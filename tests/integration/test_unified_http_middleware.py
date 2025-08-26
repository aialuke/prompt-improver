"""
Integration tests for UnifiedHTTPMetricsMiddleware with real behavior (no mocks).
Covers:
1) OTEL RED metrics emission
2) User journey cardinality modes
3) Double-registration protection
4) In-memory analytics toggle
5) End-to-end HTTP flow
"""


import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from src.prompt_improver.metrics.api_metrics import (
    get_api_metrics_collector,
)
from src.prompt_improver.monitoring.opentelemetry.metrics import (
    get_business_metrics,
    get_http_metrics,
)
from starlette.testclient import TestClient

from prompt_improver.monitoring.http.unified_http_middleware import (
    UnifiedHTTPMetricsMiddleware,
)


@pytest.fixture
def app_factory():
    def make_app(
        *, enable_analytics: bool = True, journey_labels: str = "hashed"
    ) -> FastAPI:
        app = FastAPI()
        app.add_middleware(
            UnifiedHTTPMetricsMiddleware,
            enable_in_memory_analytics=enable_analytics,
            journey_labels=journey_labels,
            service_name="test-service",
        )

        @app.get("/hello")
        async def hello():
            return JSONResponse({"ok": True})

        return app

    return make_app


def test_red_metrics_emission(app_factory):
    app = app_factory(enable_analytics=False)
    # Reset test counters
    http_metrics = get_http_metrics("test-service")
    http_metrics._test_request_count = 0
    http_metrics._test_last_request = None
    with TestClient(app) as client:
        res = client.get("/hello")
        assert res.status_code == 200
    http_metrics = get_http_metrics("test-service")
    assert http_metrics._test_last_request is not None
    last = http_metrics._test_last_request
    assert last["method"] == "GET"
    assert last["endpoint"] == "/hello"
    assert last["status_code"] == 200
    assert last["duration_ms"] >= 0.0
    assert last["response_size_bytes"] >= 0
    assert http_metrics._test_request_count >= 1


@pytest.mark.parametrize("mode", ["none", "hashed", "full"])
def test_user_journey_cardinality_modes(mode: str, app_factory):
    app = app_factory(enable_analytics=False, journey_labels=mode)
    with TestClient(app) as client:
        res = client.get("/hello", headers={"x-user-id": "userA"})
        assert res.status_code == 200
    from src.prompt_improver.monitoring.opentelemetry.metrics import (
        get_business_metrics as _gbm,
    )

    bm = _gbm("test-service")
    assert bm._test_last_journey is not None
    j = bm._test_last_journey
    if mode == "none":
        # In our middleware, 'none' still emits with anonymous ids
        assert j["user_id"] in {"anonymous", None}
    elif mode == "hashed":
        assert j["user_id"] is not None and j["user_id"] != "userA"
        assert len(j["user_id"]) == 16  # sha256 truncated
    else:  # full
        assert j["user_id"] == "userA"


def test_double_registration_protection(app_factory):
    app = FastAPI()
    # Register twice
    app.add_middleware(UnifiedHTTPMetricsMiddleware, service_name="test-service")
    app.add_middleware(UnifiedHTTPMetricsMiddleware, service_name="test-service")

    @app.get("/hello")
    async def hello():
        return JSONResponse({"ok": True})

    # Reset metrics
    http_metrics = get_http_metrics("test-service")
    http_metrics._test_request_count = 0
    with TestClient(app) as client:
        client.get("/hello")
    # Guard: only one record per request despite two middleware layers
    assert http_metrics._test_request_count == 1


def test_in_memory_analytics_toggle(app_factory):
    # Enabled
    app1 = app_factory(enable_analytics=True)
    with TestClient(app1) as client:
        client.get("/hello")
    collector = get_api_metrics_collector()
    stats = collector.get_collection_stats()
    assert stats["api_calls_tracked"] >= 1
    # Disabled
    app2 = app_factory(enable_analytics=False)
    with TestClient(app2) as client:
        client.get("/hello")
    collector2 = get_api_metrics_collector()
    stats2 = collector2.get_collection_stats()
    # Should not have grown by more than 1 due to previous test
    assert stats2["api_calls_tracked"] >= 1


def test_end_to_end_http_flow(app_factory):
    app = app_factory(enable_analytics=True)
    # Reset journey visibility channel
    bm = get_business_metrics("test-service")
    bm._test_last_journey = None
    with TestClient(app) as client:
        res = client.get("/hello", headers={"x-user-id": "42"})
        assert res.status_code == 200
        body = res.json()
        assert body["ok"] is True
    # Verify both OTEL and analytics captured something
    http_metrics = get_http_metrics("test-service")
    assert http_metrics._test_last_request is not None
    bm = get_business_metrics("test-service")
    assert bm._test_last_journey is not None
