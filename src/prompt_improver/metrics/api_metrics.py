"""API Metrics collection for real-behavior integration tests.

Provides lightweight, in-memory API usage and user journey metrics with a
simple collector interface expected by tests/integration/test_api_metrics_integration.py.

This module intentionally avoids external dependencies and persistent storage.
Optionally, a future implementation can forward metrics to OpenTelemetry, but
that is not required to satisfy current tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, cast

try:
    from prompt_improver.monitoring.opentelemetry.metrics import get_http_metrics

    _HAS_OTEL = True
except Exception:
    _HAS_OTEL = False


class HTTPMethod(StrEnum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


# Helper to emit user journey events via OTEL BusinessMetrics
def _emit_journey_event_otel(service_name: str, metric: UserJourneyMetric) -> None:
    if not _HAS_OTEL:
        return
    try:
        from prompt_improver.monitoring.opentelemetry.metrics import (
            get_business_metrics,
        )

        get_business_metrics(service_name).record_journey_event(
            user_id=metric.user_id,
            session_id=metric.session_id,
            stage=metric.journey_stage.value,
            event_type=metric.event_type,
            success=metric.success,
        )
    except Exception:
        # never fail due to OTEL emission errors
        pass


class EndpointCategory(StrEnum):
    PROMPT_IMPROVEMENT = "PROMPT_IMPROVEMENT"
    HEALTH_CHECK = "HEALTH_CHECK"
    OTHER = "OTHER"


class AuthenticationMethod(StrEnum):
    ANONYMOUS = "ANONYMOUS"
    API_KEY = "API_KEY"
    JWT_TOKEN = "JWT_TOKEN"
    OAUTH2 = "OAUTH2"


class UserJourneyStage(StrEnum):
    ONBOARDING = "ONBOARDING"
    FIRST_USE = "FIRST_USE"
    REGULAR_USE = "REGULAR_USE"
    POWER_USER = "POWER_USER"


@dataclass
class APIUsageMetric:
    endpoint: str
    method: HTTPMethod
    category: EndpointCategory
    status_code: int
    response_time_ms: float
    request_size_bytes: int
    response_size_bytes: int
    user_id: str | None = None
    session_id: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    query_parameters_count: int = 0
    payload_type: str | None = None
    rate_limited: bool = False
    cache_hit: bool = False
    authentication_method: AuthenticationMethod = AuthenticationMethod.ANONYMOUS
    api_version: str | None = None


@dataclass
class UserJourneyMetric:
    user_id: str
    session_id: str
    journey_stage: UserJourneyStage
    event_type: str
    endpoint: str | None = None
    success: bool = True
    conversion_value: float | None = None
    time_to_action_seconds: float | None = None
    previous_stage: UserJourneyStage | None = None
    feature_flags_active: list[str] = field(default_factory=list)
    cohort_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


# Additional metric types imported by tests (not functionally used in current tests)
@dataclass
class RateLimitMetric:
    endpoint: str
    rate_limited: bool
    limit_per_minute: int | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class AuthenticationMetric:
    user_id: str | None
    method: AuthenticationMethod
    success: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class APIMetricsCollector:
    """In-memory collector for API and user journey metrics."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.max_api_metrics: int = int(cfg.get("max_api_metrics", 5000))
        self.max_journey_metrics: int = int(cfg.get("max_journey_metrics", 5000))
        self.aggregation_window_minutes: int = int(
            cfg.get("aggregation_window_minutes", 1)
        )
        self.retention_hours: int = int(cfg.get("retention_hours", 24))
        self.service_name: str = str(cfg.get("service_name", "prompt-improver"))
        self.enable_otel: bool = bool(cfg.get("enable_otel", True))

        self._api_usage: list[APIUsageMetric] = []
        self._journey_events: list[UserJourneyMetric] = []

        self.endpoint_popularity_cache: dict[str, int] = {}
        self.active_sessions: dict[str, dict[str, Any]] = {}
        self.user_journey_states: dict[str, UserJourneyStage] = {}

        self.last_aggregation: datetime | None = None
        self._started: bool = False

    async def start_collection(self) -> None:
        self._started = True

    async def stop_collection(self) -> None:
        self._started = False

    def _ensure_started(self) -> None:
        # For convenience functions that record without explicit start
        if not self._started:
            # Implicitly mark started; no async work needed for in-memory collector
            self._started = True

    async def record_api_usage(self, metric: APIUsageMetric) -> None:
        self._ensure_started()
        # Keep within bounds
        if len(self._api_usage) >= self.max_api_metrics:
            self._api_usage.pop(0)
        self._api_usage.append(metric)
        # Update endpoint popularity (method:endpoint)
        key = f"{metric.method.value}:{metric.endpoint}"
        self.endpoint_popularity_cache[key] = (
            self.endpoint_popularity_cache.get(key, 0) + 1
        )
        # Update session activity
        if metric.session_id:
            sess = self.active_sessions.get(metric.session_id)
            if not sess:
                sess = {
                    "user_id": metric.user_id,
                    "endpoint_count": 0,
                    "last_seen": datetime.now(UTC),
                }
                self.active_sessions[metric.session_id] = sess
            sess = cast("dict[str, Any]", sess)
            sess["endpoint_count"] += 1
            sess["last_seen"] = datetime.now(UTC)
        # Forward to OpenTelemetry (RED method)
        if self.enable_otel and _HAS_OTEL:
            try:
                get_http_metrics(self.service_name).record_request(
                    method=metric.method.value,
                    endpoint=metric.endpoint,
                    status_code=metric.status_code,
                    duration_ms=float(metric.response_time_ms),
                    response_size_bytes=int(metric.response_size_bytes or 0),
                )
                # Also emit a lightweight journey event for visibility in end-to-end flows
                _emit_journey_event_otel(
                    self.service_name,
                    UserJourneyMetric(
                        user_id=metric.user_id or "anonymous",
                        session_id=metric.session_id
                        or f"session_{(metric.user_id or 'anon')}",
                        journey_stage=UserJourneyStage.REGULAR_USE,
                        event_type="api_request",
                        endpoint=metric.endpoint,
                        success=metric.status_code < 400,
                    ),
                )
            except Exception:
                # Never fail the collector due to OTEL issues
                pass

    async def record_user_journey(self, metric: UserJourneyMetric) -> None:
        self._ensure_started()
        if len(self._journey_events) >= self.max_journey_metrics:
            self._journey_events.pop(0)
        self._journey_events.append(metric)
        # Track the latest stage per user
        self.user_journey_states[metric.user_id] = metric.journey_stage
        # Emit OTEL counters/gauges
        if self.enable_otel:
            _emit_journey_event_otel(self.service_name, metric)

    def get_collection_stats(self) -> dict[str, Any]:
        return {
            "api_calls_tracked": len(self._api_usage),
            "journey_events_tracked": len(self._journey_events),
            "current_metrics_count": {
                "api_usage": len(self._api_usage),
                "user_journey": len(self._journey_events),
            },
            "last_aggregation": self.last_aggregation,
        }

    async def _aggregate_metrics(self) -> None:
        # For now, aggregation is a marker that could perform rollups; we only need timestamp
        self.last_aggregation = datetime.now(UTC)

    async def get_endpoint_analytics(self, hours: int = 1) -> dict[str, Any]:
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        # Filter recent API usage
        recent: list[APIUsageMetric] = [
            m for m in self._api_usage if (m.timestamp or datetime.now(UTC)) >= cutoff
        ]
        total_requests = len(recent)
        # Aggregate by endpoint (path only, not method)
        per_endpoint: dict[str, dict[str, Any]] = {}
        for m in recent:
            ep = m.endpoint
            agg = per_endpoint.get(ep)
            if not agg:
                agg = {
                    "request_count": 0,
                    "total_response_time_ms": 0.0,
                    "success_count": 0,
                    "rate_limited_count": 0,
                    "cache_hit_count": 0,
                }
                per_endpoint[ep] = agg
            agg["request_count"] += 1
            agg["total_response_time_ms"] += float(m.response_time_ms)
            if 200 <= int(m.status_code) < 400:
                agg["success_count"] += 1
            if m.rate_limited:
                agg["rate_limited_count"] += 1
            if m.cache_hit:
                agg["cache_hit_count"] += 1
        # Finalize averages and rates
        endpoint_analytics: dict[str, dict[str, Any]] = {}
        for ep, agg in per_endpoint.items():
            count = max(agg["request_count"], 1)
            endpoint_analytics[ep] = {
                "request_count": agg["request_count"],
                "avg_response_time_ms": agg["total_response_time_ms"] / count,
                "success_rate": agg["success_count"] / count,
                "rate_limit_rate": agg["rate_limited_count"] / count,
                "cache_hit_rate": agg["cache_hit_count"] / count,
            }
        return {
            "total_requests": total_requests,
            "unique_endpoints": len(per_endpoint),
            "endpoint_analytics": endpoint_analytics,
        }

    async def _cleanup_expired_sessions(self) -> None:
        # Remove sessions not seen within retention window
        if not self.active_sessions:
            return
        ttl = timedelta(hours=self.retention_hours)
        now = datetime.now(UTC)
        expired = [
            sid
            for sid, s in self.active_sessions.items()
            if (now - s.get("last_seen", now)) > ttl
        ]
        for sid in expired:
            self.active_sessions.pop(sid, None)


# Module-level singleton for convenience functions
_singleton: APIMetricsCollector | None = None


def get_api_metrics_collector(
    config: dict[str, Any] | None = None,
) -> APIMetricsCollector:
    global _singleton
    if _singleton is None:
        _singleton = APIMetricsCollector(config=config)
    return _singleton


async def record_api_request(
    *,
    endpoint: str,
    method: HTTPMethod,
    category: EndpointCategory,
    status_code: int,
    response_time_ms: float,
    user_id: str | None = None,
    session_id: str | None = None,
    request_size_bytes: int = 0,
    response_size_bytes: int = 0,
    ip_address: str | None = None,
    user_agent: str | None = None,
    query_parameters_count: int = 0,
    payload_type: str | None = None,
    rate_limited: bool = False,
    cache_hit: bool = False,
    authentication_method: AuthenticationMethod = AuthenticationMethod.ANONYMOUS,
    api_version: str | None = None,
) -> None:
    collector = get_api_metrics_collector()
    metric = APIUsageMetric(
        endpoint=endpoint,
        method=method,
        category=category,
        status_code=status_code,
        response_time_ms=response_time_ms,
        request_size_bytes=request_size_bytes,
        response_size_bytes=response_size_bytes,
        user_id=user_id,
        session_id=session_id,
        ip_address=ip_address,
        user_agent=user_agent,
        query_parameters_count=query_parameters_count,
        payload_type=payload_type,
        rate_limited=rate_limited,
        cache_hit=cache_hit,
        authentication_method=authentication_method,
        api_version=api_version,
    )
    await collector.record_api_usage(metric)


async def record_user_journey_event(
    *,
    user_id: str,
    session_id: str,
    journey_stage: UserJourneyStage,
    event_type: str,
    endpoint: str | None = None,
    success: bool = True,
    conversion_value: float | None = None,
    time_to_action_seconds: float | None = None,
    previous_stage: UserJourneyStage | None = None,
    feature_flags_active: list[str] | None = None,
    cohort_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    collector = get_api_metrics_collector()
    metric = UserJourneyMetric(
        user_id=user_id,
        session_id=session_id,
        journey_stage=journey_stage,
        event_type=event_type,
        endpoint=endpoint,
        success=success,
        conversion_value=conversion_value,
        time_to_action_seconds=time_to_action_seconds,
        previous_stage=previous_stage,
        feature_flags_active=feature_flags_active or [],
        cohort_id=cohort_id,
        metadata=metadata or {},
    )
    await collector.record_user_journey(metric)
