"""Unified HTTP metrics middleware for APES (2025 best practices).

This middleware consolidates OpenTelemetry HTTP RED metrics emission and
business user-journey event tracking into a single, consistent layer. It also
optionally updates in-memory endpoint analytics via APIMetricsCollector.

Features:
- Emits OTEL RED metrics via get_http_metrics().record_request(...)
- Emits user journey events via get_business_metrics().record_journey_event(...)
- Optional in-memory analytics via APIMetricsCollector.record_api_request(...)
- Double-count protection to avoid duplicate emission
- Cardinality controls for journey labels: 'none' | 'hashed' | 'full'
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Dict, Literal, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from prompt_improver.monitoring.opentelemetry.metrics import (
    get_business_metrics,
    get_http_metrics,
)
from src.prompt_improver.metrics.api_metrics import (
    AuthenticationMethod,
    EndpointCategory,
    HTTPMethod,
    record_api_request,
    UserJourneyStage,
)


JourneyLabelsMode = Literal["none", "hashed", "full"]


class UnifiedHTTPMetricsMiddleware(BaseHTTPMiddleware):
    """Unified middleware that replaces legacy middlewares.

    Configuration:
    - enable_in_memory_analytics: bool
    - journey_labels: 'none' | 'hashed' | 'full'
    """

    _guard_key = "_apes_unified_http_metrics_emitted"

    def __init__(
        self,
        app,
        *,
        enable_in_memory_analytics: bool = True,
        journey_labels: JourneyLabelsMode = "hashed",
        service_name: str = "prompt-improver",
    ) -> None:
        super().__init__(app)
        self.enable_in_memory_analytics = enable_in_memory_analytics
        self.journey_labels = journey_labels
        self.service_name = service_name

    async def dispatch(self, request: Request, call_next) -> Response:
        # Double-count protection per request
        if getattr(request.state, self._guard_key, False):
            return await call_next(request)
        setattr(request.state, self._guard_key, True)

        start = time.perf_counter()
        response: Response
        try:
            response = await call_next(request)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0

        # Extract request/response data safely
        method = getattr(request, "method", "UNKNOWN")
        endpoint = getattr(getattr(request, "url", None), "path", "/unknown")
        status_code = getattr(response, "status_code", 200) if 'response' in locals() else 500
        response_size = self._get_response_size(response)

        # OTEL RED metrics
        get_http_metrics(self.service_name).record_request(
            method=method,
            endpoint=endpoint,
            status_code=int(status_code),
            duration_ms=float(duration_ms),
            response_size_bytes=int(response_size),
        )

        # Emit business user-journey event with cardinality control
        user_id, session_id = self._extract_user_and_session(request)
        labels_mode = self.journey_labels
        if labels_mode == "hashed":
            user_id_h = self._hash_label(user_id) if user_id else None
            session_id_h = self._hash_label(session_id) if session_id else None
        else:
            user_id_h = user_id
            session_id_h = session_id

        # Default journey heuristics
        stage = UserJourneyStage.REGULAR_USE
        event_type = "api_request"
        from prompt_improver.monitoring.opentelemetry.metrics import get_business_metrics as _gbm
        _gbm(self.service_name).record_journey_event(
            user_id=user_id_h or "anonymous",
            session_id=session_id_h or f"session_{user_id_h or 'anon'}",
            stage=stage.value,
            event_type=event_type,
            success=200 <= int(status_code) < 400,
        )

        # Optional in-memory analytics (consistent with tests)
        if self.enable_in_memory_analytics:
            try:
                await record_api_request(
                    endpoint=endpoint,
                    method=HTTPMethod(method) if method in HTTPMethod.__members__ else HTTPMethod.GET,
                    category=EndpointCategory.OTHER,
                    status_code=int(status_code),
                    response_time_ms=float(duration_ms),
                    user_id=user_id,
                    session_id=session_id,
                    request_size_bytes=0,
                    response_size_bytes=int(response_size),
                    authentication_method=AuthenticationMethod.ANONYMOUS,
                )
            except Exception:
                # Do not fail request path for analytics
                pass

        return response

    def _get_response_size(self, response: Optional[Response]) -> int:
        if response is None:
            return 0
        try:
            val = response.headers.get("content-length")
            return int(val) if val else 0
        except Exception:
            return 0

    def _hash_label(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return value
        h = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return h[:16]

    def _extract_user_and_session(self, request: Request) -> tuple[Optional[str], Optional[str]]:
        # Minimal safe extraction; extend if needed
        user_id = None
        session_id = None
        try:
            # Common header/cookie sources; replace with real auth/session extractor if available
            user_id = request.headers.get("x-user-id") or request.headers.get("x-api-key")
            session_id = request.cookies.get("session_id")
        except Exception:
            pass
        return user_id, session_id

