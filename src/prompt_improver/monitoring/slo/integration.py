"""SLO Integration shims for tests.

Provide a minimal MetricsCollector API expected by some tests. This wraps our
OpenTelemetry layer if available, else provides no-ops. This avoids heavy
refactors and keeps compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

try:  # Prefer real OTEL metrics when present
    from prompt_improver.monitoring.opentelemetry.metrics import get_http_metrics

    _HAS_OTEL = True
except Exception:
    _HAS_OTEL = False


@dataclass
class MetricsCollector:
    service_name: str = "prompt-improver"

    def increment_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_ms: float,
        response_size_bytes: int = 0,
    ) -> None:
        if _HAS_OTEL:
            get_http_metrics(self.service_name).record_request(
                method, endpoint, status_code, duration_ms, response_size_bytes
            )

    def record_db_query(self, duration_ms: float, success: bool = True) -> None:
        if _HAS_OTEL:
            from prompt_improver.monitoring.opentelemetry.metrics import (
                get_database_metrics,
            )

            get_database_metrics(self.service_name).record_query(duration_ms, success)

    def record_cache_operation(self, duration_ms: float, hit: bool) -> None:
        # No dedicated OTEL cache metrics in wrapper; could be added if needed
        pass

    def set_custom_gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        # Minimal shim; can be expanded
        pass
