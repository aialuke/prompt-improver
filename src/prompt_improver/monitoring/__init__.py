"""APES Monitoring Package
Provides comprehensive observability for the APES ML Pipeline Orchestrator.
"""

try:
    from prompt_improver.monitoring.health_check import get_metrics  # if provided
    from prompt_improver.monitoring.http.unified_http_middleware import (
        UnifiedHTTPMetricsMiddleware,
    )

    _metrics_available = True
except Exception:
    UnifiedHTTPMetricsMiddleware = None
    get_metrics = None
    _metrics_available = False
try:
    from prompt_improver.monitoring.health_check import router as health_router

    _health_check_available = True
except ImportError:
    health_router = None
    _health_check_available = False
__all__ = []
if _metrics_available:
    __all__.extend(["UnifiedHTTPMetricsMiddleware", "get_metrics"])
if _health_check_available:
    __all__.extend(["health_router"])
