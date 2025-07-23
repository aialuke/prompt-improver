"""APES Health Check System - PHASE 3 Implementation.
Composite pattern health monitoring with Prometheus instrumentation.
"""

from .base import AggregatedHealthResult, HealthChecker, HealthResult, HealthStatus
from .checkers import (
                   AnalyticsServiceHealthChecker,
                   DatabaseHealthChecker,
                   MCPServerHealthChecker,
                   MLServiceHealthChecker,
                   QueueHealthChecker,
                   SystemResourcesHealthChecker,
)

# Import ML-specific health checkers
try:
    from .ml_specific_checkers import (
        MLModelHealthChecker,
        MLDataQualityChecker,
        MLTrainingHealthChecker,
        MLPerformanceHealthChecker,
    )
    ML_SPECIFIC_CHECKERS_AVAILABLE = True
except ImportError:
    ML_SPECIFIC_CHECKERS_AVAILABLE = False
from .metrics import (
                   PROMETHEUS_AVAILABLE,
                   get_health_metrics_summary,
                   instrument_health_check,
                   reset_health_metrics,
)
from .service import HealthService, get_health_service, reset_health_service

__all__ = [
    # Base classes and types
    "HealthChecker",
    "HealthResult",
    "HealthStatus",
    "AggregatedHealthResult",
    # Main service
    "HealthService",
    "get_health_service",
    "reset_health_service",
    # Individual checkers
    "DatabaseHealthChecker",
    "MCPServerHealthChecker",
    "AnalyticsServiceHealthChecker",
    "MLServiceHealthChecker",
    "QueueHealthChecker",
    "SystemResourcesHealthChecker",
    # Metrics
    "instrument_health_check",
    "get_health_metrics_summary",
    "reset_health_metrics",
    "PROMETHEUS_AVAILABLE",
]

# Add ML-specific checkers to __all__ if available
if ML_SPECIFIC_CHECKERS_AVAILABLE:
    __all__.extend([
        "MLModelHealthChecker",
        "MLDataQualityChecker",
        "MLTrainingHealthChecker",
        "MLPerformanceHealthChecker",
    ])
