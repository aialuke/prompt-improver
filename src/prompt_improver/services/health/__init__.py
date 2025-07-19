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
