"""APES Health Check System - PHASE 3 Implementation.
Composite pattern health monitoring with Prometheus instrumentation.
"""

from .base import HealthChecker, HealthResult, HealthStatus, AggregatedHealthResult
from .service import HealthService, get_health_service, reset_health_service
from .checkers import (
    DatabaseHealthChecker,
    MCPServerHealthChecker,
    AnalyticsServiceHealthChecker, 
    MLServiceHealthChecker,
    SystemResourcesHealthChecker
)
from .metrics import (
    instrument_health_check,
    get_health_metrics_summary,
    reset_health_metrics,
    PROMETHEUS_AVAILABLE
)

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
    "SystemResourcesHealthChecker",
    
    # Metrics
    "instrument_health_check",
    "get_health_metrics_summary",
    "reset_health_metrics",
    "PROMETHEUS_AVAILABLE"
]
