"""Health Service Protocols

Protocol definitions for health monitoring components following
clean architecture principles with focused responsibilities.
"""

from .health_service_protocols import (
    AlertingServiceProtocol,
    ConnectionMonitorProtocol,
    HealthCheckerProtocol,
    HealthFacadeProtocol,
    MetricsCollectorProtocol,
)

__all__ = [
    "AlertingServiceProtocol",
    "ConnectionMonitorProtocol", 
    "HealthCheckerProtocol",
    "HealthFacadeProtocol",
    "MetricsCollectorProtocol",
]