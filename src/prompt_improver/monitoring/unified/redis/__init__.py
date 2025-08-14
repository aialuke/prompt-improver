"""Redis health monitoring components.

Focused Redis monitoring architecture with decomposed responsibilities:
- RedisHealthChecker: Basic connectivity and health validation
- ConnectionMonitor: Connection pool management and monitoring  
- MetricsCollector: Performance metrics and alerting
- RedisHealthMonitorFacade: Unified coordination interface

Each component is <400 lines with <25ms operation targets.
"""

from .facade import RedisHealthMonitorFacade
from .health_checker import RedisHealthChecker
from .connection_monitor import ConnectionMonitor
from .metrics_collector import MetricsCollector
from .types import (
    RedisHealthStatus,
    RedisRole,
    MemoryMetrics,
    PerformanceMetrics,
    PersistenceMetrics,
    ReplicationMetrics,
    ConnectionMetrics,
    KeyspaceMetrics,
    SlowLogMetrics,
    KeyspaceInfo,
    SlowLogEntry,
)

__all__ = [
    "RedisHealthMonitorFacade",
    "RedisHealthChecker", 
    "ConnectionMonitor",
    "MetricsCollector",
    "RedisHealthStatus",
    "RedisRole",
    "MemoryMetrics",
    "PerformanceMetrics", 
    "PersistenceMetrics",
    "ReplicationMetrics",
    "ConnectionMetrics",
    "KeyspaceMetrics",
    "SlowLogMetrics",
    "KeyspaceInfo",
    "SlowLogEntry",
]