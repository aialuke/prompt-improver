"""Redis health monitoring components.

Focused Redis monitoring architecture with decomposed responsibilities:
- RedisHealthChecker: Basic connectivity and health validation
- ConnectionMonitor: Connection pool management and monitoring
- MetricsCollector: Performance metrics and alerting
- RedisHealthMonitorFacade: Unified coordination interface

Each component is <400 lines with <25ms operation targets.
"""

from prompt_improver.monitoring.unified.redis.connection_monitor import (
    ConnectionMonitor,
)
from prompt_improver.monitoring.unified.redis.facade import RedisHealthMonitorFacade
from prompt_improver.monitoring.unified.redis.health_checker import RedisHealthChecker
from prompt_improver.monitoring.unified.redis.metrics_collector import MetricsCollector
from prompt_improver.monitoring.unified.redis.types import (
    ConnectionMetrics,
    KeyspaceInfo,
    KeyspaceMetrics,
    MemoryMetrics,
    PerformanceMetrics,
    PersistenceMetrics,
    RedisHealthStatus,
    RedisRole,
    ReplicationMetrics,
    SlowLogEntry,
    SlowLogMetrics,
)

__all__ = [
    "ConnectionMetrics",
    "ConnectionMonitor",
    "KeyspaceInfo",
    "KeyspaceMetrics",
    "MemoryMetrics",
    "MetricsCollector",
    "PerformanceMetrics",
    "PersistenceMetrics",
    "RedisHealthChecker",
    "RedisHealthMonitorFacade",
    "RedisHealthStatus",
    "RedisRole",
    "ReplicationMetrics",
    "SlowLogEntry",
    "SlowLogMetrics",
]
