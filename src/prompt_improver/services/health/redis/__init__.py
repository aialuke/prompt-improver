"""Redis Health Monitoring Components

Decomposed Redis health monitoring components:
- RedisHealthChecker: Basic connectivity and health validation  
- RedisConnectionMonitor: Connection pool monitoring and metrics
- RedisMetricsCollector: Performance and operational metrics
- RedisAlerting: Health alerting and notifications
- RedisHealthFacade: Unified interface coordination

Each component is focused on a single responsibility with <350 lines.
"""

from .alerting import RedisAlerting
from .connection_monitor import RedisConnectionMonitor
from .facade import RedisHealthFacade
from .health_checker import RedisHealthChecker
from .metrics_collector import RedisMetricsCollector

# Backwards compatibility exports
from .facade import (
    RedisHealthMonitor,  # Legacy compatibility
    create_redis_health_checker,  # Legacy compatibility
    get_redis_health_summary,  # Legacy compatibility
)

__all__ = [
    "RedisAlerting",
    "RedisConnectionMonitor", 
    "RedisHealthChecker",
    "RedisHealthFacade",
    "RedisMetricsCollector",
    # Legacy compatibility
    "RedisHealthMonitor",
    "create_redis_health_checker",
    "get_redis_health_summary",
]