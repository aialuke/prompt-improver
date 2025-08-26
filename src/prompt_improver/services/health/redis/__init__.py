"""Redis Health Monitoring Components.

Decomposed Redis health monitoring components:
- RedisHealthChecker: Basic connectivity and health validation
- RedisConnectionMonitor: Connection pool monitoring and metrics
- RedisMetricsCollector: Performance and operational metrics
- RedisAlerting: Health alerting and notifications
- RedisHealthFacade: Unified interface coordination

Each component is focused on a single responsibility with <350 lines.
"""

from prompt_improver.services.health.redis.alerting import RedisAlerting
from prompt_improver.services.health.redis.connection_monitor import (
    RedisConnectionMonitor,
)

# Backwards compatibility exports
from prompt_improver.services.health.redis.facade import (
    RedisHealthFacade,
    RedisHealthMonitor,
    create_redis_health_checker,
    get_redis_health_summary,
)
from prompt_improver.services.health.redis.health_checker import RedisHealthChecker
from prompt_improver.services.health.redis.metrics_collector import (
    RedisMetricsCollector,
)

__all__ = [
    "RedisAlerting",
    "RedisConnectionMonitor",
    "RedisHealthChecker",
    "RedisHealthFacade",
    # Legacy compatibility
    "RedisHealthMonitor",
    "RedisMetricsCollector",
    "create_redis_health_checker",
    "get_redis_health_summary",
]
