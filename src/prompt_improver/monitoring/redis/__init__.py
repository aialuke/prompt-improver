"""Redis Health Monitoring Services.

Focused Redis health monitoring services following Clean Architecture patterns.
Replaces the 1473-line god object with specialized, focused services.

Services:
- RedisHealthChecker: Connection status and response time monitoring
- RedisConnectionMonitor: Connection pool management and failover detection
- RedisMetricsCollector: Performance metrics and throughput analysis
- RedisAlertingService: Incident detection and notification management
- RedisRecoveryService: Automatic recovery and circuit breaker patterns
- RedisHealthManager: Unified coordination facade for all health services

Features:
- <25ms health check operations
- Real-time monitoring with failure detection
- Protocol-based interfaces for testability
- Zero backwards compatibility - clean break strategy
"""

# Import new focused health services
from prompt_improver.monitoring.redis.health import (
    AlertEvent,
    ConnectionPoolMetrics,
    DefaultRedisClientProvider,
    HealthMetrics,
    PerformanceMetrics,
    RecoveryEvent,
    RedisAlertingService,
    RedisConnectionMonitor,
    RedisHealthChecker,
    RedisHealthManager,
    RedisHealthStatus,
    RedisMetricsCollector,
    RedisRecoveryService,
    RedisRole,
)

# Legacy services (for compatibility)
try:
    from prompt_improver.monitoring.redis.services import (
        RedisConnectionMonitor as LegacyRedisConnectionMonitor,
    )
    from prompt_improver.shared.interfaces.protocols.monitoring import (
        RedisConnectionMonitorProtocol,
        RedisHealthCheckerProtocol,
    )
except ImportError:
    # Legacy services not available
    pass
# Legacy types (for compatibility)
try:
    from prompt_improver.monitoring.redis.types import (
        ConnectionHealthStatus,
        RedisAlertLevel,
        RedisHealthConfig,
        RedisHealthMetrics,
        RedisHealthResult,
        RedisPerformanceMetrics,
    )
except ImportError:
    # Legacy types not available
    pass

__all__ = [
    "AlertEvent",
    "ConnectionPoolMetrics",
    "DefaultRedisClientProvider",
    "HealthMetrics",
    "PerformanceMetrics",
    "RecoveryEvent",
    "RedisAlertingService",
    "RedisConnectionMonitor",
    # Legacy compatibility (if available)
    "RedisConnectionMonitorProtocol",
    # New focused services (recommended)
    "RedisHealthChecker",
    "RedisHealthCheckerProtocol",
    "RedisHealthManager",
    # New types
    "RedisHealthStatus",
    "RedisMetricsCollector",
    "RedisRecoveryService",
    "RedisRole",
]
