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
from .health import (
    RedisHealthChecker,
    RedisConnectionMonitor,
    RedisMetricsCollector,
    RedisAlertingService,
    RedisRecoveryService,
    RedisHealthManager,
    DefaultRedisClientProvider,
    RedisHealthStatus,
    RedisRole,
    HealthMetrics,
    ConnectionPoolMetrics,
    PerformanceMetrics,
    AlertEvent,
    RecoveryEvent,
)

# Legacy services (for compatibility)
try:
    from .protocols import (
        RedisConnectionMonitorProtocol,
        RedisHealthCheckerProtocol,
    )
    from .services import RedisConnectionMonitor as LegacyRedisConnectionMonitor
except ImportError:
    # Legacy services not available
    pass
# Legacy types (for compatibility)
try:
    from .types import (
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
    # New focused services (recommended)
    "RedisHealthChecker",
    "RedisConnectionMonitor",
    "RedisMetricsCollector", 
    "RedisAlertingService",
    "RedisRecoveryService",
    "RedisHealthManager",
    "DefaultRedisClientProvider",
    
    # New types
    "RedisHealthStatus",
    "RedisRole",
    "HealthMetrics",
    "ConnectionPoolMetrics",
    "PerformanceMetrics",
    "AlertEvent",
    "RecoveryEvent",
    
    # Legacy compatibility (if available)
    "RedisConnectionMonitorProtocol",
    "RedisHealthCheckerProtocol",
]