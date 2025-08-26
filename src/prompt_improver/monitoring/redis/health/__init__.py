"""Redis Health Monitoring Services.

Focused Redis health monitoring components following SRE best practices:

- RedisHealthChecker: Connection status and response time monitoring
- RedisConnectionMonitor: Connection pool management and failover detection
- RedisMetricsCollector: Performance metrics and throughput analysis
- RedisAlertingService: Incident detection and notification management
- RedisRecoveryService: Automatic recovery and circuit breaker patterns
- RedisHealthManager: Unified coordination facade for all health services

All services are designed for <25ms operations with real-time monitoring capabilities.
"""

from prompt_improver.monitoring.redis.health.alerting_service import (
    RedisAlertingService,
)
from prompt_improver.monitoring.redis.health.connection_monitor import (
    RedisConnectionMonitor,
)
from prompt_improver.monitoring.redis.health.health_checker import RedisHealthChecker
from prompt_improver.monitoring.redis.health.health_manager import (
    DefaultRedisClientProvider,
    RedisHealthManager,
)
from prompt_improver.monitoring.redis.health.metrics_collector import (
    RedisMetricsCollector,
)
from prompt_improver.monitoring.redis.health.recovery_service import (
    RedisRecoveryService,
)

# Core health monitoring types
from prompt_improver.monitoring.redis.health.types import (
    AlertEvent,
    AlertLevel,
    ConnectionPoolMetrics,
    ConnectionStatus,
    HealthMetrics,
    PerformanceMetrics,
    RecoveryAction,
    RecoveryEvent,
    RedisHealthStatus,
    RedisRole,
)

# Protocols for service interfaces
from prompt_improver.shared.interfaces.protocols.monitoring import (
    RedisAlertingServiceProtocol,
    RedisConnectionMonitorProtocol,
    RedisHealthCheckerProtocol,
    RedisHealthManagerProtocol,
    RedisMetricsCollectorProtocol,
    RedisRecoveryServiceProtocol,
)

__all__ = [
    "AlertEvent",
    "AlertLevel",
    "ConnectionPoolMetrics",
    "ConnectionStatus",
    "DefaultRedisClientProvider",
    "HealthMetrics",
    "PerformanceMetrics",
    "RecoveryAction",
    "RecoveryEvent",
    "RedisAlertingService",
    "RedisAlertingServiceProtocol",
    "RedisConnectionMonitor",
    "RedisConnectionMonitorProtocol",
    # Main services
    "RedisHealthChecker",
    # Protocols
    "RedisHealthCheckerProtocol",
    "RedisHealthManager",
    "RedisHealthManagerProtocol",
    # Types
    "RedisHealthStatus",
    "RedisMetricsCollector",
    "RedisMetricsCollectorProtocol",
    "RedisRecoveryService",
    "RedisRecoveryServiceProtocol",
    "RedisRole",
]
