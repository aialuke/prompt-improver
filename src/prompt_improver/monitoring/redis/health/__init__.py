"""Redis Health Monitoring Services

Focused Redis health monitoring components following SRE best practices:

- RedisHealthChecker: Connection status and response time monitoring
- RedisConnectionMonitor: Connection pool management and failover detection  
- RedisMetricsCollector: Performance metrics and throughput analysis
- RedisAlertingService: Incident detection and notification management
- RedisRecoveryService: Automatic recovery and circuit breaker patterns
- RedisHealthManager: Unified coordination facade for all health services

All services are designed for <25ms operations with real-time monitoring capabilities.
"""

from .health_checker import RedisHealthChecker
from .connection_monitor import RedisConnectionMonitor
from .metrics_collector import RedisMetricsCollector
from .alerting_service import RedisAlertingService
from .recovery_service import RedisRecoveryService
from .health_manager import RedisHealthManager, DefaultRedisClientProvider

# Core health monitoring types
from .types import (
    RedisHealthStatus,
    RedisRole,
    HealthMetrics,
    ConnectionPoolMetrics,
    PerformanceMetrics,
    AlertEvent,
    RecoveryEvent,
    ConnectionStatus,
    AlertLevel,
    RecoveryAction,
)

# Protocols for service interfaces
from .protocols import (
    RedisHealthCheckerProtocol,
    RedisConnectionMonitorProtocol,
    RedisMetricsCollectorProtocol,
    RedisAlertingServiceProtocol,
    RedisRecoveryServiceProtocol,
    RedisHealthManagerProtocol,
)

__all__ = [
    # Main services
    "RedisHealthChecker",
    "RedisConnectionMonitor", 
    "RedisMetricsCollector",
    "RedisAlertingService",
    "RedisRecoveryService",
    "RedisHealthManager",
    "DefaultRedisClientProvider",
    
    # Types
    "RedisHealthStatus",
    "RedisRole",
    "HealthMetrics",
    "ConnectionPoolMetrics",
    "PerformanceMetrics", 
    "AlertEvent",
    "RecoveryEvent",
    "ConnectionStatus", 
    "AlertLevel",
    "RecoveryAction",
    
    # Protocols
    "RedisHealthCheckerProtocol",
    "RedisConnectionMonitorProtocol",
    "RedisMetricsCollectorProtocol", 
    "RedisAlertingServiceProtocol",
    "RedisRecoveryServiceProtocol",
    "RedisHealthManagerProtocol",
]