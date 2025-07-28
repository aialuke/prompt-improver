"""Protocol Interfaces Package

This package contains Protocol interfaces that define contracts for various
system components. Using protocols eliminates circular imports by providing
interface definitions without implementation dependencies.

2025 Best Practice: Use typing.Protocol for dependency inversion.
"""

from .retry_protocols import (
    RetryStrategy,
    RetryableErrorType,
    RetryConfigProtocol,
    RetryManagerProtocol,
    MetricsRegistryProtocol,
    BackgroundTaskProtocol,
    CircuitBreakerProtocol,
    RetryDecoratorProtocol,
    RetryConfigFactoryProtocol,
    RetryObserverProtocol,
    RetryableOperation,
    RetryCallback,
    RetryPredicate,
    AnyRetryConfig,
    AnyMetricsRegistry,
)

from .datetime_protocol import (
    DateTimeServiceProtocol,
    TimeZoneServiceProtocol,
    DateTimeUtilsProtocol
)

from .database_protocol import (
    DatabaseSessionProtocol,
    DatabaseConfigProtocol,
    QueryOptimizerProtocol,
    DatabaseHealthProtocol,
    DatabaseProtocol
)

from .cache_protocol import (
    BasicCacheProtocol,
    AdvancedCacheProtocol,
    CacheHealthProtocol,
    CacheSubscriptionProtocol,
    CacheLockProtocol,
    RedisCacheProtocol,
    MultiLevelCacheProtocol
)

from .monitoring_protocol import (
    HealthStatus,
    HealthCheckResult,
    BasicHealthCheckProtocol,
    AdvancedHealthCheckProtocol,
    PerformanceMonitorProtocol,
    CircuitBreakerProtocol as MonitoringCircuitBreakerProtocol,
    AlertingProtocol,
    SLAMonitorProtocol,
    MetricsCollectorProtocol,
    HealthServiceProtocol
)

from .ml_protocol import (
    ModelProtocol,
    ModelRegistryProtocol,
    ExperimentTrackingProtocol,
    FeatureStoreProtocol,
    DataPipelineProtocol,
    ModelTrainingProtocol,
    ModelServingProtocol,
    AutoMLProtocol,
    MLMonitoringProtocol,
    MLPlatformProtocol
)

__all__ = [
    # Retry protocols (existing)
    "RetryStrategy",
    "RetryableErrorType",
    "RetryConfigProtocol",
    "RetryManagerProtocol",
    "MetricsRegistryProtocol",
    "BackgroundTaskProtocol",
    "CircuitBreakerProtocol",
    "RetryDecoratorProtocol",
    "RetryConfigFactoryProtocol",
    "RetryObserverProtocol",
    "RetryableOperation",
    "RetryCallback",
    "RetryPredicate",
    "AnyRetryConfig",
    "AnyMetricsRegistry",
    
    # DateTime protocols
    'DateTimeServiceProtocol',
    'TimeZoneServiceProtocol', 
    'DateTimeUtilsProtocol',
    
    # Database protocols
    'DatabaseSessionProtocol',
    'DatabaseConfigProtocol',
    'QueryOptimizerProtocol',
    'DatabaseHealthProtocol',
    'DatabaseProtocol',
    
    # Cache protocols
    'BasicCacheProtocol',
    'AdvancedCacheProtocol',
    'CacheHealthProtocol',
    'CacheSubscriptionProtocol',
    'CacheLockProtocol',
    'RedisCacheProtocol',
    'MultiLevelCacheProtocol',
    
    # Monitoring protocols
    'HealthStatus',
    'HealthCheckResult',
    'BasicHealthCheckProtocol',
    'AdvancedHealthCheckProtocol',
    'PerformanceMonitorProtocol',
    'MonitoringCircuitBreakerProtocol',
    'AlertingProtocol',
    'SLAMonitorProtocol',
    'MetricsCollectorProtocol',
    'HealthServiceProtocol',
    
    # ML protocols
    'ModelProtocol',
    'ModelRegistryProtocol',
    'ExperimentTrackingProtocol',
    'FeatureStoreProtocol',
    'DataPipelineProtocol',
    'ModelTrainingProtocol',
    'ModelServingProtocol',
    'AutoMLProtocol',
    'MLMonitoringProtocol',
    'MLPlatformProtocol'
]
