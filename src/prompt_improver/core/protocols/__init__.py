"""Protocol Interfaces Package

This package contains Protocol interfaces that define contracts for various
system components. Using protocols eliminates circular imports by providing
interface definitions without implementation dependencies.

2025 Best Practice: Use typing.Protocol for dependency inversion.
"""

from prompt_improver.core.protocols.cache_protocol import (
    AdvancedCacheProtocol,
    BasicCacheProtocol,
    CacheHealthProtocol,
    CacheLockProtocol,
    CacheSubscriptionProtocol,
    MultiLevelCacheProtocol,
    RedisCacheProtocol,
)
from prompt_improver.core.protocols.connection_protocol import (
    ConnectionManagerProtocol,
    ConnectionMode,
)
from prompt_improver.core.protocols.database_protocol import (
    DatabaseConfigProtocol,
    DatabaseHealthProtocol,
    DatabaseProtocol,
    DatabaseSessionProtocol,
    QueryOptimizerProtocol,
)
from prompt_improver.core.protocols.datetime_protocol import (
    DateTimeServiceProtocol,
    DateTimeUtilsProtocol,
    TimeZoneServiceProtocol,
)
from prompt_improver.core.protocols.health_protocol import (
    HealthCheckResult as SimpleHealthCheckResult,
    HealthMonitorProtocol,
    HealthStatus as SimpleHealthStatus,
)
from prompt_improver.core.protocols.ml_protocol import (
    AutoMLProtocol,
    DataPipelineProtocol,
    ExperimentTrackingProtocol,
    FeatureStoreProtocol,
    MLMonitoringProtocol,
    MLPlatformProtocol,
    ModelProtocol,
    ModelRegistryProtocol,
    ModelServingProtocol,
    ModelTrainingProtocol,
)
from prompt_improver.core.protocols.monitoring_protocol import (
    AdvancedHealthCheckProtocol,
    AlertingProtocol,
    BasicHealthCheckProtocol,
    CircuitBreakerProtocol as MonitoringCircuitBreakerProtocol,
    HealthCheckResult,
    HealthServiceProtocol,
    HealthStatus,
    MetricsCollectorProtocol,
    PerformanceMonitorProtocol,
    SLAMonitorProtocol,
)
from prompt_improver.core.protocols.retry_protocols import (
    AnyMetricsRegistry,
    AnyRetryConfig,
    BackgroundTaskProtocol,
    CircuitBreakerProtocol,
    MetricsRegistryProtocol,
    RetryableErrorType,
    RetryableOperation,
    RetryCallback,
    RetryConfigFactoryProtocol,
    RetryConfigProtocol,
    RetryDecoratorProtocol,
    RetryManagerProtocol,
    RetryManagerProtocol as SimpleRetryManagerProtocol,
    RetryObserverProtocol,
    RetryPredicate,
    RetryStrategy,
    RetryStrategy as SimpleRetryStrategy,
)

__all__ = [
    "AdvancedCacheProtocol",
    "AdvancedHealthCheckProtocol",
    "AlertingProtocol",
    "AnyMetricsRegistry",
    "AnyRetryConfig",
    "AutoMLProtocol",
    "BackgroundTaskProtocol",
    "BasicCacheProtocol",
    "BasicHealthCheckProtocol",
    "CacheHealthProtocol",
    "CacheLockProtocol",
    "CacheSubscriptionProtocol",
    "CircuitBreakerProtocol",
    "ConnectionManagerProtocol",
    "ConnectionMode",
    "DataPipelineProtocol",
    "DatabaseConfigProtocol",
    "DatabaseHealthProtocol",
    "DatabaseProtocol",
    "DatabaseSessionProtocol",
    "DateTimeServiceProtocol",
    "DateTimeUtilsProtocol",
    "ExperimentTrackingProtocol",
    "FeatureStoreProtocol",
    "HealthCheckResult",
    "HealthMonitorProtocol",
    "HealthServiceProtocol",
    "HealthStatus",
    "MLMonitoringProtocol",
    "MLPlatformProtocol",
    "MetricsCollectorProtocol",
    "MetricsRegistryProtocol",
    "ModelProtocol",
    "ModelRegistryProtocol",
    "ModelServingProtocol",
    "ModelTrainingProtocol",
    "MonitoringCircuitBreakerProtocol",
    "MultiLevelCacheProtocol",
    "PerformanceMonitorProtocol",
    "QueryOptimizerProtocol",
    "RedisCacheProtocol",
    "RetryCallback",
    "RetryConfigFactoryProtocol",
    "RetryConfigProtocol",
    "RetryDecoratorProtocol",
    "RetryManagerProtocol",
    "RetryObserverProtocol",
    "RetryPredicate",
    "RetryStrategy",
    "RetryableErrorType",
    "RetryableOperation",
    "SLAMonitorProtocol",
    "SimpleHealthCheckResult",
    "SimpleHealthStatus",
    "SimpleRetryManagerProtocol",
    "SimpleRetryStrategy",
    "TimeZoneServiceProtocol",
]
