"""Protocol Interfaces Package

This package contains Protocol interfaces that define contracts for various
system components. Using protocols eliminates circular imports by providing
interface definitions without implementation dependencies.

2025 Best Practice: Use typing.Protocol for dependency inversion.
"""

# Heavy ML imports moved to TYPE_CHECKING and lazy loading to avoid torch dependency
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

def _get_ml_protocols():
    """Lazy load ML protocols when needed to avoid torch import on database access."""
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
    return {
        'AutoMLProtocol': AutoMLProtocol,
        'DataPipelineProtocol': DataPipelineProtocol,
        'ExperimentTrackingProtocol': ExperimentTrackingProtocol,
        'FeatureStoreProtocol': FeatureStoreProtocol,
        'MLMonitoringProtocol': MLMonitoringProtocol,
        'MLPlatformProtocol': MLPlatformProtocol,
        'ModelProtocol': ModelProtocol,
        'ModelRegistryProtocol': ModelRegistryProtocol,
        'ModelServingProtocol': ModelServingProtocol,
        'ModelTrainingProtocol': ModelTrainingProtocol,
    }

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

# ML protocols are lazy-loaded - use _get_ml_protocols() to access them
def __getattr__(name: str):
    """Lazy loading for ML protocols to avoid torch import on database access."""
    ml_protocol_names = {
        'AutoMLProtocol', 'DataPipelineProtocol', 'ExperimentTrackingProtocol',
        'FeatureStoreProtocol', 'MLMonitoringProtocol', 'MLPlatformProtocol',
        'ModelProtocol', 'ModelRegistryProtocol', 'ModelServingProtocol',
        'ModelTrainingProtocol'
    }
    
    if name in ml_protocol_names:
        ml_protocols = _get_ml_protocols()
        return ml_protocols[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "AdvancedCacheProtocol",
    "AdvancedHealthCheckProtocol",
    "AlertingProtocol",
    "AnyMetricsRegistry",
    "AnyRetryConfig",
    "BackgroundTaskProtocol",
    "BasicCacheProtocol",
    "BasicHealthCheckProtocol",
    "CacheHealthProtocol",
    "CacheLockProtocol",
    "CacheSubscriptionProtocol",
    "CircuitBreakerProtocol",
    "ConnectionManagerProtocol",
    "ConnectionMode",
    "DatabaseConfigProtocol",
    "DatabaseHealthProtocol",
    "DatabaseProtocol",
    "DatabaseSessionProtocol",
    "DateTimeServiceProtocol",
    "DateTimeUtilsProtocol",
    "HealthCheckResult",
    "HealthMonitorProtocol",
    "HealthServiceProtocol",
    "HealthStatus",
    "MetricsCollectorProtocol",
    "MetricsRegistryProtocol",
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
    # ML protocols are available via lazy loading
    "AutoMLProtocol",
    "DataPipelineProtocol", 
    "ExperimentTrackingProtocol",
    "FeatureStoreProtocol",
    "MLMonitoringProtocol",
    "MLPlatformProtocol",
    "ModelProtocol",
    "ModelRegistryProtocol",
    "ModelServingProtocol",
    "ModelTrainingProtocol",
]
