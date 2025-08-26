"""Database connection management services."""

from prompt_improver.database.services.connection.connection_metrics import (
    ConnectionMetrics,
)

# Export decomposed components for direct access if needed
from prompt_improver.database.services.connection.connection_pool_core import (
    ConnectionPoolCore,
)
from prompt_improver.database.services.connection.pool_monitoring_service import (
    PoolMonitoringService,
)
from prompt_improver.database.services.connection.pool_scaler import (
    BurstDetector,
    PoolScaler,
    PoolScalerProtocol,
    ScalingAction,
    ScalingConfiguration,
    ScalingDecision,
    ScalingMetrics,
    ScalingReason,
)
from prompt_improver.database.services.connection.pool_scaling_manager import (
    PoolScalingManager,
)
from prompt_improver.database.services.connection.pool_shared_context import (
    PoolSharedContext,
)
from prompt_improver.database.services.connection.postgres_pool_manager import (
    ConnectionInfo,
    ConnectionMode,
    DatabaseConfig,
    HealthStatus,
    PoolConfiguration,
    PoolState,
    PostgreSQLPoolManager,
    PostgreSQLPoolManager as PostgresPoolManager,
)

# RedisManager removed - use CacheFacade from services.cache for Redis operations
from prompt_improver.database.services.connection.sentinel_manager import (
    FailoverEvent,
    FailoverEventInfo,
    MasterInfo,
    SentinelConfig,
    SentinelInfo,
    SentinelManager,
    SentinelState,
)
from prompt_improver.shared.interfaces.protocols.database import (
    ConnectionPoolCoreProtocol,
    PoolManagerFacadeProtocol,
    PoolMonitoringServiceProtocol,
    PoolScalingManagerProtocol,
)

__all__ = [
    "BurstDetector",
    "ConnectionInfo",
    "ConnectionMetrics",
    "ConnectionMode",
    "ConnectionPoolCore",
    "ConnectionPoolCoreProtocol",
    "DatabaseConfig",
    "FailoverEvent",
    "FailoverEventInfo",
    "HealthStatus",
    "MasterInfo",
    "PoolConfiguration",
    "PoolManagerFacadeProtocol",
    "PoolMonitoringService",
    "PoolMonitoringServiceProtocol",
    "PoolScaler",
    "PoolScalerProtocol",
    "PoolScalingManager",
    "PoolScalingManagerProtocol",
    "PoolSharedContext",
    "PoolState",
    "PostgreSQLPoolManager",
    "PostgresPoolManager",  # Backward compatibility alias
    "ScalingAction",
    "ScalingConfiguration",
    "ScalingDecision",
    "ScalingMetrics",
    "ScalingReason",
    "SentinelConfig",
    "SentinelInfo",
    # "RedisManager" - removed, use CacheFacade from services.cache
    "SentinelManager",
    "SentinelState",
]
