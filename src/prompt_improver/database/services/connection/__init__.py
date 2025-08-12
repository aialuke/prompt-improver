"""Database connection management services."""

from .connection_metrics import ConnectionMetrics
from .pool_scaler import (
    BurstDetector,
    PoolScaler,
    PoolScalerProtocol,
    ScalingAction,
    ScalingConfiguration,
    ScalingDecision,
    ScalingMetrics,
    ScalingReason,
)
from .postgres_pool_manager import (
    ConnectionInfo,
    ConnectionMode,
    DatabaseConfig,
    HealthStatus,
    PoolConfiguration,
    PoolState,
    PostgreSQLPoolManager,
    PostgreSQLPoolManager as PostgresPoolManager,  # Alias for backward compatibility
)
from .redis_manager import (
    RedisConfig,
    RedisHealthStatus,
    RedisManager,
    RedisMode,
    RedisNodeInfo,
)
from .sentinel_manager import (
    FailoverEvent,
    FailoverEventInfo,
    MasterInfo,
    SentinelConfig,
    SentinelInfo,
    SentinelManager,
    SentinelState,
)

__all__ = [
    "ConnectionMetrics",
    "PoolScaler",
    "ScalingConfiguration",
    "ScalingMetrics",
    "ScalingDecision",
    "ScalingAction",
    "ScalingReason",
    "BurstDetector",
    "PoolScalerProtocol",
    "PostgreSQLPoolManager",
    "PostgresPoolManager",  # Backward compatibility alias
    "DatabaseConfig",
    "PoolConfiguration",
    "PoolState",
    "HealthStatus",
    "ConnectionMode",
    "ConnectionInfo",
    "RedisManager",
    "RedisConfig",
    "RedisMode",
    "RedisHealthStatus",
    "RedisNodeInfo",
    "SentinelManager",
    "SentinelConfig",
    "SentinelState",
    "FailoverEvent",
    "SentinelInfo",
    "MasterInfo",
    "FailoverEventInfo",
]
