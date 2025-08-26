"""Shared database types and enums following 2025 Python best practices.

This module contains core enums and types extracted from the monolithic
UnifiedConnectionManager to support clean service composition architecture.

Following research-validated best practices:
- UPPER_CASE enum naming convention
- Type-safe enum comparisons
- StrEnum for string-based constants (Python 3.11+)
- Clear, descriptive enum values
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TypedDict


class ManagerMode(Enum):
    """Manager operation modes optimized for different use cases."""

    MCP_SERVER = "mcp_server"
    ML_TRAINING = "ml_training"
    ADMIN = "admin"
    ASYNC_MODERN = "async_modern"
    HIGH_AVAILABILITY = "ha"


class PoolState(Enum):
    """Connection pool operational states."""

    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    SCALING_UP = "scaling_up"
    SCALING_DOWN = "scaling_down"
    DEGRADED = "degraded"
    FAILED = "failed"


class ConnectionMode(Enum):
    """Database connection access modes."""

    READ_WRITE = "read_write"
    READ_ONLY = "read_only"
    ADMIN = "admin"
    BATCH = "batch"


@dataclass
class ConnectionInfo:
    """Information about an active database connection."""

    connection_id: str
    created_at: datetime
    last_used: datetime
    query_count: int
    mode: ConnectionMode
    pool_name: str
    is_active: bool = True


class DatabaseMetrics(TypedDict):
    """Type-safe dictionary for database metrics."""

    total_connections: int
    active_connections: int
    pool_size: int
    query_count: int
    error_count: int
    avg_response_time: float
    health_status: str


class CacheMetrics(TypedDict):
    """Type-safe dictionary for cache metrics."""

    l1_hits: int
    l1_misses: int
    l2_hits: int
    l2_misses: int
    total_operations: int
    hit_ratio: float


class ServiceConfiguration(TypedDict, total=False):
    """Type-safe service configuration dictionary."""

    mode: str
    enable_caching: bool
    enable_health_checks: bool
    max_connections: int
    timeout_seconds: int
    retry_attempts: int
    debug_logging: bool


@dataclass
class PoolConfiguration:
    """Intelligent pool configuration based on usage patterns."""

    mode: ManagerMode
    pg_pool_size: int
    pg_max_overflow: int
    pg_timeout: float
    redis_pool_size: int
    enable_ha: bool = False
    enable_circuit_breaker: bool = False
    enable_l1_cache: bool = True
    l1_cache_size: int = 1000
    l2_cache_ttl: int = 3600
    enable_cache_warming: bool = True
    warming_threshold: float = 2.0
    warming_interval: int = 300
    max_warming_keys: int = 50

    def __post_init__(self):
        """Validate pool configuration values."""
        if self.pg_pool_size <= 0:
            raise ValueError(
                f"pg_pool_size must be greater than 0, got: {self.pg_pool_size}"
            )
        if self.pg_max_overflow < 0:
            raise ValueError(
                f"pg_max_overflow must be >= 0, got: {self.pg_max_overflow}"
            )
        if self.pg_timeout <= 0:
            raise ValueError(
                f"pg_timeout must be greater than 0, got: {self.pg_timeout}"
            )
        if self.redis_pool_size <= 0:
            raise ValueError(
                f"redis_pool_size must be greater than 0, got: {self.redis_pool_size}"
            )
        if self.l1_cache_size <= 0:
            raise ValueError(
                f"l1_cache_size must be greater than 0, got: {self.l1_cache_size}"
            )
        if self.l2_cache_ttl <= 0:
            raise ValueError(
                f"l2_cache_ttl must be greater than 0, got: {self.l2_cache_ttl}"
            )
        if self.warming_interval <= 0:
            raise ValueError(
                f"warming_interval must be greater than 0, got: {self.warming_interval}"
            )
        if self.max_warming_keys <= 0:
            raise ValueError(
                f"max_warming_keys must be greater than 0, got: {self.max_warming_keys}"
            )

    @classmethod
    def for_mode(cls, mode: ManagerMode) -> "PoolConfiguration":
        """Create pool configuration optimized for specific mode."""
        configs = {
            ManagerMode.MCP_SERVER: cls(
                mode=mode,
                pg_pool_size=20,
                pg_max_overflow=10,
                pg_timeout=0.2,
                redis_pool_size=10,
                enable_circuit_breaker=True,
                enable_l1_cache=True,
                l1_cache_size=2000,
                l2_cache_ttl=900,
                enable_cache_warming=True,
                warming_threshold=3.0,
                warming_interval=120,
            ),
            ManagerMode.ML_TRAINING: cls(
                mode=mode,
                pg_pool_size=15,
                pg_max_overflow=10,
                pg_timeout=5.0,
                redis_pool_size=8,
                enable_ha=True,
                enable_l1_cache=True,
                l1_cache_size=1500,
                l2_cache_ttl=1800,
                enable_cache_warming=True,
                warming_threshold=1.5,
                warming_interval=300,
            ),
            ManagerMode.ADMIN: cls(
                mode=mode,
                pg_pool_size=5,
                pg_max_overflow=2,
                pg_timeout=10.0,
                redis_pool_size=3,
                enable_ha=True,
                enable_l1_cache=True,
                l1_cache_size=500,
                l2_cache_ttl=7200,
                enable_cache_warming=False,
            ),
            ManagerMode.ASYNC_MODERN: cls(
                mode=mode,
                pg_pool_size=12,
                pg_max_overflow=8,
                pg_timeout=5.0,
                redis_pool_size=6,
                enable_circuit_breaker=True,
                enable_l1_cache=True,
                l1_cache_size=1000,
                l2_cache_ttl=3600,
                enable_cache_warming=True,
                warming_threshold=2.0,
                warming_interval=300,
            ),
            ManagerMode.HIGH_AVAILABILITY: cls(
                mode=mode,
                pg_pool_size=20,
                pg_max_overflow=20,
                pg_timeout=10.0,
                redis_pool_size=10,
                enable_ha=True,
                enable_circuit_breaker=True,
                enable_l1_cache=True,
                l1_cache_size=2500,
                l2_cache_ttl=1800,
                enable_cache_warming=True,
                warming_threshold=1.0,
                warming_interval=180,
            ),
        }
        return configs.get(mode, configs[ManagerMode.ASYNC_MODERN])


# Security-related types moved from unified_connection_manager
@dataclass
class SecurityThreatScore:
    """Security threat assessment for operations."""

    level: str = "low"
    score: float = 0.0
    factors: list[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


@dataclass
class SecurityValidationResult:
    """Comprehensive security validation results."""

    validated: bool = False
    validation_method: str = "none"
    validation_timestamp: float = field(default_factory=time.time)
    validation_duration_ms: float = 0.0
    security_incidents: list[str] = field(default_factory=list)
    rate_limit_status: str = "unknown"
    encryption_required: bool = False
    audit_trail_id: str | None = None


@dataclass
class SecurityPerformanceMetrics:
    """Security operation performance tracking."""

    authentication_time_ms: float = 0.0
    authorization_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    total_security_overhead_ms: float = 0.0
    operations_count: int = 0
    last_performance_check: float = field(default_factory=time.time)


# Security exceptions moved from unified_connection_manager
class RedisSecurityError(Exception):
    """Redis security operation error."""
