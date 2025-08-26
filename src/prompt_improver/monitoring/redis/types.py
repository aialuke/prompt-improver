"""Redis Health Monitoring Types.

Type definitions for Redis health monitoring services.
Provides clean, focused data structures for each monitoring aspect.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RedisHealthStatus(Enum):
    """Redis health status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class ConnectionHealthStatus(Enum):
    """Redis connection health status."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class RedisAlertLevel(Enum):
    """Redis alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class RedisHealthConfig:
    """Configuration for Redis health monitoring."""

    # Connection settings
    host: str = "localhost"
    port: int = 6379
    timeout_seconds: float = 5.0

    # Health check settings
    check_interval_seconds: float = 30.0
    max_connection_attempts: int = 3

    # Performance thresholds
    max_latency_ms: float = 100.0
    min_hit_rate_percentage: float = 80.0
    max_memory_usage_percentage: float = 85.0
    max_fragmentation_ratio: float = 2.0

    # Alerting settings
    enable_alerts: bool = True
    alert_cooldown_seconds: int = 300

    # Metrics settings
    metrics_retention_hours: int = 24
    large_key_threshold_mb: float = 10.0


@dataclass
class RedisPerformanceMetrics:
    """Redis performance metrics."""

    # Latency metrics
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # Throughput metrics
    ops_per_second: float = 0.0
    instantaneous_ops_per_sec: int = 0
    total_commands_processed: int = 0

    # Cache performance
    hit_rate_percentage: float = 0.0
    keyspace_hits: int = 0
    keyspace_misses: int = 0

    # Resource metrics
    memory_usage_mb: float = 0.0
    memory_usage_percentage: float = 0.0
    fragmentation_ratio: float = 1.0

    # Connection metrics
    connected_clients: int = 0
    blocked_clients: int = 0
    rejected_connections: int = 0


@dataclass
class RedisHealthMetrics:
    """Comprehensive Redis health metrics."""

    # Connection status
    connection_status: ConnectionHealthStatus = ConnectionHealthStatus.DISCONNECTED
    connection_latency_ms: float = 0.0
    connection_uptime_seconds: float = 0.0

    # Performance metrics
    performance: RedisPerformanceMetrics = field(default_factory=RedisPerformanceMetrics)

    # Health indicators
    is_healthy: bool = False
    health_status: RedisHealthStatus = RedisHealthStatus.FAILED

    # System information
    redis_version: str = ""
    uptime_seconds: int = 0

    # Persistence status
    last_save_time: datetime | None = None
    unsaved_changes: int = 0

    # Replication status
    role: str = "unknown"
    connected_slaves: int = 0
    replication_lag_seconds: float = 0.0

    # Large keys and slow queries
    large_keys_count: int = 0
    slow_queries_count: int = 0

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RedisHealthResult:
    """Result of a Redis health check operation."""

    # Basic result information
    success: bool = False
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Health metrics
    metrics: RedisHealthMetrics | None = None

    # Status and issues
    status: RedisHealthStatus = RedisHealthStatus.FAILED
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    # Error information
    error: str | None = None
    error_details: dict[str, Any] | None = None

    # Component information
    component_name: str = "redis"
    check_type: str = "health"


@dataclass
class RedisAlert:
    """Redis monitoring alert."""

    level: RedisAlertLevel
    title: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    component: str = "redis"
    resolved: bool = False
    alert_id: str = ""


@dataclass
class RedisConnectionInfo:
    """Redis connection information."""

    host: str
    port: int
    is_connected: bool = False
    connection_time_ms: float = 0.0
    last_ping_ms: float = 0.0
    error_count: int = 0
    last_error: str | None = None
    uptime_seconds: float = 0.0
    connection_pool_size: int = 0
    active_connections: int = 0
