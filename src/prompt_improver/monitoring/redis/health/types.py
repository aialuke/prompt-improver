"""Redis Health Monitoring Types.

Focused types for Redis health monitoring services following SRE best practices.
All types are designed for real-time monitoring with <25ms operations.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class RedisHealthStatus(Enum):
    """Redis health status levels for SRE incident response."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class RedisRole(Enum):
    """Redis server roles for replication monitoring."""

    MASTER = "master"
    SLAVE = "slave"
    REPLICA = "replica"
    STANDALONE = "standalone"


class ConnectionStatus(Enum):
    """Connection pool status for monitoring."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"
    RECOVERING = "recovering"


class AlertLevel(Enum):
    """Alert severity levels for incident management."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RecoveryAction(Enum):
    """Automated recovery actions for circuit breaker patterns."""

    NONE = "none"
    RETRY = "retry"
    FAILOVER = "failover"
    CIRCUIT_BREAK = "circuit_break"
    ESCALATE = "escalate"


@dataclass
class HealthMetrics:
    """Core health metrics for monitoring."""

    # Connection metrics
    ping_latency_ms: float = 0.0
    connection_count: int = 0
    max_connections: int = 0
    connection_utilization: float = 0.0

    # Performance metrics
    hit_rate: float = 0.0
    ops_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    memory_fragmentation_ratio: float = 1.0

    # Status indicators
    status: RedisHealthStatus = RedisHealthStatus.HEALTHY
    is_available: bool = True
    last_check_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    check_duration_ms: float = 0.0

    # Error tracking
    error_count: int = 0
    last_error: str | None = None
    consecutive_failures: int = 0

    def is_healthy(self) -> bool:
        """Check if metrics indicate healthy state."""
        return (
            self.is_available and
            self.ping_latency_ms < 100.0 and
            self.connection_utilization < 90.0 and
            self.hit_rate > 50.0 and
            self.consecutive_failures < 3
        )

    def get_status(self) -> RedisHealthStatus:
        """Get health status based on metrics."""
        if not self.is_available or self.consecutive_failures >= 5:
            return RedisHealthStatus.FAILED

        if not self.is_healthy():
            return RedisHealthStatus.CRITICAL

        if (self.ping_latency_ms > 50.0 or
            self.connection_utilization > 70.0 or
            self.hit_rate < 80.0 or
            self.consecutive_failures > 0):
            return RedisHealthStatus.WARNING

        return RedisHealthStatus.HEALTHY


@dataclass
class ConnectionPoolMetrics:
    """Connection pool health metrics."""

    # Pool status
    status: ConnectionStatus = ConnectionStatus.CONNECTED
    active_connections: int = 0
    idle_connections: int = 0
    max_pool_size: int = 0
    pool_utilization: float = 0.0

    # Connection lifecycle
    connections_created: int = 0
    connections_closed: int = 0
    connection_failures: int = 0
    connection_timeouts: int = 0

    # Timing metrics
    avg_connection_time_ms: float = 0.0
    max_connection_time_ms: float = 0.0

    def calculate_utilization(self):
        """Calculate pool utilization percentage."""
        if self.max_pool_size > 0:
            total_connections = self.active_connections + self.idle_connections
            self.pool_utilization = (total_connections / self.max_pool_size) * 100.0


@dataclass
class PerformanceMetrics:
    """Detailed performance metrics for monitoring."""

    # Command performance
    total_commands: int = 0
    instantaneous_ops_per_sec: int = 0
    avg_ops_per_sec: float = 0.0

    # Cache performance
    keyspace_hits: int = 0
    keyspace_misses: int = 0
    hit_rate_percentage: float = 0.0

    # Memory metrics
    used_memory_bytes: int = 0
    peak_memory_bytes: int = 0
    fragmentation_ratio: float = 1.0
    fragmentation_bytes: int = 0

    # Latency tracking
    latency_samples: list[float] = field(default_factory=list)
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    def add_latency_sample(self, latency_ms: float):
        """Add latency sample and update percentiles."""
        self.latency_samples.append(latency_ms)

        # Keep only last 1000 samples for performance
        if len(self.latency_samples) > 1000:
            self.latency_samples = self.latency_samples[-1000:]

        # Update percentiles
        if self.latency_samples:
            sorted_samples = sorted(self.latency_samples)
            n = len(sorted_samples)

            self.avg_latency_ms = sum(sorted_samples) / n
            self.p50_latency_ms = sorted_samples[int(n * 0.5)]
            self.p95_latency_ms = sorted_samples[int(n * 0.95)]
            self.p99_latency_ms = sorted_samples[int(n * 0.99)]
            self.max_latency_ms = max(sorted_samples)

    def calculate_hit_rate(self):
        """Calculate cache hit rate percentage."""
        total = self.keyspace_hits + self.keyspace_misses
        self.hit_rate_percentage = (self.keyspace_hits / total * 100) if total > 0 else 0.0


@dataclass
class AlertEvent:
    """Alert event for incident management."""

    id: str
    level: AlertLevel
    message: str
    source_service: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metrics: dict[str, Any] | None = None
    resolved: bool = False
    resolution_time: datetime | None = None

    @property
    def duration_seconds(self) -> float | None:
        """Get alert duration in seconds."""
        if self.resolved and self.resolution_time:
            return (self.resolution_time - self.timestamp).total_seconds()
        return None


@dataclass
class RecoveryEvent:
    """Recovery action event for tracking."""

    id: str
    action: RecoveryAction
    trigger_reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    success: bool = False
    error_message: str | None = None
    completion_time: datetime | None = None

    @property
    def execution_time_ms(self) -> float | None:
        """Get recovery execution time in milliseconds."""
        if self.completion_time:
            return (self.completion_time - self.timestamp).total_seconds() * 1000
        return None


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for failure protection."""

    state: str = "closed"  # closed, open, half_open
    failure_count: int = 0
    last_failure_time: datetime | None = None
    next_attempt_time: datetime | None = None
    failure_threshold: int = 5
    timeout_seconds: int = 60

    def can_attempt(self) -> bool:
        """Check if operation can be attempted."""
        if self.state == "closed":
            return True
        if self.state == "open":
            return bool(self.next_attempt_time and datetime.now(UTC) >= self.next_attempt_time)
        # half_open
        return True

    def record_success(self):
        """Record successful operation."""
        self.state = "closed"
        self.failure_count = 0
        self.last_failure_time = None
        self.next_attempt_time = None

    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(UTC)

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.next_attempt_time = datetime.now(UTC).replace(
                second=datetime.now(UTC).second + self.timeout_seconds
            )
