"""Unified connection metrics for PostgreSQL and Redis connections.

Consolidates metrics from:
- database.unified_connection_manager.ConnectionMetrics
- cache.redis_health.ConnectionMetrics
- metrics.system_metrics.ConnectionMetrics

This is the single source of truth for all connection-related metrics.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ConnectionMetrics:
    """Comprehensive connection metrics for all connection types.

    Combines PostgreSQL, Redis, and system connection metrics into a
    single unified structure for consistent monitoring and reporting.
    """

    # Core connection counts
    active_connections: int = 0
    idle_connections: int = 0
    total_connections: int = 0
    connection_errors: int = 0

    # Pool management
    pool_utilization: float = 0.0
    connection_pool_size: int = 0
    connection_pool_available: int = 0
    avg_connection_age: float = 0.0

    # Performance metrics
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0
    failed_connections: int = 0
    wait_time_ms: float = 0.0

    # Connection lifecycle
    connections_created: int = 0
    connections_closed: int = 0
    connection_failures: int = 0
    connection_reuse_count: int = 0
    connections_saved: int = 0

    # Query metrics
    queries_executed: int = 0
    queries_failed: int = 0
    query_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    connection_times: deque = field(default_factory=lambda: deque(maxlen=1000))

    # High availability
    last_failover: float | None = None
    failover_count: int = 0
    health_check_failures: int = 0

    # Circuit breaker
    circuit_breaker_state: str = "closed"
    circuit_breaker_failures: int = 0

    # SLA and efficiency
    sla_compliance_rate: float = 100.0
    pool_efficiency: float = 0.0
    database_load_reduction_percent: float = 0.0

    # Scaling metrics
    last_scale_event: datetime | None = None

    # Redis-specific metrics
    redis_connected_clients: int = 0
    redis_client_recent_max_input_buffer: int = 0
    redis_client_recent_max_output_buffer: int = 0
    redis_blocked_clients: int = 0
    redis_tracking_clients: int = 0
    redis_clients_in_timeout_table: int = 0
    redis_total_connections_received: int = 0
    redis_rejected_connections: int = 0
    redis_maxclients: int = 10000
    redis_connection_utilization: float = 0.0
    redis_connections_per_second: float = 0.0

    # Multi-system coordination
    http_pool_health: bool = True
    redis_pool_health: bool = True
    multi_pool_coordination_active: bool = False

    # Cache metrics integration
    cache_l1_hits: int = 0
    cache_l2_hits: int = 0
    cache_total_requests: int = 0
    cache_hit_rate: float = 0.0
    cache_l1_size: int = 0
    cache_l1_utilization: float = 0.0
    cache_warming_enabled: bool = False
    cache_warming_cycles: int = 0
    cache_warming_keys_warmed: int = 0
    cache_warming_hit_rate: float = 0.0
    cache_response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    cache_operation_stats: dict[str, Any] = field(default_factory=dict)
    cache_health_status: str = "healthy"

    # Legacy/compatibility fields
    mode_specific_metrics: dict[str, Any] = field(default_factory=dict)
    registry_conflicts: int = 0
    registered_models: int = 0

    def record_connection(self) -> None:
        """Record a new connection being established."""
        self.active_connections += 1
        self.total_connections += 1
        self.connections_created += 1

    def record_disconnection(self) -> None:
        """Record a connection being closed."""
        self.active_connections = max(0, self.active_connections - 1)
        self.connections_closed += 1

    def record_query(self, duration_ms: float, success: bool = True) -> None:
        """Record a query execution with timing."""
        self.queries_executed += 1
        self.query_times.append(duration_ms)

        if not success:
            self.queries_failed += 1
            self.error_rate = self.queries_failed / self.queries_executed

        # Update average response time using exponential moving average
        if self.avg_response_time_ms == 0:
            self.avg_response_time_ms = duration_ms
        else:
            alpha = 0.1
            self.avg_response_time_ms = (
                alpha * duration_ms + (1 - alpha) * self.avg_response_time_ms
            )

    def record_connection_time(self, duration_ms: float) -> None:
        """Record connection establishment time."""
        self.connection_times.append(duration_ms)

    def calculate_pool_utilization(self) -> float:
        """Calculate current pool utilization percentage."""
        if self.connection_pool_size > 0:
            self.pool_utilization = (
                self.active_connections / self.connection_pool_size * 100
            )
        return self.pool_utilization

    def calculate_redis_utilization(self) -> None:
        """Calculate Redis connection pool utilization."""
        if self.redis_maxclients > 0:
            self.redis_connection_utilization = (
                self.redis_connected_clients / self.redis_maxclients * 100.0
            )

    def is_healthy(self) -> bool:
        """Check if connection metrics indicate healthy state."""
        # Update utilization metrics
        self.calculate_pool_utilization()
        self.calculate_redis_utilization()

        # Check PostgreSQL health
        if self.pool_utilization > 90.0:
            return False
        if self.error_rate > 0.1:  # 10% error rate threshold
            return False
        if self.circuit_breaker_state == "open":
            return False

        # Check Redis health
        if self.redis_connection_utilization > 90.0:
            return False
        if self.redis_blocked_clients > self.redis_connected_clients * 0.5:
            return False
        return not self.redis_rejected_connections > 0

    def get_efficiency_metrics(self) -> dict[str, float]:
        """Get connection efficiency and optimization metrics."""
        if self.connections_created > 0:
            reuse_rate = self.connection_reuse_count / self.connections_created
            self.pool_efficiency = reuse_rate * 100

            base_connections = self.connection_reuse_count + self.connections_created
            if base_connections > 0:
                self.database_load_reduction_percent = (
                    (base_connections - self.connections_created)
                    / base_connections
                    * 100
                )
                self.connections_saved = self.connection_reuse_count

        return {
            "pool_efficiency": self.pool_efficiency,
            "database_load_reduction_percent": self.database_load_reduction_percent,
            "connections_saved": self.connections_saved,
            "reuse_rate": self.connection_reuse_count
            / max(1, self.connections_created),
        }

    def reset_counters(self) -> None:
        """Reset cumulative counters (useful for periodic reporting)."""
        self.connections_created = 0
        self.connections_closed = 0
        self.connection_failures = 0
        self.queries_executed = 0
        self.queries_failed = 0
        self.error_rate = 0.0
        self.health_check_failures = 0
        self.cache_l1_hits = 0
        self.cache_l2_hits = 0
        self.cache_total_requests = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "total_connections": self.total_connections,
            "pool_utilization": self.pool_utilization,
            "avg_response_time_ms": self.avg_response_time_ms,
            "error_rate": self.error_rate,
            "failed_connections": self.failed_connections,
            "queries_executed": self.queries_executed,
            "queries_failed": self.queries_failed,
            "pool_efficiency": self.pool_efficiency,
            "database_load_reduction_percent": self.database_load_reduction_percent,
            "sla_compliance_rate": self.sla_compliance_rate,
            "circuit_breaker_state": self.circuit_breaker_state,
            "redis_connection_utilization": self.redis_connection_utilization,
            "redis_connected_clients": self.redis_connected_clients,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_health_status": self.cache_health_status,
            "is_healthy": self.is_healthy(),
            "last_scale_event": self.last_scale_event.isoformat()
            if self.last_scale_event
            else None,
        }
