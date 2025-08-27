"""Database service protocol definitions.

Consolidated high-performance database protocols supporting PostgreSQL 15+
with JSONB optimization, connection pooling, and repository patterns.

Consolidated from:
- /core/protocols/database_protocol.py
- /database/protocols.py
- /database/services/connection/pool_protocols.py
- /database/health/services/health_protocols.py

This consolidation maintains strict interface compatibility for SessionManagerProtocol
which is critical for 24+ dependent files across the system.
"""

from contextlib import AbstractAsyncContextManager
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncContextManager,
    Protocol,
    runtime_checkable,
)

# TYPE_CHECKING imports - these are not imported at runtime, only for static analysis
if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class ConnectionMode(Enum):
    """Connection mode for database operations."""
    READ_WRITE = "read_write"
    READ_ONLY = "read_only"
    BATCH = "batch"


class HealthStatus:
    """Health status enumeration for database components."""


# =============================================================================
# SESSION MANAGEMENT PROTOCOLS (HIGHEST PRIORITY - 24+ DEPENDENCIES)
# =============================================================================

@runtime_checkable
class SessionProtocol(Protocol):
    """Protocol for database session operations.

    Provides basic database operations interface without coupling to
    specific database implementation details.
    """

    async def execute(self, query: Any, parameters: dict[str, Any] | None = None) -> Any:
        """Execute a query with optional parameters."""
        ...

    async def fetch_one(self, query: Any, parameters: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Fetch a single row from query result."""
        ...

    async def fetch_all(self, query: Any, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Fetch all rows from query result."""
        ...

    async def commit(self) -> None:
        """Commit current transaction."""
        ...

    async def rollback(self) -> None:
        """Rollback current transaction."""
        ...

    async def close(self) -> None:
        """Close the session."""
        ...


@runtime_checkable
class SessionManagerProtocol(Protocol):
    """Protocol for database session management.

    CRITICAL: This protocol interface must be preserved exactly as it affects 24+ files.
    Any changes to this interface require coordination across the entire codebase.
    """

    async def get_session(self) -> SessionProtocol:
        """Get a new database session."""
        ...

    def session_context(self) -> AbstractAsyncContextManager[SessionProtocol]:
        """Get async context manager for session with automatic cleanup."""
        ...

    def transaction_context(self) -> AbstractAsyncContextManager[SessionProtocol]:
        """Get async context manager for transactional session."""
        ...

    async def health_check(self) -> bool:
        """Check if database connection is healthy."""
        ...

    async def get_connection_info(self) -> dict[str, Any]:
        """Get connection information and statistics."""
        ...

    async def close_all_sessions(self) -> None:
        """Close all active sessions."""
        ...


@runtime_checkable
class QueryExecutorProtocol(Protocol):
    """Protocol for executing database queries without session management."""

    async def execute_query(
        self,
        query: Any,
        parameters: dict[str, Any] | None = None
    ) -> Any:
        """Execute a query and return result."""
        ...

    async def fetch_scalar(
        self,
        query: Any,
        parameters: dict[str, Any] | None = None
    ) -> Any:
        """Execute query and return scalar value."""
        ...

    async def fetch_one_dict(
        self,
        query: Any,
        parameters: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Execute query and return single row as dict."""
        ...

    async def fetch_all_dict(
        self,
        query: Any,
        parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute query and return all rows as list of dicts."""
        ...

    async def execute_in_transaction(
        self,
        queries: list[tuple[Any, dict[str, Any] | None]]
    ) -> list[Any]:
        """Execute multiple queries in a transaction."""
        ...


# =============================================================================
# CORE DATABASE OPERATIONS
# =============================================================================

@runtime_checkable
class DatabaseSessionProtocol(Protocol):
    """Protocol for database session management using SQLAlchemy."""

    async def get_session(self) -> "AsyncContextManager[AsyncSession]":
        """Get async database session context manager."""
        ...

    async def get_session_manager(self) -> Any:
        """Get the session manager instance."""
        ...

    async def health_check(self) -> bool:
        """Check database health."""
        ...


@runtime_checkable
class DatabaseServiceProtocol(Protocol):
    """Protocol for database connection services."""

    async def get_session(
        self, mode: ConnectionMode = ConnectionMode.READ_WRITE
    ) -> "AsyncContextManager[AsyncSession]":
        """Get a database session with specified access mode."""
        ...

    async def health_check(self) -> HealthStatus:
        """Check database connection health."""
        ...

    async def get_metrics(self) -> dict[str, Any]:
        """Get database connection metrics."""
        ...

    async def initialize(self) -> None:
        """Initialize the database service."""
        ...

    async def shutdown(self) -> None:
        """Shutdown the database service and cleanup resources."""
        ...


@runtime_checkable
class DatabaseConfigProtocol(Protocol):
    """Protocol for database configuration."""

    def get_database_url(self) -> str:
        """Get database connection URL."""
        ...

    def get_connection_pool_config(self) -> dict[str, Any]:
        """Get connection pool configuration."""
        ...

    def get_retry_config(self) -> dict[str, Any]:
        """Get retry configuration."""
        ...


# =============================================================================
# POSTGRESQL QUERY OPTIMIZATION
# =============================================================================

@runtime_checkable
class QueryOptimizerProtocol(Protocol):
    """Protocol for PostgreSQL query optimization services."""

    async def optimize_query(self, query: str, params: dict | None = None) -> str:
        """Optimize SQL query for PostgreSQL."""
        ...

    async def analyze_performance(self, query: str) -> dict[str, Any]:
        """Analyze query performance using EXPLAIN."""
        ...

    async def get_execution_plan(self, query: str) -> dict[str, Any]:
        """Get PostgreSQL query execution plan."""
        ...


# =============================================================================
# CONNECTION POOL PROTOCOLS (HIGH PERFORMANCE)
# =============================================================================

@runtime_checkable
class ConnectionPoolCoreProtocol(Protocol):
    """Protocol for core connection pool functionality.

    Handles basic connection management, session creation, and query execution
    with PostgreSQL optimization.
    """

    async def initialize(self) -> bool:
        """Initialize the connection pool core."""
        ...

    async def get_session(
        self, mode: ConnectionMode = ConnectionMode.READ_WRITE
    ) -> "AsyncContextManager[AsyncSession]":
        """Get async SQLAlchemy session with automatic transaction management."""
        ...

    async def get_ha_connection(
        self, pool_name: str = "primary"
    ) -> AsyncContextManager[Any]:  # asyncpg.Connection when available
        """Get direct asyncpg connection from HA pools."""
        ...

    async def execute_cached_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        cache_ttl_seconds: int = 300,
        cache_key_prefix: str = "pg_query"
    ) -> Any:
        """Execute a query with result caching for performance optimization."""
        ...

    async def test_connections(self) -> None:
        """Test all connection types to ensure they work."""
        ...

    async def shutdown(self) -> None:
        """Shutdown the connection pool core and cleanup resources."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Check if the connection pool core is initialized."""
        ...


@runtime_checkable
class PoolScalingManagerProtocol(Protocol):
    """Protocol for dynamic pool scaling management.

    Handles automatic pool size optimization based on utilization patterns
    for high-performance PostgreSQL operations.
    """

    async def optimize_pool_size(self) -> dict[str, Any]:
        """Dynamically optimize pool size based on load patterns."""
        ...

    async def evaluate_scaling_need(self) -> dict[str, Any]:
        """Evaluate if pool scaling is needed based on current metrics."""
        ...

    async def scale_pool(self, new_size: int) -> None:
        """Scale the connection pool to new size."""
        ...

    async def collect_pool_metrics(self) -> dict[str, Any]:
        """Collect current pool metrics from SQLAlchemy engine."""
        ...

    def set_scaling_thresholds(
        self,
        scale_up_threshold: float,
        scale_down_threshold: float
    ) -> None:
        """Set scaling thresholds for automatic scaling."""
        ...

    def get_scaling_metrics(self) -> dict[str, Any]:
        """Get scaling-specific metrics and status."""
        ...

    @property
    def current_pool_size(self) -> int:
        """Get current pool size."""
        ...


@runtime_checkable
class PoolMonitoringServiceProtocol(Protocol):
    """Protocol for pool health monitoring and metrics collection.

    Handles connection monitoring, health checks, and performance tracking
    for PostgreSQL connection pools.
    """

    async def health_check(self) -> dict[str, Any]:
        """Comprehensive health check of the pool manager."""
        ...

    async def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive pool metrics."""
        ...

    def is_healthy(self) -> bool:
        """Quick health status check."""
        ...

    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        ...

    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        ...

    def get_connection_registry(self) -> dict[str, Any]:
        """Get connection registry information."""
        ...

    def record_performance_event(
        self, event_type: str, duration_ms: float, success: bool
    ) -> None:
        """Record a performance event in the monitoring system."""
        ...

    def handle_connection_failure(self, error: Exception) -> None:
        """Handle connection failure and update circuit breaker."""
        ...

    @property
    def circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        ...


@runtime_checkable
class PoolManagerFacadeProtocol(Protocol):
    """Protocol for the unified pool manager facade.

    Combines all pool management components into a single interface
    that orchestrates clean architecture components efficiently.
    """

    async def initialize(self) -> bool:
        """Initialize the pool manager with all components."""
        ...

    async def get_session(
        self, mode: ConnectionMode = ConnectionMode.READ_WRITE
    ) -> "AsyncContextManager[AsyncSession]":
        """Get async SQLAlchemy session with automatic transaction management."""
        ...

    async def get_ha_connection(
        self, pool_name: str = "primary"
    ) -> AsyncContextManager[Any]:  # asyncpg.Connection when available
        """Get direct asyncpg connection from HA pools."""
        ...

    async def optimize_pool_size(self) -> dict[str, Any]:
        """Dynamically optimize pool size based on load patterns."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Comprehensive health check of the pool manager."""
        ...

    async def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive pool metrics."""
        ...

    async def execute_cached_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        cache_ttl_seconds: int = 300,
        cache_key_prefix: str = "pg_query"
    ) -> Any:
        """Execute a query with result caching for performance optimization."""
        ...

    async def shutdown(self) -> None:
        """Shutdown the pool manager and cleanup resources."""
        ...

    def __repr__(self) -> str:
        """String representation of the pool manager."""
        ...


# =============================================================================
# DATABASE HEALTH MONITORING
# =============================================================================

@runtime_checkable
class DatabaseHealthProtocol(Protocol):
    """Protocol for database health monitoring."""

    async def check_connection_pool(self) -> dict[str, Any]:
        """Check connection pool health."""
        ...

    async def check_query_performance(self) -> dict[str, Any]:
        """Check query performance metrics."""
        ...

    async def check_table_health(self) -> dict[str, Any]:
        """Check table health metrics."""
        ...


@runtime_checkable
class DatabaseConnectionServiceProtocol(Protocol):
    """Protocol for database connection monitoring and health assessment.

    Handles connection pool monitoring, connection lifecycle management,
    and connection health assessment for PostgreSQL.
    """

    async def collect_connection_metrics(self) -> dict[str, Any]:
        """Collect comprehensive connection pool metrics."""
        ...

    async def get_connection_details(self) -> list[dict[str, Any]]:
        """Get detailed connection information from pg_stat_activity."""
        ...

    async def get_pool_health_summary(self) -> dict[str, Any]:
        """Get connection pool health summary."""
        ...

    def analyze_connection_ages(self, connections: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze connection age distribution."""
        ...

    def analyze_connection_states(self, connections: list[dict[str, Any]]) -> dict[str, int]:
        """Analyze connection state distribution."""
        ...

    def identify_problematic_connections(self, connections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify connections that may be problematic."""
        ...


@runtime_checkable
class HealthMetricsServiceProtocol(Protocol):
    """Protocol for health metrics collection and performance tracking.

    Handles health metrics collection, performance tracking, and
    PostgreSQL query analysis with optimization recommendations.
    """

    async def collect_query_performance_metrics(self) -> dict[str, Any]:
        """Collect comprehensive query performance metrics."""
        ...

    async def analyze_slow_queries(self) -> list[dict[str, Any]]:
        """Analyze slow queries from pg_stat_statements."""
        ...

    async def analyze_frequent_queries(self) -> list[dict[str, Any]]:
        """Identify frequently executed queries."""
        ...

    async def analyze_io_intensive_queries(self) -> list[dict[str, Any]]:
        """Identify I/O intensive queries."""
        ...

    async def analyze_cache_performance(self) -> dict[str, Any]:
        """Analyze database cache performance."""
        ...

    async def collect_storage_metrics(self) -> dict[str, Any]:
        """Collect storage-related metrics including table sizes and bloat."""
        ...

    async def collect_replication_metrics(self) -> dict[str, Any]:
        """Collect PostgreSQL replication metrics including lag monitoring."""
        ...

    async def collect_lock_metrics(self) -> dict[str, Any]:
        """Collect lock-related metrics including current locks and deadlocks."""
        ...

    async def collect_transaction_metrics(self) -> dict[str, Any]:
        """Collect transaction-related metrics including commit/rollback rates."""
        ...


@runtime_checkable
class AlertingServiceProtocol(Protocol):
    """Protocol for health alerting and notifications.

    Handles health alerting, threshold monitoring, and
    alert escalation and management.
    """

    def calculate_health_score(self, metrics: dict[str, Any]) -> float:
        """Calculate overall health score based on all metrics (0-100)."""
        ...

    def identify_health_issues(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """Identify specific health issues based on metrics."""
        ...

    def generate_recommendations(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate actionable recommendations based on metrics and identified issues."""
        ...

    def check_thresholds(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """Check if any metrics exceed defined thresholds."""
        ...

    async def send_alert(self, alert: dict[str, Any]) -> bool:
        """Send alert notification."""
        ...

    def set_threshold(self, metric_name: str, warning_threshold: float, critical_threshold: float) -> None:
        """Set threshold for a specific metric."""
        ...


@runtime_checkable
class HealthReportingServiceProtocol(Protocol):
    """Protocol for health reporting and dashboards.

    Handles health reporting, historical analysis, and report generation.
    """

    def add_metrics_to_history(self, metrics: dict[str, Any]) -> None:
        """Add metrics to history for trend analysis."""
        ...

    def get_health_trends(self, hours: int = 24) -> dict[str, Any]:
        """Get health trends over the specified time period."""
        ...

    def generate_health_report(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive health report."""
        ...

    def generate_trend_summary(self, recent_metrics: list[dict[str, Any]]) -> str:
        """Generate a human-readable trend summary."""
        ...

    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        ...

    def get_metrics_history(self) -> list[dict[str, Any]]:
        """Get historical metrics data."""
        ...


@runtime_checkable
class DatabaseHealthServiceProtocol(Protocol):
    """Protocol for unified database health monitoring service.

    Combines all health monitoring components into a single interface
    that orchestrates clean architecture components efficiently.
    """

    async def collect_comprehensive_metrics(self) -> dict[str, Any]:
        """Collect comprehensive database health metrics from all services."""
        ...

    async def get_comprehensive_health(self) -> dict[str, Any]:
        """Single unified endpoint for all database health metrics with parallel execution."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Quick health check with essential metrics."""
        ...

    def get_health_trends(self, hours: int = 24) -> dict[str, Any]:
        """Get health trends over the specified time period."""
        ...

    async def get_connection_pool_health_summary(self) -> dict[str, Any]:
        """Get connection pool health summary."""
        ...

    async def analyze_query_performance(self) -> dict[str, Any]:
        """Analyze query performance."""
        ...


# =============================================================================
# COMBINED DATABASE PROTOCOLS
# =============================================================================

@runtime_checkable
class DatabaseProtocol(
    DatabaseSessionProtocol,
    DatabaseConfigProtocol,
    QueryOptimizerProtocol,
    DatabaseHealthProtocol,
    Protocol,
):
    """Combined protocol for all database operations.

    Integrates session management, configuration, query optimization,
    and health monitoring into a single comprehensive interface.
    """


@runtime_checkable
class DatabaseServicesProtocol(Protocol):
    """Protocol for composed database services following composition pattern.

    Represents the complete database services suite including caching,
    locking, pub/sub, and health monitoring capabilities.
    """

    # Core services
    database: DatabaseServiceProtocol
    # Note: Multi-level cache, lock manager, pubsub protocols are in their respective files

    async def initialize_all(self) -> None:
        """Initialize all composed services."""
        ...

    async def shutdown_all(self) -> None:
        """Shutdown all composed services."""
        ...

    async def health_check_all(self) -> dict[str, HealthStatus]:
        """Health check all services."""
        ...

    async def get_metrics_all(self) -> dict[str, Any]:
        """Get metrics from all services."""
        ...
