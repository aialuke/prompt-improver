"""Protocol definitions for PostgreSQL pool manager components.

Defines protocol interfaces for the decomposed pool manager components
following clean architecture principles and dependency inversion.
"""

# Removed core.protocols imports that trigger DI container chain
# Local definitions to maintain database layer isolation
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession


class ConnectionMode(Enum):
    """Connection mode for database operations."""
    READ_WRITE = "read_write"
    READ_ONLY = "read_only"
    BATCH = "batch"


@runtime_checkable
class ConnectionPoolCoreProtocol(Protocol):
    """Protocol for core connection pool functionality.

    Handles basic connection management, session creation, and query execution.
    """

    async def initialize(self) -> bool:
        """Initialize the connection pool core."""
        ...

    async def get_session(
        self, mode: ConnectionMode = ConnectionMode.READ_WRITE
    ) -> AbstractAsyncContextManager[AsyncSession]:
        """Get async SQLAlchemy session with automatic transaction management."""
        ...

    async def get_ha_connection(
        self, pool_name: str = "primary"
    ) -> AbstractAsyncContextManager[asyncpg.Connection]:
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

    Handles automatic pool size optimization based on utilization patterns.
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

    Handles connection monitoring, health checks, and performance tracking.
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
    ) -> AbstractAsyncContextManager[AsyncSession]:
        """Get async SQLAlchemy session with automatic transaction management."""
        ...

    async def get_ha_connection(
        self, pool_name: str = "primary"
    ) -> AbstractAsyncContextManager[asyncpg.Connection]:
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
