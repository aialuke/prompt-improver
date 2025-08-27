"""PostgreSQL Pool Manager Facade - Unified interface for decomposed components.

Orchestrates decomposed pool management components following clean architecture
principles while providing a single, efficient interface for all operations.
"""

import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.database.services.connection.connection_pool_core import (
    ConnectionPoolCore,
)
from prompt_improver.database.services.connection.pool_monitoring_service import (
    PoolMonitoringService,
)
from prompt_improver.database.services.connection.pool_scaling_manager import (
    PoolScalingManager,
)
from prompt_improver.database.services.connection.pool_shared_context import (
    ConnectionMode,
    DatabaseConfig,
    HealthStatus,
    PoolConfiguration,
    PoolSharedContext,
    PoolState,
)
from prompt_improver.services.cache.cache_facade import CacheFacade as CacheManager

logger = logging.getLogger(__name__)


class PostgreSQLPoolManager:
    """Unified PostgreSQL connection pool manager facade.

    Orchestrates decomposed components for high-performance database operations:

    - ConnectionPoolCore: Core connection management and session creation
    - PoolScalingManager: Dynamic pool scaling and optimization
    - PoolMonitoringService: Health monitoring and metrics collection

    Provides clean architecture with separated concerns, optimized performance,
    and comprehensive connection management capabilities.
    """

    def __init__(
        self,
        db_config: DatabaseConfig,
        pool_config: PoolConfiguration,
        service_name: str = "postgres_pool_manager",
        cache_manager: CacheManager | None = None,
    ) -> None:
        # Initialize shared context
        self.shared_context = PoolSharedContext(
            db_config=db_config,
            pool_config=pool_config,
            service_name=service_name,
            cache_manager=cache_manager,
        )

        # Initialize components with dependency injection
        self.connection_core = ConnectionPoolCore(self.shared_context)
        self.scaling_manager = PoolScalingManager(self.shared_context)
        self.monitoring_service = PoolMonitoringService(
            self.shared_context,
            scaling_manager=self.scaling_manager,
        )

        # Maintain compatibility properties
        self.db_config = db_config
        self.pool_config = pool_config
        self.service_name = service_name
        self.cache_manager = cache_manager

        logger.info(f"PostgreSQL PoolManager facade initialized: {service_name}")
        logger.info(
            f"Pool config: size={pool_config.pool_size}, "
            f"max_overflow={pool_config.max_overflow}, timeout={pool_config.timeout}s"
        )

    async def initialize(self) -> bool:
        """Initialize the pool manager with all components."""
        try:
            # Initialize connection core first
            success = await self.connection_core.initialize()
            if not success:
                return False

            # Mark as initialized in shared context
            self.shared_context.is_initialized = True
            self.shared_context.pool_state = PoolState.HEALTHY
            self.shared_context.health_status = HealthStatus.HEALTHY

            # Start monitoring
            await self.monitoring_service.start_monitoring()

            logger.info(
                f"PostgreSQL PoolManager initialized successfully: {self.service_name}"
            )
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize PostgreSQL PoolManager: {e}")
            self.shared_context.pool_state = PoolState.UNHEALTHY
            self.shared_context.health_status = HealthStatus.UNHEALTHY
            raise

    @contextlib.asynccontextmanager
    async def get_session(
        self, mode: ConnectionMode = ConnectionMode.READ_WRITE
    ) -> AsyncIterator[AsyncSession]:
        """Get async SQLAlchemy session with automatic transaction management."""
        async with self.connection_core.get_session(mode) as session:
            yield session

    @contextlib.asynccontextmanager
    async def get_ha_connection(
        self, pool_name: str = "primary"
    ) -> AsyncIterator[asyncpg.Connection]:
        """Get direct asyncpg connection from HA pools."""
        async with self.connection_core.get_ha_connection(pool_name) as conn:
            yield conn

    async def optimize_pool_size(self) -> dict[str, Any]:
        """Dynamically optimize pool size based on load patterns."""
        return await self.scaling_manager.optimize_pool_size()

    async def health_check(self) -> dict[str, Any]:
        """Comprehensive health check of the pool manager."""
        return await self.monitoring_service.health_check()

    async def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive pool metrics."""
        return await self.monitoring_service.get_metrics()

    async def execute_cached_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        cache_ttl_seconds: int = 300,
        cache_key_prefix: str = "pg_query"
    ) -> Any:
        """Execute a query with result caching for performance optimization."""
        return await self.connection_core.execute_cached_query(
            query, params, cache_ttl_seconds, cache_key_prefix
        )

    async def shutdown(self) -> None:
        """Shutdown the pool manager and cleanup resources."""
        logger.info(f"Shutting down PostgreSQL PoolManager: {self.service_name}")

        try:
            # Stop monitoring first
            await self.monitoring_service.stop_monitoring()

            # Shutdown connection core
            await self.connection_core.shutdown()

            # Clear shared context state
            self.shared_context.is_initialized = False
            self.shared_context.performance_window.clear()

            logger.info(
                f"PostgreSQL PoolManager shutdown complete: {self.service_name}"
            )

        except Exception as e:
            logger.exception(f"Error during PostgreSQL PoolManager shutdown: {e}")

    # Compatibility properties for tests and legacy interfaces
    @property
    def _pool_state(self):
        """Pool state from shared context."""
        return self.shared_context.pool_state

    @property
    def _health_status(self):
        """Health status from shared context."""
        return self.shared_context.health_status

    @property
    def _is_initialized(self):
        """Initialization status from shared context."""
        return self.shared_context.is_initialized

    @property
    def min_pool_size(self):
        """Minimum pool size from configuration."""
        return self.pool_config.pool_size

    @property
    def max_pool_size(self):
        """Maximum pool size (calculated from pool size and max overflow)."""
        return min(self.pool_config.pool_size * 5, 100)  # Same logic as original

    @property
    def current_pool_size(self):
        """Current pool size from shared context."""
        return self.shared_context.current_pool_size

    @property
    def scale_up_threshold(self):
        """Scale up threshold from scaling manager."""
        return 0.8  # Default from original implementation

    @property
    def scale_down_threshold(self):
        """Scale down threshold from scaling manager."""
        return 0.3  # Default from original implementation

    @property
    def scale_cooldown_seconds(self):
        """Scale cooldown seconds from scaling manager."""
        return 60  # Default from original implementation

    @property
    def _circuit_breaker_threshold(self):
        """Circuit breaker threshold from monitoring service."""
        return getattr(self, '__circuit_breaker_threshold', 5)  # Default from original

    @_circuit_breaker_threshold.setter
    def _circuit_breaker_threshold(self, value) -> None:
        """Set circuit breaker threshold."""
        self.__circuit_breaker_threshold = value

    @property
    def _circuit_breaker_timeout(self):
        """Circuit breaker timeout from monitoring service."""
        return getattr(self, '__circuit_breaker_timeout', 60)  # Default from original

    @_circuit_breaker_timeout.setter
    def _circuit_breaker_timeout(self, value) -> None:
        """Set circuit breaker timeout."""
        self.__circuit_breaker_timeout = value

    @property
    def _circuit_breaker_failures(self):
        """Circuit breaker failures count from monitoring service."""
        return getattr(self, '__circuit_breaker_failures', 0)  # Default from original

    @_circuit_breaker_failures.setter
    def _circuit_breaker_failures(self, value) -> None:
        """Set circuit breaker failures count."""
        self.__circuit_breaker_failures = value

    def _is_circuit_breaker_open(self):
        """Check if circuit breaker is open based on failure threshold."""
        return self._circuit_breaker_failures >= self._circuit_breaker_threshold

    def __repr__(self) -> str:
        """String representation maintaining original format."""
        return (
            f"PostgreSQLPoolManager(service={self.service_name}, "
            f"state={self.shared_context.pool_state.value}, "
            f"pool_size={self.shared_context.current_pool_size})"
        )
