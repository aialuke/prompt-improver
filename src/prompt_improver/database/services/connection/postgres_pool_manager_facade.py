"""PostgreSQL Pool Manager Facade - Unified interface for decomposed components.

Provides backward compatibility with the original PostgreSQLPoolManager
while orchestrating the decomposed components following clean architecture.
"""

import contextlib
import logging
from typing import Any, AsyncIterator, Dict, Optional

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.database.services.cache.cache_manager import CacheManager
from prompt_improver.database.services.connection.connection_pool_core import (
    ConnectionPoolCore,
)
from prompt_improver.database.services.connection.pool_monitoring_service import (
    PoolMonitoringService,
)
from prompt_improver.database.services.connection.pool_protocols import (
    PoolManagerFacadeProtocol,
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

logger = logging.getLogger(__name__)


class PostgreSQLPoolManager:
    """Unified PostgreSQL connection pool manager facade.
    
    Orchestrates the decomposed components while maintaining identical
    functionality and interface to the original god object:
    
    - ConnectionPoolCore: Core connection management and session creation
    - PoolScalingManager: Dynamic pool scaling and optimization
    - PoolMonitoringService: Health monitoring and metrics collection
    
    This facade provides backward compatibility while enabling clean
    architecture with separated concerns.
    """
    
    def __init__(
        self,
        db_config: DatabaseConfig,
        pool_config: PoolConfiguration,
        service_name: str = "postgres_pool_manager",
        cache_manager: Optional[CacheManager] = None,
    ):
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
            logger.error(f"Failed to initialize PostgreSQL PoolManager: {e}")
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
    
    async def optimize_pool_size(self) -> Dict[str, Any]:
        """Dynamically optimize pool size based on load patterns."""
        return await self.scaling_manager.optimize_pool_size()
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the pool manager."""
        return await self.monitoring_service.health_check()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pool metrics."""
        return await self.monitoring_service.get_metrics()
    
    async def execute_cached_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
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
            logger.error(f"Error during PostgreSQL PoolManager shutdown: {e}")
    
    # Compatibility properties and methods for backward compatibility
    
    @property
    def _is_initialized(self) -> bool:
        """Backward compatibility property."""
        return self.shared_context.is_initialized
    
    @property
    def _pool_state(self) -> PoolState:
        """Backward compatibility property."""
        return self.shared_context.pool_state
    
    @property
    def _health_status(self) -> HealthStatus:
        """Backward compatibility property."""
        return self.shared_context.health_status
    
    @property
    def current_pool_size(self) -> int:
        """Get current pool size."""
        return self.shared_context.current_pool_size
    
    @property
    def min_pool_size(self) -> int:
        """Get minimum pool size."""
        return self.shared_context.min_pool_size
    
    @property
    def max_pool_size(self) -> int:
        """Get maximum pool size."""
        return self.shared_context.max_pool_size
    
    @property
    def metrics(self):
        """Get metrics object for backward compatibility."""
        return self.shared_context.metrics
    
    @property
    def performance_window(self):
        """Get performance window for backward compatibility."""
        return self.shared_context.performance_window
    
    # Additional backward compatibility properties for tests
    
    @property
    def scale_up_threshold(self) -> float:
        """Get scale up threshold for backward compatibility."""
        return self.shared_context.scale_up_threshold
    
    @property
    def scale_down_threshold(self) -> float:
        """Get scale down threshold for backward compatibility."""
        return self.shared_context.scale_down_threshold
    
    @property
    def scale_cooldown_seconds(self) -> int:
        """Get scale cooldown seconds for backward compatibility."""
        return self.shared_context.scale_cooldown_seconds
    
    @property
    def _circuit_breaker_threshold(self) -> int:
        """Get circuit breaker threshold for backward compatibility."""
        return self.shared_context.circuit_breaker_threshold
    
    @property
    def _circuit_breaker_failures(self) -> int:
        """Get circuit breaker failures for backward compatibility."""
        return self.shared_context.circuit_breaker_failures
    
    @property
    def _circuit_breaker_timeout(self) -> int:
        """Get circuit breaker timeout for backward compatibility."""
        return self.shared_context.circuit_breaker_timeout
    
    @property
    def _connection_registry(self):
        """Get connection registry for backward compatibility."""
        return self.shared_context.connection_registry
    
    @property
    def _async_engine(self):
        """Get async engine for backward compatibility."""
        return self.shared_context.async_engine
    
    @property
    def _async_session_factory(self):
        """Get async session factory for backward compatibility."""
        return self.shared_context.async_session_factory
    
    def _is_circuit_breaker_open(self) -> bool:
        """Backward compatibility method."""
        return self.shared_context.is_circuit_breaker_open()
    
    def _handle_connection_failure(self, error: Exception) -> None:
        """Backward compatibility method."""
        self.monitoring_service.handle_connection_failure(error)
    
    def _record_performance_event(
        self, event_type: str, duration_ms: float, success: bool
    ) -> None:
        """Backward compatibility method."""
        self.monitoring_service.record_performance_event(
            event_type, duration_ms, success
        )
    
    def _update_response_time(self, response_time_ms: float) -> None:
        """Backward compatibility method - now handled internally."""
        # This method is now handled internally by the connection core
        # Keep for backward compatibility but delegate to shared metrics
        alpha = 0.1
        if self.shared_context.metrics.avg_response_time_ms == 0:
            self.shared_context.metrics.avg_response_time_ms = response_time_ms
        else:
            self.shared_context.metrics.avg_response_time_ms = (
                alpha * response_time_ms
                + (1 - alpha) * self.shared_context.metrics.avg_response_time_ms
            )
    
    async def _collect_pool_metrics(self) -> Dict[str, Any]:
        """Backward compatibility method."""
        return await self.scaling_manager.collect_pool_metrics()
    
    async def _scale_pool(self, new_size: int) -> None:
        """Backward compatibility method."""
        await self.scaling_manager.scale_pool(new_size)
    
    async def _evaluate_scaling(self) -> None:
        """Backward compatibility method."""
        await self.scaling_manager.perform_background_scaling_evaluation()
    
    def __repr__(self) -> str:
        """String representation maintaining original format."""
        return (
            f"PostgreSQLPoolManager(service={self.service_name}, "
            f"state={self.shared_context.pool_state.value}, "
            f"pool_size={self.shared_context.current_pool_size})"
        )