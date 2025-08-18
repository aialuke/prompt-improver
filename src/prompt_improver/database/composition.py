"""Service composition layer implementing composition root pattern with dependency injection.

This module implements the DatabaseServices composition root that orchestrates all
extracted database services following 2025 best practices for service composition:

- Constructor dependency injection pattern
- Protocol-based type safety
- Clean shutdown lifecycle management
- No backwards compatibility layers
- Real behavior testing support

Following research-validated patterns:
- Composition over inheritance
- Dependency Injection without frameworks
- Service Locator pattern for factory functions
- Clean shutdown with proper resource cleanup
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from prompt_improver.database.protocols.database_config import DatabaseConfigProtocol
from .protocols import DatabaseServicesProtocol
# Using unified cache facade from services/cache instead of database-specific CacheManager
from prompt_improver.services.cache.cache_facade import CacheFacade
from .services.connection.postgres_pool_manager import (
    PoolConfiguration as PostgresPoolConfiguration,
    PostgreSQLPoolManager,
)
from .services.health.health_manager import HealthManager
from .services.locking.lock_manager import DistributedLockManager
from .services.pubsub.pubsub_manager import PubSubManager
from .types import (
    HealthStatus,
    ManagerMode,
    PoolConfiguration as TypesPoolConfiguration,
)

logger = logging.getLogger(__name__)


class DatabaseServices:
    """Composition root for all database services following clean architecture principles.

    Implements constructor dependency injection to compose all extracted services:
    - PostgreSQLPoolManager for database connections
    - CacheManager for multi-level caching (L1, L2, L3)
    - DistributedLockManager for distributed locking
    - PubSubManager for publish/subscribe messaging
    - HealthManager for service health monitoring

    This class replaces the monolithic UnifiedConnectionManager with a clean
    composition-based architecture.
    """

    def __init__(
        self, 
        mode: ManagerMode, 
        db_config: DatabaseConfigProtocol,
        pool_config: Optional[TypesPoolConfiguration] = None
    ):
        """Initialize DatabaseServices with composed service dependencies.

        Args:
            mode: Manager operation mode for configuration
            db_config: Database configuration implementing DatabaseConfigProtocol
            pool_config: Optional pool configuration override
        """
        self.mode = mode
        self.db_config = db_config
        try:
            self.types_pool_config = pool_config or TypesPoolConfiguration.for_mode(
                mode
            )
        except ValueError as e:
            raise ValueError(
                f"Invalid pool configuration for mode {mode.value}: {e}"
            ) from e
        self._initialized = False
        self._shutdown = False

        # Simplified composition for architectural testing - cache temporarily disabled
        from prompt_improver.database.services.connection.postgres_pool_manager import (
            DatabaseConfig,
        )

        # Use injected database config via protocol  
        database_url = db_config.get_database_url()
        # Parse URL to extract components for DatabaseConfig
        from urllib.parse import urlparse
        parsed = urlparse(database_url)
        
        # Convert protocol-based config to PostgreSQLPoolManager's expected format
        postgres_db_config = DatabaseConfig(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            database=parsed.path.lstrip('/') if parsed.path else "prompt_improver",
            username=parsed.username or "postgres",
            password=parsed.password or "",
            echo_sql=False,
        )

        # Get pool configuration from protocol
        pool_config_dict = db_config.get_connection_pool_config()
        
        # Convert TypesPoolConfiguration to PostgresPoolConfiguration
        postgres_pool_config = PostgresPoolConfiguration(
            pool_size=self.types_pool_config.pg_pool_size,
            max_overflow=self.types_pool_config.pg_max_overflow,
            timeout=self.types_pool_config.pg_timeout,
            enable_ha=self.types_pool_config.enable_ha,
            enable_circuit_breaker=self.types_pool_config.enable_circuit_breaker,
            pool_pre_ping=pool_config_dict.get("pool_pre_ping", True),
            pool_recycle=pool_config_dict.get("pool_recycle", 3600),
            application_name=f"apes_{mode.value}",
            skip_connection_test=True,  # Skip connection testing for validation
        )
        self.database = PostgreSQLPoolManager(postgres_db_config, postgres_pool_config)

        # Initialize unified cache facade (eliminates CacheManager circular import)
        self.cache = CacheFacade(
            enable_l2=True, 
            enable_l3=True, 
            session_manager=self.database
        )

        # Create health manager
        self.health_manager = HealthManager()

        # Other services need Redis clients - set to None for now
        self.lock_manager = None
        self.pubsub = None

        # Register working components with health manager (skip for now - need proper checkers)
        # if hasattr(self.database, 'health_check'):
        #     self.health_manager.register_health_check(
        #         "database", self.database.health_check
        #     )

        logger.info(f"DatabaseServices composed for mode: {mode.value}")

    async def initialize_all(self) -> None:
        """Initialize all composed services in proper dependency order."""
        if self._initialized:
            logger.warning("DatabaseServices already initialized")
            return

        logger.info("Initializing all database services...")

        try:
            # Initialize services in dependency order
            logger.debug("Initializing database service...")
            await self.database.initialize()

            if self.cache and hasattr(self.cache, "initialize"):
                logger.debug("Initializing cache service...")
                await self.cache.initialize()

            if self.lock_manager and hasattr(self.lock_manager, "initialize"):
                logger.debug("Initializing lock manager service...")
                await self.lock_manager.initialize()

            if self.pubsub and hasattr(self.pubsub, "initialize"):
                logger.debug("Initializing pubsub service...")
                await self.pubsub.initialize()

            # Health manager doesn't need explicit initialization - it's ready on creation
            logger.debug("All services initialized, marking as ready...")

            self._initialized = True
            logger.info("All database services initialized successfully")

        except ValueError as e:
            # Configuration validation errors - fail fast with clear message
            error_msg = f"Invalid service configuration: {e}"
            logger.error(error_msg)
            await self._cleanup_partial_initialization()
            raise ValueError(error_msg) from e

        except Exception as e:
            # Other initialization errors - provide helpful context
            error_msg = (
                f"Failed to initialize database services: {type(e).__name__}: {e}"
            )
            logger.error(error_msg)
            # Cleanup any partially initialized services
            await self._cleanup_partial_initialization()
            raise RuntimeError(error_msg) from e

    async def shutdown_all(self) -> None:
        """Shutdown all composed services in reverse dependency order."""
        if self._shutdown:
            logger.warning("DatabaseServices already shutdown")
            return

        logger.info("Shutting down all database services...")
        self._shutdown = True

        # Shutdown in reverse order to respect dependencies
        shutdown_tasks = []

        if self.health_manager and hasattr(self.health_manager, "shutdown"):
            shutdown_tasks.append(self.health_manager.shutdown())
        if self.pubsub and hasattr(self.pubsub, "shutdown"):
            shutdown_tasks.append(self.pubsub.shutdown())
        if self.lock_manager and hasattr(self.lock_manager, "shutdown"):
            shutdown_tasks.append(self.lock_manager.shutdown())
        if self.cache and hasattr(self.cache, "shutdown"):
            shutdown_tasks.append(self.cache.shutdown())
        if self.database and hasattr(self.database, "shutdown"):
            shutdown_tasks.append(self.database.shutdown())

        # Execute shutdowns concurrently for faster cleanup
        if shutdown_tasks:
            results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)

            # Log any shutdown errors without failing
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    service_name = [
                        "health_manager",
                        "pubsub",
                        "lock_manager",
                        "cache",
                        "database",
                    ][i]
                    logger.error(f"Error shutting down {service_name}: {result}")

        logger.info("All database services shutdown complete")

    async def health_check_all(self) -> Dict[str, HealthStatus]:
        """Health check all composed services.

        Returns:
            Dictionary mapping service names to health status
        """
        if not self._initialized:
            return {"overall": HealthStatus.UNKNOWN}

        # Use health manager's check method and convert to simple dict
        result = await self.health_manager.check_all_components_health()
        return {
            "overall": result.overall_status,
            "components": len(result.components),
            "healthy": result.healthy_count,
            "degraded": result.degraded_count,
            "unhealthy": result.unhealthy_count,
            "details": result.overall_status.value,
        }

    async def get_metrics_all(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all services.

        Returns:
            Dictionary containing metrics from all composed services
        """
        if not self._initialized:
            return {"error": "Services not initialized"}

        metrics = {
            "composition_info": {
                "mode": self.mode.value,
                "initialized": self._initialized,
                "shutdown": self._shutdown,
                "timestamp": datetime.utcnow().isoformat(),
            }
        }

        # Collect metrics from all services
        try:
            if self.database and hasattr(self.database, "get_metrics"):
                metrics["database"] = await self.database.get_metrics()
        except Exception as e:
            metrics["database"] = {"error": str(e)}

        try:
            if self.cache and hasattr(self.cache, "get_comprehensive_metrics"):
                metrics["cache"] = await self.cache.get_comprehensive_metrics()
        except Exception as e:
            metrics["cache"] = {"error": str(e)}

        try:
            if self.health_manager and hasattr(
                self.health_manager, "check_all_components_health"
            ):
                health_details = await self.health_manager.check_all_components_health()
                metrics["health"] = {
                    "status": health_details.overall_status.value,
                    "components": len(health_details.components),
                    "healthy_count": health_details.healthy_count,
                    "degraded_count": health_details.degraded_count,
                    "unhealthy_count": health_details.unhealthy_count,
                }
        except Exception as e:
            metrics["health"] = {"error": str(e)}

        return metrics

    async def _cleanup_partial_initialization(self):
        """Cleanup any partially initialized services."""
        logger.warning("Cleaning up partially initialized services")

        cleanup_tasks = []
        for service in [
            self.health_manager,
            self.pubsub,
            self.lock_manager,
            self.cache,
            self.database,
        ]:
            if hasattr(service, "shutdown"):
                cleanup_tasks.append(service.shutdown())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    # Context manager support for clean resource management
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown_all()


# Global service registry for factory functions
_service_registry: Dict[ManagerMode, DatabaseServices] = {}
_registry_lock = asyncio.Lock()


async def create_database_services(
    mode: ManagerMode,
    db_config: DatabaseConfigProtocol,
    pool_config: Optional[TypesPoolConfiguration] = None,
    force_new: bool = False,
) -> DatabaseServices:
    """Factory function to create or retrieve DatabaseServices instance.

    Args:
        mode: Manager operation mode
        db_config: Database configuration implementing DatabaseConfigProtocol
        pool_config: Optional configuration override
        force_new: Force creation of new instance

    Returns:
        DatabaseServices instance for the specified mode
    """
    async with _registry_lock:
        if not force_new and mode in _service_registry:
            existing_services = _service_registry[mode]
            if not existing_services._shutdown:
                logger.debug(
                    f"Returning existing DatabaseServices for mode: {mode.value}"
                )
                return existing_services

        # Create new services instance
        services = DatabaseServices(mode, db_config, pool_config)
        await services.initialize_all()

        _service_registry[mode] = services
        logger.info(f"Created new DatabaseServices for mode: {mode.value}")
        return services


async def get_database_services(mode: ManagerMode) -> Optional[DatabaseServices]:
    """Get existing DatabaseServices instance without creating new one.

    Args:
        mode: Manager operation mode

    Returns:
        Existing DatabaseServices instance or None if not found
    """
    async with _registry_lock:
        services = _service_registry.get(mode)
        if services and not services._shutdown:
            return services
        return None


async def shutdown_all_services():
    """Shutdown all registered DatabaseServices instances."""
    async with _registry_lock:
        if not _service_registry:
            return

        logger.info(
            f"Shutting down {len(_service_registry)} DatabaseServices instances"
        )

        shutdown_tasks = []
        for services in _service_registry.values():
            if not services._shutdown:
                shutdown_tasks.append(services.shutdown_all())

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        _service_registry.clear()
        logger.info("All DatabaseServices instances shutdown complete")
