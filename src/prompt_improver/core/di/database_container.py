"""Database Services Dependency Injection Container (2025).

Specialized DI container for database services including connection management,
caching, repository patterns, and data access layer services.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional, Type, TypeVar

from prompt_improver.core.di.protocols import (
    ContainerRegistryProtocol,
    DatabaseContainerProtocol,
)

if TYPE_CHECKING:
    from prompt_improver.core.protocols.ml_protocols import (
        CacheServiceProtocol,
        DatabaseServiceProtocol,
        ServiceConnectionInfo,
        ServiceStatus,
    )

T = TypeVar("T")
logger = logging.getLogger(__name__)


def _get_ml_protocols():
    """Lazy import ML protocols to avoid torch dependencies."""
    try:
        from prompt_improver.core.protocols.ml_protocols import (
            CacheServiceProtocol,
            DatabaseServiceProtocol,
            ServiceConnectionInfo,
            ServiceStatus,
        )
        return CacheServiceProtocol, DatabaseServiceProtocol, ServiceConnectionInfo, ServiceStatus
    except ImportError:
        logger.info("ML protocols not available (torch not installed)")
        return None, None, None, None


class ServiceLifetime(Enum):
    """Service lifetime management options."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


@dataclass
class DatabaseServiceRegistration:
    """Database service registration information."""
    interface: Type[Any]
    implementation: Type[Any] | None
    lifetime: ServiceLifetime
    factory: Callable[[], Any] | None = None
    initialized: bool = False
    instance: Any = None
    tags: set[str] = field(default_factory=set)
    health_check: Callable[[], Any] | None = None


class DatabaseContainer(DatabaseContainerProtocol, ContainerRegistryProtocol):
    """Specialized DI container for database services.
    
    Manages database-related services including:
    - PostgreSQL connection management and pooling
    - Redis cache service integration
    - Repository pattern implementations
    - Session management and transactions
    - Connection health monitoring
    
    Follows clean architecture with protocol-based dependencies.
    """

    def __init__(self, name: str = "database"):
        """Initialize database services container.
        
        Args:
            name: Container identifier for logging
        """
        self.name = name
        self.logger = logger.getChild(f"container.{name}")
        self._services: dict[Type[Any], DatabaseServiceRegistration] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._initialization_order: list[Type[Any]] = []
        self._register_default_services()
        self.logger.debug(f"Database container '{self.name}' initialized")

    def _register_default_services(self) -> None:
        """Register default database services."""
        # Self-registration for dependency injection
        self.register_instance(DatabaseContainer, self, tags={"container", "database"})
        
        # Database service factory (PostgreSQL)
        self.register_database_service_factory()
        
        # Cache service factory (Redis)
        self.register_cache_service_factory()
        
        # Connection manager factory
        self.register_connection_manager_factory()
        
        # Session manager factory
        self.register_session_manager_factory()
        
        # Repository factory
        self.register_repository_factory()
        
        # Migration service factory
        self.register_migration_service_factory()
        
        # Transaction manager factory
        self.register_transaction_manager_factory()

    def register_singleton(
        self,
        interface: Type[T],
        implementation: Type[T],
        tags: set[str] | None = None,
    ) -> None:
        """Register a singleton service.
        
        Args:
            interface: Service interface/protocol
            implementation: Concrete implementation class
            tags: Optional tags for service categorization
        """
        self._services[interface] = DatabaseServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SINGLETON,
            tags=tags or set(),
        )
        self.logger.debug(
            f"Registered singleton: {interface.__name__} -> {implementation.__name__}"
        )

    def register_transient(
        self,
        interface: Type[T],
        implementation_or_factory: Any,
        tags: set[str] | None = None,
    ) -> None:
        """Register a transient service.
        
        Args:
            interface: Service interface/protocol
            implementation_or_factory: Implementation class or factory
            tags: Optional tags for service categorization
        """
        self._services[interface] = DatabaseServiceRegistration(
            interface=interface,
            implementation=implementation_or_factory if not callable(implementation_or_factory) else None,
            factory=implementation_or_factory if callable(implementation_or_factory) else None,
            lifetime=ServiceLifetime.TRANSIENT,
            tags=tags or set(),
        )
        self.logger.debug(f"Registered transient: {interface.__name__}")

    def register_factory(
        self,
        interface: Type[T],
        factory: Any,
        tags: set[str] | None = None,
    ) -> None:
        """Register a service factory.
        
        Args:
            interface: Service interface/protocol
            factory: Factory function to create service
            tags: Optional tags for service categorization
        """
        self._services[interface] = DatabaseServiceRegistration(
            interface=interface,
            implementation=None,
            lifetime=ServiceLifetime.SINGLETON,
            factory=factory,
            tags=tags or set(),
        )
        interface_name = interface.__name__ if hasattr(interface, "__name__") else str(interface)
        self.logger.debug(f"Registered factory: {interface_name}")

    def register_instance(
        self,
        interface: Type[T],
        instance: T,
        tags: set[str] | None = None,
    ) -> None:
        """Register a pre-created service instance.
        
        Args:
            interface: Service interface/protocol
            instance: Pre-created service instance
            tags: Optional tags for service categorization
        """
        registration = DatabaseServiceRegistration(
            interface=interface,
            implementation=type(instance),
            lifetime=ServiceLifetime.SINGLETON,
            initialized=True,
            instance=instance,
            tags=tags or set(),
        )
        self._services[interface] = registration
        self.logger.debug(f"Registered instance: {interface.__name__}")

    async def get(self, interface: Type[T]) -> T:
        """Resolve service instance.
        
        Args:
            interface: Service interface to resolve
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service is not registered
        """
        async with self._lock:
            return await self._resolve_service(interface)

    async def _resolve_service(self, interface: Type[T]) -> T:
        """Internal service resolution with lifecycle management.
        
        Args:
            interface: Service interface to resolve
            
        Returns:
            Service instance
        """
        if interface not in self._services:
            raise KeyError(f"Database service not registered: {interface.__name__}")
            
        registration = self._services[interface]
        
        # Return existing singleton instance
        if (registration.lifetime == ServiceLifetime.SINGLETON and 
            registration.initialized and registration.instance is not None):
            return registration.instance
            
        # Create new instance
        if registration.factory:
            instance = await self._create_from_factory(registration.factory)
        elif registration.implementation:
            instance = await self._create_from_class(registration.implementation)
        else:
            raise ValueError(f"No factory or implementation for {interface.__name__}")
            
        # Initialize if needed
        if hasattr(instance, "initialize") and asyncio.iscoroutinefunction(instance.initialize):
            await instance.initialize()
            
        # Store singleton
        if registration.lifetime == ServiceLifetime.SINGLETON:
            registration.instance = instance
            registration.initialized = True
            self._initialization_order.append(interface)
            
        self.logger.debug(f"Resolved database service: {interface.__name__}")
        return instance

    async def _create_from_factory(self, factory: Callable[[], Any]) -> Any:
        """Create service instance from factory.
        
        Args:
            factory: Factory function
            
        Returns:
            Service instance
        """
        if asyncio.iscoroutinefunction(factory):
            return await factory()
        return factory()

    async def _create_from_class(self, implementation: Type[Any]) -> Any:
        """Create service instance from class constructor.
        
        Args:
            implementation: Implementation class
            
        Returns:
            Service instance
        """
        import inspect
        
        sig = inspect.signature(implementation.__init__)
        kwargs = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param.annotation != inspect.Parameter.empty:
                try:
                    dependency = await self._resolve_service(param.annotation)
                    kwargs[param_name] = dependency
                except KeyError:
                    if param.default != inspect.Parameter.empty:
                        continue
                    raise
                    
        return implementation(**kwargs)

    def is_registered(self, interface: Type[T]) -> bool:
        """Check if service is registered.
        
        Args:
            interface: Service interface to check
            
        Returns:
            True if registered, False otherwise
        """
        return interface in self._services

    # Database service factory methods
    def register_database_service_factory(self, config: Any = None) -> None:
        """Register factory for PostgreSQL database service with connection pooling.
        
        Args:
            config: PostgreSQL service connection configuration
        """
        def _get_config():
            """Lazy load config objects to avoid ML protocol imports."""
            protocols = _get_ml_protocols()
            if protocols[0] is None:  # ML protocols not available
                return {
                    "service_name": "postgresql",
                    "connection_status": "healthy",
                    "connection_details": {
                        "host": os.getenv("POSTGRES_HOST", "postgres"),
                        "port": int(os.getenv("POSTGRES_PORT", "5432"))
                    }
                }
            
            CacheServiceProtocol, DatabaseServiceProtocol, ServiceConnectionInfo, ServiceStatus = protocols
            return ServiceConnectionInfo(
                service_name="postgresql",
                connection_status=ServiceStatus.HEALTHY,
                connection_details={
                    "host": os.getenv("POSTGRES_HOST", "postgres"),
                    "port": int(os.getenv("POSTGRES_PORT", "5432"))
                }
            )
        
        if config is None:
            config = _get_config()

        async def create_database_service():
            try:
                from prompt_improver.integrations.postgresql_service import PostgreSQLService
                service = PostgreSQLService(config)
                await service.initialize()
                return service
            except ImportError:
                # Fallback to mock database service for testing
                from prompt_improver.database.services.mock_database_service import MockDatabaseService
                return MockDatabaseService()

        async def cleanup_database_service(service):
            if hasattr(service, "shutdown"):
                await service.shutdown()

        # Use lazy loading for DatabaseServiceProtocol to avoid import issues
        def _get_database_protocol():
            protocols = _get_ml_protocols()
            if protocols[1] is not None:  # DatabaseServiceProtocol available
                return protocols[1]
            return "database_service"  # Use string key as fallback

        self.register_factory(
            _get_database_protocol(),
            create_database_service,
            tags={"postgresql", "database", "external"}
        )
        self.logger.debug("Registered database service factory")

    def register_cache_service_factory(self, config: Any = None) -> None:
        """Register factory for Redis cache service with connection pooling.
        
        Args:
            config: Redis service connection configuration
        """
        def _get_cache_config():
            """Lazy load config objects to avoid ML protocol imports."""
            protocols = _get_ml_protocols()
            if protocols[0] is None:  # ML protocols not available
                return {
                    "service_name": "redis",
                    "connection_status": "healthy",
                    "connection_details": {
                        "host": os.getenv("REDIS_HOST", "redis"),
                        "port": int(os.getenv("REDIS_PORT", "6379"))
                    }
                }
            
            CacheServiceProtocol, DatabaseServiceProtocol, ServiceConnectionInfo, ServiceStatus = protocols
            return ServiceConnectionInfo(
                service_name="redis",
                connection_status=ServiceStatus.HEALTHY,
                connection_details={
                    "host": os.getenv("REDIS_HOST", "redis"),
                    "port": int(os.getenv("REDIS_PORT", "6379"))
                }
            )
        
        if config is None:
            config = _get_cache_config()

        async def create_cache_service():
            try:
                from prompt_improver.integrations.redis_service import RedisService
                service = RedisService(config)
                await service.initialize()
                return service
            except ImportError:
                # Fallback to memory cache service
                from prompt_improver.services.cache.l1_cache_service import L1CacheService as MemoryCacheService
                return MemoryCacheService()

        async def cleanup_cache_service(service):
            if hasattr(service, "shutdown"):
                await service.shutdown()

        # Use lazy loading for CacheServiceProtocol to avoid import issues
        def _get_cache_protocol():
            protocols = _get_ml_protocols()
            if protocols[0] is not None:  # CacheServiceProtocol available
                return protocols[0]
            return "cache_service"  # Use string key as fallback

        self.register_factory(
            _get_cache_protocol(),
            create_cache_service,
            tags={"redis", "cache", "external"}
        )
        self.logger.debug("Registered cache service factory")

    def register_connection_manager_factory(self) -> None:
        """Register factory for database connection manager."""
        def create_connection_manager():
            try:
                from prompt_improver.database.services.connection.postgres_pool_manager_facade import (
                    PostgreSQLPoolManagerFacade,
                )
                return PostgreSQLPoolManagerFacade()
            except ImportError:
                # Fallback to basic connection manager
                from prompt_improver.database.services.connection.basic_connection_manager import BasicConnectionManager
                return BasicConnectionManager()

        self.register_factory(
            "connection_manager",
            create_connection_manager,
            tags={"connection", "database", "pool"}
        )
        self.logger.debug("Registered connection manager factory")

    def register_session_manager_factory(self) -> None:
        """Register factory for database session manager."""
        def create_session_manager():
            try:
                from prompt_improver.database.registry import SessionManagerProtocol
                from prompt_improver.database.services.session_manager import DatabaseSessionManager
                return DatabaseSessionManager()
            except ImportError:
                # Fallback to basic session manager
                from prompt_improver.database.services.basic_session_manager import BasicSessionManager
                return BasicSessionManager()

        self.register_factory(
            "session_manager",
            create_session_manager,
            tags={"session", "database", "transaction"}
        )
        self.logger.debug("Registered session manager factory")

    def register_repository_factory(self) -> None:
        """Register factory for repository pattern implementations."""
        def create_repository_factory():
            try:
                from prompt_improver.repositories.impl.repository_factory import RepositoryFactory
                return RepositoryFactory()
            except ImportError:
                # Fallback to basic repository factory
                from prompt_improver.database.services.basic_repository_factory import BasicRepositoryFactory
                return BasicRepositoryFactory()

        self.register_factory(
            "repository_factory",
            create_repository_factory,
            tags={"repository", "database", "pattern"}
        )
        self.logger.debug("Registered repository factory")

    def register_migration_service_factory(self) -> None:
        """Register factory for database migration service."""
        def create_migration_service():
            try:
                from prompt_improver.database.services.migration_service import MigrationService
                return MigrationService()
            except ImportError:
                # Fallback to no-op migration service
                from prompt_improver.database.services.noop_migration_service import NoOpMigrationService
                return NoOpMigrationService()

        self.register_factory(
            "migration_service",
            create_migration_service,
            tags={"migration", "database", "schema"}
        )
        self.logger.debug("Registered migration service factory")

    def register_transaction_manager_factory(self) -> None:
        """Register factory for transaction manager."""
        def create_transaction_manager():
            try:
                from prompt_improver.database.services.transaction_manager import TransactionManager
                return TransactionManager()
            except ImportError:
                # Fallback to basic transaction manager
                from prompt_improver.database.services.basic_transaction_manager import BasicTransactionManager
                return BasicTransactionManager()

        self.register_factory(
            "transaction_manager",
            create_transaction_manager,
            tags={"transaction", "database", "acid"}
        )
        self.logger.debug("Registered transaction manager factory")

    # DatabaseContainerProtocol implementation
    async def get_connection_manager(self) -> Any:
        """Get database connection manager instance."""
        return await self.get("connection_manager")

    async def get_cache_service(self):
        """Get cache service instance."""
        # Use lazy loading to avoid import issues
        protocols = _get_ml_protocols()
        if protocols[0] is not None:
            return await self.get(protocols[0])  # CacheServiceProtocol
        return await self.get("cache_service")

    async def get_session_manager(self) -> Any:
        """Get session manager instance."""
        return await self.get("session_manager")

    async def get_repository_factory(self) -> Any:
        """Get repository factory instance."""
        return await self.get("repository_factory")

    async def get_database_service(self):
        """Get database service instance."""
        # Use lazy loading to avoid import issues
        protocols = _get_ml_protocols()
        if protocols[1] is not None:
            return await self.get(protocols[1])  # DatabaseServiceProtocol
        return await self.get("database_service")

    async def get_migration_service(self) -> Any:
        """Get migration service instance."""
        return await self.get("migration_service")

    async def get_transaction_manager(self) -> Any:
        """Get transaction manager instance."""
        return await self.get("transaction_manager")

    # Container lifecycle management
    async def initialize(self) -> None:
        """Initialize all database services."""
        if self._initialized:
            return
            
        self.logger.info(f"Initializing database container '{self.name}'")
        
        # Initialize all registered services
        for interface in list(self._services.keys()):
            try:
                await self.get(interface)
            except Exception as e:
                self.logger.error(f"Failed to initialize {interface}: {e}")
                raise
                
        self._initialized = True
        self.logger.info(f"Database container '{self.name}' initialization complete")

    async def shutdown(self) -> None:
        """Shutdown all database services gracefully."""
        self.logger.info(f"Shutting down database container '{self.name}'")
        
        # Shutdown in reverse initialization order
        for interface in reversed(self._initialization_order):
            registration = self._services.get(interface)
            if registration and registration.instance:
                try:
                    if hasattr(registration.instance, "shutdown"):
                        if asyncio.iscoroutinefunction(registration.instance.shutdown):
                            await registration.instance.shutdown()
                        else:
                            registration.instance.shutdown()
                    self.logger.debug(f"Shutdown service: {interface.__name__}")
                except Exception as e:
                    self.logger.error(f"Error shutting down {interface.__name__}: {e}")
                    
        self._services.clear()
        self._initialization_order.clear()
        self._initialized = False
        self.logger.info(f"Database container '{self.name}' shutdown complete")

    async def health_check(self) -> dict[str, Any]:
        """Check health of all database services."""
        results = {
            "container_status": "healthy",
            "container_name": self.name,
            "initialized": self._initialized,
            "registered_services": len(self._services),
            "services": {},
        }
        
        for interface, registration in self._services.items():
            service_name = interface.__name__ if hasattr(interface, "__name__") else str(interface)
            try:
                if (registration.initialized and registration.instance and
                    hasattr(registration.instance, "health_check")):
                    
                    health_check = registration.instance.health_check
                    if asyncio.iscoroutinefunction(health_check):
                        service_health = await health_check()
                    else:
                        service_health = health_check()
                    results["services"][service_name] = service_health
                else:
                    results["services"][service_name] = {
                        "status": "healthy",
                        "initialized": registration.initialized,
                    }
            except Exception as e:
                results["services"][service_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                results["container_status"] = "degraded"
                
        return results

    def get_registration_info(self) -> dict[str, Any]:
        """Get information about registered database services."""
        info = {
            "container_name": self.name,
            "initialized": self._initialized,
            "services": {},
        }
        
        for interface, registration in self._services.items():
            service_name = interface.__name__ if hasattr(interface, "__name__") else str(interface)
            info["services"][service_name] = {
                "implementation": registration.implementation.__name__ if registration.implementation else "Factory",
                "lifetime": registration.lifetime.value,
                "initialized": registration.initialized,
                "has_instance": registration.instance is not None,
                "tags": list(registration.tags),
            }
            
        return info

    @asynccontextmanager
    async def managed_lifecycle(self):
        """Managed lifecycle context for database container."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.shutdown()


# Global database container instance
_database_container: Optional[DatabaseContainer] = None


def get_database_container() -> DatabaseContainer:
    """Get the global database container instance.
    
    Returns:
        DatabaseContainer: Global database container instance
    """
    global _database_container
    if _database_container is None:
        _database_container = DatabaseContainer()
    return _database_container


async def shutdown_database_container() -> None:
    """Shutdown the global database container."""
    global _database_container
    if _database_container:
        await _database_container.shutdown()
        _database_container = None