"""Clean Dependency Injection Container for Core Services.

This container follows clean architecture principles by:
- Only depending on repository interfaces, not implementations
- No imports of infrastructure layers (database, cache, monitoring)
- Using dependency injection to wire up clean services
- Supporting both synchronous and asynchronous service resolution

Features:
- Clean architecture compliance
- Repository interface injection
- Service lifecycle management  
- Async service initialization
- Type-safe service resolution
- Easy testing with mock repositories
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Protocol, Type, TypeVar
from contextlib import asynccontextmanager
from collections.abc import Awaitable

from prompt_improver.repositories.interfaces.repository_interfaces import (
    IPersistenceRepository,
    IRulesRepository, 
    IMLRepository,
    IUserFeedbackRepository,
    IAnalyticsRepository,
    IHealthRepository,
    IMonitoringRepository,
    ISessionManager,
)
from prompt_improver.core.services.persistence_service_clean import CleanPersistenceService
from prompt_improver.core.services.rule_selection_service_clean import CleanRuleSelectionService

logger = logging.getLogger(__name__)
T = TypeVar("T")


class ServiceNotRegisteredError(Exception):
    """Raised when attempting to resolve unregistered service."""
    
    def __init__(self, service_type: Type[Any]):
        self.service_type = service_type
        super().__init__(f"Service {service_type.__name__} is not registered")


class CleanDIContainer:
    """Clean dependency injection container following clean architecture."""

    def __init__(self):
        """Initialize the clean DI container."""
        self._singletons: Dict[Type[Any], Any] = {}
        self._factories: Dict[Type[Any], Any] = {}
        self._repositories: Dict[Type[Any], Any] = {}
        self._initialized = False

    def register_repository(self, interface: Type[T], implementation: T) -> None:
        """Register a repository implementation for an interface."""
        self._repositories[interface] = implementation
        logger.debug(f"Registered repository {interface.__name__} -> {type(implementation).__name__}")

    def register_singleton(self, interface: Type[T], factory: Any) -> None:
        """Register a singleton service with its factory."""
        self._factories[interface] = factory
        logger.debug(f"Registered singleton {interface.__name__}")

    def get_repository(self, interface: Type[T]) -> T:
        """Get a repository instance by interface."""
        if interface not in self._repositories:
            raise ServiceNotRegisteredError(interface)
        return self._repositories[interface]

    def get_service(self, interface: Type[T]) -> T:
        """Get a service instance by interface (singleton pattern)."""
        # Check if already instantiated
        if interface in self._singletons:
            return self._singletons[interface]

        # Check if factory is registered
        if interface not in self._factories:
            raise ServiceNotRegisteredError(interface)

        # Create instance using factory
        factory = self._factories[interface]
        try:
            if callable(factory):
                instance = factory(self)
            else:
                instance = factory
            
            self._singletons[interface] = instance
            logger.debug(f"Created singleton instance of {interface.__name__}")
            return instance

        except Exception as e:
            logger.error(f"Failed to create instance of {interface.__name__}: {e}")
            raise

    async def initialize_services(self) -> None:
        """Initialize all registered services asynchronously."""
        if self._initialized:
            return

        logger.info("Initializing clean DI container services...")

        # Initialize repositories first (they have no dependencies)
        for interface, implementation in self._repositories.items():
            if hasattr(implementation, 'initialize') and callable(implementation.initialize):
                try:
                    if asyncio.iscoroutinefunction(implementation.initialize):
                        await implementation.initialize()
                    else:
                        implementation.initialize()
                    logger.debug(f"Initialized repository {interface.__name__}")
                except Exception as e:
                    logger.error(f"Failed to initialize repository {interface.__name__}: {e}")

        # Initialize services (they depend on repositories)
        for interface in self._factories:
            try:
                # Trigger lazy instantiation
                service = self.get_service(interface)
                
                # Call async initializer if present
                if hasattr(service, 'initialize') and callable(service.initialize):
                    if asyncio.iscoroutinefunction(service.initialize):
                        await service.initialize()
                    else:
                        service.initialize()
                    logger.debug(f"Initialized service {interface.__name__}")

            except Exception as e:
                logger.error(f"Failed to initialize service {interface.__name__}: {e}")

        self._initialized = True
        logger.info("Clean DI container initialization complete")

    async def shutdown_services(self) -> None:
        """Shutdown all services gracefully."""
        logger.info("Shutting down clean DI container services...")

        # Shutdown services first
        for interface, instance in self._singletons.items():
            if hasattr(instance, 'shutdown') and callable(instance.shutdown):
                try:
                    if asyncio.iscoroutinefunction(instance.shutdown):
                        await instance.shutdown()
                    else:
                        instance.shutdown()
                    logger.debug(f"Shutdown service {interface.__name__}")
                except Exception as e:
                    logger.error(f"Failed to shutdown service {interface.__name__}: {e}")

        # Shutdown repositories
        for interface, implementation in self._repositories.items():
            if hasattr(implementation, 'shutdown') and callable(implementation.shutdown):
                try:
                    if asyncio.iscoroutinefunction(implementation.shutdown):
                        await implementation.shutdown()
                    else:
                        implementation.shutdown()
                    logger.debug(f"Shutdown repository {interface.__name__}")
                except Exception as e:
                    logger.error(f"Failed to shutdown repository {interface.__name__}: {e}")

        self._initialized = False
        logger.info("Clean DI container shutdown complete")

    @asynccontextmanager
    async def lifespan(self):
        """Async context manager for container lifecycle."""
        try:
            await self.initialize_services()
            yield self
        finally:
            await self.shutdown_services()

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all registered services."""
        health_status = {
            "container_initialized": self._initialized,
            "registered_repositories": len(self._repositories),
            "registered_services": len(self._factories),
            "instantiated_singletons": len(self._singletons),
            "repository_health": {},
            "service_health": {},
        }

        # Check repository health
        for interface, implementation in self._repositories.items():
            try:
                if hasattr(implementation, 'health_check') and callable(implementation.health_check):
                    health_result = implementation.health_check()
                    if asyncio.iscoroutine(health_result):
                        # Skip async health checks in sync method
                        health_status["repository_health"][interface.__name__] = "async_check_skipped"
                    else:
                        health_status["repository_health"][interface.__name__] = health_result
                else:
                    health_status["repository_health"][interface.__name__] = "no_health_check"
            except Exception as e:
                health_status["repository_health"][interface.__name__] = f"error: {e}"

        # Check service health  
        for interface, instance in self._singletons.items():
            try:
                if hasattr(instance, 'health_check') and callable(instance.health_check):
                    health_result = instance.health_check()
                    if asyncio.iscoroutine(health_result):
                        health_status["service_health"][interface.__name__] = "async_check_skipped"
                    else:
                        health_status["service_health"][interface.__name__] = health_result
                else:
                    health_status["service_health"][interface.__name__] = "no_health_check"
            except Exception as e:
                health_status["service_health"][interface.__name__] = f"error: {e}"

        return health_status


# Global container instance
_clean_container: Optional[CleanDIContainer] = None


def get_clean_container() -> CleanDIContainer:
    """Get the global clean DI container instance."""
    global _clean_container
    if _clean_container is None:
        _clean_container = CleanDIContainer()
        _configure_clean_container(_clean_container)
    return _clean_container


def _configure_clean_container(container: CleanDIContainer) -> None:
    """Configure the clean DI container with default services."""
    
    # Register core clean services
    container.register_singleton(
        CleanPersistenceService,
        lambda c: CleanPersistenceService(
            persistence_repository=c.get_repository(IPersistenceRepository)
        )
    )
    
    container.register_singleton(
        CleanRuleSelectionService,
        lambda c: CleanRuleSelectionService(
            rules_repository=c.get_repository(IRulesRepository)
        )
    )

    logger.info("Clean DI container configured with default services")


# Convenience functions
async def resolve_service(interface: Type[T]) -> T:
    """Resolve a service from the global container."""
    container = get_clean_container()
    return container.get_service(interface)


async def resolve_repository(interface: Type[T]) -> T:
    """Resolve a repository from the global container."""
    container = get_clean_container()
    return container.get_repository(interface)


def register_test_repositories(repositories: Dict[Type[Any], Any]) -> None:
    """Register test repositories for testing purposes."""
    container = get_clean_container()
    for interface, implementation in repositories.items():
        container.register_repository(interface, implementation)


async def initialize_clean_container() -> None:
    """Initialize the global clean container."""
    container = get_clean_container()
    await container.initialize_services()


async def shutdown_clean_container() -> None:
    """Shutdown the global clean container."""
    global _clean_container
    if _clean_container:
        await _clean_container.shutdown_services()
        _clean_container = None