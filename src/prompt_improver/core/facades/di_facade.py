"""Dependency Injection Facade - Reduces Container Orchestrator Coupling

This facade provides a simplified interface to the container orchestrator
while reducing direct imports from 12 to 2 internal dependencies.

Design:
- Protocol-based interface for loose coupling
- Lazy initialization of container components
- Simplified service resolution methods
- Domain-specific access patterns
- Zero circular import dependencies
"""

import logging
from typing import Any, Protocol, TypeVar, runtime_checkable

logger = logging.getLogger(__name__)

T = TypeVar("T")


@runtime_checkable
class DIFacadeProtocol(Protocol):
    """Protocol for dependency injection facade."""
    
    async def get_service(self, service_type: type[T]) -> T:
        """Get service instance by type."""
        ...
    
    async def get_core_service(self, service_name: str) -> Any:
        """Get core service by name."""
        ...
    
    async def get_ml_service(self, service_name: str) -> Any:
        """Get ML service by name."""
        ...
    
    async def get_database_service(self, service_type: type[T]) -> T:
        """Get database service by type."""
        ...
    
    async def get_security_service(self, service_type: type[T]) -> T:
        """Get security service by type."""
        ...
    
    async def initialize(self) -> None:
        """Initialize all container services."""
        ...
    
    async def shutdown(self) -> None:
        """Shutdown all container services."""
        ...


class DIFacade(DIFacadeProtocol):
    """Dependency injection facade with minimal coupling.
    
    Reduces container orchestrator coupling from 12 internal imports to 2.
    Provides unified interface for service resolution across all domains.
    """

    def __init__(self):
        """Initialize facade with lazy loading."""
        self._container = None
        self._initialized = False
        self._core_container = None
        self._ml_container = None
        self._database_container = None
        self._security_container = None
        self._monitoring_container = None
        logger.debug("DIFacade initialized with lazy loading")

    async def _ensure_container(self):
        """Ensure container is initialized (lazy loading)."""
        if self._container is None:
            # Only import when needed to reduce coupling
            from prompt_improver.core.di.container_orchestrator import get_container
            self._container = await get_container()

    async def _ensure_domain_container(self, domain: str):
        """Ensure domain-specific container is available."""
        await self._ensure_container()
        
        if domain == "core" and self._core_container is None:
            self._core_container = self._container.get_core_container()
        elif domain == "ml" and self._ml_container is None:
            self._ml_container = self._container.get_ml_container()
        elif domain == "database" and self._database_container is None:
            self._database_container = self._container.get_database_container()
        elif domain == "security" and self._security_container is None:
            self._security_container = self._container.get_security_container()
        elif domain == "monitoring" and self._monitoring_container is None:
            self._monitoring_container = self._container.get_monitoring_container()

    async def get_service(self, service_type: type[T]) -> T:
        """Get service instance by type with intelligent routing."""
        await self._ensure_container()
        return await self._container.get(service_type)

    async def get_core_service(self, service_name: str) -> Any:
        """Get core service by name."""
        await self._ensure_domain_container("core")
        return await self._core_container.get_service(service_name)

    async def get_ml_service(self, service_name: str) -> Any:
        """Get ML service by name."""
        await self._ensure_domain_container("ml")
        return await self._ml_container.get_service(service_name)

    async def get_database_service(self, service_type: type[T]) -> T:
        """Get database service by type."""
        await self._ensure_domain_container("database")
        return await self._database_container.get(service_type)

    async def get_security_service(self, service_type: type[T]) -> T:
        """Get security service by type."""
        await self._ensure_domain_container("security")
        return await self._security_container.get(service_type)

    async def register_singleton(self, interface: type[T], implementation: type[T]) -> None:
        """Register singleton service."""
        await self._ensure_container()
        self._container.register_singleton(interface, implementation)

    async def register_transient(self, interface: type[T], implementation: type[T]) -> None:
        """Register transient service."""
        await self._ensure_container()
        self._container.register_transient(interface, implementation)

    async def initialize(self) -> None:
        """Initialize all container services."""
        if self._initialized:
            return
            
        await self._ensure_container()
        await self._container.initialize()
        self._initialized = True
        logger.info("DIFacade initialization complete")

    async def shutdown(self) -> None:
        """Shutdown all container services."""
        if not self._initialized:
            return
            
        if self._container:
            await self._container.shutdown()
        
        self._container = None
        self._core_container = None
        self._ml_container = None
        self._database_container = None
        self._security_container = None
        self._monitoring_container = None
        self._initialized = False
        logger.info("DIFacade shutdown complete")

    async def health_check(self) -> dict[str, Any]:
        """Perform health check across all containers."""
        await self._ensure_container()
        return await self._container.health_check_all()


# Global facade instance
_di_facade: DIFacade | None = None


def get_di_facade() -> DIFacade:
    """Get global DI facade instance.
    
    Returns:
        DIFacade with lazy initialization and minimal coupling
    """
    global _di_facade
    if _di_facade is None:
        _di_facade = DIFacade()
    return _di_facade


async def initialize_di_facade() -> None:
    """Initialize the global DI facade."""
    facade = get_di_facade()
    await facade.initialize()


async def shutdown_di_facade() -> None:
    """Shutdown the global DI facade."""
    global _di_facade
    if _di_facade:
        await _di_facade.shutdown()
        _di_facade = None


__all__ = [
    "DIFacadeProtocol",
    "DIFacade", 
    "get_di_facade",
    "initialize_di_facade",
    "shutdown_di_facade",
]