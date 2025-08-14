"""Repository Access Facade - Reduces Repository Factory Coupling

This facade provides unified repository access while reducing direct imports
from 11 to 2 internal dependencies through lazy initialization.

Design:
- Protocol-based interface for loose coupling
- Lazy loading of repository implementations  
- Domain-specific repository access patterns
- Health check coordination
- Zero circular import dependencies
"""

import logging
from typing import Any, Protocol, TypeVar, runtime_checkable

logger = logging.getLogger(__name__)

T = TypeVar("T")


@runtime_checkable
class RepositoryFacadeProtocol(Protocol):
    """Protocol for repository access facade."""
    
    async def get_analytics_repository(self) -> Any:
        """Get analytics repository instance."""
        ...
    
    async def get_apriori_repository(self) -> Any:
        """Get apriori repository instance."""
        ...
    
    async def get_ml_repository(self) -> Any:
        """Get ML repository instance."""
        ...
    
    async def get_user_feedback_repository(self) -> Any:
        """Get user feedback repository instance."""
        ...
    
    async def get_health_repository(self) -> Any:
        """Get health repository instance."""
        ...
    
    async def get_repository(self, protocol_type: type[T]) -> T:
        """Get repository by protocol type."""
        ...
    
    async def health_check(self) -> dict[str, bool]:
        """Perform health check on all repositories."""
        ...
    
    def cleanup(self) -> None:
        """Cleanup repository resources."""
        ...


class RepositoryFacade(RepositoryFacadeProtocol):
    """Repository access facade with minimal coupling.
    
    Reduces repository factory coupling from 11 internal imports to 2.
    Provides unified interface for all repository access patterns.
    """

    def __init__(self):
        """Initialize facade with lazy loading."""
        self._factory = None
        self._connection_manager = None
        self._initialized = False
        logger.debug("RepositoryFacade initialized with lazy loading")

    async def _ensure_connection_manager(self):
        """Ensure database connection manager is available."""
        if self._connection_manager is None:
            # Only import when needed to reduce coupling
            from prompt_improver.database import get_database_services
            self._connection_manager = await get_database_services()

    async def _ensure_factory(self):
        """Ensure repository factory is initialized."""
        if self._factory is None:
            await self._ensure_connection_manager()
            from prompt_improver.repositories.factory import get_repository_factory
            self._factory = get_repository_factory(self._connection_manager)
            
        if not self._initialized:
            self._factory.initialize()
            self._initialized = True

    async def get_analytics_repository(self) -> Any:
        """Get analytics repository instance."""
        await self._ensure_factory()
        return self._factory.get_analytics_repository()

    async def get_apriori_repository(self) -> Any:
        """Get apriori repository instance."""
        await self._ensure_factory()
        return self._factory.get_apriori_repository()

    async def get_ml_repository(self) -> Any:
        """Get ML repository instance.""" 
        await self._ensure_factory()
        return self._factory.get_ml_repository()

    async def get_user_feedback_repository(self) -> Any:
        """Get user feedback repository instance."""
        await self._ensure_factory()
        return self._factory.get_user_feedback_repository()

    async def get_health_repository(self) -> Any:
        """Get health repository instance."""
        await self._ensure_factory()
        return self._factory.get_health_repository()

    async def get_repository(self, protocol_type: type[T]) -> T:
        """Get repository by protocol type."""
        await self._ensure_factory()
        return self._factory.get_repository(protocol_type)

    async def health_check(self) -> dict[str, bool]:
        """Perform health check on all repositories."""
        await self._ensure_factory()
        return await self._factory.health_check()

    def cleanup(self) -> None:
        """Cleanup repository resources."""
        if self._factory:
            self._factory.cleanup()
        self._factory = None
        self._connection_manager = None
        self._initialized = False
        logger.info("RepositoryFacade cleanup complete")

    async def setup_repository_dependencies(self) -> None:
        """Setup repository dependencies for the application."""
        await self._ensure_connection_manager()
        from prompt_improver.repositories.factory import setup_repository_dependencies
        await setup_repository_dependencies(self._connection_manager)

    async def get_repository_by_name(self, name: str) -> Any:
        """Get repository by name for dynamic access."""
        name_mapping = {
            "analytics": self.get_analytics_repository,
            "apriori": self.get_apriori_repository,
            "ml": self.get_ml_repository,
            "user_feedback": self.get_user_feedback_repository,
            "health": self.get_health_repository,
        }
        
        getter = name_mapping.get(name.lower())
        if getter:
            return await getter()
        
        raise ValueError(f"Unknown repository name: {name}")

    def get_available_repositories(self) -> list[str]:
        """Get list of available repository names."""
        return ["analytics", "apriori", "ml", "user_feedback", "health"]

    async def initialize_all(self) -> None:
        """Initialize all repositories."""
        await self._ensure_factory()
        
        # Pre-load all repositories for better performance
        repositories = [
            self.get_analytics_repository(),
            self.get_apriori_repository(), 
            self.get_ml_repository(),
            self.get_user_feedback_repository(),
            self.get_health_repository(),
        ]
        
        # Wait for all to initialize
        import asyncio
        await asyncio.gather(*repositories, return_exceptions=True)
        logger.info("All repositories initialized")


# Global facade instance
_repository_facade: RepositoryFacade | None = None


def get_repository_facade() -> RepositoryFacade:
    """Get global repository facade instance.
    
    Returns:
        RepositoryFacade with lazy initialization and minimal coupling
    """
    global _repository_facade
    if _repository_facade is None:
        _repository_facade = RepositoryFacade()
    return _repository_facade


async def initialize_repository_facade() -> None:
    """Initialize the global repository facade."""
    facade = get_repository_facade()
    await facade.initialize_all()


def reset_repository_facade() -> None:
    """Reset the global repository facade (primarily for testing)."""
    global _repository_facade
    if _repository_facade:
        _repository_facade.cleanup()
    _repository_facade = None


__all__ = [
    "RepositoryFacadeProtocol", 
    "RepositoryFacade",
    "get_repository_facade",
    "initialize_repository_facade",
    "reset_repository_facade",
]