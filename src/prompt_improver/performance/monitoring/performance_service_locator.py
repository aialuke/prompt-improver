"""Performance Service Locator - 2025 Architecture

Service locator pattern for performance monitoring dependencies to eliminate
circular imports while maintaining clean dependency injection. Follows the
established factory pattern architecture.
"""

import asyncio
import logging
from typing import Any, AsyncContextManager, Dict, Optional, TypeVar, cast

from sqlalchemy.ext.asyncio import AsyncSession

from .protocols import (
    ConfigurationServiceProtocol,
    DatabaseServiceProtocol,
    MLEventBusServiceProtocol,
    PromptImprovementServiceProtocol,
    SessionStoreServiceProtocol,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


class PerformanceServiceLocator:
    """Service locator for performance monitoring dependencies.
    
    Provides dependency injection for performance monitoring components without
    requiring direct imports of DI containers or MCP server components.
    
    Features:
    - Lazy loading of services
    - Caching for performance
    - Mock injection for testing
    - Graceful fallbacks for missing dependencies
    - Protocol-based interfaces for clean architecture
    """
    
    def __init__(self):
        """Initialize service locator with empty cache."""
        self._services: Dict[type, Any] = {}
        self._factories: Dict[type, Any] = {}
        self._initialized = False
        self.logger = logging.getLogger(__name__)
        
    def register_service(self, interface: type[T], instance: T) -> None:
        """Register a service instance.
        
        Args:
            interface: Service interface/protocol
            instance: Service instance
        """
        self._services[interface] = instance
        self.logger.debug(f"Registered service: {interface.__name__}")
        
    def register_factory(self, interface: type[T], factory: Any) -> None:
        """Register a service factory function.
        
        Args:
            interface: Service interface/protocol
            factory: Factory function to create service
        """
        self._factories[interface] = factory
        self.logger.debug(f"Registered factory: {interface.__name__}")
        
    async def get_service(self, interface: type[T]) -> T:
        """Get service instance by interface.
        
        Args:
            interface: Service interface to resolve
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service is not registered and no factory available
        """
        # Return cached instance if available
        if interface in self._services:
            return cast(T, self._services[interface])
            
        # Create from factory if available
        if interface in self._factories:
            factory = self._factories[interface]
            if asyncio.iscoroutinefunction(factory):
                instance = await factory()
            else:
                instance = factory()
            self._services[interface] = instance
            self.logger.debug(f"Created service from factory: {interface.__name__}")
            return cast(T, instance)
            
        raise ValueError(f"No service or factory registered for {interface.__name__}")
        
    def has_service(self, interface: type) -> bool:
        """Check if service is available.
        
        Args:
            interface: Service interface to check
            
        Returns:
            True if service or factory is registered
        """
        return interface in self._services or interface in self._factories
        
    def clear_cache(self) -> None:
        """Clear service cache (useful for testing)."""
        self._services.clear()
        self.logger.debug("Service cache cleared")


class DatabaseServiceAdapter:
    """Adapter for database service access."""
    
    def __init__(self, get_session_func: Any):
        """Initialize with session factory function.
        
        Args:
            get_session_func: Function that returns database session
        """
        self._get_session_func = get_session_func
        
    async def get_session(self) -> AsyncContextManager[AsyncSession]:
        """Get database session.
        
        Returns:
            Async context manager for database session
        """
        return self._get_session_func()


class PromptImprovementServiceAdapter:
    """Adapter for prompt improvement service access."""
    
    def __init__(self, prompt_service: Any):
        """Initialize with prompt service.
        
        Args:
            prompt_service: Prompt improvement service instance
        """
        self._prompt_service = prompt_service
        
    async def improve_prompt(
        self,
        prompt: str,
        context: dict[str, Any],
        session_id: str,
        rate_limit_remaining: Optional[int] = None
    ) -> Any:
        """Improve prompt using the service.
        
        Args:
            prompt: Input prompt to improve
            context: Context dictionary for improvement
            session_id: Session identifier for tracking
            rate_limit_remaining: Optional rate limit info
            
        Returns:
            Prompt improvement result
        """
        # Use the prompt service facade directly to avoid MCP server dependency
        return await self._prompt_service.improve_prompt(
            prompt=prompt,
            context=context,
            session_id=session_id
        )


class ConfigurationServiceAdapter:
    """Adapter for configuration service access."""
    
    def __init__(self, config_dict: Optional[dict[str, Any]] = None):
        """Initialize with configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict or {}
        
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)
        
    def get_performance_config(self) -> dict[str, Any]:
        """Get performance-specific configuration.
        
        Returns:
            Performance configuration dictionary
        """
        return self._config.get("performance", {})


class MLEventBusServiceAdapter:
    """Adapter for ML event bus service access."""
    
    def __init__(self, event_bus_getter: Any):
        """Initialize with event bus getter function.
        
        Args:
            event_bus_getter: Function that returns event bus instance
        """
        self._event_bus_getter = event_bus_getter
        
    async def publish(self, event: Any) -> bool:
        """Publish event to ML event bus.
        
        Args:
            event: Event to publish
            
        Returns:
            True if event was published successfully
        """
        event_bus = await self.get_event_bus()
        return await event_bus.publish(event)
        
    async def get_event_bus(self) -> Any:
        """Get the ML event bus instance.
        
        Returns:
            ML event bus instance
        """
        return await self._event_bus_getter()


class SessionStoreServiceAdapter:
    """Adapter for session store service access."""
    
    def __init__(self, session_store: Any):
        """Initialize with session store.
        
        Args:
            session_store: Session store instance
        """
        self._session_store = session_store
        
    async def set(self, session_id: str, data: dict[str, Any]) -> None:
        """Set session data.
        
        Args:
            session_id: Session identifier
            data: Data to store
        """
        await self._session_store.set_session(session_id, data, ttl=3600)
        
    async def get(self, session_id: str) -> dict[str, Any] | None:
        """Get session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found
        """
        return await self._session_store.get_session(session_id)
        
    async def touch(self, session_id: str) -> None:
        """Touch session to update last access time.
        
        Args:
            session_id: Session identifier
        """
        await self._session_store.touch_session(session_id, ttl=3600)
        
    async def delete(self, session_id: str) -> None:
        """Delete session data.
        
        Args:
            session_id: Session identifier
        """
        await self._session_store.delete_session(session_id)


# No-op implementations for testing/fallback

class NoOpDatabaseService:
    """No-op database service for testing."""
    
    async def get_session(self) -> AsyncContextManager[AsyncSession]:
        """Return no-op session context manager."""
        from contextlib import asynccontextmanager
        
        @asynccontextmanager
        async def noop_session():
            yield None
            
        return noop_session()


class NoOpPromptImprovementService:
    """No-op prompt improvement service for testing."""
    
    async def improve_prompt(
        self,
        prompt: str,
        context: dict[str, Any],
        session_id: str,
        rate_limit_remaining: Optional[int] = None
    ) -> dict[str, Any]:
        """Return mock improvement result."""
        return {
            "improved_prompt": prompt,
            "improvements": [],
            "context": context,
            "session_id": session_id
        }


class NoOpSessionStoreService:
    """No-op session store service for testing."""
    
    def __init__(self):
        self._data: dict[str, dict[str, Any]] = {}
        
    async def set(self, session_id: str, data: dict[str, Any]) -> None:
        """Store data in memory."""
        self._data[session_id] = data
        
    async def get(self, session_id: str) -> dict[str, Any] | None:
        """Get data from memory."""
        return self._data.get(session_id)
        
    async def touch(self, session_id: str) -> None:
        """No-op touch operation."""
        pass
        
    async def delete(self, session_id: str) -> None:
        """Delete data from memory."""
        self._data.pop(session_id, None)


def create_performance_service_locator(
    database_session_factory: Optional[Any] = None,
    prompt_service: Optional[Any] = None,
    config: Optional[dict[str, Any]] = None,
    event_bus_getter: Optional[Any] = None,
    session_store: Optional[Any] = None
) -> PerformanceServiceLocator:
    """Create and configure performance service locator.
    
    Args:
        database_session_factory: Function to get database sessions
        prompt_service: Prompt improvement service instance
        config: Configuration dictionary
        event_bus_getter: Function to get event bus
        session_store: Session store instance
        
    Returns:
        Configured PerformanceServiceLocator instance
    """
    locator = PerformanceServiceLocator()
    
    # Register database service
    if database_session_factory:
        database_service = DatabaseServiceAdapter(database_session_factory)
        locator.register_service(DatabaseServiceProtocol, database_service)
    else:
        locator.register_service(DatabaseServiceProtocol, NoOpDatabaseService())
        
    # Register prompt improvement service
    if prompt_service:
        prompt_service_adapter = PromptImprovementServiceAdapter(prompt_service)
        locator.register_service(PromptImprovementServiceProtocol, prompt_service_adapter)
    else:
        locator.register_service(PromptImprovementServiceProtocol, NoOpPromptImprovementService())
        
    # Register configuration service
    config_service = ConfigurationServiceAdapter(config)
    locator.register_service(ConfigurationServiceProtocol, config_service)
    
    # Register ML event bus service
    if event_bus_getter:
        event_bus_service = MLEventBusServiceAdapter(event_bus_getter)
        locator.register_service(MLEventBusServiceProtocol, event_bus_service)
    else:
        # Create no-op event bus adapter
        async def noop_event_bus():
            class NoOpEventBus:
                async def publish(self, event): return True
            return NoOpEventBus()
            
        noop_adapter = MLEventBusServiceAdapter(noop_event_bus)
        locator.register_service(MLEventBusServiceProtocol, noop_adapter)
        
    # Register session store service
    if session_store:
        session_store_service = SessionStoreServiceAdapter(session_store)
        locator.register_service(SessionStoreServiceProtocol, session_store_service)
    else:
        locator.register_service(SessionStoreServiceProtocol, NoOpSessionStoreService())
        
    logger.debug("Performance service locator created and configured")
    return locator