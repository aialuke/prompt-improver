"""Database Optional Services Registry.

Provides service registry pattern for optional ML services registration.
Enables graceful degradation when ML components are unavailable.

Architecture:
- Database layer defines registry interface
- ML layer optionally registers event handlers
- Database checks registry before dispatching events
- Clean separation with no backwards compatibility layers
"""

import logging
from contextlib import asynccontextmanager

from prompt_improver.database.protocols.events import (
    DatabaseEventProtocol,
    EventData,
    EventType,
    OptimizationEventProtocol,
)

logger = logging.getLogger(__name__)


class DatabaseOptionalServices:
    """Registry for optional ML and optimization services.

    Enables database layer to interact with ML services when available
    without creating circular dependencies or architectural violations.
    """

    def __init__(self) -> None:
        self._event_dispatcher: DatabaseEventProtocol | None = None
        self._optimization_handler: OptimizationEventProtocol | None = None
        self._registered_services: set[str] = set()

    def register_event_dispatcher(
        self,
        dispatcher: DatabaseEventProtocol
    ) -> None:
        """Register ML event dispatcher service.

        Args:
            dispatcher: Service implementing DatabaseEventProtocol
        """
        self._event_dispatcher = dispatcher
        self._registered_services.add("event_dispatcher")
        logger.info("ML event dispatcher registered successfully")

    def register_optimization_handler(
        self,
        handler: OptimizationEventProtocol
    ) -> None:
        """Register optimization event handler service.

        Args:
            handler: Service implementing OptimizationEventProtocol
        """
        self._optimization_handler = handler
        self._registered_services.add("optimization_handler")
        logger.info("ML optimization handler registered successfully")

    def unregister_all(self) -> None:
        """Unregister all optional services."""
        self._event_dispatcher = None
        self._optimization_handler = None
        self._registered_services.clear()
        logger.info("All optional services unregistered")

    async def dispatch_event_if_available(
        self,
        event_type: EventType,
        event_data: EventData
    ) -> bool:
        """Dispatch event if dispatcher is available.

        Args:
            event_type: Type of event to dispatch
            event_data: Event payload data

        Returns:
            True if event was dispatched, False if no dispatcher available
        """
        if self._event_dispatcher is None:
            logger.debug(f"No event dispatcher available for {event_type.value}")
            return False

        try:
            await self._event_dispatcher.dispatch_event(event_type, event_data)
            logger.debug(f"Event {event_type.value} dispatched successfully")
            return True
        except Exception as e:
            logger.exception(f"Error dispatching event {event_type.value}: {e}")
            return False

    async def handle_performance_degradation_if_available(
        self,
        metrics: EventData
    ) -> bool:
        """Handle performance degradation if handler available.

        Args:
            metrics: Performance metrics

        Returns:
            True if handled, False if no handler available
        """
        if self._optimization_handler is None:
            logger.debug("No optimization handler available for performance degradation")
            return False

        try:
            await self._optimization_handler.handle_performance_degradation(metrics)
            logger.debug("Performance degradation handled successfully")
            return True
        except Exception as e:
            logger.exception(f"Error handling performance degradation: {e}")
            return False

    async def handle_cache_optimization_if_available(
        self,
        cache_stats: EventData
    ) -> bool:
        """Handle cache optimization if handler available.

        Args:
            cache_stats: Cache performance statistics

        Returns:
            True if handled, False if no handler available
        """
        if self._optimization_handler is None:
            logger.debug("No optimization handler available for cache optimization")
            return False

        try:
            await self._optimization_handler.handle_cache_optimization(cache_stats)
            logger.debug("Cache optimization handled successfully")
            return True
        except Exception as e:
            logger.exception(f"Error handling cache optimization: {e}")
            return False

    async def handle_query_optimization_if_available(
        self,
        query_data: EventData
    ) -> bool:
        """Handle query optimization if handler available.

        Args:
            query_data: Query performance data

        Returns:
            True if handled, False if no handler available
        """
        if self._optimization_handler is None:
            logger.debug("No optimization handler available for query optimization")
            return False

        try:
            await self._optimization_handler.handle_query_optimization(query_data)
            logger.debug("Query optimization handled successfully")
            return True
        except Exception as e:
            logger.exception(f"Error handling query optimization: {e}")
            return False

    @property
    def has_event_dispatcher(self) -> bool:
        """Check if event dispatcher is available."""
        return self._event_dispatcher is not None

    @property
    def has_optimization_handler(self) -> bool:
        """Check if optimization handler is available."""
        return self._optimization_handler is not None

    @property
    def registered_services(self) -> set[str]:
        """Get set of registered service names."""
        return self._registered_services.copy()

    def get_status(self) -> dict[str, bool]:
        """Get status of all optional services.

        Returns:
            Dictionary with service availability status
        """
        return {
            "event_dispatcher_available": self.has_event_dispatcher,
            "optimization_handler_available": self.has_optimization_handler,
            "total_services_registered": len(self._registered_services),
        }


# Global registry instance
_global_registry: DatabaseOptionalServices | None = None


def get_optional_services_registry() -> DatabaseOptionalServices:
    """Get global optional services registry instance.

    Returns:
        Global DatabaseOptionalServices registry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = DatabaseOptionalServices()
    return _global_registry


@asynccontextmanager
async def optional_ml_services_context():
    """Context manager for optional ML services lifecycle.

    Ensures proper cleanup of registered services.
    """
    registry = get_optional_services_registry()
    try:
        yield registry
    finally:
        registry.unregister_all()
