"""Database Event Protocol Interfaces.

Defines protocol-based interfaces for database layer to dispatch events
without importing ML orchestration components directly.

This eliminates architectural violations where database (infrastructure)
layer was importing ML (application) layer components.
"""

from enum import Enum
from typing import Any, Protocol, runtime_checkable


class EventType(Enum):
    """Database event types that can be dispatched."""

    DATABASE_SLOW_QUERY_DETECTED = "database_slow_query_detected"
    DATABASE_CACHE_HIT_RATIO_LOW = "database_cache_hit_ratio_low"
    DATABASE_PERFORMANCE_DEGRADED = "database_performance_degraded"
    DATABASE_OPTIMIZATION_COMPLETED = "database_optimization_completed"
    DATABASE_CONNECTION_POOL_STRESSED = "database_connection_pool_stressed"


EventData = dict[str, Any]


@runtime_checkable
class DatabaseEventProtocol(Protocol):
    """Protocol for database event dispatching.

    Allows database layer to dispatch events without importing ML components.
    ML layer can optionally implement this protocol to receive events.
    """

    async def dispatch_event(
        self,
        event_type: EventType,
        event_data: EventData
    ) -> None:
        """Dispatch a database event.

        Args:
            event_type: Type of event being dispatched
            event_data: Event payload data
        """
        ...


@runtime_checkable
class OptimizationEventProtocol(Protocol):
    """Protocol for database optimization event handling.

    Enables database layer to trigger optimization workflows
    without direct ML orchestration imports.
    """

    async def handle_performance_degradation(
        self,
        metrics: EventData
    ) -> None:
        """Handle database performance degradation event.

        Args:
            metrics: Performance metrics triggering the event
        """
        ...

    async def handle_cache_optimization(
        self,
        cache_stats: EventData
    ) -> None:
        """Handle cache optimization request.

        Args:
            cache_stats: Cache performance statistics
        """
        ...

    async def handle_query_optimization(
        self,
        query_data: EventData
    ) -> None:
        """Handle slow query optimization request.

        Args:
            query_data: Query performance data requiring optimization
        """
        ...
