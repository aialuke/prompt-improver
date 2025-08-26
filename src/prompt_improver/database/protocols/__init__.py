"""Database Protocol Interfaces.

Provides protocol-based interfaces for database layer to interact with
application services without creating circular dependencies.

Clean Architecture compliance:
- Database (Infrastructure) defines protocols
- Application services implement protocols
- No direct imports from Infrastructure to Application layer
"""

from prompt_improver.database.protocols.database_services import (
    DatabaseServicesProtocol,
)
from prompt_improver.database.protocols.events import (
    DatabaseEventProtocol,
    EventData,
    EventType,
    OptimizationEventProtocol,
)

__all__ = [
    "DatabaseEventProtocol",
    "DatabaseServicesProtocol",
    "EventData",
    "EventType",
    "OptimizationEventProtocol",
]
