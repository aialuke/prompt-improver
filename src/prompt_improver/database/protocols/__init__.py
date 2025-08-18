"""Database Protocol Interfaces.

Provides protocol-based interfaces for database layer to interact with 
application services without creating circular dependencies.

Clean Architecture compliance:
- Database (Infrastructure) defines protocols
- Application services implement protocols  
- No direct imports from Infrastructure to Application layer
"""

from .events import (
    DatabaseEventProtocol,
    OptimizationEventProtocol,
    EventData,
    EventType,
)
from .database_services import DatabaseServicesProtocol
from .database_config import DatabaseConfigProtocol

__all__ = [
    "DatabaseEventProtocol",
    "OptimizationEventProtocol", 
    "EventData",
    "EventType",
    "DatabaseServicesProtocol",
    "DatabaseConfigProtocol",
]