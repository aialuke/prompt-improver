"""Database Services Protocol.

Protocol interface for the composed database services following 
the composition pattern.
"""

from typing import Protocol, Optional, Dict, Any
from prompt_improver.database.types import HealthStatus


class DatabaseServicesProtocol(Protocol):
    """Protocol for composed database services following composition pattern."""

    async def initialize_all(self) -> None:
        """Initialize all composed services in proper dependency order."""
        ...

    async def shutdown_all(self) -> None:
        """Shutdown all composed services in reverse dependency order."""
        ...

    async def health_check_all(self) -> Dict[str, HealthStatus]:
        """Health check all composed services."""
        ...

    async def get_metrics_all(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all services."""
        ...