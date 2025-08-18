"""Database Configuration Protocol.

Simple protocol for database configuration to avoid importing 
from core.protocols which triggers the DI container chain.
"""

from typing import Dict, Any, Protocol


class DatabaseConfigProtocol(Protocol):
    """Protocol for database configuration providers."""
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        ...
        
    def get_connection_pool_config(self) -> Dict[str, Any]:
        """Get connection pool configuration."""
        ...