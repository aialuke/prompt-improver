"""Connection Manager Protocol - Unified Interface for Connection Management

Provides a simplified, unified protocol for connection management across
different backends (PostgreSQL, Redis, etc.) following 2025 best practices.
"""

from typing import Protocol, AsyncContextManager, Any, Optional, Dict
from enum import Enum


class ConnectionMode(Enum):
    """Connection operation modes"""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    BATCH = "batch"
    TRANSACTIONAL = "transactional"


class ConnectionManagerProtocol(Protocol):
    """
    Unified protocol for connection management.
    
    This protocol provides a common interface for different connection
    managers (database, cache, etc.) enabling dependency inversion
    and eliminating circular imports.
    """

    async def get_connection(
        self, 
        mode: ConnectionMode = ConnectionMode.READ_WRITE,
        **kwargs
    ) -> AsyncContextManager[Any]:
        """
        Get a connection with specified mode.
        
        Args:
            mode: Connection operation mode
            **kwargs: Additional connection parameters
            
        Returns:
            Async context manager for the connection
        """
        ...

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform connection health check.
        
        Returns:
            Dictionary containing health status and metrics
        """
        ...

    async def close(self) -> None:
        """
        Close all connections and cleanup resources.
        """
        ...

    async def get_connection_info(self) -> Dict[str, Any]:
        """
        Get current connection pool information.
        
        Returns:
            Dictionary with pool status, active connections, etc.
        """
        ...

    def is_healthy(self) -> bool:
        """
        Quick health status check.
        
        Returns:
            True if connection manager is healthy
        """
        ...