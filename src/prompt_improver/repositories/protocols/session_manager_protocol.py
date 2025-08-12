"""Session manager protocol for database session management.

Defines clean interfaces for database session operations without coupling to
specific database implementation details. Provides transaction management and
connection handling abstractions.
"""

from contextlib import AbstractAsyncContextManager
from typing import Any, AsyncGenerator, Protocol, runtime_checkable
from abc import ABC, abstractmethod


@runtime_checkable
class SessionProtocol(Protocol):
    """Protocol for database session operations."""
    
    async def execute(self, query: Any, parameters: dict[str, Any] | None = None) -> Any:
        """Execute a query with optional parameters."""
        ...
    
    async def fetch_one(self, query: Any, parameters: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Fetch a single row from query result."""
        ...
    
    async def fetch_all(self, query: Any, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Fetch all rows from query result."""
        ...
    
    async def commit(self) -> None:
        """Commit current transaction."""
        ...
    
    async def rollback(self) -> None:
        """Rollback current transaction."""
        ...
    
    async def close(self) -> None:
        """Close the session."""
        ...


@runtime_checkable
class SessionManagerProtocol(Protocol):
    """Protocol for database session management."""
    
    async def get_session(self) -> SessionProtocol:
        """Get a new database session."""
        ...
    
    def session_context(self) -> AbstractAsyncContextManager[SessionProtocol]:
        """Get async context manager for session with automatic cleanup."""
        ...
    
    def transaction_context(self) -> AbstractAsyncContextManager[SessionProtocol]:
        """Get async context manager for transactional session."""
        ...
    
    async def health_check(self) -> bool:
        """Check if database connection is healthy."""
        ...
    
    async def get_connection_info(self) -> dict[str, Any]:
        """Get connection information and statistics."""
        ...
    
    async def close_all_sessions(self) -> None:
        """Close all active sessions."""
        ...


@runtime_checkable  
class QueryExecutorProtocol(Protocol):
    """Protocol for executing database queries without session management."""
    
    async def execute_query(
        self, 
        query: Any, 
        parameters: dict[str, Any] | None = None
    ) -> Any:
        """Execute a query and return result."""
        ...
    
    async def fetch_scalar(
        self, 
        query: Any, 
        parameters: dict[str, Any] | None = None
    ) -> Any:
        """Execute query and return scalar value."""
        ...
    
    async def fetch_one_dict(
        self, 
        query: Any, 
        parameters: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Execute query and return single row as dict."""
        ...
    
    async def fetch_all_dict(
        self, 
        query: Any, 
        parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute query and return all rows as list of dicts."""
        ...
    
    async def execute_in_transaction(
        self,
        queries: list[tuple[Any, dict[str, Any] | None]]
    ) -> list[Any]:
        """Execute multiple queries in a transaction."""
        ...