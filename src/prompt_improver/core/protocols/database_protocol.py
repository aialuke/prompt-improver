"""Protocol definitions for database operations.

Provides type-safe interface contracts for database access,
enabling dependency inversion and improved testability.
"""
from typing import Any, AsyncContextManager, Dict, Optional, Protocol
from sqlalchemy.ext.asyncio import AsyncSession

class DatabaseSessionProtocol(Protocol):
    """Protocol for database session management"""

    async def get_session(self) -> AsyncContextManager[AsyncSession]:
        """Get async database session context manager"""
        ...

    async def get_session_manager(self) -> Any:
        """Get the session manager instance"""
        ...

    async def health_check(self) -> bool:
        """Check database health"""
        ...

class DatabaseConfigProtocol(Protocol):
    """Protocol for database configuration"""

    def get_database_url(self) -> str:
        """Get database connection URL"""
        ...

    def get_connection_pool_config(self) -> dict[str, Any]:
        """Get connection pool configuration"""
        ...

    def get_retry_config(self) -> dict[str, Any]:
        """Get retry configuration"""
        ...

class QueryOptimizerProtocol(Protocol):
    """Protocol for query optimization services"""

    async def optimize_query(self, query: str, params: dict | None=None) -> str:
        """Optimize SQL query"""
        ...

    async def analyze_performance(self, query: str) -> dict[str, Any]:
        """Analyze query performance"""
        ...

    async def get_execution_plan(self, query: str) -> dict[str, Any]:
        """Get query execution plan"""
        ...

class DatabaseHealthProtocol(Protocol):
    """Protocol for database health monitoring"""

    async def check_connection_pool(self) -> dict[str, Any]:
        """Check connection pool health"""
        ...

    async def check_query_performance(self) -> dict[str, Any]:
        """Check query performance metrics"""
        ...

    async def check_table_health(self) -> dict[str, Any]:
        """Check table health metrics"""
        ...

class DatabaseProtocol(DatabaseSessionProtocol, DatabaseConfigProtocol, QueryOptimizerProtocol, DatabaseHealthProtocol):
    """Combined protocol for all database operations"""
