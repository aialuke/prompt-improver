"""
Real database service for testing.

This module contains database service implementations for testing,
extracted from conftest.py to maintain clean architecture.
"""
from typing import Any, Dict, List, Optional
from sqlalchemy import text
from prompt_improver.shared.interfaces.protocols.ml import ServiceStatus


class RealDatabaseService:
    """Real PostgreSQL database service for testing."""
    
    def __init__(self, container):
        self.container = container
        self._query_count = 0
        
    async def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute query against real PostgreSQL database."""
        self._query_count += 1
        
        async with self.container.get_session() as session:
            result = await session.execute(text(query), parameters or {})
            
            # Handle different result types
            if result.returns_rows:
                rows = result.fetchall()
                if rows:
                    # Convert rows to dictionaries
                    columns = list(result.keys())
                    return [dict(zip(columns, row)) for row in rows]
                return []
            else:
                # For INSERT/UPDATE/DELETE, return affected rows
                return [{"affected_rows": result.rowcount}]

    async def execute_transaction(
        self, queries: List[str], parameters: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Execute transaction against real PostgreSQL database."""
        async with self.container.get_session() as session:
            try:
                for i, query in enumerate(queries):
                    query_params = parameters[i] if parameters and i < len(parameters) else {}
                    await session.execute(text(query), query_params)
                    self._query_count += 1
                
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def health_check(self) -> ServiceStatus:
        """Check database health using real connection."""
        try:
            async with self.container.get_session() as session:
                await session.execute(text("SELECT 1"))
            return ServiceStatus.HEALTHY
        except Exception:
            return ServiceStatus.ERROR

    async def get_connection_pool_stats(self) -> Dict[str, Any]:
        """Get real connection pool statistics."""
        # Get actual pool stats from the engine
        engine = self.container._engine
        pool = engine.pool if engine else None
        
        return {
            "active_connections": pool.checkedout() if pool else 0,
            "idle_connections": pool.checkedin() if pool else 0, 
            "max_connections": pool.size() if pool else 0,
            "queries_executed": self._query_count,
            "pool_overflow": pool.overflow() if pool else 0,
        }
        
    def get_connection_url(self) -> str:
        """Get database connection URL."""
        return self.container.get_connection_url()
        
    async def truncate_all_tables(self):
        """Clean all data for test isolation."""
        await self.container.truncate_all_tables()
        
    async def create_test_data(self, table_name: str, data: List[Dict[str, Any]]):
        """Create test data in specific table."""
        if not data:
            return
            
        async with self.container.get_session() as session:
            columns = list(data[0].keys())
            placeholders = ", ".join([f":{col}" for col in columns])
            sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
            
            for row in data:
                await session.execute(text(sql), row)
            
            await session.commit()