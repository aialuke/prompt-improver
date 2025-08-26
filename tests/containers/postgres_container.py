"""PostgreSQL testcontainer infrastructure for real database testing.

This module provides real PostgreSQL container integration for comprehensive database testing,
replacing mock database operations with actual PostgreSQL instances to ensure real behavior testing.

Features:
- Real PostgreSQL instances using testcontainers
- Automatic schema migration and initialization
- Connection pooling and management
- Test isolation and cleanup
- Performance testing capabilities
- Multi-version PostgreSQL support
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlmodel import SQLModel
from testcontainers.postgres import PostgresContainer

from prompt_improver.database import (
    DatabaseServices,
)

logger = logging.getLogger(__name__)


class PostgreSQLTestContainer:
    """Enhanced PostgreSQL testcontainer for real database testing.

    Provides comprehensive PostgreSQL testing infrastructure with:
    - Real PostgreSQL instances via testcontainers
    - Automatic schema creation and migration
    - Connection pooling and session management
    - Test isolation and parallel execution support
    - Performance and constraint testing capabilities
    """

    def __init__(
        self,
        postgres_version: str = "16",
        database_name: str | None = None,
        username: str = "test_user",
        password: str = "test_pass",
        port: int | None = None,
    ):
        """Initialize PostgreSQL testcontainer.

        Args:
            postgres_version: PostgreSQL version to use (default: 16)
            database_name: Database name (auto-generated if None)
            username: Database username
            password: Database password
            port: Container port (auto-assigned if None)
        """
        self.postgres_version = postgres_version
        self.database_name = database_name or f"test_db_{uuid.uuid4().hex[:8]}"
        self.username = username
        self.password = password
        self.port = port

        self._container: PostgresContainer | None = None
        self._engine: AsyncEngine | None = None
        self._connection_manager: DatabaseServices | None = None
        self._connection_url: str | None = None

    async def start(self) -> "PostgreSQLTestContainer":
        """Start PostgreSQL container and initialize database."""
        try:
            # Create and start PostgreSQL container
            self._container = PostgresContainer(
                image=f"postgres:{self.postgres_version}",
                username=self.username,
                password=self.password,
                dbname=self.database_name,
            )

            if self.port:
                self._container = self._container.with_bind_ports(5432, self.port)

            self._container.start()

            # Get connection details
            host = self._container.get_container_host_ip()
            port = self._container.get_exposed_port(5432)

            self._connection_url = (
                f"postgresql+asyncpg://{self.username}:{self.password}@"
                f"{host}:{port}/{self.database_name}"
            )

            # Create async engine with optimized settings for testing
            self._engine = create_async_engine(
                self._connection_url,
                echo=False,  # Set to True for SQL debugging
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={
                    "server_settings": {
                        "application_name": "prompt_improver_test",
                        "jit": "off",  # Disable JIT for faster test startup
                    }
                }
            )

            # Initialize DatabaseServices for compatibility (not needed for testing)
            self._connection_manager = None

            # Wait for container to be ready and create schema
            await self._wait_for_readiness()
            await self._initialize_schema()

            logger.info(
                f"PostgreSQL testcontainer started: {self.database_name} "
                f"(version {self.postgres_version}, port {port})"
            )

            return self

        except Exception as e:
            logger.exception(f"Failed to start PostgreSQL testcontainer: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop PostgreSQL container and clean up resources."""
        try:
            if self._engine:
                await self._engine.dispose()
                self._engine = None

            if self._connection_manager:
                # DatabaseServices doesn't have cleanup method - just clear reference
                self._connection_manager = None

            if self._container:
                self._container.stop()
                self._container = None

            logger.info(f"PostgreSQL testcontainer stopped: {self.database_name}")

        except Exception as e:
            logger.warning(f"Error stopping PostgreSQL testcontainer: {e}")

    async def _wait_for_readiness(self, max_retries: int = 30, retry_delay: float = 1.0):
        """Wait for PostgreSQL to be ready for connections."""
        # Convert asyncpg URL to basic PostgreSQL URL for asyncpg
        basic_url = self._connection_url.replace("postgresql+asyncpg://", "postgresql://")

        for attempt in range(max_retries):
            try:
                conn = await asyncpg.connect(basic_url)
                await conn.execute("SELECT 1")
                await conn.close()
                logger.debug(f"PostgreSQL ready after {attempt + 1} attempts")
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"PostgreSQL not ready after {max_retries} attempts: {e}")
                await asyncio.sleep(retry_delay)

    async def _initialize_schema(self):
        """Initialize database schema with all models."""
        try:
            async with self._engine.begin() as conn:
                # Create all tables defined in SQLModel
                await conn.run_sync(SQLModel.metadata.create_all)

            logger.debug(f"Schema initialized for database: {self.database_name}")

        except Exception as e:
            logger.exception(f"Failed to initialize schema: {e}")
            raise

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with proper cleanup."""
        if not self._engine:
            raise RuntimeError("Container not started. Call start() first.")

        async with AsyncSession(self._engine) as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    @asynccontextmanager
    async def get_connection_manager(self) -> AsyncGenerator[DatabaseServices, None]:
        """Get DatabaseServices for compatibility with existing code."""
        if not self._connection_manager:
            raise RuntimeError("Container not started. Call start() first.")

        yield self._connection_manager

    async def execute_sql(self, sql: str, parameters: dict[str, Any] | None = None) -> Any:
        """Execute raw SQL for testing purposes."""
        async with self.get_session() as session:
            result = await session.execute(text(sql), parameters or {})
            await session.commit()
            return result

    async def get_table_count(self, table_name: str) -> int:
        """Get row count for a specific table."""
        result = await self.execute_sql(f"SELECT COUNT(*) FROM {table_name}")
        return result.scalar()

    async def truncate_all_tables(self):
        """Truncate all tables for test cleanup."""
        async with self.get_session() as session:
            # Get all table names
            result = await session.execute(
                text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                """)
            )
            tables = [row[0] for row in result.fetchall()]

            if tables:
                # Disable foreign key checks temporarily
                await session.execute(text("SET session_replication_role = replica"))

                # Truncate all tables
                for table in tables:
                    await session.execute(text(f"TRUNCATE TABLE {table} CASCADE"))

                # Re-enable foreign key checks
                await session.execute(text("SET session_replication_role = DEFAULT"))

                await session.commit()

            logger.debug(f"Truncated {len(tables)} tables in {self.database_name}")

    def get_connection_url(self) -> str:
        """Get database connection URL."""
        if not self._connection_url:
            raise RuntimeError("Container not started. Call start() first.")
        return self._connection_url

    def get_connection_info(self) -> dict[str, Any]:
        """Get connection information for manual connections."""
        if not self._container:
            raise RuntimeError("Container not started. Call start() first.")

        return {
            "host": self._container.get_container_host_ip(),
            "port": self._container.get_exposed_port(5432),
            "database": self.database_name,
            "username": self.username,
            "password": self.password,
            "connection_url": self._connection_url,
        }

    async def __aenter__(self) -> "PostgreSQLTestContainer":
        """Async context manager entry."""
        return await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


class PostgreSQLTestFixture:
    """Test fixture helper for PostgreSQL testcontainers.

    Provides convenience methods for common test patterns and database operations.
    """

    def __init__(self, container: PostgreSQLTestContainer):
        self.container = container

    async def create_test_data(self, table_name: str, data: list[dict[str, Any]]):
        """Create test data in specified table."""
        if not data:
            return

        async with self.container.get_session() as session:
            # Build INSERT statement
            columns = list(data[0].keys())
            placeholders = ", ".join([f":{col}" for col in columns])
            sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

            for row in data:
                await session.execute(text(sql), row)

            await session.commit()

        logger.debug(f"Created {len(data)} test records in {table_name}")

    async def verify_database_constraints(self, table_name: str, constraint_tests: dict[str, Any]):
        """Verify database constraints are properly enforced."""
        results = {}

        for constraint_name, test_data in constraint_tests.items():
            try:
                await self.create_test_data(table_name, [test_data])
                results[constraint_name] = "passed_unexpectedly"
            except Exception as e:
                results[constraint_name] = f"correctly_rejected: {type(e).__name__}"

        return results

    async def measure_query_performance(self, sql: str, parameters: dict | None = None) -> dict[str, Any]:
        """Measure query execution time and explain plan."""
        import time

        async with self.container.get_session() as session:
            # Get explain plan
            explain_sql = f"EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS) {sql}"
            start_time = time.perf_counter()

            result = await session.execute(text(explain_sql), parameters or {})
            explain_data = result.fetchone()[0]

            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds

            return {
                "execution_time_ms": execution_time,
                "explain_plan": explain_data,
                "sql": sql,
                "parameters": parameters,
            }

    async def test_connection_pooling(self, concurrent_connections: int = 10) -> dict[str, Any]:
        """Test database connection pooling behavior."""
        import asyncio
        import time

        async def make_connection():
            start_time = time.perf_counter()
            async with self.container.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                await asyncio.sleep(0.1)  # Simulate work
                return time.perf_counter() - start_time

        start_time = time.perf_counter()
        tasks = [make_connection() for _ in range(concurrent_connections)]
        connection_times = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        return {
            "concurrent_connections": concurrent_connections,
            "total_time_ms": total_time * 1000,
            "avg_connection_time_ms": sum(connection_times) / len(connection_times) * 1000,
            "min_connection_time_ms": min(connection_times) * 1000,
            "max_connection_time_ms": max(connection_times) * 1000,
        }
