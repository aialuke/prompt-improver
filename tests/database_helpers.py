"""Database test helpers following pytest-postgresql best practices."""

import asyncio
import logging
from typing import Any, Optional

import asyncpg
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel

logger = logging.getLogger(__name__)


async def wait_for_postgres_async(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    max_retries: int = 30,
    retry_delay: float = 1.0,
) -> bool:
    """Wait for PostgreSQL to be ready with async connection.

    Best practice: Always verify database is ready before running tests.
    """
    for attempt in range(max_retries):
        try:
            # Try to connect directly with asyncpg
            conn = await asyncpg.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                timeout=5.0,
            )
            await conn.close()
            logger.info(f"PostgreSQL ready after {attempt + 1} attempts")
            return True
        except (asyncpg.PostgresError, OSError) as e:
            if attempt == max_retries - 1:
                logger.error(f"PostgreSQL not ready after {max_retries} attempts: {e}")
                return False
            await asyncio.sleep(retry_delay)
    return False


def wait_for_postgres_sync(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    max_retries: int = 30,
    retry_delay: float = 1.0,
) -> bool:
    """Wait for PostgreSQL to be ready using async in sync context.

    Used for initial setup and verification.
    """
    try:
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            wait_for_postgres_async(
                host, port, user, password, database, max_retries, retry_delay
            )
        )
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Failed to check PostgreSQL readiness: {e}")
        return False


async def ensure_test_database_exists(
    host: str, port: int, user: str, password: str, test_db_name: str = "apes_test"
) -> bool:
    """Ensure test database exists, create if necessary.

    Best practice: Use template database for faster test setup.
    """
    try:
        # Connect to default postgres database
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres",
            timeout=5.0,
        )

        # Check if test database exists
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", test_db_name
        )

        if not exists:
            # Create test database
            await conn.execute(f'CREATE DATABASE "{test_db_name}"')
            logger.info(f"Created test database: {test_db_name}")

        await conn.close()
        return True

    except Exception as e:
        logger.error(f"Failed to ensure test database exists: {e}")
        return False


async def create_test_engine_with_retry(
    db_url: str, max_retries: int = 3, **engine_kwargs: Any
) -> Optional[AsyncEngine]:
    """Create SQLAlchemy async engine with retry logic.

    Best practice: Handle transient connection failures gracefully.
    """
    for attempt in range(max_retries):
        try:
            engine = create_async_engine(
                db_url, echo=False, future=True, **engine_kwargs
            )

            # Test the connection
            async with engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)

            return engine

        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(
                    f"Failed to create engine after {max_retries} attempts: {e}"
                )
                raise
            await asyncio.sleep(1)

    return None
