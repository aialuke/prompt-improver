"""
Modern SQLAlchemy 2.0 async connection management following 2025 best practices
"""

import contextlib
from typing import AsyncIterator, Optional
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel

from .config import DatabaseConfig


class DatabaseSessionManager:
    """
    Modern async session manager following SQLAlchemy 2.0 best practices
    Handles connection pooling and session lifecycle automatically
    """

    def __init__(self, database_url: str, echo: bool = False):
        # Configure engine based on environment
        engine_kwargs = {
            "echo": echo,
        }

        # Use NullPool for localhost development to avoid connection issues
        if "localhost" in database_url:
            engine_kwargs["poolclass"] = NullPool
        else:
            # Production settings with research-validated connection pooling
            config_obj = DatabaseConfig()
            engine_kwargs.update({
                "pool_size": config_obj.pool_min_size,
                "max_overflow": config_obj.pool_max_size - config_obj.pool_min_size,
                "pool_timeout": config_obj.pool_timeout,
                "pool_recycle": config_obj.pool_max_lifetime,
                "pool_pre_ping": True,  # Verify connections before use
                "pool_reset_on_return": "commit",  # Clean state for reused connections
            })

        self.engine = create_async_engine(database_url, **engine_kwargs)
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def close(self):
        """Close the database engine and all connections"""
        if self.engine:
            await self.engine.dispose()

    @contextlib.asynccontextmanager
    async def connect(self) -> AsyncIterator[AsyncConnection]:
        """Get a raw async connection"""
        async with self.engine.begin() as connection:
            try:
                yield connection
            except Exception:
                await connection.rollback()
                raise

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Get an async session with automatic commit/rollback"""
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def create_all(self, connection: AsyncConnection):
        """Create all tables (for testing/development)"""
        await connection.run_sync(SQLModel.metadata.create_all)

    async def drop_all(self, connection: AsyncConnection):
        """Drop all tables (for testing)"""
        await connection.run_sync(SQLModel.metadata.drop_all)


# Global configuration and session manager
config = DatabaseConfig()
sessionmanager = DatabaseSessionManager(config.database_url, echo=config.echo_sql)

# For backwards compatibility
engine = sessionmanager.engine


async def get_session() -> AsyncIterator[AsyncSession]:
    """
    FastAPI dependency function to get database session
    Use with Depends(get_session) in FastAPI endpoints
    """
    async with sessionmanager.session() as session:
        yield session
