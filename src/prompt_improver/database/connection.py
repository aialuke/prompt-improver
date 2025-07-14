
# Public API with proper typing
__all__ = [
    # Core classes
    "DatabaseManager",
    "DatabaseSessionManager",
    # Type aliases
    "SyncSessionFactory",
    "AsyncSessionFactory",
    # Protocols
    "SessionProvider",
    # Main session providers
    "get_session",
    "get_async_session_factory",
    # Legacy compatibility
    "get_async_session",
    "get_legacy_session",
    "sessionmanager",
    "engine",
    "async_engine",
    "SessionLocal",
    "async_session_factory",
    # Utilities
    "get_database_url",
]

"""Modern SQLAlchemy 2.0 async connection management following 2025 best practices"""

import contextlib
import logging
import os
from typing import AsyncIterator, Iterator, Optional, Protocol, runtime_checkable

from sqlalchemy import create_engine, Engine
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine,
)
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel
from typing_extensions import Annotated
from fastapi import Depends

from .config import DatabaseConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class SessionProvider(Protocol):
    """Protocol for session providers to ensure consistent interface"""
    
    async def get_session(self) -> AsyncIterator[AsyncSession]:
        """Get an async session"""
        ...
    
    async def close(self) -> None:
        """Close the session provider"""
        ...


# Type aliases for better clarity
SyncSessionFactory = sessionmaker[Session]
AsyncSessionFactory = async_sessionmaker[AsyncSession]


class DatabaseManager:
    """Synchronous database manager for operations that don't need async.
    
    Provides synchronous database connections for components like AprioriAnalyzer
    that work better with traditional synchronous database operations.
    """
    
    def __init__(self, database_url: str, echo: bool = False):
        # Keep psycopg3 URL for synchronous operations, only convert asyncpg
        if database_url.startswith('postgresql+asyncpg://'):
            database_url = database_url.replace('postgresql+asyncpg://', 'postgresql+psycopg://')
        
        self.engine = create_engine(
            database_url, 
            echo=echo,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        self._session_factory: SyncSessionFactory = sessionmaker(bind=self.engine)
    
    @property
    def session_factory(self) -> SyncSessionFactory:
        """Get the synchronous session factory with accurate typing"""
        return self._session_factory
    
    @contextlib.contextmanager
    def get_connection(self) -> Iterator[object]:
        """Get a database connection using context manager."""
        connection = self.engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    @contextlib.contextmanager
    def get_session(self) -> Iterator[Session]:
        """Get a database session using context manager."""
        session = self._session_factory()
        try:
            yield session
        finally:
            session.close()
    
    def close(self):
        """Close the database engine."""
        self.engine.dispose()


class DatabaseSessionManager:
    """Modern async database session manager following SQLAlchemy 2.0 patterns"""

    def __init__(self, database_url: str, echo: bool = False):
        # Configure engine based on environment
        engine_kwargs = {
            "echo": echo,
            "poolclass": NullPool,  # For development/testing
        }
        
        # Use provided database URL
        self._engine = create_async_engine(database_url, **engine_kwargs)
        self._sessionmaker: AsyncSessionFactory = async_sessionmaker(
            bind=self._engine, expire_on_commit=False
        )

    @property
    def session_factory(self) -> AsyncSessionFactory:
        """Get the async session factory with accurate typing"""
        return self._sessionmaker

    async def close(self):
        """Close the database engine and all connections"""
        if self._engine:
            await self._engine.dispose()

    @contextlib.asynccontextmanager
    async def connect(self) -> AsyncIterator[AsyncConnection]:
        """Get a raw async connection"""
        async with self._engine.begin() as connection:
            try:
                yield connection
            except Exception:
                await connection.rollback()
                raise

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Get an async session with automatic commit/rollback"""
        session = self._sessionmaker()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def get_session(self) -> AsyncIterator[AsyncSession]:
        """Get an async session - implements SessionProvider protocol"""
        async with self.session() as session:
            yield session

    async def create_all(self, connection: AsyncConnection):
        """Create all tables (for testing/development)"""
        await connection.run_sync(SQLModel.metadata.create_all)

    async def drop_all(self, connection: AsyncConnection):
        """Drop all tables (for testing)"""
        await connection.run_sync(SQLModel.metadata.drop_all)


# Global session manager with Optional guard
_global_sessionmanager: Optional[DatabaseSessionManager] = None


def _get_global_sessionmanager() -> DatabaseSessionManager:
    """Get or create the global session manager with proper error handling"""
    global _global_sessionmanager
    
    if _global_sessionmanager is None:
        try:
            default_database_url = os.getenv(
                'DATABASE_URL', 
                'postgresql+psycopg://apes_user:apes_secure_password_2024@localhost:5432/apes_production'
            )
            _global_sessionmanager = DatabaseSessionManager(default_database_url, echo=False)
        except Exception as e:
            logger.error(f"Failed to initialize global sessionmanager: {e}")
            raise RuntimeError(f"Database session manager initialization failed: {e}") from e
    
    return _global_sessionmanager


def get_async_session_factory() -> AsyncSessionFactory:
    """Get the async session factory for other modules
    
    Returns:
        AsyncSessionFactory: The async session factory instance
    """
    return _get_global_sessionmanager().session_factory


async def get_session() -> AsyncIterator[AsyncSession]:
    """Database session factory for async operations
    
    Provides async database sessions with automatic commit/rollback.
    This is the main session provider for FastAPI endpoints.
    """
    session_manager = _get_global_sessionmanager()
    async with session_manager.session() as session:
        yield session


# Legacy compatibility - use Optional guard
sessionmanager: Optional[DatabaseSessionManager] = None
try:
    sessionmanager = _get_global_sessionmanager()
    engine = sessionmanager._engine
except Exception as e:
    logger.warning(f"Legacy sessionmanager initialization failed: {e}")
    engine = None


# Legacy support - maintained for backward compatibility
from prompt_improver.database.config import get_database_config


def get_database_url(async_driver: bool = False) -> str:
    """Get database URL with driver selection"""
    db_config = get_database_config()
    db_user = db_config.postgres_username
    db_password = db_config.postgres_password
    db_host = db_config.postgres_host
    db_port = db_config.postgres_port
    db_name = db_config.postgres_database

    if async_driver:
        return f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    else:
        return f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


# Legacy global instances - use proper session manager instead
engine = create_engine(get_database_url(), pool_pre_ping=True)
async_engine = create_async_engine(get_database_url(async_driver=True))

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
async_session_factory = async_sessionmaker(async_engine, expire_on_commit=False)


def get_legacy_session() -> Iterator[Session]:
    """Legacy synchronous session - use DatabaseManager instead"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_session() -> AsyncIterator[AsyncSession]:
    """Legacy async session - use get_session() instead"""
    async with async_session_factory() as session:
        yield session
