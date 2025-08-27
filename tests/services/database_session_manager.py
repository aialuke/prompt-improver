"""Test database session manager implementing SessionManagerProtocol.

Provides clean architecture compliant database access for tests by implementing
the SessionManagerProtocol interface instead of using direct database imports.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from prompt_improver.database.composition import DatabaseServices
from prompt_improver.shared.interfaces.protocols.database import (
    SessionManagerProtocol,
    SessionProtocol,
)

logger = logging.getLogger(__name__)


class TestDatabaseSessionManager(SessionManagerProtocol):
    """Test implementation of SessionManagerProtocol for clean architecture compliance.

    This replaces direct database imports in tests with proper protocol-based
    dependency injection, achieving the repository pattern goals.
    """

    def __init__(self, database_services: DatabaseServices) -> None:
        """Initialize test session manager with DatabaseServices dependency."""
        self._database_services = database_services
        self._initialized = False
        logger.debug("TestDatabaseSessionManager initialized")

    async def initialize(self) -> None:
        """Initialize the database services if needed."""
        if not self._initialized:
            await self._database_services.initialize()
            self._initialized = True
            logger.debug("TestDatabaseSessionManager database services initialized")

    async def get_session(self) -> SessionProtocol:
        """Get a new database session."""
        if not self._initialized:
            await self.initialize()
        return await self._database_services.database.get_session()

    @asynccontextmanager
    async def session_context(self) -> AsyncGenerator[SessionProtocol, None]:
        """Get async context manager for session with automatic cleanup."""
        if not self._initialized:
            await self.initialize()
        async with self._database_services.database.session_context() as session:
            yield session

    @asynccontextmanager
    async def transaction_context(self) -> AsyncGenerator[SessionProtocol, None]:
        """Get async context manager for transactional session."""
        if not self._initialized:
            await self.initialize()
        async with self._database_services.database.transaction_context() as session:
            yield session

    async def health_check(self) -> bool:
        """Perform health check on database connection."""
        if not self._initialized:
            await self.initialize()
        return await self._database_services.database.health_check()

    async def shutdown(self) -> None:
        """Shutdown database services."""
        if self._initialized:
            await self._database_services.shutdown_all()
            self._initialized = False
            logger.debug("TestDatabaseSessionManager shutdown completed")


async def create_test_session_manager(connection_string: str) -> TestDatabaseSessionManager:
    """Factory function to create test session manager with proper configuration.

    Args:
        connection_string: PostgreSQL connection string for test database

    Returns:
        Initialized TestDatabaseSessionManager implementing SessionManagerProtocol
    """
    from prompt_improver.database.composition import create_database_services

    # Create DatabaseServices with test configuration
    database_services = await create_database_services(
        connection_string=connection_string,
        pool_size=2,  # Small pool for tests
        max_overflow=1,
        echo=False  # Quiet logging for tests
    )

    # Create and initialize session manager
    session_manager = TestDatabaseSessionManager(database_services)
    await session_manager.initialize()

    logger.info("Test session manager created and initialized")
    return session_manager
