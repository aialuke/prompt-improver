"""Database package for APES (Adaptive Prompt Enhancement System)
Modern SQLModel + SQLAlchemy 2.0 async implementation following 2025 best practices

Service Composition Architecture (New):
Uses clean service composition with DatabaseServices replacing the monolithic
UnifiedConnectionManager. Zero backwards compatibility - clean break approach.
"""

import contextlib
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.core.config import AppConfig
from prompt_improver.database.composition import (
    DatabaseServices,
    create_database_services,
    get_database_services,
)
from prompt_improver.database.factories import (
    SecurityContext,
    SecurityTier,
    create_security_context,
    create_security_context_from_auth_result,
)
# Security integration imports removed to avoid circular imports
# Import directly from security_integration module if needed
from prompt_improver.database.models import (
    ABExperiment,
    DiscoveredPattern,
    ImprovementSession,
    MLModelPerformance,
    RuleMetadata,
    RulePerformance,
    UserFeedback,
)
from prompt_improver.database.types import (
    ConnectionMode,
    HealthStatus,
    ManagerMode,
    PoolConfiguration,
    PoolState,
    RedisSecurityError,
    SecurityPerformanceMetrics,
    SecurityThreatScore,
    SecurityValidationResult,
)
from prompt_improver.database.utils import scalar

_global_services = None


async def _get_global_services():
    """Get or create the global DatabaseServices instance."""
    global _global_services
    if _global_services is None or _global_services._shutdown:
        _global_services = await create_database_services(ManagerMode.ASYNC_MODERN)
    return _global_services


@contextlib.asynccontextmanager
async def get_session():
    """Database session factory using DatabaseServices composition layer."""
    services = await _get_global_services()
    async with services.database.get_session() as session:
        yield session


@contextlib.asynccontextmanager
async def get_session_context():
    """Database session context using DatabaseServices composition layer."""
    services = await _get_global_services()
    async with services.database.get_session() as session:
        yield session


async def get_async_session_factory():
    """Get async session factory from database service."""
    services = await _get_global_services()
    if hasattr(services.database, "_async_session_factory"):
        return services.database._async_session_factory
    return None


async def _get_global_sessionmanager():
    """Get global session manager (DatabaseServices)."""
    return await _get_global_services()


async def get_sessionmanager():
    """Get the global database session manager (DatabaseServices)."""
    return await _get_global_services()


# Using DatabaseServices (clean composition architecture)


async def get_sessionmanager_sync() -> DatabaseServices:
    """Get the global database session manager.

    Returns:
        DatabaseServices: The composed database services instance

    Raises:
        RuntimeError: If service initialization fails
    """
    return await _get_global_sessionmanager()


async def get_database_services_dependency(
    mode: ManagerMode = ManagerMode.ASYNC_MODERN,
) -> DatabaseServices:
    """FastAPI dependency function to get database services instance."""
    services = await get_database_services(mode)
    if services is None:
        services = await create_database_services(mode)
    return services


# MCP session function using clean architecture
async def get_mcp_session():
    """Get MCP session using the new composition layer."""
    services = await get_database_services(ManagerMode.MCP_SERVER)
    if services is None:
        services = await create_database_services(ManagerMode.MCP_SERVER)
    return services.database.get_session(ConnectionMode.read_write)


# FastAPI dependency annotations (needs sync wrapper functions)
def get_session_dependency():
    """Sync wrapper for FastAPI dependency injection."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(get_session())
    except RuntimeError:
        # No event loop - return factory function
        return get_session


def get_database_services_dependency_sync(
    mode: ManagerMode = ManagerMode.ASYNC_MODERN,
):
    """Sync wrapper for FastAPI dependency injection."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(get_database_services_dependency(mode))
    except RuntimeError:
        # No event loop - needs to be called within async context
        return lambda: get_database_services_dependency(mode)


db_session = Annotated[AsyncSession, Depends(get_session_dependency)]
database_services = Annotated[
    DatabaseServices, Depends(get_database_services_dependency_sync)
]

__all__ = [
    "ABExperiment",
    "AppConfig",
    "ConnectionMode",
    "DatabaseServices",
    "DiscoveredPattern",
    "HealthStatus",
    "ImprovementSession",
    "MLModelPerformance",
    "ManagerMode",
    "PoolConfiguration",
    "PoolState",
    "RedisSecurityError",
    "RuleMetadata",
    "RulePerformance",
    "SecurityContext",
    "SecurityPerformanceMetrics",
    "SecurityThreatScore",
    "SecurityTier",
    "SecurityValidationResult",
    "UserFeedback",
    "create_database_services",
    "create_security_context",
    "create_security_context_from_auth_result",
    # "create_security_context_from_security_manager", # Removed to avoid circular imports
    "database_services",
    "db_session",
    "get_async_session_factory",
    "get_database_services",
    "get_database_services_dependency",
    "get_mcp_session",
    "get_session",
    "get_session_context",
    "get_sessionmanager",
    "scalar",
]
