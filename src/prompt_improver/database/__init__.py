"""Database package for APES (Adaptive Prompt Enhancement System)
Modern SQLModel + SQLAlchemy 2.0 async implementation following 2025 best practices

Unified Connection Manager (Default):
Uses consolidated UnifiedConnectionManager for all database operations.
Clean, direct implementation without legacy compatibility layers.
"""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import AppConfig

# Use unified connection manager by default (no feature flag needed)
from .unified_connection_manager import (
    get_unified_manager,
    ManagerMode,
    UnifiedConnectionManager,
)

# Global manager instance
_global_manager = UnifiedConnectionManager(ManagerMode.ASYNC_MODERN)

# Note: DatabaseManager and DatabaseSessionManager aliases removed - use UnifiedConnectionManager directly

# Session providers using unified manager directly
async def get_session():
    """Database session factory using unified manager"""
    async with _global_manager.get_async_session() as session:
        yield session

async def get_session_context():
    """Database session context using unified manager"""
    async with _global_manager.get_async_session() as session:
        yield session

def get_async_session_factory():
    """Get async session factory from unified manager"""
    return _global_manager._async_session_factory

def _get_global_sessionmanager():
    """Get global session manager (unified manager)"""
    return _global_manager

# Type aliases for compatibility
AsyncSessionFactory = type(get_async_session_factory())

# Protocol import - Unified manager is the SessionProvider
SessionProvider = UnifiedConnectionManager

print("🔄 Using UnifiedConnectionManager (default, consolidated connection management)")

# Direct imports without aliases
from .unified_connection_manager import (
    ManagerMode,
    get_unified_manager,
)
# Import ConnectionMode from unified_connection_manager to avoid circular imports
from .unified_connection_manager import ConnectionMode

# Import session functions from their respective modules
from .mcp_connection_pool import get_mcp_session
from .models import (
    ABExperiment,
    DiscoveredPattern,
    ImprovementSession,
    MLModelPerformance,
    RuleMetadata,
    RulePerformance,
    UserFeedback,
)
from .utils import scalar

def get_sessionmanager() -> UnifiedConnectionManager:
    """Get the global database session manager.

    Returns:
        UnifiedConnectionManager: The non-optional session manager instance

    Raises:
        RuntimeError: If session manager initialization fails
    """
    return _get_global_sessionmanager()

# Annotated type for use in FastAPI endpoints
db_session = Annotated[AsyncSession, Depends(get_session)]

__all__ = [
    # Models
    "RulePerformance",
    "UserFeedback",
    "ImprovementSession",
    "MLModelPerformance",
    "DiscoveredPattern",
    "RuleMetadata",
    "ABExperiment",
    # Core classes (use UnifiedConnectionManager directly)
    # Unified Manager
    "UnifiedConnectionManager",
    "get_unified_manager",
    "ManagerMode",
    # Type aliases
    "AsyncSessionFactory",
    # Protocols
    "SessionProvider",
    # Main session providers
    "get_session",
    "get_session_context",
    "get_async_session_factory",
    "get_sessionmanager",
    # Annotated types for endpoints
    "db_session",
    # Configuration (access via AppConfig)
    "AppConfig",
    # Utilities
    "scalar",
    # MCP session support
    "get_mcp_session",
]
