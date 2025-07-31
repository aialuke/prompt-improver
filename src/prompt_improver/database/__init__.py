"""Database package for APES (Adaptive Prompt Enhancement System)
Modern SQLModel + SQLAlchemy 2.0 async implementation following 2025 best practices

Unified Connection Manager (Default):
Uses consolidated UnifiedConnectionManager for all database operations.
Clean, direct implementation without legacy compatibility layers.
"""

from typing import Annotated
import contextlib

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
@contextlib.asynccontextmanager
async def get_session():
    """Database session factory using unified manager"""
    async with _global_manager.get_async_session() as session:
        yield session

@contextlib.asynccontextmanager
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
from .unified_connection_manager import get_mcp_session
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

# ========== FastAPI Dependency Functions ==========

def get_unified_manager_dependency(mode: ManagerMode = ManagerMode.ASYNC_MODERN) -> UnifiedConnectionManager:
    """FastAPI dependency function to get unified manager instance"""
    return get_unified_manager(mode)

def get_unified_manager_async_modern() -> UnifiedConnectionManager:
    """FastAPI dependency for ASYNC_MODERN mode (general purpose)"""
    return get_unified_manager(ManagerMode.ASYNC_MODERN)

def get_unified_manager_ml_training() -> UnifiedConnectionManager:
    """FastAPI dependency for ML_TRAINING mode"""
    return get_unified_manager(ManagerMode.ML_TRAINING)

def get_unified_manager_mcp_server() -> UnifiedConnectionManager:
    """FastAPI dependency for MCP_SERVER mode"""
    return get_unified_manager(ManagerMode.MCP_SERVER)

# Annotated types for use in FastAPI endpoints
db_session = Annotated[AsyncSession, Depends(get_session)]
unified_manager = Annotated[UnifiedConnectionManager, Depends(get_unified_manager_async_modern)]

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
    # FastAPI dependency functions
    "get_unified_manager_dependency",
    "get_unified_manager_async_modern",
    "get_unified_manager_ml_training",
    "get_unified_manager_mcp_server",
    # Annotated types for endpoints
    "db_session",
    "unified_manager",
    # Configuration (access via AppConfig)
    "AppConfig",
    # Utilities
    "scalar",
    # MCP session support
    "get_mcp_session",
]
