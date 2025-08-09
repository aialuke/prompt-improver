"""Database package for APES (Adaptive Prompt Enhancement System)
Modern SQLModel + SQLAlchemy 2.0 async implementation following 2025 best practices

Unified Connection Manager (Default):
Uses consolidated UnifiedConnectionManager for all database operations.
Clean, direct implementation without legacy compatibility layers.
"""
import contextlib
from typing import Annotated
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from prompt_improver.core.config import AppConfig
from prompt_improver.database.models import ABExperiment, DiscoveredPattern, ImprovementSession, MLModelPerformance, RuleMetadata, RulePerformance, UserFeedback
from prompt_improver.database.unified_connection_manager import ConnectionMode, ManagerMode, UnifiedConnectionManager, get_mcp_session, get_unified_manager
from prompt_improver.database.utils import scalar
_global_manager = None

def _get_global_manager():
    """Get or create the global manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = UnifiedConnectionManager(ManagerMode.ASYNC_MODERN)
    return _global_manager

@contextlib.asynccontextmanager
async def get_session():
    """Database session factory using unified manager"""
    manager = _get_global_manager()
    async with manager.get_async_session() as session:
        yield session

@contextlib.asynccontextmanager
async def get_session_context():
    """Database session context using unified manager"""
    manager = _get_global_manager()
    async with manager.get_async_session() as session:
        yield session

def get_async_session_factory():
    """Get async session factory from unified manager"""
    manager = _get_global_manager()
    return manager._async_session_factory

def _get_global_sessionmanager():
    """Get global session manager (unified manager)"""
    return _get_global_manager()

def get_sessionmanager():
    """Get the global session manager (UnifiedConnectionManager)"""
    return _get_global_manager()
SessionProvider = UnifiedConnectionManager
print('🔄 Using UnifiedConnectionManager (default, consolidated connection management)')

def get_sessionmanager() -> UnifiedConnectionManager:
    """Get the global database session manager.

    Returns:
        UnifiedConnectionManager: The non-optional session manager instance

    Raises:
        RuntimeError: If session manager initialization fails
    """
    return _get_global_sessionmanager()

def get_unified_manager_dependency(mode: ManagerMode=ManagerMode.ASYNC_MODERN) -> UnifiedConnectionManager:
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
db_session = Annotated[AsyncSession, Depends(get_session)]
unified_manager = Annotated[UnifiedConnectionManager, Depends(get_unified_manager_async_modern)]
__all__ = ['RulePerformance', 'UserFeedback', 'ImprovementSession', 'MLModelPerformance', 'DiscoveredPattern', 'RuleMetadata', 'ABExperiment', 'UnifiedConnectionManager', 'get_unified_manager', 'ManagerMode', 'AsyncSessionFactory', 'SessionProvider', 'get_session', 'get_session_context', 'get_async_session_factory', 'get_sessionmanager', 'get_unified_manager_dependency', 'get_unified_manager_async_modern', 'get_unified_manager_ml_training', 'get_unified_manager_mcp_server', 'db_session', 'unified_manager', 'AppConfig', 'scalar', 'get_mcp_session']
