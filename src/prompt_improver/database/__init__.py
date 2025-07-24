"""Database package for APES (Adaptive Prompt Enhancement System)
Modern SQLModel + SQLAlchemy 2.0 async implementation following 2025 best practices
"""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from .config import DatabaseConfig
from .connection import (
    AsyncSessionFactory,
    DatabaseManager,
    DatabaseSessionManager,
    SessionProvider,
    SyncSessionFactory,
    _get_global_sessionmanager,
    get_async_session_factory,
    get_session,
    get_session_context,
)
from .models import (
    ABExperiment,
    ImprovementSession,
    MLModelPerformance,
    RuleMetadata,
    RulePerformance,
    UserFeedback,
)
from .utils import scalar

def get_sessionmanager() -> DatabaseSessionManager:
    """Get the global database session manager.

    Returns:
        DatabaseSessionManager: The non-optional session manager instance

    Raises:
        RuntimeError: If session manager initialization fails
    """
    return _get_global_sessionmanager()

# Annotated type for use in FastAPI endpoints
db_session = Annotated[AsyncSession, Depends(get_session)]

__all__ = [
    # Models
    "RulePerformance",
    "RuleCombination",
    "UserFeedback",
    "ImprovementSession",
    "MLModelPerformance",
    "DiscoveredPattern",
    "RuleMetadata",
    "ABExperiment",
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
    "get_session_context",
    "get_async_session_factory",
    "get_sessionmanager",
    # Annotated types for endpoints
    "db_session",
    # Configuration
    "DatabaseConfig",
    # Utilities
    "scalar",
]
