"""Database package for APES (Adaptive Prompt Enhancement System)
Modern SQLModel + SQLAlchemy 2.0 async implementation following 2025 best practices"""

from typing_extensions import Annotated
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from .config import DatabaseConfig
from .connection import (
    DatabaseManager,
    DatabaseSessionManager,
    SessionProvider,
    SyncSessionFactory,
    AsyncSessionFactory,
    get_session,
    get_async_session_factory,
    sessionmanager,
    engine,
)
from .models import *

# Annotated type for use in FastAPI endpoints
DBSession = Annotated[AsyncSession, Depends(get_session)]

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
    "get_async_session_factory",
    # Annotated types for endpoints
    "DBSession",
    # Legacy compatibility
    "sessionmanager",
    "engine",
    # Configuration
    "DatabaseConfig",
]
