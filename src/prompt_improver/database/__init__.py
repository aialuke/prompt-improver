"""
Database package for APES (Adaptive Prompt Enhancement System)
Modern SQLModel + SQLAlchemy 2.0 async implementation following 2025 best practices
"""

from .models import *
from .connection import get_session, engine, sessionmanager
from .config import DatabaseConfig

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
    # Database connection
    "get_session",
    "engine",
    "sessionmanager",
    # Configuration
    "DatabaseConfig",
]
