"""Database package for APES (Adaptive Prompt Enhancement System)
Modern SQLModel + SQLAlchemy 2.0 async implementation following 2025 best practices
"""

from .config import DatabaseConfig
from .connection import engine, get_session, sessionmanager
from .models import *

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
