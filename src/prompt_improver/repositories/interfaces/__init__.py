"""Repository interface definitions for dependency injection.

This module provides concrete interface classes that combine the protocol
definitions with implementation requirements, enabling clean dependency
injection and testability.
"""

from prompt_improver.repositories.interfaces.repository_interfaces import (
    IAnalyticsRepository,
    IAprioriRepository,
    IHealthRepository,
    IMLRepository,
    IRepositoryFactory,
    IRulesRepository,
    IUserFeedbackRepository,
)

__all__ = [
    "IAnalyticsRepository",
    "IAprioriRepository",
    "IHealthRepository",
    "IMLRepository",
    "IRepositoryFactory",
    "IRulesRepository",
    "IUserFeedbackRepository",
]
