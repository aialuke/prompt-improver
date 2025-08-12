"""Repository Layer Architecture for Clean Database Access.

This module provides a clean repository pattern implementation that eliminates
direct database access from the API layer, following clean architecture principles:

Architecture: API → Service → Repository → Database

Key Components:
- Repository Protocols: Abstract interfaces for all repository operations
- Repository Implementations: Concrete implementations with database logic
- Repository Factory: Dependency injection system for repository creation
- Transaction Management: Cross-repository transaction support
- Query Builders: Type-safe query construction utilities

Usage:
    from prompt_improver.repositories import IRepositoryFactory, get_repository_factory

    # Get repository factory
    factory = await get_repository_factory()

    # Create domain repositories
    analytics_repo = await factory.create_analytics_repository()
    rules_repo = await factory.create_rules_repository()

    # Use repositories in service layer
    sessions = await analytics_repo.get_prompt_sessions(limit=10)
    rules = await rules_repo.get_rules(enabled_only=True)

Benefits:
- Clean separation of concerns
- Improved testability with repository mocking
- Consistent database access patterns
- Transaction management across repositories
- Type-safe query building
- Performance optimization at repository level
"""

# Repository Interfaces
from prompt_improver.repositories.interfaces import (
    IAnalyticsRepository,
    IAprioriRepository,
    IHealthRepository,
    IMLRepository,
    IRepositoryFactory,
    IRulesRepository,
    IUserFeedbackRepository,
)

# Repository Protocols
from prompt_improver.repositories.protocols import (
    AnalyticsRepositoryProtocol,
    AprioriRepositoryProtocol,
    BaseRepositoryProtocol,
    HealthRepositoryProtocol,
    MLRepositoryProtocol,
    QueryBuilderProtocol,
    RulesRepositoryProtocol,
    TransactionManagerProtocol,
    UserFeedbackRepositoryProtocol,
)

__all__ = [
    # Repository Interfaces for DI
    "IAnalyticsRepository",
    "IAprioriRepository",
    "IHealthRepository",
    "IMLRepository",
    "IRepositoryFactory",
    "IRulesRepository",
    "IUserFeedbackRepository",
    # Repository Protocols for Implementation
    "AnalyticsRepositoryProtocol",
    "AprioriRepositoryProtocol",
    "BaseRepositoryProtocol",
    "HealthRepositoryProtocol",
    "MLRepositoryProtocol",
    "QueryBuilderProtocol",
    "RulesRepositoryProtocol",
    "TransactionManagerProtocol",
    "UserFeedbackRepositoryProtocol",
]


# Factory function will be implemented in the impl module
# Placeholder for now - actual implementation will be in impl/repository_factory.py
async def get_repository_factory() -> IRepositoryFactory:
    """Get repository factory instance.

    This will be implemented once the concrete repository implementations
    are created in the impl/ directory.

    Returns:
        IRepositoryFactory: Repository factory for creating repository instances
    """
    raise NotImplementedError(
        "Repository factory implementation pending - "
        "will be available after concrete repositories are implemented"
    )
