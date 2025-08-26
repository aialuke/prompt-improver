"""Repository factory for dependency injection and centralized repository management.

Provides factory functions and dependency injection setup for all repository
implementations, ensuring consistent configuration and lifecycle management.
"""

import logging
from typing import TypeVar

from prompt_improver.database import DatabaseServices
from prompt_improver.repositories.impl.analytics_repository import AnalyticsRepository
from prompt_improver.repositories.impl.apriori_repository import AprioriRepository
from prompt_improver.repositories.impl.health_repository import HealthRepository
from prompt_improver.repositories.impl.ml_repository_service import MLRepositoryFacade
from prompt_improver.repositories.impl.user_feedback_repository import (
    UserFeedbackRepository,
)
from prompt_improver.repositories.protocols.analytics_repository_protocol import (
    AnalyticsRepositoryProtocol,
)
from prompt_improver.repositories.protocols.apriori_repository_protocol import (
    AprioriRepositoryProtocol,
)
from prompt_improver.repositories.protocols.health_repository_protocol import (
    HealthRepositoryProtocol,
)
from prompt_improver.repositories.protocols.ml_repository_protocol import (
    MLRepositoryProtocol,
)
from prompt_improver.repositories.protocols.user_feedback_repository_protocol import (
    UserFeedbackRepositoryProtocol,
)

logger = logging.getLogger(__name__)

# Type variable for repository protocols
RepositoryProtocol = TypeVar("RepositoryProtocol")


class RepositoryFactory:
    """Factory for creating and managing repository instances with dependency injection."""

    def __init__(self, connection_manager: DatabaseServices) -> None:
        """Initialize repository factory with connection manager.

        Args:
            connection_manager: DatabaseServices instance for database operations
        """
        self.connection_manager = connection_manager
        self._repository_cache: dict[type, object] = {}
        self._initialized = False
        logger.info("Repository factory initialized")

    def initialize(self) -> None:
        """Initialize all repositories and prepare factory for use."""
        if self._initialized:
            return

        try:
            # Pre-create all repositories for better performance
            self._repository_cache[AnalyticsRepositoryProtocol] = AnalyticsRepository(
                self.connection_manager
            )
            self._repository_cache[AprioriRepositoryProtocol] = AprioriRepository(
                self.connection_manager
            )
            self._repository_cache[MLRepositoryProtocol] = MLRepositoryFacade(
                self.connection_manager
            )
            self._repository_cache[UserFeedbackRepositoryProtocol] = (
                UserFeedbackRepository(self.connection_manager)
            )
            self._repository_cache[HealthRepositoryProtocol] = HealthRepository(
                self.connection_manager
            )

            self._initialized = True
            logger.info("Repository factory initialization complete")

        except Exception as e:
            logger.exception(f"Repository factory initialization failed: {e}")
            raise

    def get_analytics_repository(self) -> AnalyticsRepositoryProtocol:
        """Get analytics repository instance.

        Returns:
            AnalyticsRepositoryProtocol implementation
        """
        if not self._initialized:
            self.initialize()

        return self._repository_cache[AnalyticsRepositoryProtocol]

    def get_apriori_repository(self) -> AprioriRepositoryProtocol:
        """Get apriori repository instance.

        Returns:
            AprioriRepositoryProtocol implementation
        """
        if not self._initialized:
            self.initialize()

        return self._repository_cache[AprioriRepositoryProtocol]

    def get_ml_repository(self) -> MLRepositoryProtocol:
        """Get ML repository instance.

        Returns:
            MLRepositoryProtocol implementation
        """
        if not self._initialized:
            self.initialize()

        return self._repository_cache[MLRepositoryProtocol]

    def get_user_feedback_repository(self) -> UserFeedbackRepositoryProtocol:
        """Get user feedback repository instance.

        Returns:
            UserFeedbackRepositoryProtocol implementation
        """
        if not self._initialized:
            self.initialize()

        return self._repository_cache[UserFeedbackRepositoryProtocol]

    def get_health_repository(self) -> HealthRepositoryProtocol:
        """Get health repository instance.

        Returns:
            HealthRepositoryProtocol implementation
        """
        if not self._initialized:
            self.initialize()

        return self._repository_cache[HealthRepositoryProtocol]

    def get_repository(
        self, protocol_type: type[RepositoryProtocol]
    ) -> RepositoryProtocol:
        """Get repository by protocol type (generic method).

        Args:
            protocol_type: Repository protocol type to retrieve

        Returns:
            Repository implementation matching the protocol

        Raises:
            ValueError: If protocol type is not supported
        """
        if not self._initialized:
            self.initialize()

        if protocol_type not in self._repository_cache:
            raise ValueError(f"Unsupported repository protocol: {protocol_type}")

        return self._repository_cache[protocol_type]

    async def health_check(self) -> dict[str, bool]:
        """Perform health check on all repositories.

        Returns:
            Dictionary mapping repository names to health status
        """
        health_status = {}

        try:
            if not self._initialized:
                self.initialize()

            # Check each repository
            repositories = {
                "analytics": self.get_analytics_repository(),
                "apriori": self.get_apriori_repository(),
                "ml": self.get_ml_repository(),
                "user_feedback": self.get_user_feedback_repository(),
                "health": self.get_health_repository(),
            }

            for name, repo in repositories.items():
                try:
                    # Use base repository health check if available
                    if hasattr(repo, "health_check"):
                        health_result = await repo.health_check()
                        health_status[name] = health_result.get("status") == "healthy"
                    # Basic connectivity test for health repository
                    elif name == "health":
                        health_check_result = await repo.check_database_health()
                        health_status[name] = health_check_result.status == "healthy"
                    else:
                        health_status[name] = (
                            True  # Assume healthy if no check available
                        )

                except Exception as e:
                    logger.exception(f"Health check failed for {name} repository: {e}")
                    health_status[name] = False

        except Exception as e:
            logger.exception(f"Repository factory health check failed: {e}")
            return {"factory": False}

        return health_status

    def cleanup(self) -> None:
        """Cleanup repository factory resources."""
        try:
            self._repository_cache.clear()
            self._initialized = False
            logger.info("Repository factory cleanup complete")
        except Exception as e:
            logger.exception(f"Repository factory cleanup failed: {e}")


# Global factory instance - will be initialized by dependency injection
_factory_instance: RepositoryFactory | None = None


def get_repository_factory(
    connection_manager: DatabaseServices | None = None,
) -> RepositoryFactory:
    """Get or create the global repository factory instance.

    Args:
        connection_manager: Connection manager instance. Required on first call.

    Returns:
        RepositoryFactory instance

    Raises:
        ValueError: If connection_manager is None and factory not initialized
    """
    global _factory_instance

    if _factory_instance is None:
        if connection_manager is None:
            raise ValueError("Connection manager required for factory initialization")
        _factory_instance = RepositoryFactory(connection_manager)

    return _factory_instance


def reset_repository_factory() -> None:
    """Reset the global repository factory instance (primarily for testing)."""
    global _factory_instance
    if _factory_instance:
        _factory_instance.cleanup()
    _factory_instance = None


# Convenience functions for direct repository access
async def get_analytics_repository(
    connection_manager: DatabaseServices | None = None,
) -> AnalyticsRepositoryProtocol:
    """Get analytics repository instance.

    Args:
        connection_manager: Connection manager (required on first call)

    Returns:
        AnalyticsRepositoryProtocol implementation
    """
    factory = get_repository_factory(connection_manager)
    return factory.get_analytics_repository()


async def get_apriori_repository(
    connection_manager: DatabaseServices | None = None,
) -> AprioriRepositoryProtocol:
    """Get apriori repository instance.

    Args:
        connection_manager: Connection manager (required on first call)

    Returns:
        AprioriRepositoryProtocol implementation
    """
    factory = get_repository_factory(connection_manager)
    return factory.get_apriori_repository()


async def get_ml_repository(
    connection_manager: DatabaseServices | None = None,
) -> MLRepositoryProtocol:
    """Get ML repository instance.

    Args:
        connection_manager: Connection manager (required on first call)

    Returns:
        MLRepositoryProtocol implementation
    """
    factory = get_repository_factory(connection_manager)
    return factory.get_ml_repository()


async def get_user_feedback_repository(
    connection_manager: DatabaseServices | None = None,
) -> UserFeedbackRepositoryProtocol:
    """Get user feedback repository instance.

    Args:
        connection_manager: Connection manager (required on first call)

    Returns:
        UserFeedbackRepositoryProtocol implementation
    """
    factory = get_repository_factory(connection_manager)
    return factory.get_user_feedback_repository()


async def get_health_repository(
    connection_manager: DatabaseServices | None = None,
) -> HealthRepositoryProtocol:
    """Get health repository instance.

    Args:
        connection_manager: Connection manager (required on first call)

    Returns:
        HealthRepositoryProtocol implementation
    """
    factory = get_repository_factory(connection_manager)
    return factory.get_health_repository()


# Dependency injection setup for FastAPI
async def setup_repository_dependencies(
    connection_manager: DatabaseServices,
) -> None:
    """Setup repository dependencies for the application.

    Args:
        connection_manager: Database connection manager
    """
    try:
        factory = get_repository_factory(connection_manager)
        factory.initialize()

        # Perform health check to ensure all repositories are working
        health_status = await factory.health_check()
        unhealthy_repos = [
            name for name, healthy in health_status.items() if not healthy
        ]

        if unhealthy_repos:
            logger.warning(f"Some repositories are unhealthy: {unhealthy_repos}")
        else:
            logger.info("All repositories initialized and healthy")

    except Exception as e:
        logger.exception(f"Repository dependency setup failed: {e}")
        raise


__all__ = [
    "RepositoryFactory",
    "get_analytics_repository",
    "get_apriori_repository",
    "get_health_repository",
    "get_ml_repository",
    "get_repository_factory",
    "get_user_feedback_repository",
    "reset_repository_factory",
    "setup_repository_dependencies",
]
