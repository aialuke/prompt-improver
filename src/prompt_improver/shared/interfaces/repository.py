"""Repository interfaces following clean architecture patterns."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class IRepository(Generic[T], ABC):
    """Generic repository interface following repository pattern.

    Provides abstraction over data persistence to enable testing
    and switching between different storage implementations.
    """

    @abstractmethod
    async def get_by_id(self, id: str) -> T | None:
        """Get entity by ID.

        Args:
            id: Entity identifier

        Returns:
            Entity if found, None otherwise
        """
        ...

    @abstractmethod
    async def save(self, entity: T) -> T:
        """Save entity.

        Args:
            entity: Entity to save

        Returns:
            Saved entity with updated fields
        """
        ...

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete entity by ID.

        Args:
            id: Entity identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    @abstractmethod
    async def list_all(self, limit: int | None = None, offset: int = 0) -> list[T]:
        """List all entities with pagination.

        Args:
            limit: Maximum number of entities to return
            offset: Number of entities to skip

        Returns:
            List of entities
        """
        ...


class IPromptRepository(IRepository, ABC):
    """Repository interface specifically for prompts.

    Extends generic repository with prompt-specific operations.
    """

    @abstractmethod
    async def find_by_user(self, user_id: str) -> list[Any]:
        """Find prompts by user ID.

        Args:
            user_id: User identifier

        Returns:
            List of prompts for the user
        """
        ...

    @abstractmethod
    async def find_by_content_hash(self, content_hash: str) -> Any | None:
        """Find prompt by content hash for deduplication.

        Args:
            content_hash: Hash of prompt content

        Returns:
            Existing prompt with same content, if any
        """
        ...

    @abstractmethod
    async def search_by_content(self, search_term: str, limit: int = 10) -> list[Any]:
        """Search prompts by content.

        Args:
            search_term: Term to search for in prompt content
            limit: Maximum number of results

        Returns:
            List of matching prompts
        """
        ...


class ISessionRepository(IRepository, ABC):
    """Repository interface for improvement sessions."""

    @abstractmethod
    async def find_by_user(self, user_id: str) -> list[Any]:
        """Find sessions by user ID.

        Args:
            user_id: User identifier

        Returns:
            List of sessions for the user
        """
        ...

    @abstractmethod
    async def find_active_sessions(self) -> list[Any]:
        """Find currently active sessions.

        Returns:
            List of active sessions
        """
        ...


class IMetricsRepository(IRepository, ABC):
    """Repository interface for metrics and analytics data."""

    @abstractmethod
    async def save_metric(
        self, metric_name: str, value: float, tags: dict[str, str]
    ) -> None:
        """Save a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Tags for metric categorization
        """
        ...

    @abstractmethod
    async def get_metrics_by_name(
        self,
        metric_name: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get metrics by name within time range.

        Args:
            metric_name: Name of metric to retrieve
            start_time: Start time for filtering (ISO format)
            end_time: End time for filtering (ISO format)

        Returns:
            List of metric data points
        """
        ...
