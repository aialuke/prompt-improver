"""Base repository protocol definitions.

Provides foundational interfaces for all repository implementations,
ensuring consistent patterns for database access, transaction management,
and query building across all domains.
"""

from abc import ABC, abstractmethod
try:
    from contextlib import AsyncContextManager
except ImportError:
    # Python 3.13+ compatibility
    from typing import AsyncContextManager
from typing import Any, Generic, Optional, Protocol, TypeVar, runtime_checkable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select
from sqlmodel import SQLModel

# Generic type for SQLModel entities
T = TypeVar("T", bound=SQLModel)
CreateT = TypeVar("CreateT")
UpdateT = TypeVar("UpdateT")


@runtime_checkable
class QueryBuilderProtocol(Protocol):
    """Protocol for building type-safe database queries."""

    def select(self, model_class: type[T]) -> Select[tuple[T]]:
        """Create a select query for the specified model."""
        ...

    def filter_by_id(self, query: Select[tuple[T]], entity_id: int) -> Select[tuple[T]]:
        """Add ID filter to query."""
        ...

    def filter_by_field(
        self, query: Select[tuple[T]], field_name: str, value: Any
    ) -> Select[tuple[T]]:
        """Add field filter to query."""
        ...

    def order_by(
        self, query: Select[tuple[T]], field_name: str, desc: bool = False
    ) -> Select[tuple[T]]:
        """Add ordering to query."""
        ...

    def limit_offset(
        self, query: Select[tuple[T]], limit: int, offset: int = 0
    ) -> Select[tuple[T]]:
        """Add pagination to query."""
        ...


@runtime_checkable
class TransactionManagerProtocol(Protocol):
    """Protocol for managing database transactions."""

    async def begin_transaction(self) -> AsyncContextManager[AsyncSession]:
        """Begin a new transaction."""
        ...

    async def commit(self, session: AsyncSession) -> None:
        """Commit the current transaction."""
        ...

    async def rollback(self, session: AsyncSession) -> None:
        """Rollback the current transaction."""
        ...

    async def savepoint(self, session: AsyncSession, name: str) -> None:
        """Create a savepoint within the transaction."""
        ...

    async def rollback_to_savepoint(self, session: AsyncSession, name: str) -> None:
        """Rollback to a specific savepoint."""
        ...


@runtime_checkable
class BaseRepositoryProtocol(Protocol[T]):
    """Base protocol for all repository implementations.

    Provides standard CRUD operations and common database patterns
    that all domain repositories should implement.
    """

    # Basic CRUD Operations
    async def create(self, entity_data: CreateT) -> T:
        """Create a new entity in the database."""
        ...

    async def get_by_id(self, entity_id: int) -> T | None:
        """Retrieve an entity by its ID."""
        ...

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[T]:
        """Retrieve all entities with pagination."""
        ...

    async def update(self, entity_id: int, update_data: UpdateT) -> T | None:
        """Update an entity by ID."""
        ...

    async def delete(self, entity_id: int) -> bool:
        """Delete an entity by ID."""
        ...

    # Query Operations
    async def exists(self, entity_id: int) -> bool:
        """Check if an entity exists."""
        ...

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count entities with optional filters."""
        ...

    async def find_by(self, **filters: Any) -> list[T]:
        """Find entities by field filters."""
        ...

    async def find_one_by(self, **filters: Any) -> T | None:
        """Find single entity by field filters."""
        ...

    # Batch Operations
    async def create_batch(self, entities_data: list[CreateT]) -> list[T]:
        """Create multiple entities in a single transaction."""
        ...

    async def update_batch(self, updates: list[tuple[int, UpdateT]]) -> list[T]:
        """Update multiple entities in a single transaction."""
        ...

    async def delete_batch(self, entity_ids: list[int]) -> int:
        """Delete multiple entities by IDs, returns count deleted."""
        ...

    # Health and Diagnostics
    async def health_check(self) -> dict[str, Any]:
        """Perform repository health check."""
        ...

    async def get_connection_info(self) -> dict[str, Any]:
        """Get database connection information."""
        ...


@runtime_checkable
class RepositoryFactoryProtocol(Protocol):
    """Protocol for repository factory that creates repository instances."""

    async def create_analytics_repository(self) -> "AnalyticsRepositoryProtocol":
        """Create analytics repository instance."""
        ...

    async def create_apriori_repository(self) -> "AprioriRepositoryProtocol":
        """Create Apriori repository instance."""
        ...

    async def create_ml_repository(self) -> "MLRepositoryProtocol":
        """Create ML repository instance."""
        ...

    async def create_rules_repository(self) -> "RulesRepositoryProtocol":
        """Create rules repository instance."""
        ...

    async def create_user_feedback_repository(self) -> "UserFeedbackRepositoryProtocol":
        """Create user feedback repository instance."""
        ...

    async def create_health_repository(self) -> "HealthRepositoryProtocol":
        """Create health repository instance."""
        ...

    async def create_transaction_manager(self) -> TransactionManagerProtocol:
        """Create transaction manager instance."""
        ...
