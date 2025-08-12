"""Base repository implementation with common database patterns.

Provides foundational repository functionality that all domain repositories
can inherit from, including CRUD operations, query building, transaction
management, and connection handling.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select
from sqlmodel import SQLModel

from prompt_improver.core.protocols.database_protocol import DatabaseSessionProtocol
from prompt_improver.database import DatabaseServices
from prompt_improver.repositories.protocols.base_repository_protocol import (
    BaseRepositoryProtocol,
    QueryBuilderProtocol,
    TransactionManagerProtocol,
)

logger = logging.getLogger(__name__)

# Generic types for repository operations
T = TypeVar("T", bound=SQLModel)
CreateT = TypeVar("CreateT")
UpdateT = TypeVar("UpdateT")


class QueryBuilder(QueryBuilderProtocol):
    """Type-safe query builder implementation."""

    def select(self, model_class: type[T]) -> Select[tuple[T]]:
        """Create a select query for the specified model."""
        return select(model_class)

    def filter_by_id(self, query: Select[tuple[T]], entity_id: int) -> Select[tuple[T]]:
        """Add ID filter to query."""
        # Extract the model from the query
        model_class = query.column_descriptions[0]["type"]
        return query.where(model_class.id == entity_id)

    def filter_by_field(
        self, query: Select[tuple[T]], field_name: str, value: Any
    ) -> Select[tuple[T]]:
        """Add field filter to query."""
        model_class = query.column_descriptions[0]["type"]
        field = getattr(model_class, field_name)
        return query.where(field == value)

    def order_by(
        self, query: Select[tuple[T]], field_name: str, desc: bool = False
    ) -> Select[tuple[T]]:
        """Add ordering to query."""
        model_class = query.column_descriptions[0]["type"]
        field = getattr(model_class, field_name)
        if desc:
            return query.order_by(field.desc())
        return query.order_by(field)

    def limit_offset(
        self, query: Select[tuple[T]], limit: int, offset: int = 0
    ) -> Select[tuple[T]]:
        """Add pagination to query."""
        return query.limit(limit).offset(offset)


class TransactionManager(TransactionManagerProtocol):
    """Transaction manager implementation."""

    def __init__(self, connection_manager: DatabaseServices):
        self.connection_manager = connection_manager

    @asynccontextmanager
    async def begin_transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """Begin a new transaction."""
        async with self.connection_manager.get_session() as session:
            async with session.begin():
                try:
                    yield session
                except Exception:
                    # SQLAlchemy will auto-rollback on exception within begin() context
                    raise

    async def commit(self, session: AsyncSession) -> None:
        """Commit the current transaction."""
        await session.commit()

    async def rollback(self, session: AsyncSession) -> None:
        """Rollback the current transaction."""
        await session.rollback()

    async def savepoint(self, session: AsyncSession, name: str) -> None:
        """Create a savepoint within the transaction."""
        savepoint = await session.begin_nested()
        # Store savepoint with name for later reference
        setattr(session, f"_savepoint_{name}", savepoint)

    async def rollback_to_savepoint(self, session: AsyncSession, name: str) -> None:
        """Rollback to a specific savepoint."""
        savepoint = getattr(session, f"_savepoint_{name}", None)
        if savepoint:
            await savepoint.rollback()


class BaseRepository(BaseRepositoryProtocol[T], Generic[T]):
    """Base repository implementation with common database operations."""

    def __init__(
        self,
        model_class: type[T],
        connection_manager: DatabaseServices,
        create_model_class: type[CreateT] | None = None,
        update_model_class: type[UpdateT] | None = None,
    ):
        self.model_class = model_class
        self.connection_manager = connection_manager
        self.create_model_class = create_model_class
        self.update_model_class = update_model_class
        self.query_builder = QueryBuilder()
        self.transaction_manager = TransactionManager(connection_manager)

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session context manager."""
        async with self.connection_manager.get_session() as session:
            yield session

    # Basic CRUD Operations

    async def create(self, entity_data: CreateT) -> T:
        """Create a new entity in the database."""
        async with self.get_session() as session:
            # Convert create data to model instance
            if isinstance(entity_data, dict):
                entity = self.model_class(**entity_data)
            else:
                entity = self.model_class(**entity_data.model_dump())

            session.add(entity)
            await session.commit()
            await session.refresh(entity)
            return entity

    async def get_by_id(self, entity_id: int) -> T | None:
        """Retrieve an entity by its ID."""
        async with self.get_session() as session:
            query = select(self.model_class).where(self.model_class.id == entity_id)
            result = await session.execute(query)
            return result.scalar_one_or_none()

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[T]:
        """Retrieve all entities with pagination."""
        async with self.get_session() as session:
            query = select(self.model_class).limit(limit).offset(offset)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def update(self, entity_id: int, update_data: UpdateT) -> T | None:
        """Update an entity by ID."""
        async with self.get_session() as session:
            # Get update data as dict
            if isinstance(update_data, dict):
                update_dict = update_data
            else:
                update_dict = update_data.model_dump(exclude_unset=True)

            # Update entity
            query = (
                update(self.model_class)
                .where(self.model_class.id == entity_id)
                .values(**update_dict)
            )
            result = await session.execute(query)

            if result.rowcount == 0:
                return None

            await session.commit()

            # Return updated entity
            return await self.get_by_id(entity_id)

    async def delete(self, entity_id: int) -> bool:
        """Delete an entity by ID."""
        async with self.get_session() as session:
            query = delete(self.model_class).where(self.model_class.id == entity_id)
            result = await session.execute(query)
            await session.commit()
            return result.rowcount > 0

    # Query Operations

    async def exists(self, entity_id: int) -> bool:
        """Check if an entity exists."""
        async with self.get_session() as session:
            query = select(func.count(self.model_class.id)).where(
                self.model_class.id == entity_id
            )
            result = await session.execute(query)
            return result.scalar() > 0

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count entities with optional filters."""
        async with self.get_session() as session:
            query = select(func.count(self.model_class.id))

            # Apply filters if provided
            if filters:
                for field_name, value in filters.items():
                    if hasattr(self.model_class, field_name):
                        field = getattr(self.model_class, field_name)
                        query = query.where(field == value)

            result = await session.execute(query)
            return result.scalar()

    async def find_by(self, **filters: Any) -> list[T]:
        """Find entities by field filters."""
        async with self.get_session() as session:
            query = select(self.model_class)

            # Apply filters
            for field_name, value in filters.items():
                if hasattr(self.model_class, field_name):
                    field = getattr(self.model_class, field_name)
                    query = query.where(field == value)

            result = await session.execute(query)
            return list(result.scalars().all())

    async def find_one_by(self, **filters: Any) -> T | None:
        """Find single entity by field filters."""
        entities = await self.find_by(**filters)
        return entities[0] if entities else None

    # Batch Operations

    async def create_batch(self, entities_data: list[CreateT]) -> list[T]:
        """Create multiple entities in a single transaction."""
        async with self.transaction_manager.begin_transaction() as session:
            entities = []

            for entity_data in entities_data:
                if isinstance(entity_data, dict):
                    entity = self.model_class(**entity_data)
                else:
                    entity = self.model_class(**entity_data.model_dump())
                entities.append(entity)
                session.add(entity)

            await session.commit()

            # Refresh all entities to get IDs
            for entity in entities:
                await session.refresh(entity)

            return entities

    async def update_batch(self, updates: list[tuple[int, UpdateT]]) -> list[T]:
        """Update multiple entities in a single transaction."""
        async with self.transaction_manager.begin_transaction() as session:
            updated_entities = []

            for entity_id, update_data in updates:
                # Get update data as dict
                if isinstance(update_data, dict):
                    update_dict = update_data
                else:
                    update_dict = update_data.model_dump(exclude_unset=True)

                # Update entity
                query = (
                    update(self.model_class)
                    .where(self.model_class.id == entity_id)
                    .values(**update_dict)
                )
                await session.execute(query)

            await session.commit()

            # Get updated entities
            for entity_id, _ in updates:
                entity = await self.get_by_id(entity_id)
                if entity:
                    updated_entities.append(entity)

            return updated_entities

    async def delete_batch(self, entity_ids: list[int]) -> int:
        """Delete multiple entities by IDs, returns count deleted."""
        async with self.transaction_manager.begin_transaction() as session:
            query = delete(self.model_class).where(self.model_class.id.in_(entity_ids))
            result = await session.execute(query)
            await session.commit()
            return result.rowcount

    # Health and Diagnostics

    async def health_check(self) -> dict[str, Any]:
        """Perform repository health check."""
        try:
            async with self.get_session() as session:
                # Simple query to test connection
                query = select(func.count(self.model_class.id))
                result = await session.execute(query)
                count = result.scalar()

                return {
                    "status": "healthy",
                    "model": self.model_class.__name__,
                    "total_records": count,
                    "connection_status": "active",
                }
        except Exception as e:
            logger.error(f"Health check failed for {self.model_class.__name__}: {e}")
            return {
                "status": "unhealthy",
                "model": self.model_class.__name__,
                "error": str(e),
                "connection_status": "failed",
            }

    async def get_connection_info(self) -> dict[str, Any]:
        """Get database connection information."""
        try:
            connection_info = await self.connection_manager.get_connection_info()
            return {
                "model": self.model_class.__name__,
                "connection_info": connection_info,
                "status": "connected",
            }
        except Exception as e:
            return {
                "model": self.model_class.__name__,
                "error": str(e),
                "status": "error",
            }
