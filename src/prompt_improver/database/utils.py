"""Database utilities for SQLAlchemy 2.0 Core statements.

This module provides helper functions for common database operations
following SQLAlchemy 2.0 best practices with precise type narrowing.
"""

from typing import Any, TypeVar

from sqlalchemy import Executable
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")

async def scalar(session: AsyncSession, stmt: Executable) -> Any:
    """Execute a statement and return a scalar result.

    This helper function provides a concise pattern for executing
    SQLAlchemy Core statements that return a single scalar value.

    Args:
        session: The async database session
        stmt: The SQLAlchemy Executable statement (text() or select())

    Returns:
        The scalar result from the query

    Example:
        ```python
        from sqlalchemy import text, select
        from .utils import scalar

        # Using text() for raw SQL
        result = await scalar(session, text("SELECT 1"))

        # Using model-based select()
        result = await scalar(session, select(func.count(Model.id)))
        ```
    """
    result = await session.execute(stmt)
    return result.scalar()

async def scalar_with_type(session: AsyncSession, stmt: Executable) -> Row[tuple[int]] | None:
    """Execute a statement and return a typed scalar result.

    Provides precise type narrowing for count queries and similar scalar operations.

    Args:
        session: The async database session
        stmt: The SQLAlchemy Executable statement

    Returns:
        Row with tuple typing or None if no result

    Example:
        ```python
        from sqlalchemy import text, func, select
        from .utils import scalar_with_type

        # Count query with precise typing
        row: Row[tuple[int]] | None = await scalar_with_type(
            session,
            select(func.count(Model.id))
        )
        if row is None:
            count = 0
        else:
            count: int = row[0]
        ```
    """
    result = await session.execute(stmt)
    return result.first()

async def fetch_one_row(session: AsyncSession, stmt: Executable) -> Row[Any] | None:
    """Execute a statement and return a single row with type narrowing.

    Provides precise type narrowing for single row queries with named columns.

    Args:
        session: The async database session
        stmt: The SQLAlchemy Executable statement

    Returns:
        Row with Any typing or None if no result

    Example:
        ```python
        from sqlalchemy import text
        from .utils import fetch_one_row

        # Query with named columns
        row: Row[Any] | None = await fetch_one_row(
            session,
            text("SELECT COUNT(*) as high_quality_count FROM rule_performance WHERE improvement_score > 0.8")
        )
        if row is None:
            count = 0
        else:
            count: int = row.high_quality_count
        ```
    """
    result = await session.execute(stmt)
    return result.first()

async def fetch_all_rows(session: AsyncSession, stmt: Executable, parameters: dict[str, Any] | None = None) -> list[Row[Any]]:
    """Execute a statement and return all rows with type narrowing.

    Provides precise type narrowing for multi-row queries.

    Args:
        session: The async database session
        stmt: The SQLAlchemy Executable statement
        parameters: Optional parameters for the query

    Returns:
        List of Row objects with Any typing

    Example:
        ```python
        from sqlalchemy import text
        from .utils import fetch_all_rows

        # Query multiple rows with named columns
        rows: list[Row[Any]] = await fetch_all_rows(
            session,
            text("SELECT rule_id, AVG(improvement_score) as avg_score FROM rule_performance GROUP BY rule_id"),
            {"cutoff_date": cutoff_date}
        )
        for row in rows:
            rule_id: str = row.rule_id
            avg_score: float = row.avg_score
        ```
    """
    result = await session.execute(stmt, parameters or {})
    return result.fetchall()
