from typing import Type, TypeVar, cast

from sqlalchemy import Executable
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")


async def fetch_scalar(session: AsyncSession, stmt: Executable, typ: type[T]) -> T:
    """Execute a statement and return a scalar result of a specified type.

    Args:
        session: The async database session
        stmt: The SQLAlchemy Executable statement (text() or select())
        typ: The type to cast the result into

    Returns:
        The scalar result from the query cast to the specified type
    """
    result = await session.execute(stmt)
    scalar_result = result.scalar()
    if scalar_result is not None:
        return cast(T, scalar_result)
    raise ValueError("No result fetched")
