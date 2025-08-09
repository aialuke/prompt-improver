import asyncio
import pytest
from sqlalchemy import text
from prompt_improver.database.unified_connection_manager import ManagerMode, get_unified_manager

@pytest.mark.asyncio
async def test_async_db_interaction():
    db_url = 'postgresql+asyncpg://user:password@localhost/test_async_db'
    db_manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
    async with db_manager.session() as session:
        result = await session.execute(text('SELECT version();'))
        version = result.fetchone()
        assert version is not None
        assert 'postgresql' in db_url
