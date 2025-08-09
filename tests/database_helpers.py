"""Database test helpers following pytest-postgresql best practices."""
import asyncio
import logging
from typing import Any, Optional
import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel
from prompt_improver.database.unified_connection_manager import ManagerMode, get_unified_manager
logger = logging.getLogger(__name__)

async def wait_for_postgres_async(host: str, port: int, user: str, password: str, database: str, max_retries: int=30, retry_delay: float=1.0) -> bool:
    """Wait for PostgreSQL to be ready with async connection.

    Best practice: Always verify database is ready before running tests.
    2025 enhancement: Use main database instead of 'postgres' for Docker compatibility.
    """
    try:
        manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
        health_info = await manager.get_health_info()
        if health_info.get('status') == 'healthy':
            logger.info('PostgreSQL ready via UnifiedConnectionManager (database: %s)', database)
            return True
    except Exception as e:
        logger.debug('UnifiedConnectionManager health check failed, falling back to direct connection: %s', e)
    for attempt in range(max_retries):
        try:
            conn = await asyncpg.connect(host=host, port=port, user=user, password=password, database=database, timeout=5.0)
            await conn.close()
            logger.info('PostgreSQL ready after %s attempts (connected to %s)', attempt + 1, database)
            return True
        except (asyncpg.PostgresError, OSError) as e:
            if attempt == max_retries - 1:
                logger.error('PostgreSQL not ready after {max_retries} attempts: %s', e)
                return False
            await asyncio.sleep(retry_delay)
    return False

def wait_for_postgres_sync(host: str, port: int, user: str, password: str, database: str, max_retries: int=30, retry_delay: float=1.0) -> bool:
    """Wait for PostgreSQL to be ready using async in sync context.

    Used for initial setup and verification.
    """
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(wait_for_postgres_async(host, port, user, password, database, max_retries, retry_delay))
        loop.close()
        return result
    except Exception as e:
        logger.error('Failed to check PostgreSQL readiness: %s', e)
        return False

async def ensure_test_database_exists(host: str, port: int, user: str, password: str, test_db_name: str='apes_test') -> bool:
    """Ensure test database exists, create if necessary.

    Best practice: Use template database for faster test setup.
    """
    try:
        admin_database = 'apes_production'
        try:
            manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
            async with manager.get_async_session() as session:
                result = await session.execute(text('SELECT 1 FROM pg_database WHERE datname = :db_name'), {'db_name': test_db_name})
                exists = result.scalar()
                if exists:
                    logger.info('Test database %s already exists (verified via UnifiedConnectionManager)', test_db_name)
                    return None
        except Exception as e:
            logger.debug('UnifiedConnectionManager database check failed, using direct connection: %s', e)
        conn = await asyncpg.connect(host=host, port=port, user=user, password=password, database=admin_database, timeout=5.0)
        exists = await conn.fetchval('SELECT 1 FROM pg_database WHERE datname = $1', test_db_name)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{test_db_name}"')
            logger.info('Created test database: %s', test_db_name)
        await conn.close()
        return True
    except Exception as e:
        logger.error('Failed to ensure test database exists: %s', e)
        return False

async def cleanup_test_database(host: str, port: int, user: str, password: str, test_db_name: str='apes_test') -> bool:
    """Clean up test database by dropping and recreating it.

    2025 best practice: Ensure completely clean state between test runs
    by recreating the entire database with proper connection handling.
    Uses Docker exec fallback for DDL operations due to pg_hba.conf restrictions.
    """
    conn = None
    try:
        admin_database = 'apes_production'
        try:
            manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
            async with manager.get_async_session() as session:
                result = await session.execute(text('SELECT 1 FROM pg_database WHERE datname = :db_name'), {'db_name': test_db_name})
                exists = result.scalar()
                if not exists:
                    logger.info('Test database %s does not exist (verified via UnifiedConnectionManager)', test_db_name)
                    return None
        except Exception as e:
            logger.debug('UnifiedConnectionManager verification failed, proceeding with direct connection: %s', e)
        conn = await asyncpg.connect(host=host, port=port, user=user, password=password, database=admin_database, timeout=10.0)
        await conn.execute('SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1 AND pid <> pg_backend_pid()', test_db_name)
        await conn.execute(f'DROP DATABASE IF EXISTS "{test_db_name}"')
        logger.info('Dropped test database: %s', test_db_name)
        await asyncio.sleep(0.2)
        await conn.execute(f"""CREATE DATABASE "{test_db_name}" WITH TEMPLATE=template0 ENCODING='UTF8'""")
        logger.info('Created fresh test database: %s', test_db_name)
        return True
    except Exception as e:
        logger.warning('asyncpg cleanup failed: %s. Trying Docker exec fallback...', e)
        return await _cleanup_test_database_docker_fallback(test_db_name, user)
    finally:
        if conn:
            await conn.close()

async def _cleanup_test_database_docker_fallback(test_db_name: str, user: str) -> bool:
    """Fallback cleanup using Docker exec for DDL operations.

    This handles cases where asyncpg connections have permission issues
    due to pg_hba.conf authentication restrictions.
    """
    import subprocess
    try:
        terminate_result = subprocess.run(['docker', 'exec', 'apes_postgres', 'psql', '-U', user, '-d', 'apes_production', '-c', f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{test_db_name}' AND pid <> pg_backend_pid();"], check=False, capture_output=True, text=True)
        logger.info('Connection termination result: %s', terminate_result.returncode)
        drop_result = subprocess.run(['docker', 'exec', 'apes_postgres', 'psql', '-U', user, '-d', 'apes_production', '-c', f'DROP DATABASE IF EXISTS "{test_db_name}";'], check=False, capture_output=True, text=True)
        if drop_result.returncode == 0:
            logger.info('Successfully dropped test database: %s', test_db_name)
        else:
            logger.warning('Drop database warning: %s', drop_result.stderr)
        await asyncio.sleep(0.2)
        create_result = subprocess.run(['docker', 'exec', 'apes_postgres', 'psql', '-U', user, '-d', 'apes_production', '-c', f'''CREATE DATABASE "{test_db_name}" WITH TEMPLATE=template0 ENCODING='UTF8';'''], check=False, capture_output=True, text=True)
        if create_result.returncode == 0:
            logger.info('Successfully created fresh test database: %s', test_db_name)
            return True
        logger.error('Failed to create test database: %s', create_result.stderr)
        logger.error('Create command stdout: %s', create_result.stdout)
        return False
    except Exception as e:
        logger.error('Unexpected error in Docker exec cleanup: %s', e)
        return False

async def reflection_based_cleanup(conn):
    """Use SQLAlchemy reflection to discover and drop existing schema objects.

    2025 best practice: Use SQLAlchemy's Inspector to dynamically discover
    existing schema objects instead of hardcoding names.
    Enhanced to handle index conflicts properly.
    """
    try:
        from sqlalchemy import MetaData, text
        from sqlalchemy.engine import reflection
        reflected_metadata = MetaData()
        await conn.run_sync(reflected_metadata.reflect)
        if reflected_metadata.tables:
            logger.info('Dropping %s existing tables with CASCADE', len(reflected_metadata.tables))
            await conn.run_sync(reflected_metadata.drop_all)
        try:
            index_query = text("\n                SELECT indexname FROM pg_indexes\n                WHERE schemaname = 'public'\n                AND indexname LIKE 'ix_%'\n            ")
            result = await conn.execute(index_query)
            indexes = [row[0] for row in result.fetchall()]
            for index_name in indexes:
                try:
                    await conn.execute(text(f'DROP INDEX IF EXISTS "{index_name}" CASCADE'))
                    logger.debug('Dropped index: %s', index_name)
                except Exception as idx_error:
                    logger.debug('Failed to drop index {index_name}: %s', idx_error)
            if indexes:
                logger.info('Cleaned up %s remaining indexes', len(indexes))
        except Exception as index_cleanup_error:
            logger.debug('Index cleanup failed: %s', index_cleanup_error)
    except Exception as e:
        logger.debug('Reflection-based cleanup failed: %s', e)
        try:
            await conn.execute(text('DROP SCHEMA IF EXISTS public CASCADE'))
            await conn.execute(text('CREATE SCHEMA public'))
            await conn.execute(text('GRANT ALL ON SCHEMA public TO public'))
            logger.info('Successfully recreated public schema')
        except Exception as schema_error:
            logger.warning('Schema recreation failed: %s', schema_error)

async def create_test_engine_with_retry(db_url: str, max_retries: int=3, **engine_kwargs: Any) -> AsyncEngine | None:
    """Create SQLAlchemy async engine with retry logic.

    2025 best practice: Use reflection-based cleanup and proper transaction isolation.
    """
    for attempt in range(max_retries):
        try:
            engine = create_async_engine(db_url, echo=False, future=True, pool_pre_ping=True, **engine_kwargs)
            try:
                async with engine.connect() as cleanup_conn:
                    async with cleanup_conn.begin():
                        await reflection_based_cleanup(cleanup_conn)
                await asyncio.sleep(0.1)
                async with engine.connect() as create_conn:
                    async with create_conn.begin():
                        try:
                            await create_conn.run_sync(SQLModel.metadata.create_all)
                            logger.info('Successfully created all database tables and indexes')
                        except Exception as create_error:
                            if 'already exists' in str(create_error).lower():
                                logger.info('Handling existing indexes - creating tables individually')
                                await _create_tables_individually(create_conn)
                            else:
                                raise
            except Exception as table_error:
                logger.warning('Schema setup issue (attempt %s): %s', attempt + 1, table_error)
                if any((keyword in str(table_error).lower() for keyword in ['already exists', 'duplicate', 'aborted', 'relation', 'index'])):
                    if attempt < max_retries - 1:
                        logger.info('Retrying after transient error: %s', table_error)
                        await engine.dispose()
                        await asyncio.sleep(1 * (attempt + 1))
                        continue
                    logger.warning('Final attempt: using nuclear schema cleanup')
                    try:
                        async with engine.connect() as nuclear_conn:
                            async with nuclear_conn.begin():
                                from sqlalchemy import text
                                await nuclear_conn.execute(text('DROP SCHEMA IF EXISTS public CASCADE'))
                                await nuclear_conn.execute(text('CREATE SCHEMA public'))
                                await nuclear_conn.execute(text('GRANT ALL ON SCHEMA public TO public'))
                        async with engine.connect() as create_conn:
                            async with create_conn.begin():
                                await create_conn.run_sync(SQLModel.metadata.create_all)
                                logger.info('Successfully created tables after nuclear cleanup')
                    except Exception as nuclear_error:
                        logger.error('Nuclear cleanup failed: %s', nuclear_error)
                        raise table_error
                else:
                    raise
            return engine
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error('Failed to create engine after %s attempts: %s', max_retries, e)
                raise
            await asyncio.sleep(1 * (attempt + 1))
    return None

async def _create_tables_individually(conn):
    """Create tables individually, handling index conflicts gracefully.

    2025 best practice: Handle schema conflicts by creating objects individually
    and catching/ignoring specific "already exists" errors for indexes.
    """
    from sqlalchemy import text
    from sqlmodel import SQLModel
    tables = SQLModel.metadata.sorted_tables
    for table in tables:
        try:
            table.create(conn.sync_connection, checkfirst=True)
            logger.debug('Created table: %s', table.name)
            for index in table.indexes:
                try:
                    index_sql = str(index.create(conn.sync_connection).compile())
                    if 'CREATE UNIQUE INDEX' in index_sql:
                        index_name = index.name
                        modified_sql = f'''CREATE UNIQUE INDEX IF NOT EXISTS "{index_name}" ON {table.name} ({', '.join((col.name for col in index.columns))})'''
                        await conn.execute(text(modified_sql))
                    elif 'CREATE INDEX' in index_sql:
                        index_name = index.name
                        modified_sql = f'''CREATE INDEX IF NOT EXISTS "{index_name}" ON {table.name} ({', '.join((col.name for col in index.columns))})'''
                        await conn.execute(text(modified_sql))
                    else:
                        try:
                            index.create(conn.sync_connection, checkfirst=True)
                        except Exception as idx_error:
                            if 'already exists' not in str(idx_error).lower():
                                raise
                    logger.debug('Created index: %s', index.name)
                except Exception as idx_error:
                    if 'already exists' in str(idx_error).lower():
                        logger.debug('Index %s already exists, skipping', index.name)
                    else:
                        logger.warning('Failed to create index %s: %s', index.name, idx_error)
        except Exception as table_error:
            if 'already exists' in str(table_error).lower():
                logger.debug('Table %s already exists, skipping', table.name)
            else:
                logger.error('Failed to create table {table.name}: %s', table_error)
                raise
    logger.info('Successfully created all tables and indexes individually')
