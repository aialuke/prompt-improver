"""Database test helpers following pytest-postgresql best practices."""

import asyncio
import logging
from typing import Any, Optional

import asyncpg
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy import text
from sqlmodel import SQLModel

# Import UnifiedConnectionManager for health checks
from prompt_improver.database.unified_connection_manager import (
    get_unified_manager, ManagerMode
)

logger = logging.getLogger(__name__)


async def wait_for_postgres_async(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    max_retries: int = 30,
    retry_delay: float = 1.0,
) -> bool:
    """Wait for PostgreSQL to be ready with async connection.

    Best practice: Always verify database is ready before running tests.
    2025 enhancement: Use main database instead of 'postgres' for Docker compatibility.
    """
    # Use UnifiedConnectionManager for health checks when possible
    try:
        manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
        health_info = await manager.get_health_info()
        if health_info.get('status') == 'healthy':
            logger.info(f"PostgreSQL ready via UnifiedConnectionManager (database: {database})")
            return True
    except Exception as e:
        logger.debug(f"UnifiedConnectionManager health check failed, falling back to direct connection: {e}")
    
    # Fallback to direct connection for compatibility with test infrastructure
    for attempt in range(max_retries):
        try:
            # Try to connect directly with asyncpg for test infrastructure compatibility
            conn = await asyncpg.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                timeout=5.0,
            )
            await conn.close()
            logger.info(f"PostgreSQL ready after {attempt + 1} attempts (connected to {database})")
            return True
        except (asyncpg.PostgresError, OSError) as e:
            if attempt == max_retries - 1:
                logger.error(f"PostgreSQL not ready after {max_retries} attempts: {e}")
                return False
            await asyncio.sleep(retry_delay)
    return False


def wait_for_postgres_sync(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    max_retries: int = 30,
    retry_delay: float = 1.0,
) -> bool:
    """Wait for PostgreSQL to be ready using async in sync context.

    Used for initial setup and verification.
    """
    try:
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            wait_for_postgres_async(
                host, port, user, password, database, max_retries, retry_delay
            )
        )
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Failed to check PostgreSQL readiness: {e}")
        return False


async def ensure_test_database_exists(
    host: str, port: int, user: str, password: str, test_db_name: str = "apes_test"
) -> bool:
    """Ensure test database exists, create if necessary.

    Best practice: Use template database for faster test setup.
    """
    try:
        # Connect to main database (apes_production in Docker setup)
        # instead of 'postgres' which doesn't exist in our Docker container
        admin_database = "apes_production"
        # Try UnifiedConnectionManager for database existence check first
        try:
            manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
            async with manager.get_async_session() as session:
                # Check if test database exists using unified manager
                result = await session.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                    {"db_name": test_db_name}
                )
                exists = result.scalar()
                
                if exists:
                    logger.info(f"Test database {test_db_name} already exists (verified via UnifiedConnectionManager)")
                    return
        except Exception as e:
            logger.debug(f"UnifiedConnectionManager database check failed, using direct connection: {e}")
        
        # Use direct connection for DDL operations (CREATE DATABASE requires special privileges)
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=admin_database,
            timeout=5.0,
        )

        # Check if test database exists
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", test_db_name
        )

        if not exists:
            # Create test database
            await conn.execute(f'CREATE DATABASE "{test_db_name}"')
            logger.info(f"Created test database: {test_db_name}")

        await conn.close()
        return True

    except Exception as e:
        logger.error(f"Failed to ensure test database exists: {e}")
        return False


async def cleanup_test_database(
    host: str, port: int, user: str, password: str, test_db_name: str = "apes_test"
) -> bool:
    """Clean up test database by dropping and recreating it.
    
    2025 best practice: Ensure completely clean state between test runs
    by recreating the entire database with proper connection handling.
    Uses Docker exec fallback for DDL operations due to pg_hba.conf restrictions.
    """
    # First try asyncpg approach
    conn = None
    try:
        # Connect to the main database (apes_production in Docker setup)
        admin_database = "apes_production"
        # Try to verify database existence using UnifiedConnectionManager first
        try:
            manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
            async with manager.get_async_session() as session:
                result = await session.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                    {"db_name": test_db_name}
                )
                exists = result.scalar()
                if not exists:
                    logger.info(f"Test database {test_db_name} does not exist (verified via UnifiedConnectionManager)")
                    return
        except Exception as e:
            logger.debug(f"UnifiedConnectionManager verification failed, proceeding with direct connection: {e}")
        
        # Use direct connection for DDL operations (DROP DATABASE requires special privileges)
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=admin_database,
            timeout=10.0,
        )

        # Terminate active connections to test database
        await conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1 AND pid <> pg_backend_pid()",
            test_db_name
        )

        # Drop test database if it exists
        await conn.execute(f'DROP DATABASE IF EXISTS "{test_db_name}"')
        logger.info(f"Dropped test database: {test_db_name}")
        
        # Brief pause to ensure PostgreSQL processes the drop
        await asyncio.sleep(0.2)
        
        # Create fresh test database
        await conn.execute(
            f'CREATE DATABASE "{test_db_name}" WITH TEMPLATE=template0 ENCODING=\'UTF8\''
        )
        logger.info(f"Created fresh test database: {test_db_name}")

        return True

    except Exception as e:
        logger.warning(f"asyncpg cleanup failed: {e}. Trying Docker exec fallback...")
        
        # Fallback to Docker exec approach for DDL operations
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
        # Terminate active connections
        terminate_result = subprocess.run([
            "docker", "exec", "apes_postgres", "psql", "-U", user, "-d", "apes_production",
            "-c", f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{test_db_name}' AND pid <> pg_backend_pid();"
        ], check=False, capture_output=True, text=True)
        
        logger.info(f"Connection termination result: {terminate_result.returncode}")
        
        # Drop test database
        drop_result = subprocess.run([
            "docker", "exec", "apes_postgres", "psql", "-U", user, "-d", "apes_production",
            "-c", f'DROP DATABASE IF EXISTS "{test_db_name}";'
        ], check=False, capture_output=True, text=True)
        
        if drop_result.returncode == 0:
            logger.info(f"Successfully dropped test database: {test_db_name}")
        else:
            logger.warning(f"Drop database warning: {drop_result.stderr}")
        
        # Brief pause
        await asyncio.sleep(0.2)
        
        # Create fresh test database
        create_result = subprocess.run([
            "docker", "exec", "apes_postgres", "psql", "-U", user, "-d", "apes_production",
            "-c", f'CREATE DATABASE "{test_db_name}" WITH TEMPLATE=template0 ENCODING=\'UTF8\';'
        ], check=False, capture_output=True, text=True)
        
        if create_result.returncode == 0:
            logger.info(f"Successfully created fresh test database: {test_db_name}")
            return True
        else:
            logger.error(f"Failed to create test database: {create_result.stderr}")
            logger.error(f"Create command stdout: {create_result.stdout}")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error in Docker exec cleanup: {e}")
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
        
        # Create a separate metadata object for reflection
        reflected_metadata = MetaData()
        
        # Reflect existing tables to get all schema objects
        await conn.run_sync(reflected_metadata.reflect)
        
        # Drop all reflected tables with CASCADE to handle dependencies and indexes
        if reflected_metadata.tables:
            logger.info(f"Dropping {len(reflected_metadata.tables)} existing tables with CASCADE")
            await conn.run_sync(reflected_metadata.drop_all)
            
        # Additional cleanup: Drop any remaining indexes that might cause conflicts
        try:
            # Get list of all indexes in the public schema
            index_query = text("""
                SELECT indexname FROM pg_indexes
                WHERE schemaname = 'public'
                AND indexname LIKE 'ix_%'
            """)
            result = await conn.execute(index_query)
            indexes = [row[0] for row in result.fetchall()]
            
            # Drop each index individually
            for index_name in indexes:
                try:
                    await conn.execute(text(f'DROP INDEX IF EXISTS "{index_name}" CASCADE'))
                    logger.debug(f"Dropped index: {index_name}")
                except Exception as idx_error:
                    logger.debug(f"Failed to drop index {index_name}: {idx_error}")
                    
            if indexes:
                logger.info(f"Cleaned up {len(indexes)} remaining indexes")
                
        except Exception as index_cleanup_error:
            logger.debug(f"Index cleanup failed: {index_cleanup_error}")
            
    except Exception as e:
        logger.debug(f"Reflection-based cleanup failed: {e}")
        
        # Fallback: Try to drop the entire schema and recreate it
        try:
            await conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
            await conn.execute(text("GRANT ALL ON SCHEMA public TO public"))
            logger.info("Successfully recreated public schema")
        except Exception as schema_error:
            logger.warning(f"Schema recreation failed: {schema_error}")


async def create_test_engine_with_retry(
    db_url: str, max_retries: int = 3, **engine_kwargs: Any
) -> AsyncEngine | None:
    """Create SQLAlchemy async engine with retry logic.

    2025 best practice: Use reflection-based cleanup and proper transaction isolation.
    """
    for attempt in range(max_retries):
        try:
            engine = create_async_engine(
                db_url, echo=False, future=True, pool_pre_ping=True, **engine_kwargs
            )

            # Use separate connections for cleanup and creation to avoid transaction conflicts
            try:
                # First connection: Clean up existing schema objects
                async with engine.connect() as cleanup_conn:
                    async with cleanup_conn.begin():
                        await reflection_based_cleanup(cleanup_conn)
                
                # Brief pause to ensure PostgreSQL processes the cleanup
                await asyncio.sleep(0.1)
                
                # Second connection: Create new schema objects with error handling
                async with engine.connect() as create_conn:
                    async with create_conn.begin():
                        # Try to create tables individually to handle index conflicts
                        try:
                            await create_conn.run_sync(SQLModel.metadata.create_all)
                            logger.info("Successfully created all database tables and indexes")
                        except Exception as create_error:
                            # Handle index conflicts by creating tables individually
                            if "already exists" in str(create_error).lower():
                                logger.info("Handling existing indexes - creating tables individually")
                                await _create_tables_individually(create_conn)
                            else:
                                raise
                    
            except Exception as table_error:
                logger.warning(f"Schema setup issue (attempt {attempt + 1}): {table_error}")
                
                # Check if this is a known transient error
                if any(keyword in str(table_error).lower() for keyword in [
                    "already exists", "duplicate", "aborted", "relation", "index"
                ]):
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying after transient error: {table_error}")
                        await engine.dispose()
                        await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        # Final attempt - try nuclear option
                        logger.warning("Final attempt: using nuclear schema cleanup")
                        try:
                            async with engine.connect() as nuclear_conn:
                                async with nuclear_conn.begin():
                                    from sqlalchemy import text
                                    await nuclear_conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
                                    await nuclear_conn.execute(text("CREATE SCHEMA public"))
                                    await nuclear_conn.execute(text("GRANT ALL ON SCHEMA public TO public"))
                                    
                            # Try creating tables again after nuclear cleanup
                            async with engine.connect() as create_conn:
                                async with create_conn.begin():
                                    await create_conn.run_sync(SQLModel.metadata.create_all)
                                    logger.info("Successfully created tables after nuclear cleanup")
                        except Exception as nuclear_error:
                            logger.error(f"Nuclear cleanup failed: {nuclear_error}")
                            raise table_error
                else:
                    raise

            return engine

        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(
                    f"Failed to create engine after {max_retries} attempts: {e}"
                )
                raise
            await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

    return None


async def _create_tables_individually(conn):
    """Create tables individually, handling index conflicts gracefully.
    
    2025 best practice: Handle schema conflicts by creating objects individually
    and catching/ignoring specific "already exists" errors for indexes.
    """
    from sqlalchemy import text
    from sqlmodel import SQLModel
    
    # Get all table objects from metadata
    tables = SQLModel.metadata.sorted_tables
    
    for table in tables:
        try:
            # Create table without indexes first
            table.create(conn.sync_connection, checkfirst=True)
            logger.debug(f"Created table: {table.name}")
            
            # Create indexes individually with IF NOT EXISTS handling
            for index in table.indexes:
                try:
                    index_sql = str(index.create(conn.sync_connection).compile())
                    # Add IF NOT EXISTS for PostgreSQL
                    if "CREATE UNIQUE INDEX" in index_sql:
                        index_name = index.name
                        modified_sql = f'CREATE UNIQUE INDEX IF NOT EXISTS "{index_name}" ON {table.name} ({", ".join(col.name for col in index.columns)})'
                        await conn.execute(text(modified_sql))
                    elif "CREATE INDEX" in index_sql:
                        index_name = index.name
                        modified_sql = f'CREATE INDEX IF NOT EXISTS "{index_name}" ON {table.name} ({", ".join(col.name for col in index.columns)})'
                        await conn.execute(text(modified_sql))
                    else:
                        # Fallback - try original approach but catch conflicts
                        try:
                            index.create(conn.sync_connection, checkfirst=True)
                        except Exception as idx_error:
                            if "already exists" not in str(idx_error).lower():
                                raise
                    logger.debug(f"Created index: {index.name}")
                except Exception as idx_error:
                    if "already exists" in str(idx_error).lower():
                        logger.debug(f"Index {index.name} already exists, skipping")
                    else:
                        logger.warning(f"Failed to create index {index.name}: {idx_error}")
                        
        except Exception as table_error:
            if "already exists" in str(table_error).lower():
                logger.debug(f"Table {table.name} already exists, skipping")
            else:
                logger.error(f"Failed to create table {table.name}: {table_error}")
                raise
    
    logger.info("Successfully created all tables and indexes individually")
