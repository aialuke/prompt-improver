"""Core connection pool functionality for PostgreSQL.

Handles basic connection management, session creation, and query execution.
Extracted from PostgreSQLPoolManager following single responsibility principle.
"""

import contextlib
import hashlib
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from prompt_improver.database.services.connection.pool_shared_context import (
    ConnectionMode,
    PoolSharedContext,
)

logger = logging.getLogger(__name__)


class ConnectionPoolCore:
    """Core connection pool functionality.

    Responsible for:
    - AsyncEngine and async_sessionmaker creation and management
    - Basic connection establishment and testing
    - Session and HA connection context managers
    - Cached query execution
    - Connection lifecycle management
    """

    def __init__(self, shared_context: PoolSharedContext) -> None:
        self.context = shared_context

        logger.info(
            f"ConnectionPoolCore initialized for service: {self.context.service_name}"
        )

    async def initialize(self) -> bool:
        """Initialize the connection pool core."""
        async with self.context.initialization_lock:
            if self.context.is_initialized:
                return True

            try:
                await self._setup_database_connections()

                if self.context.pool_config.enable_ha:
                    await self._setup_ha_pools()

                logger.info(
                    f"ConnectionPoolCore initialized successfully: {self.context.service_name}"
                )
                return True

            except Exception as e:
                logger.exception(f"Failed to initialize ConnectionPoolCore: {e}")
                raise

    async def _setup_database_connections(self) -> None:
        """Setup primary database connections with SQLAlchemy AsyncEngine."""
        db_config = self.context.db_config
        pool_config = self.context.pool_config

        async_url = (
            f"postgresql+asyncpg://{db_config.username}:{db_config.password}"
            f"@{db_config.host}:{db_config.port}/{db_config.database}"
        )

        engine_kwargs = {
            "pool_size": pool_config.pool_size,
            "max_overflow": pool_config.max_overflow,
            "pool_timeout": pool_config.timeout,
            "pool_pre_ping": pool_config.pool_pre_ping,
            "pool_recycle": pool_config.pool_recycle,
            "echo": db_config.echo_sql,
            "future": True,
            "connect_args": {
                "server_settings": {
                    "application_name": pool_config.application_name,
                    "timezone": "UTC",
                },
                "command_timeout": pool_config.timeout,
                "connect_timeout": 10,
            },
        }

        self.context.async_engine = create_async_engine(async_url, **engine_kwargs)
        self.context.async_session_factory = async_sessionmaker(
            bind=self.context.async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False,
        )

        self._setup_connection_monitoring()

        if not self.context.pool_config.skip_connection_test:
            await self.test_connections()
        else:
            logger.info("Skipping connection testing for validation")

        logger.info(
            f"Primary database engine created with pool_size={pool_config.pool_size}"
        )

    async def _setup_ha_pools(self) -> None:
        """Setup high availability asyncpg pools for direct database access."""
        if not self.context.pool_config.enable_ha:
            return

        db_config = self.context.db_config
        pool_config = self.context.pool_config

        try:
            # Create primary pool
            primary_dsn = (
                f"postgresql://{db_config.username}:{db_config.password}"
                f"@{db_config.host}:{db_config.port}/{db_config.database}"
            )

            primary_pool = await asyncpg.create_pool(
                dsn=primary_dsn,
                min_size=2,
                max_size=pool_config.pool_size,
                command_timeout=pool_config.timeout,
                max_inactive_connection_lifetime=3600,
                server_settings={
                    "application_name": f"{pool_config.application_name}_primary",
                    "timezone": "UTC",
                },
            )

            self.context.pg_pools["primary"] = primary_pool
            logger.info("HA primary pool created")

            # Setup replica pools if configured
            replica_hosts = self._get_replica_hosts()
            for i, (host, port) in enumerate(replica_hosts):
                replica_dsn = (
                    f"postgresql://{db_config.username}:{db_config.password}"
                    f"@{host}:{port}/{db_config.database}"
                )

                replica_pool = await asyncpg.create_pool(
                    dsn=replica_dsn,
                    min_size=1,
                    max_size=pool_config.pool_size // 2,
                    command_timeout=pool_config.timeout,
                    server_settings={
                        "application_name": f"{pool_config.application_name}_replica_{i}",
                        "timezone": "UTC",
                    },
                )

                self.context.pg_pools[f"replica_{i}"] = replica_pool
                logger.info(f"HA replica pool {i} created for {host}:{port}")

        except Exception as e:
            logger.warning(f"HA setup failed, continuing with single pool: {e}")

    def _get_replica_hosts(self) -> list[tuple[str, int]]:
        """Get replica database hosts from configuration or environment."""
        # This would typically come from configuration or service discovery
        # For now, return empty list - can be extended based on deployment setup
        return []

    def _setup_connection_monitoring(self) -> None:
        """Setup SQLAlchemy connection monitoring events."""
        if not self.context.async_engine:
            return

        @event.listens_for(self.context.async_engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record) -> None:
            self.context.metrics.record_connection()
            connection_id = self.context.register_connection()

            # Store connection ID in the connection record for later retrieval
            if hasattr(connection_record, "info"):
                connection_record.info["pool_manager_id"] = connection_id

            logger.debug(f"New database connection created: {connection_id}")

        @event.listens_for(self.context.async_engine.sync_engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy) -> None:
            self.context.metrics.active_connections += 1
            self.context.update_pool_utilization()

            connection_id = None
            if (
                hasattr(connection_record, "info")
                and "pool_manager_id" in connection_record.info
            ):
                connection_id = connection_record.info["pool_manager_id"]
                self.context.update_connection(connection_id, "checkout")

        @event.listens_for(self.context.async_engine.sync_engine, "checkin")
        def on_checkin(dbapi_connection, connection_record) -> None:
            self.context.metrics.active_connections = max(
                0, self.context.metrics.active_connections - 1
            )
            self.context.update_pool_utilization()

            connection_id = None
            if (
                hasattr(connection_record, "info")
                and "pool_manager_id" in connection_record.info
            ):
                connection_id = connection_record.info["pool_manager_id"]
                self.context.update_connection(connection_id, "checkin")

        @event.listens_for(self.context.async_engine.sync_engine, "invalidate")
        def on_invalidate(dbapi_connection, connection_record, exception) -> None:
            self.context.record_circuit_breaker_failure(exception)

            connection_id = None
            if (
                hasattr(connection_record, "info")
                and "pool_manager_id" in connection_record.info
            ):
                connection_id = connection_record.info["pool_manager_id"]
                self.context.unregister_connection(connection_id)

            logger.warning(f"Connection invalidated: {exception}")

    async def test_connections(self) -> None:
        """Test all connection types to ensure they work."""
        # Test SQLAlchemy async session
        async with self.get_session() as session:
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
            logger.debug("SQLAlchemy async session test passed")

        # Test HA pools if available
        if self.context.pg_pools:
            primary_pool = self.context.pg_pools.get("primary")
            if primary_pool:
                async with primary_pool.acquire() as conn:
                    result = await conn.fetchval("SELECT 1")
                    assert result == 1
                    logger.debug("HA primary pool test passed")

    @contextlib.asynccontextmanager
    async def get_session(
        self, mode: ConnectionMode = ConnectionMode.READ_WRITE
    ) -> AsyncIterator[AsyncSession]:
        """Get async SQLAlchemy session with automatic transaction management."""
        if not self.context.is_initialized:
            await self.initialize()

        if not self.context.async_session_factory:
            raise RuntimeError("Async session factory not initialized")

        if self.context.is_circuit_breaker_open():
            raise RuntimeError("Circuit breaker is open - database unavailable")

        session = self.context.async_session_factory()
        start_time = time.time()

        try:
            if mode == ConnectionMode.READ_ONLY:
                await session.execute(text("SET TRANSACTION READ ONLY"))

            yield session
            await session.commit()

            # Record successful operation
            duration_ms = (time.time() - start_time) * 1000
            self.context.metrics.record_query(duration_ms, success=True)
            self._update_response_time(duration_ms)
            self.context.record_performance_event("session", duration_ms, True)

        except Exception as e:
            await session.rollback()
            duration_ms = (time.time() - start_time) * 1000
            self.context.metrics.record_query(duration_ms, success=False)
            self.context.record_circuit_breaker_failure(e)
            self.context.record_performance_event("session", duration_ms, False)
            logger.exception(f"Session error in {self.context.service_name}: {e}")
            raise
        finally:
            await session.close()

    @contextlib.asynccontextmanager
    async def get_ha_connection(
        self, pool_name: str = "primary"
    ) -> AsyncIterator[asyncpg.Connection]:
        """Get direct asyncpg connection from HA pools."""
        if not self.context.is_initialized:
            await self.initialize()

        if pool_name not in self.context.pg_pools:
            raise ValueError(f"Pool '{pool_name}' not available")

        pool = self.context.pg_pools[pool_name]
        start_time = time.time()

        try:
            async with pool.acquire() as conn:
                yield conn

                # Record successful operation
                duration_ms = (time.time() - start_time) * 1000
                self.context.metrics.record_query(duration_ms, success=True)
                self.context.record_performance_event("ha_connection", duration_ms, True)

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.context.metrics.record_query(duration_ms, success=False)
            self.context.record_circuit_breaker_failure(e)
            self.context.record_performance_event("ha_connection", duration_ms, False)
            logger.exception(f"HA connection error for pool {pool_name}: {e}")
            raise

    async def execute_cached_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        cache_ttl_seconds: int = 300,
        cache_key_prefix: str = "pg_query"
    ) -> Any:
        """Execute a query with result caching for performance optimization."""
        # Generate cache key
        cache_key = None
        cache_enabled = self.context.cache_manager is not None

        if cache_enabled:
            params_str = str(sorted((params or {}).items()))
            content = f"{cache_key_prefix}:{query}:{params_str}"
            cache_key = f"pg_cache:{hashlib.md5(content.encode()).hexdigest()}"

            # Check cache first
            try:
                cached_result = await self.context.cache_manager.get(cache_key)
                if cached_result:
                    logger.debug(f"Cache hit for PostgreSQL query: {query[:50]}...")
                    return cached_result
            except Exception as e:
                logger.warning(f"Cache lookup failed for query: {e}")

        # Execute query against database
        try:
            async with self.get_session() as session:
                if params:
                    result = await session.execute(text(query), params)
                else:
                    result = await session.execute(text(query))

                # Convert result to cacheable format
                if result.returns_rows:
                    rows = result.fetchall()
                    # Convert Row objects to dictionaries for caching
                    cached_data = [dict(row._mapping) for row in rows]
                else:
                    cached_data = result.rowcount

                # Cache the result
                if cache_enabled and cache_key:
                    try:
                        await self.context.cache_manager.set(
                            cache_key,
                            cached_data,
                            ttl_seconds=cache_ttl_seconds
                        )
                        logger.debug(f"Cached PostgreSQL query result: {query[:50]}...")
                    except Exception as e:
                        logger.warning(f"Failed to cache query result: {e}")

                return cached_data

        except Exception as e:
            logger.exception(f"PostgreSQL cached query failed: {e}")
            raise

    def _update_response_time(self, response_time_ms: float) -> None:
        """Update average response time using exponential moving average."""
        alpha = 0.1
        if self.context.metrics.avg_response_time_ms == 0:
            self.context.metrics.avg_response_time_ms = response_time_ms
        else:
            self.context.metrics.avg_response_time_ms = (
                alpha * response_time_ms
                + (1 - alpha) * self.context.metrics.avg_response_time_ms
            )

    async def shutdown(self) -> None:
        """Shutdown the connection pool core and cleanup resources."""
        logger.info(f"Shutting down ConnectionPoolCore: {self.context.service_name}")

        try:
            # Close HA pools
            for pool_name, pool in self.context.pg_pools.items():
                try:
                    await pool.close()
                    logger.info(f"HA pool {pool_name} closed")
                except Exception as e:
                    logger.warning(f"Error closing HA pool {pool_name}: {e}")

            # Dispose SQLAlchemy engine
            if self.context.async_engine:
                await self.context.async_engine.dispose()
                logger.info("SQLAlchemy async engine disposed")

            # Clear state
            self.context.connection_registry.clear()

            logger.info(
                f"ConnectionPoolCore shutdown complete: {self.context.service_name}"
            )

        except Exception as e:
            logger.exception(f"Error during ConnectionPoolCore shutdown: {e}")

    @property
    def is_initialized(self) -> bool:
        """Check if the connection pool core is initialized."""
        return self.context.is_initialized
