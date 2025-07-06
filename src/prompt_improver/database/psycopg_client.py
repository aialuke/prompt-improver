"""Type-safe database client using psycopg3 + Pydantic for zero serialization overhead.
Research-validated patterns for high-performance database operations.
"""

import contextlib
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ValidationError

from .config import DatabaseConfig

T = TypeVar("T", bound=BaseModel)


class QueryMetrics:
    """Track query performance metrics for Phase 2 requirements"""

    def __init__(self):
        self.query_times: list[float] = []
        self.slow_queries: list[dict[str, Any]] = []
        self.total_queries = 0

    def record_query(self, query: str, duration_ms: float, params: dict | None = None):
        """Record query execution metrics"""
        self.total_queries += 1
        self.query_times.append(duration_ms)

        # Track slow queries (>50ms target)
        if duration_ms > 50:
            self.slow_queries.append({
                "query": query[:100] + "..." if len(query) > 100 else query,
                "duration_ms": duration_ms,
                "timestamp": datetime.utcnow().isoformat(),
                "params_count": len(params) if params else 0,
            })

    @property
    def avg_query_time(self) -> float:
        """Average query time in milliseconds"""
        return sum(self.query_times) / len(self.query_times) if self.query_times else 0

    @property
    def queries_under_50ms(self) -> float:
        """Percentage of queries under 50ms target"""
        if not self.query_times:
            return 0
        under_target = sum(1 for t in self.query_times if t <= 50)
        return (under_target / len(self.query_times)) * 100


class TypeSafePsycopgClient:
    """High-performance type-safe database client using psycopg3 + Pydantic.

    Features:
    - Zero serialization overhead with direct SQL
    - Automatic type safety with Pydantic models
    - Research-validated connection pooling
    - Real-time performance monitoring
    """

    def __init__(self, config: DatabaseConfig | None = None):
        self.config = config or DatabaseConfig()
        self.metrics = QueryMetrics()

        # Build connection string for psycopg3
        self.conninfo = f"postgresql://{self.config.postgres_username}:{self.config.postgres_password}@{self.config.postgres_host}:{self.config.postgres_port}/{self.config.postgres_database}"

        # Initialize connection pool with research-validated settings
        self.pool = AsyncConnectionPool(
            conninfo=self.conninfo,
            min_size=self.config.pool_min_size,
            max_size=self.config.pool_max_size,
            timeout=self.config.pool_timeout,
            max_lifetime=self.config.pool_max_lifetime,
            max_idle=self.config.pool_max_idle,
            # Performance optimizations
            kwargs={
                "row_factory": dict_row,  # Return dict rows for Pydantic mapping
                "prepare_threshold": 5,  # Prepare frequently used queries
                "autocommit": False,  # Explicit transaction control
            },
        )

    async def __aenter__(self):
        """Async context manager entry"""
        await self.pool.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.pool.__aexit__(exc_type, exc_val, exc_tb)

    @contextlib.asynccontextmanager
    async def connection(self):
        """Get a connection from the pool"""
        async with self.pool.connection() as conn:
            yield conn

    async def fetch_models(
        self, model_class: type[T], query: str, params: dict[str, Any] | None = None
    ) -> list[T]:
        """Execute query and return typed Pydantic models.
        Zero serialization overhead with direct row mapping.
        """
        start_time = time.perf_counter()

        try:
            async with self.connection() as conn, conn.cursor() as cur:
                await cur.execute(query, params or {})
                rows = await cur.fetchall()

                # Direct Pydantic model creation from dict rows
                models = []
                for row in rows:
                    try:
                        models.append(model_class.model_validate(row))
                    except ValidationError as e:
                        # Log validation error but continue processing
                        print(f"Validation error for {model_class.__name__}: {e}")
                        continue

                # Record performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.record_query(query, duration_ms, params)

                return models

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_query(query, duration_ms, params)
            raise

    async def fetch_one_model(
        self, model_class: type[T], query: str, params: dict[str, Any] | None = None
    ) -> T | None:
        """Execute query and return single typed Pydantic model."""
        start_time = time.perf_counter()

        try:
            async with self.connection() as conn, conn.cursor() as cur:
                await cur.execute(query, params or {})
                row = await cur.fetchone()

                if row is None:
                    return None

                # Direct Pydantic model creation
                model = model_class.model_validate(row)

                # Record performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.record_query(query, duration_ms, params)

                return model

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_query(query, duration_ms, params)
            raise

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> int:
        """Execute non-SELECT query and return affected row count."""
        start_time = time.perf_counter()

        try:
            async with self.connection() as conn, conn.cursor() as cur:
                await cur.execute(query, params or {})

                # Record performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.record_query(query, duration_ms, params)

                return cur.rowcount

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_query(query, duration_ms, params)
            raise

    async def fetch_raw(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute query and return raw dictionary results.
        For cases where Pydantic models aren't needed.
        """
        start_time = time.perf_counter()

        try:
            async with self.connection() as conn, conn.cursor() as cur:
                await cur.execute(query, params or {})
                rows = await cur.fetchall()

                # Record performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.record_query(query, duration_ms, params)

                return rows

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_query(query, duration_ms, params)
            raise

    async def get_performance_stats(self) -> dict[str, Any]:
        """Get current performance statistics"""
        # Get PostgreSQL cache hit ratio
        cache_hit_query = """
        SELECT 
            CASE 
                WHEN (blks_hit + blks_read) = 0 THEN 0
                ELSE blks_hit::float / (blks_hit + blks_read)
            END as cache_hit_ratio
        FROM pg_stat_database 
        WHERE datname = current_database()
        """

        cache_stats = await self.fetch_raw(cache_hit_query)
        cache_hit_ratio = cache_stats[0]["cache_hit_ratio"] if cache_stats else 0

        return {
            "total_queries": self.metrics.total_queries,
            "avg_query_time_ms": round(self.metrics.avg_query_time, 2),
            "queries_under_50ms_percent": round(self.metrics.queries_under_50ms, 1),
            "slow_query_count": len(self.metrics.slow_queries),
            "cache_hit_ratio": round(cache_hit_ratio, 3) if cache_hit_ratio else 0,
            "target_query_time_ms": self.config.target_query_time_ms,
            "target_cache_hit_ratio": self.config.target_cache_hit_ratio,
            "performance_status": "GOOD"
            if (
                self.metrics.avg_query_time <= self.config.target_query_time_ms
                and cache_hit_ratio >= self.config.target_cache_hit_ratio
            )
            else "NEEDS_ATTENTION",
        }

    def reset_metrics(self):
        """Reset performance metrics (useful for testing)"""
        self.metrics = QueryMetrics()


# Global client instance
_client: TypeSafePsycopgClient | None = None


async def get_psycopg_client() -> TypeSafePsycopgClient:
    """Get or create the global psycopg client"""
    global _client
    if _client is None:
        _client = TypeSafePsycopgClient()
        await _client.__aenter__()
    return _client


async def close_psycopg_client():
    """Close the global psycopg client"""
    global _client
    if _client is not None:
        await _client.__aexit__(None, None, None)
        _client = None
