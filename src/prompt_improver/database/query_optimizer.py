"""Advanced database query optimization for sub-50ms response times.

This module implements 2025 best practices for PostgreSQL query optimization
using the unified cache system for optimal performance.
"""

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.database import (
    ManagerMode,
    get_database_services,
)

logger = logging.getLogger(__name__)


def _get_performance_measure():
    """Lazy import performance measurement to avoid ML chain during database.models import."""
    try:
        from prompt_improver.performance.optimization.performance_optimizer import (
            measure_database_operation,
        )
        return measure_database_operation
    except ImportError:
        # Fallback context manager when performance optimization unavailable
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def fallback_measure(operation_name):
            yield {"operation": operation_name, "available": False}

        return fallback_measure


@dataclass
class QueryPerformanceMetrics:
    """Metrics for query performance tracking."""

    query_hash: str
    execution_time_ms: float
    rows_returned: int
    cache_hit: bool
    timestamp: datetime

    def meets_target(self, target_ms: float = 50) -> bool:
        """Check if query meets performance target."""
        return self.execution_time_ms <= target_ms


class PreparedStatementCache:
    """Cache for prepared statements to improve query performance."""

    def __init__(self, max_size: int = 100) -> None:
        self._statements: dict[str, str] = {}
        self._usage_count: dict[str, int] = {}
        self._max_size = max_size

    def get_or_create_statement(self, query: str, params: dict[str, Any]) -> str:
        """Get or create a prepared statement for the query."""
        query_hash = self._hash_query_structure(query)
        if query_hash not in self._statements:
            if len(self._statements) >= self._max_size:
                least_used = min(self._usage_count.items(), key=lambda x: x[1])
                del self._statements[least_used[0]]
                del self._usage_count[least_used[0]]
            self._statements[query_hash] = query
            self._usage_count[query_hash] = 0
        self._usage_count[query_hash] += 1
        return self._statements[query_hash]

    def _hash_query_structure(self, query: str) -> str:
        """Create a hash of the query structure for caching."""
        import hashlib

        normalized = " ".join(query.lower().split())
        return hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()

    async def run_orchestrated_analysis(
        self, analysis_type: str = "cache_performance", **kwargs
    ) -> dict[str, Any]:
        """Run orchestrated analysis for PreparedStatementCache component.

        Compatible with ML orchestrator integration patterns.

        Args:
            analysis_type: Type of analysis to run
            **kwargs: Additional analysis parameters

        Returns:
            Dictionary containing analysis results
        """
        if analysis_type == "cache_performance":
            return await self._analyze_cache_performance(**kwargs)
        if analysis_type == "query_optimization":
            return await self._analyze_query_optimization(**kwargs)
        if analysis_type == "cache_efficiency":
            return await self._analyze_cache_efficiency(**kwargs)
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    async def _analyze_cache_performance(self, **kwargs) -> dict[str, Any]:
        """Analyze cache performance metrics."""
        return {
            "component": "PreparedStatementCache",
            "analysis_type": "cache_performance",
            "cache_size": len(self._statements),
            "max_cache_size": self._max_size,
            "cache_utilization": len(self._statements) / self._max_size
            if self._max_size > 0
            else 0,
            "total_entries": len(self._statements),
            "usage_statistics": dict(self._usage_count),
            "most_used_queries": sorted(
                self._usage_count.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "cache_efficiency": "high"
            if len(self._statements) / self._max_size > 0.7
            else "medium"
            if len(self._statements) / self._max_size > 0.3
            else "low",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def _analyze_query_optimization(self, **kwargs) -> dict[str, Any]:
        """Analyze query optimization potential."""
        query_complexity = {}
        for query_hash, query in self._statements.items():
            complexity = self._calculate_query_complexity(query)
            query_complexity[query_hash] = complexity
        return {
            "component": "PreparedStatementCache",
            "analysis_type": "query_optimization",
            "total_cached_queries": len(self._statements),
            "query_complexity_distribution": query_complexity,
            "optimization_recommendations": self._generate_optimization_recommendations(),
            "cache_hit_potential": sum(self._usage_count.values())
            / len(self._usage_count)
            if self._usage_count
            else 0,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def _analyze_cache_efficiency(self, **kwargs) -> dict[str, Any]:
        """Analyze cache efficiency and provide recommendations."""
        if not self._usage_count:
            return {
                "component": "PreparedStatementCache",
                "analysis_type": "cache_efficiency",
                "status": "no_data",
                "message": "No usage data available for analysis",
            }
        usage_values = list(self._usage_count.values())
        total_usage = sum(usage_values)
        avg_usage = total_usage / len(usage_values)
        hot_queries = {k: v for k, v in self._usage_count.items() if v > avg_usage * 2}
        cold_queries = {
            k: v for k, v in self._usage_count.items() if v < avg_usage * 0.5
        }
        return {
            "component": "PreparedStatementCache",
            "analysis_type": "cache_efficiency",
            "total_usage": total_usage,
            "average_usage": avg_usage,
            "hot_queries_count": len(hot_queries),
            "cold_queries_count": len(cold_queries),
            "efficiency_score": min(1.0, len(hot_queries) / len(self._usage_count) * 2)
            if self._usage_count
            else 0,
            "recommendations": self._generate_cache_recommendations(
                hot_queries, cold_queries
            ),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _calculate_query_complexity(self, query: str) -> str:
        """Calculate query complexity based on SQL keywords."""
        query_lower = query.lower()
        complexity_score = 0
        if any(
            keyword in query_lower
            for keyword in ["select", "insert", "update", "delete"]
        ):
            complexity_score += 1
        if any(
            keyword in query_lower
            for keyword in ["join", "inner join", "left join", "right join"]
        ):
            complexity_score += 2
        if any(keyword in query_lower for keyword in ["with", "exists", "in ("]):
            complexity_score += 2
        if any(
            keyword in query_lower
            for keyword in ["group by", "having", "count", "sum", "avg"]
        ):
            complexity_score += 1
        if "over(" in query_lower.replace(" ", ""):
            complexity_score += 3
        if complexity_score >= 5:
            return "complex"
        if complexity_score >= 3:
            return "moderate"
        return "simple"

    def _generate_optimization_recommendations(self) -> list[str]:
        """Generate optimization recommendations based on cache state."""
        recommendations = []
        if len(self._statements) >= self._max_size * 0.9:
            recommendations.append("Consider increasing cache size to reduce eviction")
        if self._usage_count:
            usage_values = list(self._usage_count.values())
            if max(usage_values) > sum(usage_values) * 0.5:
                recommendations.append("Cache shows good hot query concentration")
            else:
                recommendations.append(
                    "Consider implementing query-specific optimization"
                )
        if len(self._statements) < self._max_size * 0.3:
            recommendations.append("Cache is underutilized, consider reducing size")
        return recommendations

    def _generate_cache_recommendations(
        self, hot_queries: dict[str, int], cold_queries: dict[str, int]
    ) -> list[str]:
        """Generate cache-specific recommendations."""
        recommendations = []
        if len(hot_queries) > len(self._usage_count) * 0.2:
            recommendations.append("Good query locality detected - cache is effective")
        if len(cold_queries) > len(self._usage_count) * 0.5:
            recommendations.append(
                "Many cold queries detected - consider cache cleanup"
            )
        if len(hot_queries) < 3:
            recommendations.append("Limited hot queries - consider query optimization")
        return recommendations


class OptimizedQueryExecutor:
    """High-performance query executor with caching and optimization."""

    def __init__(self) -> None:
        self._prepared_cache = PreparedStatementCache()
        self._query_cache = None  # Will be initialized on first use
        self._performance_metrics: list[QueryPerformanceMetrics] = []
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "cache_time_saved_ms": 0.0,
            "total_cached_queries": 0,
        }

    @asynccontextmanager
    async def execute_optimized_query(
        self,
        session: AsyncSession,
        query: str,
        params: dict[str, Any] | None = None,
        cache_ttl: int = 300,
        enable_cache: bool = True,
    ):
        """Execute a query with optimization and caching.

        Args:
            session: Database session
            query: SQL query string
            params: Query parameters
            cache_ttl: Cache time-to-live in seconds
            enable_cache: Whether to use query result caching
        """
        params = params or {}
        cache_key = self._generate_cache_key(query, params) if enable_cache else None
        if cache_key and enable_cache:
            cache_start = time.perf_counter()
            try:
                if self._query_cache is None:
                    from prompt_improver.services.cache.cache_factory import (
                        CacheFactory,
                    )
                    # Initialize utility cache for fast prepared statement caching (L1+L2, no warming)
                    self._query_cache = CacheFactory.get_utility_cache()
                cached_result = await self._query_cache.get(cache_key, l1_ttl=cache_ttl)
                if cached_result:
                    cache_time = (time.perf_counter() - cache_start) * 1000
                    self._cache_stats["hits"] += 1
                    self._cache_stats["cache_time_saved_ms"] += max(0, 50 - cache_time)
                    # Calculate rows returned safely
                    try:
                        if isinstance(cached_result, (list, tuple)) or hasattr(cached_result, '__len__'):
                            rows_returned = len(cached_result)
                        else:
                            rows_returned = 1
                    except (TypeError, AttributeError):
                        rows_returned = 1

                    query_metrics = QueryPerformanceMetrics(
                        query_hash=self._prepared_cache._hash_query_structure(query),
                        execution_time_ms=cache_time,
                        rows_returned=rows_returned,
                        cache_hit=True,
                        timestamp=datetime.now(UTC),
                    )
                    self._performance_metrics.append(query_metrics)
                    logger.debug(
                        f"Cache hit for query {query[:50]}... (saved ~{50 - cache_time:.2f}ms)"
                    )
                    yield {
                        "result": cached_result,
                        "cache_hit": True,
                        "execution_time_ms": cache_time,
                    }
                    return
                else:
                    self._cache_stats["misses"] += 1
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}, proceeding without cache")
                self._cache_stats["misses"] += 1
        measure_func = _get_performance_measure()
        async with measure_func("optimized_query") as perf_metrics:
            start_time = time.perf_counter()
            try:
                prepared_query = self._prepared_cache.get_or_create_statement(
                    query, params
                )
                result = await session.execute(text(prepared_query), params)
                rows = result.fetchall()
                execution_time = (time.perf_counter() - start_time) * 1000
                if cache_key and enable_cache and (execution_time < 1000):
                    try:
                        import json

                        # Serialize rows safely
                        try:
                            serialized_rows = json.dumps(
                                [
                                    {
                                        k: str(v)
                                        if not isinstance(
                                            v, (str, int, float, bool, type(None))
                                        )
                                        else v
                                        for k, v in row.items()
                                    }
                                    for row in rows
                                ],
                                default=str,
                            )
                        except Exception as serialization_error:
                            # Fallback to string representation if serialization fails
                            logger.warning(f"JSON serialization failed: {serialization_error}, using string fallback")
                            serialized_rows = str(rows)
                        if self._query_cache is None:
                            from prompt_improver.services.cache.cache_factory import (
                                CacheFactory,
                            )
                            # Initialize utility cache for query result caching
                            self._query_cache = CacheFactory.get_utility_cache()
                        await self._query_cache.set(
                            cache_key, serialized_rows, l1_ttl=cache_ttl
                        )
                        self._cache_stats["total_cached_queries"] += 1
                        logger.debug(
                            f"Cached query result: {query[:50]}... ({len(rows)} rows, TTL: {cache_ttl}s)"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to cache query result: {e}")
                query_metrics = QueryPerformanceMetrics(
                    query_hash=self._prepared_cache._hash_query_structure(query),
                    execution_time_ms=execution_time,
                    rows_returned=len(rows),
                    cache_hit=False,
                    timestamp=datetime.now(UTC),
                )
                self._performance_metrics.append(query_metrics)
                if len(self._performance_metrics) > 1000:
                    self._performance_metrics = self._performance_metrics[-1000:]
                if execution_time > 50:
                    logger.warning(
                        f"Slow query detected: {execution_time:.2f}ms (target: <50ms) - {query[:100]}..."
                    )
                yield {
                    "result": rows,
                    "cache_hit": False,
                    "execution_time_ms": execution_time,
                }
            except Exception as e:
                logger.exception(f"Query execution failed: {e}")
                raise

    def _generate_cache_key(self, query: str, params: dict[str, Any]) -> str:
        """Generate a cache key for the query and parameters."""
        import hashlib
        import json

        cache_data = {
            "query": query.strip(),
            "params": sorted(params.items()) if params else [],
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"query_cache:{hashlib.md5(cache_string.encode(), usedforsecurity=False).hexdigest()}"

    async def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary for executed queries."""
        if not self._performance_metrics:
            return {"message": "No query metrics available"}
        recent_metrics = [
            m
            for m in self._performance_metrics
            if m.timestamp > datetime.now(UTC) - timedelta(hours=1)
        ]
        if not recent_metrics:
            return {"message": "No recent query metrics available"}
        execution_times = [m.execution_time_ms for m in recent_metrics]
        cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
        cache_misses = len(recent_metrics) - cache_hits
        # Get actual cache hit rate from unified cache coordinator
        cache_hit_rate = 0.0
        if self._query_cache and hasattr(self._query_cache, 'get_performance_stats'):
            try:
                cache_stats = self._query_cache.get_performance_stats()
                cache_hit_rate = cache_stats.get('overall_hit_rate', 0.0)
            except Exception:
                # Fallback to manual tracking if unified cache stats unavailable
                total_cache_operations = self._cache_stats["hits"] + self._cache_stats["misses"]
                cache_hit_rate = (
                    self._cache_stats["hits"] / total_cache_operations
                    if total_cache_operations > 0
                    else 0
                )
        estimated_db_load_reduction = cache_hit_rate * 100
        return {
            "total_queries": len(recent_metrics),
            "avg_execution_time_ms": sum(execution_times) / len(execution_times),
            "max_execution_time_ms": max(execution_times),
            "min_execution_time_ms": min(execution_times),
            "queries_meeting_target": sum(1 for t in execution_times if t <= 50),
            "target_compliance_rate": sum(1 for t in execution_times if t <= 50)
            / len(execution_times),
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self._cache_stats["hits"],
            "cache_misses": self._cache_stats["misses"],
            "cache_time_saved_ms": self._cache_stats["cache_time_saved_ms"],
            "total_cached_queries": self._cache_stats["total_cached_queries"],
            "estimated_db_load_reduction_percent": round(
                estimated_db_load_reduction, 1
            ),
            "slow_queries_count": sum(1 for t in execution_times if t > 50),
        }


class DatabaseConnectionOptimizer:
    """Optimizer for database connection settings and pool configuration with dynamic resource detection."""

    def __init__(self, event_bus=None) -> None:
        """Initialize the optimizer with optional event bus integration."""
        self.event_bus = event_bus

    @staticmethod
    def _get_system_resources() -> dict[str, Any]:
        """Get current system resource information for dynamic optimization."""
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            return {
                "total_memory_gb": memory.total / 1024**3,
                "available_memory_gb": memory.available / 1024**3,
                "memory_percent": memory.percent,
                "cpu_count": cpu_count,
                "logical_cpu_count": psutil.cpu_count(logical=True),
            }
        except Exception as e:
            logger.warning(f"Failed to get system resources: {e}")
            return {
                "total_memory_gb": 4.0,
                "available_memory_gb": 2.0,
                "memory_percent": 50.0,
                "cpu_count": 2,
                "logical_cpu_count": 4,
            }

    @staticmethod
    def _calculate_optimal_memory_settings(
        system_resources: dict[str, Any],
    ) -> dict[str, str]:
        """Calculate optimal memory settings based on available system resources."""
        total_memory_gb = system_resources["total_memory_gb"]
        available_memory_gb = system_resources["available_memory_gb"]
        work_mem_mb = min(max(int(available_memory_gb * 256 * 0.25), 16), 256)
        effective_cache_size_gb = min(max(int(total_memory_gb * 0.75), 1), 8)
        return {
            "work_mem": f"{work_mem_mb}MB",
            "effective_cache_size": f"{effective_cache_size_gb}GB",
        }

    async def optimize_connection_settings(self):
        """Apply optimal connection settings for performance with dynamic resource detection."""
        manager = await get_database_services(ManagerMode.ASYNC_MODERN)
        system_resources = self._get_system_resources()
        memory_settings = self._calculate_optimal_memory_settings(system_resources)
        logger.info(
            f"Optimizing database settings for system with {system_resources['total_memory_gb']:.1f}GB total memory, {system_resources['cpu_count']:.1f} CPU cores"
        )
        optimization_queries = [
            "SET plan_cache_mode = 'force_generic_plan'",
            f"SET work_mem = '{memory_settings['work_mem']}'",
            f"SET effective_cache_size = '{memory_settings['effective_cache_size']}'",
            "SET random_page_cost = 1.1",
            f"SET max_parallel_workers_per_gather = {min(system_resources['cpu_count'], 4)}",
            "SET checkpoint_completion_target = 0.9",
            "SET jit = on",
            "SET jit_above_cost = 100000",
        ]
        async with manager.get_session() as session:
            applied_settings = []
            for query in optimization_queries:
                try:
                    await session.execute(text(query))
                    logger.debug(f"Applied optimization: {query}")
                    applied_settings.append(query)
                except Exception as e:
                    logger.warning(f"Failed to apply optimization {query}: {e}")
        await self._emit_optimization_event(
            applied_settings, system_resources, memory_settings
        )

    async def create_performance_indexes(self):
        """Create indexes for optimal query performance."""
        manager = await get_database_services(ManagerMode.ASYNC_MODERN)
        index_queries = [
            "\n            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rule_effectiveness_lookup\n            ON rule_effectiveness (rule_id, created_at DESC)\n            WHERE active = true\n            ",
            "\n            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_active_lookup\n            ON sessions (session_id, updated_at DESC)\n            WHERE active = true\n            ",
            "\n            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_prompt_improvements_recent\n            ON prompt_improvements (created_at DESC, session_id)\n            WHERE created_at > NOW() - INTERVAL '7 days'\n            ",
            "\n            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analytics_recent_data\n            ON analytics_events (event_type, created_at DESC)\n            WHERE created_at > NOW() - INTERVAL '30 days'\n            ",
        ]
        async with manager.get_session() as session:
            created_indexes = []
            for index_query in index_queries:
                try:
                    await session.execute(text(index_query))
                    logger.info("Created performance index")
                    created_indexes.append(index_query.strip())
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")
        await self._emit_index_creation_event(created_indexes)

    async def _emit_optimization_event(
        self, applied_settings: list, system_resources: dict, memory_settings: dict
    ) -> None:
        """Emit database connection optimization event using protocol-based approach."""
        from prompt_improver.database.protocols.events import EventType
        from prompt_improver.database.services.optional_registry import (
            get_optional_services_registry,
        )

        registry = get_optional_services_registry()

        event_data = {
            "applied_settings": applied_settings,
            "system_resources": system_resources,
            "memory_settings": memory_settings,
            "optimization_timestamp": datetime.now().isoformat(),
            "source": "database_connection_optimizer",
        }

        # Use optional registry - graceful degradation if ML services unavailable
        success = await registry.dispatch_event_if_available(
            EventType.DATABASE_OPTIMIZATION_COMPLETED,
            event_data
        )

        if success:
            logger.debug("Database optimization event dispatched successfully")
        else:
            logger.debug("No ML event dispatcher available - optimization completed without ML integration")

    async def _emit_index_creation_event(self, created_indexes: list) -> None:
        """Emit database index creation event using protocol-based approach."""
        from prompt_improver.database.protocols.events import EventType
        from prompt_improver.database.services.optional_registry import (
            get_optional_services_registry,
        )

        registry = get_optional_services_registry()

        event_data = {
            "created_indexes": created_indexes,
            "index_count": len(created_indexes),
            "creation_timestamp": datetime.now().isoformat(),
            "source": "database_connection_optimizer",
        }

        # Use optional registry for graceful degradation
        success = await registry.dispatch_event_if_available(
            EventType.DATABASE_OPTIMIZATION_COMPLETED,
            event_data
        )

        if success:
            logger.debug(f"Database index creation event dispatched for {len(created_indexes)} indexes")
        else:
            logger.debug(f"No ML event dispatcher available - {len(created_indexes)} indexes created without ML integration")

    async def _emit_resource_analysis_event(
        self, system_resources: dict, memory_settings: dict
    ) -> None:
        """Emit database resource analysis event using protocol-based approach."""
        from prompt_improver.database.protocols.events import EventType
        from prompt_improver.database.services.optional_registry import (
            get_optional_services_registry,
        )

        registry = get_optional_services_registry()

        event_data = {
            "system_resources": system_resources,
            "recommended_settings": memory_settings,
            "analysis_timestamp": datetime.now().isoformat(),
            "source": "database_connection_optimizer",
        }

        # Use optional registry for graceful degradation
        success = await registry.dispatch_event_if_available(
            EventType.DATABASE_OPTIMIZATION_COMPLETED,
            event_data
        )

        if success:
            logger.debug("Database resource analysis event dispatched successfully")
        else:
            logger.debug("No ML event dispatcher available - resource analysis completed without ML integration")


_global_executor: OptimizedQueryExecutor | None = None


def get_query_executor() -> OptimizedQueryExecutor:
    """Get the global optimized query executor."""
    global _global_executor
    if _global_executor is None:
        _global_executor = OptimizedQueryExecutor()
    return _global_executor


async def execute_optimized_query(
    session: AsyncSession,
    query: str,
    params: dict[str, Any] | None = None,
    cache_ttl: int = 300,
    enable_cache: bool = True,
):
    """Execute an optimized query with caching and performance monitoring."""
    executor = get_query_executor()
    async with executor.execute_optimized_query(
        session, query, params, cache_ttl, enable_cache
    ) as result:
        return result
