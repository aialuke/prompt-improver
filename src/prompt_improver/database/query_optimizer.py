"""Advanced database query optimization for sub-50ms response times.

This module implements 2025 best practices for PostgreSQL query optimization
including prepared statements, connection pooling enhancements, and query caching.
"""

import asyncio
import logging
import psutil
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Executable

from prompt_improver.database.psycopg_client import get_psycopg_client
from ..performance.optimization.performance_optimizer import measure_database_operation
from prompt_improver.utils.redis_cache import RedisCache

logger = logging.getLogger(__name__)

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

    def __init__(self, max_size: int = 100):
        self._statements: Dict[str, str] = {}
        self._usage_count: Dict[str, int] = {}
        self._max_size = max_size

    def get_or_create_statement(self, query: str, params: Dict[str, Any]) -> str:
        """Get or create a prepared statement for the query."""
        # Create a hash of the query structure (without parameter values)
        query_hash = self._hash_query_structure(query)

        if query_hash not in self._statements:
            if len(self._statements) >= self._max_size:
                # Remove least used statement
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
        # Normalize query by removing extra whitespace and converting to lowercase
        normalized = " ".join(query.lower().split())
        return hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()

    async def run_orchestrated_analysis(self, analysis_type: str = "cache_performance", **kwargs) -> Dict[str, Any]:
        """
        Run orchestrated analysis for PreparedStatementCache component.

        Compatible with ML orchestrator integration patterns.

        Args:
            analysis_type: Type of analysis to run
            **kwargs: Additional analysis parameters

        Returns:
            Dictionary containing analysis results
        """
        if analysis_type == "cache_performance":
            return await self._analyze_cache_performance(**kwargs)
        elif analysis_type == "query_optimization":
            return await self._analyze_query_optimization(**kwargs)
        elif analysis_type == "cache_efficiency":
            return await self._analyze_cache_efficiency(**kwargs)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

    async def _analyze_cache_performance(self, **kwargs) -> Dict[str, Any]:
        """Analyze cache performance metrics."""
        return {
            "component": "PreparedStatementCache",
            "analysis_type": "cache_performance",
            "cache_size": len(self._statements),
            "max_cache_size": self._max_size,
            "cache_utilization": len(self._statements) / self._max_size if self._max_size > 0 else 0,
            "total_entries": len(self._statements),
            "usage_statistics": dict(self._usage_count),
            "most_used_queries": sorted(self._usage_count.items(), key=lambda x: x[1], reverse=True)[:5],
            "cache_efficiency": "high" if len(self._statements) / self._max_size > 0.7 else "medium" if len(self._statements) / self._max_size > 0.3 else "low",
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _analyze_query_optimization(self, **kwargs) -> Dict[str, Any]:
        """Analyze query optimization potential."""
        # Calculate query complexity distribution
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
            "cache_hit_potential": sum(self._usage_count.values()) / len(self._usage_count) if self._usage_count else 0,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _analyze_cache_efficiency(self, **kwargs) -> Dict[str, Any]:
        """Analyze cache efficiency and provide recommendations."""
        if not self._usage_count:
            return {
                "component": "PreparedStatementCache",
                "analysis_type": "cache_efficiency",
                "status": "no_data",
                "message": "No usage data available for analysis"
            }

        # Calculate efficiency metrics
        usage_values = list(self._usage_count.values())
        total_usage = sum(usage_values)
        avg_usage = total_usage / len(usage_values)

        # Identify hot and cold queries
        hot_queries = {k: v for k, v in self._usage_count.items() if v > avg_usage * 2}
        cold_queries = {k: v for k, v in self._usage_count.items() if v < avg_usage * 0.5}

        return {
            "component": "PreparedStatementCache",
            "analysis_type": "cache_efficiency",
            "total_usage": total_usage,
            "average_usage": avg_usage,
            "hot_queries_count": len(hot_queries),
            "cold_queries_count": len(cold_queries),
            "efficiency_score": min(1.0, (len(hot_queries) / len(self._usage_count)) * 2) if self._usage_count else 0,
            "recommendations": self._generate_cache_recommendations(hot_queries, cold_queries),
            "timestamp": datetime.utcnow().isoformat()
        }

    def _calculate_query_complexity(self, query: str) -> str:
        """Calculate query complexity based on SQL keywords."""
        query_lower = query.lower()
        complexity_score = 0

        # Basic queries
        if any(keyword in query_lower for keyword in ['select', 'insert', 'update', 'delete']):
            complexity_score += 1

        # Joins increase complexity
        if any(keyword in query_lower for keyword in ['join', 'inner join', 'left join', 'right join']):
            complexity_score += 2

        # Subqueries and CTEs
        if any(keyword in query_lower for keyword in ['with', 'exists', 'in (']):
            complexity_score += 2

        # Aggregation
        if any(keyword in query_lower for keyword in ['group by', 'having', 'count', 'sum', 'avg']):
            complexity_score += 1

        # Window functions
        if 'over(' in query_lower.replace(' ', ''):
            complexity_score += 3

        if complexity_score >= 5:
            return "complex"
        elif complexity_score >= 3:
            return "moderate"
        else:
            return "simple"

    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on cache state."""
        recommendations = []

        if len(self._statements) >= self._max_size * 0.9:
            recommendations.append("Consider increasing cache size to reduce eviction")

        if self._usage_count:
            usage_values = list(self._usage_count.values())
            if max(usage_values) > sum(usage_values) * 0.5:
                recommendations.append("Cache shows good hot query concentration")
            else:
                recommendations.append("Consider implementing query-specific optimization")

        if len(self._statements) < self._max_size * 0.3:
            recommendations.append("Cache is underutilized, consider reducing size")

        return recommendations

    def _generate_cache_recommendations(self, hot_queries: Dict[str, int], cold_queries: Dict[str, int]) -> List[str]:
        """Generate cache-specific recommendations."""
        recommendations = []

        if len(hot_queries) > len(self._usage_count) * 0.2:
            recommendations.append("Good query locality detected - cache is effective")

        if len(cold_queries) > len(self._usage_count) * 0.5:
            recommendations.append("Many cold queries detected - consider cache cleanup")

        if len(hot_queries) < 3:
            recommendations.append("Limited hot queries - consider query optimization")

        return recommendations

class OptimizedQueryExecutor:
    """High-performance query executor with caching and optimization."""

    def __init__(self):
        self._prepared_cache = PreparedStatementCache()
        self._query_cache = RedisCache()
        self._performance_metrics: List[QueryPerformanceMetrics] = []
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "cache_time_saved_ms": 0.0,
            "total_cached_queries": 0
        }

    @asynccontextmanager
    async def execute_optimized_query(
        self,
        session: AsyncSession,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        cache_ttl: int = 300,  # 5 minutes default
        enable_cache: bool = True
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

        # Generate cache key for query result caching
        cache_key = self._generate_cache_key(query, params) if enable_cache else None

        # Check cache first
        if cache_key and enable_cache:
            cache_start = time.perf_counter()
            try:
                cached_result = await self._query_cache.get(cache_key)
                if cached_result:
                    cache_time = (time.perf_counter() - cache_start) * 1000
                    
                    # Update cache statistics
                    self._cache_stats["hits"] += 1
                    self._cache_stats["cache_time_saved_ms"] += max(0, 50 - cache_time)  # Assuming 50ms baseline
                    
                    # Record cache hit metric
                    query_metrics = QueryPerformanceMetrics(
                        query_hash=self._prepared_cache._hash_query_structure(query),
                        execution_time_ms=cache_time,
                        rows_returned=len(cached_result) if isinstance(cached_result, list) else 1,
                        cache_hit=True,
                        timestamp=datetime.utcnow()
                    )
                    self._performance_metrics.append(query_metrics)
                    
                    logger.debug(f"Cache hit for query {query[:50]}... (saved ~{50-cache_time:.1f}ms)")
                    
                    yield {
                        "result": cached_result,
                        "cache_hit": True,
                        "execution_time_ms": cache_time
                    }
                    return
                else:
                    self._cache_stats["misses"] += 1
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}, proceeding without cache")
                self._cache_stats["misses"] += 1

        # Execute query with performance measurement
        async with measure_database_operation("optimized_query") as perf_metrics:
            start_time = time.perf_counter()

            try:
                # Use prepared statement if beneficial
                prepared_query = self._prepared_cache.get_or_create_statement(query, params)

                # Execute the query
                result = await session.execute(text(prepared_query), params)
                rows = result.fetchall()

                execution_time = (time.perf_counter() - start_time) * 1000

                # Cache the result if enabled
                if cache_key and enable_cache and execution_time < 1000:  # Only cache fast queries
                    try:
                        # Serialize the result for caching
                        import json
                        serialized_rows = json.dumps(
                            [{k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                              for k, v in row.items()} for row in rows],
                            default=str
                        ).encode()
                        
                        await self._query_cache.set(cache_key, serialized_rows, expire=cache_ttl)
                        self._cache_stats["total_cached_queries"] += 1
                        
                        logger.debug(f"Cached query result: {query[:50]}... ({len(rows)} rows, TTL: {cache_ttl}s)")
                    except Exception as e:
                        logger.warning(f"Failed to cache query result: {e}")

                # Record performance metrics
                query_metrics = QueryPerformanceMetrics(
                    query_hash=self._prepared_cache._hash_query_structure(query),
                    execution_time_ms=execution_time,
                    rows_returned=len(rows),
                    cache_hit=False,
                    timestamp=datetime.utcnow()
                )
                self._performance_metrics.append(query_metrics)

                # Keep only last 1000 metrics
                if len(self._performance_metrics) > 1000:
                    self._performance_metrics = self._performance_metrics[-1000:]

                # Log slow queries
                if execution_time > 50:
                    logger.warning(
                        f"Slow query detected: {execution_time:.2f}ms "
                        f"(target: <50ms) - {query[:100]}..."
                    )

                yield {
                    "result": rows,
                    "cache_hit": False,
                    "execution_time_ms": execution_time
                }

            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise

    def _generate_cache_key(self, query: str, params: Dict[str, Any]) -> str:
        """Generate a cache key for the query and parameters."""
        import hashlib
        import json

        # Create a deterministic string from query and params
        cache_data = {
            "query": query.strip(),
            "params": sorted(params.items()) if params else []
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"query_cache:{hashlib.md5(cache_string.encode(), usedforsecurity=False).hexdigest()}"

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for executed queries."""
        if not self._performance_metrics:
            return {"message": "No query metrics available"}

        recent_metrics = [
            m for m in self._performance_metrics
            if m.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]

        if not recent_metrics:
            return {"message": "No recent query metrics available"}

        execution_times = [m.execution_time_ms for m in recent_metrics]
        cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
        cache_misses = len(recent_metrics) - cache_hits

        # Calculate database load reduction
        total_cache_operations = self._cache_stats["hits"] + self._cache_stats["misses"]
        cache_hit_rate = self._cache_stats["hits"] / total_cache_operations if total_cache_operations > 0 else 0
        
        # Estimate database load reduction (cache hits avoid DB queries)
        estimated_db_load_reduction = cache_hit_rate * 100  # Each cache hit is a DB query avoided

        return {
            "total_queries": len(recent_metrics),
            "avg_execution_time_ms": sum(execution_times) / len(execution_times),
            "max_execution_time_ms": max(execution_times),
            "min_execution_time_ms": min(execution_times),
            "queries_meeting_target": sum(1 for t in execution_times if t <= 50),
            "target_compliance_rate": sum(1 for t in execution_times if t <= 50) / len(execution_times),
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self._cache_stats["hits"],
            "cache_misses": self._cache_stats["misses"],
            "cache_time_saved_ms": self._cache_stats["cache_time_saved_ms"],
            "total_cached_queries": self._cache_stats["total_cached_queries"],
            "estimated_db_load_reduction_percent": round(estimated_db_load_reduction, 1),
            "slow_queries_count": sum(1 for t in execution_times if t > 50)
        }

class DatabaseConnectionOptimizer:
    """Optimizer for database connection settings and pool configuration with dynamic resource detection."""

    def __init__(self, event_bus=None):
        """Initialize the optimizer with optional event bus integration."""
        self.event_bus = event_bus

    @staticmethod
    def _get_system_resources() -> Dict[str, Any]:
        """Get current system resource information for dynamic optimization."""
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()

            return {
                "total_memory_gb": memory.total / (1024**3),
                "available_memory_gb": memory.available / (1024**3),
                "memory_percent": memory.percent,
                "cpu_count": cpu_count,
                "logical_cpu_count": psutil.cpu_count(logical=True)
            }
        except Exception as e:
            logger.warning(f"Failed to get system resources: {e}")
            # Fallback to conservative defaults
            return {
                "total_memory_gb": 4.0,
                "available_memory_gb": 2.0,
                "memory_percent": 50.0,
                "cpu_count": 2,
                "logical_cpu_count": 4
            }

    @staticmethod
    def _calculate_optimal_memory_settings(system_resources: Dict[str, Any]) -> Dict[str, str]:
        """Calculate optimal memory settings based on available system resources."""
        total_memory_gb = system_resources["total_memory_gb"]
        available_memory_gb = system_resources["available_memory_gb"]

        # Conservative memory allocation (following 2025 best practices)
        # Use 25% of available memory for database operations, with limits
        work_mem_mb = min(max(int(available_memory_gb * 256 * 0.25), 16), 256)  # 16MB-256MB
        effective_cache_size_gb = min(max(int(total_memory_gb * 0.75), 1), 8)  # 1GB-8GB

        return {
            "work_mem": f"{work_mem_mb}MB",
            "effective_cache_size": f"{effective_cache_size_gb}GB"
        }

    async def optimize_connection_settings(self):
        """Apply optimal connection settings for performance with dynamic resource detection."""
        client = await get_psycopg_client()

        # Get system resources for dynamic optimization
        system_resources = self._get_system_resources()
        memory_settings = self._calculate_optimal_memory_settings(system_resources)

        logger.info(f"Optimizing database settings for system with {system_resources['total_memory_gb']:.1f}GB total memory, {system_resources['cpu_count']} CPU cores")

        # Dynamic optimization queries based on system resources
        optimization_queries = [
            # Enable query plan caching
            "SET plan_cache_mode = 'force_generic_plan'",

            # Dynamic work memory for sorting and hashing
            f"SET work_mem = '{memory_settings['work_mem']}'",

            # Dynamic effective cache size
            f"SET effective_cache_size = '{memory_settings['effective_cache_size']}'",

            # Optimize random page cost for SSD (2025 standard)
            "SET random_page_cost = 1.1",

            # Dynamic parallel workers based on CPU count
            f"SET max_parallel_workers_per_gather = {min(system_resources['cpu_count'], 4)}",

            # Optimize checkpoint settings
            "SET checkpoint_completion_target = 0.9",

            # Enable JIT compilation for complex queries (PostgreSQL 11+)
            "SET jit = on",
            "SET jit_above_cost = 100000",
        ]

        async with client.connection() as conn:
            applied_settings = []
            for query in optimization_queries:
                try:
                    await conn.execute(query)
                    logger.debug(f"Applied optimization: {query}")
                    applied_settings.append(query)
                except Exception as e:
                    logger.warning(f"Failed to apply optimization {query}: {e}")

        # Emit optimization event for orchestrator coordination
        await self._emit_optimization_event(applied_settings, system_resources, memory_settings)

    async def create_performance_indexes(self):
        """Create indexes for optimal query performance."""
        client = await get_psycopg_client()

        # Performance-critical indexes
        index_queries = [
            # Index for rule effectiveness queries
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rule_effectiveness_lookup
            ON rule_effectiveness (rule_id, created_at DESC)
            WHERE active = true
            """,

            # Index for session queries
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_active_lookup
            ON sessions (session_id, updated_at DESC)
            WHERE active = true
            """,

            # Index for prompt improvement queries
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_prompt_improvements_recent
            ON prompt_improvements (created_at DESC, session_id)
            WHERE created_at > NOW() - INTERVAL '7 days'
            """,

            # Partial index for analytics queries
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analytics_recent_data
            ON analytics_events (event_type, created_at DESC)
            WHERE created_at > NOW() - INTERVAL '30 days'
            """
        ]

        async with client.connection() as conn:
            created_indexes = []
            for index_query in index_queries:
                try:
                    await conn.execute(index_query)
                    logger.info(f"Created performance index")
                    created_indexes.append(index_query.strip())
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")

        # Emit index creation event for orchestrator coordination
        await self._emit_index_creation_event(created_indexes)

    async def _emit_optimization_event(self, applied_settings: list, system_resources: dict, memory_settings: dict):
        """Emit database connection optimization event."""
        if not self.event_bus:
            return

        try:
            # Import here to avoid circular imports
            from ..ml.orchestration.events.event_types import EventType, MLEvent

            await self.event_bus.emit(MLEvent(
                event_type=EventType.DATABASE_CONNECTION_OPTIMIZED,
                source="database_connection_optimizer",
                data={
                    "applied_settings": applied_settings,
                    "system_resources": system_resources,
                    "memory_settings": memory_settings,
                    "optimization_timestamp": datetime.now().isoformat()
                }
            ))
        except Exception as e:
            # Don't fail optimization if event emission fails
            logger.debug(f"Failed to emit optimization event: {e}")

    async def _emit_index_creation_event(self, created_indexes: list):
        """Emit database index creation event."""
        if not self.event_bus:
            return

        try:
            # Import here to avoid circular imports
            from ..ml.orchestration.events.event_types import EventType, MLEvent

            await self.event_bus.emit(MLEvent(
                event_type=EventType.DATABASE_INDEXES_CREATED,
                source="database_connection_optimizer",
                data={
                    "created_indexes": created_indexes,
                    "index_count": len(created_indexes),
                    "creation_timestamp": datetime.now().isoformat()
                }
            ))
        except Exception as e:
            # Don't fail index creation if event emission fails
            logger.debug(f"Failed to emit index creation event: {e}")

    async def _emit_resource_analysis_event(self, system_resources: dict, memory_settings: dict):
        """Emit database resource analysis event."""
        if not self.event_bus:
            return

        try:
            # Import here to avoid circular imports
            from ..ml.orchestration.events.event_types import EventType, MLEvent

            await self.event_bus.emit(MLEvent(
                event_type=EventType.DATABASE_RESOURCE_ANALYSIS_COMPLETED,
                source="database_connection_optimizer",
                data={
                    "system_resources": system_resources,
                    "recommended_settings": memory_settings,
                    "analysis_timestamp": datetime.now().isoformat()
                }
            ))
        except Exception as e:
            # Don't fail analysis if event emission fails
            logger.debug(f"Failed to emit resource analysis event: {e}")

# Global query executor instance
_global_executor: Optional[OptimizedQueryExecutor] = None

def get_query_executor() -> OptimizedQueryExecutor:
    """Get the global optimized query executor."""
    global _global_executor
    if _global_executor is None:
        _global_executor = OptimizedQueryExecutor()
    return _global_executor

# Convenience function for optimized queries
async def execute_optimized_query(
    session: AsyncSession,
    query: str,
    params: Optional[Dict[str, Any]] = None,
    cache_ttl: int = 300,
    enable_cache: bool = True
):
    """Execute an optimized query with caching and performance monitoring."""
    executor = get_query_executor()
    async with executor.execute_optimized_query(
        session, query, params, cache_ttl, enable_cache
    ) as result:
        return result
