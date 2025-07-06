"""Real-time database performance monitoring using pg_stat_statements.
Comprehensive monitoring for Phase 2 <50ms query time and 90% cache hit ratio requirements.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .psycopg_client import TypeSafePsycopgClient, get_psycopg_client


@dataclass
class QueryPerformanceMetric:
    """Individual query performance metric"""

    query_text: str
    calls: int
    total_exec_time: float
    mean_exec_time: float
    max_exec_time: float
    min_exec_time: float
    rows_affected: int
    cache_hit_ratio: float


@dataclass
class DatabasePerformanceSnapshot:
    """Complete database performance snapshot"""

    timestamp: datetime
    cache_hit_ratio: float
    active_connections: int
    total_queries: int
    avg_query_time_ms: float
    slow_queries_count: int
    top_slow_queries: list[QueryPerformanceMetric]
    database_size_mb: float
    index_hit_ratio: float


class DatabasePerformanceMonitor:
    """Real-time database performance monitoring using PostgreSQL statistics.

    Features:
    - Real-time cache hit ratio monitoring (target: >90%)
    - Query performance tracking (target: <50ms)
    - Automatic slow query detection
    - Connection pool monitoring
    - Index effectiveness analysis
    """

    def __init__(self, client: TypeSafePsycopgClient | None = None):
        self.client = client
        self._monitoring = False
        self._snapshots: list[DatabasePerformanceSnapshot] = []

    async def get_client(self) -> TypeSafePsycopgClient:
        """Get database client"""
        if self.client is None:
            return await get_psycopg_client()
        return self.client

    async def get_cache_hit_ratio(self) -> float:
        """Get current cache hit ratio from pg_stat_database"""
        client = await self.get_client()

        query = """
        SELECT 
            CASE 
                WHEN (blks_hit + blks_read) = 0 THEN 0
                ELSE ROUND(blks_hit::numeric / (blks_hit + blks_read) * 100, 2)
            END as cache_hit_ratio
        FROM pg_stat_database 
        WHERE datname = current_database()
        """

        result = await client.fetch_raw(query)
        return float(result[0]["cache_hit_ratio"]) if result else 0.0

    async def get_index_hit_ratio(self) -> float:
        """Get index hit ratio for table access patterns"""
        client = await self.get_client()

        query = """
        SELECT 
            CASE 
                WHEN (idx_blks_hit + idx_blks_read) = 0 THEN 0
                ELSE ROUND(idx_blks_hit::numeric / (idx_blks_hit + idx_blks_read) * 100, 2)
            END as index_hit_ratio
        FROM pg_statio_user_indexes
        """

        result = await client.fetch_raw(query)
        if result:
            return float(result[0]["index_hit_ratio"])
        return 0.0

    async def get_active_connections(self) -> int:
        """Get current active connection count"""
        client = await self.get_client()

        query = """
        SELECT count(*) as active_connections
        FROM pg_stat_activity 
        WHERE state = 'active' AND datname = current_database()
        """

        result = await client.fetch_raw(query)
        return int(result[0]["active_connections"]) if result else 0

    async def get_database_size(self) -> float:
        """Get database size in MB"""
        client = await self.get_client()

        query = """
        SELECT 
            ROUND(pg_database_size(current_database()) / 1024.0 / 1024.0, 2) as size_mb
        """

        result = await client.fetch_raw(query)
        return float(result[0]["size_mb"]) if result else 0.0

    async def get_slow_queries(
        self, min_calls: int = 10
    ) -> list[QueryPerformanceMetric]:
        """Get slow queries from pg_stat_statements (if available).
        Fallback to basic query analysis if extension not installed.
        """
        client = await self.get_client()

        # Check if pg_stat_statements is available
        extension_check = """
        SELECT EXISTS (
            SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
        ) as has_extension
        """

        ext_result = await client.fetch_raw(extension_check)
        has_extension = ext_result[0]["has_extension"] if ext_result else False

        if has_extension:
            # Use pg_stat_statements for detailed query analysis
            query = """
            SELECT 
                query as query_text,
                calls,
                total_exec_time,
                mean_exec_time,
                max_exec_time,
                min_exec_time,
                rows
            FROM pg_stat_statements 
            WHERE calls >= %s
            AND query NOT LIKE '%%pg_stat_statements%%'
            ORDER BY mean_exec_time DESC 
            LIMIT 10
            """

            result = await client.fetch_raw(query, {"calls": min_calls})

            metrics = []
            for row in result:
                metrics.append(
                    QueryPerformanceMetric(
                        query_text=row["query_text"][:200] + "..."
                        if len(row["query_text"]) > 200
                        else row["query_text"],
                        calls=row["calls"],
                        total_exec_time=row["total_exec_time"],
                        mean_exec_time=row["mean_exec_time"],
                        max_exec_time=row["max_exec_time"],
                        min_exec_time=row["min_exec_time"],
                        rows_affected=row["rows"],
                        cache_hit_ratio=0.0,  # Not available in pg_stat_statements
                    )
                )

            return metrics
        # Fallback: Use client metrics
        client_stats = await client.get_performance_stats()

        # Create synthetic metrics from client data
        return [
            QueryPerformanceMetric(
                query_text="Client-tracked queries (pg_stat_statements not available)",
                calls=client_stats["total_queries"],
                total_exec_time=client_stats["avg_query_time_ms"]
                * client_stats["total_queries"],
                mean_exec_time=client_stats["avg_query_time_ms"],
                max_exec_time=0.0,
                min_exec_time=0.0,
                rows_affected=0,
                cache_hit_ratio=client_stats["cache_hit_ratio"],
            )
        ]

    async def take_performance_snapshot(self) -> DatabasePerformanceSnapshot:
        """Take a complete performance snapshot"""
        client = await self.get_client()

        # Gather all metrics concurrently for efficiency
        (
            cache_hit_ratio,
            index_hit_ratio,
            active_connections,
            database_size,
            slow_queries,
        ) = await asyncio.gather(
            self.get_cache_hit_ratio(),
            self.get_index_hit_ratio(),
            self.get_active_connections(),
            self.get_database_size(),
            self.get_slow_queries(),
        )

        # Get client performance stats
        client_stats = await client.get_performance_stats()

        snapshot = DatabasePerformanceSnapshot(
            timestamp=datetime.utcnow(),
            cache_hit_ratio=cache_hit_ratio,
            active_connections=active_connections,
            total_queries=client_stats["total_queries"],
            avg_query_time_ms=client_stats["avg_query_time_ms"],
            slow_queries_count=client_stats["slow_query_count"],
            top_slow_queries=slow_queries[:5],  # Top 5 slowest
            database_size_mb=database_size,
            index_hit_ratio=index_hit_ratio,
        )

        # Store snapshot for historical tracking
        self._snapshots.append(snapshot)

        # Keep only last 100 snapshots
        if len(self._snapshots) > 100:
            self._snapshots = self._snapshots[-100:]

        return snapshot

    async def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous monitoring"""
        self._monitoring = True

        while self._monitoring:
            try:
                snapshot = await self.take_performance_snapshot()

                # Log warning if performance targets not met
                if snapshot.cache_hit_ratio < 90:
                    print(
                        f"⚠️  Cache hit ratio below target: {snapshot.cache_hit_ratio:.1f}% (target: >90%)"
                    )

                if snapshot.avg_query_time_ms > 50:
                    print(
                        f"⚠️  Average query time above target: {snapshot.avg_query_time_ms:.1f}ms (target: <50ms)"
                    )

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(interval_seconds)

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self._monitoring = False

    async def get_performance_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get performance summary for the last N hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_snapshots = [s for s in self._snapshots if s.timestamp >= cutoff_time]

        if not recent_snapshots:
            return {"error": "No recent performance data available"}

        # Calculate averages
        avg_cache_hit = sum(s.cache_hit_ratio for s in recent_snapshots) / len(
            recent_snapshots
        )
        avg_query_time = sum(s.avg_query_time_ms for s in recent_snapshots) / len(
            recent_snapshots
        )
        avg_connections = sum(s.active_connections for s in recent_snapshots) / len(
            recent_snapshots
        )

        return {
            "period_hours": hours,
            "snapshots_analyzed": len(recent_snapshots),
            "avg_cache_hit_ratio": round(avg_cache_hit, 2),
            "avg_query_time_ms": round(avg_query_time, 2),
            "avg_active_connections": round(avg_connections, 1),
            "latest_database_size_mb": recent_snapshots[-1].database_size_mb,
            "performance_status": "GOOD"
            if avg_cache_hit >= 90 and avg_query_time <= 50
            else "NEEDS_ATTENTION",
            "target_compliance": {
                "cache_hit_ratio": avg_cache_hit >= 90,
                "query_time": avg_query_time <= 50,
            },
        }

    async def get_recommendations(self) -> list[str]:
        """Get performance optimization recommendations"""
        if not self._snapshots:
            return ["No performance data available. Run monitoring first."]

        latest = self._snapshots[-1]
        recommendations = []

        if latest.cache_hit_ratio < 90:
            recommendations.append(
                f"Cache hit ratio is {latest.cache_hit_ratio:.1f}% (target: >90%). Consider increasing shared_buffers."
            )

        if latest.avg_query_time_ms > 50:
            recommendations.append(
                f"Average query time is {latest.avg_query_time_ms:.1f}ms (target: <50ms). Review slow queries and add indexes."
            )

        if latest.index_hit_ratio < 95:
            recommendations.append(
                f"Index hit ratio is {latest.index_hit_ratio:.1f}% (target: >95%). Review query plans and index usage."
            )

        if latest.active_connections > 20:
            recommendations.append(
                f"High connection count: {latest.active_connections}. Consider connection pooling optimization."
            )

        if not recommendations:
            recommendations.append("✅ Performance targets are being met!")

        return recommendations


# Global monitor instance
_monitor: DatabasePerformanceMonitor | None = None


async def get_performance_monitor() -> DatabasePerformanceMonitor:
    """Get or create global performance monitor"""
    global _monitor
    if _monitor is None:
        _monitor = DatabasePerformanceMonitor()
    return _monitor
