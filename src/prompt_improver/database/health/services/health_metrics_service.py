"""Health Metrics Collection Service.

Provides comprehensive health metrics collection, performance tracking, and
query analysis with optimization recommendations. This service focuses on
database performance monitoring and analysis.

Features:
- Query performance analysis and slow query detection
- Cache performance monitoring and optimization
- Storage metrics and bloat detection
- Replication lag monitoring
- Lock analysis and deadlock detection
- Transaction metrics and long-running transaction detection
- I/O intensive query identification
"""

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from prompt_improver.core.common import get_logger
from prompt_improver.database.health.services.health_types import (
    QueryPerformanceMetrics,
)
from prompt_improver.shared.interfaces.protocols.database import SessionManagerProtocol

logger = get_logger(__name__)


class HealthMetricsService:
    """Service for comprehensive health metrics collection and analysis.

    This service provides detailed monitoring of database performance metrics,
    including query performance, cache efficiency, storage utilization, and more.
    """

    def __init__(self, session_manager: SessionManagerProtocol) -> None:
        """Initialize the health metrics service.

        Args:
            session_manager: Database session manager for executing queries
        """
        self.session_manager = session_manager

        # Configuration thresholds
        self.slow_query_threshold_ms = 1000.0       # 1 second
        self.replication_lag_threshold_seconds = 60.0  # 1 minute
        self.cache_hit_ratio_threshold = 95.0       # 95%
        self.lock_wait_threshold_ms = 5000.0        # 5 seconds

    async def collect_query_performance_metrics(self) -> dict[str, Any]:
        """Collect comprehensive query performance metrics.

        Returns:
            Dictionary containing detailed query performance analysis
        """
        logger.debug("Collecting query performance metrics")
        start_time = time.perf_counter()

        try:
            # Check if pg_stat_statements is available
            has_pg_stat_statements = await self._check_pg_stat_statements()

            if has_pg_stat_statements:
                # Collect all query-related metrics in parallel
                results = await asyncio.gather(
                    self.analyze_slow_queries(),
                    self.analyze_frequent_queries(),
                    self.analyze_io_intensive_queries(),
                    self.analyze_cache_performance(),
                    self._get_current_activity(),
                    return_exceptions=True,
                )

                slow_queries = results[0] if not isinstance(results[0], Exception) else []
                frequent_queries = results[1] if not isinstance(results[1], Exception) else []
                io_intensive = results[2] if not isinstance(results[2], Exception) else []
                cache_analysis = results[3] if not isinstance(results[3], Exception) else {}
                current_activity = results[4] if not isinstance(results[4], Exception) else []
            else:
                # Limited analysis without pg_stat_statements
                slow_queries = []
                frequent_queries = []
                io_intensive = []
                cache_analysis = {}
                current_activity = await self._get_current_activity()

            # Generate summaries and assessments
            performance_summary = self._generate_performance_summary(
                slow_queries, frequent_queries, cache_analysis
            )
            overall_assessment = self._assess_overall_performance(slow_queries, cache_analysis)
            recommendations = self._generate_optimization_recommendations(
                slow_queries, frequent_queries, cache_analysis, current_activity
            )

            metrics = QueryPerformanceMetrics(
                pg_stat_statements_available=has_pg_stat_statements,
                slow_queries=slow_queries[:10],  # Limit for performance
                frequent_queries=frequent_queries[:10],
                io_intensive_queries=io_intensive[:5],
                cache_performance=cache_analysis,
                current_activity=current_activity[:20],
                performance_summary=performance_summary,
                overall_assessment=overall_assessment,
                recommendations=recommendations,
                collection_time_ms=round((time.perf_counter() - start_time) * 1000, 2)
            )

            return self._metrics_to_dict(metrics)

        except Exception as e:
            logger.exception(f"Failed to collect query performance metrics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
                "collection_time_ms": (time.perf_counter() - start_time) * 1000,
            }

    async def analyze_slow_queries(self) -> list[dict[str, Any]]:
        """Analyze slow queries from pg_stat_statements.

        Returns:
            List of slow queries with performance details
        """
        try:
            async with self.session_manager.session_context() as session:
                query = text("""
                    SELECT
                        query,
                        calls,
                        total_exec_time,
                        mean_exec_time,
                        stddev_exec_time,
                        min_exec_time,
                        max_exec_time,
                        rows,
                        shared_blks_hit,
                        shared_blks_read,
                        shared_blks_dirtied,
                        shared_blks_written,
                        local_blks_hit,
                        local_blks_read,
                        temp_blks_read,
                        temp_blks_written,
                        blk_read_time,
                        blk_write_time
                    FROM pg_stat_statements
                    WHERE mean_exec_time > :threshold
                    ORDER BY mean_exec_time DESC
                    LIMIT 50
                """)

                result = await session.execute(
                    query, {"threshold": self.slow_query_threshold_ms}
                )
                rows = result.fetchall()

                slow_queries = []
                for row in rows:
                    query_stats = {
                        "query_text": row[0],
                        "calls": row[1],
                        "total_time_ms": float(row[2]),
                        "mean_time_ms": float(row[3]),
                        "stddev_time_ms": float(row[4]) if row[4] else 0.0,
                        "min_time_ms": float(row[5]),
                        "max_time_ms": float(row[6]),
                        "rows_returned": row[7],
                        "shared_blks_hit": row[8],
                        "shared_blks_read": row[9],
                        "cache_hit_ratio": self._calculate_cache_hit_ratio(row[8], row[9]),
                        "io_time_ratio": self._calculate_io_time_ratio(
                            row[16], row[17], row[2]
                        ),
                    }
                    slow_queries.append(query_stats)

                return slow_queries

        except Exception as e:
            logger.exception(f"Failed to analyze slow queries: {e}")
            return []

    async def analyze_frequent_queries(self) -> list[dict[str, Any]]:
        """Identify frequently executed queries from pg_stat_statements.

        Returns:
            List of frequently executed queries with statistics
        """
        try:
            async with self.session_manager.session_context() as session:
                query = text("""
                    SELECT
                        queryid,
                        query,
                        calls,
                        total_exec_time as total_time,
                        mean_exec_time as mean_time,
                        rows,
                        shared_blks_hit,
                        shared_blks_read,
                        (shared_blks_hit + shared_blks_read) as total_blocks
                    FROM pg_stat_statements
                    WHERE calls > 100
                    ORDER BY calls DESC
                    LIMIT 15
                """)

                result = await session.execute(query)
                frequent_queries = []

                for row in result:
                    query_info = {
                        "query_id": str(row[0]) if row[0] else None,
                        "query_text": row[1][:300] + "..." if row[1] and len(row[1]) > 300 else row[1],
                        "calls": int(row[2]),
                        "total_time_ms": float(row[3]),
                        "mean_time_ms": float(row[4]),
                        "avg_rows_per_call": float(row[5]) / int(row[2]) if int(row[2]) > 0 else 0,
                        "cache_hit_ratio": self._calculate_cache_hit_ratio(row[6], row[7]),
                        "total_blocks_accessed": int(row[8]),
                        "optimization_potential": (
                            "high" if row[4] > 100 and row[2] > 1000 else
                            "medium" if row[4] > 50 else "low"
                        ),
                    }
                    frequent_queries.append(query_info)

                return frequent_queries

        except Exception as e:
            logger.exception(f"Failed to analyze frequent queries: {e}")
            return []

    async def analyze_io_intensive_queries(self) -> list[dict[str, Any]]:
        """Identify I/O intensive queries from pg_stat_statements.

        Returns:
            List of I/O intensive queries with I/O statistics
        """
        try:
            async with self.session_manager.session_context() as session:
                query = text("""
                    SELECT
                        queryid,
                        query,
                        calls,
                        total_exec_time as total_time,
                        mean_exec_time as mean_time,
                        shared_blks_read,
                        shared_blks_written,
                        local_blks_read,
                        local_blks_written,
                        temp_blks_read,
                        temp_blks_written,
                        blk_read_time,
                        blk_write_time,
                        (blk_read_time + blk_write_time) as total_io_time
                    FROM pg_stat_statements
                    WHERE (blk_read_time + blk_write_time) > 0
                        AND total_exec_time > 0
                        AND ((blk_read_time + blk_write_time) / total_exec_time * 100) > 20
                    ORDER BY (blk_read_time + blk_write_time) DESC
                    LIMIT 15
                """)

                result = await session.execute(query)
                io_intensive = []

                for row in result:
                    total_io_time = float(row[13])
                    total_time = float(row[3])
                    io_ratio = total_io_time / total_time * 100 if total_time > 0 else 0

                    query_info = {
                        "query_id": str(row[0]) if row[0] else None,
                        "query_text": row[1][:200] + "..." if row[1] and len(row[1]) > 200 else row[1],
                        "calls": int(row[2]),
                        "total_time_ms": total_time,
                        "mean_time_ms": float(row[4]),
                        "io_time_ms": total_io_time,
                        "io_ratio_percent": io_ratio,
                        "blocks_read": int(row[5]),
                        "blocks_written": int(row[6]),
                        "temp_blocks_read": int(row[9]),
                        "temp_blocks_written": int(row[10]),
                        "read_time_ms": float(row[11]),
                        "write_time_ms": float(row[12]),
                        "optimization_priority": (
                            "critical" if io_ratio > 80 else
                            "high" if io_ratio > 50 else "medium"
                        ),
                    }
                    io_intensive.append(query_info)

                return io_intensive

        except Exception as e:
            logger.exception(f"Failed to analyze IO-intensive queries: {e}")
            return []

    async def analyze_cache_performance(self) -> dict[str, Any]:
        """Analyze database cache performance from PostgreSQL statistics.

        Returns:
            Dictionary with cache performance analysis
        """
        try:
            async with self.session_manager.session_context() as session:
                # Analyze table and index cache performance
                cache_query = text("""
                    SELECT
                        sum(heap_blks_read) as heap_read,
                        sum(heap_blks_hit) as heap_hit,
                        sum(idx_blks_read) as idx_read,
                        sum(idx_blks_hit) as idx_hit,
                        sum(toast_blks_read) as toast_read,
                        sum(toast_blks_hit) as toast_hit,
                        sum(tidx_blks_read) as tidx_read,
                        sum(tidx_blks_hit) as tidx_hit
                    FROM pg_statio_user_tables
                """)

                result = await session.execute(cache_query)
                row = result.fetchone()

                if row:
                    heap_read, heap_hit = (int(row[0] or 0), int(row[1] or 0))
                    idx_read, idx_hit = (int(row[2] or 0), int(row[3] or 0))
                    toast_read, toast_hit = (int(row[4] or 0), int(row[5] or 0))
                    tidx_read, tidx_hit = (int(row[6] or 0), int(row[7] or 0))

                    # Calculate hit ratios
                    total_heap = heap_read + heap_hit
                    total_idx = idx_read + idx_hit
                    total_toast = toast_read + toast_hit
                    total_tidx = tidx_read + tidx_hit

                    heap_hit_ratio = heap_hit / total_heap * 100 if total_heap > 0 else 100.0
                    idx_hit_ratio = idx_hit / total_idx * 100 if total_idx > 0 else 100.0
                    toast_hit_ratio = toast_hit / total_toast * 100 if total_toast > 0 else 100.0
                    tidx_hit_ratio = tidx_hit / total_tidx * 100 if total_tidx > 0 else 100.0

                    total_read = heap_read + idx_read + toast_read + tidx_read
                    total_hit = heap_hit + idx_hit + toast_hit + tidx_hit
                    overall_hit_ratio = (
                        total_hit / (total_read + total_hit) * 100
                        if total_read + total_hit > 0 else 100.0
                    )
                else:
                    heap_hit_ratio = idx_hit_ratio = toast_hit_ratio = tidx_hit_ratio = overall_hit_ratio = 100.0
                    total_read = total_hit = 0

                # Get shared buffers setting
                buffer_query = text("""
                    SELECT
                        setting as shared_buffers,
                        unit
                    FROM pg_settings
                    WHERE name = 'shared_buffers'
                """)

                buffer_result = await session.execute(buffer_query)
                buffer_row = buffer_result.fetchone()
                shared_buffers_setting = f"{buffer_row[0]} {buffer_row[1]}" if buffer_row else "unknown"

                cache_analysis = {
                    "overall_cache_hit_ratio": round(overall_hit_ratio, 2),
                    "heap_cache_hit_ratio": round(heap_hit_ratio, 2),
                    "index_cache_hit_ratio": round(idx_hit_ratio, 2),
                    "toast_cache_hit_ratio": round(toast_hit_ratio, 2),
                    "toast_index_cache_hit_ratio": round(tidx_hit_ratio, 2),
                    "cache_efficiency": (
                        "excellent" if overall_hit_ratio >= 99 else
                        "good" if overall_hit_ratio >= 95 else "poor"
                    ),
                    "total_cache_misses": total_read,
                    "total_cache_hits": total_hit,
                    "shared_buffers_setting": shared_buffers_setting,
                    "recommendations": [],
                }

                # Generate cache-specific recommendations
                if overall_hit_ratio < 95:
                    cache_analysis["recommendations"].append(
                        "Consider increasing shared_buffers to improve cache hit ratio"
                    )
                if heap_hit_ratio < 90:
                    cache_analysis["recommendations"].append(
                        "Low heap cache hit ratio - consider query optimization or more memory"
                    )
                if idx_hit_ratio < 95:
                    cache_analysis["recommendations"].append(
                        "Low index cache hit ratio - check index usage and size"
                    )

                return cache_analysis

        except Exception as e:
            logger.exception(f"Failed to analyze cache performance: {e}")
            return {
                "error": str(e),
                "overall_cache_hit_ratio": 0.0,
                "cache_efficiency": "unknown",
            }

    async def collect_storage_metrics(self) -> dict[str, Any]:
        """Collect storage-related metrics including table sizes and bloat.

        Returns:
            Dictionary with storage metrics and analysis
        """
        try:
            async with self.session_manager.session_context() as session:
                # Get database size
                db_size_query = text("""
                    SELECT pg_database_size(current_database()) as db_size_bytes
                """)
                result = await session.execute(db_size_query)
                row = result.fetchone()
                db_size_bytes = int(row[0]) if row else 0

                # Get table sizes
                table_sizes_query = text("""
                    SELECT
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size_pretty,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes,
                        pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size_pretty,
                        pg_relation_size(schemaname||'.'||tablename) as table_size_bytes,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) as index_size_pretty,
                        pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename) as index_size_bytes
                    FROM pg_tables
                    WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                    LIMIT 20
                """)

                result = await session.execute(table_sizes_query)
                table_sizes = []
                total_table_size = 0
                total_index_size = 0

                for row in result:
                    table_info = {
                        "schema": row[0],
                        "table": row[1],
                        "total_size_pretty": row[2],
                        "total_size_bytes": int(row[3]),
                        "table_size_pretty": row[4],
                        "table_size_bytes": int(row[5]),
                        "index_size_pretty": row[6],
                        "index_size_bytes": int(row[7]),
                    }
                    table_sizes.append(table_info)
                    total_table_size += table_info["table_size_bytes"]
                    total_index_size += table_info["index_size_bytes"]

                return {
                    "database_size_bytes": db_size_bytes,
                    "database_size_pretty": self._format_bytes(db_size_bytes),
                    "total_table_size_bytes": total_table_size,
                    "total_index_size_bytes": total_index_size,
                    "largest_tables": table_sizes,
                    "index_to_table_ratio": (
                        total_index_size / total_table_size * 100
                        if total_table_size > 0 else 0
                    ),
                }

        except Exception as e:
            logger.exception(f"Failed to collect storage metrics: {e}")
            return {"error": str(e)}

    async def collect_replication_metrics(self) -> dict[str, Any]:
        """Collect PostgreSQL replication metrics including lag monitoring.

        Returns:
            Dictionary with replication status and metrics
        """
        try:
            async with self.session_manager.session_context() as session:
                # Check if this is a replica
                is_replica_query = text("SELECT pg_is_in_recovery() as is_replica")
                result = await session.execute(is_replica_query)
                row = result.fetchone()
                is_replica = row[0] if row else False

                metrics = {
                    "is_replica": is_replica,
                    "replication_enabled": False,
                    "lag_seconds": 0.0,
                    "lag_bytes": 0,
                    "replica_count": 0,
                    "streaming_replicas": [],
                    "replication_slots": [],
                }

                if is_replica:
                    # Get replica lag information
                    lag_query = text("""
                        SELECT
                            CASE
                                WHEN pg_last_wal_receive_lsn() = pg_last_wal_replay_lsn() THEN 0
                                ELSE EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))
                            END as lag_seconds,
                            pg_wal_lsn_diff(pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn()) as lag_bytes
                    """)
                    result = await session.execute(lag_query)
                    row = result.fetchone()

                    if row:
                        metrics["lag_seconds"] = float(row[0] or 0)
                        metrics["lag_bytes"] = int(row[1] or 0)
                        metrics["replication_enabled"] = True
                else:
                    # Get replica information (primary server)
                    replica_query = text("""
                        SELECT
                            client_addr,
                            application_name,
                            state,
                            sent_lsn,
                            write_lsn,
                            flush_lsn,
                            replay_lsn,
                            write_lag,
                            flush_lag,
                            replay_lag,
                            sync_state
                        FROM pg_stat_replication
                    """)
                    result = await session.execute(replica_query)
                    replicas = []

                    for row in result:
                        replica_info = {
                            "client_addr": row[0],
                            "application_name": row[1],
                            "state": row[2],
                            "sent_lsn": str(row[3]),
                            "write_lsn": str(row[4]),
                            "flush_lsn": str(row[5]),
                            "replay_lsn": str(row[6]),
                            "write_lag_ms": float(row[7].total_seconds() * 1000) if row[7] else 0,
                            "flush_lag_ms": float(row[8].total_seconds() * 1000) if row[8] else 0,
                            "replay_lag_ms": float(row[9].total_seconds() * 1000) if row[9] else 0,
                            "sync_state": row[10],
                        }
                        replicas.append(replica_info)

                        # Track maximum lag
                        max_lag = max(
                            replica_info["write_lag_ms"],
                            replica_info["flush_lag_ms"],
                            replica_info["replay_lag_ms"],
                        )
                        metrics["lag_seconds"] = max(metrics["lag_seconds"], max_lag / 1000)

                    metrics["replica_count"] = len(replicas)
                    metrics["streaming_replicas"] = replicas
                    metrics["replication_enabled"] = len(replicas) > 0

                return metrics

        except Exception as e:
            logger.exception(f"Failed to collect replication metrics: {e}")
            return {"error": str(e)}

    async def collect_lock_metrics(self) -> dict[str, Any]:
        """Collect lock-related metrics including current locks and deadlocks.

        Returns:
            Dictionary with lock analysis and metrics
        """
        try:
            async with self.session_manager.session_context() as session:
                # Get current locks
                locks_query = text("""
                    SELECT
                        pl.locktype,
                        pl.database,
                        pl.relation,
                        pl.page,
                        pl.tuple,
                        pl.virtualxid,
                        pl.transactionid,
                        pl.classid,
                        pl.objid,
                        pl.objsubid,
                        pl.virtualtransaction,
                        pl.pid,
                        pl.mode,
                        pl.granted,
                        pa.query,
                        pa.state,
                        pa.backend_start,
                        pa.query_start,
                        pa.state_change,
                        EXTRACT(EPOCH FROM (now() - pa.query_start)) as query_duration_seconds
                    FROM pg_locks pl
                    LEFT JOIN pg_stat_activity pa ON pl.pid = pa.pid
                    WHERE pl.pid IS NOT NULL
                    ORDER BY pa.query_start
                """)

                result = await session.execute(locks_query)
                locks = []
                blocking_locks = 0
                long_running_locks = 0

                for row in result:
                    lock_info = {
                        "locktype": row[0],
                        "database": row[1],
                        "relation": row[2],
                        "pid": row[11],
                        "mode": row[12],
                        "granted": row[13],
                        "query": row[14][:100] + "..." if row[14] and len(row[14]) > 100 else row[14],
                        "state": row[15],
                        "query_duration_seconds": float(row[19]) if row[19] else 0,
                    }
                    locks.append(lock_info)

                    if not lock_info["granted"]:
                        blocking_locks += 1

                    if lock_info["query_duration_seconds"] > self.lock_wait_threshold_ms / 1000:
                        long_running_locks += 1

                # Get blocking queries
                blocking_query = text("""
                    SELECT
                        blocked_locks.pid AS blocked_pid,
                        blocked_activity.usename AS blocked_user,
                        blocking_locks.pid AS blocking_pid,
                        blocking_activity.usename AS blocking_user,
                        blocked_activity.query AS blocked_statement,
                        blocking_activity.query AS current_statement_in_blocking_process,
                        blocked_activity.application_name AS blocked_application,
                        blocking_activity.application_name AS blocking_application
                    FROM pg_catalog.pg_locks blocked_locks
                    JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
                    JOIN pg_catalog.pg_locks blocking_locks
                        ON blocking_locks.locktype = blocked_locks.locktype
                        AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
                        AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
                        AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
                        AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
                        AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
                        AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
                        AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
                        AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
                        AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
                        AND blocking_locks.pid != blocked_locks.pid
                    JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
                    WHERE NOT blocked_locks.granted
                """)

                result = await session.execute(blocking_query)
                blocking_queries = []

                for row in result:
                    blocking_info = {
                        "blocked_pid": row[0],
                        "blocked_user": row[1],
                        "blocking_pid": row[2],
                        "blocking_user": row[3],
                        "blocked_query": row[4][:100] + "..." if row[4] and len(row[4]) > 100 else row[4],
                        "blocking_query": row[5][:100] + "..." if row[5] and len(row[5]) > 100 else row[5],
                        "blocked_application": row[6],
                        "blocking_application": row[7],
                    }
                    blocking_queries.append(blocking_info)

                return {
                    "total_locks": len(locks),
                    "blocking_locks": blocking_locks,
                    "long_running_locks": long_running_locks,
                    "blocking_queries": blocking_queries,
                    "current_locks": locks[:20],  # Limit for performance
                    "lock_types_distribution": self._analyze_lock_distribution(locks),
                }

        except Exception as e:
            logger.exception(f"Failed to collect lock metrics: {e}")
            return {"error": str(e)}

    async def collect_transaction_metrics(self) -> dict[str, Any]:
        """Collect transaction-related metrics including commit/rollback rates.

        Returns:
            Dictionary with transaction analysis and metrics
        """
        try:
            async with self.session_manager.session_context() as session:
                # Get database transaction statistics
                db_stats_query = text("""
                    SELECT
                        xact_commit,
                        xact_rollback,
                        blks_read,
                        blks_hit,
                        tup_returned,
                        tup_fetched,
                        tup_inserted,
                        tup_updated,
                        tup_deleted,
                        conflicts,
                        temp_files,
                        temp_bytes,
                        deadlocks,
                        stats_reset
                    FROM pg_stat_database
                    WHERE datname = current_database()
                """)

                result = await session.execute(db_stats_query)
                row = result.fetchone()

                db_metrics = (
                    {
                        "commits": int(row[0]) if row else 0,
                        "rollbacks": int(row[1]) if row else 0,
                        "blocks_read": int(row[2]) if row else 0,
                        "blocks_hit": int(row[3]) if row else 0,
                        "tuples_returned": int(row[4]) if row else 0,
                        "tuples_fetched": int(row[5]) if row else 0,
                        "tuples_inserted": int(row[6]) if row else 0,
                        "tuples_updated": int(row[7]) if row else 0,
                        "tuples_deleted": int(row[8]) if row else 0,
                        "conflicts": int(row[9]) if row else 0,
                        "temp_files": int(row[10]) if row else 0,
                        "temp_bytes": int(row[11]) if row else 0,
                        "deadlocks": int(row[12]) if row else 0,
                        "stats_reset": row[13].isoformat() if row and row[13] else None,
                    }
                    if row else {}
                )

                # Get long-running transactions
                long_txn_query = text("""
                    SELECT
                        pid,
                        usename,
                        application_name,
                        client_addr,
                        backend_start,
                        xact_start,
                        query_start,
                        state_change,
                        state,
                        query,
                        EXTRACT(EPOCH FROM (now() - xact_start)) as txn_duration_seconds,
                        EXTRACT(EPOCH FROM (now() - query_start)) as query_duration_seconds
                    FROM pg_stat_activity
                    WHERE state IN ('active', 'idle in transaction')
                        AND xact_start IS NOT NULL
                        AND EXTRACT(EPOCH FROM (now() - xact_start)) > 300  -- 5 minutes
                    ORDER BY xact_start
                """)

                result = await session.execute(long_txn_query)
                long_transactions = []

                for row in result:
                    txn_info = {
                        "pid": row[0],
                        "username": row[1],
                        "application_name": row[2],
                        "client_addr": str(row[3]) if row[3] else None,
                        "transaction_start": row[5].isoformat() if row[5] else None,
                        "query_start": row[6].isoformat() if row[6] else None,
                        "state": row[8],
                        "query": row[9][:200] + "..." if row[9] and len(row[9]) > 200 else row[9],
                        "transaction_duration_seconds": float(row[10]) if row[10] else 0,
                        "query_duration_seconds": float(row[11]) if row[11] else 0,
                    }
                    long_transactions.append(txn_info)

                # Calculate transaction ratios
                total_transactions = db_metrics["commits"] + db_metrics["rollbacks"]
                commit_ratio = (
                    db_metrics["commits"] / total_transactions * 100
                    if total_transactions > 0 else 100.0
                )
                rollback_ratio = (
                    db_metrics["rollbacks"] / total_transactions * 100
                    if total_transactions > 0 else 0.0
                )

                return {
                    "database_stats": db_metrics,
                    "total_transactions": total_transactions,
                    "commit_ratio_percent": commit_ratio,
                    "rollback_ratio_percent": rollback_ratio,
                    "long_running_transactions": long_transactions,
                    "longest_transaction_seconds": max(
                        [t["transaction_duration_seconds"] for t in long_transactions],
                        default=0,
                    ),
                    "transaction_health": (
                        "excellent" if rollback_ratio < 5 else
                        "good" if rollback_ratio < 10 else "needs_attention"
                    ),
                }

        except Exception as e:
            logger.exception(f"Failed to collect transaction metrics: {e}")
            return {"error": str(e)}

    async def _check_pg_stat_statements(self) -> bool:
        """Check if pg_stat_statements extension is available.

        Returns:
            True if extension is available, False otherwise
        """
        try:
            async with self.session_manager.session_context() as session:
                query = text("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_extension
                        WHERE extname = 'pg_stat_statements'
                    ) as has_extension
                """)
                result = await session.execute(query)
                row = result.fetchone()
                return bool(row[0]) if row else False
        except Exception as e:
            logger.warning(f"Could not check for pg_stat_statements: {e}")
            return False

    async def _get_current_activity(self) -> list[dict[str, Any]]:
        """Get current database activity from pg_stat_activity.

        Returns:
            List of current active sessions and queries
        """
        try:
            async with self.session_manager.session_context() as session:
                query = text("""
                    SELECT
                        pid,
                        usename,
                        application_name,
                        client_addr,
                        backend_start,
                        xact_start,
                        query_start,
                        state_change,
                        state,
                        query,
                        wait_event_type,
                        wait_event
                    FROM pg_stat_activity
                    WHERE state != 'idle'
                        AND backend_type = 'client backend'
                        AND pid != pg_backend_pid()
                    ORDER BY
                        CASE
                            WHEN state = 'active' THEN 1
                            WHEN state = 'idle in transaction' THEN 2
                            ELSE 3
                        END,
                        query_start ASC NULLS LAST
                    LIMIT 50
                """)

                result = await session.execute(query)
                current_activity = []

                for row in result:
                    activity_info = {
                        "pid": int(row[0]),
                        "username": row[1],
                        "application_name": row[2],
                        "client_addr": str(row[3]) if row[3] else None,
                        "backend_start": row[4].isoformat() if row[4] else None,
                        "transaction_start": row[5].isoformat() if row[5] else None,
                        "query_start": row[6].isoformat() if row[6] else None,
                        "state": row[8],
                        "query": row[9][:500] + "..." if row[9] and len(row[9]) > 500 else row[9],
                        "wait_event_type": row[10],
                        "wait_event": row[11],
                    }

                    # Calculate durations
                    now = datetime.now(UTC)
                    if row[6]:  # query_start
                        query_duration = (now - row[6].replace(tzinfo=UTC)).total_seconds()
                        activity_info["query_duration_seconds"] = query_duration
                        activity_info["is_long_running"] = query_duration > 300

                    if row[5]:  # xact_start
                        transaction_duration = (now - row[5].replace(tzinfo=UTC)).total_seconds()
                        activity_info["transaction_duration_seconds"] = transaction_duration
                        activity_info["is_long_transaction"] = transaction_duration > 600

                    current_activity.append(activity_info)

                return current_activity

        except Exception as e:
            logger.exception(f"Failed to get current activity: {e}")
            return []

    def _calculate_cache_hit_ratio(self, blocks_hit: int, blocks_read: int) -> float:
        """Calculate cache hit ratio percentage."""
        total_blocks = blocks_hit + blocks_read
        if total_blocks == 0:
            return 100.0
        return blocks_hit / total_blocks * 100.0

    def _calculate_io_time_ratio(
        self, read_time: float, write_time: float, total_time: float
    ) -> float:
        """Calculate IO time as percentage of total time."""
        if total_time == 0:
            return 0.0
        io_time = read_time + write_time
        return io_time / total_time * 100.0

    def _analyze_lock_distribution(self, locks: list[dict[str, Any]]) -> dict[str, int]:
        """Analyze distribution of lock types."""
        distribution = {}
        for lock in locks:
            lock_type = lock.get("locktype", "unknown")
            distribution[lock_type] = distribution.get(lock_type, 0) + 1
        return distribution

    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes in human-readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"

    def _generate_performance_summary(
        self, slow_queries, frequent_queries, cache_analysis
    ) -> str:
        """Generate performance summary."""
        cache_efficiency = cache_analysis.get("cache_efficiency", "unknown")
        return (
            f"Performance Summary: {len(slow_queries)} slow queries, "
            f"{len(frequent_queries)} frequent queries, cache efficiency: {cache_efficiency}"
        )

    def _assess_overall_performance(self, slow_queries, cache_analysis) -> str:
        """Assess overall performance status."""
        if len(slow_queries) > 10:
            return "poor"
        if len(slow_queries) > 5:
            return "moderate"

        cache_hit_ratio = cache_analysis.get("overall_cache_hit_ratio", 100)
        if cache_hit_ratio < 90:
            return "poor"
        if cache_hit_ratio < 95:
            return "moderate"

        return "good"

    def _generate_optimization_recommendations(
        self, slow_queries, frequent_queries, cache_analysis, current_activity
    ) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if len(slow_queries) > 0:
            recommendations.append(
                f"Optimize {len(slow_queries)} slow queries for better performance"
            )

        if len(frequent_queries) > 5:
            recommendations.append(
                f"Review {len(frequent_queries)} frequently executed queries for optimization opportunities"
            )

        cache_hit_ratio = cache_analysis.get("overall_cache_hit_ratio", 100)
        if cache_hit_ratio < 95:
            recommendations.append(
                f"Improve cache hit ratio from {cache_hit_ratio:.1f}% to >95%"
            )

        long_running_count = len([
            activity for activity in current_activity
            if activity.get("is_long_running", False)
        ])
        if long_running_count > 0:
            recommendations.append(
                f"Investigate {long_running_count} long-running queries"
            )

        return recommendations

    def _metrics_to_dict(self, metrics: QueryPerformanceMetrics) -> dict[str, Any]:
        """Convert QueryPerformanceMetrics dataclass to dictionary."""
        return {
            "timestamp": metrics.timestamp.isoformat(),
            "pg_stat_statements_available": metrics.pg_stat_statements_available,
            "slow_queries": metrics.slow_queries,
            "frequent_queries": metrics.frequent_queries,
            "io_intensive_queries": metrics.io_intensive_queries,
            "cache_performance": metrics.cache_performance,
            "current_activity": metrics.current_activity,
            "performance_summary": metrics.performance_summary,
            "overall_assessment": metrics.overall_assessment,
            "recommendations": metrics.recommendations,
            "collection_time_ms": metrics.collection_time_ms,
            "slow_queries_count": len(metrics.slow_queries),
            "missing_indexes_count": 0,  # Placeholder for compatibility
        }
