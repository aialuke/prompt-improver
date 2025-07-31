"""
Comprehensive Database Health Monitor with Real PostgreSQL Metrics

Provides deep database health insights including:
- Connection pool metrics with age tracking
- Query performance analysis with execution plans
- Replication lag monitoring
- Storage utilization and table bloat detection
- Lock monitoring and deadlock detection
- Cache hit rates and buffer analysis
- Transaction metrics and longest transactions
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from ..unified_connection_manager import get_unified_manager, ManagerMode
from .connection_pool_monitor import ConnectionPoolMonitor
from .query_performance_analyzer import QueryPerformanceAnalyzer
from .index_health_assessor import IndexHealthAssessor
from .table_bloat_detector import TableBloatDetector

# Use consolidated common utilities
from ...core.common import get_logger

logger = get_logger(__name__)

@dataclass
class DatabaseHealthMetrics:
    """Comprehensive database health metrics"""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Connection Pool Metrics
    connection_pool: Dict[str, Any] = field(default_factory=dict)
    
    # Query Performance Metrics
    query_performance: Dict[str, Any] = field(default_factory=dict)
    
    # Replication Metrics
    replication: Dict[str, Any] = field(default_factory=dict)
    
    # Storage Metrics
    storage: Dict[str, Any] = field(default_factory=dict)
    
    # Lock Metrics
    locks: Dict[str, Any] = field(default_factory=dict)
    
    # Cache Metrics
    cache: Dict[str, Any] = field(default_factory=dict)
    
    # Transaction Metrics
    transactions: Dict[str, Any] = field(default_factory=dict)
    
    # Overall Health Score (0-100)
    health_score: float = 100.0
    
    # Health Issues
    issues: List[Dict[str, Any]] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[Dict[str, Any]] = field(default_factory=list)

class DatabaseHealthMonitor:
    """
    Comprehensive database health monitor using real PostgreSQL system catalogs
    and performance statistics to provide deep insights into database health.
    """
    
    def __init__(self, client: Optional[Any] = None):
        self.client = client
        self.pool_monitor = ConnectionPoolMonitor(client)
        self.query_analyzer = QueryPerformanceAnalyzer(client)
        self.index_assessor = IndexHealthAssessor(client)
        self.bloat_detector = TableBloatDetector(client)
        
        # Health thresholds
        self.connection_pool_utilization_threshold = 80.0
        self.slow_query_threshold_ms = 1000.0
        self.replication_lag_threshold_seconds = 60.0
        self.table_bloat_threshold_percent = 20.0
        self.index_bloat_threshold_percent = 30.0
        self.cache_hit_ratio_threshold = 95.0
        self.lock_wait_threshold_ms = 5000.0
        
        # Metrics history for trend analysis
        self._metrics_history: List[DatabaseHealthMetrics] = []
        self._max_history_size = 1000
    
    async def get_client(self):
        """Get database client"""
        if self.client is None:
            return get_unified_manager(ManagerMode.ASYNC_MODERN)
        return self.client
    
    async def collect_comprehensive_metrics(self) -> DatabaseHealthMetrics:
        """
        Collect comprehensive database health metrics from all monitors
        """
        logger.info("Starting comprehensive database health metrics collection")
        start_time = time.perf_counter()
        
        metrics = DatabaseHealthMetrics()
        
        try:
            # Collect metrics from all subsystems in parallel
            results = await asyncio.gather(
                self.pool_monitor.collect_connection_pool_metrics(),
                self.query_analyzer.analyze_query_performance(),
                self._collect_replication_metrics(),
                self._collect_storage_metrics(), 
                self._collect_lock_metrics(),
                self._collect_cache_metrics(),
                self._collect_transaction_metrics(),
                return_exceptions=True
            )
            
            # Process results
            metrics.connection_pool = results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])}
            metrics.query_performance = results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])}
            metrics.replication = results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])}
            metrics.storage = results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])}
            metrics.locks = results[4] if not isinstance(results[4], Exception) else {"error": str(results[4])}
            metrics.cache = results[5] if not isinstance(results[5], Exception) else {"error": str(results[5])}
            metrics.transactions = results[6] if not isinstance(results[6], Exception) else {"error": str(results[6])}
            
            # Calculate overall health score
            metrics.health_score = self._calculate_health_score(metrics)
            
            # Generate issues and recommendations
            metrics.issues = self._identify_health_issues(metrics)
            metrics.recommendations = self._generate_recommendations(metrics)
            
            # Store in history
            self._add_to_history(metrics)
            
            collection_time = (time.perf_counter() - start_time) * 1000
            logger.info(f"Database health metrics collection completed in {collection_time:.2f}ms")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect comprehensive database metrics: {e}")
            metrics.health_score = 0.0
            metrics.issues.append({
                "severity": "critical",
                "category": "monitoring",
                "message": f"Health monitoring system failure: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return metrics
    
    async def _collect_replication_metrics(self) -> Dict[str, Any]:
        """
        Collect PostgreSQL replication metrics including lag monitoring
        """
        async with get_session_context() as session:
            # Check if we're in a replication setup
            is_replica_query = text("""
                SELECT pg_is_in_recovery() as is_replica
            """)
            
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
                "replication_slots": []
            }
            
            if is_replica:
                # We're on a replica - get lag information
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
                # We're on a primary - get replica information
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
                        "sync_state": row[10]
                    }
                    replicas.append(replica_info)
                    # Calculate max lag for overall metrics
                    max_lag = max(replica_info["write_lag_ms"], replica_info["flush_lag_ms"], replica_info["replay_lag_ms"])
                    metrics["lag_seconds"] = max(metrics["lag_seconds"], max_lag / 1000)
                
                metrics["replica_count"] = len(replicas)
                metrics["streaming_replicas"] = replicas
                metrics["replication_enabled"] = len(replicas) > 0
                
                # Get replication slots
                slots_query = text("""
                    SELECT 
                        slot_name,
                        plugin,
                        slot_type,
                        datoid,
                        active,
                        restart_lsn,
                        confirmed_flush_lsn,
                        wal_status,
                        safe_wal_size
                    FROM pg_replication_slots
                """)
                
                result = await session.execute(slots_query)
                slots = []
                for row in result:
                    slot_info = {
                        "slot_name": row[0],
                        "plugin": row[1],
                        "slot_type": row[2],
                        "datoid": row[3],
                        "active": row[4],
                        "restart_lsn": str(row[5]) if row[5] else None,
                        "confirmed_flush_lsn": str(row[6]) if row[6] else None,
                        "wal_status": row[7],
                        "safe_wal_size": int(row[8]) if row[8] else None
                    }
                    slots.append(slot_info)
                
                metrics["replication_slots"] = slots
            
            return metrics
    
    async def _collect_storage_metrics(self) -> Dict[str, Any]:
        """
        Collect storage-related metrics including table sizes and bloat
        """
        async with get_session_context() as session:
            # Database size
            db_size_query = text("""
                SELECT pg_database_size(current_database()) as db_size_bytes
            """)
            
            result = await session.execute(db_size_query)
            row = result.fetchone()
            db_size_bytes = int(row[0]) if row else 0
            
            # Table sizes (top 20 largest tables)
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
                    "index_size_bytes": int(row[7])
                }
                table_sizes.append(table_info)
                total_table_size += table_info["table_size_bytes"]
                total_index_size += table_info["index_size_bytes"]
            
            # Get bloat information
            bloat_metrics = await self.bloat_detector.detect_table_bloat()
            
            # Disk usage for the database cluster
            disk_usage_query = text("""
                SELECT 
                    setting as data_directory
                FROM pg_settings 
                WHERE name = 'data_directory'
            """)
            
            result = await session.execute(disk_usage_query)
            row = result.fetchone()
            data_directory = row[0] if row else None
            
            return {
                "database_size_bytes": db_size_bytes,
                "database_size_pretty": self._format_bytes(db_size_bytes),
                "total_table_size_bytes": total_table_size,
                "total_index_size_bytes": total_index_size,
                "largest_tables": table_sizes,
                "bloat_metrics": bloat_metrics,
                "data_directory": data_directory,
                "index_to_table_ratio": (total_index_size / total_table_size * 100) if total_table_size > 0 else 0
            }
    
    async def _collect_lock_metrics(self) -> Dict[str, Any]:
        """
        Collect lock-related metrics including current locks and deadlocks
        """
        async with get_session_context() as session:
            # Current locks
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
                    "query_duration_seconds": float(row[19]) if row[19] else 0
                }
                locks.append(lock_info)
                
                if not lock_info["granted"]:
                    blocking_locks += 1
                
                if lock_info["query_duration_seconds"] > self.lock_wait_threshold_ms / 1000:
                    long_running_locks += 1
            
            # Blocking queries
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
                    "blocking_application": row[7]
                }
                blocking_queries.append(blocking_info)
            
            return {
                "total_locks": len(locks),
                "blocking_locks": blocking_locks,
                "long_running_locks": long_running_locks,
                "blocking_queries": blocking_queries,
                "current_locks": locks[:20],  # Limit to 20 most recent
                "lock_types_distribution": self._analyze_lock_distribution(locks)
            }
    
    async def _collect_cache_metrics(self) -> Dict[str, Any]:
        """
        Collect cache-related metrics including buffer cache and query plan cache
        """
        async with get_session_context() as session:
            # Buffer cache hit ratio
            buffer_cache_query = text("""
                SELECT 
                    sum(heap_blks_read) as heap_read,
                    sum(heap_blks_hit) as heap_hit,
                    sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) * 100 as hit_ratio
                FROM pg_statio_user_tables
                WHERE heap_blks_read + heap_blks_hit > 0
            """)
            
            result = await session.execute(buffer_cache_query)
            row = result.fetchone()
            
            buffer_metrics = {
                "heap_blocks_read": int(row[0]) if row and row[0] else 0,
                "heap_blocks_hit": int(row[1]) if row and row[1] else 0,
                "hit_ratio_percent": float(row[2]) if row and row[2] else 0.0
            }
            
            # Index cache hit ratio
            index_cache_query = text("""
                SELECT 
                    sum(idx_blks_read) as idx_read,
                    sum(idx_blks_hit) as idx_hit,
                    CASE 
                        WHEN sum(idx_blks_read) + sum(idx_blks_hit) = 0 THEN 100
                        ELSE sum(idx_blks_hit) / (sum(idx_blks_hit) + sum(idx_blks_read)) * 100 
                    END as hit_ratio
                FROM pg_statio_user_indexes
            """)
            
            result = await session.execute(index_cache_query)
            row = result.fetchone()
            
            index_metrics = {
                "index_blocks_read": int(row[0]) if row and row[0] else 0,
                "index_blocks_hit": int(row[1]) if row and row[1] else 0,
                "hit_ratio_percent": float(row[2]) if row and row[2] else 100.0
            }
            
            # Shared buffer statistics
            shared_buffers_query = text("""
                SELECT 
                    setting as shared_buffers_setting,
                    unit
                FROM pg_settings 
                WHERE name = 'shared_buffers'
            """)
            
            result = await session.execute(shared_buffers_query)
            row = result.fetchone()
            shared_buffers_setting = f"{row[0]}{row[1]}" if row else "unknown"
            
            # Background writer statistics
            bgwriter_query = text("""
                SELECT 
                    checkpoints_timed,
                    checkpoints_req,
                    checkpoint_write_time,
                    checkpoint_sync_time,
                    buffers_checkpoint,
                    buffers_clean,
                    maxwritten_clean,
                    buffers_backend,
                    buffers_backend_fsync,
                    buffers_alloc,
                    stats_reset
                FROM pg_stat_bgwriter
            """)
            
            result = await session.execute(bgwriter_query)
            row = result.fetchone()
            
            bgwriter_metrics = {
                "checkpoints_timed": int(row[0]) if row else 0,
                "checkpoints_req": int(row[1]) if row else 0,
                "checkpoint_write_time_ms": float(row[2]) if row else 0,
                "checkpoint_sync_time_ms": float(row[3]) if row else 0,
                "buffers_checkpoint": int(row[4]) if row else 0,
                "buffers_clean": int(row[5]) if row else 0,
                "buffers_backend": int(row[7]) if row else 0,
                "buffers_alloc": int(row[9]) if row else 0,
                "stats_reset": row[10].isoformat() if row and row[10] else None
            } if row else {}
            
            # Calculate overall cache hit ratio
            total_reads = buffer_metrics["heap_blocks_read"] + index_metrics["index_blocks_read"]
            total_hits = buffer_metrics["heap_blocks_hit"] + index_metrics["index_blocks_hit"]
            overall_hit_ratio = (total_hits / (total_hits + total_reads) * 100) if (total_hits + total_reads) > 0 else 100.0
            
            return {
                "buffer_cache": buffer_metrics,
                "index_cache": index_metrics,
                "overall_cache_hit_ratio_percent": overall_hit_ratio,
                "shared_buffers_setting": shared_buffers_setting,
                "bgwriter_stats": bgwriter_metrics,
                "cache_efficiency": "excellent" if overall_hit_ratio >= 95 else "good" if overall_hit_ratio >= 90 else "needs_attention"
            }
    
    async def _collect_transaction_metrics(self) -> Dict[str, Any]:
        """
        Collect transaction-related metrics including commit/rollback rates
        """
        async with get_session_context() as session:
            # Database transaction statistics
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
            
            db_metrics = {
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
                "stats_reset": row[13].isoformat() if row and row[13] else None
            } if row else {}
            
            # Long-running transactions
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
                    "query_duration_seconds": float(row[11]) if row[11] else 0
                }
                long_transactions.append(txn_info)
            
            # Calculate transaction rates and ratios
            total_transactions = db_metrics["commits"] + db_metrics["rollbacks"]
            commit_ratio = (db_metrics["commits"] / total_transactions * 100) if total_transactions > 0 else 100.0
            rollback_ratio = (db_metrics["rollbacks"] / total_transactions * 100) if total_transactions > 0 else 0.0
            
            return {
                "database_stats": db_metrics,
                "total_transactions": total_transactions,
                "commit_ratio_percent": commit_ratio,
                "rollback_ratio_percent": rollback_ratio,
                "long_running_transactions": long_transactions,
                "longest_transaction_seconds": max([t["transaction_duration_seconds"] for t in long_transactions], default=0),
                "transaction_health": "excellent" if rollback_ratio < 5 else "good" if rollback_ratio < 10 else "needs_attention"
            }
    
    def _calculate_health_score(self, metrics: DatabaseHealthMetrics) -> float:
        """
        Calculate overall health score based on all metrics (0-100)
        """
        score = 100.0
        
        # Connection pool health (20% weight)
        pool_metrics = metrics.connection_pool
        if isinstance(pool_metrics, dict) and "utilization_percent" in pool_metrics:
            utilization = pool_metrics["utilization_percent"]
            if utilization > self.connection_pool_utilization_threshold:
                score -= 10 * (utilization - self.connection_pool_utilization_threshold) / 20
        
        # Query performance health (25% weight)
        query_metrics = metrics.query_performance
        if isinstance(query_metrics, dict) and "slow_queries_count" in query_metrics:
            slow_queries = query_metrics["slow_queries_count"]
            if slow_queries > 0:
                score -= min(15, slow_queries * 2)  # Max 15 point deduction
        
        # Replication lag health (15% weight)
        replication_metrics = metrics.replication
        if isinstance(replication_metrics, dict) and "lag_seconds" in replication_metrics:
            lag_seconds = replication_metrics["lag_seconds"]
            if lag_seconds > self.replication_lag_threshold_seconds:
                score -= min(10, (lag_seconds - self.replication_lag_threshold_seconds) / 60 * 5)
        
        # Cache hit ratio health (20% weight)
        cache_metrics = metrics.cache
        if isinstance(cache_metrics, dict) and "overall_cache_hit_ratio_percent" in cache_metrics:
            hit_ratio = cache_metrics["overall_cache_hit_ratio_percent"]
            if hit_ratio < self.cache_hit_ratio_threshold:
                score -= (self.cache_hit_ratio_threshold - hit_ratio) * 2
        
        # Lock health (10% weight)
        lock_metrics = metrics.locks
        if isinstance(lock_metrics, dict):
            blocking_locks = lock_metrics.get("blocking_locks", 0)
            long_running_locks = lock_metrics.get("long_running_locks", 0)
            score -= min(5, blocking_locks + long_running_locks)
        
        # Transaction health (10% weight)
        txn_metrics = metrics.transactions
        if isinstance(txn_metrics, dict) and "rollback_ratio_percent" in txn_metrics:
            rollback_ratio = txn_metrics["rollback_ratio_percent"]
            if rollback_ratio > 10:  # More than 10% rollbacks is concerning
                score -= min(5, (rollback_ratio - 10) / 2)
        
        return max(0.0, min(100.0, score))
    
    def _identify_health_issues(self, metrics: DatabaseHealthMetrics) -> List[Dict[str, Any]]:
        """
        Identify specific health issues based on metrics
        """
        issues = []
        
        # Connection pool issues
        pool_metrics = metrics.connection_pool
        if isinstance(pool_metrics, dict):
            utilization = pool_metrics.get("utilization_percent", 0)
            if utilization > 90:
                issues.append({
                    "severity": "critical",
                    "category": "connection_pool",
                    "message": f"Connection pool utilization very high: {utilization:.1f}%",
                    "metric_value": utilization,
                    "threshold": 90.0
                })
            elif utilization > self.connection_pool_utilization_threshold:
                issues.append({
                    "severity": "warning",
                    "category": "connection_pool",
                    "message": f"Connection pool utilization high: {utilization:.1f}%",
                    "metric_value": utilization,
                    "threshold": self.connection_pool_utilization_threshold
                })
        
        # Query performance issues
        query_metrics = metrics.query_performance
        if isinstance(query_metrics, dict):
            slow_queries = query_metrics.get("slow_queries_count", 0)
            if slow_queries > 10:
                issues.append({
                    "severity": "critical",
                    "category": "query_performance",
                    "message": f"High number of slow queries: {slow_queries}",
                    "metric_value": slow_queries,
                    "threshold": 10
                })
            elif slow_queries > 0:
                issues.append({
                    "severity": "warning",
                    "category": "query_performance",
                    "message": f"Slow queries detected: {slow_queries}",
                    "metric_value": slow_queries,
                    "threshold": 1
                })
        
        # Cache hit ratio issues
        cache_metrics = metrics.cache
        if isinstance(cache_metrics, dict):
            hit_ratio = cache_metrics.get("overall_cache_hit_ratio_percent", 100)
            if hit_ratio < 90:
                issues.append({
                    "severity": "critical",
                    "category": "cache_performance",
                    "message": f"Cache hit ratio very low: {hit_ratio:.1f}%",
                    "metric_value": hit_ratio,
                    "threshold": 90.0
                })
            elif hit_ratio < self.cache_hit_ratio_threshold:
                issues.append({
                    "severity": "warning",
                    "category": "cache_performance",
                    "message": f"Cache hit ratio below target: {hit_ratio:.1f}%",
                    "metric_value": hit_ratio,
                    "threshold": self.cache_hit_ratio_threshold
                })
        
        # Replication lag issues
        replication_metrics = metrics.replication
        if isinstance(replication_metrics, dict) and replication_metrics.get("replication_enabled"):
            lag_seconds = replication_metrics.get("lag_seconds", 0)
            if lag_seconds > 300:  # 5 minutes
                issues.append({
                    "severity": "critical",
                    "category": "replication",
                    "message": f"Replication lag very high: {lag_seconds:.1f} seconds",
                    "metric_value": lag_seconds,
                    "threshold": 300.0
                })
            elif lag_seconds > self.replication_lag_threshold_seconds:
                issues.append({
                    "severity": "warning",
                    "category": "replication",
                    "message": f"Replication lag elevated: {lag_seconds:.1f} seconds",
                    "metric_value": lag_seconds,
                    "threshold": self.replication_lag_threshold_seconds
                })
        
        # Lock issues
        lock_metrics = metrics.locks
        if isinstance(lock_metrics, dict):
            blocking_locks = lock_metrics.get("blocking_locks", 0)
            if blocking_locks > 0:
                issues.append({
                    "severity": "warning" if blocking_locks < 5 else "critical",
                    "category": "locks",
                    "message": f"Blocking locks detected: {blocking_locks}",
                    "metric_value": blocking_locks,
                    "threshold": 0
                })
        
        # Transaction issues
        txn_metrics = metrics.transactions
        if isinstance(txn_metrics, dict):
            rollback_ratio = txn_metrics.get("rollback_ratio_percent", 0)
            if rollback_ratio > 20:
                issues.append({
                    "severity": "critical",
                    "category": "transactions",
                    "message": f"High rollback ratio: {rollback_ratio:.1f}%",
                    "metric_value": rollback_ratio,
                    "threshold": 20.0
                })
            elif rollback_ratio > 10:
                issues.append({
                    "severity": "warning",
                    "category": "transactions",
                    "message": f"Elevated rollback ratio: {rollback_ratio:.1f}%",
                    "metric_value": rollback_ratio,
                    "threshold": 10.0
                })
            
            long_txns = len(txn_metrics.get("long_running_transactions", []))
            if long_txns > 0:
                issues.append({
                    "severity": "warning",
                    "category": "transactions",
                    "message": f"Long-running transactions: {long_txns}",
                    "metric_value": long_txns,
                    "threshold": 0
                })
        
        return issues
    
    def _generate_recommendations(self, metrics: DatabaseHealthMetrics) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations based on metrics and identified issues
        """
        recommendations = []
        
        # Connection pool recommendations
        pool_metrics = metrics.connection_pool
        if isinstance(pool_metrics, dict):
            utilization = pool_metrics.get("utilization_percent", 0)
            if utilization > 80:
                recommendations.append({
                    "category": "connection_pool",
                    "priority": "high" if utilization > 90 else "medium",
                    "action": "increase_pool_size",
                    "description": f"Consider increasing connection pool size from {pool_metrics.get('pool_size', 'unknown')} connections",
                    "expected_impact": "Reduced connection wait times and improved throughput"
                })
            
            if pool_metrics.get("waiting_requests", 0) > 0:
                recommendations.append({
                    "category": "connection_pool",
                    "priority": "high",
                    "action": "optimize_connection_usage",
                    "description": "Implement connection pooling best practices and review long-running queries",
                    "expected_impact": "Reduced connection contention"
                })
        
        # Query performance recommendations
        query_metrics = metrics.query_performance
        if isinstance(query_metrics, dict):
            slow_queries = query_metrics.get("slow_queries_count", 0)
            if slow_queries > 0:
                recommendations.append({
                    "category": "query_performance",
                    "priority": "high",
                    "action": "optimize_slow_queries",
                    "description": f"Analyze and optimize {slow_queries} slow queries identified",
                    "expected_impact": "Improved response times and reduced resource usage"
                })
            
            if query_metrics.get("missing_indexes_count", 0) > 0:
                recommendations.append({
                    "category": "indexing",
                    "priority": "medium",
                    "action": "add_missing_indexes",
                    "description": "Consider adding indexes for frequently queried columns",
                    "expected_impact": "Faster query execution and reduced I/O"
                })
        
        # Cache recommendations
        cache_metrics = metrics.cache
        if isinstance(cache_metrics, dict):
            hit_ratio = cache_metrics.get("overall_cache_hit_ratio_percent", 100)
            if hit_ratio < 95:
                recommendations.append({
                    "category": "cache_tuning",
                    "priority": "medium",
                    "action": "increase_shared_buffers",
                    "description": f"Consider increasing shared_buffers (current hit ratio: {hit_ratio:.1f}%)",
                    "expected_impact": "Improved cache hit ratio and reduced I/O"
                })
        
        # Storage recommendations
        storage_metrics = metrics.storage
        if isinstance(storage_metrics, dict) and "bloat_metrics" in storage_metrics:
            bloat_info = storage_metrics["bloat_metrics"]
            if isinstance(bloat_info, dict):
                bloated_tables = bloat_info.get("bloated_tables_count", 0)
                if bloated_tables > 0:
                    recommendations.append({
                        "category": "maintenance",
                        "priority": "medium",
                        "action": "vacuum_analyze",
                        "description": f"Run VACUUM ANALYZE on {bloated_tables} bloated tables",
                        "expected_impact": "Reclaimed storage space and improved query performance"
                    })
        
        # Replication recommendations
        replication_metrics = metrics.replication
        if isinstance(replication_metrics, dict) and replication_metrics.get("replication_enabled"):
            lag_seconds = replication_metrics.get("lag_seconds", 0)
            if lag_seconds > 60:
                recommendations.append({
                    "category": "replication",
                    "priority": "high" if lag_seconds > 300 else "medium",
                    "action": "investigate_replication_lag",
                    "description": f"Investigate replication lag of {lag_seconds:.1f} seconds",
                    "expected_impact": "Improved data consistency and reduced lag"
                })
        
        return recommendations
    
    def _analyze_lock_distribution(self, locks: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Analyze distribution of lock types
        """
        distribution = {}
        for lock in locks:
            lock_type = lock.get("locktype", "unknown")
            distribution[lock_type] = distribution.get(lock_type, 0) + 1
        return distribution
    
    def _format_bytes(self, bytes_value: int) -> str:
        """
        Format bytes in human-readable format
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"
    
    def _add_to_history(self, metrics: DatabaseHealthMetrics) -> None:
        """
        Add metrics to history for trend analysis
        """
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self._max_history_size:
            self._metrics_history = self._metrics_history[-self._max_history_size:]
    
    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get health trends over the specified time period
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_metrics = [
            m for m in self._metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if len(recent_metrics) < 2:
            return {"status": "insufficient_data", "message": f"Need at least 2 data points in last {hours} hours"}
        
        # Calculate trends
        health_scores = [m.health_score for m in recent_metrics]
        connection_utilizations = []
        cache_hit_ratios = []
        
        for m in recent_metrics:
            if isinstance(m.connection_pool, dict) and "utilization_percent" in m.connection_pool:
                connection_utilizations.append(m.connection_pool["utilization_percent"])
            
            if isinstance(m.cache, dict) and "overall_cache_hit_ratio_percent" in m.cache:
                cache_hit_ratios.append(m.cache["overall_cache_hit_ratio_percent"])
        
        def calculate_trend(values: List[float]) -> str:
            if len(values) < 2:
                return "unknown"
            
            recent_avg = sum(values[-3:]) / len(values[-3:]) if len(values) >= 3 else values[-1]
            earlier_avg = sum(values[:3]) / len(values[:3]) if len(values) >= 3 else values[0]
            
            if recent_avg > earlier_avg * 1.05:
                return "improving"
            elif recent_avg < earlier_avg * 0.95:
                return "degrading"
            else:
                return "stable"
        
        return {
            "period_hours": hours,
            "data_points": len(recent_metrics),
            "health_score_trend": calculate_trend(health_scores),
            "current_health_score": health_scores[-1] if health_scores else 0,
            "avg_health_score": sum(health_scores) / len(health_scores) if health_scores else 0,
            "connection_utilization_trend": calculate_trend(connection_utilizations),
            "cache_hit_ratio_trend": calculate_trend(cache_hit_ratios),
            "summary": self._generate_trend_summary(recent_metrics)
        }
    
    def _generate_trend_summary(self, recent_metrics: List[DatabaseHealthMetrics]) -> str:
        """
        Generate a human-readable trend summary
        """
        if not recent_metrics:
            return "No data available"
        
        latest = recent_metrics[-1]
        issues_count = len(latest.issues)
        
        if latest.health_score >= 90:
            health_status = "excellent"
        elif latest.health_score >= 75:
            health_status = "good"
        elif latest.health_score >= 50:
            health_status = "fair"
        else:
            health_status = "poor"
        
        return f"Database health is {health_status} (score: {latest.health_score:.1f}) with {issues_count} active issues."

# Global health monitor instance
_health_monitor: Optional[DatabaseHealthMonitor] = None

def get_database_health_monitor() -> DatabaseHealthMonitor:
    """Get or create global database health monitor"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = DatabaseHealthMonitor()
    return _health_monitor