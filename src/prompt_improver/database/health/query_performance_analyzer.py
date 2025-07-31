"""
Query Performance Analyzer with pg_stat_statements Integration

Provides comprehensive query performance analysis including:
- Slow query detection with execution plans
- Query statistics from pg_stat_statements
- Index usage analysis
- Query optimization recommendations
- Performance trend analysis
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

from sqlalchemy import text

# psycopg_client removed in Phase 1 - using unified_connection_manager instead
from ..unified_connection_manager import get_unified_manager, ManagerMode
from .. import get_session_context

logger = logging.getLogger(__name__)

@dataclass
class QueryStatistics:
    """Query performance statistics"""
    query_id: Optional[str] = None
    query_text: Optional[str] = None
    calls: int = 0
    total_time_ms: float = 0.0
    mean_time_ms: float = 0.0
    stddev_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0
    rows_returned: int = 0
    rows_affected: int = 0
    shared_blks_hit: int = 0
    shared_blks_read: int = 0
    shared_blks_dirtied: int = 0
    shared_blks_written: int = 0
    local_blks_hit: int = 0
    local_blks_read: int = 0
    temp_blks_read: int = 0
    temp_blks_written: int = 0
    blk_read_time_ms: float = 0.0
    blk_write_time_ms: float = 0.0
    
    @property
    def cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio for this query"""
        total_blocks = self.shared_blks_hit + self.shared_blks_read
        return (self.shared_blks_hit / total_blocks * 100) if total_blocks > 0 else 100.0
    
    @property
    def io_time_ratio(self) -> float:
        """Calculate I/O time as percentage of total time"""
        total_io_time = self.blk_read_time_ms + self.blk_write_time_ms
        return (total_io_time / self.total_time_ms * 100) if self.total_time_ms > 0 else 0.0

@dataclass
class ExecutionPlan:
    """Query execution plan information"""
    query_text: str
    plan_json: Optional[Dict[str, Any]] = None
    plan_text: Optional[str] = None
    execution_time_ms: float = 0.0
    planning_time_ms: float = 0.0
    total_cost: float = 0.0
    rows_estimate: int = 0
    width_estimate: int = 0
    
    # Plan analysis
    has_seq_scan: bool = False
    has_index_scan: bool = False
    has_bitmap_scan: bool = False
    has_nested_loop: bool = False
    has_hash_join: bool = False
    has_merge_join: bool = False
    has_sort: bool = False
    has_aggregate: bool = False
    
    expensive_operations: List[str] = field(default_factory=list)
    missing_indexes_suggestions: List[str] = field(default_factory=list)

class QueryPerformanceAnalyzer:
    """
    Analyze query performance using pg_stat_statements and execution plans
    """
    
    def __init__(self, client: Optional[Any] = None):
        self.client = client
        
        # Performance thresholds
        self.slow_query_threshold_ms = 1000.0  # 1 second
        self.frequent_query_threshold = 100    # 100+ calls
        self.high_io_threshold_percent = 50.0  # 50% of time in I/O
        self.low_cache_hit_threshold = 90.0    # Below 90% cache hit
        
        # Track query patterns
        self._query_patterns = {}
        self._analysis_cache = {}
    
    async def get_client(self):
        """Get database client"""
        if self.client is None:
            return get_unified_manager(ManagerMode.ASYNC_MODERN)
        return self.client
    
    async def analyze_query_performance(self) -> Dict[str, Any]:
        """
        Comprehensive query performance analysis
        """
        logger.debug("Starting query performance analysis")
        start_time = time.perf_counter()
        
        try:
            # Check if pg_stat_statements is available
            pg_stat_available = await self._check_pg_stat_statements()
            
            if pg_stat_available:
                # Comprehensive analysis with pg_stat_statements
                results = await asyncio.gather(
                    self._analyze_slow_queries(),
                    self._analyze_frequent_queries(),
                    self._analyze_io_intensive_queries(),
                    self._analyze_cache_performance(),
                    self._get_current_activity(),
                    return_exceptions=True
                )
                
                # Type-safe exception handling for asyncio.gather results
                slow_queries = cast(List[Dict[str, Any]], results[0] if not isinstance(results[0], Exception) else [])
                frequent_queries = cast(List[Dict[str, Any]], results[1] if not isinstance(results[1], Exception) else [])
                io_intensive = cast(List[Dict[str, Any]], results[2] if not isinstance(results[2], Exception) else [])
                cache_analysis = cast(Dict[str, Any], results[3] if not isinstance(results[3], Exception) else {})
                current_activity = cast(List[Dict[str, Any]], results[4] if not isinstance(results[4], Exception) else [])
                
            else:
                # Fallback analysis without pg_stat_statements
                logger.warning("pg_stat_statements not available, using limited analysis")
                slow_queries = []
                frequent_queries = []
                io_intensive = []
                cache_analysis = {}
                current_activity = await self._get_current_activity()
            
            # Generate comprehensive report
            analysis_report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pg_stat_statements_available": pg_stat_available,
                "slow_queries": {
                    "count": len(slow_queries),
                    "threshold_ms": self.slow_query_threshold_ms,
                    "queries": slow_queries[:10]  # Top 10 slowest
                },
                "frequent_queries": {
                    "count": len(frequent_queries),
                    "threshold_calls": self.frequent_query_threshold,
                    "queries": frequent_queries[:10]  # Top 10 most frequent
                },
                "io_intensive_queries": {
                    "count": len(io_intensive),
                    "threshold_percent": self.high_io_threshold_percent,
                    "queries": io_intensive[:10]  # Top 10 I/O intensive
                },
                "cache_performance": cache_analysis,
                "current_activity": {
                    "active_queries": len([q for q in current_activity if q.get("state") == "active"]),
                    "long_running_queries": len([q for q in current_activity if q.get("duration_seconds", 0) > 60]),
                    "blocked_queries": len([q for q in current_activity if q.get("wait_event_type") and "Lock" in q.get("wait_event_type", "")]),
                    "queries": current_activity[:5]  # Top 5 current queries
                },
                "performance_summary": self._generate_performance_summary(slow_queries, frequent_queries, io_intensive, cache_analysis),
                "recommendations": self._generate_optimization_recommendations(slow_queries, frequent_queries, io_intensive, cache_analysis)
            }
            
            collection_time = (time.perf_counter() - start_time) * 1000
            analysis_report["analysis_time_ms"] = round(collection_time, 2)
            
            return analysis_report
            
        except Exception as e:
            logger.error(f"Query performance analysis failed: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "analysis_time_ms": (time.perf_counter() - start_time) * 1000
            }
    
    async def _check_pg_stat_statements(self) -> bool:
        """
        Check if pg_stat_statements extension is available and enabled
        """
        try:
            async with get_session_context() as session:
                # Check if extension exists
                check_query = text("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
                    ) as extension_exists
                """)
                
                result = await session.execute(check_query)
                row = result.fetchone()
                extension_exists = row[0] if row else False
                
                if not extension_exists:
                    return False
                
                # Check if we can query the view
                test_query = text("SELECT COUNT(*) FROM pg_stat_statements LIMIT 1")
                await session.execute(test_query)
                
                return True
                
        except Exception as e:
            logger.debug(f"pg_stat_statements not available: {e}")
            return False
    
    async def _analyze_slow_queries(self) -> List[Dict[str, Any]]:
        """
        Identify and analyze slow queries using pg_stat_statements
        """
        async with get_session_context() as session:
            query = text("""
                SELECT 
                    queryid,
                    query,
                    calls,
                    total_exec_time as total_time,
                    mean_exec_time as mean_time,
                    stddev_exec_time as stddev_time,
                    min_exec_time as min_time,
                    max_exec_time as max_time,
                    rows,
                    shared_blks_hit,
                    shared_blks_read,
                    shared_blks_dirtied,
                    shared_blks_written,
                    local_blks_hit,
                    local_blks_read,
                    local_blks_dirtied,
                    local_blks_written,
                    temp_blks_read,
                    temp_blks_written,
                    blk_read_time,
                    blk_write_time
                FROM pg_stat_statements 
                WHERE mean_exec_time > :threshold
                ORDER BY mean_exec_time DESC
                LIMIT 20
            """)
            
            result = await session.execute(query, {"threshold": self.slow_query_threshold_ms})
            slow_queries = []
            
            for row in result:
                query_stats = {
                    "query_id": str(row[0]) if row[0] else None,
                    "query_text": row[1][:500] + "..." if row[1] and len(row[1]) > 500 else row[1],
                    "calls": int(row[2]),
                    "total_time_ms": float(row[3]),
                    "mean_time_ms": float(row[4]),
                    "stddev_time_ms": float(row[5]) if row[5] else 0.0,
                    "min_time_ms": float(row[6]),
                    "max_time_ms": float(row[7]),
                    "rows_returned": int(row[8]),
                    "cache_hit_ratio": self._calculate_cache_hit_ratio(row[9], row[10]),
                    "io_time_ratio": self._calculate_io_time_ratio(row[19], row[20], row[3]),
                    "blocks_read": int(row[10]),
                    "blocks_hit": int(row[9]),
                    "temp_blocks_used": int(row[17]) + int(row[18]),
                    "optimization_priority": "high" if row[4] > 5000 else "medium" if row[4] > 2000 else "low"
                }
                
                # Get execution plan for this query if possible
                if row[1]:  # If we have query text
                    try:
                        plan_info = await self._get_query_execution_plan(row[1])
                        query_stats["execution_plan"] = plan_info
                    except Exception as e:
                        logger.debug(f"Could not get execution plan: {e}")
                
                slow_queries.append(query_stats)
            
            return slow_queries
    
    async def _analyze_frequent_queries(self) -> List[Dict[str, Any]]:
        """
        Identify frequently executed queries
        """
        async with get_session_context() as session:
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
                WHERE calls > :threshold
                ORDER BY calls DESC
                LIMIT 15
            """)
            
            result = await session.execute(query, {"threshold": self.frequent_query_threshold})
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
                    "optimization_potential": "high" if row[4] > 100 and row[2] > 1000 else "medium" if row[4] > 50 else "low"
                }
                frequent_queries.append(query_info)
            
            return frequent_queries
    
    async def _analyze_io_intensive_queries(self) -> List[Dict[str, Any]]:
        """
        Identify I/O intensive queries
        """
        async with get_session_context() as session:
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
                    AND ((blk_read_time + blk_write_time) / total_exec_time * 100) > :threshold
                ORDER BY (blk_read_time + blk_write_time) DESC
                LIMIT 15
            """)
            
            result = await session.execute(query, {"threshold": self.high_io_threshold_percent})
            io_intensive = []
            
            for row in result:
                total_io_time = float(row[13])
                total_time = float(row[3])
                io_ratio = (total_io_time / total_time * 100) if total_time > 0 else 0
                
                query_info = {
                    "query_id": str(row[0]) if row[0] else None,
                    "query_text": row[1][:400] + "..." if row[1] and len(row[1]) > 400 else row[1],
                    "calls": int(row[2]),
                    "total_time_ms": total_time,
                    "mean_time_ms": float(row[4]),
                    "total_io_time_ms": total_io_time,
                    "io_time_ratio_percent": io_ratio,
                    "blocks_read": int(row[5]) + int(row[7]) + int(row[9]),
                    "blocks_written": int(row[6]) + int(row[8]) + int(row[10]),
                    "read_time_ms": float(row[11]),
                    "write_time_ms": float(row[12]),
                    "optimization_priority": "critical" if io_ratio > 80 else "high" if io_ratio > 60 else "medium"
                }
                io_intensive.append(query_info)
            
            return io_intensive
    
    async def _analyze_cache_performance(self) -> Dict[str, Any]:
        """
        Analyze overall cache performance from query statistics
        """
        async with get_session_context() as session:
            # Overall cache statistics
            cache_query = text("""
                SELECT 
                    sum(shared_blks_hit) as total_hit,
                    sum(shared_blks_read) as total_read,
                    sum(shared_blks_hit + shared_blks_read) as total_blocks,
                    CASE 
                        WHEN sum(shared_blks_hit + shared_blks_read) = 0 THEN 100
                        ELSE (sum(shared_blks_hit)::float / sum(shared_blks_hit + shared_blks_read) * 100)
                    END as overall_hit_ratio,
                    count(*) as total_queries
                FROM pg_stat_statements
            """)
            
            result = await session.execute(cache_query)
            row = result.fetchone()

            if row:
                cache_stats = {
                    "total_blocks_hit": int(row[0]) if row[0] else 0,
                    "total_blocks_read": int(row[1]) if row[1] else 0,
                    "total_blocks_accessed": int(row[2]) if row[2] else 0,
                    "overall_hit_ratio_percent": float(row[3]) if row[3] else 100.0,
                    "total_queries_analyzed": int(row[4]) if row[4] else 0
                }
            else:
                cache_stats = {
                    "total_blocks_hit": 0,
                    "total_blocks_read": 0,
                    "total_blocks_accessed": 0,
                    "overall_hit_ratio_percent": 100.0,
                    "total_queries_analyzed": 0
                }
            
            # Queries with poor cache performance
            poor_cache_query = text("""
                SELECT 
                    query,
                    calls,
                    shared_blks_hit,
                    shared_blks_read,
                    CASE 
                        WHEN (shared_blks_hit + shared_blks_read) = 0 THEN 100
                        ELSE (shared_blks_hit::float / (shared_blks_hit + shared_blks_read) * 100)
                    END as hit_ratio
                FROM pg_stat_statements 
                WHERE (shared_blks_hit + shared_blks_read) > 100  -- Only queries with significant block access
                    AND (
                        CASE 
                            WHEN (shared_blks_hit + shared_blks_read) = 0 THEN 100
                            ELSE (shared_blks_hit::float / (shared_blks_hit + shared_blks_read) * 100)
                        END
                    ) < :threshold
                ORDER BY (shared_blks_hit + shared_blks_read) DESC
                LIMIT 10
            """)
            
            result = await session.execute(poor_cache_query, {"threshold": self.low_cache_hit_threshold})
            poor_cache_queries = []
            
            for row in result:
                poor_cache_queries.append({
                    "query_text": row[0][:200] + "..." if row[0] and len(row[0]) > 200 else row[0],
                    "calls": int(row[1]),
                    "blocks_hit": int(row[2]),
                    "blocks_read": int(row[3]),
                    "hit_ratio_percent": float(row[4])
                })
            
            cache_stats["poor_cache_queries"] = poor_cache_queries
            cache_stats["cache_health"] = (
                "excellent" if cache_stats["overall_hit_ratio_percent"] >= 95
                else "good" if cache_stats["overall_hit_ratio_percent"] >= 90
                else "needs_attention"
            )
            
            return cache_stats
    
    async def _get_current_activity(self) -> List[Dict[str, Any]]:
        """
        Get current query activity from pg_stat_activity
        """
        async with get_session_context() as session:
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
                    backend_xid,
                    backend_xmin,
                    query,
                    backend_type,
                    wait_event_type,
                    wait_event,
                    EXTRACT(EPOCH FROM (now() - query_start)) as duration_seconds,
                    EXTRACT(EPOCH FROM (now() - xact_start)) as txn_duration_seconds
                FROM pg_stat_activity 
                WHERE state != 'idle'
                    AND pid != pg_backend_pid()  -- Exclude current session
                    AND backend_type = 'client backend'
                ORDER BY query_start NULLS LAST
                LIMIT 20
            """)
            
            result = await session.execute(query)
            current_queries = []
            
            for row in result:
                query_info = {
                    "pid": row[0],
                    "username": row[1],
                    "application_name": row[2],
                    "client_addr": str(row[3]) if row[3] else None,
                    "query_start": row[6].isoformat() if row[6] else None,
                    "state": row[8],
                    "query_text": row[11][:300] + "..." if row[11] and len(row[11]) > 300 else row[11],
                    "wait_event_type": row[13],
                    "wait_event": row[14],
                    "duration_seconds": float(row[15]) if row[15] else 0,
                    "transaction_duration_seconds": float(row[16]) if row[16] else 0,
                    "is_long_running": (row[15] and float(row[15]) > 60),
                    "is_blocked": (row[13] and "Lock" in row[13])
                }
                current_queries.append(query_info)
            
            return current_queries
    
    async def _get_query_execution_plan(self, query_text: str) -> Dict[str, Any]:
        """
        Get execution plan for a query
        """
        try:
            async with get_session_context() as session:
                # Use EXPLAIN (without ANALYZE for safety)
                explain_query = f"EXPLAIN (FORMAT JSON, BUFFERS, VERBOSE) {query_text}"
                
                result = await session.execute(text(explain_query))
                row = result.fetchone()
                
                if row and row[0]:
                    plan_json = row[0][0] if isinstance(row[0], list) else row[0]
                    
                    # Extract key plan information
                    plan_info = {
                        "plan_json": plan_json,
                        "total_cost": plan_json.get("Total Cost", 0),
                        "rows_estimate": plan_json.get("Plan Rows", 0),
                        "width_estimate": plan_json.get("Plan Width", 0),
                        "node_type": plan_json.get("Node Type", "Unknown"),
                        "expensive_operations": self._identify_expensive_operations(plan_json),
                        "has_sequential_scan": self._plan_has_operation(plan_json, "Seq Scan"),
                        "has_index_scan": self._plan_has_operation(plan_json, "Index Scan"),
                        "optimization_suggestions": self._generate_plan_suggestions(plan_json)
                    }
                    
                    return plan_info
                
        except Exception as e:
            logger.debug(f"Could not get execution plan: {e}")
        
        return {"error": "Could not retrieve execution plan"}
    
    def _calculate_cache_hit_ratio(self, blocks_hit: int, blocks_read: int) -> float:
        """Calculate cache hit ratio"""
        total_blocks = blocks_hit + blocks_read
        return (blocks_hit / total_blocks * 100) if total_blocks > 0 else 100.0
    
    def _calculate_io_time_ratio(self, read_time: float, write_time: float, total_time: float) -> float:
        """Calculate I/O time as percentage of total time"""
        io_time = (read_time or 0) + (write_time or 0)
        return (io_time / total_time * 100) if total_time > 0 else 0.0
    
    def _identify_expensive_operations(self, plan_node: Dict[str, Any]) -> List[str]:
        """Identify expensive operations in execution plan"""
        expensive_ops = []
        
        node_type = plan_node.get("Node Type", "")
        
        # Check for expensive operations
        if "Seq Scan" in node_type:
            expensive_ops.append("Sequential Scan (consider adding index)")
        
        if "Sort" in node_type and plan_node.get("Sort Method") == "external sort":
            expensive_ops.append("External Sort (increase work_mem)")
        
        if "Nested Loop" in node_type and plan_node.get("Total Cost", 0) > 10000:
            expensive_ops.append("Expensive Nested Loop (consider different join method)")
        
        if "Hash" in node_type and plan_node.get("Hash Buckets Original", 0) != plan_node.get("Hash Buckets", 0):
            expensive_ops.append("Hash table resize (increase work_mem)")
        
        # Recursively check child plans
        for child_plan in plan_node.get("Plans", []):
            expensive_ops.extend(self._identify_expensive_operations(child_plan))
        
        return expensive_ops
    
    def _plan_has_operation(self, plan_node: Dict[str, Any], operation: str) -> bool:
        """Check if plan contains specific operation"""
        if operation in plan_node.get("Node Type", ""):
            return True
        
        # Check child plans
        for child_plan in plan_node.get("Plans", []):
            if self._plan_has_operation(child_plan, operation):
                return True
        
        return False
    
    def _generate_plan_suggestions(self, plan_node: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on execution plan"""
        suggestions = []
        
        # Sequential scan suggestions
        if self._plan_has_operation(plan_node, "Seq Scan"):
            suggestions.append("Consider adding indexes for sequential scans")
        
        # Sort suggestions
        if self._plan_has_operation(plan_node, "Sort"):
            suggestions.append("Consider adding ORDER BY index or increasing work_mem")
        
        # Hash join suggestions
        if self._plan_has_operation(plan_node, "Hash Join"):
            if plan_node.get("Total Cost", 0) > 5000:
                suggestions.append("High-cost hash join - consider index optimization")
        
        return suggestions
    
    def _generate_performance_summary(self, slow_queries: List, frequent_queries: List, 
                                    io_intensive: List, cache_analysis: Dict) -> Dict[str, Any]:
        """Generate performance summary"""
        return {
            "total_slow_queries": len(slow_queries),
            "total_frequent_queries": len(frequent_queries),
            "total_io_intensive_queries": len(io_intensive),
            "overall_cache_hit_ratio": cache_analysis.get("overall_hit_ratio_percent", 0),
            "queries_needing_attention": len(slow_queries) + len([q for q in frequent_queries if q["mean_time_ms"] > 100]),
            "performance_status": self._assess_overall_performance(slow_queries, cache_analysis)
        }
    
    def _assess_overall_performance(self, slow_queries: List, cache_analysis: Dict) -> str:
        """Assess overall query performance status"""
        slow_count = len(slow_queries)
        cache_hit_ratio = cache_analysis.get("overall_hit_ratio_percent", 100)
        
        if slow_count > 10 or cache_hit_ratio < 85:
            return "needs_immediate_attention"
        elif slow_count > 5 or cache_hit_ratio < 90:
            return "needs_optimization"
        elif slow_count > 0 or cache_hit_ratio < 95:
            return "good_with_room_for_improvement"
        else:
            return "excellent"
    
    def _generate_optimization_recommendations(self, slow_queries: List, frequent_queries: List,
                                             io_intensive: List, cache_analysis: Dict) -> List[str]:
        """Generate actionable optimization recommendations"""
        recommendations = []
        
        # Slow query recommendations
        if len(slow_queries) > 0:
            recommendations.append(f"Optimize {len(slow_queries)} slow queries (>{self.slow_query_threshold_ms}ms)")
            
            # Check for common patterns
            seq_scan_queries = sum(1 for q in slow_queries if q.get("execution_plan", {}).get("has_sequential_scan", False))
            if seq_scan_queries > 0:
                recommendations.append(f"Add indexes for {seq_scan_queries} queries using sequential scans")
        
        # Cache recommendations
        cache_hit_ratio = cache_analysis.get("overall_hit_ratio_percent", 100)
        if cache_hit_ratio < 90:
            recommendations.append(f"Improve cache hit ratio from {cache_hit_ratio:.1f}% (increase shared_buffers)")
        
        # I/O intensive query recommendations
        if len(io_intensive) > 0:
            recommendations.append(f"Optimize {len(io_intensive)} I/O intensive queries")
        
        # Frequent query recommendations
        high_frequency_slow = [q for q in frequent_queries if q["mean_time_ms"] > 100]
        if high_frequency_slow:
            recommendations.append(f"Prioritize optimization of {len(high_frequency_slow)} frequently-called slow queries")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Query performance appears healthy - continue monitoring")
        else:
            recommendations.append("Consider enabling pg_stat_statements if not already enabled for detailed analysis")
        
        return recommendations
    
    async def get_query_performance_summary(self) -> Dict[str, Any]:
        """
        Get a concise summary of query performance health
        """
        analysis = await self.analyze_query_performance()
        
        if "error" in analysis:
            return {
                "status": "error",
                "message": "Failed to analyze query performance",
                "error": analysis["error"]
            }
        
        summary = analysis.get("performance_summary", {})
        slow_count = summary.get("total_slow_queries", 0)
        cache_hit_ratio = summary.get("overall_cache_hit_ratio", 100)
        performance_status = summary.get("performance_status", "unknown")
        
        # Determine overall status
        if performance_status in ["needs_immediate_attention"]:
            status = "critical"
        elif performance_status in ["needs_optimization"]:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "performance_status": performance_status,
            "slow_queries_count": slow_count,
            "cache_hit_ratio_percent": cache_hit_ratio,
            "pg_stat_statements_available": analysis.get("pg_stat_statements_available", False),
            "key_recommendations": analysis.get("recommendations", [])[:3],
            "timestamp": analysis.get("timestamp")
        }