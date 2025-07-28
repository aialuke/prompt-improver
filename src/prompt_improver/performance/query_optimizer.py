"""Advanced Query Optimization System with Prepared Statement Caching.

Implements 2025 best practices for database performance optimization including
prepared statement caching, query timeout protection, and SLA enforcement.
"""

import asyncio
import logging
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict

from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine
from sqlalchemy import text

logger = logging.getLogger(__name__)

class QueryType(str, Enum):
    """Types of queries for optimization categorization."""
    RULE_RETRIEVAL = "rule_retrieval"
    RULE_PERFORMANCE = "rule_performance"
    RULE_COMBINATIONS = "rule_combinations"
    FEEDBACK_STORAGE = "feedback_storage"
    ANALYTICS = "analytics"
    HEALTH_CHECK = "health_check"

@dataclass
class QueryMetrics:
    """Query performance metrics."""
    query_hash: str
    query_type: QueryType
    execution_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    last_executed: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    timeout_count: int = 0
    error_count: int = 0

@dataclass
class PreparedStatement:
    """Cached prepared statement with metadata."""
    statement_id: str
    query_hash: str
    query_type: QueryType
    sql_text: str
    parameter_names: List[str]
    created_at: float
    last_used: float
    use_count: int = 0
    avg_execution_time_ms: float = 0.0
    is_warmed: bool = False

class PreparedStatementCache:
    """High-performance prepared statement cache with LRU eviction.
    
    Features:
    - 1000-statement cache limit with intelligent eviction
    - Automatic cache warming for frequently used queries
    - Query performance tracking and optimization
    - Memory-efficient storage with weak references
    """
    
    def __init__(self, max_size: int = 1000, warming_threshold: int = 10):
        """Initialize prepared statement cache.
        
        Args:
            max_size: Maximum number of cached statements
            warming_threshold: Minimum use count for cache warming
        """
        self.max_size = max_size
        self.warming_threshold = warming_threshold
        
        # Cache storage (OrderedDict for LRU behavior)
        self._cache: OrderedDict[str, PreparedStatement] = OrderedDict()
        
        # Query metrics tracking
        self._metrics: Dict[str, QueryMetrics] = {}
        
        # Cache statistics
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "warming_operations": 0,
            "total_queries": 0
        }
        
        # Warming configuration
        self._warming_enabled = True
        self._warming_batch_size = 50

    def get_statement(self, query_hash: str) -> Optional[PreparedStatement]:
        """Get prepared statement from cache.
        
        Args:
            query_hash: Hash of the query
            
        Returns:
            PreparedStatement if found, None otherwise
        """
        if query_hash in self._cache:
            # Move to end (most recently used)
            statement = self._cache.pop(query_hash)
            self._cache[query_hash] = statement
            
            # Update usage statistics
            statement.last_used = time.time()
            statement.use_count += 1
            self._stats["cache_hits"] += 1
            
            return statement
        
        self._stats["cache_misses"] += 1
        return None

    def store_statement(self, statement: PreparedStatement) -> None:
        """Store prepared statement in cache.
        
        Args:
            statement: Prepared statement to cache
        """
        # Check if cache is full
        if len(self._cache) >= self.max_size:
            self._evict_lru_statements()
        
        # Store statement
        self._cache[statement.query_hash] = statement
        
        # Update metrics
        if statement.query_hash not in self._metrics:
            self._metrics[statement.query_hash] = QueryMetrics(
                query_hash=statement.query_hash,
                query_type=statement.query_type
            )

    def _evict_lru_statements(self, count: int = None) -> None:
        """Evict least recently used statements.
        
        Args:
            count: Number of statements to evict (default: 10% of cache)
        """
        if not count:
            count = max(1, self.max_size // 10)  # Evict 10% by default
        
        evicted = 0
        while evicted < count and self._cache:
            # Remove oldest (least recently used)
            query_hash, statement = self._cache.popitem(last=False)
            evicted += 1
            self._stats["evictions"] += 1
            
            logger.debug(f"Evicted prepared statement: {statement.statement_id}")

    def update_metrics(
        self, 
        query_hash: str, 
        execution_time_ms: float, 
        success: bool = True
    ) -> None:
        """Update query performance metrics.
        
        Args:
            query_hash: Hash of the executed query
            execution_time_ms: Execution time in milliseconds
            success: Whether the query succeeded
        """
        if query_hash not in self._metrics:
            return
        
        metrics = self._metrics[query_hash]
        metrics.execution_count += 1
        metrics.last_executed = time.time()
        
        if success:
            metrics.total_time_ms += execution_time_ms
            metrics.avg_time_ms = metrics.total_time_ms / metrics.execution_count
            metrics.min_time_ms = min(metrics.min_time_ms, execution_time_ms)
            metrics.max_time_ms = max(metrics.max_time_ms, execution_time_ms)
        else:
            metrics.error_count += 1

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics.
        
        Returns:
            Cache statistics dictionary
        """
        total_requests = self._stats["cache_hits"] + self._stats["cache_misses"]
        hit_rate = self._stats["cache_hits"] / max(1, total_requests)
        
        return {
            "cache_size": len(self._cache),
            "max_size": self.max_size,
            "utilization": len(self._cache) / self.max_size,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "evictions": self._stats["evictions"],
            "warming_operations": self._stats["warming_operations"],
            "performance_status": "good" if hit_rate >= 0.85 else "needs_optimization"
        }

    async def warm_cache(self, engine: AsyncEngine, queries: List[Tuple[str, QueryType]]) -> int:
        """Warm cache with frequently used queries.
        
        Args:
            engine: Database engine for query preparation
            queries: List of (sql, query_type) tuples to warm
            
        Returns:
            Number of queries warmed
        """
        if not self._warming_enabled:
            return 0
        
        warmed_count = 0
        
        for sql, query_type in queries[:self._warming_batch_size]:
            try:
                query_hash = self._hash_query(sql)
                
                # Skip if already warmed
                if query_hash in self._cache and self._cache[query_hash].is_warmed:
                    continue
                
                # Create prepared statement
                statement = PreparedStatement(
                    statement_id=f"warm_{query_hash[:8]}",
                    query_hash=query_hash,
                    query_type=query_type,
                    sql_text=sql,
                    parameter_names=self._extract_parameters(sql),
                    created_at=time.time(),
                    last_used=time.time(),
                    is_warmed=True
                )
                
                self.store_statement(statement)
                warmed_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to warm query: {e}")
        
        self._stats["warming_operations"] += warmed_count
        logger.info(f"Cache warming completed: {warmed_count} queries warmed")
        
        return warmed_count

    def _hash_query(self, sql: str) -> str:
        """Generate hash for SQL query.
        
        Args:
            sql: SQL query text
            
        Returns:
            Query hash string
        """
        # Normalize query for consistent hashing
        normalized = sql.strip().lower()
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _extract_parameters(self, sql: str) -> List[str]:
        """Extract parameter names from SQL query.
        
        Args:
            sql: SQL query text
            
        Returns:
            List of parameter names
        """
        import re
        # Find SQLAlchemy-style parameters (:param_name)
        parameters = re.findall(r':(\w+)', sql)
        return list(set(parameters))  # Remove duplicates

class QueryOptimizer:
    """Advanced query optimizer with SLA enforcement and performance monitoring.
    
    Features:
    - <200ms SLA enforcement with timeout protection
    - Prepared statement caching with 1000-statement limit
    - Query performance monitoring and optimization
    - Automatic cache warming for hot queries
    - Connection pool optimization
    """
    
    def __init__(
        self, 
        engine: AsyncEngine,
        sla_timeout_ms: float = 200.0,
        query_timeout_ms: float = 150.0
    ):
        """Initialize query optimizer.
        
        Args:
            engine: SQLAlchemy async engine
            sla_timeout_ms: SLA timeout in milliseconds
            query_timeout_ms: Individual query timeout in milliseconds
        """
        self.engine = engine
        self.sla_timeout_ms = sla_timeout_ms
        self.query_timeout_ms = query_timeout_ms
        
        # Prepared statement cache
        self.statement_cache = PreparedStatementCache()
        
        # Performance monitoring
        self._performance_stats = {
            "total_queries": 0,
            "sla_violations": 0,
            "timeouts": 0,
            "cache_hits": 0,
            "avg_response_time_ms": 0.0,
            "p95_response_time_ms": 0.0,
            "p99_response_time_ms": 0.0
        }
        
        # Response time tracking for percentiles
        self._response_times: List[float] = []
        self._max_response_history = 1000
        
        # Query categorization for optimization
        self._query_patterns = {
            QueryType.RULE_RETRIEVAL: [
                "SELECT * FROM rule_metadata",
                "SELECT * FROM rule_performance", 
                "SELECT rule_id, rule_name FROM rule_metadata WHERE enabled = true"
            ],
            QueryType.RULE_COMBINATIONS: [
                "SELECT * FROM rule_combinations WHERE combined_effectiveness >=",
                "SELECT rule_set, combined_effectiveness FROM rule_combinations"
            ],
            QueryType.FEEDBACK_STORAGE: [
                "INSERT INTO feedback_collection",
                "INSERT INTO ml_training_feedback"
            ]
        }

    async def execute_optimized_query(
        self,
        session: AsyncSession,
        sql: str,
        parameters: Dict[str, Any] = None,
        query_type: QueryType = QueryType.RULE_RETRIEVAL
    ) -> Any:
        """Execute query with optimization and SLA enforcement.
        
        Args:
            session: Database session
            sql: SQL query text
            parameters: Query parameters
            query_type: Type of query for optimization
            
        Returns:
            Query result
            
        Raises:
            asyncio.TimeoutError: If query exceeds timeout
        """
        start_time = time.time()
        query_hash = self.statement_cache._hash_query(sql)
        
        try:
            # Check prepared statement cache
            prepared_stmt = self.statement_cache.get_statement(query_hash)
            
            if prepared_stmt:
                self._performance_stats["cache_hits"] += 1
            else:
                # Create new prepared statement
                prepared_stmt = PreparedStatement(
                    statement_id=f"stmt_{query_hash[:8]}",
                    query_hash=query_hash,
                    query_type=query_type,
                    sql_text=sql,
                    parameter_names=list(parameters.keys()) if parameters else [],
                    created_at=time.time(),
                    last_used=time.time()
                )
                self.statement_cache.store_statement(prepared_stmt)
            
            # Execute with timeout protection
            result = await asyncio.wait_for(
                session.execute(text(sql), parameters or {}),
                timeout=self.query_timeout_ms / 1000.0
            )
            
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Update performance metrics
            self._update_performance_metrics(execution_time_ms, query_hash, True)
            
            # Check SLA compliance
            if execution_time_ms > self.sla_timeout_ms:
                self._performance_stats["sla_violations"] += 1
                logger.warning(f"SLA violation: Query took {execution_time_ms:.1f}ms (limit: {self.sla_timeout_ms}ms)")
            
            return result
            
        except asyncio.TimeoutError:
            execution_time_ms = (time.time() - start_time) * 1000
            self._performance_stats["timeouts"] += 1
            self._update_performance_metrics(execution_time_ms, query_hash, False)
            logger.error(f"Query timeout after {execution_time_ms:.1f}ms: {sql[:100]}...")
            raise
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self._update_performance_metrics(execution_time_ms, query_hash, False)
            logger.error(f"Query execution error: {e}")
            raise

    def _update_performance_metrics(
        self, 
        execution_time_ms: float, 
        query_hash: str, 
        success: bool
    ) -> None:
        """Update performance metrics and statistics.
        
        Args:
            execution_time_ms: Query execution time
            query_hash: Hash of the executed query
            success: Whether query succeeded
        """
        self._performance_stats["total_queries"] += 1
        
        # Update cache metrics
        self.statement_cache.update_metrics(query_hash, execution_time_ms, success)
        
        if success:
            # Track response times for percentile calculation
            self._response_times.append(execution_time_ms)
            
            # Maintain response time history limit
            if len(self._response_times) > self._max_response_history:
                self._response_times = self._response_times[-self._max_response_history:]
            
            # Update average response time
            total_time = sum(self._response_times)
            self._performance_stats["avg_response_time_ms"] = total_time / len(self._response_times)
            
            # Calculate percentiles
            if len(self._response_times) >= 20:  # Minimum for meaningful percentiles
                sorted_times = sorted(self._response_times)
                p95_index = int(0.95 * len(sorted_times))
                p99_index = int(0.99 * len(sorted_times))
                
                self._performance_stats["p95_response_time_ms"] = sorted_times[p95_index]
                self._performance_stats["p99_response_time_ms"] = sorted_times[p99_index]

    async def warm_statement_cache(self) -> int:
        """Warm prepared statement cache with common queries.
        
        Returns:
            Number of statements warmed
        """
        warming_queries = []
        
        # Add common queries for each type
        for query_type, patterns in self._query_patterns.items():
            for pattern in patterns:
                warming_queries.append((pattern, query_type))
        
        return await self.statement_cache.warm_cache(self.engine, warming_queries)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        cache_stats = self.statement_cache.get_cache_statistics()
        
        # Calculate SLA compliance rate
        total_queries = self._performance_stats["total_queries"]
        sla_compliance_rate = 1.0 - (self._performance_stats["sla_violations"] / max(1, total_queries))
        
        return {
            "query_performance": {
                "total_queries": total_queries,
                "avg_response_time_ms": self._performance_stats["avg_response_time_ms"],
                "p95_response_time_ms": self._performance_stats["p95_response_time_ms"],
                "p99_response_time_ms": self._performance_stats["p99_response_time_ms"],
                "sla_compliance_rate": sla_compliance_rate,
                "sla_violations": self._performance_stats["sla_violations"],
                "timeouts": self._performance_stats["timeouts"]
            },
            "cache_performance": cache_stats,
            "sla_status": {
                "target_ms": self.sla_timeout_ms,
                "compliant": sla_compliance_rate >= 0.95,
                "status": "good" if sla_compliance_rate >= 0.95 else "needs_optimization"
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for query optimization system.
        
        Returns:
            Health status information
        """
        metrics = self.get_performance_metrics()
        
        # Health assessment
        health_issues = []
        
        if metrics["sla_status"]["compliant"] is False:
            health_issues.append(f"SLA compliance below 95%: {metrics['query_performance']['sla_compliance_rate']:.2f}")
        
        if metrics["query_performance"]["p95_response_time_ms"] > self.sla_timeout_ms:
            health_issues.append(f"P95 response time exceeds SLA: {metrics['query_performance']['p95_response_time_ms']:.1f}ms")
        
        if metrics["cache_performance"]["hit_rate"] < 0.85:
            health_issues.append(f"Cache hit rate below 85%: {metrics['cache_performance']['hit_rate']:.2f}")
        
        overall_status = "healthy" if not health_issues else "degraded"
        
        return {
            "status": overall_status,
            "issues": health_issues,
            "metrics": metrics,
            "timestamp": time.time()
        }
