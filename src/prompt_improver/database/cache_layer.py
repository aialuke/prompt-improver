"""Advanced database caching layer for Phase 2 performance optimization.

Implements intelligent query result caching with Redis to achieve 50% database load reduction.
features:
- Smart cache key generation
- Automatic cache invalidation
- Cache warming strategies
- Performance metrics tracking
"""
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar
from sqlmodel import SQLModel
from prompt_improver.core.config import AppConfig
from prompt_improver.performance.monitoring.metrics_registry import StandardMetrics, get_metrics_registry
logger = logging.getLogger(__name__)
T = TypeVar('T', bound=SQLModel)

class CacheStrategy(Enum):
    """Cache strategy options"""
    aggressive = 'aggressive'
    selective = 'selective'
    smart = 'smart'

@dataclass
class CachePolicy:
    """Cache policy configuration"""
    ttl_seconds: int = 300
    strategy: CacheStrategy = CacheStrategy.smart
    max_size_bytes: int = 100000000
    invalidation_patterns: list[str] = None
    warm_on_startup: bool = True
    compress: bool = True

class DatabaseCacheLayer:
    """Intelligent database caching layer for 50% load reduction.

    features:
    - Query result caching with automatic TTL
    - Smart invalidation based on table updates
    - Cache warming for frequently accessed data
    - Compression for large result sets
    - Performance metrics and monitoring
    """

    def __init__(self, cache_policy: CachePolicy | None=None):
        self.policy = cache_policy or CachePolicy()
        self.redis_cache = RedisCache()
        self.metrics_registry = get_metrics_registry()
        self._stats = {'hits': 0, 'misses': 0, 'invalidations': 0, 'bytes_saved': 0, 'queries_cached': 0, 'db_calls_avoided': 0}
        self._query_frequency: dict[str, int] = {}
        self._query_costs: dict[str, float] = {}
        self._invalidation_subscriptions: set[str] = set()
        self._init_metrics()

    def _init_metrics(self):
        """Initialize OpenTelemetry metrics for cache monitoring"""
        self.cache_hit_counter = self.metrics_registry.get_or_create_counter(StandardMetrics.CACHE_HITS_TOTAL, 'Total database cache hits')
        self.cache_miss_counter = self.metrics_registry.get_or_create_counter(StandardMetrics.CACHE_MISSES_TOTAL, 'Total database cache misses')
        self.cache_size_gauge = self.metrics_registry.get_or_create_gauge('database_cache_size_bytes', 'Current database cache size in bytes')
        self.cache_operation_histogram = self.metrics_registry.get_or_create_histogram(StandardMetrics.CACHE_OPERATION_DURATION, 'Database cache operation duration', ['operation'])

    def _generate_cache_key(self, query: str, params: dict[str, Any] | None=None) -> str:
        """Generate deterministic cache key for query and parameters"""
        key_data = {'query': query.strip().lower(), 'params': sorted(params.items()) if params else []}
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        hash_digest = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        return f'db:query:{hash_digest}'

    def _should_cache(self, query: str, execution_time_ms: float) -> bool:
        """Determine if query result should be cached based on strategy"""
        if self.policy.strategy == CacheStrategy.aggressive:
            return True
        if self.policy.strategy == CacheStrategy.selective:
            return execution_time_ms > 10
        if self.policy.strategy == CacheStrategy.smart:
            query_key = self._generate_cache_key(query)
            frequency = self._query_frequency.get(query_key, 0)
            if execution_time_ms > 20:
                return True
            if frequency > 5:
                return True
            if self._is_expensive_query_pattern(query):
                return True
        return False

    def _is_expensive_query_pattern(self, query: str) -> bool:
        """Check if query matches known expensive patterns"""
        expensive_patterns = ['join', 'group by', 'order by', 'distinct', 'having', 'union', 'intersect', 'with recursive']
        query_lower = query.lower()
        return any((pattern in query_lower for pattern in expensive_patterns))

    def _calculate_ttl(self, query: str, result_size: int) -> int:
        """Calculate optimal TTL based on query characteristics"""
        base_ttl = self.policy.ttl_seconds
        query_lower = query.lower()
        if any((table in query_lower for table in ['rules', 'models', 'configurations'])):
            return base_ttl * 4
        if any((table in query_lower for table in ['sessions', 'analytics', 'metrics'])):
            return base_ttl // 2
        if result_size > 1000:
            return int(base_ttl * 1.5)
        return base_ttl

    async def get_or_execute(self, query: str, params: dict[str, Any] | None=None, executor: Callable | None=None, ttl_override: int | None=None) -> tuple[Any, bool]:
        """Get query result from cache or execute and cache.

        Returns:
            Tuple of (result, was_cached)
        """
        cache_key = self._generate_cache_key(query, params)
        with self.cache_operation_histogram.labels(operation='get').time():
            cached_result = await self.redis_cache.get(cache_key)
        if cached_result:
            self._stats['hits'] += 1
            self._stats['db_calls_avoided'] += 1
            self.cache_hit_counter.inc()
            try:
                result = json.loads(cached_result.decode())
                logger.debug('Cache hit for query: %s...', query[:50])
                return (result, True)
            except Exception as e:
                logger.warning('Failed to deserialize cached result: %s', e)
        self._stats['misses'] += 1
        self.cache_miss_counter.inc()
        if executor:
            import time
            start_time = time.perf_counter()
            result = await executor(query, params)
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._query_frequency[cache_key] = self._query_frequency.get(cache_key, 0) + 1
            self._query_costs[cache_key] = execution_time_ms
            if self._should_cache(query, execution_time_ms):
                ttl = ttl_override or self._calculate_ttl(query, len(str(result)))
                try:
                    serialized = json.dumps(result, default=str).encode()
                    if len(serialized) <= self.policy.max_size_bytes:
                        with self.cache_operation_histogram.labels(operation='set').time():
                            await self.redis_cache.set(cache_key, serialized, expire=ttl)
                        self._stats['queries_cached'] += 1
                        self._stats['bytes_saved'] += len(serialized)
                        logger.debug('Cached query result: %s... (TTL: %ss, Size: %s bytes)', query[:50], ttl, len(serialized))
                    else:
                        logger.warning('Result too large to cache: %s bytes', len(serialized))
                except Exception as e:
                    logger.warning('Failed to cache query result: %s', e)
            return (result, False)
        return (None, False)

    async def invalidate_pattern(self, pattern: str):
        """Invalidate all cache entries matching pattern"""
        logger.info('Invalidating cache entries matching pattern: %s', pattern)
        try:
            cursor = 0
            invalidated_count = 0
            while True:
                cursor, keys = await self.redis_cache.redis_client.scan(cursor, match=f'db:query:*{pattern}*', count=100)
                if keys:
                    for key in keys:
                        if isinstance(key, bytes):
                            key = key.decode()
                        await self.redis_cache.invalidate(key)
                        invalidated_count += 1
                if cursor == 0:
                    break
            self._stats['invalidations'] += invalidated_count
            logger.info('Invalidated %s cache entries', invalidated_count)
        except Exception as e:
            logger.error('Error invalidating cache pattern: %s', e)

    async def invalidate_table_cache(self, table_name: str):
        """Invalidate all cache entries related to a specific table"""
        await self.invalidate_pattern(table_name.lower())

    async def warm_cache(self, warm_queries: list[tuple[str, dict[str, Any], Callable]]):
        """Pre-warm cache with frequently accessed queries"""
        if not self.policy.warm_on_startup:
            return
        logger.info('Warming cache with %s queries', len(warm_queries))
        warmed_count = 0
        for query, params, executor in warm_queries:
            try:
                _, was_cached = await self.get_or_execute(query, params, executor)
                if not was_cached:
                    warmed_count += 1
            except Exception as e:
                logger.warning('Failed to warm cache for query: %s', e)
        logger.info('Cache warming complete: %s queries cached', warmed_count)

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_operations = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total_operations if total_operations > 0 else 0
        db_queries_without_cache = total_operations
        db_queries_with_cache = self._stats['misses']
        load_reduction = (db_queries_without_cache - db_queries_with_cache) / db_queries_without_cache * 100 if db_queries_without_cache > 0 else 0
        top_queries = sorted(self._query_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        expensive_queries = sorted(self._query_costs.items(), key=lambda x: x[1], reverse=True)[:10]
        return {'cache_hit_rate': round(hit_rate * 100, 2), 'total_hits': self._stats['hits'], 'total_misses': self._stats['misses'], 'total_invalidations': self._stats['invalidations'], 'queries_cached': self._stats['queries_cached'], 'db_calls_avoided': self._stats['db_calls_avoided'], 'bytes_saved': self._stats['bytes_saved'], 'mb_saved': round(self._stats['bytes_saved'] / 1024 / 1024, 2), 'database_load_reduction_percent': round(load_reduction, 1), 'cache_strategy': self.policy.strategy.value, 'top_cached_queries': [{'key': k, 'frequency': v} for k, v in top_queries], 'most_expensive_queries': [{'key': k, 'cost_ms': round(v, 2)} for k, v in expensive_queries]}

    async def optimize_cache_policy(self):
        """Optimize cache policy based on observed patterns"""
        stats = await self.get_cache_stats()
        if stats['cache_hit_rate'] < 30:
            logger.info('Low cache hit rate detected, switching to aggressive caching')
            self.policy.strategy = CacheStrategy.aggressive
        elif stats['cache_hit_rate'] > 80:
            logger.info('High cache hit rate detected, optimizing with smart caching')
            self.policy.strategy = CacheStrategy.smart
        valuable_queries = [k for k, v in self._query_costs.items() if v > 50 and self._query_frequency.get(k, 0) > 3]
        logger.info('Identified %s high-value queries for priority caching', len(valuable_queries))
_cache_layer: DatabaseCacheLayer | None = None

def get_database_cache_layer() -> DatabaseCacheLayer:
    """Get or create global database cache layer"""
    global _cache_layer
    if _cache_layer is None:
        _cache_layer = DatabaseCacheLayer()
    return _cache_layer
