"""Multi-level cache management services.

This package provides a comprehensive caching solution extracted from the monolithic
unified_connection_manager.py with clear separation of concerns:

L1 Cache - In-Memory (memory_cache.py):
    - High-performance LRU cache for hot data
    - Sub-millisecond access times
    - Configurable eviction policies
    - Access pattern tracking

L2 Cache - Redis (redis_cache.py):
    - Distributed caching across instances
    - Persistence and durability
    - Cluster and Sentinel support
    - Security context validation

L3 Cache - Database Fallback (database_cache.py):
    - Ultimate fallback for cache misses
    - Query result caching
    - Long-term storage optimization
    - Transaction-safe operations

Cache Management (cache_manager.py):
    - Multi-level cache orchestration
    - Intelligent cache warming
    - Performance monitoring and metrics
    - Automatic failover between levels

Cache Warming (cache_warmer.py):
    - Predictive cache population
    - Access pattern analysis
    - Background warming processes
    - Priority-based warming algorithms

This architecture provides:
- High performance through intelligent multi-level caching
- Reliability through automatic fallback mechanisms
- Scalability through distributed Redis caching
- Observability through comprehensive metrics
- Security through context validation
"""

from .cache_manager import (
    CacheConsistencyMode,
    CacheFallbackStrategy,
    CacheLevel,
    CacheManager,
    CacheManagerConfig,
    CacheManagerMetrics,
    create_cache_manager,
)
from .cache_warmer import (
    AccessPattern,
    CacheWarmer,
    CacheWarmerConfig,
    PatternType,
    WarmingPriority,
    WarmingTask,
    create_cache_warmer,
)
from .database_cache import DatabaseCache, DatabaseCacheConfig

# Import all implemented cache components
from .memory_cache import CacheEntry, EvictionPolicy, MemoryCache
from .redis_cache import RedisCache, RedisCacheConfig

__all__ = [
    # Cache Warmer
    "CacheWarmer",
    "CacheWarmerConfig",
    "WarmingPriority",
    "PatternType",
    "AccessPattern",
    "WarmingTask",
    "create_cache_warmer",
    # Memory Cache (L1)
    "MemoryCache",
    "CacheEntry",
    "EvictionPolicy",
    # Redis Cache (L2)
    "RedisCache",
    "RedisCacheConfig",
    # Database Cache (L3)
    "DatabaseCache",
    "DatabaseCacheConfig",
    # Cache Manager
    "CacheManager",
    "CacheManagerConfig",
    "CacheLevel",
    "CacheFallbackStrategy",
    "CacheConsistencyMode",
    "CacheManagerMetrics",
    "create_cache_manager",
]
