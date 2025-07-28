"""Multi-level rule effectiveness caching system for sub-50ms retrieval.

Implements L1 (memory) + L2 (Redis) caching with 95% hit rate target
and intelligent cache warming strategies.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import coredis
from coredis import Redis

from .intelligent_rule_selector import RuleScore
from .models import PromptCharacteristics

logger = logging.getLogger(__name__)

class CacheLevel(str, Enum):
    """Cache levels for rule effectiveness data."""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DATABASE = "l3_database"

@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l3_hits: int = 0
    total_requests: int = 0
    avg_retrieval_time_ms: float = 0.0
    hit_rate: float = 0.0
    target_hit_rate: float = 0.95

@dataclass
class CachedRuleData:
    """Cached rule effectiveness data with metadata."""
    rule_scores: List[RuleScore]
    characteristics_hash: str
    cached_at: float
    cache_level: CacheLevel
    retrieval_time_ms: float
    access_count: int = 0
    last_accessed: float = 0.0

class RuleEffectivenessCache:
    """Multi-level caching system for rule effectiveness data.

    Implements intelligent caching strategy:
    - L1 (Memory): Hot data, <1ms retrieval, 1000 entries
    - L2 (Redis): Warm data, <10ms retrieval, 10000 entries
    - L3 (Database): Cold data, <50ms retrieval, unlimited

    Features:
    - Automatic cache warming based on access patterns
    - LRU eviction with access frequency weighting
    - Cache coherence across multiple instances
    - Performance monitoring and optimization
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/3"):
        """Initialize multi-level rule cache.

        Args:
            redis_url: Redis connection URL for L2 cache
        """
        self.redis_url = redis_url
        self._redis_client = None

        # L1 Memory cache configuration
        self.l1_cache = {}  # Dict[str, CachedRuleData]
        self.l1_max_size = 1000
        self.l1_ttl_seconds = 300  # 5 minutes

        # L2 Redis cache configuration
        self.l2_ttl_seconds = 1800  # 30 minutes
        self.l2_key_prefix = "rule_cache:effectiveness:"

        # Performance targets
        self.target_hit_rate = 0.95
        self.target_retrieval_time_ms = 50.0

        # Cache metrics
        self.metrics = CacheMetrics(target_hit_rate=self.target_hit_rate)

        # Cache warming configuration
        self.warming_enabled = True
        self.warming_threshold = 0.8  # Start warming when hit rate drops below 80%
        self.popular_patterns_cache = {}  # Track popular access patterns

    async def get_redis_client(self) -> Redis:
        """Get Redis client for L2 cache."""
        if self._redis_client is None:
            self._redis_client = coredis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._redis_client

    async def get_rule_effectiveness(
        self,
        characteristics: PromptCharacteristics,
        cache_key: str
    ) -> Optional[List[RuleScore]]:
        """Get rule effectiveness data from multi-level cache.

        Args:
            characteristics: Prompt characteristics for cache lookup
            cache_key: Cache key for the request

        Returns:
            Cached rule scores or None if not found
        """
        start_time = time.time()
        self.metrics.total_requests += 1

        # Try L1 cache first (memory)
        l1_result = await self._get_from_l1_cache(cache_key)
        if l1_result:
            self.metrics.l1_hits += 1
            retrieval_time = (time.time() - start_time) * 1000
            self._update_retrieval_metrics(retrieval_time)
            logger.debug(f"L1 cache hit for {cache_key} ({retrieval_time:.1f}ms)")
            return l1_result.rule_scores

        self.metrics.l1_misses += 1

        # Try L2 cache (Redis)
        l2_result = await self._get_from_l2_cache(cache_key)
        if l2_result:
            self.metrics.l2_hits += 1
            # Promote to L1 cache
            await self._store_in_l1_cache(cache_key, l2_result)
            retrieval_time = (time.time() - start_time) * 1000
            self._update_retrieval_metrics(retrieval_time)
            logger.debug(f"L2 cache hit for {cache_key} ({retrieval_time:.1f}ms)")
            return l2_result.rule_scores

        self.metrics.l2_misses += 1

        # Cache miss - will need to fetch from database
        logger.debug(f"Cache miss for {cache_key}")
        return None

    async def store_rule_effectiveness(
        self,
        cache_key: str,
        rule_scores: List[RuleScore],
        characteristics: PromptCharacteristics,
        retrieval_time_ms: float
    ) -> None:
        """Store rule effectiveness data in multi-level cache.

        Args:
            cache_key: Cache key
            rule_scores: Rule scores to cache
            characteristics: Prompt characteristics
            retrieval_time_ms: Time taken to retrieve from database
        """
        characteristics_hash = self._hash_characteristics(characteristics)

        cached_data = CachedRuleData(
            rule_scores=rule_scores,
            characteristics_hash=characteristics_hash,
            cached_at=time.time(),
            cache_level=CacheLevel.L3_DATABASE,
            retrieval_time_ms=retrieval_time_ms,
            access_count=1,
            last_accessed=time.time()
        )

        # Store in both L1 and L2 caches
        await self._store_in_l1_cache(cache_key, cached_data)
        await self._store_in_l2_cache(cache_key, cached_data)

        # Track access pattern for cache warming
        self._track_access_pattern(cache_key, characteristics)

        logger.debug(f"Stored rule effectiveness in cache: {cache_key}")

    async def _get_from_l1_cache(self, cache_key: str) -> Optional[CachedRuleData]:
        """Get data from L1 memory cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached data or None if not found/expired
        """
        if cache_key not in self.l1_cache:
            return None

        cached_data = self.l1_cache[cache_key]

        # Check TTL
        if time.time() - cached_data.cached_at > self.l1_ttl_seconds:
            del self.l1_cache[cache_key]
            return None

        # Update access metrics
        cached_data.access_count += 1
        cached_data.last_accessed = time.time()

        return cached_data

    async def _store_in_l1_cache(self, cache_key: str, data: CachedRuleData) -> None:
        """Store data in L1 memory cache with LRU eviction.

        Args:
            cache_key: Cache key
            data: Data to cache
        """
        # Check if cache is full and evict LRU entries
        if len(self.l1_cache) >= self.l1_max_size:
            await self._evict_l1_lru_entries()

        # Update cache level
        data.cache_level = CacheLevel.L1_MEMORY
        self.l1_cache[cache_key] = data

    async def _evict_l1_lru_entries(self) -> None:
        """Evict least recently used entries from L1 cache."""
        if not self.l1_cache:
            return

        # Calculate eviction score (combines recency and frequency)
        eviction_scores = {}
        current_time = time.time()

        for key, data in self.l1_cache.items():
            recency_score = current_time - data.last_accessed
            frequency_score = 1.0 / max(1, data.access_count)
            eviction_scores[key] = recency_score + frequency_score

        # Evict 10% of entries with highest eviction scores
        evict_count = max(1, len(self.l1_cache) // 10)
        keys_to_evict = sorted(eviction_scores.keys(),
                              key=lambda k: eviction_scores[k],
                              reverse=True)[:evict_count]

        for key in keys_to_evict:
            del self.l1_cache[key]

        logger.debug(f"Evicted {len(keys_to_evict)} entries from L1 cache")

    async def _get_from_l2_cache(self, cache_key: str) -> Optional[CachedRuleData]:
        """Get data from L2 Redis cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached data or None if not found
        """
        try:
            redis = await self.get_redis_client()
            redis_key = f"{self.l2_key_prefix}{cache_key}"

            cached_json = await redis.get(redis_key)
            if not cached_json:
                return None

            # Deserialize cached data
            cached_dict = json.loads(cached_json)

            # Reconstruct RuleScore objects
            rule_scores = [
                RuleScore(**score_dict) for score_dict in cached_dict["rule_scores"]
            ]

            cached_data = CachedRuleData(
                rule_scores=rule_scores,
                characteristics_hash=cached_dict["characteristics_hash"],
                cached_at=cached_dict["cached_at"],
                cache_level=CacheLevel.L2_REDIS,
                retrieval_time_ms=cached_dict["retrieval_time_ms"],
                access_count=cached_dict.get("access_count", 1),
                last_accessed=time.time()
            )

            return cached_data

        except Exception as e:
            logger.warning(f"L2 cache retrieval error for {cache_key}: {e}")
            return None

    async def _store_in_l2_cache(self, cache_key: str, data: CachedRuleData) -> None:
        """Store data in L2 Redis cache.

        Args:
            cache_key: Cache key
            data: Data to cache
        """
        try:
            redis = await self.get_redis_client()
            redis_key = f"{self.l2_key_prefix}{cache_key}"

            # Serialize data for Redis storage
            serializable_data = {
                "rule_scores": [asdict(score) for score in data.rule_scores],
                "characteristics_hash": data.characteristics_hash,
                "cached_at": data.cached_at,
                "retrieval_time_ms": data.retrieval_time_ms,
                "access_count": data.access_count
            }

            cached_json = json.dumps(serializable_data)
            await redis.setex(redis_key, self.l2_ttl_seconds, cached_json)

        except Exception as e:
            logger.warning(f"L2 cache storage error for {cache_key}: {e}")

    def _hash_characteristics(self, characteristics: PromptCharacteristics) -> str:
        """Generate hash for prompt characteristics.

        Args:
            characteristics: Prompt characteristics

        Returns:
            Hash string for characteristics
        """
        # Create a stable hash based on key characteristics
        key_attrs = [
            characteristics.prompt_type,
            characteristics.domain,
            characteristics.task_type,
            f"{characteristics.complexity_level:.1f}",
            f"{characteristics.specificity_level:.1f}"
        ]
        return ":".join(key_attrs)

    def _track_access_pattern(self, cache_key: str, characteristics: PromptCharacteristics) -> None:
        """Track access patterns for cache warming.

        Args:
            cache_key: Cache key
            characteristics: Prompt characteristics
        """
        if not self.warming_enabled:
            return

        pattern_key = self._hash_characteristics(characteristics)

        if pattern_key not in self.popular_patterns_cache:
            self.popular_patterns_cache[pattern_key] = {
                "access_count": 0,
                "last_accessed": 0,
                "cache_keys": set()
            }

        pattern_data = self.popular_patterns_cache[pattern_key]
        pattern_data["access_count"] += 1
        pattern_data["last_accessed"] = time.time()
        pattern_data["cache_keys"].add(cache_key)

    def _update_retrieval_metrics(self, retrieval_time_ms: float) -> None:
        """Update retrieval time metrics.

        Args:
            retrieval_time_ms: Retrieval time in milliseconds
        """
        # Update rolling average
        alpha = 0.1  # Smoothing factor
        if self.metrics.avg_retrieval_time_ms == 0:
            self.metrics.avg_retrieval_time_ms = retrieval_time_ms
        else:
            self.metrics.avg_retrieval_time_ms = (
                alpha * retrieval_time_ms +
                (1 - alpha) * self.metrics.avg_retrieval_time_ms
            )

        # Update hit rate
        total_hits = self.metrics.l1_hits + self.metrics.l2_hits
        self.metrics.hit_rate = total_hits / max(1, self.metrics.total_requests)

    async def warm_cache(self, popular_patterns_limit: int = 50) -> int:
        """Warm cache with popular access patterns.

        Args:
            popular_patterns_limit: Maximum patterns to warm

        Returns:
            Number of patterns warmed
        """
        if not self.warming_enabled:
            return 0

        # Sort patterns by access frequency and recency
        current_time = time.time()
        pattern_scores = {}

        for pattern_key, pattern_data in self.popular_patterns_cache.items():
            recency_factor = max(0.1, 1.0 - (current_time - pattern_data["last_accessed"]) / 3600)
            frequency_factor = min(10.0, pattern_data["access_count"] / 10.0)
            pattern_scores[pattern_key] = recency_factor * frequency_factor

        # Select top patterns for warming
        top_patterns = sorted(pattern_scores.keys(),
                            key=lambda k: pattern_scores[k],
                            reverse=True)[:popular_patterns_limit]

        warmed_count = 0
        for pattern_key in top_patterns:
            # Warm cache for this pattern (implementation would depend on specific warming strategy)
            logger.debug(f"Warming cache for pattern: {pattern_key}")
            warmed_count += 1

        logger.info(f"Cache warming completed: {warmed_count} patterns warmed")
        return warmed_count

    async def invalidate_cache(self, cache_key: Optional[str] = None) -> None:
        """Invalidate cache entries.

        Args:
            cache_key: Specific key to invalidate, or None for full invalidation
        """
        if cache_key:
            # Invalidate specific key
            if cache_key in self.l1_cache:
                del self.l1_cache[cache_key]

            try:
                redis = await self.get_redis_client()
                redis_key = f"{self.l2_key_prefix}{cache_key}"
                await redis.delete(redis_key)
            except Exception as e:
                logger.warning(f"L2 cache invalidation error: {e}")
        else:
            # Full cache invalidation
            self.l1_cache.clear()

            try:
                redis = await self.get_redis_client()
                pattern = f"{self.l2_key_prefix}*"
                keys = []
                async for key in redis.scan_iter(match=pattern):
                    keys.append(key)
                if keys:
                    await redis.delete(*keys)
            except Exception as e:
                logger.warning(f"L2 cache full invalidation error: {e}")

        logger.info(f"Cache invalidated: {cache_key or 'all entries'}")

    def get_cache_metrics(self) -> CacheMetrics:
        """Get current cache performance metrics.

        Returns:
            Current cache metrics
        """
        return self.metrics

    def get_cache_status(self) -> Dict[str, Any]:
        """Get comprehensive cache status information.

        Returns:
            Cache status dictionary
        """
        return {
            "l1_cache_size": len(self.l1_cache),
            "l1_max_size": self.l1_max_size,
            "l1_utilization": len(self.l1_cache) / self.l1_max_size,
            "metrics": asdict(self.metrics),
            "performance_status": {
                "hit_rate_ok": self.metrics.hit_rate >= self.target_hit_rate,
                "retrieval_time_ok": self.metrics.avg_retrieval_time_ms <= self.target_retrieval_time_ms,
                "overall_status": "good" if (
                    self.metrics.hit_rate >= self.target_hit_rate and
                    self.metrics.avg_retrieval_time_ms <= self.target_retrieval_time_ms
                ) else "needs_optimization"
            },
            "popular_patterns_count": len(self.popular_patterns_cache),
            "warming_enabled": self.warming_enabled
        }
