"""Cache Coordinator Service for orchestrating multi-level cache operations.

Coordinates between L1, L2, and L3 cache services to provide unified caching
interface with intelligent cache warming and fallback strategies. Maintains
strict performance targets while ensuring data consistency across levels.
"""

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # AccessPattern class is defined below

from .l1_cache_service import L1CacheService
from .l2_redis_service import L2RedisService
from .l3_database_service import L3DatabaseService

logger = logging.getLogger(__name__)


class CacheCoordinatorService:
    """Coordinates multi-level cache operations with intelligent warming.
    
    Orchestrates between L1 (memory), L2 (Redis), and L3 (database) cache
    levels to provide optimal performance while maintaining data consistency.
    Includes intelligent cache warming based on access patterns.
    
    Performance targets:
    - Overall operation: <50ms (with fallback chain)
    - L1 hit: <1ms
    - L2 hit: <10ms  
    - L3 hit: <50ms
    """

    def __init__(self, l1_cache: L1CacheService | None = None, l2_cache: L2RedisService | None = None, l3_cache: L3DatabaseService | None = None, enable_warming: bool = True, warming_threshold: float = 2.0, warming_interval: int = 300, max_warming_keys: int = 50) -> None:
        """Initialize cache coordinator."""
        self._l1_cache, self._l2_cache, self._l3_cache = l1_cache or L1CacheService(), l2_cache, l3_cache
        self._enable_warming, self._warming_threshold, self._warming_interval, self._max_warming_keys = enable_warming, warming_threshold, warming_interval, max_warming_keys
        self._access_patterns: dict[str, AccessPattern] = {}
        self._last_warming_time = datetime.now(UTC)
        self._background_warming_task, self._warming_stop_event = None, asyncio.Event()
        self._l1_hits = self._l2_hits = self._l3_hits = self._total_requests = 0
        self._total_response_time, self._created_at = 0.0, datetime.now(UTC)
        if self._enable_warming: self._start_background_warming()

    def _track_operation(self, start_time: float, operation: str, key: str = "", pattern: str = "") -> None:
        """Track operation performance and log slow operations."""
        response_time = time.perf_counter() - start_time
        self._total_response_time += response_time
        
        # Log slow operations (should be <50ms for coordinator)
        if response_time > 0.05:
            identifier = key or pattern
            logger.warning(f"Cache coordinator {operation} took {response_time*1000:.2f}ms ({identifier[:50]}...)")

    async def get(
        self,
        key: str,
        fallback_func: Callable[[], Any] | None = None,
        l2_ttl: int | None = None,
        l1_ttl: int | None = None,
    ) -> Any | None:
        """Get value from multi-level cache with fallback.
        
        Tries L1 -> L2 -> L3 -> fallback_func in order.
        Promotes cache hits to higher levels for future performance.
        
        Args:
            key: Cache key
            fallback_func: Function to call if not in any cache level
            l2_ttl: TTL for L2 cache when promoting/setting
            l1_ttl: TTL for L1 cache when promoting/setting
            
        Returns:
            Cached or computed value
        """
        operation_start = time.perf_counter()
        self._total_requests += 1
        
        # Record access pattern for warming
        self._record_access_pattern(key)
        
        try:
            # Try L1 cache first
            l1_value = await self._l1_cache.get(key)
            if l1_value is not None:
                self._l1_hits += 1
                return l1_value
            
            # Try L2 cache
            if self._l2_cache:
                l2_value = await self._l2_cache.get(key)
                if l2_value is not None:
                    self._l2_hits += 1
                    # Promote to L1
                    await self._l1_cache.set(key, l2_value, l1_ttl)
                    return l2_value
            
            # Try L3 cache
            if self._l3_cache:
                l3_value = await self._l3_cache.get(key)
                if l3_value is not None:
                    self._l3_hits += 1
                    # Promote to L1 and L2
                    await self._l1_cache.set(key, l3_value, l1_ttl)
                    if self._l2_cache:
                        await self._l2_cache.set(key, l3_value, l2_ttl)
                    return l3_value
            
            # Use fallback function if provided
            if fallback_func:
                result = fallback_func()
                value = await result if inspect.isawaitable(result) else result
                
                if value is not None:
                    # Cache the computed value in all levels
                    await self.set(key, value, l2_ttl, l1_ttl)
                    return value
            
            return None
            
        finally:
            self._track_operation(operation_start, "GET", key)

    async def set(
        self,
        key: str,
        value: Any,
        l2_ttl: int | None = None,
        l1_ttl: int | None = None,
    ) -> None:
        """Set value in multi-level cache."""
        operation_start = time.perf_counter()
        
        try:
            tasks = [self._l1_cache.set(key, value, l1_ttl)]
            
            if self._l2_cache:
                tasks.append(self._l2_cache.set(key, value, l2_ttl))
            if self._l3_cache:
                tasks.append(self._l3_cache.set(key, value, l2_ttl))
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        finally:
            self._track_operation(operation_start, "SET", key)

    async def delete(self, key: str) -> None:
        """Delete key from all cache levels."""
        operation_start = time.perf_counter()
        
        try:
            tasks = [self._l1_cache.delete(key)]
            
            if self._l2_cache:
                tasks.append(self._l2_cache.delete(key))
            if self._l3_cache:
                tasks.append(self._l3_cache.delete(key))
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        finally:
            self._track_operation(operation_start, "DELETE", key)

    async def clear(self) -> None:
        """Clear all cache levels."""
        tasks = [self._l1_cache.clear()]
        if self._l2_cache: tasks.append(self._l2_cache.clear())
        if self._l3_cache: tasks.append(self._l3_cache.clear())
        await asyncio.gather(*tasks, return_exceptions=True)
        self._l1_hits = self._l2_hits = self._l3_hits = self._total_requests = 0
        self._total_response_time = 0.0
        self._access_patterns.clear()

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern across all levels."""
        operation_start = time.perf_counter()
        total_invalidated = 0
        
        try:
            tasks = [self._invalidate_l1_pattern(pattern)]
            
            if self._l2_cache:
                tasks.append(self._invalidate_l2_pattern(pattern))
            if self._l3_cache:
                tasks.append(self._invalidate_l3_pattern(pattern))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, int):
                    total_invalidated += result
                elif isinstance(result, Exception):
                    logger.warning(f"Pattern invalidation failed: {result}")
            
            logger.debug(f"Pattern invalidation '{pattern}' removed {total_invalidated} entries")
            return total_invalidated
            
        finally:
            self._track_operation(operation_start, "INVALIDATE_PATTERN", pattern=pattern)

    async def warm_cache(self, keys: list[str]) -> dict[str, bool]:
        """Manually warm cache with specific keys."""
        results = {}
        for key in keys:
            try:
                warmed = False
                if self._l3_cache and (l3_val := await self._l3_cache.get(key)):
                    await self._l1_cache.set(key, l3_val)
                    if self._l2_cache: await self._l2_cache.set(key, l3_val)
                    warmed = True
                elif self._l2_cache and (l2_val := await self._l2_cache.get(key)):
                    await self._l1_cache.set(key, l2_val)
                    warmed = True
                results[key] = warmed
            except Exception as e:
                logger.warning(f"Failed to warm cache key {key}: {e}")
                results[key] = False
        return results

    def _start_background_warming(self) -> None:
        """Start background cache warming task."""
        if self._background_warming_task is None: self._background_warming_task = asyncio.create_task(self._background_warming_loop())

    async def _background_warming_loop(self) -> None:
        """Background loop for intelligent cache warming."""
        logger.info("Starting cache warming background task")
        try:
            while not self._warming_stop_event.is_set():
                try:
                    await asyncio.wait_for(self._warming_stop_event.wait(), timeout=self._warming_interval)
                except asyncio.TimeoutError:
                    await self._perform_warming_cycle()
        except asyncio.CancelledError:
            logger.info("Cache warming background task cancelled")
        except Exception as e:
            logger.error(f"Error in cache warming loop: {e}")

    async def _perform_warming_cycle(self) -> None:
        """Perform one cycle of intelligent cache warming."""
        if not self._enable_warming or not (hot_keys := self._get_hot_keys_for_warming()): return
        logger.debug(f"Warming {len(hot_keys)} hot cache keys")
        batch_size = min(10, self._max_warming_keys)
        for i in range(0, len(hot_keys), batch_size):
            await self.warm_cache([key for key, _ in hot_keys[i:i + batch_size]])
            await asyncio.sleep(0.1)
        self._last_warming_time = datetime.now(UTC)

    def _get_hot_keys_for_warming(self) -> list[tuple[str, "AccessPattern"]]:
        """Get keys that should be warmed based on access patterns."""
        now = datetime.now(UTC)
        hot_keys = [(k, p) for k, p in self._access_patterns.items() if (now - p.last_access).total_seconds() <= 3600 and p.access_frequency >= self._warming_threshold]
        return sorted(hot_keys, key=lambda x: x[1].warming_priority, reverse=True)[:self._max_warming_keys]

    def _record_access_pattern(self, key: str) -> None:
        """Record access pattern for intelligent warming."""
        if not self._enable_warming: return
        if key not in self._access_patterns: self._access_patterns[key] = AccessPattern(key=key)
        self._access_patterns[key].record_access()
        if len(self._access_patterns) > 10000: self._cleanup_old_patterns()

    def _cleanup_old_patterns(self) -> None:
        """Clean up old access patterns to prevent memory growth."""
        cutoff = datetime.now(UTC) - timedelta(days=1)
        old_keys = [k for k, p in self._access_patterns.items() if p.last_access < cutoff]
        for key in old_keys: del self._access_patterns[key]
        if old_keys: logger.debug(f"Cleaned up {len(old_keys)} old access patterns")

    async def stop_warming(self) -> None:
        """Stop background cache warming."""
        self._enable_warming = False
        self._warming_stop_event.set()
        if self._background_warming_task and not self._background_warming_task.done():
            self._background_warming_task.cancel()
            try: await self._background_warming_task
            except asyncio.CancelledError: pass
        self._background_warming_task = None
        logger.info("Cache warming stopped")

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive cache performance statistics."""
        total_hits = self._l1_hits + self._l2_hits + self._l3_hits
        reqs = self._total_requests
        hit_rate = total_hits / reqs if reqs > 0 else 0
        resp_time = self._total_response_time / reqs if reqs > 0 else 0
        
        return {
            "total_requests": reqs, "overall_hit_rate": hit_rate, "avg_response_time_ms": resp_time * 1000,
            "l1_hits": self._l1_hits, "l2_hits": self._l2_hits, "l3_hits": self._l3_hits,
            "l1_hit_rate": self._l1_hits / reqs if reqs > 0 else 0,
            "l2_hit_rate": self._l2_hits / reqs if reqs > 0 else 0,
            "l3_hit_rate": self._l3_hits / reqs if reqs > 0 else 0,
            "warming_enabled": self._enable_warming, "tracked_patterns": len(self._access_patterns),
            "last_warming_time": self._last_warming_time.isoformat(),
            "l1_cache_stats": self._l1_cache.get_stats(),
            "l2_cache_stats": self._l2_cache.get_stats() if self._l2_cache else None,
            "l3_cache_stats": self._l3_cache.get_stats() if self._l3_cache else None,
            "health_status": self._get_health_status(),
            "uptime_seconds": (datetime.now(UTC) - self._created_at).total_seconds(),
        }

    def _get_health_status(self) -> str:
        """Get overall health status based on all cache levels."""
        if self._total_requests == 0: return "healthy"
        healths = [self._l1_cache.get_stats()["health_status"], self._l2_cache.get_stats()["health_status"] if self._l2_cache else "healthy", self._l3_cache.get_stats()["health_status"] if self._l3_cache else "healthy"]
        avg_time = self._total_response_time / self._total_requests
        return "unhealthy" if "unhealthy" in healths or avg_time > 0.1 else "degraded" if "degraded" in healths or avg_time > 0.05 else "healthy"

    async def _invalidate_l1_pattern(self, pattern: str) -> int:
        """Invalidate L1 cache entries matching pattern."""
        try:
            return await self._l1_cache.invalidate_pattern(pattern)
        except Exception as e:
            logger.warning(f"L1 pattern invalidation failed '{pattern}': {e}")
            return 0

    async def _invalidate_l2_pattern(self, pattern: str) -> int:
        """Invalidate L2 cache entries matching pattern."""
        try:
            if self._l2_cache and hasattr(self._l2_cache, 'invalidate_pattern'):
                return await self._l2_cache.invalidate_pattern(pattern)
            return 0
        except Exception as e:
            logger.warning(f"L2 pattern invalidation failed '{pattern}': {e}")
            return 0

    async def _invalidate_l3_pattern(self, pattern: str) -> int:
        """Invalidate L3 cache entries matching pattern."""
        try:
            if self._l3_cache and hasattr(self._l3_cache, 'invalidate_pattern'):
                return await self._l3_cache.invalidate_pattern(pattern)
            return 0
        except Exception as e:
            logger.warning(f"L3 pattern invalidation failed '{pattern}': {e}")
            return 0


class AccessPattern:
    """Access pattern tracking for cache warming."""
    def __init__(self, key: str) -> None:
        self.key = key
        self.access_count = 0
        self.last_access = datetime.now(UTC)
        self.access_frequency = 0.0
        self.access_times: list[datetime] = []
        self.warming_priority = 0.0

    def record_access(self) -> None:
        """Record access and update frequency metrics."""
        now = datetime.now(UTC)
        self.access_count += 1
        self.last_access = now
        self.access_times.append(now)
        cutoff = now - timedelta(hours=24)
        self.access_times = [t for t in self.access_times if t > cutoff]
        if len(self.access_times) >= 2:
            time_span = (self.access_times[-1] - self.access_times[0]).total_seconds() / 3600
            self.access_frequency = len(self.access_times) / max(time_span, 0.1)
        recency_weight = max(0.0, 1 - (now - self.last_access).total_seconds() / 3600)
        self.warming_priority = self.access_frequency * (1 + recency_weight)