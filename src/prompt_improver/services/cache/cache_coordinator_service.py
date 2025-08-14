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
from typing import Any, Optional

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

    def __init__(
        self,
        l1_cache: L1CacheService | None = None,
        l2_cache: L2RedisService | None = None,
        l3_cache: L3DatabaseService | None = None,
        enable_warming: bool = True,
        warming_threshold: float = 2.0,
        warming_interval: int = 300,
        max_warming_keys: int = 50,
    ) -> None:
        """Initialize cache coordinator.
        
        Args:
            l1_cache: L1 cache service instance
            l2_cache: L2 cache service instance  
            l3_cache: L3 cache service instance
            enable_warming: Enable intelligent cache warming
            warming_threshold: Access frequency threshold for warming
            warming_interval: Warming cycle interval in seconds
            max_warming_keys: Maximum keys to warm per cycle
        """
        # Cache level services
        self._l1_cache = l1_cache or L1CacheService()
        self._l2_cache = l2_cache
        self._l3_cache = l3_cache
        
        # Cache warming configuration
        self._enable_warming = enable_warming
        self._warming_threshold = warming_threshold
        self._warming_interval = warming_interval
        self._max_warming_keys = max_warming_keys
        
        # Access pattern tracking for warming
        self._access_patterns: dict[str, AccessPattern] = {}
        self._last_warming_time = datetime.now(UTC)
        
        # Background warming task
        self._background_warming_task: asyncio.Task | None = None
        self._warming_stop_event = asyncio.Event()
        
        # Performance tracking
        self._l1_hits = 0
        self._l2_hits = 0
        self._l3_hits = 0
        self._total_requests = 0
        self._total_response_time = 0.0
        self._created_at = datetime.now(UTC)
        
        # Start warming if enabled
        if self._enable_warming:
            self._start_background_warming()

    async def get(
        self,
        key: str,
        fallback_func: Optional[Callable[[], Any]] = None,
        l2_ttl: Optional[int] = None,
        l1_ttl: Optional[int] = None,
    ) -> Optional[Any]:
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
            response_time = time.perf_counter() - operation_start
            self._total_response_time += response_time
            
            # Log slow operations
            if response_time > 0.05:  # 50ms
                logger.warning(
                    f"Cache coordinator GET operation took {response_time*1000:.2f}ms (key: {key[:50]}...)"
                )

    async def set(
        self,
        key: str,
        value: Any,
        l2_ttl: Optional[int] = None,
        l1_ttl: Optional[int] = None,
    ) -> None:
        """Set value in multi-level cache.
        
        Sets value in all available cache levels for consistency.
        
        Args:
            key: Cache key
            value: Value to cache
            l2_ttl: TTL for L2 cache
            l1_ttl: TTL for L1 cache
        """
        operation_start = time.perf_counter()
        
        try:
            # Set in all cache levels concurrently
            tasks = []
            
            # Always set in L1
            tasks.append(self._l1_cache.set(key, value, l1_ttl))
            
            # Set in L2 if available
            if self._l2_cache:
                tasks.append(self._l2_cache.set(key, value, l2_ttl))
            
            # Set in L3 if available
            if self._l3_cache:
                tasks.append(self._l3_cache.set(key, value, l2_ttl))
            
            # Execute all sets concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
            
        finally:
            response_time = time.perf_counter() - operation_start
            
            if response_time > 0.05:
                logger.warning(
                    f"Cache coordinator SET operation took {response_time*1000:.2f}ms (key: {key[:50]}...)"
                )

    async def delete(self, key: str) -> None:
        """Delete key from all cache levels.
        
        Args:
            key: Cache key to delete
        """
        operation_start = time.perf_counter()
        
        try:
            # Delete from all cache levels concurrently
            tasks = []
            
            tasks.append(self._l1_cache.delete(key))
            
            if self._l2_cache:
                tasks.append(self._l2_cache.delete(key))
            
            if self._l3_cache:
                tasks.append(self._l3_cache.delete(key))
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        finally:
            response_time = time.perf_counter() - operation_start
            
            if response_time > 0.05:
                logger.warning(
                    f"Cache coordinator DELETE operation took {response_time*1000:.2f}ms (key: {key[:50]}...)"
                )

    async def clear(self) -> None:
        """Clear all cache levels."""
        tasks = []
        
        tasks.append(self._l1_cache.clear())
        
        if self._l2_cache:
            tasks.append(self._l2_cache.clear())
        
        if self._l3_cache:
            tasks.append(self._l3_cache.clear())
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Reset statistics
        self._l1_hits = 0
        self._l2_hits = 0
        self._l3_hits = 0
        self._total_requests = 0
        self._total_response_time = 0.0
        self._access_patterns.clear()

    async def warm_cache(self, keys: list[str]) -> dict[str, bool]:
        """Manually warm cache with specific keys.
        
        Attempts to promote keys from L3 -> L2 -> L1.
        
        Args:
            keys: List of cache keys to warm
            
        Returns:
            Dictionary mapping keys to warming success status
        """
        results = {}
        
        for key in keys:
            try:
                # Try to warm from L3 to L2 and L1
                warmed = False
                
                if self._l3_cache:
                    l3_value = await self._l3_cache.get(key)
                    if l3_value is not None:
                        # Promote to L1 and L2
                        await self._l1_cache.set(key, l3_value)
                        if self._l2_cache:
                            await self._l2_cache.set(key, l3_value)
                        warmed = True
                
                # If not in L3, try L2 to L1
                if not warmed and self._l2_cache:
                    l2_value = await self._l2_cache.get(key)
                    if l2_value is not None:
                        await self._l1_cache.set(key, l2_value)
                        warmed = True
                
                results[key] = warmed
                
            except Exception as e:
                logger.warning(f"Failed to warm cache key {key}: {e}")
                results[key] = False
        
        return results

    def _start_background_warming(self) -> None:
        """Start background cache warming task."""
        if self._background_warming_task is None:
            self._background_warming_task = asyncio.create_task(
                self._background_warming_loop()
            )

    async def _background_warming_loop(self) -> None:
        """Background loop for intelligent cache warming."""
        logger.info("Starting cache warming background task")
        
        try:
            while not self._warming_stop_event.is_set():
                try:
                    # Wait for warming interval or stop event
                    await asyncio.wait_for(
                        self._warming_stop_event.wait(),
                        timeout=self._warming_interval
                    )
                except asyncio.TimeoutError:
                    # Perform warming cycle
                    await self._perform_warming_cycle()
                    
        except asyncio.CancelledError:
            logger.info("Cache warming background task cancelled")
        except Exception as e:
            logger.error(f"Error in cache warming loop: {e}")

    async def _perform_warming_cycle(self) -> None:
        """Perform one cycle of intelligent cache warming."""
        if not self._enable_warming:
            return
            
        warming_start = time.perf_counter()
        hot_keys = self._get_hot_keys_for_warming()
        
        if not hot_keys:
            return
        
        logger.debug(f"Warming {len(hot_keys)} hot cache keys")
        
        # Warm keys in batches to avoid overwhelming the system
        batch_size = min(10, self._max_warming_keys)
        for i in range(0, len(hot_keys), batch_size):
            batch = hot_keys[i:i + batch_size]
            batch_keys = [key for key, _ in batch]
            
            # Warm batch
            await self.warm_cache(batch_keys)
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        warming_duration = time.perf_counter() - warming_start
        self._last_warming_time = datetime.now(UTC)
        
        logger.debug(
            f"Cache warming cycle completed: processed {len(hot_keys)} keys in {warming_duration:.2f}s"
        )

    def _get_hot_keys_for_warming(self) -> list[tuple[str, "AccessPattern"]]:
        """Get keys that should be warmed based on access patterns."""
        now = datetime.now(UTC)
        hot_keys = []
        
        for key, pattern in self._access_patterns.items():
            # Skip keys accessed too long ago
            if (now - pattern.last_access).total_seconds() > 3600:  # 1 hour
                continue
                
            # Check if access frequency meets threshold
            if pattern.access_frequency >= self._warming_threshold:
                hot_keys.append((key, pattern))
        
        # Sort by warming priority (frequency + recency)
        hot_keys.sort(key=lambda x: x[1].warming_priority, reverse=True)
        
        return hot_keys[:self._max_warming_keys]

    def _record_access_pattern(self, key: str) -> None:
        """Record access pattern for intelligent warming."""
        if not self._enable_warming:
            return
            
        if key not in self._access_patterns:
            self._access_patterns[key] = AccessPattern(key=key)
        
        self._access_patterns[key].record_access()
        
        # Cleanup old patterns periodically
        if len(self._access_patterns) > 10000:
            self._cleanup_old_patterns()

    def _cleanup_old_patterns(self) -> None:
        """Clean up old access patterns to prevent memory growth."""
        now = datetime.now(UTC)
        cutoff = now - timedelta(days=1)
        
        old_keys = [
            key for key, pattern in self._access_patterns.items()
            if pattern.last_access < cutoff
        ]
        
        for key in old_keys:
            del self._access_patterns[key]
        
        logger.debug(f"Cleaned up {len(old_keys)} old access patterns")

    async def stop_warming(self) -> None:
        """Stop background cache warming."""
        self._enable_warming = False
        self._warming_stop_event.set()
        
        if self._background_warming_task and not self._background_warming_task.done():
            self._background_warming_task.cancel()
            try:
                await self._background_warming_task
            except asyncio.CancelledError:
                pass
        
        self._background_warming_task = None
        logger.info("Cache warming stopped")

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive cache performance statistics.
        
        Returns:
            Performance statistics across all cache levels
        """
        total_hits = self._l1_hits + self._l2_hits + self._l3_hits
        overall_hit_rate = total_hits / self._total_requests if self._total_requests > 0 else 0
        avg_response_time = (
            self._total_response_time / self._total_requests 
            if self._total_requests > 0 else 0
        )
        
        return {
            # Overall metrics
            "total_requests": self._total_requests,
            "overall_hit_rate": overall_hit_rate,
            "avg_response_time_ms": avg_response_time * 1000,
            
            # Level-specific hits
            "l1_hits": self._l1_hits,
            "l2_hits": self._l2_hits,
            "l3_hits": self._l3_hits,
            
            # Hit rates by level
            "l1_hit_rate": self._l1_hits / self._total_requests if self._total_requests > 0 else 0,
            "l2_hit_rate": self._l2_hits / self._total_requests if self._total_requests > 0 else 0,
            "l3_hit_rate": self._l3_hits / self._total_requests if self._total_requests > 0 else 0,
            
            # Cache warming
            "warming_enabled": self._enable_warming,
            "tracked_patterns": len(self._access_patterns),
            "last_warming_time": self._last_warming_time.isoformat(),
            
            # Individual cache stats
            "l1_cache_stats": self._l1_cache.get_stats(),
            "l2_cache_stats": self._l2_cache.get_stats() if self._l2_cache else None,
            "l3_cache_stats": self._l3_cache.get_stats() if self._l3_cache else None,
            
            # Health and uptime
            "health_status": self._get_health_status(),
            "uptime_seconds": (datetime.now(UTC) - self._created_at).total_seconds(),
        }

    def _get_health_status(self) -> str:
        """Get overall health status based on all cache levels."""
        if self._total_requests == 0:
            return "healthy"
            
        # Check individual cache health
        l1_health = self._l1_cache.get_stats()["health_status"]
        l2_health = self._l2_cache.get_stats()["health_status"] if self._l2_cache else "healthy"
        l3_health = self._l3_cache.get_stats()["health_status"] if self._l3_cache else "healthy"
        
        # Overall response time
        avg_response_time = self._total_response_time / self._total_requests
        
        # Determine overall health
        healths = [l1_health, l2_health, l3_health]
        
        if "unhealthy" in healths or avg_response_time > 0.1:  # 100ms
            return "unhealthy"
        elif "degraded" in healths or avg_response_time > 0.05:  # 50ms
            return "degraded"
        else:
            return "healthy"


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
        """Record a new access and update frequency metrics."""
        now = datetime.now(UTC)
        self.access_count += 1
        self.last_access = now
        self.access_times.append(now)
        
        # Keep only recent access times (24 hours)
        cutoff = now - timedelta(hours=24)
        self.access_times = [t for t in self.access_times if t > cutoff]
        
        # Calculate frequency (accesses per hour)
        if len(self.access_times) >= 2:
            time_span = (self.access_times[-1] - self.access_times[0]).total_seconds() / 3600
            self.access_frequency = len(self.access_times) / max(time_span, 0.1)
        
        # Calculate warming priority (frequency + recency weight)
        recency_weight = max(0.0, 1 - (now - self.last_access).total_seconds() / 3600)
        self.warming_priority = self.access_frequency * (1 + recency_weight)