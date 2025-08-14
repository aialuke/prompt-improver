"""Cache Facade for unified multi-level cache access.

Provides a clean, unified interface to the decomposed cache services while
maintaining backward compatibility and optimal performance. Replaces the
original god object with focused, testable services.
"""

import logging
from typing import Any, Optional

from prompt_improver.core.types import CacheType

from .cache_coordinator_service import CacheCoordinatorService
from .cache_monitoring_service import CacheMonitoringService
from .l1_cache_service import L1CacheService
from .l2_redis_service import L2RedisService
from .l3_database_service import L3DatabaseService

logger = logging.getLogger(__name__)


class CacheFacade:
    """Unified facade for multi-level cache operations.
    
    Provides backward-compatible interface while using decomposed services
    internally. Maintains the same API as the original MultiLevelCache
    while achieving better separation of concerns and testability.
    
    Performance targets maintained:
    - L1 operations: <1ms
    - L2 operations: <10ms
    - L3 operations: <50ms
    - Overall operations: <200ms
    """

    def __init__(
        self,
        l1_max_size: int = 1000,
        l2_default_ttl: int = 3600,
        enable_l2: bool = True,
        enable_l3: bool = False,
        enable_warming: bool = True,
        session_manager: Any = None,
    ) -> None:
        """Initialize cache facade with decomposed services.
        
        Args:
            l1_max_size: Maximum size for L1 cache
            l2_default_ttl: Default TTL for L2 cache
            enable_l2: Enable L2 (Redis) cache
            enable_l3: Enable L3 (Database) cache
            enable_warming: Enable intelligent cache warming
            session_manager: Database session manager for L3 cache
        """
        # Initialize individual cache services
        self._l1_cache = L1CacheService(max_size=l1_max_size)
        self._l2_cache = L2RedisService() if enable_l2 else None
        self._l3_cache = L3DatabaseService(session_manager) if enable_l3 else None
        
        # Initialize coordinator with services
        self._coordinator = CacheCoordinatorService(
            l1_cache=self._l1_cache,
            l2_cache=self._l2_cache,
            l3_cache=self._l3_cache,
            enable_warming=enable_warming,
        )
        
        # Initialize monitoring service
        self._monitoring = CacheMonitoringService(self._coordinator)
        
        # Configuration
        self._l2_default_ttl = l2_default_ttl
        self._enable_l2 = enable_l2
        self._enable_l3 = enable_l3

    async def get(
        self,
        key: str,
        fallback_func: Any = None,
        l2_ttl: Optional[int] = None,
        l1_ttl: Optional[int] = None,
    ) -> Optional[Any]:
        """Get value from multi-level cache with fallback.
        
        Args:
            key: Cache key
            fallback_func: Function to call if not in cache
            l2_ttl: TTL for L2 cache
            l1_ttl: TTL for L1 cache
            
        Returns:
            Cached or computed value
        """
        return await self._coordinator.get(
            key=key,
            fallback_func=fallback_func,
            l2_ttl=l2_ttl or self._l2_default_ttl,
            l1_ttl=l1_ttl,
        )

    async def set(
        self,
        key: str,
        value: Any,
        l2_ttl: Optional[int] = None,
        l1_ttl: Optional[int] = None,
    ) -> None:
        """Set value in multi-level cache.
        
        Args:
            key: Cache key
            value: Value to cache
            l2_ttl: TTL for L2 cache
            l1_ttl: TTL for L1 cache
        """
        await self._coordinator.set(
            key=key,
            value=value,
            l2_ttl=l2_ttl or self._l2_default_ttl,
            l1_ttl=l1_ttl,
        )

    async def delete(self, key: str) -> None:
        """Delete key from all cache levels.
        
        Args:
            key: Cache key
        """
        await self._coordinator.delete(key)

    async def clear(self) -> None:
        """Clear all cache levels."""
        await self._coordinator.clear()

    async def warm_cache(self, keys: list[str]) -> dict[str, bool]:
        """Manually warm cache with specific keys.
        
        Args:
            keys: List of cache keys to warm
            
        Returns:
            Dictionary mapping keys to warming success status
        """
        return await self._coordinator.warm_cache(keys)

    async def get_warming_candidates(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get current warming candidates based on access patterns.
        
        Args:
            limit: Maximum number of candidates to return
            
        Returns:
            List of warming candidate information
        """
        # Get access patterns from coordinator
        hot_keys = self._coordinator._get_hot_keys_for_warming()
        candidates = []
        
        for key, pattern in hot_keys[:limit]:
            candidates.append({
                "key": key,
                "access_count": pattern.access_count,
                "access_frequency": pattern.access_frequency,
                "last_access": pattern.last_access.isoformat(),
                "warming_priority": pattern.warming_priority,
                "in_l1_cache": await self._l1_cache.exists(key),
            })
        
        return candidates

    async def stop_warming(self) -> None:
        """Stop background cache warming."""
        await self._coordinator.stop_warming()

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check.
        
        Returns:
            Health check results
        """
        return await self._monitoring.health_check()

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive cache performance statistics.
        
        Returns:
            Performance statistics across all cache levels
        """
        return self._coordinator.get_performance_stats()

    def get_monitoring_metrics(self) -> dict[str, Any]:
        """Get metrics optimized for monitoring systems.
        
        Returns:
            Monitoring metrics dictionary
        """
        return self._monitoring.get_monitoring_metrics()

    def get_alert_metrics(self) -> dict[str, Any]:
        """Get metrics for alerting systems with thresholds.
        
        Returns:
            Alert metrics with threshold comparisons
        """
        return self._monitoring.get_alert_metrics()

    def calculate_slo_compliance(self) -> dict[str, Any]:
        """Calculate SLO compliance metrics.
        
        Returns:
            SLO compliance statistics
        """
        return self._monitoring.calculate_slo_compliance()

    async def close(self) -> None:
        """Close all cache connections and cleanup resources."""
        await self._coordinator.stop_warming()
        
        if self._l2_cache:
            await self._l2_cache.close()

    # Backward compatibility methods
    
    def get_health_check(self) -> dict[str, Any]:
        """Backward compatibility for get_health_check."""
        import asyncio
        return asyncio.create_task(self.health_check())

    def get_stats(self) -> dict[str, Any]:
        """Backward compatibility for get_stats."""
        return self._l1_cache.get_stats()


class SpecializedCaches:
    """Specialized cache instances for different data types.
    
    Maintains backward compatibility with the original SpecializedCaches
    while using the new decomposed cache services internally.
    """

    def __init__(self) -> None:
        self.rule_cache = CacheFacade(
            l1_max_size=500,
            l2_default_ttl=7200,
            enable_l2=True,
            enable_l3=False,
        )
        self.session_cache = CacheFacade(
            l1_max_size=1000,
            l2_default_ttl=1800,
            enable_l2=True,
            enable_l3=False,
        )
        self.analytics_cache = CacheFacade(
            l1_max_size=200,
            l2_default_ttl=3600,
            enable_l2=True,
            enable_l3=False,
        )
        self.prompt_cache = CacheFacade(
            l1_max_size=2000,
            l2_default_ttl=900,
            enable_l2=True,
            enable_l3=False,
        )

    def get_cache_for_type(self, cache_type: CacheType) -> CacheFacade:
        """Get specialized cache by canonical CacheType."""
        cache_map = {
            CacheType.RULE: self.rule_cache,
            CacheType.SESSION: self.session_cache,
            CacheType.ANALYTICS: self.analytics_cache,
            CacheType.PROMPT: self.prompt_cache,
        }
        return cache_map.get(cache_type, self.prompt_cache)

    async def get_all_stats(self) -> dict[str, Any]:
        """Get statistics for all specialized caches."""
        return {
            "rule_cache": self.rule_cache.get_performance_stats(),
            "session_cache": self.session_cache.get_performance_stats(),
            "analytics_cache": self.analytics_cache.get_performance_stats(),
            "prompt_cache": self.prompt_cache.get_performance_stats(),
        }

    async def close_all(self) -> None:
        """Close all specialized cache instances."""
        await self.rule_cache.close()
        await self.session_cache.close()
        await self.analytics_cache.close()
        await self.prompt_cache.close()


# Global instances for backward compatibility
_global_caches: SpecializedCaches | None = None


def get_specialized_caches() -> SpecializedCaches:
    """Get the global specialized cache instances."""
    global _global_caches
    if _global_caches is None:
        _global_caches = SpecializedCaches()
    return _global_caches


def get_cache(cache_type: CacheType = CacheType.PROMPT) -> CacheFacade:
    """Get a specialized cache by type."""
    caches = get_specialized_caches()
    return caches.get_cache_for_type(cache_type)


async def cached_get(
    key: str,
    fallback_func: Any,
    cache_type: CacheType = CacheType.PROMPT,
    ttl: int = 300,
) -> Any:
    """Get value with caching using specialized cache."""
    cache = get_cache(cache_type)
    return await cache.get(key, fallback_func, l2_ttl=ttl, l1_ttl=ttl // 2)


async def cached_set(
    key: str,
    value: Any,
    cache_type: CacheType = CacheType.PROMPT,
    ttl: int = 300,
) -> None:
    """Set value in specialized cache."""
    cache = get_cache(cache_type)
    await cache.set(key, value, l2_ttl=ttl, l1_ttl=ttl // 2)


# Alias for backward compatibility
MultiLevelCache = CacheFacade