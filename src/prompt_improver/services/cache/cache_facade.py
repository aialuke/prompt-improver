"""Cache Facade for unified multi-level cache access.

Provides a clean, unified interface to the decomposed cache services while
maintaining backward compatibility and optimal performance. Replaces the
original god object with focused, testable services.
"""

import logging
from typing import Any, Optional

from .protocols import CacheType

from .cache_coordinator_service import CacheCoordinatorService
from .cache_monitoring_service import CacheMonitoringService
from .l1_cache_service import L1CacheService
from .l2_redis_service import L2RedisService
from .l3_database_service import L3DatabaseService

# Import secure cache wrapper for session encryption
try:
    from .secure_cache_wrapper import SecureCacheWrapper
    SECURE_CACHING_AVAILABLE = True
except ImportError:
    SECURE_CACHING_AVAILABLE = False
    SecureCacheWrapper = None

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
        session_encryption_key: Optional[str] = None,
    ) -> None:
        """Initialize cache facade with decomposed services.
        
        Args:
            l1_max_size: Maximum size for L1 cache
            l2_default_ttl: Default TTL for L2 cache
            enable_l2: Enable L2 (Redis) cache
            enable_l3: Enable L3 (Database) cache
            enable_warming: Enable intelligent cache warming
            session_manager: Database session manager for L3 cache
            session_encryption_key: Encryption key for session data (auto-generated if None)
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
        
        # Initialize secure session wrapper if available
        if SECURE_CACHING_AVAILABLE and SecureCacheWrapper:
            try:
                import os
                encryption_key = session_encryption_key or os.getenv("SESSION_ENCRYPTION_KEY")
                self._secure_wrapper = SecureCacheWrapper(self._coordinator, encryption_key)
                logger.info("Secure session encryption enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize secure session wrapper: {e}")
                self._secure_wrapper = None
        else:
            logger.warning("Secure session encryption not available - cryptography library required")
            self._secure_wrapper = None

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

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern across all levels.
        
        Args:
            pattern: Pattern to match (supports wildcards like 'prefix:*')
            
        Returns:
            Number of cache entries invalidated
        """
        return await self._coordinator.invalidate_pattern(pattern)

    async def get_or_set(
        self,
        key: str,
        value_func: Any,
        l2_ttl: Optional[int] = None,
        l1_ttl: Optional[int] = None,
    ) -> Any:
        """Get value from cache or compute and set if not found.
        
        Args:
            key: Cache key
            value_func: Function to compute value if not cached
            l2_ttl: TTL for L2 cache
            l1_ttl: TTL for L1 cache
            
        Returns:
            Cached or computed value
        """
        # Try to get from cache first
        cached_value = await self.get(key)
        if cached_value is not None:
            return cached_value
            
        # Compute value and cache it
        if callable(value_func):
            import asyncio
            if asyncio.iscoroutinefunction(value_func):
                computed_value = await value_func()
            else:
                computed_value = value_func()
        else:
            computed_value = value_func
            
        await self.set(key, computed_value, l2_ttl, l1_ttl)
        return computed_value

    # Session management convenience methods
    
    async def set_session(self, session_id: str, data: Any, ttl: int = 3600) -> bool:
        """Set session data with secure encryption and session namespace.
        
        Args:
            session_id: Session identifier
            data: Session data to store
            ttl: Time to live in seconds (default: 1 hour)
            
        Returns:
            True if successful
        """
        try:
            # SECURITY: Use encrypted storage for session data
            if self._secure_wrapper:
                return await self._secure_wrapper.set_encrypted(
                    f"session:{session_id}", 
                    data, 
                    ttl,
                    metadata={"session_id": session_id, "type": "session"}
                )
            else:
                # Fallback to unencrypted (with warning)
                logger.warning(f"Storing session {session_id} without encryption - security risk")
                await self.set(f"session:{session_id}", data, l2_ttl=ttl, l1_ttl=ttl // 4)
                return True
        except Exception as e:
            logger.error(f"Failed to set session {session_id}: {e}")
            return False

    async def get_session(self, session_id: str) -> Optional[Any]:
        """Get session data by session ID with secure decryption.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found
        """
        try:
            # SECURITY: Use encrypted retrieval for session data
            if self._secure_wrapper:
                return await self._secure_wrapper.get_encrypted(f"session:{session_id}")
            else:
                # Fallback to unencrypted (with warning)
                logger.warning(f"Retrieving session {session_id} without decryption")
                return await self.get(f"session:{session_id}")
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None

    async def delete_session(self, session_id: str) -> bool:
        """Delete session data by session ID with secure cleanup.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        try:
            # SECURITY: Use encrypted deletion for session data
            if self._secure_wrapper:
                return await self._secure_wrapper.delete_encrypted(f"session:{session_id}")
            else:
                # Fallback to unencrypted deletion
                await self.delete(f"session:{session_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    async def touch_session(self, session_id: str, ttl: int = 3600) -> bool:
        """Extend session TTL by re-setting the data.
        
        Args:
            session_id: Session identifier
            ttl: New TTL in seconds
            
        Returns:
            True if successful, False if session not found
        """
        try:
            session_data = await self.get_session(session_id)
            if session_data is not None:
                return await self.set_session(session_id, session_data, ttl)
            return False
        except Exception as e:
            logger.error(f"Failed to touch session {session_id}: {e}")
            return False

    async def clear_sessions(self, pattern: str = "session:*") -> int:
        """Clear session data matching pattern.
        
        Args:
            pattern: Session pattern to clear (default: all sessions)
            
        Returns:
            Number of sessions cleared
        """
        return await self.invalidate_pattern(pattern)

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