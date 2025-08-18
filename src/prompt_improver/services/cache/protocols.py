"""Protocols for cache services following clean architecture principles.

These protocols define the contracts for cache services, enabling dependency
injection and testability while maintaining strict separation of concerns.
"""

from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable


class CacheType(Enum):
    """Cache type enumeration for specialized cache instances.
    
    Local definition to avoid circular imports with core.types.
    """
    RULE = "rule"
    SESSION = "session" 
    ANALYTICS = "analytics"
    PROMPT = "prompt"
    APPLICATION = "application"


@runtime_checkable
class CacheServiceProtocol(Protocol):
    """Protocol for individual cache service operations."""

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        ...

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        ...

    async def delete(self, key: str) -> bool:
        """Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted, False if not found
        """
        ...

    async def clear(self) -> None:
        """Clear all cache entries."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        ...

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern.
        
        Args:
            pattern: Pattern to match against cache keys (supports wildcards)
            
        Returns:
            Number of entries invalidated
        """
        ...

    def get_stats(self) -> dict[str, Any]:
        """Get cache performance statistics.
        
        Returns:
            Dictionary with statistics like hit_rate, size, etc.
        """
        ...


@runtime_checkable
class CacheCoordinatorProtocol(Protocol):
    """Protocol for multi-level cache coordination."""

    async def get(
        self,
        key: str,
        fallback_func: Optional[Callable[[], Any]] = None,
        l2_ttl: Optional[int] = None,
        l1_ttl: Optional[int] = None,
    ) -> Optional[Any]:
        """Get value from multi-level cache with fallback.
        
        Args:
            key: Cache key
            fallback_func: Function to call if not in any cache level
            l2_ttl: TTL for L2 cache
            l1_ttl: TTL for L1 cache
            
        Returns:
            Cached or computed value
        """
        ...

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
        ...

    async def delete(self, key: str) -> None:
        """Delete key from all cache levels.
        
        Args:
            key: Cache key
        """
        ...

    async def warm_cache(self, keys: list[str]) -> dict[str, bool]:
        """Warm cache with specific keys.
        
        Args:
            keys: List of cache keys to warm
            
        Returns:
            Dictionary mapping keys to warming success status
        """
        ...

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive cache performance statistics.
        
        Returns:
            Performance statistics across all cache levels
        """
        ...


@runtime_checkable
class CacheMonitoringProtocol(Protocol):
    """Protocol for cache monitoring and health checks."""

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check.
        
        Returns:
            Health check results with status for each cache level
        """
        ...

    def get_monitoring_metrics(self) -> dict[str, Any]:
        """Get metrics optimized for monitoring systems.
        
        Returns:
            Monitoring metrics dictionary
        """
        ...

    def get_alert_metrics(self) -> dict[str, Any]:
        """Get metrics for alerting systems with thresholds.
        
        Returns:
            Alert metrics with threshold comparisons
        """
        ...

    def calculate_slo_compliance(self) -> dict[str, Any]:
        """Calculate SLO compliance metrics.
        
        Returns:
            SLO compliance statistics
        """
        ...

    def get_health_status(self) -> dict[str, str]:
        """Get current health status for all components.
        
        Returns:
            Health status dictionary with component statuses
        """
        ...


@runtime_checkable
class CacheWarmingProtocol(Protocol):
    """Protocol for intelligent cache warming operations."""

    async def start_warming(self) -> None:
        """Start background cache warming."""
        ...

    async def stop_warming(self) -> None:
        """Stop background cache warming."""
        ...

    async def perform_warming_cycle(self) -> None:
        """Perform one cycle of cache warming."""
        ...

    def record_access_pattern(self, key: str) -> None:
        """Record access pattern for warming intelligence.
        
        Args:
            key: Cache key that was accessed
        """
        ...

    def get_warming_candidates(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get current warming candidates.
        
        Args:
            limit: Maximum number of candidates
            
        Returns:
            List of warming candidate information
        """
        ...