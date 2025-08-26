"""Cache service protocol definitions.

Consolidated high-performance cache protocols supporting multi-level
caching (L1/L2/L3), distributed operations, and performance optimization.

Performance targets:
- L1 Memory Cache: <1ms response time
- L2 Redis Cache: 1-10ms response time
- Overall system: <2ms achieved on critical paths
- Cache hit rate: 96.67% achieved

Consolidated from:
- /core/protocols/cache_protocol.py
- /core/protocols/cache_service/cache_protocols.py
- /services/cache/protocols.py
"""

from collections.abc import Callable
from enum import Enum, StrEnum
from typing import Any, Protocol, runtime_checkable


class CacheType(StrEnum):
    """Canonical cache domain types used across APES.

    Clean-break consolidation replacing scattered cache type enums and strings.
    Defined here to avoid circular imports with core.types.
    """

    LINGUISTIC = "linguistic"
    DOMAIN = "domain"
    SESSION = "session"
    PROMPT = "prompt"
    RULE = "rule"
    ANALYTICS = "analytics"
    ML_ANALYSIS = "ml_analysis"
    PATTERN_DISCOVERY = "pattern_discovery"
    ALL = "all"


class CacheLevel(Enum):
    """Cache level enumeration for multi-level cache architecture."""
    L1_MEMORY = "L1_MEMORY"
    L2_REDIS = "L2_REDIS"
    L3_DATABASE = "L3_DATABASE"


# =============================================================================
# Base Cache Protocols
# =============================================================================

@runtime_checkable
class BasicCacheProtocol(Protocol):
    """Protocol for fundamental cache operations with high performance."""

    async def get(self, key: str, namespace: str | None = None) -> Any | None:
        """Get value from cache with optional namespace isolation.

        Args:
            key: Cache key
            namespace: Optional namespace for multi-tenant isolation

        Returns:
            Cached value or None if not found
        """
        ...

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        namespace: str | None = None
    ) -> bool:
        """Set value in cache with optional TTL and namespace.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            namespace: Optional namespace

        Returns:
            True if successful, False otherwise
        """
        ...

    async def delete(self, key: str, namespace: str | None = None) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key
            namespace: Optional namespace

        Returns:
            True if key was deleted, False if not found
        """
        ...

    async def exists(self, key: str, namespace: str | None = None) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key
            namespace: Optional namespace

        Returns:
            True if key exists, False otherwise
        """
        ...

    async def clear(self, namespace: str | None = None) -> int:
        """Clear cache entries.

        Args:
            namespace: Optional namespace to clear (all if None)

        Returns:
            Number of entries cleared
        """
        ...


@runtime_checkable
class AdvancedCacheProtocol(Protocol):
    """Protocol for advanced cache operations including batch processing."""

    async def get_many(
        self,
        keys: list[str],
        namespace: str | None = None
    ) -> dict[str, Any]:
        """Get multiple values from cache efficiently.

        Args:
            keys: List of cache keys
            namespace: Optional namespace

        Returns:
            Dictionary of key-value pairs found
        """
        ...

    async def set_many(
        self,
        items: dict[str, Any],
        ttl: int | None = None,
        namespace: str | None = None
    ) -> bool:
        """Set multiple values in cache efficiently.

        Args:
            items: Dictionary of key-value pairs
            ttl: Time to live in seconds
            namespace: Optional namespace

        Returns:
            True if successful, False otherwise
        """
        ...

    async def delete_many(
        self,
        keys: list[str],
        namespace: str | None = None
    ) -> int:
        """Delete multiple keys from cache.

        Args:
            keys: List of cache keys
            namespace: Optional namespace

        Returns:
            Number of keys successfully deleted
        """
        ...

    async def delete_pattern(
        self,
        pattern: str,
        namespace: str | None = None
    ) -> int:
        """Delete keys matching pattern (supports wildcards).

        Args:
            pattern: Pattern to match against cache keys
            namespace: Optional namespace

        Returns:
            Number of keys deleted
        """
        ...

    async def get_or_set(
        self,
        key: str,
        default_func: Callable[[], Any],
        ttl: int | None = None,
        namespace: str | None = None
    ) -> Any:
        """Get value or set it using default function if not found.

        Args:
            key: Cache key
            default_func: Function to generate value if key not found
            ttl: Time to live in seconds
            namespace: Optional namespace

        Returns:
            Cached or newly generated value
        """
        ...

    async def increment(
        self,
        key: str,
        delta: int = 1,
        namespace: str | None = None
    ) -> int:
        """Increment numeric value in cache.

        Args:
            key: Cache key
            delta: Increment amount
            namespace: Optional namespace

        Returns:
            New value after increment
        """
        ...

    async def expire(
        self,
        key: str,
        seconds: int,
        namespace: str | None = None
    ) -> bool:
        """Set expiration for existing key.

        Args:
            key: Cache key
            seconds: Expiration time in seconds
            namespace: Optional namespace

        Returns:
            True if expiration set, False if key not found
        """
        ...


# =============================================================================
# Level-Specific Cache Protocols
# =============================================================================

@runtime_checkable
class L1CacheServiceProtocol(BasicCacheProtocol, Protocol):
    """Protocol for L1 in-memory cache service (<1ms response time)."""

    async def get_stats(self) -> dict[str, Any]:
        """Get L1 cache statistics for performance monitoring.

        Returns:
            Statistics including hit rate, size, evictions, memory usage
        """
        ...

    async def warm_up(
        self,
        keys: list[str],
        source: Any | None = None
    ) -> int:
        """Warm up cache with frequently accessed keys.

        Args:
            keys: Keys to pre-load
            source: Optional data source for loading

        Returns:
            Number of keys successfully loaded
        """
        ...


@runtime_checkable
class L2CacheServiceProtocol(BasicCacheProtocol, AdvancedCacheProtocol, Protocol):
    """Protocol for L2 Redis cache service (1-10ms response time)."""

    async def mget(
        self,
        keys: list[str],
        namespace: str | None = None
    ) -> dict[str, Any]:
        """Redis-optimized multi-get operation.

        Args:
            keys: List of cache keys
            namespace: Optional namespace

        Returns:
            Dictionary of key-value pairs
        """
        ...

    async def mset(
        self,
        items: dict[str, Any],
        ttl: int | None = None,
        namespace: str | None = None
    ) -> bool:
        """Redis-optimized multi-set operation.

        Args:
            items: Dictionary of key-value pairs
            ttl: Time to live in seconds
            namespace: Optional namespace

        Returns:
            Success status
        """
        ...

    async def get_connection_status(self) -> dict[str, Any]:
        """Get Redis connection health and performance metrics.

        Returns:
            Connection status, latency, and health information
        """
        ...


# =============================================================================
# Advanced Feature Protocols
# =============================================================================

@runtime_checkable
class CacheHealthProtocol(Protocol):
    """Protocol for comprehensive cache health monitoring and metrics."""

    async def ping(self) -> bool:
        """Ping cache service for basic connectivity.

        Returns:
            True if cache is responsive, False otherwise
        """
        ...

    async def get_info(self) -> dict[str, Any]:
        """Get detailed cache service information.

        Returns:
            Comprehensive cache service information
        """
        ...

    async def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache performance statistics.

        Returns:
            Performance metrics including hit rates, response times, memory usage
        """
        ...

    async def get_memory_usage(self) -> dict[str, Any]:
        """Get detailed memory usage statistics.

        Returns:
            Memory usage breakdown and allocation information
        """
        ...

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check across all cache levels.

        Returns:
            Health status for each cache level and component
        """
        ...

    def get_monitoring_metrics(self) -> dict[str, Any]:
        """Get metrics optimized for monitoring systems.

        Returns:
            Monitoring-specific metrics dictionary
        """
        ...

    def get_alert_metrics(self) -> dict[str, Any]:
        """Get metrics for alerting systems with threshold comparisons.

        Returns:
            Alert metrics with threshold status
        """
        ...

    def calculate_slo_compliance(self) -> dict[str, Any]:
        """Calculate SLO compliance metrics for cache performance.

        Returns:
            SLO compliance statistics and error budget information
        """
        ...


@runtime_checkable
class CacheSubscriptionProtocol(Protocol):
    """Protocol for cache pub/sub operations (Redis-specific)."""

    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel.

        Args:
            channel: Channel name
            message: Message to publish

        Returns:
            Number of subscribers that received the message
        """
        ...

    async def subscribe(self, channels: list[str]) -> Any:
        """Subscribe to channels for cache invalidation events.

        Args:
            channels: List of channel names

        Returns:
            Subscription handle
        """
        ...

    async def unsubscribe(self, channels: list[str]) -> bool:
        """Unsubscribe from channels.

        Args:
            channels: List of channel names

        Returns:
            True if successful, False otherwise
        """
        ...


@runtime_checkable
class CacheLockProtocol(Protocol):
    """Protocol for distributed locking operations."""

    async def acquire_lock(
        self,
        key: str,
        timeout: int = 10,
        namespace: str | None = None
    ) -> str | None:
        """Acquire distributed lock with timeout.

        Args:
            key: Lock key
            timeout: Lock timeout in seconds
            namespace: Optional namespace

        Returns:
            Lock token if acquired, None if failed
        """
        ...

    async def release_lock(
        self,
        key: str,
        token: str,
        namespace: str | None = None
    ) -> bool:
        """Release distributed lock using token.

        Args:
            key: Lock key
            token: Lock token from acquire_lock
            namespace: Optional namespace

        Returns:
            True if released, False if token invalid
        """
        ...

    async def extend_lock(
        self,
        key: str,
        token: str,
        timeout: int,
        namespace: str | None = None
    ) -> bool:
        """Extend lock timeout.

        Args:
            key: Lock key
            token: Lock token
            timeout: New timeout in seconds
            namespace: Optional namespace

        Returns:
            True if extended, False if token invalid
        """
        ...


@runtime_checkable
class CacheWarmingProtocol(Protocol):
    """Protocol for intelligent cache warming operations."""

    async def start_warming(self) -> None:
        """Start background cache warming process."""
        ...

    async def stop_warming(self) -> None:
        """Stop background cache warming process."""
        ...

    async def perform_warming_cycle(self) -> None:
        """Perform one cycle of intelligent cache warming."""
        ...

    def record_access_pattern(self, key: str) -> None:
        """Record access pattern for warming intelligence.

        Args:
            key: Cache key that was accessed
        """
        ...

    def get_warming_candidates(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get current cache warming candidates based on access patterns.

        Args:
            limit: Maximum number of candidates

        Returns:
            List of warming candidate information with priority scores
        """
        ...


# =============================================================================
# Multi-Level Cache Coordination Protocols
# =============================================================================

@runtime_checkable
class MultiLevelCacheProtocol(Protocol):
    """Protocol for multi-level cache systems with intelligent coordination."""

    async def get_from_level(
        self,
        key: str,
        level: int,
        namespace: str | None = None
    ) -> Any | None:
        """Get value from specific cache level.

        Args:
            key: Cache key
            level: Cache level (1=L1, 2=L2, 3=L3)
            namespace: Optional namespace

        Returns:
            Value from specified level or None
        """
        ...

    async def set_to_level(
        self,
        key: str,
        value: Any,
        level: int,
        ttl: int | None = None,
        namespace: str | None = None
    ) -> bool:
        """Set value to specific cache level.

        Args:
            key: Cache key
            value: Value to cache
            level: Cache level (1=L1, 2=L2, 3=L3)
            ttl: Time to live in seconds
            namespace: Optional namespace

        Returns:
            True if successful, False otherwise
        """
        ...

    async def invalidate_levels(
        self,
        key: str,
        levels: list[int],
        namespace: str | None = None
    ) -> bool:
        """Invalidate key from specific cache levels.

        Args:
            key: Cache key
            levels: List of cache levels to invalidate
            namespace: Optional namespace

        Returns:
            True if invalidated from all requested levels
        """
        ...

    async def get_cache_hierarchy(self) -> list[str]:
        """Get cache level hierarchy information.

        Returns:
            Ordered list of cache level descriptions
        """
        ...


@runtime_checkable
class CacheServiceFacadeProtocol(Protocol):
    """Protocol for unified cache service facade with strategy support."""

    async def get(
        self,
        key: str,
        strategy: str = "cascade",
        namespace: str | None = None
    ) -> Any | None:
        """Get value using specified retrieval strategy.

        Args:
            key: Cache key
            strategy: Retrieval strategy ('cascade', 'l1_only', 'l2_only')
            namespace: Optional namespace

        Returns:
            Cached value or None if not found
        """
        ...

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        strategy: str = "write_through",
        namespace: str | None = None
    ) -> bool:
        """Set value using specified write strategy.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            strategy: Write strategy ('write_through', 'write_back', 'write_around')
            namespace: Optional namespace

        Returns:
            True if successful, False otherwise
        """
        ...

    async def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics for all cache levels.

        Returns:
            Unified statistics including per-level metrics and overall performance
        """
        ...

    async def configure(self, config: dict[str, Any]) -> bool:
        """Configure cache behavior and performance parameters.

        Args:
            config: Configuration parameters for cache optimization

        Returns:
            True if configuration applied successfully
        """
        ...


# =============================================================================
# Combined Protocols for Specific Cache Types
# =============================================================================

@runtime_checkable
class RedisCacheProtocol(
    BasicCacheProtocol,
    AdvancedCacheProtocol,
    CacheHealthProtocol,
    CacheSubscriptionProtocol,
    CacheLockProtocol,
    Protocol,
):
    """Combined protocol for comprehensive Redis cache operations."""


@runtime_checkable
class CacheServiceProtocol(Protocol):
    """Protocol for individual cache service operations following clean architecture principles."""

    async def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        ...

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
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
class ComprehensiveCacheProtocol(
    BasicCacheProtocol,
    AdvancedCacheProtocol,
    CacheHealthProtocol,
    CacheWarmingProtocol,
    MultiLevelCacheProtocol,
    Protocol,
):
    """Combined protocol for full-featured cache implementations."""
