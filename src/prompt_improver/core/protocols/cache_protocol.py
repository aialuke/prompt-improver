"""Protocol definitions for cache operations.

Provides type-safe interface contracts for caching systems,
enabling dependency inversion and improved testability.
"""
from typing import Any, Dict, List, Optional, Protocol

class BasicCacheProtocol(Protocol):
    """Protocol for basic cache operations"""

    async def get(self, key: str) -> Any | None:
        """Get value from cache"""
        ...

    async def set(self, key: str, value: Any, ttl: int | None=None) -> bool:
        """Set value in cache with optional TTL"""
        ...

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        ...

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        ...

    async def clear(self) -> bool:
        """Clear all cache entries"""
        ...

class AdvancedCacheProtocol(Protocol):
    """Protocol for advanced cache operations"""

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from cache"""
        ...

    async def set_many(self, mapping: dict[str, Any], ttl: int | None=None) -> bool:
        """Set multiple values in cache"""
        ...

    async def delete_many(self, keys: list[str]) -> int:
        """Delete multiple keys from cache"""
        ...

    async def get_or_set(self, key: str, default_func, ttl: int | None=None) -> Any:
        """Get value or set it using default function"""
        ...

    async def increment(self, key: str, delta: int=1) -> int:
        """Increment numeric value in cache"""
        ...

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for key"""
        ...

class CacheHealthProtocol(Protocol):
    """Protocol for cache health monitoring"""

    async def ping(self) -> bool:
        """Ping cache service"""
        ...

    async def get_info(self) -> dict[str, Any]:
        """Get cache service information"""
        ...

    async def get_stats(self) -> dict[str, Any]:
        """Get cache performance statistics"""
        ...

    async def get_memory_usage(self) -> dict[str, Any]:
        """Get memory usage statistics"""
        ...

class CacheSubscriptionProtocol(Protocol):
    """Protocol for cache pub/sub operations"""

    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel"""
        ...

    async def subscribe(self, channels: list[str]) -> Any:
        """Subscribe to channels"""
        ...

    async def unsubscribe(self, channels: list[str]) -> bool:
        """Unsubscribe from channels"""
        ...

class CacheLockProtocol(Protocol):
    """Protocol for distributed locking"""

    async def acquire_lock(self, key: str, timeout: int=10) -> str | None:
        """Acquire distributed lock"""
        ...

    async def release_lock(self, key: str, token: str) -> bool:
        """Release distributed lock"""
        ...

    async def extend_lock(self, key: str, token: str, timeout: int) -> bool:
        """Extend lock timeout"""
        ...

class RedisCacheProtocol(BasicCacheProtocol, AdvancedCacheProtocol, CacheHealthProtocol, CacheSubscriptionProtocol, CacheLockProtocol):
    """Combined protocol for Redis cache operations"""

class MultiLevelCacheProtocol(Protocol):
    """Protocol for multi-level cache systems"""

    async def get_from_level(self, key: str, level: int) -> Any | None:
        """Get value from specific cache level"""
        ...

    async def set_to_level(self, key: str, value: Any, level: int, ttl: int | None=None) -> bool:
        """Set value to specific cache level"""
        ...

    async def invalidate_levels(self, key: str, levels: list[int]) -> bool:
        """Invalidate key from specific levels"""
        ...

    async def get_cache_hierarchy(self) -> list[str]:
        """Get cache level hierarchy"""
        ...
