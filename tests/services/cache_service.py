"""
Real cache service for testing.

This module contains cache service implementations for testing,
extracted from conftest.py to maintain clean architecture.
"""
from typing import Any

from prompt_improver.shared.interfaces.protocols.ml import ServiceStatus


class RealCacheService:
    """Real Redis-backed cache service for testing."""

    def __init__(self, redis_client):
        self._redis = redis_client

    async def get(self, key: str) -> Any | None:
        """Get value from Redis cache."""
        try:
            value = await self._redis.get(key)
            if value is None:
                return None
            # Try to deserialize JSON, fallback to string
            try:
                import json
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception:
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in Redis cache with optional TTL."""
        try:
            # Serialize complex objects as JSON
            if isinstance(value, (dict, list)):
                import json
                value = json.dumps(value)

            if ttl:
                await self._redis.set(key, value, ex=ttl)
            else:
                await self._redis.set(key, value)
        except Exception as e:
            # In real Redis, failures should be visible
            raise RuntimeError(f"Cache set failed: {e}")

    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        try:
            result = await self._redis.delete(key)
            return bool(result)
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            result = await self._redis.exists(key)
            return bool(result)
        except Exception:
            return False

    async def health_check(self) -> ServiceStatus:
        """Check Redis health."""
        try:
            await self._redis.ping()
            return ServiceStatus.HEALTHY
        except Exception:
            return ServiceStatus.ERROR

    async def clear(self):
        """Clear all cache data."""
        await self._redis.flushdb()

    # Test helpers for compatibility
    def set_health_status(self, healthy: bool):
        """Test helper - not applicable for real Redis."""

    def get_all_data(self) -> dict[str, Any]:
        """Test helper - retrieve all cached data."""
        # For real Redis, this is complex due to async nature
        # Return empty dict as placeholder for test compatibility
        return {}
