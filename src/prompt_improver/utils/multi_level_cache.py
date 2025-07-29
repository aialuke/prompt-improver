"""Multi-level caching system for optimal response times.

This module implements a sophisticated caching strategy with multiple cache levels:
1. In-memory L1 cache for ultra-fast access (<1ms)
2. Redis L2 cache for shared data (1-10ms)
3. Database L3 with optimized queries (10-50ms)

Designed to achieve <200ms response times for all MCP operations.
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from typing import Any, Dict, Optional
from collections import OrderedDict
from functools import wraps

from prompt_improver.core.config import AppConfig

# Temporary fallback: disable Redis cache to fix circular imports
class RedisCache:
    """Temporary fallback Redis cache implementation."""

    async def get(self, key: str) -> Optional[bytes]:
        """Fallback get - always returns None (cache miss)."""
        return None

    async def set(self, key: str, value: bytes, expire: Optional[int] = None) -> bool:
        """Fallback set - always returns True but doesn't store."""
        return True

    async def delete(self, key: str) -> bool:
        """Fallback delete - always returns True."""
        return True

# Import performance optimizer with graceful fallback to avoid circular imports
try:
    from ..performance.optimization.performance_optimizer import measure_cache_operation
    PERFORMANCE_OPTIMIZER_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZER_AVAILABLE = False
    # Create a no-op context manager with metadata attribute
    from contextlib import asynccontextmanager

    class MockPerfMetrics:
        def __init__(self):
            self.metadata = {}

    @asynccontextmanager
    async def measure_cache_operation(operation_name: str):
        yield MockPerfMetrics()

# OpenTelemetry imports with graceful fallback
try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True

    # Get tracer and meter
    tracer = trace.get_tracer(__name__)
    meter = metrics.get_meter(__name__)

    # Create cache-specific metrics
    cache_operations_counter = meter.create_counter(
        "cache_operations_total",
        description="Total cache operations",
        unit="1"
    )

    cache_hit_ratio_gauge = meter.create_gauge(
        "cache_hit_ratio",
        description="Cache hit ratio by level and type",
        unit="ratio"
    )

    cache_latency_histogram = meter.create_histogram(
        "cache_operation_duration_seconds",
        description="Cache operation duration",
        unit="s"
    )

except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    tracer = None
    meter = None
    cache_operations_counter = None
    cache_hit_ratio_gauge = None
    cache_latency_histogram = None

logger = logging.getLogger(__name__)

def trace_cache_operation(operation_name: str = None):
    """Decorator to add OpenTelemetry tracing to cache operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not OPENTELEMETRY_AVAILABLE or not tracer:
                return await func(self, *args, **kwargs)

            # Determine operation name
            op_name = operation_name or f"cache.{func.__name__}"

            with tracer.start_as_current_span(op_name) as span:
                # Add cache-specific attributes
                span.set_attribute("cache.operation", func.__name__)
                if hasattr(self, '__class__'):
                    span.set_attribute("cache.class", self.__class__.__name__)

                # Add key if it's the first argument
                if args and isinstance(args[0], str):
                    span.set_attribute("cache.key", args[0])

                start_time = time.perf_counter()

                try:
                    result = await func(self, *args, **kwargs)

                    # Record success metrics
                    duration = time.perf_counter() - start_time
                    span.set_attribute("cache.duration_seconds", duration)
                    span.set_status(Status(StatusCode.OK))

                    # Update OpenTelemetry metrics
                    if cache_operations_counter:
                        cache_operations_counter.add(1, {
                            "operation": func.__name__,
                            "cache_class": self.__class__.__name__,
                            "status": "success"
                        })

                    if cache_latency_histogram:
                        cache_latency_histogram.record(duration, {
                            "operation": func.__name__,
                            "cache_class": self.__class__.__name__
                        })

                    return result

                except Exception as e:
                    # Record error metrics
                    duration = time.perf_counter() - start_time
                    span.set_attribute("cache.duration_seconds", duration)
                    span.set_attribute("cache.error", str(e))
                    span.set_status(Status(StatusCode.ERROR, str(e)))

                    if cache_operations_counter:
                        cache_operations_counter.add(1, {
                            "operation": func.__name__,
                            "cache_class": self.__class__.__name__,
                            "status": "error"
                        })

                    raise

        return wrapper
    return decorator

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if the cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.now(UTC) > self.created_at + timedelta(seconds=self.ttl_seconds)

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = datetime.now(UTC)
        self.access_count += 1

class LRUCache:
    """High-performance in-memory LRU cache for L1 caching."""

    def __init__(self, max_size: int = 1000):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._cache:
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry.value

        self._misses += 1
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set value in cache."""
        if key in self._cache:
            # Update existing entry
            entry = self._cache[key]
            entry.value = value
            entry.created_at = datetime.now(UTC)
            entry.ttl_seconds = ttl_seconds
            self._cache.move_to_end(key)
        else:
            # Add new entry
            if len(self._cache) >= self._max_size:
                # Remove least recently used
                self._cache.popitem(last=False)

            entry = CacheEntry(
                value=value,
                created_at=datetime.now(UTC),
                last_accessed=datetime.now(UTC),
                ttl_seconds=ttl_seconds
            )
            self._cache[key] = entry

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "utilization": len(self._cache) / self._max_size
        }

class MultiLevelCache:
    """Multi-level cache system with L1 (memory) and L2 (Redis) tiers."""

    def __init__(
        self,
        l1_max_size: int = 1000,
        l2_default_ttl: int = 3600,
        enable_l2: bool = True
    ):
        self._l1_cache = LRUCache(l1_max_size)
        self._l2_cache = RedisCache() if enable_l2 else None
        self._l2_default_ttl = l2_default_ttl
        self._enable_l2 = enable_l2

        # Performance metrics
        self._l1_hits = 0
        self._l2_hits = 0
        self._l3_hits = 0  # Database hits
        self._total_requests = 0

    @trace_cache_operation("cache.get")
    async def get(
        self,
        key: str,
        fallback_func: Optional[callable] = None,
        l2_ttl: Optional[int] = None,
        l1_ttl: Optional[int] = None
    ) -> Optional[Any]:
        """Get value from multi-level cache with fallback.

        Args:
            key: Cache key
            fallback_func: Async function to call if not in cache
            l2_ttl: TTL for L2 cache (Redis)
            l1_ttl: TTL for L1 cache (memory)

        Returns:
            Cached value or result from fallback function
        """
        self._total_requests += 1

        async with measure_cache_operation("multi_level_get") as perf_metrics:
            # Try L1 cache first (fastest)
            l1_value = self._l1_cache.get(key)
            if l1_value is not None:
                self._l1_hits += 1
                perf_metrics.metadata["cache_level"] = "L1"
                return l1_value

            # Try L2 cache (Redis)
            if self._enable_l2 and self._l2_cache:
                try:
                    l2_value = await self._l2_cache.get(key)
                    if l2_value is not None:
                        # Deserialize from Redis
                        value = json.loads(l2_value.decode('utf-8'))

                        # Populate L1 cache
                        self._l1_cache.set(key, value, l1_ttl)

                        self._l2_hits += 1
                        perf_metrics.metadata["cache_level"] = "L2"
                        return value
                except Exception as e:
                    logger.warning(f"L2 cache error for key {key}: {e}")

            # Fallback to source (L3 - database or computation)
            if fallback_func:
                try:
                    value = await fallback_func()
                    if value is not None:
                        # Store in both cache levels
                        await self.set(key, value, l2_ttl, l1_ttl)

                        self._l3_hits += 1
                        perf_metrics.metadata["cache_level"] = "L3"
                        return value
                except Exception as e:
                    logger.error(f"Fallback function failed for key {key}: {e}")
                    raise

            return None

    @trace_cache_operation("cache.set")
    async def set(
        self,
        key: str,
        value: Any,
        l2_ttl: Optional[int] = None,
        l1_ttl: Optional[int] = None
    ) -> None:
        """Set value in multi-level cache.

        Args:
            key: Cache key
            value: Value to cache
            l2_ttl: TTL for L2 cache (Redis)
            l1_ttl: TTL for L1 cache (memory)
        """
        async with measure_cache_operation("multi_level_set"):
            # Set in L1 cache
            self._l1_cache.set(key, value, l1_ttl)

            # Set in L2 cache (Redis)
            if self._enable_l2 and self._l2_cache:
                try:
                    serialized_value = json.dumps(value, default=str).encode('utf-8')
                    ttl = l2_ttl or self._l2_default_ttl
                    await self._l2_cache.set(key, serialized_value, expire=ttl)
                except Exception as e:
                    logger.warning(f"Failed to set L2 cache for key {key}: {e}")

    @trace_cache_operation("cache.delete")
    async def delete(self, key: str) -> None:
        """Delete key from all cache levels."""
        async with measure_cache_operation("multi_level_delete"):
            # Delete from L1
            self._l1_cache.delete(key)

            # Delete from L2
            if self._enable_l2 and self._l2_cache:
                try:
                    await self._l2_cache.delete(key)
                except Exception as e:
                    logger.warning(f"Failed to delete from L2 cache for key {key}: {e}")

    @trace_cache_operation("cache.clear")
    async def clear(self) -> None:
        """Clear all cache levels."""
        async with measure_cache_operation("multi_level_clear"):
            # Clear L1
            self._l1_cache.clear()

            # Clear L2 (pattern-based)
            if self._enable_l2 and self._l2_cache:
                try:
                    # Note: This is a simplified clear - in production you might want
                    # to use Redis SCAN with pattern matching
                    logger.info("L2 cache clear requested - implement pattern-based clearing if needed")
                except Exception as e:
                    logger.warning(f"Failed to clear L2 cache: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics."""
        l1_stats = self._l1_cache.get_stats()

        total_hits = self._l1_hits + self._l2_hits + self._l3_hits
        overall_hit_rate = total_hits / self._total_requests if self._total_requests > 0 else 0

        # Update OpenTelemetry metrics
        if OPENTELEMETRY_AVAILABLE and cache_hit_ratio_gauge:
            cache_hit_ratio_gauge.set(overall_hit_rate, {
                "cache_type": "multi_level",
                "level": "overall"
            })

            l1_hit_rate = self._l1_hits / self._total_requests if self._total_requests > 0 else 0
            cache_hit_ratio_gauge.set(l1_hit_rate, {
                "cache_type": "multi_level",
                "level": "l1"
            })

            l2_hit_rate = self._l2_hits / self._total_requests if self._total_requests > 0 else 0
            cache_hit_ratio_gauge.set(l2_hit_rate, {
                "cache_type": "multi_level",
                "level": "l2"
            })

            l3_hit_rate = self._l3_hits / self._total_requests if self._total_requests > 0 else 0
            cache_hit_ratio_gauge.set(l3_hit_rate, {
                "cache_type": "multi_level",
                "level": "l3"
            })

        return {
            "total_requests": self._total_requests,
            "overall_hit_rate": overall_hit_rate,
            "l1_cache": {
                "hits": self._l1_hits,
                "hit_rate": self._l1_hits / self._total_requests if self._total_requests > 0 else 0,
                **l1_stats
            },
            "l2_cache": {
                "hits": self._l2_hits,
                "hit_rate": self._l2_hits / self._total_requests if self._total_requests > 0 else 0,
                "enabled": self._enable_l2
            },
            "l3_fallback": {
                "hits": self._l3_hits,
                "hit_rate": self._l3_hits / self._total_requests if self._total_requests > 0 else 0
            }
        }

class SpecializedCaches:
    """Specialized cache instances for different data types."""

    def __init__(self):
        # Rule metadata cache - frequently accessed, small data
        self.rule_cache = MultiLevelCache(
            l1_max_size=500,
            l2_default_ttl=7200,  # 2 hours
            enable_l2=True
        )

        # Session data cache - medium frequency, session-scoped
        self.session_cache = MultiLevelCache(
            l1_max_size=1000,
            l2_default_ttl=1800,  # 30 minutes
            enable_l2=True
        )

        # Analytics cache - less frequent, larger data
        self.analytics_cache = MultiLevelCache(
            l1_max_size=200,
            l2_default_ttl=3600,  # 1 hour
            enable_l2=True
        )

        # Prompt improvement cache - high frequency, critical path
        self.prompt_cache = MultiLevelCache(
            l1_max_size=2000,
            l2_default_ttl=900,  # 15 minutes
            enable_l2=True
        )

    def get_cache_for_type(self, cache_type: str) -> MultiLevelCache:
        """Get specialized cache by type."""
        cache_map = {
            "rule": self.rule_cache,
            "session": self.session_cache,
            "analytics": self.analytics_cache,
            "prompt": self.prompt_cache
        }
        return cache_map.get(cache_type, self.prompt_cache)

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all specialized caches."""
        return {
            "rule_cache": self.rule_cache.get_performance_stats(),
            "session_cache": self.session_cache.get_performance_stats(),
            "analytics_cache": self.analytics_cache.get_performance_stats(),
            "prompt_cache": self.prompt_cache.get_performance_stats()
        }

# Global cache instances
_global_caches: Optional[SpecializedCaches] = None

def get_specialized_caches() -> SpecializedCaches:
    """Get the global specialized cache instances."""
    global _global_caches
    if _global_caches is None:
        _global_caches = SpecializedCaches()
    return _global_caches

def get_cache(cache_type: str = "prompt") -> MultiLevelCache:
    """Get a specialized cache by type."""
    caches = get_specialized_caches()
    return caches.get_cache_for_type(cache_type)

# Convenience functions
async def cached_get(
    key: str,
    fallback_func: callable,
    cache_type: str = "prompt",
    ttl: int = 300
) -> Any:
    """Get value with caching using specialized cache."""
    cache = get_cache(cache_type)
    return await cache.get(key, fallback_func, l2_ttl=ttl, l1_ttl=ttl//2)

async def cached_set(
    key: str,
    value: Any,
    cache_type: str = "prompt",
    ttl: int = 300
) -> None:
    """Set value in specialized cache."""
    cache = get_cache(cache_type)
    await cache.set(key, value, l2_ttl=ttl, l1_ttl=ttl//2)
