"""Multi-level caching system for optimal response times.

This module implements a sophisticated caching strategy with multiple cache levels:
1. In-memory L1 cache for ultra-fast access (<1ms)
2. Redis L2 cache for shared data (1-10ms)
3. Database L3 with optimized queries (10-50ms)

Designed to achieve <200ms response times for all MCP operations.
"""

import asyncio
import contextlib
import inspect
import json
import logging
import os
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

from prompt_improver.core.config import AppConfig
from prompt_improver.core.types import CacheType


class RedisCache:
    """Direct Redis cache implementation using coredis for L2 caching.

    This implementation creates a direct Redis connection using existing
    configuration patterns and handles connection errors gracefully.
    Avoids circular imports by accessing config directly.
    """

    def __init__(self) -> None:
        self._client: coredis.Redis | None = None
        self._connection_error_logged = False
        # Track connection cycles for health signals
        self._ever_connected: bool = False
        self._last_reconnect: bool = False

    async def _get_client(self) -> Optional["coredis.Redis"]:
        """Get or create Redis client with graceful error handling."""
        if self._client is not None:
            return self._client
        try:
            import coredis

            try:
                from prompt_improver.core.config import get_config

                config = get_config()
                redis_config = config.redis
                self._client = coredis.Redis(
                    host=redis_config.host,
                    port=redis_config.port,
                    db=redis_config.database,
                    password=redis_config.password,
                    username=redis_config.username or None,
                    socket_connect_timeout=redis_config.connection_timeout,
                    socket_timeout=redis_config.socket_timeout,
                    max_connections=redis_config.max_connections,
                    decode_responses=False,
                )
                await self._client.ping()
                logger.info("Redis cache connection established successfully")
                self._ever_connected = True
                self._last_reconnect = True
                self._connection_error_logged = False
            except ImportError:
                redis_host = os.getenv("REDIS_HOST", "redis.external.service")
                redis_port = int(os.getenv("REDIS_PORT", "6379"))
                redis_db = int(os.getenv("REDIS_DB", "0"))
                redis_password = os.getenv("REDIS_PASSWORD")
                self._client = coredis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    password=redis_password,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    max_connections=20,
                    decode_responses=False,
                )
                await self._client.ping()
                logger.info(
                    "Redis cache connection established using environment variables"
                )
                self._connection_error_logged = False
                self._ever_connected = True
                self._last_reconnect = True
        except ImportError:
            if not self._connection_error_logged:
                logger.warning("coredis not available - Redis L2 cache disabled")
                self._connection_error_logged = True
            return None
        except Exception as e:
            if not self._connection_error_logged:
                logger.warning(f"Failed to connect to Redis - L2 cache disabled: {e}")
                self._connection_error_logged = True
            self._client = None
            self._last_reconnect = False
            return None
        return self._client

    async def get(self, key: str) -> bytes | None:
        """Get value from Redis cache."""
        client = await self._get_client()
        if client is None:
            return None
        try:
            return await client.get(key)
        except Exception as e:
            logger.warning(f"Redis get error for key {key}: {e}")
            return None

    async def set(self, key: str, value: bytes, expire: int | None = None) -> bool:
        """Set value in Redis cache with optional expiration."""
        client = await self._get_client()
        if client is None:
            return False
        try:
            if isinstance(expire, int) and expire > 0:
                await client.set(key, value, ex=expire)
            else:
                await client.set(key, value)
            return True
        except Exception as e:
            logger.warning(f"Redis set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        client = await self._get_client()
        if client is None:
            return False
        try:
            # coredis delete expects an iterable of keys in some versions; use list
            result = await client.delete([key])
            return (result or 0) > 0
        except Exception as e:
            logger.warning(f"Redis delete error for key {key}: {e}")
            return False

    async def close(self) -> None:
        """Close Redis connection gracefully."""
        if self._client is not None:
            try:
                # Prefer async close if available
                if hasattr(self._client, "close") and inspect.iscoroutinefunction(
                    self._client.close
                ):
                    await self._client.close()
                elif hasattr(self._client, "aclose") and inspect.iscoroutinefunction(
                    self._client.aclose
                ):
                    await self._client.aclose()
                elif hasattr(self._client, "close"):
                    # Sync close fallback
                    try:
                        self._client.close()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                elif hasattr(self._client, "connection_pool"):
                    pool = getattr(self._client, "connection_pool", None)
                    if pool is not None and hasattr(pool, "disconnect"):
                        try:
                            pool.disconnect()
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")
            finally:
                self._client = None


try:
    from prompt_improver.performance.optimization.performance_optimizer import (
        measure_cache_operation,
    )

    PERFORMANCE_OPTIMIZER_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZER_AVAILABLE = False
    from contextlib import asynccontextmanager

    class MockPerfMetrics:
        def __init__(self) -> None:
            self.metadata = {}

    @asynccontextmanager
    async def measure_cache_operation(operation_name: str):
        yield MockPerfMetrics()


try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode

    OPENTELEMETRY_AVAILABLE = True
    tracer = trace.get_tracer(__name__)
    meter = metrics.get_meter(__name__)
    cache_operations_counter = meter.create_counter(
        "cache_operations_total",
        description="Total cache operations by type, level, and status",
        unit="1",
    )
    cache_hit_ratio_gauge = meter.create_gauge(
        "cache_hit_ratio", description="Cache hit ratio by level and type", unit="ratio"
    )
    cache_latency_histogram = meter.create_histogram(
        "cache_operation_duration_seconds",
        description="Cache operation duration by type and level",
        unit="s",
    )
    cache_size_gauge = meter.create_gauge(
        "cache_size_current", description="Current cache size by level", unit="1"
    )
    cache_warming_counter = meter.create_counter(
        "cache_warming_operations_total",
        description="Cache warming operations by status",
        unit="1",
    )
    cache_memory_usage_gauge = meter.create_gauge(
        "cache_memory_usage_bytes",
        description="Estimated cache memory usage",
        unit="bytes",
    )
    cache_error_counter = meter.create_counter(
        "cache_errors_total", description="Cache errors by type and operation", unit="1"
    )
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    tracer = None
    meter = None
    cache_operations_counter = None
    cache_hit_ratio_gauge = None
    cache_latency_histogram = None
    cache_size_gauge = None
    cache_warming_counter = None
    cache_memory_usage_gauge = None
    cache_error_counter = None
logger = logging.getLogger(__name__)


def trace_cache_operation(operation_name: str | None = None):
    """Enhanced decorator to add comprehensive OpenTelemetry tracing to cache operations."""

    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not OPENTELEMETRY_AVAILABLE or not tracer:
                return await func(self, *args, **kwargs)
            op_name = operation_name or f"cache.{func.__name__}"
            with tracer.start_as_current_span(op_name) as span:
                span.set_attribute("cache.operation", func.__name__)
                if hasattr(self, "__class__"):
                    span.set_attribute("cache.class", self.__class__.__name__)
                if args and isinstance(args[0], str):
                    cache_key = args[0]
                    span.set_attribute("cache.key", cache_key)
                    span.set_attribute("cache.key_length", len(cache_key))
                    span.set_attribute("cache.key_hash", hash(cache_key) % 10000)
                if hasattr(self, "_l1_cache"):
                    span.set_attribute("cache.l1_enabled", True)
                    span.set_attribute(
                        "cache.l1_max_size", getattr(self._l1_cache, "_max_size", 0)
                    )
                    span.set_attribute(
                        "cache.l1_current_size",
                        len(getattr(self._l1_cache, "_cache", {})),
                    )
                if hasattr(self, "_enable_l2"):
                    span.set_attribute("cache.l2_enabled", self._enable_l2)
                if hasattr(self, "_enable_warming"):
                    span.set_attribute("cache.warming_enabled", self._enable_warming)
                start_time = time.perf_counter()
                cache_level_hit = None
                try:
                    result = await func(self, *args, **kwargs)
                    duration = time.perf_counter() - start_time
                    span.set_attribute("cache.duration_seconds", duration)
                    span.set_attribute("cache.duration_ms", duration * 1000)
                    span.set_status(Status(StatusCode.OK))
                    if hasattr(self, "_total_requests"):
                        span.set_attribute("cache.total_requests", self._total_requests)
                        span.set_attribute(
                            "cache.l1_hits", getattr(self, "_l1_hits", 0)
                        )
                        span.set_attribute(
                            "cache.l2_hits", getattr(self, "_l2_hits", 0)
                        )
                        span.set_attribute(
                            "cache.l3_hits", getattr(self, "_l3_hits", 0)
                        )
                    if result is not None:
                        span.set_attribute("cache.result_found", True)
                        if isinstance(result, (dict, list)):
                            span.set_attribute("cache.result_size", len(result))
                    else:
                        span.set_attribute("cache.result_found", False)
                    if cache_operations_counter:
                        cache_operations_counter.add(
                            1,
                            {
                                "operation": func.__name__,
                                "cache_class": self.__class__.__name__,
                                "status": "success",
                                "cache_level": cache_level_hit or "unknown",
                            },
                        )
                    if cache_latency_histogram:
                        cache_latency_histogram.record(
                            duration,
                            {
                                "operation": func.__name__,
                                "cache_class": self.__class__.__name__,
                                "cache_level": cache_level_hit or "unknown",
                            },
                        )
                    if cache_size_gauge and hasattr(self, "_l1_cache"):
                        cache_size_gauge.set(
                            len(getattr(self._l1_cache, "_cache", {})),
                            {"cache_class": self.__class__.__name__, "level": "l1"},
                        )
                    return result
                except Exception as e:
                    duration = time.perf_counter() - start_time
                    span.set_attribute("cache.duration_seconds", duration)
                    span.set_attribute("cache.duration_ms", duration * 1000)
                    span.set_attribute("cache.error", str(e))
                    span.set_attribute("cache.error_type", type(e).__name__)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    if cache_operations_counter:
                        cache_operations_counter.add(
                            1,
                            {
                                "operation": func.__name__,
                                "cache_class": self.__class__.__name__,
                                "status": "error",
                            },
                        )
                    if cache_error_counter:
                        cache_error_counter.add(
                            1,
                            {
                                "operation": func.__name__,
                                "cache_class": self.__class__.__name__,
                                "error_type": type(e).__name__,
                            },
                        )
                    raise

        return wrapper

    return decorator


@dataclass
class WarmingAccessPattern:
    """Access pattern tracking for cache warming (renamed to avoid enum name conflict)."""

    key: str
    access_count: int = 0
    last_access: datetime | None = None
    access_frequency: float = 0.0
    access_times: list[datetime] | None = None
    warming_priority: float = 0.0

    def __post_init__(self):
        if self.access_times is None:
            self.access_times = []
        if self.last_access is None:
            self.last_access = datetime.now(UTC)

    def record_access(self) -> None:
        """Record a new access and update frequency metrics."""
        now = datetime.now(UTC)
        self.access_count += 1
        self.last_access = now
        self.access_times.append(now)
        cutoff = now - timedelta(hours=24)
        self.access_times = [t for t in self.access_times if t > cutoff]
        if len(self.access_times) >= 2:
            time_span = (
                self.access_times[-1] - self.access_times[0]
            ).total_seconds() / 3600
            self.access_frequency = len(self.access_times) / max(time_span, 0.1)
        recency_weight = max(0.0, 1 - (now - self.last_access).total_seconds() / 3600)
        self.warming_priority = self.access_frequency * (1 + recency_weight)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: int | None = None

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

    def __init__(self, max_size: int = 1000) -> None:
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if key in self._cache:
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry.value
        self._misses += 1
        return None

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Set value in cache."""
        if key in self._cache:
            entry = self._cache[key]
            entry.value = value
            entry.created_at = datetime.now(UTC)
            entry.ttl_seconds = ttl_seconds
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            entry = CacheEntry(
                value=value,
                created_at=datetime.now(UTC),
                last_accessed=datetime.now(UTC),
                ttl_seconds=ttl_seconds,
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

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "utilization": len(self._cache) / self._max_size,
        }


class MultiLevelCache:
    """Multi-level cache system with L1 (memory) and L2 (Redis) tiers."""

    def __init__(
        self,
        l1_max_size: int = 1000,
        l2_default_ttl: int = 3600,
        enable_l2: bool = True,
        enable_warming: bool = True,
        warming_threshold: float = 2.0,
        warming_interval: int = 300,
        max_warming_keys: int = 50,
    ) -> None:
        self._l1_cache = LRUCache(l1_max_size)
        self._l2_cache = RedisCache() if enable_l2 else None
        self._l2_default_ttl = l2_default_ttl
        self._enable_l2 = enable_l2
        self._access_patterns: dict[str, WarmingAccessPattern] = {}
        self._enable_warming = enable_warming
        self._warming_threshold = warming_threshold
        self._warming_interval = warming_interval
        self._max_warming_keys = max_warming_keys
        self._last_warming_time = datetime.now(UTC)
        self._background_warming_task: asyncio.Task | None = None
        self._warming_stop_event: asyncio.Event = asyncio.Event()
        self._warming_stats = {
            "cycles_completed": 0,
            "keys_warmed": 0,
            "warming_hits": 0,
            "warming_errors": 0,
        }
        self._l1_hits = 0
        self._l2_hits = 0
        self._l3_hits = 0
        self._total_requests = 0
        self._last_warming_hits_reported = 0
        self._last_warming_errors_reported = 0
        self._operation_stats = defaultdict(
            lambda: {
                "count": 0,
                "total_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
                "error_count": 0,
            }
        )
        self._response_times = []
        self._max_response_time_samples = 1000
        self._health_status = {
            "overall_health": "healthy",
            "l1_health": "healthy",
            "l2_health": "healthy",
            "warming_health": "healthy",
            "last_health_check": datetime.now(UTC),
        }
        self._error_stats = defaultdict(int)
        self._connection_failures = 0
        self._consecutive_errors = 0
        self._last_error_time = None
        if self._enable_warming:
            self._start_background_warming()

    def _start_background_warming(self) -> None:
        """Start background cache warming task using unified infrastructure."""

    def _ensure_warming_started(self) -> None:
        """Ensure background warming is started (lazy initialization)."""
        if not self._enable_warming or self._background_warming_task is not None:
            return
        try:
            from prompt_improver.performance.monitoring.health.background_manager import (
                TaskPriority,
                get_background_task_manager,
            )

            task_manager = get_background_task_manager()
            # Use NORMAL priority (no BACKGROUND level defined) and submit once
            asyncio.create_task(
                task_manager.submit_enhanced_task(
                    task_id=f"cache_warming_{id(self)}",
                    coroutine=self._background_warming_loop(),
                    priority=TaskPriority.NORMAL,
                    tags={
                        "service": "cache",
                        "type": "warming",
                        "component": "multi_level_cache",
                    },
                )
            )
            self._background_warming_task = True
        except ImportError:
            logger.warning(
                "Background task manager not available, using direct task creation for cache warming"
            )
            self._background_warming_task = asyncio.create_task(
                self._background_warming_loop()
            )
        except Exception as e:
            logger.error(f"Failed to start background warming: {e}")
            self._background_warming_task = True

    async def _background_warming_loop(self) -> None:
        """Background loop for intelligent cache warming."""
        logger.info("Starting intelligent cache warming background task")
        try:
            max_cycles = int(os.getenv("CACHE_WARMING_MAX_CYCLES", "0"))
            cycles = 0
            while not self._warming_stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._warming_stop_event.wait(), timeout=self._warming_interval
                    )
                except TimeoutError:
                    try:
                        await self._perform_warming_cycle()
                    except Exception as e:
                        logger.error(f"Error in cache warming cycle: {e}")
                        self._warming_stats["warming_errors"] += 1
                    cycles += 1
                    if max_cycles and cycles >= max_cycles:
                        break
        except asyncio.CancelledError:
            logger.info("Cache warming background task cancelled")
            raise
        except Exception as e:
            logger.error(f"Fatal error in cache warming loop: {e}")
            raise

    async def _perform_warming_cycle(self) -> None:
        """Perform one cycle of intelligent cache warming with enhanced metrics."""
        if not self._enable_warming or not self._enable_l2:
            return
        warming_start = time.perf_counter()
        now = datetime.now(UTC)
        hot_keys = self._get_hot_keys_for_warming()
        if not hot_keys:
            self._warming_stats["cycles_completed"] += 1
            return
        logger.debug(f"Warming {len(hot_keys)} hot cache keys")
        warmed_count = 0
        warming_attempts = 0
        for key, _pattern in hot_keys[: self._max_warming_keys]:
            warming_attempts += 1
            try:
                if self._l1_cache.get(key) is not None:
                    continue
                warm_start = time.perf_counter()
                if await self._warm_key_from_l2(key):
                    warmed_count += 1
                    self._warming_stats["keys_warmed"] += 1
                    warm_duration = time.perf_counter() - warm_start
                    self._record_operation_stats("warming_success", warm_duration)
                    if OPENTELEMETRY_AVAILABLE and cache_warming_counter:
                        cache_warming_counter.add(
                            1, {"status": "success", "cache_type": "multi_level"}
                        )
                else:
                    warm_duration = time.perf_counter() - warm_start
                    self._record_operation_stats("warming_miss", warm_duration)
            except Exception as e:
                logger.warning(f"Failed to warm key {key}: {e}")
                self._warming_stats["warming_errors"] += 1
                self._handle_cache_error("warming", e)
                if OPENTELEMETRY_AVAILABLE and cache_warming_counter:
                    cache_warming_counter.add(
                        1, {"status": "error", "cache_type": "multi_level"}
                    )
        warming_duration = time.perf_counter() - warming_start
        self._warming_stats["cycles_completed"] += 1
        self._warming_stats["last_cycle_duration"] = warming_duration
        self._warming_stats["last_cycle_attempts"] = warming_attempts
        self._warming_stats["last_cycle_successes"] = warmed_count
        self._last_warming_time = now
        self._record_operation_stats("warming_cycle", warming_duration)
        warming_success_rate = warmed_count / max(warming_attempts, 1)
        if warming_success_rate < 0.1 and warming_attempts > 5:
            self._health_status["warming_health"] = "degraded"
        elif warming_success_rate >= 0.5:
            self._health_status["warming_health"] = "healthy"
        if warmed_count > 0:
            logger.debug(
                f"Cache warming cycle completed: warmed {warmed_count}/{warming_attempts} keys in {warming_duration:.2f}s"
            )

    def _get_hot_keys_for_warming(self) -> list[tuple[str, WarmingAccessPattern]]:
        """Get keys that should be warmed based on access patterns."""
        now = datetime.now(UTC)
        hot_keys = []
        for key, pattern in self._access_patterns.items():
            if (now - pattern.last_access).total_seconds() > 3600:
                continue
            if pattern.access_frequency < self._warming_threshold:
                continue
            hot_keys.append((key, pattern))
        hot_keys.sort(key=lambda x: x[1].warming_priority, reverse=True)
        return hot_keys

    async def _warm_key_from_l2(self, key: str) -> bool:
        """Warm a specific key from L2 to L1 cache."""
        if not self._l2_cache:
            return False
        try:
            l2_value = await self._l2_cache.get(key)
            if l2_value is not None:
                value = json.loads(l2_value.decode("utf-8"))
                self._l1_cache.set(key, value)
                self._warming_stats["warming_hits"] += 1
                return True
        except Exception as e:
            logger.warning(f"Failed to warm key {key} from L2: {e}")
        return False

    def _record_access_pattern(self, key: str) -> None:
        """Record access pattern for intelligent warming."""
        if not self._enable_warming:
            return
        if key not in self._access_patterns:
            self._access_patterns[key] = WarmingAccessPattern(key=key)
        self._access_patterns[key].record_access()
        if len(self._access_patterns) > 10000:
            self._cleanup_old_patterns()

    def _cleanup_old_patterns(self) -> None:
        """Clean up old access patterns to prevent memory growth."""
        now = datetime.now(UTC)
        cutoff = now - timedelta(days=1)
        old_keys = [
            key
            for key, pattern in self._access_patterns.items()
            if pattern.last_access < cutoff
        ]
        for key in old_keys:
            del self._access_patterns[key]
        logger.debug(f"Cleaned up {len(old_keys)} old access patterns")

    @trace_cache_operation("cache.get")
    async def get(
        self,
        key: str,
        fallback_func: Callable[[], object] | None = None,
        l2_ttl: int | None = None,
        l1_ttl: int | None = None,
    ) -> Any | None:
        """Get value from multi-level cache with fallback.

        Args:
            key: Cache key
            fallback_func: Async function to call if not in cache
            l2_ttl: TTL for L2 cache (Redis)
            l1_ttl: TTL for L1 cache (memory)

        Returns:
            Cached value or result from fallback function
        """
        operation_start = time.perf_counter()
        self._total_requests += 1
        self._ensure_warming_started()
        self._record_access_pattern(key)
        try:
            async with measure_cache_operation("multi_level_get") as perf_metrics:
                l1_value = self._l1_cache.get(key)
                if l1_value is not None:
                    self._l1_hits += 1
                    self._record_operation_stats(
                        "get_l1", time.perf_counter() - operation_start
                    )
                    perf_metrics.metadata["cache_level"] = "L1"
                    self._update_health_indicators("l1", True)
                    return l1_value
                if self._enable_l2 and self._l2_cache:
                    try:
                        l2_start = time.perf_counter()
                        l2_value = await self._l2_cache.get(key)
                        l2_duration = time.perf_counter() - l2_start
                        if l2_value is not None:
                            value = json.loads(l2_value.decode("utf-8"))
                            self._l1_cache.set(key, value, l1_ttl)
                            self._l2_hits += 1
                            self._record_operation_stats(
                                "get_l2", time.perf_counter() - operation_start
                            )
                            perf_metrics.metadata["cache_level"] = "L2"
                            self._update_health_indicators("l2", True, l2_duration)
                            return value
                        self._update_health_indicators("l2", True, l2_duration)
                    except Exception as e:
                        self._handle_cache_error("l2_get", e)
                        logger.warning(f"L2 cache error for key {key}: {e}")
                if fallback_func:
                    try:
                        l3_start = time.perf_counter()
                        result = fallback_func()
                        value = await result if inspect.isawaitable(result) else result
                        l3_duration = time.perf_counter() - l3_start
                        if value is not None:
                            await self.set(key, value, l2_ttl, l1_ttl)
                            self._l3_hits += 1
                            self._record_operation_stats(
                                "get_l3", time.perf_counter() - operation_start
                            )
                            perf_metrics.metadata["cache_level"] = "L3"
                            if l3_duration > 1.0:
                                self._update_health_indicators("l3", False, l3_duration)
                            else:
                                self._update_health_indicators("l3", True, l3_duration)
                            return value
                    except Exception as e:
                        self._handle_cache_error("l3_fallback", e)
                        logger.error(f"Fallback function failed for key {key}: {e}")
                        raise
                return None
        finally:
            total_duration = time.perf_counter() - operation_start
            self._record_response_time(total_duration)
            self._record_operation_stats("get_total", total_duration)

    @trace_cache_operation("cache.set")
    async def set(
        self, key: str, value: Any, l2_ttl: int | None = None, l1_ttl: int | None = None
    ) -> None:
        """Set value in multi-level cache with enhanced error tracking.

        Args:
            key: Cache key
            value: Value to cache
            l2_ttl: TTL for L2 cache (Redis)
            l1_ttl: TTL for L1 cache (memory)
        """
        operation_start = time.perf_counter()
        # Count all cache operations (not just gets)
        self._total_requests += 1
        try:
            async with measure_cache_operation("multi_level_set"):
                l1_start = time.perf_counter()
                self._l1_cache.set(key, value, l1_ttl)
                self._record_operation_stats("set_l1", time.perf_counter() - l1_start)
                self._update_health_indicators("l1", True)
                if self._enable_l2 and self._l2_cache:
                    try:
                        l2_start = time.perf_counter()
                        serialized_value = json.dumps(value, default=str).encode(
                            "utf-8"
                        )
                        ttl = l2_ttl or self._l2_default_ttl
                        success = await self._l2_cache.set(
                            key, serialized_value, expire=ttl
                        )
                        l2_duration = time.perf_counter() - l2_start
                        self._record_operation_stats("set_l2", l2_duration, success)
                        self._update_health_indicators("l2", success, l2_duration)
                        if not success:
                            logger.warning(f"L2 cache set returned False for key {key}")
                    except Exception as e:
                        l2_duration = time.perf_counter() - l2_start
                        self._record_operation_stats("set_l2", l2_duration, False)
                        self._handle_cache_error("l2_set", e)
                        # Mark L2 as degraded immediately if we previously connected but now fail
                        if (
                            hasattr(self._l2_cache, "_ever_connected")
                            and self._l2_cache._ever_connected
                        ):
                            self._health_status["l2_health"] = "degraded"
                        logger.warning(f"Failed to set L2 cache for key {key}: {e}")
        finally:
            total_duration = time.perf_counter() - operation_start
            self._record_response_time(total_duration)
            self._record_operation_stats("set_total", total_duration)

    @trace_cache_operation("cache.delete")
    async def delete(self, key: str) -> None:
        """Delete key from all cache levels with enhanced error tracking."""
        operation_start = time.perf_counter()
        try:
            async with measure_cache_operation("multi_level_delete"):
                l1_start = time.perf_counter()
                l1_success = self._l1_cache.delete(key)
                self._record_operation_stats(
                    "delete_l1", time.perf_counter() - l1_start, l1_success
                )
                self._update_health_indicators("l1", True)
                if self._enable_l2 and self._l2_cache:
                    try:
                        l2_start = time.perf_counter()
                        l2_success = await self._l2_cache.delete(key)
                        l2_duration = time.perf_counter() - l2_start
                        self._record_operation_stats(
                            "delete_l2", l2_duration, l2_success
                        )
                        self._update_health_indicators("l2", l2_success, l2_duration)
                    except Exception as e:
                        l2_duration = time.perf_counter() - l2_start
                        self._record_operation_stats("delete_l2", l2_duration, False)
                        self._handle_cache_error("l2_delete", e)
                        logger.warning(
                            f"Failed to delete from L2 cache for key {key}: {e}"
                        )
        finally:
            total_duration = time.perf_counter() - operation_start
            self._record_response_time(total_duration)
            self._record_operation_stats("delete_total", total_duration)

    @trace_cache_operation("cache.clear")
    async def clear(self) -> None:
        """Clear all cache levels with enhanced tracking."""
        operation_start = time.perf_counter()
        try:
            async with measure_cache_operation("multi_level_clear"):
                l1_start = time.perf_counter()
                self._l1_cache.clear()
                self._record_operation_stats("clear_l1", time.perf_counter() - l1_start)
                self._update_health_indicators("l1", True)
                if self._enable_l2 and self._l2_cache:
                    try:
                        l2_start = time.perf_counter()
                        logger.info(
                            "L2 cache clear requested - implement pattern-based clearing if needed"
                        )
                        l2_duration = time.perf_counter() - l2_start
                        self._record_operation_stats("clear_l2", l2_duration)
                        self._update_health_indicators("l2", True, l2_duration)
                    except Exception as e:
                        l2_duration = time.perf_counter() - l2_start
                        self._record_operation_stats("clear_l2", l2_duration, False)
                        self._handle_cache_error("l2_clear", e)
                        logger.warning(f"Failed to clear L2 cache: {e}")
                self._l1_hits = 0
                self._l2_hits = 0
                self._l3_hits = 0
                self._total_requests = 0
                self._response_times.clear()
                self._operation_stats.clear()
                self._error_stats.clear()
        finally:
            total_duration = time.perf_counter() - operation_start
            self._record_operation_stats("clear_total", total_duration)

    async def warm_cache(self, keys: list[str]) -> dict[str, bool]:
        """Manually warm specific keys in the cache.

        Args:
            keys: List of cache keys to warm

        Returns:
            Dictionary mapping keys to warming success status
        """
        results = {}
        for key in keys:
            try:
                success = await self._warm_key_from_l2(key)
                results[key] = success
            except Exception as e:
                logger.warning(f"Failed to manually warm key {key}: {e}")
                results[key] = False
        return results

    async def get_warming_candidates(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get current warming candidates based on access patterns.

        Args:
            limit: Maximum number of candidates to return

        Returns:
            List of warming candidate information
        """
        hot_keys = self._get_hot_keys_for_warming()
        candidates = []
        for key, pattern in hot_keys[:limit]:
            candidates.append({
                "key": key,
                "access_count": pattern.access_count,
                "access_frequency": pattern.access_frequency,
                "last_access": pattern.last_access.isoformat(),
                "warming_priority": pattern.warming_priority,
                "in_l1_cache": self._l1_cache.get(key) is not None,
            })
        return candidates

    async def stop_warming(self) -> None:
        """Stop background cache warming."""
        self._enable_warming = False
        self._warming_stop_event.set()
        task = (
            self._background_warming_task
            if isinstance(self._background_warming_task, asyncio.Task)
            else None
        )
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._background_warming_task = None
        logger.info("Cache warming stopped")

    def _record_operation_stats(
        self, operation: str, duration: float, success: bool = True
    ) -> None:
        """Record detailed operation statistics for performance analysis."""
        stats = self._operation_stats[operation]
        stats["count"] += 1
        stats["total_time"] += duration
        stats["min_time"] = min(stats["min_time"], duration)
        stats["max_time"] = max(stats["max_time"], duration)
        if not success:
            stats["error_count"] += 1

    def _record_response_time(self, duration: float) -> None:
        """Record response time for SLO monitoring and percentile calculation."""
        self._response_times.append(duration)
        if len(self._response_times) > self._max_response_time_samples:
            self._response_times = self._response_times[
                -self._max_response_time_samples // 2 :
            ]

    def _handle_cache_error(self, operation: str, error: Exception) -> None:
        """Handle and track cache errors for monitoring and alerting."""
        error_type = type(error).__name__
        self._error_stats[f"{operation}_{error_type}"] += 1
        self._consecutive_errors += 1
        self._last_error_time = datetime.now(UTC)
        if self._consecutive_errors > 5:
            self._health_status["overall_health"] = "degraded"
        if self._consecutive_errors > 10:
            self._health_status["overall_health"] = "unhealthy"
        logger.warning(
            f"Cache error in {operation}: {error} (consecutive errors: {self._consecutive_errors})"
        )

    def _update_health_indicators(
        self, level: str, success: bool, duration: float | None = None
    ) -> None:
        """Update health indicators based on operation results."""
        if success:
            self._consecutive_errors = 0
            if level == "l2" and duration and (duration < 0.1):
                self._health_status["l2_health"] = "healthy"
            elif level == "l1":
                self._health_status["l1_health"] = "healthy"
        elif level == "l2":
            self._connection_failures += 1
            # Degrade immediately on first observed failure; escalate to unhealthy after repeated failures
            if self._connection_failures >= 1:
                self._health_status["l2_health"] = "degraded"
            if self._connection_failures > 10:
                self._health_status["l2_health"] = "unhealthy"
        self._health_status["last_health_check"] = datetime.now(UTC)

    def _calculate_response_time_percentiles(self) -> dict[str, float]:
        """Calculate response time percentiles for SLO monitoring."""
        if not self._response_times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
        import statistics

        sorted_times = sorted(self._response_times)
        n = len(sorted_times)
        return {
            "p50": sorted_times[int(n * 0.5)] if n > 0 else 0.0,
            "p95": sorted_times[int(n * 0.95)] if n > 0 else 0.0,
            "p99": sorted_times[int(n * 0.99)] if n > 0 else 0.0,
            "mean": statistics.mean(sorted_times),
            "max": max(sorted_times),
            "count": n,
        }

    def _get_operation_statistics(self) -> dict[str, dict[str, float]]:
        """Get detailed statistics for each cache operation type."""
        stats = {}
        for operation, data in self._operation_stats.items():
            if data["count"] > 0:
                stats[operation] = {
                    "count": data["count"],
                    "avg_time": data["total_time"] / data["count"],
                    "min_time": data["min_time"]
                    if data["min_time"] != float("inf")
                    else 0.0,
                    "max_time": data["max_time"],
                    "total_time": data["total_time"],
                    "error_count": data["error_count"],
                    "success_rate": 1 - data["error_count"] / data["count"],
                }
        return stats

    def _calculate_cache_efficiency(self) -> dict[str, float]:
        """Calculate cache efficiency metrics."""
        if self._total_requests == 0:
            return {"overall": 0.0, "l1_efficiency": 0.0, "l2_efficiency": 0.0}
        l1_efficiency = self._l1_hits / self._total_requests
        l2_requests = self._total_requests - self._l1_hits
        l2_efficiency = self._l2_hits / l2_requests if l2_requests > 0 else 0.0
        overall_efficiency = (self._l1_hits + self._l2_hits) / self._total_requests
        return {
            "overall": overall_efficiency,
            "l1_efficiency": l1_efficiency,
            "l2_efficiency": l2_efficiency,
            "cache_penetration_rate": self._l3_hits / self._total_requests,
        }

    def _calculate_slo_compliance(self) -> dict[str, Any]:
        """Calculate SLO compliance metrics for cache operations."""
        if not self._response_times:
            return {"compliant": True, "slo_target_ms": 200, "compliance_rate": 1.0}
        slo_target = 0.2
        compliant_requests = sum(1 for t in self._response_times if t <= slo_target)
        compliance_rate = compliant_requests / len(self._response_times)
        return {
            "compliant": compliance_rate >= 0.95,
            "slo_target_ms": slo_target * 1000,
            "compliance_rate": compliance_rate,
            "compliant_requests": compliant_requests,
            "total_requests": len(self._response_times),
            "violations": len(self._response_times) - compliant_requests,
        }

    def _calculate_error_rate(self) -> dict[str, float]:
        """Calculate error rates for monitoring and alerting."""
        total_errors = sum(self._error_stats.values())
        error_rate = total_errors / max(self._total_requests, 1)
        return {
            "overall_error_rate": error_rate,
            "consecutive_errors": self._consecutive_errors,
            "total_errors": total_errors,
            "error_threshold_exceeded": error_rate > 0.05,
        }

    def _estimate_l1_memory_usage(self) -> int:
        """Estimate L1 cache memory usage in bytes."""
        try:
            import sys

            total_size = 0
            for key, entry in self._l1_cache._cache.items():
                total_size += sys.getsizeof(key)
                total_size += sys.getsizeof(entry.value)
                total_size += sys.getsizeof(entry)
            return total_size
        except Exception:
            return len(self._l1_cache._cache) * 1024

    def _get_health_status(self) -> dict[str, str]:
        """Get comprehensive health status for all cache components."""
        now = datetime.now(UTC)
        if self._enable_warming:
            warming_errors = self._warming_stats.get("warming_errors", 0)
            warming_cycles = self._warming_stats.get("cycles_completed", 1)
            warming_error_rate = warming_errors / max(warming_cycles, 1)
            if warming_error_rate > 0.1:
                self._health_status["warming_health"] = "degraded"
            if warming_error_rate > 0.3:
                self._health_status["warming_health"] = "unhealthy"
        # If L2 had to reconnect since last check, briefly mark it degraded to reflect instability
        try:
            if (
                self._enable_l2
                and self._l2_cache is not None
                and getattr(self._l2_cache, "_last_reconnect", False)
            ):
                self._health_status["l2_health"] = "degraded"
                self._l2_cache._last_reconnect = False
        except Exception:
            pass
        component_healths = [
            self._health_status["l1_health"],
            self._health_status["l2_health"],
            self._health_status["warming_health"],
        ]
        if "unhealthy" in component_healths:
            self._health_status["overall_health"] = "unhealthy"
        elif "degraded" in component_healths:
            self._health_status["overall_health"] = "degraded"
        else:
            self._health_status["overall_health"] = "healthy"
        return self._health_status.copy()

    def _update_opentelemetry_metrics(self, overall_hit_rate: float) -> None:
        """Update comprehensive OpenTelemetry metrics."""
        if not OPENTELEMETRY_AVAILABLE:
            return
        if cache_hit_ratio_gauge:
            cache_hit_ratio_gauge.set(
                overall_hit_rate, {"cache_type": "multi_level", "level": "overall"}
            )
            l1_hit_rate = (
                self._l1_hits / self._total_requests if self._total_requests > 0 else 0
            )
            cache_hit_ratio_gauge.set(
                l1_hit_rate, {"cache_type": "multi_level", "level": "l1"}
            )
            l2_hit_rate = (
                self._l2_hits / self._total_requests if self._total_requests > 0 else 0
            )
            cache_hit_ratio_gauge.set(
                l2_hit_rate, {"cache_type": "multi_level", "level": "l2"}
            )
            l3_hit_rate = (
                self._l3_hits / self._total_requests if self._total_requests > 0 else 0
            )
            cache_hit_ratio_gauge.set(
                l3_hit_rate, {"cache_type": "multi_level", "level": "l3"}
            )
        if cache_size_gauge:
            cache_size_gauge.set(
                len(self._l1_cache._cache), {"cache_type": "multi_level", "level": "l1"}
            )
            cache_size_gauge.set(
                len(self._access_patterns),
                {"cache_type": "multi_level", "level": "access_patterns"},
            )
        if cache_memory_usage_gauge:
            cache_memory_usage_gauge.set(
                self._estimate_l1_memory_usage(),
                {"cache_type": "multi_level", "level": "l1"},
            )
        if cache_warming_counter:
            cache_warming_counter.add(
                self._warming_stats.get("warming_hits", 0)
                - getattr(self, "_last_warming_hits_reported", 0),
                {"status": "success", "cache_type": "multi_level"},
            )
            cache_warming_counter.add(
                self._warming_stats.get("warming_errors", 0)
                - getattr(self, "_last_warming_errors_reported", 0),
                {"status": "error", "cache_type": "multi_level"},
            )
            self._last_warming_hits_reported = self._warming_stats.get(
                "warming_hits", 0
            )
            self._last_warming_errors_reported = self._warming_stats.get(
                "warming_errors", 0
            )

    async def get_health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check for monitoring systems."""
        health_check_start = time.perf_counter()
        try:
            test_key = f"health_check_{int(time.time())}"
            test_value = {"test": True, "timestamp": time.time()}
            self._l1_cache.set(test_key, test_value, ttl_seconds=60)
            l1_result = self._l1_cache.get(test_key)
            l1_healthy = l1_result is not None and l1_result["test"] is True
            l2_healthy = True
            l2_latency = 0.0
            if self._enable_l2 and self._l2_cache:
                try:
                    l2_start = time.perf_counter()
                    redis_test_value = json.dumps(test_value).encode("utf-8")
                    await self._l2_cache.set(test_key, redis_test_value, expire=60)
                    l2_result = await self._l2_cache.get(test_key)
                    l2_latency = time.perf_counter() - l2_start
                    l2_healthy = l2_result is not None
                    if l2_healthy:
                        retrieved_value = json.loads(l2_result.decode("utf-8"))
                        l2_healthy = retrieved_value.get("test") is True
                except Exception as e:
                    l2_healthy = False
                    logger.warning(f"L2 health check failed: {e}")
            self._l1_cache.delete(test_key)
            if self._enable_l2 and self._l2_cache:
                with contextlib.suppress(Exception):
                    await self._l2_cache.delete(test_key)
            health_check_duration = time.perf_counter() - health_check_start
            overall_healthy = l1_healthy and (l2_healthy or not self._enable_l2)
            return {
                "healthy": overall_healthy,
                "checks": {
                    "l1_cache": {
                        "healthy": l1_healthy,
                        "status": "healthy" if l1_healthy else "unhealthy",
                    },
                    "l2_cache": {
                        "healthy": l2_healthy,
                        "enabled": self._enable_l2,
                        "latency_seconds": l2_latency,
                        "status": "healthy" if l2_healthy else "unhealthy",
                    },
                    "warming_service": {
                        "healthy": self._health_status["warming_health"] == "healthy",
                        "enabled": self._enable_warming,
                        "status": self._health_status["warming_health"],
                    },
                },
                "performance": {
                    "health_check_duration_seconds": health_check_duration,
                    "l2_latency_seconds": l2_latency,
                    "overall_hit_rate": (self._l1_hits + self._l2_hits)
                    / max(self._total_requests, 1),
                    "error_rate": sum(self._error_stats.values())
                    / max(self._total_requests, 1),
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def get_monitoring_metrics(self) -> dict[str, Any]:
        """Get metrics optimized for monitoring systems and dashboards."""
        stats = self.get_performance_stats()
        return {
            "cache.hit_rate.overall": stats["overall_hit_rate"],
            "cache.hit_rate.l1": stats["l1_cache"]["hit_rate"],
            "cache.hit_rate.l2": stats["l2_cache"]["hit_rate"],
            "cache.hit_rate.l3": stats["l3_fallback"]["hit_rate"],
            "cache.size.l1_current": stats["l1_cache"]["size"],
            "cache.size.l1_max": stats["l1_cache"]["max_size"],
            "cache.memory.l1_bytes": stats["l1_cache"]["estimated_memory_usage_bytes"],
            "cache.requests.total": stats["total_requests"],
            "cache.requests.l1_hits": stats["l1_cache"]["hits"],
            "cache.requests.l2_hits": stats["l2_cache"]["hits"],
            "cache.requests.l3_hits": stats["l3_fallback"]["hits"],
            "cache.performance.response_time_p50": stats["performance_metrics"][
                "response_times"
            ]["p50"],
            "cache.performance.response_time_p95": stats["performance_metrics"][
                "response_times"
            ]["p95"],
            "cache.performance.response_time_p99": stats["performance_metrics"][
                "response_times"
            ]["p99"],
            "cache.performance.error_rate": stats["performance_metrics"]["error_rate"][
                "overall_error_rate"
            ],
            "cache.performance.slo_compliance": stats["performance_metrics"][
                "slo_compliance"
            ]["compliance_rate"],
            "cache.warming.enabled": stats["intelligent_warming"]["enabled"],
            "cache.warming.keys_warmed": stats["intelligent_warming"]["warming_stats"][
                "keys_warmed"
            ],
            "cache.warming.warming_errors": stats["intelligent_warming"][
                "warming_stats"
            ]["warming_errors"],
            "cache.warming.hit_rate": stats["intelligent_warming"]["warming_hit_rate"],
            "cache.health.overall": stats["health_monitoring"]["overall_health"],
            "cache.health.l1": stats["health_monitoring"]["l1_health"],
            "cache.health.l2": stats["health_monitoring"]["l2_health"],
            "cache.health.warming": stats["health_monitoring"]["warming_health"],
        }

    def get_alert_metrics(self) -> dict[str, Any]:
        """Get metrics specifically for alerting systems with thresholds."""
        stats = self.get_performance_stats()
        THRESHOLDS = {
            "hit_rate_critical": 0.5,
            "hit_rate_warning": 0.7,
            "error_rate_critical": 0.1,
            "error_rate_warning": 0.05,
            "response_time_critical": 1.0,
            "response_time_warning": 0.5,
            "slo_compliance_critical": 0.9,
            "memory_usage_warning": 0.8,
            "consecutive_errors_critical": 10,
            "connection_failures_critical": 5,
        }
        overall_hit_rate = stats["overall_hit_rate"]
        error_rate = stats["performance_metrics"]["error_rate"]["overall_error_rate"]
        response_time_p95 = stats["performance_metrics"]["response_times"]["p95"]
        slo_compliance = stats["performance_metrics"]["slo_compliance"][
            "compliance_rate"
        ]
        memory_utilization = stats["l1_cache"]["utilization"]
        consecutive_errors = stats["performance_metrics"]["error_rate"][
            "consecutive_errors"
        ]
        connection_failures = stats["l2_cache"].get("connection_failures", 0)
        return {
            "alerts": {
                "hit_rate_critical": overall_hit_rate < THRESHOLDS["hit_rate_critical"],
                "hit_rate_warning": overall_hit_rate < THRESHOLDS["hit_rate_warning"],
                "error_rate_critical": error_rate > THRESHOLDS["error_rate_critical"],
                "error_rate_warning": error_rate > THRESHOLDS["error_rate_warning"],
                "response_time_critical": response_time_p95
                > THRESHOLDS["response_time_critical"],
                "response_time_warning": response_time_p95
                > THRESHOLDS["response_time_warning"],
                "slo_compliance_critical": slo_compliance
                < THRESHOLDS["slo_compliance_critical"],
                "memory_usage_warning": memory_utilization
                > THRESHOLDS["memory_usage_warning"],
                "consecutive_errors_critical": consecutive_errors
                > THRESHOLDS["consecutive_errors_critical"],
                "connection_failures_critical": connection_failures
                > THRESHOLDS["connection_failures_critical"],
                "cache_unhealthy": stats["health_monitoring"]["overall_health"]
                == "unhealthy",
                "cache_degraded": stats["health_monitoring"]["overall_health"]
                == "degraded",
            },
            "values": {
                "hit_rate": overall_hit_rate,
                "error_rate": error_rate,
                "response_time_p95": response_time_p95,
                "slo_compliance": slo_compliance,
                "memory_utilization": memory_utilization,
                "consecutive_errors": consecutive_errors,
                "connection_failures": connection_failures,
                "health_status": stats["health_monitoring"]["overall_health"],
            },
            "thresholds": THRESHOLDS,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive cache performance statistics with enhanced observability."""
        l1_stats = self._l1_cache.get_stats()
        total_hits = self._l1_hits + self._l2_hits + self._l3_hits
        overall_hit_rate = (
            total_hits / self._total_requests if self._total_requests > 0 else 0
        )
        warming_hit_rate = self._warming_stats["warming_hits"] / max(
            self._warming_stats["keys_warmed"], 1
        )
        response_time_stats = self._calculate_response_time_percentiles()
        operation_stats = self._get_operation_statistics()
        self._update_opentelemetry_metrics(overall_hit_rate)
        health_status = self._get_health_status()
        # Ensure canonical l1 hits from MultiLevelCache counters are not overridden by LRU internal stats
        l1_combined = {
            **l1_stats,
            "hits": self._l1_hits,
            "hit_rate": self._l1_hits / self._total_requests
            if self._total_requests > 0
            else 0,
            "estimated_memory_usage_bytes": self._estimate_l1_memory_usage(),
        }
        return {
            "total_requests": self._total_requests,
            "overall_hit_rate": overall_hit_rate,
            "cache_efficiency": self._calculate_cache_efficiency(),
            "l1_cache": l1_combined,
            "l2_cache": {
                "hits": self._l2_hits,
                "hit_rate": self._l2_hits / self._total_requests
                if self._total_requests > 0
                else 0,
                "enabled": self._enable_l2,
                "connection_failures": self._connection_failures,
                "health_status": health_status["l2_health"],
            },
            "l3_fallback": {
                "hits": self._l3_hits,
                "hit_rate": self._l3_hits / self._total_requests
                if self._total_requests > 0
                else 0,
                "health_status": health_status.get("l3_health", "unknown"),
            },
            "intelligent_warming": {
                "enabled": self._enable_warming,
                "warming_threshold": self._warming_threshold,
                "warming_interval_seconds": self._warming_interval,
                "max_warming_keys": self._max_warming_keys,
                "tracked_patterns": len(self._access_patterns),
                "last_warming_time": self._last_warming_time.isoformat()
                if self._last_warming_time
                else None,
                "warming_stats": self._warming_stats.copy(),
                "warming_hit_rate": warming_hit_rate,
                "health_status": health_status["warming_health"],
            },
            "performance_metrics": {
                "response_times": response_time_stats,
                "operation_stats": operation_stats,
                "slo_compliance": self._calculate_slo_compliance(),
                "error_rate": self._calculate_error_rate(),
            },
            "health_monitoring": health_status,
            "error_statistics": dict(self._error_stats),
            "observability": {
                "opentelemetry_enabled": OPENTELEMETRY_AVAILABLE,
                "performance_optimizer_enabled": PERFORMANCE_OPTIMIZER_AVAILABLE,
                "last_health_check": self._health_status[
                    "last_health_check"
                ].isoformat(),
            },
        }


class SpecializedCaches:
    """Specialized cache instances for different data types."""

    def __init__(self) -> None:
        self.rule_cache = MultiLevelCache(
            l1_max_size=500, l2_default_ttl=7200, enable_l2=True
        )
        self.session_cache = MultiLevelCache(
            l1_max_size=1000, l2_default_ttl=1800, enable_l2=True
        )
        self.analytics_cache = MultiLevelCache(
            l1_max_size=200, l2_default_ttl=3600, enable_l2=True
        )
        self.prompt_cache = MultiLevelCache(
            l1_max_size=2000, l2_default_ttl=900, enable_l2=True
        )

    def get_cache_for_type(self, cache_type: CacheType) -> MultiLevelCache:
        """Get specialized cache by canonical CacheType (clean-break enums only)."""
        cache_map = {
            CacheType.RULE: self.rule_cache,
            CacheType.SESSION: self.session_cache,
            CacheType.ANALYTICS: self.analytics_cache,
            CacheType.PROMPT: self.prompt_cache,
        }
        return cache_map.get(cache_type, self.prompt_cache)

    def get_all_stats(self) -> dict[str, Any]:
        """Get statistics for all specialized caches."""
        return {
            "rule_cache": self.rule_cache.get_performance_stats(),
            "session_cache": self.session_cache.get_performance_stats(),
            "analytics_cache": self.analytics_cache.get_performance_stats(),
            "prompt_cache": self.prompt_cache.get_performance_stats(),
        }


_global_caches: SpecializedCaches | None = None


def get_specialized_caches() -> SpecializedCaches:
    """Get the global specialized cache instances."""
    global _global_caches
    if _global_caches is None:
        _global_caches = SpecializedCaches()
    return _global_caches


def get_cache(cache_type: CacheType = CacheType.PROMPT) -> MultiLevelCache:
    """Get a specialized cache by type (clean-break: enums only)."""
    caches = get_specialized_caches()
    return caches.get_cache_for_type(cache_type)


async def cached_get(
    key: str,
    fallback_func: callable,
    cache_type: CacheType = CacheType.PROMPT,
    ttl: int = 300,
) -> Any:
    """Get value with caching using specialized cache (clean-break: enums only)."""
    cache = get_cache(cache_type)
    return await cache.get(key, fallback_func, l2_ttl=ttl, l1_ttl=ttl // 2)


async def cached_set(
    key: str, value: Any, cache_type: CacheType = CacheType.PROMPT, ttl: int = 300
) -> None:
    """Set value in specialized cache (clean-break: enums only)."""
    cache = get_cache(cache_type)
    await cache.set(key, value, l2_ttl=ttl, l1_ttl=ttl // 2)
