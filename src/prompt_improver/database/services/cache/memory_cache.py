"""L1 In-memory cache implementation with LRU eviction and access pattern tracking.

This module provides high-performance in-memory caching extracted from
unified_connection_manager.py, implementing:

- MemoryCache: Thread-safe LRU cache with TTL support and advanced features
- CacheEntry: Rich cache entry with access metadata and expiration
- AccessPattern: Intelligent access pattern tracking for cache warming
- Performance monitoring and metrics integration
- Multiple eviction policies (LRU, LFU, TTL-based)

Designed for sub-millisecond access times with comprehensive observability.
"""

import asyncio
import logging
import threading
import time
import warnings
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, Union

try:
    from opentelemetry import metrics, trace
    from opentelemetry.metrics import Counter, Gauge, Histogram

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    warnings.warn(
        "OpenTelemetry not available. Install with: pip install opentelemetry-api"
    )
    OPENTELEMETRY_AVAILABLE = False
    # Mock classes for type hints
    Counter = Any
    Gauge = Any
    Histogram = Any

logger = logging.getLogger(__name__)


class EvictionPolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live based
    FIFO = "fifo"  # First In First Out


class CacheEventType(Enum):
    """Cache event types for monitoring."""

    HIT = "hit"
    MISS = "miss"
    EVICTION = "eviction"
    EXPIRATION = "expiration"
    SET = "set"
    DELETE = "delete"
    CLEAR = "clear"


@dataclass
class CacheEntry:
    """Cache entry with comprehensive metadata for L1 memory cache.

    Enhanced from unified_connection_manager.py with additional tracking
    for performance optimization and observability.
    """

    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    tags: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Calculate entry size if not provided."""
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size()

    def is_expired(self) -> bool:
        """Check if the cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.now(UTC) > self.created_at + timedelta(seconds=self.ttl_seconds)

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = datetime.now(UTC)
        self.access_count += 1

    def time_until_expiry(self) -> Optional[float]:
        """Get seconds until expiry, or None if no TTL."""
        if self.ttl_seconds is None:
            return None
        expires_at = self.created_at + timedelta(seconds=self.ttl_seconds)
        remaining = (expires_at - datetime.now(UTC)).total_seconds()
        return max(0, remaining)

    def _calculate_size(self) -> int:
        """Estimate memory size of the cached value."""
        try:
            import sys

            return sys.getsizeof(self.value)
        except Exception:
            # Fallback estimation
            if isinstance(self.value, str):
                return len(self.value.encode("utf-8"))
            elif isinstance(self.value, (list, tuple)):
                return sum(sys.getsizeof(item) for item in self.value)
            elif isinstance(self.value, dict):
                return sum(
                    sys.getsizeof(k) + sys.getsizeof(v) for k, v in self.value.items()
                )
            else:
                return 64  # Default estimate


@dataclass
class AccessPattern:
    """Access pattern tracking for intelligent cache warming.

    Enhanced from unified_connection_manager.py with additional metrics
    and prediction capabilities.
    """

    key: str
    access_count: int = 0
    last_access: Optional[datetime] = None
    access_frequency: float = 0.0
    access_times: List[datetime] = field(default_factory=list)
    warming_priority: float = 0.0
    access_intervals: deque = field(default_factory=lambda: deque(maxlen=50))
    predicted_next_access: Optional[datetime] = None

    def record_access(self) -> None:
        """Record a new access and update frequency metrics."""
        now = datetime.now(UTC)

        # Update basic metrics
        if self.last_access:
            interval = (now - self.last_access).total_seconds()
            self.access_intervals.append(interval)

        self.access_count += 1
        self.last_access = now
        self.access_times.append(now)

        # Keep only recent access times (24 hours)
        cutoff = now - timedelta(hours=24)
        self.access_times = [t for t in self.access_times if t > cutoff]

        # Calculate access frequency
        if len(self.access_times) >= 2:
            time_span = (
                self.access_times[-1] - self.access_times[0]
            ).total_seconds() / 3600  # Convert to hours
            self.access_frequency = len(self.access_times) / max(time_span, 0.1)

        # Calculate warming priority with recency weighting
        if self.last_access:
            recency_weight = max(0, 1 - (now - self.last_access).total_seconds() / 3600)
        else:
            recency_weight = 1.0  # First access gets full recency weight
        self.warming_priority = self.access_frequency * (1 + recency_weight)

        # Predict next access time based on intervals
        self._predict_next_access()

    def _predict_next_access(self) -> None:
        """Predict when this key will likely be accessed next."""
        if len(self.access_intervals) < 3:
            self.predicted_next_access = None
            return

        # Simple moving average of intervals
        avg_interval = sum(self.access_intervals) / len(self.access_intervals)
        self.predicted_next_access = self.last_access + timedelta(seconds=avg_interval)

    def should_warm(self, threshold: float = 0.5) -> bool:
        """Determine if this pattern suggests cache warming is beneficial."""
        return self.warming_priority > threshold and self.access_frequency > 0.1


class CacheMetrics:
    """Comprehensive cache metrics tracking."""

    def __init__(self, service_name: str = "memory_cache"):
        self.service_name = service_name
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0
        self.sets = 0
        self.deletes = 0
        self.clears = 0

        # Performance metrics
        self.response_times = deque(maxlen=1000)
        self.size_history = deque(maxlen=100)

        # OpenTelemetry setup
        self.operations_counter: Optional[Counter] = None
        self.hit_ratio_gauge: Optional[Gauge] = None
        self.response_time_histogram: Optional[Histogram] = None

        if OPENTELEMETRY_AVAILABLE:
            self._setup_telemetry()

    def _setup_telemetry(self) -> None:
        """Setup OpenTelemetry metrics."""
        try:
            meter = metrics.get_meter(f"prompt_improver.cache.{self.service_name}")

            self.operations_counter = meter.create_counter(
                "cache_operations_total",
                description="Total cache operations by type and result",
                unit="1",
            )

            self.hit_ratio_gauge = meter.create_gauge(
                "cache_hit_ratio", description="Cache hit ratio", unit="1"
            )

            self.response_time_histogram = meter.create_histogram(
                "cache_response_time_seconds",
                description="Cache operation response times",
                unit="s",
            )

            logger.debug(f"OpenTelemetry metrics initialized for {self.service_name}")
        except Exception as e:
            logger.warning(f"Failed to setup OpenTelemetry metrics: {e}")

    def record_operation(
        self, event_type: CacheEventType, duration_ms: float = 0
    ) -> None:
        """Record cache operation metrics."""
        # Update counters
        if event_type == CacheEventType.HIT:
            self.hits += 1
        elif event_type == CacheEventType.MISS:
            self.misses += 1
        elif event_type == CacheEventType.EVICTION:
            self.evictions += 1
        elif event_type == CacheEventType.EXPIRATION:
            self.expirations += 1
        elif event_type == CacheEventType.SET:
            self.sets += 1
        elif event_type == CacheEventType.DELETE:
            self.deletes += 1
        elif event_type == CacheEventType.CLEAR:
            self.clears += 1

        # Record response time
        if duration_ms > 0:
            self.response_times.append(duration_ms)

        # Record OpenTelemetry metrics
        if self.operations_counter:
            self.operations_counter.add(1, {"event_type": event_type.value})

        if self.response_time_histogram and duration_ms > 0:
            self.response_time_histogram.record(duration_ms / 1000.0)

        # Update hit ratio gauge
        if self.hit_ratio_gauge:
            total = self.hits + self.misses
            hit_ratio = self.hits / total if total > 0 else 0
            self.hit_ratio_gauge.set(hit_ratio)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_operations = self.hits + self.misses
        hit_rate = self.hits / total_operations if total_operations > 0 else 0

        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times
            else 0
        )

        return {
            "service": self.service_name,
            "operations": {
                "hits": self.hits,
                "misses": self.misses,
                "sets": self.sets,
                "deletes": self.deletes,
                "evictions": self.evictions,
                "expirations": self.expirations,
                "clears": self.clears,
                "total": total_operations,
            },
            "performance": {
                "hit_rate": hit_rate,
                "avg_response_time_ms": avg_response_time,
                "p95_response_time_ms": self._percentile(95)
                if self.response_times
                else 0,
                "p99_response_time_ms": self._percentile(99)
                if self.response_times
                else 0,
            },
            "memory": {
                "current_size": self.size_history[-1] if self.size_history else 0,
                "avg_size": sum(self.size_history) / len(self.size_history)
                if self.size_history
                else 0,
                "max_size": max(self.size_history) if self.size_history else 0,
            },
        }

    def _percentile(self, p: int) -> float:
        """Calculate percentile of response times."""
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        index = int((p / 100.0) * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)]


class MemoryCache:
    """High-performance thread-safe in-memory LRU cache for L1 caching.

    Enhanced from unified_connection_manager.py with:
    - Thread safety for concurrent access
    - Multiple eviction policies
    - Comprehensive metrics and monitoring
    - Access pattern tracking
    - Memory usage tracking
    - Tag-based operations
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_bytes: Optional[int] = None,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        default_ttl: Optional[int] = None,
        enable_metrics: bool = True,
        service_name: str = "memory_cache",
    ):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._max_memory_bytes = max_memory_bytes or (
            max_size * 1024
        )  # 1KB per entry default
        self._eviction_policy = eviction_policy
        self._default_ttl = default_ttl
        self._lock = threading.RLock()  # Reentrant lock for thread safety

        # Access patterns for warming
        self._access_patterns: Dict[str, AccessPattern] = {}

        # Current memory usage
        self._current_memory_bytes = 0

        # Metrics
        self.metrics = CacheMetrics(service_name) if enable_metrics else None

        # Background cleanup
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_background_cleanup()

        logger.info(
            f"MemoryCache initialized: max_size={max_size}, "
            f"max_memory={max_memory_bytes}, policy={eviction_policy.value}"
        )

    def get(self, key: str, record_pattern: bool = True) -> Any:
        """Get value from cache with thread safety."""
        start_time = time.time()

        with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                # Check expiration
                if entry.is_expired():
                    del self._cache[key]
                    self._current_memory_bytes -= entry.size_bytes
                    if key in self._access_patterns:
                        del self._access_patterns[key]

                    duration_ms = (time.time() - start_time) * 1000
                    if self.metrics:
                        self.metrics.record_operation(
                            CacheEventType.EXPIRATION, duration_ms
                        )
                    return None

                # Update access metadata
                if self._eviction_policy == EvictionPolicy.LRU:
                    self._cache.move_to_end(key)

                entry.touch()

                # Record access pattern
                if record_pattern:
                    if key not in self._access_patterns:
                        self._access_patterns[key] = AccessPattern(key=key)
                    self._access_patterns[key].record_access()

                duration_ms = (time.time() - start_time) * 1000
                if self.metrics:
                    self.metrics.record_operation(CacheEventType.HIT, duration_ms)

                return entry.value

            # Cache miss
            duration_ms = (time.time() - start_time) * 1000
            if self.metrics:
                self.metrics.record_operation(CacheEventType.MISS, duration_ms)

            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[Set[str]] = None,
    ) -> bool:
        """Set value in cache with thread safety."""
        start_time = time.time()
        ttl = ttl_seconds or self._default_ttl

        with self._lock:
            # Create new entry
            entry = CacheEntry(
                value=value,
                created_at=datetime.now(UTC),
                last_accessed=datetime.now(UTC),
                ttl_seconds=ttl,
                tags=tags or set(),
            )

            # Update existing entry
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_memory_bytes -= old_entry.size_bytes

                entry.value = value
                entry.created_at = datetime.now(UTC)
                entry.ttl_seconds = ttl
                entry.tags = tags or set()

                self._cache[key] = entry
                self._current_memory_bytes += entry.size_bytes

                if self._eviction_policy == EvictionPolicy.LRU:
                    self._cache.move_to_end(key)
            else:
                # Check capacity before adding
                if not self._make_space_for_entry(entry):
                    return False

                self._cache[key] = entry
                self._current_memory_bytes += entry.size_bytes

            duration_ms = (time.time() - start_time) * 1000
            if self.metrics:
                self.metrics.record_operation(CacheEventType.SET, duration_ms)

            return True

    def delete(self, key: str) -> bool:
        """Delete key from cache with thread safety."""
        start_time = time.time()

        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                del self._cache[key]
                self._current_memory_bytes -= entry.size_bytes

                # Remove access pattern
                if key in self._access_patterns:
                    del self._access_patterns[key]

                duration_ms = (time.time() - start_time) * 1000
                if self.metrics:
                    self.metrics.record_operation(CacheEventType.DELETE, duration_ms)

                return True

            return False

    def clear(self) -> None:
        """Clear all cache entries with thread safety."""
        start_time = time.time()

        with self._lock:
            self._cache.clear()
            self._access_patterns.clear()
            self._current_memory_bytes = 0

            duration_ms = (time.time() - start_time) * 1000
            if self.metrics:
                self.metrics.record_operation(CacheEventType.CLEAR, duration_ms)

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            if key not in self._cache:
                return False

            entry = self._cache[key]
            if entry.is_expired():
                # Clean up expired entry
                del self._cache[key]
                self._current_memory_bytes -= entry.size_bytes
                if key in self._access_patterns:
                    del self._access_patterns[key]
                return False

            return True

    def get_by_tags(self, tags: Set[str]) -> Dict[str, Any]:
        """Get all entries that have any of the specified tags."""
        result = {}

        with self._lock:
            for key, entry in self._cache.items():
                if not entry.is_expired() and entry.tags & tags:
                    result[key] = entry.value
                    entry.touch()

        return result

    def delete_by_tags(self, tags: Set[str]) -> int:
        """Delete all entries that have any of the specified tags."""
        deleted_count = 0
        keys_to_delete = []

        with self._lock:
            for key, entry in self._cache.items():
                if entry.tags & tags:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                if self.delete(key):
                    deleted_count += 1

        return deleted_count

    def get_access_patterns(self) -> Dict[str, AccessPattern]:
        """Get current access patterns (copy for thread safety)."""
        with self._lock:
            return dict(self._access_patterns)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            base_stats = {
                "size": len(self._cache),
                "max_size": self._max_size,
                "memory_bytes": self._current_memory_bytes,
                "max_memory_bytes": self._max_memory_bytes,
                "utilization": len(self._cache) / self._max_size,
                "memory_utilization": self._current_memory_bytes
                / self._max_memory_bytes,
                "eviction_policy": self._eviction_policy.value,
                "access_patterns": len(self._access_patterns),
            }

            if self.metrics:
                self.metrics.size_history.append(len(self._cache))
                metrics_stats = self.metrics.get_stats()
                base_stats.update(metrics_stats)

            return base_stats

    def _make_space_for_entry(self, entry: CacheEntry) -> bool:
        """Make space for a new entry by evicting if necessary."""
        # Check size limits
        while (
            len(self._cache) >= self._max_size
            or self._current_memory_bytes + entry.size_bytes > self._max_memory_bytes
        ):
            if not self._cache:
                return False

            # Evict based on policy
            evicted_key = self._select_eviction_candidate()
            if not evicted_key:
                return False

            evicted_entry = self._cache[evicted_key]
            del self._cache[evicted_key]
            self._current_memory_bytes -= evicted_entry.size_bytes

            # Remove from access patterns
            if evicted_key in self._access_patterns:
                del self._access_patterns[evicted_key]

            if self.metrics:
                self.metrics.record_operation(CacheEventType.EVICTION)

        return True

    def _select_eviction_candidate(self) -> Optional[str]:
        """Select which entry to evict based on policy."""
        if not self._cache:
            return None

        if self._eviction_policy == EvictionPolicy.LRU:
            # OrderedDict with LRU: first item is least recently used
            return next(iter(self._cache))

        elif self._eviction_policy == EvictionPolicy.LFU:
            # Find least frequently used
            min_count = float("inf")
            lfu_key = None
            for key, entry in self._cache.items():
                if entry.access_count < min_count:
                    min_count = entry.access_count
                    lfu_key = key
            return lfu_key

        elif self._eviction_policy == EvictionPolicy.TTL:
            # Find entry with shortest time to expiry
            shortest_ttl = float("inf")
            ttl_key = None
            for key, entry in self._cache.items():
                if entry.ttl_seconds:
                    time_left = entry.time_until_expiry()
                    if time_left is not None and time_left < shortest_ttl:
                        shortest_ttl = time_left
                        ttl_key = key
            return ttl_key or next(iter(self._cache))  # Fallback to FIFO

        else:  # FIFO
            return next(iter(self._cache))

    def _start_background_cleanup(self) -> None:
        """Start background task for expired entry cleanup."""
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(self._cleanup_expired_entries())
        except RuntimeError:
            # No event loop running, cleanup will happen on access
            pass

    async def _cleanup_expired_entries(self) -> None:
        """Background task to clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute

                expired_keys = []
                with self._lock:
                    for key, entry in self._cache.items():
                        if entry.is_expired():
                            expired_keys.append(key)

                # Clean up expired entries
                for key in expired_keys:
                    with self._lock:
                        if key in self._cache and self._cache[key].is_expired():
                            entry = self._cache[key]
                            del self._cache[key]
                            self._current_memory_bytes -= entry.size_bytes
                            if key in self._access_patterns:
                                del self._access_patterns[key]

                            if self.metrics:
                                self.metrics.record_operation(CacheEventType.EXPIRATION)

                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup task: {e}")

    async def shutdown(self) -> None:
        """Shutdown cache and cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self.clear()
        logger.info("MemoryCache shutdown complete")

    def __repr__(self) -> str:
        return (
            f"MemoryCache(size={len(self._cache)}, "
            f"max_size={self._max_size}, "
            f"memory={self._current_memory_bytes}, "
            f"policy={self._eviction_policy.value})"
        )


# Convenience aliases for backward compatibility
LRUCache = MemoryCache  # For easy migration from monolithic code
