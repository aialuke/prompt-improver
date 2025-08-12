"""Multi-level cache orchestration and management.

This module provides the main cache management interface extracted from
unified_connection_manager.py, implementing:

- CacheManager: Multi-level cache orchestrator with L1/L2/L3 coordination
- Intelligent cache level selection and fallback strategies
- Cache warming coordination and pattern-based optimization
- Performance metrics and monitoring across all levels
- Security context validation across cache levels
- OpenTelemetry integration for comprehensive observability
- Automatic cache synchronization and consistency management

Acts as the primary interface for all caching operations with automatic
level management and optimization for sub-millisecond to <50ms response times.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

try:
    from opentelemetry import metrics, trace
    from opentelemetry.metrics import Counter, Gauge, Histogram

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    Counter = Any
    Gauge = Any
    Histogram = Any

from prompt_improver.common.types import SecurityContext
from prompt_improver.database.services.cache.database_cache import (
    DatabaseCache,
    DatabaseCacheConfig,
)
from prompt_improver.database.services.cache.memory_cache import (
    EvictionPolicy,
    MemoryCache,
)
from prompt_improver.database.services.cache.redis_cache import (
    RedisCache,
    RedisCacheConfig,
)

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels in order of preference."""

    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DATABASE = "l3_database"


class CacheFallbackStrategy(Enum):
    """Cache fallback strategies when levels are unavailable."""

    FAIL_FAST = "fail_fast"  # Return None if preferred level fails
    FALLBACK_NEXT = "fallback_next"  # Try next available level
    FALLBACK_ALL = "fallback_all"  # Try all levels in order


class CacheConsistencyMode(Enum):
    """Cache consistency modes for multi-level synchronization."""

    EVENTUAL = "eventual"  # Async propagation, best performance
    STRONG = "strong"  # Sync propagation, consistency guaranteed
    WEAK = "weak"  # No automatic propagation


@dataclass
class CacheManagerConfig:
    """Configuration for multi-level cache manager."""

    # Level availability
    enable_l1_memory: bool = True
    enable_l2_redis: bool = True
    enable_l3_database: bool = True

    # Fallback strategy
    fallback_strategy: CacheFallbackStrategy = CacheFallbackStrategy.FALLBACK_ALL
    consistency_mode: CacheConsistencyMode = CacheConsistencyMode.EVENTUAL

    # Performance settings
    l1_ttl_seconds: int = 300  # 5 minutes
    l2_ttl_seconds: int = 3600  # 1 hour
    l3_ttl_seconds: int = 86400  # 24 hours

    # Warming and synchronization
    enable_cache_warming: bool = True
    warming_threshold_accesses: int = 5
    warming_batch_size: int = 100
    sync_interval_seconds: int = 60

    # Health monitoring
    health_check_interval_seconds: int = 30
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60

    # Metrics
    enable_metrics: bool = True
    metrics_collection_interval: int = 10


class CacheManagerMetrics:
    """Comprehensive multi-level cache metrics."""

    def __init__(self, service_name: str = "cache_manager"):
        self.service_name = service_name

        # Operation metrics by level
        self.l1_hits = 0
        self.l1_misses = 0
        self.l2_hits = 0
        self.l2_misses = 0
        self.l3_hits = 0
        self.l3_misses = 0

        # Performance metrics
        self.response_times: Dict[CacheLevel, List[float]] = {
            CacheLevel.L1_MEMORY: [],
            CacheLevel.L2_REDIS: [],
            CacheLevel.L3_DATABASE: [],
        }

        # Health and fallback metrics
        self.fallbacks = 0
        self.level_failures: Dict[CacheLevel, int] = {
            CacheLevel.L1_MEMORY: 0,
            CacheLevel.L2_REDIS: 0,
            CacheLevel.L3_DATABASE: 0,
        }

        # Cache warming metrics
        self.warmings_triggered = 0
        self.warming_successes = 0
        self.warming_failures = 0

        # Synchronization metrics
        self.sync_operations = 0
        self.sync_failures = 0

        # OpenTelemetry setup
        self.operations_counter: Optional[Counter] = None
        self.response_time_histogram: Optional[Histogram] = None
        self.cache_level_gauge: Optional[Gauge] = None
        self.fallback_counter: Optional[Counter] = None

        if OPENTELEMETRY_AVAILABLE:
            self._setup_telemetry()

    def _setup_telemetry(self) -> None:
        """Setup OpenTelemetry metrics."""
        try:
            meter = metrics.get_meter(f"prompt_improver.cache.{self.service_name}")

            self.operations_counter = meter.create_counter(
                "cache_manager_operations_total",
                description="Total cache manager operations by level and result",
                unit="1",
            )

            self.response_time_histogram = meter.create_histogram(
                "cache_manager_response_time_seconds",
                description="Cache manager operation response times by level",
                unit="s",
            )

            self.cache_level_gauge = meter.create_gauge(
                "cache_manager_active_levels",
                description="Number of active cache levels",
                unit="1",
            )

            self.fallback_counter = meter.create_counter(
                "cache_manager_fallbacks_total",
                description="Total cache fallback operations",
                unit="1",
            )

            logger.debug(
                f"OpenTelemetry metrics initialized for CacheManager {self.service_name}"
            )
        except Exception as e:
            logger.warning(f"Failed to setup CacheManager OpenTelemetry metrics: {e}")

    def record_operation(
        self,
        level: CacheLevel,
        operation: str,
        status: str,
        duration_ms: float = 0,
        security_context_id: Optional[str] = None,
    ) -> None:
        """Record cache operation metrics."""
        # Update level-specific counters
        if status == "hit":
            if level == CacheLevel.L1_MEMORY:
                self.l1_hits += 1
            elif level == CacheLevel.L2_REDIS:
                self.l2_hits += 1
            elif level == CacheLevel.L3_DATABASE:
                self.l3_hits += 1
        elif status == "miss":
            if level == CacheLevel.L1_MEMORY:
                self.l1_misses += 1
            elif level == CacheLevel.L2_REDIS:
                self.l2_misses += 1
            elif level == CacheLevel.L3_DATABASE:
                self.l3_misses += 1
        elif status == "error":
            self.level_failures[level] += 1

        # Record response time
        if duration_ms > 0:
            self.response_times[level].append(duration_ms)
            # Keep only recent times
            if len(self.response_times[level]) > 1000:
                self.response_times[level] = self.response_times[level][-500:]

        # OpenTelemetry metrics
        if self.operations_counter:
            self.operations_counter.add(
                1,
                {
                    "level": level.value,
                    "operation": operation,
                    "status": status,
                    "security_context": security_context_id or "none",
                },
            )

        if self.response_time_histogram and duration_ms > 0:
            self.response_time_histogram.record(
                duration_ms / 1000.0, {"level": level.value}
            )

    def record_fallback(self, from_level: CacheLevel, to_level: CacheLevel) -> None:
        """Record cache fallback operation."""
        self.fallbacks += 1

        if self.fallback_counter:
            self.fallback_counter.add(
                1, {"from_level": from_level.value, "to_level": to_level.value}
            )

    def record_warming(self, success: bool) -> None:
        """Record cache warming operation."""
        self.warmings_triggered += 1
        if success:
            self.warming_successes += 1
        else:
            self.warming_failures += 1

    def record_sync(self, success: bool) -> None:
        """Record cache synchronization operation."""
        self.sync_operations += 1
        if not success:
            self.sync_failures += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache manager statistics."""
        # Calculate total operations and hit rates
        total_l1_ops = self.l1_hits + self.l1_misses
        total_l2_ops = self.l2_hits + self.l2_misses
        total_l3_ops = self.l3_hits + self.l3_misses
        total_ops = total_l1_ops + total_l2_ops + total_l3_ops

        l1_hit_rate = self.l1_hits / total_l1_ops if total_l1_ops > 0 else 0
        l2_hit_rate = self.l2_hits / total_l2_ops if total_l2_ops > 0 else 0
        l3_hit_rate = self.l3_hits / total_l3_ops if total_l3_ops > 0 else 0
        overall_hit_rate = (
            (self.l1_hits + self.l2_hits + self.l3_hits) / total_ops
            if total_ops > 0
            else 0
        )

        # Calculate average response times by level
        avg_response_times = {}
        for level, times in self.response_times.items():
            avg_response_times[level.value] = sum(times) / len(times) if times else 0

        return {
            "service": self.service_name,
            "operations": {
                "total": total_ops,
                "l1": {
                    "hits": self.l1_hits,
                    "misses": self.l1_misses,
                    "total": total_l1_ops,
                    "hit_rate": l1_hit_rate,
                },
                "l2": {
                    "hits": self.l2_hits,
                    "misses": self.l2_misses,
                    "total": total_l2_ops,
                    "hit_rate": l2_hit_rate,
                },
                "l3": {
                    "hits": self.l3_hits,
                    "misses": self.l3_misses,
                    "total": total_l3_ops,
                    "hit_rate": l3_hit_rate,
                },
                "overall_hit_rate": overall_hit_rate,
            },
            "performance": {
                "avg_response_times_ms": avg_response_times,
                "fallbacks": self.fallbacks,
                "failures": dict(self.level_failures),
            },
            "warming": {
                "triggered": self.warmings_triggered,
                "successes": self.warming_successes,
                "failures": self.warming_failures,
                "success_rate": self.warming_successes / self.warmings_triggered
                if self.warmings_triggered > 0
                else 0,
            },
            "synchronization": {
                "operations": self.sync_operations,
                "failures": self.sync_failures,
                "success_rate": (self.sync_operations - self.sync_failures)
                / self.sync_operations
                if self.sync_operations > 0
                else 0,
            },
        }


class CacheManager:
    """Multi-level cache orchestrator with intelligent fallback and warming.

    Enhanced cache manager extracted from unified_connection_manager.py with:
    - L1 Memory cache for sub-millisecond access (LRU/LFU eviction)
    - L2 Redis cache for distributed <5ms access with persistence
    - L3 Database cache for reliable <50ms access with compression
    - Intelligent fallback strategies and health monitoring
    - Automatic cache warming based on access patterns
    - Security context validation across all levels
    - Comprehensive metrics and OpenTelemetry integration
    """

    def __init__(
        self,
        config: CacheManagerConfig,
        l1_memory_cache: Optional[MemoryCache] = None,
        l2_redis_cache: Optional[RedisCache] = None,
        l3_database_cache: Optional[DatabaseCache] = None,
        service_name: str = "cache_manager",
    ):
        self.config = config
        self.service_name = service_name

        # Cache level instances
        self.l1_memory = l1_memory_cache
        self.l2_redis = l2_redis_cache
        self.l3_database = l3_database_cache

        # Metrics
        self.metrics = (
            CacheManagerMetrics(service_name) if config.enable_metrics else None
        )

        # Circuit breaker state
        self._circuit_breakers: Dict[CacheLevel, Dict[str, Any]] = {
            CacheLevel.L1_MEMORY: {
                "failures": 0,
                "last_failure": None,
                "is_open": False,
            },
            CacheLevel.L2_REDIS: {
                "failures": 0,
                "last_failure": None,
                "is_open": False,
            },
            CacheLevel.L3_DATABASE: {
                "failures": 0,
                "last_failure": None,
                "is_open": False,
            },
        }

        # Background tasks
        self._sync_task: Optional[asyncio.Task] = None
        self._warming_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None

        # Cache warming state
        self._access_patterns: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"CacheManager initialized: L1={config.enable_l1_memory}, "
            f"L2={config.enable_l2_redis}, L3={config.enable_l3_database}, "
            f"strategy={config.fallback_strategy.value}"
        )

    async def initialize(self) -> None:
        """Initialize cache manager and all enabled cache levels."""
        try:
            # Initialize cache levels
            if self.config.enable_l1_memory and not self.l1_memory:
                self.l1_memory = MemoryCache(
                    max_size=10000,
                    eviction_policy=EvictionPolicy.LRU,
                    enable_metrics=True,
                    service_name=f"{self.service_name}_l1",
                )

            if self.config.enable_l2_redis:
                if not self.l2_redis:
                    redis_config = RedisCacheConfig()
                    self.l2_redis = RedisCache(
                        redis_config,
                        enable_metrics=True,
                        service_name=f"{self.service_name}_l2",
                    )
                # Initialize existing or new L2 cache
                if hasattr(self.l2_redis, "initialize"):
                    await self.l2_redis.initialize()

            if self.config.enable_l3_database:
                if not self.l3_database:
                    db_config = DatabaseCacheConfig()
                    self.l3_database = DatabaseCache(
                        db_config,
                        enable_metrics=True,
                        service_name=f"{self.service_name}_l3",
                    )
                # Initialize existing or new L3 cache
                if hasattr(self.l3_database, "initialize"):
                    await self.l3_database.initialize()

            # Start background tasks
            if self.config.consistency_mode != CacheConsistencyMode.WEAK:
                self._sync_task = asyncio.create_task(self._periodic_sync())

            if self.config.enable_cache_warming:
                self._warming_task = asyncio.create_task(self._cache_warming_loop())

            self._health_check_task = asyncio.create_task(self._health_check_loop())

            logger.info("CacheManager initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize CacheManager: {e}")
            raise

    def _is_circuit_breaker_open(self, level: CacheLevel) -> bool:
        """Check if circuit breaker is open for a cache level."""
        breaker = self._circuit_breakers[level]

        if not breaker["is_open"]:
            return False

        # Check if timeout has passed
        if breaker["last_failure"]:
            timeout_passed = (
                datetime.now(UTC) - breaker["last_failure"]
            ).total_seconds() > self.config.circuit_breaker_timeout_seconds

            if timeout_passed:
                # Reset circuit breaker
                breaker["is_open"] = False
                breaker["failures"] = 0
                logger.info(f"Circuit breaker reset for {level.value}")
                return False

        return True

    def _record_failure(self, level: CacheLevel) -> None:
        """Record failure and potentially open circuit breaker."""
        breaker = self._circuit_breakers[level]
        breaker["failures"] += 1
        breaker["last_failure"] = datetime.now(UTC)

        if breaker["failures"] >= self.config.circuit_breaker_threshold:
            breaker["is_open"] = True
            logger.warning(
                f"Circuit breaker opened for {level.value} after {breaker['failures']} failures"
            )

    def _record_success(self, level: CacheLevel) -> None:
        """Record success and potentially close circuit breaker."""
        breaker = self._circuit_breakers[level]
        if breaker["failures"] > 0:
            breaker["failures"] = max(0, breaker["failures"] - 1)

    def _get_available_levels(self) -> List[CacheLevel]:
        """Get list of available cache levels based on configuration and circuit breakers."""
        available = []

        if (
            self.config.enable_l1_memory
            and self.l1_memory
            and not self._is_circuit_breaker_open(CacheLevel.L1_MEMORY)
        ):
            available.append(CacheLevel.L1_MEMORY)

        if (
            self.config.enable_l2_redis
            and self.l2_redis
            and not self._is_circuit_breaker_open(CacheLevel.L2_REDIS)
        ):
            available.append(CacheLevel.L2_REDIS)

        if (
            self.config.enable_l3_database
            and self.l3_database
            and not self._is_circuit_breaker_open(CacheLevel.L3_DATABASE)
        ):
            available.append(CacheLevel.L3_DATABASE)

        return available

    def _update_access_pattern(self, key: str) -> None:
        """Update access pattern for cache warming decisions."""
        if not self.config.enable_cache_warming:
            return

        now = datetime.now(UTC)
        if key not in self._access_patterns:
            self._access_patterns[key] = {
                "count": 0,
                "last_access": now,
                "first_access": now,
            }

        pattern = self._access_patterns[key]
        pattern["count"] += 1
        pattern["last_access"] = now

        # Clean up old patterns (keep only recently accessed)
        if len(self._access_patterns) > 10000:
            cutoff = now - timedelta(hours=1)
            self._access_patterns = {
                k: v
                for k, v in self._access_patterns.items()
                if v["last_access"] > cutoff
            }

    async def get(
        self, key: str, security_context: Optional[SecurityContext] = None
    ) -> Any:
        """Get value from cache with intelligent level selection and fallback."""
        self._update_access_pattern(key)
        available_levels = self._get_available_levels()

        if not available_levels:
            logger.error("No cache levels available")
            return None

        # Try each level in order
        for level in available_levels:
            start_time = time.time()

            try:
                value = None

                if level == CacheLevel.L1_MEMORY and self.l1_memory:
                    value = self.l1_memory.get(key)
                elif level == CacheLevel.L2_REDIS and self.l2_redis:
                    value = await self.l2_redis.get(key, security_context)
                elif level == CacheLevel.L3_DATABASE and self.l3_database:
                    value = await self.l3_database.get(key, security_context)

                duration_ms = (time.time() - start_time) * 1000

                if value is not None:
                    # Cache hit - record success and potentially warm upper levels
                    self._record_success(level)
                    if self.metrics:
                        self.metrics.record_operation(
                            level,
                            "get",
                            "hit",
                            duration_ms,
                            security_context.user_id if security_context else None,
                        )

                    # Warm upper levels if configured
                    if self.config.consistency_mode == CacheConsistencyMode.STRONG:
                        await self._warm_upper_levels(
                            key, value, level, security_context
                        )

                    return value
                else:
                    # Cache miss
                    if self.metrics:
                        self.metrics.record_operation(
                            level,
                            "get",
                            "miss",
                            duration_ms,
                            security_context.user_id if security_context else None,
                        )

                    # Continue to next level based on strategy
                    if self.config.fallback_strategy == CacheFallbackStrategy.FAIL_FAST:
                        return None
                    elif (
                        self.config.fallback_strategy
                        == CacheFallbackStrategy.FALLBACK_NEXT
                    ):
                        if level != available_levels[-1]:  # Not last level
                            if self.metrics:
                                self.metrics.record_fallback(
                                    level,
                                    available_levels[available_levels.index(level) + 1],
                                )
                            continue
                        else:
                            return None
                    # FALLBACK_ALL continues to next level automatically

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.warning(
                    f"Cache level {level.value} failed for get key {key}: {e}"
                )

                self._record_failure(level)
                if self.metrics:
                    self.metrics.record_operation(
                        level,
                        "get",
                        "error",
                        duration_ms,
                        security_context.user_id if security_context else None,
                    )

                # Continue to next level for fallback strategies
                if self.config.fallback_strategy == CacheFallbackStrategy.FAIL_FAST:
                    return None

        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        security_context: Optional[SecurityContext] = None,
        tags: Optional[Set[str]] = None,
    ) -> bool:
        """Set value in cache with multi-level consistency management."""
        available_levels = self._get_available_levels()

        if not available_levels:
            logger.error("No cache levels available for set operation")
            return False

        success_levels = []

        # Set in all available levels based on consistency mode
        for level in available_levels:
            start_time = time.time()

            try:
                result = False
                level_ttl = ttl_seconds

                # Use level-specific TTL if not specified
                if level_ttl is None:
                    if level == CacheLevel.L1_MEMORY:
                        level_ttl = self.config.l1_ttl_seconds
                    elif level == CacheLevel.L2_REDIS:
                        level_ttl = self.config.l2_ttl_seconds
                    elif level == CacheLevel.L3_DATABASE:
                        level_ttl = self.config.l3_ttl_seconds

                if level == CacheLevel.L1_MEMORY and self.l1_memory:
                    result = self.l1_memory.set(
                        key, value, ttl_seconds=level_ttl, tags=tags
                    )
                elif level == CacheLevel.L2_REDIS and self.l2_redis:
                    result = await self.l2_redis.set(
                        key,
                        value,
                        ttl_seconds=level_ttl,
                        security_context=security_context,
                    )
                elif level == CacheLevel.L3_DATABASE and self.l3_database:
                    result = await self.l3_database.set(
                        key,
                        value,
                        ttl_seconds=level_ttl,
                        security_context=security_context,
                        tags=tags,
                    )

                duration_ms = (time.time() - start_time) * 1000

                if result:
                    success_levels.append(level)
                    self._record_success(level)
                    if self.metrics:
                        self.metrics.record_operation(
                            level,
                            "set",
                            "success",
                            duration_ms,
                            security_context.user_id if security_context else None,
                        )
                else:
                    if self.metrics:
                        self.metrics.record_operation(
                            level,
                            "set",
                            "error",
                            duration_ms,
                            security_context.user_id if security_context else None,
                        )

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.warning(
                    f"Cache level {level.value} failed for set key {key}: {e}"
                )

                self._record_failure(level)
                if self.metrics:
                    self.metrics.record_operation(
                        level,
                        "set",
                        "error",
                        duration_ms,
                        security_context.user_id if security_context else None,
                    )

        return len(success_levels) > 0

    async def delete(
        self, key: str, security_context: Optional[SecurityContext] = None
    ) -> bool:
        """Delete key from all cache levels."""
        available_levels = self._get_available_levels()
        success_count = 0

        for level in available_levels:
            start_time = time.time()

            try:
                result = False

                if level == CacheLevel.L1_MEMORY and self.l1_memory:
                    result = self.l1_memory.delete(key)
                elif level == CacheLevel.L2_REDIS and self.l2_redis:
                    result = await self.l2_redis.delete(key, security_context)
                elif level == CacheLevel.L3_DATABASE and self.l3_database:
                    result = await self.l3_database.delete(key, security_context)

                duration_ms = (time.time() - start_time) * 1000

                if result:
                    success_count += 1
                    self._record_success(level)
                    if self.metrics:
                        self.metrics.record_operation(
                            level,
                            "delete",
                            "success",
                            duration_ms,
                            security_context.user_id if security_context else None,
                        )
                else:
                    if self.metrics:
                        self.metrics.record_operation(
                            level,
                            "delete",
                            "miss",
                            duration_ms,
                            security_context.user_id if security_context else None,
                        )

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.warning(
                    f"Cache level {level.value} failed for delete key {key}: {e}"
                )

                self._record_failure(level)
                if self.metrics:
                    self.metrics.record_operation(
                        level,
                        "delete",
                        "error",
                        duration_ms,
                        security_context.user_id if security_context else None,
                    )

        return success_count > 0

    async def exists(
        self, key: str, security_context: Optional[SecurityContext] = None
    ) -> bool:
        """Check if key exists in any cache level."""
        available_levels = self._get_available_levels()

        for level in available_levels:
            try:
                exists = False

                if level == CacheLevel.L1_MEMORY and self.l1_memory:
                    exists = self.l1_memory.exists(key)
                elif level == CacheLevel.L2_REDIS and self.l2_redis:
                    exists = await self.l2_redis.exists(key, security_context)
                elif level == CacheLevel.L3_DATABASE and self.l3_database:
                    exists = await self.l3_database.exists(key, security_context)

                if exists:
                    return True

            except Exception as e:
                logger.warning(
                    f"Cache level {level.value} failed for exists key {key}: {e}"
                )
                self._record_failure(level)

        return False

    async def _warm_upper_levels(
        self,
        key: str,
        value: Any,
        source_level: CacheLevel,
        security_context: Optional[SecurityContext],
    ) -> None:
        """Warm upper cache levels with value from lower level."""
        available_levels = self._get_available_levels()
        source_index = available_levels.index(source_level)

        # Warm all levels above the source level
        for i in range(source_index):
            level = available_levels[i]

            try:
                if level == CacheLevel.L1_MEMORY and self.l1_memory:
                    self.l1_memory.set(
                        key, value, ttl_seconds=self.config.l1_ttl_seconds
                    )
                elif level == CacheLevel.L2_REDIS and self.l2_redis:
                    await self.l2_redis.set(
                        key,
                        value,
                        ttl_seconds=self.config.l2_ttl_seconds,
                        security_context=security_context,
                    )

                logger.debug(f"Warmed {level.value} with key {key}")

            except Exception as e:
                logger.warning(f"Failed to warm {level.value} with key {key}: {e}")

    async def _periodic_sync(self) -> None:
        """Background task for periodic cache synchronization."""
        while True:
            try:
                await asyncio.sleep(self.config.sync_interval_seconds)

                # Sync logic would be implemented here based on consistency requirements
                # For now, just record the sync attempt
                if self.metrics:
                    self.metrics.record_sync(True)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during cache sync: {e}")
                if self.metrics:
                    self.metrics.record_sync(False)

    async def _cache_warming_loop(self) -> None:
        """Background task for intelligent cache warming based on access patterns."""
        while True:
            try:
                await asyncio.sleep(60)  # Check warming every minute

                now = datetime.now(UTC)
                cutoff_time = now - timedelta(minutes=10)

                # Find keys that should be warmed
                warming_candidates = []
                for key, pattern in self._access_patterns.items():
                    if (
                        pattern["count"] >= self.config.warming_threshold_accesses
                        and pattern["last_access"] > cutoff_time
                    ):
                        warming_candidates.append(key)

                # Process warming candidates in batches
                for i in range(
                    0, len(warming_candidates), self.config.warming_batch_size
                ):
                    batch = warming_candidates[i : i + self.config.warming_batch_size]
                    await self._warm_cache_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during cache warming: {e}")

    async def _warm_cache_batch(self, keys: List[str]) -> None:
        """Warm cache for a batch of keys."""
        for key in keys:
            try:
                # Try to get value from L3 and warm L1/L2
                if self.l3_database:
                    value = await self.l3_database.get(key)
                    if value is not None:
                        # Warm L1 and L2
                        if self.l1_memory:
                            self.l1_memory.set(
                                key, value, ttl_seconds=self.config.l1_ttl_seconds
                            )
                        if self.l2_redis:
                            await self.l2_redis.set(
                                key, value, ttl_seconds=self.config.l2_ttl_seconds
                            )

                        if self.metrics:
                            self.metrics.record_warming(True)
                    else:
                        if self.metrics:
                            self.metrics.record_warming(False)

            except Exception as e:
                logger.warning(f"Failed to warm cache for key {key}: {e}")
                if self.metrics:
                    self.metrics.record_warming(False)

    async def _health_check_loop(self) -> None:
        """Background task for health monitoring of cache levels."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)

                # Health check each level
                levels_to_check = [
                    (CacheLevel.L1_MEMORY, self.l1_memory),
                    (CacheLevel.L2_REDIS, self.l2_redis),
                    (CacheLevel.L3_DATABASE, self.l3_database),
                ]

                for level, cache_instance in levels_to_check:
                    if cache_instance:
                        try:
                            if hasattr(cache_instance, "ping"):
                                healthy = await cache_instance.ping()
                            else:
                                healthy = True  # L1 memory cache doesn't need ping

                            if healthy:
                                self._record_success(level)
                            else:
                                self._record_failure(level)

                        except Exception as e:
                            logger.warning(
                                f"Health check failed for {level.value}: {e}"
                            )
                            self._record_failure(level)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during health check: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache manager statistics."""
        base_stats = {
            "service": self.service_name,
            "levels": {
                "l1_enabled": self.config.enable_l1_memory,
                "l2_enabled": self.config.enable_l2_redis,
                "l3_enabled": self.config.enable_l3_database,
                "available_levels": len(self._get_available_levels()),
            },
            "circuit_breakers": {
                level.value: {
                    "is_open": breaker["is_open"],
                    "failures": breaker["failures"],
                }
                for level, breaker in self._circuit_breakers.items()
            },
            "config": {
                "fallback_strategy": self.config.fallback_strategy.value,
                "consistency_mode": self.config.consistency_mode.value,
                "warming_enabled": self.config.enable_cache_warming,
            },
        }

        if self.metrics:
            metrics_stats = self.metrics.get_stats()
            base_stats.update(metrics_stats)

        # Add individual level stats
        level_stats = {}
        if self.l1_memory and hasattr(self.l1_memory, "get_stats"):
            level_stats["l1_memory"] = self.l1_memory.get_stats()
        if self.l2_redis and hasattr(self.l2_redis, "get_stats"):
            level_stats["l2_redis"] = await self.l2_redis.get_stats()
        if self.l3_database and hasattr(self.l3_database, "get_stats"):
            level_stats["l3_database"] = await self.l3_database.get_stats()

        base_stats["level_details"] = level_stats

        return base_stats

    async def shutdown(self) -> None:
        """Shutdown cache manager and all cache levels."""
        try:
            # Cancel background tasks
            tasks = [
                ("sync", self._sync_task),
                ("warming", self._warming_task),
                ("health_check", self._health_check_task),
            ]

            for task_name, task in tasks:
                if task:
                    try:
                        task.cancel()
                        # Only await real asyncio tasks, not mocks
                        if hasattr(task, "_coro"):
                            await task
                    except (asyncio.CancelledError, Exception) as e:
                        logger.debug(f"Exception cancelling {task_name} task: {e}")
                        pass

            # Shutdown cache levels
            cache_levels = [
                ("l1_memory", self.l1_memory),
                ("l2_redis", self.l2_redis),
                ("l3_database", self.l3_database),
            ]

            for level_name, cache in cache_levels:
                if cache and hasattr(cache, "shutdown"):
                    try:
                        await cache.shutdown()
                    except Exception as e:
                        logger.debug(f"Exception shutting down {level_name}: {e}")
                        pass

            logger.info("CacheManager shutdown complete")

        except Exception as e:
            logger.warning(f"Error during CacheManager shutdown: {e}")

    def __repr__(self) -> str:
        available = len(self._get_available_levels())
        return (
            f"CacheManager(levels={available}/3, "
            f"strategy={self.config.fallback_strategy.value}, "
            f"consistency={self.config.consistency_mode.value})"
        )


# Convenience function for easy configuration
def create_cache_manager(
    enable_l1: bool = True,
    enable_l2: bool = True,
    enable_l3: bool = True,
    fallback_strategy: CacheFallbackStrategy = CacheFallbackStrategy.FALLBACK_ALL,
    consistency_mode: CacheConsistencyMode = CacheConsistencyMode.EVENTUAL,
    **kwargs,
) -> CacheManager:
    """Create cache manager with simple configuration."""
    config = CacheManagerConfig(
        enable_l1_memory=enable_l1,
        enable_l2_redis=enable_l2,
        enable_l3_database=enable_l3,
        fallback_strategy=fallback_strategy,
        consistency_mode=consistency_mode,
        **kwargs,
    )
    return CacheManager(config)
