"""Intelligent cache warming with predictive algorithms.

This module provides cache warming functionality extracted from
unified_connection_manager.py:

- CacheWarmer: Predictive cache population engine
- Access pattern analysis and prediction
- Priority-based warming algorithms
- Background warming processes
- Integration with all cache levels
- Resource-aware warming strategies

Proactively populates caches based on usage patterns to minimize cache misses.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# OpenTelemetry optional integration
try:
    from opentelemetry import metrics, trace

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

logger = logging.getLogger(__name__)


class WarmingPriority(Enum):
    """Priority levels for cache warming operations."""

    CRITICAL = 1  # Immediate warming required
    HIGH = 2  # High importance, warm soon
    NORMAL = 3  # Standard priority
    LOW = 4  # Low priority, warm when resources available
    BACKGROUND = 5  # Background warming only


class PatternType(Enum):
    """Types of access patterns for prediction."""

    SEQUENTIAL = "sequential"  # Sequential key access
    TEMPORAL = "temporal"  # Time-based patterns
    FREQUENCY = "frequency"  # Frequency-based patterns
    SPATIAL = "spatial"  # Spatial/geographic patterns
    USER_BASED = "user_based"  # User-specific patterns
    SESSION_BASED = "session_based"  # Session-based patterns


@dataclass
class AccessPattern:
    """Represents a detected access pattern."""

    pattern_type: PatternType
    keys: List[str]
    confidence: float  # 0.0 to 1.0
    frequency: float  # Accesses per second
    last_seen: datetime
    prediction_accuracy: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if pattern is still active (seen recently)."""
        return (datetime.now(UTC) - self.last_seen).seconds < 300  # 5 minutes


@dataclass
class WarmingTask:
    """Represents a cache warming task."""

    keys: List[str]
    priority: WarmingPriority
    pattern: Optional[AccessPattern] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    estimated_duration: Optional[float] = None
    cache_levels: List[str] = field(default_factory=list)

    @property
    def age_seconds(self) -> float:
        """Get age of warming task in seconds."""
        return (datetime.now(UTC) - self.created_at).total_seconds()


@dataclass
class CacheWarmerConfig:
    """Configuration for CacheWarmer."""

    # Pattern detection settings
    pattern_detection_enabled: bool = True
    min_pattern_confidence: float = 0.7
    pattern_history_size: int = 1000

    # Warming settings
    max_concurrent_tasks: int = 5
    max_keys_per_task: int = 100
    warming_batch_size: int = 20

    # Resource limits
    max_memory_usage_mb: int = 100
    cpu_throttle_threshold: float = 0.8

    # Background task settings
    background_interval_seconds: float = 30.0
    cleanup_interval_seconds: float = 300.0  # 5 minutes

    # Prediction settings
    prediction_window_minutes: int = 15
    prediction_accuracy_threshold: float = 0.6

    # Performance settings
    enable_metrics: bool = True
    log_pattern_stats: bool = False


class CacheWarmer:
    """Intelligent cache warming with predictive algorithms.

    Analyzes cache access patterns and proactively warms caches to minimize
    cache misses. Integrates with all cache levels and provides resource-aware
    warming strategies.
    """

    def __init__(self, cache_manager, config: Optional[CacheWarmerConfig] = None):
        self.cache_manager = cache_manager
        self.config = config or CacheWarmerConfig()

        # Pattern tracking
        self._access_history: deque = deque(maxlen=self.config.pattern_history_size)
        self._detected_patterns: Dict[str, AccessPattern] = {}
        self._pattern_predictors: Dict[PatternType, Callable] = {
            PatternType.SEQUENTIAL: self._predict_sequential,
            PatternType.TEMPORAL: self._predict_temporal,
            PatternType.FREQUENCY: self._predict_frequency,
        }

        # Task management
        self._warming_tasks: Dict[str, WarmingTask] = {}
        self._active_warming: Set[str] = set()
        self._task_semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)

        # Background tasks
        self._background_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Performance metrics
        self.patterns_detected = 0
        self.warming_tasks_completed = 0
        self.warming_tasks_failed = 0
        self.cache_hits_improved = 0
        self.total_keys_warmed = 0

        # OpenTelemetry metrics
        if OPENTELEMETRY_AVAILABLE:
            meter = metrics.get_meter(__name__)
            self._patterns_counter = meter.create_counter(
                "cache_warmer_patterns_detected",
                description="Total patterns detected by type",
            )
            self._warming_counter = meter.create_counter(
                "cache_warmer_operations_total",
                description="Total warming operations by status",
            )
            self._warming_histogram = meter.create_histogram(
                "cache_warmer_duration_seconds",
                description="Cache warming operation duration",
            )
        else:
            self._patterns_counter = None
            self._warming_counter = None
            self._warming_histogram = None

        logger.info(
            f"CacheWarmer initialized with {self.config.max_concurrent_tasks} max concurrent tasks"
        )

    async def start_background_tasks(self) -> None:
        """Start background tasks for pattern detection and warming."""
        if self._background_task and not self._background_task.done():
            logger.warning("Background tasks already running")
            return

        self._shutdown_event.clear()
        self._background_task = asyncio.create_task(self._background_warming_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Started cache warmer background tasks")

    async def stop_background_tasks(self) -> None:
        """Stop background tasks gracefully."""
        self._shutdown_event.set()

        # Wait for tasks to complete
        for task in [self._background_task, self._cleanup_task]:
            if task and not task.done():
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Background task did not stop gracefully")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        logger.info("Stopped cache warmer background tasks")

    async def record_access(
        self, key: str, cache_level: str = "unknown", user_id: Optional[str] = None
    ) -> None:
        """Record cache access for pattern detection.

        Args:
            key: Cache key that was accessed
            cache_level: Which cache level served the request
            user_id: Optional user ID for user-based patterns
        """
        access_record = {
            "key": key,
            "timestamp": datetime.now(UTC),
            "cache_level": cache_level,
            "user_id": user_id,
        }

        self._access_history.append(access_record)

        # Trigger pattern detection periodically
        if (
            self.config.pattern_detection_enabled
            and len(self._access_history) % 10 == 0
        ):
            await self._detect_patterns()

    async def _detect_patterns(self) -> None:
        """Analyze access history to detect patterns."""
        if len(self._access_history) < 10:
            return

        # Get recent access data
        recent_accesses = list(self._access_history)[-100:]  # Last 100 accesses

        # Detect different pattern types
        await self._detect_sequential_patterns(recent_accesses)
        await self._detect_temporal_patterns(recent_accesses)
        await self._detect_frequency_patterns(recent_accesses)

        # Clean up old patterns
        current_time = datetime.now(UTC)
        inactive_patterns = [
            pattern_id
            for pattern_id, pattern in self._detected_patterns.items()
            if (current_time - pattern.last_seen).seconds > 600  # 10 minutes
        ]

        for pattern_id in inactive_patterns:
            del self._detected_patterns[pattern_id]

    async def _detect_sequential_patterns(self, accesses: List[Dict[str, Any]]) -> None:
        """Detect sequential access patterns."""
        if len(accesses) < 5:
            return

        # Look for sequences of keys
        key_sequence = [access["key"] for access in accesses[-10:]]

        # Find repeating subsequences
        for seq_len in range(2, 6):  # Check sequences of length 2-5
            if seq_len > len(key_sequence):
                continue

            for i in range(len(key_sequence) - seq_len + 1):
                subseq = key_sequence[i : i + seq_len]

                # Check if this subsequence appears multiple times
                count = 0
                for j in range(len(key_sequence) - seq_len + 1):
                    if key_sequence[j : j + seq_len] == subseq:
                        count += 1

                if count >= 2:  # Pattern detected
                    pattern_id = f"seq_{hash(tuple(subseq))}"
                    confidence = min(0.9, count * 0.3)

                    if confidence >= self.config.min_pattern_confidence:
                        self._detected_patterns[pattern_id] = AccessPattern(
                            pattern_type=PatternType.SEQUENTIAL,
                            keys=subseq,
                            confidence=confidence,
                            frequency=count / 10.0,  # Approximate frequency
                            last_seen=datetime.now(UTC),
                        )

                        self.patterns_detected += 1

                        if self._patterns_counter:
                            self._patterns_counter.add(1, {"type": "sequential"})

                        logger.debug(
                            f"Detected sequential pattern: {subseq} (confidence: {confidence:.2f})"
                        )

    async def _detect_temporal_patterns(self, accesses: List[Dict[str, Any]]) -> None:
        """Detect time-based patterns."""
        if len(accesses) < 10:
            return

        # Group accesses by key and analyze timing
        key_times = defaultdict(list)
        for access in accesses:
            key_times[access["key"]].append(access["timestamp"])

        for key, timestamps in key_times.items():
            if len(timestamps) < 3:
                continue

            # Calculate intervals between accesses
            intervals = []
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i - 1]).total_seconds()
                intervals.append(interval)

            if intervals:
                avg_interval = mean(intervals)
                interval_variance = sum(
                    (x - avg_interval) ** 2 for x in intervals
                ) / len(intervals)
                consistency = 1.0 / (1.0 + interval_variance / (avg_interval**2))

                if consistency >= self.config.min_pattern_confidence:
                    pattern_id = f"temporal_{hash(key)}_{int(avg_interval)}"

                    self._detected_patterns[pattern_id] = AccessPattern(
                        pattern_type=PatternType.TEMPORAL,
                        keys=[key],
                        confidence=consistency,
                        frequency=1.0 / avg_interval if avg_interval > 0 else 0.0,
                        last_seen=timestamps[-1],
                        metadata={
                            "avg_interval": avg_interval,
                            "variance": interval_variance,
                        },
                    )

                    self.patterns_detected += 1

                    if self._patterns_counter:
                        self._patterns_counter.add(1, {"type": "temporal"})

                    logger.debug(
                        f"Detected temporal pattern for {key}: {avg_interval:.1f}s interval (confidence: {consistency:.2f})"
                    )

    async def _detect_frequency_patterns(self, accesses: List[Dict[str, Any]]) -> None:
        """Detect frequency-based patterns."""
        if len(accesses) < 10:
            return

        # Count key frequencies
        key_counts = defaultdict(int)
        for access in accesses:
            key_counts[access["key"]] += 1

        total_accesses = len(accesses)

        # Identify frequently accessed keys
        for key, count in key_counts.items():
            frequency = count / total_accesses

            if frequency >= 0.1:  # Accessed in >10% of samples
                pattern_id = f"freq_{hash(key)}"
                confidence = min(0.95, frequency * 2)

                if confidence >= self.config.min_pattern_confidence:
                    self._detected_patterns[pattern_id] = AccessPattern(
                        pattern_type=PatternType.FREQUENCY,
                        keys=[key],
                        confidence=confidence,
                        frequency=frequency * 10,  # Scale to per-second estimate
                        last_seen=datetime.now(UTC),
                        metadata={"access_count": count, "frequency_ratio": frequency},
                    )

                    self.patterns_detected += 1

                    if self._patterns_counter:
                        self._patterns_counter.add(1, {"type": "frequency"})

                    logger.debug(
                        f"Detected frequency pattern for {key}: {frequency:.2f} ratio (confidence: {confidence:.2f})"
                    )

    async def create_warming_task(
        self,
        keys: List[str],
        priority: WarmingPriority = WarmingPriority.NORMAL,
        cache_levels: Optional[List[str]] = None,
    ) -> str:
        """Create a cache warming task.

        Args:
            keys: List of keys to warm
            priority: Warming priority
            cache_levels: Specific cache levels to warm (default: all)

        Returns:
            Task ID
        """
        if len(keys) > self.config.max_keys_per_task:
            logger.warning(
                f"Too many keys ({len(keys)}), truncating to {self.config.max_keys_per_task}"
            )
            keys = keys[: self.config.max_keys_per_task]

        task_id = f"warm_{int(time.time())}_{hash(tuple(keys))}"
        cache_levels = cache_levels or ["l1_memory", "l2_redis"]

        task = WarmingTask(keys=keys, priority=priority, cache_levels=cache_levels)

        self._warming_tasks[task_id] = task

        # Schedule immediate execution for critical priority
        if priority == WarmingPriority.CRITICAL:
            asyncio.create_task(self._execute_warming_task(task_id))

        logger.debug(
            f"Created warming task {task_id} with {len(keys)} keys (priority: {priority.name})"
        )
        return task_id

    async def warm_from_patterns(self, max_predictions: int = 10) -> int:
        """Create warming tasks from detected patterns.

        Args:
            max_predictions: Maximum number of predictions to generate

        Returns:
            Number of warming tasks created
        """
        if not self._detected_patterns:
            return 0

        tasks_created = 0
        predictions_made = 0

        # Sort patterns by confidence and recency
        sorted_patterns = sorted(
            self._detected_patterns.values(),
            key=lambda p: (p.confidence, p.last_seen),
            reverse=True,
        )

        for pattern in sorted_patterns[:max_predictions]:
            if predictions_made >= max_predictions:
                break

            # Generate prediction based on pattern type
            predicted_keys = await self._predict_from_pattern(pattern)

            if predicted_keys:
                # Determine priority based on pattern confidence and frequency
                if pattern.confidence >= 0.9 and pattern.frequency > 1.0:
                    priority = WarmingPriority.HIGH
                elif pattern.confidence >= 0.8:
                    priority = WarmingPriority.NORMAL
                else:
                    priority = WarmingPriority.LOW

                await self.create_warming_task(predicted_keys, priority)
                tasks_created += 1
                predictions_made += 1

                logger.debug(
                    f"Created warming task from {pattern.pattern_type.value} pattern (confidence: {pattern.confidence:.2f})"
                )

        return tasks_created

    async def _predict_from_pattern(self, pattern: AccessPattern) -> List[str]:
        """Generate key predictions from a pattern."""
        predictor = self._pattern_predictors.get(pattern.pattern_type)
        if not predictor:
            return pattern.keys  # Fallback to pattern keys

        try:
            return await predictor(pattern)
        except Exception as e:
            logger.warning(f"Pattern prediction failed for {pattern.pattern_type}: {e}")
            return pattern.keys

    async def _predict_sequential(self, pattern: AccessPattern) -> List[str]:
        """Predict next keys in a sequential pattern."""
        # For sequential patterns, predict the next keys in the sequence
        base_keys = pattern.keys
        predicted_keys = []

        # Try to identify numerical sequences
        try:
            # Check if keys follow a numerical pattern
            if all(
                key.replace("_", "").replace("-", "").isdigit()
                for key in base_keys[-2:]
            ):
                last_nums = [
                    int(key.replace("_", "").replace("-", "")) for key in base_keys[-2:]
                ]
                if len(last_nums) >= 2:
                    step = last_nums[1] - last_nums[0]
                    next_num = last_nums[1] + step

                    # Generate format similar to original keys
                    template = base_keys[-1]
                    for i in range(3):  # Predict next 3 keys
                        predicted_key = template.replace(
                            str(last_nums[1]), str(next_num + i * step)
                        )
                        predicted_keys.append(predicted_key)
        except (ValueError, IndexError):
            pass

        # If no numerical pattern found, repeat the sequence
        if not predicted_keys:
            predicted_keys = base_keys * 2  # Repeat pattern

        return predicted_keys[: self.config.warming_batch_size]

    async def _predict_temporal(self, pattern: AccessPattern) -> List[str]:
        """Predict keys based on temporal patterns."""
        # For temporal patterns, the same keys are likely to be accessed again
        return pattern.keys

    async def _predict_frequency(self, pattern: AccessPattern) -> List[str]:
        """Predict keys based on frequency patterns."""
        # High frequency keys should continue to be accessed
        return pattern.keys

    async def _background_warming_loop(self) -> None:
        """Background loop for periodic cache warming."""
        while not self._shutdown_event.is_set():
            try:
                # Generate warming tasks from patterns
                tasks_created = await self.warm_from_patterns()

                if tasks_created > 0:
                    logger.debug(f"Background warming created {tasks_created} tasks")

                # Execute pending tasks
                await self._execute_pending_tasks()

                # Wait for next iteration
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.background_interval_seconds,
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal timeout, continue loop

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background warming loop: {e}")
                await asyncio.sleep(30.0)  # Back off on error

    async def _cleanup_loop(self) -> None:
        """Background loop for cleaning up completed tasks."""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.now(UTC)
                old_tasks = []

                # Find tasks older than 1 hour
                for task_id, task in self._warming_tasks.items():
                    if (current_time - task.created_at).seconds > 3600:
                        old_tasks.append(task_id)

                # Remove old tasks
                for task_id in old_tasks:
                    del self._warming_tasks[task_id]
                    self._active_warming.discard(task_id)

                if old_tasks:
                    logger.debug(f"Cleaned up {len(old_tasks)} old warming tasks")

                # Wait for next cleanup
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.cleanup_interval_seconds,
                    )
                    break
                except asyncio.TimeoutError:
                    continue

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60.0)

    async def _execute_pending_tasks(self) -> None:
        """Execute pending warming tasks based on priority."""
        # Get tasks sorted by priority
        pending_tasks = [
            (task_id, task)
            for task_id, task in self._warming_tasks.items()
            if task_id not in self._active_warming
        ]

        # Sort by priority (lower enum value = higher priority)
        pending_tasks.sort(key=lambda x: (x[1].priority.value, x[1].age_seconds))

        # Execute high priority tasks immediately
        execution_tasks = []
        for task_id, task in pending_tasks:
            if len(execution_tasks) >= self.config.max_concurrent_tasks:
                break

            if task.priority in [WarmingPriority.CRITICAL, WarmingPriority.HIGH]:
                execution_tasks.append(self._execute_warming_task(task_id))
            elif len(execution_tasks) < self.config.max_concurrent_tasks // 2:
                # Execute normal/low priority if we have capacity
                execution_tasks.append(self._execute_warming_task(task_id))

        if execution_tasks:
            await asyncio.gather(*execution_tasks, return_exceptions=True)

    async def _execute_warming_task(self, task_id: str) -> bool:
        """Execute a warming task.

        Args:
            task_id: ID of task to execute

        Returns:
            True if successful, False otherwise
        """
        if task_id not in self._warming_tasks:
            logger.warning(f"Warming task {task_id} not found")
            return False

        if task_id in self._active_warming:
            logger.debug(f"Warming task {task_id} already active")
            return False

        async with self._task_semaphore:
            self._active_warming.add(task_id)
            task = self._warming_tasks[task_id]

            try:
                start_time = time.time()

                # Warm cache levels in batches
                keys_warmed = 0
                for i in range(0, len(task.keys), self.config.warming_batch_size):
                    batch_keys = task.keys[i : i + self.config.warming_batch_size]

                    # Warm each cache level
                    for cache_level in task.cache_levels:
                        try:
                            await self._warm_cache_level(batch_keys, cache_level)
                            keys_warmed += len(batch_keys)
                        except Exception as e:
                            logger.warning(
                                f"Failed to warm {cache_level} for task {task_id}: {e}"
                            )

                duration = time.time() - start_time

                # Update statistics
                self.warming_tasks_completed += 1
                self.total_keys_warmed += keys_warmed

                # Record metrics
                if self._warming_counter:
                    self._warming_counter.add(1, {"status": "success"})

                if self._warming_histogram:
                    self._warming_histogram.record(duration)

                logger.debug(
                    f"Completed warming task {task_id}: {keys_warmed} keys in {duration:.3f}s"
                )
                return True

            except Exception as e:
                logger.error(f"Warming task {task_id} failed: {e}")
                self.warming_tasks_failed += 1

                if self._warming_counter:
                    self._warming_counter.add(1, {"status": "error"})

                return False

            finally:
                self._active_warming.discard(task_id)

    async def _warm_cache_level(self, keys: List[str], cache_level: str) -> None:
        """Warm specific cache level with keys.

        Args:
            keys: Keys to warm
            cache_level: Cache level identifier
        """
        # This would integrate with the actual cache manager
        # For now, simulate warming by checking if keys exist

        if not hasattr(self.cache_manager, "get_cache_by_level"):
            logger.warning("Cache manager does not support level-specific access")
            return

        cache = getattr(self.cache_manager, cache_level, None)
        if not cache:
            logger.warning(f"Cache level {cache_level} not available")
            return

        # Batch fetch to warm cache
        for key in keys:
            try:
                # Try to get value from lower-level caches or generate placeholder
                await cache.get(key)
            except Exception as e:
                logger.debug(f"Could not warm key {key} in {cache_level}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache warmer statistics."""
        active_patterns = len([
            p for p in self._detected_patterns.values() if p.is_active
        ])
        total_patterns = len(self._detected_patterns)

        pattern_stats = {}
        for pattern_type in PatternType:
            count = len([
                p
                for p in self._detected_patterns.values()
                if p.pattern_type == pattern_type
            ])
            pattern_stats[pattern_type.value] = count

        return {
            "warmer": {
                "background_tasks_running": not self._shutdown_event.is_set(),
                "max_concurrent_tasks": self.config.max_concurrent_tasks,
            },
            "patterns": {
                "total_detected": self.patterns_detected,
                "active_patterns": active_patterns,
                "total_stored_patterns": total_patterns,
                "by_type": pattern_stats,
            },
            "tasks": {
                "pending_tasks": len(self._warming_tasks) - len(self._active_warming),
                "active_tasks": len(self._active_warming),
                "completed_tasks": self.warming_tasks_completed,
                "failed_tasks": self.warming_tasks_failed,
            },
            "performance": {
                "total_keys_warmed": self.total_keys_warmed,
                "cache_hits_improved": self.cache_hits_improved,
                "success_rate": self.warming_tasks_completed
                / max(1, self.warming_tasks_completed + self.warming_tasks_failed),
            },
            "config": {
                "pattern_detection_enabled": self.config.pattern_detection_enabled,
                "min_pattern_confidence": self.config.min_pattern_confidence,
                "warming_batch_size": self.config.warming_batch_size,
            },
        }

    async def shutdown(self) -> None:
        """Shutdown cache warmer and cleanup resources."""
        logger.info("Shutting down CacheWarmer")

        await self.stop_background_tasks()

        # Cancel any active warming tasks
        active_tasks = list(self._active_warming)
        for task_id in active_tasks:
            logger.debug(f"Cancelling active warming task: {task_id}")

        self._warming_tasks.clear()
        self._active_warming.clear()
        self._detected_patterns.clear()

        logger.info("CacheWarmer shutdown complete")

    def __repr__(self) -> str:
        return (
            f"CacheWarmer(active_patterns={len(self._detected_patterns)}, "
            f"pending_tasks={len(self._warming_tasks) - len(self._active_warming)}, "
            f"success_rate={self.warming_tasks_completed / (max(1, self.warming_tasks_completed + self.warming_tasks_failed)):.2f})"
        )


# Convenience function for creating cache warmers
def create_cache_warmer(
    cache_manager,
    pattern_detection_enabled: bool = True,
    max_concurrent_tasks: int = 5,
    warming_batch_size: int = 20,
    **kwargs,
) -> CacheWarmer:
    """Create a cache warmer with simple configuration."""
    config = CacheWarmerConfig(
        pattern_detection_enabled=pattern_detection_enabled,
        max_concurrent_tasks=max_concurrent_tasks,
        warming_batch_size=warming_batch_size,
        **kwargs,
    )
    return CacheWarmer(cache_manager, config)
