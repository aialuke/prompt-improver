"""Memory optimization utilities for performance-critical analytics components.

This module provides memory-efficient data structures and automatic cleanup
mechanisms to prevent memory leaks and ensure bounded memory usage.
"""

import asyncio
import gc
import logging
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from weakref import WeakSet

import psutil

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics"""

    current_memory_mb: float
    peak_memory_mb: float
    memory_threshold_mb: float
    cleanup_count: int
    total_freed_mb: float
    event_buffer_size: int
    event_buffer_max: int
    indexed_experiments: int
    cached_windows: int
    active_tasks: int


class MemoryOptimizedBuffer:
    """Memory-optimized buffer with automatic cleanup"""

    def __init__(self, maxlen: int = 1000, cleanup_threshold: float = 0.8):
        self.buffer = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.cleanup_threshold = cleanup_threshold
        self.cleanup_count = 0
        self._lock = asyncio.Lock()

    async def append(self, item: Any):
        """Add item to buffer with automatic cleanup"""
        async with self._lock:
            self.buffer.append(item)

            # Trigger cleanup when buffer is near capacity
            if len(self.buffer) >= self.maxlen * self.cleanup_threshold:
                await self._cleanup_old_items()

    async def _cleanup_old_items(self):
        """Remove oldest items to maintain performance"""
        cleanup_count = int(self.maxlen * 0.1)  # Remove 10% of items

        async with self._lock:
            for _ in range(min(cleanup_count, len(self.buffer))):
                if self.buffer:
                    self.buffer.popleft()

            self.cleanup_count += 1
            logger.debug(
                f"Buffer cleanup #{self.cleanup_count}: removed {cleanup_count} old items"
            )

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        with self._lock:
            return iter(list(self.buffer))


class MemoryOptimizedAnalyticsService:
    """Memory-optimized analytics service with efficient data structures"""

    def __init__(
        self,
        memory_threshold_mb: int = 500,
        memory_warning_threshold_mb: int = 300,
        event_buffer_size: int = 1000,
        anomaly_retention_per_experiment: int = 50,
    ):
        # Memory-optimized data structures
        self.event_buffer = MemoryOptimizedBuffer(maxlen=event_buffer_size)
        self.stream_windows_cache = defaultdict(
            lambda: deque(maxlen=100)
        )  # Max 100 windows per experiment
        self.anomaly_index = defaultdict(
            lambda: deque(maxlen=anomaly_retention_per_experiment)
        )

        # Weak references to prevent memory leaks
        self.active_tasks: WeakSet = weakref.WeakSet()
        self.cleanup_tasks = []

        # Memory monitoring configuration
        self.memory_threshold_mb = memory_threshold_mb
        self.memory_warning_threshold_mb = memory_warning_threshold_mb
        self.last_gc_time = time.time()
        self.last_memory_check = time.time()

        # Performance metrics
        self.memory_cleanup_count = 0
        self.memory_freed_total_mb = 0.0
        self.peak_memory_mb = 0.0

        # Monitoring task
        self._monitoring_active = True
        self._monitoring_task = None
        self._start_monitoring()

    def _start_monitoring(self):
        """Start the async monitoring task"""
        try:
            loop = asyncio.get_running_loop()
            self._monitoring_task = loop.create_task(self._memory_monitoring_loop())
        except RuntimeError:
            # No event loop running, will start when async context is available
            pass

    async def add_event_optimized(self, event: Any):
        """Memory-optimized event addition with automatic cleanup"""
        # Ensure monitoring task is running
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._memory_monitoring_loop())

        # Add to buffer (automatic size limiting via MemoryOptimizedBuffer)
        await self.event_buffer.append(event)

        # Update anomaly index for O(1) lookups (with size limits)
        if (
            hasattr(event, "experiment_id")
            and hasattr(event, "anomalies")
            and event.anomalies
        ):
            self.anomaly_index[event.experiment_id].extend(event.anomalies)

        # Periodic memory cleanup (non-blocking)
        await self._periodic_memory_cleanup()

    async def _periodic_memory_cleanup(self):
        """Automatic memory cleanup based on thresholds and time"""
        current_time = time.time()

        # Check memory every 30 seconds
        if current_time - self.last_memory_check > 30:
            await self._check_memory_usage()
            self.last_memory_check = current_time

        # Run full cleanup every 60 seconds
        if current_time - self.last_gc_time > 60:
            await self._memory_cleanup()
            self.last_gc_time = current_time

    async def _check_memory_usage(self):
        """Check current memory usage and trigger cleanup if needed"""
        try:
            process = psutil.process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            # Update peak memory
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)

            if memory_mb > self.memory_threshold_mb:
                logger.warning(
                    f"Memory usage {memory_mb:.1f}MB exceeds threshold {self.memory_threshold_mb}MB"
                )
                await self._memory_cleanup()
            elif memory_mb > self.memory_warning_threshold_mb:
                logger.info(f"Memory usage {memory_mb:.1f}MB approaching threshold")

        except Exception as e:
            logger.error(f"Memory check failed: {e}")

    async def _memory_cleanup(self):
        """Comprehensive memory cleanup with metrics"""
        try:
            # Get memory usage before cleanup
            process = psutil.process()
            memory_before_mb = process.memory_info().rss / 1024 / 1024

            cleanup_actions = 0

            # Clean up old windows (automatically limited by deque maxlen)
            windows_cleaned = 0
            for exp_id in list(self.stream_windows_cache.keys()):
                windows = self.stream_windows_cache[exp_id]
                if len(windows) >= 90:  # Near capacity
                    # Deque will automatically remove old items
                    windows_cleaned += 1

            # Clean up old anomalies (automatically limited by deque maxlen)
            anomalies_cleaned = 0
            for exp_id in list(self.anomaly_index.keys()):
                anomalies = self.anomaly_index[exp_id]
                if len(anomalies) >= 45:  # Near capacity
                    anomalies_cleaned += 1

            # Clean up weak references
            self.active_tasks = weakref.WeakSet()

            # Clean up empty collections
            empty_experiments = [
                exp_id
                for exp_id, windows in self.stream_windows_cache.items()
                if len(windows) == 0
            ]
            for exp_id in empty_experiments:
                del self.stream_windows_cache[exp_id]
                if (
                    exp_id in self.anomaly_index
                    and len(self.anomaly_index[exp_id]) == 0
                ):
                    del self.anomaly_index[exp_id]

            cleanup_actions = (
                windows_cleaned + anomalies_cleaned + len(empty_experiments)
            )

            # Force garbage collection
            collected_objects = gc.collect()

            # Get memory usage after cleanup
            memory_after_mb = process.memory_info().rss / 1024 / 1024
            memory_freed = memory_before_mb - memory_after_mb

            # Update metrics
            self.memory_cleanup_count += 1
            self.memory_freed_total_mb += max(
                0, memory_freed
            )  # Don't count negative frees

            logger.info(
                f"Memory cleanup #{self.memory_cleanup_count}: "
                f"freed {memory_freed:.1f}MB, "
                f"collected {collected_objects} objects, "
                f"cleaned {cleanup_actions} collections, "
                f"current usage: {memory_after_mb:.1f}MB"
            )

        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")

    async def _memory_monitoring_loop(self):
        """Background memory monitoring loop"""
        while self._monitoring_active:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024

                # Update peak memory
                self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)

                # Emergency cleanup if memory is critically high
                if memory_mb > self.memory_threshold_mb * 1.2:  # 20% above threshold
                    logger.critical(
                        f"Emergency memory cleanup triggered: {memory_mb:.1f}MB"
                    )
                    # Run synchronous cleanup in thread to avoid blocking
                    await asyncio.to_thread(self._sync_memory_cleanup)

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Memory monitoring loop error: {e}")
                await asyncio.sleep(30)  # Wait longer on error

    def _sync_memory_cleanup(self):
        """Synchronous memory cleanup for emergency situations"""
        try:
            # Aggressive cleanup
            self.event_buffer.buffer.clear()
            self.stream_windows_cache.clear()

            # Keep only recent anomalies
            for exp_id in list(self.anomaly_index.keys()):
                anomalies = self.anomaly_index[exp_id]
                if len(anomalies) > 10:
                    # Keep only last 10 anomalies
                    recent_anomalies = list(anomalies)[-10:]
                    anomalies.clear()
                    anomalies.extend(recent_anomalies)

            # Force garbage collection
            collected = gc.collect()
            logger.warning(
                f"Emergency cleanup completed: collected {collected} objects"
            )

        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")

    def get_anomalies_optimized(self, experiment_id: str) -> list[dict[str, Any]]:
        """O(1) anomaly retrieval using index with memory efficiency"""
        anomalies_deque = self.anomaly_index.get(experiment_id)
        if not anomalies_deque:
            return []

        # Convert to list efficiently
        # Standardized serialization using duck typing instead of hasattr
        from dataclasses import asdict, is_dataclass

        serialized_anomalies = []
        for a in anomalies_deque:
            try:
                # Try Pydantic model_dump first (most common case)
                serialized_anomalies.append(a.model_dump())
            except AttributeError:
                try:
                    # Try dataclass asdict
                    if is_dataclass(a):
                        serialized_anomalies.append(asdict(a))
                    else:
                        # Assume it's already serializable (dict, basic types)
                        serialized_anomalies.append(a)
                except Exception as e:
                    logger.warning(
                        f"Failed to serialize anomaly of type {type(a).__name__}: {e}"
                    )
                    serialized_anomalies.append({
                        "error": f"Serialization failed: {a!s}"
                    })
        return serialized_anomalies

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics"""
        try:
            process = psutil.process()
            current_memory_mb = process.memory_info().rss / 1024 / 1024

            return MemoryStats(
                current_memory_mb=current_memory_mb,
                peak_memory_mb=self.peak_memory_mb,
                memory_threshold_mb=self.memory_threshold_mb,
                cleanup_count=self.memory_cleanup_count,
                total_freed_mb=self.memory_freed_total_mb,
                event_buffer_size=len(self.event_buffer),
                event_buffer_max=self.event_buffer.maxlen,
                indexed_experiments=len(self.anomaly_index),
                cached_windows=sum(
                    len(windows) for windows in self.stream_windows_cache.values()
                ),
                active_tasks=len(self.active_tasks)
                if hasattr(self.active_tasks, "__len__")
                else 0,
            )
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return MemoryStats(
                current_memory_mb=0.0,
                peak_memory_mb=self.peak_memory_mb,
                memory_threshold_mb=self.memory_threshold_mb,
                cleanup_count=self.memory_cleanup_count,
                total_freed_mb=self.memory_freed_total_mb,
                event_buffer_size=0,
                event_buffer_max=0,
                indexed_experiments=0,
                cached_windows=0,
                active_tasks=0,
            )

    def get_memory_efficiency_report(self) -> dict[str, Any]:
        """Get comprehensive memory efficiency report"""
        stats = self.get_memory_stats()

        # Calculate efficiency metrics
        memory_utilization = stats.current_memory_mb / stats.memory_threshold_mb
        buffer_utilization = (
            stats.event_buffer_size / stats.event_buffer_max
            if stats.event_buffer_max > 0
            else 0
        )
        cleanup_efficiency = (
            stats.total_freed_mb / stats.cleanup_count if stats.cleanup_count > 0 else 0
        )

        return {
            "memory_utilization_percent": memory_utilization * 100,
            "buffer_utilization_percent": buffer_utilization * 100,
            "cleanup_efficiency_mb_per_cleanup": cleanup_efficiency,
            "memory_saved_from_peak_mb": stats.peak_memory_mb - stats.current_memory_mb,
            "total_memory_managed_mb": stats.total_freed_mb + stats.current_memory_mb,
            "performance_status": "excellent"
            if memory_utilization < 0.6
            else "good"
            if memory_utilization < 0.8
            else "warning"
            if memory_utilization < 1.0
            else "critical",
            "recommendations": self._get_memory_recommendations(stats),
        }

    def _get_memory_recommendations(self, stats: MemoryStats) -> list[str]:
        """Get memory optimization recommendations"""
        recommendations = []

        memory_utilization = stats.current_memory_mb / stats.memory_threshold_mb

        if memory_utilization > 0.9:
            recommendations.append(
                "Critical: Memory usage is very high. Consider increasing memory limits or reducing buffer sizes."
            )
        elif memory_utilization > 0.7:
            recommendations.append(
                "Warning: Memory usage is approaching limits. Monitor closely."
            )

        if stats.event_buffer_size >= stats.event_buffer_max * 0.9:
            recommendations.append(
                "Event buffer is near capacity. Consider processing events more frequently."
            )

        if stats.cleanup_count == 0:
            recommendations.append(
                "No memory cleanups have been performed. Increase monitoring frequency if memory usage is high."
            )

        if stats.total_freed_mb < 10 and stats.cleanup_count > 5:
            recommendations.append(
                "Memory cleanup is not freeing much memory. Review data structure efficiency."
            )

        if not recommendations:
            recommendations.append(
                "Memory usage is optimal. Continue current configuration."
            )

        return recommendations

    def shutdown(self):
        """Shutdown memory monitoring and cleanup"""
        self._monitoring_active = False
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()

        # Final cleanup
        self._sync_memory_cleanup()
        logger.info("Memory optimization service shut down")


# Global memory optimizer instance
_memory_optimizer: MemoryOptimizedAnalyticsService | None = None


def get_memory_optimizer(
    memory_threshold_mb: int = 500, event_buffer_size: int = 1000
) -> MemoryOptimizedAnalyticsService:
    """Get singleton memory optimizer instance"""
    global _memory_optimizer

    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizedAnalyticsService(
            memory_threshold_mb=memory_threshold_mb, event_buffer_size=event_buffer_size
        )

    return _memory_optimizer


def shutdown_memory_optimizer():
    """Shutdown the global memory optimizer"""
    global _memory_optimizer

    if _memory_optimizer is not None:
        _memory_optimizer.shutdown()
        _memory_optimizer = None
