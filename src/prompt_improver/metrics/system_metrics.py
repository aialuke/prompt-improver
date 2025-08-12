"""Comprehensive System Metrics Implementation - Phase 1 Missing Metrics

Provides real-time tracking of:
1. Connection Age Tracking - Real connection lifecycle monitoring
2. Memory Usage Patterns - Detailed memory consumption analysis
3. Cache Hit Ratios - Performance optimization insights
4. Request Processing Times - End-to-end latency tracking
5. Error Rate Analysis - System reliability monitoring

This implementation uses OpenTelemetry for modern observability patterns.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import psutil
from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram, UpDownCounter

logger = logging.getLogger(__name__)


@dataclass
class SystemMetricsConfig:
    """Configuration for system metrics collection."""

    collection_interval: float = 30.0
    memory_threshold_mb: int = 1000
    cpu_threshold_percent: float = 80.0
    disk_threshold_percent: float = 85.0
    enable_detailed_metrics: bool = True


@dataclass
class ConnectionMetrics:
    """Connection lifecycle metrics."""

    active_connections: int = 0
    total_connections: int = 0
    connection_errors: int = 0
    avg_connection_age: float = 0.0
    connection_pool_size: int = 0
    connection_pool_available: int = 0


@dataclass
class MemoryMetrics:
    """Memory usage pattern metrics."""

    total_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    available_memory_mb: float = 0.0
    memory_percent: float = 0.0
    swap_used_mb: float = 0.0
    swap_percent: float = 0.0


@dataclass
class CacheMetrics:
    """Cache performance metrics."""

    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_ratio: float = 0.0
    cache_size: int = 0
    cache_evictions: int = 0


@dataclass
class RequestMetrics:
    """Request processing metrics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0


@dataclass
class SystemHealth:
    """Overall system health indicators."""

    cpu_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_mb: float = 0.0
    disk_io_mb: float = 0.0
    load_average: float = 0.0
    uptime_seconds: float = 0.0


class SystemMetricsCollector:
    """Comprehensive system metrics collector using OpenTelemetry.

    Provides real-time monitoring of system resources, connection health,
    cache performance, and request processing metrics.
    """

    def __init__(self, config: SystemMetricsConfig | None = None):
        self.config = config or SystemMetricsConfig()
        self._meter = metrics.get_meter(__name__)
        self._start_time = time.time()
        self._is_collecting = False
        self._collection_task: asyncio.Task | None = None
        self._init_metrics()
        self.connection_metrics = ConnectionMetrics()
        self.memory_metrics = MemoryMetrics()
        self.cache_metrics = CacheMetrics()
        self.request_metrics = RequestMetrics()
        self.system_health = SystemHealth()
        logger.info("SystemMetricsCollector initialized with OpenTelemetry")

    def _init_metrics(self) -> None:
        """Initialize OpenTelemetry metrics instruments."""
        self.connection_counter = self._meter.create_counter(
            name="system_connections_total",
            description="Total number of connections",
            unit="1",
        )
        self.connection_age_histogram = self._meter.create_histogram(
            name="system_connection_age_seconds",
            description="Connection age distribution",
            unit="s",
        )
        self.memory_usage_gauge = self._meter.create_up_down_counter(
            name="system_memory_usage_bytes",
            description="Current memory usage",
            unit="By",
        )
        self.cache_hit_counter = self._meter.create_counter(
            name="system_cache_hits_total", description="Total cache hits", unit="1"
        )
        self.cache_miss_counter = self._meter.create_counter(
            name="system_cache_misses_total", description="Total cache misses", unit="1"
        )
        self.request_duration_histogram = self._meter.create_histogram(
            name="system_request_duration_seconds",
            description="Request processing time distribution",
            unit="s",
        )
        self.cpu_usage_gauge = self._meter.create_up_down_counter(
            name="system_cpu_usage_percent",
            description="Current CPU usage percentage",
            unit="%",
        )
        self.disk_usage_gauge = self._meter.create_up_down_counter(
            name="system_disk_usage_percent",
            description="Current disk usage percentage",
            unit="%",
        )

    async def start_collection(self) -> None:
        """Start continuous metrics collection."""
        if self._is_collecting:
            logger.warning("Metrics collection already started")
            return
        self._is_collecting = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started system metrics collection")

    async def stop_collection(self) -> None:
        """Stop metrics collection."""
        if not self._is_collecting:
            return
        self._is_collecting = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped system metrics collection")

    async def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        while self._is_collecting:
            try:
                await self._collect_all_metrics()
                await asyncio.sleep(self.config.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.config.collection_interval)

    async def _collect_all_metrics(self) -> None:
        """Collect all system metrics."""
        await asyncio.gather(
            self._collect_memory_metrics(),
            self._collect_system_health_metrics(),
            self._update_opentelemetry_metrics(),
            return_exceptions=True,
        )

    async def _collect_memory_metrics(self) -> None:
        """Collect memory usage metrics."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            self.memory_metrics.total_memory_mb = memory.total / (1024 * 1024)
            self.memory_metrics.used_memory_mb = memory.used / (1024 * 1024)
            self.memory_metrics.available_memory_mb = memory.available / (1024 * 1024)
            self.memory_metrics.memory_percent = memory.percent
            self.memory_metrics.swap_used_mb = swap.used / (1024 * 1024)
            self.memory_metrics.swap_percent = swap.percent
        except Exception as e:
            logger.error(f"Error collecting memory metrics: {e}")

    async def _collect_system_health_metrics(self) -> None:
        """Collect system health indicators."""
        try:
            self.system_health.cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage("/")
            self.system_health.disk_usage_percent = disk.used / disk.total * 100
            try:
                load_avg = psutil.getloadavg()
                self.system_health.load_average = load_avg[0]
            except AttributeError:
                self.system_health.load_average = 0.0
            self.system_health.uptime_seconds = time.time() - self._start_time
        except Exception as e:
            logger.error(f"Error collecting system health metrics: {e}")

    async def _update_opentelemetry_metrics(self) -> None:
        """Update OpenTelemetry metrics with current values."""
        try:
            current_memory_bytes = self.memory_metrics.used_memory_mb * 1024 * 1024
            self.memory_usage_gauge.add(current_memory_bytes, {"type": "used"})
            self.cpu_usage_gauge.add(
                self.system_health.cpu_percent, {"resource": "cpu"}
            )
            self.disk_usage_gauge.add(
                self.system_health.disk_usage_percent, {"resource": "disk"}
            )
        except Exception as e:
            logger.error(f"Error updating OpenTelemetry metrics: {e}")

    def record_connection_event(
        self, event_type: str, connection_age: float | None = None
    ) -> None:
        """Record connection-related events."""
        try:
            self.connection_counter.add(1, {"event_type": event_type})
            if connection_age is not None:
                self.connection_age_histogram.record(connection_age)
            if event_type == "created":
                self.connection_metrics.total_connections += 1
                self.connection_metrics.active_connections += 1
            elif event_type == "closed":
                self.connection_metrics.active_connections = max(
                    0, self.connection_metrics.active_connections - 1
                )
            elif event_type == "error":
                self.connection_metrics.connection_errors += 1
        except Exception as e:
            logger.error(f"Error recording connection event: {e}")

    def record_cache_event(self, event_type: str) -> None:
        """Record cache-related events."""
        try:
            if event_type == "hit":
                self.cache_hit_counter.add(1)
                self.cache_metrics.cache_hits += 1
            elif event_type == "miss":
                self.cache_miss_counter.add(1)
                self.cache_metrics.cache_misses += 1
            elif event_type == "eviction":
                self.cache_metrics.cache_evictions += 1
            total_requests = (
                self.cache_metrics.cache_hits + self.cache_metrics.cache_misses
            )
            if total_requests > 0:
                self.cache_metrics.cache_hit_ratio = (
                    self.cache_metrics.cache_hits / total_requests
                )
        except Exception as e:
            logger.error(f"Error recording cache event: {e}")

    def record_request_duration(
        self, duration_seconds: float, status: str = "success"
    ) -> None:
        """Record request processing duration."""
        try:
            self.request_duration_histogram.record(duration_seconds, {"status": status})
            self.request_metrics.total_requests += 1
            if status == "success":
                self.request_metrics.successful_requests += 1
            else:
                self.request_metrics.failed_requests += 1
        except Exception as e:
            logger.error(f"Error recording request duration: {e}")

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get a summary of all collected metrics."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "collection_interval": self.config.collection_interval,
            "connection_metrics": {
                "active_connections": self.connection_metrics.active_connections,
                "total_connections": self.connection_metrics.total_connections,
                "connection_errors": self.connection_metrics.connection_errors,
                "avg_connection_age": self.connection_metrics.avg_connection_age,
            },
            "memory_metrics": {
                "used_memory_mb": round(self.memory_metrics.used_memory_mb, 2),
                "memory_percent": round(self.memory_metrics.memory_percent, 2),
                "swap_used_mb": round(self.memory_metrics.swap_used_mb, 2),
                "swap_percent": round(self.memory_metrics.swap_percent, 2),
            },
            "cache_metrics": {
                "cache_hits": self.cache_metrics.cache_hits,
                "cache_misses": self.cache_metrics.cache_misses,
                "cache_hit_ratio": round(self.cache_metrics.cache_hit_ratio, 4),
                "cache_evictions": self.cache_metrics.cache_evictions,
            },
            "request_metrics": {
                "total_requests": self.request_metrics.total_requests,
                "successful_requests": self.request_metrics.successful_requests,
                "failed_requests": self.request_metrics.failed_requests,
                "success_rate": self.request_metrics.successful_requests
                / max(1, self.request_metrics.total_requests),
            },
            "system_health": {
                "cpu_percent": round(self.system_health.cpu_percent, 2),
                "disk_usage_percent": round(self.system_health.disk_usage_percent, 2),
                "load_average": round(self.system_health.load_average, 2),
                "uptime_hours": round(self.system_health.uptime_seconds / 3600, 2),
            },
        }

    def check_health_thresholds(self) -> dict[str, bool]:
        """Check if system metrics exceed configured thresholds."""
        return {
            "memory_healthy": self.memory_metrics.memory_percent
            < self.config.memory_threshold_mb,
            "cpu_healthy": self.system_health.cpu_percent
            < self.config.cpu_threshold_percent,
            "disk_healthy": self.system_health.disk_usage_percent
            < self.config.disk_threshold_percent,
            "overall_healthy": all([
                self.memory_metrics.memory_percent < self.config.memory_threshold_mb,
                self.system_health.cpu_percent < self.config.cpu_threshold_percent,
                self.system_health.disk_usage_percent
                < self.config.disk_threshold_percent,
            ]),
        }


_metrics_collector: SystemMetricsCollector | None = None


def get_metrics_collector() -> SystemMetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = SystemMetricsCollector()
    return _metrics_collector


async def initialize_metrics_collection(
    config: SystemMetricsConfig | None = None,
) -> None:
    """Initialize and start metrics collection."""
    collector = get_metrics_collector()
    if config:
        collector.config = config
    await collector.start_collection()


async def shutdown_metrics_collection() -> None:
    """Shutdown metrics collection."""
    global _metrics_collector
    if _metrics_collector:
        await _metrics_collector.stop_collection()
        _metrics_collector = None
