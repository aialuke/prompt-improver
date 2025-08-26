"""Metrics Collection Service.

Focused service for collecting, processing, and managing metrics across
all monitoring components. Extracted from unified_monitoring_manager.py.
"""

import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from typing import Any

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True

    metrics_tracer = trace.get_tracer(__name__ + ".metrics_collector")
    metrics_meter = metrics.get_meter(__name__ + ".metrics_collector")

    metrics_collected_total = metrics_meter.create_counter(
        "unified_metrics_collected_total",
        description="Total metrics collected by type",
        unit="1",
    )

    metric_collection_duration = metrics_meter.create_histogram(
        "metric_collection_duration_seconds",
        description="Time taken to collect metrics",
        unit="s",
    )

except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    metrics_tracer = None
    metrics_meter = None
    metrics_collected_total = None
    metric_collection_duration = None

from prompt_improver.monitoring.unified.types import (
    MetricPoint,
    MetricType,
    MonitoringConfig,
)
from prompt_improver.shared.interfaces.protocols.monitoring import (
    MonitoringRepositoryProtocol,
)

logger = logging.getLogger(__name__)


class MetricsCollectorService:
    """Service for collecting and processing metrics across all components.

    Consolidates metrics collection from:
    - System resources (CPU, memory, disk)
    - Application performance
    - Cache operations
    - Health check results
    - Custom application metrics
    """

    def __init__(
        self,
        config: MonitoringConfig,
        repository: MonitoringRepositoryProtocol | None = None,
    ) -> None:
        self.config = config
        self.repository = repository

        # Metrics storage
        self._custom_metrics: list[MetricPoint] = []
        self._metric_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Performance tracking
        self._collection_stats = {
            "total_collections": 0,
            "successful_collections": 0,
            "failed_collections": 0,
            "last_collection_time": None,
            "avg_collection_duration_ms": 0.0,
        }

        # Metric processors
        self._metric_processors = []

        logger.info("MetricsCollectorService initialized")

    async def collect_all_metrics(self) -> list[MetricPoint]:
        """Collect all available metrics from all sources."""
        start_time = time.time()
        all_metrics = []

        try:
            self._collection_stats["total_collections"] += 1

            if self.config.metrics_collection_enabled:
                # Collect from various sources
                system_metrics = await self._collect_system_metrics()
                app_metrics = await self._collect_application_metrics()
                cache_metrics = await self._collect_cache_metrics()
                health_metrics = await self._collect_health_metrics()

                all_metrics.extend(system_metrics)
                all_metrics.extend(app_metrics)
                all_metrics.extend(cache_metrics)
                all_metrics.extend(health_metrics)

            # Add custom metrics
            all_metrics.extend(self._custom_metrics)

            # Process metrics through registered processors
            processed_metrics = await self._process_metrics(all_metrics)

            # Store metrics
            if self.repository and processed_metrics:
                await self._store_metrics_batch(processed_metrics)

            # Update collection stats
            duration_ms = (time.time() - start_time) * 1000
            self._update_collection_stats(True, duration_ms)

            # Record telemetry
            if OPENTELEMETRY_AVAILABLE and metrics_collected_total:
                metrics_collected_total.add(
                    len(processed_metrics),
                    {"source": "all_metrics", "status": "success"}
                )

            logger.debug(f"Collected {len(processed_metrics)} metrics in {duration_ms:.2f}ms")
            return processed_metrics

        except Exception as e:
            logger.exception(f"Failed to collect metrics: {e}")
            self._update_collection_stats(False, (time.time() - start_time) * 1000)

            if OPENTELEMETRY_AVAILABLE and metrics_collected_total:
                metrics_collected_total.add(
                    0,
                    {"source": "all_metrics", "status": "error"}
                )

            return []

    async def _collect_system_metrics(self) -> list[MetricPoint]:
        """Collect system resource metrics."""
        metrics = []

        try:
            import psutil

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()

            metrics.extend([
                MetricPoint(
                    name="system.cpu.usage_percent",
                    value=cpu_percent,
                    metric_type=MetricType.GAUGE,
                    unit="percent",
                    description="CPU usage percentage",
                    tags={"cpu_count": str(cpu_count)},
                ),
                MetricPoint(
                    name="system.cpu.count",
                    value=cpu_count,
                    metric_type=MetricType.GAUGE,
                    unit="count",
                    description="Number of CPU cores",
                ),
            ])

            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            metrics.extend([
                MetricPoint(
                    name="system.memory.usage_percent",
                    value=memory.percent,
                    metric_type=MetricType.GAUGE,
                    unit="percent",
                    description="Memory usage percentage",
                ),
                MetricPoint(
                    name="system.memory.total_bytes",
                    value=memory.total,
                    metric_type=MetricType.GAUGE,
                    unit="bytes",
                    description="Total memory in bytes",
                ),
                MetricPoint(
                    name="system.memory.available_bytes",
                    value=memory.available,
                    metric_type=MetricType.GAUGE,
                    unit="bytes",
                    description="Available memory in bytes",
                ),
                MetricPoint(
                    name="system.swap.usage_percent",
                    value=swap.percent,
                    metric_type=MetricType.GAUGE,
                    unit="percent",
                    description="Swap usage percentage",
                ),
            ])

            # Disk metrics
            disk = psutil.disk_usage("/")
            metrics.extend([
                MetricPoint(
                    name="system.disk.usage_percent",
                    value=(disk.used / disk.total) * 100,
                    metric_type=MetricType.GAUGE,
                    unit="percent",
                    description="Disk usage percentage",
                ),
                MetricPoint(
                    name="system.disk.total_bytes",
                    value=disk.total,
                    metric_type=MetricType.GAUGE,
                    unit="bytes",
                    description="Total disk space in bytes",
                ),
                MetricPoint(
                    name="system.disk.free_bytes",
                    value=disk.free,
                    metric_type=MetricType.GAUGE,
                    unit="bytes",
                    description="Free disk space in bytes",
                ),
            ])

        except ImportError:
            logger.warning("psutil not available for system metrics collection")
        except Exception as e:
            logger.exception(f"Failed to collect system metrics: {e}")

        return metrics

    async def _collect_application_metrics(self) -> list[MetricPoint]:
        """Collect application-level metrics."""
        metrics = []

        try:
            import os
            import threading

            import psutil

            process = psutil.Process(os.getpid())

            # Process metrics
            memory_info = process.memory_info()
            metrics.extend([
                MetricPoint(
                    name="app.process.cpu_percent",
                    value=process.cpu_percent(),
                    metric_type=MetricType.GAUGE,
                    unit="percent",
                    description="Process CPU usage percentage",
                ),
                MetricPoint(
                    name="app.process.memory_rss_bytes",
                    value=memory_info.rss,
                    metric_type=MetricType.GAUGE,
                    unit="bytes",
                    description="Process RSS memory usage",
                ),
                MetricPoint(
                    name="app.process.memory_vms_bytes",
                    value=memory_info.vms,
                    metric_type=MetricType.GAUGE,
                    unit="bytes",
                    description="Process VMS memory usage",
                ),
                MetricPoint(
                    name="app.process.open_files",
                    value=len(process.open_files()),
                    metric_type=MetricType.GAUGE,
                    unit="count",
                    description="Number of open files",
                ),
                MetricPoint(
                    name="app.process.threads",
                    value=process.num_threads(),
                    metric_type=MetricType.GAUGE,
                    unit="count",
                    description="Number of threads",
                ),
            ])

            # Python-specific metrics
            metrics.extend([
                MetricPoint(
                    name="app.python.thread_count",
                    value=threading.active_count(),
                    metric_type=MetricType.GAUGE,
                    unit="count",
                    description="Active Python thread count",
                ),
            ])

        except ImportError:
            logger.warning("psutil not available for application metrics collection")
        except Exception as e:
            logger.exception(f"Failed to collect application metrics: {e}")

        return metrics

    async def _collect_cache_metrics(self) -> list[MetricPoint]:
        """Collect cache performance metrics."""
        metrics = []

        try:
            # This would integrate with cache monitoring components
            # For now, return placeholder metrics
            cache_stats = self._get_cache_statistics()

            if cache_stats:
                metrics.extend([
                    MetricPoint(
                        name="cache.hit_rate",
                        value=cache_stats.get("hit_rate", 0.0),
                        metric_type=MetricType.GAUGE,
                        unit="ratio",
                        description="Overall cache hit rate",
                    ),
                    MetricPoint(
                        name="cache.operations_total",
                        value=cache_stats.get("total_operations", 0),
                        metric_type=MetricType.COUNTER,
                        unit="count",
                        description="Total cache operations",
                    ),
                    MetricPoint(
                        name="cache.avg_response_time_ms",
                        value=cache_stats.get("avg_response_time_ms", 0.0),
                        metric_type=MetricType.GAUGE,
                        unit="milliseconds",
                        description="Average cache response time",
                    ),
                ])

        except Exception as e:
            logger.exception(f"Failed to collect cache metrics: {e}")

        return metrics

    async def _collect_health_metrics(self) -> list[MetricPoint]:
        """Collect health check metrics."""
        metrics = []

        try:
            # Health metrics would come from health service
            health_stats = self._get_health_statistics()

            if health_stats:
                metrics.extend([
                    MetricPoint(
                        name="health.components_total",
                        value=health_stats.get("total_components", 0),
                        metric_type=MetricType.GAUGE,
                        unit="count",
                        description="Total health check components",
                    ),
                    MetricPoint(
                        name="health.components_healthy",
                        value=health_stats.get("healthy_components", 0),
                        metric_type=MetricType.GAUGE,
                        unit="count",
                        description="Healthy components count",
                    ),
                    MetricPoint(
                        name="health.check_duration_ms",
                        value=health_stats.get("last_check_duration_ms", 0.0),
                        metric_type=MetricType.GAUGE,
                        unit="milliseconds",
                        description="Last health check duration",
                    ),
                ])

        except Exception as e:
            logger.exception(f"Failed to collect health metrics: {e}")

        return metrics

    def record_metric(self, metric: MetricPoint) -> None:
        """Record a custom metric."""
        try:
            self._custom_metrics.append(metric)

            # Store in history for trend analysis
            self._metric_history[metric.name].append({
                "timestamp": time.time(),
                "value": metric.value,
                "tags": metric.tags,
            })

            # Store immediately if configured for real-time
            if self.config.metrics_realtime_enabled and self.repository:
                asyncio.create_task(self._store_metrics_batch([metric]))

            logger.debug(f"Recorded custom metric: {metric.name} = {metric.value}")

        except Exception as e:
            logger.exception(f"Failed to record metric {metric.name}: {e}")

    def record_metric_batch(self, metrics: list[MetricPoint]) -> None:
        """Record multiple metrics at once."""
        for metric in metrics:
            self.record_metric(metric)

    async def _process_metrics(self, metrics: list[MetricPoint]) -> list[MetricPoint]:
        """Process metrics through registered processors."""
        processed_metrics = metrics[:]

        for processor in self._metric_processors:
            try:
                processed_metrics = await processor.process_metrics(processed_metrics)
            except Exception as e:
                logger.exception(f"Metric processor failed: {e}")

        return processed_metrics

    async def _store_metrics_batch(self, metrics: list[MetricPoint]) -> None:
        """Store metrics batch in repository."""
        try:
            if self.repository:
                await self.repository.store_metrics(metrics)
                logger.debug(f"Stored {len(metrics)} metrics in repository")
        except Exception as e:
            logger.exception(f"Failed to store metrics batch: {e}")

    def _update_collection_stats(self, success: bool, duration_ms: float) -> None:
        """Update collection statistics."""
        if success:
            self._collection_stats["successful_collections"] += 1
        else:
            self._collection_stats["failed_collections"] += 1

        self._collection_stats["last_collection_time"] = time.time()

        # Update running average of duration
        current_avg = self._collection_stats["avg_collection_duration_ms"]
        total_collections = self._collection_stats["total_collections"]

        self._collection_stats["avg_collection_duration_ms"] = (
            (current_avg * (total_collections - 1) + duration_ms) / total_collections
        )

    def _get_cache_statistics(self) -> dict[str, Any]:
        """Get cache statistics from cache monitoring components."""
        # This would integrate with actual cache monitoring
        # For now, return empty dict
        return {}

    def _get_health_statistics(self) -> dict[str, Any]:
        """Get health statistics from health monitoring components."""
        # This would integrate with actual health monitoring
        # For now, return empty dict
        return {}

    def get_metric_trends(self, metric_name: str, hours: int = 1) -> dict[str, Any]:
        """Get trend analysis for a specific metric."""
        if metric_name not in self._metric_history:
            return {}

        history = list(self._metric_history[metric_name])
        if not history:
            return {}

        # Filter by time window
        cutoff_time = time.time() - (hours * 3600)
        recent_history = [h for h in history if h["timestamp"] >= cutoff_time]

        if not recent_history:
            return {}

        values = [h["value"] for h in recent_history]

        return {
            "metric_name": metric_name,
            "time_window_hours": hours,
            "data_points": len(values),
            "min_value": min(values),
            "max_value": max(values),
            "avg_value": statistics.mean(values),
            "median_value": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "latest_value": values[-1],
            "trend": "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable",
        }

    def get_collection_stats(self) -> dict[str, Any]:
        """Get metrics collection statistics."""
        return {
            **self._collection_stats,
            "success_rate": (
                self._collection_stats["successful_collections"] /
                max(self._collection_stats["total_collections"], 1)
            ),
            "active_custom_metrics": len(self._custom_metrics),
            "tracked_metrics": len(self._metric_history),
        }

    def add_metric_processor(self, processor) -> None:
        """Add a metric processor."""
        self._metric_processors.append(processor)
        logger.info(f"Added metric processor: {processor.__class__.__name__}")

    def clear_custom_metrics(self) -> None:
        """Clear stored custom metrics."""
        self._custom_metrics.clear()
        logger.debug("Cleared custom metrics")

    async def cleanup_old_metrics(self, retention_hours: int) -> int:
        """Clean up old metric history data."""
        cutoff_time = time.time() - (retention_hours * 3600)
        cleaned_count = 0

        for metric_name, history in self._metric_history.items():
            original_len = len(history)

            # Filter out old entries
            filtered_history = deque([
                h for h in history if h["timestamp"] >= cutoff_time
            ], maxlen=1000)

            cleaned_count += original_len - len(filtered_history)
            self._metric_history[metric_name] = filtered_history

        logger.info(f"Cleaned up {cleaned_count} old metric history entries")
        return cleaned_count
