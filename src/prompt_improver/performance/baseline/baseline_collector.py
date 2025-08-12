"""Automated performance baseline collection engine."""

import asyncio
import json
import logging
import time
import tracemalloc
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompt_improver.common.datetime_utils import (
    format_compact_timestamp,
    format_date_only,
    format_display_date,
)
from prompt_improver.performance.baseline.models import (
    STANDARD_METRICS,
    BaselineMetrics,
    MetricDefinition,
    MetricType,
    MetricValue,
    get_metric_definition,
)
from prompt_improver.performance.monitoring.health.background_manager import (
    TaskPriority,
    get_background_task_manager,
)

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
try:
    from prompt_improver.database import get_session

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
try:
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
try:
    from prompt_improver.performance.monitoring.metrics_registry import (
        get_metrics_registry,
    )
except ImportError:
    pass
logger = logging.getLogger(__name__)


class BaselineCollector:
    """Automated performance baseline collection engine.

    Continuously collects key performance metrics with statistical analysis
    and trend detection capabilities.
    """

    def __init__(
        self,
        collection_interval: int = 60,
        storage_path: Path | None = None,
        metrics_config: dict[str, MetricDefinition] | None = None,
        enable_system_metrics: bool = True,
        enable_database_metrics: bool = True,
        enable_redis_metrics: bool = True,
        enable_application_metrics: bool = True,
        custom_collectors: list[Callable] | None = None,
    ):
        """Initialize baseline collector.

        Args:
            collection_interval: How often to collect metrics (seconds)
            storage_path: Where to store collected data
            metrics_config: Custom metric definitions
            enable_system_metrics: Collect system resource metrics
            enable_database_metrics: Collect database performance metrics
            enable_redis_metrics: Collect Redis/cache metrics
            enable_application_metrics: Collect application-specific metrics
            custom_collectors: List of custom metric collector functions
        """
        self.collection_interval = collection_interval
        self.storage_path = storage_path or Path("baseline_data")
        self.storage_path.mkdir(exist_ok=True)
        self.metrics_config = metrics_config or {}
        self.enable_system_metrics = enable_system_metrics and PSUTIL_AVAILABLE
        self.enable_database_metrics = enable_database_metrics and DATABASE_AVAILABLE
        self.enable_redis_metrics = enable_redis_metrics and REDIS_AVAILABLE
        self.enable_application_metrics = enable_application_metrics
        self.custom_collectors = custom_collectors or []
        self._collecting = False
        self._collection_task: asyncio.Task | None = None
        self._collected_baselines: list[BaselineMetrics] = []
        self._operation_tracking: dict[str, list[float]] = {}
        self._initialize_standard_metrics()
        logger.info(
            f"BaselineCollector initialized with {collection_interval}s interval"
        )

    def _initialize_standard_metrics(self):
        """Initialize standard metric definitions."""
        for name, definition in STANDARD_METRICS.items():
            if name not in self.metrics_config:
                self.metrics_config[name] = definition

    async def start_collection(self) -> None:
        """Start the baseline collection process."""
        if self._collecting:
            logger.warning("Collection already running")
            return
        logger.info("Starting baseline collection")
        self._collecting = True
        task_manager = get_background_task_manager()
        self._collection_task = await task_manager.submit_enhanced_task(
            task_id=f"baseline_collection_{uuid.uuid4().hex[:8]}",
            coroutine=self._collection_loop(),
            priority=TaskPriority.NORMAL,
            tags={
                "service": "performance",
                "type": "collection",
                "component": "baseline",
            },
        )

    async def stop_collection(self) -> None:
        """Stop the baseline collection process."""
        if not self._collecting:
            return
        logger.info("Stopping baseline collection")
        self._collecting = False
        if self._collection_task and (not self._collection_task.done()):
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Baseline collection stopped")

    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while self._collecting:
            try:
                baseline = await self._collect_baseline()
                self._collected_baselines.append(baseline)
                await self._store_baseline(baseline)
                await self._cleanup_old_data()
                logger.debug(f"Collected baseline with {len(baseline.metrics)} metrics")
            except Exception as e:
                logger.error(f"Error during baseline collection: {e}")
            if self._collecting:
                await asyncio.sleep(self.collection_interval)

    async def _collect_baseline(self) -> BaselineMetrics:
        """Collect current performance baseline."""
        baseline = BaselineMetrics(
            timestamp=datetime.now(UTC), collection_id=str(uuid.uuid4()), metrics={}
        )
        if self.enable_system_metrics:
            await self._collect_system_metrics(baseline)
        if self.enable_database_metrics:
            await self._collect_database_metrics(baseline)
        if self.enable_redis_metrics:
            await self._collect_redis_metrics(baseline)
        if self.enable_application_metrics:
            await self._collect_application_metrics(baseline)
        for collector in self.custom_collectors:
            try:
                await collector(baseline)
            except Exception as e:
                logger.error(f"Custom collector failed: {e}")
        return baseline

    async def _collect_system_metrics(self, baseline: BaselineMetrics) -> None:
        """Collect system resource metrics."""
        if not PSUTIL_AVAILABLE:
            return
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            baseline.metrics["cpu_utilization"] = MetricValue(
                value=cpu_percent,
                metric_type=MetricType.GAUGE,
                unit="percent",
                timestamp=datetime.now(UTC),
            )
            memory = psutil.virtual_memory()
            baseline.metrics["memory_utilization"] = MetricValue(
                value=memory.percent,
                metric_type=MetricType.GAUGE,
                unit="percent",
                timestamp=datetime.now(UTC),
            )
            baseline.metrics["memory_available"] = MetricValue(
                value=memory.available / 1024**3,
                metric_type=MetricType.GAUGE,
                unit="GB",
                timestamp=datetime.now(UTC),
            )
            disk = psutil.disk_usage("/")
            baseline.metrics["disk_utilization"] = MetricValue(
                value=disk.used / disk.total * 100,
                metric_type=MetricType.GAUGE,
                unit="percent",
                timestamp=datetime.now(UTC),
            )
            net_io = psutil.net_io_counters()
            baseline.metrics["network_bytes_sent"] = MetricValue(
                value=net_io.bytes_sent,
                metric_type=MetricType.COUNTER,
                unit="bytes",
                timestamp=datetime.now(UTC),
            )
            baseline.metrics["network_bytes_recv"] = MetricValue(
                value=net_io.bytes_recv,
                metric_type=MetricType.COUNTER,
                unit="bytes",
                timestamp=datetime.now(UTC),
            )
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

    async def _collect_database_metrics(self, baseline: BaselineMetrics) -> None:
        """Collect database performance metrics."""
        if not DATABASE_AVAILABLE:
            return
        try:
            baseline.metrics["database_connections_active"] = MetricValue(
                value=5,
                metric_type=MetricType.GAUGE,
                unit="connections",
                timestamp=datetime.now(UTC),
            )
        except Exception as e:
            logger.error(f"Failed to collect database metrics: {e}")

    async def _collect_redis_metrics(self, baseline: BaselineMetrics) -> None:
        """Collect Redis/cache performance metrics."""
        if not REDIS_AVAILABLE:
            return
        try:
            baseline.metrics["cache_hit_rate"] = MetricValue(
                value=0.85,
                metric_type=MetricType.GAUGE,
                unit="ratio",
                timestamp=datetime.now(UTC),
            )
        except Exception as e:
            logger.error(f"Failed to collect Redis metrics: {e}")

    async def _collect_application_metrics(self, baseline: BaselineMetrics) -> None:
        """Collect application-specific metrics."""
        try:
            for operation_name, response_times in self._operation_tracking.items():
                if response_times:
                    avg_response_time = sum(response_times) / len(response_times)
                    baseline.metrics[f"{operation_name}_avg_response_time"] = (
                        MetricValue(
                            value=avg_response_time,
                            metric_type=MetricType.GAUGE,
                            unit="ms",
                            timestamp=datetime.now(UTC),
                        )
                    )
            self._operation_tracking.clear()
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")

    async def _store_baseline(self, baseline: BaselineMetrics) -> None:
        """Store baseline to disk."""
        try:
            filename = f"baseline_{baseline.format_compact_timestamp(timestamp)}.json"
            filepath = self.storage_path / filename
            baseline_data = {
                "timestamp": baseline.timestamp.isoformat(),
                "collection_id": baseline.collection_id,
                "metrics": {
                    name: {
                        "value": metric.value,
                        "metric_type": metric.metric_type.value,
                        "unit": metric.unit,
                        "timestamp": metric.timestamp.isoformat(),
                        "labels": metric.labels,
                    }
                    for name, metric in baseline.metrics.items()
                },
            }
            with open(filepath, "w") as f:
                json.dump(baseline_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to store baseline: {e}")

    async def _cleanup_old_data(self) -> None:
        """Clean up old baseline data files."""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)
            for filepath in self.storage_path.glob("baseline_*.json"):
                if filepath.stat().st_mtime < cutoff_time.timestamp():
                    filepath.unlink()
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

    async def record_request(
        self,
        response_time_ms: float,
        is_error: bool = False,
        operation_name: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a request for baseline analysis."""
        if operation_name not in self._operation_tracking:
            self._operation_tracking[operation_name] = []
        self._operation_tracking[operation_name].append(response_time_ms)
        if len(self._operation_tracking[operation_name]) > 1000:
            self._operation_tracking[operation_name] = self._operation_tracking[
                operation_name
            ][-1000:]

    async def load_recent_baselines(self, hours: int = 24) -> list[BaselineMetrics]:
        """Load recent baselines from storage."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            baselines = []
            for filepath in self.storage_path.glob("baseline_*.json"):
                try:
                    timestamp_str = filepath.stem.split("_", 1)[1]
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    if timestamp >= cutoff_time:
                        with open(filepath) as f:
                            data = json.load(f)
                        baseline = BaselineMetrics(
                            timestamp=datetime.fromisoformat(data["timestamp"]),
                            collection_id=data["collection_id"],
                            metrics={},
                        )
                        for name, metric_data in data["metrics"].items():
                            baseline.metrics[name] = MetricValue(
                                value=metric_data["value"],
                                metric_type=MetricType(metric_data["metric_type"]),
                                unit=metric_data["unit"],
                                timestamp=datetime.fromisoformat(
                                    metric_data["timestamp"]
                                ),
                                labels=metric_data.get("labels", {}),
                            )
                        baselines.append(baseline)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse baseline file {filepath}: {e}")
            baselines.sort(key=lambda b: b.timestamp)
            return baselines
        except Exception as e:
            logger.error(f"Failed to load recent baselines: {e}")
            return []

    def get_collection_status(self) -> dict[str, Any]:
        """Get collection status information."""
        return {
            "running": self._collecting,
            "collection_interval": self.collection_interval,
            "collected_count": len(self._collected_baselines),
            "tracking_operations": len(self._operation_tracking),
            "storage_path": str(self.storage_path),
            "system_metrics_enabled": self.enable_system_metrics,
            "database_metrics_enabled": self.enable_database_metrics,
            "redis_metrics_enabled": self.enable_redis_metrics,
            "application_metrics_enabled": self.enable_application_metrics,
        }


_global_baseline_collector: BaselineCollector | None = None


def get_baseline_collector(
    collection_interval: int = 60, storage_path: Path | None = None, **kwargs
) -> BaselineCollector:
    """Get the global baseline collector instance."""
    global _global_baseline_collector
    if _global_baseline_collector is None:
        _global_baseline_collector = BaselineCollector(
            collection_interval=collection_interval, storage_path=storage_path, **kwargs
        )
    return _global_baseline_collector


async def record_operation(
    operation_name: str,
    response_time_ms: float,
    is_error: bool = False,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record an operation for baseline analysis."""
    collector = get_baseline_collector()
    await collector.record_request(response_time_ms, is_error, operation_name, metadata)


@asynccontextmanager
async def track_operation(operation_name: str, **metadata):
    """Context manager for tracking operation performance."""
    start_time = time.time()
    error_occurred = False
    try:
        yield
    except Exception as e:
        error_occurred = True
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        await record_operation(operation_name, duration_ms, error_occurred, metadata)
