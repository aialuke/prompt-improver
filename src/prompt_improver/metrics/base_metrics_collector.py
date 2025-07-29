"""
Base MetricsCollector Implementation - 2025 Modern Python Design

Provides a foundational base class for all metrics collectors using modern Python
patterns including Protocol-based interfaces, composition over inheritance,
async-first design, and structured error handling.

Follows 2025 best practices:
- Protocol-based structural subtyping
- Composition over inheritance
- Async-first with proper resource management
- Modern type system with TypeVar bounds
- JSONB-compatible data structures
- Dependency injection patterns
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import (
    Any, Dict, Optional, TypeVar, Generic, Protocol, runtime_checkable,
    Callable, Awaitable, Union, Deque, DefaultDict, TYPE_CHECKING
)

# Modern 2025 approach: TYPE_CHECKING for type hints, lazy imports for runtime
if TYPE_CHECKING:
    from ..performance.monitoring.metrics_registry import MetricsRegistry


# Type definitions for modern Python design
MetricData = TypeVar('MetricData')
ConfigType = TypeVar('ConfigType', bound=Dict[str, Any])
ErrorHandler = Callable[[Exception, str], Awaitable[None]]


@dataclass
class CollectionStats:
    """JSONB-compatible collection statistics structure."""
    total_metrics_collected: int = 0
    collection_errors: int = 0
    last_collection_time: Optional[str] = None
    last_aggregation_time: Optional[str] = None
    is_running: bool = False
    background_tasks_active: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSONB-compatible dictionary."""
        return {
            "total_metrics_collected": self.total_metrics_collected,
            "collection_errors": self.collection_errors,
            "last_collection_time": self.last_collection_time,
            "last_aggregation_time": self.last_aggregation_time,
            "is_running": self.is_running,
            "background_tasks_active": self.background_tasks_active
        }


@dataclass
class MetricsConfig:
    """Configuration for metrics collectors with sensible defaults."""
    aggregation_window_minutes: int = 5
    retention_hours: int = 24
    max_metrics_per_type: int = 10000
    enable_prometheus: bool = True
    enable_background_aggregation: bool = True
    error_threshold: int = 100
    collection_interval_seconds: float = 1.0

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-like access for backward compatibility."""
        return getattr(self, key, default)


@runtime_checkable
class MetricsCollectorProtocol(Protocol[MetricData]):
    """Protocol defining the interface for metrics collectors."""

    def collect_metric(self, metric: MetricData) -> None:
        """Collect a single metric."""
        ...

    async def start_collection(self) -> None:
        """Start background collection processes."""
        ...

    async def stop_collection(self) -> None:
        """Stop background collection processes."""
        ...

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get current collection statistics."""
        ...


class BaseMetricsCollector(Generic[MetricData], ABC):
    """
    Modern base class for metrics collectors using 2025 Python best practices.

    Features:
    - Protocol-based design with composition over inheritance
    - Async-first with proper resource management
    - JSONB-compatible data structures
    - Modern error handling and logging
    - Dependency injection for configuration and registry
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], MetricsConfig]] = None,
        metrics_registry: Optional["MetricsRegistry"] = None,
        error_handler: Optional[ErrorHandler] = None
    ) -> None:
        """Initialize base metrics collector with dependency injection."""
        # Configuration handling with modern patterns
        if isinstance(config, dict):
            self.config = MetricsConfig(**{k: v for k, v in config.items()
                                         if hasattr(MetricsConfig, k)})
        else:
            self.config = config or MetricsConfig()

        # Dependency injection with lazy loading
        self.metrics_registry = metrics_registry or self._get_default_metrics_registry()
        self.error_handler = error_handler or self._default_error_handler

        # Modern logging setup
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Collection statistics with JSONB compatibility
        self.collection_stats = CollectionStats()

        # Metrics storage with composition pattern
        self._metrics_storage: DefaultDict[str, Deque[MetricData]] = defaultdict(
            lambda: deque(maxlen=self.config.max_metrics_per_type)
        )

        # Background task management
        self._background_tasks: Dict[str, Optional[asyncio.Task]] = {}
        self._shutdown_event = asyncio.Event()

        # Initialize Prometheus metrics if enabled
        if self.config.enable_prometheus:
            self._initialize_prometheus_metrics()

    def _get_default_metrics_registry(self) -> "MetricsRegistry":
        """Lazy import and return default metrics registry to avoid circular imports."""
        from ..performance.monitoring.metrics_registry import get_metrics_registry
        return get_metrics_registry()

    @abstractmethod
    def _initialize_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics - must be implemented by subclasses."""
        pass

    @abstractmethod
    def collect_metric(self, metric: MetricData) -> None:
        """Collect a single metric - must be implemented by subclasses."""
        pass

    async def start_collection(self) -> None:
        """Start background collection processes with modern async patterns."""
        if self.collection_stats.is_running:
            self.logger.warning("Collection already running")
            return

        self.collection_stats.is_running = True
        self._shutdown_event.clear()

        # Start background tasks if enabled
        if self.config.enable_background_aggregation:
            self._background_tasks["aggregation"] = asyncio.create_task(
                self._aggregation_loop()
            )
            self.collection_stats.background_tasks_active += 1

        self.logger.info(f"Started {self.__class__.__name__} collection")

    async def stop_collection(self) -> None:
        """Stop background collection with proper cleanup."""
        if not self.collection_stats.is_running:
            return

        self.collection_stats.is_running = False
        self._shutdown_event.set()

        # Cancel and cleanup background tasks
        for task_name, task in self._background_tasks.items():
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                self.collection_stats.background_tasks_active = max(
                    0, self.collection_stats.background_tasks_active - 1
                )

        self._background_tasks.clear()
        self.logger.info(f"Stopped {self.__class__.__name__} collection")

    async def _aggregation_loop(self) -> None:
        """Background aggregation loop with modern error handling."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    await self._aggregate_metrics()
                    self.collection_stats.last_aggregation_time = (
                        datetime.now(timezone.utc).isoformat()
                    )
                except Exception as e:
                    await self.error_handler(e, "aggregation_loop")
                    self.collection_stats.collection_errors += 1

                # Use event-based waiting for responsive shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.aggregation_window_minutes * 60
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal timeout, continue loop

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Fatal error in aggregation loop: {e}")

    @abstractmethod
    async def _aggregate_metrics(self) -> None:
        """Aggregate collected metrics - must be implemented by subclasses."""
        pass

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get current collection statistics in JSONB-compatible format."""
        stats = self.collection_stats.to_dict()
        stats.update({
            "metrics_storage_counts": {
                metric_type: len(storage)
                for metric_type, storage in self._metrics_storage.items()
            },
            "config": {
                "aggregation_window_minutes": self.config.aggregation_window_minutes,
                "retention_hours": self.config.retention_hours,
                "max_metrics_per_type": self.config.max_metrics_per_type
            }
        })
        return stats

    def _update_collection_stats(self, metric_type: str) -> None:
        """Update collection statistics for a metric type."""
        self.collection_stats.total_metrics_collected += 1
        self.collection_stats.last_collection_time = (
            datetime.now(timezone.utc).isoformat()
        )

    async def _default_error_handler(self, error: Exception, context: str) -> None:
        """Default error handler with structured logging."""
        self.logger.error(
            f"Error in {context}: {error}",
            extra={
                "error_type": type(error).__name__,
                "context": context,
                "collector_class": self.__class__.__name__
            }
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_collection()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        await self.stop_collection()


# Utility mixins for common functionality
class PrometheusMetricsMixin:
    """Mixin providing common Prometheus metrics creation patterns."""

    def create_counter(self, name: str, description: str, labels: list[str] = None):
        """Create a Prometheus counter with standard naming."""
        return self.metrics_registry.get_or_create_counter(
            name, description, labels or []
        )

    def create_histogram(self, name: str, description: str, labels: list[str] = None,
                        buckets: list[float] = None):
        """Create a Prometheus histogram with standard buckets."""
        default_buckets = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        return self.metrics_registry.get_or_create_histogram(
            name, description, labels or [], buckets or default_buckets
        )

    def create_gauge(self, name: str, description: str, labels: list[str] = None):
        """Create a Prometheus gauge."""
        return self.metrics_registry.get_or_create_gauge(
            name, description, labels or []
        )


class MetricsStorageMixin:
    """Mixin providing common metrics storage operations."""

    def store_metric(self, metric_type: str, metric: Any) -> None:
        """Store a metric in the appropriate storage."""
        self._metrics_storage[metric_type].append(metric)
        self._update_collection_stats(metric_type)

    def get_recent_metrics(self, metric_type: str, hours: int = 1) -> list[Any]:
        """Get recent metrics of a specific type."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        metrics = self._metrics_storage.get(metric_type, deque())

        return [
            metric for metric in metrics
            if hasattr(metric, 'timestamp') and metric.timestamp >= cutoff_time
        ]

    def clear_old_metrics(self, hours: int = None) -> None:
        """Clear metrics older than specified hours."""
        retention_hours = hours or self.config.retention_hours
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=retention_hours)

        for metric_type, storage in self._metrics_storage.items():
            # Create new deque with only recent metrics
            recent_metrics = deque(
                (metric for metric in storage
                 if hasattr(metric, 'timestamp') and metric.timestamp >= cutoff_time),
                maxlen=storage.maxlen
            )
            self._metrics_storage[metric_type] = recent_metrics


# Factory function for dependency injection
def create_metrics_collector(
    collector_class: type[BaseMetricsCollector],
    config: Optional[Union[Dict[str, Any], MetricsConfig]] = None,
    metrics_registry: Optional["MetricsRegistry"] = None,
    error_handler: Optional[ErrorHandler] = None
) -> BaseMetricsCollector:
    """Factory function for creating metrics collectors with dependency injection."""
    return collector_class(
        config=config,
        metrics_registry=metrics_registry,
        error_handler=error_handler
    )


# Singleton management for global collectors
_global_collectors: Dict[str, BaseMetricsCollector] = {}


def get_or_create_collector(
    collector_class: type[BaseMetricsCollector],
    config: Optional[Union[Dict[str, Any], MetricsConfig]] = None,
    metrics_registry: Optional["MetricsRegistry"] = None,
    error_handler: Optional[ErrorHandler] = None
) -> BaseMetricsCollector:
    """Get or create a singleton collector instance."""
    collector_key = collector_class.__name__

    if collector_key not in _global_collectors:
        _global_collectors[collector_key] = create_metrics_collector(
            collector_class, config, metrics_registry, error_handler
        )

    return _global_collectors[collector_key]
