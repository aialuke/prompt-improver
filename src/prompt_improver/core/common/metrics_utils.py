"""Centralized metrics utilities to eliminate duplicate metrics registry access patterns.

Consolidates the pattern: self.metrics_registry = get_metrics_registry()
Found in 15+ files across the codebase.
"""

import asyncio
from typing import Any

from prompt_improver.core.common.logging_utils import LoggerMixin, get_logger
from prompt_improver.services.cache.cache_facade import CacheFacade
from prompt_improver.services.cache.cache_factory import CacheFactory

logger = get_logger(__name__)


def get_metrics_cache():
    """Get optimized metrics cache using singleton factory pattern.

    Resolves performance issues by using CacheFactory singleton
    instead of creating new instances per call.
    """
    return CacheFactory.get_utility_cache()


def get_metrics_safely():
    """Safely get metrics registry with fallback handling using unified cache.

    Consolidates the common pattern:
    try:
        self.metrics_registry = get_metrics_registry()
    except Exception:
        # Fallback or None

    Returns:
        Metrics registry instance or None on failure
    """
    cache_key = "util:metrics:registry"
    cache = get_metrics_cache()

    # Try to run in existing event loop or create new one
    try:
        loop = asyncio.get_running_loop()
        # Create task for async cache operation
        task = asyncio.create_task(_get_metrics_cached(cache_key, cache))
        # Run in current loop context with shorter timeout
        return asyncio.run_coroutine_threadsafe(task, loop).result(timeout=1.0)
    except RuntimeError:
        # No event loop, create one with timeout
        try:
            async def run_with_timeout():
                return await asyncio.wait_for(
                    _get_metrics_cached(cache_key, cache),
                    timeout=1.0
                )
            return asyncio.run(run_with_timeout())
        except Exception as e:
            logger.warning(f"Async cache operation failed: {e}")
            return _load_metrics_direct()
    except Exception as e:
        logger.warning(f"Cache operation failed: {e}")
        # Fallback to direct execution
        return _load_metrics_direct()


async def _get_metrics_cached(cache_key: str, cache: CacheFacade) -> Any | None:
    """Get metrics registry from cache or load if not cached."""
    try:
        # Try cache first
        cached_result = await cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Cache miss - load metrics and cache result
        result = _load_metrics_direct()

        # Only cache non-None results
        if result is not None:
            # Cache for 4 hours with shorter L1 TTL for performance
            await cache.set(cache_key, result, l2_ttl=14400, l1_ttl=3600)

        return result
    except Exception as e:
        logger.exception(f"Metrics cache operation failed: {e}")
        return _load_metrics_direct()


def _load_metrics_direct() -> Any | None:
    """Load metrics registry directly without caching."""
    try:
        from prompt_improver.performance.monitoring.metrics_registry import (
            get_metrics_registry,
        )

        return get_metrics_registry()
    except Exception as e:
        logger.warning(f"Failed to load metrics registry: {e}")
        return None


class MetricsMixin(LoggerMixin):
    """Mixin class to provide consistent metrics registry access pattern.

    Eliminates duplicate metrics initialization logic in classes.
    Inherits from LoggerMixin to ensure logger access for metrics logging.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._metrics_registry: Any = None
        self._metrics_available: bool | None = None

    @property
    def metrics_registry(self):
        """Get metrics registry with caching and fallback handling."""
        if self._metrics_registry is None:
            self._metrics_registry = get_metrics_safely()
            self._metrics_available = self._metrics_registry is not None
            if not self._metrics_available:
                self.logger.warning(
                    "Metrics registry not available, metrics will be disabled"
                )
        return self._metrics_registry

    @property
    def metrics_available(self) -> bool:
        """Check if metrics are available."""
        if self._metrics_available is None:
            _ = self.metrics_registry
        return self._metrics_available

    def record_metric(
        self, metric_name: str, value: Any, labels: dict[str, str] | None = None
    ) -> bool:
        """Record a metric value safely.

        Args:
            metric_name: Name of the metric
            value: Metric value to record
            labels: Optional labels for the metric

        Returns:
            True if metric was recorded, False if metrics unavailable
        """
        if not self.metrics_available:
            return False
        try:
            registry = self.metrics_registry
            if hasattr(registry, "record_metric"):
                registry.record_metric(metric_name, value, labels or {})
                return True
            if hasattr(registry, metric_name):
                metric = getattr(registry, metric_name)
                if labels:
                    if hasattr(metric, "labels"):
                        metric.labels(**labels).set(value) if hasattr(
                            metric, "set"
                        ) else metric.labels(**labels).observe(value)
                    elif hasattr(metric, "set"):
                        metric.set(value)
                    elif hasattr(metric, "observe"):
                        metric.observe(value)
                    elif hasattr(metric, "inc"):
                        metric.inc(value)
                elif hasattr(metric, "set"):
                    metric.set(value)
                elif hasattr(metric, "observe"):
                    metric.observe(value)
                elif hasattr(metric, "inc"):
                    metric.inc(value)
                return True
            self.logger.debug(f"Metric {metric_name} not found in registry")
            return False
        except Exception as e:
            self.logger.exception(f"Failed to record metric {metric_name}: {e}")
            return False

    def increment_counter(
        self,
        counter_name: str,
        labels: dict[str, str] | None = None,
        amount: float = 1.0,
    ) -> bool:
        """Increment a counter metric safely.

        Args:
            counter_name: Name of the counter metric
            labels: Optional labels for the metric
            amount: Amount to increment by

        Returns:
            True if counter was incremented, False if metrics unavailable
        """
        if not self.metrics_available:
            return False
        try:
            registry = self.metrics_registry
            if hasattr(registry, counter_name):
                counter = getattr(registry, counter_name)
                if labels and hasattr(counter, "labels"):
                    counter.labels(**labels).inc(amount)
                else:
                    counter.inc(amount)
                return True
            self.logger.debug(f"Counter {counter_name} not found in registry")
            return False
        except Exception as e:
            self.logger.exception(f"Failed to increment counter {counter_name}: {e}")
            return False

    def observe_histogram(
        self, histogram_name: str, value: float, labels: dict[str, str] | None = None
    ) -> bool:
        """Observe a histogram metric safely.

        Args:
            histogram_name: Name of the histogram metric
            value: Value to observe
            labels: Optional labels for the metric

        Returns:
            True if value was observed, False if metrics unavailable
        """
        if not self.metrics_available:
            return False
        try:
            registry = self.metrics_registry
            if hasattr(registry, histogram_name):
                histogram = getattr(registry, histogram_name)
                if labels and hasattr(histogram, "labels"):
                    histogram.labels(**labels).observe(value)
                else:
                    histogram.observe(value)
                return True
            self.logger.debug(f"Histogram {histogram_name} not found in registry")
            return False
        except Exception as e:
            self.logger.exception(f"Failed to observe histogram {histogram_name}: {e}")
            return False

    def set_gauge(
        self, gauge_name: str, value: float, labels: dict[str, str] | None = None
    ) -> bool:
        """Set a gauge metric value safely.

        Args:
            gauge_name: Name of the gauge metric
            value: Value to set
            labels: Optional labels for the metric

        Returns:
            True if gauge was set, False if metrics unavailable
        """
        if not self.metrics_available:
            return False
        try:
            registry = self.metrics_registry
            if hasattr(registry, gauge_name):
                gauge = getattr(registry, gauge_name)
                if labels and hasattr(gauge, "labels"):
                    gauge.labels(**labels).set(value)
                else:
                    gauge.set(value)
                return True
            self.logger.debug(f"Gauge {gauge_name} not found in registry")
            return False
        except Exception as e:
            self.logger.exception(f"Failed to set gauge {gauge_name}: {e}")
            return False


def create_metrics_context(prefix: str = ""):
    """Create a metrics recording context with optional prefix.

    Args:
        prefix: Optional prefix for all metric names

    Returns:
        Context manager for metrics recording
    """

    class MetricsContext:
        def __init__(self, prefix: str) -> None:
            self.prefix = prefix
            self.registry = get_metrics_safely()
            self.available = self.registry is not None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def record(self, name: str, value: Any, labels: dict[str, str] | None = None) -> bool | None:
            """Record a metric with optional prefix."""
            if not self.available:
                return False
            full_name = f"{self.prefix}_{name}" if self.prefix else name
            try:
                if hasattr(self.registry, full_name):
                    metric = getattr(self.registry, full_name)
                    if labels and hasattr(metric, "labels"):
                        labeled_metric = metric.labels(**labels)
                        if hasattr(labeled_metric, "set"):
                            labeled_metric.set(value)
                        elif hasattr(labeled_metric, "observe"):
                            labeled_metric.observe(value)
                        elif hasattr(labeled_metric, "inc"):
                            labeled_metric.inc(value)
                    elif hasattr(metric, "set"):
                        metric.set(value)
                    elif hasattr(metric, "observe"):
                        metric.observe(value)
                    elif hasattr(metric, "inc"):
                        metric.inc(value)
                    return True
                return False
            except Exception as e:
                logger.exception(f"Failed to record metric {full_name}: {e}")
                return False

    return MetricsContext(prefix)
