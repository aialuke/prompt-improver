"""
Centralized metrics utilities to eliminate duplicate metrics registry access patterns.

Consolidates the pattern: self.metrics_registry = get_metrics_registry()
Found in 15+ files across the codebase.
"""

from typing import Optional, Any, Dict
from functools import lru_cache
from .logging_utils import get_logger, LoggerMixin

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_metrics_safely():
    """
    Safely get metrics registry with fallback handling.

    Consolidates the common pattern:
    try:
        self.metrics_registry = get_metrics_registry()
    except Exception:
        # Fallback or None

    Returns:
        Metrics registry instance or None on failure
    """
    try:
        from ...performance.monitoring.metrics_registry import get_metrics_registry
        return get_metrics_registry()
    except Exception as e:
        logger.warning(f"Failed to load metrics registry: {e}")
        return None


class MetricsMixin(LoggerMixin):
    """
    Mixin class to provide consistent metrics registry access pattern.

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
                self.logger.warning("Metrics registry not available, metrics will be disabled")

        return self._metrics_registry

    @property
    def metrics_available(self) -> bool:
        """Check if metrics are available."""
        if self._metrics_available is None:
            # Trigger metrics loading
            _ = self.metrics_registry
        return self._metrics_available

    def record_metric(self, metric_name: str, value: Any, labels: Optional[Dict[str, str]] = None) -> bool:
        """
        Record a metric value safely.

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
            # Try to record metric through registry
            registry = self.metrics_registry
            if hasattr(registry, 'record_metric'):
                registry.record_metric(metric_name, value, labels or {})
                return True
            elif hasattr(registry, metric_name):
                metric = getattr(registry, metric_name)
                if labels:
                    if hasattr(metric, 'labels'):
                        metric.labels(**labels).set(value) if hasattr(metric, 'set') else metric.labels(**labels).observe(value)
                    else:
                        # Fallback for metrics without label support
                        if hasattr(metric, 'set'):
                            metric.set(value)
                        elif hasattr(metric, 'observe'):
                            metric.observe(value)
                        elif hasattr(metric, 'inc'):
                            metric.inc(value)
                else:
                    if hasattr(metric, 'set'):
                        metric.set(value)
                    elif hasattr(metric, 'observe'):
                        metric.observe(value)
                    elif hasattr(metric, 'inc'):
                        metric.inc(value)
                return True
            else:
                self.logger.debug(f"Metric {metric_name} not found in registry")
                return False
        except Exception as e:
            self.logger.error(f"Failed to record metric {metric_name}: {e}")
            return False

    def increment_counter(self, counter_name: str, labels: Optional[Dict[str, str]] = None, amount: float = 1.0) -> bool:
        """
        Increment a counter metric safely.

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
                if labels and hasattr(counter, 'labels'):
                    counter.labels(**labels).inc(amount)
                else:
                    counter.inc(amount)
                return True
            else:
                self.logger.debug(f"Counter {counter_name} not found in registry")
                return False
        except Exception as e:
            self.logger.error(f"Failed to increment counter {counter_name}: {e}")
            return False

    def observe_histogram(self, histogram_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> bool:
        """
        Observe a histogram metric safely.

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
                if labels and hasattr(histogram, 'labels'):
                    histogram.labels(**labels).observe(value)
                else:
                    histogram.observe(value)
                return True
            else:
                self.logger.debug(f"Histogram {histogram_name} not found in registry")
                return False
        except Exception as e:
            self.logger.error(f"Failed to observe histogram {histogram_name}: {e}")
            return False

    def set_gauge(self, gauge_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> bool:
        """
        Set a gauge metric value safely.

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
                if labels and hasattr(gauge, 'labels'):
                    gauge.labels(**labels).set(value)
                else:
                    gauge.set(value)
                return True
            else:
                self.logger.debug(f"Gauge {gauge_name} not found in registry")
                return False
        except Exception as e:
            self.logger.error(f"Failed to set gauge {gauge_name}: {e}")
            return False


def create_metrics_context(prefix: str = ""):
    """
    Create a metrics recording context with optional prefix.

    Args:
        prefix: Optional prefix for all metric names

    Returns:
        Context manager for metrics recording
    """
    class MetricsContext:
        def __init__(self, prefix: str):
            self.prefix = prefix
            self.registry = get_metrics_safely()
            self.available = self.registry is not None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def record(self, name: str, value: Any, labels: Optional[Dict[str, str]] = None):
            """Record a metric with optional prefix."""
            if not self.available:
                return False

            full_name = f"{self.prefix}_{name}" if self.prefix else name
            try:
                # Use MetricsMixin-like logic for recording
                if hasattr(self.registry, full_name):
                    metric = getattr(self.registry, full_name)
                    if labels and hasattr(metric, 'labels'):
                        labeled_metric = metric.labels(**labels)
                        if hasattr(labeled_metric, 'set'):
                            labeled_metric.set(value)
                        elif hasattr(labeled_metric, 'observe'):
                            labeled_metric.observe(value)
                        elif hasattr(labeled_metric, 'inc'):
                            labeled_metric.inc(value)
                    else:
                        if hasattr(metric, 'set'):
                            metric.set(value)
                        elif hasattr(metric, 'observe'):
                            metric.observe(value)
                        elif hasattr(metric, 'inc'):
                            metric.inc(value)
                    return True
                return False
            except Exception as e:
                logger.error(f"Failed to record metric {full_name}: {e}")
                return False

    return MetricsContext(prefix)
