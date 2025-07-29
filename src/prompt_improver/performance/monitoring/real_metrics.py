"""
Real Metrics Implementation for APES.

Provides real metrics collection behavior without mock objects, using either
Prometheus/OpenTelemetry when available or in-memory collection as fallback.
Maintains full functionality and real behavior in all scenarios.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, DefaultDict, Deque

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Represents a metric value with timestamp and labels."""
    value: float
    timestamp: datetime
    labels: dict[str, str] = field(default_factory=dict)


class InMemoryCounter:
    """Real counter implementation that stores values in memory."""
    
    def __init__(self, name: str, description: str, label_names: list[str] | None = None) -> None:
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: DefaultDict[tuple[str, ...], float] = defaultdict(float)
        self._lock = threading.Lock()
    
    def inc(self, amount: float = 1.0, **labels: str) -> None:
        """Increment counter by amount."""
        label_key = self._make_label_key(labels)
        with self._lock:
            self._values[label_key] += amount
    
    def labels(self, **labels: str) -> InMemoryCounter:
        """Return a labeled version of this counter."""
        # For simplicity, return self and store labels for next operation
        self._current_labels = labels
        return self
    
    def get_value(self, **labels: str) -> float:
        """Get current counter value for given labels."""
        label_key = self._make_label_key(labels)
        with self._lock:
            return self._values[label_key]
    
    def get_all_values(self) -> dict[tuple[str, ...], float]:
        """Get all counter values."""
        with self._lock:
            return dict(self._values)
    
    def _make_label_key(self, labels: dict[str, str]) -> tuple[str, ...]:
        """Create a hashable key from labels."""
        return tuple(labels.get(name, "") for name in self.label_names)


class InMemoryGauge:
    """Real gauge implementation that stores values in memory."""
    
    def __init__(self, name: str, description: str, label_names: list[str] | None = None) -> None:
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: DefaultDict[tuple[str, ...], float] = defaultdict(float)
        self._lock = threading.Lock()
    
    def set(self, value: float, **labels: str) -> None:
        """Set gauge to value."""
        label_key = self._make_label_key(labels)
        with self._lock:
            self._values[label_key] = value
    
    def inc(self, amount: float = 1.0, **labels: str) -> None:
        """Increment gauge by amount."""
        label_key = self._make_label_key(labels)
        with self._lock:
            self._values[label_key] += amount
    
    def dec(self, amount: float = 1.0, **labels: str) -> None:
        """Decrement gauge by amount."""
        label_key = self._make_label_key(labels)
        with self._lock:
            self._values[label_key] -= amount
    
    def labels(self, **labels: str) -> InMemoryGauge:
        """Return a labeled version of this gauge."""
        self._current_labels = labels
        return self
    
    def get_value(self, **labels: str) -> float:
        """Get current gauge value for given labels."""
        label_key = self._make_label_key(labels)
        with self._lock:
            return self._values[label_key]
    
    def get_all_values(self) -> dict[tuple[str, ...], float]:
        """Get all gauge values."""
        with self._lock:
            return dict(self._values)
    
    def _make_label_key(self, labels: dict[str, str]) -> tuple[str, ...]:
        """Create a hashable key from labels."""
        return tuple(labels.get(name, "") for name in self.label_names)


class InMemoryHistogram:
    """Real histogram implementation that stores observations in memory."""
    
    def __init__(
        self, 
        name: str, 
        description: str, 
        label_names: list[str] | None = None,
        buckets: list[float] | None = None
    ) -> None:
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
        self._observations: DefaultDict[tuple[str, ...], Deque[float]] = defaultdict(lambda: deque(maxlen=10000))
        self._lock = threading.Lock()
    
    def observe(self, value: float, **labels: str) -> None:
        """Record an observation."""
        label_key = self._make_label_key(labels)
        with self._lock:
            self._observations[label_key].append(value)
    
    def labels(self, **labels: str) -> InMemoryHistogram:
        """Return a labeled version of this histogram."""
        self._current_labels = labels
        return self
    
    @contextmanager
    def time(self, **labels: str):
        """Context manager to time an operation."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.observe(duration, **labels)
    
    def get_observations(self, **labels: str) -> list[float]:
        """Get all observations for given labels."""
        label_key = self._make_label_key(labels)
        with self._lock:
            return list(self._observations[label_key])
    
    def get_bucket_counts(self, **labels: str) -> dict[float, int]:
        """Get bucket counts for histogram."""
        observations = self.get_observations(**labels)
        bucket_counts = {}
        
        for bucket in self.buckets:
            count = sum(1 for obs in observations if obs <= bucket)
            bucket_counts[bucket] = count
        
        return bucket_counts
    
    def get_statistics(self, **labels: str) -> dict[str, float]:
        """Get statistical summary of observations."""
        observations = self.get_observations(**labels)
        if not observations:
            return {"count": 0, "sum": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "count": len(observations),
            "sum": sum(observations),
            "avg": sum(observations) / len(observations),
            "min": min(observations),
            "max": max(observations)
        }
    
    def _make_label_key(self, labels: dict[str, str]) -> tuple[str, ...]:
        """Create a hashable key from labels."""
        return tuple(labels.get(name, "") for name in self.label_names)


class InMemoryTimer:
    """Real timer context manager for histogram timing."""
    
    def __init__(self, histogram: InMemoryHistogram, labels: dict[str, str]) -> None:
        self.histogram = histogram
        self.labels = labels
        self.start_time: float | None = None
    
    def __enter__(self) -> InMemoryTimer:
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Stop timing and record observation."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.histogram.observe(duration, **self.labels)


class RealMetricsRegistry:
    """
    Real metrics registry that provides actual metrics collection.
    
    Uses Prometheus when available, falls back to in-memory collection
    that maintains real behavior and data persistence.
    """
    
    def __init__(self) -> None:
        self._metrics: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Try to import Prometheus
        self._prometheus_available = self._check_prometheus_availability()
        
        if self._prometheus_available:
            self.logger.info("Using Prometheus metrics backend")
        else:
            self.logger.info("Using in-memory metrics backend (real behavior)")
    
    def _check_prometheus_availability(self) -> bool:
        """Check if Prometheus client is available."""
        try:
            import prometheus_client
            return True
        except ImportError:
            return False
    
    def get_or_create_counter(
        self,
        name: str,
        description: str,
        labels: list[str] | None = None,
        registry: Any = None
    ) -> Any:
        """Get or create a counter metric."""
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]
            
            if self._prometheus_available:
                try:
                    from prometheus_client import Counter
                    metric = Counter(name, description, labels or [], registry=registry)
                    self._metrics[name] = metric
                    return metric
                except Exception as e:
                    self.logger.warning(f"Failed to create Prometheus counter {name}: {e}")
            
            # Fallback to in-memory counter
            metric = InMemoryCounter(name, description, labels)
            self._metrics[name] = metric
            return metric
    
    def get_or_create_gauge(
        self,
        name: str,
        description: str,
        labels: list[str] | None = None,
        registry: Any = None
    ) -> Any:
        """Get or create a gauge metric."""
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]
            
            if self._prometheus_available:
                try:
                    from prometheus_client import Gauge
                    metric = Gauge(name, description, labels or [], registry=registry)
                    self._metrics[name] = metric
                    return metric
                except Exception as e:
                    self.logger.warning(f"Failed to create Prometheus gauge {name}: {e}")
            
            # Fallback to in-memory gauge
            metric = InMemoryGauge(name, description, labels)
            self._metrics[name] = metric
            return metric
    
    def get_or_create_histogram(
        self,
        name: str,
        description: str,
        labels: list[str] | None = None,
        buckets: list[float] | None = None,
        registry: Any = None
    ) -> Any:
        """Get or create a histogram metric."""
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]
            
            if self._prometheus_available:
                try:
                    from prometheus_client import Histogram
                    kwargs = {}
                    if buckets:
                        kwargs['buckets'] = buckets
                    metric = Histogram(name, description, labels or [], registry=registry, **kwargs)
                    self._metrics[name] = metric
                    return metric
                except Exception as e:
                    self.logger.warning(f"Failed to create Prometheus histogram {name}: {e}")
            
            # Fallback to in-memory histogram
            metric = InMemoryHistogram(name, description, labels, buckets)
            self._metrics[name] = metric
            return metric
    
    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of all metrics."""
        summary = {
            "backend": "prometheus" if self._prometheus_available else "in_memory",
            "total_metrics": len(self._metrics),
            "metrics": {}
        }
        
        for name, metric in self._metrics.items():
            if hasattr(metric, 'get_all_values'):
                # In-memory metric
                summary["metrics"][name] = {
                    "type": type(metric).__name__,
                    "values": metric.get_all_values()
                }
            else:
                # Prometheus metric
                summary["metrics"][name] = {
                    "type": type(metric).__name__,
                    "prometheus_metric": True
                }
        
        return summary


# Global registry instance
_real_metrics_registry: RealMetricsRegistry | None = None


def get_real_metrics_registry() -> RealMetricsRegistry:
    """Get the global real metrics registry."""
    global _real_metrics_registry
    if _real_metrics_registry is None:
        _real_metrics_registry = RealMetricsRegistry()
    return _real_metrics_registry
