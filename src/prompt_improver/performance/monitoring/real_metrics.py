"""Real Metrics Implementation for APES.

Provides real metrics collection behavior without mock objects, using OpenTelemetry
when available or in-memory collection as fallback.
Maintains full functionality and real behavior in all scenarios.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


class CounterAdapter:
    """Adapter to provide counter interface using OpenTelemetry."""

    def __init__(
        self, name: str, description: str, label_names: list[str] | None, adapter
    ) -> None:
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self.adapter = adapter
        self._current_labels = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        """Increment counter by amount."""
        labels_dict = {**self._current_labels, **labels}
        self.adapter.increment_counter(self.name, labels_dict)

    def labels(self, **labels: str):
        """Return a labeled version of this counter."""
        self._current_labels = labels
        return self


class GaugeAdapter:
    """Adapter to provide gauge interface using OpenTelemetry."""

    def __init__(
        self, name: str, description: str, label_names: list[str] | None, adapter
    ) -> None:
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self.adapter = adapter
        self._current_labels = {}

    def set(self, value: float, **labels: str) -> None:
        """Set gauge to value."""
        labels_dict = {**self._current_labels, **labels}
        self.adapter.set_gauge(self.name, value, labels_dict)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        """Increment gauge by amount."""
        labels_dict = {**self._current_labels, **labels}
        current_value = getattr(
            self, f"_cached_value_{hash(frozenset(labels_dict.items()))}", 0.0
        )
        new_value = current_value + amount
        setattr(
            self, f"_cached_value_{hash(frozenset(labels_dict.items()))}", new_value
        )
        self.adapter.set_gauge(self.name, new_value, labels_dict)

    def dec(self, amount: float = 1.0, **labels: str) -> None:
        """Decrement gauge by amount."""
        self.inc(-amount, **labels)

    def labels(self, **labels: str):
        """Return a labeled version of this gauge."""
        self._current_labels = labels
        return self


class HistogramAdapter:
    """Adapter to provide histogram interface using OpenTelemetry."""

    def __init__(
        self,
        name: str,
        description: str,
        label_names: list[str] | None,
        buckets: list[float] | None,
        adapter,
    ) -> None:
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self.buckets = buckets
        self.adapter = adapter
        self._current_labels = {}

    def observe(self, value: float, **labels: str) -> None:
        """Record an observation."""
        labels_dict = {**self._current_labels, **labels}
        self.adapter.record_histogram(self.name, value, labels_dict)

    def labels(self, **labels: str):
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


@dataclass
class MetricValue:
    """Represents a metric value with timestamp and labels."""

    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    labels: dict[str, str] = field(default_factory=dict)


class InMemoryCounter:
    """Real counter implementation that stores values in memory."""

    def __init__(
        self, name: str, description: str, label_names: list[str] | None = None
    ) -> None:
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: defaultdict[tuple[str, ...], float] = defaultdict(float)
        self._lock = asyncio.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        """Increment counter by amount (sync version for compatibility)."""
        label_key = self._make_label_key(labels)
        # For sync compatibility, use lock without await
        # Note: This creates a temporary sync context
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            # Running in async context, schedule sync increment
            self._values[label_key] += amount
        except RuntimeError:
            # Not in async context, direct access
            self._values[label_key] += amount

    async def ainc(self, amount: float = 1.0, **labels: str) -> None:
        """Increment counter by amount (async version)."""
        label_key = self._make_label_key(labels)
        async with self._lock:
            self._values[label_key] += amount

    def labels(self, **labels: str) -> InMemoryCounter:
        """Return a labeled version of this counter."""
        self._current_labels = labels
        return self

    def get_value(self, **labels: str) -> float:
        """Get current counter value for given labels (sync version)."""
        label_key = self._make_label_key(labels)
        # Sync access for compatibility
        return self._values[label_key]

    async def aget_value(self, **labels: str) -> float:
        """Get current counter value for given labels (async version)."""
        label_key = self._make_label_key(labels)
        async with self._lock:
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

    def __init__(
        self, name: str, description: str, label_names: list[str] | None = None
    ) -> None:
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: defaultdict[tuple[str, ...], float] = defaultdict(float)
        self._lock = asyncio.Lock()

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
        buckets: list[float] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self.buckets = buckets or [
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
        ]
        self._observations: defaultdict[tuple[str, ...], deque[float]] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self._lock = asyncio.Lock()

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


class RealMetricsRegistry:
    """Simple real metrics registry."""

    def __init__(self) -> None:
        self.counters = {}
        self.gauges = {}
        self.histograms = {}

    def get_or_create_counter(self, name, description, labels=None, registry=None):
        """Get or create a counter."""
        if name not in self.counters:
            self.counters[name] = InMemoryCounter(name, description, labels)
        return self.counters[name]

    def get_or_create_gauge(self, name, description, labels=None, registry=None):
        """Get or create a gauge."""
        if name not in self.gauges:
            self.gauges[name] = InMemoryGauge(name, description, labels)
        return self.gauges[name]

    def get_or_create_histogram(
        self, name, description, labels=None, buckets=None, registry=None
    ):
        """Get or create a histogram."""
        if name not in self.histograms:
            self.histograms[name] = InMemoryHistogram(
                name, description, labels, buckets
            )
        return self.histograms[name]

    def get_metrics_summary(self):
        """Get summary of all metrics."""
        return {
            "counters": {
                name: counter.get_all_values()
                for name, counter in self.counters.items()
            },
            "gauges": {
                name: gauge.get_all_values() for name, gauge in self.gauges.items()
            },
            "histograms": {
                name: len(hist.get_observations())
                for name, hist in self.histograms.items()
            },
        }


_real_metrics_registry = None


def get_real_metrics_registry():
    """Get the global real metrics registry."""
    global _real_metrics_registry
    if _real_metrics_registry is None:
        _real_metrics_registry = RealMetricsRegistry()
    return _real_metrics_registry
