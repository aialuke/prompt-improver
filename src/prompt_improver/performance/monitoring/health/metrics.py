"""OpenTelemetry metrics instrumentation for APES health checks.
Health Check Consolidation - Metrics Integration
"""

import time
from collections.abc import Callable
from functools import wraps


class _Timer:
    """Context manager for timing code execution"""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.duration = self.end - self.start
        return False


class MockMetric:
    """Mock metric class for when metrics are not available"""

    def __init__(self, *args, **kwargs):
        pass

    def inc(self, *args, **kwargs):
        pass

    def set(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass

    def labels(self, *args, **kwargs):
        return self

    def time(self):
        return _Timer()


Counter = Gauge = Histogram = Summary = MockMetric


def _create_metric_safe(metric_class, *args, **kwargs):
    """Create a metric with graceful duplicate handling."""
