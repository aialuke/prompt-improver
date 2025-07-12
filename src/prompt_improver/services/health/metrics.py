"""Prometheus metrics instrumentation for APES health checks.
PHASE 3: Health Check Consolidation - Metrics Integration
"""

from functools import wraps
from typing import Callable, Any
import time


class _Timer:
    """Context manager for timing code execution"""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.duration = self.end - self.start
        return False  # Don't suppress exceptions


try:
    from prometheus_client import Counter, Gauge, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    # Graceful degradation when prometheus_client is not available
    PROMETHEUS_AVAILABLE = False
    
    class MockMetric:
        """Mock metric class for when Prometheus is not available"""
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


# Health check metrics with graceful duplicate handling
def _create_metric_safe(metric_class, *args, **kwargs):
    """Create a Prometheus metric with graceful duplicate handling."""
    if not PROMETHEUS_AVAILABLE:
        # Create a MockMetric class locally when needed
        class LocalMockMetric:
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
        return LocalMockMetric()
    
    try:
        return metric_class(*args, **kwargs)
    except ValueError as e:
        if "Duplicated timeseries" in str(e):
            # Metric already exists, create a mock instead
            class LocalMockMetric:
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
            return LocalMockMetric()
        else:
            raise

HEALTH_CHECK_DURATION = _create_metric_safe(
    Summary,
    'health_check_duration_seconds', 
    'Time spent performing health checks',
    ['component']
)

HEALTH_CHECK_STATUS = _create_metric_safe(
    Gauge,
    'health_check_status',
    'Health check status (1=healthy, 0.5=warning, 0=failed)',
    ['component']
)

HEALTH_CHECKS_TOTAL = _create_metric_safe(
    Counter,
    'health_checks_total',
    'Total number of health checks performed',
    ['component', 'status']
)

HEALTH_CHECK_RESPONSE_TIME = _create_metric_safe(
    Histogram,
    'health_check_response_time_milliseconds',
    'Health check response time distribution',
    ['component'],
    buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000]
)


def instrument_health_check(component_name: str):
    """Decorator to instrument health checks with Prometheus metrics"""
    def decorator(check_func: Callable) -> Callable:
        if not PROMETHEUS_AVAILABLE:
            return check_func
            
        @wraps(check_func)
        async def wrapper(*args, **kwargs):
            # Time the health check
            with HEALTH_CHECK_DURATION.labels(component=component_name).time():
                result = await check_func(*args, **kwargs)
                
                # Record metrics based on result
                if hasattr(result, 'status'):
                    # Map status to numeric value
                    status_value = {
                        'healthy': 1.0,
                        'warning': 0.5,
                        'failed': 0.0
                    }.get(result.status.value if hasattr(result.status, 'value') else str(result.status), 0.0)
                    
                    HEALTH_CHECK_STATUS.labels(component=component_name).set(status_value)
                    
                    status_str = result.status.value if hasattr(result.status, 'value') else str(result.status)
                    HEALTH_CHECKS_TOTAL.labels(
                        component=component_name,
                        status=status_str
                    ).inc()
                    
                    # Record response time if available
                    if hasattr(result, 'response_time_ms') and result.response_time_ms is not None:
                        HEALTH_CHECK_RESPONSE_TIME.labels(component=component_name).observe(
                            result.response_time_ms
                        )
                
                return result
        return wrapper
    return decorator


def get_health_metrics_summary() -> dict:
    """Get current health metrics summary"""
    if not PROMETHEUS_AVAILABLE:
        return {"prometheus_available": False}
    
    try:
        from prometheus_client import REGISTRY, generate_latest
        # This would typically be called by the metrics endpoint
        return {"prometheus_available": True, "metrics_registered": True}
    except Exception:
        return {"prometheus_available": True, "error": "Failed to generate metrics"}


def reset_health_metrics():
    """Reset health check metrics (useful for testing)"""
    if not PROMETHEUS_AVAILABLE:
        return
    
    try:
        # Clear counters and gauges
        HEALTH_CHECK_STATUS.clear()
        # Note: Counters cannot be reset in Prometheus, this is by design
        # Histograms and Summaries also cannot be reset
    except Exception:
        pass  # Gracefully handle any issues
