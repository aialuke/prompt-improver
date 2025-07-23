"""
Centralized Metrics Registry for Prometheus.

Prevents duplicate metric registration and provides a single source of truth
for all metrics across the application.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from threading import Lock

# Define MockMetric classes first (always available)
class MockMetric:
    def inc(self, *args, **kwargs): pass
    def set(self, *args, **kwargs): pass
    def observe(self, *args, **kwargs): pass
    def labels(self, *args, **kwargs): return self
    def time(self): return MockTimer()

class MockTimer:
    def __enter__(self): return self
    def __exit__(self, *args): pass

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Gauge = Histogram = Summary = MockMetric
    CollectorRegistry = REGISTRY = None

logger = logging.getLogger(__name__)


class MetricsRegistry:
    """
    Centralized registry for all Prometheus metrics.
    
    Prevents duplicate registrations and provides consistent metric management.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
        self._metrics: Dict[str, Any] = {}
        self._lock = Lock()
        
    def get_or_create_counter(
        self, 
        name: str, 
        description: str, 
        labels: Optional[List[str]] = None,
        registry: Optional[CollectorRegistry] = None
    ):
        """Get existing or create new Counter metric."""
        return self._get_or_create_metric(
            Counter, name, description, labels, registry
        )
    
    def get_or_create_gauge(
        self, 
        name: str, 
        description: str, 
        labels: Optional[List[str]] = None,
        registry: Optional[CollectorRegistry] = None
    ):
        """Get existing or create new Gauge metric."""
        return self._get_or_create_metric(
            Gauge, name, description, labels, registry
        )
    
    def get_or_create_histogram(
        self, 
        name: str, 
        description: str, 
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
        registry: Optional[CollectorRegistry] = None
    ):
        """Get existing or create new Histogram metric."""
        kwargs = {}
        if buckets:
            kwargs['buckets'] = buckets
        
        return self._get_or_create_metric(
            Histogram, name, description, labels, registry, **kwargs
        )
    
    def get_or_create_summary(
        self, 
        name: str, 
        description: str, 
        labels: Optional[List[str]] = None,
        registry: Optional[CollectorRegistry] = None
    ):
        """Get existing or create new Summary metric."""
        return self._get_or_create_metric(
            Summary, name, description, labels, registry
        )
    
    def _get_or_create_metric(
        self,
        metric_class,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        registry: Optional[CollectorRegistry] = None,
        **kwargs
    ):
        """Internal method to get or create metrics safely."""
        if not PROMETHEUS_AVAILABLE:
            return MockMetric()
        
        with self._lock:
            # Check if metric already exists
            if name in self._metrics:
                logger.debug(f"Returning existing metric: {name}")
                return self._metrics[name]
            
            try:
                # Create new metric
                metric_registry = registry or self.registry
                
                if labels:
                    metric = metric_class(name, description, labels, registry=metric_registry, **kwargs)
                else:
                    metric = metric_class(name, description, registry=metric_registry, **kwargs)
                
                self._metrics[name] = metric
                logger.debug(f"Created new metric: {name}")
                return metric
                
            except ValueError as e:
                if "Duplicated timeseries" in str(e) or "already registered" in str(e):
                    # Metric exists in registry but not in our cache
                    # Try to find it in the registry
                    existing_metric = self._find_existing_metric(name)
                    if existing_metric:
                        self._metrics[name] = existing_metric
                        logger.debug(f"Found existing metric in registry: {name}")
                        return existing_metric
                    
                    # If we can't find it, create a mock to prevent errors
                    logger.warning(f"Metric {name} already exists but couldn't be retrieved. Using mock.")
                    mock_metric = MockMetric()
                    self._metrics[name] = mock_metric
                    return mock_metric
                else:
                    logger.error(f"Failed to create metric {name}: {e}")
                    raise
    
    def _find_existing_metric(self, name: str):
        """Try to find existing metric in the registry."""
        if not PROMETHEUS_AVAILABLE or not self.registry:
            return None
        
        try:
            # Look through registered collectors
            for collector in self.registry._collector_to_names.keys():
                if hasattr(collector, '_name') and collector._name == name:
                    return collector
                # Some metrics might have different attribute names
                if hasattr(collector, 'describe'):
                    for metric_family in collector.describe():
                        if metric_family.name == name:
                            return collector
        except Exception as e:
            logger.debug(f"Error finding existing metric {name}: {e}")
        
        return None
    
    def get_metric(self, name: str):
        """Get existing metric by name."""
        return self._metrics.get(name)
    
    def list_metrics(self) -> List[str]:
        """List all registered metric names."""
        return list(self._metrics.keys())
    
    def clear_metrics(self):
        """Clear all metrics (for testing)."""
        with self._lock:
            self._metrics.clear()
    
    def get_metrics_info(self) -> Dict[str, Any]:
        """Get information about registered metrics."""
        return {
            "total_metrics": len(self._metrics),
            "metric_names": list(self._metrics.keys()),
            "prometheus_available": PROMETHEUS_AVAILABLE
        }


# Global metrics registry instance
_global_metrics_registry: Optional[MetricsRegistry] = None


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry instance."""
    global _global_metrics_registry
    if _global_metrics_registry is None:
        _global_metrics_registry = MetricsRegistry()
    return _global_metrics_registry


def set_metrics_registry(registry: MetricsRegistry):
    """Set the global metrics registry instance."""
    global _global_metrics_registry
    _global_metrics_registry = registry


# Convenience functions for common metric types
def get_counter(name: str, description: str, labels: Optional[List[str]] = None):
    """Get or create a Counter metric."""
    return get_metrics_registry().get_or_create_counter(name, description, labels)


def get_gauge(name: str, description: str, labels: Optional[List[str]] = None):
    """Get or create a Gauge metric."""
    return get_metrics_registry().get_or_create_gauge(name, description, labels)


def get_histogram(name: str, description: str, labels: Optional[List[str]] = None, buckets: Optional[List[float]] = None):
    """Get or create a Histogram metric."""
    return get_metrics_registry().get_or_create_histogram(name, description, labels, buckets)


def get_summary(name: str, description: str, labels: Optional[List[str]] = None):
    """Get or create a Summary metric."""
    return get_metrics_registry().get_or_create_summary(name, description, labels)


# Standard metric definitions used across the application
class StandardMetrics:
    """Standard metric definitions to prevent naming conflicts."""
    
    # Database metrics
    DATABASE_CONNECTIONS_ACTIVE = "database_connections_active"
    DATABASE_POOL_UTILIZATION = "database_pool_utilization"
    DATABASE_QUERY_DURATION = "database_query_duration_seconds"
    DATABASE_ERRORS_TOTAL = "database_errors_total"
    
    # Retry metrics
    RETRY_ATTEMPTS_TOTAL = "retry_attempts_total"
    RETRY_SUCCESS_TOTAL = "retry_success_total"
    RETRY_FAILURE_TOTAL = "retry_failure_total"
    RETRY_DURATION = "retry_duration_seconds"
    RETRY_DELAY = "retry_delay_seconds"
    
    # Circuit breaker metrics
    CIRCUIT_BREAKER_STATE = "circuit_breaker_state"
    CIRCUIT_BREAKER_TRIPS_TOTAL = "circuit_breaker_trips_total"
    CIRCUIT_BREAKER_EVENTS_TOTAL = "circuit_breaker_events_total"
    
    # Health check metrics
    HEALTH_CHECK_DURATION = "health_check_duration_seconds"
    HEALTH_CHECK_STATUS = "health_check_status"
    HEALTH_CHECK_TOTAL = "health_checks_total"
    
    # Background task metrics
    BACKGROUND_TASKS_TOTAL = "background_tasks_total"
    BACKGROUND_TASK_DURATION = "background_task_duration_seconds"
    BACKGROUND_TASKS_ACTIVE = "background_tasks_active"
    BACKGROUND_TASK_RETRIES_TOTAL = "background_task_retries_total"
    
    # Cache metrics
    CACHE_HITS_TOTAL = "cache_hits_total"
    CACHE_MISSES_TOTAL = "cache_misses_total"
    CACHE_OPERATION_DURATION = "cache_operation_duration_seconds"
    
    # HTTP metrics
    HTTP_REQUESTS_TOTAL = "http_requests_total"
    HTTP_REQUEST_DURATION = "http_request_duration_seconds"
    HTTP_ERRORS_TOTAL = "http_errors_total"


def initialize_standard_metrics():
    """Initialize all standard metrics to prevent registration conflicts."""
    registry = get_metrics_registry()
    
    # Database metrics
    registry.get_or_create_gauge(
        StandardMetrics.DATABASE_CONNECTIONS_ACTIVE,
        "Number of active database connections",
        ["pool_name"]
    )
    
    registry.get_or_create_gauge(
        StandardMetrics.DATABASE_POOL_UTILIZATION,
        "Database connection pool utilization percentage",
        ["pool_name"]
    )
    
    registry.get_or_create_histogram(
        StandardMetrics.DATABASE_QUERY_DURATION,
        "Database query execution duration",
        ["operation", "table"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    
    registry.get_or_create_counter(
        StandardMetrics.DATABASE_ERRORS_TOTAL,
        "Total database errors",
        ["error_type", "operation"]
    )
    
    # Retry metrics
    registry.get_or_create_counter(
        StandardMetrics.RETRY_ATTEMPTS_TOTAL,
        "Total retry attempts",
        ["operation", "strategy", "attempt"]
    )
    
    registry.get_or_create_counter(
        StandardMetrics.RETRY_SUCCESS_TOTAL,
        "Total successful retries",
        ["operation", "strategy"]
    )
    
    registry.get_or_create_counter(
        StandardMetrics.RETRY_FAILURE_TOTAL,
        "Total failed retries",
        ["operation", "strategy", "error_type"]
    )
    
    registry.get_or_create_histogram(
        StandardMetrics.RETRY_DURATION,
        "Retry operation duration",
        ["operation", "strategy"],
        buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
    )
    
    # Circuit breaker metrics
    registry.get_or_create_gauge(
        StandardMetrics.CIRCUIT_BREAKER_STATE,
        "Circuit breaker state (0=closed, 1=half_open, 2=open)",
        ["operation"]
    )
    
    registry.get_or_create_counter(
        StandardMetrics.CIRCUIT_BREAKER_TRIPS_TOTAL,
        "Total circuit breaker trips",
        ["operation"]
    )
    
    logger.info("Standard metrics initialized successfully")


# Initialize standard metrics on module import
if PROMETHEUS_AVAILABLE:
    initialize_standard_metrics()
