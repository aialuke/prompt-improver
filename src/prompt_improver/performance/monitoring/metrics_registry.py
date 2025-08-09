"""Centralized Metrics Registry for OpenTelemetry.

Prevents duplicate metric registration and provides a single source of truth
for all metrics across the application. Uses real metrics collection with
OpenTelemetry as the standard observability backend.
"""
from __future__ import annotations
import logging
from threading import Lock
from typing import Any
from prompt_improver.performance.monitoring.real_metrics import get_real_metrics_registry
logger = logging.getLogger(__name__)

class MetricsRegistry:
    """Centralized registry for all metrics.

    Uses real metrics collection with OpenTelemetry backend for
    comprehensive observability and monitoring.
    """

    def __init__(self, registry: Any=None) -> None:
        self._real_registry = get_real_metrics_registry()
        self._lock = Lock()
        logger.info('Initialized MetricsRegistry with real metrics backend')

    def get_or_create_counter(self, name: str, description: str, labels: list[str] | None=None, registry: Any=None) -> Any:
        """Get existing or create new counter metric."""
        return self._real_registry.get_or_create_counter(name, description, labels, registry)

    def get_or_create_gauge(self, name: str, description: str, labels: list[str] | None=None, registry: Any=None) -> Any:
        """Get existing or create new gauge metric."""
        return self._real_registry.get_or_create_gauge(name, description, labels, registry)

    def get_or_create_histogram(self, name: str, description: str, labels: list[str] | None=None, buckets: list[float] | None=None, registry: Any=None) -> Any:
        """Get existing or create new histogram metric."""
        return self._real_registry.get_or_create_histogram(name, description, labels, buckets, registry)

    def get_or_create_summary(self, name: str, description: str, labels: list[str] | None=None, registry: Any=None) -> Any:
        """Get existing or create new summary metric."""
        return self._real_registry.get_or_create_histogram(name, description, labels, None, registry)

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of all metrics."""
        return self._real_registry.get_metrics_summary()

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all registered metrics."""
        return self._real_registry.get_metrics_summary()

    def clear_metrics(self) -> None:
        """Clear all metrics (useful for testing)."""
        global _global_metrics_registry
        _global_metrics_registry = None
        logger.debug('Reset metrics registry')
_global_metrics_registry: MetricsRegistry | None = None

def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry instance."""
    global _global_metrics_registry
    if _global_metrics_registry is None:
        _global_metrics_registry = MetricsRegistry()
    return _global_metrics_registry

def set_metrics_registry(registry: MetricsRegistry) -> None:
    """Set the global metrics registry instance."""
    global _global_metrics_registry
    _global_metrics_registry = registry

def get_counter(name: str, description: str, labels: list[str] | None=None) -> Any:
    """Get or create a counter metric."""
    return get_metrics_registry().get_or_create_counter(name, description, labels)

def get_gauge(name: str, description: str, labels: list[str] | None=None) -> Any:
    """Get or create a gauge metric."""
    return get_metrics_registry().get_or_create_gauge(name, description, labels)

def get_histogram(name: str, description: str, labels: list[str] | None=None, buckets: list[float] | None=None) -> Any:
    """Get or create a histogram metric."""
    return get_metrics_registry().get_or_create_histogram(name, description, labels, buckets)

def get_summary(name: str, description: str, labels: list[str] | None=None) -> Any:
    """Get or create a summary metric."""
    return get_metrics_registry().get_or_create_summary(name, description, labels)

class StandardMetrics:
    """Standard metric definitions to prevent naming conflicts."""
    DATABASE_CONNECTIONS_ACTIVE = 'database_connections_active'
    DATABASE_POOL_UTILIZATION = 'database_pool_utilization'
    DATABASE_QUERY_DURATION = 'database_query_duration_seconds'
    DATABASE_ERRORS_TOTAL = 'database_errors_total'
    RETRY_ATTEMPTS_TOTAL = 'retry_attempts_total'
    RETRY_SUCCESS_TOTAL = 'retry_success_total'
    RETRY_FAILURE_TOTAL = 'retry_failure_total'
    RETRY_DURATION = 'retry_duration_seconds'
    RETRY_DELAY = 'retry_delay_seconds'
    CIRCUIT_BREAKER_STATE = 'circuit_breaker_state'
    CIRCUIT_BREAKER_TRIPS_TOTAL = 'circuit_breaker_trips_total'
    CIRCUIT_BREAKER_EVENTS_TOTAL = 'circuit_breaker_events_total'
    HEALTH_CHECK_DURATION = 'health_check_duration_seconds'
    HEALTH_CHECK_STATUS = 'health_check_status'
    HEALTH_CHECK_TOTAL = 'health_checks_total'
    BACKGROUND_TASKS_TOTAL = 'background_tasks_total'
    BACKGROUND_TASK_DURATION = 'background_task_duration_seconds'
    BACKGROUND_TASKS_ACTIVE = 'background_tasks_active'
    BACKGROUND_TASK_RETRIES_TOTAL = 'background_task_retries_total'
    CACHE_HITS_TOTAL = 'cache_hits_total'
    CACHE_MISSES_TOTAL = 'cache_misses_total'
    CACHE_OPERATION_DURATION = 'cache_operation_duration_seconds'
    HTTP_REQUESTS_TOTAL = 'http_requests_total'
    HTTP_REQUEST_DURATION = 'http_request_duration_seconds'
    HTTP_ERRORS_TOTAL = 'http_errors_total'

def initialize_standard_metrics():
    """Initialize all standard metrics to prevent registration conflicts."""
    registry = get_metrics_registry()
    registry.get_or_create_gauge(StandardMetrics.DATABASE_CONNECTIONS_ACTIVE, 'Number of active database connections', ['pool_name'])
    registry.get_or_create_gauge(StandardMetrics.DATABASE_POOL_UTILIZATION, 'Database connection pool utilization percentage', ['pool_name'])
    registry.get_or_create_histogram(StandardMetrics.DATABASE_QUERY_DURATION, 'Database query execution duration', ['operation', 'table'], buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    registry.get_or_create_counter(StandardMetrics.DATABASE_ERRORS_TOTAL, 'Total database errors', ['error_type', 'operation'])
    registry.get_or_create_counter(StandardMetrics.RETRY_ATTEMPTS_TOTAL, 'Total retry attempts', ['operation', 'strategy', 'attempt'])
    registry.get_or_create_counter(StandardMetrics.RETRY_SUCCESS_TOTAL, 'Total successful retries', ['operation', 'strategy'])
    registry.get_or_create_counter(StandardMetrics.RETRY_FAILURE_TOTAL, 'Total failed retries', ['operation', 'strategy', 'error_type'])
    registry.get_or_create_histogram(StandardMetrics.RETRY_DURATION, 'Retry operation duration', ['operation', 'strategy'], buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0])
    registry.get_or_create_gauge(StandardMetrics.CIRCUIT_BREAKER_STATE, 'Circuit breaker state (0=closed, 1=half_open, 2=open)', ['operation'])
    registry.get_or_create_counter(StandardMetrics.CIRCUIT_BREAKER_TRIPS_TOTAL, 'Total circuit breaker trips', ['operation'])
    logger.info('Standard metrics initialized successfully')
initialize_standard_metrics()
