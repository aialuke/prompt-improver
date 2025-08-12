"""Unified metrics collection service with OpenTelemetry integration.

Extracted from database.database services architecture to provide:
- OpenTelemetry metrics collection (counters, gauges, histograms)
- Cache operation metrics with multi-level tracking
- Connection pool metrics with real-time updates
- Security context integration for operation tracking
- Performance metrics aggregation and reporting

This centralizes all metrics collection patterns from the monolithic manager.
"""

import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# OpenTelemetry imports with fallback
OPENTELEMETRY_AVAILABLE: bool = False
cache_tracer = None
cache_meter = None
cache_operations_counter = None
cache_hit_ratio_gauge = None
cache_latency_histogram = None
connection_pool_gauge = None
query_duration_histogram = None
security_operations_counter = None

try:
    from opentelemetry import metrics, trace

    OPENTELEMETRY_AVAILABLE = True
    cache_tracer = trace.get_tracer(__name__ + ".cache")
    cache_meter = metrics.get_meter(__name__ + ".cache")

    # Cache metrics
    cache_operations_counter = cache_meter.create_counter(
        "unified_cache_operations_total",
        description="Total unified cache operations by type, level, and status",
        unit="1",
    )
    cache_hit_ratio_gauge = cache_meter.create_gauge(
        "unified_cache_hit_ratio",
        description="Unified cache hit ratio by level",
        unit="ratio",
    )
    cache_latency_histogram = cache_meter.create_histogram(
        "unified_cache_operation_duration_seconds",
        description="Unified cache operation duration by type and level",
        unit="s",
    )

    # Connection pool metrics
    connection_pool_gauge = cache_meter.create_gauge(
        "connection_pool_active_connections",
        description="Number of active connections in the pool",
        unit="1",
    )

    # Query performance metrics
    query_duration_histogram = cache_meter.create_histogram(
        "database_query_duration_seconds",
        description="Database query duration in seconds",
        unit="s",
    )

    # Security metrics
    security_operations_counter = cache_meter.create_counter(
        "security_operations_total",
        description="Total security operations by type and status",
        unit="1",
    )

except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.info("OpenTelemetry not available - metrics will be collected locally only")


@dataclass
class OperationStats:
    """Statistics for a specific operation type."""

    count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    error_count: int = 0
    success_count: int = 0

    @property
    def avg_time(self) -> float:
        """Calculate average operation time."""
        return self.total_time / self.count if self.count > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        return (self.success_count / self.count * 100) if self.count > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        return (self.error_count / self.count * 100) if self.count > 0 else 0.0


@dataclass
class CacheMetrics:
    """Cache-specific metrics tracking."""

    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    total_requests: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    operation_stats: Dict[str, OperationStats] = field(
        default_factory=lambda: defaultdict(OperationStats)
    )

    @property
    def hit_rate(self) -> float:
        """Calculate overall cache hit rate."""
        total_hits = self.l1_hits + self.l2_hits + self.l3_hits
        return (
            (total_hits / self.total_requests * 100) if self.total_requests > 0 else 0.0
        )

    @property
    def avg_response_time(self) -> float:
        """Calculate average cache response time."""
        return statistics.mean(self.response_times) if self.response_times else 0.0


@dataclass
class ConnectionMetrics:
    """Connection pool metrics tracking."""

    active_connections: int = 0
    idle_connections: int = 0
    total_connections: int = 0
    pool_size: int = 0
    max_pool_size: int = 0
    connections_created: int = 0
    connections_closed: int = 0
    connection_failures: int = 0
    queries_executed: int = 0
    queries_failed: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))

    @property
    def pool_utilization(self) -> float:
        """Calculate pool utilization percentage."""
        if self.pool_size > 0:
            return (self.active_connections / self.pool_size) * 100
        return 0.0

    @property
    def success_rate(self) -> float:
        """Calculate query success rate."""
        total_queries = self.queries_executed + self.queries_failed
        return (
            (self.queries_executed / total_queries * 100) if total_queries > 0 else 0.0
        )

    @property
    def avg_response_time(self) -> float:
        """Calculate average response time."""
        return statistics.mean(self.response_times) if self.response_times else 0.0


@dataclass
class SecurityMetrics:
    """Security operation metrics tracking."""

    authentication_operations: int = 0
    authorization_operations: int = 0
    validation_operations: int = 0
    security_failures: int = 0
    context_validations: int = 0
    context_rejections: int = 0
    threat_assessments: int = 0
    audit_events_logged: int = 0

    @property
    def total_security_operations(self) -> int:
        """Total security operations."""
        return (
            self.authentication_operations
            + self.authorization_operations
            + self.validation_operations
        )

    @property
    def security_success_rate(self) -> float:
        """Calculate security operation success rate."""
        total = self.total_security_operations
        successes = total - self.security_failures
        return (successes / total * 100) if total > 0 else 0.0


class MetricsCollector:
    """Unified metrics collection service with OpenTelemetry integration.

    Centralizes all metrics collection from database services architecture:
    - Cache operation metrics with OpenTelemetry integration
    - Connection pool monitoring with real-time updates
    - Security context tracking and validation metrics
    - Performance aggregation with statistical analysis
    """

    def __init__(self, service_name: str = "unified_connection_manager"):
        self.service_name = service_name
        self.cache_metrics = CacheMetrics()
        self.connection_metrics = ConnectionMetrics()
        self.security_metrics = SecurityMetrics()
        self.started_at = datetime.now(UTC)

        # Performance tracking
        self.performance_window = deque(maxlen=100)
        self.last_metrics_update = time.time()

        logger.info(f"MetricsCollector initialized for {service_name}")
        if OPENTELEMETRY_AVAILABLE:
            logger.info("OpenTelemetry metrics enabled")
        else:
            logger.info("OpenTelemetry not available - using local metrics only")

    def record_cache_operation(
        self,
        operation: str,
        level: str,
        duration_ms: float,
        status: str = "success",
        security_context: Optional[Any] = None,
    ) -> None:
        """Record a cache operation with OpenTelemetry integration.

        Args:
            operation: Operation type (get, set, delete, exists)
            level: Cache level (l1, l2, l3, miss)
            duration_ms: Operation duration in milliseconds
            status: Operation status (success, error, not_found)
            security_context: Optional security context for tracking
        """
        # Update local metrics
        self.cache_metrics.total_requests += 1
        self.cache_metrics.response_times.append(duration_ms)

        # Update operation stats
        op_stats = self.cache_metrics.operation_stats[operation]
        op_stats.count += 1
        op_stats.total_time += duration_ms
        op_stats.min_time = min(op_stats.min_time, duration_ms)
        op_stats.max_time = max(op_stats.max_time, duration_ms)

        if status == "success":
            op_stats.success_count += 1
            # Track cache hits by level
            if level == "l1":
                self.cache_metrics.l1_hits += 1
            elif level == "l2":
                self.cache_metrics.l2_hits += 1
            elif level == "l3":
                self.cache_metrics.l3_hits += 1
        else:
            op_stats.error_count += 1

        # OpenTelemetry metrics
        if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
            cache_operations_counter.add(
                1,
                {
                    "operation": operation,
                    "level": level,
                    "status": status,
                    "service": self.service_name,
                    "security_context": security_context.agent_id
                    if security_context
                    else "none",
                },
            )

        if OPENTELEMETRY_AVAILABLE and cache_latency_histogram:
            cache_latency_histogram.record(
                duration_ms / 1000.0,  # Convert to seconds
                {
                    "operation": operation,
                    "level": level,
                    "service": self.service_name,
                },
            )

        # Update hit ratio gauge
        if OPENTELEMETRY_AVAILABLE and cache_hit_ratio_gauge:
            cache_hit_ratio_gauge.set(
                self.cache_metrics.hit_rate / 100.0,  # Convert to ratio
                {"service": self.service_name},
            )

    def record_connection_event(
        self, event_type: str, duration_ms: Optional[float] = None, success: bool = True
    ) -> None:
        """Record a connection pool event.

        Args:
            event_type: Event type (connect, disconnect, query, checkout, checkin)
            duration_ms: Optional operation duration
            success: Whether the operation was successful
        """
        if event_type == "connect":
            self.connection_metrics.active_connections += 1
            self.connection_metrics.total_connections += 1
            self.connection_metrics.connections_created += 1
        elif event_type == "disconnect":
            self.connection_metrics.active_connections = max(
                0, self.connection_metrics.active_connections - 1
            )
            self.connection_metrics.connections_closed += 1
        elif event_type == "query":
            if success:
                self.connection_metrics.queries_executed += 1
            else:
                self.connection_metrics.queries_failed += 1
                self.connection_metrics.connection_failures += 1
        elif event_type in ["checkout", "checkin"]:
            # Pool utilization will be calculated in properties
            pass

        if duration_ms is not None:
            self.connection_metrics.response_times.append(duration_ms)
            self.performance_window.append({
                "timestamp": time.time(),
                "duration_ms": duration_ms,
                "event_type": event_type,
                "success": success,
            })

            # OpenTelemetry query duration
            if (
                OPENTELEMETRY_AVAILABLE
                and query_duration_histogram
                and event_type == "query"
            ):
                query_duration_histogram.record(
                    duration_ms / 1000.0,  # Convert to seconds
                    {
                        "service": self.service_name,
                        "status": "success" if success else "error",
                    },
                )

        # Update connection pool gauge
        if OPENTELEMETRY_AVAILABLE and connection_pool_gauge:
            connection_pool_gauge.set(
                self.connection_metrics.active_connections,
                {"service": self.service_name},
            )

    def record_security_operation(
        self,
        operation_type: str,
        duration_ms: float,
        success: bool = True,
        security_context: Optional[Any] = None,
    ) -> None:
        """Record a security operation.

        Args:
            operation_type: Type of security operation
            duration_ms: Operation duration
            success: Whether operation was successful
            security_context: Optional security context
        """
        if operation_type == "authentication":
            self.security_metrics.authentication_operations += 1
        elif operation_type == "authorization":
            self.security_metrics.authorization_operations += 1
        elif operation_type == "validation":
            self.security_metrics.validation_operations += 1
        elif operation_type == "context_validation":
            self.security_metrics.context_validations += 1
            if not success:
                self.security_metrics.context_rejections += 1
        elif operation_type == "threat_assessment":
            self.security_metrics.threat_assessments += 1
        elif operation_type == "audit_event":
            self.security_metrics.audit_events_logged += 1

        if not success:
            self.security_metrics.security_failures += 1

        # OpenTelemetry security metrics
        if OPENTELEMETRY_AVAILABLE and security_operations_counter:
            security_operations_counter.add(
                1,
                {
                    "operation_type": operation_type,
                    "status": "success" if success else "failure",
                    "service": self.service_name,
                    "security_context": security_context.agent_id
                    if security_context
                    else "none",
                },
            )

    def update_connection_pool_config(self, pool_size: int, max_pool_size: int) -> None:
        """Update connection pool configuration metrics."""
        self.connection_metrics.pool_size = pool_size
        self.connection_metrics.max_pool_size = max_pool_size

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        operation_summaries = {}
        for op_name, stats in self.cache_metrics.operation_stats.items():
            if stats.count > 0:
                operation_summaries[op_name] = {
                    "count": stats.count,
                    "avg_time_ms": stats.avg_time,
                    "min_time_ms": stats.min_time
                    if stats.min_time != float("inf")
                    else 0,
                    "max_time_ms": stats.max_time,
                    "success_rate": stats.success_rate,
                    "error_count": stats.error_count,
                }

        return {
            "hit_rate_percent": self.cache_metrics.hit_rate,
            "total_requests": self.cache_metrics.total_requests,
            "l1_hits": self.cache_metrics.l1_hits,
            "l2_hits": self.cache_metrics.l2_hits,
            "l3_hits": self.cache_metrics.l3_hits,
            "avg_response_time_ms": self.cache_metrics.avg_response_time,
            "operations": operation_summaries,
            "response_time_samples": len(self.cache_metrics.response_times),
        }

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get comprehensive connection pool statistics."""
        return {
            "active_connections": self.connection_metrics.active_connections,
            "idle_connections": self.connection_metrics.idle_connections,
            "total_connections": self.connection_metrics.total_connections,
            "pool_size": self.connection_metrics.pool_size,
            "max_pool_size": self.connection_metrics.max_pool_size,
            "pool_utilization_percent": self.connection_metrics.pool_utilization,
            "connections_created": self.connection_metrics.connections_created,
            "connections_closed": self.connection_metrics.connections_closed,
            "connection_failures": self.connection_metrics.connection_failures,
            "queries_executed": self.connection_metrics.queries_executed,
            "queries_failed": self.connection_metrics.queries_failed,
            "query_success_rate_percent": self.connection_metrics.success_rate,
            "avg_response_time_ms": self.connection_metrics.avg_response_time,
            "response_time_samples": len(self.connection_metrics.response_times),
        }

    def get_security_stats(self) -> Dict[str, Any]:
        """Get comprehensive security operation statistics."""
        return {
            "authentication_operations": self.security_metrics.authentication_operations,
            "authorization_operations": self.security_metrics.authorization_operations,
            "validation_operations": self.security_metrics.validation_operations,
            "total_security_operations": self.security_metrics.total_security_operations,
            "security_failures": self.security_metrics.security_failures,
            "security_success_rate_percent": self.security_metrics.security_success_rate,
            "context_validations": self.security_metrics.context_validations,
            "context_rejections": self.security_metrics.context_rejections,
            "context_success_rate_percent": (
                (
                    self.security_metrics.context_validations
                    - self.security_metrics.context_rejections
                )
                / self.security_metrics.context_validations
                * 100
                if self.security_metrics.context_validations > 0
                else 0.0
            ),
            "threat_assessments": self.security_metrics.threat_assessments,
            "audit_events_logged": self.security_metrics.audit_events_logged,
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        uptime_seconds = (datetime.now(UTC) - self.started_at).total_seconds()

        # Calculate recent performance metrics
        recent_performance = []
        cutoff_time = time.time() - 300  # Last 5 minutes
        for event in self.performance_window:
            if event["timestamp"] > cutoff_time:
                recent_performance.append(event)

        recent_avg_duration = 0.0
        recent_success_rate = 0.0
        if recent_performance:
            recent_avg_duration = statistics.mean([
                e["duration_ms"] for e in recent_performance
            ])
            recent_successes = sum(1 for e in recent_performance if e["success"])
            recent_success_rate = (recent_successes / len(recent_performance)) * 100

        return {
            "service": self.service_name,
            "uptime_seconds": uptime_seconds,
            "uptime_hours": uptime_seconds / 3600,
            "last_metrics_update": self.last_metrics_update,
            "opentelemetry_available": OPENTELEMETRY_AVAILABLE,
            "cache_stats": self.get_cache_stats(),
            "connection_stats": self.get_connection_stats(),
            "security_stats": self.get_security_stats(),
            "recent_performance": {
                "avg_duration_ms": recent_avg_duration,
                "success_rate_percent": recent_success_rate,
                "sample_count": len(recent_performance),
                "window_minutes": 5,
            },
            "performance_window_size": len(self.performance_window),
        }

    def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing or periodic resets)."""
        self.cache_metrics = CacheMetrics()
        self.connection_metrics = ConnectionMetrics()
        self.security_metrics = SecurityMetrics()
        self.performance_window.clear()
        self.last_metrics_update = time.time()
        logger.info(f"Metrics reset for {self.service_name}")

    def export_metrics_for_telemetry(self) -> Dict[str, Any]:
        """Export metrics in a format suitable for external telemetry systems."""
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "service_name": self.service_name,
            "metrics": self.get_performance_summary(),
            "opentelemetry_enabled": OPENTELEMETRY_AVAILABLE,
        }
