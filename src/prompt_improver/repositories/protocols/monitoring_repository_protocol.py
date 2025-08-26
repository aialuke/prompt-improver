"""Monitoring repository protocol for metrics and health data operations.

Defines clean interfaces for monitoring operations without coupling to
infrastructure implementation details. Supports metrics collection and health monitoring.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class HealthStatus(StrEnum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class MetricType(StrEnum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class HealthCheckResult(BaseModel):
    """Domain model for health check results."""

    check_name: str
    status: HealthStatus
    message: str | None = None
    details: dict[str, Any] | None = None
    check_duration_ms: float
    timestamp: datetime
    tags: dict[str, str] | None = None


class MetricData(BaseModel):
    """Domain model for metric data."""

    metric_name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    tags: dict[str, str] | None = None
    unit: str | None = None
    description: str | None = None


class PerformanceMetrics(BaseModel):
    """Domain model for performance metrics."""

    component_name: str
    response_time_ms: float
    throughput_rps: float
    error_rate: float
    cpu_usage_percent: float | None = None
    memory_usage_mb: float | None = None
    active_connections: int | None = None
    queue_depth: int | None = None
    timestamp: datetime


class SystemHealthSummary(BaseModel):
    """Domain model for system health summary."""

    overall_status: HealthStatus
    component_count: int
    healthy_components: int
    degraded_components: int
    unhealthy_components: int
    last_check: datetime
    critical_issues: list[str] | None = None
    warnings: list[str] | None = None


class AlertData(BaseModel):
    """Domain model for alert data."""

    alert_id: str
    alert_name: str
    severity: str  # critical, warning, info
    component: str
    message: str
    details: dict[str, Any] | None = None
    triggered_at: datetime
    resolved_at: datetime | None = None
    tags: dict[str, str] | None = None


@runtime_checkable
class MonitoringRepositoryProtocol(Protocol):
    """Protocol for monitoring data operations without infrastructure coupling."""

    # Health Check Operations
    async def store_health_check(self, health_result: HealthCheckResult) -> bool:
        """Store health check result."""
        ...

    async def get_current_health_status(
        self,
        component_name: str | None = None
    ) -> dict[str, HealthCheckResult]:
        """Get current health status for components."""
        ...

    async def get_health_history(
        self,
        component_name: str,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 1000
    ) -> list[HealthCheckResult]:
        """Get health check history for component."""
        ...

    async def get_system_health_summary(self) -> SystemHealthSummary:
        """Get overall system health summary."""
        ...

    async def get_unhealthy_components(self) -> list[HealthCheckResult]:
        """Get all currently unhealthy components."""
        ...

    # Metrics Operations
    async def store_metric(self, metric_data: MetricData) -> bool:
        """Store metric data point."""
        ...

    async def store_metrics_batch(self, metrics: list[MetricData]) -> int:
        """Store multiple metrics, return count stored."""
        ...

    async def get_metric_values(
        self,
        metric_name: str,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        tags_filter: dict[str, str] | None = None,
        limit: int = 1000
    ) -> list[MetricData]:
        """Get metric values with optional filtering."""
        ...

    async def get_metric_summary(
        self,
        metric_name: str,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        tags_filter: dict[str, str] | None = None
    ) -> dict[str, float]:
        """Get metric summary statistics (avg, min, max, count)."""
        ...

    async def get_latest_metrics(
        self,
        metric_names: list[str],
        tags_filter: dict[str, str] | None = None
    ) -> dict[str, MetricData]:
        """Get latest values for specified metrics."""
        ...

    # Performance Metrics
    async def store_performance_metrics(
        self,
        performance_data: PerformanceMetrics
    ) -> bool:
        """Store performance metrics for a component."""
        ...

    async def get_performance_history(
        self,
        component_name: str,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 1000
    ) -> list[PerformanceMetrics]:
        """Get performance history for component."""
        ...

    async def get_performance_summary(
        self,
        component_names: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None
    ) -> dict[str, dict[str, float]]:
        """Get performance summary for components."""
        ...

    async def detect_performance_anomalies(
        self,
        component_name: str,
        metric_name: str,
        threshold_stddev: float = 2.0,
        window_hours: int = 24
    ) -> list[dict[str, Any]]:
        """Detect performance anomalies for component metric."""
        ...

    # Alert Management
    async def store_alert(self, alert_data: AlertData) -> bool:
        """Store alert data."""
        ...

    async def get_active_alerts(
        self,
        severity: str | None = None,
        component: str | None = None
    ) -> list[AlertData]:
        """Get active alerts with optional filtering."""
        ...

    async def resolve_alert(
        self,
        alert_id: str,
        resolved_at: datetime | None = None
    ) -> bool:
        """Mark alert as resolved."""
        ...

    async def get_alert_history(
        self,
        component: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 1000
    ) -> list[AlertData]:
        """Get alert history with optional filtering."""
        ...

    # Analytics and Reporting
    async def get_availability_metrics(
        self,
        component_name: str,
        date_from: datetime | None = None,
        date_to: datetime | None = None
    ) -> dict[str, float]:
        """Get availability metrics (uptime, downtime, etc.)."""
        ...

    async def get_error_rate_trends(
        self,
        component_names: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        granularity: str = "hour"  # hour, day, week
    ) -> dict[str, list[dict[str, Any]]]:
        """Get error rate trends over time."""
        ...

    async def get_performance_percentiles(
        self,
        component_name: str,
        metric_name: str,
        percentiles: list[float] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None
    ) -> dict[float, float]:
        """Get performance percentiles for component metric."""
        ...

    # Cleanup and Maintenance
    async def cleanup_old_metrics(self, days_old: int = 30) -> int:
        """Clean up old metric data, return count deleted."""
        ...

    async def cleanup_old_health_checks(self, days_old: int = 7) -> int:
        """Clean up old health check data, return count deleted."""
        ...

    async def get_monitoring_storage_metrics(self) -> dict[str, int]:
        """Get monitoring data storage usage metrics."""
        ...
