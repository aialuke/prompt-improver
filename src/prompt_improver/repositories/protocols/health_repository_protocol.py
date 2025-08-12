"""Health repository protocol for system health monitoring and diagnostics.

Defines the interface for health check operations, including:
- Database health monitoring
- Connection pool status
- Performance metrics
- System diagnostics
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel


class HealthStatus(BaseModel):
    """Health check status information."""

    component: str
    status: str  # "healthy", "warning", "critical", "unknown"
    timestamp: datetime
    response_time_ms: float | None
    details: dict[str, Any] | None
    error_message: str | None


class DatabaseHealthMetrics(BaseModel):
    """Database-specific health metrics."""

    connection_pool_size: int
    active_connections: int
    idle_connections: int
    max_connections: int
    connection_utilization: float
    avg_query_time_ms: float
    slow_query_count: int
    deadlock_count: int
    table_sizes: dict[str, int]
    index_usage_stats: dict[str, dict[str, Any]]


class SystemHealthSummary(BaseModel):
    """Overall system health summary."""

    overall_status: str
    components_checked: int
    healthy_components: int
    warning_components: int
    critical_components: int
    last_check_time: datetime
    uptime_seconds: float
    performance_score: float
    recommendations: list[str]


class PerformanceMetrics(BaseModel):
    """System performance metrics."""

    cpu_usage_percent: float
    memory_usage_mb: float
    disk_usage_percent: float
    network_io_mbps: float
    database_connections: int
    cache_hit_ratio: float
    avg_response_time_ms: float
    requests_per_second: float


class HealthAlert(BaseModel):
    """Health monitoring alert."""

    alert_id: str
    component: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    details: dict[str, Any]
    first_occurred: datetime
    last_occurred: datetime
    occurrence_count: int
    is_resolved: bool
    resolved_at: datetime | None


@runtime_checkable
class HealthRepositoryProtocol(Protocol):
    """Protocol for health monitoring data access operations."""

    # Basic Health Checks
    async def check_database_health(self) -> HealthStatus:
        """Perform comprehensive database health check."""
        ...

    async def check_connection_pool_health(self) -> HealthStatus:
        """Check database connection pool health."""
        ...

    async def check_cache_health(self) -> HealthStatus:
        """Check cache system health."""
        ...

    async def check_external_services_health(self) -> list[HealthStatus]:
        """Check health of external service dependencies."""
        ...

    async def perform_full_health_check(self) -> SystemHealthSummary:
        """Perform comprehensive system health check."""
        ...

    # Database-Specific Health Monitoring
    async def get_database_metrics(self) -> DatabaseHealthMetrics:
        """Get detailed database health metrics."""
        ...

    async def check_table_health(
        self, table_names: list[str] | None = None
    ) -> dict[str, HealthStatus]:
        """Check health of specific database tables."""
        ...

    async def check_index_health(self) -> dict[str, dict[str, Any]]:
        """Check database index usage and health."""
        ...

    async def detect_table_bloat(
        self, bloat_threshold_percent: float = 20.0
    ) -> list[dict[str, Any]]:
        """Detect table bloat issues."""
        ...

    async def analyze_slow_queries(
        self, min_duration_ms: int = 100, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Analyze slow-performing queries."""
        ...

    # Performance Monitoring
    async def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics."""
        ...

    async def get_performance_history(
        self,
        hours_back: int = 24,
        granularity: str = "hour",  # "minute", "hour", "day"
    ) -> list[dict[str, Any]]:
        """Get historical performance data."""
        ...

    async def get_performance_trends(
        self, metric_name: str, days_back: int = 7
    ) -> list[dict[str, Any]]:
        """Get performance trends for specific metric."""
        ...

    async def detect_performance_anomalies(
        self, metric_name: str, threshold_stddev: float = 2.0, lookback_hours: int = 24
    ) -> list[dict[str, Any]]:
        """Detect performance anomalies."""
        ...

    # Connection Monitoring
    async def get_active_connections_info(self) -> list[dict[str, Any]]:
        """Get information about active database connections."""
        ...

    async def get_connection_pool_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        ...

    async def check_connection_leaks(
        self, max_connection_age_minutes: int = 30
    ) -> list[dict[str, Any]]:
        """Check for potential connection leaks."""
        ...

    async def monitor_connection_patterns(self, hours_back: int = 4) -> dict[str, Any]:
        """Monitor connection usage patterns."""
        ...

    # Alert Management
    async def create_health_alert(self, alert_data: dict[str, Any]) -> HealthAlert:
        """Create a health monitoring alert."""
        ...

    async def get_active_alerts(
        self, severity: str | None = None, component: str | None = None
    ) -> list[HealthAlert]:
        """Get active health alerts."""
        ...

    async def resolve_alert(
        self, alert_id: str, resolution_notes: str | None = None
    ) -> bool:
        """Resolve a health alert."""
        ...

    async def get_alert_history(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        component: str | None = None,
    ) -> list[HealthAlert]:
        """Get health alert history."""
        ...

    # Diagnostic Tools
    async def run_database_diagnostics(self) -> dict[str, Any]:
        """Run comprehensive database diagnostics."""
        ...

    async def analyze_query_patterns(self, hours_back: int = 2) -> dict[str, Any]:
        """Analyze database query patterns."""
        ...

    async def check_data_integrity(
        self, table_names: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Check data integrity for specified tables."""
        ...

    async def validate_foreign_key_constraints(self) -> list[dict[str, Any]]:
        """Validate all foreign key constraints."""
        ...

    async def check_disk_usage(self) -> dict[str, Any]:
        """Check database disk usage by table/index."""
        ...

    # Maintenance Operations
    async def run_table_maintenance(
        self,
        table_name: str,
        operation: str,  # "vacuum", "reindex", "analyze"
    ) -> dict[str, Any]:
        """Run maintenance operation on table."""
        ...

    async def optimize_database_performance(self) -> dict[str, list[str]]:
        """Get database optimization recommendations."""
        ...

    async def cleanup_old_health_data(self, days_to_keep: int = 90) -> int:
        """Clean up old health monitoring data."""
        ...

    # Reporting and Export
    async def generate_health_report(
        self,
        report_type: str,  # "summary", "detailed", "performance"
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, Any]:
        """Generate health monitoring report."""
        ...

    async def export_health_metrics(
        self,
        format_type: str,  # "csv", "json", "prometheus"
        date_from: datetime,
        date_to: datetime,
    ) -> bytes:
        """Export health metrics in specified format."""
        ...

    # Advanced Monitoring
    async def setup_health_monitoring(
        self, check_interval_seconds: int = 60, alert_thresholds: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Setup continuous health monitoring."""
        ...

    async def get_system_capacity_analysis(self) -> dict[str, Any]:
        """Analyze system capacity and scaling needs."""
        ...

    async def predict_system_load(self, hours_ahead: int = 4) -> dict[str, Any]:
        """Predict future system load based on patterns."""
        ...
