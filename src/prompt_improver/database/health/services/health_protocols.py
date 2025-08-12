"""Protocol definitions for database health monitoring services.

Defines protocol interfaces for the decomposed health monitoring components
following clean architecture principles and dependency inversion.
"""

from datetime import datetime
from typing import Any, Dict, List, Protocol, runtime_checkable

from prompt_improver.repositories.protocols.session_manager_protocol import SessionManagerProtocol


@runtime_checkable
class DatabaseConnectionServiceProtocol(Protocol):
    """Protocol for database connection monitoring and health assessment.
    
    Handles connection pool monitoring, connection lifecycle management,
    and connection health assessment.
    """
    
    async def collect_connection_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive connection pool metrics."""
        ...
    
    async def get_connection_details(self) -> List[Dict[str, Any]]:
        """Get detailed connection information from pg_stat_activity."""
        ...
    
    async def get_pool_health_summary(self) -> Dict[str, Any]:
        """Get connection pool health summary."""
        ...
    
    def analyze_connection_ages(self, connections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze connection age distribution."""
        ...
    
    def analyze_connection_states(self, connections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze connection state distribution."""
        ...
    
    def identify_problematic_connections(self, connections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify connections that may be problematic."""
        ...


@runtime_checkable
class HealthMetricsServiceProtocol(Protocol):
    """Protocol for health metrics collection and performance tracking.
    
    Handles health metrics collection, performance tracking, and
    query analysis with optimization recommendations.
    """
    
    async def collect_query_performance_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive query performance metrics."""
        ...
    
    async def analyze_slow_queries(self) -> List[Dict[str, Any]]:
        """Analyze slow queries from pg_stat_statements."""
        ...
    
    async def analyze_frequent_queries(self) -> List[Dict[str, Any]]:
        """Identify frequently executed queries."""
        ...
    
    async def analyze_io_intensive_queries(self) -> List[Dict[str, Any]]:
        """Identify I/O intensive queries."""
        ...
    
    async def analyze_cache_performance(self) -> Dict[str, Any]:
        """Analyze database cache performance."""
        ...
    
    async def collect_storage_metrics(self) -> Dict[str, Any]:
        """Collect storage-related metrics including table sizes and bloat."""
        ...
    
    async def collect_replication_metrics(self) -> Dict[str, Any]:
        """Collect PostgreSQL replication metrics including lag monitoring."""
        ...
    
    async def collect_lock_metrics(self) -> Dict[str, Any]:
        """Collect lock-related metrics including current locks and deadlocks."""
        ...
    
    async def collect_transaction_metrics(self) -> Dict[str, Any]:
        """Collect transaction-related metrics including commit/rollback rates."""
        ...


@runtime_checkable
class AlertingServiceProtocol(Protocol):
    """Protocol for health alerting and notifications.
    
    Handles health alerting, threshold monitoring, and 
    alert escalation and management.
    """
    
    def calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall health score based on all metrics (0-100)."""
        ...
    
    def identify_health_issues(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific health issues based on metrics."""
        ...
    
    def generate_recommendations(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on metrics and identified issues."""
        ...
    
    def check_thresholds(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if any metrics exceed defined thresholds."""
        ...
    
    async def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert notification."""
        ...
    
    def set_threshold(self, metric_name: str, warning_threshold: float, critical_threshold: float) -> None:
        """Set threshold for a specific metric."""
        ...


@runtime_checkable
class HealthReportingServiceProtocol(Protocol):
    """Protocol for health reporting and dashboards.
    
    Handles health reporting, historical analysis, and report generation.
    """
    
    def add_metrics_to_history(self, metrics: Dict[str, Any]) -> None:
        """Add metrics to history for trend analysis."""
        ...
    
    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends over the specified time period."""
        ...
    
    def generate_health_report(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        ...
    
    def generate_trend_summary(self, recent_metrics: List[Dict[str, Any]]) -> str:
        """Generate a human-readable trend summary."""
        ...
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        ...
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get historical metrics data."""
        ...


@runtime_checkable
class DatabaseHealthServiceProtocol(Protocol):
    """Protocol for unified database health monitoring service.
    
    Combines all health monitoring components into a single interface
    that maintains backward compatibility with the original god object.
    """
    
    async def collect_comprehensive_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive database health metrics from all services."""
        ...
    
    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """Single unified endpoint for all database health metrics with parallel execution."""
        ...
    
    async def health_check(self) -> Dict[str, Any]:
        """Quick health check with essential metrics."""
        ...
    
    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends over the specified time period."""
        ...
    
    async def get_connection_pool_health_summary(self) -> Dict[str, Any]:
        """Get connection pool health summary."""
        ...
    
    async def analyze_query_performance(self) -> Dict[str, Any]:
        """Analyze query performance."""
        ...