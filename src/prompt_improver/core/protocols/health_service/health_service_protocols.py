"""Health Service Protocols for Redis Health Monitoring

Protocol definitions for the decomposed Redis health monitoring components.
Each protocol defines focused responsibilities for single-purpose components.
"""

from abc import abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from dataclasses import dataclass


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class HealthCheckResult:
    """Standard health check result."""
    status: HealthStatus
    component_name: str
    message: str
    timestamp: datetime
    response_time_ms: float
    details: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class ConnectionHealth:
    """Connection health metrics."""
    connected_clients: int
    blocked_clients: int
    connection_utilization: float
    rejected_connections: int
    max_clients: int
    is_healthy: bool
    status: HealthStatus


@dataclass
class PerformanceMetrics:
    """Performance metrics data."""
    ops_per_sec: float
    hit_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    keyspace_hits: int
    keyspace_misses: int
    memory_usage_mb: float


@dataclass
class AlertConfiguration:
    """Alert configuration settings."""
    memory_threshold_percent: float = 85.0
    latency_threshold_ms: float = 100.0
    hit_rate_threshold_percent: float = 80.0
    connection_threshold_percent: float = 90.0
    enable_email_alerts: bool = False
    enable_slack_alerts: bool = False


@runtime_checkable
class HealthCheckerProtocol(Protocol):
    """Protocol for basic health checking operations."""
    
    @abstractmethod
    async def check_connectivity(self) -> HealthCheckResult:
        """Check basic Redis connectivity."""
        ...
    
    @abstractmethod
    async def check_basic_operations(self) -> HealthCheckResult:
        """Test basic Redis operations (get/set/delete)."""
        ...
    
    @abstractmethod
    async def get_server_info(self) -> Dict[str, Any]:
        """Get Redis server information."""
        ...


@runtime_checkable
class ConnectionMonitorProtocol(Protocol):
    """Protocol for connection pool monitoring."""
    
    @abstractmethod
    async def monitor_connections(self) -> ConnectionHealth:
        """Monitor connection pool health."""
        ...
    
    @abstractmethod
    async def get_connection_metrics(self) -> Dict[str, Any]:
        """Get detailed connection metrics."""
        ...
    
    @abstractmethod
    async def check_connection_limits(self) -> HealthCheckResult:
        """Check connection limits and utilization."""
        ...


@runtime_checkable
class MetricsCollectorProtocol(Protocol):
    """Protocol for performance metrics collection."""
    
    @abstractmethod
    async def collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect Redis performance metrics."""
        ...
    
    @abstractmethod
    async def collect_memory_metrics(self) -> Dict[str, Any]:
        """Collect memory usage metrics."""
        ...
    
    @abstractmethod
    async def collect_persistence_metrics(self) -> Dict[str, Any]:
        """Collect persistence health metrics."""
        ...
    
    @abstractmethod
    async def collect_replication_metrics(self) -> Dict[str, Any]:
        """Collect replication metrics."""
        ...
    
    @abstractmethod
    async def collect_slowlog_metrics(self) -> Dict[str, Any]:
        """Collect slow log metrics."""
        ...


@runtime_checkable
class AlertingServiceProtocol(Protocol):
    """Protocol for health alerting operations."""
    
    @abstractmethod
    async def process_health_results(self, results: List[HealthCheckResult]) -> None:
        """Process health check results and generate alerts."""
        ...
    
    @abstractmethod
    async def send_alert(self, alert_type: str, message: str, details: Dict[str, Any]) -> bool:
        """Send health alert notification."""
        ...
    
    @abstractmethod
    def configure_alerts(self, config: AlertConfiguration) -> None:
        """Configure alert thresholds and settings."""
        ...
    
    @abstractmethod
    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        ...


@runtime_checkable
class HealthFacadeProtocol(Protocol):
    """Protocol for unified health monitoring facade."""
    
    @abstractmethod
    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health status from all components."""
        ...
    
    @abstractmethod
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get quick health summary for integration."""
        ...
    
    @abstractmethod
    async def check_component_health(self, component: str) -> HealthCheckResult:
        """Check health of specific component."""
        ...
    
    @abstractmethod
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        ...
    
    @abstractmethod
    def configure_monitoring(self, config: Dict[str, Any]) -> None:
        """Configure monitoring settings."""
        ...