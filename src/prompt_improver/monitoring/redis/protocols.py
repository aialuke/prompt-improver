"""Redis Health Monitoring Protocols.

Protocol definitions for Redis health monitoring services.
Enables clean dependency injection and testability.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .types import (
    RedisAlert,
    RedisAlertLevel,
    RedisConnectionInfo,
    RedisHealthConfig,
    RedisHealthMetrics,
    RedisHealthResult,
    RedisPerformanceMetrics,
)


@runtime_checkable
class RedisConnectionMonitorProtocol(Protocol):
    """Protocol for Redis connection monitoring."""
    
    @abstractmethod
    async def check_connection_health(self) -> RedisHealthResult:
        """Check Redis connection health and failover status."""
        ...
    
    @abstractmethod
    async def get_connection_info(self) -> RedisConnectionInfo:
        """Get detailed connection information."""
        ...
    
    @abstractmethod
    async def test_connectivity(self) -> bool:
        """Test basic Redis connectivity."""
        ...
    
    @abstractmethod
    async def check_failover_status(self) -> Dict[str, Any]:
        """Check Redis failover and sentinel status."""
        ...
    
    @abstractmethod
    def get_connection_metrics(self) -> Dict[str, float]:
        """Get connection performance metrics."""
        ...


@runtime_checkable
class RedisPerformanceMonitorProtocol(Protocol):
    """Protocol for Redis performance monitoring."""
    
    @abstractmethod
    async def collect_performance_metrics(self) -> RedisPerformanceMetrics:
        """Collect comprehensive Redis performance metrics."""
        ...
    
    @abstractmethod
    async def measure_latency(self) -> float:
        """Measure current Redis latency in milliseconds."""
        ...
    
    @abstractmethod
    async def get_throughput_metrics(self) -> Dict[str, float]:
        """Get Redis throughput and operations metrics."""
        ...
    
    @abstractmethod
    async def analyze_slow_queries(self) -> List[Dict[str, Any]]:
        """Analyze Redis slow query log."""
        ...
    
    @abstractmethod
    def get_performance_trends(self, hours: int = 1) -> Dict[str, List[float]]:
        """Get performance trends over time."""
        ...


@runtime_checkable
class RedisHealthCheckerProtocol(Protocol):
    """Protocol for Redis health status checking."""
    
    @abstractmethod
    async def check_overall_health(self) -> RedisHealthResult:
        """Perform comprehensive Redis health check."""
        ...
    
    @abstractmethod
    async def check_memory_health(self) -> RedisHealthResult:
        """Check Redis memory usage and fragmentation."""
        ...
    
    @abstractmethod
    async def check_persistence_health(self) -> RedisHealthResult:
        """Check Redis persistence (RDB/AOF) health."""
        ...
    
    @abstractmethod
    async def check_replication_health(self) -> RedisHealthResult:
        """Check Redis replication status and lag."""
        ...
    
    @abstractmethod
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get quick health summary for integration."""
        ...


@runtime_checkable
class RedisMetricsCollectorProtocol(Protocol):
    """Protocol for Redis metrics collection."""
    
    @abstractmethod
    async def collect_all_metrics(self) -> RedisHealthMetrics:
        """Collect all Redis health metrics."""
        ...
    
    @abstractmethod
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect Redis system-level metrics."""
        ...
    
    @abstractmethod
    async def collect_keyspace_metrics(self) -> Dict[str, Any]:
        """Collect Redis keyspace analytics."""
        ...
    
    @abstractmethod
    async def collect_client_metrics(self) -> Dict[str, Any]:
        """Collect Redis client connection metrics."""
        ...
    
    @abstractmethod
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        ...
    
    @abstractmethod
    async def validate_slo_compliance(self) -> Dict[str, bool]:
        """Validate SLO compliance for Redis metrics."""
        ...


@runtime_checkable
class RedisAlertingServiceProtocol(Protocol):
    """Protocol for Redis alerting service."""
    
    @abstractmethod
    async def process_health_result(self, result: RedisHealthResult) -> List[RedisAlert]:
        """Process health result and generate alerts."""
        ...
    
    @abstractmethod
    async def send_alert(self, alert: RedisAlert) -> bool:
        """Send Redis alert notification."""
        ...
    
    @abstractmethod
    async def check_alert_conditions(self, metrics: RedisHealthMetrics) -> List[RedisAlert]:
        """Check current metrics against alert conditions."""
        ...
    
    @abstractmethod
    def get_active_alerts(self) -> List[RedisAlert]:
        """Get currently active Redis alerts."""
        ...
    
    @abstractmethod
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a Redis alert."""
        ...
    
    @abstractmethod
    def configure_alert_rules(self, rules: Dict[str, Any]) -> None:
        """Configure Redis alerting rules."""
        ...


@runtime_checkable
class RedisHealthOrchestratorProtocol(Protocol):
    """Protocol for Redis health orchestration."""
    
    @abstractmethod
    async def run_comprehensive_health_check(self) -> RedisHealthResult:
        """Run comprehensive health check across all services."""
        ...
    
    @abstractmethod
    async def get_unified_health_report(self) -> Dict[str, Any]:
        """Get unified health report from all monitoring services."""
        ...
    
    @abstractmethod
    async def start_monitoring(self) -> None:
        """Start continuous Redis monitoring."""
        ...
    
    @abstractmethod
    async def stop_monitoring(self) -> None:
        """Stop Redis monitoring services."""
        ...
    
    @abstractmethod
    def configure_monitoring(self, config: RedisHealthConfig) -> None:
        """Configure Redis monitoring settings."""
        ...
    
    @abstractmethod
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        ...
    
    @abstractmethod
    async def emergency_health_check(self) -> RedisHealthResult:
        """Perform emergency health check with minimal overhead."""
        ...