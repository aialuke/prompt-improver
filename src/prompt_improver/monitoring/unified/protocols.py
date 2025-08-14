"""Unified monitoring protocols.

Protocol definitions for the unified monitoring facade,
providing clean interfaces for dependency injection and testability.
"""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .types import (
    ComponentHealthChecker,
    HealthCheckResult,
    MetricPoint,
    MonitoringConfig,
    SystemHealthSummary,
)


@runtime_checkable
class HealthCheckComponentProtocol(Protocol):
    """Protocol for individual health check components."""
    
    @abstractmethod
    async def check_health(self) -> HealthCheckResult:
        """Perform health check and return result."""
        ...
    
    @abstractmethod
    def get_component_name(self) -> str:
        """Get the name of this health check component.""" 
        ...
    
    @abstractmethod
    def get_timeout_seconds(self) -> float:
        """Get timeout for this health check in seconds."""
        ...


@runtime_checkable  
class MetricsCollectionProtocol(Protocol):
    """Protocol for metrics collection operations."""
    
    @abstractmethod
    async def collect_system_metrics(self) -> List[MetricPoint]:
        """Collect system-level metrics (CPU, memory, disk)."""
        ...
    
    @abstractmethod
    async def collect_application_metrics(self) -> List[MetricPoint]:
        """Collect application-level metrics (request counts, errors)."""
        ...
    
    @abstractmethod  
    async def collect_component_metrics(self, component_name: str) -> List[MetricPoint]:
        """Collect metrics for specific component."""
        ...
    
    @abstractmethod
    def record_metric(self, metric: MetricPoint) -> None:
        """Record a single metric point."""
        ...


@runtime_checkable
class MonitoringRepositoryProtocol(Protocol):
    """Protocol for monitoring data persistence."""
    
    @abstractmethod
    async def store_health_result(self, result: HealthCheckResult) -> None:
        """Store health check result."""
        ...
    
    @abstractmethod
    async def store_metrics(self, metrics: List[MetricPoint]) -> None:
        """Store multiple metric points.""" 
        ...
    
    @abstractmethod
    async def get_health_history(
        self, 
        component_name: str, 
        hours_back: int = 24
    ) -> List[HealthCheckResult]:
        """Get health check history for component."""
        ...
    
    @abstractmethod
    async def get_metrics_history(
        self,
        metric_name: str,
        hours_back: int = 24,
        tags: Optional[Dict[str, str]] = None
    ) -> List[MetricPoint]:
        """Get metrics history."""
        ...
    
    @abstractmethod
    async def cleanup_old_data(self, retention_hours: int) -> int:
        """Clean up old monitoring data, return number of records removed."""
        ...


@runtime_checkable
class CacheMonitoringProtocol(Protocol):
    """Protocol for cache monitoring operations."""
    
    @abstractmethod
    def record_cache_operation(
        self, operation: str, cache_level: str, hit: bool,
        duration_ms: float, key: str, value_size: Optional[int] = None
    ) -> None:
        """Record a cache operation for monitoring."""
        ...
    
    @abstractmethod
    async def get_cache_performance_report(self) -> Dict[str, Any]:
        """Generate cache performance report."""
        ...
    
    @abstractmethod
    def get_cache_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of cache metrics."""
        ...


@runtime_checkable
class HealthReporterProtocol(Protocol):
    """Protocol for health reporting operations."""
    
    @abstractmethod
    async def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        ...
    
    @abstractmethod
    def get_health_trends(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health trends over time."""
        ...
    
    @abstractmethod
    async def export_health_data(self, format: str = "json") -> str:
        """Export health data in specified format."""
        ...


@runtime_checkable
class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collection operations."""
    
    @abstractmethod
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        ...
    
    @abstractmethod
    async def collect_all_metrics(self) -> List[MetricPoint]:
        """Collect all metrics."""
        ...
    
    @abstractmethod
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        ...


@runtime_checkable
class AlertingServiceProtocol(Protocol):
    """Protocol for monitoring alerting service."""
    
    @abstractmethod
    async def process_health_alerts(self, health_results: List[HealthCheckResult]) -> None:
        """Process health check results and generate alerts."""
        ...
    
    @abstractmethod
    async def send_alert(self, alert_type: str, message: str, details: Dict[str, Any] = None) -> bool:
        """Send a specific alert."""
        ...
    
    @abstractmethod
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        ...


@runtime_checkable
class HealthCheckServiceProtocol(Protocol):
    """Protocol for health checking coordination."""
    
    @abstractmethod
    async def run_all_checks(self) -> SystemHealthSummary:
        """Run all registered health checks."""
        ...
    
    @abstractmethod  
    async def run_component_check(self, component_name: str) -> HealthCheckResult:
        """Run health check for specific component."""
        ...
    
    @abstractmethod
    def register_component(self, checker: HealthCheckComponentProtocol) -> None:
        """Register health check component."""
        ...
    
    @abstractmethod
    def unregister_component(self, component_name: str) -> bool:
        """Unregister health check component."""
        ...
    
    @abstractmethod
    def get_registered_components(self) -> List[str]:
        """Get list of registered component names."""
        ...


@runtime_checkable
class MonitoringOrchestratorProtocol(Protocol):
    """Protocol for monitoring orchestration operations."""
    
    @abstractmethod
    async def start_monitoring(self) -> None:
        """Start monitoring operations."""
        ...
    
    @abstractmethod
    async def stop_monitoring(self) -> None:
        """Stop monitoring operations."""
        ...
    
    @abstractmethod
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        ...


@runtime_checkable
class UnifiedMonitoringFacadeProtocol(Protocol):
    """Main protocol for the unified monitoring facade."""
    
    @abstractmethod
    async def get_system_health(self) -> SystemHealthSummary:
        """Get overall system health status."""
        ...
    
    @abstractmethod
    async def check_component_health(self, component_name: str) -> HealthCheckResult:
        """Check health of specific component."""
        ...
    
    @abstractmethod
    async def collect_all_metrics(self) -> List[MetricPoint]:
        """Collect all available metrics."""
        ...
    
    @abstractmethod
    def record_custom_metric(
        self, 
        name: str, 
        value: float, 
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record custom application metric."""
        ...
    
    @abstractmethod
    def register_health_checker(
        self, 
        checker: HealthCheckComponentProtocol
    ) -> None:
        """Register custom health checker component."""
        ...
    
    @abstractmethod
    async def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        ...
    
    @abstractmethod
    async def cleanup_old_monitoring_data(self) -> int:
        """Clean up old monitoring data."""
        ...