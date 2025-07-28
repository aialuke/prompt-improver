"""
Protocol definitions for monitoring and health check operations.

Provides type-safe interface contracts for monitoring systems,
enabling dependency inversion and improved testability.
"""

from typing import Protocol, Any, Dict, Optional
from datetime import datetime
from enum import Enum

class HealthStatus(Enum):
    """Health check status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class HealthCheckResult:
    """Standardized health check result"""
    
    def __init__(
        self, 
        status: HealthStatus,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        self.status = status
        self.message = message
        self.details = details or {}
        self.timestamp = timestamp

class BasicHealthCheckProtocol(Protocol):
    """Protocol for basic health check operations"""
    
    async def check_health(self) -> HealthCheckResult:
        """Perform health check"""
        ...
    
    def get_check_name(self) -> str:
        """Get the name of this health check"""
        ...
    
    def get_timeout(self) -> int:
        """Get health check timeout in seconds"""
        ...

class AdvancedHealthCheckProtocol(Protocol):
    """Protocol for advanced health check operations"""
    
    async def check_dependencies(self) -> Dict[str, HealthCheckResult]:
        """Check health of dependencies"""
        ...
    
    async def get_health_metrics(self) -> Dict[str, Any]:
        """Get detailed health metrics"""
        ...
    
    async def perform_deep_check(self) -> HealthCheckResult:
        """Perform comprehensive health check"""
        ...

class PerformanceMonitorProtocol(Protocol):
    """Protocol for performance monitoring"""
    
    async def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a performance metric"""
        ...
    
    async def record_timer(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record timing metric"""
        ...
    
    async def record_counter(self, name: str, count: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Record counter metric"""
        ...
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        ...

class CircuitBreakerProtocol(Protocol):
    """Protocol for circuit breaker operations"""
    
    async def call(self, func, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        ...
    
    def get_state(self) -> str:
        """Get current circuit breaker state"""
        ...
    
    def get_failure_count(self) -> int:
        """Get current failure count"""
        ...
    
    def reset(self) -> None:
        """Reset circuit breaker"""
        ...

class AlertingProtocol(Protocol):
    """Protocol for alerting operations"""
    
    async def send_alert(self, level: str, message: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Send alert notification"""
        ...
    
    async def send_health_alert(self, check_name: str, result: HealthCheckResult) -> bool:
        """Send health check alert"""
        ...
    
    def configure_alert_rules(self, rules: Dict[str, Any]) -> None:
        """Configure alerting rules"""
        ...

class SLAMonitorProtocol(Protocol):
    """Protocol for SLA monitoring"""
    
    async def record_sla_event(self, service: str, success: bool, response_time: float) -> None:
        """Record SLA event"""
        ...
    
    async def get_sla_metrics(self, service: str, timeframe: str) -> Dict[str, Any]:
        """Get SLA metrics for service"""
        ...
    
    async def check_sla_compliance(self, service: str) -> bool:
        """Check if service meets SLA requirements"""
        ...

class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collection"""
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics"""
        ...
    
    async def collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-level metrics"""
        ...
    
    async def collect_business_metrics(self) -> Dict[str, Any]:
        """Collect business-level metrics"""
        ...
    
    async def export_metrics(self, format: str = "prometheus") -> str:
        """Export metrics in specified format"""
        ...

class HealthServiceProtocol(
    BasicHealthCheckProtocol,
    AdvancedHealthCheckProtocol,
    PerformanceMonitorProtocol,
    CircuitBreakerProtocol,
    AlertingProtocol,
    SLAMonitorProtocol,
    MetricsCollectorProtocol
):
    """Combined protocol for comprehensive health monitoring"""
    
    async def get_overall_health(self) -> HealthCheckResult:
        """Get overall system health status"""
        ...
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all configured health checks"""
        ...
    
    def register_health_check(self, check: BasicHealthCheckProtocol) -> None:
        """Register a new health check"""
        ...
    
    def unregister_health_check(self, check_name: str) -> None:
        """Unregister a health check"""
        ...