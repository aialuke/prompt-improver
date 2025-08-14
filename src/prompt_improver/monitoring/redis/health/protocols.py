"""Redis Health Monitoring Protocols

Protocol interfaces for Redis health monitoring services following SRE best practices.
All protocols are designed for dependency injection and service composition.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol
import coredis

from .types import (
    AlertEvent,
    AlertLevel,
    CircuitBreakerState,
    ConnectionPoolMetrics,
    HealthMetrics,
    PerformanceMetrics,
    RecoveryAction,
    RecoveryEvent,
    RedisHealthStatus,
)


class RedisHealthCheckerProtocol(Protocol):
    """Protocol for Redis health checking and connectivity monitoring."""
    
    @abstractmethod
    async def check_health(self) -> HealthMetrics:
        """Perform comprehensive health check.
        
        Returns:
            Current health metrics with <25ms operation time
        """
        ...
    
    @abstractmethod
    async def ping(self) -> float:
        """Check Redis connectivity with latency measurement.
        
        Returns:
            Ping latency in milliseconds
        """
        ...
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Quick availability check for circuit breaker logic.
        
        Returns:
            True if Redis is available
        """
        ...
    
    @abstractmethod
    def get_last_metrics(self) -> Optional[HealthMetrics]:
        """Get cached health metrics for fast access.
        
        Returns:
            Last collected health metrics
        """
        ...


class RedisConnectionMonitorProtocol(Protocol):
    """Protocol for Redis connection pool monitoring and management."""
    
    @abstractmethod
    async def monitor_connections(self) -> ConnectionPoolMetrics:
        """Monitor connection pool health and utilization.
        
        Returns:
            Current connection pool metrics
        """
        ...
    
    @abstractmethod
    async def detect_connection_issues(self) -> List[str]:
        """Detect connection-related issues and bottlenecks.
        
        Returns:
            List of detected issues
        """
        ...
    
    @abstractmethod
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get detailed connection statistics.
        
        Returns:
            Connection statistics dictionary
        """
        ...
    
    @abstractmethod
    async def validate_connection_pool(self) -> bool:
        """Validate connection pool health.
        
        Returns:
            True if connection pool is healthy
        """
        ...


class RedisMetricsCollectorProtocol(Protocol):
    """Protocol for Redis performance metrics collection and analysis."""
    
    @abstractmethod
    async def collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics.
        
        Returns:
            Current performance metrics
        """
        ...
    
    @abstractmethod
    async def collect_memory_metrics(self) -> Dict[str, Any]:
        """Collect memory usage and fragmentation metrics.
        
        Returns:
            Memory metrics dictionary
        """
        ...
    
    @abstractmethod
    async def collect_throughput_metrics(self) -> Dict[str, Any]:
        """Collect throughput and command statistics.
        
        Returns:
            Throughput metrics dictionary
        """
        ...
    
    @abstractmethod
    async def analyze_slow_queries(self) -> List[Dict[str, Any]]:
        """Analyze slow queries for performance optimization.
        
        Returns:
            List of slow query analysis results
        """
        ...
    
    @abstractmethod
    def get_metrics_history(self, duration_minutes: int = 10) -> List[PerformanceMetrics]:
        """Get metrics history for trend analysis.
        
        Args:
            duration_minutes: How far back to retrieve metrics
            
        Returns:
            List of historical performance metrics
        """
        ...


class RedisAlertingServiceProtocol(Protocol):
    """Protocol for Redis incident detection and alerting."""
    
    @abstractmethod
    async def check_thresholds(self, metrics: HealthMetrics) -> List[AlertEvent]:
        """Check metrics against alerting thresholds.
        
        Args:
            metrics: Current health metrics to evaluate
            
        Returns:
            List of triggered alert events
        """
        ...
    
    @abstractmethod
    async def send_alert(self, alert: AlertEvent) -> bool:
        """Send alert notification through configured channels.
        
        Args:
            alert: Alert event to send
            
        Returns:
            True if alert was sent successfully
        """
        ...
    
    @abstractmethod
    async def resolve_alert(self, alert_id: str) -> bool:
        """Mark alert as resolved.
        
        Args:
            alert_id: ID of alert to resolve
            
        Returns:
            True if alert was resolved successfully
        """
        ...
    
    @abstractmethod
    def get_active_alerts(self) -> List[AlertEvent]:
        """Get all active unresolved alerts.
        
        Returns:
            List of active alert events
        """
        ...
    
    @abstractmethod
    def configure_thresholds(self, thresholds: Dict[str, Any]) -> None:
        """Configure alerting thresholds.
        
        Args:
            thresholds: Threshold configuration dictionary
        """
        ...


class RedisRecoveryServiceProtocol(Protocol):
    """Protocol for Redis automatic recovery and circuit breaker patterns."""
    
    @abstractmethod
    async def attempt_recovery(self, reason: str) -> RecoveryEvent:
        """Attempt automatic recovery from failure.
        
        Args:
            reason: Reason for recovery attempt
            
        Returns:
            Recovery event with results
        """
        ...
    
    @abstractmethod
    async def execute_failover(self) -> bool:
        """Execute failover to backup Redis instance.
        
        Returns:
            True if failover was successful
        """
        ...
    
    @abstractmethod
    def get_circuit_breaker_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state.
        
        Returns:
            Current circuit breaker state
        """
        ...
    
    @abstractmethod
    def record_operation_result(self, success: bool) -> None:
        """Record operation result for circuit breaker logic.
        
        Args:
            success: Whether the operation was successful
        """
        ...
    
    @abstractmethod
    async def validate_recovery(self) -> bool:
        """Validate that recovery was successful.
        
        Returns:
            True if system has recovered
        """
        ...
    
    @abstractmethod
    def get_recovery_history(self) -> List[RecoveryEvent]:
        """Get history of recovery attempts.
        
        Returns:
            List of recovery events
        """
        ...


class RedisHealthManagerProtocol(Protocol):
    """Protocol for unified Redis health management coordination."""
    
    @abstractmethod
    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health status from all services.
        
        Returns:
            Complete health status dictionary
        """
        ...
    
    @abstractmethod
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        ...
    
    @abstractmethod
    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        ...
    
    @abstractmethod
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get quick health summary for integration.
        
        Returns:
            Health summary dictionary
        """
        ...
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Quick health check for load balancer integration.
        
        Returns:
            True if Redis is healthy overall
        """
        ...
    
    @abstractmethod
    async def handle_incident(self, severity: AlertLevel) -> List[RecoveryAction]:
        """Handle incident with appropriate recovery actions.
        
        Args:
            severity: Incident severity level
            
        Returns:
            List of recovery actions taken
        """
        ...
    
    @abstractmethod
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get status of monitoring services.
        
        Returns:
            Monitoring services status
        """
        ...


class RedisClientProviderProtocol(Protocol):
    """Protocol for Redis client provisioning."""
    
    @abstractmethod
    async def get_client(self) -> Optional[coredis.Redis]:
        """Get Redis client for health monitoring.
        
        Returns:
            Redis client instance or None if unavailable
        """
        ...
    
    @abstractmethod
    async def create_backup_client(self) -> Optional[coredis.Redis]:
        """Create backup Redis client for failover.
        
        Returns:
            Backup Redis client instance or None
        """
        ...
    
    @abstractmethod
    async def validate_client(self, client: coredis.Redis) -> bool:
        """Validate that Redis client is working.
        
        Args:
            client: Redis client to validate
            
        Returns:
            True if client is working
        """
        ...