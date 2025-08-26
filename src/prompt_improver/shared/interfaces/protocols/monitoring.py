"""Monitoring and observability protocol definitions.

Consolidated protocols for health checks, metrics collection,
performance monitoring, and comprehensive observability systems.

Consolidated from:
- /monitoring/unified/protocols.py (10 protocols)
- /monitoring/redis/protocols.py (6 protocols)
- /monitoring/redis/health/protocols.py (7 protocols, 4 unique after deduplication)
- /performance/monitoring/protocols.py (5 protocols)
- /performance/monitoring/protocols_simple.py (5 duplicates, using more specific versions)

Total: 27 unique protocols across 5 categories
"""

from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncContextManager,
    Protocol,
    runtime_checkable,
)

# TYPE_CHECKING imports - these are not imported at runtime, only for static analysis
if TYPE_CHECKING:
    import coredis
    from sqlalchemy.ext.asyncio import AsyncSession

    from prompt_improver.monitoring.redis.health.types import (
        AlertEvent,
        AlertLevel,
        CircuitBreakerState,
        ConnectionPoolMetrics,
        HealthMetrics,
        PerformanceMetrics,
        RecoveryAction,
        RecoveryEvent,
    )
    from prompt_improver.monitoring.redis.types import (
        RedisAlert,
        RedisConnectionInfo,
        RedisHealthConfig,
        RedisHealthMetrics,
        RedisHealthResult,
        RedisPerformanceMetrics,
    )
    from prompt_improver.monitoring.unified.types import (
        HealthCheckResult,
        MetricPoint,
        SystemHealthSummary,
    )


# ============================================================================
# Core System Monitoring Protocols
# ============================================================================

@runtime_checkable
class HealthMonitorProtocol(Protocol):
    """Protocol for health monitoring operations."""

    @abstractmethod
    async def check_health(self) -> "HealthCheckResult":
        """Perform comprehensive health check."""
        ...

    @abstractmethod
    def register_checker(self, checker: "HealthCheckComponentProtocol") -> None:
        """Register health check component."""
        ...

    @abstractmethod
    def unregister_checker(self, component_name: str) -> bool:
        """Unregister health check component."""
        ...

    @abstractmethod
    def get_registered_checkers(self) -> list[str]:
        """Get list of registered checker names."""
        ...

    @abstractmethod
    async def get_overall_health(self) -> dict[str, Any]:
        """Get overall system health status."""
        ...


@runtime_checkable
class HealthCheckComponentProtocol(Protocol):
    """Protocol for individual health check components."""

    @abstractmethod
    async def check_health(self) -> "HealthCheckResult":
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
    async def collect_system_metrics(self) -> "list[MetricPoint]":
        """Collect system-level metrics (CPU, memory, disk)."""
        ...

    @abstractmethod
    async def collect_application_metrics(self) -> "list[MetricPoint]":
        """Collect application-level metrics (request counts, errors)."""
        ...

    @abstractmethod
    async def collect_component_metrics(self, component_name: str) -> "list[MetricPoint]":
        """Collect metrics for specific component."""
        ...

    @abstractmethod
    def record_metric(self, metric: "MetricPoint") -> None:
        """Record a single metric point."""
        ...


@runtime_checkable
class MonitoringRepositoryProtocol(Protocol):
    """Protocol for monitoring data persistence."""

    @abstractmethod
    async def store_health_result(self, result: "HealthCheckResult") -> None:
        """Store health check result."""
        ...

    @abstractmethod
    async def store_metrics(self, metrics: "list[MetricPoint]") -> None:
        """Store multiple metric points."""
        ...

    @abstractmethod
    async def get_health_history(
        self,
        component_name: str,
        hours_back: int = 24
    ) -> "list[HealthCheckResult]":
        """Get health check history for component."""
        ...

    @abstractmethod
    async def get_metrics_history(
        self,
        metric_name: str,
        hours_back: int = 24,
        tags: dict[str, str] | None = None
    ) -> "list[MetricPoint]":
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
        duration_ms: float, key: str, value_size: int | None = None
    ) -> None:
        """Record a cache operation for monitoring."""
        ...

    @abstractmethod
    async def get_cache_performance_report(self) -> dict[str, Any]:
        """Generate cache performance report."""
        ...

    @abstractmethod
    def get_cache_metrics_summary(self) -> dict[str, Any]:
        """Get summary of cache metrics."""
        ...


@runtime_checkable
class HealthReporterProtocol(Protocol):
    """Protocol for health reporting operations."""

    @abstractmethod
    async def generate_health_report(self) -> dict[str, Any]:
        """Generate comprehensive health report."""
        ...

    @abstractmethod
    def get_health_trends(self, hours: int = 24) -> list[dict[str, Any]]:
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
    def record_metric(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record a metric value."""
        ...

    @abstractmethod
    async def collect_all_metrics(self) -> "list[MetricPoint]":
        """Collect all metrics."""
        ...

    @abstractmethod
    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of collected metrics."""
        ...


@runtime_checkable
class AlertingServiceProtocol(Protocol):
    """Protocol for monitoring alerting service."""

    @abstractmethod
    async def process_health_alerts(self, health_results: "list[HealthCheckResult]") -> None:
        """Process health check results and generate alerts."""
        ...

    @abstractmethod
    async def send_alert(self, alert_type: str, message: str, details: dict[str, Any] | None = None) -> bool:
        """Send a specific alert."""
        ...

    @abstractmethod
    def get_alert_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent alert history."""
        ...


@runtime_checkable
class HealthCheckServiceProtocol(Protocol):
    """Protocol for health checking coordination."""

    @abstractmethod
    async def run_all_checks(self) -> "SystemHealthSummary":
        """Run all registered health checks."""
        ...

    @abstractmethod
    async def run_component_check(self, component_name: str) -> "HealthCheckResult":
        """Run health check for specific component."""
        ...

    @abstractmethod
    def register_component(self, checker: "HealthCheckComponentProtocol") -> None:
        """Register health check component."""
        ...

    @abstractmethod
    def unregister_component(self, component_name: str) -> bool:
        """Unregister health check component."""
        ...

    @abstractmethod
    def get_registered_components(self) -> list[str]:
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
    def get_orchestrator_status(self) -> dict[str, Any]:
        """Get orchestrator status."""
        ...


@runtime_checkable
class UnifiedMonitoringFacadeProtocol(Protocol):
    """Main protocol for the unified monitoring facade."""

    @abstractmethod
    async def get_system_health(self) -> "SystemHealthSummary":
        """Get overall system health status."""
        ...

    @abstractmethod
    async def check_component_health(self, component_name: str) -> "HealthCheckResult":
        """Check health of specific component."""
        ...

    @abstractmethod
    async def collect_all_metrics(self) -> "list[MetricPoint]":
        """Collect all available metrics."""
        ...

    @abstractmethod
    def record_custom_metric(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None
    ) -> None:
        """Record custom application metric."""
        ...

    @abstractmethod
    def register_health_checker(
        self,
        checker: "HealthCheckComponentProtocol"
    ) -> None:
        """Register custom health checker component."""
        ...

    @abstractmethod
    async def get_monitoring_summary(self) -> dict[str, Any]:
        """Get comprehensive monitoring summary."""
        ...

    @abstractmethod
    async def cleanup_old_monitoring_data(self) -> int:
        """Clean up old monitoring data."""
        ...


# ============================================================================
# Redis-Specific Monitoring Protocols
# ============================================================================

@runtime_checkable
class RedisConnectionMonitorProtocol(Protocol):
    """Protocol for Redis connection monitoring."""

    @abstractmethod
    async def check_connection_health(self) -> "RedisHealthResult":
        """Check Redis connection health and failover status."""
        ...

    @abstractmethod
    async def get_connection_info(self) -> "RedisConnectionInfo":
        """Get detailed connection information."""
        ...

    @abstractmethod
    async def test_connectivity(self) -> bool:
        """Test basic Redis connectivity."""
        ...

    @abstractmethod
    async def check_failover_status(self) -> dict[str, Any]:
        """Check Redis failover and sentinel status."""
        ...

    @abstractmethod
    def get_connection_metrics(self) -> dict[str, float]:
        """Get connection performance metrics."""
        ...


@runtime_checkable
class RedisConnectionPoolMonitorProtocol(Protocol):
    """Protocol for Redis connection pool monitoring and management."""

    @abstractmethod
    async def monitor_connections(self) -> "ConnectionPoolMetrics":
        """Monitor connection pool health and utilization."""
        ...

    @abstractmethod
    async def detect_connection_issues(self) -> list[str]:
        """Detect connection-related issues and bottlenecks."""
        ...

    @abstractmethod
    async def get_connection_stats(self) -> dict[str, Any]:
        """Get detailed connection statistics."""
        ...

    @abstractmethod
    async def validate_connection_pool(self) -> bool:
        """Validate connection pool health."""
        ...


@runtime_checkable
class RedisPerformanceMonitorProtocol(Protocol):
    """Protocol for Redis performance monitoring."""

    @abstractmethod
    async def collect_performance_metrics(self) -> "RedisPerformanceMetrics":
        """Collect comprehensive Redis performance metrics."""
        ...

    @abstractmethod
    async def measure_latency(self) -> float:
        """Measure current Redis latency in milliseconds."""
        ...

    @abstractmethod
    async def get_throughput_metrics(self) -> dict[str, float]:
        """Get Redis throughput and operations metrics."""
        ...

    @abstractmethod
    async def analyze_slow_queries(self) -> list[dict[str, Any]]:
        """Analyze Redis slow query log."""
        ...

    @abstractmethod
    def get_performance_trends(self, hours: int = 1) -> dict[str, list[float]]:
        """Get performance trends over time."""
        ...


@runtime_checkable
class RedisHealthCheckerProtocol(Protocol):
    """Protocol for Redis health status checking."""

    @abstractmethod
    async def check_overall_health(self) -> "RedisHealthResult":
        """Perform comprehensive Redis health check."""
        ...

    @abstractmethod
    async def check_memory_health(self) -> "RedisHealthResult":
        """Check Redis memory usage and fragmentation."""
        ...

    @abstractmethod
    async def check_persistence_health(self) -> "RedisHealthResult":
        """Check Redis persistence (RDB/AOF) health."""
        ...

    @abstractmethod
    async def check_replication_health(self) -> "RedisHealthResult":
        """Check Redis replication status and lag."""
        ...

    @abstractmethod
    async def get_health_summary(self) -> dict[str, Any]:
        """Get quick health summary for integration."""
        ...


@runtime_checkable
class RedisBasicHealthCheckerProtocol(Protocol):
    """Protocol for basic Redis health checking and connectivity monitoring."""

    @abstractmethod
    async def check_health(self) -> "HealthMetrics":
        """Perform comprehensive health check with <25ms operation time."""
        ...

    @abstractmethod
    async def ping(self) -> float:
        """Check Redis connectivity with latency measurement."""
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Quick availability check for circuit breaker logic."""
        ...

    @abstractmethod
    def get_last_metrics(self) -> "HealthMetrics | None":
        """Get cached health metrics for fast access."""
        ...


@runtime_checkable
class RedisMetricsCollectorProtocol(Protocol):
    """Protocol for Redis metrics collection."""

    @abstractmethod
    async def collect_all_metrics(self) -> "RedisHealthMetrics":
        """Collect all Redis health metrics."""
        ...

    @abstractmethod
    async def collect_system_metrics(self) -> dict[str, Any]:
        """Collect Redis system-level metrics."""
        ...

    @abstractmethod
    async def collect_keyspace_metrics(self) -> dict[str, Any]:
        """Collect Redis keyspace analytics."""
        ...

    @abstractmethod
    async def collect_client_metrics(self) -> dict[str, Any]:
        """Collect Redis client connection metrics."""
        ...

    @abstractmethod
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        ...

    @abstractmethod
    async def validate_slo_compliance(self) -> dict[str, bool]:
        """Validate SLO compliance for Redis metrics."""
        ...


@runtime_checkable
class RedisPerformanceAnalysisProtocol(Protocol):
    """Protocol for Redis performance metrics collection and analysis."""

    @abstractmethod
    async def collect_performance_metrics(self) -> "PerformanceMetrics":
        """Collect comprehensive performance metrics."""
        ...

    @abstractmethod
    async def collect_memory_metrics(self) -> dict[str, Any]:
        """Collect memory usage and fragmentation metrics."""
        ...

    @abstractmethod
    async def collect_throughput_metrics(self) -> dict[str, Any]:
        """Collect throughput and command statistics."""
        ...

    @abstractmethod
    async def analyze_slow_queries(self) -> list[dict[str, Any]]:
        """Analyze slow queries for performance optimization."""
        ...

    @abstractmethod
    def get_metrics_history(self, duration_minutes: int = 10) -> "list[PerformanceMetrics]":
        """Get metrics history for trend analysis."""
        ...


@runtime_checkable
class RedisAlertingServiceProtocol(Protocol):
    """Protocol for Redis alerting service (consolidated from two implementations)."""

    # Methods from first implementation
    @abstractmethod
    async def process_health_result(self, result: "RedisHealthResult") -> "list[RedisAlert]":
        """Process health result and generate alerts."""
        ...

    @abstractmethod
    async def check_alert_conditions(self, metrics: "RedisHealthMetrics") -> "list[RedisAlert]":
        """Check current metrics against alert conditions."""
        ...

    @abstractmethod
    def configure_alert_rules(self, rules: dict[str, Any]) -> None:
        """Configure Redis alerting rules."""
        ...

    # Methods from second implementation
    @abstractmethod
    async def check_thresholds(self, metrics: "HealthMetrics") -> "list[AlertEvent]":
        """Check metrics against alerting thresholds."""
        ...

    @abstractmethod
    def configure_thresholds(self, thresholds: dict[str, Any]) -> None:
        """Configure alerting thresholds."""
        ...

    # Common methods (consolidated)
    @abstractmethod
    async def send_alert(self, alert: Any) -> bool:
        """Send alert notification (supports both RedisAlert and AlertEvent)."""
        ...

    @abstractmethod
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a Redis alert."""
        ...

    @abstractmethod
    def get_active_alerts(self) -> list[Any]:
        """Get currently active Redis alerts (returns RedisAlert or AlertEvent)."""
        ...


@runtime_checkable
class RedisHealthOrchestratorProtocol(Protocol):
    """Protocol for Redis health orchestration."""

    @abstractmethod
    async def run_comprehensive_health_check(self) -> "RedisHealthResult":
        """Run comprehensive health check across all services."""
        ...

    @abstractmethod
    async def get_unified_health_report(self) -> dict[str, Any]:
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
    def configure_monitoring(self, config: "RedisHealthConfig") -> None:
        """Configure Redis monitoring settings."""
        ...

    @abstractmethod
    def get_monitoring_status(self) -> dict[str, Any]:
        """Get current monitoring status."""
        ...

    @abstractmethod
    async def emergency_health_check(self) -> "RedisHealthResult":
        """Perform emergency health check with minimal overhead."""
        ...


@runtime_checkable
class RedisRecoveryServiceProtocol(Protocol):
    """Protocol for Redis automatic recovery and circuit breaker patterns."""

    @abstractmethod
    async def attempt_recovery(self, reason: str) -> "RecoveryEvent":
        """Attempt automatic recovery from failure."""
        ...

    @abstractmethod
    async def execute_failover(self) -> bool:
        """Execute failover to backup Redis instance."""
        ...

    @abstractmethod
    def get_circuit_breaker_state(self) -> "CircuitBreakerState":
        """Get current circuit breaker state."""
        ...

    @abstractmethod
    def record_operation_result(self, success: bool) -> None:
        """Record operation result for circuit breaker logic."""
        ...

    @abstractmethod
    async def validate_recovery(self) -> bool:
        """Validate that recovery was successful."""
        ...

    @abstractmethod
    def get_recovery_history(self) -> "list[RecoveryEvent]":
        """Get history of recovery attempts."""
        ...


@runtime_checkable
class RedisHealthManagerProtocol(Protocol):
    """Protocol for unified Redis health management coordination."""

    @abstractmethod
    async def get_comprehensive_health(self) -> dict[str, Any]:
        """Get comprehensive health status from all services."""
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
    async def get_health_summary(self) -> dict[str, Any]:
        """Get quick health summary for integration."""
        ...

    @abstractmethod
    def is_healthy(self) -> bool:
        """Quick health check for load balancer integration."""
        ...

    @abstractmethod
    async def handle_incident(self, severity: "AlertLevel") -> "list[RecoveryAction]":
        """Handle incident with appropriate recovery actions."""
        ...

    @abstractmethod
    def get_monitoring_status(self) -> dict[str, Any]:
        """Get status of monitoring services."""
        ...


@runtime_checkable
class RedisClientProviderProtocol(Protocol):
    """Protocol for Redis client provisioning."""

    @abstractmethod
    async def get_client(self) -> "coredis.Redis | None":
        """Get Redis client for health monitoring."""
        ...

    @abstractmethod
    async def create_backup_client(self) -> "coredis.Redis | None":
        """Create backup Redis client for failover."""
        ...

    @abstractmethod
    async def validate_client(self, client: "coredis.Redis") -> bool:
        """Validate that Redis client is working."""
        ...


# ============================================================================
# Performance Service Protocols
# ============================================================================

@runtime_checkable
class DatabaseServiceProtocol(Protocol):
    """Protocol for database session access in performance monitoring."""

    @abstractmethod
    async def get_session(self) -> "AsyncContextManager[AsyncSession]":
        """Get database session for performance monitoring operations."""
        ...


@runtime_checkable
class PromptImprovementServiceProtocol(Protocol):
    """Protocol for prompt improvement operations in performance monitoring."""

    @abstractmethod
    async def improve_prompt(
        self,
        prompt: str,
        context: dict[str, Any],
        session_id: str,
        rate_limit_remaining: int | None = None
    ) -> Any:
        """Improve a prompt using the core prompt improvement service."""
        ...


@runtime_checkable
class ConfigurationServiceProtocol(Protocol):
    """Protocol for configuration access in performance monitoring."""

    @abstractmethod
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        ...

    @abstractmethod
    def get_performance_config(self) -> dict[str, Any]:
        """Get performance-specific configuration."""
        ...


@runtime_checkable
class MLEventBusServiceProtocol(Protocol):
    """Protocol for ML event bus access in performance monitoring."""

    @abstractmethod
    async def publish(self, event: Any) -> bool:
        """Publish event to ML event bus."""
        ...

    @abstractmethod
    async def get_event_bus(self) -> Any:
        """Get the ML event bus instance."""
        ...


@runtime_checkable
class SessionStoreServiceProtocol(Protocol):
    """Protocol for session store operations in performance monitoring."""

    @abstractmethod
    async def set(self, session_id: str, data: dict[str, Any]) -> None:
        """Set session data."""
        ...

    @abstractmethod
    async def get(self, session_id: str) -> dict[str, Any] | None:
        """Get session data."""
        ...

    @abstractmethod
    async def touch(self, session_id: str) -> None:
        """Touch session to update last access time."""
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Delete session data."""
        ...
