"""Unified Monitoring System.

Consolidates 8+ health checkers and 5+ metrics systems into a single
UnifiedMonitoringFacade following clean architecture principles.

Main Components:
- UnifiedMonitoringFacade: Main facade providing unified interface
- HealthCheckService: Internal service for health checking coordination
- MetricsCollectionService: Internal service for metrics collection
- MonitoringRepository: Repository for persistence using database services
- Health checker components for database, Redis, ML models, system resources

Key Benefits:
- Single clean interface for all monitoring operations
- Protocol-based design for dependency injection
- Performance optimized (<10ms per operation)
- Maintains all existing monitoring capabilities
- Clean architecture with proper separation of concerns
"""

from prompt_improver.monitoring.unified.facade import UnifiedMonitoringFacade
from prompt_improver.monitoring.unified.types import (
    ComponentCategory,
    ComponentHealthChecker,
    HealthCheckResult,
    HealthStatus,
    MetricPoint,
    MetricType,
    MonitoringConfig,
    SystemHealthSummary,
)
from prompt_improver.shared.interfaces.protocols.monitoring import (
    AlertingServiceProtocol,
    CacheMonitoringProtocol,
    HealthCheckComponentProtocol,
    HealthCheckServiceProtocol,
    HealthReporterProtocol,
    MetricsCollectionProtocol,
    MetricsCollectorProtocol,
    MonitoringOrchestratorProtocol,
    MonitoringRepositoryProtocol,
    UnifiedMonitoringFacadeProtocol,
)


# Main factory functions
async def create_monitoring_facade(
    config: MonitoringConfig | None = None,
    manager_mode: str = "high_availability"
) -> UnifiedMonitoringFacade:
    """Create a unified monitoring facade with default configuration.

    Args:
        config: Optional monitoring configuration
        manager_mode: Database manager mode ('high_availability', 'standard', etc.)

    Returns:
        Configured UnifiedMonitoringFacade instance
    """
    from prompt_improver.database import ManagerMode

    # Convert string to enum
    mode_mapping = {
        "high_availability": ManagerMode.HIGH_AVAILABILITY,
        "standard": ManagerMode.STANDARD,
        "basic": ManagerMode.BASIC,
    }

    manager_mode_enum = mode_mapping.get(manager_mode, ManagerMode.HIGH_AVAILABILITY)

    return UnifiedMonitoringFacade(
        config=config or MonitoringConfig(),
        manager_mode=manager_mode_enum
    )


def create_monitoring_config(
    health_check_timeout_seconds: float = 30.0,
    parallel_enabled: bool = True,
    max_concurrent_checks: int = 10,
    metrics_enabled: bool = True,
    cache_results_seconds: int = 60,
    **kwargs
) -> MonitoringConfig:
    """Create monitoring configuration with common settings.

    Args:
        health_check_timeout_seconds: Timeout for health checks
        parallel_enabled: Whether to run health checks in parallel
        max_concurrent_checks: Maximum concurrent health checks
        metrics_enabled: Whether to enable metrics collection
        cache_results_seconds: How long to cache health results
        **kwargs: Additional configuration options

    Returns:
        MonitoringConfig instance
    """
    return MonitoringConfig(
        health_check_timeout_seconds=health_check_timeout_seconds,
        health_check_parallel_enabled=parallel_enabled,
        max_concurrent_checks=max_concurrent_checks,
        metrics_collection_enabled=metrics_enabled,
        cache_health_results_seconds=cache_results_seconds,
        **kwargs
    )


# Compatibility exports for existing code
from prompt_improver.monitoring.unified.health_checkers import (
    DatabaseHealthChecker,
    MLModelsHealthChecker,
    RedisHealthChecker,
    SystemResourcesHealthChecker,
)

__all__ = [
    # Protocols
    "AlertingServiceProtocol",
    "CacheMonitoringProtocol",
    # Types
    "ComponentCategory",
    "ComponentHealthChecker",
    # Health checkers
    "DatabaseHealthChecker",
    "HealthCheckComponentProtocol",
    "HealthCheckResult",
    "HealthCheckServiceProtocol",
    "HealthReporterProtocol",
    "HealthStatus",
    "MLModelsHealthChecker",
    "MetricPoint",
    "MetricType",
    "MetricsCollectionProtocol",
    "MetricsCollectorProtocol",
    "MonitoringConfig",
    "MonitoringOrchestratorProtocol",
    "MonitoringRepositoryProtocol",
    "RedisHealthChecker",
    "SystemHealthSummary",
    "SystemResourcesHealthChecker",
    # Main facade
    "UnifiedMonitoringFacade",
    "UnifiedMonitoringFacadeProtocol",
    "create_monitoring_config",
    # Factory functions
    "create_monitoring_facade",
]
