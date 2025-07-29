"""APES Health Check System - PHASE 3 Implementation.
Unified plugin-based health monitoring with comprehensive observability.
"""

from .base import AggregatedHealthResult, HealthChecker, HealthResult, HealthStatus

# Modern health checkers
from .checkers import (
                   AnalyticsServiceHealthChecker,
                   DatabaseHealthChecker,
                   MCPServerHealthChecker,
                   MLServiceHealthChecker,
                   QueueHealthChecker,
                   RedisHealthChecker,
                   SystemResourcesHealthChecker,
)

# Import ML-specific health checkers
try:
    from .ml_specific_checkers import (
        MLModelHealthChecker,
        MLDataQualityChecker,
        MLTrainingHealthChecker,
        MLPerformanceHealthChecker,
    )
    ML_SPECIFIC_CHECKERS_AVAILABLE = True
except ImportError:
    ML_SPECIFIC_CHECKERS_AVAILABLE = False

from .metrics import (
                   PROMETHEUS_AVAILABLE,
                   get_health_metrics_summary,
                   instrument_health_check,
                   reset_health_metrics,
)
# Unified health system - modern implementation only
from .unified_health_system import (
    HealthCheckPlugin,
    HealthCheckCategory,
    HealthCheckPluginConfig,
    HealthProfile,
    UnifiedHealthMonitor,
    get_unified_health_monitor,
    register_health_plugin,
    create_simple_health_plugin
)

from .health_config import (
    HealthConfigurationManager,
    HealthMonitoringPolicy,
    CategoryThresholds,
    EnvironmentType,
    get_health_config,
    get_default_profile,
    get_critical_profile,
    create_category_config
)

from .plugin_adapters import (
    create_all_plugins,
    create_ml_plugins,
    create_database_plugins,
    create_redis_plugins,
    create_api_plugins,
    create_system_plugins,
    register_all_plugins
)

__all__ = [
    # Base classes and types
    "HealthChecker",
    "HealthResult",
    "HealthStatus",
    "AggregatedHealthResult",
    
    # Health checkers
    "DatabaseHealthChecker",
    "MCPServerHealthChecker",
    "AnalyticsServiceHealthChecker",
    "MLServiceHealthChecker",
    "QueueHealthChecker",
    "RedisHealthChecker",
    "SystemResourcesHealthChecker",
    
    # Metrics
    "instrument_health_check",
    "get_health_metrics_summary",
    "reset_health_metrics",
    "PROMETHEUS_AVAILABLE",
    
    # Unified health system
    "HealthCheckPlugin",
    "HealthCheckCategory",
    "HealthCheckPluginConfig",
    "HealthProfile",
    "UnifiedHealthMonitor",
    "get_unified_health_monitor",
    "register_health_plugin",
    "create_simple_health_plugin",
    
    # Configuration management
    "HealthConfigurationManager",
    "HealthMonitoringPolicy",
    "CategoryThresholds",
    "EnvironmentType",
    "get_health_config",
    "get_default_profile",
    "get_critical_profile",
    "create_category_config",
    
    # Plugin utilities
    "create_all_plugins",
    "create_ml_plugins",
    "create_database_plugins",
    "create_redis_plugins",
    "create_api_plugins",
    "create_system_plugins",
    "register_all_plugins"
]

# Add ML-specific checkers to __all__ if available
if ML_SPECIFIC_CHECKERS_AVAILABLE:
    __all__.extend([
        "MLModelHealthChecker",
        "MLDataQualityChecker",
        "MLTrainingHealthChecker",
        "MLPerformanceHealthChecker",
    ])
