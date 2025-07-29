"""
Health Check Configuration Management

Provides centralized configuration for health monitoring:
- Environment-specific health check profiles
- Plugin configuration templates
- Performance tuning parameters
- Monitoring policies and thresholds
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Any
from enum import Enum

from .unified_health_system import (
    HealthCheckCategory,
    HealthCheckPluginConfig, 
    HealthProfile
)


class EnvironmentType(Enum):
    """Environment types for different deployment contexts"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"


@dataclass
class HealthMonitoringPolicy:
    """Health monitoring policy configuration"""
    enabled: bool = True
    global_timeout_seconds: float = 30.0
    parallel_execution: bool = True
    max_concurrent_checks: int = 10
    failure_retry_count: int = 1
    failure_retry_delay_seconds: float = 2.0
    critical_alert_threshold: int = 1  # Number of critical failures before alert
    degraded_alert_threshold: int = 3  # Number of degraded checks before alert
    health_check_interval_seconds: float = 60.0  # Default periodic check interval
    metrics_retention_hours: int = 24
    enable_telemetry: bool = True
    enable_circuit_breaker: bool = True
    enable_sla_monitoring: bool = True


@dataclass 
class CategoryThresholds:
    """Performance thresholds for health check categories"""
    timeout_seconds: float = 10.0
    critical_response_time_ms: float = 5000.0
    warning_response_time_ms: float = 2000.0
    max_failure_rate: float = 0.1  # 10% failure rate threshold
    retry_count: int = 1
    retry_delay_seconds: float = 1.0


class HealthConfigurationManager:
    """
    Manages health check configuration across environments
    """
    
    def __init__(self, environment: Optional[EnvironmentType] = None):
        self.environment = environment or self._detect_environment()
        self._policies = self._create_default_policies()
        self._category_thresholds = self._create_default_thresholds()
        self._profiles = self._create_default_profiles()
        
    def _detect_environment(self) -> EnvironmentType:
        """Auto-detect environment from environment variables"""
        env_name = os.getenv('ENVIRONMENT', os.getenv('ENV', 'development')).lower()
        
        if env_name in ['prod', 'production']:
            return EnvironmentType.PRODUCTION
        elif env_name in ['stage', 'staging']:
            return EnvironmentType.STAGING
        elif env_name in ['test', 'testing']:
            return EnvironmentType.TESTING
        elif env_name in ['local']:
            return EnvironmentType.LOCAL
        else:
            return EnvironmentType.DEVELOPMENT
    
    def _create_default_policies(self) -> Dict[EnvironmentType, HealthMonitoringPolicy]:
        """Create default monitoring policies for each environment"""
        return {
            EnvironmentType.DEVELOPMENT: HealthMonitoringPolicy(
                global_timeout_seconds=60.0,
                parallel_execution=True,
                max_concurrent_checks=5,
                failure_retry_count=0,  # No retries in dev
                health_check_interval_seconds=120.0,  # Check every 2 minutes
                enable_telemetry=False,  # Disable telemetry in dev
                enable_circuit_breaker=False,  # Disable circuit breaker in dev
                enable_sla_monitoring=False
            ),
            EnvironmentType.TESTING: HealthMonitoringPolicy(
                global_timeout_seconds=30.0,
                parallel_execution=True,
                max_concurrent_checks=8,
                failure_retry_count=1,
                health_check_interval_seconds=60.0,
                enable_telemetry=True,
                enable_circuit_breaker=True,
                enable_sla_monitoring=True
            ),
            EnvironmentType.STAGING: HealthMonitoringPolicy(
                global_timeout_seconds=25.0,
                parallel_execution=True,
                max_concurrent_checks=12,
                failure_retry_count=2,
                critical_alert_threshold=1,
                degraded_alert_threshold=2,
                health_check_interval_seconds=45.0,
                enable_telemetry=True,
                enable_circuit_breaker=True,
                enable_sla_monitoring=True
            ),
            EnvironmentType.PRODUCTION: HealthMonitoringPolicy(
                global_timeout_seconds=20.0,
                parallel_execution=True,
                max_concurrent_checks=15,
                failure_retry_count=3,
                failure_retry_delay_seconds=1.0,
                critical_alert_threshold=1,
                degraded_alert_threshold=2,
                health_check_interval_seconds=30.0,  # Check every 30 seconds
                metrics_retention_hours=72,  # Keep metrics for 3 days
                enable_telemetry=True,
                enable_circuit_breaker=True,
                enable_sla_monitoring=True
            ),
            EnvironmentType.LOCAL: HealthMonitoringPolicy(
                global_timeout_seconds=45.0,
                parallel_execution=False,  # Sequential for easier debugging
                max_concurrent_checks=3,
                failure_retry_count=0,
                health_check_interval_seconds=300.0,  # Check every 5 minutes
                enable_telemetry=False,
                enable_circuit_breaker=False,
                enable_sla_monitoring=False
            )
        }
    
    def _create_default_thresholds(self) -> Dict[HealthCheckCategory, CategoryThresholds]:
        """Create default thresholds for each health check category"""
        base_thresholds = {
            EnvironmentType.DEVELOPMENT: {
                HealthCheckCategory.ML: CategoryThresholds(
                    timeout_seconds=30.0,
                    critical_response_time_ms=10000.0,
                    warning_response_time_ms=5000.0,
                    retry_count=0
                ),
                HealthCheckCategory.DATABASE: CategoryThresholds(
                    timeout_seconds=15.0,
                    critical_response_time_ms=3000.0,
                    warning_response_time_ms=1000.0,
                    retry_count=1
                ),
                HealthCheckCategory.REDIS: CategoryThresholds(
                    timeout_seconds=10.0,
                    critical_response_time_ms=1000.0,
                    warning_response_time_ms=500.0,
                    retry_count=1
                ),
                HealthCheckCategory.API: CategoryThresholds(
                    timeout_seconds=20.0,
                    critical_response_time_ms=5000.0,
                    warning_response_time_ms=2000.0,
                    retry_count=1
                ),
                HealthCheckCategory.SYSTEM: CategoryThresholds(
                    timeout_seconds=10.0,
                    critical_response_time_ms=2000.0,
                    warning_response_time_ms=1000.0,
                    retry_count=0
                ),
                HealthCheckCategory.EXTERNAL: CategoryThresholds(
                    timeout_seconds=30.0,
                    critical_response_time_ms=15000.0,
                    warning_response_time_ms=10000.0,
                    retry_count=2
                ),
                HealthCheckCategory.CUSTOM: CategoryThresholds(
                    timeout_seconds=15.0,
                    retry_count=1
                )
            }
        }
        
        # Production has stricter thresholds
        production_adjustments = {
            HealthCheckCategory.ML: CategoryThresholds(
                timeout_seconds=20.0,
                critical_response_time_ms=5000.0,
                warning_response_time_ms=2000.0,
                retry_count=2
            ),
            HealthCheckCategory.DATABASE: CategoryThresholds(
                timeout_seconds=10.0,
                critical_response_time_ms=1000.0,
                warning_response_time_ms=500.0,
                retry_count=2
            ),
            HealthCheckCategory.REDIS: CategoryThresholds(
                timeout_seconds=5.0,
                critical_response_time_ms=500.0,
                warning_response_time_ms=200.0,
                retry_count=2
            ),
            HealthCheckCategory.API: CategoryThresholds(
                timeout_seconds=15.0,
                critical_response_time_ms=3000.0,
                warning_response_time_ms=1000.0,
                retry_count=2
            ),
            HealthCheckCategory.SYSTEM: CategoryThresholds(
                timeout_seconds=8.0,
                critical_response_time_ms=1000.0,
                warning_response_time_ms=500.0,
                retry_count=1
            ),
            HealthCheckCategory.EXTERNAL: CategoryThresholds(
                timeout_seconds=25.0,
                critical_response_time_ms=10000.0,
                warning_response_time_ms=5000.0,
                retry_count=3
            )
        }
        
        base_thresholds[EnvironmentType.PRODUCTION] = production_adjustments
        base_thresholds[EnvironmentType.STAGING] = production_adjustments
        
        # Use development settings for testing and local
        base_thresholds[EnvironmentType.TESTING] = base_thresholds[EnvironmentType.DEVELOPMENT]
        base_thresholds[EnvironmentType.LOCAL] = base_thresholds[EnvironmentType.DEVELOPMENT]
        
        return base_thresholds[self.environment]
    
    def _create_default_profiles(self) -> Dict[str, HealthProfile]:
        """Create default health check profiles"""
        policy = self._policies[self.environment]
        
        profiles = {
            "full": HealthProfile(
                name="full",
                enabled_plugins={
                    # ML plugins
                    "ml_service", "enhanced_ml_service", "ml_model", "ml_data_quality",
                    "ml_training", "ml_performance", "ml_orchestrator", "ml_component_registry",
                    "ml_resource_manager", "ml_workflow_engine", "ml_event_bus",
                    # Database plugins  
                    "database", "database_connection_pool", "database_query_performance",
                    "database_index_health", "database_bloat",
                    # Redis plugins
                    "redis", "redis_detailed", "redis_memory",
                    # API plugins
                    "analytics_service", "enhanced_analytics_service", "mcp_server",
                    # System plugins
                    "system_resources", "queue_service"
                },
                global_timeout=policy.global_timeout_seconds,
                parallel_execution=policy.parallel_execution,
                critical_only=False
            ),
            "critical": HealthProfile(
                name="critical",
                enabled_plugins={
                    "enhanced_ml_service", "database", "redis",
                    "enhanced_analytics_service", "mcp_server"
                },
                global_timeout=policy.global_timeout_seconds * 0.7,  # Faster for critical
                parallel_execution=True,
                critical_only=True
            ),
            "minimal": HealthProfile(
                name="minimal",
                enabled_plugins={
                    "database", "redis", "system_resources"
                },
                global_timeout=15.0,
                parallel_execution=True,
                critical_only=False
            ),
            "ml_focused": HealthProfile(
                name="ml_focused",
                enabled_plugins={
                    "enhanced_ml_service", "ml_model", "ml_data_quality",
                    "ml_training", "ml_performance", "ml_orchestrator",
                    "database", "redis"  # Include core infrastructure
                },
                global_timeout=policy.global_timeout_seconds,
                parallel_execution=True,
                critical_only=False
            ),
            "infrastructure": HealthProfile(
                name="infrastructure",
                enabled_plugins={
                    "database", "database_connection_pool", "database_query_performance",
                    "redis", "redis_detailed", "redis_memory",
                    "system_resources", "queue_service"
                },
                global_timeout=policy.global_timeout_seconds,
                parallel_execution=True,
                critical_only=False
            )
        }
        
        # Adjust profiles based on environment
        if self.environment == EnvironmentType.PRODUCTION:
            # In production, use critical profile by default
            profiles["default"] = profiles["critical"]
        elif self.environment == EnvironmentType.DEVELOPMENT:
            # In development, use minimal profile by default
            profiles["default"] = profiles["minimal"]
        else:
            # In other environments, use full profile by default  
            profiles["default"] = profiles["full"]
            
        return profiles
    
    def get_policy(self) -> HealthMonitoringPolicy:
        """Get monitoring policy for current environment"""
        return self._policies[self.environment]
    
    def get_category_thresholds(self, category: HealthCheckCategory) -> CategoryThresholds:
        """Get thresholds for a specific health check category"""
        return self._category_thresholds.get(category, CategoryThresholds())
    
    def get_profile(self, profile_name: str = "default") -> Optional[HealthProfile]:
        """Get a health check profile by name"""
        return self._profiles.get(profile_name)
    
    def get_available_profiles(self) -> list[str]:
        """Get list of available profile names"""
        return list(self._profiles.keys())
    
    def create_plugin_config(
        self,
        category: HealthCheckCategory,
        critical: bool = False,
        **overrides
    ) -> HealthCheckPluginConfig:
        """Create plugin configuration based on category and environment"""
        thresholds = self.get_category_thresholds(category)
        policy = self.get_policy()
        
        config = HealthCheckPluginConfig(
            enabled=True,
            timeout_seconds=thresholds.timeout_seconds,
            critical=critical,
            retry_count=thresholds.retry_count,
            retry_delay_seconds=thresholds.retry_delay_seconds,
            tags=set(),
            metadata={
                "environment": self.environment.value,
                "category": category.value,
                "policy_version": "1.0"
            }
        )
        
        # Apply any overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def optimize_for_performance(self) -> Dict[str, Any]:
        """Get performance optimization settings"""
        policy = self.get_policy()
        
        return {
            "parallel_execution": policy.parallel_execution,
            "max_concurrent_checks": policy.max_concurrent_checks,
            "global_timeout": policy.global_timeout_seconds,
            "batch_size": min(policy.max_concurrent_checks, 8),
            "connection_pooling": True,
            "cache_results": True,
            "cache_ttl_seconds": 30.0,
            "performance_target_ms": 10.0,  # Target <10ms per health check
            "memory_limit_mb": 100.0
        }
    
    def get_alerting_config(self) -> Dict[str, Any]:
        """Get alerting configuration"""
        policy = self.get_policy()
        
        return {
            "enabled": self.environment in [EnvironmentType.STAGING, EnvironmentType.PRODUCTION],
            "critical_threshold": policy.critical_alert_threshold,
            "degraded_threshold": policy.degraded_alert_threshold,
            "notification_channels": self._get_notification_channels(),
            "escalation_timeout_minutes": 15,
            "recovery_notification": True
        }
    
    def _get_notification_channels(self) -> list[str]:
        """Get notification channels based on environment"""
        if self.environment == EnvironmentType.PRODUCTION:
            return ["email", "slack", "pagerduty"]
        elif self.environment == EnvironmentType.STAGING:
            return ["email", "slack"]
        else:
            return ["email"]


# Global configuration manager instance
_config_manager: Optional[HealthConfigurationManager] = None


def get_health_config() -> HealthConfigurationManager:
    """Get the global health configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = HealthConfigurationManager()
    return _config_manager


def reset_health_config() -> None:
    """Reset the global health configuration manager"""
    global _config_manager
    _config_manager = None


def load_config_from_environment() -> HealthConfigurationManager:
    """Load configuration from environment variables"""
    return HealthConfigurationManager()


# Convenience functions
def get_default_profile() -> Optional[HealthProfile]:
    """Get the default health profile for current environment"""
    return get_health_config().get_profile("default")


def get_critical_profile() -> Optional[HealthProfile]:
    """Get the critical health profile"""
    return get_health_config().get_profile("critical")


def create_category_config(
    category: HealthCheckCategory,
    critical: bool = False,
    **overrides
) -> HealthCheckPluginConfig:
    """Create plugin configuration for a category"""
    return get_health_config().create_plugin_config(category, critical, **overrides)