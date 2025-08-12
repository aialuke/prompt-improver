"""Production Optimization Guide for Performance Baseline System.

Provides configuration recommendations, deployment strategies, and optimization
techniques for running the baseline system efficiently in production environments.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Target deployment environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    HIGH_TRAFFIC = "high_traffic"


class OptimizationLevel(Enum):
    """Optimization levels for different performance requirements."""

    MINIMAL = "minimal"
    BALANCED = "balanced"
    COMPREHENSIVE = "comprehensive"
    ENTERPRISE = "enterprise"


@dataclass
class EnvironmentConfig:
    """Configuration optimized for specific environments."""

    environment: DeploymentEnvironment
    optimization_level: OptimizationLevel
    collection_interval_seconds: int
    retention_days: int
    max_memory_usage_mb: int
    cpu_usage_limit_percent: int
    enable_real_time_dashboard: bool
    enable_load_testing: bool
    alert_sensitivity: str
    data_compression: bool
    batch_processing: bool


class ProductionOptimizationGuide:
    """Comprehensive guide for optimizing the performance baseline system
    for production deployment with minimal performance impact.
    """

    def __init__(self):
        self.environment_configs = self._create_environment_configs()
        self.optimization_strategies = self._define_optimization_strategies()
        self.monitoring_recommendations = self._create_monitoring_recommendations()

    def _create_environment_configs(
        self,
    ) -> dict[DeploymentEnvironment, EnvironmentConfig]:
        """Create optimized configurations for different environments."""
        return {
            DeploymentEnvironment.DEVELOPMENT: EnvironmentConfig(
                environment=DeploymentEnvironment.DEVELOPMENT,
                optimization_level=OptimizationLevel.COMPREHENSIVE,
                collection_interval_seconds=30,
                retention_days=7,
                max_memory_usage_mb=500,
                cpu_usage_limit_percent=20,
                enable_real_time_dashboard=True,
                enable_load_testing=True,
                alert_sensitivity="low",
                data_compression=False,
                batch_processing=False,
            ),
            DeploymentEnvironment.STAGING: EnvironmentConfig(
                environment=DeploymentEnvironment.STAGING,
                optimization_level=OptimizationLevel.BALANCED,
                collection_interval_seconds=60,
                retention_days=14,
                max_memory_usage_mb=300,
                cpu_usage_limit_percent=10,
                enable_real_time_dashboard=True,
                enable_load_testing=True,
                alert_sensitivity="medium",
                data_compression=True,
                batch_processing=True,
            ),
            DeploymentEnvironment.PRODUCTION: EnvironmentConfig(
                environment=DeploymentEnvironment.PRODUCTION,
                optimization_level=OptimizationLevel.BALANCED,
                collection_interval_seconds=120,
                retention_days=30,
                max_memory_usage_mb=200,
                cpu_usage_limit_percent=5,
                enable_real_time_dashboard=False,
                enable_load_testing=False,
                alert_sensitivity="high",
                data_compression=True,
                batch_processing=True,
            ),
            DeploymentEnvironment.HIGH_TRAFFIC: EnvironmentConfig(
                environment=DeploymentEnvironment.HIGH_TRAFFIC,
                optimization_level=OptimizationLevel.MINIMAL,
                collection_interval_seconds=300,
                retention_days=7,
                max_memory_usage_mb=100,
                cpu_usage_limit_percent=2,
                enable_real_time_dashboard=False,
                enable_load_testing=False,
                alert_sensitivity="high",
                data_compression=True,
                batch_processing=True,
            ),
        }

    def _define_optimization_strategies(self) -> dict[str, dict[str, Any]]:
        """Define optimization strategies for different scenarios."""
        return {
            "memory_optimization": {
                "description": "Minimize memory usage and prevent memory leaks",
                "techniques": [
                    "Implement data retention policies with automatic cleanup",
                    "Use memory-efficient data structures (e.g., deque for time series)",
                    "Enable data compression for historical metrics",
                    "Implement lazy loading for large datasets",
                    "Use memory pooling for frequent object allocations",
                ],
                "configuration": {
                    "max_data_points": 1000,
                    "compression_algorithm": "gzip",
                    "cleanup_interval_hours": 6,
                    "memory_monitoring": True,
                },
            },
            "cpu_optimization": {
                "description": "Minimize CPU usage and optimize processing efficiency",
                "techniques": [
                    "Use async/await for non-blocking operations",
                    "Implement batch processing for bulk operations",
                    "Optimize statistical calculations with numpy/scipy",
                    "Use caching for expensive computations",
                    "Implement sampling for high-frequency data",
                ],
                "configuration": {
                    "batch_size": 100,
                    "sampling_rate": 0.1,
                    "cache_ttl_seconds": 300,
                    "parallel_processing": True,
                },
            },
            "network_optimization": {
                "description": "Optimize network usage and reduce bandwidth",
                "techniques": [
                    "Use connection pooling for database connections",
                    "Implement request batching for external APIs",
                    "Enable data compression for network transfers",
                    "Use local caching to reduce remote calls",
                    "Implement circuit breakers for external dependencies",
                ],
                "configuration": {
                    "connection_pool_size": 10,
                    "request_timeout_seconds": 30,
                    "retry_attempts": 3,
                    "circuit_breaker_threshold": 5,
                },
            },
            "storage_optimization": {
                "description": "Optimize data storage and I/O operations",
                "techniques": [
                    "Use time-series databases for metrics storage",
                    "Implement data partitioning by time periods",
                    "Use write-ahead logging for durability",
                    "Implement data aggregation to reduce storage",
                    "Use SSD storage for better I/O performance",
                ],
                "configuration": {
                    "partition_interval": "daily",
                    "aggregation_levels": ["1m", "5m", "1h", "1d"],
                    "wal_enabled": True,
                    "compression_ratio": 0.3,
                },
            },
        }

    def _create_monitoring_recommendations(self) -> dict[str, Any]:
        """Create monitoring recommendations for production."""
        return {
            "key_metrics": [
                "baseline_collection_duration_ms",
                "memory_usage_mb",
                "cpu_utilization_percent",
                "disk_io_operations_per_second",
                "network_bytes_per_second",
                "error_rate_percent",
                "alert_count_per_hour",
            ],
            "alert_thresholds": {
                "collection_duration_ms": 200,
                "memory_usage_mb": 500,
                "cpu_utilization_percent": 10,
                "error_rate_percent": 1,
                "disk_usage_percent": 80,
            },
            "health_checks": [
                "baseline_collector_health",
                "statistical_analyzer_health",
                "regression_detector_health",
                "database_connection_health",
                "cache_connection_health",
            ],
            "logging_configuration": {
                "log_level": "INFO",
                "log_format": "json",
                "log_rotation": "daily",
                "max_log_files": 30,
                "sensitive_data_masking": True,
            },
        }

    def get_optimized_config(
        self, environment: DeploymentEnvironment
    ) -> dict[str, Any]:
        """Get optimized configuration for specific environment."""
        env_config = self.environment_configs.get(environment)
        if not env_config:
            raise ValueError(
                f"No configuration available for environment: {environment}"
            )
        config = {
            "environment": {
                "name": env_config.environment.value,
                "optimization_level": env_config.optimization_level.value,
            },
            "baseline_collection": {
                "interval_seconds": env_config.collection_interval_seconds,
                "batch_processing": env_config.batch_processing,
                "compression_enabled": env_config.data_compression,
                "max_memory_mb": env_config.max_memory_usage_mb,
                "cpu_limit_percent": env_config.cpu_usage_limit_percent,
            },
            "data_retention": {
                "retention_days": env_config.retention_days,
                "compression_after_hours": 24,
                "aggregation_after_days": 7,
                "cleanup_interval_hours": 6,
            },
            "performance_targets": {
                "collection_max_duration_ms": 100,
                "analysis_max_duration_ms": 150,
                "dashboard_max_response_ms": 200,
                "memory_growth_limit_mb": 50,
                "cpu_usage_limit_percent": env_config.cpu_usage_limit_percent,
            },
            "features": {
                "real_time_dashboard": env_config.enable_real_time_dashboard,
                "load_testing": env_config.enable_load_testing,
                "advanced_analytics": env_config.optimization_level
                in [OptimizationLevel.COMPREHENSIVE, OptimizationLevel.ENTERPRISE],
            },
            "alerts": {
                "sensitivity": env_config.alert_sensitivity,
                "channels": ["email", "webhook"],
                "rate_limiting": True,
                "escalation_enabled": environment == DeploymentEnvironment.PRODUCTION,
            },
        }
        if environment == DeploymentEnvironment.PRODUCTION:
            config.update(self._get_production_specific_config())
        elif environment == DeploymentEnvironment.HIGH_TRAFFIC:
            config.update(self._get_high_traffic_config())
        return config

    def _get_production_specific_config(self) -> dict[str, Any]:
        """Get production-specific configuration optimizations."""
        return {
            "reliability": {
                "circuit_breaker_enabled": True,
                "retry_policy": {
                    "max_attempts": 3,
                    "backoff_multiplier": 2,
                    "max_delay_seconds": 60,
                },
                "health_check_interval_seconds": 30,
                "graceful_shutdown_timeout_seconds": 30,
            },
            "security": {
                "data_encryption_at_rest": True,
                "data_encryption_in_transit": True,
                "audit_logging": True,
                "access_control": True,
            },
            "scalability": {
                "horizontal_scaling": True,
                "load_balancing": True,
                "auto_scaling_enabled": True,
                "max_instances": 5,
            },
        }

    def _get_high_traffic_config(self) -> dict[str, Any]:
        """Get high-traffic specific configuration optimizations."""
        return {
            "ultra_low_overhead": {
                "sampling_rate": 0.01,
                "metrics_aggregation": True,
                "lazy_initialization": True,
                "memory_mapped_storage": True,
            },
            "performance_first": {
                "disable_real_time_features": True,
                "batch_only_processing": True,
                "async_only_operations": True,
                "cache_everything": True,
            },
        }

    def generate_deployment_checklist(
        self, environment: DeploymentEnvironment
    ) -> list[str]:
        """Generate deployment checklist for specific environment."""
        base_checklist = [
            "✓ Performance validation tests passed",
            "✓ Memory usage under target limits",
            "✓ CPU utilization optimized",
            "✓ Database connections configured",
            "✓ Monitoring and alerting configured",
            "✓ Log rotation and retention policies set",
            "✓ Health check endpoints available",
            "✓ Graceful shutdown handlers implemented",
            "✓ Configuration management in place",
            "✓ Security policies applied",
        ]
        env_config = self.environment_configs.get(environment)
        if not env_config:
            return base_checklist
        env_specific = []
        if environment == DeploymentEnvironment.PRODUCTION:
            env_specific.extend([
                "✓ Load balancer configuration verified",
                "✓ SSL/TLS certificates installed",
                "✓ Backup and recovery procedures tested",
                "✓ Incident response plan documented",
                "✓ Performance benchmarks established",
                "✓ Capacity planning completed",
                "✓ Security audit passed",
                "✓ Disaster recovery plan tested",
            ])
        if environment == DeploymentEnvironment.HIGH_TRAFFIC:
            env_specific.extend([
                "✓ Auto-scaling policies configured",
                "✓ Circuit breakers tested",
                "✓ Rate limiting configured",
                "✓ CDN integration verified",
                "✓ Database sharding configured",
                "✓ Cache warming strategies implemented",
            ])
        if env_config.enable_real_time_dashboard:
            env_specific.extend([
                "✓ WebSocket connections tested",
                "✓ Dashboard performance verified",
                "✓ Real-time update latency measured",
            ])
        return base_checklist + env_specific

    def get_optimization_recommendations(
        self, current_metrics: dict[str, float]
    ) -> list[str]:
        """Get optimization recommendations based on current performance metrics."""
        recommendations = []
        if current_metrics.get("memory_usage_mb", 0) > 300:
            recommendations.extend([
                "Enable data compression to reduce memory usage",
                "Implement more aggressive data retention policies",
                "Consider using memory-mapped files for large datasets",
                "Review data structures for memory efficiency",
            ])
        if current_metrics.get("cpu_utilization_percent", 0) > 10:
            recommendations.extend([
                "Increase collection interval to reduce CPU load",
                "Enable batch processing for bulk operations",
                "Implement sampling for high-frequency metrics",
                "Consider moving analytics to background workers",
            ])
        if current_metrics.get("collection_duration_ms", 0) > 200:
            recommendations.extend([
                "Optimize database queries with proper indexing",
                "Enable connection pooling",
                "Consider caching frequently accessed data",
                "Review and optimize statistical calculations",
            ])
        if current_metrics.get("disk_usage_percent", 0) > 80:
            recommendations.extend([
                "Enable data compression and aggregation",
                "Implement automated data cleanup",
                "Consider time-series database for better compression",
                "Archive old data to cold storage",
            ])
        return recommendations

    def validate_production_readiness(
        self, environment: DeploymentEnvironment, metrics: dict[str, float]
    ) -> dict[str, Any]:
        """Validate if system is ready for production deployment."""
        env_config = self.environment_configs.get(environment)
        if not env_config:
            return {"ready": False, "error": "Unknown environment"}
        issues = []
        warnings = []
        if metrics.get("collection_duration_ms", 0) > 200:
            issues.append("Collection duration exceeds 200ms target")
        if metrics.get("memory_usage_mb", 0) > env_config.max_memory_usage_mb:
            issues.append(
                f"Memory usage exceeds {env_config.max_memory_usage_mb}MB limit"
            )
        if (
            metrics.get("cpu_utilization_percent", 0)
            > env_config.cpu_usage_limit_percent
        ):
            issues.append(
                f"CPU usage exceeds {env_config.cpu_usage_limit_percent}% limit"
            )
        if metrics.get("error_rate_percent", 0) > 0.1:
            warnings.append("Error rate above 0.1%")
        if metrics.get("disk_usage_percent", 0) > 70:
            warnings.append("Disk usage above 70%")
        readiness_score = max(0, 100 - len(issues) * 25 - len(warnings) * 5)
        return {
            "ready": len(issues) == 0,
            "readiness_score": readiness_score,
            "issues": issues,
            "warnings": warnings,
            "environment": environment.value,
            "recommendations": self.get_optimization_recommendations(metrics),
        }


_optimization_guide: ProductionOptimizationGuide | None = None


def get_optimization_guide() -> ProductionOptimizationGuide:
    """Get global optimization guide instance."""
    global _optimization_guide
    if _optimization_guide is None:
        _optimization_guide = ProductionOptimizationGuide()
    return _optimization_guide


def get_production_config() -> dict[str, Any]:
    """Get optimized configuration for production environment."""
    guide = get_optimization_guide()
    return guide.get_optimized_config(DeploymentEnvironment.PRODUCTION)


def get_development_config() -> dict[str, Any]:
    """Get optimized configuration for development environment."""
    guide = get_optimization_guide()
    return guide.get_optimized_config(DeploymentEnvironment.DEVELOPMENT)


def validate_for_production(metrics: dict[str, float]) -> dict[str, Any]:
    """Validate if system is ready for production deployment."""
    guide = get_optimization_guide()
    return guide.validate_production_readiness(
        DeploymentEnvironment.PRODUCTION, metrics
    )


def generate_production_checklist() -> list[str]:
    """Generate production deployment checklist."""
    guide = get_optimization_guide()
    return guide.generate_deployment_checklist(DeploymentEnvironment.PRODUCTION)
