"""Configuration Validation Module

Comprehensive validation for all configuration components with
real behavior testing and environment-specific validation rules.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional, Tuple

from prompt_improver.core.config.app_config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a configuration validation check."""

    component: str
    is_valid: bool
    message: str
    details: dict[str, Any] | None = None
    critical: bool = True
    
    
@dataclass  
class ValidationReport:
    """Complete configuration validation report."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    environment: str = "unknown"
    overall_valid: bool = False
    results: list[ValidationResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    critical_failures: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "environment": self.environment,
            "overall_valid": self.overall_valid,
            "results": [
                {
                    "component": r.component,
                    "is_valid": r.is_valid,
                    "message": r.message,
                    "details": r.details,
                    "critical": r.critical,
                }
                for r in self.results
            ],
            "warnings": self.warnings,
            "critical_failures": self.critical_failures,
        }


class ConfigurationValidator:
    """Comprehensive configuration validator with real behavior testing."""
    
    def __init__(self, config: AppConfig):
        """Initialize validator with configuration."""
        self.config = config
        
    async def validate_all(self) -> ValidationReport:
        """Perform comprehensive configuration validation."""
        report = ValidationReport(environment=self.config.environment)
        
        validators = [
            self._validate_app_config,
            self._validate_database_config,
            self._validate_database_connectivity,
            self._validate_security_config,
            self._validate_monitoring_config,
            self._validate_ml_config,
            self._validate_environment_consistency,
        ]
        
        for validator_func in validators:
            try:
                result = await validator_func()
                if isinstance(result, list):
                    report.results.extend(result)
                else:
                    report.results.append(result)
            except Exception as e:
                logger.exception(f"Validation error in {validator_func.__name__}")
                report.results.append(
                    ValidationResult(
                        component=validator_func.__name__,
                        is_valid=False,
                        message=f"Validation error: {str(e)}",
                        critical=True,
                    )
                )
        
        # Categorize results
        critical_failures = [r for r in report.results if not r.is_valid and r.critical]
        warnings = [r for r in report.results if not r.is_valid and not r.critical]
        
        report.critical_failures = [f"{r.component}: {r.message}" for r in critical_failures]
        report.warnings = [f"{r.component}: {r.message}" for r in warnings]
        report.overall_valid = len(critical_failures) == 0
        
        return report
    
    async def _validate_app_config(self) -> ValidationResult:
        """Validate main application configuration."""
        try:
            issues = []
            
            # Validate environment
            allowed_envs = ["development", "testing", "staging", "production"]
            if self.config.environment not in allowed_envs:
                issues.append(f"Invalid environment: {self.config.environment}")
            
            # Validate log level  
            allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if self.config.log_level not in allowed_levels:
                issues.append(f"Invalid log level: {self.config.log_level}")
                
            # Validate MCP settings
            if self.config.mcp_port < 1 or self.config.mcp_port > 65535:
                issues.append(f"Invalid MCP port: {self.config.mcp_port}")
                
            is_valid = len(issues) == 0
            message = "App configuration valid" if is_valid else f"Issues: {'; '.join(issues)}"
            
            return ValidationResult(
                component="app_config",
                is_valid=is_valid,
                message=message,
                details={"issues": issues} if issues else None,
            )
            
        except Exception as e:
            return ValidationResult(
                component="app_config",
                is_valid=False,
                message=f"App config validation error: {str(e)}",
            )
    
    async def _validate_database_config(self) -> ValidationResult:
        """Validate database configuration parameters."""
        try:
            db_config = self.config.database
            issues = []
            
            # Validate connection parameters
            if not db_config.postgres_host:
                issues.append("Database host is required")
                
            if db_config.postgres_port < 1 or db_config.postgres_port > 65535:
                issues.append(f"Invalid database port: {db_config.postgres_port}")
                
            if not db_config.postgres_database:
                issues.append("Database name is required")
                
            if not db_config.postgres_username:
                issues.append("Database username is required")
                
            # Validate pool settings
            if db_config.pool_min_size > db_config.pool_max_size:
                issues.append(f"Pool min size ({db_config.pool_min_size}) exceeds max size ({db_config.pool_max_size})")
                
            if db_config.pool_timeout <= 0:
                issues.append("Pool timeout must be positive")
                
            is_valid = len(issues) == 0
            message = "Database configuration valid" if is_valid else f"Issues: {'; '.join(issues)}"
            
            return ValidationResult(
                component="database_config",
                is_valid=is_valid,
                message=message,
                details={
                    "host": db_config.postgres_host,
                    "port": db_config.postgres_port,
                    "database": db_config.postgres_database,
                    "pool_settings": {
                        "min_size": db_config.pool_min_size,
                        "max_size": db_config.pool_max_size,
                        "timeout": db_config.pool_timeout,
                    },
                    "issues": issues,
                } if issues else {
                    "host": db_config.postgres_host,
                    "port": db_config.postgres_port,
                    "database": db_config.postgres_database,
                    "pool_settings": {
                        "min_size": db_config.pool_min_size,
                        "max_size": db_config.pool_max_size,
                        "timeout": db_config.pool_timeout,
                    },
                },
            )
            
        except Exception as e:
            return ValidationResult(
                component="database_config",
                is_valid=False,
                message=f"Database config validation error: {str(e)}",
            )
    
    async def _validate_database_connectivity(self) -> ValidationResult:
        """Test database connectivity using existing health service."""
        try:
            from prompt_improver.core.services.service_registry import get_database_health_service
            
            # Use existing database health service for connectivity testing
            try:
                health_service = get_database_health_service()
                health_status = await health_service.health_check()
            except ValueError:
                # Health service not registered, try fallback approach
                return ValidationResult(
                    component="database_connectivity",
                    is_valid=False,
                    message="Database health service not available for connectivity test",
                    critical=False,
                )
            
            database_healthy = health_status.get("overall_status") in ["healthy", "warning"]
            response_time_ms = health_status.get("check_time_ms", 0)
            overall_status = health_status.get("overall_status", "unknown")
            
            if database_healthy and overall_status in ["healthy", "warning"]:
                return ValidationResult(
                    component="database_connectivity",
                    is_valid=True,
                    message="Database connectivity test passed",
                    details={
                        "response_time_ms": round(response_time_ms, 2),
                        "overall_status": overall_status,
                        "health_components": health_status.get("components", {}),
                    },
                )
            
            # Build error message
            error_details = []
            for component, status in health_status.get("components", {}).items():
                if status.get("status") in ["error", "critical"]:
                    error_details.append(f"{component}: {status.get('status', 'unknown')}")
                    
            error_message = f"Database connectivity failed - Status: {overall_status}"
            if error_details:
                error_message += f" - Issues: {'; '.join(error_details)}"
                
            return ValidationResult(
                component="database_connectivity",
                is_valid=False,
                message=error_message,
                details={
                    "response_time_ms": round(response_time_ms, 2),
                    "overall_status": overall_status,
                    "health_components": health_status.get("components", {}),
                    "error": health_status.get("error"),
                },
            )
            
        except ImportError:
            return ValidationResult(
                component="database_connectivity",
                is_valid=False,
                message="Service registry not available for connectivity test",
                critical=False,
            )
        except Exception as e:
            return ValidationResult(
                component="database_connectivity", 
                is_valid=False,
                message=f"Database connectivity test failed: {str(e)}",
                details={"error_type": type(e).__name__},
            )
    
    async def _validate_security_config(self) -> ValidationResult:
        """Validate security configuration."""
        try:
            security_config = self.config.security
            issues = []
            
            # Validate secret key
            if not security_config.secret_key:
                issues.append("Secret key is required")
            elif len(security_config.secret_key) < 32:
                issues.append(f"Secret key must be at least 32 characters, got {len(security_config.secret_key)}")
                
            # Validate token settings
            if security_config.token_expiry_seconds <= 0:
                issues.append("Token expiry must be positive")
                
            if security_config.hash_rounds < 4 or security_config.hash_rounds > 20:
                issues.append("Hash rounds must be between 4 and 20")
                
            # Validate authentication settings
            auth_config = security_config.authentication
            if auth_config.api_key_length_bytes < 16:
                issues.append("API key length must be at least 16 bytes")
                
            if auth_config.max_failed_attempts_per_hour < 1:
                issues.append("Max failed attempts must be at least 1")
                
            # Validate rate limiting
            rate_config = security_config.rate_limiting
            if rate_config.basic_tier_rate_limit < 1:
                issues.append("Basic tier rate limit must be at least 1")
                
            if rate_config.basic_tier_burst_capacity < rate_config.basic_tier_rate_limit:
                issues.append("Burst capacity must be >= rate limit")
                
            is_valid = len(issues) == 0
            message = "Security configuration valid" if is_valid else f"Issues: {'; '.join(issues)}"
            
            return ValidationResult(
                component="security_config",
                is_valid=is_valid,
                message=message,
                details={"issues": issues} if issues else None,
            )
            
        except Exception as e:
            return ValidationResult(
                component="security_config",
                is_valid=False,
                message=f"Security config validation error: {str(e)}",
            )
    
    async def _validate_monitoring_config(self) -> ValidationResult:
        """Validate monitoring configuration."""
        try:
            monitoring_config = self.config.monitoring
            issues = []
            
            # Validate health check settings
            health_config = monitoring_config.health_checks
            if health_config.interval_seconds <= 0:
                issues.append("Health check interval must be positive")
                
            if health_config.timeout_seconds >= health_config.interval_seconds:
                issues.append("Health check timeout should be less than interval")
                
            # Validate metrics settings
            metrics_config = monitoring_config.metrics  
            if metrics_config.collection_interval_seconds <= 0:
                issues.append("Metrics collection interval must be positive")
                
            # Validate logging settings
            logging_config = monitoring_config.logging
            allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if logging_config.level not in allowed_levels:
                issues.append(f"Invalid log level: {logging_config.level}")
                
            allowed_formats = ["json", "text", "structured"]
            if logging_config.format not in allowed_formats:
                issues.append(f"Invalid log format: {logging_config.format}")
                
            is_valid = len(issues) == 0
            message = "Monitoring configuration valid" if is_valid else f"Issues: {'; '.join(issues)}"
            
            return ValidationResult(
                component="monitoring_config",
                is_valid=is_valid,
                message=message,
                details={"issues": issues} if issues else None,
            )
            
        except Exception as e:
            return ValidationResult(
                component="monitoring_config",
                is_valid=False,
                message=f"Monitoring config validation error: {str(e)}",
            )
    
    async def _validate_ml_config(self) -> ValidationResult:
        """Validate ML configuration.""" 
        try:
            ml_config = self.config.ml
            issues = []
            
            # Validate model serving settings
            serving_config = ml_config.model_serving
            if serving_config.model_timeout <= 0:
                issues.append("Model timeout must be positive")
                
            if serving_config.max_batch_size < 1:
                issues.append("Max batch size must be at least 1")
                
            # Validate training settings
            training_config = ml_config.training
            if training_config.max_training_time_hours < 1:
                issues.append("Max training time must be at least 1 hour")
                
            # Validate feature engineering
            feature_config = ml_config.feature_engineering
            if feature_config.max_feature_dimension < 1:
                issues.append("Max feature dimension must be positive")
                
            if not 0 <= feature_config.feature_selection_threshold <= 1:
                issues.append("Feature selection threshold must be between 0 and 1")
                
            # Validate resource limits
            if ml_config.memory_limit_gb < 1:
                issues.append("Memory limit must be at least 1 GB")
                
            if ml_config.cpu_limit < 1:
                issues.append("CPU limit must be at least 1")
                
            is_valid = len(issues) == 0
            message = "ML configuration valid" if is_valid else f"Issues: {'; '.join(issues)}"
            
            return ValidationResult(
                component="ml_config",
                is_valid=is_valid,
                message=message,
                details={"issues": issues} if issues else None,
                critical=False,  # ML config issues are not critical for core app
            )
            
        except Exception as e:
            return ValidationResult(
                component="ml_config",
                is_valid=False,
                message=f"ML config validation error: {str(e)}",
                critical=False,
            )
    
    async def _validate_environment_consistency(self) -> ValidationResult:
        """Validate environment-specific consistency."""
        try:
            issues = []
            
            # Production-specific validations
            if self.config.environment == "production":
                if self.config.debug:
                    issues.append("Debug mode should be disabled in production")
                    
                if self.config.security.security_profile.value != "production":
                    issues.append("Security profile should be 'production' in production environment")
                    
                if "dev" in self.config.security.secret_key.lower():
                    issues.append("Development secret key detected in production")
                    
            # Development-specific validations
            elif self.config.environment == "development":
                if self.config.security.security_profile.value == "high_security":
                    issues.append("High security profile may be excessive for development")
                    
            is_valid = len(issues) == 0
            message = "Environment consistency valid" if is_valid else f"Issues: {'; '.join(issues)}"
            
            return ValidationResult(
                component="environment_consistency",
                is_valid=is_valid,
                message=message,
                details={"issues": issues} if issues else None,
                critical=self.config.environment == "production",  # Critical only in production
            )
            
        except Exception as e:
            return ValidationResult(
                component="environment_consistency",
                is_valid=False,
                message=f"Environment consistency validation error: {str(e)}",
            )


async def validate_configuration(config: AppConfig | None = None) -> tuple[bool, ValidationReport]:
    """Main entry point for configuration validation.
    
    Args:
        config: Configuration to validate (defaults to global config)
        
    Returns:
        Tuple of (is_valid, validation_report)
    """
    if config is None:
        from prompt_improver.core.config.app_config import get_config
        config = get_config()
        
    validator = ConfigurationValidator(config)
    report = await validator.validate_all()
    
    if report.overall_valid:
        logger.info("✅ Configuration validation passed")
        logger.info(f"Environment: {report.environment}")
        logger.info(f"Validated {len(report.results)} components")
        
        if report.warnings:
            logger.warning(f"Found {len(report.warnings)} warnings:")
            for warning in report.warnings:
                logger.warning(f"  ⚠️  {warning}")
    else:
        logger.error("❌ Configuration validation failed")
        logger.error(f"Environment: {report.environment}")
        logger.error(f"Critical failures: {len(report.critical_failures)}")
        
        for failure in report.critical_failures:
            logger.error(f"  🔥 {failure}")
            
        for warning in report.warnings:
            logger.warning(f"  ⚠️  {warning}")
    
    return (report.overall_valid, report)