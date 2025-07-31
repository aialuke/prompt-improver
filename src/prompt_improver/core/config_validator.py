"""
Comprehensive configuration validation system for startup integrity checks.
Ensures all services can start successfully with the provided configuration.
"""

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime

# Import internal configuration modules
from prompt_improver.security.config_validator import SecurityConfigValidator
from prompt_improver.core.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a configuration validation check."""
    component: str
    is_valid: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    critical: bool = True  # Whether failure should prevent startup


@dataclass
class ConfigurationValidationReport:
    """Complete configuration validation report."""
    timestamp: datetime = field(default_factory=datetime.now)
    environment: str = "unknown"
    overall_valid: bool = False
    results: List[ValidationResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    critical_failures: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
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
                    "critical": r.critical
                }
                for r in self.results
            ],
            "warnings": self.warnings,
            "critical_failures": self.critical_failures
        }


class ConfigurationValidator:
    """
    Comprehensive startup configuration validator.
    
    Validates all configuration values at startup and performs connectivity tests
    to ensure all services can start successfully.
    """
    
    def __init__(self, environment: Optional[str] = None):
        """Initialize validator with optional environment override."""
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.security_validator = SecurityConfigValidator()
        self.db_config = AppConfig().database
        
    async def validate_all(self) -> ConfigurationValidationReport:
        """Perform comprehensive configuration validation."""
        report = ConfigurationValidationReport(environment=self.environment)
        
        # Run all validation checks
        validators = [
            self._validate_environment_setup,
            self._validate_security_configuration,
            self._validate_database_configuration,
            self._validate_database_connectivity,
            self._validate_redis_configuration,
            self._validate_redis_connectivity,
            self._validate_ml_configuration,
            self._validate_api_configuration,
            self._validate_health_check_configuration,
            self._validate_monitoring_configuration
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
                report.results.append(ValidationResult(
                    component=validator_func.__name__,
                    is_valid=False,
                    message=f"Validation error: {str(e)}",
                    critical=True
                ))
        
        # Analyze results
        critical_failures = [r for r in report.results if not r.is_valid and r.critical]
        warnings = [r for r in report.results if not r.is_valid and not r.critical]
        
        report.critical_failures = [f"{r.component}: {r.message}" for r in critical_failures]
        report.warnings = [f"{r.component}: {r.message}" for r in warnings]
        report.overall_valid = len(critical_failures) == 0
        
        return report
    
    async def _validate_environment_setup(self) -> ValidationResult:
        """Validate basic environment setup."""
        required_vars = [
            "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DATABASE",
            "POSTGRES_USERNAME", "POSTGRES_PASSWORD"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            return ValidationResult(
                component="environment_setup",
                is_valid=False,
                message=f"Missing required environment variables: {', '.join(missing_vars)}",
                details={"missing_variables": missing_vars}
            )
        
        return ValidationResult(
            component="environment_setup",
            is_valid=True,
            message="All required environment variables are set"
        )
    
    async def _validate_security_configuration(self) -> ValidationResult:
        """Validate security configuration using existing validator."""
        is_valid, issues = self.security_validator.validate_environment()
        
        return ValidationResult(
            component="security",
            is_valid=is_valid,
            message="Security validation passed" if is_valid else f"Security issues found: {len(issues)}",
            details={"issues": issues} if issues else None
        )
    
    async def _validate_database_configuration(self) -> ValidationResult:
        """Validate database configuration parameters."""
        try:
            # Test database config instantiation
            db_config = self.db_config
            
            # Validate connection parameters
            if db_config.postgres_port < 1 or db_config.postgres_port > 65535:
                return ValidationResult(
                    component="database_config",
                    is_valid=False,
                    message=f"Invalid database port: {db_config.postgres_port}"
                )
            
            # Validate pool settings
            if db_config.pool_min_size > db_config.pool_max_size:
                return ValidationResult(
                    component="database_config",
                    is_valid=False,
                    message=f"Pool min size ({db_config.pool_min_size}) cannot exceed max size ({db_config.pool_max_size})"
                )
            
            return ValidationResult(
                component="database_config",
                is_valid=True,
                message="Database configuration is valid",
                details={
                    "host": db_config.postgres_host,
                    "port": db_config.postgres_port,
                    "database": db_config.postgres_database,
                    "pool_settings": {
                        "min_size": db_config.pool_min_size,
                        "max_size": db_config.pool_max_size,
                        "timeout": db_config.pool_timeout
                    }
                }
            )
            
        except Exception as e:
            return ValidationResult(
                component="database_config",
                is_valid=False,
                message=f"Database configuration error: {str(e)}"
            )
    
    async def _validate_database_connectivity(self) -> ValidationResult:
        """Test database connectivity with configuration values."""
        try:
            import asyncpg

            # Test basic connectivity
            conn_string = self.db_config.database_url
            timeout = 5  # seconds
            
            # Convert asyncpg URL format for connection
            if "postgresql+asyncpg://" in conn_string:
                conn_string = conn_string.replace("postgresql+asyncpg://", "postgresql://")

            conn = await asyncpg.connect(conn_string, timeout=timeout)
            try:
                # Test basic query
                result = await conn.fetchval("SELECT 1")

                if result == 1:
                    return ValidationResult(
                        component="database_connectivity",
                        is_valid=True,
                        message="Database connectivity test passed",
                        details={"response_time_ms": timeout * 1000}
                    )
            finally:
                await conn.close()
            
            return ValidationResult(
                component="database_connectivity",
                is_valid=False,
                message="Database connectivity test failed - no valid response"
            )
            
        except ImportError:
            return ValidationResult(
                component="database_connectivity",
                is_valid=False,
                message="psycopg not available for connectivity test",
                critical=False  # Not critical if module not installed yet
            )
        except Exception as e:
            return ValidationResult(
                component="database_connectivity",
                is_valid=False,
                message=f"Database connectivity failed: {str(e)}"
            )
    
    async def _validate_redis_configuration(self) -> ValidationResult:
        """Validate Redis configuration parameters."""
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            return ValidationResult(
                component="redis_config",
                is_valid=False,
                message="REDIS_URL not configured",
                critical=False  # Redis might be optional
            )
        
        # Validate URL format
        if not redis_url.startswith(("redis://", "rediss://")):
            return ValidationResult(
                component="redis_config",
                is_valid=False,
                message=f"Invalid Redis URL format: {redis_url}"
            )
        
        return ValidationResult(
            component="redis_config",
            is_valid=True,
            message="Redis configuration is valid",
            details={"url_configured": bool(redis_url)}
        )
    
    async def _validate_redis_connectivity(self) -> ValidationResult:
        """Test Redis connectivity."""
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            return ValidationResult(
                component="redis_connectivity",
                is_valid=False,
                message="Redis URL not configured - skipping connectivity test",
                critical=False
            )
        
        try:
            import coredis
            
            # Test Redis connectivity
            client = coredis.Redis.from_url(redis_url)
            
            # Test basic operations
            await client.ping()
            test_key = "config_validator_test"
            await client.set(test_key, "test_value", ex=5)  # 5 second expiry
            value = await client.get(test_key)
            await client.delete(test_key)
            
            if value == b"test_value":
                return ValidationResult(
                    component="redis_connectivity",
                    is_valid=True,
                    message="Redis connectivity test passed"
                )
            else:
                return ValidationResult(
                    component="redis_connectivity",
                    is_valid=False,
                    message="Redis connectivity test failed - data integrity issue"
                )
                
        except ImportError:
            return ValidationResult(
                component="redis_connectivity",
                is_valid=False,
                message="coredis not available for connectivity test",
                critical=False
            )
        except Exception as e:
            return ValidationResult(
                component="redis_connectivity",
                is_valid=False,
                message=f"Redis connectivity failed: {str(e)}",
                critical=False  # Redis failure shouldn't prevent startup
            )
    
    async def _validate_ml_configuration(self) -> ValidationResult:
        """Validate ML model configuration and accessibility."""
        ml_model_path = os.getenv("ML_MODEL_PATH", "./models")
        
        # Check if model path exists
        model_path = Path(ml_model_path)
        if not model_path.exists():
            return ValidationResult(
                component="ml_config",
                is_valid=False,
                message=f"ML model path does not exist: {ml_model_path}",
                critical=False  # ML might be optional
            )
        
        if not model_path.is_dir():
            return ValidationResult(
                component="ml_config",
                is_valid=False,
                message=f"ML model path is not a directory: {ml_model_path}",
                critical=False
            )
        
        # Check for common model files
        model_files = list(model_path.glob("*.bin")) + list(model_path.glob("*.pkl"))
        
        return ValidationResult(
            component="ml_config",
            is_valid=True,
            message=f"ML configuration valid - found {len(model_files)} model files",
            details={
                "model_path": str(model_path),
                "model_files_count": len(model_files),
                "model_files": [f.name for f in model_files[:5]]  # First 5 files
            }
        )
    
    async def _validate_api_configuration(self) -> ValidationResult:
        """Validate API rate limits and security settings."""
        results = []
        
        # Validate rate limiting settings
        rate_limit = os.getenv("API_RATE_LIMIT_PER_MINUTE")
        if rate_limit:
            try:
                rate_limit_val = int(rate_limit)
                if rate_limit_val <= 0:
                    results.append("Rate limit must be positive")
                elif rate_limit_val > 10000:
                    results.append("Rate limit seems unusually high (>10000/min)")
            except ValueError:
                results.append(f"Invalid rate limit value: {rate_limit}")
        
        # Validate timeout settings
        timeout = os.getenv("API_TIMEOUT_SECONDS")
        if timeout:
            try:
                timeout_val = int(timeout)
                if timeout_val <= 0:
                    results.append("API timeout must be positive")
                elif timeout_val > 300:
                    results.append("API timeout seems unusually high (>300s)")
            except ValueError:
                results.append(f"Invalid timeout value: {timeout}")
        
        is_valid = len(results) == 0
        message = "API configuration is valid" if is_valid else f"API configuration issues: {'; '.join(results)}"
        
        return ValidationResult(
            component="api_config",
            is_valid=is_valid,
            message=message,
            details={"issues": results} if results else None,
            critical=False
        )
    
    async def _validate_health_check_configuration(self) -> ValidationResult:
        """Validate health check threshold settings."""
        health_timeout = os.getenv("HEALTH_CHECK_TIMEOUT_SECONDS", "10")
        db_timeout = os.getenv("HEALTH_CHECK_DB_QUERY_TIMEOUT", "5")
        redis_timeout = os.getenv("HEALTH_CHECK_REDIS_TIMEOUT", "2")
        
        try:
            health_timeout_val = int(health_timeout)
            db_timeout_val = int(db_timeout)
            redis_timeout_val = int(redis_timeout)
            
            issues = []
            
            if health_timeout_val <= 0:
                issues.append("Health check timeout must be positive")
            
            if db_timeout_val >= health_timeout_val:
                issues.append("DB timeout should be less than overall health timeout")
            
            if redis_timeout_val >= health_timeout_val:
                issues.append("Redis timeout should be less than overall health timeout")
            
            is_valid = len(issues) == 0
            message = "Health check configuration is valid" if is_valid else f"Issues: {'; '.join(issues)}"
            
            return ValidationResult(
                component="health_check_config",
                is_valid=is_valid,
                message=message,
                details={
                    "health_timeout": health_timeout_val,
                    "db_timeout": db_timeout_val,
                    "redis_timeout": redis_timeout_val,
                    "issues": issues
                } if issues else {
                    "health_timeout": health_timeout_val,
                    "db_timeout": db_timeout_val,
                    "redis_timeout": redis_timeout_val
                },
                critical=False
            )
            
        except ValueError as e:
            return ValidationResult(
                component="health_check_config",
                is_valid=False,
                message=f"Invalid health check timeout values: {str(e)}",
                critical=False
            )
    
    async def _validate_monitoring_configuration(self) -> ValidationResult:
        """Validate monitoring and observability settings."""
        monitoring_enabled = os.getenv("MONITORING_ENABLED", "false").lower() == "true"
        
        if not monitoring_enabled:
            return ValidationResult(
                component="monitoring_config",
                is_valid=True,
                message="Monitoring disabled - no validation needed",
                critical=False
            )
        
        # Check for monitoring-related environment variables
        monitoring_vars = [
            "METRICS_EXPORT_INTERVAL_SECONDS",
            "TRACING_ENABLED",
            "ALERT_WEBHOOK_URL"
        ]
        
        missing_vars = [var for var in monitoring_vars if not os.getenv(var)]
        
        if missing_vars:
            return ValidationResult(
                component="monitoring_config",
                is_valid=False,
                message=f"Monitoring enabled but missing configuration: {', '.join(missing_vars)}",
                details={"missing_vars": missing_vars},
                critical=False
            )
        
        return ValidationResult(
            component="monitoring_config",
            is_valid=True,
            message="Monitoring configuration is valid"
        )


async def validate_startup_configuration(environment: Optional[str] = None) -> Tuple[bool, ConfigurationValidationReport]:
    """
    Main entry point for startup configuration validation.
    
    Args:
        environment: Optional environment override
        
    Returns:
        Tuple of (is_valid, validation_report)
    """
    validator = ConfigurationValidator(environment)
    report = await validator.validate_all()
    
    # Log results
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
    
    return report.overall_valid, report


def save_validation_report(report: ConfigurationValidationReport, output_file: Optional[str] = None) -> None:
    """Save validation report to JSON file."""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"config_validation_report_{timestamp}.json"
    
    output_path = Path(output_file)
    output_path.write_text(json.dumps(report.to_dict(), indent=2))
    logger.info(f"Configuration validation report saved to: {output_path}")


if __name__ == "__main__":
    """CLI entry point for configuration validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate application configuration")
    parser.add_argument("--environment", help="Environment to validate")
    parser.add_argument("--output", help="Output file for validation report")
    parser.add_argument("--exit-on-failure", action="store_true", 
                       help="Exit with non-zero code on validation failure")
    
    args = parser.parse_args()
    
    async def main():
        is_valid, report = await validate_startup_configuration(args.environment)
        
        if args.output:
            save_validation_report(report, args.output)
        
        if args.exit_on_failure and not is_valid:
            sys.exit(1)
    
    asyncio.run(main())