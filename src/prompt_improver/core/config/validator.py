"""Comprehensive configuration validation system for startup integrity checks.
Ensures all services can start successfully with the provided configuration.

Uses ValidationServiceProtocol to break circular dependencies with core.config.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_improver.core.config.validation import ValidationReport, ValidationResult

if TYPE_CHECKING:
    from prompt_improver.shared.interfaces.protocols.application import (
        ValidationServiceProtocol,
    )

logger = logging.getLogger(__name__)


# Use ValidationReport from validation module instead of duplicate


class ConfigurationValidator:
    """Comprehensive startup configuration validator.

    Uses ValidationServiceProtocol to break circular dependencies.
    Validates all configuration values at startup and performs connectivity tests
    to ensure all services can start successfully.
    """

    def __init__(
        self,
        validation_service: "ValidationServiceProtocol",
        environment: str | None = None
    ) -> None:
        """Initialize validator with validation service and optional environment override."""
        self.validation_service = validation_service
        self.environment = environment or os.getenv("ENVIRONMENT", "development")

    async def validate_all(self) -> ValidationReport:
        """Perform comprehensive configuration validation."""
        report = ValidationReport(environment=self.environment)

        # Use ValidationServiceProtocol methods for core validation
        validation_methods = [
            ("startup_configuration", self.validation_service.validate_startup_configuration),
            ("database_configuration", self.validation_service.validate_database_configuration),
            ("security_configuration", self.validation_service.validate_security_configuration),
            ("monitoring_configuration", self.validation_service.validate_monitoring_configuration),
        ]

        # Execute validation methods from the protocol
        for component_name, validation_method in validation_methods:
            try:
                if component_name == "startup_configuration":
                    result = await validation_method(self.environment)
                elif component_name == "database_configuration":
                    result = await validation_method(test_connectivity=True)
                elif component_name == "security_configuration":
                    result = await validation_method()
                elif component_name == "monitoring_configuration":
                    result = await validation_method(include_connectivity_tests=False)

                report.results.append(result)
            except Exception as e:
                logger.exception(f"Validation error in {component_name}")
                report.results.append(
                    ValidationResult(
                        component=component_name,
                        is_valid=False,
                        message=f"Validation error: {e!s}",
                        critical=True,
                    )
                )

        # Add additional comprehensive validation checks
        additional_validators = [
            self._validate_environment_setup,
            self._validate_redis_configuration,
            self._validate_redis_connectivity,
            self._validate_ml_configuration,
            self._validate_api_configuration,
            self._validate_health_check_configuration,
        ]

        for validator_func in additional_validators:
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
                        message=f"Validation error: {e!s}",
                        critical=True,
                    )
                )

        critical_failures = [r for r in report.results if not r.is_valid and r.critical]
        warnings = [r for r in report.results if not r.is_valid and (not r.critical)]
        report.critical_failures = [
            f"{r.component}: {r.message}" for r in critical_failures
        ]
        report.warnings = [f"{r.component}: {r.message}" for r in warnings]
        report.overall_valid = len(critical_failures) == 0
        return report

    async def _validate_environment_setup(self) -> ValidationResult:
        """Validate basic environment setup."""
        required_vars = [
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_DATABASE",
            "POSTGRES_USERNAME",
            "POSTGRES_PASSWORD",
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            return ValidationResult(
                component="environment_setup",
                is_valid=False,
                message=f"Missing required environment variables: {', '.join(missing_vars)}",
                details={"missing_variables": missing_vars},
            )
        return ValidationResult(
            component="environment_setup",
            is_valid=True,
            message="All required environment variables are set",
        )

    # Database and security configuration validation now handled by ValidationServiceProtocol

    async def _validate_redis_configuration(self) -> ValidationResult:
        """Validate Redis configuration parameters."""
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            return ValidationResult(
                component="redis_config",
                is_valid=False,
                message="REDIS_URL not configured",
                critical=False,
            )
        if not redis_url.startswith(("redis://", "rediss://")):
            return ValidationResult(
                component="redis_config",
                is_valid=False,
                message=f"Invalid Redis URL format: {redis_url}",
            )
        return ValidationResult(
            component="redis_config",
            is_valid=True,
            message="Redis configuration is valid",
            details={"url_configured": bool(redis_url)},
        )

    async def _validate_redis_connectivity(self) -> ValidationResult:
        """Test Redis connectivity."""
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            return ValidationResult(
                component="redis_connectivity",
                is_valid=False,
                message="Redis URL not configured - skipping connectivity test",
                critical=False,
            )
        try:
            # Use direct Redis connection instead of database services
            # to avoid circular dependency
            import coredis

            client = coredis.Redis.from_url(redis_url)
            await client.ping()
            test_key = "config_validator_test"
            await client.set(test_key, "test_value", ex=5)
            value = await client.get(test_key)
            await client.delete(test_key)
            await client.close()

            if value == b"test_value":
                return ValidationResult(
                    component="redis_connectivity",
                    is_valid=True,
                    message="Redis connectivity test passed",
                    details={"test_method": "direct_coredis_connection"},
                )
            return ValidationResult(
                component="redis_connectivity",
                is_valid=False,
                message="Redis connectivity test failed - data integrity issue",
                details={"test_method": "direct_coredis_connection"},
            )
        except ImportError:
            return ValidationResult(
                component="redis_connectivity",
                is_valid=False,
                message="coredis not available for connectivity test",
                critical=False,
            )
        except Exception as e:
            return ValidationResult(
                component="redis_connectivity",
                is_valid=False,
                message=f"Redis connectivity failed: {e!s}",
                critical=False,
            )

    async def _validate_ml_configuration(self) -> ValidationResult:
        """Validate ML model configuration and accessibility."""
        ml_model_path = os.getenv("ML_MODEL_PATH", "./models")
        model_path = Path(ml_model_path)
        if not model_path.exists():
            return ValidationResult(
                component="ml_config",
                is_valid=False,
                message=f"ML model path does not exist: {ml_model_path}",
                critical=False,
            )
        if not model_path.is_dir():
            return ValidationResult(
                component="ml_config",
                is_valid=False,
                message=f"ML model path is not a directory: {ml_model_path}",
                critical=False,
            )
        model_files = list(model_path.glob("*.bin")) + list(model_path.glob("*.pkl"))
        return ValidationResult(
            component="ml_config",
            is_valid=True,
            message=f"ML configuration valid - found {len(model_files)} model files",
            details={
                "model_path": str(model_path),
                "model_files_count": len(model_files),
                "model_files": [f.name for f in model_files[:5]],
            },
        )

    async def _validate_api_configuration(self) -> ValidationResult:
        """Validate API rate limits and security settings."""
        results = []
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
        message = (
            "API configuration is valid"
            if is_valid
            else f"API configuration issues: {'; '.join(results)}"
        )
        return ValidationResult(
            component="api_config",
            is_valid=is_valid,
            message=message,
            details={"issues": results} if results else None,
            critical=False,
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
                issues.append(
                    "Redis timeout should be less than overall health timeout"
                )
            is_valid = len(issues) == 0
            message = (
                "Health check configuration is valid"
                if is_valid
                else f"Issues: {'; '.join(issues)}"
            )
            return ValidationResult(
                component="health_check_config",
                is_valid=is_valid,
                message=message,
                details={
                    "health_timeout": health_timeout_val,
                    "db_timeout": db_timeout_val,
                    "redis_timeout": redis_timeout_val,
                    "issues": issues,
                }
                if issues
                else {
                    "health_timeout": health_timeout_val,
                    "db_timeout": db_timeout_val,
                    "redis_timeout": redis_timeout_val,
                },
                critical=False,
            )
        except ValueError as e:
            return ValidationResult(
                component="health_check_config",
                is_valid=False,
                message=f"Invalid health check timeout values: {e!s}",
                critical=False,
            )

    async def _validate_monitoring_configuration(self) -> ValidationResult:
        """Validate monitoring and observability settings."""
        monitoring_enabled = os.getenv("MONITORING_ENABLED", "false").lower() == "true"
        if not monitoring_enabled:
            return ValidationResult(
                component="monitoring_config",
                is_valid=True,
                message="Monitoring disabled - no validation needed",
                critical=False,
            )
        monitoring_vars = [
            "METRICS_EXPORT_INTERVAL_SECONDS",
            "TRACING_ENABLED",
            "ALERT_WEBHOOK_URL",
        ]
        missing_vars = [var for var in monitoring_vars if not os.getenv(var)]
        if missing_vars:
            return ValidationResult(
                component="monitoring_config",
                is_valid=False,
                message=f"Monitoring enabled but missing configuration: {', '.join(missing_vars)}",
                details={"missing_vars": missing_vars},
                critical=False,
            )
        return ValidationResult(
            component="monitoring_config",
            is_valid=True,
            message="Monitoring configuration is valid",
        )


async def validate_startup_configuration(
    environment: str | None = None,
    validation_service: "ValidationServiceProtocol | None" = None
) -> tuple[bool, ValidationReport]:
    """Main entry point for startup configuration validation.

    Args:
        environment: Optional environment override
        validation_service: Optional validation service (will create default if not provided)

    Returns:
        Tuple of (is_valid, validation_report)
    """
    # Create default validation service if none provided
    if validation_service is None:
        from prompt_improver.services.prompt.validation_service import ValidationService
        validation_service = ValidationService()

    validator = ConfigurationValidator(validation_service, environment)
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


def save_validation_report(
    report: ValidationReport, output_file: str | None = None
) -> None:
    """Save validation report to JSON file."""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"config_validation_report_{timestamp}.json"
    output_path = Path(output_file)
    output_path.write_text(json.dumps(report.to_dict(), indent=2))
    logger.info(f"Configuration validation report saved to: {output_path}")


if __name__ == "__main__":
    "CLI entry point for configuration validation."
    import argparse

    parser = argparse.ArgumentParser(description="Validate application configuration")
    parser.add_argument("--environment", help="Environment to validate")
    parser.add_argument("--output", help="Output file for validation report")
    parser.add_argument(
        "--exit-on-failure",
        action="store_true",
        help="Exit with non-zero code on validation failure",
    )
    args = parser.parse_args()

    async def main():
        # Create validation service for CLI usage
        from prompt_improver.services.prompt.validation_service import ValidationService
        validation_service = ValidationService()

        is_valid, report = await validate_startup_configuration(
            args.environment, validation_service
        )
        if args.output:
            save_validation_report(report, args.output)
        if args.exit_on_failure and (not is_valid):
            sys.exit(1)

    asyncio.run(main())
