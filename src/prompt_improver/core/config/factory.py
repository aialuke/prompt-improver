"""Configuration Factory and Loader Module.

Centralized configuration loading with environment-specific settings,
schema migration, and comprehensive error handling.
"""

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from prompt_improver.core.config.app_config import AppConfig
from prompt_improver.core.config.validation import (
    ValidationReport,
    validate_configuration,
)

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Configuration-related errors."""

    def __init__(self, message: str, config_section: str | None = None, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.config_section = config_section
        self.details = details or {}


class ConfigurationFactory:
    """Factory for creating and managing application configurations."""

    def __init__(self) -> None:
        """Initialize configuration factory."""
        self._cached_configs: dict[str, AppConfig] = {}

    def create_config(
        self,
        environment: str | None = None,
        env_file: str | None = None,
        validate: bool = True,
    ) -> AppConfig:
        """Create application configuration for specified environment.

        Args:
            environment: Target environment (development, testing, staging, production)
            env_file: Path to environment file (defaults to .env)
            validate: Whether to validate configuration after creation

        Returns:
            Configured AppConfig instance

        Raises:
            ConfigurationError: If configuration creation or validation fails
        """
        try:
            # Determine environment
            env = environment or os.getenv("ENVIRONMENT", "development")

            # Cache key for this configuration
            cache_key = f"{env}_{env_file or '.env'}"

            if cache_key in self._cached_configs:
                logger.debug(f"Returning cached configuration for {cache_key}")
                return self._cached_configs[cache_key]

            # Set environment variables if needed
            original_env = os.environ.get("ENVIRONMENT")
            if environment:
                os.environ["ENVIRONMENT"] = environment

            try:
                # Create configuration with optional env file
                config_kwargs = {}
                if env_file and Path(env_file).exists():
                    config_kwargs["_env_file"] = env_file
                    logger.info(f"Loading configuration from {env_file}")

                config = AppConfig(**config_kwargs)

                # Apply environment-specific overrides
                self._apply_environment_overrides(config, env)

                # Validate configuration if requested
                if validate:
                    import asyncio
                    is_valid, report = asyncio.run(validate_configuration(config))
                    if not is_valid:
                        error_msg = f"Configuration validation failed for environment '{env}'"
                        logger.error(error_msg)
                        for failure in report.critical_failures:
                            logger.error(f"  - {failure}")
                        raise ConfigurationError(
                            error_msg,
                            config_section="validation",
                            details={"failures": report.critical_failures}
                        )

                # Cache the configuration
                self._cached_configs[cache_key] = config
                logger.info(f"Successfully created configuration for environment: {env}")

                return config

            finally:
                # Restore original environment
                if original_env is not None:
                    os.environ["ENVIRONMENT"] = original_env
                elif "ENVIRONMENT" in os.environ:
                    del os.environ["ENVIRONMENT"]

        except ValidationError as e:
            error_msg = f"Configuration validation failed: {e!s}"
            logger.exception(error_msg)
            raise ConfigurationError(error_msg, config_section="pydantic_validation", details={"validation_errors": e.errors()})
        except Exception as e:
            error_msg = f"Failed to create configuration: {e!s}"
            logger.exception(error_msg)
            raise ConfigurationError(error_msg, details={"original_error": str(e)})

    def _apply_environment_overrides(self, config: AppConfig, environment: str) -> None:
        """Apply environment-specific configuration overrides."""
        logger.debug(f"Applying environment overrides for: {environment}")

        if environment == "development":
            # Development-specific overrides
            config.debug = True
            config.log_level = "DEBUG"
            config.security.security_profile = "development"
            config.monitoring.opentelemetry.sampling_rate = 1.0

        elif environment == "testing":
            # Testing-specific overrides
            config.debug = False
            config.log_level = "WARNING"
            config.security.security_profile = "testing"
            config.monitoring.opentelemetry.sampling_rate = 0.5
            config.database.pool_min_size = 2
            config.database.pool_max_size = 5

        elif environment == "staging":
            # Staging-specific overrides
            config.debug = False
            config.log_level = "INFO"
            config.security.security_profile = "staging"
            config.monitoring.opentelemetry.sampling_rate = 0.2

        elif environment == "production":
            # Production-specific overrides
            config.debug = False
            config.log_level = "INFO"
            config.security.security_profile = "production"
            config.monitoring.opentelemetry.sampling_rate = 0.1
            config.monitoring.enable_alerting = True

        logger.debug(f"Applied {environment} overrides")

    def create_for_testing(self, overrides: dict[str, Any] | None = None) -> AppConfig:
        """Create configuration optimized for testing.

        Args:
            overrides: Optional configuration overrides

        Returns:
            Test-optimized AppConfig instance
        """
        # Set test-specific environment variables
        test_env_vars = {
            "ENVIRONMENT": "testing",
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DATABASE": "test_prompt_improver",
            "POSTGRES_USERNAME": "postgres",
            "POSTGRES_PASSWORD": "postgres",
            "SECURITY_SECRET_KEY": "test-secret-key-for-testing-only-32chars-min",
        }

        # Store original values to restore later
        original_values = {}
        for key, value in test_env_vars.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            config = self.create_config(environment="testing", validate=False)

            # Apply test-specific overrides
            config.database.pool_min_size = 1
            config.database.pool_max_size = 3
            config.database.pool_timeout = 5
            config.security.authentication.max_failed_attempts_per_hour = 1000
            config.ml.enabled = False  # Disable ML for most tests

            # Apply custom overrides
            if overrides:
                for key, value in overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                    else:
                        logger.warning(f"Unknown configuration override: {key}")

            return config

        finally:
            # Restore original environment variables
            for key, original_value in original_values.items():
                if original_value is not None:
                    os.environ[key] = original_value
                elif key in os.environ:
                    del os.environ[key]

    def reload_config(self, environment: str | None = None) -> AppConfig:
        """Reload configuration, clearing cache.

        Args:
            environment: Target environment

        Returns:
            Freshly loaded AppConfig instance
        """
        env = environment or os.getenv("ENVIRONMENT", "development")

        # Clear relevant cached configurations
        keys_to_remove = [key for key in self._cached_configs if key.startswith(env)]
        for key in keys_to_remove:
            del self._cached_configs[key]

        logger.info(f"Cleared cached configurations for environment: {env}")

        return self.create_config(environment=environment)

    def clear_cache(self) -> None:
        """Clear all cached configurations."""
        self._cached_configs.clear()
        logger.info("Cleared all cached configurations")

    def get_environment_configs(self) -> dict[str, AppConfig]:
        """Get configurations for all environments (for testing/comparison).

        Returns:
            Dictionary mapping environment names to their configurations
        """
        environments = ["development", "testing", "staging", "production"]
        configs = {}

        for env in environments:
            try:
                configs[env] = self.create_config(environment=env, validate=False)
                logger.debug(f"Created configuration for environment: {env}")
            except Exception as e:
                logger.warning(f"Failed to create configuration for {env}: {e}")

        return configs


# Global factory instance
_config_factory: ConfigurationFactory | None = None


def get_config_factory() -> ConfigurationFactory:
    """Get the global configuration factory instance."""
    global _config_factory
    if _config_factory is None:
        _config_factory = ConfigurationFactory()
    return _config_factory


def create_config(
    environment: str | None = None,
    env_file: str | None = None,
    validate: bool = True,
) -> AppConfig:
    """Create application configuration using the global factory.

    Args:
        environment: Target environment
        env_file: Path to environment file
        validate: Whether to validate configuration

    Returns:
        Configured AppConfig instance
    """
    factory = get_config_factory()
    return factory.create_config(environment=environment, env_file=env_file, validate=validate)


def create_test_config(overrides: dict[str, Any] | None = None) -> AppConfig:
    """Create test configuration using the global factory.

    Args:
        overrides: Optional configuration overrides

    Returns:
        Test-optimized AppConfig instance
    """
    factory = get_config_factory()
    return factory.create_for_testing(overrides=overrides)


async def validate_config_file(config_path: str) -> ValidationReport:
    """Validate a configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        Validation report
    """
    try:
        config = create_config(env_file=config_path, validate=False)
        _is_valid, report = await validate_configuration(config)
        return report
    except Exception as e:
        # Create an error report
        from datetime import datetime

        from prompt_improver.core.config.validation import (
            ValidationReport,
            ValidationResult,
        )

        report = ValidationReport(
            timestamp=datetime.now(),
            environment="unknown",
            overall_valid=False,
        )

        report.results.append(
            ValidationResult(
                component="config_file",
                is_valid=False,
                message=f"Failed to load configuration file: {e!s}",
                critical=True,
            )
        )

        report.critical_failures = [f"config_file: Failed to load configuration file: {e!s}"]

        return report
