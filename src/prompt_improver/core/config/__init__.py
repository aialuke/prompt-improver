"""Core Configuration Module

Unified configuration system with hierarchical structure, environment-based settings,
comprehensive validation, and centralized management.

This module consolidates all configuration concerns into a clean, testable hierarchy
that replaces multiple overlapping configuration systems throughout the codebase.

Example:
    ```python
    from prompt_improver.core.config import get_config, create_test_config
    
    # Get production configuration
    config = get_config()
    
    # Access domain-specific configs
    db_config = config.database
    security_config = config.security
    
    # Create test configuration
    test_config = create_test_config({"debug": True})
    ```

Key Components:
    - AppConfig: Main application configuration with hierarchical structure
    - DatabaseConfig: Database connection and pooling settings
    - SecurityConfig: Authentication, validation, and cryptography settings
    - MonitoringConfig: Observability, metrics, and health check settings
    - MLConfig: Machine learning pipeline and model serving settings
    - ConfigurationFactory: Environment-aware configuration creation
    - validate_configuration: Comprehensive configuration validation
"""

# Main configuration classes
from prompt_improver.core.config.app_config import (
    AppConfig,
    get_config,
    get_database_config,
    get_ml_config,
    get_monitoring_config,
    get_security_config,
    reload_config,
)

# Domain-specific configurations
from prompt_improver.core.config.database_config import DatabaseConfig
from prompt_improver.core.config.ml_config import MLConfig
from prompt_improver.core.config.monitoring_config import MonitoringConfig
from prompt_improver.core.config.security_config import SecurityConfig

# Factory and validation
from prompt_improver.core.config.factory import (
    ConfigurationError,
    ConfigurationFactory,
    create_config,
    create_test_config,
    get_config_factory,
    validate_config_file,
)
from prompt_improver.core.config.validation import (
    ConfigurationValidator,
    ValidationReport,
    ValidationResult,
    validate_configuration,
)

# Consolidated configuration components (new in 2025)
from prompt_improver.core.config.schema import (
    ConfigSchemaVersion,
    ConfigMigration,
    ConfigurationSchema,
    UnifiedConfigService,
    schema_manager,
    migrate_configuration,
)
from prompt_improver.core.config.logging import (
    ConfigurationEvent,
    ConfigurationLogger,
    get_config_logger,
    setup_config_logging,
)
from prompt_improver.core.config.validator import (
    ConfigurationValidator,
    validate_startup_configuration,
    save_validation_report,
)
from prompt_improver.core.config.textstat import (
    TextStatConfig,
    TextStatMetrics,
    TextStatWrapper,
    get_textstat_wrapper,
    text_analysis,
)
from prompt_improver.core.config.retry import (
    RetryStrategy,
    RetryConfig,
    StandardRetryConfigs,
    create_retry_config,
)

__all__ = [
    # Main configuration
    "AppConfig",
    "get_config",
    "reload_config",
    
    # Domain-specific configurations
    "DatabaseConfig", 
    "SecurityConfig",
    "MonitoringConfig",
    "MLConfig",
    
    # Convenience getters
    "get_database_config",
    "get_security_config", 
    "get_monitoring_config",
    "get_ml_config",
    
    # Factory and creation
    "ConfigurationFactory",
    "ConfigurationError",
    "get_config_factory",
    "create_config",
    "create_test_config",
    
    # Validation
    "ConfigurationValidator",
    "ValidationResult",
    "ValidationReport", 
    "validate_configuration",
    "validate_config_file",
    "validate_startup_configuration",
    "save_validation_report",
    
    # Schema management and migration
    "ConfigSchemaVersion",
    "ConfigMigration",
    "ConfigurationSchema",
    "UnifiedConfigService",
    "schema_manager",
    "migrate_configuration",
    
    # Configuration logging
    "ConfigurationEvent",
    "ConfigurationLogger",
    "get_config_logger",
    "setup_config_logging",
    
    # TextStat configuration
    "TextStatConfig",
    "TextStatMetrics", 
    "TextStatWrapper",
    "get_textstat_wrapper",
    "text_analysis",
    
    # Retry configuration
    "RetryStrategy",
    "RetryConfig",
    "StandardRetryConfigs", 
    "create_retry_config",
]