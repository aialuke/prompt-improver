"""Comprehensive Centralized Pydantic Configuration System
Following 2025 best practices for production-ready configuration management.

This module provides a unified configuration system with environment-based settings,
validation, and secure credential management.

DEPRECATION NOTICE: This module is being migrated to the new configuration hierarchy.
Use `from prompt_improver.core.config import get_config` instead of importing classes directly.
"""

import os
import logging
from typing import Any, Dict, List, Optional

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings

# Import from new unified system
from .config import AppConfig as NewAppConfig, get_config as get_new_config

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseSettings):
    """Database configuration with secure defaults."""

    database_url: str = Field(
        description="PostgreSQL database URL - REQUIRED environment variable",
        min_length=1,
    )
    database_pool_size: int = Field(
        default=10, ge=1, le=100, description="Database connection pool size"
    )
    database_max_overflow: int = Field(
        default=20, ge=0, le=200, description="Database max overflow connections"
    )
    database_pool_timeout: int = Field(
        default=30, ge=1, le=300, description="Database pool timeout in seconds"
    )

    # Additional fields for granular control
    postgres_host: str = Field(
        min_length=1, description="PostgreSQL host - REQUIRED environment variable"
    )
    postgres_port: int = Field(
        default=5432, ge=1, le=65535, description="PostgreSQL port"
    )
    postgres_database: str = Field(
        default="prompt_improver", min_length=1, description="PostgreSQL database name"
    )
    postgres_username: str = Field(
        default="postgres", min_length=1, description="PostgreSQL username"
    )
    postgres_password: str = Field(
        description="PostgreSQL password - REQUIRED environment variable"
    )
    pool_min_size: int = Field(default=4, ge=1, le=50, description="Minimum pool size")
    pool_max_size: int = Field(
        default=16, ge=1, le=100, description="Maximum pool size"
    )
    pool_timeout: int = Field(
        default=10, ge=1, le=120, description="Pool timeout in seconds"
    )

    @field_validator("pool_max_size")
    @classmethod
    def validate_pool_sizes(cls, v, info):
        """Ensure max pool size is not less than min pool size."""
        if "pool_min_size" in info.data and v < info.data["pool_min_size"]:
            raise ValueError(
                "pool_max_size must be greater than or equal to pool_min_size"
            )
        return v

    model_config = {
        "env_prefix": "POSTGRES_",
        "case_sensitive": False,
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


class RedisConfig(BaseSettings):
    """Redis configuration for external Redis service."""

    redis_url: str = Field(
        description="Redis connection URL - REQUIRED environment variable",
        min_length=1,
    )
    redis_max_connections: int = Field(
        default=100, ge=1, le=1000, description="Redis max connections"
    )
    redis_timeout: int = Field(
        default=5, ge=1, le=60, description="Redis timeout in seconds"
    )

    # Additional fields for granular control
    host: str = Field(
        description="Redis host - REQUIRED environment variable",
        min_length=1,
    )
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    password: str | None = Field(default=None, description="Redis password")
    ssl: bool = Field(default=False, description="Use SSL connection")
    max_connections: int = Field(
        default=10, ge=1, le=1000, description="Maximum connections"
    )

    model_config = {
        "env_prefix": "REDIS_",
        "case_sensitive": False,
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


class MCPConfig(BaseSettings):
    """MCP Server configuration."""

    mcp_host: str = Field(
        default="0.0.0.0", description="MCP server host (0.0.0.0 for external access)"
    )
    mcp_port: int = Field(default=8080, description="MCP server port")
    mcp_timeout: int = Field(default=30, description="MCP timeout in seconds")
    mcp_batch_size: int = Field(default=100, description="MCP batch processing size")
    mcp_session_maxsize: int = Field(
        default=1000, description="MCP session cache max size"
    )
    mcp_session_ttl: int = Field(default=3600, description="MCP session TTL in seconds")
    mcp_session_cleanup_interval: int = Field(
        default=300, description="MCP session cleanup interval in seconds"
    )

    model_config = {
        "env_prefix": "MCP_",
        "case_sensitive": False,
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


class SecurityConfig(BaseSettings):
    """Security configuration."""

    secret_key: str = Field(
        ...,  # Explicitly required, no default
        min_length=32,
        description="Application secret key (min 32 chars) - REQUIRED environment variable SECURITY_SECRET_KEY",
    )
    encryption_key: str | None = Field(
        default=None, description="Encryption key for sensitive data"
    )
    token_expiry_seconds: int = Field(
        default=3600, ge=1, le=86400, description="Token expiry time in seconds"
    )
    hash_rounds: int = Field(
        default=12, ge=4, le=20, description="Password hash rounds"
    )
    max_login_attempts: int = Field(
        default=5, ge=1, le=20, description="Maximum login attempts"
    )

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v):
        """Validate secret key requirements."""
        if not v or len(v.strip()) == 0:
            raise ValueError(
                "secret_key is required. Set the SECURITY_SECRET_KEY environment variable with at least 32 characters."
            )
        if len(v) < 32:
            raise ValueError(
                f"secret_key must be at least 32 characters long, got {len(v)} characters. "
                f"Set a longer value for SECURITY_SECRET_KEY environment variable."
            )
        if (
            "dev-secret-key" in v.lower()
            and os.getenv("ENVIRONMENT", "development") == "production"
        ):
            raise ValueError(
                "Development secret key detected in production environment. "
                "Set a production-safe SECURITY_SECRET_KEY environment variable."
            )
        return v

    model_config = {
        "env_prefix": "SECURITY_",
        "case_sensitive": False,
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration."""

    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    health_check_interval: float = Field(
        default=30.0, gt=0, le=300, description="Health check interval in seconds"
    )
    metrics_collection_interval: float = Field(
        default=60.0,
        gt=0,
        le=3600,
        description="Metrics collection interval in seconds",
    )
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format")
    opentelemetry_endpoint: str | None = Field(
        default=None, description="OpenTelemetry endpoint URL"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v):
        allowed = ["json", "text"]
        if v.lower() not in allowed:
            raise ValueError(f"Log format must be one of {allowed}")
        return v.lower()

    @field_validator("opentelemetry_endpoint")
    @classmethod
    def validate_url(cls, v):
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("OpenTelemetry endpoint must be a valid URL")
        return v

    model_config = {
        "env_prefix": "MONITORING_",
        "case_sensitive": False,
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


class MLConfig(BaseSettings):
    """Machine Learning services configuration."""

    # ML Database Configuration (Optional - only needed for ML features)
    ml_postgres_host: str = Field(
        default_factory=lambda: os.getenv("ML_POSTGRES_HOST", "ml-postgres"),
        description="ML PostgreSQL host - from ML_POSTGRES_HOST env var, defaults to ml-postgres",
        min_length=1,
    )
    ml_postgres_port: int = Field(
        default=5432, ge=1, le=65535, description="ML PostgreSQL port"
    )
    ml_postgres_database: str = Field(
        default="ml_automl", min_length=1, description="ML PostgreSQL database name"
    )
    ml_postgres_username: str = Field(
        default="postgres",
        description="ML PostgreSQL username - optional, defaults to postgres",
        min_length=1,
    )
    ml_postgres_password: str = Field(
        default="postgres",
        description="ML PostgreSQL password - optional, defaults to postgres",
    )

    # ML Redis Configuration (Optional - only needed for ML features)
    ml_redis_host: str = Field(
        default_factory=lambda: os.getenv("ML_REDIS_HOST", "ml-redis"),
        description="ML Redis host - from ML_REDIS_HOST env var, defaults to ml-redis",
        min_length=1,
    )
    ml_redis_port: int = Field(
        default=6379, ge=1, le=65535, description="ML Redis port"
    )
    ml_redis_password: str | None = Field(default=None, description="ML Redis password")
    ml_redis_db: int = Field(
        default=1, ge=0, le=15, description="ML Redis database number"
    )

    # MLFlow Configuration (Optional - only needed for ML features)
    mlflow_tracking_uri: str = Field(
        default_factory=lambda: os.getenv(
            "ML_MLFLOW_TRACKING_URI", "http://mlflow:5000"
        ),
        description="MLFlow tracking server URI - from ML_MLFLOW_TRACKING_URI env var, defaults to mlflow:5000",
        min_length=1,
    )
    mlflow_host: str = Field(
        default_factory=lambda: os.getenv("ML_MLFLOW_HOST", "mlflow"),
        description="MLFlow server host - from ML_MLFLOW_HOST env var, defaults to mlflow",
        min_length=1,
    )
    mlflow_port: int = Field(
        default=5000, ge=1, le=65535, description="MLFlow server port"
    )

    model_config = {
        "env_prefix": "ML_",
        "case_sensitive": False,
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


class ExternalAPIConfig(BaseSettings):
    """External API services configuration."""

    # OpenAI API Configuration
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_api_url: str = Field(
        default="https://api.openai.com/v1", description="OpenAI API base URL"
    )

    # HuggingFace Configuration
    huggingface_api_key: str | None = Field(
        default=None, description="HuggingFace API key"
    )
    huggingface_api_url: str = Field(
        default="https://huggingface.co/api", description="HuggingFace API base URL"
    )

    # Generic external service configurations
    apes_api_url: str = Field(
        default="https://api.apes.local", description="APES API base URL"
    )

    model_config = {
        "env_prefix": "EXTERNAL_API_",
        "case_sensitive": False,
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


class TelemetryConfig(BaseSettings):
    """OpenTelemetry and observability configuration."""

    # OpenTelemetry Core Configuration
    otel_service_name: str = Field(
        default="apes-prompt-improver", description="OpenTelemetry service name"
    )
    otel_service_version: str = Field(
        default="1.0.0", description="Service version for telemetry"
    )
    otel_environment: str = Field(
        default="development", description="Environment for telemetry"
    )

    # OpenTelemetry Exporter Configuration (Optional - defaults to console)
    otel_exporter_otlp_endpoint: str = Field(
        default_factory=lambda: os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"
        ),
        description="OpenTelemetry OTLP gRPC endpoint - from OTEL_EXPORTER_OTLP_ENDPOINT env var, defaults to otel-collector:4317",
        min_length=1,
    )
    otel_exporter_otlp_http_endpoint: str = Field(
        default_factory=lambda: os.getenv(
            "OTEL_EXPORTER_OTLP_HTTP_ENDPOINT", "http://otel-collector:4318"
        ),
        description="OpenTelemetry OTLP HTTP endpoint - from OTEL_EXPORTER_OTLP_HTTP_ENDPOINT env var, defaults to otel-collector:4318",
        min_length=1,
    )
    otel_exporter_otlp_insecure: bool = Field(
        default=True, description="Use insecure connection for OTLP"
    )

    # Tracing Configuration
    otel_trace_exporter: str = Field(
        default="console", description="Trace exporter type"
    )
    otel_metric_exporter: str = Field(
        default="console", description="Metric exporter type"
    )
    otel_sampling_strategy: str = Field(
        default="parent_based", description="Sampling strategy"
    )
    otel_sampling_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Sampling rate"
    )
    otel_rate_limit_per_second: int = Field(
        default=100, ge=1, le=10000, description="Rate limit for telemetry"
    )

    # Feature Toggles
    otel_tracing_enabled: bool = Field(
        default=True, description="Enable distributed tracing"
    )
    otel_metrics_enabled: bool = Field(
        default=True, description="Enable metrics collection"
    )
    otel_logging_enabled: bool = Field(
        default=True, description="Enable structured logging"
    )
    otel_auto_instrumentation_enabled: bool = Field(
        default=True, description="Enable automatic instrumentation"
    )

    model_config = {
        "env_prefix": "OTEL_",
        "case_sensitive": False,
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


class AppConfig(BaseSettings):
    """Main application configuration."""

    environment: str = Field(
        default="development", description="Application environment"
    )
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    database: DatabaseConfig = Field(default_factory=lambda: DatabaseConfig())
    redis: RedisConfig = Field(default_factory=lambda: RedisConfig())
    mcp: MCPConfig = Field(default_factory=lambda: MCPConfig())
    security: SecurityConfig = Field(default_factory=lambda: SecurityConfig())
    monitoring: MonitoringConfig = Field(default_factory=lambda: MonitoringConfig())
    ml: MLConfig = Field(default_factory=lambda: MLConfig())
    external_api: ExternalAPIConfig = Field(default_factory=lambda: ExternalAPIConfig())
    telemetry: TelemetryConfig = Field(default_factory=lambda: TelemetryConfig())

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = ["development", "testing", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Allow extra fields to be ignored
    }


# Backward compatibility - delegate to new system
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get the global configuration instance.
    
    DEPRECATED: Use `from prompt_improver.core.config import get_config` instead.
    """
    global _config
    if _config is None:
        # Try to use new system first, fallback to old system
        try:
            new_config = get_new_config()
            # Convert to old AppConfig format for backward compatibility
            _config = AppConfig(
                environment=new_config.environment,
                debug=new_config.debug,
                log_level=new_config.log_level,
                database=DatabaseConfig(
                    database_url=new_config.database.get_connection_url(),
                    database_pool_size=new_config.database.pool_max_size,
                    database_max_overflow=new_config.database.database_max_overflow,
                    database_pool_timeout=new_config.database.database_pool_timeout,
                    postgres_host=new_config.database.postgres_host,
                    postgres_port=new_config.database.postgres_port,
                    postgres_database=new_config.database.postgres_database,
                    postgres_username=new_config.database.postgres_username,
                    postgres_password=new_config.database.postgres_password,
                    pool_min_size=new_config.database.pool_min_size,
                    pool_max_size=new_config.database.pool_max_size,
                    pool_timeout=new_config.database.pool_timeout,
                ),
                security=SecurityConfig(
                    secret_key=new_config.security.secret_key,
                    encryption_key=new_config.security.encryption_key,
                    token_expiry_seconds=new_config.security.token_expiry_seconds,
                    hash_rounds=new_config.security.hash_rounds,
                    max_login_attempts=new_config.security.max_login_attempts,
                ),
                monitoring=MonitoringConfig(
                    metrics_enabled=new_config.monitoring.enabled,
                    health_check_interval=new_config.monitoring.health_checks.interval_seconds,
                    metrics_collection_interval=new_config.monitoring.metrics.collection_interval_seconds,
                    log_level=new_config.monitoring.logging.level,
                    log_format=new_config.monitoring.logging.format,
                    opentelemetry_endpoint=new_config.monitoring.opentelemetry.otlp_endpoint,
                ),
                ml=MLConfig(
                    ml_postgres_host=new_config.ml.external_services.postgres_host,
                    ml_postgres_port=new_config.ml.external_services.postgres_port,
                    ml_postgres_database=new_config.ml.external_services.postgres_database,
                    ml_postgres_username=new_config.ml.external_services.postgres_username,
                    ml_postgres_password=new_config.ml.external_services.postgres_password,
                    ml_redis_host=new_config.ml.external_services.redis_host,
                    ml_redis_port=new_config.ml.external_services.redis_port,
                    ml_redis_password=new_config.ml.external_services.redis_password,
                    ml_redis_db=new_config.ml.external_services.redis_db,
                    mlflow_tracking_uri=new_config.ml.external_services.mlflow_tracking_uri,
                    mlflow_host=new_config.ml.external_services.mlflow_host,
                    mlflow_port=new_config.ml.external_services.mlflow_port,
                ),
                mcp_host=new_config.mcp_host,
                mcp_port=new_config.mcp_port,
                mcp_timeout=new_config.mcp_timeout,
                mcp_batch_size=new_config.mcp_batch_size,
            )
            logger.info("Using new configuration system with backward compatibility")
        except Exception as e:
            logger.warning(f"Failed to load new config system, falling back to old system: {e}")
            _config = AppConfig()
    return _config


def reload_config() -> AppConfig:
    """Reload the configuration from environment.
    
    DEPRECATED: Use `from prompt_improver.core.config import reload_config` instead.
    """
    global _config
    _config = None
    return get_config()
