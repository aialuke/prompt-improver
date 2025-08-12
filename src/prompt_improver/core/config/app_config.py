"""Main Application Configuration Module

Unified configuration system with hierarchical structure and environment-based settings.
Consolidates all configuration concerns into a single entry point with proper validation.
"""

import os
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from prompt_improver.core.config.database_config import DatabaseConfig
from prompt_improver.core.config.ml_config import MLConfig  
from prompt_improver.core.config.monitoring_config import MonitoringConfig
from prompt_improver.core.config.security_config import SecurityConfig


class AppConfig(BaseSettings):
    """Main application configuration with hierarchical structure."""

    # Core Application Settings
    environment: str = Field(
        default="development", description="Application environment"
    )
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Global logging level")

    # Domain-specific configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    ml: MLConfig = Field(default_factory=MLConfig)

    # MCP Server Configuration
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
        "extra": "ignore",
        "validate_assignment": True,
    }


# Global configuration instance - lazy initialized
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reload_config() -> AppConfig:
    """Reload the configuration from environment."""
    global _config
    _config = AppConfig()
    return _config


def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return get_config().database


def get_security_config() -> SecurityConfig:
    """Get security configuration."""
    return get_config().security


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration."""
    return get_config().monitoring


def get_ml_config() -> MLConfig:
    """Get ML configuration."""
    return get_config().ml