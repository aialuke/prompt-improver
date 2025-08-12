"""Base configuration classes for the prompt-improver system."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel
from sqlmodel import Field, SQLModel

from prompt_improver.common.exceptions import ConfigurationError
from prompt_improver.common.types import ConfigDict

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel, ABC):
    """Base configuration class with common functionality."""

    def __init__(self, **data: Any):
        """Initialize and validate configuration.

        Args:
            **data: Model field values for this configuration instance.
        """
        super().__init__(**data)
        self.validate()

    @abstractmethod
    def validate(self) -> None:
        """Validate the configuration."""

    @classmethod
    def from_env(cls: type[T], prefix: str = "") -> T:
        """Create configuration from environment variables."""
        config_dict = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix) :].lower()
                config_dict[config_key] = value
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls: type[T], config_dict: ConfigDict) -> T:
        """Create configuration from dictionary."""
        field_names = set(cls.model_fields.keys())
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        try:
            return cls(**filtered_dict)
        except TypeError as e:
            raise ConfigurationError(f"Invalid configuration for {cls.__name__}: {e}")

    def model_dump(self) -> ConfigDict:
        """Convert configuration to dictionary."""
        result = {}
        for field_name in self.model_fields.keys():
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        return result

    def merge(self: T, other: T) -> T:
        """Merge with another configuration instance."""
        if not isinstance(other, self.__class__):
            raise ConfigurationError(
                f"Cannot merge {self.__class__} with {other.__class__}"
            )
        merged_dict = self.model_dump()
        other_dict = other.model_dump()
        for key, value in other_dict.items():
            if value is not None:
                merged_dict[key] = value
        return self.__class__.from_dict(merged_dict)


class DatabaseConfig(BaseConfig):
    """Database configuration."""

    host: str = Field(description="Database host - REQUIRED environment variable")
    port: int = Field(default=5432)
    database: str = Field(default="prompt_improver")
    username: str = Field(default="postgres")
    password: str = Field(default="")
    ssl_mode: str = Field(default="prefer")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    pool_timeout: int = Field(default=30)
    pool_recycle: int = Field(default=3600)

    def validate(self) -> None:
        """Validate database configuration."""
        if not self.host:
            raise ConfigurationError("Database host is required")
        if not self.database:
            raise ConfigurationError("Database name is required")
        if not self.username:
            raise ConfigurationError("Database username is required")
        if self.port <= 0 or self.port > 65535:
            raise ConfigurationError("Database port must be between 1 and 65535")
        if self.pool_size <= 0:
            raise ConfigurationError("Pool size must be positive")


class RedisConfig(BaseConfig):
    """Redis configuration."""

    host: str = Field(description="Redis host - REQUIRED environment variable")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: str | None = Field(default=None)
    ssl: bool = Field(default=False)
    max_connections: int = Field(default=10)
    socket_timeout: float = Field(default=5.0)
    socket_connect_timeout: float = Field(default=5.0)

    def validate(self) -> None:
        """Validate Redis configuration."""
        if not self.host:
            raise ConfigurationError("Redis host is required")
        if self.port <= 0 or self.port > 65535:
            raise ConfigurationError("Redis port must be between 1 and 65535")
        if self.db < 0:
            raise ConfigurationError("Redis database number must be non-negative")
        if self.max_connections <= 0:
            raise ConfigurationError("Max connections must be positive")


class SecurityConfig(BaseConfig):
    """Security configuration."""

    secret_key: str = Field(default="")
    token_expiry_seconds: int = Field(default=3600)
    hash_rounds: int = Field(default=12)
    max_login_attempts: int = Field(default=5)
    rate_limit_requests: int = Field(default=100)
    rate_limit_window: int = Field(default=60)

    def validate(self) -> None:
        """Validate security configuration."""
        if not self.secret_key:
            raise ConfigurationError("Secret key is required")
        if len(self.secret_key) < 32:
            raise ConfigurationError("Secret key must be at least 32 characters")
        if self.token_expiry_seconds <= 0:
            raise ConfigurationError("Token expiry must be positive")
        if self.hash_rounds < 4 or self.hash_rounds > 20:
            raise ConfigurationError("Hash rounds must be between 4 and 20")


class MLConfig(BaseConfig):
    """ML configuration."""

    model_timeout: float = Field(default=120.0)
    max_feature_dimension: int = Field(default=10000)
    default_batch_size: int = Field(default=32)
    max_batch_size: int = Field(default=1000)
    model_cache_size: int = Field(default=100)
    training_data_path: str = Field(default="data/training")
    model_storage_path: str = Field(default="models")

    def validate(self) -> None:
        """Validate ML configuration."""
        if self.model_timeout <= 0:
            raise ConfigurationError("Model timeout must be positive")
        if self.max_feature_dimension <= 0:
            raise ConfigurationError("Max feature dimension must be positive")
        if self.default_batch_size <= 0:
            raise ConfigurationError("Default batch size must be positive")
        if self.max_batch_size < self.default_batch_size:
            raise ConfigurationError("Max batch size must be >= default batch size")


class MonitoringConfig(BaseConfig):
    """Monitoring configuration."""

    metrics_enabled: bool = Field(default=True)
    health_check_interval: float = Field(default=30.0)
    metrics_collection_interval: float = Field(default=60.0)
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    opentelemetry_endpoint: str | None = Field(default=None)

    def validate(self) -> None:
        """Validate monitoring configuration."""
        if self.health_check_interval <= 0:
            raise ConfigurationError("Health check interval must be positive")
        if self.metrics_collection_interval <= 0:
            raise ConfigurationError("Metrics collection interval must be positive")
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ConfigurationError("Invalid log level")
        if self.log_format not in ["json", "text"]:
            raise ConfigurationError("Log format must be 'json' or 'text'")


def merge_configs(*configs: BaseConfig) -> ConfigDict:
    """Merge multiple configuration objects into a single dictionary."""
    merged = {}
    for config in configs:
        if config is not None:
            config_dict = config.model_dump()
            merged.update(config_dict)
    return merged
