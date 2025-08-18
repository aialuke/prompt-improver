"""Database Configuration Module

Comprehensive database configuration with connection pooling, health checks,
and environment-specific settings.
"""

from typing import Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration with secure defaults and validation."""

    # Main database connection URL (takes precedence if provided)
    database_url: str | None = Field(
        default=None,
        description="Complete PostgreSQL database URL - overrides individual settings if provided",
    )

    # Individual connection parameters
    postgres_host: str = Field(
        description="PostgreSQL host - REQUIRED environment variable",
        min_length=1,
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

    # Connection Pool Configuration
    pool_min_size: int = Field(default=4, ge=1, le=50, description="Minimum pool size")
    pool_max_size: int = Field(
        default=16, ge=1, le=100, description="Maximum pool size"
    )
    pool_timeout: int = Field(
        default=10, ge=1, le=120, description="Pool timeout in seconds"
    )
    
    # Legacy pool settings for backward compatibility
    database_pool_size: int = Field(
        default=10, ge=1, le=100, description="Database connection pool size"
    )
    database_max_overflow: int = Field(
        default=20, ge=0, le=200, description="Database max overflow connections"
    )
    database_pool_timeout: int = Field(
        default=30, ge=1, le=300, description="Database pool timeout in seconds"
    )

    # Connection Settings
    ssl_mode: str = Field(default="prefer", description="SSL connection mode")
    pool_recycle: int = Field(
        default=3600, ge=300, le=86400, description="Pool recycle time in seconds"
    )
    
    # Health Check Settings
    health_check_timeout: float = Field(
        default=5.0, gt=0, le=60, description="Health check timeout in seconds"
    )
    health_check_query: str = Field(
        default="SELECT 1", description="Health check SQL query"
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

    @field_validator("ssl_mode")
    @classmethod
    def validate_ssl_mode(cls, v):
        """Validate SSL mode."""
        allowed = ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
        if v not in allowed:
            raise ValueError(f"ssl_mode must be one of {allowed}")
        return v

    def get_connection_url(self) -> str:
        """Get the complete database connection URL."""
        if self.database_url:
            return self.database_url
        
        # Build URL from individual components
        return (
            f"postgresql://{self.postgres_username}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
            f"?sslmode={self.ssl_mode}"
        )

    # DatabaseConfigProtocol implementation
    def get_database_url(self) -> str:
        """Get database connection URL (DatabaseConfigProtocol implementation)."""
        return self.get_connection_url()

    def get_connection_pool_config(self) -> dict[str, Any]:
        """Get connection pool configuration (DatabaseConfigProtocol implementation)."""
        return {
            "pool_size": self.pool_min_size,
            "max_overflow": self.pool_max_size - self.pool_min_size,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "pool_pre_ping": True,
            # Legacy compatibility
            "database_pool_size": self.database_pool_size,
            "database_max_overflow": self.database_max_overflow,
            "database_pool_timeout": self.database_pool_timeout,
        }

    def get_retry_config(self) -> dict[str, Any]:
        """Get retry configuration (DatabaseConfigProtocol implementation)."""
        return {
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 60.0,
            "exponential_base": 2.0,
            "jitter": True,
            "timeout": self.health_check_timeout,
            "health_check_timeout": self.health_check_timeout,
        }

    model_config = {
        "env_prefix": "POSTGRES_",
        "case_sensitive": False,
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "validate_assignment": True,
    }