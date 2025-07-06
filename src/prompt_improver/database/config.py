"""Database configuration using Pydantic Settings following 2025 best practices"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database configuration with environment variable support"""

    # PostgreSQL connection settings
    postgres_host: str = Field(default="localhost", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    postgres_database: str = Field(
        default="apes_production", validation_alias="POSTGRES_DATABASE"
    )
    postgres_username: str = Field(
        default="apes_user", validation_alias="POSTGRES_USERNAME"
    )
    postgres_password: str = Field(
        default="apes_secure_password_2024", validation_alias="POSTGRES_PASSWORD"
    )

    # Advanced connection pool settings (research-validated patterns)
    pool_min_size: int = Field(
        default=4, validation_alias="DB_POOL_MIN_SIZE"
    )  # Keep connections warm
    pool_max_size: int = Field(
        default=16, validation_alias="DB_POOL_MAX_SIZE"
    )  # Handle traffic spikes
    pool_timeout: int = Field(
        default=10, validation_alias="DB_POOL_TIMEOUT"
    )  # Fail fast pattern
    pool_max_lifetime: int = Field(
        default=1800, validation_alias="DB_POOL_MAX_LIFETIME"
    )  # 30min max age
    pool_max_idle: int = Field(
        default=300, validation_alias="DB_POOL_MAX_IDLE"
    )  # 5min idle timeout

    # Legacy SQLAlchemy pool settings (for backward compatibility)
    pool_size: int = Field(default=5, validation_alias="DB_POOL_SIZE")
    max_overflow: int = Field(default=10, validation_alias="DB_MAX_OVERFLOW")
    pool_recycle: int = Field(default=3600, validation_alias="DB_POOL_RECYCLE")

    # SQLAlchemy settings
    echo_sql: bool = Field(default=False, validation_alias="DB_ECHO_SQL")
    echo_pool: bool = Field(default=False, validation_alias="DB_ECHO_POOL")

    # Performance settings
    statement_timeout: int = Field(default=30, validation_alias="DB_STATEMENT_TIMEOUT")

    # Query performance targets (Phase 2 requirements)
    target_query_time_ms: int = Field(
        default=50, validation_alias="DB_TARGET_QUERY_TIME_MS"
    )
    target_cache_hit_ratio: float = Field(
        default=0.90, validation_alias="DB_TARGET_CACHE_HIT_RATIO"
    )

    @property
    def database_url(self) -> str:
        """Generate PostgreSQL connection URL"""
        return f"postgresql+asyncpg://{self.postgres_username}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"

    @property
    def database_url_sync(self) -> str:
        """Generate synchronous PostgreSQL connection URL for migrations"""
        return f"postgresql://{self.postgres_username}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
