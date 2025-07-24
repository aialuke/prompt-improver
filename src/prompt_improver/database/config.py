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
        default="apes_user",
        validation_alias="POSTGRES_USERNAME",
        description="PostgreSQL username - defaults to 'apes_user' to match Docker setup",
    )
    postgres_password: str = Field(
        validation_alias="POSTGRES_PASSWORD",
        description="PostgreSQL password from environment variable (required for security)",
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
        """Generate PostgreSQL connection URL using psycopg3 async driver"""
        return f"postgresql+psycopg://{self.postgres_username}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"

    @property
    def database_url_sync(self) -> str:
        """Generate synchronous PostgreSQL connection URL for migrations"""
        return f"postgresql+psycopg://{self.postgres_username}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"

    @property
    def psycopg_connection_string(self) -> str:
        """Generate psycopg3 connection string for direct psycopg.connect() usage"""
        return f"postgresql://{self.postgres_username}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"

    # Additional environment variables (allowing extra fields for flexibility)
    external_database_url: str | None = Field(
        default=None, validation_alias="DATABASE_URL"
    )
    test_db_name: str | None = Field(default=None, validation_alias="TEST_DB_NAME")
    test_database_url: str | None = Field(
        default=None, validation_alias="TEST_DATABASE_URL"
    )

    # MCP configuration
    mcp_postgres_enabled: bool = Field(
        default=True, validation_alias="MCP_POSTGRES_ENABLED"
    )
    mcp_postgres_connection_string: str | None = Field(
        default=None, validation_alias="MCP_POSTGRES_CONNECTION_STRING"
    )

    # General application settings
    development_mode: bool = Field(default=True, validation_alias="DEVELOPMENT_MODE")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    enable_performance_monitoring: bool = Field(
        default=True, validation_alias="ENABLE_PERFORMANCE_MONITORING"
    )
    slow_query_threshold: int = Field(
        default=1000, validation_alias="SLOW_QUERY_THRESHOLD"
    )

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

def get_database_config() -> DatabaseConfig:
    """Get database configuration instance."""
    return DatabaseConfig()
