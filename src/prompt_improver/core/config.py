"""
Comprehensive Centralized Pydantic Configuration System
Following 2025 best practices for production-ready configuration management.

This module provides a unified configuration system with:
- Environment-specific loading with .env file support
- Type validation with comprehensive defaults
- Nested configuration models for different system components
- Hot-reload capability using file watchers
- Startup validation and health checks
- Support for secrets management and cloud integrations
"""

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    PostgresDsn,
    RedisDsn,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Models (Nested Categories)
# =============================================================================

class EnvironmentConfig(BaseModel):
    """Core environment and application settings."""
    
    environment: str = Field(
        default="development",
        description="Application environment (development, staging, production)",
        validation_alias=AliasChoices("ENVIRONMENT", "APP_ENV", "ENV")
    )
    debug: bool = Field(
        default=True,
        description="Enable debug mode",
        validation_alias="DEBUG"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
        validation_alias="LOG_LEVEL"
    )
    app_name: str = Field(
        default="prompt-improver",
        description="Application name",
        validation_alias="APP_NAME"
    )
    version: str = Field(
        default="1.0.0",
        description="Application version",
        validation_alias="APP_VERSION"
    )
    secret_key: Optional[str] = Field(
        default=None,
        description="Secret key for cryptographic operations",
        validation_alias=AliasChoices("SECRET_KEY", "APP_SECRET_KEY")
    )
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        valid_envs = {"development", "staging", "production", "test"}
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v.lower()
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

class DatabaseConfig(BaseSettings):
    """PostgreSQL database configuration with connection pooling and JSONB optimization."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Connection settings
    host: str = Field(
        default="localhost",
        description="PostgreSQL host",
        validation_alias="POSTGRES_HOST"
    )
    port: int = Field(
        default=5432,
        description="PostgreSQL port",
        validation_alias="POSTGRES_PORT"
    )
    database: str = Field(
        default="apes_production",
        description="Database name",
        validation_alias=AliasChoices("POSTGRES_DATABASE", "DB_NAME")
    )
    username: str = Field(
        default="apes_user",
        description="Database username",
        validation_alias=AliasChoices("POSTGRES_USERNAME", "DB_USER")
    )
    password: str = Field(
        default="default_dev_password",
        description="Database password",
        validation_alias=AliasChoices("POSTGRES_PASSWORD", "DB_PASSWORD")
    )
    
    # Connection pool settings (2025 best practices)
    pool_min_size: int = Field(
        default=4,
        description="Minimum connection pool size",
        validation_alias="DB_POOL_MIN_SIZE",
        ge=1,
        le=50
    )
    pool_max_size: int = Field(
        default=16,
        description="Maximum connection pool size",
        validation_alias="DB_POOL_MAX_SIZE",
        ge=1,
        le=100
    )
    pool_timeout: int = Field(
        default=10,
        description="Connection timeout in seconds",
        validation_alias="DB_POOL_TIMEOUT",
        ge=1,
        le=60
    )
    pool_max_lifetime: int = Field(
        default=1800,
        description="Maximum connection lifetime in seconds (30 min)",
        validation_alias="DB_POOL_MAX_LIFETIME",
        ge=300,
        le=7200
    )
    pool_max_idle: int = Field(
        default=300,
        description="Maximum idle time in seconds (5 min)",
        validation_alias="DB_POOL_MAX_IDLE",
        ge=60,
        le=1800
    )
    
    # Performance settings
    statement_timeout: int = Field(
        default=30,
        description="Statement timeout in seconds",
        validation_alias="DB_STATEMENT_TIMEOUT",
        ge=1,
        le=300
    )
    echo_sql: bool = Field(
        default=False,
        description="Log SQL statements",
        validation_alias="DB_ECHO_SQL"
    )
    
    # Query performance targets
    target_query_time_ms: int = Field(
        default=50,
        description="Target query time in milliseconds",
        validation_alias="DB_TARGET_QUERY_TIME_MS",
        ge=1,
        le=5000
    )
    target_cache_hit_ratio: float = Field(
        default=0.90,
        description="Target cache hit ratio",
        validation_alias="DB_TARGET_CACHE_HIT_RATIO",
        ge=0.0,
        le=1.0
    )
    
    # Connection Pool Optimizer Settings (Performance Engineering)
    pool_target_utilization: float = Field(
        default=0.7,
        description="Target pool utilization percentage (70%)",
        validation_alias="DB_POOL_TARGET_UTILIZATION",
        ge=0.1,
        le=0.95
    )
    pool_high_utilization_threshold: float = Field(
        default=0.9,
        description="High utilization threshold for pool scaling (90%)",
        validation_alias="DB_POOL_HIGH_UTILIZATION_THRESHOLD",
        ge=0.5,
        le=0.99
    )
    pool_low_utilization_threshold: float = Field(
        default=0.3,
        description="Low utilization threshold for pool scaling (30%)",
        validation_alias="DB_POOL_LOW_UTILIZATION_THRESHOLD",
        ge=0.1,
        le=0.8
    )
    pool_stress_threshold: float = Field(
        default=0.8,
        description="Stress threshold for optimal pool size calculation (80%)",
        validation_alias="DB_POOL_STRESS_THRESHOLD",
        ge=0.5,
        le=0.95
    )
    pool_underutilized_threshold: float = Field(
        default=0.4,
        description="Underutilized threshold for pool size reduction (40%)",
        validation_alias="DB_POOL_UNDERUTILIZED_THRESHOLD",
        ge=0.1,
        le=0.7
    )
    pool_wait_threshold: int = Field(
        default=5,
        description="Maximum acceptable waiting requests before scaling",
        validation_alias="DB_POOL_WAIT_THRESHOLD",
        ge=1,
        le=50
    )
    pool_increase_step: int = Field(
        default=5,
        description="Number of connections to add when scaling up",
        validation_alias="DB_POOL_INCREASE_STEP",
        ge=1,
        le=20
    )
    pool_decrease_step: int = Field(
        default=3,
        description="Number of connections to remove when scaling down",
        validation_alias="DB_POOL_DECREASE_STEP",
        ge=1,
        le=10
    )
    pool_scale_up_factor: float = Field(
        default=1.2,
        description="Scale up factor for optimal pool size (20% increase)",
        validation_alias="DB_POOL_SCALE_UP_FACTOR",
        ge=1.1,
        le=2.0
    )
    pool_scale_down_factor: float = Field(
        default=0.9,
        description="Scale down factor for optimal pool size (10% decrease)",
        validation_alias="DB_POOL_SCALE_DOWN_FACTOR",
        ge=0.5,
        le=0.95
    )
    
    # Connection Age Tracking Settings
    pool_registry_size: int = Field(
        default=1000,
        description="Maximum number of connection records to track",
        validation_alias="DB_POOL_REGISTRY_SIZE",
        ge=100,
        le=5000
    )
    pool_max_connection_age_hours: int = Field(
        default=24,
        description="Maximum age for connection tracking (hours)",
        validation_alias="DB_POOL_MAX_CONNECTION_AGE_HOURS",
        ge=1,
        le=168  # 1 week
    )
    
    @model_validator(mode='after')
    def validate_pool_settings(self) -> 'DatabaseConfig':
        """Validate pool configuration consistency."""
        if self.pool_max_size < self.pool_min_size:
            raise ValueError("pool_max_size must be >= pool_min_size")
        return self
    
    @property
    def database_url(self) -> str:
        """Generate async PostgreSQL connection URL with JSONB optimization."""
        base_url = f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        # Add JSONB optimizations for 2025 best practices
        return f"{base_url}?options=-c%20default_text_search_config=pg_catalog.english"
    
    @property
    def database_url_sync(self) -> str:
        """Generate sync PostgreSQL connection URL with JSONB optimization."""
        base_url = f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        # Add JSONB optimizations for 2025 best practices
        return f"{base_url}?options=-c%20default_text_search_config=pg_catalog.english"

class RedisConfig(BaseModel):
    """Redis configuration for caching and rate limiting with enhanced security."""
    
    # Connection settings
    host: str = Field(
        default="localhost",
        description="Redis host",
        validation_alias="REDIS_HOST"
    )
    port: int = Field(
        default=6379,
        description="Redis port",
        validation_alias="REDIS_PORT",
        ge=1,
        le=65535
    )
    database: int = Field(
        default=0,
        description="Redis database number",
        validation_alias="REDIS_DB",
        ge=0,
        le=15
    )
    password: Optional[str] = Field(
        default=None,
        description="Redis password (required for production)",
        validation_alias="REDIS_PASSWORD"
    )
    username: Optional[str] = Field(
        default=None,
        description="Redis username (Redis 6.0+)",
        validation_alias="REDIS_USERNAME"
    )
    
    # Security settings
    require_auth: bool = Field(
        default=True,
        description="Require authentication for Redis connections",
        validation_alias="REDIS_REQUIRE_AUTH"
    )
    use_ssl: bool = Field(
        default=False,
        description="Use SSL/TLS for Redis connections",
        validation_alias="REDIS_USE_SSL"
    )
    ssl_cert_path: Optional[str] = Field(
        default=None,
        description="Path to SSL certificate file",
        validation_alias="REDIS_SSL_CERT_PATH"
    )
    ssl_key_path: Optional[str] = Field(
        default=None,
        description="Path to SSL private key file",
        validation_alias="REDIS_SSL_KEY_PATH"
    )
    ssl_ca_path: Optional[str] = Field(
        default=None,
        description="Path to SSL CA certificate file",
        validation_alias="REDIS_SSL_CA_PATH"
    )
    ssl_verify_mode: str = Field(
        default="required",
        description="SSL certificate verification mode (none, optional, required)",
        validation_alias="REDIS_SSL_VERIFY_MODE"
    )
    
    # Connection pool settings
    max_connections: int = Field(
        default=20,
        description="Maximum Redis connections",
        validation_alias="REDIS_MAX_CONNECTIONS",
        ge=1,
        le=100
    )
    connection_timeout: int = Field(
        default=5,
        description="Connection timeout in seconds",
        validation_alias="REDIS_CONNECT_TIMEOUT",
        ge=1,
        le=30
    )
    socket_timeout: int = Field(
        default=5,
        description="Socket timeout in seconds",
        validation_alias="REDIS_SOCKET_TIMEOUT",
        ge=1,
        le=30
    )
    
    # Memory and performance settings
    max_memory: str = Field(
        default="256mb",
        description="Maximum memory usage",
        validation_alias="REDIS_MAX_MEMORY"
    )
    max_memory_policy: str = Field(
        default="allkeys-lru",
        description="Memory eviction policy",
        validation_alias="REDIS_MAX_MEMORY_POLICY"
    )
    
    # Cache settings
    default_ttl: int = Field(
        default=3600,
        description="Default TTL in seconds (1 hour)",
        validation_alias="REDIS_DEFAULT_TTL",
        ge=60,
        le=86400
    )
    
    # High availability settings
    sentinel_enabled: bool = Field(
        default=False,
        description="Enable Redis Sentinel",
        validation_alias="REDIS_SENTINEL_ENABLED"
    )
    sentinel_hosts: List[str] = Field(
        default_factory=list,
        description="Redis Sentinel hosts",
        validation_alias="REDIS_SENTINEL_HOSTS"
    )
    sentinel_service_name: str = Field(
        default="mymaster",
        description="Redis Sentinel service name",
        validation_alias="REDIS_SENTINEL_SERVICE_NAME"
    )
    
    @field_validator("password")
    @classmethod
    def validate_password_security(cls, v: Optional[str], info) -> Optional[str]:
        """Validate password meets security requirements."""
        if v and len(v) < 12:
            logger.warning("Redis password should be at least 12 characters for security")
        return v
    
    @field_validator("ssl_verify_mode")
    @classmethod
    def validate_ssl_verify_mode(cls, v: str) -> str:
        """Validate SSL verification mode."""
        valid_modes = {"none", "optional", "required"}
        if v not in valid_modes:
            raise ValueError(f"SSL verify mode must be one of: {valid_modes}")
        return v
    
    @model_validator(mode='after')
    def validate_security_configuration(self) -> 'RedisConfig':
        """Validate security configuration consistency."""
        
        # Check authentication requirements
        if self.require_auth and not self.password:
            if self.host != "localhost":
                raise ValueError("Redis password required when authentication is enabled for non-localhost connections")
            else:
                logger.warning("Redis authentication disabled for localhost - acceptable for development only")
        
        # Check SSL configuration
        if self.use_ssl:
            if self.ssl_verify_mode == "required" and not self.ssl_ca_path:
                logger.warning("SSL verification set to 'required' but no CA certificate path provided")
        
        return self
    
    def validate_production_security(self) -> List[str]:
        """Validate configuration meets production security requirements."""
        issues = []
        
        # Authentication checks
        if not self.password:
            issues.append("Redis password not configured - required for production")
        elif len(self.password) < 16:
            issues.append("Redis password should be at least 16 characters for production")
        
        # SSL/TLS checks
        if not self.use_ssl and self.host != "localhost":
            issues.append("SSL/TLS not enabled for non-localhost Redis connection")
        
        # Host security checks
        if self.host == "localhost" and not self.password:
            issues.append("Localhost Redis without password - not suitable for production")
        
        return issues
    
    @property
    def redis_url(self) -> str:
        """Generate Redis connection URL with security considerations."""
        # Use rediss:// scheme for SSL connections
        scheme = "rediss" if self.use_ssl else "redis"
        
        auth_part = ""
        if self.username and self.password:
            auth_part = f"{self.username}:{self.password}@"
        elif self.password:
            auth_part = f":{self.password}@"
        
        return f"{scheme}://{auth_part}{self.host}:{self.port}/{self.database}"
    
    @property
    def secure_redis_url(self) -> str:
        """Generate Redis connection URL with authentication masked for logging."""
        scheme = "rediss" if self.use_ssl else "redis"
        auth_part = "***:***@" if self.password else ""
        return f"{scheme}://{auth_part}{self.host}:{self.port}/{self.database}"

class MLConfig(BaseModel):
    """Machine Learning model and inference configuration."""
    
    # Model paths and storage
    model_storage_path: Path = Field(
        default=Path("./models"),
        description="Base path for model storage",
        validation_alias="ML_MODEL_STORAGE_PATH"
    )
    model_cache_size: int = Field(
        default=5,
        description="Number of models to cache in memory",
        validation_alias="ML_MODEL_CACHE_SIZE",
        ge=1,
        le=20
    )
    
    # Inference settings
    inference_timeout: int = Field(
        default=30,
        description="Model inference timeout in seconds",
        validation_alias="ML_INFERENCE_TIMEOUT",
        ge=1,
        le=300
    )
    batch_size: int = Field(
        default=32,
        description="Default batch size for inference",
        validation_alias="ML_BATCH_SIZE",
        ge=1,
        le=1000
    )
    max_sequence_length: int = Field(
        default=2048,
        description="Maximum input sequence length",
        validation_alias="ML_MAX_SEQUENCE_LENGTH",
        ge=128,
        le=8192
    )
    
    # Resource limits
    gpu_memory_fraction: float = Field(
        default=0.8,
        description="Fraction of GPU memory to use",
        validation_alias="ML_GPU_MEMORY_FRACTION",
        ge=0.1,
        le=1.0
    )
    cpu_threads: int = Field(
        default=4,
        description="Number of CPU threads for inference",
        validation_alias="ML_CPU_THREADS",
        ge=1,
        le=32
    )
    
    # MLflow and experiment tracking
    mlflow_tracking_uri: Optional[str] = Field(
        default=None,
        description="MLflow tracking server URI",
        validation_alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="prompt-improvement",
        description="MLflow experiment name",
        validation_alias="MLFLOW_EXPERIMENT_NAME"
    )
    
    # Model serving
    model_serving_enabled: bool = Field(
        default=True,
        description="Enable model serving endpoints",
        validation_alias="ML_MODEL_SERVING_ENABLED"
    )
    model_warmup_enabled: bool = Field(
        default=True,
        description="Enable model warmup on startup",
        validation_alias="ML_MODEL_WARMUP_ENABLED"
    )



class APIConfig(BaseModel):
    """API server and HTTP configuration."""
    
    # Server settings
    host: str = Field(
        default="0.0.0.0",
        description="API server host",
        validation_alias="API_HOST"
    )
    port: int = Field(
        default=8000,
        description="API server port",
        validation_alias="API_PORT",
        ge=1,
        le=65535
    )
    workers: int = Field(
        default=1,
        description="Number of worker processes",
        validation_alias="API_WORKERS",
        ge=1,
        le=32
    )
    
    # Request/Response limits
    max_request_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum request size in bytes",
        validation_alias="API_MAX_REQUEST_SIZE",
        ge=1024,
        le=100 * 1024 * 1024
    )
    request_timeout: int = Field(
        default=30,
        description="Request timeout in seconds",
        validation_alias="API_REQUEST_TIMEOUT",
        ge=1,
        le=300
    )
    
    # Rate limiting
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting",
        validation_alias="API_RATE_LIMIT_ENABLED"
    )
    rate_limit_requests_per_minute: int = Field(
        default=60,
        description="Requests per minute per client",
        validation_alias="API_RATE_LIMIT_RPM",
        ge=1,
        le=10000
    )
    rate_limit_burst: int = Field(
        default=10,
        description="Rate limit burst capacity",
        validation_alias="API_RATE_LIMIT_BURST",
        ge=1,
        le=100
    )
    
    # CORS settings
    cors_enabled: bool = Field(
        default=True,
        description="Enable CORS",
        validation_alias="API_CORS_ENABLED"
    )
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:8080"],
        description="Allowed CORS origins",
        validation_alias="API_CORS_ORIGINS"
    )
    cors_methods: List[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed CORS methods",
        validation_alias="API_CORS_METHODS"
    )
    
    # Security settings
    enable_https: bool = Field(
        default=False,
        description="Enable HTTPS",
        validation_alias="API_ENABLE_HTTPS"
    )
    ssl_cert_path: Optional[Path] = Field(
        default=None,
        description="SSL certificate path",
        validation_alias="API_SSL_CERT_PATH"
    )
    ssl_key_path: Optional[Path] = Field(
        default=None,
        description="SSL private key path",
        validation_alias="API_SSL_KEY_PATH"
    )
    

class MCPConfig(BaseModel):
    """MCP Server specific configuration for batch processing and session management."""
    
    # Batch processor settings
    batch_size: int = Field(
        default=10,
        description="Number of records to process in each batch",
        validation_alias="MCP_BATCH_SIZE",
        ge=1,
        le=100
    )
    batch_timeout: int = Field(
        default=30,
        description="Timeout between periodic batch processing triggers in seconds",
        validation_alias="MCP_BATCH_TIMEOUT",
        ge=1,
        le=300
    )
    max_attempts: int = Field(
        default=3,
        description="Maximum retry attempts per record",
        validation_alias="MCP_MAX_ATTEMPTS",
        ge=1,
        le=10
    )
    concurrency: int = Field(
        default=3,
        description="Number of concurrent processing workers",
        validation_alias="MCP_CONCURRENCY",
        ge=1,
        le=20
    )
    
    # Session store settings
    session_maxsize: int = Field(
        default=1000,
        description="Maximum number of concurrent sessions",
        validation_alias="MCP_SESSION_MAXSIZE",
        ge=100,
        le=10000
    )
    session_ttl: int = Field(
        default=3600,
        description="Session TTL in seconds (1 hour)",
        validation_alias="MCP_SESSION_TTL",
        ge=300,
        le=86400
    )
    session_cleanup_interval: int = Field(
        default=300,
        description="Session cleanup interval in seconds (5 minutes)",
        validation_alias="MCP_SESSION_CLEANUP_INTERVAL",
        ge=60,
        le=3600
    )
    
    # Additional MCP server settings
    enable_circuit_breaker: bool = Field(
        default=True,
        description="Enable circuit breaker for batch processing",
        validation_alias="MCP_ENABLE_CIRCUIT_BREAKER"
    )
    enable_dead_letter_queue: bool = Field(
        default=True,
        description="Enable dead letter queue for failed records",
        validation_alias="MCP_ENABLE_DEAD_LETTER_QUEUE"
    )
    enable_opentelemetry: bool = Field(
        default=True,
        description="Enable OpenTelemetry monitoring",
        validation_alias="MCP_ENABLE_OPENTELEMETRY"
    )
    
    # Performance settings
    response_timeout_ms: int = Field(
        default=30000,
        description="Timeout per batch in milliseconds",
        validation_alias="MCP_RESPONSE_TIMEOUT_MS",
        ge=1000,
        le=300000
    )
    
    @model_validator(mode='after')
    def validate_mcp_settings(self) -> 'MCPConfig':
        """Validate MCP configuration consistency."""
        if self.session_cleanup_interval >= self.session_ttl:
            raise ValueError("session_cleanup_interval must be less than session_ttl")
        return self

class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration for resilience patterns."""
    
    # Circuit breaker thresholds
    failure_threshold: int = Field(
        default=5,
        description="Number of failures before opening circuit",
        validation_alias="CIRCUIT_BREAKER_FAILURE_THRESHOLD",
        ge=1,
        le=20
    )
    recovery_timeout: int = Field(
        default=60,
        description="Seconds before trying half-open state",
        validation_alias="CIRCUIT_BREAKER_RECOVERY_TIMEOUT",
        ge=10,
        le=600
    )
    half_open_max_calls: int = Field(
        default=3,
        description="Test calls in half-open state",
        validation_alias="CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS",
        ge=1,
        le=10
    )
    reset_timeout: int = Field(
        default=120,
        description="Full reset after success streak in seconds",
        validation_alias="CIRCUIT_BREAKER_RESET_TIMEOUT",
        ge=30,
        le=3600
    )
    
    # SLA-based thresholds
    response_time_threshold_ms: float = Field(
        default=1000.0,
        description="Consider slow response as failure (ms)",
        validation_alias="CIRCUIT_BREAKER_RESPONSE_TIME_THRESHOLD_MS",
        ge=100.0,
        le=10000.0
    )
    success_rate_threshold: float = Field(
        default=0.95,
        description="Minimum success rate to stay closed",
        validation_alias="CIRCUIT_BREAKER_SUCCESS_RATE_THRESHOLD",
        ge=0.5,
        le=1.0
    )

class SLAMonitoringConfig(BaseModel):
    """Service Level Agreement monitoring configuration."""
    
    # Response time SLA targets
    response_time_p50_ms: float = Field(
        default=100.0,
        description="50th percentile response time target (ms)",
        validation_alias="SLA_RESPONSE_TIME_P50_MS",
        ge=10.0,
        le=5000.0
    )
    response_time_p95_ms: float = Field(
        default=500.0,
        description="95th percentile response time target (ms)",
        validation_alias="SLA_RESPONSE_TIME_P95_MS",
        ge=50.0,
        le=10000.0
    )
    response_time_p99_ms: float = Field(
        default=1000.0,
        description="99th percentile response time target (ms)",
        validation_alias="SLA_RESPONSE_TIME_P99_MS",
        ge=100.0,
        le=15000.0
    )
    
    # Availability SLA targets
    availability_target: float = Field(
        default=0.999,
        description="Uptime availability target (99.9%)",
        validation_alias="SLA_AVAILABILITY_TARGET",
        ge=0.90,
        le=1.0
    )
    error_rate_threshold: float = Field(
        default=0.01,
        description="Maximum error rate threshold (1%)",
        validation_alias="SLA_ERROR_RATE_THRESHOLD",
        ge=0.001,
        le=0.1
    )
    
    # Measurement configuration
    measurement_window_seconds: int = Field(
        default=300,
        description="SLA measurement window in seconds (5 min)",
        validation_alias="SLA_MEASUREMENT_WINDOW_SECONDS",
        ge=60,
        le=3600
    )
    warning_threshold: float = Field(
        default=0.9,
        description="Warning threshold as fraction of target",
        validation_alias="SLA_WARNING_THRESHOLD",
        ge=0.5,
        le=1.0
    )
    critical_threshold: float = Field(
        default=1.0,
        description="Critical threshold as fraction of target",
        validation_alias="SLA_CRITICAL_THRESHOLD",
        ge=0.8,
        le=2.0
    )
    
    # Alert configuration
    alert_cooldown_seconds: int = Field(
        default=300,
        description="Alert cooldown period in seconds",
        validation_alias="SLA_ALERT_COOLDOWN_SECONDS",
        ge=60,
        le=3600
    )

class DirectoryPathsConfig(BaseModel):
    """Configuration for system directory paths."""
    
    # Temporary directories
    temp_base_dir: Path = Field(
        default=Path("/tmp"),
        description="Base temporary directory",
        validation_alias="TEMP_BASE_DIR"
    )
    pid_dir: Path = Field(
        default=Path("/tmp/prompt_improver_pids"),
        description="PID files directory",
        validation_alias="PID_DIR"
    )
    prometheus_multiproc_dir: Path = Field(
        default=Path("/tmp/prometheus_multiproc"),
        description="Prometheus multiprocess directory",
        validation_alias="PROMETHEUS_MULTIPROC_DIR"
    )
    
    # Cache directories
    cache_dir: Path = Field(
        default=Path("./cache"),
        description="Application cache directory",
        validation_alias="CACHE_DIR"
    )
    log_dir: Path = Field(
        default=Path("./logs"),
        description="Log files directory",
        validation_alias="LOG_DIR"
    )
    
    # Security configuration
    secure_temp_permissions: int = Field(
        default=0o755,
        description="Permissions for temporary directories",
        validation_alias="SECURE_TEMP_PERMISSIONS"
    )

class HealthCheckConfig(BaseModel):
    """Health check and monitoring configuration."""
    
    # Health check intervals
    health_check_interval: int = Field(
        default=30,
        description="Health check interval in seconds",
        validation_alias="HEALTH_CHECK_INTERVAL",
        ge=5,
        le=300
    )
    startup_timeout: int = Field(
        default=60,
        description="Startup timeout in seconds",
        validation_alias="STARTUP_TIMEOUT",
        ge=10,
        le=600
    )
    shutdown_timeout: int = Field(
        default=30,
        description="Graceful shutdown timeout in seconds",
        validation_alias="SHUTDOWN_TIMEOUT",
        ge=5,
        le=300
    )
    
    # Component health thresholds
    database_health_timeout: int = Field(
        default=5,
        description="Database health check timeout",
        validation_alias="DB_HEALTH_TIMEOUT",
        ge=1,
        le=30
    )
    redis_health_timeout: int = Field(
        default=3,
        description="Redis health check timeout",
        validation_alias="REDIS_HEALTH_TIMEOUT",
        ge=1,
        le=30
    )
    ml_model_health_timeout: int = Field(
        default=10,
        description="ML model health check timeout",
        validation_alias="ML_HEALTH_TIMEOUT",
        ge=1,
        le=60
    )
    
    # Performance thresholds
    response_time_threshold_ms: int = Field(
        default=1000,
        description="Response time threshold in milliseconds",
        validation_alias="RESPONSE_TIME_THRESHOLD_MS",
        ge=100,
        le=10000
    )
    memory_usage_threshold_percent: float = Field(
        default=85.0,
        description="Memory usage threshold percentage",
        validation_alias="MEMORY_USAGE_THRESHOLD",
        ge=50.0,
        le=95.0
    )
    cpu_usage_threshold_percent: float = Field(
        default=80.0,
        description="CPU usage threshold percentage",
        validation_alias="CPU_USAGE_THRESHOLD",
        ge=50.0,
        le=95.0
    )
    
    # Monitoring settings
    metrics_enabled: bool = Field(
        default=True,
        description="Enable metrics collection",
        validation_alias="METRICS_ENABLED"
    )
    metrics_port: int = Field(
        default=9090,
        description="Metrics server port",
        validation_alias="METRICS_PORT",
        ge=1024,
        le=65535
    )
    alerting_enabled: bool = Field(
        default=False,
        description="Enable alerting",
        validation_alias="ALERTING_ENABLED"
    )

# =============================================================================
# Main Configuration Class
# =============================================================================

class AppConfig(BaseSettings):
    """
    Main application configuration using Pydantic BaseSettings.
    
    This configuration system follows 2025 best practices:
    - Environment-specific loading with .env support
    - Type validation with comprehensive defaults
    - Nested configuration models
    - Hot-reload capability
    - Production-ready validation
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
        env_nested_delimiter="__"  # Enable nested parsing: DATABASE__HOST -> database.host
    )
    
    # Nested Configuration Models (2025 Best Practice - Post-initialization)
    # These are initialized after the main config to avoid environment parsing conflicts
    
    # Environment Settings (flattened for easier env var access)
    env_environment: str = Field(
        default="development",
        description="Application environment",
        validation_alias=AliasChoices("ENVIRONMENT", "APP_ENV", "ENV")
    )
    env_debug: bool = Field(default=True, validation_alias="DEBUG")
    env_log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    env_app_name: str = Field(default="prompt-improver", validation_alias="APP_NAME")
    env_version: str = Field(default="1.0.0", validation_alias="APP_VERSION")
    env_secret_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("SECRET_KEY", "APP_SECRET_KEY")
    )
    
    # Database Settings (flattened)
    db_host: str = Field(default="localhost", validation_alias="POSTGRES_HOST")
    db_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    db_database: str = Field(default="apes_production", validation_alias=AliasChoices("POSTGRES_DATABASE", "DB_NAME"))
    db_username: str = Field(default="apes_user", validation_alias=AliasChoices("POSTGRES_USERNAME", "DB_USER"))
    db_password: str = Field(default="default_dev_password", validation_alias=AliasChoices("POSTGRES_PASSWORD", "DB_PASSWORD"))
    db_pool_min_size: int = Field(default=4, validation_alias="DB_POOL_MIN_SIZE", ge=1, le=50)
    db_pool_max_size: int = Field(default=16, validation_alias="DB_POOL_MAX_SIZE", ge=1, le=100)
    db_pool_timeout: int = Field(default=10, validation_alias="DB_POOL_TIMEOUT", ge=1, le=60)
    db_pool_max_lifetime: int = Field(default=1800, validation_alias="DB_POOL_MAX_LIFETIME", ge=300, le=7200)
    db_pool_max_idle: int = Field(default=300, validation_alias="DB_POOL_MAX_IDLE", ge=60, le=1800)
    db_statement_timeout: int = Field(default=30, validation_alias="DB_STATEMENT_TIMEOUT", ge=1, le=300)
    db_echo_sql: bool = Field(default=False, validation_alias="DB_ECHO_SQL")
    db_target_query_time_ms: int = Field(default=50, validation_alias="DB_TARGET_QUERY_TIME_MS", ge=1, le=5000)
    db_target_cache_hit_ratio: float = Field(default=0.90, validation_alias="DB_TARGET_CACHE_HIT_RATIO", ge=0.0, le=1.0)
    
    # Connection Pool Optimizer Settings (flattened)
    db_pool_target_utilization: float = Field(default=0.7, validation_alias="DB_POOL_TARGET_UTILIZATION", ge=0.1, le=0.95)
    db_pool_high_utilization_threshold: float = Field(default=0.9, validation_alias="DB_POOL_HIGH_UTILIZATION_THRESHOLD", ge=0.5, le=0.99)
    db_pool_low_utilization_threshold: float = Field(default=0.3, validation_alias="DB_POOL_LOW_UTILIZATION_THRESHOLD", ge=0.1, le=0.8)
    db_pool_stress_threshold: float = Field(default=0.8, validation_alias="DB_POOL_STRESS_THRESHOLD", ge=0.5, le=0.95)
    db_pool_underutilized_threshold: float = Field(default=0.4, validation_alias="DB_POOL_UNDERUTILIZED_THRESHOLD", ge=0.1, le=0.7)
    db_pool_wait_threshold: int = Field(default=5, validation_alias="DB_POOL_WAIT_THRESHOLD", ge=1, le=50)
    db_pool_increase_step: int = Field(default=5, validation_alias="DB_POOL_INCREASE_STEP", ge=1, le=20)
    db_pool_decrease_step: int = Field(default=3, validation_alias="DB_POOL_DECREASE_STEP", ge=1, le=10)
    db_pool_scale_up_factor: float = Field(default=1.2, validation_alias="DB_POOL_SCALE_UP_FACTOR", ge=1.1, le=2.0)
    db_pool_scale_down_factor: float = Field(default=0.9, validation_alias="DB_POOL_SCALE_DOWN_FACTOR", ge=0.5, le=0.95)
    db_pool_registry_size: int = Field(default=1000, validation_alias="DB_POOL_REGISTRY_SIZE", ge=100, le=5000)
    db_pool_max_connection_age_hours: int = Field(default=24, validation_alias="DB_POOL_MAX_CONNECTION_AGE_HOURS", ge=1, le=168)
    
    # Redis Settings (flattened)
    redis_host: str = Field(default="localhost", validation_alias="REDIS_HOST")
    redis_port: int = Field(default=6379, validation_alias="REDIS_PORT", ge=1, le=65535)
    redis_database: int = Field(default=0, validation_alias="REDIS_DB", ge=0, le=15)
    redis_password: Optional[str] = Field(default=None, validation_alias="REDIS_PASSWORD")
    redis_username: Optional[str] = Field(default=None, validation_alias="REDIS_USERNAME")
    redis_require_auth: bool = Field(default=True, validation_alias="REDIS_REQUIRE_AUTH")
    redis_use_ssl: bool = Field(default=False, validation_alias="REDIS_USE_SSL")
    redis_ssl_cert_path: Optional[str] = Field(default=None, validation_alias="REDIS_SSL_CERT_PATH")
    redis_ssl_key_path: Optional[str] = Field(default=None, validation_alias="REDIS_SSL_KEY_PATH")
    redis_ssl_ca_path: Optional[str] = Field(default=None, validation_alias="REDIS_SSL_CA_PATH")
    redis_ssl_verify_mode: str = Field(default="required", validation_alias="REDIS_SSL_VERIFY_MODE")
    redis_max_connections: int = Field(default=20, validation_alias="REDIS_MAX_CONNECTIONS", ge=1, le=100)
    redis_connection_timeout: int = Field(default=5, validation_alias="REDIS_CONNECT_TIMEOUT", ge=1, le=30)
    redis_socket_timeout: int = Field(default=5, validation_alias="REDIS_SOCKET_TIMEOUT", ge=1, le=30)
    redis_max_memory: str = Field(default="256mb", validation_alias="REDIS_MAX_MEMORY")
    redis_max_memory_policy: str = Field(default="allkeys-lru", validation_alias="REDIS_MAX_MEMORY_POLICY")
    redis_default_ttl: int = Field(default=3600, validation_alias="REDIS_DEFAULT_TTL", ge=60, le=86400)
    redis_sentinel_enabled: bool = Field(default=False, validation_alias="REDIS_SENTINEL_ENABLED")
    redis_sentinel_hosts: List[str] = Field(default_factory=list, validation_alias="REDIS_SENTINEL_HOSTS")
    redis_sentinel_service_name: str = Field(default="mymaster", validation_alias="REDIS_SENTINEL_SERVICE_NAME")
    
    # ML Settings (flattened)
    ml_model_storage_path: Path = Field(default=Path("./models"), validation_alias="ML_MODEL_STORAGE_PATH")
    ml_model_cache_size: int = Field(default=5, validation_alias="ML_MODEL_CACHE_SIZE", ge=1, le=20)
    ml_inference_timeout: int = Field(default=30, validation_alias="ML_INFERENCE_TIMEOUT", ge=1, le=300)
    ml_batch_size: int = Field(default=32, validation_alias="ML_BATCH_SIZE", ge=1, le=1000)
    ml_max_sequence_length: int = Field(default=2048, validation_alias="ML_MAX_SEQUENCE_LENGTH", ge=128, le=8192)
    ml_gpu_memory_fraction: float = Field(default=0.8, validation_alias="ML_GPU_MEMORY_FRACTION", ge=0.1, le=1.0)
    ml_cpu_threads: int = Field(default=4, validation_alias="ML_CPU_THREADS", ge=1, le=32)
    ml_mlflow_tracking_uri: Optional[str] = Field(default=None, validation_alias="MLFLOW_TRACKING_URI")
    ml_mlflow_experiment_name: str = Field(default="prompt-improvement", validation_alias="MLFLOW_EXPERIMENT_NAME")
    ml_model_serving_enabled: bool = Field(default=True, validation_alias="ML_MODEL_SERVING_ENABLED")
    ml_model_warmup_enabled: bool = Field(default=True, validation_alias="ML_MODEL_WARMUP_ENABLED")


    # API Settings (flattened)
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT", ge=1, le=65535)
    api_workers: int = Field(default=1, validation_alias="API_WORKERS", ge=1, le=32)
    api_max_request_size: int = Field(default=10 * 1024 * 1024, validation_alias="API_MAX_REQUEST_SIZE", ge=1024, le=100 * 1024 * 1024)
    api_request_timeout: int = Field(default=30, validation_alias="API_REQUEST_TIMEOUT", ge=1, le=300)
    api_rate_limit_enabled: bool = Field(default=True, validation_alias="API_RATE_LIMIT_ENABLED")
    api_rate_limit_requests_per_minute: int = Field(default=60, validation_alias="API_RATE_LIMIT_RPM", ge=1, le=10000)
    api_rate_limit_burst: int = Field(default=10, validation_alias="API_RATE_LIMIT_BURST", ge=1, le=100)
    api_cors_enabled: bool = Field(default=True, validation_alias="API_CORS_ENABLED")
    api_cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:8080"], validation_alias="API_CORS_ORIGINS")
    api_cors_methods: List[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"], validation_alias="API_CORS_METHODS")
    api_enable_https: bool = Field(default=False, validation_alias="API_ENABLE_HTTPS")
    api_ssl_cert_path: Optional[Path] = Field(default=None, validation_alias="API_SSL_CERT_PATH")
    api_ssl_key_path: Optional[Path] = Field(default=None, validation_alias="API_SSL_KEY_PATH")
    
    # MCP Settings (flattened)
    mcp_batch_size: int = Field(default=10, validation_alias="MCP_BATCH_SIZE", ge=1, le=100)
    mcp_batch_timeout: int = Field(default=30, validation_alias="MCP_BATCH_TIMEOUT", ge=1, le=300)
    mcp_max_attempts: int = Field(default=3, validation_alias="MCP_MAX_ATTEMPTS", ge=1, le=10)
    mcp_concurrency: int = Field(default=3, validation_alias="MCP_CONCURRENCY", ge=1, le=20)
    mcp_session_maxsize: int = Field(default=1000, validation_alias="MCP_SESSION_MAXSIZE", ge=100, le=10000)
    mcp_session_ttl: int = Field(default=3600, validation_alias="MCP_SESSION_TTL", ge=300, le=86400)
    mcp_session_cleanup_interval: int = Field(default=300, validation_alias="MCP_SESSION_CLEANUP_INTERVAL", ge=60, le=3600)
    mcp_enable_circuit_breaker: bool = Field(default=True, validation_alias="MCP_ENABLE_CIRCUIT_BREAKER")
    mcp_enable_dead_letter_queue: bool = Field(default=True, validation_alias="MCP_ENABLE_DEAD_LETTER_QUEUE")
    mcp_enable_opentelemetry: bool = Field(default=True, validation_alias="MCP_ENABLE_OPENTELEMETRY")
    mcp_response_timeout_ms: int = Field(default=30000, validation_alias="MCP_RESPONSE_TIMEOUT_MS", ge=1000, le=300000)

    # Circuit Breaker Settings (flattened)
    circuit_breaker_failure_threshold: int = Field(default=5, validation_alias="CIRCUIT_BREAKER_FAILURE_THRESHOLD", ge=1, le=20)
    circuit_breaker_recovery_timeout: int = Field(default=60, validation_alias="CIRCUIT_BREAKER_RECOVERY_TIMEOUT", ge=10, le=600)
    circuit_breaker_half_open_max_calls: int = Field(default=3, validation_alias="CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS", ge=1, le=10)
    circuit_breaker_reset_timeout: int = Field(default=120, validation_alias="CIRCUIT_BREAKER_RESET_TIMEOUT", ge=30, le=3600)
    circuit_breaker_response_time_threshold_ms: float = Field(default=1000.0, validation_alias="CIRCUIT_BREAKER_RESPONSE_TIME_THRESHOLD_MS", ge=100.0, le=10000.0)
    circuit_breaker_success_rate_threshold: float = Field(default=0.95, validation_alias="CIRCUIT_BREAKER_SUCCESS_RATE_THRESHOLD", ge=0.5, le=1.0)
    
    # SLA Monitoring Settings (flattened)
    sla_response_time_p50_ms: float = Field(default=100.0, validation_alias="SLA_RESPONSE_TIME_P50_MS", ge=10.0, le=5000.0)
    sla_response_time_p95_ms: float = Field(default=500.0, validation_alias="SLA_RESPONSE_TIME_P95_MS", ge=50.0, le=10000.0)
    sla_response_time_p99_ms: float = Field(default=1000.0, validation_alias="SLA_RESPONSE_TIME_P99_MS", ge=100.0, le=15000.0)
    sla_availability_target: float = Field(default=0.999, validation_alias="SLA_AVAILABILITY_TARGET", ge=0.90, le=1.0)
    sla_error_rate_threshold: float = Field(default=0.01, validation_alias="SLA_ERROR_RATE_THRESHOLD", ge=0.001, le=0.1)
    sla_measurement_window_seconds: int = Field(default=300, validation_alias="SLA_MEASUREMENT_WINDOW_SECONDS", ge=60, le=3600)
    sla_warning_threshold: float = Field(default=0.9, validation_alias="SLA_WARNING_THRESHOLD", ge=0.5, le=1.0)
    sla_critical_threshold: float = Field(default=1.0, validation_alias="SLA_CRITICAL_THRESHOLD", ge=0.8, le=2.0)
    sla_alert_cooldown_seconds: int = Field(default=300, validation_alias="SLA_ALERT_COOLDOWN_SECONDS", ge=60, le=3600)
    
    # Directory Paths Settings (flattened)
    temp_base_dir: Path = Field(default=Path("/tmp"), validation_alias="TEMP_BASE_DIR")
    pid_dir: Path = Field(default=Path("/tmp/prompt_improver_pids"), validation_alias="PID_DIR")
    prometheus_multiproc_dir: Path = Field(default=Path("/tmp/prometheus_multiproc"), validation_alias="PROMETHEUS_MULTIPROC_DIR")
    cache_dir: Path = Field(default=Path("./cache"), validation_alias="CACHE_DIR")
    log_dir: Path = Field(default=Path("./logs"), validation_alias="LOG_DIR")
    secure_temp_permissions: int = Field(default=0o755, validation_alias="SECURE_TEMP_PERMISSIONS")

    # Health Check Settings (flattened)
    health_check_interval: int = Field(default=30, validation_alias="HEALTH_CHECK_INTERVAL", ge=5, le=300)
    health_startup_timeout: int = Field(default=60, validation_alias="STARTUP_TIMEOUT", ge=10, le=600)
    health_shutdown_timeout: int = Field(default=30, validation_alias="SHUTDOWN_TIMEOUT", ge=5, le=300)
    health_database_timeout: int = Field(default=5, validation_alias="DB_HEALTH_TIMEOUT", ge=1, le=30)
    health_redis_timeout: int = Field(default=3, validation_alias="REDIS_HEALTH_TIMEOUT", ge=1, le=30)
    health_ml_model_timeout: int = Field(default=10, validation_alias="ML_HEALTH_TIMEOUT", ge=1, le=60)
    health_response_time_threshold_ms: int = Field(default=1000, validation_alias="RESPONSE_TIME_THRESHOLD_MS", ge=100, le=10000)
    health_memory_usage_threshold_percent: float = Field(default=85.0, validation_alias="MEMORY_USAGE_THRESHOLD", ge=50.0, le=95.0)
    health_cpu_usage_threshold_percent: float = Field(default=80.0, validation_alias="CPU_USAGE_THRESHOLD", ge=50.0, le=95.0)
    health_metrics_enabled: bool = Field(default=True, validation_alias="METRICS_ENABLED")
    health_metrics_port: int = Field(default=9090, validation_alias="METRICS_PORT", ge=1024, le=65535)
    health_alerting_enabled: bool = Field(default=False, validation_alias="ALERTING_ENABLED")
    
    # Global settings
    config_file_path: Optional[Path] = Field(
        default=None,
        description="Path to configuration file",
        validation_alias="CONFIG_FILE_PATH"
    )
    hot_reload_enabled: bool = Field(
        default=True,
        description="Enable configuration hot-reload",
        validation_alias="CONFIG_HOT_RELOAD"
    )
    validation_strict: bool = Field(
        default=True,
        description="Enable strict validation",
        validation_alias="CONFIG_VALIDATION_STRICT"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
        use_enum_values=True,
        secrets_dir=os.environ.get("SECRETS_DIR", "/run/secrets")
    )
    
    # Environment field validators
    @field_validator("env_environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        valid_envs = {"development", "staging", "production", "test"}
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v.lower()
    
    @field_validator("env_log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @model_validator(mode='after')
    def validate_configuration(self) -> 'AppConfig':
        """Perform cross-component validation."""
        
        # Validate production requirements
        if self.is_production:
            if not self.env_secret_key:
                raise ValueError("SECRET_KEY is required in production")
            if self.env_debug:
                logger.warning("Debug mode enabled in production - consider disabling")
        
        # Validate SSL configuration
        if self.api_enable_https:
            if not self.api_ssl_cert_path or not self.api_ssl_key_path:
                raise ValueError("SSL certificate and key paths required when HTTPS is enabled")
        
        # Validate Redis sentinel configuration
        if self.redis_sentinel_enabled and not self.redis_sentinel_hosts:
            raise ValueError("Sentinel hosts required when Redis Sentinel is enabled")
        
        # Validate pool settings
        if self.db_pool_max_size < self.db_pool_min_size:
            raise ValueError("db_pool_max_size must be >= db_pool_min_size")
        
        # Validate MCP settings
        if self.mcp_session_cleanup_interval >= self.mcp_session_ttl:
            raise ValueError("mcp_session_cleanup_interval must be less than mcp_session_ttl")
        
        return self
    
    # Convenience Properties
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env_environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.env_environment == "development"
    
    @property
    def environment(self) -> EnvironmentConfig:
        """Get environment configuration as structured object."""
        return EnvironmentConfig(
            environment=self.env_environment,
            debug=self.env_debug,
            log_level=self.env_log_level,
            app_name=self.env_app_name,
            version=self.env_version,
            secret_key=self.env_secret_key
        )
    
    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration as structured object."""
        # Return standalone DatabaseConfig that loads from environment directly
        return DatabaseConfig()
    
    @property
    def redis(self) -> RedisConfig:
        """Get Redis configuration as structured object."""
        return RedisConfig(
            host=self.redis_host,
            port=self.redis_port,
            database=self.redis_database,
            password=self.redis_password,
            username=self.redis_username,
            require_auth=self.redis_require_auth,
            use_ssl=self.redis_use_ssl,
            ssl_cert_path=self.redis_ssl_cert_path,
            ssl_key_path=self.redis_ssl_key_path,
            ssl_ca_path=self.redis_ssl_ca_path,
            ssl_verify_mode=self.redis_ssl_verify_mode,
            max_connections=self.redis_max_connections,
            connection_timeout=self.redis_connection_timeout,
            socket_timeout=self.redis_socket_timeout,
            max_memory=self.redis_max_memory,
            max_memory_policy=self.redis_max_memory_policy,
            default_ttl=self.redis_default_ttl,
            sentinel_enabled=self.redis_sentinel_enabled,
            sentinel_hosts=self.redis_sentinel_hosts,
            sentinel_service_name=self.redis_sentinel_service_name
        )
    
    @property
    def ml(self) -> MLConfig:
        """Get ML configuration as structured object."""
        return MLConfig(
            model_storage_path=self.ml_model_storage_path,
            model_cache_size=self.ml_model_cache_size,
            inference_timeout=self.ml_inference_timeout,
            batch_size=self.ml_batch_size,
            max_sequence_length=self.ml_max_sequence_length,
            gpu_memory_fraction=self.ml_gpu_memory_fraction,
            cpu_threads=self.ml_cpu_threads,
            mlflow_tracking_uri=self.ml_mlflow_tracking_uri,
            mlflow_experiment_name=self.ml_mlflow_experiment_name,
            model_serving_enabled=self.ml_model_serving_enabled,
            model_warmup_enabled=self.ml_model_warmup_enabled
        )
    
    @property
    def api(self) -> APIConfig:
        """Get API configuration as structured object."""
        return APIConfig(
            host=self.api_host,
            port=self.api_port,
            workers=self.api_workers,
            max_request_size=self.api_max_request_size,
            request_timeout=self.api_request_timeout,
            rate_limit_enabled=self.api_rate_limit_enabled,
            rate_limit_requests_per_minute=self.api_rate_limit_requests_per_minute,
            rate_limit_burst=self.api_rate_limit_burst,
            cors_enabled=self.api_cors_enabled,
            cors_origins=self.api_cors_origins,
            cors_methods=self.api_cors_methods,
            enable_https=self.api_enable_https,
            ssl_cert_path=self.api_ssl_cert_path,
            ssl_key_path=self.api_ssl_key_path,
        )
    
    @property
    def mcp(self) -> MCPConfig:
        """Get MCP configuration as structured object."""
        return MCPConfig(
            batch_size=self.mcp_batch_size,
            batch_timeout=self.mcp_batch_timeout,
            max_attempts=self.mcp_max_attempts,
            concurrency=self.mcp_concurrency,
            session_maxsize=self.mcp_session_maxsize,
            session_ttl=self.mcp_session_ttl,
            session_cleanup_interval=self.mcp_session_cleanup_interval,
            enable_circuit_breaker=self.mcp_enable_circuit_breaker,
            enable_dead_letter_queue=self.mcp_enable_dead_letter_queue,
            enable_opentelemetry=self.mcp_enable_opentelemetry,
            response_timeout_ms=self.mcp_response_timeout_ms
        )
    
    @property
    def health(self) -> HealthCheckConfig:
        """Get health check configuration as structured object."""
        return HealthCheckConfig(
            health_check_interval=self.health_check_interval,
            startup_timeout=self.health_startup_timeout,
            shutdown_timeout=self.health_shutdown_timeout,
            database_health_timeout=self.health_database_timeout,
            redis_health_timeout=self.health_redis_timeout,
            ml_model_health_timeout=self.health_ml_model_timeout,
            response_time_threshold_ms=self.health_response_time_threshold_ms,
            memory_usage_threshold_percent=self.health_memory_usage_threshold_percent,
            cpu_usage_threshold_percent=self.health_cpu_usage_threshold_percent,
            metrics_enabled=self.health_metrics_enabled,
            metrics_port=self.health_metrics_port,
            alerting_enabled=self.health_alerting_enabled
        )
    
    @property
    def circuit_breaker(self) -> CircuitBreakerConfig:
        """Get circuit breaker configuration as structured object."""
        return CircuitBreakerConfig(
            failure_threshold=self.circuit_breaker_failure_threshold,
            recovery_timeout=self.circuit_breaker_recovery_timeout,
            half_open_max_calls=self.circuit_breaker_half_open_max_calls,
            reset_timeout=self.circuit_breaker_reset_timeout,
            response_time_threshold_ms=self.circuit_breaker_response_time_threshold_ms,
            success_rate_threshold=self.circuit_breaker_success_rate_threshold
        )
    
    @property
    def sla_monitoring(self) -> SLAMonitoringConfig:
        """Get SLA monitoring configuration as structured object."""
        return SLAMonitoringConfig(
            response_time_p50_ms=self.sla_response_time_p50_ms,
            response_time_p95_ms=self.sla_response_time_p95_ms,
            response_time_p99_ms=self.sla_response_time_p99_ms,
            availability_target=self.sla_availability_target,
            error_rate_threshold=self.sla_error_rate_threshold,
            measurement_window_seconds=self.sla_measurement_window_seconds,
            warning_threshold=self.sla_warning_threshold,
            critical_threshold=self.sla_critical_threshold,
            alert_cooldown_seconds=self.sla_alert_cooldown_seconds
        )
    
    @property
    def directory_paths(self) -> DirectoryPathsConfig:
        """Get directory paths configuration as structured object."""
        return DirectoryPathsConfig(
            temp_base_dir=self.temp_base_dir,
            pid_dir=self.pid_dir,
            prometheus_multiproc_dir=self.prometheus_multiproc_dir,
            cache_dir=self.cache_dir,
            log_dir=self.log_dir,
            secure_temp_permissions=self.secure_temp_permissions
        )


    def get_database_url(self, async_driver: bool = True) -> str:
        """Get database connection URL."""
        return self.database.database_url if async_driver else self.database.database_url_sync
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        return self.redis.redis_url
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        feature_flags = {
            "rate_limiting": self.api_rate_limit_enabled,
            "cors": self.api_cors_enabled,
            "metrics": self.health_metrics_enabled,
            "alerting": self.health_alerting_enabled,
            "model_serving": self.ml_model_serving_enabled,
            "hot_reload": self.hot_reload_enabled,
        }
        return feature_flags.get(feature, False)
    
    def validate_startup_requirements(self) -> List[str]:
        """
        Validate startup requirements and return list of issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        # Check required directories
        if not self.ml_model_storage_path.exists():
            issues.append(f"ML model storage path does not exist: {self.ml_model_storage_path}")
        
        # Check SSL files if HTTPS is enabled
        if self.api_enable_https:
            if self.api_ssl_cert_path and not self.api_ssl_cert_path.exists():
                issues.append(f"SSL certificate file not found: {self.api_ssl_cert_path}")
            if self.api_ssl_key_path and not self.api_ssl_key_path.exists():
                issues.append(f"SSL key file not found: {self.api_ssl_key_path}")
        
        # Check port availability (basic validation)
        if self.api_port == self.health_metrics_port:
            issues.append("API port and metrics port cannot be the same")
        
        return issues

# =============================================================================
# Configuration Hot-Reload System
# =============================================================================

class ConfigFileWatcher(FileSystemEventHandler):
    """File system event handler for configuration hot-reload."""
    
    def __init__(self, config_instance: AppConfig, reload_callback: callable):
        self.config_instance = config_instance
        self.reload_callback = reload_callback
        self._last_reload = 0
        self._reload_debounce = 1.0  # 1 second debounce
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        # Check if it's an environment file
        if event.src_path.endswith(('.env', '.env.local', '.env.production')):
            current_time = time.time()
            if current_time - self._last_reload > self._reload_debounce:
                self._last_reload = current_time
                logger.info(f"Configuration file changed: {event.src_path}")
                try:
                    self.reload_callback()
                except Exception as e:
                    logger.error(f"Failed to reload configuration: {e}")

class ConfigManager:
    """
    Configuration manager with hot-reload capability.
    
    Provides centralized configuration management with:
    - Automatic reloading on file changes
    - Validation on reload
    - Thread-safe access
    - Graceful error handling
    """
    
    def __init__(self, config_class: type = AppConfig):
        self._config_class = config_class
        self._config: Optional[AppConfig] = None
        self._observer: Optional[Observer] = None
        self._lock = threading.RLock()
        self._reload_callbacks: List[callable] = []
    
    def load_config(self) -> AppConfig:
        """Load or reload configuration."""
        with self._lock:
            try:
                new_config = self._config_class()
                
                # Validate startup requirements
                issues = new_config.validate_startup_requirements()
                if issues:
                    for issue in issues:
                        logger.warning(f"Configuration validation warning: {issue}")
                
                self._config = new_config
                logger.info("Configuration loaded successfully")
                
                # Setup hot-reload if enabled and not already watching
                if new_config.hot_reload_enabled and not self._observer:
                    self._setup_hot_reload()
                
                # Notify reload callbacks
                for callback in self._reload_callbacks:
                    try:
                        callback(self._config)
                    except Exception as e:
                        logger.error(f"Reload callback failed: {e}")
                
                return self._config
                
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                if self._config is None:
                    # If we don't have any config, create a minimal one
                    self._config = self._config_class()
                raise
    
    def get_config(self) -> AppConfig:
        """Get current configuration (load if not already loaded)."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def add_reload_callback(self, callback: callable):
        """Add callback to be called when configuration is reloaded."""
        self._reload_callbacks.append(callback)
    
    def _setup_hot_reload(self):
        """Setup file watching for hot-reload."""
        if self._observer:
            return
        
        try:
            self._observer = Observer()
            handler = ConfigFileWatcher(self._config, self.load_config)
            
            # Watch current directory for .env files
            watch_path = Path.cwd()
            self._observer.schedule(handler, str(watch_path), recursive=False)
            
            # Watch secrets directory if it exists
            secrets_dir = Path(self._config.model_config.get("secrets_dir", "/run/secrets"))
            if secrets_dir.exists():
                self._observer.schedule(handler, str(secrets_dir), recursive=False)
            
            self._observer.start()
            logger.info("Configuration hot-reload enabled")
            
        except Exception as e:
            logger.error(f"Failed to setup configuration hot-reload: {e}")
    
    def stop_watching(self):
        """Stop file watching."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Configuration watching stopped")
    
    @asynccontextmanager
    async def lifespan_manager(self) -> AsyncGenerator[AppConfig, None]:
        """Async context manager for application lifespan."""
        config = self.load_config()
        try:
            yield config
        finally:
            self.stop_watching()

# =============================================================================
# Global Configuration Instance
# =============================================================================

# Global configuration manager instance
_config_manager = ConfigManager()

def get_config() -> AppConfig:
    """
    Get the global application configuration.
    
    Returns:
        AppConfig: Current application configuration
        
    Example:
        >>> config = get_config()
        >>> db_url = config.get_database_url()
        >>> redis_url = config.get_redis_url()
    """
    return _config_manager.get_config()

def reload_config() -> AppConfig:
    """Force reload of configuration from environment/files."""
    return _config_manager.load_config()

def add_config_reload_callback(callback: callable):
    """
    Add callback to be called when configuration is reloaded.
    
    Args:
        callback: Function to call with new config as argument
    """
    _config_manager.add_reload_callback(callback)

@asynccontextmanager
async def config_lifespan() -> AsyncGenerator[AppConfig, None]:
    """
    Async context manager for configuration lifecycle.
    
    Example:
        >>> async with config_lifespan() as config:
        ...     # Use config here
        ...     db_url = config.get_database_url()
    """
    async with _config_manager.lifespan_manager() as config:
        yield config

# =============================================================================
# Utility Functions
# =============================================================================

def validate_config_environment() -> Dict[str, Any]:
    """
    Validate current environment for configuration requirements.
    
    Returns:
        Dict with validation results and recommendations
    """
    config = get_config()
    results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "recommendations": []
    }
    
    # Check environment-specific requirements
    if config.is_production:
        if config.env_debug:
            results["warnings"].append("Debug mode enabled in production")
        if not config.api_enable_https:
            results["recommendations"].append("Consider enabling HTTPS in production")
        if not config.api_rate_limit_enabled:
            results["recommendations"].append("Consider enabling rate limiting in production")
    
    # Check resource settings
    if config.db_pool_max_size < 10:
        results["recommendations"].append("Consider increasing database pool size for production")
    
    if config.redis_max_connections < 20:
        results["recommendations"].append("Consider increasing Redis connection pool for production")
    
    # Check security settings
    if not config.env_secret_key:
        results["errors"].append("SECRET_KEY not configured")
        results["valid"] = False
    
    if config.api_cors_enabled and "*" in config.api_cors_origins:
        results["warnings"].append("CORS allows all origins - consider restricting in production")
    
    return results

def get_config_summary() -> Dict[str, Any]:
    """Get a summary of current configuration (safe for logging)."""
    config = get_config()
    
    return {
        "environment": config.env_environment,
        "debug": config.env_debug,
        "database": {
            "host": config.db_host,
            "port": config.db_port,
            "database": config.db_database,
            "pool_size": f"{config.db_pool_min_size}-{config.db_pool_max_size}"
        },
        "redis": {
            "host": config.redis_host,
            "port": config.redis_port,
            "database": config.redis_database,
            "sentinel_enabled": config.redis_sentinel_enabled
        },
        "api": {
            "host": config.api_host,
            "port": config.api_port,
            "https_enabled": config.api_enable_https,
            "rate_limit_enabled": config.api_rate_limit_enabled
        },
        "features": {
            "hot_reload": config.hot_reload_enabled,
            "metrics": config.health_metrics_enabled,
            "model_serving": config.ml_model_serving_enabled,
        }
    }

# =============================================================================
# Compatibility Layer for Migration
# =============================================================================

def get_database_config() -> DatabaseConfig:
    """Get database configuration instance (compatibility function)."""
    config = get_config()
    return config.database

def get_redis_config() -> RedisConfig:
    """Get Redis configuration instance (compatibility function)."""
    config = get_config()
    return config.redis

# Legacy function aliases for smooth migration
def get_config_manager():
    """Get configuration manager (legacy compatibility)."""
    return _config_manager

# =============================================================================
# Redis Cache Configuration Integration
# =============================================================================

class LegacyRedisConfig(BaseModel):
    """Legacy Redis configuration for compatibility with utils.redis_cache."""
    
    host: str = Field(default="localhost", description="Redis server hostname")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis server port")
    cache_db: int = Field(default=0, ge=0, le=15, description="Redis database number for cache")
    pool_size: int = Field(default=10, ge=1, description="Connection pool size")
    max_connections: int = Field(default=50, ge=1, description="Maximum connections")
    connect_timeout: int = Field(default=5, ge=0, description="Connection timeout in seconds")
    socket_timeout: int = Field(default=5, ge=0, description="Socket timeout in seconds")
    socket_keepalive: bool = Field(default=True, description="Enable socket keep-alive")
    socket_keepalive_options: dict = Field(default_factory=dict, description="Socket keep-alive options")
    use_ssl: bool = Field(default=False, description="Use SSL connection")
    monitoring_enabled: bool = Field(default=True, description="Enable monitoring")
    
    @classmethod
    def from_core_config(cls, redis_config: RedisConfig) -> 'LegacyRedisConfig':
        """Create legacy config from core Redis config."""
        return cls(
            host=redis_config.host,
            port=redis_config.port,
            cache_db=redis_config.database,
            pool_size=min(redis_config.max_connections, 10),
            max_connections=redis_config.max_connections,
            connect_timeout=redis_config.connection_timeout,
            socket_timeout=redis_config.socket_timeout,
            socket_keepalive=True,
            socket_keepalive_options={},
            use_ssl=False,  # Add to core config if needed
            monitoring_enabled=True
        )
    
    @classmethod
    def load_from_yaml(cls, path: str) -> 'LegacyRedisConfig':
        """Load from YAML (compatibility method)."""
        config = get_config()
        return cls.from_core_config(config.redis)
    
    def validate_config(self) -> None:
        """Validate configuration (compatibility method)."""
        pass  # Validation is handled by Pydantic in core config

# =============================================================================
# Database Configuration Extensions
# =============================================================================

# Legacy compatibility code removed - using unified asyncpg architecture

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Environment: {config.environment.environment}")
    print(f"Database URL: {config.get_database_url()}")
    print(f"Redis URL: {config.get_redis_url()}")
    
    # Validate environment
    validation = validate_config_environment()
    if validation["errors"]:
        print("Configuration errors:", validation["errors"])
    if validation["warnings"]:
        print("Configuration warnings:", validation["warnings"])
    if validation["recommendations"]:
        print("Recommendations:", validation["recommendations"])