"""External Services Configuration for Native ML Deployment

Configuration for external PostgreSQL, Redis, and other services used in native ML deployment.
Replaces Docker-based services with external service connections.
"""
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, Optional

@dataclass
class PostgreSQLConfig:
    """PostgreSQL configuration for MLflow model registry and tracking."""
    host: str = field(default_factory=lambda: os.getenv('POSTGRES_HOST', 'localhost'))
    port: int = 5432
    database: str = 'mlflow'
    username: str = 'mlflow'
    password: str = 'mlflow'
    schema: str = 'public'
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    ssl_mode: str = 'prefer'
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    ssl_ca_path: Optional[str] = None

    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f'postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}'

    def sqlalchemy_url(self) -> str:
        """Generate SQLAlchemy URL with connection pool settings."""
        base_url = self.connection_string()
        params = []
        if self.ssl_mode != 'disable':
            params.append(f'sslmode={self.ssl_mode}')
        if self.ssl_cert_path:
            params.append(f'sslcert={self.ssl_cert_path}')
        if self.ssl_key_path:
            params.append(f'sslkey={self.ssl_key_path}')
        if self.ssl_ca_path:
            params.append(f'sslrootcert={self.ssl_ca_path}')
        if params:
            return f"{base_url}?{'&'.join(params)}"
        return base_url

@dataclass
class RedisConfig:
    """Redis configuration for ML caching and session management."""
    host: str = field(default_factory=lambda: os.getenv('REDIS_HOST', 'redis.external.service'))
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    connection_timeout: int = 5
    socket_timeout: int = 5
    max_connections: int = 50
    retry_on_timeout: bool = True
    ssl: bool = False
    ssl_cert_reqs: Optional[str] = None
    ssl_ca_certs: Optional[str] = None
    ssl_cert_file: Optional[str] = None
    ssl_key_file: Optional[str] = None
    default_ttl: int = 3600
    model_cache_ttl: int = 86400
    prediction_cache_ttl: int = 300

    def connection_params(self) -> Dict[str, Any]:
        """Generate Redis connection parameters."""
        params = {'host': self.host, 'port': self.port, 'db': self.database, 'socket_timeout': self.socket_timeout, 'socket_connect_timeout': self.connection_timeout, 'retry_on_timeout': self.retry_on_timeout, 'max_connections': self.max_connections}
        if self.password:
            params['password'] = self.password
        if self.ssl:
            params['ssl'] = True
            if self.ssl_cert_reqs:
                params['ssl_cert_reqs'] = self.ssl_cert_reqs
            if self.ssl_ca_certs:
                params['ssl_ca_certs'] = self.ssl_ca_certs
            if self.ssl_cert_file:
                params['ssl_certfile'] = self.ssl_cert_file
            if self.ssl_key_file:
                params['ssl_keyfile'] = self.ssl_key_file
        return params

@dataclass
class MLflowConfig:
    """MLflow configuration for model registry and tracking."""
    tracking_uri: Optional[str] = None
    artifact_root: str = '/var/mlflow/artifacts'
    default_artifact_root: str = '/var/mlflow/artifacts'
    registry_store_uri: Optional[str] = None
    auth_enabled: bool = False
    auth_config_path: Optional[str] = None
    serve_artifacts: bool = True
    artifacts_only: bool = False
    host: str = '0.0.0.0'
    port: int = 5000
    workers: int = 4

    def configure_tracking_uri(self, postgresql_config: PostgreSQLConfig):
        """Configure MLflow to use PostgreSQL as backend store."""
        if not self.tracking_uri:
            self.tracking_uri = postgresql_config.sqlalchemy_url()
        if not self.registry_store_uri:
            self.registry_store_uri = postgresql_config.sqlalchemy_url()

@dataclass
class NativeDeploymentConfig:
    """Configuration for native model deployment without containers."""
    process_manager: str = 'systemd'
    working_directory: str = '/var/lib/ml-models'
    log_directory: str = '/var/log/ml-models'
    pid_directory: str = '/var/run/ml-models'
    service_user: str = 'ml-service'
    service_group: str = 'ml-service'
    restart_policy: str = 'always'
    restart_delay: int = 5
    max_restart_attempts: int = 3
    load_balancer: str = 'nginx'
    load_balancer_config_path: str = '/etc/nginx/conf.d/ml-models.conf'
    health_check_interval: int = 30
    health_check_timeout: int = 10
    health_check_retries: int = 3
    worker_processes: int = 1
    worker_connections: int = 1000
    max_requests_per_worker: int = 1000
    worker_memory_limit_mb: int = 512

@dataclass
class ExternalServicesConfig:
    """Complete external services configuration for native ML deployment."""
    postgresql: PostgreSQLConfig = field(default_factory=PostgreSQLConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    deployment: NativeDeploymentConfig = field(default_factory=NativeDeploymentConfig)
    environment: str = 'production'

    def __post_init__(self):
        """Configure services after initialization."""
        self.mlflow.configure_tracking_uri(self.postgresql)
        self._load_environment_overrides()

    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables."""
        if os.getenv('ML_POSTGRES_HOST'):
            self.postgresql.host = os.getenv('ML_POSTGRES_HOST')
        if os.getenv('ML_POSTGRES_PORT'):
            self.postgresql.port = int(os.getenv('ML_POSTGRES_PORT'))
        if os.getenv('ML_POSTGRES_USER'):
            self.postgresql.username = os.getenv('ML_POSTGRES_USER')
        if os.getenv('ML_POSTGRES_PASSWORD'):
            self.postgresql.password = os.getenv('ML_POSTGRES_PASSWORD')
        if os.getenv('ML_POSTGRES_DATABASE'):
            self.postgresql.database = os.getenv('ML_POSTGRES_DATABASE')
        if os.getenv('ML_REDIS_HOST'):
            self.redis.host = os.getenv('ML_REDIS_HOST')
        if os.getenv('ML_REDIS_PORT'):
            self.redis.port = int(os.getenv('ML_REDIS_PORT'))
        if os.getenv('ML_REDIS_PASSWORD'):
            self.redis.password = os.getenv('ML_REDIS_PASSWORD')
        if os.getenv('MLFLOW_TRACKING_URI'):
            self.mlflow.tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        if os.getenv('MLFLOW_ARTIFACT_ROOT'):
            self.mlflow.artifact_root = os.getenv('MLFLOW_ARTIFACT_ROOT')
        if os.getenv('ML_WORKING_DIR'):
            self.deployment.working_directory = os.getenv('ML_WORKING_DIR')
        if os.getenv('ML_LOG_DIR'):
            self.deployment.log_directory = os.getenv('ML_LOG_DIR')

    @classmethod
    def from_environment(cls, environment: str='production') -> 'ExternalServicesConfig':
        """Create configuration from environment variables."""
        config = cls()
        config.environment = environment
        config._load_environment_overrides()
        return config

    def validate(self) -> bool:
        """Validate the configuration."""
        errors = []
        if not self.postgresql.host or not self.postgresql.database:
            errors.append('PostgreSQL host and database are required')
        if not self.redis.host:
            errors.append('Redis host is required')
        for directory in [self.deployment.working_directory, self.deployment.log_directory, self.deployment.pid_directory, self.mlflow.artifact_root]:
            path = Path(directory)
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    errors.append(f'Cannot create directory: {directory}')
            elif not os.access(directory, os.W_OK):
                errors.append(f'Directory not writable: {directory}')
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        return True

    def get_service_urls(self) -> Dict[str, str]:
        """Get service URLs for health checks and monitoring."""
        return {'postgresql': f'postgresql://{self.postgresql.host}:{self.postgresql.port}/{self.postgresql.database}', 'redis': f'redis://{self.redis.host}:{self.redis.port}/{self.redis.database}', 'mlflow': f'http://{self.mlflow.host}:{self.mlflow.port}'}
