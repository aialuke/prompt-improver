"""Comprehensive Type Definitions with Pydantic Validation for 2025.

This module provides robust type definitions using modern Python typing features
and Pydantic validation for the prompt improver system. It follows 2025 best
practices for type safety, runtime validation, and clean architecture.

Features:
- Pydantic v2 models with strict validation
- Protocol definitions for service contracts
- Generic types for ML components
- Configuration types with validation
- Performance and metrics types
- Error handling with typed exceptions
- JSON schema generation for API documentation
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any, Annotated, Callable, Dict, Generic, List, Literal, Optional, 
    Protocol, Set, Tuple, TypeVar, Union, runtime_checkable, NewType,
    TYPE_CHECKING, TypeAlias, ClassVar
)
from uuid import UUID

try:
    from pydantic import (
        BaseModel, Field, ConfigDict, validator, field_validator,
        model_validator, computed_field, ValidationError,
        AnyUrl, EmailStr, Json, StrictStr, StrictInt, StrictFloat,
        constr, conint, confloat, conlist, root_validator
    )
    from pydantic.types import PositiveInt, NonNegativeInt, PositiveFloat
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback for when Pydantic is not available
    BaseModel = object
    Field = lambda *args, **kwargs: None
    ConfigDict = lambda *args, **kwargs: {}
    ValidationError = ValueError
    PYDANTIC_AVAILABLE = False
    # Define basic constraints as type aliases
    PositiveInt = int
    NonNegativeInt = int
    PositiveFloat = float
    StrictStr = str
    StrictInt = int
    StrictFloat = float
    constr = str
    conint = int
    confloat = float
    conlist = list
    validator = lambda *args, **kwargs: lambda f: f
    field_validator = lambda *args, **kwargs: lambda f: f
    model_validator = lambda *args, **kwargs: lambda f: f
    computed_field = lambda *args, **kwargs: lambda f: f
    root_validator = lambda *args, **kwargs: lambda f: f

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Type variables for generics
T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
T_contra = TypeVar('T_contra', contravariant=True)

# Specialized type aliases
UserId = NewType('UserId', UUID)
SessionId = NewType('SessionId', str)
ModelId = NewType('ModelId', str)
MetricName = NewType('MetricName', str)
ConfigKey = NewType('ConfigKey', str)

# String constraints
PromptText: TypeAlias = Annotated[str, Field(min_length=1, max_length=10000)]
ModelName: TypeAlias = Annotated[str, Field(pattern=r'^[a-zA-Z0-9_-]+$')]
Version: TypeAlias = Annotated[str, Field(pattern=r'^\d+\.\d+\.\d+$')]
Percentage: TypeAlias = Annotated[float, Field(ge=0.0, le=100.0)]
Score: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]

# Numeric constraints  
PositiveTimeout: TypeAlias = Annotated[float, Field(gt=0, le=3600)]  # Max 1 hour
RetryCount: TypeAlias = Annotated[int, Field(ge=0, le=10)]
Priority: TypeAlias = Annotated[int, Field(ge=1, le=10)]

class BaseEnum(str, Enum):
    """Base enum with string values for better JSON serialization."""
    
    def __str__(self) -> str:
        return self.value

class EnvironmentType(BaseEnum):
    """Environment types for configuration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(BaseEnum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ServiceStatus(BaseEnum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class ModelType(BaseEnum):
    """Machine learning model types."""
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    CLUSTERER = "clusterer"
    TRANSFORMER = "transformer"
    EMBEDDER = "embedder"
    GENERATOR = "generator"

class OptimizationStatus(BaseEnum):
    """Optimization job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MetricType(BaseEnum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

if PYDANTIC_AVAILABLE:
    class BaseConfig(BaseModel):
        """Base configuration class with common settings."""
        
        model_config = ConfigDict(
            extra='forbid',
            validate_assignment=True,
            use_enum_values=True,
            frozen=False,  # Allow updates for configuration
            str_strip_whitespace=True,
            validate_default=True,
            # Performance optimization
            arbitrary_types_allowed=True,
            # JSON schema generation
            json_schema_extra={
                "examples": []
            }
        )
        
        @field_validator('*', mode='before')
        @classmethod
        def empty_strings_to_none(cls, v):
            """Convert empty strings to None for optional fields."""
            if isinstance(v, str) and v.strip() == '':
                return None
            return v
    
    class TimestampedModel(BaseConfig):
        """Base model with automatic timestamps."""
        
        created_at: datetime = Field(
            default_factory=lambda: datetime.now(timezone.utc),
            description="Creation timestamp"
        )
        updated_at: Optional[datetime] = Field(
            default=None,
            description="Last update timestamp"
        )
        
        @model_validator(mode='before')
        @classmethod
        def set_updated_at(cls, values):
            """Set updated_at when model is modified."""
            if isinstance(values, dict) and 'updated_at' not in values:
                values['updated_at'] = datetime.now(timezone.utc)
            return values

    class HealthCheckConfig(BaseConfig):
        """Configuration for health checks."""
        
        enabled: bool = Field(default=True, description="Enable health checks")
        interval_seconds: PositiveTimeout = Field(default=30.0, description="Check interval")
        timeout_seconds: PositiveTimeout = Field(default=5.0, description="Check timeout")
        failure_threshold: RetryCount = Field(default=3, description="Failures before unhealthy")
        success_threshold: RetryCount = Field(default=1, description="Successes before healthy")
        
        @field_validator('timeout_seconds')
        @classmethod
        def timeout_less_than_interval(cls, v, values):
            """Ensure timeout is less than interval."""
            if 'interval_seconds' in values and v >= values['interval_seconds']:
                raise ValueError('timeout_seconds must be less than interval_seconds')
            return v

    class DatabaseConfig(BaseConfig):
        """Database connection configuration."""
        
        host: str = Field(description="Database host")
        port: Annotated[int, Field(ge=1, le=65535)] = Field(default=5432, description="Database port")
        database: str = Field(description="Database name")
        username: str = Field(description="Database username")
        password: str = Field(description="Database password", repr=False)  # Hide in repr
        pool_size: Annotated[int, Field(ge=1, le=100)] = Field(default=10, description="Connection pool size")
        max_overflow: Annotated[int, Field(ge=0, le=50)] = Field(default=20, description="Max pool overflow")
        pool_timeout: PositiveTimeout = Field(default=30.0, description="Pool timeout")
        pool_recycle: PositiveTimeout = Field(default=3600.0, description="Pool recycle time")
        
        @computed_field
        @property
        def connection_url(self) -> str:
            """Generate connection URL."""
            return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    class RedisConfig(BaseConfig):
        """Redis configuration."""
        
        host: str = Field(default="localhost", description="Redis host")
        port: Annotated[int, Field(ge=1, le=65535)] = Field(default=6379, description="Redis port")
        db: Annotated[int, Field(ge=0, le=15)] = Field(default=0, description="Redis database")
        password: Optional[str] = Field(default=None, description="Redis password", repr=False)
        max_connections: PositiveInt = Field(default=50, description="Max connections")
        connection_timeout: PositiveTimeout = Field(default=5.0, description="Connection timeout")
        
        @computed_field
        @property
        def redis_url(self) -> str:
            """Generate Redis URL."""
            auth_part = f":{self.password}@" if self.password else ""
            return f"redis://{auth_part}{self.host}:{self.port}/{self.db}"

    class MLModelConfig(BaseConfig):
        """Machine learning model configuration."""
        
        model_id: ModelId = Field(description="Unique model identifier")
        model_type: ModelType = Field(description="Type of ML model")
        name: ModelName = Field(description="Human-readable model name")
        version: Version = Field(description="Model version")
        framework: str = Field(description="ML framework (sklearn, tensorflow, etc.)")
        
        # Training parameters
        max_training_time: PositiveTimeout = Field(default=3600.0, description="Max training time")
        validation_split: Score = Field(default=0.2, description="Validation split ratio")
        early_stopping: bool = Field(default=True, description="Enable early stopping")
        
        # Performance thresholds
        min_accuracy: Score = Field(default=0.7, description="Minimum acceptable accuracy")
        max_inference_time: PositiveTimeout = Field(default=1.0, description="Max inference time")
        
        # Resource limits
        max_memory_mb: PositiveInt = Field(default=1024, description="Max memory usage")
        max_cpu_cores: PositiveInt = Field(default=4, description="Max CPU cores")
        
        @field_validator('version')
        @classmethod
        def valid_semver(cls, v):
            """Validate semantic version format."""
            parts = v.split('.')
            if len(parts) != 3:
                raise ValueError('Version must be in format x.y.z')
            try:
                [int(part) for part in parts]
            except ValueError:
                raise ValueError('Version parts must be integers')
            return v

    class PromptRequest(TimestampedModel):
        """Request for prompt improvement."""
        
        request_id: UUID = Field(default_factory=uuid.uuid4, description="Unique request ID")
        user_id: Optional[UserId] = Field(default=None, description="User identifier")
        session_id: Optional[SessionId] = Field(default=None, description="Session identifier")
        
        original_prompt: PromptText = Field(description="Original prompt text")
        context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
        rules: List[str] = Field(default_factory=list, description="Rules to apply")
        priority: Priority = Field(default=5, description="Request priority")
        
        # Processing options
        use_ml: bool = Field(default=True, description="Use ML-based improvement")
        use_rules: bool = Field(default=True, description="Use rule-based improvement")
        max_processing_time: PositiveTimeout = Field(default=30.0, description="Max processing time")
        
        @field_validator('context')
        @classmethod
        def validate_context_size(cls, v):
            """Ensure context is not too large."""
            import json
            if len(json.dumps(v)) > 10000:  # 10KB limit
                raise ValueError('Context too large (max 10KB)')
            return v

    class PromptResponse(TimestampedModel):
        """Response from prompt improvement."""
        
        request_id: UUID = Field(description="Original request ID")
        improved_prompt: PromptText = Field(description="Improved prompt text")
        
        # Metrics
        confidence_score: Score = Field(description="Confidence in improvement")
        processing_time_ms: NonNegativeInt = Field(description="Processing time in milliseconds")
        improvement_score: Score = Field(description="Improvement quality score")
        
        # Applied changes
        rules_applied: List[str] = Field(description="Rules that were applied")
        ml_predictions: Dict[str, float] = Field(default_factory=dict, description="ML predictions")
        suggestions: List[str] = Field(default_factory=list, description="Additional suggestions")
        
        # Performance data
        token_count_original: NonNegativeInt = Field(description="Original token count")
        token_count_improved: NonNegativeInt = Field(description="Improved token count")
        readability_score: Optional[Score] = Field(default=None, description="Readability score")

    class MetricPoint(TimestampedModel):
        """A single metric data point."""
        
        name: MetricName = Field(description="Metric name")
        value: float = Field(description="Metric value")
        metric_type: MetricType = Field(description="Type of metric")
        labels: Dict[str, str] = Field(default_factory=dict, description="Metric labels")
        tags: Set[str] = Field(default_factory=set, description="Metric tags")
        
        @field_validator('labels')
        @classmethod
        def validate_labels(cls, v):
            """Validate metric labels."""
            for key, value in v.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError('Labels must be string key-value pairs')
                if len(key) > 100 or len(value) > 1000:
                    raise ValueError('Label keys/values too long')
            return v

    class HealthStatus(TimestampedModel):
        """Service health status."""
        
        service_name: str = Field(description="Service name")
        status: ServiceStatus = Field(description="Current status")
        version: Optional[Version] = Field(default=None, description="Service version")
        
        # Health metrics
        uptime_seconds: NonNegativeInt = Field(description="Service uptime")
        memory_usage_mb: NonNegativeInt = Field(description="Memory usage")
        cpu_usage_percent: Percentage = Field(description="CPU usage percentage")
        
        # Dependencies
        dependencies: Dict[str, ServiceStatus] = Field(
            default_factory=dict, 
            description="Status of dependencies"
        )
        
        # Additional details
        details: Dict[str, Any] = Field(default_factory=dict, description="Additional status details")
        error_message: Optional[str] = Field(default=None, description="Error message if unhealthy")

    class PerformanceMetrics(TimestampedModel):
        """Performance metrics collection."""
        
        component: str = Field(description="Component name")
        operation: str = Field(description="Operation name")
        
        # Timing metrics
        response_time_ms: NonNegativeInt = Field(description="Response time in milliseconds")
        cpu_time_ms: NonNegativeInt = Field(description="CPU time in milliseconds")
        memory_peak_mb: NonNegativeInt = Field(description="Peak memory usage")
        
        # Throughput metrics
        requests_per_second: PositiveFloat = Field(description="Requests per second")
        success_rate: Percentage = Field(description="Success rate percentage")
        error_rate: Percentage = Field(description="Error rate percentage")
        
        # Resource utilization
        database_queries: NonNegativeInt = Field(default=0, description="Number of database queries")
        cache_hits: NonNegativeInt = Field(default=0, description="Cache hits")
        cache_misses: NonNegativeInt = Field(default=0, description="Cache misses")
        
        @computed_field
        @property
        def cache_hit_rate(self) -> float:
            """Calculate cache hit rate."""
            total = self.cache_hits + self.cache_misses
            return (self.cache_hits / total * 100) if total > 0 else 0.0

    class ConfigurationSchema(BaseConfig):
        """Application configuration schema."""
        
        # Environment
        environment: EnvironmentType = Field(default=EnvironmentType.DEVELOPMENT)
        debug: bool = Field(default=False, description="Enable debug mode")
        log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
        
        # Service configuration
        service_name: str = Field(default="prompt-improver", description="Service name")
        service_version: Version = Field(description="Service version")
        api_host: str = Field(default="0.0.0.0", description="API host")
        api_port: Annotated[int, Field(ge=1, le=65535)] = Field(default=8000, description="API port")
        
        # Database
        database: DatabaseConfig = Field(description="Database configuration")
        redis: RedisConfig = Field(description="Redis configuration")
        
        # ML Configuration
        ml: MLModelConfig = Field(description="ML model configuration")
        
        # Health checks
        health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
        
        # Security
        secret_key: str = Field(description="Secret key for encryption", repr=False)
        
        # Rate limiting
        rate_limit_requests: PositiveInt = Field(default=100, description="Requests per window")
        rate_limit_window_minutes: PositiveInt = Field(default=1, description="Rate limit window")
        
        @model_validator(mode='after')
        def validate_configuration(self):
            """Validate configuration consistency."""
            if self.environment == EnvironmentType.PRODUCTION:
                if self.debug:
                    raise ValueError('Debug mode should not be enabled in production')
                if self.log_level == LogLevel.DEBUG:
                    raise ValueError('Debug logging should not be used in production')
            return self

else:
    # Fallback classes when Pydantic is not available
    class BaseConfig:
        """Fallback base configuration class."""
        pass
    
    class TimestampedModel:
        """Fallback timestamped model."""
        pass
    
    # Define other fallback classes as needed
    HealthCheckConfig = BaseConfig
    DatabaseConfig = BaseConfig
    RedisConfig = BaseConfig
    MLModelConfig = BaseConfig
    PromptRequest = BaseConfig
    PromptResponse = BaseConfig
    MetricPoint = BaseConfig
    HealthStatus = BaseConfig
    PerformanceMetrics = BaseConfig
    ConfigurationSchema = BaseConfig

# Protocol definitions for service contracts
@runtime_checkable
class ConfigurationProvider(Protocol):
    """Protocol for configuration providers."""
    
    def get(self, key: ConfigKey, default: Any = None) -> Any:
        """Get configuration value."""
        ...
    
    def set(self, key: ConfigKey, value: Any) -> None:
        """Set configuration value."""
        ...
    
    def reload(self) -> None:
        """Reload configuration from source."""
        ...

@runtime_checkable
class MetricsCollector(Protocol):
    """Protocol for metrics collection."""
    
    def record_metric(self, metric: MetricPoint) -> None:
        """Record a metric point."""
        ...
    
    def get_metrics(self, 
                   name_pattern: Optional[str] = None,
                   labels: Optional[Dict[str, str]] = None) -> List[MetricPoint]:
        """Get metrics matching criteria."""
        ...
    
    def clear_metrics(self, older_than: Optional[datetime] = None) -> None:
        """Clear old metrics."""
        ...

@runtime_checkable
class HealthChecker(Protocol):
    """Protocol for health checking."""
    
    async def check_health(self) -> HealthStatus:
        """Perform health check."""
        ...
    
    def get_dependencies(self) -> Dict[str, 'HealthChecker']:
        """Get health check dependencies."""
        ...

@runtime_checkable
class PerformanceMonitor(Protocol):
    """Protocol for performance monitoring."""
    
    def start_timer(self, operation: str) -> str:
        """Start performance timer."""
        ...
    
    def end_timer(self, timer_id: str) -> PerformanceMetrics:
        """End performance timer and return metrics."""
        ...
    
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        ...

# Generic types for ML components
if TYPE_CHECKING:
    MLFeatures = NDArray[np.float64]
    MLLabels = NDArray[np.int64]
    MLPredictions = Union[NDArray[np.float64], NDArray[np.int64]]
else:
    MLFeatures = Any
    MLLabels = Any
    MLPredictions = Any

@runtime_checkable
class MLModel(Protocol, Generic[T]):
    """Protocol for ML models."""
    
    def fit(self, features: MLFeatures, labels: MLLabels) -> None:
        """Train the model."""
        ...
    
    def predict(self, features: MLFeatures) -> MLPredictions:
        """Make predictions."""
        ...
    
    def get_model_info(self) -> MLModelConfig:
        """Get model configuration."""
        ...
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        ...
    
    @classmethod
    def load(cls, path: Path) -> 'MLModel[T]':
        """Load model from disk."""
        ...

# Error types with validation
class PromptImproverError(Exception):
    """Base exception for prompt improver."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(message)

class ValidationError(PromptImproverError):
    """Validation error with field details."""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Optional[Any] = None):
        self.field = field
        self.value = value
        super().__init__(message, "VALIDATION_ERROR", {"field": field, "value": value})

class ConfigurationError(PromptImproverError):
    """Configuration error."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        self.config_key = config_key
        super().__init__(message, "CONFIGURATION_ERROR", {"config_key": config_key})

class MLModelError(PromptImproverError):
    """ML model error."""
    
    def __init__(self, message: str, model_id: Optional[str] = None, 
                 operation: Optional[str] = None):
        self.model_id = model_id
        self.operation = operation
        super().__init__(message, "ML_MODEL_ERROR", {
            "model_id": model_id,
            "operation": operation
        })

# Utility functions for type conversion and validation
def validate_prompt_text(text: str) -> PromptText:
    """Validate and convert text to PromptText type."""
    if not isinstance(text, str):
        raise ValidationError("Prompt text must be a string", "prompt_text", text)
    
    text = text.strip()
    if not text:
        raise ValidationError("Prompt text cannot be empty", "prompt_text", text)
    
    if len(text) > 10000:
        raise ValidationError("Prompt text too long (max 10000 chars)", "prompt_text", text)
    
    return text

def validate_score(score: float) -> Score:
    """Validate and convert float to Score type."""
    if not isinstance(score, (int, float)):
        raise ValidationError("Score must be a number", "score", score)
    
    if not 0.0 <= score <= 1.0:
        raise ValidationError("Score must be between 0.0 and 1.0", "score", score)
    
    return float(score)

def validate_percentage(percentage: float) -> Percentage:
    """Validate and convert float to Percentage type."""
    if not isinstance(percentage, (int, float)):
        raise ValidationError("Percentage must be a number", "percentage", percentage)
    
    if not 0.0 <= percentage <= 100.0:
        raise ValidationError("Percentage must be between 0.0 and 100.0", "percentage", percentage)
    
    return float(percentage)

# Export main types and functions
__all__ = [
    # Enums
    "EnvironmentType", "LogLevel", "ServiceStatus", "ModelType", 
    "OptimizationStatus", "MetricType",
    
    # Type aliases
    "UserId", "SessionId", "ModelId", "MetricName", "ConfigKey",
    "PromptText", "ModelName", "Version", "Percentage", "Score",
    "PositiveTimeout", "RetryCount", "Priority",
    
    # Models (if Pydantic available)
    "BaseConfig", "TimestampedModel", "HealthCheckConfig", 
    "DatabaseConfig", "RedisConfig", "MLModelConfig",
    "PromptRequest", "PromptResponse", "MetricPoint", 
    "HealthStatus", "PerformanceMetrics", "ConfigurationSchema",
    
    # Protocols
    "ConfigurationProvider", "MetricsCollector", "HealthChecker", 
    "PerformanceMonitor", "MLModel",
    
    # ML types
    "MLFeatures", "MLLabels", "MLPredictions",
    
    # Exceptions
    "PromptImproverError", "ValidationError", "ConfigurationError", "MLModelError",
    
    # Utility functions
    "validate_prompt_text", "validate_score", "validate_percentage",
    
    # Constants
    "PYDANTIC_AVAILABLE"
]