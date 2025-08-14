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

import os
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    NewType,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)
from uuid import UUID

from pydantic import BaseModel, Field

# Import SecurityContext from shared types for compatibility
from prompt_improver.shared.types.core import SecurityContext

# SQLModel availability check
try:
    import sqlmodel
    SQLMODEL_AVAILABLE = True
except ImportError:
    SQLMODEL_AVAILABLE = False

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
UserId = NewType("UserId", UUID)
SessionId = NewType("SessionId", str)
ModelId = NewType("ModelId", str)
MetricName = NewType("MetricName", str)
ConfigKey = NewType("ConfigKey", str)
PromptText = str
ModelName = str
Version = str
Percentage = float
Score = float
PositiveTimeout = float
RetryCount = int
Priority = int


class BaseEnum(StrEnum):
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


# Canonical cache-related enums (Clean Break - 2025)
class CacheType(BaseEnum):
    """Canonical cache domain types used across APES.

    Clean-break consolidation replacing scattered cache type enums and strings.
    """

    LINGUISTIC = "linguistic"
    DOMAIN = "domain"
    SESSION = "session"
    PROMPT = "prompt"
    RULE = "rule"
    ANALYTICS = "analytics"
    ML_ANALYSIS = "ml_analysis"
    PATTERN_DISCOVERY = "pattern_discovery"
    ALL = "all"


class CacheLevel(BaseEnum):
    """Canonical cache levels."""

    L1 = "l1"
    L2 = "l2"
    L3 = "l3"
    ALL = "all"


class InvalidationType(BaseEnum):
    """What/why classification of invalidation events."""

    PATTERN = "pattern"
    DEPENDENCY = "dependency"
    CASCADE = "cascade"
    MANUAL = "manual"
    TTL_EXPIRED = "ttl_expired"
    CAPACITY = "capacity"


class InvalidationStrategy(BaseEnum):
    """How invalidation is performed (strategy)."""

    IMMEDIATE = "immediate"
    BATCH = "batch"
    TIME_BASED = "time_based"
    DEPENDENCY = "dependency"


class CoordinationAction(BaseEnum):
    """Cross-level coordination actions."""

    PROMOTE = "promote"
    DEMOTE = "demote"
    REPLICATE = "replicate"
    INVALIDATE = "invalidate"
    WARM = "warm"


class AccessPattern(BaseEnum):
    """Cache access pattern classification."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    TEMPORAL = "temporal"
    BURST = "burst"


class CacheBackend(BaseEnum):
    """Backend semantics associated with a cache level (optional)."""

    MEMORY = "memory"
    REDIS = "redis"
    DATABASE = "database"


class BaseConfig(BaseModel):
    """Base configuration class with common settings."""


class TimestampedModel(BaseConfig):
    """Base model with automatic timestamps."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = Field(default=None)


class HealthCheckConfig(BaseConfig):
    """Configuration for health checks."""

    enabled: bool = Field(default=True)
    interval_seconds: float = Field(default=30.0)
    timeout_seconds: float = Field(default=5.0)
    failure_threshold: int = Field(default=3)
    success_threshold: int = Field(default=1)


class DatabaseConfig(BaseConfig):
    """Database connection configuration."""

    host: str = Field()
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field()
    username: str = Field()
    password: str = Field(repr=False)
    pool_size: int = Field(default=10, ge=1, le=100)
    max_overflow: int = Field(default=20, ge=0, le=50)
    pool_timeout: float = Field(default=30.0)
    pool_recycle: float = Field(default=3600.0)

    def connection_url(self) -> str:
        """Generate connection URL."""
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseConfig):
    """Redis configuration."""

    host: str = Field(default_factory=lambda: os.getenv("REDIS_HOST", "redis"))
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0, le=15)
    password: str | None = Field(default=None, repr=False)
    max_connections: int = Field(default=50)
    connection_timeout: float = Field(default=5.0)

    def redis_url(self) -> str:
        """Generate Redis URL."""
        auth_part = f":{self.password}@" if self.password else ""
        return f"redis://{auth_part}{self.host}:{self.port}/{self.db}"


class MLModelConfig(BaseConfig):
    """Machine learning model configuration."""

    model_id: str = Field()
    model_type: ModelType = Field()
    name: str = Field()
    version: str = Field()
    framework: str = Field()
    max_training_time: float = Field(default=3600.0)
    validation_split: float = Field(default=0.2, ge=0.0, le=1.0)
    early_stopping: bool = Field(default=True)
    min_accuracy: float = Field(default=0.7, ge=0.0, le=1.0)
    max_inference_time: float = Field(default=1.0)
    max_memory_mb: int = Field(default=1024)
    max_cpu_cores: int = Field(default=4)


class PromptRequest(TimestampedModel):
    """Request for prompt improvement."""

    request_id: UUID = Field(default_factory=uuid.uuid4)
    user_id: UUID | None = Field(default=None)
    session_id: str | None = Field(default=None)
    original_prompt: str = Field(min_length=1, max_length=10000)
    context: dict[str, str | int | float | bool | list[str]] = Field(
        default_factory=dict
    )
    rules: list[str] = Field(default_factory=list)
    priority: int = Field(default=5, ge=1, le=10)
    use_ml: bool = Field(default=True)
    use_rules: bool = Field(default=True)
    max_processing_time: float = Field(default=30.0, gt=0, le=3600)


class PromptResponse(TimestampedModel):
    """Response from prompt improvement."""

    request_id: UUID = Field()
    improved_prompt: str = Field(min_length=1, max_length=10000)
    confidence_score: float = Field(ge=0.0, le=1.0)
    processing_time_ms: int = Field(ge=0)
    improvement_score: float = Field(ge=0.0, le=1.0)
    rules_applied: list[str] = Field()
    ml_predictions: dict[str, float] = Field(default_factory=dict)
    suggestions: list[str] = Field(default_factory=list)
    token_count_original: int = Field(ge=0)
    token_count_improved: int = Field(ge=0)
    readability_score: float | None = Field(default=None)


class MetricPoint(TimestampedModel):
    """A single metric data point."""

    name: str = Field()
    value: float = Field()
    metric_type: MetricType = Field()
    labels: dict[str, str] = Field(default_factory=dict)
    tags: set[str] = Field(default_factory=set)


class HealthStatus(TimestampedModel):
    """Service health status."""

    service_name: str = Field()
    status: ServiceStatus = Field()
    version: str | None = Field(default=None)
    uptime_seconds: int = Field(ge=0)
    memory_usage_mb: int = Field(ge=0)
    cpu_usage_percent: float = Field(ge=0.0, le=100.0)
    dependencies: dict[str, ServiceStatus] = Field(default_factory=dict)
    details: dict[str, str | int | float | bool] = Field(default_factory=dict)
    error_message: str | None = Field(default=None)


class PerformanceMetrics(TimestampedModel):
    """Performance metrics collection."""

    component: str = Field()
    operation: str = Field()
    response_time_ms: int = Field(ge=0)
    cpu_time_ms: int = Field(ge=0)
    memory_peak_mb: int = Field(ge=0)
    requests_per_second: float = Field(gt=0)
    success_rate: float = Field(ge=0.0, le=100.0)
    error_rate: float = Field(ge=0.0, le=100.0)
    database_queries: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)

    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total * 100 if total > 0 else 0.0


class ConfigurationSchema(BaseConfig):
    """Application configuration schema."""

    environment: EnvironmentType = Field(default=EnvironmentType.DEVELOPMENT)
    debug: bool = Field(default=False)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    service_name: str = Field(default="prompt-improver")
    service_version: str = Field()
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    database: DatabaseConfig = Field()
    redis: RedisConfig = Field()
    ml: MLModelConfig = Field()
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
    secret_key: str = Field(repr=False)
    rate_limit_requests: int = Field(default=100)
    rate_limit_window_minutes: int = Field(default=1)


@runtime_checkable
class ConfigurationProvider(Protocol):
    """Protocol for configuration providers."""

    def get(
        self, key: ConfigKey, default: str | int | float | bool | None = None
    ) -> str | int | float | bool | None:
        """Get configuration value."""
        ...

    def set(self, key: ConfigKey, value: str | int | float | bool) -> None:
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

    def get_metrics(
        self, name_pattern: str | None = None, labels: dict[str, str] | None = None
    ) -> list[MetricPoint]:
        """Get metrics matching criteria."""
        ...

    def clear_metrics(self, older_than: datetime | None = None) -> None:
        """Clear old metrics."""
        ...


@runtime_checkable
class HealthChecker(Protocol):
    """Protocol for health checking."""

    async def check_health(self) -> HealthStatus:
        """Perform health check."""
        ...

    def get_dependencies(self) -> dict[str, "HealthChecker"]:
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
    def load(cls, path: Path) -> "MLModel[T]":
        """Load model from disk."""
        ...


class PromptImproverError(Exception):
    """Base exception for prompt improver."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, str | int | float | bool] | None = None,
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(message)


class ValidationError(PromptImproverError):
    """Validation error with field details."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: str | int | float | bool | None = None,
    ):
        self.field = field
        self.value = value
        details = {"field": field} if field else {}
        if value is not None:
            details["value"] = str(value)
        super().__init__(message, "VALIDATION_ERROR", details)


class ConfigurationError(PromptImproverError):
    """Configuration error."""

    def __init__(self, message: str, config_key: str | None = None):
        self.config_key = config_key
        super().__init__(message, "CONFIGURATION_ERROR", {"config_key": config_key})


class MLModelError(PromptImproverError):
    """ML model error."""

    def __init__(
        self, message: str, model_id: str | None = None, operation: str | None = None
    ):
        self.model_id = model_id
        self.operation = operation
        super().__init__(
            message, "ML_MODEL_ERROR", {"model_id": model_id, "operation": operation}
        )


def validate_prompt_text(text: str) -> PromptText:
    """Validate and convert text to PromptText type."""
    if not isinstance(text, str):
        raise ValidationError("Prompt text must be a string", "prompt_text", text)
    text = text.strip()
    if not text:
        raise ValidationError("Prompt text cannot be empty", "prompt_text", text)
    if len(text) > 10000:
        raise ValidationError(
            "Prompt text too long (max 10000 chars)", "prompt_text", text
        )
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
        raise ValidationError(
            "Percentage must be between 0.0 and 100.0", "percentage", percentage
        )
    return float(percentage)


__all__ = [
    "SQLMODEL_AVAILABLE",
    "AccessPattern",
    "BaseConfig",
    "CacheBackend",
    "CacheLevel",
    "CacheType",
    "ConfigKey",
    "ConfigurationError",
    "ConfigurationProvider",
    "ConfigurationSchema",
    "CoordinationAction",
    "DatabaseConfig",
    "EnvironmentType",
    "HealthCheckConfig",
    "HealthChecker",
    "HealthStatus",
    "InvalidationStrategy",
    "InvalidationType",
    "LogLevel",
    "MLFeatures",
    "MLLabels",
    "MLModel",
    "MLModelConfig",
    "MLModelError",
    "MLPredictions",
    "MetricName",
    "MetricPoint",
    "MetricType",
    "MetricsCollector",
    "ModelId",
    "ModelName",
    "ModelType",
    "OptimizationStatus",
    "Percentage",
    "PerformanceMetrics",
    "PerformanceMonitor",
    "PositiveTimeout",
    "Priority",
    "PromptImproverError",
    "PromptRequest",
    "PromptResponse",
    "PromptText",
    "RedisConfig",
    "RetryCount",
    "Score",
    "SecurityContext",
    "ServiceStatus",
    "SessionId",
    "TimestampedModel",
    "UserId",
    "ValidationError",
    "Version",
    "validate_percentage",
    "validate_prompt_text",
    "validate_score",
]
