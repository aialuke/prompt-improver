"""Shared type definitions for the prompt-improver system.

This module contains common type aliases and type definitions
used across multiple modules.
"""
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from sqlmodel import Field, SQLModel
ConfigDict = dict[str, Any]
MetricsDict = dict[str, int | float | str]
HeadersDict = dict[str, str]
QueryParams = dict[str, str | int | float | bool]

class ConnectionParams(SQLModel):
    """Database connection parameters."""
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: str = Field(default='prefer')
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)

class RedisConnectionParams(SQLModel):
    """Redis connection parameters."""
    host: str
    port: int
    db: int = Field(default=0)
    password: str | None = Field(default=None)
    ssl: bool = Field(default=False)
    max_connections: int = Field(default=10)

class ModelConfig(SQLModel):
    """ML model configuration."""
    name: str
    version: str
    parameters: ConfigDict
    timeout: float = Field(default=120.0)
    batch_size: int = Field(default=32)

class FeatureVector(SQLModel):
    """Feature vector for ML processing."""
    features: list[float]
    feature_metadata: ConfigDict
    timestamp: str | None = Field(default=None)

class HealthStatus(Enum):
    """Health check status enumeration."""
    HEALTHY = 'healthy'
    UNHEALTHY = 'unhealthy'
    DEGRADED = 'degraded'
    UNKNOWN = 'unknown'

class HealthCheckResult(SQLModel):
    """Health check result."""
    status: HealthStatus
    message: str
    details: ConfigDict
    timestamp: str
    response_time_ms: float

class MetricPoint(SQLModel):
    """Single metric data point."""
    name: str
    value: int | float
    tags: dict[str, str]
    timestamp: str
    unit: str | None = Field(default=None)
AsyncCallback = Callable[..., Awaitable[Any]]
SyncCallback = Callable[..., Any]
EventHandler = Union[AsyncCallback, SyncCallback]

class APIRequest(SQLModel):
    """Generic API request structure."""
    method: str
    path: str
    headers: HeadersDict
    params: QueryParams
    body: Any | None = Field(default=None)

class APIResponse(SQLModel):
    """Generic API response structure."""
    status_code: int
    headers: HeadersDict
    body: Any
    response_time_ms: float

class CacheEntry(SQLModel):
    """Cache entry with metadata."""
    key: str
    value: Any
    ttl: int
    created_at: str
    accessed_at: str
    hit_count: int = Field(default=0)

class AuthToken(SQLModel):
    """Authentication token."""
    token: str
    expires_at: str
    user_id: str
    permissions: list[str]

class SecurityContext(SQLModel):
    """Security context for requests."""
    user_id: str | None = Field(default=None)
    permissions: list[str] | None = Field(default=None)
    token: AuthToken | None = Field(default=None)
    ip_address: str | None = Field(default=None)

class ValidationError(SQLModel):
    """Validation error details."""
    field: str
    message: str
    code: str
    value: Any

class ErrorContext(SQLModel):
    """Error context information."""
    error_id: str
    timestamp: str
    request_id: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    additional_data: ConfigDict | None = Field(default=None)
