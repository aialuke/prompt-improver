"""Core shared types for cross-layer infrastructure contracts.

This module contains fundamental types used across multiple layers
of the application for infrastructure concerns like connections,
health checks, metrics, and API communication.
"""

from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, Union

from pydantic import BaseModel, Field

# Type aliases for common data structures
ConfigDict = dict[str, Any]
MetricsDict = dict[str, int | float | str]
HeadersDict = dict[str, str]
QueryParams = dict[str, str | int | float | bool]

# Callback types
AsyncCallback = Callable[..., Awaitable[Any]]
SyncCallback = Callable[..., Any]
EventHandler = Union[AsyncCallback, SyncCallback]


class ConnectionParams(BaseModel):
    """Database connection parameters."""

    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: str = Field(default="prefer")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)


class RedisConnectionParams(BaseModel):
    """Redis connection parameters."""

    host: str
    port: int
    db: int = Field(default=0)
    password: str | None = Field(default=None)
    ssl: bool = Field(default=False)
    max_connections: int = Field(default=10)


class HealthStatus(Enum):
    """Health check status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class HealthCheckResult(BaseModel):
    """Health check result."""

    status: HealthStatus
    message: str
    details: ConfigDict
    timestamp: str
    response_time_ms: float


class MetricPoint(BaseModel):
    """Single metric data point."""

    name: str
    value: int | float
    tags: dict[str, str]
    timestamp: str
    unit: str | None = Field(default=None)


class APIRequest(BaseModel):
    """Generic API request structure."""

    method: str
    path: str
    headers: HeadersDict
    params: QueryParams
    body: Any | None = Field(default=None)


class APIResponse(BaseModel):
    """Generic API response structure."""

    status_code: int
    headers: HeadersDict
    body: Any
    response_time_ms: float


class CacheEntry(BaseModel):
    """Cache entry with metadata."""

    key: str
    value: Any
    ttl: int
    created_at: str
    accessed_at: str
    hit_count: int = Field(default=0)


class AuthToken(BaseModel):
    """Authentication token."""

    token: str
    expires_at: str
    user_id: str
    permissions: list[str]


class SecurityContext(BaseModel):
    """Security context for requests."""

    user_id: str | None = Field(default=None)
    permissions: list[str] | None = Field(default=None)
    token: AuthToken | None = Field(default=None)
    ip_address: str | None = Field(default=None)


class ValidationError(BaseModel):
    """Validation error details."""

    field: str
    message: str
    code: str
    value: Any


class ErrorContext(BaseModel):
    """Error context information."""

    error_id: str
    timestamp: str
    request_id: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    additional_data: ConfigDict | None = Field(default=None)
