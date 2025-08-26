"""Shared types and data structures."""

from prompt_improver.shared.types.core import (
    APIRequest,
    APIResponse,
    AsyncCallback,
    AuthToken,
    CacheEntry,
    ConfigDict,
    ConnectionParams,
    ErrorContext,
    EventHandler,
    HeadersDict,
    HealthCheckResult,
    HealthStatus,
    MetricPoint,
    MetricsDict,
    QueryParams,
    RedisConnectionParams,
    SecurityContext,
    SyncCallback,
    ValidationError,
)
from prompt_improver.shared.types.signals import (
    EmergencyOperation,
    OperationResult,
    ShutdownReason,
    SignalContext,
    SignalOperation,
)

__all__ = [
    # Core infrastructure types
    "APIRequest",
    "APIResponse",
    "AsyncCallback",
    "AuthToken",
    "CacheEntry",
    "ConfigDict",
    "ConnectionParams",
    # Signal handling types
    "EmergencyOperation",
    "ErrorContext",
    "EventHandler",
    "HeadersDict",
    "HealthCheckResult",
    "HealthStatus",
    "MetricPoint",
    "MetricsDict",
    "OperationResult",
    "QueryParams",
    "RedisConnectionParams",
    "SecurityContext",
    "ShutdownReason",
    "SignalContext",
    "SignalOperation",
    "SyncCallback",
    "ValidationError",
]
