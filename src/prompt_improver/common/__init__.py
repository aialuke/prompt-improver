"""Common utilities and domain-specific components for the prompt-improver system.

This package contains domain-specific utilities, constants, types, and exceptions
that are used across multiple modules. Cross-layer infrastructure contracts
have been moved to prompt_improver.shared.
"""

from prompt_improver.common.config import *
from prompt_improver.common.constants import *
from prompt_improver.common.datetime_utils import *
from prompt_improver.common.exceptions import *
from prompt_improver.common.protocols import *  # For backward compatibility
from prompt_improver.common.types import *

# Import shared types for backward compatibility
from prompt_improver.shared.types import (
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
    ValidationError as SharedValidationError,
)

__all__ = [
    # Constants
    "CACHE_TTL",
    "CONNECTION_TIMEOUT",
    "DB_POOL_SIZE",
    "DEFAULT_TIMEOUT",
    "HEALTH_CHECK_TIMEOUT",
    "MAX_RETRIES",
    "REQUEST_TIMEOUT",
    # Configuration classes
    "BaseConfig",
    "DatabaseConfig",
    "MLConfig",
    "MonitoringConfig",
    "RedisConfig",
    "SecurityConfig",
    # Domain-specific types
    "FeatureVector",
    "ModelConfig",
    # Shared types (for backward compatibility)
    "APIRequest",
    "APIResponse",
    "AsyncCallback",
    "AuthToken",
    "CacheEntry",
    "ConfigDict",
    "ConnectionParams",
    "ErrorContext",
    "EventHandler",
    "HeadersDict",
    "HealthCheckResult",
    "HealthStatus",
    "MetricPoint",
    "MetricsDict",
    "QueryParams",
    "RedisConnectionParams",
    "SecurityContext",
    "SyncCallback",
    # Exceptions
    "AuthenticationError",
    "AuthorizationError",
    "CacheError",
    "ConfigurationError",
    "ConnectionError",
    "DataError",
    "DatabaseError",
    "MLError",
    "PromptImproverError",
    "RateLimitError",
    "ResourceError",
    "SecurityError",
    "ServiceUnavailableError",
    "TimeoutError",
    "ValidationError",
    # Protocols (for backward compatibility)
    "CacheProtocol",
    "ConfigManagerProtocol",
    "ConnectionManagerProtocol",
    "DatabaseManagerProtocol",
    "EventBusProtocol",
    "HealthCheckerProtocol",
    "LoggerProtocol",
    "MetricsCollectorProtocol",
    "MLModelProtocol",
    "RepositoryProtocol",
    "RetryManagerProtocol",
    "SecurityManagerProtocol",
    "ServiceProtocol",
    "WorkflowProtocol",
    # Utility functions
    "format_compact_timestamp",
    "format_date_only",
    "format_display_date",
    "format_duration",
    "format_log_timestamp",
    "format_time_only",
    "format_timestamp",
    "format_utc_timestamp",
    "merge_configs",
]
