"""Retry Protocols - 2025 Best Practice for Circular Import Prevention

This module defines Protocol interfaces for retry functionality using Python's
typing.Protocol feature (Python 3.8+). This follows the Dependency Inversion
Principle and eliminates circular imports by defining contracts without implementations.

Key Benefits:
- Eliminates circular imports through interface segregation
- Follows SOLID principles (especially Dependency Inversion)
- Type-safe without runtime dependencies
- Compatible with structural typing (duck typing)
- Zero runtime overhead
"""

from collections.abc import Callable
from enum import Enum
from typing import (
    Any,
    List,
    Optional,
    ParamSpec,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

P = ParamSpec("P")
T = TypeVar("T")


class RetryStrategy(Enum):
    """Retry strategy enumeration - shared across all implementations"""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    CUSTOM = "custom"


class RetryableErrorType(Enum):
    """Types of retryable errors - shared across all implementations"""

    TRANSIENT = "transient"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"


@runtime_checkable
class RetryConfigProtocol(Protocol):
    """Protocol for retry configuration objects.

    This protocol defines the interface that any retry configuration must implement.
    It eliminates circular imports by defining the contract without implementation.
    """

    max_attempts: int
    strategy: RetryStrategy
    base_delay: float
    max_delay: float
    jitter: bool
    jitter_factor: float
    backoff_multiplier: float
    retry_on_exceptions: list[type]
    retry_condition: Callable[[Exception], bool] | None
    operation_timeout: float | None
    total_timeout: float | None
    log_attempts: bool
    log_level: str
    track_metrics: bool

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number"""
        ...

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an operation should be retried"""
        ...


@runtime_checkable
class RetryManagerProtocol(Protocol):
    """Protocol for retry manager implementations.

    This protocol defines the interface for retry managers without creating
    circular dependencies between ML orchestration and performance monitoring.
    """

    async def execute_with_retry(
        self,
        operation: Callable[..., Any],
        config: RetryConfigProtocol,
        *args,
        **kwargs,
    ) -> Any:
        """Execute an operation with retry logic"""
        ...

    async def execute_with_circuit_breaker(
        self,
        operation: Callable[..., Any],
        config: RetryConfigProtocol,
        *args,
        **kwargs,
    ) -> Any:
        """Execute an operation with circuit breaker pattern"""
        ...

    def get_retry_stats(self, operation_name: str) -> dict[str, Any]:
        """Get retry statistics for an operation"""
        ...


@runtime_checkable
class MetricsRegistryProtocol(Protocol):
    """Protocol for metrics registry implementations.

    This eliminates the circular import between retry managers and metrics registry.
    """

    def increment_counter(self, name: str, labels: dict | None = None) -> None:
        """Increment a counter metric"""
        ...

    def record_histogram(
        self, name: str, value: float, labels: dict | None = None
    ) -> None:
        """Record a histogram value"""
        ...

    def set_gauge(self, name: str, value: float, labels: dict | None = None) -> None:
        """Set a gauge value"""
        ...

    def get_metric(self, name: str) -> Any | None:
        """Get a metric by name"""
        ...


@runtime_checkable
class BackgroundTaskProtocol(Protocol):
    """Protocol for background task implementations.

    This eliminates circular imports between background managers and retry systems.
    """

    task_id: str
    priority: int
    retry_config: RetryConfigProtocol

    async def execute(self) -> Any:
        """Execute the background task"""
        ...

    def should_retry_on_error(self, error: Exception) -> bool:
        """Determine if task should retry on specific error"""
        ...


@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    """Protocol for circuit breaker implementations.

    This provides a clean interface for circuit breaker functionality
    without creating dependencies on specific implementations.
    """

    async def call(
        self, operation: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Execute operation through circuit breaker"""
        ...

    def is_open(self) -> bool:
        """Check if circuit breaker is open"""
        ...

    def get_state(self) -> str:
        """Get current circuit breaker state"""
        ...

    def reset(self) -> None:
        """Reset circuit breaker to closed state"""
        ...


RetryableOperation = Callable[..., Any]
RetryCallback = Callable[[int, Exception], None]
RetryPredicate = Callable[[Exception], bool]
AnyRetryConfig = Union[RetryConfigProtocol, dict[str, Any]]
AnyMetricsRegistry = Union[MetricsRegistryProtocol, None]


@runtime_checkable
class RetryDecoratorProtocol(Protocol):
    """Protocol for retry decorators.

    This allows different retry decorator implementations to be used
    interchangeably without circular dependencies.
    """

    def __call__(
        self,
        config: RetryConfigProtocol | None = None,
        operation_name: str | None = None,
        metrics_registry: MetricsRegistryProtocol | None = None,
    ) -> Callable[[Callable], Callable]:
        """Decorator that adds retry functionality to a function"""
        ...


@runtime_checkable
class RetryConfigFactoryProtocol(Protocol):
    """Protocol for retry configuration factories.

    This allows different configuration creation strategies without
    coupling to specific implementations.
    """

    def create_config(
        self,
        operation_type: str = "medium",
        max_attempts: int | None = None,
        base_delay: float | None = None,
        **kwargs,
    ) -> RetryConfigProtocol:
        """Create a retry configuration with sensible defaults"""
        ...

    def get_standard_config(self, config_name: str) -> RetryConfigProtocol:
        """Get a predefined standard configuration"""
        ...


@runtime_checkable
class RetryObserverProtocol(Protocol):
    """Protocol for observing retry events"""

    def on_retry_attempt(
        self, operation_name: str, attempt: int, delay: float, error: Exception
    ) -> None:
        """Called when a retry attempt is made"""
        ...

    def on_retry_success(self, operation_name: str, total_attempts: int) -> None:
        """Called when retry succeeds"""
        ...

    def on_retry_failure(
        self, operation_name: str, total_attempts: int, final_error: Exception
    ) -> None:
        """Called when all retry attempts fail"""
        ...


__all__ = [
    "AnyMetricsRegistry",
    "AnyRetryConfig",
    "BackgroundTaskProtocol",
    "CircuitBreakerProtocol",
    "MetricsRegistryProtocol",
    "RetryCallback",
    "RetryConfigFactoryProtocol",
    "RetryConfigProtocol",
    "RetryDecoratorProtocol",
    "RetryManagerProtocol",
    "RetryObserverProtocol",
    "RetryPredicate",
    "RetryStrategy",
    "RetryableErrorType",
    "RetryableOperation",
]
