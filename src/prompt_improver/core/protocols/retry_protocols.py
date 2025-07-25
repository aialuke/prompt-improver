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

from typing import Protocol, Any, Callable, List, Optional, Union
from enum import Enum


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


class RetryConfigProtocol(Protocol):
    """
    Protocol for retry configuration objects.

    This protocol defines the interface that any retry configuration must implement.
    It eliminates circular imports by defining the contract without implementation.
    """

    # Basic retry settings
    max_attempts: int
    strategy: RetryStrategy

    # Delay settings
    base_delay: float  # seconds
    max_delay: float   # seconds

    # Advanced settings
    jitter: bool
    jitter_factor: float
    backoff_multiplier: float

    # Conditional retry settings
    retry_on_exceptions: List[type]
    retry_condition: Optional[Callable[[Exception], bool]]

    # Timeout settings
    operation_timeout: Optional[float]
    total_timeout: Optional[float]

    # Logging and monitoring
    log_attempts: bool
    log_level: str
    track_metrics: bool

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number"""
        ...

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an operation should be retried"""
        ...


class RetryManagerProtocol(Protocol):
    """
    Protocol for retry manager implementations.

    This protocol defines the interface for retry managers without creating
    circular dependencies between ML orchestration and performance monitoring.
    """

    async def execute_with_retry(
        self,
        operation: Callable[..., Any],
        config: RetryConfigProtocol,
        *args,
        **kwargs
    ) -> Any:
        """Execute an operation with retry logic"""
        ...

    async def execute_with_circuit_breaker(
        self,
        operation: Callable[..., Any],
        config: RetryConfigProtocol,
        *args,
        **kwargs
    ) -> Any:
        """Execute an operation with circuit breaker pattern"""
        ...

    def get_retry_stats(self, operation_name: str) -> dict[str, Any]:
        """Get retry statistics for an operation"""
        ...


class MetricsRegistryProtocol(Protocol):
    """
    Protocol for metrics registry implementations.

    This eliminates the circular import between retry managers and metrics registry.
    """

    def increment_counter(self, name: str, labels: Optional[dict] = None) -> None:
        """Increment a counter metric"""
        ...

    def record_histogram(self, name: str, value: float, labels: Optional[dict] = None) -> None:
        """Record a histogram value"""
        ...

    def set_gauge(self, name: str, value: float, labels: Optional[dict] = None) -> None:
        """Set a gauge value"""
        ...

    def get_metric(self, name: str) -> Optional[Any]:
        """Get a metric by name"""
        ...


class BackgroundTaskProtocol(Protocol):
    """
    Protocol for background task implementations.

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


class CircuitBreakerProtocol(Protocol):
    """
    Protocol for circuit breaker implementations.

    This provides a clean interface for circuit breaker functionality
    without creating dependencies on specific implementations.
    """

    async def call(self, operation: Callable[..., Any], *args, **kwargs) -> Any:
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


# Type aliases for common protocol combinations
RetryableOperation = Callable[..., Any]
RetryCallback = Callable[[int, Exception], None]
RetryPredicate = Callable[[Exception], bool]

# Union types for flexibility
AnyRetryConfig = Union[RetryConfigProtocol, dict[str, Any]]
AnyMetricsRegistry = Union[MetricsRegistryProtocol, None]


class RetryDecoratorProtocol(Protocol):
    """
    Protocol for retry decorators.

    This allows different retry decorator implementations to be used
    interchangeably without circular dependencies.
    """

    def __call__(
        self,
        config: Optional[RetryConfigProtocol] = None,
        operation_name: Optional[str] = None,
        metrics_registry: Optional[MetricsRegistryProtocol] = None
    ) -> Callable[[Callable], Callable]:
        """Decorator that adds retry functionality to a function"""
        ...


# Factory protocol for creating retry configurations
class RetryConfigFactoryProtocol(Protocol):
    """
    Protocol for retry configuration factories.

    This allows different configuration creation strategies without
    coupling to specific implementations.
    """

    def create_config(
        self,
        operation_type: str = "medium",
        max_attempts: Optional[int] = None,
        base_delay: Optional[float] = None,
        **kwargs
    ) -> RetryConfigProtocol:
        """Create a retry configuration with sensible defaults"""
        ...

    def get_standard_config(self, config_name: str) -> RetryConfigProtocol:
        """Get a predefined standard configuration"""
        ...


# Utility protocols for advanced features
class RetryObserverProtocol(Protocol):
    """Protocol for observing retry events"""

    def on_retry_attempt(
        self,
        operation_name: str,
        attempt: int,
        delay: float,
        error: Exception
    ) -> None:
        """Called when a retry attempt is made"""
        ...

    def on_retry_success(self, operation_name: str, total_attempts: int) -> None:
        """Called when retry succeeds"""
        ...

    def on_retry_failure(
        self,
        operation_name: str,
        total_attempts: int,
        final_error: Exception
    ) -> None:
        """Called when all retry attempts fail"""
        ...


# Export all protocols for easy importing
__all__ = [
    "RetryStrategy",
    "RetryableErrorType",
    "RetryConfigProtocol",
    "RetryManagerProtocol",
    "MetricsRegistryProtocol",
    "BackgroundTaskProtocol",
    "CircuitBreakerProtocol",
    "RetryDecoratorProtocol",
    "RetryConfigFactoryProtocol",
    "RetryObserverProtocol",
    "RetryableOperation",
    "RetryCallback",
    "RetryPredicate",
    "AnyRetryConfig",
    "AnyMetricsRegistry",
]
