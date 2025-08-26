"""Core cross-cutting protocol definitions.

Consolidated protocols for fundamental services that span multiple domains,
including generic service patterns, health checking, and system-wide interfaces.
"""

from abc import abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ParamSpec, Protocol, TypeVar, Union, runtime_checkable

# Core protocols will be migrated here from:
# - /core/protocols/facade_protocols.py
# - /core/protocols/health_protocol.py
# - /shared/interfaces/protocols.py (existing generic protocols)


@runtime_checkable
class ServiceProtocol(Protocol):
    """Base protocol for all services in the system."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the service gracefully."""
        ...


@runtime_checkable
class HealthCheckProtocol(Protocol):
    """Protocol for health checking capabilities."""

    @abstractmethod
    async def check_health(self) -> dict[str, Any]:
        """Check the health status of the service."""
        ...

    @abstractmethod
    def is_healthy(self) -> bool:
        """Return True if the service is healthy."""
        ...


@runtime_checkable
class EventBusProtocol(Protocol):
    """Protocol for event bus service."""

    @abstractmethod
    async def publish(self, event_type: str, event_data: dict[str, Any]) -> None:
        """Publish event to the bus."""
        ...

    @abstractmethod
    async def subscribe(self, event_type: str, handler: Any) -> str:
        """Subscribe to event type and return subscription ID."""
        ...

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from events."""
        ...


# Retry/Resilience Protocols - Migrated from core/protocols/retry_protocols.py

P = ParamSpec("P")
T = TypeVar("T")


class RetryStrategy(Enum):
    """Retry strategy enumeration - shared across all implementations."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    CUSTOM = "custom"


class RetryableErrorType(Enum):
    """Types of retryable errors - shared across all implementations."""

    TRANSIENT = "transient"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"


@runtime_checkable
class RetryConfigProtocol(Protocol):
    """Protocol for retry configuration objects."""

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
        """Calculate delay for a given attempt number."""
        ...

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an operation should be retried."""
        ...


@runtime_checkable
class MetricsRegistryProtocol(Protocol):
    """Protocol for metrics registry implementations."""

    def increment_counter(self, name: str, labels: dict | None = None) -> None:
        """Increment a counter metric."""
        ...

    def record_histogram(
        self, name: str, value: float, labels: dict | None = None
    ) -> None:
        """Record a histogram value."""
        ...

    def set_gauge(self, name: str, value: float, labels: dict | None = None) -> None:
        """Set a gauge value."""
        ...

    def get_metric(self, name: str) -> Any | None:
        """Get a metric by name."""
        ...


@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    """Protocol for circuit breaker implementations."""

    async def call(
        self, operation: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Execute operation through circuit breaker."""
        ...

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        ...

    def get_state(self) -> str:
        """Get current circuit breaker state."""
        ...

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        ...


@runtime_checkable
class RetryObserverProtocol(Protocol):
    """Protocol for observing retry events."""

    def on_retry_attempt(
        self, operation_name: str, attempt: int, delay: float, error: Exception
    ) -> None:
        """Called when a retry attempt is made."""
        ...

    def on_retry_success(self, operation_name: str, total_attempts: int) -> None:
        """Called when retry succeeds."""
        ...

    def on_retry_failure(
        self, operation_name: str, total_attempts: int, final_error: Exception
    ) -> None:
        """Called when all retry attempts fail."""
        ...


@runtime_checkable
class RetryManagerProtocol(Protocol):
    """Protocol for retry manager implementations."""

    @abstractmethod
    async def retry_operation(
        self,
        operation: Callable[P, T],
        config: RetryConfigProtocol | None = None,
        operation_name: str | None = None,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> T:
        """Execute operation with retry logic."""
        ...

    @abstractmethod
    def create_retry_config(
        self,
        max_attempts: int = 3,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        **kwargs: Any
    ) -> RetryConfigProtocol:
        """Create retry configuration."""
        ...

    @abstractmethod
    def add_observer(self, observer: RetryObserverProtocol) -> None:
        """Add retry observer."""
        ...

    @abstractmethod
    def remove_observer(self, observer: RetryObserverProtocol) -> None:
        """Remove retry observer."""
        ...


# Type aliases for retry functionality
AnyRetryConfig = Union[RetryConfigProtocol, dict[str, Any]]
AnyMetricsRegistry = Union[MetricsRegistryProtocol, None]

# Additional core protocols to be migrated during consolidation phase

# DI Container Protocols (migrated from core/di/protocols.py)


@runtime_checkable
class ContainerProtocol(Protocol):
    """Base protocol for all specialized DI containers."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize all services in the container."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown all services gracefully."""
        ...

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Check health of all services in container."""
        ...

    @abstractmethod
    def get_registration_info(self) -> dict[str, Any]:
        """Get information about registered services."""
        ...


@runtime_checkable
class CoreContainerProtocol(Protocol):
    """Protocol for core services container."""

    # Base container methods
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize all services in the container."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown all services gracefully."""
        ...

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Check health of all services in container."""
        ...

    @abstractmethod
    def get_registration_info(self) -> dict[str, Any]:
        """Get information about registered services."""
        ...

    # Core-specific methods
    @abstractmethod
    async def get_datetime_service(self) -> Any:
        """Get datetime service instance."""
        ...

    @abstractmethod
    async def get_metrics_registry(self) -> Any:
        """Get metrics registry instance."""
        ...


@runtime_checkable
class ContainerRegistryProtocol(Protocol):
    """Protocol for container registry services."""

    @abstractmethod
    def register_container(self, name: str, container: Any) -> None:
        """Register a container with the registry."""
        ...

    @abstractmethod
    def get_container(self, name: str) -> Any:
        """Get a container from the registry."""
        ...

    @abstractmethod
    def list_containers(self) -> list[str]:
        """List all registered container names."""
        ...


# DateTime Service Protocols - Migrated from core/protocols/datetime_protocol.py

@runtime_checkable
class DateTimeServiceProtocol(Protocol):
    """Protocol for datetime service operations."""

    def aware_utc_now(self) -> datetime:
        """Get current UTC datetime with timezone awareness."""
        ...

    def naive_utc_now(self) -> datetime:
        """Get current UTC datetime without timezone awareness."""
        ...

    def to_aware_utc(self, dt: datetime) -> datetime:
        """Convert datetime to timezone-aware UTC."""
        ...

    def to_naive_utc(self, dt: datetime) -> datetime:
        """Convert datetime to naive UTC."""
        ...

    def format_iso(self, dt: datetime) -> str:
        """Format datetime as ISO string."""
        ...

    def parse_iso(self, iso_string: str) -> datetime:
        """Parse ISO string to datetime."""
        ...


@runtime_checkable
class TimeZoneServiceProtocol(Protocol):
    """Protocol for timezone operations."""

    def get_utc_timezone(self) -> timezone:
        """Get UTC timezone object."""
        ...

    def convert_timezone(self, dt: datetime, target_tz: timezone) -> datetime:
        """Convert datetime to target timezone."""
        ...

    def is_aware(self, dt: datetime) -> bool:
        """Check if datetime is timezone-aware."""
        ...


@runtime_checkable
class DateTimeUtilsProtocol(DateTimeServiceProtocol, TimeZoneServiceProtocol, Protocol):
    """Combined protocol for all datetime utilities."""
    ...
