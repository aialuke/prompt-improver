"""Protocol Interfaces Package

This package contains Protocol interfaces that define contracts for various
system components. Using protocols eliminates circular imports by providing
interface definitions without implementation dependencies.

2025 Best Practice: Use typing.Protocol for dependency inversion.
"""

from .retry_protocols import (
    RetryStrategy,
    RetryableErrorType,
    RetryConfigProtocol,
    RetryManagerProtocol,
    MetricsRegistryProtocol,
    BackgroundTaskProtocol,
    CircuitBreakerProtocol,
    RetryDecoratorProtocol,
    RetryConfigFactoryProtocol,
    RetryObserverProtocol,
    RetryableOperation,
    RetryCallback,
    RetryPredicate,
    AnyRetryConfig,
    AnyMetricsRegistry,
)

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
