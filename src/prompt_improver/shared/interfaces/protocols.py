"""Core protocol definitions for cross-layer service contracts.

This module contains protocol interfaces that define contracts
for various system components across different layers.
"""

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from prompt_improver.shared.types import (
    ConfigDict,
    ConnectionParams,
    HealthCheckResult,
    MetricPoint,
    SecurityContext,
)


@runtime_checkable
class ConnectionManagerProtocol(Protocol):
    """Protocol for connection managers."""

    @abstractmethod
    async def get_connection(self) -> Any:
        """Get a connection from the pool."""
        ...

    @abstractmethod
    async def close_connection(self, connection: Any) -> None:
        """Return a connection to the pool."""
        ...

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """Perform a health check on the connection pool."""
        ...


@runtime_checkable
class RetryManagerProtocol(Protocol):
    """Protocol for retry managers."""

    @abstractmethod
    async def retry_operation(
        self,
        operation: Any,
        max_retries: int | None = None,
        backoff_factor: float | None = None,
    ) -> Any:
        """Retry an operation with exponential backoff."""
        ...

    @abstractmethod
    def get_retry_stats(self) -> dict[str, Any]:
        """Get retry statistics."""
        ...


@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for cache implementations."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get a value from cache."""
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set a value in cache."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        ...

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        ...


@runtime_checkable
class ServiceProtocol(Protocol):
    """Protocol for service implementations."""

    @abstractmethod
    async def start(self) -> None:
        """Start the service."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the service."""
        ...

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """Perform a health check."""
        ...

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """Get service status."""
        ...


@runtime_checkable
class HealthCheckerProtocol(Protocol):
    """Protocol for health checkers."""

    @abstractmethod
    async def check_health(self) -> HealthCheckResult:
        """Perform a health check."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Get the health checker name."""
        ...


@runtime_checkable
class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collectors."""

    @abstractmethod
    async def collect_metrics(self) -> list[MetricPoint]:
        """Collect metrics."""
        ...

    @abstractmethod
    def record_metric(
        self, name: str, value: float, tags: dict[str, str] | None = None
    ) -> None:
        """Record a single metric."""
        ...


@runtime_checkable
class ConfigManagerProtocol(Protocol):
    """Protocol for configuration managers."""

    @abstractmethod
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        ...

    @abstractmethod
    def set_config(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        ...

    @abstractmethod
    def get_all_config(self) -> ConfigDict:
        """Get all configuration."""
        ...

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate configuration."""
        ...


@runtime_checkable
class SecurityManagerProtocol(Protocol):
    """Protocol for security managers."""

    @abstractmethod
    async def authenticate(self, credentials: dict[str, Any]) -> SecurityContext | None:
        """Authenticate a user."""
        ...

    @abstractmethod
    async def authorize(
        self, context: SecurityContext, resource: str, action: str
    ) -> bool:
        """Authorize an action."""
        ...

    @abstractmethod
    async def validate_token(self, token: str) -> SecurityContext | None:
        """Validate an authentication token."""
        ...


@runtime_checkable
class DatabaseManagerProtocol(Protocol):
    """Protocol for database managers."""

    @abstractmethod
    async def execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Execute a database query."""
        ...

    @abstractmethod
    async def execute_transaction(self, operations: list[Any]) -> Any:
        """Execute a database transaction."""
        ...

    @abstractmethod
    async def get_connection_info(self) -> ConnectionParams:
        """Get connection information."""
        ...


@runtime_checkable
class MLModelProtocol(Protocol):
    """Protocol for ML models."""

    @abstractmethod
    async def predict(self, features: list[float]) -> Any:
        """Make a prediction."""
        ...

    @abstractmethod
    async def train(self, training_data: Any) -> None:
        """Train the model."""
        ...

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        ...


@runtime_checkable
class EventBusProtocol(Protocol):
    """Protocol for event bus implementations."""

    @abstractmethod
    async def publish(self, event_type: str, data: Any) -> None:
        """Publish an event."""
        ...

    @abstractmethod
    async def subscribe(self, event_type: str, handler: Any) -> None:
        """Subscribe to an event type."""
        ...

    @abstractmethod
    async def unsubscribe(self, event_type: str, handler: Any) -> None:
        """Unsubscribe from an event type."""
        ...


@runtime_checkable
class LoggerProtocol(Protocol):
    """Protocol for logger implementations."""

    @abstractmethod
    def debug(self, message: str, **kwargs: Mapping[str, Any]) -> None:
        """Log a debug message."""
        ...

    @abstractmethod
    def info(self, message: str, **kwargs: Mapping[str, Any]) -> None:
        """Log an info message."""
        ...

    @abstractmethod
    def warning(self, message: str, **kwargs: Mapping[str, Any]) -> None:
        """Log a warning message."""
        ...

    @abstractmethod
    def error(self, message: str, **kwargs: Mapping[str, Any]) -> None:
        """Log an error message."""
        ...

    @abstractmethod
    def critical(self, message: str, **kwargs: Mapping[str, Any]) -> None:
        """Log a critical message."""
        ...


@runtime_checkable
class WorkflowProtocol(Protocol):
    """Protocol for workflow implementations."""

    @abstractmethod
    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the workflow."""
        ...

    @abstractmethod
    def get_workflow_info(self) -> dict[str, Any]:
        """Get workflow information."""
        ...

    @abstractmethod
    async def validate(self, context: dict[str, Any]) -> bool:
        """Validate workflow context."""
        ...


@runtime_checkable
class RepositoryProtocol(Protocol):
    """Protocol for repository implementations."""

    @abstractmethod
    async def create(self, entity: Any) -> Any:
        """Create a new entity."""
        ...

    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Any | None:
        """Get an entity by ID."""
        ...

    @abstractmethod
    async def update(self, entity: Any) -> Any:
        """Update an entity."""
        ...

    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Delete an entity."""
        ...

    @abstractmethod
    async def list_all(self, filters: dict[str, Any] | None = None) -> list[Any]:
        """List all entities with optional filters."""
        ...
