"""Protocol Interfaces for ML Pipeline Dependency Injection (2025).

Defines protocol interfaces for all major dependencies following modern Python
architecture patterns and the dependency inversion principle.
"""

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    AsyncContextManager,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    runtime_checkable,
)


class ServiceStatus(Enum):
    """Service health status enumeration."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    INITIALIZING = "initializing"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ComponentSpec:
    """Specification for a component that can be loaded."""

    name: str
    module_path: str
    class_name: str
    tier: str
    dependencies: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    enabled: bool = True


@runtime_checkable
class MLflowServiceProtocol(Protocol):
    """Protocol for MLflow service interactions."""

    @abstractmethod
    async def log_experiment(
        self, experiment_name: str, parameters: dict[str, Any]
    ) -> str:
        """Log an ML experiment and return run ID."""
        ...

    @abstractmethod
    async def log_model(
        self, model_name: str, model_data: Any, metadata: dict[str, Any]
    ) -> str:
        """Log a model and return model URI."""
        ...

    @abstractmethod
    async def get_model_metadata(self, model_id: str) -> dict[str, Any]:
        """Retrieve model metadata by ID."""
        ...

    @abstractmethod
    async def start_trace(
        self, trace_name: str, attributes: dict[str, Any] | None = None
    ) -> str:
        """Start MLflow tracing and return trace ID."""
        ...

    @abstractmethod
    async def end_trace(
        self, trace_id: str, outputs: dict[str, Any] | None = None
    ) -> None:
        """End MLflow trace with outputs."""
        ...


@runtime_checkable
class CacheServiceProtocol(Protocol):
    """Protocol for cache service (Redis) interactions."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value by key."""
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set key-value with optional TTL."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key and return success status."""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...

    @abstractmethod
    async def health_check(self) -> ServiceStatus:
        """Check cache service health."""
        ...


@runtime_checkable
class DatabaseServiceProtocol(Protocol):
    """Protocol for database service interactions."""

    @abstractmethod
    async def execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute query and return results."""
        ...

    @abstractmethod
    async def execute_transaction(
        self, queries: list[str], parameters: list[dict[str, Any]] | None = None
    ) -> None:
        """Execute multiple queries in transaction."""
        ...

    @abstractmethod
    async def health_check(self) -> ServiceStatus:
        """Check database service health."""
        ...

    @abstractmethod
    async def get_connection_pool_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
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


@runtime_checkable
class WorkflowEngineProtocol(Protocol):
    """Protocol for workflow execution engine."""

    @abstractmethod
    async def execute_workflow(
        self, workflow_id: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute workflow and return results."""
        ...

    @abstractmethod
    async def get_workflow_status(self, workflow_id: str) -> str:
        """Get workflow execution status."""
        ...

    @abstractmethod
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel running workflow."""
        ...


@runtime_checkable
class ResourceManagerProtocol(Protocol):
    """Protocol for resource management."""

    @abstractmethod
    async def allocate_resources(self, resource_spec: dict[str, Any]) -> str:
        """Allocate resources and return allocation ID."""
        ...

    @abstractmethod
    async def release_resources(self, allocation_id: str) -> None:
        """Release allocated resources."""
        ...

    @abstractmethod
    async def get_resource_usage(self) -> dict[str, Any]:
        """Get current resource usage statistics."""
        ...


@runtime_checkable
class HealthMonitorProtocol(Protocol):
    """Protocol for health monitoring service."""

    @abstractmethod
    async def check_service_health(self, service_name: str) -> ServiceStatus:
        """Check individual service health."""
        ...

    @abstractmethod
    async def get_overall_health(self) -> dict[str, ServiceStatus]:
        """Get overall system health status."""
        ...

    @abstractmethod
    async def register_health_check(
        self, service_name: str, health_check_func: Any
    ) -> None:
        """Register custom health check for service."""
        ...


@runtime_checkable
class MLPipelineFactoryProtocol(Protocol):
    """Protocol for ML pipeline factory."""

    @abstractmethod
    async def create_orchestrator(self, config: dict[str, Any] | None = None) -> Any:
        """Create ML pipeline orchestrator with dependencies."""
        ...

    @abstractmethod
    async def create_component_loader(
        self, config: dict[str, Any] | None = None
    ) -> Any:
        """Create component loader."""
        ...

    @abstractmethod
    async def create_workflow_engine(
        self, config: dict[str, Any] | None = None
    ) -> WorkflowEngineProtocol:
        """Create workflow execution engine."""
        ...


@runtime_checkable
class ServiceContainerProtocol(Protocol):
    """Protocol for dependency injection container."""

    @abstractmethod
    async def register_service(self, service_name: str, service_instance: Any) -> None:
        """Register service instance."""
        ...

    @abstractmethod
    async def get_service(self, service_name: str) -> Any:
        """Get service by name."""
        ...

    @abstractmethod
    async def initialize_all_services(self) -> None:
        """Initialize all registered services."""
        ...

    @abstractmethod
    async def shutdown_all_services(self) -> None:
        """Shutdown all services gracefully."""
        ...


@runtime_checkable
class ComponentRegistryProtocol(Protocol):
    """Protocol for component registry service."""

    @abstractmethod
    async def discover_components(self, tier: str | None = None) -> list[ComponentSpec]:
        """Discover available components, optionally filtered by tier."""
        ...

    @abstractmethod
    async def register_component(self, spec: ComponentSpec) -> None:
        """Register a component specification."""
        ...

    @abstractmethod
    async def get_component_spec(self, component_name: str) -> ComponentSpec | None:
        """Get component specification by name."""
        ...

    @abstractmethod
    async def list_components_by_tier(self, tier: str) -> list[ComponentSpec]:
        """List all components for a specific tier."""
        ...


@runtime_checkable
class ComponentFactoryProtocol(Protocol):
    """Protocol for component factory service."""

    @abstractmethod
    async def create_component(
        self, spec: ComponentSpec, dependencies: dict[str, Any] | None = None
    ) -> Any:
        """Create a component instance from specification with dependency injection."""
        ...

    @abstractmethod
    async def get_component_class(self, spec: ComponentSpec) -> type:
        """Get component class from specification."""
        ...

    @abstractmethod
    async def validate_dependencies(
        self, spec: ComponentSpec, dependencies: dict[str, Any]
    ) -> bool:
        """Validate that all required dependencies are provided."""
        ...


@runtime_checkable
class ComponentLoaderProtocol(Protocol):
    """Protocol for component loading and management."""

    @abstractmethod
    async def load_component(self, component_name: str, tier: str | None = None) -> Any:
        """Load a component by name, optionally filtered by tier."""
        ...

    @abstractmethod
    async def load_components_by_tier(self, tier: str) -> dict[str, Any]:
        """Load all components for a specific tier."""
        ...

    @abstractmethod
    async def get_available_components(self, tier: str | None = None) -> list[str]:
        """Get list of available component names."""
        ...


@runtime_checkable
class ComponentInvokerProtocol(Protocol):
    """Protocol for component invocation and execution."""

    @abstractmethod
    async def invoke_component(
        self, component_name: str, method_name: str, *args, **kwargs
    ) -> Any:
        """Invoke a method on a component by name."""
        ...

    @abstractmethod
    async def invoke_components_batch(
        self, invocations: list[dict[str, Any]]
    ) -> list[Any]:
        """Invoke multiple component methods in batch."""
        ...

    @abstractmethod
    async def get_component_methods(self, component_name: str) -> list[str]:
        """Get available methods for a component."""
        ...


@runtime_checkable
class ExternalServicesConfigProtocol(Protocol):
    """Protocol for external services configuration."""

    @abstractmethod
    def get_mlflow_config(self) -> dict[str, Any]:
        """Get MLflow service configuration."""
        ...

    @abstractmethod
    def get_redis_config(self) -> dict[str, Any]:
        """Get Redis service configuration."""
        ...

    @abstractmethod
    def get_database_config(self) -> dict[str, Any]:
        """Get database service configuration."""
        ...


@dataclass
class ServiceConnectionInfo:
    """Information about service connections."""

    service_name: str
    connection_status: ServiceStatus
    connection_details: dict[str, Any] | None = None
    last_check: str | None = None
    error_message: str | None = None
