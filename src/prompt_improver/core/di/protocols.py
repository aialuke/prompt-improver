"""Protocol interfaces for specialized DI containers (2025).

Defines the contract for domain-specific dependency injection containers
with clean boundaries and protocol-based service registration.
"""

from abc import abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, Protocol, TypeVar

T = TypeVar("T")


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


class CoreContainerProtocol(ContainerProtocol):
    """Protocol for core services container."""

    @abstractmethod
    async def get_datetime_service(self) -> Any:
        """Get datetime service instance."""
        ...

    @abstractmethod
    async def get_metrics_registry(self) -> Any:
        """Get metrics registry instance."""
        ...

    @abstractmethod
    async def get_health_monitor(self) -> Any:
        """Get health monitor instance."""
        ...

    @abstractmethod
    async def get_config_service(self) -> Any:
        """Get configuration service instance."""
        ...


class MLContainerProtocol(ContainerProtocol):
    """Protocol for ML services container."""

    @abstractmethod
    async def get_ml_pipeline_orchestrator(self) -> Any:
        """Get ML pipeline orchestrator instance."""
        ...

    @abstractmethod
    async def get_event_bus(self) -> Any:
        """Get event bus instance."""
        ...

    @abstractmethod
    async def get_workflow_engine(self) -> Any:
        """Get workflow engine instance."""
        ...

    @abstractmethod
    async def get_resource_manager(self) -> Any:
        """Get resource manager instance."""
        ...

    @abstractmethod
    async def get_component_registry(self) -> Any:
        """Get component registry instance."""
        ...


class SecurityContainerProtocol(ContainerProtocol):
    """Protocol for security services container."""

    @abstractmethod
    async def get_authentication_service(self) -> Any:
        """Get authentication service instance."""
        ...

    @abstractmethod
    async def get_authorization_service(self) -> Any:
        """Get authorization service instance."""
        ...

    @abstractmethod
    async def get_crypto_service(self) -> Any:
        """Get cryptography service instance."""
        ...

    @abstractmethod
    async def get_validation_service(self) -> Any:
        """Get validation service instance."""
        ...


class DatabaseContainerProtocol(ContainerProtocol):
    """Protocol for database services container."""

    @abstractmethod
    async def get_connection_manager(self) -> Any:
        """Get database connection manager instance."""
        ...

    @abstractmethod
    async def get_cache_service(self) -> Any:
        """Get cache service instance."""
        ...

    @abstractmethod
    async def get_session_manager(self) -> Any:
        """Get session manager instance."""
        ...

    @abstractmethod
    async def get_repository_factory(self) -> Any:
        """Get repository factory instance."""
        ...


class ContainerRegistryProtocol(Protocol):
    """Protocol for container registration and dependency management."""

    @abstractmethod
    def register_singleton(
        self,
        interface: type[T],
        implementation_or_factory: Any,
        tags: set[str] | None = None,
    ) -> None:
        """Register a singleton service."""
        ...

    @abstractmethod
    def register_transient(
        self,
        interface: type[T],
        implementation_or_factory: Any,
        tags: set[str] | None = None,
    ) -> None:
        """Register a transient service."""
        ...

    @abstractmethod
    def register_factory(
        self,
        interface: type[T],
        factory: Any,
        tags: set[str] | None = None,
    ) -> None:
        """Register a service factory."""
        ...

    @abstractmethod
    async def get(self, interface: type[T]) -> T:
        """Resolve service instance."""
        ...

    @abstractmethod
    def is_registered(self, interface: type[T]) -> bool:
        """Check if service is registered."""
        ...


class MonitoringContainerProtocol(ContainerProtocol):
    """Protocol for monitoring services container.

    Provides access to monitoring and observability services including:
    - Metrics collection and registry (OpenTelemetry)
    - Health check systems and unified health monitoring
    - A/B testing and experimentation services
    - Performance monitoring and benchmarking
    - Alerting and notification systems
    - Distributed tracing and logging
    - Observability dashboards and visualization

    Designed to eliminate circular dependencies through protocol-based interfaces.
    """

    @abstractmethod
    async def get_metrics_registry(self) -> Any:
        """Get metrics registry instance for OpenTelemetry collection."""
        ...

    @abstractmethod
    async def get_health_monitor(self) -> Any:
        """Get health monitor instance with circuit breaker patterns."""
        ...

    @abstractmethod
    async def get_ab_testing_service(self) -> Any:
        """Get A/B testing service instance for experimentation."""
        ...

    @abstractmethod
    async def get_performance_monitor(self) -> Any:
        """Get performance monitor instance for benchmarking.

        Note: Performance monitoring may have lazy loading to avoid
        circular dependencies with MCP server components.
        """
        ...

    @abstractmethod
    async def get_alert_manager(self) -> Any:
        """Get alert manager instance for notifications."""
        ...

    @abstractmethod
    async def get_tracing_service(self) -> Any:
        """Get distributed tracing service instance."""
        ...

    @abstractmethod
    async def get_observability_dashboard(self) -> Any:
        """Get observability dashboard instance for monitoring visualization."""
        ...

    @abstractmethod
    def register_metrics_collector_factory(self, collector_type: str = "opentelemetry") -> None:
        """Register factory for metrics collectors optimized for ML workloads."""
        ...

    @abstractmethod
    def register_health_monitor_factory(self) -> None:
        """Register factory for health monitors with circuit breaker patterns."""
        ...

    @abstractmethod
    def register_ab_testing_service_factory(self) -> None:
        """Register factory for A/B testing service."""
        ...

    @abstractmethod
    def register_performance_monitoring_factory(self) -> None:
        """Register factory for performance monitoring service.

        Implementation should use lazy loading to avoid circular dependencies.
        """
        ...

    @abstractmethod
    def register_alert_manager_factory(self) -> None:
        """Register factory for alert management service."""
        ...

    @abstractmethod
    def register_tracing_service_factory(self) -> None:
        """Register factory for distributed tracing service."""
        ...

    @abstractmethod
    def register_observability_dashboard_factory(self) -> None:
        """Register factory for observability dashboard service."""
        ...


class ContainerFacadeProtocol(Protocol):
    """Protocol for unified container facade management."""

    @abstractmethod
    def get_core_container(self) -> CoreContainerProtocol:
        """Get core services container."""
        ...

    @abstractmethod
    def get_ml_container(self) -> MLContainerProtocol:
        """Get ML services container."""
        ...

    @abstractmethod
    def get_security_container(self) -> SecurityContainerProtocol:
        """Get security services container."""
        ...

    @abstractmethod
    def get_database_container(self) -> DatabaseContainerProtocol:
        """Get database services container."""
        ...

    @abstractmethod
    def get_monitoring_container(self) -> MonitoringContainerProtocol:
        """Get monitoring services container."""
        ...

    @abstractmethod
    @asynccontextmanager
    async def managed_lifecycle(self) -> AsyncContextManager["ContainerFacadeProtocol"]:
        """Managed lifecycle for all containers."""
        ...

    @abstractmethod
    async def health_check_all(self) -> dict[str, Any]:
        """Health check across all containers."""
        ...


class ServiceLifecycleProtocol(Protocol):
    """Protocol for service lifecycle management."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the service gracefully."""
        ...

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Check service health."""
        ...
