"""Container Orchestrator - Unified DI Container Facade (2025).

This orchestrator implements the same interface as the original DIContainer
god object but delegates to specialized domain containers following clean
architecture principles and the single responsibility principle.
"""

import asyncio
import inspect
import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, TypeVar

if TYPE_CHECKING:
    from prompt_improver.core.di.cli_container import CLIContainer
    from prompt_improver.core.di.ml_container import MLServiceContainer
    from prompt_improver.core.events.ml_event_bus import MLEventBus

# Core non-ML imports - safe to import at module level
from prompt_improver.core.di.core_container import CoreContainer, get_core_container
from prompt_improver.core.di.database_container import (
    DatabaseContainer,
    get_database_container,
)

# from prompt_improver.shared.interfaces.protocols.core import ContainerFacadeProtocol, Any
from prompt_improver.core.di.security_container import (
    SecurityContainer,
    get_security_container,
)
from prompt_improver.core.interfaces.datetime_service import DateTimeServiceProtocol
from prompt_improver.core.interfaces.ml_interface import (
    MLAnalysisInterface,
    MLHealthInterface,
    MLModelInterface,
    MLServiceInterface,
    MLTrainingInterface,
)
from prompt_improver.core.services.ml_service import EventBasedMLService
from prompt_improver.shared.interfaces.ab_testing import IABTestingService
from prompt_improver.shared.interfaces.protocols.core import MetricsRegistryProtocol
from prompt_improver.shared.interfaces.protocols.ml import (
    CacheServiceProtocol,
    DatabaseServiceProtocol,
    ExternalServicesConfigProtocol,
    MLflowServiceProtocol,
    ServiceConnectionInfo,
    ServiceStatus,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


class ServiceLifetime(Enum):
    """Service lifetime management options."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"
    RESOURCE = "resource"


@dataclass
class ServiceRegistration:
    """Enhanced service registration information."""
    interface: type[Any]
    implementation: type[Any] | None
    lifetime: ServiceLifetime
    factory: Callable[[], Any] | None = None
    initializer: Callable[[Any], Any] | None = None
    finalizer: Callable[[Any], Any] | None = None
    initialized: bool = False
    instance: Any = None
    dependencies: set[type[Any]] = field(default_factory=set)
    tags: set[str] = field(default_factory=set)
    priority: int = 0
    health_check: Callable[[], Any] | None = None


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected."""

    def __init__(self, cycle: str) -> None:
        self.cycle = cycle
        super().__init__(f"Circular dependency detected: {cycle}")


class ServiceNotRegisteredError(Exception):
    """Raised when attempting to resolve unregistered service."""

    def __init__(self, service_type: type[Any]) -> None:
        self.service_type = service_type
        super().__init__(f"Service not registered: {service_type.__name__}")


class ServiceInitializationError(Exception):
    """Raised when service initialization fails."""

    def __init__(self, service_type: type[Any], original_error: Exception) -> None:
        self.service_type = service_type
        self.original_error = original_error
        super().__init__(
            f"Failed to initialize {service_type.__name__}: {original_error}"
        )


class ResourceManagementError(Exception):
    """Raised when resource management operations fail."""


class DIContainer:
    """Enhanced dependency injection container orchestrator with 2025 best practices.

    This orchestrator maintains the same interface as the original DIContainer god object
    but delegates to specialized domain containers following clean architecture principles.

    Provides comprehensive service registration, resolution, and lifecycle management
    for ML Pipeline Orchestrator systems with async support, resource management,
    and factory patterns optimized for ML workloads.

    Features:
    - Async service initialization and cleanup
    - Resource lifecycle management (connections, models, etc.)
    - Factory patterns for metrics collectors and health monitors
    - Scoped service lifetimes for request/session isolation
    - Performance monitoring and health checks
    - Delegation to specialized containers by domain
    """

    def __init__(self, logger: logging.Logger | None = None, name: str = "orchestrator") -> None:
        """Initialize the container orchestrator.

        Args:
            logger: Optional logger for debugging and monitoring
            name: Container name for identification in logs
        """
        self.name = name
        self.logger = logger or logging.getLogger(__name__)

        # Initialize specialized containers
        self._core_container = get_core_container()
        self._security_container = get_security_container()
        self._database_container = get_database_container()
        self._monitoring_container = self._create_monitoring_container()
        self._ml_container = None  # Lazy-loaded to avoid torch dependencies
        self._cli_container = None  # Lazy-loaded for CLI services

        # Service routing maps
        self._domain_routing = self._build_domain_routing()

        # Legacy interface support
        self._scoped_services: dict[str, dict[type[Any], Any]] = {}
        self._resources: set[Any] = set()
        self._resolution_stack: set[type[Any]] = set()
        self._lock = asyncio.Lock()
        self._metrics_registry: MetricsRegistryProtocol | None = None
        self._health_checks: dict[str, Callable[[], Any]] = {}
        self._resolution_times: dict[type[Any], float] = {}
        self._initialization_order: list[type[Any]] = []

        self.logger.debug(f"Container orchestrator '{self.name}' initialized")

    def _create_monitoring_container(self) -> Any:
        """Create monitoring container with lazy loading to avoid circular dependencies.

        This factory method delays the import of the concrete MonitoringContainer
        until needed, breaking the circular dependency chain.
        """
        from prompt_improver.core.di.monitoring_container import MonitoringContainer
        return MonitoringContainer()

    def _create_ml_container(self) -> Optional["MLServiceContainer"]:
        """Create ML container with lazy loading to avoid torch dependencies.

        This factory method delays the import of ML services until needed,
        breaking the torch import chain. Returns None if torch is not available.
        """
        try:
            from prompt_improver.core.di.ml_container import MLServiceContainer
            return MLServiceContainer()
        except ImportError as e:
            self.logger.info(f"ML container not available (torch not installed): {e}")
            return None
        except Exception as e:
            self.logger.warning(f"ML container initialization failed: {e}")
            return None

    def _create_cli_container(self) -> Optional["CLIContainer"]:
        """Create CLI container with lazy loading.

        This factory method delays the import of CLI container until needed,
        following the pattern established for ML container.
        """
        try:
            from prompt_improver.core.di.cli_container import CLIContainer
            return CLIContainer()
        except ImportError as e:
            self.logger.exception(f"CLI container not available: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"CLI container initialization failed: {e}")
            return None

    def _build_domain_routing(self) -> dict[type[Any], str]:
        """Build service type to container domain routing map.

        Note: ML service types are handled by name-based routing since they're
        in TYPE_CHECKING blocks to avoid torch import chain.
        CLI services follow the same pattern for consistency.
        """
        # Import CLI protocols here to avoid circular imports
        from prompt_improver.shared.interfaces.protocols.cli import (
            CLIFacadeProtocol,
            CLIOrchestratorProtocol,
            CLIServiceProtocol,
            ProgressServiceProtocol,
            SessionServiceProtocol,
            TrainingServiceProtocol,
            WorkflowServiceProtocol,
        )

        return {
            # Core services
            DateTimeServiceProtocol: "core",

            # Database services
            DatabaseServiceProtocol: "database",
            CacheServiceProtocol: "database",

            # ML services (explicit routing to prevent keyword conflicts)
            MLServiceInterface: "ml",
            MLAnalysisInterface: "ml",
            MLTrainingInterface: "ml",
            MLHealthInterface: "ml",  # Explicit routing to prevent "health" keyword conflict
            MLModelInterface: "ml",

            # Monitoring services
            MetricsRegistryProtocol: "monitoring",
            IABTestingService: "monitoring",

            # CLI services
            CLIFacadeProtocol: "cli",
            CLIServiceProtocol: "cli",
            CLIOrchestratorProtocol: "cli",
            WorkflowServiceProtocol: "cli",
            ProgressServiceProtocol: "cli",
            SessionServiceProtocol: "cli",
            TrainingServiceProtocol: "cli",

            # External services configuration
            ExternalServicesConfigProtocol: "core",

            # String-based services (legacy support)
            str: "core",  # Default for string-based service names
        }

    def _get_container_for_service(self, interface: type[Any]) -> Any:
        """Get the appropriate specialized container for a service interface."""
        # Check direct mapping first
        domain = self._domain_routing.get(interface)
        if domain:
            return self._get_container_by_domain(domain)

        # Check by service name pattern
        service_name = interface.__name__ if hasattr(interface, "__name__") else str(interface)

        if any(keyword in service_name.lower() for keyword in ["auth", "security", "crypto", "validation"]):
            return self._security_container
        if any(keyword in service_name.lower() for keyword in ["database", "cache", "repository", "session"]):
            return self._database_container
        if any(keyword in service_name.lower() for keyword in ["metric", "health", "monitor", "alert", "trace"]):
            return self._monitoring_container
        if any(keyword in service_name.lower() for keyword in ["ml", "model", "training", "workflow", "pipeline"]):
            return self.get_ml_container()
        # Default to core container
        return self._core_container

    def _get_container_by_domain(self, domain: str) -> Any:
        """Get container by domain name."""
        if domain == "ml":
            return self.get_ml_container()
        if domain == "cli":
            return self.get_cli_container()

        containers = {
            "core": self._core_container,
            "security": self._security_container,
            "database": self._database_container,
            "monitoring": self._monitoring_container,
        }
        return containers.get(domain, self._core_container)

    def _map_protocol_to_service_name(self, interface: type[Any]) -> str:
        """Map protocol type to service name for ML and CLI containers.

        This method provides the translation layer between protocol-based
        service resolution and name-based container service storage.
        """
        # Check if it's already a string
        if isinstance(interface, str):
            return interface

        interface_name = interface.__name__ if hasattr(interface, "__name__") else str(interface)

        # CLI protocol mappings
        cli_protocol_mappings = {
            "CLIOrchestratorProtocol": "cli_orchestrator",
            "WorkflowServiceProtocol": "workflow_service",
            "ProgressServiceProtocol": "progress_service",
            "SessionServiceProtocol": "session_service",
            "TrainingServiceProtocol": "training_service",
            "SignalHandlerProtocol": "signal_handler",
            "BackgroundManagerProtocol": "background_manager",
            "EmergencyServiceProtocol": "emergency_service",
            "RuleValidationServiceProtocol": "rule_validation_service",
            "ProcessServiceProtocol": "process_service",
            "SystemStateReporterProtocol": "system_state_reporter",
        }

        # Check CLI mappings first
        if interface_name in cli_protocol_mappings:
            return cli_protocol_mappings[interface_name]

        # ML protocol mappings (maintain existing behavior)
        ml_protocol_mappings = {
            "MLflowServiceProtocol": "mlflow_service",
            "EventBusProtocol": "event_bus",
            "WorkflowEngineProtocol": "workflow_engine",
            "ResourceManagerProtocol": "resource_manager",
            "DatabaseServiceProtocol": "database_service",
            "CacheServiceProtocol": "cache_service",
        }

        if interface_name in ml_protocol_mappings:
            return ml_protocol_mappings[interface_name]

        # Default to the interface name
        return interface_name

    # ContainerFacadeProtocol implementation
    def get_core_container(self) -> CoreContainer:
        """Get core services container."""
        return self._core_container

    def get_ml_container(self) -> Optional["MLServiceContainer"]:
        """Get ML services container with lazy loading.

        Returns:
            MLServiceContainer if torch is available, None otherwise
        """
        if self._ml_container is None:
            self._ml_container = self._create_ml_container()
        return self._ml_container

    def get_security_container(self) -> SecurityContainer:
        """Get security services container."""
        return self._security_container

    def get_database_container(self) -> DatabaseContainer:
        """Get database services container."""
        return self._database_container

    def get_monitoring_container(self) -> Any:
        """Get monitoring services container."""
        return self._monitoring_container

    def get_cli_container(self) -> Optional["CLIContainer"]:
        """Get CLI services container with lazy loading.

        Returns:
            CLIContainer if CLI services are available, None otherwise
        """
        if self._cli_container is None:
            self._cli_container = self._create_cli_container()
        return self._cli_container

    @asynccontextmanager
    async def managed_lifecycle(self):
        """Managed lifecycle for all containers."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.shutdown()

    async def health_check_all(self) -> dict[str, Any]:
        """Health check across all containers."""
        results = {
            "orchestrator_status": "healthy",
            "orchestrator_name": self.name,
            "containers": {},
        }

        containers = {
            "core": self._core_container,
            "security": self._security_container,
            "database": self._database_container,
            "monitoring": self._monitoring_container,
            "ml": self._ml_container,
            "cli": self._cli_container,
        }

        for name, container in containers.items():
            try:
                if hasattr(container, "health_check"):
                    health_result = await container.health_check()
                    results["containers"][name] = health_result
                else:
                    results["containers"][name] = {"status": "no_health_check"}
            except Exception as e:
                results["containers"][name] = {"status": "unhealthy", "error": str(e)}
                results["orchestrator_status"] = "degraded"

        return results

    # Unified container interface methods
    def register_singleton(
        self,
        interface: type[T],
        implementation: type[T],
        tags: set[str] | None = None,
        initializer: Callable[[Any], Any] | None = None,
        finalizer: Callable[[Any], Any] | None = None,
        health_check: Callable[[], Any] | None = None,
    ) -> None:
        """Register a service as singleton."""
        container = self._get_container_for_service(interface)
        if hasattr(container, "register_singleton"):
            container.register_singleton(interface, implementation, tags)
        else:
            # Fallback for ML container
            container.register_service(interface.__name__, implementation())

    def register_transient(
        self, interface: type[T], implementation: type[T], tags: set[str] | None = None
    ) -> None:
        """Register a service as transient."""
        container = self._get_container_for_service(interface)
        if hasattr(container, "register_transient"):
            container.register_transient(interface, implementation, tags)
        else:
            # Fallback for ML container
            container.register_factory(interface.__name__, implementation, singleton=False)

    def register_factory(
        self,
        interface: type[T],
        factory: Callable[[], T],
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
        tags: set[str] | None = None,
        initializer: Callable[[Any], Any] | None = None,
        finalizer: Callable[[Any], Any] | None = None,
    ) -> None:
        """Register a service with custom factory function."""
        container = self._get_container_for_service(interface)
        if hasattr(container, "register_factory"):
            container.register_factory(interface, factory, tags)
        else:
            # Fallback for ML container
            container.register_factory(interface.__name__, factory, singleton=(lifetime == ServiceLifetime.SINGLETON))

    def register_instance(
        self,
        interface: type[T],
        instance: T,
        tags: set[str] | None = None,
        finalizer: Callable[[Any], Any] | None = None,
        health_check: Callable[[], Any] | None = None,
    ) -> None:
        """Register a pre-created service instance."""
        container = self._get_container_for_service(interface)
        if hasattr(container, "register_instance"):
            container.register_instance(interface, instance, tags)
        else:
            # Fallback for ML container
            container.register_service(interface.__name__, instance)

    async def get(self, interface: type[T], scope_id: str | None = None) -> T:
        """Resolve service instance."""
        import time

        start_time = time.perf_counter()
        try:
            async with self._lock:
                result = await self._resolve_service(interface, scope_id)
                resolution_time = time.perf_counter() - start_time
                self._resolution_times[interface] = resolution_time
                if self._metrics_registry:
                    self._metrics_registry.record_histogram(
                        "di_container_resolution_time",
                        resolution_time,
                        {"service": interface.__name__, "container": self.name},
                    )
                return result
        except Exception as e:
            if self._metrics_registry:
                self._metrics_registry.increment_counter(
                    "di_container_resolution_errors",
                    {
                        "service": interface.__name__,
                        "container": self.name,
                        "error": type(e).__name__,
                    },
                )
            raise

    async def _resolve_service(
        self, interface: type[T], scope_id: str | None = None
    ) -> T:
        """Internal service resolution with enhanced lifecycle management."""
        container = self._get_container_for_service(interface)

        try:
            if hasattr(container, "get"):
                return await container.get(interface)
            if hasattr(container, "get_service"):
                # ML and CLI containers use get_service
                service_name = self._map_protocol_to_service_name(interface)
                if asyncio.iscoroutinefunction(container.get_service):
                    return await container.get_service(service_name)
                return container.get_service(service_name)
            raise ServiceNotRegisteredError(interface)
        except KeyError:
            raise ServiceNotRegisteredError(interface)

    # Container factory methods
    def register_metrics_collector_factory(
        self, collector_type: str = "opentelemetry"
    ) -> None:
        """Register factory for metrics collectors optimized for ML workloads."""
        # Delegate to monitoring container
        self._monitoring_container.register_metrics_collector_factory(collector_type)

    def register_health_monitor_factory(self) -> None:
        """Register factory for health monitors with circuit breaker patterns."""
        # Delegate to monitoring container
        self._monitoring_container.register_health_monitor_factory()

    def register_mlflow_service_factory(self, config: ServiceConnectionInfo) -> None:
        """Register factory for MLflow service with proper lifecycle management."""
        # Delegate to ML container
        self._ml_container.register_mlflow_service_factory(
            lambda: self._create_mlflow_service(config)
        )

    def register_cache_service_factory(self, config: ServiceConnectionInfo) -> None:
        """Register factory for Redis cache service with connection pooling."""
        # Delegate to database container
        self._database_container.register_cache_service_factory(config)

    def register_database_service_factory(self, config: ServiceConnectionInfo) -> None:
        """Register factory for PostgreSQL database service with connection pooling."""
        # Delegate to database container
        self._database_container.register_database_service_factory(config)

    def register_external_services_config_factory(
        self, environment: str = "production"
    ) -> None:
        """Register factory for external services configuration."""
        # Add to core container
        def create_services_config() -> ExternalServicesConfigProtocol:
            from prompt_improver.integrations.services_config import (
                ExternalServicesConfigImpl,
            )
            return ExternalServicesConfigImpl.from_environment(environment)

        self._core_container.register_factory(
            ExternalServicesConfigProtocol,
            create_services_config,
            tags={"config", "external"}
        )

    def register_event_bus_factory(self, config: Any = None) -> None:
        """Register factory for event bus with proper initialization."""
        self._ml_container.register_event_bus_factory(
            lambda: self._create_event_bus(config)
        )

    def register_workflow_engine_factory(self, config: Any = None) -> None:
        """Register factory for workflow execution engine."""
        self._ml_container.register_workflow_engine_factory(
            lambda: self._create_workflow_engine(config)
        )

    def register_resource_manager_factory(self, config: Any = None) -> None:
        """Register factory for resource manager with monitoring capabilities."""
        self._ml_container.register_resource_manager_factory(
            lambda: self._create_resource_manager(config)
        )

    def register_component_registry_factory(self, config: Any = None) -> None:
        """Register factory for ML component registry."""
        # These would need to be implemented based on the actual ML services

    def register_component_loader_factory(self) -> None:
        """Register factory for ML component loader."""

    def register_component_invoker_factory(self) -> None:
        """Register factory for ML component invoker with dependency injection."""

    def register_ml_pipeline_factory(self) -> None:
        """Register factory for complete ML pipeline orchestrator with all dependencies."""
        self._ml_container.register_ml_pipeline_factory(
            self._create_ml_pipeline_orchestrator
        )

    def register_ab_testing_service_factory(self) -> None:
        """Register factory for A/B testing service."""
        # Delegate to monitoring container
        self._monitoring_container.register_ab_testing_service_factory()

    def register_all_ml_services(self, environment: str = "production") -> None:
        """Register all ML pipeline services with proper dependency injection."""
        self.logger.info(
            f"Registering ML pipeline services for {environment} environment"
        )

        # Register external services config in core
        self.register_external_services_config_factory(environment)

        # Register database and cache services
        config_placeholder = ServiceConnectionInfo(
            service_name="placeholder",
            connection_status=ServiceStatus.HEALTHY,
            connection_details={
                "host": os.getenv("POSTGRES_HOST", "postgres"),
                "port": 5432
            }
        )
        self.register_mlflow_service_factory(config_placeholder)
        self.register_cache_service_factory(config_placeholder)
        self.register_database_service_factory(config_placeholder)

        # Register ML services
        self.register_event_bus_factory()
        self.register_component_loader_factory()
        self.register_workflow_engine_factory()
        self.register_resource_manager_factory()
        self.register_component_registry_factory()
        self.register_component_invoker_factory()

        # Register monitoring services
        self.register_health_monitor_factory()
        self.register_ab_testing_service_factory()
        self.register_ml_pipeline_factory()

        self.logger.info("ML pipeline services registration complete")

    def register_ml_interfaces(self) -> None:
        """Register ML interface factories that use event bus communication."""
        async def create_ml_event_bus():
            from prompt_improver.core.events.ml_event_bus import get_ml_event_bus
            return await get_ml_event_bus()

        async def cleanup_ml_event_bus(event_bus: "MLEventBus") -> None:
            await event_bus.shutdown()

        def get_ml_event_bus_type():
            from prompt_improver.core.events.ml_event_bus import MLEventBus
            return MLEventBus

        self._core_container.register_factory(
            get_ml_event_bus_type(),
            create_ml_event_bus,
            tags={"ml", "events", "communication"}
        )

        def create_ml_service_interface():
            return EventBasedMLService()

        # Register all ML interfaces - ensure ML container is initialized
        ml_container = self.get_ml_container()
        if ml_container is not None:
            for interface in [MLServiceInterface, MLAnalysisInterface, MLTrainingInterface,
                             MLHealthInterface, MLModelInterface]:
                ml_container.register_factory(
                    interface.__name__,
                    create_ml_service_interface,
                    singleton=True
                )

    # Helper methods for service creation
    async def _create_mlflow_service(self, config: ServiceConnectionInfo):
        from prompt_improver.integrations.mlflow_service import MLflowService
        service = MLflowService(config)
        await service.initialize()
        return service

    async def _create_event_bus(self, config: Any):
        from prompt_improver.ml.orchestration.events.adaptive_event_bus import (
            AdaptiveEventBus,
        )
        event_bus = AdaptiveEventBus(config)
        await event_bus.initialize()
        return event_bus

    async def _create_workflow_engine(self, config: Any):
        from prompt_improver.ml.orchestration.core.workflow_execution_engine import (
            WorkflowExecutionEngine,
        )
        engine = WorkflowExecutionEngine(config)
        await engine.initialize()
        return engine

    async def _create_resource_manager(self, config: Any):
        from prompt_improver.ml.orchestration.core.resource_manager import (
            ResourceManager,
        )
        manager = ResourceManager(config)
        await manager.initialize()
        return manager

    def _create_ml_pipeline_orchestrator(self, services: dict[str, Any]):
        from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import (
            MLPipelineOrchestrator,
        )
        return MLPipelineOrchestrator(**services)

    # Health check methods
    async def _check_mlflow_health(self) -> dict[str, Any]:
        """Check MLflow service health."""
        try:
            mlflow_service = await self.get(MLflowServiceProtocol)
            return await mlflow_service.health_check()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _check_cache_health(self) -> dict[str, Any]:
        """Check cache service health."""
        try:
            cache_service = await self.get(CacheServiceProtocol)
            return await cache_service.health_check()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _check_database_health(self) -> dict[str, Any]:
        """Check database service health."""
        try:
            db_service = await self.get(DatabaseServiceProtocol)
            return await db_service.health_check()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    # Scoped service support
    @asynccontextmanager
    async def scope(self, scope_id: str):
        """Create a scoped context for scoped services."""
        self._scoped_services[scope_id] = {}
        try:
            yield
        finally:
            scoped_services = self._scoped_services.pop(scope_id, {})
            for service in scoped_services.values():
                if hasattr(service, "cleanup") and callable(service.cleanup):
                    try:
                        if inspect.iscoroutinefunction(service.cleanup):
                            await service.cleanup()
                        else:
                            service.cleanup()
                    except Exception as e:
                        self.logger.exception(f"Error cleaning up scoped service: {e}")

    def set_metrics_registry(self, metrics_registry: MetricsRegistryProtocol) -> None:
        """Set the metrics registry for performance tracking."""
        self._metrics_registry = metrics_registry

    def get_services_by_tag(self, tag: str) -> dict[type[Any], Any]:
        """Get all services registered with a specific tag."""
        # This would need to aggregate across all containers
        return {}

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for the DI container."""
        return {
            "resolution_times": dict(self._resolution_times),
            "initialization_order": [t.__name__ for t in self._initialization_order],
            "registered_services_count": sum(len(getattr(container, "_services", {}))
                for container in [self._core_container, self._security_container,
                                self._database_container, self._monitoring_container]),
            "scoped_contexts_count": len(self._scoped_services),
            "active_resources_count": len(self._resources),
        }

    async def health_check(self) -> dict:
        """Perform health check on all registered services."""
        return await self.health_check_all()

    def get_registration_info(self) -> dict:
        """Get information about registered services."""
        info = {
            "orchestrator_name": self.name,
            "containers": {},
        }

        containers = {
            "core": self._core_container,
            "security": self._security_container,
            "database": self._database_container,
            "monitoring": self._monitoring_container,
        }

        for name, container in containers.items():
            if hasattr(container, "get_registration_info"):
                info["containers"][name] = container.get_registration_info()

        return info

    async def initialize(self) -> None:
        """Initialize all containers."""
        containers = [
            self._core_container,
            self._security_container,
            self._database_container,
            self._monitoring_container,
        ]

        for container in containers:
            if hasattr(container, "initialize"):
                await container.initialize()

        # Initialize ML container separately
        if self._ml_container and hasattr(self._ml_container, "initialize_all_services"):
            await self._ml_container.initialize_all_services()

        # Initialize CLI container separately
        if self._cli_container and hasattr(self._cli_container, "initialize"):
            await self._cli_container.initialize()

    async def shutdown(self):
        """Enhanced shutdown with comprehensive resource cleanup."""
        self.logger.info(f"Shutting down container orchestrator '{self.name}'")

        # Shutdown ML container first
        if self._ml_container and hasattr(self._ml_container, "shutdown_all_services"):
            await self._ml_container.shutdown_all_services()

        # Shutdown CLI container
        if self._cli_container and hasattr(self._cli_container, "shutdown"):
            await self._cli_container.shutdown()

        # Shutdown other containers in reverse order
        containers = [
            self._monitoring_container,
            self._database_container,
            self._security_container,
            self._core_container,
        ]

        for container in containers:
            if hasattr(container, "shutdown"):
                await container.shutdown()

        # Clear local state
        self._scoped_services.clear()
        self._resources.clear()
        self._resolution_stack.clear()
        self._resolution_times.clear()
        self._initialization_order.clear()
        self._health_checks.clear()

        self.logger.info(f"Container orchestrator '{self.name}' shutdown complete")


# Global container instance
_container: DIContainer | None = None


async def get_container() -> DIContainer:
    """Get the global DI container instance.

    Returns:
        DIContainer: Global container orchestrator instance
    """
    global _container
    if _container is None:
        _container = DIContainer()
    return _container


async def get_datetime_service() -> DateTimeServiceProtocol:
    """Get datetime service instance.

    Convenience function for getting the datetime service.

    Returns:
        DateTimeServiceProtocol: DateTime service instance
    """
    container = await get_container()
    return await container.get(DateTimeServiceProtocol)


async def shutdown_container():
    """Shutdown the global container."""
    global _container
    if _container:
        await _container.shutdown()
        _container = None


# Container initialization utility
async def initialize_container():
    """Initialize the global container (legacy support)."""
    container = await get_container()
    await container.initialize()
