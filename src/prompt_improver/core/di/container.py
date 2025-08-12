"""Enhanced Dependency Injection Container following 2025 best practices.

This module provides a comprehensive dependency injection system that combines
the power of python-dependency-injector with async-compatible extensions
for ML pipeline orchestration and modern Python applications.

Features:
- Modern dependency-injector integration
- Async service initialization and lifecycle management
- ML component-specific factory patterns
- Resource management for database connections
- Type-safe service resolution with protocols
- Circular dependency detection
- Service health monitoring
- Easy testing with mock services
- Performance metrics collection
"""

import asyncio
import inspect
import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Type, TypeVar

if TYPE_CHECKING:
    from prompt_improver.core.events.ml_event_bus import MLEventBus
from prompt_improver.core.interfaces.datetime_service import DateTimeServiceProtocol
from prompt_improver.core.interfaces.ml_interface import (
    MLAnalysisInterface,
    MLHealthInterface,
    MLModelInterface,
    MLServiceInterface,
    MLTrainingInterface,
)
from prompt_improver.core.protocols.ml_protocols import (
    CacheServiceProtocol,
    ComponentInvokerProtocol,
    ComponentLoaderProtocol,
    ComponentRegistryProtocol,
    DatabaseServiceProtocol,
    EventBusProtocol,
    ExternalServicesConfigProtocol,
    HealthMonitorProtocol,
    MLflowServiceProtocol,
    MLPipelineFactoryProtocol,
    ResourceManagerProtocol,
    ServiceConnectionInfo,
    WorkflowEngineProtocol,
)
from prompt_improver.core.protocols.retry_protocols import MetricsRegistryProtocol
from prompt_improver.core.services.datetime_service import DateTimeService
from prompt_improver.core.services.ml_service import EventBasedMLService
from prompt_improver.shared.interfaces.ab_testing import IABTestingService

DEPENDENCY_INJECTOR_AVAILABLE = False
T = TypeVar("T")


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

    def __init__(self, cycle: str):
        self.cycle = cycle
        super().__init__(f"Circular dependency detected: {cycle}")


class ServiceNotRegisteredError(Exception):
    """Raised when attempting to resolve unregistered service."""

    def __init__(self, service_type: type[Any]):
        self.service_type = service_type
        super().__init__(f"Service not registered: {service_type.__name__}")


class ServiceInitializationError(Exception):
    """Raised when service initialization fails."""

    def __init__(self, service_type: type[Any], original_error: Exception):
        self.service_type = service_type
        self.original_error = original_error
        super().__init__(
            f"Failed to initialize {service_type.__name__}: {original_error}"
        )


class ResourceManagementError(Exception):
    """Raised when resource management operations fail."""


class DIContainer:
    """Enhanced dependency injection container with 2025 best practices.

    Provides comprehensive service registration, resolution, and lifecycle management
    for ML Pipeline Orchestrator systems with async support, resource management,
    and factory patterns optimized for ML workloads.

    Features:
    - Async service initialization and cleanup
    - Resource lifecycle management (connections, models, etc.)
    - Factory patterns for metrics collectors and health monitors
    - Scoped service lifetimes for request/session isolation
    - Performance monitoring and health checks
    - Integration with python-dependency-injector when available
    """

    def __init__(self, logger: logging.Logger | None = None, name: str = "default"):
        """Initialize the enhanced DI container.

        Args:
            logger: Optional logger for debugging and monitoring
            name: Container name for identification in logs
        """
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self._services: dict[type[Any], ServiceRegistration] = {}
        self._scoped_services: dict[str, dict[type[Any], Any]] = {}
        self._resources: set[Any] = set()
        self._resolution_stack: set[type[Any]] = set()
        self._lock = asyncio.Lock()
        self._metrics_registry: MetricsRegistryProtocol | None = None
        self._health_checks: dict[str, Callable[[], Any]] = {}
        self._resolution_times: dict[type[Any], float] = {}
        self._initialization_order: list[type[Any]] = []
        self._register_default_services()
        self.logger.debug(f"Enhanced DIContainer '{self.name}' initialized")

    def _register_default_services(self):
        """Register default system services."""
        self.register_singleton(DateTimeServiceProtocol, DateTimeService)
        self.register_instance(DIContainer, self)
        self.register_ml_interfaces()

    def register_singleton(
        self,
        interface: type[T],
        implementation: type[T],
        tags: set[str] | None = None,
        initializer: Callable[[Any], Any] | None = None,
        finalizer: Callable[[Any], Any] | None = None,
        health_check: Callable[[], Any] | None = None,
    ) -> None:
        """Register a service as singleton.

        Args:
            interface: Service interface/protocol
            implementation: Concrete implementation class
            tags: Optional tags for service categorization
            initializer: Optional async initializer function
            finalizer: Optional cleanup function
            health_check: Optional health check function
        """
        self._services[interface] = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SINGLETON,
            initializer=initializer,
            finalizer=finalizer,
            tags=tags or set(),
            health_check=health_check,
        )
        self.logger.debug(
            f"Registered singleton: {interface.__name__} -> {implementation.__name__}"
        )

    def register_transient(
        self, interface: type[T], implementation: type[T], tags: set[str] | None = None
    ) -> None:
        """Register a service as transient (new instance each time).

        Args:
            interface: Service interface/protocol
            implementation: Concrete implementation class
            tags: Optional tags for service categorization
        """
        self._services[interface] = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.TRANSIENT,
            tags=tags or set(),
        )
        self.logger.debug(
            f"Registered transient: {interface.__name__} -> {implementation.__name__}"
        )

    def register_factory(
        self,
        interface: type[T],
        factory: Callable[[], T],
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
        tags: set[str] | None = None,
        initializer: Callable[[Any], Any] | None = None,
        finalizer: Callable[[Any], Any] | None = None,
    ) -> None:
        """Register a service with custom factory function.

        Args:
            interface: Service interface/protocol
            factory: Factory function to create service instance
            lifetime: Service lifetime (singleton, transient, or scoped)
            tags: Optional tags for service categorization
            initializer: Optional async initializer function
            finalizer: Optional cleanup function
        """
        self._services[interface] = ServiceRegistration(
            interface=interface,
            implementation=None,
            lifetime=lifetime,
            factory=factory,
            initializer=initializer,
            finalizer=finalizer,
            tags=tags or set(),
        )
        self.logger.debug(
            f"Registered factory: {interface.__name__} (lifetime: {lifetime.value})"
        )

    def register_instance(
        self,
        interface: type[T],
        instance: T,
        tags: set[str] | None = None,
        finalizer: Callable[[Any], Any] | None = None,
        health_check: Callable[[], Any] | None = None,
    ) -> None:
        """Register a pre-created service instance.

        Args:
            interface: Service interface/protocol
            instance: Pre-created service instance
            tags: Optional tags for service categorization
            finalizer: Optional cleanup function
            health_check: Optional health check function
        """
        registration = ServiceRegistration(
            interface=interface,
            implementation=type(instance),
            lifetime=ServiceLifetime.SINGLETON,
            initialized=True,
            instance=instance,
            finalizer=finalizer,
            tags=tags or set(),
            health_check=health_check,
        )
        self._services[interface] = registration
        self.logger.debug(f"Registered instance: {interface.__name__}")

    async def get(self, interface: type[T], scope_id: str | None = None) -> T:
        """Resolve service instance.

        Args:
            interface: Service interface to resolve
            scope_id: Optional scope identifier for scoped services

        Returns:
            Service instance

        Raises:
            ServiceNotRegisteredError: If service is not registered
            CircularDependencyError: If circular dependency detected
            ServiceInitializationError: If service initialization fails
        """
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
        """Internal service resolution with enhanced lifecycle management.

        Args:
            interface: Service interface to resolve
            scope_id: Optional scope identifier for scoped services

        Returns:
            Service instance
        """
        if interface not in self._services:
            raise ServiceNotRegisteredError(interface)
        if interface in self._resolution_stack:
            cycle = (
                " -> ".join([t.__name__ for t in self._resolution_stack])
                + f" -> {interface.__name__}"
            )
            raise CircularDependencyError(cycle)
        registration = self._services[interface]
        if registration.lifetime == ServiceLifetime.SINGLETON:
            if registration.initialized and registration.instance is not None:
                return registration.instance
        elif registration.lifetime == ServiceLifetime.SCOPED:
            if scope_id and scope_id in self._scoped_services:
                if interface in self._scoped_services[scope_id]:
                    return self._scoped_services[scope_id][interface]
        self._resolution_stack.add(interface)
        try:
            if registration.factory:
                instance = await self._create_from_factory(registration.factory)
            elif registration.implementation:
                instance = await self._create_from_class(registration.implementation)
            else:
                raise ServiceInitializationError(
                    interface, Exception("No factory or implementation provided")
                )
            if registration.initializer:
                try:
                    if inspect.iscoroutinefunction(registration.initializer):
                        await registration.initializer(instance)
                    else:
                        registration.initializer(instance)
                except Exception as e:
                    raise ServiceInitializationError(interface, e)
            if registration.lifetime == ServiceLifetime.SINGLETON:
                registration.instance = instance
                registration.initialized = True
                self._initialization_order.append(interface)
            elif registration.lifetime == ServiceLifetime.SCOPED and scope_id:
                if scope_id not in self._scoped_services:
                    self._scoped_services[scope_id] = {}
                self._scoped_services[scope_id][interface] = instance
            elif registration.lifetime == ServiceLifetime.RESOURCE:
                self._resources.add(instance)
                registration.instance = instance
                registration.initialized = True
            self.logger.debug(
                f"Resolved service: {interface.__name__} (lifetime: {registration.lifetime.value})"
            )
            return instance
        except Exception as e:
            if not isinstance(e, (ServiceInitializationError, CircularDependencyError)):
                raise ServiceInitializationError(interface, e)
            raise
        finally:
            self._resolution_stack.discard(interface)

    async def _create_from_factory(self, factory: Callable[[], Any]) -> Any:
        """Create service instance from factory function.

        Args:
            factory: Factory function

        Returns:
            Service instance
        """
        if inspect.iscoroutinefunction(factory):
            return await factory()
        return factory()

    async def _create_from_class(self, implementation: type[Any]) -> Any:
        """Create service instance from class constructor.

        Args:
            implementation: Implementation class

        Returns:
            Service instance
        """
        sig = inspect.signature(implementation.__init__)
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param.annotation != inspect.Parameter.empty:
                try:
                    dependency = await self._resolve_service(param.annotation)
                    kwargs[param_name] = dependency
                except ServiceNotRegisteredError:
                    if param.default != inspect.Parameter.empty:
                        continue
                    raise
        instance = implementation(**kwargs)
        if hasattr(instance, "initialize") and inspect.iscoroutinefunction(
            instance.initialize
        ):
            await instance.initialize()
        return instance

    async def health_check(self) -> dict:
        """Perform health check on all registered services.

        Returns:
            dict: Health check results
        """
        results = {
            "container_status": "healthy",
            "registered_services": len(self._services),
            "services": {},
        }
        for interface, registration in self._services.items():
            service_name = interface.__name__
            try:
                if (
                    registration.lifetime == ServiceLifetime.SINGLETON
                    and registration.initialized
                    and (registration.instance is not None)
                ):
                    instance = registration.instance
                    if hasattr(instance, "health_check") and callable(
                        instance.health_check
                    ):
                        if inspect.iscoroutinefunction(instance.health_check):
                            service_health = await instance.health_check()
                        else:
                            service_health = instance.health_check()
                        results["services"][service_name] = service_health
                    else:
                        results["services"][service_name] = {
                            "status": "healthy",
                            "note": "No health check method available",
                        }
                else:
                    results["services"][service_name] = {
                        "status": "not_initialized",
                        "lifetime": registration.lifetime.value,
                    }
            except Exception as e:
                results["services"][service_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                results["container_status"] = "degraded"
        return results

    def get_registration_info(self) -> dict:
        """Get information about registered services.

        Returns:
            dict: Registration information
        """
        info = {}
        for interface, registration in self._services.items():
            info[interface.__name__] = {
                "implementation": registration.implementation.__name__
                if registration.implementation
                else "Factory",
                "lifetime": registration.lifetime.value,
                "initialized": registration.initialized,
                "has_instance": registration.instance is not None,
            }
        return info

    def register_scoped(
        self, interface: type[T], implementation: type[T], tags: set[str] | None = None
    ) -> None:
        """Register a service as scoped (one instance per scope).

        Args:
            interface: Service interface/protocol
            implementation: Concrete implementation class
            tags: Optional tags for service categorization
        """
        self._services[interface] = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SCOPED,
            tags=tags or set(),
        )
        self.logger.debug(
            f"Registered scoped: {interface.__name__} -> {implementation.__name__}"
        )

    def register_resource(
        self,
        interface: type[T],
        factory: Callable[[], T],
        initializer: Callable | None = None,
        finalizer: Callable | None = None,
        tags: set[str] | None = None,
    ) -> None:
        """Register a resource that requires lifecycle management.

        Args:
            interface: Service interface/protocol
            factory: Factory function to create resource
            initializer: Optional async initializer function
            finalizer: Required cleanup function for resource
            tags: Optional tags for service categorization
        """
        self._services[interface] = ServiceRegistration(
            interface=interface,
            implementation=None,
            lifetime=ServiceLifetime.RESOURCE,
            factory=factory,
            initializer=initializer,
            finalizer=finalizer,
            tags=tags or set(),
        )
        self.logger.debug(f"Registered resource: {interface.__name__}")

    def register_metrics_collector_factory(
        self, collector_type: str = "opentelemetry"
    ) -> None:
        """Register factory for metrics collectors optimized for ML workloads.

        Args:
            collector_type: Type of metrics collector (opentelemetry only)
        """

        def create_metrics_collector() -> MetricsRegistryProtocol:
            if collector_type == "opentelemetry":
                from prompt_improver.performance.monitoring.metrics_registry import (
                    MetricsRegistry,
                )

                return MetricsRegistry()
            raise ValueError(
                f"Unsupported metrics collector type: {collector_type}. Only 'opentelemetry' is supported."
            )

        self.register_factory(
            MetricsRegistryProtocol,
            create_metrics_collector,
            ServiceLifetime.SINGLETON,
            tags={"metrics", "monitoring"},
        )
        self.logger.debug(f"Registered metrics collector factory: {collector_type}")

    def register_health_monitor_factory(self) -> None:
        """Register factory for health monitors with circuit breaker patterns."""

        def create_health_monitor():
            from prompt_improver.performance.monitoring.health.unified_health_system import (
                get_unified_health_monitor,
            )

            return get_unified_health_monitor()

        from prompt_improver.performance.monitoring.health.unified_health_system import (
            UnifiedHealthMonitor,
        )

        self.register_factory(
            UnifiedHealthMonitor,
            create_health_monitor,
            ServiceLifetime.SINGLETON,
            tags={"health", "monitoring"},
        )
        self.logger.debug("Registered health monitor factory")

    def register_mlflow_service_factory(self, config: ServiceConnectionInfo) -> None:
        """Register factory for MLflow service with proper lifecycle management.

        Args:
            config: MLflow service connection configuration
        """

        async def create_mlflow_service() -> MLflowServiceProtocol:
            from prompt_improver.integrations.mlflow_service import MLflowService

            service = MLflowService(config)
            await service.initialize()
            return service

        async def cleanup_mlflow_service(service: MLflowServiceProtocol):
            await service.shutdown()

        self.register_factory(
            MLflowServiceProtocol,
            create_mlflow_service,
            ServiceLifetime.SINGLETON,
            tags={"mlflow", "tracking", "external"},
            finalizer=cleanup_mlflow_service,
            health_check=lambda: asyncio.create_task(self._check_mlflow_health()),
        )
        self.logger.debug("Registered MLflow service factory")

    def register_cache_service_factory(self, config: ServiceConnectionInfo) -> None:
        """Register factory for Redis cache service with connection pooling.

        Args:
            config: Redis service connection configuration
        """

        async def create_cache_service() -> CacheServiceProtocol:
            from prompt_improver.integrations.redis_service import RedisService

            service = RedisService(config)
            await service.initialize()
            return service

        async def cleanup_cache_service(service: CacheServiceProtocol):
            await service.shutdown()

        self.register_factory(
            CacheServiceProtocol,
            create_cache_service,
            ServiceLifetime.SINGLETON,
            tags={"redis", "cache", "external"},
            finalizer=cleanup_cache_service,
            health_check=lambda: asyncio.create_task(self._check_cache_health()),
        )
        self.logger.debug("Registered cache service factory")

    def register_database_service_factory(self, config: ServiceConnectionInfo) -> None:
        """Register factory for PostgreSQL database service with connection pooling.

        Args:
            config: PostgreSQL service connection configuration
        """

        async def create_database_service() -> DatabaseServiceProtocol:
            from prompt_improver.integrations.postgresql_service import (
                PostgreSQLService,
            )

            service = PostgreSQLService(config)
            await service.initialize()
            return service

        async def cleanup_database_service(service: DatabaseServiceProtocol):
            await service.shutdown()

        self.register_factory(
            DatabaseServiceProtocol,
            create_database_service,
            ServiceLifetime.SINGLETON,
            tags={"postgresql", "database", "external"},
            finalizer=cleanup_database_service,
            health_check=lambda: asyncio.create_task(self._check_database_health()),
        )
        self.logger.debug("Registered database service factory")

    def register_external_services_config_factory(
        self, environment: str = "production"
    ) -> None:
        """Register factory for external services configuration.

        Args:
            environment: Deployment environment (development, staging, production)
        """

        def create_services_config() -> ExternalServicesConfigProtocol:
            from prompt_improver.integrations.services_config import (
                ExternalServicesConfigImpl,
            )

            return ExternalServicesConfigImpl.from_environment(environment)

        self.register_factory(
            ExternalServicesConfigProtocol,
            create_services_config,
            ServiceLifetime.SINGLETON,
            tags={"config", "external"},
        )
        self.logger.debug(
            f"Registered external services config factory (env: {environment})"
        )

    def register_event_bus_factory(self, config: Any = None) -> None:
        """Register factory for event bus with proper initialization.

        Args:
            config: Optional event bus configuration
        """

        async def create_event_bus() -> EventBusProtocol:
            from prompt_improver.ml.orchestration.events.adaptive_event_bus import (
                AdaptiveEventBus,
            )

            event_bus = AdaptiveEventBus(config)
            await event_bus.initialize()
            return event_bus

        async def cleanup_event_bus(event_bus: EventBusProtocol):
            await event_bus.shutdown()

        self.register_factory(
            EventBusProtocol,
            create_event_bus,
            ServiceLifetime.SINGLETON,
            tags={"events", "messaging", "core"},
            finalizer=cleanup_event_bus,
        )
        self.logger.debug("Registered event bus factory")

    def register_workflow_engine_factory(self, config: Any = None) -> None:
        """Register factory for workflow execution engine.

        Args:
            config: Optional workflow engine configuration
        """

        async def create_workflow_engine() -> WorkflowEngineProtocol:
            from prompt_improver.ml.orchestration.core.workflow_execution_engine import (
                WorkflowExecutionEngine,
            )

            engine = WorkflowExecutionEngine(config)
            await engine.initialize()
            return engine

        async def cleanup_workflow_engine(engine: WorkflowEngineProtocol):
            await engine.shutdown()

        self.register_factory(
            WorkflowEngineProtocol,
            create_workflow_engine,
            ServiceLifetime.SINGLETON,
            tags={"workflow", "execution", "core"},
            finalizer=cleanup_workflow_engine,
        )
        self.logger.debug("Registered workflow engine factory")

    def register_resource_manager_factory(self, config: Any = None) -> None:
        """Register factory for resource manager with monitoring capabilities.

        Args:
            config: Optional resource manager configuration
        """

        async def create_resource_manager() -> ResourceManagerProtocol:
            from prompt_improver.ml.orchestration.core.resource_manager import (
                ResourceManager,
            )

            manager = ResourceManager(config)
            await manager.initialize()
            return manager

        async def cleanup_resource_manager(manager: ResourceManagerProtocol):
            await manager.shutdown()

        self.register_factory(
            ResourceManagerProtocol,
            create_resource_manager,
            ServiceLifetime.SINGLETON,
            tags={"resources", "monitoring", "core"},
            finalizer=cleanup_resource_manager,
        )
        self.logger.debug("Registered resource manager factory")

    def register_component_registry_factory(self, config: Any = None) -> None:
        """Register factory for ML component registry.

        Args:
            config: Optional component registry configuration
        """

        async def create_component_registry() -> ComponentRegistryProtocol:
            from prompt_improver.ml.orchestration.core.component_registry import (
                ComponentRegistry,
            )

            registry = ComponentRegistry(config)
            await registry.initialize()
            return registry

        async def cleanup_component_registry(registry: ComponentRegistryProtocol):
            await registry.shutdown()

        self.register_factory(
            ComponentRegistryProtocol,
            create_component_registry,
            ServiceLifetime.SINGLETON,
            tags={"components", "registry", "core"},
            finalizer=cleanup_component_registry,
        )
        self.logger.debug("Registered component registry factory")

    def register_component_loader_factory(self) -> None:
        """Register factory for ML component loader."""

        def create_component_loader() -> ComponentLoaderProtocol:
            from prompt_improver.ml.orchestration.integration.direct_component_loader import (
                DirectComponentLoader,
            )

            return DirectComponentLoader()

        self.register_factory(
            ComponentLoaderProtocol,
            create_component_loader,
            ServiceLifetime.SINGLETON,
            tags={"components", "loader", "integration"},
        )
        self.logger.debug("Registered component loader factory")

    def register_component_invoker_factory(self) -> None:
        """Register factory for ML component invoker with dependency injection."""

        async def create_component_invoker() -> ComponentInvokerProtocol:
            component_loader = await self.get(ComponentLoaderProtocol)
            from prompt_improver.ml.orchestration.integration.component_invoker import (
                ComponentInvoker,
            )

            return ComponentInvoker(component_loader)

        self.register_factory(
            ComponentInvokerProtocol,
            create_component_invoker,
            ServiceLifetime.SINGLETON,
            tags={"components", "invoker", "integration"},
        )
        self.logger.debug("Registered component invoker factory")

    def register_ml_pipeline_factory(self) -> None:
        """Register factory for complete ML pipeline orchestrator with all dependencies."""

        async def create_ml_pipeline_orchestrator():
            config = await self.get(ExternalServicesConfigProtocol)
            event_bus = await self.get(EventBusProtocol)
            workflow_engine = await self.get(WorkflowEngineProtocol)
            resource_manager = await self.get(ResourceManagerProtocol)
            component_registry = await self.get(ComponentRegistryProtocol)
            component_loader = await self.get(ComponentLoaderProtocol)
            component_invoker = await self.get(ComponentInvokerProtocol)
            from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import (
                MLPipelineOrchestrator,
            )

            orchestrator = MLPipelineOrchestrator(
                config=config,
                event_bus=event_bus,
                workflow_engine=workflow_engine,
                resource_manager=resource_manager,
                component_registry=component_registry,
                component_loader=component_loader,
                component_invoker=component_invoker,
            )
            await orchestrator.initialize()
            return orchestrator

        async def cleanup_orchestrator(orchestrator):
            await orchestrator.shutdown()

        self.register_factory(
            MLPipelineFactoryProtocol,
            create_ml_pipeline_orchestrator,
            ServiceLifetime.SINGLETON,
            tags={"orchestrator", "ml", "pipeline", "core"},
            finalizer=cleanup_orchestrator,
        )
        self.logger.debug("Registered ML pipeline orchestrator factory")

    def register_ab_testing_service_factory(self) -> None:
        """Register factory for A/B testing service.

        This creates an IABTestingService instance that uses the performance layer
        implementation through the adapter pattern, maintaining clean architecture.
        """

        def create_ab_testing_service() -> IABTestingService:
            try:
                from prompt_improver.performance.testing.ab_testing_adapter import (
                    create_ab_testing_service_adapter,
                )
                from prompt_improver.performance.testing.ab_testing_service import (
                    ModernABConfig,
                )

                # Use default configuration optimized for prompt improvement
                config = ModernABConfig(
                    confidence_level=0.95,
                    statistical_power=0.8,
                    minimum_detectable_effect=0.05,
                    practical_significance_threshold=0.02,
                    enable_early_stopping=True,
                    enable_sequential_testing=True,
                )

                return create_ab_testing_service_adapter(config)
            except ImportError as e:
                # Fallback to no-op implementation if performance layer is unavailable
                from prompt_improver.shared.interfaces.ab_testing import (
                    NoOpABTestingService,
                )

                self.logger.warning(
                    f"A/B testing service unavailable, using no-op: {e}"
                )
                return NoOpABTestingService()

        self.register_factory(
            IABTestingService,
            create_ab_testing_service,
            ServiceLifetime.SINGLETON,
            tags={"ab_testing", "statistics", "experimentation"},
        )
        self.logger.debug("Registered A/B testing service factory")

    def register_all_ml_services(self, environment: str = "production") -> None:
        """Register all ML pipeline services with proper dependency injection.

        This method sets up the complete ML pipeline dependency graph following
        2025 best practices with constructor injection and proper lifecycle management.

        Args:
            environment: Deployment environment for service configuration
        """
        self.logger.info(
            f"Registering ML pipeline services for {environment} environment"
        )
        self.register_external_services_config_factory(environment)
        config_placeholder = ServiceConnectionInfo(
            host=os.getenv("POSTGRES_HOST", "postgres"), port=5432
        )
        self.register_mlflow_service_factory(config_placeholder)
        self.register_cache_service_factory(config_placeholder)
        self.register_database_service_factory(config_placeholder)
        self.register_event_bus_factory()
        self.register_component_loader_factory()
        self.register_workflow_engine_factory()
        self.register_resource_manager_factory()
        self.register_component_registry_factory()
        self.register_component_invoker_factory()
        self.register_health_monitor_factory()
        self.register_ab_testing_service_factory()
        self.register_ml_pipeline_factory()
        self.logger.info("ML pipeline services registration complete")

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

    def register_ml_interfaces(self) -> None:
        """Register ML interface factories that use event bus communication.

        This creates ML service interfaces that communicate via events without
        direct ML imports, ensuring clean MCP-ML boundary separation.
        """

        async def create_ml_event_bus():
            from prompt_improver.core.events.ml_event_bus import get_ml_event_bus

            return await get_ml_event_bus()

        async def cleanup_ml_event_bus(event_bus: "MLEventBus") -> None:
            await event_bus.shutdown()

        def get_ml_event_bus_type():
            from prompt_improver.core.events.ml_event_bus import MLEventBus

            return MLEventBus

        self.register_factory(
            get_ml_event_bus_type(),
            create_ml_event_bus,
            ServiceLifetime.SINGLETON,
            tags={"ml", "events", "communication"},
            finalizer=cleanup_ml_event_bus,
        )

        def create_ml_service_interface():
            return EventBasedMLService()

        self.register_factory(
            MLServiceInterface,
            create_ml_service_interface,
            ServiceLifetime.SINGLETON,
            tags={"ml", "unified"},
        )
        self.register_factory(
            MLAnalysisInterface,
            create_ml_service_interface,
            ServiceLifetime.SINGLETON,
            tags={"ml", "analysis"},
        )
        self.register_factory(
            MLTrainingInterface,
            create_ml_service_interface,
            ServiceLifetime.SINGLETON,
            tags={"ml", "training"},
        )
        self.register_factory(
            MLHealthInterface,
            create_ml_service_interface,
            ServiceLifetime.SINGLETON,
            tags={"ml", "health"},
        )
        self.register_factory(
            MLModelInterface,
            create_ml_service_interface,
            ServiceLifetime.SINGLETON,
            tags={"ml", "model"},
        )
        self.logger.info(
            "Registered ML interface factories with event bus communication"
        )

    @asynccontextmanager
    async def scope(self, scope_id: str):
        """Create a scoped context for scoped services.

        Args:
            scope_id: Unique identifier for this scope
        """
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
                        self.logger.error(f"Error cleaning up scoped service: {e}")

    def set_metrics_registry(self, metrics_registry: MetricsRegistryProtocol) -> None:
        """Set the metrics registry for performance tracking.

        Args:
            metrics_registry: Metrics registry implementation
        """
        self._metrics_registry = metrics_registry
        self.logger.debug("Metrics registry set for DI container")

    def get_services_by_tag(self, tag: str) -> dict[type[Any], ServiceRegistration]:
        """Get all services registered with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            Dictionary of service types to their registrations
        """
        return {
            service_type: registration
            for service_type, registration in self._services.items()
            if tag in registration.tags
        }

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for the DI container.

        Returns:
            Dictionary containing performance metrics
        """
        return {
            "resolution_times": dict(self._resolution_times),
            "initialization_order": [t.__name__ for t in self._initialization_order],
            "registered_services_count": len(self._services),
            "scoped_contexts_count": len(self._scoped_services),
            "active_resources_count": len(self._resources),
        }

    async def shutdown(self):
        """Enhanced shutdown with comprehensive resource cleanup."""
        self.logger.info(f"Shutting down enhanced DI container '{self.name}'")
        for resource in reversed(list(self._resources)):
            try:
                for registration in self._services.values():
                    if registration.instance is resource and registration.finalizer:
                        if inspect.iscoroutinefunction(registration.finalizer):
                            await registration.finalizer(resource)
                        else:
                            registration.finalizer(resource)
                        break
                else:
                    if hasattr(resource, "shutdown") and callable(resource.shutdown):
                        if inspect.iscoroutinefunction(resource.shutdown):
                            await resource.shutdown()
                        else:
                            resource.shutdown()
            except Exception as e:
                self.logger.error(
                    f"Error shutting down resource {type(resource).__name__}: {e}"
                )
        for interface in reversed(self._initialization_order):
            registration = self._services.get(interface)
            if registration and registration.instance:
                try:
                    if registration.finalizer:
                        if inspect.iscoroutinefunction(registration.finalizer):
                            await registration.finalizer(registration.instance)
                        else:
                            registration.finalizer(registration.instance)
                    elif hasattr(registration.instance, "shutdown") and callable(
                        registration.instance.shutdown
                    ):
                        if inspect.iscoroutinefunction(registration.instance.shutdown):
                            await registration.instance.shutdown()
                        else:
                            registration.instance.shutdown()
                except Exception as e:
                    self.logger.error(
                        f"Error shutting down service {interface.__name__}: {e}"
                    )
        for scope_id, scoped_services in self._scoped_services.items():
            self.logger.debug(f"Cleaning up scoped services for scope: {scope_id}")
            for service in scoped_services.values():
                try:
                    if hasattr(service, "cleanup") and callable(service.cleanup):
                        if inspect.iscoroutinefunction(service.cleanup):
                            await service.cleanup()
                        else:
                            service.cleanup()
                except Exception as e:
                    self.logger.error(f"Error cleaning up scoped service: {e}")
        self._services.clear()
        self._scoped_services.clear()
        self._resources.clear()
        self._resolution_stack.clear()
        self._resolution_times.clear()
        self._initialization_order.clear()
        self._health_checks.clear()
        self.logger.info(f"Enhanced DI container '{self.name}' shutdown complete")


_container: DIContainer | None = None


async def get_container() -> DIContainer:
    """Get the global DI container instance.

    Returns:
        DIContainer: Global container instance
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
