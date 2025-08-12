"""ML Pipeline Dependency Injection Container (2025).

Modern async DI container for ML pipeline orchestration with service lifecycle
management, health monitoring, and factory patterns.
"""

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, Dict, Optional, Type, TypeVar

from prompt_improver.core.protocols.ml_protocols import (
    CacheServiceProtocol,
    DatabaseServiceProtocol,
    EventBusProtocol,
    HealthMonitorProtocol,
    MLflowServiceProtocol,
    ResourceManagerProtocol,
    ServiceContainerProtocol,
    ServiceStatus,
    WorkflowEngineProtocol,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


class MLServiceContainer:
    """Async dependency injection container for ML pipeline services.

    Follows 2025 best practices with:
    - Protocol-based service registration
    - Async service lifecycle management
    - Factory pattern support
    - Health monitoring integration
    - Resource cleanup
    """

    def __init__(self):
        """Initialize the service container."""
        self._services: dict[str, Any] = {}
        self._factories: dict[str, Callable[[], Any]] = {}
        self._singletons: dict[str, Any] = {}
        self._initialized_services: set = set()
        self._health_checks: dict[str, Callable] = {}
        self._is_initialized = False
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def register_service(self, service_name: str, service_instance: Any) -> None:
        """Register a service instance."""
        self._services[service_name] = service_instance
        self._logger.debug(f"Registered service: {service_name}")

    def register_factory(
        self, service_name: str, factory: Callable[[], Any], singleton: bool = True
    ) -> None:
        """Register a service factory function."""
        self._factories[service_name] = factory
        if singleton:
            self._singletons[service_name] = None
        self._logger.debug(
            f"Registered factory for service: {service_name} (singleton={singleton})"
        )

    def register_mlflow_service_factory(
        self, factory: Callable[[], MLflowServiceProtocol]
    ) -> None:
        """Register MLflow service factory."""
        self.register_factory("mlflow_service", factory, singleton=True)

    def register_cache_service_factory(
        self, factory: Callable[[], CacheServiceProtocol]
    ) -> None:
        """Register cache service (Redis) factory."""
        self.register_factory("cache_service", factory, singleton=True)

    def register_database_service_factory(
        self, factory: Callable[[], DatabaseServiceProtocol]
    ) -> None:
        """Register database service factory."""
        self.register_factory("database_service", factory, singleton=True)

    def register_event_bus_factory(
        self, factory: Callable[[], EventBusProtocol]
    ) -> None:
        """Register event bus factory."""
        self.register_factory("event_bus", factory, singleton=True)

    def register_workflow_engine_factory(
        self, factory: Callable[[], WorkflowEngineProtocol]
    ) -> None:
        """Register workflow engine factory."""
        self.register_factory("workflow_engine", factory, singleton=True)

    def register_resource_manager_factory(
        self, factory: Callable[[], ResourceManagerProtocol]
    ) -> None:
        """Register resource manager factory."""
        self.register_factory("resource_manager", factory, singleton=True)

    def register_health_monitor_factory(
        self, factory: Callable[[], HealthMonitorProtocol]
    ) -> None:
        """Register health monitor factory."""
        self.register_factory("health_monitor", factory, singleton=True)

    def register_ml_pipeline_factory(
        self, factory: Callable[[dict[str, Any]], Any]
    ) -> None:
        """Register ML pipeline orchestrator factory."""
        self.register_factory(
            "ml_pipeline_orchestrator",
            lambda: factory(self.get_all_services()),
            singleton=False,
        )

    async def get_service(self, service_name: str) -> Any:
        """Get service by name, creating via factory if needed."""
        if service_name in self._services:
            return self._services[service_name]
        if (
            service_name in self._singletons
            and self._singletons[service_name] is not None
        ):
            return self._singletons[service_name]
        if service_name in self._factories:
            factory = self._factories[service_name]
            try:
                service_instance = factory()
                if hasattr(
                    service_instance, "initialize"
                ) and asyncio.iscoroutinefunction(service_instance.initialize):
                    await service_instance.initialize()
                    self._initialized_services.add(service_name)
                if service_name in self._singletons:
                    self._singletons[service_name] = service_instance
                self._logger.info(f"Created service via factory: {service_name}")
                return service_instance
            except Exception as e:
                self._logger.error(f"Failed to create service '{service_name}': {e}")
                raise RuntimeError(f"Service creation failed for '{service_name}': {e}")
        raise KeyError(f"Service not found: {service_name}")

    def get_all_services(self) -> dict[str, Any]:
        """Get all registered services for dependency injection."""
        all_services = {}
        all_services.update(self._services)
        all_services.update({
            k: v for k, v in self._singletons.items() if v is not None
        })
        return all_services

    async def initialize_all_services(self) -> None:
        """Initialize all registered services asynchronously."""
        if self._is_initialized:
            return
        self._logger.info("Initializing all ML pipeline services...")
        initialization_tasks = []
        for service_name in list(self._factories.keys()) + list(self._services.keys()):
            if service_name not in self._initialized_services:
                task = self._initialize_service(service_name)
                initialization_tasks.append(task)
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
        failed_services = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                service_name = (
                    list(self._factories.keys())[i]
                    if i < len(self._factories)
                    else list(self._services.keys())[i - len(self._factories)]
                )
                failed_services.append((service_name, result))
                self._logger.error(
                    f"Failed to initialize service '{service_name}': {result}"
                )
        if failed_services:
            raise RuntimeError(
                f"Failed to initialize services: {[name for name, _ in failed_services]}"
            )
        self._is_initialized = True
        self._logger.info("All ML pipeline services initialized successfully")

    async def _initialize_service(self, service_name: str) -> None:
        """Initialize a single service."""
        try:
            service = await self.get_service(service_name)
            if hasattr(service, "health_check"):
                self._health_checks[service_name] = service.health_check
            self._logger.debug(f"Initialized service: {service_name}")
        except Exception as e:
            self._logger.error(
                f"Service initialization failed for '{service_name}': {e}"
            )
            raise

    async def shutdown_all_services(self) -> None:
        """Gracefully shutdown all services."""
        self._logger.info("Shutting down all ML pipeline services...")
        shutdown_tasks = []
        all_services = list(self._services.items()) + [
            (k, v) for k, v in self._singletons.items() if v is not None
        ]
        for service_name, service in reversed(all_services):
            if hasattr(service, "shutdown") and asyncio.iscoroutinefunction(
                service.shutdown
            ):
                task = self._shutdown_service(service_name, service)
                shutdown_tasks.append(task)
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        self._services.clear()
        self._singletons = dict.fromkeys(self._singletons)
        self._initialized_services.clear()
        self._is_initialized = False
        self._logger.info("All services shutdown completed")

    async def _shutdown_service(self, service_name: str, service: Any) -> None:
        """Shutdown a single service."""
        try:
            await service.shutdown()
            self._logger.debug(f"Shutdown service: {service_name}")
        except Exception as e:
            self._logger.error(f"Service shutdown failed for '{service_name}': {e}")

    async def health_check_all_services(self) -> dict[str, ServiceStatus]:
        """Check health of all registered services."""
        health_results = {}
        health_tasks = []
        for service_name, health_check_func in self._health_checks.items():
            task = self._check_service_health(service_name, health_check_func)
            health_tasks.append(task)
        results = await asyncio.gather(*health_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            service_name = list(self._health_checks.keys())[i]
            if isinstance(result, Exception):
                health_results[service_name] = ServiceStatus.ERROR
                self._logger.error(
                    f"Health check failed for '{service_name}': {result}"
                )
            else:
                health_results[service_name] = result
        return health_results

    async def _check_service_health(
        self, service_name: str, health_check_func: Callable
    ) -> ServiceStatus:
        """Check health of a single service."""
        try:
            if asyncio.iscoroutinefunction(health_check_func):
                return await health_check_func()
            return health_check_func()
        except Exception as e:
            self._logger.error(f"Health check error for '{service_name}': {e}")
            return ServiceStatus.ERROR

    def is_initialized(self) -> bool:
        """Check if container is fully initialized."""
        return self._is_initialized

    @asynccontextmanager
    async def managed_container(self) -> AsyncContextManager["MLServiceContainer"]:
        """Context manager for automatic service lifecycle management."""
        try:
            await self.initialize_all_services()
            yield self
        finally:
            await self.shutdown_all_services()


def register_all_ml_services(
    container: MLServiceContainer, config: dict[str, Any] | None = None
) -> None:
    """Register all standard ML pipeline services with the container.

    Args:
        container: Service container to register with
        config: Optional configuration dictionary
    """
    config = config or {}
    from prompt_improver.core.factories.ml_pipeline_factory import (
        create_ml_pipeline_orchestrator,
    )
    from prompt_improver.core.services.ml_service_adapters import (
        create_cache_service,
        create_database_service,
        create_event_bus,
        create_health_monitor,
        create_mlflow_service,
        create_resource_manager,
        create_workflow_engine,
    )

    container.register_mlflow_service_factory(
        lambda: create_mlflow_service(config.get("mlflow", {}))
    )
    container.register_cache_service_factory(
        lambda: create_cache_service(config.get("cache", {}))
    )
    container.register_database_service_factory(
        lambda: create_database_service(config.get("database", {}))
    )
    container.register_event_bus_factory(
        lambda: create_event_bus(config.get("event_bus", {}))
    )
    container.register_workflow_engine_factory(
        lambda: create_workflow_engine(config.get("workflow", {}))
    )
    container.register_resource_manager_factory(
        lambda: create_resource_manager(config.get("resources", {}))
    )
    container.register_health_monitor_factory(
        lambda: create_health_monitor(config.get("health", {}))
    )
    container.register_ml_pipeline_factory(
        lambda services: create_ml_pipeline_orchestrator(
            services, config.get("orchestrator", {})
        )
    )
    logger.info("All ML pipeline services registered with DI container")
