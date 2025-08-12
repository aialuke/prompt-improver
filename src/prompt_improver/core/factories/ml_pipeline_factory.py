"""ML Pipeline Factory with Dependency Injection (2025).

Factory pattern implementation for creating ML pipeline components with
proper dependency injection, following modern architecture patterns.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Dict, Optional

from prompt_improver.core.protocols.ml_protocols import (
    CacheServiceProtocol,
    DatabaseServiceProtocol,
    EventBusProtocol,
    HealthMonitorProtocol,
    MLflowServiceProtocol,
    ResourceManagerProtocol,
    ServiceContainerProtocol,
    WorkflowEngineProtocol,
)

if TYPE_CHECKING:
    from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import (
        MLPipelineOrchestrator,
    )
logger = logging.getLogger(__name__)


class MLPipelineOrchestratorFactory:
    """Factory for creating ML Pipeline Orchestrators with dependency injection.

    Follows 2025 architecture patterns:
    - Constructor injection instead of service locator
    - Protocol-based dependencies
    - Async initialization
    - Resource lifecycle management
    """

    @staticmethod
    def create_with_dependencies(
        mlflow_service: MLflowServiceProtocol,
        cache_service: CacheServiceProtocol,
        database_service: DatabaseServiceProtocol,
        event_bus: EventBusProtocol,
        workflow_engine: WorkflowEngineProtocol,
        resource_manager: ResourceManagerProtocol,
        health_monitor: HealthMonitorProtocol,
        config: dict[str, Any] | None = None,
    ) -> "MLPipelineOrchestrator":
        """Create orchestrator with all dependencies injected.

        Args:
            mlflow_service: MLflow service for experiment tracking
            cache_service: Cache service for performance
            database_service: Database service for persistence
            event_bus: Event bus for loose coupling
            workflow_engine: Workflow execution engine
            resource_manager: Resource allocation manager
            health_monitor: Health monitoring service
            config: Optional configuration

        Returns:
            Configured ML pipeline orchestrator
        """
        from prompt_improver.ml.orchestration.config.external_services_config import (
            ExternalServicesConfig,
        )
        from prompt_improver.ml.orchestration.config.orchestrator_config import (
            OrchestratorConfig,
        )
        from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import (
            MLPipelineOrchestrator,
        )

        orch_config = OrchestratorConfig()
        if config:
            for key, value in config.items():
                if hasattr(orch_config, key):
                    setattr(orch_config, key, value)
        external_config = ExternalServicesConfig()
        return MLPipelineOrchestrator(
            config=orch_config,
            external_services_config=external_config,
            mlflow_service=mlflow_service,
            cache_service=cache_service,
            database_service=database_service,
            event_bus=event_bus,
            workflow_engine=workflow_engine,
            resource_manager=resource_manager,
            health_monitor=health_monitor,
        )

    @staticmethod
    async def create_from_container(
        service_container: ServiceContainerProtocol,
        config: dict[str, Any] | None = None,
    ) -> "MLPipelineOrchestrator":
        """Create orchestrator from service container with pure dependency injection.

        Args:
            service_container: Container with all registered services
            config: Optional configuration overrides

        Returns:
            Configured ML pipeline orchestrator

        Raises:
            RuntimeError: If required services not registered in container
        """
        logger.info("Creating ML Pipeline Orchestrator from service container")
        try:
            event_bus = await service_container.get_service("event_bus")
            workflow_engine = await service_container.get_service("workflow_engine")
            resource_manager = await service_container.get_service("resource_manager")
            component_registry = await service_container.get_service(
                "component_registry"
            )
            component_factory = await service_container.get_service("component_factory")
            mlflow_service = await service_container.get_service("mlflow_service")
            cache_service = await service_container.get_service("cache_service")
            database_service = await service_container.get_service("database_service")
            orchestrator_config = await service_container.get_service(
                "orchestrator_config"
            )
            external_config = await service_container.get_service(
                "external_services_config"
            )
            health_monitor = None
            retry_manager = None
            input_sanitizer = None
            memory_guard = None
            try:
                health_monitor = await service_container.get_service("health_monitor")
                retry_manager = await service_container.get_service("retry_manager")
                input_sanitizer = await service_container.get_service("input_sanitizer")
                memory_guard = await service_container.get_service("memory_guard")
            except KeyError:
                logger.debug("Optional services not available in container")
            if config:
                for key, value in config.items():
                    if hasattr(orchestrator_config, key):
                        setattr(orchestrator_config, key, value)
            from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import (
                MLPipelineOrchestrator,
            )

            orchestrator = MLPipelineOrchestrator(
                event_bus=event_bus,
                workflow_engine=workflow_engine,
                resource_manager=resource_manager,
                component_registry=component_registry,
                component_factory=component_factory,
                mlflow_service=mlflow_service,
                cache_service=cache_service,
                database_service=database_service,
                config=orchestrator_config,
                external_services_config=external_config,
                health_monitor=health_monitor,
                retry_manager=retry_manager,
                input_sanitizer=input_sanitizer,
                memory_guard=memory_guard,
            )
            logger.info(
                "ML Pipeline Orchestrator created successfully from service container"
            )
            return orchestrator
        except KeyError as e:
            raise RuntimeError(f"Required service not found in container: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to create orchestrator from container: {e}")


class ComponentLoaderFactory:
    """Factory for creating component loaders."""

    @staticmethod
    def create_component_loader(
        cache_service: CacheServiceProtocol, config: dict[str, Any] | None = None
    ) -> Any:
        """Create component loader with cache service."""
        from prompt_improver.ml.orchestration.integration.direct_component_loader import (
            DirectComponentLoader,
        )

        return DirectComponentLoader(cache_service=cache_service, config=config or {})


def create_ml_pipeline_orchestrator(
    services: dict[str, Any], config: dict[str, Any] | None = None
) -> "MLPipelineOrchestrator":
    """Create orchestrator from service dictionary (used by DI container).

    Args:
        services: Dictionary of service instances
        config: Optional configuration

    Returns:
        Configured orchestrator
    """
    return MLPipelineOrchestratorFactory.create_with_dependencies(
        mlflow_service=services["mlflow_service"],
        cache_service=services["cache_service"],
        database_service=services["database_service"],
        event_bus=services["event_bus"],
        workflow_engine=services["workflow_engine"],
        resource_manager=services["resource_manager"],
        health_monitor=services["health_monitor"],
        config=config,
    )


async def create_production_orchestrator(
    config_file: str | None = None,
) -> "MLPipelineOrchestrator":
    """Create production-ready orchestrator with full configuration.

    Args:
        config_file: Path to configuration file

    Returns:
        Production-configured orchestrator
    """
    config = {}
    if config_file:
        import yaml

        with open(config_file) as f:
            config = yaml.safe_load(f)
    return await MLPipelineOrchestratorFactory.create_with_external_services(config)


@asynccontextmanager
async def ml_pipeline_context(config: dict[str, Any] | None = None):
    """Async context manager for ML pipeline with automatic lifecycle management.

    Usage:
        async with ml_pipeline_context(config) as orchestrator:
            await orchestrator.run_workflow(...)
    """
    orchestrator = None
    try:
        orchestrator = (
            await MLPipelineOrchestratorFactory.create_with_external_services(config)
        )
        yield orchestrator
    finally:
        if orchestrator and hasattr(orchestrator, "shutdown"):
            await orchestrator.shutdown()
            logger.info("ML Pipeline Orchestrator shutdown completed")


class ExternalServiceFactory:
    """Factory for creating external service adapters."""

    @staticmethod
    def create_mlflow_adapter(config: dict[str, Any]) -> MLflowServiceProtocol:
        """Create MLflow service adapter."""
        from prompt_improver.core.services.ml_service_adapters import (
            create_mlflow_service,
        )

        return create_mlflow_service(config)

    @staticmethod
    def create_redis_adapter(config: dict[str, Any]) -> CacheServiceProtocol:
        """Create Redis cache service adapter."""
        from prompt_improver.core.services.ml_service_adapters import (
            create_cache_service,
        )

        return create_cache_service(config)

    @staticmethod
    def create_postgresql_adapter(config: dict[str, Any]) -> DatabaseServiceProtocol:
        """Create PostgreSQL database service adapter."""
        from prompt_improver.core.services.ml_service_adapters import (
            create_database_service,
        )

        return create_database_service(config)

    @staticmethod
    def create_event_bus_adapter(config: dict[str, Any]) -> EventBusProtocol:
        """Create event bus service adapter."""
        from prompt_improver.core.services.ml_service_adapters import create_event_bus

        return create_event_bus(config)
