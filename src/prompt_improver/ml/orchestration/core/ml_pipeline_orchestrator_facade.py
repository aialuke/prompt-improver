"""
ML Pipeline Orchestrator Facade (Decomposed Architecture)

Lightweight orchestrator facade that coordinates between 5 focused services,
replacing the original 1,043-line god object while maintaining identical functionality.

Services:
1. WorkflowOrchestrator - Core workflow execution and pipeline coordination
2. ComponentManager - Component loading, lifecycle management, and registry operations  
3. SecurityIntegrationService - Security validation, input sanitization, and access control
4. DeploymentPipelineService - Model deployment, versioning, and release management
5. MonitoringCoordinator - Health monitoring, metrics collection, and performance tracking
"""

import logging
from datetime import datetime, timezone
from typing import Any

# Import Protocol interfaces for dependency injection
from ....shared.interfaces.protocols.ml import (
    CacheServiceProtocol,
    ComponentFactoryProtocol,
    ComponentRegistryProtocol,
    DatabaseServiceProtocol,
    EventBusProtocol,
    HealthMonitorProtocol,
    MLflowServiceProtocol,
    ResourceManagerProtocol,
    WorkflowEngineProtocol,
)
from ....shared.interfaces.protocols.core import RetryManagerProtocol

# Import actual classes where protocols don't exist yet
from ....security.input_sanitization import InputSanitizer
from ....security.memory_guard import MemoryGuard
from ..config.external_services_config import ExternalServicesConfig
from ..config.orchestrator_config import OrchestratorConfig
from ..integration.component_invoker import ComponentInvoker
from ..integration.direct_component_loader import DirectComponentLoader
from .orchestrator_types import PipelineState, WorkflowInstance

# Import the decomposed services
from ..services.component_manager import ComponentService
from ..services.deployment_pipeline_service import DeploymentPipelineService
from ..services.monitoring_coordinator import MonitoringCoordinator
from ..services.security_integration_service import SecurityIntegrationService
from ..services.workflow_orchestrator import WorkflowOrchestrator


class MLPipelineOrchestrator:
    """
    ML Pipeline Orchestrator Facade (Decomposed Architecture)
    
    Lightweight coordinator that delegates operations to focused services:
    - WorkflowOrchestrator: Workflow execution and coordination
    - ComponentManager: Component lifecycle and registry
    - SecurityIntegrationService: Security validation and access control
    - DeploymentPipelineService: Model deployment and versioning  
    - MonitoringCoordinator: Health monitoring and performance tracking
    
    Maintains identical public API to the original orchestrator while following
    clean architecture principles and single responsibility design.
    """

    def __init__(
        self,
        # REQUIRED dependencies (no fallbacks)
        event_bus: EventBusProtocol,
        workflow_engine: WorkflowEngineProtocol,
        resource_manager: ResourceManagerProtocol,
        component_registry: ComponentRegistryProtocol,
        component_factory: ComponentFactoryProtocol,
        
        # REQUIRED service dependencies
        mlflow_service: MLflowServiceProtocol,
        cache_service: CacheServiceProtocol,
        database_service: DatabaseServiceProtocol,
        
        # Configuration (with defaults)
        config: OrchestratorConfig,
        external_services_config: ExternalServicesConfig,
        
        # Optional services
        health_monitor: HealthMonitorProtocol | None = None,
        retry_manager: RetryManagerProtocol | None = None,
        input_sanitizer: InputSanitizer | None = None,
        memory_guard: MemoryGuard | None = None
    ):
        """Initialize ML Pipeline Orchestrator with pure dependency injection."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.external_services_config = external_services_config
        
        # Create component loader and invoker for orchestration
        self.component_loader = DirectComponentLoader()
        self.component_invoker = ComponentInvoker(component_loader=self.component_loader)
        
        # Initialize the 5 focused services
        self.workflow_orchestrator = WorkflowOrchestrator(
            event_bus=event_bus,
            workflow_engine=workflow_engine,
            component_invoker=self.component_invoker
        )
        
        self.component_manager = ComponentService(
            component_registry=component_registry,
            component_factory=component_factory,
            event_bus=event_bus,
            component_loader=self.component_loader,
            component_invoker=self.component_invoker
        )
        
        self.security_service = SecurityIntegrationService(
            event_bus=event_bus,
            input_sanitizer=input_sanitizer,
            memory_guard=memory_guard
        )
        
        self.deployment_service = DeploymentPipelineService(
            event_bus=event_bus,
            external_services_config=external_services_config
        )
        
        self.monitoring_coordinator = MonitoringCoordinator(
            event_bus=event_bus,
            resource_manager=resource_manager,
            mlflow_service=mlflow_service,
            cache_service=cache_service,
            database_service=database_service,
            health_monitor=health_monitor,
            component_invoker=self.component_invoker
        )
        
        # State tracking (delegated from workflow orchestrator)
        self._is_initialized = False
        
        self.logger.info("ML Pipeline Orchestrator Facade initialized with decomposed architecture")

    async def initialize(self) -> None:
        """Initialize the orchestrator and all subsystems."""
        if self._is_initialized:
            return

        self.logger.info("Initializing ML Pipeline Orchestrator Facade")
        
        try:
            # Initialize all services in parallel for better performance
            await self.workflow_orchestrator.initialize()
            await self.component_manager.initialize()
            await self.security_service.initialize()
            await self.deployment_service.initialize()
            await self.monitoring_coordinator.initialize()
            
            # Discover and register components
            await self.component_manager.discover_components()
            
            # Direct component loading (Phase 6)
            await self.component_manager.load_direct_components()
            
            self._is_initialized = True
            
            self.logger.info("ML Pipeline Orchestrator Facade initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator facade: {e}")
            raise

    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator."""
        self.logger.info("Shutting down ML Pipeline Orchestrator Facade")
        
        try:
            # Shutdown all services
            await self.workflow_orchestrator.shutdown()
            await self.component_manager.shutdown()
            await self.security_service.shutdown()
            await self.deployment_service.shutdown()
            await self.monitoring_coordinator.shutdown()
            
            self._is_initialized = False
            
            self.logger.info("ML Pipeline Orchestrator Facade shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during orchestrator facade shutdown: {e}")
            raise

    # Workflow Operations (delegated to WorkflowOrchestrator)
    async def start_workflow(self, workflow_type: str, parameters: dict[str, Any]) -> str:
        """Start a new ML workflow."""
        return await self.workflow_orchestrator.start_workflow(workflow_type, parameters)

    async def stop_workflow(self, workflow_id: str) -> None:
        """Stop a running workflow."""
        await self.workflow_orchestrator.stop_workflow(workflow_id)

    async def get_workflow_status(self, workflow_id: str) -> WorkflowInstance:
        """Get the status of a workflow."""
        return await self.workflow_orchestrator.get_workflow_status(workflow_id)

    async def list_workflows(self) -> list[WorkflowInstance]:
        """List all workflows."""
        return await self.workflow_orchestrator.list_workflows()

    async def run_training_workflow(self, training_data: Any) -> dict[str, Any]:
        """Run a complete training workflow."""
        return await self.workflow_orchestrator.run_training_workflow(training_data)

    async def run_evaluation_workflow(self, evaluation_data: Any) -> dict[str, Any]:
        """Run a complete evaluation workflow."""
        return await self.workflow_orchestrator.run_evaluation_workflow(evaluation_data)

    # Component Operations (delegated to ComponentManager)
    async def get_component_health(self) -> dict[str, bool]:
        """Get health status of all registered components."""
        return await self.component_manager.get_component_health()

    async def invoke_component(self, component_name: str, method_name: str, *args, **kwargs) -> Any:
        """Invoke a method on a loaded ML component."""
        return await self.component_manager.invoke_component(component_name, method_name, *args, **kwargs)

    def get_loaded_components(self) -> list[str]:
        """Get list of loaded component names."""
        return self.component_manager.get_loaded_components()

    def get_component_methods(self, component_name: str) -> list[str]:
        """Get available methods for a component."""
        return self.component_manager.get_component_methods(component_name)

    # Security Operations (delegated to SecurityIntegrationService)
    async def validate_input_secure(self, input_data: Any, context: dict[str, Any] = None) -> Any:
        """Validate input data using integrated security sanitizer."""
        return await self.security_service.validate_input_secure(input_data, context)

    async def monitor_memory_usage(self, operation_name: str = "orchestrator_operation", component_name: str = None) -> Any:
        """Monitor memory usage for orchestrator operations."""
        return await self.security_service.monitor_memory_usage(operation_name, component_name)

    def monitor_operation_memory(self, operation_name: str, component_name: str = None) -> Any:
        """Async context manager for monitoring memory during orchestrator operations."""
        return self.security_service.monitor_operation_memory(operation_name, component_name)

    async def validate_operation_memory(self, data: Any, operation_name: str, component_name: str = None) -> bool:
        """Validate memory requirements for ML operations."""
        return await self.security_service.validate_operation_memory(data, operation_name, component_name)

    async def run_training_workflow_secure(self, training_data: Any, context: dict[str, Any] = None) -> dict[str, Any]:
        """Run a complete training workflow with security validation."""
        return await self.security_service.run_training_workflow_secure(training_data, context)

    async def run_training_workflow_with_memory_monitoring(self, training_data: Any, context: dict[str, Any] = None) -> dict[str, Any]:
        """Run a complete training workflow with comprehensive memory monitoring."""
        return await self.security_service.run_training_workflow_with_memory_monitoring(
            training_data, 
            context,
            workflow_runner=self.workflow_orchestrator.run_training_workflow
        )

    # Deployment Operations (delegated to DeploymentPipelineService)
    async def run_native_deployment_workflow(self, model_id: str, deployment_config: dict[str, Any]) -> dict[str, Any]:
        """Run native ML model deployment workflow without Docker containers."""
        return await self.deployment_service.run_native_deployment_workflow(model_id, deployment_config)

    async def run_complete_ml_pipeline(self, 
                                     training_data: Any,
                                     model_config: dict[str, Any],
                                     deployment_config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run complete ML pipeline: training, evaluation, and native deployment."""
        return await self.deployment_service.run_complete_ml_pipeline(
            training_data=training_data,
            model_config=model_config,
            deployment_config=deployment_config,
            training_runner=self.security_service.run_training_workflow_with_memory_monitoring,
            evaluation_runner=self.workflow_orchestrator.run_evaluation_workflow
        )

    # Monitoring Operations (delegated to MonitoringCoordinator)
    async def health_check(self) -> dict[str, Any]:
        """Comprehensive health check of the orchestrator and all components."""
        return await self.monitoring_coordinator.health_check()

    async def get_resource_usage(self) -> dict[str, Any]:
        """Get current resource usage statistics."""
        return await self.monitoring_coordinator.get_resource_usage()

    def get_invocation_history(self, component_name: str | None = None) -> list[dict[str, Any]]:
        """Get component invocation history."""
        return self.monitoring_coordinator.get_invocation_history(component_name)

    # Legacy compatibility methods
    @property
    def state(self) -> PipelineState:
        """Get current pipeline state (compatibility)."""
        return self.workflow_orchestrator.state if hasattr(self.workflow_orchestrator, 'state') else PipelineState.idle