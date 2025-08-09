"""Refactored ML Pipeline Orchestrator with Constructor Injection

This demonstrates how to refactor the existing MLPipelineOrchestrator to use
dependency injection instead of hard-coded dependencies, following 2025 best practices.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import logging
import time
from typing import Any, Callable, Dict, List, Optional
from ....core.protocols.ml_protocols import ComponentInvokerProtocol, ComponentLoaderProtocol, ComponentRegistryProtocol, EventBusProtocol, ExternalServicesConfigProtocol, HealthMonitorProtocol, ResourceManagerProtocol, WorkflowEngineProtocol

class PipelineState(Enum):
    """Pipeline execution states."""
    idle = 'idle'
    INITIALIZING = 'initializing'
    RUNNING = 'running'
    PAUSED = 'paused'
    STOPPING = 'stopping'
    ERROR = 'error'
    COMPLETED = 'completed'

@dataclass
class WorkflowInstance:
    """Represents a running workflow instance."""
    workflow_id: str
    workflow_type: str
    state: PipelineState
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MLPipelineOrchestratorRefactored:
    """
    Refactored ML Pipeline Orchestrator with Constructor Dependency Injection
    
    This version demonstrates 2025 best practices:
    - Constructor injection of all dependencies
    - Protocol-based abstractions for external services
    - Proper async initialization patterns
    - Resource lifecycle management
    - Comprehensive error handling
    
    Key improvements over the original:
    1. All dependencies injected via constructor (testable, flexible)
    2. Protocol interfaces enable easy mocking and testing
    3. Proper separation of concerns
    4. Resource management through DI container
    5. Health monitoring integration
    """

    def __init__(self, config: ExternalServicesConfigProtocol, event_bus: EventBusProtocol, workflow_engine: WorkflowEngineProtocol, resource_manager: ResourceManagerProtocol, component_registry: ComponentRegistryProtocol, component_loader: ComponentLoaderProtocol, component_invoker: ComponentInvokerProtocol, health_monitor: Optional[HealthMonitorProtocol]=None, logger: Optional[logging.Logger]=None):
        """
        Initialize ML Pipeline Orchestrator with injected dependencies.
        
        Args:
            config: External services configuration
            event_bus: Event bus for messaging
            workflow_engine: Workflow execution engine
            resource_manager: Resource management service
            component_registry: ML component registry
            component_loader: ML component loader
            component_invoker: ML component invoker
            health_monitor: Optional health monitoring service
            logger: Optional logger instance
        """
        self.config = config
        self.event_bus = event_bus
        self.workflow_engine = workflow_engine
        self.resource_manager = resource_manager
        self.component_registry = component_registry
        self.component_loader = component_loader
        self.component_invoker = component_invoker
        self.health_monitor = health_monitor
        self.logger = logger or logging.getLogger(__name__)
        self.state = PipelineState.idle
        self.active_workflows: Dict[str, WorkflowInstance] = {}
        self.component_health: Dict[str, bool] = {}
        self._is_initialized = False
        self._setup_component_references()

    def _setup_component_references(self) -> None:
        """Setup references between injected components."""
        if hasattr(self.workflow_engine, 'set_event_bus'):
            self.workflow_engine.set_event_bus(self.event_bus)
        if hasattr(self.resource_manager, 'set_event_bus'):
            self.resource_manager.set_event_bus(self.event_bus)
        self._setup_event_handlers()

    async def initialize(self) -> None:
        """Initialize the orchestrator and all subsystems."""
        if self._is_initialized:
            return
        self.logger.info('Initializing ML Pipeline Orchestrator (Refactored)')
        self.state = PipelineState.INITIALIZING
        try:
            initialization_tasks = [('component_registry', self.component_registry.initialize()), ('resource_manager', self.resource_manager.initialize()), ('workflow_engine', self.workflow_engine.initialize()), ('event_bus', self.event_bus.initialize())]
            for component_name, init_task in initialization_tasks:
                try:
                    await init_task
                    self.logger.debug('Initialized %s', component_name)
                except Exception as e:
                    self.logger.error('Failed to initialize {component_name}: %s', e)
                    raise
            await self._discover_components()
            await self._load_direct_components()
            if self.health_monitor:
                await self._setup_health_monitoring()
            await self._setup_monitoring()
            self.state = PipelineState.idle
            self._is_initialized = True
            await self._emit_orchestrator_event('ORCHESTRATOR_INITIALIZED', {'timestamp': datetime.now(timezone.utc).isoformat(), 'components_loaded': len(self.component_health)})
            self.logger.info('ML Pipeline Orchestrator initialized successfully (Refactored)')
        except Exception as e:
            self.state = PipelineState.ERROR
            self.logger.error('Failed to initialize orchestrator: %s', e)
            raise

    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator."""
        self.logger.info('Shutting down ML Pipeline Orchestrator (Refactored)')
        self.state = PipelineState.STOPPING
        try:
            await self._stop_all_workflows()
            shutdown_tasks = [('workflow_engine', self.workflow_engine.shutdown()), ('resource_manager', self.resource_manager.shutdown()), ('component_registry', self.component_registry.shutdown()), ('event_bus', self.event_bus.shutdown())]
            for component_name, shutdown_task in shutdown_tasks:
                try:
                    await shutdown_task
                    self.logger.debug('Shutdown %s', component_name)
                except Exception as e:
                    self.logger.error('Error shutting down {component_name}: %s', e)
            if self.health_monitor:
                try:
                    await self.health_monitor.shutdown()
                    self.logger.debug('Shutdown health monitor')
                except Exception as e:
                    self.logger.error('Error shutting down health monitor: %s', e)
            self.state = PipelineState.idle
            self._is_initialized = False
            self.logger.info('ML Pipeline Orchestrator shutdown complete (Refactored)')
        except Exception as e:
            self.logger.error('Error during orchestrator shutdown: %s', e)
            raise

    async def start_workflow(self, workflow_type: str, parameters: Dict[str, Any]) -> str:
        """Start a new ML workflow."""
        if not self._is_initialized:
            raise RuntimeError('Orchestrator not initialized')
        if self.state != PipelineState.idle:
            raise RuntimeError(f'Cannot start workflow in state: {self.state}')
        workflow_id = f"{workflow_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        workflow_instance = WorkflowInstance(workflow_id=workflow_id, workflow_type=workflow_type, state=PipelineState.INITIALIZING, created_at=datetime.now(timezone.utc), metadata=parameters)
        self.active_workflows[workflow_id] = workflow_instance
        try:
            await self.workflow_engine.start_workflow(workflow_id, workflow_type, parameters)
            workflow_instance.state = PipelineState.RUNNING
            workflow_instance.started_at = datetime.now(timezone.utc)
            await self._emit_orchestrator_event('WORKFLOW_STARTED', {'workflow_id': workflow_id, 'workflow_type': workflow_type, 'parameters': parameters})
            self.logger.info('Started workflow {workflow_id} of type %s', workflow_type)
            return workflow_id
        except Exception as e:
            workflow_instance.state = PipelineState.ERROR
            workflow_instance.error_message = str(e)
            self.logger.error('Failed to start workflow {workflow_id}: %s', e)
            raise

    async def stop_workflow(self, workflow_id: str) -> None:
        """Stop a running workflow."""
        if workflow_id not in self.active_workflows:
            raise ValueError(f'Workflow {workflow_id} not found')
        workflow_instance = self.active_workflows[workflow_id]
        try:
            await self.workflow_engine.stop_workflow(workflow_id)
            workflow_instance.state = PipelineState.COMPLETED
            workflow_instance.completed_at = datetime.now(timezone.utc)
            await self._emit_orchestrator_event('WORKFLOW_STOPPED', {'workflow_id': workflow_id})
            self.logger.info('Stopped workflow %s', workflow_id)
        except Exception as e:
            workflow_instance.state = PipelineState.ERROR
            workflow_instance.error_message = str(e)
            self.logger.error('Failed to stop workflow {workflow_id}: %s', e)
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the orchestrator and all components."""
        try:
            health_results = {'healthy': True, 'status': 'healthy', 'orchestrator_version': '2025.1-refactored', 'timestamp': datetime.now(timezone.utc).isoformat(), 'components': {}, 'external_services': {}, 'active_workflows': len(self.active_workflows)}
            component_health = await self.get_component_health()
            health_results['components'] = component_health
            if self.health_monitor:
                try:
                    service_health = await self.health_monitor.check_all_services()
                    health_results['external_services'] = service_health
                except Exception as e:
                    self.logger.error('Health monitor check failed: %s', e)
                    health_results['external_services'] = {'error': str(e)}
            try:
                resource_usage = await self.resource_manager.get_usage_stats()
                health_results['resource_usage'] = resource_usage
                memory_ok = resource_usage.get('memory_usage_percent', 0) < 90
                if not memory_ok:
                    health_results['healthy'] = False
                    health_results['status'] = 'degraded'
            except Exception as e:
                self.logger.error('Resource usage check failed: %s', e)
                health_results['resource_usage'] = {'error': str(e)}
                health_results['healthy'] = False
                health_results['status'] = 'degraded'
            all_components_healthy = all(component_health.values()) if component_health else True
            if not all_components_healthy:
                health_results['healthy'] = False
                health_results['status'] = 'degraded'
            return health_results
        except Exception as e:
            self.logger.error('Health check failed: %s', e)
            return {'healthy': False, 'status': 'error', 'error': str(e), 'timestamp': datetime.now(timezone.utc).isoformat()}

    async def get_component_health(self) -> Dict[str, bool]:
        """Get health status of all registered components."""
        components = await self.component_registry.list_components()
        health_status = {}
        for component in components:
            health_status[component.name] = component.status.value in ['healthy', 'starting']
        return health_status

    async def invoke_component(self, component_name: str, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a method on a loaded ML component through the injected invoker."""
        if not self._is_initialized:
            raise RuntimeError('Orchestrator not initialized')
        result = await self.component_invoker.invoke_component_method(component_name, method_name, *args, **kwargs)
        if not result.success:
            self.logger.error('Component invocation failed: %s', result.error)
            raise RuntimeError(f'Component {component_name}.{method_name} failed: {result.error}')
        return result.result

    async def run_training_workflow(self, training_data: Any) -> Dict[str, Any]:
        """Run a complete training workflow using the injected component invoker."""
        self.logger.info('Running training workflow (Refactored)')
        try:
            results = await self.component_invoker.invoke_training_workflow(training_data)
            for step, result in results.items():
                if result.success:
                    self.logger.info("Training step '%s' completed successfully", step)
                else:
                    self.logger.error("Training step '{step}' failed: %s", result.error)
            return {step: result.result for step, result in results.items() if result.success}
        except Exception as e:
            self.logger.error('Training workflow failed: %s', e)
            raise

    async def run_evaluation_workflow(self, evaluation_data: Any) -> Dict[str, Any]:
        """Run a complete evaluation workflow using the injected component invoker."""
        self.logger.info('Running evaluation workflow (Refactored)')
        try:
            results = await self.component_invoker.invoke_evaluation_workflow(evaluation_data)
            for step, result in results.items():
                if result.success:
                    self.logger.info("Evaluation step '%s' completed successfully", step)
                else:
                    self.logger.error("Evaluation step '{step}' failed: %s", result.error)
            return {step: result.result for step, result in results.items() if result.success}
        except Exception as e:
            self.logger.error('Evaluation workflow failed: %s', e)
            raise

    def get_loaded_components(self) -> List[str]:
        """Get list of loaded component names."""
        return list(self.component_loader.get_all_loaded_components().keys())

    def get_component_methods(self, component_name: str) -> List[str]:
        """Get available methods for a component."""
        return self.component_invoker.get_available_methods(component_name)

    def get_invocation_history(self, component_name: Optional[str]=None) -> List[Dict[str, Any]]:
        """Get component invocation history."""
        history = self.component_invoker.get_invocation_history(component_name)
        return [{'component_name': result.component_name, 'method_name': result.method_name, 'success': result.success, 'execution_time': result.execution_time, 'timestamp': result.timestamp.isoformat(), 'error': result.error} for result in history]

    def _setup_event_handlers(self) -> None:
        """Setup event handlers for orchestrator events."""
        self.event_bus.subscribe('COMPONENT_HEALTH_CHANGED', self._handle_component_health)
        self.event_bus.subscribe('WORKFLOW_COMPLETED', self._handle_workflow_completed)
        self.event_bus.subscribe('WORKFLOW_FAILED', self._handle_workflow_failed)
        self.event_bus.subscribe('RESOURCE_EXHAUSTED', self._handle_resource_exhausted)

    async def _emit_orchestrator_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an orchestrator event through the injected event bus."""
        await self.event_bus.publish(event_type=event_type, data={'source': 'ml_pipeline_orchestrator_refactored', **data})

    async def _handle_component_health(self, event: Any) -> None:
        """Handle component health change events."""
        component_name = event.data.get('component_name')
        is_healthy = event.data.get('is_healthy', False)
        if component_name:
            self.component_health[component_name] = is_healthy
            self.logger.info('Component {component_name} health: %s', is_healthy)

    async def _handle_workflow_completed(self, event: Any) -> None:
        """Handle workflow completion events."""
        workflow_id = event.data.get('workflow_id')
        if workflow_id in self.active_workflows:
            workflow_instance = self.active_workflows[workflow_id]
            workflow_instance.state = PipelineState.COMPLETED
            workflow_instance.completed_at = datetime.now(timezone.utc)
            self.logger.info('Workflow %s completed successfully', workflow_id)

    async def _handle_workflow_failed(self, event: Any) -> None:
        """Handle workflow failure events."""
        workflow_id = event.data.get('workflow_id')
        error_message = event.data.get('error_message', 'Unknown error')
        if workflow_id in self.active_workflows:
            workflow_instance = self.active_workflows[workflow_id]
            workflow_instance.state = PipelineState.ERROR
            workflow_instance.error_message = error_message
            workflow_instance.completed_at = datetime.now(timezone.utc)
            self.logger.error('Workflow {workflow_id} failed: %s', error_message)

    async def _handle_resource_exhausted(self, event: Any) -> None:
        """Handle resource exhaustion events."""
        resource_type = event.data.get('resource_type')
        self.logger.warning('Resource exhausted: %s', resource_type)
        await self.resource_manager.handle_resource_exhaustion(resource_type)

    async def _discover_components(self) -> None:
        """Discover and register all ML components."""
        self.logger.info('Discovering ML components across all tiers')
        components = await self.component_registry.discover_components()
        for component in components:
            self.component_health[component.name] = False
        self.logger.info('Discovered %s ML components', len(components))

    async def _load_direct_components(self) -> None:
        """Load ML components directly using the injected component loader."""
        self.logger.info('Loading direct ML components (Refactored)')
        try:
            loaded_components = await self.component_loader.load_all_components()
            self.logger.info('Successfully loaded %s ML components', len(loaded_components))
            core_components = ['training_data_loader', 'ml_integration', 'rule_optimizer', 'multi_armed_bandit', 'batch_processor', 'apes_service_manager', 'unified_retry_manager', 'input_sanitizer', 'memory_guard', 'apriori_analyzer', 'context_learner']
            for component_name in core_components:
                if component_name in loaded_components:
                    success = await self.component_loader.initialize_component(component_name)
                    if success:
                        self.logger.info('Initialized core component: %s', component_name)
                    else:
                        self.logger.warning('Failed to initialize core component: %s', component_name)
            await self._emit_orchestrator_event('COMPONENT_REGISTERED', {'loaded_components': len(loaded_components), 'initialized_components': len(core_components)})
        except Exception as e:
            self.logger.error('Failed to load direct components: %s', e)
            raise

    async def _setup_health_monitoring(self) -> None:
        """Setup health monitoring using the injected health monitor."""
        if not self.health_monitor:
            return
        try:
            await self.health_monitor.initialize()
            self.logger.info('Health monitoring setup complete')
        except Exception as e:
            self.logger.warning('Failed to setup health monitoring: %s', e)

    async def _setup_monitoring(self) -> None:
        """Setup general monitoring for the orchestrator."""
        self.logger.info('Setting up orchestrator monitoring (Refactored)')

    async def _stop_all_workflows(self) -> None:
        """Stop all active workflows during shutdown."""
        for workflow_id in list(self.active_workflows.keys()):
            try:
                await self.stop_workflow(workflow_id)
            except Exception as e:
                self.logger.error('Error stopping workflow {workflow_id}: %s', e)

async def create_orchestrator_with_di(container) -> MLPipelineOrchestratorRefactored:
    """Create a refactored orchestrator using dependency injection.
    
    This demonstrates how the factory would create the orchestrator with all
    dependencies properly injected.
    
    Args:
        container: DI container with registered services
        
    Returns:
        Configured orchestrator with all dependencies injected
    """
    config = await container.get(ExternalServicesConfigProtocol)
    event_bus = await container.get(EventBusProtocol)
    workflow_engine = await container.get(WorkflowEngineProtocol)
    resource_manager = await container.get(ResourceManagerProtocol)
    component_registry = await container.get(ComponentRegistryProtocol)
    component_loader = await container.get(ComponentLoaderProtocol)
    component_invoker = await container.get(ComponentInvokerProtocol)
    health_monitor = None
    try:
        health_monitor = await container.get(HealthMonitorProtocol)
    except:
        pass
    orchestrator = MLPipelineOrchestratorRefactored(config=config, event_bus=event_bus, workflow_engine=workflow_engine, resource_manager=resource_manager, component_registry=component_registry, component_loader=component_loader, component_invoker=component_invoker, health_monitor=health_monitor)
    return orchestrator
