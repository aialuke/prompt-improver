"""
Central ML Pipeline Orchestrator

Main orchestrator class that coordinates all ML components in the prompt-improver system.
Implements hybrid central orchestration following Kubeflow patterns.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone

from .workflow_execution_engine import WorkflowExecutionEngine
from .resource_manager import ResourceManager
from .component_registry import ComponentRegistry
from ..events.event_bus import EventBus
from ..events.event_types import EventType, MLEvent
from ..config.orchestrator_config import OrchestratorConfig
from ..integration.direct_component_loader import DirectComponentLoader
from ..integration.component_invoker import ComponentInvoker

class PipelineState(Enum):
    """Pipeline execution states."""
    idle = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    COMPLETED = "completed"

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

class MLPipelineOrchestrator:
    """
    Central ML Pipeline Orchestrator

    Coordinates all ML components across 6 tiers:
    - Tier 1: Core ML Pipeline (11 components)
    - Tier 2: Optimization & Learning (8 components)
    - Tier 3: Evaluation & Analysis (10 components)
    - Tier 4: Performance & Testing (8 components)
    - Tier 5: Model & Infrastructure (6 components)
    - Tier 6: Security & Advanced (7+ components)
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """Initialize the ML Pipeline Orchestrator."""
        self.config = config or OrchestratorConfig()
        self.logger = logging.getLogger(__name__)

        # Core components
        self.event_bus = EventBus(self.config)
        self.workflow_engine = WorkflowExecutionEngine(self.config)
        self.resource_manager = ResourceManager(self.config)
        self.component_registry = ComponentRegistry(self.config)

        # Direct component integration (Phase 6)
        self.component_loader = DirectComponentLoader()
        self.component_invoker = ComponentInvoker(self.component_loader)

        # Retry management for resilient operations
        self.retry_manager = None  # Will be initialized during startup

        # Security input validation for all operations
        self.input_sanitizer = None  # Will be initialized during startup

        # Memory management and resource monitoring for all operations
        self.memory_guard = None  # Will be initialized during startup

        # Set event bus references
        self.workflow_engine.set_event_bus(self.event_bus)
        self.resource_manager.set_event_bus(self.event_bus)

        # State management
        self.state = PipelineState.idle
        self.active_workflows: Dict[str, WorkflowInstance] = {}
        self.component_health: Dict[str, bool] = {}

        # Event handlers
        self._setup_event_handlers()

        # Startup flag
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the orchestrator and all subsystems."""
        if self._is_initialized:
            return

        self.logger.info("Initializing ML Pipeline Orchestrator")
        self.state = PipelineState.INITIALIZING

        try:
            # Initialize core components
            await self.component_registry.initialize()
            await self.resource_manager.initialize()
            await self.workflow_engine.initialize()
            await self.event_bus.initialize()

            # Discover and register components
            await self._discover_components()

            # Direct component loading (Phase 6)
            await self._load_direct_components()

            # Setup monitoring
            await self._setup_monitoring()

            self.state = PipelineState.idle
            self._is_initialized = True

            # Emit initialization complete event
            await self.event_bus.emit(MLEvent(
                event_type=EventType.ORCHESTRATOR_INITIALIZED,
                source="ml_pipeline_orchestrator",
                data={"timestamp": datetime.now(timezone.utc).isoformat()}
            ))

            self.logger.info("ML Pipeline Orchestrator initialized successfully")

        except Exception as e:
            self.state = PipelineState.ERROR
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            raise

    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator."""
        self.logger.info("Shutting down ML Pipeline Orchestrator")
        self.state = PipelineState.STOPPING

        try:
            # Stop all active workflows
            await self._stop_all_workflows()

            # Shutdown components
            await self.workflow_engine.shutdown()
            await self.resource_manager.shutdown()
            await self.event_bus.shutdown()
            await self.component_registry.shutdown()

            self.state = PipelineState.idle
            self._is_initialized = False

            self.logger.info("ML Pipeline Orchestrator shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during orchestrator shutdown: {e}")
            raise

    async def start_workflow(self, workflow_type: str, parameters: Dict[str, Any]) -> str:
        """Start a new ML workflow."""
        if not self._is_initialized:
            raise RuntimeError("Orchestrator not initialized")

        if self.state != PipelineState.idle:
            raise RuntimeError(f"Cannot start workflow in state: {self.state}")

        # Generate workflow ID
        workflow_id = f"{workflow_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        # Create workflow instance
        workflow_instance = WorkflowInstance(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            state=PipelineState.INITIALIZING,
            created_at=datetime.now(timezone.utc),
            metadata=parameters
        )

        self.active_workflows[workflow_id] = workflow_instance

        try:
            # Start workflow execution
            await self.workflow_engine.start_workflow(workflow_id, workflow_type, parameters)

            workflow_instance.state = PipelineState.RUNNING
            workflow_instance.started_at = datetime.now(timezone.utc)

            # Emit workflow started event
            await self.event_bus.emit(MLEvent(
                event_type=EventType.WORKFLOW_STARTED,
                source="ml_pipeline_orchestrator",
                data={
                    "workflow_id": workflow_id,
                    "workflow_type": workflow_type,
                    "parameters": parameters
                }
            ))

            self.logger.info(f"Started workflow {workflow_id} of type {workflow_type}")
            return workflow_id

        except Exception as e:
            workflow_instance.state = PipelineState.ERROR
            workflow_instance.error_message = str(e)
            self.logger.error(f"Failed to start workflow {workflow_id}: {e}")
            raise

    async def stop_workflow(self, workflow_id: str) -> None:
        """Stop a running workflow."""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow_instance = self.active_workflows[workflow_id]

        try:
            await self.workflow_engine.stop_workflow(workflow_id)

            workflow_instance.state = PipelineState.COMPLETED
            workflow_instance.completed_at = datetime.now(timezone.utc)

            # Emit workflow stopped event
            await self.event_bus.emit(MLEvent(
                event_type=EventType.WORKFLOW_STOPPED,
                source="ml_pipeline_orchestrator",
                data={"workflow_id": workflow_id}
            ))

            self.logger.info(f"Stopped workflow {workflow_id}")

        except Exception as e:
            workflow_instance.state = PipelineState.ERROR
            workflow_instance.error_message = str(e)
            self.logger.error(f"Failed to stop workflow {workflow_id}: {e}")
            raise

    async def get_workflow_status(self, workflow_id: str) -> WorkflowInstance:
        """Get the status of a workflow."""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        return self.active_workflows[workflow_id]

    async def list_workflows(self) -> List[WorkflowInstance]:
        """List all workflows."""
        return list(self.active_workflows.values())

    async def get_component_health(self) -> Dict[str, bool]:
        """Get health status of all registered components."""
        # Get current health from component registry
        components = await self.component_registry.list_components()
        health_status = {}

        for component in components:
            # Convert component status to boolean health status
            health_status[component.name] = component.status.value in ["healthy", "starting"]

        return health_status

    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check of the orchestrator and all components.

        Returns:
            Dictionary with overall health status and component details
        """
        try:
            # Get component health
            component_health = await self.get_component_health()

            # Get resource usage
            resource_usage = await self.get_resource_usage()

            # Check active workflows
            active_workflows = len(self.active_workflows)

            # Determine overall health
            all_components_healthy = all(component_health.values()) if component_health else True
            resource_ok = resource_usage.get("memory_usage_percent", 0) < 90  # Less than 90% memory usage

            overall_healthy = all_components_healthy and resource_ok

            return {
                "healthy": overall_healthy,
                "status": "healthy" if overall_healthy else "degraded",
                "components": component_health,
                "resource_usage": resource_usage,
                "active_workflows": active_workflows,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "orchestrator_version": "2025.1"
            }

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage statistics."""
        return await self.resource_manager.get_usage_stats()

    def _setup_event_handlers(self) -> None:
        """Setup event handlers for orchestrator events."""
        # Component health events
        self.event_bus.subscribe(EventType.COMPONENT_HEALTH_CHANGED, self._handle_component_health)

        # Workflow events
        self.event_bus.subscribe(EventType.WORKFLOW_COMPLETED, self._handle_workflow_completed)
        self.event_bus.subscribe(EventType.WORKFLOW_FAILED, self._handle_workflow_failed)

        # Resource events
        self.event_bus.subscribe(EventType.RESOURCE_EXHAUSTED, self._handle_resource_exhausted)

    async def _handle_component_health(self, event: MLEvent) -> None:
        """Handle component health change events."""
        component_name = event.data.get("component_name")
        is_healthy = event.data.get("is_healthy", False)

        if component_name:
            self.component_health[component_name] = is_healthy
            self.logger.info(f"Component {component_name} health: {is_healthy}")

    async def _handle_workflow_completed(self, event: MLEvent) -> None:
        """Handle workflow completion events."""
        workflow_id = event.data.get("workflow_id")
        if workflow_id in self.active_workflows:
            workflow_instance = self.active_workflows[workflow_id]
            workflow_instance.state = PipelineState.COMPLETED
            workflow_instance.completed_at = datetime.now(timezone.utc)

            self.logger.info(f"Workflow {workflow_id} completed successfully")

    async def _handle_workflow_failed(self, event: MLEvent) -> None:
        """Handle workflow failure events."""
        workflow_id = event.data.get("workflow_id")
        error_message = event.data.get("error_message", "Unknown error")

        if workflow_id in self.active_workflows:
            workflow_instance = self.active_workflows[workflow_id]
            workflow_instance.state = PipelineState.ERROR
            workflow_instance.error_message = error_message
            workflow_instance.completed_at = datetime.now(timezone.utc)

            self.logger.error(f"Workflow {workflow_id} failed: {error_message}")

    async def _handle_resource_exhausted(self, event: MLEvent) -> None:
        """Handle resource exhaustion events."""
        resource_type = event.data.get("resource_type")
        self.logger.warning(f"Resource exhausted: {resource_type}")

        # Implement resource management strategies
        await self.resource_manager.handle_resource_exhaustion(resource_type)

    async def _discover_components(self) -> None:
        """Discover and register all ML components."""
        self.logger.info("Discovering ML components across all tiers")

        # This will be implemented to discover components from all 6 tiers
        # For now, we'll initialize the component health tracking
        components = await self.component_registry.discover_components()

        for component in components:
            self.component_health[component.name] = False  # Initially unhealthy until verified

        self.logger.info(f"Discovered {len(components)} ML components")

    async def _setup_monitoring(self) -> None:
        """Setup monitoring for the orchestrator."""
        # This will be implemented to setup health checks and metrics collection
        self.logger.info("Setting up orchestrator monitoring")

    async def _stop_all_workflows(self) -> None:
        """Stop all active workflows during shutdown."""
        for workflow_id in list(self.active_workflows.keys()):
            try:
                await self.stop_workflow(workflow_id)
            except Exception as e:
                self.logger.error(f"Error stopping workflow {workflow_id}: {e}")

    async def _load_direct_components(self) -> None:
        """Load ML components directly for Phase 6 integration."""
        self.logger.info("Loading direct ML components (Phase 6)")

        try:
            # Load all components across all tiers
            loaded_components = await self.component_loader.load_all_components()

            self.logger.info(f"Successfully loaded {len(loaded_components)} ML components")

            # Initialize core components for immediate use
            core_components = [
                "training_data_loader", "ml_integration", "rule_optimizer",
                "multi_armed_bandit", "batch_processor", "apes_service_manager",
                "unified_retry_manager", "input_sanitizer", "memory_guard",
                "apriori_analyzer", "context_learner"
            ]

            # Initialize evaluation components for workflows
            evaluation_components = [
                "statistical_analyzer", "advanced_statistical_validator",
                "causal_inference_analyzer", "pattern_significance_analyzer",
                "structural_analyzer"
            ]

            for component_name in core_components:
                if component_name in loaded_components:
                    # Special initialization for APESServiceManager with event bus
                    if component_name == "apes_service_manager":
                        success = await self.component_loader.initialize_component(
                            component_name,
                            event_bus=self.event_bus
                        )
                    else:
                        success = await self.component_loader.initialize_component(component_name)

                    if success:
                        self.logger.info(f"Initialized core component: {component_name}")

                        # Set retry manager reference for orchestrator use
                        if component_name == "unified_retry_manager":
                            retry_component = self.component_loader.get_loaded_component("unified_retry_manager")
                            if retry_component and retry_component.instance:
                                self.retry_manager = retry_component.instance
                                # Integrate retry manager with component invoker and workflow engine
                                self.component_invoker.set_retry_manager(self.retry_manager)
                                self.workflow_engine.set_retry_manager(self.retry_manager)
                                self.logger.info("RetryManager integrated with orchestrator, component invoker, and workflow engine")

                        # Set input sanitizer reference for orchestrator use
                        elif component_name == "input_sanitizer":
                            sanitizer_component = self.component_loader.get_loaded_component("input_sanitizer")
                            if sanitizer_component and sanitizer_component.instance:
                                self.input_sanitizer = sanitizer_component.instance
                                # Integrate input sanitizer with event bus for security monitoring
                                self.input_sanitizer.set_event_bus(self.event_bus)
                                # Integrate input sanitizer with component invoker and workflow engine
                                self.component_invoker.set_input_sanitizer(self.input_sanitizer)
                                self.workflow_engine.set_input_sanitizer(self.input_sanitizer)
                                self.logger.info("InputSanitizer integrated with orchestrator, event bus, component invoker, and workflow engine")

                        # Set memory guard reference for orchestrator use
                        elif component_name == "memory_guard":
                            memory_guard_component = self.component_loader.get_loaded_component("memory_guard")
                            if memory_guard_component and memory_guard_component.instance:
                                self.memory_guard = memory_guard_component.instance
                                # Integrate memory guard with event bus for resource monitoring
                                self.memory_guard.set_event_bus(self.event_bus)
                                # Integrate memory guard with component invoker and workflow engine
                                self.component_invoker.set_memory_guard(self.memory_guard)
                                self.workflow_engine.set_memory_guard(self.memory_guard)
                                self.logger.info("MemoryGuard integrated with orchestrator, event bus, component invoker, and workflow engine")
                    else:
                        self.logger.warning(f"Failed to initialize core component: {component_name}")

            # Initialize evaluation components for workflows
            for component_name in evaluation_components:
                if component_name in loaded_components:
                    success = await self.component_loader.initialize_component(component_name)
                    if success:
                        self.logger.info(f"Initialized evaluation component: {component_name}")
                    else:
                        self.logger.warning(f"Failed to initialize evaluation component: {component_name}")

            # Emit component loading complete event
            await self.event_bus.emit(MLEvent(
                event_type=EventType.COMPONENT_REGISTERED,
                source="ml_pipeline_orchestrator",
                data={
                    "loaded_components": len(loaded_components),
                    "initialized_components": len(core_components),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ))

        except Exception as e:
            self.logger.error(f"Failed to load direct components: {e}")
            raise

    async def invoke_component(self, component_name: str, method_name: str, *args, **kwargs) -> Any:
        """
        Invoke a method on a loaded ML component.

        Args:
            component_name: Name of the component
            method_name: Method to invoke
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of the component method invocation
        """
        if not self._is_initialized:
            raise RuntimeError("Orchestrator not initialized")

        result = await self.component_invoker.invoke_component_method(
            component_name, method_name, *args, **kwargs
        )

        if not result.success:
            self.logger.error(f"Component invocation failed: {result.error}")
            raise RuntimeError(f"Component {component_name}.{method_name} failed: {result.error}")

        return result.result

    async def run_training_workflow(self, training_data: Any) -> Dict[str, Any]:
        """
        Run a complete training workflow using direct component integration.

        Args:
            training_data: Input training data

        Returns:
            Dictionary of workflow results
        """
        self.logger.info("Running direct training workflow")

        try:
            results = await self.component_invoker.invoke_training_workflow(training_data)

            # Log results
            for step, result in results.items():
                if result.success:
                    self.logger.info(f"Training step '{step}' completed successfully")
                else:
                    self.logger.error(f"Training step '{step}' failed: {result.error}")

            return {step: result.result for step, result in results.items() if result.success}

        except Exception as e:
            self.logger.error(f"Training workflow failed: {e}")
            raise

    async def execute_with_retry(self, operation, operation_name: str, **retry_config_kwargs):
        """
        Execute an operation with unified retry logic.

        Args:
            operation: Async operation to execute with retry
            operation_name: Name for observability and logging
            **retry_config_kwargs: Additional retry configuration parameters

        Returns:
            Result of successful operation

        Raises:
            RuntimeError: If retry manager not available
            Exception: Last exception if all retries failed
        """
        if not self.retry_manager:
            self.logger.warning(f"Retry manager not available for operation: {operation_name}")
            # Fallback to direct execution
            return await operation()

        from ....core.retry_manager import RetryConfig

        # Create retry configuration
        config = RetryConfig(
            operation_name=operation_name,
            **retry_config_kwargs
        )

        return await self.retry_manager.retry_async(operation, config=config)

    async def validate_input_secure(self, input_data: Any, context: Dict[str, Any] = None):
        """
        Validate input data using integrated security sanitizer.

        Args:
            input_data: Data to validate
            context: Validation context (user_id, source_ip, etc.)

        Returns:
            ValidationResult with security assessment

        Raises:
            RuntimeError: If input sanitizer not available
            SecurityError: If critical security threats detected
        """
        if not self.input_sanitizer:
            self.logger.warning("Input sanitizer not available, skipping validation")
            # Return a basic validation result
            from ..security.input_sanitization import ValidationResult, SecurityThreatLevel
            return ValidationResult(
                is_valid=True,
                sanitized_value=input_data,
                threat_level=SecurityThreatLevel.LOW,
                message="Input sanitizer not available"
            )

        # Perform comprehensive security validation
        result = await self.input_sanitizer.validate_input_async(input_data, context)

        # Log security validation results
        if result.threats_detected:
            self.logger.warning(f"Security threats detected in input: {result.threats_detected}")

        # Raise exception for critical threats
        from prompt_improver.security.input_sanitization import SecurityError, SecurityThreatLevel
        if result.threat_level == SecurityThreatLevel.CRITICAL:
            raise SecurityError(f"Critical security threat detected: {result.threats_detected}")

        return result

    async def run_training_workflow_secure(self, training_data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run a complete training workflow with security validation.

        Args:
            training_data: Input training data
            context: Security context (user_id, source_ip, etc.)

        Returns:
            Dictionary of workflow results
        """
        self.logger.info("Running secure training workflow with input validation")

        try:
            # Validate training data for security threats
            validation_result = await self.validate_input_secure(training_data, context)

            if not validation_result.is_valid:
                raise RuntimeError(f"Training data validation failed: {validation_result.message}")

            # Use sanitized data for training
            sanitized_data = validation_result.sanitized_value

            # Run the training workflow with validated data
            results = await self.component_invoker.invoke_training_workflow(sanitized_data)

            # Log results
            for step, result in results.items():
                if result.success:
                    self.logger.info(f"Secure training step '{step}' completed successfully")
                else:
                    self.logger.error(f"Secure training step '{step}' failed: {result.error}")

            return {step: result.result for step, result in results.items() if result.success}

        except Exception as e:
            self.logger.error(f"Secure training workflow failed: {e}")
            raise

    async def monitor_memory_usage(self, operation_name: str = "orchestrator_operation", component_name: str = None):
        """
        Monitor memory usage for orchestrator operations.

        Args:
            operation_name: Name of the operation being monitored
            component_name: Name of the component if applicable

        Returns:
            ResourceStats with current memory information
        """
        if not self.memory_guard:
            self.logger.warning("Memory guard not available, skipping memory monitoring")
            return None

        return await self.memory_guard.check_memory_usage_async(operation_name, component_name)

    def monitor_operation_memory(self, operation_name: str, component_name: str = None):
        """
        Async context manager for monitoring memory during orchestrator operations.

        Args:
            operation_name: Name of the operation to monitor
            component_name: Name of the component if applicable

        Returns:
            AsyncMemoryMonitor context manager
        """
        if not self.memory_guard:
            self.logger.warning("Memory guard not available for operation monitoring")
            # Return a no-op context manager
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def noop_monitor():
                yield None

            return noop_monitor()

        return self.memory_guard.monitor_operation_async(operation_name, component_name)

    async def validate_operation_memory(self, data: Any, operation_name: str, component_name: str = None) -> bool:
        """
        Validate memory requirements for ML operations.

        Args:
            data: Data to validate for memory requirements
            operation_name: Name of the operation
            component_name: Name of the component performing the operation

        Returns:
            True if memory validation passes

        Raises:
            MemoryError: If memory requirements exceed limits
        """
        if not self.memory_guard:
            self.logger.warning("Memory guard not available, skipping memory validation")
            return True

        return await self.memory_guard.validate_ml_operation_memory(data, operation_name, component_name)

    async def run_training_workflow_with_memory_monitoring(self, training_data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run a complete training workflow with comprehensive memory monitoring.

        Args:
            training_data: Input training data
            context: Security and monitoring context

        Returns:
            Dictionary of workflow results
        """
        self.logger.info("Running training workflow with memory monitoring")

        async with self.monitor_operation_memory("training_workflow", "ml_pipeline_orchestrator"):
            try:
                # Validate memory requirements for training data
                await self.validate_operation_memory(training_data, "training_data_validation", "ml_pipeline_orchestrator")

                # Run secure training workflow with memory monitoring
                if self.input_sanitizer:
                    return await self.run_training_workflow_secure(training_data, context)
                else:
                    return await self.run_training_workflow(training_data)

            except Exception as e:
                self.logger.error(f"Memory-monitored training workflow failed: {e}")
                raise

    async def run_evaluation_workflow(self, evaluation_data: Any) -> Dict[str, Any]:
        """
        Run a complete evaluation workflow using direct component integration.

        Args:
            evaluation_data: Input evaluation data

        Returns:
            Dictionary of workflow results
        """
        self.logger.info("Running direct evaluation workflow")

        try:
            results = await self.component_invoker.invoke_evaluation_workflow(evaluation_data)

            # Log results
            for step, result in results.items():
                if result.success:
                    self.logger.info(f"Evaluation step '{step}' completed successfully")
                else:
                    self.logger.error(f"Evaluation step '{step}' failed: {result.error}")

            return {step: result.result for step, result in results.items() if result.success}

        except Exception as e:
            self.logger.error(f"Evaluation workflow failed: {e}")
            raise

    def get_loaded_components(self) -> List[str]:
        """Get list of loaded component names."""
        return list(self.component_loader.get_all_loaded_components().keys())

    def get_component_methods(self, component_name: str) -> List[str]:
        """Get available methods for a component."""
        return self.component_invoker.get_available_methods(component_name)

    def get_invocation_history(self, component_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get component invocation history."""
        history = self.component_invoker.get_invocation_history(component_name)
        return [
            {
                "component_name": result.component_name,
                "method_name": result.method_name,
                "success": result.success,
                "execution_time": result.execution_time,
                "timestamp": result.timestamp.isoformat(),
                "error": result.error
            }
            for result in history
        ]
