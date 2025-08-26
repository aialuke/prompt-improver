"""Workflow Orchestrator Service.

Focused service responsible for core workflow execution and pipeline coordination.
Handles workflow lifecycle management, training/evaluation workflows, and workflow state.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from ....shared.interfaces.protocols.ml import (
    ComponentInvokerProtocol,
    EventBusProtocol,
    WorkflowEngineProtocol,
)
from ..core.orchestrator_types import PipelineState, WorkflowInstance
from ..core.orchestrator_service_protocols import WorkflowOrchestratorProtocol
from ..events.event_types import EventType, MLEvent


class WorkflowOrchestrator:
    """
    Workflow Orchestrator Service.
    
    Responsible for:
    - Core workflow execution and lifecycle management
    - Training and evaluation workflow coordination
    - Workflow state management and tracking
    """

    def __init__(
        self,
        event_bus: EventBusProtocol,
        workflow_engine: WorkflowEngineProtocol,
        component_invoker: ComponentInvokerProtocol | None = None,
    ):
        """Initialize WorkflowOrchestrator with required dependencies.
        
        Args:
            event_bus: Event bus for inter-component communication
            workflow_engine: Workflow execution engine
            component_invoker: Optional component invoker for workflows
        """
        self.event_bus = event_bus
        self.workflow_engine = workflow_engine
        self.component_invoker = component_invoker
        self.logger = logging.getLogger(__name__)
        
        # Workflow state management
        self.state = PipelineState.idle
        self.active_workflows: dict[str, WorkflowInstance] = {}
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the workflow orchestrator."""
        if self._is_initialized:
            return
        
        self.logger.info("Initializing Workflow Orchestrator")
        
        try:
            # Initialize workflow engine
            await self.workflow_engine.initialize()
            
            # Setup event handlers
            self._setup_event_handlers()
            
            self.state = PipelineState.idle
            self._is_initialized = True
            
            # Emit initialization event
            await self.event_bus.emit(MLEvent(
                event_type=EventType.ORCHESTRATOR_INITIALIZED,
                source="workflow_orchestrator", 
                data={"component": "workflow_orchestrator", "timestamp": datetime.now(timezone.utc).isoformat()}
            ))
            
            self.logger.info("Workflow Orchestrator initialized successfully")
            
        except Exception as e:
            self.state = PipelineState.ERROR
            self.logger.error(f"Failed to initialize workflow orchestrator: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the workflow orchestrator."""
        self.logger.info("Shutting down Workflow Orchestrator")
        self.state = PipelineState.STOPPING
        
        try:
            # Stop all active workflows
            await self._stop_all_workflows()
            
            # Shutdown workflow engine
            await self.workflow_engine.shutdown()
            
            self.state = PipelineState.idle
            self._is_initialized = False
            
            self.logger.info("Workflow Orchestrator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during workflow orchestrator shutdown: {e}")
            raise

    async def start_workflow(self, workflow_type: str, parameters: dict[str, Any]) -> str:
        """Start a new ML workflow."""
        if not self._is_initialized:
            raise RuntimeError("Workflow orchestrator not initialized")
        
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
                source="workflow_orchestrator",
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
                source="workflow_orchestrator",
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

    async def list_workflows(self) -> list[WorkflowInstance]:
        """List all workflows."""
        return list(self.active_workflows.values())

    async def run_training_workflow(self, training_data: Any) -> dict[str, Any]:
        """
        Run a complete training workflow using direct component integration.
        
        Args:
            training_data: Input training data
            
        Returns:
            Dictionary of workflow results
        """
        if not self.component_invoker:
            raise RuntimeError("Component invoker not available for training workflow")
        
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

    async def run_evaluation_workflow(self, evaluation_data: Any) -> dict[str, Any]:
        """
        Run a complete evaluation workflow using direct component integration.
        
        Args:
            evaluation_data: Input evaluation data
            
        Returns:
            Dictionary of workflow results
        """
        if not self.component_invoker:
            raise RuntimeError("Component invoker not available for evaluation workflow")
        
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

    def _setup_event_handlers(self) -> None:
        """Setup event handlers for workflow events."""
        # Workflow completion events
        self.event_bus.subscribe(EventType.WORKFLOW_COMPLETED, self._handle_workflow_completed)
        self.event_bus.subscribe(EventType.WORKFLOW_FAILED, self._handle_workflow_failed)

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

    async def _stop_all_workflows(self) -> None:
        """Stop all active workflows during shutdown."""
        for workflow_id in list(self.active_workflows.keys()):
            try:
                await self.stop_workflow(workflow_id)
            except Exception as e:
                self.logger.error(f"Error stopping workflow {workflow_id}: {e}")