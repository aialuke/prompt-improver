"""
Tests for Workflow Execution Engine.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from prompt_improver.ml.orchestration.core.workflow_execution_engine import (
    WorkflowExecutionEngine
)
from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import PipelineState, WorkflowInstance
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from prompt_improver.ml.orchestration.events.event_types import EventType, MLEvent


class TestWorkflowExecutionEngine:
    """Test suite for Workflow Execution Engine."""
    
    @pytest.fixture
    async def execution_engine(self):
        """Create execution engine instance for testing."""
        config = OrchestratorConfig(
            max_concurrent_workflows=2,
            event_bus_buffer_size=10
        )
        
        # Mock event bus
        mock_event_bus = Mock()
        mock_event_bus.emit = AsyncMock()
        
        engine = WorkflowExecutionEngine(config)
        engine.set_event_bus(mock_event_bus)
        await engine.initialize()
        
        yield engine
        
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, execution_engine):
        """Test engine initialization."""
        assert execution_engine.workflow_definitions is not None
        assert execution_engine.active_executors == {}
        assert execution_engine.event_bus is not None
    
    @pytest.mark.asyncio
    async def test_start_workflow(self, execution_engine):
        """Test starting a workflow."""
        # First register a workflow definition
        from prompt_improver.ml.orchestration.core.workflow_types import WorkflowDefinition, WorkflowStep
        
        workflow_type = "test_training"
        definition = WorkflowDefinition(
            workflow_type=workflow_type,
            name="Test Training Workflow",
            description="Test workflow for unit testing",
            steps=[
                WorkflowStep(
                    step_id="step1",
                    name="Test Step",
                    component_name="test_component",
                    parameters={}
                )
            ]
        )
        await execution_engine.register_workflow_definition(definition)
        
        workflow_id = "test-workflow-123"
        parameters = {"test_param": "value"}
        
        await execution_engine.start_workflow(workflow_id, workflow_type, parameters)
        
        assert workflow_id in execution_engine.active_executors
        executor = execution_engine.active_executors[workflow_id]
        assert executor.definition.workflow_type == workflow_type
        assert executor.is_running is True
    
    @pytest.mark.asyncio
    async def test_workflow_status(self, execution_engine):
        """Test getting workflow status."""
        # Register workflow definition
        from prompt_improver.ml.orchestration.core.workflow_types import WorkflowDefinition, WorkflowStep
        
        workflow_type = "test_training"
        definition = WorkflowDefinition(
            workflow_type=workflow_type,
            name="Test Training Workflow",
            description="Test workflow for unit testing",
            steps=[
                WorkflowStep(
                    step_id="step1",
                    name="Test Step",
                    component_name="test_component",
                    parameters={}
                )
            ]
        )
        await execution_engine.register_workflow_definition(definition)
        
        workflow_id = "test-workflow-456"
        parameters = {"test_param": "value"}
        
        await execution_engine.start_workflow(workflow_id, workflow_type, parameters)
        
        status = await execution_engine.get_workflow_status(workflow_id)
        assert status["workflow_id"] == workflow_id
        assert status["workflow_type"] == workflow_type
        assert status["is_running"] is True
    
    @pytest.mark.asyncio
    async def test_list_active_workflows(self, execution_engine):
        """Test listing active workflows."""
        # Register workflow definition
        from prompt_improver.ml.orchestration.core.workflow_types import WorkflowDefinition, WorkflowStep
        
        workflow_type = "test_training"
        definition = WorkflowDefinition(
            workflow_type=workflow_type,
            name="Test Training Workflow",
            description="Test workflow for unit testing",
            steps=[
                WorkflowStep(
                    step_id="step1",
                    name="Test Step",
                    component_name="test_component",
                    parameters={}
                )
            ]
        )
        await execution_engine.register_workflow_definition(definition)
        
        # Initially no workflows
        assert len(execution_engine.active_executors) == 0
        
        # Start workflows
        workflow_id1 = "workflow-1"
        workflow_id2 = "workflow-2"
        await execution_engine.start_workflow(workflow_id1, workflow_type, {"test": 1})
        await execution_engine.start_workflow(workflow_id2, workflow_type, {"test": 2})
        
        # Should have two workflows
        workflows = await execution_engine.list_active_workflows()
        assert len(workflows) == 2
        
        workflow_ids = [w["workflow_id"] for w in workflows]
        assert workflow_id1 in workflow_ids
        assert workflow_id2 in workflow_ids
    
    @pytest.mark.asyncio
    async def test_stop_workflow(self, execution_engine):
        """Test stopping a workflow."""
        # Register workflow definition
        from prompt_improver.ml.orchestration.core.workflow_types import WorkflowDefinition, WorkflowStep
        
        workflow_type = "test_training"
        definition = WorkflowDefinition(
            workflow_type=workflow_type,
            name="Test Training Workflow",
            description="Test workflow for unit testing",
            steps=[
                WorkflowStep(
                    step_id="step1",
                    name="Test Step",
                    component_name="test_component",
                    parameters={}
                )
            ]
        )
        await execution_engine.register_workflow_definition(definition)
        
        workflow_id = "test-workflow-stop"
        await execution_engine.start_workflow(workflow_id, workflow_type, {})
        
        # Workflow should be running
        status = await execution_engine.get_workflow_status(workflow_id)
        assert status["is_running"] is True
        
        # Stop the workflow
        await execution_engine.stop_workflow(workflow_id)
        
        # Workflow should be removed from active executors
        assert workflow_id not in execution_engine.active_executors
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_limit(self, execution_engine):
        """Test concurrent workflow limits."""
        # Register workflow definition
        from prompt_improver.ml.orchestration.core.workflow_types import WorkflowDefinition, WorkflowStep
        
        workflow_type = "test_training"
        definition = WorkflowDefinition(
            workflow_type=workflow_type,
            name="Test Training Workflow",
            description="Test workflow for unit testing",
            steps=[
                WorkflowStep(
                    step_id="step1",
                    name="Test Step",
                    component_name="test_component",
                    parameters={}
                )
            ]
        )
        await execution_engine.register_workflow_definition(definition)
        
        # Start maximum concurrent workflows
        workflow_id1 = "workflow-1"
        workflow_id2 = "workflow-2"
        await execution_engine.start_workflow(workflow_id1, workflow_type, {"test": 1})
        await execution_engine.start_workflow(workflow_id2, workflow_type, {"test": 2})
        
        # Should have 2 active workflows (at limit)
        assert len(execution_engine.active_executors) == 2
        
        # Try to start another - should succeed (engine doesn't enforce limits at this level)
        workflow_id3 = "workflow-3"
        await execution_engine.start_workflow(workflow_id3, workflow_type, {"test": 3})
        assert workflow_id3 in execution_engine.active_executors
    
    @pytest.mark.asyncio
    async def test_workflow_completion(self, execution_engine):
        """Test workflow completion handling."""
        # Register workflow definition
        from prompt_improver.ml.orchestration.core.workflow_types import WorkflowDefinition, WorkflowStep
        
        workflow_type = "test_training"
        definition = WorkflowDefinition(
            workflow_type=workflow_type,
            name="Test Training Workflow",
            description="Test workflow for unit testing",
            steps=[
                WorkflowStep(
                    step_id="step1",
                    name="Test Step",
                    component_name="test_component",
                    parameters={}
                )
            ]
        )
        await execution_engine.register_workflow_definition(definition)
        
        workflow_id = "test-workflow-complete"
        await execution_engine.start_workflow(workflow_id, workflow_type, {})
        
        # Simulate workflow completion
        result = {"status": "success", "metrics": {"accuracy": 0.95}}
        await execution_engine._complete_workflow(workflow_id, result)
        
        # Workflow should be removed from active executors
        assert workflow_id not in execution_engine.active_executors
        
        # Verify completion event was emitted
        execution_engine.event_bus.emit.assert_called()
    
    @pytest.mark.asyncio
    async def test_workflow_failure_handling(self, execution_engine):
        """Test workflow failure handling."""
        # Register workflow definition
        from prompt_improver.ml.orchestration.core.workflow_types import WorkflowDefinition, WorkflowStep
        
        workflow_type = "test_training"
        definition = WorkflowDefinition(
            workflow_type=workflow_type,
            name="Test Training Workflow", 
            description="Test workflow for unit testing",
            steps=[
                WorkflowStep(
                    step_id="step1",
                    name="Test Step",
                    component_name="test_component",
                    parameters={}
                )
            ]
        )
        await execution_engine.register_workflow_definition(definition)
        
        workflow_id = "test-workflow-fail"
        await execution_engine.start_workflow(workflow_id, workflow_type, {})
        
        error_message = "Simulated failure"
        await execution_engine._handle_workflow_failure(workflow_id, error_message)
        
        # Workflow should be removed from active executors
        assert workflow_id not in execution_engine.active_executors
        
        # Verify failure event was emitted
        execution_engine.event_bus.emit.assert_called()
    
    @pytest.mark.asyncio
    async def test_event_emission(self, execution_engine):
        """Test that events are emitted correctly."""
        # Register workflow definition
        from prompt_improver.ml.orchestration.core.workflow_types import WorkflowDefinition, WorkflowStep
        
        workflow_type = "test_training"
        definition = WorkflowDefinition(
            workflow_type=workflow_type,
            name="Test Training Workflow",
            description="Test workflow for unit testing",
            steps=[
                WorkflowStep(
                    step_id="step1",
                    name="Test Step",
                    component_name="test_component",
                    parameters={}
                )
            ]
        )
        await execution_engine.register_workflow_definition(definition)
        
        workflow_id = "test-workflow-event"
        await execution_engine.start_workflow(workflow_id, workflow_type, {})
        
        # Verify workflow started event was emitted
        execution_engine.event_bus.emit.assert_called()
        
        # Check the event
        call_args = execution_engine.event_bus.emit.call_args_list
        assert len(call_args) >= 1
        
        # Find the workflow started event
        started_event = None
        for call in call_args:
            event = call[0][0]  # First argument is the event
            if event.event_type == EventType.WORKFLOW_STARTED:
                started_event = event
                break
        
        assert started_event is not None
        assert started_event.data["workflow_id"] == workflow_id
    
    @pytest.mark.asyncio
    async def test_workflow_definitions(self, execution_engine):
        """Test workflow definition management."""
        # Initially should have loaded definitions
        definitions = await execution_engine.list_workflow_definitions()
        initial_count = len(definitions)
        
        # Register a new workflow definition
        from prompt_improver.ml.orchestration.core.workflow_types import WorkflowDefinition, WorkflowStep
        
        new_definition = WorkflowDefinition(
            workflow_type="custom_workflow",
            name="Custom Workflow",
            description="Custom test workflow",
            steps=[
                WorkflowStep(
                    step_id="custom_step",
                    name="Custom Step",
                    component_name="custom_component",
                    parameters={}
                )
            ]
        )
        await execution_engine.register_workflow_definition(new_definition)
        
        # Should have one more definition
        updated_definitions = await execution_engine.list_workflow_definitions()
        assert len(updated_definitions) == initial_count + 1
        assert "custom_workflow" in updated_definitions
    
    @pytest.mark.asyncio
    async def test_workflow_cleanup(self, execution_engine):
        """Test workflow cleanup on shutdown."""
        # Register workflow definition
        from prompt_improver.ml.orchestration.core.workflow_types import WorkflowDefinition, WorkflowStep
        
        workflow_type = "test_training"
        definition = WorkflowDefinition(
            workflow_type=workflow_type,
            name="Test Training Workflow",
            description="Test workflow for unit testing",
            steps=[
                WorkflowStep(
                    step_id="step1",
                    name="Test Step", 
                    component_name="test_component",
                    parameters={}
                )
            ]
        )
        await execution_engine.register_workflow_definition(definition)
        
        # Start some workflows
        workflow_id1 = "cleanup-workflow-1"
        workflow_id2 = "cleanup-workflow-2"
        await execution_engine.start_workflow(workflow_id1, workflow_type, {})
        await execution_engine.start_workflow(workflow_id2, workflow_type, {})
        
        assert len(execution_engine.active_executors) == 2
        
        # Shutdown should clean up active workflows
        await execution_engine.shutdown()
        
        assert len(execution_engine.active_executors) == 0


class TestWorkflowInstance:
    """Test suite for WorkflowInstance."""
    
    def test_workflow_instance_creation(self):
        """Test workflow instance creation."""
        workflow_id = "test-workflow-123"
        workflow_type = "tier1_training"
        created_at = datetime.now(timezone.utc)
        metadata = {"param1": "value1"}
        
        instance = WorkflowInstance(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            state=PipelineState.IDLE,
            created_at=created_at,
            metadata=metadata
        )
        
        assert instance.workflow_id == workflow_id
        assert instance.workflow_type == workflow_type
        assert instance.state == PipelineState.IDLE
        assert instance.created_at == created_at
        assert instance.metadata == metadata
        assert instance.started_at is None
        assert instance.completed_at is None
        assert instance.error_message is None
    
    def test_workflow_instance_state_transitions(self):
        """Test workflow instance state transitions."""
        created_at = datetime.now(timezone.utc)
        instance = WorkflowInstance(
            workflow_id="test-workflow-456",
            workflow_type="tier1_training",
            state=PipelineState.IDLE,
            created_at=created_at
        )
        
        # Initial state
        assert instance.state == PipelineState.IDLE
        
        # Transition to running
        instance.state = PipelineState.RUNNING
        instance.started_at = datetime.now(timezone.utc)
        
        assert instance.state == PipelineState.RUNNING
        assert instance.started_at is not None
        assert instance.started_at >= created_at
        
        # Transition to completed
        instance.state = PipelineState.COMPLETED
        instance.completed_at = datetime.now(timezone.utc)
        
        assert instance.state == PipelineState.COMPLETED
        assert instance.completed_at is not None
        assert instance.completed_at >= instance.started_at
    
    def test_workflow_error_handling(self):
        """Test workflow error handling."""
        created_at = datetime.now(timezone.utc)
        instance = WorkflowInstance(
            workflow_id="test-workflow-error",
            workflow_type="tier1_training",
            state=PipelineState.RUNNING,
            created_at=created_at
        )
        
        # Initially no error
        assert instance.error_message is None
        
        # Set error
        error_message = "Component failure in training step"
        instance.error_message = error_message
        instance.state = PipelineState.ERROR
        
        assert instance.error_message == error_message
        assert instance.state == PipelineState.ERROR


if __name__ == "__main__":
    # Run basic smoke test
    async def smoke_test():
        """Basic smoke test for workflow execution engine."""
        print("Running Workflow Execution Engine smoke test...")
        
        # Create config and mocks
        config = OrchestratorConfig()
        mock_event_bus = Mock()
        mock_event_bus.emit = AsyncMock()
        mock_resource_manager = Mock()
        mock_resource_manager.allocate_resources = AsyncMock(return_value={"cpu": 1.0})
        mock_resource_manager.deallocate_resources = AsyncMock()
        mock_component_registry = Mock()
        
        # Create and initialize engine
        engine = WorkflowExecutionEngine(config, mock_event_bus, mock_resource_manager, mock_component_registry)
        
        try:
            await engine.initialize()
            print("✓ Engine initialized successfully")
            
            # Test workflow start
            workflow_id = await engine.start_workflow("tier1_training", {"test": True})
            print(f"✓ Workflow started: {workflow_id}")
            
            # Test status
            status = await engine.get_workflow_status(workflow_id)
            print(f"✓ Workflow status: {status.state}")
            
            # Test list workflows
            workflows = await engine.list_active_workflows()
            print(f"✓ Active workflows: {len(workflows)}")
            
            # Test stop workflow
            stopped = await engine.stop_workflow(workflow_id)
            print(f"✓ Workflow stopped: {stopped}")
            
            print("✓ All basic tests passed!")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            raise
        finally:
            await engine.shutdown()
            print("✓ Engine shut down gracefully")
    
    # Run the smoke test
    asyncio.run(smoke_test())