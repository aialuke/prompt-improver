"""
Tests for Training Workflow Coordinator.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from prompt_improver.ml.orchestration.coordinators.training_workflow_coordinator import (
    TrainingWorkflowCoordinator
)
from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import PipelineState
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from prompt_improver.ml.orchestration.events.event_types import EventType, MLEvent


class TestTrainingWorkflowCoordinator:
    """Test suite for Training Workflow Coordinator."""
    
    @pytest.fixture
    async def coordinator(self):
        """Create coordinator instance for testing."""
        config = OrchestratorConfig()
        
        # Mock dependencies
        mock_event_bus = Mock()
        mock_event_bus.emit = AsyncMock()
        
        mock_resource_manager = Mock()
        mock_resource_manager.allocate_resources = AsyncMock(return_value=Mock(allocation_id="alloc-123"))
        mock_resource_manager.deallocate_resources = AsyncMock(return_value=True)
        
        mock_component_registry = Mock()
        mock_component_registry.get_component = AsyncMock()
        
        coordinator = TrainingWorkflowCoordinator(config, mock_event_bus, mock_resource_manager, mock_component_registry)
        await coordinator.initialize()
        
        yield coordinator
        
        await coordinator.shutdown()
    
    @pytest.mark.asyncio
    async def test_coordinator_initialization(self, coordinator):
        """Test coordinator initialization."""
        assert coordinator._is_initialized is True
        assert coordinator.active_workflows == {}
        assert coordinator.workflow_history == []
    
    @pytest.mark.asyncio
    async def test_start_training_workflow(self, coordinator):
        """Test starting a training workflow."""
        workflow_id = "train-workflow-123"
        parameters = {
            "model_type": "transformer",
            "dataset": "test_dataset",
            "epochs": 10,
            "learning_rate": 0.001
        }
        
        # Mock component responses
        coordinator.component_registry.get_component.return_value = Mock(
            execute=AsyncMock(return_value={"status": "success", "data": {"processed": 100}})
        )
        
        result = await coordinator.start_training_workflow(workflow_id, parameters)
        
        assert result is not None
        assert result.workflow_id == workflow_id
        assert result.state == WorkflowState.RUNNING
        assert result.current_step == TrainingStep.DATA_LOADING
    
    @pytest.mark.asyncio
    async def test_training_pipeline_execution(self, coordinator):
        """Test complete training pipeline execution."""
        workflow_id = "train-pipeline-456"
        parameters = {
            "model_type": "bert",
            "training_data": "samples.json",
            "validation_split": 0.2
        }
        
        # Mock successful component responses for each step
        mock_component = Mock()
        mock_component.execute = AsyncMock(side_effect=[
            # Data loading step
            {"status": "success", "data": {"samples_loaded": 1000, "validation_samples": 200}},
            # Model training step
            {"status": "success", "data": {"model_trained": True, "accuracy": 0.92, "loss": 0.15}},
            # Rule optimization step
            {"status": "success", "data": {"rules_optimized": 25, "improvement": 0.08}}
        ])
        coordinator.component_registry.get_component.return_value = mock_component
        
        # Start workflow
        workflow = await coordinator.start_training_workflow(workflow_id, parameters)
        
        # Execute all steps
        await coordinator._execute_training_pipeline(workflow_id, parameters)
        
        # Verify all steps were executed
        assert mock_component.execute.call_count == 3
        
        # Verify events were emitted
        coordinator.event_bus.emit.assert_called()
        
        # Check final workflow state
        workflow_status = await coordinator.get_workflow_status(workflow_id)
        assert workflow_status.state == WorkflowState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_data_loading_step(self, coordinator):
        """Test data loading step execution."""
        workflow_id = "data-load-test"
        parameters = {"dataset": "test_data.json", "batch_size": 32}
        
        # Mock training data loader component
        mock_loader = Mock()
        mock_loader.execute = AsyncMock(return_value={
            "status": "success",
            "data": {
                "samples_loaded": 500,
                "features_extracted": 128,
                "preprocessing_time": 2.5
            }
        })
        coordinator.component_registry.get_component.return_value = mock_loader
        
        result = await coordinator._execute_data_loading(workflow_id, parameters)
        
        assert result["status"] == "success"
        assert result["data"]["samples_loaded"] == 500
        assert result["data"]["features_extracted"] == 128
        
        # Verify correct component was called
        coordinator.component_registry.get_component.assert_called_with("training_data_loader")
    
    @pytest.mark.asyncio
    async def test_model_training_step(self, coordinator):
        """Test model training step execution."""
        workflow_id = "model-train-test"
        parameters = {
            "model_architecture": "transformer",
            "epochs": 5,
            "learning_rate": 0.001,
            "batch_size": 16
        }
        
        # Mock ML integration component
        mock_ml_service = Mock()
        mock_ml_service.execute = AsyncMock(return_value={
            "status": "success",
            "data": {
                "model_trained": True,
                "final_accuracy": 0.94,
                "final_loss": 0.12,
                "training_time": 120.5,
                "epochs_completed": 5
            }
        })
        coordinator.component_registry.get_component.return_value = mock_ml_service
        
        result = await coordinator._execute_model_training(workflow_id, parameters)
        
        assert result["status"] == "success"
        assert result["data"]["model_trained"] is True
        assert result["data"]["final_accuracy"] == 0.94
        
        # Verify correct component was called
        coordinator.component_registry.get_component.assert_called_with("ml_integration")
    
    @pytest.mark.asyncio
    async def test_rule_optimization_step(self, coordinator):
        """Test rule optimization step execution."""
        workflow_id = "rule-opt-test"
        parameters = {
            "optimization_algorithm": "genetic",
            "population_size": 50,
            "generations": 20
        }
        
        # Mock rule optimizer component
        mock_optimizer = Mock()
        mock_optimizer.execute = AsyncMock(return_value={
            "status": "success",
            "data": {
                "rules_optimized": 30,
                "performance_improvement": 0.15,
                "optimization_time": 45.2,
                "best_fitness": 0.87
            }
        })
        coordinator.component_registry.get_component.return_value = mock_optimizer
        
        result = await coordinator._execute_rule_optimization(workflow_id, parameters)
        
        assert result["status"] == "success"
        assert result["data"]["rules_optimized"] == 30
        assert result["data"]["performance_improvement"] == 0.15
        
        # Verify correct component was called
        coordinator.component_registry.get_component.assert_called_with("rule_optimizer")
    
    @pytest.mark.asyncio
    async def test_training_failure_handling(self, coordinator):
        """Test training failure handling."""
        workflow_id = "fail-test"
        parameters = {"invalid": "parameters"}
        
        # Mock component failure
        mock_component = Mock()
        mock_component.execute = AsyncMock(side_effect=Exception("Component failure"))
        coordinator.component_registry.get_component.return_value = mock_component
        
        # Start workflow
        workflow = await coordinator.start_training_workflow(workflow_id, parameters)
        
        # Execute pipeline - should handle failure gracefully
        await coordinator._execute_training_pipeline(workflow_id, parameters)
        
        # Check workflow was marked as failed
        status = await coordinator.get_workflow_status(workflow_id)
        assert status.state == WorkflowState.FAILED
        assert status.error is not None
    
    @pytest.mark.asyncio
    async def test_training_metrics_collection(self, coordinator):
        """Test training metrics collection."""
        workflow_id = "metrics-test"
        
        # Create workflow and simulate training completion
        await coordinator.start_training_workflow(workflow_id, {})
        
        # Mock training metrics
        metrics = TrainingMetrics(
            workflow_id=workflow_id,
            accuracy=0.95,
            loss=0.08,
            training_time=300.5,
            samples_processed=2000,
            epochs_completed=10
        )
        
        # Record metrics
        await coordinator._record_training_metrics(workflow_id, metrics)
        
        # Retrieve metrics
        recorded_metrics = await coordinator.get_training_metrics(workflow_id)
        
        assert recorded_metrics is not None
        assert recorded_metrics.accuracy == 0.95
        assert recorded_metrics.loss == 0.08
        assert recorded_metrics.training_time == 300.5
    
    @pytest.mark.asyncio
    async def test_resource_allocation_for_training(self, coordinator):
        """Test resource allocation for training workflows."""
        workflow_id = "resource-test"
        parameters = {
            "model_size": "large",
            "expected_training_time": "2h",
            "memory_requirements": "8GB"
        }
        
        # Start workflow - should allocate resources
        await coordinator.start_training_workflow(workflow_id, parameters)
        
        # Verify resource allocation was requested
        coordinator.resource_manager.allocate_resources.assert_called()
        
        # Stop workflow - should deallocate resources
        await coordinator.stop_training_workflow(workflow_id)
        
        # Verify resource deallocation was requested
        coordinator.resource_manager.deallocate_resources.assert_called()
    
    @pytest.mark.asyncio
    async def test_concurrent_training_workflows(self, coordinator):
        """Test handling multiple concurrent training workflows."""
        workflow_ids = ["concurrent-1", "concurrent-2", "concurrent-3"]
        parameters = {"model": "test", "data": "test.json"}
        
        # Mock component responses
        mock_component = Mock()
        mock_component.execute = AsyncMock(return_value={"status": "success", "data": {}})
        coordinator.component_registry.get_component.return_value = mock_component
        
        # Start multiple workflows concurrently
        tasks = [
            coordinator.start_training_workflow(workflow_id, parameters)
            for workflow_id in workflow_ids
        ]
        
        workflows = await asyncio.gather(*tasks)
        
        # All workflows should be started
        assert len(workflows) == 3
        assert all(w.state == WorkflowState.RUNNING for w in workflows)
        
        # Check active workflows
        active = await coordinator.list_active_workflows()
        assert len(active) == 3
    
    @pytest.mark.asyncio
    async def test_workflow_status_updates(self, coordinator):
        """Test workflow status updates during execution."""
        workflow_id = "status-test"
        parameters = {"test": True}
        
        # Start workflow
        workflow = await coordinator.start_training_workflow(workflow_id, parameters)
        assert workflow.state == WorkflowState.RUNNING
        assert workflow.current_step == TrainingStep.DATA_LOADING
        
        # Simulate step progression
        await coordinator._update_workflow_step(workflow_id, TrainingStep.MODEL_TRAINING)
        status = await coordinator.get_workflow_status(workflow_id)
        assert status.current_step == TrainingStep.MODEL_TRAINING
        
        await coordinator._update_workflow_step(workflow_id, TrainingStep.RULE_OPTIMIZATION)
        status = await coordinator.get_workflow_status(workflow_id)
        assert status.current_step == TrainingStep.RULE_OPTIMIZATION


class TestTrainingMetrics:
    """Test suite for TrainingMetrics."""
    
    def test_metrics_creation(self):
        """Test training metrics creation."""
        metrics = TrainingMetrics(
            workflow_id="test-workflow",
            accuracy=0.92,
            loss=0.15,
            training_time=240.5,
            samples_processed=1500,
            epochs_completed=8
        )
        
        assert metrics.workflow_id == "test-workflow"
        assert metrics.accuracy == 0.92
        assert metrics.loss == 0.15
        assert metrics.training_time == 240.5
        assert metrics.samples_processed == 1500
        assert metrics.epochs_completed == 8
        assert metrics.timestamp is not None
    
    def test_metrics_serialization(self):
        """Test metrics to/from dict conversion."""
        metrics = TrainingMetrics(
            workflow_id="serialize-test",
            accuracy=0.88,
            loss=0.22,
            training_time=180.0,
            samples_processed=1000,
            epochs_completed=5
        )
        
        # Convert to dict and back
        metrics_dict = metrics.to_dict()
        restored_metrics = TrainingMetrics.from_dict(metrics_dict)
        
        assert restored_metrics.workflow_id == metrics.workflow_id
        assert restored_metrics.accuracy == metrics.accuracy
        assert restored_metrics.loss == metrics.loss
        assert restored_metrics.training_time == metrics.training_time
        assert restored_metrics.samples_processed == metrics.samples_processed
        assert restored_metrics.epochs_completed == metrics.epochs_completed
    
    def test_metrics_validation(self):
        """Test metrics validation."""
        # Valid metrics
        valid_metrics = TrainingMetrics(
            workflow_id="valid-test",
            accuracy=0.85,
            loss=0.18,
            training_time=120.0,
            samples_processed=800,
            epochs_completed=3
        )
        assert valid_metrics.is_valid()
        
        # Invalid metrics (negative values)
        invalid_metrics = TrainingMetrics(
            workflow_id="invalid-test",
            accuracy=-0.1,  # Invalid: negative accuracy
            loss=0.18,
            training_time=120.0,
            samples_processed=800,
            epochs_completed=3
        )
        assert not invalid_metrics.is_valid()


if __name__ == "__main__":
    # Run basic smoke test
    async def smoke_test():
        """Basic smoke test for training workflow coordinator."""
        print("Running Training Workflow Coordinator smoke test...")
        
        # Create mocks
        config = OrchestratorConfig()
        mock_event_bus = Mock()
        mock_event_bus.emit = AsyncMock()
        mock_resource_manager = Mock()
        mock_resource_manager.allocate_resources = AsyncMock(return_value=Mock(allocation_id="test-alloc"))
        mock_resource_manager.deallocate_resources = AsyncMock(return_value=True)
        mock_component_registry = Mock()
        mock_component_registry.get_component = AsyncMock(return_value=Mock(
            execute=AsyncMock(return_value={"status": "success", "data": {}})
        ))
        
        # Create and initialize coordinator
        coordinator = TrainingWorkflowCoordinator(config, mock_event_bus, mock_resource_manager, mock_component_registry)
        
        try:
            await coordinator.initialize()
            print("✓ Coordinator initialized successfully")
            
            # Test workflow start
            workflow = await coordinator.start_training_workflow("smoke-test", {"model": "test"})
            print(f"✓ Training workflow started: {workflow.workflow_id}")
            
            # Test status
            status = await coordinator.get_workflow_status("smoke-test")
            print(f"✓ Workflow status: {status.state}")
            
            # Test active workflows
            active = await coordinator.list_active_workflows()
            print(f"✓ Active workflows: {len(active)}")
            
            # Test stop workflow
            stopped = await coordinator.stop_training_workflow("smoke-test")
            print(f"✓ Workflow stopped: {stopped}")
            
            print("✓ All basic tests passed!")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            raise
        finally:
            await coordinator.shutdown()
            print("✓ Coordinator shut down gracefully")
    
    # Run the smoke test
    asyncio.run(smoke_test())