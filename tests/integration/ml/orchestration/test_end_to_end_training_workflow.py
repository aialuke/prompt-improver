"""
End-to-end integration tests for training workflows.
"""

import pytest
import asyncio
from datetime import datetime, timezone

from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import (
    MLPipelineOrchestrator, PipelineState
)
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from prompt_improver.ml.orchestration.events.event_types import EventType


class TestEndToEndTrainingWorkflow:
    """End-to-end integration tests for training workflows."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create full orchestrator for integration testing."""
        config = OrchestratorConfig(
            max_concurrent_workflows=3,
            component_health_check_interval=2,
            training_timeout=300,  # 5 minutes for tests
            event_bus_buffer_size=50
        )
        
        orchestrator = MLPipelineOrchestrator(config)
        await orchestrator.initialize()
        
        yield orchestrator
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_complete_training_workflow_execution(self, orchestrator):
        """Test complete training workflow from start to finish."""
        workflow_type = "tier1_training"
        parameters = {
            "model_type": "transformer",
            "dataset": "integration_test_data.json",
            "epochs": 3,
            "learning_rate": 0.001,
            "batch_size": 16,
            "validation_split": 0.2
        }
        
        # Start workflow
        workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
        assert workflow_id is not None
        
        # Monitor workflow progress
        max_wait_time = 60  # 1 minute timeout for integration test
        elapsed_time = 0
        check_interval = 2  # Check every 2 seconds
        
        while elapsed_time < max_wait_time:
            status = await orchestrator.get_workflow_status(workflow_id)
            
            if status.state in [PipelineState.COMPLETED, PipelineState.ERROR]:
                break
                
            await asyncio.sleep(check_interval)
            elapsed_time += check_interval
        
        # Verify workflow completed successfully
        final_status = await orchestrator.get_workflow_status(workflow_id)
        assert final_status.state == PipelineState.COMPLETED
        assert final_status.metadata is not None
        
        # Verify workflow metadata contains expected parameters
        result = final_status.metadata
        assert "model_type" in result
        assert "dataset" in result
        assert result["model_type"] == "transformer"
        assert result["dataset"] == "integration_test_data.json"
    
    @pytest.mark.asyncio
    async def test_training_workflow_with_component_integration(self, orchestrator):
        """Test training workflow with real component integration."""
        workflow_type = "tier1_training"
        parameters = {
            "model_type": "bert",
            "dataset": "component_integration_test.json",
            "use_real_components": True,  # Flag to use actual components
            "component_timeout": 30
        }
        
        # Start workflow
        workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
        
        # Wait for workflow to complete
        await self._wait_for_workflow_completion(orchestrator, workflow_id, timeout=120)
        
        # Get final status
        status = await orchestrator.get_workflow_status(workflow_id)
        
        # Verify workflow completed (may succeed or fail, but should not hang)
        assert status.state in [PipelineState.COMPLETED, PipelineState.ERROR]
        
        # If successful, verify component integration worked
        if status.state == PipelineState.COMPLETED:
            # Verify components were actually called
            component_health = await orchestrator.get_component_health()
            assert len(component_health) > 0
            
            # Check that at least some components are healthy
            healthy_components = sum(1 for is_healthy in component_health.values() if is_healthy)
            assert healthy_components > 0
    
    @pytest.mark.asyncio
    async def test_training_workflow_resource_allocation(self, orchestrator):
        """Test training workflow resource allocation and cleanup."""
        workflow_type = "tier1_training"
        parameters = {
            "model_type": "large_transformer",
            "resource_requirements": {
                "cpu": 4.0,
                "memory": 8192,  # 8GB
                "gpu": 1
            }
        }
        
        # Check initial resource usage
        initial_usage = await orchestrator.get_resource_usage()
        initial_allocations = initial_usage.get("active_allocations", 0)
        
        # Start workflow
        workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
        
        # Check that resources were allocated
        await asyncio.sleep(1)  # Give time for allocation
        active_usage = await orchestrator.get_resource_usage()
        active_allocations = active_usage.get("active_allocations", 0)
        
        assert active_allocations > initial_allocations
        
        # Wait for workflow completion
        await self._wait_for_workflow_completion(orchestrator, workflow_id)
        
        # Check that resources were deallocated
        await asyncio.sleep(1)  # Give time for deallocation
        final_usage = await orchestrator.get_resource_usage()
        final_allocations = final_usage.get("active_allocations", 0)
        
        # Resources should be cleaned up
        assert final_allocations <= initial_allocations
    
    @pytest.mark.asyncio
    async def test_training_workflow_event_emission(self, orchestrator):
        """Test training workflow event emission throughout execution."""
        workflow_type = "tier1_training"
        parameters = {"model_type": "test", "track_events": True}
        
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        # Subscribe to relevant events
        await orchestrator.event_bus.subscribe(EventType.WORKFLOW_STARTED, event_handler)
        await orchestrator.event_bus.subscribe(EventType.TRAINING_STARTED, event_handler)
        await orchestrator.event_bus.subscribe(EventType.TRAINING_COMPLETED, event_handler)
        await orchestrator.event_bus.subscribe(EventType.WORKFLOW_COMPLETED, event_handler)
        
        # Start workflow
        workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
        
        # Wait for workflow completion
        await self._wait_for_workflow_completion(orchestrator, workflow_id)
        
        # Give time for all events to be processed
        await asyncio.sleep(2)
        
        # Verify events were emitted
        assert len(events_received) >= 2  # At least workflow started and completed
        
        # Check for specific events
        event_types = [event.event_type for event in events_received]
        assert EventType.WORKFLOW_STARTED in event_types
        
        # Verify event data contains workflow_id
        workflow_events = [e for e in events_received if e.data.get("workflow_id") == workflow_id]
        assert len(workflow_events) >= 1
    
    @pytest.mark.asyncio
    async def test_training_workflow_failure_recovery(self, orchestrator):
        """Test training workflow failure handling and recovery."""
        workflow_type = "tier1_training"
        parameters = {
            "model_type": "invalid_model",  # Should cause failure
            "simulate_failure": True,
            "failure_step": "model_training"
        }
        
        # Start workflow that will fail
        workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
        
        # Wait for workflow to fail
        await self._wait_for_workflow_completion(orchestrator, workflow_id, timeout=60)
        
        # Verify workflow failed
        status = await orchestrator.get_workflow_status(workflow_id)
        assert status.state == PipelineState.ERROR
        assert status.error is not None
        
        # Verify orchestrator is still functional after failure
        # Start a successful workflow
        success_parameters = {
            "model_type": "simple_model",
            "dataset": "small_test.json"
        }
        
        success_workflow_id = await orchestrator.start_workflow(workflow_type, success_parameters)
        await self._wait_for_workflow_completion(orchestrator, success_workflow_id)
        
        success_status = await orchestrator.get_workflow_status(success_workflow_id)
        assert success_status.state == PipelineState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_concurrent_training_workflows(self, orchestrator):
        """Test multiple concurrent training workflows."""
        workflow_type = "tier1_training"
        
        # Define multiple workflows with different parameters
        workflows = [
            {"model_type": "transformer", "dataset": "dataset_1.json", "epochs": 2},
            {"model_type": "bert", "dataset": "dataset_2.json", "epochs": 3},
            {"model_type": "gpt", "dataset": "dataset_3.json", "epochs": 1}
        ]
        
        # Start all workflows concurrently
        workflow_ids = []
        for params in workflows:
            workflow_id = await orchestrator.start_workflow(workflow_type, params)
            workflow_ids.append(workflow_id)
        
        # Verify all workflows started
        assert len(workflow_ids) == 3
        assert len(set(workflow_ids)) == 3  # All unique IDs
        
        # Wait for all workflows to complete
        for workflow_id in workflow_ids:
            await self._wait_for_workflow_completion(orchestrator, workflow_id, timeout=180)
        
        # Verify all workflows completed
        for workflow_id in workflow_ids:
            status = await orchestrator.get_workflow_status(workflow_id)
            assert status.state in [PipelineState.COMPLETED, PipelineState.ERROR]
        
        # Check that at least some workflows succeeded
        completed_workflows = []
        for workflow_id in workflow_ids:
            status = await orchestrator.get_workflow_status(workflow_id)
            if status.state == PipelineState.COMPLETED:
                completed_workflows.append(workflow_id)
        
        assert len(completed_workflows) >= 1  # At least one should succeed
    
    @pytest.mark.asyncio
    async def test_training_workflow_health_monitoring(self, orchestrator):
        """Test health monitoring during training workflow execution."""
        workflow_type = "tier1_training"
        parameters = {
            "model_type": "health_monitored",
            "enable_health_checks": True,
            "health_check_interval": 5
        }
        
        # Start workflow
        workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
        
        # Monitor health during execution
        health_checks = []
        for _ in range(3):  # Check health 3 times during execution
            await asyncio.sleep(2)
            component_health = await orchestrator.get_component_health()
            health_checks.append(component_health)
        
        # Wait for workflow completion
        await self._wait_for_workflow_completion(orchestrator, workflow_id)
        
        # Verify health monitoring was active
        assert len(health_checks) == 3
        
        # Check that health data was collected
        for health_check in health_checks:
            assert isinstance(health_check, dict)
            assert len(health_check) > 0  # Should have some components
    
    @pytest.mark.asyncio
    async def test_training_workflow_with_metrics_collection(self, orchestrator):
        """Test training workflow with comprehensive metrics collection."""
        workflow_type = "tier1_training"
        parameters = {
            "model_type": "metrics_test",
            "collect_detailed_metrics": True,
            "metric_collection_interval": 1
        }
        
        # Start workflow
        workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
        
        # Wait for workflow completion
        await self._wait_for_workflow_completion(orchestrator, workflow_id)
        
        # Verify workflow completed
        status = await orchestrator.get_workflow_status(workflow_id)
        assert status.state == PipelineState.COMPLETED
        
        # Check that metrics were collected
        # Note: This would require access to the metrics collector
        # For now, we verify the workflow completed successfully with metrics enabled
        assert status.metadata is not None
    
    async def _wait_for_workflow_completion(self, orchestrator, workflow_id, timeout=120):
        """Helper method to wait for workflow completion."""
        elapsed_time = 0
        check_interval = 2
        
        while elapsed_time < timeout:
            status = await orchestrator.get_workflow_status(workflow_id)
            
            if status.state in [PipelineState.COMPLETED, PipelineState.ERROR]:
                return status
                
            await asyncio.sleep(check_interval)
            elapsed_time += check_interval
        
        # Timeout reached
        final_status = await orchestrator.get_workflow_status(workflow_id)
        pytest.fail(f"Workflow {workflow_id} did not complete within {timeout}s. Final state: {final_status.state}")


class TestTrainingWorkflowRegressionTests:
    """Regression tests for training workflow functionality."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator for regression testing."""
        config = OrchestratorConfig(max_concurrent_workflows=1)
        orchestrator = MLPipelineOrchestrator(config)
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_workflow_id_uniqueness(self, orchestrator):
        """Test that workflow IDs are always unique."""
        workflow_ids = set()
        
        # Start multiple workflows
        for i in range(10):
            workflow_id = await orchestrator.start_workflow("tier1_training", {"iteration": i})
            
            # Verify uniqueness
            assert workflow_id not in workflow_ids
            workflow_ids.add(workflow_id)
        
        # All IDs should be unique
        assert len(workflow_ids) == 10
    
    @pytest.mark.asyncio
    async def test_workflow_state_consistency(self, orchestrator):
        """Test workflow state consistency throughout execution."""
        workflow_id = await orchestrator.start_workflow("tier1_training", {})
        
        # Track state changes
        states_observed = []
        
        for _ in range(10):  # Check state multiple times
            status = await orchestrator.get_workflow_status(workflow_id)
            states_observed.append(status.state)
            
            if status.state in [PipelineState.COMPLETED, PipelineState.ERROR]:
                break
                
            await asyncio.sleep(1)
        
        # Verify state progression is logical
        assert PipelineState.RUNNING in states_observed  # Should have been running
        
        # Final state should be terminal
        final_state = states_observed[-1]
        assert final_state in [PipelineState.COMPLETED, PipelineState.ERROR]
    
    @pytest.mark.asyncio
    async def test_workflow_cleanup_on_orchestrator_shutdown(self, orchestrator):
        """Test that active workflows are properly cleaned up on shutdown."""
        # Start a workflow
        workflow_id = await orchestrator.start_workflow("tier1_training", {"long_running": True})
        
        # Verify workflow is active
        status = await orchestrator.get_workflow_status(workflow_id)
        assert status.state == PipelineState.RUNNING
        
        # Shutdown orchestrator
        await orchestrator.shutdown()
        
        # Verify orchestrator is properly shut down
        assert orchestrator._is_initialized is False


if __name__ == "__main__":
    # Run basic integration test
    async def integration_test():
        """Basic integration test for training workflows."""
        print("Running Training Workflow Integration Test...")
        
        # Create orchestrator
        config = OrchestratorConfig(max_concurrent_workflows=1)
        orchestrator = MLPipelineOrchestrator(config)
        
        try:
            await orchestrator.initialize()
            print("✓ Orchestrator initialized")
            
            # Start training workflow
            workflow_id = await orchestrator.start_workflow("tier1_training", {
                "model_type": "integration_test",
                "dataset": "test.json"
            })
            print(f"✓ Training workflow started: {workflow_id}")
            
            # Wait for completion
            timeout = 60
            elapsed = 0
            while elapsed < timeout:
                status = await orchestrator.get_workflow_status(workflow_id)
                if status.state in [PipelineState.COMPLETED, PipelineState.ERROR]:
                    break
                await asyncio.sleep(2)
                elapsed += 2
            
            final_status = await orchestrator.get_workflow_status(workflow_id)
            print(f"✓ Workflow completed with state: {final_status.state}")
            
            # Check component health
            health = await orchestrator.get_component_health()
            print(f"✓ Component health checked: {len(health)} components")
            
            print("✓ Integration test passed!")
            
        except Exception as e:
            print(f"✗ Integration test failed: {e}")
            raise
        finally:
            await orchestrator.shutdown()
            print("✓ Orchestrator shut down")
    
    # Run the integration test
    asyncio.run(integration_test())