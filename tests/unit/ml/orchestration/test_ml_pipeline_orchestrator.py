"""
Tests for ML Pipeline Orchestrator - Phase 1 Foundation.
"""

import pytest
import asyncio
from datetime import datetime, timezone

from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import (
    MLPipelineOrchestrator, PipelineState, WorkflowInstance
)
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from prompt_improver.ml.orchestration.events.event_types import EventType, MLEvent


class TestMLPipelineOrchestrator:
    """Test suite for ML Pipeline Orchestrator."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator instance for testing."""
        config = OrchestratorConfig(
            # Use smaller values for testing
            max_concurrent_workflows=2,
            component_health_check_interval=1,
            event_bus_buffer_size=10
        )
        
        orchestrator = MLPipelineOrchestrator(config)
        await orchestrator.initialize()
        
        yield orchestrator
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.state == PipelineState.IDLE
        assert orchestrator._is_initialized is True
        assert orchestrator.event_bus is not None
        assert orchestrator.workflow_engine is not None
        assert orchestrator.resource_manager is not None
        assert orchestrator.component_registry is not None
    
    @pytest.mark.asyncio
    async def test_start_workflow(self, orchestrator):
        """Test starting a workflow."""
        workflow_type = "tier1_training"
        parameters = {"test_param": "value"}
        
        workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
        
        assert workflow_id is not None
        assert workflow_id in orchestrator.active_workflows
        
        workflow_instance = orchestrator.active_workflows[workflow_id]
        assert workflow_instance.workflow_type == workflow_type
        assert workflow_instance.state == PipelineState.RUNNING
        assert workflow_instance.metadata == parameters
    
    @pytest.mark.asyncio
    async def test_workflow_status(self, orchestrator):
        """Test getting workflow status."""
        workflow_type = "tier1_training"
        parameters = {"test_param": "value"}
        
        workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
        
        status = await orchestrator.get_workflow_status(workflow_id)
        assert status.workflow_id == workflow_id
        assert status.workflow_type == workflow_type
        assert status.state == PipelineState.RUNNING
    
    @pytest.mark.asyncio
    async def test_list_workflows(self, orchestrator):
        """Test listing workflows."""
        # Initially no workflows
        workflows = await orchestrator.list_workflows()
        assert len(workflows) == 0
        
        # Start a workflow
        workflow_id = await orchestrator.start_workflow("tier1_training", {})
        
        # Should have one workflow
        workflows = await orchestrator.list_workflows()
        assert len(workflows) == 1
        assert workflows[0].workflow_id == workflow_id
    
    @pytest.mark.asyncio
    async def test_component_health(self, orchestrator):
        """Test component health monitoring."""
        health = await orchestrator.get_component_health()
        
        # Should have registered components from Tier 1
        assert len(health) > 0
        
        # All components should initially be unhealthy until checked
        for component_name, is_healthy in health.items():
            assert isinstance(is_healthy, bool)
    
    @pytest.mark.asyncio
    async def test_resource_usage(self, orchestrator):
        """Test resource usage statistics."""
        usage = await orchestrator.get_resource_usage()
        
        assert isinstance(usage, dict)
        # Should have usage stats for different resource types
        assert len(usage) > 0
    
    @pytest.mark.asyncio
    async def test_event_handling(self, orchestrator):
        """Test event handling."""
        events_received = []
        
        def event_handler(event: MLEvent):
            events_received.append(event)
        
        # Subscribe to workflow events
        orchestrator.event_bus.subscribe(EventType.WORKFLOW_STARTED, event_handler)
        
        # Start a workflow to trigger event
        await orchestrator.start_workflow("tier1_training", {})
        
        # Give some time for event processing
        await asyncio.sleep(0.1)
        
        # Should have received workflow started event
        assert len(events_received) >= 1
        assert events_received[0].event_type == EventType.WORKFLOW_STARTED
    
    @pytest.mark.asyncio 
    async def test_shutdown_gracefully(self, orchestrator):
        """Test graceful shutdown."""
        # Start a workflow
        workflow_id = await orchestrator.start_workflow("tier1_training", {})
        
        # Shutdown should stop all workflows
        await orchestrator.shutdown()
        
        assert orchestrator.state == PipelineState.IDLE
        assert orchestrator._is_initialized is False


class TestEventSystem:
    """Test suite for Event System."""
    
    @pytest.mark.asyncio
    async def test_event_creation(self):
        """Test MLEvent creation."""
        event = MLEvent(
            event_type=EventType.TRAINING_STARTED,
            source="test_component",
            data={"test": "data"}
        )
        
        assert event.event_type == EventType.TRAINING_STARTED
        assert event.source == "test_component"
        assert event.data == {"test": "data"}
        assert event.timestamp is not None
        assert event.event_id is not None
    
    @pytest.mark.asyncio
    async def test_event_serialization(self):
        """Test event to/from dict conversion."""
        original_event = MLEvent(
            event_type=EventType.TRAINING_STARTED,
            source="test_component",
            data={"test": "data"},
            correlation_id="test_correlation"
        )
        
        # Convert to dict and back
        event_dict = original_event.to_dict()
        restored_event = MLEvent.from_dict(event_dict)
        
        assert restored_event.event_type == original_event.event_type
        assert restored_event.source == original_event.source
        assert restored_event.data == original_event.data
        assert restored_event.correlation_id == original_event.correlation_id


class TestComponentRegistry:
    """Test suite for Component Registry."""
    
    @pytest.fixture
    async def component_registry(self):
        """Create component registry for testing."""
        from prompt_improver.ml.orchestration.core.component_registry import ComponentRegistry
        from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
        
        config = OrchestratorConfig(component_health_check_interval=1)
        registry = ComponentRegistry(config)
        await registry.initialize()
        
        yield registry
        
        await registry.shutdown()
    
    @pytest.mark.asyncio
    async def test_component_registration(self, component_registry):
        """Test component registration."""
        components = await component_registry.list_components()
        
        # Should have registered Tier 1 components during initialization
        assert len(components) > 0
        
        # Check for some expected components
        component_names = [c.name for c in components]
        assert "training_data_loader" in component_names
        assert "ml_integration" in component_names
    
    @pytest.mark.asyncio
    async def test_health_summary(self, component_registry):
        """Test health summary."""
        summary = await component_registry.get_health_summary()
        
        assert "total_components" in summary
        assert "status_distribution" in summary
        assert "tier_health" in summary
        assert "overall_health_percentage" in summary
        
        assert summary["total_components"] > 0


if __name__ == "__main__":
    # Run basic smoke test
    async def smoke_test():
        """Basic smoke test for orchestrator."""
        print("Running ML Pipeline Orchestrator smoke test...")
        
        # Create and initialize orchestrator
        config = OrchestratorConfig(max_concurrent_workflows=1)
        orchestrator = MLPipelineOrchestrator(config)
        
        try:
            await orchestrator.initialize()
            print("✓ Orchestrator initialized successfully")
            
            # Test workflow start
            workflow_id = await orchestrator.start_workflow("tier1_training", {"test": True})
            print(f"✓ Workflow started: {workflow_id}")
            
            # Test status
            status = await orchestrator.get_workflow_status(workflow_id)
            print(f"✓ Workflow status: {status.state}")
            
            # Test resource usage
            usage = await orchestrator.get_resource_usage()
            print(f"✓ Resource usage retrieved: {len(usage)} resource types")
            
            # Test component health
            health = await orchestrator.get_component_health()
            print(f"✓ Component health retrieved: {len(health)} components")
            
            print("✓ All basic tests passed!")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            raise
        finally:
            await orchestrator.shutdown()
            print("✓ Orchestrator shut down gracefully")
    
    # Run the smoke test
    asyncio.run(smoke_test())