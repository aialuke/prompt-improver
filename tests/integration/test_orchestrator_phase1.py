#!/usr/bin/env python3
"""
Standalone test for ML Pipeline Orchestrator Phase 1 Implementation.

This test verifies the core orchestration components work independently.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_phase1_implementation():
    """Test Phase 1 orchestrator implementation."""
    print("🚀 Testing ML Pipeline Orchestrator Phase 1 Implementation")
    print("=" * 60)
    
    try:
        # Test 1: Import core components
        print("1. Testing imports...")
        from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
        from prompt_improver.ml.orchestration.events.event_types import EventType, MLEvent
        from prompt_improver.ml.orchestration.events.event_bus import EventBus
        from prompt_improver.ml.orchestration.core.component_registry import ComponentRegistry
        from prompt_improver.ml.orchestration.core.resource_manager import ResourceManager
        from prompt_improver.ml.orchestration.core.workflow_execution_engine import WorkflowExecutionEngine
        from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator, PipelineState
        print("   ✓ All core components imported successfully")
        
        # Test 2: Configuration system
        print("\n2. Testing configuration system...")
        config = OrchestratorConfig(
            max_concurrent_workflows=2,
            component_health_check_interval=5,
            event_bus_buffer_size=100
        )
        
        # Validate configuration
        errors = config.validate()
        assert len(errors) == 0, f"Configuration validation failed: {errors}"
        print("   ✓ Configuration created and validated")
        
        # Test 3: Event system
        print("\n3. Testing event system...")
        event_bus = EventBus(config)
        await event_bus.initialize()
        
        # Test event creation and emission
        test_events = []
        def event_handler(event):
            test_events.append(event)
        
        event_bus.subscribe(EventType.TRAINING_STARTED, event_handler)
        
        test_event = MLEvent(
            event_type=EventType.TRAINING_STARTED,
            source="test",
            data={"test": "data"}
        )
        
        await event_bus.emit(test_event)
        await asyncio.sleep(0.1)  # Allow event processing
        
        assert len(test_events) >= 1, "Event not received"
        print("   ✓ Event system working correctly")
        
        await event_bus.shutdown()
        
        # Test 4: Component Registry
        print("\n4. Testing component registry...")
        registry = ComponentRegistry(config)
        await registry.initialize()
        
        # Check component registration
        components = await registry.list_components()
        assert len(components) > 0, "No components registered"
        print(f"   ✓ Component registry initialized with {len(components)} components")
        
        # Check health summary
        health_summary = await registry.get_health_summary()
        assert "total_components" in health_summary, "Health summary missing required fields"
        print(f"   ✓ Health monitoring active for {health_summary['total_components']} components")
        
        await registry.shutdown()
        
        # Test 5: Resource Manager
        print("\n5. Testing resource manager...")
        resource_manager = ResourceManager(config)
        await resource_manager.initialize()
        
        # Test resource allocation
        from prompt_improver.ml.orchestration.core.resource_manager import ResourceType
        
        allocation_id = await resource_manager.allocate_resource(
            ResourceType.CPU,
            1.0,
            "test_component"
        )
        assert allocation_id is not None, "Resource allocation failed"
        print("   ✓ Resource allocation working")
        
        # Test usage stats
        usage_stats = await resource_manager.get_usage_stats()
        assert len(usage_stats) > 0, "No usage stats available"
        print(f"   ✓ Resource monitoring active for {len(usage_stats)} resource types")
        
        # Release resource
        released = await resource_manager.release_resource(allocation_id)
        assert released, "Resource release failed"
        print("   ✓ Resource release working")
        
        await resource_manager.shutdown()
        
        # Test 6: Workflow Engine
        print("\n6. Testing workflow execution engine...")
        workflow_engine = WorkflowExecutionEngine(config)
        await workflow_engine.initialize()
        
        # Check workflow definitions
        definitions = await workflow_engine.list_workflow_definitions()
        assert len(definitions) > 0, "No workflow definitions loaded"
        print(f"   ✓ Workflow engine initialized with {len(definitions)} workflow types")
        
        await workflow_engine.shutdown()
        
        # Test 7: Full Orchestrator Integration
        print("\n7. Testing full orchestrator integration...")
        orchestrator = MLPipelineOrchestrator(config)
        await orchestrator.initialize()
        
        assert orchestrator.state == PipelineState.IDLE, "Orchestrator not in IDLE state"
        assert orchestrator._is_initialized, "Orchestrator not initialized"
        print("   ✓ Orchestrator initialized successfully")
        
        # Test workflow start
        workflow_id = await orchestrator.start_workflow("tier1_training", {"test": True})
        assert workflow_id is not None, "Workflow start failed"
        print(f"   ✓ Workflow started: {workflow_id}")
        
        # Test workflow status
        status = await orchestrator.get_workflow_status(workflow_id)
        assert status.workflow_id == workflow_id, "Workflow status mismatch"
        print(f"   ✓ Workflow status: {status.state.value}")
        
        # Test component health
        health = await orchestrator.get_component_health()
        assert len(health) > 0, "No component health data"
        print(f"   ✓ Component health monitoring: {len(health)} components")
        
        # Test resource usage
        usage = await orchestrator.get_resource_usage()
        assert len(usage) > 0, "No resource usage data"
        print(f"   ✓ Resource usage monitoring: {len(usage)} resource types")
        
        await orchestrator.shutdown()
        print("   ✓ Orchestrator shutdown successfully")
        
        # Test 8: Performance verification
        print("\n8. Testing performance characteristics...")
        start_time = asyncio.get_event_loop().time()
        
        # Quick initialization test
        orchestrator = MLPipelineOrchestrator(config)
        await orchestrator.initialize()
        
        init_time = asyncio.get_event_loop().time() - start_time
        assert init_time < 5.0, f"Initialization too slow: {init_time:.2f}s"
        print(f"   ✓ Initialization time: {init_time:.3f}s")
        
        # Quick workflow start test
        start_time = asyncio.get_event_loop().time()
        workflow_id = await orchestrator.start_workflow("tier1_training", {})
        workflow_time = asyncio.get_event_loop().time() - start_time
        
        assert workflow_time < 1.0, f"Workflow start too slow: {workflow_time:.2f}s"
        print(f"   ✓ Workflow start time: {workflow_time:.3f}s")
        
        await orchestrator.shutdown()
        
        print("\n" + "=" * 60)
        print("🎉 ALL PHASE 1 TESTS PASSED!")
        print("✅ Core orchestrator structure implemented")
        print("✅ Event system functional")
        print("✅ Component registry operational")
        print("✅ Resource management working")
        print("✅ Workflow execution engine ready")
        print("✅ Full integration successful")
        print("✅ Performance targets met")
        print("\n🚀 Phase 1 Foundation is complete and ready for Phase 2!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_phase1_implementation())
    sys.exit(0 if success else 1)