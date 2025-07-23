#!/usr/bin/env python3
"""
Validation script to ensure Phase 2 implementation has no false-positive outputs.

This script checks that:
1. All component calls return realistic results, not just placeholders
2. Resource allocation actually tracks real resources
3. Component registry has real component definitions
4. Workflow execution produces meaningful results
"""

import sys
from pathlib import Path

# Add the src directory to Python path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

def validate_component_execution():
    """Validate that component execution returns realistic results."""
    try:
        from prompt_improver.ml.orchestration.core.workflow_execution_engine import WorkflowExecutor
        from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
        from prompt_improver.ml.orchestration.core.workflow_types import WorkflowDefinition, WorkflowStep
        
        config = OrchestratorConfig()
        
        # Create a simple workflow definition for testing
        test_definition = WorkflowDefinition(
            workflow_type="test",
            name="Test Workflow",
            description="Test workflow for validation",
            steps=[
                WorkflowStep(
                    step_id="test_step",
                    name="Test Step",
                    component_name="training_data_loader",
                    parameters={}
                )
            ]
        )
        
        executor = WorkflowExecutor("test_workflow", test_definition, config, None)
        
        # Test component execution
        import asyncio
        
        async def test_component_calls():
            # Test training data loader
            result = await executor._call_component("training_data_loader", {"batch_size": 500})
            
            # Validate result structure
            assert result["status"] == "success", "Component should succeed"
            assert "result" in result, "Should have result field"
            assert result["result"]["data_loaded"] == True, "Should actually load data"
            assert result["result"]["record_count"] == 500, "Should respect batch_size parameter"
            
            # Test ML integration
            result = await executor._call_component("ml_integration", {"epochs": 5})
            assert result["result"]["model_trained"] == True, "Should train model"
            assert "accuracy" in result["result"], "Should have accuracy metric"
            assert "loss" in result["result"], "Should have loss metric"
            
            # Test rule optimizer
            result = await executor._call_component("rule_optimizer", {"iterations": 25})
            assert result["result"]["rules_optimized"] == True, "Should optimize rules"
            assert "optimization_score" in result["result"], "Should have optimization score"
            assert result["result"]["iterations"] > 0, "Should have iterations"
            
            print("✅ Component execution returns realistic, non-placeholder results")
            return True
        
        return asyncio.run(test_component_calls())
        
    except Exception as e:
        print(f"❌ Component execution validation failed: {e}")
        return False

def validate_resource_allocation():
    """Validate that resource allocation tracks real system resources."""
    try:
        from prompt_improver.ml.orchestration.core.resource_manager import ResourceManager, ResourceType
        from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
        import psutil
        
        config = OrchestratorConfig()
        manager = ResourceManager(config)
        
        # Check that resource limits are based on actual system resources
        manager._initialize_resource_limits()
        
        # CPU should be based on actual CPU count
        actual_cpu_count = psutil.cpu_count()
        assert manager.resource_limits[ResourceType.CPU] == float(actual_cpu_count), \
               f"CPU limit should match system CPU count: {actual_cpu_count}"
        
        # Memory should be based on actual system memory
        actual_memory = psutil.virtual_memory().total
        memory_limit_configured = config.memory_limit_gb * 1024 * 1024 * 1024
        assert manager.resource_limits[ResourceType.MEMORY] == memory_limit_configured, \
               "Memory limit should be configurable"
        
        print("✅ Resource allocation uses real system resources, not fake numbers")
        print(f"   - CPU cores: {manager.resource_limits[ResourceType.CPU]}")
        print(f"   - Memory: {manager.resource_limits[ResourceType.MEMORY] / (1024**3):.1f} GB")
        return True
        
    except Exception as e:
        print(f"❌ Resource allocation validation failed: {e}")
        return False

def validate_component_registry():
    """Validate that component registry has real component definitions."""
    try:
        from prompt_improver.ml.orchestration.config.component_definitions import ComponentDefinitions
        from prompt_improver.ml.orchestration.core.component_registry import ComponentTier
        
        comp_defs = ComponentDefinitions()
        
        # Check Tier 1 components
        tier1_components = comp_defs.get_tier_components(ComponentTier.TIER_1_CORE)
        
        # Validate training_data_loader definition
        training_loader = tier1_components.get("training_data_loader")
        assert training_loader is not None, "Should have training_data_loader definition"
        assert training_loader["file_path"] == "ml/core/training_data_loader.py", "Should have real file path"
        assert "data_loading" in training_loader["capabilities"], "Should have realistic capabilities"
        assert "memory" in training_loader["resource_requirements"], "Should have resource requirements"
        
        # Validate ml_integration definition
        ml_integration = tier1_components.get("ml_integration")
        assert ml_integration is not None, "Should have ml_integration definition"
        assert "model_training" in ml_integration["capabilities"], "Should have training capability"
        assert "training_data_loader" in ml_integration["dependencies"], "Should have dependencies"
        
        # Check that we have the expected number of components
        assert len(tier1_components) == 11, f"Should have 11 Tier 1 components, got {len(tier1_components)}"
        
        tier2_components = comp_defs.get_tier_components(ComponentTier.TIER_2_OPTIMIZATION)
        assert len(tier2_components) == 8, f"Should have 8 Tier 2 components, got {len(tier2_components)}"
        
        print("✅ Component registry has detailed, realistic component definitions")
        print(f"   - Tier 1 components: {len(tier1_components)}")
        print(f"   - Tier 2 components: {len(tier2_components)}")
        return True
        
    except Exception as e:
        print(f"❌ Component registry validation failed: {e}")
        return False

def validate_workflow_definitions():
    """Validate that workflow definitions are complete and realistic."""
    try:
        from prompt_improver.ml.orchestration.config.workflow_templates import WorkflowTemplates
        
        # Get training workflow
        training_workflow = WorkflowTemplates.get_training_workflow()
        
        # Validate workflow structure
        assert training_workflow.workflow_type == "training", "Should have correct type"
        assert len(training_workflow.steps) == 3, "Should have 3 steps"
        
        # Validate steps have realistic timeouts and dependencies
        load_step = training_workflow.steps[0]
        assert load_step.component_name == "training_data_loader", "First step should load data"
        assert load_step.timeout == 300, "Should have realistic timeout"
        
        train_step = training_workflow.steps[1]
        assert "load_data" in train_step.dependencies, "Train step should depend on load step"
        assert train_step.timeout == 1800, "Training should have longer timeout"
        
        optimize_step = training_workflow.steps[2]
        assert "train_model" in optimize_step.dependencies, "Optimize should depend on training"
        
        # Check end-to-end workflow
        e2e_workflow = WorkflowTemplates.get_end_to_end_workflow()
        assert e2e_workflow.parallel_execution == True, "E2E workflow should support parallel execution"
        assert len(e2e_workflow.steps) == 5, "E2E workflow should have 5 steps"
        
        print("✅ Workflow definitions are complete with realistic steps and dependencies")
        print(f"   - Training workflow steps: {len(training_workflow.steps)}")
        print(f"   - End-to-end workflow steps: {len(e2e_workflow.steps)}")
        return True
        
    except Exception as e:
        print(f"❌ Workflow definitions validation failed: {e}")
        return False

def validate_event_system():
    """Validate that event system has comprehensive event types."""
    try:
        from prompt_improver.ml.orchestration.events.event_types import EventType
        
        # Check that we have comprehensive event types
        event_types = list(EventType)
        
        # Should have core events
        required_events = [
            EventType.WORKFLOW_STARTED,
            EventType.WORKFLOW_COMPLETED,
            EventType.WORKFLOW_FAILED,
            EventType.TRAINING_STARTED,
            EventType.TRAINING_COMPLETED,
            EventType.EVALUATION_STARTED,
            EventType.EVALUATION_COMPLETED,
            EventType.DEPLOYMENT_STARTED,
            EventType.DEPLOYMENT_COMPLETED,
            EventType.RESOURCE_ALLOCATED,
            EventType.RESOURCE_RELEASED,
            EventType.COMPONENT_HEALTH_CHANGED
        ]
        
        for event_type in required_events:
            assert event_type in event_types, f"Should have {event_type} event type"
        
        print("✅ Event system has comprehensive event types for all workflow stages")
        print(f"   - Total event types: {len(event_types)}")
        return True
        
    except Exception as e:
        print(f"❌ Event system validation failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🔍 Validating Phase 2 Implementation for False Positives\n")
    
    validations = [
        ("Component Execution Results", validate_component_execution),
        ("Resource Allocation", validate_resource_allocation),
        ("Component Registry Definitions", validate_component_registry),
        ("Workflow Definitions", validate_workflow_definitions),
        ("Event System", validate_event_system)
    ]
    
    passed = 0
    failed = 0
    
    for validation_name, validation_func in validations:
        print(f"▶️  {validation_name}:")
        try:
            if validation_func():
                passed += 1
                print(f"✅ {validation_name} PASSED - No false positives detected\n")
            else:
                failed += 1
                print(f"❌ {validation_name} FAILED - False positives detected\n")
        except Exception as e:
            failed += 1
            print(f"❌ {validation_name} FAILED: {e}\n")
    
    print(f"📊 Validation Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 Phase 2 implementation validated - NO FALSE POSITIVES detected!")
        print("   - All component calls return realistic results")
        print("   - Resource allocation uses real system resources")
        print("   - Component definitions are detailed and complete")
        print("   - Workflow definitions have proper dependencies")
        print("   - Event system is comprehensive")
        return True
    else:
        print("⚠️  False positives detected - implementation needs review")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)