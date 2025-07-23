#!/usr/bin/env python3
"""
Simple Phase 2 test to verify core functionality without complex async operations.
"""

import sys
from pathlib import Path

# Add the src directory to Python path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all Phase 2 components can be imported."""
    try:
        from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
        from prompt_improver.ml.orchestration.core.workflow_execution_engine import WorkflowExecutionEngine
        from prompt_improver.ml.orchestration.core.resource_manager import ResourceManager
        from prompt_improver.ml.orchestration.core.component_registry import ComponentRegistry
        from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
        from prompt_improver.ml.orchestration.config.workflow_templates import WorkflowTemplates
        
        print("✅ All Phase 2 imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_workflow_templates():
    """Test workflow template creation."""
    try:
        from prompt_improver.ml.orchestration.config.workflow_templates import WorkflowTemplates
        
        templates = WorkflowTemplates.get_all_workflow_templates()
        print(f"✅ Found {len(templates)} workflow templates")
        
        # Check if 'training' workflow exists
        training_found = any(t.workflow_type == "training" for t in templates)
        if training_found:
            print("✅ Training workflow template found")
            return True
        else:
            print("❌ Training workflow template not found")
            return False
            
    except Exception as e:
        print(f"❌ Workflow template test failed: {e}")
        return False

def test_component_definitions():
    """Test component definitions."""
    try:
        from prompt_improver.ml.orchestration.config.component_definitions import ComponentDefinitions
        from prompt_improver.ml.orchestration.core.component_registry import ComponentTier
        
        comp_defs = ComponentDefinitions()
        tier1_components = comp_defs.get_tier_components(ComponentTier.TIER_1_CORE)
        tier2_components = comp_defs.get_tier_components(ComponentTier.TIER_2_OPTIMIZATION)
        
        print(f"✅ Tier 1 components: {len(tier1_components)}")
        print(f"✅ Tier 2 components: {len(tier2_components)}")
        
        # Test component info creation
        sample_component = comp_defs.create_component_info(
            "test_component", 
            {"description": "Test", "capabilities": ["test"]},
            ComponentTier.TIER_1_CORE
        )
        
        print(f"✅ Component info creation successful: {sample_component.name}")
        return True
        
    except Exception as e:
        print(f"❌ Component definitions test failed: {e}")
        return False

def test_orchestrator_creation():
    """Test orchestrator creation without initialization."""
    try:
        from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
        from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
        
        config = OrchestratorConfig()
        orchestrator = MLPipelineOrchestrator(config)
        
        print(f"✅ Orchestrator created successfully")
        print(f"   - Event bus: {type(orchestrator.event_bus).__name__}")
        print(f"   - Workflow engine: {type(orchestrator.workflow_engine).__name__}")
        print(f"   - Resource manager: {type(orchestrator.resource_manager).__name__}")
        print(f"   - Component registry: {type(orchestrator.component_registry).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator creation failed: {e}")
        return False

def main():
    """Run all simple tests."""
    print("🚀 Running Simple Phase 2 Tests\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Workflow Templates", test_workflow_templates),
        ("Component Definitions", test_component_definitions),
        ("Orchestrator Creation", test_orchestrator_creation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"▶️  {test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED\n")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED: {e}\n")
    
    print(f"📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Phase 2 core tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)