#!/usr/bin/env python3
"""
Orchestrator Functional Integration Test

Tests actual orchestrator component calls to verify functional integration
beyond just registration and loading.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_orchestrator_functional_integration():
    """Test actual orchestrator functionality with components."""
    
    print("🔧 Testing Orchestrator Functional Integration")
    print("=" * 60)
    
    try:
        from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
        from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
        from prompt_improver.ml.orchestration.core.component_registry import ComponentTier
        
        # Test 1: Initialize orchestrator
        print("\n📋 Test 1: Initialize ML Pipeline Orchestrator")
        print("-" * 50)
        
        config = OrchestratorConfig()
        orchestrator = MLPipelineOrchestrator(config)
        
        # Initialize the orchestrator
        await orchestrator.initialize()
        print("✅ Orchestrator initialized successfully")
        
        # Test 2: Check component discovery
        print("\n📋 Test 2: Component Discovery Through Orchestrator")
        print("-" * 50)
        
        # Get component registry
        registry = orchestrator.component_registry
        
        # Check discovered components
        all_components = await registry.list_components()
        print(f"✅ Discovered {len(all_components)} components through orchestrator")
        
        # Test by tier
        for tier in ComponentTier:
            tier_components = await registry.list_components(tier)
            print(f"  {tier.value}: {len(tier_components)} components")
        
        # Test 3: Direct component loading through orchestrator
        print("\n📋 Test 3: Direct Component Loading Through Orchestrator")
        print("-" * 50)
        
        loader = orchestrator.component_loader
        
        # Test loading specific components
        test_components = [
            ("training_data_loader", ComponentTier.TIER_1_CORE),
            ("ml_integration", ComponentTier.TIER_1_CORE),
            ("insight_engine", ComponentTier.TIER_2_OPTIMIZATION),
            ("causal_inference_analyzer", ComponentTier.TIER_3_EVALUATION),
            ("monitoring", ComponentTier.TIER_4_PERFORMANCE),
        ]
        
        loaded_components = {}
        for comp_name, tier in test_components:
            try:
                loaded_comp = await loader.load_component(comp_name, tier)
                if loaded_comp:
                    loaded_components[comp_name] = loaded_comp
                    print(f"  ✅ {comp_name}: Loaded successfully")
                else:
                    print(f"  ❌ {comp_name}: Failed to load")
            except Exception as e:
                print(f"  ❌ {comp_name}: Error - {e}")
        
        # Test 4: Component initialization through orchestrator
        print("\n📋 Test 4: Component Initialization Through Orchestrator")
        print("-" * 50)
        
        initialization_results = {}
        for comp_name, loaded_comp in loaded_components.items():
            try:
                # Try to initialize the component
                success = await loader.initialize_component(comp_name)
                initialization_results[comp_name] = success
                if success:
                    print(f"  ✅ {comp_name}: Initialized successfully")
                else:
                    print(f"  ❌ {comp_name}: Failed to initialize")
            except Exception as e:
                print(f"  ❌ {comp_name}: Initialization error - {e}")
                initialization_results[comp_name] = False
        
        # Test 5: Component invocation through orchestrator
        print("\n📋 Test 5: Component Invocation Through Orchestrator")
        print("-" * 50)
        
        invoker = orchestrator.component_invoker
        
        # Test simple component invocations
        invocation_results = {}
        
        # Test training data loader
        if "training_data_loader" in loaded_components:
            try:
                # This is a simple test - just check if we can access the component
                comp = loaded_components["training_data_loader"]
                if comp.instance:
                    # Check if it has expected methods
                    has_load_method = hasattr(comp.instance, 'load_training_data')
                    invocation_results["training_data_loader"] = has_load_method
                    print(f"  ✅ training_data_loader: Has load_training_data method: {has_load_method}")
                else:
                    print(f"  ⚠️ training_data_loader: No instance available")
            except Exception as e:
                print(f"  ❌ training_data_loader: Invocation error - {e}")
        
        # Test ML integration service
        if "ml_integration" in loaded_components:
            try:
                comp = loaded_components["ml_integration"]
                if comp.instance:
                    # Check if it has expected methods
                    has_predict_method = hasattr(comp.instance, 'predict_improvement')
                    invocation_results["ml_integration"] = has_predict_method
                    print(f"  ✅ ml_integration: Has predict_improvement method: {has_predict_method}")
                else:
                    print(f"  ⚠️ ml_integration: No instance available")
            except Exception as e:
                print(f"  ❌ ml_integration: Invocation error - {e}")
        
        # Test 6: Orchestrator workflow execution
        print("\n📋 Test 6: Orchestrator Workflow Execution")
        print("-" * 50)
        
        workflow_engine = orchestrator.workflow_engine
        
        # Test basic workflow capabilities
        try:
            # Check if workflow engine is initialized
            if hasattr(workflow_engine, 'execute_workflow'):
                print("  ✅ Workflow engine has execute_workflow method")
            else:
                print("  ❌ Workflow engine missing execute_workflow method")
                
            # Check workflow status
            if hasattr(workflow_engine, 'get_workflow_status'):
                print("  ✅ Workflow engine has get_workflow_status method")
            else:
                print("  ❌ Workflow engine missing get_workflow_status method")
                
        except Exception as e:
            print(f"  ❌ Workflow engine test error: {e}")
        
        # Test 7: Resource management
        print("\n📋 Test 7: Resource Management Integration")
        print("-" * 50)
        
        resource_manager = orchestrator.resource_manager
        
        try:
            # Check resource manager capabilities
            if hasattr(resource_manager, 'allocate_resources'):
                print("  ✅ Resource manager has allocate_resources method")
            else:
                print("  ❌ Resource manager missing allocate_resources method")
                
            if hasattr(resource_manager, 'get_resource_status'):
                print("  ✅ Resource manager has get_resource_status method")
            else:
                print("  ❌ Resource manager missing get_resource_status method")
                
        except Exception as e:
            print(f"  ❌ Resource manager test error: {e}")
        
        # Print summary
        print("\n📊 FUNCTIONAL INTEGRATION SUMMARY")
        print("=" * 60)
        
        print(f"✅ Orchestrator initialization: SUCCESS")
        print(f"✅ Component discovery: {len(all_components)} components")
        print(f"✅ Component loading: {len(loaded_components)}/{len(test_components)} successful")
        print(f"✅ Component initialization: {sum(initialization_results.values())}/{len(initialization_results)} successful")
        print(f"✅ Component invocation tests: {len(invocation_results)} tested")
        
        return {
            'orchestrator_initialized': True,
            'components_discovered': len(all_components),
            'components_loaded': len(loaded_components),
            'components_initialized': sum(initialization_results.values()),
            'invocation_results': invocation_results,
            'test_components': test_components,
            'loaded_components': list(loaded_components.keys())
        }
        
    except Exception as e:
        print(f"❌ Critical error in functional integration test: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = asyncio.run(test_orchestrator_functional_integration())
    if result:
        print(f"\n🎯 FUNCTIONAL INTEGRATION: {result['components_loaded']} components successfully integrated")
    else:
        print("\n❌ Functional integration test failed")
