#!/usr/bin/env python3
"""
Validate Phase 6 implementation by actually running the components.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_direct_component_loading():
    """Test that DirectComponentLoader can actually load real components."""
    print("\n🔍 Testing DirectComponentLoader with real components...")
    
    from prompt_improver.ml.orchestration.integration.direct_component_loader import DirectComponentLoader, ComponentTier
    
    loader = DirectComponentLoader()
    
    # Test loading specific components that should exist
    test_components = [
        ("training_data_loader", ComponentTier.TIER_1_CORE),
        ("ml_integration", ComponentTier.TIER_1_CORE),
        ("rule_optimizer", ComponentTier.TIER_1_CORE),
    ]
    
    loaded_count = 0
    for comp_name, tier in test_components:
        try:
            loaded = await loader.load_component(comp_name, tier)
            if loaded:
                print(f"  ✅ Loaded {comp_name}: {loaded.component_class.__name__}")
                loaded_count += 1
            else:
                print(f"  ❌ Failed to load {comp_name}")
        except Exception as e:
            print(f"  ❌ Error loading {comp_name}: {e}")
    
    print(f"\n  Summary: Loaded {loaded_count}/{len(test_components)} components")
    return loaded_count == len(test_components)


async def test_orchestrator_initialization():
    """Test that the orchestrator can initialize with direct components."""
    print("\n🎯 Testing MLPipelineOrchestrator initialization...")
    
    from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
    from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
    
    try:
        config = OrchestratorConfig()
        orchestrator = MLPipelineOrchestrator(config)
        
        print(f"  Initial state: {orchestrator.state.value}")
        
        # Initialize orchestrator
        await orchestrator.initialize()
        
        print(f"  ✅ Initialized: {orchestrator._is_initialized}")
        
        # Check loaded components
        components = orchestrator.get_loaded_components()
        print(f"  ✅ Loaded components: {len(components)}")
        
        if components:
            print(f"  Sample components: {components[:5]}")
        
        # Test shutdown
        await orchestrator.shutdown()
        print("  ✅ Shutdown successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_component_invocation():
    """Test that we can invoke methods on components."""
    print("\n🚀 Testing component method invocation...")
    
    from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
    from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
    
    try:
        config = OrchestratorConfig()
        orchestrator = MLPipelineOrchestrator(config)
        await orchestrator.initialize()
        
        # Get a component to test
        components = orchestrator.get_loaded_components()
        if not components:
            print("  ❌ No components loaded")
            return False
        
        test_component = components[0]
        methods = orchestrator.get_component_methods(test_component)
        
        print(f"  Testing component: {test_component}")
        print(f"  Available methods: {methods}")
        
        # Try to get invocation history (should be empty initially)
        history = orchestrator.get_invocation_history()
        print(f"  Initial history: {len(history)} entries")
        
        await orchestrator.shutdown()
        return True
        
    except Exception as e:
        print(f"  ❌ Component invocation test failed: {e}")
        return False


async def test_cli_command():
    """Test the orchestrator CLI command."""
    print("\n🖥️  Testing CLI orchestrator command...")
    
    import subprocess
    
    try:
        # Test orchestrator status command
        result = subprocess.run(
            ["python3", "-m", "prompt_improver.cli", "orchestrator", "status"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("  ✅ CLI orchestrator status command works")
            print(f"  Output preview: {result.stdout[:200]}...")
            return True
        else:
            print(f"  ❌ CLI command failed with code {result.returncode}")
            print(f"  Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ CLI test failed: {e}")
        return False


async def main():
    """Run all validation tests."""
    print("🧪 Phase 6 Implementation Validation")
    print("=" * 50)
    
    tests = [
        ("Direct Component Loading", test_direct_component_loading),
        ("Orchestrator Initialization", test_orchestrator_initialization),
        ("Component Invocation", test_component_invocation),
        ("CLI Command", test_cli_command),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        print("-" * 40)
        
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"💥 Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Validation Results")
    print("-" * 40)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed < total:
        print("\n⚠️  Some tests failed - Phase 6 needs fixes!")
    else:
        print("\n✅ All validation tests passed!")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)