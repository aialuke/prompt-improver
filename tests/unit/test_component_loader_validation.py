#!/usr/bin/env python3
"""
Comprehensive Component Loader Validation Test

Tests the DirectComponentLoader to verify which components actually load successfully
and validates the integration audit analysis accuracy.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_component_loader():
    """Test DirectComponentLoader functionality comprehensively."""
    
    print("🧪 Testing DirectComponentLoader Functionality")
    print("=" * 60)
    
    try:
        from prompt_improver.ml.orchestration.integration.direct_component_loader import DirectComponentLoader
        from prompt_improver.ml.orchestration.core.component_registry import ComponentTier
        
        loader = DirectComponentLoader()
        
        # Test 1: Load all components across all tiers
        print("\n📋 Test 1: Loading All Components Across All Tiers")
        print("-" * 50)
        
        all_loaded = await loader.load_all_components()
        
        print(f"✅ Total components loaded: {len(all_loaded)}")
        
        # Test 2: Load components by tier
        print("\n📋 Test 2: Loading Components by Tier")
        print("-" * 50)
        
        tier_results = {}
        for tier in ComponentTier:
            try:
                tier_components = await loader.load_tier_components(tier)
                tier_results[tier] = {
                    'loaded': len(tier_components),
                    'components': list(tier_components.keys()),
                    'failed': []
                }
                print(f"  {tier.value}: {len(tier_components)} components loaded")
                
                # Test individual component loading
                for comp_name in loader.component_paths.get(tier, {}):
                    if comp_name not in tier_components:
                        tier_results[tier]['failed'].append(comp_name)
                        
            except Exception as e:
                print(f"  ❌ {tier.value}: Failed to load - {e}")
                tier_results[tier] = {'error': str(e)}
        
        # Test 3: Verify specific components from audit
        print("\n📋 Test 3: Verifying Specific Components from Audit")
        print("-" * 50)
        
        # Test components I identified as INTEGRATED
        integrated_test_cases = [
            ("training_data_loader", ComponentTier.TIER_1_CORE),
            ("ml_integration", ComponentTier.TIER_1_CORE),
            ("rule_optimizer", ComponentTier.TIER_1_CORE),
            ("multi_armed_bandit", ComponentTier.TIER_1_CORE),
            ("insight_engine", ComponentTier.TIER_2_OPTIMIZATION),
            ("causal_inference_analyzer", ComponentTier.TIER_3_EVALUATION),
            ("monitoring", ComponentTier.TIER_4_PERFORMANCE),
            ("adversarial_defense", ComponentTier.TIER_6_SECURITY),
        ]
        
        integration_validation = {}
        for comp_name, tier in integrated_test_cases:
            try:
                loaded_comp = await loader.load_component(comp_name, tier)
                if loaded_comp:
                    integration_validation[comp_name] = "✅ CONFIRMED INTEGRATED"
                else:
                    integration_validation[comp_name] = "❌ FAILED TO LOAD"
            except Exception as e:
                integration_validation[comp_name] = f"❌ ERROR: {e}"
        
        # Test 4: Check components I identified as NOT INTEGRATED
        print("\n📋 Test 4: Checking Components Identified as NOT INTEGRATED")
        print("-" * 50)
        
        not_integrated_test_cases = [
            "context_cache_manager",
            "composite_extractor", 
            "linguistic_extractor",
            "performance_monitor",
            "health_monitor",
        ]
        
        not_integrated_validation = {}
        for comp_name in not_integrated_test_cases:
            # Try to find this component in any tier
            found_in_tier = None
            for tier in ComponentTier:
                if comp_name in loader.component_paths.get(tier, {}):
                    found_in_tier = tier
                    break
            
            if found_in_tier:
                not_integrated_validation[comp_name] = f"⚠️ FOUND IN {found_in_tier.value} - AUDIT ERROR"
            else:
                not_integrated_validation[comp_name] = "✅ CONFIRMED NOT IN LOADER"
        
        # Print detailed results
        print("\n📊 DETAILED VALIDATION RESULTS")
        print("=" * 60)
        
        print("\n🔍 Tier-by-Tier Loading Results:")
        for tier, result in tier_results.items():
            if 'error' in result:
                print(f"  {tier.value}: ❌ {result['error']}")
            else:
                print(f"  {tier.value}: {result['loaded']} loaded")
                if result['failed']:
                    print(f"    Failed: {result['failed']}")
        
        print("\n🔍 Integration Validation Results:")
        for comp, status in integration_validation.items():
            print(f"  {comp}: {status}")
        
        print("\n🔍 Not-Integrated Validation Results:")
        for comp, status in not_integrated_validation.items():
            print(f"  {comp}: {status}")
        
        return {
            'total_loaded': len(all_loaded),
            'tier_results': tier_results,
            'integration_validation': integration_validation,
            'not_integrated_validation': not_integrated_validation,
            'all_loaded_components': list(all_loaded.keys())
        }
        
    except Exception as e:
        print(f"❌ Critical error in component loader test: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = asyncio.run(test_component_loader())
    if result:
        print(f"\n🎯 SUMMARY: {result['total_loaded']} components successfully loaded through DirectComponentLoader")
    else:
        print("\n❌ Component loader test failed")
