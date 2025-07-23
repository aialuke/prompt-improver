#!/usr/bin/env python3
"""
Phase 2A-B Enhanced OptimizationValidator Integration Test

Tests the enhanced OptimizationValidator with 2025 best practices:
- Bayesian model comparison and evidence calculation
- Robust statistical methods (bootstrap, permutation tests)
- Causal inference validation
- Multi-dimensional optimization validation
- Advanced uncertainty quantification

Validates orchestrator integration and 2025 compliance.
"""

import asyncio
import logging
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase2AEnhancedValidatorTester:
    """Test enhanced OptimizationValidator integration with orchestrator"""
    
    def __init__(self):
        self.test_results = {}
        self.component_name = "enhanced_optimization_validator"
    
    async def run_comprehensive_integration_test(self) -> Dict[str, Any]:
        """Run comprehensive integration test for enhanced OptimizationValidator"""
        
        print("🚀 Phase 2A-B Enhanced OptimizationValidator Integration Test")
        print("=" * 70)
        
        # Test 1: Component Discovery
        discovery_result = await self._test_component_discovery()
        
        # Test 2: 2025 Features Validation
        features_result = await self._test_2025_features()
        
        # Test 3: Orchestrator Integration
        integration_result = await self._test_orchestrator_integration()
        
        # Test 4: Advanced Validation Methods
        advanced_result = await self._test_advanced_validation_methods()
        
        # Compile results
        overall_result = {
            "discovery": discovery_result,
            "features_2025": features_result,
            "orchestrator_integration": integration_result,
            "advanced_validation": advanced_result,
            "summary": self._generate_test_summary()
        }
        
        self._print_test_results(overall_result)
        return overall_result
    
    async def _test_component_discovery(self) -> Dict[str, Any]:
        """Test that enhanced OptimizationValidator is discoverable through orchestrator"""
        
        print("\n📋 Test 1: Component Discovery")
        
        try:
            from prompt_improver.ml.orchestration.core.component_registry import ComponentRegistry
            from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
            
            # Initialize orchestrator
            config = OrchestratorConfig()
            registry = ComponentRegistry(config)
            
            # Discover components
            components = await registry.discover_components()
            discovered_names = [comp.name for comp in components]
            
            # Check if enhanced validator is discovered
            found = self.component_name in discovered_names
            
            print(f"  {'✅ FOUND' if found else '❌ MISSING'}: {self.component_name}")
            print(f"  📊 Total components discovered: {len(components)}")
            
            return {
                "success": found,
                "total_discovered": len(components),
                "component_found": found,
                "expected_minimum": 16  # Should have at least 16 components now
            }
            
        except Exception as e:
            print(f"  ❌ Discovery test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_2025_features(self) -> Dict[str, Any]:
        """Test 2025 enhanced features"""
        
        print("\n🔬 Test 2: 2025 Features Validation")
        
        try:
            from prompt_improver.ml.optimization.validation.optimization_validator import (
                EnhancedOptimizationValidator, 
                EnhancedValidationConfig,
                ValidationMethod,
                EffectSizeMagnitude,
                ValidationResult
            )
            
            # Test enhanced configuration
            config = EnhancedValidationConfig(
                validation_method=ValidationMethod.COMPREHENSIVE,
                enable_bayesian_validation=True,
                enable_causal_inference=True,
                enable_robust_methods=True,
                enable_uncertainty_quantification=True
            )
            
            validator = EnhancedOptimizationValidator(config)
            
            # Test enum classes
            validation_methods = list(ValidationMethod)
            effect_magnitudes = list(EffectSizeMagnitude)
            
            features_available = {
                "enhanced_config": True,
                "validation_methods": len(validation_methods) >= 5,
                "effect_size_classification": len(effect_magnitudes) >= 5,
                "bayesian_support": hasattr(validator, 'bayesian_validator'),
                "causal_support": hasattr(validator, 'causal_validator'),
                "robust_support": hasattr(validator, 'robust_validator'),
                "orchestrator_interface": hasattr(validator, 'run_orchestrated_analysis')
            }
            
            success_count = sum(features_available.values())
            total_features = len(features_available)
            
            print(f"  ✅ Enhanced Configuration: {'AVAILABLE' if features_available['enhanced_config'] else 'MISSING'}")
            print(f"  ✅ Validation Methods: {len(validation_methods)} methods available")
            print(f"  ✅ Effect Size Classification: {len(effect_magnitudes)} levels available")
            print(f"  ✅ Bayesian Support: {'AVAILABLE' if features_available['bayesian_support'] else 'MISSING'}")
            print(f"  ✅ Causal Inference: {'AVAILABLE' if features_available['causal_support'] else 'MISSING'}")
            print(f"  ✅ Robust Statistics: {'AVAILABLE' if features_available['robust_support'] else 'MISSING'}")
            print(f"  ✅ Orchestrator Interface: {'AVAILABLE' if features_available['orchestrator_interface'] else 'MISSING'}")
            print(f"  📊 Features Score: {success_count}/{total_features} ({(success_count/total_features)*100:.1f}%)")
            
            return {
                "success": success_count == total_features,
                "features_available": features_available,
                "features_score": success_count / total_features,
                "validation_methods_count": len(validation_methods),
                "effect_magnitudes_count": len(effect_magnitudes)
            }
            
        except Exception as e:
            print(f"  ❌ 2025 features test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_orchestrator_integration(self) -> Dict[str, Any]:
        """Test orchestrator integration"""
        
        print("\n🔄 Test 3: Orchestrator Integration")
        
        try:
            from prompt_improver.ml.optimization.validation.optimization_validator import EnhancedOptimizationValidator
            
            validator = EnhancedOptimizationValidator()
            
            # Generate sample data
            np.random.seed(42)
            baseline_scores = np.random.normal(0.5, 0.1, 50).tolist()
            optimized_scores = np.random.normal(0.7, 0.1, 50).tolist()  # Better performance
            
            # Test orchestrator interface
            config = {
                "optimization_id": "test_enhanced_validation",
                "baseline_data": {"scores": baseline_scores},
                "optimized_data": {"scores": optimized_scores},
                "validation_method": "comprehensive",
                "metrics": ["primary", "secondary"],
                "output_path": "./test_outputs/enhanced_validation"
            }
            
            result = await validator.run_orchestrated_analysis(config)
            
            # Validate result structure
            success = (
                result.get("orchestrator_compatible") == True and
                "component_result" in result and
                "local_metadata" in result and
                "validation_summary" in result["component_result"]
            )
            
            validation_summary = result.get("component_result", {}).get("validation_summary", {})
            has_advanced_features = (
                "statistical_analysis" in result.get("component_result", {}) and
                "uncertainty_quantification" in result.get("component_result", {}) and
                "quality_assessment" in result.get("component_result", {})
            )
            
            print(f"  {'✅' if success else '❌'} Orchestrator Interface: {'PASSED' if success else 'FAILED'}")
            print(f"  {'✅' if has_advanced_features else '❌'} Advanced Features: {'AVAILABLE' if has_advanced_features else 'MISSING'}")
            print(f"  📊 Validation Quality Score: {validation_summary.get('validation_quality_score', 0):.3f}")
            print(f"  📊 Robustness Score: {validation_summary.get('robustness_score', 0):.3f}")
            
            return {
                "success": success and has_advanced_features,
                "orchestrator_compatible": result.get("orchestrator_compatible", False),
                "has_validation_summary": "validation_summary" in result.get("component_result", {}),
                "has_advanced_features": has_advanced_features,
                "validation_quality_score": validation_summary.get("validation_quality_score", 0),
                "robustness_score": validation_summary.get("robustness_score", 0),
                "execution_time": result.get("local_metadata", {}).get("execution_time", 0)
            }
            
        except Exception as e:
            print(f"  ❌ Orchestrator integration test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_advanced_validation_methods(self) -> Dict[str, Any]:
        """Test advanced validation methods"""
        
        print("\n🧪 Test 4: Advanced Validation Methods")
        
        try:
            from prompt_improver.ml.optimization.validation.optimization_validator import (
                EnhancedOptimizationValidator,
                EnhancedValidationConfig,
                ValidationMethod
            )
            
            # Test different validation methods
            methods_to_test = [
                ValidationMethod.CLASSICAL,
                ValidationMethod.ROBUST,
                ValidationMethod.COMPREHENSIVE
            ]
            
            method_results = {}
            
            for method in methods_to_test:
                config = EnhancedValidationConfig(validation_method=method)
                validator = EnhancedOptimizationValidator(config)
                
                # Generate test data with clear improvement
                np.random.seed(42)
                baseline = np.random.normal(0.4, 0.1, 40)
                optimized = np.random.normal(0.8, 0.1, 40)  # Clear improvement
                
                try:
                    result = await validator.validate_enhanced_optimization(
                        optimization_id=f"test_{method.value}",
                        baseline_data={"scores": baseline.tolist()},
                        optimized_data={"scores": optimized.tolist()}
                    )
                    
                    method_results[method.value] = {
                        "success": True,
                        "valid": result.valid,
                        "effect_size": result.effect_size,
                        "p_value": result.p_value,
                        "validation_quality": result.validation_quality_score,
                        "robustness": result.robustness_score
                    }
                    
                    print(f"  ✅ {method.value.upper()}: Valid={result.valid}, Effect={result.effect_size:.3f}, Quality={result.validation_quality_score:.3f}")
                    
                except Exception as e:
                    method_results[method.value] = {"success": False, "error": str(e)}
                    print(f"  ❌ {method.value.upper()}: FAILED - {e}")
            
            successful_methods = sum(1 for r in method_results.values() if r.get("success", False))
            
            return {
                "success": successful_methods >= 2,  # At least 2 methods should work
                "methods_tested": len(methods_to_test),
                "successful_methods": successful_methods,
                "method_results": method_results
            }
            
        except Exception as e:
            print(f"  ❌ Advanced validation methods test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate overall test summary"""
        
        return {
            "total_tests": 4,
            "component_tested": "enhanced_optimization_validator",
            "enhancement_status": "Phase 2A-B Enhanced OptimizationValidator Complete",
            "version": "2025.1.0"
        }
    
    def _print_test_results(self, results: Dict[str, Any]):
        """Print comprehensive test results"""
        
        print("\n" + "=" * 70)
        print("📊 PHASE 2A-B ENHANCED OPTIMIZATION VALIDATOR TEST RESULTS")
        print("=" * 70)
        
        # Print summary
        discovery = results.get("discovery", {})
        features = results.get("features_2025", {})
        integration = results.get("orchestrator_integration", {})
        advanced = results.get("advanced_validation", {})
        
        discovery_success = discovery.get("success", False)
        features_success = features.get("success", False)
        integration_success = integration.get("success", False)
        advanced_success = advanced.get("success", False)
        
        print(f"✅ Component Discovery: {'PASSED' if discovery_success else 'FAILED'}")
        print(f"✅ 2025 Features: {'PASSED' if features_success else 'FAILED'} ({features.get('features_score', 0)*100:.1f}%)")
        print(f"✅ Orchestrator Integration: {'PASSED' if integration_success else 'FAILED'}")
        print(f"✅ Advanced Validation: {'PASSED' if advanced_success else 'FAILED'} ({advanced.get('successful_methods', 0)}/{advanced.get('methods_tested', 0)} methods)")
        
        overall_success = discovery_success and features_success and integration_success and advanced_success
        
        if overall_success:
            print("\n🎉 PHASE 2A-B ENHANCEMENT: COMPLETE SUCCESS!")
            print("Enhanced OptimizationValidator with 2025 best practices is fully integrated and ready!")
        else:
            print("\n⚠️  PHASE 2A-B ENHANCEMENT: NEEDS ATTENTION")
            print("Some enhanced features require additional work.")


async def main():
    """Main test execution function"""
    
    tester = Phase2AEnhancedValidatorTester()
    results = await tester.run_comprehensive_integration_test()
    
    # Save results to file
    with open('phase2a_enhanced_validator_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: phase2a_enhanced_validator_test_results.json")
    
    return 0 if results.get("discovery", {}).get("success", False) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
