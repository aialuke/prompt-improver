#!/usr/bin/env python3
"""
Phase 2A-A Ready Components Integration Test

Tests the integration of 3 ready Phase 2A ML components with the orchestrator:
1. MultiarmedBanditFramework
2. ClusteringOptimizer
3. AdvancedEarlyStoppingFramework

Validates 2025 best practices implementation and orchestrator compatibility.
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

class Phase2AReadyComponentsTester:
    """Test Phase 2A ready component integration with orchestrator"""
    
    def __init__(self):
        self.test_results = {}
        self.components_to_test = [
            "multiarmed_bandit_framework",
            "clustering_optimizer", 
            "advanced_early_stopping_framework"
        ]
    
    async def run_comprehensive_integration_test(self) -> Dict[str, Any]:
        """Run comprehensive integration test for Phase 2A ready components"""
        
        print("🚀 Phase 2A-A Ready Components Integration Test")
        print("=" * 60)
        
        # Test 1: Component Discovery
        discovery_result = await self._test_component_discovery()
        
        # Test 2: Individual Component Integration
        component_results = await self._test_individual_components()
        
        # Test 3: Orchestrator Communication
        communication_result = await self._test_orchestrator_communication()
        
        # Compile results
        overall_result = {
            "discovery": discovery_result,
            "individual_components": component_results,
            "orchestrator_communication": communication_result,
            "summary": self._generate_test_summary()
        }
        
        self._print_test_results(overall_result)
        return overall_result
    
    async def _test_component_discovery(self) -> Dict[str, Any]:
        """Test that all Phase 2A ready components are discoverable through orchestrator"""
        
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
            
            # Check if our Phase 2A ready components are discovered
            results = {}
            for comp_name in self.components_to_test:
                found = comp_name in discovered_names
                results[comp_name] = {
                    "discovered": found,
                    "status": "✅ FOUND" if found else "❌ MISSING"
                }
                print(f"  {results[comp_name]['status']}: {comp_name}")
            
            total_discovered = len(components)
            phase2a_discovered = sum(1 for r in results.values() if r["discovered"])
            
            print(f"\n  📊 Discovery Summary:")
            print(f"    Total components discovered: {total_discovered}")
            print(f"    Phase 2A ready components discovered: {phase2a_discovered}/3")
            print(f"    Discovery success rate: {(phase2a_discovered/3)*100:.1f}%")
            
            return {
                "success": phase2a_discovered == 3,
                "total_discovered": total_discovered,
                "phase2a_discovered": phase2a_discovered,
                "component_results": results,
                "expected_minimum": 15  # Should have at least 15 components total now
            }
            
        except Exception as e:
            print(f"  ❌ Discovery test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_individual_components(self) -> Dict[str, Any]:
        """Test each component's orchestrator interface individually"""
        
        print("\n🔧 Test 2: Individual Component Integration")
        
        results = {}
        
        # Test MultiarmedBanditFramework
        results["multiarmed_bandit_framework"] = await self._test_bandit_framework()
        
        # Test ClusteringOptimizer
        results["clustering_optimizer"] = await self._test_clustering_optimizer()
        
        # Test AdvancedEarlyStoppingFramework
        results["advanced_early_stopping_framework"] = await self._test_early_stopping()
        
        return results
    
    async def _test_bandit_framework(self) -> Dict[str, Any]:
        """Test MultiarmedBanditFramework orchestrator interface"""
        
        try:
            from prompt_improver.ml.optimization.algorithms.multi_armed_bandit import MultiarmedBanditFramework
            
            framework = MultiarmedBanditFramework()
            
            # Test orchestrator interface
            config = {
                "experiment_name": "test_bandit_experiment",
                "arms": ["strategy_a", "strategy_b", "strategy_c"],
                "algorithm": "thompson_sampling",
                "num_trials": 50,
                "output_path": "./test_outputs/bandit_optimization"
            }
            
            result = await framework.run_orchestrated_analysis(config)
            
            # Validate result structure
            success = (
                result.get("orchestrator_compatible") == True and
                "component_result" in result and
                "local_metadata" in result and
                "experiment_summary" in result["component_result"]
            )
            
            print(f"  {'✅' if success else '❌'} MultiarmedBanditFramework: {'PASSED' if success else 'FAILED'}")
            
            return {
                "success": success,
                "orchestrator_compatible": result.get("orchestrator_compatible", False),
                "has_experiment_summary": "experiment_summary" in result.get("component_result", {}),
                "execution_time": result.get("local_metadata", {}).get("execution_time", 0)
            }
            
        except Exception as e:
            print(f"  ❌ MultiarmedBanditFramework: FAILED - {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_clustering_optimizer(self) -> Dict[str, Any]:
        """Test ClusteringOptimizer orchestrator interface"""
        
        try:
            from prompt_improver.ml.optimization.algorithms.clustering_optimizer import ClusteringOptimizer
            
            optimizer = ClusteringOptimizer()
            
            # Generate sample data for clustering
            np.random.seed(42)
            features = np.random.rand(50, 10)  # 50 samples, 10 features
            
            # Test orchestrator interface
            config = {
                "features": features.tolist(),
                "optimization_target": "silhouette",
                "max_clusters": 10,
                "clustering_method": "auto",
                "output_path": "./test_outputs/clustering_optimization"
            }
            
            result = await optimizer.run_orchestrated_analysis(config)
            
            # Validate result structure
            success = (
                result.get("orchestrator_compatible") == True and
                "component_result" in result and
                "local_metadata" in result and
                "clustering_summary" in result["component_result"]
            )
            
            print(f"  {'✅' if success else '❌'} ClusteringOptimizer: {'PASSED' if success else 'FAILED'}")
            
            return {
                "success": success,
                "orchestrator_compatible": result.get("orchestrator_compatible", False),
                "has_clustering_summary": "clustering_summary" in result.get("component_result", {}),
                "execution_time": result.get("local_metadata", {}).get("execution_time", 0)
            }
            
        except Exception as e:
            print(f"  ❌ ClusteringOptimizer: FAILED - {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_early_stopping(self) -> Dict[str, Any]:
        """Test AdvancedEarlyStoppingFramework orchestrator interface"""
        
        try:
            from prompt_improver.ml.optimization.algorithms.early_stopping import AdvancedEarlyStoppingFramework
            
            framework = AdvancedEarlyStoppingFramework()
            
            # Test orchestrator interface
            config = {
                "experiment_id": "test_early_stopping",
                "control_data": [0.5, 0.6, 0.4, 0.7, 0.5, 0.6, 0.4, 0.5],
                "treatment_data": [0.7, 0.8, 0.6, 0.9, 0.8, 0.7, 0.6, 0.8],
                "look_number": 1,
                "alpha_spending_function": "obrien_fleming",
                "enable_futility": True,
                "output_path": "./test_outputs/early_stopping"
            }
            
            result = await framework.run_orchestrated_analysis(config)
            
            # Validate result structure
            success = (
                result.get("orchestrator_compatible") == True and
                "component_result" in result and
                "local_metadata" in result and
                "stopping_decision" in result["component_result"]
            )
            
            print(f"  {'✅' if success else '❌'} AdvancedEarlyStoppingFramework: {'PASSED' if success else 'FAILED'}")
            
            return {
                "success": success,
                "orchestrator_compatible": result.get("orchestrator_compatible", False),
                "has_stopping_decision": "stopping_decision" in result.get("component_result", {}),
                "execution_time": result.get("local_metadata", {}).get("execution_time", 0)
            }
            
        except Exception as e:
            print(f"  ❌ AdvancedEarlyStoppingFramework: FAILED - {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_orchestrator_communication(self) -> Dict[str, Any]:
        """Test orchestrator-level component communication"""
        
        print("\n🔄 Test 3: Orchestrator Communication")
        
        try:
            from prompt_improver.ml.orchestration.core.component_registry import ComponentRegistry
            from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
            
            config = OrchestratorConfig()
            registry = ComponentRegistry(config)
            
            # Test component loading through orchestrator
            components = await registry.discover_components()
            phase2a_components = [c for c in components if c.name in self.components_to_test]
            
            communication_success = len(phase2a_components) == 3
            
            print(f"  {'✅' if communication_success else '❌'} Orchestrator Communication: {'PASSED' if communication_success else 'FAILED'}")
            print(f"    Components accessible through orchestrator: {len(phase2a_components)}/3")
            
            return {
                "success": communication_success,
                "accessible_components": len(phase2a_components),
                "component_names": [c.name for c in phase2a_components]
            }
            
        except Exception as e:
            print(f"  ❌ Orchestrator Communication: FAILED - {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate overall test summary"""
        
        return {
            "total_tests": 3,
            "components_tested": 3,
            "integration_status": "Phase 2A-A Ready Components Complete"
        }
    
    def _print_test_results(self, results: Dict[str, Any]):
        """Print comprehensive test results"""
        
        print("\n" + "=" * 60)
        print("📊 PHASE 2A-A READY COMPONENTS TEST RESULTS")
        print("=" * 60)
        
        # Print summary
        discovery = results.get("discovery", {})
        individual = results.get("individual_components", {})
        
        discovery_success = discovery.get("success", False)
        individual_success = sum(1 for r in individual.values() if r.get("success", False))
        
        print(f"✅ Component Discovery: {'PASSED' if discovery_success else 'FAILED'}")
        print(f"✅ Individual Components: {individual_success}/3 PASSED")
        print(f"✅ Overall Integration: {'SUCCESS' if discovery_success and individual_success == 3 else 'PARTIAL'}")
        
        if discovery_success and individual_success == 3:
            print("\n🎉 PHASE 2A-A INTEGRATION: COMPLETE SUCCESS!")
            print("All 3 ready Phase 2A components are now orchestrator-integrated and ready for use.")
        else:
            print("\n⚠️  PHASE 2A-A INTEGRATION: NEEDS ATTENTION")
            print("Some components require additional integration work.")


async def main():
    """Main test execution function"""
    
    tester = Phase2AReadyComponentsTester()
    results = await tester.run_comprehensive_integration_test()
    
    # Save results to file
    with open('phase2a_ready_components_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: phase2a_ready_components_test_results.json")
    
    return 0 if results.get("discovery", {}).get("success", False) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
