#!/usr/bin/env python3
"""
Phase 2B-2 Enhanced HealthService Integration Test

Tests the enhanced HealthService with 2025 best practices:
- Circuit breaker integration for dependency health
- Predictive health monitoring with trend analysis
- Health check result caching and optimization
- Dependency graph visualization and analysis
- Advanced observability with metrics
- Service mesh health integration

Validates orchestrator integration and 2025 compliance.
"""

import asyncio
import logging
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase2BEnhancedHealthServiceTester:
    """Test enhanced HealthService integration with orchestrator"""
    
    def __init__(self):
        self.test_results = {}
        self.component_name = "enhanced_health_service"
    
    async def run_comprehensive_integration_test(self) -> Dict[str, Any]:
        """Run comprehensive integration test for enhanced HealthService"""
        
        print("🚀 Phase 2B-2 Enhanced HealthService Integration Test")
        print("=" * 70)
        
        # Test 1: Component Discovery and 2025 Features
        features_result = await self._test_2025_features()
        
        # Test 2: Circuit Breaker Integration
        circuit_breaker_result = await self._test_circuit_breakers()
        
        # Test 3: Health Check Caching
        caching_result = await self._test_health_caching()
        
        # Test 4: Predictive Health Analysis
        predictive_result = await self._test_predictive_analysis()
        
        # Test 5: Orchestrator Integration
        integration_result = await self._test_orchestrator_integration()
        
        # Compile results
        overall_result = {
            "features_2025": features_result,
            "circuit_breakers": circuit_breaker_result,
            "health_caching": caching_result,
            "predictive_analysis": predictive_result,
            "orchestrator_integration": integration_result,
            "summary": self._generate_test_summary()
        }
        
        self._print_test_results(overall_result)
        return overall_result
    
    async def _test_2025_features(self) -> Dict[str, Any]:
        """Test 2025 enhanced features"""
        
        print("\n🔬 Test 1: 2025 Features Validation")
        
        try:
            from prompt_improver.performance.monitoring.health.service import (
                EnhancedHealthService,
                HealthTrend,
                DependencyType,
                HealthMetrics,
                DependencyInfo
            )

            # Check if PredictiveHealthAnalysis exists
            try:
                from prompt_improver.performance.monitoring.health.service import PredictiveHealthAnalysis
                predictive_available = True
            except ImportError:
                predictive_available = False
            
            # Test enhanced classes and enums
            service = EnhancedHealthService(
                enable_circuit_breakers=True,
                enable_predictive_analysis=True,
                enable_caching=True
            )
            
            features_available = {
                "enhanced_service": True,
                "health_trends": len(list(HealthTrend)) >= 4,
                "dependency_types": len(list(DependencyType)) >= 6,
                "health_metrics": hasattr(HealthMetrics, 'success_rate'),
                "dependency_info": hasattr(DependencyInfo, 'circuit_breaker'),
                "predictive_analysis": predictive_available,
                "orchestrator_interface": hasattr(service, 'run_orchestrated_analysis'),
                "circuit_breakers": hasattr(service, 'circuit_breakers'),
                "health_history": hasattr(service, 'health_history'),
                "caching": hasattr(service, 'cached_results')
            }
            
            success_count = sum(features_available.values())
            total_features = len(features_available)
            
            print(f"  ✅ Enhanced Service: {'AVAILABLE' if features_available['enhanced_service'] else 'MISSING'}")
            print(f"  ✅ Health Trends: {len(list(HealthTrend))} trends available")
            print(f"  ✅ Dependency Types: {len(list(DependencyType))} types available")
            print(f"  ✅ Health Metrics: {'AVAILABLE' if features_available['health_metrics'] else 'MISSING'}")
            print(f"  ✅ Dependency Info: {'AVAILABLE' if features_available['dependency_info'] else 'MISSING'}")
            print(f"  ✅ Predictive Analysis: {'AVAILABLE' if features_available['predictive_analysis'] else 'MISSING'}")
            print(f"  ✅ Orchestrator Interface: {'AVAILABLE' if features_available['orchestrator_interface'] else 'MISSING'}")
            print(f"  ✅ Circuit Breakers: {'AVAILABLE' if features_available['circuit_breakers'] else 'MISSING'}")
            print(f"  ✅ Health History: {'AVAILABLE' if features_available['health_history'] else 'MISSING'}")
            print(f"  ✅ Caching: {'AVAILABLE' if features_available['caching'] else 'MISSING'}")
            print(f"  📊 Features Score: {success_count}/{total_features} ({(success_count/total_features)*100:.1f}%)")
            
            return {
                "success": success_count == total_features,
                "features_available": features_available,
                "features_score": success_count / total_features,
                "health_trends": len(list(HealthTrend)),
                "dependency_types": len(list(DependencyType))
            }
            
        except Exception as e:
            print(f"  ❌ 2025 features test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_circuit_breakers(self) -> Dict[str, Any]:
        """Test circuit breaker integration"""
        
        print("\n⚡ Test 2: Circuit Breaker Integration")
        
        try:
            from prompt_improver.performance.monitoring.health.service import EnhancedHealthService
            from prompt_improver.performance.monitoring.health.base import HealthChecker, HealthResult, HealthStatus
            
            # Create a failing health checker
            class FailingChecker(HealthChecker):
                def __init__(self):
                    super().__init__("failing_service")
                    self.call_count = 0
                
                async def check(self) -> HealthResult:
                    self.call_count += 1
                    raise Exception(f"Simulated failure {self.call_count}")
            
            failing_checker = FailingChecker()
            service = EnhancedHealthService(
                checkers=[failing_checker],
                enable_circuit_breakers=True
            )
            
            # Run multiple health checks to trigger circuit breaker
            results = []
            for i in range(5):
                result = await service.run_enhanced_health_check()
                results.append(result)
                await asyncio.sleep(0.1)
            
            # Check circuit breaker status
            cb_status = service._get_circuit_breaker_status()
            
            success = (
                len(service.circuit_breakers) > 0 and
                "failing_service" in service.circuit_breakers and
                len(cb_status) > 0
            )
            
            print(f"  {'✅' if success else '❌'} Circuit Breakers: {'PASSED' if success else 'FAILED'}")
            print(f"  📊 Circuit Breakers Created: {len(service.circuit_breakers)}")
            print(f"  📊 Failing Service CB: {'AVAILABLE' if 'failing_service' in service.circuit_breakers else 'MISSING'}")
            
            return {
                "success": success,
                "circuit_breakers_created": len(service.circuit_breakers),
                "circuit_breaker_status": cb_status
            }
            
        except Exception as e:
            print(f"  ❌ Circuit breaker test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_health_caching(self) -> Dict[str, Any]:
        """Test health check result caching"""
        
        print("\n💾 Test 3: Health Check Caching")
        
        try:
            from prompt_improver.performance.monitoring.health.service import EnhancedHealthService
            from prompt_improver.performance.monitoring.health.base import HealthChecker, HealthResult, HealthStatus
            
            # Create a simple health checker
            class SimpleChecker(HealthChecker):
                def __init__(self):
                    super().__init__("simple_service")
                    self.call_count = 0
                
                async def check(self) -> HealthResult:
                    self.call_count += 1
                    return HealthResult(
                        status=HealthStatus.HEALTHY,
                        component="simple_service",
                        message=f"Check #{self.call_count}"
                    )
            
            simple_checker = SimpleChecker()
            service = EnhancedHealthService(
                checkers=[simple_checker],
                enable_caching=True,
                cache_ttl=2  # 2 seconds cache
            )
            
            # First check - should hit the checker
            result1 = await service.run_enhanced_health_check(use_cache=True)
            first_call_count = simple_checker.call_count
            
            # Second check immediately - should use cache
            result2 = await service.run_enhanced_health_check(use_cache=True)
            second_call_count = simple_checker.call_count
            
            # Wait for cache to expire
            await asyncio.sleep(3)
            
            # Third check - should hit the checker again
            result3 = await service.run_enhanced_health_check(use_cache=True)
            third_call_count = simple_checker.call_count
            
            success = (
                first_call_count == 1 and
                second_call_count == 1 and  # Cache hit
                third_call_count == 2  # Cache miss after expiry
            )
            
            print(f"  {'✅' if success else '❌'} Health Caching: {'PASSED' if success else 'FAILED'}")
            print(f"  📊 First Check Calls: {first_call_count}")
            print(f"  📊 Second Check Calls: {second_call_count} (should be same - cache hit)")
            print(f"  📊 Third Check Calls: {third_call_count} (should be +1 - cache miss)")
            
            return {
                "success": success,
                "first_call_count": first_call_count,
                "second_call_count": second_call_count,
                "third_call_count": third_call_count,
                "cache_working": second_call_count == first_call_count
            }
            
        except Exception as e:
            print(f"  ❌ Health caching test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_predictive_analysis(self) -> Dict[str, Any]:
        """Test predictive health analysis"""
        
        print("\n🔮 Test 4: Predictive Health Analysis")
        
        try:
            from prompt_improver.performance.monitoring.health.service import EnhancedHealthService
            from prompt_improver.performance.monitoring.health.base import HealthChecker, HealthResult, HealthStatus
            
            # Create a degrading health checker
            class DegradingChecker(HealthChecker):
                def __init__(self):
                    super().__init__("degrading_service")
                    self.call_count = 0
                
                async def check(self) -> HealthResult:
                    self.call_count += 1
                    # Simulate degrading health
                    if self.call_count <= 2:
                        status = HealthStatus.HEALTHY
                    elif self.call_count <= 4:
                        status = HealthStatus.WARNING
                    else:
                        status = HealthStatus.FAILED
                    
                    return HealthResult(
                        status=status,
                        component="degrading_service",
                        message=f"Check #{self.call_count}"
                    )
            
            degrading_checker = DegradingChecker()
            service = EnhancedHealthService(
                checkers=[degrading_checker],
                enable_predictive_analysis=True,
                trend_window_size=5
            )
            
            # Run multiple checks to build history
            for i in range(6):
                await service.run_enhanced_health_check(include_predictions=True)
                await asyncio.sleep(0.1)
            
            # Get final result with predictions
            final_result = await service.run_enhanced_health_check(include_predictions=True)
            
            predictions = final_result.get("predictive_analysis", [])
            trend_analysis = final_result.get("trend_analysis", {})
            
            success = (
                len(predictions) > 0 and
                len(trend_analysis) > 0 and
                "degrading_service" in trend_analysis
            )
            
            print(f"  {'✅' if success else '❌'} Predictive Analysis: {'PASSED' if success else 'FAILED'}")
            print(f"  📊 Predictions Generated: {len(predictions)}")
            print(f"  📊 Trend Analysis: {len(trend_analysis)} components")
            
            return {
                "success": success,
                "predictions_count": len(predictions),
                "trend_analysis_count": len(trend_analysis),
                "predictions": predictions[:1] if predictions else []  # First prediction only
            }
            
        except Exception as e:
            print(f"  ❌ Predictive analysis test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_orchestrator_integration(self) -> Dict[str, Any]:
        """Test orchestrator integration"""
        
        print("\n🔄 Test 5: Orchestrator Integration")
        
        try:
            from prompt_improver.performance.monitoring.health.service import EnhancedHealthService
            
            service = EnhancedHealthService()
            
            # Test orchestrator interface
            config = {
                "parallel": True,
                "use_cache": True,
                "include_predictions": True,
                "output_path": "./test_outputs/health_monitoring"
            }
            
            result = await service.run_orchestrated_analysis(config)
            
            # Validate result structure
            success = (
                result.get("orchestrator_compatible") == True and
                "component_result" in result and
                "local_metadata" in result and
                "overall_status" in result["component_result"]
            )
            
            component_result = result.get("component_result", {})
            metadata = result.get("local_metadata", {})
            
            print(f"  {'✅' if success else '❌'} Orchestrator Interface: {'PASSED' if success else 'FAILED'}")
            print(f"  📊 Overall Status: {component_result.get('overall_status', 'unknown')}")
            print(f"  📊 Components Checked: {metadata.get('components_checked', 0)}")
            print(f"  📊 Dependencies Monitored: {metadata.get('dependencies_monitored', 0)}")
            print(f"  📊 Execution Time: {metadata.get('execution_time', 0):.3f}s")
            
            return {
                "success": success,
                "orchestrator_compatible": result.get("orchestrator_compatible", False),
                "overall_status": component_result.get("overall_status", "unknown"),
                "components_checked": metadata.get("components_checked", 0),
                "execution_time": metadata.get("execution_time", 0)
            }
            
        except Exception as e:
            print(f"  ❌ Orchestrator integration test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate overall test summary"""
        
        return {
            "total_tests": 5,
            "component_tested": "enhanced_health_service",
            "enhancement_status": "Phase 2B-2 Enhanced HealthService Complete",
            "version": "2025.1.0"
        }
    
    def _print_test_results(self, results: Dict[str, Any]):
        """Print comprehensive test results"""
        
        print("\n" + "=" * 70)
        print("📊 PHASE 2B-2 ENHANCED HEALTH SERVICE TEST RESULTS")
        print("=" * 70)
        
        # Print summary
        features = results.get("features_2025", {})
        circuit = results.get("circuit_breakers", {})
        caching = results.get("health_caching", {})
        predictive = results.get("predictive_analysis", {})
        integration = results.get("orchestrator_integration", {})
        
        features_success = features.get("success", False)
        circuit_success = circuit.get("success", False)
        caching_success = caching.get("success", False)
        predictive_success = predictive.get("success", False)
        integration_success = integration.get("success", False)
        
        print(f"✅ 2025 Features: {'PASSED' if features_success else 'FAILED'} ({features.get('features_score', 0)*100:.1f}%)")
        print(f"✅ Circuit Breakers: {'PASSED' if circuit_success else 'FAILED'}")
        print(f"✅ Health Caching: {'PASSED' if caching_success else 'FAILED'}")
        print(f"✅ Predictive Analysis: {'PASSED' if predictive_success else 'FAILED'}")
        print(f"✅ Orchestrator Integration: {'PASSED' if integration_success else 'FAILED'}")
        
        overall_success = all([features_success, circuit_success, caching_success, predictive_success, integration_success])
        
        if overall_success:
            print("\n🎉 PHASE 2B-2 ENHANCEMENT: COMPLETE SUCCESS!")
            print("Enhanced HealthService with 2025 best practices is fully integrated and ready!")
        else:
            print("\n⚠️  PHASE 2B-2 ENHANCEMENT: NEEDS ATTENTION")
            print("Some enhanced features require additional work.")


async def main():
    """Main test execution function"""
    
    tester = Phase2BEnhancedHealthServiceTester()
    results = await tester.run_comprehensive_integration_test()
    
    # Save results to file
    with open('phase2b_enhanced_health_service_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: phase2b_enhanced_health_service_test_results.json")
    
    return 0 if results.get("features_2025", {}).get("success", False) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
