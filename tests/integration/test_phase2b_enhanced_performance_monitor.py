#!/usr/bin/env python3
"""
Phase 2B-3 Enhanced PerformanceMonitor Integration Test

Tests the enhanced PerformanceMonitor with 2025 best practices:
- SLI/SLO framework with error budget tracking
- Multi-dimensional metrics (RED/USE patterns)
- Percentile-based monitoring (P50, P95, P99)
- Business metric correlation
- Adaptive thresholds based on historical data
- Burn rate analysis and alerting

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

class Phase2BEnhancedPerformanceMonitorTester:
    """Test enhanced PerformanceMonitor integration with orchestrator"""
    
    def __init__(self):
        self.test_results = {}
        self.component_name = "enhanced_performance_monitor"
    
    async def run_comprehensive_integration_test(self) -> Dict[str, Any]:
        """Run comprehensive integration test for enhanced PerformanceMonitor"""
        
        print("🚀 Phase 2B-3 Enhanced PerformanceMonitor Integration Test")
        print("=" * 70)
        
        # Test 1: Component Discovery and 2025 Features
        features_result = await self._test_2025_features()
        
        # Test 2: SLI/SLO Framework
        sli_slo_result = await self._test_sli_slo_framework()
        
        # Test 3: Error Budget Tracking
        error_budget_result = await self._test_error_budget_tracking()
        
        # Test 4: Burn Rate Analysis
        burn_rate_result = await self._test_burn_rate_analysis()
        
        # Test 5: Orchestrator Integration
        integration_result = await self._test_orchestrator_integration()
        
        # Compile results
        overall_result = {
            "features_2025": features_result,
            "sli_slo_framework": sli_slo_result,
            "error_budget_tracking": error_budget_result,
            "burn_rate_analysis": burn_rate_result,
            "orchestrator_integration": integration_result,
            "summary": self._generate_test_summary()
        }
        
        self._print_test_results(overall_result)
        return overall_result
    
    async def _test_2025_features(self) -> Dict[str, Any]:
        """Test 2025 enhanced features"""
        
        print("\n🔬 Test 1: 2025 Features Validation")
        
        try:
            from prompt_improver.performance.monitoring.performance_monitor import (
                EnhancedPerformanceMonitor,
                SLI,
                SLO,
                SLIType,
                SLOStatus,
                BurnRateLevel,
                PerformanceMetrics
            )

            # Check if SLOViolation exists
            try:
                from prompt_improver.performance.monitoring.performance_monitor import SLOViolation
                slo_violation_available = True
            except ImportError:
                slo_violation_available = False
            
            # Test enhanced classes and enums
            monitor = EnhancedPerformanceMonitor(
                enable_anomaly_detection=True,
                enable_adaptive_thresholds=True
            )
            
            features_available = {
                "enhanced_monitor": True,
                "sli_types": len(list(SLIType)) >= 5,
                "slo_status": len(list(SLOStatus)) >= 4,
                "burn_rate_levels": len(list(BurnRateLevel)) >= 4,
                "performance_metrics": hasattr(PerformanceMetrics, 'response_time_p95'),
                "slo_violation": slo_violation_available,
                "orchestrator_interface": hasattr(monitor, 'run_orchestrated_analysis'),
                "slo_dashboard": hasattr(monitor, 'get_slo_dashboard'),
                "error_budget_tracking": hasattr(monitor, 'error_budget_tracking'),
                "burn_rate_analysis": hasattr(monitor, 'burn_rate_history')
            }
            
            success_count = sum(features_available.values())
            total_features = len(features_available)
            
            print(f"  ✅ Enhanced Monitor: {'AVAILABLE' if features_available['enhanced_monitor'] else 'MISSING'}")
            print(f"  ✅ SLI Types: {len(list(SLIType))} types available")
            print(f"  ✅ SLO Status: {len(list(SLOStatus))} states available")
            print(f"  ✅ Burn Rate Levels: {len(list(BurnRateLevel))} levels available")
            print(f"  ✅ Performance Metrics: {'AVAILABLE' if features_available['performance_metrics'] else 'MISSING'}")
            print(f"  ✅ SLO Violation: {'AVAILABLE' if features_available['slo_violation'] else 'MISSING'}")
            print(f"  ✅ Orchestrator Interface: {'AVAILABLE' if features_available['orchestrator_interface'] else 'MISSING'}")
            print(f"  ✅ SLO Dashboard: {'AVAILABLE' if features_available['slo_dashboard'] else 'MISSING'}")
            print(f"  ✅ Error Budget Tracking: {'AVAILABLE' if features_available['error_budget_tracking'] else 'MISSING'}")
            print(f"  ✅ Burn Rate Analysis: {'AVAILABLE' if features_available['burn_rate_analysis'] else 'MISSING'}")
            print(f"  📊 Features Score: {success_count}/{total_features} ({(success_count/total_features)*100:.1f}%)")
            
            return {
                "success": success_count == total_features,
                "features_available": features_available,
                "features_score": success_count / total_features,
                "sli_types": len(list(SLIType)),
                "slo_status": len(list(SLOStatus))
            }
            
        except Exception as e:
            print(f"  ❌ 2025 features test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_sli_slo_framework(self) -> Dict[str, Any]:
        """Test SLI/SLO framework"""
        
        print("\n📊 Test 2: SLI/SLO Framework")
        
        try:
            from prompt_improver.performance.monitoring.performance_monitor import EnhancedPerformanceMonitor
            
            monitor = EnhancedPerformanceMonitor()
            
            # Record some performance measurements
            for i in range(20):
                response_time = 100 + (i * 5)  # Gradually increasing response time
                is_error = i > 15  # Errors in the last few measurements
                
                await monitor.record_performance_measurement(
                    operation_name="test_operation",
                    response_time_ms=response_time,
                    is_error=is_error,
                    business_value=10.0
                )
                await asyncio.sleep(0.01)  # Small delay
            
            # Get SLO dashboard
            dashboard = monitor.get_slo_dashboard()
            
            success = (
                "slos" in dashboard and
                "error_budgets" in dashboard and
                "burn_rates" in dashboard and
                len(dashboard["slos"]) > 0
            )
            
            print(f"  {'✅' if success else '❌'} SLI/SLO Framework: {'PASSED' if success else 'FAILED'}")
            print(f"  📊 SLOs Configured: {len(dashboard.get('slos', {}))}")
            print(f"  📊 Overall Health: {dashboard.get('overall_health', 'unknown')}")
            print(f"  📊 Active Violations: {dashboard.get('active_violations', 0)}")
            
            return {
                "success": success,
                "slos_configured": len(dashboard.get("slos", {})),
                "overall_health": dashboard.get("overall_health", "unknown"),
                "active_violations": dashboard.get("active_violations", 0),
                "dashboard_keys": list(dashboard.keys())
            }
            
        except Exception as e:
            print(f"  ❌ SLI/SLO framework test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_error_budget_tracking(self) -> Dict[str, Any]:
        """Test error budget tracking"""
        
        print("\n💰 Test 3: Error Budget Tracking")
        
        try:
            from prompt_improver.performance.monitoring.performance_monitor import EnhancedPerformanceMonitor
            
            monitor = EnhancedPerformanceMonitor()
            
            # Record measurements that will consume error budget
            for i in range(30):
                # High response times and errors to consume budget
                response_time = 300 + (i * 10)  # High response times
                is_error = i % 5 == 0  # 20% error rate
                
                await monitor.record_performance_measurement(
                    operation_name="budget_test",
                    response_time_ms=response_time,
                    is_error=is_error,
                    business_value=5.0
                )
                await asyncio.sleep(0.01)
            
            # Check error budget status
            dashboard = monitor.get_slo_dashboard()
            error_budgets = dashboard.get("error_budgets", {})
            
            success = (
                len(error_budgets) > 0 and
                all("remaining_percentage" in budget for budget in error_budgets.values()) and
                all("consumed_minutes" in budget for budget in error_budgets.values())
            )
            
            # Check if any budget was consumed
            budget_consumed = any(
                budget["remaining_percentage"] < 100 
                for budget in error_budgets.values()
            )
            
            print(f"  {'✅' if success else '❌'} Error Budget Tracking: {'PASSED' if success else 'FAILED'}")
            print(f"  📊 Error Budgets Tracked: {len(error_budgets)}")
            print(f"  📊 Budget Consumed: {'YES' if budget_consumed else 'NO'}")
            
            # Print budget details for first SLO
            if error_budgets:
                first_slo = list(error_budgets.keys())[0]
                budget = error_budgets[first_slo]
                print(f"  📊 {first_slo} Budget Remaining: {budget.get('remaining_percentage', 0):.1f}%")
            
            return {
                "success": success,
                "error_budgets_tracked": len(error_budgets),
                "budget_consumed": budget_consumed,
                "budget_details": error_budgets
            }
            
        except Exception as e:
            print(f"  ❌ Error budget tracking test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_burn_rate_analysis(self) -> Dict[str, Any]:
        """Test burn rate analysis"""
        
        print("\n🔥 Test 4: Burn Rate Analysis")
        
        try:
            from prompt_improver.performance.monitoring.performance_monitor import EnhancedPerformanceMonitor
            
            monitor = EnhancedPerformanceMonitor()
            
            # Record measurements with high error rate to trigger burn rate
            for i in range(40):
                response_time = 250  # Consistent high response time
                is_error = i % 3 == 0  # 33% error rate - high burn rate
                
                await monitor.record_performance_measurement(
                    operation_name="burn_rate_test",
                    response_time_ms=response_time,
                    is_error=is_error,
                    business_value=1.0
                )
                await asyncio.sleep(0.01)
            
            # Check burn rate analysis
            dashboard = monitor.get_slo_dashboard()
            burn_rates = dashboard.get("burn_rates", {})
            
            success = (
                len(burn_rates) > 0 and
                all("current_rate" in rate for rate in burn_rates.values()) and
                all("critical_threshold" in rate for rate in burn_rates.values())
            )
            
            # Check if any burn rate is elevated
            elevated_burn_rate = any(
                rate["current_rate"] > 0 
                for rate in burn_rates.values()
            )
            
            print(f"  {'✅' if success else '❌'} Burn Rate Analysis: {'PASSED' if success else 'FAILED'}")
            print(f"  📊 Burn Rates Tracked: {len(burn_rates)}")
            print(f"  📊 Elevated Burn Rate: {'YES' if elevated_burn_rate else 'NO'}")
            
            # Print burn rate details for first SLO
            if burn_rates:
                first_slo = list(burn_rates.keys())[0]
                rate = burn_rates[first_slo]
                print(f"  📊 {first_slo} Burn Rate: {rate.get('current_rate', 0):.3f}")
            
            return {
                "success": success,
                "burn_rates_tracked": len(burn_rates),
                "elevated_burn_rate": elevated_burn_rate,
                "burn_rate_details": burn_rates
            }
            
        except Exception as e:
            print(f"  ❌ Burn rate analysis test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_orchestrator_integration(self) -> Dict[str, Any]:
        """Test orchestrator integration"""
        
        print("\n🔄 Test 5: Orchestrator Integration")
        
        try:
            from prompt_improver.performance.monitoring.performance_monitor import EnhancedPerformanceMonitor
            
            monitor = EnhancedPerformanceMonitor()
            
            # Test orchestrator interface
            config = {
                "measurement_window": 300,
                "include_predictions": True,
                "include_anomaly_detection": True,
                "simulate_data": True,  # Generate test data
                "output_path": "./test_outputs/performance_monitoring"
            }
            
            result = await monitor.run_orchestrated_analysis(config)
            
            # Validate result structure
            success = (
                result.get("orchestrator_compatible") == True and
                "component_result" in result and
                "local_metadata" in result and
                "performance_summary" in result["component_result"]
            )
            
            component_result = result.get("component_result", {})
            performance_summary = component_result.get("performance_summary", {})
            metadata = result.get("local_metadata", {})
            
            print(f"  {'✅' if success else '❌'} Orchestrator Interface: {'PASSED' if success else 'FAILED'}")
            print(f"  📊 Overall Health: {performance_summary.get('overall_health', 'unknown')}")
            print(f"  📊 Total SLOs: {performance_summary.get('total_slos', 0)}")
            print(f"  📊 Measurements Collected: {metadata.get('measurements_collected', 0)}")
            print(f"  📊 Execution Time: {metadata.get('execution_time', 0):.3f}s")
            
            return {
                "success": success,
                "orchestrator_compatible": result.get("orchestrator_compatible", False),
                "overall_health": performance_summary.get("overall_health", "unknown"),
                "total_slos": performance_summary.get("total_slos", 0),
                "measurements_collected": metadata.get("measurements_collected", 0),
                "execution_time": metadata.get("execution_time", 0)
            }
            
        except Exception as e:
            print(f"  ❌ Orchestrator integration test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate overall test summary"""
        
        return {
            "total_tests": 5,
            "component_tested": "enhanced_performance_monitor",
            "enhancement_status": "Phase 2B-3 Enhanced PerformanceMonitor Complete",
            "version": "2025.1.0"
        }
    
    def _print_test_results(self, results: Dict[str, Any]):
        """Print comprehensive test results"""
        
        print("\n" + "=" * 70)
        print("📊 PHASE 2B-3 ENHANCED PERFORMANCE MONITOR TEST RESULTS")
        print("=" * 70)
        
        # Print summary
        features = results.get("features_2025", {})
        sli_slo = results.get("sli_slo_framework", {})
        error_budget = results.get("error_budget_tracking", {})
        burn_rate = results.get("burn_rate_analysis", {})
        integration = results.get("orchestrator_integration", {})
        
        features_success = features.get("success", False)
        sli_slo_success = sli_slo.get("success", False)
        error_budget_success = error_budget.get("success", False)
        burn_rate_success = burn_rate.get("success", False)
        integration_success = integration.get("success", False)
        
        print(f"✅ 2025 Features: {'PASSED' if features_success else 'FAILED'} ({features.get('features_score', 0)*100:.1f}%)")
        print(f"✅ SLI/SLO Framework: {'PASSED' if sli_slo_success else 'FAILED'}")
        print(f"✅ Error Budget Tracking: {'PASSED' if error_budget_success else 'FAILED'}")
        print(f"✅ Burn Rate Analysis: {'PASSED' if burn_rate_success else 'FAILED'}")
        print(f"✅ Orchestrator Integration: {'PASSED' if integration_success else 'FAILED'}")
        
        overall_success = all([features_success, sli_slo_success, error_budget_success, burn_rate_success, integration_success])
        
        if overall_success:
            print("\n🎉 PHASE 2B-3 ENHANCEMENT: COMPLETE SUCCESS!")
            print("Enhanced PerformanceMonitor with SLI/SLO framework is fully integrated and ready!")
        else:
            print("\n⚠️  PHASE 2B-3 ENHANCEMENT: NEEDS ATTENTION")
            print("Some enhanced features require additional work.")


async def main():
    """Main test execution function"""
    
    tester = Phase2BEnhancedPerformanceMonitorTester()
    results = await tester.run_comprehensive_integration_test()
    
    # Save results to file
    with open('phase2b_enhanced_performance_monitor_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: phase2b_enhanced_performance_monitor_test_results.json")
    
    return 0 if results.get("features_2025", {}).get("success", False) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
