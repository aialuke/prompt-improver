#!/usr/bin/env python3
"""
Phase 2C Enhanced Components Integration Test

Tests the enhanced Phase 2C components with 2025 best practices:
1. Enhanced RealTimeAnalyticsService - Event-driven architecture
2. Enhanced CanaryTestingService - Progressive delivery

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

class Phase2CEnhancedComponentsTester:
    """Test enhanced Phase 2C components integration with orchestrator"""
    
    def __init__(self):
        self.test_results = {}
        self.component_name = "phase2c_enhanced_components"
    
    async def run_comprehensive_integration_test(self) -> Dict[str, Any]:
        """Run comprehensive integration test for enhanced Phase 2C components"""
        
        print("🚀 Phase 2C Enhanced Components Integration Test")
        print("=" * 70)
        
        # Test 1: Enhanced RealTimeAnalyticsService
        analytics_result = await self._test_enhanced_analytics_service()
        
        # Test 2: Enhanced CanaryTestingService
        canary_result = await self._test_enhanced_canary_service()
        
        # Test 3: Cross-Component Integration
        integration_result = await self._test_cross_component_integration()
        
        # Test 4: Orchestrator Compliance
        compliance_result = await self._test_orchestrator_compliance()
        
        # Compile results
        overall_result = {
            "enhanced_analytics": analytics_result,
            "enhanced_canary": canary_result,
            "cross_integration": integration_result,
            "orchestrator_compliance": compliance_result,
            "summary": self._generate_test_summary()
        }
        
        self._print_test_results(overall_result)
        return overall_result
    
    async def _test_enhanced_analytics_service(self) -> Dict[str, Any]:
        """Test enhanced RealTimeAnalyticsService"""
        
        print("\n📊 Test 1: Enhanced RealTimeAnalyticsService")
        print("-" * 50)
        
        try:
            from prompt_improver.performance.analytics.real_time_analytics import (
                EnhancedRealTimeAnalyticsService,
                AnalyticsEvent,
                EventType,
                StreamProcessingMode,
                AnomalyDetection
            )
            
            # Create mock database session
            class MockSession:
                pass
            
            # Initialize service
            service = EnhancedRealTimeAnalyticsService(
                db_session=MockSession(),
                enable_stream_processing=True,
                enable_anomaly_detection=True,
                processing_mode=StreamProcessingMode.NEAR_REAL_TIME
            )
            
            # Test orchestrator interface
            config = {
                "experiment_ids": ["test_exp_001"],
                "enable_streaming": True,
                "enable_anomaly_detection": True,
                "simulate_data": True,
                "output_path": "./test_outputs/analytics"
            }
            
            result = await service.run_orchestrated_analysis(config)
            
            # Validate result
            success = (
                result.get("orchestrator_compatible") == True and
                "component_result" in result and
                "analytics_summary" in result["component_result"] and
                "capabilities" in result["component_result"]
            )
            
            capabilities = result.get("component_result", {}).get("capabilities", {})
            analytics_summary = result.get("component_result", {}).get("analytics_summary", {})
            
            print(f"  {'✅' if success else '❌'} Enhanced Analytics: {'PASSED' if success else 'FAILED'}")
            print(f"  📊 Stream Processing: {'ENABLED' if capabilities.get('stream_processing', False) else 'DISABLED'}")
            print(f"  📊 ML Anomaly Detection: {'AVAILABLE' if capabilities.get('ml_anomaly_detection', False) else 'UNAVAILABLE'}")
            print(f"  📊 Events Processed: {analytics_summary.get('events_processed', 0)}")
            print(f"  📊 Active Windows: {analytics_summary.get('active_windows', 0)}")
            
            return {
                "success": success,
                "orchestrator_compatible": result.get("orchestrator_compatible", False),
                "stream_processing_enabled": capabilities.get("stream_processing", False),
                "ml_anomaly_detection": capabilities.get("ml_anomaly_detection", False),
                "events_processed": analytics_summary.get("events_processed", 0),
                "execution_time": result.get("local_metadata", {}).get("execution_time", 0)
            }
            
        except Exception as e:
            print(f"  ❌ Enhanced analytics test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_enhanced_canary_service(self) -> Dict[str, Any]:
        """Test enhanced CanaryTestingService"""
        
        print("\n🚢 Test 2: Enhanced CanaryTestingService")
        print("-" * 50)
        
        try:
            from prompt_improver.performance.testing.canary_testing import (
                EnhancedCanaryTestingService,
                DeploymentStrategy,
                SLITarget,
                CanaryPhase
            )
            
            # Initialize service
            service = EnhancedCanaryTestingService(
                enable_service_mesh=True,
                enable_gitops=True,
                enable_sli_monitoring=True
            )
            
            # Test orchestrator interface
            config = {
                "deployment_name": "test_deployment_001",
                "strategy": "canary",
                "initial_percentage": 10.0,
                "enable_sli_monitoring": True,
                "simulate_deployment": True,
                "output_path": "./test_outputs/canary"
            }
            
            result = await service.run_orchestrated_analysis(config)
            
            # Validate result
            success = (
                result.get("orchestrator_compatible") == True and
                "component_result" in result and
                "canary_summary" in result["component_result"] and
                "capabilities" in result["component_result"]
            )
            
            capabilities = result.get("component_result", {}).get("capabilities", {})
            canary_summary = result.get("component_result", {}).get("canary_summary", {})
            
            print(f"  {'✅' if success else '❌'} Enhanced Canary: {'PASSED' if success else 'FAILED'}")
            print(f"  📊 Progressive Delivery: {'ENABLED' if capabilities.get('progressive_delivery', False) else 'DISABLED'}")
            print(f"  📊 Service Mesh Integration: {'ENABLED' if capabilities.get('service_mesh_integration', False) else 'DISABLED'}")
            print(f"  📊 SLI/SLO Monitoring: {'ENABLED' if capabilities.get('sli_slo_monitoring', False) else 'DISABLED'}")
            print(f"  📊 Active Deployments: {canary_summary.get('active_deployments', 0)}")
            
            return {
                "success": success,
                "orchestrator_compatible": result.get("orchestrator_compatible", False),
                "progressive_delivery": capabilities.get("progressive_delivery", False),
                "service_mesh_integration": capabilities.get("service_mesh_integration", False),
                "sli_slo_monitoring": capabilities.get("sli_slo_monitoring", False),
                "active_deployments": canary_summary.get("active_deployments", 0),
                "execution_time": result.get("local_metadata", {}).get("execution_time", 0)
            }
            
        except Exception as e:
            print(f"  ❌ Enhanced canary test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_cross_component_integration(self) -> Dict[str, Any]:
        """Test cross-component integration"""
        
        print("\n🔗 Test 3: Cross-Component Integration")
        print("-" * 50)
        
        try:
            # Test that both components can work together
            from prompt_improver.performance.analytics.real_time_analytics import EnhancedRealTimeAnalyticsService
            from prompt_improver.performance.testing.canary_testing import EnhancedCanaryTestingService
            
            # Mock session
            class MockSession:
                pass
            
            # Initialize both services
            analytics_service = EnhancedRealTimeAnalyticsService(
                db_session=MockSession(),
                enable_stream_processing=False  # Disable for integration test
            )
            
            canary_service = EnhancedCanaryTestingService(
                enable_service_mesh=False  # Disable for integration test
            )
            
            # Test that they can both run orchestrated analysis
            analytics_config = {"simulate_data": True}
            canary_config = {"simulate_deployment": True}
            
            analytics_result = await analytics_service.run_orchestrated_analysis(analytics_config)
            canary_result = await canary_service.run_orchestrated_analysis(canary_config)
            
            success = (
                analytics_result.get("orchestrator_compatible", False) and
                canary_result.get("orchestrator_compatible", False)
            )
            
            print(f"  {'✅' if success else '❌'} Cross-Component Integration: {'PASSED' if success else 'FAILED'}")
            print(f"  📊 Analytics Compatible: {'YES' if analytics_result.get('orchestrator_compatible', False) else 'NO'}")
            print(f"  📊 Canary Compatible: {'YES' if canary_result.get('orchestrator_compatible', False) else 'NO'}")
            
            return {
                "success": success,
                "analytics_compatible": analytics_result.get("orchestrator_compatible", False),
                "canary_compatible": canary_result.get("orchestrator_compatible", False),
                "both_services_working": success
            }
            
        except Exception as e:
            print(f"  ❌ Cross-component integration test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_orchestrator_compliance(self) -> Dict[str, Any]:
        """Test orchestrator compliance"""
        
        print("\n🏛️ Test 4: Orchestrator Compliance")
        print("-" * 50)
        
        compliance_checks = []
        
        # Check 1: Both components implement run_orchestrated_analysis
        print("Checking orchestrated analysis interface...")
        interface_compliance = True
        
        try:
            from prompt_improver.performance.analytics.real_time_analytics import EnhancedRealTimeAnalyticsService
            from prompt_improver.performance.testing.canary_testing import EnhancedCanaryTestingService
            
            # Mock session
            class MockSession:
                pass
            
            components = [
                EnhancedRealTimeAnalyticsService(db_session=MockSession()),
                EnhancedCanaryTestingService()
            ]
            
            for component in components:
                if not hasattr(component, 'run_orchestrated_analysis'):
                    interface_compliance = False
                    print(f"  ⚠️  {component.__class__.__name__}: Missing run_orchestrated_analysis method")
            
        except Exception as e:
            interface_compliance = False
            print(f"  ❌ Interface check failed: {e}")
        
        compliance_checks.append(("interface_compliance", interface_compliance))
        print(f"  {'✅' if interface_compliance else '❌'} Interface Compliance: {'COMPLIANT' if interface_compliance else 'NON_COMPLIANT'}")
        
        # Check 2: 2025 features implementation
        print("Checking 2025 features implementation...")
        features_compliance = True
        
        try:
            # Check for 2025 feature enums and classes
            from prompt_improver.performance.analytics.real_time_analytics import EventType, StreamProcessingMode
            from prompt_improver.performance.testing.canary_testing import DeploymentStrategy, RollbackTrigger
            
            features_available = {
                "event_types": len(list(EventType)) >= 5,
                "stream_processing_modes": len(list(StreamProcessingMode)) >= 3,
                "deployment_strategies": len(list(DeploymentStrategy)) >= 4,
                "rollback_triggers": len(list(RollbackTrigger)) >= 5
            }
            
            features_compliance = all(features_available.values())
            
        except Exception as e:
            features_compliance = False
            print(f"  ❌ Features check failed: {e}")
        
        compliance_checks.append(("features_compliance", features_compliance))
        print(f"  {'✅' if features_compliance else '❌'} 2025 Features: {'COMPLIANT' if features_compliance else 'NON_COMPLIANT'}")
        
        passed_checks = sum(1 for _, check in compliance_checks if check)
        total_checks = len(compliance_checks)
        
        print(f"\n📊 Orchestrator Compliance: {passed_checks}/{total_checks} checks passed")
        
        return {
            "success": passed_checks == total_checks,
            "checks_performed": total_checks,
            "checks_passed": passed_checks,
            "individual_checks": dict(compliance_checks)
        }
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate overall test summary"""
        
        return {
            "total_tests": 4,
            "components_tested": 2,
            "enhancement_status": "Phase 2C High Priority Components Enhanced",
            "version": "2025.1.0"
        }
    
    def _print_test_results(self, results: Dict[str, Any]):
        """Print comprehensive test results"""
        
        print("\n" + "=" * 70)
        print("📊 PHASE 2C ENHANCED COMPONENTS TEST RESULTS")
        print("=" * 70)
        
        # Print summary
        analytics = results.get("enhanced_analytics", {})
        canary = results.get("enhanced_canary", {})
        integration = results.get("cross_integration", {})
        compliance = results.get("orchestrator_compliance", {})
        
        analytics_success = analytics.get("success", False)
        canary_success = canary.get("success", False)
        integration_success = integration.get("success", False)
        compliance_success = compliance.get("success", False)
        
        print(f"✅ Enhanced Analytics: {'PASSED' if analytics_success else 'FAILED'}")
        print(f"✅ Enhanced Canary: {'PASSED' if canary_success else 'FAILED'}")
        print(f"✅ Cross-Component Integration: {'PASSED' if integration_success else 'FAILED'}")
        print(f"✅ Orchestrator Compliance: {'PASSED' if compliance_success else 'FAILED'}")
        
        overall_success = all([analytics_success, canary_success, integration_success, compliance_success])
        
        if overall_success:
            print("\n🎉 PHASE 2C HIGH PRIORITY ENHANCEMENTS: COMPLETE SUCCESS!")
            print("Enhanced components with 2025 best practices are fully integrated!")
        else:
            print("\n⚠️  PHASE 2C HIGH PRIORITY ENHANCEMENTS: NEEDS ATTENTION")
            print("Some enhanced features require additional work.")


async def main():
    """Main test execution function"""
    
    tester = Phase2CEnhancedComponentsTester()
    results = await tester.run_comprehensive_integration_test()
    
    # Save results to file
    with open('phase2c_enhanced_components_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: phase2c_enhanced_components_test_results.json")
    
    return 0 if all(
        results.get(key, {}).get("success", False) 
        for key in ["enhanced_analytics", "enhanced_canary", "cross_integration", "orchestrator_compliance"]
    ) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
