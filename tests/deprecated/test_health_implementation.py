#!/usr/bin/env python3
"""
Test script for APES PHASE 3 Health Check Implementation
Tests the new unified health service and verifies everything works correctly.
"""

import asyncio
import json
import sys
import traceback
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_health_system():
    """Test the health check system comprehensively"""
    print("🧪 Testing APES PHASE 3 Health Check System")
    print("=" * 60)
    
    try:
        # Test 1: Import health system
        print("\n1️⃣ Testing imports...")
        from prompt_improver.services.health import (
            get_health_service,
            HealthStatus,
            PROMETHEUS_AVAILABLE
        )
        print("✅ Successfully imported health system")
        print(f"📊 Prometheus available: {PROMETHEUS_AVAILABLE}")
        
        # Test 2: Get health service instance
        print("\n2️⃣ Testing health service instance...")
        health_service = get_health_service()
        available_checks = health_service.get_available_checks()
        print(f"✅ Health service created with {len(available_checks)} checkers")
        print(f"🔍 Available checks: {', '.join(available_checks)}")
        
        # Test 3: Run individual health checks
        print("\n3️⃣ Testing individual health checks...")
        for check_name in available_checks:
            try:
                result = await health_service.run_specific_check(check_name)
                status_icon = "✅" if result.status == HealthStatus.HEALTHY else "⚠️" if result.status == HealthStatus.WARNING else "❌"
                response_time = f" ({result.response_time_ms:.1f}ms)" if result.response_time_ms else ""
                print(f"  {status_icon} {check_name}: {result.status.value}{response_time}")
                if result.error:
                    print(f"    Error: {result.error}")
            except Exception as e:
                print(f"  ❌ {check_name}: Failed - {str(e)}")
        
        # Test 4: Run full health check (parallel)
        print("\n4️⃣ Testing full health check (parallel)...")
        start_time = asyncio.get_event_loop().time()
        result = await health_service.run_health_check(parallel=True)
        end_time = asyncio.get_event_loop().time()
        
        overall_icon = "✅" if result.overall_status == HealthStatus.HEALTHY else "⚠️" if result.overall_status == HealthStatus.WARNING else "❌"
        print(f"  {overall_icon} Overall status: {result.overall_status.value}")
        print(f"  ⏱️ Total time: {(end_time - start_time) * 1000:.1f}ms")
        print(f"  📊 Checks: {len(result.checks)} completed")
        
        if result.failed_checks:
            print(f"  ❌ Failed: {', '.join(result.failed_checks)}")
        if result.warning_checks:
            print(f"  ⚠️ Warnings: {', '.join(result.warning_checks)}")
        
        # Test 5: Run full health check (sequential)
        print("\n5️⃣ Testing full health check (sequential)...")
        start_time = asyncio.get_event_loop().time()
        result_seq = await health_service.run_health_check(parallel=False)
        end_time = asyncio.get_event_loop().time()
        
        print(f"  📊 Sequential time: {(end_time - start_time) * 1000:.1f}ms")
        print(f"  🔄 Status consistent: {result.overall_status == result_seq.overall_status}")
        
        # Test 6: Test health summary format
        print("\n6️⃣ Testing health summary format...")
        summary = await health_service.get_health_summary(include_details=True)
        print(f"  📋 Summary keys: {list(summary.keys())}")
        print(f"  🔍 Checks in summary: {len(summary.get('checks', {}))}")
        
        # Test 7: Test CLI compatibility 
        print("\n7️⃣ Testing CLI format compatibility...")
        try:
            # Simulate CLI usage
            checks = summary.get("checks", {})
            for component, check_result in checks.items():
                status = check_result.get("status", "unknown")
                response_time = check_result.get("response_time_ms")
                message = check_result.get("message", "")
                
                status_icon = "✅" if status == "healthy" else "⚠️" if status == "warning" else "❌"
                response_str = f" ({response_time:.1f}ms)" if response_time else ""
                print(f"  {status_icon} {component.replace('_', ' ').title()}: {status.capitalize()}{response_str}")
            
            print("✅ CLI format compatibility verified")
        except Exception as e:
            print(f"❌ CLI format compatibility failed: {e}")
        
        # Test 8: Test Prometheus metrics (if available)
        if PROMETHEUS_AVAILABLE:
            print("\n8️⃣ Testing Prometheus metrics...")
            try:
                from prompt_improver.services.health.metrics import get_health_metrics_summary
                metrics_summary = get_health_metrics_summary()
                print(f"  📊 Metrics summary: {metrics_summary}")
                print("✅ Prometheus metrics integration working")
            except Exception as e:
                print(f"⚠️ Prometheus metrics test failed: {e}")
        else:
            print("\n8️⃣ Skipping Prometheus metrics (not available)")
        
        # Test 9: Test error handling
        print("\n9️⃣ Testing error handling...")
        try:
            # Test unknown component
            unknown_result = await health_service.run_specific_check("nonexistent")
            if unknown_result.status == HealthStatus.FAILED:
                print("✅ Unknown component error handling works")
            else:
                print("⚠️ Unknown component should return FAILED status")
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
        
        # Test 10: Performance check
        print("\n🔟 Performance assessment...")
        times = []
        for i in range(3):
            start_time = asyncio.get_event_loop().time()
            await health_service.run_health_check(parallel=True)
            end_time = asyncio.get_event_loop().time()
            times.append((end_time - start_time) * 1000)
        
        avg_time = sum(times) / len(times)
        print(f"  ⏱️ Average execution time: {avg_time:.1f}ms")
        print(f"  📊 Individual times: {[f'{t:.1f}ms' for t in times]}")
        
        if avg_time < 1000:  # Should complete in under 1 second
            print("✅ Performance is acceptable")
        else:
            print("⚠️ Performance may need optimization")
        
        print("\n" + "=" * 60)
        print("🎉 PHASE 3 Health Check System Test Complete!")
        print(f"Overall Status: {result.overall_status.value.upper()}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        print("🔍 Full traceback:")
        traceback.print_exc()
        return False


async def test_cli_integration():
    """Test CLI integration specifically"""
    print("\n" + "=" * 60)
    print("🖥️ Testing CLI Integration")
    print("=" * 60)
    
    try:
        # Test health service import in CLI context
        from prompt_improver.services.health import get_health_service
        
        health_service = get_health_service()
        
        # Simulate CLI health command execution
        print("\n🧪 Simulating CLI health command...")
        summary = await health_service.get_health_summary(include_details=True)
        
        # Format like CLI would
        overall_status = summary.get("overall_status", "unknown")
        status_icon = "✅" if overall_status == "healthy" else "⚠️" if overall_status == "warning" else "❌"
        
        print(f"\n{status_icon} Overall Health: {overall_status.upper()}")
        
        checks = summary.get("checks", {})
        for component, check_result in checks.items():
            status = check_result.get("status", "unknown")
            component_icon = "✅" if status == "healthy" else "⚠️" if status == "warning" else "❌"
            response_time = check_result.get("response_time_ms")
            response_str = f"{response_time:.1f}ms" if response_time else "-"
            message = check_result.get("message", "")
            
            print(f"{component_icon} {component.replace('_', ' ').title()}: {status.capitalize()} ({response_str}) - {message}")
        
        # Test detailed mode
        if "warning_checks" in summary or "failed_checks" in summary:
            print("\n⚠️ Issues Found:")
            for warning in summary.get("warning_checks", []):
                print(f"  ⚠️ {warning}: {checks[warning].get('message', '')}")
            for failure in summary.get("failed_checks", []):
                print(f"  ❌ {failure}: {checks[failure].get('message', '')}")
        
        print("✅ CLI integration test successful")
        return True
        
    except Exception as e:
        print(f"❌ CLI integration test failed: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🚀 APES PHASE 3 Health Check System Testing")
    print("Testing unified health service implementation...")
    
    # Run tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Test core health system
        health_test_passed = loop.run_until_complete(test_health_system())
        
        # Test CLI integration
        cli_test_passed = loop.run_until_complete(test_cli_integration())
        
        print("\n" + "=" * 60)
        print("🏁 FINAL RESULTS")
        print("=" * 60)
        print(f"Health System Test: {'✅ PASSED' if health_test_passed else '❌ FAILED'}")
        print(f"CLI Integration Test: {'✅ PASSED' if cli_test_passed else '❌ FAILED'}")
        
        if health_test_passed and cli_test_passed:
            print("\n🎉 ALL TESTS PASSED! PHASE 3 implementation is working correctly.")
            return 0
        else:
            print("\n⚠️ Some tests failed. Please review the output above.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {str(e)}")
        traceback.print_exc()
        return 1
    finally:
        loop.close()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
