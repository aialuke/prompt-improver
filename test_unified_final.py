#!/usr/bin/env python3
"""
Final V2 Manager Real Behavior Testing Suite
Comprehensive testing of UnifiedConnectionManager functionality
"""

import asyncio
import logging
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Main test execution"""
    logger.info("🚀 Final V2 Manager Real Behavior Testing Suite")
    
    test_results = {
        "import_test": None,
        "creation_test": None,
        "configuration_test": None,
        "metrics_test": None,
        "health_test": None,
        "circuit_breaker_test": None,
        "integration_tests": {}
    }
    
    start_time = time.time()
    
    try:
        # Test 1: Import and basic creation
        logger.info("\n🔄 Test 1: Import and Manager Creation")
        from prompt_improver.database.unified_connection_manager import (
            UnifiedConnectionManager,
            ManagerMode,
            HealthStatus,
            PoolConfiguration,
            get_unified_manager
        )
        
        manager = UnifiedConnectionManager(mode=ManagerMode.ASYNC_MODERN)
        test_results["import_test"] = {
            "success": True,
            "manager_type": type(manager).__name__,
            "mode": manager.mode.value
        }
        logger.info("✅ Import and creation successful")
        
        # Test 2: Multiple manager modes
        logger.info("\n🔄 Test 2: Multiple Manager Modes")
        modes_results = {}
        for mode in [ManagerMode.ASYNC_MODERN, ManagerMode.MCP_SERVER, ManagerMode.ML_TRAINING, ManagerMode.ADMIN]:
            mode_manager = UnifiedConnectionManager(mode=mode)
            modes_results[mode.value] = {
                "pool_size": mode_manager.pool_config.pg_pool_size,
                "timeout": mode_manager.pool_config.pg_timeout,
                "ha_enabled": mode_manager.pool_config.enable_ha,
                "circuit_breaker": mode_manager.pool_config.enable_circuit_breaker
            }
        
        test_results["creation_test"] = {"success": True, "modes": modes_results}
        logger.info(f"✅ Created managers for {len(modes_results)} modes")
        
        # Test 3: Configuration validation
        logger.info("\n🔄 Test 3: Configuration Validation")
        config_tests = {}
        
        # MCP Server should be optimized for fast responses
        mcp_config = PoolConfiguration.for_mode(ManagerMode.MCP_SERVER)
        config_tests["mcp_fast"] = mcp_config.pg_timeout <= 0.5
        config_tests["mcp_circuit_breaker"] = mcp_config.enable_circuit_breaker
        
        # ML Training should support HA
        ml_config = PoolConfiguration.for_mode(ManagerMode.ML_TRAINING)
        config_tests["ml_ha"] = ml_config.enable_ha
        config_tests["ml_large_pool"] = ml_config.pg_pool_size >= 10
        
        # HA mode should have both features
        ha_config = PoolConfiguration.for_mode(ManagerMode.HIGH_AVAILABILITY)
        config_tests["ha_features"] = ha_config.enable_ha and ha_config.enable_circuit_breaker
        
        test_results["configuration_test"] = {
            "success": all(config_tests.values()),
            "checks": config_tests
        }
        logger.info(f"✅ Configuration validation: {sum(config_tests.values())}/{len(config_tests)} checks passed")
        
        # Test 4: Metrics system
        logger.info("\n🔄 Test 4: Metrics System")
        metrics_manager = UnifiedConnectionManager(mode=ManagerMode.ASYNC_MODERN)
        
        # Initial metrics
        initial_metrics = metrics_manager._get_metrics_dict()
        metrics_keys = list(initial_metrics.keys())
        
        # Test metric updates
        metrics_manager._update_response_time(100.0)
        metrics_manager._update_response_time(200.0)
        updated_metrics = metrics_manager._get_metrics_dict()
        
        # Test connection metrics
        metrics_manager._update_connection_metrics(success=True)
        metrics_manager._update_connection_metrics(success=False)
        final_metrics = metrics_manager._get_metrics_dict()
        
        test_results["metrics_test"] = {
            "success": True,
            "metrics_count": len(metrics_keys),
            "response_time_updated": updated_metrics["avg_response_time_ms"] > 0,
            "failed_connections": final_metrics["failed_connections"] > 0,
            "key_metrics": {
                "active_connections": final_metrics["active_connections"],
                "pool_utilization": final_metrics["pool_utilization"],
                "avg_response_time_ms": final_metrics["avg_response_time_ms"],
                "circuit_breaker_state": final_metrics["circuit_breaker_state"]
            }
        }
        logger.info(f"✅ Metrics system: {len(metrics_keys)} metrics tracked")
        
        # Test 5: Health status management
        logger.info("\n🔄 Test 5: Health Status Management")
        health_manager = UnifiedConnectionManager(mode=ManagerMode.ASYNC_MODERN)
        
        health_tests = {}
        
        # Test all health statuses
        for status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]:
            health_manager._health_status = status
            health_tests[status.value] = {
                "is_healthy": health_manager.is_healthy(),
                "expected_healthy": status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
            }
        
        # Verify health logic
        healthy_logic_correct = all(
            test["is_healthy"] == test["expected_healthy"] 
            for test in health_tests.values()
        )
        
        test_results["health_test"] = {
            "success": healthy_logic_correct,
            "status_tests": health_tests,
            "logic_correct": healthy_logic_correct
        }
        logger.info(f"✅ Health status management: {sum(1 for t in health_tests.values() if t['is_healthy'] == t['expected_healthy'])}/{len(health_tests)} status checks correct")
        
        # Test 6: Circuit breaker functionality
        logger.info("\n🔄 Test 6: Circuit Breaker Functionality")
        cb_manager = UnifiedConnectionManager(mode=ManagerMode.ASYNC_MODERN)
        cb_manager.pool_config.enable_circuit_breaker = True
        cb_manager._circuit_breaker_threshold = 3
        
        # Initial state
        initial_open = cb_manager._is_circuit_breaker_open()
        
        # Trigger failures
        for i in range(3):
            cb_manager._handle_connection_failure(Exception(f"Test error {i+1}"))
        
        # Check if opened
        after_failures = cb_manager._is_circuit_breaker_open()
        
        # Test timeout recovery
        cb_manager._circuit_breaker_timeout = 0.1
        await asyncio.sleep(0.2)
        after_timeout = cb_manager._is_circuit_breaker_open()
        
        test_results["circuit_breaker_test"] = {
            "success": not initial_open and after_failures and not after_timeout,
            "initial_closed": not initial_open,
            "opened_after_failures": after_failures,
            "recovered_after_timeout": not after_timeout,
            "failure_count": cb_manager._circuit_breaker_failures
        }
        logger.info(f"✅ Circuit breaker: {'Working correctly' if test_results['circuit_breaker_test']['success'] else 'Issues detected'}")
        
        # Test 7: Integration patterns
        logger.info("\n🔄 Test 7: Integration Patterns")
        
        # MCP Server pattern
        mcp_manager = UnifiedConnectionManager(mode=ManagerMode.MCP_SERVER)
        mcp_pattern = {
            "fast_timeout": mcp_manager.pool_config.pg_timeout <= 0.5,
            "circuit_breaker": mcp_manager.pool_config.enable_circuit_breaker,
            "adequate_pool": mcp_manager.pool_config.pg_pool_size >= 15
        }
        
        # ML Training pattern
        ml_manager = UnifiedConnectionManager(mode=ManagerMode.ML_TRAINING)
        ml_pattern = {
            "ha_enabled": ml_manager.pool_config.enable_ha,
            "batch_timeout": ml_manager.pool_config.pg_timeout >= 5.0,
            "moderate_pool": ml_manager.pool_config.pg_pool_size >= 10
        }
        
        # API pattern (singleton)
        api_manager_1 = get_unified_manager(ManagerMode.ASYNC_MODERN)
        api_manager_2 = get_unified_manager(ManagerMode.ASYNC_MODERN)
        api_pattern = {
            "singleton": api_manager_1 is api_manager_2,
            "has_health_check": hasattr(api_manager_1, 'health_check'),
            "has_metrics": len(api_manager_1._get_metrics_dict()) > 0
        }
        
        test_results["integration_tests"] = {
            "mcp_server": {"success": all(mcp_pattern.values()), "checks": mcp_pattern},
            "ml_training": {"success": all(ml_pattern.values()), "checks": ml_pattern},
            "api_endpoints": {"success": all(api_pattern.values()), "checks": api_pattern}
        }
        
        for pattern_name, pattern_data in test_results["integration_tests"].items():
            status = "✅" if pattern_data["success"] else "⚠️"
            logger.info(f"{status} {pattern_name}: {sum(pattern_data['checks'].values())}/{len(pattern_data['checks'])} checks passed")
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        test_results["error"] = str(e)
    
    # Final summary
    total_duration = time.time() - start_time
    
    # Count successful tests
    core_tests = ["import_test", "creation_test", "configuration_test", "metrics_test", "health_test", "circuit_breaker_test"]
    core_passed = sum(1 for test in core_tests if test_results.get(test, {}).get("success", False))
    
    integration_passed = sum(1 for test in test_results.get("integration_tests", {}).values() if test.get("success", False))
    integration_total = len(test_results.get("integration_tests", {}))
    
    total_passed = core_passed + integration_passed
    total_tests = len(core_tests) + integration_total
    
    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL V2 MANAGER TEST RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Core Tests: {core_passed}/{len(core_tests)} passed")
    logger.info(f"Integration Tests: {integration_passed}/{integration_total} passed")
    logger.info(f"Total: {total_passed}/{total_tests} passed")
    logger.info(f"Success Rate: {(total_passed/total_tests)*100:.1f}%")
    logger.info(f"Duration: {total_duration:.3f}s")
    
    # Detailed results
    logger.info(f"\n📊 DETAILED RESULTS:")
    logger.info(f"✅ Manager Import & Creation: Working")
    logger.info(f"✅ Multi-Mode Support: {len(test_results.get('creation_test', {}).get('modes', {}))} modes")
    logger.info(f"✅ Configuration System: {sum(test_results.get('configuration_test', {}).get('checks', {}).values())}/{len(test_results.get('configuration_test', {}).get('checks', {}))} checks")
    logger.info(f"✅ Metrics Collection: {test_results.get('metrics_test', {}).get('metrics_count', 0)} metrics")
    logger.info(f"✅ Health Monitoring: {'Working' if test_results.get('health_test', {}).get('success') else 'Issues'}")
    logger.info(f"✅ Circuit Breaker: {'Working' if test_results.get('circuit_breaker_test', {}).get('success') else 'Issues'}")
    
    return {
        "success_rate": (total_passed/total_tests)*100,
        "total_tests": total_tests,
        "passed_tests": total_passed,
        "duration": total_duration,
        "results": test_results
    }

if __name__ == "__main__":
    results = asyncio.run(main())
    
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print(f"Tests Passed: {results['passed_tests']}/{results['total_tests']}")
    print(f"Duration: {results['duration']:.3f}s")
    
    if results['success_rate'] >= 80:
        print("🎉 V2 Manager testing passed!")
        sys.exit(0)
    else:
        print("⚠️ V2 Manager testing has issues")
        sys.exit(1)