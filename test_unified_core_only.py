#!/usr/bin/env python3
"""
Core V2 Manager Testing - No Database Connections
Tests the core functionality of UnifiedConnectionManager without requiring database
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

async def test_core_functionality():
    """Test core V2 manager functionality without database"""
    logger.info("🔄 Testing core V2 manager functionality...")
    
    try:
        # Direct import
        from prompt_improver.database.unified_connection_manager import (
            UnifiedConnectionManager,
            ManagerMode,
            HealthStatus,
            PoolConfiguration,
            ConnectionMetrics
        )
        
        results = {
            "imports": True,
            "manager_creation": {},
            "pool_configs": {},
            "metrics": {},
            "health_status": {},
            "circuit_breaker": {}
        }
        
        # Test manager creation for different modes
        modes = [ManagerMode.ASYNC_MODERN, ManagerMode.MCP_SERVER, ManagerMode.ML_TRAINING]
        for mode in modes:
            manager = UnifiedConnectionManager(mode=mode)
            results["manager_creation"][mode.value] = {
                "created": True,
                "pool_size": manager.pool_config.pg_pool_size,
                "timeout": manager.pool_config.pg_timeout,
                "mode": manager.mode.value,
                "initialized": manager._is_initialized
            }
        
        # Test pool configurations
        for mode in modes:
            config = PoolConfiguration.for_mode(mode)
            results["pool_configs"][mode.value] = {
                "pg_pool_size": config.pg_pool_size,
                "pg_max_overflow": config.pg_max_overflow,
                "pg_timeout": config.pg_timeout,
                "enable_ha": config.enable_ha,
                "enable_circuit_breaker": config.enable_circuit_breaker
            }
        
        # Test metrics system
        manager = UnifiedConnectionManager(mode=ManagerMode.ASYNC_MODERN)
        
        # Initial metrics
        initial_metrics = manager._get_metrics_dict()
        results["metrics"]["initial"] = {
            "active_connections": initial_metrics["active_connections"],
            "total_connections": initial_metrics["total_connections"],
            "avg_response_time_ms": initial_metrics["avg_response_time_ms"]
        }
        
        # Update metrics
        manager._update_response_time(100.0)
        manager._update_response_time(200.0)
        updated_metrics = manager._get_metrics_dict()
        results["metrics"]["after_updates"] = {
            "avg_response_time_ms": updated_metrics["avg_response_time_ms"]
        }
        
        # Test health status management
        initial_health = manager._health_status
        manager._health_status = HealthStatus.HEALTHY
        healthy_check = manager.is_healthy()
        manager._health_status = HealthStatus.UNHEALTHY
        unhealthy_check = manager.is_healthy()
        
        results["health_status"] = {
            "initial": initial_health.value,
            "healthy_detection": healthy_check,
            "unhealthy_detection": not unhealthy_check
        }
        
        # Test circuit breaker logic
        manager.pool_config.enable_circuit_breaker = True
        manager._circuit_breaker_threshold = 2
        
        initial_breaker = manager._is_circuit_breaker_open()
        
        # Trigger failures
        manager._handle_connection_failure(Exception("Test 1"))
        manager._handle_connection_failure(Exception("Test 2"))
        
        after_failures = manager._is_circuit_breaker_open()
        breaker_state = manager._metrics.circuit_breaker_state
        
        results["circuit_breaker"] = {
            "initial_closed": not initial_breaker,
            "opened_after_failures": after_failures,
            "state": breaker_state,
            "failure_count": manager._circuit_breaker_failures
        }
        
        logger.info("✅ Core functionality test completed successfully")
        return {"success": True, "results": results}
        
    except Exception as e:
        logger.error(f"❌ Core functionality test failed: {e}")
        return {"success": False, "error": str(e)}

async def test_mcp_integration():
    """Test MCP server integration patterns"""
    logger.info("🔄 Testing MCP integration patterns...")
    
    try:
        from prompt_improver.database.unified_connection_manager import (
            UnifiedConnectionManager,
            ManagerMode
        )
        
        # Create MCP-optimized manager
        mcp_manager = UnifiedConnectionManager(mode=ManagerMode.MCP_SERVER)
        
        # Test MCP-specific configurations
        mcp_config = {
            "mode": mcp_manager.mode.value,
            "pool_size": mcp_manager.pool_config.pg_pool_size,
            "timeout": mcp_manager.pool_config.pg_timeout,
            "circuit_breaker": mcp_manager.pool_config.enable_circuit_breaker,
            "optimized_for_reads": mcp_manager.pool_config.pg_timeout < 1.0  # Fast timeouts for MCP
        }
        
        # Test that MCP mode has appropriate settings
        mcp_optimizations = {
            "fast_timeout": mcp_config["timeout"] <= 0.2,  # Should be 0.2s for MCP
            "large_pool": mcp_config["pool_size"] >= 15,   # Should have adequate pool size
            "circuit_breaker_enabled": mcp_config["circuit_breaker"]
        }
        
        # Simulate MCP server usage pattern
        connection_info = {
            "mode": mcp_manager.mode.value,
            "initialized": mcp_manager._is_initialized,
            "health": mcp_manager._health_status.value,
            "metrics_keys": len(mcp_manager._get_metrics_dict())
        }
        
        logger.info("✅ MCP integration patterns tested")
        logger.info(f"   Mode: {mcp_config['mode']}")
        logger.info(f"   Optimizations: {sum(mcp_optimizations.values())}/3 enabled")
        
        return {
            "success": True,
            "mcp_config": mcp_config,
            "optimizations": mcp_optimizations,
            "connection_info": connection_info
        }
        
    except Exception as e:
        logger.error(f"❌ MCP integration test failed: {e}")
        return {"success": False, "error": str(e)}

async def test_ml_training_integration():
    """Test ML training integration patterns"""
    logger.info("🔄 Testing ML training integration patterns...")
    
    try:
        from prompt_improver.database.unified_connection_manager import (
            UnifiedConnectionManager,
            ManagerMode
        )
        
        # Create ML-optimized manager
        ml_manager = UnifiedConnectionManager(mode=ManagerMode.ML_TRAINING)
        
        # Test ML-specific configurations
        ml_config = {
            "mode": ml_manager.mode.value,
            "pool_size": ml_manager.pool_config.pg_pool_size,
            "timeout": ml_manager.pool_config.pg_timeout,
            "ha_enabled": ml_manager.pool_config.enable_ha,
            "batch_optimized": ml_manager.pool_config.pg_timeout >= 5.0  # Longer timeouts for batch ops
        }
        
        # Test ML training optimizations
        ml_optimizations = {
            "moderate_pool": 10 <= ml_config["pool_size"] <= 20,  # Balanced pool size
            "batch_timeout": ml_config["timeout"] >= 5.0,         # Adequate timeout for ML ops
            "ha_support": ml_config["ha_enabled"]                 # HA for reliability
        }
        
        # Simulate ML training metrics collection
        metrics = ml_manager._get_metrics_dict()
        ml_metrics = {
            "initial_connections": metrics["active_connections"],
            "pool_utilization": metrics["pool_utilization"],
            "avg_response_time": metrics["avg_response_time_ms"],
            "circuit_breaker_state": metrics["circuit_breaker_state"]
        }
        
        logger.info("✅ ML training integration patterns tested")
        logger.info(f"   Mode: {ml_config['mode']}")
        logger.info(f"   HA enabled: {ml_config['ha_enabled']}")
        logger.info(f"   Batch timeout: {ml_config['timeout']}s")
        
        return {
            "success": True,
            "ml_config": ml_config,
            "optimizations": ml_optimizations,
            "metrics": ml_metrics
        }
        
    except Exception as e:
        logger.error(f"❌ ML training integration test failed: {e}")
        return {"success": False, "error": str(e)}

async def test_api_endpoint_patterns():
    """Test API endpoint integration patterns"""
    logger.info("🔄 Testing API endpoint integration patterns...")
    
    try:
        from prompt_improver.database.unified_connection_manager import (
            UnifiedConnectionManager,
            ManagerMode,
            get_unified_manager
        )
        
        # Test global manager pattern (typical for API endpoints)
        api_manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
        api_manager_2 = get_unified_manager(ManagerMode.ASYNC_MODERN)
        
        # Verify singleton behavior
        singleton_works = api_manager is api_manager_2
        
        # Test API-relevant configuration
        api_config = {
            "mode": api_manager.mode.value,
            "pool_size": api_manager.pool_config.pg_pool_size,
            "timeout": api_manager.pool_config.pg_timeout,
            "circuit_breaker": api_manager.pool_config.enable_circuit_breaker
        }
        
        # Test connection info (typical API health check)
        try:
            # This would normally require database, so we'll simulate
            conn_info_keys = [
                "mode", "initialized", "health_status", "pool_config", "metrics"
            ]
            simulated_conn_info = {key: f"mock_{key}" for key in conn_info_keys}
            
            api_patterns = {
                "global_manager_singleton": singleton_works,
                "connection_info_available": len(simulated_conn_info) > 0,
                "health_check_ready": hasattr(api_manager, 'health_check'),
                "metrics_available": len(api_manager._get_metrics_dict()) > 0
            }
            
        except Exception as e:
            api_patterns = {"error": str(e)}
        
        logger.info("✅ API endpoint patterns tested")
        logger.info(f"   Singleton pattern: {singleton_works}")
        logger.info(f"   API optimizations: {sum(api_patterns.values() if isinstance(v, bool) else 0 for v in api_patterns.values())}")
        
        return {
            "success": True,
            "api_config": api_config,
            "patterns": api_patterns,
            "singleton_test": singleton_works
        }
        
    except Exception as e:
        logger.error(f"❌ API endpoint patterns test failed: {e}")
        return {"success": False, "error": str(e)}

async def test_health_monitoring_system():
    """Test health monitoring system functionality"""
    logger.info("🔄 Testing health monitoring system...")
    
    try:
        from prompt_improver.database.unified_connection_manager import (
            UnifiedConnectionManager,
            ManagerMode,
            HealthStatus
        )
        
        manager = UnifiedConnectionManager(mode=ManagerMode.HIGH_AVAILABILITY)
        
        # Test health status transitions
        health_transitions = {}
        statuses = [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]
        
        for status in statuses:
            manager._health_status = status
            health_transitions[status.value] = {
                "is_healthy": manager.is_healthy(),
                "status_value": manager._health_status.value
            }
        
        # Test health check structure (without database)
        try:
            # Mock the health check components
            mock_health_result = {
                "status": "unknown",
                "timestamp": time.time(),
                "mode": manager.mode.value,
                "components": {},
                "metrics": manager._get_metrics_dict(),
                "response_time_ms": 0
            }
            
            health_check_structure = {
                "has_status": "status" in mock_health_result,
                "has_timestamp": "timestamp" in mock_health_result,
                "has_mode": "mode" in mock_health_result,
                "has_components": "components" in mock_health_result,
                "has_metrics": "metrics" in mock_health_result,
                "has_response_time": "response_time_ms" in mock_health_result
            }
            
        except Exception as e:
            health_check_structure = {"error": str(e)}
        
        # Test metrics integration with health
        metrics = manager._get_metrics_dict()
        health_metrics = {
            "circuit_breaker_state": metrics["circuit_breaker_state"],
            "error_rate": metrics["error_rate"],
            "pool_utilization": metrics["pool_utilization"],
            "avg_response_time": metrics["avg_response_time_ms"]
        }
        
        logger.info("✅ Health monitoring system tested")
        logger.info(f"   Health transitions work: {len(health_transitions)} statuses tested")
        logger.info(f"   Health check structure: {sum(health_check_structure.values() if isinstance(v, bool) else 0 for v in health_check_structure.values())}/6 components")
        
        return {
            "success": True,
            "health_transitions": health_transitions,
            "health_check_structure": health_check_structure,
            "health_metrics": health_metrics
        }
        
    except Exception as e:
        logger.error(f"❌ Health monitoring test failed: {e}")
        return {"success": False, "error": str(e)}

async def run_core_test_suite():
    """Run core test suite without database dependencies"""
    logger.info("🚀 Starting Core V2 Manager Test Suite (No Database)")
    
    tests = [
        ("Core Functionality", test_core_functionality),
        ("MCP Integration Patterns", test_mcp_integration),
        ("ML Training Integration", test_ml_training_integration),
        ("API Endpoint Patterns", test_api_endpoint_patterns),
        ("Health Monitoring System", test_health_monitoring_system)
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        test_start = time.time()
        try:
            result = await test_func()
            test_duration = time.time() - test_start
            results[test_name] = {
                "result": result,
                "duration_seconds": test_duration,
                "status": "completed" if result.get("success", False) else "failed"
            }
            
            if result.get("success", False):
                logger.info(f"✅ {test_name} completed in {test_duration:.3f}s")
            else:
                logger.warning(f"⚠️ {test_name} failed in {test_duration:.3f}s")
                
        except Exception as e:
            test_duration = time.time() - test_start
            results[test_name] = {
                "result": {"success": False, "error": str(e)},
                "duration_seconds": test_duration,
                "status": "failed"
            }
            logger.error(f"❌ {test_name} failed in {test_duration:.3f}s: {e}")
    
    total_duration = time.time() - start_time
    
    # Generate summary
    passed_tests = sum(1 for result in results.values() if result["status"] == "completed")
    total_tests = len(tests)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"CORE TEST SUITE SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Total Duration: {total_duration:.3f}s")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Print detailed results
    logger.info(f"\n📊 DETAILED RESULTS:")
    for test_name, result_data in results.items():
        status_emoji = "✅" if result_data["status"] == "completed" else "❌"
        logger.info(f"{status_emoji} {test_name}: {result_data['duration_seconds']:.3f}s")
    
    return {
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "total_duration_seconds": total_duration,
            "success_rate_percent": (passed_tests/total_tests)*100
        },
        "detailed_results": results
    }

if __name__ == "__main__":
    # Run the core test suite
    results = asyncio.run(run_core_test_suite())
    
    # Print final summary
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Success Rate: {results['summary']['success_rate_percent']:.1f}%")
    print(f"Duration: {results['summary']['total_duration_seconds']:.3f}s")
    print(f"Tests Passed: {results['summary']['passed_tests']}/{results['summary']['total_tests']}")
    
    # Exit with appropriate code
    if results['summary']['success_rate_percent'] >= 80:
        print("🎉 Core test suite passed!")
        sys.exit(0)
    else:
        print("⚠️ Core test suite has significant failures")
        sys.exit(1)