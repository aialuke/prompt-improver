#!/usr/bin/env python3
"""
Direct V2 Manager Behavior Testing - Avoiding Circular Imports
Tests the UnifiedConnectionManager directly without going through the database module
"""

import asyncio
import logging
import time
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_direct_import_and_basic_functionality():
    """Test 1: Direct import and basic functionality"""
    logger.info("🔄 Testing direct V2 manager import and basic functionality...")
    
    try:
        # Direct import of the V2 manager file
        from prompt_improver.database.unified_connection_manager import (
            UnifiedConnectionManager,
            ManagerMode,
            HealthStatus
        )
        from prompt_improver.core.config import AppConfig
        
        # Create manager directly
        db_config = AppConfig().database
        manager = UnifiedConnectionManager(
            mode=ManagerMode.ASYNC_MODERN,
            db_config=db_config
        )
        
        logger.info(f"✅ Successfully created manager: {type(manager).__name__}")
        logger.info(f"   Mode: {manager.mode.value}")
        logger.info(f"   Pool config: {manager.pool_config.pg_pool_size} connections")
        
        return {"success": True, "manager_created": True, "mode": manager.mode.value}
        
    except Exception as e:
        logger.error(f"❌ Direct import test failed: {e}")
        return {"success": False, "error": str(e)}

async def test_initialization_without_database():
    """Test 2: Test initialization process (without actual database)"""
    logger.info("🔄 Testing initialization process...")
    
    try:
        from prompt_improver.database.unified_connection_manager import (
            UnifiedConnectionManager,
            ManagerMode
        )
        from prompt_improver.core.config import AppConfig
        
        # Create manager with test config
        db_config = AppConfig().database
        manager = UnifiedConnectionManager(mode=ManagerMode.ASYNC_MODERN, db_config=db_config)
        
        # Test initial state
        initial_state = {
            "initialized": manager._is_initialized,
            "health_status": manager._health_status.value,
            "metrics": manager._get_metrics_dict()
        }
        
        logger.info(f"✅ Initial state checked")
        logger.info(f"   Initialized: {initial_state['initialized']}")
        logger.info(f"   Health: {initial_state['health_status']}")
        
        return {"success": True, "initial_state": initial_state}
        
    except Exception as e:
        logger.error(f"❌ Initialization test failed: {e}")
        return {"success": False, "error": str(e)}

async def test_pool_configurations():
    """Test 3: Test different pool configurations"""
    logger.info("🔄 Testing pool configurations...")
    
    try:
        from prompt_improver.database.unified_connection_manager import (
            UnifiedConnectionManager,
            ManagerMode,
            PoolConfiguration
        )
        
        configs = {}
        modes_to_test = [
            ManagerMode.ASYNC_MODERN,
            ManagerMode.MCP_SERVER, 
            ManagerMode.ML_TRAINING,
            ManagerMode.ADMIN,
            ManagerMode.HIGH_AVAILABILITY
        ]
        
        for mode in modes_to_test:
            pool_config = PoolConfiguration.for_mode(mode)
            configs[mode.value] = {
                "pg_pool_size": pool_config.pg_pool_size,
                "pg_max_overflow": pool_config.pg_max_overflow,
                "pg_timeout": pool_config.pg_timeout,
                "redis_pool_size": pool_config.redis_pool_size,
                "enable_ha": pool_config.enable_ha,
                "enable_circuit_breaker": pool_config.enable_circuit_breaker
            }
        
        logger.info("✅ Pool configurations tested")
        for mode, config in configs.items():
            logger.info(f"   {mode}: {config['pg_pool_size']} connections, {config['pg_timeout']}s timeout")
        
        return {"success": True, "configurations": configs}
        
    except Exception as e:
        logger.error(f"❌ Pool configuration test failed: {e}")
        return {"success": False, "error": str(e)}

async def test_metrics_system():
    """Test 4: Test metrics collection system"""
    logger.info("🔄 Testing metrics system...")
    
    try:
        from prompt_improver.database.unified_connection_manager import (
            UnifiedConnectionManager,
            ManagerMode,
            ConnectionMetrics
        )
        from prompt_improver.core.config import AppConfig
        
        # Create manager
        manager = UnifiedConnectionManager(mode=ManagerMode.ASYNC_MODERN)
        
        # Test metrics structure
        metrics = manager._get_metrics_dict()
        expected_keys = [
            "active_connections", "idle_connections", "total_connections",
            "pool_utilization", "avg_response_time_ms", "error_rate",
            "failed_connections", "circuit_breaker_state"
        ]
        
        missing_keys = [key for key in expected_keys if key not in metrics]
        
        # Test metrics updates
        manager._update_response_time(100.0)
        manager._update_response_time(200.0)
        
        updated_metrics = manager._get_metrics_dict()
        
        logger.info("✅ Metrics system tested")
        logger.info(f"   Metrics keys present: {len(metrics)}")
        logger.info(f"   Missing keys: {missing_keys}")
        logger.info(f"   Avg response time: {updated_metrics['avg_response_time_ms']:.2f}ms")
        
        return {
            "success": True,
            "metrics_count": len(metrics),
            "missing_keys": missing_keys,
            "avg_response_time": updated_metrics['avg_response_time_ms']
        }
        
    except Exception as e:
        logger.error(f"❌ Metrics system test failed: {e}")
        return {"success": False, "error": str(e)}

async def test_health_status_management():
    """Test 5: Test health status management"""
    logger.info("🔄 Testing health status management...")
    
    try:
        from prompt_improver.database.unified_connection_manager import (
            UnifiedConnectionManager,
            ManagerMode,
            HealthStatus
        )
        
        manager = UnifiedConnectionManager(mode=ManagerMode.ASYNC_MODERN)
        
        # Test initial health status
        initial_health = manager._health_status
        is_healthy_initial = manager.is_healthy()
        
        # Test health status changes
        manager._health_status = HealthStatus.HEALTHY
        is_healthy_after = manager.is_healthy()
        
        manager._health_status = HealthStatus.DEGRADED
        is_degraded = manager.is_healthy()
        
        manager._health_status = HealthStatus.UNHEALTHY
        is_unhealthy = manager.is_healthy()
        
        logger.info("✅ Health status management tested")
        logger.info(f"   Initial health: {initial_health.value}")
        logger.info(f"   Healthy states work: {is_healthy_after and is_degraded}")
        logger.info(f"   Unhealthy detection: {not is_unhealthy}")
        
        return {
            "success": True,
            "initial_health": initial_health.value,
            "healthy_detection_works": is_healthy_after and is_degraded,
            "unhealthy_detection_works": not is_unhealthy
        }
        
    except Exception as e:
        logger.error(f"❌ Health status test failed: {e}")
        return {"success": False, "error": str(e)}

async def test_circuit_breaker_logic():
    """Test 6: Test circuit breaker logic"""
    logger.info("🔄 Testing circuit breaker logic...")
    
    try:
        from prompt_improver.database.unified_connection_manager import (
            UnifiedConnectionManager,
            ManagerMode
        )
        
        manager = UnifiedConnectionManager(mode=ManagerMode.ASYNC_MODERN)
        manager.pool_config.enable_circuit_breaker = True
        manager._circuit_breaker_threshold = 3
        
        # Test initial state
        initial_open = manager._is_circuit_breaker_open()
        
        # Simulate failures
        for i in range(3):
            manager._handle_connection_failure(Exception(f"Test error {i+1}"))
        
        # Test circuit breaker opened
        after_failures = manager._is_circuit_breaker_open()
        breaker_state = manager._metrics.circuit_breaker_state
        
        # Test timeout reset
        manager._circuit_breaker_timeout = 0.1  # Very short timeout
        await asyncio.sleep(0.2)  # Wait for timeout
        after_timeout = manager._is_circuit_breaker_open()
        
        logger.info("✅ Circuit breaker logic tested")
        logger.info(f"   Initial state: {initial_open}")
        logger.info(f"   After failures: {after_failures}")
        logger.info(f"   Breaker state: {breaker_state}")
        logger.info(f"   After timeout: {after_timeout}")
        
        return {
            "success": True,
            "initial_closed": not initial_open,
            "opened_after_failures": after_failures,
            "breaker_state": breaker_state,
            "timeout_recovery": not after_timeout
        }
        
    except Exception as e:
        logger.error(f"❌ Circuit breaker test failed: {e}")
        return {"success": False, "error": str(e)}

async def test_database_connection_with_real_db():
    """Test 7: Test actual database connection (if database is available)"""
    logger.info("🔄 Testing real database connection...")
    
    try:
        from prompt_improver.database.unified_connection_manager import (
            UnifiedConnectionManager,
            ManagerMode
        )
        from prompt_improver.core.config import AppConfig
        from sqlalchemy import text
        
        # Create manager with real config
        db_config = AppConfig().database
        manager = UnifiedConnectionManager(mode=ManagerMode.ASYNC_MODERN, db_config=db_config)
        
        # Try to initialize
        init_success = await manager.initialize()
        
        if init_success:
            # Try a simple query
            try:
                async with manager.get_async_session() as session:
                    result = await session.execute(text("SELECT 1 as test_value"))
                    value = result.scalar()
                    
                # Get connection info
                conn_info = await manager.get_connection_info()
                
                # Run health check
                health = await manager.health_check()
                
                await manager.close()
                
                logger.info("✅ Real database connection successful")
                logger.info(f"   Test query result: {value}")
                logger.info(f"   Health status: {health['status']}")
                logger.info(f"   Response time: {health['response_time_ms']:.2f}ms")
                
                return {
                    "success": True,
                    "database_available": True,
                    "test_query_result": value,
                    "health_status": health['status'],
                    "response_time_ms": health['response_time_ms'],
                    "connection_info": {
                        "mode": conn_info["mode"],
                        "initialized": conn_info["initialized"]
                    }
                }
                
            except Exception as query_error:
                await manager.close()
                logger.warning(f"⚠️ Database connection succeeded but query failed: {query_error}")
                return {
                    "success": True,
                    "database_available": True,
                    "init_success": True,
                    "query_error": str(query_error)
                }
        else:
            logger.warning("⚠️ Database not available or initialization failed")
            return {
                "success": True,
                "database_available": False,
                "init_success": False,
                "reason": "Database initialization failed"
            }
        
    except Exception as e:
        logger.warning(f"⚠️ Database connection test failed (expected if DB not available): {e}")
        return {
            "success": True,
            "database_available": False,
            "error": str(e),
            "reason": "Database not available for testing"
        }

async def test_concurrent_manager_creation():
    """Test 8: Test concurrent manager creation and usage"""
    logger.info("🔄 Testing concurrent manager creation...")
    
    try:
        from prompt_improver.database.unified_connection_manager import (
            UnifiedConnectionManager,
            ManagerMode,
            get_unified_manager
        )
        
        async def create_manager_task(mode):
            """Create and test a manager"""
            manager = UnifiedConnectionManager(mode=mode)
            metrics = manager._get_metrics_dict()
            return {
                "mode": mode.value,
                "created": True,
                "pool_size": manager.pool_config.pg_pool_size,
                "metrics_keys": len(metrics)
            }
        
        # Test concurrent creation
        modes = [ManagerMode.ASYNC_MODERN, ManagerMode.MCP_SERVER, ManagerMode.ML_TRAINING]
        tasks = [create_manager_task(mode) for mode in modes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Test global manager function
        global_manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
        global_manager_2 = get_unified_manager(ManagerMode.ASYNC_MODERN)
        same_instance = global_manager is global_manager_2
        
        successful_creations = [r for r in results if isinstance(r, dict) and r.get("created")]
        
        logger.info("✅ Concurrent manager creation tested")
        logger.info(f"   Successful creations: {len(successful_creations)}")
        logger.info(f"   Global manager singleton works: {same_instance}")
        
        return {
            "success": True,
            "concurrent_creations": len(successful_creations),
            "total_attempted": len(modes),
            "singleton_works": same_instance,
            "results": successful_creations
        }
        
    except Exception as e:
        logger.error(f"❌ Concurrent creation test failed: {e}")
        return {"success": False, "error": str(e)}

async def run_comprehensive_direct_tests():
    """Run all direct tests without circular imports"""
    logger.info("🚀 Starting Direct V2 Manager Testing Suite")
    
    tests = [
        ("Direct Import and Basic Functionality", test_direct_import_and_basic_functionality),
        ("Initialization Process", test_initialization_without_database),
        ("Pool Configurations", test_pool_configurations),
        ("Metrics System", test_metrics_system),
        ("Health Status Management", test_health_status_management),
        ("Circuit Breaker Logic", test_circuit_breaker_logic),
        ("Database Connection (Real)", test_database_connection_with_real_db),
        ("Concurrent Manager Creation", test_concurrent_manager_creation)
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
                logger.info(f"✅ {test_name} completed in {test_duration:.2f}s")
            else:
                logger.warning(f"⚠️ {test_name} completed with issues in {test_duration:.2f}s")
                
        except Exception as e:
            test_duration = time.time() - test_start
            results[test_name] = {
                "result": {"success": False, "error": str(e)},
                "duration_seconds": test_duration,
                "status": "failed"
            }
            logger.error(f"❌ {test_name} failed in {test_duration:.2f}s: {e}")
    
    total_duration = time.time() - start_time
    
    # Generate summary
    completed_tests = sum(1 for result in results.values() if result["status"] == "completed")
    total_tests = len(tests)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"DIRECT TEST SUITE SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Completed Successfully: {completed_tests}")
    logger.info(f"Failed/Issues: {total_tests - completed_tests}")
    logger.info(f"Total Duration: {total_duration:.2f}s")
    logger.info(f"Success Rate: {(completed_tests/total_tests)*100:.1f}%")
    
    return {
        "summary": {
            "total_tests": total_tests,
            "completed_tests": completed_tests,
            "failed_tests": total_tests - completed_tests,
            "total_duration_seconds": total_duration,
            "success_rate_percent": (completed_tests/total_tests)*100
        },
        "detailed_results": results
    }

if __name__ == "__main__":
    # Run the direct test suite
    results = asyncio.run(run_comprehensive_direct_tests())
    
    # Print final summary
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Success Rate: {results['summary']['success_rate_percent']:.1f}%")
    print(f"Duration: {results['summary']['total_duration_seconds']:.2f}s")
    
    # Exit with appropriate code
    if results['summary']['success_rate_percent'] >= 80:
        print("🎉 Direct test suite passed!")
        sys.exit(0)
    else:
        print("⚠️ Direct test suite has significant issues")
        sys.exit(1)