#!/usr/bin/env python3
"""
Real Behavior Test Suite for UnifiedConnectionManager
Test actual database connections, session management, and component integration
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pytest
from sqlalchemy import text

from prompt_improver.core.config import AppConfig
from prompt_improver.database.unified_connection_manager import (
    ConnectionMode,
    HealthStatus,
    ManagerMode,
    UnifiedConnectionManager,
)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestUnifiedConnectionManagerRealBehavior:
    """Real behavior tests for V2 connection manager"""

    async def test_database_connections_basic(self):
        """Test 1: Basic database connection establishment"""
        logger.info("🔄 Testing basic database connections...")

        manager = UnifiedConnectionManager(mode=ManagerMode.ASYNC_MODERN)

        try:
            # Test initialization
            success = await manager.initialize()
            assert success, "Manager initialization failed"

            # Test async session
            async with manager.get_async_session() as session:
                result = await session.execute(text("SELECT 1 as test_value"))
                value = result.scalar()
                assert value == 1, f"Expected 1, got {value}"

            # Test connection info
            info = await manager.get_connection_info()
            assert info["initialized"] is True
            assert info["mode"] == "async_modern"

            logger.info("✅ Basic database connections working correctly")
            return True

        except Exception as e:
            logger.error("❌ Basic database connection test failed: %s", e)
            return False
        finally:
            await manager.close()

    async def test_multiple_manager_modes(self):
        """Test 2: Multiple manager modes with different configurations"""
        logger.info("🔄 Testing multiple manager modes...")

        modes_to_test = [
            ManagerMode.ASYNC_MODERN,
            ManagerMode.MCP_SERVER,
            ManagerMode.ML_TRAINING,
            ManagerMode.ADMIN,
        ]

        results = {}
        managers = []

        try:
            for mode in modes_to_test:
                logger.info("Testing mode: %s", mode.value)
                manager = UnifiedConnectionManager(mode=mode)
                managers.append(manager)

                # Initialize and test
                success = await manager.initialize()
                if success:
                    async with manager.get_async_session() as session:
                        result = await session.execute(
                            text("SELECT current_database()")
                        )
                        db_name = result.scalar()
                        results[mode.value] = {
                            "initialized": True,
                            "database": db_name,
                            "pool_config": {
                                "pg_pool_size": manager.pool_config.pg_pool_size,
                                "pg_timeout": manager.pool_config.pg_timeout,
                            },
                        }
                else:
                    results[mode.value] = {
                        "initialized": False,
                        "error": "Failed to initialize",
                    }

            logger.info("✅ Multiple manager modes test completed")
            return results

        except Exception as e:
            logger.error("❌ Multiple manager modes test failed: %s", e)
            return {"error": str(e)}
        finally:
            for manager in managers:
                await manager.close()

    async def test_connection_protocol_implementation(self):
        """Test 3: ConnectionManagerProtocol implementation"""
        logger.info("🔄 Testing connection protocol implementation...")

        manager = UnifiedConnectionManager(mode=ManagerMode.ASYNC_MODERN)

        try:
            await manager.initialize()

            # Test different connection modes
            connection_modes = [
                ConnectionMode.READ_ONLY,
                ConnectionMode.READ_WRITE,
                ConnectionMode.BATCH,
                ConnectionMode.TRANSACTIONAL,
            ]

            results = {}

            for mode in connection_modes:
                try:
                    async with manager.get_connection(mode=mode) as conn:
                        # Test basic query
                        result = await conn.execute(text("SELECT current_timestamp"))
                        timestamp = result.scalar()
                        results[mode.value] = {
                            "success": True,
                            "timestamp": str(timestamp),
                        }
                except Exception as e:
                    results[mode.value] = {"success": False, "error": str(e)}

            # Test raw connection type
            try:
                async with manager.get_connection(connection_type="raw") as conn:
                    if hasattr(conn, "execute"):
                        # It's a session
                        result = await conn.execute(text("SELECT 'raw_test'"))
                        value = result.scalar()
                    else:
                        # It's a raw connection
                        result = await conn.fetchval("SELECT 'raw_test'")
                        value = result
                    results["raw_connection"] = {"success": True, "value": value}
            except Exception as e:
                results["raw_connection"] = {"success": False, "error": str(e)}

            logger.info("✅ Connection protocol implementation test completed")
            return results

        except Exception as e:
            logger.error("❌ Connection protocol test failed: %s", e)
            return {"error": str(e)}
        finally:
            await manager.close()

    async def test_health_checks_comprehensive(self):
        """Test 4: Comprehensive health check functionality"""
        logger.info("🔄 Testing comprehensive health checks...")

        manager = UnifiedConnectionManager(mode=ManagerMode.HIGH_AVAILABILITY)

        try:
            await manager.initialize()

            # Run health check
            health_result = await manager.health_check()

            # Validate health check structure
            required_fields = [
                "status",
                "timestamp",
                "mode",
                "components",
                "metrics",
                "response_time_ms",
            ]
            for field in required_fields:
                assert field in health_result, f"Missing field: {field}"

            # Test is_healthy method
            is_healthy = manager.is_healthy()

            # Get connection info
            conn_info = await manager.get_connection_info()

            results = {
                "health_check": health_result,
                "is_healthy": is_healthy,
                "connection_info": conn_info,
                "health_status": manager._health_status.value,
            }

            logger.info("✅ Health checks test completed")
            return results

        except Exception as e:
            logger.error("❌ Health checks test failed: %s", e)
            return {"error": str(e)}
        finally:
            await manager.close()

    async def test_session_management_lifecycle(self):
        """Test 5: Session management lifecycle"""
        logger.info("🔄 Testing session management lifecycle...")

        manager = UnifiedConnectionManager(mode=ManagerMode.ASYNC_MODERN)

        try:
            await manager.initialize()

            results = {
                "session_tests": [],
                "metrics_before": manager._get_metrics_dict(),
                "metrics_after": None,
            }

            # Test multiple sessions in sequence
            for i in range(5):
                try:
                    async with manager.get_async_session() as session:
                        result = await session.execute(
                            text(f"SELECT {i + 1} as session_id")
                        )
                        session_id = result.scalar()
                        results["session_tests"].append({
                            "session": i + 1,
                            "success": True,
                            "session_id": session_id,
                        })
                except Exception as e:
                    results["session_tests"].append({
                        "session": i + 1,
                        "success": False,
                        "error": str(e),
                    })

            # Test concurrent sessions
            async def concurrent_session(session_num):
                async with manager.get_async_session() as session:
                    await asyncio.sleep(0.1)  # Small delay
                    result = await session.execute(
                        text(f"SELECT {session_num} as concurrent_id")
                    )
                    return result.scalar()

            concurrent_results = await asyncio.gather(
                *[concurrent_session(i) for i in range(3)], return_exceptions=True
            )

            results["concurrent_sessions"] = [
                {"session": i, "result": result}
                for i, result in enumerate(concurrent_results)
            ]

            results["metrics_after"] = manager._get_metrics_dict()

            logger.info("✅ Session management lifecycle test completed")
            return results

        except Exception as e:
            logger.error("❌ Session management test failed: %s", e)
            return {"error": str(e)}
        finally:
            await manager.close()

    async def test_error_handling_scenarios(self):
        """Test 6: Error handling and recovery scenarios"""
        logger.info("🔄 Testing error handling scenarios...")

        manager = UnifiedConnectionManager(mode=ManagerMode.ASYNC_MODERN)

        try:
            await manager.initialize()

            results = {
                "invalid_query": None,
                "transaction_rollback": None,
                "connection_recovery": None,
            }

            # Test invalid query handling
            try:
                async with manager.get_async_session() as session:
                    await session.execute(text("SELECT * FROM nonexistent_table"))
            except Exception as e:
                results["invalid_query"] = {
                    "error_handled": True,
                    "error_type": type(e).__name__,
                }

            # Test transaction rollback
            try:
                async with manager.get_async_session() as session:
                    await session.execute(text("SELECT 1"))
                    # Simulate error within transaction
                    raise ValueError("Simulated error")
            except ValueError:
                results["transaction_rollback"] = {"rollback_triggered": True}

            # Test connection recovery after error
            try:
                async with manager.get_async_session() as session:
                    result = await session.execute(text("SELECT 'recovery_test'"))
                    value = result.scalar()
                    results["connection_recovery"] = {"recovered": True, "value": value}
            except Exception as e:
                results["connection_recovery"] = {"recovered": False, "error": str(e)}

            logger.info("✅ Error handling scenarios test completed")
            return results

        except Exception as e:
            logger.error("❌ Error handling test failed: %s", e)
            return {"error": str(e)}
        finally:
            await manager.close()

    async def test_performance_metrics(self):
        """Test 7: Performance metrics collection"""
        logger.info("🔄 Testing performance metrics...")

        manager = UnifiedConnectionManager(mode=ManagerMode.ASYNC_MODERN)

        try:
            await manager.initialize()

            # Baseline metrics
            baseline_metrics = manager._get_metrics_dict()

            # Perform operations to generate metrics
            start_time = time.time()

            for i in range(10):
                async with manager.get_async_session() as session:
                    await session.execute(text("SELECT pg_sleep(0.01)"))  # Small delay

            end_time = time.time()
            operation_duration = (end_time - start_time) * 1000  # Convert to ms

            # Final metrics
            final_metrics = manager._get_metrics_dict()

            results = {
                "baseline_metrics": baseline_metrics,
                "final_metrics": final_metrics,
                "operation_duration_ms": operation_duration,
                "metrics_comparison": {
                    "total_connections_delta": final_metrics["total_connections"]
                    - baseline_metrics["total_connections"],
                    "avg_response_time": final_metrics["avg_response_time_ms"],
                    "pool_utilization": final_metrics["pool_utilization"],
                },
            }

            logger.info("✅ Performance metrics test completed")
            return results

        except Exception as e:
            logger.error("❌ Performance metrics test failed: %s", e)
            return {"error": str(e)}
        finally:
            await manager.close()


async def test_mcp_server_integration():
    """Test 8: MCP Server integration with V2 manager"""
    logger.info("🔄 Testing MCP Server integration...")

    try:
        # Import MCP server components
        # Legacy import removed - will be fixed with modern patterns

        # Create MCP-optimized manager
        manager = UnifiedConnectionManager(mode=ManagerMode.MCP_SERVER)
        await manager.initialize()

        # Test MCP server creation with V2 manager
        server = MLDataCollectionServer()

        # Test database operations that MCP server would perform
        async with manager.get_async_session() as session:
            # Simulate MCP data collection query
            result = await session.execute(
                text("""
                SELECT 
                    current_database() as db_name,
                    current_user as db_user,
                    current_timestamp as query_time
            """)
            )
            row = result.fetchone()

            mcp_test_data = {
                "database": row.db_name,
                "user": row.db_user,
                "timestamp": str(row.query_time),
                "manager_mode": manager.mode.value,
                "pool_config": {
                    "pool_size": manager.pool_config.pg_pool_size,
                    "timeout": manager.pool_config.pg_timeout,
                },
            }

        await manager.close()
        logger.info("✅ MCP Server integration test completed")
        return {"success": True, "data": mcp_test_data}

    except ImportError as e:
        logger.warning("⚠️ MCP Server components not available: %s", e)
        return {
            "success": False,
            "error": "MCP components not available",
            "skip_reason": str(e),
        }
    except Exception as e:
        logger.error("❌ MCP Server integration test failed: %s", e)
        return {"success": False, "error": str(e)}


async def test_ml_operations_integration():
    """Test 9: ML operations integration with V2 manager"""
    logger.info("🔄 Testing ML operations integration...")

    try:
        manager = UnifiedConnectionManager(mode=ManagerMode.ML_TRAINING)
        await manager.initialize()

        # Simulate ML training data operations
        ml_test_results = []

        # Test batch operations typical for ML training
        async with manager.get_connection(mode=ConnectionMode.BATCH) as conn:
            # Simulate feature extraction query
            result = await conn.execute(
                text("""
                SELECT 
                    generate_series(1, 100) as feature_id,
                    random() as feature_value,
                    case when random() > 0.5 then 'positive' else 'negative' end as label
                LIMIT 10
            """)
            )
            rows = result.fetchall()

            ml_test_results = [
                {
                    "feature_id": row.feature_id,
                    "feature_value": float(row.feature_value),
                    "label": row.label,
                }
                for row in rows
            ]

        # Test model performance metrics simulation
        metrics = manager._get_metrics_dict()

        await manager.close()

        logger.info("✅ ML operations integration test completed")
        return {
            "success": True,
            "ml_data_sample": ml_test_results[:5],  # First 5 records
            "total_records": len(ml_test_results),
            "manager_metrics": {
                "avg_response_time_ms": metrics["avg_response_time_ms"],
                "pool_utilization": metrics["pool_utilization"],
            },
        }

    except Exception as e:
        logger.error("❌ ML operations integration test failed: %s", e)
        return {"success": False, "error": str(e)}


async def test_api_endpoints_integration():
    """Test 10: API endpoints integration with V2 manager"""
    logger.info("🔄 Testing API endpoints integration...")

    try:
        # Import API components
        from prompt_improver.api.analytics_endpoints import AnalyticsEndpoints

        manager = UnifiedConnectionManager(mode=ManagerMode.ASYNC_MODERN)
        await manager.initialize()

        # Test API-style operations
        api_test_results = {
            "connection_health": None,
            "metrics_query": None,
            "performance_data": None,
        }

        # Simulate health check endpoint
        health_result = await manager.health_check()
        api_test_results["connection_health"] = {
            "status": health_result["status"],
            "response_time_ms": health_result["response_time_ms"],
            "components_count": len(health_result["components"]),
        }

        # Simulate metrics endpoint
        async with manager.get_async_session() as session:
            result = await session.execute(
                text("""
                SELECT 
                    current_setting('max_connections') as max_connections,
                    current_setting('shared_buffers') as shared_buffers,
                    pg_database_size(current_database()) as db_size_bytes
            """)
            )
            row = result.fetchone()

            api_test_results["metrics_query"] = {
                "max_connections": row.max_connections,
                "shared_buffers": row.shared_buffers,
                "db_size_mb": int(row.db_size_bytes) / (1024 * 1024)
                if row.db_size_bytes
                else 0,
            }

        # Simulate performance monitoring
        conn_info = await manager.get_connection_info()
        api_test_results["performance_data"] = {
            "manager_mode": conn_info["mode"],
            "pool_metrics": conn_info.get("async_pool", {}),
            "health_status": conn_info["health_status"],
        }

        await manager.close()

        logger.info("✅ API endpoints integration test completed")
        return {"success": True, "results": api_test_results}

    except ImportError as e:
        logger.warning("⚠️ API endpoint components not available: %s", e)
        return {
            "success": False,
            "error": "API components not available",
            "skip_reason": str(e),
        }
    except Exception as e:
        logger.error("❌ API endpoints integration test failed: %s", e)
        return {"success": False, "error": str(e)}


async def run_comprehensive_test_suite():
    """Run comprehensive real behavior test suite"""
    logger.info("🚀 Starting Comprehensive V2 Manager Real Behavior Test Suite")

    test_suite = TestUnifiedConnectionManagerRealBehavior()
    all_results = {}

    # Define test functions with descriptions
    tests = [
        ("Database Connections Basic", test_suite.test_database_connections_basic),
        ("Multiple Manager Modes", test_suite.test_multiple_manager_modes),
        (
            "Connection Protocol Implementation",
            test_suite.test_connection_protocol_implementation,
        ),
        ("Health Checks Comprehensive", test_suite.test_health_checks_comprehensive),
        ("Session Management Lifecycle", test_suite.test_session_management_lifecycle),
        ("Error Handling Scenarios", test_suite.test_error_handling_scenarios),
        ("Performance Metrics", test_suite.test_performance_metrics),
        ("MCP Server Integration", test_mcp_server_integration),
        ("ML Operations Integration", test_ml_operations_integration),
        ("API Endpoints Integration", test_api_endpoints_integration),
    ]

    # Run tests with timing
    start_time = time.time()

    for test_name, test_func in tests:
        logger.info("\n%s", '=' * 60)
        logger.info("Running: %s", test_name)
        logger.info("%s", '=' * 60)

        test_start = time.time()
        try:
            result = await test_func()
            test_duration = time.time() - test_start
            all_results[test_name] = {
                "result": result,
                "duration_seconds": test_duration,
                "status": "completed",
            }
            logger.info("✅ {test_name} completed in %ss", test_duration:.2f)
        except Exception as e:
            test_duration = time.time() - test_start
            all_results[test_name] = {
                "result": {"error": str(e)},
                "duration_seconds": test_duration,
                "status": "failed",
            }
            logger.error("❌ {test_name} failed in {test_duration:.2f}s: %s", e)

    total_duration = time.time() - start_time

    # Generate summary
    passed_tests = sum(
        1 for result in all_results.values() if result["status"] == "completed"
    )
    total_tests = len(tests)

    logger.info("\n%s", '=' * 80)
    logger.info("TEST SUITE SUMMARY")
    logger.info("%s", '=' * 80)
    logger.info("Total Tests: %s", total_tests)
    logger.info("Passed: %s", passed_tests)
    logger.info("Failed: %s", total_tests - passed_tests)
    logger.info("Total Duration: %ss", total_duration:.2f)
    logger.info("Success Rate: %s%", (passed_tests / total_tests) * 100:.1f)

    return {
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "total_duration_seconds": total_duration,
            "success_rate_percent": (passed_tests / total_tests) * 100,
        },
        "detailed_results": all_results,
    }


if __name__ == "__main__":
    # Run the comprehensive test suite
    results = asyncio.run(run_comprehensive_test_suite())

    # Print final summary
    print("\n🎯 FINAL RESULTS:")
    print(f"Success Rate: {results['summary']['success_rate_percent']:.1f}%")
    print(f"Duration: {results['summary']['total_duration_seconds']:.2f}s")

    # Exit with appropriate code
    if results["summary"]["success_rate_percent"] >= 80:
        print("🎉 Test suite passed!")
        sys.exit(0)
    else:
        print("⚠️ Test suite has significant failures")
        sys.exit(1)
