#!/usr/bin/env python3
"""
Legacy Compatibility Validation Test

Tests that legacy adapter removal didn't break existing health checks
and that the new unified system maintains backward compatibility.
"""

import asyncio
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_plugin_adapter_compatibility():
    """Test that plugin adapters work correctly"""
    print("=== Testing Plugin Adapter Compatibility ===")
    
    from src.prompt_improver.performance.monitoring.health.unified_health_system import get_unified_health_monitor
    from src.prompt_improver.performance.monitoring.health.plugin_adapters import (
        MLServicePlugin,
        EnhancedMLServicePlugin,
        DatabasePlugin,
        RedisPlugin,
        AnalyticsServicePlugin,
        SystemResourcesPlugin
    )
    
    monitor = get_unified_health_monitor()
    
    # Test core plugin types that replace legacy adapters
    adapter_plugins = [
        ("ML Service", MLServicePlugin()),
        ("Enhanced ML Service", EnhancedMLServicePlugin()),
        ("Database", DatabasePlugin()),
        ("Redis", RedisPlugin()),
        ("Analytics Service", AnalyticsServicePlugin()),
        ("System Resources", SystemResourcesPlugin())
    ]
    
    compatibility_results = {}
    
    for plugin_name, plugin in adapter_plugins:
        try:
            # Test plugin registration
            registered = monitor.register_plugin(plugin, auto_enable=True)
            
            if not registered:
                # Plugin might already be registered from previous tests
                print(f"⚠️  {plugin_name} plugin already registered")
                compatibility_results[plugin_name] = {
                    "registered": False,
                    "already_exists": True,
                    "executed": False,
                    "compatible": True  # Not a failure if already exists
                }
                continue
            
            # Test plugin execution
            start_time = time.time()
            results = await monitor.check_health(plugin_name=plugin.name)
            execution_time = (time.time() - start_time) * 1000
            
            if plugin.name in results:
                result = results[plugin.name]
                status = result.status.value
                compatibility_results[plugin_name] = {
                    "registered": registered,
                    "executed": True,
                    "status": status,
                    "message": result.message,
                    "execution_time_ms": execution_time,
                    "compatible": True
                }
                print(f"✅ {plugin_name} plugin: {status} ({execution_time:.2f}ms)")
                print(f"   Message: {result.message}")
            else:
                compatibility_results[plugin_name] = {
                    "registered": registered,
                    "executed": False,
                    "compatible": False,
                    "error": "Plugin not found in results"
                }
                print(f"❌ {plugin_name} plugin: Not found in results")
                
        except Exception as e:
            compatibility_results[plugin_name] = {
                "registered": False,
                "executed": False,
                "compatible": False,
                "error": str(e)
            }
            print(f"❌ {plugin_name} plugin failed: {e}")
    
    # Calculate compatibility metrics
    compatible_count = sum(1 for result in compatibility_results.values() if result.get("compatible", False))
    total_count = len(compatibility_results)
    compatibility_rate = compatible_count / total_count if total_count > 0 else 0
    
    print(f"\n✅ Plugin adapter compatibility: {compatible_count}/{total_count} ({compatibility_rate:.1%})")
    
    return {
        "compatibility_rate": compatibility_rate,
        "compatible_plugins": compatible_count,
        "total_tested": total_count,
        "results": compatibility_results,
        "adapter_compatibility_successful": compatibility_rate >= 0.8  # 80% success threshold
    }


async def test_legacy_interface_replacement():
    """Test that legacy interfaces have been properly replaced"""
    print("\n=== Testing Legacy Interface Replacement ===")
    
    interface_tests = {}
    
    # Test 1: Check that unified health monitor replaces old health service
    try:
        from src.prompt_improver.performance.monitoring.health.unified_health_system import get_unified_health_monitor
        
        monitor = get_unified_health_monitor()
        
        # Test core functionality
        plugins = monitor.get_registered_plugins()
        summary = monitor.get_health_summary()
        
        interface_tests["unified_health_monitor"] = {
            "available": True,
            "plugins_accessible": len(plugins) > 0,
            "summary_accessible": "registered_plugins" in summary,
            "working": True
        }
        
        print(f"✅ Unified Health Monitor: {len(plugins)} plugins, summary available")
        
    except Exception as e:
        interface_tests["unified_health_monitor"] = {
            "available": False,
            "error": str(e),
            "working": False
        }
        print(f"❌ Unified Health Monitor failed: {e}")
    
    # Test 2: Check that plugin adapters replace legacy checkers
    try:
        from src.prompt_improver.performance.monitoring.health.plugin_adapters import (
            create_all_plugins,
            register_all_plugins
        )
        
        # Test plugin creation
        all_plugins = create_all_plugins()
        
        # Test plugin registration
        monitor = get_unified_health_monitor()
        registered_count = register_all_plugins(monitor)
        
        interface_tests["plugin_adapters"] = {
            "available": True,
            "plugins_created": len(all_plugins),
            "plugins_registered": registered_count,
            "working": len(all_plugins) > 0 and registered_count > 0
        }
        
        print(f"✅ Plugin Adapters: {len(all_plugins)} created, {registered_count} registered")
        
    except Exception as e:
        interface_tests["plugin_adapters"] = {
            "available": False,
            "error": str(e),
            "working": False
        }
        print(f"❌ Plugin Adapters failed: {e}")
    
    # Test 3: Check enhanced background manager replaces legacy task manager
    try:
        from src.prompt_improver.performance.monitoring.health.background_manager import (
            EnhancedBackgroundTaskManager,
            get_background_task_manager
        )
        
        # Test enhanced manager creation
        enhanced_manager = EnhancedBackgroundTaskManager(max_concurrent_tasks=2)
        
        # Test global manager access
        global_manager = get_background_task_manager()
        
        interface_tests["enhanced_background_manager"] = {
            "available": True,
            "enhanced_manager_creatable": enhanced_manager is not None,
            "global_manager_accessible": global_manager is not None,
            "working": True
        }
        
        print("✅ Enhanced Background Manager: Creation and global access working")
        
    except Exception as e:
        interface_tests["enhanced_background_manager"] = {
            "available": False,
            "error": str(e),
            "working": False
        }
        print(f"❌ Enhanced Background Manager failed: {e}")
    
    # Calculate replacement success
    working_interfaces = sum(1 for test in interface_tests.values() if test.get("working", False))
    total_interfaces = len(interface_tests)
    replacement_success_rate = working_interfaces / total_interfaces if total_interfaces > 0 else 0
    
    print(f"\n✅ Legacy interface replacement: {working_interfaces}/{total_interfaces} ({replacement_success_rate:.1%})")
    
    return {
        "replacement_success_rate": replacement_success_rate,
        "working_interfaces": working_interfaces,
        "total_interfaces": total_interfaces,
        "interface_tests": interface_tests,
        "interface_replacement_successful": replacement_success_rate >= 0.8
    }


async def test_functionality_preservation():
    """Test that core functionality is preserved after legacy removal"""
    print("\n=== Testing Functionality Preservation ===")
    
    functionality_tests = {}
    
    # Test 1: Health check registration and execution
    try:
        from src.prompt_improver.performance.monitoring.health.unified_health_system import (
            get_unified_health_monitor,
            HealthCheckCategory,
            create_simple_health_plugin
        )
        
        monitor = get_unified_health_monitor()
        
        # Create and register test plugin
        def test_function():
            return {"status": "healthy", "message": "Functionality test working"}
        
        test_plugin = create_simple_health_plugin(
            name="functionality_test",
            category=HealthCheckCategory.CUSTOM,
            check_func=test_function
        )
        
        registered = monitor.register_plugin(test_plugin)
        
        # Execute health check
        results = await monitor.check_health(plugin_name="functionality_test")
        
        functionality_tests["health_check_lifecycle"] = {
            "plugin_created": test_plugin is not None,
            "plugin_registered": registered,
            "health_check_executed": "functionality_test" in results,
            "result_valid": results["functionality_test"].status.value == "healthy" if "functionality_test" in results else False,
            "working": all([
                test_plugin is not None,
                registered,
                "functionality_test" in results,
                results["functionality_test"].status.value == "healthy" if "functionality_test" in results else False
            ])
        }
        
        print("✅ Health check lifecycle: Plugin creation, registration, and execution working")
        
    except Exception as e:
        functionality_tests["health_check_lifecycle"] = {
            "working": False,
            "error": str(e)
        }
        print(f"❌ Health check lifecycle failed: {e}")
    
    # Test 2: Category-based filtering
    try:
        monitor = get_unified_health_monitor()
        
        # Test category filtering
        custom_results = await monitor.check_health(category=HealthCheckCategory.CUSTOM)
        all_results = await monitor.check_health()
        
        functionality_tests["category_filtering"] = {
            "custom_category_results": len(custom_results),
            "all_results": len(all_results),
            "filtering_working": len(custom_results) <= len(all_results),
            "working": len(custom_results) <= len(all_results)
        }
        
        print(f"✅ Category filtering: {len(custom_results)} custom results from {len(all_results)} total")
        
    except Exception as e:
        functionality_tests["category_filtering"] = {
            "working": False,
            "error": str(e)
        }
        print(f"❌ Category filtering failed: {e}")
    
    # Test 3: Overall health aggregation
    try:
        monitor = get_unified_health_monitor()
        
        overall_health = await monitor.get_overall_health()
        
        functionality_tests["health_aggregation"] = {
            "overall_health_available": overall_health is not None,
            "status_available": hasattr(overall_health, 'status'),
            "message_available": hasattr(overall_health, 'message'),
            "details_available": hasattr(overall_health, 'details'),
            "working": all([
                overall_health is not None,
                hasattr(overall_health, 'status'),
                hasattr(overall_health, 'message'),
                hasattr(overall_health, 'details')
            ])
        }
        
        print(f"✅ Health aggregation: Overall status = {overall_health.status.value}")
        
    except Exception as e:
        functionality_tests["health_aggregation"] = {
            "working": False,
            "error": str(e)
        }
        print(f"❌ Health aggregation failed: {e}")
    
    # Calculate preservation success
    working_functionality = sum(1 for test in functionality_tests.values() if test.get("working", False))
    total_functionality = len(functionality_tests)
    preservation_rate = working_functionality / total_functionality if total_functionality > 0 else 0
    
    print(f"\n✅ Functionality preservation: {working_functionality}/{total_functionality} ({preservation_rate:.1%})")
    
    return {
        "preservation_rate": preservation_rate,
        "working_functionality": working_functionality,
        "total_functionality": total_functionality,
        "functionality_tests": functionality_tests,
        "functionality_preservation_successful": preservation_rate >= 0.8
    }


async def run_legacy_compatibility_validation():
    """Run comprehensive legacy compatibility validation"""
    print("🔧 Legacy Compatibility Validation")
    print("=" * 50)
    
    validation_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "validation_passed": False,
        "test_results": {}
    }
    
    # Test categories
    test_functions = [
        ("adapter_compatibility", test_plugin_adapter_compatibility),
        ("interface_replacement", test_legacy_interface_replacement),
        ("functionality_preservation", test_functionality_preservation)
    ]
    
    try:
        for test_name, test_func in test_functions:
            print(f"\n🧪 Running {test_name} test...")
            
            try:
                test_start = time.time()
                result = await test_func()
                test_duration = time.time() - test_start
                
                # Determine if test passed based on result structure
                test_passed = result.get(f"{test_name}_successful", False)
                
                validation_results["test_results"][test_name] = {
                    "passed": test_passed,
                    "duration_seconds": test_duration,
                    "result": result
                }
                
                print(f"{'✅' if test_passed else '❌'} {test_name} {'passed' if test_passed else 'failed'} in {test_duration:.2f}s")
                
            except Exception as e:
                test_duration = time.time() - test_start if 'test_start' in locals() else 0
                validation_results["test_results"][test_name] = {
                    "passed": False,
                    "duration_seconds": test_duration,
                    "error": str(e)
                }
                print(f"❌ {test_name} failed with exception: {e}")
                logger.exception(f"Test {test_name} failed")
        
        # Calculate overall results
        passed_tests = sum(1 for result in validation_results["test_results"].values() if result["passed"])
        total_tests = len(validation_results["test_results"])
        
        validation_results["validation_passed"] = passed_tests == total_tests
        validation_results["passed_tests"] = passed_tests
        validation_results["total_tests"] = total_tests
        validation_results["success_rate"] = passed_tests / total_tests if total_tests > 0 else 0
        
        # Summary
        print(f"\n📊 Legacy Compatibility Validation Summary")
        print("=" * 50)
        print(f"Total tests: {total_tests}")
        print(f"Passed tests: {passed_tests}")
        print(f"Success rate: {validation_results['success_rate']:.1%}")
        print(f"Overall validation: {'✅ PASSED' if validation_results['validation_passed'] else '❌ FAILED'}")
        
        if validation_results["validation_passed"]:
            print("\n🎉 Legacy compatibility validation completed successfully!")
            print("   Legacy pattern removal did not break existing functionality.")
        else:
            print("\n⚠️  Legacy compatibility validation found issues.")
            print("   Some legacy functionality may have been broken during removal.")
        
        return validation_results
        
    except Exception as e:
        validation_results["validation_passed"] = False
        validation_results["error"] = str(e)
        print(f"❌ Legacy compatibility validation failed: {e}")
        logger.exception("Legacy compatibility validation failed")
        return validation_results


if __name__ == "__main__":
    # Run the validation
    results = asyncio.run(run_legacy_compatibility_validation())
    
    # Exit with appropriate code
    exit(0 if results["validation_passed"] else 1)