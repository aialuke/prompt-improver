"""
Real Behavior Health Monitoring Scenarios Test

Tests real-world health monitoring scenarios to ensure the system
works correctly in production-like conditions after legacy removal.
"""

import asyncio
import logging
import time
from datetime import UTC, datetime, timezone
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_system_startup_scenario():
    """Test system startup health validation scenario"""
    print("=== Testing System Startup Scenario ===")
    from src.prompt_improver.performance.monitoring.health.plugin_adapters import (
        register_all_plugins,
    )
    from src.prompt_improver.performance.monitoring.health.unified_health_system import (
        get_unified_health_monitor,
    )

    monitor = get_unified_health_monitor()
    print("🚀 Simulating system startup...")
    start_time = time.time()
    registered_count = register_all_plugins(monitor)
    registration_time = (time.time() - start_time) * 1000
    print(
        f"✅ Plugin registration: {registered_count} plugins in {registration_time:.2f}ms"
    )
    startup_check_start = time.time()
    startup_results = await monitor.check_health()
    startup_check_time = (time.time() - startup_check_start) * 1000
    print(
        f"✅ Startup health check: {len(startup_results)} checks in {startup_check_time:.2f}ms"
    )
    critical_plugins = [
        name for name, plugin in monitor._plugins.items() if plugin.config.critical
    ]
    critical_results = {
        name: result
        for name, result in startup_results.items()
        if name in critical_plugins
    }
    critical_failures = [
        name
        for name, result in critical_results.items()
        if result.status.value == "unhealthy"
    ]
    startup_viable = len(critical_failures) == 0
    performance_acceptable = startup_check_time < 2000
    print(
        f"   Critical systems: {len(critical_results)} checks, {len(critical_failures)} failures"
    )
    print(f"   System startup viable: {startup_viable}")
    print(f"   Performance acceptable: {performance_acceptable}")
    return {
        "plugins_registered": registered_count,
        "registration_time_ms": registration_time,
        "startup_checks": len(startup_results),
        "startup_check_time_ms": startup_check_time,
        "critical_systems": len(critical_results),
        "critical_failures": len(critical_failures),
        "startup_viable": startup_viable,
        "performance_acceptable": performance_acceptable,
        "startup_scenario_successful": startup_viable and performance_acceptable,
    }


async def test_continuous_monitoring_scenario():
    """Test continuous monitoring scenario"""
    print("\n=== Testing Continuous Monitoring Scenario ===")
    from src.prompt_improver.performance.monitoring.health.unified_health_system import (
        get_unified_health_monitor,
    )

    monitor = get_unified_health_monitor()
    print("⏱️  Simulating continuous monitoring (5 cycles)...")
    monitoring_cycles = []
    for cycle in range(5):
        cycle_start = time.time()
        cycle_results = await monitor.check_health()
        overall_health = await monitor.get_overall_health()
        cycle_time = (time.time() - cycle_start) * 1000
        cycle_data = {
            "cycle": cycle + 1,
            "checks_executed": len(cycle_results),
            "execution_time_ms": cycle_time,
            "overall_status": overall_health.status.value,
            "healthy_count": sum(
                1 for r in cycle_results.values() if r.status.value == "healthy"
            ),
            "degraded_count": sum(
                1 for r in cycle_results.values() if r.status.value == "degraded"
            ),
            "unhealthy_count": sum(
                1 for r in cycle_results.values() if r.status.value == "unhealthy"
            ),
        }
        monitoring_cycles.append(cycle_data)
        print(
            f"   Cycle {cycle + 1}: {len(cycle_results)} checks, {cycle_time:.2f}ms, {overall_health.status.value}"
        )
        await asyncio.sleep(0.2)
    avg_cycle_time = sum(c["execution_time_ms"] for c in monitoring_cycles) / len(
        monitoring_cycles
    )
    max_cycle_time = max(c["execution_time_ms"] for c in monitoring_cycles)
    min_cycle_time = min(c["execution_time_ms"] for c in monitoring_cycles)
    time_variance = (
        (max_cycle_time - min_cycle_time) / avg_cycle_time if avg_cycle_time > 0 else 0
    )
    consistent_performance = time_variance < 2.0
    print("✅ Monitoring consistency:")
    print(f"   Average cycle time: {avg_cycle_time:.2f}ms")
    print(f"   Time variance: {time_variance:.1%}")
    print(f"   Consistent performance: {consistent_performance}")
    return {
        "monitoring_cycles": len(monitoring_cycles),
        "average_cycle_time_ms": avg_cycle_time,
        "min_cycle_time_ms": min_cycle_time,
        "max_cycle_time_ms": max_cycle_time,
        "time_variance": time_variance,
        "consistent_performance": consistent_performance,
        "cycle_results": monitoring_cycles,
        "continuous_monitoring_successful": consistent_performance
        and avg_cycle_time < 1000,
    }


async def test_profile_switching_scenario():
    """Test profile switching under load scenario"""
    print("\n=== Testing Profile Switching Scenario ===")
    from src.prompt_improver.monitoring.unified_monitoring_manager import (
        get_health_config,
    )
    from src.prompt_improver.performance.monitoring.health.unified_health_system import (
        get_unified_health_monitor,
    )

    monitor = get_unified_health_monitor()
    config = get_health_config()
    profiles_to_test = ["minimal", "critical", "full"]
    profile_results = {}
    for profile_name in profiles_to_test:
        profile = config.get_profile(profile_name)
        if not profile:
            print(f"⚠️  Profile {profile_name} not available")
            continue
        print(f"🔄 Testing {profile_name} profile...")
        monitor.activate_profile(profile_name)
        profile_start = time.time()
        load_results = []
        for load_cycle in range(3):
            load_start = time.time()
            results = await monitor.check_health()
            load_time = (time.time() - load_start) * 1000
            load_results.append({
                "load_cycle": load_cycle + 1,
                "checks": len(results),
                "time_ms": load_time,
            })
        total_profile_time = (time.time() - profile_start) * 1000
        avg_load_time = sum(r["time_ms"] for r in load_results) / len(load_results)
        total_checks = sum(r["checks"] for r in load_results)
        profile_results[profile_name] = {
            "enabled_plugins": len(profile.enabled_plugins),
            "load_cycles": len(load_results),
            "total_checks": total_checks,
            "total_time_ms": total_profile_time,
            "average_load_time_ms": avg_load_time,
            "load_handling_acceptable": total_profile_time < 3000,
            "load_results": load_results,
        }
        print(
            f"   Profile {profile_name}: {total_checks} total checks, {total_profile_time:.2f}ms total"
        )
        print(
            f"   Load handling: {('✅ acceptable' if profile_results[profile_name]['load_handling_acceptable'] else '❌ slow')}"
        )
    all_profiles_responsive = all(
        result["load_handling_acceptable"] for result in profile_results.values()
    )
    return {
        "profiles_tested": len(profile_results),
        "profile_results": profile_results,
        "all_profiles_responsive": all_profiles_responsive,
        "profile_switching_successful": all_profiles_responsive
        and len(profile_results) > 0,
    }


async def test_error_recovery_scenario():
    """Test error recovery and resilience scenario"""
    print("\n=== Testing Error Recovery Scenario ===")
    from src.prompt_improver.core.protocols.health_protocol import HealthStatus
    from src.prompt_improver.performance.monitoring.health.unified_health_system import (
        HealthCheckCategory,
        HealthCheckPluginConfig,
        create_simple_health_plugin,
        get_unified_health_monitor,
    )

    monitor = get_unified_health_monitor()
    failure_count = 0

    def intermittent_failure():
        nonlocal failure_count
        failure_count += 1
        if failure_count % 3 == 0:
            raise Exception("Intermittent failure")
        return {"status": "healthy", "message": "Working normally"}

    async def timeout_simulation():
        import asyncio

        await asyncio.sleep(2.0)
        return True

    def recovery_check():
        return {"status": "healthy", "message": "System recovered"}

    error_plugins = [
        ("intermittent_failure", intermittent_failure, 5.0),
        ("timeout_test", timeout_simulation, 1.0),
        ("recovery_test", recovery_check, 5.0),
    ]
    for name, check_func, timeout in error_plugins:
        plugin = create_simple_health_plugin(
            name=name,
            category=HealthCheckCategory.CUSTOM,
            check_func=check_func,
            config=HealthCheckPluginConfig(timeout_seconds=timeout, retry_count=1),
        )
        monitor.register_plugin(plugin)
        print(f"✅ Registered error test plugin: {name}")
    recovery_cycles = []
    for cycle in range(4):
        cycle_start = time.time()
        try:
            cycle_results = await monitor.check_health()
            healthy_count = sum(
                1 for r in cycle_results.values() if r.status.value == "healthy"
            )
            error_count = sum(
                1 for r in cycle_results.values() if r.status.value == "unhealthy"
            )
            cycle_time = (time.time() - cycle_start) * 1000
            recovery_cycles.append({
                "cycle": cycle + 1,
                "execution_time_ms": cycle_time,
                "total_checks": len(cycle_results),
                "healthy_checks": healthy_count,
                "error_checks": error_count,
                "recovery_rate": healthy_count / len(cycle_results)
                if cycle_results
                else 0,
            })
            print(
                f"   Recovery cycle {cycle + 1}: {healthy_count}/{len(cycle_results)} healthy ({cycle_time:.2f}ms)"
            )
        except Exception as e:
            print(f"   Recovery cycle {cycle + 1} failed: {e}")
            recovery_cycles.append({
                "cycle": cycle + 1,
                "execution_time_ms": 0,
                "total_checks": 0,
                "healthy_checks": 0,
                "error_checks": 1,
                "recovery_rate": 0,
                "error": str(e),
            })
        await asyncio.sleep(0.5)
    avg_recovery_rate = sum(c["recovery_rate"] for c in recovery_cycles) / len(
        recovery_cycles
    )
    system_resilient = avg_recovery_rate > 0.6
    print("✅ Error recovery analysis:")
    print(f"   Average recovery rate: {avg_recovery_rate:.1%}")
    print(f"   System resilient: {system_resilient}")
    return {
        "recovery_cycles": len(recovery_cycles),
        "average_recovery_rate": avg_recovery_rate,
        "system_resilient": system_resilient,
        "cycle_results": recovery_cycles,
        "error_recovery_successful": system_resilient,
    }


async def test_integration_scenario():
    """Test integration with other system components"""
    print("\n=== Testing Integration Scenario ===")
    from src.prompt_improver.performance.monitoring.health.background_manager import (
        get_background_task_manager,
    )
    from src.prompt_improver.performance.monitoring.health.unified_health_system import (
        get_unified_health_monitor,
    )

    monitor = get_unified_health_monitor()
    integration_tests = {}
    try:
        task_manager = get_background_task_manager()

        async def health_monitoring_task():
            results = await monitor.check_health()
            return f"Monitored {len(results)} health checks"

        task_id = await task_manager.submit_enhanced_task(
            task_id="health_monitoring_integration", coroutine=health_monitoring_task
        )
        await asyncio.sleep(1.0)
        task_status = task_manager.get_enhanced_task_status(task_id)
        task_completed = task_status and task_status["status"] == "completed"
        integration_tests["background_manager"] = {
            "task_submitted": task_id is not None,
            "task_completed": task_completed,
            "result": task_status.get("result") if task_status else None,
            "working": task_completed,
        }
        print(
            f"✅ Background manager integration: {('working' if task_completed else 'failed')}"
        )
    except Exception as e:
        integration_tests["background_manager"] = {"working": False, "error": str(e)}
        print(f"❌ Background manager integration failed: {e}")
    try:
        from src.prompt_improver.performance.monitoring.metrics_registry import (
            get_metrics_registry,
        )

        metrics_registry = get_metrics_registry()
        before_check = time.time()
        health_results = await monitor.check_health()
        after_check = time.time()
        metrics_available = metrics_registry is not None
        health_executed = len(health_results) > 0
        timing_reasonable = after_check - before_check < 5.0
        integration_tests["metrics_registry"] = {
            "metrics_registry_available": metrics_available,
            "health_checks_executed": health_executed,
            "timing_reasonable": timing_reasonable,
            "working": all([metrics_available, health_executed, timing_reasonable]),
        }
        print(
            f"✅ Metrics registry integration: {('working' if integration_tests['metrics_registry']['working'] else 'failed')}"
        )
    except Exception as e:
        integration_tests["metrics_registry"] = {"working": False, "error": str(e)}
        print(f"❌ Metrics registry integration failed: {e}")
    try:
        health_summary = monitor.get_health_summary()
        overall_health = await monitor.get_overall_health()
        summary_complete = all([
            "registered_plugins" in health_summary,
            "enabled_plugins" in health_summary,
            "categories" in health_summary,
        ])
        overall_health_valid = all([
            hasattr(overall_health, "status"),
            hasattr(overall_health, "message"),
            hasattr(overall_health, "details"),
        ])
        integration_tests["health_reporting"] = {
            "summary_complete": summary_complete,
            "overall_health_valid": overall_health_valid,
            "working": summary_complete and overall_health_valid,
        }
        print(
            f"✅ Health reporting integration: {('working' if integration_tests['health_reporting']['working'] else 'failed')}"
        )
    except Exception as e:
        integration_tests["health_reporting"] = {"working": False, "error": str(e)}
        print(f"❌ Health reporting integration failed: {e}")
    working_integrations = sum(
        1 for test in integration_tests.values() if test.get("working", False)
    )
    total_integrations = len(integration_tests)
    integration_success_rate = (
        working_integrations / total_integrations if total_integrations > 0 else 0
    )
    return {
        "integrations_tested": total_integrations,
        "working_integrations": working_integrations,
        "integration_success_rate": integration_success_rate,
        "integration_tests": integration_tests,
        "integration_scenario_successful": integration_success_rate >= 0.67,
    }


async def run_real_behavior_validation():
    """Run comprehensive real behavior scenarios validation"""
    print("🔧 Real Behavior Health Monitoring Scenarios")
    print("=" * 60)
    validation_results = {
        "timestamp": datetime.now(UTC).isoformat(),
        "validation_passed": False,
        "test_results": {},
    }
    test_functions = [
        ("startup_scenario", test_system_startup_scenario),
        ("continuous_monitoring", test_continuous_monitoring_scenario),
        ("profile_switching", test_profile_switching_scenario),
        ("error_recovery", test_error_recovery_scenario),
        ("integration", test_integration_scenario),
    ]
    try:
        for test_name, test_func in test_functions:
            print(f"\n🧪 Running {test_name} scenario...")
            try:
                test_start = time.time()
                result = await test_func()
                test_duration = time.time() - test_start
                test_passed = result.get(f"{test_name}_successful", False)
                validation_results["test_results"][test_name] = {
                    "passed": test_passed,
                    "duration_seconds": test_duration,
                    "result": result,
                }
                print(
                    f"{('✅' if test_passed else '❌')} {test_name} {('passed' if test_passed else 'failed')} in {test_duration:.2f}s"
                )
            except Exception as e:
                test_duration = (
                    time.time() - test_start if "test_start" in locals() else 0
                )
                validation_results["test_results"][test_name] = {
                    "passed": False,
                    "duration_seconds": test_duration,
                    "error": str(e),
                }
                print(f"❌ {test_name} failed with exception: {e}")
                logger.exception(f"Scenario {test_name} failed")
        passed_tests = sum(
            1
            for result in validation_results["test_results"].values()
            if result["passed"]
        )
        total_tests = len(validation_results["test_results"])
        validation_results["validation_passed"] = passed_tests == total_tests
        validation_results["passed_tests"] = passed_tests
        validation_results["total_tests"] = total_tests
        validation_results["success_rate"] = (
            passed_tests / total_tests if total_tests > 0 else 0
        )
        print("\n📊 Real Behavior Scenarios Validation Summary")
        print("=" * 60)
        print(f"Total scenarios: {total_tests}")
        print(f"Passed scenarios: {passed_tests}")
        print(f"Success rate: {validation_results['success_rate']:.1%}")
        print(
            f"Overall validation: {('✅ PASSED' if validation_results['validation_passed'] else '❌ FAILED')}"
        )
        if validation_results["validation_passed"]:
            print("\n🎉 Real behavior scenarios validation completed successfully!")
            print(
                "   Health monitoring system works correctly in production-like conditions."
            )
        else:
            print("\n⚠️  Real behavior scenarios validation found issues.")
            print("   Some production scenarios may not be working correctly.")
        return validation_results
    except Exception as e:
        validation_results["validation_passed"] = False
        validation_results["error"] = str(e)
        print(f"❌ Real behavior validation failed: {e}")
        logger.exception("Real behavior validation failed")
        return validation_results


if __name__ == "__main__":
    results = asyncio.run(run_real_behavior_validation())
    exit(0 if results["validation_passed"] else 1)
