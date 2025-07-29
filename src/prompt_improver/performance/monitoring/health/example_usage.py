"""
Usage Examples for Unified Health System

Demonstrates how to use the plugin-based health monitoring system:
- Basic plugin registration and usage
- Environment-specific configuration
- Custom health check plugins
- Performance monitoring and validation
"""

import asyncio
import time
from typing import Dict, Any

from .unified_health_system import (
    HealthCheckPlugin,
    HealthCheckCategory,
    HealthCheckPluginConfig,
    get_unified_health_monitor,
    create_simple_health_plugin
)
from .health_config import (
    get_health_config,
    EnvironmentType,
    create_category_config
)
from .plugin_adapters import (
    create_all_plugins,
    register_all_plugins
)
from ....core.protocols.health_protocol import HealthCheckResult, HealthStatus


async def basic_usage_example():
    """Basic usage of the unified health system"""
    print("=== Basic Usage Example ===")
    
    # Get the unified health monitor
    monitor = get_unified_health_monitor()
    
    # Create a simple health check plugin
    def simple_check():
        return {"status": "healthy", "message": "Service is running"}
    
    simple_plugin = create_simple_health_plugin(
        name="simple_service",
        category=HealthCheckCategory.CUSTOM,
        check_func=simple_check,
        config=HealthCheckPluginConfig(timeout_seconds=5.0)
    )
    
    # Register the plugin
    monitor.register_plugin(simple_plugin)
    
    # Perform health check
    results = await monitor.check_health()
    print(f"Health check results: {results}")
    
    # Get overall health
    overall_health = await monitor.get_overall_health()
    print(f"Overall health: {overall_health.status.value} - {overall_health.message}")
    
    print()


async def environment_configuration_example():
    """Example of environment-specific configuration"""
    print("=== Environment Configuration Example ===")
    
    # Get configuration for current environment
    config = get_health_config()
    print(f"Current environment: {config.environment.value}")
    
    # Get monitoring policy
    policy = config.get_policy()
    print(f"Global timeout: {policy.global_timeout_seconds}s")
    print(f"Parallel execution: {policy.parallel_execution}")
    print(f"Max concurrent checks: {policy.max_concurrent_checks}")
    
    # Get category-specific thresholds
    ml_thresholds = config.get_category_thresholds(HealthCheckCategory.ML)
    print(f"ML category timeout: {ml_thresholds.timeout_seconds}s")
    print(f"ML category retry count: {ml_thresholds.retry_count}")
    
    # Get available profiles
    profiles = config.get_available_profiles()
    print(f"Available profiles: {profiles}")
    
    # Get default profile
    default_profile = config.get_profile("default")
    if default_profile:
        print(f"Default profile enabled plugins: {len(default_profile.enabled_plugins)}")
    
    print()


async def custom_plugin_example():
    """Example of creating custom health check plugins"""
    print("=== Custom Plugin Example ===")
    
    class CustomDatabasePlugin(HealthCheckPlugin):
        """Custom database health check with advanced logic"""
        
        def __init__(self):
            config = create_category_config(
                HealthCheckCategory.DATABASE,
                critical=True,
                timeout_seconds=8.0
            )
            super().__init__(
                name="custom_database",
                category=HealthCheckCategory.DATABASE,
                config=config
            )
            
        async def execute_check(self) -> HealthCheckResult:
            # Simulate database check
            start_time = time.time()
            await asyncio.sleep(0.1)  # Simulate database query
            duration = (time.time() - start_time) * 1000
            
            # Check connection and performance
            if duration > 100:
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message=f"Database slow: {duration:.1f}ms",
                    details={"duration_ms": duration, "threshold_ms": 100},
                    check_name=self.name
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message=f"Database healthy: {duration:.1f}ms",
                    details={"duration_ms": duration},
                    check_name=self.name
                )
    
    monitor = get_unified_health_monitor()
    
    # Register custom plugin
    custom_plugin = CustomDatabasePlugin()
    monitor.register_plugin(custom_plugin)
    
    # Check the custom plugin
    results = await monitor.check_health(plugin_name="custom_database")
    print(f"Custom plugin result: {results}")
    
    print()


async def performance_validation_example():
    """Example of performance monitoring and validation"""
    print("=== Performance Validation Example ===")
    
    monitor = get_unified_health_monitor()
    config = get_health_config()
    
    # Register a few test plugins
    test_plugins = []
    for i in range(5):
        def quick_check():
            return True
            
        plugin = create_simple_health_plugin(
            name=f"test_plugin_{i}",
            category=HealthCheckCategory.SYSTEM,
            check_func=quick_check,
            config=HealthCheckPluginConfig(timeout_seconds=1.0)
        )
        
        test_plugins.append(plugin)
        monitor.register_plugin(plugin)
    
    # Measure performance
    start_time = time.time()
    results = await monitor.check_health()
    total_duration = (time.time() - start_time) * 1000
    
    print(f"Total health check duration: {total_duration:.2f}ms")
    print(f"Number of checks: {len(results)}")
    print(f"Average per check: {total_duration / max(len(results), 1):.2f}ms")
    
    # Performance optimization settings
    perf_settings = config.optimize_for_performance()
    print(f"Performance target: {perf_settings['performance_target_ms']}ms per check")
    print(f"Batch size: {perf_settings['batch_size']}")
    print(f"Cache enabled: {perf_settings['cache_results']}")
    
    # Validate performance target
    avg_duration = total_duration / max(len(results), 1)
    target_ms = perf_settings['performance_target_ms']
    
    if avg_duration <= target_ms:
        print(f"✅ Performance target met: {avg_duration:.2f}ms <= {target_ms}ms")
    else:
        print(f"❌ Performance target missed: {avg_duration:.2f}ms > {target_ms}ms")
    
    print()


async def full_system_example():
    """Example of registering all available plugins"""
    print("=== Full System Example ===")
    
    monitor = get_unified_health_monitor()
    
    # Register all available plugins
    registered_count = register_all_plugins(monitor)
    print(f"Registered {registered_count} health check plugins")
    
    # Get health summary
    summary = monitor.get_health_summary()
    print(f"Total registered plugins: {summary['registered_plugins']}")
    print(f"Plugins by category: {summary['categories']}")
    print(f"Active profile: {summary['active_profile']}")
    
    # Switch to critical profile for faster checks
    config = get_health_config()
    critical_profile = config.get_profile("critical")
    if critical_profile:
        monitor.activate_profile("critical")
        print("Switched to critical profile")
        
        # Run critical checks only
        results = await monitor.check_health()
        print(f"Critical health checks: {len(results)} completed")
        
        # Get overall health
        overall = await monitor.get_overall_health()
        print(f"Overall system health: {overall.status.value}")
    
    print()


async def profile_switching_example():
    """Example of switching between health check profiles"""
    print("=== Profile Switching Example ===")
    
    monitor = get_unified_health_monitor()
    config = get_health_config()
    
    # Register some plugins
    register_all_plugins(monitor)
    
    # Test different profiles
    profiles_to_test = ["minimal", "critical", "full"]
    
    for profile_name in profiles_to_test:
        profile = config.get_profile(profile_name)
        if not profile:
            continue
            
        monitor.activate_profile(profile_name)
        print(f"\n--- Testing {profile_name} profile ---")
        print(f"Enabled plugins: {len(profile.enabled_plugins)}")
        print(f"Global timeout: {profile.global_timeout}s")
        print(f"Critical only: {profile.critical_only}")
        
        # Run health checks
        start_time = time.time()
        results = await monitor.check_health()
        duration = (time.time() - start_time) * 1000
        
        print(f"Health checks completed: {len(results)} in {duration:.2f}ms")
        
        # Show results summary
        healthy_count = sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED)
        unhealthy_count = sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY)
        
        print(f"Results: {healthy_count} healthy, {degraded_count} degraded, {unhealthy_count} unhealthy")
    
    print()




async def main():
    """Run all examples"""
    print("Unified Health System Usage Examples")
    print("=" * 50)
    
    try:
        await basic_usage_example()
        await environment_configuration_example()
        await custom_plugin_example()
        await performance_validation_example()
        await full_system_example()
        await profile_switching_example()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())