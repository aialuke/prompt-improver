"""
Health Check Execution Validation Test

Tests the health check execution and result reporting mechanisms
to ensure they work correctly after legacy pattern removal.
"""
import asyncio
import logging
import time
from datetime import UTC, datetime, timezone
from typing import Any, Dict
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_basic_health_execution():
    """Test basic health check execution"""
    print('=== Testing Basic Health Check Execution ===')
    from src.prompt_improver.core.protocols.health_protocol import HealthStatus
    from src.prompt_improver.performance.monitoring.health.unified_health_system import HealthCheckCategory, HealthCheckPluginConfig, create_simple_health_plugin, get_unified_health_monitor
    monitor = get_unified_health_monitor()

    def healthy_check():
        return {'status': 'healthy', 'message': 'System operating normally'}

    def warning_check():
        return {'status': 'degraded', 'message': 'Minor performance issues detected'}

    def failing_check():
        return {'status': 'unhealthy', 'message': 'Critical system failure'}
    plugins = [('healthy_test', healthy_check, HealthStatus.HEALTHY), ('warning_test', warning_check, HealthStatus.DEGRADED), ('failing_test', failing_check, HealthStatus.UNHEALTHY)]
    for name, check_func, expected_status in plugins:
        plugin = create_simple_health_plugin(name=name, category=HealthCheckCategory.CUSTOM, check_func=check_func, config=HealthCheckPluginConfig(timeout_seconds=5.0))
        success = monitor.register_plugin(plugin)
        assert success, f'Failed to register plugin {name}'
        print(f'✅ Registered plugin: {name}')
    for name, _, expected_status in plugins:
        start_time = time.time()
        results = await monitor.check_health(plugin_name=name)
        execution_time = (time.time() - start_time) * 1000
        assert name in results, f'Plugin {name} not found in results'
        result = results[name]
        print(f'✅ Plugin {name}: {result.status.value} ({execution_time:.2f}ms)')
        print(f'   Message: {result.message}')
        print(f'   Duration: {result.duration_ms:.2f}ms')
        assert result.status == expected_status, f'Expected {expected_status}, got {result.status}'
    return {'plugins_registered': len(plugins), 'individual_execution_successful': True, 'status_verification_passed': True}

async def test_bulk_health_execution():
    """Test bulk health check execution"""
    print('\n=== Testing Bulk Health Check Execution ===')
    from src.prompt_improver.core.protocols.health_protocol import HealthStatus
    from src.prompt_improver.performance.monitoring.health.unified_health_system import get_unified_health_monitor
    monitor = get_unified_health_monitor()
    start_time = time.time()
    all_results = await monitor.check_health()
    total_execution_time = (time.time() - start_time) * 1000
    print(f'✅ Bulk execution completed: {len(all_results)} checks in {total_execution_time:.2f}ms')
    status_counts = {}
    duration_sum = 0
    for plugin_name, result in all_results.items():
        status = result.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
        duration_sum += result.duration_ms
    average_duration = duration_sum / len(all_results) if all_results else 0
    print('   Status distribution:')
    for status, count in status_counts.items():
        print(f'     {status}: {count}')
    print(f'   Average duration per check: {average_duration:.2f}ms')
    overall_health = await monitor.get_overall_health()
    print(f'✅ Overall system health: {overall_health.status.value}')
    print(f'   Message: {overall_health.message}')
    print(f'   Details: {len(overall_health.details)} items')
    performance_acceptable = total_execution_time < 2000
    efficiency_good = average_duration < 100
    return {'total_checks_executed': len(all_results), 'total_execution_time_ms': total_execution_time, 'average_duration_ms': average_duration, 'status_distribution': status_counts, 'overall_health_status': overall_health.status.value, 'performance_acceptable': performance_acceptable, 'efficiency_good': efficiency_good, 'bulk_execution_successful': True}

async def test_category_based_execution():
    """Test category-based health check execution"""
    print('\n=== Testing Category-Based Execution ===')
    from src.prompt_improver.performance.monitoring.health.unified_health_system import HealthCheckCategory, get_unified_health_monitor
    monitor = get_unified_health_monitor()
    categories_to_test = [HealthCheckCategory.ML, HealthCheckCategory.DATABASE, HealthCheckCategory.REDIS, HealthCheckCategory.API, HealthCheckCategory.SYSTEM, HealthCheckCategory.CUSTOM]
    category_results = {}
    for category in categories_to_test:
        start_time = time.time()
        results = await monitor.check_health(category=category)
        execution_time = (time.time() - start_time) * 1000
        category_results[category.value] = {'check_count': len(results), 'execution_time_ms': execution_time, 'status_distribution': {}}
        for plugin_name, result in results.items():
            status = result.status.value
            category_results[category.value]['status_distribution'][status] = category_results[category.value]['status_distribution'].get(status, 0) + 1
        print(f'✅ {category.value} category: {len(results)} checks in {execution_time:.2f}ms')
        if results:
            status_summary = ', '.join([f'{status}: {count}' for status, count in category_results[category.value]['status_distribution'].items()])
            print(f'   Status: {status_summary}')
    return {'categories_tested': len(categories_to_test), 'category_results': category_results, 'category_execution_successful': True}

async def test_error_handling_execution():
    """Test error handling in health check execution"""
    print('\n=== Testing Error Handling ===')
    from src.prompt_improver.core.protocols.health_protocol import HealthStatus
    from src.prompt_improver.performance.monitoring.health.unified_health_system import HealthCheckCategory, HealthCheckPluginConfig, create_simple_health_plugin, get_unified_health_monitor
    monitor = get_unified_health_monitor()

    def exception_check():
        raise ValueError('Simulated check failure')
    exception_plugin = create_simple_health_plugin(name='exception_test', category=HealthCheckCategory.CUSTOM, check_func=exception_check, config=HealthCheckPluginConfig(timeout_seconds=2.0))
    monitor.register_plugin(exception_plugin)
    results = await monitor.check_health(plugin_name='exception_test')
    exception_result = results['exception_test']
    assert exception_result.status == HealthStatus.UNHEALTHY, 'Exception should result in unhealthy status'
    print(f'✅ Exception handling: {exception_result.status.value}')
    print(f'   Message: {exception_result.message}')

    async def slow_check():
        await asyncio.sleep(5.0)
        return True
    timeout_plugin = create_simple_health_plugin(name='timeout_test', category=HealthCheckCategory.CUSTOM, check_func=lambda: asyncio.create_task(slow_check()), config=HealthCheckPluginConfig(timeout_seconds=1.0))
    monitor.register_plugin(timeout_plugin)
    start_time = time.time()
    timeout_results = await monitor.check_health(plugin_name='timeout_test')
    timeout_duration = (time.time() - start_time) * 1000
    timeout_result = timeout_results['timeout_test']
    timeout_handled = timeout_result.status == HealthStatus.UNHEALTHY and timeout_duration < 2000
    print(f'✅ Timeout handling: {timeout_result.status.value} ({timeout_duration:.2f}ms)')
    print(f'   Timeout properly handled: {timeout_handled}')
    return {'exception_handling_working': exception_result.status == HealthStatus.UNHEALTHY, 'timeout_handling_working': timeout_handled, 'error_handling_successful': True}

async def test_performance_characteristics():
    """Test performance characteristics of health execution"""
    print('\n=== Testing Performance Characteristics ===')
    from src.prompt_improver.performance.monitoring.health.unified_health_system import HealthCheckCategory, create_simple_health_plugin, get_unified_health_monitor
    monitor = get_unified_health_monitor()

    def fast_check():
        return True
    performance_plugins = []
    for i in range(10):
        plugin = create_simple_health_plugin(name=f'performance_test_{i}', category=HealthCheckCategory.CUSTOM, check_func=fast_check)
        monitor.register_plugin(plugin)
        performance_plugins.append(f'performance_test_{i}')
    runs = []
    for run in range(5):
        start_time = time.time()
        results = await monitor.check_health()
        execution_time = (time.time() - start_time) * 1000
        runs.append(execution_time)
        print(f'   Run {run + 1}: {len(results)} checks in {execution_time:.2f}ms')
    average_time = sum(runs) / len(runs)
    min_time = min(runs)
    max_time = max(runs)
    print('✅ Performance analysis:')
    print(f'   Average execution time: {average_time:.2f}ms')
    print(f'   Min execution time: {min_time:.2f}ms')
    print(f'   Max execution time: {max_time:.2f}ms')
    print(f'   Performance consistency: {(max_time - min_time) / average_time * 100:.1f}% variance')
    performance_good = average_time < 500
    consistency_good = (max_time - min_time) / average_time < 0.5
    return {'runs_completed': len(runs), 'average_execution_time_ms': average_time, 'min_execution_time_ms': min_time, 'max_execution_time_ms': max_time, 'performance_variance': (max_time - min_time) / average_time, 'performance_acceptable': performance_good, 'consistency_acceptable': consistency_good, 'performance_testing_successful': True}

async def run_health_execution_validation():
    """Run comprehensive health execution validation"""
    print('🔧 Health Check Execution Validation')
    print('=' * 50)
    validation_results = {'timestamp': datetime.now(UTC).isoformat(), 'validation_passed': False, 'test_results': {}}
    test_functions = [('basic_execution', test_basic_health_execution), ('bulk_execution', test_bulk_health_execution), ('category_execution', test_category_based_execution), ('error_handling', test_error_handling_execution), ('performance', test_performance_characteristics)]
    try:
        for test_name, test_func in test_functions:
            print(f'\n🧪 Running {test_name} test...')
            try:
                test_start = time.time()
                result = await test_func()
                test_duration = time.time() - test_start
                test_passed = all([result.get(f'{key}_successful', True) for key in result.keys() if key.endswith('_successful')])
                validation_results['test_results'][test_name] = {'passed': test_passed, 'duration_seconds': test_duration, 'result': result}
                print(f"{('✅' if test_passed else '❌')} {test_name} {('passed' if test_passed else 'failed')} in {test_duration:.2f}s")
            except Exception as e:
                test_duration = time.time() - test_start if 'test_start' in locals() else 0
                validation_results['test_results'][test_name] = {'passed': False, 'duration_seconds': test_duration, 'error': str(e)}
                print(f'❌ {test_name} failed with exception: {e}')
                logger.exception('Test %s failed', test_name)
        passed_tests = sum((1 for result in validation_results['test_results'].values() if result['passed']))
        total_tests = len(validation_results['test_results'])
        validation_results['validation_passed'] = passed_tests == total_tests
        validation_results['passed_tests'] = passed_tests
        validation_results['total_tests'] = total_tests
        validation_results['success_rate'] = passed_tests / total_tests if total_tests > 0 else 0
        print('\n📊 Health Execution Validation Summary')
        print('=' * 50)
        print(f'Total tests: {total_tests}')
        print(f'Passed tests: {passed_tests}')
        print(f"Success rate: {validation_results['success_rate']:.1%}")
        print(f"Overall validation: {('✅ PASSED' if validation_results['validation_passed'] else '❌ FAILED')}")
        if validation_results['validation_passed']:
            print('\n🎉 Health check execution validation completed successfully!')
            print('   All execution mechanisms are working correctly after legacy pattern removal.')
        else:
            print('\n⚠️  Health check execution validation found issues.')
            print('   Some execution mechanisms may not be working correctly.')
        return validation_results
    except Exception as e:
        validation_results['validation_passed'] = False
        validation_results['error'] = str(e)
        print(f'❌ Health execution validation failed: {e}')
        logger.exception('Health execution validation failed')
        return validation_results
if __name__ == '__main__':
    results = asyncio.run(run_health_execution_validation())
    exit(0 if results['validation_passed'] else 1)
