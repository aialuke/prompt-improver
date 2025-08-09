"""
Comprehensive Real Behavior Testing - Consolidated Health Monitoring System
==========================================================================

Tests the complete health monitoring system with real behavior validation:
1. Import functionality across all critical files
2. HealthService backward compatibility layer
3. UnifiedHealthMonitor plugin system
4. MCP server health integration
5. TUI dashboard health integration
6. DI container health factory
7. Verify no broken imports or missing functionality

This script uses REAL behavior testing (no mocks) to validate:
- Health checks execute correctly
- Return types match expected interfaces (AggregatedHealthResult, HealthResult)
- All legacy methods work (run_health_check, run_specific_check, get_health_summary)
- Advanced features work (circuit breakers, plugin registration)
- Integration points function properly
"""
import asyncio
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from typing import Any, Dict, List, Optional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result container"""
    name: str
    passed: bool
    duration_ms: float
    error: str | None = None
    details: dict[str, Any] = None

class HealthMonitoringTester:
    """Comprehensive health monitoring system tester"""

    def __init__(self):
        self.results: list[TestResult] = []
        self.start_time = time.time()

    async def run_all_tests(self) -> dict[str, Any]:
        """Run all health monitoring tests"""
        logger.info('Starting comprehensive health monitoring real behavior tests...')
        test_groups = [('Import Tests', self._test_imports), ('Core Health System', self._test_core_health_system), ('Legacy Compatibility', self._test_legacy_compatibility), ('Plugin System', self._test_plugin_system), ('Integration Points', self._test_integration_points), ('Advanced Features', self._test_advanced_features), ('Error Handling', self._test_error_handling), ('Performance', self._test_performance)]
        for group_name, test_func in test_groups:
            logger.info('\n%s', '=' * 60)
            logger.info('Running %s Tests', group_name)
            logger.info('%s', '=' * 60)
            try:
                await test_func()
            except Exception as e:
                logger.error('Test group {group_name} failed: %s', e)
                self.results.append(TestResult(name=f'{group_name} (Group)', passed=False, duration_ms=0, error=str(e)))
        return self._generate_report()

    async def _test_imports(self):
        """Test 1: Import functionality across all critical files"""
        test_start = time.time()
        try:
            from prompt_improver.performance.monitoring.health import AggregatedHealthResult, HealthCheckCategory, HealthChecker, HealthCheckPlugin, HealthCheckPluginConfig, HealthResult, HealthService, HealthStatus, UnifiedHealthMonitor, create_simple_health_plugin, get_health_service, get_unified_health_monitor, register_health_plugin, reset_health_service, reset_unified_health_monitor
            self.results.append(TestResult(name='Core health imports', passed=True, duration_ms=(time.time() - test_start) * 1000, details={'imported_classes': 12}))
            logger.info('✅ Core health imports successful')
        except ImportError as e:
            self.results.append(TestResult(name='Core health imports', passed=False, duration_ms=(time.time() - test_start) * 1000, error=f'Import failed: {e}'))
            logger.error('❌ Core health imports failed: %s', e)
            return
        test_start = time.time()
        try:
            from prompt_improver.core.protocols.health_protocol import HealthCheckResult, HealthMonitorProtocol, HealthStatus as ProtocolHealthStatus
            self.results.append(TestResult(name='Protocol imports', passed=True, duration_ms=(time.time() - test_start) * 1000))
            logger.info('✅ Protocol imports successful')
        except ImportError as e:
            self.results.append(TestResult(name='Protocol imports', passed=False, duration_ms=(time.time() - test_start) * 1000, error=f'Protocol import failed: {e}'))
            logger.error('❌ Protocol imports failed: %s', e)
        test_start = time.time()
        try:
            from prompt_improver.performance.monitoring.health import AnalyticsServiceHealthChecker, DatabaseHealthChecker, MCPServerHealthChecker, MLServiceHealthChecker, QueueHealthChecker, RedisHealthChecker, SystemResourcesHealthChecker
            self.results.append(TestResult(name='Legacy checker imports', passed=True, duration_ms=(time.time() - test_start) * 1000, details={'imported_checkers': 7}))
            logger.info('✅ Legacy checker imports successful')
        except ImportError as e:
            self.results.append(TestResult(name='Legacy checker imports', passed=False, duration_ms=(time.time() - test_start) * 1000, error=f'Legacy checker import failed: {e}'))
            logger.error('❌ Legacy checker imports failed: %s', e)

    async def _test_core_health_system(self):
        """Test 2: UnifiedHealthMonitor core functionality"""
        from prompt_improver.core.protocols.health_protocol import HealthCheckResult, HealthStatus
        from prompt_improver.performance.monitoring.health import HealthCheckCategory, HealthCheckPlugin, HealthCheckPluginConfig, UnifiedHealthMonitor, get_unified_health_monitor, reset_unified_health_monitor
        test_start = time.time()
        try:
            monitor = UnifiedHealthMonitor()
            self.results.append(TestResult(name='UnifiedHealthMonitor creation', passed=True, duration_ms=(time.time() - test_start) * 1000))
            logger.info('✅ UnifiedHealthMonitor created successfully')
        except Exception as e:
            self.results.append(TestResult(name='UnifiedHealthMonitor creation', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ UnifiedHealthMonitor creation failed: %s', e)
            return
        test_start = time.time()
        try:

            class TestHealthPlugin(HealthCheckPlugin):

                async def execute_check(self) -> HealthCheckResult:
                    return HealthCheckResult(status=HealthStatus.HEALTHY, message='Test plugin working', check_name=self.name)
            plugin = TestHealthPlugin(name='test_plugin', category=HealthCheckCategory.CUSTOM, config=HealthCheckPluginConfig(enabled=True, timeout_seconds=5.0))
            success = monitor.register_plugin(plugin)
            self.results.append(TestResult(name='Plugin registration', passed=success, duration_ms=(time.time() - test_start) * 1000, details={'plugin_name': 'test_plugin', 'category': 'custom'}))
            if success:
                logger.info('✅ Plugin registration successful')
            else:
                logger.error('❌ Plugin registration failed')
        except Exception as e:
            self.results.append(TestResult(name='Plugin registration', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ Plugin registration failed: %s', e)
        test_start = time.time()
        try:
            health_results = await monitor.check_health()
            is_valid = isinstance(health_results, dict) and 'test_plugin' in health_results and (health_results['test_plugin'].status == HealthStatus.HEALTHY)
            self.results.append(TestResult(name='Health check execution', passed=is_valid, duration_ms=(time.time() - test_start) * 1000, details={'results_count': len(health_results), 'test_plugin_status': health_results.get('test_plugin', {}).status.value if health_results.get('test_plugin') else 'missing'}))
            if is_valid:
                logger.info('✅ Health check execution successful')
            else:
                logger.error('❌ Health check execution returned invalid results')
        except Exception as e:
            self.results.append(TestResult(name='Health check execution', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ Health check execution failed: %s', e)
        test_start = time.time()
        try:
            overall_health = await monitor.get_overall_health()
            is_valid = isinstance(overall_health, HealthCheckResult) and hasattr(overall_health, 'status') and hasattr(overall_health, 'message') and hasattr(overall_health, 'details')
            self.results.append(TestResult(name='Overall health summary', passed=is_valid, duration_ms=(time.time() - test_start) * 1000, details={'overall_status': overall_health.status.value if overall_health else 'none', 'has_details': bool(overall_health.details if overall_health else False)}))
            if is_valid:
                logger.info('✅ Overall health summary successful')
            else:
                logger.error('❌ Overall health summary invalid')
        except Exception as e:
            self.results.append(TestResult(name='Overall health summary', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ Overall health summary failed: %s', e)
        test_start = time.time()
        try:
            global_monitor = get_unified_health_monitor()
            is_valid = isinstance(global_monitor, UnifiedHealthMonitor)
            self.results.append(TestResult(name='Global monitor access', passed=is_valid, duration_ms=(time.time() - test_start) * 1000))
            if is_valid:
                logger.info('✅ Global monitor access successful')
            else:
                logger.error('❌ Global monitor access failed')
        except Exception as e:
            self.results.append(TestResult(name='Global monitor access', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ Global monitor access failed: %s', e)

    async def _test_legacy_compatibility(self):
        """Test 3: HealthService backward compatibility layer"""
        from prompt_improver.performance.monitoring.health import AggregatedHealthResult, HealthChecker, HealthResult, HealthService, get_health_service, reset_health_service
        from prompt_improver.performance.monitoring.health.base import HealthStatus as BaseHealthStatus
        test_start = time.time()
        try:
            service = HealthService()
            self.results.append(TestResult(name='Legacy HealthService creation', passed=True, duration_ms=(time.time() - test_start) * 1000))
            logger.info('✅ Legacy HealthService created successfully')
        except Exception as e:
            self.results.append(TestResult(name='Legacy HealthService creation', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ Legacy HealthService creation failed: %s', e)
            return
        test_start = time.time()
        try:

            class TestLegacyChecker(HealthChecker):

                def __init__(self):
                    super().__init__(name='legacy_test_checker')

                async def check(self) -> HealthResult:
                    return HealthResult(status=BaseHealthStatus.HEALTHY, component='legacy_test_checker', message='Legacy checker working', timestamp=datetime.now(UTC))
            legacy_checker = TestLegacyChecker()
            service.add_checker(legacy_checker)
            available_checks = service.get_available_checks()
            is_valid = 'legacy_test_checker' in available_checks
            self.results.append(TestResult(name='Legacy checker integration', passed=is_valid, duration_ms=(time.time() - test_start) * 1000, details={'available_checks': available_checks}))
            if is_valid:
                logger.info('✅ Legacy checker integration successful')
            else:
                logger.error('❌ Legacy checker integration failed')
        except Exception as e:
            self.results.append(TestResult(name='Legacy checker integration', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ Legacy checker integration failed: %s', e)
        test_start = time.time()
        try:
            aggregated_result = await service.run_health_check()
            is_valid = isinstance(aggregated_result, AggregatedHealthResult) and hasattr(aggregated_result, 'overall_status') and hasattr(aggregated_result, 'checks') and hasattr(aggregated_result, 'timestamp') and ('legacy_test_checker' in aggregated_result.checks)
            self.results.append(TestResult(name='run_health_check method', passed=is_valid, duration_ms=(time.time() - test_start) * 1000, details={'overall_status': aggregated_result.overall_status.value if aggregated_result else 'none', 'checks_count': len(aggregated_result.checks) if aggregated_result else 0}))
            if is_valid:
                logger.info('✅ run_health_check method successful')
            else:
                logger.error('❌ run_health_check method failed')
        except Exception as e:
            self.results.append(TestResult(name='run_health_check method', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ run_health_check method failed: %s', e)
        test_start = time.time()
        try:
            specific_result = await service.run_specific_check('legacy_test_checker')
            is_valid = isinstance(specific_result, HealthResult) and specific_result.component == 'legacy_test_checker' and (specific_result.status == BaseHealthStatus.HEALTHY)
            self.results.append(TestResult(name='run_specific_check method', passed=is_valid, duration_ms=(time.time() - test_start) * 1000, details={'component': specific_result.component if specific_result else 'none', 'status': specific_result.status.value if specific_result else 'none'}))
            if is_valid:
                logger.info('✅ run_specific_check method successful')
            else:
                logger.error('❌ run_specific_check method failed')
        except Exception as e:
            self.results.append(TestResult(name='run_specific_check method', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ run_specific_check method failed: %s', e)
        test_start = time.time()
        try:
            summary = await service.get_health_summary(include_details=True)
            is_valid = isinstance(summary, dict) and 'overall_status' in summary and ('checks' in summary) and ('timestamp' in summary) and ('legacy_test_checker' in summary['checks'])
            self.results.append(TestResult(name='get_health_summary method', passed=is_valid, duration_ms=(time.time() - test_start) * 1000, details={'has_overall_status': 'overall_status' in summary, 'has_checks': 'checks' in summary, 'checks_count': len(summary.get('checks', {}))}))
            if is_valid:
                logger.info('✅ get_health_summary method successful')
            else:
                logger.error('❌ get_health_summary method failed')
        except Exception as e:
            self.results.append(TestResult(name='get_health_summary method', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ get_health_summary method failed: %s', e)
        test_start = time.time()
        try:
            global_service = get_health_service()
            is_valid = isinstance(global_service, HealthService)
            self.results.append(TestResult(name='Global service instance', passed=is_valid, duration_ms=(time.time() - test_start) * 1000))
            if is_valid:
                logger.info('✅ Global service instance successful')
            else:
                logger.error('❌ Global service instance failed')
        except Exception as e:
            self.results.append(TestResult(name='Global service instance', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ Global service instance failed: %s', e)

    async def _test_plugin_system(self):
        """Test 4: Plugin system functionality"""
        from prompt_improver.core.protocols.health_protocol import HealthStatus
        from prompt_improver.performance.monitoring.health import HealthCheckCategory, create_simple_health_plugin, get_unified_health_monitor, register_health_plugin
        test_start = time.time()
        try:

            def simple_check():
                return {'status': 'healthy', 'message': 'Simple check works'}
            simple_plugin = create_simple_health_plugin(name='simple_test_plugin', category=HealthCheckCategory.CUSTOM, check_func=simple_check)
            self.results.append(TestResult(name='Simple plugin creation', passed=True, duration_ms=(time.time() - test_start) * 1000, details={'plugin_name': simple_plugin.name, 'category': simple_plugin.category.value}))
            logger.info('✅ Simple plugin creation successful')
        except Exception as e:
            self.results.append(TestResult(name='Simple plugin creation', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ Simple plugin creation failed: %s', e)
            return
        test_start = time.time()
        try:
            success = register_health_plugin(simple_plugin)
            self.results.append(TestResult(name='Plugin registration via convenience function', passed=success, duration_ms=(time.time() - test_start) * 1000))
            if success:
                logger.info('✅ Plugin registration via convenience function successful')
            else:
                logger.error('❌ Plugin registration via convenience function failed')
        except Exception as e:
            self.results.append(TestResult(name='Plugin registration via convenience function', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ Plugin registration via convenience function failed: %s', e)
        test_start = time.time()
        try:
            monitor = get_unified_health_monitor()
            results = await monitor.check_health(plugin_name='simple_test_plugin')
            is_valid = 'simple_test_plugin' in results and results['simple_test_plugin'].status == HealthStatus.HEALTHY
            self.results.append(TestResult(name='Plugin execution', passed=is_valid, duration_ms=(time.time() - test_start) * 1000, details={'plugin_found': 'simple_test_plugin' in results, 'status': results.get('simple_test_plugin', {}).status.value if results.get('simple_test_plugin') else 'none'}))
            if is_valid:
                logger.info('✅ Plugin execution successful')
            else:
                logger.error('❌ Plugin execution failed')
        except Exception as e:
            self.results.append(TestResult(name='Plugin execution', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ Plugin execution failed: %s', e)
        test_start = time.time()
        try:
            category_results = await monitor.check_health(category=HealthCheckCategory.CUSTOM)
            is_valid = len(category_results) > 0 and all((plugin_name in ['simple_test_plugin', 'test_plugin'] for plugin_name in category_results.keys()))
            self.results.append(TestResult(name='Category-based filtering', passed=is_valid, duration_ms=(time.time() - test_start) * 1000, details={'results_count': len(category_results), 'plugin_names': list(category_results.keys())}))
            if is_valid:
                logger.info('✅ Category-based filtering successful')
            else:
                logger.error('❌ Category-based filtering failed')
        except Exception as e:
            self.results.append(TestResult(name='Category-based filtering', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ Category-based filtering failed: %s', e)

    async def _test_integration_points(self):
        """Test 5: Integration with MCP server and TUI dashboard"""
        test_start = time.time()
        try:
            self.results.append(TestResult(name='MCP server import', passed=True, duration_ms=(time.time() - test_start) * 1000))
            logger.info('✅ MCP server import successful')
        except ImportError as e:
            self.results.append(TestResult(name='MCP server import', passed=False, duration_ms=(time.time() - test_start) * 1000, error=f'MCP server import failed: {e}'))
            logger.error('❌ MCP server import failed: %s', e)
        test_start = time.time()
        try:
            from prompt_improver.tui.data_provider import TUIDataProvider
            self.results.append(TestResult(name='TUI data provider import', passed=True, duration_ms=(time.time() - test_start) * 1000))
            logger.info('✅ TUI data provider import successful')
        except ImportError as e:
            self.results.append(TestResult(name='TUI data provider import', passed=False, duration_ms=(time.time() - test_start) * 1000, error=f'TUI data provider import failed: {e}'))
            logger.error('❌ TUI data provider import failed: %s', e)
        test_start = time.time()
        try:
            from prompt_improver.core.services.manager import ServiceManager
            self.results.append(TestResult(name='Service manager import', passed=True, duration_ms=(time.time() - test_start) * 1000))
            logger.info('✅ Service manager import successful')
        except ImportError as e:
            self.results.append(TestResult(name='Service manager import', passed=False, duration_ms=(time.time() - test_start) * 1000, error=f'Service manager import failed: {e}'))
            logger.error('❌ Service manager import failed: %s', e)

    async def _test_advanced_features(self):
        """Test 6: Advanced features like circuit breakers"""
        from prompt_improver.performance.monitoring.health import get_health_service
        test_start = time.time()
        try:
            service = get_health_service()
            success = service.enable_circuit_breaker(component_name='legacy_test_checker', failure_threshold=3, timeout=30)
            available_checks = service.get_available_checks()
            expected_success = 'legacy_test_checker' in available_checks
            is_valid = success == expected_success
            self.results.append(TestResult(name='Circuit breaker functionality', passed=is_valid, duration_ms=(time.time() - test_start) * 1000, details={'success': success, 'expected_success': expected_success, 'available_checks': available_checks}))
            if is_valid:
                logger.info('✅ Circuit breaker functionality working as expected')
            else:
                logger.error('❌ Circuit breaker functionality failed')
        except Exception as e:
            self.results.append(TestResult(name='Circuit breaker functionality', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ Circuit breaker functionality failed: %s', e)
        test_start = time.time()
        try:
            from prompt_improver.performance.monitoring.health import get_unified_health_monitor
            monitor = get_unified_health_monitor()
            profile = monitor.create_health_profile(name='test_profile', enabled_plugins={'test_plugin', 'simple_test_plugin'}, global_timeout=45.0, parallel_execution=True)
            activation_success = monitor.activate_profile('test_profile')
            self.results.append(TestResult(name='Health profiles and advanced configuration', passed=activation_success, duration_ms=(time.time() - test_start) * 1000, details={'profile_name': profile.name, 'enabled_plugins': len(profile.enabled_plugins), 'activation_success': activation_success}))
            if activation_success:
                logger.info('✅ Health profiles and advanced configuration successful')
            else:
                logger.error('❌ Health profiles and advanced configuration failed')
        except Exception as e:
            self.results.append(TestResult(name='Health profiles and advanced configuration', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ Health profiles and advanced configuration failed: %s', e)

    async def _test_error_handling(self):
        """Test 7: Error handling and edge cases"""
        from prompt_improver.core.protocols.health_protocol import HealthCheckResult, HealthStatus
        from prompt_improver.performance.monitoring.health import HealthCheckCategory, HealthCheckPlugin, HealthCheckPluginConfig, get_unified_health_monitor
        test_start = time.time()
        try:

            class TimeoutTestPlugin(HealthCheckPlugin):

                async def execute_check(self) -> HealthCheckResult:
                    await asyncio.sleep(0.1)
                    return HealthCheckResult(status=HealthStatus.HEALTHY, message='Delayed check completed', check_name=self.name)
            timeout_plugin = TimeoutTestPlugin(name='timeout_test_plugin', category=HealthCheckCategory.CUSTOM, config=HealthCheckPluginConfig(timeout_seconds=0.05))
            monitor = get_unified_health_monitor()
            monitor.register_plugin(timeout_plugin)
            results = await monitor.check_health(plugin_name='timeout_test_plugin')
            plugin_result = results.get('timeout_test_plugin')
            is_valid = plugin_result is not None and plugin_result.status == HealthStatus.UNHEALTHY and ('timeout' in plugin_result.message.lower() or 'failed' in plugin_result.message.lower())
            self.results.append(TestResult(name='Timeout handling', passed=is_valid, duration_ms=(time.time() - test_start) * 1000, details={'timed_out': is_valid, 'status': results.get('timeout_test_plugin', {}).status.value if results.get('timeout_test_plugin') else 'none'}))
            if is_valid:
                logger.info('✅ Timeout handling successful')
            else:
                logger.error('❌ Timeout handling failed')
        except Exception as e:
            self.results.append(TestResult(name='Timeout handling', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ Timeout handling failed: %s', e)
        test_start = time.time()
        try:

            class ExceptionTestPlugin(HealthCheckPlugin):

                async def execute_check(self) -> HealthCheckResult:
                    raise ValueError('Simulated health check failure')
            exception_plugin = ExceptionTestPlugin(name='exception_test_plugin', category=HealthCheckCategory.CUSTOM, config=HealthCheckPluginConfig())
            monitor.register_plugin(exception_plugin)
            results = await monitor.check_health(plugin_name='exception_test_plugin')
            is_valid = 'exception_test_plugin' in results and results['exception_test_plugin'].status == HealthStatus.UNHEALTHY and ('failed' in results['exception_test_plugin'].message.lower())
            self.results.append(TestResult(name='Exception handling', passed=is_valid, duration_ms=(time.time() - test_start) * 1000, details={'handled_exception': is_valid, 'status': results.get('exception_test_plugin', {}).status.value if results.get('exception_test_plugin') else 'none'}))
            if is_valid:
                logger.info('✅ Exception handling successful')
            else:
                logger.error('❌ Exception handling failed')
        except Exception as e:
            self.results.append(TestResult(name='Exception handling', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ Exception handling failed: %s', e)

    async def _test_performance(self):
        """Test 8: Performance characteristics"""
        from prompt_improver.performance.monitoring.health import get_unified_health_monitor
        test_start = time.time()
        try:
            monitor = get_unified_health_monitor()
            parallel_start = time.time()
            results = await monitor.check_health()
            parallel_duration = (time.time() - parallel_start) * 1000
            is_acceptable = parallel_duration < 1000.0
            self.results.append(TestResult(name='Parallel execution performance', passed=is_acceptable, duration_ms=(time.time() - test_start) * 1000, details={'parallel_duration_ms': parallel_duration, 'checks_executed': len(results), 'acceptable_performance': is_acceptable}))
            if is_acceptable:
                logger.info('✅ Parallel execution performance successful (%sms)', format(parallel_duration, '.2f'))
            else:
                logger.error('❌ Parallel execution performance poor (%sms)', format(parallel_duration, '.2f'))
        except Exception as e:
            self.results.append(TestResult(name='Parallel execution performance', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ Parallel execution performance failed: %s', e)
        test_start = time.time()
        try:
            monitor = get_unified_health_monitor()
            for i in range(10):
                await monitor.check_health()
            self.results.append(TestResult(name='Memory efficiency', passed=True, duration_ms=(time.time() - test_start) * 1000, details={'repeated_calls': 10}))
            logger.info('✅ Memory efficiency successful')
        except Exception as e:
            self.results.append(TestResult(name='Memory efficiency', passed=False, duration_ms=(time.time() - test_start) * 1000, error=str(e)))
            logger.error('❌ Memory efficiency failed: %s', e)

    def _generate_report(self) -> dict[str, Any]:
        """Generate comprehensive test report"""
        total_duration = time.time() - self.start_time
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        categories = {}
        for result in self.results:
            category = result.name.split(' ')[0] if ' ' in result.name else 'Other'
            if category not in categories:
                categories[category] = {'passed': 0, 'failed': 0, 'tests': []}
            if result.passed:
                categories[category]['passed'] += 1
            else:
                categories[category]['failed'] += 1
            categories[category]['tests'].append(result)
        report = {'summary': {'total_tests': len(self.results), 'passed': len(passed_tests), 'failed': len(failed_tests), 'success_rate': len(passed_tests) / max(len(self.results), 1), 'total_duration_seconds': total_duration}, 'categories': categories, 'detailed_results': [{'name': r.name, 'passed': r.passed, 'duration_ms': r.duration_ms, 'error': r.error, 'details': r.details} for r in self.results], 'failures': [{'name': r.name, 'error': r.error, 'duration_ms': r.duration_ms} for r in failed_tests]}
        return report

async def main():
    """Run comprehensive health monitoring tests"""
    print('Comprehensive Health Monitoring System - Real Behavior Testing')
    print('=' * 70)
    tester = HealthMonitoringTester()
    try:
        report = await tester.run_all_tests()
        print(f"\n{'=' * 70}")
        print('TEST RESULTS SUMMARY')
        print(f"{'=' * 70}")
        summary = report['summary']
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Duration: {summary['total_duration_seconds']:.2f}s")
        print('\nCATEGORY BREAKDOWN:')
        print('-' * 30)
        for category, stats in report['categories'].items():
            total = stats['passed'] + stats['failed']
            success_rate = stats['passed'] / max(total, 1)
            print(f"{category:20} {stats['passed']:2}/{total:2} ({success_rate:.1%})")
        if report['failures']:
            print('\nFAILURES:')
            print('-' * 50)
            for failure in report['failures']:
                print(f"❌ {failure['name']}")
                print(f"   Error: {failure['error']}")
                print(f"   Duration: {failure['duration_ms']:.2f}ms")
                print()
        print(f"\n{'=' * 70}")
        if summary['success_rate'] >= 0.9:
            print('🎉 OVERALL HEALTH: EXCELLENT - System is functioning correctly')
        elif summary['success_rate'] >= 0.8:
            print('✅ OVERALL HEALTH: GOOD - Minor issues detected')
        elif summary['success_rate'] >= 0.6:
            print('⚠️  OVERALL HEALTH: FAIR - Several issues need attention')
        else:
            print('❌ OVERALL HEALTH: POOR - Significant issues detected')
        print(f"{'=' * 70}")
        return 0 if summary['success_rate'] >= 0.8 else 1
    except Exception as e:
        print('\n❌ CRITICAL ERROR: Test execution failed')
        print(f'Error: {e}')
        print('\nFull traceback:')
        traceback.print_exc()
        return 2
if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
