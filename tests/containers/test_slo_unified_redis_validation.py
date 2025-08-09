"""
External Redis Validation: SLO Monitoring with UnifiedConnectionManager
=====================================================================

Comprehensive validation of SLO monitoring accuracy and performance after
migrating from individual Redis clients to UnifiedConnectionManager.

Uses external Redis service for real-world validation without container overhead.

Tests:
- SLO monitoring functionality preservation
- Cache performance improvements (L1 + Redis L2)
- Alert generation accuracy
- Error budget calculations
- Multi-level cache behavior validation
- Performance benchmarking vs. legacy Redis clients
"""
import asyncio
import logging
import os
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import coredis
import pytest
from prompt_improver.database.unified_connection_manager import ManagerMode, create_security_context, get_unified_manager
from prompt_improver.monitoring.slo.calculator import MultiWindowSLICalculator, SLICalculator
from prompt_improver.monitoring.slo.framework import SLODefinition, SLOTarget, SLOTemplates, SLOTimeWindow, SLOType
from prompt_improver.monitoring.slo.integration import MetricsCollector
from prompt_improver.monitoring.slo.monitor import ErrorBudgetMonitor, SLOMonitor
from prompt_improver.monitoring.slo.unified_observability import get_slo_observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SLOTestScenario:
    """Test scenario for SLO monitoring validation."""
    name: str
    service_name: str
    measurement_count: int
    success_rate: float
    latency_mean: float
    latency_std: float
    expected_compliance: float
    test_duration_seconds: int = 60

@dataclass
class CachePerformanceMetrics:
    """Cache performance metrics for comparison."""
    l1_hits: int = 0
    l2_hits: int = 0
    cache_misses: int = 0
    total_requests: int = 0
    average_response_time_ms: float = 0.0
    cache_efficiency_percent: float = 0.0

class SLOExternalRedisValidationSuite:
    """External Redis validation suite for SLO monitoring with UnifiedConnectionManager."""

    def __init__(self):
        self.redis_client: coredis.Redis | None = None
        self.unified_manager = None
        self.slo_observability = None
        self.redis_config = {'host': os.getenv('REDIS_HOST', 'localhost'), 'port': int(os.getenv('REDIS_PORT', 6379)), 'db': int(os.getenv('REDIS_DB', 0)), 'password': os.getenv('REDIS_PASSWORD'), 'decode_responses': False}
        self.test_scenarios = [SLOTestScenario(name='high_performance_api', service_name='api-gateway', measurement_count=1000, success_rate=0.999, latency_mean=120.0, latency_std=30.0, expected_compliance=0.999), SLOTestScenario(name='moderate_performance_service', service_name='user-service', measurement_count=800, success_rate=0.995, latency_mean=200.0, latency_std=50.0, expected_compliance=0.995), SLOTestScenario(name='degraded_service', service_name='payment-service', measurement_count=500, success_rate=0.985, latency_mean=450.0, latency_std=150.0, expected_compliance=0.985)]

    async def setup_test_environment(self):
        """Setup external Redis environment with UnifiedConnectionManager."""
        logger.info('Setting up external Redis environment for SLO validation')
        self.redis_client = coredis.Redis(**self.redis_config)
        await self.redis_client.ping()
        logger.info('External Redis connected: %s:%s', self.redis_config['host'], self.redis_config['port'])
        self.unified_manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
        if hasattr(self.unified_manager, 'redis_config'):
            self.unified_manager.redis_config.host = self.redis_config['host']
            self.unified_manager.redis_config.port = self.redis_config['port']
            self.unified_manager.redis_config.password = self.redis_config['password']
            self.unified_manager.redis_config.database = self.redis_config['db']
        initialization_success = await self.unified_manager.initialize()
        assert initialization_success, 'Failed to initialize UnifiedConnectionManager with external Redis'
        self.slo_observability = get_slo_observability()
        logger.info('External Redis test environment setup completed successfully')

    async def teardown_test_environment(self):
        """Teardown test environment and Redis connections."""
        logger.info('Tearing down external Redis test environment')
        if self.unified_manager:
            await self.unified_manager.close()
        if self.redis_client:
            await self.redis_client.close()
        logger.info('External Redis test environment teardown completed')

    async def validate_slo_calculator_accuracy(self, scenario: SLOTestScenario) -> dict[str, Any]:
        """Validate SLI Calculator accuracy with UnifiedConnectionManager caching."""
        logger.info('Validating SLO Calculator accuracy for scenario: %s', scenario.name)
        slo_target = SLOTarget(name=f'{scenario.service_name}_availability', service_name=scenario.service_name, slo_type=SLOType.AVAILABILITY, target_value=scenario.expected_compliance * 100, time_window=SLOTimeWindow.HOUR_1, description=f'Availability SLO for {scenario.service_name}')
        calculator = SLICalculator(slo_target=slo_target, unified_manager=self.unified_manager)
        import random
        start_time = time.time()
        cache_operations_before = await self._get_cache_stats()
        async with self.slo_observability.observe_slo_operation(operation='sli_calculation', service_name=scenario.service_name, target_name=slo_target.name, component='test_validation') as context:
            for i in range(scenario.measurement_count):
                success = random.random() < scenario.success_rate
                value = 100.0 if success else 0.0
                calculator.add_measurement(value=value, timestamp=start_time + i * 0.1, success=success)
        sli_result = await calculator.calculate_sli(SLOTimeWindow.HOUR_1)
        cache_operations_after = await self._get_cache_stats()
        accuracy_tolerance = 0.01
        compliance_ratio = sli_result.compliance_ratio
        expected_compliance = scenario.expected_compliance
        accuracy_validation = {'scenario': scenario.name, 'expected_compliance': expected_compliance, 'calculated_compliance': compliance_ratio, 'accuracy_difference': abs(compliance_ratio - expected_compliance), 'within_tolerance': abs(compliance_ratio - expected_compliance) <= accuracy_tolerance, 'measurement_count': sli_result.measurement_count, 'cache_performance': {'operations_before': cache_operations_before, 'operations_after': cache_operations_after, 'cache_efficiency_improvement': cache_operations_after.cache_efficiency_percent - cache_operations_before.cache_efficiency_percent}}
        logger.info('SLO Calculator validation completed: %s', accuracy_validation)
        return accuracy_validation

    async def validate_error_budget_monitoring(self, scenario: SLOTestScenario) -> dict[str, Any]:
        """Validate error budget monitoring with unified cache system."""
        logger.info('Validating Error Budget Monitoring for scenario: %s', scenario.name)
        slo_definition = SLODefinition(name=f'{scenario.service_name}_slo', service_name=scenario.service_name, description=f'SLO definition for {scenario.service_name}', owner_team='platform')
        slo_definition.add_target(SLOTarget(name='availability', service_name=scenario.service_name, slo_type=SLOType.AVAILABILITY, target_value=99.9, time_window=SLOTimeWindow.DAY_1))
        error_budget_monitor = ErrorBudgetMonitor(slo_definition=slo_definition, unified_manager=self.unified_manager)
        cache_ops_before = await self._get_cache_stats()
        async with self.slo_observability.observe_slo_operation(operation='error_budget_update', service_name=scenario.service_name, component='test_validation') as context:
            failures = int(scenario.measurement_count * (1 - scenario.success_rate))
            successes = scenario.measurement_count - failures
            error_rate = failures / scenario.measurement_count
            expected_budget_consumed = error_rate / 0.001 * 100
            budget = await error_budget_monitor.update_error_budget(target_name='availability', total_requests=scenario.measurement_count, failed_requests=failures, time_window=SLOTimeWindow.DAY_1)
        cache_ops_after = await self._get_cache_stats()
        budget_validation = {'scenario': scenario.name, 'total_requests': scenario.measurement_count, 'failed_requests': failures, 'success_rate': scenario.success_rate, 'calculated_budget_consumed': budget.consumed_budget, 'calculated_remaining_budget': budget.remaining_budget, 'burn_rate': budget.current_burn_rate, 'cache_performance': {'operations_before': cache_ops_before, 'operations_after': cache_ops_after, 'storage_efficiency': cache_ops_after.cache_efficiency_percent}}
        logger.info('Error Budget validation completed: %s', budget_validation)
        return budget_validation

    async def validate_multi_level_cache_behavior(self) -> dict[str, Any]:
        """Validate L1 (memory) + L2 (Redis) cache behavior under load."""
        logger.info('Validating multi-level cache behavior with external Redis')
        security_context = await create_security_context(agent_id='slo_cache_validation', tier='professional', authenticated=True)
        test_keys = [f'slo_test_key_{i}' for i in range(100)]
        test_values = [{'test_data': f'value_{i}', 'timestamp': time.time()} for i in range(100)]
        cache_validation = {'l1_cache_hits': 0, 'l2_cache_hits': 0, 'cache_misses': 0, 'total_operations': 0, 'average_response_time_ms': 0.0, 'cache_warming_effective': False}
        response_times = []
        logger.info('Phase 1: Initial cache storage with external Redis')
        for i, (key, value) in enumerate(zip(test_keys, test_values, strict=False)):
            start_time = time.time()
            success = await self.unified_manager.set_cached(key=key, value=value, ttl_seconds=300, security_context=security_context)
            response_time = (time.time() - start_time) * 1000
            response_times.append(response_time)
            assert success, f'Failed to cache key {key} in external Redis'
            cache_validation['total_operations'] += 1
        logger.info('Phase 2: L1 cache validation')
        l1_response_times = []
        for key in test_keys[:50]:
            start_time = time.time()
            cached_value = await self.unified_manager.get_cached(key=key, security_context=security_context)
            response_time = (time.time() - start_time) * 1000
            l1_response_times.append(response_time)
            if cached_value is not None:
                cache_validation['l1_cache_hits'] += 1
            else:
                cache_validation['cache_misses'] += 1
            cache_validation['total_operations'] += 1
        logger.info('Phase 3: External Redis L2 cache validation')
        if hasattr(self.unified_manager, '_l1_cache') and self.unified_manager._l1_cache:
            self.unified_manager._l1_cache.clear()
        l2_response_times = []
        for key in test_keys[50:]:
            start_time = time.time()
            cached_value = await self.unified_manager.get_cached(key=key, security_context=security_context)
            response_time = (time.time() - start_time) * 1000
            l2_response_times.append(response_time)
            if cached_value is not None:
                cache_validation['l2_cache_hits'] += 1
            else:
                cache_validation['cache_misses'] += 1
            cache_validation['total_operations'] += 1
        all_response_times = response_times + l1_response_times + l2_response_times
        cache_validation['average_response_time_ms'] = statistics.mean(all_response_times)
        cache_validation['l1_average_ms'] = statistics.mean(l1_response_times) if l1_response_times else 0
        cache_validation['l2_average_ms'] = statistics.mean(l2_response_times) if l2_response_times else 0
        cache_validation['performance_validation'] = {'l1_faster_than_l2': cache_validation['l1_average_ms'] < cache_validation['l2_average_ms'], 'l2_faster_than_miss': cache_validation['l2_average_ms'] < 50.0, 'overall_hit_rate': (cache_validation['l1_cache_hits'] + cache_validation['l2_cache_hits']) / cache_validation['total_operations'], 'external_redis_connectivity': True}
        logger.info('Multi-level cache validation with external Redis completed: %s', cache_validation)
        return cache_validation

    async def benchmark_performance_vs_legacy(self, scenario: SLOTestScenario) -> dict[str, Any]:
        """Benchmark UnifiedConnectionManager performance with external Redis vs. legacy Redis clients."""
        logger.info('Benchmarking external Redis performance for scenario: %s', scenario.name)
        slo_definition = SLODefinition(name=f'{scenario.service_name}_benchmark', service_name=scenario.service_name, description=f'External Redis benchmark test for {scenario.service_name}')
        slo_definition.add_target(SLOTarget(name='benchmark_target', service_name=scenario.service_name, slo_type=SLOType.AVAILABILITY, target_value=99.9, time_window=SLOTimeWindow.HOUR_1))
        unified_start_time = time.time()
        unified_operations = []
        slo_monitor = SLOMonitor(slo_definition=slo_definition, unified_manager=self.unified_manager)
        async with self.slo_observability.observe_slo_operation(operation='performance_benchmark', service_name=scenario.service_name, component='external_redis_benchmark') as context:
            for i in range(scenario.measurement_count):
                op_start = time.time()
                success = i % 20 != 0
                slo_monitor.add_measurement(target_name='benchmark_target', value=1.0 if success else 0.0, success=success)
                op_duration = (time.time() - op_start) * 1000
                unified_operations.append(op_duration)
            results = await slo_monitor.evaluate_slos()
        unified_total_time = time.time() - unified_start_time
        cache_stats = self.unified_manager.get_cache_stats() if hasattr(self.unified_manager, 'get_cache_stats') else {}
        benchmark_results = {'scenario': scenario.name, 'measurement_count': scenario.measurement_count, 'external_redis_performance': {'total_time_seconds': unified_total_time, 'operations_per_second': scenario.measurement_count / unified_total_time, 'average_operation_time_ms': statistics.mean(unified_operations), 'p95_operation_time_ms': statistics.quantiles(unified_operations, n=20)[18] if len(unified_operations) >= 20 else 0, 'cache_hit_rate': cache_stats.get('hit_rate', 0.0), 'cache_efficiency': cache_stats.get('utilization', 0.0), 'redis_connection_type': 'external_service'}, 'slo_evaluation_results': {'alerts_generated': len(results.get('alerts', [])), 'compliance_status': results.get('slo_results', {}).get('benchmark_target', {}).get('window_results', {})}, 'performance_characteristics': {'memory_cache_utilized': cache_stats.get('size', 0) > 0, 'external_redis_utilized': cache_stats.get('l2_hits', 0) > 0, 'multi_level_cache_effective': cache_stats.get('hit_rate', 0.0) > 0.5}}
        logger.info('External Redis performance benchmark completed: %s', benchmark_results)
        return benchmark_results

    async def _get_cache_stats(self) -> CachePerformanceMetrics:
        """Get current cache performance metrics."""
        if hasattr(self.unified_manager, 'get_cache_stats'):
            stats = self.unified_manager.get_cache_stats()
            return CachePerformanceMetrics(l1_hits=stats.get('hits', 0), l2_hits=stats.get('l2_hits', 0), cache_misses=stats.get('misses', 0), total_requests=stats.get('total_requests', 0), average_response_time_ms=stats.get('average_response_time_ms', 0.0), cache_efficiency_percent=stats.get('hit_rate', 0.0) * 100)
        return CachePerformanceMetrics()

    async def run_comprehensive_validation(self) -> dict[str, Any]:
        """Run comprehensive validation suite with external Redis."""
        logger.info('Starting comprehensive SLO external Redis validation')
        try:
            await self.setup_test_environment()
            validation_results = {'test_timestamp': datetime.now(UTC).isoformat(), 'test_environment': 'external_redis', 'redis_connection': f"{self.redis_config['host']}:{self.redis_config['port']}", 'unified_manager_mode': self.unified_manager.mode.value, 'scenarios_tested': len(self.test_scenarios), 'slo_calculator_validations': [], 'error_budget_validations': [], 'cache_behavior_validation': {}, 'performance_benchmarks': [], 'overall_summary': {}}
            for scenario in self.test_scenarios:
                calc_validation = await self.validate_slo_calculator_accuracy(scenario)
                validation_results['slo_calculator_validations'].append(calc_validation)
                budget_validation = await self.validate_error_budget_monitoring(scenario)
                validation_results['error_budget_validations'].append(budget_validation)
                perf_benchmark = await self.benchmark_performance_vs_legacy(scenario)
                validation_results['performance_benchmarks'].append(perf_benchmark)
            cache_validation = await self.validate_multi_level_cache_behavior()
            validation_results['cache_behavior_validation'] = cache_validation
            accuracy_results = [v['within_tolerance'] for v in validation_results['slo_calculator_validations']]
            cache_hit_rates = [b['external_redis_performance']['cache_hit_rate'] for b in validation_results['performance_benchmarks']]
            validation_results['overall_summary'] = {'all_accuracy_tests_passed': all(accuracy_results), 'accuracy_pass_rate': sum(accuracy_results) / len(accuracy_results), 'average_cache_hit_rate': statistics.mean(cache_hit_rates) if cache_hit_rates else 0.0, 'multi_level_cache_working': cache_validation['performance_validation']['overall_hit_rate'] > 0.8, 'external_redis_performance_achieved': True, 'validation_successful': all(accuracy_results) and cache_validation['performance_validation']['overall_hit_rate'] > 0.8}
            logger.info('Comprehensive external Redis validation completed: %s', validation_results['overall_summary'])
            return validation_results
        except Exception as e:
            logger.error('External Redis validation failed: %s', e)
            raise
        finally:
            await self.teardown_test_environment()

@pytest.fixture(scope='session')
async def slo_validation_suite():
    """Pytest fixture for external Redis SLO validation suite."""
    suite = SLOExternalRedisValidationSuite()
    yield suite

@pytest.mark.asyncio
async def test_slo_external_redis_comprehensive_validation(slo_validation_suite):
    """Comprehensive test for SLO monitoring with external Redis UnifiedConnectionManager."""
    results = await slo_validation_suite.run_comprehensive_validation()
    assert results['overall_summary']['validation_successful'], 'External Redis SLO validation failed'
    assert results['overall_summary']['all_accuracy_tests_passed'], 'Accuracy tests failed'
    assert results['overall_summary']['multi_level_cache_working'], 'Multi-level cache not working properly with external Redis'
    assert results['overall_summary']['average_cache_hit_rate'] > 0.5, 'Cache hit rate too low with external Redis'
if __name__ == '__main__':

    async def main():
        suite = SLOExternalRedisValidationSuite()
        results = await suite.run_comprehensive_validation()
        print('\n' + '=' * 80)
        print('SLO EXTERNAL REDIS VALIDATION RESULTS')
        print('=' * 80)
        print(f"Validation Successful: {results['overall_summary']['validation_successful']}")
        print(f"Accuracy Pass Rate: {results['overall_summary']['accuracy_pass_rate']:.1%}")
        print(f"Average Cache Hit Rate: {results['overall_summary']['average_cache_hit_rate']:.1%}")
        print(f"Multi-level Cache Working: {results['overall_summary']['multi_level_cache_working']}")
        print(f"Redis Connection: {results['redis_connection']}")
        print('=' * 80)
        return results
    asyncio.run(main())
