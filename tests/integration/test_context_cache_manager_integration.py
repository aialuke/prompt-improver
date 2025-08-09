"""
Integration tests for ContextCacheManager with ML Pipeline Orchestrator.

Tests real behavior of ContextCacheManager integration including:
- Component loading through orchestrator
- Cache operations with TTL behavior
- OpenTelemetry tracing integration
- Performance metrics collection
- Integration with context learning workflows
"""
import asyncio
import logging
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ContextCacheManagerIntegrationTest:
    """Integration tests for ContextCacheManager."""

    def __init__(self):
        self.test_results = []

    def log_test(self, test_name: str, passed: bool, details: str=''):
        """Log test result."""
        status = '✅ PASS' if passed else '❌ FAIL'
        message = f'{status}: {test_name}'
        if details:
            message += f' - {details}'
        logger.info(message)
        self.test_results.append((test_name, passed, details))

    async def test_orchestrator_component_loading(self):
        """Test ContextCacheManager loads through orchestrator."""
        logger.info('🧪 Testing ContextCacheManager orchestrator integration...')
        try:
            from prompt_improver.ml.orchestration.core.component_registry import ComponentTier
            from prompt_improver.ml.orchestration.integration.direct_component_loader import DirectComponentLoader
            loader = DirectComponentLoader()
            loaded_component = await loader.load_component('context_cache_manager', ComponentTier.TIER_2)
            self.log_test('Component loads through orchestrator', loaded_component is not None)
            if loaded_component:
                from prompt_improver.ml.learning.algorithms.context_cache_manager import ContextCacheManager
                self.log_test('Correct class loaded', loaded_component.component_class == ContextCacheManager)
                success = await loader.initialize_component('context_cache_manager')
                self.log_test('Component initializes successfully', success)
                return True
            return False
        except Exception as e:
            self.log_test('Orchestrator integration', False, f'Error: {e}')
            return False

    async def test_cache_functionality(self):
        """Test real cache behavior with TTL and metrics."""
        logger.info('🧪 Testing ContextCacheManager cache functionality...')
        try:
            from prompt_improver.ml.learning.algorithms.context_cache_manager import ContextCacheManager
            cache_manager = ContextCacheManager(linguistic_cache_size=10, domain_cache_size=5, ttl_seconds=2)
            cache_manager.update_linguistic_cache('test_key1', {'features': [1, 2, 3]})
            cache_manager.update_linguistic_cache('test_key2', {'features': [4, 5, 6]})
            result1 = cache_manager.get_linguistic_cache('test_key1')
            self.log_test('Linguistic cache stores and retrieves', result1 is not None)
            self.log_test('Linguistic cache returns correct data', result1 == {'features': [1, 2, 3]})
            cache_manager.update_domain_cache('domain_key1', {'domain': 'technical', 'confidence': 0.9})
            domain_result = cache_manager.get_domain_cache('domain_key1')
            self.log_test('Domain cache stores and retrieves', domain_result is not None)
            stats = cache_manager.get_cache_stats()
            self.log_test('Cache statistics available', 'linguistic_cache_size' in stats)
            self.log_test('Hit rate tracking works', 'linguistic_hit_rate' in stats)
            logger.info('Testing TTL expiration (waiting 3 seconds)...')
            time.sleep(3)
            expired_result = cache_manager.get_linguistic_cache('test_key1')
            self.log_test('TTL expiration works', expired_result is None)
            cache_manager.update_linguistic_cache('expire_test', 'data')
            expire_stats = cache_manager.expire_cache_entries()
            self.log_test('Manual expiration works', 'linguistic_expired' in expire_stats)
            return True
        except Exception as e:
            self.log_test('Cache functionality', False, f'Error: {e}')
            return False

    async def test_cache_performance_and_metrics(self):
        """Test cache performance characteristics and metrics collection."""
        logger.info('🧪 Testing ContextCacheManager performance and metrics...')
        try:
            from prompt_improver.ml.learning.algorithms.context_cache_manager import ContextCacheManager
            cache_manager = ContextCacheManager(linguistic_cache_size=100, domain_cache_size=50, ttl_seconds=300)
            start_time = time.time()
            for i in range(50):
                cache_manager.update_linguistic_cache(f'perf_key_{i}', {'data': f'value_{i}'})
                cache_manager.update_domain_cache(f'domain_{i}', {'domain': f'type_{i}'})
            for i in range(50):
                cache_manager.get_linguistic_cache(f'perf_key_{i}')
                cache_manager.get_domain_cache(f'domain_{i}')
            end_time = time.time()
            operation_time = end_time - start_time
            self.log_test('Bulk operations complete quickly', operation_time < 1.0, f'Time: {operation_time:.3f}s')
            stats = cache_manager.get_cache_stats()
            self.log_test('Cache utilization calculated', stats['linguistic_cache_utilization'] > 0)
            self.log_test('Hit rate calculated', stats['linguistic_hit_rate'] >= 0)
            clear_stats = cache_manager.clear_all_caches()
            self.log_test('Cache clearing works', clear_stats['linguistic_cleared'] > 0)
            post_clear_stats = cache_manager.get_cache_stats()
            self.log_test('Caches empty after clearing', post_clear_stats['linguistic_cache_size'] == 0)
            return True
        except Exception as e:
            self.log_test('Performance and metrics', False, f'Error: {e}')
            return False

    async def test_integration_with_context_learner(self):
        """Test ContextCacheManager integration with ContextLearner workflow."""
        logger.info('🧪 Testing ContextCacheManager integration with ContextLearner...')
        try:
            from prompt_improver.ml.learning.algorithms.context_cache_manager import ContextCacheManager
            from prompt_improver.ml.learning.algorithms.context_learner import ContextConfig, ContextLearner
            cache_manager = ContextCacheManager()
            config = ContextConfig(cache_enabled=True)
            context_learner = ContextLearner(config)
            self.log_test('ContextCacheManager and ContextLearner coexist', True)
            linguistic_features = {'pos_tags': ['NOUN', 'VERB'], 'sentiment': 0.8}
            cache_manager.update_linguistic_cache('sample_text_hash', linguistic_features)
            cached_features = cache_manager.get_linguistic_cache('sample_text_hash')
            self.log_test('Linguistic features cached and retrieved', cached_features == linguistic_features)
            domain_info = {'domain': 'technical', 'confidence': 0.95, 'keywords': ['API', 'function']}
            cache_manager.update_domain_cache('domain_analysis_hash', domain_info)
            cached_domain = cache_manager.get_domain_cache('domain_analysis_hash')
            self.log_test('Domain info cached and retrieved', cached_domain == domain_info)
            return True
        except Exception as e:
            self.log_test('ContextLearner integration', False, f'Error: {e}')
            return False

    async def run_all_tests(self):
        """Run all integration tests."""
        logger.info('🚀 Starting ContextCacheManager Integration Tests')
        logger.info('=' * 60)
        test_methods = [self.test_orchestrator_component_loading, self.test_cache_functionality, self.test_cache_performance_and_metrics, self.test_integration_with_context_learner]
        all_passed = True
        for test_method in test_methods:
            try:
                result = await test_method()
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error('Test {test_method.__name__} failed with exception: %s', e)
                all_passed = False
        logger.info('=' * 60)
        logger.info('🏁 Test Summary')
        passed_count = sum((1 for _, passed, _ in self.test_results if passed))
        total_count = len(self.test_results)
        logger.info('Tests passed: {passed_count}/%s', total_count)
        if all_passed:
            logger.info('✅ All ContextCacheManager integration tests PASSED!')
        else:
            logger.info('❌ Some ContextCacheManager integration tests FAILED!')
        return all_passed

async def main():
    """Run the integration tests."""
    test_runner = ContextCacheManagerIntegrationTest()
    success = await test_runner.run_all_tests()
    return 0 if success else 1
if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
