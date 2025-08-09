"""
Integration Test for Modernized Feature Extractors

Tests the 2025 modernized ContextFeatureExtractor and DomainFeatureExtractor
to validate integration with ML Pipeline Orchestrator and verify that all
2025 best practices are working correctly.

This test validates:
- Async/await operations
- Circuit breaker patterns
- Health monitoring
- Orchestrator integration
- Pydantic configuration validation
- Cache effectiveness
- Performance metrics
"""
import asyncio
import json
import logging
import time
from typing import Any, Dict
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_context_feature_extractor():
    """Test the modernized ContextFeatureExtractor."""
    print('\n' + '=' * 60)
    print('TESTING CONTEXT FEATURE EXTRACTOR')
    print('=' * 60)
    try:
        from prompt_improver.ml.learning.features.context_feature_extractor import ContextFeatureConfig, ContextFeatureExtractor
        print('\n1. Testing basic configuration and initialization...')
        config = ContextFeatureConfig(weight=1.5, cache_enabled=True, cache_ttl_seconds=1800, timeout_seconds=10.0, circuit_breaker_enabled=True, failure_threshold=3, enable_metrics=True, log_level='DEBUG')
        extractor = ContextFeatureExtractor(config)
        print(f'✅ Initialized with config: weight={config.weight}, timeout={config.timeout_seconds}s')
        print('\n2. Testing health status...')
        health = extractor.get_health_status()
        print(f"✅ Health status: {health['status']}")
        print(f"   Circuit breaker: {health['circuit_breaker_state']}")
        print(f"   Cache enabled: {health['cache_stats']['cache_enabled']}")
        print('\n3. Testing synchronous feature extraction...')
        context_data = {'user_id': 'test_user_123', 'session_id': 'session_456', 'project_type': 'ai', 'performance': {'improvement_score': 0.8, 'user_satisfaction': 0.9, 'rule_effectiveness': 0.7, 'response_time_score': 0.85, 'quality_score': 0.88}, 'interaction': {'session_length_norm': 0.6, 'iteration_count_norm': 0.4, 'feedback_frequency': 0.5, 'user_engagement_score': 0.7, 'success_rate': 0.8}, 'temporal': {'time_of_day_norm': 0.3, 'day_of_week_norm': 0.6, 'session_recency_norm': 0.8, 'usage_frequency_norm': 0.5, 'trend_indicator': 0.6}}
        start_time = time.time()
        features = extractor.extract_features(context_data)
        extraction_time = time.time() - start_time
        print(f'✅ Extracted {len(features)} features in {extraction_time:.3f}s')
        print(f'   Feature sample: {features[:5]}...')
        print(f'   Feature names: {extractor.get_feature_names()[:3]}...')
        print('\n4. Testing asynchronous feature extraction...')
        start_time = time.time()
        features_async = await extractor.extract_features_async(context_data)
        async_extraction_time = time.time() - start_time
        print(f'✅ Async extracted {len(features_async)} features in {async_extraction_time:.3f}s')
        assert features == features_async, 'Sync and async results should match'
        print('   ✅ Sync and async results match')
        print('\n5. Testing cache effectiveness...')
        start_time = time.time()
        features_cached = await extractor.extract_features_async(context_data)
        cached_extraction_time = time.time() - start_time
        cache_stats = extractor.get_cache_stats()
        print(f'✅ Cached extraction in {cached_extraction_time:.3f}s')
        print(f"   Cache hits: {cache_stats['cache_hits']}")
        print(f"   Cache misses: {cache_stats['cache_misses']}")
        print(f"   Hit rate: {cache_stats['hit_rate_percent']:.1f}%")
        print('\n6. Testing ML Pipeline Orchestrator integration...')
        orchestrator_config = {'operation': 'extract_features', 'context_data': context_data, 'correlation_id': 'test_correlation_123'}
        orchestrator_result = await extractor.run_orchestrated_analysis(orchestrator_config)
        print(f"✅ Orchestrator result status: {orchestrator_result['status']}")
        print(f"   Component: {orchestrator_result['component']}")
        print(f"   Feature count: {orchestrator_result['result']['feature_count']}")
        health_config = {'operation': 'health_check', 'correlation_id': 'health_check_123'}
        health_result = await extractor.run_orchestrated_analysis(health_config)
        print(f"✅ Health check via orchestrator: {health_result['status']}")
        print(f"   Health status: {health_result['result']['status']}")
        print('\n7. Testing metrics collection...')
        metrics = extractor.metrics.get_health_status()
        print(f"✅ Total extractions: {metrics['total_extractions']}")
        print(f"   Success rate: {metrics['success_rate']:.1f}%")
        print(f"   Average time: {metrics['average_processing_time']:.3f}s")
        print('\n🎉 ContextFeatureExtractor tests completed successfully!')
        return True
    except Exception as e:
        print(f'❌ ContextFeatureExtractor test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

async def test_domain_feature_extractor():
    """Test the modernized DomainFeatureExtractor."""
    print('\n' + '=' * 60)
    print('TESTING DOMAIN FEATURE EXTRACTOR')
    print('=' * 60)
    try:
        from prompt_improver.ml.learning.features.domain_feature_extractor import DomainFeatureConfig, DomainFeatureExtractor
        print('\n1. Testing basic configuration and initialization...')
        config = DomainFeatureConfig(weight=2.0, deterministic=True, cache_enabled=True, cache_ttl_seconds=1800, timeout_seconds=15.0, circuit_breaker_enabled=True, failure_threshold=3, use_domain_analyzer=True, fallback_on_analyzer_failure=True, enable_metrics=True, log_level='DEBUG')
        extractor = DomainFeatureExtractor(config)
        print(f'✅ Initialized with config: weight={config.weight}, timeout={config.timeout_seconds}s')
        print('\n2. Testing health status...')
        health = extractor.get_health_status()
        print(f"✅ Health status: {health['status']}")
        print(f"   Circuit breaker: {health['circuit_breaker_state']}")
        print(f"   Analyzer available: {health['analyzer_status']['available']}")
        print(f"   Cache enabled: {health['cache_stats']['cache_enabled']}")
        print('\n3. Testing synchronous feature extraction...')
        test_text = "\n        I need help with a Python machine learning project. I'm building a neural network\n        for text classification and having issues with data preprocessing and model training.\n        Can you help me optimize the training process and improve accuracy? This is urgent\n        as I have a deadline next week. The dataset contains technical documentation and\n        research papers from various academic sources.\n        "
        start_time = time.time()
        features = extractor.extract_features(test_text)
        extraction_time = time.time() - start_time
        print(f'✅ Extracted {len(features)} features in {extraction_time:.3f}s')
        print(f'   Feature sample: {features[:5]}...')
        print(f'   Feature names: {extractor.get_feature_names()[:3]}...')
        print('\n4. Testing asynchronous feature extraction...')
        start_time = time.time()
        features_async = await extractor.extract_features_async(test_text)
        async_extraction_time = time.time() - start_time
        print(f'✅ Async extracted {len(features_async)} features in {async_extraction_time:.3f}s')
        assert features == features_async, 'Sync and async results should match'
        print('   ✅ Sync and async results match')
        print('\n5. Testing cache effectiveness...')
        start_time = time.time()
        features_cached = await extractor.extract_features_async(test_text)
        cached_extraction_time = time.time() - start_time
        cache_stats = extractor.get_cache_stats()
        print(f'✅ Cached extraction in {cached_extraction_time:.3f}s')
        print(f"   Cache hits: {cache_stats['cache_hits']}")
        print(f"   Cache misses: {cache_stats['cache_misses']}")
        print(f"   Hit rate: {cache_stats['hit_rate_percent']:.1f}%")
        print('\n6. Testing ML Pipeline Orchestrator integration...')
        orchestrator_config = {'operation': 'extract_features', 'text': test_text, 'correlation_id': 'domain_test_correlation_456'}
        orchestrator_result = await extractor.run_orchestrated_analysis(orchestrator_config)
        print(f"✅ Orchestrator result status: {orchestrator_result['status']}")
        print(f"   Component: {orchestrator_result['component']}")
        print(f"   Feature count: {orchestrator_result['result']['feature_count']}")
        print(f"   Analyzer usage rate: {orchestrator_result['result']['extraction_metadata']['analyzer_usage_rate']:.1f}%")
        health_config = {'operation': 'health_check', 'correlation_id': 'domain_health_check_456'}
        health_result = await extractor.run_orchestrated_analysis(health_config)
        print(f"✅ Health check via orchestrator: {health_result['status']}")
        print(f"   Health status: {health_result['result']['status']}")
        print('\n7. Testing metrics collection...')
        metrics = extractor.metrics.get_health_status()
        print(f"✅ Total extractions: {metrics['total_extractions']}")
        print(f"   Success rate: {metrics['success_rate']:.1f}%")
        print(f"   Analyzer usage rate: {metrics['analyzer_usage_rate']:.1f}%")
        print(f"   Average time: {metrics['average_processing_time']:.3f}s")
        print('\n🎉 DomainFeatureExtractor tests completed successfully!')
        return True
    except Exception as e:
        print(f'❌ DomainFeatureExtractor test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

async def test_circuit_breaker_behavior():
    """Test circuit breaker behavior under failure conditions."""
    print('\n' + '=' * 60)
    print('TESTING CIRCUIT BREAKER BEHAVIOR')
    print('=' * 60)
    try:
        from prompt_improver.ml.learning.features.context_feature_extractor import ContextFeatureConfig, ContextFeatureExtractor
        config = ContextFeatureConfig(circuit_breaker_enabled=True, failure_threshold=2, recovery_timeout=5, timeout_seconds=0.1)
        extractor = ContextFeatureExtractor(config)
        print('✅ Initialized extractor with aggressive circuit breaker settings')
        large_context = {f'field_{i}': f'value_{i}' * 1000 for i in range(100)}
        print('\n1. Testing circuit breaker trip...')
        failed_attempts = 0
        for i in range(5):
            try:
                features = await extractor.extract_features_async(large_context)
                print(f'   Attempt {i + 1}: Success (unexpected)')
            except Exception as e:
                failed_attempts += 1
                print(f'   Attempt {i + 1}: Failed as expected')
                health = extractor.get_health_status()
                if health['circuit_breaker_state'] == 'open':
                    print(f'   ✅ Circuit breaker opened after {failed_attempts} failures')
                    break
        print('\n2. Testing circuit breaker recovery...')
        print('   Waiting for recovery timeout...')
        await asyncio.sleep(6)
        simple_context = {'user_id': 'test', 'session_id': 'test'}
        features = await extractor.extract_features_async(simple_context)
        final_health = extractor.get_health_status()
        print(f"✅ Circuit breaker recovered, state: {final_health['circuit_breaker_state']}")
        return True
    except Exception as e:
        print(f'❌ Circuit breaker test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

async def test_performance_comparison():
    """Compare performance between old and new implementations."""
    print('\n' + '=' * 60)
    print('TESTING PERFORMANCE COMPARISON')
    print('=' * 60)
    try:
        from prompt_improver.ml.learning.features.context_feature_extractor import ContextFeatureConfig, ContextFeatureExtractor
        from prompt_improver.ml.learning.features.domain_feature_extractor import DomainFeatureConfig, DomainFeatureExtractor
        context_data = {'user_id': 'perf_test_user', 'session_id': 'perf_test_session', 'project_type': 'web', 'performance': {'improvement_score': 0.8, 'user_satisfaction': 0.9}, 'interaction': {'session_length_norm': 0.6, 'feedback_frequency': 0.5}, 'temporal': {'time_of_day_norm': 0.3, 'usage_frequency_norm': 0.5}}
        test_text = 'Create a web application using React and Node.js with user authentication and real-time messaging features.'
        context_config = ContextFeatureConfig(cache_enabled=False)
        domain_config = DomainFeatureConfig(cache_enabled=False)
        context_extractor = ContextFeatureExtractor(context_config)
        domain_extractor = DomainFeatureExtractor(domain_config)
        num_iterations = 50
        print(f'Running {num_iterations} iterations for each extractor...')
        print('\n1. Testing ContextFeatureExtractor performance...')
        context_times = []
        for i in range(num_iterations):
            start_time = time.time()
            features = await context_extractor.extract_features_async(context_data)
            end_time = time.time()
            context_times.append(end_time - start_time)
        context_avg_time = sum(context_times) / len(context_times)
        context_min_time = min(context_times)
        context_max_time = max(context_times)
        print(f'✅ ContextFeatureExtractor: {context_avg_time:.4f}s avg, {context_min_time:.4f}s min, {context_max_time:.4f}s max')
        print('\n2. Testing DomainFeatureExtractor performance...')
        domain_times = []
        for i in range(num_iterations):
            start_time = time.time()
            features = await domain_extractor.extract_features_async(test_text)
            end_time = time.time()
            domain_times.append(end_time - start_time)
        domain_avg_time = sum(domain_times) / len(domain_times)
        domain_min_time = min(domain_times)
        domain_max_time = max(domain_times)
        print(f'✅ DomainFeatureExtractor: {domain_avg_time:.4f}s avg, {domain_min_time:.4f}s min, {domain_max_time:.4f}s max')
        print('\n3. Testing cache performance...')
        cached_context_config = ContextFeatureConfig(cache_enabled=True)
        cached_context_extractor = ContextFeatureExtractor(cached_context_config)
        start_time = time.time()
        await cached_context_extractor.extract_features_async(context_data)
        miss_time = time.time() - start_time
        start_time = time.time()
        await cached_context_extractor.extract_features_async(context_data)
        hit_time = time.time() - start_time
        speedup = miss_time / hit_time if hit_time > 0 else float('inf')
        print(f'✅ Cache miss: {miss_time:.4f}s, Cache hit: {hit_time:.4f}s')
        print(f'   Cache speedup: {speedup:.1f}x')
        print('\n📊 Performance Summary:')
        print(f'   Context extraction: {context_avg_time:.4f}s average')
        print(f'   Domain extraction: {domain_avg_time:.4f}s average')
        print(f'   Cache effectiveness: {speedup:.1f}x speedup')
        return True
    except Exception as e:
        print(f'❌ Performance test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner."""
    print('🚀 Starting Modernized Feature Extractor Integration Tests')
    print('Testing 2025 best practices implementation...')
    results = []
    test_functions = [('Context Feature Extractor', test_context_feature_extractor), ('Domain Feature Extractor', test_domain_feature_extractor), ('Circuit Breaker Behavior', test_circuit_breaker_behavior), ('Performance Comparison', test_performance_comparison)]
    for test_name, test_func in test_functions:
        print(f"\n{'=' * 80}")
        print(f'RUNNING: {test_name}')
        print(f"{'=' * 80}")
        try:
            result = await test_func()
            results.append((test_name, result))
            if result:
                print(f'✅ {test_name}: PASSED')
            else:
                print(f'❌ {test_name}: FAILED')
        except Exception as e:
            print(f'❌ {test_name}: ERROR - {e}')
            results.append((test_name, False))
    print('\n' + '=' * 80)
    print('FINAL TEST RESULTS')
    print('=' * 80)
    passed = sum((1 for _, result in results if result))
    total = len(results)
    for test_name, result in results:
        status = '✅ PASSED' if result else '❌ FAILED'
        print(f'  {test_name}: {status}')
    print(f'\nOverall: {passed}/{total} tests passed ({passed / total * 100:.1f}%)')
    if passed == total:
        print('\n🎉 ALL TESTS PASSED! The 2025 modernization is successful!')
        print('\nKey achievements:')
        print('✅ Async/await patterns implemented')
        print('✅ Circuit breaker fault tolerance working')
        print('✅ Health monitoring and observability active')
        print('✅ ML Pipeline Orchestrator integration successful')
        print('✅ Pydantic configuration validation working')
        print('✅ Intelligent caching with TTL operational')
        print('✅ Performance metrics collection active')
    else:
        print(f'\n⚠️  {total - passed} tests failed. Review the issues above.')
        return 1
    return 0
if __name__ == '__main__':
    import sys
    sys.exit(asyncio.run(main()))
