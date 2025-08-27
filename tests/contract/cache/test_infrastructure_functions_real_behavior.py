"""Real Behavior Testing for Test Infrastructure Functions Cache Migration

This module provides comprehensive real behavior tests for the 4 test infrastructure functions
migrated from @lru_cache to module-level caching in conftest.py.

Test Infrastructure Functions Tested (4 functions):
1. get_database_models() - Lazy database model loading
2. detect_external_redis() - Docker container Redis detection
3. check_ml_libraries() - ML library availability checking
4. lazy_import() - Lazy import utility with caching

SLA Targets (from baseline):
- Cache Hit: ≤ 0.1ms
- Cache Miss: ≤ 1.0ms
- Hit Rate: ≥ 95%

These functions use simple module-level caching instead of @lru_cache for better
test infrastructure performance and explicit cache management.
"""

import time

import pytest
from tests.contract.cache.test_lru_migration_real_behavior import RealBehaviorTestBase
from tests.fixtures.application.cache import reset_test_caches
from tests.fixtures.foundation.utils import (
    check_ml_libraries,
    detect_external_redis,
    get_cache_status,
    get_database_models,
    lazy_import,
)


@pytest.mark.real_behavior
@pytest.mark.cache_migration
@pytest.mark.test_infrastructure
class TestInfrastructureFunctionsCacheMigration(RealBehaviorTestBase):
    """Real behavior tests for Test Infrastructure functions cache migration.

    Tests all 4 test infrastructure functions migrated from @lru_cache to
    simple module-level caching for better performance and explicit management.
    """

    def setup_method(self):
        """Setup for test infrastructure tests."""
        super().setup_method()
        # Clear all test caches to ensure clean state
        reset_test_caches()

    def teardown_method(self):
        """Cleanup for test infrastructure tests."""
        super().teardown_method()

        # Log cache status for debugging
        cache_status = get_cache_status()
        self.logger.info(f"Final cache status: {cache_status}")

        # Clear caches for test isolation
        reset_test_caches()

    def test_get_database_models_real_behavior(self):
        """Test get_database_models() with real module-level caching."""
        # First call should be cache miss (slower due to SQLAlchemy imports)
        result1, duration1 = self.measure_performance(
            "get_database_models_miss", get_database_models
        )

        # Validate result structure
        assert isinstance(result1, dict), f"Expected dict result, got {type(result1)}"

        # Should contain expected database models
        expected_models = {
            'ABExperiment', 'ImprovementSession', 'RuleIntelligenceCache',
            'RuleMetadata', 'RulePerformance', 'SQLModel', 'UserFeedback'
        }

        result_keys = set(result1.keys())
        common_models = expected_models.intersection(result_keys)
        assert len(common_models) >= 5, f"Expected at least 5 database models, got {common_models}"

        # Validate model types (should be SQLAlchemy/SQLModel classes)
        for model_name, model_class in result1.items():
            assert hasattr(model_class, '__name__'), f"Model {model_name} should be a class"
            assert hasattr(model_class, '__module__'), f"Model {model_name} should have module info"

        # First call should be slower (importing SQLAlchemy models)
        assert duration1 >= 0.001, f"First call too fast: {duration1}ms (expected import overhead)"

        # Second call should be cache hit (much faster)
        result2, duration2 = self.measure_performance(
            "get_database_models_hit", get_database_models
        )

        # Results should be identical (same dict reference due to module-level caching)
        assert result1 is result2, "Module-level cache should return same dict reference"

        # Cache hit should meet test infrastructure SLA
        self.validate_cache_hit_performance(duration2, "test")

        # Cache hit should be significantly faster
        assert duration2 < duration1, f"Cache hit ({duration2:.6f}ms) should be faster than miss ({duration1:.6f}ms)"

        self.logger.info(f"Database Models: miss={duration1:.6f}ms, hit={duration2:.6f}ms")
        self.logger.info(f"Models loaded: {list(result1.keys())}")

    def test_detect_external_redis_real_behavior(self):
        """Test detect_external_redis() with real module-level caching."""
        # First call should be cache miss (potentially slow due to Docker subprocess)
        result1, duration1 = self.measure_performance(
            "detect_external_redis_miss", detect_external_redis
        )

        # Validate result structure
        assert isinstance(result1, tuple), f"Expected tuple result, got {type(result1)}"
        assert len(result1) == 2, f"Expected 2-tuple (host, port), got {len(result1)}-tuple"

        # Should be (host, port) tuple where both are strings or both are None
        host, port = result1
        assert (host is None and port is None) or (isinstance(host, str) and isinstance(port, str)), \
            f"Expected (str, str) or (None, None), got ({type(host)}, {type(port)})"

        if host and port:
            self.logger.info(f"External Redis detected: {host}:{port}")
        else:
            self.logger.info("No external Redis container detected")

        # Second call should be cache hit (much faster)
        result2, duration2 = self.measure_performance(
            "detect_external_redis_hit", detect_external_redis
        )

        # Results should be identical
        assert result1 == result2, f"Cache hit result mismatch: {result1} != {result2}"

        # Cache hit should meet test infrastructure SLA
        self.validate_cache_hit_performance(duration2, "test")

        # Cache hit should be faster (especially if Docker calls were made)
        assert duration2 <= duration1, f"Cache hit ({duration2:.6f}ms) should not be slower than miss ({duration1:.6f}ms)"

        self.logger.info(f"Redis Detection: miss={duration1:.6f}ms, hit={duration2:.6f}ms, result={result1}")

    def test_check_ml_libraries_real_behavior(self):
        """Test check_ml_libraries() with real module-level caching."""
        # First call should be cache miss (slower due to import attempts)
        result1, duration1 = self.measure_performance(
            "check_ml_libraries_miss", check_ml_libraries
        )

        # Validate result structure
        assert isinstance(result1, dict), f"Expected dict result, got {type(result1)}"

        # Should contain expected ML libraries
        expected_libs = {'sklearn', 'deap', 'pymc', 'umap', 'hdbscan'}
        result_keys = set(result1.keys())
        common_libs = expected_libs.intersection(result_keys)
        assert len(common_libs) >= 3, f"Expected at least 3 ML libraries checked, got {common_libs}"

        # All values should be boolean
        for lib_name, available in result1.items():
            assert isinstance(available, bool), f"Library {lib_name} availability should be bool, got {type(available)}"

        # Log library availability
        available_libs = [lib for lib, available in result1.items() if available]
        unavailable_libs = [lib for lib, available in result1.items() if not available]
        self.logger.info(f"Available ML libraries: {available_libs}")
        self.logger.info(f"Unavailable ML libraries: {unavailable_libs}")

        # Second call should be cache hit (much faster)
        result2, duration2 = self.measure_performance(
            "check_ml_libraries_hit", check_ml_libraries
        )

        # Results should be identical
        assert result1 == result2, f"Cache hit result mismatch: {result1} != {result2}"

        # Cache hit should meet test infrastructure SLA
        self.validate_cache_hit_performance(duration2, "test")

        # Cache hit should be faster
        assert duration2 < duration1, f"Cache hit ({duration2:.6f}ms) should be faster than miss ({duration1:.6f}ms)"

        self.logger.info(f"ML Libraries: miss={duration1:.6f}ms, hit={duration2:.6f}ms")

    def test_lazy_import_real_behavior(self):
        """Test lazy_import() utility with real module-level caching."""
        import_tests = [
            ('os_module', 'os', None),
            ('sys_module', 'sys', None),
            ('json_dumps', 'json', 'dumps'),
            ('time_sleep', 'time', 'sleep'),
        ]

        for test_name, module_name, attribute in import_tests:
            self.logger.info(f"Testing lazy import: {test_name}")

            # Create lazy importer
            lazy_importer = lazy_import(module_name, attribute)

            # First call should be cache miss (import overhead)
            result1, duration1 = self.measure_performance(
                f"lazy_import_{test_name}_miss", lazy_importer
            )

            # Validate result
            if attribute:
                # Should be a specific attribute from the module
                assert callable(result1) or hasattr(result1, '__name__'), \
                    f"Expected callable or named object for {module_name}.{attribute}"
            else:
                # Should be the module itself
                assert hasattr(result1, '__name__'), f"Expected module object for {module_name}"
                assert result1.__name__ == module_name, f"Module name mismatch: {result1.__name__} != {module_name}"

            # Second call should be cache hit (faster)
            result2, duration2 = self.measure_performance(
                f"lazy_import_{test_name}_hit", lazy_importer
            )

            # Results should be identical (same object reference)
            assert result1 is result2, f"Lazy import cache should return same object reference for {test_name}"

            # Cache hit should meet test infrastructure SLA
            self.validate_cache_hit_performance(duration2, "test")

            # Cache hit should be faster
            assert duration2 < duration1, f"Cache hit ({duration2:.6f}ms) should be faster than miss ({duration1:.6f}ms) for {test_name}"

            self.logger.info(f"Lazy import {test_name}: miss={duration1:.6f}ms, hit={duration2:.6f}ms")

    def test_lazy_import_error_handling(self):
        """Test lazy_import() error handling with invalid modules."""
        # Test with non-existent module
        invalid_importer = lazy_import('nonexistent_module_12345')

        with pytest.raises(ImportError, match="Failed to import nonexistent_module_12345"):
            invalid_importer()

        # Test with non-existent attribute
        invalid_attr_importer = lazy_import('os', 'nonexistent_attribute_12345')

        with pytest.raises((ImportError, AttributeError)):
            invalid_attr_importer()

    def test_cache_isolation_and_reset(self):
        """Test cache isolation and reset functionality."""
        # Prime all caches
        models1 = get_database_models()
        redis1 = detect_external_redis()
        ml_libs1 = check_ml_libraries()

        # Check cache status
        status_before = get_cache_status()
        self.logger.info(f"Cache status before reset: {status_before}")

        # All caches should be populated
        assert status_before['database_models_loaded'], "Database models cache should be loaded"
        assert status_before['redis_detection_cached'], "Redis detection cache should be loaded"
        assert status_before['ml_libraries_cached'], "ML libraries cache should be loaded"
        assert status_before['import_cache_size'] > 0, "Import cache should have entries"

        # Reset caches
        reset_test_caches()

        # Check cache status after reset
        status_after = get_cache_status()
        self.logger.info(f"Cache status after reset: {status_after}")

        # All caches should be cleared
        assert not status_after['database_models_loaded'], "Database models cache should be cleared"
        assert not status_after['redis_detection_cached'], "Redis detection cache should be cleared"
        assert not status_after['ml_libraries_cached'], "ML libraries cache should be cleared"
        assert status_after['import_cache_size'] == 0, "Import cache should be empty"

        # Next calls should be cache misses again (slower)
        start_time = time.time() * 1000
        models2 = get_database_models()
        models_time = (time.time() * 1000) - start_time

        start_time = time.time() * 1000
        redis2 = detect_external_redis()
        redis_time = (time.time() * 1000) - start_time

        start_time = time.time() * 1000
        ml_libs2 = check_ml_libraries()
        ml_libs_time = (time.time() * 1000) - start_time

        # Results should be the same as before reset
        assert models1 == models2, "Database models should be identical after cache reset"
        assert redis1 == redis2, "Redis detection should be identical after cache reset"
        assert ml_libs1 == ml_libs2, "ML libraries should be identical after cache reset"

        # Timing should indicate cache misses (slower than hits)
        self.logger.info(f"After reset - models: {models_time:.6f}ms, redis: {redis_time:.6f}ms, ml: {ml_libs_time:.6f}ms")

    def test_infrastructure_functions_concurrent_access(self):
        """Test infrastructure functions under concurrent access."""
        import queue
        import threading

        results_queue = queue.Queue()
        errors_queue = queue.Queue()

        def worker(worker_id: int):
            try:
                # Access all infrastructure functions
                models = get_database_models()
                redis = detect_external_redis()
                ml_libs = check_ml_libraries()

                # Test lazy import
                json_importer = lazy_import('json', 'dumps')
                json_dumps = json_importer()

                results_queue.put({
                    'worker_id': worker_id,
                    'models_count': len(models),
                    'redis_result': redis,
                    'ml_libs_count': len(ml_libs),
                    'json_dumps_available': callable(json_dumps),
                })

            except Exception as e:
                errors_queue.put(f"Worker {worker_id}: {e}")

        # Run 10 concurrent workers
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        errors = []
        while not errors_queue.empty():
            errors.append(errors_queue.get())

        # Validate results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"

        # All workers should get identical results (same cached objects)
        first_result = results[0]
        for result in results[1:]:
            assert result['models_count'] == first_result['models_count'], "Models count should be consistent"
            assert result['redis_result'] == first_result['redis_result'], "Redis result should be consistent"
            assert result['ml_libs_count'] == first_result['ml_libs_count'], "ML libs count should be consistent"
            assert result['json_dumps_available'] == first_result['json_dumps_available'], "JSON dumps should be consistent"

        self.logger.info(f"Concurrent access test: {len(results)} workers successful, 0 errors")

    def test_infrastructure_performance_characteristics(self):
        """Test performance characteristics of infrastructure functions."""
        # Clear caches for clean measurement
        reset_test_caches()

        # Test multiple cache miss -> hit cycles
        functions = [
            ('database_models', get_database_models),
            ('redis_detection', detect_external_redis),
            ('ml_libraries', check_ml_libraries),
        ]

        performance_data = {}

        for func_name, func in functions:
            # Single cache miss
            _, miss_time = self.measure_performance(f"{func_name}_miss", func)

            # Multiple cache hits
            hit_times = []
            for i in range(5):
                _, hit_time = self.measure_performance(f"{func_name}_hit_{i}", func)
                hit_times.append(hit_time)
                self.validate_cache_hit_performance(hit_time, "test")

            avg_hit_time = sum(hit_times) / len(hit_times)
            hit_variance = sum((t - avg_hit_time) ** 2 for t in hit_times) / len(hit_times)

            performance_data[func_name] = {
                'miss_time': miss_time,
                'avg_hit_time': avg_hit_time,
                'hit_variance': hit_variance,
                'speedup_ratio': miss_time / avg_hit_time if avg_hit_time > 0 else 0
            }

            # All hits should meet SLA
            assert avg_hit_time <= 0.1, f"Average {func_name} hit time {avg_hit_time:.6f}ms > 0.1ms SLA"

            # Clear cache for next function test
            reset_test_caches()

        # Log performance summary
        for func_name, data in performance_data.items():
            self.logger.info(
                f"{func_name}: miss={data['miss_time']:.6f}ms, "
                f"avg_hit={data['avg_hit_time']:.6f}ms, "
                f"speedup={data['speedup_ratio']:.1f}x"
            )

        # Verify meaningful performance improvement from caching
        for func_name, data in performance_data.items():
            assert data['speedup_ratio'] >= 2.0, f"{func_name} cache speedup {data['speedup_ratio']:.1f}x < 2.0x minimum"

    def test_infrastructure_cache_memory_efficiency(self):
        """Test that infrastructure caches don't consume excessive memory."""
        # Clear caches
        reset_test_caches()

        # Prime all caches
        models = get_database_models()
        redis = detect_external_redis()
        ml_libs = check_ml_libraries()

        # Create several lazy importers
        importers = []
        for module in ['os', 'sys', 'json', 'time', 'random']:
            importer = lazy_import(module)
            imported_module = importer()
            importers.append((importer, imported_module))

        # Check cache status
        status = get_cache_status()

        # Caches should be populated but not excessive
        assert status['database_models_loaded'], "Database models should be cached"
        assert status['redis_detection_cached'], "Redis detection should be cached"
        assert status['ml_libraries_cached'], "ML libraries should be cached"
        assert 1 <= status['import_cache_size'] <= 20, f"Import cache size {status['import_cache_size']} should be reasonable"

        # Verify that repeated calls don't increase cache size
        original_cache_size = status['import_cache_size']

        # Call functions multiple times
        for _ in range(5):
            get_database_models()
            detect_external_redis()
            check_ml_libraries()

            for importer, _ in importers:
                importer()

        # Cache size should remain stable
        new_status = get_cache_status()
        assert new_status['import_cache_size'] == original_cache_size, \
            f"Import cache size increased from {original_cache_size} to {new_status['import_cache_size']}"

        self.logger.info(f"Cache memory efficiency validated - stable size: {original_cache_size}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-k", "infrastructure"])
