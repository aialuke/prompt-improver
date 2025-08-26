"""Real Behavior Testing for TextStat Functions Cache Migration

This module provides comprehensive real behavior tests for all 9 TextStat functions
migrated from @lru_cache to the unified cache infrastructure.

TextStat Functions Tested (9 functions):
1. flesch_reading_ease() - Readability scoring calculation
2. flesch_kincaid_grade() - Grade level calculation
3. syllable_count() - Syllable counting with CMUdict optimization
4. sentence_count() - Sentence counting
5. lexicon_count() - Word counting with punctuation handling
6. automated_readability_index() - ARI calculation
7. coleman_liau_index() - Coleman-Liau calculation
8. gunning_fog() - Gunning Fog index calculation
9. smog_index() - SMOG index calculation

SLA Targets (from baseline):
- Cache Hit: ≤ 0.5ms
- Cache Miss: ≤ 50ms
- Hit Rate: ≥ 80%

Based on baseline showing sub-millisecond cache hit performance.
"""

import time

import pytest
from tests.contract.cache.test_lru_migration_real_behavior import RealBehaviorTestBase

from prompt_improver.core.config.textstat import TextStatConfig, TextStatWrapper


@pytest.mark.real_behavior
@pytest.mark.cache_migration
@pytest.mark.textstat
class TestTextStatFunctionsCacheMigration(RealBehaviorTestBase):
    """Real behavior tests for TextStat functions cache migration.

    Tests all 9 TextStat functions migrated from @lru_cache to unified cache.
    Validates performance, cache behavior, and integration with real text analysis.
    """

    # Test texts for different complexity levels
    TEST_TEXTS = {
        'simple': "This is a simple test sentence.",
        'medium': "This is a medium complexity sentence with several clauses, designed to test readability metrics and provide more realistic text analysis scenarios for comprehensive validation.",
        'complex': "In the intricate and multifaceted domain of computational linguistics and natural language processing, sophisticated algorithms and methodologies are continuously being developed and refined to enhance the accuracy and effectiveness of automated text analysis systems, particularly in the context of readability assessment and linguistic complexity evaluation.",
        'technical': "The machine learning model utilizes transformer architecture with multi-head attention mechanisms to process sequential data, implementing gradient descent optimization with adaptive learning rates and regularization techniques including dropout and batch normalization.",
        'empty': "",
        'single_word': "Test",
        'punctuation_heavy': "Hello! How are you? I'm fine, thanks. What about you? That's great! See you later.",
    }

    def setup_method(self):
        """Setup for TextStat tests."""
        super().setup_method()

        # Create fresh TextStat wrapper for each test to ensure clean cache state
        self.textstat_config = TextStatConfig(
            language="en_US",
            enable_caching=True,
            cache_size=2000,
            suppress_warnings=True,
            enable_metrics=True
        )
        self.textstat_wrapper = TextStatWrapper(self.textstat_config)

        # Clear any existing cache state
        try:
            self.textstat_wrapper.clear_cache()
        except Exception as e:
            self.logger.warning(f"Failed to clear TextStat cache in setup: {e}")

    def teardown_method(self):
        """Cleanup for TextStat tests."""
        super().teardown_method()

        # Get performance metrics for logging
        try:
            metrics = self.textstat_wrapper.get_metrics()
            self.logger.info(f"TextStat performance metrics: {metrics.get('metrics', {})}")
        except Exception as e:
            self.logger.warning(f"Failed to get TextStat metrics: {e}")

    def test_flesch_reading_ease_real_behavior(self):
        """Test flesch_reading_ease() with real cache behavior."""
        text = self.TEST_TEXTS['medium']

        # First call should be cache miss (slower)
        result1, duration1 = self.measure_performance(
            "flesch_reading_ease_miss", self.textstat_wrapper.flesch_reading_ease, text
        )

        # Validate result
        assert isinstance(result1, (int, float)), f"Expected numeric result, got {type(result1)}"
        assert 0 <= result1 <= 100, f"Flesch Reading Ease should be 0-100, got {result1}"

        # Second call should be cache hit (faster)
        result2, duration2 = self.measure_performance(
            "flesch_reading_ease_hit", self.textstat_wrapper.flesch_reading_ease, text
        )

        # Results should be identical
        assert result1 == result2, f"Cache hit result mismatch: {result1} != {result2}"

        # Cache hit should meet SLA
        self.validate_cache_hit_performance(duration2, "textstat")

        # Cache hit should be significantly faster
        assert duration2 < duration1, f"Cache hit ({duration2:.6f}ms) should be faster than miss ({duration1:.6f}ms)"

        self.logger.info(f"Flesch Reading Ease: miss={duration1:.6f}ms, hit={duration2:.6f}ms, result={result1}")

    def test_flesch_kincaid_grade_real_behavior(self):
        """Test flesch_kincaid_grade() with real cache behavior."""
        text = self.TEST_TEXTS['medium']

        # First call should be cache miss
        result1, duration1 = self.measure_performance(
            "flesch_kincaid_grade_miss", self.textstat_wrapper.flesch_kincaid_grade, text
        )

        # Validate result
        assert isinstance(result1, (int, float)), f"Expected numeric result, got {type(result1)}"
        assert result1 >= 0, f"Flesch-Kincaid grade should be positive, got {result1}"

        # Second call should be cache hit
        result2, duration2 = self.measure_performance(
            "flesch_kincaid_grade_hit", self.textstat_wrapper.flesch_kincaid_grade, text
        )

        # Results should be identical
        assert result1 == result2, f"Cache hit result mismatch: {result1} != {result2}"

        # Cache hit should meet SLA
        self.validate_cache_hit_performance(duration2, "textstat")

        self.logger.info(f"Flesch-Kincaid Grade: miss={duration1:.6f}ms, hit={duration2:.6f}ms, result={result1}")

    def test_syllable_count_real_behavior(self):
        """Test syllable_count() with real cache behavior and CMUdict optimization."""
        text = self.TEST_TEXTS['complex']

        # First call should be cache miss
        result1, duration1 = self.measure_performance(
            "syllable_count_miss", self.textstat_wrapper.syllable_count, text
        )

        # Validate result
        assert isinstance(result1, int), f"Expected integer result, got {type(result1)}"
        assert result1 > 0, f"Syllable count should be positive for non-empty text, got {result1}"

        # Second call should be cache hit
        result2, duration2 = self.measure_performance(
            "syllable_count_hit", self.textstat_wrapper.syllable_count, text
        )

        # Results should be identical
        assert result1 == result2, f"Cache hit result mismatch: {result1} != {result2}"

        # Cache hit should meet SLA
        self.validate_cache_hit_performance(duration2, "textstat")

        self.logger.info(f"Syllable Count: miss={duration1:.6f}ms, hit={duration2:.6f}ms, result={result1}")

    def test_sentence_count_real_behavior(self):
        """Test sentence_count() with real cache behavior."""
        text = self.TEST_TEXTS['punctuation_heavy']

        # First call should be cache miss
        result1, duration1 = self.measure_performance(
            "sentence_count_miss", self.textstat_wrapper.sentence_count, text
        )

        # Validate result
        assert isinstance(result1, int), f"Expected integer result, got {type(result1)}"
        assert result1 > 0, f"Sentence count should be positive for non-empty text, got {result1}"

        # Second call should be cache hit
        result2, duration2 = self.measure_performance(
            "sentence_count_hit", self.textstat_wrapper.sentence_count, text
        )

        # Results should be identical
        assert result1 == result2, f"Cache hit result mismatch: {result1} != {result2}"

        # Cache hit should meet SLA
        self.validate_cache_hit_performance(duration2, "textstat")

        self.logger.info(f"Sentence Count: miss={duration1:.6f}ms, hit={duration2:.6f}ms, result={result1}")

    def test_lexicon_count_real_behavior(self):
        """Test lexicon_count() with real cache behavior and parameter variations."""
        text = self.TEST_TEXTS['technical']

        # Test with default parameters (removepunct=True)
        result1, duration1 = self.measure_performance(
            "lexicon_count_default_miss", self.textstat_wrapper.lexicon_count, text
        )

        # Validate result
        assert isinstance(result1, int), f"Expected integer result, got {type(result1)}"
        assert result1 > 0, f"Lexicon count should be positive for non-empty text, got {result1}"

        # Cache hit with same parameters
        result2, duration2 = self.measure_performance(
            "lexicon_count_default_hit", self.textstat_wrapper.lexicon_count, text
        )

        assert result1 == result2, f"Cache hit result mismatch: {result1} != {result2}"
        self.validate_cache_hit_performance(duration2, "textstat")

        # Test with different parameters (removepunct=False) - should be cache miss
        result3, duration3 = self.measure_performance(
            "lexicon_count_nopunct_miss", self.textstat_wrapper.lexicon_count, text, False
        )

        # Should be different result due to different parameters
        assert isinstance(result3, int), f"Expected integer result, got {type(result3)}"

        # Cache hit with same new parameters
        result4, duration4 = self.measure_performance(
            "lexicon_count_nopunct_hit", self.textstat_wrapper.lexicon_count, text, False
        )

        assert result3 == result4, f"Cache hit result mismatch: {result3} != {result4}"
        self.validate_cache_hit_performance(duration4, "textstat")

        self.logger.info(f"Lexicon Count: default={result1} ({duration1:.6f}ms miss, {duration2:.6f}ms hit), "
                        f"nopunct={result3} ({duration3:.6f}ms miss, {duration4:.6f}ms hit)")

    def test_automated_readability_index_real_behavior(self):
        """Test automated_readability_index() with real cache behavior."""
        text = self.TEST_TEXTS['medium']

        # First call should be cache miss
        result1, duration1 = self.measure_performance(
            "automated_readability_index_miss", self.textstat_wrapper.automated_readability_index, text
        )

        # Validate result
        assert isinstance(result1, (int, float)), f"Expected numeric result, got {type(result1)}"
        assert result1 >= 0, f"ARI should be positive, got {result1}"

        # Second call should be cache hit
        result2, duration2 = self.measure_performance(
            "automated_readability_index_hit", self.textstat_wrapper.automated_readability_index, text
        )

        # Results should be identical
        assert result1 == result2, f"Cache hit result mismatch: {result1} != {result2}"

        # Cache hit should meet SLA
        self.validate_cache_hit_performance(duration2, "textstat")

        self.logger.info(f"ARI: miss={duration1:.6f}ms, hit={duration2:.6f}ms, result={result1}")

    def test_coleman_liau_index_real_behavior(self):
        """Test coleman_liau_index() with real cache behavior."""
        text = self.TEST_TEXTS['complex']

        # First call should be cache miss
        result1, duration1 = self.measure_performance(
            "coleman_liau_index_miss", self.textstat_wrapper.coleman_liau_index, text
        )

        # Validate result
        assert isinstance(result1, (int, float)), f"Expected numeric result, got {type(result1)}"

        # Second call should be cache hit
        result2, duration2 = self.measure_performance(
            "coleman_liau_index_hit", self.textstat_wrapper.coleman_liau_index, text
        )

        # Results should be identical
        assert result1 == result2, f"Cache hit result mismatch: {result1} != {result2}"

        # Cache hit should meet SLA
        self.validate_cache_hit_performance(duration2, "textstat")

        self.logger.info(f"Coleman-Liau: miss={duration1:.6f}ms, hit={duration2:.6f}ms, result={result1}")

    def test_gunning_fog_real_behavior(self):
        """Test gunning_fog() with real cache behavior."""
        text = self.TEST_TEXTS['technical']

        # First call should be cache miss
        result1, duration1 = self.measure_performance(
            "gunning_fog_miss", self.textstat_wrapper.gunning_fog, text
        )

        # Validate result
        assert isinstance(result1, (int, float)), f"Expected numeric result, got {type(result1)}"
        assert result1 >= 0, f"Gunning Fog should be positive, got {result1}"

        # Second call should be cache hit
        result2, duration2 = self.measure_performance(
            "gunning_fog_hit", self.textstat_wrapper.gunning_fog, text
        )

        # Results should be identical
        assert result1 == result2, f"Cache hit result mismatch: {result1} != {result2}"

        # Cache hit should meet SLA
        self.validate_cache_hit_performance(duration2, "textstat")

        self.logger.info(f"Gunning Fog: miss={duration1:.6f}ms, hit={duration2:.6f}ms, result={result1}")

    def test_smog_index_real_behavior(self):
        """Test smog_index() with real cache behavior."""
        text = self.TEST_TEXTS['complex']

        # First call should be cache miss
        result1, duration1 = self.measure_performance(
            "smog_index_miss", self.textstat_wrapper.smog_index, text
        )

        # Validate result
        assert isinstance(result1, (int, float)), f"Expected numeric result, got {type(result1)}"
        assert result1 >= 0, f"SMOG index should be positive, got {result1}"

        # Second call should be cache hit
        result2, duration2 = self.measure_performance(
            "smog_index_hit", self.textstat_wrapper.smog_index, text
        )

        # Results should be identical
        assert result1 == result2, f"Cache hit result mismatch: {result1} != {result2}"

        # Cache hit should meet SLA
        self.validate_cache_hit_performance(duration2, "textstat")

        self.logger.info(f"SMOG Index: miss={duration1:.6f}ms, hit={duration2:.6f}ms, result={result1}")

    def test_comprehensive_analysis_real_behavior(self):
        """Test comprehensive_analysis() which uses all 9 TextStat functions."""
        text = self.TEST_TEXTS['medium']

        # First comprehensive analysis - all functions should be cache misses
        result1, duration1 = self.measure_performance(
            "comprehensive_analysis_miss", self.textstat_wrapper.comprehensive_analysis, text
        )

        # Validate comprehensive result structure
        expected_keys = {
            'flesch_reading_ease', 'flesch_kincaid_grade', 'syllable_count',
            'sentence_count', 'lexicon_count', 'automated_readability_index',
            'coleman_liau_index', 'gunning_fog', 'smog_index',
            'analysis_timestamp', 'config_language', 'caching_enabled'
        }

        assert isinstance(result1, dict), f"Expected dict result, got {type(result1)}"
        assert expected_keys.issubset(set(result1.keys())), f"Missing expected keys: {expected_keys - set(result1.keys())}"

        # Validate all numeric results
        numeric_keys = expected_keys - {'analysis_timestamp', 'config_language', 'caching_enabled'}
        for key in numeric_keys:
            assert isinstance(result1[key], (int, float)), f"Expected numeric value for {key}, got {type(result1[key])}"

        # Second comprehensive analysis - all functions should be cache hits
        result2, duration2 = self.measure_performance(
            "comprehensive_analysis_hit", self.textstat_wrapper.comprehensive_analysis, text
        )

        # Results should be nearly identical (timestamps may differ)
        for key in numeric_keys:
            assert result1[key] == result2[key], f"Cache hit result mismatch for {key}: {result1[key]} != {result2[key]}"

        # Cache hit should be significantly faster
        assert duration2 < duration1, f"Cache hit ({duration2:.6f}ms) should be faster than miss ({duration1:.6f}ms)"

        # Individual cache hits should meet SLA (not the overall comprehensive analysis)
        self.logger.info(f"Comprehensive Analysis: miss={duration1:.6f}ms, hit={duration2:.6f}ms")
        self.logger.info(f"Results: FRE={result1['flesch_reading_ease']}, FK={result1['flesch_kincaid_grade']}, Syllables={result1['syllable_count']}")

    def test_textstat_functions_with_edge_cases(self):
        """Test TextStat functions with edge case inputs."""
        edge_cases = [
            ('empty', self.TEST_TEXTS['empty']),
            ('single_word', self.TEST_TEXTS['single_word']),
            ('punctuation_heavy', self.TEST_TEXTS['punctuation_heavy']),
        ]

        for case_name, text in edge_cases:
            self.logger.info(f"Testing edge case: {case_name}")

            # Test a few key functions with edge cases
            functions = [
                ('flesch_reading_ease', self.textstat_wrapper.flesch_reading_ease),
                ('syllable_count', self.textstat_wrapper.syllable_count),
                ('sentence_count', self.textstat_wrapper.sentence_count),
                ('lexicon_count', self.textstat_wrapper.lexicon_count),
            ]

            for func_name, func in functions:
                try:
                    result, _duration = self.measure_performance(f"{func_name}_{case_name}", func, text)

                    # Validate result type
                    if func_name in {'syllable_count', 'sentence_count', 'lexicon_count'}:
                        assert isinstance(result, int), f"{func_name} should return int for {case_name}"
                    else:
                        assert isinstance(result, (int, float)), f"{func_name} should return number for {case_name}"

                    # Cache hit test
                    result2, duration2 = self.measure_performance(f"{func_name}_{case_name}_hit", func, text)
                    assert result == result2, f"{func_name} cache hit mismatch for {case_name}"
                    self.validate_cache_hit_performance(duration2, "textstat")

                except Exception as e:
                    self.logger.exception(f"Edge case {case_name} failed for {func_name}: {e}")
                    raise

    def test_textstat_cache_isolation_between_texts(self):
        """Test that different texts are cached separately."""
        text1 = self.TEST_TEXTS['simple']
        text2 = self.TEST_TEXTS['complex']

        # Get results for both texts
        result1_text1, _ = self.measure_performance("isolation_text1", self.textstat_wrapper.flesch_reading_ease, text1)
        result1_text2, _ = self.measure_performance("isolation_text2", self.textstat_wrapper.flesch_reading_ease, text2)

        # Results should be different for different texts
        assert result1_text1 != result1_text2, f"Different texts should have different results: {result1_text1} == {result1_text2}"

        # Cache hits should return same results
        result2_text1, duration1 = self.measure_performance("isolation_text1_hit", self.textstat_wrapper.flesch_reading_ease, text1)
        result2_text2, duration2 = self.measure_performance("isolation_text2_hit", self.textstat_wrapper.flesch_reading_ease, text2)

        assert result1_text1 == result2_text1, "Text1 cache hit mismatch"
        assert result1_text2 == result2_text2, "Text2 cache hit mismatch"

        # Both should meet SLA
        self.validate_cache_hit_performance(duration1, "textstat")
        self.validate_cache_hit_performance(duration2, "textstat")

    def test_textstat_cache_performance_under_load(self):
        """Test TextStat cache performance under high load."""
        import queue
        import threading

        results_queue = queue.Queue()
        errors_queue = queue.Queue()

        def worker(worker_id: int, text: str):
            try:
                start_time = time.time() * 1000

                # Perform multiple TextStat operations
                results = {
                    'flesch_reading_ease': self.textstat_wrapper.flesch_reading_ease(text),
                    'syllable_count': self.textstat_wrapper.syllable_count(text),
                    'sentence_count': self.textstat_wrapper.sentence_count(text),
                    'lexicon_count': self.textstat_wrapper.lexicon_count(text),
                }

                end_time = time.time() * 1000
                total_time = end_time - start_time

                results_queue.put({
                    'worker_id': worker_id,
                    'results': results,
                    'total_time_ms': total_time
                })

            except Exception as e:
                errors_queue.put(f"Worker {worker_id}: {e}")

        # Use same text for all workers to test cache sharing
        test_text = self.TEST_TEXTS['medium']

        # Prime the cache with first call
        self.textstat_wrapper.flesch_reading_ease(test_text)

        # Run 20 concurrent workers
        threads = []
        for i in range(20):
            thread = threading.Thread(target=worker, args=(i, test_text))
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
        assert len(results) == 20, f"Expected 20 results, got {len(results)}"

        # All workers should get identical results (same text, cached)
        first_results = results[0]['results']
        for result in results[1:]:
            for key in first_results:
                assert first_results[key] == result['results'][key], f"Inconsistent results for {key}"

        # Most operations should be fast (cache hits)
        fast_operations = sum(1 for r in results if r['total_time_ms'] < 5.0)  # 5ms threshold
        cache_hit_rate = fast_operations / len(results)

        assert cache_hit_rate >= 0.8, f"Cache hit rate {cache_hit_rate:.2%} below 80% threshold"

        avg_time = sum(r['total_time_ms'] for r in results) / len(results)
        self.logger.info(f"Concurrent load test: {len(results)} workers, {cache_hit_rate:.1%} fast operations, avg {avg_time:.2f}ms")

    def test_textstat_cache_invalidation_real_behavior(self):
        """Test cache invalidation for TextStat functions."""
        text = self.TEST_TEXTS['medium']

        # Prime cache
        original_result = self.textstat_wrapper.flesch_reading_ease(text)

        # Verify cache hit
        cached_result, hit_duration = self.measure_performance(
            "before_invalidation", self.textstat_wrapper.flesch_reading_ease, text
        )
        assert original_result == cached_result
        self.validate_cache_hit_performance(hit_duration, "textstat")

        # Clear cache
        clear_result = self.textstat_wrapper.clear_cache()
        assert clear_result['status'] == 'success', f"Cache clear failed: {clear_result}"
        self.logger.info(f"Cache cleared: {clear_result}")

        # Next call should be cache miss (slower)
        new_result, miss_duration = self.measure_performance(
            "after_invalidation", self.textstat_wrapper.flesch_reading_ease, text
        )

        # Result should be same but timing should indicate cache miss
        assert original_result == new_result, "Results should be identical after cache clear"
        assert miss_duration > hit_duration, f"Post-invalidation call ({miss_duration:.6f}ms) should be slower than cache hit ({hit_duration:.6f}ms)"

    def test_textstat_health_check_real_behavior(self):
        """Test TextStat wrapper health check functionality."""
        health_result = self.textstat_wrapper.health_check()

        # Validate health check structure
        assert isinstance(health_result, dict), "Health check should return dict"
        assert 'status' in health_result, "Health check missing status"
        assert 'component' in health_result, "Health check missing component"
        assert 'test_result' in health_result, "Health check missing test_result"

        # Should be healthy
        assert health_result['status'] == 'healthy', f"TextStat wrapper unhealthy: {health_result}"
        assert health_result['component'] == 'textstat_wrapper'
        assert isinstance(health_result['test_result'], (int, float)), "Test result should be numeric"

        self.logger.info(f"TextStat health check: {health_result['status']}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-k", "textstat"])
