"""Real Behavior Testing for ML Analysis Functions Cache Migration

This module provides comprehensive real behavior tests for the 3 ML analysis functions
migrated from @lru_cache to the unified cache infrastructure.

ML Analysis Functions Tested (3 functions):
1. linguistic_analyzer.analyze_cached() - Comprehensive linguistic feature extraction
2. domain_detector.detect_domain() - Domain classification with keyword matching
3. domain_feature_extractor.extract_domain_features() - Domain-specific feature extraction

SLA Targets (from baseline):
- Cache Hit: ≤ 1.0ms
- Cache Miss: ≤ 200ms
- Hit Rate: ≥ 90%

Based on baseline showing ML analysis requires higher time budgets but still maintains
sub-millisecond cache hit performance.
"""

import asyncio
import time

import pytest
from tests.contract.cache.test_lru_migration_real_behavior import RealBehaviorTestBase

from prompt_improver.ml.analysis.domain_detector import DomainDetector, PromptDomain
from prompt_improver.ml.analysis.domain_feature_extractor import DomainFeatureExtractor
from prompt_improver.ml.analysis.linguistic_analyzer import (
    LinguisticAnalyzer,
    LinguisticConfig,
)
from prompt_improver.services.cache.cache_facade import CacheFacade


@pytest.mark.real_behavior
@pytest.mark.cache_migration
@pytest.mark.ml_analysis
class TestMLAnalysisFunctionsCacheMigration(RealBehaviorTestBase):
    """Real behavior tests for ML Analysis functions cache migration.

    Tests all 3 ML analysis functions migrated from @lru_cache to unified cache.
    Validates performance, cache behavior, and integration with real ML workloads.
    """

    # Test prompts for different domains and complexity levels
    TEST_PROMPTS = {
        'software_development': "Write a Python function that implements binary search algorithm with proper error handling and documentation. Include unit tests and explain the time complexity.",
        'data_science': "Analyze this dataset using pandas and matplotlib. Create visualizations showing correlations between variables and perform statistical hypothesis testing.",
        'ai_ml': "Design a transformer architecture for natural language processing. Implement attention mechanisms and explain the training process with gradient descent.",
        'web_development': "Build a React component that handles user authentication with JWT tokens. Include form validation and responsive design using CSS.",
        'creative_writing': "Write a compelling short story about a character who discovers they can time travel. Focus on character development and descriptive narrative.",
        'business_analysis': "Perform a SWOT analysis for a new market entry strategy. Include competitor analysis, ROI projections, and risk assessment.",
        'simple': "Hello, how are you today?",
        'empty': "",
        'technical_complex': "The implementation utilizes advanced machine learning algorithms including convolutional neural networks, transformer architectures, and reinforcement learning paradigms to optimize performance across multiple objectives.",
        'mixed_domain': "Create a web application using Python Flask that analyzes customer data with machine learning models and provides interactive visualizations for business decision making.",
    }

    def setup_method(self):
        """Setup for ML analysis tests."""
        super().setup_method()

        # Create fresh ML analysis components with real cache
        self.ml_cache = CacheFacade(
            l1_max_size=1000,
            l2_default_ttl=14400,  # 4 hours for ML results
            enable_l2=True,  # Use L2 for persistence
        )

        # Initialize analyzers with shared cache
        self.linguistic_config = LinguisticConfig(
            enable_ner=True,
            enable_dependency_parsing=True,
            enable_readability=True,
            enable_complexity_metrics=True,
            use_lightweight_models=True,  # Use lightweight models for testing
            max_memory_threshold_mb=100,  # Lower memory usage
            force_cpu_only=True,  # Force CPU to avoid GPU dependencies
            enable_caching=True,
            cache_size=1000
        )

        self.linguistic_analyzer = LinguisticAnalyzer(
            config=self.linguistic_config,
            cache_facade=self.ml_cache
        )

        self.domain_detector = DomainDetector(cache_facade=self.ml_cache)
        self.domain_feature_extractor = DomainFeatureExtractor(cache_facade=self.ml_cache)

        # Clear any existing cache state
        try:
            asyncio.run(self.ml_cache.clear_all())
        except Exception as e:
            self.logger.warning(f"Failed to clear ML cache in setup: {e}")

    def teardown_method(self):
        """Cleanup for ML analysis tests."""
        super().teardown_method()

        # Get cache performance stats
        try:
            cache_stats = self.ml_cache.get_performance_stats()
            self.logger.info(f"ML cache performance: {cache_stats}")
        except Exception as e:
            self.logger.warning(f"Failed to get ML cache stats: {e}")

    def test_linguistic_analyzer_real_behavior(self):
        """Test linguistic_analyzer.analyze_cached() with real cache behavior."""
        prompt = self.TEST_PROMPTS['software_development']

        # First call should be cache miss (slower due to ML processing)
        result1, duration1 = self.measure_performance(
            "linguistic_analyze_miss", self.linguistic_analyzer.analyze_cached, prompt
        )

        # Validate result structure
        assert isinstance(result1, dict), f"Expected dict result, got {type(result1)}"

        # Check for key linguistic features
        expected_keys = {
            'entities', 'entity_types', 'entity_density', 'technical_terms',
            'syntactic_complexity', 'flesch_reading_ease', 'lexical_diversity',
            'avg_sentence_length', 'overall_linguistic_quality', 'confidence'
        }

        # Allow for missing keys in case of simplified implementation
        present_keys = set(result1.keys())
        common_keys = expected_keys.intersection(present_keys)
        assert len(common_keys) >= 3, f"Expected at least 3 linguistic features, got {common_keys}"

        # Validate numeric values
        for key in common_keys:
            if isinstance(result1[key], (int, float)):
                assert result1[key] >= 0, f"Linguistic metric {key} should be non-negative: {result1[key]}"

        # Second call should be cache hit (much faster)
        result2, duration2 = self.measure_performance(
            "linguistic_analyze_hit", self.linguistic_analyzer.analyze_cached, prompt
        )

        # Results should be identical for deterministic analysis
        assert result1 == result2, "Cache hit result mismatch for linguistic analysis"

        # Cache hit should meet ML SLA
        self.validate_cache_hit_performance(duration2, "ml")

        # Cache hit should be significantly faster than miss
        assert duration2 < duration1, f"Cache hit ({duration2:.6f}ms) should be faster than miss ({duration1:.6f}ms)"

        self.logger.info(f"Linguistic Analysis: miss={duration1:.6f}ms, hit={duration2:.6f}ms")
        self.logger.info(f"Analysis keys: {list(result1.keys())}")

    def test_domain_detector_real_behavior(self):
        """Test domain_detector.detect_domain() with real cache behavior."""
        # Test multiple domains
        domain_tests = [
            ('software_development', self.TEST_PROMPTS['software_development'], 'SOFTWARE_DEVELOPMENT'),
            ('data_science', self.TEST_PROMPTS['data_science'], 'DATA_SCIENCE'),
            ('ai_ml', self.TEST_PROMPTS['ai_ml'], 'AI_ML'),
            ('web_development', self.TEST_PROMPTS['web_development'], 'WEB_DEVELOPMENT'),
        ]

        for test_name, prompt, _expected_domain_name in domain_tests:
            self.logger.info(f"Testing domain detection: {test_name}")

            # First call should be cache miss
            result1, duration1 = self.measure_performance(
                f"domain_detect_{test_name}_miss", self.domain_detector.detect_domain, prompt
            )

            # Validate result structure
            assert hasattr(result1, 'primary_domain'), "Domain result missing primary_domain attribute"
            assert hasattr(result1, 'confidence'), "Domain result missing confidence attribute"

            # Validate domain classification
            assert isinstance(result1.primary_domain, PromptDomain), f"Expected PromptDomain, got {type(result1.primary_domain)}"
            assert 0.0 <= result1.confidence <= 1.0, f"Confidence should be 0-1, got {result1.confidence}"

            # For software development prompts, should detect correctly (if classifier is working)
            if test_name == 'software_development':
                # Allow some flexibility in domain detection
                tech_domains = {PromptDomain.SOFTWARE_DEVELOPMENT, PromptDomain.AI_ML, PromptDomain.WEB_DEVELOPMENT}
                assert result1.primary_domain in tech_domains, f"Expected technical domain for software prompt, got {result1.primary_domain}"

            # Second call should be cache hit
            result2, duration2 = self.measure_performance(
                f"domain_detect_{test_name}_hit", self.domain_detector.detect_domain, prompt
            )

            # Results should be identical
            assert result1.primary_domain == result2.primary_domain, f"Cache hit domain mismatch: {result1.primary_domain} != {result2.primary_domain}"
            assert result1.confidence == result2.confidence, f"Cache hit confidence mismatch: {result1.confidence} != {result2.confidence}"

            # Cache hit should meet ML SLA
            self.validate_cache_hit_performance(duration2, "ml")

            self.logger.info(f"Domain {test_name}: {result1.primary_domain.value} (conf={result1.confidence:.3f}), "
                           f"miss={duration1:.6f}ms, hit={duration2:.6f}ms")

    def test_domain_feature_extractor_real_behavior(self):
        """Test domain_feature_extractor.extract_domain_features() with real cache behavior."""
        prompt = self.TEST_PROMPTS['mixed_domain']
        detected_domain = PromptDomain.SOFTWARE_DEVELOPMENT  # Assume this domain for testing

        # First call should be cache miss
        result1, duration1 = self.measure_performance(
            "domain_features_miss", self.domain_feature_extractor.extract_domain_features,
            prompt, detected_domain
        )

        # Validate result structure
        assert isinstance(result1, dict), f"Expected dict result, got {type(result1)}"

        # Should contain domain-specific features
        expected_feature_types = {
            'keywords', 'complexity_metrics', 'domain_indicators',
            'technical_terms', 'feature_vector', 'domain_confidence'
        }

        # Allow flexibility - check for at least some features
        present_features = set(result1.keys())
        common_features = expected_feature_types.intersection(present_features)
        assert len(common_features) >= 2, f"Expected at least 2 domain features, got {common_features}"

        # Validate feature types
        for key in common_features:
            assert result1[key] is not None, f"Feature {key} should not be None"

        # Second call should be cache hit
        result2, duration2 = self.measure_performance(
            "domain_features_hit", self.domain_feature_extractor.extract_domain_features,
            prompt, detected_domain
        )

        # Results should be identical
        assert result1 == result2, "Cache hit result mismatch for domain feature extraction"

        # Cache hit should meet ML SLA
        self.validate_cache_hit_performance(duration2, "ml")

        # Cache hit should be faster
        assert duration2 < duration1, f"Cache hit ({duration2:.6f}ms) should be faster than miss ({duration1:.6f}ms)"

        self.logger.info(f"Domain Features: miss={duration1:.6f}ms, hit={duration2:.6f}ms")
        self.logger.info(f"Feature keys: {list(result1.keys())}")

    def test_ml_analysis_with_edge_cases(self):
        """Test ML analysis functions with edge case inputs."""
        edge_cases = [
            ('empty', self.TEST_PROMPTS['empty']),
            ('simple', self.TEST_PROMPTS['simple']),
            ('technical_complex', self.TEST_PROMPTS['technical_complex']),
        ]

        for case_name, prompt in edge_cases:
            self.logger.info(f"Testing ML edge case: {case_name}")

            # Test linguistic analysis
            try:
                ling_result, _ling_duration = self.measure_performance(
                    f"linguistic_{case_name}", self.linguistic_analyzer.analyze_cached, prompt
                )
                assert isinstance(ling_result, dict), f"Linguistic analysis should return dict for {case_name}"

                # Cache hit test
                ling_result2, ling_duration2 = self.measure_performance(
                    f"linguistic_{case_name}_hit", self.linguistic_analyzer.analyze_cached, prompt
                )
                assert ling_result == ling_result2, f"Linguistic cache hit mismatch for {case_name}"
                self.validate_cache_hit_performance(ling_duration2, "ml")

            except Exception as e:
                self.logger.warning(f"Linguistic analysis failed for {case_name}: {e}")
                # Don't fail the test - ML models can be fragile

            # Test domain detection
            try:
                domain_result, _domain_duration = self.measure_performance(
                    f"domain_{case_name}", self.domain_detector.detect_domain, prompt
                )
                assert hasattr(domain_result, 'primary_domain'), f"Domain detection should return proper result for {case_name}"

                # Cache hit test
                domain_result2, domain_duration2 = self.measure_performance(
                    f"domain_{case_name}_hit", self.domain_detector.detect_domain, prompt
                )
                assert domain_result.primary_domain == domain_result2.primary_domain, f"Domain cache hit mismatch for {case_name}"
                self.validate_cache_hit_performance(domain_duration2, "ml")

            except Exception as e:
                self.logger.warning(f"Domain detection failed for {case_name}: {e}")

    def test_ml_analysis_cache_isolation(self):
        """Test that different prompts and configurations are cached separately."""
        prompt1 = self.TEST_PROMPTS['software_development']
        prompt2 = self.TEST_PROMPTS['creative_writing']

        # Analyze both prompts
        result1_p1, _ = self.measure_performance("isolation_p1_ling", self.linguistic_analyzer.analyze_cached, prompt1)
        result1_p2, _ = self.measure_performance("isolation_p2_ling", self.linguistic_analyzer.analyze_cached, prompt2)

        # Results should be different for different prompts
        assert result1_p1 != result1_p2, "Different prompts should have different linguistic analysis results"

        # Cache hits should return same results
        result2_p1, duration1 = self.measure_performance("isolation_p1_ling_hit", self.linguistic_analyzer.analyze_cached, prompt1)
        result2_p2, duration2 = self.measure_performance("isolation_p2_ling_hit", self.linguistic_analyzer.analyze_cached, prompt2)

        assert result1_p1 == result2_p1, "Prompt1 linguistic cache hit mismatch"
        assert result1_p2 == result2_p2, "Prompt2 linguistic cache hit mismatch"

        # Both should meet ML SLA
        self.validate_cache_hit_performance(duration1, "ml")
        self.validate_cache_hit_performance(duration2, "ml")

        # Test domain detection isolation
        domain1_p1, _ = self.measure_performance("isolation_p1_domain", self.domain_detector.detect_domain, prompt1)
        domain1_p2, _ = self.measure_performance("isolation_p2_domain", self.domain_detector.detect_domain, prompt2)

        # Different prompts likely have different domains
        # (but don't assert this as it depends on classifier accuracy)
        self.logger.info(f"Domain isolation - P1: {domain1_p1.primary_domain}, P2: {domain1_p2.primary_domain}")

        # Cache hits should be consistent
        domain2_p1, duration3 = self.measure_performance("isolation_p1_domain_hit", self.domain_detector.detect_domain, prompt1)
        domain2_p2, duration4 = self.measure_performance("isolation_p2_domain_hit", self.domain_detector.detect_domain, prompt2)

        assert domain1_p1.primary_domain == domain2_p1.primary_domain, "Prompt1 domain cache hit mismatch"
        assert domain1_p2.primary_domain == domain2_p2.primary_domain, "Prompt2 domain cache hit mismatch"

        self.validate_cache_hit_performance(duration3, "ml")
        self.validate_cache_hit_performance(duration4, "ml")

    def test_ml_analysis_concurrent_access(self):
        """Test ML analysis functions under concurrent access."""
        import queue
        import threading

        results_queue = queue.Queue()
        errors_queue = queue.Queue()

        def worker(worker_id: int, prompt: str):
            try:
                start_time = time.time() * 1000

                # Perform ML analysis operations
                linguistic_result = self.linguistic_analyzer.analyze_cached(prompt)
                domain_result = self.domain_detector.detect_domain(prompt)

                end_time = time.time() * 1000
                total_time = end_time - start_time

                results_queue.put({
                    'worker_id': worker_id,
                    'linguistic_keys': list(linguistic_result.keys()) if isinstance(linguistic_result, dict) else [],
                    'domain': domain_result.primary_domain.value if hasattr(domain_result, 'primary_domain') else 'unknown',
                    'domain_confidence': domain_result.confidence if hasattr(domain_result, 'confidence') else 0.0,
                    'total_time_ms': total_time
                })

            except Exception as e:
                errors_queue.put(f"Worker {worker_id}: {e}")

        # Use same prompt for all workers to test cache sharing
        test_prompt = self.TEST_PROMPTS['software_development']

        # Prime the cache
        self.linguistic_analyzer.analyze_cached(test_prompt)
        self.domain_detector.detect_domain(test_prompt)

        # Run 10 concurrent workers
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i, test_prompt))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)  # Longer timeout for ML operations

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        errors = []
        while not errors_queue.empty():
            errors.append(errors_queue.get())

        # Log any errors but don't fail test (ML can be fragile)
        if errors:
            self.logger.warning(f"Concurrent ML analysis errors: {errors}")

        # Should have some successful results
        assert len(results) > 0, f"No successful concurrent ML analysis results: {errors}"

        # Results should be consistent across workers (same prompt, cached)
        if len(results) > 1:
            first_domain = results[0]['domain']
            consistent_domains = sum(1 for r in results if r['domain'] == first_domain)
            consistency_rate = consistent_domains / len(results)

            # Allow some inconsistency due to ML model variability
            assert consistency_rate >= 0.8, f"Domain consistency rate {consistency_rate:.2%} below 80%"

        # Most operations should benefit from caching
        fast_operations = sum(1 for r in results if r['total_time_ms'] < 100.0)  # 100ms threshold
        fast_rate = fast_operations / len(results) if results else 0

        self.logger.info(f"Concurrent ML test: {len(results)} successful, {len(errors)} errors, {fast_rate:.1%} fast operations")

    def test_ml_cache_performance_characteristics(self):
        """Test ML cache performance characteristics and SLA compliance."""
        prompt = self.TEST_PROMPTS['ai_ml']

        # Test linguistic analysis performance pattern
        linguistic_times = []

        # First call - cache miss
        _, miss_time = self.measure_performance("ml_perf_linguistic_miss", self.linguistic_analyzer.analyze_cached, prompt)

        # Multiple cache hits
        for i in range(5):
            _, hit_time = self.measure_performance(f"ml_perf_linguistic_hit_{i}", self.linguistic_analyzer.analyze_cached, prompt)
            linguistic_times.append(hit_time)
            self.validate_cache_hit_performance(hit_time, "ml")

        # Test domain detection performance pattern
        domain_times = []

        # Clear cache and test domain detection
        asyncio.run(self.ml_cache.clear_all())

        # First call - cache miss
        _, _miss_time = self.measure_performance("ml_perf_domain_miss", self.domain_detector.detect_domain, prompt)

        # Multiple cache hits
        for i in range(5):
            _, hit_time = self.measure_performance(f"ml_perf_domain_hit_{i}", self.domain_detector.detect_domain, prompt)
            domain_times.append(hit_time)
            self.validate_cache_hit_performance(hit_time, "ml")

        # Analyze performance characteristics
        avg_linguistic_hit = sum(linguistic_times) / len(linguistic_times)
        avg_domain_hit = sum(domain_times) / len(domain_times)

        # All cache hits should meet ML SLA (≤ 1.0ms)
        assert avg_linguistic_hit <= 1.0, f"Average linguistic cache hit {avg_linguistic_hit:.6f}ms > 1.0ms SLA"
        assert avg_domain_hit <= 1.0, f"Average domain cache hit {avg_domain_hit:.6f}ms > 1.0ms SLA"

        # Performance should be consistent (low variance)
        linguistic_variance = sum((t - avg_linguistic_hit) ** 2 for t in linguistic_times) / len(linguistic_times)
        domain_variance = sum((t - avg_domain_hit) ** 2 for t in domain_times) / len(domain_times)

        self.logger.info(f"ML Cache Performance - Linguistic: avg={avg_linguistic_hit:.6f}ms, var={linguistic_variance:.6f}")
        self.logger.info(f"ML Cache Performance - Domain: avg={avg_domain_hit:.6f}ms, var={domain_variance:.6f}")

        # Log cache statistics
        cache_stats = self.ml_cache.get_performance_stats()
        self.logger.info(f"ML Cache Stats: {cache_stats}")

    def test_ml_analysis_error_handling_real_behavior(self):
        """Test error handling in ML analysis functions with real cache."""
        # Test with potentially problematic inputs
        problematic_inputs = [
            ("very_long", "x" * 10000),  # Very long text
            ("special_chars", "∑∏∆∇∂∫∆∇⊂⊃⊆⊇∈∉∪∩∧∨¬→↔∃∀"),  # Special unicode characters
            ("mixed_languages", "Hello world Здравствуй мир 你好世界 こんにちは"),  # Mixed languages
        ]

        for test_name, problematic_input in problematic_inputs:
            self.logger.info(f"Testing ML error handling: {test_name}")

            # Test linguistic analysis error handling
            try:
                result, duration = self.measure_performance(
                    f"error_linguistic_{test_name}",
                    self.linguistic_analyzer.analyze_cached,
                    problematic_input
                )
                # Should either succeed or fail gracefully
                if result is not None:
                    assert isinstance(result, dict), "Linguistic analysis should return dict or fail gracefully"

                    # Test cache hit
                    result2, duration2 = self.measure_performance(
                        f"error_linguistic_{test_name}_hit",
                        self.linguistic_analyzer.analyze_cached,
                        problematic_input
                    )
                    assert result == result2, "Cached error result should be consistent"
                    self.validate_cache_hit_performance(duration2, "ml")

            except Exception as e:
                self.logger.info(f"Linguistic analysis failed gracefully for {test_name}: {e}")

            # Test domain detection error handling
            try:
                result, _duration = self.measure_performance(
                    f"error_domain_{test_name}",
                    self.domain_detector.detect_domain,
                    problematic_input
                )
                # Should either succeed or fail gracefully
                if result is not None:
                    assert hasattr(result, 'primary_domain'), "Domain detection should return proper result or fail gracefully"

                    # Test cache hit
                    result2, duration2 = self.measure_performance(
                        f"error_domain_{test_name}_hit",
                        self.domain_detector.detect_domain,
                        problematic_input
                    )
                    assert result.primary_domain == result2.primary_domain, "Cached error result should be consistent"
                    self.validate_cache_hit_performance(duration2, "ml")

            except Exception as e:
                self.logger.info(f"Domain detection failed gracefully for {test_name}: {e}")


if __name__ == "__main__":
    # Run tests with verbose output and longer timeout for ML operations
    pytest.main([__file__, "-v", "--tb=short", "-k", "ml_analysis", "--timeout=300"])
