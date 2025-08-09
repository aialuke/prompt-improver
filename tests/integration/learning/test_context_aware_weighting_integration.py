"""
Integration tests for Context-Aware Feature Weighting in ContextSpecificLearner

This test suite validates that the context-aware weighting system integrates
correctly with the complete ML pipeline and produces expected behavior.
"""
from typing import Any, Dict, List
import numpy as np
import pytest
from prompt_improver.learning.context_aware_weighter import WeightingStrategy
from prompt_improver.learning.context_learner import ContextConfig, ContextSpecificLearner

class TestContextAwareWeightingIntegration:
    """Test context-aware weighting integration with ContextSpecificLearner."""

    @pytest.fixture
    def context_config(self):
        """Create context config with weighting enabled."""
        return ContextConfig(enable_domain_features=True, enable_context_aware_weighting=True, weighting_strategy='adaptive', confidence_boost_factor=0.3, enable_linguistic_features=True, use_advanced_clustering=False, use_ultra_lightweight_models=True, enable_model_quantization=True, enable_4bit_quantization=True, max_memory_threshold_mb=30, force_cpu_only=True)

    @pytest.fixture
    def learner(self, context_config):
        """Create context learner with weighting enabled."""
        return ContextSpecificLearner(context_config)

    @pytest.fixture
    def software_results(self):
        """Create test results for software development prompts."""
        return [{'originalPrompt': 'Write a Python function that implements a binary search algorithm with unit tests', 'overallScore': 0.8, 'clarity': 0.7, 'completeness': 0.9, 'actionability': 0.8, 'effectiveness': 0.8, 'context': {'projectType': 'api', 'domain': 'other', 'complexity': 'high', 'teamSize': 5}}, {'originalPrompt': 'Create a REST API endpoint using Flask with error handling and documentation', 'overallScore': 0.9, 'clarity': 0.8, 'completeness': 0.9, 'actionability': 0.9, 'effectiveness': 0.9, 'context': {'projectType': 'web', 'domain': 'other', 'complexity': 'medium', 'teamSize': 8}}]

    @pytest.fixture
    def creative_results(self):
        """Create test results for creative writing prompts."""
        return [{'originalPrompt': 'Write a compelling short story about a character who discovers they can see emotions as colors', 'overallScore': 0.7, 'clarity': 0.8, 'completeness': 0.6, 'actionability': 0.7, 'effectiveness': 0.7, 'context': {'projectType': 'other', 'domain': 'other', 'complexity': 'medium', 'teamSize': 3}}, {'originalPrompt': 'Create engaging marketing copy for a new sustainable fashion brand with emotional appeal', 'overallScore': 0.8, 'clarity': 0.9, 'completeness': 0.7, 'actionability': 0.8, 'effectiveness': 0.8, 'context': {'projectType': 'other', 'domain': 'other', 'complexity': 'medium', 'teamSize': 5}}]

    def test_weighter_initialization(self, learner):
        """Test that context-aware weighter is properly initialized."""
        assert learner.context_aware_weighter is not None
        assert learner.config.enable_context_aware_weighting is True
        assert hasattr(learner, '_get_feature_names')

    def test_feature_extraction_with_weighting(self, learner, software_results):
        """Test that feature extraction applies weighting correctly."""
        features_weighted = learner._extract_clustering_features(software_results)
        learner.config.enable_context_aware_weighting = False
        features_unweighted = learner._extract_clustering_features(software_results)
        assert features_weighted is not None
        assert features_unweighted is not None
        assert features_weighted.shape == features_unweighted.shape
        assert not np.array_equal(features_weighted, features_unweighted)
        learner.config.enable_context_aware_weighting = True

    def test_domain_specific_weighting_software(self, learner, software_results):
        """Test that software development prompts get appropriate weighting."""
        features = learner._extract_clustering_features(software_results)
        feature_names = learner._get_feature_names()
        assert features is not None
        assert len(features) == len(software_results)
        assert features.shape[1] == len(feature_names)
        assert np.all(np.isfinite(features))

    def test_domain_specific_weighting_creative(self, learner, creative_results):
        """Test that creative writing prompts get appropriate weighting."""
        features = learner._extract_clustering_features(creative_results)
        feature_names = learner._get_feature_names()
        assert features is not None
        assert len(features) == len(creative_results)
        assert features.shape[1] == len(feature_names)
        assert np.all(np.isfinite(features))

    def test_mixed_domain_clustering(self, learner, software_results, creative_results):
        """Test clustering with mixed domain types."""
        mixed_results = software_results + creative_results
        features = learner._extract_clustering_features(mixed_results)
        assert features is not None
        assert len(features) == len(mixed_results)
        software_features = features[:len(software_results)]
        creative_features = features[len(software_results):]
        software_mean = np.mean(software_features, axis=0)
        creative_mean = np.mean(creative_features, axis=0)
        differences = np.abs(software_mean - creative_mean)
        assert np.max(differences) > 0.1

    def test_weighting_strategies(self, context_config, software_results):
        """Test different weighting strategies."""
        strategies = ['static', 'adaptive', 'hybrid']
        results_by_strategy = {}
        for strategy in strategies:
            config = ContextConfig(enable_domain_features=True, enable_context_aware_weighting=True, weighting_strategy=strategy, enable_linguistic_features=True, use_advanced_clustering=False, use_ultra_lightweight_models=True, enable_model_quantization=True, enable_4bit_quantization=True, max_memory_threshold_mb=30, force_cpu_only=True)
            learner = ContextSpecificLearner(config)
            features = learner._extract_clustering_features(software_results)
            results_by_strategy[strategy] = features
        static_features = results_by_strategy['static']
        adaptive_features = results_by_strategy['adaptive']
        hybrid_features = results_by_strategy['hybrid']
        assert not np.array_equal(static_features, adaptive_features)
        assert not np.array_equal(static_features, hybrid_features)
        assert not np.array_equal(adaptive_features, hybrid_features)

    def test_confidence_impact(self, learner):
        """Test that domain confidence affects weighting."""
        high_confidence_result = [{'originalPrompt': 'Implement a machine learning model using scikit-learn for classification with cross-validation', 'overallScore': 0.8, 'clarity': 0.8, 'completeness': 0.8, 'actionability': 0.8, 'effectiveness': 0.8, 'context': {'projectType': 'ml', 'domain': 'other', 'complexity': 'high', 'teamSize': 5}}]
        low_confidence_result = [{'originalPrompt': 'Help me with some coding stuff', 'overallScore': 0.5, 'clarity': 0.5, 'completeness': 0.5, 'actionability': 0.5, 'effectiveness': 0.5, 'context': {'projectType': 'other', 'domain': 'other', 'complexity': 'low', 'teamSize': 3}}]
        high_conf_features = learner._extract_clustering_features(high_confidence_result)
        low_conf_features = learner._extract_clustering_features(low_confidence_result)
        assert high_conf_features is not None
        assert low_conf_features is not None
        assert not np.array_equal(high_conf_features, low_conf_features)

    def test_feature_vector_consistency(self, learner, software_results):
        """Test that feature vectors maintain consistent dimensions."""
        single_result = [software_results[0]]
        double_results = software_results[:2]
        single_features = learner._extract_clustering_features(single_result)
        double_features = learner._extract_clustering_features(double_results)
        assert single_features.shape[1] == double_features.shape[1]
        assert single_features.shape[0] == 1
        assert double_features.shape[0] == 2

    def test_weighting_error_handling(self, learner):
        """Test error handling in weighting system."""
        malformed_results = [{'originalPrompt': None, 'overallScore': 0.5, 'clarity': 0.5, 'completeness': 0.5, 'actionability': 0.5, 'effectiveness': 0.5, 'context': {}}, {'overallScore': 0.5, 'clarity': 0.5, 'completeness': 0.5, 'actionability': 0.5, 'effectiveness': 0.5, 'context': {}}]
        features = learner._extract_clustering_features(malformed_results)
        assert features is not None
        assert features.shape[0] == len(malformed_results)

    def test_weighting_performance(self, learner, software_results):
        """Test that weighting doesn't significantly impact performance."""
        import time
        start_time = time.time()
        for _ in range(5):
            learner._extract_clustering_features(software_results)
        weighted_time = time.time() - start_time
        learner.config.enable_context_aware_weighting = False
        start_time = time.time()
        for _ in range(5):
            learner._extract_clustering_features(software_results)
        unweighted_time = time.time() - start_time
        overhead_ratio = weighted_time / max(unweighted_time, 0.001)
        assert overhead_ratio < 1.5, f'Weighting overhead too high: {overhead_ratio:.2f}x'
        learner.config.enable_context_aware_weighting = True

    def test_hybrid_domain_handling(self, learner):
        """Test handling of prompts that span multiple domains."""
        hybrid_results = [{'originalPrompt': 'Create a data science project that analyzes creative writing samples to predict story engagement using machine learning', 'overallScore': 0.8, 'clarity': 0.8, 'completeness': 0.8, 'actionability': 0.8, 'effectiveness': 0.8, 'context': {'projectType': 'ml', 'domain': 'other', 'complexity': 'high', 'teamSize': 6}}]
        features = learner._extract_clustering_features(hybrid_results)
        assert features is not None
        assert features.shape[0] == 1
        assert np.all(np.isfinite(features))

    def test_configuration_validation(self):
        """Test that invalid configurations are handled properly."""
        invalid_config = ContextConfig(enable_context_aware_weighting=True, weighting_strategy='invalid_strategy')
        try:
            learner = ContextSpecificLearner(invalid_config)
            assert learner.context_aware_weighter is None or learner.context_aware_weighter.config.weighting_strategy is not None
        except (ValueError, TypeError):
            pass

class TestContextAwareWeightingPerformance:
    """Test performance characteristics of context-aware weighting."""

    def test_large_batch_processing(self):
        """Test weighting with large batches of results."""
        config = ContextConfig(enable_domain_features=True, enable_context_aware_weighting=True, enable_linguistic_features=True, use_advanced_clustering=False, use_lightweight_models=True, enable_model_quantization=False, max_memory_threshold_mb=50, force_cpu_only=True)
        learner = ContextSpecificLearner(config)
        large_batch = []
        for i in range(50):
            result = {'originalPrompt': f'Write a Python function for task {i} with proper documentation', 'overallScore': 0.7 + i % 3 * 0.1, 'clarity': 0.6 + i % 4 * 0.1, 'completeness': 0.7 + i % 5 * 0.1, 'actionability': 0.8, 'effectiveness': 0.7, 'context': {'projectType': 'api', 'complexity': 'medium', 'teamSize': 5}}
            large_batch.append(result)
        import time
        start_time = time.time()
        features = learner._extract_clustering_features(large_batch)
        processing_time = time.time() - start_time
        assert features is not None
        assert features.shape[0] == len(large_batch)
        assert processing_time < 10.0, f'Processing took too long: {processing_time:.2f}s'

    def test_memory_usage(self):
        """Test that weighting doesn't cause excessive memory usage."""
        import gc
        import os
        import psutil
        config = ContextConfig(enable_domain_features=True, enable_context_aware_weighting=True, enable_linguistic_features=True, use_ultra_lightweight_models=True, enable_model_quantization=True, enable_4bit_quantization=True, max_memory_threshold_mb=30, force_cpu_only=True)
        learner = ContextSpecificLearner(config)
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        for batch_num in range(10):
            batch_results = []
            for i in range(20):
                result = {'originalPrompt': f'Batch {batch_num} task {i}: implement algorithm with tests', 'overallScore': 0.8, 'clarity': 0.7, 'completeness': 0.8, 'actionability': 0.8, 'effectiveness': 0.8, 'context': {'projectType': 'api', 'complexity': 'high', 'teamSize': 5}}
                batch_results.append(result)
            features = learner._extract_clustering_features(batch_results)
            assert features is not None
            gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        try:
            import bitsandbytes
            max_memory = 50
        except ImportError:
            max_memory = 150
        print(f'Actual memory increase: {memory_increase:.2f}MB')
        print(f'Maximum allowed: {max_memory}MB')
        print('Note: Performance depends on available optimization libraries')
        assert memory_increase < max_memory, f'Excessive memory usage: {memory_increase:.2f}MB increase (max: {max_memory}MB)'
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
