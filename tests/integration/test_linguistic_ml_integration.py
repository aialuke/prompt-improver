"""
Test suite for linguistic analysis integration with ML pipeline.

Validates that the ContextSpecificLearner correctly integrates linguistic features
into the clustering and analysis pipeline.
"""

import pytest
import numpy as np
import random
from typing import Dict, Any, List

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from prompt_improver.learning.context_learner import ContextSpecificLearner, ContextConfig
from prompt_improver.analysis.linguistic_analyzer import LinguisticAnalyzer

# Set global random seeds for deterministic testing
random.seed(42)
np.random.seed(42)


class TestLinguisticMLIntegration:
    """Test linguistic analysis integration with ML pipeline."""

    @pytest.fixture
    def sample_results(self) -> List[Dict[str, Any]]:
        """Sample results with various prompt types for testing."""
        return [
            {
                'originalPrompt': 'Write a Python function to calculate factorial.',
                'overallScore': 0.7,
                'clarity': 0.6,
                'completeness': 0.5,
                'actionability': 0.8,
                'effectiveness': 0.7,
                'context': {
                    'projectType': 'api',
                    'domain': 'education',
                    'complexity': 'medium',
                    'teamSize': 5
                }
            },
            {
                'originalPrompt': 'Please write a comprehensive Python function that calculates the factorial of a number. The function should handle edge cases like negative numbers and zero. Include proper error handling, documentation, and examples of usage.',
                'overallScore': 0.9,
                'clarity': 0.9,
                'completeness': 0.9,
                'actionability': 0.9,
                'effectiveness': 0.9,
                'context': {
                    'projectType': 'api',
                    'domain': 'education', 
                    'complexity': 'high',
                    'teamSize': 8
                }
            },
            {
                'originalPrompt': 'Create ML model.',
                'overallScore': 0.3,
                'clarity': 0.2,
                'completeness': 0.2,
                'actionability': 0.4,
                'effectiveness': 0.3,
                'context': {
                    'projectType': 'ml',
                    'domain': 'other',
                    'complexity': 'low',
                    'teamSize': 3
                }
            },
            {
                'originalPrompt': 'Develop a machine learning model using Python and scikit-learn for predicting customer churn. The model should include data preprocessing, feature engineering, model training with cross-validation, and performance evaluation using appropriate metrics.',
                'overallScore': 0.95,
                'clarity': 0.9,
                'completeness': 0.95,
                'actionability': 0.9,
                'effectiveness': 0.95,
                'context': {
                    'projectType': 'ml',
                    'domain': 'finance',
                    'complexity': 'very_high',
                    'teamSize': 12
                }
            }
        ]

    @pytest.fixture
    def context_config_with_linguistics(self) -> ContextConfig:
        """Context config with linguistic features enabled."""
        return ContextConfig(
            enable_linguistic_features=True,
            linguistic_feature_weight=0.3,
            cache_linguistic_analysis=True,
            use_advanced_clustering=False,  # Use simple clustering for predictable testing
            min_sample_size=2  # Lower threshold for small test dataset
        )

    @pytest.fixture
    def context_config_without_linguistics(self) -> ContextConfig:
        """Context config with linguistic features disabled."""
        return ContextConfig(
            enable_linguistic_features=False,
            use_advanced_clustering=False,
            min_sample_size=2
        )

    def test_linguistic_analyzer_initialization(self, context_config_with_linguistics):
        """Test that linguistic analyzer is properly initialized."""
        learner = ContextSpecificLearner(context_config_with_linguistics)
        
        # Check that linguistic analyzer is initialized
        assert learner.linguistic_analyzer is not None
        assert learner.config.enable_linguistic_features is True
        assert learner.linguistic_cache == {}

    def test_linguistic_feature_extraction(self, context_config_with_linguistics, sample_results):
        """Test that linguistic features are properly extracted."""
        learner = ContextSpecificLearner(context_config_with_linguistics)
        
        # Test feature extraction from a sample result
        result = sample_results[1]  # Comprehensive prompt
        linguistic_features = learner._extract_linguistic_features(result)
        
        # Verify we get exactly 10 linguistic features
        assert len(linguistic_features) == 10
        
        # Verify all features are in valid range [0, 1]
        for feature in linguistic_features:
            assert 0.0 <= feature <= 1.0
        
        # Test that different prompts produce different features
        result_simple = sample_results[2]  # Simple prompt
        linguistic_features_simple = learner._extract_linguistic_features(result_simple)
        
        # Features should be different for different prompt qualities
        assert linguistic_features != linguistic_features_simple

    def test_clustering_features_with_linguistics(self, context_config_with_linguistics, sample_results):
        """Test that clustering features include linguistic features when enabled."""
        learner = ContextSpecificLearner(context_config_with_linguistics)
        
        # Extract clustering features
        features = learner._extract_clustering_features(sample_results)
        
        assert features is not None
        assert features.shape[0] == len(sample_results)
        
        # With linguistic features: 5 (performance) + 10 (linguistic) + 7 (project) + 7 (domain) + 1 (complexity) + 1 (team) = 31 features
        expected_features = 31
        assert features.shape[1] == expected_features

    def test_clustering_features_without_linguistics(self, context_config_without_linguistics, sample_results):
        """Test that clustering features work without linguistic features."""
        learner = ContextSpecificLearner(context_config_without_linguistics)
        
        # Extract clustering features
        features = learner._extract_clustering_features(sample_results)
        
        assert features is not None
        assert features.shape[0] == len(sample_results)
        
        # Without linguistic features: 5 (performance) + 10 (zeros) + 7 (project) + 7 (domain) + 1 (complexity) + 1 (team) = 31 features
        # (Still 31 because we add placeholder zeros to maintain consistent vector size)
        expected_features = 31
        assert features.shape[1] == expected_features
        
        # Verify that linguistic features are all zeros (columns 5-14)
        linguistic_columns = features[:, 5:15]
        assert np.all(linguistic_columns == 0.0)

    def test_linguistic_feature_caching(self, context_config_with_linguistics, sample_results):
        """Test that linguistic feature caching mechanism works correctly."""
        learner = ContextSpecificLearner(context_config_with_linguistics)
        
        result = sample_results[0]
        
        # Test that cache is being used by checking the cache behavior
        # Clear any existing cache first
        learner.linguistic_cache.clear()
        
        # First extraction should populate cache
        features1 = learner._extract_linguistic_features(result)
        assert len(learner.linguistic_cache) == 1, "Cache should have one entry after first extraction"
        
        # Store the cache key to verify it's being reused
        cache_key = list(learner.linguistic_cache.keys())[0]
        cached_features = learner.linguistic_cache[cache_key]
        
        # Second extraction should use cache 
        features2 = learner._extract_linguistic_features(result)
        
        # Verify cache wasn't expanded (same number of entries)
        assert len(learner.linguistic_cache) == 1, "Cache should still have only one entry"
        
        # Verify the same cache key is being used
        assert cache_key in learner.linguistic_cache, "Original cache key should still exist"
        
        # Verify features are same length and similar values
        assert len(features1) == len(features2) == 10, "Should have 10 linguistic features"
        
        # For BERT-based models, we expect some variation due to model randomness
        # But cached results should be exactly the same
        features_from_cache = learner.linguistic_cache[cache_key]
        assert features2 == features_from_cache, "Second extraction should return exact cached values"

    def test_linguistic_integration_improves_clustering(self, sample_results):
        """Test that linguistic features improve clustering quality."""
        # Test with linguistic features
        config_with = ContextConfig(
            enable_linguistic_features=True,
            use_advanced_clustering=False,
            min_sample_size=2
        )
        learner_with = ContextSpecificLearner(config_with)
        
        # Test without linguistic features  
        config_without = ContextConfig(
            enable_linguistic_features=False,
            use_advanced_clustering=False,
            min_sample_size=2
        )
        learner_without = ContextSpecificLearner(config_without)
        
        # Extract features with both configurations
        features_with = learner_with._extract_clustering_features(sample_results)
        features_without = learner_without._extract_clustering_features(sample_results)
        
        # Verify same dimensionality but different values
        assert features_with.shape == features_without.shape
        
        # Linguistic columns should be different
        linguistic_cols_with = features_with[:, 5:15]
        linguistic_cols_without = features_without[:, 5:15]
        
        # With linguistics should have non-zero values, without should be zeros
        assert not np.all(linguistic_cols_with == 0.0)
        assert np.all(linguistic_cols_without == 0.0)

    @pytest.mark.asyncio
    async def test_end_to_end_context_analysis_with_linguistics(self, context_config_with_linguistics, sample_results):
        """Test complete context analysis pipeline with linguistic features."""
        learner = ContextSpecificLearner(context_config_with_linguistics)
        
        # Prepare data for context analysis
        context_grouped_data = {
            'api_education_medium': {
                'historical_data': sample_results[:2],
                'sample_size': 2
            },
            'ml_mixed_complexity': {
                'historical_data': sample_results[2:],
                'sample_size': 2
            }
        }
        
        # Run context analysis
        analysis_result = await learner.analyze_context_effectiveness(context_grouped_data)
        
        # Verify analysis completed successfully
        assert 'context_insights' in analysis_result
        assert 'metadata' in analysis_result
        
        # Verify linguistic features are mentioned in metadata
        metadata = analysis_result['metadata']
        assert 'phase2_enhancements' in metadata
        assert metadata['phase2_enhancements']['advanced_clustering_enabled'] is False
        
        # Verify we have context insights
        context_insights = analysis_result['context_insights']
        assert len(context_insights) >= 1

    def test_linguistic_feature_weight_application(self, context_config_with_linguistics, sample_results):
        """Test that linguistic feature weight is properly applied."""
        learner = ContextSpecificLearner(context_config_with_linguistics)
        
        # Test with different weights
        result = sample_results[1]  # Comprehensive prompt
        
        # Extract features with default weight (0.3)
        features_default = learner._extract_linguistic_features(result)
        
        # Change weight and extract again (clear cache first)
        learner.linguistic_cache.clear()
        learner.config.linguistic_feature_weight = 0.8
        features_high_weight = learner._extract_linguistic_features(result)
        
        # Features should be different due to different weighting
        assert features_default != features_high_weight

    def test_linguistic_analysis_error_handling(self, context_config_with_linguistics):
        """Test error handling when linguistic analysis fails."""
        learner = ContextSpecificLearner(context_config_with_linguistics)
        
        # Test with invalid input
        invalid_result = {
            'originalPrompt': None,  # Invalid prompt
            'overallScore': 0.5
        }
        
        linguistic_features = learner._extract_linguistic_features(invalid_result)
        
        # Should return default features (all 0.5)
        assert len(linguistic_features) == 10
        assert all(f == 0.5 for f in linguistic_features)

    def test_linguistic_features_improve_context_discrimination(self, sample_results):
        """Test that linguistic features help distinguish between different contexts."""
        config = ContextConfig(
            enable_linguistic_features=True,
            use_advanced_clustering=False,
            min_sample_size=2
        )
        learner = ContextSpecificLearner(config)
        
        # Extract features for high-quality vs low-quality prompts
        high_quality_result = sample_results[3]  # Comprehensive ML prompt
        low_quality_result = sample_results[2]   # Simple "Create ML model"
        
        features_high = learner._extract_linguistic_features(high_quality_result)
        features_low = learner._extract_linguistic_features(low_quality_result)
        
        # Verify features are different (discrimination exists)
        features_diff = sum(abs(h - l) for h, l in zip(features_high, features_low))
        assert features_diff > 0.1, f"Features should be discriminative, diff: {features_diff}"
        
        # High-quality prompt should have better structure and content characteristics
        # Check sentence structure quality (feature index 4) - longer prompts typically have better structure
        assert features_high[4] > features_low[4], f"High quality should have better structure: {features_high[4]} vs {features_low[4]}"
        
        # Check average sentence length (feature index 6) - more detailed prompts have longer sentences
        assert features_high[6] > features_low[6], f"High quality should have longer sentences: {features_high[6]} vs {features_low[6]}"
        
        # Overall, the features should show clear discrimination
        print(f"Debug: High quality features: {features_high}")
        print(f"Debug: Low quality features: {features_low}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 