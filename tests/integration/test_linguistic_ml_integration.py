"""
Test suite for linguistic analysis integration with ML pipeline.

Validates that the ContextSpecificLearner correctly integrates linguistic features
into the clustering and analysis pipeline.
"""

import os
import random
import sys
from typing import Any, Dict, List

import numpy as np
import pytest

from prompt_improver.analysis.linguistic_analyzer import LinguisticAnalyzer
from prompt_improver.learning.context_learner import (
    ContextConfig,
    ContextSpecificLearner,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
random.seed(42)
np.random.seed(42)


class TestLinguisticMLIntegration:
    """Test linguistic analysis integration with ML pipeline."""

    @pytest.fixture
    def sample_results(self) -> list[dict[str, Any]]:
        """Sample results with various prompt types for testing."""
        return [
            {
                "originalPrompt": "Write a Python function to calculate factorial.",
                "overallScore": 0.7,
                "clarity": 0.6,
                "completeness": 0.5,
                "actionability": 0.8,
                "effectiveness": 0.7,
                "context": {
                    "projectType": "api",
                    "domain": "education",
                    "complexity": "medium",
                    "teamSize": 5,
                },
            },
            {
                "originalPrompt": "Please write a comprehensive Python function that calculates the factorial of a number. The function should handle edge cases like negative numbers and zero. Include proper error handling, documentation, and examples of usage.",
                "overallScore": 0.9,
                "clarity": 0.9,
                "completeness": 0.9,
                "actionability": 0.9,
                "effectiveness": 0.9,
                "context": {
                    "projectType": "api",
                    "domain": "education",
                    "complexity": "high",
                    "teamSize": 8,
                },
            },
            {
                "originalPrompt": "Create ML model.",
                "overallScore": 0.3,
                "clarity": 0.2,
                "completeness": 0.2,
                "actionability": 0.4,
                "effectiveness": 0.3,
                "context": {
                    "projectType": "ml",
                    "domain": "other",
                    "complexity": "low",
                    "teamSize": 3,
                },
            },
            {
                "originalPrompt": "Develop a machine learning model using Python and scikit-learn for predicting customer churn. The model should include data preprocessing, feature engineering, model training with cross-validation, and performance evaluation using appropriate metrics.",
                "overallScore": 0.95,
                "clarity": 0.9,
                "completeness": 0.95,
                "actionability": 0.9,
                "effectiveness": 0.95,
                "context": {
                    "projectType": "ml",
                    "domain": "finance",
                    "complexity": "very_high",
                    "teamSize": 12,
                },
            },
        ]

    @pytest.fixture
    def context_config_with_linguistics(self) -> ContextConfig:
        """Context config with linguistic features enabled."""
        return ContextConfig(
            enable_linguistic_features=True,
            linguistic_feature_weight=0.3,
            cache_linguistic_analysis=True,
            use_advanced_clustering=False,
            min_sample_size=2,
        )

    @pytest.fixture
    def context_config_without_linguistics(self) -> ContextConfig:
        """Context config with linguistic features disabled."""
        return ContextConfig(
            enable_linguistic_features=False,
            use_advanced_clustering=False,
            min_sample_size=2,
        )

    def test_linguistic_analyzer_initialization(self, context_config_with_linguistics):
        """Test that linguistic analyzer is properly initialized."""
        learner = ContextSpecificLearner(context_config_with_linguistics)
        assert learner.linguistic_analyzer is not None
        assert learner.config.enable_linguistic_features is True
        assert learner.linguistic_cache == {}

    def test_linguistic_feature_extraction(
        self, context_config_with_linguistics, sample_results
    ):
        """Test that linguistic features are properly extracted."""
        learner = ContextSpecificLearner(context_config_with_linguistics)
        result = sample_results[1]
        linguistic_features = learner._extract_linguistic_features(result)
        assert len(linguistic_features) == 10
        for feature in linguistic_features:
            assert 0.0 <= feature <= 1.0
        result_simple = sample_results[2]
        linguistic_features_simple = learner._extract_linguistic_features(result_simple)
        assert linguistic_features != linguistic_features_simple

    def test_clustering_features_with_linguistics(
        self, context_config_with_linguistics, sample_results
    ):
        """Test that clustering features include linguistic features when enabled."""
        learner = ContextSpecificLearner(context_config_with_linguistics)
        features = learner._extract_clustering_features(sample_results)
        assert features is not None
        assert features.shape[0] == len(sample_results)
        expected_features = 31
        assert features.shape[1] == expected_features

    def test_clustering_features_without_linguistics(
        self, context_config_without_linguistics, sample_results
    ):
        """Test that clustering features work without linguistic features."""
        learner = ContextSpecificLearner(context_config_without_linguistics)
        features = learner._extract_clustering_features(sample_results)
        assert features is not None
        assert features.shape[0] == len(sample_results)
        expected_features = 31
        assert features.shape[1] == expected_features
        linguistic_columns = features[:, 5:15]
        assert np.all(linguistic_columns == 0.0)

    def test_linguistic_feature_caching(
        self, context_config_with_linguistics, sample_results
    ):
        """Test that linguistic feature caching mechanism works correctly."""
        learner = ContextSpecificLearner(context_config_with_linguistics)
        result = sample_results[0]
        learner.linguistic_cache.clear()
        features1 = learner._extract_linguistic_features(result)
        assert len(learner.linguistic_cache) == 1, (
            "Cache should have one entry after first extraction"
        )
        cache_key = list(learner.linguistic_cache.keys())[0]
        cached_features = learner.linguistic_cache[cache_key]
        features2 = learner._extract_linguistic_features(result)
        assert len(learner.linguistic_cache) == 1, (
            "Cache should still have only one entry"
        )
        assert cache_key in learner.linguistic_cache, (
            "Original cache key should still exist"
        )
        assert len(features1) == len(features2) == 10, (
            "Should have 10 linguistic features"
        )
        features_from_cache = learner.linguistic_cache[cache_key]
        assert features2 == features_from_cache, (
            "Second extraction should return exact cached values"
        )

    def test_linguistic_integration_improves_clustering(self, sample_results):
        """Test that linguistic features improve clustering quality."""
        config_with = ContextConfig(
            enable_linguistic_features=True,
            use_advanced_clustering=False,
            min_sample_size=2,
        )
        learner_with = ContextSpecificLearner(config_with)
        config_without = ContextConfig(
            enable_linguistic_features=False,
            use_advanced_clustering=False,
            min_sample_size=2,
        )
        learner_without = ContextSpecificLearner(config_without)
        features_with = learner_with._extract_clustering_features(sample_results)
        features_without = learner_without._extract_clustering_features(sample_results)
        assert features_with.shape == features_without.shape
        linguistic_cols_with = features_with[:, 5:15]
        linguistic_cols_without = features_without[:, 5:15]
        assert not np.all(linguistic_cols_with == 0.0)
        assert np.all(linguistic_cols_without == 0.0)

    @pytest.mark.asyncio
    async def test_end_to_end_context_analysis_with_linguistics(
        self, context_config_with_linguistics, sample_results
    ):
        """Test complete context analysis pipeline with linguistic features."""
        learner = ContextSpecificLearner(context_config_with_linguistics)
        context_grouped_data = {
            "api_education_medium": {
                "historical_data": sample_results[:2],
                "sample_size": 2,
            },
            "ml_mixed_complexity": {
                "historical_data": sample_results[2:],
                "sample_size": 2,
            },
        }
        analysis_result = await learner.analyze_context_effectiveness(
            context_grouped_data
        )
        assert "context_insights" in analysis_result
        assert "metadata" in analysis_result
        metadata = analysis_result["metadata"]
        assert "phase2_enhancements" in metadata
        assert metadata["phase2_enhancements"]["advanced_clustering_enabled"] is False
        context_insights = analysis_result["context_insights"]
        assert len(context_insights) >= 1

    def test_linguistic_feature_weight_application(
        self, context_config_with_linguistics, sample_results
    ):
        """Test that linguistic feature weight is properly applied."""
        learner = ContextSpecificLearner(context_config_with_linguistics)
        result = sample_results[1]
        features_default = learner._extract_linguistic_features(result)
        learner.linguistic_cache.clear()
        learner.config.linguistic_feature_weight = 0.8
        features_high_weight = learner._extract_linguistic_features(result)
        assert features_default != features_high_weight

    def test_linguistic_analysis_error_handling(self, context_config_with_linguistics):
        """Test error handling when linguistic analysis fails."""
        learner = ContextSpecificLearner(context_config_with_linguistics)
        invalid_result = {"originalPrompt": None, "overallScore": 0.5}
        linguistic_features = learner._extract_linguistic_features(invalid_result)
        assert len(linguistic_features) == 10
        assert all(f == 0.5 for f in linguistic_features)

    def test_linguistic_features_improve_context_discrimination(self, sample_results):
        """Test that linguistic features help distinguish between different contexts."""
        config = ContextConfig(
            enable_linguistic_features=True,
            use_advanced_clustering=False,
            min_sample_size=2,
        )
        learner = ContextSpecificLearner(config)
        high_quality_result = sample_results[3]
        low_quality_result = sample_results[2]
        features_high = learner._extract_linguistic_features(high_quality_result)
        features_low = learner._extract_linguistic_features(low_quality_result)
        features_diff = sum(
            (abs(h - l) for h, l in zip(features_high, features_low, strict=False))
        )
        assert features_diff > 0.1, (
            f"Features should be discriminative, diff: {features_diff}"
        )
        assert features_high[4] > features_low[4], (
            f"High quality should have better structure: {features_high[4]} vs {features_low[4]}"
        )
        assert features_high[6] > features_low[6], (
            f"High quality should have longer sentences: {features_high[6]} vs {features_low[6]}"
        )
        print(f"Debug: High quality features: {features_high}")
        print(f"Debug: Low quality features: {features_low}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
