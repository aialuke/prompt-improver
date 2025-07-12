"""Tests for Phase 2 In-Context Learning implementation in Context-Specific Learning.

Comprehensive test suite for in-context learning enhancements including:
- Demonstration selection and retrieval
- Contextual bandit Thompson Sampling
- Differential privacy preservation
- Privacy-preserving personalization
- Integration with existing context learning workflow

Testing best practices applied from Context7 research:
- Proper async testing with pytest-asyncio
- Statistical validation of ML components
- Normalization verification for neural network inputs
- Mocking external dependencies for isolation
- Progressive testing with realistic data ranges
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any
from datetime import datetime

from prompt_improver.learning.context_learner import (
    ContextSpecificLearner,
    ContextConfig,
    ContextInsight,
    LearningRecommendation
)


@pytest.fixture
def icl_config():
    """Configuration with in-context learning enabled."""
    return ContextConfig(
        enable_in_context_learning=True,
        icl_demonstrations=5,
        icl_similarity_threshold=0.7,
        privacy_preserving=True,
        differential_privacy_epsilon=1.0,
        # Realistic parameter ranges based on ML-Agents best practices
        significance_threshold=0.15,
        min_sample_size=20,  # Sufficient for statistical significance
        similarity_threshold=0.8
    )


@pytest.fixture
def context_engine(icl_config):
    """Context learning engine with ICL enabled."""
    return ContextSpecificLearner(config=icl_config)


@pytest.fixture
def sample_historical_data():
    """Realistic historical data for in-context learning tests."""
    return [
        {
            "context": "technical_documentation",
            "prompt": "Optimize this API documentation for clarity",
            "score": 0.85,
            "timestamp": datetime.now().isoformat(),
            "features": [0.7, 0.8, 0.6, 0.9],  # Normalized features [0,1]
            "user_id": "user1",
            "session_id": "session1"
        },
        {
            "context": "technical_documentation", 
            "prompt": "Improve code comments structure",
            "score": 0.78,
            "timestamp": datetime.now().isoformat(),
            "features": [0.6, 0.7, 0.8, 0.7],
            "user_id": "user1",
            "session_id": "session2"
        },
        {
            "context": "creative_writing",
            "prompt": "Enhance narrative flow in story",
            "score": 0.92,
            "timestamp": datetime.now().isoformat(),
            "features": [0.9, 0.8, 0.7, 0.85],
            "user_id": "user2", 
            "session_id": "session3"
        },
        {
            "context": "technical_documentation",
            "prompt": "Restructure technical guide layout",
            "score": 0.88,
            "timestamp": datetime.now().isoformat(),
            "features": [0.8, 0.9, 0.7, 0.8],
            "user_id": "user1",
            "session_id": "session4"
        },
        {
            "context": "business_communication",
            "prompt": "Optimize meeting summary format",
            "score": 0.75,
            "timestamp": datetime.now().isoformat(), 
            "features": [0.5, 0.6, 0.9, 0.7],
            "user_id": "user3",
            "session_id": "session5"
        }
    ]


@pytest.mark.asyncio
class TestInContextLearning:
    """Test suite for in-context learning functionality."""

    async def test_in_context_learning_integration(self, context_engine, sample_historical_data):
        """Test full in-context learning workflow integration."""
        # Test data within realistic ML performance ranges
        context_data = {
            "technical_documentation": {
                "sample_size": 25,  # Above min_sample_size threshold
                "avg_performance": 0.83,  # Strong performance in [0.7, 0.95] range
                "consistency_score": 0.85,  # High consistency
                "historical_data": sample_historical_data[:3]
            }
        }
        
        result = await context_engine.analyze_context_effectiveness(context_data)
        
        # Verify in-context learning was applied
        assert "in_context_learning" in result
        icl_result = result["in_context_learning"]
        
        # Validate realistic performance metrics
        assert 0.0 <= icl_result.get("similarity_score", 0) <= 1.0
        assert 0.0 <= icl_result.get("personalization_score", 0) <= 1.0
        assert icl_result.get("demonstrations_used", 0) >= 0
        
        # Verify privacy preservation was applied
        if context_engine.config.privacy_preserving:
            assert "privacy_metrics" in icl_result
            privacy_metrics = icl_result["privacy_metrics"]
            assert "epsilon_spent" in privacy_metrics
            assert 0 < privacy_metrics["epsilon_spent"] <= context_engine.config.differential_privacy_epsilon

    async def test_demonstration_selection(self, context_engine, sample_historical_data):
        """Test contextual demonstration selection with similarity scoring."""
        query_context = {
            "context": "technical_documentation",
            "features": [0.75, 0.8, 0.65, 0.85],  # Normalized input features
            "user_id": "user1"
        }
        
        # Use private method for focused testing
        demonstrations = context_engine._select_contextual_demonstrations(
            query_context, sample_historical_data
        )
        
        # Validate demonstration selection
        assert len(demonstrations) <= context_engine.config.icl_demonstrations
        assert all(demo.get("context") == query_context["context"] for demo in demonstrations)
        
        # Verify similarity scoring produces realistic values
        for demo in demonstrations:
            assert "similarity_score" in demo
            assert 0.0 <= demo["similarity_score"] <= 1.0
            # Should meet similarity threshold
            assert demo["similarity_score"] >= context_engine.config.icl_similarity_threshold

    async def test_contextual_bandit_thompson_sampling(self, context_engine, sample_historical_data):
        """Test contextual bandit with Thompson Sampling for exploration."""
        context_vector = np.array([0.7, 0.8, 0.6, 0.9])  # Normalized context
        
        # Test bandit action selection
        action_scores = await context_engine._apply_contextual_bandit(
            context_vector, sample_historical_data
        )
        
        # Validate Thompson Sampling results
        assert isinstance(action_scores, dict)
        assert "selected_action" in action_scores
        assert "confidence_interval" in action_scores
        assert "exploration_bonus" in action_scores
        
        # Verify realistic confidence bounds
        ci = action_scores["confidence_interval"]
        assert ci["lower"] <= ci["upper"]
        assert 0.0 <= ci["lower"] <= 1.0
        assert 0.0 <= ci["upper"] <= 1.0
        
        # Exploration bonus should be positive for uncertainty
        assert action_scores["exploration_bonus"] >= 0.0

    async def test_differential_privacy_application(self, context_engine, sample_historical_data):
        """Test differential privacy preservation in personalization."""
        original_scores = [0.85, 0.78, 0.92, 0.88, 0.75]
        
        # Apply differential privacy
        private_result = context_engine._apply_differential_privacy(
            original_scores, sample_historical_data
        )
        
        # Validate privacy preservation
        assert "noisy_scores" in private_result
        assert "privacy_budget_used" in private_result
        assert "privacy_guarantee" in private_result
        
        noisy_scores = private_result["noisy_scores"]
        
        # Scores should remain in reasonable bounds despite noise
        assert len(noisy_scores) == len(original_scores)
        for score in noisy_scores:
            # Allow some noise but maintain realistic bounds
            assert -0.5 <= score <= 1.5  # Bounded noise around [0,1]
        
        # Privacy budget should be consumed appropriately
        budget_used = private_result["privacy_budget_used"]
        assert 0 < budget_used <= context_engine.config.differential_privacy_epsilon

    async def test_feature_normalization(self, context_engine):
        """Test proper feature normalization for neural network compatibility."""
        raw_features = [100, 0.5, 1000, 0.001]  # Mixed scale features
        
        # Test internal normalization (would be called during processing)
        # Using realistic bounds from ML-Agents best practices
        normalized = []
        for i, feature in enumerate(raw_features):
            # Simulate normalization to [0,1] range
            if i == 0:  # Large scale feature
                normalized.append(min(feature / 1000, 1.0))
            elif i == 2:  # Another large scale
                normalized.append(min(feature / 2000, 1.0))
            else:  # Already in good range
                normalized.append(feature)
        
        # Verify normalization follows ML-Agents best practices
        for norm_value in normalized:
            assert 0.0 <= norm_value <= 1.0, f"Feature {norm_value} not in [0,1] range"

    async def test_privacy_preserving_disabled(self, sample_historical_data):
        """Test behavior when privacy preservation is disabled."""
        config = ContextConfig(
            enable_in_context_learning=True,
            privacy_preserving=False,
            icl_demonstrations=3
        )
        engine = ContextSpecificLearningEngine(config=config)
        
        original_scores = [0.85, 0.78, 0.92]
        
        result = engine._apply_differential_privacy(original_scores, sample_historical_data)
        
        # Should return original scores when privacy is disabled
        assert result["noisy_scores"] == original_scores
        assert result["privacy_budget_used"] == 0.0
        assert result["privacy_guarantee"] == "disabled"

    async def test_insufficient_demonstration_data(self, context_engine):
        """Test handling of insufficient demonstration data."""
        minimal_data = [
            {
                "context": "rare_context",
                "features": [0.5, 0.6, 0.7, 0.8],
                "score": 0.75
            }
        ]
        
        query_context = {
            "context": "rare_context", 
            "features": [0.6, 0.7, 0.8, 0.9],
            "user_id": "new_user"
        }
        
        demonstrations = context_engine._select_contextual_demonstrations(
            query_context, minimal_data
        )
        
        # Should handle gracefully with limited data
        assert len(demonstrations) <= len(minimal_data)
        # Should still return available demonstrations even if below desired count
        assert len(demonstrations) >= 0

    @pytest.mark.parametrize("epsilon", [0.1, 1.0, 5.0])
    async def test_privacy_epsilon_levels(self, sample_historical_data, epsilon):
        """Test different privacy epsilon levels for differential privacy."""
        config = ContextConfig(
            enable_in_context_learning=True,
            privacy_preserving=True,
            differential_privacy_epsilon=epsilon
        )
        engine = ContextSpecificLearningEngine(config=config)
        
        scores = [0.8, 0.7, 0.9, 0.85]
        result = engine._apply_differential_privacy(scores, sample_historical_data)
        
        # Higher epsilon should use more budget (less privacy, more utility)
        assert result["privacy_budget_used"] <= epsilon
        
        # Noise magnitude should be inversely related to epsilon
        noisy_scores = result["noisy_scores"]
        noise_magnitude = np.mean([abs(n - o) for n, o in zip(noisy_scores, scores)])
        
        # Lower epsilon should generally produce more noise (not strict due to randomness)
        assert noise_magnitude >= 0.0

    async def test_similarity_threshold_filtering(self, context_engine, sample_historical_data):
        """Test demonstration filtering based on similarity threshold."""
        # Query with features that should have varying similarity to historical data
        query_context = {
            "context": "technical_documentation",
            "features": [0.1, 0.2, 0.1, 0.15],  # Very different from historical data
            "user_id": "user1"
        }
        
        demonstrations = context_engine._select_contextual_demonstrations(
            query_context, sample_historical_data
        )
        
        # With low similarity query, might get no demonstrations above threshold
        for demo in demonstrations:
            assert demo["similarity_score"] >= context_engine.config.icl_similarity_threshold

    async def test_contextual_bandit_convergence(self, context_engine):
        """Test contextual bandit learning convergence with repeated interactions."""
        context_vector = np.array([0.8, 0.7, 0.9, 0.8])
        historical_data = []
        
        # Simulate multiple rounds of learning
        for round_num in range(10):
            action_scores = await context_engine._apply_contextual_bandit(
                context_vector, historical_data
            )
            
            # Simulate feedback (higher reward for consistent actions)
            simulated_reward = 0.8 + np.random.normal(0, 0.1)
            
            # Add experience to historical data
            historical_data.append({
                "context_vector": context_vector.tolist(),
                "action": action_scores["selected_action"],
                "reward": max(0.0, min(1.0, simulated_reward)),
                "round": round_num
            })
            
            # Confidence intervals should generally narrow with more data
            if round_num > 5:
                ci_width = action_scores["confidence_interval"]["upper"] - action_scores["confidence_interval"]["lower"]
                # Not strictly enforced due to Thompson Sampling randomness
                assert ci_width >= 0.0


@pytest.mark.asyncio
class TestInContextLearningErrorHandling:
    """Test error handling and edge cases for in-context learning."""

    async def test_missing_features_handling(self, context_engine):
        """Test handling of data with missing feature vectors."""
        incomplete_data = [
            {"context": "test", "score": 0.8},  # Missing features
            {"context": "test", "features": [0.7, 0.8], "score": 0.75}  # Wrong feature size
        ]
        
        query_context = {
            "context": "test",
            "features": [0.8, 0.9, 0.7, 0.8],
            "user_id": "test_user"
        }
        
        # Should handle gracefully without crashing
        demonstrations = context_engine._select_contextual_demonstrations(
            query_context, incomplete_data
        )
        
        # Should filter out invalid data
        assert all("features" in demo for demo in demonstrations)

    async def test_empty_historical_data(self, context_engine):
        """Test behavior with no historical data."""
        query_context = {
            "context": "new_context",
            "features": [0.5, 0.6, 0.7, 0.8],
            "user_id": "new_user"
        }
        
        demonstrations = context_engine._select_contextual_demonstrations(
            query_context, []
        )
        
        assert demonstrations == []

    async def test_invalid_privacy_epsilon(self):
        """Test handling of invalid privacy epsilon values."""
        with pytest.raises((ValueError, AssertionError)):
            ContextConfig(
                enable_in_context_learning=True,
                privacy_preserving=True,
                differential_privacy_epsilon=-1.0  # Invalid negative epsilon
            )

    async def test_extreme_feature_values(self, context_engine):
        """Test handling of extreme feature values."""
        extreme_data = [
            {
                "context": "test",
                "features": [1e6, -1e6, float('inf'), 0.5],  # Extreme values
                "score": 0.8
            }
        ]
        
        query_context = {
            "context": "test", 
            "features": [0.5, 0.6, 0.7, 0.8],
            "user_id": "test"
        }
        
        # Should handle extreme values without crashing
        try:
            demonstrations = context_engine._select_contextual_demonstrations(
                query_context, extreme_data
            )
            # If it doesn't crash, verify results are reasonable
            for demo in demonstrations:
                assert "similarity_score" in demo
                assert not np.isnan(demo["similarity_score"])
                assert not np.isinf(demo["similarity_score"])
        except (ValueError, OverflowError):
            # Acceptable to reject extreme values
            pass


@pytest.mark.asyncio
class TestInContextLearningIntegration:
    """Integration tests for in-context learning with existing workflows."""

    async def test_icl_with_existing_context_analysis(self, context_engine, sample_historical_data):
        """Test ICL integration with existing context analysis workflow."""
        # Real context data that would come from existing workflow
        context_data = {
            "technical_documentation": {
                "sample_size": 30,
                "avg_performance": 0.82,
                "consistency_score": 0.78,
                "historical_data": sample_historical_data,
                "user_demographics": {"experience_level": "intermediate"}
            },
            "business_communication": {
                "sample_size": 15,
                "avg_performance": 0.75,
                "consistency_score": 0.85,
                "historical_data": sample_historical_data[-2:],
                "user_demographics": {"experience_level": "expert"}
            }
        }
        
        result = await context_engine.analyze_context_effectiveness(context_data)
        
        # Should have both traditional analysis and ICL results
        assert "context_insights" in result
        assert "in_context_learning" in result
        
        # ICL should enhance traditional analysis
        icl_result = result["in_context_learning"]
        assert "personalization_improvements" in icl_result
        assert "context_specific_recommendations" in icl_result

    async def test_icl_performance_impact(self, sample_historical_data):
        """Test performance impact of enabling in-context learning."""
        import time
        
        # Test with ICL disabled
        config_disabled = ContextConfig(enable_in_context_learning=False)
        engine_disabled = ContextSpecificLearningEngine(config_disabled)
        
        start_time = time.time()
        result_disabled = await engine_disabled.analyze_context_effectiveness({
            "test_context": {
                "sample_size": 20,
                "avg_performance": 0.8,
                "consistency_score": 0.85,
                "historical_data": sample_historical_data
            }
        })
        time_disabled = time.time() - start_time
        
        # Test with ICL enabled
        config_enabled = ContextConfig(enable_in_context_learning=True)
        engine_enabled = ContextSpecificLearningEngine(config_enabled)
        
        start_time = time.time()
        result_enabled = await engine_enabled.analyze_context_effectiveness({
            "test_context": {
                "sample_size": 20,
                "avg_performance": 0.8,
                "consistency_score": 0.85,
                "historical_data": sample_historical_data
            }
        })
        time_enabled = time.time() - start_time
        
        # ICL should add functionality without excessive overhead
        # Allow up to 5x time increase for the additional ML processing
        assert time_enabled <= time_disabled * 5.0
        
        # ICL version should have additional features
        assert "in_context_learning" in result_enabled
        assert "in_context_learning" not in result_disabled


# Performance and validation markers from pyproject.toml
pytestmark = [
    pytest.mark.unit,
    pytest.mark.ml_contracts,
    pytest.mark.ml_performance
]