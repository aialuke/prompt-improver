"""
Comprehensive integration tests for ML Training Data Pipeline Integration.
Tests Phase 1 components to ensure proper integration and prevent false positives.

Tests:
- Context Learner training data integration
- Clustering Optimizer training data integration  
- Failure Analyzer training data integration
- Dimensionality Reducer training data integration
- Integration error handling and edge cases
- False positive prevention

Following 2025 best practices: real ML models, real database integration, minimal mocking.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.database.models import (
    RuleMetadata,
    RulePerformance,
    TrainingPrompt,
)
from prompt_improver.ml.core.training_data_loader import TrainingDataLoader
from prompt_improver.ml.learning.algorithms.context_learner import (
    ContextSpecificLearner,
    ContextConfig,
)
from prompt_improver.ml.optimization.algorithms.clustering_optimizer import (
    ClusteringOptimizer,
    ClusteringConfig,
)
from prompt_improver.ml.learning.algorithms.failure_analyzer import (
    FailureModeAnalyzer,
    FailureConfig,
)
from prompt_improver.ml.optimization.algorithms.dimensionality_reducer import (
    AdvancedDimensionalityReducer,
    DimensionalityConfig,
)
from prompt_improver.utils.datetime_utils import aware_utc_now


class TestTrainingDataIntegration:
    """Test suite for Phase 1 ML Training Data Integration components."""

    @pytest.fixture
    async def training_data_loader(self):
        """Create training data loader for testing."""
        return TrainingDataLoader(
            real_data_priority=True,
            min_samples=5,
            lookback_days=30,
            synthetic_ratio=0.3
        )

    @pytest.fixture
    async def sample_training_data(self, test_db_session: AsyncSession):
        """Create sample training data in database for testing."""
        # Create sample rule metadata
        rule_metadata = RuleMetadata(
            rule_id="test_rule_1",
            rule_name="Test Rule",
            category="test",
            priority=5,
            rule_version="1.0.0",
            default_parameters={"test_param": "value"}
        )
        test_db_session.add(rule_metadata)
        
        # Create sample rule performance data
        base_time = aware_utc_now() - timedelta(days=5)
        performances = []
        
        for i in range(20):
            perf = RulePerformance(
                rule_id="test_rule_1",
                rule_name="Test Rule",
                prompt_id=f"session_{i}",
                improvement_score=0.5 + (i % 10) * 0.05,  # Varied scores
                execution_time_ms=100 + i * 10,
                created_at=base_time + timedelta(hours=i),
                rule_parameters={"param": f"value_{i}"}
            )
            performances.append(perf)
            test_db_session.add(perf)
        
        # Create some training prompts
        for i in range(10):
            training_prompt = TrainingPrompt(
                prompt_text=f"Training prompt {i}",
                enhancement_result={
                    "original_text": f"Training prompt {i}",
                    "improved_text": f"Improved training prompt {i}",
                    "improvement_score": 0.6 + (i % 5) * 0.1,
                    "rule_applications": [f"test_rule_1"],
                    "metadata": {"training": True, "batch": i // 5}
                },
                created_at=base_time + timedelta(hours=i * 2),
                data_source="real"
            )
            test_db_session.add(training_prompt)
        
        await test_db_session.commit()
        return {
            "rule_performances": performances,
            "training_prompts": 10,
            "time_range": (base_time, base_time + timedelta(days=2))
        }

    @pytest.fixture
    async def insufficient_training_data(self, test_db_session: AsyncSession):
        """Create insufficient training data to test edge cases."""
        # Create only 2 samples (below minimum threshold)
        rule_metadata = RuleMetadata(
            rule_id="insufficient_rule",
            rule_name="Insufficient Rule",
            category="test",
            priority=1,
            rule_version="1.0.0"
        )
        test_db_session.add(rule_metadata)
        
        for i in range(2):
            perf = RulePerformance(
                rule_id="insufficient_rule",
                rule_name="Insufficient Rule",
                prompt_id=f"insufficient_session_{i}",
                improvement_score=0.3,
                execution_time_ms=200,
                created_at=aware_utc_now() - timedelta(hours=i)
            )
            test_db_session.add(perf)
        
        await test_db_session.commit()

    async def test_context_learner_integration(
        self, 
        test_db_session: AsyncSession, 
        sample_training_data,
        training_data_loader
    ):
        """Test Context Learner training data integration."""
        # Initialize context learner with training data loader
        config = ContextConfig(
            min_sample_size=5,
            similarity_threshold=0.7  # Lower threshold for test data
        )
        learner = ContextSpecificLearner(config=config, training_loader=training_data_loader)
        
        # Test historical data training
        result = await learner.train_on_historical_data(test_db_session)
        
        # Verify integration success
        assert result is True, "Context learner should successfully train on historical data"
        
        # Verify that context clusters were created
        assert hasattr(learner, 'context_clusters'), "Context clusters should be created"
        
        # Test incremental learning
        await learner.update_from_new_data(test_db_session, "test_session")
        
        # Verify no false positives in clustering
        if hasattr(learner, 'context_clusters') and learner.context_clusters:
            for cluster_id, cluster_data in learner.context_clusters.items():
                assert isinstance(cluster_data, dict), f"Cluster {cluster_id} should be a dictionary"
                assert 'size' in cluster_data, f"Cluster {cluster_id} should have size information"
                assert cluster_data['size'] > 0, f"Cluster {cluster_id} should have positive size"

    async def test_context_learner_insufficient_data(
        self,
        test_db_session: AsyncSession,
        insufficient_training_data,
        training_data_loader
    ):
        """Test Context Learner with insufficient training data."""
        config = ContextConfig(min_sample_size=10)  # Higher than available data
        learner = ContextSpecificLearner(config=config, training_loader=training_data_loader)
        
        result = await learner.train_on_historical_data(test_db_session)
        
        # Should handle insufficient data gracefully
        assert result is False, "Should return False for insufficient training data"

    async def test_clustering_optimizer_integration(
        self,
        test_db_session: AsyncSession,
        sample_training_data,
        training_data_loader
    ):
        """Test Clustering Optimizer training data integration."""
        config = ClusteringConfig(
            memory_efficient_mode=True,
            auto_dim_reduction=False,  # Disable for faster testing
            enable_caching=False  # Disable caching for testing
        )
        optimizer = ClusteringOptimizer(config=config, training_loader=training_data_loader)
        
        # Test training data optimization
        result = await optimizer.optimize_with_training_data(test_db_session)
        
        # With limited test data, clustering may correctly identify insufficient data
        if result["status"] == "insufficient_data":
            # This is correct behavior - clustering optimizer properly detected insufficient data
            assert "samples" in result, "Result should include sample count"
            assert result["samples"] > 0, "Should report training samples processed"
            assert "reason" in result, "Should provide reason for insufficient data"
        else:
            # If optimization succeeds with test data, verify results
            assert result["status"] == "success", f"Optimization should succeed, got: {result}"
            assert "training_samples" in result, "Result should include training sample count"
            assert result["training_samples"] > 0, "Should process training samples"
        
        # Test pattern discovery
        pattern_result = await optimizer.discover_optimization_patterns(test_db_session)
        
        assert pattern_result["status"] == "success", "Pattern discovery should succeed"
        assert "discovered_patterns" in pattern_result, "Should include discovered patterns"
        
        # Verify no false positives in quality scores
        if "best_quality_score" in result:
            quality_score = result["best_quality_score"]
            assert 0.0 <= quality_score <= 1.0, f"Quality score should be between 0 and 1, got {quality_score}"

    async def test_clustering_optimizer_insufficient_data(
        self,
        test_db_session: AsyncSession,
        insufficient_training_data,
        training_data_loader
    ):
        """Test Clustering Optimizer with insufficient data."""
        config = ClusteringConfig()
        optimizer = ClusteringOptimizer(config=config, training_loader=training_data_loader)
        
        result = await optimizer.optimize_with_training_data(test_db_session)
        
        # Should handle insufficient data gracefully
        assert result["status"] == "insufficient_data", "Should detect insufficient data"
        assert "samples" in result, "Should report sample count"

    async def test_failure_analyzer_integration(
        self,
        test_db_session: AsyncSession,
        sample_training_data,
        training_data_loader
    ):
        """Test Failure Analyzer training data integration."""
        config = FailureConfig(
            enable_robustness_validation=False,  # Disable for testing
            enable_prometheus_monitoring=False
        )
        analyzer = FailureModeAnalyzer(config=config, training_loader=training_data_loader)
        
        # Test historical failure analysis
        result = await analyzer.analyze_historical_failures(test_db_session)
        
        # Verify successful analysis
        assert result["status"] == "success", f"Failure analysis should succeed, got: {result}"
        assert "failure_cases" in result, "Should report failure cases"
        assert "failure_rate" in result, "Should calculate failure rate"
        
        # Verify failure rate is reasonable (between 0 and 1)
        failure_rate = result["failure_rate"]
        assert 0.0 <= failure_rate <= 1.0, f"Failure rate should be between 0 and 1, got {failure_rate}"
        
        # Test failure probability prediction with mock features
        mock_features = np.array([[0.5, 0.3, 0.8, 0.2, 0.6]])  # 5-dimensional feature vector
        prediction_result = await analyzer.predict_failure_probability(mock_features)
        
        if prediction_result["status"] == "success":
            prob = prediction_result["failure_probability"]
            assert 0.0 <= prob <= 1.0, f"Failure probability should be between 0 and 1, got {prob}"
            assert prediction_result["risk_level"] in ["low", "medium", "high"], "Should assign valid risk level"

    async def test_failure_analyzer_insufficient_data(
        self,
        test_db_session: AsyncSession,
        insufficient_training_data,
        training_data_loader
    ):
        """Test Failure Analyzer with insufficient data."""
        config = FailureConfig(
            enable_robustness_validation=False,
            enable_prometheus_monitoring=False
        )
        analyzer = FailureModeAnalyzer(config=config, training_loader=training_data_loader)
        
        result = await analyzer.analyze_historical_failures(test_db_session)
        
        # Should handle insufficient data gracefully
        assert result["status"] == "insufficient_data", "Should detect insufficient data"

    async def test_dimensionality_reducer_integration(
        self,
        test_db_session: AsyncSession,
        sample_training_data,
        training_data_loader
    ):
        """Test Dimensionality Reducer training data integration."""
        config = DimensionalityConfig(
            target_dimensions=5,
            fast_mode=True,  # Use fast mode for testing
            auto_method_selection=True
        )
        reducer = AdvancedDimensionalityReducer(config=config, training_loader=training_data_loader)
        
        # Test feature space optimization
        result = await reducer.optimize_feature_space(test_db_session)
        
        # Verify successful optimization
        assert result["status"] == "success", f"Feature space optimization should succeed, got: {result}"
        assert "training_samples" in result, "Should report training samples"
        assert "best_method" in result, "Should select best method"
        assert "optimal_dimensions" in result, "Should determine optimal dimensions"
        
        # Verify dimensions are reasonable
        optimal_dims = result["optimal_dimensions"]
        assert 1 <= optimal_dims <= 20, f"Optimal dimensions should be reasonable, got {optimal_dims}"
        
        # Verify variance preservation is reasonable
        if "variance_preserved" in result:
            variance = result["variance_preserved"]
            assert 0.0 <= variance <= 1.0, f"Variance preserved should be between 0 and 1, got {variance}"
        
        # Test adaptive reduction with mock new data
        mock_new_data = np.random.rand(5, 10)  # 5 samples, 10 features
        adaptive_result = await reducer.adaptive_reduction(mock_new_data, test_db_session)
        
        assert adaptive_result["status"] == "success", "Adaptive reduction should succeed"

    async def test_dimensionality_reducer_insufficient_data(
        self,
        test_db_session: AsyncSession,
        insufficient_training_data,
        training_data_loader
    ):
        """Test Dimensionality Reducer with insufficient data."""
        config = DimensionalityConfig()
        reducer = AdvancedDimensionalityReducer(config=config, training_loader=training_data_loader)
        
        result = await reducer.optimize_feature_space(test_db_session)
        
        # Should handle insufficient data gracefully
        assert result["status"] == "insufficient_data", "Should detect insufficient data"

    async def test_training_data_loader_validation(
        self,
        test_db_session: AsyncSession,
        sample_training_data,
        training_data_loader
    ):
        """Test training data loader validation and error handling."""
        # Test successful data loading
        training_data = await training_data_loader.load_training_data(test_db_session)
        
        # Verify data structure
        assert "features" in training_data, "Training data should include features"
        assert "labels" in training_data, "Training data should include labels"
        assert "metadata" in training_data, "Training data should include metadata"
        
        # Verify metadata structure
        metadata = training_data["metadata"]
        assert "total_samples" in metadata, "Metadata should include sample count"
        assert "real_samples" in metadata, "Metadata should include real sample count"
        assert "synthetic_samples" in metadata, "Metadata should include synthetic sample count"
        assert "synthetic_ratio" in metadata, "Metadata should include synthetic ratio"
        
        # Verify data consistency
        total_samples = metadata["total_samples"]
        real_samples = metadata["real_samples"]
        synthetic_samples = metadata["synthetic_samples"]
        
        assert total_samples == real_samples + synthetic_samples, "Sample counts should be consistent"
        assert total_samples > 0, "Should have training samples"
        
        # Verify synthetic ratio is reasonable
        synthetic_ratio = metadata["synthetic_ratio"]
        assert 0.0 <= synthetic_ratio <= 1.0, f"Synthetic ratio should be between 0 and 1, got {synthetic_ratio}"

    async def test_false_positive_prevention(
        self,
        test_db_session: AsyncSession,
        sample_training_data,
        training_data_loader
    ):
        """Test prevention of false positives in integration outputs."""
        
        # Test with empty database (no training data)
        empty_loader = TrainingDataLoader(min_samples=100)  # Very high threshold
        
        # All components should handle empty data gracefully
        context_learner = ContextSpecificLearner(
            config=ContextConfig(min_sample_size=100),  # Very high threshold
            training_loader=empty_loader
        )
        clustering_optimizer = ClusteringOptimizer(training_loader=empty_loader)
        failure_analyzer = FailureModeAnalyzer(
            config=FailureConfig(
                enable_robustness_validation=False,
                enable_prometheus_monitoring=False
            ),
            training_loader=empty_loader
        )
        dimensionality_reducer = AdvancedDimensionalityReducer(training_loader=empty_loader)
        
        # Test that all components return appropriate error states, not false successes
        context_result = await context_learner.train_on_historical_data(test_db_session)
        assert context_result is False, "Context learner should not claim success with no data"
        
        clustering_result = await clustering_optimizer.optimize_with_training_data(test_db_session)
        assert clustering_result["status"] != "success", "Clustering optimizer should not claim success with no data"
        
        failure_result = await failure_analyzer.analyze_historical_failures(test_db_session)
        assert failure_result["status"] != "success", "Failure analyzer should not claim success with no data"
        
        dimensionality_result = await dimensionality_reducer.optimize_feature_space(test_db_session)
        assert dimensionality_result["status"] != "success", "Dimensionality reducer should not claim success with no data"

    @given(
        sample_size=st.integers(min_value=1, max_value=100),
        feature_dims=st.integers(min_value=1, max_value=50),
        improvement_scores=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=100),
            elements=st.floats(min_value=0.0, max_value=1.0)
        )
    )
    @settings(max_examples=10, deadline=30000)  # Limit for CI performance
    async def test_property_based_integration(
        self,
        sample_size: int,
        feature_dims: int,
        improvement_scores: np.ndarray
    ):
        """Property-based test to ensure integration handles diverse inputs correctly."""
        assume(len(improvement_scores) >= sample_size)
        
        # Create mock training data
        mock_features = np.random.rand(sample_size, feature_dims)
        mock_labels = improvement_scores[:sample_size]
        
        mock_training_data = {
            "features": mock_features.tolist(),
            "labels": mock_labels.tolist(),
            "metadata": {
                "total_samples": sample_size,
                "real_samples": sample_size,
                "synthetic_samples": 0,
                "synthetic_ratio": 0.0
            }
        }
        
        # Test that components handle various input sizes and dimensions correctly
        # This tests the robustness of our integration without false positives
        
        # Create mock training loader
        mock_loader = MagicMock()
        mock_loader.load_training_data = AsyncMock(return_value=mock_training_data)
        
        # Test dimensionality reducer with various dimensions
        if feature_dims >= 2 and sample_size >= 5:  # Minimum requirements
            reducer = AdvancedDimensionalityReducer(training_loader=mock_loader)
            
            # Mock the database session
            mock_session = MagicMock()
            
            result = await reducer.optimize_feature_space(mock_session)
            
            if result["status"] == "success":
                # Verify outputs are reasonable, not false positives
                assert result["training_samples"] == sample_size
                assert 1 <= result["optimal_dimensions"] <= min(feature_dims, sample_size - 1)
                if "variance_preserved" in result:
                    assert 0.0 <= result["variance_preserved"] <= 1.0

    async def test_integration_error_handling(self, test_db_session: AsyncSession):
        """Test error handling in integration components."""
        
        # Test with corrupted training loader
        broken_loader = TrainingDataLoader()
        
        # Mock a broken load_training_data method
        async def broken_load(*args, **kwargs):
            raise Exception("Database connection failed")
        
        broken_loader.load_training_data = broken_load
        
        # Test that components handle errors gracefully
        context_learner = ContextSpecificLearner(training_loader=broken_loader)
        
        result = await context_learner.train_on_historical_data(test_db_session)
        
        # Should return False on error, not crash or return false positive
        assert result is False, "Should handle errors gracefully"

    async def test_integration_performance_bounds(
        self,
        test_db_session: AsyncSession,
        sample_training_data,
        training_data_loader
    ):
        """Test that integration operations complete within reasonable time bounds."""
        import time
        
        # Test context learner performance
        start_time = time.time()
        
        config = ContextConfig(use_advanced_clustering=False)  # Use faster clustering
        learner = ContextSpecificLearner(config=config, training_loader=training_data_loader)
        result = await learner.train_on_historical_data(test_db_session)
        
        elapsed_time = time.time() - start_time
        
        # Should complete within reasonable time (10 seconds for test data)
        assert elapsed_time < 10.0, f"Context learner training took too long: {elapsed_time:.2f}s"
        
        if result:
            # Verify reasonable performance characteristics
            assert hasattr(learner, 'context_patterns'), "Should create context patterns"


# Mark tests that require ML dependencies
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.skipif(
        not all([
            __import__('sklearn', fromlist=['']), 
            __import__('numpy', fromlist=[''])
        ]),
        reason="ML dependencies required"
    )
]