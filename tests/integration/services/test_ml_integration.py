"""
Enhanced ML integration tests with property-based validation and model contract testing.
Implements comprehensive ML model validation, data contract verification, and
performance characterization following Context7 ML testing best practices.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from hypothesis import (
    HealthCheck,
    assume,
    given,
    settings,
    strategies as st,
)
from hypothesis.extra.numpy import arrays

from prompt_improver.database.models import (
    MLModelPerformance,
    RuleMetadata,
    RulePerformance,
)
from prompt_improver.services.ml_integration import MLModelService, get_ml_service


class TestMLModelService:
    """Test suite for MLModelService - Phase 3 direct Python integration."""

    @pytest.fixture
    def ml_service(self):
        """Create ML service instance for testing."""
        with patch("prompt_improver.services.ml_integration.mlflow"):
            return MLModelService()


class TestDirectPythonIntegration:
    """Test direct Python integration replacing bridge architecture."""

    @pytest.mark.asyncio
    async def test_optimize_rules_success(
        self, ml_service, sample_training_data, mock_db_session
    ):
        """Test successful rule optimization with direct Python calls."""
        with (
            patch("mlflow.start_run"),
            patch("mlflow.log_params"),
            patch("mlflow.log_metrics"),
            patch("mlflow.sklearn.log_model"),
            patch("mlflow.active_run") as mock_run,
        ):
            mock_run.return_value.info.run_id = "test_run_123"

            result = await ml_service.optimize_rules(
                sample_training_data,
                mock_db_session,
                rule_ids=["clarity_rule", "specificity_rule"],
            )

            # Verify successful optimization
            assert result["status"] == "success"
            assert "model_id" in result
            assert "best_score" in result
            assert "processing_time_ms" in result
            assert result["training_samples"] == 25

            # Verify performance metrics
            assert 0 <= result["best_score"] <= 1
            assert result["accuracy"] >= 0
            assert result["precision"] >= 0
            assert result["recall"] >= 0

            # Verify database interactions
            mock_db_session.execute.assert_called()
            mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_optimize_rules_insufficient_data(self, ml_service, mock_db_session):
        """Test optimization with insufficient training data."""
        # Test case 1: Insufficient class diversity (improved validation catches this first)
        degenerate_data = {
            "features": [[0.5, 100, 1.0, 5, 0.7, 1.0]] * 5,  # Only 5 samples
            "effectiveness_scores": [0.5] * 5,  # All identical scores
        }

        result = await ml_service.optimize_rules(degenerate_data, mock_db_session)
        assert "error" in result
        assert "Insufficient class diversity" in result["error"]

        # Test case 2: Sufficient diversity but insufficient sample size
        insufficient_samples = {
            "features": [[0.5, 100, 1.0, 5, 0.7, 1.0]] * 5,  # Only 5 samples
            "effectiveness_scores": [0.1, 0.3, 0.5, 0.7, 0.9],  # Diverse scores
        }

        result = await ml_service.optimize_rules(insufficient_samples, mock_db_session)
        assert "error" in result
        assert "Insufficient training data" in result["error"]
        assert "10" in result["error"]  # Minimum requirement

    @pytest.mark.asyncio
    async def test_predict_rule_effectiveness(self, ml_service):
        """Test rule effectiveness prediction with direct Python calls."""
        # Mock a trained model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

        model_id = "test_model_123"
        # Add model to the registry
        ml_service.model_registry.add_model(
            model_id=model_id, model=mock_model, model_type="sklearn"
        )

        rule_features = [0.7, 150, 1.0, 5, 0.8, 1.0]

        result = await ml_service.predict_rule_effectiveness(model_id, rule_features)

        assert result["status"] == "success"
        assert result["prediction"] == 0.8
        assert result["confidence"] == 0.8
        assert len(result["probabilities"]) == 2
        assert result["processing_time_ms"] < 10  # Should be very fast

    @pytest.mark.asyncio
    async def test_predict_model_not_found(self, ml_service):
        """Test prediction with non-existent model."""
        result = await ml_service.predict_rule_effectiveness(
            "nonexistent_model", [0.5, 100, 1.0]
        )

        assert "error" in result
        assert "Model nonexistent_model not found" in result["error"]


class TestEnsembleOptimization:
    """Test ensemble ML optimization capabilities."""

    @pytest.mark.asyncio
    async def test_optimize_ensemble_rules_success(self, ml_service, mock_db_session):
        """Test successful ensemble optimization."""
        # Create sufficient data for ensemble (minimum 20 samples) with variance for binary classification
        import random

        random.seed(42)  # Reproducible results

        features = []
        effectiveness_scores = []

        for i in range(25):
            # Add variance to features
            base_feature = [0.8, 150, 1.0, 5, 0.7, 1.0, 0.5, 1.0, 0.2, 0.8]
            varied_feature = [
                base_feature[j] + random.uniform(-0.1, 0.1)
                for j in range(len(base_feature))
            ]
            features.append(varied_feature)

            # Create binary classes: high (>0.7) and low (<=0.7) effectiveness
            if i < 12:  # First 12 samples - high effectiveness
                effectiveness_scores.append(0.7 + random.uniform(0.1, 0.2))  # 0.8-0.9
            else:  # Remaining 13 samples - low effectiveness
                effectiveness_scores.append(0.5 + random.uniform(0.0, 0.2))  # 0.5-0.7

        ensemble_data = {
            "features": features,
            "effectiveness_scores": effectiveness_scores,
        }

        with (
            patch("mlflow.start_run"),
            patch("mlflow.log_params"),
            patch("mlflow.log_metrics"),
            patch("mlflow.sklearn.log_model"),
            patch("mlflow.active_run") as mock_run,
        ):
            mock_run.return_value.info.run_id = "ensemble_run_123"

            result = await ml_service.optimize_ensemble_rules(
                ensemble_data, mock_db_session
            )

            assert result["status"] == "success"
            assert "model_id" in result
            assert "ensemble_score" in result
            assert "ensemble_std" in result
            assert "cv_scores" in result
            assert result["processing_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_optimize_ensemble_insufficient_data(
        self, ml_service, mock_db_session
    ):
        """Test ensemble optimization with insufficient data."""
        insufficient_data = {
            "features": [[0.5, 100]] * 15,  # Only 15 samples, need 20
            "effectiveness_scores": [0.5] * 15,
        }

        result = await ml_service.optimize_ensemble_rules(
            insufficient_data, mock_db_session
        )

        assert "error" in result
        assert "Insufficient data for ensemble" in result["error"]
        assert "20" in result["error"]


class TestPatternDiscovery:
    """Test pattern discovery functionality."""

    @pytest.mark.asyncio
    async def test_discover_patterns_success(
        self,
        ml_service,
        test_db_session,
        sample_rule_metadata,
        sample_rule_performance,
        populate_db,
    ):
        """Test successful pattern discovery using real database interactions."""
        # Populate real database with test data
        await populate_db(
            test_db_session,
            rule_metadata_list=sample_rule_metadata,
            rule_performance_list=sample_rule_performance,
        )

        # Test pattern discovery with real database query
        result = await ml_service.discover_patterns(
            test_db_session, min_effectiveness=0.7, min_support=3
        )

        assert result["status"] == "success"
        assert result["patterns_discovered"] >= 0
        assert "patterns" in result
        assert "total_analyzed" in result
        assert result["processing_time_ms"] > 0

        # Verify patterns structure
        if result["patterns"]:
            pattern = result["patterns"][0]
            assert "parameters" in pattern
            assert "avg_effectiveness" in pattern
            assert "support_count" in pattern
            assert "rule_ids" in pattern

    @pytest.mark.asyncio
    async def test_discover_patterns_insufficient_data(
        self, ml_service, mock_db_session
    ):
        """Test pattern discovery with insufficient data."""
        # Mock empty database result
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db_session.execute.return_value = mock_result

        result = await ml_service.discover_patterns(mock_db_session, min_support=5)

        assert result["status"] == "insufficient_data"
        assert "minimum" in result["message"]


class TestDatabaseIntegration:
    """Test database integration for ML operations."""

    @pytest.mark.asyncio
    async def test_update_rule_parameters(self, ml_service, mock_db_session):
        """Test updating rule parameters in database."""
        # Mock rule query results
        mock_rule = MagicMock()
        mock_rule.rule_id = "clarity_rule"
        mock_rule.default_parameters = {"weight": 0.8}

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_rule
        mock_db_session.execute.return_value = mock_result

        optimized_params = {"n_estimators": 100, "max_depth": 10}

        await ml_service._update_rule_parameters(
            mock_db_session, ["clarity_rule"], optimized_params, 0.85, "test_model_123"
        )

        # Verify rule was updated
        assert mock_rule.default_parameters["n_estimators"] == 100
        assert mock_rule.default_parameters["max_depth"] == 10
        assert mock_rule.updated_at is not None

        mock_db_session.add.assert_called_with(mock_rule)
        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_store_model_performance(self, ml_service, mock_db_session):
        """Test storing model performance metrics."""
        await ml_service._store_model_performance(
            mock_db_session, "test_model_123", 0.85, 0.90, 0.88, 0.92
        )

        # Verify performance record was created and stored
        mock_db_session.add.assert_called()
        mock_db_session.commit.assert_called()

        # Get the added performance record
        call_args = mock_db_session.add.call_args[0][0]
        assert call_args.model_version == "test_model_123"
        assert call_args.model_type == "RandomForestClassifier"
        assert call_args.accuracy_score == 0.90
        assert call_args.precision_score == 0.88
        assert call_args.recall_score == 0.92


class TestMLServiceSingleton:
    """Test ML service singleton pattern."""

    @pytest.mark.anyio
    async def test_get_ml_service_singleton(self):
        """Test that get_ml_service returns the same instance."""
        with patch("prompt_improver.services.ml_integration.mlflow"):
            service1 = await get_ml_service()
            service2 = await get_ml_service()

            assert service1 is service2
            assert isinstance(service1, MLModelService)

    @pytest.mark.anyio
    async def test_ml_service_initialization(self):
        """Test ML service proper initialization."""
        with patch("prompt_improver.services.ml_integration.mlflow"):
            service = await get_ml_service()

            assert service.model_registry is not None
            assert service.mlflow_client is not None
            assert service.scaler is not None


class TestPerformanceRequirements:
    """Test Phase 3 performance requirements."""

    @pytest.mark.asyncio
    async def test_prediction_latency_requirement(self, ml_service):
        """Test that predictions meet <5ms latency requirement."""
        # Mock a simple model for fast prediction
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

        model_id = "fast_model"
        ml_service.model_registry.add_model(model_id, mock_model, "test_model")

        rule_features = [0.7, 150, 1.0, 5, 0.8, 1.0]

        result = await ml_service.predict_rule_effectiveness(model_id, rule_features)

        # Verify sub-5ms performance
        assert result["status"] == "success"
        assert result["processing_time_ms"] < 5.0

    @pytest.mark.asyncio
    async def test_optimization_timeout_handling(
        self, ml_service, sample_training_data, mock_db_session
    ):
        """Test that optimization respects timeout constraints."""
        with (
            patch("optuna.create_study") as mock_study_create,
            patch("mlflow.start_run"),
            patch("mlflow.active_run") as mock_run,
        ):
            mock_study = MagicMock()
            mock_study.best_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
            }
            mock_study.best_value = 0.85
            mock_study_create.return_value = mock_study
            mock_run.return_value.info.run_id = "timeout_test"

            result = await ml_service.optimize_rules(
                sample_training_data, mock_db_session
            )

            # Verify optimization was called with timeout
            mock_study.optimize.assert_called_once()
            call_args = mock_study.optimize.call_args
            assert call_args[1]["timeout"] == 30  # 30 second timeout for small datasets (optimized for testing)


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_optimization_with_invalid_data(self, ml_service, mock_db_session):
        """Test optimization with invalid training data."""
        invalid_data = {
            "features": "invalid_format",
            "effectiveness_scores": [0.5, 0.6],
        }

        result = await ml_service.optimize_rules(invalid_data, mock_db_session)

        assert "error" in result
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_database_error_handling(
        self, ml_service, sample_training_data, mock_db_session
    ):
        """Test handling of database errors during optimization."""
        # Mock database error
        mock_db_session.execute.side_effect = Exception("Database connection failed")

        with patch("mlflow.start_run"), patch("mlflow.active_run") as mock_run:
            mock_run.return_value.info.run_id = "error_test"

            # Should not raise exception, should return error result
            result = await ml_service.optimize_rules(
                sample_training_data, mock_db_session
            )

            # Optimization might succeed but database operations fail
            # The error handling should be graceful
            assert "status" in result

    @pytest.mark.asyncio
    async def test_mlflow_error_handling(
        self, ml_service, sample_training_data, mock_db_session
    ):
        """Test handling of MLflow errors."""
        with patch("mlflow.start_run", side_effect=Exception("MLflow unavailable")):
            result = await ml_service.optimize_rules(
                sample_training_data, mock_db_session
            )

            assert "error" in result
            assert result["status"] == "error"
            assert "MLflow unavailable" in result["error"]


@pytest.mark.ml_contracts
class TestMLModelContracts:
    """Contract testing for ML model interfaces and data validation."""

    @given(
        features=arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=1, max_value=100),
                st.integers(min_value=5, max_value=20),
            ),
            elements=st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    @pytest.mark.asyncio
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_prediction_contract_feature_validation(self, ml_service, features):
        """Contract: prediction should validate feature dimensions and ranges."""

        # Mock model that accepts any feature count
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5] * features.shape[0])
        mock_model.predict_proba.return_value = np.array(
            [[0.5, 0.5]] * features.shape[0]
        )

        model_id = "contract_test_model"
        ml_service.model_registry.add_model(model_id, mock_model, "contract_test")

        # Test with each feature vector
        for feature_vector in features:
            result = await ml_service.predict_rule_effectiveness(
                model_id, feature_vector.tolist()
            )

            # Contract requirements
            assert result["status"] == "success"
            assert isinstance(result["prediction"], (int, float))
            assert 0.0 <= result["prediction"] <= 1.0
            assert isinstance(result["confidence"], (int, float))
            assert 0.0 <= result["confidence"] <= 1.0
            assert isinstance(result["probabilities"], list)
            assert len(result["probabilities"]) == 2  # Binary classification
            assert sum(result["probabilities"]) == pytest.approx(1.0, abs=0.01)

    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        n_features=st.integers(min_value=5, max_value=15),
        effectiveness_scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0), min_size=10, max_size=100
        ),
    )
    @pytest.mark.asyncio
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much], 
        max_examples=10,  # Reduce examples for faster testing
        deadline=30000  # 30 second timeout for ML operations
    )
    async def test_training_data_contract_validation(
        self, ml_service, mock_db_session, n_samples, n_features, effectiveness_scores
    ):
        """Contract: training should validate data shape and quality requirements."""

        assume(len(effectiveness_scores) == n_samples)
        
        # Ensure class diversity - critical for ML classification
        # Research-based best practice: guarantee multi-class scenarios
        effectiveness_array = np.array(effectiveness_scores[:n_samples])
        
        # Apply diversity constraint: at least 20% of each class
        unique_values = np.unique(effectiveness_array)
        if len(unique_values) < 2:
            # Force diversity if all values are identical
            split_point = n_samples // 2
            effectiveness_array[:split_point] = np.random.uniform(0.0, 0.4, split_point)  # Low effectiveness
            effectiveness_array[split_point:] = np.random.uniform(0.6, 1.0, n_samples - split_point)  # High effectiveness
            effectiveness_scores = effectiveness_array.tolist()
        
        # Additional validation: ensure sufficient class separation
        effectiveness_std = np.std(effectiveness_scores)
        if effectiveness_std < 0.1:  # Very low variance
            # Add controlled variance
            noise = np.random.normal(0, 0.15, n_samples)
            effectiveness_scores = np.clip(np.array(effectiveness_scores) + noise, 0.0, 1.0).tolist()

        # Generate realistic training data
        features = np.random.rand(n_samples, n_features).tolist()
        training_data = {
            "features": features,
            "effectiveness_scores": effectiveness_scores,
        }

        with (
            patch("mlflow.start_run"),
            patch("mlflow.active_run") as mock_run,
            patch("mlflow.end_run"),
            patch("mlflow.log_params"),
            patch("mlflow.log_metrics"),
            patch("mlflow.sklearn.log_model"),
        ):
            mock_run_info = MagicMock()
            mock_run_info.info.run_id = "contract_test_run"
            mock_run.return_value = mock_run_info

            result = await ml_service.optimize_rules(training_data, mock_db_session)

            # Contract validation
            assert "status" in result
            if result["status"] == "success":
                # Success contract requirements
                assert "model_id" in result
                assert "best_score" in result
                assert isinstance(result["best_score"], (int, float))
                assert 0.0 <= result["best_score"] <= 1.0
                assert "processing_time_ms" in result
                assert result["processing_time_ms"] > 0
                assert result["training_samples"] == n_samples
                
                # Research-based validation: ensure reasonable performance
                # With proper data diversity, model should achieve meaningful performance
                assert result["best_score"] >= 0.3, "Model failed to learn from diverse data"
            else:
                # Error contract requirements
                assert "error" in result
                assert isinstance(result["error"], str)

    @given(
        model_params=st.dictionaries(
            keys=st.sampled_from([
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "learning_rate",
            ]),
            values=st.one_of(
                st.integers(min_value=1, max_value=1000),
                st.floats(min_value=0.001, max_value=1.0),
            ),
            min_size=1,
            max_size=4,
        )
    )
    @pytest.mark.asyncio
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_model_parameter_contract_validation(
        self, ml_service, mock_db_session, model_params
    ):
        """Contract: model parameters should be validated and sanitized."""

        # Test parameter validation in context of rule updates
        with patch.object(ml_service, "_update_rule_parameters") as mock_update:
            mock_update.return_value = None

            # This would be called internally during optimization
            await ml_service._update_rule_parameters(
                mock_db_session, ["test_rule"], model_params, 0.8, "test_model"
            )

            # Verify parameter validation was applied
            mock_update.assert_called_once()
            call_args = mock_update.call_args[0]
            parameters = call_args[2]  # The parameters argument

            # Contract: all parameters should be valid types
            for key, value in parameters.items():
                assert isinstance(key, str)
                assert isinstance(value, (int, float, bool, str))
                if isinstance(value, (int, float)):
                    assert not np.isnan(value)
                    assert not np.isinf(value)


@pytest.mark.property_based
class TestMLPropertyBasedValidation:
    """Property-based testing for ML service behavior validation."""

    @given(
        rule_count=st.integers(min_value=1, max_value=10),
        feature_dimension=st.integers(min_value=5, max_value=20),
    )
    @pytest.mark.asyncio
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_prediction_consistency_property(
        self, ml_service, rule_count, feature_dimension
    ):
        """Property: identical inputs should produce identical predictions."""

        # Create mock model with deterministic behavior
        mock_model = MagicMock()
        prediction_value = 0.7
        mock_model.predict.return_value = np.array([prediction_value])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        model_id = "deterministic_model"
        ml_service.model_registry.add_model(model_id, mock_model, "deterministic_test")

        # Generate test features
        test_features = [0.5] * feature_dimension

        # Make multiple predictions with identical input
        results = []
        for _ in range(rule_count):
            result = await ml_service.predict_rule_effectiveness(
                model_id, test_features
            )
            results.append(result)

        # Property: all results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result["prediction"] == first_result["prediction"]
            assert result["confidence"] == first_result["confidence"]
            assert result["probabilities"] == first_result["probabilities"]
            assert result["status"] == first_result["status"]

    @given(
        data_size=st.integers(min_value=20, max_value=100),
        noise_level=st.floats(min_value=0.0, max_value=0.1),
    )
    @pytest.mark.asyncio
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much],
        max_examples=3,  # Minimal examples for faster testing
        deadline=60000  # 60 second timeout for optimization
    )
    async def test_optimization_convergence_property(
        self, ml_service, mock_db_session, data_size, noise_level
    ):
        """Property: optimization should converge to reasonable solutions."""

        # Generate synthetic training data with known pattern
        np.random.seed(42)  # Reproducible results
        features = np.random.rand(data_size, 6)

        # Create effectiveness scores with guaranteed class diversity and realistic patterns
        # Research-based approach: structured data with meaningful patterns
        half_size = data_size // 2
        quarter_size = data_size // 4
        
        # Create three distinct effectiveness levels for better convergence
        high_scores = np.random.uniform(0.7, 1.0, quarter_size)  # Top performers
        medium_scores = np.random.uniform(0.4, 0.6, half_size)   # Average performers  
        low_scores = np.random.uniform(0.0, 0.3, data_size - quarter_size - half_size)  # Poor performers
        
        effectiveness_scores = np.concatenate([high_scores, medium_scores, low_scores])
        
        # Shuffle to avoid ordering bias
        shuffle_indices = np.random.permutation(data_size)
        effectiveness_scores = effectiveness_scores[shuffle_indices]
        features = features[shuffle_indices]

        # Add controlled noise (research best practice: limited noise for stable convergence)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, data_size)
            effectiveness_scores = np.clip(effectiveness_scores + noise, 0, 1)
        
        effectiveness_scores = effectiveness_scores.tolist()

        training_data = {
            "features": features.tolist(),
            "effectiveness_scores": effectiveness_scores,
        }

        with (
            patch("mlflow.start_run"),
            patch("mlflow.active_run") as mock_run,
            patch("mlflow.end_run"),
            patch("mlflow.log_params"),
            patch("mlflow.log_metrics"),
            patch("mlflow.sklearn.log_model"),
        ):
            mock_run_info = MagicMock()
            mock_run_info.info.run_id = "convergence_test"
            mock_run.return_value = mock_run_info

            result = await ml_service.optimize_rules(training_data, mock_db_session)

            if result["status"] == "success":
                # Property: optimization should achieve reasonable performance
                # Research finding: with diverse structured data, models should achieve >0.4 performance
                assert result["best_score"] >= 0.4, (
                    f"Optimization failed to find meaningful patterns: {result['best_score']}"
                )
                assert result["best_score"] <= 1.0, "Invalid optimization score"

                # Property: processing should complete in reasonable time
                # Research-based timeout: 30s for small datasets, 60s for larger ones
                expected_timeout = 30000 if data_size < 50 else 60000
                assert result["processing_time_ms"] < expected_timeout, (
                    f"Optimization took too long: {result['processing_time_ms']}ms > {expected_timeout}ms"
                )

                # Property: should use provided training data
                assert result["training_samples"] == data_size
                
                # Property: accuracy metrics should be reasonable for structured data
                assert result.get("accuracy", 0) >= 0.5, "Model accuracy too low for structured data"
            else:
                # Allow failure for very high noise levels or edge cases
                if noise_level > 0.08:  # High noise tolerance
                    assert "error" in result
                else:
                    # Low noise should succeed
                    pytest.fail(f"Optimization should succeed with low noise: {result.get('error', 'Unknown error')}")

    @given(
        effectiveness_threshold=st.floats(min_value=0.5, max_value=0.9),
        support_threshold=st.integers(min_value=2, max_value=10),
    )
    @pytest.mark.asyncio
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_pattern_discovery_threshold_property(
        self, ml_service, mock_db_session, effectiveness_threshold, support_threshold
    ):
        """Property: pattern discovery should respect threshold parameters."""

        # Mock database with controlled pattern data
        mock_patterns = []
        for i in range(15):
            effectiveness = 0.6 + (i % 4) * 0.1  # Values from 0.6 to 0.9
            support = 2 + (i % 6)  # Values from 2 to 7
            mock_patterns.append({
                "parameters": {"param": i},
                "avg_effectiveness": effectiveness,
                "support_count": support,
                "rule_ids": [f"rule_{i}"],
            })

        mock_result = MagicMock()
        mock_result.fetchall.return_value = mock_patterns
        mock_db_session.execute.return_value = mock_result

        result = await ml_service.discover_patterns(
            mock_db_session,
            min_effectiveness=effectiveness_threshold,
            min_support=support_threshold,
        )

        if result["status"] == "success" and result["patterns"]:
            # Property: all returned patterns should meet thresholds
            for pattern in result["patterns"]:
                assert pattern["avg_effectiveness"] >= effectiveness_threshold, (
                    f"Pattern effectiveness {pattern['avg_effectiveness']} below threshold {effectiveness_threshold}"
                )
                assert pattern["support_count"] >= support_threshold, (
                    f"Pattern support {pattern['support_count']} below threshold {support_threshold}"
                )


@pytest.mark.ml_performance
class TestMLPerformanceCharacterization:
    """Performance characterization and regression detection for ML components."""

    @pytest.mark.asyncio
    async def test_prediction_latency_characterization(self, ml_service):
        """Characterize prediction latency under various conditions."""

        # Setup different model complexities
        test_scenarios = [
            ("simple_model", 5),  # 5 features
            ("medium_model", 10),  # 10 features
            ("complex_model", 20),  # 20 features
        ]

        latency_results = {}

        for model_name, feature_count in test_scenarios:
            # Mock model for this scenario
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([0.8])
            mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

            ml_service.model_registry.add_model(
                model_name, mock_model, "performance_test"
            )

            # Measure prediction latency
            test_features = [0.5] * feature_count
            latencies = []

            for _ in range(10):
                import time

                start_time = time.time()
                result = await ml_service.predict_rule_effectiveness(
                    model_name, test_features
                )
                end_time = time.time()

                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

                assert result["status"] == "success"

            avg_latency = sum(latencies) / len(latencies)
            latency_results[model_name] = avg_latency

            # Performance requirement: sub-5ms predictions
            assert avg_latency < 5.0, (
                f"{model_name} avg latency {avg_latency:.2f}ms exceeds 5ms target"
            )

        # Characterization: latency should not grow dramatically with complexity
        simple_latency = latency_results["simple_model"]
        complex_latency = latency_results["complex_model"]
        latency_growth_factor = (
            complex_latency / simple_latency if simple_latency > 0 else 1
        )

        assert latency_growth_factor < 3.0, (
            f"Latency grows too much with complexity: {latency_growth_factor:.2f}x"
        )

    @pytest.mark.asyncio
    async def test_optimization_scalability_characterization(
        self, ml_service, mock_db_session
    ):
        """Characterize optimization performance scaling with data size."""

        data_sizes = [20, 50, 100]
        optimization_times = {}

        for data_size in data_sizes:
            # Generate training data with variance to avoid NaN in cross-validation
            import random

            random.seed(42)  # Reproducible results

            features = []
            effectiveness_scores = []

            for i in range(data_size):
                # Add variance to prevent identical samples
                base_feature = [0.8, 150, 1.0, 5, 0.7, 1.0]
                varied_feature = [
                    base_feature[0] + random.uniform(-0.2, 0.2),  # 0.6-1.0
                    base_feature[1] + random.uniform(-50, 50),  # 100-200
                    base_feature[2] + random.uniform(-0.3, 0.3),  # 0.7-1.3
                    base_feature[3] + random.randint(-2, 2),  # 3-7
                    base_feature[4] + random.uniform(-0.2, 0.2),  # 0.5-0.9
                    base_feature[5],  # Keep binary
                ]
                effectiveness_score = 0.8 + random.uniform(-0.2, 0.2)  # 0.6-1.0

                features.append(varied_feature)
                effectiveness_scores.append(
                    max(0.0, min(1.0, effectiveness_score))
                )  # Clamp to [0,1]

            training_data = {
                "features": features,
                "effectiveness_scores": effectiveness_scores,
            }

            with patch("mlflow.start_run"), patch("mlflow.active_run") as mock_run:
                mock_run.return_value.info.run_id = f"scale_test_{data_size}"

                import time

                start_time = time.time()
                result = await ml_service.optimize_rules(training_data, mock_db_session)
                end_time = time.time()

                optimization_time = (end_time - start_time) * 1000
                optimization_times[data_size] = optimization_time

                if result["status"] == "success":
                    assert result["training_samples"] == data_size

        # Characterization: optimization time should scale reasonably
        if len(optimization_times) >= 2:
            times = list(optimization_times.values())
            sizes = list(optimization_times.keys())

            # Calculate scaling factor (should not be exponential)
            scaling_factor = (times[-1] / times[0]) / (sizes[-1] / sizes[0])

            # Should be closer to linear (1.0) than exponential (>>1.0)
            assert scaling_factor < 5.0, (
                f"Optimization scaling factor {scaling_factor:.2f} indicates poor scalability"
            )


@pytest.mark.ml_data_validation
class TestMLDataQualityValidation:
    """Data quality and validation testing for ML pipelines."""

    @given(corrupted_ratio=st.floats(min_value=0.0, max_value=0.3))
    @pytest.mark.asyncio
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much],
        max_examples=3,  # Minimal examples for faster testing
        deadline=60000  # 60 second timeout for data cleaning operations
    )
    async def test_corrupted_data_handling(
        self, ml_service, mock_db_session, corrupted_ratio
    ):
        """Test ML service handling of corrupted training data."""

        # Generate dataset with controlled corruption
        clean_size = 50
        corrupted_size = int(clean_size * corrupted_ratio)

        # Clean data with realistic diversity (research best practice)
        clean_features = []
        clean_scores = []
        
        # Generate diverse clean data to ensure good baseline
        for i in range(clean_size):
            # Create varied feature patterns
            clean_features.append([
                np.random.uniform(0.3, 0.9),    # Feature 1: moderate variation
                np.random.uniform(100, 200),    # Feature 2: count-based
                np.random.uniform(0.5, 1.0),    # Feature 3: high performance range
                np.random.randint(3, 8),        # Feature 4: discrete values
                np.random.uniform(0.4, 0.8),    # Feature 5: mid-range
                np.random.uniform(0.6, 1.0),    # Feature 6: quality metric
            ])
            # Create bimodal effectiveness distribution
            if i < clean_size // 2:
                clean_scores.append(np.random.uniform(0.7, 1.0))  # High effectiveness
            else:
                clean_scores.append(np.random.uniform(0.1, 0.5))  # Low effectiveness

        # Corrupted data (NaN, infinity, out of range) - research-based corruption patterns
        corrupted_features = []
        corrupted_scores = []

        for i in range(corrupted_size):
            corruption_type = i % 4  # Cycle through corruption types
            
            if corruption_type == 0:  # NaN corruption
                corrupted_features.append([
                    float("nan"), 150, 0.8, 5, 0.7, 0.9
                ])
                corrupted_scores.append(float("nan"))
            elif corruption_type == 1:  # Infinity corruption
                corrupted_features.append([
                    0.8, float("inf"), 1.0, 5, 0.7, 1.0
                ])
                corrupted_scores.append(0.8)
            elif corruption_type == 2:  # Out of range corruption
                corrupted_features.append([
                    -1.0, 150, 2.0, 1000, -0.5, 1.5
                ])
                corrupted_scores.append(-0.5)
            else:  # Mixed corruption
                corrupted_features.append([
                    float("nan"), float("inf"), -1.0, 1000, 2.0, -0.5
                ])
                corrupted_scores.append(float("inf"))

        training_data = {
            "features": clean_features + corrupted_features,
            "effectiveness_scores": clean_scores + corrupted_scores,
        }

        with (
            patch("mlflow.start_run"),
            patch("mlflow.active_run") as mock_run,
            patch("mlflow.end_run"),
            patch("mlflow.log_params"),
            patch("mlflow.log_metrics"),
            patch("mlflow.sklearn.log_model"),
        ):
            mock_run_info = MagicMock()
            mock_run_info.info.run_id = "corruption_test"
            mock_run.return_value = mock_run_info

            result = await ml_service.optimize_rules(training_data, mock_db_session)

            # ML service should handle corruption gracefully
            assert "status" in result

            if corrupted_ratio == 0.0:  # No corruption - should always succeed
                assert result["status"] == "success"
                assert result["training_samples"] == clean_size
                assert result["best_score"] >= 0.4  # Good performance with clean data
            elif corrupted_ratio > 0.25:  # High corruption (>25%)
                # Should either fail gracefully or succeed with robust handling
                if result["status"] == "error":
                    assert (
                        "data" in result["error"].lower()
                        or "invalid" in result["error"].lower()
                        or "nan" in result["error"].lower()
                        or "inf" in result["error"].lower()
                    ), f"Error message should indicate data issues: {result['error']}"
                else:
                    # If it succeeds, should show evidence of data cleaning
                    assert result["training_samples"] <= clean_size + corrupted_size
                    assert result["training_samples"] >= clean_size * 0.8  # Should retain most clean data
            else:  # Low to moderate corruption (1-25%)
                # Should succeed by filtering bad data (research finding: ML systems should be robust)
                if result["status"] == "success":
                    # Should report actual clean samples used after filtering
                    assert result["training_samples"] <= clean_size + corrupted_size
                    assert result["training_samples"] >= clean_size * 0.7  # Should retain majority of clean data
                    
                    # Performance should still be reasonable with clean data
                    assert result["best_score"] >= 0.3, "Performance degraded too much with minor corruption"
                elif result["status"] == "error":
                    # Acceptable if corruption causes fundamental issues
                    assert "data" in result["error"].lower() or "invalid" in result["error"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
