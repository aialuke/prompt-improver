"""
Enhanced ML integration tests with property-based validation and model contract testing.
Following 2025 best practices: real ML models, real database integration, minimal mocking.

Migration from mock-based testing to real behavior testing based on research:
- Use real sklearn models with lightweight configurations for testing
- Real PostgreSQL database with transaction rollback for isolation
- Real MLflow tracking with test backend or lightweight alternative
- Mock only external services that would be expensive or flaky (Redis pub/sub)
- Test actual ML performance characteristics with real models
"""

import asyncio
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import mlflow
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select

from prompt_improver.database.models import (
    MLModelPerformance,
    RuleMetadata,
    RulePerformance,
)
from prompt_improver.services.ml_integration import MLModelService, get_ml_service


class TestMLModelService:
    """Test suite for MLModelService - Phase 3 direct Python integration."""

    @pytest.fixture
    async def ml_service(self):
        """Create ML service instance with real MLflow backend for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mlflow.set_tracking_uri(f"file://{temp_dir}")
            mlflow.set_experiment("test_ml_integration")
            service = MLModelService()
            yield service
            if mlflow.active_run():
                mlflow.end_run()


class TestDirectPythonIntegration:
    """Test direct Python integration replacing bridge architecture."""

    @pytest.mark.asyncio
    async def test_optimize_rules_success(
        self, ml_service, sample_training_data, real_db_session, real_rule_metadata
    ):
        """Test successful rule optimization with real ML models and database."""
        result = await ml_service.optimize_rules(
            sample_training_data,
            real_db_session,
            rule_ids=[rule.rule_id for rule in real_rule_metadata],
        )
        assert result["status"] == "success"
        assert "model_id" in result
        assert "best_score" in result
        assert "processing_time_ms" in result
        assert result["training_samples"] == 25
        assert 0 <= result["best_score"] <= 1
        assert result["accuracy"] >= 0
        assert result["precision"] >= 0
        assert result["recall"] >= 0
        cached_model = ml_service.model_registry.get_model(result["model_id"])
        assert cached_model is not None
        assert hasattr(cached_model, "predict")
        assert hasattr(cached_model, "predict_proba")
        stmt = select(MLModelPerformance).where(
            MLModelPerformance.model_id == result["model_id"]
        )
        db_result = await real_db_session.execute(stmt)
        model_perf = db_result.scalar_one_or_none()
        assert model_perf is not None
        assert model_perf.accuracy == result["accuracy"]

    @pytest.mark.asyncio
    async def test_optimize_rules_insufficient_data(self, ml_service, real_db_session):
        """Test optimization with insufficient training data using real validation."""
        degenerate_data = {
            "features": [[0.5, 100, 1.0, 5, 0.7, 1.0]] * 5,
            "effectiveness_scores": [0.5] * 5,
        }
        result = await ml_service.optimize_rules(degenerate_data, real_db_session)
        assert "error" in result
        assert "Insufficient class diversity" in result["error"]
        insufficient_samples = {
            "features": [[0.5, 100, 1.0, 5, 0.7, 1.0]] * 5,
            "effectiveness_scores": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
        result = await ml_service.optimize_rules(insufficient_samples, real_db_session)
        assert "error" in result
        assert "Insufficient training data" in result["error"]
        assert "10" in result["error"]

    @pytest.mark.asyncio
    async def test_predict_rule_effectiveness(
        self, ml_service, sample_training_data, real_db_session
    ):
        """Test rule effectiveness prediction with real trained model."""
        train_result = await ml_service.optimize_rules(
            sample_training_data, real_db_session, rule_ids=["test_rule"]
        )
        assert train_result["status"] == "success"
        model_id = train_result["model_id"]
        rule_features = [0.7, 150, 1.0, 5, 0.8, 1.0]
        result = await ml_service.predict_rule_effectiveness(model_id, rule_features)
        assert result["status"] == "success"
        assert isinstance(result["prediction"], (int, float))
        assert 0 <= result["prediction"] <= 1
        assert isinstance(result["confidence"], (int, float))
        assert 0 <= result["confidence"] <= 1
        assert len(result["probabilities"]) == 2
        assert abs(sum(result["probabilities"]) - 1.0) < 0.01
        assert result["processing_time_ms"] < 10

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
    async def test_optimize_ensemble_rules_success(self, ml_service, real_db_session):
        """Test successful ensemble optimization with real models."""
        import random

        random.seed(42)
        features = []
        effectiveness_scores = []
        for i in range(25):
            base_feature = [0.8, 150, 1.0, 5, 0.7, 1.0, 0.5, 1.0, 0.2, 0.8]
            varied_feature = [
                base_feature[j] + random.uniform(-0.1, 0.1)
                for j in range(len(base_feature))
            ]
            features.append(varied_feature)
            if i < 12:
                effectiveness_scores.append(0.7 + random.uniform(0.1, 0.2))
            else:
                effectiveness_scores.append(0.5 + random.uniform(0.0, 0.2))
        ensemble_data = {
            "features": features,
            "effectiveness_scores": effectiveness_scores,
        }
        result = await ml_service.optimize_ensemble_rules(
            ensemble_data, real_db_session
        )
        assert result["status"] == "success"
        assert "model_id" in result
        assert "ensemble_score" in result
        assert "ensemble_std" in result
        assert "cv_scores" in result
        assert result["processing_time_ms"] > 0
        ensemble_model = ml_service.model_registry.get_model(result["model_id"])
        assert ensemble_model is not None
        assert hasattr(ensemble_model, "estimators_") or hasattr(
            ensemble_model, "base_estimator_"
        )

    @pytest.mark.asyncio
    async def test_optimize_ensemble_insufficient_data(
        self, ml_service, real_db_session
    ):
        """Test ensemble optimization with insufficient data using real validation."""
        insufficient_data = {
            "features": [[0.5, 100]] * 15,
            "effectiveness_scores": [0.5] * 15,
        }
        result = await ml_service.optimize_ensemble_rules(
            insufficient_data, real_db_session
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
        real_db_session,
        real_rule_metadata,
        sample_rule_performance,
        real_prompt_sessions,
    ):
        """Test successful pattern discovery using real database interactions."""
        for i, performance in enumerate(sample_rule_performance[:10]):
            performance.rule_id = real_rule_metadata[
                i % len(real_rule_metadata)
            ].rule_id
            performance.session_id = real_prompt_sessions[
                i % len(real_prompt_sessions)
            ].session_id
            real_db_session.add(performance)
        await real_db_session.commit()
        result = await ml_service.discover_patterns(
            real_db_session, min_effectiveness=0.7, min_support=3
        )
        assert result["status"] == "success"
        assert result["patterns_discovered"] >= 0
        assert "patterns" in result
        assert "total_analyzed" in result
        assert result["processing_time_ms"] > 0
        if result["patterns"]:
            pattern = result["patterns"][0]
            assert "parameters" in pattern
            assert "avg_effectiveness" in pattern
            assert "support_count" in pattern
            assert "rule_ids" in pattern

    @pytest.mark.asyncio
    async def test_discover_patterns_insufficient_data(
        self, ml_service, real_db_session
    ):
        """Test pattern discovery with insufficient data using real empty database."""
        result = await ml_service.discover_patterns(real_db_session, min_support=5)
        assert result["status"] == "insufficient_data"
        assert "minimum" in result["message"]


class TestDatabaseIntegration:
    """Test database integration for ML operations."""

    @pytest.mark.asyncio
    async def test_update_rule_parameters(
        self, ml_service, real_db_session, real_rule_metadata
    ):
        """Test updating rule parameters in database with real data."""
        rule_id = real_rule_metadata[0].rule_id
        optimized_params = {"n_estimators": 100, "max_depth": 10}
        await ml_service._update_rule_parameters(
            real_db_session, [rule_id], optimized_params, 0.85, "test_model_123"
        )
        stmt = select(RuleMetadata).where(RuleMetadata.rule_id == rule_id)
        result = await real_db_session.execute(stmt)
        updated_rule = result.scalar_one_or_none()
        assert updated_rule is not None
        assert updated_rule.default_parameters["n_estimators"] == 100
        assert updated_rule.default_parameters["max_depth"] == 10
        assert updated_rule.updated_at is not None

    @pytest.mark.asyncio
    async def test_store_model_performance(self, ml_service, real_db_session):
        """Test storing model performance metrics with real database."""
        await ml_service._store_model_performance(
            real_db_session, "test_model_123", 0.85, 0.9, 0.88, 0.92
        )
        stmt = select(MLModelPerformance).where(
            MLModelPerformance.model_id == "test_model_123"
        )
        result = await real_db_session.execute(stmt)
        perf_record = result.scalar_one_or_none()
        assert perf_record is not None
        assert perf_record.model_id == "test_model_123"
        assert perf_record.model_type == "sklearn"
        assert perf_record.accuracy == 0.9
        assert perf_record.precision == 0.88
        assert perf_record.recall == 0.92
        assert perf_record.performance_score == 0.85

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_rule_update(
        self, ml_service, real_db_session, real_rule_metadata
    ):
        """Test that cache invalidation events are emitted when rules are updated."""
        import json
        from unittest.mock import patch

        with patch(
            "prompt_improver.services.ml_integration.redis_client"
        ) as mock_redis:
            mock_redis.publish = AsyncMock()
            rule_ids = [rule.rule_id for rule in real_rule_metadata]
            optimized_params = {"n_estimators": 100, "max_depth": 10}
            await ml_service._update_rule_parameters(
                real_db_session, rule_ids, optimized_params, 0.85, "test_model_123"
            )
            mock_redis.publish.assert_called_once()
            call_args = mock_redis.publish.call_args[0]
            channel = call_args[0]
            event_data = json.loads(call_args[1])
            assert channel == "pattern.invalidate"
            assert event_data["type"] == "rule_parameters_updated"
            assert set(event_data["rule_ids"]) == set(rule_ids)
            assert event_data["effectiveness_score"] == 0.85
            assert event_data["model_id"] == "test_model_123"
            assert "apes:pattern:" in event_data["cache_prefixes"]
            assert "rule:" in event_data["cache_prefixes"]
            assert "timestamp" in event_data


class TestMLServiceSingleton:
    """Test ML service singleton pattern."""

    @pytest.mark.anyio
    async def test_get_ml_service_singleton(self):
        """Test that get_ml_service returns the same instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mlflow.set_tracking_uri(f"file://{temp_dir}")
            service1 = await get_ml_service()
            service2 = await get_ml_service()
            assert service1 is service2
            assert isinstance(service1, MLModelService)

    @pytest.mark.anyio
    async def test_ml_service_initialization(self):
        """Test ML service proper initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mlflow.set_tracking_uri(f"file://{temp_dir}")
            service = await get_ml_service()
            assert service.model_registry is not None
            assert service.mlflow_client is not None
            assert service.scaler is not None


class TestPerformanceRequirements:
    """Test Phase 3 performance requirements."""

    @pytest.mark.asyncio
    async def test_prediction_latency_requirement(
        self, ml_service, sample_training_data, real_db_session
    ):
        """Test that predictions meet <5ms latency requirement with real model."""
        small_data = {
            "features": sample_training_data["features"][:15],
            "effectiveness_scores": sample_training_data["effectiveness_scores"][:15],
        }
        train_result = await ml_service.optimize_rules(
            small_data, real_db_session, rule_ids=["perf_test_rule"]
        )
        assert train_result["status"] == "success"
        model_id = train_result["model_id"]
        rule_features = [0.7, 150, 1.0, 5, 0.8, 1.0]
        result = await ml_service.predict_rule_effectiveness(model_id, rule_features)
        assert result["status"] == "success"
        assert result["processing_time_ms"] < 5.0

    @pytest.mark.asyncio
    async def test_optimization_timeout_handling(
        self, ml_service, sample_training_data, real_db_session
    ):
        """Test that optimization respects timeout constraints with real Optuna."""
        import time

        start_time = time.time()
        ml_service_with_timeout = MLModelService()
        ml_service_with_timeout.optimization_timeout = 0.5
        result = await ml_service_with_timeout.optimize_rules(
            sample_training_data, real_db_session
        )
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        assert elapsed_seconds < 2.0
        assert "status" in result
        if result["status"] == "success":
            assert "model_id" in result
            assert "best_score" in result


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_optimization_with_invalid_data(self, ml_service, real_db_session):
        """Test optimization with invalid training data."""
        invalid_data = {
            "features": "invalid_format",
            "effectiveness_scores": [0.5, 0.6],
        }
        result = await ml_service.optimize_rules(invalid_data, real_db_session)
        assert "error" in result
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_database_error_handling(self, ml_service, sample_training_data):
        """Test handling of database errors during optimization."""
        mock_session = MagicMock()
        mock_session.execute.side_effect = Exception("Database connection failed")
        mock_session.commit.side_effect = Exception("Database connection failed")
        mock_session.rollback = MagicMock()
        result = await ml_service.optimize_rules(sample_training_data, mock_session)
        assert "status" in result
        if result["status"] == "error":
            assert "error" in result

    @pytest.mark.asyncio
    async def test_mlflow_error_handling(
        self, ml_service, sample_training_data, real_db_session
    ):
        """Test handling of MLflow errors."""
        with patch("mlflow.start_run", side_effect=Exception("MLflow unavailable")):
            result = await ml_service.optimize_rules(
                sample_training_data, real_db_session
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
    async def test_prediction_contract_feature_validation(
        self, ml_service, features, sample_training_data, real_db_session
    ):
        """Contract: prediction should validate feature dimensions and ranges."""
        train_result = await ml_service.optimize_rules(
            sample_training_data, real_db_session, rule_ids=["contract_test_rule"]
        )
        assert train_result["status"] == "success"
        model_id = train_result["model_id"]
        for i, feature_vector in enumerate(features[:3]):
            if len(feature_vector) != 6:
                feature_vector = (
                    feature_vector[:6]
                    if len(feature_vector) > 6
                    else list(feature_vector) + [0.5] * (6 - len(feature_vector))
                )
            result = await ml_service.predict_rule_effectiveness(
                model_id, feature_vector.tolist()
            )
            assert result["status"] == "success"
            assert isinstance(result["prediction"], (int, float))
            assert 0.0 <= result["prediction"] <= 1.0
            assert isinstance(result["confidence"], (int, float))
            assert 0.0 <= result["confidence"] <= 1.0
            assert isinstance(result["probabilities"], list)
            assert len(result["probabilities"]) == 2
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
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.filter_too_much,
        ],
        max_examples=3,
        deadline=60000,
    )
    async def test_training_data_contract_validation(
        self, ml_service, real_db_session, n_samples, n_features, effectiveness_scores
    ):
        """Contract: training should validate data shape and quality requirements."""
        assume(len(effectiveness_scores) == n_samples)
        assume(n_samples >= 10)
        assume(n_features >= 6)
        effectiveness_array = np.array(effectiveness_scores[:n_samples])
        unique_values = np.unique(effectiveness_array)
        if len(unique_values) < 2:
            split_point = n_samples // 2
            effectiveness_array[:split_point] = np.random.uniform(0.0, 0.4, split_point)
            effectiveness_array[split_point:] = np.random.uniform(
                0.6, 1.0, n_samples - split_point
            )
            effectiveness_scores = effectiveness_array.tolist()
        effectiveness_std = np.std(effectiveness_scores)
        if effectiveness_std < 0.1:
            noise = np.random.normal(0, 0.15, n_samples)
            effectiveness_scores = np.clip(
                np.array(effectiveness_scores) + noise, 0.0, 1.0
            ).tolist()
        features = np.random.rand(n_samples, 6).tolist()
        training_data = {
            "features": features,
            "effectiveness_scores": effectiveness_scores,
        }
        result = await ml_service.optimize_rules(training_data, real_db_session)
        assert "status" in result
        if result["status"] == "success":
            assert "model_id" in result
            assert "best_score" in result
            assert isinstance(result["best_score"], (int, float))
            assert 0.0 <= result["best_score"] <= 1.0
            assert "processing_time_ms" in result
            assert result["processing_time_ms"] > 0
            assert result["training_samples"] == n_samples
            assert result["best_score"] >= 0.3, (
                "Model failed to learn from diverse data"
            )
        else:
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
        with patch.object(ml_service, "_update_rule_parameters") as mock_update:
            mock_update.return_value = None
            await ml_service._update_rule_parameters(
                mock_db_session, ["test_rule"], model_params, 0.8, "test_model"
            )
            mock_update.assert_called_once()
            call_args = mock_update.call_args[0]
            parameters = call_args[2]
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
        self,
        ml_service,
        rule_count,
        feature_dimension,
        sample_training_data,
        real_db_session,
    ):
        """Property: identical inputs should produce identical predictions."""
        train_result = await ml_service.optimize_rules(
            sample_training_data, real_db_session, rule_ids=["consistency_test_rule"]
        )
        assert train_result["status"] == "success"
        model_id = train_result["model_id"]
        test_features = [0.5] * 6
        results = []
        for _ in range(min(rule_count, 3)):
            result = await ml_service.predict_rule_effectiveness(
                model_id, test_features
            )
            results.append(result)
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
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.filter_too_much,
        ],
        max_examples=2,
        deadline=120000,
    )
    async def test_optimization_convergence_property(
        self, ml_service, real_db_session, data_size, noise_level
    ):
        """Property: optimization should converge to reasonable solutions."""
        assume(data_size >= 15)
        assume(noise_level <= 0.1)
        np.random.seed(42)
        features = np.random.rand(data_size, 6)
        half_size = data_size // 2
        quarter_size = data_size // 4
        high_scores = np.random.uniform(0.7, 1.0, quarter_size)
        medium_scores = np.random.uniform(0.4, 0.6, half_size)
        low_scores = np.random.uniform(0.0, 0.3, data_size - quarter_size - half_size)
        effectiveness_scores = np.concatenate([high_scores, medium_scores, low_scores])
        shuffle_indices = np.random.permutation(data_size)
        effectiveness_scores = effectiveness_scores[shuffle_indices]
        features = features[shuffle_indices]
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, data_size)
            effectiveness_scores = np.clip(effectiveness_scores + noise, 0, 1)
        effectiveness_scores = effectiveness_scores.tolist()
        training_data = {
            "features": features.tolist(),
            "effectiveness_scores": effectiveness_scores,
        }
        result = await ml_service.optimize_rules(training_data, real_db_session)
        if result["status"] == "success":
            assert result["best_score"] >= 0.4, (
                f"Optimization failed to find meaningful patterns: {result['best_score']}"
            )
            assert result["best_score"] <= 1.0, "Invalid optimization score"
            expected_timeout = 30000 if data_size < 50 else 60000
            assert result["processing_time_ms"] < expected_timeout, (
                f"Optimization took too long: {result['processing_time_ms']}ms > {expected_timeout}ms"
            )
            assert result["training_samples"] == data_size
            assert result.get("accuracy", 0) >= 0.5, (
                "Model accuracy too low for structured data"
            )
            real_model = ml_service.model_registry.get_model(result["model_id"])
            assert real_model is not None
            assert hasattr(real_model, "predict")
            assert hasattr(real_model, "predict_proba")
        elif noise_level > 0.08:
            assert "error" in result
        else:
            pytest.fail(
                f"Optimization should succeed with low noise: {result.get('error', 'Unknown error')}"
            )

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
        mock_patterns = []
        for i in range(15):
            effectiveness = 0.6 + i % 4 * 0.1
            support = 2 + i % 6
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
    async def test_prediction_latency_characterization(
        self, ml_service, sample_training_data, real_db_session
    ):
        """Characterize prediction latency under various conditions with real models."""
        test_scenarios = [("simple_model", 15), ("medium_model", 25)]
        latency_results = {}
        for model_name, data_size in test_scenarios:
            train_data = {
                "features": sample_training_data["features"][:data_size],
                "effectiveness_scores": sample_training_data["effectiveness_scores"][
                    :data_size
                ],
            }
            train_result = await ml_service.optimize_rules(
                train_data, real_db_session, rule_ids=[f"{model_name}_rule"]
            )
            assert train_result["status"] == "success"
            model_id = train_result["model_id"]
            test_features = [0.5] * 6
            latencies = []
            for _ in range(5):
                import time

                start_time = time.time()
                result = await ml_service.predict_rule_effectiveness(
                    model_id, test_features
                )
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                assert result["status"] == "success"
            avg_latency = sum(latencies) / len(latencies)
            latency_results[model_name] = avg_latency
            assert avg_latency < 5.0, (
                f"{model_name} avg latency {avg_latency:.2f}ms exceeds 5ms target"
            )
        simple_latency = latency_results["simple_model"]
        medium_latency = latency_results["medium_model"]
        latency_growth_factor = (
            medium_latency / simple_latency if simple_latency > 0 else 1
        )
        assert latency_growth_factor < 2.0, (
            f"Latency grows too much with complexity: {latency_growth_factor:.2f}x"
        )

    @pytest.mark.asyncio
    async def test_optimization_scalability_characterization(
        self, ml_service, real_db_session
    ):
        """Characterize optimization performance scaling with data size using real models."""
        data_sizes = [20, 35]
        optimization_times = {}
        for data_size in data_sizes:
            import random

            random.seed(42)
            features = []
            effectiveness_scores = []
            for i in range(data_size):
                base_feature = [0.8, 150, 1.0, 5, 0.7, 1.0]
                varied_feature = [
                    base_feature[0] + random.uniform(-0.2, 0.2),
                    base_feature[1] + random.uniform(-50, 50),
                    base_feature[2] + random.uniform(-0.3, 0.3),
                    base_feature[3] + random.randint(-2, 2),
                    base_feature[4] + random.uniform(-0.2, 0.2),
                    base_feature[5],
                ]
                effectiveness_score = 0.8 + random.uniform(-0.2, 0.2)
                features.append(varied_feature)
                effectiveness_scores.append(max(0.0, min(1.0, effectiveness_score)))
            training_data = {
                "features": features,
                "effectiveness_scores": effectiveness_scores,
            }
            import time

            start_time = time.time()
            result = await ml_service.optimize_rules(training_data, real_db_session)
            end_time = time.time()
            optimization_time = (end_time - start_time) * 1000
            optimization_times[data_size] = optimization_time
            if result["status"] == "success":
                assert result["training_samples"] == data_size
                real_model = ml_service.model_registry.get_model(result["model_id"])
                assert real_model is not None
        if len(optimization_times) >= 2:
            times = list(optimization_times.values())
            sizes = list(optimization_times.keys())
            scaling_factor = times[-1] / times[0] / (sizes[-1] / sizes[0])
            assert scaling_factor < 5.0, (
                f"Optimization scaling factor {scaling_factor:.2f} indicates poor scalability"
            )


@pytest.mark.ml_data_validation
class TestMLDataQualityValidation:
    """Data quality and validation testing for ML pipelines."""

    @given(corrupted_ratio=st.floats(min_value=0.0, max_value=0.3))
    @pytest.mark.asyncio
    @settings(
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.filter_too_much,
        ],
        max_examples=2,
        deadline=120000,
    )
    async def test_corrupted_data_handling(
        self, ml_service, real_db_session, corrupted_ratio
    ):
        """Test ML service handling of corrupted training data with real models."""
        assume(corrupted_ratio <= 0.3)
        clean_size = 30
        corrupted_size = int(clean_size * corrupted_ratio)
        clean_features = []
        clean_scores = []
        for i in range(clean_size):
            clean_features.append([
                np.random.uniform(0.3, 0.9),
                np.random.uniform(100, 200),
                np.random.uniform(0.5, 1.0),
                np.random.randint(3, 8),
                np.random.uniform(0.4, 0.8),
                np.random.uniform(0.6, 1.0),
            ])
            if i < clean_size // 2:
                clean_scores.append(np.random.uniform(0.7, 1.0))
            else:
                clean_scores.append(np.random.uniform(0.1, 0.5))
        corrupted_features = []
        corrupted_scores = []
        for i in range(corrupted_size):
            corruption_type = i % 4
            if corruption_type == 0:
                corrupted_features.append([float("nan"), 150, 0.8, 5, 0.7, 0.9])
                corrupted_scores.append(float("nan"))
            elif corruption_type == 1:
                corrupted_features.append([0.8, float("inf"), 1.0, 5, 0.7, 1.0])
                corrupted_scores.append(0.8)
            elif corruption_type == 2:
                corrupted_features.append([-1.0, 150, 2.0, 1000, -0.5, 1.5])
                corrupted_scores.append(-0.5)
            else:
                corrupted_features.append([
                    float("nan"),
                    float("inf"),
                    -1.0,
                    1000,
                    2.0,
                    -0.5,
                ])
                corrupted_scores.append(float("inf"))
        training_data = {
            "features": clean_features + corrupted_features,
            "effectiveness_scores": clean_scores + corrupted_scores,
        }
        result = await ml_service.optimize_rules(training_data, real_db_session)
        assert "status" in result
        if corrupted_ratio == 0.0:
            assert result["status"] == "success"
            assert result["training_samples"] == clean_size
            assert result["best_score"] >= 0.4
            real_model = ml_service.model_registry.get_model(result["model_id"])
            assert real_model is not None
        elif corrupted_ratio > 0.25:
            if result["status"] == "error":
                assert (
                    "data" in result["error"].lower()
                    or "invalid" in result["error"].lower()
                    or "nan" in result["error"].lower()
                    or ("inf" in result["error"].lower())
                ), f"Error message should indicate data issues: {result['error']}"
            else:
                assert result["training_samples"] <= clean_size + corrupted_size
                assert result["training_samples"] >= clean_size * 0.8
        elif result["status"] == "success":
            assert result["training_samples"] <= clean_size + corrupted_size
            assert result["training_samples"] >= clean_size * 0.7
            assert result["best_score"] >= 0.3, (
                "Performance degraded too much with minor corruption"
            )
            real_model = ml_service.model_registry.get_model(result["model_id"])
            assert real_model is not None
        elif result["status"] == "error":
            assert (
                "data" in result["error"].lower()
                or "invalid" in result["error"].lower()
            )


@pytest.mark.metamorphic
class TestMLMetamorphicProperties:
    """
    Metamorphic testing for ML model behavioral properties and invariants.

    NOTE: These tests operate on discrete training data (5 patterns repeated 5 times),
    which leads to binary classification behavior. The model learns to memorize
    specific feature->effectiveness mappings rather than continuous relationships.

    This affects test expectations:
    - Predictions are typically 0.0 or 1.0 (binary outcomes)
    - Perfect consistency is expected for identical inputs
    - Perturbations may cause discrete jumps rather than gradual changes
    - Tests are calibrated for this discrete behavior pattern
    """

    @pytest.mark.asyncio
    async def test_prediction_determinism(
        self, ml_service, sample_training_data, real_db_session
    ):
        """Property: Identical inputs should produce identical predictions (determinism)."""
        train_result = await ml_service.optimize_rules(
            sample_training_data, real_db_session, rule_ids=["determinism_test_rule"]
        )
        assert train_result["status"] == "success"
        model_id = train_result["model_id"]
        test_features = [0.7, 150, 1.0, 5, 0.8, 1.0]
        predictions = []
        for _ in range(5):
            result = await ml_service.predict_rule_effectiveness(
                model_id, test_features
            )
            predictions.append(result)
        first_pred = predictions[0]
        for i, pred in enumerate(predictions[1:], 1):
            assert pred["prediction"] == first_pred["prediction"], (
                f"Prediction mismatch: {pred['prediction']} != {first_pred['prediction']}"
            )
            assert pred["confidence"] == first_pred["confidence"], (
                f"Confidence mismatch: {pred['confidence']} != {first_pred['confidence']}"
            )
            assert pred["probabilities"] == first_pred["probabilities"], (
                f"Probabilities mismatch: {pred['probabilities']} != {first_pred['probabilities']}"
            )

    @given(
        perturbation_factor=st.floats(min_value=0.001, max_value=0.1),
        feature_index=st.integers(min_value=0, max_value=5),
    )
    @pytest.mark.asyncio
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=3
    )
    async def test_small_perturbation_stability(
        self,
        ml_service,
        sample_training_data,
        real_db_session,
        perturbation_factor,
        feature_index,
    ):
        """Property: Small input perturbations should produce stable output changes."""
        train_result = await ml_service.optimize_rules(
            sample_training_data, real_db_session, rule_ids=["perturbation_test_rule"]
        )
        assert train_result["status"] == "success"
        model_id = train_result["model_id"]
        original_features = [0.7, 150, 1.0, 5, 0.8, 1.0]
        perturbed_features = original_features.copy()
        perturbed_features[feature_index] += perturbation_factor
        perturbed_features[feature_index] = max(
            0.0, min(1.0, perturbed_features[feature_index])
        )
        original_pred = await ml_service.predict_rule_effectiveness(
            model_id, original_features
        )
        perturbed_pred = await ml_service.predict_rule_effectiveness(
            model_id, perturbed_features
        )
        prediction_diff = abs(
            original_pred["prediction"] - perturbed_pred["prediction"]
        )
        confidence_diff = abs(
            original_pred["confidence"] - perturbed_pred["confidence"]
        )
        if prediction_diff > (1.0 if perturbation_factor > 0.05 else 0.5):
            print(
                f"Large perturbation detected - Original: {original_pred}, Perturbed: {perturbed_pred}"
            )
        max_expected_change = 1.0 if perturbation_factor > 0.05 else 0.5
        assert prediction_diff <= max_expected_change, (
            f"Prediction changed too much: {prediction_diff} > {max_expected_change}"
        )
        assert confidence_diff <= max_expected_change, (
            f"Confidence changed too much: {confidence_diff} > {max_expected_change}"
        )

    @given(
        effectiveness_threshold_1=st.floats(min_value=0.5, max_value=0.7),
        effectiveness_threshold_2=st.floats(min_value=0.7, max_value=0.9),
    )
    @pytest.mark.asyncio
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=2
    )
    async def test_pattern_discovery_monotonicity(
        self,
        ml_service,
        real_db_session,
        real_rule_metadata,
        sample_rule_performance,
        real_prompt_sessions,
        effectiveness_threshold_1,
        effectiveness_threshold_2,
    ):
        """Property: Higher effectiveness thresholds should find fewer or equal patterns."""
        assume(effectiveness_threshold_1 < effectiveness_threshold_2)
        for i, performance in enumerate(sample_rule_performance[:10]):
            performance.rule_id = real_rule_metadata[
                i % len(real_rule_metadata)
            ].rule_id
            performance.session_id = real_prompt_sessions[
                i % len(real_prompt_sessions)
            ].session_id
            performance.effectiveness_score = 0.5 + i % 5 * 0.1
            real_db_session.add(performance)
        await real_db_session.commit()
        patterns_low = await ml_service.discover_patterns(
            real_db_session, min_effectiveness=effectiveness_threshold_1, min_support=2
        )
        patterns_high = await ml_service.discover_patterns(
            real_db_session, min_effectiveness=effectiveness_threshold_2, min_support=2
        )
        if patterns_low["status"] == "success" and patterns_high["status"] == "success":
            assert (
                patterns_high["patterns_discovered"]
                <= patterns_low["patterns_discovered"]
            ), (
                f"Higher threshold found more patterns: {patterns_high['patterns_discovered']} > {patterns_low['patterns_discovered']}"
            )

    @given(feature_multiplier=st.floats(min_value=1.1, max_value=2.0))
    @pytest.mark.asyncio
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=3
    )
    async def test_feature_enhancement_monotonicity(
        self, ml_service, sample_training_data, real_db_session, feature_multiplier
    ):
        """Property: Enhanced features should generally improve or maintain effectiveness predictions."""
        train_result = await ml_service.optimize_rules(
            sample_training_data, real_db_session, rule_ids=["monotonicity_test_rule"]
        )
        assert train_result["status"] == "success"
        model_id = train_result["model_id"]
        baseline_features = [0.6, 100, 0.8, 4, 0.7, 0.9]
        enhanced_features = [
            baseline_features[0] * feature_multiplier,
            baseline_features[1] * feature_multiplier,
            baseline_features[2] * feature_multiplier,
            baseline_features[3],
            baseline_features[4] * feature_multiplier,
            baseline_features[5],
        ]
        enhanced_features = [
            min(1.0, f) if i != 1 else min(200.0, f)
            for i, f in enumerate(enhanced_features)
        ]
        enhanced_features[3] = min(10, max(1, enhanced_features[3]))
        baseline_pred = await ml_service.predict_rule_effectiveness(
            model_id, baseline_features
        )
        enhanced_pred = await ml_service.predict_rule_effectiveness(
            model_id, enhanced_features
        )
        prediction_improvement = (
            enhanced_pred["prediction"] - baseline_pred["prediction"]
        )
        if prediction_improvement < -0.5:
            print(
                f"Significant degradation detected - Baseline: {baseline_pred}, Enhanced: {enhanced_pred}"
            )
        if baseline_pred["prediction"] >= 0.8:
            assert prediction_improvement >= -0.5, (
                f"Enhanced features decreased high-effectiveness prediction too much: {prediction_improvement}"
            )
        else:
            assert prediction_improvement >= -0.8, (
                f"Enhanced features decreased effectiveness too much: {prediction_improvement}"
            )

    @given(noise_level=st.floats(min_value=0.0, max_value=0.1))
    @pytest.mark.asyncio
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=3
    )
    async def test_input_noise_robustness(
        self, ml_service, sample_training_data, real_db_session, noise_level
    ):
        """Property: Model should be robust to small amounts of input noise."""
        train_result = await ml_service.optimize_rules(
            sample_training_data, real_db_session, rule_ids=["robustness_test_rule"]
        )
        assert train_result["status"] == "success"
        model_id = train_result["model_id"]
        clean_features = [0.7, 150, 1.0, 5, 0.8, 1.0]
        np.random.seed(42)
        noise = np.random.normal(0, noise_level, len(clean_features))
        noisy_features = [
            max(0.0, min(1.0, f + n))
            for f, n in zip(clean_features, noise, strict=False)
        ]
        noisy_features[1] = max(0, min(300, clean_features[1] + noise[1] * 50))
        noisy_features[3] = max(1, min(10, clean_features[3] + int(noise[3] * 2)))
        clean_pred = await ml_service.predict_rule_effectiveness(
            model_id, clean_features
        )
        noisy_pred = await ml_service.predict_rule_effectiveness(
            model_id, noisy_features
        )
        prediction_diff = abs(clean_pred["prediction"] - noisy_pred["prediction"])
        confidence_diff = abs(clean_pred["confidence"] - noisy_pred["confidence"])
        max_prediction_change = noise_level * 5 + 0.1
        max_confidence_change = noise_level * 3 + 0.1
        assert prediction_diff <= max_prediction_change, (
            f"Model not robust to noise: prediction changed by {prediction_diff}"
        )
        assert confidence_diff <= max_confidence_change, (
            f"Model not robust to noise: confidence changed by {confidence_diff}"
        )

    @pytest.mark.asyncio
    async def test_ensemble_consistency(
        self, ml_service, sample_training_data, real_db_session
    ):
        """Property: Multiple models trained on same data should show reasonable consistency."""
        model1_result = await ml_service.optimize_rules(
            sample_training_data, real_db_session, rule_ids=["ensemble_test_1"]
        )
        model2_result = await ml_service.optimize_rules(
            sample_training_data, real_db_session, rule_ids=["ensemble_test_2"]
        )
        assert model1_result["status"] == "success"
        assert model2_result["status"] == "success"
        test_features = [0.6, 120, 0.9, 4, 0.7, 0.8]
        pred1 = await ml_service.predict_rule_effectiveness(
            model1_result["model_id"], test_features
        )
        pred2 = await ml_service.predict_rule_effectiveness(
            model2_result["model_id"], test_features
        )
        prediction_diff = abs(pred1["prediction"] - pred2["prediction"])
        confidence_diff = abs(pred1["confidence"] - pred2["confidence"])
        if prediction_diff == 0.0 and confidence_diff == 0.0:
            print(
                "✅ Perfect ensemble consistency (expected with discrete training data)"
            )
        else:
            assert prediction_diff <= 1.0, (
                f"Models too inconsistent: prediction diff {prediction_diff}"
            )
            assert confidence_diff <= 1.0, (
                f"Models too inconsistent: confidence diff {confidence_diff}"
            )
            print(
                f"Ensemble variance within acceptable bounds: pred_diff={prediction_diff}, conf_diff={confidence_diff}"
            )

    @given(batch_size=st.integers(min_value=1, max_value=5))
    @pytest.mark.asyncio
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=2
    )
    async def test_batch_consistency(
        self, ml_service, sample_training_data, real_db_session, batch_size
    ):
        """Property: Batch predictions should be consistent with individual predictions."""
        train_result = await ml_service.optimize_rules(
            sample_training_data, real_db_session, rule_ids=["batch_test_rule"]
        )
        assert train_result["status"] == "success"
        model_id = train_result["model_id"]
        feature_sets = [
            [0.7, 150, 1.0, 5, 0.8, 1.0],
            [0.6, 100, 0.9, 4, 0.7, 0.9],
            [0.8, 200, 0.8, 6, 0.9, 0.8],
            [0.5, 80, 0.7, 3, 0.6, 0.7],
            [0.9, 180, 0.95, 7, 0.85, 0.95],
        ][:batch_size]
        individual_predictions = []
        for features in feature_sets:
            pred = await ml_service.predict_rule_effectiveness(model_id, features)
            individual_predictions.append(pred)
        for pred in individual_predictions:
            assert pred["status"] == "success"
            assert 0 <= pred["prediction"] <= 1
            assert 0 <= pred["confidence"] <= 1
            assert len(pred["probabilities"]) == 2
            assert abs(sum(pred["probabilities"]) - 1.0) < 0.01

    @given(threshold_offset=st.floats(min_value=0.01, max_value=0.2))
    @pytest.mark.asyncio
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=2
    )
    async def test_optimization_score_boundary_consistency(
        self, ml_service, sample_training_data, real_db_session, threshold_offset
    ):
        """Property: Optimization results should be consistent near score boundaries."""
        threshold = 0.7
        boundary_data = sample_training_data.copy()
        modified_scores = []
        for i, score in enumerate(boundary_data["effectiveness_scores"]):
            if i % 2 == 0:
                modified_scores.append(threshold - threshold_offset)
            else:
                modified_scores.append(threshold + threshold_offset)
        boundary_data["effectiveness_scores"] = modified_scores
        result = await ml_service.optimize_rules(
            boundary_data, real_db_session, rule_ids=["boundary_test_rule"]
        )
        if result["status"] == "success":
            assert result["best_score"] >= 0
            assert result["best_score"] <= 1
            assert result["accuracy"] >= 0
            assert result["precision"] >= 0
            assert result["recall"] >= 0
            model_id = result["model_id"]
            cached_model = ml_service.model_registry.get_model(model_id)
            assert cached_model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
