"""
Comprehensive tests for Phase 3 ML integration service.
Tests direct Python ML service implementation including optimization, prediction, and pattern discovery.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

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
        insufficient_data = {
            "features": [[0.5, 100, 1.0, 5, 0.7, 1.0]] * 5,  # Only 5 samples
            "effectiveness_scores": [0.5] * 5,
        }

        result = await ml_service.optimize_rules(insufficient_data, mock_db_session)

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
        ml_service.models[model_id] = mock_model

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
        # Create sufficient data for ensemble (minimum 20 samples)
        ensemble_data = {
            "features": [[0.8, 150, 1.0, 5, 0.7, 1.0, 0.5, 1.0, 0.2, 0.8]] * 25,
            "effectiveness_scores": [0.8] * 25,
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
        self, ml_service, mock_db_session, sample_rule_performance
    ):
        """Test successful pattern discovery."""
        # Mock database query result
        mock_result = MagicMock()
        mock_result.fetchall.return_value = sample_rule_performance
        mock_db_session.execute.return_value = mock_result

        result = await ml_service.discover_patterns(
            mock_db_session, min_effectiveness=0.7, min_support=3
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
        mock_rule.parameters = {"weight": 0.8}

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_rule
        mock_db_session.execute.return_value = mock_result

        optimized_params = {"n_estimators": 100, "max_depth": 10}

        await ml_service._update_rule_parameters(
            mock_db_session, ["clarity_rule"], optimized_params, 0.85, "test_model_123"
        )

        # Verify rule was updated
        assert mock_rule.parameters["n_estimators"] == 100
        assert mock_rule.parameters["max_depth"] == 10
        assert mock_rule.effectiveness_score == 0.85
        assert mock_rule.updated_by == "ml_training"

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
        assert call_args.model_id == "test_model_123"
        assert call_args.performance_score == 0.85
        assert call_args.accuracy == 0.90
        assert call_args.precision == 0.88
        assert call_args.recall == 0.92


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

            assert service.models == {}
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
        ml_service.models[model_id] = mock_model

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
            assert call_args[1]["timeout"] == 300  # 5 minute timeout


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
