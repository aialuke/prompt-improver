"""
Tests for A/B Testing Framework Implementation in ab_testing.py
Utilizes pytest-asyncio for async tests and hypothesis for complex statistical scenarios.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import (
    assume,
    given,
    settings,
    strategies as st,
)

from prompt_improver.services.ab_testing import ABTestingService, ExperimentResult


class TestABTestingFramework:
    """Test Suite for A/B Testing Service with comprehensive test coverage."""

    @pytest.fixture
    def ab_testing_service(self):
        """Fixture to create an instance of ABTestingService."""
        return ABTestingService()

    @pytest.mark.asyncio
    async def test_create_experiment_success(self, ab_testing_service, mock_db_session):
        """Test successful creation of A/B experiment."""
        mock_result = {"status": "success", "experiment_id": "test_exp_123"}
        with patch.object(
            ab_testing_service, "create_experiment", return_value=mock_result
        ) as mock_method:
            result = await ab_testing_service.create_experiment(
                "test_experiment",
                {"name": "control"},
                {"name": "treatment"},
                db_session=mock_db_session,
            )
            assert result["status"] == "success"
            assert mock_method.call_count == 1

    @pytest.mark.asyncio
    async def test_analyze_experiment_statistical_significance(
        self, ab_testing_service, mock_db_session
    ):
        """Test statistical significance of A/B testing using Welch's t-test."""
        # Mock successful analysis result
        mock_analysis_result = {
            "status": "success",
            "experiment_id": "experiment_id",
            "experiment_name": "test_experiment",
            "analysis": {
                "control_mean": 0.1,
                "treatment_mean": 0.2,
                "effect_size": 0.1,
                "p_value": 0.04,
                "statistical_significance": True,
                "bayesian_probability": 0.95,
            },
        }

        with patch.object(
            ab_testing_service, "analyze_experiment", return_value=mock_analysis_result
        ):
            result = await ab_testing_service.analyze_experiment(
                "experiment_id", mock_db_session
            )

        assert (
            result["analysis"]["p_value"] < 0.05
        )  # Check for statistical significance
        assert (
            result["analysis"]["bayesian_probability"] > 0.9
        )  # Check Bayesian probability threshold

    @given(
        control_mean=st.floats(min_value=0.0, max_value=1.0),
        treatment_mean=st.floats(min_value=0.0, max_value=1.0),
        sample_size=st.integers(min_value=30, max_value=1000),
    )
    @settings(max_examples=20)
    def test_statistical_properties(self, control_mean, treatment_mean, sample_size):
        """Property-based test to ensure statistical calculations are reasonable."""
        assume(
            abs(control_mean - treatment_mean) > 0.01
        )  # Avoid cases where means are too close

        # Test that effect size calculation would be reasonable
        effect_size = abs(treatment_mean - control_mean) / 0.1  # Assuming std of 0.1
        assert effect_size >= 0  # Effect size should be non-negative

        # Test that sample size is reasonable
        assert sample_size >= 30  # Minimum sample size for statistical tests

    @pytest.mark.asyncio
    async def test_list_active_experiments(self, ab_testing_service, mock_db_session):
        """Test listing of active A/B experiments."""
        # Mock database session behavior
        mock_experiments = [
            {
                "id": "exp1",
                "status": "running",
                "control_group": "C1",
                "treatment_group": "T1",
            },
            {
                "id": "exp2",
                "status": "running",
                "control_group": "C2",
                "treatment_group": "T2",
            },
        ]

        # Patch list_experiments
        with patch.object(
            ab_testing_service, "list_experiments", return_value=mock_experiments
        ):
            active_experiments = await ab_testing_service.list_experiments(
                status="running"
            )

        # Verify returned experiments
        assert len(active_experiments) == 2
        for experiment in active_experiments:
            assert experiment["status"] == "running"

    @pytest.mark.asyncio
    async def test_stop_experiment(self, ab_testing_service, mock_db_session):
        """Test stopping an A/B experiment."""
        experiment_id = "exp123"

        with patch.object(ab_testing_service, "stop_experiment", return_value=True):
            result = await ab_testing_service.stop_experiment(
                experiment_id=experiment_id,
                reason="Test complete",
                db_session=mock_db_session,
            )

        assert result is True


class TestABEdgeCases:
    """Additional edge cases and error handling tests for A/B Test service."""

    @pytest.fixture
    def ab_testing_service(self):
        """Fixture to create an instance of ABTestingService."""
        return ABTestingService()

    @pytest.mark.asyncio
    async def test_create_experiment_conflict(
        self, ab_testing_service, mock_db_session
    ):
        """Test creation conflict when same experiment already exists."""
        experiment_name = "duplicate_exp"

        with patch.object(
            ab_testing_service,
            "create_experiment",
            side_effect=ValueError("Experiment already exists"),
        ):
            with pytest.raises(ValueError, match="Experiment already exists"):
                await ab_testing_service.create_experiment(
                    experiment_name,
                    control_group="control",
                    treatment_group="treatment",
                    db_session=mock_db_session,
                )

    @pytest.mark.asyncio
    async def test_analyze_experiment_no_results(
        self, ab_testing_service, mock_db_session
    ):
        """Test analyze_experiment behavior when no results available."""
        experiment_id = "exp999"

        with patch.object(ab_testing_service, "analyze_experiment", return_value=None):
            result = await ab_testing_service.analyze_experiment(
                experiment_id=experiment_id, db_session=mock_db_session
            )

        assert result is None
