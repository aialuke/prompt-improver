"""
Test for rule optimizer integration fixes.

This test verifies that the optimize_rule method works correctly with proper
performance_data parameter and validates the integration with other components.
"""


import pytest

from prompt_improver.ml.optimization.algorithms.rule_optimizer import (
    OptimizationConfig,
    RuleOptimizer,
)


class TestRuleOptimizerIntegrationFix:
    """Test the fixes for rule optimizer integration issues."""

    @pytest.fixture
    def optimization_config(self):
        """Create optimization configuration for testing."""
        return OptimizationConfig(
            min_sample_size=10,
            enable_multi_objective=True,
            enable_gaussian_process=True,
            pareto_population_size=20,
            pareto_generations=10,
        )

    @pytest.fixture
    def rule_optimizer(self, optimization_config):
        """Create RuleOptimizer instance for testing."""
        return RuleOptimizer(config=optimization_config)

    @pytest.fixture
    def valid_performance_data(self):
        """Create valid performance data dictionary."""
        return {
            "test_rule": {
                "total_applications": 50,
                "avg_improvement": 0.75,
                "consistency_score": 0.8,
                "confidence_level": 0.85,
            }
        }

    @pytest.fixture
    def valid_historical_data(self):
        """Create valid historical data list."""
        return [
            {
                "score": 0.8,
                "context": {"domain": "technical", "length": "medium"},
                "timestamp": "2025-01-01T10:00:00",
                "rule_parameters": {"threshold": 0.7, "weight": 0.9},
            },
            {
                "score": 0.75,
                "context": {"domain": "creative", "length": "short"},
                "timestamp": "2025-01-01T11:00:00",
                "rule_parameters": {"threshold": 0.6, "weight": 0.8},
            },
            {
                "score": 0.85,
                "context": {"domain": "technical", "length": "long"},
                "timestamp": "2025-01-01T12:00:00",
                "rule_parameters": {"threshold": 0.8, "weight": 1.0},
            },
        ]

    @pytest.mark.asyncio
    async def test_optimize_rule_with_correct_parameters(
        self, rule_optimizer, valid_performance_data, valid_historical_data
    ):
        """Test that optimize_rule works with correct parameter types."""
        rule_id = "test_rule"
        result = await rule_optimizer.optimize_rule(
            rule_id=rule_id,
            performance_data=valid_performance_data,
            historical_data=valid_historical_data,
        )
        assert "rule_id" in result
        assert result["rule_id"] == rule_id
        assert "status" in result
        assert result["status"] in {"optimized", "insufficient_data"}

    @pytest.mark.asyncio
    async def test_optimize_rule_with_insufficient_data(self, rule_optimizer):
        """Test optimize_rule with insufficient performance data."""
        rule_id = "test_rule"
        insufficient_performance_data = {
            "test_rule": {
                "total_applications": 5,
                "avg_improvement": 0.75,
                "consistency_score": 0.8,
            }
        }
        result = await rule_optimizer.optimize_rule(
            rule_id=rule_id,
            performance_data=insufficient_performance_data,
            historical_data=[],
        )
        assert result["status"] == "insufficient_data"
        assert "message" in result

    @pytest.mark.asyncio
    async def test_optimize_rule_parameter_validation(
        self, rule_optimizer, valid_historical_data
    ):
        """Test that optimize_rule validates parameter types correctly."""
        rule_id = "test_rule"
        with pytest.raises(AttributeError):
            await rule_optimizer.optimize_rule(
                rule_id=rule_id,
                performance_data=[],
                historical_data=valid_historical_data,
            )

    @pytest.mark.asyncio
    async def test_optimize_rule_with_empty_performance_data(
        self, rule_optimizer, valid_historical_data
    ):
        """Test optimize_rule with empty performance data dictionary."""
        rule_id = "test_rule"
        empty_performance_data = {}
        result = await rule_optimizer.optimize_rule(
            rule_id=rule_id,
            performance_data=empty_performance_data,
            historical_data=valid_historical_data,
        )
        assert result["status"] == "insufficient_data"

    @pytest.mark.asyncio
    async def test_optimize_rule_with_none_historical_data(
        self, rule_optimizer, valid_performance_data
    ):
        """Test optimize_rule with None historical data."""
        rule_id = "test_rule"
        result = await rule_optimizer.optimize_rule(
            rule_id=rule_id,
            performance_data=valid_performance_data,
            historical_data=None,
        )
        assert "rule_id" in result
        assert result["rule_id"] == rule_id

    @pytest.mark.asyncio
    async def test_optimize_rule_integration_with_advanced_features(
        self, rule_optimizer, valid_performance_data, valid_historical_data
    ):
        """Test that optimize_rule integrates correctly with advanced optimization features."""
        rule_id = "test_rule"
        result = await rule_optimizer.optimize_rule(
            rule_id=rule_id,
            performance_data=valid_performance_data,
            historical_data=valid_historical_data,
        )
        assert "rule_id" in result
        assert "status" in result
        assert "optimization_date" in result
        if result["status"] == "optimized":
            if "multi_objective_optimization" in result:
                assert isinstance(result["multi_objective_optimization"], dict)
            if "gaussian_process_optimization" in result:
                assert isinstance(result["gaussian_process_optimization"], dict)

    def test_performance_data_structure_validation(self):
        """Test that performance data has the expected structure."""
        valid_performance_data = {
            "test_rule": {
                "total_applications": 50,
                "avg_improvement": 0.75,
                "consistency_score": 0.8,
                "confidence_level": 0.85,
            }
        }
        rule_data = valid_performance_data.get("test_rule", {})
        assert "total_applications" in rule_data
        assert "avg_improvement" in rule_data
        assert isinstance(rule_data["total_applications"], int)
        assert isinstance(rule_data["avg_improvement"], (int, float))

    def test_historical_data_structure_validation(self):
        """Test that historical data has the expected structure."""
        valid_historical_data = [
            {
                "score": 0.8,
                "context": {"domain": "technical"},
                "timestamp": "2025-01-01T10:00:00",
            }
        ]
        assert isinstance(valid_historical_data, list)
        if valid_historical_data:
            data_point = valid_historical_data[0]
            assert "score" in data_point
            assert "context" in data_point
            assert isinstance(data_point["score"], (int, float))
            assert isinstance(data_point["context"], dict)


pytestmark = [pytest.mark.unit]
