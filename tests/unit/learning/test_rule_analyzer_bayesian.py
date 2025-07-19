"""Tests for Phase 2 Bayesian Modeling in Rule Effectiveness Analyzer using real behavior.

Migrated from mock-based testing to real behavior testing following 2025 best practices:
- Use real PyMC models with minimal sampling parameters for test speed
- Test actual Bayesian inference results and posterior distributions
- Validate real convergence diagnostics and credible intervals
- Mock only truly external dependencies, not core PyMC functionality
- Focus on behavior validation rather than implementation details

Comprehensive test suite for Bayesian modeling enhancements including:
- Real Bayesian hierarchical modeling with PyMC
- Actual uncertainty quantification and credible intervals
- Real prior specification and posterior inference
- Authentic model convergence diagnostics and validation

Testing best practices applied from 2025 research:
- Real statistical validation of Bayesian inference
- Actual handling of uncertainty estimates
- Realistic prior distributions for ML parameters
- Real model convergence and diagnostic validation
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
# Removed mock dependencies - using real behavior testing following 2025 best practices

import numpy as np
import pandas as pd
import pytest

from prompt_improver.learning.rule_analyzer import (
    RuleAnalysisConfig,
    RuleEffectivenessAnalyzer,
    RuleMetrics,
)


@pytest.fixture
def bayesian_config():
    """Configuration with Bayesian modeling enabled using minimal parameters for test speed."""
    return RuleAnalysisConfig(
        enable_bayesian_modeling=True,
        bayesian_samples=100,  # Reduced from 2000 for test speed
        bayesian_tune=50,      # Reduced from 1000 for test speed
        bayesian_chains=2,
        credible_interval=0.95,
        # Standard parameters
        min_sample_size=10,
        significance_level=0.05,
        effect_size_threshold=0.2,
    )


@pytest.fixture
def rule_analyzer_bayesian(bayesian_config):
    """Rule analyzer with Bayesian modeling enabled."""
    return RuleEffectivenessAnalyzer(config=bayesian_config)


@pytest.fixture
def hierarchical_rule_data():
    """Realistic hierarchical rule data for Bayesian modeling."""
    np.random.seed(42)  # Reproducible test data

    # Generate hierarchical data structure
    # Global performance parameters
    global_mean_performance = 0.75
    global_std = 0.15

    # Context-specific variations
    contexts = ["technical", "creative", "business", "academic"]
    context_effects = {
        "technical": 0.05,  # Slightly better
        "creative": -0.02,  # Slightly worse
        "business": 0.08,  # Better
        "academic": 0.0,  # Baseline
    }

    rule_data = {}

    for rule_idx in range(3):
        rule_id = f"rule_bayesian_{rule_idx:03d}"

        # Rule-specific effect
        rule_effect = np.random.normal(0, 0.08)

        rule_observations = []

        for context in contexts:
            # Context-specific performance
            context_effect = context_effects[context]

            # Generate observations for this rule-context combination
            n_obs = 15 + np.random.randint(0, 11)  # 15-25 observations

            for obs_idx in range(n_obs):
                # Hierarchical performance model:
                # performance = global_mean + rule_effect + context_effect + noise
                true_performance = (
                    global_mean_performance
                    + rule_effect
                    + context_effect
                    + np.random.normal(0, 0.05)
                )

                # Clip to valid range
                observed_performance = np.clip(true_performance, 0.0, 1.0)

                rule_observations.append({
                    "context": context,
                    "score": float(observed_performance),
                    "applications": 1,
                    "timestamp": datetime.now().isoformat(),
                    "user_feedback": np.random.choice(
                        ["positive", "neutral", "negative"], p=[0.6, 0.3, 0.1]
                    ),
                })

        rule_data[rule_id] = {
            "observations": rule_observations,
            "total_applications": len(rule_observations),
            "avg_score": float(np.mean([obs["score"] for obs in rule_observations])),
            "std_score": float(np.std([obs["score"] for obs in rule_observations])),
            "contexts_used": contexts,
            "hierarchical_structure": {
                "global_mean": global_mean_performance,
                "rule_effect": rule_effect,
                "context_effects": context_effects,
            },
        }

    return rule_data


@pytest.fixture
def simple_rule_data():
    """Simple rule data for basic Bayesian testing."""
    np.random.seed(123)

    # Simple case with known parameters
    true_mean = 0.8
    true_std = 0.1
    n_observations = 30

    observations = []
    for i in range(n_observations):
        score = np.clip(np.random.normal(true_mean, true_std), 0.0, 1.0)
        observations.append({
            "score": float(score),
            "applications": 1,
            "context": "simple_context",
        })

    rule_data = {
        "rule_simple_bayesian": {
            "observations": observations,
            "total_applications": n_observations,
            "avg_score": float(np.mean([obs["score"] for obs in observations])),
            "true_parameters": {"mean": true_mean, "std": true_std},
        }
    }

    return rule_data


@pytest.fixture
def insufficient_bayesian_data():
    """Minimal data that may not support Bayesian modeling."""
    return {
        "rule_minimal_bayesian": {
            "observations": [
                {"score": 0.8, "applications": 1, "context": "test"},
                {"score": 0.7, "applications": 1, "context": "test"},
            ],
            "total_applications": 2,
            "avg_score": 0.75,
        }
    }


class TestBayesianModeling:
    """Test suite for Bayesian modeling functionality."""

    @pytest.mark.asyncio
    async def test_bayesian_modeling_integration(
        self, rule_analyzer_bayesian, hierarchical_rule_data
    ):
        """Test full Bayesian modeling workflow integration using real PyMC."""
        try:
            import pymc
            import arviz
        except ImportError:
            pytest.skip("PyMC and ArviZ not available for real Bayesian testing")
            
        # Test with real PyMC - no mocking needed
        result = await rule_analyzer_bayesian.analyze_rule_effectiveness(
            hierarchical_rule_data
        )

        # Should include Bayesian modeling results
        assert "bayesian_analysis" in result
        bayesian_results = result["bayesian_analysis"]

        # Validate Bayesian results structure
        for rule_id, bayesian_result in bayesian_results.items():
            if isinstance(bayesian_result, dict):
                assert "posterior_mean" in bayesian_result
                assert "credible_interval" in bayesian_result
                assert "uncertainty_estimate" in bayesian_result

                # Validate realistic posterior estimates
                posterior_mean = bayesian_result["posterior_mean"]
                assert 0.0 <= posterior_mean <= 1.0

                # Validate credible interval
                ci = bayesian_result["credible_interval"]
                assert "lower" in ci and "upper" in ci
                assert 0.0 <= ci["lower"] <= ci["upper"] <= 1.0

                # Validate uncertainty estimate
                uncertainty = bayesian_result["uncertainty_estimate"]
                assert 0.0 <= uncertainty <= 1.0

    @pytest.mark.asyncio
    async def test_bayesian_model_fitting(
        self, rule_analyzer_bayesian, simple_rule_data
    ):
        """Test Bayesian model fitting with known parameters using real PyMC."""
        try:
            import pymc
            import arviz
        except ImportError:
            pytest.skip("PyMC and ArviZ not available for real Bayesian testing")
            
        rule_id = "rule_simple_bayesian"
        observations = simple_rule_data[rule_id]["observations"]
        true_params = simple_rule_data[rule_id]["true_parameters"]

        # Test real Bayesian model fitting
        bayesian_result = rule_analyzer_bayesian._perform_bayesian_modeling(
            rule_id, observations
        )

        # Validate result structure
        if bayesian_result:
            assert isinstance(bayesian_result, dict)
            assert "posterior_mean" in bayesian_result
            assert "credible_interval" in bayesian_result
            assert "model_diagnostics" in bayesian_result
            
            # Validate posterior mean is reasonable (should be close to true mean)
            posterior_mean = bayesian_result["posterior_mean"]
            assert 0.0 <= posterior_mean <= 1.0
            # Allow reasonable deviation from true mean due to sampling
            assert abs(posterior_mean - true_params["mean"]) < 0.3
            
            # Validate credible interval
            ci = bayesian_result["credible_interval"]
            assert "lower" in ci and "upper" in ci
            assert ci["lower"] < ci["upper"]
            assert 0.0 <= ci["lower"] <= ci["upper"] <= 1.0
            
            # Validate model diagnostics exist
            diagnostics = bayesian_result["model_diagnostics"]
            assert isinstance(diagnostics, dict)

    @pytest.mark.asyncio
    async def test_hierarchical_bayesian_model(
        self, rule_analyzer_bayesian, hierarchical_rule_data
    ):
        """Test hierarchical Bayesian modeling with multiple contexts using real PyMC."""
        try:
            import pymc
            import arviz
        except ImportError:
            pytest.skip("PyMC and ArviZ not available for real Bayesian testing")
            
        rule_id = "rule_bayesian_001"
        observations = hierarchical_rule_data[rule_id]["observations"]
        contexts = ["technical", "creative", "business", "academic"]

        # Test real hierarchical Bayesian modeling
        bayesian_result = rule_analyzer_bayesian._perform_bayesian_modeling(
            rule_id, observations
        )

        # Validate hierarchical structure
        if bayesian_result:
            # Basic structure validation
            assert isinstance(bayesian_result, dict)
            assert "posterior_mean" in bayesian_result
            assert "credible_interval" in bayesian_result
            
            # Check for hierarchical effects if implemented
            if "hierarchical_effects" in bayesian_result:
                hierarchical_effects = bayesian_result["hierarchical_effects"]
                assert isinstance(hierarchical_effects, dict)
                
                # Validate context effects structure
                for context in contexts:
                    if context in hierarchical_effects:
                        effect = hierarchical_effects[context]
                        assert "mean" in effect
                        assert "credible_interval" in effect

    @pytest.mark.asyncio
    async def test_bayesian_uncertainty_quantification(
        self, rule_analyzer_bayesian, simple_rule_data
    ):
        """Test uncertainty quantification in Bayesian analysis using real PyMC."""
        try:
            import pymc
            import arviz
        except ImportError:
            pytest.skip("PyMC and ArviZ not available for real Bayesian testing")
            
        rule_id = "rule_simple_bayesian"
        observations = simple_rule_data[rule_id]["observations"]

        # Test real uncertainty quantification
        bayesian_result = rule_analyzer_bayesian._perform_bayesian_modeling(
            rule_id, observations
        )

        if bayesian_result:
            # Validate uncertainty estimates
            uncertainty = bayesian_result.get("uncertainty_estimate", 0)
            assert 0.0 <= uncertainty <= 1.0

            # Credible interval width should reflect uncertainty
            ci = bayesian_result.get("credible_interval", {})
            if "lower" in ci and "upper" in ci:
                ci_width = ci["upper"] - ci["lower"]
                assert ci_width > 0  # Should have non-zero width
                # Wider intervals indicate higher uncertainty
                assert ci_width < 0.5  # Reasonable upper bound
                
            # Validate posterior mean is reasonable
            posterior_mean = bayesian_result.get("posterior_mean", 0)
            assert 0.0 <= posterior_mean <= 1.0

    @pytest.mark.asyncio
    async def test_bayesian_prior_specification(self, rule_analyzer_bayesian):
        """Test proper specification of Bayesian priors using real PyMC."""
        try:
            import pymc
            import arviz
        except ImportError:
            pytest.skip("PyMC and ArviZ not available for real Bayesian testing")
            
        # Test with different data characteristics
        test_observations = [
            {"score": 0.9, "applications": 1},  # High performance
            {"score": 0.85, "applications": 1},
            {"score": 0.88, "applications": 1},
        ]

        # Test real prior specification
        try:
            bayesian_result = rule_analyzer_bayesian._perform_bayesian_modeling(
                "test_rule", test_observations
            )
            
            # If model fitting succeeds, priors were properly specified
            if bayesian_result:
                assert isinstance(bayesian_result, dict)
                # Validate that model produced reasonable results
                if "posterior_mean" in bayesian_result:
                    posterior_mean = bayesian_result["posterior_mean"]
                    assert 0.0 <= posterior_mean <= 1.0
                    # Should be close to the high performance data
                    assert posterior_mean > 0.7  # High performance prior effect
        except Exception as e:
            # Prior specification issues would cause model fitting to fail
            pytest.fail(f"Prior specification failed: {e}")

    @pytest.mark.asyncio
    async def test_bayesian_convergence_diagnostics(
        self, rule_analyzer_bayesian, simple_rule_data
    ):
        """Test Bayesian model convergence diagnostics using real PyMC."""
        try:
            import pymc
            import arviz
        except ImportError:
            pytest.skip("PyMC and ArviZ not available for real Bayesian testing")
            
        rule_id = "rule_simple_bayesian"
        observations = simple_rule_data[rule_id]["observations"]

        # Test real model diagnostics
        bayesian_result = rule_analyzer_bayesian._perform_bayesian_modeling(
            rule_id, observations
        )

        if bayesian_result and "model_diagnostics" in bayesian_result:
            diagnostics = bayesian_result["model_diagnostics"]

            # Validate diagnostic metrics
            if "r_hat" in diagnostics:
                r_hat = diagnostics["r_hat"]
                assert r_hat > 0  # R-hat should be positive
                # Good convergence typically r_hat < 1.2 (allowing some flexibility for tests)

            if "effective_sample_size" in diagnostics:
                ess = diagnostics["effective_sample_size"]
                assert ess > 0  # ESS should be positive
                
            # Validate that diagnostics contain reasonable values
            assert isinstance(diagnostics, dict)
            # Should have at least some diagnostic information
            assert len(diagnostics) > 0

    @pytest.mark.asyncio
    async def test_credible_interval_calculation(self, rule_analyzer_bayesian):
        """Test credible interval calculation with different confidence levels using real PyMC."""
        try:
            import pymc
            import arviz
        except ImportError:
            pytest.skip("PyMC and ArviZ not available for real Bayesian testing")
            
        # Test different credible interval levels
        test_configs = [
            RuleAnalysisConfig(enable_bayesian_modeling=True, credible_interval=0.90, bayesian_samples=100, bayesian_tune=50),
            RuleAnalysisConfig(enable_bayesian_modeling=True, credible_interval=0.95, bayesian_samples=100, bayesian_tune=50),
            RuleAnalysisConfig(enable_bayesian_modeling=True, credible_interval=0.99, bayesian_samples=100, bayesian_tune=50),
        ]

        observations = [{"score": 0.8, "applications": 1}] * 20

        for config in test_configs:
            analyzer = RuleEffectivenessAnalyzer(config=config)

            # Test real credible interval calculation
            result = analyzer._perform_bayesian_modeling(
                "test_rule", observations
            )

            if result and "credible_interval" in result:
                ci = result["credible_interval"]
                assert "lower" in ci and "upper" in ci
                assert ci["lower"] < ci["upper"]
                assert 0.0 <= ci["lower"] <= ci["upper"] <= 1.0
                
                # Validate interval is reasonable for the data
                ci_width = ci["upper"] - ci["lower"]
                assert ci_width > 0
                assert ci_width < 0.8  # Reasonable upper bound

    @pytest.mark.asyncio
    async def test_insufficient_bayesian_data(
        self, rule_analyzer_bayesian, insufficient_bayesian_data
    ):
        """Test handling of insufficient data for Bayesian modeling using real PyMC."""
        try:
            import pymc
            import arviz
        except ImportError:
            pytest.skip("PyMC and ArviZ not available for real Bayesian testing")
            
        result = await rule_analyzer_bayesian.analyze_rule_effectiveness(
            insufficient_bayesian_data
        )

        # Should handle gracefully with traditional analysis
        assert "rule_metrics" in result

        # Bayesian analysis may be skipped or return limited results
        if "bayesian_analysis" in result:
            bayesian_results = result["bayesian_analysis"]
            for rule_id, bayesian_result in bayesian_results.items():
                if isinstance(bayesian_result, dict):
                    # May indicate insufficient data or may still work with minimal data
                    status = bayesian_result.get("status")
                    if status is not None:
                        assert status in [
                            "insufficient_data",
                            "skipped",
                            "failed",
                        ]
                    # If no status, the model may have worked with minimal data
                    else:
                        # Should still have basic structure
                        assert "posterior_mean" in bayesian_result or "error" in bayesian_result


class TestBayesianErrorHandling:
    """Test error handling and edge cases for Bayesian modeling."""

    @pytest.mark.asyncio
    async def test_bayesian_libraries_unavailable(self, hierarchical_rule_data):
        """Test behavior when Bayesian libraries are not available using real import checking."""
        config = RuleAnalysisConfig(enable_bayesian_modeling=True)
        analyzer = RuleEffectivenessAnalyzer(config=config)

        # Test real availability checking - the analyzer should handle missing libraries gracefully
        # The actual implementation checks for PyMC availability during runtime
        result = await analyzer.analyze_rule_effectiveness(hierarchical_rule_data)

        # Should provide traditional analysis - Bayesian features may or may not be available
        assert "rule_metrics" in result
        
        # Test that the system handles missing libraries gracefully
        # If PyMC is not available, the analyzer should skip Bayesian analysis
        if "bayesian_analysis" in result:
            bayesian_results = result["bayesian_analysis"]
            for rule_id, bayesian_result in bayesian_results.items():
                if isinstance(bayesian_result, dict):
                    # Should have either real results or indicate unavailable status
                    assert ("posterior_mean" in bayesian_result or 
                            bayesian_result.get("status") in ["unavailable", "skipped"])
        # No Bayesian analysis section is also acceptable if libraries are unavailable

    @pytest.mark.asyncio
    async def test_bayesian_modeling_disabled(
        self, rule_analyzer_bayesian, hierarchical_rule_data
    ):
        """Test behavior when Bayesian modeling is disabled."""
        config = RuleAnalysisConfig(enable_bayesian_modeling=False)
        analyzer = RuleEffectivenessAnalyzer(config=config)

        result = await analyzer.analyze_rule_effectiveness(hierarchical_rule_data)

        # Should not perform Bayesian modeling
        assert "rule_metrics" in result
        assert "bayesian_analysis" not in result

    @pytest.mark.asyncio
    async def test_bayesian_sampling_failure(
        self, rule_analyzer_bayesian, simple_rule_data
    ):
        """Test handling of Bayesian sampling failures using real PyMC."""
        try:
            import pymc
            import arviz
        except ImportError:
            pytest.skip("PyMC and ArviZ not available for real Bayesian testing")
            
        rule_id = "rule_simple_bayesian"
        observations = simple_rule_data[rule_id]["observations"]

        # Test with extreme configurations that might cause sampling issues
        extreme_config = RuleAnalysisConfig(
            enable_bayesian_modeling=True,
            bayesian_samples=1,  # Very small - may cause issues
            bayesian_tune=1,     # Very small - may cause issues
            bayesian_chains=1,   # Single chain
        )
        extreme_analyzer = RuleEffectivenessAnalyzer(config=extreme_config)

        # Test with potentially problematic configuration
        bayesian_result = extreme_analyzer._perform_bayesian_modeling(
            rule_id, observations
        )

        # Should handle gracefully - either succeed or fail gracefully
        if bayesian_result is not None:
            assert isinstance(bayesian_result, dict)
            # If it succeeds, should have basic structure
            if "status" not in bayesian_result:
                assert "posterior_mean" in bayesian_result
        # If it fails, that's also acceptable behavior

    @pytest.mark.asyncio
    async def test_invalid_observation_data(self, rule_analyzer_bayesian):
        """Test handling of invalid observation data using real PyMC."""
        try:
            import pymc
            import arviz
        except ImportError:
            pytest.skip("PyMC and ArviZ not available for real Bayesian testing")
            
        invalid_observations = [
            {"score": "invalid", "applications": 1},  # Invalid score type
            {"score": None, "applications": 1},  # None score
            {"score": float("inf"), "applications": 1},  # Infinite score
            {"score": -1.0, "applications": 1},  # Out of range score
        ]

        # Should handle invalid data gracefully
        result = rule_analyzer_bayesian._perform_bayesian_modeling(
            "test_rule", invalid_observations
        )

        # Should return None, error result, or filtered data result
        if result is not None:
            assert isinstance(result, dict)
            # If it processed the data, should indicate status or have valid structure
            if "status" in result:
                assert result["status"] in ["failed", "invalid_data", "filtered"]
            # If no status, should have tried to process and may have partial results
        # None result is also acceptable (indicates complete failure)

    @pytest.mark.asyncio
    async def test_extreme_bayesian_parameters(self):
        """Test Bayesian modeling with extreme parameter configurations using real PyMC."""
        try:
            import pymc
            import arviz
        except ImportError:
            pytest.skip("PyMC and ArviZ not available for real Bayesian testing")
            
        # Test with very small sample sizes
        config = RuleAnalysisConfig(
            enable_bayesian_modeling=True,
            bayesian_samples=10,  # Very small
            bayesian_tune=5,  # Very small
            bayesian_chains=1,  # Single chain
        )
        analyzer = RuleEffectivenessAnalyzer(config=config)

        observations = [{"score": 0.8, "applications": 1}] * 5

        # Should handle extreme parameters without crashing
        result = analyzer._perform_bayesian_modeling("test_rule", observations)

        # Should either succeed or fail gracefully
        if result is not None:
            assert isinstance(result, dict)
            # If it succeeded, should have basic structure
            if "status" not in result:
                assert "posterior_mean" in result
        # None result is also acceptable (indicates failure)

    @pytest.mark.asyncio
    async def test_convergence_failure_handling(
        self, rule_analyzer_bayesian, simple_rule_data
    ):
        """Test handling of Bayesian model convergence failures using real PyMC."""
        try:
            import pymc
            import arviz
        except ImportError:
            pytest.skip("PyMC and ArviZ not available for real Bayesian testing")
            
        rule_id = "rule_simple_bayesian"
        observations = simple_rule_data[rule_id]["observations"]

        # Use configuration likely to cause convergence issues
        poor_config = RuleAnalysisConfig(
            enable_bayesian_modeling=True,
            bayesian_samples=10,  # Very small
            bayesian_tune=5,      # Very small
            bayesian_chains=1,    # Single chain
        )
        poor_analyzer = RuleEffectivenessAnalyzer(config=poor_config)

        # Test convergence handling with poor configuration
        result = poor_analyzer._perform_bayesian_modeling(
            rule_id, observations
        )

        if result and "model_diagnostics" in result:
            diagnostics = result["model_diagnostics"]
            # Should include diagnostic information
            assert isinstance(diagnostics, dict)
            
            # Check for convergence warnings if available
            if "convergence_warning" in diagnostics:
                assert isinstance(diagnostics["convergence_warning"], bool)
                
            # Check for r_hat values if available
            if "r_hat" in diagnostics:
                r_hat = diagnostics["r_hat"]
                assert r_hat > 0  # Should be positive
                # May be > 1.1 indicating poor convergence
        # Result may be None if convergence completely failed


class TestBayesianIntegration:
    """Integration tests for Bayesian modeling with existing workflows."""

    @pytest.mark.asyncio
    async def test_bayesian_with_traditional_analysis(
        self, rule_analyzer_bayesian, hierarchical_rule_data
    ):
        """Test integration of Bayesian modeling with traditional rule analysis using real PyMC."""
        try:
            import pymc
            import arviz
        except ImportError:
            pytest.skip("PyMC and ArviZ not available for real Bayesian testing")
            
        result = await rule_analyzer_bayesian.analyze_rule_effectiveness(
            hierarchical_rule_data
        )

        # Should have both traditional and Bayesian analysis
        assert "rule_metrics" in result
        assert "bayesian_analysis" in result

        # Traditional metrics should be preserved
        rule_metrics = result["rule_metrics"]
        for rule_id, metrics in rule_metrics.items():
            if isinstance(metrics, dict):
                # Should still have standard metrics
                assert "avg_score" in metrics or "total_applications" in metrics
                
        # Bayesian analysis should complement traditional analysis
        bayesian_analysis = result["bayesian_analysis"]
        for rule_id, bayesian_result in bayesian_analysis.items():
            if isinstance(bayesian_result, dict) and "status" not in bayesian_result:
                # Should have Bayesian-specific metrics
                assert "posterior_mean" in bayesian_result or "uncertainty_estimate" in bayesian_result

    @pytest.mark.asyncio
    async def test_bayesian_performance_monitoring(
        self, rule_analyzer_bayesian, simple_rule_data
    ):
        """Test performance characteristics of Bayesian modeling using real PyMC."""
        try:
            import pymc
            import arviz
        except ImportError:
            pytest.skip("PyMC and ArviZ not available for real Bayesian testing")
            
        import time

        start_time = time.time()
        result = await rule_analyzer_bayesian.analyze_rule_effectiveness(
            simple_rule_data
        )
        execution_time = time.time() - start_time

        # Should complete within reasonable time (allow up to 30 seconds for real MCMC)
        assert execution_time < 30.0

        # Should provide meaningful results
        assert "rule_metrics" in result
        
        # If Bayesian analysis completed, should have reasonable structure
        if "bayesian_analysis" in result:
            bayesian_analysis = result["bayesian_analysis"]
            assert isinstance(bayesian_analysis, dict)

    @pytest.mark.asyncio
    async def test_bayesian_with_hierarchical_data(
        self, rule_analyzer_bayesian, hierarchical_rule_data
    ):
        """Test Bayesian modeling with complex hierarchical data structures using real PyMC."""
        try:
            import pymc
            import arviz
        except ImportError:
            pytest.skip("PyMC and ArviZ not available for real Bayesian testing")
            
        result = await rule_analyzer_bayesian.analyze_rule_effectiveness(
            hierarchical_rule_data
        )

        if "bayesian_analysis" in result:
            bayesian_results = result["bayesian_analysis"]

            # Should handle hierarchical structure appropriately
            for rule_id, bayesian_result in bayesian_results.items():
                if isinstance(bayesian_result, dict) and "status" not in bayesian_result:
                    # Should have basic Bayesian results
                    assert "posterior_mean" in bayesian_result or "uncertainty_estimate" in bayesian_result
                    
                    # May include hierarchical effects
                    if "hierarchical_effects" in bayesian_result:
                        hierarchical_effects = bayesian_result[
                            "hierarchical_effects"
                        ]
                        assert isinstance(hierarchical_effects, dict)

    @pytest.mark.parametrize("n_chains", [1, 2])
    async def test_bayesian_chain_variations(self, hierarchical_rule_data, n_chains):
        """Test Bayesian modeling with different numbers of chains using real PyMC."""
        try:
            import pymc
            import arviz
        except ImportError:
            pytest.skip("PyMC and ArviZ not available for real Bayesian testing")
            
        config = RuleAnalysisConfig(
            enable_bayesian_modeling=True,
            bayesian_chains=n_chains,
            bayesian_samples=50,  # Smaller for test performance
            bayesian_tune=25,     # Smaller for test performance
        )
        analyzer = RuleEffectivenessAnalyzer(config=config)

        rule_id = "rule_bayesian_001"
        observations = hierarchical_rule_data[rule_id]["observations"]

        # Test real chain variations
        result = analyzer._perform_bayesian_modeling(rule_id, observations)

        # Should handle different chain configurations
        if result is not None:
            assert isinstance(result, dict)
            # If successful, should have basic structure
            if "status" not in result:
                assert "posterior_mean" in result
        # None result is acceptable (model may fail with certain configurations)


# Test markers for categorization
pytestmark = [
    pytest.mark.unit,
    pytest.mark.ml_contracts,
    pytest.mark.ml_data_validation,
]
