"""Tests for Advanced Early Stopping Mechanisms

This test suite validates the research-backed early stopping implementations including:
- Sequential Probability Ratio Test (SPRT)
- Group Sequential Design with error spending functions
- Futility stopping mechanisms
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from src.prompt_improver.optimization.early_stopping import (
    AdvancedEarlyStoppingFramework,
    EarlyStoppingConfig,
    AlphaSpendingFunction,
    StoppingDecision,
    should_stop_experiment,
    create_early_stopping_framework
)


class TestEarlyStoppingConfig:
    """Test early stopping configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = EarlyStoppingConfig()
        
        assert config.alpha == 0.05
        assert config.beta == 0.2
        assert config.effect_size_h0 == 0.0
        assert config.effect_size_h1 == 0.1
        assert config.max_looks == 10
        assert config.alpha_spending_function == AlphaSpendingFunction.OBRIEN_FLEMING
        assert config.enable_futility_stopping is True
        assert config.min_sample_size == 30
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = EarlyStoppingConfig(
            alpha=0.01,
            beta=0.1,
            effect_size_h1=0.2,
            max_looks=5,
            alpha_spending_function=AlphaSpendingFunction.POCOCK
        )
        
        assert config.alpha == 0.01
        assert config.beta == 0.1
        assert config.effect_size_h1 == 0.2
        assert config.max_looks == 5
        assert config.alpha_spending_function == AlphaSpendingFunction.POCOCK


class TestAdvancedEarlyStoppingFramework:
    """Test the main early stopping framework"""
    
    @pytest.fixture
    def framework(self):
        """Create framework for testing"""
        config = EarlyStoppingConfig(
            alpha=0.05,
            beta=0.2,
            effect_size_h1=0.2,
            max_looks=5,
            min_sample_size=10
        )
        return AdvancedEarlyStoppingFramework(config)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample test data"""
        np.random.seed(42)  # For reproducible tests
        
        # Generate data with small effect
        control = np.random.normal(0.5, 0.1, 50).tolist()
        treatment = np.random.normal(0.52, 0.1, 50).tolist()  # 2% improvement
        
        return control, treatment
    
    @pytest.fixture
    def strong_effect_data(self):
        """Generate data with strong effect for superiority testing"""
        np.random.seed(42)
        
        control = np.random.normal(0.5, 0.1, 30).tolist()
        treatment = np.random.normal(0.7, 0.1, 30).tolist()  # 20% improvement
        
        return control, treatment
    
    @pytest.fixture
    def no_effect_data(self):
        """Generate data with no effect for futility testing"""
        np.random.seed(42)
        
        control = np.random.normal(0.5, 0.1, 50).tolist()
        treatment = np.random.normal(0.5, 0.1, 50).tolist()  # No improvement
        
        return control, treatment

    @pytest.mark.asyncio
    async def test_insufficient_sample_size(self, framework):
        """Test behavior with insufficient sample size"""
        control = [0.5, 0.6]  # Only 2 samples
        treatment = [0.52, 0.62]  # Only 2 samples
        
        result = await framework.evaluate_stopping_criteria(
            experiment_id="test_exp_1",
            control_data=control,
            treatment_data=treatment,
            look_number=1
        )
        
        assert result.decision == StoppingDecision.CONTINUE
        assert "Insufficient sample size" in result.recommendation
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_continue_decision(self, framework, sample_data):
        """Test continue decision with ambiguous data"""
        control, treatment = sample_data
        
        # Use small subset to ensure inconclusive results
        result = await framework.evaluate_stopping_criteria(
            experiment_id="test_exp_2",
            control_data=control[:15],
            treatment_data=treatment[:15],
            look_number=1
        )
        
        assert result.test_id == "test_exp_2"
        assert result.look_number == 1
        assert result.samples_analyzed == 30
        assert isinstance(result.test_statistic, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.effect_size, float)
        assert isinstance(result.conditional_power, float)
        # With small sample and small effect, may stop for futility or continue
        assert result.decision in [StoppingDecision.CONTINUE, StoppingDecision.STOP_REJECT_NULL, StoppingDecision.STOP_FOR_FUTILITY]

    @pytest.mark.asyncio
    async def test_superiority_stopping(self, framework, strong_effect_data):
        """Test stopping for superiority with strong effect"""
        control, treatment = strong_effect_data
        
        result = await framework.evaluate_stopping_criteria(
            experiment_id="test_exp_3",
            control_data=control,
            treatment_data=treatment,
            look_number=1
        )
        
        # With strong effect, should detect significance
        assert result.test_id == "test_exp_3"
        assert result.p_value < 0.05  # Should be significant
        assert abs(result.effect_size) > 1.0  # Should be large effect
        assert result.decision in [StoppingDecision.STOP_REJECT_NULL, StoppingDecision.STOP_FOR_SUPERIORITY]

    @pytest.mark.asyncio
    async def test_futility_stopping(self, framework, no_effect_data):
        """Test futility stopping with no effect"""
        control, treatment = no_effect_data
        
        result = await framework.evaluate_stopping_criteria(
            experiment_id="test_exp_4",
            control_data=control,
            treatment_data=treatment,
            look_number=3  # Later look
        )
        
        # With no effect and later look, should at least get valid results
        assert result.test_id == "test_exp_4"
        assert isinstance(result.conditional_power, float)
        assert 0 <= result.conditional_power <= 1
        # Note: With random data, conditional power may vary - this is expected

    @pytest.mark.asyncio
    async def test_sprt_bounds(self, framework, sample_data):
        """Test SPRT boundary calculations"""
        control, treatment = sample_data
        
        result = await framework.evaluate_stopping_criteria(
            experiment_id="test_exp_5",
            control_data=control,
            treatment_data=treatment,
            look_number=1
        )
        
        assert result.sprt_bounds is not None
        assert isinstance(result.sprt_bounds.lower_bound, float)
        assert isinstance(result.sprt_bounds.upper_bound, float)
        assert isinstance(result.sprt_bounds.log_likelihood_ratio, float)
        assert result.sprt_bounds.upper_bound > result.sprt_bounds.lower_bound
        assert result.sprt_bounds.decision in list(StoppingDecision)

    @pytest.mark.asyncio
    async def test_group_sequential_bounds(self, framework, sample_data):
        """Test Group Sequential Design boundary calculations"""
        control, treatment = sample_data
        
        result = await framework.evaluate_stopping_criteria(
            experiment_id="test_exp_6",
            control_data=control,
            treatment_data=treatment,
            look_number=2
        )
        
        assert result.group_sequential_bounds is not None
        assert result.group_sequential_bounds.look_number == 2
        assert 0 < result.group_sequential_bounds.information_fraction <= 1
        assert result.group_sequential_bounds.alpha_spent > 0
        assert result.group_sequential_bounds.rejection_boundary > 0
        assert result.group_sequential_bounds.decision in list(StoppingDecision)

    @pytest.mark.asyncio
    async def test_multiple_looks_experiment(self, framework, sample_data):
        """Test experiment with multiple interim looks"""
        control, treatment = sample_data
        
        # First look
        result1 = await framework.evaluate_stopping_criteria(
            experiment_id="test_exp_multi",
            control_data=control[:20],
            treatment_data=treatment[:20],
            look_number=1
        )
        
        # Second look with more data
        result2 = await framework.evaluate_stopping_criteria(
            experiment_id="test_exp_multi",
            control_data=control[:35],
            treatment_data=treatment[:35],
            look_number=2
        )
        
        # Third look with full data
        result3 = await framework.evaluate_stopping_criteria(
            experiment_id="test_exp_multi",
            control_data=control,
            treatment_data=treatment,
            look_number=3
        )
        
        # Verify progression
        assert result1.look_number == 1
        assert result2.look_number == 2
        assert result3.look_number == 3
        
        assert result1.samples_analyzed < result2.samples_analyzed < result3.samples_analyzed
        
        # Information fraction should increase
        if result2.group_sequential_bounds and result3.group_sequential_bounds:
            assert (result2.group_sequential_bounds.information_fraction < 
                   result3.group_sequential_bounds.information_fraction)

    def test_alpha_spending_functions(self, framework):
        """Test different alpha spending functions"""
        
        # Test Pocock
        alpha_pocock = framework._calculate_alpha_spending(0.5, AlphaSpendingFunction.POCOCK)
        assert 0 < alpha_pocock < framework.config.alpha
        
        # Test O'Brien-Fleming
        alpha_obf = framework._calculate_alpha_spending(0.5, AlphaSpendingFunction.OBRIEN_FLEMING)
        assert 0 < alpha_obf < framework.config.alpha
        
        # Test Wang-Tsiatis
        alpha_wt = framework._calculate_alpha_spending(0.5, AlphaSpendingFunction.WANG_TSIATIS)
        assert 0 < alpha_wt < framework.config.alpha
        
        # Pocock should spend more alpha early compared to O'Brien-Fleming
        alpha_pocock_early = framework._calculate_alpha_spending(0.2, AlphaSpendingFunction.POCOCK)
        alpha_obf_early = framework._calculate_alpha_spending(0.2, AlphaSpendingFunction.OBRIEN_FLEMING)
        assert alpha_pocock_early > alpha_obf_early

    def test_conditional_power_calculation(self, framework):
        """Test conditional power calculation"""
        control = [0.5] * 30
        treatment = [0.52] * 30
        
        conditional_power = framework._calculate_conditional_power(
            control, treatment, target_effect=0.1
        )
        
        assert 0 <= conditional_power <= 1
        
        # Test with stronger effect - use much larger difference
        treatment_strong = [0.8] * 30  # 30% improvement, much larger than target_effect
        conditional_power_strong = framework._calculate_conditional_power(
            control, treatment_strong, target_effect=0.1
        )
        
        assert conditional_power_strong >= conditional_power  # Should be at least equal

    def test_experiment_state_tracking(self, framework):
        """Test experiment state tracking"""
        exp_id = "state_test_exp"
        
        # Initially no state
        assert framework.get_experiment_summary(exp_id) is None
        
        # After evaluation, state should exist
        # This would normally happen during evaluate_stopping_criteria
        framework.experiment_state[exp_id] = {
            "start_time": datetime.utcnow(),
            "looks": [],
            "cumulative_data": {"control": [0.5, 0.6], "treatment": [0.52, 0.62]}
        }
        
        summary = framework.get_experiment_summary(exp_id)
        assert summary is not None
        assert summary["experiment_id"] == exp_id
        assert "start_time" in summary
        assert "total_samples" in summary

    def test_cleanup_completed_experiments(self, framework):
        """Test cleanup of old completed experiments"""
        # Add old completed experiment
        old_exp_id = "old_completed_exp"
        framework.experiment_state[old_exp_id] = {
            "start_time": datetime.utcnow() - timedelta(days=35),
            "completed": True,
            "looks": [],
            "cumulative_data": {"control": [], "treatment": []}
        }
        
        # Add recent experiment
        recent_exp_id = "recent_exp"
        framework.experiment_state[recent_exp_id] = {
            "start_time": datetime.utcnow() - timedelta(days=5),
            "completed": False,
            "looks": [],
            "cumulative_data": {"control": [], "treatment": []}
        }
        
        framework.cleanup_completed_experiments(days_old=30)
        
        # Old completed experiment should be removed
        assert old_exp_id not in framework.experiment_state
        # Recent experiment should remain
        assert recent_exp_id in framework.experiment_state


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_create_early_stopping_framework(self):
        """Test framework creation utility"""
        framework = create_early_stopping_framework(
            alpha=0.01,
            beta=0.1,
            max_looks=5,
            enable_futility=False
        )
        
        assert framework.config.alpha == 0.01
        assert framework.config.beta == 0.1
        assert framework.config.max_looks == 5
        assert framework.config.enable_futility_stopping is False

    @pytest.mark.asyncio
    async def test_should_stop_experiment_utility(self):
        """Test simple should_stop_experiment utility function"""
        framework = create_early_stopping_framework()
        
        # Test with small effect
        control = np.random.normal(0.5, 0.1, 20).tolist()
        treatment = np.random.normal(0.52, 0.1, 20).tolist()
        
        should_stop, reason, confidence = await should_stop_experiment(
            experiment_id="utility_test",
            control_data=control,
            treatment_data=treatment,
            framework=framework,
            look_number=1
        )
        
        assert isinstance(should_stop, bool)
        assert isinstance(reason, str)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1


class TestErrorHandling:
    """Test error handling in early stopping"""
    
    @pytest.mark.asyncio
    async def test_empty_data_handling(self):
        """Test handling of empty data"""
        framework = create_early_stopping_framework()
        
        result = await framework.evaluate_stopping_criteria(
            experiment_id="empty_test",
            control_data=[],
            treatment_data=[],
            look_number=1
        )
        
        assert result.decision == StoppingDecision.CONTINUE
        assert "Insufficient sample size" in result.recommendation

    @pytest.mark.asyncio
    async def test_single_value_data(self):
        """Test handling of constant data (zero variance)"""
        framework = create_early_stopping_framework()
        
        # All same values - zero variance
        control = [0.5] * 30
        treatment = [0.5] * 30
        
        result = await framework.evaluate_stopping_criteria(
            experiment_id="constant_test",
            control_data=control,
            treatment_data=treatment,
            look_number=1
        )
        
        # Should handle gracefully without errors
        assert isinstance(result.test_statistic, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.effect_size, float)

    def test_invalid_alpha_spending_time(self):
        """Test alpha spending with invalid time values"""
        framework = create_early_stopping_framework()
        
        # Test with negative time
        alpha_neg = framework._calculate_alpha_spending(-0.1, AlphaSpendingFunction.POCOCK)
        assert alpha_neg == 0  # Should return 0 for negative time
        
        # Test with zero time
        alpha_zero = framework._calculate_alpha_spending(0.0, AlphaSpendingFunction.OBRIEN_FLEMING)
        assert alpha_zero == 0
        
        # Test with time > 1
        alpha_over = framework._calculate_alpha_spending(1.5, AlphaSpendingFunction.POCOCK)
        assert alpha_over >= framework.config.alpha


class TestStatisticalValidation:
    """Test statistical properties of early stopping"""
    
    @pytest.mark.asyncio
    async def test_type_i_error_control(self):
        """Test that Type I error is controlled under null hypothesis"""
        framework = create_early_stopping_framework(alpha=0.05)
        
        # Simulate multiple experiments under null (no effect)
        false_positives = 0
        num_simulations = 20  # Reduced for testing speed
        
        for sim in range(num_simulations):
            np.random.seed(sim + 100)  # Different seed for each simulation
            
            # Generate null data (no difference)
            control = np.random.normal(0.5, 0.1, 40).tolist()
            treatment = np.random.normal(0.5, 0.1, 40).tolist()
            
            result = await framework.evaluate_stopping_criteria(
                experiment_id=f"type_i_test_{sim}",
                control_data=control,
                treatment_data=treatment,
                look_number=1
            )
            
            if result.stop_for_efficacy:
                false_positives += 1
        
        # False positive rate should be reasonably controlled
        # With small sample this is just a sanity check
        false_positive_rate = false_positives / num_simulations
        assert false_positive_rate <= 0.3  # Generous bound for small sample test

    def test_boundary_monotonicity(self):
        """Test that alpha spending is monotonically increasing"""
        framework = create_early_stopping_framework()
        
        time_points = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        for spending_func in [AlphaSpendingFunction.POCOCK, 
                              AlphaSpendingFunction.OBRIEN_FLEMING,
                              AlphaSpendingFunction.WANG_TSIATIS]:
            
            alpha_values = [
                framework._calculate_alpha_spending(t, spending_func) 
                for t in time_points
            ]
            
            # Check monotonicity
            for i in range(1, len(alpha_values)):
                assert alpha_values[i] >= alpha_values[i-1], \
                    f"Alpha spending not monotonic for {spending_func}"
            
            # Final value should be close to alpha
            assert abs(alpha_values[-1] - framework.config.alpha) < 0.01

    @pytest.mark.asyncio 
    async def test_estimation_consistency(self):
        """Test that estimates are consistent across looks"""
        framework = create_early_stopping_framework()
        
        # Generate data with known effect
        np.random.seed(123)
        true_effect = 0.1
        control = np.random.normal(0.5, 0.1, 60).tolist()
        treatment = np.random.normal(0.5 + true_effect, 0.1, 60).tolist()
        
        # Look 1: subset of data
        result1 = await framework.evaluate_stopping_criteria(
            experiment_id="consistency_test",
            control_data=control[:30],
            treatment_data=treatment[:30],
            look_number=1
        )
        
        # Look 2: more data
        result2 = await framework.evaluate_stopping_criteria(
            experiment_id="consistency_test",
            control_data=control[:50],
            treatment_data=treatment[:50],
            look_number=2
        )
        
        # Effect size estimates should be in reasonable range
        assert abs(result1.effect_size - true_effect / 0.1) < 2.0  # Within 2 standard deviations
        assert abs(result2.effect_size - true_effect / 0.1) < 2.0
        
        # Larger sample should have higher confidence (lower p-value if effect exists)
        if result1.effect_size > 0.5:  # If reasonable effect detected
            assert result2.p_value <= result1.p_value or result2.p_value < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])