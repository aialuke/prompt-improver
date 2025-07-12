"""
Tests for CUPED Variance Reduction Implementation

Tests the enhanced A/B Testing Framework CUPED functionality for variance reduction
in controlled experiments using pre-experiment data.

Follows pytest best practices:
- Comprehensive fixture-based test setup
- Parametrized tests for multiple scenarios  
- Statistical validation with proper error handling
- Performance benchmarking
- Integration testing patterns
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from scipy import stats
import warnings

from src.prompt_improver.services.ab_testing import ABTestingService, ExperimentResult


class TestCUPEDVarianceReduction:
    """Test suite for CUPED variance reduction with realistic experimental scenarios"""
    
    @pytest.fixture(scope="class")
    def ab_testing_service(self):
        """Create an A/B testing service instance for testing"""
        return ABTestingService()
    
    @pytest.fixture(scope="function")
    def random_seed(self):
        """Set reproducible random seed for each test"""
        np.random.seed(42)
        return 42
    
    @pytest.fixture
    def sample_sizes(self):
        """Standard sample sizes for testing"""
        return {
            'small': 30,
            'medium': 100, 
            'large': 500
        }
    
    @pytest.fixture(params=[
        (0.3, 0.01, 0.09),   # weak correlation, small effect
        (0.6, 0.02, 0.36),   # medium correlation, medium effect
        (0.8, 0.03, 0.64),   # strong correlation, large effect
    ])
    def correlation_scenario(self, request):
        """Parametrized fixture for different correlation scenarios"""
        correlation, treatment_effect, expected_variance_reduction = request.param
        return {
            'correlation': correlation,
            'treatment_effect': treatment_effect,
            'expected_variance_reduction': expected_variance_reduction
        }
    
    @pytest.fixture
    def experimental_data_factory(self, random_seed, sample_sizes):
        """Factory fixture for generating experimental data with varying parameters"""
        def _create_data(correlation=0.6, treatment_effect=0.02, n_per_group=100, baseline=0.15, noise_scale=0.03):
            """
            Create experimental data with specified parameters
            
            Args:
                correlation: Correlation between pre-experiment and outcome data
                treatment_effect: True treatment effect size
                n_per_group: Sample size per group
                baseline: Baseline conversion rate
                noise_scale: Scale of random noise
            """
            # Generate pre-experiment values
            control_pre = np.random.normal(baseline, baseline/3, n_per_group)
            treatment_pre = np.random.normal(baseline, baseline/3, n_per_group)
            
            # Generate correlated outcomes with controlled noise
            noise_variance = noise_scale * np.sqrt(1 - correlation**2)
            
            control_outcome = (correlation * control_pre + 
                             np.random.normal(baseline/3, noise_variance, n_per_group))
            treatment_outcome = (correlation * treatment_pre + treatment_effect +
                               np.random.normal(baseline/3, noise_variance, n_per_group))
            
            return {
                'control': {
                    'outcome': control_outcome,
                    'pre_value': control_pre
                },
                'treatment': {
                    'outcome': treatment_outcome, 
                    'pre_value': treatment_pre
                },
                'expected_treatment_effect': treatment_effect,
                'expected_variance_reduction': correlation**2,
                'true_correlation': correlation
            }
        return _create_data
    
    @pytest.fixture  
    def experimental_data(self, experimental_data_factory):
        """Default experimental data for most tests"""
        return experimental_data_factory()
    
    def test_cuped_basic_functionality(self, ab_testing_service, experimental_data):
        """Test basic CUPED variance reduction calculation"""
        control_data = experimental_data['control']
        treatment_data = experimental_data['treatment']
        
        cuped_results = ab_testing_service._apply_cuped_analysis(treatment_data, control_data)
        
        # Verify all required fields are present
        required_fields = [
            'treatment_effect_cuped', 'p_value', 'confidence_interval',
            'variance_reduction_percent', 'theta_coefficient', 'original_effect',
            'power_improvement_factor', 'recommendation'
        ]
        
        for field in required_fields:
            assert field in cuped_results, f"Missing required field: {field}"
        
        # Verify data types
        assert isinstance(cuped_results['treatment_effect_cuped'], float)
        assert isinstance(cuped_results['p_value'], float)
        assert isinstance(cuped_results['confidence_interval'], list)
        assert len(cuped_results['confidence_interval']) == 2
        assert isinstance(cuped_results['variance_reduction_percent'], float)
        assert isinstance(cuped_results['theta_coefficient'], float)
        assert isinstance(cuped_results['recommendation'], str)
    
    def test_cuped_variance_reduction_effectiveness(self, ab_testing_service, experimental_data):
        """Test that CUPED achieves meaningful variance reduction"""
        control_data = experimental_data['control']
        treatment_data = experimental_data['treatment']
        
        cuped_results = ab_testing_service._apply_cuped_analysis(treatment_data, control_data)
        
        # Validate variance reduction using statistical bounds
        expected_reduction = experimental_data['expected_variance_reduction']
        actual_reduction = cuped_results['variance_reduction_percent'] / 100
        
        # Should achieve variance reduction within reasonable bounds of theoretical expectation
        reduction_tolerance = 0.15  # Allow 15% tolerance due to finite sample effects
        assert actual_reduction >= max(0.05, expected_reduction - reduction_tolerance), \
            f"Variance reduction {actual_reduction:.3f} below expected {expected_reduction:.3f}"
        
        # Power improvement factor should be consistent with variance reduction
        expected_power_factor = 1 / np.sqrt(1 - actual_reduction)
        assert abs(cuped_results['power_improvement_factor'] - expected_power_factor) < 0.1
        
        # Theta coefficient should reflect true correlation (with finite sample noise)
        true_correlation = experimental_data['true_correlation']
        theta_magnitude = abs(cuped_results['theta_coefficient'])
        assert 0.5 * true_correlation <= theta_magnitude <= 1.5 * true_correlation, \
            f"Theta {theta_magnitude:.3f} not consistent with correlation {true_correlation:.3f}"
        
        # Treatment effect should be statistically unbiased
        expected_effect = experimental_data['expected_treatment_effect']
        actual_effect = cuped_results['treatment_effect_cuped']
        effect_tolerance = 2 * np.sqrt(actual_reduction) * 0.01  # Scale tolerance with noise level
        assert abs(actual_effect - expected_effect) < max(0.005, effect_tolerance), \
            f"Effect bias {abs(actual_effect - expected_effect):.4f} exceeds tolerance {effect_tolerance:.4f}"
    
    def test_cuped_unbiased_estimation(self, ab_testing_service, experimental_data):
        """Test that CUPED preserves unbiased treatment effect estimation"""
        control_data = experimental_data['control']
        treatment_data = experimental_data['treatment']
        
        # Calculate original treatment effect
        original_effect = np.mean(treatment_data['outcome']) - np.mean(control_data['outcome'])
        
        cuped_results = ab_testing_service._apply_cuped_analysis(treatment_data, control_data)
        cuped_effect = cuped_results['treatment_effect_cuped']
        original_effect_reported = cuped_results['original_effect']
        
        # Original effect should match calculation
        assert abs(original_effect - original_effect_reported) < 1e-10
        
        # CUPED effect should be close to original (unbiased)
        bias = abs(cuped_effect - original_effect)
        assert bias < 0.005  # Less than 0.5% bias tolerance
    
    def test_cuped_confidence_intervals(self, ab_testing_service, experimental_data):
        """Test CUPED confidence interval calculation"""
        control_data = experimental_data['control']
        treatment_data = experimental_data['treatment']
        
        cuped_results = ab_testing_service._apply_cuped_analysis(treatment_data, control_data)
        
        ci_lower, ci_upper = cuped_results['confidence_interval']
        treatment_effect = cuped_results['treatment_effect_cuped']
        
        # CI should contain the treatment effect
        assert ci_lower <= treatment_effect <= ci_upper
        
        # CI should be reasonable width (not too narrow or too wide)
        ci_width = ci_upper - ci_lower
        assert 0.005 < ci_width < 0.1  # Between 0.5% and 10%
        
        # Lower bound should be less than upper bound
        assert ci_lower < ci_upper
    
    def test_cuped_statistical_significance(self, ab_testing_service, experimental_data):
        """Test CUPED statistical significance testing"""
        control_data = experimental_data['control']
        treatment_data = experimental_data['treatment']
        
        cuped_results = ab_testing_service._apply_cuped_analysis(treatment_data, control_data)
        
        # P-value should be valid
        assert 0 <= cuped_results['p_value'] <= 1
        
        # With 2% treatment effect and variance reduction, should be significant
        assert cuped_results['p_value'] < 0.05
        
        # Recommendation should reflect significance
        recommendation = cuped_results['recommendation'].lower()
        assert 'significant' in recommendation
    
    def test_cuped_with_weak_covariate(self, ab_testing_service):
        """Test CUPED with weakly correlated pre-experiment data"""
        np.random.seed(42)
        
        # Generate data with weak correlation (0.2)
        n = 50
        control_pre = np.random.normal(0.15, 0.05, n)
        treatment_pre = np.random.normal(0.15, 0.05, n)
        
        # Weak correlation + noise
        control_outcome = 0.2 * control_pre + np.random.normal(0.1, 0.05, n)
        treatment_outcome = 0.2 * treatment_pre + 0.02 + np.random.normal(0.1, 0.05, n)
        
        control_data = {'outcome': control_outcome, 'pre_value': control_pre}
        treatment_data = {'outcome': treatment_outcome, 'pre_value': treatment_pre}
        
        cuped_results = ab_testing_service._apply_cuped_analysis(treatment_data, control_data)
        
        # Should still work but with limited variance reduction
        assert cuped_results['variance_reduction_percent'] < 20
        
        # Recommendation should mention weak covariate
        recommendation = cuped_results['recommendation'].lower()
        assert 'weak' in recommendation or 'consider alternative' in recommendation
    
    def test_cuped_with_no_correlation(self, ab_testing_service):
        """Test CUPED with uncorrelated pre-experiment data"""
        np.random.seed(42)
        
        # Generate uncorrelated data
        n = 50
        control_pre = np.random.normal(0.15, 0.05, n)
        treatment_pre = np.random.normal(0.15, 0.05, n)
        
        # No correlation - independent outcomes
        control_outcome = np.random.normal(0.12, 0.05, n)
        treatment_outcome = np.random.normal(0.14, 0.05, n)
        
        control_data = {'outcome': control_outcome, 'pre_value': control_pre}
        treatment_data = {'outcome': treatment_outcome, 'pre_value': treatment_pre}
        
        cuped_results = ab_testing_service._apply_cuped_analysis(treatment_data, control_data)
        
        # Should have minimal or no variance reduction
        assert cuped_results['variance_reduction_percent'] < 10
        
        # Theta coefficient should be close to zero
        assert abs(cuped_results['theta_coefficient']) < 0.5
    
    def test_cuped_insufficient_data(self, ab_testing_service):
        """Test CUPED with insufficient data"""
        # Very small sample sizes
        control_data = {
            'outcome': np.array([0.1, 0.12]),
            'pre_value': np.array([0.08, 0.09])
        }
        treatment_data = {
            'outcome': np.array([0.13, 0.15]),
            'pre_value': np.array([0.10, 0.11])
        }
        
        cuped_results = ab_testing_service._apply_cuped_analysis(treatment_data, control_data)
        
        # Should return empty dict for insufficient data
        assert cuped_results == {}
    
    def test_cuped_edge_cases(self, ab_testing_service):
        """Test CUPED with edge cases"""
        np.random.seed(42)
        
        # Case 1: Zero variance in pre-experiment data
        control_data_zero_var = {
            'outcome': np.random.normal(0.12, 0.02, 30),
            'pre_value': np.full(30, 0.15)  # All same values
        }
        treatment_data_zero_var = {
            'outcome': np.random.normal(0.14, 0.02, 30),
            'pre_value': np.full(30, 0.15)  # All same values
        }
        
        cuped_results = ab_testing_service._apply_cuped_analysis(treatment_data_zero_var, control_data_zero_var)
        
        # Should handle zero variance gracefully
        assert 'theta_coefficient' in cuped_results
        assert cuped_results['theta_coefficient'] == 0  # Should be zero due to zero variance
        
        # Case 2: Identical outcome values
        control_data_identical = {
            'outcome': np.full(30, 0.12),
            'pre_value': np.random.normal(0.15, 0.02, 30)
        }
        treatment_data_identical = {
            'outcome': np.full(30, 0.12),
            'pre_value': np.random.normal(0.15, 0.02, 30)
        }
        
        cuped_results_identical = ab_testing_service._apply_cuped_analysis(treatment_data_identical, control_data_identical)
        
        # Should detect no treatment effect
        assert abs(cuped_results_identical['treatment_effect_cuped']) < 1e-10
    
    def test_cuped_error_handling(self, ab_testing_service):
        """Test CUPED error handling with invalid data"""
        # Test with invalid data types
        invalid_data = {
            'outcome': "invalid",
            'pre_value': [1, 2, 3]
        }
        
        cuped_results = ab_testing_service._apply_cuped_analysis(invalid_data, invalid_data)
        
        # Should return empty dict on error
        assert cuped_results == {}
        
        # Test with NaN values
        control_data_nan = {
            'outcome': np.array([0.1, np.nan, 0.12]),
            'pre_value': np.array([0.08, 0.09, np.nan])
        }
        treatment_data_nan = {
            'outcome': np.array([0.13, 0.15, np.nan]),
            'pre_value': np.array([0.10, np.nan, 0.11])
        }
        
        cuped_results_nan = ab_testing_service._apply_cuped_analysis(treatment_data_nan, control_data_nan)
        
        # Should handle NaN values gracefully
        assert cuped_results_nan == {}
    
    def test_cuped_recommendation_quality(self, ab_testing_service):
        """Test quality and informativeness of CUPED recommendations"""
        np.random.seed(42)
        
        # High variance reduction scenario
        n = 100
        control_pre = np.random.normal(0.15, 0.05, n)
        treatment_pre = np.random.normal(0.15, 0.05, n)
        
        # Strong correlation (0.8) for high variance reduction
        control_outcome = 0.8 * control_pre + np.random.normal(0.02, 0.02, n)
        treatment_outcome = 0.8 * treatment_pre + 0.03 + np.random.normal(0.02, 0.02, n)
        
        control_data = {'outcome': control_outcome, 'pre_value': control_pre}
        treatment_data = {'outcome': treatment_outcome, 'pre_value': treatment_pre}
        
        cuped_results = ab_testing_service._apply_cuped_analysis(treatment_data, control_data)
        
        recommendation = cuped_results['recommendation']
        
        # Should mention variance reduction percentage
        assert f"{cuped_results['variance_reduction_percent']:.1f}%" in recommendation
        
        # Should provide actionable guidance
        assert any(keyword in recommendation.lower() for keyword in [
            'excellent', 'good', 'continue', 'future', 'significant'
        ])
        
        # Should be properly formatted
        assert len(recommendation) > 50  # Sufficiently detailed
        assert recommendation.endswith('.')  # Proper sentence structure


@pytest.mark.integration
class TestCUPEDIntegration:
    """Integration tests for CUPED with A/B testing framework"""
    
    @pytest.fixture
    def ab_testing_service(self):
        """Create an A/B testing service instance for integration testing"""
        return ABTestingService()
    
    @pytest.mark.asyncio
    async def test_cuped_integration_with_experiment_analysis(self, ab_testing_service):
        """Test CUPED integration with full experiment analysis workflow"""
        # Mock database session and experiment data
        mock_session = AsyncMock()
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "test-experiment-123"
        mock_experiment.experiment_name = "CUPED Test Experiment"
        mock_experiment.control_rules = {"rule_ids": ["control-rule-1"]}
        mock_experiment.treatment_rules = {"rule_ids": ["treatment-rule-1"]}
        mock_experiment.target_metric = "improvement_score"
        mock_experiment.started_at = datetime.now() - timedelta(days=7)
        
        # Create realistic experimental data
        np.random.seed(42)
        control_data = np.random.normal(0.65, 0.15, 50).tolist()  # 65% baseline score
        treatment_data = np.random.normal(0.68, 0.15, 50).tolist()  # 68% treatment score
        
        # Mock the experiment data retrieval
        with patch.object(ab_testing_service, '_get_experiment_data') as mock_get_data:
            mock_get_data.side_effect = [control_data, treatment_data]
            
            # Mock the pre-experiment data for CUPED
            with patch.object(ab_testing_service, '_get_pre_experiment_data') as mock_get_pre_data:
                # Generate correlated pre-experiment data
                control_pre = np.random.normal(0.63, 0.12, 45).tolist()
                treatment_pre = np.random.normal(0.63, 0.12, 45).tolist()
                
                mock_get_pre_data.return_value = {
                    'control': control_pre,
                    'treatment': treatment_pre
                }
                
                # Mock database queries
                mock_session.execute.return_value.scalar_one_or_none.return_value = mock_experiment
                
                # Mock experiment results storage
                with patch.object(ab_testing_service, '_store_experiment_results') as mock_store:
                    mock_store.return_value = None
                    
                    # Analyze experiment
                    result = await ab_testing_service.analyze_experiment(
                        "test-experiment-123", mock_session
                    )
        
        # Verify successful analysis
        assert result["status"] == "success"
        assert "analysis" in result
        
        # Verify standard A/B testing metrics are present
        analysis = result["analysis"]
        assert "control_mean" in analysis
        assert "treatment_mean" in analysis
        assert "effect_size" in analysis
        assert "p_value" in analysis
        assert "statistical_significance" in analysis
        
        # Verify CUPED integration doesn't break existing functionality
        assert isinstance(analysis["control_mean"], float)
        assert isinstance(analysis["treatment_mean"], float)
        assert analysis["treatment_mean"] > analysis["control_mean"]  # Treatment should be better
    
    def test_cuped_performance_benchmarks(self, ab_testing_service):
        """Test CUPED performance with realistic dataset sizes"""
        import time
        
        # Large dataset simulation (1000 samples each group)
        np.random.seed(42)
        n_large = 1000
        
        control_pre_large = np.random.normal(0.15, 0.05, n_large)
        treatment_pre_large = np.random.normal(0.15, 0.05, n_large)
        
        control_outcome_large = 0.6 * control_pre_large + np.random.normal(0.05, 0.03, n_large)
        treatment_outcome_large = 0.6 * treatment_pre_large + 0.02 + np.random.normal(0.05, 0.03, n_large)
        
        control_data_large = {'outcome': control_outcome_large, 'pre_value': control_pre_large}
        treatment_data_large = {'outcome': treatment_outcome_large, 'pre_value': treatment_pre_large}
        
        # Measure execution time
        start_time = time.time()
        cuped_results = ab_testing_service._apply_cuped_analysis(treatment_data_large, control_data_large)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete in reasonable time (< 1 second for 2000 samples)
        assert execution_time < 1.0
        
        # Should still produce valid results
        assert cuped_results['variance_reduction_percent'] > 0
        assert 'treatment_effect_cuped' in cuped_results
    
    @pytest.mark.parametrize("correlation,expected_min_reduction,tolerance", [
        pytest.param(0.3, 5, 5, id="weak_correlation"),
        pytest.param(0.5, 15, 10, id="medium_correlation"),  
        pytest.param(0.7, 35, 15, id="strong_correlation"),
        pytest.param(0.9, 70, 20, id="very_strong_correlation", marks=pytest.mark.slow),
    ])
    def test_cuped_variance_reduction_by_correlation(self, ab_testing_service, experimental_data_factory, 
                                                   correlation, expected_min_reduction, tolerance):
        """Test that CUPED variance reduction scales with pre-experiment correlation
        
        Uses parametrized tests to validate theoretical relationship between
        correlation strength and achievable variance reduction.
        """
        # Generate data with specific correlation using factory
        experimental_data = experimental_data_factory(
            correlation=correlation, 
            treatment_effect=0.02,
            n_per_group=200  # Larger sample for more stable estimates
        )
        
        cuped_results = ab_testing_service._apply_cuped_analysis(
            experimental_data['treatment'], 
            experimental_data['control']
        )
        
        # Theoretical variance reduction is correlation^2
        theoretical_reduction = correlation**2 * 100
        actual_reduction = cuped_results['variance_reduction_percent']
        
        # Should achieve reduction within tolerance of theoretical expectation
        assert actual_reduction >= expected_min_reduction, \
            f"Expected at least {expected_min_reduction}% reduction with {correlation} correlation, got {actual_reduction:.1f}%"
        
        # Should be reasonably close to theoretical value
        assert abs(actual_reduction - theoretical_reduction) <= tolerance, \
            f"Actual reduction {actual_reduction:.1f}% differs from theoretical {theoretical_reduction:.1f}% by more than {tolerance}%"
        
        # Power improvement should follow theoretical relationship
        power_improvement = cuped_results['power_improvement_factor']
        expected_power = 1 / np.sqrt(1 - correlation**2)
        assert abs(power_improvement - expected_power) <= 0.3, \
            f"Power improvement {power_improvement:.2f} differs from expected {expected_power:.2f}"
    
    @pytest.mark.parametrize("sample_size", [30, 100, 500])
    def test_cuped_robustness_across_sample_sizes(self, ab_testing_service, experimental_data_factory, sample_size):
        """Test CUPED performance consistency across different sample sizes"""
        experimental_data = experimental_data_factory(
            correlation=0.6,
            treatment_effect=0.02, 
            n_per_group=sample_size
        )
        
        cuped_results = ab_testing_service._apply_cuped_analysis(
            experimental_data['treatment'],
            experimental_data['control']
        )
        
        # Should work consistently across sample sizes
        assert cuped_results['variance_reduction_percent'] > 20, \
            f"Sample size {sample_size}: variance reduction too low"
        
        # Treatment effect estimation should remain unbiased
        effect_bias = abs(cuped_results['treatment_effect_cuped'] - 0.02)
        max_bias = 0.01 * np.sqrt(100 / sample_size)  # Scale with sample size
        assert effect_bias <= max_bias, \
            f"Sample size {sample_size}: effect bias {effect_bias:.4f} exceeds limit {max_bias:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])