"""
Unit tests for Advanced Statistical Validator
Tests comprehensive statistical validation with 2025 best practices
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, Mock

from src.prompt_improver.evaluation.advanced_statistical_validator import (
    AdvancedStatisticalValidator,
    CorrectionMethod,
    EffectSizeMagnitude,
    StatisticalTestResult,
    AdvancedValidationResult,
    quick_validation
)


class TestAdvancedStatisticalValidator:
    """Test suite for AdvancedStatisticalValidator"""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance for testing"""
        return AdvancedStatisticalValidator(
            alpha=0.05,
            power_threshold=0.8,
            min_effect_size=0.1,
            bootstrap_samples=1000
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)  # For reproducible tests
        
        # Control group: normal distribution, mean=10, std=2
        control = np.random.normal(10, 2, 100)
        
        # Treatment group: slightly higher mean=10.5, std=2
        treatment = np.random.normal(10.5, 2, 100)
        
        return control.tolist(), treatment.tolist()
    
    @pytest.fixture
    def large_effect_data(self):
        """Create data with large effect size"""
        np.random.seed(42)
        
        # Control group
        control = np.random.normal(10, 2, 100)
        
        # Treatment group with large effect
        treatment = np.random.normal(12, 2, 100)  # 1 standard deviation difference
        
        return control.tolist(), treatment.tolist()
    
    def test_initialization(self):
        """Test validator initialization"""
        validator = AdvancedStatisticalValidator(
            alpha=0.01,
            power_threshold=0.9,
            min_effect_size=0.2,
            bootstrap_samples=5000
        )
        
        assert validator.alpha == 0.01
        assert validator.power_threshold == 0.9
        assert validator.min_effect_size == 0.2
        assert validator.bootstrap_samples == 5000
    
    def test_input_validation(self, validator):
        """Test input data validation"""
        # Empty data
        with pytest.raises(ValueError, match="cannot be empty"):
            validator._validate_input_data(np.array([]), np.array([1, 2, 3]))
        
        # Insufficient data
        with pytest.raises(ValueError, match="Minimum 3 observations"):
            validator._validate_input_data(np.array([1, 2]), np.array([3, 4]))
        
        # Invalid values
        with pytest.raises(ValueError, match="infinite or NaN"):
            validator._validate_input_data(
                np.array([1, 2, np.inf]), 
                np.array([3, 4, 5])
            )
        
        # Zero variance in both groups with identical values
        with pytest.raises(ValueError, match="identical constant values"):
            validator._validate_input_data(
                np.array([5, 5, 5]), 
                np.array([5, 5, 5])
            )
    
    def test_primary_statistical_test(self, validator, sample_data):
        """Test primary statistical test (Welch's t-test)"""
        control, treatment = sample_data
        control_array = np.array(control)
        treatment_array = np.array(treatment)
        
        result = validator._perform_primary_test(control_array, treatment_array)
        
        assert isinstance(result, StatisticalTestResult)
        assert result.test_name == "Welch's t-test"
        assert result.statistic is not None
        assert result.p_value is not None
        assert result.degrees_of_freedom is not None
        assert result.effect_size is not None
        assert result.effect_size_type == "Cohen's d"
        assert result.confidence_interval is not None
        assert result.minimum_detectable_effect is not None
        assert len(result.notes) > 0
    
    def test_cohens_d_calculation(self, validator):
        """Test Cohen's d effect size calculation"""
        control = np.array([8, 9, 10, 11, 12])
        treatment = np.array([10, 11, 12, 13, 14])
        
        cohens_d = validator._calculate_cohens_d(control, treatment)
        
        # Expected Cohen's d for this data should be approximately 1.0 (allow larger tolerance)
        assert abs(cohens_d - 1.0) < 0.3
        
        # Test with identical groups (should be 0)
        identical = np.array([10, 10, 10, 10, 10])
        cohens_d_zero = validator._calculate_cohens_d(identical, identical)
        assert cohens_d_zero == 0.0
    
    def test_effect_size_magnitude_classification(self, validator):
        """Test effect size magnitude classification"""
        control = np.array([10, 10, 10, 10, 10])
        
        # Negligible effect
        treatment_negligible = np.array([10.05, 10.05, 10.05, 10.05, 10.05])
        magnitude, practical, clinical = validator._analyze_effect_size(
            control, treatment_negligible, 0.05
        )
        assert magnitude == EffectSizeMagnitude.NEGLIGIBLE
        assert not practical
        
        # Small effect
        treatment_small = np.array([10.2, 10.2, 10.2, 10.2, 10.2])
        magnitude, practical, clinical = validator._analyze_effect_size(
            control, treatment_small, 0.2
        )
        assert magnitude == EffectSizeMagnitude.SMALL
        
        # Large effect
        treatment_large = np.array([11.5, 11.5, 11.5, 11.5, 11.5])
        magnitude, practical, clinical = validator._analyze_effect_size(
            control, treatment_large, 0.6
        )
        assert magnitude == EffectSizeMagnitude.LARGE
    
    def test_confidence_interval_calculation(self, validator, sample_data):
        """Test confidence interval calculation"""
        control, treatment = sample_data
        control_array = np.array(control)
        treatment_array = np.array(treatment)
        
        ci = validator._calculate_difference_ci(control_array, treatment_array)
        
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < ci[1]  # Lower bound < Upper bound
        
        # CI should contain the true difference (approximately 0.5)
        true_diff = np.mean(treatment_array) - np.mean(control_array)
        assert ci[0] <= true_diff <= ci[1]
    
    def test_normality_assumptions(self, validator, sample_data):
        """Test normality assumption testing"""
        control, treatment = sample_data
        control_array = np.array(control)
        treatment_array = np.array(treatment)
        
        normality_tests = validator._test_normality_assumptions(control_array, treatment_array)
        
        assert isinstance(normality_tests, dict)
        assert 'shapiro_control' in normality_tests
        assert 'shapiro_treatment' in normality_tests
        assert 'ks_control' in normality_tests
        assert 'ks_treatment' in normality_tests
        
        for test_name, test_result in normality_tests.items():
            assert isinstance(test_result, StatisticalTestResult)
            assert test_result.statistic is not None
            assert test_result.p_value is not None
    
    def test_homogeneity_assumptions(self, validator, sample_data):
        """Test homogeneity of variance assumption testing"""
        control, treatment = sample_data
        control_array = np.array(control)
        treatment_array = np.array(treatment)
        
        homogeneity_tests = validator._test_homogeneity_assumptions(control_array, treatment_array)
        
        assert isinstance(homogeneity_tests, dict)
        assert 'levene' in homogeneity_tests
        assert 'bartlett' in homogeneity_tests
        
        for test_name, test_result in homogeneity_tests.items():
            assert isinstance(test_result, StatisticalTestResult)
            assert test_result.statistic is not None
            assert test_result.p_value is not None
    
    def test_non_parametric_tests(self, validator, sample_data):
        """Test non-parametric robustness checks"""
        control, treatment = sample_data
        control_array = np.array(control)
        treatment_array = np.array(treatment)
        
        non_parametric_tests = validator._perform_non_parametric_tests(control_array, treatment_array)
        
        assert isinstance(non_parametric_tests, dict)
        assert 'mann_whitney' in non_parametric_tests
        
        mann_whitney = non_parametric_tests['mann_whitney']
        assert isinstance(mann_whitney, StatisticalTestResult)
        assert mann_whitney.test_name == "Mann-Whitney U"
        assert mann_whitney.effect_size is not None
        assert mann_whitney.effect_size_type == "Rank-biserial correlation"
    
    def test_power_analysis(self, validator, sample_data):
        """Test statistical power analysis"""
        control, treatment = sample_data
        control_array = np.array(control)
        treatment_array = np.array(treatment)
        
        effect_size = validator._calculate_cohens_d(control_array, treatment_array)
        
        # Post-hoc power
        post_hoc_power = validator._calculate_post_hoc_power(control_array, treatment_array, effect_size)
        assert 0 <= post_hoc_power <= 1
        
        # Prospective power
        prospective_power = validator._calculate_prospective_power(control_array, treatment_array)
        assert 0 <= prospective_power <= 1
    
    def test_bootstrap_analysis(self, validator, sample_data):
        """Test bootstrap analysis"""
        control, treatment = sample_data
        control_array = np.array(control)
        treatment_array = np.array(treatment)
        
        # Use smaller bootstrap samples for faster testing
        validator.bootstrap_samples = 100
        
        bootstrap_results = validator._perform_bootstrap_analysis(control_array, treatment_array)
        
        assert isinstance(bootstrap_results, dict)
        assert 'n_bootstrap_samples' in bootstrap_results
        assert 'mean_difference' in bootstrap_results
        assert 'effect_size' in bootstrap_results
        
        mean_diff = bootstrap_results['mean_difference']
        assert 'bootstrap_mean' in mean_diff
        assert 'bootstrap_std' in mean_diff
        assert 'confidence_interval_95' in mean_diff
        assert len(mean_diff['confidence_interval_95']) == 2
        
        effect_size = bootstrap_results['effect_size']
        assert 'bootstrap_mean' in effect_size
        assert 'confidence_interval_95' in effect_size
    
    def test_sensitivity_analysis(self, validator):
        """Test sensitivity analysis with outliers"""
        # Create data with outliers
        np.random.seed(42)
        control = np.concatenate([np.random.normal(10, 1, 95), [20, 25]])  # 2 outliers
        treatment = np.concatenate([np.random.normal(11, 1, 95), [22, 27]])  # 2 outliers
        
        sensitivity_results = validator._perform_sensitivity_analysis(control, treatment)
        
        assert isinstance(sensitivity_results, dict)
        assert 'outliers_removed' in sensitivity_results
        assert 'effect_size_comparison' in sensitivity_results
        assert 'p_value_comparison' in sensitivity_results
        
        outliers = sensitivity_results['outliers_removed']
        assert outliers['control'] > 0  # Should detect outliers
        assert outliers['treatment'] > 0
        
        effect_comparison = sensitivity_results['effect_size_comparison']
        assert 'original' in effect_comparison
        assert 'cleaned' in effect_comparison
        assert 'robust_to_outliers' in effect_comparison
    
    def test_multiple_testing_correction(self, validator):
        """Test multiple testing correction"""
        p_values = [0.01, 0.03, 0.045, 0.06, 0.12]
        
        correction = validator._apply_multiple_testing_correction(
            p_values, CorrectionMethod.BENJAMINI_HOCHBERG
        )
        
        assert correction.method == CorrectionMethod.BENJAMINI_HOCHBERG
        assert correction.original_p_values == p_values
        assert len(correction.corrected_p_values) == len(p_values)
        assert len(correction.rejected) == len(p_values)
        assert correction.family_wise_error_rate is not None
        assert correction.false_discovery_rate is not None
    
    def test_comprehensive_validation(self, validator, sample_data):
        """Test complete validation workflow"""
        control, treatment = sample_data
        
        result = validator.validate_ab_test(
            control_data=control,
            treatment_data=treatment,
            correction_method=CorrectionMethod.BENJAMINI_HOCHBERG,
            validate_assumptions=True,
            include_bootstrap=True,
            include_sensitivity=True
        )
        
        assert isinstance(result, AdvancedValidationResult)
        assert result.validation_id is not None
        assert result.timestamp is not None
        assert isinstance(result.primary_test, StatisticalTestResult)
        assert isinstance(result.effect_size_magnitude, EffectSizeMagnitude)
        assert isinstance(result.practical_significance, bool)
        assert isinstance(result.clinical_significance, bool)
        
        # Check optional components
        assert result.normality_tests is not None
        assert result.homogeneity_tests is not None
        assert result.non_parametric_tests is not None
        assert result.bootstrap_results is not None
        assert result.sensitivity_analysis is not None
        
        # Check recommendations and warnings
        assert isinstance(result.recommendations, list)
        assert isinstance(result.warnings, list)
        assert 0 <= result.validation_quality_score <= 1
    
    def test_recommendations_generation(self, validator, large_effect_data):
        """Test recommendation generation"""
        control, treatment = large_effect_data
        
        result = validator.validate_ab_test(
            control_data=control,
            treatment_data=treatment,
            validate_assumptions=False,
            include_bootstrap=False,
            include_sensitivity=False
        )
        
        recommendations = result.recommendations
        assert len(recommendations) > 0
        
        # Should recommend deployment for large effect
        deploy_recommendation = any("DEPLOY" in rec for rec in recommendations)
        assert deploy_recommendation
    
    def test_quality_score_calculation(self, validator, sample_data):
        """Test validation quality score calculation"""
        control, treatment = sample_data
        
        result = validator.validate_ab_test(
            control_data=control,
            treatment_data=treatment
        )
        
        quality_score = validator._calculate_validation_quality_score(result)
        
        assert 0 <= quality_score <= 1
        assert quality_score == result.validation_quality_score
    
    def test_quick_validation_utility(self, sample_data):
        """Test quick validation utility function"""
        control, treatment = sample_data
        
        result = quick_validation(control, treatment, alpha=0.05)
        
        assert isinstance(result, dict)
        assert 'statistically_significant' in result
        assert 'p_value' in result
        assert 'effect_size' in result
        assert 'effect_magnitude' in result
        assert 'practical_significance' in result
        assert 'recommendations' in result
        assert 'quality_score' in result
        
        assert isinstance(result['statistically_significant'], bool)
        assert isinstance(result['p_value'], float)
        assert isinstance(result['effect_size'], float)
        assert isinstance(result['practical_significance'], bool)
        assert isinstance(result['recommendations'], list)
        assert 0 <= result['quality_score'] <= 1
    
    def test_error_handling(self, validator):
        """Test error handling in validation"""
        # Test with invalid data that should raise an error
        with pytest.raises(ValueError):
            validator.validate_ab_test(
                control_data=[],  # Empty data
                treatment_data=[1, 2, 3]
            )
    
    def test_edge_cases(self, validator):
        """Test edge cases"""
        # Test with different constant groups (zero variance but different means)
        control_data = [10, 10, 10, 10, 10]
        treatment_data = [12, 12, 12, 12, 12]
        
        result = validator.validate_ab_test(
            control_data=control_data,
            treatment_data=treatment_data,
            validate_assumptions=False,
            include_bootstrap=False,
            include_sensitivity=False
        )
        
        # Should handle constant groups with different means (effect size should be very large)
        assert abs(result.primary_test.effect_size) > 5.0
        assert result.effect_size_magnitude != EffectSizeMagnitude.NEGLIGIBLE
        
        # Very small differences
        control_small = [10.0, 10.0, 10.0, 10.0, 10.0]
        treatment_small = [10.001, 10.001, 10.001, 10.001, 10.001]
        
        result_small = validator.validate_ab_test(
            control_data=control_small,
            treatment_data=treatment_small,
            validate_assumptions=False,
            include_bootstrap=False,
            include_sensitivity=False
        )
        
        # Should detect negligible effect
        assert result_small.effect_size_magnitude == EffectSizeMagnitude.NEGLIGIBLE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])