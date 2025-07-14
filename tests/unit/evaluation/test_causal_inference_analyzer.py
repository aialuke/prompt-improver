"""
Unit tests for Causal Inference Analyzer
Tests causal analysis and counterfactual reasoning for A/B experiments
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, Mock

from src.prompt_improver.evaluation.causal_inference_analyzer import (
    CausalInferenceAnalyzer,
    CausalMethod,
    TreatmentAssignment,
    CausalAssumption,
    CausalEffect,
    CausalInferenceResult,
    quick_causal_analysis
)


class TestCausalInferenceAnalyzer:
    """Test suite for CausalInferenceAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing"""
        return CausalInferenceAnalyzer(
            significance_level=0.05,
            minimum_effect_size=0.1,
            bootstrap_samples=100,  # Reduced for faster testing
            enable_sensitivity_analysis=True
        )
    
    @pytest.fixture
    def randomized_experiment_data(self):
        """Create sample randomized experiment data"""
        np.random.seed(42)
        
        n = 200
        # Randomized treatment assignment
        treatment = np.random.binomial(1, 0.5, n)
        
        # Outcome with true treatment effect of 0.5
        noise = np.random.normal(0, 1, n)
        outcome = 2.0 + 0.5 * treatment + noise
        
        # Some covariates (not confounders in randomized setting)
        covariates = np.random.normal(0, 1, (n, 3))
        
        return outcome, treatment, covariates
    
    @pytest.fixture
    def observational_data(self):
        """Create sample observational data with confounding"""
        np.random.seed(42)
        
        n = 200
        # Confounder affects both treatment and outcome
        confounder = np.random.normal(0, 1, n)
        
        # Treatment assignment depends on confounder
        treatment_prob = 1 / (1 + np.exp(-(0.5 * confounder)))
        treatment = np.random.binomial(1, treatment_prob)
        
        # Outcome depends on both treatment and confounder
        noise = np.random.normal(0, 1, n)
        outcome = 2.0 + 0.3 * treatment + 0.4 * confounder + noise
        
        # Observable covariates (including the confounder)
        covariates = np.column_stack([confounder, np.random.normal(0, 1, (n, 2))])
        
        return outcome, treatment, covariates
    
    @pytest.fixture
    def did_data(self):
        """Create sample difference-in-differences data"""
        np.random.seed(42)
        
        n_units = 100
        n_periods = 2
        n = n_units * n_periods
        
        # Unit and time identifiers
        units = np.repeat(np.arange(n_units), n_periods)
        time_periods = np.tile([0, 1], n_units)
        
        # Treatment group (half of units)
        treatment_group = units < n_units // 2
        post_treatment = (treatment_group & (time_periods == 1)).astype(int)
        
        # Outcome with parallel trends in pre-period and treatment effect in post-period
        unit_effects = np.random.normal(0, 0.5, n_units)[units]
        time_trend = 0.2 * time_periods
        treatment_effect = 0.4 * post_treatment
        noise = np.random.normal(0, 0.3, n)
        
        outcome = 2.0 + unit_effects + time_trend + treatment_effect + noise
        
        return outcome, post_treatment, time_periods
    
    def test_initialization(self):
        """Test analyzer initialization"""
        analyzer = CausalInferenceAnalyzer(
            significance_level=0.01,
            minimum_effect_size=0.2,
            bootstrap_samples=500,
            enable_sensitivity_analysis=False
        )
        
        assert analyzer.significance_level == 0.01
        assert analyzer.minimum_effect_size == 0.2
        assert analyzer.bootstrap_samples == 500
        assert not analyzer.enable_sensitivity_analysis
    
    def test_data_validation(self, analyzer):
        """Test causal data validation"""
        # Valid data (increase sample size to meet minimum requirements)
        outcome = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        treatment = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        
        validated = analyzer._validate_causal_data(outcome, treatment, None, None, None)
        
        assert validated['n_total'] == 10
        assert validated['n_treated'] == 5
        assert validated['n_control'] == 5
        assert np.array_equal(validated['outcome'], outcome)
        assert np.array_equal(validated['treatment'], treatment)
    
    def test_data_validation_errors(self, analyzer):
        """Test data validation error handling"""
        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            analyzer._validate_causal_data(
                np.array([1, 2, 3]), 
                np.array([0, 1]), 
                None, None, None
            )
        
        # Insufficient sample size
        with pytest.raises(ValueError, match="Insufficient sample size"):
            analyzer._validate_causal_data(
                np.array([1, 2]), 
                np.array([0, 1]), 
                None, None, None
            )
        
        # Non-finite values (increase sample size)
        with pytest.raises(ValueError, match="finite"):
            analyzer._validate_causal_data(
                np.array([1, 2, np.inf, 4, 5, 6, 7, 8, 9, 10]), 
                np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0]), 
                None, None, None
            )
    
    def test_data_validation_with_covariates(self, analyzer):
        """Test data validation with covariates"""
        outcome = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        treatment = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        covariates = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
        
        validated = analyzer._validate_causal_data(outcome, treatment, covariates, None, None)
        
        assert 'covariates' in validated
        assert validated['n_covariates'] == 2
        assert validated['covariates'].shape == (10, 2)
    
    def test_overlap_assumption_basic(self, analyzer):
        """Test overlap assumption testing without covariates"""
        data = {
            'treatment': np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            'n_treated': 5,
            'n_control': 5
        }
        
        assumption = analyzer._test_overlap_assumption(data)
        
        assert assumption.name == "overlap"
        assert assumption.testable
        assert not assumption.violated
        assert assumption.test_result['adequate_overlap']
    
    def test_overlap_assumption_insufficient(self, analyzer):
        """Test overlap assumption with insufficient groups"""
        data = {
            'treatment': np.array([0, 0, 0, 1, 1]),
            'n_treated': 2,
            'n_control': 3
        }
        
        assumption = analyzer._test_overlap_assumption(data)
        
        assert assumption.violated  # n_treated < 5
        assert assumption.severity == "high"
        assert len(assumption.recommendations) > 0
    
    def test_balance_assumption(self, analyzer):
        """Test covariate balance assumption"""
        np.random.seed(42)
        
        # Balanced data
        covariates = np.random.normal(0, 1, (100, 2))
        treatment = np.random.binomial(1, 0.5, 100)
        
        data = {
            'covariates': covariates,
            'treatment': treatment
        }
        
        assumption = analyzer._test_balance_assumption(data)
        
        assert assumption.name == "balance"
        assert assumption.testable
        assert 'balance_tests' in assumption.test_result
        assert len(assumption.test_result['balance_tests']) == 2
    
    def test_balance_assumption_no_covariates(self, analyzer):
        """Test balance assumption without covariates"""
        data = {'treatment': np.array([0, 1, 0, 1])}
        
        assumption = analyzer._test_balance_assumption(data)
        
        assert assumption.name == "balance"
        assert not assumption.testable
        assert not assumption.violated
    
    def test_simple_difference_estimation(self, analyzer, randomized_experiment_data):
        """Test simple difference in means estimation"""
        outcome, treatment, _ = randomized_experiment_data
        
        data = {
            'outcome': outcome,
            'treatment': treatment,
            'n_total': len(outcome),
            'n_treated': np.sum(treatment),
            'n_control': len(outcome) - np.sum(treatment)
        }
        
        effect = analyzer._estimate_simple_difference(data)
        
        assert isinstance(effect, CausalEffect)
        assert effect.effect_name == "Average Treatment Effect"
        assert effect.statistical_significance or effect.p_value < 0.1  # Allow some randomness
        assert effect.sample_size == len(outcome)
        assert effect.confidence_interval[0] < effect.confidence_interval[1]
        assert abs(effect.point_estimate - 0.5) < 0.2  # Should be close to true effect
    
    def test_did_estimation(self, analyzer, did_data):
        """Test difference-in-differences estimation"""
        outcome, treatment, time_periods = did_data
        
        data = {
            'outcome': outcome,
            'treatment': treatment,
            'time_periods': time_periods,
            'n_total': len(outcome),
            'n_treated': np.sum(treatment),
            'n_control': len(outcome) - np.sum(treatment)
        }
        
        effect = analyzer._estimate_did_effect(data)
        
        assert isinstance(effect, CausalEffect)
        assert effect.effect_name == "Difference-in-Differences Estimate"
        assert effect.method == CausalMethod.DIFFERENCE_IN_DIFFERENCES
        assert effect.sample_size == len(outcome)
        assert 'regression_coefficients' in effect.metadata
        # True DiD effect is 0.4, should be reasonably close
        assert abs(effect.point_estimate - 0.4) < 0.2
    
    def test_psm_estimation(self, analyzer, observational_data):
        """Test propensity score matching estimation"""
        outcome, treatment, covariates = observational_data
        
        data = {
            'outcome': outcome,
            'treatment': treatment,
            'covariates': covariates,
            'n_total': len(outcome),
            'n_treated': np.sum(treatment),
            'n_control': len(outcome) - np.sum(treatment)
        }
        
        effect = analyzer._estimate_psm_effect(data)
        
        assert isinstance(effect, CausalEffect)
        assert effect.effect_name == "Propensity Score Matching Estimate (ATT)"
        assert effect.method == CausalMethod.PROPENSITY_SCORE_MATCHING
        assert 'n_matched_pairs' in effect.metadata
        assert 'mean_propensity_score_treated' in effect.metadata
    
    def test_doubly_robust_estimation(self, analyzer, observational_data):
        """Test doubly robust estimation"""
        outcome, treatment, covariates = observational_data
        
        data = {
            'outcome': outcome,
            'treatment': treatment,
            'covariates': covariates,
            'n_total': len(outcome),
            'n_treated': np.sum(treatment),
            'n_control': len(outcome) - np.sum(treatment)
        }
        
        effect = analyzer._estimate_doubly_robust_effect(data)
        
        assert isinstance(effect, CausalEffect)
        assert effect.effect_name == "Doubly Robust Estimate"
        assert effect.method == CausalMethod.DOUBLY_ROBUST
        assert 'regression_component' in effect.metadata
        assert 'ipw_component' in effect.metadata
    
    def test_iv_estimation(self, analyzer):
        """Test instrumental variables estimation"""
        np.random.seed(42)
        
        n = 100
        # Create instrument
        instrument = np.random.normal(0, 1, n)
        # Treatment depends on instrument
        treatment = (instrument + np.random.normal(0, 0.5, n) > 0).astype(int)
        # Outcome depends on treatment
        outcome = 2.0 + 0.3 * treatment + np.random.normal(0, 1, n)
        
        data = {
            'outcome': outcome,
            'treatment': treatment,
            'instruments': instrument.reshape(-1, 1),
            'n_total': len(outcome),
            'n_treated': np.sum(treatment),
            'n_control': len(outcome) - np.sum(treatment)
        }
        
        effect = analyzer._estimate_iv_effect(data)
        
        assert isinstance(effect, CausalEffect)
        assert effect.effect_name == "Instrumental Variables Estimate (2SLS)"
        assert effect.method == CausalMethod.INSTRUMENTAL_VARIABLES
        assert 'first_stage_f_stat' in effect.metadata
    
    def test_comprehensive_causal_analysis(self, analyzer, randomized_experiment_data):
        """Test complete causal inference analysis"""
        outcome, treatment, covariates = randomized_experiment_data
        
        result = analyzer.analyze_causal_effect(
            outcome_data=outcome,
            treatment_data=treatment,
            covariates=covariates,
            assignment_mechanism=TreatmentAssignment.RANDOMIZED,
            method=CausalMethod.DIFFERENCE_IN_DIFFERENCES
        )
        
        assert isinstance(result, CausalInferenceResult)
        assert result.analysis_id is not None
        assert result.timestamp is not None
        assert result.treatment_assignment == TreatmentAssignment.RANDOMIZED
        
        # Check primary effect
        assert isinstance(result.average_treatment_effect, CausalEffect)
        assert result.average_treatment_effect.sample_size == len(outcome)
        
        # Check assumptions testing
        assert isinstance(result.assumptions_tested, list)
        assert len(result.assumptions_tested) > 0
        
        # Check quality scores
        assert 0 <= result.internal_validity_score <= 1
        assert 0 <= result.external_validity_score <= 1
        assert 0 <= result.overall_quality_score <= 1
        assert 0 <= result.robustness_score <= 1
        
        # Check interpretations
        assert isinstance(result.causal_interpretation, str)
        assert isinstance(result.business_recommendations, list)
        assert isinstance(result.statistical_warnings, list)
    
    def test_sensitivity_analysis(self, analyzer, randomized_experiment_data):
        """Test sensitivity analysis for unmeasured confounding"""
        outcome, treatment, _ = randomized_experiment_data
        
        data = {
            'outcome': outcome,
            'treatment': treatment,
            'n_total': len(outcome),
            'n_treated': np.sum(treatment),
            'n_control': len(outcome) - np.sum(treatment)
        }
        
        sensitivity = analyzer._perform_sensitivity_analysis(data, CausalMethod.DIFFERENCE_IN_DIFFERENCES)
        
        assert isinstance(sensitivity, dict)
        assert 'baseline_effect' in sensitivity
        assert 'adjusted_effects' in sensitivity
        assert 'robust_to_small_bias' in sensitivity
        assert isinstance(sensitivity['robust_to_small_bias'], bool)
    
    def test_placebo_tests(self, analyzer, randomized_experiment_data):
        """Test placebo tests"""
        outcome, treatment, _ = randomized_experiment_data
        
        data = {
            'outcome': outcome,
            'treatment': treatment,
            'n_total': len(outcome),
            'n_treated': np.sum(treatment),
            'n_control': len(outcome) - np.sum(treatment)
        }
        
        placebo = analyzer._perform_placebo_tests(data, CausalMethod.DIFFERENCE_IN_DIFFERENCES)
        
        assert isinstance(placebo, dict)
        assert 'placebo_tests' in placebo
        assert 'overall_passes' in placebo
        assert 'original_effect' in placebo
        assert isinstance(placebo['placebo_tests'], list)
        assert len(placebo['placebo_tests']) > 0
    
    def test_confounding_assessment(self, analyzer, observational_data):
        """Test confounding assessment"""
        outcome, treatment, _ = observational_data
        
        assessment = analyzer._assess_confounding(
            {'outcome': outcome, 'treatment': treatment},
            TreatmentAssignment.OBSERVATIONAL
        )
        
        assert isinstance(assessment, dict)
        assert assessment['assignment_mechanism'] == 'observational'
        assert assessment['confounding_risk'] == 'high'
        assert 'treated_outcome_mean' in assessment
        assert 'control_outcome_mean' in assessment
        assert 'treatment_prevalence' in assessment
    
    def test_covariate_balance_assessment(self, analyzer, observational_data):
        """Test covariate balance assessment"""
        outcome, treatment, covariates = observational_data
        
        data = {
            'treatment': treatment,
            'covariates': covariates
        }
        
        balance = analyzer._assess_covariate_balance(data)
        
        assert isinstance(balance, dict)
        assert 'balance_results' in balance
        assert 'n_covariates' in balance
        assert 'overall_balanced' in balance
        assert balance['n_covariates'] == covariates.shape[1]
        assert isinstance(balance['balance_results'], list)
    
    def test_score_calculations(self, analyzer, randomized_experiment_data):
        """Test quality score calculations"""
        outcome, treatment, covariates = randomized_experiment_data
        
        # Create mock objects for testing
        primary_effect = Mock()
        primary_effect.point_estimate = 0.5
        primary_effect.statistical_significance = True
        
        sensitivity_results = {'robust_to_small_bias': True}
        placebo_results = {'overall_passes': True}
        
        robustness_score = analyzer._calculate_robustness_score(
            primary_effect, sensitivity_results, placebo_results
        )
        
        assert 0 <= robustness_score <= 1
        assert robustness_score > 0.7  # Should be high for good results
        
        # Test internal validity score
        assumptions = [
            Mock(violated=False, severity="low"),
            Mock(violated=True, severity="medium")
        ]
        confounding_assessment = {'confounding_risk': 'low'}
        
        internal_validity = analyzer._calculate_internal_validity_score(
            assumptions, confounding_assessment
        )
        
        assert 0 <= internal_validity <= 1
        
        # Test external validity score
        data = {
            'n_total': len(outcome),
            'n_treated': np.sum(treatment),
            'n_control': len(outcome) - np.sum(treatment)
        }
        
        external_validity = analyzer._calculate_external_validity_score(
            data, CausalMethod.DIFFERENCE_IN_DIFFERENCES
        )
        
        assert 0 <= external_validity <= 1
    
    def test_interpretation_generation(self, analyzer):
        """Test causal interpretation generation"""
        # Mock effect with strong evidence
        strong_effect = Mock()
        strong_effect.statistical_significance = True
        strong_effect.practical_significance = True
        strong_effect.effect_size_interpretation = "large effect"
        strong_effect.point_estimate = 0.8
        
        # No violated assumptions
        good_assumptions = [Mock(violated=False, severity="low")]
        
        interpretation = analyzer._generate_causal_interpretation(strong_effect, good_assumptions)
        
        assert isinstance(interpretation, str)
        assert "strong evidence" in interpretation.lower()
        assert "causal effect" in interpretation.lower()
        
        # Test with violated assumptions
        bad_assumptions = [Mock(violated=True, severity="high")]
        weak_interpretation = analyzer._generate_causal_interpretation(strong_effect, bad_assumptions)
        
        assert "suggestive evidence" in weak_interpretation.lower() or "violated" in weak_interpretation.lower()
    
    def test_business_recommendations(self, analyzer):
        """Test business recommendations generation"""
        # Strong effect with good robustness
        strong_effect = Mock()
        strong_effect.statistical_significance = True
        strong_effect.practical_significance = True
        
        good_assumptions = [Mock(violated=False, severity="low")]
        
        recommendations = analyzer._generate_business_recommendations(
            strong_effect, good_assumptions, robustness_score=0.8
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("STRONG RECOMMENDATION" in rec for rec in recommendations)
        assert any("Deploy" in rec for rec in recommendations)
        
        # Weak effect
        weak_effect = Mock()
        weak_effect.statistical_significance = False
        weak_effect.practical_significance = False
        
        weak_recommendations = analyzer._generate_business_recommendations(
            weak_effect, good_assumptions, robustness_score=0.3
        )
        
        assert any("NOT RECOMMENDED" in rec for rec in weak_recommendations)
    
    def test_statistical_warnings(self, analyzer):
        """Test statistical warnings generation"""
        # Critical assumption violations
        bad_assumptions = [
            Mock(violated=True, severity="high", name="critical_assumption", 
                 description="This is critical")
        ]
        
        # High confounding risk
        confounding_assessment = {
            'confounding_risk': 'high',
            'extreme_difference_detected': True,
            'treatment_prevalence': 0.05
        }
        
        warnings = analyzer._generate_statistical_warnings(bad_assumptions, confounding_assessment)
        
        assert isinstance(warnings, list)
        assert len(warnings) > 0
        assert any("CRITICAL" in warning for warning in warnings)
        assert any("HIGH CONFOUNDING RISK" in warning for warning in warnings)
        assert any("EXTREME DIFFERENCES" in warning for warning in warnings)
        assert any("SMALL TREATMENT GROUP" in warning for warning in warnings)
    
    def test_quick_causal_analysis_utility(self, randomized_experiment_data):
        """Test quick causal analysis utility function"""
        outcome, treatment, covariates = randomized_experiment_data
        
        result = quick_causal_analysis(
            outcome_data=outcome.tolist(),
            treatment_data=treatment.tolist(),
            covariates=covariates.tolist(),
            method="simple_difference"
        )
        
        assert isinstance(result, dict)
        assert 'causal_effect' in result
        assert 'confidence_interval' in result
        assert 'p_value' in result
        assert 'statistical_significance' in result
        assert 'practical_significance' in result
        assert 'causal_interpretation' in result
        assert 'business_recommendations' in result
        assert 'robustness_score' in result
        assert 'overall_quality' in result
        
        assert isinstance(result['statistical_significance'], bool)
        assert isinstance(result['practical_significance'], bool)
        assert isinstance(result['business_recommendations'], list)
        assert 0 <= result['robustness_score'] <= 1
        assert 0 <= result['overall_quality'] <= 1
    
    def test_error_handling(self, analyzer):
        """Test error handling in causal analysis"""
        # Test with invalid method
        outcome = np.random.normal(0, 1, 50)
        treatment = np.random.binomial(1, 0.5, 50)
        
        # Should fall back to simple difference if method fails
        result = analyzer.analyze_causal_effect(
            outcome_data=outcome,
            treatment_data=treatment,
            method=CausalMethod.INSTRUMENTAL_VARIABLES  # Without instruments
        )
        
        # Should complete without error (may fall back to simple method)
        assert isinstance(result, CausalInferenceResult)
        assert result.average_treatment_effect is not None
        
        # Test quick analysis with error
        error_result = quick_causal_analysis(
            outcome_data=[],  # Empty data
            treatment_data=[],
            method="invalid_method"
        )
        
        assert 'error' in error_result
    
    def test_edge_cases(self, analyzer):
        """Test edge cases in causal analysis"""
        # Perfect separation (all treatment or all control)
        outcome = np.array([1, 2, 3, 4, 5])
        treatment_all_control = np.array([0, 0, 0, 0, 0])
        
        # Should handle gracefully (though not meaningful)
        try:
            result = analyzer.analyze_causal_effect(
                outcome_data=outcome,
                treatment_data=treatment_all_control
            )
            # If it completes, check it's a valid result
            assert isinstance(result, CausalInferenceResult)
        except ValueError:
            # It's also acceptable to raise an error for this case
            pass
        
        # Identical outcomes
        identical_outcome = np.array([5, 5, 5, 5, 5, 5])
        balanced_treatment = np.array([0, 0, 0, 1, 1, 1])
        
        result_identical = analyzer.analyze_causal_effect(
            outcome_data=identical_outcome,
            treatment_data=balanced_treatment
        )
        
        # Should detect no effect
        assert abs(result_identical.average_treatment_effect.point_estimate) < 0.001
        assert not result_identical.average_treatment_effect.practical_significance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])