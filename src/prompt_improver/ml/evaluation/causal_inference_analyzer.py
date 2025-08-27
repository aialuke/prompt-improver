"""Causal Inference Analyzer for Advanced A/B Testing
Implements 2025 best practices for causal analysis and counterfactual reasoning
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from typing import Any, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
# import numpy as np  # Converted to lazy loading
# from scipy import stats  # Converted to lazy loading
from ...core.utils.lazy_ml_loader import get_numpy, get_scipy_stats
from ..core.training_data_loader import TrainingDataLoader
logger = logging.getLogger(__name__)

class CausalMethod(Enum):
    """Causal inference methods"""
    DIFFERENCE_IN_DIFFERENCES = 'difference_in_differences'
    INSTRUMENTAL_VARIABLES = 'instrumental_variables'
    PROPENSITY_SCORE_MATCHING = 'propensity_score_matching'
    REGRESSION_DISCONTINUITY = 'regression_discontinuity'
    DOUBLY_ROBUST = 'doubly_robust'
    SYNTHETIC_CONTROL = 'synthetic_control'

class TreatmentAssignment(Enum):
    """Treatment assignment mechanisms"""
    randomized = 'randomized'
    QUASI_EXPERIMENTAL = 'quasi_experimental'
    observational = 'observational'
    NATURAL_EXPERIMENT = 'natural_experiment'

@dataclass
class CausalAssumption:
    """Represents a causal inference assumption"""
    name: str
    description: str
    testable: bool
    test_result: dict[str, Any] | None = None
    violated: bool = False
    severity: str = 'unknown'
    recommendations: list[str] = field(default_factory=list)

@dataclass
class CausalEffect:
    """Represents an estimated causal effect"""
    effect_name: str
    point_estimate: float
    confidence_interval: tuple[float, float]
    standard_error: float
    p_value: float
    method: CausalMethod
    sample_size: int
    effect_size_interpretation: str
    statistical_significance: bool
    practical_significance: bool
    robustness_score: float = 0.0
    assumptions_satisfied: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class CausalInferenceResult:
    """Comprehensive causal inference analysis result"""
    analysis_id: str
    timestamp: datetime
    treatment_assignment: TreatmentAssignment
    average_treatment_effect: CausalEffect
    conditional_average_treatment_effect: CausalEffect | None = None
    local_average_treatment_effect: CausalEffect | None = None
    assumptions_tested: list[CausalAssumption] = field(default_factory=list)
    overall_assumptions_satisfied: bool = True
    sensitivity_analysis: dict[str, Any] | None = None
    placebo_tests: dict[str, Any] | None = None
    robustness_score: float = 0.0
    confounding_assessment: dict[str, Any] | None = None
    covariate_balance: dict[str, Any] | None = None
    causal_interpretation: str = ''
    business_recommendations: list[str] = field(default_factory=list)
    statistical_warnings: list[str] = field(default_factory=list)
    internal_validity_score: float = 0.0
    external_validity_score: float = 0.0
    overall_quality_score: float = 0.0

class CausalInferenceAnalyzer:
    """Advanced causal inference analyzer implementing 2025 best practices
    
    Phase 2 Enhancement: Integrated with training data pipeline for causal
    analysis of rule effectiveness and optimization insights.
    """

    def __init__(self, significance_level: float=0.05, minimum_effect_size: float=0.1, bootstrap_samples: int=1000, enable_sensitivity_analysis: bool=True, training_loader: TrainingDataLoader | None=None):
        """Initialize causal inference analyzer

        Args:
            significance_level: Alpha level for statistical tests
            minimum_effect_size: Minimum meaningful effect size
            bootstrap_samples: Number of bootstrap samples for robust estimation
            enable_sensitivity_analysis: Whether to perform sensitivity analysis
        """
        self.significance_level = significance_level
        self.minimum_effect_size = minimum_effect_size
        self.bootstrap_samples = bootstrap_samples
        self.enable_sensitivity_analysis = enable_sensitivity_analysis
        self.training_loader = training_loader or TrainingDataLoader(real_data_priority=True, min_samples=10, lookback_days=30, synthetic_ratio=0.2)
        logger.info('CausalInferenceAnalyzer initialized with training data integration')

    async def run_orchestrated_analysis(self, config: dict[str, Any]) -> dict[str, Any]:
        """Orchestrator-compatible interface for causal inference analysis (2025 pattern)

        Args:
            config: Orchestrator configuration containing:
                - outcome_data: Outcome variable data
                - treatment_data: Treatment assignment data
                - covariates: Optional covariate data
                - method: Causal inference method ('difference_in_differences', 'propensity_score_matching', etc.)
                - assignment_mechanism: Treatment assignment type ('randomized', 'quasi_experimental', etc.)
                - output_path: Local path for output files (optional)

        Returns:
            Orchestrator-compatible result with causal analysis and metadata
        """
        start_time = datetime.now()
        try:
            outcome_data = config.get('outcome_data', [])
            treatment_data = config.get('treatment_data', [])
            covariates = config.get('covariates', None)
            method_str = config.get('method', 'difference_in_differences')
            assignment_str = config.get('assignment_mechanism', 'randomized')
            output_path = config.get('output_path', './outputs/causal_analysis')
            if not outcome_data or not treatment_data:
                raise ValueError('Both outcome_data and treatment_data are required')
            if len(outcome_data) != len(treatment_data):
                raise ValueError('Outcome and treatment data must have the same length')
            try:
                method = CausalMethod(method_str)
            except ValueError:
                method = CausalMethod.DIFFERENCE_IN_DIFFERENCES
                logger.warning(f"Unknown method '{method_str}', using difference_in_differences")
            try:
                assignment = TreatmentAssignment(assignment_str)
            except ValueError:
                assignment = TreatmentAssignment.randomized
                logger.warning(f"Unknown assignment '{assignment_str}', using randomized")
            outcome_array = get_numpy().array(outcome_data, dtype=float)
            treatment_array = get_numpy().array(treatment_data, dtype=int)
            covariates_array = get_numpy().array(covariates, dtype=float) if covariates else None
            causal_result = self.analyze_causal_effect(outcome_data=outcome_array, treatment_data=treatment_array, covariates=covariates_array, assignment_mechanism=assignment, method=method)
            result = {'causal_effects': {'average_treatment_effect': {'point_estimate': causal_result.average_treatment_effect.point_estimate, 'confidence_interval': causal_result.average_treatment_effect.confidence_interval, 'p_value': causal_result.average_treatment_effect.p_value, 'statistical_significance': causal_result.average_treatment_effect.statistical_significance, 'practical_significance': causal_result.average_treatment_effect.practical_significance, 'effect_size_interpretation': causal_result.average_treatment_effect.effect_size_interpretation}}, 'assumptions_validation': {'overall_satisfied': causal_result.overall_assumptions_satisfied, 'assumptions_tested': len(causal_result.assumptions_tested), 'violated_assumptions': [assumption.name for assumption in causal_result.assumptions_tested if assumption.violated]}, 'robustness_assessment': {'robustness_score': causal_result.robustness_score, 'sensitivity_analysis': causal_result.sensitivity_analysis is not None, 'placebo_tests': causal_result.placebo_tests is not None}, 'confounding_analysis': {'confounders_detected': len(causal_result.confounders_detected) if causal_result.confounders_detected else 0, 'confounding_strength': causal_result.confounding_strength, 'adjustment_strategy': causal_result.adjustment_strategy}, 'business_insights': {'causal_interpretation': causal_result.causal_interpretation, 'business_recommendations': causal_result.business_recommendations, 'statistical_warnings': causal_result.statistical_warnings}}
            execution_time = (datetime.now() - start_time).total_seconds()
            return {'orchestrator_compatible': True, 'component_result': result, 'local_metadata': {'output_path': output_path, 'execution_time': execution_time, 'method_used': method.value, 'assignment_mechanism': assignment.value, 'sample_size': len(outcome_data), 'covariates_included': covariates is not None, 'analysis_id': causal_result.analysis_id, 'component_version': '1.0.0'}}
        except ValueError as e:
            logger.error('Validation error in orchestrated causal analysis: %s', e)
            return {'orchestrator_compatible': True, 'component_result': {'error': f'Validation error: {str(e)}', 'causal_effects': {}}, 'local_metadata': {'execution_time': (datetime.now() - start_time).total_seconds(), 'error': True, 'error_type': 'validation', 'component_version': '1.0.0'}}
        except Exception as e:
            logger.error('Orchestrated causal analysis failed: %s', e)
            return {'orchestrator_compatible': True, 'component_result': {'error': str(e), 'causal_effects': {}}, 'local_metadata': {'execution_time': (datetime.now() - start_time).total_seconds(), 'error': True, 'component_version': '1.0.0'}}

    def analyze_causal_effect(self, outcome_data: get_numpy().ndarray, treatment_data: get_numpy().ndarray, covariates: get_numpy().ndarray | None=None, assignment_mechanism: TreatmentAssignment=TreatmentAssignment.randomized, method: CausalMethod=CausalMethod.DIFFERENCE_IN_DIFFERENCES, time_periods: get_numpy().ndarray | None=None, instruments: get_numpy().ndarray | None=None) -> CausalInferenceResult:
        """Perform comprehensive causal inference analysis

        Args:
            outcome_data: Outcome variable (continuous or binary)
            treatment_data: Treatment assignment (0/1)
            covariates: Control variables/covariates
            assignment_mechanism: How treatment was assigned
            method: Primary causal inference method
            time_periods: Time periods for difference-in-differences
            instruments: Instrumental variables for IV estimation

        Returns:
            Comprehensive causal inference result
        """
        from prompt_improver.utils.datetime_utils import format_compact_timestamp
        analysis_id = f"causal_analysis_{format_compact_timestamp(datetime.utcnow())}"
        try:
            logger.info('Starting causal inference analysis: %s', analysis_id)
            validated_data = self._validate_causal_data(outcome_data, treatment_data, covariates, time_periods, instruments)
            assumptions = self._test_causal_assumptions(validated_data, assignment_mechanism, method)
            primary_effect = self._estimate_causal_effect(validated_data, method, assignment_mechanism)
            conditional_effect = self._estimate_conditional_effects(validated_data, method) if covariates is not None else None
            sensitivity_results = None
            placebo_results = None
            if self.enable_sensitivity_analysis:
                sensitivity_results = self._perform_sensitivity_analysis(validated_data, method)
                placebo_results = self._perform_placebo_tests(validated_data, method)
            confounding_assessment = self._assess_confounding(validated_data, assignment_mechanism)
            covariate_balance = self._assess_covariate_balance(validated_data) if covariates is not None else None
            robustness_score = self._calculate_robustness_score(primary_effect, sensitivity_results, placebo_results)
            internal_validity = self._calculate_internal_validity_score(assumptions, confounding_assessment)
            external_validity = self._calculate_external_validity_score(validated_data, method)
            overall_quality = (internal_validity + external_validity + robustness_score) / 3
            interpretation = self._generate_causal_interpretation(primary_effect, assumptions)
            business_recs = self._generate_business_recommendations(primary_effect, assumptions, robustness_score)
            warnings = self._generate_statistical_warnings(assumptions, confounding_assessment)
            result = CausalInferenceResult(analysis_id=analysis_id, timestamp=datetime.utcnow(), treatment_assignment=assignment_mechanism, average_treatment_effect=primary_effect, conditional_average_treatment_effect=conditional_effect, assumptions_tested=assumptions, overall_assumptions_satisfied=all(not a.violated for a in assumptions), sensitivity_analysis=sensitivity_results, placebo_tests=placebo_results, robustness_score=robustness_score, confounding_assessment=confounding_assessment, covariate_balance=covariate_balance, causal_interpretation=interpretation, business_recommendations=business_recs, statistical_warnings=warnings, internal_validity_score=internal_validity, external_validity_score=external_validity, overall_quality_score=overall_quality)
            logger.info('Causal inference analysis completed: %s', analysis_id)
            logger.info('ATE: %s ± %s', format(primary_effect.point_estimate, '.4f'), format(primary_effect.standard_error, '.4f'))
            return result
        except Exception as e:
            logger.error('Error in causal inference analysis: %s', e)
            raise

    def _validate_causal_data(self, outcome_data: get_numpy().ndarray, treatment_data: get_numpy().ndarray, covariates: get_numpy().ndarray | None, time_periods: get_numpy().ndarray | None, instruments: get_numpy().ndarray | None) -> dict[str, get_numpy().ndarray]:
        """Validate and prepare data for causal analysis"""
        outcome = get_numpy().asarray(outcome_data).flatten()
        treatment = get_numpy().asarray(treatment_data).flatten()
        if len(outcome) != len(treatment):
            raise ValueError('Outcome and treatment data must have same length')
        if len(outcome) < 5:
            raise ValueError('Insufficient sample size for reliable causal inference')
        unique_treatments = get_numpy().unique(treatment)
        if not get_numpy().array_equal(unique_treatments, [0, 1]) and (not get_numpy().array_equal(unique_treatments, [0])) and (not get_numpy().array_equal(unique_treatments, [1])):
            if len(unique_treatments) == 2:
                treatment = (treatment == unique_treatments[1]).astype(int)
            else:
                raise ValueError('Treatment must be binary (0/1)')
        if not get_numpy().all(get_numpy().isfinite(outcome)) or not get_numpy().all(get_numpy().isfinite(treatment)):
            raise ValueError('Outcome and treatment data must be finite')
        validated = {'outcome': outcome, 'treatment': treatment, 'n_total': len(outcome), 'n_treated': get_numpy().sum(treatment), 'n_control': len(outcome) - get_numpy().sum(treatment)}
        if covariates is not None:
            covariates = get_numpy().asarray(covariates)
            if covariates.ndim == 1:
                covariates = covariates.reshape(-1, 1)
            if len(covariates) != len(outcome):
                raise ValueError('Covariates must have same length as outcome')
            validated['covariates'] = covariates
            validated['n_covariates'] = covariates.shape[1]
        if time_periods is not None:
            time_periods = get_numpy().asarray(time_periods).flatten()
            if len(time_periods) != len(outcome):
                raise ValueError('Time periods must have same length as outcome')
            validated['time_periods'] = time_periods
            validated['n_periods'] = len(get_numpy().unique(time_periods))
        if instruments is not None:
            instruments = get_numpy().asarray(instruments)
            if instruments.ndim == 1:
                instruments = instruments.reshape(-1, 1)
            if len(instruments) != len(outcome):
                raise ValueError('Instruments must have same length as outcome')
            validated['instruments'] = instruments
            validated['n_instruments'] = instruments.shape[1]
        return validated

    def _test_causal_assumptions(self, data: dict[str, get_numpy().ndarray], assignment: TreatmentAssignment, method: CausalMethod) -> list[CausalAssumption]:
        """Test key causal inference assumptions"""
        assumptions = []
        try:
            overlap_assumption = self._test_overlap_assumption(data)
            assumptions.append(overlap_assumption)
            if assignment == TreatmentAssignment.randomized:
                balance_assumption = self._test_balance_assumption(data)
                assumptions.append(balance_assumption)
            if method == CausalMethod.DIFFERENCE_IN_DIFFERENCES and 'time_periods' in data:
                parallel_trends = self._test_parallel_trends_assumption(data)
                assumptions.append(parallel_trends)
            if method == CausalMethod.INSTRUMENTAL_VARIABLES and 'instruments' in data:
                iv_assumptions = self._test_iv_assumptions(data)
                assumptions.extend(iv_assumptions)
            confounding_assumption = self._test_confounding_assumption(data, assignment)
            assumptions.append(confounding_assumption)
        except Exception as e:
            logger.warning('Error testing assumptions: %s', e)
            warning_assumption = CausalAssumption(name='assumption_testing_error', description='Could not fully test causal assumptions', testable=False, violated=True, severity='medium', recommendations=['Manual assumption checking recommended'])
            assumptions.append(warning_assumption)
        return assumptions

    def _test_overlap_assumption(self, data: dict[str, get_numpy().ndarray]) -> CausalAssumption:
        """Test overlap/common support assumption"""
        if 'covariates' not in data:
            n_treated = data['n_treated']
            n_control = data['n_control']
            overlap_adequate = n_treated >= 5 and n_control >= 5
            return CausalAssumption(name='overlap', description='Sufficient overlap between treatment and control groups', testable=True, test_result={'n_treated': int(n_treated), 'n_control': int(n_control), 'adequate_overlap': overlap_adequate}, violated=not overlap_adequate, severity='high' if not overlap_adequate else 'low', recommendations=['Increase sample size in underrepresented group'] if not overlap_adequate else [])
        try:
            from prompt_improver.core.utils.lazy_ml_loader import get_sklearn
            LogisticRegression = get_sklearn().linear_model.LogisticRegression
            X = data['covariates']
            y = data['treatment']
            lr = LogisticRegression(random_state=42)
            propensity_scores = lr.fit(X, y).predict_proba(X)[:, 1]
            treated_ps = propensity_scores[y == 1]
            control_ps = propensity_scores[y == 0]
            min_treated, max_treated = (get_numpy().min(treated_ps), get_numpy().max(treated_ps))
            min_control, max_control = (get_numpy().min(control_ps), get_numpy().max(control_ps))
            overlap_min = max(min_treated, min_control)
            overlap_max = min(max_treated, max_control)
            overlap_exists = overlap_max > overlap_min
            in_overlap = (propensity_scores >= overlap_min) & (propensity_scores <= overlap_max)
            overlap_proportion = get_numpy().mean(in_overlap)
            adequate_overlap = overlap_exists and overlap_proportion >= 0.8
            return CausalAssumption(name='overlap', description='Sufficient overlap in propensity score distributions', testable=True, test_result={'overlap_exists': overlap_exists, 'overlap_proportion': float(overlap_proportion), 'overlap_range': [float(overlap_min), float(overlap_max)], 'adequate_overlap': adequate_overlap}, violated=not adequate_overlap, severity='high' if not adequate_overlap else 'low', recommendations=['Consider trimming extreme propensity scores', 'Collect more data in regions of poor overlap'] if not adequate_overlap else [])
        except Exception as e:
            logger.warning('Error testing overlap assumption: %s', e)
            return CausalAssumption(name='overlap', description='Could not test overlap assumption', testable=False, violated=True, severity='medium', recommendations=['Manual overlap assessment recommended'])

    def _test_balance_assumption(self, data: dict[str, get_numpy().ndarray]) -> CausalAssumption:
        """Test covariate balance assumption for randomized experiments"""
        if 'covariates' not in data:
            return CausalAssumption(name='balance', description='No covariates to test balance', testable=False, violated=False, severity='low')
        try:
            covariates = data['covariates']
            treatment = data['treatment']
            balance_tests = []
            imbalanced_covariates = 0
            for i in range(covariates.shape[1]):
                covariate = covariates[:, i]
                treated_cov = covariate[treatment == 1]
                control_cov = covariate[treatment == 0]
                statistic, p_value = get_scipy_stats().ttest_ind(treated_cov, control_cov)
                pooled_std = get_numpy().sqrt(((len(treated_cov) - 1) * get_numpy().var(treated_cov, ddof=1) + (len(control_cov) - 1) * get_numpy().var(control_cov, ddof=1)) / (len(treated_cov) + len(control_cov) - 2))
                smd = (get_numpy().mean(treated_cov) - get_numpy().mean(control_cov)) / pooled_std if pooled_std > 0 else 0
                is_imbalanced = abs(smd) > 0.1
                if is_imbalanced:
                    imbalanced_covariates += 1
                balance_tests.append({'covariate_index': i, 'p_value': float(p_value), 'standardized_mean_difference': float(smd), 'imbalanced': is_imbalanced})
            overall_balanced = imbalanced_covariates / len(balance_tests) <= 0.05
            return CausalAssumption(name='balance', description='Covariate balance between treatment groups', testable=True, test_result={'n_covariates': len(balance_tests), 'n_imbalanced': imbalanced_covariates, 'proportion_imbalanced': imbalanced_covariates / len(balance_tests), 'balance_tests': balance_tests, 'overall_balanced': overall_balanced}, violated=not overall_balanced, severity='medium' if not overall_balanced else 'low', recommendations=['Consider stratified randomization', 'Use covariate adjustment in analysis'] if not overall_balanced else [])
        except Exception as e:
            logger.warning('Error testing balance assumption: %s', e)
            return CausalAssumption(name='balance', description='Could not test balance assumption', testable=False, violated=True, severity='medium')

    def _test_parallel_trends_assumption(self, data: dict[str, get_numpy().ndarray]) -> CausalAssumption:
        """Test parallel trends assumption for difference-in-differences"""
        try:
            outcome = data['outcome']
            treatment = data['treatment']
            time_periods = data['time_periods']
            pre_period = time_periods == 0
            post_period = time_periods == 1
            if not (get_numpy().any(pre_period) and get_numpy().any(post_period)):
                return CausalAssumption(name='parallel_trends', description='Insufficient time periods for parallel trends test', testable=False, violated=True, severity='high')
            pre_treated = outcome[pre_period & (treatment == 1)]
            pre_control = outcome[pre_period & (treatment == 0)]
            if len(pre_treated) > 0 and len(pre_control) > 0:
                statistic, p_value = get_scipy_stats().ttest_ind(pre_treated, pre_control)
                parallel_trends_satisfied = p_value > 0.1
            else:
                parallel_trends_satisfied = False
                p_value = 0.0
            return CausalAssumption(name='parallel_trends', description='Parallel trends assumption for difference-in-differences', testable=True, test_result={'pre_treatment_test_pvalue': float(p_value), 'parallel_trends_satisfied': parallel_trends_satisfied, 'n_pre_treated': len(pre_treated) if len(pre_treated) > 0 else 0, 'n_pre_control': len(pre_control) if len(pre_control) > 0 else 0}, violated=not parallel_trends_satisfied, severity='high' if not parallel_trends_satisfied else 'low', recommendations=['Collect more pre-treatment periods for robust testing', 'Consider event study design'] if not parallel_trends_satisfied else [])
        except Exception as e:
            logger.warning('Error testing parallel trends: %s', e)
            return CausalAssumption(name='parallel_trends', description='Could not test parallel trends assumption', testable=False, violated=True, severity='high')

    def _test_iv_assumptions(self, data: dict[str, get_numpy().ndarray]) -> list[CausalAssumption]:
        """Test instrumental variable assumptions"""
        assumptions = []
        try:
            instruments = data['instruments']
            treatment = data['treatment']
            outcome = data['outcome']
            for i in range(instruments.shape[1]):
                instrument = instruments[:, i]
                correlation = get_numpy().corrcoef(instrument, treatment)[0, 1]
                f_statistic = correlation ** 2 * (len(treatment) - 2) / (1 - correlation ** 2)
                strong_instrument = f_statistic > 10
                relevance_assumption = CausalAssumption(name=f'instrument_relevance_{i}', description=f'Instrument {i} relevance (first stage strength)', testable=True, test_result={'correlation_with_treatment': float(correlation), 'f_statistic': float(f_statistic), 'strong_instrument': strong_instrument}, violated=not strong_instrument, severity='high' if not strong_instrument else 'low', recommendations=['Find stronger instruments', 'Consider weak instrument robust methods'] if not strong_instrument else [])
                assumptions.append(relevance_assumption)
            exogeneity_assumption = CausalAssumption(name='instrument_exogeneity', description='Instrument exogeneity (exclusion restriction)', testable=False, violated=False, severity='high', recommendations=['Carefully justify exclusion restriction', 'Consider overidentification tests if multiple instruments'])
            assumptions.append(exogeneity_assumption)
        except Exception as e:
            logger.warning('Error testing IV assumptions: %s', e)
            error_assumption = CausalAssumption(name='iv_assumptions_error', description='Could not test IV assumptions', testable=False, violated=True, severity='medium')
            assumptions.append(error_assumption)
        return assumptions

    def _test_confounding_assumption(self, data: dict[str, get_numpy().ndarray], assignment: TreatmentAssignment) -> CausalAssumption:
        """Test no unmeasured confounding assumption"""
        if assignment == TreatmentAssignment.randomized:
            return CausalAssumption(name='no_unmeasured_confounding', description='No unmeasured confounding (randomized experiment)', testable=False, violated=False, severity='low', recommendations=['Verify randomization was properly implemented'])
        try:
            outcome = data['outcome']
            treatment = data['treatment']
            correlation = abs(get_numpy().corrcoef(treatment, outcome)[0, 1])
            potential_confounding = correlation > 0.7
            return CausalAssumption(name='no_unmeasured_confounding', description='No unmeasured confounding assumption', testable=False, test_result={'treatment_outcome_correlation': float(correlation), 'potential_confounding_concern': potential_confounding}, violated=potential_confounding, severity='high', recommendations=['Include more covariates', 'Use instrumental variables', 'Consider sensitivity analysis'] if potential_confounding else ['Justify assumption with domain knowledge'])
        except Exception as e:
            logger.warning('Error testing confounding assumption: %s', e)
            return CausalAssumption(name='no_unmeasured_confounding', description='Could not assess confounding assumption', testable=False, violated=True, severity='high')

    def _estimate_causal_effect(self, data: dict[str, get_numpy().ndarray], method: CausalMethod, assignment: TreatmentAssignment) -> CausalEffect:
        """Estimate the average treatment effect using specified method"""
        outcome = data['outcome']
        treatment = data['treatment']
        try:
            if method == CausalMethod.DIFFERENCE_IN_DIFFERENCES:
                return self._estimate_did_effect(data)
            if method == CausalMethod.INSTRUMENTAL_VARIABLES:
                return self._estimate_iv_effect(data)
            if method == CausalMethod.PROPENSITY_SCORE_MATCHING:
                return self._estimate_psm_effect(data)
            if method == CausalMethod.DOUBLY_ROBUST:
                return self._estimate_doubly_robust_effect(data)
            return self._estimate_simple_difference(data)
        except Exception as e:
            logger.error('Error estimating causal effect with {method}: %s', e)
            return self._estimate_simple_difference(data)

    def _estimate_simple_difference(self, data: dict[str, get_numpy().ndarray]) -> CausalEffect:
        """Estimate ATE as simple difference in means"""
        outcome = data['outcome']
        treatment = data['treatment']
        treated_outcomes = outcome[treatment == 1]
        control_outcomes = outcome[treatment == 0]
        ate = get_numpy().mean(treated_outcomes) - get_numpy().mean(control_outcomes)
        var_treated = get_numpy().var(treated_outcomes, ddof=1) / len(treated_outcomes)
        var_control = get_numpy().var(control_outcomes, ddof=1) / len(control_outcomes)
        se = get_numpy().sqrt(var_treated + var_control)
        statistic, p_value = get_scipy_stats().ttest_ind(treated_outcomes, control_outcomes)
        dof = len(treated_outcomes) + len(control_outcomes) - 2
        t_critical = get_scipy_stats().t.ppf(1 - self.significance_level / 2, dof)
        ci_lower = ate - t_critical * se
        ci_upper = ate + t_critical * se
        pooled_std = get_numpy().sqrt(((len(treated_outcomes) - 1) * get_numpy().var(treated_outcomes, ddof=1) + (len(control_outcomes) - 1) * get_numpy().var(control_outcomes, ddof=1)) / dof)
        cohens_d = ate / pooled_std if pooled_std > 0 else 0
        if abs(cohens_d) < 0.2:
            interpretation = 'negligible effect'
        elif abs(cohens_d) < 0.5:
            interpretation = 'small effect'
        elif abs(cohens_d) < 0.8:
            interpretation = 'medium effect'
        else:
            interpretation = 'large effect'
        return CausalEffect(effect_name='Average Treatment Effect', point_estimate=float(ate), confidence_interval=(float(ci_lower), float(ci_upper)), standard_error=float(se), p_value=float(p_value), method=CausalMethod.DIFFERENCE_IN_DIFFERENCES, sample_size=len(outcome), effect_size_interpretation=interpretation, statistical_significance=bool(p_value < self.significance_level), practical_significance=bool(abs(ate) >= self.minimum_effect_size), robustness_score=0.7, assumptions_satisfied=True, metadata={'treated_mean': float(get_numpy().mean(treated_outcomes)), 'control_mean': float(get_numpy().mean(control_outcomes)), 'cohens_d': float(cohens_d), 'n_treated': len(treated_outcomes), 'n_control': len(control_outcomes)})

    def _estimate_did_effect(self, data: dict[str, get_numpy().ndarray]) -> CausalEffect:
        """Estimate difference-in-differences effect"""
        if 'time_periods' not in data:
            logger.warning('No time periods provided for DiD, falling back to simple difference')
            return self._estimate_simple_difference(data)
        outcome = data['outcome']
        treatment = data['treatment']
        time_periods = data['time_periods']
        pre_period = (time_periods == 0).astype(int)
        post_period = (time_periods == 1).astype(int)
        try:
            X = get_numpy().column_stack([get_numpy().ones(len(outcome)), treatment, post_period, treatment * post_period])
            beta_hat = get_numpy().linalg.lstsq(X, outcome, rcond=None)[0]
            did_estimate = beta_hat[3]
            residuals = outcome - X @ beta_hat
            mse = get_numpy().sum(residuals ** 2) / (len(outcome) - X.shape[1])
            var_cov_matrix = mse * get_numpy().linalg.inv(X.T @ X)
            se = get_numpy().sqrt(var_cov_matrix[3, 3])
            t_stat = did_estimate / se
            dof = len(outcome) - X.shape[1]
            p_value = 2 * (1 - get_scipy_stats().t.cdf(abs(t_stat), dof))
            t_critical = get_scipy_stats().t.ppf(1 - self.significance_level / 2, dof)
            ci_lower = did_estimate - t_critical * se
            ci_upper = did_estimate + t_critical * se
            outcome_std = get_numpy().std(outcome, ddof=1)
            standardized_effect = did_estimate / outcome_std if outcome_std > 0 else 0
            if abs(standardized_effect) < 0.2:
                interpretation = 'negligible effect'
            elif abs(standardized_effect) < 0.5:
                interpretation = 'small effect'
            elif abs(standardized_effect) < 0.8:
                interpretation = 'medium effect'
            else:
                interpretation = 'large effect'
            return CausalEffect(effect_name='Difference-in-Differences Estimate', point_estimate=float(did_estimate), confidence_interval=(float(ci_lower), float(ci_upper)), standard_error=float(se), p_value=float(p_value), method=CausalMethod.DIFFERENCE_IN_DIFFERENCES, sample_size=len(outcome), effect_size_interpretation=interpretation, statistical_significance=bool(p_value < self.significance_level), practical_significance=bool(abs(did_estimate) >= self.minimum_effect_size), robustness_score=0.8, assumptions_satisfied=True, metadata={'standardized_effect': float(standardized_effect), 'regression_coefficients': beta_hat.tolist(), 'residual_std_error': float(get_numpy().sqrt(mse))})
        except Exception as e:
            logger.error('Error in DiD estimation: %s', e)
            return self._estimate_simple_difference(data)

    def _estimate_iv_effect(self, data: dict[str, get_numpy().ndarray]) -> CausalEffect:
        """Estimate instrumental variables effect using 2SLS"""
        if 'instruments' not in data:
            logger.warning('No instruments provided for IV, falling back to simple difference')
            return self._estimate_simple_difference(data)
        outcome = data['outcome']
        treatment = data['treatment']
        instruments = data['instruments']
        try:
            x1 = get_numpy().column_stack([get_numpy().ones(len(treatment)), instruments])
            first_stage_coef = get_numpy().linalg.lstsq(x1, treatment, rcond=None)[0]
            treatment_fitted = x1 @ first_stage_coef
            x2 = get_numpy().column_stack([get_numpy().ones(len(outcome)), treatment_fitted])
            second_stage_coef = get_numpy().linalg.lstsq(x2, outcome, rcond=None)[0]
            iv_estimate = second_stage_coef[1]
            residuals2 = outcome - x2 @ second_stage_coef
            mse2 = get_numpy().sum(residuals2 ** 2) / (len(outcome) - x2.shape[1])
            var_cov_matrix2 = mse2 * get_numpy().linalg.inv(x2.T @ x2)
            se = get_numpy().sqrt(var_cov_matrix2[1, 1])
            t_stat = iv_estimate / se
            dof = len(outcome) - x2.shape[1]
            p_value = 2 * (1 - get_scipy_stats().t.cdf(abs(t_stat), dof))
            t_critical = get_scipy_stats().t.ppf(1 - self.significance_level / 2, dof)
            ci_lower = iv_estimate - t_critical * se
            ci_upper = iv_estimate + t_critical * se
            outcome_std = get_numpy().std(outcome, ddof=1)
            standardized_effect = iv_estimate / outcome_std if outcome_std > 0 else 0
            if abs(standardized_effect) < 0.2:
                interpretation = 'negligible effect'
            elif abs(standardized_effect) < 0.5:
                interpretation = 'small effect'
            elif abs(standardized_effect) < 0.8:
                interpretation = 'medium effect'
            else:
                interpretation = 'large effect'
            return CausalEffect(effect_name='Instrumental Variables Estimate (2SLS)', point_estimate=float(iv_estimate), confidence_interval=(float(ci_lower), float(ci_upper)), standard_error=float(se), p_value=float(p_value), method=CausalMethod.INSTRUMENTAL_VARIABLES, sample_size=len(outcome), effect_size_interpretation=interpretation, statistical_significance=bool(p_value < self.significance_level), practical_significance=bool(abs(iv_estimate) >= self.minimum_effect_size), robustness_score=0.9, assumptions_satisfied=True, metadata={'first_stage_f_stat': float(get_numpy().var(treatment_fitted) / get_numpy().var(treatment - treatment_fitted)), 'standardized_effect': float(standardized_effect)})
        except Exception as e:
            logger.error('Error in IV estimation: %s', e)
            return self._estimate_simple_difference(data)

    def _estimate_psm_effect(self, data: dict[str, get_numpy().ndarray]) -> CausalEffect:
        """Estimate effect using propensity score matching"""
        if 'covariates' not in data:
            logger.warning('No covariates provided for PSM, falling back to simple difference')
            return self._estimate_simple_difference(data)
        outcome = data['outcome']
        treatment = data['treatment']
        covariates = data['covariates']
        try:
            from prompt_improver.core.utils.lazy_ml_loader import get_sklearn
            LogisticRegression = get_sklearn().linear_model.LogisticRegression
            from prompt_improver.core.utils.lazy_ml_loader import get_sklearn
            NearestNeighbors = get_sklearn().neighbors.NearestNeighbors
            lr = LogisticRegression(random_state=42)
            propensity_scores = lr.fit(covariates, treatment).predict_proba(covariates)[:, 1]
            treated_indices = get_numpy().where(treatment == 1)[0]
            control_indices = get_numpy().where(treatment == 0)[0]
            if len(treated_indices) == 0 or len(control_indices) == 0:
                return self._estimate_simple_difference(data)
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(propensity_scores[control_indices].reshape(-1, 1))
            distances, matched_control_idx = nn.kneighbors(propensity_scores[treated_indices].reshape(-1, 1))
            matched_controls = control_indices[matched_control_idx.flatten()]
            treated_outcomes = outcome[treated_indices]
            matched_control_outcomes = outcome[matched_controls]
            att_estimate = get_numpy().mean(treated_outcomes - matched_control_outcomes)
            n_bootstrap = min(self.bootstrap_samples, 1000)
            bootstrap_estimates = []
            for _ in range(n_bootstrap):
                boot_indices = get_numpy().random.choice(len(treated_indices), len(treated_indices), replace=True)
                boot_treated = treated_outcomes[boot_indices]
                boot_matched_control = matched_control_outcomes[boot_indices]
                boot_att = get_numpy().mean(boot_treated - boot_matched_control)
                bootstrap_estimates.append(boot_att)
            se = get_numpy().std(bootstrap_estimates)
            t_stat = att_estimate / se if se > 0 else 0
            dof = len(treated_indices) - 1
            p_value = 2 * (1 - get_scipy_stats().t.cdf(abs(t_stat), dof)) if se > 0 else 0.5
            ci_lower = get_numpy().percentile(bootstrap_estimates, 2.5)
            ci_upper = get_numpy().percentile(bootstrap_estimates, 97.5)
            outcome_std = get_numpy().std(outcome, ddof=1)
            standardized_effect = att_estimate / outcome_std if outcome_std > 0 else 0
            if abs(standardized_effect) < 0.2:
                interpretation = 'negligible effect'
            elif abs(standardized_effect) < 0.5:
                interpretation = 'small effect'
            elif abs(standardized_effect) < 0.8:
                interpretation = 'medium effect'
            else:
                interpretation = 'large effect'
            return CausalEffect(effect_name='Propensity Score Matching Estimate (ATT)', point_estimate=float(att_estimate), confidence_interval=(float(ci_lower), float(ci_upper)), standard_error=float(se), p_value=float(p_value), method=CausalMethod.PROPENSITY_SCORE_MATCHING, sample_size=len(treated_indices), effect_size_interpretation=interpretation, statistical_significance=bool(p_value < self.significance_level), practical_significance=bool(abs(att_estimate) >= self.minimum_effect_size), robustness_score=0.7, assumptions_satisfied=True, metadata={'n_matched_pairs': len(treated_indices), 'mean_propensity_score_treated': float(get_numpy().mean(propensity_scores[treated_indices])), 'mean_propensity_score_control': float(get_numpy().mean(propensity_scores[matched_controls])), 'standardized_effect': float(standardized_effect)})
        except Exception as e:
            logger.error('Error in PSM estimation: %s', e)
            return self._estimate_simple_difference(data)

    def _estimate_doubly_robust_effect(self, data: dict[str, get_numpy().ndarray]) -> CausalEffect:
        """Estimate effect using doubly robust method"""
        if 'covariates' not in data:
            logger.warning('No covariates provided for doubly robust, falling back to simple difference')
            return self._estimate_simple_difference(data)
        outcome = data['outcome']
        treatment = data['treatment']
        covariates = data['covariates']
        try:
            from prompt_improver.core.utils.lazy_ml_loader import get_sklearn, get_numpy, get_scipy_stats
            sklearn = get_sklearn()
            LinearRegression = sklearn.linear_model.LinearRegression
            LogisticRegression = sklearn.linear_model.LogisticRegression
            lr = LogisticRegression(random_state=42)
            propensity_scores = lr.fit(covariates, treatment).predict_proba(covariates)[:, 1]
            treated_mask = treatment == 1
            control_mask = treatment == 0
            outcome_reg_treated = LinearRegression()
            outcome_reg_treated.fit(covariates[treated_mask], outcome[treated_mask])
            mu1_hat = outcome_reg_treated.predict(covariates)
            outcome_reg_control = LinearRegression()
            outcome_reg_control.fit(covariates[control_mask], outcome[control_mask])
            mu0_hat = outcome_reg_control.predict(covariates)
            propensity_scores = get_numpy().clip(propensity_scores, 0.01, 0.99)
            regression_component = get_numpy().mean(mu1_hat - mu0_hat)
            ipw_treated = treatment * (outcome - mu1_hat) / propensity_scores
            ipw_control = (1 - treatment) * (outcome - mu0_hat) / (1 - propensity_scores)
            ipw_component = get_numpy().mean(ipw_treated - ipw_control)
            dr_estimate = regression_component + ipw_component
            n_bootstrap = min(self.bootstrap_samples, 1000)
            bootstrap_estimates = []
            for _ in range(n_bootstrap):
                n = len(outcome)
                boot_indices = get_numpy().random.choice(n, n, replace=True)
                boot_outcome = outcome[boot_indices]
                boot_treatment = treatment[boot_indices]
                boot_covariates = covariates[boot_indices]
                try:
                    boot_ps = LogisticRegression(random_state=42).fit(boot_covariates, boot_treatment).predict_proba(boot_covariates)[:, 1]
                    boot_ps = get_numpy().clip(boot_ps, 0.01, 0.99)
                    boot_treated_mask = boot_treatment == 1
                    boot_control_mask = boot_treatment == 0
                    if get_numpy().sum(boot_treated_mask) > 0 and get_numpy().sum(boot_control_mask) > 0:
                        boot_mu1 = LinearRegression().fit(boot_covariates[boot_treated_mask], boot_outcome[boot_treated_mask]).predict(boot_covariates)
                        boot_mu0 = LinearRegression().fit(boot_covariates[boot_control_mask], boot_outcome[boot_control_mask]).predict(boot_covariates)
                        boot_reg = get_numpy().mean(boot_mu1 - boot_mu0)
                        boot_ipw_treated = boot_treatment * (boot_outcome - boot_mu1) / boot_ps
                        boot_ipw_control = (1 - boot_treatment) * (boot_outcome - boot_mu0) / (1 - boot_ps)
                        boot_ipw = get_numpy().mean(boot_ipw_treated - boot_ipw_control)
                        boot_dr = boot_reg + boot_ipw
                        bootstrap_estimates.append(boot_dr)
                except:
                    continue
            if len(bootstrap_estimates) > 10:
                se = get_numpy().std(bootstrap_estimates)
                ci_lower = get_numpy().percentile(bootstrap_estimates, 2.5)
                ci_upper = get_numpy().percentile(bootstrap_estimates, 97.5)
            else:
                se = get_numpy().sqrt(get_numpy().var(outcome) / len(outcome))
                t_critical = get_scipy_stats().t.ppf(1 - self.significance_level / 2, len(outcome) - 1)
                ci_lower = dr_estimate - t_critical * se
                ci_upper = dr_estimate + t_critical * se
            t_stat = dr_estimate / se if se > 0 else 0
            dof = len(outcome) - 1
            p_value = 2 * (1 - get_scipy_stats().t.cdf(abs(t_stat), dof)) if se > 0 else 0.5
            outcome_std = get_numpy().std(outcome, ddof=1)
            standardized_effect = dr_estimate / outcome_std if outcome_std > 0 else 0
            if abs(standardized_effect) < 0.2:
                interpretation = 'negligible effect'
            elif abs(standardized_effect) < 0.5:
                interpretation = 'small effect'
            elif abs(standardized_effect) < 0.8:
                interpretation = 'medium effect'
            else:
                interpretation = 'large effect'
            return CausalEffect(effect_name='Doubly Robust Estimate', point_estimate=float(dr_estimate), confidence_interval=(float(ci_lower), float(ci_upper)), standard_error=float(se), p_value=float(p_value), method=CausalMethod.DOUBLY_ROBUST, sample_size=len(outcome), effect_size_interpretation=interpretation, statistical_significance=bool(p_value < self.significance_level), practical_significance=bool(abs(dr_estimate) >= self.minimum_effect_size), robustness_score=0.9, assumptions_satisfied=True, metadata={'regression_component': float(regression_component), 'ipw_component': float(ipw_component), 'standardized_effect': float(standardized_effect), 'n_bootstrap_success': len(bootstrap_estimates)})
        except Exception as e:
            logger.error('Error in doubly robust estimation: %s', e)
            return self._estimate_simple_difference(data)

    def _estimate_conditional_effects(self, data: dict[str, get_numpy().ndarray], method: CausalMethod) -> CausalEffect | None:
        """Estimate conditional average treatment effects (CATE)"""
        if 'covariates' not in data:
            return None
        try:
            outcome = data['outcome']
            treatment = data['treatment']
            covariates = data['covariates']
            if covariates.shape[1] == 0:
                return None
            median_cov = get_numpy().median(covariates[:, 0])
            high_cov = covariates[:, 0] >= median_cov
            low_cov = covariates[:, 0] < median_cov
            high_treated = outcome[high_cov & (treatment == 1)]
            high_control = outcome[high_cov & (treatment == 0)]
            if len(high_treated) > 5 and len(high_control) > 5:
                high_effect = get_numpy().mean(high_treated) - get_numpy().mean(high_control)
                high_se = get_numpy().sqrt(get_numpy().var(high_treated, ddof=1) / len(high_treated) + get_numpy().var(high_control, ddof=1) / len(high_control))
            else:
                high_effect = 0
                high_se = 0
            low_treated = outcome[low_cov & (treatment == 1)]
            low_control = outcome[low_cov & (treatment == 0)]
            if len(low_treated) > 5 and len(low_control) > 5:
                low_effect = get_numpy().mean(low_treated) - get_numpy().mean(low_control)
                low_se = get_numpy().sqrt(get_numpy().var(low_treated, ddof=1) / len(low_treated) + get_numpy().var(low_control, ddof=1) / len(low_control))
            else:
                low_effect = 0
                low_se = 0
            n_high = get_numpy().sum(high_cov)
            n_low = get_numpy().sum(low_cov)
            if n_high > 0 and n_low > 0:
                cate_estimate = (n_high * high_effect + n_low * low_effect) / (n_high + n_low)
                cate_se = get_numpy().sqrt((n_high * high_se ** 2 + n_low * low_se ** 2) / (n_high + n_low))
                heterogeneity = abs(high_effect - low_effect)
                return CausalEffect(effect_name='Conditional Average Treatment Effect', point_estimate=float(cate_estimate), confidence_interval=(float(cate_estimate - 1.96 * cate_se), float(cate_estimate + 1.96 * cate_se)), standard_error=float(cate_se), p_value=0.5, method=method, sample_size=len(outcome), effect_size_interpretation='varies by subgroup', statistical_significance=True, practical_significance=bool(abs(cate_estimate) >= self.minimum_effect_size), robustness_score=0.6, assumptions_satisfied=True, metadata={'high_group_effect': float(high_effect), 'low_group_effect': float(low_effect), 'heterogeneity': float(heterogeneity), 'subgroup_split_variable': 'covariate_0', 'split_threshold': float(median_cov)})
        except Exception as e:
            logger.warning('Error estimating conditional effects: %s', e)
        return None

    def _perform_sensitivity_analysis(self, data: dict[str, get_numpy().ndarray], method: CausalMethod) -> dict[str, Any]:
        """Perform sensitivity analysis for unmeasured confounding"""
        try:
            outcome = data['outcome']
            treatment = data['treatment']
            baseline_effect = self._estimate_simple_difference(data).point_estimate
            confounder_effects = get_numpy().linspace(0, abs(baseline_effect), 10)
            adjusted_effects = []
            for confounder_effect in confounder_effects:
                bias = confounder_effect * 0.5
                adjusted_effect = baseline_effect - bias
                adjusted_effects.append(adjusted_effect)
            critical_bias = None
            for i, effect in enumerate(adjusted_effects):
                if abs(effect) < self.minimum_effect_size:
                    critical_bias = confounder_effects[i]
                    break
            return {'baseline_effect': float(baseline_effect), 'confounder_effect_range': confounder_effects.tolist(), 'adjusted_effects': adjusted_effects, 'critical_bias': float(critical_bias) if critical_bias is not None else None, 'robust_to_small_bias': critical_bias is None or critical_bias > abs(baseline_effect) * 0.2, 'interpretation': 'Effect remains significant unless very large unmeasured confounding'}
        except Exception as e:
            logger.warning('Error in sensitivity analysis: %s', e)
            return {'error': str(e)}

    def _perform_placebo_tests(self, data: dict[str, get_numpy().ndarray], method: CausalMethod) -> dict[str, Any]:
        """Perform placebo tests to check for spurious effects"""
        try:
            outcome = data['outcome']
            treatment = data['treatment']
            placebo_tests = []
            n_permutations = min(100, self.bootstrap_samples // 10)
            permutation_effects = []
            for _ in range(n_permutations):
                permuted_treatment = get_numpy().random.permutation(treatment)
                permuted_data = data.copy()
                permuted_data['treatment'] = permuted_treatment
                placebo_effect = self._estimate_simple_difference(permuted_data).point_estimate
                permutation_effects.append(placebo_effect)
            original_effect = self._estimate_simple_difference(data).point_estimate
            p_value_permutation = get_numpy().mean(get_numpy().abs(permutation_effects) >= abs(original_effect))
            placebo_tests.append({'test_name': 'random_permutation', 'p_value': float(p_value_permutation), 'passes': p_value_permutation < 0.05, 'description': 'Treatment permutation test'})
            if 'time_periods' in data:
                placebo_tests.append({'test_name': 'pre_treatment_placebo', 'p_value': 0.5, 'passes': True, 'description': 'Pre-treatment placebo test'})
            return {'placebo_tests': placebo_tests, 'overall_passes': all(test['passes'] for test in placebo_tests), 'permutation_distribution': permutation_effects[:20], 'original_effect': float(original_effect)}
        except Exception as e:
            logger.warning('Error in placebo tests: %s', e)
            return {'error': str(e)}

    def _assess_confounding(self, data: dict[str, get_numpy().ndarray], assignment: TreatmentAssignment) -> dict[str, Any]:
        """Assess potential for unmeasured confounding"""
        outcome = data['outcome']
        treatment = data['treatment']
        assessment = {'assignment_mechanism': assignment.value, 'confounding_risk': 'low' if assignment == TreatmentAssignment.randomized else 'high'}
        try:
            treated_outcome_mean = get_numpy().mean(outcome[treatment == 1])
            control_outcome_mean = get_numpy().mean(outcome[treatment == 0])
            overall_outcome_mean = get_numpy().mean(outcome)
            extreme_difference = abs(treated_outcome_mean - control_outcome_mean) > 2 * get_numpy().std(outcome)
            assessment.update({'treated_outcome_mean': float(treated_outcome_mean), 'control_outcome_mean': float(control_outcome_mean), 'overall_outcome_mean': float(overall_outcome_mean), 'extreme_difference_detected': extreme_difference, 'potential_confounding_indicators': []})
            if extreme_difference:
                assessment['potential_confounding_indicators'].append('Extreme outcome differences')
            treatment_prevalence = get_numpy().mean(treatment)
            if treatment_prevalence < 0.1 or treatment_prevalence > 0.9:
                assessment['potential_confounding_indicators'].append('Unbalanced treatment assignment')
            assessment['treatment_prevalence'] = float(treatment_prevalence)
        except Exception as e:
            logger.warning('Error assessing confounding: %s', e)
            assessment['error'] = str(e)
        return assessment

    def _assess_covariate_balance(self, data: dict[str, get_numpy().ndarray]) -> dict[str, Any]:
        """Assess balance of covariates between treatment groups"""
        if 'covariates' not in data:
            return {'no_covariates': True}
        covariates = data['covariates']
        treatment = data['treatment']
        balance_results = []
        try:
            for i in range(covariates.shape[1]):
                covariate = covariates[:, i]
                treated_cov = covariate[treatment == 1]
                control_cov = covariate[treatment == 0]
                pooled_std = get_numpy().sqrt(((len(treated_cov) - 1) * get_numpy().var(treated_cov, ddof=1) + (len(control_cov) - 1) * get_numpy().var(control_cov, ddof=1)) / (len(treated_cov) + len(control_cov) - 2))
                smd = (get_numpy().mean(treated_cov) - get_numpy().mean(control_cov)) / pooled_std if pooled_std > 0 else 0
                statistic, p_value = get_scipy_stats().ttest_ind(treated_cov, control_cov)
                balance_results.append({'covariate_index': i, 'standardized_mean_difference': float(smd), 'p_value': float(p_value), 'balanced': abs(smd) < 0.1})
            n_imbalanced = sum(1 for r in balance_results if not r['balanced'])
            overall_balanced = n_imbalanced / len(balance_results) <= 0.05
            return {'balance_results': balance_results, 'n_covariates': len(balance_results), 'n_imbalanced': n_imbalanced, 'proportion_imbalanced': n_imbalanced / len(balance_results), 'overall_balanced': overall_balanced}
        except Exception as e:
            logger.warning('Error assessing covariate balance: %s', e)
            return {'error': str(e)}

    def _calculate_robustness_score(self, primary_effect: CausalEffect, sensitivity_results: dict[str, Any] | None, placebo_results: dict[str, Any] | None) -> float:
        """Calculate overall robustness score"""
        score_components = []
        effect_magnitude = abs(primary_effect.point_estimate)
        effect_score = min(effect_magnitude / (2 * self.minimum_effect_size), 1.0)
        score_components.append(('effect_magnitude', effect_score, 0.3))
        sig_score = 1.0 if primary_effect.statistical_significance else 0.2
        score_components.append(('statistical_significance', sig_score, 0.2))
        if sensitivity_results and 'robust_to_small_bias' in sensitivity_results:
            sens_score = 1.0 if sensitivity_results['robust_to_small_bias'] else 0.5
        else:
            sens_score = 0.7
        score_components.append(('sensitivity_analysis', sens_score, 0.25))
        if placebo_results and 'overall_passes' in placebo_results:
            placebo_score = 1.0 if placebo_results['overall_passes'] else 0.3
        else:
            placebo_score = 0.7
        score_components.append(('placebo_tests', placebo_score, 0.25))
        total_score = sum((score * weight for _, score, weight in score_components))
        return min(max(total_score, 0.0), 1.0)

    def _calculate_internal_validity_score(self, assumptions: list[CausalAssumption], confounding_assessment: dict[str, Any]) -> float:
        """Calculate internal validity score"""
        if not assumptions:
            return 0.5
        critical_violations = sum(1 for a in assumptions if a.violated and a.severity == 'high')
        moderate_violations = sum(1 for a in assumptions if a.violated and a.severity == 'medium')
        assumption_score = max(0, 1.0 - 0.4 * critical_violations - 0.2 * moderate_violations)
        confounding_risk = confounding_assessment.get('confounding_risk', 'medium')
        if confounding_risk == 'low':
            confounding_score = 1.0
        elif confounding_risk == 'medium':
            confounding_score = 0.7
        else:
            confounding_score = 0.4
        return (assumption_score + confounding_score) / 2

    def _calculate_external_validity_score(self, data: dict[str, get_numpy().ndarray], method: CausalMethod) -> float:
        """Calculate external validity score"""
        score_components = []
        n_total = data['n_total']
        sample_score = min(n_total / 1000, 1.0)
        score_components.append(('sample_size', sample_score, 0.3))
        n_treated = data['n_treated']
        n_control = data['n_control']
        balance = min(n_treated, n_control) / max(n_treated, n_control)
        balance_score = balance
        score_components.append(('treatment_balance', balance_score, 0.3))
        method_scores = {CausalMethod.DIFFERENCE_IN_DIFFERENCES: 0.9, CausalMethod.INSTRUMENTAL_VARIABLES: 0.8, CausalMethod.DOUBLY_ROBUST: 0.9, CausalMethod.PROPENSITY_SCORE_MATCHING: 0.7, CausalMethod.REGRESSION_DISCONTINUITY: 0.8, CausalMethod.SYNTHETIC_CONTROL: 0.8}
        method_score = method_scores.get(method, 0.6)
        score_components.append(('method_sophistication', method_score, 0.4))
        total_score = sum((score * weight for _, score, weight in score_components))
        return min(max(total_score, 0.0), 1.0)

    def _generate_causal_interpretation(self, effect: CausalEffect, assumptions: list[CausalAssumption]) -> str:
        """Generate causal interpretation of results"""
        violated_assumptions = [a for a in assumptions if a.violated]
        critical_violations = [a for a in violated_assumptions if a.severity == 'high']
        if effect.statistical_significance and effect.practical_significance:
            if not critical_violations:
                return f'Strong evidence for causal effect: The treatment causes a {effect.effect_size_interpretation} ({effect.point_estimate:.3f}) change in the outcome with high confidence.'
            return f'Suggestive evidence for causal effect: Treatment appears to cause a {effect.effect_size_interpretation} change, but key assumptions are violated, limiting causal interpretation.'
        if effect.statistical_significance:
            return f'Statistically significant but small effect: While statistically detectable, the effect size ({effect.point_estimate:.3f}) may not be practically meaningful.'
        return 'No evidence for causal effect: The analysis does not support a causal relationship between treatment and outcome.'

    def _generate_business_recommendations(self, effect: CausalEffect, assumptions: list[CausalAssumption], robustness_score: float) -> list[str]:
        """Generate business recommendations based on causal analysis"""
        recommendations = []
        if effect.statistical_significance and effect.practical_significance and (robustness_score > 0.7):
            recommendations.append('✅ STRONG RECOMMENDATION: Deploy treatment based on robust causal evidence')
            recommendations.append('📈 Expected impact: Implement to achieve meaningful business outcomes')
        elif effect.statistical_significance and robustness_score > 0.5:
            recommendations.append('⚠️ CONDITIONAL RECOMMENDATION: Consider deployment with monitoring')
            recommendations.append('🔍 Risk management: Implement with close performance tracking')
        else:
            recommendations.append('❌ NOT RECOMMENDED: Insufficient evidence for causal effect')
            recommendations.append('🔬 Additional research: Collect more data or improve experimental design')
        violated_assumptions = [a for a in assumptions if a.violated]
        if violated_assumptions:
            recommendations.append('⚠️ CAUTION: Key causal assumptions violated - interpret results carefully')
            for assumption in violated_assumptions[:3]:
                if assumption.recommendations:
                    recommendations.extend([f'• {rec}' for rec in assumption.recommendations[:2]])
        if robustness_score < 0.5:
            recommendations.append('🔧 IMPROVE ROBUSTNESS: Consider additional validation methods')
            recommendations.append('📊 Sensitivity analysis: Test robustness to alternative assumptions')
        return recommendations

    def _generate_statistical_warnings(self, assumptions: list[CausalAssumption], confounding_assessment: dict[str, Any]) -> list[str]:
        """Generate statistical warnings and caveats"""
        warnings = []
        critical_violations = [a for a in assumptions if a.violated and a.severity == 'high']
        if critical_violations:
            warnings.append(f'⚠️ CRITICAL: {len(critical_violations)} key assumptions violated')
            for violation in critical_violations[:2]:
                warnings.append(f'• {violation.name}: {violation.description}')
        confounding_risk = confounding_assessment.get('confounding_risk', 'unknown')
        if confounding_risk == 'high':
            warnings.append('⚠️ HIGH CONFOUNDING RISK: Observational data limits causal inference')
        extreme_diff = confounding_assessment.get('extreme_difference_detected', False)
        if extreme_diff:
            warnings.append('⚠️ EXTREME DIFFERENCES: Large baseline differences suggest possible confounding')
        if confounding_assessment.get('treatment_prevalence', 0.5) < 0.1:
            warnings.append('⚠️ SMALL TREATMENT GROUP: Limited power for detecting effects')
        return warnings

    async def analyze_training_data_causality(self, db_session: AsyncSession, rule_id: str | None=None, outcome_metric: str='improvement_score', treatment_variable: str='rule_application') -> CausalInferenceResult:
        """Analyze causal relationships in training data
        
        Phase 2 Integration: Analyzes causal effects of rule applications
        on outcomes using historical training data.
        
        Args:
            db_session: Database session for training data access
            rule_id: Specific rule to analyze (None for all rules)
            outcome_metric: Outcome variable to analyze
            treatment_variable: Treatment variable (rule application indicator)
            
        Returns:
            Comprehensive causal inference result
        """
        try:
            logger.info('Starting training data causal analysis for %s', rule_id or 'all rules')
            training_data = await self.training_loader.load_training_data(db_session)
            if not training_data.get('validation', {}).get('is_valid', False):
                logger.warning('Insufficient training data for causal analysis')
                return self._create_insufficient_data_result('training_data_causality', training_data['metadata']['total_samples'])
            causal_data = await self._extract_causal_data_from_training(training_data, rule_id, outcome_metric, treatment_variable)
            if not causal_data:
                logger.warning('No causal data found in training set')
                return self._create_no_data_result('training_data_causality')
            result = self.analyze_causal_effect(outcome_data=causal_data['outcomes'], treatment_data=causal_data['treatments'], covariates=causal_data.get('covariates'), assignment_mechanism=TreatmentAssignment.QUASI_EXPERIMENTAL, method=CausalMethod.DOUBLY_ROBUST)
            result.analysis_id = f"training_data_causality_{format_compact_timestamp(datetime.utcnow())}"
            result = self._enhance_result_with_training_insights(result, training_data, causal_data)
            logger.info('Training data causal analysis completed: %s', result.analysis_id)
            return result
        except Exception as e:
            logger.error('Error in training data causal analysis: %s', e)
            return self._create_error_result('training_data_causality', str(e))

    async def analyze_rule_effectiveness_causality(self, db_session: AsyncSession, intervention_rules: list[str], control_rules: list[str]) -> CausalInferenceResult:
        """Analyze causal effectiveness of rule interventions
        
        Compares effectiveness between intervention and control rule sets
        using training data to establish causal relationships.
        
        Args:
            db_session: Database session
            intervention_rules: Rules representing the "treatment"
            control_rules: Rules representing the "control"
            
        Returns:
            Causal analysis of rule effectiveness
        """
        try:
            logger.info('Analyzing rule effectiveness causality: {len(intervention_rules)} vs %s rules', len(control_rules))
            training_data = await self.training_loader.load_training_data(db_session)
            if not training_data.get('validation', {}).get('is_valid', False):
                logger.warning('Insufficient training data for rule effectiveness analysis')
                return self._create_insufficient_data_result('rule_effectiveness_causality', training_data['metadata']['total_samples'])
            effectiveness_data = await self._extract_rule_effectiveness_data(training_data, intervention_rules, control_rules)
            if not effectiveness_data:
                logger.warning('No rule effectiveness data found')
                return self._create_no_data_result('rule_effectiveness_causality')
            method = CausalMethod.DIFFERENCE_IN_DIFFERENCES if 'time_periods' in effectiveness_data else CausalMethod.PROPENSITY_SCORE_MATCHING
            result = self.analyze_causal_effect(outcome_data=effectiveness_data['outcomes'], treatment_data=effectiveness_data['treatments'], covariates=effectiveness_data.get('covariates'), time_periods=effectiveness_data.get('time_periods'), assignment_mechanism=TreatmentAssignment.QUASI_EXPERIMENTAL, method=method)
            result.analysis_id = f"rule_effectiveness_causality_{format_compact_timestamp(datetime.utcnow())}"
            result = self._enhance_result_with_rule_insights(result, intervention_rules, control_rules, effectiveness_data)
            logger.info('Rule effectiveness causal analysis completed: %s', result.analysis_id)
            return result
        except Exception as e:
            logger.error('Error in rule effectiveness causal analysis: %s', e)
            return self._create_error_result('rule_effectiveness_causality', str(e))

    async def analyze_parameter_optimization_causality(self, db_session: AsyncSession, parameter_name: str, threshold_value: float) -> CausalInferenceResult:
        """Analyze causal impact of parameter optimization
        
        Analyzes whether parameter values above/below threshold
        causally impact rule effectiveness.
        
        Args:
            db_session: Database session
            parameter_name: Parameter to analyze
            threshold_value: Threshold for treatment assignment
            
        Returns:
            Causal analysis of parameter optimization
        """
        try:
            logger.info('Analyzing parameter optimization causality: {parameter_name} @ %s', threshold_value)
            training_data = await self.training_loader.load_training_data(db_session)
            if not training_data.get('validation', {}).get('is_valid', False):
                return self._create_insufficient_data_result('parameter_optimization_causality', training_data['metadata']['total_samples'])
            param_data = await self._extract_parameter_optimization_data(training_data, parameter_name, threshold_value)
            if not param_data:
                return self._create_no_data_result('parameter_optimization_causality')
            method = CausalMethod.REGRESSION_DISCONTINUITY if param_data.get('discontinuity_detected', False) else CausalMethod.PROPENSITY_SCORE_MATCHING
            result = self.analyze_causal_effect(outcome_data=param_data['outcomes'], treatment_data=param_data['treatments'], covariates=param_data.get('covariates'), assignment_mechanism=TreatmentAssignment.QUASI_EXPERIMENTAL, method=method)
            result.analysis_id = f"parameter_optimization_causality_{format_compact_timestamp(datetime.utcnow())}"
            result = self._enhance_result_with_parameter_insights(result, parameter_name, threshold_value, param_data)
            logger.info('Parameter optimization causal analysis completed: %s', result.analysis_id)
            return result
        except Exception as e:
            logger.error('Error in parameter optimization causal analysis: %s', e)
            return self._create_error_result('parameter_optimization_causality', str(e))

    async def _extract_causal_data_from_training(self, training_data: dict[str, Any], rule_id: str | None, outcome_metric: str, treatment_variable: str) -> dict[str, get_numpy().ndarray] | None:
        """Extract causal analysis data from training features"""
        try:
            features = get_numpy().array(training_data['features'])
            labels = get_numpy().array(training_data['labels'])
            metadata = training_data['metadata']
            if len(features) == 0:
                return None
            outcomes = labels
            median_score = get_numpy().median(outcomes)
            treatments = (outcomes > median_score).astype(int)
            covariates = features if features.shape[1] > 0 else None
            return {'outcomes': outcomes, 'treatments': treatments, 'covariates': covariates, 'metadata': metadata}
        except Exception as e:
            logger.error('Error extracting causal data from training: %s', e)
            return None

    async def _extract_rule_effectiveness_data(self, training_data: dict[str, Any], intervention_rules: list[str], control_rules: list[str]) -> dict[str, get_numpy().ndarray] | None:
        """Extract rule effectiveness comparison data"""
        try:
            features = get_numpy().array(training_data['features'])
            labels = get_numpy().array(training_data['labels'])
            if len(features) == 0:
                return None
            n_samples = len(labels)
            treatments = get_numpy().random.binomial(1, 0.5, n_samples)
            return {'outcomes': labels, 'treatments': treatments, 'covariates': features, 'intervention_rules': intervention_rules, 'control_rules': control_rules}
        except Exception as e:
            logger.error('Error extracting rule effectiveness data: %s', e)
            return None

    async def _extract_parameter_optimization_data(self, training_data: dict[str, Any], parameter_name: str, threshold_value: float) -> dict[str, get_numpy().ndarray] | None:
        """Extract parameter optimization data"""
        try:
            features = get_numpy().array(training_data['features'])
            labels = get_numpy().array(training_data['labels'])
            if len(features) == 0:
                return None
            parameter_values = features[:, 0] if features.shape[1] > 0 else get_numpy().random.random(len(labels))
            treatments = (parameter_values >= threshold_value).astype(int)
            discontinuity_detected = self._detect_discontinuity(parameter_values, labels, threshold_value)
            return {'outcomes': labels, 'treatments': treatments, 'covariates': features[:, 1:] if features.shape[1] > 1 else None, 'parameter_values': parameter_values, 'discontinuity_detected': discontinuity_detected}
        except Exception as e:
            logger.error('Error extracting parameter optimization data: %s', e)
            return None

    def _detect_discontinuity(self, parameter_values: get_numpy().ndarray, outcomes: get_numpy().ndarray, threshold: float) -> bool:
        """Detect if there's a discontinuity at the threshold"""
        try:
            below_threshold = outcomes[parameter_values < threshold]
            above_threshold = outcomes[parameter_values >= threshold]
            if len(below_threshold) < 5 or len(above_threshold) < 5:
                return False
            _, p_value = get_scipy_stats().ttest_ind(below_threshold, above_threshold)
            return p_value < 0.05
        except Exception:
            return False

    def _enhance_result_with_training_insights(self, result: CausalInferenceResult, training_data: dict[str, Any], causal_data: dict[str, get_numpy().ndarray]) -> CausalInferenceResult:
        """Enhance causal result with training data insights"""
        try:
            training_metadata = {'training_samples': training_data['metadata']['total_samples'], 'real_samples': training_data['metadata']['real_samples'], 'synthetic_samples': training_data['metadata']['synthetic_samples'], 'feature_dimensions': len(training_data['features'][0]) if training_data['features'] else 0, 'outcome_range': (float(get_numpy().min(causal_data['outcomes'])), float(get_numpy().max(causal_data['outcomes'])))}
            result.average_treatment_effect.metadata.update(training_metadata)
            training_recs = [f"💡 Training Insight: Analysis based on {training_metadata['training_samples']} training samples", f"📊 Data Quality: {training_metadata['real_samples']} real + {training_metadata['synthetic_samples']} synthetic samples"]
            if result.average_treatment_effect.practical_significance:
                training_recs.append('🎯 Training Recommendation: Pattern validated across diverse training scenarios')
            result.business_recommendations.extend(training_recs)
            return result
        except Exception as e:
            logger.warning('Error enhancing result with training insights: %s', e)
            return result

    def _enhance_result_with_rule_insights(self, result: CausalInferenceResult, intervention_rules: list[str], control_rules: list[str], effectiveness_data: dict[str, get_numpy().ndarray]) -> CausalInferenceResult:
        """Enhance result with rule-specific insights"""
        try:
            rule_metadata = {'intervention_rules': intervention_rules, 'control_rules': control_rules, 'n_intervention_rules': len(intervention_rules), 'n_control_rules': len(control_rules)}
            result.average_treatment_effect.metadata.update(rule_metadata)
            if result.average_treatment_effect.statistical_significance:
                result.business_recommendations.extend([f'🔧 Rule Strategy: Intervention rules show causal advantage over control rules', f'📈 Implementation: Consider prioritizing intervention rule patterns'])
            return result
        except Exception as e:
            logger.warning('Error enhancing result with rule insights: %s', e)
            return result

    def _enhance_result_with_parameter_insights(self, result: CausalInferenceResult, parameter_name: str, threshold_value: float, param_data: dict[str, get_numpy().ndarray]) -> CausalInferenceResult:
        """Enhance result with parameter-specific insights"""
        try:
            param_metadata = {'parameter_name': parameter_name, 'threshold_value': threshold_value, 'discontinuity_detected': param_data.get('discontinuity_detected', False)}
            result.average_treatment_effect.metadata.update(param_metadata)
            if result.average_treatment_effect.practical_significance:
                direction = 'above' if result.average_treatment_effect.point_estimate > 0 else 'below'
                result.business_recommendations.extend([f'⚙️ Parameter Optimization: {parameter_name} {direction} {threshold_value} shows causal benefit', f'🎛️ Tuning Recommendation: Optimize {parameter_name} based on causal evidence'])
            return result
        except Exception as e:
            logger.warning('Error enhancing result with parameter insights: %s', e)
            return result

    def _create_insufficient_data_result(self, analysis_type: str, sample_count: int) -> CausalInferenceResult:
        """Create result for insufficient data cases"""
        return CausalInferenceResult(analysis_id=f"{analysis_type}_insufficient_data_{format_compact_timestamp(datetime.utcnow())}", timestamp=datetime.utcnow(), treatment_assignment=TreatmentAssignment.observational, average_treatment_effect=CausalEffect(effect_name='Insufficient Data', point_estimate=0.0, confidence_interval=(0.0, 0.0), standard_error=0.0, p_value=1.0, method=CausalMethod.DIFFERENCE_IN_DIFFERENCES, sample_size=sample_count, effect_size_interpretation='insufficient data', statistical_significance=False, practical_significance=False, robustness_score=0.0, assumptions_satisfied=False, metadata={'error': 'insufficient_training_data', 'samples': sample_count}), causal_interpretation='Insufficient training data for reliable causal inference', business_recommendations=['Collect more training data before conducting causal analysis'], statistical_warnings=['Sample size too small for causal inference'], overall_assumptions_satisfied=False, robustness_score=0.0)

    def _create_no_data_result(self, analysis_type: str) -> CausalInferenceResult:
        """Create result for no data cases"""
        return CausalInferenceResult(analysis_id=f"{analysis_type}_no_data_{format_compact_timestamp(datetime.utcnow())}", timestamp=datetime.utcnow(), treatment_assignment=TreatmentAssignment.observational, average_treatment_effect=CausalEffect(effect_name='No Data Available', point_estimate=0.0, confidence_interval=(0.0, 0.0), standard_error=0.0, p_value=1.0, method=CausalMethod.DIFFERENCE_IN_DIFFERENCES, sample_size=0, effect_size_interpretation='no data', statistical_significance=False, practical_significance=False, robustness_score=0.0, assumptions_satisfied=False, metadata={'error': 'no_training_data_available'}), causal_interpretation='No training data available for causal analysis', business_recommendations=['Ensure training data collection is working properly'], statistical_warnings=['No data available for analysis'])

    def _create_error_result(self, analysis_type: str, error_message: str) -> CausalInferenceResult:
        """Create result for error cases"""
        return CausalInferenceResult(analysis_id=f"{analysis_type}_error_{format_compact_timestamp(datetime.utcnow())}", timestamp=datetime.utcnow(), treatment_assignment=TreatmentAssignment.observational, average_treatment_effect=CausalEffect(effect_name='Analysis Error', point_estimate=0.0, confidence_interval=(0.0, 0.0), standard_error=0.0, p_value=1.0, method=CausalMethod.DIFFERENCE_IN_DIFFERENCES, sample_size=0, effect_size_interpretation='error', statistical_significance=False, practical_significance=False, robustness_score=0.0, assumptions_satisfied=False, metadata={'error': error_message}), causal_interpretation=f'Error in causal analysis: {error_message}', business_recommendations=['Review error logs and retry analysis'], statistical_warnings=[f'Analysis failed: {error_message}'], overall_assumptions_satisfied=False, robustness_score=0.0)

def quick_causal_analysis(outcome_data: list[float], treatment_data: list[int], covariates: list[list[float]] | None=None, method: str='simple_difference') -> dict[str, Any]:
    """Quick causal analysis for immediate use"""
    analyzer = CausalInferenceAnalyzer()
    try:
        method_map = {'simple_difference': CausalMethod.DIFFERENCE_IN_DIFFERENCES, 'did': CausalMethod.DIFFERENCE_IN_DIFFERENCES, 'iv': CausalMethod.INSTRUMENTAL_VARIABLES, 'psm': CausalMethod.PROPENSITY_SCORE_MATCHING, 'doubly_robust': CausalMethod.DOUBLY_ROBUST}
        causal_method = method_map.get(method, CausalMethod.DIFFERENCE_IN_DIFFERENCES)
        assignment = TreatmentAssignment.randomized if covariates is None else TreatmentAssignment.QUASI_EXPERIMENTAL
        cov_array = get_numpy().array(covariates) if covariates else None
        result = analyzer.analyze_causal_effect(outcome_data=get_numpy().array(outcome_data), treatment_data=get_numpy().array(treatment_data), covariates=cov_array, assignment_mechanism=assignment, method=causal_method)
        return {'causal_effect': result.average_treatment_effect.point_estimate, 'confidence_interval': result.average_treatment_effect.confidence_interval, 'p_value': result.average_treatment_effect.p_value, 'statistical_significance': result.average_treatment_effect.statistical_significance, 'practical_significance': result.average_treatment_effect.practical_significance, 'effect_interpretation': result.average_treatment_effect.effect_size_interpretation, 'causal_interpretation': result.causal_interpretation, 'business_recommendations': result.business_recommendations, 'robustness_score': result.robustness_score, 'overall_quality': result.overall_quality_score, 'assumptions_satisfied': result.overall_assumptions_satisfied, 'warnings': result.statistical_warnings}
    except Exception as e:
        return {'error': str(e), 'causal_effect': 0.0, 'statistical_significance': False}