"""Advanced Statistical Validator for A/B Testing
Implements 2025 best practices for statistical validation and multiple testing corrections
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
# Lazy loading for statsmodels imports
# from statsmodels.stats.multitest import multipletests
# from statsmodels.stats.power import ttest_power  
# from statsmodels.stats.weightstats import ztest
# import numpy as np  # Converted to lazy loading
import pandas as pd
from prompt_improver.core.utils.lazy_ml_loader import get_numpy, get_scipy
from prompt_improver.core.utils.lazy_ml_loader import get_scipy_stats
# from scipy import stats  # Converted to lazy loading
# from get_scipy().stats import chi2_contingency, kruskal, mannwhitneyu  # Converted to lazy loading
logger = logging.getLogger(__name__)

class CorrectionMethod(Enum):
    """Multiple testing correction methods"""
    bonferroni = 'bonferroni'
    holm = 'holm'
    sidak = 'sidak'
    BENJAMINI_HOCHBERG = 'fdr_bh'
    BENJAMINI_YEKUTIELI = 'fdr_by'
    FALSE_DISCOVERY_RATE = 'fdr_tsbh'

class EffectSizeMagnitude(Enum):
    """Effect size magnitude classifications"""
    negligible = 'negligible'
    small = 'small'
    MEDIUM = 'medium'
    large = 'large'
    VERY_LARGE = 'very_large'

@dataclass
class StatisticalTestResult:
    """Comprehensive statistical test result"""
    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: float | None = None
    effect_size: float | None = None
    effect_size_type: str | None = None
    confidence_interval: tuple[float, float] | None = None
    power: float | None = None
    minimum_detectable_effect: float | None = None
    assumptions_met: dict[str, bool] | None = None
    notes: list[str] = field(default_factory=list)

@dataclass
class MultipleTestingCorrection:
    """Results of multiple testing correction"""
    method: CorrectionMethod
    original_p_values: list[float]
    corrected_p_values: list[float]
    rejected: list[bool]
    alpha_corrected: float
    family_wise_error_rate: float
    false_discovery_rate: float | None = None

@dataclass
class AdvancedValidationResult:
    """Comprehensive validation result with 2025 best practices"""
    validation_id: str
    timestamp: datetime
    primary_test: StatisticalTestResult
    effect_size_magnitude: EffectSizeMagnitude
    practical_significance: bool
    clinical_significance: bool
    multiple_testing_correction: MultipleTestingCorrection | None = None
    sensitivity_analysis: dict[str, Any] | None = None
    post_hoc_power: float | None = None
    prospective_power: float | None = None
    normality_tests: dict[str, StatisticalTestResult] | None = None
    homogeneity_tests: dict[str, StatisticalTestResult] | None = None
    non_parametric_tests: dict[str, StatisticalTestResult] | None = None
    bootstrap_results: dict[str, Any] | None = None
    recommendations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validation_quality_score: float = 0.0

class AdvancedStatisticalValidator:
    """Advanced statistical validator implementing 2025 best practices"""

    def __init__(self, alpha: float=0.05, power_threshold: float=0.8, min_effect_size: float=0.1, bootstrap_samples: int=10000):
        """Initialize advanced statistical validator

        Args:
            alpha: Significance level
            power_threshold: Minimum acceptable statistical power
            min_effect_size: Minimum practically significant effect size
            bootstrap_samples: Number of bootstrap samples for resampling
        """
        self.alpha = alpha
        self.power_threshold = power_threshold
        self.min_effect_size = min_effect_size
        self.bootstrap_samples = bootstrap_samples

    async def run_orchestrated_analysis(self, config: dict[str, Any]) -> dict[str, Any]:
        """Orchestrator-compatible interface for advanced statistical validation (2025 pattern)

        Args:
            config: Orchestrator configuration containing:
                - control_data: Control group data
                - treatment_data: Treatment group data
                - test_names: Optional list of test names
                - correction_method: Multiple testing correction method
                - validate_assumptions: Whether to validate statistical assumptions
                - include_bootstrap: Whether to include bootstrap analysis
                - include_sensitivity: Whether to include sensitivity analysis
                - output_path: Local path for output files (optional)

        Returns:
            Orchestrator-compatible result with statistical validation and metadata
        """
        start_time = datetime.now()
        try:
            control_data = config.get('control_data', [])
            treatment_data = config.get('treatment_data', [])
            test_names = config.get('test_names', None)
            correction_method_str = config.get('correction_method', 'benjamini_hochberg')
            validate_assumptions = config.get('validate_assumptions', True)
            include_bootstrap = config.get('include_bootstrap', True)
            include_sensitivity = config.get('include_sensitivity', True)
            output_path = config.get('output_path', './outputs/statistical_validation')
            if not control_data or not treatment_data:
                raise ValueError('Both control_data and treatment_data are required')
            try:
                correction_method = CorrectionMethod(correction_method_str)
            except ValueError:
                correction_method = CorrectionMethod.BENJAMINI_HOCHBERG
                logger.warning(f"Unknown correction method '{correction_method_str}', using benjamini_hochberg")
            validation_result = self.validate_ab_test(control_data=control_data, treatment_data=treatment_data, test_names=test_names, correction_method=correction_method, validate_assumptions=validate_assumptions, include_bootstrap=include_bootstrap, include_sensitivity=include_sensitivity)
            result = {'primary_test': {'test_name': validation_result.primary_test.test_name, 'statistic': validation_result.primary_test.statistic, 'p_value': validation_result.primary_test.p_value, 'effect_size': validation_result.primary_test.effect_size, 'effect_size_type': validation_result.primary_test.effect_size_type, 'confidence_interval': validation_result.primary_test.confidence_interval, 'power': validation_result.primary_test.power, 'assumptions_met': validation_result.primary_test.assumptions_met}, 'effect_size_analysis': {'magnitude': validation_result.effect_size_magnitude.value, 'practical_significance': validation_result.practical_significance, 'clinical_significance': validation_result.clinical_significance}, 'multiple_testing': {'correction_applied': validation_result.multiple_testing_correction is not None, 'method': correction_method.value, 'family_wise_error_rate': validation_result.multiple_testing_correction.family_wise_error_rate if validation_result.multiple_testing_correction else None, 'false_discovery_rate': validation_result.multiple_testing_correction.false_discovery_rate if validation_result.multiple_testing_correction else None}, 'power_analysis': {'post_hoc_power': validation_result.post_hoc_power, 'prospective_power': validation_result.prospective_power}, 'quality_assessment': {'validation_quality_score': validation_result.validation_quality_score, 'recommendations': validation_result.recommendations, 'warnings': validation_result.warnings}}
            execution_time = (datetime.now() - start_time).total_seconds()
            return {'orchestrator_compatible': True, 'component_result': result, 'local_metadata': {'output_path': output_path, 'execution_time': execution_time, 'validation_id': validation_result.validation_id, 'control_sample_size': len(control_data), 'treatment_sample_size': len(treatment_data), 'correction_method': correction_method.value, 'assumptions_validated': validate_assumptions, 'bootstrap_included': include_bootstrap, 'sensitivity_included': include_sensitivity, 'component_version': '1.0.0'}}
        except ValueError as e:
            logger.error('Validation error in orchestrated statistical validation: %s', e)
            return {'orchestrator_compatible': True, 'component_result': {'error': f'Validation error: {str(e)}', 'primary_test': {}}, 'local_metadata': {'execution_time': (datetime.now() - start_time).total_seconds(), 'error': True, 'error_type': 'validation', 'component_version': '1.0.0'}}
        except Exception as e:
            logger.error('Orchestrated statistical validation failed: %s', e)
            return {'orchestrator_compatible': True, 'component_result': {'error': str(e), 'primary_test': {}}, 'local_metadata': {'execution_time': (datetime.now() - start_time).total_seconds(), 'error': True, 'component_version': '1.0.0'}}

    def validate_ab_test(self, control_data: list[float], treatment_data: list[float], test_names: list[str] | None=None, correction_method: CorrectionMethod=CorrectionMethod.BENJAMINI_HOCHBERG, validate_assumptions: bool=True, include_bootstrap: bool=True, include_sensitivity: bool=True) -> AdvancedValidationResult:
        """Perform comprehensive A/B test validation with 2025 best practices

        Args:
            control_data: Control group measurements
            treatment_data: Treatment group measurements
            test_names: Optional list of test names for multiple comparisons
            correction_method: Multiple testing correction method
            validate_assumptions: Whether to test statistical assumptions
            include_bootstrap: Whether to include bootstrap analysis
            include_sensitivity: Whether to include sensitivity analysis

        Returns:
            Comprehensive validation result
        """
        validation_id = f"validation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        try:
            control_array = get_numpy().array(control_data, dtype=float)
            treatment_array = get_numpy().array(treatment_data, dtype=float)
            self._validate_input_data(control_array, treatment_array)
            primary_test = self._perform_primary_test(control_array, treatment_array)
            effect_size_magnitude, practical_sig, clinical_sig = self._analyze_effect_size(control_array, treatment_array, primary_test.effect_size)
            result = AdvancedValidationResult(validation_id=validation_id, timestamp=datetime.utcnow(), primary_test=primary_test, effect_size_magnitude=effect_size_magnitude, practical_significance=practical_sig, clinical_significance=clinical_sig)
            if test_names and len(test_names) > 1:
                result.multiple_testing_correction = self._apply_multiple_testing_correction([primary_test.p_value], correction_method)
            if validate_assumptions:
                result.normality_tests = self._test_normality_assumptions(control_array, treatment_array)
                result.homogeneity_tests = self._test_homogeneity_assumptions(control_array, treatment_array)
            result.non_parametric_tests = self._perform_non_parametric_tests(control_array, treatment_array)
            result.post_hoc_power = self._calculate_post_hoc_power(control_array, treatment_array, primary_test.effect_size)
            result.prospective_power = self._calculate_prospective_power(control_array, treatment_array)
            if include_bootstrap:
                result.bootstrap_results = self._perform_bootstrap_analysis(control_array, treatment_array)
            if include_sensitivity:
                result.sensitivity_analysis = self._perform_sensitivity_analysis(control_array, treatment_array)
            result.recommendations = self._generate_recommendations(result)
            result.warnings = self._generate_warnings(result)
            result.validation_quality_score = self._calculate_validation_quality_score(result)
            logger.info('Advanced validation completed: %s', validation_id)
            return result
        except Exception as e:
            logger.error('Error in advanced validation: %s', e)
            raise

    def _validate_input_data(self, control: get_numpy().ndarray, treatment: get_numpy().ndarray):
        """Validate input data quality"""
        if len(control) == 0 or len(treatment) == 0:
            raise ValueError('Control and treatment groups cannot be empty')
        if len(control) < 3 or len(treatment) < 3:
            raise ValueError('Minimum 3 observations required per group for statistical analysis')
        if not get_numpy().all(get_numpy().isfinite(control)) or not get_numpy().all(get_numpy().isfinite(treatment)):
            raise ValueError('Data contains infinite or NaN values')
        if get_numpy().var(control) == 0 and get_numpy().var(treatment) == 0 and (get_numpy().mean(control) == get_numpy().mean(treatment)):
            raise ValueError('Both groups have identical constant values - no variation to analyze')

    def _perform_primary_test(self, control: get_numpy().ndarray, treatment: get_numpy().ndarray) -> StatisticalTestResult:
        """Perform primary statistical test (Welch's t-test)"""
        try:
            statistic, p_value = get_scipy_stats().ttest_ind(treatment, control, equal_var=False)
            effect_size = self._calculate_cohens_d(control, treatment)
            confidence_interval = self._calculate_difference_ci(control, treatment)
            n1, n2 = (len(control), len(treatment))
            s1_sq, s2_sq = (get_numpy().var(control, ddof=1), get_numpy().var(treatment, ddof=1))
            if s1_sq == 0 and s2_sq == 0:
                df = n1 + n2 - 2
            else:
                df = (s1_sq / n1 + s2_sq / n2) ** 2 / ((s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1))
            mde = self._calculate_mde(control, treatment)
            return StatisticalTestResult(test_name="Welch's t-test", statistic=float(statistic), p_value=float(p_value), degrees_of_freedom=float(df), effect_size=effect_size, effect_size_type="Cohen's d", confidence_interval=confidence_interval, minimum_detectable_effect=mde, notes=["Welch's t-test assumes unequal variances", 'Robust to heteroscedasticity'])
        except Exception as e:
            logger.error('Error in primary test: %s', e)
            raise

    def _calculate_cohens_d(self, control: get_numpy().ndarray, treatment: get_numpy().ndarray) -> float:
        """Calculate Cohen's d effect size"""
        try:
            mean_diff = get_numpy().mean(treatment) - get_numpy().mean(control)
            n1, n2 = (len(control), len(treatment))
            s1_sq, s2_sq = (get_numpy().var(control, ddof=1), get_numpy().var(treatment, ddof=1))
            pooled_std = get_numpy().sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2))
            if pooled_std == 0:
                if abs(mean_diff) > 1e-10:
                    mean_magnitude = max(abs(get_numpy().mean(control)), abs(get_numpy().mean(treatment)), 1.0)
                    relative_diff = abs(mean_diff) / mean_magnitude
                    if relative_diff > 0.01:
                        return get_numpy().sign(mean_diff) * 10.0
                    return get_numpy().sign(mean_diff) * 0.05
                return 0.0
            return mean_diff / pooled_std
        except Exception as e:
            logger.warning(f"Error calculating Cohen's d: {e}")
            return 0.0

    def _calculate_difference_ci(self, control: get_numpy().ndarray, treatment: get_numpy().ndarray, confidence_level: float=0.95) -> tuple[float, float]:
        """Calculate confidence interval for difference in means"""
        try:
            mean_diff = get_numpy().mean(treatment) - get_numpy().mean(control)
            n1, n2 = (len(control), len(treatment))
            se_control = get_numpy().std(control, ddof=1) / get_numpy().sqrt(n1)
            se_treatment = get_numpy().std(treatment, ddof=1) / get_numpy().sqrt(n2)
            se_diff = get_numpy().sqrt(se_control ** 2 + se_treatment ** 2)
            s1_sq, s2_sq = (get_numpy().var(control, ddof=1), get_numpy().var(treatment, ddof=1))
            df = (s1_sq / n1 + s2_sq / n2) ** 2 / ((s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1))
            alpha = 1 - confidence_level
            t_critical = get_scipy_stats().t.ppf(1 - alpha / 2, df)
            margin_of_error = t_critical * se_diff
            return (mean_diff - margin_of_error, mean_diff + margin_of_error)
        except Exception as e:
            logger.warning('Error calculating confidence interval: %s', e)
            return (0.0, 0.0)

    def _calculate_mde(self, control: get_numpy().ndarray, treatment: get_numpy().ndarray) -> float:
        """Calculate minimum detectable effect size"""
        try:
            n1, n2 = (len(control), len(treatment))
            pooled_var = ((n1 - 1) * get_numpy().var(control, ddof=1) + (n2 - 1) * get_numpy().var(treatment, ddof=1)) / (n1 + n2 - 2)
            se = get_numpy().sqrt(pooled_var * (1 / n1 + 1 / n2))
            t_alpha = get_scipy_stats().t.ppf(1 - self.alpha / 2, n1 + n2 - 2)
            t_beta = get_scipy_stats().t.ppf(self.power_threshold, n1 + n2 - 2)
            mde = (t_alpha + t_beta) * se
            return float(mde)
        except Exception as e:
            logger.warning('Error calculating MDE: %s', e)
            return 0.0

    def _analyze_effect_size(self, control: get_numpy().ndarray, treatment: get_numpy().ndarray, effect_size: float) -> tuple[EffectSizeMagnitude, bool, bool]:
        """Analyze effect size magnitude and significance"""
        try:
            abs_effect = abs(effect_size)
            if get_numpy().isinf(abs_effect) or abs_effect >= 5.0:
                magnitude = EffectSizeMagnitude.VERY_LARGE
            elif abs_effect < 0.1:
                magnitude = EffectSizeMagnitude.negligible
            elif abs_effect < 0.3:
                magnitude = EffectSizeMagnitude.small
            elif abs_effect < 0.5:
                magnitude = EffectSizeMagnitude.MEDIUM
            elif abs_effect < 0.8:
                magnitude = EffectSizeMagnitude.large
            else:
                magnitude = EffectSizeMagnitude.VERY_LARGE
            practical_significance = bool(get_numpy().isinf(abs_effect) or abs_effect >= self.min_effect_size)
            mean_improvement = (get_numpy().mean(treatment) - get_numpy().mean(control)) / get_numpy().mean(control) if get_numpy().mean(control) != 0 else 0
            clinical_significance = bool(abs(mean_improvement) >= 0.1)
            return (magnitude, practical_significance, clinical_significance)
        except Exception as e:
            logger.warning('Error analyzing effect size: %s', e)
            return (EffectSizeMagnitude.negligible, False, False)

    def _apply_multiple_testing_correction(self, p_values: list[float], method: CorrectionMethod) -> MultipleTestingCorrection:
        """Apply multiple testing correction"""
        try:
            p_array = get_numpy().array(p_values)
            rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(p_array, alpha=self.alpha, method=method.value)
            if method == CorrectionMethod.bonferroni:
                fwer = min(len(p_values) * self.alpha, 1.0)
            elif method == CorrectionMethod.sidak:
                fwer = 1 - (1 - self.alpha) ** len(p_values)
            else:
                fwer = self.alpha
            fdr = None
            if method.value.startswith('fdr'):
                fdr = self.alpha
            return MultipleTestingCorrection(method=method, original_p_values=p_values, corrected_p_values=p_corrected.tolist(), rejected=rejected.tolist(), alpha_corrected=alpha_bonf if method == CorrectionMethod.bonferroni else self.alpha, family_wise_error_rate=fwer, false_discovery_rate=fdr)
        except Exception as e:
            logger.error('Error in multiple testing correction: %s', e)
            raise

    def _test_normality_assumptions(self, control: get_numpy().ndarray, treatment: get_numpy().ndarray) -> dict[str, StatisticalTestResult]:
        """Test normality assumptions"""
        results = {}
        try:
            if len(control) < 5000:
                stat_c, p_c = get_scipy_stats().shapiro(control)
                results['shapiro_control'] = StatisticalTestResult(test_name='Shapiro-Wilk (Control)', statistic=float(stat_c), p_value=float(p_c), notes=['H0: Data is normally distributed'])
            if len(treatment) < 5000:
                stat_t, p_t = get_scipy_stats().shapiro(treatment)
                results['shapiro_treatment'] = StatisticalTestResult(test_name='Shapiro-Wilk (Treatment)', statistic=float(stat_t), p_value=float(p_t), notes=['H0: Data is normally distributed'])
            stat_c, p_c = get_scipy_stats().kstest(control, 'norm', args=(get_numpy().mean(control), get_numpy().std(control, ddof=1)))
            results['ks_control'] = StatisticalTestResult(test_name='Kolmogorov-Smirnov (Control)', statistic=float(stat_c), p_value=float(p_c), notes=['H0: Data follows normal distribution'])
            stat_t, p_t = get_scipy_stats().kstest(treatment, 'norm', args=(get_numpy().mean(treatment), get_numpy().std(treatment, ddof=1)))
            results['ks_treatment'] = StatisticalTestResult(test_name='Kolmogorov-Smirnov (Treatment)', statistic=float(stat_t), p_value=float(p_t), notes=['H0: Data follows normal distribution'])
        except Exception as e:
            logger.warning('Error in normality testing: %s', e)
        return results

    def _test_homogeneity_assumptions(self, control: get_numpy().ndarray, treatment: get_numpy().ndarray) -> dict[str, StatisticalTestResult]:
        """Test homogeneity of variance assumptions"""
        results = {}
        try:
            stat, p_value = get_scipy_stats().levene(control, treatment)
            results['levene'] = StatisticalTestResult(test_name="Levene's Test", statistic=float(stat), p_value=float(p_value), notes=['H0: Equal variances', 'Robust to non-normality'])
            stat, p_value = get_scipy_stats().bartlett(control, treatment)
            results['bartlett'] = StatisticalTestResult(test_name="Bartlett's Test", statistic=float(stat), p_value=float(p_value), notes=['H0: Equal variances', 'Assumes normality'])
        except Exception as e:
            logger.warning('Error in homogeneity testing: %s', e)
        return results

    def _perform_non_parametric_tests(self, control: get_numpy().ndarray, treatment: get_numpy().ndarray) -> dict[str, StatisticalTestResult]:
        """Perform non-parametric robustness checks"""
        results = {}
        try:
            stat, p_value = mannwhitneyu(treatment, control, alternative='two-sided')
            results['mann_whitney'] = StatisticalTestResult(test_name='Mann-Whitney U', statistic=float(stat), p_value=float(p_value), notes=['Non-parametric', 'Tests median differences', 'Robust to outliers'])
            n1, n2 = (len(control), len(treatment))
            effect_size = 1 - 2 * stat / (n1 * n2)
            results['mann_whitney'].effect_size = effect_size
            results['mann_whitney'].effect_size_type = 'Rank-biserial correlation'
        except Exception as e:
            logger.warning('Error in non-parametric testing: %s', e)
        return results

    def _calculate_post_hoc_power(self, control: get_numpy().ndarray, treatment: get_numpy().ndarray, effect_size: float) -> float:
        """Calculate post-hoc statistical power"""
        try:
            n1, n2 = (len(control), len(treatment))
            power = ttest_power(effect_size, n1, self.alpha, alternative='two-sided')
            return float(min(max(power, 0.0), 1.0))
        except Exception as e:
            logger.warning('Error calculating post-hoc power: %s', e)
            return 0.0

    def _calculate_prospective_power(self, control: get_numpy().ndarray, treatment: get_numpy().ndarray) -> float:
        """Calculate prospective power for minimum effect size"""
        try:
            n1, n2 = (len(control), len(treatment))
            power = ttest_power(self.min_effect_size, n1, self.alpha, alternative='two-sided')
            return float(min(max(power, 0.0), 1.0))
        except Exception as e:
            logger.warning('Error calculating prospective power: %s', e)
            return 0.0

    def _perform_bootstrap_analysis(self, control: get_numpy().ndarray, treatment: get_numpy().ndarray) -> dict[str, Any]:
        """Perform bootstrap analysis for robust confidence intervals"""
        try:
            bootstrap_diffs = []
            bootstrap_effect_sizes = []
            for _ in range(self.bootstrap_samples):
                control_boot = get_numpy().random.choice(control, size=len(control), replace=True)
                treatment_boot = get_numpy().random.choice(treatment, size=len(treatment), replace=True)
                diff = get_numpy().mean(treatment_boot) - get_numpy().mean(control_boot)
                bootstrap_diffs.append(diff)
                effect_size = self._calculate_cohens_d(control_boot, treatment_boot)
                bootstrap_effect_sizes.append(effect_size)
            bootstrap_diffs = get_numpy().array(bootstrap_diffs)
            bootstrap_effect_sizes = get_numpy().array(bootstrap_effect_sizes)
            ci_lower_diff = get_numpy().percentile(bootstrap_diffs, 2.5)
            ci_upper_diff = get_numpy().percentile(bootstrap_diffs, 97.5)
            ci_lower_effect = get_numpy().percentile(bootstrap_effect_sizes, 2.5)
            ci_upper_effect = get_numpy().percentile(bootstrap_effect_sizes, 97.5)
            return {'n_bootstrap_samples': self.bootstrap_samples, 'mean_difference': {'bootstrap_mean': float(get_numpy().mean(bootstrap_diffs)), 'bootstrap_std': float(get_numpy().std(bootstrap_diffs)), 'confidence_interval_95': [float(ci_lower_diff), float(ci_upper_diff)]}, 'effect_size': {'bootstrap_mean': float(get_numpy().mean(bootstrap_effect_sizes)), 'bootstrap_std': float(get_numpy().std(bootstrap_effect_sizes)), 'confidence_interval_95': [float(ci_lower_effect), float(ci_upper_effect)]}}
        except Exception as e:
            logger.warning('Error in bootstrap analysis: %s', e)
            return {}

    def _perform_sensitivity_analysis(self, control: get_numpy().ndarray, treatment: get_numpy().ndarray) -> dict[str, Any]:
        """Perform sensitivity analysis by removing outliers"""
        try:

            def remove_outliers(data, multiplier=1.5):
                q1 = get_numpy().percentile(data, 25)
                q3 = get_numpy().percentile(data, 75)
                iqr = q3 - q1
                lower = q1 - multiplier * iqr
                upper = q3 + multiplier * iqr
                return data[(data >= lower) & (data <= upper)]
            control_clean = remove_outliers(control)
            treatment_clean = remove_outliers(treatment)
            if len(control_clean) < 3 or len(treatment_clean) < 3:
                return {'error': 'Insufficient data after outlier removal'}
            original_effect = self._calculate_cohens_d(control, treatment)
            cleaned_effect = self._calculate_cohens_d(control_clean, treatment_clean)
            _, original_p = get_scipy_stats().ttest_ind(treatment, control, equal_var=False)
            _, cleaned_p = get_scipy_stats().ttest_ind(treatment_clean, control_clean, equal_var=False)
            return {'outliers_removed': {'control': len(control) - len(control_clean), 'treatment': len(treatment) - len(treatment_clean), 'total': len(control) + len(treatment) - (len(control_clean) + len(treatment_clean))}, 'effect_size_comparison': {'original': float(original_effect), 'cleaned': float(cleaned_effect), 'difference': float(abs(original_effect - cleaned_effect)), 'robust_to_outliers': abs(original_effect - cleaned_effect) < 0.1}, 'p_value_comparison': {'original': float(original_p), 'cleaned': float(cleaned_p), 'robust_to_outliers': abs(original_p - cleaned_p) < 0.01}}
        except Exception as e:
            logger.warning('Error in sensitivity analysis: %s', e)
            return {}

    def _generate_recommendations(self, result: AdvancedValidationResult) -> list[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        try:
            if result.primary_test.p_value < self.alpha:
                if result.practical_significance:
                    recommendations.append('✅ DEPLOY: Statistically and practically significant improvement detected')
                else:
                    recommendations.append('⚠️ CAUTION: Statistically significant but effect size below practical threshold')
            else:
                recommendations.append('❌ DO NOT DEPLOY: No statistical significance detected')
            if result.effect_size_magnitude == EffectSizeMagnitude.VERY_LARGE:
                recommendations.append('🎯 Excellent effect size - strong business impact expected')
            elif result.effect_size_magnitude == EffectSizeMagnitude.large:
                recommendations.append('📈 Good effect size - meaningful business impact expected')
            elif result.effect_size_magnitude == EffectSizeMagnitude.MEDIUM:
                recommendations.append('📊 Moderate effect size - monitor performance closely')
            else:
                recommendations.append('📉 Small effect size - consider testing larger changes')
            if result.post_hoc_power and result.post_hoc_power < self.power_threshold:
                recommendations.append(f'⚡ Low statistical power ({result.post_hoc_power:.2f}) - consider increasing sample size')
            if result.normality_tests:
                normal_violations = [test for test in result.normality_tests.values() if test.p_value < 0.05]
                if normal_violations:
                    recommendations.append('📋 Normality assumption violated - non-parametric results may be more reliable')
            if result.bootstrap_results and result.primary_test.confidence_interval:
                bootstrap_ci = result.bootstrap_results.get('mean_difference', {}).get('confidence_interval_95', [])
                parametric_ci = result.primary_test.confidence_interval
                if bootstrap_ci and parametric_ci:
                    if abs(bootstrap_ci[0] - parametric_ci[0]) > 0.1 or abs(bootstrap_ci[1] - parametric_ci[1]) > 0.1:
                        recommendations.append('🔍 Bootstrap and parametric results differ - investigate further')
        except Exception as e:
            logger.warning('Error generating recommendations: %s', e)
            recommendations.append('⚠️ Could not generate complete recommendations due to analysis errors')
        return recommendations

    def _generate_warnings(self, result: AdvancedValidationResult) -> list[str]:
        """Generate warnings based on validation results"""
        warnings = []
        try:
            total_n = result.primary_test.degrees_of_freedom + 2 if result.primary_test.degrees_of_freedom else 0
            if total_n < 30:
                warnings.append('⚠️ Small sample size may affect reliability of statistical tests')
            if result.multiple_testing_correction:
                if not any(result.multiple_testing_correction.rejected):
                    warnings.append('⚠️ No significant results after multiple testing correction')
            if result.normality_tests:
                violations = sum(1 for test in result.normality_tests.values() if test.p_value < 0.05)
                if violations > 0:
                    warnings.append(f'⚠️ {violations} normality assumption violation(s) detected')
            if result.homogeneity_tests:
                violations = sum(1 for test in result.homogeneity_tests.values() if test.p_value < 0.05)
                if violations > 0:
                    warnings.append(f'⚠️ {violations} homogeneity assumption violation(s) detected')
            if result.sensitivity_analysis and 'effect_size_comparison' in result.sensitivity_analysis:
                if not result.sensitivity_analysis['effect_size_comparison'].get('robust_to_outliers', True):
                    warnings.append('⚠️ Results sensitive to outliers - interpret with caution')
        except Exception as e:
            logger.warning('Error generating warnings: %s', e)
            warnings.append('⚠️ Could not generate complete warnings due to analysis errors')
        return warnings

    def _calculate_validation_quality_score(self, result: AdvancedValidationResult) -> float:
        """Calculate overall validation quality score (0-1)"""
        try:
            score_components = []
            if result.primary_test.p_value is not None:
                rigor_score = 1.0 if result.primary_test.p_value < self.alpha else 0.5
                score_components.append(('rigor', rigor_score, 0.3))
            if result.effect_size_magnitude:
                effect_scores = {EffectSizeMagnitude.negligible: 0.1, EffectSizeMagnitude.small: 0.4, EffectSizeMagnitude.MEDIUM: 0.7, EffectSizeMagnitude.large: 0.9, EffectSizeMagnitude.VERY_LARGE: 1.0}
                effect_score = effect_scores.get(result.effect_size_magnitude, 0.0)
                score_components.append(('effect', effect_score, 0.25))
            if result.post_hoc_power:
                power_score = min(result.post_hoc_power / self.power_threshold, 1.0)
                score_components.append(('power', power_score, 0.2))
            assumption_score = 1.0
            if result.normality_tests:
                violations = sum(1 for test in result.normality_tests.values() if test.p_value < 0.05)
                assumption_score *= max(0.5, 1.0 - violations * 0.2)
            score_components.append(('assumptions', assumption_score, 0.15))
            robustness_score = 1.0
            if result.sensitivity_analysis and 'effect_size_comparison' in result.sensitivity_analysis:
                if not result.sensitivity_analysis['effect_size_comparison'].get('robust_to_outliers', True):
                    robustness_score = 0.7
            score_components.append(('robustness', robustness_score, 0.1))
            total_score = sum((score * weight for _, score, weight in score_components))
            total_weight = sum((weight for _, _, weight in score_components))
            return total_score / total_weight if total_weight > 0 else 0.0
        except Exception as e:
            logger.warning('Error calculating validation quality score: %s', e)
            return 0.0

def quick_validation(control_data: list[float], treatment_data: list[float], alpha: float=0.05) -> dict[str, Any]:
    """Quick validation for immediate use"""
    validator = AdvancedStatisticalValidator(alpha=alpha)
    result = validator.validate_ab_test(control_data, treatment_data, validate_assumptions=False, include_bootstrap=False, include_sensitivity=False)
    return {'statistically_significant': result.primary_test.p_value < alpha, 'p_value': result.primary_test.p_value, 'effect_size': result.primary_test.effect_size, 'effect_magnitude': result.effect_size_magnitude.value, 'practical_significance': result.practical_significance, 'recommendations': result.recommendations, 'quality_score': result.validation_quality_score}