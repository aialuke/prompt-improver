"""Advanced Statistical Validator for A/B Testing
Implements 2025 best practices for statistical validation and multiple testing corrections
"""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import ttest_power
from statsmodels.stats.weightstats import ztest

logger = logging.getLogger(__name__)


class CorrectionMethod(Enum):
    """Multiple testing correction methods"""

    BONFERRONI = "bonferroni"
    HOLM = "holm"  # Step-down Holm-Bonferroni
    SIDAK = "sidak"
    BENJAMINI_HOCHBERG = "fdr_bh"  # FDR Benjamini-Hochberg
    BENJAMINI_YEKUTIELI = "fdr_by"  # FDR Benjamini-Yekutieli
    FALSE_DISCOVERY_RATE = "fdr_tsbh"  # Two-stage FDR


class EffectSizeMagnitude(Enum):
    """Effect size magnitude classifications"""

    NEGLIGIBLE = "negligible"  # < 0.1
    SMALL = "small"  # 0.1 - 0.3
    MEDIUM = "medium"  # 0.3 - 0.5
    LARGE = "large"  # 0.5 - 0.8
    VERY_LARGE = "very_large"  # > 0.8


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
    rejected: list[bool]  # Which null hypotheses are rejected
    alpha_corrected: float
    family_wise_error_rate: float
    false_discovery_rate: float | None = None


@dataclass
class AdvancedValidationResult:
    """Comprehensive validation result with 2025 best practices"""

    validation_id: str
    timestamp: datetime

    # Primary analysis
    primary_test: StatisticalTestResult

    # Effect size analysis
    effect_size_magnitude: EffectSizeMagnitude
    practical_significance: bool
    clinical_significance: bool

    # Multiple testing correction
    multiple_testing_correction: MultipleTestingCorrection | None = None

    # Sensitivity analysis
    sensitivity_analysis: dict[str, Any] | None = None

    # Power analysis
    post_hoc_power: float | None = None
    prospective_power: float | None = None

    # Assumptions testing
    normality_tests: dict[str, StatisticalTestResult] | None = None
    homogeneity_tests: dict[str, StatisticalTestResult] | None = None

    # Robustness checks
    non_parametric_tests: dict[str, StatisticalTestResult] | None = None

    # Bootstrap analysis
    bootstrap_results: dict[str, Any] | None = None

    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Quality metrics
    validation_quality_score: float = 0.0


class AdvancedStatisticalValidator:
    """Advanced statistical validator implementing 2025 best practices"""

    def __init__(
        self,
        alpha: float = 0.05,
        power_threshold: float = 0.8,
        min_effect_size: float = 0.1,
        bootstrap_samples: int = 10000,
    ):
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

    async def run_orchestrated_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
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
            # Extract configuration from orchestrator
            control_data = config.get("control_data", [])
            treatment_data = config.get("treatment_data", [])
            test_names = config.get("test_names", None)
            correction_method_str = config.get("correction_method", "benjamini_hochberg")
            validate_assumptions = config.get("validate_assumptions", True)
            include_bootstrap = config.get("include_bootstrap", True)
            include_sensitivity = config.get("include_sensitivity", True)
            output_path = config.get("output_path", "./outputs/statistical_validation")

            # Validate input data
            if not control_data or not treatment_data:
                raise ValueError("Both control_data and treatment_data are required")

            # Convert correction method string to enum
            try:
                correction_method = CorrectionMethod(correction_method_str)
            except ValueError:
                correction_method = CorrectionMethod.BENJAMINI_HOCHBERG
                logger.warning(f"Unknown correction method '{correction_method_str}', using benjamini_hochberg")

            # Perform statistical validation using existing method
            validation_result = self.validate_ab_test(
                control_data=control_data,
                treatment_data=treatment_data,
                test_names=test_names,
                correction_method=correction_method,
                validate_assumptions=validate_assumptions,
                include_bootstrap=include_bootstrap,
                include_sensitivity=include_sensitivity
            )

            # Prepare orchestrator-compatible result
            result = {
                "primary_test": {
                    "test_name": validation_result.primary_test.test_name,
                    "statistic": validation_result.primary_test.statistic,
                    "p_value": validation_result.primary_test.p_value,
                    "effect_size": validation_result.primary_test.effect_size,
                    "effect_size_type": validation_result.primary_test.effect_size_type,
                    "confidence_interval": validation_result.primary_test.confidence_interval,
                    "power": validation_result.primary_test.power,
                    "assumptions_met": validation_result.primary_test.assumptions_met
                },
                "effect_size_analysis": {
                    "magnitude": validation_result.effect_size_magnitude.value,
                    "practical_significance": validation_result.practical_significance,
                    "clinical_significance": validation_result.clinical_significance
                },
                "multiple_testing": {
                    "correction_applied": validation_result.multiple_testing_correction is not None,
                    "method": correction_method.value,
                    "family_wise_error_rate": validation_result.multiple_testing_correction.family_wise_error_rate if validation_result.multiple_testing_correction else None,
                    "false_discovery_rate": validation_result.multiple_testing_correction.false_discovery_rate if validation_result.multiple_testing_correction else None
                },
                "power_analysis": {
                    "post_hoc_power": validation_result.post_hoc_power,
                    "prospective_power": validation_result.prospective_power
                },
                "quality_assessment": {
                    "validation_quality_score": validation_result.validation_quality_score,
                    "recommendations": validation_result.recommendations,
                    "warnings": validation_result.warnings
                }
            }

            # Calculate execution metadata
            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "orchestrator_compatible": True,
                "component_result": result,
                "local_metadata": {
                    "output_path": output_path,
                    "execution_time": execution_time,
                    "validation_id": validation_result.validation_id,
                    "control_sample_size": len(control_data),
                    "treatment_sample_size": len(treatment_data),
                    "correction_method": correction_method.value,
                    "assumptions_validated": validate_assumptions,
                    "bootstrap_included": include_bootstrap,
                    "sensitivity_included": include_sensitivity,
                    "component_version": "1.0.0"
                }
            }

        except ValueError as e:
            logger.error(f"Validation error in orchestrated statistical validation: {e}")
            return {
                "orchestrator_compatible": True,
                "component_result": {"error": f"Validation error: {str(e)}", "primary_test": {}},
                "local_metadata": {
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "error": True,
                    "error_type": "validation",
                    "component_version": "1.0.0"
                }
            }
        except Exception as e:
            logger.error(f"Orchestrated statistical validation failed: {e}")
            return {
                "orchestrator_compatible": True,
                "component_result": {"error": str(e), "primary_test": {}},
                "local_metadata": {
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "error": True,
                    "component_version": "1.0.0"
                }
            }

    def validate_ab_test(
        self,
        control_data: list[float],
        treatment_data: list[float],
        test_names: list[str] | None = None,
        correction_method: CorrectionMethod = CorrectionMethod.BENJAMINI_HOCHBERG,
        validate_assumptions: bool = True,
        include_bootstrap: bool = True,
        include_sensitivity: bool = True,
    ) -> AdvancedValidationResult:
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
            # Convert to numpy arrays
            control_array = np.array(control_data, dtype=float)
            treatment_array = np.array(treatment_data, dtype=float)

            # Validate input data
            self._validate_input_data(control_array, treatment_array)

            # Primary statistical analysis
            primary_test = self._perform_primary_test(control_array, treatment_array)

            # Effect size analysis
            effect_size_magnitude, practical_sig, clinical_sig = (
                self._analyze_effect_size(
                    control_array, treatment_array, primary_test.effect_size
                )
            )

            # Initialize result
            result = AdvancedValidationResult(
                validation_id=validation_id,
                timestamp=datetime.utcnow(),
                primary_test=primary_test,
                effect_size_magnitude=effect_size_magnitude,
                practical_significance=practical_sig,
                clinical_significance=clinical_sig,
            )

            # Multiple testing correction (if applicable)
            if test_names and len(test_names) > 1:
                result.multiple_testing_correction = (
                    self._apply_multiple_testing_correction(
                        [primary_test.p_value], correction_method
                    )
                )

            # Assumption testing
            if validate_assumptions:
                result.normality_tests = self._test_normality_assumptions(
                    control_array, treatment_array
                )
                result.homogeneity_tests = self._test_homogeneity_assumptions(
                    control_array, treatment_array
                )

            # Non-parametric robustness checks
            result.non_parametric_tests = self._perform_non_parametric_tests(
                control_array, treatment_array
            )

            # Power analysis
            result.post_hoc_power = self._calculate_post_hoc_power(
                control_array, treatment_array, primary_test.effect_size
            )
            result.prospective_power = self._calculate_prospective_power(
                control_array, treatment_array
            )

            # Bootstrap analysis
            if include_bootstrap:
                result.bootstrap_results = self._perform_bootstrap_analysis(
                    control_array, treatment_array
                )

            # Sensitivity analysis
            if include_sensitivity:
                result.sensitivity_analysis = self._perform_sensitivity_analysis(
                    control_array, treatment_array
                )

            # Generate recommendations and warnings
            result.recommendations = self._generate_recommendations(result)
            result.warnings = self._generate_warnings(result)

            # Calculate validation quality score
            result.validation_quality_score = self._calculate_validation_quality_score(
                result
            )

            logger.info(f"Advanced validation completed: {validation_id}")

            return result

        except Exception as e:
            logger.error(f"Error in advanced validation: {e}")
            raise

    def _validate_input_data(self, control: np.ndarray, treatment: np.ndarray):
        """Validate input data quality"""
        if len(control) == 0 or len(treatment) == 0:
            raise ValueError("Control and treatment groups cannot be empty")

        if len(control) < 3 or len(treatment) < 3:
            raise ValueError(
                "Minimum 3 observations required per group for statistical analysis"
            )

        # Check for infinite or NaN values
        if not np.all(np.isfinite(control)) or not np.all(np.isfinite(treatment)):
            raise ValueError("Data contains infinite or NaN values")

        # Check for constant values (zero variance) - only error if both are constant AND identical
        if (
            np.var(control) == 0
            and np.var(treatment) == 0
            and np.mean(control) == np.mean(treatment)
        ):
            raise ValueError(
                "Both groups have identical constant values - no variation to analyze"
            )

    def _perform_primary_test(
        self, control: np.ndarray, treatment: np.ndarray
    ) -> StatisticalTestResult:
        """Perform primary statistical test (Welch's t-test)"""
        try:
            # Welch's t-test (unequal variances assumed)
            statistic, p_value = stats.ttest_ind(treatment, control, equal_var=False)

            # Calculate effect size (Cohen's d)
            effect_size = self._calculate_cohens_d(control, treatment)

            # Calculate confidence interval for difference in means
            confidence_interval = self._calculate_difference_ci(control, treatment)

            # Calculate degrees of freedom for Welch's test
            n1, n2 = len(control), len(treatment)
            s1_sq, s2_sq = np.var(control, ddof=1), np.var(treatment, ddof=1)

            if s1_sq == 0 and s2_sq == 0:
                df = n1 + n2 - 2
            else:
                df = (s1_sq / n1 + s2_sq / n2) ** 2 / (
                    (s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1)
                )

            # Calculate minimum detectable effect
            mde = self._calculate_mde(control, treatment)

            return StatisticalTestResult(
                test_name="Welch's t-test",
                statistic=float(statistic),
                p_value=float(p_value),
                degrees_of_freedom=float(df),
                effect_size=effect_size,
                effect_size_type="Cohen's d",
                confidence_interval=confidence_interval,
                minimum_detectable_effect=mde,
                notes=[
                    "Welch's t-test assumes unequal variances",
                    "Robust to heteroscedasticity",
                ],
            )

        except Exception as e:
            logger.error(f"Error in primary test: {e}")
            raise

    def _calculate_cohens_d(self, control: np.ndarray, treatment: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        try:
            mean_diff = np.mean(treatment) - np.mean(control)

            # Pooled standard deviation
            n1, n2 = len(control), len(treatment)
            s1_sq, s2_sq = np.var(control, ddof=1), np.var(treatment, ddof=1)

            pooled_std = np.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2))

            if pooled_std == 0:
                # If both groups are constant but have different means
                if (
                    abs(mean_diff) > 1e-10
                ):  # Use a reasonable threshold for numerical precision
                    # For constant groups, assess if the difference is meaningful relative to the scale
                    mean_magnitude = max(
                        abs(np.mean(control)), abs(np.mean(treatment)), 1.0
                    )
                    relative_diff = abs(mean_diff) / mean_magnitude

                    if relative_diff > 0.01:  # 1% relative difference is meaningful
                        return np.sign(mean_diff) * 10.0  # Large effect size
                    # Very small relative difference - treat as negligible
                    return np.sign(mean_diff) * 0.05  # Small effect size
                # If both groups are identical (within numerical precision), no effect
                return 0.0

            return mean_diff / pooled_std

        except Exception as e:
            logger.warning(f"Error calculating Cohen's d: {e}")
            return 0.0

    def _calculate_difference_ci(
        self, control: np.ndarray, treatment: np.ndarray, confidence_level: float = 0.95
    ) -> tuple[float, float]:
        """Calculate confidence interval for difference in means"""
        try:
            mean_diff = np.mean(treatment) - np.mean(control)

            # Standard error of difference
            n1, n2 = len(control), len(treatment)
            se_control = np.std(control, ddof=1) / np.sqrt(n1)
            se_treatment = np.std(treatment, ddof=1) / np.sqrt(n2)
            se_diff = np.sqrt(se_control**2 + se_treatment**2)

            # Degrees of freedom (Welch's)
            s1_sq, s2_sq = np.var(control, ddof=1), np.var(treatment, ddof=1)
            df = (s1_sq / n1 + s2_sq / n2) ** 2 / (
                (s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1)
            )

            # Critical value
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha / 2, df)

            margin_of_error = t_critical * se_diff

            return (mean_diff - margin_of_error, mean_diff + margin_of_error)

        except Exception as e:
            logger.warning(f"Error calculating confidence interval: {e}")
            return (0.0, 0.0)

    def _calculate_mde(self, control: np.ndarray, treatment: np.ndarray) -> float:
        """Calculate minimum detectable effect size"""
        try:
            n1, n2 = len(control), len(treatment)
            pooled_var = (
                (n1 - 1) * np.var(control, ddof=1)
                + (n2 - 1) * np.var(treatment, ddof=1)
            ) / (n1 + n2 - 2)

            se = np.sqrt(pooled_var * (1 / n1 + 1 / n2))

            # Critical values for alpha and beta
            t_alpha = stats.t.ppf(1 - self.alpha / 2, n1 + n2 - 2)
            t_beta = stats.t.ppf(self.power_threshold, n1 + n2 - 2)

            mde = (t_alpha + t_beta) * se

            return float(mde)

        except Exception as e:
            logger.warning(f"Error calculating MDE: {e}")
            return 0.0

    def _analyze_effect_size(
        self, control: np.ndarray, treatment: np.ndarray, effect_size: float
    ) -> tuple[EffectSizeMagnitude, bool, bool]:
        """Analyze effect size magnitude and significance"""
        try:
            # Classify effect size magnitude
            abs_effect = abs(effect_size)

            # Handle very large effect sizes (constant groups with different means)
            if np.isinf(abs_effect) or abs_effect >= 5.0:
                magnitude = EffectSizeMagnitude.VERY_LARGE
            elif abs_effect < 0.1:
                magnitude = EffectSizeMagnitude.NEGLIGIBLE
            elif abs_effect < 0.3:
                magnitude = EffectSizeMagnitude.SMALL
            elif abs_effect < 0.5:
                magnitude = EffectSizeMagnitude.MEDIUM
            elif abs_effect < 0.8:
                magnitude = EffectSizeMagnitude.LARGE
            else:
                magnitude = EffectSizeMagnitude.VERY_LARGE

            # Practical significance (very large effects are always practically significant)
            practical_significance = bool(
                np.isinf(abs_effect) or abs_effect >= self.min_effect_size
            )

            # Clinical significance (domain-specific thresholds)
            # For prompt improvement, we might consider 10% improvement clinically significant
            mean_improvement = (
                (np.mean(treatment) - np.mean(control)) / np.mean(control)
                if np.mean(control) != 0
                else 0
            )
            clinical_significance = bool(
                abs(mean_improvement) >= 0.1
            )  # 10% relative improvement

            return magnitude, practical_significance, clinical_significance

        except Exception as e:
            logger.warning(f"Error analyzing effect size: {e}")
            return EffectSizeMagnitude.NEGLIGIBLE, False, False

    def _apply_multiple_testing_correction(
        self, p_values: list[float], method: CorrectionMethod
    ) -> MultipleTestingCorrection:
        """Apply multiple testing correction"""
        try:
            p_array = np.array(p_values)

            # Apply correction
            rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                p_array, alpha=self.alpha, method=method.value
            )

            # Calculate family-wise error rate
            if method == CorrectionMethod.BONFERRONI:
                fwer = min(len(p_values) * self.alpha, 1.0)
            elif method == CorrectionMethod.SIDAK:
                fwer = 1 - (1 - self.alpha) ** len(p_values)
            else:
                fwer = self.alpha  # For FDR methods

            # Calculate false discovery rate for FDR methods
            fdr = None
            if method.value.startswith("fdr"):
                fdr = self.alpha  # Expected FDR level

            return MultipleTestingCorrection(
                method=method,
                original_p_values=p_values,
                corrected_p_values=p_corrected.tolist(),
                rejected=rejected.tolist(),
                alpha_corrected=alpha_bonf
                if method == CorrectionMethod.BONFERRONI
                else self.alpha,
                family_wise_error_rate=fwer,
                false_discovery_rate=fdr,
            )

        except Exception as e:
            logger.error(f"Error in multiple testing correction: {e}")
            raise

    def _test_normality_assumptions(
        self, control: np.ndarray, treatment: np.ndarray
    ) -> dict[str, StatisticalTestResult]:
        """Test normality assumptions"""
        results = {}

        try:
            # Shapiro-Wilk test (for n < 5000)
            if len(control) < 5000:
                stat_c, p_c = stats.shapiro(control)
                results["shapiro_control"] = StatisticalTestResult(
                    test_name="Shapiro-Wilk (Control)",
                    statistic=float(stat_c),
                    p_value=float(p_c),
                    notes=["H0: Data is normally distributed"],
                )

            if len(treatment) < 5000:
                stat_t, p_t = stats.shapiro(treatment)
                results["shapiro_treatment"] = StatisticalTestResult(
                    test_name="Shapiro-Wilk (Treatment)",
                    statistic=float(stat_t),
                    p_value=float(p_t),
                    notes=["H0: Data is normally distributed"],
                )

            # Kolmogorov-Smirnov test
            stat_c, p_c = stats.kstest(
                control, "norm", args=(np.mean(control), np.std(control, ddof=1))
            )
            results["ks_control"] = StatisticalTestResult(
                test_name="Kolmogorov-Smirnov (Control)",
                statistic=float(stat_c),
                p_value=float(p_c),
                notes=["H0: Data follows normal distribution"],
            )

            stat_t, p_t = stats.kstest(
                treatment, "norm", args=(np.mean(treatment), np.std(treatment, ddof=1))
            )
            results["ks_treatment"] = StatisticalTestResult(
                test_name="Kolmogorov-Smirnov (Treatment)",
                statistic=float(stat_t),
                p_value=float(p_t),
                notes=["H0: Data follows normal distribution"],
            )

        except Exception as e:
            logger.warning(f"Error in normality testing: {e}")

        return results

    def _test_homogeneity_assumptions(
        self, control: np.ndarray, treatment: np.ndarray
    ) -> dict[str, StatisticalTestResult]:
        """Test homogeneity of variance assumptions"""
        results = {}

        try:
            # Levene's test
            stat, p_value = stats.levene(control, treatment)
            results["levene"] = StatisticalTestResult(
                test_name="Levene's Test",
                statistic=float(stat),
                p_value=float(p_value),
                notes=["H0: Equal variances", "Robust to non-normality"],
            )

            # Bartlett's test (sensitive to non-normality)
            stat, p_value = stats.bartlett(control, treatment)
            results["bartlett"] = StatisticalTestResult(
                test_name="Bartlett's Test",
                statistic=float(stat),
                p_value=float(p_value),
                notes=["H0: Equal variances", "Assumes normality"],
            )

        except Exception as e:
            logger.warning(f"Error in homogeneity testing: {e}")

        return results

    def _perform_non_parametric_tests(
        self, control: np.ndarray, treatment: np.ndarray
    ) -> dict[str, StatisticalTestResult]:
        """Perform non-parametric robustness checks"""
        results = {}

        try:
            # Mann-Whitney U test
            stat, p_value = mannwhitneyu(treatment, control, alternative="two-sided")
            results["mann_whitney"] = StatisticalTestResult(
                test_name="Mann-Whitney U",
                statistic=float(stat),
                p_value=float(p_value),
                notes=[
                    "Non-parametric",
                    "Tests median differences",
                    "Robust to outliers",
                ],
            )

            # Wilcoxon rank-sum test (equivalent to Mann-Whitney)
            # We'll compute rank-biserial correlation as effect size
            n1, n2 = len(control), len(treatment)
            effect_size = 1 - (2 * stat) / (n1 * n2)  # Rank-biserial correlation
            results["mann_whitney"].effect_size = effect_size
            results["mann_whitney"].effect_size_type = "Rank-biserial correlation"

        except Exception as e:
            logger.warning(f"Error in non-parametric testing: {e}")

        return results

    def _calculate_post_hoc_power(
        self, control: np.ndarray, treatment: np.ndarray, effect_size: float
    ) -> float:
        """Calculate post-hoc statistical power"""
        try:
            n1, n2 = len(control), len(treatment)

            # Use the observed effect size to calculate power
            power = ttest_power(effect_size, n1, self.alpha, alternative="two-sided")

            return float(min(max(power, 0.0), 1.0))

        except Exception as e:
            logger.warning(f"Error calculating post-hoc power: {e}")
            return 0.0

    def _calculate_prospective_power(
        self, control: np.ndarray, treatment: np.ndarray
    ) -> float:
        """Calculate prospective power for minimum effect size"""
        try:
            n1, n2 = len(control), len(treatment)

            # Use minimum effect size threshold
            power = ttest_power(
                self.min_effect_size, n1, self.alpha, alternative="two-sided"
            )

            return float(min(max(power, 0.0), 1.0))

        except Exception as e:
            logger.warning(f"Error calculating prospective power: {e}")
            return 0.0

    def _perform_bootstrap_analysis(
        self, control: np.ndarray, treatment: np.ndarray
    ) -> dict[str, Any]:
        """Perform bootstrap analysis for robust confidence intervals"""
        try:
            bootstrap_diffs = []
            bootstrap_effect_sizes = []

            for _ in range(self.bootstrap_samples):
                # Bootstrap resample
                control_boot = np.random.choice(
                    control, size=len(control), replace=True
                )
                treatment_boot = np.random.choice(
                    treatment, size=len(treatment), replace=True
                )

                # Calculate statistics
                diff = np.mean(treatment_boot) - np.mean(control_boot)
                bootstrap_diffs.append(diff)

                effect_size = self._calculate_cohens_d(control_boot, treatment_boot)
                bootstrap_effect_sizes.append(effect_size)

            bootstrap_diffs = np.array(bootstrap_diffs)
            bootstrap_effect_sizes = np.array(bootstrap_effect_sizes)

            # Calculate percentile confidence intervals
            ci_lower_diff = np.percentile(bootstrap_diffs, 2.5)
            ci_upper_diff = np.percentile(bootstrap_diffs, 97.5)

            ci_lower_effect = np.percentile(bootstrap_effect_sizes, 2.5)
            ci_upper_effect = np.percentile(bootstrap_effect_sizes, 97.5)

            return {
                "n_bootstrap_samples": self.bootstrap_samples,
                "mean_difference": {
                    "bootstrap_mean": float(np.mean(bootstrap_diffs)),
                    "bootstrap_std": float(np.std(bootstrap_diffs)),
                    "confidence_interval_95": [
                        float(ci_lower_diff),
                        float(ci_upper_diff),
                    ],
                },
                "effect_size": {
                    "bootstrap_mean": float(np.mean(bootstrap_effect_sizes)),
                    "bootstrap_std": float(np.std(bootstrap_effect_sizes)),
                    "confidence_interval_95": [
                        float(ci_lower_effect),
                        float(ci_upper_effect),
                    ],
                },
            }

        except Exception as e:
            logger.warning(f"Error in bootstrap analysis: {e}")
            return {}

    def _perform_sensitivity_analysis(
        self, control: np.ndarray, treatment: np.ndarray
    ) -> dict[str, Any]:
        """Perform sensitivity analysis by removing outliers"""
        try:
            # Remove outliers using IQR method
            def remove_outliers(data, multiplier=1.5):
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                lower = q1 - multiplier * iqr
                upper = q3 + multiplier * iqr
                return data[(data >= lower) & (data <= upper)]

            control_clean = remove_outliers(control)
            treatment_clean = remove_outliers(treatment)

            if len(control_clean) < 3 or len(treatment_clean) < 3:
                return {"error": "Insufficient data after outlier removal"}

            # Recalculate key statistics
            original_effect = self._calculate_cohens_d(control, treatment)
            cleaned_effect = self._calculate_cohens_d(control_clean, treatment_clean)

            # Test difference
            _, original_p = stats.ttest_ind(treatment, control, equal_var=False)
            _, cleaned_p = stats.ttest_ind(
                treatment_clean, control_clean, equal_var=False
            )

            return {
                "outliers_removed": {
                    "control": len(control) - len(control_clean),
                    "treatment": len(treatment) - len(treatment_clean),
                    "total": (len(control) + len(treatment))
                    - (len(control_clean) + len(treatment_clean)),
                },
                "effect_size_comparison": {
                    "original": float(original_effect),
                    "cleaned": float(cleaned_effect),
                    "difference": float(abs(original_effect - cleaned_effect)),
                    "robust_to_outliers": abs(original_effect - cleaned_effect) < 0.1,
                },
                "p_value_comparison": {
                    "original": float(original_p),
                    "cleaned": float(cleaned_p),
                    "robust_to_outliers": abs(original_p - cleaned_p) < 0.01,
                },
            }

        except Exception as e:
            logger.warning(f"Error in sensitivity analysis: {e}")
            return {}

    def _generate_recommendations(self, result: AdvancedValidationResult) -> list[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []

        try:
            # Statistical significance recommendations
            if result.primary_test.p_value < self.alpha:
                if result.practical_significance:
                    recommendations.append(
                        "✅ DEPLOY: Statistically and practically significant improvement detected"
                    )
                else:
                    recommendations.append(
                        "⚠️ CAUTION: Statistically significant but effect size below practical threshold"
                    )
            else:
                recommendations.append(
                    "❌ DO NOT DEPLOY: No statistical significance detected"
                )

            # Effect size recommendations
            if result.effect_size_magnitude == EffectSizeMagnitude.VERY_LARGE:
                recommendations.append(
                    "🎯 Excellent effect size - strong business impact expected"
                )
            elif result.effect_size_magnitude == EffectSizeMagnitude.LARGE:
                recommendations.append(
                    "📈 Good effect size - meaningful business impact expected"
                )
            elif result.effect_size_magnitude == EffectSizeMagnitude.MEDIUM:
                recommendations.append(
                    "📊 Moderate effect size - monitor performance closely"
                )
            else:
                recommendations.append(
                    "📉 Small effect size - consider testing larger changes"
                )

            # Power analysis recommendations
            if result.post_hoc_power and result.post_hoc_power < self.power_threshold:
                recommendations.append(
                    f"⚡ Low statistical power ({result.post_hoc_power:.2f}) - consider increasing sample size"
                )

            # Assumption violations
            if result.normality_tests:
                normal_violations = [
                    test
                    for test in result.normality_tests.values()
                    if test.p_value < 0.05
                ]
                if normal_violations:
                    recommendations.append(
                        "📋 Normality assumption violated - non-parametric results may be more reliable"
                    )

            # Bootstrap vs parametric agreement
            if result.bootstrap_results and result.primary_test.confidence_interval:
                bootstrap_ci = result.bootstrap_results.get("mean_difference", {}).get(
                    "confidence_interval_95", []
                )
                parametric_ci = result.primary_test.confidence_interval

                if bootstrap_ci and parametric_ci:
                    # Check if confidence intervals are reasonably similar
                    if (
                        abs(bootstrap_ci[0] - parametric_ci[0]) > 0.1
                        or abs(bootstrap_ci[1] - parametric_ci[1]) > 0.1
                    ):
                        recommendations.append(
                            "🔍 Bootstrap and parametric results differ - investigate further"
                        )

        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
            recommendations.append(
                "⚠️ Could not generate complete recommendations due to analysis errors"
            )

        return recommendations

    def _generate_warnings(self, result: AdvancedValidationResult) -> list[str]:
        """Generate warnings based on validation results"""
        warnings = []

        try:
            # Small sample size warnings
            total_n = (
                result.primary_test.degrees_of_freedom + 2
                if result.primary_test.degrees_of_freedom
                else 0
            )
            if total_n < 30:
                warnings.append(
                    "⚠️ Small sample size may affect reliability of statistical tests"
                )

            # Multiple testing warnings
            if result.multiple_testing_correction:
                if not any(result.multiple_testing_correction.rejected):
                    warnings.append(
                        "⚠️ No significant results after multiple testing correction"
                    )

            # Assumption violations
            if result.normality_tests:
                violations = sum(
                    1 for test in result.normality_tests.values() if test.p_value < 0.05
                )
                if violations > 0:
                    warnings.append(
                        f"⚠️ {violations} normality assumption violation(s) detected"
                    )

            if result.homogeneity_tests:
                violations = sum(
                    1
                    for test in result.homogeneity_tests.values()
                    if test.p_value < 0.05
                )
                if violations > 0:
                    warnings.append(
                        f"⚠️ {violations} homogeneity assumption violation(s) detected"
                    )

            # Outlier sensitivity
            if (
                result.sensitivity_analysis
                and "effect_size_comparison" in result.sensitivity_analysis
            ):
                if not result.sensitivity_analysis["effect_size_comparison"].get(
                    "robust_to_outliers", True
                ):
                    warnings.append(
                        "⚠️ Results sensitive to outliers - interpret with caution"
                    )

        except Exception as e:
            logger.warning(f"Error generating warnings: {e}")
            warnings.append(
                "⚠️ Could not generate complete warnings due to analysis errors"
            )

        return warnings

    def _calculate_validation_quality_score(
        self, result: AdvancedValidationResult
    ) -> float:
        """Calculate overall validation quality score (0-1)"""
        try:
            score_components = []

            # Statistical rigor (30%)
            if result.primary_test.p_value is not None:
                rigor_score = 1.0 if result.primary_test.p_value < self.alpha else 0.5
                score_components.append(("rigor", rigor_score, 0.3))

            # Effect size adequacy (25%)
            if result.effect_size_magnitude:
                effect_scores = {
                    EffectSizeMagnitude.NEGLIGIBLE: 0.1,
                    EffectSizeMagnitude.SMALL: 0.4,
                    EffectSizeMagnitude.MEDIUM: 0.7,
                    EffectSizeMagnitude.LARGE: 0.9,
                    EffectSizeMagnitude.VERY_LARGE: 1.0,
                }
                effect_score = effect_scores.get(result.effect_size_magnitude, 0.0)
                score_components.append(("effect", effect_score, 0.25))

            # Statistical power (20%)
            if result.post_hoc_power:
                power_score = min(result.post_hoc_power / self.power_threshold, 1.0)
                score_components.append(("power", power_score, 0.2))

            # Assumption validity (15%)
            assumption_score = 1.0
            if result.normality_tests:
                violations = sum(
                    1 for test in result.normality_tests.values() if test.p_value < 0.05
                )
                assumption_score *= max(0.5, 1.0 - violations * 0.2)
            score_components.append(("assumptions", assumption_score, 0.15))

            # Robustness (10%)
            robustness_score = 1.0
            if (
                result.sensitivity_analysis
                and "effect_size_comparison" in result.sensitivity_analysis
            ):
                if not result.sensitivity_analysis["effect_size_comparison"].get(
                    "robust_to_outliers", True
                ):
                    robustness_score = 0.7
            score_components.append(("robustness", robustness_score, 0.1))

            # Calculate weighted average
            total_score = sum(score * weight for _, score, weight in score_components)
            total_weight = sum(weight for _, _, weight in score_components)

            return total_score / total_weight if total_weight > 0 else 0.0

        except Exception as e:
            logger.warning(f"Error calculating validation quality score: {e}")
            return 0.0


# Utility functions for external use
def quick_validation(
    control_data: list[float], treatment_data: list[float], alpha: float = 0.05
) -> dict[str, Any]:
    """Quick validation for immediate use"""
    validator = AdvancedStatisticalValidator(alpha=alpha)
    result = validator.validate_ab_test(
        control_data,
        treatment_data,
        validate_assumptions=False,
        include_bootstrap=False,
        include_sensitivity=False,
    )

    return {
        "statistically_significant": result.primary_test.p_value < alpha,
        "p_value": result.primary_test.p_value,
        "effect_size": result.primary_test.effect_size,
        "effect_magnitude": result.effect_size_magnitude.value,
        "practical_significance": result.practical_significance,
        "recommendations": result.recommendations,
        "quality_score": result.validation_quality_score,
    }
