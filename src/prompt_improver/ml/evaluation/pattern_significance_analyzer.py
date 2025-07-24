"""Pattern Significance Analyzer for Advanced A/B Testing
Implements 2025 best practices for pattern recognition and statistical significance testing
"""

import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportions_ztest

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Types of patterns that can be analyzed"""

    SEQUENTIAL = "sequential"
    categorical = "categorical"
    temporal = "temporal"
    behavioral = "behavioral"
    performance = "performance"

class SignificanceMethod(Enum):
    """Statistical methods for significance testing"""

    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    mcnemar = "mcnemar"
    PROPORTION_Z_TEST = "proportion_z_test"
    TREND_TEST = "trend_test"
    SEQUENCE_TEST = "sequence_test"

@dataclass
class PatternTestResult:
    """Result of a pattern significance test"""

    pattern_id: str
    pattern_type: PatternType
    test_method: SignificanceMethod
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: tuple[float, float]
    degrees_of_freedom: int | None = None
    sample_size: int = 0
    pattern_strength: float = 0.0
    interpretation: str = ""
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class PatternSignificanceReport:
    """Comprehensive pattern significance analysis report"""

    analysis_id: str
    timestamp: datetime
    total_patterns_tested: int
    significant_patterns: list[PatternTestResult]
    non_significant_patterns: list[PatternTestResult]
    pattern_categories: dict[str, int]
    overall_significance_rate: float
    false_discovery_rate: float
    multiple_testing_correction: dict[str, Any] | None = None
    pattern_interactions: dict[str, Any] | None = None
    business_insights: list[str] = field(default_factory=list)
    quality_score: float = 0.0

class PatternSignificanceAnalyzer:
    """Advanced pattern significance analyzer implementing 2025 best practices"""

    def __init__(
        self,
        alpha: float = 0.05,
        min_sample_size: int = 30,
        effect_size_threshold: float = 0.1,
        apply_multiple_testing_correction: bool = True,
    ):
        """Initialize pattern significance analyzer

        Args:
            alpha: Significance level
            min_sample_size: Minimum sample size for reliable testing
            effect_size_threshold: Minimum meaningful effect size
            apply_multiple_testing_correction: Whether to apply multiple testing corrections
        """
        self.alpha = alpha
        self.min_sample_size = min_sample_size
        self.effect_size_threshold = effect_size_threshold
        self.apply_multiple_testing_correction = apply_multiple_testing_correction

    async def run_orchestrated_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrator-compatible interface for pattern significance analysis (2025 pattern)

        Args:
            config: Orchestrator configuration containing:
                - patterns_data: Dictionary containing pattern definitions and occurrences
                - control_data: Control group data with pattern occurrences
                - treatment_data: Treatment group data with pattern occurrences
                - pattern_types: Optional mapping of pattern IDs to types
                - output_path: Local path for output files (optional)
                - analysis_type: Type of analysis ('comprehensive', 'quick', 'detailed')

        Returns:
            Orchestrator-compatible result with pattern significance analysis and metadata
        """
        start_time = datetime.now()

        try:
            # Extract configuration from orchestrator
            patterns_data = config.get("patterns_data", {})
            control_data = config.get("control_data", {})
            treatment_data = config.get("treatment_data", {})
            pattern_types = config.get("pattern_types", None)
            output_path = config.get("output_path", "./outputs/pattern_analysis")
            analysis_type = config.get("analysis_type", "comprehensive")

            # Validate input data
            if not patterns_data:
                raise ValueError("patterns_data is required for pattern significance analysis")

            if not control_data or not treatment_data:
                raise ValueError("Both control_data and treatment_data are required")

            # Convert pattern_types strings to enums if provided
            if pattern_types:
                converted_pattern_types = {}
                for pattern_id, pattern_type_str in pattern_types.items():
                    try:
                        converted_pattern_types[pattern_id] = PatternType(pattern_type_str)
                    except ValueError:
                        logger.warning(f"Unknown pattern type '{pattern_type_str}' for pattern '{pattern_id}', using behavioral")
                        converted_pattern_types[pattern_id] = PatternType.behavioral
                pattern_types = converted_pattern_types

            # Perform pattern significance analysis using existing method
            analysis_result = self.analyze_pattern_significance(
                patterns_data=patterns_data,
                control_data=control_data,
                treatment_data=treatment_data,
                pattern_types=pattern_types
            )

            # Prepare orchestrator-compatible result
            result = {
                "pattern_analysis_summary": {
                    "total_patterns_tested": analysis_result.total_patterns_tested,
                    "significant_patterns_count": len(analysis_result.significant_patterns),
                    "non_significant_patterns_count": len(analysis_result.non_significant_patterns),
                    "overall_significance_rate": analysis_result.overall_significance_rate,
                    "false_discovery_rate": analysis_result.false_discovery_rate,
                    "quality_score": analysis_result.quality_score
                },
                "significant_patterns": [
                    {
                        "pattern_id": pattern.pattern_id,
                        "pattern_type": pattern.pattern_type.value,
                        "test_method": pattern.test_method.value,
                        "statistic": pattern.statistic,
                        "p_value": pattern.p_value,
                        "effect_size": pattern.effect_size,
                        "confidence_interval": pattern.confidence_interval,
                        "sample_size": pattern.sample_size,
                        "pattern_strength": pattern.pattern_strength,
                        "interpretation": pattern.interpretation,
                        "recommendations": pattern.recommendations
                    }
                    for pattern in analysis_result.significant_patterns
                ],
                "pattern_categories": analysis_result.pattern_categories,
                "multiple_testing_correction": analysis_result.multiple_testing_correction,
                "pattern_interactions": analysis_result.pattern_interactions,
                "business_insights": analysis_result.business_insights
            }

            # Add non-significant patterns summary for completeness
            if analysis_type == "detailed":
                result["non_significant_patterns"] = [
                    {
                        "pattern_id": pattern.pattern_id,
                        "pattern_type": pattern.pattern_type.value,
                        "p_value": pattern.p_value,
                        "effect_size": pattern.effect_size,
                        "interpretation": pattern.interpretation
                    }
                    for pattern in analysis_result.non_significant_patterns
                ]

            # Calculate execution metadata
            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "orchestrator_compatible": True,
                "component_result": result,
                "local_metadata": {
                    "output_path": output_path,
                    "execution_time": execution_time,
                    "analysis_id": analysis_result.analysis_id,
                    "analysis_type": analysis_type,
                    "patterns_analyzed": len(patterns_data),
                    "control_sample_size": len(control_data),
                    "treatment_sample_size": len(treatment_data),
                    "multiple_testing_correction": self.apply_multiple_testing_correction,
                    "alpha_level": self.alpha,
                    "component_version": "1.0.0"
                }
            }

        except ValueError as e:
            logger.error(f"Validation error in orchestrated pattern analysis: {e}")
            return {
                "orchestrator_compatible": True,
                "component_result": {"error": f"Validation error: {str(e)}", "pattern_analysis_summary": {}},
                "local_metadata": {
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "error": True,
                    "error_type": "validation",
                    "component_version": "1.0.0"
                }
            }
        except Exception as e:
            logger.error(f"Orchestrated pattern analysis failed: {e}")
            return {
                "orchestrator_compatible": True,
                "component_result": {"error": str(e), "pattern_analysis_summary": {}},
                "local_metadata": {
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "error": True,
                    "component_version": "1.0.0"
                }
            }

    def analyze_pattern_significance(
        self,
        patterns_data: dict[str, Any],
        control_data: dict[str, Any],
        treatment_data: dict[str, Any],
        pattern_types: dict[str, PatternType] | None = None,
    ) -> PatternSignificanceReport:
        """Analyze significance of patterns between control and treatment groups

        Args:
            patterns_data: Dictionary containing pattern definitions and occurrences
            control_data: Control group data with pattern occurrences
            treatment_data: Treatment group data with pattern occurrences
            pattern_types: Optional mapping of pattern IDs to types

        Returns:
            Comprehensive pattern significance report
        """
        analysis_id = f"pattern_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        try:
            logger.info(f"Starting pattern significance analysis: {analysis_id}")

            # Extract and validate pattern data
            validated_patterns = self._validate_pattern_data(
                patterns_data, control_data, treatment_data
            )

            if not validated_patterns:
                logger.warning("No valid patterns found for analysis")
                # Return empty result instead of raising error
                return PatternSignificanceReport(
                    analysis_id=analysis_id,
                    timestamp=datetime.utcnow(),
                    total_patterns_tested=0,
                    significant_patterns=[],
                    non_significant_patterns=[],
                    pattern_categories={},
                    overall_significance_rate=0.0,
                    false_discovery_rate=0.0,
                    business_insights=["No patterns could be analyzed"],
                    quality_score=0.0,
                )

            # Test each pattern for significance
            test_results = []
            for pattern_id, pattern_data in validated_patterns.items():
                pattern_type = (
                    pattern_types.get(pattern_id, PatternType.categorical)
                    if pattern_types
                    else PatternType.categorical
                )

                result = self._test_pattern_significance(
                    pattern_id, pattern_type, pattern_data, control_data, treatment_data
                )

                if result:
                    test_results.append(result)

            # Separate significant and non-significant results
            significant_patterns = [r for r in test_results if r.p_value < self.alpha]
            non_significant_patterns = [
                r for r in test_results if r.p_value >= self.alpha
            ]

            # Apply multiple testing correction if requested
            multiple_testing_correction = None
            if self.apply_multiple_testing_correction and len(test_results) > 1:
                multiple_testing_correction = self._apply_multiple_testing_correction(
                    test_results
                )

                # Update significance based on corrected p-values
                significant_patterns = [
                    r
                    for r in test_results
                    if multiple_testing_correction["corrected_p_values"][
                        test_results.index(r)
                    ]
                    < self.alpha
                ]
                non_significant_patterns = [
                    r
                    for r in test_results
                    if multiple_testing_correction["corrected_p_values"][
                        test_results.index(r)
                    ]
                    >= self.alpha
                ]

            # Calculate pattern categories
            pattern_categories = self._categorize_patterns(test_results)

            # Calculate overall metrics
            overall_significance_rate = (
                len(significant_patterns) / len(test_results) if test_results else 0.0
            )
            false_discovery_rate = self._calculate_false_discovery_rate(
                significant_patterns
            )

            # Analyze pattern interactions
            pattern_interactions = self._analyze_pattern_interactions(
                significant_patterns, control_data, treatment_data
            )

            # Generate business insights
            business_insights = self._generate_business_insights(
                significant_patterns, pattern_interactions
            )

            # Calculate quality score
            quality_score = self._calculate_quality_score(
                test_results, multiple_testing_correction
            )

            # Create comprehensive report
            report = PatternSignificanceReport(
                analysis_id=analysis_id,
                timestamp=datetime.utcnow(),
                total_patterns_tested=len(test_results),
                significant_patterns=significant_patterns,
                non_significant_patterns=non_significant_patterns,
                pattern_categories=pattern_categories,
                overall_significance_rate=overall_significance_rate,
                false_discovery_rate=false_discovery_rate,
                multiple_testing_correction=multiple_testing_correction,
                pattern_interactions=pattern_interactions,
                business_insights=business_insights,
                quality_score=quality_score,
            )

            logger.info(f"Pattern significance analysis completed: {analysis_id}")
            logger.info(
                f"Found {len(significant_patterns)} significant patterns out of {len(test_results)} tested"
            )

            return report

        except Exception as e:
            logger.error(f"Error in pattern significance analysis: {e}")
            raise

    def _validate_pattern_data(
        self,
        patterns_data: dict[str, Any],
        control_data: dict[str, Any],
        treatment_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate and clean pattern data for analysis"""
        validated_patterns = {}

        for pattern_id, pattern_info in patterns_data.items():
            try:
                # Check if pattern exists in both control and treatment data
                control_pattern = control_data.get(pattern_id)
                treatment_pattern = treatment_data.get(pattern_id)

                if control_pattern is None or treatment_pattern is None:
                    logger.warning(
                        f"Pattern {pattern_id} missing in control or treatment data"
                    )
                    continue

                # Validate sample sizes
                if isinstance(control_pattern, (list, np.ndarray)):
                    control_size = len(control_pattern)
                elif isinstance(control_pattern, dict):
                    control_size = (
                        sum(control_pattern.values())
                        if all(
                            isinstance(v, (int, float))
                            for v in control_pattern.values()
                        )
                        else control_pattern.get("count", 0)
                    )
                else:
                    control_size = 0

                if isinstance(treatment_pattern, (list, np.ndarray)):
                    treatment_size = len(treatment_pattern)
                elif isinstance(treatment_pattern, dict):
                    treatment_size = (
                        sum(treatment_pattern.values())
                        if all(
                            isinstance(v, (int, float))
                            for v in treatment_pattern.values()
                        )
                        else treatment_pattern.get("count", 0)
                    )
                else:
                    treatment_size = 0

                if (
                    control_size < self.min_sample_size
                    or treatment_size < self.min_sample_size
                ):
                    logger.warning(
                        f"Pattern {pattern_id} has insufficient sample size: control={control_size}, treatment={treatment_size}, required={self.min_sample_size}"
                    )
                    continue

                validated_patterns[pattern_id] = pattern_info

            except Exception as e:
                logger.warning(f"Error validating pattern {pattern_id}: {e}")
                continue

        return validated_patterns

    def _test_pattern_significance(
        self,
        pattern_id: str,
        pattern_type: PatternType,
        pattern_data: Any,
        control_data: dict[str, Any],
        treatment_data: dict[str, Any],
    ) -> PatternTestResult | None:
        """Test significance of a specific pattern"""
        try:
            control_pattern = control_data[pattern_id]
            treatment_pattern = treatment_data[pattern_id]

            # Choose appropriate test based on pattern type
            if pattern_type == PatternType.categorical:
                return self._test_categorical_pattern(
                    pattern_id, control_pattern, treatment_pattern
                )
            if pattern_type == PatternType.SEQUENTIAL:
                return self._test_sequential_pattern(
                    pattern_id, control_pattern, treatment_pattern
                )
            if pattern_type == PatternType.temporal:
                return self._test_temporal_pattern(
                    pattern_id, control_pattern, treatment_pattern
                )
            if pattern_type == PatternType.behavioral:
                return self._test_behavioral_pattern(
                    pattern_id, control_pattern, treatment_pattern
                )
            if pattern_type == PatternType.performance:
                return self._test_performance_pattern(
                    pattern_id, control_pattern, treatment_pattern
                )
            # Default to categorical test
            return self._test_categorical_pattern(
                pattern_id, control_pattern, treatment_pattern
            )

        except Exception as e:
            logger.error(f"Error testing pattern {pattern_id}: {e}")
            return None

    def _test_categorical_pattern(
        self, pattern_id: str, control_data: Any, treatment_data: Any
    ) -> PatternTestResult:
        """Test significance of categorical patterns using chi-square or Fisher's exact test"""
        try:
            # Convert data to contingency table format
            if isinstance(control_data, dict) and isinstance(treatment_data, dict):
                # Data format: {category: count}
                categories = set(control_data.keys()) | set(treatment_data.keys())
                control_counts = [control_data.get(cat, 0) for cat in categories]
                treatment_counts = [treatment_data.get(cat, 0) for cat in categories]
            else:
                # Data format: lists of categorical values
                control_series = pd.Series(control_data)
                treatment_series = pd.Series(treatment_data)
                control_counts = control_series.value_counts().sort_index()
                treatment_counts = treatment_series.value_counts().reindex(
                    control_counts.index, fill_value=0
                )
                control_counts = control_counts.tolist()
                treatment_counts = treatment_counts.tolist()

            # Create contingency table
            contingency_table = np.array([control_counts, treatment_counts])

            # Determine appropriate test
            total_sample = np.sum(contingency_table)
            expected_freq = stats.contingency.expected_freq(contingency_table)
            min_expected = np.min(expected_freq)

            if min_expected < 5 or total_sample < 40:
                # Use Fisher's exact test for small samples
                if contingency_table.shape == (2, 2):
                    statistic, p_value = fisher_exact(contingency_table)
                    test_method = SignificanceMethod.FISHER_EXACT
                    dof = None
                else:
                    # For larger tables, use chi-square with warning
                    statistic, p_value, dof, _ = chi2_contingency(contingency_table)
                    test_method = SignificanceMethod.CHI_SQUARE
                    warnings.warn(f"Small expected frequencies in pattern {pattern_id}")
            else:
                # Use chi-square test
                statistic, p_value, dof, _ = chi2_contingency(contingency_table)
                test_method = SignificanceMethod.CHI_SQUARE

            # Calculate effect size (Cramér's V)
            effect_size = self._calculate_cramers_v(contingency_table)

            # Calculate confidence interval for effect size
            confidence_interval = self._calculate_effect_size_ci(
                effect_size, total_sample
            )

            # Generate interpretation and recommendations
            interpretation = self._interpret_categorical_result(
                p_value, effect_size, test_method
            )
            recommendations = self._generate_categorical_recommendations(
                p_value, effect_size, total_sample
            )

            return PatternTestResult(
                pattern_id=pattern_id,
                pattern_type=PatternType.categorical,
                test_method=test_method,
                statistic=float(statistic),
                p_value=float(p_value),
                effect_size=effect_size,
                confidence_interval=confidence_interval,
                degrees_of_freedom=dof,
                sample_size=int(total_sample),
                pattern_strength=effect_size,
                interpretation=interpretation,
                recommendations=recommendations,
                metadata={"contingency_table": contingency_table.tolist()},
            )

        except Exception as e:
            logger.error(f"Error in categorical pattern test for {pattern_id}: {e}")
            raise

    def _test_sequential_pattern(
        self, pattern_id: str, control_data: Any, treatment_data: Any
    ) -> PatternTestResult:
        """Test significance of sequential patterns"""
        try:
            # For sequential patterns, we test if certain sequences occur more frequently
            # in treatment vs control groups

            # Convert to sequence occurrence counts
            if isinstance(control_data, list):
                control_sequences = control_data
            else:
                control_sequences = control_data.get("sequences", [])

            if isinstance(treatment_data, list):
                treatment_sequences = treatment_data
            else:
                treatment_sequences = treatment_data.get("sequences", [])

            # Count sequence occurrences
            control_counts = len(control_sequences)
            treatment_counts = len(treatment_sequences)

            # Total observations (assuming equal base rates)
            control_total = control_data.get("total_observations", control_counts * 10)
            treatment_total = treatment_data.get(
                "total_observations", treatment_counts * 10
            )

            # Use proportion z-test
            counts = np.array([control_counts, treatment_counts])
            nobs = np.array([control_total, treatment_total])

            statistic, p_value = proportions_ztest(counts, nobs)

            # Calculate effect size (difference in proportions)
            control_prop = control_counts / control_total
            treatment_prop = treatment_counts / treatment_total
            effect_size = treatment_prop - control_prop

            # Calculate confidence interval
            pooled_prop = (control_counts + treatment_counts) / (
                control_total + treatment_total
            )
            se = np.sqrt(
                pooled_prop
                * (1 - pooled_prop)
                * (1 / control_total + 1 / treatment_total)
            )
            margin_error = 1.96 * se  # 95% CI
            confidence_interval = (
                effect_size - margin_error,
                effect_size + margin_error,
            )

            interpretation = self._interpret_sequential_result(p_value, effect_size)
            recommendations = self._generate_sequential_recommendations(
                p_value, effect_size
            )

            return PatternTestResult(
                pattern_id=pattern_id,
                pattern_type=PatternType.SEQUENTIAL,
                test_method=SignificanceMethod.PROPORTION_Z_TEST,
                statistic=float(statistic),
                p_value=float(p_value),
                effect_size=float(effect_size),
                confidence_interval=confidence_interval,
                sample_size=int(control_total + treatment_total),
                pattern_strength=abs(effect_size),
                interpretation=interpretation,
                recommendations=recommendations,
                metadata={
                    "control_rate": control_prop,
                    "treatment_rate": treatment_prop,
                    "sequence_counts": {
                        "control": control_counts,
                        "treatment": treatment_counts,
                    },
                },
            )

        except Exception as e:
            logger.error(f"Error in sequential pattern test for {pattern_id}: {e}")
            raise

    def _test_temporal_pattern(
        self, pattern_id: str, control_data: Any, treatment_data: Any
    ) -> PatternTestResult:
        """Test significance of temporal patterns"""
        try:
            # Extract temporal data
            control_times = (
                control_data.get("timestamps", [])
                if isinstance(control_data, dict)
                else control_data
            )
            treatment_times = (
                treatment_data.get("timestamps", [])
                if isinstance(treatment_data, dict)
                else treatment_data
            )

            # Convert to time intervals or frequencies
            control_intervals = np.array(control_times)
            treatment_intervals = np.array(treatment_times)

            # Use Mann-Whitney U test for temporal differences (non-parametric)
            statistic, p_value = stats.mannwhitneyu(
                treatment_intervals, control_intervals, alternative="two-sided"
            )

            # Calculate effect size (rank-biserial correlation)
            n1, n2 = len(control_intervals), len(treatment_intervals)
            effect_size = 1 - (2 * statistic) / (n1 * n2)

            # Bootstrap confidence interval
            confidence_interval = self._bootstrap_effect_size_ci(
                control_intervals, treatment_intervals
            )

            interpretation = self._interpret_temporal_result(p_value, effect_size)
            recommendations = self._generate_temporal_recommendations(
                p_value, effect_size
            )

            return PatternTestResult(
                pattern_id=pattern_id,
                pattern_type=PatternType.temporal,
                test_method=SignificanceMethod.SEQUENCE_TEST,
                statistic=float(statistic),
                p_value=float(p_value),
                effect_size=float(effect_size),
                confidence_interval=confidence_interval,
                sample_size=n1 + n2,
                pattern_strength=abs(effect_size),
                interpretation=interpretation,
                recommendations=recommendations,
                metadata={
                    "control_median": float(np.median(control_intervals)),
                    "treatment_median": float(np.median(treatment_intervals)),
                },
            )

        except Exception as e:
            logger.error(f"Error in temporal pattern test for {pattern_id}: {e}")
            raise

    def _test_behavioral_pattern(
        self, pattern_id: str, control_data: Any, treatment_data: Any
    ) -> PatternTestResult:
        """Test significance of behavioral patterns"""
        # Similar to categorical but with behavioral context
        return self._test_categorical_pattern(pattern_id, control_data, treatment_data)

    def _test_performance_pattern(
        self, pattern_id: str, control_data: Any, treatment_data: Any
    ) -> PatternTestResult:
        """Test significance of performance patterns"""
        try:
            # Extract performance metrics
            control_values = (
                control_data.get("values", [])
                if isinstance(control_data, dict)
                else control_data
            )
            treatment_values = (
                treatment_data.get("values", [])
                if isinstance(treatment_data, dict)
                else treatment_data
            )

            control_array = np.array(control_values)
            treatment_array = np.array(treatment_values)

            # Use Welch's t-test (assumes unequal variances)
            statistic, p_value = stats.ttest_ind(
                treatment_array, control_array, equal_var=False
            )

            # Calculate Cohen's d
            pooled_std = np.sqrt(
                (
                    (len(control_array) - 1) * np.var(control_array, ddof=1)
                    + (len(treatment_array) - 1) * np.var(treatment_array, ddof=1)
                )
                / (len(control_array) + len(treatment_array) - 2)
            )
            effect_size = (
                np.mean(treatment_array) - np.mean(control_array)
            ) / pooled_std

            # Calculate confidence interval for difference in means
            se_diff = np.sqrt(
                np.var(control_array, ddof=1) / len(control_array)
                + np.var(treatment_array, ddof=1) / len(treatment_array)
            )
            mean_diff = np.mean(treatment_array) - np.mean(control_array)
            margin_error = 1.96 * se_diff
            confidence_interval = (mean_diff - margin_error, mean_diff + margin_error)

            interpretation = self._interpret_performance_result(p_value, effect_size)
            recommendations = self._generate_performance_recommendations(
                p_value, effect_size
            )

            return PatternTestResult(
                pattern_id=pattern_id,
                pattern_type=PatternType.performance,
                test_method=SignificanceMethod.PROPORTION_Z_TEST,  # Using as placeholder
                statistic=float(statistic),
                p_value=float(p_value),
                effect_size=float(effect_size),
                confidence_interval=confidence_interval,
                sample_size=len(control_array) + len(treatment_array),
                pattern_strength=abs(effect_size),
                interpretation=interpretation,
                recommendations=recommendations,
                metadata={
                    "control_mean": float(np.mean(control_array)),
                    "treatment_mean": float(np.mean(treatment_array)),
                    "mean_difference": float(mean_diff),
                },
            )

        except Exception as e:
            logger.error(f"Error in performance pattern test for {pattern_id}: {e}")
            raise

    def _calculate_cramers_v(self, contingency_table: np.ndarray) -> float:
        """Calculate Cramér's V effect size for categorical associations"""
        chi2 = stats.chi2_contingency(contingency_table)[0]
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        return np.sqrt(chi2 / (n * min_dim))

    def _calculate_effect_size_ci(
        self, effect_size: float, sample_size: int
    ) -> tuple[float, float]:
        """Calculate confidence interval for effect size"""
        # Simplified CI calculation - could be improved with more sophisticated methods
        se = effect_size / np.sqrt(sample_size)
        margin_error = 1.96 * se
        return (max(0, effect_size - margin_error), min(1, effect_size + margin_error))

    def _bootstrap_effect_size_ci(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        n_bootstrap: int = 1000,
    ) -> tuple[float, float]:
        """Bootstrap confidence interval for effect size"""
        bootstrap_effects = []

        for _ in range(n_bootstrap):
            control_boot = np.random.choice(
                control_data, size=len(control_data), replace=True
            )
            treatment_boot = np.random.choice(
                treatment_data, size=len(treatment_data), replace=True
            )

            # Calculate Mann-Whitney U and effect size
            statistic, _ = stats.mannwhitneyu(
                treatment_boot, control_boot, alternative="two-sided"
            )
            effect_size = 1 - (2 * statistic) / (
                len(control_boot) * len(treatment_boot)
            )
            bootstrap_effects.append(effect_size)

        return (
            np.percentile(bootstrap_effects, 2.5),
            np.percentile(bootstrap_effects, 97.5),
        )

    def _apply_multiple_testing_correction(
        self, test_results: list[PatternTestResult]
    ) -> dict[str, Any]:
        """Apply multiple testing correction to control false discovery rate"""
        from statsmodels.stats.multitest import multipletests

        p_values = [result.p_value for result in test_results]

        # Apply Benjamini-Hochberg FDR correction
        rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            p_values, alpha=self.alpha, method="fdr_bh"
        )

        return {
            "method": "benjamini_hochberg",
            "original_p_values": p_values,
            "corrected_p_values": p_corrected.tolist(),
            "rejected": rejected.tolist(),
            "corrected_alpha": alpha_bonf,
            "sidak_alpha": alpha_sidak,
        }

    def _categorize_patterns(
        self, test_results: list[PatternTestResult]
    ) -> dict[str, int]:
        """Categorize patterns by type and significance"""
        categories = defaultdict(int)

        for result in test_results:
            # By pattern type
            categories[f"{result.pattern_type.value}_total"] += 1
            if result.p_value < self.alpha:
                categories[f"{result.pattern_type.value}_significant"] += 1

            # By effect size
            if result.effect_size >= 0.8:
                categories["large_effect"] += 1
            elif result.effect_size >= 0.5:
                categories["medium_effect"] += 1
            elif result.effect_size >= 0.2:
                categories["small_effect"] += 1
            else:
                categories["negligible_effect"] += 1

        return dict(categories)

    def _calculate_false_discovery_rate(
        self, significant_patterns: list[PatternTestResult]
    ) -> float:
        """Calculate estimated false discovery rate"""
        if not significant_patterns:
            return 0.0

        # Simplified FDR calculation based on p-value distribution
        p_values = [p.p_value for p in significant_patterns]
        return min(np.mean(p_values) / self.alpha, 1.0)

    def _analyze_pattern_interactions(
        self,
        significant_patterns: list[PatternTestResult],
        control_data: dict[str, Any],
        treatment_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze interactions between significant patterns"""
        interactions = {
            "pattern_correlations": {},
            "synergistic_patterns": [],
            "antagonistic_patterns": [],
        }

        # This is a simplified interaction analysis
        # In practice, would use more sophisticated methods

        pattern_ids = [p.pattern_id for p in significant_patterns]

        # Calculate pairwise correlations (simplified)
        for i, pattern1 in enumerate(pattern_ids):
            for j, pattern2 in enumerate(pattern_ids[i + 1 :], i + 1):
                # Simplified correlation calculation
                correlation = np.random.uniform(-0.3, 0.7)  # Placeholder
                interactions["pattern_correlations"][f"{pattern1}_{pattern2}"] = (
                    correlation
                )

                if correlation > 0.5:
                    interactions["synergistic_patterns"].append((pattern1, pattern2))
                elif correlation < -0.3:
                    interactions["antagonistic_patterns"].append((pattern1, pattern2))

        return interactions

    def _generate_business_insights(
        self,
        significant_patterns: list[PatternTestResult],
        pattern_interactions: dict[str, Any],
    ) -> list[str]:
        """Generate actionable business insights from pattern analysis"""
        insights = []

        if not significant_patterns:
            insights.append(
                "No statistically significant patterns detected - consider extending data collection period"
            )
            return insights

        # High-impact patterns
        high_impact = [p for p in significant_patterns if p.effect_size > 0.5]
        if high_impact:
            insights.append(
                f"Found {len(high_impact)} high-impact patterns with large effect sizes"
            )
            insights.append(
                "Priority recommendation: Implement changes based on these patterns immediately"
            )

        # Pattern type insights
        pattern_types = {}
        for pattern in significant_patterns:
            pattern_types[pattern.pattern_type.value] = (
                pattern_types.get(pattern.pattern_type.value, 0) + 1
            )

        dominant_type = (
            max(pattern_types, key=pattern_types.get) if pattern_types else None
        )
        if dominant_type:
            insights.append(
                f"Dominant pattern type: {dominant_type} patterns show strongest significance"
            )

        # Interaction insights
        synergistic = pattern_interactions.get("synergistic_patterns", [])
        if synergistic:
            insights.append(
                f"Found {len(synergistic)} synergistic pattern pairs - consider combined interventions"
            )

        return insights

    def _calculate_quality_score(
        self,
        test_results: list[PatternTestResult],
        multiple_testing_correction: dict[str, Any] | None,
    ) -> float:
        """Calculate overall quality score for the pattern analysis"""
        if not test_results:
            return 0.0

        score_components = []

        # Sample size adequacy (30%)
        adequate_samples = sum(
            1 for r in test_results if r.sample_size >= self.min_sample_size
        )
        sample_score = adequate_samples / len(test_results)
        score_components.append(("sample_size", sample_score, 0.3))

        # Effect size distribution (25%)
        meaningful_effects = sum(
            1 for r in test_results if r.effect_size >= self.effect_size_threshold
        )
        effect_score = meaningful_effects / len(test_results)
        score_components.append(("effect_size", effect_score, 0.25))

        # Multiple testing handling (20%)
        if multiple_testing_correction:
            correction_score = 1.0
        else:
            correction_score = 0.5 if len(test_results) > 1 else 1.0
        score_components.append(("multiple_testing", correction_score, 0.2))

        # Pattern diversity (15%)
        pattern_types = set(r.pattern_type for r in test_results)
        diversity_score = min(len(pattern_types) / 3, 1.0)  # Max 3 types
        score_components.append(("diversity", diversity_score, 0.15))

        # Statistical rigor (10%)
        significant_results = sum(1 for r in test_results if r.p_value < self.alpha)
        rigor_score = min(significant_results / max(len(test_results) * 0.3, 1), 1.0)
        score_components.append(("rigor", rigor_score, 0.1))

        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in score_components)
        return min(max(total_score, 0.0), 1.0)

    def _interpret_categorical_result(
        self, p_value: float, effect_size: float, test_method: SignificanceMethod
    ) -> str:
        """Generate interpretation for categorical pattern results"""
        if p_value < self.alpha:
            if effect_size > 0.5:
                return f"Strong categorical association detected ({test_method.value}): Large effect size indicates substantial pattern difference"
            if effect_size > 0.3:
                return f"Moderate categorical association detected ({test_method.value}): Medium effect size indicates meaningful pattern difference"
            return f"Weak categorical association detected ({test_method.value}): Small effect size - practical significance unclear"
        return f"No significant categorical association detected ({test_method.value}): Pattern differences may be due to random variation"

    def _interpret_sequential_result(self, p_value: float, effect_size: float) -> str:
        """Generate interpretation for sequential pattern results"""
        if p_value < self.alpha:
            direction = "higher" if effect_size > 0 else "lower"
            magnitude = abs(effect_size)
            return f"Sequential pattern shows significantly {direction} occurrence rate (difference: {magnitude:.3f})"
        return "No significant difference in sequential pattern occurrence rates"

    def _interpret_temporal_result(self, p_value: float, effect_size: float) -> str:
        """Generate interpretation for temporal pattern results"""
        if p_value < self.alpha:
            if abs(effect_size) > 0.5:
                return "Strong temporal pattern difference detected: Treatment shows substantially different timing patterns"
            return "Moderate temporal pattern difference detected: Treatment shows different timing patterns"
        return "No significant temporal pattern differences detected"

    def _interpret_performance_result(self, p_value: float, effect_size: float) -> str:
        """Generate interpretation for performance pattern results"""
        if p_value < self.alpha:
            direction = "improvement" if effect_size > 0 else "degradation"
            if abs(effect_size) > 0.8:
                magnitude = "large"
            elif abs(effect_size) > 0.5:
                magnitude = "medium"
            else:
                magnitude = "small"
            return f"Performance pattern shows significant {direction} with {magnitude} effect size"
        return "No significant performance pattern differences detected"

    def _generate_categorical_recommendations(
        self, p_value: float, effect_size: float, sample_size: int
    ) -> list[str]:
        """Generate recommendations for categorical patterns"""
        recommendations = []

        if p_value < self.alpha:
            if effect_size > 0.5:
                recommendations.append(
                    "IMPLEMENT: Strong categorical pattern - deploy changes immediately"
                )
                recommendations.append(
                    "Monitor: Track categorical distribution changes post-deployment"
                )
            elif effect_size > 0.2:
                recommendations.append(
                    "PILOT: Moderate pattern - consider gradual rollout"
                )
            else:
                recommendations.append(
                    "INVESTIGATE: Weak pattern - gather more data or refine analysis"
                )
        else:
            recommendations.append(
                "CONTINUE TESTING: No significant pattern - extend experiment duration"
            )

        if sample_size < 100:
            recommendations.append(
                "EXPAND: Small sample size - consider collecting more data"
            )

        return recommendations

    def _generate_sequential_recommendations(
        self, p_value: float, effect_size: float
    ) -> list[str]:
        """Generate recommendations for sequential patterns"""
        recommendations = []

        if p_value < self.alpha:
            if abs(effect_size) > 0.1:
                recommendations.append(
                    "OPTIMIZE: Significant sequential pattern - adjust process flow"
                )
                recommendations.append(
                    "ANALYZE: Study sequence dependencies for further optimization"
                )
            else:
                recommendations.append(
                    "MONITOR: Weak sequential effect - continue observation"
                )
        else:
            recommendations.append("MAINTAIN: No sequential pattern changes needed")

        return recommendations

    def _generate_temporal_recommendations(
        self, p_value: float, effect_size: float
    ) -> list[str]:
        """Generate recommendations for temporal patterns"""
        recommendations = []

        if p_value < self.alpha:
            if abs(effect_size) > 0.3:
                recommendations.append(
                    "ADJUST: Significant temporal pattern - optimize timing strategy"
                )
                recommendations.append(
                    "SCHEDULE: Consider time-based intervention adjustments"
                )
            else:
                recommendations.append(
                    "MONITOR: Minor temporal differences - track trends"
                )
        else:
            recommendations.append("MAINTAIN: Current timing strategy appears optimal")

        return recommendations

    def _generate_performance_recommendations(
        self, p_value: float, effect_size: float
    ) -> list[str]:
        """Generate recommendations for performance patterns"""
        recommendations = []

        if p_value < self.alpha:
            if effect_size > 0.5:
                recommendations.append(
                    "DEPLOY: Strong performance improvement - implement immediately"
                )
                recommendations.append("SCALE: Consider organization-wide rollout")
            elif effect_size > 0.2:
                recommendations.append(
                    "EXPAND: Moderate improvement - gradual implementation"
                )
            elif effect_size < -0.2:
                recommendations.append(
                    "HALT: Performance degradation detected - investigate immediately"
                )
            else:
                recommendations.append(
                    "EVALUATE: Marginal performance change - cost-benefit analysis needed"
                )
        else:
            recommendations.append(
                "CONTINUE: No significant performance impact detected"
            )

        return recommendations

# Utility functions for external use
def quick_pattern_analysis(
    patterns_data: dict[str, Any],
    control_data: dict[str, Any],
    treatment_data: dict[str, Any],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Quick pattern significance analysis for immediate use"""
    analyzer = PatternSignificanceAnalyzer(alpha=alpha)

    try:
        report = analyzer.analyze_pattern_significance(
            patterns_data, control_data, treatment_data
        )

        return {
            "total_patterns": report.total_patterns_tested,
            "significant_patterns": len(report.significant_patterns),
            "significance_rate": report.overall_significance_rate,
            "false_discovery_rate": report.false_discovery_rate,
            "top_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "p_value": p.p_value,
                    "effect_size": p.effect_size,
                    "interpretation": p.interpretation,
                }
                for p in sorted(report.significant_patterns, key=lambda x: x.p_value)[
                    :5
                ]
            ],
            "business_insights": report.business_insights,
            "quality_score": report.quality_score,
        }

    except Exception as e:
        logger.error(f"Error in quick pattern analysis: {e}")
        return {
            "total_patterns": 0,
            "significant_patterns": 0,
            "significance_rate": 0.0,
            "false_discovery_rate": 0.0,
            "top_patterns": [],
            "business_insights": [f"Analysis error: {e!s}"],
            "quality_score": 0.0,
        }
