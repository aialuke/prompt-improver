"""Advanced A/B Testing Framework

Enhanced A/B testing framework with stratified sampling, comprehensive statistical analysis,
and test orchestration for rule modification validation. Extends the basic A/B testing
service with advanced features for enterprise-grade experimentation.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats


from sqlalchemy.ext.asyncio import AsyncSession

from ..database.models import ABExperiment, ABExperimentCreate
from ..services.ab_testing import ABTestingService
from ..services.analytics import AnalyticsService
from ..utils.error_handlers import handle_database_errors

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RecommendationLevel(Enum):
    """Deployment recommendation levels"""

    STRONG_DEPLOY = "strong_deploy"
    CONDITIONAL_DEPLOY = "conditional_deploy"
    MONITOR_DEPLOY = "monitor_deploy"
    DO_NOT_DEPLOY = "do_not_deploy"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class AdvancedConfig:
    """Configuration for advanced A/B testing"""

    # Statistical parameters
    minimum_sample_size: int = 30
    significance_level: float = 0.05
    power_threshold: float = 0.8
    confidence_level: float = 0.95

    # Test design parameters
    max_test_duration_days: int = 7
    min_effect_size: float = 0.1
    balance_tolerance_percent: float = 5

    # Batch processing
    max_concurrent_tests: int = 5
    batch_size: int = 10

    # Quality controls
    outliers_threshold: float = 3.0
    consistency_threshold: float = 0.1

    # Advanced features
    enable_stratified_sampling: bool = True
    enable_sequential_testing: bool = True
    enable_bayesian_analysis: bool = True
    enable_multiple_testing_correction: bool = True


@dataclass
class GroupResults:
    """Results for a test group"""

    group_type: str
    sample_size: int
    successful_tests: int
    results: list[dict[str, Any]]
    summary: dict[str, Any]
    execution_time_ms: float


@dataclass
class QualityChecks:
    """Quality assurance checks for A/B test"""

    group_balance_ok: bool
    outliers_detected: int
    consistency_score: float
    data_quality_score: float
    issues: list[str] = field(default_factory=list)


@dataclass
class AdvancedAnalysis:
    """Advanced statistical analysis results"""

    # Basic analysis
    descriptive_stats: dict[str, Any]
    significance_test: dict[str, Any]
    effect_size: dict[str, Any]
    power_analysis: dict[str, Any]
    confidence_interval: dict[str, Any]

    # Advanced analysis
    bayesian_analysis: dict[str, Any] | None = None
    sequential_analysis: dict[str, Any] | None = None
    multiple_testing_correction: dict[str, Any] | None = None
    sensitivity_analysis: dict[str, Any] | None = None


@dataclass
class DeploymentRecommendation:
    """Deployment recommendation with reasoning"""

    level: RecommendationLevel
    deploy: bool
    confidence: float
    reasoning: list[str]
    conditions: list[str] = field(default_factory=list)
    monitoring_requirements: list[str] = field(default_factory=list)


@dataclass
class AdvancedTestResult:
    """Complete A/B test results with advanced analysis"""

    test_id: str
    metadata: dict[str, Any]
    control_results: GroupResults
    treatment_results: GroupResults
    analysis: AdvancedAnalysis
    recommendation: DeploymentRecommendation
    deployment_ready: bool
    quality_checks: QualityChecks
    validation_status: dict[str, Any]
    execution_summary: dict[str, Any]


class AdvancedABTestingFramework:
    """Advanced A/B Testing Framework for rule modification validation"""

    def __init__(
        self,
        config: AdvancedConfig | None = None,
        db_session: AsyncSession | None = None,
    ):
        """Initialize the advanced A/B testing framework

        Args:
            config: Advanced configuration settings
            db_session: Database session for persistence
        """
        self.config = config or AdvancedConfig()
        self.db_session = db_session
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Test tracking
        self.active_tests: dict[str, dict[str, Any]] = {}
        self.test_history: list[AdvancedTestResult] = []
        self.statistical_cache: dict[str, Any] = {}

        # Initialize underlying services
        self.ab_service = ABTestingService()
        self.analytics_service = AnalyticsService()

    async def test_rule_modification(
        self,
        original_rule: dict[str, Any],
        modified_rule: dict[str, Any],
        test_set: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> AdvancedTestResult:
        """Execute comprehensive A/B test for rule modification

        Args:
            original_rule: Current rule implementation
            modified_rule: Proposed rule modification
            test_set: Test cases for evaluation
            options: Additional test options

        Returns:
            Complete A/B test results with advanced statistical analysis
        """
        options = options or {}

        # Validate input parameters
        self._validate_test_inputs(original_rule, modified_rule, test_set)

        # Check sample size adequacy
        min_required = self.config.minimum_sample_size * 2
        if len(test_set) < min_required:
            raise ValueError(
                f"Insufficient sample size: {len(test_set)} < {min_required}"
            )

        test_id = self._generate_test_id(original_rule, modified_rule)
        test_start_time = datetime.now()

        try:
            self.logger.info(
                f"Starting A/B test {test_id}",
                extra={
                    "original_rule": original_rule.get("name", "unknown"),
                    "modified_rule": modified_rule.get("name", "modified"),
                    "sample_size": len(test_set),
                },
            )

            # Create balanced test groups using stratified sampling
            group_a, group_b = self._create_balanced_groups(test_set, options)

            # Track test start
            self.active_tests[test_id] = {
                "start_time": test_start_time,
                "status": TestStatus.RUNNING,
                "original_rule": original_rule,
                "modified_rule": modified_rule,
                "group_sizes": [len(group_a), len(group_b)],
            }

            # Execute tests in parallel for efficiency
            execution_start = datetime.now()
            control_results, treatment_results = await asyncio.gather(
                self._execute_test_group(group_a, original_rule, "control"),
                self._execute_test_group(group_b, modified_rule, "treatment"),
            )
            execution_time = (datetime.now() - execution_start).total_seconds() * 1000

            # Comprehensive statistical analysis
            analysis = await self._perform_advanced_analysis(
                control_results, treatment_results, test_id, options
            )

            # Generate deployment recommendation
            recommendation = self._generate_deployment_recommendation(analysis)

            # Perform quality checks
            quality_checks = self._perform_quality_checks(
                control_results, treatment_results
            )

            # Validate test results
            validation_status = self._validate_test_results(analysis, quality_checks)

            # Create test result
            test_result = AdvancedTestResult(
                test_id=test_id,
                metadata={
                    "original_rule": original_rule.get("name", "unknown"),
                    "modified_rule": modified_rule.get("name", "modified"),
                    "test_date": test_start_time.isoformat(),
                    "sample_size_per_group": [len(group_a), len(group_b)],
                    "test_duration_ms": (
                        datetime.now() - test_start_time
                    ).total_seconds()
                    * 1000,
                    "execution_time_ms": execution_time,
                    "config": {
                        "significance_level": self.config.significance_level,
                        "confidence_level": self.config.confidence_level,
                        "min_effect_size": self.config.min_effect_size,
                        "stratified_sampling": self.config.enable_stratified_sampling,
                        "bayesian_analysis": self.config.enable_bayesian_analysis,
                    },
                },
                control_results=control_results,
                treatment_results=treatment_results,
                analysis=analysis,
                recommendation=recommendation,
                deployment_ready=recommendation.deploy,
                quality_checks=quality_checks,
                validation_status=validation_status,
                execution_summary={
                    "total_test_cases": len(test_set),
                    "successful_control": control_results.successful_tests,
                    "successful_treatment": treatment_results.successful_tests,
                    "success_rate_control": control_results.successful_tests
                    / control_results.sample_size
                    if control_results.sample_size > 0
                    else 0,
                    "success_rate_treatment": treatment_results.successful_tests
                    / treatment_results.sample_size
                    if treatment_results.sample_size > 0
                    else 0,
                    "overall_improvement": analysis.descriptive_stats.get(
                        "overall_improvement", 0
                    ),
                },
            )

            # Update tracking
            self.active_tests[test_id]["status"] = TestStatus.COMPLETED
            del self.active_tests[test_id]
            self.test_history.append(test_result)

            # Persist results if database session available
            if self.db_session:
                await self._persist_test_results(test_result)

            self.logger.info(
                f"A/B test {test_id} completed",
                extra={
                    "deployment_ready": test_result.deployment_ready,
                    "recommendation": recommendation.level.value,
                    "effect_size": analysis.effect_size.get("cohens_d", 0),
                    "p_value": analysis.significance_test.get("p_value", 1),
                },
            )

            return test_result

        except Exception as error:
            self.logger.error(f"A/B test {test_id} failed: {error}")
            if test_id in self.active_tests:
                self.active_tests[test_id]["status"] = TestStatus.FAILED
                del self.active_tests[test_id]
            raise error

    def _validate_test_inputs(
        self,
        original_rule: dict[str, Any],
        modified_rule: dict[str, Any],
        test_set: list[dict[str, Any]],
    ):
        """Validate test inputs"""
        if not original_rule:
            raise ValueError("Original rule cannot be empty")
        if not modified_rule:
            raise ValueError("Modified rule cannot be empty")
        if not test_set:
            raise ValueError("Test set cannot be empty")
        if len(test_set) < self.config.minimum_sample_size:
            self.logger.warning(
                f"Sample size {len(test_set)} below minimum {self.config.minimum_sample_size}"
            )

    def _generate_test_id(
        self, original_rule: dict[str, Any], modified_rule: dict[str, Any]
    ) -> str:
        """Generate unique test ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = original_rule.get("name", "unknown")[:10]
        modified_name = modified_rule.get("name", "modified")[:10]
        unique_id = str(uuid.uuid4())[:8]
        return f"abtest_{original_name}_{modified_name}_{timestamp}_{unique_id}"

    def _create_balanced_groups(
        self, test_set: list[dict[str, Any]], options: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Create balanced test groups using stratified sampling"""
        if not self.config.enable_stratified_sampling:
            # Simple random split
            np.random.shuffle(test_set)
            midpoint = len(test_set) // 2
            return test_set[:midpoint], test_set[midpoint:]

        # Stratified sampling by key variables
        stratify_by = options.get("stratify_by", ["category", "complexity", "context"])
        strata = self._create_strata(test_set, stratify_by)

        group_a = []
        group_b = []

        # Randomly assign within each stratum
        for stratum in strata:
            if len(stratum) == 1:
                # Single item - assign randomly
                if np.random.random() < 0.5:
                    group_a.extend(stratum)
                else:
                    group_b.extend(stratum)
            else:
                # Multiple items - split evenly
                shuffled = np.random.permutation(stratum).tolist()
                midpoint = len(shuffled) // 2
                group_a.extend(shuffled[:midpoint])
                group_b.extend(shuffled[midpoint:])

        # Validate group balance
        self._validate_group_balance(group_a, group_b)

        return group_a, group_b

    def _create_strata(
        self, test_set: list[dict[str, Any]], stratify_by: list[str]
    ) -> list[list[dict[str, Any]]]:
        """Create stratified samples based on specified variables"""
        strata_map = {}

        for test_case in test_set:
            stratum_key = "|".join([
                self._get_stratum_value(test_case, variable) for variable in stratify_by
            ])

            if stratum_key not in strata_map:
                strata_map[stratum_key] = []
            strata_map[stratum_key].append(test_case)

        return list(strata_map.values())

    def _get_stratum_value(self, test_case: dict[str, Any], variable: str) -> str:
        """Get stratum value for a test case variable"""
        if variable == "category":
            return test_case.get("category", "unknown")
        if variable == "complexity":
            return test_case.get("complexity", "unknown")
        if variable == "context":
            return test_case.get("context", {}).get("projectType", "unknown")
        return test_case.get(variable, "unknown")

    def _validate_group_balance(
        self, group_a: list[dict[str, Any]], group_b: list[dict[str, Any]]
    ):
        """Validate that groups are balanced within tolerance"""
        total_size = len(group_a) + len(group_b)
        if total_size == 0:
            raise ValueError("Both groups are empty")

        size_diff_percent = abs(len(group_a) - len(group_b)) / total_size * 100
        if size_diff_percent > self.config.balance_tolerance_percent:
            self.logger.warning(
                f"Group imbalance: {size_diff_percent:.1f}% > {self.config.balance_tolerance_percent}%"
            )

    async def _execute_test_group(
        self, group: list[dict[str, Any]], rule: dict[str, Any], group_type: str
    ) -> GroupResults:
        """Execute tests for a specific group"""
        start_time = datetime.now()
        results = []

        # Process in batches for efficiency
        for i in range(0, len(group), self.config.batch_size):
            batch = group[i : i + self.config.batch_size]
            batch_results = await asyncio.gather(
                *[self._execute_test_case(test_case, rule) for test_case in batch],
                return_exceptions=True,
            )

            for j, result in enumerate(batch_results):
                test_case = batch[j]
                if isinstance(result, Exception):
                    results.append({
                        "test_case_id": test_case.get("id", f"test_{i + j}"),
                        "group_type": group_type,
                        "error": str(result),
                        "success": False,
                        "execution_time_ms": 0,
                    })
                else:
                    results.append({
                        "test_case_id": test_case.get("id", f"test_{i + j}"),
                        "group_type": group_type,
                        **result,
                        "success": result.get("success", True),
                    })

        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        successful_tests = sum(1 for r in results if r.get("success", False))

        return GroupResults(
            group_type=group_type,
            sample_size=len(results),
            successful_tests=successful_tests,
            results=results,
            summary=self._summarize_group_results(results),
            execution_time_ms=execution_time,
        )

    async def _execute_test_case(
        self, test_case: dict[str, Any], rule: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute individual test case with a specific rule"""
        start_time = datetime.now()

        try:
            # Simulate rule application - in reality, this would use the actual rule engine
            improvement = self._simulate_rule_application(test_case, rule)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return {
                "original_prompt": test_case.get("originalPrompt", ""),
                "improved_prompt": improvement["improved_prompt"],
                "improvement_score": improvement["score"],
                "quality_metrics": improvement["metrics"],
                "execution_time_ms": execution_time,
                "success": True,
            }
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return {
                "error": str(e),
                "execution_time_ms": execution_time,
                "success": False,
            }

    def _simulate_rule_application(
        self, test_case: dict[str, Any], rule: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate rule application for testing purposes"""
        # Mock implementation - in reality, this would use the actual rule engine
        base_score = 0.7 + (np.random.random() * 0.3)  # Random score between 0.7-1.0
        rule_modifier = rule.get("expectedImprovement", 0)

        final_score = min(1.0, base_score + rule_modifier + np.random.normal(0, 0.05))

        return {
            "improved_prompt": f"{test_case.get('originalPrompt', '')} [improved by {rule.get('name', 'rule')}]",
            "score": final_score,
            "metrics": {
                "clarity": final_score + np.random.normal(0, 0.02),
                "completeness": final_score + np.random.normal(0, 0.02),
                "actionability": final_score + np.random.normal(0, 0.02),
                "effectiveness": final_score + np.random.normal(0, 0.02),
            },
        }

    def _summarize_group_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize results for a group"""
        successful_results = [r for r in results if r.get("success", False)]

        if not successful_results:
            return {
                "success_rate": 0.0,
                "avg_score": 0.0,
                "avg_execution_time_ms": 0.0,
                "total_tests": len(results),
            }

        scores = [r.get("improvement_score", 0) for r in successful_results]
        execution_times = [r.get("execution_time_ms", 0) for r in successful_results]

        return {
            "success_rate": len(successful_results) / len(results),
            "avg_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            "avg_execution_time_ms": np.mean(execution_times),
            "total_tests": len(results),
            "successful_tests": len(successful_results),
        }

    async def _perform_advanced_analysis(
        self,
        control_results: GroupResults,
        treatment_results: GroupResults,
        test_id: str,
        options: dict[str, Any],
    ) -> AdvancedAnalysis:
        """Perform comprehensive statistical analysis"""
        # Extract scores for analysis
        control_scores = [
            r.get("improvement_score", 0)
            for r in control_results.results
            if r.get("success", False)
        ]
        treatment_scores = [
            r.get("improvement_score", 0)
            for r in treatment_results.results
            if r.get("success", False)
        ]

        if len(control_scores) < 3 or len(treatment_scores) < 3:
            raise ValueError(
                "Insufficient successful test cases for statistical analysis"
            )

        # Basic statistical analysis
        descriptive_stats = self._calculate_descriptive_statistics(
            control_scores, treatment_scores
        )
        significance_test = self._perform_significance_test(
            control_scores, treatment_scores
        )
        effect_size = self._calculate_effect_size(control_scores, treatment_scores)
        power_analysis = self._perform_power_analysis(control_scores, treatment_scores)
        confidence_interval = self._calculate_confidence_interval(
            control_scores, treatment_scores
        )

        analysis = AdvancedAnalysis(
            descriptive_stats=descriptive_stats,
            significance_test=significance_test,
            effect_size=effect_size,
            power_analysis=power_analysis,
            confidence_interval=confidence_interval,
        )

        # Advanced analysis (if enabled)
        if self.config.enable_bayesian_analysis:
            analysis.bayesian_analysis = self._perform_bayesian_analysis(
                control_scores, treatment_scores
            )

        if self.config.enable_sequential_testing:
            analysis.sequential_analysis = self._perform_sequential_analysis(
                control_scores, treatment_scores
            )

        if self.config.enable_multiple_testing_correction:
            analysis.multiple_testing_correction = (
                self._apply_multiple_testing_correction(significance_test)
            )

        # Sensitivity analysis
        analysis.sensitivity_analysis = self._perform_sensitivity_analysis(
            control_scores, treatment_scores
        )

        return analysis

    def _calculate_descriptive_statistics(
        self, control: list[float], treatment: list[float]
    ) -> dict[str, Any]:
        """Calculate descriptive statistics for both groups"""

        def calc_stats(scores):
            if not scores:
                return None
            return {
                "n": len(scores),
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "std": float(np.std(scores, ddof=1)),
                "var": float(np.var(scores, ddof=1)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "q1": float(np.percentile(scores, 25)),
                "q3": float(np.percentile(scores, 75)),
                "skewness": float(stats.skew(scores)),
                "kurtosis": float(stats.kurtosis(scores)),
            }

        control_stats = calc_stats(control)
        treatment_stats = calc_stats(treatment)

        overall_improvement = (
            (treatment_stats["mean"] - control_stats["mean"])
            if control_stats and treatment_stats
            else 0
        )

        return {
            "control": control_stats,
            "treatment": treatment_stats,
            "overall_improvement": overall_improvement,
            "relative_improvement": overall_improvement / control_stats["mean"]
            if control_stats and control_stats["mean"] != 0
            else 0,
        }

    def _perform_significance_test(
        self, control: list[float], treatment: list[float]
    ) -> dict[str, Any]:
        """Perform statistical significance test"""
        try:
            # Welch's t-test (unequal variances assumed)
            statistic, p_value = stats.ttest_ind(treatment, control, equal_var=False)

            # Effect direction
            effect_direction = (
                "positive" if np.mean(treatment) > np.mean(control) else "negative"
            )

            return {
                "test_type": "welch_t_test",
                "t_statistic": float(statistic),
                "p_value": float(p_value),
                "significant": p_value < self.config.significance_level,
                "significance_level": self.config.significance_level,
                "effect_direction": effect_direction,
                "degrees_of_freedom": len(control) + len(treatment) - 2,
            }
        except Exception as e:
            return {"error": str(e)}

    def _calculate_effect_size(
        self, control: list[float], treatment: list[float]
    ) -> dict[str, Any]:
        """Calculate effect size (Cohen's d)"""
        try:
            control_mean = np.mean(control)
            treatment_mean = np.mean(treatment)

            # Pooled standard deviation
            pooled_std = np.sqrt(
                (
                    (len(control) - 1) * np.var(control, ddof=1)
                    + (len(treatment) - 1) * np.var(treatment, ddof=1)
                )
                / (len(control) + len(treatment) - 2)
            )

            if pooled_std == 0:
                return {"cohens_d": 0, "magnitude": "none", "error": "zero_variance"}

            cohens_d = (treatment_mean - control_mean) / pooled_std
            magnitude = self._interpret_effect_size(abs(cohens_d))

            return {
                "cohens_d": float(cohens_d),
                "magnitude": magnitude,
                "meaningful_effect": abs(cohens_d) >= self.config.min_effect_size,
                "control_mean": float(control_mean),
                "treatment_mean": float(treatment_mean),
                "pooled_std": float(pooled_std),
            }
        except Exception as e:
            return {"error": str(e)}

    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude"""
        if effect_size >= 0.8:
            return "large"
        if effect_size >= 0.5:
            return "medium"
        if effect_size >= 0.2:
            return "small"
        return "negligible"

    def _perform_power_analysis(
        self, control: list[float], treatment: list[float]
    ) -> dict[str, Any]:
        """Perform power analysis"""
        try:
            effect_size_result = self._calculate_effect_size(control, treatment)
            if "error" in effect_size_result:
                return effect_size_result

            effect_size = abs(effect_size_result["cohens_d"])
            n = min(len(control), len(treatment))

            # Simplified power calculation using normal approximation
            z_alpha = stats.norm.ppf(1 - self.config.significance_level / 2)
            z_beta = stats.norm.ppf(self.config.power_threshold)

            # Calculate achieved power
            delta = effect_size * np.sqrt(n / 2)
            z_score = delta - z_alpha
            power = stats.norm.cdf(z_score)

            # Required sample size for desired power
            required_n_per_group = (
                ((z_alpha + z_beta) / effect_size) ** 2 * 2
                if effect_size > 0
                else float("inf")
            )

            return {
                "achieved_power": float(max(0, power)),
                "adequate_power": power >= self.config.power_threshold,
                "required_sample_size_per_group": int(min(required_n_per_group, 10000)),
                "current_sample_size": n,
                "effect_size_used": effect_size,
            }
        except Exception as e:
            return {"error": str(e)}

    def _calculate_confidence_interval(
        self, control: list[float], treatment: list[float]
    ) -> dict[str, Any]:
        """Calculate confidence interval for difference in means"""
        try:
            control_mean = np.mean(control)
            treatment_mean = np.mean(treatment)
            mean_difference = treatment_mean - control_mean

            # Standard error of difference
            se_control = np.std(control, ddof=1) / np.sqrt(len(control))
            se_treatment = np.std(treatment, ddof=1) / np.sqrt(len(treatment))
            se_difference = np.sqrt(se_control**2 + se_treatment**2)

            # Degrees of freedom for Welch's t-test
            df = (se_control**2 + se_treatment**2) ** 2 / (
                se_control**4 / (len(control) - 1)
                + se_treatment**4 / (len(treatment) - 1)
            )

            # Critical value
            alpha = 1 - self.config.confidence_level
            t_critical = stats.t.ppf(1 - alpha / 2, df)

            margin_of_error = t_critical * se_difference

            return {
                "mean_difference": float(mean_difference),
                "standard_error": float(se_difference),
                "margin_of_error": float(margin_of_error),
                "lower_bound": float(mean_difference - margin_of_error),
                "upper_bound": float(mean_difference + margin_of_error),
                "confidence_level": self.config.confidence_level,
                "degrees_of_freedom": float(df),
            }
        except Exception as e:
            return {"error": str(e)}

    def _perform_bayesian_analysis(
        self, control: list[float], treatment: list[float]
    ) -> dict[str, Any]:
        """Perform Bayesian analysis of the difference"""
        try:
            # Simple Bayesian analysis using normal priors
            # This is a simplified implementation - full Bayesian analysis would be more complex

            control_mean = np.mean(control)
            treatment_mean = np.mean(treatment)
            control_var = np.var(control, ddof=1)
            treatment_var = np.var(treatment, ddof=1)

            # Posterior distributions (assuming non-informative priors)
            n_control = len(control)
            n_treatment = len(treatment)

            # Monte Carlo simulation for posterior
            n_samples = 10000

            # Sample from posterior distributions
            control_posterior = np.random.normal(
                control_mean, np.sqrt(control_var / n_control), n_samples
            )
            treatment_posterior = np.random.normal(
                treatment_mean, np.sqrt(treatment_var / n_treatment), n_samples
            )

            # Calculate probability of improvement
            differences = treatment_posterior - control_posterior
            prob_improvement = np.mean(differences > 0)
            prob_meaningful = np.mean(differences > self.config.min_effect_size)

            # Credible interval for difference
            credible_interval = np.percentile(differences, [2.5, 97.5])

            return {
                "probability_of_improvement": float(prob_improvement),
                "probability_of_meaningful_improvement": float(prob_meaningful),
                "credible_interval_95": [
                    float(credible_interval[0]),
                    float(credible_interval[1]),
                ],
                "posterior_mean_difference": float(np.mean(differences)),
                "posterior_std_difference": float(np.std(differences)),
                "monte_carlo_samples": n_samples,
            }
        except Exception as e:
            return {"error": str(e)}

    def _perform_sequential_analysis(
        self, control: list[float], treatment: list[float]
    ) -> dict[str, Any]:
        """Perform sequential analysis for early stopping"""
        try:
            # Simple sequential probability ratio test (SPRT)
            # This is a simplified implementation

            alpha = self.config.significance_level
            beta = 1 - self.config.power_threshold

            # Log likelihood ratio bounds
            a = np.log(beta / (1 - alpha))  # Lower bound (accept H0)
            b = np.log((1 - beta) / alpha)  # Upper bound (accept H1)

            # Calculate cumulative log likelihood ratio
            control_mean = np.mean(control)
            treatment_mean = np.mean(treatment)
            pooled_var = (np.var(control, ddof=1) + np.var(treatment, ddof=1)) / 2

            if pooled_var == 0:
                return {"error": "zero_variance"}

            # Simplified calculation
            log_likelihood_ratio = (
                (treatment_mean - control_mean)
                / np.sqrt(pooled_var)
                * np.sqrt(len(treatment))
            )

            decision = "continue"
            if log_likelihood_ratio <= a:
                decision = "stop_accept_null"
            elif log_likelihood_ratio >= b:
                decision = "stop_accept_alternative"

            return {
                "log_likelihood_ratio": float(log_likelihood_ratio),
                "lower_bound": float(a),
                "upper_bound": float(b),
                "decision": decision,
                "samples_analyzed": len(control) + len(treatment),
                "early_stopping_recommended": decision != "continue",
            }
        except Exception as e:
            return {"error": str(e)}

    def _apply_multiple_testing_correction(
        self, significance_test: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply multiple testing correction"""
        try:
            # Bonferroni correction (conservative)
            # In practice, you'd track multiple tests being run
            num_tests = len(self.active_tests) + 1  # Current test + active tests

            corrected_alpha = self.config.significance_level / num_tests
            original_p_value = significance_test.get("p_value", 1.0)

            return {
                "method": "bonferroni",
                "original_alpha": self.config.significance_level,
                "corrected_alpha": corrected_alpha,
                "original_p_value": original_p_value,
                "corrected_significant": original_p_value < corrected_alpha,
                "number_of_tests": num_tests,
            }
        except Exception as e:
            return {"error": str(e)}

    def _perform_sensitivity_analysis(
        self, control: list[float], treatment: list[float]
    ) -> dict[str, Any]:
        """Perform sensitivity analysis by removing outliers"""
        try:

            def remove_outliers(data, threshold=3):
                z_scores = np.abs(stats.zscore(data))
                return [x for i, x in enumerate(data) if z_scores[i] < threshold]

            # Remove outliers
            control_no_outliers = remove_outliers(
                control, self.config.outliers_threshold
            )
            treatment_no_outliers = remove_outliers(
                treatment, self.config.outliers_threshold
            )

            if len(control_no_outliers) < 3 or len(treatment_no_outliers) < 3:
                return {"error": "insufficient_data_after_outlier_removal"}

            # Recompute key statistics
            original_effect = self._calculate_effect_size(control, treatment)
            outliers_removed_effect = self._calculate_effect_size(
                control_no_outliers, treatment_no_outliers
            )

            original_test = self._perform_significance_test(control, treatment)
            outliers_removed_test = self._perform_significance_test(
                control_no_outliers, treatment_no_outliers
            )

            return {
                "outliers_removed_control": len(control) - len(control_no_outliers),
                "outliers_removed_treatment": len(treatment)
                - len(treatment_no_outliers),
                "original_effect_size": original_effect.get("cohens_d", 0),
                "outliers_removed_effect_size": outliers_removed_effect.get(
                    "cohens_d", 0
                ),
                "original_p_value": original_test.get("p_value", 1),
                "outliers_removed_p_value": outliers_removed_test.get("p_value", 1),
                "robust_to_outliers": abs(
                    original_effect.get("cohens_d", 0)
                    - outliers_removed_effect.get("cohens_d", 0)
                )
                < 0.1,
            }
        except Exception as e:
            return {"error": str(e)}

    def _generate_deployment_recommendation(
        self, analysis: AdvancedAnalysis
    ) -> DeploymentRecommendation:
        """Generate deployment recommendation based on analysis"""
        reasoning = []
        conditions = []
        monitoring_requirements = []

        # Extract key metrics
        significant = analysis.significance_test.get("significant", False)
        p_value = analysis.significance_test.get("p_value", 1.0)
        effect_size = analysis.effect_size.get("cohens_d", 0)
        effect_magnitude = analysis.effect_size.get("magnitude", "negligible")
        power = analysis.power_analysis.get("achieved_power", 0)
        improvement = analysis.descriptive_stats.get("overall_improvement", 0)

        # Bayesian probability if available
        bayesian_prob = None
        if analysis.bayesian_analysis:
            bayesian_prob = analysis.bayesian_analysis.get(
                "probability_of_meaningful_improvement", 0
            )

        # Confidence calculation
        confidence = 0.0

        # Decision logic
        if not significant and effect_magnitude == "negligible":
            level = RecommendationLevel.DO_NOT_DEPLOY
            reasoning.append(
                f"No statistical significance (p={p_value:.3f}) and negligible effect size"
            )
            confidence = 0.9

        elif significant and effect_magnitude in ["large", "medium"]:
            if power >= self.config.power_threshold:
                level = RecommendationLevel.STRONG_DEPLOY
                reasoning.append(
                    f"Strong statistical evidence: p={p_value:.3f}, effect size={effect_magnitude}"
                )
                confidence = 0.95
            else:
                level = RecommendationLevel.CONDITIONAL_DEPLOY
                reasoning.append(f"Significant but underpowered (power={power:.2f})")
                conditions.append("Increase sample size for confirmation")
                confidence = 0.7

        elif significant and effect_magnitude == "small":
            level = RecommendationLevel.MONITOR_DEPLOY
            reasoning.append(
                f"Statistically significant but small effect (d={effect_size:.3f})"
            )
            monitoring_requirements.append(
                "Monitor performance closely after deployment"
            )
            confidence = 0.6

        elif effect_magnitude in ["medium", "large"] and not significant:
            level = RecommendationLevel.CONDITIONAL_DEPLOY
            reasoning.append(
                "Large effect size but not statistically significant (may be underpowered)"
            )
            conditions.append("Collect more data to achieve significance")
            confidence = 0.5

        else:
            level = RecommendationLevel.INSUFFICIENT_DATA
            reasoning.append("Insufficient evidence for reliable recommendation")
            conditions.append("Collect more data or refine experimental design")
            confidence = 0.3

        # Bayesian adjustment
        if bayesian_prob and bayesian_prob > 0.8:
            reasoning.append(
                f"Bayesian analysis shows {bayesian_prob:.1%} probability of meaningful improvement"
            )
            confidence = min(confidence + 0.1, 1.0)

        # Improvement direction check
        if improvement < 0:
            level = RecommendationLevel.DO_NOT_DEPLOY
            reasoning.append("Treatment shows worse performance than control")
            confidence = 0.95

        # Final deployment decision
        deploy = level in [
            RecommendationLevel.STRONG_DEPLOY,
            RecommendationLevel.CONDITIONAL_DEPLOY,
            RecommendationLevel.MONITOR_DEPLOY,
        ]

        return DeploymentRecommendation(
            level=level,
            deploy=deploy,
            confidence=confidence,
            reasoning=reasoning,
            conditions=conditions,
            monitoring_requirements=monitoring_requirements,
        )

    def _perform_quality_checks(
        self, control_results: GroupResults, treatment_results: GroupResults
    ) -> QualityChecks:
        """Perform quality assurance checks"""
        issues = []

        # Group balance check
        total_size = control_results.sample_size + treatment_results.sample_size
        size_diff_percent = (
            abs(control_results.sample_size - treatment_results.sample_size)
            / total_size
            * 100
        )
        group_balance_ok = size_diff_percent <= self.config.balance_tolerance_percent

        if not group_balance_ok:
            issues.append(f"Group imbalance: {size_diff_percent:.1f}% difference")

        # Outlier detection
        all_scores = []
        for results in [control_results.results, treatment_results.results]:
            scores = [
                r.get("improvement_score", 0)
                for r in results
                if r.get("success", False)
            ]
            all_scores.extend(scores)

        if all_scores:
            z_scores = np.abs(stats.zscore(all_scores))
            outliers_detected = np.sum(z_scores > self.config.outliers_threshold)

            if outliers_detected > len(all_scores) * 0.1:  # More than 10% outliers
                issues.append(f"High number of outliers detected: {outliers_detected}")
        else:
            outliers_detected = 0
            issues.append("No successful test cases for outlier analysis")

        # Consistency check
        control_scores = [
            r.get("improvement_score", 0)
            for r in control_results.results
            if r.get("success", False)
        ]
        treatment_scores = [
            r.get("improvement_score", 0)
            for r in treatment_results.results
            if r.get("success", False)
        ]

        if control_scores and treatment_scores:
            control_cv = (
                np.std(control_scores) / np.mean(control_scores)
                if np.mean(control_scores) != 0
                else float("inf")
            )
            treatment_cv = (
                np.std(treatment_scores) / np.mean(treatment_scores)
                if np.mean(treatment_scores) != 0
                else float("inf")
            )
            consistency_score = 1 - min(control_cv, treatment_cv, 1.0)

            if consistency_score < self.config.consistency_threshold:
                issues.append(f"Low consistency score: {consistency_score:.3f}")
        else:
            consistency_score = 0.0
            issues.append("Cannot calculate consistency - insufficient data")

        # Overall data quality score
        quality_factors = [
            1.0 if group_balance_ok else 0.5,
            1.0
            if outliers_detected / len(all_scores) < 0.1
            else 0.7
            if all_scores
            else 0.0,
            consistency_score,
            control_results.successful_tests / control_results.sample_size
            if control_results.sample_size > 0
            else 0.0,
            treatment_results.successful_tests / treatment_results.sample_size
            if treatment_results.sample_size > 0
            else 0.0,
        ]

        data_quality_score = np.mean(quality_factors)

        return QualityChecks(
            group_balance_ok=group_balance_ok,
            outliers_detected=outliers_detected,
            consistency_score=consistency_score,
            data_quality_score=data_quality_score,
            issues=issues,
        )

    def _validate_test_results(
        self, analysis: AdvancedAnalysis, quality_checks: QualityChecks
    ) -> dict[str, Any]:
        """Validate test results for reliability"""
        validation = {
            "overall_valid": True,
            "warnings": [],
            "errors": [],
            "quality_score": quality_checks.data_quality_score,
        }

        # Check sample sizes
        control_n = analysis.descriptive_stats.get("control", {}).get("n", 0)
        treatment_n = analysis.descriptive_stats.get("treatment", {}).get("n", 0)

        if (
            control_n < self.config.minimum_sample_size
            or treatment_n < self.config.minimum_sample_size
        ):
            validation["warnings"].append(
                f"Sample sizes below minimum: control={control_n}, treatment={treatment_n}"
            )

        # Check power
        power = analysis.power_analysis.get("achieved_power", 0)
        if power < self.config.power_threshold:
            validation["warnings"].append(f"Low statistical power: {power:.2f}")

        # Check effect size reliability
        if "error" in analysis.effect_size:
            validation["errors"].append(
                f"Effect size calculation failed: {analysis.effect_size['error']}"
            )
            validation["overall_valid"] = False

        # Check significance test
        if "error" in analysis.significance_test:
            validation["errors"].append(
                f"Significance test failed: {analysis.significance_test['error']}"
            )
            validation["overall_valid"] = False

        # Quality checks
        if quality_checks.data_quality_score < 0.5:
            validation["warnings"].append(
                f"Low data quality score: {quality_checks.data_quality_score:.2f}"
            )

        if quality_checks.issues:
            validation["warnings"].extend(quality_checks.issues)

        return validation

    @handle_database_errors(
        rollback_session=True,
        return_format="none",
        operation_name="persist_test_results",
    )
    async def _persist_test_results(self, test_result: AdvancedTestResult):
        """Persist test results to database"""
        if not self.db_session:
            return

        try:
            # Create AB experiment record
            experiment_data = ABExperimentCreate(
                experiment_name=f"rule_test_{test_result.test_id}",
                control_rule_id=test_result.metadata["original_rule"],
                treatment_rule_id=test_result.metadata["modified_rule"],
                sample_size_control=test_result.control_results.sample_size,
                sample_size_treatment=test_result.treatment_results.sample_size,
                significance_level=self.config.significance_level,
                effect_size=test_result.analysis.effect_size.get("cohens_d", 0),
                p_value=test_result.analysis.significance_test.get("p_value", 1),
                confidence_interval_lower=test_result.analysis.confidence_interval.get(
                    "lower_bound", 0
                ),
                confidence_interval_upper=test_result.analysis.confidence_interval.get(
                    "upper_bound", 0
                ),
                statistical_power=test_result.analysis.power_analysis.get(
                    "achieved_power", 0
                ),
                recommendation=test_result.recommendation.level.value,
                metadata_=test_result.metadata,
            )

            experiment = ABExperiment(**experiment_data.model_dump())
            self.db_session.add(experiment)
            await self.db_session.commit()

            self.logger.info(f"Persisted test results for {test_result.test_id}")

        except Exception as e:
            self.logger.error(f"Failed to persist test results: {e}")
            await self.db_session.rollback()

    def get_test_history(self, limit: int = 10) -> list[AdvancedTestResult]:
        """Get recent test history"""
        return self.test_history[-limit:]

    def get_active_tests(self) -> dict[str, dict[str, Any]]:
        """Get currently active tests"""
        return self.active_tests.copy()

    async def cancel_test(self, test_id: str) -> bool:
        """Cancel an active test"""
        if test_id in self.active_tests:
            self.active_tests[test_id]["status"] = TestStatus.CANCELLED
            del self.active_tests[test_id]
            self.logger.info(f"Cancelled test {test_id}")
            return True
        return False
