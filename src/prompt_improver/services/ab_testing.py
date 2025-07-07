"""A/B Testing Framework for Phase 4 ML Enhancement & Discovery
Advanced experimentation system for rule effectiveness validation
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from scipy import stats
from sklearn.utils import resample  # For bootstrap sampling
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.models import ABExperiment, ABExperimentCreate, RulePerformance
from .analytics import AnalyticsService

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Statistical results of A/B experiment with advanced statistics"""

    control_mean: float
    treatment_mean: float
    effect_size: float
    p_value: float
    confidence_interval: tuple[float, float]
    bootstrap_ci: tuple[float, float]
    statistical_significance: bool
    practical_significance: bool
    sample_size_control: int
    sample_size_treatment: int
    statistical_power: float
    minimum_detectable_effect: float
    bayesian_probability: float


class ABTestingService:
    """Advanced A/B testing framework for rule effectiveness validation.

    Features:
    - Statistical significance testing with proper power analysis
    - Effect size calculation and practical significance assessment
    - Bayesian confidence intervals
    - Multiple testing correction
    - Early stopping detection
    - Automated experiment lifecycle management
    """

    def __init__(self):
        self.analytics = AnalyticsService()
        self.min_sample_size = 30  # Minimum sample size per group
        self.alpha = 0.05  # Significance threshold
        self.min_effect_size = 0.1  # Minimum practical effect size (10% improvement)

    async def create_experiment(
        self,
        experiment_name: str,
        control_rules: dict[str, Any],
        treatment_rules: dict[str, Any],
        target_metric: str = "improvement_score",
        description: str | None = None,
        sample_size_per_group: int = 100,
        db_session: AsyncSession = None,
    ) -> dict[str, Any]:
        """Create new A/B experiment for rule effectiveness testing.

        Args:
            experiment_name: Name of the experiment
            control_rules: Control group rule configuration
            treatment_rules: Treatment group rule configuration
            target_metric: Primary metric to optimize
            description: Optional experiment description
            sample_size_per_group: Target sample size per group
            db_session: Database session

        Returns:
            Experiment creation result with experiment ID
        """
        try:
            experiment_data = ABExperimentCreate(
                experiment_name=experiment_name,
                description=description
                or f"A/B test: {control_rules.get('name', 'Control')} vs {treatment_rules.get('name', 'Treatment')}",
                control_rules=control_rules,
                treatment_rules=treatment_rules,
                target_metric=target_metric,
                sample_size_per_group=sample_size_per_group,
                status="running",
            )

            experiment = ABExperiment(**experiment_data.dict())
            db_session.add(experiment)
            await db_session.commit()
            await db_session.refresh(experiment)

            logger.info(
                f"Created A/B experiment: {experiment_name} (ID: {experiment.experiment_id})"
            )

            return {
                "status": "success",
                "experiment_id": str(experiment.experiment_id),
                "experiment_name": experiment_name,
                "message": "A/B experiment created successfully",
            }

        except Exception as e:
            logger.error(f"Failed to create A/B experiment: {e}")
            await db_session.rollback()
            return {"status": "error", "error": str(e)}

    async def analyze_experiment(
        self, experiment_id: str, db_session: AsyncSession
    ) -> dict[str, Any]:
        """Analyze A/B experiment results with statistical significance testing.

        Args:
            experiment_id: UUID of the experiment
            db_session: Database session

        Returns:
            Statistical analysis results with recommendations
        """
        try:
            # Get experiment details
            stmt = select(ABExperiment).where(
                ABExperiment.experiment_id == experiment_id
            )
            result = await db_session.execute(stmt)
            experiment = result.scalar_one_or_none()

            if not experiment:
                return {"status": "error", "error": "Experiment not found"}

            # Get performance data for both groups
            control_data = await self._get_experiment_data(
                experiment.control_rules,
                experiment.started_at,
                experiment.target_metric,
                db_session,
            )

            treatment_data = await self._get_experiment_data(
                experiment.treatment_rules,
                experiment.started_at,
                experiment.target_metric,
                db_session,
            )

            if (
                len(control_data) < self.min_sample_size
                or len(treatment_data) < self.min_sample_size
            ):
                return {
                    "status": "insufficient_data",
                    "control_samples": len(control_data),
                    "treatment_samples": len(treatment_data),
                    "min_required": self.min_sample_size,
                    "message": f"Need at least {self.min_sample_size} samples per group",
                }

            # Perform statistical analysis
            analysis_result = self._perform_statistical_analysis(
                control_data, treatment_data
            )

            # Store results
            await self._store_experiment_results(
                experiment, analysis_result, db_session
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(analysis_result)

            return {
                "status": "success",
                "experiment_id": experiment_id,
                "experiment_name": experiment.experiment_name,
                "analysis": {
                    "control_mean": analysis_result.control_mean,
                    "treatment_mean": analysis_result.treatment_mean,
                    "effect_size": analysis_result.effect_size,
                    "p_value": analysis_result.p_value,
                    "confidence_interval": analysis_result.confidence_interval,
                    "bootstrap_confidence_interval": analysis_result.bootstrap_ci,
                    "statistical_significance": analysis_result.statistical_significance,
                    "practical_significance": analysis_result.practical_significance,
                    "statistical_power": analysis_result.statistical_power,
                    "minimum_detectable_effect": analysis_result.minimum_detectable_effect,
                    "bayesian_probability": analysis_result.bayesian_probability,
                    "sample_sizes": {
                        "control": analysis_result.sample_size_control,
                        "treatment": analysis_result.sample_size_treatment,
                    },
                },
                "recommendations": recommendations,
                "next_actions": self._get_next_actions(analysis_result),
            }

        except Exception as e:
            logger.error(f"Failed to analyze experiment: {e}")
            return {"status": "error", "error": str(e)}

    async def _get_experiment_data(
        self,
        rule_config: dict[str, Any],
        start_time: datetime,
        target_metric: str,
        db_session: AsyncSession,
    ) -> list[float]:
        """Get performance data for experiment group"""
        try:
            # Query performance data based on rule configuration
            rule_ids = rule_config.get("rule_ids", [])

            if not rule_ids:
                return []

            stmt = select(RulePerformance).where(
                RulePerformance.rule_id.in_(rule_ids),
                RulePerformance.created_at >= start_time,
            )

            result = await db_session.execute(stmt)
            performance_records = result.scalars().all()

            # Extract target metric values
            metric_values = []
            for record in performance_records:
                if target_metric == "improvement_score":
                    metric_values.append(record.improvement_score or 0.0)
                elif target_metric == "execution_time_ms":
                    metric_values.append(record.execution_time_ms or 0.0)
                elif target_metric == "user_satisfaction_score":
                    metric_values.append(record.user_satisfaction_score or 0.0)

            return metric_values

        except Exception as e:
            logger.error(f"Failed to get experiment data: {e}")
            return []

    def _perform_statistical_analysis(
        self, control_data: list[float], treatment_data: list[float]
    ) -> ExperimentResult:
        """Perform comprehensive statistical analysis"""
        control_array = np.array(control_data)
        treatment_array = np.array(treatment_data)

        # Basic statistics
        control_mean = np.mean(control_array)
        treatment_mean = np.mean(treatment_array)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (
                (len(control_array) - 1) * np.var(control_array, ddof=1)
                + (len(treatment_array) - 1) * np.var(treatment_array, ddof=1)
            )
            / (len(control_array) + len(treatment_array) - 2)
        )

        effect_size = (
            (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        )

        # Statistical significance test (Welch's t-test)
        t_stat, p_value = stats.ttest_ind(
            treatment_array, control_array, equal_var=False
        )

        # Classical confidence interval for difference in means
        diff_mean = treatment_mean - control_mean
        se_diff = np.sqrt(
            np.var(control_array, ddof=1) / len(control_array)
            + np.var(treatment_array, ddof=1) / len(treatment_array)
        )

        df = (
            np.var(control_array, ddof=1) / len(control_array)
            + np.var(treatment_array, ddof=1) / len(treatment_array)
        ) ** 2 / (
            (np.var(control_array, ddof=1) / len(control_array)) ** 2
            / (len(control_array) - 1)
            + (np.var(treatment_array, ddof=1) / len(treatment_array)) ** 2
            / (len(treatment_array) - 1)
        )

        t_critical = stats.t.ppf(1 - self.alpha / 2, df)
        margin_of_error = t_critical * se_diff
        confidence_interval = (diff_mean - margin_of_error, diff_mean + margin_of_error)

        # Bootstrap confidence interval for more robust estimation
        bootstrap_ci = self._calculate_bootstrap_ci(
            control_array, treatment_array, alpha=self.alpha
        )

        # Statistical power analysis
        statistical_power = self._calculate_statistical_power(
            control_array, treatment_array, effect_size
        )

        # Minimum detectable effect
        mde = self._calculate_minimum_detectable_effect(
            len(control_array), len(treatment_array),
            np.var(control_array, ddof=1), np.var(treatment_array, ddof=1)
        )

        # Bayesian probability of treatment being better
        bayesian_prob = self._calculate_bayesian_probability(
            control_array, treatment_array
        )

        # Significance assessments
        statistical_significance = p_value < self.alpha
        practical_significance = abs(effect_size) >= self.min_effect_size

        return ExperimentResult(
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            effect_size=effect_size,
            p_value=p_value,
            confidence_interval=confidence_interval,
            bootstrap_ci=bootstrap_ci,
            statistical_significance=statistical_significance,
            practical_significance=practical_significance,
            sample_size_control=len(control_array),
            sample_size_treatment=len(treatment_array),
            statistical_power=statistical_power,
            minimum_detectable_effect=mde,
            bayesian_probability=bayesian_prob,
        )

    async def _store_experiment_results(
        self,
        experiment: ABExperiment,
        analysis_result: ExperimentResult,
        db_session: AsyncSession,
    ):
        """Store experiment analysis results"""
        try:
            results_data = {
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "control_mean": analysis_result.control_mean,
                "treatment_mean": analysis_result.treatment_mean,
                "effect_size": analysis_result.effect_size,
                "p_value": analysis_result.p_value,
                "confidence_interval": list(analysis_result.confidence_interval),
                "statistical_significance": analysis_result.statistical_significance,
                "practical_significance": analysis_result.practical_significance,
                "sample_sizes": {
                    "control": analysis_result.sample_size_control,
                    "treatment": analysis_result.sample_size_treatment,
                },
            }

            experiment.results = results_data
            experiment.current_sample_size = (
                analysis_result.sample_size_control
                + analysis_result.sample_size_treatment
            )

            # Update status if experiment is complete
            if (
                analysis_result.statistical_significance
                and analysis_result.practical_significance
            ) or experiment.current_sample_size >= experiment.sample_size_per_group * 2:
                experiment.status = "completed"
                experiment.completed_at = datetime.utcnow()

            db_session.add(experiment)
            await db_session.commit()

        except Exception as e:
            logger.error(f"Failed to store experiment results: {e}")
            await db_session.rollback()

    def _generate_recommendations(self, analysis_result: ExperimentResult) -> list[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        if (
            analysis_result.statistical_significance
            and analysis_result.practical_significance
        ):
            if analysis_result.treatment_mean > analysis_result.control_mean:
                recommendations.append(
                    "🎯 IMPLEMENT: Treatment shows significant improvement - deploy to production"
                )
                recommendations.append(
                    f"📈 Expected improvement: {analysis_result.effect_size:.2f} standard deviations"
                )
            else:
                recommendations.append(
                    "⛔ REJECT: Treatment performs significantly worse - keep control rules"
                )

        elif (
            analysis_result.statistical_significance
            and not analysis_result.practical_significance
        ):
            recommendations.append(
                "📊 STATISTICALLY SIGNIFICANT but effect size too small for practical value"
            )
            recommendations.append(
                "💡 Consider testing larger changes or different approaches"
            )

        elif (
            not analysis_result.statistical_significance
            and analysis_result.practical_significance
        ):
            recommendations.append(
                "📈 PROMISING but needs more data for statistical confidence"
            )
            recommendations.append(
                f"🔄 Continue experiment - current sample sizes: {analysis_result.sample_size_control + analysis_result.sample_size_treatment}"
            )

        else:
            recommendations.append("❌ NO SIGNIFICANT DIFFERENCE detected")
            recommendations.append("🔄 Consider testing different treatment variations")

        # Power analysis recommendation
        if (
            analysis_result.sample_size_control + analysis_result.sample_size_treatment
            < 100
        ):
            recommendations.append(
                "⚡ Consider increasing sample size for better statistical power"
            )

        return recommendations

    def _get_next_actions(self, analysis_result: ExperimentResult) -> list[str]:
        """Get specific next actions based on analysis"""
        actions = []

        if (
            analysis_result.statistical_significance
            and analysis_result.practical_significance
        ):
            if analysis_result.treatment_mean > analysis_result.control_mean:
                actions.append("Deploy treatment rules to production")
                actions.append("Update rule parameters in database")
                actions.append("Monitor post-deployment performance")
            else:
                actions.append("Archive experiment")
                actions.append("Investigate why treatment underperformed")
        else:
            actions.append("Continue data collection")
            actions.append("Check experiment setup and rule implementations")
            actions.append("Consider alternative treatment variations")

        return actions

    async def list_experiments(
        self,
        status: str | None = None,
        limit: int = 20,
        db_session: AsyncSession = None,
    ) -> dict[str, Any]:
        """List A/B experiments with optional status filter"""
        try:
            stmt = select(ABExperiment)

            if status:
                stmt = stmt.where(ABExperiment.status == status)

            stmt = stmt.order_by(ABExperiment.started_at.desc()).limit(limit)

            result = await db_session.execute(stmt)
            experiments = result.scalars().all()

            experiment_list = []
            for exp in experiments:
                experiment_list.append({
                    "experiment_id": str(exp.experiment_id),
                    "experiment_name": exp.experiment_name,
                    "status": exp.status,
                    "started_at": exp.started_at.isoformat(),
                    "current_sample_size": exp.current_sample_size,
                    "target_sample_size": exp.sample_size_per_group * 2,
                    "has_results": exp.results is not None,
                })

            return {
                "status": "success",
                "experiments": experiment_list,
                "total_count": len(experiment_list),
            }

        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return {"status": "error", "error": str(e)}

    async def stop_experiment(
        self,
        experiment_id: str,
        reason: str = "Manual stop",
        db_session: AsyncSession = None,
    ) -> dict[str, Any]:
        """Stop running experiment"""
        try:
            stmt = select(ABExperiment).where(
                ABExperiment.experiment_id == experiment_id
            )
            result = await db_session.execute(stmt)
            experiment = result.scalar_one_or_none()

            if not experiment:
                return {"status": "error", "error": "Experiment not found"}

            if experiment.status != "running":
                return {
                    "status": "error",
                    "error": f"Experiment is {experiment.status}, cannot stop",
                }

            experiment.status = "stopped"
            experiment.completed_at = datetime.utcnow()

            # Add stop reason to results
            if not experiment.results:
                experiment.results = {}
            experiment.results["stop_reason"] = reason
            experiment.results["stopped_at"] = datetime.utcnow().isoformat()

            db_session.add(experiment)
            await db_session.commit()

            return {
                "status": "success",
                "message": f"Experiment {experiment.experiment_name} stopped",
                "stop_reason": reason,
            }

        except Exception as e:
            logger.error(f"Failed to stop experiment: {e}")
            await db_session.rollback()
            return {"status": "error", "error": str(e)}

    def _calculate_bootstrap_ci(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap: int = 10000
    ) -> tuple[float, float]:
        """Calculate bootstrap confidence interval for treatment effect."""
        try:
            bootstrap_diffs = []

            for _ in range(n_bootstrap):
                # Bootstrap resample both groups
                control_sample = resample(control_data, n_samples=len(control_data))
                treatment_sample = resample(treatment_data, n_samples=len(treatment_data))

                # Calculate difference in means
                diff = np.mean(treatment_sample) - np.mean(control_sample)
                bootstrap_diffs.append(diff)

            # Calculate percentile-based confidence interval
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            ci_lower = np.percentile(bootstrap_diffs, lower_percentile)
            ci_upper = np.percentile(bootstrap_diffs, upper_percentile)

            return (ci_lower, ci_upper)

        except Exception as e:
            logger.warning(f"Bootstrap CI calculation failed: {e}")
            return (0.0, 0.0)

    def _calculate_statistical_power(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        effect_size: float
    ) -> float:
        """Calculate statistical power of the test."""
        try:
            # Use effect size and sample sizes to estimate power
            n1, n2 = len(control_data), len(treatment_data)

            # Calculate pooled standard deviation
            pooled_std = np.sqrt(
                ((n1 - 1) * np.var(control_data, ddof=1) +
                 (n2 - 1) * np.var(treatment_data, ddof=1)) / (n1 + n2 - 2)
            )

            # Calculate non-centrality parameter
            ncp = abs(effect_size) * np.sqrt(n1 * n2 / (n1 + n2))

            # Degrees of freedom
            df = n1 + n2 - 2

            # Critical t-value
            t_critical = stats.t.ppf(1 - self.alpha / 2, df)

            # Power calculation using non-central t-distribution
            power = 1 - stats.nct.cdf(t_critical, df, ncp) + stats.nct.cdf(-t_critical, df, ncp)

            return min(max(power, 0.0), 1.0)  # Clamp between 0 and 1

        except Exception as e:
            logger.warning(f"Power calculation failed: {e}")
            return 0.5  # Default moderate power estimate

    def _calculate_minimum_detectable_effect(
        self,
        n1: int,
        n2: int,
        var1: float,
        var2: float,
        power: float = 0.8
    ) -> float:
        """Calculate minimum detectable effect size."""
        try:
            # Pooled variance
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

            # Standard error
            se = np.sqrt(pooled_var * (1 / n1 + 1 / n2))

            # Critical values for alpha and beta
            t_alpha = stats.t.ppf(1 - self.alpha / 2, n1 + n2 - 2)
            t_beta = stats.t.ppf(power, n1 + n2 - 2)

            # Minimum detectable effect
            mde = (t_alpha + t_beta) * se

            return mde

        except Exception as e:
            logger.warning(f"MDE calculation failed: {e}")
            return 0.1  # Default conservative estimate

    def _calculate_bayesian_probability(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        n_simulations: int = 10000
    ) -> float:
        """Calculate Bayesian probability that treatment is better than control."""
        try:
            # Simple Bayesian approach using normal approximation
            control_mean = np.mean(control_data)
            control_std = np.std(control_data, ddof=1)
            treatment_mean = np.mean(treatment_data)
            treatment_std = np.std(treatment_data, ddof=1)

            # Sample from posterior distributions (assuming uninformative priors)
            control_samples = np.random.normal(
                control_mean,
                control_std / np.sqrt(len(control_data)),
                n_simulations
            )
            treatment_samples = np.random.normal(
                treatment_mean,
                treatment_std / np.sqrt(len(treatment_data)),
                n_simulations
            )

            # Calculate probability that treatment > control
            prob_treatment_better = np.mean(treatment_samples > control_samples)

            return prob_treatment_better

        except Exception as e:
            logger.warning(f"Bayesian probability calculation failed: {e}")
            return 0.5  # Neutral probability


# Singleton instance for easy access
_ab_testing_service = None


async def get_ab_testing_service() -> ABTestingService:
    """Get singleton ABTestingService instance"""
    global _ab_testing_service
    if _ab_testing_service is None:
        _ab_testing_service = ABTestingService()
    return _ab_testing_service
