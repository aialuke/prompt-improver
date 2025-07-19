"""A/B Testing Framework for Phase 4 ML Enhancement & Discovery
Advanced experimentation system for rule effectiveness validation
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats
from sklearn.utils import resample  # For bootstrap sampling
from sqlalchemy import select
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.utils.datetime_utils import aware_utc_now

from ..database.models import ABExperiment, ABExperimentCreate, RulePerformance
from ..optimization.early_stopping import (
    AdvancedEarlyStoppingFramework,
    EarlyStoppingConfig,
    StoppingDecision,
    should_stop_experiment,
)
from ..optimization.multi_armed_bandit import (
    ArmResult,
    BanditAlgorithm,
    BanditConfig,
    MultiarmedBanditFramework,
    create_rule_optimization_bandit,
    intelligent_rule_selection,
)
from ..utils.error_handlers import handle_database_errors
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
    - Advanced early stopping mechanisms (SPRT, Group Sequential Design)
    - Futility stopping to prevent wasted resources
    - Automated experiment lifecycle management
    """

    def __init__(self, enable_early_stopping: bool = True, enable_bandits: bool = True):
        self.analytics = AnalyticsService()
        self.min_sample_size = 30  # Minimum sample size per group
        self.alpha = 0.05  # Significance threshold
        self.min_effect_size = 0.1  # Minimum practical effect size (10% improvement)

        # Early stopping framework
        self.enable_early_stopping = enable_early_stopping
        self.early_stopping_framework: AdvancedEarlyStoppingFramework | None = None
        if enable_early_stopping:
            early_stopping_config = EarlyStoppingConfig(
                alpha=self.alpha,
                beta=0.2,  # 80% power
                effect_size_h1=self.min_effect_size,
                max_looks=10,
                enable_futility_stopping=True,
                min_sample_size=self.min_sample_size,
            )
            self.early_stopping_framework = AdvancedEarlyStoppingFramework(
                early_stopping_config
            )

        # Multi-armed bandit framework
        self.enable_bandits = enable_bandits
        self.bandit_framework: MultiarmedBanditFramework | None = None
        self.bandit_experiments: dict[
            str, str
        ] = {}  # Maps experiment_name -> bandit_experiment_id
        if enable_bandits:
            bandit_config = BanditConfig(
                algorithm=BanditAlgorithm.THOMPSON_SAMPLING,
                epsilon=0.1,
                epsilon_decay=0.99,
                min_epsilon=0.01,
                ucb_confidence=2.0,
                prior_alpha=1.0,
                prior_beta=1.0,
                min_samples_per_arm=self.min_sample_size,
                warmup_trials=50,
                enable_regret_tracking=True,
                confidence_level=0.95,
            )
            self.bandit_framework = MultiarmedBanditFramework(bandit_config)

    async def create_experiment(
        self,
        experiment_name: str,
        control_rules: dict[str, Any],
        treatment_rules: dict[str, Any],
        db_session: AsyncSession,
        target_metric: str = "improvement_score",
        description: str | None = None,
        sample_size_per_group: int = 100,
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

        except OSError as e:
            logger.error(f"Database I/O error creating A/B experiment: {e}")
            await db_session.rollback()
            return {"status": "error", "error": str(e)}
        except (ValueError, TypeError) as e:
            logger.error(f"Data validation error creating A/B experiment: {e}")
            await db_session.rollback()
            return {"status": "error", "error": str(e)}
        except KeyboardInterrupt:
            logger.warning("A/B experiment creation cancelled by user")
            await db_session.rollback()
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating A/B experiment: {e}")
            import logging

            logging.exception("Unexpected error in create_experiment")
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
            # Get experiment details - SQLAlchemy metaclass magic for column access
            stmt = select(ABExperiment).where(
                ABExperiment.experiment_id == experiment_id  # type: ignore[arg-type]  # SQLAlchemy column comparison
            )
            result = await db_session.execute(stmt)
            experiment = result.scalar_one_or_none()

            if not experiment:
                return {"status": "error", "error": "Experiment not found"}

            if experiment.status != "running":
                return {
                    "status": "error",
                    "error": f"Experiment is {experiment.status}, cannot analyze early stopping",
                }

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

        except OSError as e:
            logger.error(f"Database I/O error analyzing experiment: {e}")
            return {"status": "error", "error": str(e)}
        except (ValueError, TypeError) as e:
            logger.error(f"Data validation error analyzing experiment: {e}")
            return {"status": "error", "error": str(e)}
        except (AttributeError, KeyError) as e:
            logger.error(f"Data structure error analyzing experiment: {e}")
            return {"status": "error", "error": str(e)}
        except KeyboardInterrupt:
            logger.warning("Experiment analysis cancelled by user")
            raise
        except Exception as e:
            logger.error(f"Unexpected error analyzing experiment: {e}")
            import logging

            logging.exception("Unexpected error in analyze_experiment")
            return {"status": "error", "error": str(e)}

    @handle_database_errors(
        rollback_session=False,  # This is a read operation, no rollback needed
        return_format="none",  # Return empty list on error instead of dict
        operation_name="get_experiment_data",
    )
    async def _get_experiment_data(
        self,
        rule_config: dict[str, Any],
        start_time: datetime,
        target_metric: str,
        db_session: AsyncSession,
    ) -> list[float]:
        """Get performance data for experiment group.

        Now uses standardized error handling decorator that provides:
        - Categorized exception handling (Database I/O, Validation, etc.)
        - Consistent logging patterns
        - Graceful error recovery (returns empty list on any error)
        """
        # Query performance data based on rule configuration
        rule_ids: list[str] = rule_config.get("rule_ids", [])

        if not rule_ids:
            return []

        # SQLAlchemy column access with proper typing - metaclass magic confuses type checker
        stmt = select(RulePerformance).where(
            RulePerformance.rule_id.in_(rule_ids),  # type: ignore[attr-defined]  # SQLAlchemy column method
            RulePerformance.created_at >= start_time,  # type: ignore[arg-type]  # SQLAlchemy column comparison
        )

        result = await db_session.execute(stmt)
        performance_records = result.scalars().all()

        # Extract target metric values - only use existing attributes
        metric_values = []
        for record in performance_records:
            if target_metric == "improvement_score":
                metric_values.append(record.improvement_score or 0.0)
            elif target_metric == "execution_time_ms":
                metric_values.append(record.execution_time_ms or 0.0)
            elif target_metric == "confidence_level":
                metric_values.append(record.confidence_level or 0.0)

        return metric_values

    async def _get_pre_experiment_data(
        self,
        control_rules: dict[str, Any],
        treatment_rules: dict[str, Any],
        experiment_start: datetime,
        target_metric: str,
        db_session: AsyncSession,
    ) -> dict[str, list[float]]:
        """Get pre-experiment data for CUPED analysis"""
        try:
            # Get data from 30 days before experiment start
            from datetime import timedelta

            pre_start = experiment_start - timedelta(days=30)
            pre_end = experiment_start

            # Get control group pre-experiment data
            control_rule_ids: list[str] = control_rules.get("rule_ids", [])
            control_pre_data = []

            if control_rule_ids:
                # SQLAlchemy column access - type checker struggles with metaclass magic
                stmt = select(RulePerformance).where(
                    RulePerformance.rule_id.in_(control_rule_ids),  # type: ignore[attr-defined]  # SQLAlchemy column method
                    RulePerformance.created_at >= pre_start,  # type: ignore[arg-type]  # SQLAlchemy column comparison
                    RulePerformance.created_at < pre_end,  # type: ignore[arg-type]  # SQLAlchemy column comparison
                )
                result = await db_session.execute(stmt)
                control_records = result.scalars().all()

                for record in control_records:
                    if target_metric == "improvement_score":
                        control_pre_data.append(record.improvement_score or 0.0)
                    elif target_metric == "execution_time_ms":
                        control_pre_data.append(record.execution_time_ms or 0.0)
                    elif target_metric == "confidence_level":
                        control_pre_data.append(record.confidence_level or 0.0)

            # Get treatment group pre-experiment data
            treatment_rule_ids: list[str] = treatment_rules.get("rule_ids", [])
            treatment_pre_data = []

            if treatment_rule_ids:
                # SQLAlchemy column access - type checker struggles with metaclass magic
                stmt = select(RulePerformance).where(
                    RulePerformance.rule_id.in_(treatment_rule_ids),  # type: ignore[attr-defined]  # SQLAlchemy column method
                    RulePerformance.created_at >= pre_start,  # type: ignore[arg-type]  # SQLAlchemy column comparison
                    RulePerformance.created_at < pre_end,  # type: ignore[arg-type]  # SQLAlchemy column comparison
                )
                result = await db_session.execute(stmt)
                treatment_records = result.scalars().all()

                for record in treatment_records:
                    if target_metric == "improvement_score":
                        treatment_pre_data.append(record.improvement_score or 0.0)
                    elif target_metric == "execution_time_ms":
                        treatment_pre_data.append(record.execution_time_ms or 0.0)
                    elif target_metric == "confidence_level":
                        treatment_pre_data.append(record.confidence_level or 0.0)

            return {"control": control_pre_data, "treatment": treatment_pre_data}

        except Exception as e:
            logger.warning(f"Failed to get pre-experiment data: {e}")
            return {"control": [], "treatment": []}

    def _perform_statistical_analysis(
        self, control_data: list[float], treatment_data: list[float]
    ) -> ExperimentResult:
        """Perform comprehensive statistical analysis"""
        control_array = np.array(control_data)
        treatment_array = np.array(treatment_data)

        # Basic statistics
        control_mean = float(np.mean(control_array))
        treatment_mean = float(np.mean(treatment_array))

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

        # Statistical significance test (Welch's t-test) - PROPER 2025 attribute access
        ttest_result = stats.ttest_ind(treatment_array, control_array, equal_var=False)
        # Fix: Use proper attribute access, not tuple unpacking - scipy TtestResult object
        t_stat = float(ttest_result.statistic)  # type: ignore[attr-defined]  # scipy TtestResult attribute
        p_value = float(ttest_result.pvalue)  # type: ignore[attr-defined]  # scipy TtestResult attribute

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
            len(control_array),
            len(treatment_array),
            float(np.var(control_array, ddof=1)),
            float(np.var(treatment_array, ddof=1)),
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
                "analysis_timestamp": aware_utc_now().isoformat(),
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
            target_sample_size = (experiment.sample_size_per_group or 100) * 2
            if (
                analysis_result.statistical_significance
                and analysis_result.practical_significance
            ) or experiment.current_sample_size >= target_sample_size:
                experiment.status = "completed"
                experiment.completed_at = aware_utc_now()

            db_session.add(experiment)
            await db_session.commit()

        except OSError as e:
            logger.error(f"Database I/O error storing experiment results: {e}")
            await db_session.rollback()
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Data processing error storing experiment results: {e}")
            await db_session.rollback()
        except KeyboardInterrupt:
            logger.warning("Experiment results storage cancelled by user")
            await db_session.rollback()
            raise
        except Exception as e:
            logger.error(f"Unexpected error storing experiment results: {e}")
            import logging

            logging.exception("Unexpected error in _store_experiment_results")
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

    async def check_early_stopping(
        self,
        experiment_id: str,
        look_number: int = 1,
        db_session: AsyncSession | None = None,
    ) -> dict[str, Any]:
        """Check if experiment should stop early using advanced early stopping mechanisms

        Args:
            experiment_id: UUID of the experiment
            look_number: Current interim analysis number
            db_session: Database session

        Returns:
            Early stopping analysis with recommendation
        """
        if db_session is None:
            return {"status": "error", "error": "Database session is required"}

        if not self.enable_early_stopping or not self.early_stopping_framework:
            return {
                "status": "disabled",
                "message": "Early stopping is not enabled for this service",
                "should_stop": False,
            }

        try:
            # Get experiment details - SQLAlchemy metaclass magic for column access
            stmt = select(ABExperiment).where(
                ABExperiment.experiment_id == experiment_id  # type: ignore[arg-type]  # SQLAlchemy column comparison
            )
            result = await db_session.execute(stmt)
            experiment = result.scalar_one_or_none()

            if not experiment:
                return {"status": "error", "error": "Experiment not found"}

            if experiment.status != "running":
                return {
                    "status": "error",
                    "error": f"Experiment is {experiment.status}, cannot analyze early stopping",
                }

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
                    "should_stop": False,
                    "message": f"Need at least {self.min_sample_size} samples per group for early stopping analysis",
                }

            # Perform early stopping analysis
            early_stopping_result = (
                await self.early_stopping_framework.evaluate_stopping_criteria(
                    experiment_id=experiment_id,
                    control_data=control_data,
                    treatment_data=treatment_data,
                    look_number=look_number,
                    metadata={
                        "experiment_name": experiment.experiment_name,
                        "target_metric": experiment.target_metric,
                        "started_at": experiment.started_at.isoformat(),
                    },
                )
            )

            # Update experiment status if stopping
            should_stop = early_stopping_result.decision != StoppingDecision.CONTINUE
            if should_stop:
                # Update experiment status
                experiment.status = "stopped_early"
                experiment.completed_at = aware_utc_now()

                # Store early stopping results
                if not experiment.results:
                    experiment.results = {}
                experiment.results.update({
                    "early_stopping": {
                        "stopped_early": True,
                        "look_number": look_number,
                        "decision": early_stopping_result.decision.value,
                        "recommendation": early_stopping_result.recommendation,
                        "confidence": early_stopping_result.confidence,
                        "stop_for_efficacy": early_stopping_result.stop_for_efficacy,
                        "stop_for_futility": early_stopping_result.stop_for_futility,
                        "conditional_power": early_stopping_result.conditional_power,
                        "samples_analyzed": early_stopping_result.samples_analyzed,
                        "analysis_time": early_stopping_result.analysis_time.isoformat(),
                    }
                })

                db_session.add(experiment)
                await db_session.commit()

            return {
                "status": "success",
                "experiment_id": experiment_id,
                "look_number": look_number,
                "should_stop": should_stop,
                "early_stopping_analysis": {
                    "decision": early_stopping_result.decision.value,
                    "recommendation": early_stopping_result.recommendation,
                    "confidence": early_stopping_result.confidence,
                    "stop_for_efficacy": early_stopping_result.stop_for_efficacy,
                    "stop_for_futility": early_stopping_result.stop_for_futility,
                    "conditional_power": early_stopping_result.conditional_power,
                    "test_statistic": early_stopping_result.test_statistic,
                    "p_value": early_stopping_result.p_value,
                    "effect_size": early_stopping_result.effect_size,
                    "samples_analyzed": early_stopping_result.samples_analyzed,
                    "estimated_remaining_samples": early_stopping_result.estimated_remaining_samples,
                    "sprt_analysis": {
                        "lower_bound": early_stopping_result.sprt_bounds.lower_bound
                        if early_stopping_result.sprt_bounds
                        else None,
                        "upper_bound": early_stopping_result.sprt_bounds.upper_bound
                        if early_stopping_result.sprt_bounds
                        else None,
                        "log_likelihood_ratio": early_stopping_result.sprt_bounds.log_likelihood_ratio
                        if early_stopping_result.sprt_bounds
                        else None,
                        "decision": early_stopping_result.sprt_bounds.decision.value
                        if early_stopping_result.sprt_bounds
                        else None,
                    }
                    if early_stopping_result.sprt_bounds
                    else None,
                    "group_sequential_analysis": {
                        "information_fraction": early_stopping_result.group_sequential_bounds.information_fraction
                        if early_stopping_result.group_sequential_bounds
                        else None,
                        "alpha_spent": early_stopping_result.group_sequential_bounds.alpha_spent
                        if early_stopping_result.group_sequential_bounds
                        else None,
                        "rejection_boundary": early_stopping_result.group_sequential_bounds.rejection_boundary
                        if early_stopping_result.group_sequential_bounds
                        else None,
                        "futility_boundary": early_stopping_result.group_sequential_bounds.futility_boundary
                        if early_stopping_result.group_sequential_bounds
                        else None,
                        "decision": early_stopping_result.group_sequential_bounds.decision.value
                        if early_stopping_result.group_sequential_bounds
                        else None,
                    }
                    if early_stopping_result.group_sequential_bounds
                    else None,
                },
                "next_actions": self._get_early_stopping_next_actions(
                    early_stopping_result
                ),
            }

        except OSError as e:
            logger.error(f"Database I/O error in early stopping analysis: {e}")
            return {"status": "error", "error": str(e)}
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Data processing error in early stopping analysis: {e}")
            return {"status": "error", "error": str(e)}
        except KeyboardInterrupt:
            logger.warning("Early stopping analysis cancelled by user")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in early stopping analysis: {e}")
            import logging

            logging.exception("Unexpected error in check_early_stopping")
            return {"status": "error", "error": str(e)}

    def _get_early_stopping_next_actions(self, result) -> list[str]:
        """Get recommended next actions based on early stopping analysis"""
        actions = []

        if result.decision == StoppingDecision.STOP_REJECT_NULL:
            actions.append("🎯 Deploy treatment - significant effect detected")
            actions.append("📊 Conduct final analysis and document results")
            actions.append("🔄 Monitor post-deployment performance")

        elif result.decision == StoppingDecision.STOP_FOR_SUPERIORITY:
            actions.append("🚀 Deploy treatment immediately - strong superiority shown")
            actions.append("📈 Scale to full population")
            actions.append("📊 Document superior performance metrics")

        elif result.decision in [
            StoppingDecision.STOP_FOR_FUTILITY,
            StoppingDecision.STOP_ACCEPT_NULL,
        ]:
            actions.append("❌ Do not deploy treatment - no effect or negative effect")
            actions.append("🔍 Investigate why treatment underperformed")
            actions.append("💡 Consider alternative treatment variations")
            actions.append("📋 Archive experiment and document learnings")

        elif result.conditional_power < 0.3:
            actions.append(
                "⚠️ Consider stopping for futility - low probability of success"
            )
            actions.append(
                f"📊 Current conditional power: {result.conditional_power:.1%}"
            )
            actions.append(
                f"🔢 Estimated {result.estimated_remaining_samples} more samples needed"
            )

        else:
            actions.append("▶️ Continue experiment - insufficient evidence for stopping")
            actions.append(
                f"🎯 Target {result.estimated_remaining_samples} additional samples"
            )
            actions.append("📅 Schedule next interim analysis")

        return actions

    async def list_experiments(
        self,
        status: str | None = None,
        limit: int = 20,
        db_session: AsyncSession | None = None,
    ) -> dict[str, Any]:
        """List A/B experiments with optional status filter"""
        if db_session is None:
            return {"status": "error", "error": "Database session is required"}

        try:
            stmt = select(ABExperiment)

            if status:
                # SQLAlchemy column comparison - type checker struggles with metaclass magic
                stmt = stmt.where(ABExperiment.status == status)  # type: ignore[arg-type]  # SQLAlchemy column comparison

            # SQLAlchemy column method access - type checker struggles with metaclass magic
            stmt = stmt.order_by(ABExperiment.started_at.desc()).limit(limit)  # type: ignore[attr-defined]  # SQLAlchemy column method

            result = await db_session.execute(stmt)
            experiments = result.scalars().all()

            experiment_list = []
            for exp in experiments:
                target_sample_size = (exp.sample_size_per_group or 100) * 2
                experiment_list.append({
                    "experiment_id": str(exp.experiment_id),
                    "experiment_name": exp.experiment_name,
                    "status": exp.status,
                    "started_at": exp.started_at.isoformat(),
                    "current_sample_size": exp.current_sample_size,
                    "target_sample_size": target_sample_size,
                    "has_results": exp.results is not None,
                })

            return {
                "status": "success",
                "experiments": experiment_list,
                "total_count": len(experiment_list),
            }

        except OSError as e:
            logger.error(f"Database I/O error listing experiments: {e}")
            return {"status": "error", "error": str(e)}
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Data processing error listing experiments: {e}")
            return {"status": "error", "error": str(e)}
        except KeyboardInterrupt:
            logger.warning("Experiment listing cancelled by user")
            raise
        except Exception as e:
            logger.error(f"Unexpected error listing experiments: {e}")
            import logging

            logging.exception("Unexpected error in list_experiments")
            return {"status": "error", "error": str(e)}

    async def stop_experiment(
        self,
        experiment_id: str,
        reason: str = "Manual stop",
        db_session: AsyncSession | None = None,
    ) -> dict[str, Any]:
        """Stop running experiment"""
        if db_session is None:
            return {"status": "error", "error": "Database session is required"}

        try:
            # SQLAlchemy column comparison - type checker struggles with metaclass magic
            stmt = select(ABExperiment).where(
                ABExperiment.experiment_id == experiment_id  # type: ignore[arg-type]  # SQLAlchemy column comparison
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
            experiment.completed_at = aware_utc_now()

            # Add stop reason to results
            if not experiment.results:
                experiment.results = {}
            experiment.results["stop_reason"] = reason
            experiment.results["stopped_at"] = aware_utc_now().isoformat()

            db_session.add(experiment)
            await db_session.commit()

            return {
                "status": "success",
                "message": f"Experiment {experiment.experiment_name} stopped",
                "stop_reason": reason,
            }

        except OSError as e:
            logger.error(f"Database I/O error stopping experiment: {e}")
            await db_session.rollback()
            return {"status": "error", "error": str(e)}
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Data processing error stopping experiment: {e}")
            await db_session.rollback()
            return {"status": "error", "error": str(e)}
        except KeyboardInterrupt:
            logger.warning("Experiment stopping cancelled by user")
            await db_session.rollback()
            raise
        except Exception as e:
            logger.error(f"Unexpected error stopping experiment: {e}")
            import logging

            logging.exception("Unexpected error in stop_experiment")
            await db_session.rollback()
            return {"status": "error", "error": str(e)}

    def _calculate_bootstrap_ci(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap: int = 10000,
    ) -> tuple[float, float]:
        """Calculate bootstrap confidence interval for treatment effect."""
        try:
            bootstrap_diffs = []

            for _ in range(n_bootstrap):
                # Bootstrap resample both groups
                control_sample = resample(control_data, n_samples=len(control_data))
                treatment_sample = resample(
                    treatment_data, n_samples=len(treatment_data)
                )

                # Calculate difference in means - ensure numpy arrays for proper typing
                control_sample_array = np.asarray(
                    control_sample
                )  # Explicit array conversion for type safety
                treatment_sample_array = np.asarray(
                    treatment_sample
                )  # Explicit array conversion for type safety
                diff = float(
                    np.mean(treatment_sample_array) - np.mean(control_sample_array)
                )
                bootstrap_diffs.append(diff)

            # Calculate percentile-based confidence interval
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            ci_lower = np.percentile(bootstrap_diffs, lower_percentile)
            ci_upper = np.percentile(bootstrap_diffs, upper_percentile)

            return (float(ci_lower), float(ci_upper))

        except (ValueError, TypeError) as e:
            logger.warning(f"Data validation error in bootstrap CI calculation: {e}")
            return (0.0, 0.0)
        except (MemoryError, OverflowError) as e:
            logger.warning(f"Resource error in bootstrap CI calculation: {e}")
            return (0.0, 0.0)
        except KeyboardInterrupt:
            logger.warning("Bootstrap CI calculation cancelled by user")
            raise
        except Exception as e:
            logger.warning(f"Unexpected error in bootstrap CI calculation: {e}")
            import logging

            logging.exception("Unexpected error in _calculate_bootstrap_ci")
            return (0.0, 0.0)

    def _calculate_statistical_power(
        self, control_data: np.ndarray, treatment_data: np.ndarray, effect_size: float
    ) -> float:
        """Calculate statistical power of the test."""
        try:
            # Use effect size and sample sizes to estimate power
            n1, n2 = len(control_data), len(treatment_data)

            # Calculate pooled standard deviation
            pooled_std = np.sqrt(
                (
                    (n1 - 1) * np.var(control_data, ddof=1)
                    + (n2 - 1) * np.var(treatment_data, ddof=1)
                )
                / (n1 + n2 - 2)
            )

            # Calculate non-centrality parameter
            ncp = abs(effect_size) * np.sqrt(n1 * n2 / (n1 + n2))

            # Degrees of freedom
            df = n1 + n2 - 2

            # Critical t-value
            t_critical = stats.t.ppf(1 - self.alpha / 2, df)

            # Power calculation using non-central t-distribution
            power = (
                1
                - stats.nct.cdf(t_critical, df, ncp)
                + stats.nct.cdf(-t_critical, df, ncp)
            )

            # Ensure proper float conversion for NDArray result
            power_float = float(
                np.asarray(power).item()
            )  # Convert NDArray to float safely
            return min(max(power_float, 0.0), 1.0)  # Clamp between 0 and 1

        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.warning(f"Mathematical error in power calculation: {e}")
            return 0.5  # Default moderate power estimate
        except (OverflowError, FloatingPointError) as e:
            logger.warning(f"Numerical error in power calculation: {e}")
            return 0.5  # Default moderate power estimate
        except KeyboardInterrupt:
            logger.warning("Power calculation cancelled by user")
            raise
        except Exception as e:
            logger.warning(f"Unexpected error in power calculation: {e}")
            import logging

            logging.exception("Unexpected error in _calculate_statistical_power")
            return 0.5  # Default moderate power estimate

    def _calculate_minimum_detectable_effect(
        self, n1: int, n2: int, var1: float, var2: float, power: float = 0.8
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

        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.warning(f"Mathematical error in MDE calculation: {e}")
            return 0.1  # Default conservative estimate
        except (OverflowError, FloatingPointError) as e:
            logger.warning(f"Numerical error in MDE calculation: {e}")
            return 0.1  # Default conservative estimate
        except KeyboardInterrupt:
            logger.warning("MDE calculation cancelled by user")
            raise
        except Exception as e:
            logger.warning(f"Unexpected error in MDE calculation: {e}")
            import logging

            logging.exception(
                "Unexpected error in _calculate_minimum_detectable_effect"
            )
            return 0.1  # Default conservative estimate

    def _calculate_bayesian_probability(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        n_simulations: int = 10000,
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
                control_mean, control_std / np.sqrt(len(control_data)), n_simulations
            )
            treatment_samples = np.random.normal(
                treatment_mean,
                treatment_std / np.sqrt(len(treatment_data)),
                n_simulations,
            )

            # Calculate probability that treatment > control
            prob_treatment_better = np.mean(treatment_samples > control_samples)

            return float(prob_treatment_better)

        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.warning(
                f"Mathematical error in Bayesian probability calculation: {e}"
            )
            return 0.5  # Neutral probability
        except (OverflowError, FloatingPointError) as e:
            logger.warning(f"Numerical error in Bayesian probability calculation: {e}")
            return 0.5  # Neutral probability
        except KeyboardInterrupt:
            logger.warning("Bayesian probability calculation cancelled by user")
            raise
        except Exception as e:
            logger.warning(f"Unexpected error in Bayesian probability calculation: {e}")
            import logging

            logging.exception("Unexpected error in _calculate_bayesian_probability")
            return 0.5  # Neutral probability

    def _apply_cuped_analysis(self, treatment_data: dict, control_data: dict) -> dict:
        """CUPED variance reduction technique - 2025 industry standard.

        Reduces variance by 40-50% using pre-experiment data.

        Key insights from 2025 research:
        - Works with any pre-experiment covariate correlated with outcome
        - Maintains unbiased treatment effect estimation
        - Dramatically improves statistical power
        - Essential for modern A/B testing platforms
        """
        try:
            # Combine all data for covariate regression
            all_outcomes = np.concatenate([
                treatment_data["outcome"],
                control_data["outcome"],
            ])
            all_pre_values = np.concatenate([
                treatment_data["pre_value"],
                control_data["pre_value"],
            ])

            # Check for sufficient data
            if len(all_outcomes) < 10 or len(all_pre_values) < 10:
                return {}

            # Step 1: Estimate theta (covariate coefficient) from all data
            # This preserves randomization and remains unbiased
            covariance = np.cov(all_outcomes, all_pre_values)[0, 1]
            pre_variance = np.var(all_pre_values)
            theta = covariance / pre_variance if pre_variance > 0 else 0

            # Step 2: Calculate CUPED-adjusted outcomes
            # Y_cuped = Y - theta * (X_pre - E[X_pre])
            overall_pre_mean = np.mean(all_pre_values)

            treatment_cuped = treatment_data["outcome"] - theta * (
                treatment_data["pre_value"] - overall_pre_mean
            )
            control_cuped = control_data["outcome"] - theta * (
                control_data["pre_value"] - overall_pre_mean
            )

            # Step 3: Standard analysis on CUPED-adjusted outcomes
            treatment_effect_cuped = np.mean(treatment_cuped) - np.mean(control_cuped)

            # Calculate variance reduction achieved
            original_variance = np.var(all_outcomes)
            cuped_variance = np.var(np.concatenate([treatment_cuped, control_cuped]))
            variance_reduction = (
                1 - (cuped_variance / original_variance) if original_variance > 0 else 0
            )

            # Statistical test on adjusted outcomes (maintains Type I error) - proper 2025 scipy usage
            ttest_result = stats.ttest_ind(treatment_cuped, control_cuped)
            # Handle scipy return format properly - can be tuple or TtestResult object
            if hasattr(ttest_result, "statistic"):
                # Modern scipy TtestResult object
                t_stat = float(ttest_result.statistic)  # type: ignore[attr-defined]  # scipy TtestResult attribute
                p_value_float = float(ttest_result.pvalue)  # type: ignore[attr-defined]  # scipy TtestResult attribute
            else:
                # Legacy tuple format - explicit type handling for unpacking
                t_stat_raw, p_value_raw = ttest_result  # type: ignore[misc]  # scipy tuple unpacking
                t_stat = float(t_stat_raw)  # type: ignore[arg-type]  # scipy tuple element to float
                p_value_float = float(p_value_raw)  # type: ignore[arg-type]  # scipy tuple element to float

            # Confidence interval for CUPED estimate
            pooled_se = np.sqrt(
                np.var(treatment_cuped, ddof=1) / len(treatment_cuped)
                + np.var(control_cuped, ddof=1) / len(control_cuped)
            )
            ci_lower = treatment_effect_cuped - 1.96 * pooled_se
            ci_upper = treatment_effect_cuped + 1.96 * pooled_se

            return {
                "treatment_effect_cuped": float(treatment_effect_cuped),
                "p_value": p_value_float,
                "confidence_interval": [float(ci_lower), float(ci_upper)],
                "variance_reduction_percent": float(variance_reduction * 100),
                "theta_coefficient": float(theta),
                "original_effect": float(
                    np.mean(treatment_data["outcome"])
                    - np.mean(control_data["outcome"])
                ),
                "power_improvement_factor": float(1 / np.sqrt(1 - variance_reduction))
                if variance_reduction < 1
                else 1.0,
                "recommendation": self._interpret_cuped_results(
                    float(variance_reduction), p_value_float
                ),
            }

        except Exception as e:
            logger.warning(f"CUPED analysis failed: {e}")
            return {}

    def _interpret_cuped_results(
        self, variance_reduction: float, p_value: float
    ) -> str:
        """Provide actionable interpretation of CUPED results"""
        interpretation = (
            f"CUPED achieved {variance_reduction * 100:.1f}% variance reduction. "
        )

        if variance_reduction > 0.3:
            interpretation += "Excellent covariate - continue using for future tests. "
        elif variance_reduction > 0.1:
            interpretation += "Good covariate - provides meaningful power improvement. "
        else:
            interpretation += (
                "Weak covariate - consider alternative pre-experiment variables. "
            )

        if p_value < 0.05:
            interpretation += (
                "Treatment effect is statistically significant with CUPED adjustment."
            )
        else:
            interpretation += (
                "No significant treatment effect detected even with variance reduction."
            )

        return interpretation

    # Multi-Armed Bandit Methods

    async def create_bandit_experiment(
        self,
        experiment_name: str,
        rule_ids: list[str],
        algorithm: BanditAlgorithm = BanditAlgorithm.THOMPSON_SAMPLING,
        context_features: list[str] | None = None,
        db_session: AsyncSession | None = None,
    ) -> dict[str, Any]:
        """Create a multi-armed bandit experiment for rule optimization

        Args:
            experiment_name: Name of the bandit experiment
            rule_ids: List of rule IDs to test as arms
            algorithm: Bandit algorithm to use
            context_features: Optional contextual features to consider
            db_session: Database session

        Returns:
            Experiment creation result with bandit experiment ID
        """
        if not self.enable_bandits:
            return {"status": "error", "error": "Multi-armed bandits not enabled"}

        if len(rule_ids) < 2:
            return {
                "status": "error",
                "error": "At least 2 rules required for bandit experiment",
            }

        try:
            if not self.bandit_framework:
                return {"status": "error", "error": "Bandit framework not enabled"}

            # Create bandit experiment
            bandit_experiment_id = await self.bandit_framework.create_experiment(
                experiment_name=experiment_name, arms=rule_ids, algorithm=algorithm
            )

            # Store mapping
            self.bandit_experiments[experiment_name] = bandit_experiment_id

            # Also create traditional A/B experiment record for tracking
            experiment_data = ABExperimentCreate(
                experiment_name=f"bandit_{experiment_name}",
                description=f"Multi-armed bandit experiment: {algorithm.value} with {len(rule_ids)} rules",
                control_rules={"rule_id": rule_ids[0], "type": "bandit_arm"},
                treatment_rules={"rule_ids": rule_ids[1:], "type": "bandit_arms"},
                target_metric="improvement_score",
                sample_size_per_group=1000,  # Large sample for bandit
                status="running",
            )

            if db_session:
                experiment = ABExperiment(**experiment_data.dict())
                db_session.add(experiment)
                await db_session.commit()
                await db_session.refresh(experiment)

                traditional_experiment_id = str(experiment.experiment_id)
            else:
                traditional_experiment_id = None

            logger.info(
                f"Created bandit experiment: {experiment_name} with {len(rule_ids)} arms using {algorithm.value}"
            )

            return {
                "status": "success",
                "bandit_experiment_id": bandit_experiment_id,
                "traditional_experiment_id": traditional_experiment_id,
                "experiment_name": experiment_name,
                "algorithm": algorithm.value,
                "arms": rule_ids,
                "message": f"Multi-armed bandit experiment created with {len(rule_ids)} arms",
            }

        except Exception as e:
            logger.error(f"Error creating bandit experiment: {e}")
            if db_session:
                await db_session.rollback()
            return {"status": "error", "error": str(e)}

    async def select_bandit_arm(
        self, experiment_name: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Select an arm (rule) using bandit algorithm

        Args:
            experiment_name: Name of the bandit experiment
            context: Optional contextual information about the prompt

        Returns:
            Selected arm information with confidence
        """
        if not self.enable_bandits:
            return {"status": "error", "error": "Multi-armed bandits not enabled"}

        if experiment_name not in self.bandit_experiments:
            return {
                "status": "error",
                "error": f"Unknown bandit experiment: {experiment_name}",
            }

        try:
            if not self.bandit_framework:
                return {"status": "error", "error": "Bandit framework not enabled"}

            bandit_experiment_id = self.bandit_experiments[experiment_name]

            # Select arm using bandit algorithm
            arm_result = await self.bandit_framework.select_arm(
                bandit_experiment_id, context
            )

            logger.debug(
                f"Bandit selected arm {arm_result.arm_id} for experiment {experiment_name} with confidence {arm_result.confidence:.3f}"
            )

            return {
                "status": "success",
                "selected_rule_id": arm_result.arm_id,
                "confidence": arm_result.confidence,
                "uncertainty": arm_result.uncertainty,
                "experiment_id": bandit_experiment_id,
                "context_used": context is not None,
                "metadata": arm_result.metadata,
            }

        except Exception as e:
            logger.error(f"Error selecting bandit arm: {e}")
            return {"status": "error", "error": str(e)}

    async def update_bandit_reward(
        self,
        experiment_name: str,
        selected_rule_id: str,
        reward: float,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update bandit with observed reward from rule application

        Args:
            experiment_name: Name of the bandit experiment
            selected_rule_id: Rule ID that was applied
            reward: Observed reward (improvement score, user rating, etc.)
            context: Optional contextual information

        Returns:
            Update result with current bandit state
        """
        if not self.enable_bandits:
            return {"status": "error", "error": "Multi-armed bandits not enabled"}

        if experiment_name not in self.bandit_experiments:
            return {
                "status": "error",
                "error": f"Unknown bandit experiment: {experiment_name}",
            }

        try:
            if not self.bandit_framework:
                return {"status": "error", "error": "Bandit framework not enabled"}

            bandit_experiment_id = self.bandit_experiments[experiment_name]

            # Create arm result for update
            context_array = None
            if context:
                # Convert context to array if needed
                context_array = self.bandit_framework._context_to_array(context)

            arm_result = ArmResult(
                arm_id=selected_rule_id,
                reward=reward,
                context=context_array,
                metadata={"experiment_name": experiment_name},
            )

            # Update bandit with reward
            await self.bandit_framework.update_reward(
                bandit_experiment_id, arm_result, reward
            )

            # Get updated experiment summary
            summary = self.bandit_framework.get_experiment_summary(bandit_experiment_id)

            logger.debug(
                f"Updated bandit reward for {selected_rule_id}: {reward:.3f}, total trials: {summary['total_trials']}"
            )

            return {
                "status": "success",
                "updated_rule_id": selected_rule_id,
                "reward": reward,
                "total_trials": summary["total_trials"],
                "average_reward": summary["average_reward"],
                "best_arm": summary["best_arm"],
                "regret": summary.get("total_regret", 0.0),
                "message": f"Bandit updated with reward {reward:.3f} for rule {selected_rule_id}",
            }

        except Exception as e:
            logger.error(f"Error updating bandit reward: {e}")
            return {"status": "error", "error": str(e)}

    async def get_bandit_performance(self, experiment_name: str) -> dict[str, Any]:
        """Get comprehensive performance analysis of bandit experiment

        Args:
            experiment_name: Name of the bandit experiment

        Returns:
            Detailed performance metrics and recommendations
        """
        if not self.enable_bandits:
            return {"status": "error", "error": "Multi-armed bandits not enabled"}

        if experiment_name not in self.bandit_experiments:
            return {
                "status": "error",
                "error": f"Unknown bandit experiment: {experiment_name}",
            }

        try:
            if not self.bandit_framework:
                return {"status": "error", "error": "Bandit framework not enabled"}

            bandit_experiment_id = self.bandit_experiments[experiment_name]
            summary = self.bandit_framework.get_experiment_summary(bandit_experiment_id)

            # Calculate additional metrics
            arm_stats = summary["arm_statistics"]

            # Find top performing arms
            top_arms = sorted(
                arm_stats.items(), key=lambda x: x[1]["mean_reward"], reverse=True
            )[:3]

            # Calculate exploration vs exploitation ratio
            total_pulls = sum(stats["pulls"] for stats in arm_stats.values())
            best_arm_pulls = arm_stats[summary["best_arm"]]["pulls"]
            exploitation_ratio = best_arm_pulls / max(1, total_pulls)

            # Generate recommendations
            recommendations = []

            if summary["total_trials"] < 100:
                recommendations.append(
                    "Experiment is in early stage - continue collecting data"
                )
            elif exploitation_ratio > 0.8:
                recommendations.append(
                    "High exploitation detected - consider more exploration"
                )
            elif exploitation_ratio < 0.3:
                recommendations.append(
                    "High exploration detected - algorithm may be converging"
                )

            if summary["total_regret"] > summary["total_reward"] * 0.2:
                recommendations.append(
                    "High regret detected - consider switching to more exploitative algorithm"
                )

            return {
                "status": "success",
                "experiment_summary": summary,
                "top_performing_arms": top_arms,
                "exploitation_ratio": exploitation_ratio,
                "exploration_ratio": 1 - exploitation_ratio,
                "recommendations": recommendations,
                "convergence_indicator": exploitation_ratio > 0.7,
                "statistical_confidence": "high"
                if summary["total_trials"] > 200
                else "medium"
                if summary["total_trials"] > 50
                else "low",
            }

        except Exception as e:
            logger.error(f"Error getting bandit performance: {e}")
            return {"status": "error", "error": str(e)}

    async def stop_bandit_experiment(self, experiment_name: str) -> dict[str, Any]:
        """Stop a running bandit experiment

        Args:
            experiment_name: Name of the bandit experiment

        Returns:
            Final experiment results and recommendations
        """
        if not self.enable_bandits:
            return {"status": "error", "error": "Multi-armed bandits not enabled"}

        if experiment_name not in self.bandit_experiments:
            return {
                "status": "error",
                "error": f"Unknown bandit experiment: {experiment_name}",
            }

        try:
            if not self.bandit_framework:
                return {"status": "error", "error": "Bandit framework not enabled"}

            bandit_experiment_id = self.bandit_experiments[experiment_name]

            # Get final performance before stopping
            final_performance = await self.get_bandit_performance(experiment_name)

            # Stop the experiment
            self.bandit_framework.stop_experiment(bandit_experiment_id)

            logger.info(f"Stopped bandit experiment: {experiment_name}")

            return {
                "status": "success",
                "experiment_name": experiment_name,
                "final_performance": final_performance,
                "message": f"Bandit experiment {experiment_name} stopped successfully",
            }

        except Exception as e:
            logger.error(f"Error stopping bandit experiment: {e}")
            return {"status": "error", "error": str(e)}


# Singleton instance for easy access
_ab_testing_service = None


async def get_ab_testing_service() -> ABTestingService:
    """Get singleton ABTestingService instance"""
    global _ab_testing_service
    if _ab_testing_service is None:
        _ab_testing_service = ABTestingService()
    return _ab_testing_service
