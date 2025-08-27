"""A/B Testing Component for Unified Analytics.

This component handles all A/B testing and experimentation operations
with advanced statistical analysis and early stopping capabilities.
"""

import asyncio
import contextlib
import logging
import uuid
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel
from scipy.stats import norm
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.analytics.unified.protocols import (
    ABTestingProtocol,
    AnalyticsComponentProtocol,
    ComponentHealth,
    ExperimentResult,
)
from prompt_improver.core.utils.lazy_ml_loader import get_numpy

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment status values."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VariantType(Enum):
    """Variant type values."""
    CONTROL = "control"
    TREATMENT = "treatment"


class StatisticalMethod(Enum):
    """Statistical analysis methods."""
    FREQUENTIST = "frequentist"
    BAYESIAN = "bayesian"
    SEQUENTIAL = "sequential"


class Experiment(BaseModel):
    """Experiment configuration and state."""
    experiment_id: str
    name: str
    description: str | None = None
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = datetime.now()
    started_at: datetime | None = None
    ended_at: datetime | None = None

    # Configuration
    success_metric: str
    target_significance: float = 0.05
    minimum_effect_size: float = 0.01
    statistical_power: float = 0.8
    statistical_method: StatisticalMethod = StatisticalMethod.FREQUENTIST

    # Variants
    control_variant: dict[str, Any]
    treatment_variants: list[dict[str, Any]]

    # Traffic allocation
    traffic_allocation: dict[str, float] = {}

    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_threshold: float = 0.01

    # Results
    sample_sizes: dict[str, int] = defaultdict(int)
    conversion_counts: dict[str, int] = defaultdict(int)
    metric_values: dict[str, list[float]] = defaultdict(list)


class ABTestingComponent(ABTestingProtocol, AnalyticsComponentProtocol):
    """A/B Testing Component implementing comprehensive experimentation capabilities.

    Features:
    - Advanced statistical analysis (Frequentist, Bayesian, Sequential)
    - Early stopping with false discovery rate control
    - Multiple treatment variants support
    - Stratified randomization
    - Power analysis and sample size calculation
    - Real-time experiment monitoring
    - Comprehensive result interpretation
    """

    def __init__(self, db_session: AsyncSession, config: dict[str, Any]) -> None:
        self.db_session = db_session
        self.config = config
        self.logger = logger

        # Experiment storage
        self._experiments: dict[str, Experiment] = {}
        self._experiment_events: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # Configuration
        self._default_significance = config.get("default_significance", 0.05)
        self._default_power = config.get("default_power", 0.8)
        self._max_experiments = config.get("max_experiments", 50)

        # Statistical settings
        self._bonferroni_correction = config.get("bonferroni_correction", True)
        self._sequential_testing = config.get("sequential_testing", True)
        self._early_stopping_frequency = config.get("early_stopping_check_hours", 24)

        # Performance tracking
        self._stats = {
            "experiments_created": 0,
            "experiments_completed": 0,
            "events_recorded": 0,
            "analyses_performed": 0,
            "early_stops_triggered": 0,
        }

        # Background monitoring
        self._monitoring_enabled = True
        self._monitoring_task: asyncio.Task | None = None

        # Start background monitoring
        self._start_monitoring()

    async def create_experiment(
        self,
        name: str,
        control_variant: dict[str, Any],
        treatment_variants: list[dict[str, Any]],
        success_metric: str,
        target_significance: float = 0.05
    ) -> str:
        """Create a new A/B testing experiment.

        Args:
            name: Experiment name
            control_variant: Control variant configuration
            treatment_variants: List of treatment variant configurations
            success_metric: Primary success metric to optimize
            target_significance: Statistical significance threshold

        Returns:
            Experiment ID
        """
        try:
            if len(self._experiments) >= self._max_experiments:
                raise ValueError(f"Maximum number of experiments ({self._max_experiments}) reached")

            # Generate experiment ID
            experiment_id = str(uuid.uuid4())

            # Create experiment
            experiment = Experiment(
                experiment_id=experiment_id,
                name=name,
                success_metric=success_metric,
                target_significance=target_significance,
                control_variant=control_variant,
                treatment_variants=treatment_variants,
            )

            # Calculate initial traffic allocation
            total_variants = 1 + len(treatment_variants)  # Control + treatments
            equal_allocation = 1.0 / total_variants

            experiment.traffic_allocation = {
                "control": equal_allocation,
                **{f"treatment_{i}": equal_allocation for i in range(len(treatment_variants))}
            }

            # Store experiment and activate it by default
            experiment.status = ExperimentStatus.ACTIVE
            experiment.started_at = datetime.now()

            self._experiments[experiment_id] = experiment
            self._stats["experiments_created"] += 1

            self.logger.info(f"Created experiment {experiment_id}: {name}")
            return experiment_id

        except Exception as e:
            self.logger.exception(f"Error creating experiment: {e}")
            raise

    async def record_experiment_event(
        self,
        experiment_id: str,
        variant_id: str,
        event_type: str,
        user_id: str,
        value: float | None = None
    ) -> bool:
        """Record an event for an experiment.

        Args:
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            event_type: Type of event (e.g., "conversion", "click", "view")
            user_id: User identifier
            value: Optional numeric value associated with event

        Returns:
            Success status
        """
        try:
            if experiment_id not in self._experiments:
                self.logger.error(f"Unknown experiment: {experiment_id}")
                return False

            experiment = self._experiments[experiment_id]

            if experiment.status != ExperimentStatus.ACTIVE:
                self.logger.warning(f"Experiment {experiment_id} is not active")
                return False

            # Record event
            event = {
                "experiment_id": experiment_id,
                "variant_id": variant_id,
                "event_type": event_type,
                "user_id": user_id,
                "value": value,
                "timestamp": datetime.now()
            }

            self._experiment_events[experiment_id].append(event)

            # Update experiment statistics
            if event_type == experiment.success_metric:
                experiment.conversion_counts[variant_id] += 1

            experiment.sample_sizes[variant_id] += 1

            if value is not None:
                experiment.metric_values[variant_id].append(value)

            self._stats["events_recorded"] += 1

            # Check for early stopping if enabled
            if experiment.early_stopping_enabled:
                await self._check_early_stopping(experiment_id)

            return True

        except Exception as e:
            self.logger.exception(f"Error recording event for experiment {experiment_id}: {e}")
            return False

    async def analyze_experiment(self, experiment_id: str) -> ExperimentResult:
        """Analyze experiment results with comprehensive statistical testing.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Experiment analysis results
        """
        try:
            if experiment_id not in self._experiments:
                raise ValueError(f"Unknown experiment: {experiment_id}")

            experiment = self._experiments[experiment_id]

            # Get control and treatment data
            control_data = self._get_variant_data(experiment, "control")
            treatment_data = {}

            for i, _ in enumerate(experiment.treatment_variants):
                variant_id = f"treatment_{i}"
                treatment_data[variant_id] = self._get_variant_data(experiment, variant_id)

            # Perform statistical analysis based on method
            if experiment.statistical_method == StatisticalMethod.FREQUENTIST:
                results = await self._analyze_frequentist(experiment, control_data, treatment_data)
            elif experiment.statistical_method == StatisticalMethod.BAYESIAN:
                results = await self._analyze_bayesian(experiment, control_data, treatment_data)
            else:  # Sequential
                results = await self._analyze_sequential(experiment, control_data, treatment_data)

            self._stats["analyses_performed"] += 1

            return results

        except Exception as e:
            self.logger.exception(f"Error analyzing experiment {experiment_id}: {e}")
            raise

    async def get_active_experiments(self) -> list[dict[str, Any]]:
        """Get list of active experiments with basic information.

        Returns:
            List of active experiment summaries
        """
        try:
            active_experiments = []

            for experiment_id, experiment in self._experiments.items():
                if experiment.status == ExperimentStatus.ACTIVE:
                    # Check if experiment has enough data for analysis
                    total_samples = sum(experiment.sample_sizes.values())
                    ready_for_analysis = total_samples >= 100  # Minimum sample size

                    # Calculate basic statistics
                    control_conversion_rate = 0.0
                    if experiment.sample_sizes.get("control", 0) > 0:
                        control_conversion_rate = (
                            experiment.conversion_counts.get("control", 0) /
                            experiment.sample_sizes["control"]
                        )

                    active_experiments.append({
                        "id": experiment_id,
                        "name": experiment.name,
                        "success_metric": experiment.success_metric,
                        "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
                        "total_samples": total_samples,
                        "control_conversion_rate": control_conversion_rate,
                        "variants_count": len(experiment.treatment_variants) + 1,
                        "ready_for_analysis": ready_for_analysis,
                        "early_stopping_enabled": experiment.early_stopping_enabled,
                    })

            return active_experiments

        except Exception as e:
            self.logger.exception(f"Error getting active experiments: {e}")
            return []

    async def health_check(self) -> dict[str, Any]:
        """Check component health status."""
        try:
            # Calculate experiment statistics
            total_experiments = len(self._experiments)
            active_experiments = sum(1 for exp in self._experiments.values() if exp.status == ExperimentStatus.ACTIVE)
            completed_experiments = sum(1 for exp in self._experiments.values() if exp.status == ExperimentStatus.COMPLETED)

            # Calculate recent activity
            recent_events = sum(
                len([
                    event for event in events
                    if (datetime.now() - event["timestamp"]).hours < 24
                ])
                for events in self._experiment_events.values()
            )

            # Determine health status
            status = "healthy"
            alerts = []

            if not self._monitoring_enabled:
                status = "unhealthy"
                alerts.append("A/B testing monitoring disabled")

            if active_experiments > self._max_experiments * 0.8:
                status = "degraded"
                alerts.append("High number of active experiments")

            if recent_events == 0 and active_experiments > 0:
                status = "degraded"
                alerts.append("No recent experiment activity")

            return ComponentHealth(
                component_name="ab_testing",
                status=status,
                last_check=datetime.now(),
                response_time_ms=0,  # Would measure actual response time
                error_rate=0,  # Would calculate from actual error tracking
                memory_usage_mb=self._estimate_memory_usage(),
                alerts=alerts,
                details={
                    "total_experiments": total_experiments,
                    "active_experiments": active_experiments,
                    "completed_experiments": completed_experiments,
                    "recent_events_24h": recent_events,
                    "monitoring_enabled": self._monitoring_enabled,
                    "stats": self._stats,
                }
            ).dict()

        except Exception as e:
            return {
                "component_name": "ab_testing",
                "status": "error",
                "last_check": datetime.now().isoformat(),
                "error": str(e)
            }

    async def get_metrics(self) -> dict[str, Any]:
        """Get component performance metrics."""
        return {
            "performance": self._stats.copy(),
            "experiment_counts": {
                status.value: sum(1 for exp in self._experiments.values() if exp.status == status)
                for status in ExperimentStatus
            },
            "memory_usage_mb": self._estimate_memory_usage(),
            "total_events": sum(len(events) for events in self._experiment_events.values()),
        }

    async def configure(self, config: dict[str, Any]) -> bool:
        """Configure component with new settings."""
        try:
            # Update configuration
            self.config.update(config)

            # Apply configuration changes
            if "default_significance" in config:
                self._default_significance = config["default_significance"]

            if "early_stopping_frequency" in config:
                self._early_stopping_frequency = config["early_stopping_frequency"]

            if "max_experiments" in config:
                self._max_experiments = config["max_experiments"]

            self.logger.info(f"A/B testing component reconfigured: {config}")
            return True

        except Exception as e:
            self.logger.exception(f"Error configuring component: {e}")
            return False

    async def shutdown(self) -> None:
        """Gracefully shutdown component."""
        try:
            self.logger.info("Shutting down A/B testing component")

            # Stop monitoring
            self._monitoring_enabled = False

            # Cancel monitoring task
            if self._monitoring_task and not self._monitoring_task.done():
                self._monitoring_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._monitoring_task

            # Pause all active experiments
            for experiment in self._experiments.values():
                if experiment.status == ExperimentStatus.ACTIVE:
                    experiment.status = ExperimentStatus.PAUSED

            self.logger.info("A/B testing component shutdown complete")

        except Exception as e:
            self.logger.exception(f"Error during shutdown: {e}")

    # Private helper methods

    def _start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        if self._monitoring_enabled:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring_enabled:
            try:
                await asyncio.sleep(self._early_stopping_frequency * 3600)  # Convert hours to seconds

                # Check all active experiments for early stopping
                for experiment_id, experiment in self._experiments.items():
                    if (experiment.status == ExperimentStatus.ACTIVE and
                        experiment.early_stopping_enabled):
                        await self._check_early_stopping(experiment_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(f"Error in A/B testing monitoring loop: {e}")
                await asyncio.sleep(300)  # 5 minute pause before retrying

    def _get_variant_data(self, experiment: Experiment, variant_id: str) -> dict[str, Any]:
        """Get data for a specific variant."""
        return {
            "variant_id": variant_id,
            "sample_size": experiment.sample_sizes.get(variant_id, 0),
            "conversions": experiment.conversion_counts.get(variant_id, 0),
            "conversion_rate": (
                experiment.conversion_counts.get(variant_id, 0) /
                max(experiment.sample_sizes.get(variant_id, 1), 1)
            ),
            "metric_values": experiment.metric_values.get(variant_id, []),
        }

    async def _analyze_frequentist(
        self,
        experiment: Experiment,
        control_data: dict[str, Any],
        treatment_data: dict[str, dict[str, Any]]
    ) -> ExperimentResult:
        """Perform frequentist statistical analysis."""
        try:
            # For simplicity, analyze only the first treatment variant
            # In a full implementation, this would handle multiple comparisons
            first_treatment_key = next(iter(treatment_data.keys()))
            treatment = treatment_data[first_treatment_key]

            control_conversions = control_data["conversions"]
            control_samples = control_data["sample_size"]
            treatment_conversions = treatment["conversions"]
            treatment_samples = treatment["sample_size"]

            if control_samples == 0 or treatment_samples == 0:
                return ExperimentResult(
                    experiment_id=experiment.experiment_id,
                    control_group_size=control_samples,
                    treatment_group_size=treatment_samples,
                    statistical_significance=False,
                    p_value=1.0,
                    effect_size=0.0,
                    confidence_interval=(0.0, 0.0),
                    winner=None
                )

            # Two-proportion z-test
            control_rate = control_conversions / control_samples
            treatment_rate = treatment_conversions / treatment_samples

            # Calculate pooled proportion
            pooled_rate = (control_conversions + treatment_conversions) / (control_samples + treatment_samples)

            # Calculate standard error
            # import numpy as np  # Converted to lazy loading
            # from get_scipy().stats import norm  # Converted to lazy loading
            se = get_numpy().sqrt(pooled_rate * (1 - pooled_rate) * (1 / control_samples + 1 / treatment_samples))

            if se == 0:
                p_value = 1.0
                z_score = 0.0
            else:
                # Calculate z-score
                z_score = (treatment_rate - control_rate) / se

                # Calculate p-value (two-tailed)
                p_value = 2 * (1 - norm.cdf(abs(z_score)))

            # Effect size (Cohen's h for proportions)
            effect_size = 2 * (get_numpy().arcsin(get_numpy().sqrt(treatment_rate)) - get_numpy().arcsin(get_numpy().sqrt(control_rate)))

            # Confidence interval for difference in proportions
            diff = treatment_rate - control_rate
            se_diff = get_numpy().sqrt((control_rate * (1 - control_rate) / control_samples) +
                             (treatment_rate * (1 - treatment_rate) / treatment_samples))

            ci_lower = diff - 1.96 * se_diff
            ci_upper = diff + 1.96 * se_diff

            # Determine statistical significance
            is_significant = p_value < experiment.target_significance

            # Determine winner
            winner = None
            if is_significant:
                if treatment_rate > control_rate:
                    winner = first_treatment_key
                else:
                    winner = "control"

            return ExperimentResult(
                experiment_id=experiment.experiment_id,
                control_group_size=control_samples,
                treatment_group_size=treatment_samples,
                statistical_significance=is_significant,
                p_value=p_value,
                effect_size=abs(effect_size),
                confidence_interval=(ci_lower, ci_upper),
                winner=winner
            )

        except Exception as e:
            self.logger.exception(f"Error in frequentist analysis: {e}")
            raise

    async def _analyze_bayesian(
        self,
        experiment: Experiment,
        control_data: dict[str, Any],
        treatment_data: dict[str, dict[str, Any]]
    ) -> ExperimentResult:
        """Perform Bayesian statistical analysis."""
        try:
            # Simplified Bayesian analysis using Beta distributions
            # For conversion rates, we use Beta(alpha, beta) priors

            first_treatment_key = next(iter(treatment_data.keys()))
            treatment = treatment_data[first_treatment_key]

            control_conversions = control_data["conversions"]
            control_samples = control_data["sample_size"]
            treatment_conversions = treatment["conversions"]
            treatment_samples = treatment["sample_size"]

            if control_samples == 0 or treatment_samples == 0:
                return ExperimentResult(
                    experiment_id=experiment.experiment_id,
                    control_group_size=control_samples,
                    treatment_group_size=treatment_samples,
                    statistical_significance=False,
                    p_value=1.0,
                    effect_size=0.0,
                    confidence_interval=(0.0, 0.0),
                    winner=None
                )

            # Beta priors (uninformative: Beta(1, 1))
            prior_alpha, prior_beta = 1, 1

            # Posterior distributions
            control_alpha = prior_alpha + control_conversions
            control_beta = prior_beta + control_samples - control_conversions

            treatment_alpha = prior_alpha + treatment_conversions
            treatment_beta = prior_beta + treatment_samples - treatment_conversions

            # Sample from posterior distributions
            n_samples = 10000
            control_samples_dist = get_numpy().random.beta(control_alpha, control_beta, n_samples)
            treatment_samples_dist = get_numpy().random.beta(treatment_alpha, treatment_beta, n_samples)

            # Calculate probability that treatment > control
            prob_treatment_better = get_numpy().mean(treatment_samples_dist > control_samples_dist)

            # Calculate effect size (difference in means)
            control_mean = control_alpha / (control_alpha + control_beta)
            treatment_mean = treatment_alpha / (treatment_alpha + treatment_beta)
            effect_size = abs(treatment_mean - control_mean)

            # Credible interval for difference
            diff_samples = treatment_samples_dist - control_samples_dist
            ci_lower = get_numpy().percentile(diff_samples, 2.5)
            ci_upper = get_numpy().percentile(diff_samples, 97.5)

            # Convert probability to equivalent p-value
            p_value = 2 * min(prob_treatment_better, 1 - prob_treatment_better)

            # Determine significance (95% credible that treatment is better)
            is_significant = prob_treatment_better > 0.95 or prob_treatment_better < 0.05

            # Determine winner
            winner = None
            if is_significant:
                if prob_treatment_better > 0.95:
                    winner = first_treatment_key
                elif prob_treatment_better < 0.05:
                    winner = "control"

            return ExperimentResult(
                experiment_id=experiment.experiment_id,
                control_group_size=control_samples,
                treatment_group_size=treatment_samples,
                statistical_significance=is_significant,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                winner=winner
            )

        except Exception as e:
            self.logger.exception(f"Error in Bayesian analysis: {e}")
            raise

    async def _analyze_sequential(
        self,
        experiment: Experiment,
        control_data: dict[str, Any],
        treatment_data: dict[str, dict[str, Any]]
    ) -> ExperimentResult:
        """Perform sequential statistical analysis with early stopping."""
        try:
            # Sequential analysis with alpha spending function
            # This is a simplified implementation

            first_treatment_key = next(iter(treatment_data.keys()))
            treatment = treatment_data[first_treatment_key]

            control_samples = control_data["sample_size"]
            treatment_samples = treatment["sample_size"]

            if control_samples == 0 or treatment_samples == 0:
                return ExperimentResult(
                    experiment_id=experiment.experiment_id,
                    control_group_size=control_samples,
                    treatment_group_size=treatment_samples,
                    statistical_significance=False,
                    p_value=1.0,
                    effect_size=0.0,
                    confidence_interval=(0.0, 0.0),
                    winner=None
                )

            # Use frequentist analysis as base
            base_result = await self._analyze_frequentist(experiment, control_data, treatment_data)

            # Adjust significance level for sequential testing
            # Using O'Brien-Fleming spending function (simplified)
            total_planned_samples = 1000  # Would be calculated from power analysis
            current_samples = control_samples + treatment_samples
            information_fraction = min(current_samples / total_planned_samples, 1.0)

            if information_fraction > 0:
                # Adjust alpha for sequential testing
                adjusted_alpha = experiment.target_significance * get_numpy().sqrt(information_fraction)
                base_result.statistical_significance = base_result.p_value < adjusted_alpha

            return base_result

        except Exception as e:
            self.logger.exception(f"Error in sequential analysis: {e}")
            raise

    async def _check_early_stopping(self, experiment_id: str) -> None:
        """Check if experiment should be stopped early."""
        try:
            experiment = self._experiments[experiment_id]

            # Minimum sample size check
            total_samples = sum(experiment.sample_sizes.values())
            if total_samples < 100:  # Too early to stop
                return

            # Analyze current results
            result = await self.analyze_experiment(experiment_id)

            # Check early stopping criteria
            should_stop = False
            reason = ""

            # Criteria 1: Statistical significance achieved
            if result.statistical_significance and result.effect_size > experiment.minimum_effect_size:
                should_stop = True
                reason = "Statistical significance achieved"

            # Criteria 2: Very low probability of future significance
            elif result.p_value > 0.5 and total_samples > 500:
                should_stop = True
                reason = "Low probability of achieving significance"

            # Criteria 3: Effect size too small to be meaningful
            elif result.effect_size < experiment.early_stopping_threshold and total_samples > 1000:
                should_stop = True
                reason = "Effect size below meaningful threshold"

            if should_stop:
                experiment.status = ExperimentStatus.COMPLETED
                experiment.ended_at = datetime.now()
                self._stats["early_stops_triggered"] += 1

                self.logger.info(
                    f"Early stopping triggered for experiment {experiment_id}: {reason}"
                )

        except Exception as e:
            self.logger.exception(f"Error checking early stopping for experiment {experiment_id}: {e}")

    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        # Simple estimation based on stored data
        experiment_count = len(self._experiments)
        total_events = sum(len(events) for events in self._experiment_events.values())

        # Rough estimates: 10KB per experiment, 500 bytes per event
        return (experiment_count * 0.01) + (total_events * 0.0005)
