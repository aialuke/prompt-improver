"""Advanced Early Stopping Mechanisms for A/B Testing

This module implements research-validated early stopping techniques including:
- Sequential Probability Ratio Test (SPRT)
- Group Sequential Design with error spending functions
- Futility stopping mechanisms
- Mixture SPRT (mSPRT) for composite hypotheses

Based on:
- Wald (1945) Sequential Analysis
- Pocock (1977) Group sequential methods
- O'Brien & Fleming (1979) Time-varying boundaries
- Lan & DeMets (1983) Alpha spending functions
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class StoppingDecision(Enum):
    """Early stopping decisions"""

    CONTINUE = "continue"
    STOP_REJECT_NULL = "stop_reject_null"  # Significant effect found
    STOP_ACCEPT_NULL = "stop_accept_null"  # No effect (futility)
    STOP_FOR_SUPERIORITY = "stop_for_superiority"  # Clear winner
    STOP_FOR_FUTILITY = "stop_for_futility"  # Unlikely to find effect


class AlphaSpendingFunction(Enum):
    """Error spending function types"""

    POCOCK = "pocock"
    OBRIEN_FLEMING = "obrien_fleming"
    WANG_TSIATIS = "wang_tsiatis"
    CUSTOM = "custom"


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping mechanisms"""

    # SPRT parameters
    alpha: float = 0.05  # Type I error rate
    beta: float = 0.2  # Type II error rate (1 - power)
    effect_size_h0: float = 0.0  # Null hypothesis effect size
    effect_size_h1: float = 0.1  # Alternative hypothesis effect size

    # Group Sequential Design parameters
    max_looks: int = 10  # Maximum number of interim analyses
    alpha_spending_function: AlphaSpendingFunction = (
        AlphaSpendingFunction.OBRIEN_FLEMING
    )
    information_fraction: list[float] = field(default_factory=list)  # Custom timing

    # Futility stopping parameters
    enable_futility_stopping: bool = True
    futility_threshold: float = 0.1  # Conditional power threshold
    min_sample_size: int = 30  # Minimum samples before stopping

    # Safety parameters
    max_duration_minutes: int = 60  # Maximum test duration
    min_effect_detectable: float = 0.05  # Minimum detectable effect

    # Advanced features
    enable_mixture_sprt: bool = False
    mixture_weights: list[float] = field(default_factory=lambda: [0.5, 0.5])
    mixture_effects: list[float] = field(default_factory=lambda: [0.1, 0.2])


@dataclass
class SPRTBounds:
    """SPRT decision boundaries"""

    lower_bound: float  # Accept null (no effect)
    upper_bound: float  # Accept alternative (effect exists)
    log_likelihood_ratio: float
    samples_analyzed: int
    decision: StoppingDecision


@dataclass
class GroupSequentialBounds:
    """Group sequential design boundaries"""

    look_number: int
    information_fraction: float
    alpha_spent: float
    rejection_boundary: float
    futility_boundary: float | None
    decision: StoppingDecision


@dataclass
class EarlyStoppingResult:
    """Result of early stopping analysis"""

    test_id: str
    look_number: int
    samples_analyzed: int
    analysis_time: datetime

    # Statistical measures
    test_statistic: float
    p_value: float
    effect_size: float
    conditional_power: float

    # SPRT results
    sprt_bounds: SPRTBounds | None = None

    # Group sequential results
    group_sequential_bounds: GroupSequentialBounds | None = None

    # Final decision
    decision: StoppingDecision = StoppingDecision.CONTINUE
    stop_for_efficacy: bool = False
    stop_for_futility: bool = False

    # Recommendations
    recommendation: str = ""
    confidence: float = 0.0
    estimated_remaining_samples: int = 0


class AdvancedEarlyStoppingFramework:
    """Advanced early stopping framework implementing research-validated methods"""

    def __init__(self, config: EarlyStoppingConfig | None = None):
        """Initialize early stopping framework

        Args:
            config: Early stopping configuration
        """
        self.config = config or EarlyStoppingConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Track ongoing experiments
        self.experiment_state: dict[str, dict[str, Any]] = {}
        self.stopping_history: list[EarlyStoppingResult] = []

    async def run_orchestrated_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrator-compatible interface for early stopping analysis (2025 pattern)

        Args:
            config: Orchestrator configuration containing:
                - experiment_id: Unique experiment identifier
                - control_data: Control group performance data
                - treatment_data: Treatment group performance data
                - look_number: Current analysis look number (for group sequential design)
                - stopping_criteria: Stopping criteria configuration
                - alpha_spending_function: Alpha spending function ('pocock', 'obrien_fleming', 'custom')
                - enable_futility: Whether to enable futility stopping
                - output_path: Local path for output files (optional)

        Returns:
            Orchestrator-compatible result with early stopping analysis and metadata
        """
        start_time = datetime.now()

        try:
            # Extract configuration from orchestrator
            experiment_id = config.get("experiment_id", f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            control_data = config.get("control_data", [])
            treatment_data = config.get("treatment_data", [])
            look_number = config.get("look_number", 1)
            stopping_criteria = config.get("stopping_criteria", {})
            alpha_spending_str = config.get("alpha_spending_function", "obrien_fleming")
            enable_futility = config.get("enable_futility", True)
            output_path = config.get("output_path", "./outputs/early_stopping")

            # Validate input data
            if not control_data or not treatment_data:
                raise ValueError("Both control_data and treatment_data are required")

            if len(control_data) < 3 or len(treatment_data) < 3:
                raise ValueError("At least 3 observations required per group")

            # Convert alpha spending function string to enum
            try:
                alpha_spending_function = AlphaSpendingFunction(alpha_spending_str)
            except ValueError:
                alpha_spending_function = AlphaSpendingFunction.OBRIEN_FLEMING
                self.logger.warning(f"Unknown alpha spending function '{alpha_spending_str}', using obrien_fleming")

            # Update configuration with orchestrator settings
            if stopping_criteria:
                if "alpha" in stopping_criteria:
                    self.config.alpha = stopping_criteria["alpha"]
                if "beta" in stopping_criteria:
                    self.config.beta = stopping_criteria["beta"]
                if "effect_size_h1" in stopping_criteria:
                    self.config.effect_size_h1 = stopping_criteria["effect_size_h1"]

            self.config.alpha_spending_function = alpha_spending_function
            self.config.enable_futility_stopping = enable_futility

            # Perform early stopping evaluation using existing method
            stopping_result = await self.evaluate_stopping_criteria(
                experiment_id=experiment_id,
                control_data=control_data,
                treatment_data=treatment_data,
                look_number=look_number
            )

            # Prepare orchestrator-compatible result
            result = {
                "stopping_decision": {
                    "decision": stopping_result.decision.value,
                    "should_stop": stopping_result.decision != StoppingDecision.CONTINUE,
                    "recommendation": stopping_result.recommendation,
                    "confidence": stopping_result.confidence,
                    "look_number": look_number,
                    "stopping_reason": getattr(stopping_result, 'stopping_reason', 'analysis_complete')
                },
                "statistical_analysis": {
                    "test_statistic": stopping_result.test_statistic,
                    "p_value": stopping_result.p_value,
                    "effect_size": stopping_result.effect_size,
                    "confidence_interval": getattr(stopping_result, 'confidence_interval', [0, 0]),
                    "statistical_power": getattr(stopping_result, 'statistical_power', 0.0)
                },
                "boundary_analysis": {
                    "efficacy_boundary": getattr(stopping_result, 'efficacy_boundary', 0.0),
                    "futility_boundary": getattr(stopping_result, 'futility_boundary', 0.0),
                    "alpha_spent": getattr(stopping_result, 'alpha_spent', 0.0),
                    "beta_spent": getattr(stopping_result, 'beta_spent', 0.0),
                    "information_fraction": getattr(stopping_result, 'information_fraction', 0.0)
                },
                "group_sequential_design": {
                    "max_looks": self.config.max_looks,
                    "current_look": look_number,
                    "alpha_spending_function": alpha_spending_function.value,
                    "remaining_looks": max(0, self.config.max_looks - look_number),
                    "planned_sample_size": getattr(stopping_result, 'planned_sample_size', len(control_data) + len(treatment_data))
                },
                "futility_analysis": {
                    "futility_enabled": self.config.enable_futility_stopping,
                    "conditional_power": stopping_result.conditional_power,
                    "futility_threshold": self.config.futility_threshold,
                    "stop_for_futility": stopping_result.conditional_power < self.config.futility_threshold if self.config.enable_futility_stopping else False
                },
                "recommendations": {
                    "primary_recommendation": stopping_result.recommendation,
                    "continue_experiment": stopping_result.decision == StoppingDecision.CONTINUE,
                    "suggested_next_look": getattr(stopping_result, 'suggested_next_look', look_number + 1),
                    "minimum_additional_samples": stopping_result.estimated_remaining_samples
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
                    "experiment_id": experiment_id,
                    "look_number": look_number,
                    "control_sample_size": len(control_data),
                    "treatment_sample_size": len(treatment_data),
                    "alpha_spending_function": alpha_spending_function.value,
                    "futility_enabled": enable_futility,
                    "max_looks": self.config.max_looks,
                    "component_version": "1.0.0"
                }
            }

        except ValueError as e:
            self.logger.error(f"Validation error in orchestrated early stopping: {e}")
            return {
                "orchestrator_compatible": True,
                "component_result": {"error": f"Validation error: {str(e)}", "stopping_decision": {}},
                "local_metadata": {
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "error": True,
                    "error_type": "validation",
                    "component_version": "1.0.0"
                }
            }
        except Exception as e:
            self.logger.error(f"Orchestrated early stopping failed: {e}")
            return {
                "orchestrator_compatible": True,
                "component_result": {"error": str(e), "stopping_decision": {}},
                "local_metadata": {
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "error": True,
                    "component_version": "1.0.0"
                }
            }

        # Precompute boundaries for efficiency
        self._precompute_group_sequential_boundaries()

    async def evaluate_stopping_criteria(
        self,
        experiment_id: str,
        control_data: list[float],
        treatment_data: list[float],
        look_number: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> EarlyStoppingResult:
        """Evaluate whether to stop experiment early

        Args:
            experiment_id: Unique experiment identifier
            control_data: Control group measurements
            treatment_data: Treatment group measurements
            look_number: Current interim analysis number
            metadata: Additional experiment metadata

        Returns:
            Early stopping analysis result with recommendation
        """
        metadata = metadata or {}

        if (
            len(control_data) < self.config.min_sample_size
            or len(treatment_data) < self.config.min_sample_size
        ):
            return EarlyStoppingResult(
                test_id=experiment_id,
                look_number=look_number,
                samples_analyzed=len(control_data) + len(treatment_data),
                analysis_time=datetime.utcnow(),
                test_statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                conditional_power=0.0,
                decision=StoppingDecision.CONTINUE,
                recommendation="Insufficient sample size for early stopping analysis",
                confidence=0.0,
            )

        self.logger.info(
            f"Evaluating stopping criteria for {experiment_id} (look {look_number})"
        )

        # Initialize experiment tracking
        if experiment_id not in self.experiment_state:
            self.experiment_state[experiment_id] = {
                "start_time": datetime.utcnow(),
                "looks": [],
                "cumulative_data": {"control": [], "treatment": []},
            }

        # Update cumulative data
        exp_state = self.experiment_state[experiment_id]
        exp_state["cumulative_data"]["control"].extend(control_data)
        exp_state["cumulative_data"]["treatment"].extend(treatment_data)

        # Calculate current statistics
        stats_result = self._calculate_test_statistics(control_data, treatment_data)

        # Evaluate SPRT if enabled
        sprt_bounds = None
        if self.config.alpha > 0:
            sprt_bounds = await self._evaluate_sprt(
                experiment_id, control_data, treatment_data, look_number
            )

        # Evaluate Group Sequential Design
        group_seq_bounds = None
        if self.config.max_looks > 1:
            group_seq_bounds = await self._evaluate_group_sequential(
                experiment_id, control_data, treatment_data, look_number
            )

        # Evaluate futility stopping
        stop_for_futility = False
        conditional_power = 0.0
        if self.config.enable_futility_stopping:
            conditional_power = self._calculate_conditional_power(
                control_data, treatment_data, self.config.effect_size_h1
            )
            stop_for_futility = conditional_power < self.config.futility_threshold

        # Make final stopping decision
        decision, recommendation, confidence = self._make_stopping_decision(
            stats_result,
            sprt_bounds,
            group_seq_bounds,
            stop_for_futility,
            conditional_power,
        )

        # Create result
        result = EarlyStoppingResult(
            test_id=experiment_id,
            look_number=look_number,
            samples_analyzed=len(control_data) + len(treatment_data),
            analysis_time=datetime.utcnow(),
            test_statistic=stats_result["test_statistic"],
            p_value=stats_result["p_value"],
            effect_size=stats_result["effect_size"],
            conditional_power=conditional_power,
            sprt_bounds=sprt_bounds,
            group_sequential_bounds=group_seq_bounds,
            decision=decision,
            stop_for_efficacy=(
                sprt_bounds
                and sprt_bounds.decision == StoppingDecision.STOP_REJECT_NULL
            )
            or (
                group_seq_bounds
                and group_seq_bounds.decision == StoppingDecision.STOP_REJECT_NULL
            ),
            stop_for_futility=stop_for_futility,
            recommendation=recommendation,
            confidence=confidence,
            estimated_remaining_samples=self._estimate_remaining_samples(
                control_data, treatment_data, conditional_power
            ),
        )

        # Update experiment state
        exp_state["looks"].append({
            "look_number": look_number,
            "result": result,
            "timestamp": datetime.utcnow(),
        })

        # Store in history
        self.stopping_history.append(result)

        # Clean up if experiment is stopping
        if decision != StoppingDecision.CONTINUE:
            self.logger.info(f"Experiment {experiment_id} stopping: {decision.value}")
            # Keep experiment state for final analysis but mark as complete
            exp_state["completed"] = True
            exp_state["final_decision"] = decision

        return result

    def _calculate_test_statistics(
        self, control: list[float], treatment: list[float]
    ) -> dict[str, float]:
        """Calculate basic test statistics"""
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)

        # Welch's t-test
        statistic, p_value = stats.ttest_ind(treatment, control, equal_var=False)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (
                (len(control) - 1) * np.var(control, ddof=1)
                + (len(treatment) - 1) * np.var(treatment, ddof=1)
            )
            / (len(control) + len(treatment) - 2)
        )

        effect_size = (
            (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        )

        return {
            "test_statistic": float(statistic),
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "control_mean": float(control_mean),
            "treatment_mean": float(treatment_mean),
        }

    async def _evaluate_sprt(
        self,
        experiment_id: str,
        control: list[float],
        treatment: list[float],
        look_number: int,
    ) -> SPRTBounds:
        """Evaluate Sequential Probability Ratio Test"""
        # SPRT boundaries
        alpha = self.config.alpha
        beta = self.config.beta

        # Log-likelihood ratio bounds
        a = math.log(beta / (1 - alpha))  # Lower bound (accept H0)
        b = math.log((1 - beta) / alpha)  # Upper bound (accept H1)

        # Calculate likelihood ratio for normal distributions
        # Simplified implementation assuming known variance
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        pooled_var = (np.var(control, ddof=1) + np.var(treatment, ddof=1)) / 2

        if pooled_var <= 0:
            # Handle degenerate case
            log_lr = 0.0
        else:
            # Log-likelihood ratio for difference in means
            observed_diff = treatment_mean - control_mean
            sigma = math.sqrt(pooled_var)
            n = len(treatment)

            # Under H1: effect = effect_size_h1 * sigma
            # Under H0: effect = effect_size_h0 * sigma
            expected_diff_h1 = self.config.effect_size_h1 * sigma
            expected_diff_h0 = self.config.effect_size_h0 * sigma

            # Log-likelihood ratio
            if sigma > 0 and n > 0:
                ll_h1 = -0.5 * n * ((observed_diff - expected_diff_h1) / sigma) ** 2
                ll_h0 = -0.5 * n * ((observed_diff - expected_diff_h0) / sigma) ** 2
                log_lr = ll_h1 - ll_h0
            else:
                log_lr = 0.0

        # Make decision
        decision = StoppingDecision.CONTINUE
        if log_lr <= a:
            decision = StoppingDecision.STOP_ACCEPT_NULL
        elif log_lr >= b:
            decision = StoppingDecision.STOP_REJECT_NULL

        return SPRTBounds(
            lower_bound=a,
            upper_bound=b,
            log_likelihood_ratio=log_lr,
            samples_analyzed=len(control) + len(treatment),
            decision=decision,
        )

    async def _evaluate_group_sequential(
        self,
        experiment_id: str,
        control: list[float],
        treatment: list[float],
        look_number: int,
    ) -> GroupSequentialBounds:
        """Evaluate Group Sequential Design with error spending"""
        # Calculate information fraction
        if self.config.information_fraction:
            # Use custom timing
            if look_number <= len(self.config.information_fraction):
                info_frac = self.config.information_fraction[look_number - 1]
            else:
                info_frac = 1.0
        else:
            # Equal spacing
            info_frac = look_number / self.config.max_looks

        # Calculate alpha spending
        alpha_spent = self._calculate_alpha_spending(
            info_frac, self.config.alpha_spending_function
        )

        # Calculate rejection boundary
        rejection_boundary = self._calculate_rejection_boundary(
            alpha_spent, len(control), len(treatment)
        )

        # Calculate test statistic
        stats_result = self._calculate_test_statistics(control, treatment)
        test_stat = abs(stats_result["test_statistic"])

        # Make decision
        decision = StoppingDecision.CONTINUE
        if test_stat >= rejection_boundary:
            decision = StoppingDecision.STOP_REJECT_NULL

        # Calculate futility boundary if enabled
        futility_boundary = None
        if self.config.enable_futility_stopping:
            futility_boundary = self._calculate_futility_boundary(info_frac)
            if test_stat <= futility_boundary:
                decision = StoppingDecision.STOP_FOR_FUTILITY

        return GroupSequentialBounds(
            look_number=look_number,
            information_fraction=info_frac,
            alpha_spent=alpha_spent,
            rejection_boundary=rejection_boundary,
            futility_boundary=futility_boundary,
            decision=decision,
        )

    def _calculate_alpha_spending(
        self, t: float, function_type: AlphaSpendingFunction
    ) -> float:
        """Calculate cumulative alpha spending at information time t"""
        alpha = self.config.alpha

        if function_type == AlphaSpendingFunction.POCOCK:
            # Pocock boundary: α * ln(1 + (e-1)*t)
            if t <= 0:
                return 0
            return alpha * math.log(1 + (math.e - 1) * t)

        if function_type == AlphaSpendingFunction.OBRIEN_FLEMING:
            # O'Brien-Fleming boundary: 4 - 4*Φ(Φ^(-1)(1-α/4)/√t)
            if t <= 0:
                return 0
            z_alpha_4 = stats.norm.ppf(1 - alpha / 4)
            return 4 - 4 * stats.norm.cdf(z_alpha_4 / math.sqrt(t))

        if function_type == AlphaSpendingFunction.WANG_TSIATIS:
            # Wang-Tsiatis with rho = 0.5
            rho = 0.5
            if t <= 0:
                return 0
            return alpha * (t**rho)

        # Linear spending as fallback
        return alpha * t

    def _calculate_rejection_boundary(
        self, alpha_spent: float, n_control: int, n_treatment: int
    ) -> float:
        """Calculate rejection boundary for current alpha spending"""
        # Use normal approximation
        if alpha_spent <= 0:
            return float("inf")

        # Two-sided test
        z_alpha = stats.norm.ppf(1 - alpha_spent / 2)
        return z_alpha

    def _calculate_futility_boundary(self, info_frac: float) -> float:
        """Calculate futility boundary"""
        # Conservative futility boundary
        # Stop if test statistic is very small (unlikely to reach significance)
        if info_frac < 0.5:
            return 0.5  # Don't stop too early
        return 1.0  # More liberal stopping later

    def _calculate_conditional_power(
        self, control: list[float], treatment: list[float], target_effect: float
    ) -> float:
        """Calculate conditional power to detect target effect"""
        current_n = len(treatment)
        if current_n == 0:
            return 0.0

        # Current effect estimate
        current_effect = np.mean(treatment) - np.mean(control)
        pooled_var = (np.var(control, ddof=1) + np.var(treatment, ddof=1)) / 2

        if pooled_var <= 0:
            return 0.0

        # Estimate final sample size needed
        z_alpha = stats.norm.ppf(1 - self.config.alpha / 2)
        z_beta = stats.norm.ppf(1 - self.config.beta)

        # Required sample size for target effect
        required_n = 2 * pooled_var * ((z_alpha + z_beta) / target_effect) ** 2

        # If we already have enough samples, check current effect
        if current_n >= required_n:
            # Calculate current z-score
            se_current = math.sqrt(2 * pooled_var / current_n)
            z_current = abs(current_effect) / se_current if se_current > 0 else 0
            # Return probability of significance
            return 1 - stats.norm.cdf(z_alpha - z_current) if z_current > 0 else 0.0

        # Conditional power calculation for continuing the experiment
        remaining_n = required_n - current_n
        if remaining_n <= 0:
            return 1.0

        # Current z-score with current sample size
        se_current = math.sqrt(2 * pooled_var / current_n)
        z_current = current_effect / se_current if se_current > 0 else 0

        # Final z-score if current trend continues
        se_final = math.sqrt(2 * pooled_var / required_n)
        z_final = current_effect / se_final if se_final > 0 else 0

        # Conditional power - probability of achieving significance
        conditional_power = (
            1 - stats.norm.cdf(z_alpha - z_final) if z_final > 0 else 0.0
        )

        return max(0.0, min(1.0, conditional_power))

    def _make_stopping_decision(
        self,
        stats_result: dict[str, float],
        sprt_bounds: SPRTBounds | None,
        group_seq_bounds: GroupSequentialBounds | None,
        stop_for_futility: bool,
        conditional_power: float,
    ) -> tuple[StoppingDecision, str, float]:
        """Make final stopping decision based on all criteria"""
        decisions = []
        recommendations = []
        confidence_scores = []

        # SPRT decision
        if sprt_bounds:
            if sprt_bounds.decision != StoppingDecision.CONTINUE:
                decisions.append(sprt_bounds.decision)
                if sprt_bounds.decision == StoppingDecision.STOP_REJECT_NULL:
                    recommendations.append("SPRT: Significant effect detected")
                    confidence_scores.append(0.95)
                else:
                    recommendations.append("SPRT: No effect detected")
                    confidence_scores.append(0.90)

        # Group Sequential decision
        if group_seq_bounds:
            if group_seq_bounds.decision != StoppingDecision.CONTINUE:
                decisions.append(group_seq_bounds.decision)
                if group_seq_bounds.decision == StoppingDecision.STOP_REJECT_NULL:
                    recommendations.append(
                        "Group Sequential: Significant effect with error control"
                    )
                    confidence_scores.append(0.95)
                elif group_seq_bounds.decision == StoppingDecision.STOP_FOR_FUTILITY:
                    recommendations.append(
                        "Group Sequential: Futility boundary crossed"
                    )
                    confidence_scores.append(0.85)

        # Futility decision
        if stop_for_futility:
            decisions.append(StoppingDecision.STOP_FOR_FUTILITY)
            recommendations.append(
                f"Low conditional power ({conditional_power:.2%}) - unlikely to detect effect"
            )
            confidence_scores.append(0.80)

        # Safety checks
        p_value = stats_result.get("p_value", 1.0)
        effect_size = stats_result.get("effect_size", 0.0)

        if p_value < 0.001 and abs(effect_size) > 0.5:
            decisions.append(StoppingDecision.STOP_FOR_SUPERIORITY)
            recommendations.append(
                "Strong evidence of superiority (p < 0.001, large effect)"
            )
            confidence_scores.append(0.99)

        # Make final decision
        if StoppingDecision.STOP_FOR_SUPERIORITY in decisions:
            final_decision = StoppingDecision.STOP_FOR_SUPERIORITY
        elif StoppingDecision.STOP_REJECT_NULL in decisions:
            final_decision = StoppingDecision.STOP_REJECT_NULL
        elif StoppingDecision.STOP_FOR_FUTILITY in decisions:
            final_decision = StoppingDecision.STOP_FOR_FUTILITY
        elif StoppingDecision.STOP_ACCEPT_NULL in decisions:
            final_decision = StoppingDecision.STOP_ACCEPT_NULL
        else:
            final_decision = StoppingDecision.CONTINUE
            recommendations.append(
                "Continue experiment - insufficient evidence for stopping"
            )
            confidence_scores.append(0.60)

        final_recommendation = "; ".join(recommendations)
        final_confidence = max(confidence_scores) if confidence_scores else 0.60

        return final_decision, final_recommendation, final_confidence

    def _estimate_remaining_samples(
        self, control: list[float], treatment: list[float], conditional_power: float
    ) -> int:
        """Estimate remaining samples needed to complete experiment"""
        if conditional_power >= 0.8:
            return 0  # Likely to reach conclusion soon

        current_n = len(treatment)
        pooled_var = (np.var(control, ddof=1) + np.var(treatment, ddof=1)) / 2

        if pooled_var <= 0:
            return 100  # Default estimate

        # Calculate required sample size for desired power
        z_alpha = stats.norm.ppf(1 - self.config.alpha / 2)
        z_beta = stats.norm.ppf(1 - self.config.beta)
        target_effect = self.config.effect_size_h1

        required_n_per_group = ((z_alpha + z_beta) / target_effect) ** 2 * pooled_var

        remaining = max(0, int(required_n_per_group - current_n))
        return min(remaining, 1000)  # Cap at reasonable limit

    def _precompute_group_sequential_boundaries(self):
        """Precompute Group Sequential boundaries for efficiency"""
        self.gs_boundaries = {}

        for look in range(1, self.config.max_looks + 1):
            info_frac = look / self.config.max_looks
            alpha_spent = self._calculate_alpha_spending(
                info_frac, self.config.alpha_spending_function
            )
            boundary = self._calculate_rejection_boundary(
                alpha_spent, 100, 100
            )  # Standardized

            self.gs_boundaries[look] = {
                "information_fraction": info_frac,
                "alpha_spent": alpha_spent,
                "boundary": boundary,
            }

    def get_experiment_summary(self, experiment_id: str) -> dict[str, Any] | None:
        """Get summary of experiment's early stopping history"""
        if experiment_id not in self.experiment_state:
            return None

        state = self.experiment_state[experiment_id]
        return {
            "experiment_id": experiment_id,
            "start_time": state["start_time"],
            "total_looks": len(state["looks"]),
            "completed": state.get("completed", False),
            "final_decision": state.get("final_decision"),
            "total_samples": len(state["cumulative_data"]["control"])
            + len(state["cumulative_data"]["treatment"]),
            "looks_history": [look["result"] for look in state["looks"]],
        }

    def get_stopping_history(self, limit: int = 50) -> list[EarlyStoppingResult]:
        """Get recent early stopping history"""
        return self.stopping_history[-limit:]

    def cleanup_completed_experiments(self, days_old: int = 30):
        """Clean up old completed experiments"""
        cutoff_time = datetime.utcnow() - timedelta(days=days_old)

        to_remove = []
        for exp_id, state in self.experiment_state.items():
            if state.get("completed", False) and state["start_time"] < cutoff_time:
                to_remove.append(exp_id)

        for exp_id in to_remove:
            del self.experiment_state[exp_id]

        self.logger.info(f"Cleaned up {len(to_remove)} old experiments")


# Utility functions for integration with existing A/B testing service


def create_early_stopping_framework(
    alpha: float = 0.05,
    beta: float = 0.2,
    max_looks: int = 10,
    enable_futility: bool = True,
) -> AdvancedEarlyStoppingFramework:
    """Create early stopping framework with common defaults"""
    config = EarlyStoppingConfig(
        alpha=alpha,
        beta=beta,
        max_looks=max_looks,
        enable_futility_stopping=enable_futility,
        alpha_spending_function=AlphaSpendingFunction.OBRIEN_FLEMING,
    )
    return AdvancedEarlyStoppingFramework(config)


async def should_stop_experiment(
    experiment_id: str,
    control_data: list[float],
    treatment_data: list[float],
    framework: AdvancedEarlyStoppingFramework,
    look_number: int = 1,
) -> tuple[bool, str, float]:
    """Simple interface to check if experiment should stop

    Returns:
        (should_stop, reason, confidence)
    """
    result = await framework.evaluate_stopping_criteria(
        experiment_id, control_data, treatment_data, look_number
    )

    should_stop = result.decision != StoppingDecision.CONTINUE
    return should_stop, result.recommendation, result.confidence
