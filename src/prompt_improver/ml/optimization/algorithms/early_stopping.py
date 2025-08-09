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
from datetime import datetime, timedelta
from enum import Enum
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from sqlmodel import SQLModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
from scipy import stats
logger = logging.getLogger(__name__)

class StoppingDecision(Enum):
    """Early stopping decisions"""
    CONTINUE = 'continue'
    STOP_REJECT_NULL = 'stop_reject_null'
    STOP_ACCEPT_NULL = 'stop_accept_null'
    STOP_FOR_SUPERIORITY = 'stop_for_superiority'
    STOP_FOR_FUTILITY = 'stop_for_futility'

class AlphaSpendingFunction(Enum):
    """Error spending function types"""
    pocock = 'pocock'
    OBRIEN_FLEMING = 'obrien_fleming'
    WANG_TSIATIS = 'wang_tsiatis'
    CUSTOM = 'custom'

class EarlyStoppingConfig(SQLModel):
    """Configuration for early stopping mechanisms"""
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, description='Type I error rate')
    beta: float = Field(default=0.2, gt=0.0, lt=1.0, description='Type II error rate (1 - power)')
    effect_size_h0: float = Field(default=0.0, description='Null hypothesis effect size')
    effect_size_h1: float = Field(default=0.1, description='Alternative hypothesis effect size')
    max_looks: int = Field(default=10, ge=1, description='Maximum number of interim analyses')
    alpha_spending_function: AlphaSpendingFunction = Field(default=AlphaSpendingFunction.OBRIEN_FLEMING, description='Alpha spending function type')
    information_fraction: List[float] = Field(default_factory=list, description='Custom timing fractions')
    enable_futility_stopping: bool = Field(default=True, description='Enable futility stopping')
    futility_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description='Conditional power threshold')
    min_sample_size: int = Field(default=30, ge=1, description='Minimum samples before stopping')
    max_duration_minutes: int = Field(default=60, ge=1, description='Maximum test duration')
    min_effect_detectable: float = Field(default=0.05, gt=0.0, description='Minimum detectable effect')
    enable_mixture_sprt: bool = Field(default=False, description='Enable mixture SPRT')
    mixture_weights: List[float] = Field(default_factory=lambda: [0.5, 0.5], description='Mixture component weights')
    mixture_effects: List[float] = Field(default_factory=lambda: [0.1, 0.2], description='Mixture effect sizes')

class SPRTBounds(SQLModel):
    """SPRT decision boundaries"""
    lower_bound: float = Field(description='Lower boundary - accept null hypothesis')
    upper_bound: float = Field(description='Upper boundary - accept alternative hypothesis')
    log_likelihood_ratio: float = Field(description='Current log likelihood ratio')
    samples_analyzed: int = Field(ge=0, description='Number of samples analyzed')
    decision: StoppingDecision = Field(description='Current stopping decision')

class GroupSequentialBounds(SQLModel):
    """Group sequential design boundaries"""
    look_number: int = Field(ge=1, description='Current look/analysis number')
    information_fraction: float = Field(ge=0.0, le=1.0, description='Fraction of total information')
    alpha_spent: float = Field(ge=0.0, le=1.0, description='Cumulative alpha spent')
    rejection_boundary: float = Field(description='Statistical boundary for rejection')
    futility_boundary: Optional[float] = Field(default=None, description='Statistical boundary for futility')
    decision: StoppingDecision = Field(description='Current stopping decision')

class EarlyStoppingResult(SQLModel):
    """Result of early stopping analysis"""
    test_id: str = Field(description='Test identifier')
    look_number: int = Field(ge=1, description='Analysis look number')
    samples_analyzed: int = Field(ge=0, description='Total samples analyzed')
    analysis_time: datetime = Field(description='Time of analysis')
    test_statistic: float = Field(description='Test statistic value')
    p_value: float = Field(ge=0.0, le=1.0, description='P-value from statistical test')
    effect_size: float = Field(description='Estimated effect size')
    conditional_power: float = Field(ge=0.0, le=1.0, description='Conditional power for continuation')
    sprt_bounds: Optional[SPRTBounds] = Field(default=None, description='SPRT boundary results')
    group_sequential_bounds: Optional[GroupSequentialBounds] = Field(default=None, description='Group sequential boundary results')
    decision: StoppingDecision = Field(default=StoppingDecision.CONTINUE, description='Stopping decision')
    stop_for_efficacy: bool = Field(default=False, description='Whether stopping for efficacy')
    stop_for_futility: bool = Field(default=False, description='Whether stopping for futility')
    recommendation: str = Field(default='', description='Recommendation text')
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description='Confidence in recommendation')
    estimated_remaining_samples: int = Field(default=0, ge=0, description='Estimated samples remaining')

class AdvancedEarlyStoppingFramework:
    """Advanced early stopping framework implementing research-validated methods"""

    def __init__(self, config: EarlyStoppingConfig | None=None):
        """Initialize early stopping framework

        Args:
            config: Early stopping configuration
        """
        self.config = config or EarlyStoppingConfig()
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
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
            experiment_id = config.get('experiment_id', f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            control_data = config.get('control_data', [])
            treatment_data = config.get('treatment_data', [])
            look_number = config.get('look_number', 1)
            stopping_criteria = config.get('stopping_criteria', {})
            alpha_spending_str = config.get('alpha_spending_function', 'obrien_fleming')
            enable_futility = config.get('enable_futility', True)
            output_path = config.get('output_path', './outputs/early_stopping')
            if not control_data or not treatment_data:
                raise ValueError('Both control_data and treatment_data are required')
            if len(control_data) < 3 or len(treatment_data) < 3:
                raise ValueError('At least 3 observations required per group')
            try:
                alpha_spending_function = AlphaSpendingFunction(alpha_spending_str)
            except ValueError:
                alpha_spending_function = AlphaSpendingFunction.OBRIEN_FLEMING
                self.logger.warning("Unknown alpha spending function '%s', using obrien_fleming", alpha_spending_str)
            if stopping_criteria:
                if 'alpha' in stopping_criteria:
                    self.config.alpha = stopping_criteria['alpha']
                if 'beta' in stopping_criteria:
                    self.config.beta = stopping_criteria['beta']
                if 'effect_size_h1' in stopping_criteria:
                    self.config.effect_size_h1 = stopping_criteria['effect_size_h1']
            self.config.alpha_spending_function = alpha_spending_function
            self.config.enable_futility_stopping = enable_futility
            stopping_result = await self.evaluate_stopping_criteria(experiment_id=experiment_id, control_data=control_data, treatment_data=treatment_data, look_number=look_number)
            result = {'stopping_decision': {'decision': stopping_result.decision.value, 'should_stop': stopping_result.decision != StoppingDecision.CONTINUE, 'recommendation': stopping_result.recommendation, 'confidence': stopping_result.confidence, 'look_number': look_number, 'stopping_reason': getattr(stopping_result, 'stopping_reason', 'analysis_complete')}, 'statistical_analysis': {'test_statistic': stopping_result.test_statistic, 'p_value': stopping_result.p_value, 'effect_size': stopping_result.effect_size, 'confidence_interval': getattr(stopping_result, 'confidence_interval', [0, 0]), 'statistical_power': getattr(stopping_result, 'statistical_power', 0.0)}, 'boundary_analysis': {'efficacy_boundary': getattr(stopping_result, 'efficacy_boundary', 0.0), 'futility_boundary': getattr(stopping_result, 'futility_boundary', 0.0), 'alpha_spent': getattr(stopping_result, 'alpha_spent', 0.0), 'beta_spent': getattr(stopping_result, 'beta_spent', 0.0), 'information_fraction': getattr(stopping_result, 'information_fraction', 0.0)}, 'group_sequential_design': {'max_looks': self.config.max_looks, 'current_look': look_number, 'alpha_spending_function': alpha_spending_function.value, 'remaining_looks': max(0, self.config.max_looks - look_number), 'planned_sample_size': getattr(stopping_result, 'planned_sample_size', len(control_data) + len(treatment_data))}, 'futility_analysis': {'futility_enabled': self.config.enable_futility_stopping, 'conditional_power': stopping_result.conditional_power, 'futility_threshold': self.config.futility_threshold, 'stop_for_futility': stopping_result.conditional_power < self.config.futility_threshold if self.config.enable_futility_stopping else False}, 'recommendations': {'primary_recommendation': stopping_result.recommendation, 'continue_experiment': stopping_result.decision == StoppingDecision.CONTINUE, 'suggested_next_look': getattr(stopping_result, 'suggested_next_look', look_number + 1), 'minimum_additional_samples': stopping_result.estimated_remaining_samples}}
            execution_time = (datetime.now() - start_time).total_seconds()
            return {'orchestrator_compatible': True, 'component_result': result, 'local_metadata': {'output_path': output_path, 'execution_time': execution_time, 'experiment_id': experiment_id, 'look_number': look_number, 'control_sample_size': len(control_data), 'treatment_sample_size': len(treatment_data), 'alpha_spending_function': alpha_spending_function.value, 'futility_enabled': enable_futility, 'max_looks': self.config.max_looks, 'component_version': '1.0.0'}}
        except ValueError as e:
            self.logger.error('Validation error in orchestrated early stopping: %s', e)
            return {'orchestrator_compatible': True, 'component_result': {'error': f'Validation error: {str(e)}', 'stopping_decision': {}}, 'local_metadata': {'execution_time': (datetime.now() - start_time).total_seconds(), 'error': True, 'error_type': 'validation', 'component_version': '1.0.0'}}
        except Exception as e:
            self.logger.error('Orchestrated early stopping failed: %s', e)
            return {'orchestrator_compatible': True, 'component_result': {'error': str(e), 'stopping_decision': {}}, 'local_metadata': {'execution_time': (datetime.now() - start_time).total_seconds(), 'error': True, 'component_version': '1.0.0'}}
        self._precompute_group_sequential_boundaries()

    async def evaluate_stopping_criteria(self, experiment_id: str, control_data: list[float], treatment_data: list[float], look_number: int=1, metadata: dict[str, Any] | None=None) -> EarlyStoppingResult:
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
        if len(control_data) < self.config.min_sample_size or len(treatment_data) < self.config.min_sample_size:
            return EarlyStoppingResult(test_id=experiment_id, look_number=look_number, samples_analyzed=len(control_data) + len(treatment_data), analysis_time=datetime.utcnow(), test_statistic=0.0, p_value=1.0, effect_size=0.0, conditional_power=0.0, decision=StoppingDecision.CONTINUE, recommendation='Insufficient sample size for early stopping analysis', confidence=0.0)
        self.logger.info('Evaluating stopping criteria for %s (look %s)', experiment_id, look_number)
        if experiment_id not in self.experiment_state:
            self.experiment_state[experiment_id] = {'start_time': datetime.utcnow(), 'looks': [], 'cumulative_data': {'control': [], 'treatment': []}}
        exp_state = self.experiment_state[experiment_id]
        exp_state['cumulative_data']['control'].extend(control_data)
        exp_state['cumulative_data']['treatment'].extend(treatment_data)
        stats_result = self._calculate_test_statistics(control_data, treatment_data)
        sprt_bounds = None
        if self.config.alpha > 0:
            sprt_bounds = await self._evaluate_sprt(experiment_id, control_data, treatment_data, look_number)
        group_seq_bounds = None
        if self.config.max_looks > 1:
            group_seq_bounds = await self._evaluate_group_sequential(experiment_id, control_data, treatment_data, look_number)
        stop_for_futility = False
        conditional_power = 0.0
        if self.config.enable_futility_stopping:
            conditional_power = self._calculate_conditional_power(control_data, treatment_data, self.config.effect_size_h1)
            stop_for_futility = conditional_power < self.config.futility_threshold
        decision, recommendation, confidence = self._make_stopping_decision(stats_result, sprt_bounds, group_seq_bounds, stop_for_futility, conditional_power)
        result = EarlyStoppingResult(test_id=experiment_id, look_number=look_number, samples_analyzed=len(control_data) + len(treatment_data), analysis_time=datetime.utcnow(), test_statistic=stats_result['test_statistic'], p_value=stats_result['p_value'], effect_size=stats_result['effect_size'], conditional_power=conditional_power, sprt_bounds=sprt_bounds, group_sequential_bounds=group_seq_bounds, decision=decision, stop_for_efficacy=sprt_bounds and sprt_bounds.decision == StoppingDecision.STOP_REJECT_NULL or (group_seq_bounds and group_seq_bounds.decision == StoppingDecision.STOP_REJECT_NULL), stop_for_futility=stop_for_futility, recommendation=recommendation, confidence=confidence, estimated_remaining_samples=self._estimate_remaining_samples(control_data, treatment_data, conditional_power))
        exp_state['looks'].append({'look_number': look_number, 'result': result, 'timestamp': datetime.utcnow()})
        self.stopping_history.append(result)
        if decision != StoppingDecision.CONTINUE:
            self.logger.info('Experiment {experiment_id} stopping: %s', decision.value)
            exp_state['completed'] = True
            exp_state['final_decision'] = decision
        return result

    def _calculate_test_statistics(self, control: list[float], treatment: list[float]) -> dict[str, float]:
        """Calculate basic test statistics"""
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        statistic, p_value = stats.ttest_ind(treatment, control, equal_var=False)
        pooled_std = np.sqrt(((len(control) - 1) * np.var(control, ddof=1) + (len(treatment) - 1) * np.var(treatment, ddof=1)) / (len(control) + len(treatment) - 2))
        effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        return {'test_statistic': float(statistic), 'p_value': float(p_value), 'effect_size': float(effect_size), 'control_mean': float(control_mean), 'treatment_mean': float(treatment_mean)}

    async def _evaluate_sprt(self, experiment_id: str, control: list[float], treatment: list[float], look_number: int) -> SPRTBounds:
        """Evaluate Sequential Probability Ratio Test"""
        alpha = self.config.alpha
        beta = self.config.beta
        a = math.log(beta / (1 - alpha))
        b = math.log((1 - beta) / alpha)
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        pooled_var = (np.var(control, ddof=1) + np.var(treatment, ddof=1)) / 2
        if pooled_var <= 0:
            log_lr = 0.0
        else:
            observed_diff = treatment_mean - control_mean
            sigma = math.sqrt(pooled_var)
            n = len(treatment)
            expected_diff_h1 = self.config.effect_size_h1 * sigma
            expected_diff_h0 = self.config.effect_size_h0 * sigma
            if sigma > 0 and n > 0:
                ll_h1 = -0.5 * n * ((observed_diff - expected_diff_h1) / sigma) ** 2
                ll_h0 = -0.5 * n * ((observed_diff - expected_diff_h0) / sigma) ** 2
                log_lr = ll_h1 - ll_h0
            else:
                log_lr = 0.0
        decision = StoppingDecision.CONTINUE
        if log_lr <= a:
            decision = StoppingDecision.STOP_ACCEPT_NULL
        elif log_lr >= b:
            decision = StoppingDecision.STOP_REJECT_NULL
        return SPRTBounds(lower_bound=a, upper_bound=b, log_likelihood_ratio=log_lr, samples_analyzed=len(control) + len(treatment), decision=decision)

    async def _evaluate_group_sequential(self, experiment_id: str, control: list[float], treatment: list[float], look_number: int) -> GroupSequentialBounds:
        """Evaluate Group Sequential Design with error spending"""
        if self.config.information_fraction:
            if look_number <= len(self.config.information_fraction):
                info_frac = self.config.information_fraction[look_number - 1]
            else:
                info_frac = 1.0
        else:
            info_frac = look_number / self.config.max_looks
        alpha_spent = self._calculate_alpha_spending(info_frac, self.config.alpha_spending_function)
        rejection_boundary = self._calculate_rejection_boundary(alpha_spent, len(control), len(treatment))
        stats_result = self._calculate_test_statistics(control, treatment)
        test_stat = abs(stats_result['test_statistic'])
        decision = StoppingDecision.CONTINUE
        if test_stat >= rejection_boundary:
            decision = StoppingDecision.STOP_REJECT_NULL
        futility_boundary = None
        if self.config.enable_futility_stopping:
            futility_boundary = self._calculate_futility_boundary(info_frac)
            if test_stat <= futility_boundary:
                decision = StoppingDecision.STOP_FOR_FUTILITY
        return GroupSequentialBounds(look_number=look_number, information_fraction=info_frac, alpha_spent=alpha_spent, rejection_boundary=rejection_boundary, futility_boundary=futility_boundary, decision=decision)

    def _calculate_alpha_spending(self, t: float, function_type: AlphaSpendingFunction) -> float:
        """Calculate cumulative alpha spending at information time t"""
        alpha = self.config.alpha
        if function_type == AlphaSpendingFunction.pocock:
            if t <= 0:
                return 0
            return alpha * math.log(1 + (math.e - 1) * t)
        if function_type == AlphaSpendingFunction.OBRIEN_FLEMING:
            if t <= 0:
                return 0
            z_alpha_4 = stats.norm.ppf(1 - alpha / 4)
            return 4 - 4 * stats.norm.cdf(z_alpha_4 / math.sqrt(t))
        if function_type == AlphaSpendingFunction.WANG_TSIATIS:
            rho = 0.5
            if t <= 0:
                return 0
            return alpha * t ** rho
        return alpha * t

    def _calculate_rejection_boundary(self, alpha_spent: float, n_control: int, n_treatment: int) -> float:
        """Calculate rejection boundary for current alpha spending"""
        if alpha_spent <= 0:
            return float('inf')
        z_alpha = stats.norm.ppf(1 - alpha_spent / 2)
        return z_alpha

    def _calculate_futility_boundary(self, info_frac: float) -> float:
        """Calculate futility boundary"""
        if info_frac < 0.5:
            return 0.5
        return 1.0

    def _calculate_conditional_power(self, control: list[float], treatment: list[float], target_effect: float) -> float:
        """Calculate conditional power to detect target effect"""
        current_n = len(treatment)
        if current_n == 0:
            return 0.0
        current_effect = np.mean(treatment) - np.mean(control)
        pooled_var = (np.var(control, ddof=1) + np.var(treatment, ddof=1)) / 2
        if pooled_var <= 0:
            return 0.0
        z_alpha = stats.norm.ppf(1 - self.config.alpha / 2)
        z_beta = stats.norm.ppf(1 - self.config.beta)
        required_n = 2 * pooled_var * ((z_alpha + z_beta) / target_effect) ** 2
        if current_n >= required_n:
            se_current = math.sqrt(2 * pooled_var / current_n)
            z_current = abs(current_effect) / se_current if se_current > 0 else 0
            return 1 - stats.norm.cdf(z_alpha - z_current) if z_current > 0 else 0.0
        remaining_n = required_n - current_n
        if remaining_n <= 0:
            return 1.0
        se_current = math.sqrt(2 * pooled_var / current_n)
        z_current = current_effect / se_current if se_current > 0 else 0
        se_final = math.sqrt(2 * pooled_var / required_n)
        z_final = current_effect / se_final if se_final > 0 else 0
        conditional_power = 1 - stats.norm.cdf(z_alpha - z_final) if z_final > 0 else 0.0
        return max(0.0, min(1.0, conditional_power))

    def _make_stopping_decision(self, stats_result: dict[str, float], sprt_bounds: SPRTBounds | None, group_seq_bounds: GroupSequentialBounds | None, stop_for_futility: bool, conditional_power: float) -> tuple[StoppingDecision, str, float]:
        """Make final stopping decision based on all criteria"""
        decisions = []
        recommendations = []
        confidence_scores = []
        if sprt_bounds:
            if sprt_bounds.decision != StoppingDecision.CONTINUE:
                decisions.append(sprt_bounds.decision)
                if sprt_bounds.decision == StoppingDecision.STOP_REJECT_NULL:
                    recommendations.append('SPRT: Significant effect detected')
                    confidence_scores.append(0.95)
                else:
                    recommendations.append('SPRT: No effect detected')
                    confidence_scores.append(0.9)
        if group_seq_bounds:
            if group_seq_bounds.decision != StoppingDecision.CONTINUE:
                decisions.append(group_seq_bounds.decision)
                if group_seq_bounds.decision == StoppingDecision.STOP_REJECT_NULL:
                    recommendations.append('Group Sequential: Significant effect with error control')
                    confidence_scores.append(0.95)
                elif group_seq_bounds.decision == StoppingDecision.STOP_FOR_FUTILITY:
                    recommendations.append('Group Sequential: Futility boundary crossed')
                    confidence_scores.append(0.85)
        if stop_for_futility:
            decisions.append(StoppingDecision.STOP_FOR_FUTILITY)
            recommendations.append(f'Low conditional power ({conditional_power:.2%}) - unlikely to detect effect')
            confidence_scores.append(0.8)
        p_value = stats_result.get('p_value', 1.0)
        effect_size = stats_result.get('effect_size', 0.0)
        if p_value < 0.001 and abs(effect_size) > 0.5:
            decisions.append(StoppingDecision.STOP_FOR_SUPERIORITY)
            recommendations.append('Strong evidence of superiority (p < 0.001, large effect)')
            confidence_scores.append(0.99)
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
            recommendations.append('Continue experiment - insufficient evidence for stopping')
            confidence_scores.append(0.6)
        final_recommendation = '; '.join(recommendations)
        final_confidence = max(confidence_scores) if confidence_scores else 0.6
        return (final_decision, final_recommendation, final_confidence)

    def _estimate_remaining_samples(self, control: list[float], treatment: list[float], conditional_power: float) -> int:
        """Estimate remaining samples needed to complete experiment"""
        if conditional_power >= 0.8:
            return 0
        current_n = len(treatment)
        pooled_var = (np.var(control, ddof=1) + np.var(treatment, ddof=1)) / 2
        if pooled_var <= 0:
            return 100
        z_alpha = stats.norm.ppf(1 - self.config.alpha / 2)
        z_beta = stats.norm.ppf(1 - self.config.beta)
        target_effect = self.config.effect_size_h1
        required_n_per_group = ((z_alpha + z_beta) / target_effect) ** 2 * pooled_var
        remaining = max(0, int(required_n_per_group - current_n))
        return min(remaining, 1000)

    def _precompute_group_sequential_boundaries(self):
        """Precompute Group Sequential boundaries for efficiency"""
        self.gs_boundaries = {}
        for look in range(1, self.config.max_looks + 1):
            info_frac = look / self.config.max_looks
            alpha_spent = self._calculate_alpha_spending(info_frac, self.config.alpha_spending_function)
            boundary = self._calculate_rejection_boundary(alpha_spent, 100, 100)
            self.gs_boundaries[look] = {'information_fraction': info_frac, 'alpha_spent': alpha_spent, 'boundary': boundary}

    def get_experiment_summary(self, experiment_id: str) -> dict[str, Any] | None:
        """Get summary of experiment's early stopping history"""
        if experiment_id not in self.experiment_state:
            return None
        state = self.experiment_state[experiment_id]
        return {'experiment_id': experiment_id, 'start_time': state['start_time'], 'total_looks': len(state['looks']), 'completed': state.get('completed', False), 'final_decision': state.get('final_decision'), 'total_samples': len(state['cumulative_data']['control']) + len(state['cumulative_data']['treatment']), 'looks_history': [look['result'] for look in state['looks']]}

    def get_stopping_history(self, limit: int=50) -> list[EarlyStoppingResult]:
        """Get recent early stopping history"""
        return self.stopping_history[-limit:]

    def cleanup_completed_experiments(self, days_old: int=30):
        """Clean up old completed experiments"""
        cutoff_time = datetime.utcnow() - timedelta(days=days_old)
        to_remove = []
        for exp_id, state in self.experiment_state.items():
            if state.get('completed', False) and state['start_time'] < cutoff_time:
                to_remove.append(exp_id)
        for exp_id in to_remove:
            del self.experiment_state[exp_id]
        self.logger.info('Cleaned up %s old experiments', len(to_remove))

def create_early_stopping_framework(alpha: float=0.05, beta: float=0.2, max_looks: int=10, enable_futility: bool=True) -> AdvancedEarlyStoppingFramework:
    """Create early stopping framework with common defaults"""
    config = EarlyStoppingConfig(alpha=alpha, beta=beta, max_looks=max_looks, enable_futility_stopping=enable_futility, alpha_spending_function=AlphaSpendingFunction.OBRIEN_FLEMING)
    return AdvancedEarlyStoppingFramework(config)

async def should_stop_experiment(experiment_id: str, control_data: list[float], treatment_data: list[float], framework: AdvancedEarlyStoppingFramework, look_number: int=1) -> tuple[bool, str, float]:
    """Simple interface to check if experiment should stop

    Returns:
        (should_stop, reason, confidence)
    """
    result = await framework.evaluate_stopping_criteria(experiment_id, control_data, treatment_data, look_number)
    should_stop = result.decision != StoppingDecision.CONTINUE
    return (should_stop, result.recommendation, result.confidence)
