"""Experiment Orchestrator for Advanced A/B Testing
Coordinates and manages complex multi-variate experiments with causal inference
"""
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
import pandas as pd
from prompt_improver.utils.datetime_utils import aware_utc_now
from prompt_improver.common.datetime_utils import format_compact_timestamp, format_display_date, format_date_only
from ...core.services.analytics_factory import get_analytics_router
from ...database.models import ABExperiment, RulePerformance
from ...performance.monitoring.health.background_manager import EnhancedBackgroundTaskManager, TaskPriority, get_background_task_manager
from .advanced_statistical_validator import AdvancedStatisticalValidator, AdvancedValidationResult
from .causal_inference_analyzer import CausalInferenceAnalyzer, CausalInferenceResult, CausalMethod
from .pattern_significance_analyzer import PatternSignificanceAnalyzer, PatternSignificanceReport
logger = logging.getLogger(__name__)

class BayesianExperimentConfig:
    """Configuration for Bayesian experiment analysis."""

    def __init__(self, enable_real_time_analysis: bool=True, confidence_threshold: float=0.95, harm_threshold: float=0.05, minimum_sample_size: int=100, prior_alpha: float=1.0, prior_beta: float=1.0):
        self.enable_real_time_analysis = enable_real_time_analysis
        self.confidence_threshold = confidence_threshold
        self.harm_threshold = harm_threshold
        self.minimum_sample_size = minimum_sample_size
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

class ExperimentType(Enum):
    """Types of experiments that can be orchestrated"""
    SIMPLE_AB = 'simple_ab'
    multivariate = 'multivariate'
    factorial = 'factorial'
    SEQUENTIAL = 'sequential'
    bandits = 'bandits'
    CAUSAL_INFERENCE = 'causal_inference'

class ExperimentStatus(Enum):
    """Experiment lifecycle states"""
    planned = 'planned'
    active = 'active'
    PAUSED = 'paused'
    COMPLETED = 'completed'
    stopped = 'stopped'
    analyzing = 'analyzing'
    ARCHIVED = 'archived'

class StoppingRule(Enum):
    """Early stopping criteria"""
    STATISTICAL_SIGNIFICANCE = 'statistical_significance'
    PRACTICAL_SIGNIFICANCE = 'practical_significance'
    SEQUENTIAL_TESTING = 'sequential_testing'
    BAYESIAN_DECISION = 'bayesian_decision'
    SAMPLE_SIZE_REACHED = 'sample_size_reached'
    TIME_LIMIT = 'time_limit'

@dataclass
class ExperimentArm:
    """Represents a single experiment arm/variant"""
    arm_id: str
    arm_name: str
    description: str
    rules: dict[str, Any]
    allocation_weight: float = 1.0
    current_allocation: float = 0.0
    sample_size: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentConfiguration:
    """Configuration for experiment setup"""
    experiment_id: str
    experiment_name: str
    experiment_type: ExperimentType
    description: str
    arms: list[ExperimentArm]
    target_population: dict[str, Any] = field(default_factory=dict)
    traffic_allocation: float = 1.0
    minimum_sample_size: int = 100
    maximum_sample_size: int = 10000
    statistical_power: float = 0.8
    effect_size_threshold: float = 0.1
    significance_level: float = 0.05
    stopping_rules: list[StoppingRule] = field(default_factory=list)
    max_duration_days: int = 30
    early_stopping_enabled: bool = True
    primary_metric: str = 'improvement_score'
    secondary_metrics: list[str] = field(default_factory=list)
    causal_analysis_enabled: bool = True
    pattern_analysis_enabled: bool = True
    sequential_testing_enabled: bool = False
    adaptive_allocation_enabled: bool = False
    multi_armed_bandit_config: dict[str, Any] | None = None
    guardrail_metrics: list[str] = field(default_factory=list)
    quality_thresholds: dict[str, float] = field(default_factory=dict)

@dataclass
class ExperimentResult:
    """Comprehensive experiment analysis result"""
    experiment_id: str
    analysis_id: str
    timestamp: datetime
    experiment_status: ExperimentStatus
    statistical_validation: AdvancedValidationResult
    pattern_significance: PatternSignificanceReport | None = None
    causal_inference: CausalInferenceResult | None = None
    arm_performance: dict[str, dict[str, float]] = field(default_factory=dict)
    relative_performance: dict[str, float] = field(default_factory=dict)
    stopping_recommendation: str = ''
    business_decision: str = ''
    confidence_level: float = 0.0
    risk_assessment: dict[str, Any] = field(default_factory=dict)
    data_quality_score: float = 0.0
    analysis_quality_score: float = 0.0
    overall_experiment_quality: float = 0.0
    actionable_insights: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    lessons_learned: list[str] = field(default_factory=list)
    analysis_duration_seconds: float = 0.0
    sample_sizes: dict[str, int] = field(default_factory=dict)
    experiment_duration_days: float = 0.0

class ExperimentOrchestrator:
    """Advanced experiment orchestrator implementing 2025 best practices"""

    def __init__(self, db_session: AsyncSession, statistical_validator: AdvancedStatisticalValidator | None=None, pattern_analyzer: PatternSignificanceAnalyzer | None=None, causal_analyzer: CausalInferenceAnalyzer | None=None, real_time_service: Any | None=None, task_manager: EnhancedBackgroundTaskManager | None=None):
        """Initialize experiment orchestrator

        Args:
            db_session: Database session for experiment management
            statistical_validator: Advanced statistical validator
            pattern_analyzer: Pattern significance analyzer
            causal_analyzer: Causal inference analyzer
            real_time_service: Real-time analytics service (deprecated - use analytics_router instead)
            task_manager: Enhanced background task manager for centralized task management
        """
        self.db_session = db_session
        self.statistical_validator = statistical_validator or AdvancedStatisticalValidator()
        self.pattern_analyzer = pattern_analyzer or PatternSignificanceAnalyzer()
        self.causal_analyzer = causal_analyzer or CausalInferenceAnalyzer()
        self.real_time_service = real_time_service
        self.task_manager = task_manager or get_background_task_manager()
        self.active_experiments: dict[str, ExperimentConfiguration] = {}
        self.experiment_tasks: dict[str, str] = {}
        self.bayesian_analyzer = None
        self.bayesian_config = None
        self.bayesian_monitoring_enabled = False
        self.task_monitoring_enabled = True
        self.experiment_task_timeout = 24 * 3600
        logger.info('Experiment Orchestrator initialized with task manager: %s', type(self.task_manager).__name__)

    async def setup_experiment(self, config: ExperimentConfiguration) -> dict[str, Any]:
        """Set up and validate a new experiment

        Args:
            config: Experiment configuration

        Returns:
            Setup result with validation and recommendations
        """
        try:
            logger.info('Setting up experiment: %s', config.experiment_id)
            validation_result = self._validate_experiment_config(config)
            if not validation_result['valid']:
                return {'success': False, 'errors': validation_result['errors'], 'experiment_id': config.experiment_id}
            sample_size_analysis = self._calculate_sample_size_requirements(config)
            experiment_record = await self._create_experiment_record(config)
            if self.real_time_service:
                await self.real_time_service.start_experiment_monitoring(experiment_id=config.experiment_id, update_interval=60)
            self.active_experiments[config.experiment_id] = config
            monitoring_task_id = f'experiment_monitor_{config.experiment_id}_{int(datetime.now().timestamp())}'
            task_id = await self.task_manager.submit_enhanced_task(task_id=monitoring_task_id, coroutine=self._monitor_experiment(config.experiment_id), priority=TaskPriority.NORMAL, timeout=config.max_duration_days * 24 * 3600, tags={'type': 'experiment_monitoring', 'experiment_id': config.experiment_id, 'experiment_name': config.experiment_name})
            self.experiment_tasks[config.experiment_id] = task_id
            logger.info('Experiment setup completed: %s', config.experiment_id)
            return {'success': True, 'experiment_id': config.experiment_id, 'experiment_record_id': experiment_record.experiment_id, 'sample_size_analysis': sample_size_analysis, 'validation_warnings': validation_result.get('warnings', []), 'estimated_duration_days': sample_size_analysis.get('estimated_duration_days'), 'monitoring_enabled': self.real_time_service is not None}
        except Exception as e:
            logger.error('Error setting up experiment {config.experiment_id}: %s', e)
            return {'success': False, 'error': str(e), 'experiment_id': config.experiment_id}

    async def analyze_experiment(self, experiment_id: str, force_analysis: bool=False) -> ExperimentResult:
        """Perform comprehensive experiment analysis

        Args:
            experiment_id: Experiment identifier
            force_analysis: Force analysis even if stopping criteria not met

        Returns:
            Comprehensive experiment analysis result
        """
        analysis_start = aware_utc_now()
        analysis_id = f"analysis_{experiment_id}_{format_compact_timestamp(analysis_start)}"
        try:
            logger.info('Starting experiment analysis: %s', experiment_id)
            config = self.active_experiments.get(experiment_id)
            if not config:
                config = await self._load_experiment_config(experiment_id)
            if not config:
                raise ValueError(f'Experiment {experiment_id} not found')
            experiment_data = await self._collect_experiment_data(experiment_id, config)
            if not experiment_data['sufficient_data']:
                return self._create_insufficient_data_result(experiment_id, analysis_id, experiment_data)
            logger.info('Performing statistical validation for %s', experiment_id)
            statistical_result = await self._perform_statistical_analysis(experiment_data, config)
            pattern_result = None
            if config.pattern_analysis_enabled and len(experiment_data['arms']) > 1:
                logger.info('Performing pattern analysis for %s', experiment_id)
                pattern_result = await self._perform_pattern_analysis(experiment_data, config)
            causal_result = None
            if config.causal_analysis_enabled:
                logger.info('Performing causal analysis for %s', experiment_id)
                causal_result = await self._perform_causal_analysis(experiment_data, config)
            performance_metrics = self._calculate_performance_metrics(experiment_data, statistical_result)
            stopping_recommendation = self._generate_stopping_recommendation(config, statistical_result, causal_result, performance_metrics)
            business_decision = self._generate_business_decision(config, statistical_result, causal_result, performance_metrics)
            quality_scores = self._calculate_quality_scores(experiment_data, statistical_result, pattern_result, causal_result)
            insights = self._generate_actionable_insights(config, statistical_result, pattern_result, causal_result, performance_metrics)
            analysis_duration = (aware_utc_now() - analysis_start).total_seconds()
            result = ExperimentResult(experiment_id=experiment_id, analysis_id=analysis_id, timestamp=aware_utc_now(), experiment_status=self._determine_experiment_status(config, stopping_recommendation), statistical_validation=statistical_result, pattern_significance=pattern_result, causal_inference=causal_result, arm_performance=performance_metrics['arm_performance'], relative_performance=performance_metrics['relative_performance'], stopping_recommendation=stopping_recommendation['recommendation'], business_decision=business_decision['decision'], confidence_level=stopping_recommendation['confidence'], risk_assessment=business_decision['risk_assessment'], data_quality_score=quality_scores['data_quality'], analysis_quality_score=quality_scores['analysis_quality'], overall_experiment_quality=quality_scores['overall_quality'], actionable_insights=insights['actionable_insights'], next_steps=insights['next_steps'], lessons_learned=insights['lessons_learned'], analysis_duration_seconds=analysis_duration, sample_sizes=experiment_data['sample_sizes'], experiment_duration_days=experiment_data['duration_days'])
            await self._store_analysis_result(result)
            logger.info('Experiment analysis completed: %s', experiment_id)
            logger.info('Recommendation: %s', stopping_recommendation['recommendation'])
            return result
        except Exception as e:
            logger.error('Error analyzing experiment {experiment_id}: %s', e)
            raise

    async def stop_experiment(self, experiment_id: str, reason: str='manual') -> dict[str, Any]:
        """Stop an active experiment

        Args:
            experiment_id: Experiment identifier
            reason: Reason for stopping

        Returns:
            Stop operation result
        """
        try:
            logger.info('Stopping experiment {experiment_id}, reason: %s', reason)
            final_analysis = await self.analyze_experiment(experiment_id, force_analysis=True)
            await self._update_experiment_status(experiment_id, ExperimentStatus.COMPLETED)
            if self.real_time_service:
                await self.real_time_service.stop_experiment_monitoring(experiment_id)
            if experiment_id in self.experiment_tasks:
                task_id = self.experiment_tasks[experiment_id]
                await self.task_manager.cancel_task(task_id)
                del self.experiment_tasks[experiment_id]
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]
            logger.info('Experiment stopped successfully: %s', experiment_id)
            return {'success': True, 'experiment_id': experiment_id, 'stop_reason': reason, 'final_analysis': final_analysis, 'status': ExperimentStatus.COMPLETED.value}
        except Exception as e:
            logger.error('Error stopping experiment {experiment_id}: %s', e)
            return {'success': False, 'error': str(e), 'experiment_id': experiment_id}

    async def get_experiment_status(self, experiment_id: str) -> dict[str, Any]:
        """Get current experiment status and metrics

        Args:
            experiment_id: Experiment identifier

        Returns:
            Current experiment status and metrics
        """
        try:
            config = self.active_experiments.get(experiment_id)
            if not config:
                return {'experiment_id': experiment_id, 'status': 'not_found', 'active': False}
            real_time_metrics = None
            if self.real_time_service:
                real_time_metrics = await self.real_time_service.get_real_time_metrics(experiment_id)
            experiment_data = await self._collect_experiment_data(experiment_id, config)
            task_status = None
            if experiment_id in self.experiment_tasks:
                task_id = self.experiment_tasks[experiment_id]
                task_status = self.task_manager.get_enhanced_task_status(task_id)
            return {'experiment_id': experiment_id, 'status': 'active', 'active': True, 'experiment_name': config.experiment_name, 'experiment_type': config.experiment_type.value, 'duration_days': experiment_data['duration_days'], 'sample_sizes': experiment_data['sample_sizes'], 'total_sample_size': sum(experiment_data['sample_sizes'].values()), 'arms': [arm.arm_name for arm in config.arms], 'real_time_metrics': real_time_metrics.__dict__ if real_time_metrics else None, 'sufficient_data': experiment_data['sufficient_data'], 'minimum_sample_size_reached': experiment_data['total_sample_size'] >= config.minimum_sample_size, 'estimated_completion_date': self._estimate_completion_date(config, experiment_data), 'task_management': {'task_manager_integration': True, 'monitoring_task_status': task_status['status'] if task_status else 'unknown', 'task_execution_time': task_status.get('metrics', {}).get('total_duration', 0) if task_status else 0, 'task_retry_count': task_status.get('retry_count', 0) if task_status else 0}}
        except Exception as e:
            logger.error('Error getting experiment status {experiment_id}: %s', e)
            return {'experiment_id': experiment_id, 'status': 'error', 'error': str(e)}

    async def cleanup(self):
        """Clean up orchestrator resources"""
        try:
            logger.info('Cleaning up experiment orchestrator')
            cleanup_tasks = []
            for experiment_id, task_id in list(self.experiment_tasks.items()):
                cleanup_tasks.append(self.task_manager.cancel_task(task_id))
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            if self.real_time_service:
                for experiment_id in list(self.active_experiments.keys()):
                    await self.real_time_service.stop_experiment_monitoring(experiment_id)
                await self.real_time_service.cleanup()
            self.experiment_tasks.clear()
            self.active_experiments.clear()
            logger.info('Experiment orchestrator cleanup completed')
        except Exception as e:
            logger.error('Error during orchestrator cleanup: %s', e)

    def _validate_experiment_config(self, config: ExperimentConfiguration) -> dict[str, Any]:
        """Validate experiment configuration"""
        errors = []
        warnings = []
        if not config.arms or len(config.arms) < 2:
            errors.append('Experiment must have at least 2 arms')
        if config.minimum_sample_size < 20:
            warnings.append('Minimum sample size is very small, consider increasing')
        if config.effect_size_threshold < 0.05:
            warnings.append('Effect size threshold is very small, may lead to false positives')
        if config.statistical_power < 0.7:
            warnings.append('Statistical power is low, consider increasing sample size')
        total_weight = sum(arm.allocation_weight for arm in config.arms)
        if abs(total_weight - len(config.arms)) > 0.01:
            warnings.append('Arm allocation weights may not be balanced')
        if not config.stopping_rules:
            warnings.append('No stopping rules defined, experiment may run indefinitely')
        if config.max_duration_days > 90:
            warnings.append('Very long experiment duration, consider shorter periods')
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}

    def _calculate_sample_size_requirements(self, config: ExperimentConfiguration) -> dict[str, Any]:
        """Calculate required sample sizes for experiment"""
        try:
            from scipy import stats
            alpha = config.significance_level
            power = config.statistical_power
            effect_size = config.effect_size_threshold
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            z_beta = stats.norm.ppf(power)
            n_per_group = (z_alpha + z_beta) ** 2 * 2 / effect_size ** 2
            n_per_group = int(np.ceil(n_per_group))
            n_total = n_per_group * len(config.arms)
            if len(config.arms) > 2:
                corrected_alpha = alpha / (len(config.arms) - 1)
                z_alpha_corrected = stats.norm.ppf(1 - corrected_alpha / 2)
                n_per_group_corrected = (z_alpha_corrected + z_beta) ** 2 * 2 / effect_size ** 2
                n_per_group_corrected = int(np.ceil(n_per_group_corrected))
                n_total_corrected = n_per_group_corrected * len(config.arms)
            else:
                n_per_group_corrected = n_per_group
                n_total_corrected = n_total
            daily_traffic = config.traffic_allocation * 1000
            estimated_days = n_total_corrected / daily_traffic
            return {'required_sample_size_per_group': n_per_group, 'required_total_sample_size': n_total, 'corrected_sample_size_per_group': n_per_group_corrected, 'corrected_total_sample_size': n_total_corrected, 'estimated_duration_days': max(1, int(np.ceil(estimated_days))), 'recommended_minimum': max(config.minimum_sample_size, n_total_corrected), 'power_analysis': {'alpha': alpha, 'power': power, 'effect_size': effect_size, 'n_arms': len(config.arms)}}
        except Exception as e:
            logger.warning('Error calculating sample size: %s', e)
            return {'required_sample_size_per_group': config.minimum_sample_size // len(config.arms), 'required_total_sample_size': config.minimum_sample_size, 'estimated_duration_days': 14, 'error': str(e)}

    async def _create_experiment_record(self, config: ExperimentConfiguration) -> ABExperiment:
        """Create experiment record in database"""
        experiment = ABExperiment(experiment_name=config.experiment_name, description=config.description, control_rules={'arm_id': config.arms[0].arm_id, 'rules': config.arms[0].rules}, treatment_rules={'arms': [{'arm_id': arm.arm_id, 'rules': arm.rules} for arm in config.arms[1:]]}, target_metric=config.primary_metric, sample_size_per_group=config.minimum_sample_size // len(config.arms), status='running', started_at=aware_utc_now(), metadata={'experiment_type': config.experiment_type.value, 'arms': [{'arm_id': arm.arm_id, 'arm_name': arm.arm_name} for arm in config.arms], 'stopping_rules': [rule.value for rule in config.stopping_rules], 'analysis_config': {'causal_analysis_enabled': config.causal_analysis_enabled, 'pattern_analysis_enabled': config.pattern_analysis_enabled, 'statistical_power': config.statistical_power, 'effect_size_threshold': config.effect_size_threshold}})
        self.db_session.add(experiment)
        await self.db_session.commit()
        await self.db_session.refresh(experiment)
        return experiment

    async def _monitor_experiment(self, experiment_id: str):
        """Background task to monitor experiment progress via managed task system"""
        try:
            logger.info('Starting managed experiment monitoring: %s', experiment_id)
            while experiment_id in self.active_experiments:
                try:
                    config = self.active_experiments[experiment_id]
                    should_stop, reason = await self._check_stopping_criteria(experiment_id, config)
                    if should_stop:
                        logger.info('Stopping criteria met for %s: %s', experiment_id, reason)
                        await self.stop_experiment(experiment_id, reason)
                        break
                    await asyncio.sleep(300)
                except asyncio.CancelledError:
                    logger.info('Managed monitoring cancelled for %s', experiment_id)
                    break
                except Exception as e:
                    logger.error('Error in managed experiment monitoring {experiment_id}: %s', e)
                    await asyncio.sleep(300)
        except Exception as e:
            logger.error('Fatal error in managed experiment monitoring {experiment_id}: %s', e)

    async def _check_stopping_criteria(self, experiment_id: str, config: ExperimentConfiguration) -> tuple[bool, str]:
        """Check if experiment should be stopped"""
        try:
            if StoppingRule.TIME_LIMIT in config.stopping_rules:
                experiment_data = await self._collect_experiment_data(experiment_id, config)
                if experiment_data['duration_days'] >= config.max_duration_days:
                    return (True, 'Maximum duration reached')
            if StoppingRule.SAMPLE_SIZE_REACHED in config.stopping_rules:
                experiment_data = await self._collect_experiment_data(experiment_id, config)
                if experiment_data['total_sample_size'] >= config.maximum_sample_size:
                    return (True, 'Maximum sample size reached')
            if StoppingRule.STATISTICAL_SIGNIFICANCE in config.stopping_rules and config.early_stopping_enabled:
                try:
                    analysis = await self.analyze_experiment(experiment_id)
                    if analysis.statistical_validation.primary_test.p_value < config.significance_level and analysis.statistical_validation.practical_significance:
                        return (True, 'Statistical and practical significance achieved')
                except Exception as e:
                    logger.warning('Error checking statistical significance: %s', e)
            return (False, '')
        except Exception as e:
            logger.error('Error checking stopping criteria: %s', e)
            return (False, '')

    async def _collect_experiment_data(self, experiment_id: str, config: ExperimentConfiguration) -> dict[str, Any]:
        """Collect experiment data for analysis"""
        try:
            from sqlalchemy import and_, select
            query = select(RulePerformance).where(and_(RulePerformance.created_at >= aware_utc_now() - timedelta(days=config.max_duration_days)))
            result = await self.db_session.execute(query)
            performance_records = result.scalars().all()
            arms_data = {}
            sample_sizes = {}
            for i, arm in enumerate(config.arms):
                arm_records = performance_records[i::len(config.arms)]
                arms_data[arm.arm_id] = {'outcomes': [r.improvement_score for r in arm_records], 'metrics': {'improvement_score': [r.improvement_score for r in arm_records], 'execution_time_ms': [r.execution_time_ms for r in arm_records], 'user_satisfaction_score': [r.user_satisfaction_score for r in arm_records]}, 'sample_size': len(arm_records)}
                sample_sizes[arm.arm_id] = len(arm_records)
            total_sample_size = sum(sample_sizes.values())
            duration_days = min(config.max_duration_days, (aware_utc_now() - aware_utc_now().replace(hour=0, minute=0, second=0)).days + 1)
            return {'arms': arms_data, 'sample_sizes': sample_sizes, 'total_sample_size': total_sample_size, 'duration_days': duration_days, 'sufficient_data': total_sample_size >= config.minimum_sample_size, 'primary_metric': config.primary_metric, 'secondary_metrics': config.secondary_metrics}
        except Exception as e:
            logger.error('Error collecting experiment data: %s', e)
            return {'arms': {}, 'sample_sizes': {}, 'total_sample_size': 0, 'duration_days': 0, 'sufficient_data': False}

    async def _perform_statistical_analysis(self, experiment_data: dict[str, Any], config: ExperimentConfiguration) -> AdvancedValidationResult:
        """Perform statistical analysis of experiment"""
        try:
            arms = list(experiment_data['arms'].items())
            if len(arms) < 2:
                raise ValueError('Insufficient arms for statistical analysis')
            control_arm_id, control_data = arms[0]
            treatment_arm_id, treatment_data = arms[1]
            control_outcomes = control_data['outcomes']
            treatment_outcomes = treatment_data['outcomes']
            if len(control_outcomes) < 3 or len(treatment_outcomes) < 3:
                raise ValueError('Insufficient data for statistical analysis')
            result = self.statistical_validator.validate_ab_test(control_data=control_outcomes, treatment_data=treatment_outcomes, validate_assumptions=True, include_bootstrap=True, include_sensitivity=True)
            return result
        except Exception as e:
            logger.error('Error in statistical analysis: %s', e)
            raise

    async def _perform_pattern_analysis(self, experiment_data: dict[str, Any], config: ExperimentConfiguration) -> PatternSignificanceReport:
        """Perform pattern significance analysis"""
        try:
            patterns_data = {}
            control_patterns = {}
            treatment_patterns = {}
            for arm_id, arm_data in experiment_data['arms'].items():
                if not arm_data['outcomes']:
                    continue
                outcomes = np.array(arm_data['outcomes'])
                pattern_id = f'performance_distribution_{arm_id}'
                patterns_data[pattern_id] = {'type': 'performance_distribution', 'description': f'Performance distribution for {arm_id}'}
                high_perf = np.sum(outcomes > np.percentile(outcomes, 75))
                med_perf = np.sum((outcomes >= np.percentile(outcomes, 25)) & (outcomes <= np.percentile(outcomes, 75)))
                low_perf = np.sum(outcomes < np.percentile(outcomes, 25))
                if arm_id == list(experiment_data['arms'].keys())[0]:
                    control_patterns[pattern_id] = {'high': high_perf, 'medium': med_perf, 'low': low_perf}
                else:
                    treatment_patterns[pattern_id] = {'high': high_perf, 'medium': med_perf, 'low': low_perf}
            if not patterns_data:
                return None
            result = self.pattern_analyzer.analyze_pattern_significance(patterns_data=patterns_data, control_data=control_patterns, treatment_data=treatment_patterns)
            return result
        except Exception as e:
            logger.error('Error in pattern analysis: %s', e)
            return None

    async def _perform_causal_analysis(self, experiment_data: dict[str, Any], config: ExperimentConfiguration) -> CausalInferenceResult:
        """Perform causal inference analysis"""
        try:
            arms = list(experiment_data['arms'].items())
            if len(arms) < 2:
                return None
            all_outcomes = []
            all_treatments = []
            for i, (arm_id, arm_data) in enumerate(arms):
                outcomes = arm_data['outcomes']
                treatments = [1 if i > 0 else 0] * len(outcomes)
                all_outcomes.extend(outcomes)
                all_treatments.extend(treatments)
            if len(all_outcomes) < 20:
                return None
            result = self.causal_analyzer.analyze_causal_effect(outcome_data=np.array(all_outcomes), treatment_data=np.array(all_treatments), assignment_mechanism=TreatmentAssignment.RANDOMIZED, method=CausalMethod.DIFFERENCE_IN_DIFFERENCES)
            return result
        except Exception as e:
            logger.error('Error in causal analysis: %s', e)
            return None

    def _calculate_performance_metrics(self, experiment_data: dict[str, Any], statistical_result: AdvancedValidationResult) -> dict[str, Any]:
        """Calculate arm performance metrics"""
        arm_performance = {}
        relative_performance = {}
        try:
            control_arm_id = list(experiment_data['arms'].keys())[0]
            control_outcomes = experiment_data['arms'][control_arm_id]['outcomes']
            control_mean = np.mean(control_outcomes) if control_outcomes else 0
            for arm_id, arm_data in experiment_data['arms'].items():
                if not arm_data['outcomes']:
                    continue
                outcomes = arm_data['outcomes']
                arm_mean = np.mean(outcomes)
                arm_std = np.std(outcomes, ddof=1) if len(outcomes) > 1 else 0
                arm_performance[arm_id] = {'mean': float(arm_mean), 'std': float(arm_std), 'sample_size': len(outcomes), 'confidence_interval': self._calculate_ci(outcomes) if len(outcomes) > 1 else (arm_mean, arm_mean)}
                if control_mean > 0:
                    relative_lift = (arm_mean - control_mean) / control_mean
                    relative_performance[arm_id] = {'absolute_difference': float(arm_mean - control_mean), 'relative_lift': float(relative_lift), 'relative_lift_percent': float(relative_lift * 100)}
                else:
                    relative_performance[arm_id] = {'absolute_difference': float(arm_mean - control_mean), 'relative_lift': 0.0, 'relative_lift_percent': 0.0}
        except Exception as e:
            logger.warning('Error calculating performance metrics: %s', e)
        return {'arm_performance': arm_performance, 'relative_performance': relative_performance}

    def _calculate_ci(self, data: list[float], confidence: float=0.95) -> tuple[float, float]:
        """Calculate confidence interval for data"""
        try:
            from scipy import stats
            data_array = np.array(data)
            mean = np.mean(data_array)
            se = stats.sem(data_array)
            h = se * stats.t.ppf((1 + confidence) / 2.0, len(data_array) - 1)
            return (float(mean - h), float(mean + h))
        except Exception:
            mean = np.mean(data)
            return (mean, mean)

    def _generate_stopping_recommendation(self, config: ExperimentConfiguration, statistical_result: AdvancedValidationResult, causal_result: CausalInferenceResult | None, performance_metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate stopping recommendation"""
        try:
            statistically_significant = statistical_result.primary_test.p_value < config.significance_level
            practically_significant = statistical_result.practical_significance
            validation_quality = statistical_result.validation_quality_score
            causal_quality = causal_result.overall_quality_score if causal_result else 0.5
            confidence = (validation_quality + causal_quality + (0.3 if statistically_significant else 0) + (0.2 if practically_significant else 0)) / 2
            if statistically_significant and practically_significant and (validation_quality > 0.7):
                recommendation = 'STOP_FOR_SUCCESS'
                reason = 'Strong evidence for treatment effect with high confidence'
            elif statistically_significant and practically_significant:
                recommendation = 'STOP_WITH_CAUTION'
                reason = 'Treatment effect detected but some quality concerns'
            elif validation_quality > 0.8 and (not statistically_significant):
                recommendation = 'STOP_FOR_FUTILITY'
                reason = 'High quality analysis shows no meaningful effect'
            else:
                recommendation = 'CONTINUE'
                reason = 'Insufficient evidence to make decision'
            return {'recommendation': recommendation, 'reason': reason, 'confidence': float(min(max(confidence, 0.0), 1.0)), 'criteria_met': {'statistical_significance': statistically_significant, 'practical_significance': practically_significant, 'validation_quality': validation_quality > 0.7, 'causal_quality': causal_quality > 0.7 if causal_result else True}}
        except Exception as e:
            logger.error('Error generating stopping recommendation: %s', e)
            return {'recommendation': 'CONTINUE', 'reason': 'Error in analysis', 'confidence': 0.0}

    def _generate_business_decision(self, config: ExperimentConfiguration, statistical_result: AdvancedValidationResult, causal_result: CausalInferenceResult | None, performance_metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate business decision framework"""
        try:
            effect_size = statistical_result.primary_test.effect_size
            p_value = statistical_result.primary_test.p_value
            risk_factors = []
            if p_value > 0.01:
                risk_factors.append('Moderate statistical evidence')
            if abs(effect_size) < 0.2:
                risk_factors.append('Small effect size')
            if statistical_result.validation_quality_score < 0.7:
                risk_factors.append('Quality concerns in analysis')
            if statistical_result.practical_significance and len(risk_factors) <= 1:
                decision = 'IMPLEMENT'
                rationale = 'Strong evidence supports implementation'
            elif statistical_result.practical_significance:
                decision = 'PILOT'
                rationale = 'Consider gradual rollout with monitoring'
            else:
                decision = 'NO_ACTION'
                rationale = 'Insufficient evidence for business impact'
            return {'decision': decision, 'rationale': rationale, 'risk_assessment': {'risk_level': 'low' if len(risk_factors) <= 1 else 'medium' if len(risk_factors) <= 2 else 'high', 'risk_factors': risk_factors, 'mitigation_strategies': ['Monitor key metrics closely', 'Set up early warning systems', 'Plan rollback procedures']}}
        except Exception as e:
            logger.error('Error generating business decision: %s', e)
            return {'decision': 'NO_ACTION', 'rationale': 'Error in analysis'}

    def _calculate_quality_scores(self, experiment_data: dict[str, Any], statistical_result: AdvancedValidationResult, pattern_result: PatternSignificanceReport | None, causal_result: CausalInferenceResult | None) -> dict[str, float]:
        """Calculate overall quality scores"""
        try:
            data_quality_components = []
            total_sample = experiment_data['total_sample_size']
            sample_score = min(total_sample / 1000, 1.0)
            data_quality_components.append(('sample_size', sample_score, 0.4))
            sample_sizes = list(experiment_data['sample_sizes'].values())
            if sample_sizes:
                balance_score = min(sample_sizes) / max(sample_sizes)
                data_quality_components.append(('balance', balance_score, 0.3))
            completeness_score = 1.0 if experiment_data['sufficient_data'] else 0.5
            data_quality_components.append(('completeness', completeness_score, 0.3))
            data_quality = sum((score * weight for _, score, weight in data_quality_components))
            analysis_quality = statistical_result.validation_quality_score
            quality_components = [('data_quality', data_quality, 0.4), ('statistical_analysis', analysis_quality, 0.4)]
            if pattern_result:
                quality_components.append(('pattern_analysis', pattern_result.quality_score, 0.1))
            if causal_result:
                quality_components.append(('causal_analysis', causal_result.overall_quality_score, 0.1))
            total_weight = sum((weight for _, _, weight in quality_components))
            overall_quality = sum((score * weight / total_weight for _, score, weight in quality_components))
            return {'data_quality': float(min(max(data_quality, 0.0), 1.0)), 'analysis_quality': float(min(max(analysis_quality, 0.0), 1.0)), 'overall_quality': float(min(max(overall_quality, 0.0), 1.0))}
        except Exception as e:
            logger.error('Error calculating quality scores: %s', e)
            return {'data_quality': 0.5, 'analysis_quality': 0.5, 'overall_quality': 0.5}

    def _generate_actionable_insights(self, config: ExperimentConfiguration, statistical_result: AdvancedValidationResult, pattern_result: PatternSignificanceReport | None, causal_result: CausalInferenceResult | None, performance_metrics: dict[str, Any]) -> dict[str, list[str]]:
        """Generate actionable insights and recommendations"""
        actionable_insights = []
        next_steps = []
        lessons_learned = []
        try:
            if statistical_result.practical_significance:
                effect_size = statistical_result.primary_test.effect_size
                actionable_insights.append(f'Treatment shows {statistical_result.effect_size_magnitude.value} effect size ({effect_size:.3f})')
                if statistical_result.primary_test.p_value < 0.001:
                    actionable_insights.append('Very strong statistical evidence (p < 0.001)')
            if pattern_result and pattern_result.significant_patterns:
                actionable_insights.append(f'Detected {len(pattern_result.significant_patterns)} significant behavioral patterns')
                for insight in pattern_result.business_insights[:3]:
                    actionable_insights.append(f'Pattern insight: {insight}')
            if causal_result and causal_result.causal_interpretation:
                actionable_insights.append(f'Causal analysis: {causal_result.causal_interpretation}')
                for rec in causal_result.business_recommendations[:2]:
                    actionable_insights.append(rec)
            if statistical_result.practical_significance:
                next_steps.extend(['Plan implementation rollout strategy', 'Set up monitoring for key metrics', 'Prepare stakeholder communication'])
            else:
                next_steps.extend(["Investigate why treatment didn't show effect", 'Consider alternative approaches', 'Plan follow-up experiments'])
            if statistical_result.validation_quality_score > 0.8:
                lessons_learned.append('High-quality experimental design enabled reliable conclusions')
            if config.causal_analysis_enabled and causal_result:
                lessons_learned.append('Causal analysis provided deeper understanding beyond correlation')
            if pattern_result and pattern_result.pattern_interactions:
                lessons_learned.append('Pattern analysis revealed interaction effects worth investigating')
        except Exception as e:
            logger.error('Error generating insights: %s', e)
            actionable_insights.append('Error generating insights - manual review recommended')
        return {'actionable_insights': actionable_insights, 'next_steps': next_steps, 'lessons_learned': lessons_learned}

    def _determine_experiment_status(self, config: ExperimentConfiguration, stopping_recommendation: dict[str, Any]) -> ExperimentStatus:
        """Determine experiment status based on recommendation"""
        recommendation = stopping_recommendation.get('recommendation', 'CONTINUE')
        if recommendation in ['STOP_FOR_SUCCESS', 'STOP_WITH_CAUTION', 'STOP_FOR_FUTILITY']:
            return ExperimentStatus.analyzing
        return ExperimentStatus.active

    async def _store_analysis_result(self, result: ExperimentResult):
        """Store analysis result in database"""
        try:
            from sqlalchemy import select, update
            query = select(ABExperiment).where(ABExperiment.experiment_name == result.experiment_id)
            db_result = await self.db_session.execute(query)
            experiment = db_result.scalar_one_or_none()
            if experiment:
                update_query = update(ABExperiment).where(ABExperiment.experiment_id == experiment.experiment_id).values(analysis_results={'analysis_id': result.analysis_id, 'timestamp': result.timestamp.isoformat(), 'statistical_significance': result.statistical_validation.primary_test.p_value < 0.05, 'effect_size': result.statistical_validation.primary_test.effect_size, 'business_decision': result.business_decision, 'confidence_level': result.confidence_level, 'quality_score': result.overall_experiment_quality}, completed_at=aware_utc_now() if result.experiment_status == ExperimentStatus.COMPLETED else None)
                await self.db_session.execute(update_query)
                await self.db_session.commit()
        except Exception as e:
            logger.error('Error storing analysis result: %s', e)

    async def _update_experiment_status(self, experiment_id: str, status: ExperimentStatus):
        """Update experiment status in database"""
        try:
            from sqlalchemy import select, update
            query = select(ABExperiment).where(ABExperiment.experiment_name == experiment_id)
            result = await self.db_session.execute(query)
            experiment = result.scalar_one_or_none()
            if experiment:
                update_query = update(ABExperiment).where(ABExperiment.experiment_id == experiment.experiment_id).values(status=status.value, completed_at=aware_utc_now() if status == ExperimentStatus.COMPLETED else None)
                await self.db_session.execute(update_query)
                await self.db_session.commit()
        except Exception as e:
            logger.error('Error updating experiment status: %s', e)

    async def _load_experiment_config(self, experiment_id: str) -> ExperimentConfiguration | None:
        """Load experiment configuration from database"""
        return None

    def _create_insufficient_data_result(self, experiment_id: str, analysis_id: str, experiment_data: dict[str, Any]) -> ExperimentResult:
        """Create result for insufficient data scenario"""
        from .advanced_statistical_validator import EffectSizeMagnitude, StatisticalTestResult
        placeholder_test = StatisticalTestResult(test_name='Insufficient Data', statistic=0.0, p_value=1.0, effect_size=0.0, effect_size_type='None')
        placeholder_validation = AdvancedValidationResult(validation_id=analysis_id, timestamp=aware_utc_now(), primary_test=placeholder_test, effect_size_magnitude=EffectSizeMagnitude.NEGLIGIBLE, practical_significance=False, clinical_significance=False, validation_quality_score=0.0)
        return ExperimentResult(experiment_id=experiment_id, analysis_id=analysis_id, timestamp=aware_utc_now(), experiment_status=ExperimentStatus.active, statistical_validation=placeholder_validation, stopping_recommendation='CONTINUE - Insufficient data', business_decision='WAIT - Collect more data', confidence_level=0.0, data_quality_score=0.3, analysis_quality_score=0.0, overall_experiment_quality=0.2, actionable_insights=['Insufficient data for analysis', 'Continue data collection'], next_steps=['Wait for more participants', 'Check data collection issues'], sample_sizes=experiment_data['sample_sizes'], experiment_duration_days=experiment_data['duration_days'])

    def _estimate_completion_date(self, config: ExperimentConfiguration, experiment_data: dict[str, Any]) -> str | None:
        """Estimate experiment completion date"""
        try:
            current_sample = experiment_data['total_sample_size']
            target_sample = config.minimum_sample_size
            if current_sample >= target_sample:
                return 'Ready for analysis'
            duration_days = experiment_data['duration_days']
            if duration_days > 0:
                daily_rate = current_sample / duration_days
                if daily_rate > 0:
                    days_remaining = (target_sample - current_sample) / daily_rate
                    completion_date = aware_utc_now() + timedelta(days=days_remaining)
                    return format_date_only(completion_date)
            return 'Unable to estimate'
        except Exception as e:
            logger.warning('Error estimating completion date: %s', e)
            return 'Unknown'

    async def get_orchestrator_status(self) -> dict[str, Any]:
        """Get comprehensive orchestrator status with task management information."""
        try:
            task_stats = self.task_manager.get_statistics() if self.task_manager else {}
            experiment_task_details = []
            for experiment_id, task_id in self.experiment_tasks.items():
                task_status = self.task_manager.get_enhanced_task_status(task_id)
                if task_status:
                    experiment_task_details.append({'experiment_id': experiment_id, 'task_id': task_id, 'status': task_status['status'], 'execution_time': task_status.get('metrics', {}).get('total_duration', 0), 'retry_count': task_status.get('retry_count', 0), 'created_at': task_status.get('created_at'), 'started_at': task_status.get('started_at')})
            return {'orchestrator_status': {'active_experiments': len(self.active_experiments), 'monitoring_tasks': len(self.experiment_tasks), 'task_manager_integration': True, 'bayesian_analysis_enabled': self.bayesian_monitoring_enabled, 'task_monitoring_enabled': self.task_monitoring_enabled}, 'experiment_details': [{'experiment_id': exp_id, 'experiment_name': config.experiment_name, 'experiment_type': config.experiment_type.value, 'arms': len(config.arms), 'max_duration_days': config.max_duration_days} for exp_id, config in self.active_experiments.items()], 'task_management': {'task_manager_type': type(self.task_manager).__name__, 'task_manager_stats': task_stats, 'experiment_tasks': experiment_task_details, 'total_tasks_managed': task_stats.get('total_submitted', 0), 'currently_running_tasks': task_stats.get('currently_running', 0)}, 'performance_metrics': {'task_success_rate': task_stats.get('total_completed', 0) / max(1, task_stats.get('total_submitted', 0)), 'task_failure_rate': task_stats.get('total_failed', 0) / max(1, task_stats.get('total_submitted', 0)), 'task_retry_rate': task_stats.get('total_retries', 0) / max(1, task_stats.get('total_submitted', 0))}}
        except Exception as e:
            logger.error('Error getting orchestrator status: %s', e)
            return {'orchestrator_status': {'error': str(e)}, 'active_experiments': len(self.active_experiments), 'monitoring_tasks': len(self.experiment_tasks)}

    async def enable_bayesian_analysis(self, config: BayesianExperimentConfig=None) -> bool:
        """Enable Bayesian analysis capabilities for experiments.

        Args:
            config: Bayesian experiment configuration

        Returns:
            True if successfully enabled, False otherwise
        """
        try:
            from ...performance.testing.ab_testing_service import ModernABTestingService
            self.bayesian_analyzer = ModernABTestingService()
            self.bayesian_config = config or BayesianExperimentConfig()
            self.bayesian_monitoring_enabled = True
            logger.info('Bayesian analysis enabled for ExperimentOrchestrator')
            return True
        except ImportError as e:
            logger.error('Failed to enable Bayesian analysis: %s', e)
            return False

    async def run_bayesian_experiment(self, config: ExperimentConfiguration) -> dict[str, Any]:
        """Run experiment with integrated Bayesian analysis.

        Args:
            config: Experiment configuration

        Returns:
            Experiment results with Bayesian analysis
        """
        if not self.bayesian_monitoring_enabled:
            await self.enable_bayesian_analysis()
        try:
            setup_result = await self.setup_experiment(config)
            if not setup_result.get('success', False):
                return setup_result
            experiment = await self._create_experiment_record(config)
            experiment_results = await self._run_experiment_with_bayesian_monitoring(experiment, config)
            return {'success': True, 'experiment_id': config.experiment_id, 'setup_result': setup_result, 'experiment_results': experiment_results, 'bayesian_analysis': experiment_results.get('bayesian_analysis'), 'recommendations': self._generate_bayesian_recommendations(experiment_results)}
        except Exception as e:
            logger.error('Bayesian experiment failed: %s', e)
            return {'success': False, 'error': str(e), 'experiment_id': config.experiment_id}

    async def _run_experiment_with_bayesian_monitoring(self, experiment: ABExperiment, config: ExperimentConfiguration) -> dict[str, Any]:
        """Run experiment with real-time Bayesian monitoring."""
        experiment_data = {'phases': [], 'bayesian_analysis': None, 'early_stopping': False, 'final_recommendation': None}
        for phase_num in range(1, 6):
            phase_data = await self._simulate_experiment_phase(experiment, phase_num)
            experiment_data['phases'].append(phase_data)
            if phase_data['total_samples'] >= self.bayesian_config.minimum_sample_size:
                bayesian_result = await self._analyze_experiment_bayesian(phase_data)
                experiment_data['bayesian_analysis'] = bayesian_result
                if await self._should_stop_experiment_bayesian(bayesian_result):
                    experiment_data['early_stopping'] = True
                    experiment_data['stopping_reason'] = self._get_stopping_reason(bayesian_result)
                    break
        if experiment_data['bayesian_analysis']:
            experiment_data['final_recommendation'] = self._generate_final_recommendation(experiment_data['bayesian_analysis'])
        return experiment_data

    async def _analyze_experiment_bayesian(self, experiment_data: dict[str, Any]) -> dict[str, Any]:
        """Perform Bayesian analysis on experiment data."""
        if not self.bayesian_analyzer:
            return {'error': 'Bayesian analyzer not available'}
        try:
            control_data = experiment_data.get('control', {})
            treatment_data = experiment_data.get('treatment', {})
            bayesian_result = await self.bayesian_analyzer._bayesian_analysis(control_conversions=control_data.get('conversions', 0), control_visitors=control_data.get('visitors', 1), treatment_conversions=treatment_data.get('conversions', 0), treatment_visitors=treatment_data.get('visitors', 1))
            return {'probability_of_improvement': getattr(bayesian_result, 'probability_of_improvement', 0.5), 'expected_loss': getattr(bayesian_result, 'expected_loss', 0.0), 'confidence_interval': getattr(bayesian_result, 'confidence_interval', [0.0, 1.0]), 'statistical_power': getattr(bayesian_result, 'statistical_power', 0.8), 'method': 'beta_binomial_conjugate_prior', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            logger.error('Bayesian analysis failed: %s', e)
            return {'error': str(e)}

    async def _should_stop_experiment_bayesian(self, bayesian_result: dict[str, Any]) -> bool:
        """Determine if experiment should be stopped based on Bayesian analysis."""
        if 'error' in bayesian_result:
            return False
        probability_of_improvement = bayesian_result.get('probability_of_improvement', 0.5)
        if probability_of_improvement >= self.bayesian_config.confidence_threshold:
            return True
        probability_of_harm = 1 - probability_of_improvement
        if probability_of_harm >= 1 - self.bayesian_config.harm_threshold:
            return True
        return False

    def _get_stopping_reason(self, bayesian_result: dict[str, Any]) -> str:
        """Get reason for early stopping."""
        probability_of_improvement = bayesian_result.get('probability_of_improvement', 0.5)
        if probability_of_improvement >= self.bayesian_config.confidence_threshold:
            return f'High confidence of improvement ({probability_of_improvement:.3f})'
        probability_of_harm = 1 - probability_of_improvement
        if probability_of_harm >= 1 - self.bayesian_config.harm_threshold:
            return f'High confidence of harm ({probability_of_harm:.3f})'
        return 'Unknown'

    def _generate_bayesian_recommendations(self, experiment_results: dict[str, Any]) -> list[str]:
        """Generate recommendations based on Bayesian analysis."""
        recommendations = []
        bayesian_analysis = experiment_results.get('bayesian_analysis')
        if not bayesian_analysis or 'error' in bayesian_analysis:
            recommendations.append('Bayesian analysis unavailable - use traditional statistical methods')
            return recommendations
        probability_of_improvement = bayesian_analysis.get('probability_of_improvement', 0.5)
        if probability_of_improvement >= 0.95:
            recommendations.append('Strong evidence for treatment - recommend full rollout')
        elif probability_of_improvement >= 0.8:
            recommendations.append('Moderate evidence for treatment - consider gradual rollout')
        elif probability_of_improvement <= 0.2:
            recommendations.append('Strong evidence against treatment - do not implement')
        elif probability_of_improvement <= 0.5:
            recommendations.append('Weak evidence against treatment - consider alternative approaches')
        else:
            recommendations.append('Inconclusive results - consider extending experiment or increasing sample size')
        if experiment_results.get('early_stopping'):
            recommendations.append(f"Early stopping triggered: {experiment_results.get('stopping_reason')}")
        return recommendations

    async def _simulate_experiment_phase(self, experiment: ABExperiment, phase_num: int) -> dict[str, Any]:
        """Simulate experiment phase data (replace with real data collection in production)."""
        import random
        base_samples = 50 * phase_num
        control_visitors = base_samples + random.randint(-10, 10)
        treatment_visitors = base_samples + random.randint(-10, 10)
        control_rate = 0.15 + random.uniform(-0.02, 0.02)
        treatment_rate = 0.18 + random.uniform(-0.02, 0.02)
        control_conversions = int(control_visitors * control_rate)
        treatment_conversions = int(treatment_visitors * treatment_rate)
        return {'phase': phase_num, 'control': {'visitors': control_visitors, 'conversions': control_conversions, 'conversion_rate': control_conversions / control_visitors if control_visitors > 0 else 0}, 'treatment': {'visitors': treatment_visitors, 'conversions': treatment_conversions, 'conversion_rate': treatment_conversions / treatment_visitors if treatment_visitors > 0 else 0}, 'total_samples': control_visitors + treatment_visitors, 'timestamp': datetime.now().isoformat()}

    def _generate_final_recommendation(self, bayesian_analysis: dict[str, Any]) -> str:
        """Generate final experiment recommendation."""
        if 'error' in bayesian_analysis:
            return 'Unable to generate recommendation due to analysis error'
        probability_of_improvement = bayesian_analysis.get('probability_of_improvement', 0.5)
        if probability_of_improvement >= 0.95:
            return 'IMPLEMENT: Very strong evidence for treatment effectiveness'
        elif probability_of_improvement >= 0.8:
            return 'CONSIDER: Strong evidence for treatment effectiveness'
        elif probability_of_improvement >= 0.6:
            return 'MONITOR: Moderate evidence for treatment effectiveness'
        elif probability_of_improvement >= 0.4:
            return 'INCONCLUSIVE: Insufficient evidence for decision'
        elif probability_of_improvement >= 0.2:
            return 'RECONSIDER: Weak evidence against treatment'
        else:
            return 'REJECT: Strong evidence against treatment effectiveness'

async def quick_experiment_setup(experiment_name: str, control_rules: dict[str, Any], treatment_rules: dict[str, Any], db_session: AsyncSession, task_manager: EnhancedBackgroundTaskManager | None=None, **kwargs) -> dict[str, Any]:
    """Quick experiment setup for immediate use"""
    try:
        control_arm = ExperimentArm(arm_id='control', arm_name='Control', description='Control arm', rules=control_rules)
        treatment_arm = ExperimentArm(arm_id='treatment', arm_name='Treatment', description='Treatment arm', rules=treatment_rules)
        config = ExperimentConfiguration(experiment_id=experiment_name, experiment_name=experiment_name, experiment_type=ExperimentType.SIMPLE_AB, description=f'A/B test: {experiment_name}', arms=[control_arm, treatment_arm], **kwargs)
        orchestrator = ExperimentOrchestrator(db_session=db_session, task_manager=task_manager)
        result = await orchestrator.setup_experiment(config)
        return result
    except Exception as e:
        return {'success': False, 'error': str(e), 'experiment_id': experiment_name}
