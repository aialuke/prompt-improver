"""
Experiment Orchestrator for Advanced A/B Testing
Coordinates and manages complex multi-variate experiments with causal inference
"""

import logging
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from enum import Enum
import json

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from .advanced_statistical_validator import AdvancedStatisticalValidator, AdvancedValidationResult
from .pattern_significance_analyzer import PatternSignificanceAnalyzer, PatternSignificanceReport
from .causal_inference_analyzer import CausalInferenceAnalyzer, CausalInferenceResult, CausalMethod
from ..services.real_time_analytics import RealTimeAnalyticsService, RealTimeMetrics
from ..database.models import ABExperiment, RulePerformance

logger = logging.getLogger(__name__)


class ExperimentType(Enum):
    """Types of experiments that can be orchestrated"""
    SIMPLE_AB = "simple_ab"
    MULTIVARIATE = "multivariate"
    FACTORIAL = "factorial"
    SEQUENTIAL = "sequential"
    BANDITS = "bandits"
    CAUSAL_INFERENCE = "causal_inference"


class ExperimentStatus(Enum):
    """Experiment lifecycle states"""
    PLANNED = "planned"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ANALYZING = "analyzing"
    ARCHIVED = "archived"


class StoppingRule(Enum):
    """Early stopping criteria"""
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    PRACTICAL_SIGNIFICANCE = "practical_significance"
    SEQUENTIAL_TESTING = "sequential_testing"
    BAYESIAN_DECISION = "bayesian_decision"
    SAMPLE_SIZE_REACHED = "sample_size_reached"
    TIME_LIMIT = "time_limit"


@dataclass
class ExperimentArm:
    """Represents a single experiment arm/variant"""
    arm_id: str
    arm_name: str
    description: str
    rules: Dict[str, Any]
    allocation_weight: float = 1.0
    current_allocation: float = 0.0
    sample_size: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfiguration:
    """Configuration for experiment setup"""
    experiment_id: str
    experiment_name: str
    experiment_type: ExperimentType
    description: str
    
    # Arms configuration
    arms: List[ExperimentArm]
    
    # Targeting and allocation
    target_population: Dict[str, Any] = field(default_factory=dict)
    traffic_allocation: float = 1.0
    
    # Sample size and power
    minimum_sample_size: int = 100
    maximum_sample_size: int = 10000
    statistical_power: float = 0.8
    effect_size_threshold: float = 0.1
    significance_level: float = 0.05
    
    # Stopping rules
    stopping_rules: List[StoppingRule] = field(default_factory=list)
    max_duration_days: int = 30
    early_stopping_enabled: bool = True
    
    # Analysis configuration
    primary_metric: str = "improvement_score"
    secondary_metrics: List[str] = field(default_factory=list)
    causal_analysis_enabled: bool = True
    pattern_analysis_enabled: bool = True
    
    # Advanced settings
    sequential_testing_enabled: bool = False
    adaptive_allocation_enabled: bool = False
    multi_armed_bandit_config: Optional[Dict[str, Any]] = None
    
    # Quality controls
    guardrail_metrics: List[str] = field(default_factory=list)
    quality_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Comprehensive experiment analysis result"""
    experiment_id: str
    analysis_id: str
    timestamp: datetime
    experiment_status: ExperimentStatus
    
    # Statistical analysis
    statistical_validation: AdvancedValidationResult
    pattern_significance: Optional[PatternSignificanceReport] = None
    causal_inference: Optional[CausalInferenceResult] = None
    
    # Performance metrics
    arm_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    relative_performance: Dict[str, float] = field(default_factory=dict)
    
    # Decision framework
    stopping_recommendation: str = ""
    business_decision: str = ""
    confidence_level: float = 0.0
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    data_quality_score: float = 0.0
    analysis_quality_score: float = 0.0
    overall_experiment_quality: float = 0.0
    
    # Recommendations
    actionable_insights: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    
    # Metadata
    analysis_duration_seconds: float = 0.0
    sample_sizes: Dict[str, int] = field(default_factory=dict)
    experiment_duration_days: float = 0.0


class ExperimentOrchestrator:
    """Advanced experiment orchestrator implementing 2025 best practices"""
    
    def __init__(self,
                 db_session: AsyncSession,
                 statistical_validator: Optional[AdvancedStatisticalValidator] = None,
                 pattern_analyzer: Optional[PatternSignificanceAnalyzer] = None,
                 causal_analyzer: Optional[CausalInferenceAnalyzer] = None,
                 real_time_service: Optional[RealTimeAnalyticsService] = None):
        """
        Initialize experiment orchestrator
        
        Args:
            db_session: Database session for experiment management
            statistical_validator: Advanced statistical validator
            pattern_analyzer: Pattern significance analyzer
            causal_analyzer: Causal inference analyzer
            real_time_service: Real-time analytics service
        """
        self.db_session = db_session
        
        # Initialize analyzers with defaults if not provided
        self.statistical_validator = statistical_validator or AdvancedStatisticalValidator()
        self.pattern_analyzer = pattern_analyzer or PatternSignificanceAnalyzer()
        self.causal_analyzer = causal_analyzer or CausalInferenceAnalyzer()
        self.real_time_service = real_time_service
        
        # Active experiments tracking
        self.active_experiments: Dict[str, ExperimentConfiguration] = {}
        self.experiment_tasks: Dict[str, asyncio.Task] = {}
        
    async def setup_experiment(self, config: ExperimentConfiguration) -> Dict[str, Any]:
        """
        Set up and validate a new experiment
        
        Args:
            config: Experiment configuration
            
        Returns:
            Setup result with validation and recommendations
        """
        try:
            logger.info(f"Setting up experiment: {config.experiment_id}")
            
            # Validate experiment configuration
            validation_result = self._validate_experiment_config(config)
            
            if not validation_result['valid']:
                return {
                    'success': False,
                    'errors': validation_result['errors'],
                    'experiment_id': config.experiment_id
                }
            
            # Calculate sample size requirements
            sample_size_analysis = self._calculate_sample_size_requirements(config)
            
            # Create database experiment record
            experiment_record = await self._create_experiment_record(config)
            
            # Initialize real-time monitoring if available
            if self.real_time_service:
                await self.real_time_service.start_experiment_monitoring(
                    experiment_id=config.experiment_id,
                    update_interval=60  # 1 minute intervals
                )
            
            # Store in active experiments
            self.active_experiments[config.experiment_id] = config
            
            # Start experiment monitoring task
            monitoring_task = asyncio.create_task(
                self._monitor_experiment(config.experiment_id)
            )
            self.experiment_tasks[config.experiment_id] = monitoring_task
            
            logger.info(f"Experiment setup completed: {config.experiment_id}")
            
            return {
                'success': True,
                'experiment_id': config.experiment_id,
                'experiment_record_id': experiment_record.experiment_id,
                'sample_size_analysis': sample_size_analysis,
                'validation_warnings': validation_result.get('warnings', []),
                'estimated_duration_days': sample_size_analysis.get('estimated_duration_days'),
                'monitoring_enabled': self.real_time_service is not None
            }
            
        except Exception as e:
            logger.error(f"Error setting up experiment {config.experiment_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'experiment_id': config.experiment_id
            }
    
    async def analyze_experiment(self, experiment_id: str,
                               force_analysis: bool = False) -> ExperimentResult:
        """
        Perform comprehensive experiment analysis
        
        Args:
            experiment_id: Experiment identifier
            force_analysis: Force analysis even if stopping criteria not met
            
        Returns:
            Comprehensive experiment analysis result
        """
        analysis_start = datetime.utcnow()
        analysis_id = f"analysis_{experiment_id}_{analysis_start.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"Starting experiment analysis: {experiment_id}")
            
            # Get experiment configuration
            config = self.active_experiments.get(experiment_id)
            if not config:
                # Try to load from database
                config = await self._load_experiment_config(experiment_id)
            
            if not config:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            # Collect experiment data
            experiment_data = await self._collect_experiment_data(experiment_id, config)
            
            if not experiment_data['sufficient_data']:
                return self._create_insufficient_data_result(
                    experiment_id, analysis_id, experiment_data
                )
            
            # Statistical validation
            logger.info(f"Performing statistical validation for {experiment_id}")
            statistical_result = await self._perform_statistical_analysis(
                experiment_data, config
            )
            
            # Pattern significance analysis
            pattern_result = None
            if config.pattern_analysis_enabled and len(experiment_data['arms']) > 1:
                logger.info(f"Performing pattern analysis for {experiment_id}")
                pattern_result = await self._perform_pattern_analysis(
                    experiment_data, config
                )
            
            # Causal inference analysis
            causal_result = None
            if config.causal_analysis_enabled:
                logger.info(f"Performing causal analysis for {experiment_id}")
                causal_result = await self._perform_causal_analysis(
                    experiment_data, config
                )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                experiment_data, statistical_result
            )
            
            # Generate stopping recommendation
            stopping_recommendation = self._generate_stopping_recommendation(
                config, statistical_result, causal_result, performance_metrics
            )
            
            # Generate business decision framework
            business_decision = self._generate_business_decision(
                config, statistical_result, causal_result, performance_metrics
            )
            
            # Calculate quality scores
            quality_scores = self._calculate_quality_scores(
                experiment_data, statistical_result, pattern_result, causal_result
            )
            
            # Generate actionable insights
            insights = self._generate_actionable_insights(
                config, statistical_result, pattern_result, causal_result, performance_metrics
            )
            
            # Calculate analysis duration
            analysis_duration = (datetime.utcnow() - analysis_start).total_seconds()
            
            # Create comprehensive result
            result = ExperimentResult(
                experiment_id=experiment_id,
                analysis_id=analysis_id,
                timestamp=datetime.utcnow(),
                experiment_status=self._determine_experiment_status(config, stopping_recommendation),
                statistical_validation=statistical_result,
                pattern_significance=pattern_result,
                causal_inference=causal_result,
                arm_performance=performance_metrics['arm_performance'],
                relative_performance=performance_metrics['relative_performance'],
                stopping_recommendation=stopping_recommendation['recommendation'],
                business_decision=business_decision['decision'],
                confidence_level=stopping_recommendation['confidence'],
                risk_assessment=business_decision['risk_assessment'],
                data_quality_score=quality_scores['data_quality'],
                analysis_quality_score=quality_scores['analysis_quality'],
                overall_experiment_quality=quality_scores['overall_quality'],
                actionable_insights=insights['actionable_insights'],
                next_steps=insights['next_steps'],
                lessons_learned=insights['lessons_learned'],
                analysis_duration_seconds=analysis_duration,
                sample_sizes=experiment_data['sample_sizes'],
                experiment_duration_days=experiment_data['duration_days']
            )
            
            # Store analysis result
            await self._store_analysis_result(result)
            
            logger.info(f"Experiment analysis completed: {experiment_id}")
            logger.info(f"Recommendation: {stopping_recommendation['recommendation']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing experiment {experiment_id}: {e}")
            raise
    
    async def stop_experiment(self, experiment_id: str, reason: str = "manual") -> Dict[str, Any]:
        """
        Stop an active experiment
        
        Args:
            experiment_id: Experiment identifier
            reason: Reason for stopping
            
        Returns:
            Stop operation result
        """
        try:
            logger.info(f"Stopping experiment {experiment_id}, reason: {reason}")
            
            # Perform final analysis
            final_analysis = await self.analyze_experiment(experiment_id, force_analysis=True)
            
            # Update experiment status in database
            await self._update_experiment_status(experiment_id, ExperimentStatus.COMPLETED)
            
            # Stop real-time monitoring
            if self.real_time_service:
                await self.real_time_service.stop_experiment_monitoring(experiment_id)
            
            # Cancel monitoring task
            if experiment_id in self.experiment_tasks:
                self.experiment_tasks[experiment_id].cancel()
                del self.experiment_tasks[experiment_id]
            
            # Remove from active experiments
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]
            
            logger.info(f"Experiment stopped successfully: {experiment_id}")
            
            return {
                'success': True,
                'experiment_id': experiment_id,
                'stop_reason': reason,
                'final_analysis': final_analysis,
                'status': ExperimentStatus.COMPLETED.value
            }
            
        except Exception as e:
            logger.error(f"Error stopping experiment {experiment_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'experiment_id': experiment_id
            }
    
    async def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get current experiment status and metrics
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Current experiment status and metrics
        """
        try:
            config = self.active_experiments.get(experiment_id)
            if not config:
                return {
                    'experiment_id': experiment_id,
                    'status': 'not_found',
                    'active': False
                }
            
            # Get real-time metrics if available
            real_time_metrics = None
            if self.real_time_service:
                real_time_metrics = await self.real_time_service.get_real_time_metrics(experiment_id)
            
            # Get current data summary
            experiment_data = await self._collect_experiment_data(experiment_id, config)
            
            return {
                'experiment_id': experiment_id,
                'status': 'active',
                'active': True,
                'experiment_name': config.experiment_name,
                'experiment_type': config.experiment_type.value,
                'duration_days': experiment_data['duration_days'],
                'sample_sizes': experiment_data['sample_sizes'],
                'total_sample_size': sum(experiment_data['sample_sizes'].values()),
                'arms': [arm.arm_name for arm in config.arms],
                'real_time_metrics': real_time_metrics.__dict__ if real_time_metrics else None,
                'sufficient_data': experiment_data['sufficient_data'],
                'minimum_sample_size_reached': experiment_data['total_sample_size'] >= config.minimum_sample_size,
                'estimated_completion_date': self._estimate_completion_date(config, experiment_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting experiment status {experiment_id}: {e}")
            return {
                'experiment_id': experiment_id,
                'status': 'error',
                'error': str(e)
            }
    
    async def cleanup(self):
        """Clean up orchestrator resources"""
        try:
            logger.info("Cleaning up experiment orchestrator")
            
            # Cancel all monitoring tasks
            for task in self.experiment_tasks.values():
                task.cancel()
            
            # Wait for tasks to complete
            if self.experiment_tasks:
                await asyncio.gather(*self.experiment_tasks.values(), return_exceptions=True)
            
            # Stop real-time services
            if self.real_time_service:
                for experiment_id in list(self.active_experiments.keys()):
                    await self.real_time_service.stop_experiment_monitoring(experiment_id)
                await self.real_time_service.cleanup()
            
            self.experiment_tasks.clear()
            self.active_experiments.clear()
            
            logger.info("Experiment orchestrator cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during orchestrator cleanup: {e}")
    
    def _validate_experiment_config(self, config: ExperimentConfiguration) -> Dict[str, Any]:
        """Validate experiment configuration"""
        errors = []
        warnings = []
        
        # Basic validation
        if not config.arms or len(config.arms) < 2:
            errors.append("Experiment must have at least 2 arms")
        
        if config.minimum_sample_size < 20:
            warnings.append("Minimum sample size is very small, consider increasing")
        
        if config.effect_size_threshold < 0.05:
            warnings.append("Effect size threshold is very small, may lead to false positives")
        
        if config.statistical_power < 0.7:
            warnings.append("Statistical power is low, consider increasing sample size")
        
        # Arms validation
        total_weight = sum(arm.allocation_weight for arm in config.arms)
        if abs(total_weight - len(config.arms)) > 0.01:  # Allow for equal weights by default
            warnings.append("Arm allocation weights may not be balanced")
        
        # Stopping rules validation
        if not config.stopping_rules:
            warnings.append("No stopping rules defined, experiment may run indefinitely")
        
        if config.max_duration_days > 90:
            warnings.append("Very long experiment duration, consider shorter periods")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _calculate_sample_size_requirements(self, config: ExperimentConfiguration) -> Dict[str, Any]:
        """Calculate required sample sizes for experiment"""
        try:
            from scipy import stats
            
            # Basic power analysis for two-sample t-test
            alpha = config.significance_level
            power = config.statistical_power
            effect_size = config.effect_size_threshold
            
            # Calculate required sample size per group
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            
            n_per_group = ((z_alpha + z_beta) ** 2 * 2) / (effect_size ** 2)
            n_per_group = int(np.ceil(n_per_group))
            
            # Total sample size
            n_total = n_per_group * len(config.arms)
            
            # Adjust for multiple arms
            if len(config.arms) > 2:
                # Bonferroni correction for multiple comparisons
                corrected_alpha = alpha / (len(config.arms) - 1)
                z_alpha_corrected = stats.norm.ppf(1 - corrected_alpha/2)
                n_per_group_corrected = ((z_alpha_corrected + z_beta) ** 2 * 2) / (effect_size ** 2)
                n_per_group_corrected = int(np.ceil(n_per_group_corrected))
                n_total_corrected = n_per_group_corrected * len(config.arms)
            else:
                n_per_group_corrected = n_per_group
                n_total_corrected = n_total
            
            # Estimate duration based on traffic
            daily_traffic = config.traffic_allocation * 1000  # Assume 1000 daily users
            estimated_days = n_total_corrected / daily_traffic
            
            return {
                'required_sample_size_per_group': n_per_group,
                'required_total_sample_size': n_total,
                'corrected_sample_size_per_group': n_per_group_corrected,
                'corrected_total_sample_size': n_total_corrected,
                'estimated_duration_days': max(1, int(np.ceil(estimated_days))),
                'recommended_minimum': max(config.minimum_sample_size, n_total_corrected),
                'power_analysis': {
                    'alpha': alpha,
                    'power': power,
                    'effect_size': effect_size,
                    'n_arms': len(config.arms)
                }
            }
            
        except Exception as e:
            logger.warning(f"Error calculating sample size: {e}")
            return {
                'required_sample_size_per_group': config.minimum_sample_size // len(config.arms),
                'required_total_sample_size': config.minimum_sample_size,
                'estimated_duration_days': 14,
                'error': str(e)
            }
    
    async def _create_experiment_record(self, config: ExperimentConfiguration) -> ABExperiment:
        """Create experiment record in database"""
        experiment = ABExperiment(
            experiment_name=config.experiment_name,
            description=config.description,
            control_rules={"arm_id": config.arms[0].arm_id, "rules": config.arms[0].rules},
            treatment_rules={"arms": [{"arm_id": arm.arm_id, "rules": arm.rules} for arm in config.arms[1:]]},
            target_metric=config.primary_metric,
            sample_size_per_group=config.minimum_sample_size // len(config.arms),
            status="running",
            started_at=datetime.utcnow(),
            metadata={
                "experiment_type": config.experiment_type.value,
                "arms": [{"arm_id": arm.arm_id, "arm_name": arm.arm_name} for arm in config.arms],
                "stopping_rules": [rule.value for rule in config.stopping_rules],
                "analysis_config": {
                    "causal_analysis_enabled": config.causal_analysis_enabled,
                    "pattern_analysis_enabled": config.pattern_analysis_enabled,
                    "statistical_power": config.statistical_power,
                    "effect_size_threshold": config.effect_size_threshold
                }
            }
        )
        
        self.db_session.add(experiment)
        await self.db_session.commit()
        await self.db_session.refresh(experiment)
        
        return experiment
    
    async def _monitor_experiment(self, experiment_id: str):
        """Background task to monitor experiment progress"""
        try:
            logger.info(f"Starting experiment monitoring: {experiment_id}")
            
            while experiment_id in self.active_experiments:
                try:
                    config = self.active_experiments[experiment_id]
                    
                    # Check stopping criteria
                    should_stop, reason = await self._check_stopping_criteria(experiment_id, config)
                    
                    if should_stop:
                        logger.info(f"Stopping criteria met for {experiment_id}: {reason}")
                        await self.stop_experiment(experiment_id, reason)
                        break
                    
                    # Wait before next check
                    await asyncio.sleep(300)  # Check every 5 minutes
                    
                except asyncio.CancelledError:
                    logger.info(f"Monitoring cancelled for {experiment_id}")
                    break
                except Exception as e:
                    logger.error(f"Error in experiment monitoring {experiment_id}: {e}")
                    await asyncio.sleep(300)  # Continue monitoring despite errors
                    
        except Exception as e:
            logger.error(f"Fatal error in experiment monitoring {experiment_id}: {e}")
    
    async def _check_stopping_criteria(self, experiment_id: str,
                                     config: ExperimentConfiguration) -> Tuple[bool, str]:
        """Check if experiment should be stopped"""
        try:
            # Time-based stopping
            if StoppingRule.TIME_LIMIT in config.stopping_rules:
                experiment_data = await self._collect_experiment_data(experiment_id, config)
                if experiment_data['duration_days'] >= config.max_duration_days:
                    return True, "Maximum duration reached"
            
            # Sample size based stopping
            if StoppingRule.SAMPLE_SIZE_REACHED in config.stopping_rules:
                experiment_data = await self._collect_experiment_data(experiment_id, config)
                if experiment_data['total_sample_size'] >= config.maximum_sample_size:
                    return True, "Maximum sample size reached"
            
            # Statistical significance based stopping
            if (StoppingRule.STATISTICAL_SIGNIFICANCE in config.stopping_rules and 
                config.early_stopping_enabled):
                
                try:
                    analysis = await self.analyze_experiment(experiment_id)
                    if (analysis.statistical_validation.primary_test.p_value < config.significance_level and
                        analysis.statistical_validation.practical_significance):
                        return True, "Statistical and practical significance achieved"
                except Exception as e:
                    logger.warning(f"Error checking statistical significance: {e}")
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Error checking stopping criteria: {e}")
            return False, ""
    
    async def _collect_experiment_data(self, experiment_id: str,
                                     config: ExperimentConfiguration) -> Dict[str, Any]:
        """Collect experiment data for analysis"""
        try:
            from sqlalchemy import select, and_
            
            # Query experiment data from database
            query = select(RulePerformance).where(
                and_(
                    RulePerformance.created_at >= datetime.utcnow() - timedelta(days=config.max_duration_days),
                    # In practice, you'd filter by experiment participation
                )
            )
            
            result = await self.db_session.execute(query)
            performance_records = result.scalars().all()
            
            # Group data by arms (simplified - in practice, you'd have proper experiment assignment tracking)
            arms_data = {}
            sample_sizes = {}
            
            for i, arm in enumerate(config.arms):
                # Simulate arm assignment (in practice, this would be tracked properly)
                arm_records = performance_records[i::len(config.arms)]
                
                arms_data[arm.arm_id] = {
                    'outcomes': [r.improvement_score for r in arm_records],
                    'metrics': {
                        'improvement_score': [r.improvement_score for r in arm_records],
                        'execution_time_ms': [r.execution_time_ms for r in arm_records],
                        'user_satisfaction_score': [r.user_satisfaction_score for r in arm_records]
                    },
                    'sample_size': len(arm_records)
                }
                sample_sizes[arm.arm_id] = len(arm_records)
            
            total_sample_size = sum(sample_sizes.values())
            duration_days = min(config.max_duration_days, 
                               (datetime.utcnow() - datetime.utcnow().replace(hour=0, minute=0, second=0)).days + 1)
            
            return {
                'arms': arms_data,
                'sample_sizes': sample_sizes,
                'total_sample_size': total_sample_size,
                'duration_days': duration_days,
                'sufficient_data': total_sample_size >= config.minimum_sample_size,
                'primary_metric': config.primary_metric,
                'secondary_metrics': config.secondary_metrics
            }
            
        except Exception as e:
            logger.error(f"Error collecting experiment data: {e}")
            return {
                'arms': {},
                'sample_sizes': {},
                'total_sample_size': 0,
                'duration_days': 0,
                'sufficient_data': False
            }
    
    async def _perform_statistical_analysis(self, experiment_data: Dict[str, Any],
                                          config: ExperimentConfiguration) -> AdvancedValidationResult:
        """Perform statistical analysis of experiment"""
        try:
            # Get control and treatment data for primary metric
            arms = list(experiment_data['arms'].items())
            if len(arms) < 2:
                raise ValueError("Insufficient arms for statistical analysis")
            
            control_arm_id, control_data = arms[0]
            treatment_arm_id, treatment_data = arms[1]  # Use first treatment arm
            
            control_outcomes = control_data['outcomes']
            treatment_outcomes = treatment_data['outcomes']
            
            if len(control_outcomes) < 3 or len(treatment_outcomes) < 3:
                raise ValueError("Insufficient data for statistical analysis")
            
            # Perform advanced statistical validation
            result = self.statistical_validator.validate_ab_test(
                control_data=control_outcomes,
                treatment_data=treatment_outcomes,
                validate_assumptions=True,
                include_bootstrap=True,
                include_sensitivity=True
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            raise
    
    async def _perform_pattern_analysis(self, experiment_data: Dict[str, Any],
                                      config: ExperimentConfiguration) -> PatternSignificanceReport:
        """Perform pattern significance analysis"""
        try:
            # Prepare pattern data (simplified example)
            patterns_data = {}
            control_patterns = {}
            treatment_patterns = {}
            
            # Extract patterns from experiment data
            for arm_id, arm_data in experiment_data['arms'].items():
                if not arm_data['outcomes']:
                    continue
                
                # Create categorical patterns based on performance levels
                outcomes = np.array(arm_data['outcomes'])
                
                pattern_id = f"performance_distribution_{arm_id}"
                patterns_data[pattern_id] = {
                    'type': 'performance_distribution',
                    'description': f'Performance distribution for {arm_id}'
                }
                
                # Categorize outcomes
                high_perf = np.sum(outcomes > np.percentile(outcomes, 75))
                med_perf = np.sum((outcomes >= np.percentile(outcomes, 25)) & 
                                 (outcomes <= np.percentile(outcomes, 75)))
                low_perf = np.sum(outcomes < np.percentile(outcomes, 25))
                
                if arm_id == list(experiment_data['arms'].keys())[0]:  # Control
                    control_patterns[pattern_id] = {
                        'high': high_perf, 'medium': med_perf, 'low': low_perf
                    }
                else:  # Treatment
                    treatment_patterns[pattern_id] = {
                        'high': high_perf, 'medium': med_perf, 'low': low_perf
                    }
            
            if not patterns_data:
                return None
            
            # Analyze pattern significance
            result = self.pattern_analyzer.analyze_pattern_significance(
                patterns_data=patterns_data,
                control_data=control_patterns,
                treatment_data=treatment_patterns
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return None
    
    async def _perform_causal_analysis(self, experiment_data: Dict[str, Any],
                                     config: ExperimentConfiguration) -> CausalInferenceResult:
        """Perform causal inference analysis"""
        try:
            # Prepare data for causal analysis
            arms = list(experiment_data['arms'].items())
            if len(arms) < 2:
                return None
            
            # Combine all data with treatment indicators
            all_outcomes = []
            all_treatments = []
            
            for i, (arm_id, arm_data) in enumerate(arms):
                outcomes = arm_data['outcomes']
                treatments = [1 if i > 0 else 0] * len(outcomes)  # Control = 0, Treatments = 1
                
                all_outcomes.extend(outcomes)
                all_treatments.extend(treatments)
            
            if len(all_outcomes) < 20:
                return None
            
            # Perform causal analysis
            result = self.causal_analyzer.analyze_causal_effect(
                outcome_data=np.array(all_outcomes),
                treatment_data=np.array(all_treatments),
                assignment_mechanism=TreatmentAssignment.RANDOMIZED,
                method=CausalMethod.DIFFERENCE_IN_DIFFERENCES
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in causal analysis: {e}")
            return None
    
    def _calculate_performance_metrics(self, experiment_data: Dict[str, Any],
                                     statistical_result: AdvancedValidationResult) -> Dict[str, Any]:
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
                
                arm_performance[arm_id] = {
                    'mean': float(arm_mean),
                    'std': float(arm_std),
                    'sample_size': len(outcomes),
                    'confidence_interval': self._calculate_ci(outcomes) if len(outcomes) > 1 else (arm_mean, arm_mean)
                }
                
                # Relative performance vs control
                if control_mean > 0:
                    relative_lift = (arm_mean - control_mean) / control_mean
                    relative_performance[arm_id] = {
                        'absolute_difference': float(arm_mean - control_mean),
                        'relative_lift': float(relative_lift),
                        'relative_lift_percent': float(relative_lift * 100)
                    }
                else:
                    relative_performance[arm_id] = {
                        'absolute_difference': float(arm_mean - control_mean),
                        'relative_lift': 0.0,
                        'relative_lift_percent': 0.0
                    }
            
        except Exception as e:
            logger.warning(f"Error calculating performance metrics: {e}")
        
        return {
            'arm_performance': arm_performance,
            'relative_performance': relative_performance
        }
    
    def _calculate_ci(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for data"""
        try:
            from scipy import stats
            
            data_array = np.array(data)
            mean = np.mean(data_array)
            se = stats.sem(data_array)
            h = se * stats.t.ppf((1 + confidence) / 2., len(data_array) - 1)
            
            return (float(mean - h), float(mean + h))
            
        except Exception:
            mean = np.mean(data)
            return (mean, mean)
    
    def _generate_stopping_recommendation(self, config: ExperimentConfiguration,
                                        statistical_result: AdvancedValidationResult,
                                        causal_result: Optional[CausalInferenceResult],
                                        performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate stopping recommendation"""
        try:
            # Primary criteria
            statistically_significant = statistical_result.primary_test.p_value < config.significance_level
            practically_significant = statistical_result.practical_significance
            
            # Quality criteria
            validation_quality = statistical_result.validation_quality_score
            causal_quality = causal_result.overall_quality_score if causal_result else 0.5
            
            # Overall confidence
            confidence = (validation_quality + causal_quality + 
                         (0.3 if statistically_significant else 0) + 
                         (0.2 if practically_significant else 0)) / 2
            
            # Generate recommendation
            if statistically_significant and practically_significant and validation_quality > 0.7:
                recommendation = "STOP_FOR_SUCCESS"
                reason = "Strong evidence for treatment effect with high confidence"
            elif statistically_significant and practically_significant:
                recommendation = "STOP_WITH_CAUTION"
                reason = "Treatment effect detected but some quality concerns"
            elif validation_quality > 0.8 and not statistically_significant:
                recommendation = "STOP_FOR_FUTILITY"
                reason = "High quality analysis shows no meaningful effect"
            else:
                recommendation = "CONTINUE"
                reason = "Insufficient evidence to make decision"
            
            return {
                'recommendation': recommendation,
                'reason': reason,
                'confidence': float(min(max(confidence, 0.0), 1.0)),
                'criteria_met': {
                    'statistical_significance': statistically_significant,
                    'practical_significance': practically_significant,
                    'validation_quality': validation_quality > 0.7,
                    'causal_quality': causal_quality > 0.7 if causal_result else True
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating stopping recommendation: {e}")
            return {
                'recommendation': 'CONTINUE',
                'reason': 'Error in analysis',
                'confidence': 0.0
            }
    
    def _generate_business_decision(self, config: ExperimentConfiguration,
                                  statistical_result: AdvancedValidationResult,
                                  causal_result: Optional[CausalInferenceResult],
                                  performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate business decision framework"""
        try:
            # Extract key metrics
            effect_size = statistical_result.primary_test.effect_size
            p_value = statistical_result.primary_test.p_value
            
            # Risk assessment
            risk_factors = []
            if p_value > 0.01:
                risk_factors.append("Moderate statistical evidence")
            if abs(effect_size) < 0.2:
                risk_factors.append("Small effect size")
            if statistical_result.validation_quality_score < 0.7:
                risk_factors.append("Quality concerns in analysis")
            
            # Business decision
            if statistical_result.practical_significance and len(risk_factors) <= 1:
                decision = "IMPLEMENT"
                rationale = "Strong evidence supports implementation"
            elif statistical_result.practical_significance:
                decision = "PILOT"
                rationale = "Consider gradual rollout with monitoring"
            else:
                decision = "NO_ACTION"
                rationale = "Insufficient evidence for business impact"
            
            return {
                'decision': decision,
                'rationale': rationale,
                'risk_assessment': {
                    'risk_level': 'low' if len(risk_factors) <= 1 else 'medium' if len(risk_factors) <= 2 else 'high',
                    'risk_factors': risk_factors,
                    'mitigation_strategies': [
                        "Monitor key metrics closely",
                        "Set up early warning systems",
                        "Plan rollback procedures"
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating business decision: {e}")
            return {
                'decision': 'NO_ACTION',
                'rationale': 'Error in analysis'
            }
    
    def _calculate_quality_scores(self, experiment_data: Dict[str, Any],
                                statistical_result: AdvancedValidationResult,
                                pattern_result: Optional[PatternSignificanceReport],
                                causal_result: Optional[CausalInferenceResult]) -> Dict[str, float]:
        """Calculate overall quality scores"""
        try:
            # Data quality score
            data_quality_components = []
            
            # Sample size adequacy
            total_sample = experiment_data['total_sample_size']
            sample_score = min(total_sample / 1000, 1.0)
            data_quality_components.append(('sample_size', sample_score, 0.4))
            
            # Balance between arms
            sample_sizes = list(experiment_data['sample_sizes'].values())
            if sample_sizes:
                balance_score = min(sample_sizes) / max(sample_sizes)
                data_quality_components.append(('balance', balance_score, 0.3))
            
            # Data completeness
            completeness_score = 1.0 if experiment_data['sufficient_data'] else 0.5
            data_quality_components.append(('completeness', completeness_score, 0.3))
            
            data_quality = sum(score * weight for _, score, weight in data_quality_components)
            
            # Analysis quality score
            analysis_quality = statistical_result.validation_quality_score
            
            # Overall quality
            quality_components = [
                ('data_quality', data_quality, 0.4),
                ('statistical_analysis', analysis_quality, 0.4)
            ]
            
            if pattern_result:
                quality_components.append(('pattern_analysis', pattern_result.quality_score, 0.1))
            
            if causal_result:
                quality_components.append(('causal_analysis', causal_result.overall_quality_score, 0.1))
            
            # Normalize weights
            total_weight = sum(weight for _, _, weight in quality_components)
            overall_quality = sum(score * weight / total_weight for _, score, weight in quality_components)
            
            return {
                'data_quality': float(min(max(data_quality, 0.0), 1.0)),
                'analysis_quality': float(min(max(analysis_quality, 0.0), 1.0)),
                'overall_quality': float(min(max(overall_quality, 0.0), 1.0))
            }
            
        except Exception as e:
            logger.error(f"Error calculating quality scores: {e}")
            return {
                'data_quality': 0.5,
                'analysis_quality': 0.5,
                'overall_quality': 0.5
            }
    
    def _generate_actionable_insights(self, config: ExperimentConfiguration,
                                    statistical_result: AdvancedValidationResult,
                                    pattern_result: Optional[PatternSignificanceReport],
                                    causal_result: Optional[CausalInferenceResult],
                                    performance_metrics: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate actionable insights and recommendations"""
        actionable_insights = []
        next_steps = []
        lessons_learned = []
        
        try:
            # Statistical insights
            if statistical_result.practical_significance:
                effect_size = statistical_result.primary_test.effect_size
                actionable_insights.append(
                    f"Treatment shows {statistical_result.effect_size_magnitude.value} effect size ({effect_size:.3f})"
                )
                
                if statistical_result.primary_test.p_value < 0.001:
                    actionable_insights.append("Very strong statistical evidence (p < 0.001)")
                
            # Pattern insights
            if pattern_result and pattern_result.significant_patterns:
                actionable_insights.append(
                    f"Detected {len(pattern_result.significant_patterns)} significant behavioral patterns"
                )
                
                for insight in pattern_result.business_insights[:3]:
                    actionable_insights.append(f"Pattern insight: {insight}")
            
            # Causal insights
            if causal_result and causal_result.causal_interpretation:
                actionable_insights.append(f"Causal analysis: {causal_result.causal_interpretation}")
                
                for rec in causal_result.business_recommendations[:2]:
                    actionable_insights.append(rec)
            
            # Next steps
            if statistical_result.practical_significance:
                next_steps.extend([
                    "Plan implementation rollout strategy",
                    "Set up monitoring for key metrics",
                    "Prepare stakeholder communication"
                ])
            else:
                next_steps.extend([
                    "Investigate why treatment didn't show effect",
                    "Consider alternative approaches",
                    "Plan follow-up experiments"
                ])
            
            # Lessons learned
            if statistical_result.validation_quality_score > 0.8:
                lessons_learned.append("High-quality experimental design enabled reliable conclusions")
            
            if config.causal_analysis_enabled and causal_result:
                lessons_learned.append("Causal analysis provided deeper understanding beyond correlation")
            
            if pattern_result and pattern_result.pattern_interactions:
                lessons_learned.append("Pattern analysis revealed interaction effects worth investigating")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            actionable_insights.append("Error generating insights - manual review recommended")
        
        return {
            'actionable_insights': actionable_insights,
            'next_steps': next_steps,
            'lessons_learned': lessons_learned
        }
    
    def _determine_experiment_status(self, config: ExperimentConfiguration,
                                   stopping_recommendation: Dict[str, Any]) -> ExperimentStatus:
        """Determine experiment status based on recommendation"""
        recommendation = stopping_recommendation.get('recommendation', 'CONTINUE')
        
        if recommendation in ['STOP_FOR_SUCCESS', 'STOP_WITH_CAUTION', 'STOP_FOR_FUTILITY']:
            return ExperimentStatus.ANALYZING
        else:
            return ExperimentStatus.ACTIVE
    
    async def _store_analysis_result(self, result: ExperimentResult):
        """Store analysis result in database"""
        try:
            # In practice, you'd store the comprehensive result in a dedicated table
            # For now, we'll update the experiment record
            
            from sqlalchemy import select, update
            
            # Find experiment record
            query = select(ABExperiment).where(ABExperiment.experiment_name == result.experiment_id)
            db_result = await self.db_session.execute(query)
            experiment = db_result.scalar_one_or_none()
            
            if experiment:
                # Update with analysis results
                update_query = update(ABExperiment).where(
                    ABExperiment.experiment_id == experiment.experiment_id
                ).values(
                    analysis_results={
                        'analysis_id': result.analysis_id,
                        'timestamp': result.timestamp.isoformat(),
                        'statistical_significance': result.statistical_validation.primary_test.p_value < 0.05,
                        'effect_size': result.statistical_validation.primary_test.effect_size,
                        'business_decision': result.business_decision,
                        'confidence_level': result.confidence_level,
                        'quality_score': result.overall_experiment_quality
                    },
                    completed_at=datetime.utcnow() if result.experiment_status == ExperimentStatus.COMPLETED else None
                )
                
                await self.db_session.execute(update_query)
                await self.db_session.commit()
                
        except Exception as e:
            logger.error(f"Error storing analysis result: {e}")
    
    async def _update_experiment_status(self, experiment_id: str, status: ExperimentStatus):
        """Update experiment status in database"""
        try:
            from sqlalchemy import select, update
            
            query = select(ABExperiment).where(ABExperiment.experiment_name == experiment_id)
            result = await self.db_session.execute(query)
            experiment = result.scalar_one_or_none()
            
            if experiment:
                update_query = update(ABExperiment).where(
                    ABExperiment.experiment_id == experiment.experiment_id
                ).values(
                    status=status.value,
                    completed_at=datetime.utcnow() if status == ExperimentStatus.COMPLETED else None
                )
                
                await self.db_session.execute(update_query)
                await self.db_session.commit()
                
        except Exception as e:
            logger.error(f"Error updating experiment status: {e}")
    
    async def _load_experiment_config(self, experiment_id: str) -> Optional[ExperimentConfiguration]:
        """Load experiment configuration from database"""
        # This would load a stored configuration from the database
        # For now, return None as we don't have persistent config storage implemented
        return None
    
    def _create_insufficient_data_result(self, experiment_id: str, analysis_id: str,
                                       experiment_data: Dict[str, Any]) -> ExperimentResult:
        """Create result for insufficient data scenario"""
        # Create minimal statistical result for insufficient data
        from .advanced_statistical_validator import StatisticalTestResult, EffectSizeMagnitude
        
        placeholder_test = StatisticalTestResult(
            test_name="Insufficient Data",
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            effect_size_type="None"
        )
        
        placeholder_validation = AdvancedValidationResult(
            validation_id=analysis_id,
            timestamp=datetime.utcnow(),
            primary_test=placeholder_test,
            effect_size_magnitude=EffectSizeMagnitude.NEGLIGIBLE,
            practical_significance=False,
            clinical_significance=False,
            validation_quality_score=0.0
        )
        
        return ExperimentResult(
            experiment_id=experiment_id,
            analysis_id=analysis_id,
            timestamp=datetime.utcnow(),
            experiment_status=ExperimentStatus.ACTIVE,
            statistical_validation=placeholder_validation,
            stopping_recommendation="CONTINUE - Insufficient data",
            business_decision="WAIT - Collect more data",
            confidence_level=0.0,
            data_quality_score=0.3,
            analysis_quality_score=0.0,
            overall_experiment_quality=0.2,
            actionable_insights=["Insufficient data for analysis", "Continue data collection"],
            next_steps=["Wait for more participants", "Check data collection issues"],
            sample_sizes=experiment_data['sample_sizes'],
            experiment_duration_days=experiment_data['duration_days']
        )
    
    def _estimate_completion_date(self, config: ExperimentConfiguration,
                                experiment_data: Dict[str, Any]) -> Optional[str]:
        """Estimate experiment completion date"""
        try:
            current_sample = experiment_data['total_sample_size']
            target_sample = config.minimum_sample_size
            
            if current_sample >= target_sample:
                return "Ready for analysis"
            
            # Estimate based on current rate
            duration_days = experiment_data['duration_days']
            if duration_days > 0:
                daily_rate = current_sample / duration_days
                if daily_rate > 0:
                    days_remaining = (target_sample - current_sample) / daily_rate
                    completion_date = datetime.utcnow() + timedelta(days=days_remaining)
                    return completion_date.strftime('%Y-%m-%d')
            
            return "Unable to estimate"
            
        except Exception as e:
            logger.warning(f"Error estimating completion date: {e}")
            return "Unknown"


# Utility functions for external use
async def quick_experiment_setup(experiment_name: str,
                               control_rules: Dict[str, Any],
                               treatment_rules: Dict[str, Any],
                               db_session: AsyncSession,
                               **kwargs) -> Dict[str, Any]:
    """Quick experiment setup for immediate use"""
    try:
        # Create simple A/B test configuration
        control_arm = ExperimentArm(
            arm_id="control",
            arm_name="Control",
            description="Control arm",
            rules=control_rules
        )
        
        treatment_arm = ExperimentArm(
            arm_id="treatment",
            arm_name="Treatment", 
            description="Treatment arm",
            rules=treatment_rules
        )
        
        config = ExperimentConfiguration(
            experiment_id=experiment_name,
            experiment_name=experiment_name,
            experiment_type=ExperimentType.SIMPLE_AB,
            description=f"A/B test: {experiment_name}",
            arms=[control_arm, treatment_arm],
            **kwargs
        )
        
        # Set up experiment
        orchestrator = ExperimentOrchestrator(db_session)
        result = await orchestrator.setup_experiment(config)
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'experiment_id': experiment_name
        }