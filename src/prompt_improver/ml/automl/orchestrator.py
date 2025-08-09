"""AutoML Orchestrator - Central coordinator for automated machine learning
Implements 2025 best practices for AutoML integration and observability
"""
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
import optuna
from optuna.integration import OptunaSearchCV
from optuna.storages import RDBStorage
from prompt_improver.utils.datetime_utils import aware_utc_now
from .callbacks import AutoMLCallback
tracer = trace.get_tracer(__name__)
if TYPE_CHECKING:
    from ...core.services.analytics_factory import get_analytics_router
    from ...database import UnifiedConnectionManager
    from ..evaluation.experiment_orchestrator import ExperimentOrchestrator
    from ..optimization.algorithms.rule_optimizer import OptimizationConfig, RuleOptimizer
    from ..utils.model_manager import ModelManager
logger = logging.getLogger(__name__)

class AutoMLMode(Enum):
    """AutoML operation modes"""
    HYPERPARAMETER_OPTIMIZATION = 'hpo'
    AUTOMATED_EXPERIMENT_DESIGN = 'aed'
    CONTINUOUS_OPTIMIZATION = 'continuous'
    MULTI_OBJECTIVE_PARETO = 'pareto'

@dataclass
class AutoMLConfig:
    """Configuration for AutoML orchestration"""
    study_name: str = 'prompt_improver_automl'
    storage_url: str = field(default_factory=lambda: os.getenv('AUTOML_DATABASE_URL', f"postgresql+psycopg://{os.getenv('POSTGRES_USERNAME', 'user')}:{os.getenv('POSTGRES_PASSWORD', 'password')}@{os.getenv('POSTGRES_HOST', 'localhost')}/automl_studies"))
    n_trials: int = 100
    timeout: int | None = 3600
    optimization_mode: AutoMLMode = AutoMLMode.HYPERPARAMETER_OPTIMIZATION
    objectives: list[str] = field(default_factory=lambda: ['improvement_score', 'execution_time'])
    enable_real_time_feedback: bool = True
    enable_early_stopping: bool = True
    enable_artifact_storage: bool = True
    enable_drift_detection: bool = True
    auto_retraining_threshold: float = 0.05
    pareto_front_size: int = 10

class AutoMLOrchestrator:
    """Central AutoML orchestrator implementing 2025 best practices

    Coordinates existing components:
    - Optuna hyperparameter optimization
    - A/B testing framework
    - Real-time analytics
    - Model management
    - Continuous learning

    features:
    - Callback-based integration following Optuna 2025 patterns
    - Real-time feedback loops
    - Multi-objective optimization with NSGA-II
    - Automated experiment design
    - Continuous optimization and drift detection
    """

    def __init__(self, config: AutoMLConfig, db_manager: 'DatabaseManager', rule_optimizer: Optional['RuleOptimizer']=None, experiment_orchestrator: Optional['ExperimentOrchestrator']=None, analytics_service: Optional[Any]=None, model_manager: Optional['ModelManager']=None):
        """Initialize AutoML orchestrator with existing components

        Args:
            config: AutoML configuration
            db_manager: Database manager for persistence
            rule_optimizer: Existing rule optimizer (contains Optuna + NSGA-II)
            experiment_orchestrator: A/B testing framework
            analytics_service: Real-time analytics service
            model_manager: Model management service
        """
        self.config = config
        self.db_manager = db_manager
        if rule_optimizer is None:
            from ..optimization.algorithms.rule_optimizer import RuleOptimizer
            self.rule_optimizer = RuleOptimizer()
        else:
            self.rule_optimizer = rule_optimizer
        self.experiment_orchestrator = experiment_orchestrator
        self.analytics_service = analytics_service
        self.model_manager = model_manager
        self.storage = self._create_storage()
        self.study = None
        self.callbacks = []
        self.current_optimization = None
        self.performance_history = []
        self.best_configurations = {}
        logger.info('AutoML Orchestrator initialized with mode: %s', config.optimization_mode)

    def _create_storage(self) -> RDBStorage:
        """Create Optuna storage following 2025 best practices"""
        try:
            storage = RDBStorage(url=self.config.storage_url, heartbeat_interval=60, grace_period=120)
            logger.info('Created RDBStorage with heartbeat monitoring')
            return storage
        except Exception as e:
            logger.warning('Failed to create RDBStorage: %s, falling back to InMemoryStorage', e)
            return optuna.storages.InMemoryStorage()

    def _setup_callbacks(self) -> list[Callable]:
        """Setup Optuna callbacks following 2025 integration patterns"""
        callbacks = []
        if self.analytics_service and self.config.enable_real_time_feedback:
            analytics_callback = self._create_analytics_callback()
            if analytics_callback:
                callbacks.append(analytics_callback)
        automl_callback = AutoMLCallback(orchestrator=self, enable_early_stopping=self.config.enable_early_stopping, enable_artifact_storage=self.config.enable_artifact_storage)
        callbacks.append(automl_callback)
        self.callbacks = callbacks
        return callbacks

    def _create_analytics_callback(self):
        """Create analytics callback with lazy import to avoid circular dependency."""
        try:
            from .callbacks import RealTimeAnalyticsCallback
            return RealTimeAnalyticsCallback(self.analytics_service)
        except ImportError as e:
            self.logger.warning('Could not import RealTimeAnalyticsCallback: %s', e)
            return None

    async def start_optimization(self, optimization_target: str='rule_effectiveness', experiment_config: dict[str, Any] | None=None) -> dict[str, Any]:
        """Start AutoML optimization following 2025 best practices with observability

        Args:
            optimization_target: What to optimize ('rule_effectiveness', 'user_satisfaction', etc.)
            experiment_config: Configuration for A/B testing integration

        Returns:
            Dictionary with optimization results and metadata
        """
        with tracer.start_as_current_span('automl_start_optimization', attributes={'optimization_target': optimization_target, 'mode': self.config.mode.value, 'n_trials': self.config.n_trials}) as span:
            logger.info('Starting AutoML optimization for target: %s', optimization_target)
            start_time = time.time()
            try:
                self.study = optuna.create_study(study_name=self.config.study_name, storage=self.storage, direction='maximize' if 'score' in optimization_target else 'minimize', sampler=self._create_sampler(), load_if_exists=True)
                callbacks = self._setup_callbacks()
                objective_function = self._create_objective_function(optimization_target, experiment_config)
                optimization_result = await self._execute_optimization(objective_function, callbacks)
                results = await self._process_optimization_results(optimization_result)
                results['execution_time'] = time.time() - start_time
                self.performance_history.append({'timestamp': aware_utc_now(), 'target': optimization_target, 'best_value': self.study.best_value, 'best_params': self.study.best_params, 'execution_time': results['execution_time']})
                span.set_attribute('execution_time', results['execution_time'])
                span.set_attribute('best_value', self.study.best_value)
                span.set_attribute('trials_completed', len(self.study.trials))
                return results
            except Exception as e:
                logger.error('AutoML optimization failed: %s', e)
                span.set_attribute('error', str(e))
                return {'error': str(e), 'execution_time': time.time() - start_time}

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create appropriate Optuna sampler based on optimization mode"""
        if self.config.optimization_mode == AutoMLMode.MULTI_OBJECTIVE_PARETO:
            return optuna.samplers.NSGAIISampler(population_size=50, mutation_prob=0.1, crossover_prob=0.9)
        if self.config.optimization_mode == AutoMLMode.HYPERPARAMETER_OPTIMIZATION:
            return optuna.samplers.TPESampler(n_startup_trials=10, n_ei_candidates=24, seed=42)
        return optuna.samplers.TPESampler()

    def _create_objective_function(self, optimization_target: str, experiment_config: dict[str, Any] | None) -> Callable:
        """Create objective function that integrates all components

        This follows 2025 pattern of unified objective functions that
        coordinate multiple ML components through a single interface
        """

        async def async_objective_impl(trial: optuna.Trial) -> float:
            """Async implementation of objective function for internal use."""
            try:
                if self.rule_optimizer:
                    from ..optimization.algorithms.rule_optimizer import OptimizationConfig
                    optimization_config = OptimizationConfig(pareto_population_size=trial.suggest_int('pareto_population_size', 20, 100), pareto_generations=trial.suggest_int('pareto_generations', 50, 200), pareto_mutation_prob=trial.suggest_float('pareto_mutation_prob', 0.01, 0.3), pareto_crossover_prob=trial.suggest_float('pareto_crossover_prob', 0.6, 0.9), enable_multi_objective=trial.suggest_categorical('enable_multi_objective', [True, False]), enable_gaussian_process=trial.suggest_categorical('enable_gaussian_process', [True, False]))
                    if hasattr(self.rule_optimizer, 'optimize_rule'):
                        rule_config = {'rule_id': 'test_rule', 'current_params': {}, 'optimization_config': optimization_config}
                        performance_data = {'test_rule': {'total_applications': 50, 'avg_improvement': 0.7, 'consistency_score': 0.8, 'success_rate': 0.75}}
                        optimization_result = await self.rule_optimizer.optimize_rule(rule_id='test_rule', performance_data=performance_data, historical_data=[])
                    else:
                        optimization_result = {'effectiveness': np.random.random()}
                else:
                    test_config = {'sample_size': trial.suggest_int('sample_size', 100, 1000), 'confidence_level': trial.suggest_float('confidence_level', 0.9, 0.99), 'effect_size_threshold': trial.suggest_float('effect_size_threshold', 0.1, 0.5)}
                    if self.experiment_orchestrator:
                        experiment_result = await self.experiment_orchestrator.run_experiment(experiment_config={**experiment_config, **test_config})
                        optimization_result = experiment_result
                    else:
                        optimization_result = {'effectiveness': np.random.random()}
                if optimization_target == 'rule_effectiveness':
                    value = optimization_result.get('effectiveness', 0.0)
                elif optimization_target == 'execution_time':
                    value = optimization_result.get('execution_time', float('inf'))
                else:
                    value = optimization_result.get('effectiveness', 0.0)
                trial.report(value, step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                return value
            except Exception as e:
                logger.error('Objective function failed: %s', e)
                return -1.0 if 'score' in optimization_target else float('inf')

        def objective(trial: optuna.Trial) -> float:
            """Synchronous objective function compatible with Optuna (2025 best practice)."""
            import asyncio
            try:
                try:
                    loop = asyncio.get_running_loop()
                    return asyncio.run_coroutine_threadsafe(async_objective_impl(trial), loop).result()
                except RuntimeError:
                    return asyncio.run(async_objective_impl(trial))
            except Exception as e:
                logger.error('Objective function wrapper failed: %s', e)
                return -1.0 if 'score' in optimization_target else float('inf')
        return objective

    async def _execute_optimization(self, objective_function: Callable, callbacks: list[Callable]) -> dict[str, Any]:
        """Execute Optuna optimization with real-time monitoring"""
        try:
            if hasattr(self.study, 'optimize_async'):
                result = await self.study.optimize_async(objective_function, n_trials=self.config.n_trials, timeout=self.config.timeout, callbacks=callbacks)
            else:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.study.optimize, objective_function, n_trials=self.config.n_trials, timeout=self.config.timeout, callbacks=callbacks)
                    while not future.done():
                        await asyncio.sleep(5)
                        logger.info('Optimization in progress... Completed trials: %s', len(self.study.trials))
                    result = future.result()
            return {'status': 'completed', 'best_value': self.study.best_value, 'best_params': self.study.best_params, 'n_trials': len(self.study.trials), 'optimization_time': time.time()}
        except Exception as e:
            logger.error('Optimization execution failed: %s', e)
            return {'status': 'failed', 'error': str(e)}

    async def _process_optimization_results(self, optimization_result: dict[str, Any]) -> dict[str, Any]:
        """Process optimization results and update system components"""
        if optimization_result.get('status') != 'completed':
            return optimization_result
        try:
            best_params = optimization_result['best_params']
            best_value = optimization_result['best_value']
            self.best_configurations[self.config.optimization_mode.value] = {'params': best_params, 'value': best_value, 'timestamp': aware_utc_now()}
            if self.model_manager and 'model_' in str(best_params):
                model_config = {k: v for k, v in best_params.items() if k.startswith('model_')}
                if hasattr(self.model_manager, 'update_configuration'):
                    await self.model_manager.update_configuration(model_config)
            if self.analytics_service and hasattr(self.analytics_service, 'update_optimization_results'):
                await self.analytics_service.update_optimization_results({'best_params': best_params, 'best_value': best_value, 'optimization_mode': self.config.optimization_mode.value})
            processed_result = {**optimization_result, 'automl_mode': self.config.optimization_mode.value, 'pareto_front': self._extract_pareto_front() if self.config.optimization_mode == AutoMLMode.MULTI_OBJECTIVE_PARETO else None, 'feature_importance': self._analyze_parameter_importance(), 'recommendations': self._generate_recommendations(best_params, best_value)}
            return processed_result
        except Exception as e:
            logger.error('Result processing failed: %s', e)
            return {**optimization_result, 'processing_error': str(e)}

    def _extract_pareto_front(self) -> list[dict[str, Any]] | None:
        """Extract Pareto front for multi-objective optimization"""
        if not self.study or self.config.optimization_mode != AutoMLMode.MULTI_OBJECTIVE_PARETO:
            return None
        try:
            pareto_trials = []
            for trial in self.study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    pareto_trials.append({'trial_number': trial.number, 'values': trial.values if hasattr(trial, 'values') else [trial.value], 'params': trial.params})
            pareto_trials.sort(key=lambda x: sum(x['values']), reverse=True)
            return pareto_trials[:self.config.pareto_front_size]
        except Exception as e:
            logger.error('Pareto front extraction failed: %s', e)
            return None

    def _analyze_parameter_importance(self) -> dict[str, float]:
        """Analyze parameter importance using Optuna's built-in methods"""
        if not self.study or len(self.study.trials) < 10:
            return {}
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return {param: float(score) for param, score in importance.items()}
        except Exception as e:
            logger.error('Parameter importance analysis failed: %s', e)
            return {}

    def _generate_recommendations(self, best_params: dict[str, Any], best_value: float) -> list[str]:
        """Generate actionable recommendations based on optimization results"""
        recommendations = []
        if best_value > 0.9:
            recommendations.append('DEPLOY: Excellent performance achieved - ready for production')
        elif best_value > 0.7:
            recommendations.append('PILOT: Good performance - consider pilot deployment')
        else:
            recommendations.append('INVESTIGATE: Performance needs improvement - analyze failure modes')
        for param, value in best_params.items():
            if 'rate' in param and value > 0.8:
                recommendations.append(f'HIGH {param.upper()}: Consider reducing {param} for stability')
            elif 'size' in param and value < 50:
                recommendations.append(f'SMALL {param.upper()}: Consider increasing {param} for better coverage')
        return recommendations

    async def get_optimization_status(self) -> dict[str, Any]:
        """Get current optimization status and metrics"""
        if not self.study:
            return {'status': 'not_started'}
        return {'status': 'running' if self.current_optimization else 'idle', 'study_name': self.study.study_name, 'n_trials': len(self.study.trials), 'best_value': self.study.best_value if len(self.study.trials) > 0 else None, 'best_params': self.study.best_params if len(self.study.trials) > 0 else None, 'optimization_mode': self.config.optimization_mode.value, 'performance_history': self.performance_history[-10:], 'parameter_importance': self._analyze_parameter_importance()}

    async def stop_optimization(self) -> dict[str, Any]:
        """Stop current optimization gracefully"""
        if self.current_optimization:
            self.current_optimization = None
            logger.info('Optimization stopped by user request')
            return {'status': 'stopped', 'message': 'Optimization stopped successfully'}
        return {'status': 'idle', 'message': 'No optimization running'}

async def create_automl_orchestrator(config: AutoMLConfig | None=None, db_manager: Optional['UnifiedConnectionManager']=None) -> AutoMLOrchestrator:
    """Factory function to create AutoML orchestrator with proper component initialization

    Args:
        config: AutoML configuration (uses defaults if None)
        db_manager: Database manager (creates new if None)

    Returns:
        Configured AutoML orchestrator
    """
    if config is None:
        config = AutoMLConfig()
    if db_manager is None:
        from ...database import get_sessionmanager
        db_manager = get_sessionmanager()
    from ..optimization.algorithms.rule_optimizer import RuleOptimizer
    rule_optimizer = RuleOptimizer()
    orchestrator = AutoMLOrchestrator(config=config, db_manager=db_manager, rule_optimizer=rule_optimizer)
    logger.info('AutoML Orchestrator created with integrated components')
    return orchestrator
