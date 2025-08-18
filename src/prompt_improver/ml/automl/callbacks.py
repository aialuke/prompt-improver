"""AutoML Callbacks for Optuna Integration
Implements 2025 best practices for callback-based ML framework integration
"""
from datetime import UTC, datetime
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional
import optuna
from optuna.trial import TrialState
if TYPE_CHECKING:
    from ..evaluation.experiment_orchestrator import ExperimentOrchestrator
    from .shared_types import AutoMLOrchestratorProtocol
logger = logging.getLogger(__name__)

class AutoMLCallback:
    """Main AutoML callback implementing 2025 Optuna integration patterns

    features:
    - Real-time trial monitoring and reporting
    - Artifact storage for model persistence
    - Early stopping based on performance criteria
    - Integration with existing prompt improver components
    """

    def __init__(self, orchestrator: 'AutoMLOrchestrator', enable_early_stopping: bool=True, enable_artifact_storage: bool=True, performance_threshold: float=0.95):
        """Initialize AutoML callback

        Args:
            orchestrator: AutoML orchestrator instance
            enable_early_stopping: Enable early stopping for poor trials
            enable_artifact_storage: Enable model/config artifact storage
            performance_threshold: Threshold for early stopping decisions
        """
        self.orchestrator = orchestrator
        self.enable_early_stopping = enable_early_stopping
        self.enable_artifact_storage = enable_artifact_storage
        self.performance_threshold = performance_threshold
        self.trial_start_times = {}
        self.best_value_so_far = None
        self.trials_since_improvement = 0
        logger.info('AutoML callback initialized with 2025 integration patterns')

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Main callback function following Optuna 2025 patterns

        Called after each trial completion with trial results
        """
        try:
            self._update_tracking_state(study, trial)
            self._report_trial_results(study, trial)
            if self.enable_artifact_storage and trial.state == TrialState.COMPLETE:
                self._store_trial_artifacts(study, trial)
            if self.enable_early_stopping:
                self._check_early_stopping(study, trial)
            self._update_real_time_metrics(study, trial)
        except Exception as e:
            logger.error('AutoML callback failed: %s', e)

    def _update_tracking_state(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Update internal state tracking"""
        if trial.state == TrialState.COMPLETE and trial.value is not None:
            if self.best_value_so_far is None or trial.value > self.best_value_so_far:
                self.best_value_so_far = trial.value
                self.trials_since_improvement = 0
            else:
                self.trials_since_improvement += 1

    def _report_trial_results(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Report trial results to orchestrator and components"""
        try:
            trial_summary = {'trial_number': trial.number, 'trial_state': trial.state.name, 'trial_value': trial.value, 'trial_params': trial.params, 'execution_time': self._get_trial_execution_time(trial), 'study_name': study.study_name, 'timestamp': datetime.now(UTC).isoformat()}
            if hasattr(self.orchestrator, '_handle_trial_completion'):
                self.orchestrator._handle_trial_completion(trial_summary)
            if trial.state == TrialState.COMPLETE:
                logger.info('Trial %s completed: value=%s, params=%s', trial.number, format(trial.value, '.4f'), trial.params)
            elif trial.state == TrialState.PRUNED:
                logger.info('Trial %s pruned early', trial.number)
            else:
                logger.warning('Trial {trial.number} failed: %s', trial.state)
        except Exception as e:
            logger.error('Failed to report trial results: %s', e)

    def _store_trial_artifacts(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Store trial artifacts following 2025 patterns"""
        try:
            artifact_metadata = {'trial_number': trial.number, 'study_name': study.study_name, 'trial_value': trial.value, 'trial_params': trial.params, 'timestamp': datetime.now(UTC).isoformat(), 'optimization_mode': getattr(self.orchestrator.config, 'optimization_mode', 'unknown')}
            trial.set_user_attr('artifact_metadata', artifact_metadata)
            if trial.value == study.best_value:
                trial.set_user_attr('is_best_trial', True)
                if hasattr(self.orchestrator, 'best_configurations'):
                    self.orchestrator.best_configurations['latest_best'] = {'trial_number': trial.number, 'value': trial.value, 'params': trial.params, 'timestamp': datetime.now(UTC)}
        except Exception as e:
            logger.error('Failed to store trial artifacts: %s', e)

    def _check_early_stopping(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Check early stopping conditions following 2025 best practices"""
        try:
            if trial.state == TrialState.COMPLETE and trial.value is not None and (trial.value >= self.performance_threshold):
                logger.info('Early stopping: Performance threshold %s reached', self.performance_threshold)
                study.stop()
                return
            if self.trials_since_improvement >= 20:
                logger.info('Early stopping: No improvement for %s trials', self.trials_since_improvement)
                study.stop()
                return
            failed_trials = len([t for t in study.trials if t.state == TrialState.FAIL])
            if failed_trials >= 10:
                logger.warning('Early stopping: Too many failed trials (%s)', failed_trials)
                study.stop()
                return
        except Exception as e:
            logger.error('Early stopping check failed: %s', e)

    def _update_real_time_metrics(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Update real-time metrics for dashboard/monitoring"""
        try:
            metrics = {'current_trial': trial.number, 'total_trials': len(study.trials), 'best_value': study.best_value if len(study.trials) > 0 else None, 'best_params': study.best_params if len(study.trials) > 0 else {}, 'completion_rate': len([t for t in study.trials if t.state == TrialState.COMPLETE]) / max(len(study.trials), 1), 'trials_since_improvement': self.trials_since_improvement, 'timestamp': datetime.now(UTC).isoformat()}
            if hasattr(self.orchestrator, '_current_metrics'):
                self.orchestrator._current_metrics = metrics
        except Exception as e:
            logger.error('Failed to update real-time metrics: %s', e)

    def _get_trial_execution_time(self, trial: optuna.Trial) -> float | None:
        """Get trial execution time if available"""
        try:
            if hasattr(trial, 'datetime_start') and hasattr(trial, 'datetime_complete'):
                if trial.datetime_start and trial.datetime_complete:
                    delta = trial.datetime_complete - trial.datetime_start
                    return delta.total_seconds()
            return None
        except Exception:
            return None

class RealTimeAnalyticsCallback:
    """Callback for real-time analytics integration
    Streams optimization progress to WebSocket connections
    """

    def __init__(self, analytics_service):
        """Initialize real-time analytics callback

        Args:
            analytics_service: Analytics service for WebSocket streaming (from get_analytics_router)
        """
        self.analytics_service = analytics_service
        logger.info('Real-time analytics callback initialized')

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Stream trial updates to real-time analytics"""
        try:
            update_message = {'type': 'automl_trial_update', 'data': {'trial_number': trial.number, 'trial_state': trial.state.name, 'trial_value': trial.value, 'study_name': study.study_name, 'best_value': study.best_value if len(study.trials) > 0 else None, 'progress': {'completed_trials': len([t for t in study.trials if t.state == TrialState.COMPLETE]), 'total_trials': len(study.trials), 'success_rate': len([t for t in study.trials if t.state == TrialState.COMPLETE]) / max(len(study.trials), 1)}}, 'timestamp': datetime.now(UTC).isoformat()}
            if hasattr(self.analytics_service, 'broadcast_update'):
                self.analytics_service.broadcast_update(update_message)
            elif hasattr(self.analytics_service, 'send_real_time_update'):
                self.analytics_service.send_real_time_update(update_message)
        except Exception as e:
            logger.error('Real-time analytics callback failed: %s', e)

class ExperimentCallback:
    """Callback for A/B testing experiment integration
    Coordinates Optuna optimization with experiment orchestrator
    """

    def __init__(self, experiment_orchestrator: 'ExperimentOrchestrator'):
        """Initialize experiment callback

        Args:
            experiment_orchestrator: A/B testing experiment orchestrator
        """
        self.experiment_orchestrator = experiment_orchestrator
        self.active_experiments = {}
        logger.info('Experiment callback initialized')

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Handle experiment lifecycle integration"""
        try:
            if trial.state == TrialState.COMPLETE and trial.value is not None:
                if self._is_significant_improvement(study, trial):
                    self._trigger_ab_test(trial)
                self._update_experiment_tracking(study, trial)
        except Exception as e:
            logger.error('Experiment callback failed: %s', e)

    def _is_significant_improvement(self, study: optuna.Study, trial: optuna.Trial) -> bool:
        """Check if trial represents significant improvement"""
        try:
            if len(study.trials) < 10:
                return False
            recent_trials = [t for t in study.trials[-10:] if t.state == TrialState.COMPLETE and t.value is not None]
            if len(recent_trials) < 5:
                return False
            recent_values = [t.value for t in recent_trials]
            avg_recent = sum(recent_values) / len(recent_values)
            improvement = (trial.value - avg_recent) / avg_recent
            return improvement > 0.05
        except Exception:
            return False

    def _trigger_ab_test(self, trial: optuna.Trial) -> None:
        """Trigger A/B test with optimized parameters"""
        try:
            experiment_config = {'name': f'automl_optimized_trial_{trial.number}', 'parameters': trial.params, 'expected_improvement': trial.value, 'source': 'automl_optimization', 'timestamp': datetime.now(UTC).isoformat()}
            self.active_experiments[trial.number] = experiment_config
            if hasattr(self.experiment_orchestrator, 'create_experiment_from_optimization'):
                self.experiment_orchestrator.create_experiment_from_optimization(experiment_config)
            logger.info('Triggered A/B test for trial %s with improvement %s', trial.number, format(trial.value, '.4f'))
        except Exception as e:
            logger.error('Failed to trigger A/B test: %s', e)

    def _update_experiment_tracking(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Update experiment tracking metadata"""
        try:
            tracking_data = {'study_name': study.study_name, 'trial_number': trial.number, 'optimization_value': trial.value, 'active_experiments': len(self.active_experiments), 'last_update': datetime.now(UTC).isoformat()}
            trial.set_user_attr('experiment_tracking', tracking_data)
        except Exception as e:
            logger.error('Failed to update experiment tracking: %s', e)

class ModelSelectionCallback:
    """Callback for automated model selection and management
    Updates model configurations based on optimization results
    """

    def __init__(self, model_manager):
        """Initialize model selection callback

        Args:
            model_manager: Model manager for automated selection and configuration
        """
        self.model_manager = model_manager
        self.model_updates = []
        logger.info('Model selection callback initialized')

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Handle model selection based on optimization results"""
        try:
            if trial.state == TrialState.COMPLETE and trial.value is not None:
                model_params = {k: v for k, v in trial.params.items() if k.startswith('model_')}
                if model_params and trial.value == study.best_value:
                    self._update_model_configuration(trial, model_params)
                self._track_model_performance(trial, model_params)
        except Exception as e:
            logger.error('Model selection callback failed: %s', e)

    def _update_model_configuration(self, trial: optuna.Trial, model_params: dict[str, Any]) -> None:
        """Update model manager with optimized configuration"""
        try:
            if hasattr(self.model_manager, 'update_configuration'):
                config_update = {'source': 'automl_optimization', 'trial_number': trial.number, 'optimization_value': trial.value, 'parameters': model_params, 'timestamp': datetime.now(UTC).isoformat()}
                self.model_manager.update_configuration(config_update)
                self.model_updates.append(config_update)
                logger.info('Updated model configuration from trial %s', trial.number)
        except Exception as e:
            logger.error('Failed to update model configuration: %s', e)

    def _track_model_performance(self, trial: optuna.Trial, model_params: dict[str, Any]) -> None:
        """Track model performance for analysis"""
        try:
            performance_record = {'trial_number': trial.number, 'performance_value': trial.value, 'model_parameters': model_params, 'timestamp': datetime.now(UTC).isoformat()}
            trial.set_user_attr('model_performance', performance_record)
        except Exception as e:
            logger.error('Failed to track model performance: %s', e)

def create_standard_callbacks(orchestrator: 'AutoMLOrchestrator') -> list:
    """Create standard set of callbacks for AutoML optimization

    Args:
        orchestrator: AutoML orchestrator instance

    Returns:
        List of configured callbacks
    """
    callbacks = []
    callbacks.append(AutoMLCallback(orchestrator))
    from ...api.factories.analytics_api_factory import get_analytics_router
    analytics_router = get_analytics_router()
    if analytics_router:
        callbacks.append(RealTimeAnalyticsCallback(analytics_router))
    if orchestrator.experiment_orchestrator:
        callbacks.append(ExperimentCallback(orchestrator.experiment_orchestrator))
    if orchestrator.model_manager:
        callbacks.append(ModelSelectionCallback(orchestrator.model_manager))
    logger.info('Created %s standard AutoML callbacks', len(callbacks))
    return callbacks
