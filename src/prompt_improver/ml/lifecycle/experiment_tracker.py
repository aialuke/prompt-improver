"""Advanced Experiment Tracking System for ML Lifecycle Management.

This module provides comprehensive experiment tracking with:
- Hyperparameter tracking and optimization
- Parallel experiment execution
- Experiment comparison and analysis
- Automated hyperparameter tuning
- Integration with model registry
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field as PydanticField, ConfigDict
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import ParameterGrid

from prompt_improver.ml.types import (
    features, labels, hyper_parameters, metrics_dict,
    TrainingBatch, OptimizationResult
)
from prompt_improver.utils.datetime_utils import aware_utc_now

# Optional imports for advanced optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import ray
    from ray import tune
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    """Experiment execution status."""
    created = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    paused = "paused"

class OptimizationStrategy(Enum):
    """Hyperparameter optimization strategies."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    evolutionary = "evolutionary"
    optuna = "optuna"
    RAY_TUNE = "ray_tune"

class ParallelizationMode(Enum):
    """Experiment parallelization modes."""
    sequential = "sequential"
    THREAD_PARALLEL = "thread_parallel"
    PROCESS_PARALLEL = "process_parallel"
    distributed = "distributed"

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    experiment_id: str
    experiment_name: str
    description: str
    objective_metric: str
    objective_direction: str  # "minimize" or "maximize"
    
    # Hyperparameter search space
    hyperparameter_space: Dict[str, Any]
    fixed_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization settings
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN
    max_trials: int = 100
    timeout_seconds: Optional[int] = None
    early_stopping_rounds: Optional[int] = 10
    
    # Parallelization settings
    parallelization_mode: ParallelizationMode = ParallelizationMode.THREAD_PARALLEL
    n_parallel_trials: int = 4
    
    # Resource limits
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[float] = None
    
    # Tracking settings
    track_metrics: List[str] = field(default_factory=list)
    save_models: bool = True
    save_frequency: int = 10  # Save every N trials

@dataclass
class Trial:
    """Individual trial within an experiment."""
    trial_id: str
    experiment_id: str
    trial_number: int
    hyperparameters: hyper_parameters
    
    # Execution info
    status: ExperimentStatus = ExperimentStatus.created
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Results
    metrics: metrics_dict = field(default_factory=dict)
    model_id: Optional[str] = None
    
    # Resource usage
    peak_memory_mb: Optional[float] = None
    avg_cpu_percent: Optional[float] = None
    
    # Error tracking
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Additional info
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)

@dataclass
class ExperimentResults:
    """Aggregated experiment results."""
    experiment_id: str
    total_trials: int
    successful_trials: int
    failed_trials: int
    
    # Best trial info
    best_trial_id: str
    best_hyperparameters: hyper_parameters
    best_metric_value: float
    
    # Statistics
    metric_mean: float
    metric_std: float
    metric_min: float
    metric_max: float
    
    # Timing
    total_duration_seconds: float
    avg_trial_duration_seconds: float
    
    # Hyperparameter importance (if available)
    parameter_importance: Optional[Dict[str, float]] = None
    
    # Convergence info
    convergence_iteration: Optional[int] = None
    early_stopped: bool = False

class BayesianOptimizer:
    """Bayesian optimization for hyperparameter tuning."""
    
    def __init__(self, 
                 bounds: Dict[str, Tuple[float, float]],
                 objective_direction: str = "maximize"):
        """Initialize Bayesian optimizer.
        
        Args:
            bounds: Parameter bounds {param_name: (min, max)}
            objective_direction: "minimize" or "maximize"
        """
        self.bounds = bounds
        self.param_names = list(bounds.keys())
        self.objective_direction = objective_direction
        
        # Gaussian process for modeling objective function
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        # History
        self.X_observed = []
        self.y_observed = []
        
    def suggest_next(self, n_suggestions: int = 1) -> List[Dict[str, float]]:
        """Suggest next hyperparameters to try.
        
        Args:
            n_suggestions: Number of suggestions to generate
            
        Returns:
            List of hyperparameter dictionaries
        """
        suggestions = []
        
        for _ in range(n_suggestions):
            if len(self.X_observed) < 5:
                # Random exploration for initial points
                suggestion = self._random_sample()
            else:
                # Use acquisition function
                suggestion = self._optimize_acquisition()
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def update(self, params: Dict[str, float], objective_value: float):
        """Update optimizer with observed result.
        
        Args:
            params: Hyperparameters tried
            objective_value: Resulting objective value
        """
        X = [params[name] for name in self.param_names]
        y = -objective_value if self.objective_direction == "minimize" else objective_value
        
        self.X_observed.append(X)
        self.y_observed.append(y)
        
        # Refit Gaussian process
        if len(self.X_observed) >= 2:
            self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
    
    def _random_sample(self) -> Dict[str, float]:
        """Generate random sample within bounds."""
        sample = {}
        for param, (low, high) in self.bounds.items():
            sample[param] = np.random.uniform(low, high)
        return sample
    
    def _optimize_acquisition(self) -> Dict[str, float]:
        """Optimize acquisition function to find next point."""
        # Expected Improvement acquisition function
        def acquisition(X):
            X = np.atleast_2d(X)
            mu, sigma = self.gp.predict(X, return_std=True)
            
            # Current best
            y_best = np.max(self.y_observed)
            
            # Expected improvement
            with np.errstate(divide='warn'):
                z = (mu - y_best) / sigma
                ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
                ei[sigma == 0.0] = 0.0
            
            return -ei  # Minimize negative EI
        
        # Multi-start optimization
        best_x = None
        best_acq = float('inf')
        
        for _ in range(10):
            x0 = self._random_sample()
            x0_array = np.array([x0[name] for name in self.param_names])
            
            # Simple gradient-free optimization
            # In practice, use scipy.optimize
            result = x0_array  # Placeholder
            acq_value = acquisition(result)
            
            if acq_value < best_acq:
                best_acq = acq_value
                best_x = result
        
        # Convert back to dictionary
        return {name: best_x[i] for i, name in enumerate(self.param_names)}

class ExperimentTracker:
    """Advanced experiment tracking system."""
    
    def __init__(self,
                 storage_path: Path = Path("./experiments"),
                 model_registry=None):
        """Initialize experiment tracker.
        
        Args:
            storage_path: Path for storing experiment data
            model_registry: Optional model registry instance
        """
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.model_registry = model_registry
        
        # In-memory tracking
        self._experiments: Dict[str, ExperimentConfig] = {}
        self._trials: Dict[str, List[Trial]] = {}
        self._active_experiments: Set[str] = set()
        
        # Optimizers for each experiment
        self._optimizers: Dict[str, BayesianOptimizer] = {}
        
        # Execution pools
        self._thread_pool = ThreadPoolExecutor(max_workers=8)
        self._process_pool = ProcessPoolExecutor(max_workers=4)
        
        logger.info(f"Experiment tracker initialized at {storage_path}")
    
    async def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        experiment_id = config.experiment_id
        
        # Store configuration
        self._experiments[experiment_id] = config
        self._trials[experiment_id] = []
        
        # Initialize optimizer if using Bayesian optimization
        if config.optimization_strategy == OptimizationStrategy.BAYESIAN:
            bounds = self._extract_bounds(config.hyperparameter_space)
            self._optimizers[experiment_id] = BayesianOptimizer(
                bounds=bounds,
                objective_direction=config.objective_direction
            )
        
        # Save experiment config
        config_path = self.storage_path / experiment_id / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w") as f:
            json.dump(self._config_to_dict(config), f, indent=2)
        
        logger.info(f"Created experiment {experiment_id}: {config.experiment_name}")
        return experiment_id
    
    async def run_experiment(self,
                           experiment_id: str,
                           train_function: Callable,
                           train_data: Tuple[features, labels],
                           validation_data: Optional[Tuple[features, labels]] = None) -> ExperimentResults:
        """Run a complete experiment with multiple trials.
        
        Args:
            experiment_id: Experiment to run
            train_function: Function that trains a model given hyperparameters
            train_data: Training data (features, labels)
            validation_data: Optional validation data
            
        Returns:
            Experiment results
        """
        config = self._experiments.get(experiment_id)
        if not config:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        if experiment_id in self._active_experiments:
            raise RuntimeError(f"Experiment {experiment_id} is already running")
        
        self._active_experiments.add(experiment_id)
        experiment_start = time.time()
        
        try:
            # Generate trial configurations
            trial_configs = await self._generate_trial_configs(config)
            
            # Run trials based on parallelization mode
            if config.parallelization_mode == ParallelizationMode.sequential:
                results = await self._run_sequential(
                    experiment_id, trial_configs, train_function, train_data, validation_data
                )
            elif config.parallelization_mode == ParallelizationMode.THREAD_PARALLEL:
                results = await self._run_thread_parallel(
                    experiment_id, trial_configs, train_function, train_data, validation_data,
                    config.n_parallel_trials
                )
            elif config.parallelization_mode == ParallelizationMode.PROCESS_PARALLEL:
                results = await self._run_process_parallel(
                    experiment_id, trial_configs, train_function, train_data, validation_data,
                    config.n_parallel_trials
                )
            else:  # distributed
                results = await self._run_distributed(
                    experiment_id, trial_configs, train_function, train_data, validation_data
                )
            
            # Analyze results
            experiment_results = await self._analyze_experiment(
                experiment_id,
                time.time() - experiment_start
            )
            
            # Save final results
            results_path = self.storage_path / experiment_id / "results.json"
            with open(results_path, "w") as f:
                json.dump(self._results_to_dict(experiment_results), f, indent=2)
            
            return experiment_results
            
        finally:
            self._active_experiments.discard(experiment_id)
    
    async def run_trial(self,
                      experiment_id: str,
                      hyperparameters: hyper_parameters,
                      train_function: Callable,
                      train_data: Tuple[features, labels],
                      validation_data: Optional[Tuple[features, labels]] = None) -> Trial:
        """Run a single trial within an experiment.
        
        Args:
            experiment_id: Parent experiment
            hyperparameters: Hyperparameters for this trial
            train_function: Training function
            train_data: Training data
            validation_data: Optional validation data
            
        Returns:
            Trial results
        """
        config = self._experiments.get(experiment_id)
        if not config:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Create trial
        trial_number = len(self._trials[experiment_id]) + 1
        trial = Trial(
            trial_id=f"{experiment_id}_trial_{trial_number}",
            experiment_id=experiment_id,
            trial_number=trial_number,
            hyperparameters=hyperparameters
        )
        
        trial.status = ExperimentStatus.RUNNING
        trial.start_time = aware_utc_now()
        
        try:
            # Merge hyperparameters with fixed parameters
            full_params = {**config.fixed_parameters, **hyperparameters}
            
            # Run training
            start_time = time.time()
            model, metrics = await self._run_training(
                train_function,
                train_data,
                validation_data,
                full_params
            )
            duration = time.time() - start_time
            
            # Update trial
            trial.metrics = metrics
            trial.duration_seconds = duration
            trial.status = ExperimentStatus.COMPLETED
            
            # Save model if configured
            if config.save_models and self.model_registry:
                from .model_registry import ModelMetadata, ModelStatus
                
                metadata = ModelMetadata(
                    model_id="",  # Will be set by registry
                    model_name=f"{config.experiment_name}_model",
                    version=f"trial_{trial_number}",
                    created_by="experiment_tracker",
                    created_at=aware_utc_now(),
                    status=ModelStatus.VALIDATION,
                    training_hyperparameters=full_params,
                    validation_metrics=metrics,
                    experiment_id=experiment_id
                )
                
                model_id = await self.model_registry.register_model(
                    model=model,
                    model_name=metadata.model_name,
                    version=metadata.version,
                    metadata=metadata,
                    experiment_id=experiment_id
                )
                trial.model_id = model_id
            
            # Update optimizer if using Bayesian optimization
            if config.optimization_strategy == OptimizationStrategy.BAYESIAN:
                optimizer = self._optimizers.get(experiment_id)
                if optimizer:
                    objective_value = metrics.get(config.objective_metric, 0.0)
                    optimizer.update(hyperparameters, objective_value)
            
        except Exception as e:
            trial.status = ExperimentStatus.FAILED
            trial.error_message = str(e)
            import traceback
            trial.stack_trace = traceback.format_exc()
            logger.error(f"Trial {trial.trial_id} failed: {e}")
        
        finally:
            trial.end_time = aware_utc_now()
            if trial.start_time:
                trial.duration_seconds = (trial.end_time - trial.start_time).total_seconds()
        
        # Store trial
        self._trials[experiment_id].append(trial)
        
        # Save trial data
        trial_path = self.storage_path / experiment_id / "trials" / f"{trial.trial_id}.json"
        trial_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(trial_path, "w") as f:
            json.dump(self._trial_to_dict(trial), f, indent=2)
        
        return trial
    
    async def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current experiment status.
        
        Args:
            experiment_id: Experiment to check
            
        Returns:
            Status information
        """
        config = self._experiments.get(experiment_id)
        if not config:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        trials = self._trials.get(experiment_id, [])
        
        status = {
            "experiment_id": experiment_id,
            "experiment_name": config.experiment_name,
            "is_active": experiment_id in self._active_experiments,
            "total_trials": len(trials),
            "completed_trials": sum(1 for t in trials if t.status == ExperimentStatus.COMPLETED),
            "failed_trials": sum(1 for t in trials if t.status == ExperimentStatus.FAILED),
            "running_trials": sum(1 for t in trials if t.status == ExperimentStatus.RUNNING),
            "optimization_strategy": config.optimization_strategy.value,
            "max_trials": config.max_trials
        }
        
        # Add best trial info if available
        if trials:
            completed_trials = [t for t in trials if t.status == ExperimentStatus.COMPLETED]
            if completed_trials:
                objective_metric = config.objective_metric
                
                if config.objective_direction == "maximize":
                    best_trial = max(completed_trials, key=lambda t: t.metrics.get(objective_metric, float('-inf')))
                else:
                    best_trial = min(completed_trials, key=lambda t: t.metrics.get(objective_metric, float('inf')))
                
                status["best_trial"] = {
                    "trial_id": best_trial.trial_id,
                    "hyperparameters": best_trial.hyperparameters,
                    "metrics": best_trial.metrics,
                    "objective_value": best_trial.metrics.get(objective_metric)
                }
        
        return status
    
    async def compare_experiments(self,
                                experiment_ids: List[str],
                                metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare multiple experiments.
        
        Args:
            experiment_ids: Experiments to compare
            metrics: Specific metrics to compare
            
        Returns:
            Comparison results
        """
        comparison = {
            "experiments": {},
            "best_by_metric": {}
        }
        
        for exp_id in experiment_ids:
            if exp_id not in self._experiments:
                continue
            
            config = self._experiments[exp_id]
            trials = self._trials.get(exp_id, [])
            completed_trials = [t for t in trials if t.status == ExperimentStatus.COMPLETED]
            
            if completed_trials:
                # Get best trial
                objective_metric = config.objective_metric
                if config.objective_direction == "maximize":
                    best_trial = max(completed_trials, key=lambda t: t.metrics.get(objective_metric, float('-inf')))
                else:
                    best_trial = min(completed_trials, key=lambda t: t.metrics.get(objective_metric, float('inf')))
                
                exp_info = {
                    "name": config.experiment_name,
                    "total_trials": len(trials),
                    "successful_trials": len(completed_trials),
                    "best_hyperparameters": best_trial.hyperparameters,
                    "best_metrics": best_trial.metrics,
                    "optimization_strategy": config.optimization_strategy.value
                }
                
                comparison["experiments"][exp_id] = exp_info
        
        # Find best experiment for each metric
        if metrics:
            for metric in metrics:
                best_exp = None
                best_value = None
                
                for exp_id, exp_info in comparison["experiments"].items():
                    if metric in exp_info["best_metrics"]:
                        value = exp_info["best_metrics"][metric]
                        if best_value is None or value > best_value:
                            best_value = value
                            best_exp = exp_id
                
                if best_exp:
                    comparison["best_by_metric"][metric] = {
                        "experiment_id": best_exp,
                        "value": best_value
                    }
        
        return comparison
    
    # Helper methods
    
    async def _generate_trial_configs(self, config: ExperimentConfig) -> List[hyper_parameters]:
        """Generate trial configurations based on optimization strategy."""
        if config.optimization_strategy == OptimizationStrategy.GRID_SEARCH:
            return list(ParameterGrid(config.hyperparameter_space))
        
        elif config.optimization_strategy == OptimizationStrategy.RANDOM_SEARCH:
            configs = []
            for _ in range(config.max_trials):
                trial_config = {}
                for param, values in config.hyperparameter_space.items():
                    if isinstance(values, list):
                        trial_config[param] = np.random.choice(values)
                    elif isinstance(values, tuple) and len(values) == 2:
                        # Assume uniform distribution between bounds
                        trial_config[param] = np.random.uniform(values[0], values[1])
                configs.append(trial_config)
            return configs
        
        elif config.optimization_strategy == OptimizationStrategy.BAYESIAN:
            # Generate trials on-demand during execution
            return []
        
        elif config.optimization_strategy == OptimizationStrategy.optuna and OPTUNA_AVAILABLE:
            # Optuna manages its own trial generation
            return []
        
        else:
            raise NotImplementedError(f"Strategy {config.optimization_strategy} not implemented")
    
    async def _run_sequential(self,
                            experiment_id: str,
                            trial_configs: List[hyper_parameters],
                            train_function: Callable,
                            train_data: Tuple[features, labels],
                            validation_data: Optional[Tuple[features, labels]]) -> List[Trial]:
        """Run trials sequentially."""
        config = self._experiments[experiment_id]
        results = []
        
        # For Bayesian optimization, generate trials on-demand
        if config.optimization_strategy == OptimizationStrategy.BAYESIAN:
            optimizer = self._optimizers.get(experiment_id)
            if not optimizer:
                raise RuntimeError("Bayesian optimizer not initialized")
            
            for i in range(config.max_trials):
                # Get next suggestion
                suggestions = optimizer.suggest_next(1)
                hyperparameters = suggestions[0]
                
                # Run trial
                trial = await self.run_trial(
                    experiment_id,
                    hyperparameters,
                    train_function,
                    train_data,
                    validation_data
                )
                results.append(trial)
                
                # Check early stopping
                if await self._should_stop_early(experiment_id, i + 1):
                    logger.info(f"Early stopping triggered for experiment {experiment_id}")
                    break
        else:
            # Run pre-generated configs
            for i, hyperparameters in enumerate(trial_configs):
                if config.max_trials and i >= config.max_trials:
                    break
                
                trial = await self.run_trial(
                    experiment_id,
                    hyperparameters,
                    train_function,
                    train_data,
                    validation_data
                )
                results.append(trial)
        
        return results
    
    async def _run_thread_parallel(self,
                                 experiment_id: str,
                                 trial_configs: List[hyper_parameters],
                                 train_function: Callable,
                                 train_data: Tuple[features, labels],
                                 validation_data: Optional[Tuple[features, labels]],
                                 n_parallel: int) -> List[Trial]:
        """Run trials in parallel using threads."""
        config = self._experiments[experiment_id]
        results = []
        
        async def run_trial_async(hyperparameters):
            return await self.run_trial(
                experiment_id,
                hyperparameters,
                train_function,
                train_data,
                validation_data
            )
        
        # For Bayesian optimization
        if config.optimization_strategy == OptimizationStrategy.BAYESIAN:
            optimizer = self._optimizers.get(experiment_id)
            if not optimizer:
                raise RuntimeError("Bayesian optimizer not initialized")
            
            trials_completed = 0
            while trials_completed < config.max_trials:
                # Get batch of suggestions
                n_suggestions = min(n_parallel, config.max_trials - trials_completed)
                suggestions = optimizer.suggest_next(n_suggestions)
                
                # Run batch in parallel
                tasks = [run_trial_async(params) for params in suggestions]
                batch_results = await asyncio.gather(*tasks)
                
                results.extend(batch_results)
                trials_completed += len(batch_results)
                
                # Check early stopping
                if await self._should_stop_early(experiment_id, trials_completed):
                    logger.info(f"Early stopping triggered for experiment {experiment_id}")
                    break
        else:
            # Run pre-generated configs in batches
            for i in range(0, len(trial_configs), n_parallel):
                if config.max_trials and len(results) >= config.max_trials:
                    break
                
                batch = trial_configs[i:i + n_parallel]
                if config.max_trials:
                    remaining = config.max_trials - len(results)
                    batch = batch[:remaining]
                
                tasks = [run_trial_async(params) for params in batch]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
        
        return results
    
    async def _run_process_parallel(self,
                                  experiment_id: str,
                                  trial_configs: List[hyper_parameters],
                                  train_function: Callable,
                                  train_data: Tuple[features, labels],
                                  validation_data: Optional[Tuple[features, labels]],
                                  n_parallel: int) -> List[Trial]:
        """Run trials in parallel using processes."""
        # Similar to thread parallel but using process pool
        # Implementation would be similar but with process pool executor
        return await self._run_thread_parallel(
            experiment_id, trial_configs, train_function, train_data, validation_data, n_parallel
        )
    
    async def _run_distributed(self,
                             experiment_id: str,
                             trial_configs: List[hyper_parameters],
                             train_function: Callable,
                             train_data: Tuple[features, labels],
                             validation_data: Optional[Tuple[features, labels]]) -> List[Trial]:
        """Run trials in distributed mode using Ray."""
        if not RAY_AVAILABLE:
            logger.warning("Ray not available, falling back to thread parallel")
            return await self._run_thread_parallel(
                experiment_id, trial_configs, train_function, train_data, validation_data, 4
            )
        
        # Ray-based distributed execution would go here
        # For now, fallback to thread parallel
        return await self._run_thread_parallel(
            experiment_id, trial_configs, train_function, train_data, validation_data, 4
        )
    
    async def _run_training(self,
                          train_function: Callable,
                          train_data: Tuple[features, labels],
                          validation_data: Optional[Tuple[features, labels]],
                          hyperparameters: hyper_parameters) -> Tuple[Any, metrics_dict]:
        """Execute training function with error handling."""
        try:
            # Train function should return (model, metrics)
            result = await asyncio.get_event_loop().run_in_executor(
                self._thread_pool,
                train_function,
                train_data,
                validation_data,
                hyperparameters
            )
            
            if isinstance(result, tuple) and len(result) == 2:
                return result
            else:
                # If only model returned, create empty metrics
                return result, {}
                
        except Exception as e:
            logger.error(f"Training failed with params {hyperparameters}: {e}")
            raise
    
    async def _should_stop_early(self, experiment_id: str, trials_completed: int) -> bool:
        """Check if experiment should stop early."""
        config = self._experiments[experiment_id]
        
        if not config.early_stopping_rounds:
            return False
        
        trials = self._trials.get(experiment_id, [])
        if len(trials) < config.early_stopping_rounds:
            return False
        
        # Check if objective metric hasn't improved in last N rounds
        recent_trials = trials[-config.early_stopping_rounds:]
        objective_values = [
            t.metrics.get(config.objective_metric, float('inf'))
            for t in recent_trials
            if t.status == ExperimentStatus.COMPLETED
        ]
        
        if len(objective_values) < config.early_stopping_rounds:
            return False
        
        # Check for improvement
        if config.objective_direction == "maximize":
            best_recent = max(objective_values)
            best_overall = max(
                t.metrics.get(config.objective_metric, float('-inf'))
                for t in trials
                if t.status == ExperimentStatus.COMPLETED
            )
            return best_recent <= best_overall
        else:
            best_recent = min(objective_values)
            best_overall = min(
                t.metrics.get(config.objective_metric, float('inf'))
                for t in trials
                if t.status == ExperimentStatus.COMPLETED
            )
            return best_recent >= best_overall
    
    async def _analyze_experiment(self,
                                experiment_id: str,
                                total_duration: float) -> ExperimentResults:
        """Analyze experiment results."""
        config = self._experiments[experiment_id]
        trials = self._trials.get(experiment_id, [])
        
        completed_trials = [t for t in trials if t.status == ExperimentStatus.COMPLETED]
        failed_trials = [t for t in trials if t.status == ExperimentStatus.FAILED]
        
        if not completed_trials:
            raise ValueError(f"No successful trials in experiment {experiment_id}")
        
        # Find best trial
        objective_metric = config.objective_metric
        if config.objective_direction == "maximize":
            best_trial = max(completed_trials, key=lambda t: t.metrics.get(objective_metric, float('-inf')))
        else:
            best_trial = min(completed_trials, key=lambda t: t.metrics.get(objective_metric, float('inf')))
        
        # Calculate statistics
        objective_values = [
            t.metrics.get(objective_metric, 0.0)
            for t in completed_trials
        ]
        
        # Calculate parameter importance if we have enough data
        parameter_importance = None
        if len(completed_trials) >= 10:
            parameter_importance = self._calculate_parameter_importance(
                completed_trials,
                objective_metric
            )
        
        results = ExperimentResults(
            experiment_id=experiment_id,
            total_trials=len(trials),
            successful_trials=len(completed_trials),
            failed_trials=len(failed_trials),
            best_trial_id=best_trial.trial_id,
            best_hyperparameters=best_trial.hyperparameters,
            best_metric_value=best_trial.metrics.get(objective_metric, 0.0),
            metric_mean=np.mean(objective_values),
            metric_std=np.std(objective_values),
            metric_min=np.min(objective_values),
            metric_max=np.max(objective_values),
            total_duration_seconds=total_duration,
            avg_trial_duration_seconds=np.mean([
                t.duration_seconds for t in completed_trials if t.duration_seconds
            ]),
            parameter_importance=parameter_importance,
            early_stopped=len(trials) < config.max_trials
        )
        
        return results
    
    def _calculate_parameter_importance(self,
                                      trials: List[Trial],
                                      objective_metric: str) -> Dict[str, float]:
        """Calculate relative importance of hyperparameters."""
        # Simple correlation-based importance
        # In practice, use more sophisticated methods like SHAP
        
        importance = {}
        
        # Get all parameter names
        param_names = set()
        for trial in trials:
            param_names.update(trial.hyperparameters.keys())
        
        # Calculate correlation for each parameter
        objective_values = np.array([
            t.metrics.get(objective_metric, 0.0) for t in trials
        ])
        
        for param in param_names:
            param_values = []
            for trial in trials:
                value = trial.hyperparameters.get(param, 0)
                # Convert to numeric if possible
                if isinstance(value, (int, float)):
                    param_values.append(value)
                else:
                    # For categorical, use hash
                    param_values.append(hash(str(value)) % 1000)
            
            if len(set(param_values)) > 1:  # Only if parameter varies
                correlation = abs(np.corrcoef(param_values, objective_values)[0, 1])
                importance[param] = float(correlation)
            else:
                importance[param] = 0.0
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance
    
    def _extract_bounds(self, hyperparameter_space: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """Extract parameter bounds for Bayesian optimization."""
        bounds = {}
        
        for param, spec in hyperparameter_space.items():
            if isinstance(spec, tuple) and len(spec) == 2:
                bounds[param] = spec
            elif isinstance(spec, list) and all(isinstance(x, (int, float)) for x in spec):
                bounds[param] = (min(spec), max(spec))
            else:
                # Skip non-numeric parameters for now
                pass
        
        return bounds
    
    def _config_to_dict(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "experiment_id": config.experiment_id,
            "experiment_name": config.experiment_name,
            "description": config.description,
            "objective_metric": config.objective_metric,
            "objective_direction": config.objective_direction,
            "hyperparameter_space": config.hyperparameter_space,
            "fixed_parameters": config.fixed_parameters,
            "optimization_strategy": config.optimization_strategy.value,
            "max_trials": config.max_trials,
            "timeout_seconds": config.timeout_seconds,
            "early_stopping_rounds": config.early_stopping_rounds,
            "parallelization_mode": config.parallelization_mode.value,
            "n_parallel_trials": config.n_parallel_trials,
            "max_memory_mb": config.max_memory_mb,
            "max_cpu_percent": config.max_cpu_percent,
            "track_metrics": config.track_metrics,
            "save_models": config.save_models,
            "save_frequency": config.save_frequency
        }
    
    def _trial_to_dict(self, trial: Trial) -> Dict[str, Any]:
        """Convert trial to dictionary for serialization."""
        return {
            "trial_id": trial.trial_id,
            "experiment_id": trial.experiment_id,
            "trial_number": trial.trial_number,
            "hyperparameters": trial.hyperparameters,
            "status": trial.status.value,
            "start_time": trial.start_time.isoformat() if trial.start_time else None,
            "end_time": trial.end_time.isoformat() if trial.end_time else None,
            "duration_seconds": trial.duration_seconds,
            "metrics": trial.metrics,
            "model_id": trial.model_id,
            "peak_memory_mb": trial.peak_memory_mb,
            "avg_cpu_percent": trial.avg_cpu_percent,
            "error_message": trial.error_message,
            "stack_trace": trial.stack_trace,
            "tags": trial.tags,
            "artifacts": trial.artifacts
        }
    
    def _results_to_dict(self, results: ExperimentResults) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            "experiment_id": results.experiment_id,
            "total_trials": results.total_trials,
            "successful_trials": results.successful_trials,
            "failed_trials": results.failed_trials,
            "best_trial_id": results.best_trial_id,
            "best_hyperparameters": results.best_hyperparameters,
            "best_metric_value": results.best_metric_value,
            "metric_mean": results.metric_mean,
            "metric_std": results.metric_std,
            "metric_min": results.metric_min,
            "metric_max": results.metric_max,
            "total_duration_seconds": results.total_duration_seconds,
            "avg_trial_duration_seconds": results.avg_trial_duration_seconds,
            "parameter_importance": results.parameter_importance,
            "convergence_iteration": results.convergence_iteration,
            "early_stopped": results.early_stopped
        }