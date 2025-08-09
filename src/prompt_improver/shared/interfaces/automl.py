"""Shared AutoML interfaces to break circular dependencies
Following 2025 dependency injection patterns
"""
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

class AutoMLMode(Enum):
    """AutoML operation modes"""
    HYPERPARAMETER_OPTIMIZATION = 'hpo'
    AUTOMATED_EXPERIMENT_DESIGN = 'aed'
    CONTINUOUS_OPTIMIZATION = 'continuous'
    MULTI_OBJECTIVE_PARETO = 'pareto'

@dataclass
class TrialResult:
    """Result from an AutoML trial"""
    trial_id: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    duration: float
    status: str
    timestamp: datetime

class IAutoMLCallback(Protocol):
    """Interface for AutoML callbacks

    Enables loose coupling between orchestrator and callbacks
    by depending on abstractions rather than concrete classes.
    """

    def on_trial_start(self, trial_id: str, parameters: dict[str, Any]) -> None:
        """Called when a trial starts

        Args:
            trial_id: Unique identifier for the trial
            parameters: Trial parameters
        """
        ...

    def on_trial_complete(self, result: TrialResult) -> None:
        """Called when a trial completes

        Args:
            result: Complete trial result
        """
        ...

    def on_optimization_start(self, config: dict[str, Any]) -> None:
        """Called when optimization begins

        Args:
            config: Optimization configuration
        """
        ...

    def on_optimization_complete(self, results: list[TrialResult]) -> None:
        """Called when optimization completes

        Args:
            results: All trial results
        """
        ...

class IAutoMLOrchestrator(Protocol):
    """Interface for AutoML orchestrator

    Abstracts AutoML implementation details to enable testing
    and multiple AutoML framework support.
    """

    async def optimize(self, objective_function: Callable[[dict[str, Any]], float], parameter_space: dict[str, Any], n_trials: int=100) -> list[TrialResult]:
        """Run optimization with given configuration

        Args:
            objective_function: Function to optimize
            parameter_space: Parameters to search over
            n_trials: Number of trials to run

        Returns:
            List of trial results
        """
        ...

    def add_callback(self, callback: IAutoMLCallback) -> None:
        """Add callback to orchestrator

        Args:
            callback: Callback to add
        """
        ...

    def remove_callback(self, callback: IAutoMLCallback) -> None:
        """Remove callback from orchestrator

        Args:
            callback: Callback to remove
        """
        ...

    async def get_best_trial(self) -> TrialResult | None:
        """Get the best trial result so far

        Returns:
            Best trial result or None if no trials completed
        """
        ...

    async def stop_optimization(self) -> None:
        """Stop ongoing optimization"""
        ...

class IAutoMLMetrics(Protocol):
    """Interface for AutoML metrics collection"""

    async def record_trial_metric(self, trial_id: str, metric_name: str, value: float) -> None:
        """Record a metric for a trial

        Args:
            trial_id: Trial identifier
            metric_name: Name of metric
            value: Metric value
        """
        ...

    async def get_optimization_summary(self) -> dict[str, Any]:
        """Get summary of optimization progress

        Returns:
            Summary with key metrics and progress
        """
        ...
