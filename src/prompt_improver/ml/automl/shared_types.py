"""
Shared types for AutoML to avoid circular imports.

This module contains common types and protocols used by both orchestrator 
and callbacks modules, following 2025 best practices for dependency management.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol

class AutoMLOrchestratorProtocol(Protocol):
    """Protocol for AutoML orchestrator to avoid circular imports."""

    def get_current_trial_info(self) -> dict[str, Any]:
        """Get information about the current trial."""
        ...

    def update_trial_metrics(self, metrics: dict[str, float]) -> None:
        """Update metrics for the current trial."""
        ...

    def should_stop_trial(self) -> bool:
        """Check if the current trial should be stopped."""
        ...

class CallbackProtocol(Protocol):
    """Protocol for AutoML callbacks to avoid circular imports."""

    def on_trial_start(self, trial_info: dict[str, Any]) -> None:
        """Called when a trial starts."""
        ...

    def on_trial_end(self, trial_info: dict[str, Any], result: dict[str, Any]) -> None:
        """Called when a trial ends."""
        ...

    def on_optimization_start(self) -> None:
        """Called when optimization starts."""
        ...

    def on_optimization_end(self) -> None:
        """Called when optimization ends."""
        ...

class AnalyticsServiceProtocol(Protocol):
    """Protocol for analytics service to avoid circular imports."""

    def log_trial_metrics(self, trial_id: str, metrics: dict[str, float]) -> None:
        """Log metrics for a trial."""
        ...

    def get_trial_history(self, limit: int | None=None) -> list[dict[str, Any]]:
        """Get trial history."""
        ...

class TrialInfo:
    """Information about an AutoML trial."""

    def __init__(self, trial_id: str, parameters: dict[str, Any], state: str='running', metrics: dict[str, float] | None=None):
        self.trial_id = trial_id
        self.parameters = parameters
        self.state = state
        self.metrics = metrics or {}

class OptimizationResult:
    """Result of an AutoML optimization."""

    def __init__(self, best_trial_id: str, best_parameters: dict[str, Any], best_score: float, total_trials: int, optimization_time: float):
        self.best_trial_id = best_trial_id
        self.best_parameters = best_parameters
        self.best_score = best_score
        self.total_trials = total_trials
        self.optimization_time = optimization_time
