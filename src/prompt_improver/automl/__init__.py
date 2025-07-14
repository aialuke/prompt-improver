"""
AutoML Module for Prompt Improver System
Implements 2025 best practices for automated machine learning

This module provides:
- Automated hyperparameter optimization using Optuna
- Real-time A/B testing integration
- Continuous learning and model management
- Multi-objective optimization with NSGA-II
- Automated experiment design and orchestration

Architecture follows 2025 AutoML patterns:
- Callback-based integration with ML frameworks
- Real-time feedback loops
- Persistence with RDBStorage patterns
- Multi-objective Pareto optimization
"""

from .orchestrator import AutoMLOrchestrator
from .callbacks import (
    AutoMLCallback,
    RealTimeAnalyticsCallback,
    ExperimentCallback,
    ModelSelectionCallback
)

__all__ = [
    "AutoMLOrchestrator",
    "AutoMLCallback", 
    "RealTimeAnalyticsCallback",
    "ExperimentCallback",
    "ModelSelectionCallback"
]

# Version following semantic versioning
__version__ = "1.0.0"