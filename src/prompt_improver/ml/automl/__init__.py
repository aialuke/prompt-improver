"""AutoML Module for Prompt Improver System
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
from .callbacks import AutoMLCallback, ExperimentCallback, ModelSelectionCallback, RealTimeAnalyticsCallback
from .orchestrator import AutoMLOrchestrator
__all__ = ['AutoMLCallback', 'AutoMLOrchestrator', 'ExperimentCallback', 'ModelSelectionCallback', 'RealTimeAnalyticsCallback']
__version__ = '1.0.0'
