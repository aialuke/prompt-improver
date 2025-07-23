"""ML Pipeline - Unified Machine Learning Infrastructure

This module provides the core ML infrastructure for the prompt improvement system,
including training data management, model lifecycle, optimization algorithms,
and evaluation frameworks.
"""

# Core ML functionality
from .core.training_data_loader import TrainingDataLoader, get_training_data_stats
from .core.ml_integration import MLModelService

# Model management
from .models.model_manager import ModelManager
from .models.production_registry import ProductionModelRegistry

# Learning algorithms
from .learning.algorithms import ContextLearner, FailureAnalyzer, InsightGenerationEngine, RuleAnalyzer

# Optimization algorithms
from .optimization.algorithms.rule_optimizer import RuleOptimizer
from .optimization.algorithms.multi_armed_bandit import MultiarmedBanditFramework
from .optimization.algorithms.clustering_optimizer import ClusteringOptimizer
from .optimization.algorithms.dimensionality_reducer import AdvancedDimensionalityReducer as DimensionalityReducer

# AutoML
from .automl.orchestrator import AutoMLOrchestrator

# Utilities
from .utils.nltk_manager import NLTKResourceManager

__all__ = [
    # Core
    "TrainingDataLoader",
    "get_training_data_stats", 
    "MLModelService",
    # Models
    "ModelManager",
    "ProductionModelRegistry",
    # Learning
    "ContextLearner",
    "FailureAnalyzer", 
    "InsightGenerationEngine",
    "RuleAnalyzer",
    # Optimization
    "RuleOptimizer",
    "MultiarmedBanditFramework",
    "ClusteringOptimizer",
    "DimensionalityReducer",
    # AutoML
    "AutoMLOrchestrator",
    # Utilities
    "NLTKResourceManager",
]