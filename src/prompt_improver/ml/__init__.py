"""ML Pipeline - Unified Machine Learning Infrastructure

This module provides the core ML infrastructure for the prompt improvement system,
including training data management, model lifecycle, optimization algorithms,
and evaluation frameworks.
"""
from .automl.orchestrator import AutoMLOrchestrator
from .core.ml_integration import MLModelService
from .core.training_data_loader import TrainingDataLoader, get_training_data_stats
from .learning.algorithms import ContextLearner, InsightGenerationEngine, RuleAnalyzer
from .models.model_manager import ModelManager
from .models.production_registry import ProductionModelRegistry
from .optimization.algorithms.clustering_optimizer import ClusteringOptimizer
from .optimization.algorithms.dimensionality_reducer import AdvancedDimensionalityReducer as DimensionalityReducer
from .optimization.algorithms.multi_armed_bandit import MultiarmedBanditFramework
from .optimization.algorithms.rule_optimizer import RuleOptimizer
# NLTK manager removed - use EnglishNLTKManager from ml.learning.features.english_nltk_manager
__all__ = ['TrainingDataLoader', 'get_training_data_stats', 'MLModelService', 'ModelManager', 'ProductionModelRegistry', 'ContextLearner', 'InsightGenerationEngine', 'RuleAnalyzer', 'RuleOptimizer', 'MultiarmedBanditFramework', 'ClusteringOptimizer', 'DimensionalityReducer', 'AutoMLOrchestrator']
