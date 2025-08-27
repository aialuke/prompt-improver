"""ML Pipeline - Unified Machine Learning Infrastructure

This module provides the core ML infrastructure for the prompt improvement system,
including training data management, model lifecycle, optimization algorithms,
and evaluation frameworks.
"""
from .automl.orchestrator import AutoMLOrchestrator
from .core import MLModelServiceFacade
from .core.training_data_loader import TrainingDataLoader, get_training_data_stats
# Heavy ML imports moved to lazy loading and TYPE_CHECKING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .learning.algorithms import get_context_learner, get_insight_engine, get_rule_analyzer
    from .clustering.services.clustering_optimizer_facade import ClusteringOptimizerFacade
from .models.model_manager import ModelManager
from .models.production_registry import ProductionModelRegistry
# Heavy clustering import moved to lazy loading

def get_clustering_optimizer():
    """Lazy load ClusteringOptimizerFacade when needed."""
    from .clustering.services.clustering_optimizer_facade import ClusteringOptimizerFacade
    return ClusteringOptimizerFacade

# Provide lazy alias
ClusteringOptimizer = get_clustering_optimizer
# Heavy dimensionality import moved to lazy loading  
def get_dimensionality_reducer():
    """Lazy load DimensionalityReducerFacade when needed."""
    from .dimensionality.services.dimensionality_reducer_facade import DimensionalityReducerFacade
    return DimensionalityReducerFacade

# Provide lazy alias
DimensionalityReducer = get_dimensionality_reducer
# Heavy optimization import moved to lazy loading
def get_multiarm_bandit():
    """Lazy load MultiarmedBanditFramework when needed."""
    from .optimization.algorithms.multi_armed_bandit import MultiarmedBanditFramework
    return MultiarmedBanditFramework

# Provide lazy alias  
MultiarmedBanditFramework = get_multiarm_bandit
from .optimization.algorithms.rule_optimizer import RuleOptimizer
# NLTK manager removed - use EnglishNLTKManager from ml.learning.features.english_nltk_manager
# Direct lazy loading imports - CLEAN BREAK, no backward compatibility
from .learning.algorithms import (
    get_context_learner as ContextLearner,
    get_insight_engine as InsightGenerationEngine, 
    get_rule_analyzer as RuleAnalyzer
)

__all__ = ['TrainingDataLoader', 'get_training_data_stats', 'MLModelServiceFacade', 'ModelManager', 'ProductionModelRegistry', 'ContextLearner', 'InsightGenerationEngine', 'RuleAnalyzer', 'RuleOptimizer', 'MultiarmedBanditFramework', 'ClusteringOptimizer', 'DimensionalityReducer', 'AutoMLOrchestrator']
