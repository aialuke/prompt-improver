"""Learning Algorithms and Feature Engineering

Advanced learning algorithms for context awareness, failure analysis,
and insights generation, along with feature engineering components.
"""

# Learning algorithms
from .algorithms import ContextLearner, FailureAnalyzer, InsightGenerationEngine, RuleAnalyzer

# Core learning components
from .algorithms import ContextLearner, ContextConfig
from .features import (
    CompositeFeatureExtractor,
    LinguisticFeatureExtractor,
    DomainFeatureExtractor,
    ContextFeatureExtractor,
    FeatureExtractionConfig
)
# ContextClusteringEngine merged into ClusteringOptimizer - use enhanced version
from ..optimization.algorithms.clustering_optimizer import ClusteringOptimizer as ContextClusteringEngine
from .clustering import ClusteringConfig

# Feature engineering
from .algorithms.context_aware_weighter import ContextAwareFeatureWeighter

# Quality assessment
from .quality.enhanced_scorer import EnhancedQualityScorer

__all__ = [
    # Algorithms
    "ContextLearner",
    "ContextConfig",
    "FailureAnalyzer",
    "InsightGenerationEngine",
    "RuleAnalyzer",
    # Features
    "CompositeFeatureExtractor",
    "LinguisticFeatureExtractor",
    "DomainFeatureExtractor",
    "ContextFeatureExtractor",
    "FeatureExtractionConfig",
    "ContextAwareFeatureWeighter",
    # Clustering
    "ContextClusteringEngine",
    "ClusteringConfig",
    # Quality
    "EnhancedQualityScorer",
]