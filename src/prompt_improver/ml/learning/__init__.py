"""Learning Algorithms and Feature Engineering

Advanced learning algorithms for context awareness, failure analysis,
and insights generation, along with feature engineering components.
"""
from ..clustering.services.clustering_optimizer_facade import ClusteringOptimizerFacade as ContextClusteringEngine
from .algorithms import get_context_config as ContextConfig, get_context_learner as ContextLearner, get_insight_engine as InsightGenerationEngine, get_rule_analyzer as RuleAnalyzer
from .algorithms.context_aware_weighter import ContextAwareFeatureWeighter
from .clustering import ClusteringConfig
from .features import CompositeFeatureExtractor, ContextFeatureExtractor, DomainFeatureExtractor, FeatureExtractionConfig, LinguisticFeatureExtractor
from .quality.enhanced_scorer import EnhancedQualityScorer
__all__ = ['ContextLearner', 'ContextConfig', 'InsightGenerationEngine', 'RuleAnalyzer', 'CompositeFeatureExtractor', 'LinguisticFeatureExtractor', 'DomainFeatureExtractor', 'ContextFeatureExtractor', 'FeatureExtractionConfig', 'ContextAwareFeatureWeighter', 'ContextClusteringEngine', 'ClusteringConfig', 'EnhancedQualityScorer']
