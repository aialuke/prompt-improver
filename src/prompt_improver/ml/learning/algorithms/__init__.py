"""Learning Algorithms

Advanced learning algorithms for context awareness, failure analysis,
insights generation, and rule effectiveness analysis.

Uses lazy loading to prevent heavy ML imports at module level.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context_learner import ContextConfig, ContextLearner
    from .insight_engine import InsightGenerationEngine
    from .pattern_detector import PatternDetector
    from .rule_analyzer import RuleEffectivenessAnalyzer as RuleAnalyzer

def get_context_learner():
    """Lazy load ContextLearner when needed."""
    from .context_learner import ContextLearner
    return ContextLearner

def get_context_config():
    """Lazy load ContextConfig when needed."""
    from .context_learner import ContextConfig
    return ContextConfig

def get_insight_engine():
    """Lazy load InsightGenerationEngine when needed."""
    from .insight_engine import InsightGenerationEngine
    return InsightGenerationEngine

def get_pattern_detector():
    """Lazy load PatternDetector when needed."""
    from .pattern_detector import PatternDetector
    return PatternDetector

def get_rule_analyzer():
    """Lazy load RuleEffectivenessAnalyzer when needed."""
    from .rule_analyzer import RuleEffectivenessAnalyzer as RuleAnalyzer
    return RuleAnalyzer

__all__ = [
    'get_context_learner', 
    'get_context_config', 
    'get_pattern_detector', 
    'get_insight_engine', 
    'get_rule_analyzer'
]
