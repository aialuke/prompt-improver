"""Feature Engineering Components

Components for feature extraction and domain-specific processing.
Includes specialized extractors for different types of analysis.
"""
from .composite_feature_extractor import CompositeFeatureExtractor, FeatureExtractionConfig, FeatureExtractorFactory
from .context_feature_extractor import ContextFeatureExtractor
from .domain_feature_extractor import DomainFeatureExtractor
from .linguistic_feature_extractor import LinguisticFeatureExtractor
__all__ = ['LinguisticFeatureExtractor', 'DomainFeatureExtractor', 'ContextFeatureExtractor', 'CompositeFeatureExtractor', 'FeatureExtractionConfig', 'FeatureExtractorFactory']
