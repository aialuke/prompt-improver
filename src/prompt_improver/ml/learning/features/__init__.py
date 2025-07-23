"""Feature Engineering Components

Components for feature extraction and domain-specific processing.
Includes specialized extractors for different types of analysis.
"""

# Specialized feature extractors
from .linguistic_feature_extractor import LinguisticFeatureExtractor
from .domain_feature_extractor import DomainFeatureExtractor
from .context_feature_extractor import ContextFeatureExtractor
from .composite_feature_extractor import (
    CompositeFeatureExtractor,
    FeatureExtractionConfig,
    FeatureExtractorFactory
)

__all__ = [
    # Specialized extractors
    "LinguisticFeatureExtractor",
    "DomainFeatureExtractor",
    "ContextFeatureExtractor",
    "CompositeFeatureExtractor",
    "FeatureExtractionConfig",
    "FeatureExtractorFactory"
]