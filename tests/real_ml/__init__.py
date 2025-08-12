"""Real ML Testing Module

Provides lightweight real ML implementations for integration testing,
replacing mocks with actual model training and inference.
"""

from .fixtures import *
from .lightweight_models import (
    LightweightRegressor,
    LightweightTextClassifier,
    PatternDiscoveryEngine,
    RealMLflowService,
    RealMLService,
)
from .test_data_generators import (
    MLTrainingDataGenerator,
    PromptDataGenerator,
)

__all__ = [
    # Models
    "LightweightRegressor",
    "LightweightTextClassifier", 
    "PatternDiscoveryEngine",
    "RealMLflowService",
    "RealMLService",
    
    # Data generators
    "MLTrainingDataGenerator",
    "PromptDataGenerator",
    
    # All fixtures are imported via *
]