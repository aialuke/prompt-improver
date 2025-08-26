"""Real ML Testing Module

Provides lightweight real ML implementations for integration testing,
replacing mocks with actual model training and inference.
"""

from tests.real_ml.fixtures import *
from tests.real_ml.lightweight_models import (
    LightweightRegressor,
    LightweightTextClassifier,
    PatternDiscoveryEngine,
    RealMLflowService,
    RealMLService,
)
from tests.real_ml.test_data_generators import (
    MLTrainingDataGenerator,
    PromptDataGenerator,
)

__all__ = [
    # Models
    "LightweightRegressor",
    "LightweightTextClassifier",
    # Data generators
    "MLTrainingDataGenerator",
    "PatternDiscoveryEngine",
    "PromptDataGenerator",
    "RealMLService",
    "RealMLflowService",

    # All fixtures are imported via *
]
