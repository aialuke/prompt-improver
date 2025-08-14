"""Core ML Infrastructure

Central components for ML pipeline orchestration, training data management,
and model integration.

MIGRATION NOTE: MLModelService now points to the unified facade for better architecture.
"""
from .facade import MLModelServiceFacade as MLModelService
from .training_data_loader import TrainingDataLoader, get_training_data_stats

# Export both facade and legacy name for compatibility
from .facade import MLModelServiceFacade

__all__ = ['TrainingDataLoader', 'get_training_data_stats', 'MLModelService', 'MLModelServiceFacade']
