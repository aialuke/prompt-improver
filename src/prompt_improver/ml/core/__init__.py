"""Core ML Infrastructure

Central components for ML pipeline orchestration, training data management,
and model integration.

2025 Architecture:
- Decomposed god object into focused services with clean boundaries
- Direct facade pattern with MLModelServiceFacade as primary interface
"""
from .facade import MLModelServiceFacade
from .training_data_loader import TrainingDataLoader, get_training_data_stats

__all__ = ['TrainingDataLoader', 'get_training_data_stats', 'MLModelServiceFacade']
