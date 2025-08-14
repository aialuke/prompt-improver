"""Dimensionality reduction services following clean architecture patterns.

This package contains focused services that replace the monolithic dimensionality reducer:
- NeuralReducerService: Neural network architectures (autoencoders, VAE, transformers)
- ClassicalReducerService: Traditional algorithms (PCA, t-SNE, UMAP)
- ReductionEvaluatorService: Quality evaluation and metrics
- DataPreprocessorService: Data preprocessing and validation

Each service is protocol-based and <500 lines following 2025 clean architecture standards.
"""

from typing import Protocol, runtime_checkable
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "ReductionProtocol",
    "EvaluationProtocol", 
    "PreprocessorProtocol",
    "ReductionResult",
    "EvaluationMetrics",
    "PreprocessingResult"
]

@dataclass
class ReductionResult:
    """Result of dimensionality reduction operation."""
    method: str
    original_dimensions: int
    reduced_dimensions: int
    variance_preserved: float
    reconstruction_error: float
    processing_time: float
    quality_score: float
    transformed_data: np.ndarray
    transformer: Any
    feature_importance: Optional[np.ndarray] = None
    evaluation_metrics: Optional[Dict[str, float]] = None

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for dimensionality reduction."""
    variance_preservation: float
    clustering_preservation: float
    classification_preservation: Optional[float]
    neighborhood_preservation: float
    overall_quality: float
    processing_time: float

@dataclass
class PreprocessingResult:
    """Result of data preprocessing operations."""
    status: str
    features: np.ndarray
    info: Dict[str, Any]
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    preprocessing_time: float

@runtime_checkable
class ReductionProtocol(Protocol):
    """Protocol for dimensionality reduction implementations."""
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ReductionProtocol':
        """Fit the reduction model to data."""
        ...
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to reduced dimensions."""
        ...
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform data in one step."""
        ...
    
    def get_method_info(self) -> Dict[str, Any]:
        """Get information about the reduction method."""
        ...

@runtime_checkable
class EvaluationProtocol(Protocol):
    """Protocol for reduction quality evaluation."""
    
    def evaluate_quality(self, original: np.ndarray, reduced: np.ndarray, 
                        labels: Optional[np.ndarray] = None) -> EvaluationMetrics:
        """Evaluate quality of dimensionality reduction."""
        ...
    
    def compute_variance_preservation(self, original: np.ndarray, reduced: np.ndarray) -> float:
        """Compute variance preservation score."""
        ...
    
    def assess_clustering_preservation(self, original: np.ndarray, reduced: np.ndarray) -> float:
        """Assess how well clustering structure is preserved."""
        ...

@runtime_checkable
class PreprocessorProtocol(Protocol):
    """Protocol for data preprocessing operations."""
    
    def preprocess(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> PreprocessingResult:
        """Preprocess data for dimensionality reduction."""
        ...
    
    def validate_input(self, X: np.ndarray) -> Dict[str, Any]:
        """Validate input data quality and characteristics."""
        ...
    
    def scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features for optimal reduction performance."""
        ...