"""Clustering optimization services following clean architecture patterns.

This package contains focused services that replace the monolithic clustering optimizer:
- ClusteringAlgorithmService: HDBSCAN and UMAP execution
- ClusteringParameterService: Parameter optimization and grid search
- ClusteringEvaluatorService: Quality metrics and evaluation
- ClusteringPreprocessorService: Feature preprocessing and validation

Each service is protocol-based and <500 lines following 2025 clean architecture standards.
"""

from typing import Protocol, runtime_checkable
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "ClusteringAlgorithmProtocol",
    "ParameterOptimizationProtocol",
    "ClusteringEvaluationProtocol",
    "ClusteringPreprocessorProtocol",
    "ClusteringResult",
    "ClusteringMetrics",
    "OptimizationResult",
    "ClusteringPreprocessingResult"
]

@dataclass
class ClusteringResult:
    """Result of clustering operation."""
    cluster_labels: np.ndarray
    cluster_centers: Optional[np.ndarray]
    n_clusters: int
    silhouette_score: float
    algorithm_used: str
    processing_time: float
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class ClusteringMetrics:
    """Comprehensive clustering performance metrics."""
    n_clusters: int
    n_noise_points: int
    noise_ratio: float
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    quality_score: float
    processing_time_seconds: float
    memory_usage_mb: float
    convergence_achieved: bool
    stability_score: float

@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_time: float
    parameter_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]

@dataclass
class ClusteringPreprocessingResult:
    """Result of clustering preprocessing operations."""
    status: str
    features: np.ndarray
    info: Dict[str, Any]
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    preprocessing_time: float
    feature_importance: Optional[np.ndarray] = None

@runtime_checkable
class ClusteringAlgorithmProtocol(Protocol):
    """Protocol for clustering algorithm implementations."""
    
    def fit_predict(self, X: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit clustering model and predict cluster labels."""
        ...
    
    def get_cluster_centers(self, X: np.ndarray, labels: np.ndarray) -> Optional[np.ndarray]:
        """Compute cluster centers from features and labels."""
        ...
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about the clustering algorithm."""
        ...

@runtime_checkable
class ParameterOptimizationProtocol(Protocol):
    """Protocol for clustering parameter optimization."""
    
    def optimize_parameters(self, X: np.ndarray, param_grid: Dict[str, List[Any]], 
                          labels: Optional[np.ndarray] = None) -> OptimizationResult:
        """Optimize clustering parameters using grid search."""
        ...
    
    def get_adaptive_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        """Get adaptive parameters based on data characteristics."""
        ...
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameter combinations."""
        ...

@runtime_checkable
class ClusteringEvaluationProtocol(Protocol):
    """Protocol for clustering quality evaluation."""
    
    def assess_clustering_quality(self, X: np.ndarray, labels: np.ndarray, 
                                probabilities: Optional[np.ndarray] = None) -> ClusteringMetrics:
        """Assess comprehensive clustering quality."""
        ...
    
    def compute_stability_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute cluster stability score."""
        ...
    
    def evaluate_clustering_success(self, X: np.ndarray, metrics: ClusteringMetrics) -> Tuple[str, str]:
        """Evaluate if clustering was successful."""
        ...

@runtime_checkable
class ClusteringPreprocessorProtocol(Protocol):
    """Protocol for clustering preprocessing operations."""
    
    def preprocess_features(self, X: np.ndarray, labels: Optional[np.ndarray] = None) -> ClusteringPreprocessingResult:
        """Preprocess features for optimal clustering."""
        ...
    
    def validate_inputs(self, X: np.ndarray, labels: Optional[np.ndarray] = None, 
                       sample_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Validate input data for clustering."""
        ...
    
    def apply_dimensionality_reduction(self, X: np.ndarray) -> np.ndarray:
        """Apply UMAP dimensionality reduction for clustering."""
        ...