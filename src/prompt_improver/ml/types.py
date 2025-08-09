"""Type definitions for ML module.

This module provides comprehensive type definitions for machine learning components,
including proper numpy array typing and model interfaces.
"""
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple, TypeVar, Union, runtime_checkable
import numpy as np
from numpy.typing import NDArray
float_array = NDArray[np.float64]
int_array = NDArray[np.int64]
bool_array = NDArray[np.bool]
feature_array = NDArray[np.float64]
label_array = NDArray[np.int64]
T = TypeVar('T', bound=np.generic)
generic_array = NDArray[T]
features = float_array
labels = int_array
predictions = Union[int_array, float_array]
probabilities = float_array
embeddings = float_array
weights = float_array
cluster_labels = int_array
cluster_centers = float_array
distance_matrix = float_array
reduced_features = float_array
transform_matrix = float_array
model_config = Dict[str, Any]
hyper_parameters = Dict[str, Union[int, float, str, bool]]
metrics_dict = Dict[str, float]
progress_callback = Callable[[int, int], None]
metrics_callback = Callable[[metrics_dict], None]
pipeline_step = Tuple[str, Any]
pipeline_config = List[pipeline_step]

@runtime_checkable
class Transformer(Protocol):
    """Protocol for feature transformers."""

    def fit(self, X: features, y: Optional[labels]=None) -> 'Transformer':
        """Fit the transformer to data."""
        ...

    def transform(self, X: features) -> features:
        """Transform the features."""
        ...

    def fit_transform(self, X: features, y: Optional[labels]=None) -> features:
        """Fit and transform in one step."""
        ...

@runtime_checkable
class Clusterer(Protocol):
    """Protocol for clustering algorithms."""

    def fit(self, X: features) -> 'Clusterer':
        """Fit the clusterer to data."""
        ...

    def predict(self, X: features) -> cluster_labels:
        """Predict cluster labels."""
        ...

    def fit_predict(self, X: features) -> cluster_labels:
        """Fit and predict in one step."""
        ...

@runtime_checkable
class Optimizer(Protocol):
    """Protocol for optimization algorithms."""

    def optimize(self, features: features, labels: Optional[labels]=None, **kwargs: Any) -> Dict[str, Any]:
        """Optimize the given features."""
        ...

@runtime_checkable
class FeatureExtractor(Protocol):
    """Protocol for feature extraction components."""

    def extract(self, data: Any) -> features:
        """Extract features from raw data."""
        ...

@runtime_checkable
class ModelManager(Protocol):
    """Protocol for model management."""

    def load_model(self, model_id: str) -> Any:
        """Load a model by ID."""
        ...

    def save_model(self, model: Any, model_id: str) -> None:
        """Save a model with ID."""
        ...

    def get_pipeline(self) -> Optional[Any]:
        """Get the model pipeline."""
        ...

class TrainingBatch:
    """Type-safe container for training data batches."""

    def __init__(self, features: features, labels: labels, sample_weights: Optional[weights]=None, metadata: Optional[Dict[str, Any]]=None):
        self.features = features
        self.labels = labels
        self.sample_weights = sample_weights
        self.metadata = metadata or {}

class OptimizationResult:
    """Type-safe container for optimization results."""

    def __init__(self, optimized_features: features, metrics: metrics_dict, model_params: Optional[hyper_parameters]=None, execution_time: Optional[float]=None):
        self.optimized_features = optimized_features
        self.metrics = metrics
        self.model_params = model_params
        self.execution_time = execution_time

class ClusteringResult:
    """Type-safe container for clustering results."""

    def __init__(self, labels: cluster_labels, centers: Optional[cluster_centers]=None, metrics: Optional[metrics_dict]=None, n_clusters: Optional[int]=None):
        self.labels = labels
        self.centers = centers
        self.metrics = metrics or {}
        self.n_clusters = n_clusters or len(np.unique(labels))
SilhouetteScore = float
CalinskiHarabaszScore = float
DaviesBouldinScore = float
InertiaScore = float
ValidationSplit = Tuple[int_array, int_array]
CrossValidationFolds = List[ValidationSplit]

class MLError(Exception):
    """Base exception for ML module."""
    pass

class DimensionalityError(MLError):
    """Raised when dimensionality requirements are not met."""
    pass

class ConvergenceError(MLError):
    """Raised when algorithms fail to converge."""
    pass

class DataValidationError(MLError):
    """Raised when input data validation fails."""
    pass

def is_valid_features(arr: Any) -> bool:
    """Check if array is valid features array."""
    return isinstance(arr, np.ndarray) and arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating)

def is_valid_labels(arr: Any) -> bool:
    """Check if array is valid labels array."""
    return isinstance(arr, np.ndarray) and arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer)

def ensure_features(arr: Union[List, np.ndarray]) -> features:
    """Ensure input is proper features array."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def ensure_labels(arr: Union[List, np.ndarray]) -> labels:
    """Ensure input is proper labels array."""
    return np.asarray(arr, dtype=np.int64).ravel()
