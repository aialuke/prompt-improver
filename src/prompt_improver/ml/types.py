"""Type definitions for ML module.

This module provides comprehensive type definitions for machine learning components,
including proper numpy array typing and model interfaces.
"""
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, TypeVar, Union, runtime_checkable, TYPE_CHECKING
from collections.abc import Callable
from prompt_improver.core.utils.lazy_ml_loader import get_numpy

if TYPE_CHECKING:
    from numpy.typing import NDArray
    import numpy as np
else:
    def _get_numpy_types():
        np = get_numpy()
        return np.typing.NDArray, np.float64, np.int64, np.bool_, np.generic
    
    NDArray, _float64, _int64, _bool_, _generic = _get_numpy_types()

# Type aliases using lazy loading
def _create_type_aliases():
    np = get_numpy()
    return {
        'float_array': NDArray[np.float64],
        'int_array': NDArray[np.int64], 
        'bool_array': NDArray[np.bool_],
        'feature_array': NDArray[np.float64],
        'label_array': NDArray[np.int64],
    }

# Lazy evaluation of types
if TYPE_CHECKING:
    float_array = NDArray[np.float64]
    int_array = NDArray[np.int64]
    bool_array = NDArray[np.bool_]
    feature_array = NDArray[np.float64]
    label_array = NDArray[np.int64]
    T = TypeVar('T', bound=np.generic)
    generic_array = NDArray[T]
else:
    # Runtime lazy loading
    _type_cache = None
    
    def _get_type_aliases():
        global _type_cache
        if _type_cache is None:
            np = get_numpy()
            _type_cache = {
                'float_array': NDArray[np.float64],
                'int_array': NDArray[np.int64],
                'bool_array': NDArray[np.bool_],
                'feature_array': NDArray[np.float64], 
                'label_array': NDArray[np.int64],
            }
        return _type_cache
    
    # Create module-level variables for runtime access
    def __getattr__(name: str):
        types = _get_type_aliases()
        if name in types:
            return types[name]
        elif name == 'T':
            np = get_numpy()
            return TypeVar('T', bound=np.generic)
        elif name == 'generic_array':
            np = get_numpy() 
            T = TypeVar('T', bound=np.generic)
            return NDArray[T]
        elif name in ['features', 'probabilities', 'embeddings', 'weights', 'distance_matrix', 'reduced_features', 'transform_matrix']:
            return types['float_array']
        elif name in ['labels', 'cluster_labels']:
            return types['int_array']
        elif name == 'cluster_centers':
            return types['float_array']
        elif name == 'predictions':
            return Union[types['int_array'], types['float_array']]
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
# Runtime type aliases - these will work via __getattr__ or direct assignment
if TYPE_CHECKING:
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
else:
    # These will be available through __getattr__ for runtime access
    pass
model_config = dict[str, Any]
hyper_parameters = dict[str, Union[int, float, str, bool]]
metrics_dict = dict[str, float]
progress_callback = Callable[[int, int], None]
metrics_callback = Callable[[metrics_dict], None]
pipeline_step = tuple[str, Any]
pipeline_config = list[pipeline_step]

@runtime_checkable
class Transformer(Protocol):
    """Protocol for feature transformers."""

    def fit(self, X: 'features', y: 'labels | None' = None) -> 'Transformer':
        """Fit the transformer to data."""
        ...

    def transform(self, X: 'features') -> 'features':
        """Transform the features."""
        ...

    def fit_transform(self, X: 'features', y: 'labels | None' = None) -> 'features':
        """Fit and transform in one step."""
        ...

@runtime_checkable
class Clusterer(Protocol):
    """Protocol for clustering algorithms."""

    def fit(self, X: 'features') -> 'Clusterer':
        """Fit the clusterer to data."""
        ...

    def predict(self, X: 'features') -> 'cluster_labels':
        """Predict cluster labels."""
        ...

    def fit_predict(self, X: 'features') -> 'cluster_labels':
        """Fit and predict in one step."""
        ...

@runtime_checkable
class Optimizer(Protocol):
    """Protocol for optimization algorithms."""

    def optimize(self, features: 'features', labels: 'labels | None' = None, **kwargs: Any) -> dict[str, Any]:
        """Optimize the given features."""
        ...

@runtime_checkable
class FeatureExtractor(Protocol):
    """Protocol for feature extraction components."""

    def extract(self, data: Any) -> 'features':
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

    def get_pipeline(self) -> Any | None:
        """Get the model pipeline."""
        ...

class TrainingBatch:
    """Type-safe container for training data batches."""

    def __init__(self, features: 'features', labels: 'labels', sample_weights: 'weights | None' = None, metadata: dict[str, Any] | None = None):
        self.features = features
        self.labels = labels
        self.sample_weights = sample_weights
        self.metadata = metadata or {}

class OptimizationResult:
    """Type-safe container for optimization results."""

    def __init__(self, optimized_features: 'features', metrics: 'metrics_dict', model_params: 'hyper_parameters | None' = None, execution_time: float | None = None):
        self.optimized_features = optimized_features
        self.metrics = metrics
        self.model_params = model_params
        self.execution_time = execution_time

class ClusteringResult:
    """Type-safe container for clustering results."""

    def __init__(self, labels: 'cluster_labels', centers: 'cluster_centers | None' = None, metrics: 'metrics_dict | None' = None, n_clusters: int | None = None):
        self.labels = labels
        self.centers = centers
        self.metrics = metrics or {}
        if n_clusters is not None:
            self.n_clusters = n_clusters
        else:
            np = get_numpy()
            self.n_clusters = len(np.unique(labels))
SilhouetteScore = float
CalinskiHarabaszScore = float
DaviesBouldinScore = float
InertiaScore = float
ValidationSplit = 'tuple[int_array, int_array]'
CrossValidationFolds = 'list[ValidationSplit]'

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
    np = get_numpy()
    return isinstance(arr, np.ndarray) and arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating)

def is_valid_labels(arr: Any) -> bool:
    """Check if array is valid labels array."""
    np = get_numpy()
    return isinstance(arr, np.ndarray) and arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer)

def ensure_features(arr: list | Any) -> Any:  # features type available via __getattr__
    """Ensure input is proper features array."""
    np = get_numpy()
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def ensure_labels(arr: list | Any) -> Any:  # labels type available via __getattr__
    """Ensure input is proper labels array."""
    np = get_numpy()
    return np.asarray(arr, dtype=np.int64).ravel()