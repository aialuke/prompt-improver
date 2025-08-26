"""Classical dimensionality reduction algorithms service.

Implements traditional statistical and manifold learning methods including:
- PCA and variants (Incremental, Kernel)
- Independent Component Analysis (ICA)
- t-SNE and manifold learning methods
- UMAP (if available)
- Random projection methods

Follows clean architecture patterns with protocol-based interfaces.
"""

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

from typing import TYPE_CHECKING
from prompt_improver.core.utils.lazy_ml_loader import get_numpy, get_sklearn

if TYPE_CHECKING:
    from sklearn.decomposition import PCA, FastICA, IncrementalPCA, KernelPCA, TruncatedSVD
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.manifold import TSNE, Isomap
    from sklearn.preprocessing import StandardScaler
    from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
    import numpy as np
else:
    # Runtime lazy loading
    def _get_sklearn_imports():
        sklearn = get_sklearn()
        return (
            sklearn.decomposition.PCA,
            sklearn.decomposition.FastICA, 
            sklearn.decomposition.IncrementalPCA,
            sklearn.decomposition.KernelPCA,
            sklearn.decomposition.TruncatedSVD,
            sklearn.discriminant_analysis.LinearDiscriminantAnalysis,
            sklearn.manifold.TSNE,
            sklearn.manifold.Isomap,
            sklearn.preprocessing.StandardScaler,
            sklearn.random_projection.GaussianRandomProjection,
            sklearn.random_projection.SparseRandomProjection
        )
    
    (PCA, FastICA, IncrementalPCA, KernelPCA, TruncatedSVD, 
     LinearDiscriminantAnalysis, TSNE, Isomap, StandardScaler,
     GaussianRandomProjection, SparseRandomProjection) = _get_sklearn_imports()

from . import ReductionProtocol, ReductionResult

logger = logging.getLogger(__name__)

# Optional imports
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")

class ClassicalReducerService:
    """Classical dimensionality reduction service following clean architecture."""

    def __init__(self, method: str = "pca", n_components: int = 10, random_state: int = 42, **kwargs):
        """Initialize classical reducer service.
        
        Args:
            method: Reduction method to use
            n_components: Number of components for reduction
            random_state: Random state for reproducibility
            **kwargs: Method-specific parameters
        """
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.method_params = kwargs
        
        self.reducer = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.method_registry = self._initialize_method_registry()
        
        if method not in self.method_registry:
            available_methods = list(self.method_registry.keys())
            raise ValueError(f"Unknown method '{method}'. Available: {available_methods}")
        
        logger.info(f"ClassicalReducerService initialized: {method} with {n_components} components")

    def _initialize_method_registry(self) -> Dict[str, Dict[str, Any]]:
        """Initialize registry of available reduction methods."""
        methods = {
            "pca": {
                "class": PCA,
                "type": "linear",
                "supervised": False,
                "scalable": True,
                "preserves_variance": True,
                "params": {
                    "n_components": self.n_components,
                    "random_state": self.random_state,
                    "svd_solver": "auto"
                }
            },
            "incremental_pca": {
                "class": IncrementalPCA,
                "type": "linear",
                "supervised": False,
                "scalable": True,
                "preserves_variance": True,
                "params": {
                    "n_components": self.n_components,
                    "batch_size": min(1000, max(100, 32))
                }
            },
            "kernel_pca": {
                "class": KernelPCA,
                "type": "nonlinear",
                "supervised": False,
                "scalable": False,
                "preserves_variance": False,
                "params": {
                    "n_components": self.n_components,
                    "kernel": "rbf",
                    "gamma": None,
                    "random_state": self.random_state
                }
            },
            "ica": {
                "class": FastICA,
                "type": "linear",
                "supervised": False,
                "scalable": True,
                "preserves_variance": False,
                "params": {
                    "n_components": self.n_components,
                    "random_state": self.random_state,
                    "max_iter": 1000
                }
            },
            "truncated_svd": {
                "class": TruncatedSVD,
                "type": "linear",
                "supervised": False,
                "scalable": True,
                "preserves_variance": True,
                "params": {
                    "n_components": self.n_components,
                    "random_state": self.random_state
                }
            },
            "lda": {
                "class": LinearDiscriminantAnalysis,
                "type": "linear",
                "supervised": True,
                "scalable": True,
                "preserves_variance": False,
                "params": {
                    "n_components": min(self.n_components, 2)  # LDA limited by n_classes-1
                }
            },
            "tsne": {
                "class": TSNE,
                "type": "nonlinear",
                "supervised": False,
                "scalable": False,
                "preserves_variance": False,
                "params": {
                    "n_components": min(self.n_components, 3),  # t-SNE typically 2-3D
                    "perplexity": 30.0,
                    "early_exaggeration": 12.0,
                    "random_state": self.random_state,
                    "n_jobs": 1
                }
            },
            "isomap": {
                "class": Isomap,
                "type": "nonlinear",
                "supervised": False,
                "scalable": False,
                "preserves_variance": False,
                "params": {
                    "n_components": self.n_components,
                    "n_neighbors": 10,
                    "n_jobs": 1
                }
            },
            "gaussian_rp": {
                "class": GaussianRandomProjection,
                "type": "linear",
                "supervised": False,
                "scalable": True,
                "preserves_variance": False,
                "params": {
                    "n_components": self.n_components,
                    "random_state": self.random_state
                }
            },
            "sparse_rp": {
                "class": SparseRandomProjection,
                "type": "linear",
                "supervised": False,
                "scalable": True,
                "preserves_variance": False,
                "params": {
                    "n_components": self.n_components,
                    "random_state": self.random_state
                }
            }
        }

        # Add UMAP if available
        if UMAP_AVAILABLE:
            methods["umap"] = {
                "class": umap.UMAP,
                "type": "nonlinear",
                "supervised": False,
                "scalable": True,
                "preserves_variance": False,
                "params": {
                    "n_components": self.n_components,
                    "n_neighbors": 15,
                    "min_dist": 0.1,
                    "metric": "euclidean",
                    "random_state": self.random_state
                }
            }

        # Update with user-provided parameters
        for method_name, method_info in methods.items():
            if method_name == self.method:
                method_info["params"].update(self.method_params)

        return methods

    def fit(self, X: get_numpy().ndarray, y: Optional[get_numpy().ndarray] = None) -> 'ClassicalReducerService':
        """Fit the classical dimensionality reducer."""
        start_time = time.time()
        
        method_info = self.method_registry[self.method]
        method_class = method_info["class"]
        method_params = method_info["params"].copy()

        # Validate supervised methods
        if method_info["supervised"] and y is None:
            raise ValueError(f"Method '{self.method}' requires labels (y parameter)")

        # Adjust parameters based on data characteristics
        method_params = self._adjust_parameters_for_data(X, method_params, method_info)

        # Initialize and fit reducer
        self.reducer = method_class(**method_params)

        # Scale data for most methods (except some that handle it internally)
        if self.method not in ["tsne", "umap"]:  # These handle scaling internally
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
            self.scaler = None  # Don't use scaler for these methods

        # Fit the reducer
        if method_info["supervised"] and y is not None:
            self.reducer.fit(X_scaled, y)
        else:
            self.reducer.fit(X_scaled)

        self.is_fitted = True
        training_time = time.time() - start_time
        
        logger.info(f"Classical reducer '{self.method}' fitted in {training_time:.2f}s")
        return self

    def transform(self, X: get_numpy().ndarray) -> get_numpy().ndarray:
        """Transform data to lower dimensional space."""
        if not self.is_fitted:
            raise ValueError("Reducer must be fitted before transform")

        # Apply scaling if scaler was used during fitting
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        # Transform data
        X_reduced = self.reducer.transform(X_scaled)
        
        return X_reduced

    def fit_transform(self, X: get_numpy().ndarray, y: Optional[get_numpy().ndarray] = None) -> get_numpy().ndarray:
        """Fit the reducer and transform the data."""
        return self.fit(X, y).transform(X)

    def get_method_info(self) -> Dict[str, Any]:
        """Get information about the reduction method."""
        method_info = self.method_registry[self.method].copy()
        method_info.update({
            "method_name": self.method,
            "n_components": self.n_components,
            "is_fitted": self.is_fitted,
            "has_scaler": self.scaler is not None,
            "umap_available": UMAP_AVAILABLE
        })
        return method_info

    def get_feature_importance(self) -> Optional[get_numpy().ndarray]:
        """Get feature importance if available from the fitted reducer."""
        if not self.is_fitted:
            return None

        # For linear methods like PCA, use component magnitudes
        if hasattr(self.reducer, "components_"):
            return get_numpy().abs(self.reducer.components_).mean(axis=0)
        elif hasattr(self.reducer, "feature_importances_"):
            return self.reducer.feature_importances_
        else:
            return None

    def get_explained_variance_ratio(self) -> Optional[get_numpy().ndarray]:
        """Get explained variance ratio if available."""
        if not self.is_fitted:
            return None

        if hasattr(self.reducer, "explained_variance_ratio_"):
            return self.reducer.explained_variance_ratio_
        else:
            return None

    def get_reconstruction_error(self, X: get_numpy().ndarray) -> float:
        """Calculate reconstruction error for methods that support it."""
        if not self.is_fitted:
            return 0.0

        try:
            # For methods with inverse_transform
            if hasattr(self.reducer, "inverse_transform"):
                X_transformed = self.transform(X)
                X_reconstructed = self.reducer.inverse_transform(X_transformed)
                
                # Apply inverse scaling if scaler was used
                if self.scaler is not None:
                    X_reconstructed = self.scaler.inverse_transform(X_reconstructed)
                
                error = get_numpy().mean(get_numpy().square(X - X_reconstructed))
                return float(error)

            # For PCA-like methods, use explained variance
            if hasattr(self.reducer, "explained_variance_ratio_"):
                return float(1.0 - get_numpy().sum(self.reducer.explained_variance_ratio_))

            return 0.5  # Default moderate error for methods without reconstruction

        except Exception as e:
            logger.warning(f"Could not calculate reconstruction error: {e}")
            return 0.5

    def _adjust_parameters_for_data(self, X: get_numpy().ndarray, params: Dict[str, Any], 
                                  method_info: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust method parameters based on data characteristics."""
        n_samples, n_features = X.shape
        adjusted_params = params.copy()

        # Adjust n_components based on data size
        max_components = min(n_samples - 1, n_features, self.n_components)
        
        if "n_components" in adjusted_params:
            adjusted_params["n_components"] = max_components

        # Method-specific adjustments
        if self.method == "lda":
            # LDA is limited by number of classes
            if "n_components" in adjusted_params:
                # Conservative estimate - will be adjusted when labels are provided
                adjusted_params["n_components"] = min(adjusted_params["n_components"], max_components)

        elif self.method == "umap" and UMAP_AVAILABLE:
            # Adjust UMAP parameters for data size
            if n_samples < 50:
                adjusted_params["n_neighbors"] = min(adjusted_params["n_neighbors"], n_samples // 3)
            
        elif self.method == "tsne":
            # Adjust t-SNE perplexity for small datasets
            if n_samples < 100:
                adjusted_params["perplexity"] = min(adjusted_params["perplexity"], n_samples // 4)

        elif self.method == "kernel_pca":
            # Adjust gamma for RBF kernel based on feature dimensionality
            if adjusted_params["gamma"] is None:
                adjusted_params["gamma"] = 1.0 / n_features

        return adjusted_params

    def get_available_methods(self) -> List[str]:
        """Get list of available reduction methods."""
        return list(self.method_registry.keys())

    def is_method_suitable(self, method: str, n_samples: int, n_features: int, 
                          has_labels: bool = False) -> Dict[str, Any]:
        """Check if a method is suitable for the given data characteristics."""
        if method not in self.method_registry:
            return {"suitable": False, "reason": "Method not available"}

        method_info = self.method_registry[method]

        # Check if supervised method has labels
        if method_info["supervised"] and not has_labels:
            return {"suitable": False, "reason": "Supervised method requires labels"}

        # Check scalability for large datasets
        if not method_info["scalable"] and n_samples > 5000:
            return {"suitable": False, "reason": "Method not suitable for large datasets"}

        # Check specific method constraints
        if method == "lda":
            return {"suitable": True, "note": "Will be limited by number of classes"}
        
        if method in ["tsne", "isomap"] and n_samples > 10000:
            return {"suitable": False, "reason": "Method too slow for very large datasets"}

        return {"suitable": True, "reason": "Method is suitable"}

class ClassicalReducerServiceFactory:
    """Factory for creating classical reducer services with optimal configurations."""

    @staticmethod
    def create_pca_service(n_components: int = 10, **kwargs) -> ClassicalReducerService:
        """Create PCA service with optimized parameters."""
        return ClassicalReducerService(
            method="pca",
            n_components=n_components,
            **kwargs
        )

    @staticmethod
    def create_umap_service(n_components: int = 10, n_neighbors: int = 15, 
                           min_dist: float = 0.1, **kwargs) -> ClassicalReducerService:
        """Create UMAP service with optimized parameters."""
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
            
        return ClassicalReducerService(
            method="umap",
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            **kwargs
        )

    @staticmethod
    def create_tsne_service(n_components: int = 2, perplexity: float = 30.0, **kwargs) -> ClassicalReducerService:
        """Create t-SNE service with optimized parameters."""
        return ClassicalReducerService(
            method="tsne",
            n_components=min(n_components, 3),  # t-SNE typically 2-3D
            perplexity=perplexity,
            **kwargs
        )

    @staticmethod
    def create_best_method_service(n_samples: int, n_features: int, 
                                 has_labels: bool = False, **kwargs) -> ClassicalReducerService:
        """Create service with the best method for given data characteristics."""
        # Heuristic method selection
        if has_labels and n_samples < 10000:
            method = "lda"
        elif n_features > 50 and UMAP_AVAILABLE:
            method = "umap"
        elif n_samples > 5000:
            method = "pca"  # Fast and scalable
        else:
            method = "pca"  # Safe default
        
        return ClassicalReducerService(method=method, **kwargs)