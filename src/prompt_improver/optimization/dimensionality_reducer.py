"""Advanced Dimensionality Reduction for High-Dimensional Linguistic Features.

Implements multiple dimensionality reduction techniques with intelligent method selection
and variance preservation optimization for 31-dimensional linguistic feature vectors.
"""

import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA, FastICA, KernelPCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold

from sklearn.manifold import TSNE, Isomap

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

# Advanced dimensionality reduction imports
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")

try:
    from sklearn.decomposition import FactorAnalysis

    FACTOR_ANALYSIS_AVAILABLE = True
except ImportError:
    FACTOR_ANALYSIS_AVAILABLE = False
    FactorAnalysis = None
    warnings.warn("FactorAnalysis not available in this sklearn version")

logger = logging.getLogger(__name__)


@dataclass
class DimensionalityConfig:
    """Configuration for dimensionality reduction optimization."""

    # Target dimensionality
    target_dimensions: int = 10
    min_dimensions: int = 5
    max_dimensions: int = 20
    variance_threshold: float = 0.95  # Minimum variance to preserve

    # Method selection
    auto_method_selection: bool = True
    preferred_methods: list[str] = None  # ['pca', 'umap', 'lda', 'ica', 'kernel_pca']
    enable_manifold_learning: bool = True
    enable_feature_selection: bool = True

    # Performance optimization
    fast_mode: bool = False  # Use faster but potentially less accurate methods
    memory_efficient: bool = True
    n_jobs: int = -1
    random_state: int = 42

    # Evaluation criteria
    preservation_metrics: list[str] = (
        None  # ['variance', 'structure', 'clustering', 'classification']
    )
    cross_validation_folds: int = 3
    clustering_quality_weight: float = 0.4
    classification_quality_weight: float = 0.3
    variance_preservation_weight: float = 0.3

    # Method-specific parameters
    pca_solver: str = "auto"  # 'auto', 'full', 'arpack', 'randomized'
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"

    kernel_pca_kernel: str = "rbf"  # 'linear', 'poly', 'rbf', 'sigmoid', 'cosine'
    kernel_pca_gamma: float | None = None

    tsne_perplexity: float = 30.0
    tsne_early_exaggeration: float = 12.0

    # Feature selection parameters
    feature_selection_method: str = (
        "mutual_info"  # 'f_score', 'mutual_info', 'rfe', 'variance'
    )
    feature_selection_k: int = 15

    # Caching and persistence
    enable_caching: bool = True
    cache_transformations: bool = True
    save_models: bool = False
    model_save_path: str | None = None


@dataclass
class ReductionResult:
    """Result of dimensionality reduction process."""

    method: str
    original_dimensions: int
    reduced_dimensions: int
    variance_preserved: float
    reconstruction_error: float
    processing_time: float
    quality_score: float
    transformed_data: np.ndarray
    transformer: Any
    feature_importance: np.ndarray | None = None
    evaluation_metrics: dict[str, float] | None = None


class AdvancedDimensionalityReducer:
    """Advanced dimensionality reduction with intelligent method selection."""

    def __init__(self, config: DimensionalityConfig | None = None):
        """Initialize dimensionality reducer.

        Args:
            config: Configuration for dimensionality reduction
        """
        self.config = config or DimensionalityConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize method registry
        self.available_methods = self._initialize_method_registry()

        # Performance tracking
        self.reduction_history: list[ReductionResult] = []
        self.method_performance: dict[str, list[float]] = {}

        # Caching
        self.cache: dict[str, ReductionResult] = {}
        self.scaler: Any | None = None

        # Preprocessing pipeline
        self.preprocessing_pipeline: Pipeline | None = None

        self.logger.info(
            f"Advanced dimensionality reducer initialized with {len(self.available_methods)} methods"
        )

    def _initialize_method_registry(self) -> dict[str, dict[str, Any]]:
        """Initialize registry of available dimensionality reduction methods."""
        methods = {
            "pca": {
                "class": PCA,
                "type": "linear",
                "manifold": False,
                "supervised": False,
                "scalable": True,
                "preserves_variance": True,
                "params": {
                    "n_components": self.config.target_dimensions,
                    "solver": self.config.pca_solver,
                    "random_state": self.config.random_state,
                },
            },
            "kernel_pca": {
                "class": KernelPCA,
                "type": "nonlinear",
                "manifold": False,
                "supervised": False,
                "scalable": False,
                "preserves_variance": False,
                "params": {
                    "n_components": self.config.target_dimensions,
                    "kernel": self.config.kernel_pca_kernel,
                    "gamma": self.config.kernel_pca_gamma,
                    "random_state": self.config.random_state,
                    "n_jobs": self.config.n_jobs
                    if not self.config.memory_efficient
                    else 1,
                },
            },
            "ica": {
                "class": FastICA,
                "type": "linear",
                "manifold": False,
                "supervised": False,
                "scalable": True,
                "preserves_variance": False,
                "params": {
                    "n_components": self.config.target_dimensions,
                    "random_state": self.config.random_state,
                    "max_iter": 1000,
                },
            },
            "truncated_svd": {
                "class": TruncatedSVD,
                "type": "linear",
                "manifold": False,
                "supervised": False,
                "scalable": True,
                "preserves_variance": True,
                "params": {
                    "n_components": self.config.target_dimensions,
                    "random_state": self.config.random_state,
                },
            },
            "lda": {
                "class": LinearDiscriminantAnalysis,
                "type": "linear",
                "manifold": False,
                "supervised": True,
                "scalable": True,
                "preserves_variance": False,
                "params": {
                    "n_components": min(
                        self.config.target_dimensions, 2
                    )  # LDA limited by n_classes-1
                },
            },
        }

        # Add UMAP if available
        if UMAP_AVAILABLE:
            methods["umap"] = {
                "class": umap.UMAP,
                "type": "nonlinear",
                "manifold": True,
                "supervised": False,
                "scalable": True,
                "preserves_variance": False,
                "params": {
                    "n_components": self.config.target_dimensions,
                    "n_neighbors": self.config.umap_n_neighbors,
                    "min_dist": self.config.umap_min_dist,
                    "metric": self.config.umap_metric,
                    "random_state": self.config.random_state,
                    "n_jobs": 1 if self.config.memory_efficient else self.config.n_jobs,
                },
            }

        # Add manifold learning methods if enabled
        if self.config.enable_manifold_learning and not self.config.fast_mode:
            methods.update({
                "tsne": {
                    "class": TSNE,
                    "type": "nonlinear",
                    "manifold": True,
                    "supervised": False,
                    "scalable": False,
                    "preserves_variance": False,
                    "params": {
                        "n_components": min(
                            self.config.target_dimensions, 3
                        ),  # t-SNE typically 2-3D
                        "perplexity": self.config.tsne_perplexity,
                        "early_exaggeration": self.config.tsne_early_exaggeration,
                        "random_state": self.config.random_state,
                        "n_jobs": 1
                        if self.config.memory_efficient
                        else self.config.n_jobs,
                    },
                },
                "isomap": {
                    "class": Isomap,
                    "type": "nonlinear",
                    "manifold": True,
                    "supervised": False,
                    "scalable": False,
                    "preserves_variance": False,
                    "params": {
                        "n_components": self.config.target_dimensions,
                        "n_neighbors": 10,
                        "n_jobs": 1
                        if self.config.memory_efficient
                        else self.config.n_jobs,
                    },
                },
            })

        # Add random projection methods for fast mode
        if self.config.fast_mode:
            methods.update({
                "gaussian_rp": {
                    "class": GaussianRandomProjection,
                    "type": "linear",
                    "manifold": False,
                    "supervised": False,
                    "scalable": True,
                    "preserves_variance": False,
                    "params": {
                        "n_components": self.config.target_dimensions,
                        "random_state": self.config.random_state,
                    },
                },
                "sparse_rp": {
                    "class": SparseRandomProjection,
                    "type": "linear",
                    "manifold": False,
                    "supervised": False,
                    "scalable": True,
                    "preserves_variance": False,
                    "params": {
                        "n_components": self.config.target_dimensions,
                        "random_state": self.config.random_state,
                    },
                },
            })

        return methods

    async def reduce_dimensions(
        self, X: np.ndarray, y: np.ndarray | None = None, method: str | None = None
    ) -> ReductionResult:
        """Reduce dimensionality of input data with optimal method selection.

        Args:
            X: Input feature matrix (n_samples, n_features)
            y: Optional target labels for supervised methods
            method: Specific method to use (if None, auto-select best method)

        Returns:
            ReductionResult with transformed data and evaluation metrics
        """
        start_time = time.time()

        self.logger.info(
            f"Starting dimensionality reduction: {X.shape[0]} samples x {X.shape[1]} features"
        )

        try:
            # Input validation
            if X.ndim != 2:
                raise ValueError("Input data must be 2D array")

            if X.shape[1] <= self.config.target_dimensions:
                self.logger.warning(
                    f"Input dimensionality ({X.shape[1]}) already <= target ({self.config.target_dimensions})"
                )
                return ReductionResult(
                    method="identity",
                    original_dimensions=X.shape[1],
                    reduced_dimensions=X.shape[1],
                    variance_preserved=1.0,
                    reconstruction_error=0.0,
                    processing_time=time.time() - start_time,
                    quality_score=1.0,
                    transformed_data=X.copy(),
                    transformer=None,
                )

            # Preprocessing
            X_processed = await self._preprocess_data(X)

            # Method selection
            if method is None:
                if self.config.auto_method_selection:
                    method = await self._select_optimal_method(X_processed, y)
                else:
                    method = (
                        self.config.preferred_methods[0]
                        if self.config.preferred_methods
                        else "pca"
                    )

            # Validate method availability
            if method not in self.available_methods:
                self.logger.warning(
                    f"Method '{method}' not available, falling back to PCA"
                )
                method = "pca"

            # Check cache
            cache_key = self._generate_cache_key(X, method)
            if self.config.enable_caching and cache_key in self.cache:
                self.logger.debug(f"Using cached result for method '{method}'")
                return self.cache[cache_key]

            # Perform dimensionality reduction
            result = await self._apply_reduction_method(X_processed, y, method)

            # Evaluate result quality
            evaluation_metrics = await self._evaluate_reduction_quality(
                X_processed, result.transformed_data, y
            )
            result.evaluation_metrics = evaluation_metrics
            result.quality_score = self._compute_overall_quality_score(
                result, evaluation_metrics
            )

            # Cache result
            if self.config.enable_caching:
                self.cache[cache_key] = result

            # Update performance tracking
            self._update_performance_tracking(result)

            self.logger.info(
                f"Dimensionality reduction completed with method '{method}': "
                f"{result.original_dimensions}→{result.reduced_dimensions} "
                f"(quality: {result.quality_score:.3f}, time: {result.processing_time:.2f}s)"
            )

            return result

        except Exception as e:
            self.logger.error(f"Dimensionality reduction failed: {e}", exc_info=True)
            raise

    async def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """Preprocess data for dimensionality reduction."""
        if self.scaler is None:
            # Use RobustScaler for better handling of outliers
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        # Remove constant features
        variance_selector = VarianceThreshold(threshold=1e-6)
        X_processed = variance_selector.fit_transform(X_scaled)

        if X_processed.shape[1] < X.shape[1]:
            self.logger.info(
                f"Removed {X.shape[1] - X_processed.shape[1]} constant features"
            )

        return X_processed

    async def _select_optimal_method(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> str:
        """Intelligently select the optimal dimensionality reduction method."""
        n_samples, n_features = X.shape

        # Filter methods based on data characteristics and constraints
        suitable_methods = []

        for method_name, method_info in self.available_methods.items():
            # Skip supervised methods if no labels provided
            if method_info["supervised"] and y is None:
                continue

            # Skip non-scalable methods for large datasets
            if not method_info["scalable"] and n_samples > 1000:
                continue

            # Skip manifold methods for very high-dimensional data in fast mode
            if method_info["manifold"] and n_features > 50 and self.config.fast_mode:
                continue

            # Adjust LDA components based on number of classes
            if method_name == "lda" and y is not None:
                n_classes = len(np.unique(y))
                if n_classes <= 1:
                    continue  # Skip LDA if only one class
                method_info["params"]["n_components"] = min(
                    self.config.target_dimensions, n_classes - 1
                )

            suitable_methods.append(method_name)

        if not suitable_methods:
            self.logger.warning("No suitable methods found, falling back to PCA")
            return "pca"

        # If only one method is suitable, use it
        if len(suitable_methods) == 1:
            return suitable_methods[0]

        # For multiple suitable methods, use heuristics based on data characteristics

        # Prefer faster methods for large datasets
        if n_samples > 5000 or self.config.fast_mode:
            fast_methods = [
                m
                for m in suitable_methods
                if m in ["pca", "truncated_svd", "gaussian_rp", "sparse_rp"]
            ]
            if fast_methods:
                return fast_methods[0]

        # Prefer supervised methods when labels are available
        if y is not None:
            supervised_methods = [
                m for m in suitable_methods if self.available_methods[m]["supervised"]
            ]
            if supervised_methods:
                return supervised_methods[0]

        # Prefer variance-preserving methods for interpretability
        if self.config.variance_preservation_weight > 0.5:
            variance_methods = [
                m
                for m in suitable_methods
                if self.available_methods[m]["preserves_variance"]
            ]
            if variance_methods:
                return variance_methods[0]

        # Prefer UMAP for general-purpose manifold learning
        if "umap" in suitable_methods and UMAP_AVAILABLE:
            return "umap"

        # Default to PCA as most reliable
        return "pca" if "pca" in suitable_methods else suitable_methods[0]

    async def _apply_reduction_method(
        self, X: np.ndarray, y: np.ndarray | None, method: str
    ) -> ReductionResult:
        """Apply specific dimensionality reduction method."""
        start_time = time.time()

        method_info = self.available_methods[method]
        method_class = method_info["class"]
        method_params = method_info["params"].copy()

        try:
            # Initialize reducer
            reducer = method_class(**method_params)

            # Fit and transform
            if method_info["supervised"] and y is not None:
                X_reduced = reducer.fit_transform(X, y)
            else:
                X_reduced = reducer.fit_transform(X)

            # Calculate variance preserved (for applicable methods)
            variance_preserved = 0.0
            if hasattr(reducer, "explained_variance_ratio_"):
                variance_preserved = np.sum(reducer.explained_variance_ratio_)
            elif method in ["pca", "truncated_svd"]:
                # Fallback calculation for PCA-like methods
                total_var = np.var(X, axis=0).sum()
                reduced_var = np.var(X_reduced, axis=0).sum()
                variance_preserved = min(1.0, reduced_var / total_var)

            # Calculate reconstruction error
            reconstruction_error = self._calculate_reconstruction_error(
                X, X_reduced, reducer
            )

            # Extract feature importance if available
            feature_importance = None
            if hasattr(reducer, "components_"):
                # For linear methods like PCA, use component magnitudes
                feature_importance = np.abs(reducer.components_).mean(axis=0)
            elif hasattr(reducer, "feature_importances_"):
                feature_importance = reducer.feature_importances_

            processing_time = time.time() - start_time

            return ReductionResult(
                method=method,
                original_dimensions=X.shape[1],
                reduced_dimensions=X_reduced.shape[1],
                variance_preserved=variance_preserved,
                reconstruction_error=reconstruction_error,
                processing_time=processing_time,
                quality_score=0.0,  # Will be computed later
                transformed_data=X_reduced,
                transformer=reducer,
                feature_importance=feature_importance,
            )

        except Exception as e:
            self.logger.error(f"Failed to apply method '{method}': {e}")
            raise

    def _calculate_reconstruction_error(
        self, X_original: np.ndarray, X_reduced: np.ndarray, reducer: Any
    ) -> float:
        """Calculate reconstruction error for the dimensionality reduction."""
        try:
            # For methods with inverse_transform
            if hasattr(reducer, "inverse_transform"):
                X_reconstructed = reducer.inverse_transform(X_reduced)
                error = np.mean(np.square(X_original - X_reconstructed))
                return float(error)

            # For PCA-like methods, use explained variance
            if hasattr(reducer, "explained_variance_ratio_"):
                return float(1.0 - np.sum(reducer.explained_variance_ratio_))

            # For other methods, use a simple approximation
            # Based on the ratio of preserved vs original variance
            original_var = np.var(X_original)
            reduced_var = np.var(X_reduced)
            normalized_error = max(0.0, 1.0 - (reduced_var / (original_var + 1e-8)))
            return float(normalized_error)

        except Exception as e:
            self.logger.warning(f"Could not calculate reconstruction error: {e}")
            return 0.5  # Default moderate error

    async def _evaluate_reduction_quality(
        self, X_original: np.ndarray, X_reduced: np.ndarray, y: np.ndarray | None = None
    ) -> dict[str, float]:
        """Evaluate the quality of dimensionality reduction."""
        metrics = {}

        try:
            # 1. Variance preservation (always computed)
            original_total_var = np.sum(np.var(X_original, axis=0))
            reduced_total_var = np.sum(np.var(X_reduced, axis=0))
            metrics["variance_preservation"] = min(
                1.0, reduced_total_var / (original_total_var + 1e-8)
            )

            # 2. Clustering quality preservation
            if X_original.shape[0] >= 20:  # Need sufficient samples
                try:
                    from sklearn.cluster import KMeans
                    from sklearn.metrics import adjusted_rand_score

                    # Simple clustering comparison
                    n_clusters = min(5, X_original.shape[0] // 10)
                    if n_clusters >= 2:
                        kmeans_orig = KMeans(
                            n_clusters=n_clusters, random_state=42, n_init=10
                        )
                        kmeans_reduced = KMeans(
                            n_clusters=n_clusters, random_state=42, n_init=10
                        )

                        labels_orig = kmeans_orig.fit_predict(X_original)
                        labels_reduced = kmeans_reduced.fit_predict(X_reduced)

                        metrics["clustering_preservation"] = adjusted_rand_score(
                            labels_orig, labels_reduced
                        )
                except Exception as e:
                    self.logger.debug(f"Clustering quality evaluation failed: {e}")
                    metrics["clustering_preservation"] = 0.5

            # 3. Classification quality preservation (if labels available)
            if y is not None and len(np.unique(y)) > 1:
                try:
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.model_selection import cross_val_score

                    # Simple classification comparison
                    clf = LogisticRegression(random_state=42, max_iter=1000)

                    scores_orig = cross_val_score(
                        clf, X_original, y, cv=3, scoring="accuracy"
                    )
                    scores_reduced = cross_val_score(
                        clf, X_reduced, y, cv=3, scoring="accuracy"
                    )

                    metrics["classification_preservation"] = np.mean(scores_reduced) / (
                        np.mean(scores_orig) + 1e-8
                    )
                except Exception as e:
                    self.logger.debug(f"Classification quality evaluation failed: {e}")
                    metrics["classification_preservation"] = 0.5

            # 4. Neighborhood preservation
            try:
                from sklearn.neighbors import NearestNeighbors

                k = min(10, X_original.shape[0] // 10)
                if k >= 2:
                    nn_orig = NearestNeighbors(n_neighbors=k)
                    nn_reduced = NearestNeighbors(n_neighbors=k)

                    nn_orig.fit(X_original)
                    nn_reduced.fit(X_reduced)

                    # Sample a subset for efficiency
                    sample_size = min(100, X_original.shape[0])
                    sample_indices = np.random.choice(
                        X_original.shape[0], sample_size, replace=False
                    )

                    neighbors_orig = nn_orig.kneighbors(
                        X_original[sample_indices], return_distance=False
                    )
                    neighbors_reduced = nn_reduced.kneighbors(
                        X_reduced[sample_indices], return_distance=False
                    )

                    # Calculate neighborhood preservation score
                    preservation_scores = []
                    for i in range(sample_size):
                        overlap = len(
                            set(neighbors_orig[i]) & set(neighbors_reduced[i])
                        )
                        preservation_scores.append(overlap / k)

                    metrics["neighborhood_preservation"] = np.mean(preservation_scores)
            except Exception as e:
                self.logger.debug(f"Neighborhood preservation evaluation failed: {e}")
                metrics["neighborhood_preservation"] = 0.5

        except Exception as e:
            self.logger.warning(f"Quality evaluation failed: {e}")
            # Return default metrics
            metrics = {
                "variance_preservation": 0.5,
                "clustering_preservation": 0.5,
                "classification_preservation": 0.5 if y is not None else None,
                "neighborhood_preservation": 0.5,
            }

        return metrics

    def _compute_overall_quality_score(
        self, result: ReductionResult, evaluation_metrics: dict[str, float]
    ) -> float:
        """Compute overall quality score for the dimensionality reduction."""
        try:
            score_components = []
            weights = []

            # Variance preservation
            if "variance_preservation" in evaluation_metrics:
                score_components.append(evaluation_metrics["variance_preservation"])
                weights.append(self.config.variance_preservation_weight)

            # Clustering preservation
            if "clustering_preservation" in evaluation_metrics:
                score_components.append(evaluation_metrics["clustering_preservation"])
                weights.append(self.config.clustering_quality_weight)

            # Classification preservation
            if (
                "classification_preservation" in evaluation_metrics
                and evaluation_metrics["classification_preservation"] is not None
            ):
                score_components.append(
                    evaluation_metrics["classification_preservation"]
                )
                weights.append(self.config.classification_quality_weight)

            # Neighborhood preservation
            if "neighborhood_preservation" in evaluation_metrics:
                score_components.append(evaluation_metrics["neighborhood_preservation"])
                weights.append(0.2)  # Fixed weight for neighborhood preservation

            # Processing time penalty (favor faster methods)
            time_penalty = max(
                0.0, 1.0 - result.processing_time / 60.0
            )  # Penalty after 1 minute
            score_components.append(time_penalty)
            weights.append(0.1)

            # Dimensionality efficiency (favor achieving target dimensions)
            target_efficiency = min(
                1.0, result.reduced_dimensions / self.config.target_dimensions
            )
            score_components.append(target_efficiency)
            weights.append(0.1)

            # Weighted average
            if score_components and weights:
                weights = np.array(weights)
                weights = weights / weights.sum()  # Normalize weights
                overall_score = np.average(score_components, weights=weights)
                return float(np.clip(overall_score, 0.0, 1.0))

            return 0.5  # Default score

        except Exception as e:
            self.logger.warning(f"Quality score computation failed: {e}")
            return 0.5

    def _generate_cache_key(self, X: np.ndarray, method: str) -> str:
        """Generate cache key for reduction results."""
        import hashlib

        key_data = f"{X.shape}_{method}_{self.config.target_dimensions}_{self.config.random_state}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _update_performance_tracking(self, result: ReductionResult):
        """Update performance tracking for the reduction method."""
        self.reduction_history.append(result)

        # Update method-specific performance
        if result.method not in self.method_performance:
            self.method_performance[result.method] = []

        self.method_performance[result.method].append(result.quality_score)

        # Keep only recent history to prevent memory growth
        if len(self.reduction_history) > 100:
            self.reduction_history = self.reduction_history[-50:]

        for method in self.method_performance:
            if len(self.method_performance[method]) > 50:
                self.method_performance[method] = self.method_performance[method][-25:]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.reduction_history:
            return {"status": "no_data"}

        recent_results = self.reduction_history[-10:]

        summary = {
            "status": "available",
            "total_reductions": len(self.reduction_history),
            "recent_performance": {
                "avg_quality_score": float(
                    np.mean([r.quality_score for r in recent_results])
                ),
                "avg_processing_time": float(
                    np.mean([r.processing_time for r in recent_results])
                ),
                "avg_variance_preserved": float(
                    np.mean([r.variance_preserved for r in recent_results])
                ),
                "avg_dimensionality_reduction": float(
                    np.mean([
                        (r.original_dimensions - r.reduced_dimensions)
                        / r.original_dimensions
                        for r in recent_results
                    ])
                ),
            },
            "method_performance": {},
            "available_methods": list(self.available_methods.keys()),
        }

        # Method-specific performance
        for method, scores in self.method_performance.items():
            if scores:
                summary["method_performance"][method] = {
                    "avg_quality": float(np.mean(scores)),
                    "std_quality": float(np.std(scores)),
                    "usage_count": len(scores),
                    "reliability": float(np.mean([s > 0.5 for s in scores])),
                }

        return summary

    def get_recommended_method(
        self, X_shape: tuple[int, int], y: np.ndarray | None = None
    ) -> str:
        """Get recommended method based on data characteristics and historical performance."""
        n_samples, n_features = X_shape

        # Use historical performance if available
        if self.method_performance:
            best_method = max(
                self.method_performance.keys(),
                key=lambda m: np.mean(self.method_performance[m]),
            )
            avg_quality = np.mean(self.method_performance[best_method])
            if avg_quality > 0.7:  # Good historical performance
                return best_method

        # Fall back to heuristics
        if n_samples > 10000 or self.config.fast_mode:
            return "pca"
        if y is not None and len(np.unique(y)) > 1:
            return "lda"
        if UMAP_AVAILABLE and n_features > 20:
            return "umap"
        return "pca"


# Factory function for easy integration
def get_dimensionality_reducer(
    config: DimensionalityConfig | None = None,
) -> AdvancedDimensionalityReducer:
    """Get dimensionality reducer instance.

    Args:
        config: Optional dimensionality reduction configuration

    Returns:
        AdvancedDimensionalityReducer instance
    """
    return AdvancedDimensionalityReducer(config=config)
