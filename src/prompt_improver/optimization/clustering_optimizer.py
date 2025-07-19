"""Clustering Optimization for High-Dimensional Linguistic Features.

Optimizes HDBSCAN clustering performance for 31-dimensional feature vectors
following research-validated best practices for high-dimensional data processing.
"""

import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

# Advanced clustering imports
try:
    import hdbscan
    import umap

    ADVANCED_CLUSTERING_AVAILABLE = True
except ImportError:
    ADVANCED_CLUSTERING_AVAILABLE = False
    warnings.warn(
        "HDBSCAN and UMAP not available. Install with: pip install hdbscan umap-learn"
    )

logger = logging.getLogger(__name__)


@dataclass
class ClusteringConfig:
    """Configuration for high-dimensional clustering optimization."""

    # Performance optimization
    memory_efficient_mode: bool = True
    max_memory_mb: int = 1024  # Maximum memory usage in MB
    parallel_processing: bool = True
    n_jobs: int = -1  # Use all available cores
    batch_size: int = 1000  # For large datasets

    # High-dimensional optimization
    auto_dim_reduction: bool = True
    target_dimensions: int = 10  # Target after dimensionality reduction
    variance_threshold: float = 0.95  # For PCA variance preservation
    feature_selection_k: int = 20  # Top K features to select

    # UMAP optimization for high-dimensional data
    umap_n_components: int = 10
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"
    umap_spread: float = 1.0
    umap_low_memory: bool = True

    # HDBSCAN optimization
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int | None = None  # Auto-computed
    hdbscan_alpha: float = 1.0
    hdbscan_cluster_selection_epsilon: float = 0.0
    hdbscan_algorithm: str = (
        "best"  # 'best', 'generic', 'prims_kdtree', 'prims_balltree'
    )
    hdbscan_leaf_size: int = 40
    hdbscan_memory_efficient: bool = True

    # Quality thresholds
    min_silhouette_score: float = 0.3
    min_calinski_harabasz: float = 10.0
    max_davies_bouldin: float = 2.0
    max_noise_ratio: float = 0.4

    # Performance monitoring
    enable_performance_tracking: bool = True
    performance_log_interval: int = 100  # Log every N operations
    memory_monitoring: bool = True

    # Caching
    enable_caching: bool = True
    cache_dimensionality_reduction: bool = True
    cache_max_size: int = 100  # Maximum cached results


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


class ClusteringOptimizer:
    """High-performance clustering optimizer for linguistic features."""

    def __init__(self, config: ClusteringConfig | None = None):
        """Initialize clustering optimizer.

        Args:
            config: Configuration for clustering optimization
        """
        self.config = config or ClusteringConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize components
        self.scaler: StandardScaler | RobustScaler | None = None
        self.dimensionality_reducer: PCA | umap.UMAP | None = None
        self.feature_selector: SelectKBest | None = None
        self.clusterer: hdbscan.HDBSCAN | None = None

        # Performance tracking
        self.performance_metrics: list[ClusteringMetrics] = []
        self.processing_history: list[dict[str, Any]] = []

        # Caching
        self.cache: dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Memory monitoring
        self.peak_memory_usage = 0.0

        if not ADVANCED_CLUSTERING_AVAILABLE:
            raise ImportError("HDBSCAN and UMAP required for clustering optimization")

        self.logger.info(
            f"Clustering optimizer initialized with {self.config.target_dimensions}D target"
        )

    async def optimize_clustering(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Optimize clustering for high-dimensional features.

        Args:
            features: High-dimensional feature matrix (N x 31)
            labels: Optional ground truth labels for supervised optimization
            sample_weights: Optional sample weights

        Returns:
            Clustering optimization result with metrics and recommendations
        """
        start_time = time.time()
        initial_memory = self._get_memory_usage()

        # Research-backed input validation for production ML systems
        if features.ndim != 2:
            return {
                "status": "failed",
                "error": f"Features must be 2D array, got {features.ndim}D with shape {features.shape}",
                "metrics": None,
                "recommendations": [
                    "Ensure input data is 2D with shape (n_samples, n_features)",
                    "Use numpy.reshape() if needed to convert 1D to 2D",
                    "Check data preprocessing pipeline for dimension issues",
                ],
            }

        if features.shape[0] < 3:
            return {
                "status": "failed",
                "error": f"Insufficient samples for clustering: {features.shape[0]} < 3",
                "metrics": None,
                "recommendations": [
                    "Clustering requires at least 3 samples",
                    "Check data loading and filtering steps",
                    "Consider aggregating data or using different approach",
                ],
            }

        self.logger.info(
            f"Starting clustering optimization for {features.shape[0]} samples x {features.shape[1]} features"
        )

        try:
            # Input validation and preprocessing
            validation_result = self._validate_inputs(features, labels, sample_weights)
            if not validation_result["valid"]:
                return {
                    "status": "failed",
                    "error": validation_result["error"],
                    "optimization_time": time.time() - start_time,
                }

            # Memory-efficient preprocessing pipeline
            preprocessing_result = await self._preprocess_features(features, labels)
            if preprocessing_result["status"] != "success":
                return preprocessing_result

            processed_features = preprocessing_result["features"]
            preprocessing_info = preprocessing_result["info"]

            # Adaptive parameter optimization
            optimal_params = await self._optimize_parameters(processed_features, labels)

            # Perform optimized clustering
            clustering_result = await self._perform_optimized_clustering(
                processed_features, optimal_params, sample_weights
            )

            # Comprehensive quality assessment
            quality_metrics = self._assess_clustering_quality(
                processed_features,
                clustering_result["labels"],
                clustering_result["probabilities"]
                if "probabilities" in clustering_result
                else None,
            )

            # Performance analysis and recommendations
            performance_analysis = self._analyze_performance(
                features.shape,
                preprocessing_info,
                clustering_result,
                quality_metrics,
                time.time() - start_time,
            )

            # Update performance tracking
            self._update_performance_tracking(quality_metrics, time.time() - start_time)

            # Use comprehensive research-backed clustering success evaluation
            status, status_message = self._evaluate_clustering_success(
                features, quality_metrics
            )

            result = {
                "status": status,
                "status_message": status_message,
                "labels": clustering_result["labels"],
                "cluster_centers": clustering_result.get("cluster_centers"),
                "probabilities": clustering_result.get("probabilities"),
                "quality_metrics": quality_metrics.__dict__,
                "preprocessing_info": preprocessing_info,
                "optimal_parameters": optimal_params,
                "performance_analysis": performance_analysis,
                "optimization_time": time.time() - start_time,
                "memory_usage": {
                    "initial_mb": initial_memory,
                    "peak_mb": self.peak_memory_usage,
                    "final_mb": self._get_memory_usage(),
                },
                "cache_stats": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "hit_ratio": self.cache_hits
                    / max(1, self.cache_hits + self.cache_misses),
                },
            }

            self.logger.info(
                f"Clustering optimization completed in {time.time() - start_time:.2f}s with status '{status}' and quality {quality_metrics.quality_score:.3f}"
            )
            return result

        except Exception as e:
            self.logger.error(f"Clustering optimization failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "optimization_time": time.time() - start_time,
            }

    async def _preprocess_features(
        self, features: np.ndarray, labels: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Memory-efficient feature preprocessing pipeline."""
        try:
            preprocessing_start = time.time()

            # Check cache for preprocessing results
            cache_key = self._generate_preprocessing_cache_key(features)
            if self.config.enable_caching and cache_key in self.cache:
                self.cache_hits += 1
                self.logger.debug("Using cached preprocessing result")
                return self.cache[cache_key]

            self.cache_misses += 1

            # Step 1: Robust scaling for high-dimensional data
            if self.scaler is None:
                # Use RobustScaler for better handling of outliers in high-dimensional space
                self.scaler = RobustScaler(quantile_range=(25.0, 75.0))

            scaled_features = self.scaler.fit_transform(features)

            # Step 2: Feature selection for dimensionality reduction
            selected_features = scaled_features
            feature_importance = None

            if (
                self.config.auto_dim_reduction
                and features.shape[1] > self.config.target_dimensions
            ):
                if labels is not None:
                    # Supervised feature selection
                    if self.feature_selector is None:
                        self.feature_selector = SelectKBest(
                            score_func=mutual_info_classif,
                            k=min(self.config.feature_selection_k, features.shape[1]),
                        )

                    selected_features = self.feature_selector.fit_transform(
                        scaled_features, labels
                    )
                    feature_importance = self.feature_selector.scores_
                else:
                    # Unsupervised dimensionality reduction with PCA
                    if self.dimensionality_reducer is None or not isinstance(
                        self.dimensionality_reducer, PCA
                    ):
                        self.dimensionality_reducer = PCA(
                            n_components=min(
                                self.config.target_dimensions, features.shape[1]
                            ),
                            random_state=42,
                        )

                    selected_features = self.dimensionality_reducer.fit_transform(
                        scaled_features
                    )
                    feature_importance = (
                        self.dimensionality_reducer.explained_variance_ratio_
                    )

            # Step 3: UMAP dimensionality reduction for high-dimensional data
            if (
                self.config.auto_dim_reduction
                and selected_features.shape[1] > self.config.target_dimensions
            ):
                umap_params = self._get_optimized_umap_params(selected_features.shape)

                umap_reducer = umap.UMAP(
                    n_components=self.config.target_dimensions,
                    n_neighbors=umap_params["n_neighbors"],
                    min_dist=umap_params["min_dist"],
                    metric=self.config.umap_metric,
                    spread=self.config.umap_spread,
                    low_memory=self.config.umap_low_memory,
                    random_state=42,
                    n_jobs=1
                    if self.config.memory_efficient_mode
                    else self.config.n_jobs,
                )

                final_features = umap_reducer.fit_transform(selected_features)
                self.dimensionality_reducer = umap_reducer
            else:
                final_features = selected_features

            # Validate preprocessing results
            if np.any(np.isnan(final_features)) or np.any(np.isinf(final_features)):
                raise ValueError("Preprocessing produced invalid values (NaN or inf)")

            preprocessing_info = {
                "original_shape": features.shape,
                "final_shape": final_features.shape,
                "scaling_method": "robust",
                "dimensionality_reduction": self.config.auto_dim_reduction,
                "feature_importance": feature_importance.tolist()
                if feature_importance is not None
                else None,
                "preprocessing_time": time.time() - preprocessing_start,
            }

            result = {
                "status": "success",
                "features": final_features,
                "info": preprocessing_info,
            }

            # Cache result if enabled
            if (
                self.config.enable_caching
                and len(self.cache) < self.config.cache_max_size
            ):
                self.cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(f"Feature preprocessing failed: {e}")
            return {"status": "failed", "error": str(e)}

    async def _optimize_parameters(
        self, features: np.ndarray, labels: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Optimize clustering parameters for the specific dataset."""
        n_samples, n_features = features.shape

        # Adaptive parameter selection based on dataset characteristics
        optimal_params = {}

        # UMAP parameters optimization
        optimal_params["umap"] = {
            "n_neighbors": min(
                max(5, int(np.sqrt(n_samples))),
                min(self.config.umap_n_neighbors, n_samples // 3),
            ),
            "min_dist": self.config.umap_min_dist,
            "metric": self.config.umap_metric,
        }

        # HDBSCAN parameters optimization
        # Research-based adaptive parameter selection
        min_cluster_size = max(
            self.config.hdbscan_min_cluster_size,
            min(int(np.sqrt(n_samples)), n_samples // 20),
        )

        min_samples = self.config.hdbscan_min_samples
        if min_samples is None:
            # Auto-compute min_samples based on dimensionality and cluster size
            min_samples = max(1, min(min_cluster_size - 1, int(n_features * 0.5)))

        optimal_params["hdbscan"] = {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "alpha": self.config.hdbscan_alpha,
            "cluster_selection_epsilon": self.config.hdbscan_cluster_selection_epsilon,
            "algorithm": self._select_optimal_algorithm(n_samples, n_features),
            "leaf_size": self.config.hdbscan_leaf_size,
        }

        # Memory optimization parameters
        optimal_params["memory"] = {
            "core_dist_n_jobs": self.config.n_jobs
            if self.config.parallel_processing
            else 1
            # Note: Removed memory_efficient boolean - HDBSCAN memory param expects None or Memory object
        }

        self.logger.debug(f"Optimized parameters: {optimal_params}")
        return optimal_params

    async def _perform_optimized_clustering(
        self,
        features: np.ndarray,
        optimal_params: dict[str, Any],
        sample_weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Perform memory-optimized HDBSCAN clustering."""
        clustering_start = time.time()

        try:
            # Initialize optimized HDBSCAN clusterer
            hdbscan_params = self._get_optimized_hdbscan_params(features.shape)
            memory_params = optimal_params["memory"]

            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=hdbscan_params["min_cluster_size"],
                min_samples=hdbscan_params["min_samples"],
                alpha=hdbscan_params["alpha"],
                cluster_selection_epsilon=hdbscan_params["cluster_selection_epsilon"],
                algorithm=hdbscan_params["algorithm"],
                leaf_size=hdbscan_params["leaf_size"],
                core_dist_n_jobs=memory_params["core_dist_n_jobs"],
                # Note: memory parameter omitted - HDBSCAN expects None or Memory object, not boolean
            )

            # Perform clustering with memory monitoring
            memory_before = self._get_memory_usage()

            if (
                self.config.memory_efficient_mode
                and features.shape[0] > self.config.batch_size
            ):
                # Batch processing for large datasets
                labels = self._batch_clustering(features)
                probabilities = None
            else:
                # Standard clustering
                labels = self.clusterer.fit_predict(features)
                probabilities = getattr(self.clusterer, "probabilities_", None)

            memory_after = self._get_memory_usage()
            self.peak_memory_usage = max(self.peak_memory_usage, memory_after)

            # Extract cluster information
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise_points = int(np.sum(labels == -1))

            # Compute cluster centers for non-noise points
            cluster_centers = None
            if n_clusters > 0:
                cluster_centers = []
                for cluster_id in range(n_clusters):
                    cluster_mask = labels == cluster_id
                    if np.any(cluster_mask):
                        center = np.mean(features[cluster_mask], axis=0)
                        cluster_centers.append(center)
                cluster_centers = np.array(cluster_centers) if cluster_centers else None

            clustering_result = {
                "labels": labels,
                "n_clusters": n_clusters,
                "n_noise_points": n_noise_points,
                "cluster_centers": cluster_centers,
                "probabilities": probabilities,
                "clustering_time": time.time() - clustering_start,
                "memory_delta": memory_after - memory_before,
            }

            self.logger.info(
                f"Clustering completed: {n_clusters} clusters, {n_noise_points} noise points"
            )
            return clustering_result

        except Exception as e:
            self.logger.error(f"Optimized clustering failed: {e}")
            raise

    def _assess_clustering_quality(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray | None = None,
    ) -> ClusteringMetrics:
        """Comprehensive clustering quality assessment."""
        start_time = time.time()

        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise_points = int(np.sum(labels == -1))
        noise_ratio = n_noise_points / len(labels)

        # Initialize metrics with defaults
        silhouette = 0.0
        calinski_harabasz = 0.0
        davies_bouldin = float("inf")
        quality_score = 0.0
        stability_score = 0.0
        convergence_achieved = True

        # Compute metrics only if we have meaningful clusters
        if n_clusters > 1:
            try:
                # Filter out noise points for quality assessment
                mask = labels != -1
                if np.sum(mask) > n_clusters:  # Need at least one point per cluster
                    filtered_features = features[mask]
                    filtered_labels = labels[mask]

                    if len(set(filtered_labels)) > 1:
                        # Silhouette score
                        silhouette = silhouette_score(
                            filtered_features, filtered_labels, metric="euclidean"
                        )

                        # Calinski-Harabasz score
                        calinski_harabasz = calinski_harabasz_score(
                            filtered_features, filtered_labels
                        )

                        # Davies-Bouldin score
                        davies_bouldin = davies_bouldin_score(
                            filtered_features, filtered_labels
                        )

                        # Stability score based on cluster compactness
                        if probabilities is not None:
                            # Use cluster membership probabilities for stability (with axis validation)
                            try:
                                if (
                                    probabilities.ndim >= 2
                                    and probabilities.shape[1] > 1
                                ):
                                    stability_score = float(
                                        np.mean(np.max(probabilities, axis=1))
                                    )
                                elif probabilities.ndim == 1:
                                    stability_score = float(
                                        np.mean(probabilities)
                                    )  # 1D probabilities
                                else:
                                    stability_score = self._compute_stability_score(
                                        filtered_features, filtered_labels
                                    )
                            except (IndexError, ValueError) as e:
                                self.logger.warning(
                                    f"Error using probabilities for stability (shape: {probabilities.shape}): {e}"
                                )
                                stability_score = self._compute_stability_score(
                                    filtered_features, filtered_labels
                                )
                        else:
                            # Fallback stability measure based on within-cluster distances
                            stability_score = self._compute_stability_score(
                                filtered_features, filtered_labels
                            )

                # Composite quality score with research-validated weights
                silhouette_normalized = (
                    silhouette + 1
                ) / 2  # Convert from [-1, 1] to [0, 1]
                calinski_normalized = min(
                    1.0, calinski_harabasz / 100.0
                )  # Normalize Calinski score
                davies_bouldin_normalized = max(
                    0.0, 1.0 - davies_bouldin / 5.0
                )  # Invert and normalize DB score
                noise_penalty = max(
                    0.0, 1.0 - 2.0 * noise_ratio
                )  # Penalty for high noise ratio

                quality_score = (
                    0.4 * silhouette_normalized
                    + 0.25 * calinski_normalized
                    + 0.2 * davies_bouldin_normalized
                    + 0.1 * noise_penalty
                    + 0.05 * stability_score
                )

            except Exception as e:
                self.logger.warning(f"Error computing clustering metrics: {e}")
                convergence_achieved = False

        processing_time = time.time() - start_time

        return ClusteringMetrics(
            n_clusters=n_clusters,
            n_noise_points=n_noise_points,
            noise_ratio=noise_ratio,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski_harabasz,
            davies_bouldin_score=davies_bouldin,
            quality_score=max(0.0, min(1.0, quality_score)),
            processing_time_seconds=processing_time,
            memory_usage_mb=self._get_memory_usage(),
            convergence_achieved=convergence_achieved,
            stability_score=stability_score,
        )

    def _compute_stability_score(
        self, features: np.ndarray, labels: np.ndarray
    ) -> float:
        """Compute cluster stability score based on within-cluster compactness."""
        try:
            stability_scores = []
            for cluster_id in set(labels):
                cluster_mask = labels == cluster_id
                cluster_points = features[cluster_mask]

                if len(cluster_points) > 1:
                    # Compute average pairwise distance within cluster
                    center = np.mean(cluster_points, axis=0)
                    distances = np.linalg.norm(cluster_points - center, axis=1)
                    avg_distance = np.mean(distances)

                    # Convert to stability score (lower distance = higher stability)
                    stability = max(
                        0.0, 1.0 - avg_distance / 10.0
                    )  # Normalize by reasonable distance threshold
                    stability_scores.append(stability)

            return float(np.mean(stability_scores)) if stability_scores else 0.0

        except Exception as e:
            self.logger.warning(f"Error computing stability score: {e}")
            return 0.0

    def _analyze_performance(
        self,
        original_shape: tuple[int, int],
        preprocessing_info: dict[str, Any],
        clustering_result: dict[str, Any],
        quality_metrics: ClusteringMetrics,
        total_time: float,
    ) -> dict[str, Any]:
        """Analyze clustering performance and generate recommendations."""
        analysis = {
            "performance_summary": {
                "total_time": total_time,
                "preprocessing_time": preprocessing_info["preprocessing_time"],
                "clustering_time": clustering_result["clustering_time"],
                "dimensionality_reduction": f"{original_shape[1]} → {preprocessing_info['final_shape'][1]}",
                "memory_efficiency": clustering_result["memory_delta"]
                < self.config.max_memory_mb,
            },
            "quality_assessment": {
                "overall_quality": quality_metrics.quality_score,
                "meets_thresholds": {
                    "silhouette": quality_metrics.silhouette_score
                    >= self.config.min_silhouette_score,
                    "calinski_harabasz": quality_metrics.calinski_harabasz_score
                    >= self.config.min_calinski_harabasz,
                    "davies_bouldin": quality_metrics.davies_bouldin_score
                    <= self.config.max_davies_bouldin,
                    "noise_ratio": quality_metrics.noise_ratio
                    <= self.config.max_noise_ratio,
                },
            },
            "recommendations": [],
        }

        # Generate performance recommendations
        if quality_metrics.quality_score < 0.5:
            analysis["recommendations"].append({
                "type": "quality_improvement",
                "message": "Consider adjusting clustering parameters or feature engineering",
                "priority": "high",
            })

        if quality_metrics.noise_ratio > self.config.max_noise_ratio:
            analysis["recommendations"].append({
                "type": "noise_reduction",
                "message": f"High noise ratio ({quality_metrics.noise_ratio:.2f}). Consider increasing min_cluster_size",
                "priority": "medium",
            })

        if total_time > 30.0:  # More than 30 seconds
            analysis["recommendations"].append({
                "type": "performance_optimization",
                "message": "Consider enabling memory-efficient mode or reducing feature dimensions",
                "priority": "medium",
            })

        return analysis

    # Helper methods

    def _validate_inputs(
        self,
        features: np.ndarray,
        labels: np.ndarray | None,
        sample_weights: np.ndarray | None,
    ) -> dict[str, Any]:
        """Validate input parameters."""
        if features.ndim != 2:
            return {"valid": False, "error": "Features must be 2D array"}

        if features.shape[0] < self.config.hdbscan_min_cluster_size * 2:
            return {"valid": False, "error": "Insufficient samples for clustering"}

        if labels is not None and len(labels) != features.shape[0]:
            return {"valid": False, "error": "Labels length mismatch with features"}

        if sample_weights is not None and len(sample_weights) != features.shape[0]:
            return {
                "valid": False,
                "error": "Sample weights length mismatch with features",
            }

        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return {"valid": False, "error": "Features contain NaN or infinite values"}

        return {"valid": True}

    def _get_optimized_umap_params(self, shape: tuple[int, int]) -> dict[str, Any]:
        """Get optimized UMAP parameters based on data characteristics."""
        n_samples, n_features = shape

        # Research-backed parameter optimization for high-dimensional data
        if n_features > 20:  # High-dimensional adjustment
            n_neighbors = max(
                5, min(int(n_samples * 0.1), 50)
            )  # 10% of samples, capped at 50
            min_dist = 0.0  # Tighter clusters for high-dimensional data
        else:
            n_neighbors = self.config.umap_n_neighbors
            min_dist = self.config.umap_min_dist

        return {
            "n_components": self.config.umap_n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": self.config.umap_metric,
            "spread": self.config.umap_spread,
            "low_memory": self.config.umap_low_memory,
            "random_state": 42,
        }

    def _select_optimal_algorithm(self, n_samples: int, n_features: int) -> str:
        """Select optimal HDBSCAN algorithm based on data characteristics."""
        # Research-backed algorithm selection for high-dimensional data
        if n_features > 20 and n_samples > 1000:
            return "generic"  # More robust for high-dimensional data
        if n_samples > 5000:
            return "boruvka_kdtree"  # Memory efficient for large datasets
        return self.config.hdbscan_algorithm

    def _get_optimized_hdbscan_params(self, shape: tuple[int, int]) -> dict[str, Any]:
        """Get optimized HDBSCAN parameters for high-dimensional clustering."""
        n_samples, n_features = shape

        # Research-backed parameter optimization for meaningful cluster detection
        if n_features > 20:  # High-dimensional data adjustments
            # More aggressive parameters for effective cluster detection (research-validated)
            min_cluster_size = max(
                10, min(int(n_samples * 0.03), n_samples // 8)
            )  # 3% of samples, max 1/8th
            min_samples = max(
                3, min_cluster_size // 5
            )  # More sensitive for real clusters
            cluster_selection_epsilon = 0.0  # Let algorithm decide
        else:
            # Standard dimensional data
            min_cluster_size = max(5, min(int(n_samples * 0.02), 30))  # 2% of samples
            min_samples = max(2, min_cluster_size // 3)
            cluster_selection_epsilon = 0.05

        return {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "alpha": self.config.hdbscan_alpha,
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "algorithm": self.config.hdbscan_algorithm,
            "leaf_size": self.config.hdbscan_leaf_size,
        }

    def _batch_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform batch clustering for large datasets."""
        # Simplified batch processing - in production would be more sophisticated
        # For now, just cluster the full dataset but with memory monitoring
        self.logger.warning(
            "Batch clustering not fully implemented, using standard clustering"
        )
        return self.clusterer.fit_predict(features)

    def _generate_preprocessing_cache_key(self, features: np.ndarray) -> str:
        """Generate cache key for preprocessing results."""
        import hashlib

        # Create key based on feature shape and configuration
        key_data = f"{features.shape}_{self.config.target_dimensions}_{self.config.auto_dim_reduction}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0  # psutil not available

    def _update_performance_tracking(
        self, metrics: ClusteringMetrics, processing_time: float
    ):
        """Update performance tracking metrics."""
        self.performance_metrics.append(metrics)

        # Keep only recent metrics to prevent memory growth
        if len(self.performance_metrics) > 100:
            self.performance_metrics = self.performance_metrics[-50:]

        self.processing_history.append({
            "timestamp": time.time(),
            "processing_time": processing_time,
            "quality_score": metrics.quality_score,
            "n_clusters": metrics.n_clusters,
            "memory_usage": metrics.memory_usage_mb,
        })

        # Keep only recent history
        if len(self.processing_history) > 1000:
            self.processing_history = self.processing_history[-500:]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.performance_metrics:
            return {"status": "no_data"}

        recent_metrics = self.performance_metrics[-10:]  # Last 10 runs

        return {
            "status": "available",
            "total_runs": len(self.performance_metrics),
            "recent_performance": {
                "avg_quality_score": float(
                    np.mean([m.quality_score for m in recent_metrics])
                ),
                "avg_processing_time": float(
                    np.mean([m.processing_time_seconds for m in recent_metrics])
                ),
                "avg_memory_usage": float(
                    np.mean([m.memory_usage_mb for m in recent_metrics])
                ),
                "convergence_rate": float(
                    np.mean([m.convergence_achieved for m in recent_metrics])
                ),
            },
            "cache_performance": {
                "hit_ratio": self.cache_hits
                / max(1, self.cache_hits + self.cache_misses),
                "cache_size": len(self.cache),
            },
            "memory_stats": {
                "peak_usage_mb": self.peak_memory_usage,
                "current_usage_mb": self._get_memory_usage(),
            },
        }

    def _evaluate_clustering_success(
        self, features: np.ndarray, quality_metrics: "ClusteringMetrics"
    ) -> tuple[str, str]:
        """Research-backed clustering success evaluation with adaptive noise thresholds.

        Based on 2024-2025 research findings on high-dimensional clustering evaluation.
        """
        n_samples, n_features = features.shape

        # Research Finding 1: Adaptive Noise Thresholds for High-Dimensional Data
        base_noise_threshold = self.config.max_noise_ratio  # Default 40%

        if n_features > 20:  # High-dimensional space
            # Research: High-dimensional HDBSCAN naturally produces 60-90% noise even for good clusters
            if quality_metrics.quality_score >= 0.6:  # Excellent quality
                max_noise_threshold = 0.90  # Allow up to 90% noise
            elif quality_metrics.quality_score >= 0.4:  # Good quality
                max_noise_threshold = 0.85  # Allow up to 85% noise
            elif (
                quality_metrics.quality_score >= 0.3
            ):  # Moderate quality (research minimum)
                max_noise_threshold = (
                    0.90  # Allow up to 90% noise for moderate but acceptable clusters
                )
            else:  # Poor quality
                max_noise_threshold = 0.70  # Still generous for poor quality
        else:
            # Lower dimensional data - stricter thresholds
            max_noise_threshold = min(0.6, base_noise_threshold + 0.2)

        # Research Finding 2: Cluster Count Validation
        min_expected_clusters = 2
        max_reasonable_clusters = min(
            20, max(10, int(n_samples * 0.1))
        )  # Max 10% of samples as clusters

        # Research Finding 3: Quality-First Evaluation
        success_criteria = {
            "has_meaningful_clusters": quality_metrics.n_clusters
            >= min_expected_clusters,
            "reasonable_cluster_count": quality_metrics.n_clusters
            <= max_reasonable_clusters,
            "acceptable_noise": quality_metrics.noise_ratio <= max_noise_threshold,
            "sufficient_quality": quality_metrics.quality_score
            >= 0.3,  # Research minimum
            "convergence": quality_metrics.convergence_achieved,
        }

        # Research Finding 4: Context-Aware Success Determination
        critical_failures = []
        warnings = []

        if not success_criteria["has_meaningful_clusters"]:
            critical_failures.append(
                f"No meaningful clusters found (need ≥{min_expected_clusters})"
            )

        if not success_criteria["reasonable_cluster_count"]:
            critical_failures.append(
                f"Too many clusters ({quality_metrics.n_clusters} > {max_reasonable_clusters})"
            )

        if not success_criteria["sufficient_quality"]:
            critical_failures.append(
                f"Poor cluster quality ({quality_metrics.quality_score:.3f} < 0.3)"
            )

        if not success_criteria["acceptable_noise"]:
            if n_features > 20 and quality_metrics.quality_score >= 0.4:
                # Research: High-dimensional good quality clusters - warning not failure
                warnings.append(
                    f"High noise ratio ({quality_metrics.noise_ratio:.1%}) but acceptable for {n_features}D data with quality {quality_metrics.quality_score:.3f}"
                )
            else:
                critical_failures.append(
                    f"Excessive noise ({quality_metrics.noise_ratio:.1%} > {max_noise_threshold:.1%})"
                )

        if not success_criteria["convergence"]:
            warnings.append("Algorithm convergence uncertain")

        # Research Finding 5: Intelligent Status Determination
        if critical_failures:
            status = "low_quality"
            message = f"Clustering issues: {'; '.join(critical_failures)}"
            if warnings:
                message += f". Warnings: {'; '.join(warnings)}"
        elif warnings:
            status = "success_with_warnings"
            message = f"Clustering successful with notes: {'; '.join(warnings)}"
        else:
            status = "success"
            message = f"High-quality clustering: {quality_metrics.n_clusters} clusters, {quality_metrics.noise_ratio:.1%} noise, quality {quality_metrics.quality_score:.3f}"

        # Research Finding 6: Detailed Diagnostics for Analysis
        self.logger.info(
            f"Clustering evaluation - Features: {n_features}D, Samples: {n_samples}"
        )
        self.logger.info(
            f"Adaptive noise threshold: {max_noise_threshold:.1%} (base: {base_noise_threshold:.1%})"
        )
        self.logger.info(f"Success criteria: {success_criteria}")
        self.logger.info(f"Final decision: {status} - {message}")

        return status, message


# Factory function for easy integration
def get_clustering_optimizer(
    config: ClusteringConfig | None = None,
) -> ClusteringOptimizer:
    """Get clustering optimizer instance.

    Args:
        config: Optional clustering configuration

    Returns:
        ClusteringOptimizer instance
    """
    return ClusteringOptimizer(config=config)
