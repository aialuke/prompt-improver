"""Clustering preprocessing service.

Provides specialized preprocessing for clustering including:
- Feature preprocessing and validation for clustering
- UMAP dimensionality reduction for clustering
- Input validation with clustering-specific checks  
- Data preparation optimized for clustering algorithms

Follows clean architecture patterns with protocol-based interfaces.
"""

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

# import numpy as np  # Converted to lazy loading
from prompt_improver.core.utils.lazy_ml_loader import get_numpy, get_sklearn

from . import ClusteringPreprocessorProtocol, ClusteringPreprocessingResult

logger = logging.getLogger(__name__)

# Optional imports for advanced preprocessing
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")

try:
    # Test if sklearn is available by trying to import it
    sklearn = get_sklearn()
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available for advanced preprocessing")

class ClusteringPreprocessorService:
    """Service for preprocessing data before clustering following clean architecture."""

    def __init__(self, scaling_method: str = "robust", 
                 auto_dimensionality_reduction: bool = True,
                 target_dimensions: int = 10,
                 variance_threshold: float = 1e-6,
                 umap_n_neighbors: int = 15,
                 umap_min_dist: float = 0.1,
                 memory_efficient: bool = True):
        """Initialize clustering preprocessor service.
        
        Args:
            scaling_method: Scaling method ("robust", "standard", "none")
            auto_dimensionality_reduction: Whether to apply automatic dimensionality reduction
            target_dimensions: Target number of dimensions after reduction
            variance_threshold: Minimum variance threshold for feature removal
            umap_n_neighbors: UMAP n_neighbors parameter
            umap_min_dist: UMAP min_dist parameter
            memory_efficient: Whether to use memory-efficient processing
        """
        self.scaling_method = scaling_method
        self.auto_dimensionality_reduction = auto_dimensionality_reduction
        self.target_dimensions = target_dimensions
        self.variance_threshold = variance_threshold
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.memory_efficient = memory_efficient
        
        # Initialize components
        self.scaler = self._create_scaler()
        self.variance_selector = None
        self.dimensionality_reducer = None
        self.feature_selector = None
        
        # State tracking
        self.is_fitted = False
        self.preprocessing_steps = []
        
        logger.info(f"ClusteringPreprocessorService initialized: scaling={scaling_method}, "
                   f"auto_reduction={auto_dimensionality_reduction}, target_dims={target_dimensions}")

    def preprocess_features(self, X: get_numpy().ndarray, labels: Optional[get_numpy().ndarray] = None) -> ClusteringPreprocessingResult:
        """Preprocess features for optimal clustering."""
        start_time = time.time()
        original_shape = X.shape
        
        try:
            # Step 1: Input validation
            validation_result = self.validate_inputs(X, labels)
            if not validation_result["valid"]:
                return ClusteringPreprocessingResult(
                    status="failed",
                    features=X,
                    info={"error": validation_result["error"]},
                    original_shape=original_shape,
                    final_shape=X.shape,
                    preprocessing_time=time.time() - start_time
                )
            
            # Step 2: Data cleaning and preparation
            X_cleaned = self._clean_data(X)
            self.preprocessing_steps.append(("data_cleaning", "completed"))
            
            # Step 3: Feature scaling
            X_scaled = self._apply_scaling(X_cleaned)
            self.preprocessing_steps.append(("feature_scaling", self.scaling_method))
            
            # Step 4: Remove constant/low-variance features
            X_filtered, removed_features = self._remove_constant_features(X_scaled)
            if removed_features > 0:
                self.preprocessing_steps.append(("variance_filtering", f"removed_{removed_features}"))
            
            # Step 5: Dimensionality reduction (if enabled and beneficial)
            feature_importance = None
            if self.auto_dimensionality_reduction and X_filtered.shape[1] > self.target_dimensions:
                X_reduced, feature_importance = self._apply_dimensionality_reduction(X_filtered, labels)
                self.preprocessing_steps.append(("dimensionality_reduction", "umap" if UMAP_AVAILABLE else "pca"))
            else:
                X_reduced = X_filtered
            
            # Final validation
            if get_numpy().any(get_numpy().isnan(X_reduced)) or get_numpy().any(get_numpy().isinf(X_reduced)):
                return ClusteringPreprocessingResult(
                    status="failed",
                    features=X,
                    info={"error": "Preprocessing produced invalid values"},
                    original_shape=original_shape,
                    final_shape=X.shape,
                    preprocessing_time=time.time() - start_time
                )
            
            preprocessing_time = time.time() - start_time
            self.is_fitted = True
            
            # Create comprehensive info
            info = {
                "scaling_method": self.scaling_method,
                "dimensionality_reduction_applied": X_reduced.shape[1] < X_filtered.shape[1],
                "removed_constant_features": removed_features,
                "preprocessing_steps": self.preprocessing_steps.copy(),
                "data_characteristics": self._analyze_data_characteristics(X_reduced),
                "clustering_readiness_score": self._compute_clustering_readiness_score(X_reduced)
            }
            
            return ClusteringPreprocessingResult(
                status="success",
                features=X_reduced,
                info=info,
                original_shape=original_shape,
                final_shape=X_reduced.shape,
                preprocessing_time=preprocessing_time,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            logger.error(f"Clustering preprocessing failed: {e}")
            return ClusteringPreprocessingResult(
                status="failed",
                features=X,
                info={"error": str(e)},
                original_shape=original_shape,
                final_shape=X.shape,
                preprocessing_time=time.time() - start_time
            )

    def validate_inputs(self, X: get_numpy().ndarray, labels: Optional[get_numpy().ndarray] = None, 
                       sample_weights: Optional[get_numpy().ndarray] = None) -> Dict[str, Any]:
        """Validate input data for clustering."""
        try:
            validation_result = {
                "valid": True,
                "warnings": [],
                "clustering_specific_checks": {}
            }
            
            # Basic shape validation
            if X.ndim != 2:
                return {"valid": False, "error": f"Input must be 2D array, got {X.ndim}D"}
            
            n_samples, n_features = X.shape
            
            # Clustering-specific minimum requirements
            if n_samples < 3:
                return {"valid": False, "error": f"Clustering requires at least 3 samples, got {n_samples}"}
            
            if n_features < 1:
                return {"valid": False, "error": f"Need at least 1 feature, got {n_features}"}
            
            # Check for invalid values
            if get_numpy().any(get_numpy().isnan(X)):
                validation_result["warnings"].append(f"Found {get_numpy().sum(get_numpy().isnan(X))} NaN values")
            
            if get_numpy().any(get_numpy().isinf(X)):
                validation_result["warnings"].append(f"Found {get_numpy().sum(get_numpy().isinf(X))} infinite values")
            
            # Clustering-specific validation
            validation_result["clustering_specific_checks"] = {
                "sample_to_feature_ratio": n_samples / n_features,
                "is_high_dimensional": n_features > 20,
                "sufficient_samples_for_clustering": n_samples >= 10,
                "constant_features_detected": get_numpy().sum(get_numpy().var(X, axis=0) < self.variance_threshold)
            }
            
            # Additional warnings for clustering
            if n_samples < 10:
                validation_result["warnings"].append("Very few samples - clustering may not be meaningful")
            
            if n_features > n_samples:
                validation_result["warnings"].append("More features than samples - consider dimensionality reduction")
            
            if validation_result["clustering_specific_checks"]["constant_features_detected"] > n_features * 0.5:
                validation_result["warnings"].append("Many constant features detected - data may have low variance")
            
            # Label validation if provided
            if labels is not None:
                if len(labels) != n_samples:
                    return {"valid": False, "error": "Labels length mismatch with features"}
                    
            # Sample weights validation if provided
            if sample_weights is not None:
                if len(sample_weights) != n_samples:
                    return {"valid": False, "error": "Sample weights length mismatch with features"}
                if get_numpy().any(sample_weights < 0):
                    return {"valid": False, "error": "Sample weights must be non-negative"}
            
            return validation_result
            
        except Exception as e:
            return {"valid": False, "error": f"Validation failed: {str(e)}"}

    def apply_dimensionality_reduction(self, X: get_numpy().ndarray) -> get_numpy().ndarray:
        """Apply UMAP dimensionality reduction for clustering."""
        if not self.auto_dimensionality_reduction or X.shape[1] <= self.target_dimensions:
            return X
        
        return self._apply_dimensionality_reduction(X)[0]

    def _create_scaler(self):
        """Create appropriate scaler based on configuration."""
        sklearn = get_sklearn()
        if self.scaling_method == "robust":
            return sklearn.preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
        elif self.scaling_method == "standard":
            return sklearn.preprocessing.StandardScaler()
        elif self.scaling_method == "none":
            return None
        else:
            logger.warning(f"Unknown scaling method '{self.scaling_method}', using robust")
            return sklearn.preprocessing.RobustScaler()

    def _clean_data(self, X: get_numpy().ndarray) -> get_numpy().ndarray:
        """Clean data by handling missing values and invalid entries."""
        X_cleaned = X.copy()
        
        # Handle NaN values
        if get_numpy().any(get_numpy().isnan(X_cleaned)):
            logger.info("Handling NaN values with median imputation")
            for col in range(X_cleaned.shape[1]):
                col_data = X_cleaned[:, col]
                if get_numpy().any(get_numpy().isnan(col_data)):
                    median_value = get_numpy().nanmedian(col_data)
                    X_cleaned[get_numpy().isnan(col_data), col] = median_value
        
        # Handle infinite values
        if get_numpy().any(get_numpy().isinf(X_cleaned)):
            logger.info("Handling infinite values by clipping")
            for col in range(X_cleaned.shape[1]):
                col_data = X_cleaned[:, col]
                finite_data = col_data[get_numpy().isfinite(col_data)]
                if len(finite_data) > 0:
                    lower_bound = get_numpy().percentile(finite_data, 1)
                    upper_bound = get_numpy().percentile(finite_data, 99)
                    X_cleaned[:, col] = get_numpy().clip(col_data, lower_bound, upper_bound)
        
        return X_cleaned

    def _apply_scaling(self, X: get_numpy().ndarray) -> get_numpy().ndarray:
        """Apply feature scaling optimized for clustering."""
        if self.scaler is None:
            return X
        
        if not self.is_fitted:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)

    def _remove_constant_features(self, X: get_numpy().ndarray) -> Tuple[get_numpy().ndarray, int]:
        """Remove constant and low-variance features."""
        if not SKLEARN_AVAILABLE:
            return X, 0
        
        original_features = X.shape[1]
        
        if self.variance_selector is None:
            sklearn = get_sklearn()
            self.variance_selector = sklearn.feature_selection.VarianceThreshold(threshold=self.variance_threshold)
            X_filtered = self.variance_selector.fit_transform(X)
        else:
            X_filtered = self.variance_selector.transform(X)
        
        removed_count = original_features - X_filtered.shape[1]
        if removed_count > 0:
            logger.info(f"Removed {removed_count} low-variance features for clustering")
        
        return X_filtered, removed_count

    def _apply_dimensionality_reduction(self, X: get_numpy().ndarray, 
                                      labels: Optional[get_numpy().ndarray] = None) -> Tuple[get_numpy().ndarray, Optional[get_numpy().ndarray]]:
        """Apply dimensionality reduction optimized for clustering."""
        # Prefer UMAP for clustering as it preserves local structure
        if UMAP_AVAILABLE:
            return self._apply_umap_reduction(X)
        elif SKLEARN_AVAILABLE:
            return self._apply_pca_reduction(X)
        else:
            logger.warning("No dimensionality reduction available")
            return X, None

    def _apply_umap_reduction(self, X: get_numpy().ndarray) -> Tuple[get_numpy().ndarray, Optional[get_numpy().ndarray]]:
        """Apply UMAP dimensionality reduction."""
        try:
            # Optimize UMAP parameters for clustering
            umap_params = self._get_optimized_umap_params(X.shape)
            
            if self.dimensionality_reducer is None:
                self.dimensionality_reducer = umap.UMAP(
                    n_components=self.target_dimensions,
                    n_neighbors=umap_params["n_neighbors"],
                    min_dist=umap_params["min_dist"],
                    metric="euclidean",
                    random_state=42,
                    low_memory=self.memory_efficient,
                    n_jobs=1 if self.memory_efficient else -1
                )
                X_reduced = self.dimensionality_reducer.fit_transform(X)
            else:
                X_reduced = self.dimensionality_reducer.transform(X)
            
            logger.info(f"UMAP reduction: {X.shape[1]} → {X_reduced.shape[1]} dimensions")
            return X_reduced, None
            
        except Exception as e:
            logger.error(f"UMAP reduction failed: {e}")
            return X, None

    def _apply_pca_reduction(self, X: get_numpy().ndarray) -> Tuple[get_numpy().ndarray, Optional[get_numpy().ndarray]]:
        """Apply PCA dimensionality reduction as fallback."""
        try:
            if self.dimensionality_reducer is None:
                sklearn = get_sklearn()
                self.dimensionality_reducer = sklearn.decomposition.PCA(
                    n_components=min(self.target_dimensions, X.shape[1]),
                    random_state=42
                )
                X_reduced = self.dimensionality_reducer.fit_transform(X)
                feature_importance = self.dimensionality_reducer.explained_variance_ratio_
            else:
                X_reduced = self.dimensionality_reducer.transform(X)
                feature_importance = getattr(self.dimensionality_reducer, "explained_variance_ratio_", None)
            
            logger.info(f"PCA reduction: {X.shape[1]} → {X_reduced.shape[1]} dimensions")
            return X_reduced, feature_importance
            
        except Exception as e:
            logger.error(f"PCA reduction failed: {e}")
            return X, None

    def _get_optimized_umap_params(self, shape: Tuple[int, int]) -> Dict[str, Any]:
        """Get optimized UMAP parameters based on data characteristics."""
        n_samples, n_features = shape
        
        # Optimize for clustering with high-dimensional data
        if n_features > 20:
            n_neighbors = max(5, min(int(n_samples * 0.1), 50))
            min_dist = 0.0  # Tighter clusters for high-dimensional data
        else:
            n_neighbors = min(self.umap_n_neighbors, n_samples // 3)
            min_dist = self.umap_min_dist
        
        return {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist
        }

    def _analyze_data_characteristics(self, X: get_numpy().ndarray) -> Dict[str, Any]:
        """Analyze data characteristics relevant for clustering."""
        n_samples, n_features = X.shape
        
        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "is_high_dimensional": n_features > 20,
            "feature_scale_range": {
                "min": float(get_numpy().min(X)),
                "max": float(get_numpy().max(X)),
                "std": float(get_numpy().std(X))
            },
            "feature_variances": {
                "mean_variance": float(get_numpy().mean(get_numpy().var(X, axis=0))),
                "min_variance": float(get_numpy().min(get_numpy().var(X, axis=0))),
                "max_variance": float(get_numpy().max(get_numpy().var(X, axis=0)))
            },
            "sample_to_feature_ratio": n_samples / n_features,
            "estimated_clustering_difficulty": self._estimate_clustering_difficulty(X)
        }

    def _estimate_clustering_difficulty(self, X: get_numpy().ndarray) -> str:
        """Estimate clustering difficulty based on data characteristics."""
        n_samples, n_features = X.shape
        
        # Factors that make clustering more difficult
        difficulty_score = 0
        
        if n_features > 50:
            difficulty_score += 2
        elif n_features > 20:
            difficulty_score += 1
        
        if n_samples < 100:
            difficulty_score += 2
        elif n_samples < 500:
            difficulty_score += 1
        
        if n_features > n_samples:
            difficulty_score += 2
        
        # Check data spread
        feature_vars = get_numpy().var(X, axis=0)
        if get_numpy().std(feature_vars) > get_numpy().mean(feature_vars):
            difficulty_score += 1  # High variance in feature variances
        
        if difficulty_score >= 4:
            return "high"
        elif difficulty_score >= 2:
            return "moderate"
        else:
            return "low"

    def _compute_clustering_readiness_score(self, X: get_numpy().ndarray) -> float:
        """Compute a score indicating how ready the data is for clustering."""
        try:
            score_components = []
            
            # Factor 1: No invalid values
            has_invalid = get_numpy().any(get_numpy().isnan(X)) or get_numpy().any(get_numpy().isinf(X))
            score_components.append(0.0 if has_invalid else 1.0)
            
            # Factor 2: Reasonable dimensionality
            n_samples, n_features = X.shape
            dim_ratio = min(1.0, n_samples / (n_features * 5))  # Prefer more samples per feature
            score_components.append(dim_ratio)
            
            # Factor 3: Feature variance distribution
            feature_vars = get_numpy().var(X, axis=0)
            if len(feature_vars) > 0:
                var_consistency = 1.0 - (get_numpy().std(feature_vars) / (get_numpy().mean(feature_vars) + 1e-8))
                score_components.append(max(0.0, min(1.0, var_consistency)))
            
            # Factor 4: Data scale appropriateness
            data_range = get_numpy().max(X) - get_numpy().min(X)
            scale_score = 1.0 if 0.1 <= data_range <= 10 else 0.5
            score_components.append(scale_score)
            
            # Compute overall score
            overall_score = get_numpy().mean(score_components)
            return float(get_numpy().clip(overall_score, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Clustering readiness score computation failed: {e}")
            return 0.5

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing operations performed."""
        return {
            "is_fitted": self.is_fitted,
            "scaling_method": self.scaling_method,
            "auto_dimensionality_reduction": self.auto_dimensionality_reduction,
            "target_dimensions": self.target_dimensions,
            "umap_available": UMAP_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "preprocessing_steps": self.preprocessing_steps.copy(),
            "memory_efficient": self.memory_efficient
        }

    def reset(self):
        """Reset the preprocessor to unfitted state."""
        self.scaler = self._create_scaler()
        self.variance_selector = None
        self.dimensionality_reducer = None
        self.feature_selector = None
        self.is_fitted = False
        self.preprocessing_steps = []
        logger.info("ClusteringPreprocessorService reset to unfitted state")

class ClusteringPreprocessorServiceFactory:
    """Factory for creating clustering preprocessor services with different configurations."""

    @staticmethod
    def create_minimal_preprocessor() -> ClusteringPreprocessorService:
        """Create minimal preprocessor with basic scaling only."""
        return ClusteringPreprocessorService(
            scaling_method="standard",
            auto_dimensionality_reduction=False
        )

    @staticmethod
    def create_high_dimensional_preprocessor() -> ClusteringPreprocessorService:
        """Create preprocessor optimized for high-dimensional clustering."""
        return ClusteringPreprocessorService(
            scaling_method="robust",
            auto_dimensionality_reduction=True,
            target_dimensions=10,
            umap_n_neighbors=15,
            umap_min_dist=0.0,  # Better for clustering
            memory_efficient=True
        )

    @staticmethod
    def create_memory_efficient_preprocessor() -> ClusteringPreprocessorService:
        """Create memory-efficient preprocessor for large datasets."""
        return ClusteringPreprocessorService(
            scaling_method="robust",
            auto_dimensionality_reduction=True,
            target_dimensions=8,  # Lower dimensionality for memory
            memory_efficient=True
        )

    @staticmethod
    def create_quality_focused_preprocessor() -> ClusteringPreprocessorService:
        """Create preprocessor focused on clustering quality."""
        return ClusteringPreprocessorService(
            scaling_method="robust",
            auto_dimensionality_reduction=True,
            target_dimensions=15,  # Higher dimensionality for quality
            umap_n_neighbors=20,
            umap_min_dist=0.0,
            memory_efficient=False
        )