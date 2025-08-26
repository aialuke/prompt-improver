"""Data preprocessing service for dimensionality reduction.

Provides comprehensive data preprocessing including:
- Input validation and quality checks
- Feature scaling and normalization
- Feature selection and filtering
- Data cleaning and outlier handling
- Preparation for various reduction methods

Follows clean architecture patterns with protocol-based interfaces.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from typing import TYPE_CHECKING
from prompt_improver.core.utils.lazy_ml_loader import get_numpy, get_sklearn

if TYPE_CHECKING:
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
    from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    import numpy as np
else:
    # Runtime lazy loading
    def _get_sklearn_imports():
        sklearn = get_sklearn()
        return (
            sklearn.feature_selection.VarianceThreshold,
            sklearn.preprocessing.RobustScaler,
            sklearn.preprocessing.StandardScaler,
            sklearn.preprocessing.MinMaxScaler,
            sklearn.feature_selection.SelectKBest,
            sklearn.feature_selection.f_classif,
            sklearn.feature_selection.mutual_info_classif,
            sklearn.decomposition.PCA
        )
    
    (VarianceThreshold, RobustScaler, StandardScaler, MinMaxScaler,
     SelectKBest, f_classif, mutual_info_classif, PCA) = _get_sklearn_imports()

from . import PreprocessorProtocol, PreprocessingResult

logger = logging.getLogger(__name__)

# Check sklearn availability
try:
    get_sklearn()
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available for advanced preprocessing")

class DataPreprocessorService:
    """Service for preprocessing data before dimensionality reduction following clean architecture."""

    def __init__(self, scaling_method: str = "robust", remove_constant_features: bool = True,
                 variance_threshold: float = 1e-6, feature_selection_method: Optional[str] = None,
                 feature_selection_k: int = 20, outlier_detection: bool = False):
        """Initialize data preprocessor service.
        
        Args:
            scaling_method: Scaling method ("robust", "standard", "minmax", "none")
            remove_constant_features: Whether to remove constant/low-variance features
            variance_threshold: Minimum variance threshold for feature removal
            feature_selection_method: Feature selection method ("mutual_info", "f_score", None)
            feature_selection_k: Number of top features to select
            outlier_detection: Whether to detect and handle outliers
        """
        self.scaling_method = scaling_method
        self.remove_constant_features = remove_constant_features
        self.variance_threshold = variance_threshold
        self.feature_selection_method = feature_selection_method
        self.feature_selection_k = feature_selection_k
        self.outlier_detection = outlier_detection
        
        # Initialize components
        self.scaler = self._create_scaler()
        self.variance_selector = None
        self.feature_selector = None
        
        # State tracking
        self.is_fitted = False
        self.preprocessing_pipeline = []
        
        logger.info(f"DataPreprocessorService initialized: scaling={scaling_method}, "
                   f"variance_threshold={variance_threshold}, feature_selection={feature_selection_method}")

    def preprocess(self, X: get_numpy().ndarray, y: Optional[get_numpy().ndarray] = None) -> PreprocessingResult:
        """Preprocess data for dimensionality reduction."""
        start_time = time.time()
        original_shape = X.shape
        
        try:
            # Step 1: Input validation
            validation_result = self.validate_input(X)
            if not validation_result["valid"]:
                return PreprocessingResult(
                    status="failed",
                    features=X,
                    info={"error": validation_result["error"]},
                    original_shape=original_shape,
                    final_shape=X.shape,
                    preprocessing_time=time.time() - start_time
                )
            
            # Step 2: Handle missing values and outliers
            X_cleaned = self._clean_data(X)
            self.preprocessing_pipeline.append(("data_cleaning", "completed"))
            
            # Step 3: Feature scaling
            X_scaled = self._apply_scaling(X_cleaned)
            self.preprocessing_pipeline.append(("feature_scaling", self.scaling_method))
            
            # Step 4: Remove constant features
            X_filtered = X_scaled
            removed_features_count = 0
            if self.remove_constant_features:
                X_filtered, removed_features_count = self._remove_constant_features(X_scaled)
                self.preprocessing_pipeline.append(("constant_removal", f"removed_{removed_features_count}"))
            
            # Step 5: Feature selection (if enabled and labels available)
            feature_importance = None
            if self.feature_selection_method and y is not None and SKLEARN_AVAILABLE:
                X_filtered, feature_importance = self._apply_feature_selection(X_filtered, y)
                self.preprocessing_pipeline.append(("feature_selection", self.feature_selection_method))
            
            # Final validation
            if get_numpy().any(get_numpy().isnan(X_filtered)) or get_numpy().any(get_numpy().isinf(X_filtered)):
                return PreprocessingResult(
                    status="failed",
                    features=X,
                    info={"error": "Preprocessing produced invalid values (NaN or inf)"},
                    original_shape=original_shape,
                    final_shape=X.shape,
                    preprocessing_time=time.time() - start_time
                )
            
            preprocessing_time = time.time() - start_time
            self.is_fitted = True
            
            # Create detailed info
            info = {
                "scaling_method": self.scaling_method,
                "removed_constant_features": removed_features_count,
                "feature_selection_applied": self.feature_selection_method is not None,
                "outlier_detection_applied": self.outlier_detection,
                "preprocessing_pipeline": self.preprocessing_pipeline.copy(),
                "feature_importance": feature_importance.tolist() if feature_importance is not None else None,
                "data_quality_score": self._compute_data_quality_score(X_filtered)
            }
            
            return PreprocessingResult(
                status="success",
                features=X_filtered,
                info=info,
                original_shape=original_shape,
                final_shape=X_filtered.shape,
                preprocessing_time=preprocessing_time
            )
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return PreprocessingResult(
                status="failed",
                features=X,
                info={"error": str(e)},
                original_shape=original_shape,
                final_shape=X.shape,
                preprocessing_time=time.time() - start_time
            )

    def validate_input(self, X: get_numpy().ndarray) -> Dict[str, Any]:
        """Validate input data quality and characteristics."""
        try:
            validation_result = {
                "valid": True,
                "warnings": [],
                "statistics": {}
            }
            
            # Basic shape validation
            if X.ndim != 2:
                return {
                    "valid": False,
                    "error": f"Input must be 2D array, got {X.ndim}D with shape {X.shape}"
                }
            
            n_samples, n_features = X.shape
            
            # Check minimum requirements
            if n_samples < 3:
                return {
                    "valid": False,
                    "error": f"Need at least 3 samples, got {n_samples}"
                }
            
            if n_features < 1:
                return {
                    "valid": False,
                    "error": f"Need at least 1 feature, got {n_features}"
                }
            
            # Check for invalid values
            if get_numpy().any(get_numpy().isnan(X)):
                nan_count = get_numpy().sum(get_numpy().isnan(X))
                validation_result["warnings"].append(f"Found {nan_count} NaN values")
            
            if get_numpy().any(get_numpy().isinf(X)):
                inf_count = get_numpy().sum(get_numpy().isinf(X))
                validation_result["warnings"].append(f"Found {inf_count} infinite values")
            
            # Compute statistics
            validation_result["statistics"] = {
                "n_samples": n_samples,
                "n_features": n_features,
                "data_range": {
                    "min": float(get_numpy().nanmin(X)),
                    "max": float(get_numpy().nanmax(X)),
                    "mean": float(get_numpy().nanmean(X)),
                    "std": float(get_numpy().nanstd(X))
                },
                "constant_features": int(get_numpy().sum(get_numpy().var(X, axis=0) < self.variance_threshold)),
                "missing_values": int(get_numpy().sum(get_numpy().isnan(X))),
                "infinite_values": int(get_numpy().sum(get_numpy().isinf(X)))
            }
            
            # Check data characteristics
            if n_features > n_samples:
                validation_result["warnings"].append(
                    f"High-dimensional data: {n_features} features > {n_samples} samples"
                )
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation failed: {str(e)}"
            }

    def scale_features(self, X: get_numpy().ndarray) -> get_numpy().ndarray:
        """Scale features for optimal reduction performance."""
        return self._apply_scaling(X)

    def _create_scaler(self):
        """Create appropriate scaler based on configuration."""
        if self.scaling_method == "robust":
            return RobustScaler(quantile_range=(25.0, 75.0))
        elif self.scaling_method == "standard":
            return StandardScaler()
        elif self.scaling_method == "minmax":
            return MinMaxScaler()
        elif self.scaling_method == "none":
            return None
        else:
            logger.warning(f"Unknown scaling method '{self.scaling_method}', using robust")
            return RobustScaler()

    def _clean_data(self, X: get_numpy().ndarray) -> get_numpy().ndarray:
        """Clean data by handling missing values and outliers."""
        X_cleaned = X.copy()
        
        # Handle NaN values by replacing with column median
        if get_numpy().any(get_numpy().isnan(X_cleaned)):
            logger.info("Handling NaN values with column median imputation")
            for col in range(X_cleaned.shape[1]):
                col_data = X_cleaned[:, col]
                if get_numpy().any(get_numpy().isnan(col_data)):
                    median_value = get_numpy().nanmedian(col_data)
                    X_cleaned[get_numpy().isnan(col_data), col] = median_value
        
        # Handle infinite values by clipping to reasonable bounds
        if get_numpy().any(get_numpy().isinf(X_cleaned)):
            logger.info("Handling infinite values by clipping")
            # Clip to 99.9% percentile bounds
            for col in range(X_cleaned.shape[1]):
                col_data = X_cleaned[:, col]
                finite_data = col_data[get_numpy().isfinite(col_data)]
                if len(finite_data) > 0:
                    lower_bound = get_numpy().percentile(finite_data, 0.1)
                    upper_bound = get_numpy().percentile(finite_data, 99.9)
                    X_cleaned[:, col] = get_numpy().clip(col_data, lower_bound, upper_bound)
        
        # Outlier detection and handling (if enabled)
        if self.outlier_detection:
            X_cleaned = self._handle_outliers(X_cleaned)
        
        return X_cleaned

    def _handle_outliers(self, X: get_numpy().ndarray) -> get_numpy().ndarray:
        """Handle outliers using IQR-based method."""
        X_outlier_handled = X.copy()
        
        for col in range(X.shape[1]):
            col_data = X[:, col]
            Q1 = get_numpy().percentile(col_data, 25)
            Q3 = get_numpy().percentile(col_data, 75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Clip outliers
            X_outlier_handled[:, col] = get_numpy().clip(col_data, lower_bound, upper_bound)
        
        return X_outlier_handled

    def _apply_scaling(self, X: get_numpy().ndarray) -> get_numpy().ndarray:
        """Apply feature scaling."""
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
            self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
            X_filtered = self.variance_selector.fit_transform(X)
        else:
            X_filtered = self.variance_selector.transform(X)
        
        removed_count = original_features - X_filtered.shape[1]
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} low-variance features")
        
        return X_filtered, removed_count

    def _apply_feature_selection(self, X: get_numpy().ndarray, y: get_numpy().ndarray) -> Tuple[get_numpy().ndarray, get_numpy().ndarray]:
        """Apply feature selection based on labels."""
        if not SKLEARN_AVAILABLE:
            return X, None
        
        # Determine number of features to select
        k = min(self.feature_selection_k, X.shape[1])
        
        if self.feature_selector is None:
            # Choose score function based on method
            if self.feature_selection_method == "mutual_info":
                score_func = mutual_info_classif
            elif self.feature_selection_method == "f_score":
                score_func = f_classif
            else:
                score_func = f_classif  # Default
            
            self.feature_selector = SelectKBest(score_func=score_func, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            feature_scores = self.feature_selector.scores_
        else:
            X_selected = self.feature_selector.transform(X)
            feature_scores = self.feature_selector.scores_
        
        logger.info(f"Selected {k} features using {self.feature_selection_method}")
        
        return X_selected, feature_scores

    def _compute_data_quality_score(self, X: get_numpy().ndarray) -> float:
        """Compute overall data quality score."""
        try:
            quality_factors = []
            
            # Factor 1: No invalid values
            has_invalid = get_numpy().any(get_numpy().isnan(X)) or get_numpy().any(get_numpy().isinf(X))
            quality_factors.append(0.0 if has_invalid else 1.0)
            
            # Factor 2: Feature variance (higher variance = better)
            feature_variances = get_numpy().var(X, axis=0)
            avg_variance = get_numpy().mean(feature_variances)
            variance_score = min(1.0, avg_variance / (avg_variance + 1))  # Normalize
            quality_factors.append(variance_score)
            
            # Factor 3: No constant features
            constant_features = get_numpy().sum(feature_variances < self.variance_threshold)
            constant_score = max(0.0, 1.0 - constant_features / X.shape[1])
            quality_factors.append(constant_score)
            
            # Factor 4: Reasonable data range (not too extreme)
            data_range = get_numpy().max(X) - get_numpy().min(X)
            range_score = 1.0 if 0.1 <= data_range <= 1000 else 0.5
            quality_factors.append(range_score)
            
            # Compute weighted average
            quality_score = get_numpy().mean(quality_factors)
            
            return float(get_numpy().clip(quality_score, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Quality score computation failed: {e}")
            return 0.5

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing operations performed."""
        return {
            "is_fitted": self.is_fitted,
            "scaling_method": self.scaling_method,
            "remove_constant_features": self.remove_constant_features,
            "variance_threshold": self.variance_threshold,
            "feature_selection_method": self.feature_selection_method,
            "feature_selection_k": self.feature_selection_k,
            "outlier_detection": self.outlier_detection,
            "preprocessing_pipeline": self.preprocessing_pipeline.copy() if hasattr(self, 'preprocessing_pipeline') else []
        }

    def reset(self):
        """Reset the preprocessor to unfitted state."""
        self.scaler = self._create_scaler()
        self.variance_selector = None
        self.feature_selector = None
        self.is_fitted = False
        self.preprocessing_pipeline = []
        logger.info("DataPreprocessorService reset to unfitted state")

class DataPreprocessorServiceFactory:
    """Factory for creating preprocessor services with different configurations."""

    @staticmethod
    def create_minimal_preprocessor() -> DataPreprocessorService:
        """Create minimal preprocessor with basic scaling only."""
        return DataPreprocessorService(
            scaling_method="standard",
            remove_constant_features=False,
            feature_selection_method=None,
            outlier_detection=False
        )

    @staticmethod
    def create_robust_preprocessor() -> DataPreprocessorService:
        """Create robust preprocessor for noisy data."""
        return DataPreprocessorService(
            scaling_method="robust",
            remove_constant_features=True,
            variance_threshold=1e-6,
            outlier_detection=True
        )

    @staticmethod
    def create_high_dimensional_preprocessor() -> DataPreprocessorService:
        """Create preprocessor optimized for high-dimensional data."""
        return DataPreprocessorService(
            scaling_method="robust",
            remove_constant_features=True,
            variance_threshold=1e-4,
            feature_selection_method="mutual_info",
            feature_selection_k=50,
            outlier_detection=True
        )

    @staticmethod
    def create_fast_preprocessor() -> DataPreprocessorService:
        """Create preprocessor optimized for speed."""
        return DataPreprocessorService(
            scaling_method="minmax",
            remove_constant_features=True,
            feature_selection_method=None,
            outlier_detection=False
        )