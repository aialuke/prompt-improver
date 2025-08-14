"""Dimensionality Reducer Facade.

Coordinates dimensionality reduction services following clean architecture patterns.
Replaces the monolithic AdvancedDimensionalityReducer with a facade that orchestrates:
- DataPreprocessorService: Data cleaning, validation, scaling
- ClassicalReducerService: Traditional algorithms (PCA, t-SNE, UMAP)
- NeuralReducerService: Neural network architectures
- ReductionEvaluatorService: Quality evaluation and metrics

Maintains backward compatibility while providing improved modularity and testability.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import asyncio

import numpy as np

from . import ReductionResult, EvaluationMetrics
from .data_preprocessor_service import DataPreprocessorService, DataPreprocessorServiceFactory
from .classical_reducer_service import ClassicalReducerService, ClassicalReducerServiceFactory  
from .neural_reducer_service import NeuralReducerService, NeuralReducerServiceFactory
from .reduction_evaluator_service import ReductionEvaluatorService, ReductionEvaluatorServiceFactory

logger = logging.getLogger(__name__)

class DimensionalityReducerFacade:
    """Facade coordinating dimensionality reduction services following clean architecture."""

    def __init__(self, target_dimensions: int = 10, auto_method_selection: bool = True,
                 enable_neural_methods: bool = True, enable_evaluation: bool = True,
                 preprocessing_strategy: str = "robust", evaluation_strategy: str = "comprehensive"):
        """Initialize dimensionality reducer facade.
        
        Args:
            target_dimensions: Target number of dimensions
            auto_method_selection: Whether to automatically select best method
            enable_neural_methods: Whether to enable neural network methods
            enable_evaluation: Whether to enable quality evaluation
            preprocessing_strategy: Preprocessing strategy ("minimal", "robust", "high_dimensional")
            evaluation_strategy: Evaluation strategy ("fast", "comprehensive", "variance_focused")
        """
        self.target_dimensions = target_dimensions
        self.auto_method_selection = auto_method_selection
        self.enable_neural_methods = enable_neural_methods
        self.enable_evaluation = enable_evaluation
        
        # Initialize service components
        self.preprocessor = self._create_preprocessor(preprocessing_strategy)
        self.evaluator = self._create_evaluator(evaluation_strategy) if enable_evaluation else None
        
        # Service registry for different reduction methods
        self.method_services = {}
        self.available_methods = []
        
        # Performance tracking
        self.reduction_history = []
        self.method_performance = {}
        
        # Initialize available methods
        self._initialize_method_services()
        
        logger.info(f"DimensionalityReducerFacade initialized: target_dims={target_dimensions}, "
                   f"methods={len(self.available_methods)}, neural_enabled={enable_neural_methods}")

    def _create_preprocessor(self, strategy: str) -> DataPreprocessorService:
        """Create preprocessor service based on strategy."""
        if strategy == "minimal":
            return DataPreprocessorServiceFactory.create_minimal_preprocessor()
        elif strategy == "robust":
            return DataPreprocessorServiceFactory.create_robust_preprocessor()
        elif strategy == "high_dimensional":
            return DataPreprocessorServiceFactory.create_high_dimensional_preprocessor()
        elif strategy == "fast":
            return DataPreprocessorServiceFactory.create_fast_preprocessor()
        else:
            logger.warning(f"Unknown preprocessing strategy '{strategy}', using robust")
            return DataPreprocessorServiceFactory.create_robust_preprocessor()

    def _create_evaluator(self, strategy: str) -> ReductionEvaluatorService:
        """Create evaluator service based on strategy."""
        if strategy == "fast":
            return ReductionEvaluatorServiceFactory.create_fast_evaluator()
        elif strategy == "comprehensive":
            return ReductionEvaluatorServiceFactory.create_comprehensive_evaluator()
        elif strategy == "variance_focused":
            return ReductionEvaluatorServiceFactory.create_variance_focused_evaluator()
        else:
            logger.warning(f"Unknown evaluation strategy '{strategy}', using comprehensive")
            return ReductionEvaluatorServiceFactory.create_comprehensive_evaluator()

    def _initialize_method_services(self):
        """Initialize available reduction method services."""
        # Classical methods (always available)
        classical_methods = ["pca", "incremental_pca", "kernel_pca", "ica", "truncated_svd", 
                           "lda", "tsne", "isomap", "gaussian_rp", "sparse_rp"]
        
        # Add UMAP if available
        try:
            ClassicalReducerServiceFactory.create_umap_service(n_components=2)
            classical_methods.append("umap")
        except ImportError:
            pass
        
        # Register classical methods
        for method in classical_methods:
            self.method_services[method] = {
                "type": "classical",
                "factory": ClassicalReducerServiceFactory,
                "params": {"method": method, "n_components": self.target_dimensions}
            }
            self.available_methods.append(method)
        
        # Neural methods (if enabled and available)
        if self.enable_neural_methods:
            try:
                # Test neural service availability
                NeuralReducerServiceFactory.create_autoencoder_service(latent_dim=2)
                neural_methods = ["autoencoder", "vae", "transformer"]
                
                for method in neural_methods:
                    self.method_services[method] = {
                        "type": "neural",
                        "factory": NeuralReducerServiceFactory,
                        "params": {"latent_dim": self.target_dimensions}
                    }
                    self.available_methods.append(method)
                    
            except ImportError:
                logger.warning("Neural methods not available (PyTorch not installed)")

    async def reduce_dimensions(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                              method: Optional[str] = None) -> ReductionResult:
        """Reduce dimensionality of input data with optimal method selection.
        
        Args:
            X: Input feature matrix (n_samples, n_features)
            y: Optional target labels for supervised methods
            method: Specific method to use (if None, auto-select)
            
        Returns:
            ReductionResult with transformed data and evaluation metrics
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting dimensionality reduction: {X.shape[0]} samples x {X.shape[1]} features")
            
            # Input validation
            if X.ndim != 2:
                raise ValueError("Input data must be 2D array")
            
            if X.shape[1] <= self.target_dimensions:
                logger.warning(f"Input dimensionality ({X.shape[1]}) already <= target ({self.target_dimensions})")
                return self._create_identity_result(X, start_time)
            
            # Step 1: Preprocessing
            preprocessing_result = self.preprocessor.preprocess(X, y)
            if preprocessing_result.status != "success":
                raise ValueError(f"Preprocessing failed: {preprocessing_result.info.get('error', 'Unknown error')}")
            
            X_processed = preprocessing_result.features
            
            # Step 2: Method selection
            if method is None and self.auto_method_selection:
                method = await self._select_optimal_method(X_processed, y)
            elif method is None:
                method = "pca"  # Default fallback
            
            # Validate method availability
            if method not in self.available_methods:
                logger.warning(f"Method '{method}' not available, falling back to PCA")
                method = "pca"
            
            # Step 3: Apply dimensionality reduction
            reduction_result = await self._apply_reduction_method(X_processed, y, method)
            
            # Step 4: Evaluate result quality (if enabled)
            if self.enable_evaluation and self.evaluator:
                evaluation_metrics = self.evaluator.evaluate_quality(X_processed, reduction_result.transformed_data, y)
                reduction_result.evaluation_metrics = evaluation_metrics.__dict__
                reduction_result.quality_score = evaluation_metrics.overall_quality
            
            # Step 5: Update performance tracking
            self._update_performance_tracking(reduction_result)
            
            processing_time = time.time() - start_time
            reduction_result.processing_time = processing_time
            
            logger.info(f"Dimensionality reduction completed: {reduction_result.original_dimensions}→"
                       f"{reduction_result.reduced_dimensions} (method: {method}, "
                       f"quality: {reduction_result.quality_score:.3f}, time: {processing_time:.2f}s)")
            
            return reduction_result
            
        except Exception as e:
            logger.error(f"Dimensionality reduction failed: {e}", exc_info=True)
            raise

    async def _select_optimal_method(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> str:
        """Intelligently select the optimal dimensionality reduction method."""
        n_samples, n_features = X.shape
        
        # Filter suitable methods based on data characteristics
        suitable_methods = []
        
        for method in self.available_methods:
            if self._is_method_suitable(method, n_samples, n_features, y is not None):
                suitable_methods.append(method)
        
        if not suitable_methods:
            logger.warning("No suitable methods found, falling back to PCA")
            return "pca"
        
        # Use performance history if available
        if self.method_performance:
            best_performing_method = max(
                (m for m in suitable_methods if m in self.method_performance),
                key=lambda m: np.mean(self.method_performance[m]),
                default=None
            )
            if best_performing_method and np.mean(self.method_performance[best_performing_method]) > 0.7:
                return best_performing_method
        
        # Apply heuristic selection rules
        return self._apply_method_selection_heuristics(suitable_methods, n_samples, n_features, y is not None)

    def _is_method_suitable(self, method: str, n_samples: int, n_features: int, has_labels: bool) -> bool:
        """Check if a method is suitable for the given data characteristics."""
        # Supervised methods require labels
        if method == "lda" and not has_labels:
            return False
        
        # Neural methods require sufficient data
        if method in ["autoencoder", "vae", "transformer"] and n_samples < 100:
            return False
        
        # Memory-intensive methods for large datasets
        if method in ["tsne", "isomap"] and n_samples > 5000:
            return False
        
        # High-dimensional specific constraints
        if method == "kernel_pca" and n_features > 100:
            return False
        
        return True

    def _apply_method_selection_heuristics(self, suitable_methods: List[str], 
                                         n_samples: int, n_features: int, has_labels: bool) -> str:
        """Apply heuristic rules for method selection."""
        # Prefer supervised methods when labels are available
        if has_labels and "lda" in suitable_methods:
            return "lda"
        
        # For large datasets, prefer scalable methods
        if n_samples > 5000:
            scalable_methods = [m for m in suitable_methods if m in ["pca", "incremental_pca", "sparse_rp", "gaussian_rp"]]
            if scalable_methods:
                return scalable_methods[0]
        
        # For high-dimensional data, prefer manifold learning
        if n_features > 50:
            manifold_methods = [m for m in suitable_methods if m in ["umap", "autoencoder", "vae"]]
            if manifold_methods:
                return manifold_methods[0]
        
        # For neural methods, prefer based on data size
        if self.enable_neural_methods and n_samples > 500:
            neural_methods = [m for m in suitable_methods if m in ["autoencoder", "vae", "transformer"]]
            if neural_methods:
                if n_samples > 2000:
                    return "transformer"
                else:
                    return "autoencoder"
        
        # Default preferences
        if "umap" in suitable_methods and n_features > 20:
            return "umap"
        
        return "pca"  # Safe default

    async def _apply_reduction_method(self, X: np.ndarray, y: Optional[np.ndarray], method: str) -> ReductionResult:
        """Apply specific dimensionality reduction method."""
        method_info = self.method_services[method]
        
        try:
            # Create service instance
            if method_info["type"] == "classical":
                if method == "pca":
                    service = method_info["factory"].create_pca_service(n_components=self.target_dimensions)
                elif method == "umap":
                    service = method_info["factory"].create_umap_service(n_components=self.target_dimensions)
                elif method == "tsne":
                    service = method_info["factory"].create_tsne_service(n_components=min(self.target_dimensions, 3))
                else:
                    service = ClassicalReducerService(method=method, n_components=self.target_dimensions)
                    
            elif method_info["type"] == "neural":
                if method == "autoencoder":
                    service = method_info["factory"].create_autoencoder_service(latent_dim=self.target_dimensions)
                elif method == "vae":
                    service = method_info["factory"].create_vae_service(latent_dim=self.target_dimensions)
                elif method == "transformer":
                    service = method_info["factory"].create_transformer_service(latent_dim=self.target_dimensions)
                else:
                    service = NeuralReducerService(model_type=method, latent_dim=self.target_dimensions)
            else:
                raise ValueError(f"Unknown method type: {method_info['type']}")
            
            # Fit and transform
            start_time = time.time()
            X_reduced = service.fit_transform(X, y)
            processing_time = time.time() - start_time
            
            # Calculate reconstruction error if possible
            reconstruction_error = 0.0
            if hasattr(service, 'get_reconstruction_error'):
                reconstruction_error = service.get_reconstruction_error(X)
            
            # Calculate variance preserved if possible  
            variance_preserved = 0.0
            if hasattr(service, 'get_explained_variance_ratio'):
                explained_variance = service.get_explained_variance_ratio()
                if explained_variance is not None:
                    variance_preserved = np.sum(explained_variance)
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(service, 'get_feature_importance'):
                feature_importance = service.get_feature_importance()
            
            return ReductionResult(
                method=method,
                original_dimensions=X.shape[1],
                reduced_dimensions=X_reduced.shape[1],
                variance_preserved=variance_preserved,
                reconstruction_error=reconstruction_error,
                processing_time=processing_time,
                quality_score=0.0,  # Will be computed by evaluator
                transformed_data=X_reduced,
                transformer=service,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            logger.error(f"Failed to apply method '{method}': {e}")
            raise

    def _create_identity_result(self, X: np.ndarray, start_time: float) -> ReductionResult:
        """Create identity result when no reduction is needed."""
        return ReductionResult(
            method="identity",
            original_dimensions=X.shape[1],
            reduced_dimensions=X.shape[1],
            variance_preserved=1.0,
            reconstruction_error=0.0,
            processing_time=time.time() - start_time,
            quality_score=1.0,
            transformed_data=X.copy(),
            transformer=None
        )

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

    def get_available_methods(self) -> List[str]:
        """Get list of available reduction methods."""
        return self.available_methods.copy()

    def get_method_info(self, method: str) -> Dict[str, Any]:
        """Get information about a specific method."""
        if method not in self.method_services:
            return {"available": False, "reason": "Method not found"}
        
        method_info = self.method_services[method].copy()
        method_info["available"] = True
        
        # Add performance history if available
        if method in self.method_performance:
            scores = self.method_performance[method]
            method_info["performance"] = {
                "avg_quality": float(np.mean(scores)),
                "std_quality": float(np.std(scores)),
                "usage_count": len(scores)
            }
        
        return method_info

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.reduction_history:
            return {"status": "no_data"}
        
        recent_results = self.reduction_history[-10:]
        
        summary = {
            "status": "available",
            "total_reductions": len(self.reduction_history),
            "recent_performance": {
                "avg_quality_score": float(np.mean([r.quality_score for r in recent_results])),
                "avg_processing_time": float(np.mean([r.processing_time for r in recent_results])),
                "avg_variance_preserved": float(np.mean([r.variance_preserved for r in recent_results])),
                "avg_dimensionality_reduction": float(np.mean([
                    (r.original_dimensions - r.reduced_dimensions) / r.original_dimensions
                    for r in recent_results
                ]))
            },
            "method_performance": {},
            "available_methods": self.available_methods.copy()
        }
        
        # Method-specific performance
        for method, scores in self.method_performance.items():
            if scores:
                summary["method_performance"][method] = {
                    "avg_quality": float(np.mean(scores)),
                    "std_quality": float(np.std(scores)),
                    "usage_count": len(scores),
                    "reliability": float(np.mean([s > 0.5 for s in scores]))
                }
        
        return summary

    def recommend_method(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Recommend the best method for given data characteristics."""
        n_samples, n_features = X.shape
        
        # Get suitable methods
        suitable_methods = [
            method for method in self.available_methods
            if self._is_method_suitable(method, n_samples, n_features, y is not None)
        ]
        
        # Get recommended method
        if suitable_methods:
            recommended = self._apply_method_selection_heuristics(suitable_methods, n_samples, n_features, y is not None)
        else:
            recommended = "pca"
        
        return {
            "recommended_method": recommended,
            "suitable_methods": suitable_methods,
            "data_characteristics": {
                "n_samples": n_samples,
                "n_features": n_features,
                "has_labels": y is not None,
                "is_high_dimensional": n_features > 50,
                "is_large_dataset": n_samples > 5000
            },
            "reasoning": self._get_recommendation_reasoning(recommended, n_samples, n_features, y is not None)
        }

    def _get_recommendation_reasoning(self, method: str, n_samples: int, n_features: int, has_labels: bool) -> str:
        """Get reasoning for method recommendation."""
        reasons = []
        
        if method == "lda" and has_labels:
            reasons.append("Supervised method chosen because labels are available")
        
        if method in ["pca", "incremental_pca"] and n_samples > 5000:
            reasons.append("Scalable method chosen for large dataset")
        
        if method in ["umap", "autoencoder"] and n_features > 50:
            reasons.append("Nonlinear method chosen for high-dimensional data")
        
        if method in ["autoencoder", "vae", "transformer"] and n_samples > 500:
            reasons.append("Neural method chosen for sufficient data size")
        
        if not reasons:
            reasons.append("Default method selection based on data characteristics")
        
        return "; ".join(reasons)

class DimensionalityReducerFacadeFactory:
    """Factory for creating dimensionality reducer facades with different configurations."""

    @staticmethod
    def create_fast_reducer(target_dimensions: int = 10) -> DimensionalityReducerFacade:
        """Create fast reducer optimized for speed."""
        return DimensionalityReducerFacade(
            target_dimensions=target_dimensions,
            enable_neural_methods=False,
            enable_evaluation=False,
            preprocessing_strategy="minimal",
            evaluation_strategy="fast"
        )

    @staticmethod
    def create_high_quality_reducer(target_dimensions: int = 10) -> DimensionalityReducerFacade:
        """Create reducer optimized for quality."""
        return DimensionalityReducerFacade(
            target_dimensions=target_dimensions,
            enable_neural_methods=True,
            enable_evaluation=True,
            preprocessing_strategy="robust",
            evaluation_strategy="comprehensive"
        )

    @staticmethod
    def create_high_dimensional_reducer(target_dimensions: int = 15) -> DimensionalityReducerFacade:
        """Create reducer optimized for high-dimensional data."""
        return DimensionalityReducerFacade(
            target_dimensions=target_dimensions,
            enable_neural_methods=True,
            enable_evaluation=True,
            preprocessing_strategy="high_dimensional",
            evaluation_strategy="comprehensive"
        )

    @staticmethod
    def create_neural_focused_reducer(target_dimensions: int = 10) -> DimensionalityReducerFacade:
        """Create reducer focused on neural methods."""
        return DimensionalityReducerFacade(
            target_dimensions=target_dimensions,
            enable_neural_methods=True,
            enable_evaluation=True,
            auto_method_selection=False,  # Let user choose neural methods
            preprocessing_strategy="robust"
        )