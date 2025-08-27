"""Clustering Optimizer Facade.

Coordinates clustering optimization services following clean architecture patterns.
Replaces the monolithic ClusteringOptimizer with a facade that orchestrates:
- ClusteringPreprocessorService: Feature preprocessing and validation
- ClusteringAlgorithmService: HDBSCAN and clustering execution
- ClusteringParameterService: Parameter optimization and tuning
- ClusteringEvaluatorService: Quality evaluation and metrics

Provides improved modularity and testability through focused service components.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# import numpy as np  # Converted to lazy loading

from . import ClusteringResult, ClusteringMetrics, OptimizationResult
from .clustering_preprocessor_service import ClusteringPreprocessorService, ClusteringPreprocessorServiceFactory
from .clustering_algorithm_service import ClusteringAlgorithmService, ClusteringAlgorithmServiceFactory
from .clustering_parameter_service import ClusteringParameterService, ClusteringParameterServiceFactory
from .clustering_evaluator_service import ClusteringEvaluatorService, ClusteringEvaluatorServiceFactory
from prompt_improver.core.utils.lazy_ml_loader import get_numpy

logger = logging.getLogger(__name__)

class ClusteringOptimizerFacade:
    """Facade coordinating clustering optimization services following clean architecture."""

    def __init__(self, algorithm: str = "hdbscan", 
                 preprocessing_strategy: str = "high_dimensional",
                 parameter_optimization: str = "thorough",
                 evaluation_strategy: str = "comprehensive",
                 memory_efficient: bool = True):
        """Initialize clustering optimizer facade.
        
        Args:
            algorithm: Clustering algorithm to use ("hdbscan", "kmeans", "dbscan")
            preprocessing_strategy: Preprocessing strategy ("minimal", "high_dimensional", "memory_efficient", "quality_focused")
            parameter_optimization: Parameter optimization strategy ("fast", "thorough", "hdbscan", "kmeans")
            evaluation_strategy: Evaluation strategy ("fast", "comprehensive", "stability_focused", "high_dimensional")
            memory_efficient: Whether to use memory-efficient processing
        """
        self.algorithm = algorithm
        self.memory_efficient = memory_efficient
        
        # Initialize service components
        self.preprocessor = self._create_preprocessor(preprocessing_strategy)
        self.algorithm_service = self._create_algorithm_service(algorithm, memory_efficient)
        self.parameter_service = self._create_parameter_service(algorithm, parameter_optimization)
        self.evaluator = self._create_evaluator(evaluation_strategy)
        
        # Performance tracking
        self.optimization_history = []
        self.performance_metrics = []
        
        logger.info(f"ClusteringOptimizerFacade initialized: algorithm={algorithm}, "
                   f"preprocessing={preprocessing_strategy}, optimization={parameter_optimization}")

    def _create_preprocessor(self, strategy: str) -> ClusteringPreprocessorService:
        """Create preprocessor service based on strategy."""
        if strategy == "minimal":
            return ClusteringPreprocessorServiceFactory.create_minimal_preprocessor()
        elif strategy == "high_dimensional":
            return ClusteringPreprocessorServiceFactory.create_high_dimensional_preprocessor()
        elif strategy == "memory_efficient":
            return ClusteringPreprocessorServiceFactory.create_memory_efficient_preprocessor()
        elif strategy == "quality_focused":
            return ClusteringPreprocessorServiceFactory.create_quality_focused_preprocessor()
        else:
            logger.warning(f"Unknown preprocessing strategy '{strategy}', using high_dimensional")
            return ClusteringPreprocessorServiceFactory.create_high_dimensional_preprocessor()

    def _create_algorithm_service(self, algorithm: str, memory_efficient: bool) -> ClusteringAlgorithmService:
        """Create algorithm service based on configuration."""
        if memory_efficient:
            return ClusteringAlgorithmServiceFactory.create_memory_efficient_service(algorithm=algorithm)
        else:
            return ClusteringAlgorithmServiceFactory.create_best_service_for_data(1000, 50, algorithm=algorithm)

    def _create_parameter_service(self, algorithm: str, strategy: str) -> ClusteringParameterService:
        """Create parameter service based on strategy."""
        if strategy == "fast":
            return ClusteringParameterServiceFactory.create_fast_optimizer(algorithm=algorithm)
        elif strategy == "thorough":
            return ClusteringParameterServiceFactory.create_thorough_optimizer(algorithm=algorithm)
        elif strategy == "hdbscan":
            return ClusteringParameterServiceFactory.create_hdbscan_optimizer()
        elif strategy == "kmeans":
            return ClusteringParameterServiceFactory.create_kmeans_optimizer()
        else:
            logger.warning(f"Unknown parameter optimization strategy '{strategy}', using thorough")
            return ClusteringParameterServiceFactory.create_thorough_optimizer(algorithm=algorithm)

    def _create_evaluator(self, strategy: str) -> ClusteringEvaluatorService:
        """Create evaluator service based on strategy."""
        if strategy == "fast":
            return ClusteringEvaluatorServiceFactory.create_fast_evaluator()
        elif strategy == "comprehensive":
            return ClusteringEvaluatorServiceFactory.create_comprehensive_evaluator()
        elif strategy == "stability_focused":
            return ClusteringEvaluatorServiceFactory.create_stability_focused_evaluator()
        elif strategy == "high_dimensional":
            return ClusteringEvaluatorServiceFactory.create_high_dimensional_evaluator()
        else:
            logger.warning(f"Unknown evaluation strategy '{strategy}', using comprehensive")
            return ClusteringEvaluatorServiceFactory.create_comprehensive_evaluator()

    async def optimize_clustering(self, features: get_numpy().ndarray, labels: Optional[get_numpy().ndarray] = None,
                                sample_weights: Optional[get_numpy().ndarray] = None) -> Dict[str, Any]:
        """Optimize clustering for high-dimensional features.
        
        Args:
            features: High-dimensional feature matrix
            labels: Optional ground truth labels for supervised optimization
            sample_weights: Optional sample weights
            
        Returns:
            Clustering optimization result with metrics and recommendations
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting clustering optimization: {features.shape[0]} samples x {features.shape[1]} features")
            
            # Input validation
            if features.ndim != 2:
                return {"status": "failed", "error": f"Features must be 2D array, got {features.ndim}D"}
            
            if features.shape[0] < 3:
                return {"status": "failed", "error": f"Need at least 3 samples, got {features.shape[0]}"}
            
            # Step 1: Preprocessing
            preprocessing_result = self.preprocessor.preprocess_features(features, labels)
            if preprocessing_result.status != "success":
                return {
                    "status": "failed",
                    "error": f"Preprocessing failed: {preprocessing_result.info.get('error', 'Unknown error')}",
                    "optimization_time": time.time() - start_time
                }
            
            processed_features = preprocessing_result.features
            preprocessing_info = preprocessing_result.info
            
            # Step 2: Parameter optimization
            optimal_params = await self._optimize_parameters(processed_features, labels)
            
            # Step 3: Perform optimized clustering
            clustering_result = await self._perform_optimized_clustering(
                processed_features, optimal_params, sample_weights
            )
            
            # Step 4: Comprehensive quality assessment
            quality_metrics = self.evaluator.assess_clustering_quality(
                processed_features,
                clustering_result["labels"],
                clustering_result.get("probabilities")
            )
            
            # Step 5: Success evaluation
            status, status_message = self.evaluator.evaluate_clustering_success(
                processed_features, quality_metrics
            )
            
            # Step 6: Performance analysis and recommendations
            performance_analysis = self._analyze_performance(
                features.shape, preprocessing_info, clustering_result, 
                quality_metrics, time.time() - start_time
            )
            
            # Update tracking
            self._update_performance_tracking(quality_metrics, time.time() - start_time)
            
            # Compile comprehensive result
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
                "optimization_time": time.time() - start_time
            }
            
            logger.info(f"Clustering optimization completed in {time.time() - start_time:.2f}s "
                       f"with status '{status}' and quality {quality_metrics.quality_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Clustering optimization failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "optimization_time": time.time() - start_time
            }

    async def run_orchestrated_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrator-compatible interface for clustering optimization."""
        start_time = datetime.now()
        
        try:
            # Extract orchestrator configuration
            features = config.get("features", [])
            labels = config.get("labels", None)
            sample_weights = config.get("sample_weights", None)
            optimization_target = config.get("optimization_target", "silhouette")
            max_clusters = config.get("max_clusters", 20)
            output_path = config.get("output_path", "./outputs/clustering_optimization")
            clustering_method = config.get("clustering_method", "auto")
            
            # Validate and convert input data
            if not features:
                raise ValueError("features are required for clustering optimization")
            
            if isinstance(features, list):
                features = get_numpy().array(features, dtype=float)
            
            if features.ndim != 2 or features.shape[0] < 3:
                raise ValueError("Invalid features array")
            
            # Convert optional inputs
            if labels is not None and isinstance(labels, list):
                labels = get_numpy().array(labels)
            if sample_weights is not None and isinstance(sample_weights, list):
                sample_weights = get_numpy().array(sample_weights, dtype=float)
            
            # Perform clustering optimization
            optimization_result = await self.optimize_clustering(features, labels, sample_weights)
            
            # Format orchestrator-compatible result
            result = {
                "clustering_summary": {
                    "status": optimization_result.get("status", "unknown"),
                    "samples_processed": features.shape[0],
                    "original_dimensions": features.shape[1],
                    "optimal_dimensions": optimization_result.get("preprocessing_info", {}).get("final_shape", [0, features.shape[1]])[1],
                    "clusters_found": optimization_result.get("quality_metrics", {}).get("n_clusters", 0),
                    "noise_points": optimization_result.get("quality_metrics", {}).get("n_noise_points", 0)
                },
                "optimization_results": {
                    "best_parameters": optimization_result.get("optimal_parameters", {}),
                    "clustering_quality": optimization_result.get("quality_metrics", {}).get("quality_score", 0.0),
                    "optimization_time": optimization_result.get("optimization_time", 0.0),
                    "quality_metrics": {
                        "silhouette_score": optimization_result.get("quality_metrics", {}).get("silhouette_score", 0.0),
                        "calinski_harabasz_score": optimization_result.get("quality_metrics", {}).get("calinski_harabasz_score", 0.0),
                        "davies_bouldin_score": optimization_result.get("quality_metrics", {}).get("davies_bouldin_score", 0.0)
                    }
                },
                "cluster_analysis": {
                    "cluster_sizes": self._get_cluster_sizes(optimization_result.get("labels", [])),
                    "cluster_centers": optimization_result.get("cluster_centers", []),
                    "cluster_quality_scores": [],
                    "outlier_detection": {"noise_ratio": optimization_result.get("quality_metrics", {}).get("noise_ratio", 0.0)}
                },
                "recommendations": optimization_result.get("performance_analysis", {}).get("recommendations", []),
                "dimensionality_reduction": {
                    "method_used": "UMAP",
                    "reduction_ratio": features.shape[1] / max(1, optimization_result.get("preprocessing_info", {}).get("final_shape", [0, features.shape[1]])[1]),
                    "variance_explained": 0.0
                }
            }
            
            # Add error information if optimization failed
            if optimization_result.get("status") == "failed":
                result["error_details"] = {
                    "error_message": optimization_result.get("error", "Unknown error"),
                    "error_type": "clustering_optimization_error"
                }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "orchestrator_compatible": True,
                "component_result": result,
                "local_metadata": {
                    "output_path": output_path,
                    "execution_time": execution_time,
                    "optimization_target": optimization_target,
                    "clustering_method": clustering_method,
                    "samples_processed": features.shape[0],
                    "features_count": features.shape[1],
                    "max_clusters": max_clusters,
                    "supervised_optimization": labels is not None,
                    "weighted_optimization": sample_weights is not None,
                    "component_version": "1.0.0"
                }
            }
            
        except Exception as e:
            logger.error(f"Orchestrated clustering optimization failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                "orchestrator_compatible": True,
                "component_result": {"error": str(e), "clustering_summary": {}},
                "local_metadata": {
                    "execution_time": execution_time,
                    "error": True,
                    "component_version": "1.0.0"
                }
            }

    async def cluster_contexts(self, features: get_numpy().ndarray) -> ClusteringResult:
        """Simplified async clustering interface compatible with existing code."""
        try:
            if not self._validate_features_simple(features):
                return self._get_default_result(features.shape[0])
            
            # Use existing optimization method but return structured result
            optimization_result = await self.optimize_clustering(features)
            
            if optimization_result.get("status") == "failed":
                return self._get_default_result(features.shape[0])
            
            return ClusteringResult(
                cluster_labels=optimization_result.get("labels", get_numpy().zeros(features.shape[0], dtype=int)),
                cluster_centers=optimization_result.get("cluster_centers"),
                n_clusters=optimization_result.get("quality_metrics", {}).get("n_clusters", 1),
                silhouette_score=optimization_result.get("quality_metrics", {}).get("silhouette_score", 0.0),
                algorithm_used=f"{self.algorithm.upper()}",
                processing_time=optimization_result.get("optimization_time", 0.0),
                quality_metrics=optimization_result.get("quality_metrics", {}),
                metadata=optimization_result.get("preprocessing_info", {})
            )
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return self._get_default_result(features.shape[0])

    async def _optimize_parameters(self, features: get_numpy().ndarray, labels: Optional[get_numpy().ndarray] = None) -> Dict[str, Any]:
        """Optimize clustering parameters for the specific dataset."""
        try:
            # Get adaptive parameters as baseline
            adaptive_params = self.parameter_service.get_adaptive_parameters(features)
            
            # If we have sufficient data, perform grid search optimization
            if features.shape[0] > 50:
                param_grid = self._get_parameter_grid()
                optimization_result = self.parameter_service.optimize_parameters(
                    features, param_grid, labels
                )
                
                if optimization_result.best_score > 0:
                    logger.info(f"Parameter optimization completed: score={optimization_result.best_score:.3f}")
                    return optimization_result.best_parameters
            
            logger.info("Using adaptive parameters (insufficient data for optimization)")
            return adaptive_params
            
        except Exception as e:
            logger.warning(f"Parameter optimization failed: {e}, using adaptive parameters")
            return self.parameter_service.get_adaptive_parameters(features)

    def _get_parameter_grid(self) -> Dict[str, List[Any]]:
        """Get parameter grid for optimization based on algorithm."""
        if self.algorithm == "hdbscan":
            return {
                "min_cluster_size": [3, 5, 8, 10, 15],
                "min_samples": [1, 2, 3, 5],
                "alpha": [0.5, 1.0, 1.5],
                "cluster_selection_epsilon": [0.0, 0.1, 0.2, 0.3]
            }
        elif self.algorithm == "kmeans":
            return {
                "n_clusters": [2, 3, 4, 5, 6, 8, 10, 12, 15],
                "n_init": [10, 20],
                "max_iter": [300, 500]
            }
        elif self.algorithm == "dbscan":
            return {
                "eps": [0.1, 0.3, 0.5, 0.7, 1.0],
                "min_samples": [2, 3, 5, 8, 10]
            }
        else:
            return {}

    async def _perform_optimized_clustering(self, features: get_numpy().ndarray, 
                                          optimal_params: Dict[str, Any],
                                          sample_weights: Optional[get_numpy().ndarray] = None) -> Dict[str, Any]:
        """Perform clustering with optimized parameters."""
        try:
            # Update algorithm service with optimal parameters
            self.algorithm_service = ClusteringAlgorithmService(
                algorithm=self.algorithm,
                memory_efficient=self.memory_efficient,
                **optimal_params
            )
            
            # Perform clustering
            labels = self.algorithm_service.fit_predict(features, sample_weights)
            
            # Get additional information
            cluster_centers = self.algorithm_service.get_cluster_centers(features, labels)
            probabilities = self.algorithm_service.get_cluster_probabilities()
            
            return {
                "labels": labels,
                "cluster_centers": cluster_centers,
                "probabilities": probabilities,
                "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
                "n_noise_points": int(get_numpy().sum(labels == -1))
            }
            
        except Exception as e:
            logger.error(f"Optimized clustering failed: {e}")
            # Fallback to simple clustering
            try:
                fallback_service = ClusteringAlgorithmServiceFactory.create_best_service_for_data(
                    features.shape[0], features.shape[1]
                )
                labels = fallback_service.fit_predict(features, sample_weights)
                return {
                    "labels": labels,
                    "cluster_centers": fallback_service.get_cluster_centers(features, labels),
                    "probabilities": None,
                    "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
                    "n_noise_points": int(get_numpy().sum(labels == -1))
                }
            except:
                # Ultimate fallback: single cluster
                return {
                    "labels": get_numpy().zeros(features.shape[0], dtype=int),
                    "cluster_centers": None,
                    "probabilities": None,
                    "n_clusters": 1,
                    "n_noise_points": 0
                }

    def _analyze_performance(self, original_shape: Tuple[int, int],
                           preprocessing_info: Dict[str, Any],
                           clustering_result: Dict[str, Any],
                           quality_metrics: ClusteringMetrics,
                           total_time: float) -> Dict[str, Any]:
        """Analyze clustering performance and generate recommendations."""
        return {
            "performance_summary": {
                "total_time": total_time,
                "preprocessing_time": preprocessing_info.get("preprocessing_time", 0.0),
                "clustering_time": total_time - preprocessing_info.get("preprocessing_time", 0.0),
                "dimensionality_reduction": f"{original_shape[1]} → {preprocessing_info.get('final_shape', [0, original_shape[1]])[1]}",
                "memory_efficiency": quality_metrics.memory_usage_mb < 1000
            },
            "quality_assessment": {
                "overall_quality": quality_metrics.quality_score,
                "meets_thresholds": {
                    "sufficient_clusters": quality_metrics.n_clusters >= 2,
                    "acceptable_noise": quality_metrics.noise_ratio <= 0.7,
                    "good_silhouette": quality_metrics.silhouette_score >= 0.3
                }
            },
            "recommendations": self._generate_recommendations(quality_metrics, original_shape)
        }

    def _generate_recommendations(self, metrics: ClusteringMetrics, 
                                original_shape: Tuple[int, int]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if metrics.quality_score < 0.5:
            recommendations.append("Consider adjusting clustering parameters or feature engineering")
        
        if metrics.noise_ratio > 0.6:
            recommendations.append(f"High noise ratio ({metrics.noise_ratio:.2f}). Consider increasing min_cluster_size")
        
        if metrics.n_clusters < 2:
            recommendations.append("No meaningful clusters found. Check data quality or parameters")
        
        if original_shape[1] > 50 and metrics.quality_score < 0.5:
            recommendations.append("High-dimensional data with poor quality. Consider better dimensionality reduction")
        
        if not recommendations:
            recommendations.append("Clustering results are satisfactory")
        
        return recommendations

    def _get_cluster_sizes(self, labels: get_numpy().ndarray) -> List[int]:
        """Get sizes of each cluster."""
        if len(labels) == 0:
            return []
        
        unique_labels, counts = get_numpy().unique(labels[labels != -1], return_counts=True)
        return counts.tolist()

    def _validate_features_simple(self, features: get_numpy().ndarray) -> bool:
        """Simplified feature validation."""
        try:
            return (features is not None and 
                   features.size > 0 and 
                   len(features.shape) == 2 and 
                   features.shape[0] >= 3 and
                   get_numpy().isfinite(features).all())
        except:
            return False

    def _get_default_result(self, n_samples: int) -> ClusteringResult:
        """Get default clustering result when clustering fails."""
        return ClusteringResult(
            cluster_labels=get_numpy().zeros(n_samples, dtype=int),
            cluster_centers=None,
            n_clusters=1,
            silhouette_score=0.0,
            algorithm_used='default',
            processing_time=0.0,
            quality_metrics={'default': True},
            metadata={'is_default': True, 'n_samples': n_samples}
        )

    def _update_performance_tracking(self, metrics: ClusteringMetrics, processing_time: float):
        """Update performance tracking metrics."""
        self.performance_metrics.append(metrics)
        
        # Keep only recent metrics to prevent memory growth
        if len(self.performance_metrics) > 100:
            self.performance_metrics = self.performance_metrics[-50:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.performance_metrics:
            return {"status": "no_data"}
        
        recent_metrics = self.performance_metrics[-10:]
        
        return {
            "status": "available",
            "total_runs": len(self.performance_metrics),
            "recent_performance": {
                "avg_quality_score": float(get_numpy().mean([m.quality_score for m in recent_metrics])),
                "avg_processing_time": float(get_numpy().mean([m.processing_time_seconds for m in recent_metrics])),
                "avg_memory_usage": float(get_numpy().mean([m.memory_usage_mb for m in recent_metrics])),
                "convergence_rate": float(get_numpy().mean([m.convergence_achieved for m in recent_metrics]))
            },
            "cache_performance": {
                "hit_ratio": 0.0,
                "cache_size": 0
            }
        }

    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about available clustering algorithms."""
        return self.algorithm_service.get_algorithm_info()

class ClusteringOptimizerFacadeFactory:
    """Factory for creating clustering optimizer facades with different configurations."""

    @staticmethod
    def create_fast_optimizer() -> ClusteringOptimizerFacade:
        """Create optimizer optimized for speed."""
        return ClusteringOptimizerFacade(
            preprocessing_strategy="minimal",
            parameter_optimization="fast",
            evaluation_strategy="fast",
            memory_efficient=True
        )

    @staticmethod
    def create_high_quality_optimizer() -> ClusteringOptimizerFacade:
        """Create optimizer optimized for quality."""
        return ClusteringOptimizerFacade(
            preprocessing_strategy="quality_focused",
            parameter_optimization="thorough",
            evaluation_strategy="comprehensive",
            memory_efficient=False
        )

    @staticmethod
    def create_high_dimensional_optimizer() -> ClusteringOptimizerFacade:
        """Create optimizer optimized for high-dimensional data."""
        return ClusteringOptimizerFacade(
            preprocessing_strategy="high_dimensional",
            parameter_optimization="thorough",
            evaluation_strategy="high_dimensional",
            memory_efficient=True
        )

    @staticmethod
    def create_memory_efficient_optimizer() -> ClusteringOptimizerFacade:
        """Create optimizer optimized for memory efficiency."""
        return ClusteringOptimizerFacade(
            preprocessing_strategy="memory_efficient",
            parameter_optimization="fast",
            evaluation_strategy="fast",
            memory_efficient=True
        )