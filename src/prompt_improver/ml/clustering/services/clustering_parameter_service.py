"""Clustering parameter optimization service.

Provides intelligent parameter optimization including:
- Grid search optimization
- Adaptive parameter selection based on data characteristics
- Parameter validation and constraint checking
- Performance-guided parameter tuning

Follows clean architecture patterns with protocol-based interfaces.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import itertools

import numpy as np

from . import ParameterOptimizationProtocol, OptimizationResult

logger = logging.getLogger(__name__)

# Optional imports for advanced optimization
try:
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False
    logger.warning("scikit-learn metrics not available for parameter optimization")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.warning("HDBSCAN not available for parameter optimization")

class ClusteringParameterService:
    """Service for clustering parameter optimization following clean architecture."""

    def __init__(self, algorithm: str = "hdbscan", scoring_metric: str = "silhouette",
                 max_iterations: int = 50, early_stopping: bool = True,
                 early_stopping_patience: int = 10, n_jobs: int = -1):
        """Initialize clustering parameter service.
        
        Args:
            algorithm: Clustering algorithm to optimize ("hdbscan", "kmeans", "dbscan")
            scoring_metric: Metric for optimization ("silhouette", "calinski_harabasz", "davies_bouldin")
            max_iterations: Maximum optimization iterations
            early_stopping: Whether to use early stopping
            early_stopping_patience: Patience for early stopping
            n_jobs: Number of parallel jobs
        """
        self.algorithm = algorithm
        self.scoring_metric = scoring_metric
        self.max_iterations = max_iterations
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.n_jobs = n_jobs
        
        # Optimization state
        self.optimization_history = []
        self.best_params = None
        self.best_score = -np.inf
        
        logger.info(f"ClusteringParameterService initialized: {algorithm}, "
                   f"metric={scoring_metric}, max_iter={max_iterations}")

    def optimize_parameters(self, X: np.ndarray, param_grid: Dict[str, List[Any]], 
                          labels: Optional[np.ndarray] = None) -> OptimizationResult:
        """Optimize clustering parameters using grid search."""
        start_time = time.time()
        
        try:
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(param_grid)
            
            if not param_combinations:
                return self._empty_optimization_result(start_time)
            
            logger.info(f"Starting parameter optimization with {len(param_combinations)} combinations")
            
            # Perform grid search optimization
            best_params, best_score, history = self._grid_search_optimize(
                X, param_combinations, labels
            )
            
            optimization_time = time.time() - start_time
            
            # Update service state
            self.best_params = best_params
            self.best_score = best_score
            self.optimization_history.extend(history)
            
            # Analyze convergence
            convergence_info = self._analyze_convergence(history)
            
            result = OptimizationResult(
                best_parameters=best_params,
                best_score=best_score,
                optimization_time=optimization_time,
                parameter_history=history,
                convergence_info=convergence_info
            )
            
            logger.info(f"Parameter optimization completed in {optimization_time:.2f}s: "
                       f"best_score={best_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            return self._empty_optimization_result(start_time)

    def get_adaptive_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        """Get adaptive parameters based on data characteristics."""
        n_samples, n_features = X.shape
        
        if self.algorithm == "hdbscan":
            return self._get_adaptive_hdbscan_params(n_samples, n_features)
        elif self.algorithm == "kmeans":
            return self._get_adaptive_kmeans_params(n_samples, n_features)
        elif self.algorithm == "dbscan":
            return self._get_adaptive_dbscan_params(X, n_samples, n_features)
        else:
            logger.warning(f"No adaptive parameters for algorithm: {self.algorithm}")
            return {}

    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameter combinations."""
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "suggestions": []
        }
        
        try:
            if self.algorithm == "hdbscan":
                validation_result = self._validate_hdbscan_params(parameters, validation_result)
            elif self.algorithm == "kmeans":
                validation_result = self._validate_kmeans_params(parameters, validation_result)
            elif self.algorithm == "dbscan":
                validation_result = self._validate_dbscan_params(parameters, validation_result)
            
        except Exception as e:
            validation_result["errors"].append(f"Validation failed: {str(e)}")
            validation_result["valid"] = False
        
        return validation_result

    def _generate_parameter_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid."""
        if not param_grid:
            return []
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations

    def _grid_search_optimize(self, X: np.ndarray, param_combinations: List[Dict[str, Any]],
                            labels: Optional[np.ndarray]) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Perform grid search optimization."""
        best_params = {}
        best_score = -np.inf
        history = []
        
        early_stopping_counter = 0
        
        for i, params in enumerate(param_combinations):
            try:
                # Validate parameters
                validation = self.validate_parameters(params)
                if not validation["valid"]:
                    logger.debug(f"Skipping invalid parameters: {params}")
                    continue
                
                # Evaluate parameters
                score = self._evaluate_parameters(X, params, labels)
                
                # Record in history
                history.append({
                    "iteration": i,
                    "parameters": params.copy(),
                    "score": score,
                    "is_best": score > best_score
                })
                
                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                # Early stopping check
                if (self.early_stopping and 
                    early_stopping_counter >= self.early_stopping_patience and
                    i > 20):  # Minimum iterations before early stopping
                    logger.info(f"Early stopping at iteration {i}")
                    break
                
                # Progress logging
                if i % 10 == 0:
                    logger.debug(f"Optimization progress: {i}/{len(param_combinations)}, "
                               f"best_score={best_score:.3f}")
                
                # Max iterations check
                if i >= self.max_iterations:
                    logger.info(f"Reached maximum iterations: {self.max_iterations}")
                    break
                    
            except Exception as e:
                logger.warning(f"Parameter evaluation failed for {params}: {e}")
                continue
        
        return best_params, best_score, history

    def _evaluate_parameters(self, X: np.ndarray, params: Dict[str, Any],
                           labels: Optional[np.ndarray]) -> float:
        """Evaluate clustering quality for given parameters."""
        if not SKLEARN_METRICS_AVAILABLE:
            logger.warning("Limited evaluation due to missing scikit-learn")
            return 0.0
        
        try:
            # Perform clustering with given parameters
            cluster_labels = self._cluster_with_params(X, params)
            
            # Calculate quality score
            return self._calculate_clustering_score(X, cluster_labels)
            
        except Exception as e:
            logger.debug(f"Parameter evaluation failed: {e}")
            return -1.0  # Penalty for failed clustering

    def _cluster_with_params(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Perform clustering with given parameters."""
        if self.algorithm == "hdbscan":
            if not HDBSCAN_AVAILABLE:
                raise ImportError("HDBSCAN not available")
            clusterer = hdbscan.HDBSCAN(**params)
            return clusterer.fit_predict(X)
        
        elif self.algorithm == "kmeans":
            from sklearn.cluster import KMeans
            clusterer = KMeans(**params)
            return clusterer.fit_predict(X)
        
        elif self.algorithm == "dbscan":
            from sklearn.cluster import DBSCAN
            clusterer = DBSCAN(**params)
            return clusterer.fit_predict(X)
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _calculate_clustering_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate clustering quality score."""
        try:
            # Check for valid clustering
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            if n_clusters < 2:
                return -1.0  # No meaningful clusters
            
            # Filter out noise points for scoring
            mask = labels != -1
            if np.sum(mask) < 10:  # Need minimum points for meaningful scoring
                return -1.0
            
            X_filtered = X[mask]
            labels_filtered = labels[mask]
            
            if len(set(labels_filtered)) < 2:
                return -1.0
            
            # Calculate score based on selected metric
            if self.scoring_metric == "silhouette":
                score = silhouette_score(X_filtered, labels_filtered)
                return score  # Already in [-1, 1] range
            
            elif self.scoring_metric == "calinski_harabasz":
                score = calinski_harabasz_score(X_filtered, labels_filtered)
                # Normalize to [0, 1] range
                return min(1.0, score / 100.0)
            
            elif self.scoring_metric == "davies_bouldin":
                score = davies_bouldin_score(X_filtered, labels_filtered)
                # Invert and normalize (lower DB score is better)
                return max(0.0, 1.0 - score / 5.0)
            
            else:
                logger.warning(f"Unknown scoring metric: {self.scoring_metric}")
                return 0.0
                
        except Exception as e:
            logger.debug(f"Score calculation failed: {e}")
            return -1.0

    def _get_adaptive_hdbscan_params(self, n_samples: int, n_features: int) -> Dict[str, Any]:
        """Get adaptive HDBSCAN parameters."""
        # Research-backed adaptive parameters for high-dimensional clustering
        if n_features > 20:
            min_cluster_size = max(10, min(int(n_samples * 0.03), n_samples // 8))
            min_samples = max(3, min_cluster_size // 5)
            cluster_selection_epsilon = 0.0
        else:
            min_cluster_size = max(5, min(int(n_samples * 0.02), 30))
            min_samples = max(2, min_cluster_size // 3)
            cluster_selection_epsilon = 0.05
        
        algorithm = "generic" if n_features > 20 and n_samples > 1000 else "best"
        
        return {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "alpha": 1.0,
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "algorithm": algorithm,
            "leaf_size": 40
        }

    def _get_adaptive_kmeans_params(self, n_samples: int, n_features: int) -> Dict[str, Any]:
        """Get adaptive K-means parameters."""
        # Heuristic for number of clusters
        n_clusters = min(10, max(2, int(np.sqrt(n_samples / 2))))
        
        return {
            "n_clusters": n_clusters,
            "random_state": 42,
            "n_init": 10,
            "max_iter": 300
        }

    def _get_adaptive_dbscan_params(self, X: np.ndarray, n_samples: int, n_features: int) -> Dict[str, Any]:
        """Get adaptive DBSCAN parameters."""
        # Estimate eps using k-distance method
        try:
            from sklearn.neighbors import NearestNeighbors
            k = min(10, max(4, n_samples // 100))
            nbrs = NearestNeighbors(n_neighbors=k)
            nbrs.fit(X)
            distances, _ = nbrs.kneighbors(X)
            eps = np.percentile(distances[:, -1], 90)
        except:
            eps = 0.5  # Default fallback
        
        min_samples = max(3, min(10, n_samples // 100))
        
        return {
            "eps": eps,
            "min_samples": min_samples
        }

    def _validate_hdbscan_params(self, params: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate HDBSCAN parameters."""
        if "min_cluster_size" in params:
            if params["min_cluster_size"] < 2:
                result["errors"].append("min_cluster_size must be >= 2")
                result["valid"] = False
        
        if "min_samples" in params and "min_cluster_size" in params:
            if params["min_samples"] >= params["min_cluster_size"]:
                result["warnings"].append("min_samples should be < min_cluster_size")
        
        if "alpha" in params:
            if params["alpha"] <= 0:
                result["errors"].append("alpha must be > 0")
                result["valid"] = False
        
        return result

    def _validate_kmeans_params(self, params: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate K-means parameters."""
        if "n_clusters" in params:
            if params["n_clusters"] < 1:
                result["errors"].append("n_clusters must be >= 1")
                result["valid"] = False
        
        if "max_iter" in params:
            if params["max_iter"] < 1:
                result["errors"].append("max_iter must be >= 1")
                result["valid"] = False
        
        return result

    def _validate_dbscan_params(self, params: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate DBSCAN parameters."""
        if "eps" in params:
            if params["eps"] <= 0:
                result["errors"].append("eps must be > 0")
                result["valid"] = False
        
        if "min_samples" in params:
            if params["min_samples"] < 1:
                result["errors"].append("min_samples must be >= 1")
                result["valid"] = False
        
        return result

    def _analyze_convergence(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze optimization convergence."""
        if not history:
            return {"converged": False, "reason": "No optimization history"}
        
        scores = [h["score"] for h in history]
        best_scores = []
        best_so_far = -np.inf
        
        for score in scores:
            if score > best_so_far:
                best_so_far = score
            best_scores.append(best_so_far)
        
        # Check for convergence (no improvement in last N iterations)
        convergence_window = min(10, len(scores) // 4)
        if len(scores) >= convergence_window:
            recent_improvement = best_scores[-1] - best_scores[-convergence_window]
            converged = recent_improvement < 1e-4
        else:
            converged = False
        
        return {
            "converged": converged,
            "total_iterations": len(history),
            "final_score": scores[-1] if scores else 0.0,
            "best_score": max(scores) if scores else 0.0,
            "improvement_over_baseline": max(scores) - scores[0] if len(scores) > 1 else 0.0,
            "convergence_iteration": self._find_convergence_point(best_scores)
        }

    def _find_convergence_point(self, best_scores: List[float]) -> Optional[int]:
        """Find the iteration where convergence occurred."""
        if len(best_scores) < 10:
            return None
        
        # Look for the last significant improvement
        threshold = 1e-4
        for i in range(len(best_scores) - 10, 0, -1):
            if best_scores[i] - best_scores[i-1] > threshold:
                return i
        
        return 0

    def _empty_optimization_result(self, start_time: float) -> OptimizationResult:
        """Create empty optimization result for failed optimizations."""
        return OptimizationResult(
            best_parameters={},
            best_score=-1.0,
            optimization_time=time.time() - start_time,
            parameter_history=[],
            convergence_info={"converged": False, "reason": "Optimization failed"}
        )

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        return {
            "algorithm": self.algorithm,
            "scoring_metric": self.scoring_metric,
            "best_score": self.best_score,
            "best_params": self.best_params.copy() if self.best_params else None,
            "total_evaluations": len(self.optimization_history),
            "optimization_configured": {
                "max_iterations": self.max_iterations,
                "early_stopping": self.early_stopping,
                "early_stopping_patience": self.early_stopping_patience
            }
        }

class ClusteringParameterServiceFactory:
    """Factory for creating parameter services with different configurations."""

    @staticmethod
    def create_fast_optimizer(algorithm: str = "hdbscan") -> ClusteringParameterService:
        """Create fast optimizer with limited iterations."""
        return ClusteringParameterService(
            algorithm=algorithm,
            max_iterations=20,
            early_stopping=True,
            early_stopping_patience=5
        )

    @staticmethod
    def create_thorough_optimizer(algorithm: str = "hdbscan") -> ClusteringParameterService:
        """Create thorough optimizer with extensive search."""
        return ClusteringParameterService(
            algorithm=algorithm,
            max_iterations=100,
            early_stopping=True,
            early_stopping_patience=15
        )

    @staticmethod
    def create_hdbscan_optimizer() -> ClusteringParameterService:
        """Create HDBSCAN-specific optimizer."""
        return ClusteringParameterService(
            algorithm="hdbscan",
            scoring_metric="silhouette",
            max_iterations=50
        )

    @staticmethod
    def create_kmeans_optimizer() -> ClusteringParameterService:
        """Create K-means-specific optimizer."""
        return ClusteringParameterService(
            algorithm="kmeans",
            scoring_metric="calinski_harabasz",
            max_iterations=30
        )