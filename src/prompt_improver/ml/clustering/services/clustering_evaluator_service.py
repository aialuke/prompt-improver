"""Clustering quality evaluation service.

Provides comprehensive clustering evaluation including:
- Quality metrics computation (silhouette, Calinski-Harabasz, Davies-Bouldin)
- Stability analysis and cluster compactness
- Success evaluation with adaptive thresholds
- Performance assessment and recommendations

Follows clean architecture patterns with protocol-based interfaces.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import ClusteringEvaluationProtocol, ClusteringMetrics

logger = logging.getLogger(__name__)

# Optional ML metrics imports with fallbacks
try:
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False
    logger.warning("scikit-learn metrics not available for clustering evaluation")

class ClusteringEvaluatorService:
    """Service for evaluating clustering quality following clean architecture."""

    def __init__(self, quality_weights: Optional[Dict[str, float]] = None,
                 enable_stability_analysis: bool = True,
                 adaptive_thresholds: bool = True):
        """Initialize clustering evaluator service.
        
        Args:
            quality_weights: Weights for different quality metrics
            enable_stability_analysis: Whether to compute stability metrics
            adaptive_thresholds: Whether to use adaptive quality thresholds
        """
        # Default quality metric weights
        self.quality_weights = quality_weights or {
            "silhouette": 0.4,
            "calinski_harabasz": 0.25,
            "davies_bouldin": 0.2,
            "noise_penalty": 0.1,
            "stability": 0.05
        }
        
        self.enable_stability_analysis = enable_stability_analysis
        self.adaptive_thresholds = adaptive_thresholds
        
        # Normalize weights
        total_weight = sum(self.quality_weights.values())
        if total_weight > 0:
            self.quality_weights = {k: v/total_weight for k, v in self.quality_weights.items()}
        
        logger.info(f"ClusteringEvaluatorService initialized: weights={self.quality_weights}, "
                   f"stability={enable_stability_analysis}, adaptive={adaptive_thresholds}")

    def assess_clustering_quality(self, X: np.ndarray, labels: np.ndarray, 
                                probabilities: Optional[np.ndarray] = None) -> ClusteringMetrics:
        """Assess comprehensive clustering quality."""
        start_time = time.time()
        
        try:
            # Basic cluster analysis
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise_points = int(np.sum(labels == -1))
            noise_ratio = n_noise_points / len(labels) if len(labels) > 0 else 1.0
            
            # Initialize metrics with defaults
            silhouette = 0.0
            calinski_harabasz = 0.0
            davies_bouldin = float("inf")
            stability_score = 0.0
            convergence_achieved = True
            
            # Compute metrics if we have meaningful clusters
            if n_clusters > 1 and SKLEARN_METRICS_AVAILABLE:
                silhouette, calinski_harabasz, davies_bouldin = self._compute_sklearn_metrics(
                    X, labels, n_clusters
                )
            
            # Compute stability score
            if self.enable_stability_analysis:
                stability_score = self._compute_stability_score_internal(X, labels, probabilities)
            
            # Compute overall quality score
            quality_score = self._compute_overall_quality_score(
                silhouette, calinski_harabasz, davies_bouldin, 
                noise_ratio, stability_score
            )
            
            processing_time = time.time() - start_time
            memory_usage = self._estimate_memory_usage(X.shape)
            
            return ClusteringMetrics(
                n_clusters=n_clusters,
                n_noise_points=n_noise_points,
                noise_ratio=noise_ratio,
                silhouette_score=silhouette,
                calinski_harabasz_score=calinski_harabasz,
                davies_bouldin_score=davies_bouldin,
                quality_score=max(0.0, min(1.0, quality_score)),
                processing_time_seconds=processing_time,
                memory_usage_mb=memory_usage,
                convergence_achieved=convergence_achieved,
                stability_score=stability_score
            )
            
        except Exception as e:
            logger.error(f"Clustering quality assessment failed: {e}")
            return self._default_clustering_metrics(X, labels, start_time)

    def compute_stability_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute cluster stability score."""
        return self._compute_stability_score_internal(X, labels)

    def evaluate_clustering_success(self, X: np.ndarray, metrics: ClusteringMetrics) -> Tuple[str, str]:
        """Evaluate if clustering was successful with adaptive thresholds."""
        n_samples, n_features = X.shape
        
        if self.adaptive_thresholds:
            return self._adaptive_success_evaluation(X.shape, metrics)
        else:
            return self._standard_success_evaluation(metrics)

    def _compute_sklearn_metrics(self, X: np.ndarray, labels: np.ndarray, 
                                n_clusters: int) -> Tuple[float, float, float]:
        """Compute scikit-learn clustering metrics."""
        silhouette = 0.0
        calinski_harabasz = 0.0
        davies_bouldin = float("inf")
        
        try:
            # Filter out noise points for quality assessment
            mask = labels != -1
            if np.sum(mask) > n_clusters and len(set(labels[mask])) > 1:
                X_filtered = X[mask]
                labels_filtered = labels[mask]
                
                # Silhouette score
                try:
                    silhouette = silhouette_score(X_filtered, labels_filtered, metric="euclidean")
                except Exception as e:
                    logger.debug(f"Silhouette score calculation failed: {e}")
                
                # Calinski-Harabasz score
                try:
                    calinski_harabasz = calinski_harabasz_score(X_filtered, labels_filtered)
                except Exception as e:
                    logger.debug(f"Calinski-Harabasz score calculation failed: {e}")
                
                # Davies-Bouldin score
                try:
                    davies_bouldin = davies_bouldin_score(X_filtered, labels_filtered)
                except Exception as e:
                    logger.debug(f"Davies-Bouldin score calculation failed: {e}")
                    
        except Exception as e:
            logger.warning(f"Sklearn metrics computation failed: {e}")
            
        return silhouette, calinski_harabasz, davies_bouldin

    def _compute_stability_score_internal(self, X: np.ndarray, labels: np.ndarray, 
                                        probabilities: Optional[np.ndarray] = None) -> float:
        """Compute cluster stability score based on within-cluster compactness."""
        try:
            # Use probabilities if available
            if probabilities is not None:
                try:
                    if probabilities.ndim >= 2 and probabilities.shape[1] > 1:
                        return float(np.mean(np.max(probabilities, axis=1)))
                    elif probabilities.ndim == 1:
                        return float(np.mean(probabilities))
                except (IndexError, ValueError) as e:
                    logger.debug(f"Error using probabilities for stability: {e}")
            
            # Fallback: compute based on within-cluster distances
            stability_scores = []
            unique_labels = set(labels)
            unique_labels.discard(-1)  # Remove noise label
            
            for cluster_id in unique_labels:
                cluster_mask = labels == cluster_id
                cluster_points = X[cluster_mask]
                
                if len(cluster_points) > 1:
                    # Compute average distance from centroid
                    center = np.mean(cluster_points, axis=0)
                    distances = np.linalg.norm(cluster_points - center, axis=1)
                    avg_distance = np.mean(distances)
                    
                    # Convert to stability score (lower distance = higher stability)
                    stability = max(0.0, 1.0 - avg_distance / 10.0)
                    stability_scores.append(stability)
            
            return float(np.mean(stability_scores)) if stability_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Stability score computation failed: {e}")
            return 0.0

    def _compute_overall_quality_score(self, silhouette: float, calinski_harabasz: float,
                                     davies_bouldin: float, noise_ratio: float,
                                     stability_score: float) -> float:
        """Compute weighted overall quality score."""
        try:
            # Normalize individual metrics to [0, 1] scale
            silhouette_normalized = (silhouette + 1) / 2  # From [-1, 1] to [0, 1]
            calinski_normalized = min(1.0, calinski_harabasz / 100.0)  # Normalize CH score
            davies_bouldin_normalized = max(0.0, 1.0 - davies_bouldin / 5.0)  # Invert DB score
            noise_penalty = max(0.0, 1.0 - 2.0 * noise_ratio)  # Penalty for high noise
            
            # Compute weighted score
            quality_score = (
                self.quality_weights.get("silhouette", 0.4) * silhouette_normalized +
                self.quality_weights.get("calinski_harabasz", 0.25) * calinski_normalized +
                self.quality_weights.get("davies_bouldin", 0.2) * davies_bouldin_normalized +
                self.quality_weights.get("noise_penalty", 0.1) * noise_penalty +
                self.quality_weights.get("stability", 0.05) * stability_score
            )
            
            return quality_score
            
        except Exception as e:
            logger.warning(f"Overall quality score computation failed: {e}")
            return 0.5

    def _adaptive_success_evaluation(self, data_shape: Tuple[int, int], 
                                   metrics: ClusteringMetrics) -> Tuple[str, str]:
        """Evaluate clustering success with adaptive thresholds based on data characteristics."""
        n_samples, n_features = data_shape
        
        # Adaptive noise thresholds based on dimensionality
        if n_features > 20:  # High-dimensional data
            if metrics.quality_score >= 0.6:
                max_noise_threshold = 0.90
            elif metrics.quality_score >= 0.4:
                max_noise_threshold = 0.85
            elif metrics.quality_score >= 0.3:
                max_noise_threshold = 0.90
            else:
                max_noise_threshold = 0.70
        else:
            max_noise_threshold = 0.6
        
        # Success criteria with adaptive thresholds
        min_clusters = 2
        max_clusters = min(20, max(10, int(n_samples * 0.1)))
        
        success_criteria = {
            "has_meaningful_clusters": metrics.n_clusters >= min_clusters,
            "reasonable_cluster_count": metrics.n_clusters <= max_clusters,
            "acceptable_noise": metrics.noise_ratio <= max_noise_threshold,
            "sufficient_quality": metrics.quality_score >= 0.3,
            "convergence": metrics.convergence_achieved
        }
        
        # Determine status
        critical_failures = []
        warnings = []
        
        if not success_criteria["has_meaningful_clusters"]:
            critical_failures.append(f"No meaningful clusters found (need ≥{min_clusters})")
        
        if not success_criteria["reasonable_cluster_count"]:
            critical_failures.append(f"Too many clusters ({metrics.n_clusters} > {max_clusters})")
        
        if not success_criteria["sufficient_quality"]:
            critical_failures.append(f"Poor cluster quality ({metrics.quality_score:.3f} < 0.3)")
        
        if not success_criteria["acceptable_noise"]:
            if n_features > 20 and metrics.quality_score >= 0.4:
                warnings.append(f"High noise ratio ({metrics.noise_ratio:.1%}) but acceptable for {n_features}D data")
            else:
                critical_failures.append(f"Excessive noise ({metrics.noise_ratio:.1%} > {max_noise_threshold:.1%})")
        
        if not success_criteria["convergence"]:
            warnings.append("Algorithm convergence uncertain")
        
        # Generate status and message
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
            message = f"High-quality clustering: {metrics.n_clusters} clusters, {metrics.noise_ratio:.1%} noise, quality {metrics.quality_score:.3f}"
        
        return status, message

    def _standard_success_evaluation(self, metrics: ClusteringMetrics) -> Tuple[str, str]:
        """Standard clustering success evaluation with fixed thresholds."""
        success_criteria = {
            "has_clusters": metrics.n_clusters >= 2,
            "reasonable_noise": metrics.noise_ratio <= 0.5,
            "good_silhouette": metrics.silhouette_score >= 0.3,
            "convergence": metrics.convergence_achieved
        }
        
        if all(success_criteria.values()):
            return "success", f"Clustering successful: {metrics.n_clusters} clusters, quality {metrics.quality_score:.3f}"
        else:
            issues = [k for k, v in success_criteria.items() if not v]
            return "low_quality", f"Clustering issues: {', '.join(issues)}"

    def _estimate_memory_usage(self, data_shape: Tuple[int, int]) -> float:
        """Estimate memory usage in MB based on data shape."""
        n_samples, n_features = data_shape
        # Rough estimate: 8 bytes per float64 value + overhead
        estimated_mb = (n_samples * n_features * 8) / (1024 * 1024) * 1.5  # 1.5x for overhead
        return max(1.0, estimated_mb)

    def _default_clustering_metrics(self, X: np.ndarray, labels: np.ndarray, 
                                  start_time: float) -> ClusteringMetrics:
        """Provide default metrics when full evaluation fails."""
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise_points = int(np.sum(labels == -1))
        noise_ratio = n_noise_points / len(labels) if len(labels) > 0 else 1.0
        
        return ClusteringMetrics(
            n_clusters=n_clusters,
            n_noise_points=n_noise_points,
            noise_ratio=noise_ratio,
            silhouette_score=0.0,
            calinski_harabasz_score=0.0,
            davies_bouldin_score=float("inf"),
            quality_score=0.1,  # Low default quality
            processing_time_seconds=time.time() - start_time,
            memory_usage_mb=self._estimate_memory_usage(X.shape),
            convergence_achieved=False,
            stability_score=0.0
        )

    def compare_clustering_results(self, X: np.ndarray, 
                                 results: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]) -> Dict[str, Any]:
        """Compare multiple clustering results."""
        comparison = {
            "results": {},
            "best_method": None,
            "best_score": -1.0,
            "ranking": []
        }
        
        for method_name, (labels, probabilities) in results.items():
            try:
                metrics = self.assess_clustering_quality(X, labels, probabilities)
                comparison["results"][method_name] = {
                    "metrics": metrics,
                    "quality_score": metrics.quality_score,
                    "n_clusters": metrics.n_clusters,
                    "noise_ratio": metrics.noise_ratio
                }
                
                # Track best method
                if metrics.quality_score > comparison["best_score"]:
                    comparison["best_score"] = metrics.quality_score
                    comparison["best_method"] = method_name
                    
            except Exception as e:
                logger.error(f"Failed to evaluate method '{method_name}': {e}")
        
        # Create ranking
        comparison["ranking"] = sorted(
            comparison["results"].items(),
            key=lambda x: x[1]["quality_score"],
            reverse=True
        )
        
        return comparison

    def generate_evaluation_report(self, X: np.ndarray, labels: np.ndarray,
                                 probabilities: Optional[np.ndarray] = None,
                                 method_name: str = "Unknown") -> Dict[str, Any]:
        """Generate comprehensive clustering evaluation report."""
        metrics = self.assess_clustering_quality(X, labels, probabilities)
        status, status_message = self.evaluate_clustering_success(X, metrics)
        
        report = {
            "method": method_name,
            "data_summary": {
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "data_dimensionality": "high" if X.shape[1] > 20 else "standard"
            },
            "clustering_results": {
                "n_clusters": metrics.n_clusters,
                "n_noise_points": metrics.n_noise_points,
                "noise_ratio": metrics.noise_ratio,
                "largest_cluster_size": self._get_largest_cluster_size(labels)
            },
            "quality_metrics": {
                "overall_quality": metrics.quality_score,
                "silhouette_score": metrics.silhouette_score,
                "calinski_harabasz_score": metrics.calinski_harabasz_score,
                "davies_bouldin_score": metrics.davies_bouldin_score,
                "stability_score": metrics.stability_score
            },
            "evaluation": {
                "status": status,
                "message": status_message,
                "quality_level": self._get_quality_level(metrics.quality_score)
            },
            "performance": {
                "processing_time": metrics.processing_time_seconds,
                "memory_usage_mb": metrics.memory_usage_mb,
                "convergence_achieved": metrics.convergence_achieved
            },
            "recommendations": self._generate_recommendations(metrics, X.shape, status)
        }
        
        return report

    def _get_largest_cluster_size(self, labels: np.ndarray) -> int:
        """Get size of the largest cluster."""
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        return int(np.max(counts)) if len(counts) > 0 else 0

    def _get_quality_level(self, quality_score: float) -> str:
        """Get quality level description."""
        if quality_score >= 0.8:
            return "Excellent"
        elif quality_score >= 0.6:
            return "Good"
        elif quality_score >= 0.4:
            return "Moderate"
        else:
            return "Poor"

    def _generate_recommendations(self, metrics: ClusteringMetrics, 
                                data_shape: Tuple[int, int], status: str) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        n_samples, n_features = data_shape
        
        if metrics.quality_score < 0.4:
            recommendations.append("Consider adjusting clustering parameters or preprocessing")
        
        if metrics.noise_ratio > 0.6:
            recommendations.append("High noise ratio - consider increasing min_cluster_size or improving preprocessing")
        
        if metrics.n_clusters < 2:
            recommendations.append("No meaningful clusters found - check data quality or try different parameters")
        
        if metrics.n_clusters > n_samples * 0.1:
            recommendations.append("Too many clusters - consider increasing min_cluster_size")
        
        if n_features > 20 and metrics.quality_score < 0.5:
            recommendations.append("High-dimensional data - consider dimensionality reduction before clustering")
        
        if status == "success" and not recommendations:
            recommendations.append("Clustering results are satisfactory")
        
        return recommendations

class ClusteringEvaluatorServiceFactory:
    """Factory for creating clustering evaluator services with different configurations."""

    @staticmethod
    def create_comprehensive_evaluator() -> ClusteringEvaluatorService:
        """Create evaluator with all features enabled."""
        return ClusteringEvaluatorService(
            enable_stability_analysis=True,
            adaptive_thresholds=True
        )

    @staticmethod
    def create_fast_evaluator() -> ClusteringEvaluatorService:
        """Create evaluator optimized for speed."""
        return ClusteringEvaluatorService(
            enable_stability_analysis=False,
            adaptive_thresholds=False
        )

    @staticmethod
    def create_stability_focused_evaluator() -> ClusteringEvaluatorService:
        """Create evaluator focused on stability metrics."""
        quality_weights = {
            "silhouette": 0.3,
            "calinski_harabasz": 0.2,
            "davies_bouldin": 0.15,
            "noise_penalty": 0.1,
            "stability": 0.25  # Higher weight for stability
        }
        return ClusteringEvaluatorService(
            quality_weights=quality_weights,
            enable_stability_analysis=True
        )

    @staticmethod
    def create_high_dimensional_evaluator() -> ClusteringEvaluatorService:
        """Create evaluator optimized for high-dimensional data."""
        return ClusteringEvaluatorService(
            enable_stability_analysis=True,
            adaptive_thresholds=True  # Important for high-dimensional data
        )