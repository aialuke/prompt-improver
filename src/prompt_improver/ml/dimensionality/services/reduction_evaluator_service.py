"""Dimensionality reduction quality evaluation service.

Provides comprehensive evaluation of dimensionality reduction quality including:
- Variance preservation analysis
- Clustering structure preservation  
- Classification performance preservation
- Neighborhood structure preservation
- Overall quality scoring

Follows clean architecture patterns with protocol-based interfaces.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import EvaluationProtocol, EvaluationMetrics

logger = logging.getLogger(__name__)

# Optional ML imports with fallbacks
try:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import adjusted_rand_score
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available for advanced evaluation metrics")

class ReductionEvaluatorService:
    """Service for evaluating dimensionality reduction quality following clean architecture."""

    def __init__(self, variance_weight: float = 0.3, clustering_weight: float = 0.4,
                 classification_weight: float = 0.3, enable_advanced_metrics: bool = True):
        """Initialize reduction evaluator service.
        
        Args:
            variance_weight: Weight for variance preservation in overall score
            clustering_weight: Weight for clustering preservation in overall score  
            classification_weight: Weight for classification preservation in overall score
            enable_advanced_metrics: Whether to compute advanced metrics (slower)
        """
        self.variance_weight = variance_weight
        self.clustering_weight = clustering_weight
        self.classification_weight = classification_weight
        self.enable_advanced_metrics = enable_advanced_metrics
        
        # Normalize weights
        total_weight = variance_weight + clustering_weight + classification_weight
        if total_weight > 0:
            self.variance_weight /= total_weight
            self.clustering_weight /= total_weight
            self.classification_weight /= total_weight
        
        logger.info(f"ReductionEvaluatorService initialized with weights: variance={self.variance_weight:.2f}, "
                   f"clustering={self.clustering_weight:.2f}, classification={self.classification_weight:.2f}")

    def evaluate_quality(self, original: np.ndarray, reduced: np.ndarray, 
                        labels: Optional[np.ndarray] = None) -> EvaluationMetrics:
        """Evaluate comprehensive quality of dimensionality reduction."""
        start_time = time.time()
        
        if not SKLEARN_AVAILABLE:
            logger.warning("Limited evaluation due to missing scikit-learn")
            return self._basic_evaluation(original, reduced, start_time)
        
        try:
            # Core metrics
            variance_preservation = self.compute_variance_preservation(original, reduced)
            clustering_preservation = self.assess_clustering_preservation(original, reduced)
            
            # Optional classification preservation
            classification_preservation = None
            if labels is not None:
                classification_preservation = self._assess_classification_preservation(
                    original, reduced, labels
                )
            
            # Advanced metrics if enabled
            neighborhood_preservation = 0.5  # Default
            if self.enable_advanced_metrics:
                neighborhood_preservation = self._assess_neighborhood_preservation(original, reduced)
            
            # Compute overall quality
            overall_quality = self._compute_overall_quality(
                variance_preservation, clustering_preservation, 
                classification_preservation, neighborhood_preservation
            )
            
            processing_time = time.time() - start_time
            
            return EvaluationMetrics(
                variance_preservation=variance_preservation,
                clustering_preservation=clustering_preservation,
                classification_preservation=classification_preservation,
                neighborhood_preservation=neighborhood_preservation,
                overall_quality=overall_quality,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return self._basic_evaluation(original, reduced, start_time)

    def compute_variance_preservation(self, original: np.ndarray, reduced: np.ndarray) -> float:
        """Compute how much variance is preserved in the reduction."""
        try:
            if original.size == 0 or reduced.size == 0:
                return 0.0
            
            if original.shape[0] != reduced.shape[0]:
                logger.warning("Shape mismatch in variance preservation calculation")
                return 0.0
            
            # Calculate total variance preservation
            original_var = np.sum(np.var(original, axis=0))
            reduced_var = np.sum(np.var(reduced, axis=0))
            
            if original_var == 0:
                return 1.0 if reduced_var == 0 else 0.0
            
            preservation = min(1.0, reduced_var / original_var)
            return max(0.0, preservation)
            
        except Exception as e:
            logger.warning(f"Variance preservation calculation failed: {e}")
            return 0.5

    def assess_clustering_preservation(self, original: np.ndarray, reduced: np.ndarray) -> float:
        """Assess how well clustering structure is preserved."""
        if not SKLEARN_AVAILABLE or original.shape[0] < 10:
            return 0.5
        
        try:
            # Use K-means clustering to assess structure preservation
            n_clusters = min(5, max(2, original.shape[0] // 10))
            
            # Cluster original data
            kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels_orig = kmeans_orig.fit_predict(original)
            
            # Cluster reduced data
            kmeans_reduced = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels_reduced = kmeans_reduced.fit_predict(reduced)
            
            # Measure agreement between clusterings
            agreement = adjusted_rand_score(labels_orig, labels_reduced)
            
            # Convert from [-1, 1] to [0, 1] range
            normalized_agreement = (agreement + 1) / 2
            
            return max(0.0, min(1.0, normalized_agreement))
            
        except Exception as e:
            logger.warning(f"Clustering preservation assessment failed: {e}")
            return 0.5

    def _assess_classification_preservation(self, original: np.ndarray, reduced: np.ndarray, 
                                          labels: np.ndarray) -> float:
        """Assess how well classification performance is preserved."""
        if not SKLEARN_AVAILABLE or len(np.unique(labels)) < 2:
            return 0.5
        
        try:
            # Use simple logistic regression for classification comparison
            clf = LogisticRegression(random_state=42, max_iter=1000)
            
            # Performance on original data
            scores_orig = cross_val_score(clf, original, labels, cv=3, scoring="accuracy")
            orig_score = np.mean(scores_orig)
            
            # Performance on reduced data  
            scores_reduced = cross_val_score(clf, reduced, labels, cv=3, scoring="accuracy")
            reduced_score = np.mean(scores_reduced)
            
            # Calculate preservation ratio
            if orig_score == 0:
                return 1.0 if reduced_score == 0 else 0.0
            
            preservation = reduced_score / orig_score
            return max(0.0, min(1.0, preservation))
            
        except Exception as e:
            logger.warning(f"Classification preservation assessment failed: {e}")
            return 0.5

    def _assess_neighborhood_preservation(self, original: np.ndarray, reduced: np.ndarray) -> float:
        """Assess how well local neighborhood structure is preserved."""
        if not SKLEARN_AVAILABLE:
            return 0.5
        
        try:
            n_samples = original.shape[0]
            k = min(10, max(2, n_samples // 10))
            
            if k < 2:
                return 0.5
            
            # Find k-nearest neighbors in original space
            nn_orig = NearestNeighbors(n_neighbors=k)
            nn_orig.fit(original)
            
            # Find k-nearest neighbors in reduced space
            nn_reduced = NearestNeighbors(n_neighbors=k)  
            nn_reduced.fit(reduced)
            
            # Sample points for efficiency on large datasets
            sample_size = min(100, n_samples)
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            
            # Get neighborhoods for sampled points
            neighbors_orig = nn_orig.kneighbors(
                original[sample_indices], return_distance=False
            )
            neighbors_reduced = nn_reduced.kneighbors(
                reduced[sample_indices], return_distance=False  
            )
            
            # Calculate neighborhood preservation scores
            preservation_scores = []
            for i in range(sample_size):
                # Count overlap in neighborhoods
                overlap = len(set(neighbors_orig[i]) & set(neighbors_reduced[i]))
                preservation_scores.append(overlap / k)
            
            return float(np.mean(preservation_scores))
            
        except Exception as e:
            logger.warning(f"Neighborhood preservation assessment failed: {e}")
            return 0.5

    def _compute_overall_quality(self, variance_preservation: float, 
                               clustering_preservation: float,
                               classification_preservation: Optional[float],
                               neighborhood_preservation: float) -> float:
        """Compute weighted overall quality score."""
        try:
            score_components = []
            weights = []
            
            # Always include variance and clustering preservation
            score_components.extend([variance_preservation, clustering_preservation])
            weights.extend([self.variance_weight, self.clustering_weight])
            
            # Add classification if available
            if classification_preservation is not None:
                score_components.append(classification_preservation)
                weights.append(self.classification_weight)
            
            # Add neighborhood preservation with fixed weight
            score_components.append(neighborhood_preservation)
            weights.append(0.2)
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Compute weighted average
            overall_score = np.average(score_components, weights=weights)
            
            return float(np.clip(overall_score, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Overall quality computation failed: {e}")
            return 0.5

    def _basic_evaluation(self, original: np.ndarray, reduced: np.ndarray, start_time: float) -> EvaluationMetrics:
        """Provide basic evaluation when advanced metrics are unavailable."""
        variance_preservation = self.compute_variance_preservation(original, reduced)
        processing_time = time.time() - start_time
        
        return EvaluationMetrics(
            variance_preservation=variance_preservation,
            clustering_preservation=0.5,  # Default
            classification_preservation=None,
            neighborhood_preservation=0.5,  # Default
            overall_quality=variance_preservation * 0.8,  # Conservative estimate
            processing_time=processing_time
        )

    def compare_methods(self, original: np.ndarray, reduced_results: Dict[str, np.ndarray],
                       labels: Optional[np.ndarray] = None) -> Dict[str, EvaluationMetrics]:
        """Compare multiple dimensionality reduction methods."""
        results = {}
        
        for method_name, reduced_data in reduced_results.items():
            try:
                results[method_name] = self.evaluate_quality(original, reduced_data, labels)
                logger.info(f"Evaluated method '{method_name}': quality={results[method_name].overall_quality:.3f}")
            except Exception as e:
                logger.error(f"Failed to evaluate method '{method_name}': {e}")
                
        return results

    def get_best_method(self, evaluation_results: Dict[str, EvaluationMetrics]) -> Tuple[str, EvaluationMetrics]:
        """Get the best performing method from evaluation results."""
        if not evaluation_results:
            raise ValueError("No evaluation results provided")
        
        best_method = max(
            evaluation_results.items(), 
            key=lambda x: x[1].overall_quality
        )
        
        return best_method[0], best_method[1]

    def generate_evaluation_report(self, original: np.ndarray, reduced: np.ndarray,
                                 labels: Optional[np.ndarray] = None, method_name: str = "Unknown") -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        metrics = self.evaluate_quality(original, reduced, labels)
        
        report = {
            "method": method_name,
            "data_summary": {
                "original_shape": original.shape,
                "reduced_shape": reduced.shape,
                "dimensionality_reduction": f"{original.shape[1]} → {reduced.shape[1]}",
                "reduction_ratio": reduced.shape[1] / original.shape[1] if original.shape[1] > 0 else 0,
                "has_labels": labels is not None
            },
            "quality_metrics": {
                "variance_preservation": metrics.variance_preservation,
                "clustering_preservation": metrics.clustering_preservation,
                "classification_preservation": metrics.classification_preservation,
                "neighborhood_preservation": metrics.neighborhood_preservation,
                "overall_quality": metrics.overall_quality
            },
            "quality_assessment": self._assess_quality_level(metrics.overall_quality),
            "recommendations": self._generate_recommendations(metrics, original.shape, reduced.shape),
            "processing_time": metrics.processing_time
        }
        
        return report

    def _assess_quality_level(self, overall_quality: float) -> Dict[str, Any]:
        """Assess the quality level of the reduction."""
        if overall_quality >= 0.8:
            level = "Excellent"
            description = "High-quality reduction with good preservation of data structure"
        elif overall_quality >= 0.6:
            level = "Good"  
            description = "Acceptable reduction quality with reasonable structure preservation"
        elif overall_quality >= 0.4:
            level = "Moderate"
            description = "Moderate quality reduction, consider parameter tuning"
        else:
            level = "Poor"
            description = "Low quality reduction, consider different method or parameters"
        
        return {
            "level": level,
            "score": overall_quality,
            "description": description
        }

    def _generate_recommendations(self, metrics: EvaluationMetrics, 
                                original_shape: Tuple[int, int], 
                                reduced_shape: Tuple[int, int]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        if metrics.variance_preservation < 0.5:
            recommendations.append("Consider increasing target dimensions to preserve more variance")
        
        if metrics.clustering_preservation < 0.4:
            recommendations.append("Clustering structure poorly preserved; consider manifold learning methods")
            
        if metrics.classification_preservation is not None and metrics.classification_preservation < 0.7:
            recommendations.append("Classification performance degraded; consider supervised reduction methods")
            
        if metrics.neighborhood_preservation < 0.4:
            recommendations.append("Local structure poorly preserved; consider t-SNE or UMAP")
            
        reduction_ratio = reduced_shape[1] / original_shape[1]
        if reduction_ratio > 0.8:
            recommendations.append("Limited dimensionality reduction achieved; consider more aggressive reduction")
            
        if not recommendations:
            recommendations.append("Reduction quality is satisfactory")
            
        return recommendations

class ReductionEvaluatorServiceFactory:
    """Factory for creating evaluation services with different configurations."""

    @staticmethod
    def create_fast_evaluator() -> ReductionEvaluatorService:
        """Create evaluator optimized for speed (basic metrics only)."""
        return ReductionEvaluatorService(enable_advanced_metrics=False)

    @staticmethod
    def create_comprehensive_evaluator() -> ReductionEvaluatorService:
        """Create evaluator with all advanced metrics enabled."""
        return ReductionEvaluatorService(enable_advanced_metrics=True)

    @staticmethod
    def create_variance_focused_evaluator() -> ReductionEvaluatorService:
        """Create evaluator focused on variance preservation."""
        return ReductionEvaluatorService(
            variance_weight=0.7,
            clustering_weight=0.2,
            classification_weight=0.1
        )

    @staticmethod  
    def create_clustering_focused_evaluator() -> ReductionEvaluatorService:
        """Create evaluator focused on clustering preservation."""
        return ReductionEvaluatorService(
            variance_weight=0.2,
            clustering_weight=0.6,
            classification_weight=0.2
        )