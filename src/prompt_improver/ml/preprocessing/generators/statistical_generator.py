"""Statistical Data Generator Module

Traditional statistical methods for synthetic data generation using scikit-learn.
Extracted from synthetic_data_generator.py for focused functionality.

This module contains:
- GenerationMethodMetrics: Performance tracking for generation methods
- MethodPerformanceTracker: Auto-selection of best generation methods
- Statistical generation methods using make_classification
"""
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
from typing import TYPE_CHECKING
from prompt_improver.core.utils.lazy_ml_loader import get_numpy, get_sklearn

if TYPE_CHECKING:
    from sklearn.datasets import make_classification
    import numpy as np
else:
    # Runtime lazy loading
    def _get_sklearn_imports():
        sklearn = get_sklearn()
        return sklearn.datasets.make_classification
    
    make_classification = _get_sklearn_imports()
logger = logging.getLogger(__name__)

@dataclass
class GenerationMethodMetrics:
    """Performance metrics for generation methods (2025 best practice)"""
    method_name: str
    generation_time: float
    quality_score: float
    diversity_score: float
    memory_usage_mb: float
    success_rate: float
    samples_generated: int
    timestamp: datetime
    performance_gaps_addressed: dict[str, float]

class MethodPerformanceTracker:
    """Tracks performance of different generation methods for auto-selection (2025 best practice)"""

    def __init__(self):
        self.method_history: dict[str, list[GenerationMethodMetrics]] = {}
        self.method_rankings: dict[str, float] = {}

    def record_performance(self, metrics: GenerationMethodMetrics) -> None:
        """Record performance metrics for a generation method"""
        if metrics.method_name not in self.method_history:
            self.method_history[metrics.method_name] = []
        self.method_history[metrics.method_name].append(metrics)
        self._update_rankings()

    def get_best_method(self, performance_gaps: dict[str, float]) -> str:
        """Select best method based on historical performance and current gaps"""
        if not self.method_rankings:
            return 'statistical'
        weighted_scores = {}
        for method, base_score in self.method_rankings.items():
            gap_bonus = self._calculate_gap_bonus(method, performance_gaps)
            weighted_scores[method] = base_score + gap_bonus
        return max(weighted_scores, key=weighted_scores.get)

    def _update_rankings(self) -> None:
        """Update method rankings based on recent performance"""
        for method, history in self.method_history.items():
            if not history:
                continue
            recent_metrics = history[-10:]
            weights = get_numpy().linspace(0.5, 1.0, len(recent_metrics))
            quality_scores = [m.quality_score for m in recent_metrics]
            diversity_scores = [m.diversity_score for m in recent_metrics]
            success_rates = [m.success_rate for m in recent_metrics]
            weighted_quality = get_numpy().average(quality_scores, weights=weights)
            weighted_diversity = get_numpy().average(diversity_scores, weights=weights)
            weighted_success = get_numpy().average(success_rates, weights=weights)
            self.method_rankings[method] = 0.4 * weighted_quality + 0.3 * weighted_diversity + 0.3 * weighted_success

    def _calculate_gap_bonus(self, method: str, performance_gaps: dict[str, float]) -> float:
        """Calculate bonus score based on method's effectiveness for specific gaps"""
        if method not in self.method_history:
            return 0.0
        relevant_metrics = []
        for metrics in self.method_history[method][-5:]:
            gap_similarity = self._calculate_gap_similarity(metrics.performance_gaps_addressed, performance_gaps)
            if gap_similarity > 0.5:
                relevant_metrics.append(metrics)
        if not relevant_metrics:
            return 0.0
        return get_numpy().mean([m.quality_score for m in relevant_metrics]) * 0.2

    def _calculate_gap_similarity(self, gaps1: dict[str, float], gaps2: dict[str, float]) -> float:
        """Calculate similarity between two gap patterns"""
        common_keys = set(gaps1.keys()) & set(gaps2.keys())
        if not common_keys:
            return 0.0
        similarities = []
        for key in common_keys:
            diff = abs(gaps1[key] - gaps2[key])
            similarity = max(0, 1 - diff)
            similarities.append(similarity)
        return get_numpy().mean(similarities)

class StatisticalDataGenerator:
    """Statistical data generator using scikit-learn methods"""

    def __init__(self, random_state: int=42):
        """Initialize statistical generator
        
        Args:
            random_state: Random seed for reproducible generation
        """
        self.random_state = random_state
        self.rng = get_numpy().random.RandomState(random_state)

    async def generate_statistical_samples(self, sample_count: int, data_dim: int, n_clusters_per_class: int=2, class_sep: float=0.8) -> list:
        """Generate samples using statistical methods
        
        Args:
            sample_count: Number of samples to generate
            data_dim: Dimensionality of the data
            n_clusters_per_class: Number of clusters per class for diversity
            class_sep: Separation between classes for quality control
            
        Returns:
            List of generated samples
        """
        try:
            X, _ = make_classification(n_samples=sample_count, n_features=data_dim, n_informative=data_dim, n_redundant=0, n_clusters_per_class=n_clusters_per_class, class_sep=class_sep, random_state=self.random_state)
            return X.tolist()
        except Exception as e:
            logger.error('Statistical generation failed: %s', e)
            return self.rng.randn(sample_count, data_dim).tolist()

    def generate_domain_statistical_features(self, sample_count: int, feature_names: list[str], domain_name: str='technical') -> tuple[list[list[float]], list[float]]:
        """Generate domain-specific statistical features
        
        Args:
            sample_count: Number of samples to generate
            feature_names: Names of features to generate
            domain_name: Domain type for generation characteristics
            
        Returns:
            Tuple of (features, effectiveness_scores)
        """
        base_features, base_effectiveness = make_classification(n_samples=sample_count, n_features=len(feature_names), n_informative=len(feature_names), n_redundant=0, n_clusters_per_class=3, class_sep=0.8, flip_y=0.01, weights=[0.4, 0.6], random_state=self.random_state)
        enhanced_features = self._apply_domain_enhancement(base_features, domain_name)
        normalized_effectiveness = (base_effectiveness + 1) / 2
        return (enhanced_features.tolist(), normalized_effectiveness.tolist())

    def _apply_domain_enhancement(self, features: get_numpy().ndarray, domain_name: str) -> get_numpy().ndarray:
        """Apply domain-specific enhancements to statistical features
        
        Args:
            features: Base feature matrix
            domain_name: Domain type for enhancement
            
        Returns:
            Enhanced feature matrix
        """
        enhanced = features.copy()
        if domain_name == 'technical':
            enhanced = enhanced * 1.2
            enhanced = get_numpy().clip(enhanced, -3, 3)
        elif domain_name == 'creative':
            noise = self.rng.normal(0, 0.1, enhanced.shape)
            enhanced = enhanced + noise
        elif domain_name == 'analytical':
            enhanced = enhanced * 0.9
            enhanced = get_numpy().round(enhanced, 2)
        elif domain_name == 'instructional':
            enhanced = enhanced * 0.8
            enhanced = get_numpy().clip(enhanced, -2, 2)
        elif domain_name == 'conversational':
            enhanced = enhanced + self.rng.normal(0, 0.05, enhanced.shape)
        return enhanced

class StatisticalQualityAssessor:
    """Quality assessment for statistically generated data"""

    def __init__(self, random_state: int=42):
        self.rng = get_numpy().random.RandomState(random_state)

    def assess_sample_quality(self, samples: list) -> float:
        """Assess quality of statistically generated samples
        
        Args:
            samples: Generated samples to assess
            
        Returns:
            Quality score between 0 and 1
        """
        if not samples:
            return 0.0
        try:
            data = get_numpy().array(samples)
            quality_components = []
            dist_quality = self._assess_distribution_quality(data)
            quality_components.append(dist_quality * 0.3)
            variance_quality = self._assess_variance_quality(data)
            quality_components.append(variance_quality * 0.3)
            diversity_quality = self._assess_diversity_quality(data)
            quality_components.append(diversity_quality * 0.4)
            return sum(quality_components)
        except Exception as e:
            logger.error('Quality assessment failed: %s', e)
            return 0.5

    def _assess_distribution_quality(self, data: get_numpy().ndarray) -> float:
        """Assess the distribution quality of the data"""
        try:
            means = get_numpy().mean(data, axis=0)
            stds = get_numpy().std(data, axis=0)
            mean_quality = get_numpy().mean(get_numpy().abs(means) < 2)
            std_quality = get_numpy().mean((stds > 0.1) & (stds < 3))
            return (mean_quality + std_quality) / 2
        except Exception:
            return 0.5

    def _assess_variance_quality(self, data: get_numpy().ndarray) -> float:
        """Assess the variance structure of the data"""
        try:
            variances = get_numpy().var(data, axis=0)
            variance_consistency = 1.0 - get_numpy().std(variances) / (get_numpy().mean(variances) + 1e-08)
            variance_consistency = get_numpy().clip(variance_consistency, 0, 1)
            min_variance = get_numpy().min(variances)
            variance_adequacy = min(1.0, min_variance / 0.1)
            return (variance_consistency + variance_adequacy) / 2
        except Exception:
            return 0.5

    def _assess_diversity_quality(self, data: get_numpy().ndarray) -> float:
        """Assess the diversity of generated samples"""
        try:
            n_samples = min(100, data.shape[0])
            sample_indices = self.rng.choice(data.shape[0], n_samples, replace=False)
            sample_data = data[sample_indices]
            distances = []
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    dist = get_numpy().linalg.norm(sample_data[i] - sample_data[j])
                    distances.append(dist)
            if not distances:
                return 0.5
            mean_distance = get_numpy().mean(distances)
            data_scale = get_numpy().sqrt(data.shape[1])
            normalized_distance = mean_distance / (data_scale * 2)
            return min(1.0, normalized_distance)
        except Exception:
            return 0.5