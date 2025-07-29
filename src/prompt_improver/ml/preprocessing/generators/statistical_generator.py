"""Statistical Data Generator Module

Traditional statistical methods for synthetic data generation using scikit-learn.
Extracted from synthetic_data_generator.py for focused functionality.

This module contains:
- GenerationMethodMetrics: Performance tracking for generation methods
- MethodPerformanceTracker: Auto-selection of best generation methods
- Statistical generation methods using make_classification
"""

import logging
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.datasets import make_classification

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
            return "statistical"  # Default fallback

        # Weight rankings by gap-specific performance
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

            # Calculate weighted score (recent performance weighted higher)
            recent_metrics = history[-10:]  # Last 10 generations
            weights = np.linspace(0.5, 1.0, len(recent_metrics))

            quality_scores = [m.quality_score for m in recent_metrics]
            diversity_scores = [m.diversity_score for m in recent_metrics]
            success_rates = [m.success_rate for m in recent_metrics]

            weighted_quality = np.average(quality_scores, weights=weights)
            weighted_diversity = np.average(diversity_scores, weights=weights)
            weighted_success = np.average(success_rates, weights=weights)

            # Combined score (2025 best practice weighting)
            self.method_rankings[method] = (
                0.4 * weighted_quality +
                0.3 * weighted_diversity +
                0.3 * weighted_success
            )

    def _calculate_gap_bonus(self, method: str, performance_gaps: dict[str, float]) -> float:
        """Calculate bonus score based on method's effectiveness for specific gaps"""
        if method not in self.method_history:
            return 0.0

        # Analyze historical effectiveness for similar gaps
        relevant_metrics = []
        for metrics in self.method_history[method][-5:]:  # Recent history
            gap_similarity = self._calculate_gap_similarity(
                metrics.performance_gaps_addressed, performance_gaps
            )
            if gap_similarity > 0.5:  # Similar gap patterns
                relevant_metrics.append(metrics)

        if not relevant_metrics:
            return 0.0

        # Return average effectiveness for similar gaps
        return np.mean([m.quality_score for m in relevant_metrics]) * 0.2

    def _calculate_gap_similarity(self, gaps1: dict[str, float], gaps2: dict[str, float]) -> float:
        """Calculate similarity between two gap patterns"""
        common_keys = set(gaps1.keys()) & set(gaps2.keys())
        if not common_keys:
            return 0.0

        similarities = []
        for key in common_keys:
            # Normalized difference (closer to 0 = more similar)
            diff = abs(gaps1[key] - gaps2[key])
            similarity = max(0, 1 - diff)
            similarities.append(similarity)

        return np.mean(similarities)


class StatisticalDataGenerator:
    """Statistical data generator using scikit-learn methods"""
    
    def __init__(self, random_state: int = 42):
        """Initialize statistical generator
        
        Args:
            random_state: Random seed for reproducible generation
        """
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
    async def generate_statistical_samples(
        self, 
        sample_count: int, 
        data_dim: int,
        n_clusters_per_class: int = 2,
        class_sep: float = 0.8
    ) -> list:
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
            X, _ = make_classification(
                n_samples=sample_count,
                n_features=data_dim,
                n_informative=data_dim,
                n_redundant=0,
                n_clusters_per_class=n_clusters_per_class,
                class_sep=class_sep,
                random_state=self.random_state
            )
            
            # Convert to list format for compatibility
            return X.tolist()
            
        except Exception as e:
            logger.error(f"Statistical generation failed: {e}")
            # Fallback to random generation
            return self.rng.randn(sample_count, data_dim).tolist()
    
    def generate_domain_statistical_features(
        self,
        sample_count: int,
        feature_names: list[str],
        domain_name: str = "technical"
    ) -> tuple[list[list[float]], list[float]]:
        """Generate domain-specific statistical features
        
        Args:
            sample_count: Number of samples to generate
            feature_names: Names of features to generate
            domain_name: Domain type for generation characteristics
            
        Returns:
            Tuple of (features, effectiveness_scores)
        """
        # Use optimized scikit-learn feature generation with modern parameters
        base_features, base_effectiveness = make_classification(
            n_samples=sample_count,
            n_features=len(feature_names),
            n_informative=len(feature_names),  # All features informative
            n_redundant=0,  # No redundant features
            n_clusters_per_class=3,  # Increased clusters for better diversity
            class_sep=0.8,  # Good class separation
            flip_y=0.01,  # Very low noise for quality
            weights=[0.4, 0.6],  # Realistic class imbalance
            random_state=self.random_state
        )
        
        # Enhance features based on domain characteristics
        enhanced_features = self._apply_domain_enhancement(
            base_features, domain_name
        )
        
        # Normalize effectiveness scores to [0, 1] range
        normalized_effectiveness = (base_effectiveness + 1) / 2
        
        return enhanced_features.tolist(), normalized_effectiveness.tolist()
    
    def _apply_domain_enhancement(
        self, 
        features: np.ndarray, 
        domain_name: str
    ) -> np.ndarray:
        """Apply domain-specific enhancements to statistical features
        
        Args:
            features: Base feature matrix
            domain_name: Domain type for enhancement
            
        Returns:
            Enhanced feature matrix
        """
        enhanced = features.copy()
        
        # Domain-specific statistical modifications
        if domain_name == "technical":
            # Technical domains tend to have higher precision requirements
            enhanced = enhanced * 1.2
            enhanced = np.clip(enhanced, -3, 3)  # Bounded precision
        elif domain_name == "creative":
            # Creative domains show more variance and outliers
            noise = self.rng.normal(0, 0.1, enhanced.shape)
            enhanced = enhanced + noise
        elif domain_name == "analytical":
            # Analytical domains show systematic patterns
            enhanced = enhanced * 0.9
            enhanced = np.round(enhanced, 2)  # More structured
        elif domain_name == "instructional":
            # Instructional content tends to be more moderate
            enhanced = enhanced * 0.8
            enhanced = np.clip(enhanced, -2, 2)
        elif domain_name == "conversational":
            # Conversational data shows natural variation
            enhanced = enhanced + self.rng.normal(0, 0.05, enhanced.shape)
            
        return enhanced


class StatisticalQualityAssessor:
    """Quality assessment for statistically generated data"""
    
    def __init__(self, random_state: int = 42):
        self.rng = np.random.RandomState(random_state)
    
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
            # Convert to numpy array for analysis
            data = np.array(samples)
            
            # Quality metrics for statistical data
            quality_components = []
            
            # 1. Data distribution quality (0-0.3)
            dist_quality = self._assess_distribution_quality(data)
            quality_components.append(dist_quality * 0.3)
            
            # 2. Feature variance quality (0-0.3)
            variance_quality = self._assess_variance_quality(data)
            quality_components.append(variance_quality * 0.3)
            
            # 3. Sample diversity quality (0-0.4)
            diversity_quality = self._assess_diversity_quality(data)
            quality_components.append(diversity_quality * 0.4)
            
            return sum(quality_components)
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return 0.5  # Neutral score on failure
    
    def _assess_distribution_quality(self, data: np.ndarray) -> float:
        """Assess the distribution quality of the data"""
        try:
            # Check for reasonable mean and std
            means = np.mean(data, axis=0)
            stds = np.std(data, axis=0)
            
            # Good statistical data should have reasonable spread
            mean_quality = np.mean(np.abs(means) < 2)  # Means not too extreme
            std_quality = np.mean((stds > 0.1) & (stds < 3))  # Good variance
            
            return (mean_quality + std_quality) / 2
            
        except Exception:
            return 0.5
    
    def _assess_variance_quality(self, data: np.ndarray) -> float:
        """Assess the variance structure of the data"""
        try:
            # Calculate feature-wise variance
            variances = np.var(data, axis=0)
            
            # Good variance should be consistent across features
            variance_consistency = 1.0 - np.std(variances) / (np.mean(variances) + 1e-8)
            variance_consistency = np.clip(variance_consistency, 0, 1)
            
            # Features should have sufficient variance
            min_variance = np.min(variances)
            variance_adequacy = min(1.0, min_variance / 0.1)
            
            return (variance_consistency + variance_adequacy) / 2
            
        except Exception:
            return 0.5
    
    def _assess_diversity_quality(self, data: np.ndarray) -> float:
        """Assess the diversity of generated samples"""
        try:
            # Calculate pairwise distances to assess diversity
            n_samples = min(100, data.shape[0])  # Limit for efficiency
            sample_indices = self.rng.choice(data.shape[0], n_samples, replace=False)
            sample_data = data[sample_indices]
            
            # Calculate mean pairwise distance
            distances = []
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    dist = np.linalg.norm(sample_data[i] - sample_data[j])
                    distances.append(dist)
            
            if not distances:
                return 0.5
                
            mean_distance = np.mean(distances)
            
            # Normalize based on data dimensionality and scale
            data_scale = np.sqrt(data.shape[1])  # Expected scale for random data
            normalized_distance = mean_distance / (data_scale * 2)
            
            # Good diversity should have reasonable separation
            return min(1.0, normalized_distance)
            
        except Exception:
            return 0.5