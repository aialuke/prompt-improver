"""Enhanced Quality Scoring System for Synthetic Data Generation

Multi-dimensional quality assessment framework inspired by research from:
- Firecrawl: NLP synthetic data generation best practices
- YData Profiling: Comprehensive data quality profiling
- Scikit-learn: Model evaluation and multi-metric scoring
- MAPIE: Prediction interval estimation and uncertainty quantification

Implements granular quality scoring to replace binary pass/fail system.
"""

import asyncio
import logging
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class QualityDimension:
    """Individual quality dimension with granular scoring"""

    name: str
    score: float  # 0.0 - 1.0
    weight: float  # Relative importance
    sub_metrics: dict[str, float]
    threshold_met: bool
    confidence_interval: tuple[float, float]
    interpretation: str


@dataclass
class EnhancedQualityMetrics:
    """Comprehensive quality assessment with multi-dimensional scoring"""

    # Core Dimensions (from Firecrawl research)
    fidelity: QualityDimension
    utility: QualityDimension
    privacy: QualityDimension

    # Additional Dimensions (from Context7 research)
    statistical_validity: QualityDimension
    diversity: QualityDimension
    consistency: QualityDimension

    # Composite Scores
    overall_score: float
    confidence_score: float
    recommendation_tier: str  # "EXCELLENT", "GOOD", "ADEQUATE", "POOR"

    # Metadata
    assessment_timestamp: str
    total_samples: int
    assessment_duration: float


class EnhancedQualityScorer:
    """Advanced quality scoring system with multi-dimensional assessment"""

    def __init__(self, confidence_level: float = 0.95):
        """Initialize enhanced quality scorer

        Args:
            confidence_level: Confidence level for statistical assessments
        """
        self.confidence_level = confidence_level
        self.dimension_weights = {
            "fidelity": 0.25,  # Statistical similarity to real data
            "utility": 0.25,  # ML pipeline effectiveness
            "privacy": 0.15,  # Privacy preservation
            "statistical_validity": 0.15,  # Statistical correctness
            "diversity": 0.15,  # Sample diversity and coverage
            "consistency": 0.05,  # Internal consistency
        }

        # Quality thresholds for interpretation
        self.tier_thresholds = {
            "EXCELLENT": 0.85,
            "GOOD": 0.70,
            "ADEQUATE": 0.55,
            "POOR": 0.0,
        }

    async def assess_comprehensive_quality(
        self,
        features: list[list[float]],
        effectiveness_scores: list[float],
        domain_counts: dict[str, int],
        generation_params: dict[str, Any],
    ) -> EnhancedQualityMetrics:
        """Perform comprehensive multi-dimensional quality assessment

        Args:
            features: Generated feature vectors
            effectiveness_scores: Effectiveness scores
            domain_counts: Distribution across domains
            generation_params: Generation parameters for context

        Returns:
            EnhancedQualityMetrics with detailed assessment
        """
        start_time = datetime.utcnow()

        features_array = np.array(features)
        effectiveness_array = np.array(effectiveness_scores)

        logger.info("Starting comprehensive quality assessment")

        # Assess each dimension
        fidelity = await self._assess_fidelity(features_array, effectiveness_array)
        utility = await self._assess_utility(
            features_array, effectiveness_array, generation_params
        )
        privacy = await self._assess_privacy(features_array, effectiveness_array)
        statistical_validity = await self._assess_statistical_validity(
            features_array, effectiveness_array
        )
        diversity = await self._assess_diversity(features_array, domain_counts)
        consistency = await self._assess_consistency(
            features_array, effectiveness_array
        )

        # Calculate composite scores
        overall_score = self._calculate_weighted_score([
            fidelity,
            utility,
            privacy,
            statistical_validity,
            diversity,
            consistency,
        ])

        confidence_score = self._calculate_confidence_score([
            fidelity,
            utility,
            privacy,
            statistical_validity,
            diversity,
            consistency,
        ])

        recommendation_tier = self._determine_recommendation_tier(overall_score)

        duration = (datetime.utcnow() - start_time).total_seconds()

        return EnhancedQualityMetrics(
            fidelity=fidelity,
            utility=utility,
            privacy=privacy,
            statistical_validity=statistical_validity,
            diversity=diversity,
            consistency=consistency,
            overall_score=overall_score,
            confidence_score=confidence_score,
            recommendation_tier=recommendation_tier,
            assessment_timestamp=start_time.isoformat(),
            total_samples=len(features),
            assessment_duration=duration,
        )

    async def _assess_fidelity(
        self, features: np.ndarray, effectiveness: np.ndarray
    ) -> QualityDimension:
        """Assess statistical fidelity using distribution similarity metrics"""
        sub_metrics = {}

        # Distribution similarity (Kolmogorov-Smirnov tests)
        ks_scores = []
        for i in range(features.shape[1]):
            # Compare to normal distribution as baseline
            ks_stat, ks_p = stats.kstest(
                features[:, i],
                "norm",
                args=(features[:, i].mean(), features[:, i].std()),
            )
            ks_scores.append(1.0 - ks_stat)  # Convert to similarity score

        sub_metrics["distribution_similarity"] = np.mean(ks_scores)

        # Jensen-Shannon divergence for effectiveness distribution
        # Compare to beta distribution (expected for synthetic data)
        hist, bins = np.histogram(effectiveness, bins=20, density=True)
        hist = hist / hist.sum()  # Normalize

        # Expected beta distribution
        alpha, beta_param = 2.5, 1.5  # Typical for effective prompts
        expected_beta = stats.beta.pdf(bins[:-1], alpha, beta_param)
        expected_beta = expected_beta / expected_beta.sum()

        js_divergence = jensenshannon(hist, expected_beta)
        sub_metrics["effectiveness_distribution"] = 1.0 - js_divergence

        # Feature correlation fidelity
        corr_matrix = np.corrcoef(features.T)
        # Ideal correlation matrix should be close to identity (uncorrelated features)
        identity_matrix = np.eye(corr_matrix.shape[0])
        corr_deviation = float(np.mean(np.abs(corr_matrix - identity_matrix)))
        sub_metrics["feature_independence"] = max(0.0, 1.0 - corr_deviation)

        # Aggregate fidelity score
        fidelity_score = np.mean(list(sub_metrics.values()))

        # Confidence interval using bootstrap
        confidence_interval = self._bootstrap_confidence_interval(
            lambda x: np.mean([sub_metrics[k] for k in sub_metrics]),
            [fidelity_score] * 100,  # Simplified for this metric
        )

        return QualityDimension(
            name="fidelity",
            score=float(fidelity_score),
            weight=self.dimension_weights["fidelity"],
            sub_metrics=sub_metrics,
            threshold_met=bool(fidelity_score >= 0.6),
            confidence_interval=confidence_interval,
            interpretation=self._interpret_fidelity_score(fidelity_score),
        )

    async def _assess_utility(
        self, features: np.ndarray, effectiveness: np.ndarray, params: dict
    ) -> QualityDimension:
        """Assess ML utility using predictive modeling potential"""
        sub_metrics = {}

        # Feature variance (higher variance = more informative) - normalize to 0-1
        feature_variances = np.var(features, axis=0)
        # Normalize variance to 0-1 scale (typical range 0-2 for normalized features)
        normalized_variance = min(1.0, float(np.mean(feature_variances)) / 2.0)
        sub_metrics["feature_informativeness"] = normalized_variance

        # Predictability assessment using simple clustering
        if len(features) >= 10:  # Need minimum samples
            try:
                # Standardize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)

                # PCA to assess dimensionality
                pca = PCA()
                pca.fit(features_scaled)
                explained_variance_ratio = pca.explained_variance_ratio_[:3].sum()
                sub_metrics["dimensionality_quality"] = explained_variance_ratio

                # Clustering quality (if enough samples)
                if len(features) >= 20:
                    n_clusters = min(5, len(features) // 4)
                    kmeans = KMeans(
                        n_clusters=n_clusters, random_state=42, n_init="auto"
                    )
                    cluster_labels = kmeans.fit_predict(features_scaled)

                    silhouette = silhouette_score(features_scaled, cluster_labels)
                    sub_metrics["clustering_potential"] = max(0.0, silhouette)
                else:
                    sub_metrics["clustering_potential"] = 0.5  # Neutral score

            except Exception as e:
                logger.warning(f"Utility assessment error: {e}")
                sub_metrics["dimensionality_quality"] = 0.5
                sub_metrics["clustering_potential"] = 0.5
        else:
            sub_metrics["dimensionality_quality"] = 0.3  # Low sample count penalty
            sub_metrics["clustering_potential"] = 0.3

        # Effectiveness score distribution quality
        eff_std = np.std(effectiveness)
        eff_range = np.max(effectiveness) - np.min(effectiveness)
        sub_metrics["effectiveness_variance"] = min(
            1.0, eff_std / 0.3
        )  # Normalize to 0.3 target
        sub_metrics["effectiveness_range"] = float(
            min(1.0, eff_range / 0.8)
        )  # Normalize to 0.8 target

        utility_score = np.mean(list(sub_metrics.values()))

        confidence_interval = self._bootstrap_confidence_interval(
            lambda x: np.mean(x), list(sub_metrics.values())
        )

        return QualityDimension(
            name="utility",
            score=float(utility_score),
            weight=self.dimension_weights["utility"],
            sub_metrics=sub_metrics,
            threshold_met=bool(utility_score >= 0.6),
            confidence_interval=confidence_interval,
            interpretation=self._interpret_utility_score(float(utility_score)),
        )

    async def _assess_privacy(
        self, features: np.ndarray, effectiveness: np.ndarray
    ) -> QualityDimension:
        """Assess privacy preservation through uniqueness and anonymity"""
        sub_metrics = {}

        # Sample uniqueness (all samples should be unique for synthetic data)
        unique_samples = len(np.unique(features, axis=0))
        total_samples = len(features)
        sub_metrics["sample_uniqueness"] = unique_samples / total_samples

        # Feature value uniqueness (avoid identical feature vectors)
        feature_uniqueness_scores = []
        for i in range(features.shape[1]):
            unique_values = len(np.unique(features[:, i]))
            max_possible = min(total_samples, 100)  # Cap for continuous features
            feature_uniqueness_scores.append(unique_values / max_possible)

        sub_metrics["feature_uniqueness"] = float(np.mean(feature_uniqueness_scores))

        # Effectiveness anonymity (no exact duplicates in effectiveness scores)
        unique_effectiveness = len(np.unique(effectiveness))
        sub_metrics["effectiveness_anonymity"] = unique_effectiveness / total_samples

        # Distance-based privacy (minimum distance between samples)
        if len(features) >= 2:
            from scipy.spatial.distance import pdist

            distances = pdist(features)
            min_distance = float(np.min(distances))
            mean_distance = float(np.mean(distances))
            sub_metrics["inter_sample_distance"] = min(
                1.0, min_distance / (mean_distance * 0.1)
            )
        else:
            sub_metrics["inter_sample_distance"] = 1.0

        privacy_score = np.mean(list(sub_metrics.values()))

        confidence_interval = self._bootstrap_confidence_interval(
            lambda x: np.mean(x), list(sub_metrics.values())
        )

        return QualityDimension(
            name="privacy",
            score=float(privacy_score),
            weight=self.dimension_weights["privacy"],
            sub_metrics=sub_metrics,
            threshold_met=bool(privacy_score >= 0.8),  # Higher threshold for privacy
            confidence_interval=confidence_interval,
            interpretation=self._interpret_privacy_score(float(privacy_score)),
        )

    async def _assess_statistical_validity(
        self, features: np.ndarray, effectiveness: np.ndarray
    ) -> QualityDimension:
        """Assess statistical validity and correctness"""
        sub_metrics = {}

        # No NaN or infinite values
        finite_features = np.isfinite(features).all()
        finite_effectiveness = np.isfinite(effectiveness).all()
        sub_metrics["data_validity"] = float(finite_features and finite_effectiveness)

        # Reasonable value ranges (0-1 for most features, effectiveness scores)
        features_in_range = np.all(
            (features >= -5) & (features <= 5)
        )  # Reasonable bounds
        effectiveness_in_range = np.all((effectiveness >= 0) & (effectiveness <= 1))
        sub_metrics["value_ranges"] = float(
            features_in_range and effectiveness_in_range
        )

        # Statistical moment validity (reasonable means, std devs)
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)

        # Means should be reasonable (not extreme outliers)
        reasonable_means = np.all(np.abs(feature_means) <= 3)
        # Standard deviations should be positive and reasonable
        reasonable_stds = np.all((feature_stds > 0) & (feature_stds <= 5))

        sub_metrics["statistical_moments"] = float(reasonable_means and reasonable_stds)

        # Effectiveness distribution validity
        eff_mean = np.mean(effectiveness)
        eff_std = np.std(effectiveness)

        # Should have reasonable mean (0.3-0.8) and std (0.1-0.4)
        reasonable_eff_mean = 0.2 <= eff_mean <= 0.9
        reasonable_eff_std = 0.05 <= eff_std <= 0.5

        sub_metrics["effectiveness_validity"] = float(
            reasonable_eff_mean and reasonable_eff_std
        )

        validity_score = np.mean(list(sub_metrics.values()))

        confidence_interval = self._bootstrap_confidence_interval(
            lambda x: np.mean(x), list(sub_metrics.values())
        )

        return QualityDimension(
            name="statistical_validity",
            score=float(validity_score),
            weight=self.dimension_weights["statistical_validity"],
            sub_metrics=sub_metrics,
            threshold_met=bool(validity_score >= 0.9),  # High threshold for validity
            confidence_interval=confidence_interval,
            interpretation=self._interpret_validity_score(float(validity_score)),
        )

    async def _assess_diversity(
        self, features: np.ndarray, domain_counts: dict[str, int]
    ) -> QualityDimension:
        """Assess sample diversity and domain coverage"""
        sub_metrics = {}

        # Domain distribution balance
        total_samples = sum(domain_counts.values())
        domain_proportions = [count / total_samples for count in domain_counts.values()]

        # Shannon entropy for domain diversity
        shannon_entropy = -sum(p * np.log(p) for p in domain_proportions if p > 0)
        max_entropy = np.log(len(domain_counts))
        sub_metrics["domain_diversity"] = (
            shannon_entropy / max_entropy if max_entropy > 0 else 0
        )

        # Feature space coverage using PCA
        if len(features) >= 5:
            try:
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)

                # Measure spread in first few principal components
                pca = PCA(n_components=min(3, features.shape[1]))
                pca_features = pca.fit_transform(features_scaled)

                # Coverage as average std dev across PC dimensions
                pc_coverage = np.mean(np.std(pca_features, axis=0))
                sub_metrics["feature_space_coverage"] = min(1.0, pc_coverage / 2.0)

            except Exception:
                sub_metrics["feature_space_coverage"] = 0.5
        else:
            sub_metrics["feature_space_coverage"] = 0.3

        # Effectiveness score diversity
        effectiveness_range = (
            np.max(features) - np.min(features) if len(features) > 1 else 0
        )
        sub_metrics["effectiveness_diversity"] = min(1.0, effectiveness_range / 1.0)

        # Sample count adequacy
        min_samples_per_domain = 10
        adequate_samples = all(
            count >= min_samples_per_domain for count in domain_counts.values()
        )
        sub_metrics["sample_adequacy"] = float(adequate_samples)

        diversity_score = np.mean(list(sub_metrics.values()))

        confidence_interval = self._bootstrap_confidence_interval(
            lambda x: np.mean(x), list(sub_metrics.values())
        )

        return QualityDimension(
            name="diversity",
            score=float(diversity_score),
            weight=self.dimension_weights["diversity"],
            sub_metrics=sub_metrics,
            threshold_met=bool(diversity_score >= 0.7),
            confidence_interval=confidence_interval,
            interpretation=self._interpret_diversity_score(diversity_score),
        )

    async def _assess_consistency(
        self, features: np.ndarray, effectiveness: np.ndarray
    ) -> QualityDimension:
        """Assess internal consistency and logical relationships"""
        sub_metrics = {}

        # Feature-effectiveness correlation consistency
        # Some features should correlate with effectiveness (but not too strongly)
        correlations = []
        for i in range(features.shape[1]):
            corr = np.corrcoef(features[:, i], effectiveness)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

        if correlations:
            avg_correlation = float(np.mean(correlations))
            # Ideal: some correlation (0.2-0.6) but not too strong
            consistency_score = 1.0 - abs(avg_correlation - 0.4) / 0.4
            sub_metrics["feature_effectiveness_consistency"] = max(
                0.0, consistency_score
            )
        else:
            sub_metrics["feature_effectiveness_consistency"] = 0.5

        # Internal feature consistency (reasonable correlations between features)
        if features.shape[1] > 1:
            feature_corr_matrix = np.corrcoef(features.T)
            # Remove diagonal and get absolute correlations
            mask = ~np.eye(feature_corr_matrix.shape[0], dtype=bool)
            off_diagonal_corrs = np.abs(feature_corr_matrix[mask])

            # Average correlation should be moderate (not too high, not too low)
            avg_feature_corr = (
                np.mean(off_diagonal_corrs) if len(off_diagonal_corrs) > 0 else 0
            )
            # Penalize very high correlations (>0.8) and very low ones (<0.1)
            if avg_feature_corr > 0.8:
                sub_metrics["inter_feature_consistency"] = (
                    1.0 - (avg_feature_corr - 0.8) / 0.2
                )
            elif avg_feature_corr < 0.1:
                sub_metrics["inter_feature_consistency"] = avg_feature_corr / 0.1
            else:
                sub_metrics["inter_feature_consistency"] = 1.0
        else:
            sub_metrics["inter_feature_consistency"] = 1.0

        # Monotonicity checks (some relationships should be monotonic)
        # Sort by effectiveness and check if feature trends are consistent
        sorted_indices = np.argsort(effectiveness)
        sorted_features = features[sorted_indices]

        monotonic_scores = []
        for i in range(features.shape[1]):
            # Check if feature values follow some trend with effectiveness
            trend_correlation, _ = stats.spearmanr(effectiveness, features[:, i])
            if not np.isnan(trend_correlation):
                monotonic_scores.append(abs(trend_correlation))

        if monotonic_scores:
            sub_metrics["monotonicity_consistency"] = np.mean(monotonic_scores)
        else:
            sub_metrics["monotonicity_consistency"] = 0.5

        consistency_score = float(np.mean(list(sub_metrics.values())))

        confidence_interval = self._bootstrap_confidence_interval(
            lambda x: np.mean(x), list(sub_metrics.values())
        )

        return QualityDimension(
            name="consistency",
            score=float(consistency_score),
            weight=self.dimension_weights["consistency"],
            sub_metrics=sub_metrics,
            threshold_met=bool(consistency_score >= 0.6),
            confidence_interval=confidence_interval,
            interpretation=self._interpret_consistency_score(consistency_score),
        )

    def _calculate_weighted_score(self, dimensions: list[QualityDimension]) -> float:
        """Calculate weighted overall quality score"""
        weighted_sum = sum(dim.score * dim.weight for dim in dimensions)
        total_weight = sum(dim.weight for dim in dimensions)
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_confidence_score(self, dimensions: list[QualityDimension]) -> float:
        """Calculate confidence in the overall assessment"""
        # Based on consistency of dimension scores and confidence intervals
        scores = [dim.score for dim in dimensions]
        score_std = float(np.std(scores))

        # Lower standard deviation = higher confidence
        consistency_confidence = max(0.0, 1.0 - score_std)

        # Average confidence interval width
        interval_widths = [
            dim.confidence_interval[1] - dim.confidence_interval[0]
            for dim in dimensions
        ]
        avg_interval_width = float(np.mean(interval_widths))
        interval_confidence = max(0.0, 1.0 - avg_interval_width)

        return (consistency_confidence + interval_confidence) / 2.0

    def _determine_recommendation_tier(self, overall_score: float) -> str:
        """Determine recommendation tier based on overall score"""
        for tier, threshold in self.tier_thresholds.items():
            if overall_score >= threshold:
                return tier
        return "POOR"

    def _bootstrap_confidence_interval(
        self, func, data, n_bootstrap: int = 100
    ) -> tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        if len(data) < 2:
            return (0.0, 1.0)

        bootstrap_samples = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_samples.append(func(sample))

        alpha = 1 - self.confidence_level
        lower = np.percentile(bootstrap_samples, 100 * alpha / 2)
        upper = np.percentile(bootstrap_samples, 100 * (1 - alpha / 2))

        return (float(lower), float(upper))

    # Interpretation methods
    def _interpret_fidelity_score(self, score: float) -> str:
        if score >= 0.8:
            return "Excellent statistical similarity to expected patterns"
        if score >= 0.6:
            return "Good statistical fidelity with minor deviations"
        if score >= 0.4:
            return "Adequate fidelity but noticeable distribution differences"
        return "Poor statistical fidelity, significant deviations detected"

    def _interpret_utility_score(self, score: float) -> str:
        if score >= 0.8:
            return "Excellent ML utility with high predictive potential"
        if score >= 0.6:
            return "Good utility for most ML applications"
        if score >= 0.4:
            return "Adequate for basic ML tasks, may need enhancement"
        return "Limited ML utility, significant improvements needed"

    def _interpret_privacy_score(self, score: float) -> str:
        if score >= 0.9:
            return "Excellent privacy preservation with high uniqueness"
        if score >= 0.8:
            return "Good privacy protection meets standards"
        if score >= 0.6:
            return "Adequate privacy but some concerns about uniqueness"
        return "Privacy concerns detected, review generation parameters"

    def _interpret_validity_score(self, score: float) -> str:
        if score >= 0.95:
            return "Excellent statistical validity, all checks passed"
        if score >= 0.85:
            return "Good validity with minor statistical anomalies"
        if score >= 0.7:
            return "Adequate validity but some statistical concerns"
        return "Statistical validity issues detected, review required"

    def _interpret_diversity_score(self, score: float) -> str:
        if score >= 0.8:
            return "Excellent diversity across domains and feature space"
        if score >= 0.6:
            return "Good diversity with balanced domain representation"
        if score >= 0.4:
            return "Adequate diversity but some imbalances present"
        return "Limited diversity, increase sample variation needed"

    def _interpret_consistency_score(self, score: float) -> str:
        if score >= 0.8:
            return "Excellent internal consistency and logical relationships"
        if score >= 0.6:
            return "Good consistency with expected patterns"
        if score >= 0.4:
            return "Adequate consistency but some irregularities"
        return "Consistency issues detected, review generation logic"

    def generate_quality_report(
        self, metrics: EnhancedQualityMetrics
    ) -> dict[str, Any]:
        """Generate comprehensive quality report"""
        # Convert to serializable format
        report: dict[str, Any] = {
            "assessment_summary": {
                "overall_score": round(metrics.overall_score, 3),
                "confidence_score": round(metrics.confidence_score, 3),
                "recommendation_tier": metrics.recommendation_tier,
                "total_samples": metrics.total_samples,
                "assessment_duration": round(metrics.assessment_duration, 3),
                "timestamp": metrics.assessment_timestamp,
            },
            "dimensional_analysis": {},
            "recommendations": [],
            "quality_trend": self._assess_quality_trend(metrics.overall_score),
            "action_items": self._generate_action_items(metrics),
        }

        # Add each dimension
        dimension_names = [
            "fidelity",
            "utility",
            "privacy",
            "statistical_validity",
            "diversity",
            "consistency",
        ]
        for dim_name in dimension_names:
            dimension = getattr(metrics, dim_name)
            report["dimensional_analysis"][dim_name] = {
                "score": round(dimension.score, 3),
                "weight": dimension.weight,
                "threshold_met": dimension.threshold_met,
                "confidence_interval": [
                    round(x, 3) for x in dimension.confidence_interval
                ],
                "interpretation": dimension.interpretation,
                "sub_metrics": {
                    k: round(v, 3) for k, v in dimension.sub_metrics.items()
                },
            }

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(metrics)

        return report

    def _assess_quality_trend(self, overall_score: float) -> str:
        """Assess quality trend for dashboard display"""
        if overall_score >= 0.85:
            return "IMPROVING"
        if overall_score >= 0.70:
            return "STABLE"
        if overall_score >= 0.55:
            return "DECLINING"
        return "CRITICAL"

    def _generate_action_items(self, metrics: EnhancedQualityMetrics) -> list[str]:
        """Generate specific action items based on assessment"""
        actions = []

        # Check each dimension for specific issues
        if metrics.fidelity.score < 0.6:
            actions.append(
                "Review feature generation algorithms for better statistical fidelity"
            )

        if metrics.utility.score < 0.6:
            actions.append(
                "Increase feature variance and sample size for better ML utility"
            )

        if metrics.privacy.score < 0.8:
            actions.append("Enhance sample uniqueness and inter-sample distances")

        if metrics.statistical_validity.score < 0.9:
            actions.append("Validate data ranges and statistical moments")

        if metrics.diversity.score < 0.7:
            actions.append("Improve domain balance and feature space coverage")

        if metrics.consistency.score < 0.6:
            actions.append("Review feature-effectiveness relationships for consistency")

        if not actions:
            actions.append(
                "Maintain current quality standards with periodic monitoring"
            )

        return actions

    def _generate_recommendations(self, metrics: EnhancedQualityMetrics) -> list[str]:
        """Generate high-level recommendations based on assessment"""
        recommendations = []

        if metrics.recommendation_tier == "EXCELLENT":
            recommendations.append(
                "Quality exceeds standards - consider this as reference for future generations"
            )
            recommendations.append("Monitor for consistency in future batches")

        elif metrics.recommendation_tier == "GOOD":
            recommendations.append("Quality meets production standards")
            recommendations.append(
                "Consider minor optimizations in lower-scoring dimensions"
            )

        elif metrics.recommendation_tier == "ADEQUATE":
            recommendations.append(
                "Quality adequate for basic use but improvements recommended"
            )
            recommendations.append("Focus on dimensions scoring below 0.6")
            recommendations.append(
                "Consider increasing sample size or adjusting generation parameters"
            )

        else:  # POOR
            recommendations.append(
                "Quality below acceptable standards - regeneration recommended"
            )
            recommendations.append("Review generation methodology and parameters")
            recommendations.append("Increase quality controls before production use")

        return recommendations
