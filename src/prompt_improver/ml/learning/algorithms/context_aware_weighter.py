"""Context-Aware Feature Weighting System

This module provides adaptive feature weighting based on prompt domain detection
and context analysis, enabling more targeted ML optimization by emphasizing
relevant features for different prompt types.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from ...analysis.domain_detector import DomainClassificationResult, PromptDomain
    from ...analysis.domain_feature_extractor import DomainFeatures

    DOMAIN_ANALYSIS_AVAILABLE = True
except ImportError:
    DOMAIN_ANALYSIS_AVAILABLE = False

logger = logging.getLogger(__name__)

class WeightingStrategy(Enum):
    """Available feature weighting strategies."""

    static = "static"  # Fixed weights per domain
    ADAPTIVE = "adaptive"  # Confidence-based adjustment
    dynamic = "dynamic"  # Feature importance-based
    HYBRID = "hybrid"  # Combined approach

@dataclass
class WeightingConfig:
    """Configuration for context-aware feature weighting."""

    # Core weighting settings
    enable_context_aware_weighting: bool = True
    weighting_strategy: WeightingStrategy = WeightingStrategy.ADAPTIVE

    # Adaptive weighting parameters
    confidence_boost_factor: float = 0.3
    min_weight_threshold: float = 0.1
    max_weight_threshold: float = 2.0

    # Hybrid domain handling
    secondary_domain_weight_factor: float = 0.6
    hybrid_blend_threshold: float = 0.3

    # Performance settings
    enable_weight_caching: bool = True
    normalize_weights: bool = True

class ContextAwareFeatureWeighter:
    """Adaptive feature weighting system that adjusts feature importance
    based on detected prompt domain and context.
    """

    def __init__(self, config: WeightingConfig | None = None):
        """Initialize the context-aware feature weighter."""
        self.config = config or WeightingConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize domain weight profiles
        self._domain_weight_profiles = self._initialize_weight_profiles()

        # Cache for computed weights
        self._weight_cache = {}

    def _initialize_weight_profiles(self) -> dict[PromptDomain, dict[str, float]]:
        """Initialize domain-specific weight profiles based on research."""
        return {
            # Technical domains - emphasize technical features
            PromptDomain.SOFTWARE_DEVELOPMENT: {
                "technical": 1.5,
                "creative": 0.7,
                "academic": 1.0,
                "conversational": 0.8,
            },
            PromptDomain.DATA_SCIENCE: {
                "technical": 1.4,
                "creative": 0.6,
                "academic": 1.2,
                "conversational": 0.9,
            },
            PromptDomain.AI_ML: {
                "technical": 1.6,
                "creative": 0.6,
                "academic": 1.3,
                "conversational": 0.8,
            },
            PromptDomain.WEB_DEVELOPMENT: {
                "technical": 1.3,
                "creative": 1.1,
                "academic": 0.9,
                "conversational": 1.0,
            },
            PromptDomain.SYSTEM_ADMIN: {
                "technical": 1.5,
                "creative": 0.5,
                "academic": 0.8,
                "conversational": 0.9,
            },
            PromptDomain.API_DOCUMENTATION: {
                "technical": 1.4,
                "creative": 0.8,
                "academic": 1.1,
                "conversational": 0.9,
            },
            # Creative domains - emphasize creative features
            PromptDomain.CREATIVE_WRITING: {
                "technical": 0.6,
                "creative": 1.6,
                "academic": 0.9,
                "conversational": 1.2,
            },
            PromptDomain.CONTENT_CREATION: {
                "technical": 0.7,
                "creative": 1.4,
                "academic": 0.9,
                "conversational": 1.3,
            },
            PromptDomain.MARKETING: {
                "technical": 0.8,
                "creative": 1.3,
                "academic": 1.0,
                "conversational": 1.4,
            },
            PromptDomain.STORYTELLING: {
                "technical": 0.5,
                "creative": 1.7,
                "academic": 0.8,
                "conversational": 1.3,
            },
            # Academic domains - emphasize academic features
            PromptDomain.RESEARCH: {
                "technical": 1.0,
                "creative": 0.8,
                "academic": 1.5,
                "conversational": 0.9,
            },
            PromptDomain.EDUCATION: {
                "technical": 0.9,
                "creative": 1.1,
                "academic": 1.4,
                "conversational": 1.2,
            },
            PromptDomain.ACADEMIC_WRITING: {
                "technical": 0.8,
                "creative": 0.9,
                "academic": 1.6,
                "conversational": 0.8,
            },
            PromptDomain.SCIENTIFIC: {
                "technical": 1.2,
                "creative": 0.7,
                "academic": 1.5,
                "conversational": 0.8,
            },
            # Business domains - balanced with conversational emphasis
            PromptDomain.BUSINESS_ANALYSIS: {
                "technical": 1.1,
                "creative": 1.0,
                "academic": 1.2,
                "conversational": 1.1,
            },
            PromptDomain.PROJECT_MANAGEMENT: {
                "technical": 1.0,
                "creative": 1.0,
                "academic": 1.0,
                "conversational": 1.3,
            },
            PromptDomain.CUSTOMER_SERVICE: {
                "technical": 0.8,
                "creative": 1.1,
                "academic": 0.9,
                "conversational": 1.5,
            },
            PromptDomain.SALES: {
                "technical": 0.7,
                "creative": 1.3,
                "academic": 0.9,
                "conversational": 1.4,
            },
            # Specialized domains
            PromptDomain.MEDICAL: {
                "technical": 1.1,
                "creative": 0.8,
                "academic": 1.4,
                "conversational": 1.0,
            },
            PromptDomain.LEGAL: {
                "technical": 0.9,
                "creative": 0.8,
                "academic": 1.4,
                "conversational": 1.1,
            },
            PromptDomain.HEALTHCARE: {
                "technical": 1.0,
                "creative": 0.9,
                "academic": 1.3,
                "conversational": 1.2,
            },
            # General domains - neutral weighting
            PromptDomain.CONVERSATIONAL: {
                "technical": 0.8,
                "creative": 1.1,
                "academic": 0.9,
                "conversational": 1.4,
            },
            PromptDomain.INSTRUCTIONAL: {
                "technical": 1.0,
                "creative": 1.0,
                "academic": 1.1,
                "conversational": 1.2,
            },
            PromptDomain.ANALYTICAL: {
                "technical": 1.2,
                "creative": 0.8,
                "academic": 1.2,
                "conversational": 1.0,
            },
            PromptDomain.GENERAL: {
                "technical": 1.0,
                "creative": 1.0,
                "academic": 1.0,
                "conversational": 1.0,
            },
        }

    def calculate_feature_weights(
        self, domain_features: DomainFeatures, feature_names: tuple[str]
    ) -> np.ndarray:
        """Calculate context-aware weights for feature vector.

        Args:
            domain_features: Domain classification and features
            feature_names: Names of features in the vector

        Returns:
            Numpy array of weights for each feature
        """
        if not self.config.enable_context_aware_weighting:
            return np.ones(len(feature_names))

        # Check cache first if enabled
        if self.config.enable_weight_caching:
            cache_key = self._create_cache_key(domain_features, feature_names)
            if cache_key in self._weight_cache:
                return self._weight_cache[cache_key]

        # Get base weights from domain profile
        base_weights = self._get_base_weights(domain_features)

        # Apply strategy-specific adjustments
        if self.config.weighting_strategy == WeightingStrategy.static:
            weights = self._apply_static_weights(base_weights, feature_names)
        elif self.config.weighting_strategy == WeightingStrategy.ADAPTIVE:
            weights = self._apply_adaptive_weights(
                base_weights, domain_features, feature_names
            )
        elif self.config.weighting_strategy == WeightingStrategy.dynamic:
            weights = self._apply_dynamic_weights(
                base_weights, domain_features, feature_names
            )
        else:  # HYBRID
            weights = self._apply_hybrid_weights(
                base_weights, domain_features, feature_names
            )

        # Apply thresholds and normalization
        weights = self._post_process_weights(weights)

        self.logger.debug(
            f"Calculated weights for {domain_features.domain.value}: "
            f"mean={np.mean(weights):.3f}, std={np.std(weights):.3f}"
        )

        # Cache the result if enabled
        if self.config.enable_weight_caching:
            cache_key = self._create_cache_key(domain_features, feature_names)
            self._weight_cache[cache_key] = weights

        return weights

    def _get_base_weights(self, domain_features: DomainFeatures) -> dict[str, float]:
        """Get base weight profile for the primary domain."""
        primary_profile = self._domain_weight_profiles.get(
            domain_features.domain, self._domain_weight_profiles[PromptDomain.GENERAL]
        )

        # Handle hybrid domains by blending profiles
        if (
            domain_features.hybrid_domain
            and domain_features.secondary_domains
            and len(domain_features.secondary_domains) > 0
        ):
            return self._blend_domain_profiles(
                primary_profile,
                domain_features.secondary_domains,
                domain_features.confidence,
            )

        return primary_profile.copy()

    def _blend_domain_profiles(
        self,
        primary_profile: dict[str, float],
        secondary_domains: list[tuple[PromptDomain, float]],
        primary_confidence: float,
    ) -> dict[str, float]:
        """Blend weight profiles for hybrid domains."""
        blended_profile = primary_profile.copy()

        for secondary_domain, secondary_confidence in secondary_domains:
            if secondary_confidence < self.config.hybrid_blend_threshold:
                continue

            secondary_profile = self._domain_weight_profiles.get(
                secondary_domain, self._domain_weight_profiles[PromptDomain.GENERAL]
            )

            # Calculate blend ratio
            blend_factor = (
                secondary_confidence * self.config.secondary_domain_weight_factor
            )

            # Blend profiles
            for category in blended_profile:
                if category in secondary_profile:
                    blended_profile[category] = (
                        blended_profile[category] * (1 - blend_factor)
                        + secondary_profile[category] * blend_factor
                    )

        return blended_profile

    def _apply_static_weights(
        self, base_weights: dict[str, float], feature_names: tuple[str]
    ) -> np.ndarray:
        """Apply static domain-based weights."""
        weights = np.ones(len(feature_names))

        for i, feature_name in enumerate(feature_names):
            # Determine feature category from name
            category = self._categorize_feature(feature_name)
            if category in base_weights:
                weights[i] = base_weights[category]

        return weights

    def _apply_adaptive_weights(
        self,
        base_weights: dict[str, float],
        domain_features: DomainFeatures,
        feature_names: tuple[str],
    ) -> np.ndarray:
        """Apply adaptive weights based on domain confidence."""
        weights = self._apply_static_weights(base_weights, feature_names)

        # Apply confidence-based boost
        confidence_boost = (
            domain_features.confidence * self.config.confidence_boost_factor
        )

        for i, feature_name in enumerate(feature_names):
            category = self._categorize_feature(feature_name)
            if category in base_weights and base_weights[category] > 1.0:
                # Boost important features more with higher confidence
                weights[i] *= 1 + confidence_boost
            elif category in base_weights and base_weights[category] < 1.0:
                # Reduce less important features more with higher confidence
                weights[i] *= 1 - confidence_boost * 0.5

        return weights

    def _apply_dynamic_weights(
        self,
        base_weights: dict[str, float],
        domain_features: DomainFeatures,
        feature_names: tuple[str],
    ) -> np.ndarray:
        """Apply dynamic weights based on feature importance (placeholder for future enhancement)."""
        # For now, use adaptive weighting as base
        # Future enhancement: integrate with ensemble feature importance
        return self._apply_adaptive_weights(
            base_weights, domain_features, feature_names
        )

    def _apply_hybrid_weights(
        self,
        base_weights: dict[str, float],
        domain_features: DomainFeatures,
        feature_names: tuple[str],
    ) -> np.ndarray:
        """Apply hybrid weighting combining multiple strategies."""
        # Combine static and adaptive approaches
        static_weights = self._apply_static_weights(base_weights, feature_names)
        adaptive_weights = self._apply_adaptive_weights(
            base_weights, domain_features, feature_names
        )

        # Blend based on confidence
        blend_factor = min(domain_features.confidence, 0.8)
        weights = static_weights * (1 - blend_factor) + adaptive_weights * blend_factor

        return weights

    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize a feature by name to determine appropriate weight category."""
        feature_name_lower = feature_name.lower()

        if any(
            prefix in feature_name_lower
            for prefix in ["tech_", "technical_", "code_", "api_", "algorithm"]
        ):
            return "technical"
        if any(
            prefix in feature_name_lower
            for prefix in ["creative_", "narrative_", "emotional_", "sensory_"]
        ):
            return "creative"
        if any(
            prefix in feature_name_lower
            for prefix in ["academic_", "research_", "citation_", "formal_"]
        ):
            return "academic"
        if any(
            prefix in feature_name_lower
            for prefix in ["conv_", "conversational_", "question_", "polite_"]
        ):
            return "conversational"
        return "general"

    def _post_process_weights(self, weights: np.ndarray) -> np.ndarray:
        """Apply thresholds and normalization to weights."""
        # Apply thresholds
        weights = np.clip(
            weights, self.config.min_weight_threshold, self.config.max_weight_threshold
        )

        # Normalize if requested
        if self.config.normalize_weights:
            mean_weight = np.mean(weights)
            if mean_weight > 0:
                weights = weights / mean_weight

        return weights

    def _create_cache_key(
        self, domain_features: DomainFeatures, feature_names: tuple[str]
    ) -> str:
        """Create a cache key from domain features and feature names."""
        # Create a deterministic string representation for caching
        key_parts = [
            domain_features.domain.value,
            f"conf_{domain_features.confidence:.3f}",
            f"comp_{domain_features.complexity_score:.3f}",
            f"spec_{domain_features.specificity_score:.3f}",
            f"hybrid_{domain_features.hybrid_domain}",
            f"strategy_{self.config.weighting_strategy.value}",
            f"boost_{self.config.confidence_boost_factor}",
            str(hash(feature_names)),  # Hash the feature names tuple
        ]

        # Add secondary domains if present
        if (
            hasattr(domain_features, "secondary_domains")
            and domain_features.secondary_domains
        ):
            secondary_parts = []
            for domain, conf in domain_features.secondary_domains:
                secondary_parts.append(f"{domain.value}:{conf:.3f}")
            key_parts.append(f"secondary_{'_'.join(secondary_parts)}")

        return "_".join(key_parts)

    def get_weighting_info(self, domain_features: DomainFeatures) -> dict[str, Any]:
        """Get information about the weighting configuration for a domain."""
        base_weights = self._get_base_weights(domain_features)

        return {
            "domain": domain_features.domain.value,
            "confidence": domain_features.confidence,
            "hybrid_domain": domain_features.hybrid_domain,
            "secondary_domains": [
                (d.value, c) for d, c in domain_features.secondary_domains
            ],
            "base_weights": base_weights,
            "weighting_strategy": self.config.weighting_strategy.value,
            "config": {
                "confidence_boost_factor": self.config.confidence_boost_factor,
                "min_weight_threshold": self.config.min_weight_threshold,
                "max_weight_threshold": self.config.max_weight_threshold,
            },
        }
