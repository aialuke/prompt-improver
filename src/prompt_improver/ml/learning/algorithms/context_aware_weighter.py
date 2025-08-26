"""Context-Aware Feature Weighting System

This module provides adaptive feature weighting based on prompt domain detection
and context analysis, enabling more targeted ML optimization by emphasizing
relevant features for different prompt types.
"""
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
# import numpy as np  # Converted to lazy loading
try:
    from ...analysis.domain_detector import DomainClassificationResult, PromptDomain
    from ...analysis.domain_feature_extractor import DomainFeatures
    DOMAIN_ANALYSIS_AVAILABLE = True
except ImportError:
    DOMAIN_ANALYSIS_AVAILABLE = False

try:
    from ....services.cache.cache_factory import CacheFactory
    from prompt_improver.core.utils.lazy_ml_loader import get_numpy
    CACHE_FACTORY_AVAILABLE = True
except ImportError:
    CACHE_FACTORY_AVAILABLE = False

logger = logging.getLogger(__name__)

class WeightingStrategy(Enum):
    """Available feature weighting strategies."""
    static = 'static'
    ADAPTIVE = 'adaptive'
    dynamic = 'dynamic'
    HYBRID = 'hybrid'

@dataclass
class WeightingConfig:
    """Configuration for context-aware feature weighting."""
    enable_context_aware_weighting: bool = True
    weighting_strategy: WeightingStrategy = WeightingStrategy.ADAPTIVE
    confidence_boost_factor: float = 0.3
    min_weight_threshold: float = 0.1
    max_weight_threshold: float = 2.0
    secondary_domain_weight_factor: float = 0.6
    hybrid_blend_threshold: float = 0.3
    enable_weight_caching: bool = True
    normalize_weights: bool = True

class ContextAwareFeatureWeighter:
    """Adaptive feature weighting system that adjusts feature importance
    based on detected prompt domain and context.
    """

    def __init__(self, config: WeightingConfig | None=None):
        """Initialize the context-aware feature weighter."""
        self.config = config or WeightingConfig()
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self._domain_weight_profiles = self._initialize_weight_profiles()
        
        # Initialize optimized ML analysis cache for feature weights
        if CACHE_FACTORY_AVAILABLE and self.config.enable_weight_caching:
            try:
                self._cache = CacheFactory.get_ml_analysis_cache()
                self._use_unified_cache = True
                self.logger.info("Initialized optimized ML analysis cache for feature weights (target: <10μs cache hits)")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ML analysis cache, falling back to dict: {e}")
                self._weight_cache = {}
                self._use_unified_cache = False
        else:
            self._weight_cache = {}
            self._use_unified_cache = False
            if not CACHE_FACTORY_AVAILABLE:
                self.logger.warning("CacheFactory not available, using dictionary cache")

    def _serialize_weights(self, weights: get_numpy().ndarray) -> dict[str, Any]:
        """Serialize numpy array weights for cache storage.
        
        Args:
            weights: Numpy array of feature weights
            
        Returns:
            Serializable dictionary representation
        """
        return {
            "weights": weights.tolist(),
            "dtype": str(weights.dtype),
            "shape": weights.shape,
            "type": "numpy_array"
        }
    
    def _deserialize_weights(self, data: dict[str, Any]) -> get_numpy().ndarray:
        """Deserialize cached weights back to numpy array.
        
        Args:
            data: Serialized weight data
            
        Returns:
            Reconstructed numpy array
        """
        if isinstance(data, dict) and data.get("type") == "numpy_array":
            weights_list = data["weights"]
            dtype = data.get("dtype", "float64")
            shape = data.get("shape")
            
            weights = get_numpy().array(weights_list, dtype=dtype)
            if shape and weights.shape != tuple(shape):
                weights = weights.reshape(shape)
            return weights
        else:
            # Handle legacy or direct numpy array storage
            return get_numpy().array(data) if not isinstance(data, get_numpy().ndarray) else data
    
    async def _get_cached_weights(self, cache_key: str) -> Optional[get_numpy().ndarray]:
        """Get weights from cache with proper deserialization.
        
        Args:
            cache_key: Cache key for weights
            
        Returns:
            Cached weights array or None if not found
        """
        if self._use_unified_cache:
            try:
                cached_data = await self._cache.get(cache_key)
                if cached_data is not None:
                    return self._deserialize_weights(cached_data)
            except Exception as e:
                self.logger.warning(f"Failed to get cached weights for key {cache_key}: {e}")
        else:
            # Fallback to dictionary cache
            return self._weight_cache.get(cache_key)
        
        return None
    
    async def _set_cached_weights(self, cache_key: str, weights: get_numpy().ndarray) -> None:
        """Set weights in cache with proper serialization.
        
        Args:
            cache_key: Cache key for weights
            weights: Weights array to cache
        """
        if self._use_unified_cache:
            try:
                serialized_weights = self._serialize_weights(weights)
                await self._cache.set(
                    cache_key, 
                    serialized_weights, 
                    l2_ttl=3600,  # 1 hour TTL for feature weights
                    l1_ttl=900    # 15 minute L1 TTL for hot weights
                )
            except Exception as e:
                self.logger.warning(f"Failed to cache weights for key {cache_key}: {e}")
                # Fallback to dictionary cache
                self._weight_cache[cache_key] = weights
        else:
            # Dictionary cache (synchronous)
            self._weight_cache[cache_key] = weights

    def _initialize_weight_profiles(self) -> dict[PromptDomain, dict[str, float]]:
        """Initialize domain-specific weight profiles based on research."""
        return {PromptDomain.SOFTWARE_DEVELOPMENT: {'technical': 1.5, 'creative': 0.7, 'academic': 1.0, 'conversational': 0.8}, PromptDomain.DATA_SCIENCE: {'technical': 1.4, 'creative': 0.6, 'academic': 1.2, 'conversational': 0.9}, PromptDomain.AI_ML: {'technical': 1.6, 'creative': 0.6, 'academic': 1.3, 'conversational': 0.8}, PromptDomain.WEB_DEVELOPMENT: {'technical': 1.3, 'creative': 1.1, 'academic': 0.9, 'conversational': 1.0}, PromptDomain.SYSTEM_ADMIN: {'technical': 1.5, 'creative': 0.5, 'academic': 0.8, 'conversational': 0.9}, PromptDomain.API_DOCUMENTATION: {'technical': 1.4, 'creative': 0.8, 'academic': 1.1, 'conversational': 0.9}, PromptDomain.CREATIVE_WRITING: {'technical': 0.6, 'creative': 1.6, 'academic': 0.9, 'conversational': 1.2}, PromptDomain.CONTENT_CREATION: {'technical': 0.7, 'creative': 1.4, 'academic': 0.9, 'conversational': 1.3}, PromptDomain.MARKETING: {'technical': 0.8, 'creative': 1.3, 'academic': 1.0, 'conversational': 1.4}, PromptDomain.STORYTELLING: {'technical': 0.5, 'creative': 1.7, 'academic': 0.8, 'conversational': 1.3}, PromptDomain.RESEARCH: {'technical': 1.0, 'creative': 0.8, 'academic': 1.5, 'conversational': 0.9}, PromptDomain.EDUCATION: {'technical': 0.9, 'creative': 1.1, 'academic': 1.4, 'conversational': 1.2}, PromptDomain.ACADEMIC_WRITING: {'technical': 0.8, 'creative': 0.9, 'academic': 1.6, 'conversational': 0.8}, PromptDomain.SCIENTIFIC: {'technical': 1.2, 'creative': 0.7, 'academic': 1.5, 'conversational': 0.8}, PromptDomain.BUSINESS_ANALYSIS: {'technical': 1.1, 'creative': 1.0, 'academic': 1.2, 'conversational': 1.1}, PromptDomain.PROJECT_MANAGEMENT: {'technical': 1.0, 'creative': 1.0, 'academic': 1.0, 'conversational': 1.3}, PromptDomain.CUSTOMER_SERVICE: {'technical': 0.8, 'creative': 1.1, 'academic': 0.9, 'conversational': 1.5}, PromptDomain.SALES: {'technical': 0.7, 'creative': 1.3, 'academic': 0.9, 'conversational': 1.4}, PromptDomain.MEDICAL: {'technical': 1.1, 'creative': 0.8, 'academic': 1.4, 'conversational': 1.0}, PromptDomain.LEGAL: {'technical': 0.9, 'creative': 0.8, 'academic': 1.4, 'conversational': 1.1}, PromptDomain.HEALTHCARE: {'technical': 1.0, 'creative': 0.9, 'academic': 1.3, 'conversational': 1.2}, PromptDomain.CONVERSATIONAL: {'technical': 0.8, 'creative': 1.1, 'academic': 0.9, 'conversational': 1.4}, PromptDomain.INSTRUCTIONAL: {'technical': 1.0, 'creative': 1.0, 'academic': 1.1, 'conversational': 1.2}, PromptDomain.ANALYTICAL: {'technical': 1.2, 'creative': 0.8, 'academic': 1.2, 'conversational': 1.0}, PromptDomain.GENERAL: {'technical': 1.0, 'creative': 1.0, 'academic': 1.0, 'conversational': 1.0}}

    async def calculate_feature_weights(self, domain_features: DomainFeatures, feature_names: tuple[str]) -> get_numpy().ndarray:
        """Calculate context-aware weights for feature vector.

        Args:
            domain_features: Domain classification and features
            feature_names: Names of features in the vector

        Returns:
            Numpy array of weights for each feature
        """
        if not self.config.enable_context_aware_weighting:
            return get_numpy().ones(len(feature_names))
            
        # Check cache first if enabled
        if self.config.enable_weight_caching:
            cache_key = self._create_cache_key(domain_features, feature_names)
            cached_weights = await self._get_cached_weights(cache_key)
            if cached_weights is not None:
                return cached_weights
        
        # Compute weights
        base_weights = self._get_base_weights(domain_features)
        if self.config.weighting_strategy == WeightingStrategy.static:
            weights = self._apply_static_weights(base_weights, feature_names)
        elif self.config.weighting_strategy == WeightingStrategy.ADAPTIVE:
            weights = self._apply_adaptive_weights(base_weights, domain_features, feature_names)
        elif self.config.weighting_strategy == WeightingStrategy.dynamic:
            weights = self._apply_dynamic_weights(base_weights, domain_features, feature_names)
        else:
            weights = self._apply_hybrid_weights(base_weights, domain_features, feature_names)
            
        weights = self._post_process_weights(weights)
        
        self.logger.debug(
            'Calculated weights for %s: mean=%s, std=%s', 
            domain_features.domain.value, 
            format(get_numpy().mean(weights), '.3f'), 
            format(get_numpy().std(weights), '.3f')
        )
        
        # Cache the computed weights
        if self.config.enable_weight_caching:
            cache_key = self._create_cache_key(domain_features, feature_names)
            await self._set_cached_weights(cache_key, weights)
            
        return weights
    
    def calculate_feature_weights_sync(self, domain_features: DomainFeatures, feature_names: tuple[str]) -> get_numpy().ndarray:
        """Synchronous version for backward compatibility.
        
        Note: This version won't use unified cache and falls back to dictionary cache.
        For optimal performance, use the async version.
        """
        if not self.config.enable_context_aware_weighting:
            return get_numpy().ones(len(feature_names))
            
        # Use dictionary cache for sync version
        if self.config.enable_weight_caching and not self._use_unified_cache:
            cache_key = self._create_cache_key(domain_features, feature_names)
            if cache_key in self._weight_cache:
                return self._weight_cache[cache_key]
        
        # Compute weights
        base_weights = self._get_base_weights(domain_features)
        if self.config.weighting_strategy == WeightingStrategy.static:
            weights = self._apply_static_weights(base_weights, feature_names)
        elif self.config.weighting_strategy == WeightingStrategy.ADAPTIVE:
            weights = self._apply_adaptive_weights(base_weights, domain_features, feature_names)
        elif self.config.weighting_strategy == WeightingStrategy.dynamic:
            weights = self._apply_dynamic_weights(base_weights, domain_features, feature_names)
        else:
            weights = self._apply_hybrid_weights(base_weights, domain_features, feature_names)
            
        weights = self._post_process_weights(weights)
        
        self.logger.debug(
            'Calculated weights for %s: mean=%s, std=%s', 
            domain_features.domain.value, 
            format(get_numpy().mean(weights), '.3f'), 
            format(get_numpy().std(weights), '.3f')
        )
        
        # Cache using dictionary (sync)
        if self.config.enable_weight_caching and not self._use_unified_cache:
            cache_key = self._create_cache_key(domain_features, feature_names)
            self._weight_cache[cache_key] = weights
            
        return weights

    def _get_base_weights(self, domain_features: DomainFeatures) -> dict[str, float]:
        """Get base weight profile for the primary domain."""
        primary_profile = self._domain_weight_profiles.get(domain_features.domain, self._domain_weight_profiles[PromptDomain.GENERAL])
        if domain_features.hybrid_domain and domain_features.secondary_domains and (len(domain_features.secondary_domains) > 0):
            return self._blend_domain_profiles(primary_profile, domain_features.secondary_domains, domain_features.confidence)
        return primary_profile.copy()

    def _blend_domain_profiles(self, primary_profile: dict[str, float], secondary_domains: list[tuple[PromptDomain, float]], primary_confidence: float) -> dict[str, float]:
        """Blend weight profiles for hybrid domains."""
        blended_profile = primary_profile.copy()
        for secondary_domain, secondary_confidence in secondary_domains:
            if secondary_confidence < self.config.hybrid_blend_threshold:
                continue
            secondary_profile = self._domain_weight_profiles.get(secondary_domain, self._domain_weight_profiles[PromptDomain.GENERAL])
            blend_factor = secondary_confidence * self.config.secondary_domain_weight_factor
            for category in blended_profile:
                if category in secondary_profile:
                    blended_profile[category] = blended_profile[category] * (1 - blend_factor) + secondary_profile[category] * blend_factor
        return blended_profile

    def _apply_static_weights(self, base_weights: dict[str, float], feature_names: tuple[str]) -> get_numpy().ndarray:
        """Apply static domain-based weights."""
        weights = get_numpy().ones(len(feature_names))
        for i, feature_name in enumerate(feature_names):
            category = self._categorize_feature(feature_name)
            if category in base_weights:
                weights[i] = base_weights[category]
        return weights

    def _apply_adaptive_weights(self, base_weights: dict[str, float], domain_features: DomainFeatures, feature_names: tuple[str]) -> get_numpy().ndarray:
        """Apply adaptive weights based on domain confidence."""
        weights = self._apply_static_weights(base_weights, feature_names)
        confidence_boost = domain_features.confidence * self.config.confidence_boost_factor
        for i, feature_name in enumerate(feature_names):
            category = self._categorize_feature(feature_name)
            if category in base_weights and base_weights[category] > 1.0:
                weights[i] *= 1 + confidence_boost
            elif category in base_weights and base_weights[category] < 1.0:
                weights[i] *= 1 - confidence_boost * 0.5
        return weights

    def _apply_dynamic_weights(self, base_weights: dict[str, float], domain_features: DomainFeatures, feature_names: tuple[str]) -> get_numpy().ndarray:
        """Apply dynamic weights based on feature importance (placeholder for future enhancement)."""
        return self._apply_adaptive_weights(base_weights, domain_features, feature_names)

    def _apply_hybrid_weights(self, base_weights: dict[str, float], domain_features: DomainFeatures, feature_names: tuple[str]) -> get_numpy().ndarray:
        """Apply hybrid weighting combining multiple strategies."""
        static_weights = self._apply_static_weights(base_weights, feature_names)
        adaptive_weights = self._apply_adaptive_weights(base_weights, domain_features, feature_names)
        blend_factor = min(domain_features.confidence, 0.8)
        weights = static_weights * (1 - blend_factor) + adaptive_weights * blend_factor
        return weights

    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize a feature by name to determine appropriate weight category."""
        feature_name_lower = feature_name.lower()
        if any(prefix in feature_name_lower for prefix in ['tech_', 'technical_', 'code_', 'api_', 'algorithm']):
            return 'technical'
        if any(prefix in feature_name_lower for prefix in ['creative_', 'narrative_', 'emotional_', 'sensory_']):
            return 'creative'
        if any(prefix in feature_name_lower for prefix in ['academic_', 'research_', 'citation_', 'formal_']):
            return 'academic'
        if any(prefix in feature_name_lower for prefix in ['conv_', 'conversational_', 'question_', 'polite_']):
            return 'conversational'
        return 'general'

    def _post_process_weights(self, weights: get_numpy().ndarray) -> get_numpy().ndarray:
        """Apply thresholds and normalization to weights."""
        weights = get_numpy().clip(weights, self.config.min_weight_threshold, self.config.max_weight_threshold)
        if self.config.normalize_weights:
            mean_weight = get_numpy().mean(weights)
            if mean_weight > 0:
                weights = weights / mean_weight
        return weights

    def _create_cache_key(self, domain_features: DomainFeatures, feature_names: tuple[str]) -> str:
        """Create a cache key from domain features and feature names.
        
        Cache key format: weight:{domain}:{confidence}:{strategy}:{features_hash}
        This provides hierarchical namespacing for ML feature weight caches.
        """
        key_parts = [
            "weight",  # Namespace prefix for weight caches
            domain_features.domain.value,
            f'conf_{domain_features.confidence:.3f}',
            f'comp_{domain_features.complexity_score:.3f}',
            f'spec_{domain_features.specificity_score:.3f}',
            f'hybrid_{domain_features.hybrid_domain}',
            f'strategy_{self.config.weighting_strategy.value}',
            f'boost_{self.config.confidence_boost_factor}',
            str(hash(feature_names))
        ]
        
        if hasattr(domain_features, 'secondary_domains') and domain_features.secondary_domains:
            secondary_parts = []
            for domain, conf in domain_features.secondary_domains:
                secondary_parts.append(f'{domain.value}:{conf:.3f}')
            key_parts.append(f"secondary_{'_'.join(secondary_parts)}")
            
        return ':'.join(key_parts)

    def get_weighting_info(self, domain_features: DomainFeatures) -> dict[str, Any]:
        """Get information about the weighting configuration for a domain."""
        base_weights = self._get_base_weights(domain_features)
        
        info = {
            'domain': domain_features.domain.value, 
            'confidence': domain_features.confidence, 
            'hybrid_domain': domain_features.hybrid_domain, 
            'secondary_domains': [(d.value, c) for d, c in domain_features.secondary_domains], 
            'base_weights': base_weights, 
            'weighting_strategy': self.config.weighting_strategy.value, 
            'config': {
                'confidence_boost_factor': self.config.confidence_boost_factor, 
                'min_weight_threshold': self.config.min_weight_threshold, 
                'max_weight_threshold': self.config.max_weight_threshold
            },
            'cache_info': {
                'unified_cache_enabled': self._use_unified_cache,
                'cache_factory_available': CACHE_FACTORY_AVAILABLE,
                'weight_caching_enabled': self.config.enable_weight_caching,
                'performance_target': '<10μs cache hits'
            }
        }
        
        return info
    
    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics.
        
        Returns:
            Cache performance and usage statistics
        """
        if self._use_unified_cache:
            try:
                cache_stats = self._cache.get_performance_stats()
                return {
                    "unified_cache": cache_stats,
                    "cache_type": "unified",
                    "dict_cache_size": 0  # Not used when unified cache is active
                }
            except Exception as e:
                self.logger.warning(f"Failed to get unified cache stats: {e}")
        
        # Fallback stats for dictionary cache
        return {
            "unified_cache": None,
            "cache_type": "dictionary",
            "dict_cache_size": len(self._weight_cache)
        }
        
    async def clear_cache(self) -> None:
        """Clear all cached feature weights."""
        if self._use_unified_cache:
            try:
                # Clear weight-related cache entries using pattern
                cleared_count = await self._cache.invalidate_pattern("weight:*")
                self.logger.info(f"Cleared {cleared_count} weight cache entries from unified cache")
            except Exception as e:
                self.logger.warning(f"Failed to clear unified cache: {e}")
        
        # Always clear dictionary cache as fallback
        self._weight_cache.clear()
        self.logger.info("Cleared dictionary weight cache")

    async def close(self) -> None:
        """Clean up cache resources."""
        if self._use_unified_cache:
            try:
                await self._cache.close()
                self.logger.info("Closed unified cache connection")
            except Exception as e:
                self.logger.warning(f"Error closing unified cache: {e}")
        
        # Clear dictionary cache
        self._weight_cache.clear()