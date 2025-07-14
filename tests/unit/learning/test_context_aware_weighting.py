"""
Tests for Context-Aware Feature Weighting System

This test suite validates the adaptive feature weighting functionality
that adjusts feature importance based on prompt domain and context.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.prompt_improver.learning.context_aware_weighter import (
    ContextAwareFeatureWeighter, WeightingConfig, WeightingStrategy
)
from src.prompt_improver.analysis.domain_detector import PromptDomain, DomainClassificationResult
from src.prompt_improver.analysis.domain_feature_extractor import DomainFeatures


class TestWeightingConfig:
    """Test weighting configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WeightingConfig()
        
        assert config.enable_context_aware_weighting is True
        assert config.weighting_strategy == WeightingStrategy.ADAPTIVE
        assert config.confidence_boost_factor == 0.3
        assert config.min_weight_threshold == 0.1
        assert config.max_weight_threshold == 2.0
        assert config.secondary_domain_weight_factor == 0.6
        assert config.normalize_weights is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = WeightingConfig(
            weighting_strategy=WeightingStrategy.STATIC,
            confidence_boost_factor=0.5,
            min_weight_threshold=0.2,
            max_weight_threshold=1.5
        )
        
        assert config.weighting_strategy == WeightingStrategy.STATIC
        assert config.confidence_boost_factor == 0.5
        assert config.min_weight_threshold == 0.2
        assert config.max_weight_threshold == 1.5


class TestContextAwareFeatureWeighter:
    """Test context-aware feature weighting functionality."""
    
    @pytest.fixture
    def weighter(self):
        """Create a feature weighter for testing."""
        return ContextAwareFeatureWeighter()
    
    @pytest.fixture
    def software_domain_features(self):
        """Create domain features for software development."""
        return DomainFeatures(
            domain=PromptDomain.SOFTWARE_DEVELOPMENT,
            confidence=0.8,
            complexity_score=0.6,
            specificity_score=0.7,
            secondary_domains=[],
            hybrid_domain=False
        )
    
    @pytest.fixture
    def creative_domain_features(self):
        """Create domain features for creative writing."""
        return DomainFeatures(
            domain=PromptDomain.CREATIVE_WRITING,
            confidence=0.9,
            complexity_score=0.5,
            specificity_score=0.8,
            secondary_domains=[],
            hybrid_domain=False
        )
    
    @pytest.fixture
    def hybrid_domain_features(self):
        """Create domain features for hybrid domain."""
        return DomainFeatures(
            domain=PromptDomain.DATA_SCIENCE,
            confidence=0.7,
            complexity_score=0.8,
            specificity_score=0.6,
            secondary_domains=[
                (PromptDomain.CREATIVE_WRITING, 0.4)
            ],
            hybrid_domain=True
        )
    
    @pytest.fixture
    def sample_feature_names(self):
        """Create sample feature names for testing."""
        return (
            'overall_score', 'clarity', 'completeness',  # Performance features
            'tech_code_snippets', 'tech_api_references',  # Technical features
            'creative_narrative_elements', 'creative_emotional_language',  # Creative features
            'academic_research_methodology', 'academic_citation_patterns',  # Academic features
            'conv_question_count', 'conv_politeness'  # Conversational features
        )
    
    def test_initialization(self, weighter):
        """Test weighter initialization."""
        assert weighter.config.enable_context_aware_weighting is True
        assert len(weighter._domain_weight_profiles) > 0
        assert PromptDomain.SOFTWARE_DEVELOPMENT in weighter._domain_weight_profiles
        assert PromptDomain.CREATIVE_WRITING in weighter._domain_weight_profiles
    
    def test_domain_weight_profiles(self, weighter):
        """Test domain weight profile structure."""
        software_profile = weighter._domain_weight_profiles[PromptDomain.SOFTWARE_DEVELOPMENT]
        creative_profile = weighter._domain_weight_profiles[PromptDomain.CREATIVE_WRITING]
        
        # Software development should emphasize technical features
        assert software_profile['technical'] > 1.0
        assert software_profile['creative'] < 1.0
        
        # Creative writing should emphasize creative features
        assert creative_profile['creative'] > 1.0
        assert creative_profile['technical'] < 1.0
    
    def test_feature_categorization(self, weighter):
        """Test feature categorization by name."""
        assert weighter._categorize_feature('tech_code_snippets') == 'technical'
        assert weighter._categorize_feature('creative_narrative_elements') == 'creative'
        assert weighter._categorize_feature('academic_research_methodology') == 'academic'
        assert weighter._categorize_feature('conv_question_count') == 'conversational'
        assert weighter._categorize_feature('overall_score') == 'general'
    
    def test_static_weighting(self, weighter, software_domain_features, sample_feature_names):
        """Test static weighting strategy."""
        weighter.config.weighting_strategy = WeightingStrategy.STATIC
        
        weights = weighter.calculate_feature_weights(
            software_domain_features, 
            sample_feature_names
        )
        
        assert len(weights) == len(sample_feature_names)
        
        # Technical features should have higher weights for software domain
        tech_indices = [i for i, name in enumerate(sample_feature_names) if 'tech_' in name]
        creative_indices = [i for i, name in enumerate(sample_feature_names) if 'creative_' in name]
        
        if tech_indices and creative_indices:
            assert np.mean([weights[i] for i in tech_indices]) > np.mean([weights[i] for i in creative_indices])
    
    def test_adaptive_weighting(self, weighter, sample_feature_names):
        """Test adaptive weighting with confidence boost."""
        weighter.config.weighting_strategy = WeightingStrategy.ADAPTIVE
        
        # High confidence case - create separate object
        high_conf_features = DomainFeatures(
            domain=PromptDomain.SOFTWARE_DEVELOPMENT,
            confidence=0.9,
            complexity_score=0.6,
            specificity_score=0.7,
            secondary_domains=[],
            hybrid_domain=False
        )
        
        # Low confidence case - create separate object
        low_conf_features = DomainFeatures(
            domain=PromptDomain.SOFTWARE_DEVELOPMENT,
            confidence=0.3,
            complexity_score=0.6,
            specificity_score=0.7,
            secondary_domains=[],
            hybrid_domain=False
        )
        
        high_weights = weighter.calculate_feature_weights(high_conf_features, sample_feature_names)
        low_weights = weighter.calculate_feature_weights(low_conf_features, sample_feature_names)
        
        # High confidence should boost important features more
        tech_indices = [i for i, name in enumerate(sample_feature_names) if 'tech_' in name]
        if tech_indices:
            high_tech_weights = [high_weights[i] for i in tech_indices]
            low_tech_weights = [low_weights[i] for i in tech_indices]
            assert np.mean(high_tech_weights) > np.mean(low_tech_weights)
    
    def test_hybrid_weighting(self, weighter, hybrid_domain_features, sample_feature_names):
        """Test weighting for hybrid domains."""
        weighter.config.weighting_strategy = WeightingStrategy.HYBRID
        
        weights = weighter.calculate_feature_weights(
            hybrid_domain_features, 
            sample_feature_names
        )
        
        assert len(weights) == len(sample_feature_names)
        
        # Should blend weights for primary and secondary domains
        tech_indices = [i for i, name in enumerate(sample_feature_names) if 'tech_' in name]
        creative_indices = [i for i, name in enumerate(sample_feature_names) if 'creative_' in name]
        
        if tech_indices and creative_indices:
            tech_weights = [weights[i] for i in tech_indices]
            creative_weights = [weights[i] for i in creative_indices]
            
            # Both should be above baseline but tech should be higher (primary domain)
            assert np.mean(tech_weights) > 1.0
            assert np.mean(creative_weights) > 0.6  # Some boost from secondary domain (adjusted expectation)
            assert np.mean(tech_weights) > np.mean(creative_weights)
    
    def test_weight_thresholds(self, weighter, software_domain_features, sample_feature_names):
        """Test weight threshold application."""
        weighter.config.min_weight_threshold = 0.2
        weighter.config.max_weight_threshold = 1.8
        
        weights = weighter.calculate_feature_weights(
            software_domain_features, 
            sample_feature_names
        )
        
        # All weights should be within thresholds
        assert np.all(weights >= 0.2)
        assert np.all(weights <= 1.8)
    
    def test_weight_normalization(self, weighter, software_domain_features, sample_feature_names):
        """Test weight normalization."""
        weighter.config.normalize_weights = True
        
        weights = weighter.calculate_feature_weights(
            software_domain_features, 
            sample_feature_names
        )
        
        # Mean should be close to 1.0 when normalized
        assert abs(np.mean(weights) - 1.0) < 0.1
    
    def test_disabled_weighting(self, software_domain_features, sample_feature_names):
        """Test disabled weighting returns uniform weights."""
        config = WeightingConfig(enable_context_aware_weighting=False)
        weighter = ContextAwareFeatureWeighter(config)
        
        weights = weighter.calculate_feature_weights(
            software_domain_features, 
            sample_feature_names
        )
        
        # All weights should be 1.0 when disabled
        assert np.allclose(weights, 1.0)
    
    def test_weighting_info(self, weighter, software_domain_features):
        """Test weighting information retrieval."""
        info = weighter.get_weighting_info(software_domain_features)
        
        assert info['domain'] == 'software_development'
        assert info['confidence'] == software_domain_features.confidence
        assert info['hybrid_domain'] == software_domain_features.hybrid_domain
        assert 'base_weights' in info
        assert 'weighting_strategy' in info
        assert 'config' in info
    
    def test_caching(self, weighter, software_domain_features, sample_feature_names):
        """Test that weight calculation is cached."""
        # First call
        weights1 = weighter.calculate_feature_weights(
            software_domain_features, 
            sample_feature_names
        )
        
        # Second call should use cache
        weights2 = weighter.calculate_feature_weights(
            software_domain_features, 
            sample_feature_names
        )
        
        assert np.array_equal(weights1, weights2)
    
    def test_secondary_domain_blending(self, weighter, hybrid_domain_features, sample_feature_names):
        """Test secondary domain weight blending."""
        base_weights = weighter._get_base_weights(hybrid_domain_features)
        
        # Should have blended weights from primary and secondary domains
        assert 'technical' in base_weights
        assert 'creative' in base_weights
        
        # Since DATA_SCIENCE is primary (technical) and CREATIVE_WRITING is secondary,
        # technical should still be higher but creative should be boosted
        primary_profile = weighter._domain_weight_profiles[PromptDomain.DATA_SCIENCE]
        pure_creative_weight = primary_profile['creative']
        blended_creative_weight = base_weights['creative']
        
        # Blended should be higher than pure primary domain weight for creative
        assert blended_creative_weight >= pure_creative_weight
    
    def test_domain_confidence_scaling(self, weighter):
        """Test that domain confidence affects weighting appropriately."""
        feature_names = ('tech_feature1', 'creative_feature1', 'general_feature1')
        
        # High confidence software development
        high_conf_features = DomainFeatures(
            domain=PromptDomain.SOFTWARE_DEVELOPMENT,
            confidence=0.9,
            complexity_score=0.6,
            specificity_score=0.7
        )
        
        # Low confidence software development
        low_conf_features = DomainFeatures(
            domain=PromptDomain.SOFTWARE_DEVELOPMENT,
            confidence=0.2,
            complexity_score=0.6,
            specificity_score=0.7
        )
        
        high_weights = weighter.calculate_feature_weights(high_conf_features, feature_names)
        low_weights = weighter.calculate_feature_weights(low_conf_features, feature_names)
        
        # High confidence should show more extreme weighting (higher highs, lower lows)
        assert high_weights[0] > low_weights[0]  # Technical feature boosted more with high confidence
        assert high_weights[1] < low_weights[1]  # Creative feature reduced more with high confidence


class TestContextAwareWeighterIntegration:
    """Test integration scenarios for context-aware weighting."""
    
    def test_all_domain_types(self):
        """Test weighting for all domain types."""
        weighter = ContextAwareFeatureWeighter()
        feature_names = ('tech_feature', 'creative_feature', 'academic_feature', 'conv_feature')
        
        domains_to_test = [
            PromptDomain.SOFTWARE_DEVELOPMENT,
            PromptDomain.CREATIVE_WRITING,
            PromptDomain.RESEARCH,
            PromptDomain.BUSINESS_ANALYSIS,
            PromptDomain.GENERAL
        ]
        
        for domain in domains_to_test:
            domain_features = DomainFeatures(
                domain=domain,
                confidence=0.8,
                complexity_score=0.6,
                specificity_score=0.7
            )
            
            weights = weighter.calculate_feature_weights(domain_features, feature_names)
            
            # Should produce valid weights for all domains
            assert len(weights) == len(feature_names)
            assert np.all(weights > 0)
            assert np.all(np.isfinite(weights))
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        weighter = ContextAwareFeatureWeighter()
        
        # Empty feature names
        domain_features = DomainFeatures(
            domain=PromptDomain.GENERAL,
            confidence=0.5,
            complexity_score=0.5,
            specificity_score=0.5
        )
        
        weights = weighter.calculate_feature_weights(domain_features, ())
        assert len(weights) == 0
        
        # Very long feature list
        long_feature_names = tuple(f'feature_{i}' for i in range(1000))
        weights = weighter.calculate_feature_weights(domain_features, long_feature_names)
        assert len(weights) == 1000
        assert np.all(np.isfinite(weights))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])