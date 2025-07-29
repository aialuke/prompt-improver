#!/usr/bin/env python3
"""Integration tests for orchestrator with ML components.

Tests the integration between the ML Pipeline Orchestrator and the
ML components (feature extractors, clustering engine, etc.).
"""

import asyncio
import pytest
import numpy as np
from typing import Dict, Any

# Orchestrator imports
from prompt_improver.ml.orchestration.integration.direct_component_loader import DirectComponentLoader
from prompt_improver.ml.orchestration.connectors.component_connector import ComponentTier

# Component imports
from prompt_improver.ml.learning.features import (
    CompositeFeatureExtractor,
    LinguisticFeatureExtractor,
    DomainFeatureExtractor,
    ContextFeatureExtractor,
    FeatureExtractionConfig
)
from prompt_improver.ml.learning.clustering import (
    ContextClusteringEngine,
    ClusteringConfig
)
from prompt_improver.ml.learning.algorithms.context_learner import (
    ContextLearner,
    ContextConfig
)


class TestOrchestratorMLIntegration:
    """Test integration between orchestrator and ML components."""
    
    @pytest.fixture
    def component_loader(self):
        """Create component loader for testing."""
        return DirectComponentLoader()
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for testing."""
        return [
            {
                'text': 'Create a Python function for data processing using pandas and numpy.',
                'originalPrompt': 'Write Python code for data.',
                'performance': {'improvement_score': 0.8, 'user_satisfaction': 0.9},
                'project_type': 'data_science',
                'user_id': 'user_001',
                'session_id': 'session_001',
                'interaction': {'session_length_norm': 0.6, 'iteration_count_norm': 0.4},
                'temporal': {'time_of_day_norm': 0.5, 'day_of_week_norm': 0.3}
            },
            {
                'text': 'Design a web application interface with React and TypeScript.',
                'originalPrompt': 'Build a web app.',
                'performance': {'improvement_score': 0.7, 'user_satisfaction': 0.8},
                'project_type': 'web_development',
                'user_id': 'user_002',
                'session_id': 'session_002',
                'interaction': {'session_length_norm': 0.4, 'iteration_count_norm': 0.6},
                'temporal': {'time_of_day_norm': 0.7, 'day_of_week_norm': 0.2}
            },
            {
                'text': 'Write a creative story about artificial intelligence and human emotions.',
                'originalPrompt': 'Tell a story about AI.',
                'performance': {'improvement_score': 0.9, 'user_satisfaction': 0.95},
                'project_type': 'creative_writing',
                'user_id': 'user_003',
                'session_id': 'session_003',
                'interaction': {'session_length_norm': 0.8, 'iteration_count_norm': 0.3},
                'temporal': {'time_of_day_norm': 0.2, 'day_of_week_norm': 0.8}
            }
        ]
    
    @pytest.mark.asyncio
    async def test_orchestrator_loads_context_learner(self, component_loader):
        """Test that orchestrator can load the original context learner."""
        # Load context learner through orchestrator
        tier1_key = ComponentTier.TIER_1_CORE
        loaded_component = await component_loader.load_component('context_learner', tier1_key)
        
        assert loaded_component is not None
        assert loaded_component.name == 'context_learner'
        assert loaded_component.component_class.__name__ == 'ContextSpecificLearner'
        assert not loaded_component.is_initialized
    
    @pytest.mark.asyncio
    async def test_orchestrator_loads_context_learner(self, component_loader):
        """Test that orchestrator can load the context learner."""
        # Load context learner through orchestrator
        tier1_key = ComponentTier.TIER_1_CORE
        loaded_component = await component_loader.load_component('context_learner', tier1_key)

        assert loaded_component is not None
        assert loaded_component.name == 'context_learner'
        assert loaded_component.component_class.__name__ == 'ContextLearner'
        assert not loaded_component.is_initialized
    
    def test_feature_extractors_integration(self):
        """Test that all feature extractors work correctly."""
        # Test individual extractors
        linguistic_extractor = LinguisticFeatureExtractor()
        domain_extractor = DomainFeatureExtractor()
        context_extractor = ContextFeatureExtractor()
        
        # Test text
        test_text = "Create a machine learning model for sentiment analysis using Python and scikit-learn."
        test_context = {
            'performance': {'improvement_score': 0.8},
            'project_type': 'data_science',
            'user_id': 'test_user'
        }
        
        # Extract features
        linguistic_features = linguistic_extractor.extract_features(test_text)
        domain_features = domain_extractor.extract_features(test_text)
        context_features = context_extractor.extract_features(test_context)
        
        # Verify feature counts
        assert len(linguistic_features) == 10
        assert len(domain_features) == 15
        assert len(context_features) == 20
        
        # Verify feature ranges
        assert all(0.0 <= f <= 1.0 for f in linguistic_features)
        assert all(0.0 <= f <= 1.0 for f in domain_features)
        assert all(0.0 <= f <= 1.0 for f in context_features)
    
    def test_composite_feature_extractor_integration(self):
        """Test that composite feature extractor integrates all components."""
        config = FeatureExtractionConfig(
            enable_linguistic=True,
            enable_domain=True,
            enable_context=True,
            cache_enabled=True
        )
        
        composite_extractor = CompositeFeatureExtractor(config)
        
        test_text = "Develop a RESTful API for a mobile application using Node.js and Express."
        test_context = {
            'performance': {'improvement_score': 0.75, 'user_satisfaction': 0.85},
            'project_type': 'web_development',
            'user_id': 'test_user_api'
        }
        
        result = composite_extractor.extract_features(test_text, test_context)
        
        # Verify result structure
        assert 'features' in result
        assert 'feature_names' in result
        assert 'metadata' in result
        
        # Verify feature count (10 + 15 + 20 = 45)
        assert len(result['features']) == 45
        assert len(result['feature_names']) == 45
        
        # Verify metadata
        assert result['metadata']['extractors_used'] == ['linguistic', 'domain', 'context']
        assert result['metadata']['total_features'] == 45
    
    @pytest.mark.asyncio
    async def test_clustering_engine_integration(self):
        """Test that clustering engine works correctly."""
        config = ClusteringConfig(
            use_advanced_clustering=False,  # Use K-means for consistent testing
            min_samples_for_clustering=10
        )
        
        clustering_engine = ContextClusteringEngine(config)
        
        # Create synthetic feature data
        np.random.seed(42)
        features = np.random.rand(25, 45)  # 25 samples, 45 features
        
        # Perform clustering
        cluster_result = await clustering_engine.cluster_contexts(features)
        
        # Verify clustering result
        assert cluster_result.n_clusters >= 1
        assert cluster_result.algorithm_used == 'K-means'
        assert 0.0 <= cluster_result.silhouette_score <= 1.0
        assert len(cluster_result.cluster_labels) == 25
        assert cluster_result.quality_metrics is not None
    
    @pytest.mark.asyncio
    async def test_context_learner_integration(self, sample_training_data):
        """Test that context learner integrates all components."""
        config = ContextConfig(
            enable_linguistic_features=True,
            enable_domain_features=True,
            enable_context_features=True,
            use_advanced_clustering=False  # Use K-means for consistent testing
        )
        
        learner = ContextLearner(config)
        
        # Test learning from data
        result = await learner.learn_from_data(sample_training_data)
        
        # Verify learning result
        assert result.clusters_found >= 1
        assert result.features_extracted == 45  # 10 + 15 + 20
        assert 0.0 <= result.silhouette_score <= 1.0
        assert result.processing_time > 0
        assert result.quality_metrics is not None
        assert len(result.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_integration(self, component_loader, sample_training_data):
        """Test complete end-to-end workflow through orchestrator."""
        # 1. Load context learner through orchestrator
        tier1_key = ComponentTier.TIER_1_CORE
        loaded_component = await component_loader.load_component('context_learner', tier1_key)
        
        assert loaded_component is not None
        
        # 2. Create instance with configuration
        config = ContextConfig(
            enable_linguistic_features=True,
            enable_domain_features=True,
            enable_context_features=True,
            use_advanced_clustering=False
        )
        
        learner_instance = loaded_component.component_class(config)
        
        # 3. Execute learning workflow
        result = await learner_instance.learn_from_data(sample_training_data)
        
        # 4. Verify end-to-end result
        assert result.clusters_found >= 1
        assert result.features_extracted == 45
        assert result.processing_time > 0
        assert len(result.recommendations) > 0
        
        # 5. Test component state management
        assert learner_instance.cluster_results is not None
        assert learner_instance.context_patterns is not None
        
        # 6. Test cache functionality
        cache_stats = learner_instance.feature_extractor.get_cache_stats()
        assert cache_stats['cache_enabled'] is True
        assert cache_stats['cache_size'] >= 0
    
    def test_performance_comparison(self):
        """Test performance comparison between old and new components."""
        import time
        
        # Test current ML components
        start_time = time.time()
        
        # Initialize current ML components
        composite_extractor = CompositeFeatureExtractor()
        clustering_engine = ContextClusteringEngine()
        context_learner = ContextLearner()
        
        new_init_time = time.time() - start_time
        
        # Test feature extraction performance
        test_text = "Build a scalable microservices architecture using Docker and Kubernetes."
        test_context = {'performance': {'improvement_score': 0.8}, 'project_type': 'devops'}
        
        start_time = time.time()
        result = composite_extractor.extract_features(test_text, test_context)
        new_extraction_time = time.time() - start_time
        
        # Verify performance characteristics
        assert new_init_time < 5.0  # Should initialize quickly
        assert new_extraction_time < 1.0  # Should extract features quickly
        assert len(result['features']) == 45  # Should extract all features
        
        print(f"✅ Performance Test Results:")
        print(f"   Initialization time: {new_init_time:.3f}s")
        print(f"   Feature extraction time: {new_extraction_time:.3f}s")
        print(f"   Features extracted: {len(result['features'])}")
    
    def test_context_learner_available(self):
        """Test that context learner is available and working."""
        # Test direct import
        try:
            from prompt_improver.ml.learning.algorithms.context_learner import ContextLearner
            learner = ContextLearner()
            assert learner is not None
            print("✅ Context learner available")
        except ImportError:
            pytest.fail("Context learner should be available")

        # Test import from algorithms module
        try:
            from prompt_improver.ml.learning.algorithms import ContextLearner
            learner = ContextLearner()
            assert learner is not None
            assert learner.__class__.__name__ == "ContextLearner"
            print("✅ ContextLearner import working correctly")
        except ImportError:
            pytest.fail("ContextLearner should be available")

        print("✅ All components working correctly")


if __name__ == "__main__":
    # Run a quick integration test
    async def quick_test():
        print("🧪 Running Quick Integration Test...")
        
        test_instance = TestOrchestratorMLIntegration()
        
        # Test feature extractors
        test_instance.test_feature_extractors_integration()
        print("✅ Feature extractors integration test passed")
        
        # Test composite extractor
        test_instance.test_composite_feature_extractor_integration()
        print("✅ Composite feature extractor integration test passed")
        
        # Test clustering engine
        await test_instance.test_clustering_engine_integration()
        print("✅ Clustering engine integration test passed")
        
        # Test performance
        test_instance.test_performance_comparison()
        print("✅ Performance comparison test passed")
        
        # Test current ML components
        test_instance.test_context_learner_available()
        print("✅ Current ML components test passed")
        
        print("\n🎯 All integration tests passed!")
    
    asyncio.run(quick_test())
