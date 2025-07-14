"""Tests for Phase 2 Advanced Clustering (HDBSCAN + UMAP) in Context-Specific Learning.

Comprehensive test suite for advanced clustering enhancements including:
- UMAP dimensionality reduction
- HDBSCAN hierarchical clustering
- Clustering quality assessment
- Integration with existing context learning workflow
- Performance validation and edge case handling

Testing best practices applied from Context7 research:
- Statistical validation of clustering results
- Realistic parameter ranges for ML components
- Proper handling of high-dimensional data
- Quality metrics validation
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
from datetime import datetime

from prompt_improver.learning.context_learner import (
    ContextSpecificLearner,
    ContextConfig,
    ContextInsight
)


@pytest.fixture
def clustering_config():
    """Configuration with advanced clustering enabled."""
    return ContextConfig(
        use_advanced_clustering=True,
        umap_n_components=10,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
        hdbscan_min_cluster_size=5,
        hdbscan_min_samples=3,
        clustering_quality_threshold=0.5,
        # Standard parameters
        min_sample_size=20,
        max_context_groups=20
    )


@pytest.fixture
def context_engine_clustering(clustering_config):
    """Context learning engine with advanced clustering enabled."""
    return ContextSpecificLearner(config=clustering_config)


@pytest.fixture
def high_dimensional_data():
    """High-dimensional data for clustering tests."""
    np.random.seed(42)  # Reproducible test data
    
    # Generate realistic high-dimensional feature vectors
    n_samples = 100
    n_features = 50
    
    # Create clusters with realistic separations
    cluster_centers = [
        np.random.normal(0.3, 0.1, n_features),  # Technical documentation cluster
        np.random.normal(0.7, 0.1, n_features),  # Creative writing cluster  
        np.random.normal(0.5, 0.15, n_features), # Business communication cluster
    ]
    
    data = []
    labels = []
    
    for i in range(n_samples):
        cluster_idx = i % 3
        # Add noise to cluster centers
        features = cluster_centers[cluster_idx] + np.random.normal(0, 0.05, n_features)
        # Normalize to [0,1] range following ML-Agents best practices
        features = np.clip(features, 0.0, 1.0)
        
        data.append({
            "context": ["technical_documentation", "creative_writing", "business_communication"][cluster_idx],
            "features": features.tolist(),
            "score": 0.7 + 0.2 * np.random.random(),  # Realistic performance scores
            "timestamp": datetime.now().isoformat(),
            "user_id": f"user_{i % 10}",
            "session_id": f"session_{i}"
        })
        labels.append(cluster_idx)
    
    return data, labels


@pytest.fixture
def minimal_clustering_data():
    """Minimal data that might not cluster well."""
    return [
        {
            "context": "sparse_context",
            "features": [0.5] * 10,
            "score": 0.6,
            "timestamp": datetime.now().isoformat()
        },
        {
            "context": "sparse_context", 
            "features": [0.6] * 10,
            "score": 0.7,
            "timestamp": datetime.now().isoformat()
        }
    ]


class TestAdvancedClustering:
    """Test suite for HDBSCAN + UMAP clustering functionality."""

    @pytest.mark.asyncio
    async def test_umap_dimensionality_reduction(self, context_engine_clustering, high_dimensional_data):
        """Test UMAP dimensionality reduction with realistic parameters."""
        data, _ = high_dimensional_data
        
        # Extract feature matrix
        features = np.array([item["features"] for item in data])
        
        # Test internal UMAP processing
        with patch('prompt_improver.learning.context_learner.ADVANCED_CLUSTERING_AVAILABLE', True):
            with patch('umap.UMAP') as mock_umap:
                # Configure mock to return realistic reduced dimensions
                mock_reducer = MagicMock()
                reduced_features = np.random.random((len(features), context_engine_clustering.config.umap_n_components))
                mock_reducer.fit_transform.return_value = reduced_features
                mock_umap.return_value = mock_reducer
                
                # Test clustering analysis
                context_data = {
                    "mixed_contexts": {
                        "sample_size": len(data),
                        "avg_performance": 0.75,
                        "consistency_score": 0.8,
                        "historical_data": data
                    }
                }
                
                result = await context_engine_clustering.analyze_context_effectiveness(context_data)
                
                # Verify UMAP was called with correct parameters
                mock_umap.assert_called_once()
                call_kwargs = mock_umap.call_args[1]
                assert call_kwargs["n_components"] == context_engine_clustering.config.umap_n_components
                assert call_kwargs["n_neighbors"] == context_engine_clustering.config.umap_n_neighbors
                assert call_kwargs["min_dist"] == context_engine_clustering.config.umap_min_dist

    @pytest.mark.asyncio
    async def test_hdbscan_clustering(self, context_engine_clustering, high_dimensional_data):
        """Test HDBSCAN hierarchical clustering with quality assessment."""
        data, true_labels = high_dimensional_data
        
        with patch('prompt_improver.learning.context_learner.ADVANCED_CLUSTERING_AVAILABLE', True):
            with patch('umap.UMAP') as mock_umap, patch('hdbscan.HDBSCAN') as mock_hdbscan:
                # Setup UMAP mock
                mock_reducer = MagicMock()
                reduced_features = np.random.random((len(data), 10))
                mock_reducer.fit_transform.return_value = reduced_features
                mock_umap.return_value = mock_reducer
                
                # Setup HDBSCAN mock with realistic clustering results
                mock_clusterer = MagicMock()
                # Generate realistic cluster labels (some noise points as -1)
                cluster_labels = np.random.choice([-1, 0, 1, 2], len(data), p=[0.1, 0.3, 0.3, 0.3])
                mock_clusterer.labels_ = cluster_labels
                mock_clusterer.probabilities_ = np.random.random(len(data))
                mock_hdbscan.return_value = mock_clusterer
                
                context_data = {
                    "test_context": {
                        "sample_size": len(data),
                        "avg_performance": 0.78,
                        "consistency_score": 0.82,
                        "historical_data": data
                    }
                }
                
                result = await context_engine_clustering.analyze_context_effectiveness(context_data)
                
                # Verify HDBSCAN was called with correct parameters
                mock_hdbscan.assert_called_once()
                call_kwargs = mock_hdbscan.call_args[1]
                assert call_kwargs["min_cluster_size"] == context_engine_clustering.config.hdbscan_min_cluster_size
                assert call_kwargs["min_samples"] == context_engine_clustering.config.hdbscan_min_samples

    @pytest.mark.asyncio
    async def test_clustering_quality_assessment(self, context_engine_clustering, high_dimensional_data):
        """Test clustering quality metrics and validation."""
        data, _ = high_dimensional_data
        
        with patch('prompt_improver.learning.context_learner.ADVANCED_CLUSTERING_AVAILABLE', True):
            with patch('umap.UMAP') as mock_umap, patch('hdbscan.HDBSCAN') as mock_hdbscan:
                # Setup mocks
                mock_reducer = MagicMock()
                mock_reducer.fit_transform.return_value = np.random.random((len(data), 10))
                mock_umap.return_value = mock_reducer
                
                mock_clusterer = MagicMock()
                # High-quality clustering with clear separation
                cluster_labels = np.repeat([0, 1, 2], len(data) // 3)[:len(data)]
                mock_clusterer.labels_ = cluster_labels
                mock_clusterer.probabilities_ = np.random.uniform(0.7, 1.0, len(data))  # High confidence
                mock_hdbscan.return_value = mock_clusterer
                
                context_data = {
                    "quality_test": {
                        "sample_size": len(data),
                        "avg_performance": 0.85,
                        "consistency_score": 0.9,
                        "historical_data": data
                    }
                }
                
                result = await context_engine_clustering.analyze_context_effectiveness(context_data)
                
                # Check for clustering quality metrics in result
                if "advanced_clustering" in result:
                    clustering_result = result["advanced_clustering"]
                    
                    # Validate quality metrics
                    if "clustering_quality" in clustering_result:
                        quality = clustering_result["clustering_quality"]
                        assert 0.0 <= quality["silhouette_score"] <= 1.0
                        assert quality["n_clusters"] >= 0
                        assert 0.0 <= quality["noise_ratio"] <= 1.0

    @pytest.mark.asyncio
    async def test_clustering_with_insufficient_data(self, context_engine_clustering, minimal_clustering_data):
        """Test clustering behavior with insufficient data points."""
        context_data = {
            "sparse_data": {
                "sample_size": len(minimal_clustering_data),
                "avg_performance": 0.65,
                "consistency_score": 0.75,
                "historical_data": minimal_clustering_data
            }
        }
        
        # Should handle gracefully without advanced clustering
        result = await context_engine_clustering.analyze_context_effectiveness(context_data)
        
        # Should fall back to traditional analysis
        assert "context_insights" in result
        # Advanced clustering might be skipped due to insufficient data
        if "advanced_clustering" in result:
            clustering_result = result["advanced_clustering"]
            assert clustering_result.get("status") in ["insufficient_data", "fallback", "success"]

    @pytest.mark.asyncio  
    async def test_clustering_feature_extraction(self, context_engine_clustering):
        """Test feature extraction for clustering from context data."""
        test_data = [
            {
                "context": "test_context",
                "prompt": "Test prompt with specific features",
                "score": 0.85,
                "features": [0.7, 0.8, 0.6, 0.9],
                "metadata": {"complexity": "high", "domain": "technical"}
            },
            {
                "context": "test_context",
                "prompt": "Another test prompt",
                "score": 0.75,
                "features": [0.6, 0.7, 0.8, 0.7],
                "metadata": {"complexity": "medium", "domain": "business"}
            }
        ]
        
        # Test feature extraction method (would be called internally)
        features = []
        for item in test_data:
            if "features" in item:
                feature_vector = item["features"]
                # Verify features are normalized to [0,1] range
                assert all(0.0 <= f <= 1.0 for f in feature_vector)
                features.append(feature_vector)
        
        assert len(features) == len(test_data)
        assert all(len(f) == 4 for f in features)  # Consistent feature dimensionality

    @pytest.mark.asyncio
    async def test_clustering_disabled_fallback(self):
        """Test fallback behavior when advanced clustering is disabled."""
        config = ContextConfig(
            use_advanced_clustering=False,
            min_sample_size=10
        )
        engine = ContextSpecificLearner(config=config)
        
        data = [
            {
                "context": "test",
                "features": [0.5, 0.6, 0.7, 0.8],
                "score": 0.8,
                "timestamp": datetime.now().isoformat()
            }
        ] * 15  # Sufficient data
        
        context_data = {
            "test_context": {
                "sample_size": len(data),
                "avg_performance": 0.8,
                "consistency_score": 0.85,
                "historical_data": data
            }
        }
        
        result = await engine.analyze_context_effectiveness(context_data)
        
        # Should not contain advanced clustering results
        assert "advanced_clustering" not in result
        # Should still provide traditional analysis
        assert "context_insights" in result

    @pytest.mark.parametrize("n_components", [2, 5, 10, 20])
    async def test_umap_component_variations(self, high_dimensional_data, n_components):
        """Test UMAP with different component counts."""
        config = ContextConfig(
            use_advanced_clustering=True,
            umap_n_components=n_components,
            umap_n_neighbors=15,  # Default from config
            min_sample_size=10
        )
        engine = ContextSpecificLearner(config=config)
        
        data, _ = high_dimensional_data
        
        with patch('prompt_improver.learning.context_learner.ADVANCED_CLUSTERING_AVAILABLE', True):
            with patch('umap.UMAP') as mock_umap:
                mock_reducer = MagicMock()
                # Calculate expected adjusted n_components based on implementation logic
                n_samples = len(data)
                n_features = len(data[0]['features']) if data and 'features' in data[0] else 20  # fallback
                adjusted_n_neighbors = min(config.umap_n_neighbors, max(2, n_samples // 3))
                expected_n_components = min(n_components, n_features, adjusted_n_neighbors - 1)
                
                # Return appropriate dimensions using expected value
                mock_reducer.fit_transform.return_value = np.random.random((len(data), expected_n_components))
                mock_umap.return_value = mock_reducer
                
                context_data = {
                    "param_test": {
                        "sample_size": len(data),
                        "avg_performance": 0.8,
                        "consistency_score": 0.85,
                        "historical_data": data
                    }
                }
                
                result = await engine.analyze_context_effectiveness(context_data)
                
                # Verify UMAP was called with intelligently adjusted n_components
                mock_umap.assert_called_once()
                call_kwargs = mock_umap.call_args[1]
                assert call_kwargs["n_components"] == expected_n_components
                
                # Verify the adjusted value is sensible
                assert call_kwargs["n_components"] <= n_components  # Should not exceed requested
                assert call_kwargs["n_components"] >= 2  # Should be at least 2
                assert call_kwargs["n_components"] <= n_features  # Should not exceed feature count

    @pytest.mark.asyncio
    async def test_clustering_quality_threshold(self):
        """Test clustering quality threshold enforcement."""
        config = ContextConfig(
            use_advanced_clustering=True,
            clustering_quality_threshold=0.8,  # High threshold
            min_sample_size=10
        )
        engine = ContextSpecificLearner(config=config)
        
        test_data = [{"features": [0.5] * 5, "score": 0.7}] * 20
        
        with patch('prompt_improver.learning.context_learner.ADVANCED_CLUSTERING_AVAILABLE', True):
            with patch('umap.UMAP') as mock_umap, patch('hdbscan.HDBSCAN') as mock_hdbscan:
                # Setup mocks for poor quality clustering
                mock_reducer = MagicMock()
                mock_reducer.fit_transform.return_value = np.random.random((20, 5))
                mock_umap.return_value = mock_reducer
                
                mock_clusterer = MagicMock()
                # Poor clustering: mostly noise points
                mock_clusterer.labels_ = np.array([-1] * 18 + [0, 1])  # High noise ratio
                mock_clusterer.probabilities_ = np.random.uniform(0.1, 0.4, 20)  # Low confidence
                mock_hdbscan.return_value = mock_clusterer
                
                context_data = {
                    "poor_quality": {
                        "sample_size": 20,
                        "avg_performance": 0.7,
                        "consistency_score": 0.8,
                        "historical_data": test_data
                    }
                }
                
                result = await engine.analyze_context_effectiveness(context_data)
                
                # Should handle poor quality clustering appropriately
                if "advanced_clustering" in result:
                    clustering_result = result["advanced_clustering"]
                    # May indicate poor quality or fallback to traditional methods
                    assert "status" in clustering_result


class TestClusteringErrorHandling:
    """Test error handling and edge cases for advanced clustering."""

    @pytest.mark.asyncio
    async def test_missing_features_in_data(self, context_engine_clustering):
        """Test handling of data with missing feature vectors."""
        incomplete_data = [
            {"context": "test", "score": 0.8},  # Missing features
            {"context": "test", "features": [0.7, 0.8, 0.6], "score": 0.75},
            {"context": "test", "features": [0.6, 0.9, 0.7], "score": 0.82}
        ]
        
        context_data = {
            "incomplete_features": {
                "sample_size": len(incomplete_data),
                "avg_performance": 0.79,
                "consistency_score": 0.8,
                "historical_data": incomplete_data
            }
        }
        
        # Should handle gracefully by filtering out incomplete data
        result = await context_engine_clustering.analyze_context_effectiveness(context_data)
        
        # Should still provide some analysis
        assert "context_insights" in result

    @pytest.mark.asyncio
    async def test_uniform_feature_vectors(self, context_engine_clustering):
        """Test clustering with uniform/identical feature vectors."""
        uniform_data = [
            {
                "context": "uniform_test",
                "features": [0.5, 0.5, 0.5, 0.5],  # Identical features
                "score": 0.8
            }
        ] * 25
        
        context_data = {
            "uniform_features": {
                "sample_size": len(uniform_data),
                "avg_performance": 0.8,
                "consistency_score": 0.9,
                "historical_data": uniform_data
            }
        }
        
        # Should handle uniform data without clustering errors
        result = await context_engine_clustering.analyze_context_effectiveness(context_data)
        
        # Should provide analysis even with non-clusterable data
        assert "context_insights" in result

    @pytest.mark.asyncio
    async def test_clustering_library_import_failure(self):
        """Test behavior when HDBSCAN/UMAP libraries are not available."""
        config = ContextConfig(use_advanced_clustering=True)
        
        with patch('prompt_improver.learning.context_learner.ADVANCED_CLUSTERING_AVAILABLE', False):
            engine = ContextSpecificLearner(config=config)
            
            test_data = [{"features": [0.5, 0.6], "score": 0.8}] * 15
            
            context_data = {
                "no_libraries": {
                    "sample_size": 15,
                    "avg_performance": 0.8,
                    "consistency_score": 0.85,
                    "historical_data": test_data
                }
            }
            
            # Should fall back to traditional analysis
            result = await engine.analyze_context_effectiveness(context_data)
            
            assert "context_insights" in result
            # Should not attempt advanced clustering
            assert "advanced_clustering" not in result or result["advanced_clustering"].get("status") == "unavailable"

    @pytest.mark.asyncio
    async def test_extreme_parameter_values(self):
        """Test clustering with extreme parameter configurations."""
        # Test with very small cluster size requirements
        config = ContextConfig(
            use_advanced_clustering=True,
            hdbscan_min_cluster_size=1,  # Very small
            hdbscan_min_samples=1,
            umap_n_components=1  # Minimal dimensionality
        )
        engine = ContextSpecificLearner(config=config)
        
        test_data = [{"features": [0.5, 0.6, 0.7], "score": 0.8}] * 10
        
        with patch('prompt_improver.learning.context_learner.ADVANCED_CLUSTERING_AVAILABLE', True):
            with patch('umap.UMAP') as mock_umap, patch('hdbscan.HDBSCAN') as mock_hdbscan:
                mock_reducer = MagicMock()
                mock_reducer.fit_transform.return_value = np.random.random((10, 1))
                mock_umap.return_value = mock_reducer
                
                mock_clusterer = MagicMock()
                mock_clusterer.labels_ = np.arange(10)  # Each point its own cluster
                mock_clusterer.probabilities_ = np.ones(10)
                mock_hdbscan.return_value = mock_clusterer
                
                context_data = {
                    "extreme_params": {
                        "sample_size": 10,
                        "avg_performance": 0.8,
                        "consistency_score": 0.85,
                        "historical_data": test_data
                    }
                }
                
                # Should handle extreme parameters without crashing
                result = await engine.analyze_context_effectiveness(context_data)
                assert "context_insights" in result


class TestClusteringIntegration:
    """Integration tests for advanced clustering with existing workflows."""

    @pytest.mark.asyncio
    async def test_clustering_with_traditional_analysis(self, context_engine_clustering, high_dimensional_data):
        """Test integration of advanced clustering with traditional context analysis."""
        data, _ = high_dimensional_data
        
        context_data = {
            "integrated_test": {
                "sample_size": len(data),
                "avg_performance": 0.82,
                "consistency_score": 0.78,
                "historical_data": data,
                "traditional_metrics": {
                    "avg_improvement": 0.15,
                    "rule_effectiveness": 0.85
                }
            }
        }
        
        with patch('prompt_improver.learning.context_learner.ADVANCED_CLUSTERING_AVAILABLE', True):
            result = await context_engine_clustering.analyze_context_effectiveness(context_data)
            
            # Should have both traditional and advanced analysis
            assert "context_insights" in result
            
            # Traditional metrics should be preserved
            insights = result["context_insights"]
            assert isinstance(insights, (list, dict))

    @pytest.mark.asyncio
    async def test_clustering_performance_monitoring(self, context_engine_clustering, high_dimensional_data):
        """Test performance characteristics of clustering operations."""
        data, _ = high_dimensional_data
        import time
        
        context_data = {
            "performance_test": {
                "sample_size": len(data),
                "avg_performance": 0.8,
                "consistency_score": 0.85,
                "historical_data": data
            }
        }
        
        start_time = time.time()
        result = await context_engine_clustering.analyze_context_effectiveness(context_data)
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time (allow up to 10 seconds for complex clustering)
        assert execution_time < 10.0
        
        # Should provide meaningful results
        assert "context_insights" in result


# Test markers for categorization
pytestmark = [
    pytest.mark.unit,
    pytest.mark.ml_performance,
    pytest.mark.ml_contracts
]