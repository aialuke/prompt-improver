"""Integration tests for decomposed ML services.

Tests the new service-based architecture to ensure all functionality
is preserved and performance is maintained after decomposition.
"""

import asyncio
import pytest
import numpy as np
from unittest.mock import Mock

from src.prompt_improver.ml.dimensionality.services.dimensionality_reducer_facade import (
    DimensionalityReducerFacade,
    DimensionalityReducerFacadeFactory
)
from src.prompt_improver.ml.clustering.services.clustering_optimizer_facade import (
    ClusteringOptimizerFacade,
    ClusteringOptimizerFacadeFactory
)


class TestDimensionalityReductionServices:
    """Test dimensionality reduction service decomposition."""

    @pytest.fixture
    def sample_data(self):
        """Create sample high-dimensional data for testing."""
        np.random.seed(42)
        n_samples, n_features = 100, 50
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 3, n_samples)
        return X, y

    @pytest.fixture
    def reducer_facade(self):
        """Create dimensionality reducer facade."""
        return DimensionalityReducerFacadeFactory.create_fast_reducer(target_dimensions=10)

    def test_facade_initialization(self, reducer_facade):
        """Test that facade initializes correctly."""
        assert reducer_facade.target_dimensions == 10
        assert len(reducer_facade.available_methods) > 0
        assert 'pca' in reducer_facade.available_methods

    @pytest.mark.asyncio
    async def test_dimensionality_reduction_basic(self, reducer_facade, sample_data):
        """Test basic dimensionality reduction functionality."""
        X, y = sample_data
        
        result = await reducer_facade.reduce_dimensions(X, y, method='pca')
        
        assert result.original_dimensions == 50
        assert result.reduced_dimensions == 10
        assert result.transformed_data.shape == (100, 10)
        assert result.method == 'pca'
        assert result.quality_score >= 0.0
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_auto_method_selection(self, sample_data):
        """Test automatic method selection."""
        X, y = sample_data
        reducer = DimensionalityReducerFacadeFactory.create_high_quality_reducer(target_dimensions=5)
        
        result = await reducer.reduce_dimensions(X, y)
        
        assert result.reduced_dimensions == 5
        assert result.transformed_data.shape == (100, 5)
        assert result.method in reducer.available_methods

    def test_method_recommendation(self, reducer_facade, sample_data):
        """Test method recommendation functionality."""
        X, y = sample_data
        
        recommendation = reducer_facade.recommend_method(X, y)
        
        assert 'recommended_method' in recommendation
        assert 'suitable_methods' in recommendation
        assert 'data_characteristics' in recommendation
        assert recommendation['recommended_method'] in recommendation['suitable_methods']

    def test_performance_tracking(self, reducer_facade):
        """Test performance tracking functionality."""
        summary = reducer_facade.get_performance_summary()
        
        # Should show no data initially
        assert summary['status'] == 'no_data'

    @pytest.mark.asyncio
    async def test_multiple_methods(self, reducer_facade, sample_data):
        """Test multiple reduction methods work."""
        X, y = sample_data
        
        methods_to_test = ['pca', 'incremental_pca']
        results = {}
        
        for method in methods_to_test:
            if method in reducer_facade.available_methods:
                result = await reducer_facade.reduce_dimensions(X, y, method=method)
                results[method] = result
        
        assert len(results) > 0
        for method, result in results.items():
            assert result.method == method
            assert result.transformed_data.shape[1] == 10


class TestClusteringServices:
    """Test clustering service decomposition."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for clustering."""
        np.random.seed(42)
        n_samples, n_features = 150, 20
        
        # Create some clustered data
        cluster_centers = np.random.randn(3, n_features) * 2
        X = []
        y = []
        
        for i, center in enumerate(cluster_centers):
            cluster_data = np.random.randn(50, n_features) * 0.5 + center
            X.extend(cluster_data)
            y.extend([i] * 50)
        
        return np.array(X), np.array(y)

    @pytest.fixture
    def clustering_facade(self):
        """Create clustering optimizer facade."""
        return ClusteringOptimizerFacadeFactory.create_fast_optimizer()

    def test_facade_initialization(self, clustering_facade):
        """Test that facade initializes correctly."""
        assert clustering_facade.algorithm == 'hdbscan'
        assert clustering_facade.preprocessor is not None
        assert clustering_facade.evaluator is not None

    @pytest.mark.asyncio
    async def test_clustering_basic(self, clustering_facade, sample_data):
        """Test basic clustering functionality."""
        X, y = sample_data
        
        result = await clustering_facade.optimize_clustering(X, y)
        
        assert result['status'] in ['success', 'success_with_warnings', 'low_quality']
        assert 'labels' in result
        assert 'quality_metrics' in result
        assert 'optimization_time' in result
        assert len(result['labels']) == len(X)

    @pytest.mark.asyncio
    async def test_orchestrated_interface(self, clustering_facade, sample_data):
        """Test orchestrator-compatible interface."""
        X, y = sample_data
        
        config = {
            'features': X.tolist(),
            'labels': y.tolist(),
            'optimization_target': 'silhouette',
            'max_clusters': 10
        }
        
        result = await clustering_facade.run_orchestrated_analysis(config)
        
        assert result['orchestrator_compatible'] is True
        assert 'component_result' in result
        assert 'local_metadata' in result
        assert 'clustering_summary' in result['component_result']

    @pytest.mark.asyncio
    async def test_context_clustering_interface(self, clustering_facade, sample_data):
        """Test backward-compatible context clustering interface."""
        X, y = sample_data
        
        result = await clustering_facade.cluster_contexts(X)
        
        assert hasattr(result, 'cluster_labels')
        assert hasattr(result, 'n_clusters')
        assert hasattr(result, 'silhouette_score')
        assert hasattr(result, 'algorithm_used')
        assert len(result.cluster_labels) == len(X)

    def test_performance_tracking(self, clustering_facade):
        """Test performance tracking functionality."""
        summary = clustering_facade.get_performance_summary()
        
        # Should show no data initially
        assert summary['status'] == 'no_data'

    def test_algorithm_info(self, clustering_facade):
        """Test algorithm information retrieval."""
        info = clustering_facade.get_algorithm_info()
        
        assert 'algorithm' in info
        assert 'availability' in info


class TestServiceIntegration:
    """Test integration between decomposed services."""

    @pytest.fixture
    def integrated_data(self):
        """Create data that needs both dimensionality reduction and clustering."""
        np.random.seed(42)
        n_samples, n_features = 200, 100
        
        # High-dimensional data with some structure
        X = np.random.randn(n_samples, n_features)
        # Add some correlated features to create structure
        X[:, :10] = X[:, :10] + np.random.randn(n_samples, 10) * 0.1
        
        return X

    @pytest.mark.asyncio
    async def test_reduction_then_clustering_pipeline(self, integrated_data):
        """Test pipeline: dimensionality reduction followed by clustering."""
        X = integrated_data
        
        # Step 1: Dimensionality reduction
        reducer = DimensionalityReducerFacadeFactory.create_fast_reducer(target_dimensions=15)
        reduction_result = await reducer.reduce_dimensions(X, method='pca')
        
        # Step 2: Clustering on reduced data
        clusterer = ClusteringOptimizerFacadeFactory.create_fast_optimizer()
        clustering_result = await clusterer.optimize_clustering(reduction_result.transformed_data)
        
        # Validate pipeline results
        assert reduction_result.transformed_data.shape == (200, 15)
        assert clustering_result['status'] in ['success', 'success_with_warnings', 'low_quality']
        assert len(clustering_result['labels']) == 200

    def test_factory_patterns(self):
        """Test that factory patterns work correctly."""
        # Test dimensionality reduction factories
        fast_reducer = DimensionalityReducerFacadeFactory.create_fast_reducer()
        quality_reducer = DimensionalityReducerFacadeFactory.create_high_quality_reducer()
        hd_reducer = DimensionalityReducerFacadeFactory.create_high_dimensional_reducer()
        
        assert fast_reducer.enable_neural_methods is False
        assert quality_reducer.enable_neural_methods is True
        assert hd_reducer.target_dimensions == 15
        
        # Test clustering factories
        fast_clusterer = ClusteringOptimizerFacadeFactory.create_fast_optimizer()
        quality_clusterer = ClusteringOptimizerFacadeFactory.create_high_quality_optimizer()
        memory_clusterer = ClusteringOptimizerFacadeFactory.create_memory_efficient_optimizer()
        
        assert fast_clusterer.memory_efficient is True
        assert quality_clusterer.memory_efficient is False
        assert memory_clusterer.memory_efficient is True


class TestBackwardCompatibility:
    """Test backward compatibility with existing interfaces."""

    def test_import_compatibility(self):
        """Test that old imports still work."""
        # These should work without errors
        from src.prompt_improver.ml import ClusteringOptimizer, DimensionalityReducer
        from src.prompt_improver.ml.optimization import ClusteringOptimizer as OptClusteringOptimizer
        from src.prompt_improver.ml.learning import ContextClusteringEngine
        
        assert ClusteringOptimizer is not None
        assert DimensionalityReducer is not None
        assert OptClusteringOptimizer is not None
        assert ContextClusteringEngine is not None

    @pytest.mark.asyncio
    async def test_api_compatibility(self):
        """Test that API interfaces remain compatible."""
        # Test clustering interface
        clusterer = ClusteringOptimizerFacadeFactory.create_fast_optimizer()
        
        # Should have the expected methods
        assert hasattr(clusterer, 'optimize_clustering')
        assert hasattr(clusterer, 'run_orchestrated_analysis')
        assert hasattr(clusterer, 'cluster_contexts')
        
        # Test dimensionality reduction interface
        reducer = DimensionalityReducerFacadeFactory.create_fast_reducer()
        
        # Should have the expected methods
        assert hasattr(reducer, 'reduce_dimensions')
        assert hasattr(reducer, 'get_available_methods')
        assert hasattr(reducer, 'recommend_method')


@pytest.mark.integration
class TestServicePerformance:
    """Test that decomposed services maintain acceptable performance."""

    @pytest.fixture
    def performance_data(self):
        """Create larger dataset for performance testing."""
        np.random.seed(42)
        return np.random.randn(1000, 50)

    @pytest.mark.asyncio
    async def test_reduction_performance(self, performance_data):
        """Test dimensionality reduction performance."""
        X = performance_data
        reducer = DimensionalityReducerFacadeFactory.create_fast_reducer()
        
        import time
        start_time = time.time()
        result = await reducer.reduce_dimensions(X, method='pca')
        end_time = time.time()
        
        # Should complete in reasonable time
        assert (end_time - start_time) < 5.0  # Less than 5 seconds
        assert result.quality_score > 0.0

    @pytest.mark.asyncio
    async def test_clustering_performance(self, performance_data):
        """Test clustering performance."""
        X = performance_data
        clusterer = ClusteringOptimizerFacadeFactory.create_fast_optimizer()
        
        import time
        start_time = time.time()
        result = await clusterer.optimize_clustering(X)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert (end_time - start_time) < 10.0  # Less than 10 seconds
        assert result['status'] in ['success', 'success_with_warnings', 'low_quality']


if __name__ == "__main__":
    # Run basic smoke tests
    async def smoke_test():
        """Basic smoke test for manual execution."""
        print("Running ML services decomposition smoke test...")
        
        # Test data
        X = np.random.randn(50, 20)
        y = np.random.randint(0, 3, 50)
        
        # Test dimensionality reduction
        reducer = DimensionalityReducerFacadeFactory.create_fast_reducer(target_dimensions=5)
        reduction_result = await reducer.reduce_dimensions(X, y)
        print(f"Dimensionality reduction: {reduction_result.original_dimensions}→{reduction_result.reduced_dimensions}")
        
        # Test clustering
        clusterer = ClusteringOptimizerFacadeFactory.create_fast_optimizer()
        clustering_result = await clusterer.optimize_clustering(X)
        print(f"Clustering: {clustering_result['status']}, quality: {clustering_result.get('quality_metrics', {}).get('quality_score', 0):.3f}")
        
        print("Smoke test completed successfully!")
    
    asyncio.run(smoke_test())