"""Test type safety improvements in ML module."""

import pytest
import numpy as np
from typing import TYPE_CHECKING

from ..types import (
    features,
    labels,
    cluster_labels,
    ensure_features,
    ensure_labels,
    is_valid_features,
    is_valid_labels,
    TrainingBatch,
    OptimizationResult,
    ClusteringResult,
)
from ..models.model_manager import ModelManager, model_config
from ..optimization.algorithms.clustering_optimizer import ClusteringOptimizer, ClusteringConfig

class TestMLTypes:
    """Test ML type definitions and type safety."""

    def test_numpy_array_types(self):
        """Test numpy array type aliases work correctly."""
        # Create typed arrays
        features: features = np.random.rand(100, 31).astype(np.float64)
        labels: labels = np.random.randint(0, 5, 100).astype(np.int64)
        
        # Verify types
        assert features.dtype == np.float64
        assert labels.dtype == np.int64
        assert features.shape == (100, 31)
        assert labels.shape == (100,)

    def test_type_guards(self):
        """Test type guard functions."""
        valid_features = np.random.rand(50, 10).astype(np.float64)
        valid_labels = np.array([0, 1, 2, 3, 4] * 10).astype(np.int64)
        
        assert is_valid_features(valid_features)
        assert is_valid_labels(valid_labels)
        
        # Test invalid cases
        assert not is_valid_features(valid_labels)  # Wrong dtype
        assert not is_valid_features(np.array([1, 2, 3]))  # Wrong shape
        assert not is_valid_labels(valid_features)  # Wrong dtype
        assert not is_valid_labels(np.array([[1, 2], [3, 4]]))  # Wrong shape

    def test_ensure_functions(self):
        """Test array conversion functions."""
        # Test ensure_features
        list_data = [[1, 2, 3], [4, 5, 6]]
        features = ensure_features(list_data)
        assert features.shape == (2, 3)
        assert features.dtype == np.float64
        
        # Test 1D to 2D conversion
        one_d_data = [1, 2, 3, 4]
        features_2d = ensure_features(one_d_data)
        assert features_2d.shape == (4, 1)
        
        # Test ensure_labels
        label_list = [0, 1, 2, 3, 4]
        labels = ensure_labels(label_list)
        assert labels.shape == (5,)
        assert labels.dtype == np.int64

    def test_training_batch(self):
        """Test TrainingBatch container."""
        features = np.random.rand(100, 31).astype(np.float64)
        labels = np.random.randint(0, 5, 100).astype(np.int64)
        weights = np.random.rand(100).astype(np.float64)
        
        batch = TrainingBatch(
            features=features,
            labels=labels,
            sample_weights=weights,
            metadata={"source": "test", "version": 1}
        )
        
        assert batch.features.shape == (100, 31)
        assert batch.labels.shape == (100,)
        assert batch.sample_weights is not None
        assert batch.metadata["source"] == "test"

    def test_result_containers(self):
        """Test result container types."""
        # Test OptimizationResult
        features = np.random.rand(50, 10).astype(np.float64)
        metrics = {"accuracy": 0.95, "loss": 0.05}
        
        opt_result = OptimizationResult(
            optimized_features=features,
            metrics=metrics,
            model_params={"learning_rate": 0.01},
            execution_time=1.23
        )
        
        assert opt_result.optimized_features.shape == (50, 10)
        assert opt_result.metrics["accuracy"] == 0.95
        assert opt_result.execution_time == 1.23
        
        # Test ClusteringResult
        labels = np.array([0, 1, 1, 2, 2, 2, 0, 0, 1, 2]).astype(np.int64)
        centers = np.random.rand(3, 10).astype(np.float64)
        
        cluster_result = ClusteringResult(
            labels=labels,
            centers=centers,
            metrics={"silhouette": 0.7},
            n_clusters=3
        )
        
        assert cluster_result.labels.shape == (10,)
        assert cluster_result.centers.shape == (3, 10)
        assert cluster_result.n_clusters == 3
        assert cluster_result.metrics["silhouette"] == 0.7

class TestModelManagerTypes:
    """Test type safety in ModelManager."""

    def test_model_manager_initialization(self):
        """Test ModelManager with proper types."""
        config = model_config(
            model_name="test-model",
            task="ner",
            use_quantization=False
        )
        
        manager = ModelManager(config)
        assert manager.config.model_name == "test-model"
        
        # Test pipeline type annotation
        pipeline = manager.get_pipeline()
        # Pipeline might be None if transformers not available
        assert pipeline is None or hasattr(pipeline, "__call__")

    def test_model_info_types(self):
        """Test model info return types."""
        manager = ModelManager()
        info = manager.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "memory_usage_mb" in info
        assert isinstance(info["memory_usage_mb"], (int, float))

class TestClusteringOptimizerTypes:
    """Test type safety in ClusteringOptimizer."""

    def test_clustering_optimizer_initialization(self):
        """Test ClusteringOptimizer with proper types."""
        config = ClusteringConfig(
            memory_efficient_mode=True,
            target_dimensions=10
        )
        
        optimizer = ClusteringOptimizer(config)
        assert optimizer.config.memory_efficient_mode is True

    @pytest.mark.asyncio
    async def test_optimize_clustering_types(self):
        """Test optimize_clustering with proper types."""
        optimizer = ClusteringOptimizer()
        
        # Create typed test data
        features: features = np.random.rand(100, 31).astype(np.float64)
        labels: labels = np.random.randint(0, 3, 100).astype(np.int64)
        
        # This should type check correctly
        result = await optimizer.optimize_clustering(
            features=features,
            labels=labels
        )
        
        assert isinstance(result, dict)

class TestTypeCompatibility:
    """Test compatibility with existing code."""

    def test_backward_compatibility(self):
        """Ensure type annotations don't break existing code."""
        # Old style code should still work
        features = np.array([[1, 2, 3], [4, 5, 6]])
        labels = np.array([0, 1])
        
        # Create components with numpy arrays directly
        batch = TrainingBatch(features=features, labels=labels)
        assert batch.features.shape == (2, 3)

    def test_type_annotations_runtime(self):
        """Test that type annotations don't affect runtime."""
        # Type annotations should not be enforced at runtime
        features = np.array([[1.0, 2.0]], dtype=np.float32)  # Wrong dtype
        
        # This should still work (Python doesn't enforce types at runtime)
        batch = TrainingBatch(
            features=features,  # type: ignore
            labels=np.array([0])
        )
        assert batch.features.dtype == np.float32

if __name__ == "__main__":
    pytest.main([__file__, "-v"])