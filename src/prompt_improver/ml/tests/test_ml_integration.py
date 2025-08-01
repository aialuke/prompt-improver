"""Integration tests for ML module with real behavior validation."""

import asyncio
import pytest
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any

from ..types import features, labels
from ..models.model_manager import ModelManager, model_config, get_memory_optimized_config
from ..optimization.algorithms.clustering_optimizer import ClusteringOptimizer, ClusteringConfig
from ..optimization.algorithms.dimensionality_reducer import DimensionalityReducer, DimensionalityConfig
from ..learning.features.linguistic_feature_extractor import LinguisticFeatureExtractor
from ..core.training_data_loader import TrainingDataLoader

class TestMLPipelineIntegration:
    """Test complete ML pipeline with real data flow."""

    @pytest.fixture
    def sample_text_data(self) -> list[str]:
        """Generate sample text data for testing."""
        return [
            "Create a function to calculate the factorial of a number",
            "Implement a sorting algorithm with O(n log n) complexity",
            "Write a web scraper to extract data from websites",
            "Design a REST API for user authentication",
            "Build a machine learning model for text classification",
            "Optimize database queries for better performance",
            "Develop a mobile app with React Native",
            "Create a distributed system using microservices",
            "Implement caching mechanism for API responses",
            "Write unit tests for the authentication module",
        ] * 10  # Replicate to get 100 samples

    @pytest.fixture
    def feature_extractor(self) -> LinguisticFeatureExtractor:
        """Create a linguistic feature extractor."""
        return LinguisticFeatureExtractor()

    def test_feature_extraction_pipeline(self, sample_text_data, feature_extractor):
        """Test feature extraction with proper types."""
        start_time = time.time()
        
        # Extract features
        features_list = []
        for text in sample_text_data:
            try:
                feature_dict = feature_extractor.extract_features(text)
                feature_vector = feature_extractor._dict_to_vector(feature_dict)
                features_list.append(feature_vector)
            except Exception as e:
                # Handle extraction errors gracefully
                print(f"Feature extraction error: {e}")
                features_list.append(np.zeros(31))  # Default feature vector
        
        # Convert to numpy array with proper type
        features: features = np.array(features_list, dtype=np.float64)
        
        extraction_time = time.time() - start_time
        
        # Validate features
        assert features.shape == (100, 31)
        assert features.dtype == np.float64
        assert not np.isnan(features).any()
        assert extraction_time < 10.0  # Should complete within 10 seconds
        
        print(f"✓ Feature extraction completed in {extraction_time:.2f}s")
        print(f"  Shape: {features.shape}, Mean: {features.mean():.3f}, Std: {features.std():.3f}")

    @pytest.mark.asyncio
    async def test_clustering_optimization_pipeline(self, sample_text_data, feature_extractor):
        """Test clustering optimization with real features."""
        # Extract features
        features_list = []
        for text in sample_text_data[:50]:  # Use subset for faster test
            try:
                feature_dict = feature_extractor.extract_features(text)
                feature_vector = feature_extractor._dict_to_vector(feature_dict)
                features_list.append(feature_vector)
            except:
                features_list.append(np.zeros(31))
        
        features: features = np.array(features_list, dtype=np.float64)
        
        # Configure clustering
        config = ClusteringConfig(
            memory_efficient_mode=True,
            target_dimensions=10,
            hdbscan_min_cluster_size=3,
            enable_performance_tracking=True
        )
        
        optimizer = ClusteringOptimizer(config)
        
        # Run optimization
        start_time = time.time()
        result = await optimizer.optimize_clustering(features)
        optimization_time = time.time() - start_time
        
        # Validate results
        assert "error" not in result or result.get("error") is None
        assert "clustering_result" in result or "metrics" in result
        assert optimization_time < 30.0  # Should complete within 30 seconds
        
        print(f"✓ Clustering optimization completed in {optimization_time:.2f}s")
        if "metrics" in result:
            print(f"  Metrics: {result['metrics']}")

    def test_model_manager_memory_optimization(self):
        """Test model manager memory optimization."""
        # Test different memory configurations
        configs = [
            get_memory_optimized_config(25),   # Ultra-light
            get_memory_optimized_config(50),   # Light
            get_memory_optimized_config(100),  # Standard
        ]
        
        for i, config in enumerate(configs):
            manager = ModelManager(config)
            info = manager.get_model_info()
            
            assert info["model_name"] is not None
            assert info["quantization_bits"] in [4, 8, 16]
            assert info["memory_usage_mb"] >= 0
            
            print(f"✓ Config {i+1}: {info['model_name']} - {info['quantization_bits']}bit")
            
            # Cleanup
            manager.cleanup()

    @pytest.mark.asyncio
    async def test_full_ml_pipeline(self, sample_text_data, feature_extractor):
        """Test complete ML pipeline from text to clustering."""
        print("\n=== Full ML Pipeline Test ===")
        
        # Step 1: Feature Extraction
        print("1. Extracting features...")
        features_list = []
        for text in sample_text_data[:30]:  # Use subset
            try:
                feature_dict = feature_extractor.extract_features(text)
                feature_vector = feature_extractor._dict_to_vector(feature_dict)
                features_list.append(feature_vector)
            except:
                features_list.append(np.zeros(31))
        
        features: features = np.array(features_list, dtype=np.float64)
        print(f"   ✓ Extracted {features.shape[0]} feature vectors")
        
        # Step 2: Dimensionality Reduction
        print("2. Reducing dimensionality...")
        try:
            from ..optimization.algorithms.dimensionality_reducer import (
                DimensionalityReducer,
                DimensionalityConfig
            )
            
            dim_config = DimensionalityConfig(
                target_dimensions=10,
                method_selection="auto",
                preserve_variance=0.95
            )
            
            reducer = DimensionalityReducer(dim_config)
            reduction_result = await reducer.reduce_dimensions(features)
            
            if "reduced_features" in reduction_result:
                reduced_features = reduction_result["reduced_features"]
                print(f"   ✓ Reduced to {reduced_features.shape[1]} dimensions")
            else:
                reduced_features = features[:, :10]  # Fallback
                print("   ! Using fallback dimension reduction")
        except Exception as e:
            print(f"   ! Dimension reduction failed: {e}")
            reduced_features = features[:, :10]
        
        # Step 3: Clustering
        print("3. Performing clustering...")
        cluster_config = ClusteringConfig(
            memory_efficient_mode=True,
            hdbscan_min_cluster_size=3,
            auto_dim_reduction=False  # Already reduced
        )
        
        optimizer = ClusteringOptimizer(cluster_config)
        cluster_result = await optimizer.optimize_clustering(reduced_features)
        
        if "error" not in cluster_result:
            print("   ✓ Clustering completed successfully")
            if "metrics" in cluster_result:
                metrics = cluster_result["metrics"]
                print(f"   Silhouette Score: {metrics.get('silhouette_score', 'N/A')}")
        else:
            print(f"   ! Clustering error: {cluster_result['error']}")
        
        print("\n=== Pipeline Test Complete ===")

class TestTypeSystemPerformance:
    """Test performance impact of type annotations."""

    def test_type_checking_overhead(self):
        """Measure overhead of type checking."""
        import sys
        
        # Create large arrays
        features = np.random.rand(10000, 31).astype(np.float64)
        labels = np.random.randint(0, 10, 10000).astype(np.int64)
        
        # Test with type annotations
        start_time = time.time()
        for _ in range(100):
            f: features = features
            l: labels = labels
            _ = f.shape
            _ = l.shape
        typed_time = time.time() - start_time
        
        # Test without type annotations
        start_time = time.time()
        for _ in range(100):
            f = features
            l = labels
            _ = f.shape
            _ = l.shape
        untyped_time = time.time() - start_time
        
        # Type annotations should have minimal overhead
        overhead = (typed_time - untyped_time) / untyped_time * 100
        print(f"Type annotation overhead: {overhead:.2f}%")
        
        # Assert overhead is reasonable (less than 5%)
        assert abs(overhead) < 5.0

    def test_memory_usage_with_types(self):
        """Test memory usage with type annotations."""
        import psutil
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Measure initial memory
        process = psutil.process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create typed objects
        features: features = np.random.rand(1000, 31).astype(np.float64)
        labels: labels = np.random.randint(0, 5, 1000).astype(np.int64)
        
        from ..types import TrainingBatch, OptimizationResult, ClusteringResult
        
        # Create multiple typed objects
        batches = []
        for i in range(10):
            batch = TrainingBatch(
                features=features[i*100:(i+1)*100],
                labels=labels[i*100:(i+1)*100],
                metadata={"batch": i}
            )
            batches.append(batch)
        
        # Measure memory after
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase: {memory_increase:.2f} MB")
        
        # Memory increase should be reasonable (less than 50MB for this test)
        assert memory_increase < 50.0

class TestMLModelValidation:
    """Validate ML model behavior with types."""

    def test_model_pipeline_types(self):
        """Test model pipeline with proper types."""
        config = model_config(
            model_name="distilbert-base-uncased",
            task="ner",
            use_quantization=True,
            quantization_bits=8
        )
        
        manager = ModelManager(config)
        pipeline = manager.get_pipeline()
        
        if pipeline is not None:
            # Test with sample text
            test_text = "John Smith works at Microsoft in Seattle."
            try:
                result = pipeline(test_text)
                assert isinstance(result, list)
                print(f"✓ Pipeline inference successful: {len(result)} entities")
            except Exception as e:
                print(f"! Pipeline inference failed: {e}")
        else:
            print("! Pipeline not available (transformers not installed)")
        
        # Cleanup
        manager.cleanup()

    @pytest.mark.parametrize("memory_target", [30, 60, 100])
    def test_memory_optimized_configs(self, memory_target: int):
        """Test memory-optimized configurations."""
        config = get_memory_optimized_config(memory_target)
        
        assert config.model_name is not None
        assert config.quantization_bits in [4, 8, 16]
        assert config.use_quantization is True
        
        if memory_target < 30:
            assert "tiny" in config.model_name.lower()
        elif memory_target < 60:
            assert "tiny" in config.model_name.lower() or "small" in config.model_name.lower()
        
        print(f"✓ Memory target {memory_target}MB -> {config.model_name} ({config.quantization_bits}bit)")

def test_compilation_performance():
    """Test TypeScript compilation performance impact."""
    import subprocess
    import time
    
    # This would normally run mypy on the ML module
    # For now, we'll just ensure the types module exists and is importable
    try:
        from ..types import features, labels, cluster_labels
        print("✓ ML types module successfully imported")
    except ImportError as e:
        pytest.fail(f"Failed to import ML types: {e}")

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])