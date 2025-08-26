"""Integration tests for Clustering Optimization module.

Tests high-dimensional clustering optimization, memory efficiency,
parameter tuning, and performance monitoring capabilities.
"""

import time

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification

from prompt_improver.ml.optimization.algorithms.clustering_optimizer import (
    ClusteringConfig,
    ClusteringOptimizer,
    get_clustering_optimizer,
)


@pytest.fixture
def clustering_config():
    """Create test configuration for clustering optimization."""
    return ClusteringConfig(
        memory_efficient_mode=True,
        max_memory_mb=512,
        auto_dim_reduction=True,
        target_dimensions=8,
        hdbscan_min_cluster_size=3,
        min_silhouette_score=0.2,
        enable_performance_tracking=True,
        enable_caching=True,
    )


@pytest.fixture
def high_dimensional_data():
    """Create high-dimensional test data (31 features like linguistic analysis)."""
    X, y = make_classification(
        n_samples=500,
        n_features=31,
        n_informative=20,
        n_redundant=5,
        n_clusters_per_class=2,
        class_sep=1.5,
        random_state=42,
    )
    noise = np.random.normal(0, 0.1, X.shape)
    X_noisy = X + noise
    return (X_noisy, y)


@pytest.fixture
def linguistic_like_features():
    """Create features that simulate linguistic analysis output."""
    np.random.seed(42)
    n_samples = 300
    features = []
    for _i in range(n_samples):
        feature_vector = []
        feature_vector.extend(np.random.beta(2, 2, 5))
        feature_vector.append(np.random.beta(3, 2))
        feature_vector.append(np.random.beta(2, 3))
        feature_vector.append(np.random.exponential(0.2))
        feature_vector.append(np.random.beta(2, 2))
        feature_vector.append(np.random.beta(3, 2))
        feature_vector.append(np.random.exponential(0.1))
        feature_vector.append(np.random.gamma(2, 0.3))
        feature_vector.append(np.random.beta(3, 2))
        feature_vector.append(np.random.binomial(1, 0.3))
        feature_vector.append(np.random.beta(2, 2))
        project_type_encoding = np.zeros(7)
        project_type_encoding[np.random.randint(0, 7)] = 1.0
        feature_vector.extend(project_type_encoding)
        domain_encoding = np.zeros(7)
        domain_encoding[np.random.randint(0, 7)] = 1.0
        feature_vector.extend(domain_encoding)
        feature_vector.append(np.random.choice([0.25, 0.5, 0.75, 1.0]))
        feature_vector.append(np.random.uniform(0, 1))
        features.append(feature_vector)
    return np.array(features)


@pytest.mark.asyncio
async def test_clustering_optimizer_initialization():
    """Test clustering optimizer initialization with configuration."""
    config = ClusteringConfig(target_dimensions=10, memory_efficient_mode=True)
    optimizer = ClusteringOptimizer(config)
    assert optimizer.config.target_dimensions == 10
    assert optimizer.config.memory_efficient_mode is True
    assert optimizer.scaler is None
    assert optimizer.performance_metrics == []
    assert optimizer.cache == {}


@pytest.mark.asyncio
async def test_high_dimensional_clustering_optimization(
    clustering_config, high_dimensional_data
):
    """Test clustering optimization on high-dimensional data."""
    X, _y = high_dimensional_data
    optimizer = ClusteringOptimizer(clustering_config)
    result = await optimizer.optimize_clustering(X)
    assert result["status"] == "success"
    assert "labels" in result
    assert "quality_metrics" in result
    assert "preprocessing_info" in result
    assert "optimal_parameters" in result
    preprocessing_info = result["preprocessing_info"]
    assert preprocessing_info["original_shape"] == (500, 31)
    assert preprocessing_info["final_shape"][1] <= clustering_config.target_dimensions
    labels = result["labels"]
    assert len(labels) == len(X)
    assert len(set(labels)) > 1
    quality_metrics = result["quality_metrics"]
    assert "quality_score" in quality_metrics
    assert 0.0 <= quality_metrics["quality_score"] <= 1.0


@pytest.mark.asyncio
async def test_linguistic_features_clustering(
    clustering_config, linguistic_like_features
):
    """Test clustering optimization on realistic linguistic features."""
    optimizer = ClusteringOptimizer(clustering_config)
    result = await optimizer.optimize_clustering(linguistic_like_features)
    assert result["status"] == "success"
    preprocessing_info = result["preprocessing_info"]
    assert preprocessing_info["original_shape"] == (300, 31)
    assert preprocessing_info["final_shape"][1] <= clustering_config.target_dimensions
    labels = result["labels"]
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    assert n_clusters >= 2
    assert result["optimization_time"] > 0
    assert "memory_usage" in result


@pytest.mark.asyncio
async def test_memory_efficient_processing(clustering_config):
    """Test memory-efficient processing with large datasets."""
    from sklearn.datasets import make_classification

    X, _ = make_classification(
        n_samples=2000,
        n_features=31,
        n_informative=20,
        n_redundant=5,
        n_clusters_per_class=2,
        class_sep=1.2,
        random_state=42,
    )
    clustering_config.memory_efficient_mode = True
    clustering_config.max_memory_mb = 256
    optimizer = ClusteringOptimizer(clustering_config)
    start_time = time.time()
    result = await optimizer.optimize_clustering(X)
    processing_time = time.time() - start_time
    assert result["status"] in {"success", "success_with_warnings", "low_quality"}
    assert processing_time < 60.0
    memory_info = result["memory_usage"]
    assert "initial_mb" in memory_info
    assert "peak_mb" in memory_info
    assert "final_mb" in memory_info


@pytest.mark.asyncio
async def test_parameter_optimization():
    """Test adaptive parameter optimization for different dataset sizes."""
    optimizer = ClusteringOptimizer()
    from sklearn.datasets import make_classification

    small_X, _ = make_classification(
        n_samples=50,
        n_features=31,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42,
    )
    small_result = await optimizer.optimize_clustering(small_X)
    large_X, _ = make_classification(
        n_samples=1000,
        n_features=31,
        n_informative=20,
        n_redundant=5,
        n_clusters_per_class=2,
        class_sep=1.2,
        random_state=43,
    )
    large_result = await optimizer.optimize_clustering(large_X)
    assert small_result["status"] in {"success", "success_with_warnings", "low_quality"}
    assert large_result["status"] in {"success", "success_with_warnings", "low_quality"}
    small_params = small_result["optimal_parameters"]["hdbscan"]
    large_params = large_result["optimal_parameters"]["hdbscan"]
    assert small_params["min_cluster_size"] <= large_params["min_cluster_size"]


@pytest.mark.asyncio
async def test_quality_assessment_comprehensive():
    """Test comprehensive clustering quality assessment."""
    X, _ = make_blobs(
        n_samples=400, centers=4, n_features=31, cluster_std=1.5, random_state=42
    )
    optimizer = ClusteringOptimizer()
    result = await optimizer.optimize_clustering(X)
    assert result["status"] == "success"
    quality_metrics = result["quality_metrics"]
    required_metrics = [
        "n_clusters",
        "n_noise_points",
        "noise_ratio",
        "silhouette_score",
        "calinski_harabasz_score",
        "davies_bouldin_score",
        "quality_score",
        "processing_time_seconds",
        "convergence_achieved",
    ]
    for metric in required_metrics:
        assert metric in quality_metrics
    assert quality_metrics["quality_score"] > 0.3
    assert quality_metrics["n_clusters"] >= 3
    assert quality_metrics["convergence_achieved"] is True


@pytest.mark.asyncio
async def test_preprocessing_pipeline():
    """Test the feature preprocessing pipeline."""
    X = np.random.rand(200, 31)
    X[:, :10] *= 100
    X[:, 10:20] *= 0.01
    optimizer = ClusteringOptimizer()
    result = await optimizer.optimize_clustering(X)
    assert result["status"] == "success"
    preprocessing_info = result["preprocessing_info"]
    assert preprocessing_info["scaling_method"] == "robust"
    assert preprocessing_info["dimensionality_reduction"] is True
    assert preprocessing_info["preprocessing_time"] > 0
    original_shape = preprocessing_info["original_shape"]
    final_shape = preprocessing_info["final_shape"]
    assert final_shape[1] <= original_shape[1]


@pytest.mark.asyncio
async def test_caching_functionality(clustering_config):
    """Test caching of preprocessing results."""
    X = np.random.rand(100, 31)
    clustering_config.enable_caching = True
    optimizer = ClusteringOptimizer(clustering_config)
    result1 = await optimizer.optimize_clustering(X)
    cache_stats1 = result1["cache_stats"]
    result2 = await optimizer.optimize_clustering(X)
    cache_stats2 = result2["cache_stats"]
    assert result1["status"] == "success"
    assert result2["status"] == "success"
    assert cache_stats2["hits"] > cache_stats1["hits"]
    assert cache_stats2["hit_ratio"] > cache_stats1["hit_ratio"]


@pytest.mark.asyncio
async def test_performance_tracking():
    """Test performance tracking and monitoring."""
    optimizer = ClusteringOptimizer()
    for i in range(3):
        X = np.random.rand(100 + i * 50, 31)
        result = await optimizer.optimize_clustering(X)
        assert result["status"] == "success"
    performance_summary = optimizer.get_performance_summary()
    assert performance_summary["status"] == "available"
    assert performance_summary["total_runs"] == 3
    assert "recent_performance" in performance_summary
    assert "cache_performance" in performance_summary
    recent_perf = performance_summary["recent_performance"]
    assert "avg_quality_score" in recent_perf
    assert "avg_processing_time" in recent_perf
    assert "convergence_rate" in recent_perf


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling for invalid inputs."""
    optimizer = ClusteringOptimizer()
    small_X = np.random.rand(2, 31)
    result = await optimizer.optimize_clustering(small_X)
    assert result["status"] == "failed"
    assert "error" in result
    invalid_X = np.random.rand(100, 31)
    invalid_X[0, 0] = np.nan
    result = await optimizer.optimize_clustering(invalid_X)
    assert result["status"] == "failed"
    assert "error" in result
    wrong_dim_X = np.random.rand(100)
    result = await optimizer.optimize_clustering(wrong_dim_X)
    assert result["status"] == "failed"
    assert "error" in result


@pytest.mark.asyncio
async def test_supervised_optimization():
    """Test clustering optimization with supervised labels."""
    X, y = make_classification(
        n_samples=200, n_features=31, n_classes=3, n_informative=20, random_state=42
    )
    optimizer = ClusteringOptimizer()
    result = await optimizer.optimize_clustering(X, labels=y)
    assert result["status"] == "success"
    preprocessing_info = result["preprocessing_info"]
    if preprocessing_info.get("feature_importance") is not None:
        assert len(preprocessing_info["feature_importance"]) > 0


@pytest.mark.asyncio
async def test_factory_function():
    """Test the factory function for creating optimizer instances."""
    config = ClusteringConfig(target_dimensions=8)
    optimizer = get_clustering_optimizer(config)
    assert isinstance(optimizer, ClusteringOptimizer)
    assert optimizer.config.target_dimensions == 8
    default_optimizer = get_clustering_optimizer()
    assert isinstance(default_optimizer, ClusteringOptimizer)
    assert default_optimizer.config is not None


@pytest.mark.asyncio
async def test_performance_recommendations():
    """Test performance analysis and recommendations generation."""
    from sklearn.datasets import make_classification

    X, _ = make_classification(
        n_samples=50,
        n_features=31,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=42,
    )
    config = ClusteringConfig(min_silhouette_score=0.8, max_noise_ratio=0.1)
    optimizer = ClusteringOptimizer(config)
    result = await optimizer.optimize_clustering(X)
    assert result["status"] in {"success", "success_with_warnings", "low_quality"}
    performance_analysis = result["performance_analysis"]
    assert "performance_summary" in performance_analysis
    assert "quality_assessment" in performance_analysis
    assert "recommendations" in performance_analysis
    recommendations = performance_analysis["recommendations"]
    assert isinstance(recommendations, list)


if __name__ == "__main__":

    async def run_debug_test():
        """Run a specific test for debugging."""
        config = ClusteringConfig(
            memory_efficient_mode=True,
            target_dimensions=8,
            enable_performance_tracking=True,
        )
        np.random.seed(42)
        X = np.random.rand(200, 31)
        optimizer = ClusteringOptimizer(config)
        result = await optimizer.optimize_clustering(X)
        print(f"Status: {result['status']}")
        print(f"Quality Score: {result['quality_metrics']['quality_score']:.3f}")
        print(f"Clusters Found: {result['quality_metrics']['n_clusters']}")
        print(f"Processing Time: {result['optimization_time']:.2f}s")
        print(
            f"Dimensionality: {result['preprocessing_info']['original_shape']} → {result['preprocessing_info']['final_shape']}"
        )
        if result["performance_analysis"]["recommendations"]:
            print("Recommendations:")
            for rec in result["performance_analysis"]["recommendations"]:
                print(f"  - {rec['message']} (Priority: {rec['priority']})")
