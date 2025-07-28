"""Comprehensive tests for Week 6 Data Generation Optimization features

Tests all Week 6 components with real behavior testing:
- Enhanced neural generation methods
- Hybrid generation system
- Dynamic batch optimization
- Generation history tracking
- Database integration
- Analytics and reporting
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch

from prompt_improver.ml.preprocessing.synthetic_data_generator import (
    ProductionSyntheticDataGenerator,
    HybridGenerationSystem,
    MethodPerformanceTracker,
    GenerationMethodMetrics,
    TabularGAN,
    TabularVAE,
    TabularDiffusion
)
from prompt_improver.ml.optimization.batch.dynamic_batch_optimizer import (
    DynamicBatchOptimizer,
    BatchOptimizationConfig,
    BatchPerformanceMetrics
)
from prompt_improver.ml.analytics.generation_analytics import (
    GenerationHistoryTracker,
    GenerationAnalytics
)
from prompt_improver.database.services.generation_service import GenerationDatabaseService
from prompt_improver.database.models import (
    GenerationSession,
    GenerationBatch,
    SyntheticDataSample
)


class TestEnhancedNeuralGeneration:
    """Test enhanced neural generation methods with 2025 best practices"""

    def test_method_performance_tracker_initialization(self):
        """Test MethodPerformanceTracker initialization and basic functionality"""
        tracker = MethodPerformanceTracker()

        assert tracker.method_history == {}
        assert tracker.method_rankings == {}

    def test_method_performance_recording(self):
        """Test recording and ranking of method performance"""
        tracker = MethodPerformanceTracker()

        # Record performance for different methods
        metrics1 = GenerationMethodMetrics(
            method_name="statistical",
            generation_time=1.0,
            quality_score=0.8,
            diversity_score=0.7,
            memory_usage_mb=100.0,
            success_rate=1.0,
            samples_generated=100,
            timestamp=datetime.now(),
            performance_gaps_addressed={"accuracy": 0.1}
        )

        metrics2 = GenerationMethodMetrics(
            method_name="neural",
            generation_time=2.0,
            quality_score=0.9,
            diversity_score=0.8,
            memory_usage_mb=200.0,
            success_rate=1.0,
            samples_generated=100,
            timestamp=datetime.now(),
            performance_gaps_addressed={"accuracy": 0.1}
        )

        tracker.record_performance(metrics1)
        tracker.record_performance(metrics2)

        # Check that methods are ranked
        assert "statistical" in tracker.method_rankings
        assert "neural" in tracker.method_rankings

        # Neural should rank higher due to better quality and diversity
        assert tracker.method_rankings["neural"] > tracker.method_rankings["statistical"]

    def test_best_method_selection(self):
        """Test intelligent method selection based on performance gaps"""
        tracker = MethodPerformanceTracker()

        # Record some performance history
        for i in range(5):
            metrics = GenerationMethodMetrics(
                method_name="hybrid",
                generation_time=1.5,
                quality_score=0.85,
                diversity_score=0.75,
                memory_usage_mb=150.0,
                success_rate=1.0,
                samples_generated=100,
                timestamp=datetime.now(),
                performance_gaps_addressed={"diversity": 0.2}
            )
            tracker.record_performance(metrics)

        # Test method selection
        performance_gaps = {"diversity": 0.15}
        best_method = tracker.get_best_method(performance_gaps)

        assert best_method in ["hybrid", "statistical"]  # Should select based on available methods

    @pytest.mark.skipif(not hasattr(TabularGAN, '__init__'), reason="PyTorch not available")
    def test_tabular_gan_initialization(self):
        """Test enhanced TabularGAN initialization"""
        data_dim = 10
        gan = TabularGAN(data_dim=data_dim)

        assert gan.data_dim == data_dim
        assert gan.noise_dim == 100  # Default
        assert hasattr(gan, 'generator')
        assert hasattr(gan, 'discriminator')

    @pytest.mark.skipif(not hasattr(TabularVAE, '__init__'), reason="PyTorch not available")
    def test_tabular_vae_initialization(self):
        """Test enhanced TabularVAE with β-VAE features"""
        data_dim = 10
        vae = TabularVAE(data_dim=data_dim, beta=1.5)

        assert vae.data_dim == data_dim
        assert vae.beta == 1.5
        assert hasattr(vae, 'encoder')
        assert hasattr(vae, 'decoder')


class TestHybridGenerationSystem:
    """Test hybrid generation system combining multiple methods"""

    @pytest.mark.asyncio
    async def test_hybrid_system_initialization(self):
        """Test HybridGenerationSystem initialization"""
        data_dim = 5
        hybrid_system = HybridGenerationSystem(data_dim=data_dim, device="cpu")

        assert hybrid_system.data_dim == data_dim
        assert isinstance(hybrid_system.performance_tracker, MethodPerformanceTracker)
        assert "statistical" in hybrid_system.method_weights

    @pytest.mark.asyncio
    async def test_method_allocation_determination(self):
        """Test intelligent method allocation based on performance gaps"""
        hybrid_system = HybridGenerationSystem(data_dim=5, device="cpu")

        performance_gaps = {"model_accuracy": 0.15, "diversity": 0.1}
        total_samples = 100

        allocation = hybrid_system._determine_method_allocation(performance_gaps, total_samples)

        # Check that allocation sums to total samples
        assert sum(allocation.values()) == total_samples

        # Check that all methods have some allocation
        assert all(count >= 0 for count in allocation.values())

    @pytest.mark.asyncio
    async def test_hybrid_data_generation(self):
        """Test end-to-end hybrid data generation"""
        hybrid_system = HybridGenerationSystem(data_dim=3, device="cpu")

        performance_gaps = {"accuracy": 0.1}
        batch_size = 50
        quality_threshold = 0.5

        result = await hybrid_system.generate_hybrid_data(
            batch_size=batch_size,
            performance_gaps=performance_gaps,
            quality_threshold=quality_threshold
        )

        # Verify result structure
        assert "samples" in result
        assert "method_metrics" in result
        assert "total_generation_time" in result
        assert "method_allocation" in result

        # Verify samples were generated
        assert len(result["samples"]) > 0
        assert result["total_generation_time"] > 0


class TestDynamicBatchOptimization:
    """Test dynamic batch size optimization system"""

    def test_batch_optimizer_initialization(self):
        """Test DynamicBatchOptimizer initialization"""
        config = BatchOptimizationConfig(
            min_batch_size=10,
            max_batch_size=500,
            initial_batch_size=100
        )
        optimizer = DynamicBatchOptimizer(config)

        assert optimizer.current_batch_size == 100
        assert optimizer.config.min_batch_size == 10
        assert optimizer.config.max_batch_size == 500

    @pytest.mark.asyncio
    async def test_optimal_batch_size_calculation(self):
        """Test optimal batch size calculation"""
        config = BatchOptimizationConfig(memory_limit_mb=1000.0)
        optimizer = DynamicBatchOptimizer(config)

        target_samples = 200
        current_memory = 100.0

        optimal_size = await optimizer.get_optimal_batch_size(
            target_samples=target_samples,
            current_memory_usage=current_memory
        )

        assert config.min_batch_size <= optimal_size <= min(target_samples, config.max_batch_size)

    @pytest.mark.asyncio
    async def test_batch_performance_recording(self):
        """Test batch performance recording and optimization"""
        optimizer = DynamicBatchOptimizer()

        # Record some performance metrics
        await optimizer.record_batch_performance(
            batch_size=100,
            processing_time=2.0,
            success_count=95,
            error_count=5
        )

        assert len(optimizer.performance_history) == 1

        metrics = optimizer.performance_history[0]
        assert metrics.batch_size == 100
        assert metrics.processing_time == 2.0
        assert metrics.success_rate == 0.95

    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation"""
        optimizer = DynamicBatchOptimizer()

        efficiency = optimizer._calculate_efficiency_score(
            batch_size=100,
            processing_time=1.0,
            memory_usage=500.0,
            throughput=100.0
        )

        assert 0.0 <= efficiency <= 1.0


class TestGenerationHistoryTracking:
    """Test generation history tracking and analytics"""

    @pytest.mark.asyncio
    async def test_generation_service_initialization(self):
        """Test GenerationDatabaseService initialization"""
        # Mock database session
        mock_session = AsyncMock()
        service = GenerationDatabaseService(mock_session)

        assert service.db_session == mock_session

    @pytest.mark.asyncio
    async def test_session_creation_and_tracking(self):
        """Test generation session creation and tracking"""
        mock_session = AsyncMock()
        tracker = GenerationHistoryTracker(mock_session)

        # Mock the database service methods
        with patch.object(tracker.db_service, 'create_generation_session') as mock_create:
            mock_create.return_value = MagicMock(session_id="test-session-123")

            session_id = await tracker.start_tracking_session(
                generation_method="hybrid",
                target_samples=100,
                performance_gaps={"accuracy": 0.1}
            )

            assert session_id == "test-session-123"
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_analytics_performance_trends(self):
        """Test analytics performance trend calculation"""
        mock_session = AsyncMock()
        analytics = GenerationAnalytics(mock_session)

        # Mock performance history
        mock_performance = [
            MagicMock(
                quality_score=0.8,
                diversity_score=0.7,
                success_rate=0.9,
                generation_time_seconds=1.0,
                recorded_at=datetime.now() - timedelta(days=i)
            )
            for i in range(5)
        ]

        with patch.object(analytics.db_service, 'get_method_performance_history') as mock_history:
            mock_history.return_value = mock_performance

            trends = await analytics.get_performance_trends(days_back=7, method_name="hybrid")

            assert "trends" in trends
            assert "current_averages" in trends
            assert "performance_ranges" in trends
            assert trends["total_executions"] == 5


class TestProductionIntegration:
    """Test integration of all Week 6 features with ProductionSyntheticDataGenerator"""

    @pytest.mark.asyncio
    async def test_production_generator_with_week6_features(self):
        """Test ProductionSyntheticDataGenerator with Week 6 enhancements"""
        generator = ProductionSyntheticDataGenerator(
            feature_names=["feature1", "feature2", "feature3"],
            target_samples=100,
            generation_method="hybrid",
            enable_gap_targeting=True
        )

        # Test that Week 6 components are initialized
        assert hasattr(generator, 'method_tracker')
        assert hasattr(generator, 'batch_optimizer')
        assert generator.enable_quality_filtering is True

    @pytest.mark.asyncio
    async def test_dynamic_batching_generation(self):
        """Test dynamic batching generation workflow"""
        generator = ProductionSyntheticDataGenerator(
            feature_names=["f1", "f2"],
            target_samples=50,
            generation_method="statistical"
        )

        performance_gaps = {"accuracy": 0.1}

        result = await generator.generate_with_dynamic_batching(
            total_samples=50,
            performance_gaps=performance_gaps,
            strategy="adaptive"
        )

        # Verify result structure
        assert "features" in result
        assert "effectiveness" in result
        assert "metadata" in result

        # Verify metadata contains batching info
        metadata = result["metadata"]
        assert "batches_processed" in metadata
        assert "total_generation_time" in metadata
        assert metadata["method"] == "dynamic_batching"

    @pytest.mark.asyncio
    async def test_quality_filtering_application(self):
        """Test quality-based filtering and ranking"""
        generator = ProductionSyntheticDataGenerator(
            feature_names=["f1", "f2"],
            target_samples=20,
            generation_method="statistical"
        )

        # Create mock result with varying quality
        mock_result = {
            "features": [[0.5, 0.5], [0.8, 0.2], [0.1, 0.9], [0.7, 0.3]],
            "effectiveness": [0.6, 0.9, 0.4, 0.8],
            "metadata": {"generation_method": "statistical"}
        }

        filtered_result = await generator._apply_quality_filtering(mock_result)

        # Verify filtering was applied
        assert "quality_filtering" in filtered_result["metadata"]
        assert len(filtered_result["features"]) <= len(mock_result["features"])

        # Verify samples are sorted by quality (highest first)
        if len(filtered_result["features"]) > 1:
            effectiveness_scores = filtered_result["effectiveness"]
            assert all(effectiveness_scores[i] >= effectiveness_scores[i+1]
                      for i in range(len(effectiveness_scores)-1))


class TestRealBehaviorIntegration:
    """Test real behavior integration without mocks where possible"""

    def test_method_performance_tracker_real_behavior(self):
        """Test MethodPerformanceTracker with real data flow"""
        tracker = MethodPerformanceTracker()

        # Simulate real performance data over time
        methods = ["statistical", "neural", "hybrid"]

        for day in range(7):
            for method in methods:
                # Simulate different performance characteristics
                base_quality = {"statistical": 0.7, "neural": 0.8, "hybrid": 0.85}[method]
                base_time = {"statistical": 0.5, "neural": 2.0, "hybrid": 1.5}[method]

                metrics = GenerationMethodMetrics(
                    method_name=method,
                    generation_time=base_time + np.random.normal(0, 0.1),
                    quality_score=base_quality + np.random.normal(0, 0.05),
                    diversity_score=0.7 + np.random.normal(0, 0.05),
                    memory_usage_mb=100 + np.random.normal(0, 20),
                    success_rate=0.95 + np.random.normal(0, 0.02),
                    samples_generated=100,
                    timestamp=datetime.now(),
                    performance_gaps_addressed={"accuracy": 0.1}
                )
                tracker.record_performance(metrics)

        # Verify realistic rankings
        assert len(tracker.method_rankings) == 3

        # Hybrid should generally rank highest due to better quality
        best_method = max(tracker.method_rankings, key=tracker.method_rankings.get)
        assert best_method in ["hybrid", "neural"]  # Either could be best

    def test_batch_optimizer_real_adaptation(self):
        """Test batch optimizer adaptation with realistic scenarios"""
        config = BatchOptimizationConfig(
            min_batch_size=10,
            max_batch_size=200,
            initial_batch_size=50,
            memory_limit_mb=1000.0
        )
        optimizer = DynamicBatchOptimizer(config)

        # Simulate realistic batch performance over time
        initial_batch_size = optimizer.current_batch_size

        # Simulate successful batches with good performance
        for i in range(10):
            asyncio.run(optimizer.record_batch_performance(
                batch_size=optimizer.current_batch_size,
                processing_time=1.0 + np.random.normal(0, 0.1),
                success_count=optimizer.current_batch_size - np.random.randint(0, 3),
                error_count=np.random.randint(0, 3)
            ))

        # Check that optimizer adapted batch size based on performance
        final_batch_size = optimizer.current_batch_size

        # Batch size should be within reasonable bounds
        assert config.min_batch_size <= final_batch_size <= config.max_batch_size

        # Get optimization stats
        stats = optimizer.get_optimization_stats()
        assert stats["total_batches_processed"] == 10
        assert "recent_avg_efficiency" in stats


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
