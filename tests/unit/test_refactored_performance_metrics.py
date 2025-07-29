"""
Test suite for refactored PerformanceMetricsCollector.

Tests the fully refactored PerformanceMetricsCollector to ensure it works correctly
with the new base class architecture and eliminates all redundant code patterns.
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from src.prompt_improver.metrics.performance_metrics import (
    PerformanceMetricsCollector,
    RequestPipelineMetric,
    DatabasePerformanceMetric,
    CachePerformanceMetric,
    ExternalAPIMetric,
    PipelineStage,
    DatabaseOperation,
    CacheType,
    ExternalAPIType,
    get_performance_metrics_collector
)
from src.prompt_improver.metrics.base_metrics_collector import MetricsConfig


class TestRefactoredPerformanceMetricsCollector:
    """Test the refactored PerformanceMetricsCollector implementation."""

    @pytest.fixture
    def config(self) -> MetricsConfig:
        """Create test configuration."""
        return MetricsConfig(
            enable_prometheus=False,  # Disable for testing
            max_metrics_per_type=1000,
            aggregation_window_minutes=5,
            retention_hours=24
        )

    @pytest.fixture
    def collector(self, config: MetricsConfig) -> PerformanceMetricsCollector:
        """Create test collector instance."""
        return PerformanceMetricsCollector(config)

    def test_initialization_with_base_class(self, collector: PerformanceMetricsCollector):
        """Test that collector initializes correctly with base class."""
        # Test base class initialization
        assert collector.config is not None
        assert collector.logger is not None
        assert collector._metrics_storage is not None

        # Test performance-specific initialization
        assert "pipeline" in collector._metrics_storage
        assert "database" in collector._metrics_storage
        assert "cache" in collector._metrics_storage
        assert "external_api" in collector._metrics_storage

        # Test performance-specific stats
        assert collector.performance_stats is not None
        assert collector.performance_stats["pipeline_stages_tracked"] == 0
        assert collector.performance_stats["database_queries_tracked"] == 0

    def test_collect_pipeline_metric(self, collector: PerformanceMetricsCollector):
        """Test collecting pipeline metrics using new base class storage."""
        metric = RequestPipelineMetric(
            request_id="test-123",
            stage=PipelineStage.VALIDATION,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(milliseconds=100),
            duration_ms=100.0,
            memory_usage_mb=50.0,
            cpu_usage_percent=25.0,
            success=True,
            endpoint="/api/test",
            queue_time_ms=10.0,
            error_type=None
        )

        # Collect metric
        collector.collect_metric(metric)

        # Verify storage using new base class patterns
        pipeline_metrics = collector.get_metrics_by_type("pipeline")
        assert len(pipeline_metrics) == 1
        assert pipeline_metrics[0] == metric

        # Verify stats tracking
        assert collector.performance_stats["pipeline_stages_tracked"] == 1

    def test_collect_database_metric(self, collector: PerformanceMetricsCollector):
        """Test collecting database metrics using new base class storage."""
        metric = DatabasePerformanceMetric(
            timestamp=datetime.now(timezone.utc),
            operation_type=DatabaseOperation.SELECT,
            table_name="test_table",
            execution_time_ms=50.0,
            rows_affected=100,
            query_plan_type="index_scan",
            lock_time_ms=5.0,
            active_connections=10,
            connection_pool_size=20,
            query_hash="abc123",
            success=True
        )

        # Collect metric
        collector.collect_metric(metric)

        # Verify storage using new base class patterns
        database_metrics = collector.get_metrics_by_type("database")
        assert len(database_metrics) == 1
        assert database_metrics[0] == metric

        # Verify stats tracking
        assert collector.performance_stats["database_queries_tracked"] == 1

    def test_collect_cache_metric(self, collector: PerformanceMetricsCollector):
        """Test collecting cache metrics using new base class storage."""
        metric = CachePerformanceMetric(
            timestamp=datetime.now(timezone.utc),
            cache_type=CacheType.APPLICATION,
            operation="get",
            key="test_key",
            hit=True,
            response_time_ms=5.0,
            serialization_time_ms=1.0,
            eviction_triggered=False
        )

        # Collect metric
        collector.collect_metric(metric)

        # Verify storage using new base class patterns
        cache_metrics = collector.get_metrics_by_type("cache")
        assert len(cache_metrics) == 1
        assert cache_metrics[0] == metric

        # Verify stats tracking
        assert collector.performance_stats["cache_operations_tracked"] == 1

    def test_collect_external_api_metric(self, collector: PerformanceMetricsCollector):
        """Test collecting external API metrics using new base class storage."""
        metric = ExternalAPIMetric(
            timestamp=datetime.now(timezone.utc),
            api_type=ExternalAPIType.REST,
            endpoint="/external/api",
            method="GET",
            status_code=200,
            response_time_ms=150.0,
            request_size_bytes=1024,
            response_size_bytes=2048,
            success=True,
            circuit_breaker_state="closed"
        )

        # Collect metric
        collector.collect_metric(metric)

        # Verify storage using new base class patterns
        api_metrics = collector.get_metrics_by_type("external_api")
        assert len(api_metrics) == 1
        assert api_metrics[0] == metric

        # Verify stats tracking
        assert collector.performance_stats["external_api_calls_tracked"] == 1

    def test_slow_query_detection(self, collector: PerformanceMetricsCollector):
        """Test slow query detection using new configuration patterns."""
        # Create a slow query (exceeds threshold)
        slow_metric = DatabasePerformanceMetric(
            timestamp=datetime.now(timezone.utc),
            operation_type=DatabaseOperation.SELECT,
            table_name="large_table",
            execution_time_ms=2000.0,  # Exceeds default 1000ms threshold
            rows_affected=1000000,
            query_plan_type="full_scan",
            lock_time_ms=100.0,
            active_connections=15,
            connection_pool_size=20,
            query_hash="slow123",
            success=True
        )

        # Collect metric
        collector.collect_metric(slow_metric)

        # Verify slow query detection
        assert collector.performance_stats["slow_queries_detected"] == 1

    def test_cache_miss_detection(self, collector: PerformanceMetricsCollector):
        """Test cache miss detection using new stats patterns."""
        # Create a cache miss
        miss_metric = CachePerformanceMetric(
            timestamp=datetime.now(timezone.utc),
            cache_type=CacheType.ML_MODEL,
            operation="get",
            key="model_cache_key",
            hit=False,  # Cache miss
            response_time_ms=50.0,
            serialization_time_ms=10.0,
            eviction_triggered=False
        )

        # Collect metric
        collector.collect_metric(miss_metric)

        # Verify cache miss detection
        assert collector.performance_stats["cache_misses_detected"] == 1

    @pytest.mark.asyncio
    async def test_cleanup_old_metrics(self, collector: PerformanceMetricsCollector):
        """Test cleanup of old metrics using new base class storage."""
        # Create old metrics
        old_time = datetime.now(timezone.utc) - timedelta(hours=25)  # Older than retention
        recent_time = datetime.now(timezone.utc) - timedelta(hours=1)  # Within retention

        old_metric = RequestPipelineMetric(
            request_id="old-123",
            stage=PipelineStage.PROCESSING,
            start_time=old_time,
            end_time=old_time + timedelta(milliseconds=100),
            duration_ms=100.0,
            memory_usage_mb=30.0,
            cpu_usage_percent=20.0,
            success=True,
            endpoint="/api/old",
            queue_time_ms=5.0,
            error_type=None
        )

        recent_metric = RequestPipelineMetric(
            request_id="recent-123",
            stage=PipelineStage.PROCESSING,
            start_time=recent_time,
            end_time=recent_time + timedelta(milliseconds=100),
            duration_ms=100.0,
            memory_usage_mb=30.0,
            cpu_usage_percent=20.0,
            success=True,
            endpoint="/api/recent",
            queue_time_ms=5.0,
            error_type=None
        )

        # Collect both metrics
        collector.collect_metric(old_metric)
        collector.collect_metric(recent_metric)

        # Verify both are stored
        assert len(collector.get_metrics_by_type("pipeline")) == 2

        # Run cleanup
        await collector._cleanup_old_metrics(datetime.now(timezone.utc))

        # Verify only recent metric remains
        remaining_metrics = collector.get_metrics_by_type("pipeline")
        assert len(remaining_metrics) == 1
        assert remaining_metrics[0].request_id == "recent-123"

    def test_factory_function(self, config: MetricsConfig):
        """Test the factory function works with new base class patterns."""
        collector = get_performance_metrics_collector(config)

        # Verify it's the correct type
        assert isinstance(collector, PerformanceMetricsCollector)

        # Verify it has base class functionality
        assert hasattr(collector, '_metrics_storage')
        assert hasattr(collector, 'collect_metric')
        assert hasattr(collector, 'get_metrics_by_type')

    def test_get_collection_stats(self, collector: PerformanceMetricsCollector):
        """Test collection stats using new base class patterns."""
        # Collect some metrics
        pipeline_metric = RequestPipelineMetric(
            request_id="stats-test",
            stage=PipelineStage.VALIDATION,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(milliseconds=50),
            duration_ms=50.0,
            memory_usage_mb=25.0,
            cpu_usage_percent=15.0,
            success=True,
            endpoint="/api/stats",
            queue_time_ms=5.0,
            error_type=None
        )

        collector.collect_metric(pipeline_metric)

        # Get stats
        stats = collector.get_collection_stats()

        # Verify stats structure
        assert "total_metrics_collected" in stats
        assert "pipeline_stages_tracked" in stats
        assert stats["pipeline_stages_tracked"] == 1
        assert stats["total_metrics_collected"] >= 1


if __name__ == "__main__":
    pytest.main([__file__])
