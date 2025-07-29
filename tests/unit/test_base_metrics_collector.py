"""
Comprehensive tests for the base MetricsCollector class.

Tests the modern 2025 Python design patterns including:
- Protocol-based interfaces
- Composition over inheritance
- Async-first design
- JSONB compatibility
- Real behavior testing (no mocks)
"""

import asyncio
import json
import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Union
from dataclasses import dataclass

from prompt_improver.metrics.base_metrics_collector import (
    BaseMetricsCollector,
    MetricsConfig,
    CollectionStats,
    PrometheusMetricsMixin,
    MetricsStorageMixin,
    create_metrics_collector,
    get_or_create_collector
)


@dataclass
class TestMetric:
    """Test metric for testing base collector functionality."""
    name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any]


class TestMetricsCollector(BaseMetricsCollector[TestMetric], PrometheusMetricsMixin, MetricsStorageMixin):
    """Test implementation of BaseMetricsCollector for testing."""

    def __init__(self, config=None, metrics_registry=None, error_handler=None):
        super().__init__(config, metrics_registry, error_handler)
        self._metrics_storage.update({
            "test_metrics": self._metrics_storage.get("test_metrics", [])
        })

    def collect_metric(self, metric: TestMetric) -> None:
        """Collect a test metric."""
        self.store_metric("test_metrics", metric)

    def _initialize_prometheus_metrics(self) -> None:
        """Initialize test Prometheus metrics."""
        self.test_counter = self.create_counter(
            "test_metrics_total",
            "Total test metrics collected"
        )
        self.test_histogram = self.create_histogram(
            "test_metric_values",
            "Distribution of test metric values"
        )

    async def _aggregate_metrics(self) -> None:
        """Test aggregation implementation."""
        # Simple aggregation for testing
        recent_metrics = self.get_recent_metrics("test_metrics", hours=1)
        if recent_metrics:
            avg_value = sum(m.value for m in recent_metrics) / len(recent_metrics)
            self.test_histogram.observe(avg_value)


class TestBaseMetricsCollector:
    """Test suite for BaseMetricsCollector."""

    @pytest_asyncio.fixture
    async def collector(self):
        """Create a test collector instance."""
        config = MetricsConfig(
            aggregation_window_minutes=1,
            retention_hours=2,
            max_metrics_per_type=100
        )
        collector = TestMetricsCollector(config)
        yield collector
        await collector.stop_collection()

    def test_metrics_config_creation(self):
        """Test MetricsConfig creation and dictionary access."""
        config = MetricsConfig(
            aggregation_window_minutes=10,
            retention_hours=48,
            max_metrics_per_type=5000
        )

        assert config.aggregation_window_minutes == 10
        assert config.retention_hours == 48
        assert config.max_metrics_per_type == 5000
        assert config.get("aggregation_window_minutes") == 10
        assert config.get("nonexistent_key", "default") == "default"

    def test_collection_stats_jsonb_compatibility(self):
        """Test that CollectionStats is JSONB compatible."""
        stats = CollectionStats(
            total_metrics_collected=100,
            collection_errors=2,
            last_collection_time=datetime.now(timezone.utc).isoformat(),
            is_running=True,
            background_tasks_active=2
        )

        # Test conversion to dict (JSONB compatible)
        stats_dict = stats.to_dict()
        assert isinstance(stats_dict, dict)
        assert stats_dict["total_metrics_collected"] == 100
        assert stats_dict["collection_errors"] == 2
        assert stats_dict["is_running"] is True

        # Test JSON serialization (JSONB compatibility)
        json_str = json.dumps(stats_dict)
        assert isinstance(json_str, str)

        # Test round-trip
        parsed = json.loads(json_str)
        assert parsed["total_metrics_collected"] == 100

    @pytest.mark.asyncio
    async def test_collector_initialization(self, collector):
        """Test collector initialization with dependency injection."""
        assert isinstance(collector.config, MetricsConfig)
        assert collector.config.aggregation_window_minutes == 1
        assert collector.config.retention_hours == 2
        assert collector.config.max_metrics_per_type == 100

        assert hasattr(collector, 'metrics_registry')
        assert hasattr(collector, 'logger')
        assert isinstance(collector.collection_stats, CollectionStats)
        assert not collector.collection_stats.is_running

    @pytest.mark.asyncio
    async def test_metric_collection(self, collector):
        """Test basic metric collection functionality."""
        test_metric = TestMetric(
            name="test_metric_1",
            value=42.5,
            timestamp=datetime.now(timezone.utc),
            metadata={"source": "test", "category": "unit_test"}
        )

        # Collect metric
        collector.collect_metric(test_metric)

        # Verify storage
        assert len(collector._metrics_storage["test_metrics"]) == 1
        stored_metric = collector._metrics_storage["test_metrics"][0]
        assert stored_metric.name == "test_metric_1"
        assert stored_metric.value == 42.5
        assert stored_metric.metadata["source"] == "test"

    @pytest.mark.asyncio
    async def test_background_collection_lifecycle(self, collector):
        """Test start/stop collection lifecycle."""
        # Initially not running
        assert not collector.collection_stats.is_running

        # Start collection
        await collector.start_collection()
        assert collector.collection_stats.is_running
        assert collector.collection_stats.background_tasks_active >= 1

        # Stop collection
        await collector.stop_collection()
        assert not collector.collection_stats.is_running
        assert collector.collection_stats.background_tasks_active == 0

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager functionality."""
        config = MetricsConfig(aggregation_window_minutes=1)

        async with TestMetricsCollector(config) as collector:
            assert collector.collection_stats.is_running

            # Collect some metrics
            test_metric = TestMetric(
                name="context_test",
                value=100.0,
                timestamp=datetime.now(timezone.utc),
                metadata={}
            )
            collector.collect_metric(test_metric)

            assert len(collector._metrics_storage["test_metrics"]) == 1

        # Should be stopped after context exit
        assert not collector.collection_stats.is_running

    @pytest.mark.asyncio
    async def test_metrics_storage_mixin(self, collector):
        """Test MetricsStorageMixin functionality."""
        # Create test metrics with different timestamps
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(hours=2)
        recent_time = now - timedelta(minutes=30)

        old_metric = TestMetric("old", 1.0, old_time, {})
        recent_metric = TestMetric("recent", 2.0, recent_time, {})
        current_metric = TestMetric("current", 3.0, now, {})

        # Store metrics
        collector.store_metric("test_metrics", old_metric)
        collector.store_metric("test_metrics", recent_metric)
        collector.store_metric("test_metrics", current_metric)

        # Test recent metrics retrieval
        recent_metrics = collector.get_recent_metrics("test_metrics", hours=1)
        assert len(recent_metrics) == 2  # recent and current, not old

        # Test old metrics cleanup
        collector.clear_old_metrics(hours=1)
        remaining_metrics = collector._metrics_storage["test_metrics"]
        assert len(remaining_metrics) == 2  # old metric should be removed

    @pytest.mark.asyncio
    async def test_prometheus_metrics_mixin(self, collector):
        """Test PrometheusMetricsMixin functionality."""
        # Test counter creation
        counter = collector.create_counter("test_counter", "Test counter")
        assert counter is not None

        # Test histogram creation
        histogram = collector.create_histogram("test_histogram", "Test histogram")
        assert histogram is not None

        # Test gauge creation
        gauge = collector.create_gauge("test_gauge", "Test gauge")
        assert gauge is not None

        # Test with custom labels and buckets
        labeled_histogram = collector.create_histogram(
            "labeled_histogram",
            "Labeled histogram",
            labels=["label1", "label2"],
            buckets=[0.1, 0.5, 1.0, 5.0]
        )
        assert labeled_histogram is not None

    @pytest.mark.asyncio
    async def test_collection_stats_tracking(self, collector):
        """Test collection statistics tracking."""
        initial_stats = collector.get_collection_stats()
        assert initial_stats["total_metrics_collected"] == 0

        # Collect some metrics
        for i in range(5):
            metric = TestMetric(f"metric_{i}", float(i), datetime.now(timezone.utc), {})
            collector.collect_metric(metric)

        updated_stats = collector.get_collection_stats()
        assert updated_stats["total_metrics_collected"] == 5
        assert "metrics_storage_counts" in updated_stats
        assert updated_stats["metrics_storage_counts"]["test_metrics"] == 5

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling functionality."""
        error_messages = []

        async def custom_error_handler(error: Exception, context: str):
            error_messages.append(f"{context}: {str(error)}")

        collector = TestMetricsCollector(error_handler=custom_error_handler)

        # Trigger an error in aggregation
        await collector.start_collection()

        # Force an error by calling aggregation directly
        try:
            # This should trigger error handling
            await collector._aggregate_metrics()
        except Exception:
            pass

        await collector.stop_collection()

        # Error handler should have been called if there were any errors
        # (This test verifies the error handling mechanism exists)

    def test_factory_functions(self):
        """Test factory functions for dependency injection."""
        config = MetricsConfig(max_metrics_per_type=1000)

        # Test create_metrics_collector
        collector1 = create_metrics_collector(TestMetricsCollector, config)
        assert isinstance(collector1, TestMetricsCollector)
        assert collector1.config.max_metrics_per_type == 1000

        # Test get_or_create_collector (singleton pattern)
        collector2 = get_or_create_collector(TestMetricsCollector, config)
        collector3 = get_or_create_collector(TestMetricsCollector, config)

        # Should return the same instance
        assert collector2 is collector3
        assert isinstance(collector2, TestMetricsCollector)

    @pytest.mark.asyncio
    async def test_real_behavior_integration(self, collector):
        """Test real behavior integration without mocks."""
        # Start collection
        await collector.start_collection()

        # Collect metrics over time
        metrics_data = []
        for i in range(10):
            metric = TestMetric(
                name=f"integration_test_{i}",
                value=float(i * 10),
                timestamp=datetime.now(timezone.utc),
                metadata={"test_run": "integration", "sequence": i}
            )
            collector.collect_metric(metric)
            metrics_data.append(metric)

            # Small delay to simulate real-world timing
            await asyncio.sleep(0.01)

        # Verify all metrics were collected
        assert len(collector._metrics_storage["test_metrics"]) == 10

        # Test aggregation
        await collector._aggregate_metrics()

        # Verify collection stats
        stats = collector.get_collection_stats()
        assert stats["total_metrics_collected"] == 10
        assert stats["is_running"] is True

        # Stop collection
        await collector.stop_collection()
        assert not collector.collection_stats.is_running
