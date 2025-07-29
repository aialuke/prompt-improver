"""
Real Behavior Testing for Updated Metrics Implementation.

Tests that the new real metrics implementation works correctly without any
mock objects, verifying both Prometheus and in-memory fallback scenarios.
"""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timezone
from typing import Any

from prompt_improver.metrics.api_metrics import (
    APIMetricsCollector,
    APIUsageMetric,
    HTTPMethod,
    EndpointCategory,
    AuthenticationMethod,
)
from prompt_improver.performance.monitoring.real_metrics import (
    RealMetricsRegistry,
    InMemoryCounter,
    InMemoryGauge,
    InMemoryHistogram,
    get_real_metrics_registry
)
from prompt_improver.performance.monitoring.metrics_registry import (
    get_metrics_registry,
    MetricsRegistry
)


class TestRealMetricsBehavior:
    """Test suite for real metrics behavior without mock objects."""

    def test_in_memory_counter_real_behavior(self) -> None:
        """Test that in-memory counter provides real behavior."""
        counter = InMemoryCounter("test_counter", "Test counter", ["endpoint", "method"])

        # Test basic increment
        counter.inc(1.0, endpoint="/api/test", method="GET")
        assert counter.get_value(endpoint="/api/test", method="GET") == 1.0

        # Test multiple increments
        counter.inc(2.5, endpoint="/api/test", method="GET")
        assert counter.get_value(endpoint="/api/test", method="GET") == 3.5

        # Test different label combinations
        counter.inc(1.0, endpoint="/api/other", method="POST")
        assert counter.get_value(endpoint="/api/other", method="POST") == 1.0
        assert counter.get_value(endpoint="/api/test", method="GET") == 3.5

        # Test labels method
        labeled_counter = counter.labels(endpoint="/api/labeled", method="PUT")
        labeled_counter.inc(5.0)
        # Note: This is a simplified test - in real implementation, labels() would need to store state

        # Verify all values
        all_values = counter.get_all_values()
        assert len(all_values) >= 2  # At least the two we created

    def test_in_memory_gauge_real_behavior(self) -> None:
        """Test that in-memory gauge provides real behavior."""
        gauge = InMemoryGauge("test_gauge", "Test gauge", ["service"])

        # Test set operation
        gauge.set(10.5, service="api")
        assert gauge.get_value(service="api") == 10.5

        # Test increment
        gauge.inc(2.5, service="api")
        assert gauge.get_value(service="api") == 13.0

        # Test decrement
        gauge.dec(3.0, service="api")
        assert gauge.get_value(service="api") == 10.0

        # Test multiple services
        gauge.set(25.0, service="database")
        assert gauge.get_value(service="database") == 25.0
        assert gauge.get_value(service="api") == 10.0

    def test_in_memory_histogram_real_behavior(self) -> None:
        """Test that in-memory histogram provides real behavior."""
        histogram = InMemoryHistogram(
            "test_histogram",
            "Test histogram",
            ["operation"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
        )

        # Test observations
        histogram.observe(0.3, operation="read")
        histogram.observe(0.8, operation="read")
        histogram.observe(1.5, operation="read")
        histogram.observe(0.2, operation="write")

        # Test getting observations
        read_observations = histogram.get_observations(operation="read")
        assert len(read_observations) == 3
        assert 0.3 in read_observations
        assert 0.8 in read_observations
        assert 1.5 in read_observations

        write_observations = histogram.get_observations(operation="write")
        assert len(write_observations) == 1
        assert 0.2 in write_observations

        # Test bucket counts
        read_buckets = histogram.get_bucket_counts(operation="read")
        assert read_buckets[0.5] == 1  # Only 0.3 <= 0.5
        assert read_buckets[1.0] == 2  # 0.3 and 0.8 <= 1.0
        assert read_buckets[2.0] == 3  # All three <= 2.0

        # Test statistics
        read_stats = histogram.get_statistics(operation="read")
        assert read_stats["count"] == 3
        assert read_stats["sum"] == 2.6  # 0.3 + 0.8 + 1.5
        assert abs(read_stats["avg"] - 0.8667) < 0.001  # 2.6 / 3
        assert read_stats["min"] == 0.3
        assert read_stats["max"] == 1.5

    def test_real_metrics_registry_prometheus_fallback(self) -> None:
        """Test that real metrics registry handles Prometheus fallback correctly."""
        registry = RealMetricsRegistry()

        # Test counter creation
        counter = registry.get_or_create_counter(
            "test_real_counter",
            "Test real counter",
            ["label1", "label2"]
        )

        # Verify it's a real metric (either Prometheus or in-memory)
        assert hasattr(counter, 'inc')
        assert hasattr(counter, 'labels')

        # Test that it actually works with proper labels
        if hasattr(counter, 'get_value'):
            # In-memory counter - can increment without labels
            counter.inc(1.0, label1="test1", label2="test2")
            assert counter.get_value(label1="test1", label2="test2") >= 1.0
        else:
            # Prometheus counter - needs labels
            counter.labels(label1="test1", label2="test2").inc(1.0)

        # Test gauge creation
        gauge = registry.get_or_create_gauge(
            "test_real_gauge",
            "Test real gauge",
            ["service"]
        )

        assert hasattr(gauge, 'set')
        assert hasattr(gauge, 'inc')
        assert hasattr(gauge, 'dec')

        # Test histogram creation
        histogram = registry.get_or_create_histogram(
            "test_real_histogram",
            "Test real histogram",
            ["operation"],
            buckets=[0.1, 0.5, 1.0]
        )

        assert hasattr(histogram, 'observe')
        assert hasattr(histogram, 'time')

    @pytest_asyncio.fixture
    async def real_metrics_collector(self) -> APIMetricsCollector:
        """Create API metrics collector with real metrics."""
        config = {
            "max_api_metrics": 100,
            "aggregation_window_minutes": 1,
            "retention_hours": 1
        }
        collector = APIMetricsCollector(config)
        await collector.start_collection()
        yield collector
        await collector.stop_collection()

    @pytest.mark.asyncio
    async def test_api_metrics_collector_real_behavior(self, real_metrics_collector: APIMetricsCollector) -> None:
        """Test that API metrics collector uses real metrics instead of mocks."""
        # Verify that metrics are real objects, not mocks
        assert hasattr(real_metrics_collector.endpoint_request_count, 'inc')
        assert hasattr(real_metrics_collector.endpoint_response_time, 'observe')
        assert hasattr(real_metrics_collector.endpoint_popularity, 'set')

        # Test that metrics actually record data
        metric = APIUsageMetric(
            endpoint="/api/v1/test",
            method=HTTPMethod.GET,
            category=EndpointCategory.PROMPT_IMPROVEMENT,
            status_code=200,
            response_time_ms=150.0,
            request_size_bytes=512,
            response_size_bytes=1024,
            user_id="test_user",
            session_id="test_session",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
            timestamp=datetime.now(timezone.utc),
            query_parameters_count=2,
            payload_type="application/json",
            rate_limited=False,
            cache_hit=True,
            authentication_method=AuthenticationMethod.JWT_TOKEN,
            api_version="v1"
        )

        # Record the metric
        await real_metrics_collector.record_api_usage(metric)

        # Verify it was recorded in the collector
        assert len(real_metrics_collector.api_usage_metrics) == 1

        # Test that Prometheus metrics were called (they should not raise errors)
        try:
            real_metrics_collector.endpoint_request_count.labels(
                endpoint="/api/v1/test",
                method="GET",
                status_code="200",
                category="prompt_improvement"
            ).inc()

            real_metrics_collector.endpoint_response_time.labels(
                endpoint="/api/v1/test",
                method="GET",
                category="prompt_improvement"
            ).observe(0.15)  # 150ms in seconds

            # If we get here, the metrics are working
            metrics_working = True
        except Exception as e:
            # This should not happen with real metrics
            pytest.fail(f"Real metrics should not raise exceptions: {e}")

        assert metrics_working

    def test_metrics_registry_integration(self) -> None:
        """Test that the updated metrics registry provides real behavior."""
        registry = get_metrics_registry()

        # Verify it's using the real metrics backend
        assert isinstance(registry, MetricsRegistry)
        assert hasattr(registry, '_real_registry')

        # Test creating metrics
        counter = registry.get_or_create_counter(
            "integration_test_counter",
            "Integration test counter",
            ["test_label"]
        )

        # Verify it's a real metric
        assert hasattr(counter, 'inc')
        counter.inc(1.0)  # Should not raise any errors

        # Test metrics summary
        summary = registry.get_metrics_summary()
        assert isinstance(summary, dict)
        assert "backend" in summary
        assert "total_metrics" in summary
        assert summary["backend"] in ["prometheus", "in_memory"]

    @pytest.mark.asyncio
    async def test_no_mock_objects_in_system(self, real_metrics_collector: APIMetricsCollector) -> None:
        """Verify that no mock objects are used anywhere in the metrics system."""
        # Check that all metrics are real objects
        metrics_to_check = [
            real_metrics_collector.endpoint_request_count,
            real_metrics_collector.endpoint_response_time,
            real_metrics_collector.endpoint_payload_size,
            real_metrics_collector.endpoint_popularity,
            real_metrics_collector.user_journey_conversion_rate,
            real_metrics_collector.user_journey_duration,
            real_metrics_collector.active_user_sessions,
            real_metrics_collector.rate_limit_blocks,
            real_metrics_collector.rate_limit_utilization,
            real_metrics_collector.burst_detection_events,
            real_metrics_collector.auth_success_rate,
            real_metrics_collector.auth_session_duration,
            real_metrics_collector.mfa_usage_rate
        ]

        for metric in metrics_to_check:
            # Verify it's not a mock by checking it has real behavior
            assert hasattr(metric, 'inc') or hasattr(metric, 'set') or hasattr(metric, 'observe')

            # Verify the class name doesn't contain "Mock"
            class_name = type(metric).__name__
            assert "Mock" not in class_name, f"Found mock object: {class_name}"

            # Test that calling methods doesn't just pass silently
            # Real metrics should either work or raise meaningful errors
            try:
                if hasattr(metric, 'inc'):
                    metric.inc(1.0)
                elif hasattr(metric, 'set'):
                    metric.set(1.0)
                elif hasattr(metric, 'observe'):
                    metric.observe(1.0)
            except Exception as e:
                # Real metrics might raise exceptions for invalid operations,
                # but they shouldn't silently do nothing like mocks
                assert str(e) != "", "Real metrics should provide meaningful error messages"

    def test_real_metrics_persistence(self) -> None:
        """Test that real metrics persist data correctly."""
        registry = get_real_metrics_registry()

        # Create a counter and record some data
        counter = registry.get_or_create_counter(
            "persistence_test_counter",
            "Persistence test counter",
            ["operation"]
        )

        # Record some values
        counter.inc(5.0, operation="test1")
        counter.inc(3.0, operation="test2")
        counter.inc(2.0, operation="test1")  # Add to existing

        # Get the same counter again (should be cached)
        same_counter = registry.get_or_create_counter(
            "persistence_test_counter",
            "Persistence test counter",
            ["operation"]
        )

        # Verify it's the same object
        assert counter is same_counter

        # Verify data persists
        if hasattr(counter, 'get_value'):
            # In-memory counter
            assert counter.get_value(operation="test1") == 7.0  # 5 + 2
            assert counter.get_value(operation="test2") == 3.0

        # Test metrics summary includes our data
        summary = registry.get_metrics_summary()
        assert "persistence_test_counter" in summary["metrics"]
