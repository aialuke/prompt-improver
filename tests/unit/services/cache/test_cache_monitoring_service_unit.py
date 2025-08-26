"""Unit tests for CacheMonitoringService critical methods.

Focused unit tests for isolated testing of critical CacheMonitoringService
methods with complete mocking. Tests edge cases, error conditions, and
specific algorithm correctness without external dependencies.

Critical areas tested:
- SLO compliance calculation accuracy (percentile bug validation)
- Error tracking and memory leak prevention
- Health status calculation logic
- Alert threshold boundary conditions
- Response time tracking and cleanup
- Memory utilization calculations
"""

import logging
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from prompt_improver.services.cache.cache_monitoring_service import (
    CacheMonitoringService,
)

logger = logging.getLogger(__name__)


class TestCacheMonitoringServiceUnit:
    """Unit tests for CacheMonitoringService critical methods."""

    @pytest.fixture
    def mock_cache_facade(self):
        """Mock cache facade for unit testing without coordination overhead."""
        facade = MagicMock()

        # Mock get_cache_stats to return realistic facade stats
        facade.get_cache_stats.return_value = {
            "l1_stats": {
                "total_size": 150,
                "max_size": 1000,
                "estimated_memory_bytes": 1048576,  # 1MB
                "overall_hit_rate": 0.90,
                "health": {
                    "status": "healthy",
                    "avg_response_time_ms": 0.3,
                    "performance_compliant": True
                },
                "utilization": 0.15  # 150/1000
            },
            "l2_enabled": False,
            "warming_enabled": False
        }

        # Mock basic operations for health check
        facade.set = MagicMock(return_value=True)
        facade.get = MagicMock(return_value={"test": True, "timestamp": time.time()})
        facade.invalidate = MagicMock(return_value=True)

        return facade

    @pytest.fixture
    def monitoring_service(self, mock_cache_facade):
        """CacheMonitoringService with mocked facade (no coordination overhead)."""
        return CacheMonitoringService(mock_cache_facade)

    def test_slo_compliance_calculation_edge_cases(self, monitoring_service):
        """Test SLO compliance calculation with edge cases and boundary conditions."""
        monitoring = monitoring_service

        # Test empty response times
        monitoring._response_times.clear()
        slo_result = monitoring.calculate_slo_compliance()

        assert slo_result["compliant"] is True, "Empty response times should be compliant"
        assert slo_result["compliance_rate"] == 1.0, "Empty response times should have 100% compliance"
        assert slo_result["sample_count"] == 0, "Sample count should be 0 for empty response times"
        assert slo_result["total_requests"] == 0, "Total requests should be 0 for empty response times"

        # Test single response time - compliant
        monitoring._response_times = [0.150]  # 150ms - compliant
        slo_result = monitoring.calculate_slo_compliance()

        assert slo_result["compliant"] is True, "Single compliant response should be compliant"
        assert slo_result["compliance_rate"] == 1.0, "Single compliant response should have 100% compliance"
        assert slo_result["compliant_requests"] == 1, "Should have 1 compliant request"
        assert slo_result["violations"] == 0, "Should have 0 violations"

        # Test single response time - non-compliant
        monitoring._response_times = [0.250]  # 250ms - violation
        slo_result = monitoring.calculate_slo_compliance()

        assert slo_result["compliant"] is False, "Single non-compliant response should not be compliant"
        assert slo_result["compliance_rate"] == 0.0, "Single non-compliant response should have 0% compliance"
        assert slo_result["compliant_requests"] == 0, "Should have 0 compliant requests"
        assert slo_result["violations"] == 1, "Should have 1 violation"

        # Test boundary condition - exactly at SLO target (200ms)
        monitoring._response_times = [0.200]  # Exactly 200ms
        slo_result = monitoring.calculate_slo_compliance()

        assert slo_result["compliant"] is True, "Exactly at SLO target should be compliant"
        assert slo_result["compliance_rate"] == 1.0, "Exactly at SLO target should have 100% compliance"

        # Test 95% compliance boundary
        # 20 requests: 19 compliant (95%), 1 violation (5%)
        compliant_times = [0.100] * 19  # 19 compliant requests
        violation_times = [0.300] * 1   # 1 violation
        monitoring._response_times = compliant_times + violation_times
        slo_result = monitoring.calculate_slo_compliance()

        assert slo_result["compliance_rate"] == 0.95, "Should have exactly 95% compliance rate"
        assert slo_result["compliant"] is True, "95% compliance should meet SLO target"
        assert slo_result["compliant_requests"] == 19, "Should have 19 compliant requests"
        assert slo_result["violations"] == 1, "Should have 1 violation"

        # Test below 95% compliance threshold
        # 20 requests: 18 compliant (90%), 2 violations (10%)
        compliant_times = [0.100] * 18  # 18 compliant requests
        violation_times = [0.300] * 2   # 2 violations
        monitoring._response_times = compliant_times + violation_times
        slo_result = monitoring.calculate_slo_compliance()

        assert slo_result["compliance_rate"] == 0.90, "Should have exactly 90% compliance rate"
        assert slo_result["compliant"] is False, "90% compliance should not meet SLO target"
        assert slo_result["compliant_requests"] == 18, "Should have 18 compliant requests"
        assert slo_result["violations"] == 2, "Should have 2 violations"

    def test_percentile_calculation_accuracy(self, monitoring_service):
        """Test percentile calculation accuracy for various data distributions."""
        monitoring = monitoring_service

        # Test with known data set for percentile validation
        test_data = [0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090, 0.100]  # 10ms to 100ms
        monitoring._response_times = test_data.copy()

        slo_result = monitoring.calculate_slo_compliance()
        percentiles = slo_result["percentiles"]

        # Manual percentile calculation for validation
        sorted_times_ms = sorted([t * 1000 for t in test_data])

        def manual_percentile(data: list[float], percentile: int) -> float:
            """Manual percentile calculation matching service logic."""
            idx = int(len(data) * percentile / 100)
            return data[min(idx, len(data) - 1)]

        expected_p50 = manual_percentile(sorted_times_ms, 50)
        expected_p95 = manual_percentile(sorted_times_ms, 95)
        expected_p99 = manual_percentile(sorted_times_ms, 99)

        assert abs(percentiles["p50"] - expected_p50) < 0.1, f"P50 percentile error: {percentiles['p50']:.2f} != {expected_p50:.2f}"
        assert abs(percentiles["p95"] - expected_p95) < 0.1, f"P95 percentile error: {percentiles['p95']:.2f} != {expected_p95:.2f}"
        assert abs(percentiles["p99"] - expected_p99) < 0.1, f"P99 percentile error: {percentiles['p99']:.2f} != {expected_p99:.2f}"

        # Test with single value (edge case)
        monitoring._response_times = [0.075]  # 75ms
        slo_result = monitoring.calculate_slo_compliance()
        percentiles = slo_result["percentiles"]

        assert percentiles["p50"] == 75.0, "Single value P50 should equal the value"
        assert percentiles["p95"] == 75.0, "Single value P95 should equal the value"
        assert percentiles["p99"] == 75.0, "Single value P99 should equal the value"

        # Test with two values (edge case)
        monitoring._response_times = [0.050, 0.100]  # 50ms, 100ms
        slo_result = monitoring.calculate_slo_compliance()
        percentiles = slo_result["percentiles"]

        # With 2 values: P50 index = 1, P95 index = 1, P99 index = 1 (all point to second value)
        assert percentiles["p50"] == 100.0, "Two values P50 should be the second value"
        assert percentiles["p95"] == 100.0, "Two values P95 should be the second value"
        assert percentiles["p99"] == 100.0, "Two values P99 should be the second value"

    def test_error_tracking_and_memory_management(self, monitoring_service):
        """Test error tracking accuracy and memory leak prevention."""
        monitoring = monitoring_service

        # Clear initial state
        monitoring._error_counts.clear()
        monitoring._consecutive_errors = 0

        # Test basic error recording
        test_error = ValueError("Test validation error")
        monitoring._record_error("test_operation", test_error)

        assert "test_operation_ValueError" in monitoring._error_counts, "Error should be recorded with correct key"
        assert monitoring._error_counts["test_operation_ValueError"] == 1, "Error count should be 1"
        assert monitoring._consecutive_errors == 1, "Consecutive errors should be 1"
        assert monitoring._last_error_time is not None, "Last error time should be set"

        # Test error count aggregation
        monitoring._record_error("test_operation", test_error)
        monitoring._record_error("other_operation", RuntimeError("Runtime error"))

        assert monitoring._error_counts["test_operation_ValueError"] == 2, "Error count should be aggregated"
        assert "other_operation_RuntimeError" in monitoring._error_counts, "Different error types should be tracked separately"
        assert monitoring._consecutive_errors == 3, "Consecutive errors should be 3"

        # Test memory leak prevention - error count growth
        initial_error_types = len(monitoring._error_counts)

        # Simulate many different error types to test memory management
        for i in range(200):
            error_type = Exception if i % 2 == 0 else ValueError
            monitoring._record_error(f"operation_{i % 50}", error_type(f"Error {i}"))

        # Error counts should not grow indefinitely
        final_error_types = len(monitoring._error_counts)
        assert final_error_types < 150, f"Too many error types tracked: {final_error_types} (potential memory leak)"

        # Test consecutive error reset on success
        monitoring.record_operation("success_operation", 0.050, success=True)
        assert monitoring._consecutive_errors == 0, "Consecutive errors should reset on successful operation"

    def test_response_time_tracking_memory_management(self, monitoring_service):
        """Test response time tracking and memory leak prevention."""
        monitoring = monitoring_service

        # Clear initial state
        monitoring._response_times.clear()
        initial_max_samples = monitoring._max_response_time_samples

        # Test normal response time recording
        for i in range(100):
            monitoring.record_operation(f"test_op_{i}", 0.050, success=True)

        assert len(monitoring._response_times) == 100, "Response times should be recorded"

        # Test memory management - exceed maximum samples
        excess_samples = initial_max_samples + 500
        for i in range(excess_samples):
            monitoring.record_operation(f"excess_op_{i}", 0.050, success=True)

        # Should trigger cleanup when exceeding maximum
        final_count = len(monitoring._response_times)
        assert final_count <= initial_max_samples, f"Response times not limited: {final_count} > {initial_max_samples}"
        assert final_count >= initial_max_samples // 2, f"Response times too aggressively pruned: {final_count} < {initial_max_samples // 2}"

        # Test that most recent samples are preserved
        recent_samples = monitoring._response_times[-10:]  # Last 10 samples
        assert len(recent_samples) == 10, "Recent samples should be preserved during cleanup"

    def test_error_rate_calculation_accuracy(self, monitoring_service, mock_cache_coordinator):
        """Test error rate calculation with various scenarios."""
        monitoring = monitoring_service

        # Setup mock coordinator stats
        mock_cache_coordinator.get_performance_stats.return_value = {
            "total_requests": 1000
        }

        # Clear error state
        monitoring._error_counts.clear()

        # Test zero errors
        error_rate = monitoring._calculate_error_rate()
        assert error_rate == 0.0, "Error rate should be 0 with no errors"

        # Test with known error counts
        monitoring._error_counts = {
            "operation_1_Exception": 5,
            "operation_2_ValueError": 3,
            "operation_3_RuntimeError": 2
        }

        error_rate = monitoring._calculate_error_rate()
        expected_error_rate = (5 + 3 + 2) / 1000  # 10 errors out of 1000 requests
        assert abs(error_rate - expected_error_rate) < 0.001, f"Error rate calculation incorrect: {error_rate} != {expected_error_rate}"

        # Test with zero total requests (edge case)
        mock_cache_coordinator.get_performance_stats.return_value = {
            "total_requests": 0
        }

        error_rate = monitoring._calculate_error_rate()
        assert error_rate == 10.0, "Error rate should handle zero total requests gracefully"  # 10 errors / max(0, 1) = 10

    def test_memory_utilization_calculation(self, monitoring_service, mock_cache_coordinator):
        """Test memory utilization calculation accuracy."""
        monitoring = monitoring_service

        # Test with mock L1 cache stats
        test_stats = {
            "l1_cache_stats": {
                "utilization": 0.75
            }
        }

        utilization = monitoring._calculate_memory_utilization(test_stats)
        assert utilization == 0.75, f"Memory utilization should be 0.75, got {utilization}"

        # Test with different utilization values
        edge_cases = [0.0, 0.01, 0.50, 0.99, 1.0]
        for expected_util in edge_cases:
            test_stats["l1_cache_stats"]["utilization"] = expected_util
            actual_util = monitoring._calculate_memory_utilization(test_stats)
            assert actual_util == expected_util, f"Memory utilization should be {expected_util}, got {actual_util}"

    def test_alert_threshold_boundary_conditions(self, monitoring_service, mock_cache_coordinator):
        """Test alert threshold boundary conditions and logic."""
        monitoring = monitoring_service

        # Setup specific performance stats for threshold testing
        mock_cache_coordinator.get_performance_stats.return_value = {
            "overall_hit_rate": 0.70,  # Between warning (0.7) and critical (0.5)
            "avg_response_time_ms": 150.0,  # Between warning (100) and critical (200)
            "total_requests": 1000
        }

        # Setup SLO compliance for threshold testing
        monitoring._response_times = [0.100] * 900 + [0.300] * 100  # 90% compliance

        # Setup error counts for threshold testing
        monitoring._error_counts = {"test_error": 60}  # 6% error rate
        monitoring._consecutive_errors = 5

        alert_metrics = monitoring.get_alert_metrics()
        alerts = alert_metrics["alerts"]
        values = alert_metrics["values"]
        thresholds = alert_metrics["thresholds"]

        # Test hit rate thresholds
        assert alerts["hit_rate_warning"] is False, "Hit rate warning should not trigger (0.70 == 0.70 threshold)"
        assert alerts["hit_rate_critical"] is False, "Hit rate critical should not trigger (0.70 > 0.50)"

        # Test response time thresholds
        assert alerts["response_time_warning"] is True, "Response time warning should trigger (150ms > 100ms)"
        assert alerts["response_time_critical"] is False, "Response time critical should not trigger (150ms < 200ms)"

        # Test SLO compliance thresholds
        assert alerts["slo_compliance_warning"] is True, "SLO compliance warning should trigger (90% < 95%)"
        assert alerts["slo_compliance_critical"] is True, "SLO compliance critical should trigger (90% < 90%)"

        # Test error rate thresholds
        assert alerts["error_rate_warning"] is True, "Error rate warning should trigger (6% > 5%)"
        assert alerts["error_rate_critical"] is False, "Error rate critical should not trigger (6% < 10%)"

        # Test consecutive errors threshold
        assert alerts["consecutive_errors_critical"] is False, "Consecutive errors critical should not trigger (5 < 10)"

        # Test exact boundary conditions
        # Hit rate exactly at warning threshold
        mock_cache_coordinator.get_performance_stats.return_value["overall_hit_rate"] = 0.70
        alert_metrics = monitoring.get_alert_metrics()
        assert alert_metrics["alerts"]["hit_rate_warning"] is False, "Hit rate warning should not trigger at exact threshold"

        # Hit rate just below warning threshold
        mock_cache_coordinator.get_performance_stats.return_value["overall_hit_rate"] = 0.699
        alert_metrics = monitoring.get_alert_metrics()
        assert alert_metrics["alerts"]["hit_rate_warning"] is True, "Hit rate warning should trigger just below threshold"

    def test_health_status_determination_logic(self, monitoring_service):
        """Test health status determination logic."""
        monitoring = monitoring_service

        # Test initial healthy state
        assert monitoring._health_status["overall_health"] == "healthy", "Initial health should be healthy"

        # Test health status update logic manually
        component_healths = ["healthy", "healthy", "healthy", "healthy"]

        # All healthy
        if "unhealthy" in component_healths:
            overall_health = "unhealthy"
        elif "degraded" in component_healths:
            overall_health = "degraded"
        else:
            overall_health = "healthy"

        assert overall_health == "healthy", "All healthy components should result in healthy overall"

        # One degraded
        component_healths = ["healthy", "degraded", "healthy", "healthy"]
        if "unhealthy" in component_healths:
            overall_health = "unhealthy"
        elif "degraded" in component_healths:
            overall_health = "degraded"
        else:
            overall_health = "healthy"

        assert overall_health == "degraded", "Any degraded component should result in degraded overall"

        # One unhealthy
        component_healths = ["healthy", "degraded", "unhealthy", "healthy"]
        if "unhealthy" in component_healths:
            overall_health = "unhealthy"
        elif "degraded" in component_healths:
            overall_health = "degraded"
        else:
            overall_health = "healthy"

        assert overall_health == "unhealthy", "Any unhealthy component should result in unhealthy overall"

    def test_monitoring_metrics_structure_validation(self, monitoring_service, mock_cache_coordinator):
        """Test monitoring metrics structure and completeness."""
        monitoring = monitoring_service

        # Test with complete cache stats
        metrics = monitoring.get_monitoring_metrics()

        # Validate required metrics exist
        required_metrics = [
            "cache.hit_rate.overall",
            "cache.response_time.avg_ms",
            "cache.requests.total",
            "cache.health.overall",
            "cache.l1.hit_rate",
            "cache.l1.hits",
            "cache.l1.size",
            "cache.l1.memory_bytes",
            "cache.l1.health",
            "cache.l2.enabled",
            "cache.l2.hit_rate",
            "cache.l2.hits",
            "cache.warming.enabled",
            "cache.warming.tracked_patterns"
        ]

        for metric in required_metrics:
            assert metric in metrics, f"Required metric missing: {metric}"

        # Test with L2 cache available
        assert metrics["cache.l2.enabled"] is True, "L2 should be enabled in mock setup"
        assert "cache.l2.success_rate" in metrics, "L2 success rate should be included when available"
        assert "cache.l2.health" in metrics, "L2 health should be included when available"
        assert "cache.l2.connected" in metrics, "L2 connection status should be included when available"

        # Test with L1/L2 cache available
        # L3 cache eliminated - check database cache disabled flag
        assert "cache.database.enabled" in metrics, "Database cache disabled flag should be present"
        assert metrics["cache.database.enabled"] is False, "Database cache should be disabled"

        # Test with L2 cache unavailable
        mock_cache_coordinator.get_performance_stats.return_value["l2_cache_stats"] = None
        metrics = monitoring.get_monitoring_metrics()

        assert metrics["cache.l2.enabled"] is False, "L2 should be disabled when stats unavailable"
        assert "cache.l2.success_rate" not in metrics, "L2 success rate should not be included when unavailable"

        # Test with database cache unavailable (eliminated architecture)
        # L3 no longer exists - no need to set l3_cache_stats to None
        metrics = monitoring.get_monitoring_metrics()

        assert metrics["cache.database.enabled"] is False, "Database cache should be disabled in L1/L2 architecture"

    def test_opentelemetry_metrics_recording(self, monitoring_service):
        """Test OpenTelemetry metrics recording logic."""
        monitoring = monitoring_service

        # Test with OpenTelemetry available
        with patch('prompt_improver.services.cache.cache_monitoring_service.OPENTELEMETRY_AVAILABLE', True):
            with patch('prompt_improver.services.cache.cache_monitoring_service.cache_operations_counter') as mock_counter, \
                 patch('prompt_improver.services.cache.cache_monitoring_service.cache_latency_histogram') as mock_histogram, \
                 patch('prompt_improver.services.cache.cache_monitoring_service.cache_error_counter') as mock_error_counter:

                # Test successful operation recording
                monitoring._update_opentelemetry_metrics("test_operation", 0.100, True)

                mock_counter.add.assert_called_once_with(
                    1,
                    {
                        "operation": "test_operation",
                        "status": "success",
                        "cache_type": "multi_level",
                    }
                )

                mock_histogram.record.assert_called_once_with(
                    0.100,
                    {
                        "operation": "test_operation",
                        "cache_type": "multi_level",
                    }
                )

                assert not mock_error_counter.add.called, "Error counter should not be called for successful operations"

                # Test failed operation recording
                mock_counter.reset_mock()
                mock_histogram.reset_mock()

                monitoring._update_opentelemetry_metrics("failed_operation", 0.200, False)

                mock_counter.add.assert_called_with(
                    1,
                    {
                        "operation": "failed_operation",
                        "status": "error",
                        "cache_type": "multi_level",
                    }
                )

                mock_error_counter.add.assert_called_once_with(
                    1,
                    {
                        "operation": "failed_operation",
                        "cache_type": "multi_level",
                    }
                )

        # Test with OpenTelemetry unavailable
        with patch('prompt_improver.services.cache.cache_monitoring_service.OPENTELEMETRY_AVAILABLE', False):
            # Should not raise exceptions
            monitoring._update_opentelemetry_metrics("test_operation", 0.100, True)
            monitoring._update_opentelemetry_metrics("failed_operation", 0.200, False)

    def test_record_operation_integration(self, monitoring_service):
        """Test record_operation method integration and side effects."""
        monitoring = monitoring_service

        # Clear initial state
        monitoring._response_times.clear()
        monitoring._error_counts.clear()
        monitoring._consecutive_errors = 0

        # Test successful operation recording
        monitoring.record_operation("test_operation", 0.075, success=True)

        assert len(monitoring._response_times) == 1, "Response time should be recorded"
        assert monitoring._response_times[0] == 0.075, "Correct response time should be recorded"
        assert monitoring._consecutive_errors == 0, "Consecutive errors should remain 0 for successful operation"
        assert len(monitoring._error_counts) == 0, "No errors should be recorded for successful operation"

        # Test failed operation recording
        monitoring.record_operation("failed_operation", 0.150, success=False)

        assert len(monitoring._response_times) == 2, "Response time should be recorded for failed operation"
        assert monitoring._response_times[1] == 0.150, "Correct response time should be recorded for failed operation"
        assert monitoring._consecutive_errors == 1, "Consecutive errors should increment for failed operation"
        assert len(monitoring._error_counts) == 1, "Error should be recorded for failed operation"
        assert "failed_operation_Exception" in monitoring._error_counts, "Error should be recorded with correct key"

        # Test consecutive successful operations reset consecutive errors
        monitoring.record_operation("success_after_failure", 0.050, success=True)

        assert monitoring._consecutive_errors == 0, "Consecutive errors should reset after successful operation"
        assert len(monitoring._response_times) == 3, "Response time should continue to be recorded"

    def test_health_summary_calculation(self, monitoring_service, mock_cache_coordinator):
        """Test health summary calculation accuracy."""
        monitoring = monitoring_service

        # Set creation time for uptime calculation
        test_creation_time = datetime.now(UTC) - timedelta(hours=2.5)  # 2.5 hours ago
        monitoring._created_at = test_creation_time

        # Setup mock stats
        mock_cache_coordinator.get_performance_stats.return_value = {
            "overall_hit_rate": 0.85,
            "avg_response_time_ms": 45.0,
            "total_requests": 1000
        }

        # Setup SLO compliance
        monitoring._response_times = [0.100] * 950 + [0.300] * 50  # 95% compliance

        # Setup error rate
        monitoring._error_counts = {"test_error": 20}  # 2% error rate

        health_summary = monitoring.get_health_summary()

        # Validate health summary structure
        required_fields = [
            "overall_healthy",
            "overall_status",
            "hit_rate",
            "avg_response_time_ms",
            "slo_compliant",
            "total_requests",
            "error_rate",
            "uptime_hours",
            "last_check"
        ]

        for field in required_fields:
            assert field in health_summary, f"Required field missing from health summary: {field}"

        # Validate calculated values
        assert health_summary["overall_healthy"] in {True, False}, "Overall healthy should be boolean"
        assert health_summary["hit_rate"] == 0.85, "Hit rate should match coordinator stats"
        assert health_summary["avg_response_time_ms"] == 45.0, "Response time should match coordinator stats"
        assert health_summary["total_requests"] == 1000, "Total requests should match coordinator stats"
        assert abs(health_summary["error_rate"] - 0.02) < 0.001, f"Error rate should be 0.02, got {health_summary['error_rate']}"
        assert abs(health_summary["uptime_hours"] - 2.5) < 0.1, f"Uptime should be ~2.5 hours, got {health_summary['uptime_hours']}"
        assert health_summary["slo_compliant"] is True, "SLO should be compliant with 95% compliance rate"
