"""Contract tests for CacheMonitoringService API compliance.

Validates that CacheMonitoringService maintains its API contracts and
provides consistent interfaces for monitoring systems, dashboards,
and alerting platforms. Tests schema compliance, data types, and
interface stability.

Contract areas validated:
- Health check API contract and response schema
- Monitoring metrics API contract for time-series databases
- Alert metrics API contract for alerting systems
- SLO compliance API contract for SLO monitoring
- OpenTelemetry integration contract
- Health summary API contract for dashboards
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from prompt_improver.services.cache.cache_facade import CacheFacade
from prompt_improver.services.cache.cache_monitoring_service import (
    CacheMonitoringService,
)


class TestCacheMonitoringServiceContracts:
    """Contract tests for CacheMonitoringService API compliance."""

    @pytest.fixture
    def cache_facade(self):
        """Real cache facade for contract testing with direct cache-aside pattern."""
        facade = CacheFacade(
            l1_max_size=100,
            enable_l2=False,  # Disable L2 for isolated testing
            enable_warming=False,
        )

        # Populate some test data for realistic contract testing
        import asyncio

        async def setup_test_data():
            await facade.set("contract_key_1", "value1")
            await facade.set("contract_key_2", "value2")
            await facade.get("contract_key_1")  # Generate some hits
            await facade.get("contract_key_2")

        # Run setup in test context
        asyncio.create_task(setup_test_data())
        return facade

    @pytest.fixture
    def monitoring_service(self, cache_facade):
        """CacheMonitoringService with real cache facade for contract testing."""
        return CacheMonitoringService(cache_facade)

    def test_health_check_api_contract(self, monitoring_service):
        """Test health check API contract compliance."""

        async def run_health_check_contract_test():
            health_result = await monitoring_service.health_check()

            # Validate required top-level fields
            required_fields = {
                "healthy": bool,
                "overall_status": str,
                "checks": dict,
                "performance": dict,
                "timestamp": str
            }

            for field, expected_type in required_fields.items():
                assert field in health_result, f"Required field missing: {field}"
                assert isinstance(health_result[field], expected_type), f"Field {field} should be {expected_type.__name__}, got {type(health_result[field]).__name__}"

            # Validate overall_status values
            valid_statuses = ["healthy", "degraded", "unhealthy"]
            assert health_result["overall_status"] in valid_statuses, f"Invalid overall_status: {health_result['overall_status']}"

            # Validate checks structure
            checks = health_result["checks"]
            required_check_sections = ["coordinator_operations", "cache_levels"]

            for section in required_check_sections:
                assert section in checks, f"Required check section missing: {section}"

            # Validate coordinator_operations contract
            coordinator_ops = checks["coordinator_operations"]
            coordinator_fields = {
                "healthy": bool,
                "set_time_ms": (int, float),
                "get_time_ms": (int, float),
                "delete_time_ms": (int, float),
                "value_match": bool
            }

            for field, expected_types in coordinator_fields.items():
                assert field in coordinator_ops, f"Coordinator operations field missing: {field}"
                if isinstance(expected_types, tuple):
                    assert isinstance(coordinator_ops[field], expected_types), f"Field {field} should be numeric"
                else:
                    assert isinstance(coordinator_ops[field], expected_types), f"Field {field} should be {expected_types.__name__}"

            # Validate cache_levels contract
            cache_levels = checks["cache_levels"]
            required_cache_levels = ["l1", "l2", "l3"]

            for level in required_cache_levels:
                assert level in cache_levels, f"Cache level missing: {level}"

                level_data = cache_levels[level]
                level_required_fields = ["healthy", "status"]

                for field in level_required_fields:
                    assert field in level_data, f"Cache level {level} missing field: {field}"

                assert isinstance(level_data["healthy"], bool), f"Cache level {level} healthy should be boolean"
                assert isinstance(level_data["status"], str), f"Cache level {level} status should be string"

            # Validate L1 specific fields
            l1_data = cache_levels["l1"]
            l1_specific_fields = ["hit_rate", "size"]
            for field in l1_specific_fields:
                assert field in l1_data, f"L1 cache missing field: {field}"
                assert isinstance(l1_data[field], (int, float)), f"L1 {field} should be numeric"

            # Validate performance contract
            performance = health_result["performance"]
            performance_fields = {
                "total_check_time_ms": (int, float),
                "meets_slo": bool,
                "overall_hit_rate": (int, float),
                "avg_response_time_ms": (int, float)
            }

            for field, expected_types in performance_fields.items():
                assert field in performance, f"Performance field missing: {field}"
                if isinstance(expected_types, tuple):
                    assert isinstance(performance[field], expected_types), f"Performance {field} should be numeric"
                else:
                    assert isinstance(performance[field], expected_types), f"Performance {field} should be {expected_types.__name__}"

            # Validate timestamp format (ISO 8601)
            timestamp = health_result["timestamp"]
            try:
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                pytest.fail(f"Invalid timestamp format: {timestamp}")

        # Run async test
        import asyncio
        asyncio.run(run_health_check_contract_test())

    def test_monitoring_metrics_api_contract(self, monitoring_service):
        """Test monitoring metrics API contract for time-series databases."""
        metrics = monitoring_service.get_monitoring_metrics()

        # Validate required metrics for time-series databases
        required_metrics = {
            # Overall metrics
            "cache.hit_rate.overall": (int, float),
            "cache.response_time.avg_ms": (int, float),
            "cache.requests.total": (int, float),
            "cache.health.overall": str,

            # L1 metrics
            "cache.l1.hit_rate": (int, float),
            "cache.l1.hits": (int, float),
            "cache.l1.size": (int, float),
            "cache.l1.memory_bytes": (int, float),
            "cache.l1.health": str,

            # L2 metrics
            "cache.l2.enabled": bool,
            "cache.l2.hit_rate": (int, float),
            "cache.l2.hits": (int, float),

            # Cache warming metrics
            "cache.warming.enabled": bool,
            "cache.warming.tracked_patterns": (int, float)
        }

        for metric, expected_types in required_metrics.items():
            assert metric in metrics, f"Required metric missing: {metric}"
            if isinstance(expected_types, tuple):
                assert isinstance(metrics[metric], expected_types), f"Metric {metric} should be numeric, got {type(metrics[metric]).__name__}"
            else:
                assert isinstance(metrics[metric], expected_types), f"Metric {metric} should be {expected_types.__name__}, got {type(metrics[metric]).__name__}"

        # Validate metric value ranges
        assert 0 <= metrics["cache.hit_rate.overall"] <= 1, "Overall hit rate should be between 0 and 1"
        assert metrics["cache.response_time.avg_ms"] >= 0, "Average response time should be non-negative"
        assert metrics["cache.requests.total"] >= 0, "Total requests should be non-negative"
        assert metrics["cache.l1.size"] >= 0, "L1 size should be non-negative"
        assert metrics["cache.l1.memory_bytes"] >= 0, "L1 memory usage should be non-negative"

        # Validate health status values
        valid_health_statuses = ["healthy", "degraded", "unhealthy"]
        assert metrics["cache.health.overall"] in valid_health_statuses, f"Invalid overall health status: {metrics['cache.health.overall']}"
        assert metrics["cache.l1.health"] in valid_health_statuses, f"Invalid L1 health status: {metrics['cache.l1.health']}"

        # Validate metric naming convention (dot-separated hierarchy)
        for metric_name in metrics:
            parts = metric_name.split('.')
            assert len(parts) >= 2, f"Metric name should follow hierarchical naming: {metric_name}"
            assert parts[0] == "cache", f"All metrics should start with 'cache': {metric_name}"

    def test_alert_metrics_api_contract(self, monitoring_service):
        """Test alert metrics API contract for alerting systems."""
        monitoring_service._response_times = [0.100] * 950 + [0.300] * 50  # Setup for SLO compliance
        monitoring_service._error_counts = {"test_error": 20}  # Setup error rate

        alert_metrics = monitoring_service.get_alert_metrics()

        # Validate required top-level structure
        required_sections = {
            "alerts": dict,
            "values": dict,
            "thresholds": dict,
            "timestamp": str
        }

        for section, expected_type in required_sections.items():
            assert section in alert_metrics, f"Required section missing: {section}"
            assert isinstance(alert_metrics[section], expected_type), f"Section {section} should be {expected_type.__name__}"

        # Validate alerts section contract
        alerts = alert_metrics["alerts"]
        required_alerts = [
            "hit_rate_critical",
            "hit_rate_warning",
            "response_time_critical",
            "response_time_warning",
            "slo_compliance_critical",
            "slo_compliance_warning",
            "error_rate_critical",
            "error_rate_warning",
            "system_unhealthy",
            "system_degraded",
            "consecutive_errors_critical",
            "memory_usage_warning"
        ]

        for alert in required_alerts:
            assert alert in alerts, f"Required alert missing: {alert}"
            assert isinstance(alerts[alert], bool), f"Alert {alert} should be boolean"

        # Validate values section contract
        values = alert_metrics["values"]
        required_values = {
            "hit_rate": (int, float),
            "response_time_ms": (int, float),
            "slo_compliance": (int, float),
            "error_rate": (int, float),
            "consecutive_errors": int,
            "memory_utilization": (int, float),
            "overall_health": str
        }

        for value, expected_types in required_values.items():
            assert value in values, f"Required value missing: {value}"
            if isinstance(expected_types, tuple):
                assert isinstance(values[value], expected_types), f"Value {value} should be numeric"
            else:
                assert isinstance(values[value], expected_types), f"Value {value} should be {expected_types.__name__}"

        # Validate value ranges
        assert 0 <= values["hit_rate"] <= 1, "Hit rate should be between 0 and 1"
        assert values["response_time_ms"] >= 0, "Response time should be non-negative"
        assert 0 <= values["slo_compliance"] <= 1, "SLO compliance should be between 0 and 1"
        assert 0 <= values["error_rate"] <= 1, "Error rate should be between 0 and 1"
        assert values["consecutive_errors"] >= 0, "Consecutive errors should be non-negative"
        assert 0 <= values["memory_utilization"] <= 1, "Memory utilization should be between 0 and 1"

        # Validate thresholds section contract
        thresholds = alert_metrics["thresholds"]
        required_thresholds = [
            "hit_rate_critical",
            "hit_rate_warning",
            "response_time_critical_ms",
            "response_time_warning_ms",
            "slo_compliance_critical",
            "slo_compliance_warning",
            "error_rate_critical",
            "error_rate_warning",
            "consecutive_errors_critical",
            "memory_usage_warning"
        ]

        for threshold in required_thresholds:
            assert threshold in thresholds, f"Required threshold missing: {threshold}"
            assert isinstance(thresholds[threshold], (int, float)), f"Threshold {threshold} should be numeric"

        # Validate threshold relationships
        assert thresholds["hit_rate_critical"] < thresholds["hit_rate_warning"], "Critical hit rate threshold should be lower than warning"
        assert thresholds["response_time_warning_ms"] < thresholds["response_time_critical_ms"], "Warning response time threshold should be lower than critical"
        assert thresholds["slo_compliance_critical"] < thresholds["slo_compliance_warning"], "Critical SLO threshold should be lower than warning"
        assert thresholds["error_rate_warning"] < thresholds["error_rate_critical"], "Warning error rate threshold should be lower than critical"

    def test_slo_compliance_api_contract(self, monitoring_service):
        """Test SLO compliance API contract for SLO monitoring systems."""
        # Setup test data for SLO compliance
        monitoring_service._response_times = [0.050, 0.100, 0.150, 0.200, 0.250]

        slo_compliance = monitoring_service.calculate_slo_compliance()

        # Validate required fields
        required_fields = {
            "compliant": bool,
            "slo_target_ms": (int, float),
            "compliance_rate": (int, float),
            "compliant_requests": int,
            "total_requests": int,
            "violations": int,
            "percentiles": dict,
            "mean_ms": (int, float),
            "sample_count": int
        }

        for field, expected_types in required_fields.items():
            assert field in slo_compliance, f"Required SLO field missing: {field}"
            if isinstance(expected_types, tuple):
                assert isinstance(slo_compliance[field], expected_types), f"SLO field {field} should be numeric"
            else:
                assert isinstance(slo_compliance[field], expected_types), f"SLO field {field} should be {expected_types.__name__}"

        # Validate percentiles structure
        percentiles = slo_compliance["percentiles"]
        required_percentiles = ["p50", "p95", "p99"]

        for percentile in required_percentiles:
            assert percentile in percentiles, f"Required percentile missing: {percentile}"
            assert isinstance(percentiles[percentile], (int, float)), f"Percentile {percentile} should be numeric"

        # Validate value ranges and relationships
        assert 0 <= slo_compliance["compliance_rate"] <= 1, "Compliance rate should be between 0 and 1"
        assert slo_compliance["compliant_requests"] <= slo_compliance["total_requests"], "Compliant requests should not exceed total requests"
        assert slo_compliance["violations"] + slo_compliance["compliant_requests"] == slo_compliance["total_requests"], "Violations + compliant should equal total"
        assert slo_compliance["sample_count"] == slo_compliance["total_requests"], "Sample count should equal total requests"
        assert slo_compliance["slo_target_ms"] > 0, "SLO target should be positive"
        assert slo_compliance["mean_ms"] >= 0, "Mean response time should be non-negative"

        # Validate percentile ordering
        assert percentiles["p50"] <= percentiles["p95"], "P50 should be <= P95"
        assert percentiles["p95"] <= percentiles["p99"], "P95 should be <= P99"

    def test_health_summary_api_contract(self, monitoring_service):
        """Test health summary API contract for dashboards."""
        # Setup test data
        monitoring_service._response_times = [0.100] * 19 + [0.300] * 1  # 95% compliance
        monitoring_service._error_counts = {"test_error": 10}  # Setup error rate

        health_summary = monitoring_service.get_health_summary()

        # Validate required fields
        required_fields = {
            "overall_healthy": bool,
            "overall_status": str,
            "hit_rate": (int, float),
            "avg_response_time_ms": (int, float),
            "slo_compliant": bool,
            "total_requests": int,
            "error_rate": (int, float),
            "uptime_hours": (int, float),
            "last_check": str
        }

        for field, expected_types in required_fields.items():
            assert field in health_summary, f"Required health summary field missing: {field}"
            if isinstance(expected_types, tuple):
                assert isinstance(health_summary[field], expected_types), f"Health summary field {field} should be numeric"
            else:
                assert isinstance(health_summary[field], expected_types), f"Health summary field {field} should be {expected_types.__name__}"

        # Validate value ranges
        assert 0 <= health_summary["hit_rate"] <= 1, "Hit rate should be between 0 and 1"
        assert health_summary["avg_response_time_ms"] >= 0, "Average response time should be non-negative"
        assert health_summary["total_requests"] >= 0, "Total requests should be non-negative"
        assert 0 <= health_summary["error_rate"] <= 1, "Error rate should be between 0 and 1"
        assert health_summary["uptime_hours"] >= 0, "Uptime should be non-negative"

        # Validate status values
        valid_statuses = ["healthy", "degraded", "unhealthy"]
        assert health_summary["overall_status"] in valid_statuses, f"Invalid overall status: {health_summary['overall_status']}"

        # Validate timestamp format
        try:
            datetime.fromisoformat(health_summary["last_check"])
        except ValueError:
            pytest.fail(f"Invalid last_check timestamp format: {health_summary['last_check']}")

    def test_record_operation_api_contract(self, monitoring_service):
        """Test record_operation API contract for operation tracking."""
        # Test valid operation recording
        try:
            monitoring_service.record_operation("test_operation", 0.100, success=True)
            monitoring_service.record_operation("failed_operation", 0.200, success=False)
        except Exception as e:
            pytest.fail(f"record_operation should not raise exceptions: {e}")

        # Validate that operation recording affects response times
        assert len(monitoring_service._response_times) >= 2, "Response times should be recorded"

        # Test parameter types
        with pytest.raises(TypeError):
            monitoring_service.record_operation(123, 0.100, success=True)  # Invalid operation type

        with pytest.raises(TypeError):
            monitoring_service.record_operation("test", "invalid_duration", success=True)  # Invalid duration type

        with pytest.raises(TypeError):
            monitoring_service.record_operation("test", 0.100, success="invalid_success")  # Invalid success type

    def test_opentelemetry_integration_contract(self, monitoring_service):
        """Test OpenTelemetry integration contract."""
        # Test update_cache_metrics method exists and is callable
        assert hasattr(monitoring_service, 'update_cache_metrics'), "update_cache_metrics method should exist"
        assert callable(monitoring_service.update_cache_metrics), "update_cache_metrics should be callable"

        # Test method executes without exceptions
        try:
            monitoring_service.update_cache_metrics()
        except Exception as e:
            pytest.fail(f"update_cache_metrics should not raise exceptions: {e}")

    def test_api_stability_contract(self, monitoring_service):
        """Test API stability and backward compatibility."""
        # Test that all public methods exist
        required_public_methods = [
            "health_check",
            "get_monitoring_metrics",
            "get_alert_metrics",
            "calculate_slo_compliance",
            "record_operation",
            "update_cache_metrics",
            "get_health_summary"
        ]

        for method in required_public_methods:
            assert hasattr(monitoring_service, method), f"Required public method missing: {method}"
            assert callable(getattr(monitoring_service, method)), f"Method {method} should be callable"

        # Test that private methods don't change (internal contract)
        required_private_methods = [
            "_record_error",
            "_calculate_error_rate",
            "_calculate_memory_utilization",
            "_update_opentelemetry_metrics"
        ]

        for method in required_private_methods:
            assert hasattr(monitoring_service, method), f"Required private method missing: {method}"
            assert callable(getattr(monitoring_service, method)), f"Private method {method} should be callable"

    def test_error_handling_contract(self, monitoring_service):
        """Test error handling contract and exception guarantees."""

        # Test health_check error handling
        async def test_health_check_error_handling():
            # Mock coordinator to raise exception
            monitoring_service._cache_coordinator.get = MagicMock(side_effect=Exception("Coordinator failure"))

            health_result = await monitoring_service.health_check()

            # Should return error structure instead of raising
            assert health_result["healthy"] is False, "Health check should report unhealthy on error"
            assert "error" in health_result, "Error information should be included"
            assert "error_type" in health_result, "Error type should be included"

        # Run async test
        import asyncio
        asyncio.run(test_health_check_error_handling())

        # Test monitoring methods with invalid coordinator
        monitoring_service._cache_coordinator.get_performance_stats = MagicMock(return_value=None)

        # These methods should handle None stats gracefully or raise appropriate exceptions
        try:
            monitoring_service.get_monitoring_metrics()
            pytest.fail("get_monitoring_metrics should handle None stats")
        except (AttributeError, TypeError):
            pass  # Expected behavior

        # Test with minimal valid stats
        monitoring_service._cache_coordinator.get_performance_stats = MagicMock(return_value={
            "overall_hit_rate": 0.5,
            "avg_response_time_ms": 100,
            "total_requests": 10,
            "health_status": "degraded",
            "l1_cache_stats": {"utilization": 0.5}
        })

        # Should work with minimal stats
        try:
            health_summary = monitoring_service.get_health_summary()
            assert "overall_healthy" in health_summary, "Health summary should work with minimal stats"
        except Exception as e:
            pytest.fail(f"Health summary should work with minimal stats: {e}")

    def test_thread_safety_contract(self, monitoring_service):
        """Test thread safety contract for concurrent access."""
        import threading
        import time

        # Test concurrent record_operation calls
        def record_operations():
            for i in range(100):
                monitoring_service.record_operation(f"thread_op_{i}", 0.050, success=True)
                time.sleep(0.001)  # Small delay to encourage race conditions

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=record_operations)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have recorded all operations without corruption
        assert len(monitoring_service._response_times) >= 500, "All operations should be recorded"

        # All response times should be valid
        for response_time in monitoring_service._response_times:
            assert isinstance(response_time, (int, float)), "All response times should be numeric"
            assert response_time >= 0, "All response times should be non-negative"
