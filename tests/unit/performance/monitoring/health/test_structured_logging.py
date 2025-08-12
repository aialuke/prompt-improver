"""
Test Structured Logging Implementation with Real Log Output
Validates actual log format and content without mocks
"""

import asyncio
import io
import json
import logging
import time
from unittest.mock import patch

import pytest

from prompt_improver.performance.monitoring.health.base import (
    HealthResult,
    HealthStatus,
)
from prompt_improver.performance.monitoring.health.structured_logging import (
    HealthMetricsLogger,
    LogContext,
    StructuredLogger,
    get_metrics_logger,
    log_health_check,
)


class TestStructuredLogger:
    """Test structured logger with actual log output verification"""

    def test_structured_logger_json_format(self):
        """Test that structured logger produces valid JSON output"""
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        test_logger = logging.getLogger("test_structured")
        test_logger.setLevel(logging.INFO)
        test_logger.handlers.clear()
        test_logger.addHandler(handler)
        structured = StructuredLogger("test_structured")
        structured.info(
            "Test message",
            component="test_component",
            operation="test_operation",
            custom_field="custom_value",
        )
        handler.flush()
        log_output = log_capture.getvalue().strip()
        log_data = json.loads(log_output)
        assert log_data["level"] == "INFO"
        assert log_data["message"] == "Test message"
        assert "timestamp" in log_data
        assert "timestamp_human" in log_data
        assert "correlation_id" in log_data
        assert log_data["component"] == "test_component"
        assert log_data["operation"] == "test_operation"
        assert log_data["custom_field"] == "custom_value"

    def test_structured_logger_error_with_exception(self):
        """Test error logging includes exception details"""
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        test_logger = logging.getLogger("test_error")
        test_logger.setLevel(logging.ERROR)
        test_logger.handlers.clear()
        test_logger.addHandler(handler)
        structured = StructuredLogger("test_error")
        try:
            raise ValueError("This is a test error with details")
        except ValueError as e:
            structured.error(
                "An error occurred during testing", error=e, context="unit_test"
            )
        handler.flush()
        log_output = log_capture.getvalue().strip()
        log_data = json.loads(log_output)
        assert log_data["level"] == "ERROR"
        assert log_data["message"] == "An error occurred during testing"
        assert log_data["context"] == "unit_test"
        assert "error_details" in log_data
        assert log_data["error_details"]["type"] == "ValueError"
        assert (
            log_data["error_details"]["message"] == "This is a test error with details"
        )
        assert "traceback" in log_data["error_details"]
        assert "raise ValueError" in log_data["error_details"]["traceback"]

    def test_structured_logger_context_stacking(self):
        """Test that context can be stacked and properly managed"""
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        test_logger = logging.getLogger("test_context")
        test_logger.setLevel(logging.INFO)
        test_logger.handlers.clear()
        test_logger.addHandler(handler)
        structured = StructuredLogger("test_context")
        with structured.context(service="outer_service", version="1.0"):
            with structured.context(component="inner_component", phase="testing"):
                structured.info("Nested context test")
        handler.flush()
        log_output = log_capture.getvalue().strip()
        log_data = json.loads(log_output)
        assert log_data["service"] == "outer_service"
        assert log_data["version"] == "1.0"
        assert log_data["component"] == "inner_component"
        assert log_data["phase"] == "testing"

    def test_structured_logger_different_log_levels(self):
        """Test all log levels produce correct structured output"""
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        test_logger = logging.getLogger("test_levels")
        test_logger.setLevel(logging.DEBUG)
        test_logger.handlers.clear()
        test_logger.addHandler(handler)
        structured = StructuredLogger("test_levels")
        structured.debug("Debug message", level_test="debug")
        structured.info("Info message", level_test="info")
        structured.warning("Warning message", level_test="warning")
        structured.error("Error message", level_test="error")
        handler.flush()
        log_lines = log_capture.getvalue().strip().split("\n")
        assert len(log_lines) == 4
        for i, expected_level in enumerate(["DEBUG", "INFO", "WARNING", "ERROR"]):
            log_data = json.loads(log_lines[i])
            assert log_data["level"] == expected_level
            assert log_data["level_test"] == expected_level.lower()


class TestLogHealthCheckDecorator:
    """Test the health check logging decorator"""

    @pytest.mark.asyncio
    async def test_health_check_decorator_successful_execution(self):
        """Test decorator logs successful health check execution"""
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)

        class MockHealthChecker:
            def __init__(self):
                self.__class__.__module__ = "test.module"
                self.__class__.__name__ = "MockHealthChecker"

            @log_health_check("test_component", "availability_check")
            async def check_health(self):
                await asyncio.sleep(0.05)
                return HealthResult(
                    status=HealthStatus.HEALTHY,
                    component="test_component",
                    response_time_ms=50,
                    details={"test": "data"},
                )

        test_logger = logging.getLogger("test.module.MockHealthChecker")
        test_logger.setLevel(logging.INFO)
        test_logger.handlers.clear()
        test_logger.addHandler(handler)
        checker = MockHealthChecker()
        result = await checker.check_health()
        handler.flush()
        log_lines = log_capture.getvalue().strip().split("\n")
        assert len(log_lines) >= 2
        start_log = json.loads(log_lines[0])
        assert start_log["component"] == "test_component"
        assert start_log["check_type"] == "availability_check"
        assert start_log["phase"] == "start"
        assert "Health check started" in start_log["message"]
        complete_log = json.loads(log_lines[1])
        assert complete_log["component"] == "test_component"
        assert complete_log["phase"] == "complete"
        assert complete_log["status"] == "HEALTHY"
        assert complete_log["healthy"] is True
        assert complete_log["duration_ms"] >= 50
        assert "Health check completed" in complete_log["message"]

    @pytest.mark.asyncio
    async def test_health_check_decorator_failure_logging(self):
        """Test decorator logs health check failures properly"""
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)

        class MockFailingChecker:
            def __init__(self):
                self.__class__.__module__ = "test.failing"
                self.__class__.__name__ = "MockFailingChecker"

            @log_health_check("failing_component")
            async def failing_check(self):
                await asyncio.sleep(0.02)
                raise ConnectionError("Database connection failed")

        test_logger = logging.getLogger("test.failing.MockFailingChecker")
        test_logger.setLevel(logging.INFO)
        test_logger.handlers.clear()
        test_logger.addHandler(handler)
        checker = MockFailingChecker()
        with pytest.raises(ConnectionError):
            await checker.failing_check()
        handler.flush()
        log_lines = log_capture.getvalue().strip().split("\n")
        assert len(log_lines) >= 2
        failure_log = json.loads(log_lines[1])
        assert failure_log["component"] == "failing_component"
        assert failure_log["phase"] == "failed"
        assert failure_log["duration_ms"] >= 20
        assert "Health check failed" in failure_log["message"]
        assert "error_details" in failure_log
        assert failure_log["error_details"]["type"] == "ConnectionError"
        assert failure_log["error_details"]["message"] == "Database connection failed"


class TestHealthMetricsLogger:
    """Test health metrics aggregation and logging"""

    def test_health_metrics_logger_aggregation(self):
        """Test that metrics are properly aggregated and flushed"""
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        metrics_logger_name = "health_metrics.test_component"
        test_logger = logging.getLogger(metrics_logger_name)
        test_logger.setLevel(logging.INFO)
        test_logger.handlers.clear()
        test_logger.addHandler(handler)
        metrics_logger = HealthMetricsLogger("test_component")
        metrics_logger._flush_interval = 1
        test_data = [
            ("HEALTHY", 45.2, {"connections": 10}),
            ("HEALTHY", 52.1, {"connections": 12}),
            ("WARNING", 89.5, {"connections": 8}),
            ("HEALTHY", 41.8, {"connections": 11}),
            ("FAILED", 150.0, {"connections": 5}),
        ]
        for status, response_time, details in test_data:
            metrics_logger.record_check(status, response_time, details)
        metrics_logger.flush()
        handler.flush()
        log_output = log_capture.getvalue().strip()
        log_data = json.loads(log_output)
        assert log_data["component"] == "test_component"
        assert log_data["metric_type"] == "health_summary"
        assert log_data["total_checks"] == 5
        assert 45 <= log_data["avg_response_time_ms"] <= 90
        assert log_data["min_response_time_ms"] == 41.8
        assert log_data["max_response_time_ms"] == 150.0
        status_dist = log_data["status_distribution"]
        assert status_dist["HEALTHY"] == 3
        assert status_dist["WARNING"] == 1
        assert status_dist["FAILED"] == 1
        assert log_data["success_rate"] == 0.6

    def test_health_metrics_logger_percentile_calculation(self):
        """Test P95 calculation accuracy"""
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        test_logger = logging.getLogger("health_metrics.percentile_test")
        test_logger.setLevel(logging.INFO)
        test_logger.handlers.clear()
        test_logger.addHandler(handler)
        metrics_logger = HealthMetricsLogger("percentile_test")
        response_times = list(range(1, 101))
        for rt in response_times:
            metrics_logger.record_check("HEALTHY", float(rt), {})
        metrics_logger.flush()
        handler.flush()
        log_output = log_capture.getvalue().strip()
        log_data = json.loads(log_output)
        assert log_data["p95_response_time_ms"] == 95.0
        assert log_data["min_response_time_ms"] == 1.0
        assert log_data["max_response_time_ms"] == 100.0
        assert log_data["avg_response_time_ms"] == 50.5

    def test_metrics_logger_registry(self):
        """Test global metrics logger registry"""
        logger1 = get_metrics_logger("component_a")
        assert isinstance(logger1, HealthMetricsLogger)
        assert logger1.component_name == "component_a"
        logger2 = get_metrics_logger("component_a")
        assert logger1 is logger2
        logger3 = get_metrics_logger("component_b")
        assert logger3 is not logger1
        assert logger3.component_name == "component_b"


class TestLogContext:
    """Test LogContext data class"""

    def test_log_context_to_dict_filters_none(self):
        """Test that to_dict excludes None values"""
        context = LogContext(
            component="test_component",
            operation="test_op",
            health_check_type="availability",
            correlation_id="12345",
            trace_id=None,
            user_id=None,
        )
        context_dict = context.model_dump()
        assert context_dict["component"] == "test_component"
        assert context_dict["operation"] == "test_op"
        assert context_dict["health_check_type"] == "availability"
        assert context_dict["correlation_id"] == "12345"
        assert context_dict["environment"] == "production"
        assert "trace_id" not in context_dict
        assert "user_id" not in context_dict


def test_correlation_id_integration():
    """Test integration with correlation ID context"""
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    test_logger = logging.getLogger("test_correlation")
    test_logger.setLevel(logging.INFO)
    test_logger.handlers.clear()
    test_logger.addHandler(handler)
    mock_correlation_id = "test-correlation-12345"
    with patch(
        "prompt_improver.performance.monitoring.health.structured_logging.get_correlation_id",
        return_value=mock_correlation_id,
    ):
        structured = StructuredLogger("test_correlation")
        structured.info("Test with correlation ID")
    handler.flush()
    log_output = log_capture.getvalue().strip()
    log_data = json.loads(log_output)
    assert log_data["correlation_id"] == mock_correlation_id


@pytest.mark.asyncio
async def test_structured_logging_performance():
    """Test structured logging performance under load"""
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    test_logger = logging.getLogger("test_performance")
    test_logger.setLevel(logging.INFO)
    test_logger.handlers.clear()
    test_logger.addHandler(handler)
    structured = StructuredLogger("test_performance")
    start_time = time.time()
    for i in range(100):
        structured.info(
            f"Performance test message {i}",
            i,
            iteration=i,
            batch="performance_test",
            data={"nested": {"value": i * 2}},
        )
    duration = time.time() - start_time
    handler.flush()
    log_lines = log_capture.getvalue().strip().split("\n")
    assert len(log_lines) == 100
    assert duration < 1.0
    last_log = json.loads(log_lines[-1])
    assert last_log["iteration"] == 99
    assert last_log["batch"] == "performance_test"
    assert last_log["data"]["nested"]["value"] == 198
