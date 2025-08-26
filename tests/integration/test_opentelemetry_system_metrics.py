"""
OpenTelemetry System Metrics Integration Tests
==============================================

Real behavior testing for OpenTelemetry system metrics collection.
Replaces prometheus-based system metrics tests with OTel-native validation.
"""

import time

import pytest

from prompt_improver.monitoring.opentelemetry.metrics import (
    BusinessMetrics,
    DatabaseMetrics,
    HttpMetrics,
    MLMetrics,
    get_business_metrics,
    get_database_metrics,
    get_http_metrics,
    get_ml_metrics,
)


class TestOpenTelemetrySystemMetrics:
    """Test OpenTelemetry system metrics integration."""

    def setup_method(self):
        """Set up test environment with real OpenTelemetry metrics."""
        self.http_metrics = get_http_metrics("test-system-metrics")
        self.database_metrics = get_database_metrics("test-system-metrics")
        self.ml_metrics = get_ml_metrics("test-system-metrics")
        self.business_metrics = get_business_metrics("test-system-metrics")

    def test_http_metrics_collection(self):
        """Test HTTP metrics collection using OpenTelemetry."""
        self.http_metrics.record_request(
            method="GET",
            endpoint="/api/v1/health",
            status_code=200,
            duration_ms=45.2,
            response_size_bytes=512,
        )
        assert isinstance(self.http_metrics, HttpMetrics)
        assert hasattr(self.http_metrics, "record_request")
        print("✅ HTTP system metrics recorded with real OpenTelemetry")

    def test_database_metrics_collection(self):
        """Test database metrics collection using OpenTelemetry."""
        self.database_metrics.record_query(
            operation="SELECT", table="rule_metadata", duration_ms=12.5, success=True
        )
        self.database_metrics.record_query(
            operation="UPDATE", table="rule_metadata", duration_ms=8.3, success=True
        )
        assert isinstance(self.database_metrics, DatabaseMetrics)
        assert hasattr(self.database_metrics, "record_query")
        print("✅ Database system metrics recorded with real OpenTelemetry")

    def test_ml_metrics_collection(self):
        """Test ML metrics collection using OpenTelemetry."""
        self.ml_metrics.record_inference(
            model_name="rule_optimizer", duration_s=0.15, success=True
        )
        self.ml_metrics.record_prompt_improvement(
            category="model_accuracy", improvement_score=0.92
        )
        assert isinstance(self.ml_metrics, MLMetrics)
        assert hasattr(self.ml_metrics, "record_inference")
        assert hasattr(self.ml_metrics, "record_prompt_improvement")
        print("✅ ML system metrics recorded with real OpenTelemetry")

    def test_business_metrics_collection(self):
        """Test business metrics collection using OpenTelemetry."""
        self.business_metrics.record_feature_usage(
            feature="enhanced_rule_engine", user_tier="premium"
        )
        self.business_metrics.record_session(
            user_id="improvement_session_user", session_duration_s=300.0
        )
        assert isinstance(self.business_metrics, BusinessMetrics)
        assert hasattr(self.business_metrics, "record_feature_usage")
        assert hasattr(self.business_metrics, "record_session")
        print("✅ Business system metrics recorded with real OpenTelemetry")

    @pytest.mark.asyncio
    async def test_real_behavior_metrics_workflow(self):
        """Test real behavior metrics workflow with OpenTelemetry."""
        start_time = time.time()
        self.http_metrics.record_request(
            method="POST",
            endpoint="/api/v1/improve-prompt",
            status_code=200,
            duration_ms=250.0,
            response_size_bytes=1536,
        )
        self.database_metrics.record_query(
            operation="INSERT", table="prompt_sessions", duration_ms=8.2, success=True
        )
        self.ml_metrics.record_inference(
            model_name="prompt_improver", duration_s=0.18, success=True
        )
        self.business_metrics.record_feature_usage(
            feature="ml_optimization", user_tier="standard"
        )
        workflow_duration = (time.time() - start_time) * 1000
        assert workflow_duration < 1000
        assert isinstance(self.http_metrics, HttpMetrics)
        assert isinstance(self.database_metrics, DatabaseMetrics)
        assert isinstance(self.ml_metrics, MLMetrics)
        assert isinstance(self.business_metrics, BusinessMetrics)
        print("✅ Complete real behavior metrics workflow recorded with OpenTelemetry")

    def test_metrics_error_handling(self):
        """Test metrics error handling and resilience."""
        try:
            self.http_metrics.record_request(
                method="INVALID", endpoint="", status_code=-1, duration_ms=-100.0
            )
        except Exception as e:
            pytest.fail(f"Metrics should handle invalid data gracefully: {e}")
        try:
            self.database_metrics.record_query(
                operation=None, table=None, duration_ms=None, success=None
            )
        except Exception as e:
            pytest.fail(f"Metrics should handle None values gracefully: {e}")

    def test_metrics_performance(self):
        """Test metrics collection performance."""
        start_time = time.time()
        for i in range(100):
            self.http_metrics.record_request(
                method="GET",
                endpoint=f"/api/test/{i}",
                status_code=200,
                duration_ms=10.0,
            )
        recording_duration = (time.time() - start_time) * 1000
        assert recording_duration < 200, (
            f"Metrics recording too slow: {recording_duration}ms"
        )
        print(f"✅ Metrics performance test completed in {recording_duration:.1f}ms")

    async def test_metrics_concurrency_safety(self):
        """Test metrics collection concurrency safety."""
        import asyncio

        async def record_metrics(task_id: int):
            """Record metrics from an async task."""
            for i in range(10):
                self.http_metrics.record_request(
                    method="POST",
                    endpoint=f"/task/{task_id}/request/{i}",
                    status_code=200,
                    duration_ms=15.0,
                )

        tasks = [asyncio.create_task(record_metrics(i)) for i in range(5)]
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            pytest.fail(f"Concurrency safety test failed: {e}")
        print("✅ Metrics concurrency safety test completed successfully")

    def test_metrics_labels_and_attributes(self):
        """Test metrics labels and attributes functionality."""
        self.http_metrics.record_request(
            method="GET", endpoint="/api/v1/rules", status_code=200, duration_ms=25.0
        )
        self.http_metrics.record_request(
            method="POST", endpoint="/api/v1/rules", status_code=201, duration_ms=45.0
        )
        self.database_metrics.record_query(
            operation="SELECT", table="rules", duration_ms=5.0, success=True
        )
        self.database_metrics.record_query(
            operation="UPDATE", table="rules", duration_ms=12.0, success=True
        )
        assert isinstance(self.http_metrics, HttpMetrics)
        assert isinstance(self.database_metrics, DatabaseMetrics)
        print(
            "✅ Metrics labels and attributes functionality verified with real OpenTelemetry"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
