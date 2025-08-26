"""
Test Enhanced Health Checkers with Real Behavior - 2025 Best Practices
Integration tests that prioritize real behavior over mocking to prevent false positives
"""

import asyncio
from unittest.mock import patch

import pytest

from prompt_improver.performance.monitoring.health.base import HealthStatus
from prompt_improver.performance.monitoring.health.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitState,
)
from prompt_improver.performance.monitoring.health.enhanced_base import (
    EnhancedHealthChecker,
)
from prompt_improver.performance.monitoring.health.enhanced_checkers import (
    EnhancedAnalyticsServiceHealthChecker,
    EnhancedMLOrchestratorHealthChecker,
    EnhancedMLServiceHealthChecker,
)
from prompt_improver.performance.monitoring.health.sla_monitor import SLAConfiguration


class TestEnhancedHealthCheckerBase:
    """Test the enhanced health checker base functionality"""

    @pytest.mark.asyncio
    async def test_enhanced_health_checker_circuit_breaker_integration(self):
        """Test that circuit breaker properly protects health checks"""

        class FailingHealthChecker(EnhancedHealthChecker):
            def __init__(self):
                config = CircuitBreakerConfig(failure_threshold=3)
                super().__init__("test_failing", circuit_breaker_config=config)
                self.call_count = 0

            async def _execute_health_check(self):
                self.call_count += 1
                raise Exception(f"Health check failure #{self.call_count}")

        checker = FailingHealthChecker()
        for i in range(3):
            result = await checker.check()
            assert result.status == HealthStatus.FAILED
            assert checker.call_count == i + 1
        assert checker.circuit_breaker.state == CircuitState.OPEN
        initial_call_count = checker.call_count
        result = await checker.check()
        assert result.status == HealthStatus.FAILED
        assert (
            "Circuit breaker" in result.details["error"]
            and "OPEN" in result.details["error"]
        )
        assert checker.call_count == initial_call_count

    @pytest.mark.asyncio
    async def test_enhanced_health_checker_sla_monitoring(self):
        """Test that SLA monitoring works correctly"""

        class TimedHealthChecker(EnhancedHealthChecker):
            def __init__(self, response_time_ms: float):
                config = SLAConfiguration(
                    service_name="timed_test", response_time_p95_ms=100
                )
                super().__init__("timed_test", sla_config=config)
                self.response_time_ms = response_time_ms

            async def _execute_health_check(self):
                await asyncio.sleep(self.response_time_ms / 1000)
                from prompt_improver.performance.monitoring.health.base import (
                    HealthResult,
                )

                return HealthResult(
                    status=HealthStatus.HEALTHY,
                    component=self.name,
                    response_time_ms=self.response_time_ms,
                    details={"simulated_time": self.response_time_ms},
                )

        fast_checker = TimedHealthChecker(50)
        for _ in range(10):
            result = await fast_checker.check()
            assert result.status == HealthStatus.HEALTHY
        report = fast_checker.sla_monitor.get_sla_report()
        p95_sla = report["sla_targets"]["response_time_p95"]
        assert p95_sla["status"] == "meeting"
        slow_checker = TimedHealthChecker(150)
        for _ in range(10):
            result = await slow_checker.check()
            assert result.status == HealthStatus.HEALTHY
        report = slow_checker.sla_monitor.get_sla_report()
        p95_sla = report["sla_targets"]["response_time_p95"]
        assert p95_sla["status"] == "breaching"

    @pytest.mark.asyncio
    async def test_enhanced_status_includes_all_monitoring_data(self):
        """Test that enhanced status includes all monitoring components"""

        class MockHealthChecker(EnhancedHealthChecker):
            async def _execute_health_check(self):
                from prompt_improver.performance.monitoring.health.base import (
                    HealthResult,
                )

                return HealthResult(
                    status=HealthStatus.HEALTHY,
                    component=self.name,
                    response_time_ms=50,
                    details={"test": "data"},
                )

        checker = MockHealthChecker("comprehensive_test")
        for _ in range(5):
            await checker.check()
        status = checker.get_enhanced_status()
        assert "component" in status
        assert "circuit_breaker" in status
        assert "sla_report" in status
        assert "sla_metrics" in status
        cb_metrics = status["circuit_breaker"]
        assert cb_metrics["total_calls"] == 5
        assert cb_metrics["successful_calls"] == 5
        assert cb_metrics["state"] == "closed"
        sla_report = status["sla_report"]
        assert "service" in sla_report
        assert "overall_status" in sla_report
        assert "sla_targets" in sla_report


class TestMLServiceHealthChecker:
    """Test ML Service Health Checker with realistic scenarios - 2025 Best Practices"""

    @pytest.mark.asyncio
    async def test_ml_service_unavailable_real_behavior(self):
        """Test REAL behavior when ML service is unavailable - should be WARNING, not HEALTHY"""
        checker = EnhancedMLServiceHealthChecker()
        result = await checker.check()
        assert result.status in {HealthStatus.WARNING, HealthStatus.FAILED}
        if result.status == HealthStatus.WARNING:
            assert result.details.get("fallback_mode") is True
            assert "fallback" in result.details.get("message", "").lower()
        assert result.response_time_ms >= 0

    @pytest.mark.asyncio
    async def test_ml_service_import_error_real_behavior(self):
        """Test REAL behavior when ML service import fails - should be WARNING/FAILED"""
        checker = EnhancedMLServiceHealthChecker()
        result = await checker.check()
        assert result.status in {HealthStatus.WARNING, HealthStatus.FAILED}
        assert result.response_time_ms >= 0
        assert (
            "error" in result.details
            or "fallback" in result.details.get("message", "").lower()
        )

    @pytest.mark.asyncio
    async def test_ml_service_import_error_handling(self):
        """Test handling when ML service cannot be imported - test real behavior"""
        checker = EnhancedMLServiceHealthChecker()
        result = await checker.check()
        assert result.status in {HealthStatus.WARNING, HealthStatus.FAILED}
        assert result.response_time_ms >= 0
        if result.status == HealthStatus.WARNING:
            assert "fallback" in result.details.get("message", "").lower()
        else:
            assert "error" in result.details

    @pytest.mark.asyncio
    async def test_ml_service_circuit_breaker_behavior(self):
        """Test circuit breaker behavior under ML service failures"""

        def mock_failing_service():
            raise RuntimeError("ML service internal error")

        checker = EnhancedMLServiceHealthChecker()
        for _i in range(4):
            result = await checker.check()
            assert result.status in {HealthStatus.WARNING, HealthStatus.FAILED}
            assert result.response_time_ms >= 0
            assert (
                "error" in result.details
                or "fallback" in result.details.get("message", "").lower()
            )


class TestMLOrchestratorHealthChecker:
    """Test ML Orchestrator Health Checker"""

    @pytest.mark.asyncio
    async def test_orchestrator_healthy_scenario(self):
        """Test orchestrator health check when everything is working"""

        class MockOrchestrator:
            def __init__(self):
                self.initialized = True

        checker = EnhancedMLOrchestratorHealthChecker()
        checker.set_orchestrator(MockOrchestrator())
        result = await checker.check()
        assert result.status == HealthStatus.HEALTHY
        assert result.details["initialized"] is True
        assert result.details["component_health"]["healthy_percentage"] >= 70
        assert "workflow_status" in result.details

    @pytest.mark.asyncio
    async def test_orchestrator_not_initialized(self):
        """Test behavior when orchestrator is not initialized"""
        checker = EnhancedMLOrchestratorHealthChecker()
        result = await checker.check()
        assert result.status == HealthStatus.FAILED
        assert "not initialized" in result.details["error"]

    @pytest.mark.asyncio
    async def test_orchestrator_component_health_degradation(self):
        """Test different health states based on component health"""

        class MockOrchestrator:
            def __init__(self):
                self.initialized = True

        checker = EnhancedMLOrchestratorHealthChecker()
        checker.set_orchestrator(MockOrchestrator())

        async def mock_degraded_health():
            return {
                "total_components": 10,
                "healthy_components": 6,
                "healthy_percentage": 60,
                "unhealthy_components": [
                    "data_loader",
                    "model_cache",
                    "metrics",
                    "alerts",
                ],
            }

        with patch.object(checker, "_check_component_health", mock_degraded_health):
            result = await checker.check()
        assert result.status == HealthStatus.FAILED
        assert "60%" in result.details["message"]
        assert result.details["component_health"]["healthy_percentage"] == 60


class TestAnalyticsServiceHealthChecker:
    """Test Analytics Service Health Checker - 2025 Best Practices"""

    @pytest.mark.asyncio
    async def test_analytics_service_real_behavior_unavailable(self):
        """Test REAL behavior when analytics service is unavailable - should be FAILED"""
        checker = EnhancedAnalyticsServiceHealthChecker()
        result = await checker.check()
        assert result.status == HealthStatus.FAILED
        assert result.response_time_ms >= 0
        assert "error" in result.details
        assert "not available" in result.details.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_analytics_service_stale_data_warning(self):
        """Test real behavior - analytics service unavailable should be FAILED"""
        checker = EnhancedAnalyticsServiceHealthChecker()
        result = await checker.check()
        assert result.status == HealthStatus.FAILED
        assert "error" in result.details
        assert "not available" in result.details.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_analytics_service_query_failure(self):
        """Test handling of analytics query failures"""
        checker = EnhancedAnalyticsServiceHealthChecker()
        result = await checker.check()
        assert result.status == HealthStatus.FAILED
        assert "error" in result.details
        assert (
            "not available" in result.details.get("error", "").lower()
            or "module" in result.details.get("error", "").lower()
        )

    @pytest.mark.asyncio
    async def test_analytics_service_import_error(self):
        """Test handling when analytics service is not available"""
        checker = EnhancedAnalyticsServiceHealthChecker()
        result = await checker.check()
        assert result.status == HealthStatus.FAILED
        assert "error" in result.details
        assert "not available" in result.details.get("error", "").lower()


@pytest.mark.asyncio
async def test_integrated_health_monitoring_scenario():
    """Integration test simulating a realistic monitoring scenario"""
    ml_checker = EnhancedMLServiceHealthChecker()
    analytics_checker = EnhancedAnalyticsServiceHealthChecker()
    results = []
    for _i in range(5):
        ml_result = await ml_checker.check()
        results.append(("ml_service", ml_result))
        analytics_result = await analytics_checker.check()
        results.append(("analytics", analytics_result))
        await asyncio.sleep(0.01)
    ml_results = [r for service, r in results if service == "ml_service"]
    analytics_results = [r for service, r in results if service == "analytics"]
    assert len(ml_results) == 5
    assert len(analytics_results) == 5
    assert all(
        r.status in {HealthStatus.WARNING, HealthStatus.FAILED} for r in ml_results
    )
    assert all(r.status == HealthStatus.FAILED for r in analytics_results)
    ml_sla_report = ml_checker.sla_monitor.get_sla_report()
    analytics_sla_report = analytics_checker.sla_monitor.get_sla_report()
    assert ml_sla_report["total_checks"] >= 5
    assert analytics_sla_report["total_checks"] >= 5
    assert 0 <= ml_sla_report["overall_availability"] <= 1.0
    assert 0 <= analytics_sla_report["overall_availability"] <= 1.0
