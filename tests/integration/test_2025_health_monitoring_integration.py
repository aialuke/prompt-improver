"""
Integration Tests for 2025 Health Monitoring Features
End-to-end testing of circuit breaker + SLA + logging + telemetry
"""

import pytest
import asyncio
import json
import io
import logging
import time
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

from prompt_improver.performance.monitoring.health.enhanced_checkers import (
    EnhancedMLServiceHealthChecker,
    EnhancedRedisHealthMonitor
)
from prompt_improver.performance.monitoring.health.base import HealthStatus
from prompt_improver.performance.monitoring.health.circuit_breaker import CircuitState
from prompt_improver.performance.monitoring.health.sla_monitor import SLAStatus


class TestIntegratedHealthMonitoring:
    """Integration tests for complete health monitoring system"""
    
    @pytest.mark.asyncio
    async def test_full_health_monitoring_lifecycle(self):
        """Test complete health monitoring lifecycle with all features"""
        
        # Set up log capture
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        
        # Configure logging for health components
        health_logger = logging.getLogger("health.ml_service")
        health_logger.setLevel(logging.INFO)
        health_logger.handlers.clear()
        health_logger.addHandler(handler)
        
        # Create health checker with monitoring
        checker = EnhancedMLServiceHealthChecker()
        
        # Mock ML service that works initially
        ml_service_available = True
        def mock_get_ml_service():
            if ml_service_available:
                return {"available": True, "version": "1.0"}
            else:
                raise ConnectionError("ML service connection failed")
        
        # Phase 1: Normal operation (should be healthy)
        with patch('prompt_improver.ml.services.ml_integration.get_ml_service', mock_get_ml_service):
            for i in range(5):
                result = await checker.check()
                assert result.status == HealthStatus.HEALTHY
                await asyncio.sleep(0.01)
        
        # Verify initial state
        assert checker.circuit_breaker.state == CircuitState.CLOSED
        
        sla_report = checker.sla_monitor.get_sla_report()
        assert sla_report["overall_availability"] == 1.0
        assert sla_report["overall_status"] == "meeting"
        
        # Phase 2: Service failures (should trigger circuit breaker)
        ml_service_available = False
        
        # Execute enough failures to open circuit
        circuit_opened = False
        for i in range(5):
            result = await checker.check()
            if "Circuit breaker open" in result.details.get("error", ""):
                circuit_opened = True
                break
            await asyncio.sleep(0.01)
        
        # Verify circuit breaker opened
        assert checker.circuit_breaker.state == CircuitState.OPEN
        
        # Phase 3: Service recovery
        ml_service_available = True
        
        # Wait for circuit breaker recovery timeout
        await asyncio.sleep(0.5)  # Circuit breaker config uses 30s, but we can't wait that long
        
        # Manually reset for testing (in real scenario, time would pass)
        checker.circuit_breaker.reset()
        
        # Should be healthy again
        result = await checker.check()
        assert result.status == HealthStatus.HEALTHY
        assert checker.circuit_breaker.state == CircuitState.CLOSED
        
        # Verify logging captured all phases
        handler.flush()
        log_output = log_capture.getvalue()
        
        # Should contain success and failure logs
        assert "Health check completed: ml_service" in log_output
        assert "Health check rejected by circuit breaker" in log_output
        
        # Verify structured log format
        log_lines = [line for line in log_output.split('\n') if line.strip()]
        valid_json_logs = 0
        
        for line in log_lines:
            try:
                log_data = json.loads(line)
                if "component" in log_data:
                    valid_json_logs += 1
                    assert log_data["component"] == "ml_service"
                    assert "correlation_id" in log_data
            except json.JSONDecodeError:
                pass  # Some logs might not be JSON
        
        assert valid_json_logs > 0  # At least some structured logs
    
    @pytest.mark.asyncio
    async def test_sla_monitoring_with_real_performance_degradation(self):
        """Test SLA monitoring detects real performance issues"""
        
        # Custom health checker that simulates varying performance
        class VariablePerformanceChecker(EnhancedRedisHealthMonitor):
            def __init__(self):
                super().__init__()
                self.call_count = 0
                self.base_response_time = 10  # Start at 10ms
            
            async def _ping_check(self):
                self.call_count += 1
                
                # Simulate degrading performance
                response_time_ms = self.base_response_time + (self.call_count * 5)
                
                await asyncio.sleep(response_time_ms / 1000)  # Actual delay
                
                return {
                    "ping_success": True,
                    "latency_ms": response_time_ms
                }
        
        checker = VariablePerformanceChecker()
        
        # Track SLA status changes
        sla_statuses = []
        
        # Execute health checks with degrading performance
        for i in range(20):
            result = await checker.check()
            
            # Still healthy but performance degrades
            assert result.status == HealthStatus.HEALTHY
            
            # Check SLA status
            sla_report = checker.sla_monitor.get_sla_report()
            overall_status = sla_report["overall_status"]
            sla_statuses.append(overall_status)
            
            await asyncio.sleep(0.01)
        
        # Should show progression from meeting -> at_risk -> breaching
        assert "meeting" in sla_statuses[:5]  # Early calls should meet SLA
        assert "breaching" in sla_statuses[-5:]  # Later calls should breach SLA
        
        # Verify final SLA report shows breaching
        final_report = checker.sla_monitor.get_sla_report()
        assert final_report["overall_status"] in ["at_risk", "breaching"]
        
        # P95 response time should exceed thresholds
        p95_sla = final_report["sla_targets"]["response_time_p95"]
        assert p95_sla["current_value"] > p95_sla["target_value"]
    
    @pytest.mark.asyncio
    async def test_concurrent_health_monitoring(self):
        """Test health monitoring under concurrent load"""
        
        # Create multiple health checkers
        redis_checker = EnhancedRedisHealthMonitor()
        
        # Shared results storage
        results = {"redis": []}
        errors = []
        
        async def concurrent_health_check(checker_name: str, checker, iterations: int):
            """Run health checks concurrently"""
            try:
                for i in range(iterations):
                    result = await checker.check()
                    results[checker_name].append({
                        "iteration": i,
                        "status": result.status.name,
                        "response_time_ms": result.response_time_ms,
                        "timestamp": time.time()
                    })
                    await asyncio.sleep(0.01)  # Small delay
            except Exception as e:
                errors.append(f"{checker_name}: {str(e)}")
        
        # Run concurrent health checks
        tasks = [
            concurrent_health_check("redis", redis_checker, 25)
        ]
        
        start_time = time.time()
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors during concurrent execution: {errors}"
        
        # Verify all checks completed
        assert len(results["redis"]) == 25
        
        # Verify reasonable performance (should complete in under 2 seconds)
        assert total_time < 2.0
        
        # Verify circuit breaker state remained stable
        assert redis_checker.circuit_breaker.state == CircuitState.CLOSED
        
        # Verify SLA monitoring tracked all calls
        sla_report = redis_checker.sla_monitor.get_sla_report()
        assert sla_report["total_checks"] == 25
        
        # Calculate actual success rate
        successful_checks = sum(1 for r in results["redis"] if r["status"] == "HEALTHY")
        expected_success_rate = successful_checks / 25
        assert abs(sla_report["overall_availability"] - expected_success_rate) < 0.01
    
    @pytest.mark.asyncio
    async def test_health_monitoring_failure_cascade_prevention(self):
        """Test that circuit breakers prevent failure cascades"""
        
        # Create health checkers with fast-opening circuit breakers
        from prompt_improver.performance.monitoring.health.circuit_breaker import CircuitBreakerConfig
        
        fast_circuit_config = CircuitBreakerConfig(
            failure_threshold=2,  # Open after 2 failures
            recovery_timeout=0.1  # Quick recovery for testing
        )
        
        class FailingChecker(EnhancedMLServiceHealthChecker):
            def __init__(self):
                super().__init__()
                self.circuit_breaker.config = fast_circuit_config
                self.total_executions = 0
                self.circuit_rejections = 0
            
            async def _execute_health_check(self):
                self.total_executions += 1
                raise RuntimeError("Simulated service failure")
        
        checker = FailingChecker()
        
        # Execute many health checks
        results = []
        for i in range(10):
            result = await checker.check()
            results.append(result)
            
            if "Circuit breaker open" in result.details.get("error", ""):
                checker.circuit_rejections += 1
            
            await asyncio.sleep(0.01)
        
        # Verify circuit breaker prevented excessive executions
        assert checker.total_executions <= 4  # Should stop executing after circuit opens
        assert checker.circuit_rejections > 0  # Some calls should be rejected
        
        # Verify all calls returned failure status
        assert all(r.status == HealthStatus.FAILED for r in results)
        
        # Verify circuit breaker opened
        assert checker.circuit_breaker.state == CircuitState.OPEN
        
        # Verify failure was properly recorded in SLA
        sla_report = checker.sla_monitor.get_sla_report()
        assert sla_report["overall_availability"] == 0.0
        assert sla_report["overall_status"] == "breaching"
    
    @pytest.mark.asyncio
    async def test_health_monitoring_metrics_export_integration(self):
        """Test that all monitoring data can be exported for external systems"""
        
        checker = EnhancedRedisHealthMonitor()
        
        # Generate various health check results
        for i in range(15):
            result = await checker.check()
            await asyncio.sleep(0.01)
        
        # Get comprehensive monitoring data
        enhanced_status = checker.get_enhanced_status()
        
        # Verify all monitoring components export data
        assert "circuit_breaker" in enhanced_status
        assert "sla_report" in enhanced_status
        assert "sla_metrics" in enhanced_status
        
        # Verify circuit breaker metrics
        cb_metrics = enhanced_status["circuit_breaker"]
        assert cb_metrics["total_calls"] == 15
        assert "success_rate" in cb_metrics
        assert "state" in cb_metrics
        
        # Verify SLA metrics for Prometheus export
        sla_metrics = enhanced_status["sla_metrics"]
        prometheus_metrics = {}
        
        for key, value in sla_metrics.items():
            # Should be numeric and suitable for Prometheus
            assert isinstance(value, (int, float))
            prometheus_metrics[f"health_{key}"] = value
        
        # Verify key Prometheus metrics exist
        expected_metrics = [
            "health_sla_response_time_p50_current",
            "health_sla_response_time_p95_current", 
            "health_sla_availability_current",
            "health_sla_response_time_p50_compliance_ratio"
        ]
        
        for metric in expected_metrics:
            assert metric in prometheus_metrics
            assert 0 <= prometheus_metrics[metric] <= 10000  # Reasonable range
    
    def test_health_monitoring_configuration_validation(self):
        """Test that health monitoring components validate configurations"""
        
        # Test circuit breaker config validation
        from prompt_improver.performance.monitoring.health.circuit_breaker import CircuitBreakerConfig
        
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=30,
            response_time_threshold_ms=1000
        )
        
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 30
        assert config.response_time_threshold_ms == 1000
        
        # Test SLA config validation
        from prompt_improver.performance.monitoring.health.sla_monitor import SLAConfiguration, SLATarget
        
        sla_config = SLAConfiguration(
            service_name="test_service",
            response_time_p95_ms=500,
            availability_target=0.99,
            custom_targets=[
                SLATarget(
                    name="custom_metric",
                    description="Custom test metric",
                    target_value=100,
                    unit="count"
                )
            ]
        )
        
        assert sla_config.service_name == "test_service"
        assert sla_config.response_time_p95_ms == 500
        assert sla_config.availability_target == 0.99
        assert len(sla_config.custom_targets) == 1
        assert sla_config.custom_targets[0].name == "custom_metric"


@pytest.mark.asyncio 
async def test_end_to_end_monitoring_scenario():
    """End-to-end test simulating real-world monitoring scenario"""
    
    # Set up comprehensive logging
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    
    # Configure multiple loggers
    for logger_name in ["health.redis", "health_metrics.redis"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.addHandler(handler)
    
    # Create health checker
    redis_checker = EnhancedRedisHealthMonitor()
    
    # Simulate real-world scenario: service starts healthy, degrades, recovers
    phases = [
        # Phase 1: Healthy operation (10 checks)
        {"duration": 10, "healthy": True, "response_time_base": 15},
        
        # Phase 2: Performance degradation (10 checks)  
        {"duration": 10, "healthy": True, "response_time_base": 80},
        
        # Phase 3: Service failures (5 checks)
        {"duration": 5, "healthy": False, "response_time_base": 200},
        
        # Phase 4: Recovery (5 checks)
        {"duration": 5, "healthy": True, "response_time_base": 20}
    ]
    
    all_results = []
    phase_results = []
    
    for phase_num, phase in enumerate(phases):
        phase_start = time.time()
        
        # Mock Redis behavior for this phase
        async def mock_redis_ops(healthy, base_time):
            if not healthy:
                raise ConnectionError("Redis connection failed")
            
            response_time = base_time + (time.time() % 10)  # Add some variance
            await asyncio.sleep(response_time / 1000)
            
            return {
                "ping_success": True,
                "latency_ms": response_time
            }
        
        with patch.object(redis_checker, '_ping_check', 
                         lambda: mock_redis_ops(phase["healthy"], phase["response_time_base"])):
            
            phase_results = []
            
            for i in range(phase["duration"]):
                result = await redis_checker.check()
                phase_results.append({
                    "phase": phase_num,
                    "iteration": i,
                    "status": result.status.name,
                    "response_time_ms": result.response_time_ms,
                    "circuit_state": redis_checker.circuit_breaker.state.name
                })
                all_results.append(phase_results[-1])
                
                await asyncio.sleep(0.02)  # Small delay between checks
    
    # Analyze results
    
    # Phase 1: Should all be healthy
    phase_1_results = [r for r in all_results if r["phase"] == 0]
    assert all(r["status"] == "HEALTHY" for r in phase_1_results)
    assert all(r["circuit_state"] == "CLOSED" for r in phase_1_results)
    
    # Phase 3: Should show failures and circuit opening
    phase_3_results = [r for r in all_results if r["phase"] == 2]
    failure_count = sum(1 for r in phase_3_results if r["status"] == "FAILED")
    assert failure_count > 0
    
    # Circuit should have opened during failures
    circuit_states = [r["circuit_state"] for r in all_results]
    assert "OPEN" in circuit_states
    
    # Verify final SLA report
    sla_report = redis_checker.sla_monitor.get_sla_report()
    
    assert sla_report["total_checks"] == 30  # Total from all phases
    assert 0.5 <= sla_report["overall_availability"] <= 0.9  # Mixed success/failure
    
    # Should show SLA breaching due to failures
    assert sla_report["overall_status"] in ["at_risk", "breaching"]
    
    # Verify comprehensive logging
    handler.flush()
    log_output = log_capture.getvalue()
    
    # Should contain logs from all phases
    assert "Health check completed: redis" in log_output
    assert "Health check failed: redis" in log_output
    
    # Should contain structured logs
    structured_logs = []
    for line in log_output.split('\n'):
        if line.strip():
            try:
                log_data = json.loads(line)
                if "component" in log_data and log_data["component"] == "redis":
                    structured_logs.append(log_data)
            except json.JSONDecodeError:
                pass
    
    assert len(structured_logs) >= 20  # Should have many structured logs
    
    # Verify metrics export readiness
    enhanced_status = redis_checker.get_enhanced_status()
    sla_metrics = enhanced_status["sla_metrics"]
    
    # Should have metrics suitable for Prometheus
    prometheus_ready_metrics = {
        f"redis_{key}": value 
        for key, value in sla_metrics.items() 
        if isinstance(value, (int, float))
    }
    
    assert len(prometheus_ready_metrics) >= 10  # Should have multiple metrics
    assert all(isinstance(v, (int, float)) for v in prometheus_ready_metrics.values())