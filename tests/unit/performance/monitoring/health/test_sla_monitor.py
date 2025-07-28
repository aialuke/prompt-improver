"""
Test SLA Monitor Implementation with Real Behavior
Testing actual SLA calculations and breach detection
"""

import pytest
import asyncio
import time
from typing import List, Tuple

from prompt_improver.performance.monitoring.health.sla_monitor import (
    SLAMonitor,
    SLAConfiguration,
    SLATarget,
    SLAStatus,
    SLAMeasurement,
    get_or_create_sla_monitor
)


class TestSLAMeasurement:
    """Test SLA measurement calculations with real data"""
    
    def test_sla_measurement_percentile_calculations(self):
        """Test that percentile calculations are accurate"""
        # Create a response time SLA
        target = SLATarget(
            name="response_time_p95",
            description="95th percentile response time",
            target_value=100,  # 100ms target
            unit="ms",
            measurement_window_seconds=60
        )
        
        measurement = SLAMeasurement(target)
        
        # Add real response time data
        response_times = [
            10, 20, 30, 40, 50,  # Fast responses
            60, 70, 80, 90,      # Medium responses  
            95, 98, 99,          # Near threshold
            101, 105, 150        # Some breaches
        ]
        
        current_time = time.time()
        for i, rt in enumerate(response_times):
            # Spread measurements over time
            measurement.add_measurement(rt, current_time - 30 + i * 2)
        
        # Calculate p95
        current_value = measurement.get_current_value()
        
        # Manual p95 calculation for verification
        sorted_times = sorted(response_times)
        expected_p95 = sorted_times[int(len(sorted_times) * 0.95)]
        
        assert abs(current_value - expected_p95) < 1  # Within 1ms tolerance
        
        # Check compliance status
        status = measurement.get_compliance_status()
        assert status == SLAStatus.BREACHING  # p95 exceeds 100ms target
    
    def test_sla_measurement_sliding_window(self):
        """Test that measurement window correctly filters old data"""
        target = SLATarget(
            name="availability",
            description="Service availability",
            target_value=0.99,
            unit="percent",
            measurement_window_seconds=30  # 30 second window
        )
        
        measurement = SLAMeasurement(target)
        current_time = time.time()
        
        # Add old measurements (outside window)
        for i in range(5):
            measurement.add_measurement(0.5, current_time - 60 - i)  # 50% availability
        
        # Add recent measurements (inside window)
        for i in range(10):
            measurement.add_measurement(1.0, current_time - 15 + i)  # 100% availability
        
        # Should only consider recent measurements
        current_value = measurement.get_current_value()
        assert current_value == 1.0  # Only recent 100% measurements count
        
        status = measurement.get_compliance_status()
        assert status == SLAStatus.MEETING
    
    def test_sla_alert_cooldown(self):
        """Test that alert cooldown prevents spam"""
        target = SLATarget(
            name="error_rate",
            description="Error rate",
            target_value=0.01,  # 1% max
            unit="percent",
            alert_cooldown_seconds=5
        )
        
        measurement = SLAMeasurement(target)
        
        # Add measurements that breach SLA
        for i in range(10):
            measurement.add_measurement(0.05)  # 5% error rate
        
        # First alert should trigger
        assert measurement.should_alert() is True
        measurement.last_alert_time = time.time()
        
        # Immediate second check should not alert (cooldown)
        assert measurement.should_alert() is False
        
        # Wait for cooldown
        time.sleep(target.alert_cooldown_seconds + 0.1)
        
        # Now should alert again
        assert measurement.should_alert() is True


class TestSLAMonitor:
    """Test SLA Monitor with real health check scenarios"""
    
    def test_sla_monitor_tracks_response_times(self):
        """Test that SLA monitor accurately tracks response time percentiles"""
        config = SLAConfiguration(
            service_name="test_service",
            response_time_p50_ms=50,
            response_time_p95_ms=100,
            response_time_p99_ms=200
        )
        
        monitor = SLAMonitor("test_service", config)
        
        # Simulate realistic response time distribution
        # 70% under 45ms, 26% between 45-95ms, 3% between 95-150ms, 1% over 200ms  
        response_times = (
            [25] * 35 + [40] * 35 +  # 70% fast (median will be 40ms)
            [75] * 20 + [95] * 6 +   # 26% medium (P95 will be around 95ms)
            [130] * 3 +              # 3% slow
            [250]                    # 1% very slow
        )
        
        # Record health checks
        for rt in response_times:
            monitor.record_health_check(
                success=True,
                response_time_ms=rt
            )
        
        # Get SLA report
        report = monitor.get_sla_report()
        
        # Verify calculations
        p50_data = report["sla_targets"]["response_time_p50"]
        p95_data = report["sla_targets"]["response_time_p95"]
        p99_data = report["sla_targets"]["response_time_p99"]
        
        # P50 should be around 45-50ms
        assert 40 <= p50_data["current_value"] <= 60
        assert p50_data["status"] == SLAStatus.MEETING.value
        
        # P95 should be around 95-100ms
        assert 90 <= p95_data["current_value"] <= 110
        assert p95_data["status"] in [SLAStatus.MEETING.value, SLAStatus.AT_RISK.value]
        
        # P99 should be around 150-200ms
        assert 140 <= p99_data["current_value"] <= 260
        # Could be at risk or breaching depending on exact calculation
    
    def test_sla_monitor_availability_tracking(self):
        """Test accurate availability calculation"""
        config = SLAConfiguration(
            service_name="availability_test",
            availability_target=0.99  # 99% target
        )
        
        monitor = SLAMonitor("availability_test", config)
        
        # Simulate 1000 health checks with 98.5% success rate
        success_count = 985
        failure_count = 15
        
        for _ in range(success_count):
            monitor.record_health_check(success=True, response_time_ms=50)
        
        for _ in range(failure_count):
            monitor.record_health_check(success=False, response_time_ms=50)
        
        report = monitor.get_sla_report()
        
        # Verify availability calculation
        assert report["total_checks"] == 1000
        assert report["successful_checks"] == success_count
        assert abs(report["overall_availability"] - 0.985) < 0.001
        
        # Should be at risk (98.5% < 99% target)
        availability_sla = report["sla_targets"]["availability"]
        assert availability_sla["status"] == SLAStatus.AT_RISK.value
    
    def test_sla_breach_callback(self):
        """Test that SLA breach callbacks are triggered correctly"""
        breaches_detected = []
        
        def on_breach(service_name: str, target: SLATarget, current_value: float):
            breaches_detected.append({
                "service": service_name,
                "target": target.name,
                "target_value": target.target_value,
                "current_value": current_value
            })
        
        config = SLAConfiguration(
            service_name="breach_test",
            response_time_p50_ms=50,
            error_rate_threshold=0.01  # 1% max errors
        )
        
        monitor = SLAMonitor("breach_test", config, on_sla_breach=on_breach)
        
        # Generate data that will breach error rate SLA
        for i in range(100):
            # 5% error rate (breaches 1% threshold)
            success = i % 20 != 0
            monitor.record_health_check(
                success=success,
                response_time_ms=30  # Good response time
            )
        
        # Should have triggered breach callback
        assert len(breaches_detected) > 0
        
        # Find error rate breach
        error_breach = next(
            b for b in breaches_detected 
            if b["target"] == "error_rate"
        )
        
        assert error_breach["service"] == "breach_test"
        assert error_breach["target_value"] == 0.01
        assert error_breach["current_value"] > 0.01  # Breaching
    
    def test_custom_sla_targets(self):
        """Test custom SLA targets work correctly"""
        config = SLAConfiguration(
            service_name="custom_test",
            custom_targets=[
                SLATarget(
                    name="queue_depth",
                    description="Message queue depth",
                    target_value=1000,  # Max 1000 messages
                    unit="count",
                    warning_threshold=0.8  # Warn at 800
                ),
                SLATarget(
                    name="cache_hit_rate",
                    description="Cache hit rate",
                    target_value=0.90,  # 90% minimum
                    unit="percent",
                    critical_threshold=1.11  # Below 90% is critical (inverse)
                )
            ]
        )
        
        monitor = SLAMonitor("custom_test", config)
        
        # Record health checks with custom metrics
        for i in range(50):
            queue_depth = 650 + i * 10  # Gradually increase queue (650 to 1140, mean=895)
            cache_hit_rate = 0.95 - i * 0.002  # Gradually decrease hit rate
            
            monitor.record_health_check(
                success=True,
                response_time_ms=50,
                custom_metrics={
                    "queue_depth": queue_depth,
                    "cache_hit_rate": cache_hit_rate
                }
            )
        
        report = monitor.get_sla_report()
        
        # Check queue depth (should be at risk)
        queue_sla = report["sla_targets"]["queue_depth"]
        assert queue_sla["current_value"] > 800  # Above warning
        assert queue_sla["status"] in [SLAStatus.AT_RISK.value, SLAStatus.BREACHING.value]
        
        # Check cache hit rate (should be meeting or at risk)
        cache_sla = report["sla_targets"]["cache_hit_rate"]
        assert 0.85 <= cache_sla["current_value"] <= 0.95
    
    def test_sla_metrics_export_format(self):
        """Test metrics export format for Prometheus compatibility"""
        config = SLAConfiguration(service_name="export_test")
        monitor = SLAMonitor("export_test", config)
        
        # Add some data
        for i in range(10):
            monitor.record_health_check(
                success=i % 10 != 0,  # 90% success
                response_time_ms=40 + i * 5
            )
        
        metrics = monitor.get_sla_metrics_for_export()
        
        # Verify Prometheus-compatible metric names and values
        assert "sla_response_time_p50_current" in metrics
        assert "sla_response_time_p50_target" in metrics
        assert "sla_response_time_p50_compliance_ratio" in metrics
        assert "sla_response_time_p50_status" in metrics
        
        # Status should be numeric
        assert isinstance(metrics["sla_response_time_p50_status"], (int, float))
        assert metrics["sla_response_time_p50_status"] in [0, 1, 2]
        
        # Compliance ratio should be 0-1
        assert 0 <= metrics["sla_response_time_p50_compliance_ratio"] <= 1
    
    def test_sla_monitor_registry(self):
        """Test global SLA monitor registry"""
        # Create monitor through registry
        monitor1 = get_or_create_sla_monitor("service1")
        assert monitor1.service_name == "service1"
        
        # Get same monitor
        monitor2 = get_or_create_sla_monitor("service1")
        assert monitor1 is monitor2
        
        # Create with custom config
        custom_config = SLAConfiguration(
            service_name="service2",
            response_time_p50_ms=25
        )
        monitor3 = get_or_create_sla_monitor("service2", custom_config)
        assert monitor3.config.response_time_p50_ms == 25


@pytest.mark.asyncio
async def test_sla_monitor_concurrent_updates():
    """Test SLA monitor under concurrent access"""
    config = SLAConfiguration(
        service_name="concurrent_test",
        response_time_p50_ms=100
    )
    
    monitor = SLAMonitor("concurrent_test", config)
    
    async def health_check_task(task_id: int):
        for i in range(100):
            # Vary response times by task
            response_time = 50 + task_id * 10 + i % 20
            success = i % 15 != 0  # ~93% success rate
            
            monitor.record_health_check(
                success=success,
                response_time_ms=response_time
            )
            
            await asyncio.sleep(0.001)  # Small delay
    
    # Run concurrent tasks
    tasks = [health_check_task(i) for i in range(5)]
    await asyncio.gather(*tasks)
    
    # Verify consistency
    report = monitor.get_sla_report()
    
    assert report["total_checks"] == 500  # 5 tasks * 100 checks
    assert 0.9 <= report["overall_availability"] <= 0.95  # ~93% expected
    
    # Response times should be reasonable
    p50 = report["sla_targets"]["response_time_p50"]["current_value"]
    assert 50 <= p50 <= 150  # Reasonable range given our data


def test_sla_calculation_edge_cases():
    """Test SLA calculations handle edge cases correctly"""
    config = SLAConfiguration(service_name="edge_case_test")
    monitor = SLAMonitor("edge_case_test", config)
    
    # Test with no data
    report = monitor.get_sla_report()
    assert report["total_checks"] == 0
    assert report["overall_availability"] == 0
    
    # Test with single data point
    monitor.record_health_check(success=True, response_time_ms=100)
    report = monitor.get_sla_report()
    
    assert report["total_checks"] == 1
    assert report["overall_availability"] == 1.0
    
    # All SLAs should have values
    for sla_name, sla_data in report["sla_targets"].items():
        assert sla_data["current_value"] is not None