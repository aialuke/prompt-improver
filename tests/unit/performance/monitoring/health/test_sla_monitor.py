"""
Test SLA Monitor Implementation with Real Behavior
Testing actual SLA calculations and breach detection
"""
import asyncio
import time
from typing import List, Tuple
import pytest
from prompt_improver.performance.monitoring.health.sla_monitor import SLAConfiguration, SLAMeasurement, SLAMonitor, SLAStatus, SLATarget, get_or_create_sla_monitor

class TestSLAMeasurement:
    """Test SLA measurement calculations with real data"""

    def test_sla_measurement_percentile_calculations(self):
        """Test that percentile calculations are accurate"""
        target = SLATarget(name='response_time_p95', description='95th percentile response time', target_value=100, unit='ms', measurement_window_seconds=60)
        measurement = SLAMeasurement(target)
        response_times = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99, 101, 105, 150]
        current_time = time.time()
        for i, rt in enumerate(response_times):
            measurement.add_measurement(rt, current_time - 30 + i * 2)
        current_value = measurement.get_current_value()
        sorted_times = sorted(response_times)
        expected_p95 = sorted_times[int(len(sorted_times) * 0.95)]
        assert abs(current_value - expected_p95) < 1
        status = measurement.get_compliance_status()
        assert status == SLAStatus.BREACHING

    def test_sla_measurement_sliding_window(self):
        """Test that measurement window correctly filters old data"""
        target = SLATarget(name='availability', description='Service availability', target_value=0.99, unit='percent', measurement_window_seconds=30)
        measurement = SLAMeasurement(target)
        current_time = time.time()
        for i in range(5):
            measurement.add_measurement(0.5, current_time - 60 - i)
        for i in range(10):
            measurement.add_measurement(1.0, current_time - 15 + i)
        current_value = measurement.get_current_value()
        assert current_value == 1.0
        status = measurement.get_compliance_status()
        assert status == SLAStatus.MEETING

    def test_sla_alert_cooldown(self):
        """Test that alert cooldown prevents spam"""
        target = SLATarget(name='error_rate', description='Error rate', target_value=0.01, unit='percent', alert_cooldown_seconds=5)
        measurement = SLAMeasurement(target)
        for i in range(10):
            measurement.add_measurement(0.05)
        assert measurement.should_alert() is True
        measurement.last_alert_time = time.time()
        assert measurement.should_alert() is False
        time.sleep(target.alert_cooldown_seconds + 0.1)
        assert measurement.should_alert() is True

class TestSLAMonitor:
    """Test SLA Monitor with real health check scenarios"""

    def test_sla_monitor_tracks_response_times(self):
        """Test that SLA monitor accurately tracks response time percentiles"""
        config = SLAConfiguration(service_name='test_service', response_time_p50_ms=50, response_time_p95_ms=100, response_time_p99_ms=200)
        monitor = SLAMonitor('test_service', config)
        response_times = [25] * 35 + [40] * 35 + [75] * 20 + [95] * 6 + [130] * 3 + [250]
        for rt in response_times:
            monitor.record_health_check(success=True, response_time_ms=rt)
        report = monitor.get_sla_report()
        p50_data = report['sla_targets']['response_time_p50']
        p95_data = report['sla_targets']['response_time_p95']
        p99_data = report['sla_targets']['response_time_p99']
        assert 40 <= p50_data['current_value'] <= 60
        assert p50_data['status'] == SLAStatus.MEETING.value
        assert 90 <= p95_data['current_value'] <= 110
        assert p95_data['status'] in [SLAStatus.MEETING.value, SLAStatus.AT_RISK.value]
        assert 140 <= p99_data['current_value'] <= 260

    def test_sla_monitor_availability_tracking(self):
        """Test accurate availability calculation"""
        config = SLAConfiguration(service_name='availability_test', availability_target=0.99)
        monitor = SLAMonitor('availability_test', config)
        success_count = 985
        failure_count = 15
        for _ in range(success_count):
            monitor.record_health_check(success=True, response_time_ms=50)
        for _ in range(failure_count):
            monitor.record_health_check(success=False, response_time_ms=50)
        report = monitor.get_sla_report()
        assert report['total_checks'] == 1000
        assert report['successful_checks'] == success_count
        assert abs(report['overall_availability'] - 0.985) < 0.001
        availability_sla = report['sla_targets']['availability']
        assert availability_sla['status'] == SLAStatus.AT_RISK.value

    def test_sla_breach_callback(self):
        """Test that SLA breach callbacks are triggered correctly"""
        breaches_detected = []

        def on_breach(service_name: str, target: SLATarget, current_value: float):
            breaches_detected.append({'service': service_name, 'target': target.name, 'target_value': target.target_value, 'current_value': current_value})
        config = SLAConfiguration(service_name='breach_test', response_time_p50_ms=50, error_rate_threshold=0.01)
        monitor = SLAMonitor('breach_test', config, on_sla_breach=on_breach)
        for i in range(100):
            success = i % 20 != 0
            monitor.record_health_check(success=success, response_time_ms=30)
        assert len(breaches_detected) > 0
        error_breach = next((b for b in breaches_detected if b['target'] == 'error_rate'))
        assert error_breach['service'] == 'breach_test'
        assert error_breach['target_value'] == 0.01
        assert error_breach['current_value'] > 0.01

    def test_custom_sla_targets(self):
        """Test custom SLA targets work correctly"""
        config = SLAConfiguration(service_name='custom_test', custom_targets=[SLATarget(name='queue_depth', description='Message queue depth', target_value=1000, unit='count', warning_threshold=0.8), SLATarget(name='cache_hit_rate', description='Cache hit rate', target_value=0.9, unit='percent', critical_threshold=1.11)])
        monitor = SLAMonitor('custom_test', config)
        for i in range(50):
            queue_depth = 650 + i * 10
            cache_hit_rate = 0.95 - i * 0.002
            monitor.record_health_check(success=True, response_time_ms=50, custom_metrics={'queue_depth': queue_depth, 'cache_hit_rate': cache_hit_rate})
        report = monitor.get_sla_report()
        queue_sla = report['sla_targets']['queue_depth']
        assert queue_sla['current_value'] > 800
        assert queue_sla['status'] in [SLAStatus.AT_RISK.value, SLAStatus.BREACHING.value]
        cache_sla = report['sla_targets']['cache_hit_rate']
        assert 0.85 <= cache_sla['current_value'] <= 0.95

    def test_sla_metrics_export_format(self):
        """Test metrics export format for Prometheus compatibility"""
        config = SLAConfiguration(service_name='export_test')
        monitor = SLAMonitor('export_test', config)
        for i in range(10):
            monitor.record_health_check(success=i % 10 != 0, response_time_ms=40 + i * 5)
        metrics = monitor.get_sla_metrics_for_export()
        assert 'sla_response_time_p50_current' in metrics
        assert 'sla_response_time_p50_target' in metrics
        assert 'sla_response_time_p50_compliance_ratio' in metrics
        assert 'sla_response_time_p50_status' in metrics
        assert isinstance(metrics['sla_response_time_p50_status'], (int, float))
        assert metrics['sla_response_time_p50_status'] in [0, 1, 2]
        assert 0 <= metrics['sla_response_time_p50_compliance_ratio'] <= 1

    def test_sla_monitor_registry(self):
        """Test global SLA monitor registry"""
        monitor1 = get_or_create_sla_monitor('service1')
        assert monitor1.service_name == 'service1'
        monitor2 = get_or_create_sla_monitor('service1')
        assert monitor1 is monitor2
        custom_config = SLAConfiguration(service_name='service2', response_time_p50_ms=25)
        monitor3 = get_or_create_sla_monitor('service2', custom_config)
        assert monitor3.config.response_time_p50_ms == 25

@pytest.mark.asyncio
async def test_sla_monitor_concurrent_updates():
    """Test SLA monitor under concurrent access"""
    config = SLAConfiguration(service_name='concurrent_test', response_time_p50_ms=100)
    monitor = SLAMonitor('concurrent_test', config)

    async def health_check_task(task_id: int):
        for i in range(100):
            response_time = 50 + task_id * 10 + i % 20
            success = i % 15 != 0
            monitor.record_health_check(success=success, response_time_ms=response_time)
            await asyncio.sleep(0.001)
    tasks = [health_check_task(i) for i in range(5)]
    await asyncio.gather(*tasks)
    report = monitor.get_sla_report()
    assert report['total_checks'] == 500
    assert 0.9 <= report['overall_availability'] <= 0.95
    p50 = report['sla_targets']['response_time_p50']['current_value']
    assert 50 <= p50 <= 150

def test_sla_calculation_edge_cases():
    """Test SLA calculations handle edge cases correctly"""
    config = SLAConfiguration(service_name='edge_case_test')
    monitor = SLAMonitor('edge_case_test', config)
    report = monitor.get_sla_report()
    assert report['total_checks'] == 0
    assert report['overall_availability'] == 0
    monitor.record_health_check(success=True, response_time_ms=100)
    report = monitor.get_sla_report()
    assert report['total_checks'] == 1
    assert report['overall_availability'] == 1.0
    for sla_name, sla_data in report['sla_targets'].items():
        assert sla_data['current_value'] is not None
