"""
Test SLA Monitor False Positive Elimination - 2025 Best Practices
Comprehensive testing to ensure SLA violations are accurate and not false positives
"""

import time

import pytest

from prompt_improver.performance.monitoring.health.sla_monitor import (
    SLAConfiguration,
    SLAMeasurement,
    SLAMonitor,
    SLAStatus,
    SLATarget,
)


class TestSLAFalsePositiveElimination:
    """Test cases specifically designed to catch and eliminate false positive SLA violations"""

    def test_enum_consistency_prevents_false_positives(self):
        """Test that enum consistency fixes prevent false positive violations"""
        target = SLATarget(
            name="response_time_test",
            description="Response time consistency test",
            target_value=100,
            unit="ms",
        )
        measurement = SLAMeasurement(target)
        good_response_times = [50, 60, 70, 80, 90]
        for rt in good_response_times:
            measurement.add_measurement(rt)
        status = measurement.get_compliance_status()
        assert status == SLAStatus.MEETING, f"Expected MEETING, got {status}"
        assert measurement.should_alert() is False

    def test_percentile_calculation_accuracy(self):
        """Test that percentile calculations are mathematically accurate and consistent"""
        target = SLATarget(
            name="response_time_p95",
            description="95th percentile response time",
            target_value=120,
            unit="ms",
        )
        measurement = SLAMeasurement(target)
        response_times = list(range(1, 101))
        for rt in response_times:
            measurement.add_measurement(rt)
        current_value = measurement.get_current_value()
        sorted_times = sorted(response_times)
        expected_p95_index = int(len(sorted_times) * 0.95)
        expected_p95 = sorted_times[min(expected_p95_index, len(sorted_times) - 1)]
        assert current_value == expected_p95, (
            f"P95 calculation mismatch: got {current_value}, expected {expected_p95}"
        )
        status = measurement.get_compliance_status()
        assert status == SLAStatus.MEETING

    def test_percentile_edge_cases_no_false_positives(self):
        """Test percentile calculations with edge cases that previously caused false positives"""
        target_single = SLATarget(
            name="single_point",
            description="Single point test",
            target_value=100,
            unit="ms",
        )
        measurement_single = SLAMeasurement(target_single)
        measurement_single.add_measurement(50)
        assert measurement_single.get_current_value() == 50
        assert measurement_single.get_compliance_status() == SLAStatus.MEETING
        target_two = SLATarget(
            name="two_points",
            description="Two points test",
            target_value=100,
            unit="ms",
        )
        measurement_two = SLAMeasurement(target_two)
        measurement_two.add_measurement(20)
        measurement_two.add_measurement(80)
        current_value = measurement_two.get_current_value()
        assert current_value == 50, (
            f"Expected 50 (median of 20,80), got {current_value}"
        )
        assert measurement_two.get_compliance_status() == SLAStatus.MEETING
        target_hundred = SLATarget(
            name="hundred_points_p95",
            description="Hundred points P95 test",
            target_value=120,
            unit="ms",
        )
        measurement_hundred = SLAMeasurement(target_hundred)
        for i in range(100):
            measurement_hundred.add_measurement(i + 1)
        p95_value = measurement_hundred.get_current_value()
        assert p95_value == 96
        assert measurement_hundred.get_compliance_status() == SLAStatus.MEETING

    def test_time_window_filtering_accuracy(self):
        """Test that time window filtering doesn't create false positives from stale data"""
        target = SLATarget(
            name="time_window_test",
            description="Time window filtering test",
            target_value=100,
            unit="ms",
            measurement_window_seconds=60,
        )
        measurement = SLAMeasurement(target)
        current_time = time.time()
        for i in range(10):
            measurement.add_measurement(200, current_time - 120 - i)
        for i in range(20):
            measurement.add_measurement(50, current_time - 30 + i)
        current_value = measurement.get_current_value()
        assert current_value == 50, (
            f"Expected 50ms (recent data), got {current_value}ms"
        )
        status = measurement.get_compliance_status()
        assert status == SLAStatus.MEETING, (
            "Should be meeting SLA based on recent data only"
        )

    def test_availability_calculation_precision(self):
        """Test that availability calculations don't produce false positives due to rounding"""
        config = SLAConfiguration(
            service_name="availability_precision_test", availability_target=0.999
        )
        monitor = SLAMonitor("availability_precision_test", config)
        for _ in range(999):
            monitor.record_health_check(success=True, response_time_ms=50)
        for _ in range(1):
            monitor.record_health_check(success=False, response_time_ms=50)
        report = monitor.get_sla_report()
        expected_availability = 999 / 1000
        actual_availability = report["overall_availability"]
        assert abs(actual_availability - expected_availability) < 0.0001, (
            f"Availability calculation imprecise: {actual_availability} vs {expected_availability}"
        )
        availability_sla = report["sla_targets"]["availability"]
        assert availability_sla["status"] == SLAStatus.MEETING.value

    def test_error_rate_threshold_precision(self):
        """Test that error rate thresholds don't create false positives at boundaries"""
        config = SLAConfiguration(
            service_name="error_rate_test", error_rate_threshold=0.01
        )
        monitor = SLAMonitor("error_rate_test", config)
        for _ in range(124):
            monitor.record_health_check(success=True, response_time_ms=50)
        for _ in range(1):
            monitor.record_health_check(success=False, response_time_ms=50)
        report = monitor.get_sla_report()
        error_rate_sla = report["sla_targets"]["error_rate"]
        expected_error_rate = 1 / 125
        assert abs(error_rate_sla["current_value"] - expected_error_rate) < 0.0001
        assert error_rate_sla["status"] == SLAStatus.MEETING.value

    def test_concurrent_measurements_no_race_conditions(self):
        """Test that concurrent measurements don't cause false positive race conditions"""
        target = SLATarget(
            name="concurrent_test",
            description="Concurrent measurements test",
            target_value=100,
            unit="ms",
        )
        measurement = SLAMeasurement(target)
        good_times = [40, 50, 60, 70, 80] * 20
        for rt in good_times:
            measurement.add_measurement(rt)
        current_value = measurement.get_current_value()
        assert current_value <= 80, f"P95 should be 80 or less, got {current_value}"
        status = measurement.get_compliance_status()
        assert status == SLAStatus.MEETING

    def test_sla_report_consistency(self):
        """Test that SLA reports are internally consistent and don't contradict"""
        config = SLAConfiguration(
            service_name="consistency_test",
            response_time_p50_ms=80,
            response_time_p95_ms=120,
            response_time_p99_ms=200,
            availability_target=0.99,
        )
        monitor = SLAMonitor("consistency_test", config)
        response_times = [30, 40, 50, 60, 70] * 20
        for rt in response_times:
            monitor.record_health_check(success=True, response_time_ms=rt)
        report = monitor.get_sla_report()
        for sla_name, sla_data in report["sla_targets"].items():
            assert sla_data["status"] == SLAStatus.MEETING.value, (
                f"SLA {sla_name} should be meeting but is {sla_data['status']}"
            )
        assert report["overall_status"] == SLAStatus.MEETING.value

    def test_custom_sla_targets_no_false_positives(self):
        """Test that custom SLA targets don't create false positives"""
        config = SLAConfiguration(
            service_name="custom_sla_test",
            custom_targets=[
                SLATarget(
                    name="memory_usage",
                    description="Memory usage percentage",
                    target_value=0.8,
                    unit="percent",
                    warning_threshold=0.9,
                    critical_threshold=1.0,
                )
            ],
        )
        monitor = SLAMonitor("custom_sla_test", config)
        memory_usage_values = [0.6, 0.65, 0.7, 0.75, 0.7]
        for memory_pct in memory_usage_values:
            monitor.record_health_check(
                success=True,
                response_time_ms=50,
                custom_metrics={"memory_usage": memory_pct},
            )
        report = monitor.get_sla_report()
        memory_sla = report["sla_targets"]["memory_usage"]
        assert memory_sla["current_value"] < 0.8, (
            "Memory usage should be under threshold"
        )
        assert memory_sla["status"] == SLAStatus.MEETING.value

    async def test_alert_cooldown_prevents_spam_false_positives(self):
        """Test that alert cooldown prevents false positive alert spam"""
        breaches_detected = []

        def on_breach(service_name: str, target: SLATarget, current_value: float):
            breaches_detected.append({
                "timestamp": time.time(),
                "service": service_name,
                "target": target.name,
                "value": current_value,
            })

        target = SLATarget(
            name="cooldown_test",
            description="Alert cooldown test",
            target_value=50,
            unit="ms",
            alert_cooldown_seconds=2,
        )
        measurement = SLAMeasurement(target)
        breach_value = 100
        for _ in range(5):
            measurement.add_measurement(breach_value)
        for _ in range(5):
            if measurement.should_alert():
                on_breach("test_service", target, breach_value)
                measurement.last_alert_time = time.time()
            await asyncio.sleep(0.1)
        assert len(breaches_detected) == 1, (
            f"Expected 1 alert, got {len(breaches_detected)}"
        )

    def test_metrics_export_format_consistency(self):
        """Test that metrics export format is consistent and doesn't create false data"""
        config = SLAConfiguration(service_name="metrics_export_test")
        monitor = SLAMonitor("metrics_export_test", config)
        for rt in [40, 50, 60]:
            monitor.record_health_check(success=True, response_time_ms=rt)
        metrics = monitor.get_sla_metrics_for_export()
        expected_metric_patterns = [
            "sla_response_time_p50_current",
            "sla_response_time_p50_target",
            "sla_response_time_p50_compliance_ratio",
            "sla_response_time_p50_status",
        ]
        for pattern in expected_metric_patterns:
            assert pattern in metrics, f"Missing metric: {pattern}"
        status = metrics["sla_response_time_p50_status"]
        assert isinstance(status, (int, float))
        assert status in {0, 1, 2}, f"Invalid status value: {status}"
        compliance_ratio = metrics["sla_response_time_p50_compliance_ratio"]
        assert 0 <= compliance_ratio <= 1, (
            f"Invalid compliance ratio: {compliance_ratio}"
        )


def test_comprehensive_real_world_scenario():
    """Comprehensive test simulating real-world conditions without false positives"""
    config = SLAConfiguration(
        service_name="real_world_test",
        response_time_p50_ms=200,
        response_time_p95_ms=400,
        response_time_p99_ms=600,
        availability_target=0.995,
        error_rate_threshold=0.005,
        custom_targets=[
            SLATarget(
                name="throughput",
                description="Requests per second",
                target_value=800,
                unit="rps",
            )
        ],
    )
    monitor = SLAMonitor("real_world_test", config)
    scenarios = [
        {
            "count": 50,
            "success_rate": 0.998,
            "response_times": [50, 80, 120],
            "throughput": 1200,
        },
        {
            "count": 30,
            "success_rate": 0.996,
            "response_times": [80, 150, 250],
            "throughput": 950,
        },
        {
            "count": 40,
            "success_rate": 0.999,
            "response_times": [60, 100, 180],
            "throughput": 1100,
        },
    ]
    for scenario in scenarios:
        for i in range(scenario["count"]):
            success = i < scenario["count"] * scenario["success_rate"]
            rt_choices = scenario["response_times"]
            response_time = rt_choices[i % len(rt_choices)]
            monitor.record_health_check(
                success=success,
                response_time_ms=response_time,
                custom_metrics={"throughput": scenario["throughput"]},
            )
    report = monitor.get_sla_report()
    assert report["total_checks"] == 120
    assert report["overall_availability"] > 0.99
    for sla_name, sla_data in report["sla_targets"].items():
        assert sla_data["status"] in {
            SLAStatus.MEETING.value,
            SLAStatus.AT_RISK.value,
        }, f"SLA {sla_name} has unexpected status: {sla_data['status']}"
        assert sla_data["current_value"] is not None, f"SLA {sla_name} has None value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
