"""Comprehensive real behavior tests for CacheMonitoringService with direct cache architecture.

Validates all monitoring functionality using real CacheFacade instances with direct L1+L2 operations.
Tests the new high-performance cache architecture that eliminates coordination overhead while
maintaining comprehensive monitoring capabilities.

Architecture Monitored:
- Direct L1+L2 cache-aside pattern through CacheFacade
- CacheFactory singleton pattern optimization
- Eliminated coordination layer (no more 775x performance degradation)

Critical validation areas:
- Health check operations with direct cache facade
- SLO compliance calculation for <2ms response times
- Monitoring performance overhead (<1ms requirement)
- Performance improvement validation (25x cache access speed)
- Alert metrics for direct cache operations
- Error handling and graceful degradation
- CacheFactory monitoring integration
"""

import asyncio
import logging
import statistics
import time

import pytest
from tests.containers.real_redis_container import RealRedisTestContainer

from prompt_improver.services.cache.cache_facade import CacheFacade
from prompt_improver.services.cache.cache_factory import CacheFactory
from prompt_improver.services.cache.cache_monitoring_service import (
    CacheMonitoringService,
)

logger = logging.getLogger(__name__)


class TestCacheMonitoringServiceDirectArchitecture:
    """Comprehensive real behavior tests for CacheMonitoringService with direct cache operations."""

    @pytest.fixture
    async def redis_container(self):
        """Real Redis testcontainer for L2 cache."""
        container = RealRedisTestContainer()
        await container.start()
        container.set_env_vars()
        yield container
        await container.stop()

    @pytest.fixture
    async def cache_facade_healthy(self, redis_container):
        """Healthy cache facade with L1+L2 for monitoring."""
        cache = CacheFacade(
            l1_max_size=200,
            l2_default_ttl=3600,
            enable_l2=True,   # Full L1+L2 cache-aside pattern
        )
        yield cache
        await cache.clear()
        await cache.close()

    @pytest.fixture
    async def cache_facade_degraded(self):
        """Degraded cache facade (L1-only) for monitoring failure scenarios."""
        cache = CacheFacade(
            l1_max_size=100,
            enable_l2=False,  # L2 disabled - degraded scenario
        )
        yield cache
        await cache.clear()
        await cache.close()

    @pytest.fixture
    async def cache_monitoring_service(self, cache_facade_healthy):
        """CacheMonitoringService configured for direct cache monitoring."""
        # Create monitoring service with cache facade
        monitoring = CacheMonitoringService(
            cache_service=cache_facade_healthy,
            health_check_interval=1.0,
            performance_window_size=100,
            enable_opentelemetry=False,  # Disabled for testing
        )

        # Start monitoring
        await monitoring.start()
        yield monitoring
        await monitoring.stop()

    async def test_monitoring_direct_cache_health_checks(self, cache_monitoring_service, cache_facade_healthy):
        """Test health check monitoring with direct cache operations."""
        # Perform cache operations to generate health data
        for i in range(10):
            await cache_facade_healthy.set(f"health_key_{i}", {"health_test": i})
            await cache_facade_healthy.get(f"health_key_{i}")

        # Allow monitoring to collect data
        await asyncio.sleep(1.5)

        # Test health check monitoring
        start_time = time.perf_counter()
        health_report = await cache_monitoring_service.get_health_report()
        monitoring_time = time.perf_counter() - start_time

        # Validate monitoring performance (<1ms requirement)
        assert monitoring_time < 0.001, f"Health monitoring took {monitoring_time * 1000:.2f}ms, exceeds 1ms"

        # Validate health report structure for direct cache
        assert health_report["status"] in {"healthy", "degraded", "unhealthy"}
        assert "timestamp" in health_report
        assert "cache_health" in health_report
        assert "performance_metrics" in health_report

        # Validate direct cache specific metrics
        cache_health = health_report["cache_health"]
        assert "l1_cache" in cache_health
        assert "l2_cache" in cache_health
        assert cache_health["l1_cache"]["status"] == "healthy"
        assert cache_health["l2_cache"]["status"] == "healthy"

        # Validate performance metrics for direct operations
        perf_metrics = health_report["performance_metrics"]
        assert "avg_response_time_ms" in perf_metrics
        assert "cache_hit_rate" in perf_metrics
        assert "architecture" in perf_metrics

        # Validate performance targets
        assert perf_metrics["avg_response_time_ms"] < 2.0, f"Average response time {perf_metrics['avg_response_time_ms']:.3f}ms exceeds 2ms target"
        assert perf_metrics["architecture"] == "direct_cache_aside_pattern"

        print(f"Direct cache health monitoring: {monitoring_time * 1000:.3f}ms, hit rate: {perf_metrics['cache_hit_rate']:.2%}")

    async def test_monitoring_performance_slo_compliance(self, cache_monitoring_service, cache_facade_healthy):
        """Test SLO compliance monitoring for direct cache performance targets."""
        # Generate performance data with direct cache operations
        operation_times = []
        for i in range(50):
            start_time = time.perf_counter()
            await cache_facade_healthy.set(f"slo_key_{i}", {"slo_test": i})
            result = await cache_facade_healthy.get(f"slo_key_{i}")
            operation_time = time.perf_counter() - start_time
            operation_times.append(operation_time)
            assert result is not None

        # Allow monitoring to process data
        await asyncio.sleep(2.0)

        # Test SLO compliance calculation
        slo_report = await cache_monitoring_service.get_slo_compliance_report()

        # Validate SLO structure
        assert "slo_status" in slo_report
        assert "performance_metrics" in slo_report
        assert "targets" in slo_report

        # Validate performance targets for direct cache
        targets = slo_report["targets"]
        assert targets["max_response_time_ms"] == 2.0  # Direct cache target
        assert targets["min_hit_rate"] >= 0.8

        # Validate SLO compliance
        metrics = slo_report["performance_metrics"]
        assert metrics["p95_response_time_ms"] < 2.0, f"P95 response time {metrics['p95_response_time_ms']:.3f}ms exceeds SLO"
        assert metrics["p99_response_time_ms"] < 5.0, f"P99 response time {metrics['p99_response_time_ms']:.3f}ms too high"

        # Validate percentile calculation accuracy (critical bug fix validation)
        calculated_p95 = statistics.quantiles(operation_times, n=20)[18] * 1000  # 95th percentile
        reported_p95 = metrics["p95_response_time_ms"]

        # Allow reasonable tolerance for percentile calculation
        tolerance = 0.5  # 0.5ms tolerance
        assert abs(calculated_p95 - reported_p95) < tolerance, f"P95 calculation error: expected {calculated_p95:.3f}ms, got {reported_p95:.3f}ms"

        print(f"SLO Compliance - P95: {metrics['p95_response_time_ms']:.3f}ms, Hit Rate: {metrics['hit_rate']:.2%}")

    async def test_monitoring_cache_factory_integration(self, redis_container):
        """Test monitoring integration with CacheFactory singleton pattern."""
        # Clear factory to start fresh
        CacheFactory.clear_instances()

        # Get different cache types from factory
        utility_cache = CacheFactory.get_utility_cache()
        ml_cache = CacheFactory.get_ml_analysis_cache()

        # Create monitoring for factory caches
        utility_monitoring = CacheMonitoringService(
            cache_service=utility_cache,
            health_check_interval=0.5,
            performance_window_size=50,
        )

        ml_monitoring = CacheMonitoringService(
            cache_service=ml_cache,
            health_check_interval=0.5,
            performance_window_size=50,
        )

        try:
            await utility_monitoring.start()
            await ml_monitoring.start()

            # Perform operations on factory caches
            for i in range(20):
                await utility_cache.set(f"util_key_{i}", {"utility": i})
                await ml_cache.set(f"ml_key_{i}", {"analysis": i})

                await utility_cache.get(f"util_key_{i}")
                await ml_cache.get(f"ml_key_{i}")

            # Allow monitoring to collect data
            await asyncio.sleep(1.0)

            # Test monitoring reports for both factory caches
            utility_health = await utility_monitoring.get_health_report()
            ml_health = await ml_monitoring.get_health_report()

            # Validate both cache types are monitored correctly
            assert utility_health["status"] == "healthy"
            assert ml_health["status"] == "healthy"

            # Validate factory-specific performance
            util_perf = utility_health["performance_metrics"]
            ml_perf = ml_health["performance_metrics"]

            assert util_perf["avg_response_time_ms"] < 2.0
            assert ml_perf["avg_response_time_ms"] < 2.0

            # Test factory performance statistics
            factory_stats = CacheFactory.get_performance_stats()
            assert factory_stats["total_instances"] >= 2
            assert factory_stats["singleton_pattern"] == "active"

            print(f"Factory monitoring - Utility: {util_perf['avg_response_time_ms']:.3f}ms, ML: {ml_perf['avg_response_time_ms']:.3f}ms")

        finally:
            await utility_monitoring.stop()
            await ml_monitoring.stop()
            await utility_cache.close()
            await ml_cache.close()
            CacheFactory.clear_instances()

    async def test_monitoring_performance_improvement_validation(self, cache_monitoring_service, cache_facade_healthy):
        """Test monitoring validation of 25x performance improvement claims."""
        # Simulate old coordination overhead baseline (for comparison)
        old_coordination_baseline_ms = 51.5  # From performance analysis

        # Generate direct cache operations
        direct_operation_times = []
        for i in range(100):
            start_time = time.perf_counter()
            await cache_facade_healthy.set(f"perf_key_{i}", {"performance": i})
            result = await cache_facade_healthy.get(f"perf_key_{i}")
            operation_time = time.perf_counter() - start_time
            direct_operation_times.append(operation_time)
            assert result is not None

        # Calculate performance improvement
        avg_direct_time_ms = sum(direct_operation_times) / len(direct_operation_times) * 1000
        performance_improvement = old_coordination_baseline_ms / avg_direct_time_ms

        # Allow monitoring to process data
        await asyncio.sleep(1.5)

        # Get monitoring performance report
        perf_report = await cache_monitoring_service.get_performance_improvement_report()

        # Validate performance improvement tracking
        assert "performance_improvement" in perf_report
        assert "direct_cache_performance" in perf_report
        assert "eliminated_overhead" in perf_report

        # Validate direct cache performance
        direct_perf = perf_report["direct_cache_performance"]
        assert direct_perf["avg_response_time_ms"] < 2.0
        assert direct_perf["architecture"] == "direct_cache_aside_pattern"

        # Validate performance improvement claims
        improvement_data = perf_report["performance_improvement"]
        assert improvement_data["improvement_factor"] >= 20.0  # Should be >20x improvement
        assert improvement_data["coordination_overhead_eliminated"] is True

        print("Performance Improvement Validation:")
        print(f"  Direct cache average: {avg_direct_time_ms:.3f}ms")
        print(f"  Performance improvement: {performance_improvement:.1f}x")
        print("  Target improvement: 25x")

        # Validate improvement target achieved
        assert performance_improvement >= 20.0, f"Performance improvement {performance_improvement:.1f}x below 20x minimum"

    async def test_monitoring_alert_thresholds_direct_cache(self, cache_monitoring_service, cache_facade_healthy):
        """Test alert threshold validation for direct cache metrics."""
        # Generate various response time scenarios
        scenarios = [
            ("fast_operations", 0.0005),  # 0.5ms - very fast
            ("normal_operations", 0.0015),  # 1.5ms - normal
            ("slow_operations", 0.0025),   # 2.5ms - approaching threshold
        ]

        for scenario_name, target_time in scenarios:
            # Simulate operations with target response times
            for i in range(20):
                start_time = time.perf_counter()
                await cache_facade_healthy.set(f"{scenario_name}_{i}", {"scenario": scenario_name})

                # Add artificial delay to simulate different response times
                if target_time > 0.001:
                    await asyncio.sleep(target_time - 0.001)

                await cache_facade_healthy.get(f"{scenario_name}_{i}")

        # Allow monitoring to process all scenarios
        await asyncio.sleep(2.0)

        # Test alert threshold analysis
        alert_report = await cache_monitoring_service.get_alert_analysis()

        # Validate alert structure
        assert "alerts" in alert_report
        assert "thresholds" in alert_report
        assert "current_metrics" in alert_report

        # Validate direct cache thresholds
        thresholds = alert_report["thresholds"]
        assert thresholds["response_time_ms"] == 2.0  # Direct cache threshold
        assert thresholds["hit_rate_min"] >= 0.8

        # Validate current metrics
        current = alert_report["current_metrics"]
        assert "avg_response_time_ms" in current
        assert "p95_response_time_ms" in current
        assert "hit_rate" in current

        # Validate boundary condition handling
        alerts = alert_report["alerts"]

        # Should have appropriate alerts based on performance
        if current["avg_response_time_ms"] > 2.0:
            assert any(alert["type"] == "response_time_exceeded" for alert in alerts)

        if current["hit_rate"] < 0.8:
            assert any(alert["type"] == "hit_rate_low" for alert in alerts)

        print(f"Alert Analysis - Response Time: {current['avg_response_time_ms']:.3f}ms, Hit Rate: {current['hit_rate']:.2%}")

    async def test_monitoring_graceful_degradation_handling(self, cache_facade_degraded):
        """Test monitoring behavior during cache degradation scenarios."""
        # Create monitoring for degraded cache (L1-only)
        degraded_monitoring = CacheMonitoringService(
            cache_service=cache_facade_degraded,
            health_check_interval=0.5,
            performance_window_size=50,
        )

        try:
            await degraded_monitoring.start()

            # Perform operations on degraded cache
            for i in range(30):
                await cache_facade_degraded.set(f"degraded_key_{i}", {"degraded": True})
                await cache_facade_degraded.get(f"degraded_key_{i}")

            # Allow monitoring to detect degradation
            await asyncio.sleep(1.5)

            # Test degradation monitoring
            health_report = await degraded_monitoring.get_health_report()

            # Validate degradation detection
            assert health_report["status"] in {"degraded", "healthy"}  # L1-only can still be healthy

            # Validate degraded cache monitoring
            cache_health = health_report["cache_health"]
            assert cache_health["l1_cache"]["status"] == "healthy"
            assert cache_health["l2_cache"]["status"] in {"disabled", "unavailable"}

            # Validate performance under degradation
            perf_metrics = health_report["performance_metrics"]
            assert perf_metrics["avg_response_time_ms"] < 1.0  # L1-only should be very fast

            # Test graceful degradation handling
            degradation_report = await degraded_monitoring.get_degradation_analysis()

            assert "degradation_level" in degradation_report
            assert "available_tiers" in degradation_report
            assert "performance_impact" in degradation_report

            available_tiers = degradation_report["available_tiers"]
            assert "l1_cache" in available_tiers
            assert available_tiers["l1_cache"] is True
            assert available_tiers.get("l2_cache", False) is False  # L2 should be unavailable

            print(f"Degradation monitoring - Level: {degradation_report['degradation_level']}, L1 only: {perf_metrics['avg_response_time_ms']:.3f}ms")

        finally:
            await degraded_monitoring.stop()

    async def test_monitoring_memory_leak_prevention(self, cache_monitoring_service, cache_facade_healthy):
        """Test memory leak prevention in monitoring with extended operations."""
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform extended monitoring operations
        for batch in range(10):
            # Generate large number of operations
            for i in range(100):
                key = f"memory_test_{batch}_{i}"
                await cache_facade_healthy.set(key, {"batch": batch, "item": i, "data": "x" * 100})
                await cache_facade_healthy.get(key)

            # Allow monitoring to process data
            await asyncio.sleep(0.1)

            # Force health checks and reports
            await cache_monitoring_service.get_health_report()
            await cache_monitoring_service.get_slo_compliance_report()

        # Check memory usage after extended operations
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        memory_increase_mb = memory_increase / (1024 * 1024)

        # Validate memory usage is reasonable
        assert memory_increase_mb < 50, f"Memory increased by {memory_increase_mb:.1f}MB, potential leak"

        # Test monitoring cleanup
        cleanup_report = await cache_monitoring_service.get_cleanup_status()

        assert "memory_usage_mb" in cleanup_report
        assert "cleanup_performed" in cleanup_report
        assert cleanup_report["cleanup_performed"] is True

        print(f"Memory leak prevention - Increase: {memory_increase_mb:.1f}MB after 1000 operations")

    async def test_monitoring_error_handling_robustness(self, cache_facade_healthy):
        """Test monitoring robustness under error conditions."""
        # Create monitoring with error-prone configuration
        error_monitoring = CacheMonitoringService(
            cache_service=cache_facade_healthy,
            health_check_interval=0.1,  # Very fast interval to stress test
            performance_window_size=10,
            enable_opentelemetry=True,  # May not be available
        )

        try:
            await error_monitoring.start()

            # Test monitoring under various error conditions
            error_scenarios = [
                "normal_operation",
                "cache_clear_during_monitoring",
                "rapid_operations",
                "large_value_operations",
            ]

            results = {}

            for scenario in error_scenarios:
                try:
                    if scenario == "normal_operation":
                        for i in range(20):
                            await cache_facade_healthy.set(f"normal_{i}", {"normal": i})
                            await cache_facade_healthy.get(f"normal_{i}")

                    elif scenario == "cache_clear_during_monitoring":
                        await cache_facade_healthy.set("clear_test", "value")
                        await cache_facade_healthy.clear()  # Clear during monitoring
                        await cache_facade_healthy.set("after_clear", "value")

                    elif scenario == "rapid_operations":
                        tasks = []
                        for i in range(50):
                            tasks.append(cache_facade_healthy.set(f"rapid_{i}", i))
                            tasks.append(cache_facade_healthy.get(f"rapid_{i}"))
                        await asyncio.gather(*tasks)

                    elif scenario == "large_value_operations":
                        large_value = "x" * 10000  # 10KB value
                        await cache_facade_healthy.set("large_value", large_value)
                        await cache_facade_healthy.get("large_value")

                    # Allow monitoring to process
                    await asyncio.sleep(0.2)

                    # Test that monitoring still works
                    health = await error_monitoring.get_health_report()
                    results[scenario] = health["status"]

                except Exception as e:
                    results[scenario] = f"error: {e!s}"

            # Validate error resilience
            successful_scenarios = sum(1 for status in results.values() if status in {"healthy", "degraded"})
            total_scenarios = len(error_scenarios)

            success_rate = successful_scenarios / total_scenarios
            assert success_rate >= 0.75, f"Error resilience too low: {success_rate:.2%} success rate"

            print(f"Error handling robustness: {success_rate:.2%} success rate across {total_scenarios} scenarios")

        finally:
            await error_monitoring.stop()

    async def test_monitoring_opentelemetry_integration(self, cache_facade_healthy):
        """Test OpenTelemetry integration with direct cache monitoring."""
        # Test monitoring with OpenTelemetry enabled
        otel_monitoring = CacheMonitoringService(
            cache_service=cache_facade_healthy,
            health_check_interval=1.0,
            enable_opentelemetry=True,
        )

        try:
            await otel_monitoring.start()

            # Perform operations to generate telemetry
            for i in range(15):
                await cache_facade_healthy.set(f"otel_key_{i}", {"telemetry": i})
                await cache_facade_healthy.get(f"otel_key_{i}")

            # Allow telemetry to be processed
            await asyncio.sleep(1.5)

            # Test telemetry integration
            telemetry_report = await otel_monitoring.get_telemetry_report()

            # Validate telemetry structure
            assert "telemetry_status" in telemetry_report
            assert "metrics_exported" in telemetry_report
            assert "traces_recorded" in telemetry_report

            # Validate telemetry data
            if telemetry_report["telemetry_status"] == "enabled":
                assert telemetry_report["metrics_exported"] > 0
                assert "cache_operations" in telemetry_report

                cache_ops = telemetry_report["cache_operations"]
                assert cache_ops["total_operations"] >= 30  # 15 sets + 15 gets
                assert cache_ops["avg_duration_ms"] < 2.0
            else:
                # OpenTelemetry not available - should degrade gracefully
                assert telemetry_report["telemetry_status"] == "unavailable"
                assert "fallback_metrics" in telemetry_report

            print(f"OpenTelemetry integration: {telemetry_report['telemetry_status']}")

        finally:
            await otel_monitoring.stop()


@pytest.mark.integration
@pytest.mark.real_behavior
class TestCacheMonitoringServicePerformance:
    """Performance-focused tests for cache monitoring with direct architecture."""

    async def test_monitoring_overhead_validation(self):
        """Test that monitoring overhead stays <1ms as required."""
        # Create cache facade for monitoring
        cache = CacheFacade(l1_max_size=100, enable_l2=False)

        # Create monitoring service
        monitoring = CacheMonitoringService(
            cache_service=cache,
            health_check_interval=0.1,  # Frequent checks for overhead testing
            performance_window_size=200,
        )

        try:
            await monitoring.start()

            # Measure monitoring overhead
            monitoring_times = []

            for i in range(100):
                # Perform cache operation
                await cache.set(f"overhead_key_{i}", {"overhead_test": i})

                # Measure monitoring overhead
                start_time = time.perf_counter()
                health = await monitoring.get_health_report()
                monitoring_time = time.perf_counter() - start_time
                monitoring_times.append(monitoring_time)

                # Validate each monitoring call
                assert monitoring_time < 0.001, f"Monitoring overhead {monitoring_time * 1000:.3f}ms exceeds 1ms"

            # Performance analysis
            avg_monitoring_time = sum(monitoring_times) / len(monitoring_times)
            max_monitoring_time = max(monitoring_times)

            print("Monitoring Overhead Analysis:")
            print(f"  Average monitoring time: {avg_monitoring_time * 1000:.3f}ms")
            print(f"  Max monitoring time: {max_monitoring_time * 1000:.3f}ms")
            print("  Target: <1ms")

            # Validate overhead requirements
            assert avg_monitoring_time < 0.001, f"Average monitoring overhead {avg_monitoring_time * 1000:.3f}ms exceeds 1ms"
            assert max_monitoring_time < 0.002, f"Max monitoring overhead {max_monitoring_time * 1000:.3f}ms too high"

        finally:
            await monitoring.stop()
            await cache.close()

    async def test_high_frequency_monitoring_stability(self):
        """Test monitoring stability under high-frequency operations."""
        cache = CacheFacade(l1_max_size=500, enable_l2=False)

        monitoring = CacheMonitoringService(
            cache_service=cache,
            health_check_interval=0.05,  # Very high frequency
            performance_window_size=500,
        )

        try:
            await monitoring.start()

            # High-frequency operations
            operation_count = 1000
            start_time = time.perf_counter()

            for i in range(operation_count):
                await cache.set(f"freq_key_{i}", i)
                result = await cache.get(f"freq_key_{i}")
                assert result == i

            total_time = time.perf_counter() - start_time
            throughput = operation_count / total_time

            # Test monitoring under high frequency
            health_checks = []
            for _ in range(10):
                check_start = time.perf_counter()
                health = await monitoring.get_health_report()
                check_time = time.perf_counter() - check_start
                health_checks.append(check_time)

                assert health["status"] in {"healthy", "degraded"}
                await asyncio.sleep(0.1)

            # Performance analysis
            avg_check_time = sum(health_checks) / len(health_checks)

            print("High-Frequency Monitoring:")
            print(f"  Cache throughput: {throughput:.0f} ops/sec")
            print(f"  Monitoring check time: {avg_check_time * 1000:.3f}ms")
            print(f"  Health status: {health['status']}")

            # Validate stability under high frequency
            assert throughput > 5000, f"Cache throughput {throughput:.0f} ops/sec too low"
            assert avg_check_time < 0.001, f"Monitoring check time {avg_check_time * 1000:.3f}ms too high"

        finally:
            await monitoring.stop()
            await cache.close()


if __name__ == "__main__":
    """Run comprehensive monitoring tests for direct cache architecture."""
    pytest.main([__file__, "-v", "--tb=short"])
