"""Performance tests for CacheMonitoringService SLO compliance.

Validates that CacheMonitoringService meets its performance SLO requirements
under various load conditions and ensures monitoring overhead remains minimal.
Tests critical performance paths and validates against defined SLO targets.

SLO Requirements Tested:
- Health check operations: <100ms
- Monitoring metrics collection: <10ms
- Alert metrics calculation: <20ms
- SLO compliance calculation: <50ms
- Monitoring overhead per operation: <1ms
- High-frequency operation handling: >1000 ops/sec
- Memory efficiency under load
- Concurrent operation stability
"""

import asyncio
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock

import pytest

from prompt_improver.services.cache.cache_monitoring_service import (
    CacheMonitoringService,
)

logger = logging.getLogger(__name__)


@pytest.mark.performance
class TestCacheMonitoringServicePerformance:
    """Performance tests for CacheMonitoringService SLO compliance."""

    @pytest.fixture
    def high_performance_coordinator(self):
        """High-performance mock coordinator for performance testing."""
        coordinator = MagicMock()

        # Optimized performance stats response
        coordinator.get_performance_stats.return_value = {
            "overall_hit_rate": 0.96,
            "avg_response_time_ms": 15.0,
            "total_requests": 50000,
            "l1_hit_rate": 0.98,
            "l2_hit_rate": 0.85,
            "l1_hits": 49000,
            "l2_hits": 42500,
            "health_status": "healthy",
            "warming_enabled": True,
            "tracked_patterns": 150,
            "l1_cache_stats": {
                "size": 1000,
                "estimated_memory_bytes": 10485760,  # 10MB
                "hit_rate": 0.98,
                "health_status": "healthy",
                "utilization": 0.85
            },
            "l2_cache_stats": {
                "success_rate": 0.99,
                "health_status": "healthy",
                "currently_connected": True
            }
        }

        # Fast mock operations for health check
        coordinator.set = MagicMock()
        coordinator.get = MagicMock(return_value={"test": True, "timestamp": time.time()})
        coordinator.delete = MagicMock()

        return coordinator

    @pytest.fixture
    def performance_monitoring_service(self, high_performance_coordinator):
        """High-performance CacheMonitoringService for performance testing."""
        return CacheMonitoringService(high_performance_coordinator)

    async def test_health_check_slo_compliance(self, performance_monitoring_service):
        """Test health check SLO compliance (<100ms)."""
        monitoring = performance_monitoring_service

        # Warm up
        await monitoring.health_check()

        # Performance measurement
        health_check_times = []
        iterations = 50

        for i in range(iterations):
            start_time = time.perf_counter()
            health_result = await monitoring.health_check()
            health_check_time = time.perf_counter() - start_time
            health_check_times.append(health_check_time)

            # Validate successful completion
            assert health_result["healthy"] is not None, f"Health check {i} failed to complete"

            # SLO validation per operation
            assert health_check_time < 0.1, f"Health check {i} SLO violation: {health_check_time * 1000:.2f}ms > 100ms"

        # Statistical analysis
        avg_time = statistics.mean(health_check_times)
        p95_time = statistics.quantiles(health_check_times, n=20)[18]  # 95th percentile
        p99_time = statistics.quantiles(health_check_times, n=100)[98]  # 99th percentile
        min_time = min(health_check_times)
        max_time = max(health_check_times)

        print("\nHealth Check Performance SLO Analysis:")
        print(f"  Iterations: {iterations}")
        print(f"  Average: {avg_time * 1000:.2f}ms")
        print(f"  P95: {p95_time * 1000:.2f}ms")
        print(f"  P99: {p99_time * 1000:.2f}ms")
        print(f"  Min: {min_time * 1000:.2f}ms")
        print(f"  Max: {max_time * 1000:.2f}ms")
        print("  SLO Target: <100ms")

        # SLO compliance validation
        assert avg_time < 0.05, f"Average health check time {avg_time * 1000:.2f}ms exceeds 50ms target"
        assert p95_time < 0.08, f"P95 health check time {p95_time * 1000:.2f}ms exceeds 80ms target"
        assert p99_time < 0.1, f"P99 health check time {p99_time * 1000:.2f}ms exceeds 100ms SLO"

    def test_monitoring_metrics_slo_compliance(self, performance_monitoring_service):
        """Test monitoring metrics collection SLO compliance (<10ms)."""
        monitoring = performance_monitoring_service

        # Warm up
        monitoring.get_monitoring_metrics()

        # Performance measurement
        metrics_times = []
        iterations = 1000

        for i in range(iterations):
            start_time = time.perf_counter()
            metrics = monitoring.get_monitoring_metrics()
            metrics_time = time.perf_counter() - start_time
            metrics_times.append(metrics_time)

            # Validate successful completion
            assert len(metrics) > 0, f"Metrics collection {i} returned empty result"

            # SLO validation per operation
            assert metrics_time < 0.01, f"Metrics collection {i} SLO violation: {metrics_time * 1000:.2f}ms > 10ms"

        # Statistical analysis
        avg_time = statistics.mean(metrics_times)
        p95_time = statistics.quantiles(metrics_times, n=20)[18]
        p99_time = statistics.quantiles(metrics_times, n=100)[98]

        print("\nMonitoring Metrics Performance SLO Analysis:")
        print(f"  Iterations: {iterations}")
        print(f"  Average: {avg_time * 1000:.3f}ms")
        print(f"  P95: {p95_time * 1000:.3f}ms")
        print(f"  P99: {p99_time * 1000:.3f}ms")
        print("  SLO Target: <10ms")

        # SLO compliance validation
        assert avg_time < 0.005, f"Average metrics time {avg_time * 1000:.3f}ms exceeds 5ms target"
        assert p95_time < 0.008, f"P95 metrics time {p95_time * 1000:.3f}ms exceeds 8ms target"
        assert p99_time < 0.01, f"P99 metrics time {p99_time * 1000:.3f}ms exceeds 10ms SLO"

    def test_alert_metrics_slo_compliance(self, performance_monitoring_service):
        """Test alert metrics calculation SLO compliance (<20ms)."""
        monitoring = performance_monitoring_service

        # Setup test data for alert calculation
        monitoring._response_times = [0.100] * 950 + [0.300] * 50  # 95% compliance
        monitoring._error_counts = {"test_error": 25}

        # Warm up
        monitoring.get_alert_metrics()

        # Performance measurement
        alert_times = []
        iterations = 500

        for i in range(iterations):
            start_time = time.perf_counter()
            alerts = monitoring.get_alert_metrics()
            alert_time = time.perf_counter() - start_time
            alert_times.append(alert_time)

            # Validate successful completion
            assert "alerts" in alerts, f"Alert metrics {i} missing alerts section"
            assert len(alerts["alerts"]) > 0, f"Alert metrics {i} returned empty alerts"

            # SLO validation per operation
            assert alert_time < 0.02, f"Alert metrics {i} SLO violation: {alert_time * 1000:.2f}ms > 20ms"

        # Statistical analysis
        avg_time = statistics.mean(alert_times)
        p95_time = statistics.quantiles(alert_times, n=20)[18]
        p99_time = statistics.quantiles(alert_times, n=100)[98]

        print("\nAlert Metrics Performance SLO Analysis:")
        print(f"  Iterations: {iterations}")
        print(f"  Average: {avg_time * 1000:.3f}ms")
        print(f"  P95: {p95_time * 1000:.3f}ms")
        print(f"  P99: {p99_time * 1000:.3f}ms")
        print("  SLO Target: <20ms")

        # SLO compliance validation
        assert avg_time < 0.01, f"Average alert metrics time {avg_time * 1000:.3f}ms exceeds 10ms target"
        assert p95_time < 0.015, f"P95 alert metrics time {p95_time * 1000:.3f}ms exceeds 15ms target"
        assert p99_time < 0.02, f"P99 alert metrics time {p99_time * 1000:.3f}ms exceeds 20ms SLO"

    def test_slo_compliance_calculation_slo(self, performance_monitoring_service):
        """Test SLO compliance calculation SLO compliance (<50ms)."""
        monitoring = performance_monitoring_service

        # Setup large dataset for calculation stress test
        large_response_times = []
        for i in range(10000):
            # Mix of compliant and non-compliant times
            if i % 20 == 0:  # 5% violations
                large_response_times.append(0.250)  # 250ms violation
            else:
                large_response_times.append(0.050 + (i % 100) * 0.001)  # 50-150ms range

        monitoring._response_times = large_response_times

        # Performance measurement
        slo_calc_times = []
        iterations = 100

        for i in range(iterations):
            start_time = time.perf_counter()
            slo_result = monitoring.calculate_slo_compliance()
            slo_calc_time = time.perf_counter() - start_time
            slo_calc_times.append(slo_calc_time)

            # Validate successful completion
            assert slo_result["sample_count"] > 0, f"SLO calculation {i} returned no samples"
            assert "percentiles" in slo_result, f"SLO calculation {i} missing percentiles"

            # SLO validation per operation
            assert slo_calc_time < 0.05, f"SLO calculation {i} SLO violation: {slo_calc_time * 1000:.2f}ms > 50ms"

        # Statistical analysis
        avg_time = statistics.mean(slo_calc_times)
        p95_time = statistics.quantiles(slo_calc_times, n=20)[18]
        p99_time = statistics.quantiles(slo_calc_times, n=100)[98]

        print("\nSLO Calculation Performance SLO Analysis:")
        print(f"  Dataset size: {len(large_response_times):,} samples")
        print(f"  Iterations: {iterations}")
        print(f"  Average: {avg_time * 1000:.3f}ms")
        print(f"  P95: {p95_time * 1000:.3f}ms")
        print(f"  P99: {p99_time * 1000:.3f}ms")
        print("  SLO Target: <50ms")

        # SLO compliance validation
        assert avg_time < 0.025, f"Average SLO calculation time {avg_time * 1000:.3f}ms exceeds 25ms target"
        assert p95_time < 0.04, f"P95 SLO calculation time {p95_time * 1000:.3f}ms exceeds 40ms target"
        assert p99_time < 0.05, f"P99 SLO calculation time {p99_time * 1000:.3f}ms exceeds 50ms SLO"

    def test_monitoring_overhead_slo_compliance(self, performance_monitoring_service):
        """Test monitoring overhead SLO compliance (<1ms per operation)."""
        monitoring = performance_monitoring_service

        # Performance measurement for record_operation
        record_times = []
        iterations = 10000

        for i in range(iterations):
            operation_name = f"perf_test_{i % 100}"  # 100 different operation names
            response_time = 0.050 + (i % 50) * 0.001  # 50-100ms range
            success = i % 10 != 0  # 90% success rate

            start_time = time.perf_counter()
            monitoring.record_operation(operation_name, response_time, success=success)
            record_time = time.perf_counter() - start_time
            record_times.append(record_time)

            # SLO validation per operation
            assert record_time < 0.001, f"Record operation {i} SLO violation: {record_time * 1000:.3f}ms > 1ms"

        # Statistical analysis
        avg_time = statistics.mean(record_times)
        p95_time = statistics.quantiles(record_times, n=20)[18]
        p99_time = statistics.quantiles(record_times, n=100)[98]
        max_time = max(record_times)

        print("\nMonitoring Overhead Performance SLO Analysis:")
        print(f"  Operations recorded: {iterations:,}")
        print(f"  Average: {avg_time * 1000:.4f}ms")
        print(f"  P95: {p95_time * 1000:.4f}ms")
        print(f"  P99: {p99_time * 1000:.4f}ms")
        print(f"  Max: {max_time * 1000:.4f}ms")
        print("  SLO Target: <1ms")
        print(f"  Throughput: {iterations / sum(record_times):.0f} ops/sec")

        # SLO compliance validation
        assert avg_time < 0.0005, f"Average record operation time {avg_time * 1000:.4f}ms exceeds 0.5ms target"
        assert p95_time < 0.0008, f"P95 record operation time {p95_time * 1000:.4f}ms exceeds 0.8ms target"
        assert p99_time < 0.001, f"P99 record operation time {p99_time * 1000:.4f}ms exceeds 1ms SLO"

    def test_high_frequency_operation_handling(self, performance_monitoring_service):
        """Test high-frequency operation handling (>1000 ops/sec)."""
        monitoring = performance_monitoring_service

        # High-frequency operation simulation
        operations_count = 50000
        batch_size = 1000

        total_start = time.perf_counter()

        for batch_start in range(0, operations_count, batch_size):
            batch_end = min(batch_start + batch_size, operations_count)
            batch_start_time = time.perf_counter()

            for i in range(batch_start, batch_end):
                operation_name = f"high_freq_{i % 20}"
                response_time = 0.001 + (i % 100) * 0.0001  # 1-11ms range
                success = i % 20 != 0  # 95% success rate

                monitoring.record_operation(operation_name, response_time, success=success)

            batch_time = time.perf_counter() - batch_start_time
            batch_throughput = batch_size / batch_time

            # Validate batch throughput
            assert batch_throughput > 1000, f"Batch throughput {batch_throughput:.0f} ops/sec below 1000 ops/sec target"

        total_time = time.perf_counter() - total_start
        overall_throughput = operations_count / total_time

        print("\nHigh-Frequency Operation Handling Analysis:")
        print(f"  Total operations: {operations_count:,}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Overall throughput: {overall_throughput:.0f} ops/sec")
        print("  Target throughput: >1000 ops/sec")
        print(f"  Response times tracked: {len(monitoring._response_times):,}")
        print(f"  Error types tracked: {len(monitoring._error_counts):,}")

        # Throughput validation
        assert overall_throughput > 5000, f"Overall throughput {overall_throughput:.0f} ops/sec below 5000 ops/sec target"

        # Memory management validation
        assert len(monitoring._response_times) <= monitoring._max_response_time_samples, "Response times not properly managed"
        assert len(monitoring._error_counts) < 100, "Error counts growing without limit"

    def test_concurrent_monitoring_performance(self, performance_monitoring_service):
        """Test concurrent monitoring operation performance."""
        monitoring = performance_monitoring_service

        def concurrent_worker(worker_id: int, operations_per_worker: int) -> dict[str, Any]:
            """Worker function for concurrent monitoring operations."""
            start_time = time.perf_counter()
            operation_times = []

            for i in range(operations_per_worker):
                # Mix of different monitoring operations
                if i % 50 == 0:
                    # Health check (async operation - run in sync context)
                    op_start = time.perf_counter()
                    try:
                        # Create and run event loop for this operation
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        health_result = loop.run_until_complete(monitoring.health_check())
                        loop.close()
                        op_time = time.perf_counter() - op_start
                        operation_times.append(("health_check", op_time))
                        assert health_result["healthy"] is not None
                    except Exception as e:
                        logger.warning(f"Worker {worker_id} health check failed: {e}")
                elif i % 20 == 0:
                    # Alert metrics
                    op_start = time.perf_counter()
                    alerts = monitoring.get_alert_metrics()
                    op_time = time.perf_counter() - op_start
                    operation_times.append(("alert_metrics", op_time))
                    assert len(alerts["alerts"]) > 0
                elif i % 10 == 0:
                    # Monitoring metrics
                    op_start = time.perf_counter()
                    metrics = monitoring.get_monitoring_metrics()
                    op_time = time.perf_counter() - op_start
                    operation_times.append(("monitoring_metrics", op_time))
                    assert len(metrics) > 0
                else:
                    # Record operation
                    op_start = time.perf_counter()
                    monitoring.record_operation(f"worker_{worker_id}_op_{i}", 0.050, success=True)
                    op_time = time.perf_counter() - op_start
                    operation_times.append(("record_operation", op_time))

            duration = time.perf_counter() - start_time
            return {
                "worker_id": worker_id,
                "duration": duration,
                "operations": len(operation_times),
                "operation_times": operation_times,
                "throughput": len(operation_times) / duration
            }

        # Run concurrent workers
        workers = 10
        operations_per_worker = 200

        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(concurrent_worker, i, operations_per_worker) for i in range(workers)]
            results = [future.result() for future in futures]

        total_time = time.perf_counter() - start_time

        # Analyze concurrent performance
        total_operations = sum(r["operations"] for r in results)
        overall_throughput = total_operations / total_time
        worker_throughputs = [r["throughput"] for r in results]

        # Analyze operation type performance
        operation_type_times = {}
        for result in results:
            for op_type, op_time in result["operation_times"]:
                if op_type not in operation_type_times:
                    operation_type_times[op_type] = []
                operation_type_times[op_type].append(op_time)

        print("\nConcurrent Monitoring Performance Analysis:")
        print(f"  Workers: {workers}")
        print(f"  Operations per worker: {operations_per_worker}")
        print(f"  Total operations: {total_operations:,}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Overall throughput: {overall_throughput:.0f} ops/sec")
        print(f"  Average worker throughput: {statistics.mean(worker_throughputs):.0f} ops/sec")

        for op_type, times in operation_type_times.items():
            avg_time = statistics.mean(times)
            p95_time = statistics.quantiles(times, n=20)[18] if len(times) > 20 else max(times)
            print(f"  {op_type}: avg={avg_time * 1000:.2f}ms, p95={p95_time * 1000:.2f}ms, count={len(times)}")

        # Performance validation
        assert overall_throughput > 1000, f"Concurrent throughput {overall_throughput:.0f} ops/sec below 1000 ops/sec target"
        assert min(worker_throughputs) > 50, f"Slowest worker {min(worker_throughputs):.0f} ops/sec too slow"

        # SLO validation for each operation type
        for op_type, times in operation_type_times.items():
            avg_time = statistics.mean(times)
            if op_type == "health_check":
                assert avg_time < 0.1, f"Concurrent {op_type} average {avg_time * 1000:.2f}ms exceeds 100ms SLO"
            elif op_type == "monitoring_metrics":
                assert avg_time < 0.01, f"Concurrent {op_type} average {avg_time * 1000:.2f}ms exceeds 10ms SLO"
            elif op_type == "alert_metrics":
                assert avg_time < 0.02, f"Concurrent {op_type} average {avg_time * 1000:.2f}ms exceeds 20ms SLO"
            elif op_type == "record_operation":
                assert avg_time < 0.001, f"Concurrent {op_type} average {avg_time * 1000:.3f}ms exceeds 1ms SLO"

    def test_memory_efficiency_under_load(self, performance_monitoring_service):
        """Test memory efficiency under high load conditions."""
        import os

        import psutil

        monitoring = performance_monitoring_service
        process = psutil.Process(os.getpid())

        # Get initial memory usage
        initial_memory = process.memory_info().rss

        # High-load scenario
        high_load_operations = 100000

        load_start = time.perf_counter()

        for i in range(high_load_operations):
            # Vary operation patterns to stress test memory management
            operation_name = f"load_test_{i % 1000}"  # 1000 different operation names
            response_time = 0.001 + (i % 1000) * 0.0001  # 1-101ms range
            success = i % 50 != 0  # 98% success rate

            monitoring.record_operation(operation_name, response_time, success=success)

            # Periodic monitoring operations
            if i % 1000 == 0:
                monitoring.get_monitoring_metrics()
            if i % 5000 == 0:
                monitoring.get_alert_metrics()
            if i % 10000 == 0:
                monitoring.calculate_slo_compliance()

        load_time = time.perf_counter() - load_start
        load_throughput = high_load_operations / load_time

        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        memory_per_operation = memory_increase / high_load_operations

        print("\nMemory Efficiency Under Load Analysis:")
        print(f"  High-load operations: {high_load_operations:,}")
        print(f"  Load time: {load_time:.2f}s")
        print(f"  Load throughput: {load_throughput:.0f} ops/sec")
        print(f"  Initial memory: {initial_memory / (1024 * 1024):.1f} MB")
        print(f"  Final memory: {final_memory / (1024 * 1024):.1f} MB")
        print(f"  Memory increase: {memory_increase / (1024 * 1024):.1f} MB")
        print(f"  Memory per operation: {memory_per_operation:.1f} bytes")
        print(f"  Response times tracked: {len(monitoring._response_times):,}")
        print(f"  Error types tracked: {len(monitoring._error_counts):,}")

        # Memory efficiency validation
        assert memory_per_operation < 50, f"Memory per operation {memory_per_operation:.1f} bytes too high"
        assert memory_increase < 50 * 1024 * 1024, f"Total memory increase {memory_increase / (1024 * 1024):.1f} MB too high"
        assert load_throughput > 5000, f"Load throughput {load_throughput:.0f} ops/sec too low"

        # Memory management validation
        assert len(monitoring._response_times) <= monitoring._max_response_time_samples, "Response times not properly limited"
        assert len(monitoring._error_counts) < 2000, "Error counts not properly managed"

    def test_performance_regression_detection(self, performance_monitoring_service):
        """Test performance regression detection capabilities."""
        monitoring = performance_monitoring_service

        # Establish baseline performance
        baseline_iterations = 1000
        baseline_times = {
            "health_check": [],
            "monitoring_metrics": [],
            "alert_metrics": [],
            "record_operation": []
        }

        # Collect baseline metrics
        for i in range(baseline_iterations):
            # Health check baseline
            if i % 10 == 0:
                start_time = time.perf_counter()
                asyncio.run(monitoring.health_check())
                baseline_times["health_check"].append(time.perf_counter() - start_time)

            # Monitoring metrics baseline
            start_time = time.perf_counter()
            monitoring.get_monitoring_metrics()
            baseline_times["monitoring_metrics"].append(time.perf_counter() - start_time)

            # Alert metrics baseline
            if i % 5 == 0:
                start_time = time.perf_counter()
                monitoring.get_alert_metrics()
                baseline_times["alert_metrics"].append(time.perf_counter() - start_time)

            # Record operation baseline
            start_time = time.perf_counter()
            monitoring.record_operation(f"baseline_{i}", 0.050, success=True)
            baseline_times["record_operation"].append(time.perf_counter() - start_time)

        # Calculate baseline statistics
        baseline_stats = {}
        for operation, times in baseline_times.items():
            baseline_stats[operation] = {
                "mean": statistics.mean(times),
                "p95": statistics.quantiles(times, n=20)[18] if len(times) > 20 else max(times),
                "p99": statistics.quantiles(times, n=100)[98] if len(times) > 100 else max(times)
            }

        print("\nPerformance Regression Detection Analysis:")
        print(f"  Baseline iterations: {baseline_iterations}")

        for operation, stats in baseline_stats.items():
            print(f"  {operation}:")
            print(f"    Mean: {stats['mean'] * 1000:.3f}ms")
            print(f"    P95: {stats['p95'] * 1000:.3f}ms")
            print(f"    P99: {stats['p99'] * 1000:.3f}ms")

        # Validate against SLO targets
        slo_targets = {
            "health_check": 0.1,      # 100ms
            "monitoring_metrics": 0.01,  # 10ms
            "alert_metrics": 0.02,     # 20ms
            "record_operation": 0.001  # 1ms
        }

        for operation, target in slo_targets.items():
            stats = baseline_stats[operation]
            assert stats["mean"] < target * 0.5, f"{operation} mean performance regression: {stats['mean'] * 1000:.3f}ms (target: <{target * 500:.1f}ms)"
            assert stats["p95"] < target * 0.8, f"{operation} P95 performance regression: {stats['p95'] * 1000:.3f}ms (target: <{target * 800:.1f}ms)"
            assert stats["p99"] < target, f"{operation} P99 performance regression: {stats['p99'] * 1000:.3f}ms (target: <{target * 1000:.1f}ms)"
