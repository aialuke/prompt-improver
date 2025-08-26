"""
Performance testing harness for ML testing.

This module contains performance testing utilities extracted from conftest.py
to maintain clean architecture and separation of concerns.
"""
import asyncio
import time
from collections.abc import Callable
from typing import Any

import psutil
from prompt_improver.utils.datetime_utils import aware_utc_now


class PerformanceTestHarness:
    """Performance testing harness for ML pipeline benchmarking.
    
    Provides comprehensive performance monitoring, profiling utilities,
    and benchmark comparison tools following 2025 best practices.
    """
    
    def __init__(self):
        self.benchmark_results = []
        self.performance_baselines = {}
        self.resource_monitors = []
        self.start_time = None

    async def benchmark_async_function(
        self,
        func: Callable[..., Any],
        *args: Any,
        iterations: int = 10,
        warmup_iterations: int = 2,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Benchmark async function with statistical analysis."""
        for _ in range(warmup_iterations):
            if asyncio.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)
        execution_times = []
        memory_usage = []
        for i in range(iterations):
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            start_time = time.perf_counter()
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000
            mem_after = process.memory_info().rss / 1024 / 1024
            memory_delta = mem_after - mem_before
            execution_times.append(execution_time)
            memory_usage.append(memory_delta)
        import statistics

        benchmark_result = {
            "function_name": func.__name__,
            "iterations": iterations,
            "execution_times_ms": execution_times,
            "avg_time_ms": statistics.mean(execution_times),
            "median_time_ms": statistics.median(execution_times),
            "min_time_ms": min(execution_times),
            "max_time_ms": max(execution_times),
            "std_dev_ms": statistics.stdev(execution_times)
            if len(execution_times) > 1
            else 0,
            "memory_usage_mb": memory_usage,
            "avg_memory_mb": statistics.mean(memory_usage),
            "max_memory_mb": max(memory_usage),
            "benchmarked_at": aware_utc_now().isoformat(),
        }
        self.benchmark_results.append(benchmark_result)
        return benchmark_result

    def set_performance_baseline(
        self, operation_name: str, baseline_metrics: dict[str, float]
    ) -> None:
        """Set performance baseline for comparison."""
        self.performance_baselines[operation_name] = baseline_metrics

    def compare_to_baseline(
        self, operation_name: str, actual_metrics: dict[str, float]
    ) -> dict[str, Any]:
        """Compare actual metrics to baseline."""
        if operation_name not in self.performance_baselines:
            return {"status": "no_baseline", "operation": operation_name}
        baseline = self.performance_baselines[operation_name]
        comparison = {"operation": operation_name, "comparisons": {}}
        for metric_name, actual_value in actual_metrics.items():
            if metric_name in baseline:
                baseline_value = baseline[metric_name]
                percentage_diff = (
                    (actual_value - baseline_value) / baseline_value * 100
                )
                comparison["comparisons"][metric_name] = {
                    "baseline": baseline_value,
                    "actual": actual_value,
                    "difference": actual_value - baseline_value,
                    "percentage_diff": percentage_diff,
                    "performance_status": "improved"
                    if percentage_diff < -5
                    else "degraded"
                    if percentage_diff > 5
                    else "stable",
                }
        return comparison

    async def monitor_resource_usage(
        self, duration_seconds: int = 60
    ) -> dict[str, list[float]]:
        """Monitor system resource usage over time."""
        cpu_usage = []
        memory_usage = []
        timestamps = []
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            process = psutil.Process()
            cpu_usage.append(process.cpu_percent())
            memory_usage.append(process.memory_info().rss / 1024 / 1024)
            timestamps.append(time.time() - start_time)
            await asyncio.sleep(0.1)
        return {
            "timestamps": timestamps,
            "cpu_usage_percent": cpu_usage,
            "memory_usage_mb": memory_usage,
            "avg_cpu": sum(cpu_usage) / len(cpu_usage),
            "avg_memory": sum(memory_usage) / len(memory_usage),
            "peak_memory": max(memory_usage),
        }

    def validate_sla_compliance(
        self, metrics: dict[str, float], sla_thresholds: dict[str, float]
    ) -> dict[str, Any]:
        """Validate SLA compliance for performance metrics."""
        compliance_results = {"overall_compliant": True, "violations": []}
        for metric_name, threshold in sla_thresholds.items():
            if metric_name in metrics:
                actual_value = metrics[metric_name]
                is_compliant = actual_value <= threshold
                if not is_compliant:
                    compliance_results["overall_compliant"] = False
                    compliance_results["violations"].append({
                        "metric": metric_name,
                        "threshold": threshold,
                        "actual": actual_value,
                        "violation_percentage": (actual_value - threshold)
                        / threshold
                        * 100,
                    })
        return compliance_results

    def get_benchmark_summary(self) -> dict[str, Any]:
        """Get comprehensive benchmark summary."""
        if not self.benchmark_results:
            return {"status": "no_benchmarks_run"}
        return {
            "total_benchmarks": len(self.benchmark_results),
            "functions_tested": list(
                {r["function_name"] for r in self.benchmark_results}
            ),
            "avg_execution_time_ms": sum(
                r["avg_time_ms"] for r in self.benchmark_results
            )
            / len(self.benchmark_results),
            "slowest_function": max(
                self.benchmark_results, key=lambda r: r["avg_time_ms"]
            )["function_name"],
            "fastest_function": min(
                self.benchmark_results, key=lambda r: r["avg_time_ms"]
            )["function_name"],
            "results": self.benchmark_results,
        }