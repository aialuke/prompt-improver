"""Event loop benchmark utilities for latency measurement and performance comparison.

This module provides comprehensive benchmarking capabilities for comparing
asyncio vs uvloop performance, with a focus on meeting the <200ms latency target.
"""

import asyncio
import logging
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

from prompt_improver.utils.event_loop_manager import get_event_loop_manager
from prompt_improver.utils.session_event_loop import get_session_wrapper

logger = logging.getLogger(__name__)

class EventLoopBenchmark:
    """Comprehensive event loop benchmarking suite."""

    def __init__(self):
        self.event_loop_manager = get_event_loop_manager()
        self.target_latency_ms = 200.0  # Target latency in milliseconds
        self.baseline_results: dict[str, Any] | None = None
        self.uvloop_results: dict[str, Any] | None = None

    async def run_latency_benchmark(
        self, samples: int = 100, operation_type: str = "sleep_yield"
    ) -> dict[str, Any]:
        """Run latency benchmark for current event loop.

        Args:
            samples: Number of samples to collect
            operation_type: Type of operation to benchmark

        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Starting latency benchmark with {samples} samples")

        if operation_type == "sleep_yield":
            results = await self._benchmark_sleep_yield(samples)
        elif operation_type == "task_creation":
            results = await self._benchmark_task_creation(samples)
        elif operation_type == "gather_operations":
            results = await self._benchmark_gather_operations(samples)
        elif operation_type == "context_switch":
            results = await self._benchmark_context_switch(samples)
        else:
            raise ValueError(f"Unknown operation type: {operation_type}")

        # Add loop information
        loop_info = self.event_loop_manager.get_loop_info()
        results["loop_info"] = loop_info
        results["operation_type"] = operation_type
        results["target_latency_ms"] = self.target_latency_ms
        results["meets_target"] = results["avg_ms"] < self.target_latency_ms

        return results

    async def _benchmark_sleep_yield(self, samples: int) -> dict[str, Any]:
        """Benchmark sleep/yield operations."""
        latencies = []

        for _ in range(samples):
            start = time.perf_counter()
            await asyncio.sleep(0)  # Yield control
            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

        return self._calculate_statistics(latencies)

    async def _benchmark_task_creation(self, samples: int) -> dict[str, Any]:
        """Benchmark task creation and completion."""
        latencies = []

        async def dummy_task():
            await asyncio.sleep(0.001)
            return True

        for _ in range(samples):
            start = time.perf_counter()
            task = asyncio.create_task(dummy_task())
            await task
            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

        return self._calculate_statistics(latencies)

    async def _benchmark_gather_operations(self, samples: int) -> dict[str, Any]:
        """Benchmark gather operations with multiple tasks."""
        latencies = []

        async def dummy_task():
            await asyncio.sleep(0.001)
            return True

        for _ in range(samples):
            start = time.perf_counter()

            # Create multiple tasks and gather
            tasks = [asyncio.create_task(dummy_task()) for _ in range(5)]
            await asyncio.gather(*tasks)

            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

        return self._calculate_statistics(latencies)

    async def _benchmark_context_switch(self, samples: int) -> dict[str, Any]:
        """Benchmark context switching between coroutines."""
        latencies = []

        async def ping():
            await asyncio.sleep(0)
            return "ping"

        async def pong():
            await asyncio.sleep(0)
            return "pong"

        for _ in range(samples):
            start = time.perf_counter()

            # Context switch between coroutines
            result1 = await ping()
            result2 = await pong()

            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

        return self._calculate_statistics(latencies)

    def _calculate_statistics(self, latencies: list[float]) -> dict[str, Any]:
        """Calculate comprehensive statistics for latency measurements."""
        if not latencies:
            return {
                "avg_ms": 0,
                "min_ms": 0,
                "max_ms": 0,
                "median_ms": 0,
                "std_ms": 0,
                "p95_ms": 0,
                "p99_ms": 0,
                "samples": 0,
            }

        avg_ms = statistics.mean(latencies)
        min_ms = min(latencies)
        max_ms = max(latencies)
        median_ms = statistics.median(latencies)
        std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0

        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        p95_ms = sorted_latencies[int(0.95 * len(sorted_latencies))]
        p99_ms = sorted_latencies[int(0.99 * len(sorted_latencies))]

        return {
            "avg_ms": avg_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "median_ms": median_ms,
            "std_ms": std_ms,
            "p95_ms": p95_ms,
            "p99_ms": p99_ms,
            "samples": len(latencies),
            "total_time_ms": sum(latencies),
        }

    async def run_comprehensive_benchmark(self) -> dict[str, Any]:
        """Run comprehensive benchmark suite."""
        logger.info("Starting comprehensive event loop benchmark")

        benchmark_results = {}

        # Test different operation types
        operation_types = [
            "sleep_yield",
            "task_creation",
            "gather_operations",
            "context_switch",
        ]

        for operation_type in operation_types:
            logger.info(f"Benchmarking {operation_type} operations")
            results = await self.run_latency_benchmark(
                samples=100, operation_type=operation_type
            )
            benchmark_results[operation_type] = results

        # Calculate overall performance metrics
        overall_avg = statistics.mean([
            results["avg_ms"] for results in benchmark_results.values()
        ])

        overall_p95 = statistics.mean([
            results["p95_ms"] for results in benchmark_results.values()
        ])

        meets_target = overall_avg < self.target_latency_ms

        return {
            "operation_benchmarks": benchmark_results,
            "overall_avg_ms": overall_avg,
            "overall_p95_ms": overall_p95,
            "target_latency_ms": self.target_latency_ms,
            "meets_target": meets_target,
            "timestamp": time.time(),
        }

    async def compare_asyncio_vs_uvloop(self) -> dict[str, Any]:
        """Compare asyncio baseline vs uvloop performance."""
        logger.info("Starting asyncio vs uvloop performance comparison")

        # Get current loop info
        initial_loop_info = self.event_loop_manager.get_loop_info()

        # Run benchmark with current loop (might be uvloop if already configured)
        current_results = await self.run_comprehensive_benchmark()

        # Store results based on loop type
        if initial_loop_info["uvloop_detected"]:
            self.uvloop_results = current_results
            logger.info("Benchmarked uvloop performance")
        else:
            self.baseline_results = current_results
            logger.info("Benchmarked asyncio baseline performance")

        # If we have both results, calculate comparison
        if self.baseline_results and self.uvloop_results:
            return self._calculate_comparison()

        return {
            "current_results": current_results,
            "loop_type": initial_loop_info["loop_type"],
            "uvloop_detected": initial_loop_info["uvloop_detected"],
            "comparison_available": False,
        }

    def _calculate_comparison(self) -> dict[str, Any]:
        """Calculate performance comparison between asyncio and uvloop."""
        if not self.baseline_results or not self.uvloop_results:
            return {"error": "Both baseline and uvloop results required for comparison"}

        baseline_avg = self.baseline_results["overall_avg_ms"]
        uvloop_avg = self.uvloop_results["overall_avg_ms"]

        baseline_p95 = self.baseline_results["overall_p95_ms"]
        uvloop_p95 = self.uvloop_results["overall_p95_ms"]

        # Calculate improvements
        avg_improvement = ((baseline_avg - uvloop_avg) / baseline_avg) * 100
        p95_improvement = ((baseline_p95 - uvloop_p95) / baseline_p95) * 100

        return {
            "baseline_results": self.baseline_results,
            "uvloop_results": self.uvloop_results,
            "comparison": {
                "avg_improvement_percent": avg_improvement,
                "p95_improvement_percent": p95_improvement,
                "baseline_avg_ms": baseline_avg,
                "uvloop_avg_ms": uvloop_avg,
                "baseline_p95_ms": baseline_p95,
                "uvloop_p95_ms": uvloop_p95,
                "uvloop_faster": uvloop_avg < baseline_avg,
                "both_meet_target": (
                    baseline_avg < self.target_latency_ms
                    and uvloop_avg < self.target_latency_ms
                ),
            },
            "timestamp": time.time(),
        }

    async def benchmark_prompt_improvement_simulation(
        self, prompt_count: int = 50
    ) -> dict[str, Any]:
        """Simulate prompt improvement operations for realistic benchmarking."""
        logger.info(f"Simulating {prompt_count} prompt improvement operations")

        latencies = []

        async def simulate_prompt_improvement():
            """Simulate a typical prompt improvement operation."""
            # Simulate database query
            await asyncio.sleep(0.01)

            # Simulate rule processing
            await asyncio.sleep(0.02)

            # Simulate ML model inference
            await asyncio.sleep(0.05)

            # Simulate result formatting
            await asyncio.sleep(0.01)

            return "improved_prompt"

        for i in range(prompt_count):
            start = time.perf_counter()

            # Use session wrapper for realistic session management
            session_wrapper = get_session_wrapper(f"benchmark_session_{i}")

            async with session_wrapper.performance_context("prompt_improvement"):
                result = await simulate_prompt_improvement()

            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

        stats = self._calculate_statistics(latencies)

        return {
            **stats,
            "simulation_type": "prompt_improvement",
            "prompt_count": prompt_count,
            "meets_target": stats["avg_ms"] < self.target_latency_ms,
            "target_latency_ms": self.target_latency_ms,
        }

    async def run_session_benchmark(self, session_count: int = 10) -> dict[str, Any]:
        """Benchmark session-scoped performance."""
        logger.info(f"Benchmarking {session_count} concurrent sessions")

        results = {}

        async def session_workload(session_id: str):
            """Simulate workload for a single session."""
            wrapper = get_session_wrapper(session_id)

            # Simulate multiple operations in the session
            for i in range(10):
                async with wrapper.performance_context(f"operation_{i}"):
                    await asyncio.sleep(0.01)

            return wrapper.get_metrics()

        # Run sessions concurrently
        start_time = time.perf_counter()

        tasks = [
            asyncio.create_task(session_workload(f"session_{i}"))
            for i in range(session_count)
        ]

        session_results = await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        # Aggregate session metrics
        avg_session_time = statistics.mean([
            result["avg_time_ms"] for result in session_results
        ])

        total_operations = sum([
            result["operations_count"] for result in session_results
        ])

        return {
            "session_count": session_count,
            "total_time_ms": total_time_ms,
            "avg_session_time_ms": avg_session_time,
            "total_operations": total_operations,
            "throughput_ops_per_sec": total_operations / (total_time_ms / 1000),
            "meets_target": avg_session_time < self.target_latency_ms,
            "session_results": session_results,
        }

    def get_benchmark_summary(self) -> dict[str, Any]:
        """Get summary of all benchmark results."""
        return {
            "baseline_results": self.baseline_results,
            "uvloop_results": self.uvloop_results,
            "target_latency_ms": self.target_latency_ms,
            "comparison_available": bool(self.baseline_results and self.uvloop_results),
        }

async def run_startup_benchmark() -> dict[str, Any]:
    """Run benchmark during application startup."""
    logger.info("Running startup performance benchmark")

    benchmark = EventLoopBenchmark()

    # Run quick benchmark to verify performance
    results = await benchmark.run_latency_benchmark(
        samples=50, operation_type="sleep_yield"
    )

    if results["meets_target"]:
        logger.info(
            f"✅ Event loop meets target latency: {results['avg_ms']:.2f}ms < {benchmark.target_latency_ms}ms"
        )
    else:
        logger.warning(
            f"⚠️ Event loop exceeds target latency: {results['avg_ms']:.2f}ms > {benchmark.target_latency_ms}ms"
        )

    return results

async def run_full_benchmark_suite() -> dict[str, Any]:
    """Run the complete benchmark suite."""
    logger.info("Starting full benchmark suite")

    benchmark = EventLoopBenchmark()

    # Run all benchmarks
    comprehensive_results = await benchmark.run_comprehensive_benchmark()
    prompt_simulation = await benchmark.benchmark_prompt_improvement_simulation()
    session_benchmark = await benchmark.run_session_benchmark()

    return {
        "comprehensive_benchmark": comprehensive_results,
        "prompt_simulation": prompt_simulation,
        "session_benchmark": session_benchmark,
        "summary": benchmark.get_benchmark_summary(),
        "timestamp": time.time(),
    }
