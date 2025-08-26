"""Comprehensive Validation Performance Benchmarking System.

This module provides detailed performance analysis and benchmarking for validation
bottlenecks identified in the Validation_Consolidation.md analysis.

Key Performance Targets:
- MCP Server message handling: 543μs -> 6.4μs (85x faster)
- Config instantiation: 54.3μs -> 8.4μs (6.5x faster)
- Metrics collection: 12.1μs -> 1.0μs (12x faster)
- Memory usage: 2.5KB -> 0.4KB per instance (84% reduction)

This benchmarking system provides:
1. Real-world data payload simulation
2. Memory leak detection with tracemalloc
3. Performance regression detection
4. Concurrent validation stress testing
"""

import asyncio
import contextlib
import gc
import json
import logging
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles
import psutil

logger = logging.getLogger(__name__)


@dataclass
class ValidationBenchmarkResult:
    """Results from validation performance benchmark."""

    operation_name: str
    current_performance_us: float  # microseconds
    target_performance_us: float
    improvement_factor: float
    memory_usage_kb: float
    samples_count: int
    p95_latency_us: float
    p99_latency_us: float
    success_rate: float
    timestamp: datetime
    regression_detected: bool = False
    memory_leak_detected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def meets_target(self) -> bool:
        """Check if current performance meets the target."""
        return self.current_performance_us <= self.target_performance_us

    @property
    def improvement_percent(self) -> float:
        """Calculate improvement percentage over baseline."""
        baseline_us = self.target_performance_us * self.improvement_factor
        return (
            ((baseline_us - self.current_performance_us) / baseline_us * 100)
            if baseline_us > 0
            else 0
        )

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_name": self.operation_name,
            "current_performance_us": self.current_performance_us,
            "target_performance_us": self.target_performance_us,
            "improvement_factor": self.improvement_factor,
            "memory_usage_kb": self.memory_usage_kb,
            "samples_count": self.samples_count,
            "p95_latency_us": self.p95_latency_us,
            "p99_latency_us": self.p99_latency_us,
            "success_rate": self.success_rate,
            "meets_target": self.meets_target,
            "improvement_percent": self.improvement_percent,
            "regression_detected": self.regression_detected,
            "memory_leak_detected": self.memory_leak_detected,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class MemoryProfile:
    """Memory profiling results."""

    initial_memory_mb: float
    peak_memory_mb: float
    final_memory_mb: float
    memory_growth_mb: float
    allocation_count: int
    deallocation_count: int
    top_allocations: list[tuple[str, int]]
    leak_detected: bool = False

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "initial_memory_mb": self.initial_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "final_memory_mb": self.final_memory_mb,
            "memory_growth_mb": self.memory_growth_mb,
            "allocation_count": self.allocation_count,
            "deallocation_count": self.deallocation_count,
            "top_allocations": self.top_allocations,
            "leak_detected": self.leak_detected,
        }


class ValidationBenchmarkFramework:
    """Comprehensive validation performance benchmarking framework."""

    def __init__(self) -> None:
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)

        # Performance targets from Validation_Consolidation.md
        self.performance_targets = {
            "mcp_message_decode": 6.4,  # μs - 85x improvement target
            "config_instantiation": 8.4,  # μs - 6.5x improvement target
            "metrics_collection": 1.0,  # μs - 12x improvement target
        }

        # Current baseline measurements
        self.current_baselines = {
            "mcp_message_decode": 543.0,  # μs
            "config_instantiation": 54.3,  # μs
            "metrics_collection": 12.1,  # μs
        }

        self.memory_profile: MemoryProfile | None = None

    async def run_comprehensive_benchmark(
        self,
        operations: int = 10000,
        memory_operations: int = 100000,
        concurrent_sessions: int = 100,
    ) -> dict[str, ValidationBenchmarkResult]:
        """Run comprehensive validation benchmarking suite.

        Args:
            operations: Number of operations for standard benchmarks
            memory_operations: Number of operations for memory leak testing
            concurrent_sessions: Number of concurrent sessions for stress testing

        Returns:
            Dictionary of benchmark results by operation name
        """
        logger.info("Starting comprehensive validation benchmark suite")
        logger.info(
            f"Operations: {operations}, Memory ops: {memory_operations}, Concurrent: {concurrent_sessions}"
        )

        results = {}

        # 1. MCP Server message handling benchmark
        logger.info("Benchmarking MCP message decode validation...")
        result = await self._benchmark_mcp_message_decode(operations)
        if result:
            results["mcp_message_decode"] = result

        # 2. Config instantiation benchmark
        logger.info("Benchmarking config instantiation...")
        result = await self._benchmark_config_instantiation(operations)
        if result:
            results["config_instantiation"] = result

        # 3. Metrics collection benchmark
        logger.info("Benchmarking metrics collection...")
        result = await self._benchmark_metrics_collection(operations)
        if result:
            results["metrics_collection"] = result

        # 4. High-frequency operations stress test
        logger.info("Running high-frequency operations stress test...")
        result = await self._benchmark_high_frequency_operations(operations * 2)
        if result:
            results["high_frequency_stress"] = result

        # 5. Memory leak detection
        logger.info("Running memory leak detection...")
        memory_results = await self._benchmark_memory_usage(memory_operations)
        for name, result in memory_results.items():
            results[f"{name}_memory"] = result

        # 6. Concurrent validation stress test
        logger.info("Running concurrent validation stress test...")
        result = await self._benchmark_concurrent_validation(concurrent_sessions)
        if result:
            results["concurrent_stress"] = result

        # 7. Generate comprehensive report
        await self._generate_benchmark_report(results)

        # 8. Check for performance regressions
        await self._detect_performance_regressions(results)

        logger.info("Comprehensive validation benchmark completed")
        return results

    async def _benchmark_mcp_message_decode(
        self, operations: int
    ) -> ValidationBenchmarkResult | None:
        """Benchmark MCP message decoding and validation performance."""
        try:
            # Import here to avoid circular imports
            from prompt_improver.core.types import PromptImprovementRequest
            from prompt_improver.mcp_server.server import APESMCPServer

            server = APESMCPServer()

            # Test data simulating real MCP messages
            test_messages = [
                {
                    "prompt": f"Test prompt {i}: Write a Python function to calculate fibonacci numbers efficiently.",
                    "context": {
                        "domain": "programming",
                        "language": "python",
                        "complexity": "medium",
                    },
                    "session_id": f"benchmark_session_{i % 100}",
                }
                for i in range(min(operations, 1000))  # Limit to avoid memory issues
            ]

            latencies = []
            errors = 0

            # Warmup
            for _ in range(10):
                try:
                    request = PromptImprovementRequest(**test_messages[0])
                    await server._improve_prompt_impl(
                        prompt=request.prompt,
                        context=request.context,
                        session_id=request.session_id,
                    )
                except Exception:
                    pass

            # Actual benchmark
            start_time = time.perf_counter()

            for i in range(operations):
                msg_data = test_messages[i % len(test_messages)]

                operation_start = time.perf_counter()
                try:
                    # Simulate full message decode + validation
                    request = PromptImprovementRequest(**msg_data)

                    # Simulate validation processing
                    result = await server._improve_prompt_impl(
                        prompt=request.prompt,
                        context=request.context,
                        session_id=request.session_id,
                    )

                    if not result or "improved_prompt" not in result:
                        errors += 1

                except Exception as e:
                    logger.debug(f"MCP benchmark error: {e}")
                    errors += 1

                operation_time = (
                    time.perf_counter() - operation_start
                ) * 1_000_000  # Convert to microseconds
                latencies.append(operation_time)

                # Yield control occasionally for high operation counts
                if i % 1000 == 0:
                    await asyncio.sleep(0)

            total_time = time.perf_counter() - start_time

            return self._create_benchmark_result(
                operation_name="mcp_message_decode",
                latencies=latencies,
                errors=errors,
                total_operations=operations,
                memory_kb=self._get_memory_usage_kb(),
            )

        except Exception as e:
            logger.exception(f"MCP message decode benchmark failed: {e}")
            return None

    async def _benchmark_config_instantiation(
        self, operations: int
    ) -> ValidationBenchmarkResult | None:
        """Benchmark configuration instantiation performance."""
        try:
            from prompt_improver.core.config import (
                AppConfig,
                DatabaseConfig,
                MCPConfig,
                RedisConfig,
                SecurityConfig,
            )

            latencies = []
            errors = 0

            # Test with various config scenarios
            config_scenarios = [
                {"environment": "production", "debug": False},
                {"environment": "development", "debug": True},
                {"environment": "testing", "debug": False},
                {"environment": "staging", "debug": False},
            ]

            # Warmup
            for _ in range(10):
                with contextlib.suppress(Exception):
                    config = AppConfig(**config_scenarios[0])

            start_time = time.perf_counter()

            for i in range(operations):
                scenario = config_scenarios[i % len(config_scenarios)]

                operation_start = time.perf_counter()
                try:
                    # Full config instantiation with all nested configs
                    config = AppConfig(
                        **scenario,
                        database=DatabaseConfig(),
                        redis=RedisConfig(),
                        mcp=MCPConfig(),
                        security=SecurityConfig(),
                    )

                    # Simulate config validation/access
                    _ = config.database.database_url
                    _ = config.redis.redis_url
                    _ = config.mcp.mcp_host

                except Exception as e:
                    logger.debug(f"Config instantiation error: {e}")
                    errors += 1

                operation_time = (time.perf_counter() - operation_start) * 1_000_000
                latencies.append(operation_time)

                if i % 1000 == 0:
                    await asyncio.sleep(0)

            return self._create_benchmark_result(
                operation_name="config_instantiation",
                latencies=latencies,
                errors=errors,
                total_operations=operations,
                memory_kb=self._get_memory_usage_kb(),
            )

        except Exception as e:
            logger.exception(f"Config instantiation benchmark failed: {e}")
            return None

    async def _benchmark_metrics_collection(
        self, operations: int
    ) -> ValidationBenchmarkResult | None:
        """Benchmark metrics collection performance."""
        try:
            from prompt_improver.metrics.api_metrics import (
                APIMetricsCollector,
                APIUsageMetric,
                EndpointCategory,
                HTTPMethod,
            )

            collector = APIMetricsCollector()
            await collector.start_collection()

            latencies = []
            errors = 0

            # Test metrics data
            api_metrics_data = [
                {
                    "endpoint": f"/api/improve-prompt/{i % 100}",
                    "method": HTTPMethod.POST,
                    "category": EndpointCategory.PROMPT_IMPROVEMENT,
                    "status_code": 200,
                    "response_time_ms": 150.5,
                    "request_size_bytes": 1024,
                    "response_size_bytes": 2048,
                }
                for i in range(min(operations, 1000))
            ]

            # Warmup
            for _ in range(10):
                try:
                    metric = APIUsageMetric(**api_metrics_data[0])
                    await collector.record_api_usage(metric)
                except Exception:
                    pass

            start_time = time.perf_counter()

            for i in range(operations):
                metric_data = api_metrics_data[i % len(api_metrics_data)]

                operation_start = time.perf_counter()
                try:
                    # Full metrics creation and collection
                    metric = APIUsageMetric(**metric_data)
                    await collector.record_api_usage(metric)

                except Exception as e:
                    logger.debug(f"Metrics collection error: {e}")
                    errors += 1

                operation_time = (time.perf_counter() - operation_start) * 1_000_000
                latencies.append(operation_time)

                if i % 1000 == 0:
                    await asyncio.sleep(0)

            await collector.stop_collection()

            return self._create_benchmark_result(
                operation_name="metrics_collection",
                latencies=latencies,
                errors=errors,
                total_operations=operations,
                memory_kb=self._get_memory_usage_kb(),
            )

        except Exception as e:
            logger.exception(f"Metrics collection benchmark failed: {e}")
            return None

    async def _benchmark_high_frequency_operations(
        self, operations: int
    ) -> ValidationBenchmarkResult | None:
        """Benchmark high-frequency validation operations (10k+ ops/sec target)."""
        try:
            from prompt_improver.core.types import PromptImprovementRequest

            latencies = []
            errors = 0

            # Simulate lightweight validation operations
            test_data = [
                {
                    "prompt": f"Quick validation test {i}",
                    "session_id": f"freq_test_{i % 50}",
                }
                for i in range(min(operations, 500))
            ]

            start_time = time.perf_counter()

            for i in range(operations):
                data = test_data[i % len(test_data)]

                operation_start = time.perf_counter()
                try:
                    # Fast validation operation
                    request = PromptImprovementRequest(**data)
                    # Simulate basic validation
                    if len(request.prompt) > 0 and request.session_id:
                        validation_passed = True
                    else:
                        errors += 1

                except Exception as e:
                    logger.debug(f"High frequency operation error: {e}")
                    errors += 1

                operation_time = (time.perf_counter() - operation_start) * 1_000_000
                latencies.append(operation_time)

                # Minimal yielding for very high frequency
                if i % 5000 == 0:
                    await asyncio.sleep(0)

            return self._create_benchmark_result(
                operation_name="high_frequency_stress",
                latencies=latencies,
                errors=errors,
                total_operations=operations,
                memory_kb=self._get_memory_usage_kb(),
                target_us=1.0,  # Very aggressive target for high-frequency
            )

        except Exception as e:
            logger.exception(f"High frequency operations benchmark failed: {e}")
            return None

    async def _benchmark_memory_usage(
        self, operations: int
    ) -> dict[str, ValidationBenchmarkResult]:
        """Run memory leak detection benchmark."""
        results = {}

        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        try:
            # Test each validation type for memory leaks
            for operation_type in [
                "mcp_message_decode",
                "config_instantiation",
                "metrics_collection",
            ]:
                logger.info(f"Memory testing {operation_type}...")

                # Take initial snapshot
                snapshot_start = tracemalloc.take_snapshot()
                gc.collect()
                memory_start = process.memory_info().rss / 1024 / 1024

                # Run operations
                latencies = []
                errors = 0

                for i in range(operations):
                    operation_start = time.perf_counter()

                    try:
                        if operation_type == "mcp_message_decode":
                            from prompt_improver.core.types import (
                                PromptImprovementRequest,
                            )

                            request = PromptImprovementRequest(
                                prompt=f"Memory test prompt {i}",
                                session_id=f"mem_test_{i}",
                            )

                        elif operation_type == "config_instantiation":
                            from prompt_improver.core.config import AppConfig

                            config = AppConfig(environment="testing", debug=False)

                        elif operation_type == "metrics_collection":
                            from prompt_improver.metrics.api_metrics import (
                                APIUsageMetric,
                                EndpointCategory,
                                HTTPMethod,
                            )

                            metric = APIUsageMetric(
                                endpoint=f"/test/{i}",
                                method=HTTPMethod.GET,
                                category=EndpointCategory.OTHER,
                                status_code=200,
                                response_time_ms=10.0,
                                request_size_bytes=100,
                                response_size_bytes=200,
                            )

                    except Exception as e:
                        errors += 1
                        logger.debug(f"Memory test error in {operation_type}: {e}")

                    operation_time = (time.perf_counter() - operation_start) * 1_000_000
                    latencies.append(operation_time)

                    # Force cleanup occasionally
                    if i % 10000 == 0:
                        gc.collect()
                        await asyncio.sleep(0.001)

                # Take final snapshot and analyze
                snapshot_end = tracemalloc.take_snapshot()
                gc.collect()
                memory_end = process.memory_info().rss / 1024 / 1024

                # Analyze memory growth
                top_stats = snapshot_end.compare_to(snapshot_start, "lineno")
                memory_growth = memory_end - memory_start
                leak_detected = memory_growth > 100  # MB threshold for leak detection

                # Create memory profile
                memory_profile = MemoryProfile(
                    initial_memory_mb=memory_start,
                    peak_memory_mb=max(memory_start, memory_end),
                    final_memory_mb=memory_end,
                    memory_growth_mb=memory_growth,
                    allocation_count=len([
                        stat for stat in top_stats if stat.size_diff > 0
                    ]),
                    deallocation_count=len([
                        stat for stat in top_stats if stat.size_diff < 0
                    ]),
                    top_allocations=[
                        (str(stat.traceback), stat.size_diff) for stat in top_stats[:10]
                    ],
                    leak_detected=leak_detected,
                )

                result = self._create_benchmark_result(
                    operation_name=f"{operation_type}_memory",
                    latencies=latencies,
                    errors=errors,
                    total_operations=operations,
                    memory_kb=memory_end * 1024,
                )
                result.memory_leak_detected = leak_detected
                result.metadata["memory_profile"] = asdict(memory_profile)

                results[operation_type] = result

                if leak_detected:
                    logger.warning(
                        f"Memory leak detected in {operation_type}: {memory_growth:.1f}MB growth"
                    )

        finally:
            tracemalloc.stop()

        return results

    async def _benchmark_concurrent_validation(
        self, concurrent_sessions: int
    ) -> ValidationBenchmarkResult | None:
        """Benchmark concurrent validation performance."""
        try:
            from prompt_improver.core.types import PromptImprovementRequest

            async def concurrent_validation_task(session_id: int) -> tuple[float, bool]:
                """Single concurrent validation task."""
                start_time = time.perf_counter()
                try:
                    request = PromptImprovementRequest(
                        prompt=f"Concurrent validation test from session {session_id}",
                        session_id=f"concurrent_{session_id}",
                    )
                    # Simulate validation work
                    await asyncio.sleep(0.001)  # 1ms simulated work
                    success = len(request.prompt) > 0

                except Exception:
                    success = False

                duration = (time.perf_counter() - start_time) * 1_000_000
                return (duration, success)

            # Run concurrent tasks
            tasks = [concurrent_validation_task(i) for i in range(concurrent_sessions)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            latencies = []
            errors = 0

            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                    latencies.append(10000)  # 10ms penalty for errors
                else:
                    duration, success = result
                    latencies.append(duration)
                    if not success:
                        errors += 1

            return self._create_benchmark_result(
                operation_name="concurrent_stress",
                latencies=latencies,
                errors=errors,
                total_operations=concurrent_sessions,
                memory_kb=self._get_memory_usage_kb(),
                target_us=50.0,  # 50μs target for concurrent operations
            )

        except Exception as e:
            logger.exception(f"Concurrent validation benchmark failed: {e}")
            return None

    def _create_benchmark_result(
        self,
        operation_name: str,
        latencies: list[float],
        errors: int,
        total_operations: int,
        memory_kb: float,
        target_us: float | None = None,
    ) -> ValidationBenchmarkResult:
        """Create a benchmark result from collected data."""
        if not latencies:
            latencies = [0.0]

        avg_latency = statistics.mean(latencies)
        p95_latency = self._percentile(latencies, 95)
        p99_latency = self._percentile(latencies, 99)
        success_rate = (
            (total_operations - errors) / total_operations
            if total_operations > 0
            else 0.0
        )

        # Get target and improvement factor
        if target_us is None:
            target_us = self.performance_targets.get(operation_name, avg_latency)

        baseline_us = self.current_baselines.get(operation_name, avg_latency)
        improvement_factor = baseline_us / target_us if target_us > 0 else 1.0

        return ValidationBenchmarkResult(
            operation_name=operation_name,
            current_performance_us=avg_latency,
            target_performance_us=target_us,
            improvement_factor=improvement_factor,
            memory_usage_kb=memory_kb,
            samples_count=len(latencies),
            p95_latency_us=p95_latency,
            p99_latency_us=p99_latency,
            success_rate=success_rate,
            timestamp=datetime.now(UTC),
            metadata={
                "total_operations": total_operations,
                "error_count": errors,
                "min_latency_us": min(latencies),
                "max_latency_us": max(latencies),
                "std_latency_us": statistics.stdev(latencies)
                if len(latencies) > 1
                else 0.0,
            },
        )

    def _percentile(self, data: list[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = percentile / 100 * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        lower = sorted_data[int(index)]
        upper = sorted_data[int(index) + 1]
        return lower + (upper - lower) * (index - int(index))

    def _get_memory_usage_kb(self) -> float:
        """Get current memory usage in KB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024
        except Exception:
            return 0.0

    async def _detect_performance_regressions(
        self, results: dict[str, ValidationBenchmarkResult]
    ) -> None:
        """Detect performance regressions by comparing with historical data."""
        baseline_file = self.results_dir / "baseline_results.json"

        if not baseline_file.exists():
            logger.info("No baseline file found - saving current results as baseline")
            await self._save_baseline(results)
            return

        try:
            async with aiofiles.open(baseline_file) as f:
                content = await f.read()
                baseline_data = json.loads(content)

            for operation_name, result in results.items():
                if operation_name in baseline_data:
                    baseline_perf = baseline_data[operation_name][
                        "current_performance_us"
                    ]
                    current_perf = result.current_performance_us

                    # Detect regression (>10% performance degradation)
                    regression_threshold = baseline_perf * 1.1
                    if current_perf > regression_threshold:
                        result.regression_detected = True
                        logger.warning(
                            f"Performance regression detected in {operation_name}: "
                            f"{baseline_perf:.2f}μs -> {current_perf:.2f}μs "
                            f"({((current_perf - baseline_perf) / baseline_perf * 100):+.1f}%)"
                        )

        except Exception as e:
            logger.exception(f"Error detecting regressions: {e}")

    async def _save_baseline(
        self, results: dict[str, ValidationBenchmarkResult]
    ) -> None:
        """Save current results as baseline for future regression detection."""
        baseline_file = self.results_dir / "baseline_results.json"
        baseline_data = {name: asdict(result) for name, result in results.items()}

        async with aiofiles.open(baseline_file, "w") as f:
            await f.write(json.dumps(baseline_data, indent=2))

        logger.info(f"Baseline saved to {baseline_file}")

    async def _generate_benchmark_report(
        self, results: dict[str, ValidationBenchmarkResult]
    ) -> None:
        """Generate comprehensive benchmark report."""
        report_file = (
            self.results_dir / f"validation_benchmark_report_{int(time.time())}.json"
        )

        # Calculate overall metrics
        total_operations = sum(r.samples_count for r in results.values())
        operations_meeting_target = sum(1 for r in results.values() if r.meets_target)
        regressions_detected = sum(1 for r in results.values() if r.regression_detected)
        memory_leaks_detected = sum(
            1 for r in results.values() if r.memory_leak_detected
        )

        avg_improvement = statistics.mean([
            r.improvement_percent for r in results.values()
        ])

        report = {
            "benchmark_timestamp": datetime.now(UTC).isoformat(),
            "summary": {
                "total_operations_tested": total_operations,
                "total_test_categories": len(results),
                "operations_meeting_target": operations_meeting_target,
                "target_compliance_rate": operations_meeting_target / len(results)
                if results
                else 0,
                "performance_regressions_detected": regressions_detected,
                "memory_leaks_detected": memory_leaks_detected,
                "average_improvement_percent": avg_improvement,
                "benchmark_passed": regressions_detected == 0
                and memory_leaks_detected == 0,
            },
            "performance_targets": self.performance_targets,
            "current_baselines": self.current_baselines,
            "detailed_results": {
                name: asdict(result) for name, result in results.items()
            },
            "recommendations": self._generate_recommendations(results),
        }

        async with aiofiles.open(report_file, "w") as f:
            await f.write(json.dumps(report, indent=2))

        logger.info(f"Comprehensive benchmark report saved to {report_file}")

        # Also create human-readable summary
        await self._generate_summary_report(report)

    async def _generate_summary_report(self, report_data: dict[str, Any]) -> None:
        """Generate human-readable summary report."""
        summary_file = (
            self.results_dir / f"validation_benchmark_summary_{int(time.time())}.txt"
        )

        lines = [
            "=" * 80,
            "VALIDATION PERFORMANCE BENCHMARK REPORT",
            "=" * 80,
            f"Generated: {datetime.now(UTC).isoformat()}",
            f"Total Operations Tested: {report_data['summary']['total_operations_tested']:,}",
            f"Test Categories: {report_data['summary']['total_test_categories']}",
            "",
            "PERFORMANCE TARGETS:",
            "-" * 40,
        ]

        for operation, target_us in self.performance_targets.items():
            baseline_us = self.current_baselines.get(operation, 0)
            improvement_factor = baseline_us / target_us if target_us > 0 else 1
            lines.extend([
                f"  {operation.replace('_', ' ').title()}:",
                f"    Current Baseline: {baseline_us:.1f}μs",
                f"    Target: {target_us:.1f}μs",
                f"    Target Improvement: {improvement_factor:.1f}x faster",
                "",
            ])

        lines.extend(["TEST RESULTS SUMMARY:", "-" * 40])

        for name, result_data in report_data["detailed_results"].items():
            status = "✅ PASS" if result_data["meets_target"] else "❌ FAIL"
            regression = " 🔍 REGRESSION" if result_data["regression_detected"] else ""
            memory_leak = (
                " 🚨 MEMORY LEAK" if result_data["memory_leak_detected"] else ""
            )

            lines.extend([
                f"  {name.replace('_', ' ').title()}: {status}{regression}{memory_leak}",
                f"    Performance: {result_data['current_performance_us']:.2f}μs (target: {result_data['target_performance_us']:.2f}μs)",
                f"    P95 Latency: {result_data['p95_latency_us']:.2f}μs",
                f"    Success Rate: {result_data['success_rate']:.1%}",
                f"    Memory Usage: {result_data['memory_usage_kb']:.1f}KB",
                f"    Samples: {result_data['samples_count']:,}",
                "",
            ])

        lines.extend([
            "OVERALL SUMMARY:",
            "-" * 20,
            f"Target Compliance: {report_data['summary']['target_compliance_rate']:.1%}",
            f"Average Improvement: {report_data['summary']['average_improvement_percent']:+.1f}%",
            f"Regressions Detected: {report_data['summary']['performance_regressions_detected']}",
            f"Memory Leaks Detected: {report_data['summary']['memory_leaks_detected']}",
            f"Benchmark Status: {'✅ PASSED' if report_data['summary']['benchmark_passed'] else '❌ FAILED'}",
            "",
            "RECOMMENDATIONS:",
            "-" * 20,
        ])

        lines.extend(f"  • {recommendation}" for recommendation in report_data["recommendations"])

        lines.extend(["", "=" * 80])

        async with aiofiles.open(summary_file, "w") as f:
            await f.write("\n".join(lines))

        logger.info(f"Human-readable summary saved to {summary_file}")

    def _generate_recommendations(
        self, results: dict[str, ValidationBenchmarkResult]
    ) -> list[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        for name, result in results.items():
            if not result.meets_target:
                if result.current_performance_us > result.target_performance_us * 2:
                    recommendations.append(
                        f"{name}: Critical performance gap - current {result.current_performance_us:.1f}μs "
                        f"vs target {result.target_performance_us:.1f}μs. Consider msgspec migration."
                    )
                else:
                    recommendations.append(
                        f"{name}: Minor performance gap - optimize validation logic and reduce object creation."
                    )

            if result.regression_detected:
                recommendations.append(
                    f"{name}: Performance regression detected - investigate recent changes and "
                    f"consider rolling back or optimizing."
                )

            if result.memory_leak_detected:
                recommendations.append(
                    f"{name}: Memory leak detected - review object lifecycle and ensure proper cleanup."
                )

            if result.success_rate < 0.99:
                recommendations.append(
                    f"{name}: Success rate below 99% ({result.success_rate:.1%}) - investigate error causes."
                )

        if not recommendations:
            recommendations.append(
                "All performance targets met! System is performing optimally. "
                "Consider tightening targets for continuous improvement."
            )

        return recommendations


async def run_validation_benchmark(
    operations: int = 10000,
    memory_operations: int = 100000,
    concurrent_sessions: int = 100,
) -> dict[str, ValidationBenchmarkResult]:
    """Run comprehensive validation benchmarking suite.

    Args:
        operations: Number of operations for standard benchmarks
        memory_operations: Number of operations for memory leak testing
        concurrent_sessions: Number of concurrent sessions for stress testing

    Returns:
        Dictionary of benchmark results
    """
    framework = ValidationBenchmarkFramework()
    return await framework.run_comprehensive_benchmark(
        operations=operations,
        memory_operations=memory_operations,
        concurrent_sessions=concurrent_sessions,
    )


# CLI interface for standalone benchmarking
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validation Performance Benchmarking")
    parser.add_argument(
        "--operations", type=int, default=10000, help="Operations per benchmark"
    )
    parser.add_argument(
        "--memory-operations",
        type=int,
        default=100000,
        help="Operations for memory testing",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=100,
        help="Concurrent sessions for stress testing",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmark_results", help="Output directory"
    )

    args = parser.parse_args()

    async def main():
        framework = ValidationBenchmarkFramework()
        if args.output_dir != "benchmark_results":
            framework.results_dir = Path(args.output_dir)
            framework.results_dir.mkdir(exist_ok=True)

        results = await framework.run_comprehensive_benchmark(
            operations=args.operations,
            memory_operations=args.memory_operations,
            concurrent_sessions=args.concurrent,
        )

        print("Benchmarking completed. Results saved to:", framework.results_dir)
        return results

    asyncio.run(main())
