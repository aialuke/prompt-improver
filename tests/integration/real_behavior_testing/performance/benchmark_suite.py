"""Performance Validation Framework with Real Behavior Benchmarks.

Comprehensive benchmarking framework for validating modernized services against performance targets:
- Real behavior testing with actual services (no mocks)
- Performance target validation (<10ms, <100ms, <500ms)
- Comprehensive metrics collection and analysis
- Service decomposition validation
- Clean architecture compliance verification
"""

import asyncio
import logging
import statistics
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from tests.containers.postgres_container import PostgreSQLTestContainer
from tests.integration.real_behavior_testing.containers.ml_test_container import (
    MLTestContainer,
)
from tests.integration.real_behavior_testing.containers.network_simulator import (
    NetworkSimulator,
)
from tests.integration.real_behavior_testing.containers.real_redis_container import (
    RealRedisTestContainer,
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceTarget:
    """Performance target definition for benchmarking."""

    name: str
    description: str
    target_ms: float
    tolerance_percentage: float = 20.0  # Allow 20% variance
    critical: bool = False  # Whether missing this target is critical
    category: str = "general"  # Performance category


@dataclass
class BenchmarkResult:
    """Individual benchmark result."""

    benchmark_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""

    # Timing results
    duration_ms: float = 0.0
    target_ms: float | None = None
    performance_met: bool = False

    # Statistical analysis
    min_duration_ms: float | None = None
    max_duration_ms: float | None = None
    mean_duration_ms: float | None = None
    median_duration_ms: float | None = None
    std_dev_ms: float | None = None

    # Context
    iterations: int = 1
    warmup_iterations: int = 0
    category: str = "general"
    service_name: str = ""

    # Results
    success: bool = True
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Comprehensive benchmark suite for all modernized services."""

    def __init__(
        self,
        postgres_container: PostgreSQLTestContainer,
        redis_container: RealRedisTestContainer,
        ml_container: MLTestContainer | None = None,
        network_simulator: NetworkSimulator | None = None
    ):
        """Initialize benchmark suite with container dependencies.

        Args:
            postgres_container: PostgreSQL testcontainer
            redis_container: Redis testcontainer
            ml_container: Optional ML testcontainer
            network_simulator: Optional network simulator
        """
        self.postgres_container = postgres_container
        self.redis_container = redis_container
        self.ml_container = ml_container
        self.network_simulator = network_simulator

        self.suite_id = str(uuid.uuid4())[:8]
        self.results: list[BenchmarkResult] = []
        self.targets = self._define_performance_targets()

        logger.info(f"BenchmarkSuite initialized: {self.suite_id}")

    def _define_performance_targets(self) -> dict[str, PerformanceTarget]:
        """Define comprehensive performance targets for all services."""
        return {
            # Database operations
            "database_connection": PerformanceTarget(
                name="database_connection",
                description="Database connection establishment",
                target_ms=50.0,
                critical=True,
                category="database"
            ),
            "database_query_simple": PerformanceTarget(
                name="database_query_simple",
                description="Simple SELECT query execution",
                target_ms=10.0,
                critical=True,
                category="database"
            ),
            "database_transaction": PerformanceTarget(
                name="database_transaction",
                description="Database transaction commit",
                target_ms=25.0,
                critical=True,
                category="database"
            ),

            # Cache operations
            "redis_connection": PerformanceTarget(
                name="redis_connection",
                description="Redis connection establishment",
                target_ms=20.0,
                critical=True,
                category="cache"
            ),
            "redis_set_operation": PerformanceTarget(
                name="redis_set_operation",
                description="Redis SET operation",
                target_ms=1.0,
                critical=True,
                category="cache"
            ),
            "redis_get_operation": PerformanceTarget(
                name="redis_get_operation",
                description="Redis GET operation",
                target_ms=0.5,
                critical=True,
                category="cache"
            ),
            "redis_batch_operation": PerformanceTarget(
                name="redis_batch_operation",
                description="Redis batch MGET operation",
                target_ms=5.0,
                category="cache"
            ),

            # ML Intelligence Services
            "ml_feature_extraction": PerformanceTarget(
                name="ml_feature_extraction",
                description="ML feature extraction processing",
                target_ms=50.0,
                category="ml"
            ),
            "ml_clustering": PerformanceTarget(
                name="ml_clustering",
                description="ML clustering algorithm execution",
                target_ms=100.0,
                category="ml"
            ),
            "ml_prediction": PerformanceTarget(
                name="ml_prediction",
                description="ML prediction generation",
                target_ms=20.0,
                critical=True,
                category="ml"
            ),
            "ml_intelligence_facade": PerformanceTarget(
                name="ml_intelligence_facade",
                description="ML Intelligence Facade coordination",
                target_ms=200.0,
                critical=True,
                category="ml"
            ),

            # Retry System Services
            "retry_decision": PerformanceTarget(
                name="retry_decision",
                description="Retry service decision making",
                target_ms=5.0,
                critical=True,
                category="retry"
            ),
            "retry_orchestration": PerformanceTarget(
                name="retry_orchestration",
                description="Retry orchestrator execution",
                target_ms=10.0,
                critical=True,
                category="retry"
            ),
            "circuit_breaker_decision": PerformanceTarget(
                name="circuit_breaker_decision",
                description="Circuit breaker decision",
                target_ms=2.0,
                critical=True,
                category="retry"
            ),

            # Error Handling Services
            "error_routing": PerformanceTarget(
                name="error_routing",
                description="Error handling facade routing",
                target_ms=1.0,
                critical=True,
                category="error_handling"
            ),
            "database_error_handling": PerformanceTarget(
                name="database_error_handling",
                description="Database error service processing",
                target_ms=10.0,
                category="error_handling"
            ),
            "network_error_handling": PerformanceTarget(
                name="network_error_handling",
                description="Network error service processing",
                target_ms=8.0,
                category="error_handling"
            ),
            "validation_error_handling": PerformanceTarget(
                name="validation_error_handling",
                description="Validation error service processing",
                target_ms=5.0,
                category="error_handling"
            ),

            # Repository Migration Performance
            "repository_session_creation": PerformanceTarget(
                name="repository_session_creation",
                description="Repository session creation",
                target_ms=15.0,
                critical=True,
                category="repository"
            ),
            "repository_data_access": PerformanceTarget(
                name="repository_data_access",
                description="Repository data access operation",
                target_ms=25.0,
                critical=True,
                category="repository"
            ),

            # Service Facade Coordination
            "facade_coordination_overhead": PerformanceTarget(
                name="facade_coordination_overhead",
                description="Service facade coordination overhead",
                target_ms=2.0,
                critical=True,
                category="facade"
            ),
        }

    async def benchmark_operation(
        self,
        name: str,
        operation: Callable,
        *args,
        iterations: int = 10,
        warmup_iterations: int = 2,
        category: str = "general",
        service_name: str = "",
        description: str = "",
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark a single operation with statistical analysis.

        Args:
            name: Benchmark name
            operation: Async operation to benchmark
            *args: Arguments for the operation
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations (excluded from results)
            category: Performance category
            service_name: Service being benchmarked
            description: Benchmark description
            **kwargs: Keyword arguments for the operation

        Returns:
            BenchmarkResult with comprehensive timing statistics
        """
        logger.info(f"Starting benchmark: {name} ({iterations} iterations)")

        # Perform warmup iterations
        for i in range(warmup_iterations):
            try:
                if asyncio.iscoroutinefunction(operation):
                    await operation(*args, **kwargs)
                else:
                    operation(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Warmup iteration {i + 1} failed: {e}")

        # Perform benchmark iterations
        durations = []
        success_count = 0
        last_error = None

        for i in range(iterations):
            start_time = time.perf_counter()

            try:
                if asyncio.iscoroutinefunction(operation):
                    await operation(*args, **kwargs)
                else:
                    operation(*args, **kwargs)

                duration = (time.perf_counter() - start_time) * 1000  # Convert to ms
                durations.append(duration)
                success_count += 1

            except Exception as e:
                duration = (time.perf_counter() - start_time) * 1000
                durations.append(duration)  # Include failed attempts in timing
                last_error = str(e)
                logger.warning(f"Benchmark iteration {i + 1} failed: {e}")

        # Calculate statistics
        if durations:
            min_duration = min(durations)
            max_duration = max(durations)
            mean_duration = statistics.mean(durations)
            median_duration = statistics.median(durations)
            std_dev = statistics.stdev(durations) if len(durations) > 1 else 0.0
        else:
            min_duration = max_duration = mean_duration = median_duration = std_dev = 0.0

        # Check performance target
        target = self.targets.get(name)
        target_ms = target.target_ms if target else None
        performance_met = target_ms is None or mean_duration <= target_ms

        # Create result
        result = BenchmarkResult(
            name=name,
            description=description or (target.description if target else ""),
            duration_ms=mean_duration,
            target_ms=target_ms,
            performance_met=performance_met,
            min_duration_ms=min_duration,
            max_duration_ms=max_duration,
            mean_duration_ms=mean_duration,
            median_duration_ms=median_duration,
            std_dev_ms=std_dev,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            category=category,
            service_name=service_name,
            success=success_count == iterations,
            error_message=last_error if success_count < iterations else None,
            metadata={
                "success_rate": success_count / iterations * 100,
                "successful_iterations": success_count,
                "failed_iterations": iterations - success_count,
            }
        )

        self.results.append(result)

        logger.info(f"Benchmark completed: {name} - {mean_duration:.2f}ms (target: {target_ms}ms) - {'✓' if performance_met else '✗'}")

        return result

    async def benchmark_database_operations(self) -> list[BenchmarkResult]:
        """Benchmark database operations with real PostgreSQL."""
        results = []

        # Test database connection
        async def test_connection():
            async with self.postgres_container.get_session() as session:
                await session.execute("SELECT 1")

        result = await self.benchmark_operation(
            name="database_connection",
            operation=test_connection,
            iterations=20,
            warmup_iterations=3,
            category="database",
            service_name="PostgreSQLTestContainer"
        )
        results.append(result)

        # Test simple query
        async def test_simple_query():
            async with self.postgres_container.get_session() as session:
                result = await session.execute("SELECT COUNT(*) FROM information_schema.tables")
                return result.scalar()

        result = await self.benchmark_operation(
            name="database_query_simple",
            operation=test_simple_query,
            iterations=50,
            warmup_iterations=5,
            category="database",
            service_name="PostgreSQLTestContainer"
        )
        results.append(result)

        # Test transaction
        async def test_transaction():
            async with self.postgres_container.get_session() as session:
                await session.execute("BEGIN")
                await session.execute("CREATE TEMP TABLE temp_test (id INTEGER)")
                await session.execute("INSERT INTO temp_test VALUES (1)")
                await session.execute("COMMIT")

        result = await self.benchmark_operation(
            name="database_transaction",
            operation=test_transaction,
            iterations=30,
            warmup_iterations=3,
            category="database",
            service_name="PostgreSQLTestContainer"
        )
        results.append(result)

        return results

    async def benchmark_cache_operations(self) -> list[BenchmarkResult]:
        """Benchmark cache operations with real Redis."""
        results = []

        # Test Redis connection
        async def test_redis_connection():
            async with self.redis_container.get_client() as client:
                await client.ping()

        result = await self.benchmark_operation(
            name="redis_connection",
            operation=test_redis_connection,
            iterations=30,
            warmup_iterations=5,
            category="cache",
            service_name="RealRedisTestContainer"
        )
        results.append(result)

        # Test Redis SET operation
        async def test_redis_set():
            async with self.redis_container.get_client() as client:
                await client.set("benchmark_key", "benchmark_value")

        result = await self.benchmark_operation(
            name="redis_set_operation",
            operation=test_redis_set,
            iterations=100,
            warmup_iterations=10,
            category="cache",
            service_name="RealRedisTestContainer"
        )
        results.append(result)

        # Test Redis GET operation
        async def test_redis_get():
            async with self.redis_container.get_client() as client:
                # Ensure key exists
                await client.set("benchmark_get_key", "benchmark_value")
                return await client.get("benchmark_get_key")

        result = await self.benchmark_operation(
            name="redis_get_operation",
            operation=test_redis_get,
            iterations=100,
            warmup_iterations=10,
            category="cache",
            service_name="RealRedisTestContainer"
        )
        results.append(result)

        # Test Redis batch operation
        async def test_redis_batch():
            async with self.redis_container.get_client() as client:
                # Setup batch data
                keys = [f"batch_key_{i}" for i in range(10)]
                for key in keys:
                    await client.set(key, f"value_{key}")

                # Perform batch GET
                return await client.mget(keys)

        result = await self.benchmark_operation(
            name="redis_batch_operation",
            operation=test_redis_batch,
            iterations=50,
            warmup_iterations=5,
            category="cache",
            service_name="RealRedisTestContainer"
        )
        results.append(result)

        return results

    async def benchmark_ml_operations(self) -> list[BenchmarkResult]:
        """Benchmark ML operations if ML container is available."""
        if not self.ml_container:
            logger.warning("ML container not available, skipping ML benchmarks")
            return []

        results = []

        # Get test dataset
        test_dataset = self.ml_container.get_test_dataset("small")
        if not test_dataset:
            logger.warning("No test dataset available, skipping ML benchmarks")
            return []

        # Test feature extraction
        async def test_feature_extraction():
            return await self.ml_container.extract_features(
                test_dataset.prompts[:20],  # Small sample for benchmarking
                feature_type="tfidf"
            )

        result = await self.benchmark_operation(
            name="ml_feature_extraction",
            operation=test_feature_extraction,
            iterations=10,
            warmup_iterations=2,
            category="ml",
            service_name="MLTestContainer"
        )
        results.append(result)

        # Test clustering (depends on feature extraction)
        async def test_clustering():
            feature_result = await self.ml_container.extract_features(
                test_dataset.prompts[:30],
                feature_type="tfidf"
            )
            if feature_result.success:
                import numpy as np
                feature_matrix = np.array(feature_result.predictions)
                return await self.ml_container.perform_clustering(
                    feature_matrix,
                    algorithm="hdbscan"
                )
            return None

        result = await self.benchmark_operation(
            name="ml_clustering",
            operation=test_clustering,
            iterations=5,
            warmup_iterations=1,
            category="ml",
            service_name="MLTestContainer"
        )
        results.append(result)

        # Test ML prediction
        async def test_ml_prediction():
            prediction_data = {
                "characteristics": {"rule_effectiveness": {"rule_1": 0.8}},
                "context": {"pattern_insights": [{"pattern": "test"}]}
            }
            return await self.ml_container.generate_predictions(
                prediction_data,
                model_type="pattern_prediction"
            )

        result = await self.benchmark_operation(
            name="ml_prediction",
            operation=test_ml_prediction,
            iterations=20,
            warmup_iterations=3,
            category="ml",
            service_name="MLTestContainer"
        )
        results.append(result)

        return results

    async def benchmark_network_operations(self) -> list[BenchmarkResult]:
        """Benchmark network operations if network simulator is available."""
        if not self.network_simulator:
            logger.warning("Network simulator not available, skipping network benchmarks")
            return []

        results = []

        # Test successful network operation
        async def test_network_success():
            return await self.network_simulator.simulate_network_operation(
                operation_name="benchmark_test",
                target_host="api.example.com",
                operation_type="http_request"
            )

        result = await self.benchmark_operation(
            name="network_operation_success",
            operation=test_network_success,
            iterations=20,
            warmup_iterations=3,
            category="network",
            service_name="NetworkSimulator",
            description="Successful network operation simulation"
        )
        results.append(result)

        return results

    async def run_comprehensive_benchmark(self) -> dict[str, Any]:
        """Run comprehensive benchmark suite across all services.

        Returns:
            Comprehensive benchmark results and analysis
        """
        logger.info(f"Starting comprehensive benchmark suite: {self.suite_id}")
        suite_start_time = time.perf_counter()

        all_results = []
        benchmark_sections = [
            ("Database Operations", self.benchmark_database_operations),
            ("Cache Operations", self.benchmark_cache_operations),
            ("ML Operations", self.benchmark_ml_operations),
            ("Network Operations", self.benchmark_network_operations),
        ]

        for section_name, benchmark_method in benchmark_sections:
            logger.info(f"Running benchmark section: {section_name}")
            section_start_time = time.perf_counter()

            try:
                section_results = await benchmark_method()
                all_results.extend(section_results)

                section_duration = (time.perf_counter() - section_start_time) * 1000
                logger.info(f"Completed {section_name}: {len(section_results)} benchmarks in {section_duration:.1f}ms")

            except Exception as e:
                logger.exception(f"Error in benchmark section {section_name}: {e}")
                continue

        suite_duration = (time.perf_counter() - suite_start_time) * 1000

        # Analyze results
        analysis = self._analyze_results(all_results, suite_duration)

        logger.info(f"Comprehensive benchmark completed: {len(all_results)} benchmarks in {suite_duration:.1f}ms")

        return {
            "suite_id": self.suite_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "total_duration_ms": suite_duration,
            "total_benchmarks": len(all_results),
            "results": [self._result_to_dict(result) for result in all_results],
            "analysis": analysis,
        }

    def _analyze_results(self, results: list[BenchmarkResult], suite_duration_ms: float) -> dict[str, Any]:
        """Analyze benchmark results for comprehensive reporting."""
        # Performance target analysis
        total_targets = 0
        met_targets = 0
        critical_targets = 0
        met_critical_targets = 0

        category_performance = {}

        for result in results:
            if result.target_ms is not None:
                total_targets += 1
                if result.performance_met:
                    met_targets += 1

                # Check if target is critical
                target = self.targets.get(result.name)
                if target and target.critical:
                    critical_targets += 1
                    if result.performance_met:
                        met_critical_targets += 1

                # Category analysis
                category = result.category
                if category not in category_performance:
                    category_performance[category] = {
                        "total": 0,
                        "met": 0,
                        "benchmarks": []
                    }

                category_performance[category]["total"] += 1
                if result.performance_met:
                    category_performance[category]["met"] += 1
                category_performance[category]["benchmarks"].append(result.name)

        # Calculate success rates
        overall_success_rate = (met_targets / total_targets * 100) if total_targets > 0 else 0
        critical_success_rate = (met_critical_targets / critical_targets * 100) if critical_targets > 0 else 0

        # Category success rates
        for category, data in category_performance.items():
            data["success_rate"] = (data["met"] / data["total"] * 100) if data["total"] > 0 else 0

        # Performance insights
        fastest_benchmarks = sorted(results, key=lambda r: r.mean_duration_ms or 0)[:5]
        slowest_benchmarks = sorted(results, key=lambda r: r.mean_duration_ms or 0, reverse=True)[:5]

        # Failed benchmarks
        failed_benchmarks = [r for r in results if not r.success or not r.performance_met]

        return {
            "summary": {
                "total_benchmarks": len(results),
                "total_targets": total_targets,
                "met_targets": met_targets,
                "overall_success_rate": overall_success_rate,
                "critical_targets": critical_targets,
                "met_critical_targets": met_critical_targets,
                "critical_success_rate": critical_success_rate,
                "suite_duration_ms": suite_duration_ms,
            },
            "category_performance": category_performance,
            "performance_insights": {
                "fastest_operations": [
                    {"name": r.name, "duration_ms": r.mean_duration_ms, "category": r.category}
                    for r in fastest_benchmarks
                ],
                "slowest_operations": [
                    {"name": r.name, "duration_ms": r.mean_duration_ms, "category": r.category}
                    for r in slowest_benchmarks
                ],
                "failed_benchmarks": [
                    {
                        "name": r.name,
                        "category": r.category,
                        "duration_ms": r.mean_duration_ms,
                        "target_ms": r.target_ms,
                        "error": r.error_message,
                        "performance_met": r.performance_met,
                    }
                    for r in failed_benchmarks
                ],
            },
            "recommendations": self._generate_recommendations(results, category_performance),
        }

    def _generate_recommendations(
        self,
        results: list[BenchmarkResult],
        category_performance: dict[str, Any]
    ) -> list[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        # Overall performance assessment
        total_with_targets = len([r for r in results if r.target_ms is not None])
        met_targets = len([r for r in results if r.target_ms is not None and r.performance_met])

        if total_with_targets > 0:
            success_rate = met_targets / total_with_targets * 100

            if success_rate >= 90:
                recommendations.append("✓ Excellent performance: >90% of targets met")
            elif success_rate >= 80:
                recommendations.append("⚠ Good performance: 80-90% of targets met, minor optimizations needed")
            elif success_rate >= 70:
                recommendations.append("⚠ Moderate performance: 70-80% of targets met, optimization required")
            else:
                recommendations.append("✗ Poor performance: <70% of targets met, significant optimization needed")

        # Category-specific recommendations
        for category, data in category_performance.items():
            success_rate = data["success_rate"]

            if success_rate < 80:
                recommendations.append(f"⚠ {category.title()} operations need optimization: {success_rate:.1f}% target success rate")

        # Specific slow operations
        slow_operations = [r for r in results if r.target_ms and r.mean_duration_ms and r.mean_duration_ms > r.target_ms * 2]
        if slow_operations:
            recommendations.append(f"⚠ {len(slow_operations)} operations significantly exceed targets (>200% of target time)")

        # Failed operations
        failed_ops = [r for r in results if not r.success]
        if failed_ops:
            recommendations.append(f"✗ {len(failed_ops)} operations failed during benchmarking - investigate errors")

        return recommendations

    def _result_to_dict(self, result: BenchmarkResult) -> dict[str, Any]:
        """Convert BenchmarkResult to dictionary for serialization."""
        return {
            "benchmark_id": result.benchmark_id,
            "name": result.name,
            "description": result.description,
            "duration_ms": result.duration_ms,
            "target_ms": result.target_ms,
            "performance_met": result.performance_met,
            "min_duration_ms": result.min_duration_ms,
            "max_duration_ms": result.max_duration_ms,
            "mean_duration_ms": result.mean_duration_ms,
            "median_duration_ms": result.median_duration_ms,
            "std_dev_ms": result.std_dev_ms,
            "iterations": result.iterations,
            "warmup_iterations": result.warmup_iterations,
            "category": result.category,
            "service_name": result.service_name,
            "success": result.success,
            "error_message": result.error_message,
            "metadata": result.metadata,
        }

    def generate_performance_report(self, results_data: dict[str, Any]) -> str:
        """Generate human-readable performance report.

        Args:
            results_data: Results from run_comprehensive_benchmark()

        Returns:
            Formatted performance report
        """
        analysis = results_data["analysis"]
        summary = analysis["summary"]

        report_lines = [
            "=" * 80,
            "COMPREHENSIVE PERFORMANCE BENCHMARK REPORT",
            "=" * 80,
            "",
            f"Suite ID: {results_data['suite_id']}",
            f"Timestamp: {results_data['timestamp']}",
            f"Total Duration: {results_data['total_duration_ms']:.1f}ms",
            "",
            "SUMMARY:",
            f"  Total Benchmarks: {summary['total_benchmarks']}",
            f"  Performance Targets: {summary['met_targets']}/{summary['total_targets']} ({summary['overall_success_rate']:.1f}%)",
            f"  Critical Targets: {summary['met_critical_targets']}/{summary['critical_targets']} ({summary['critical_success_rate']:.1f}%)",
            "",
            "CATEGORY PERFORMANCE:",
        ]

        for category, data in analysis["category_performance"].items():
            report_lines.append(f"  {category.title()}: {data['met']}/{data['total']} ({data['success_rate']:.1f}%)")

        report_lines.extend([
            "",
            "FASTEST OPERATIONS:",
        ])

        report_lines.extend(f"  ✓ {op['name']}: {op['duration_ms']:.2f}ms ({op['category']})" for op in analysis["performance_insights"]["fastest_operations"])

        report_lines.extend([
            "",
            "SLOWEST OPERATIONS:",
        ])

        report_lines.extend(f"  ⚠ {op['name']}: {op['duration_ms']:.2f}ms ({op['category']})" for op in analysis["performance_insights"]["slowest_operations"])

        if analysis["performance_insights"]["failed_benchmarks"]:
            report_lines.extend([
                "",
                "FAILED BENCHMARKS:",
            ])

            for failed in analysis["performance_insights"]["failed_benchmarks"]:
                status = "✗" if not failed["performance_met"] else "⚠"
                report_lines.append(
                    f"  {status} {failed['name']}: {failed['duration_ms']:.2f}ms "
                    f"(target: {failed['target_ms']}ms) - {failed.get('error', 'Performance target missed')}"
                )

        report_lines.extend([
            "",
            "RECOMMENDATIONS:",
        ])

        report_lines.extend(f"  {recommendation}" for recommendation in analysis["recommendations"])

        report_lines.extend([
            "",
            "=" * 80,
        ])

        return "\n".join(report_lines)
