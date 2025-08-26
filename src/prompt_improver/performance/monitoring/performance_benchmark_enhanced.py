"""Enhanced Performance Benchmark with Service Locator - 2025 Architecture.

Enhanced version of MCPPerformanceBenchmark that uses service locator pattern
for dependency injection to eliminate circular imports while maintaining all
functionality.
"""

import json
import logging
import time
from datetime import UTC, datetime

# Heavy ML imports moved to TYPE_CHECKING and lazy loading
from typing import Any

import aiofiles


def _get_ml_event_modules():
    """Lazy load ML event modules when needed."""
    from prompt_improver.core.events.ml_event_bus import MLEvent, MLEventType
    return {'MLEventType': MLEventType, 'MLEvent': MLEvent}


from prompt_improver.performance.monitoring.performance_service_locator import (
    PerformanceServiceLocator,
)
from prompt_improver.shared.interfaces.protocols.monitoring import (
    ConfigurationServiceProtocol,
    DatabaseServiceProtocol,
    MLEventBusServiceProtocol,
    PromptImprovementServiceProtocol,
    SessionStoreServiceProtocol,
)

logger = logging.getLogger(__name__)


class MCPPerformanceBenchmarkEnhanced:
    """Enhanced performance benchmark with service locator pattern.

    This version eliminates circular dependencies by using the service locator
    pattern for dependency injection while maintaining all original functionality.

    Features:
    - Service locator pattern for clean dependency injection
    - Protocol-based interfaces for testability
    - No direct imports of DI containers or MCP server
    - Graceful fallbacks for missing dependencies
    - Full compatibility with existing performance monitoring
    """

    def __init__(self, service_locator: PerformanceServiceLocator) -> None:
        """Initialize with service locator for dependency injection.

        Args:
            service_locator: Configured service locator instance
        """
        self.service_locator = service_locator
        self._optimizer = None

        # Test data for benchmarking
        self.test_prompts = [
            "Write a Python function to calculate fibonacci numbers",
            "Explain the concept of machine learning in simple terms",
            "Create a REST API endpoint for user authentication",
            "Describe the benefits of using async programming",
            "Write a SQL query to find duplicate records",
            "Implement a binary search algorithm",
            "Explain the difference between HTTP and HTTPS",
            "Create a responsive CSS layout",
            "Write unit tests for a calculator function",
            "Describe the SOLID principles in software engineering",
        ]
        self.test_contexts = [
            {"domain": "programming", "language": "python"},
            {"domain": "education", "level": "beginner"},
            {"domain": "web_development", "framework": "fastapi"},
            {"domain": "data_science", "tool": "pandas"},
            {"domain": "devops", "platform": "aws"},
        ]

    @property
    async def optimizer(self):
        """Lazy load performance optimizer."""
        if self._optimizer is None:
            try:
                from prompt_improver.performance.optimization.performance_optimizer import (
                    get_performance_optimizer,
                )
                self._optimizer = get_performance_optimizer()
            except ImportError:
                # Use factory fallback
                from prompt_improver.performance.monitoring.performance_benchmark_factory import (
                    get_performance_optimizer,
                )
                self._optimizer = await get_performance_optimizer()
        return self._optimizer

    async def run_baseline_benchmark(
        self, samples_per_operation: int = 50
    ) -> dict[str, Any]:
        """Run comprehensive baseline benchmark for all operations.

        Args:
            samples_per_operation: Number of samples to collect per operation

        Returns:
            Dictionary of operation names to baseline measurements
        """
        logger.info(
            f"Starting baseline benchmark with {samples_per_operation} samples per operation"
        )
        baselines = {}

        # Benchmark prompt improvement
        baseline = await self._benchmark_improve_prompt(samples_per_operation)
        if baseline:
            baselines["improve_prompt"] = baseline

        # Benchmark database operations
        baseline = await self._benchmark_database_operations(samples_per_operation)
        if baseline:
            baselines["database_query"] = baseline

        # Benchmark analytics operations
        baseline = await self._benchmark_analytics_operations(samples_per_operation)
        if baseline:
            baselines["analytics_query"] = baseline

        # Benchmark session operations
        baseline = await self._benchmark_session_operations(samples_per_operation)
        if baseline:
            baselines["session_management"] = baseline

        await self.save_benchmark_results(baselines)
        return baselines

    async def _benchmark_improve_prompt(
        self, sample_count: int
    ) -> Any | None:
        """Benchmark the improve_prompt operation using service locator."""
        logger.info("Benchmarking improve_prompt operation")

        try:
            prompt_service = await self.service_locator.get_service(PromptImprovementServiceProtocol)
        except ValueError:
            logger.warning("Prompt improvement service not available, skipping benchmark")
            return None

        async def improve_prompt_operation() -> None:
            prompt_idx = len((await self.optimizer)._measurements.get("improve_prompt", [])) % len(self.test_prompts)
            context_idx = len((await self.optimizer)._measurements.get("improve_prompt", [])) % len(self.test_contexts)

            prompt = self.test_prompts[prompt_idx]
            context = self.test_contexts[context_idx]

            await prompt_service.improve_prompt(
                prompt=prompt,
                context=context,
                session_id=f"benchmark_{int(time.time())}",
                rate_limit_remaining=None,
            )

        optimizer = await self.optimizer
        return await optimizer.run_performance_benchmark(
            "improve_prompt", improve_prompt_operation, sample_count
        )

    async def _benchmark_database_operations(
        self, sample_count: int
    ) -> Any | None:
        """Benchmark database query operations using service locator."""
        logger.info("Benchmarking database operations")

        try:
            database_service = await self.service_locator.get_service(DatabaseServiceProtocol)
        except ValueError:
            logger.warning("Database service not available, skipping benchmark")
            return None

        async def database_operation() -> None:
            async with await database_service.get_session() as db_session:
                if db_session is not None:  # Check for no-op implementation
                    from sqlalchemy import text
                    result = await db_session.execute(text("SELECT 1"))
                    await result.fetchone()

        optimizer = await self.optimizer
        return await optimizer.run_performance_benchmark(
            "database_query", database_operation, sample_count
        )

    async def _benchmark_analytics_operations(
        self, sample_count: int
    ) -> Any | None:
        """Benchmark analytics query operations using service locator."""
        logger.info("Benchmarking analytics operations")

        try:
            event_bus_service = await self.service_locator.get_service(MLEventBusServiceProtocol)
        except ValueError:
            logger.warning("ML event bus service not available, skipping benchmark")
            return None

        async def analytics_operation() -> None:
            ml_modules = _get_ml_event_modules()
            MLEvent = ml_modules['MLEvent']
            MLEventType = ml_modules['MLEventType']

            performance_request = MLEvent(
                event_type=MLEventType.PERFORMANCE_METRICS_REQUEST,
                source="performance_benchmark",
                data={
                    "metric_type": "rule_effectiveness",
                    "days": 7,
                    "min_usage_count": 1
                }
            )
            await event_bus_service.publish(performance_request)

        optimizer = await self.optimizer
        return await optimizer.run_performance_benchmark(
            "analytics_query", analytics_operation, sample_count
        )

    async def _benchmark_session_operations(
        self, sample_count: int
    ) -> Any | None:
        """Benchmark session management operations using service locator."""
        logger.info("Benchmarking session operations")

        try:
            session_store = await self.service_locator.get_service(SessionStoreServiceProtocol)
        except ValueError:
            logger.warning("Session store service not available, skipping benchmark")
            return None

        async def session_operation() -> None:
            session_id = f"benchmark_{int(time.time() * 1000)}"
            await session_store.set_session(session_id, {"test": "data"})
            await session_store.get_session(session_id)
            await session_store.set_session(session_id, {"test": "data", "updated": True})
            await session_store.touch_session(session_id)
            await session_store.delete_session(session_id)

        optimizer = await self.optimizer
        return await optimizer.run_performance_benchmark(
            "session_management", session_operation, sample_count
        )

    async def save_benchmark_results(
        self,
        baselines: dict[str, Any],
        filepath: str = "performance_baseline.json",
    ) -> None:
        """Save benchmark results to file."""
        results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "target_ms": 200,
            "baselines": {
                name: baseline.model_dump() for name, baseline in baselines.items()
            },
            "summary": {
                "total_operations": len(baselines),
                "operations_meeting_target": sum(
                    1 for baseline in baselines.values() if baseline.meets_target(200)
                ),
                "overall_success": all(
                    baseline.meets_target(200) for baseline in baselines.values()
                ),
            },
        }
        async with aiofiles.open(filepath, "w") as f:
            await f.write(json.dumps(results, indent=2))
        logger.info(f"Saved benchmark results to {filepath}")

    async def compare_with_baseline(
        self, baseline_filepath: str = "performance_baseline.json"
    ) -> dict[str, Any]:
        """Compare current performance with saved baseline."""
        try:
            async with aiofiles.open(baseline_filepath) as f:
                content = await f.read()
                baseline_data = json.loads(content)
        except FileNotFoundError:
            logger.exception(f"Baseline file {baseline_filepath} not found")
            return {"error": "Baseline file not found"}

        optimizer = await self.optimizer
        current_baselines = await optimizer.get_all_baselines()
        comparison = {
            "timestamp": datetime.now(UTC).isoformat(),
            "baseline_timestamp": baseline_data["timestamp"],
            "comparisons": {},
        }

        for operation_name, baseline_info in baseline_data["baselines"].items():
            current_baseline = current_baselines.get(operation_name)
            if current_baseline:
                comparison["comparisons"][operation_name] = {
                    "baseline_avg_ms": baseline_info["avg_duration_ms"],
                    "current_avg_ms": current_baseline.avg_duration_ms,
                    "improvement_ms": baseline_info["avg_duration_ms"]
                    - current_baseline.avg_duration_ms,
                    "improvement_percent": (
                        baseline_info["avg_duration_ms"]
                        - current_baseline.avg_duration_ms
                    )
                    / baseline_info["avg_duration_ms"]
                    * 100,
                    "baseline_p95_ms": baseline_info["p95_duration_ms"],
                    "current_p95_ms": current_baseline.p95_duration_ms,
                    "baseline_meets_target": baseline_info["meets_200ms_target"],
                    "current_meets_target": current_baseline.meets_target(200),
                }
        return comparison

    def generate_performance_report(
        self, baselines: dict[str, Any]
    ) -> str:
        """Generate a human-readable performance report."""
        report_lines = [
            "=" * 80,
            "MCP PERFORMANCE BENCHMARK REPORT (Enhanced)",
            "=" * 80,
            f"Generated: {datetime.now(UTC).isoformat()}",
            "Target: <200ms response time",
            "",
            "OPERATION PERFORMANCE SUMMARY:",
            "-" * 40,
        ]

        for operation_name, baseline in baselines.items():
            status = "✅ PASS" if baseline.meets_target(200) else "❌ FAIL"
            report_lines.extend([
                f"Operation: {operation_name}",
                f"  Status: {status}",
                f"  Average: {baseline.avg_duration_ms:.2f}ms",
                f"  P95: {baseline.p95_duration_ms:.2f}ms",
                f"  P99: {baseline.p99_duration_ms:.2f}ms",
                f"  Success Rate: {baseline.success_rate:.1%}",
                f"  Samples: {baseline.sample_count}",
                "",
            ])

        total_operations = len(baselines)
        passing_operations = sum(1 for b in baselines.values() if b.meets_target(200))
        report_lines.extend([
            "OVERALL SUMMARY:",
            "-" * 20,
            f"Total Operations: {total_operations}",
            f"Passing Target: {passing_operations}/{total_operations}",
            f"Success Rate: {passing_operations / total_operations:.1%}"
            if total_operations > 0
            else "Success Rate: N/A",
            "",
            "Architecture: Service Locator Pattern (No Circular Dependencies)",
            "=" * 80,
        ])
        return "\n".join(report_lines)

    async def health_check(self) -> dict[str, Any]:
        """Check health of performance benchmark and its dependencies."""
        health = {
            "status": "healthy",
            "benchmark_type": "enhanced_service_locator",
            "dependencies": {},
            "timestamp": datetime.now(UTC).isoformat()
        }

        # Check service locator health
        try:
            available_services = [protocol.__name__ for protocol in [
                DatabaseServiceProtocol,
                PromptImprovementServiceProtocol,
                ConfigurationServiceProtocol,
                MLEventBusServiceProtocol,
                SessionStoreServiceProtocol,
            ] if self.service_locator.has_service(protocol)]

            health["dependencies"]["service_locator"] = {
                "status": "healthy",
                "available_services": available_services
            }
        except Exception as e:
            health["dependencies"]["service_locator"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "degraded"

        # Check optimizer health
        try:
            optimizer = await self.optimizer
            health["dependencies"]["performance_optimizer"] = {
                "status": "healthy",
                "type": type(optimizer).__name__
            }
        except Exception as e:
            health["dependencies"]["performance_optimizer"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "degraded"

        return health


# Convenience function for easy access
async def run_enhanced_performance_benchmark(
    service_locator: PerformanceServiceLocator,
    samples_per_operation: int = 50,
) -> dict[str, Any]:
    """Run enhanced performance benchmark with service locator.

    Args:
        service_locator: Configured service locator instance
        samples_per_operation: Number of samples to collect per operation

    Returns:
        Dictionary of performance baselines
    """
    benchmark = MCPPerformanceBenchmarkEnhanced(service_locator)
    return await benchmark.run_baseline_benchmark(samples_per_operation)
