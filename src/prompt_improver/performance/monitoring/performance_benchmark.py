"""Comprehensive performance benchmarking tool for MCP operations.

This module provides tools to measure current performance baseline and validate
optimization improvements against the <200ms target.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiofiles

from prompt_improver.database import get_session
from prompt_improver.core.services.analytics_factory import get_analytics_interface
from ..optimization.performance_optimizer import (
    get_performance_optimizer,
    PerformanceBaseline
)

logger = logging.getLogger(__name__)

class MCPPerformanceBenchmark:
    """Comprehensive benchmark suite for MCP operations."""

    def __init__(self):
        self.optimizer = get_performance_optimizer()
        self._prompt_service = None  # Lazy loaded to avoid circular imports
        self.analytics_service = get_analytics_interface()

    @property
    def prompt_service(self):
        """Lazy load PromptImprovementService to avoid circular imports."""
        if self._prompt_service is None:
            from prompt_improver.core.services.prompt_improvement import PromptImprovementService
            self._prompt_service = PromptImprovementService()
        return self._prompt_service

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
            "Describe the SOLID principles in software engineering"
        ]

        self.test_contexts = [
            {"domain": "programming", "language": "python"},
            {"domain": "education", "level": "beginner"},
            {"domain": "web_development", "framework": "fastapi"},
            {"domain": "data_science", "tool": "pandas"},
            {"domain": "devops", "platform": "aws"}
        ]

    async def run_baseline_benchmark(
        self,
        samples_per_operation: int = 50
    ) -> Dict[str, PerformanceBaseline]:
        """Run comprehensive baseline benchmark for all MCP operations.

        Args:
            samples_per_operation: Number of samples to collect per operation

        Returns:
            Dictionary of operation names to baseline measurements
        """
        logger.info(f"Starting baseline benchmark with {samples_per_operation} samples per operation")

        baselines = {}

        # Benchmark improve_prompt operation
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

        # Save baselines for future comparison
        await self.save_benchmark_results(baselines)

        return baselines

    async def _benchmark_improve_prompt(self, sample_count: int) -> Optional[PerformanceBaseline]:
        """Benchmark the improve_prompt operation."""
        logger.info("Benchmarking improve_prompt operation")

        async def improve_prompt_operation():
            prompt = self.test_prompts[len(self.optimizer._measurements.get("improve_prompt", [])) % len(self.test_prompts)]
            context = self.test_contexts[len(self.optimizer._measurements.get("improve_prompt", [])) % len(self.test_contexts)]

            async with get_session() as db_session:
                await self.prompt_service.improve_prompt(
                    prompt=prompt,
                    user_context=context,
                    session_id=f"benchmark_{int(time.time())}",
                    db_session=db_session
                )

        return await self.optimizer.run_performance_benchmark(
            "improve_prompt",
            improve_prompt_operation,
            sample_count
        )

    async def _benchmark_database_operations(self, sample_count: int) -> Optional[PerformanceBaseline]:
        """Benchmark database query operations."""
        logger.info("Benchmarking database operations")

        async def database_operation():
            async with get_session() as db_session:
                # Simulate typical database queries
                result = await db_session.execute("SELECT 1")
                await result.fetchone()

        return await self.optimizer.run_performance_benchmark(
            "database_query",
            database_operation,
            sample_count
        )

    async def _benchmark_analytics_operations(self, sample_count: int) -> Optional[PerformanceBaseline]:
        """Benchmark analytics query operations."""
        logger.info("Benchmarking analytics operations")

        async def analytics_operation():
            async with get_session() as db_session:
                await self.analytics_service.get_rule_effectiveness(
                    days=7,
                    min_usage_count=1,
                    db_session=db_session
                )

        return await self.optimizer.run_performance_benchmark(
            "analytics_query",
            analytics_operation,
            sample_count
        )

    async def _benchmark_session_operations(self, sample_count: int) -> Optional[PerformanceBaseline]:
        """Benchmark session management operations."""
        logger.info("Benchmarking session operations")

        from prompt_improver.utils.session_store import get_session_store

        async def session_operation():
            store = get_session_store()
            session_id = f"benchmark_{int(time.time() * 1000)}"

            # Create, update, and retrieve session
            await store.create_session(session_id, {"test": "data"})
            await store.get_session(session_id)
            await store.update_session(session_id, {"updated": True})
            await store.delete_session(session_id)

        return await self.optimizer.run_performance_benchmark(
            "session_management",
            session_operation,
            sample_count
        )

    async def save_benchmark_results(
        self,
        baselines: Dict[str, PerformanceBaseline],
        filepath: str = "performance_baseline.json"
    ) -> None:
        """Save benchmark results to file."""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "target_ms": 200,
            "baselines": {
                name: baseline.to_dict()
                for name, baseline in baselines.items()
            },
            "summary": {
                "total_operations": len(baselines),
                "operations_meeting_target": sum(
                    1 for baseline in baselines.values()
                    if baseline.meets_target(200)
                ),
                "overall_success": all(
                    baseline.meets_target(200)
                    for baseline in baselines.values()
                )
            }
        }

        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(results, indent=2))

        logger.info(f"Saved benchmark results to {filepath}")

    async def compare_with_baseline(
        self,
        baseline_filepath: str = "performance_baseline.json"
    ) -> Dict[str, Any]:
        """Compare current performance with saved baseline.

        Args:
            baseline_filepath: Path to baseline file

        Returns:
            Comparison results
        """
        try:
            async with aiofiles.open(baseline_filepath, 'r') as f:
                content = await f.read()
                baseline_data = json.loads(content)
        except FileNotFoundError:
            logger.error(f"Baseline file {baseline_filepath} not found")
            return {"error": "Baseline file not found"}

        current_baselines = await self.optimizer.get_all_baselines()
        comparison = {
            "timestamp": datetime.utcnow().isoformat(),
            "baseline_timestamp": baseline_data["timestamp"],
            "comparisons": {}
        }

        for operation_name, baseline_info in baseline_data["baselines"].items():
            current_baseline = current_baselines.get(operation_name)
            if current_baseline:
                comparison["comparisons"][operation_name] = {
                    "baseline_avg_ms": baseline_info["avg_duration_ms"],
                    "current_avg_ms": current_baseline.avg_duration_ms,
                    "improvement_ms": baseline_info["avg_duration_ms"] - current_baseline.avg_duration_ms,
                    "improvement_percent": (
                        (baseline_info["avg_duration_ms"] - current_baseline.avg_duration_ms)
                        / baseline_info["avg_duration_ms"] * 100
                    ),
                    "baseline_p95_ms": baseline_info["p95_duration_ms"],
                    "current_p95_ms": current_baseline.p95_duration_ms,
                    "baseline_meets_target": baseline_info["meets_200ms_target"],
                    "current_meets_target": current_baseline.meets_target(200)
                }

        return comparison

    def generate_performance_report(
        self,
        baselines: Dict[str, PerformanceBaseline]
    ) -> str:
        """Generate a human-readable performance report.

        Args:
            baselines: Dictionary of performance baselines

        Returns:
            Formatted performance report
        """
        report_lines = [
            "=" * 80,
            "MCP PERFORMANCE BENCHMARK REPORT",
            "=" * 80,
            f"Generated: {datetime.utcnow().isoformat()}",
            f"Target: <200ms response time",
            "",
            "OPERATION PERFORMANCE SUMMARY:",
            "-" * 40
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
                ""
            ])

        # Overall summary
        total_operations = len(baselines)
        passing_operations = sum(1 for b in baselines.values() if b.meets_target(200))

        report_lines.extend([
            "OVERALL SUMMARY:",
            "-" * 20,
            f"Total Operations: {total_operations}",
            f"Passing Target: {passing_operations}/{total_operations}",
            f"Success Rate: {passing_operations/total_operations:.1%}" if total_operations > 0 else "Success Rate: N/A",
            "",
            "=" * 80
        ])

        return "\n".join(report_lines)

# Convenience function for running benchmarks
async def run_mcp_performance_benchmark(samples_per_operation: int = 50) -> Dict[str, PerformanceBaseline]:
    """Run a complete MCP performance benchmark.

    Args:
        samples_per_operation: Number of samples to collect per operation

    Returns:
        Dictionary of performance baselines
    """
    benchmark = MCPPerformanceBenchmark()
    return await benchmark.run_baseline_benchmark(samples_per_operation)
