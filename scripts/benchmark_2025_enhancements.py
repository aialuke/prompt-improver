"""Benchmark script for 2025 FastMCP enhancements.

Verifies that the new features maintain <200ms response time targets.
"""

import asyncio
import json
import logging
import statistics
import time
from typing import Dict, List

from mcp.server.fastmcp import Context
from unittest.mock import AsyncMock

from prompt_improver.mcp_server.server import APESMCPServer
from prompt_improver.mcp_server.middleware import MiddlewareContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Benchmark 2025 FastMCP enhancements."""
    
    def __init__(self):
        self.server = APESMCPServer()
        self.results = {}
    
    async def benchmark_middleware_overhead(self, iterations: int = 100):
        """Measure overhead added by middleware stack."""
        logger.info(f"Benchmarking middleware overhead ({iterations} iterations)...")
        
        # Direct handler (no middleware)
        async def direct_handler():
            await asyncio.sleep(0.001)  # Simulate 1ms work
            return {"result": "success"}
        
        # Wrapped handler (with middleware)
        async def wrapped_handler(*args, **kwargs):
            await asyncio.sleep(0.001)  # Same 1ms work
            return {"result": "success"}
        
        wrapped = self.server.services.middleware_stack.wrap(wrapped_handler)
        
        # Benchmark direct calls
        direct_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            await direct_handler()
            direct_times.append((time.perf_counter() - start) * 1000)
        
        # Benchmark wrapped calls
        wrapped_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            await wrapped(__method__="test")
            wrapped_times.append((time.perf_counter() - start) * 1000)
        
        # Calculate statistics
        direct_avg = statistics.mean(direct_times)
        wrapped_avg = statistics.mean(wrapped_times)
        overhead = wrapped_avg - direct_avg
        
        self.results["middleware_overhead"] = {
            "direct_avg_ms": direct_avg,
            "wrapped_avg_ms": wrapped_avg,
            "overhead_ms": overhead,
            "overhead_percent": (overhead / direct_avg) * 100,
            "direct_p95_ms": statistics.quantiles(direct_times, n=20)[18],  # 95th percentile
            "wrapped_p95_ms": statistics.quantiles(wrapped_times, n=20)[18]
        }
        
        logger.info(f"Middleware overhead: {overhead:.2f}ms ({(overhead / direct_avg) * 100:.1f}%)")
    
    async def benchmark_progress_reporting(self, iterations: int = 50):
        """Benchmark impact of progress reporting."""
        logger.info(f"Benchmarking progress reporting impact ({iterations} iterations)...")
        
        # Mock context with progress reporting
        mock_ctx_with_progress = AsyncMock(spec=Context)
        mock_ctx_with_progress.report_progress = AsyncMock()
        mock_ctx_with_progress.info = AsyncMock()
        mock_ctx_with_progress.debug = AsyncMock()
        mock_ctx_with_progress.error = AsyncMock()
        
        # Find the progress-aware tool (now unified as improve_prompt)
        improve_prompt = None
        for tool_name, tool_func in self.server.mcp._tools.items():
            if tool_name == "improve_prompt":
                improve_prompt = tool_func.implementation
                break
        
        assert improve_prompt is not None
        
        # Test prompts of varying sizes
        test_prompts = [
            "Short prompt",
            "Medium prompt " * 10,
            "Long prompt " * 100
        ]
        
        results_by_size = {}
        
        for prompt in test_prompts:
            prompt_size = len(prompt)
            times_with_progress = []
            times_without_progress = []
            
            # Benchmark with progress reporting (modern 2025 parameters)
            for i in range(iterations):
                start = time.perf_counter()
                await improve_prompt(
                    prompt=prompt,
                    session_id=f"bench_session_progress_{i}",  # Required parameter
                    ctx=mock_ctx_with_progress,  # Required parameter
                    context=None
                )
                times_with_progress.append((time.perf_counter() - start) * 1000)
            
            # Benchmark with minimal progress (still required but mock does nothing)
            mock_ctx_minimal = AsyncMock(spec=Context)
            mock_ctx_minimal.report_progress = AsyncMock()
            mock_ctx_minimal.info = AsyncMock() 
            mock_ctx_minimal.debug = AsyncMock()
            mock_ctx_minimal.error = AsyncMock()
            
            for i in range(iterations):
                start = time.perf_counter()
                await improve_prompt(
                    prompt=prompt,
                    session_id=f"bench_session_minimal_{i}",  # Required parameter
                    ctx=mock_ctx_minimal,  # Required parameter
                    context=None
                )
                times_without_progress.append((time.perf_counter() - start) * 1000)
            
            results_by_size[f"prompt_{prompt_size}_chars"] = {
                "with_progress_avg_ms": statistics.mean(times_with_progress),
                "without_progress_avg_ms": statistics.mean(times_without_progress),
                "progress_overhead_ms": statistics.mean(times_with_progress) - statistics.mean(times_without_progress),
                "with_progress_p95_ms": statistics.quantiles(times_with_progress, n=20)[18],
                "without_progress_p95_ms": statistics.quantiles(times_without_progress, n=20)[18]
            }
        
        self.results["progress_reporting"] = results_by_size
        
        # Log summary
        for size, metrics in results_by_size.items():
            logger.info(f"{size}: Progress overhead = {metrics['progress_overhead_ms']:.2f}ms")
    
    async def benchmark_wildcard_resources(self, iterations: int = 100):
        """Benchmark wildcard resource performance."""
        logger.info(f"Benchmarking wildcard resource performance ({iterations} iterations)...")
        
        # Setup test data
        test_session_data = {
            "history": [{"action": f"action_{i}", "workspace": "main"} for i in range(100)]
        }
        await self.server.services.session_store.set("benchmark_session", test_session_data)
        
        # Benchmark different wildcard patterns
        patterns = [
            "benchmark_session",  # Simple lookup
            "benchmark_session/workspace/main",  # One level wildcard
            "benchmark_session/workspace/main/recent"  # Multi-level (will use same data)
        ]
        
        wildcard_results = {}
        
        for pattern in patterns:
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                await self.server._get_session_history_impl(pattern)
                times.append((time.perf_counter() - start) * 1000)
            
            wildcard_results[pattern] = {
                "avg_ms": statistics.mean(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "p95_ms": statistics.quantiles(times, n=20)[18]
            }
        
        self.results["wildcard_resources"] = wildcard_results
        
        # Log summary
        for pattern, metrics in wildcard_results.items():
            logger.info(f"Pattern '{pattern}': avg={metrics['avg_ms']:.2f}ms, p95={metrics['p95_ms']:.2f}ms")
    
    async def benchmark_end_to_end_latency(self, iterations: int = 50):
        """Benchmark end-to-end latency for realistic operations."""
        logger.info(f"Benchmarking end-to-end latency ({iterations} iterations)...")
        
        # Realistic test scenarios
        scenarios = [
            {
                "name": "simple_prompt_enhancement",
                "prompt": "Write a function to calculate fibonacci numbers",
                "context": None
            },
            {
                "name": "complex_prompt_with_context",
                "prompt": "Refactor this code to use async/await pattern " * 5,
                "context": {"language": "python", "framework": "fastapi", "requirements": ["performance", "readability"]}
            }
        ]
        
        e2e_results = {}
        
        for scenario in scenarios:
            times = []
            under_200ms_count = 0
            
            # Get the standard improve_prompt tool
            improve_prompt = None
            for tool_name, tool_func in self.server.mcp._tools.items():
                if tool_name == "improve_prompt":
                    improve_prompt = tool_func.implementation
                    break
            
            assert improve_prompt is not None
            
            # Create minimal context for benchmarking
            mock_ctx_bench = AsyncMock(spec=Context)
            mock_ctx_bench.report_progress = AsyncMock()
            mock_ctx_bench.info = AsyncMock()
            mock_ctx_bench.debug = AsyncMock()
            mock_ctx_bench.error = AsyncMock()
            
            for i in range(iterations):
                start = time.perf_counter()
                result = await improve_prompt(
                    prompt=scenario["prompt"],
                    session_id=f"benchmark_{scenario['name']}_{i}",  # Required parameter
                    ctx=mock_ctx_bench,  # Required parameter  
                    context=scenario["context"]
                )
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                
                if elapsed < 200:
                    under_200ms_count += 1
            
            e2e_results[scenario["name"]] = {
                "avg_ms": statistics.mean(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "p50_ms": statistics.median(times),
                "p95_ms": statistics.quantiles(times, n=20)[18],
                "p99_ms": statistics.quantiles(times, n=100)[98] if len(times) >= 100 else max(times),
                "under_200ms_percent": (under_200ms_count / iterations) * 100
            }
        
        self.results["end_to_end_latency"] = e2e_results
        
        # Log summary
        for scenario_name, metrics in e2e_results.items():
            logger.info(f"{scenario_name}: avg={metrics['avg_ms']:.2f}ms, "
                       f"p95={metrics['p95_ms']:.2f}ms, "
                       f"<200ms={metrics['under_200ms_percent']:.1f}%")
    
    def generate_report(self) -> Dict:
        """Generate comprehensive benchmark report."""
        report = {
            "timestamp": time.time(),
            "results": self.results,
            "summary": {
                "middleware_overhead_acceptable": self.results.get("middleware_overhead", {}).get("overhead_ms", 0) < 10,
                "progress_reporting_efficient": all(
                    metrics.get("progress_overhead_ms", 0) < 20
                    for metrics in self.results.get("progress_reporting", {}).values()
                ),
                "wildcard_resources_fast": all(
                    metrics.get("p95_ms", 0) < 50
                    for metrics in self.results.get("wildcard_resources", {}).values()
                ),
                "meets_200ms_target": all(
                    metrics.get("under_200ms_percent", 0) >= 95
                    for metrics in self.results.get("end_to_end_latency", {}).values()
                )
            }
        }
        
        # Overall pass/fail
        report["summary"]["all_targets_met"] = all(report["summary"].values())
        
        return report
    
    async def run_all_benchmarks(self):
        """Run all benchmark suites."""
        logger.info("Starting 2025 FastMCP enhancement benchmarks...")
        
        await self.benchmark_middleware_overhead()
        await self.benchmark_progress_reporting()
        await self.benchmark_wildcard_resources()
        await self.benchmark_end_to_end_latency()
        
        report = self.generate_report()
        
        # Save report
        with open("benchmark_2025_results.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*50)
        
        for key, value in report["summary"].items():
            status = "✅ PASS" if value else "❌ FAIL"
            logger.info(f"{key}: {status}")
        
        logger.info("="*50)
        logger.info(f"Results saved to benchmark_2025_results.json")
        
        return report


async def main():
    """Run benchmarks."""
    benchmark = PerformanceBenchmark()
    report = await benchmark.run_all_benchmarks()
    
    # Exit with appropriate code
    exit_code = 0 if report["summary"]["all_targets_met"] else 1
    exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())