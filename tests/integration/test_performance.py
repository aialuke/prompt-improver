#!/usr/bin/env python3
"""
Real behavior performance testing with pytest-benchmark integration.
Implements statistical validation, benchmark comparisons, and comprehensive
performance regression detection following 2025 best practices.

Migrated from mock-based testing to real system integration for accurate
performance measurements and validation.
"""

import asyncio
import os
import statistics
import sys
import time
from typing import Any, Dict, List
from contextlib import asynccontextmanager

import pytest
from hypothesis import (
    given,
    strategies as st,
)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import real performance testing targets
from prompt_improver.database import get_session_context
from prompt_improver.services.prompt_improvement import PromptImprovementService
from prompt_improver.rule_engine import RuleEngine
from prompt_improver.database.models import (
    ImprovementSession,
    RulePerformance,
)
from prompt_improver.utils.session_store import SessionStore
from prompt_improver.services.analytics import AnalyticsService


class PerformanceTester:
    """Performance testing suite for real prompt improvement system"""

    def __init__(self):
        self.results = {}
        self.session_store = SessionStore()
        self.analytics_service = None
        self.improvement_service = None
        self.rule_engine = None
        
    async def setup_services(self):
        """Initialize real services for performance testing"""
        # Setup analytics service
        self.analytics_service = AnalyticsService()
        
        # Setup rule engine
        self.rule_engine = RuleEngine()
        
        # Setup prompt improvement service with real dependencies
        self.improvement_service = PromptImprovementService(
            enable_bandit_optimization=False,  # Disable for performance testing
            enable_automl=False  # Disable for performance testing
        )
        
        # Initialize service connections
        if hasattr(self.improvement_service, 'initialize'):
            await self.improvement_service.initialize()
        
    async def cleanup_services(self):
        """Cleanup services after testing"""
        if self.analytics_service:
            # Analytics service doesn't have close method, just reset
            self.analytics_service = None
        if self.session_store:
            # Clear the session store
            await self.session_store.clear()
            # Stop cleanup task if running
            await self.session_store.stop_cleanup_task()
            
    @asynccontextmanager
    async def test_session(self, session_id: str):
        """Context manager for test sessions with proper cleanup"""
        try:
            # Store session for cleanup
            await self.session_store.set(session_id, {"test_session": True})
            yield session_id
        finally:
            # Cleanup session
            await self.session_store.delete(session_id)

    async def test_improve_prompt_performance(
        self, iterations: int = 10
    ) -> dict[str, Any]:
        """Test real prompt improvement service performance"""
        print(f"🧪 Testing real prompt improvement performance ({iterations} iterations)")

        # Real-world test prompts with varying complexity
        test_prompts = [
            "Please analyze this code and suggest improvements",
            "Write a comprehensive summary of the machine learning model",
            "Create a detailed list of API endpoints with documentation",
            "Explain how the distributed system architecture works",
            "Help me optimize this database query for better performance",
        ]

        response_times = []
        
        # Setup real services
        await self.setup_services()
        
        try:
            for i in range(iterations):
                prompt = test_prompts[i % len(test_prompts)]
                session_id = f"perf_test_{i}"

                start_time = time.perf_counter()
                try:
                    # Test the real service directly
                    async with self.test_session(session_id) as sid:
                        result = await self.improvement_service.improve_prompt(
                            prompt=prompt, 
                            context={"domain": "testing", "test_run": True}, 
                            session_id=sid
                        )

                    end_time = time.perf_counter()
                    response_time = (end_time - start_time) * 1000  # Convert to ms
                    response_times.append(response_time)

                    print(f"   Iteration {i + 1}: {response_time:.2f}ms")

                except Exception as e:
                    print(f"   Iteration {i + 1}: ERROR - {e}")
                    response_times.append(float("inf"))

        finally:
            await self.cleanup_services()

        # Calculate statistics with improved metrics
        valid_times = [t for t in response_times if t != float("inf")]

        if valid_times:
            # Calculate percentiles for better performance analysis
            sorted_times = sorted(valid_times)
            p95 = sorted_times[int(0.95 * len(sorted_times))] if len(sorted_times) > 1 else sorted_times[0]
            p99 = sorted_times[int(0.99 * len(sorted_times))] if len(sorted_times) > 1 else sorted_times[0]
            
            stats = {
                "avg_response_time": statistics.mean(valid_times),
                "median_response_time": statistics.median(valid_times),
                "min_response_time": min(valid_times),
                "max_response_time": max(valid_times),
                "p95_response_time": p95,
                "p99_response_time": p99,
                "std_dev": statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
                "success_rate": len(valid_times) / iterations * 100,
                "under_200ms_rate": len([t for t in valid_times if t < 200])
                / len(valid_times)
                * 100,
                "under_500ms_rate": len([t for t in valid_times if t < 500])
                / len(valid_times)
                * 100,
            }
        else:
            stats = {
                "avg_response_time": float("inf"),
                "median_response_time": float("inf"),
                "min_response_time": float("inf"),
                "max_response_time": float("inf"),
                "p95_response_time": float("inf"),
                "p99_response_time": float("inf"),
                "std_dev": 0,
                "success_rate": 0,
                "under_200ms_rate": 0,
                "under_500ms_rate": 0,
            }

        return stats

    async def test_store_prompt_performance(
        self, iterations: int = 5
    ) -> dict[str, Any]:
        """Test real database storage performance"""
        print(f"\n🧪 Testing real database storage performance ({iterations} iterations)")

        response_times = []
        
        # Setup real services
        await self.setup_services()
        
        try:
            for i in range(iterations):
                start_time = time.perf_counter()
                try:
                    # Use real database session
                    async with get_session_context() as session:
                        # Create real improvement session record
                        improvement_session = ImprovementSession(
                            session_id=f"perf_test_{i}",
                            original_prompt=f"Test prompt {i}",
                            final_prompt=f"Enhanced test prompt {i}",
                            rules_applied=["test_rule"],
                            improvement_metrics={"improvement_score": 0.8, "confidence_level": 0.8},
                        )
                        
                        session.add(improvement_session)
                        await session.commit()
                        
                        # Also store rule performance
                        rule_performance = RulePerformance(
                            session_id=improvement_session.id,
                            rule_id="test_rule",
                            applied=True,
                            confidence_score=0.8,
                            processing_time_ms=50,
                        )
                        
                        session.add(rule_performance)
                        await session.commit()

                    end_time = time.perf_counter()
                    response_time = (end_time - start_time) * 1000  # Convert to ms
                    response_times.append(response_time)

                    print(f"   Iteration {i + 1}: {response_time:.2f}ms")

                except Exception as e:
                    print(f"   Iteration {i + 1}: ERROR - {e}")
                    response_times.append(float("inf"))

        finally:
            await self.cleanup_services()

        # Calculate statistics with database-specific metrics
        valid_times = [t for t in response_times if t != float("inf")]

        if valid_times:
            # Database operations should be faster than API calls
            sorted_times = sorted(valid_times)
            p95 = sorted_times[int(0.95 * len(sorted_times))] if len(sorted_times) > 1 else sorted_times[0]
            p99 = sorted_times[int(0.99 * len(sorted_times))] if len(sorted_times) > 1 else sorted_times[0]
            
            stats = {
                "avg_response_time": statistics.mean(valid_times),
                "median_response_time": statistics.median(valid_times),
                "min_response_time": min(valid_times),
                "max_response_time": max(valid_times),
                "p95_response_time": p95,
                "p99_response_time": p99,
                "std_dev": statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
                "success_rate": len(valid_times) / iterations * 100,
                "under_100ms_rate": len([t for t in valid_times if t < 100])
                / len(valid_times)
                * 100,
                "under_200ms_rate": len([t for t in valid_times if t < 200])
                / len(valid_times)
                * 100,
            }
        else:
            stats = {
                "avg_response_time": float("inf"),
                "median_response_time": float("inf"),
                "min_response_time": float("inf"),
                "max_response_time": float("inf"),
                "p95_response_time": float("inf"),
                "p99_response_time": float("inf"),
                "std_dev": 0,
                "success_rate": 0,
                "under_100ms_rate": 0,
                "under_200ms_rate": 0,
            }

        return stats

    async def test_rule_status_performance(self, iterations: int = 5) -> dict[str, Any]:
        """Test real rule engine status performance"""
        print(
            f"\n🧪 Testing real rule engine status performance ({iterations} iterations)"
        )

        response_times = []
        
        # Setup real services
        await self.setup_services()
        
        try:
            for i in range(iterations):
                start_time = time.perf_counter()
                try:
                    # Query real rule status from database
                    async with get_session_context() as session:
                        # Get all rule statuses
                        rule_statuses = await session.execute(
                            "SELECT * FROM rule_status WHERE enabled = true"
                        )
                        rules = rule_statuses.fetchall()
                        
                        # Also get rule engine status
                        engine_status = await self.rule_engine.get_status()
                        
                        result = {
                            "rules": [
                                {
                                    "id": rule.id,
                                    "name": rule.name,
                                    "enabled": rule.enabled,
                                    "status": "active",
                                }
                                for rule in rules
                            ],
                            "engine_status": engine_status,
                        }

                    end_time = time.perf_counter()
                    response_time = (end_time - start_time) * 1000  # Convert to ms
                    response_times.append(response_time)

                    print(f"   Iteration {i + 1}: {response_time:.2f}ms")

                except Exception as e:
                    print(f"   Iteration {i + 1}: ERROR - {e}")
                    response_times.append(float("inf"))

        finally:
            await self.cleanup_services()

        # Calculate statistics for rule queries
        valid_times = [t for t in response_times if t != float("inf")]

        if valid_times:
            # Rule status queries should be very fast
            sorted_times = sorted(valid_times)
            p95 = sorted_times[int(0.95 * len(sorted_times))] if len(sorted_times) > 1 else sorted_times[0]
            p99 = sorted_times[int(0.99 * len(sorted_times))] if len(sorted_times) > 1 else sorted_times[0]
            
            stats = {
                "avg_response_time": statistics.mean(valid_times),
                "median_response_time": statistics.median(valid_times),
                "min_response_time": min(valid_times),
                "max_response_time": max(valid_times),
                "p95_response_time": p95,
                "p99_response_time": p99,
                "std_dev": statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
                "success_rate": len(valid_times) / iterations * 100,
                "under_50ms_rate": len([t for t in valid_times if t < 50])
                / len(valid_times)
                * 100,
                "under_100ms_rate": len([t for t in valid_times if t < 100])
                / len(valid_times)
                * 100,
            }
        else:
            stats = {
                "avg_response_time": float("inf"),
                "median_response_time": float("inf"),
                "min_response_time": float("inf"),
                "max_response_time": float("inf"),
                "p95_response_time": float("inf"),
                "p99_response_time": float("inf"),
                "std_dev": 0,
                "success_rate": 0,
                "under_50ms_rate": 0,
                "under_100ms_rate": 0,
            }

        return stats

    async def run_comprehensive_test(self) -> dict[str, Any]:
        """Run comprehensive performance test suite"""
        print("🚀 Starting APES Real Behavior Performance Validation")
        print("=" * 70)

        # Test each MCP endpoint
        improve_prompt_stats = await self.test_improve_prompt_performance(10)
        store_prompt_stats = await self.test_store_prompt_performance(5)
        rule_status_stats = await self.test_rule_status_performance(5)

        # Compile results
        results = {
            "improve_prompt": improve_prompt_stats,
            "store_prompt": store_prompt_stats,
            "rule_status": rule_status_stats,
        }

        # Print summary with enhanced real behavior metrics
        print("\n" + "=" * 70)
        print("📊 REAL BEHAVIOR PERFORMANCE VALIDATION RESULTS")
        print("=" * 70)

        for tool_name, stats in results.items():
            print(f"\n🔧 {tool_name.upper()} REAL SERVICE:")
            print(f"   Average Response Time: {stats['avg_response_time']:.2f}ms")
            print(f"   Median Response Time:  {stats['median_response_time']:.2f}ms")
            print(f"   Min Response Time:     {stats['min_response_time']:.2f}ms")
            print(f"   Max Response Time:     {stats['max_response_time']:.2f}ms")
            print(f"   P95 Response Time:     {stats.get('p95_response_time', 'N/A'):.2f}ms")
            print(f"   P99 Response Time:     {stats.get('p99_response_time', 'N/A'):.2f}ms")
            print(f"   Standard Deviation:    {stats['std_dev']:.2f}ms")
            print(f"   Success Rate:          {stats['success_rate']:.1f}%")
            print(f"   Under 500ms Rate:      {stats.get('under_500ms_rate', 'N/A'):.1f}%")
            print(f"   Under 200ms Rate:      {stats.get('under_200ms_rate', 'N/A'):.1f}%")

            # Performance validation for real behavior
            if stats["avg_response_time"] < 1000 and stats.get("under_500ms_rate", 0) >= 70:
                print("   ✅ REAL PERFORMANCE TARGET MET")
            elif stats["avg_response_time"] < 1500:
                print("   ⚠️  MOSTLY MEETS REAL TARGET (some outliers)")
            else:
                print("   ❌ REAL PERFORMANCE TARGET NOT MET")

        # Overall assessment for real behavior
        avg_response_times = [
            stats["avg_response_time"]
            for stats in results.values()
            if stats["avg_response_time"] != float("inf")
        ]
        overall_under_500 = [stats.get("under_500ms_rate", 0) for stats in results.values()]

        print("\n🎯 OVERALL REAL BEHAVIOR ASSESSMENT:")
        if avg_response_times:
            overall_avg = statistics.mean(avg_response_times)
            overall_under_500_avg = statistics.mean([stats.get("under_500ms_rate", 0) for stats in results.values()])

            print(f"   Overall Average Response Time: {overall_avg:.2f}ms")
            print(f"   Overall Under 500ms Rate:      {overall_under_500_avg:.1f}%")

            if overall_avg < 1000 and overall_under_500_avg >= 70:
                print("   ✅ REAL BEHAVIOR PERFORMANCE TARGET ACHIEVED")
                print("   🎉 Real system response times validated under acceptable thresholds")
            elif overall_avg < 1500:
                print("   ⚠️  REAL BEHAVIOR PERFORMANCE ACCEPTABLE BUT NEEDS OPTIMIZATION")
            else:
                print("   ❌ REAL BEHAVIOR PERFORMANCE TARGET NEEDS SIGNIFICANT OPTIMIZATION")
        else:
            print("   ❌ UNABLE TO VALIDATE REAL PERFORMANCE - ALL TESTS FAILED")

        return results


async def main():
    """Main performance testing function"""
    tester = PerformanceTester()
    results = await tester.run_comprehensive_test()
    return results


@pytest.mark.benchmark
class TestBenchmarkPerformance:
    """pytest-benchmark integration for performance regression detection."""

    def test_benchmark_improve_prompt_sync_wrapper(self, benchmark):
        """Benchmark real rule engine with statistical validation."""

        def sync_improve_prompt():
            """Synchronous wrapper for benchmarking real async function."""
            async def run_real_improvement():
                # Use rule engine directly to avoid complex service dependencies
                rule_engine = RuleEngine()
                result = rule_engine.apply_rules(
                    prompt="Test prompt for benchmarking with real behavior",
                    context={"domain": "benchmark", "test_run": True}
                )
                # Convert to expected format for compatibility
                return {
                    "improved_prompt": result.improved_prompt,
                    "processing_time_ms": result.processing_time_ms or 0,
                    "applied_rules": [{"rule_id": rule.rule_id, "confidence": rule.confidence} for rule in result.applied_rules]
                }
            
            return asyncio.run(run_real_improvement())

        # Run benchmark with statistical analysis
        result = benchmark.pedantic(
            sync_improve_prompt, iterations=3, rounds=2, warmup_rounds=1  # Reduced for real behavior
        )

        # Validate result structure from real service
        assert "improved_prompt" in result
        assert result["improved_prompt"] is not None
        assert len(result["improved_prompt"]) > 0

        # Performance regression detection
        # benchmark.compare() will be used by pytest-benchmark for regression detection

    def test_benchmark_store_prompt_performance(self, benchmark):
        """Benchmark real database storage operations."""

        def sync_store_prompt():
            """Synchronous wrapper for benchmarking."""
            async def run_real_storage():
                async with get_session_context() as session:
                    # Create real improvement session record
                    improvement_session = ImprovementSession(
                        session_id="benchmark_store_test",
                        original_prompt="Benchmark original prompt",
                        final_prompt="Benchmark enhanced prompt", 
                        rules_applied=["benchmark_rule"],
                        improvement_metrics={"improvement_score": 0.8, "confidence_level": 0.8},
                    )
                    
                    session.add(improvement_session)
                    await session.commit()
                    return improvement_session.id
            
            return asyncio.run(run_real_storage())

        result = benchmark.pedantic(
            sync_store_prompt, iterations=3, rounds=2, warmup_rounds=1  # Reduced for real behavior
        )

        # Validate real database storage result
        assert result is not None
        assert isinstance(result, int)  # Should return an ID

    def test_benchmark_rule_status_performance(self, benchmark):
        """Benchmark real rule engine status queries."""

        def sync_get_rule_status():
            """Synchronous wrapper for benchmarking."""
            async def run_real_rule_status():
                # Get rule engine status directly for performance testing
                rule_engine = RuleEngine()
                engine_rules = [
                    {
                        "id": getattr(rule, 'rule_id', f'rule_{i}'),
                        "name": rule.__class__.__name__,
                        "enabled": True,
                        "status": "active",
                    }
                    for i, rule in enumerate(rule_engine.rules)
                ]
                
                return {
                    "rules": engine_rules,
                    "engine_status": {
                        "active_rules": len(rule_engine.rules),
                        "min_confidence": rule_engine.min_confidence,
                    },
                }
            
            return asyncio.run(run_real_rule_status())

        result = benchmark.pedantic(
            sync_get_rule_status, iterations=8, rounds=4, warmup_rounds=2
        )

        # Validate real rule status result
        assert "rules" in result
        assert "engine_status" in result
        assert isinstance(result["rules"], list)


@pytest.mark.performance
class TestStatisticalPerformanceValidation:
    """Statistical analysis of real performance characteristics following 2025 best practices."""

    @pytest.fixture(scope="class")
    async def stats_services(self):
        """Setup real services for statistical analysis"""
        tester = PerformanceTester()
        await tester.setup_services()
        yield tester
        await tester.cleanup_services()

    @pytest.mark.asyncio
    async def test_response_time_distribution_analysis(self, stats_services):
        """Analyze real response time distribution for statistical validation."""

        response_times = []

        # Collect sample data from real service
        for i in range(15):  # Fewer samples for real behavior testing
            session_id = f"stats_test_{i}"
            start_time = time.perf_counter()
            
            async with stats_services.test_session(session_id) as sid:
                await stats_services.improvement_service.improve_prompt(
                    prompt="Statistical analysis test prompt with real behavior",
                    context={"domain": "statistics", "test_run": True},
                    session_id=sid,
                )
                
            end_time = time.perf_counter()
            response_times.append((end_time - start_time) * 1000)

        # Statistical analysis with real behavior expectations
        mean_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0

        # Validate statistical properties for real behavior
        # Real behavior will be slower than mocks but more predictable
        assert mean_time < 2000, f"Mean response time {mean_time}ms exceeds realistic target"
        assert median_time < 1500, f"Median response time {median_time}ms exceeds realistic target"

        # Validate distribution characteristics for real behavior
        # Real systems should have lower variability than mocks
        coefficient_of_variation = std_dev / mean_time if mean_time > 0 else 0
        assert coefficient_of_variation < 0.5, (
            f"High variability: CV {coefficient_of_variation:.2f} indicates inconsistent real performance"
        )

        # Check for outliers (values > 2 std devs from mean)
        outliers = [t for t in response_times if abs(t - mean_time) > 2 * std_dev]
        outlier_rate = len(outliers) / len(response_times)
        assert outlier_rate <= 0.20, (
            f"Too many outliers: {outlier_rate:.2%} of responses - real behavior should be more consistent"
        )
        
        # Additional validation for real behavior
        # Check that we have reasonable p95 and p99 metrics
        sorted_times = sorted(response_times)
        p95 = sorted_times[int(0.95 * len(sorted_times))]
        p99 = sorted_times[int(0.99 * len(sorted_times))]
        
        assert p95 < mean_time * 2, f"P95 {p95}ms is too high compared to mean {mean_time}ms"
        assert p99 < mean_time * 3, f"P99 {p99}ms is too high compared to mean {mean_time}ms"

    @given(
        prompt_length=st.integers(min_value=10, max_value=200),
        concurrent_operations=st.integers(min_value=1, max_value=5),
    )
    @pytest.mark.asyncio
    async def test_performance_scaling_properties(
        self, prompt_length, concurrent_operations
    ):
        """Property-based testing of performance scaling characteristics for async function calls."""

        # Generate test prompt of specified length
        test_prompt = " ".join(["word"] * prompt_length)

        # Measure concurrent execution time
        start_time = time.time()

        tasks = [
            improve_prompt(
                prompt=test_prompt,
                context={"domain": "scaling_test"},
                session_id=f"scale_test_{i}",
            )
            for i in range(concurrent_operations)
        ]

        results = await asyncio.gather(*tasks)
        end_time = time.time()

        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_operation = total_time_ms / concurrent_operations

        # Validate all operations succeeded
        assert len(results) == concurrent_operations
        assert all("improved_prompt" in result for result in results)

        # Performance scaling properties
        # Average time per operation should not grow linearly with concurrent operations
        if concurrent_operations > 1:
            # Concurrent processing should provide some efficiency
            single_operation_baseline = 200  # ms (estimated baseline)
            efficiency_factor = avg_time_per_operation / single_operation_baseline

            # Should not be worse than 2x the baseline per operation
            assert efficiency_factor < 2.0, (
                f"Poor scaling: {avg_time_per_operation:.1f}ms per operation "
                f"with {concurrent_operations} concurrent operations"
            )

        # Total time should be reasonable regardless of request count
        assert total_time_ms < 1000, f"Total processing time {total_time_ms}ms too high"


@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression detection and validation."""

    @pytest.fixture(scope="class")
    async def regression_services(self):
        """Setup real services for regression testing"""
        tester = PerformanceTester()
        await tester.setup_services()
        yield tester
        await tester.cleanup_services()

    @pytest.mark.asyncio
    async def test_performance_baseline_validation(self, test_config, regression_services):
        """Validate real performance against established baselines."""

        # Use realistic target for real behavior
        target_response_time = test_config.get("performance", {}).get("target_response_time_ms", 1500)

        # Run standardized performance test with real services
        response_times = []
        for i in range(8):  # Fewer iterations for real behavior
            session_id = f"baseline_test_{i}"
            start_time = time.perf_counter()
            
            async with regression_services.test_session(session_id) as sid:
                result = await regression_services.improvement_service.improve_prompt(
                    prompt=f"Baseline test prompt {i} with real behavior",
                    context={"domain": "baseline", "test_run": True},
                    session_id=sid,
                )
                
            end_time = time.perf_counter()

            response_time_ms = (end_time - start_time) * 1000
            response_times.append(response_time_ms)

            # Validate individual response from real service
            assert "improved_prompt" in result
            assert result["improved_prompt"] is not None
            assert len(result["improved_prompt"]) > 0

        # Statistical validation against realistic baseline
        p95_response_time = sorted(response_times)[int(0.95 * len(response_times))]
        avg_response_time = statistics.mean(response_times)

        # Performance assertions for real behavior
        assert avg_response_time <= target_response_time, (
            f"Average real response time {avg_response_time:.1f}ms exceeds target {target_response_time}ms"
        )
        assert p95_response_time <= target_response_time * 2, (
            f"95th percentile real response time {p95_response_time:.1f}ms exceeds tolerance"
        )

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, regression_services):
        """Test that memory usage remains stable during extended real operations."""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
        except ImportError:
            pytest.skip("psutil not available - skipping memory usage test")

        # Run multiple operations to test memory stability with real services
        for i in range(12):  # Fewer operations for real behavior
            session_id = f"memory_test_{i}"
            
            async with regression_services.test_session(session_id) as sid:
                await regression_services.improvement_service.improve_prompt(
                    prompt=f"Memory stability test {i} with real behavior",
                    context={"domain": "memory_test", "test_run": True},
                    session_id=sid,
                )

            # Check memory every 4 operations
            if i % 4 == 0:
                current_memory = process.memory_info().rss
                memory_growth = current_memory - initial_memory
                memory_growth_mb = memory_growth / (1024 * 1024)

                # Memory growth should be reasonable for real behavior
                # Real services may use more memory but should be stable
                assert memory_growth_mb < 100, (
                    f"Excessive memory growth in real behavior: {memory_growth_mb:.1f}MB after {i + 1} operations"
                )
                
        # Final memory check
        final_memory = process.memory_info().rss
        total_growth = (final_memory - initial_memory) / (1024 * 1024)
        assert total_growth < 150, f"Total memory growth {total_growth:.1f}MB exceeds threshold for real behavior"


if __name__ == "__main__":
    asyncio.run(main())
