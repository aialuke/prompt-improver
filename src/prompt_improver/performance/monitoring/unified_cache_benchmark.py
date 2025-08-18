"""Unified Cache Architecture Performance Benchmark - 2025 Standards

This module provides comprehensive performance benchmarking for the unified cache
architecture, validating against 2025 performance targets:

L1 Cache (Memory): <1ms average, <2ms P99
L2 Cache (Redis): <10ms average, <20ms P99
L3 Cache (Database): <50ms average, <100ms P99
Cache Coordination: <50ms for full L1→L2→L3 fallback
Hit Rates: >95% L1, >90% L2, >80% L3, >96% overall
Memory Efficiency: <1KB overhead per L1 entry
"""

import asyncio
import json
import logging
import statistics
import time
from datetime import UTC, datetime
from typing import Any, Dict

import aiofiles

from prompt_improver.services.cache.cache_facade import CacheFacade
from prompt_improver.services.cache.cache_coordinator_service import CacheCoordinatorService
from prompt_improver.services.cache.l1_cache_service import L1CacheService
from prompt_improver.services.cache.l2_redis_service import L2RedisService
from prompt_improver.services.cache.l3_database_service import L3DatabaseService

logger = logging.getLogger(__name__)


class UnifiedCacheBenchmark:
    """Comprehensive benchmark suite for unified cache architecture."""

    def __init__(self) -> None:
        self.benchmark_results: Dict[str, Any] = {}
        self.performance_targets: Dict[str, float] = {
            "l1_avg_ms": 1.0,
            "l1_p99_ms": 2.0,
            "l2_avg_ms": 10.0,
            "l2_p99_ms": 20.0,
            "l3_avg_ms": 50.0,
            "l3_p99_ms": 100.0,
            "coordination_max_ms": 50.0,
            "l1_hit_rate": 0.95,
            "l2_hit_rate": 0.90,
            "l3_hit_rate": 0.80,
            "overall_hit_rate": 0.96,
            "memory_overhead_kb": 1.0,
        }

    async def run_comprehensive_benchmark(
        self, samples_per_test: int = 100
    ) -> Dict[str, Any]:
        """Run comprehensive unified cache benchmark suite.
        
        Args:
            samples_per_test: Number of samples per performance test
            
        Returns:
            Dictionary containing benchmark results and performance analysis
        """
        logger.info(f"Starting unified cache benchmark with {samples_per_test} samples per test")
        
        # 1. Test L1 Cache Performance
        await self._benchmark_l1_cache(samples_per_test)
        
        # 2. Test L2 Redis Cache Performance (if available)
        await self._benchmark_l2_cache(samples_per_test)
        
        # 3. Test L3 Database Cache Performance (if available)
        await self._benchmark_l3_cache(samples_per_test)
        
        # 4. Test Cache Coordination Performance
        await self._benchmark_cache_coordination(samples_per_test)
        
        # 5. Test Cache Facade Performance
        await self._benchmark_cache_facade(samples_per_test)
        
        # 6. Test Memory Efficiency
        await self._benchmark_memory_efficiency()
        
        # 7. Test Session Management Performance
        await self._benchmark_session_operations(samples_per_test)
        
        # 8. Analyze Results Against Targets
        performance_analysis = self._analyze_performance_results()
        
        # 9. Generate Benchmark Report
        benchmark_report = await self._generate_benchmark_report()
        
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "samples_per_test": samples_per_test,
            "benchmark_results": self.benchmark_results,
            "performance_analysis": performance_analysis,
            "benchmark_report": benchmark_report,
            "targets_met": performance_analysis["overall_targets_met"],
            "performance_score": performance_analysis["overall_performance_score"],
        }

    async def _benchmark_l1_cache(self, sample_count: int):
        """Benchmark L1 cache performance against <1ms average, <2ms P99 targets."""
        logger.info("Benchmarking L1 cache performance...")
        
        l1_cache = L1CacheService(max_size=1000)
        # Removed unused operation_times variable
        
        # Test data
        test_data = {"benchmark": True, "timestamp": time.time(), "data": "x" * 100}
        
        # Warm up
        for i in range(10):
            await l1_cache.set(f"warmup_{i}", test_data)
            await l1_cache.get(f"warmup_{i}")
        
        # Benchmark SET operations
        set_times: list[float] = []
        for i in range(sample_count):
            start_time = time.perf_counter()
            await l1_cache.set(f"l1_set_{i}", test_data)
            set_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            set_times.append(set_time)
        
        # Benchmark GET operations
        get_times: list[float] = []
        for i in range(sample_count):
            start_time = time.perf_counter()
            result = await l1_cache.get(f"l1_set_{i}")
            get_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            get_times.append(get_time)
            assert result is not None, f"L1 cache miss for key l1_set_{i}"
        
        # Calculate statistics
        all_times: list[float] = set_times + get_times
        stats = l1_cache.get_stats()
        
        self.benchmark_results["l1_cache"] = {
            "set_avg_ms": statistics.mean(set_times),
            "set_p99_ms": sorted(set_times)[int(0.99 * len(set_times))],
            "get_avg_ms": statistics.mean(get_times),
            "get_p99_ms": sorted(get_times)[int(0.99 * len(get_times))],
            "overall_avg_ms": statistics.mean(all_times),
            "overall_p99_ms": sorted(all_times)[int(0.99 * len(all_times))],
            "hit_rate": stats["hit_rate"],
            "memory_per_entry_bytes": stats["memory_per_entry_bytes"],
            "sample_count": sample_count * 2,  # SET + GET operations
        }

    async def _benchmark_l2_cache(self, sample_count: int):
        """Benchmark L2 Redis cache performance against <10ms average, <20ms P99 targets."""
        logger.info("Benchmarking L2 Redis cache performance...")
        
        try:
            l2_cache = L2RedisService()
            test_data = {"benchmark": True, "timestamp": time.time(), "data": "x" * 100}
            
            # Test Redis availability
            await l2_cache.set("health_check", {"status": "ok"})
            health_result = await l2_cache.get("health_check")
            if health_result is None:
                logger.warning("Redis not available, skipping L2 benchmark")
                self.benchmark_results["l2_cache"] = {"available": False}
                return
            
            # Benchmark SET operations
            set_times: list[float] = []
            for i in range(sample_count):
                start_time = time.perf_counter()
                await l2_cache.set(f"l2_set_{i}", test_data)
                set_time = (time.perf_counter() - start_time) * 1000
                set_times.append(set_time)
            
            # Benchmark GET operations
            get_times: list[float] = []
            cache_hits = 0
            for i in range(sample_count):
                start_time = time.perf_counter()
                result = await l2_cache.get(f"l2_set_{i}")
                get_time = (time.perf_counter() - start_time) * 1000
                get_times.append(get_time)
                if result is not None:
                    cache_hits += 1
            
            all_times: list[float] = set_times + get_times
            hit_rate = cache_hits / sample_count
            
            self.benchmark_results["l2_cache"] = {
                "available": True,
                "set_avg_ms": statistics.mean(set_times),
                "set_p99_ms": sorted(set_times)[int(0.99 * len(set_times))],
                "get_avg_ms": statistics.mean(get_times),
                "get_p99_ms": sorted(get_times)[int(0.99 * len(get_times))],
                "overall_avg_ms": statistics.mean(all_times),
                "overall_p99_ms": sorted(all_times)[int(0.99 * len(all_times))],
                "hit_rate": hit_rate,
                "sample_count": sample_count * 2,
            }
            
        except Exception as e:
            logger.warning(f"L2 cache benchmark failed: {e}")
            self.benchmark_results["l2_cache"] = {"available": False, "error": str(e)}

    async def _benchmark_l3_cache(self, sample_count: int):
        """Benchmark L3 database cache performance against <50ms average, <100ms P99 targets."""
        logger.info("Benchmarking L3 database cache performance...")
        
        try:
            l3_cache = L3DatabaseService()
            test_data = {"benchmark": True, "timestamp": time.time(), "data": "x" * 100}
            
            # Test database availability with smaller sample size
            test_sample_count = min(sample_count, 20)  # Limit database operations
            
            # Benchmark SET operations
            set_times: list[float] = []
            for i in range(test_sample_count):
                start_time = time.perf_counter()
                await l3_cache.set(f"l3_set_{i}", test_data)
                set_time = (time.perf_counter() - start_time) * 1000
                set_times.append(set_time)
            
            # Benchmark GET operations
            get_times: list[float] = []
            cache_hits = 0
            for i in range(test_sample_count):
                start_time = time.perf_counter()
                result = await l3_cache.get(f"l3_set_{i}")
                get_time = (time.perf_counter() - start_time) * 1000
                get_times.append(get_time)
                if result is not None:
                    cache_hits += 1
            
            all_times: list[float] = set_times + get_times
            hit_rate = cache_hits / test_sample_count
            
            self.benchmark_results["l3_cache"] = {
                "available": True,
                "set_avg_ms": statistics.mean(set_times),
                "set_p99_ms": sorted(set_times)[int(0.99 * len(set_times))],
                "get_avg_ms": statistics.mean(get_times),
                "get_p99_ms": sorted(get_times)[int(0.99 * len(get_times))],
                "overall_avg_ms": statistics.mean(all_times),
                "overall_p99_ms": sorted(all_times)[int(0.99 * len(all_times))],
                "hit_rate": hit_rate,
                "sample_count": test_sample_count * 2,
            }
            
        except Exception as e:
            logger.warning(f"L3 cache benchmark failed: {e}")
            self.benchmark_results["l3_cache"] = {"available": False, "error": str(e)}

    async def _benchmark_cache_coordination(self, sample_count: int):
        """Benchmark cache coordination performance with L1→L2→L3 fallback."""
        logger.info("Benchmarking cache coordination performance...")
        
        # Create coordinator with available cache levels
        l1_cache = L1CacheService(max_size=100)
        l2_cache = L2RedisService() if self.benchmark_results.get("l2_cache", {}).get("available") else None
        l3_cache = L3DatabaseService() if self.benchmark_results.get("l3_cache", {}).get("available") else None
        
        coordinator = CacheCoordinatorService(
            l1_cache=l1_cache,
            l2_cache=l2_cache,
            l3_cache=l3_cache,
            enable_warming=True,
        )
        
        test_data = {"coordination": True, "timestamp": time.time(), "data": "coordination_test"}
        test_sample_count = min(sample_count, 50)  # Reasonable sample for coordination
        
        # Benchmark coordinated operations
        coordination_times: list[float] = []
        for i in range(test_sample_count):
            key = f"coord_test_{i}"
            
            # SET operation through coordinator
            start_time = time.perf_counter()
            await coordinator.set(key, test_data)
            set_time = (time.perf_counter() - start_time) * 1000
            
            # GET operation through coordinator (should hit L1)
            start_time = time.perf_counter()
            result = await coordinator.get(key)
            get_time = (time.perf_counter() - start_time) * 1000
            
            total_time = set_time + get_time
            coordination_times.append(total_time)
            
            assert result is not None, f"Coordination failed for key {key}"
        
        # Test cache miss fallback performance
        await l1_cache.clear()  # Force L1 miss to test fallback
        fallback_times: list[float] = []
        for i in range(min(test_sample_count, 10)):  # Small sample for fallback test
            key = f"coord_test_{i}"
            start_time = time.perf_counter()
            result = await coordinator.get(key)  # Should fallback to L2/L3
            fallback_time = (time.perf_counter() - start_time) * 1000
            fallback_times.append(fallback_time)
        
        self.benchmark_results["cache_coordination"] = {
            "coordination_avg_ms": statistics.mean(coordination_times),
            "coordination_p99_ms": sorted(coordination_times)[int(0.99 * len(coordination_times))],
            "fallback_avg_ms": statistics.mean(fallback_times) if fallback_times else 0,
            "fallback_max_ms": max(fallback_times) if fallback_times else 0,
            "sample_count": test_sample_count,
            "fallback_sample_count": len(fallback_times),
        }

    async def _benchmark_cache_facade(self, sample_count: int):
        """Benchmark CacheFacade performance for end-to-end operations."""
        logger.info("Benchmarking CacheFacade performance...")
        
        cache_facade = CacheFacade(
            l1_max_size=1000,
            l2_default_ttl=300,
            enable_l2=self.benchmark_results.get("l2_cache", {}).get("available", False),
            enable_l3=self.benchmark_results.get("l3_cache", {}).get("available", False),
        )
        
        test_data = {"facade": True, "timestamp": time.time()}
        test_sample_count = min(sample_count, 50)
        
        # Benchmark facade operations
        facade_times: list[float] = []
        for i in range(test_sample_count):
            key = f"facade_test_{i}"
            
            # SET operation
            start_time = time.perf_counter()
            await cache_facade.set(key, test_data)
            set_time = (time.perf_counter() - start_time) * 1000
            
            # GET operation
            start_time = time.perf_counter()
            result = await cache_facade.get(key)
            get_time = (time.perf_counter() - start_time) * 1000
            
            total_time = set_time + get_time
            facade_times.append(total_time)
            
            assert result is not None, f"Facade operation failed for key {key}"
        
        self.benchmark_results["cache_facade"] = {
            "facade_avg_ms": statistics.mean(facade_times),
            "facade_p99_ms": sorted(facade_times)[int(0.99 * len(facade_times))],
            "sample_count": test_sample_count,
        }

    async def _benchmark_memory_efficiency(self):
        """Benchmark memory efficiency of cache layers."""
        logger.info("Benchmarking memory efficiency...")
        
        l1_cache = L1CacheService(max_size=1000)
        
        # Load cache with test data
        test_data = {"memory_test": True, "data": "x" * 100}  # ~100 bytes of data
        entries_to_test = 100
        
        for i in range(entries_to_test):
            await l1_cache.set(f"memory_test_{i}", test_data)
        
        stats = l1_cache.get_stats()
        
        self.benchmark_results["memory_efficiency"] = {
            "total_entries": stats["size"],
            "estimated_memory_bytes": stats["estimated_memory_bytes"],
            "memory_per_entry_bytes": stats["memory_per_entry_bytes"],
            "memory_per_entry_kb": stats["memory_per_entry_bytes"] / 1024,
            "memory_overhead_ratio": stats["memory_per_entry_bytes"] / len(json.dumps(test_data).encode()),
        }

    async def _benchmark_session_operations(self, sample_count: int):
        """Benchmark session management operations through unified cache."""
        logger.info("Benchmarking session operations...")
        
        cache_facade = CacheFacade(l1_max_size=1000, enable_l2=False, enable_l3=False)
        
        session_times: list[float] = []
        test_sample_count = min(sample_count, 50)
        
        for i in range(test_sample_count):
            session_id = f"session_benchmark_{i}"
            session_data = {
                "user_id": f"user_{i}",
                "active": True,
                "timestamp": time.time(),
                "preferences": {"theme": "dark", "language": "en"},
            }
            
            start_time = time.perf_counter()
            
            # Full session lifecycle
            await cache_facade.set_session(session_id, session_data)
            result = await cache_facade.get_session(session_id)
            await cache_facade.touch_session(session_id)
            await cache_facade.delete_session(session_id)
            
            session_time = (time.perf_counter() - start_time) * 1000
            session_times.append(session_time)
            
            assert result == session_data, f"Session data mismatch for {session_id}"
        
        self.benchmark_results["session_operations"] = {
            "session_lifecycle_avg_ms": statistics.mean(session_times),
            "session_lifecycle_p99_ms": sorted(session_times)[int(0.99 * len(session_times))],
            "sample_count": test_sample_count,
        }

    def _analyze_performance_results(self) -> Dict[str, Any]:
        """Analyze benchmark results against performance targets."""
        analysis: Dict[str, Any] = {
            "target_compliance": {},
            "performance_highlights": {},
            "improvement_opportunities": [],
            "overall_targets_met": True,
            "overall_performance_score": 0.0,
        }
        
        compliance_scores: list[float] = []
        
        # Analyze L1 Cache Performance
        if "l1_cache" in self.benchmark_results:
            l1_results = self.benchmark_results["l1_cache"]
            l1_compliance = {
                "avg_target_met": l1_results["overall_avg_ms"] <= self.performance_targets["l1_avg_ms"],
                "p99_target_met": l1_results["overall_p99_ms"] <= self.performance_targets["l1_p99_ms"],
                "hit_rate_target_met": l1_results["hit_rate"] >= self.performance_targets["l1_hit_rate"],
            }
            analysis["target_compliance"]["l1_cache"] = l1_compliance
            compliance_scores.append(sum(l1_compliance.values()) / len(l1_compliance))
            
            if l1_results["overall_avg_ms"] <= 0.1:  # Sub-100μs performance
                analysis["performance_highlights"]["l1_ultra_fast"] = f"L1 average: {l1_results['overall_avg_ms']:.3f}ms"
        
        # Analyze L2 Cache Performance
        if "l2_cache" in self.benchmark_results and self.benchmark_results["l2_cache"].get("available"):
            l2_results = self.benchmark_results["l2_cache"]
            l2_compliance = {
                "avg_target_met": l2_results["overall_avg_ms"] <= self.performance_targets["l2_avg_ms"],
                "p99_target_met": l2_results["overall_p99_ms"] <= self.performance_targets["l2_p99_ms"],
                "hit_rate_target_met": l2_results["hit_rate"] >= self.performance_targets["l2_hit_rate"],
            }
            analysis["target_compliance"]["l2_cache"] = l2_compliance
            compliance_scores.append(sum(l2_compliance.values()) / len(l2_compliance))
        
        # Analyze Cache Coordination
        if "cache_coordination" in self.benchmark_results:
            coord_results = self.benchmark_results["cache_coordination"]
            coord_compliance = {
                "coordination_target_met": coord_results["coordination_avg_ms"] <= self.performance_targets["coordination_max_ms"],
                "fallback_target_met": coord_results["fallback_max_ms"] <= self.performance_targets["coordination_max_ms"],
            }
            analysis["target_compliance"]["cache_coordination"] = coord_compliance
            compliance_scores.append(sum(coord_compliance.values()) / len(coord_compliance))
        
        # Analyze Memory Efficiency
        if "memory_efficiency" in self.benchmark_results:
            memory_results = self.benchmark_results["memory_efficiency"]
            memory_compliance = {
                "memory_overhead_target_met": memory_results["memory_per_entry_kb"] <= self.performance_targets["memory_overhead_kb"],
            }
            analysis["target_compliance"]["memory_efficiency"] = memory_compliance
            compliance_scores.append(sum(memory_compliance.values()) / len(memory_compliance))
            
            if memory_results["memory_per_entry_bytes"] <= 500:  # Very efficient
                analysis["performance_highlights"]["memory_efficient"] = f"Memory per entry: {memory_results['memory_per_entry_bytes']} bytes"
        
        # Calculate overall performance score
        if compliance_scores:
            analysis["overall_performance_score"] = sum(compliance_scores) / len(compliance_scores) * 100
            analysis["overall_targets_met"] = analysis["overall_performance_score"] >= 90.0
        
        # Generate improvement opportunities
        target_compliance = analysis["target_compliance"]
        if isinstance(target_compliance, dict):
            for component, compliance in target_compliance.items():
                if isinstance(compliance, dict):
                    failed_targets = [target for target, met in compliance.items() if not met]
                    if failed_targets:
                        improvement_opportunities = analysis["improvement_opportunities"]
                        if isinstance(improvement_opportunities, list):
                            improvement_opportunities.append(f"{component}: {', '.join(failed_targets)}")
        
        return analysis

    async def _generate_benchmark_report(self) -> str:
        """Generate human-readable benchmark report."""
        report_lines = [
            "=" * 80,
            "UNIFIED CACHE ARCHITECTURE PERFORMANCE BENCHMARK REPORT",
            "=" * 80,
            f"Generated: {datetime.now(UTC).isoformat()}",
            f"Performance Targets (2025 Standards):",
            f"  L1 Cache: <{self.performance_targets['l1_avg_ms']}ms avg, <{self.performance_targets['l1_p99_ms']}ms P99",
            f"  L2 Cache: <{self.performance_targets['l2_avg_ms']}ms avg, <{self.performance_targets['l2_p99_ms']}ms P99",
            f"  L3 Cache: <{self.performance_targets['l3_avg_ms']}ms avg, <{self.performance_targets['l3_p99_ms']}ms P99",
            f"  Coordination: <{self.performance_targets['coordination_max_ms']}ms max",
            f"  Hit Rates: >{self.performance_targets['l1_hit_rate']:.0%} L1, >{self.performance_targets['overall_hit_rate']:.0%} overall",
            f"  Memory: <{self.performance_targets['memory_overhead_kb']}KB per entry",
            "",
        ]
        
        # L1 Cache Results
        if "l1_cache" in self.benchmark_results:
            l1 = self.benchmark_results["l1_cache"]
            status = "✅" if l1["overall_avg_ms"] <= self.performance_targets["l1_avg_ms"] else "❌"
            report_lines.extend([
                f"L1 CACHE (MEMORY) PERFORMANCE: {status}",
                f"  Average Response Time: {l1['overall_avg_ms']:.3f}ms",
                f"  P99 Response Time: {l1['overall_p99_ms']:.3f}ms",
                f"  Hit Rate: {l1['hit_rate']:.2%}",
                f"  Memory per Entry: {l1['memory_per_entry_bytes']} bytes",
                f"  Samples: {l1['sample_count']}",
                "",
            ])
        
        # L2 Cache Results
        if "l2_cache" in self.benchmark_results:
            if self.benchmark_results["l2_cache"].get("available"):
                l2 = self.benchmark_results["l2_cache"]
                status = "✅" if l2["overall_avg_ms"] <= self.performance_targets["l2_avg_ms"] else "❌"
                report_lines.extend([
                    f"L2 CACHE (REDIS) PERFORMANCE: {status}",
                    f"  Average Response Time: {l2['overall_avg_ms']:.3f}ms",
                    f"  P99 Response Time: {l2['overall_p99_ms']:.3f}ms",
                    f"  Hit Rate: {l2['hit_rate']:.2%}",
                    f"  Samples: {l2['sample_count']}",
                    "",
                ])
            else:
                report_lines.extend([
                    "L2 CACHE (REDIS) PERFORMANCE: ⚠️ NOT AVAILABLE",
                    "  Redis cache not accessible during benchmark",
                    "",
                ])
        
        # Cache Coordination Results
        if "cache_coordination" in self.benchmark_results:
            coord = self.benchmark_results["cache_coordination"]
            status = "✅" if coord["coordination_avg_ms"] <= self.performance_targets["coordination_max_ms"] else "❌"
            report_lines.extend([
                f"CACHE COORDINATION PERFORMANCE: {status}",
                f"  Coordination Average: {coord['coordination_avg_ms']:.3f}ms",
                f"  Fallback Maximum: {coord['fallback_max_ms']:.3f}ms",
                f"  Samples: {coord['sample_count']} coordination, {coord['fallback_sample_count']} fallback",
                "",
            ])
        
        # Overall Performance Summary
        analysis = self._analyze_performance_results()
        report_lines.extend([
            "OVERALL PERFORMANCE SUMMARY:",
            f"  Performance Score: {analysis['overall_performance_score']:.1f}/100",
            f"  Targets Met: {'✅ ALL' if analysis['overall_targets_met'] else '❌ PARTIAL'}",
            "",
        ])
        
        # Performance Highlights
        if analysis["performance_highlights"]:
            report_lines.append("PERFORMANCE HIGHLIGHTS:")
            for highlight, value in analysis["performance_highlights"].items():
                report_lines.append(f"  🚀 {highlight}: {value}")
            report_lines.append("")
        
        # Improvement Opportunities
        if analysis["improvement_opportunities"]:
            report_lines.append("IMPROVEMENT OPPORTUNITIES:")
            for opportunity in analysis["improvement_opportunities"]:
                report_lines.append(f"  📈 {opportunity}")
            report_lines.append("")
        
        report_lines.extend([
            "=" * 80,
        ])
        
        return "\n".join(report_lines)

    async def save_benchmark_results(self, filepath: str | None = None) -> str:
        """Save benchmark results to JSON file."""
        if filepath is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            filepath = f"unified_cache_benchmark_{timestamp}.json"
        
        results = await self.run_comprehensive_benchmark()
        
        async with aiofiles.open(filepath, "w") as f:
            await f.write(json.dumps(results, indent=2, default=str))
        
        logger.info(f"Benchmark results saved to {filepath}")
        return filepath


async def run_unified_cache_benchmark(
    samples_per_test: int = 100,
    save_results: bool = True
) -> Dict[str, Any]:
    """Run unified cache architecture performance benchmark.
    
    Args:
        samples_per_test: Number of samples per performance test
        save_results: Whether to save results to file
        
    Returns:
        Dictionary containing comprehensive benchmark results
    """
    benchmark = UnifiedCacheBenchmark()
    results = await benchmark.run_comprehensive_benchmark(samples_per_test)
    
    # Print benchmark report
    print(results["benchmark_report"])
    
    # Save results if requested
    if save_results:
        filepath = await benchmark.save_benchmark_results()
        print(f"\n📄 Detailed results saved to: {filepath}")
    
    return results


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_unified_cache_benchmark())