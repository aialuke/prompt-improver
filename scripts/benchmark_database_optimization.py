#!/usr/bin/env python3
"""
Benchmark script for Phase 2 database optimization.

Measures real database load reduction from query caching and connection pooling.
Target: 50% reduction in database load.
"""

import asyncio
import time
import statistics
from datetime import datetime
from typing import List, Dict, Any
import json

from prompt_improver.database.cache_layer import DatabaseCacheLayer, CachePolicy, CacheStrategy
from prompt_improver.database.connection_pool_optimizer import ConnectionPoolOptimizer
from prompt_improver.database.query_optimizer import OptimizedQueryExecutor, get_query_executor
from prompt_improver.database.psycopg_client import get_psycopg_client
from prompt_improver.database.connection import get_session_context
from prompt_improver.database.performance_monitor import get_performance_monitor


class DatabaseOptimizationBenchmark:
    """Benchmark database optimization performance"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "baseline": {},
            "optimized": {},
            "improvement": {},
            "details": []
        }
    
    async def run_baseline_benchmark(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Run baseline benchmark without optimizations"""
        print("\n=== Running Baseline Benchmark (No Optimizations) ===")
        
        # Disable caching for baseline
        executor = get_query_executor()
        executor._cache_stats = {"hits": 0, "misses": 0, "cache_time_saved_ms": 0, "total_cached_queries": 0}
        
        metrics = {
            "total_queries": 0,
            "total_time_ms": 0,
            "query_times": [],
            "errors": 0,
            "connections_created": 0
        }
        
        # Test queries representing real workload
        test_queries = [
            ("SELECT * FROM rules WHERE active = true ORDER BY created_at DESC LIMIT 20", {}),
            ("SELECT COUNT(*) as total FROM sessions WHERE created_at > NOW() - INTERVAL '24 hours'", {}),
            ("SELECT r.id, r.name, COUNT(pi.id) as usage_count FROM rules r LEFT JOIN prompt_improvements pi ON r.id = pi.rule_id GROUP BY r.id, r.name ORDER BY usage_count DESC LIMIT 10", {}),
            ("SELECT * FROM prompt_improvements WHERE session_id = %(session_id)s", {"session_id": "test-session-1"}),
            ("SELECT AVG(improvement_score) as avg_score FROM rule_effectiveness WHERE rule_id = %(rule_id)s", {"rule_id": 1}),
        ]
        
        start_time = time.time()
        query_count = 0
        
        async with get_session_context() as session:
            while time.time() - start_time < duration_seconds:
                # Execute queries in random order
                for query, params in test_queries:
                    query_start = time.perf_counter()
                    try:
                        # Direct execution without caching
                        result = await session.execute(query, params)
                        await result.fetchall()
                        
                        query_time = (time.perf_counter() - query_start) * 1000
                        metrics["query_times"].append(query_time)
                        metrics["total_time_ms"] += query_time
                        metrics["total_queries"] += 1
                        query_count += 1
                        
                        if query_count % 50 == 0:
                            print(f"  Baseline: {query_count} queries executed...")
                            
                    except Exception as e:
                        metrics["errors"] += 1
                        print(f"  Error in baseline: {e}")
                
                # Small delay to simulate realistic load
                await asyncio.sleep(0.01)
        
        # Calculate statistics
        metrics["avg_query_time_ms"] = statistics.mean(metrics["query_times"]) if metrics["query_times"] else 0
        metrics["median_query_time_ms"] = statistics.median(metrics["query_times"]) if metrics["query_times"] else 0
        metrics["p95_query_time_ms"] = sorted(metrics["query_times"])[int(len(metrics["query_times"]) * 0.95)] if metrics["query_times"] else 0
        metrics["queries_per_second"] = metrics["total_queries"] / duration_seconds
        
        print(f"\nBaseline Results:")
        print(f"  Total Queries: {metrics['total_queries']}")
        print(f"  Average Query Time: {metrics['avg_query_time_ms']:.2f}ms")
        print(f"  Queries/Second: {metrics['queries_per_second']:.2f}")
        
        return metrics
    
    async def run_optimized_benchmark(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Run benchmark with all optimizations enabled"""
        print("\n=== Running Optimized Benchmark (With Caching & Pool Optimization) ===")
        
        # Enable aggressive caching
        cache_policy = CachePolicy(
            ttl_seconds=300,  # 5 minutes
            strategy=CacheStrategy.AGGRESSIVE,
            warm_on_startup=True
        )
        cache_layer = DatabaseCacheLayer(cache_policy)
        
        # Enable pool optimization
        pool_optimizer = ConnectionPoolOptimizer()
        await pool_optimizer.implement_connection_multiplexing()
        
        # Enable query optimization
        executor = get_query_executor()
        
        metrics = {
            "total_queries": 0,
            "total_time_ms": 0,
            "query_times": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "connections_reused": 0
        }
        
        # Same test queries
        test_queries = [
            ("SELECT * FROM rules WHERE active = true ORDER BY created_at DESC LIMIT 20", {}),
            ("SELECT COUNT(*) as total FROM sessions WHERE created_at > NOW() - INTERVAL '24 hours'", {}),
            ("SELECT r.id, r.name, COUNT(pi.id) as usage_count FROM rules r LEFT JOIN prompt_improvements pi ON r.id = pi.rule_id GROUP BY r.id, r.name ORDER BY usage_count DESC LIMIT 10", {}),
            ("SELECT * FROM prompt_improvements WHERE session_id = %(session_id)s", {"session_id": "test-session-1"}),
            ("SELECT AVG(improvement_score) as avg_score FROM rule_effectiveness WHERE rule_id = %(rule_id)s", {"rule_id": 1}),
        ]
        
        # Warm up cache with common queries
        print("  Warming up cache...")
        async with get_session_context() as session:
            for query, params in test_queries[:3]:  # Warm up with first 3 queries
                async with executor.execute_optimized_query(session, query, params, cache_ttl=300, enable_cache=True) as result:
                    pass
        
        start_time = time.time()
        query_count = 0
        
        async with get_session_context() as session:
            while time.time() - start_time < duration_seconds:
                for query, params in test_queries:
                    query_start = time.perf_counter()
                    try:
                        # Execute with caching
                        async with executor.execute_optimized_query(
                            session, query, params, cache_ttl=300, enable_cache=True
                        ) as result:
                            query_time = (time.perf_counter() - query_start) * 1000
                            metrics["query_times"].append(query_time)
                            metrics["total_time_ms"] += query_time
                            metrics["total_queries"] += 1
                            
                            if result["cache_hit"]:
                                metrics["cache_hits"] += 1
                            else:
                                metrics["cache_misses"] += 1
                            
                            query_count += 1
                            
                            if query_count % 50 == 0:
                                hit_rate = (metrics["cache_hits"] / query_count * 100) if query_count > 0 else 0
                                print(f"  Optimized: {query_count} queries, Cache Hit Rate: {hit_rate:.1f}%")
                                
                    except Exception as e:
                        metrics["errors"] += 1
                        print(f"  Error in optimized: {e}")
                
                await asyncio.sleep(0.01)
        
        # Get final optimization stats
        cache_stats = await cache_layer.get_cache_stats()
        pool_stats = await pool_optimizer.get_optimization_summary()
        executor_stats = await executor.get_performance_summary()
        
        # Calculate statistics
        metrics["avg_query_time_ms"] = statistics.mean(metrics["query_times"]) if metrics["query_times"] else 0
        metrics["median_query_time_ms"] = statistics.median(metrics["query_times"]) if metrics["query_times"] else 0
        metrics["p95_query_time_ms"] = sorted(metrics["query_times"])[int(len(metrics["query_times"]) * 0.95)] if metrics["query_times"] else 0
        metrics["queries_per_second"] = metrics["total_queries"] / duration_seconds
        metrics["cache_hit_rate"] = (metrics["cache_hits"] / metrics["total_queries"] * 100) if metrics["total_queries"] > 0 else 0
        metrics["cache_stats"] = cache_stats
        metrics["pool_stats"] = pool_stats
        
        print(f"\nOptimized Results:")
        print(f"  Total Queries: {metrics['total_queries']}")
        print(f"  Average Query Time: {metrics['avg_query_time_ms']:.2f}ms")
        print(f"  Cache Hit Rate: {metrics['cache_hit_rate']:.1f}%")
        print(f"  Queries/Second: {metrics['queries_per_second']:.2f}")
        
        return metrics
    
    async def analyze_results(self, baseline: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and compare benchmark results"""
        print("\n=== Performance Analysis ===")
        
        # Calculate improvements
        query_time_reduction = ((baseline["avg_query_time_ms"] - optimized["avg_query_time_ms"]) / baseline["avg_query_time_ms"] * 100) if baseline["avg_query_time_ms"] > 0 else 0
        
        throughput_improvement = ((optimized["queries_per_second"] - baseline["queries_per_second"]) / baseline["queries_per_second"] * 100) if baseline["queries_per_second"] > 0 else 0
        
        # Database load reduction calculation
        # Load = queries that hit the database (not cached)
        baseline_db_queries = baseline["total_queries"]  # All queries hit DB
        optimized_db_queries = optimized["cache_misses"]  # Only misses hit DB
        
        db_load_reduction = ((baseline_db_queries - optimized_db_queries) / baseline_db_queries * 100) if baseline_db_queries > 0 else 0
        
        # Additional metrics from cache and pool
        cache_contribution = optimized.get("cache_stats", {}).get("database_load_reduction_percent", 0)
        pool_contribution = optimized.get("pool_stats", {}).get("optimization", {}).get("database_load_reduction_percent", 0)
        
        analysis = {
            "query_time_reduction_percent": round(query_time_reduction, 1),
            "throughput_improvement_percent": round(throughput_improvement, 1),
            "database_load_reduction_percent": round(db_load_reduction, 1),
            "cache_contribution_percent": round(cache_contribution, 1),
            "pool_contribution_percent": round(pool_contribution, 1),
            "target_achieved": db_load_reduction >= 50,
            "details": {
                "baseline_avg_query_ms": round(baseline["avg_query_time_ms"], 2),
                "optimized_avg_query_ms": round(optimized["avg_query_time_ms"], 2),
                "baseline_qps": round(baseline["queries_per_second"], 2),
                "optimized_qps": round(optimized["queries_per_second"], 2),
                "cache_hit_rate": round(optimized["cache_hit_rate"], 1),
                "total_queries": optimized["total_queries"],
                "queries_cached": optimized["cache_hits"],
                "database_queries_avoided": optimized["cache_hits"]
            }
        }
        
        print(f"\nPerformance Improvements:")
        print(f"  Query Time Reduction: {analysis['query_time_reduction_percent']:.1f}%")
        print(f"  Throughput Improvement: {analysis['throughput_improvement_percent']:.1f}%")
        print(f"  Database Load Reduction: {analysis['database_load_reduction_percent']:.1f}%")
        print(f"    - Cache Contribution: {analysis['cache_contribution_percent']:.1f}%")
        print(f"    - Pool Contribution: {analysis['pool_contribution_percent']:.1f}%")
        print(f"\n  Target (50% reduction) Achieved: {'✅ YES' if analysis['target_achieved'] else '❌ NO'}")
        
        return analysis
    
    async def run_benchmark(self, duration_seconds: int = 30):
        """Run complete benchmark suite"""
        print(f"Starting Database Optimization Benchmark ({duration_seconds}s per test)")
        print("=" * 60)
        
        # Monitor system during benchmark
        monitor = await get_performance_monitor()
        
        # Take initial snapshot
        initial_snapshot = await monitor.take_performance_snapshot()
        
        # Run baseline benchmark
        baseline_results = await self.run_baseline_benchmark(duration_seconds)
        self.results["baseline"] = baseline_results
        
        # Allow system to stabilize
        print("\nAllowing system to stabilize...")
        await asyncio.sleep(5)
        
        # Run optimized benchmark
        optimized_results = await self.run_optimized_benchmark(duration_seconds)
        self.results["optimized"] = optimized_results
        
        # Take final snapshot
        final_snapshot = await monitor.take_performance_snapshot()
        
        # Analyze results
        analysis = await self.analyze_results(baseline_results, optimized_results)
        self.results["improvement"] = analysis
        
        # Add system metrics
        self.results["system_metrics"] = {
            "initial_cache_hit_ratio": initial_snapshot.cache_hit_ratio,
            "final_cache_hit_ratio": final_snapshot.cache_hit_ratio,
            "initial_avg_query_time": initial_snapshot.avg_query_time_ms,
            "final_avg_query_time": final_snapshot.avg_query_time_ms
        }
        
        # Save results
        filename = f"database_optimization_benchmark_{int(time.time())}.json"
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n\nBenchmark results saved to: {filename}")
        
        return self.results


async def main():
    """Run the benchmark"""
    benchmark = DatabaseOptimizationBenchmark()
    
    # Run with configurable duration
    import sys
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    
    results = await benchmark.run_benchmark(duration_seconds=duration)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Database Load Reduction: {results['improvement']['database_load_reduction_percent']:.1f}%")
    print(f"Target (50%) Achieved: {'YES ✅' if results['improvement']['target_achieved'] else 'NO ❌'}")
    print(f"Query Performance Improvement: {results['improvement']['query_time_reduction_percent']:.1f}%")
    print(f"Throughput Improvement: {results['improvement']['throughput_improvement_percent']:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())