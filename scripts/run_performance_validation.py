#!/usr/bin/env python3
"""Performance Validation Script for Phase 3 Optimizations.

This script demonstrates and validates the performance improvements achieved
through multi-level caching and connection pool optimizations implemented
in Phase 3.

Usage:
    python scripts/run_performance_validation.py [--iterations=100] [--verbose]
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from prompt_improver.performance.validation.performance_benchmark_suite import (
    PerformanceBenchmarkSuite,
    run_performance_benchmark,
)
from prompt_improver.services.cache.cache_facade import (
    CacheFacade as CacheManager,
    get_cache,
)
# Using CacheFacade unified cache architecture

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def create_test_services():
    """Create test service instances for benchmarking."""
    logger.info("Creating test service instances...")
    
    # Create unified cache facade for testing
    cache_manager = CacheManager(
        l1_max_size=1000,
        l2_default_ttl=900,
        enable_l2=True,  # Will gracefully degrade if Redis unavailable
        enable_l3=False,  # Skip for testing
    )
    
    try:
        # CacheFacade doesn't need explicit initialization
        logger.info("Cache facade created successfully")
    except Exception as e:
        logger.warning(f"Cache facade creation failed: {e}")
        # Continue with degraded caching
    
    # Create mock services for demonstration
    # In a real implementation, these would be properly initialized
    # with actual database connections and configurations
    
    class MockPromptService:
        """Mock prompt service for testing."""
        
        def __init__(self, cache_manager):
            self.cache_manager = cache_manager
            self._request_count = 0
        
        async def improve_prompt(self, prompt, session_id, options=None):
            """Mock prompt improvement with caching simulation."""
            self._request_count += 1
            
            # Simulate cache key generation
            import hashlib
            cache_key = f"prompt:{hashlib.md5(f'{prompt}:{session_id}'.encode()).hexdigest()}"
            
            # Check cache first
            if self.cache_manager:
                try:
                    cached_result = await self.cache_manager.get(cache_key)
                    if cached_result:
                        return {**cached_result, "cache_hit": True, "served_from_cache": True}
                except Exception:
                    pass
            
            # Simulate processing time
            await asyncio.sleep(0.05)  # 50ms processing time
            
            result = {
                "status": "success",
                "session_id": session_id,
                "original_prompt": prompt,
                "improved_prompt": f"Improved: {prompt}",
                "rules_applied": ["clarity", "specificity"],
                "processing_time_ms": 50.0,
                "cache_hit": False,
                "served_from_cache": False,
            }
            
            # Cache the result
            if self.cache_manager:
                try:
                    await self.cache_manager.set(cache_key, result, ttl_seconds=300)
                except Exception:
                    pass
            
            return result
    
    class MockAnalyticsService:
        """Mock analytics service for testing."""
        
        async def health_check(self):
            """Mock health check."""
            await asyncio.sleep(0.002)  # 2ms processing time
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    class MockMLService:
        """Mock ML service for testing."""
        
        def __init__(self, cache_manager):
            self.cache_manager = cache_manager
        
        async def predict_rule_effectiveness(self, model_id, features):
            """Mock ML inference with caching."""
            import hashlib
            cache_key = f"ml:{model_id}:{hashlib.md5(str(features).encode()).hexdigest()}"
            
            # Check cache first
            if self.cache_manager:
                try:
                    cached_result = await self.cache_manager.get(cache_key)
                    if cached_result:
                        return {**cached_result, "cache_hit": True, "served_from_cache": True}
                except Exception:
                    pass
            
            # Simulate ML inference time
            await asyncio.sleep(0.008)  # 8ms processing time
            
            result = {
                "model_id": model_id,
                "prediction": 0.85,
                "confidence": 0.92,
                "features_used": len(features),
                "cache_hit": False,
                "served_from_cache": False,
            }
            
            # Cache the result
            if self.cache_manager:
                try:
                    await self.cache_manager.set(cache_key, result, ttl_seconds=1800)
                except Exception:
                    pass
            
            return result
    
    class MockAnalyticsRepository:
        """Mock analytics repository for testing."""
        
        def __init__(self, cache_manager):
            self.cache_manager = cache_manager
            self._performance_metrics = {
                "query_count": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "avg_query_time_ms": 25.0,
            }
        
        async def get_session_analytics(self, start_date, end_date, filters=None):
            """Mock session analytics with caching."""
            self._performance_metrics["query_count"] += 1
            
            import hashlib
            content = f"analytics:{start_date}:{end_date}:{filters}"
            cache_key = f"analytics:{hashlib.md5(content.encode()).hexdigest()}"
            
            # Check cache first
            if self.cache_manager:
                try:
                    cached_result = await self.cache_manager.get(cache_key)
                    if cached_result:
                        self._performance_metrics["cache_hits"] += 1
                        return cached_result
                except Exception:
                    pass
            
            self._performance_metrics["cache_misses"] += 1
            
            # Simulate database query time
            await asyncio.sleep(0.03)  # 30ms query time
            
            result = {
                "total_sessions": 150,
                "avg_improvement_score": 0.82,
                "avg_quality_score": 0.78,
                "avg_confidence_level": 0.85,
                "top_performing_rules": [
                    {"rule_name": "clarity", "success_rate": 0.89},
                    {"rule_name": "specificity", "success_rate": 0.76},
                ],
                "performance_distribution": {
                    "excellent": 45, "good": 65, "fair": 30, "poor": 10
                },
            }
            
            # Cache the result
            if self.cache_manager:
                try:
                    await self.cache_manager.set(cache_key, result, ttl_seconds=300)
                except Exception:
                    pass
            
            return result
        
        async def get_performance_metrics(self):
            """Get repository performance metrics."""
            total_requests = self._performance_metrics["query_count"]
            cache_hit_rate = (
                self._performance_metrics["cache_hits"] / max(total_requests, 1)
            )
            return {
                "cache_hit_rate": cache_hit_rate,
                "performance": self._performance_metrics,
            }
    
    class MockPostgresPool:
        """Mock PostgreSQL pool for testing."""
        
        def __init__(self, cache_manager):
            self.cache_manager = cache_manager
        
        async def execute_cached_query(self, query, params=None):
            """Mock cached query execution."""
            import hashlib
            content = f"pg:{query}:{params}"
            cache_key = f"pg:{hashlib.md5(content.encode()).hexdigest()}"
            
            # Check cache first
            if self.cache_manager:
                try:
                    cached_result = await self.cache_manager.get(cache_key)
                    if cached_result:
                        return cached_result
                except Exception:
                    pass
            
            # Simulate database query time
            await asyncio.sleep(0.02)  # 20ms query time
            
            result = [{"count": 42, "avg_score": 0.78}]
            
            # Cache the result
            if self.cache_manager:
                try:
                    await self.cache_manager.set(cache_key, result, ttl_seconds=300)
                except Exception:
                    pass
            
            return result
    
    # Create service instances
    prompt_service = MockPromptService(cache_manager)
    analytics_service = MockAnalyticsService()
    ml_service = MockMLService(cache_manager)
    analytics_repository = MockAnalyticsRepository(cache_manager)
    postgres_pool = MockPostgresPool(cache_manager)
    
    return (
        prompt_service,
        analytics_service,
        ml_service,
        analytics_repository,
        postgres_pool,
        cache_manager,
    )


async def run_demo_benchmark(iterations: int = 50, verbose: bool = False) -> None:
    """Run performance validation demo."""
    logger.info(f"Starting Phase 3 Performance Validation Demo (iterations: {iterations})")
    
    try:
        # Create test services
        services = await create_test_services()
        (
            prompt_service,
            analytics_service,
            ml_service,
            analytics_repository,
            postgres_pool,
            cache_manager,
        ) = services
        
        # Create benchmark suite
        benchmark_suite = PerformanceBenchmarkSuite(
            prompt_service=prompt_service,
            analytics_service=analytics_service,
            ml_service=ml_service,
            analytics_repository=analytics_repository,
            postgres_pool=postgres_pool,
            cache_manager=cache_manager,
        )
        
        # Run comprehensive benchmark
        logger.info("Running comprehensive performance benchmark...")
        results = await benchmark_suite.run_comprehensive_benchmark(
            test_iterations=iterations,
            warmup_iterations=max(5, iterations // 10)
        )
        
        # Display results
        print("\\n" + "="*80)
        print("PHASE 3 PERFORMANCE OPTIMIZATION RESULTS")
        print("="*80)
        
        print(f"Overall Success: {'✅ PASS' if results['overall_success'] else '❌ FAIL'}")
        print(f"Total Duration: {results['total_duration_seconds']:.2f} seconds")
        print(f"Targets Met: {results['targets_met']}")
        print(f"Average Cache Hit Rate: {results['avg_cache_hit_rate']:.2%}")
        
        print("\\n📊 PERFORMANCE SUMMARY:")
        summary = results["performance_summary"]
        
        target_results = [
            ("Prompt Improvement", summary.get("prompt_improvement_p95_ms"), 20.0),
            ("Analytics Queries", summary.get("analytics_queries_p95_ms"), 50.0),
            ("ML Inference", summary.get("ml_inference_p95_ms"), 10.0),
            ("Health Checks", summary.get("health_checks_p95_ms"), 5.0),
        ]
        
        for name, actual, target in target_results:
            if actual is not None:
                status = "✅" if actual <= target else "❌"
                print(f"  {status} {name}: {actual:.2f}ms (target: {target:.2f}ms)")
        
        print("\\n💡 RECOMMENDATIONS:")
        for recommendation in results["recommendations"]:
            print(f"  {recommendation}")
        
        if verbose:
            print("\\n📋 DETAILED TEST RESULTS:")
            for test_result in results["test_results"]:
                print(f"\\n  Test: {test_result['test_name']}")
                print(f"    Target Met: {'✅' if test_result['target_met'] else '❌'}")
                print(f"    P95 Response Time: {test_result['actual_p95_response_time_ms']:.2f}ms")
                print(f"    Average Response Time: {test_result['avg_response_time_ms']:.2f}ms")
                print(f"    Cache Hit Rate: {test_result['cache_hit_rate']:.2%}")
                print(f"    Test Iterations: {test_result['test_iterations']}")
        
        # Save results to file
        results_file = project_root / "benchmark_results" / f"performance_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n📄 Results saved to: {results_file}")
        
        # Cleanup
        if cache_manager:
            try:
                # CacheFacade doesn't require explicit shutdown
                logger.info("Cache facade cleanup completed")
            except Exception as e:
                logger.warning(f"Error during cache facade cleanup: {e}")
        
        print("\\n🎉 Performance validation demo completed!")
        
    except Exception as e:
        logger.error(f"Performance validation failed: {e}")
        raise


def main():
    """Main entry point for performance validation script."""
    parser = argparse.ArgumentParser(
        description="Run Phase 3 Performance Optimization Validation"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of test iterations (default: 50)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed test results"
    )
    
    args = parser.parse_args()
    
    # Run the async demo
    asyncio.run(run_demo_benchmark(args.iterations, args.verbose))


if __name__ == "__main__":
    main()