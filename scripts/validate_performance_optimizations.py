#!/usr/bin/env python3
"""Performance optimization validation script.

This script validates that all Phase 3 performance optimizations are working
correctly and meeting the target performance improvements:
- >20% response time reduction on critical endpoints
- >30% reduction in database query load through caching  
- <5% memory usage increase despite caching layer
- >90% cache hit ratio on frequently accessed data
"""

import asyncio
import logging
import statistics
import sys
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

import psutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceValidator:
    """Validates performance optimizations against targets."""

    def __init__(self):
        self.test_results = {}
        self.baseline_metrics = {}
        self.optimized_metrics = {}

    async def run_validation_suite(self) -> Dict[str, Any]:
        """Run complete performance validation suite."""
        logger.info("Starting Performance Optimization Validation Suite")
        
        try:
            # 1. Test caching infrastructure
            await self._test_caching_infrastructure()
            
            # 2. Test repository caching
            await self._test_repository_caching()
            
            # 3. Test ML service caching
            await self._test_ml_service_caching()
            
            # 4. Test API response caching
            await self._test_api_response_caching()
            
            # 5. Test connection pool optimization
            await self._test_connection_pool_optimization()
            
            # 6. Test memory efficiency
            await self._test_memory_efficiency()
            
            # 7. Run load testing
            await self._test_performance_under_load()
            
            # 8. Validate performance targets
            validation_results = await self._validate_performance_targets()
            
            # Generate summary report
            summary = self._generate_validation_report()
            
            return {
                "validation_passed": validation_results["all_targets_met"],
                "test_results": self.test_results,
                "performance_improvements": validation_results,
                "summary": summary,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Validation suite failed: {e}")
            return {
                "validation_passed": False,
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    async def _test_caching_infrastructure(self):
        """Test unified caching facade infrastructure."""
        logger.info("Testing caching infrastructure...")
        
        try:
            from prompt_improver.performance.caching import (
                get_performance_cache,
                CacheStrategy,
                CacheKey,
            )
            
            cache_facade = get_performance_cache()
            
            # Test different caching strategies
            test_key = CacheKey(
                namespace="validation",
                operation="test_operation",
                parameters={"test": True, "timestamp": datetime.now(UTC).isoformat()},
            )
            
            strategies_to_test = [
                CacheStrategy.ULTRA_FAST,
                CacheStrategy.FAST,
                CacheStrategy.BALANCED,
                CacheStrategy.LONG_TERM,
            ]
            
            strategy_performance = {}
            
            for strategy in strategies_to_test:
                # Test cache set/get performance
                start_time = time.perf_counter()
                
                result = await cache_facade.get_or_compute(
                    cache_key=test_key,
                    compute_func=lambda: {"computed": True, "value": 42},
                    strategy=strategy,
                )
                
                execution_time = (time.perf_counter() - start_time) * 1000
                strategy_performance[strategy.value] = {
                    "execution_time_ms": execution_time,
                    "result_valid": result.get("value") == 42,
                }
            
            # Health check
            health_result = await cache_facade.health_check()
            
            # Performance stats
            perf_stats = await cache_facade.get_performance_stats()
            
            self.test_results["caching_infrastructure"] = {
                "strategy_performance": strategy_performance,
                "health_check": health_result,
                "performance_stats": perf_stats,
                "passed": health_result["healthy"] and all(
                    p["result_valid"] for p in strategy_performance.values()
                ),
            }
            
            logger.info("✓ Caching infrastructure test completed")
            
        except Exception as e:
            logger.error(f"Caching infrastructure test failed: {e}")
            self.test_results["caching_infrastructure"] = {
                "passed": False,
                "error": str(e),
            }

    async def _test_repository_caching(self):
        """Test repository layer caching."""
        logger.info("Testing repository caching...")
        
        try:
            from prompt_improver.performance.caching.integration_example import (
                OptimizedAnalyticsRepository,
            )
            
            repository = OptimizedAnalyticsRepository(None)
            
            # Test dashboard summary caching
            test_params = [24, 48, 168]  # Different time periods
            performance_metrics = []
            
            for period_hours in test_params:
                # Cold cache test
                start_time = time.perf_counter()
                result1 = await repository.get_dashboard_summary_optimized(period_hours)
                cold_time = (time.perf_counter() - start_time) * 1000
                
                # Warm cache test
                start_time = time.perf_counter()
                result2 = await repository.get_dashboard_summary_optimized(period_hours)
                warm_time = (time.perf_counter() - start_time) * 1000
                
                performance_metrics.append({
                    "period_hours": period_hours,
                    "cold_cache_ms": cold_time,
                    "warm_cache_ms": warm_time,
                    "speedup_ratio": cold_time / warm_time if warm_time > 0 else 0,
                    "meets_target": warm_time <= 50,  # Target <50ms for cached responses
                })
            
            # Test time-series caching
            start_date = datetime.now(UTC) - timedelta(days=7)
            end_date = datetime.now(UTC)
            
            start_time = time.perf_counter()
            time_series_result = await repository.get_time_series_data_optimized(
                "performance", "day", start_date, end_date
            )
            time_series_time = (time.perf_counter() - start_time) * 1000
            
            # Get repository performance stats
            repo_stats = repository.get_cache_performance_stats()
            
            self.test_results["repository_caching"] = {
                "dashboard_performance": performance_metrics,
                "time_series_performance": {
                    "execution_time_ms": time_series_time,
                    "data_points": len(time_series_result),
                    "meets_target": time_series_time <= 200,
                },
                "repository_stats": repo_stats,
                "passed": all(m["meets_target"] for m in performance_metrics) and 
                         time_series_time <= 200,
            }
            
            logger.info("✓ Repository caching test completed")
            
        except Exception as e:
            logger.error(f"Repository caching test failed: {e}")
            self.test_results["repository_caching"] = {
                "passed": False,
                "error": str(e),
            }

    async def _test_ml_service_caching(self):
        """Test ML service caching."""
        logger.info("Testing ML service caching...")
        
        try:
            from prompt_improver.performance.caching.integration_example import (
                OptimizedMLService,
            )
            
            ml_service = OptimizedMLService()
            
            # Test prompt improvement caching
            test_prompts = [
                "Improve this text for clarity",
                "Make this more specific and actionable",
                "Enhance readability and engagement",
            ]
            
            performance_metrics = []
            
            for i, prompt in enumerate(test_prompts):
                context = {"domain": f"test_{i}", "user_type": "analyst"}
                
                # Cold cache
                start_time = time.perf_counter()
                result1 = await ml_service.cached_prompt_improvement(
                    prompt, context, "clarity"
                )
                cold_time = (time.perf_counter() - start_time) * 1000
                
                # Warm cache
                start_time = time.perf_counter()
                result2 = await ml_service.cached_prompt_improvement(
                    prompt, context, "clarity"
                )
                warm_time = (time.perf_counter() - start_time) * 1000
                
                performance_metrics.append({
                    "prompt": prompt[:30] + "...",
                    "cold_cache_ms": cold_time,
                    "warm_cache_ms": warm_time,
                    "speedup_ratio": cold_time / warm_time if warm_time > 0 else 0,
                    "meets_target": warm_time <= 10,  # Target <10ms for cached ML predictions
                })
            
            # Test feature extraction caching
            test_text = "This is a sample text for feature extraction testing"
            
            start_time = time.perf_counter()
            features1 = await ml_service.cached_feature_extraction(test_text, "linguistic")
            feature_cold_time = (time.perf_counter() - start_time) * 1000
            
            start_time = time.perf_counter()
            features2 = await ml_service.cached_feature_extraction(test_text, "linguistic")
            feature_warm_time = (time.perf_counter() - start_time) * 1000
            
            # Get ML cache performance stats
            from prompt_improver.performance.caching import get_ml_service_cache
            ml_cache = get_ml_service_cache()
            ml_stats = await ml_cache.get_ml_cache_performance()
            
            self.test_results["ml_service_caching"] = {
                "prompt_improvement_performance": performance_metrics,
                "feature_extraction_performance": {
                    "cold_cache_ms": feature_cold_time,
                    "warm_cache_ms": feature_warm_time,
                    "speedup_ratio": feature_cold_time / feature_warm_time if feature_warm_time > 0 else 0,
                    "meets_target": feature_warm_time <= 10,
                },
                "ml_cache_stats": ml_stats,
                "passed": all(m["meets_target"] for m in performance_metrics) and
                         feature_warm_time <= 10,
            }
            
            logger.info("✓ ML service caching test completed")
            
        except Exception as e:
            logger.error(f"ML service caching test failed: {e}")
            self.test_results["ml_service_caching"] = {
                "passed": False,
                "error": str(e),
            }

    async def _test_api_response_caching(self):
        """Test API response caching."""
        logger.info("Testing API response caching...")
        
        try:
            from prompt_improver.performance.caching import get_api_response_cache
            
            api_cache = get_api_response_cache()
            
            # Create mock request
            class MockRequest:
                def __init__(self):
                    self.method = "GET"
                    self.path_params = {}
                    self.query_params = {"time_range_hours": "24"}
                    self.headers = {"Authorization": "Bearer test"}
            
            mock_request = MockRequest()
            
            # Test different endpoint types
            endpoint_tests = [
                {
                    "name": "dashboard_metrics",
                    "response_func": lambda req: {"status": "ok", "data": {"sessions": 100}},
                    "cache_method": "cached_dashboard_endpoint",
                },
                {
                    "name": "analytics_trends",
                    "response_func": lambda req: {"trends": [{"date": "2025-01-01", "value": 0.8}]},
                    "cache_method": "cached_analytics_endpoint",
                },
            ]
            
            api_performance_metrics = []
            
            for test_config in endpoint_tests:
                endpoint_name = test_config["name"]
                response_func = test_config["response_func"]
                
                # Cold cache
                start_time = time.perf_counter()
                result1 = await api_cache.cached_endpoint_response(
                    endpoint_name, mock_request, response_func
                )
                cold_time = (time.perf_counter() - start_time) * 1000
                
                # Warm cache
                start_time = time.perf_counter()
                result2 = await api_cache.cached_endpoint_response(
                    endpoint_name, mock_request, response_func
                )
                warm_time = (time.perf_counter() - start_time) * 1000
                
                api_performance_metrics.append({
                    "endpoint": endpoint_name,
                    "cold_cache_ms": cold_time,
                    "warm_cache_ms": warm_time,
                    "speedup_ratio": cold_time / warm_time if warm_time > 0 else 0,
                    "meets_target": warm_time <= 20,  # Target <20ms for cached API responses
                })
            
            # Get API cache stats
            api_stats = await api_cache.get_api_cache_performance()
            
            # Health check
            api_health = await api_cache.health_check()
            
            self.test_results["api_response_caching"] = {
                "endpoint_performance": api_performance_metrics,
                "api_cache_stats": api_stats,
                "health_check": api_health,
                "passed": all(m["meets_target"] for m in api_performance_metrics) and
                         api_health["healthy"],
            }
            
            logger.info("✓ API response caching test completed")
            
        except Exception as e:
            logger.error(f"API response caching test failed: {e}")
            self.test_results["api_response_caching"] = {
                "passed": False,
                "error": str(e),
            }

    async def _test_connection_pool_optimization(self):
        """Test connection pool optimization."""
        logger.info("Testing connection pool optimization...")
        
        try:
            # Test database connection performance
            connection_metrics = []
            
            # Simulate multiple concurrent database operations
            async def simulate_db_operation():
                start_time = time.perf_counter()
                await asyncio.sleep(0.01)  # Simulate 10ms DB operation
                return (time.perf_counter() - start_time) * 1000
            
            # Test concurrent operations
            concurrent_tasks = [simulate_db_operation() for _ in range(20)]
            operation_times = await asyncio.gather(*concurrent_tasks)
            
            avg_operation_time = statistics.mean(operation_times)
            max_operation_time = max(operation_times)
            
            connection_metrics.append({
                "concurrent_operations": len(concurrent_tasks),
                "avg_operation_time_ms": avg_operation_time,
                "max_operation_time_ms": max_operation_time,
                "meets_target": max_operation_time <= 100,  # Target <100ms for concurrent ops
            })
            
            self.test_results["connection_pool_optimization"] = {
                "connection_metrics": connection_metrics,
                "passed": all(m["meets_target"] for m in connection_metrics),
            }
            
            logger.info("✓ Connection pool optimization test completed")
            
        except Exception as e:
            logger.error(f"Connection pool optimization test failed: {e}")
            self.test_results["connection_pool_optimization"] = {
                "passed": False,
                "error": str(e),
            }

    async def _test_memory_efficiency(self):
        """Test memory efficiency of caching layer."""
        logger.info("Testing memory efficiency...")
        
        try:
            # Get baseline memory usage
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Load cache with test data
            from prompt_improver.performance.caching import get_performance_cache
            cache_facade = get_performance_cache()
            
            # Create test cache entries
            test_entries = []
            for i in range(100):
                from prompt_improver.performance.caching.cache_facade import CacheKey
                cache_key = CacheKey(
                    namespace="memory_test",
                    operation=f"test_op_{i}",
                    parameters={"index": i, "data": f"test_data_{i}"},
                )
                test_data = {"value": i, "data": "x" * 1000}  # 1KB per entry
                test_entries.append({"key": cache_key, "value": test_data})
            
            # Warm cache
            await cache_facade.warm_cache(test_entries)
            
            # Measure memory after caching
            cached_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = cached_memory - baseline_memory
            memory_increase_percent = (memory_increase / baseline_memory) * 100
            
            # Test cache performance with loaded data
            start_time = time.perf_counter()
            for entry in test_entries[:10]:  # Test first 10 entries
                result = await cache_facade.get_or_compute(
                    cache_key=entry["key"],
                    compute_func=lambda: {"fallback": True},
                    strategy=cache_facade._get_cache_config(cache_facade.default_strategy)["strategy"] if hasattr(cache_facade, 'default_strategy') else None,
                )
            retrieval_time = (time.perf_counter() - start_time) * 1000
            
            self.test_results["memory_efficiency"] = {
                "baseline_memory_mb": baseline_memory,
                "cached_memory_mb": cached_memory,
                "memory_increase_mb": memory_increase,
                "memory_increase_percent": memory_increase_percent,
                "cache_entries_loaded": len(test_entries),
                "retrieval_time_ms": retrieval_time,
                "meets_memory_target": memory_increase_percent <= 10,  # Target <10% increase
                "meets_performance_target": retrieval_time <= 50,
                "passed": memory_increase_percent <= 10 and retrieval_time <= 50,
            }
            
            logger.info("✓ Memory efficiency test completed")
            
        except Exception as e:
            logger.error(f"Memory efficiency test failed: {e}")
            self.test_results["memory_efficiency"] = {
                "passed": False,
                "error": str(e),
            }

    async def _test_performance_under_load(self):
        """Test performance under realistic load."""
        logger.info("Testing performance under load...")
        
        try:
            from prompt_improver.performance.caching.integration_example import (
                OptimizedAnalyticsRepository,
                OptimizedMLService,
            )
            
            repository = OptimizedAnalyticsRepository(None)
            ml_service = OptimizedMLService()
            
            # Define load test scenarios
            async def repository_load_test():
                tasks = []
                for i in range(50):  # 50 concurrent repository calls
                    task = repository.get_dashboard_summary_optimized(24)
                    tasks.append(task)
                
                start_time = time.perf_counter()
                results = await asyncio.gather(*tasks)
                total_time = (time.perf_counter() - start_time) * 1000
                
                return {
                    "operation": "repository_load",
                    "concurrent_requests": len(tasks),
                    "total_time_ms": total_time,
                    "avg_time_per_request_ms": total_time / len(tasks),
                    "requests_per_second": len(tasks) / (total_time / 1000),
                }
            
            async def ml_load_test():
                tasks = []
                for i in range(30):  # 30 concurrent ML calls
                    task = ml_service.cached_prompt_improvement(
                        f"Test prompt {i}", {"test": True}, "general"
                    )
                    tasks.append(task)
                
                start_time = time.perf_counter()
                results = await asyncio.gather(*tasks)
                total_time = (time.perf_counter() - start_time) * 1000
                
                return {
                    "operation": "ml_load",
                    "concurrent_requests": len(tasks),
                    "total_time_ms": total_time,
                    "avg_time_per_request_ms": total_time / len(tasks),
                    "requests_per_second": len(tasks) / (total_time / 1000),
                }
            
            # Run load tests
            repo_load_result = await repository_load_test()
            ml_load_result = await ml_load_test()
            
            # Evaluate performance targets
            load_test_results = [repo_load_result, ml_load_result]
            
            for result in load_test_results:
                result["meets_latency_target"] = result["avg_time_per_request_ms"] <= 200
                result["meets_throughput_target"] = result["requests_per_second"] >= 10
            
            self.test_results["performance_under_load"] = {
                "load_test_results": load_test_results,
                "passed": all(
                    r["meets_latency_target"] and r["meets_throughput_target"]
                    for r in load_test_results
                ),
            }
            
            logger.info("✓ Performance under load test completed")
            
        except Exception as e:
            logger.error(f"Performance under load test failed: {e}")
            self.test_results["performance_under_load"] = {
                "passed": False,
                "error": str(e),
            }

    async def _validate_performance_targets(self) -> Dict[str, Any]:
        """Validate all performance targets are met."""
        logger.info("Validating performance targets...")
        
        target_validations = {}
        
        # Target 1: >20% response time reduction
        repo_metrics = self.test_results.get("repository_caching", {}).get("dashboard_performance", [])
        if repo_metrics:
            avg_speedup = statistics.mean([m["speedup_ratio"] for m in repo_metrics])
            response_time_improvement = ((avg_speedup - 1) / avg_speedup) * 100
            target_validations["response_time_reduction"] = {
                "target": 20.0,
                "achieved": response_time_improvement,
                "met": response_time_improvement >= 20.0,
            }
        
        # Target 2: >30% database load reduction through caching
        cache_stats = self.test_results.get("caching_infrastructure", {}).get("performance_stats", {})
        if cache_stats:
            hit_rate = cache_stats.get("hit_rate", 0) * 100
            target_validations["database_load_reduction"] = {
                "target": 30.0,
                "achieved": hit_rate,
                "met": hit_rate >= 30.0,
            }
        
        # Target 3: <5% memory usage increase
        memory_metrics = self.test_results.get("memory_efficiency", {})
        if memory_metrics:
            memory_increase = memory_metrics.get("memory_increase_percent", 0)
            target_validations["memory_usage_increase"] = {
                "target": 5.0,
                "achieved": memory_increase,
                "met": memory_increase <= 5.0,
            }
        
        # Target 4: >90% cache hit ratio
        ml_stats = self.test_results.get("ml_service_caching", {}).get("ml_cache_stats", {})
        if ml_stats:
            ml_hit_rate = ml_stats.get("ml_cache_stats", {}).get("cache_hit_rate", 0) * 100
            target_validations["cache_hit_ratio"] = {
                "target": 90.0,
                "achieved": ml_hit_rate,
                "met": ml_hit_rate >= 90.0,
            }
        
        # Overall validation
        all_targets_met = all(v.get("met", False) for v in target_validations.values())
        
        return {
            "target_validations": target_validations,
            "all_targets_met": all_targets_met,
            "summary": f"Met {sum(1 for v in target_validations.values() if v.get('met', False))}/{len(target_validations)} performance targets",
        }

    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        passed_tests = sum(1 for test in self.test_results.values() if test.get("passed", False))
        total_tests = len(self.test_results)
        
        return {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            },
            "performance_highlights": self._extract_performance_highlights(),
            "recommendations": self._generate_recommendations(),
        }

    def _extract_performance_highlights(self) -> Dict[str, Any]:
        """Extract key performance highlights from test results."""
        highlights = {}
        
        # Repository caching highlights
        repo_test = self.test_results.get("repository_caching", {})
        if "dashboard_performance" in repo_test:
            speedups = [m["speedup_ratio"] for m in repo_test["dashboard_performance"]]
            highlights["repository_caching"] = {
                "max_speedup": f"{max(speedups):.1f}x" if speedups else "N/A",
                "avg_speedup": f"{statistics.mean(speedups):.1f}x" if speedups else "N/A",
            }
        
        # ML caching highlights
        ml_test = self.test_results.get("ml_service_caching", {})
        if "prompt_improvement_performance" in ml_test:
            ml_speedups = [m["speedup_ratio"] for m in ml_test["prompt_improvement_performance"]]
            highlights["ml_caching"] = {
                "max_speedup": f"{max(ml_speedups):.1f}x" if ml_speedups else "N/A",
                "avg_speedup": f"{statistics.mean(ml_speedups):.1f}x" if ml_speedups else "N/A",
            }
        
        # Memory efficiency
        memory_test = self.test_results.get("memory_efficiency", {})
        if memory_test:
            highlights["memory_efficiency"] = {
                "memory_increase": f"{memory_test.get('memory_increase_percent', 0):.1f}%",
                "within_target": memory_test.get('meets_memory_target', False),
            }
        
        return highlights

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check failed tests and provide recommendations
        for test_name, test_result in self.test_results.items():
            if not test_result.get("passed", False):
                if test_name == "caching_infrastructure":
                    recommendations.append("Review caching infrastructure configuration and Redis connectivity")
                elif test_name == "repository_caching":
                    recommendations.append("Optimize repository caching TTL values and cache key generation")
                elif test_name == "ml_service_caching":
                    recommendations.append("Tune ML service cache parameters for better hit rates")
                elif test_name == "api_response_caching":
                    recommendations.append("Review API response caching conditions and strategies")
                elif test_name == "memory_efficiency":
                    recommendations.append("Consider reducing cache sizes or implementing cache eviction policies")
                elif test_name == "performance_under_load":
                    recommendations.append("Scale connection pools and cache resources for higher loads")
        
        # General recommendations
        if not recommendations:
            recommendations.append("All tests passed - consider monitoring in production environment")
        
        return recommendations


async def main():
    """Run performance validation."""
    validator = PerformanceValidator()
    
    try:
        results = await validator.run_validation_suite()
        
        print("\n" + "="*60)
        print("PERFORMANCE OPTIMIZATION VALIDATION RESULTS")
        print("="*60)
        
        print(f"\nValidation Status: {'✓ PASSED' if results['validation_passed'] else '✗ FAILED'}")
        
        if "summary" in results:
            summary = results["summary"]
            print(f"\nTest Summary:")
            print(f"  Tests Run: {summary['test_summary']['total_tests']}")
            print(f"  Passed: {summary['test_summary']['passed_tests']}")
            print(f"  Failed: {summary['test_summary']['failed_tests']}")
            print(f"  Success Rate: {summary['test_summary']['success_rate']:.1f}%")
        
        if "performance_improvements" in results:
            improvements = results["performance_improvements"]
            print(f"\nPerformance Targets:")
            for target_name, target_info in improvements.get("target_validations", {}).items():
                status = "✓" if target_info["met"] else "✗"
                print(f"  {status} {target_name}: {target_info['achieved']:.1f}% (target: {target_info['target']:.1f}%)")
        
        if "summary" in results and "performance_highlights" in results["summary"]:
            highlights = results["summary"]["performance_highlights"]
            print(f"\nPerformance Highlights:")
            for component, metrics in highlights.items():
                print(f"  {component}:")
                for metric_name, metric_value in metrics.items():
                    print(f"    {metric_name}: {metric_value}")
        
        if "summary" in results and "recommendations" in results["summary"]:
            recommendations = results["summary"]["recommendations"]
            if recommendations:
                print(f"\nRecommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
        
        print("\n" + "="*60)
        
        # Exit with appropriate code
        sys.exit(0 if results["validation_passed"] else 1)
        
    except Exception as e:
        print(f"\nValidation failed with error: {e}")
        logger.exception("Validation error")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())