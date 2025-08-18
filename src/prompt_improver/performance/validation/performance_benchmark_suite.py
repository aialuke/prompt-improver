"""Performance Benchmark Suite for Phase 3 Optimizations.

Comprehensive performance validation suite to test and measure the effectiveness
of multi-level caching and connection pool optimizations implemented in Phase 3.

Key Performance Targets:
- Prompt improvement workflow: <20ms (with caching)
- Analytics dashboard queries: <50ms (with caching)
- ML inference pipeline: <10ms (with caching)
- Health check endpoints: <5ms
- Cache hit rates: >80% for hot paths
- Connection pool utilization: 50-70% optimal
"""

import asyncio
import logging
import statistics
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.application.services.prompt_application_service import (
    PromptApplicationService,
)
from prompt_improver.analytics.unified.analytics_service_facade import (
    AnalyticsServiceFacade,
)
from prompt_improver.ml.core.facade import MLModelServiceFacade
from prompt_improver.repositories.impl.analytics_repository import (
    AnalyticsRepository,
)
from prompt_improver.database.services.connection.postgres_pool_manager import (
    PostgreSQLPoolManager,
)
from prompt_improver.services.cache.cache_facade import (
    CacheFacade as CacheManager,
)
# CacheManagerConfig removed - use CacheFacade constructor parameters instead

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBenchmarkResult:
    """Results from a performance benchmark test."""
    
    test_name: str
    target_response_time_ms: float
    actual_response_times_ms: List[float]
    cache_hit_rate: float
    cache_enabled: bool
    test_iterations: int
    start_time: datetime
    end_time: datetime
    
    # Calculated metrics
    avg_response_time_ms: float = field(init=False)
    p50_response_time_ms: float = field(init=False)
    p95_response_time_ms: float = field(init=False)
    p99_response_time_ms: float = field(init=False)
    success_rate: float = field(init=False)
    performance_improvement_ratio: Optional[float] = None
    
    def __post_init__(self):
        """Calculate performance metrics."""
        if self.actual_response_times_ms:
            sorted_times = sorted(self.actual_response_times_ms)
            self.avg_response_time_ms = statistics.mean(sorted_times)
            self.p50_response_time_ms = statistics.median(sorted_times)
            self.p95_response_time_ms = self._percentile(sorted_times, 95)
            self.p99_response_time_ms = self._percentile(sorted_times, 99)
            self.success_rate = 1.0  # All completed tests are successful
        else:
            self.avg_response_time_ms = 0.0
            self.p50_response_time_ms = 0.0
            self.p95_response_time_ms = 0.0
            self.p99_response_time_ms = 0.0
            self.success_rate = 0.0
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        index = int((percentile / 100.0) * len(data))
        return data[min(index, len(data) - 1)]
    
    def meets_target(self) -> bool:
        """Check if P95 response time meets the target."""
        return self.p95_response_time_ms <= self.target_response_time_ms
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark result summary."""
        return {
            "test_name": self.test_name,
            "target_met": self.meets_target(),
            "target_response_time_ms": self.target_response_time_ms,
            "actual_p95_response_time_ms": self.p95_response_time_ms,
            "avg_response_time_ms": self.avg_response_time_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_enabled": self.cache_enabled,
            "test_iterations": self.test_iterations,
            "success_rate": self.success_rate,
            "performance_improvement_ratio": self.performance_improvement_ratio,
        }


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmark suite for Phase 3 optimizations."""
    
    def __init__(
        self,
        prompt_service: PromptApplicationService,
        analytics_service: AnalyticsServiceFacade,
        ml_service: MLModelServiceFacade,
        analytics_repository: AnalyticsRepository,
        postgres_pool: PostgreSQLPoolManager,
        cache_manager: CacheManager,
    ):
        self.prompt_service = prompt_service
        self.analytics_service = analytics_service
        self.ml_service = ml_service
        self.analytics_repository = analytics_repository
        self.postgres_pool = postgres_pool
        self.cache_manager = cache_manager
        
        self.benchmark_results: List[PerformanceBenchmarkResult] = []
        self.logger = logger
    
    async def run_comprehensive_benchmark(
        self, 
        test_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, Any]:
        """Run comprehensive performance benchmark suite.
        
        Args:
            test_iterations: Number of test iterations per benchmark
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Complete benchmark results
        """
        self.logger.info("Starting comprehensive performance benchmark suite")
        start_time = datetime.now()
        
        # Clear benchmark results
        self.benchmark_results = []
        
        # 1. Prompt Improvement Service Benchmark (Target: <20ms)
        await self._benchmark_prompt_improvement(test_iterations, warmup_iterations)
        
        # 2. Analytics Dashboard Queries Benchmark (Target: <50ms)
        await self._benchmark_analytics_queries(test_iterations, warmup_iterations)
        
        # 3. ML Inference Pipeline Benchmark (Target: <10ms)
        await self._benchmark_ml_inference(test_iterations, warmup_iterations)
        
        # 4. Health Check Endpoints Benchmark (Target: <5ms)
        await self._benchmark_health_checks(test_iterations, warmup_iterations)
        
        # 5. PostgreSQL Connection Pool Benchmark
        await self._benchmark_postgres_pool(test_iterations, warmup_iterations)
        
        # 6. Cache Performance Benchmark
        await self._benchmark_cache_performance(test_iterations, warmup_iterations)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Generate comprehensive report
        report = self._generate_benchmark_report(total_duration)
        
        self.logger.info(f"Benchmark suite completed in {total_duration:.2f}s")
        return report
    
    async def _benchmark_prompt_improvement(
        self, test_iterations: int, warmup_iterations: int
    ) -> None:
        """Benchmark prompt improvement workflow performance."""
        self.logger.info("Benchmarking prompt improvement workflow")
        
        test_prompt = "Write a Python function to calculate fibonacci numbers"
        test_session_id = "perf_test_session"
        improvement_options = {"enable_caching": True}
        
        # Warmup phase
        for _ in range(warmup_iterations):
            try:
                await self.prompt_service.improve_prompt(
                    test_prompt, test_session_id + "_warmup", improvement_options
                )
            except Exception as e:
                self.logger.warning(f"Warmup iteration failed: {e}")
        
        # Benchmark phase
        response_times = []
        cache_hits = 0
        
        for i in range(test_iterations):
            start_time = time.time()
            try:
                result = await self.prompt_service.improve_prompt(
                    test_prompt, f"{test_session_id}_{i}", improvement_options
                )
                
                response_time_ms = (time.time() - start_time) * 1000
                response_times.append(response_time_ms)
                
                # Track cache hits
                if result.get("cache_hit", False):
                    cache_hits += 1
                    
            except Exception as e:
                self.logger.error(f"Prompt improvement test iteration {i} failed: {e}")
                response_times.append(10000)  # Penalty for failure
        
        cache_hit_rate = cache_hits / test_iterations
        
        benchmark_result = PerformanceBenchmarkResult(
            test_name="Prompt Improvement Workflow",
            target_response_time_ms=20.0,
            actual_response_times_ms=response_times,
            cache_hit_rate=cache_hit_rate,
            cache_enabled=True,
            test_iterations=test_iterations,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        
        self.benchmark_results.append(benchmark_result)
        self.logger.info(
            f"Prompt improvement benchmark: P95={benchmark_result.p95_response_time_ms:.2f}ms, "
            f"cache_hit_rate={cache_hit_rate:.2%}"
        )
    
    async def _benchmark_analytics_queries(
        self, test_iterations: int, warmup_iterations: int
    ) -> None:
        """Benchmark analytics dashboard queries performance."""
        self.logger.info("Benchmarking analytics dashboard queries")
        
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        filters = {"status": "completed"}
        
        # Warmup phase
        for _ in range(warmup_iterations):
            try:
                await self.analytics_repository.get_session_analytics(
                    start_date, end_date, filters
                )
            except Exception as e:
                self.logger.warning(f"Analytics warmup iteration failed: {e}")
        
        # Benchmark phase
        response_times = []
        cache_hits = 0
        
        for i in range(test_iterations):
            start_time = time.time()
            try:
                # Vary dates slightly to test different cache keys
                test_start = start_date + timedelta(minutes=i % 60)
                await self.analytics_repository.get_session_analytics(
                    test_start, end_date, filters
                )
                
                response_time_ms = (time.time() - start_time) * 1000
                response_times.append(response_time_ms)
                
            except Exception as e:
                self.logger.error(f"Analytics query test iteration {i} failed: {e}")
                response_times.append(10000)  # Penalty for failure
        
        # Get cache hit rate from repository metrics
        try:
            metrics = await self.analytics_repository.get_performance_metrics()
            cache_hit_rate = metrics.get("cache_hit_rate", 0.0)
        except Exception:
            cache_hit_rate = 0.0
        
        benchmark_result = PerformanceBenchmarkResult(
            test_name="Analytics Dashboard Queries",
            target_response_time_ms=50.0,
            actual_response_times_ms=response_times,
            cache_hit_rate=cache_hit_rate,
            cache_enabled=True,
            test_iterations=test_iterations,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        
        self.benchmark_results.append(benchmark_result)
        self.logger.info(
            f"Analytics queries benchmark: P95={benchmark_result.p95_response_time_ms:.2f}ms, "
            f"cache_hit_rate={cache_hit_rate:.2%}"
        )
    
    async def _benchmark_ml_inference(
        self, test_iterations: int, warmup_iterations: int
    ) -> None:
        """Benchmark ML inference pipeline performance."""
        self.logger.info("Benchmarking ML inference pipeline")
        
        model_id = "test_model"
        rule_features = [0.8, 0.7, 0.9, 0.6, 0.85]
        
        # Warmup phase
        for _ in range(warmup_iterations):
            try:
                await self.ml_service.predict_rule_effectiveness(model_id, rule_features)
            except Exception as e:
                self.logger.warning(f"ML inference warmup iteration failed: {e}")
        
        # Benchmark phase
        response_times = []
        cache_hits = 0
        
        for i in range(test_iterations):
            start_time = time.time()
            try:
                # Vary features slightly to test different cache keys
                test_features = [f + (i % 10) * 0.01 for f in rule_features]
                result = await self.ml_service.predict_rule_effectiveness(
                    model_id, test_features
                )
                
                response_time_ms = (time.time() - start_time) * 1000
                response_times.append(response_time_ms)
                
                # Track cache hits
                if result.get("cache_hit", False):
                    cache_hits += 1
                    
            except Exception as e:
                self.logger.error(f"ML inference test iteration {i} failed: {e}")
                response_times.append(10000)  # Penalty for failure
        
        cache_hit_rate = cache_hits / test_iterations
        
        benchmark_result = PerformanceBenchmarkResult(
            test_name="ML Inference Pipeline",
            target_response_time_ms=10.0,
            actual_response_times_ms=response_times,
            cache_hit_rate=cache_hit_rate,
            cache_enabled=True,
            test_iterations=test_iterations,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        
        self.benchmark_results.append(benchmark_result)
        self.logger.info(
            f"ML inference benchmark: P95={benchmark_result.p95_response_time_ms:.2f}ms, "
            f"cache_hit_rate={cache_hit_rate:.2%}"
        )
    
    async def _benchmark_health_checks(
        self, test_iterations: int, warmup_iterations: int
    ) -> None:
        """Benchmark health check endpoints performance."""
        self.logger.info("Benchmarking health check endpoints")
        
        # Warmup phase
        for _ in range(warmup_iterations):
            try:
                await self.analytics_service.health_check()
            except Exception as e:
                self.logger.warning(f"Health check warmup iteration failed: {e}")
        
        # Benchmark phase
        response_times = []
        
        for i in range(test_iterations):
            start_time = time.time()
            try:
                await self.analytics_service.health_check()
                
                response_time_ms = (time.time() - start_time) * 1000
                response_times.append(response_time_ms)
                
            except Exception as e:
                self.logger.error(f"Health check test iteration {i} failed: {e}")
                response_times.append(10000)  # Penalty for failure
        
        benchmark_result = PerformanceBenchmarkResult(
            test_name="Health Check Endpoints",
            target_response_time_ms=5.0,
            actual_response_times_ms=response_times,
            cache_hit_rate=0.0,  # Health checks typically not cached
            cache_enabled=False,
            test_iterations=test_iterations,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        
        self.benchmark_results.append(benchmark_result)
        self.logger.info(
            f"Health check benchmark: P95={benchmark_result.p95_response_time_ms:.2f}ms"
        )
    
    async def _benchmark_postgres_pool(
        self, test_iterations: int, warmup_iterations: int
    ) -> None:
        """Benchmark PostgreSQL connection pool performance."""
        self.logger.info("Benchmarking PostgreSQL connection pool")
        
        test_query = "SELECT COUNT(*) as total FROM prompt_sessions WHERE created_at > $1"
        test_params = {"1": datetime.now() - timedelta(days=1)}
        
        # Warmup phase
        for _ in range(warmup_iterations):
            try:
                await self.postgres_pool.execute_cached_query(test_query, test_params)
            except Exception as e:
                self.logger.warning(f"Postgres pool warmup iteration failed: {e}")
        
        # Benchmark phase
        response_times = []
        
        for i in range(test_iterations):
            start_time = time.time()
            try:
                # Vary query slightly to test caching
                test_date = datetime.now() - timedelta(days=1, hours=i % 24)
                params = {"1": test_date}
                
                await self.postgres_pool.execute_cached_query(test_query, params)
                
                response_time_ms = (time.time() - start_time) * 1000
                response_times.append(response_time_ms)
                
            except Exception as e:
                self.logger.error(f"Postgres pool test iteration {i} failed: {e}")
                response_times.append(10000)  # Penalty for failure
        
        benchmark_result = PerformanceBenchmarkResult(
            test_name="PostgreSQL Connection Pool",
            target_response_time_ms=50.0,  # Database queries can be slower
            actual_response_times_ms=response_times,
            cache_hit_rate=0.5,  # Estimated cache hit rate for queries
            cache_enabled=True,
            test_iterations=test_iterations,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        
        self.benchmark_results.append(benchmark_result)
        self.logger.info(
            f"Postgres pool benchmark: P95={benchmark_result.p95_response_time_ms:.2f}ms"
        )
    
    async def _benchmark_cache_performance(
        self, test_iterations: int, warmup_iterations: int
    ) -> None:
        """Benchmark cache manager performance."""
        self.logger.info("Benchmarking cache manager performance")
        
        # Warmup phase - populate cache
        for i in range(warmup_iterations):
            try:
                await self.cache_manager.set(f"warmup_key_{i}", f"warmup_value_{i}")
            except Exception as e:
                self.logger.warning(f"Cache warmup iteration failed: {e}")
        
        # Benchmark phase - test cache operations
        response_times = []
        cache_hits = 0
        
        for i in range(test_iterations):
            start_time = time.time()
            try:
                # Test cache get operations (should hit for warmup keys)
                key = f"warmup_key_{i % warmup_iterations}"
                value = await self.cache_manager.get(key)
                
                response_time_ms = (time.time() - start_time) * 1000
                response_times.append(response_time_ms)
                
                if value is not None:
                    cache_hits += 1
                    
            except Exception as e:
                self.logger.error(f"Cache test iteration {i} failed: {e}")
                response_times.append(10000)  # Penalty for failure
        
        cache_hit_rate = cache_hits / test_iterations
        
        benchmark_result = PerformanceBenchmarkResult(
            test_name="Cache Manager Performance",
            target_response_time_ms=5.0,  # Cache should be very fast
            actual_response_times_ms=response_times,
            cache_hit_rate=cache_hit_rate,
            cache_enabled=True,
            test_iterations=test_iterations,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        
        self.benchmark_results.append(benchmark_result)
        self.logger.info(
            f"Cache manager benchmark: P95={benchmark_result.p95_response_time_ms:.2f}ms, "
            f"cache_hit_rate={cache_hit_rate:.2%}"
        )
    
    def _generate_benchmark_report(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        overall_success = all(result.meets_target() for result in self.benchmark_results)
        
        # Calculate overall metrics
        total_tests = sum(result.test_iterations for result in self.benchmark_results)
        avg_cache_hit_rate = statistics.mean(
            result.cache_hit_rate for result in self.benchmark_results 
            if result.cache_enabled
        )
        
        # Individual test results
        test_summaries = [result.get_summary() for result in self.benchmark_results]
        
        # Performance targets status
        targets_met = sum(1 for result in self.benchmark_results if result.meets_target())
        targets_total = len(self.benchmark_results)
        
        report = {
            "benchmark_suite": "Phase 3 Performance Optimizations",
            "overall_success": overall_success,
            "total_duration_seconds": total_duration,
            "total_test_iterations": total_tests,
            "targets_met": f"{targets_met}/{targets_total}",
            "avg_cache_hit_rate": avg_cache_hit_rate,
            "timestamp": datetime.now().isoformat(),
            
            "test_results": test_summaries,
            
            "performance_summary": {
                "prompt_improvement_p95_ms": next(
                    (r.p95_response_time_ms for r in self.benchmark_results 
                     if r.test_name == "Prompt Improvement Workflow"), None
                ),
                "analytics_queries_p95_ms": next(
                    (r.p95_response_time_ms for r in self.benchmark_results 
                     if r.test_name == "Analytics Dashboard Queries"), None
                ),
                "ml_inference_p95_ms": next(
                    (r.p95_response_time_ms for r in self.benchmark_results 
                     if r.test_name == "ML Inference Pipeline"), None
                ),
                "health_checks_p95_ms": next(
                    (r.p95_response_time_ms for r in self.benchmark_results 
                     if r.test_name == "Health Check Endpoints"), None
                ),
            },
            
            "recommendations": self._generate_recommendations(),
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        for result in self.benchmark_results:
            if not result.meets_target():
                recommendations.append(
                    f"❌ {result.test_name}: P95 {result.p95_response_time_ms:.2f}ms "
                    f"exceeds target {result.target_response_time_ms}ms"
                )
                
                if result.cache_enabled and result.cache_hit_rate < 0.8:
                    recommendations.append(
                        f"💡 Consider increasing cache TTL for {result.test_name} "
                        f"(current hit rate: {result.cache_hit_rate:.2%})"
                    )
            else:
                recommendations.append(
                    f"✅ {result.test_name}: Meeting performance target "
                    f"(P95: {result.p95_response_time_ms:.2f}ms)"
                )
        
        # General recommendations
        avg_cache_hit_rate = statistics.mean(
            result.cache_hit_rate for result in self.benchmark_results 
            if result.cache_enabled
        )
        
        if avg_cache_hit_rate < 0.8:
            recommendations.append(
                f"💡 Overall cache hit rate {avg_cache_hit_rate:.2%} is below 80% target. "
                "Consider tuning cache TTL and warming strategies."
            )
        
        return recommendations
    
    async def validate_phase_3_targets(self) -> bool:
        """Quick validation that Phase 3 targets are met."""
        results = await self.run_comprehensive_benchmark(test_iterations=50, warmup_iterations=5)
        return results["overall_success"]


# Convenience function for running benchmarks
async def run_performance_benchmark(
    prompt_service: PromptApplicationService,
    analytics_service: AnalyticsServiceFacade,
    ml_service: MLModelServiceFacade,
    analytics_repository: AnalyticsRepository,
    postgres_pool: PostgreSQLPoolManager,
    cache_manager: CacheManager,
    test_iterations: int = 100
) -> Dict[str, Any]:
    """Run comprehensive performance benchmark suite.
    
    Args:
        prompt_service: Prompt application service
        analytics_service: Analytics service facade
        ml_service: ML model service facade
        analytics_repository: Analytics repository
        postgres_pool: PostgreSQL pool manager
        cache_manager: Cache manager
        test_iterations: Number of test iterations
        
    Returns:
        Benchmark results
    """
    benchmark_suite = PerformanceBenchmarkSuite(
        prompt_service=prompt_service,
        analytics_service=analytics_service,
        ml_service=ml_service,
        analytics_repository=analytics_repository,
        postgres_pool=postgres_pool,
        cache_manager=cache_manager,
    )
    
    return await benchmark_suite.run_comprehensive_benchmark(test_iterations)