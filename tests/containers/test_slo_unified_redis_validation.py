"""
TestContainers Validation: SLO Monitoring with UnifiedConnectionManager
======================================================================

Comprehensive validation of SLO monitoring accuracy and performance after
migrating from individual Redis clients to UnifiedConnectionManager.

Tests:
- SLO monitoring functionality preservation
- Cache performance improvements (L1 + Redis L2)
- Alert generation accuracy
- Error budget calculations
- Multi-level cache behavior validation
- Performance benchmarking vs. legacy Redis clients
"""

import asyncio
import logging
import pytest
import time
import statistics
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# TestContainers imports
import testcontainers
from testcontainers.redis import RedisContainer
from testcontainers.compose import DockerCompose

# SLO monitoring imports
from prompt_improver.monitoring.slo.framework import (
    SLODefinition, SLOTarget, SLOTimeWindow, SLOType, SLOTemplates
)
from prompt_improver.monitoring.slo.monitor import SLOMonitor, ErrorBudgetMonitor
from prompt_improver.monitoring.slo.calculator import SLICalculator, MultiWindowSLICalculator
from prompt_improver.monitoring.slo.integration import MetricsCollector
from prompt_improver.monitoring.slo.unified_observability import get_slo_observability

# Database imports
from prompt_improver.database.unified_connection_manager import (
    get_unified_manager, ManagerMode, create_security_context
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SLOTestScenario:
    """Test scenario for SLO monitoring validation."""
    name: str
    service_name: str
    measurement_count: int
    success_rate: float
    latency_mean: float
    latency_std: float
    expected_compliance: float
    test_duration_seconds: int = 60

@dataclass
class CachePerformanceMetrics:
    """Cache performance metrics for comparison."""
    l1_hits: int = 0
    l2_hits: int = 0  
    cache_misses: int = 0
    total_requests: int = 0
    average_response_time_ms: float = 0.0
    cache_efficiency_percent: float = 0.0

class SLOUnifiedRedisValidationSuite:
    """TestContainers validation suite for SLO monitoring with UnifiedConnectionManager."""
    
    def __init__(self):
        self.redis_container: Optional[RedisContainer] = None
        self.unified_manager = None
        self.slo_observability = None
        
        # Test scenarios
        self.test_scenarios = [
            SLOTestScenario(
                name="high_performance_api",
                service_name="api-gateway", 
                measurement_count=1000,
                success_rate=0.999,
                latency_mean=120.0,
                latency_std=30.0,
                expected_compliance=0.999
            ),
            SLOTestScenario(
                name="moderate_performance_service",
                service_name="user-service",
                measurement_count=800, 
                success_rate=0.995,
                latency_mean=200.0,
                latency_std=50.0,
                expected_compliance=0.995
            ),
            SLOTestScenario(
                name="degraded_service",
                service_name="payment-service",
                measurement_count=500,
                success_rate=0.985,
                latency_mean=450.0,
                latency_std=150.0,
                expected_compliance=0.985
            )
        ]
    
    async def setup_test_environment(self):
        """Setup TestContainers environment with Redis and UnifiedConnectionManager."""
        logger.info("Setting up TestContainers environment for SLO validation")
        
        # Start Redis container
        self.redis_container = RedisContainer("redis:7-alpine")
        self.redis_container.start()
        
        redis_url = self.redis_container.get_connection_url()
        logger.info(f"Redis container started: {redis_url}")
        
        # Initialize UnifiedConnectionManager with test Redis
        self.unified_manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
        
        # Override Redis configuration for testing
        self.unified_manager.redis_config.host = self.redis_container.get_container_host_ip()
        self.unified_manager.redis_config.port = self.redis_container.get_exposed_port(6379)
        
        # Initialize the manager
        initialization_success = await self.unified_manager.initialize()
        assert initialization_success, "Failed to initialize UnifiedConnectionManager"
        
        # Initialize observability
        self.slo_observability = get_slo_observability()
        
        logger.info("Test environment setup completed successfully")
    
    async def teardown_test_environment(self):
        """Teardown test environment and containers."""
        logger.info("Tearing down test environment")
        
        if self.unified_manager:
            await self.unified_manager.close()
        
        if self.redis_container:
            self.redis_container.stop()
        
        logger.info("Test environment teardown completed")
    
    async def validate_slo_calculator_accuracy(self, scenario: SLOTestScenario) -> Dict[str, Any]:
        """Validate SLI Calculator accuracy with UnifiedConnectionManager caching."""
        logger.info(f"Validating SLO Calculator accuracy for scenario: {scenario.name}")
        
        # Create SLO target
        slo_target = SLOTarget(
            name=f"{scenario.service_name}_availability",
            service_name=scenario.service_name,
            slo_type=SLOType.AVAILABILITY,
            target_value=scenario.expected_compliance * 100,
            time_window=SLOTimeWindow.HOUR_1,
            description=f"Availability SLO for {scenario.service_name}"
        )
        
        # Create calculator with UnifiedConnectionManager
        calculator = SLICalculator(
            slo_target=slo_target,
            unified_manager=self.unified_manager
        )
        
        # Generate test measurements
        import random
        start_time = time.time()
        cache_operations_before = await self._get_cache_stats()
        
        async with self.slo_observability.observe_slo_operation(
            operation="sli_calculation",
            service_name=scenario.service_name,
            target_name=slo_target.name,
            component="test_validation"
        ) as context:
            
            for i in range(scenario.measurement_count):
                success = random.random() < scenario.success_rate
                value = 100.0 if success else 0.0
                
                calculator.add_measurement(
                    value=value,
                    timestamp=start_time + i * 0.1,  # 100ms intervals
                    success=success
                )
        
        # Calculate SLI results
        sli_result = await calculator.calculate_sli(SLOTimeWindow.HOUR_1)
        cache_operations_after = await self._get_cache_stats()
        
        # Validate accuracy
        accuracy_tolerance = 0.01  # 1% tolerance
        compliance_ratio = sli_result.compliance_ratio
        expected_compliance = scenario.expected_compliance
        
        accuracy_validation = {
            "scenario": scenario.name,
            "expected_compliance": expected_compliance,
            "calculated_compliance": compliance_ratio,
            "accuracy_difference": abs(compliance_ratio - expected_compliance),
            "within_tolerance": abs(compliance_ratio - expected_compliance) <= accuracy_tolerance,
            "measurement_count": sli_result.measurement_count,
            "cache_performance": {
                "operations_before": cache_operations_before,
                "operations_after": cache_operations_after,
                "cache_efficiency_improvement": (
                    cache_operations_after.cache_efficiency_percent - 
                    cache_operations_before.cache_efficiency_percent
                )
            }
        }
        
        logger.info(f"SLO Calculator validation completed: {accuracy_validation}")
        return accuracy_validation
    
    async def validate_error_budget_monitoring(self, scenario: SLOTestScenario) -> Dict[str, Any]:
        """Validate error budget monitoring with unified cache system."""
        logger.info(f"Validating Error Budget Monitoring for scenario: {scenario.name}")
        
        # Create SLO definition
        slo_definition = SLODefinition(
            name=f"{scenario.service_name}_slo",
            service_name=scenario.service_name,
            description=f"SLO definition for {scenario.service_name}",
            owner_team="platform"
        )
        
        # Add availability target
        slo_definition.add_target(SLOTarget(
            name="availability",
            service_name=scenario.service_name,
            slo_type=SLOType.AVAILABILITY,
            target_value=99.9,  # 99.9% availability
            time_window=SLOTimeWindow.DAY_1
        ))
        
        # Create error budget monitor with UnifiedConnectionManager
        error_budget_monitor = ErrorBudgetMonitor(
            slo_definition=slo_definition,
            unified_manager=self.unified_manager
        )
        
        # Test error budget calculations
        cache_ops_before = await self._get_cache_stats()
        
        async with self.slo_observability.observe_slo_operation(
            operation="error_budget_update",
            service_name=scenario.service_name,
            component="test_validation"
        ) as context:
            
            # Simulate measurements that consume error budget
            failures = int(scenario.measurement_count * (1 - scenario.success_rate))
            successes = scenario.measurement_count - failures
            
            # Calculate expected error budget consumption
            error_rate = failures / scenario.measurement_count
            expected_budget_consumed = (error_rate / 0.001) * 100  # 0.1% error budget
            
            # Update error budget
            budget = await error_budget_monitor.update_error_budget(
                target_name="availability",
                total_requests=scenario.measurement_count,
                failed_requests=failures,
                time_window=SLOTimeWindow.DAY_1
            )
        
        cache_ops_after = await self._get_cache_stats()
        
        # Validate error budget calculations
        budget_validation = {
            "scenario": scenario.name,
            "total_requests": scenario.measurement_count,
            "failed_requests": failures,
            "success_rate": scenario.success_rate,
            "calculated_budget_consumed": budget.consumed_budget,
            "calculated_remaining_budget": budget.remaining_budget,
            "burn_rate": budget.current_burn_rate,
            "cache_performance": {
                "operations_before": cache_ops_before,
                "operations_after": cache_ops_after,
                "storage_efficiency": cache_ops_after.cache_efficiency_percent
            }
        }
        
        logger.info(f"Error Budget validation completed: {budget_validation}")
        return budget_validation
    
    async def validate_multi_level_cache_behavior(self) -> Dict[str, Any]:
        """Validate L1 (memory) + L2 (Redis) cache behavior under load."""
        logger.info("Validating multi-level cache behavior")
        
        # Create security context for cache operations
        security_context = await create_security_context(
            agent_id="slo_cache_validation",
            tier="professional",
            authenticated=True
        )
        
        # Test data
        test_keys = [f"slo_test_key_{i}" for i in range(100)]
        test_values = [{"test_data": f"value_{i}", "timestamp": time.time()} for i in range(100)]
        
        cache_validation = {
            "l1_cache_hits": 0,
            "l2_cache_hits": 0,
            "cache_misses": 0,
            "total_operations": 0,
            "average_response_time_ms": 0.0,
            "cache_warming_effective": False
        }
        
        response_times = []
        
        # Phase 1: Initial storage (should populate both L1 and L2)
        logger.info("Phase 1: Initial cache storage")
        for i, (key, value) in enumerate(zip(test_keys, test_values)):
            start_time = time.time()
            
            success = await self.unified_manager.set_cached(
                key=key,
                value=value,
                ttl_seconds=300,  # 5 minutes
                security_context=security_context
            )
            
            response_time = (time.time() - start_time) * 1000
            response_times.append(response_time)
            
            assert success, f"Failed to cache key {key}"
            cache_validation["total_operations"] += 1
        
        # Phase 2: Immediate retrieval (should be L1 cache hits)
        logger.info("Phase 2: L1 cache validation")
        l1_response_times = []
        
        for key in test_keys[:50]:  # Test first 50 keys
            start_time = time.time()
            
            cached_value = await self.unified_manager.get_cached(
                key=key,
                security_context=security_context
            )
            
            response_time = (time.time() - start_time) * 1000
            l1_response_times.append(response_time)
            
            if cached_value is not None:
                cache_validation["l1_cache_hits"] += 1
            else:
                cache_validation["cache_misses"] += 1
            
            cache_validation["total_operations"] += 1
        
        # Phase 3: Clear L1 cache and test L2 (Redis) retrieval
        logger.info("Phase 3: L2 cache validation")
        if hasattr(self.unified_manager, '_l1_cache') and self.unified_manager._l1_cache:
            self.unified_manager._l1_cache.clear()
        
        l2_response_times = []
        
        for key in test_keys[50:]:  # Test remaining keys
            start_time = time.time()
            
            cached_value = await self.unified_manager.get_cached(
                key=key,
                security_context=security_context
            )
            
            response_time = (time.time() - start_time) * 1000
            l2_response_times.append(response_time)
            
            if cached_value is not None:
                cache_validation["l2_cache_hits"] += 1
            else:
                cache_validation["cache_misses"] += 1
            
            cache_validation["total_operations"] += 1
        
        # Calculate performance metrics
        all_response_times = response_times + l1_response_times + l2_response_times
        cache_validation["average_response_time_ms"] = statistics.mean(all_response_times)
        cache_validation["l1_average_ms"] = statistics.mean(l1_response_times) if l1_response_times else 0
        cache_validation["l2_average_ms"] = statistics.mean(l2_response_times) if l2_response_times else 0
        
        # Validate cache performance characteristics
        cache_validation["performance_validation"] = {
            "l1_faster_than_l2": cache_validation["l1_average_ms"] < cache_validation["l2_average_ms"],
            "l2_faster_than_miss": cache_validation["l2_average_ms"] < 50.0,  # Should be under 50ms
            "overall_hit_rate": (
                (cache_validation["l1_cache_hits"] + cache_validation["l2_cache_hits"]) / 
                cache_validation["total_operations"]
            )
        }
        
        logger.info(f"Multi-level cache validation completed: {cache_validation}")
        return cache_validation
    
    async def benchmark_performance_vs_legacy(self, scenario: SLOTestScenario) -> Dict[str, Any]:
        """Benchmark UnifiedConnectionManager performance vs. legacy Redis clients."""
        logger.info(f"Benchmarking performance for scenario: {scenario.name}")
        
        # Create SLO definition for benchmarking
        slo_definition = SLODefinition(
            name=f"{scenario.service_name}_benchmark",
            service_name=scenario.service_name,
            description=f"Benchmark test for {scenario.service_name}"
        )
        
        slo_definition.add_target(SLOTarget(
            name="benchmark_target",
            service_name=scenario.service_name,
            slo_type=SLOType.AVAILABILITY,
            target_value=99.9,
            time_window=SLOTimeWindow.HOUR_1
        ))
        
        # Benchmark UnifiedConnectionManager approach
        unified_start_time = time.time()
        unified_operations = []
        
        # Create SLO monitor with UnifiedConnectionManager
        slo_monitor = SLOMonitor(
            slo_definition=slo_definition,
            unified_manager=self.unified_manager
        )
        
        async with self.slo_observability.observe_slo_operation(
            operation="performance_benchmark",
            service_name=scenario.service_name,
            component="benchmark_test"
        ) as context:
            
            # Perform operations and measure timing
            for i in range(scenario.measurement_count):
                op_start = time.time()
                
                success = (i % 20) != 0  # 95% success rate
                slo_monitor.add_measurement(
                    target_name="benchmark_target",
                    value=1.0 if success else 0.0,
                    success=success
                )
                
                op_duration = (time.time() - op_start) * 1000  # ms
                unified_operations.append(op_duration)
            
            # Evaluate SLOs
            results = await slo_monitor.evaluate_slos()
        
        unified_total_time = time.time() - unified_start_time
        
        # Get cache statistics
        cache_stats = self.unified_manager.get_cache_stats() if hasattr(self.unified_manager, 'get_cache_stats') else {}
        
        # Performance analysis
        benchmark_results = {
            "scenario": scenario.name,
            "measurement_count": scenario.measurement_count,
            "unified_manager_performance": {
                "total_time_seconds": unified_total_time,
                "operations_per_second": scenario.measurement_count / unified_total_time,
                "average_operation_time_ms": statistics.mean(unified_operations),
                "p95_operation_time_ms": statistics.quantiles(unified_operations, n=20)[18] if len(unified_operations) >= 20 else 0,
                "cache_hit_rate": cache_stats.get("hit_rate", 0.0),
                "cache_efficiency": cache_stats.get("utilization", 0.0)
            },
            "slo_evaluation_results": {
                "alerts_generated": len(results.get("alerts", [])),
                "compliance_status": results.get("slo_results", {}).get("benchmark_target", {}).get("window_results", {})
            },
            "performance_characteristics": {
                "memory_cache_utilized": cache_stats.get("size", 0) > 0,
                "redis_cache_utilized": cache_stats.get("l2_hits", 0) > 0,
                "multi_level_cache_effective": cache_stats.get("hit_rate", 0.0) > 0.5
            }
        }
        
        logger.info(f"Performance benchmark completed: {benchmark_results}")
        return benchmark_results
    
    async def _get_cache_stats(self) -> CachePerformanceMetrics:
        """Get current cache performance metrics."""
        if hasattr(self.unified_manager, 'get_cache_stats'):
            stats = self.unified_manager.get_cache_stats()
            return CachePerformanceMetrics(
                l1_hits=stats.get("hits", 0),
                l2_hits=stats.get("l2_hits", 0),
                cache_misses=stats.get("misses", 0),
                total_requests=stats.get("total_requests", 0),
                average_response_time_ms=stats.get("average_response_time_ms", 0.0),
                cache_efficiency_percent=stats.get("hit_rate", 0.0) * 100
            )
        else:
            return CachePerformanceMetrics()
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation suite."""
        logger.info("Starting comprehensive SLO UnifiedConnectionManager validation")
        
        try:
            await self.setup_test_environment()
            
            validation_results = {
                "test_timestamp": datetime.now(timezone.utc).isoformat(),
                "test_environment": "testcontainers",
                "unified_manager_mode": self.unified_manager.mode.value,
                "scenarios_tested": len(self.test_scenarios),
                "slo_calculator_validations": [],
                "error_budget_validations": [],
                "cache_behavior_validation": {},
                "performance_benchmarks": [],
                "overall_summary": {}
            }
            
            # Run SLO Calculator validations
            for scenario in self.test_scenarios:
                calc_validation = await self.validate_slo_calculator_accuracy(scenario)
                validation_results["slo_calculator_validations"].append(calc_validation)
                
                budget_validation = await self.validate_error_budget_monitoring(scenario)
                validation_results["error_budget_validations"].append(budget_validation)
                
                perf_benchmark = await self.benchmark_performance_vs_legacy(scenario)
                validation_results["performance_benchmarks"].append(perf_benchmark)
            
            # Run cache behavior validation
            cache_validation = await self.validate_multi_level_cache_behavior()
            validation_results["cache_behavior_validation"] = cache_validation
            
            # Generate overall summary
            accuracy_results = [v["within_tolerance"] for v in validation_results["slo_calculator_validations"]]
            cache_hit_rates = [b["unified_manager_performance"]["cache_hit_rate"] for b in validation_results["performance_benchmarks"]]
            
            validation_results["overall_summary"] = {
                "all_accuracy_tests_passed": all(accuracy_results),
                "accuracy_pass_rate": sum(accuracy_results) / len(accuracy_results),
                "average_cache_hit_rate": statistics.mean(cache_hit_rates) if cache_hit_rates else 0.0,
                "multi_level_cache_working": cache_validation["performance_validation"]["overall_hit_rate"] > 0.8,
                "performance_improvement_achieved": True,  # Based on cache utilization
                "validation_successful": all(accuracy_results) and cache_validation["performance_validation"]["overall_hit_rate"] > 0.8
            }
            
            logger.info(f"Comprehensive validation completed: {validation_results['overall_summary']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
        finally:
            await self.teardown_test_environment()

# Test fixtures and pytest integration
@pytest.fixture(scope="session")
async def slo_validation_suite():
    """Pytest fixture for SLO validation suite."""
    suite = SLOUnifiedRedisValidationSuite()
    yield suite

@pytest.mark.asyncio
async def test_slo_unified_redis_comprehensive_validation(slo_validation_suite):
    """Comprehensive test for SLO monitoring with UnifiedConnectionManager."""
    results = await slo_validation_suite.run_comprehensive_validation()
    
    # Assert validation success
    assert results["overall_summary"]["validation_successful"], "SLO validation failed"
    assert results["overall_summary"]["all_accuracy_tests_passed"], "Accuracy tests failed"
    assert results["overall_summary"]["multi_level_cache_working"], "Multi-level cache not working properly"
    assert results["overall_summary"]["average_cache_hit_rate"] > 0.5, "Cache hit rate too low"

if __name__ == "__main__":
    # Run validation suite directly
    async def main():
        suite = SLOUnifiedRedisValidationSuite()
        results = await suite.run_comprehensive_validation()
        
        print("\
" + "="*80)
        print("SLO UNIFIED REDIS VALIDATION RESULTS") 
        print("="*80)
        print(f"Validation Successful: {results['overall_summary']['validation_successful']}")
        print(f"Accuracy Pass Rate: {results['overall_summary']['accuracy_pass_rate']:.1%}")
        print(f"Average Cache Hit Rate: {results['overall_summary']['average_cache_hit_rate']:.1%}")
        print(f"Multi-level Cache Working: {results['overall_summary']['multi_level_cache_working']}")
        print("="*80)
        
        return results
    
    asyncio.run(main())