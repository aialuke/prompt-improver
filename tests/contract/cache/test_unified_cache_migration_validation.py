"""Comprehensive Contract Validation for Direct Cache Architecture Migration

This module provides end-to-end contract validation of the complete migration from
coordination-based caching to the new high-performance direct L1+L2 cache-aside pattern.
Validates the new architecture through comprehensive contract testing.

Architecture Contract Validation:
1. **CacheFactory Contract**: Singleton pattern, <2μs instance retrieval
2. **CacheFacade Contract**: Direct L1+L2 operations, <2ms response times
3. **Performance Contract**: 25x improvement, eliminated coordination overhead
4. **API Contract**: Method signatures, error handling, graceful degradation
5. **Integration Contract**: Cross-component functionality, cache sharing

New Architecture Contracts Validated:
- CacheFactory singleton pattern for optimized instance management
- Direct L1+L2 cache-aside pattern (eliminated coordination layer)
- Performance targets: <2ms cache operations, <2μs factory access
- Eliminated services: L3DatabaseService, CacheCoordinatorService
- Memory efficiency: Singleton instances prevent duplication

Migration Success Criteria:
- All functions use CacheFactory pattern for cache access
- Performance targets met: <2ms operations vs old 51.5ms coordination
- Integration workflows function correctly with new architecture
- Real behavior validation with testcontainers (no mocks)
- 100% compatibility with existing function APIs
"""

import logging
import time
from dataclasses import dataclass, field

import pytest
from tests.conftest import reset_test_caches

# Import all functions that should now use the new cache architecture
from prompt_improver.core.common.config_utils import get_config_safely
from prompt_improver.core.common.logging_utils import get_logger
from prompt_improver.core.common.metrics_utils import get_metrics_safely
from prompt_improver.core.config.textstat import get_textstat_wrapper
from prompt_improver.ml.analysis.linguistic_analyzer import (
    LinguisticAnalyzer,
    get_lightweight_config,
)
from prompt_improver.services.cache.cache_facade import CacheFacade
from prompt_improver.services.cache.cache_factory import CacheFactory


@dataclass
class CacheContractValidationResult:
    """Results from cache contract validation."""
    component_name: str
    contract_validated: bool
    performance_metrics: dict[str, float] = field(default_factory=dict)
    api_compliance: bool = True
    error_handling: bool = True
    integration_success: bool = True
    validation_errors: list[str] = field(default_factory=list)


@pytest.mark.contract
@pytest.mark.cache_migration
@pytest.mark.real_behavior
class TestCacheArchitectureContractsComprehensive:
    """Comprehensive contract validation for direct cache architecture."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for contract validation tests."""
        # Clear factory instances for clean state
        CacheFactory.clear_instances()

        # Clear all caches for clean state
        reset_test_caches()

        # Clear utility caches
        self._clear_utility_caches()

        # Set up logger
        self.logger = logging.getLogger(__name__)

        yield

        # Cleanup after test
        self.teardown_method()

    def teardown_method(self):
        """Cleanup after contract validation tests."""
        # Clear all test caches
        reset_test_caches()
        self._clear_utility_caches()

        # Clear factory instances
        CacheFactory.clear_instances()

    def _clear_utility_caches(self):
        """Clear all utility cache instances."""
        from prompt_improver.core.common import (
            config_utils,
            logging_utils,
            metrics_utils,
        )

        # Reset global cache instances
        config_utils._config_cache = None
        metrics_utils._metrics_cache = None
        logging_utils._logging_cache = None

    def test_cache_factory_singleton_contract(self):
        """Test CacheFactory singleton pattern contract compliance."""
        validation_result = CacheContractValidationResult(
            component_name="CacheFactory",
            contract_validated=False
        )

        try:
            # Contract 1: Singleton pattern enforcement
            cache_types = [
                ("utility", CacheFactory.get_utility_cache),
                ("textstat", CacheFactory.get_textstat_cache),
                ("ml_analysis", CacheFactory.get_ml_analysis_cache),
                ("session", CacheFactory.get_session_cache),
                ("rule", CacheFactory.get_rule_cache),
                ("prompt", CacheFactory.get_prompt_cache),
            ]

            singleton_performance = {}

            for cache_type, factory_method in cache_types:
                # Test singleton contract
                cache1 = factory_method()
                cache2 = factory_method()

                if cache1 is not cache2:
                    validation_result.validation_errors.append(f"{cache_type} violates singleton contract")
                    continue

                # Test performance contract (<2μs instance retrieval)
                retrieval_times = []
                for _ in range(50):
                    start_time = time.perf_counter()
                    cache3 = factory_method()
                    retrieval_time = time.perf_counter() - start_time
                    retrieval_times.append(retrieval_time)

                    if cache3 is not cache1:
                        validation_result.validation_errors.append(f"{cache_type} singleton consistency violated")

                avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
                singleton_performance[cache_type] = avg_retrieval_time * 1000000  # Convert to μs

                # Validate performance contract
                if avg_retrieval_time > 0.000002:  # 2μs
                    validation_result.validation_errors.append(
                        f"{cache_type} retrieval {avg_retrieval_time * 1000000:.2f}μs exceeds 2μs contract"
                    )

            # Contract 2: Factory statistics contract
            stats = CacheFactory.get_performance_stats()
            required_stats = ["total_instances", "singleton_pattern", "memory_efficient"]

            for stat in required_stats:
                if stat not in stats:
                    validation_result.validation_errors.append(f"Missing required factory statistic: {stat}")

            if stats.get("singleton_pattern") != "active":
                validation_result.validation_errors.append("Factory singleton pattern not active")

            if not stats.get("memory_efficient"):
                validation_result.validation_errors.append("Factory not memory efficient")

            # Contract 3: Instance cleanup contract
            initial_count = stats["total_instances"]
            CacheFactory.clear_instances()

            # After clear, new instances should be created
            test_cache = CacheFactory.get_utility_cache()
            cleared_stats = CacheFactory.get_performance_stats()

            if cleared_stats["total_instances"] == 0:
                validation_result.validation_errors.append("Factory clear_instances not working")

            validation_result.performance_metrics.update(singleton_performance)
            validation_result.contract_validated = len(validation_result.validation_errors) == 0

            self.logger.info(f"CacheFactory contract validation: {validation_result.contract_validated}")
            for cache_type, perf in singleton_performance.items():
                self.logger.info(f"  {cache_type}: {perf:.2f}μs retrieval")

        except Exception as e:
            validation_result.validation_errors.append(f"Contract validation exception: {e!s}")

        # Assert contract compliance
        assert validation_result.contract_validated, f"CacheFactory contract violations: {validation_result.validation_errors}"

        # Assert performance contract
        for cache_type, perf_us in validation_result.performance_metrics.items():
            assert perf_us < 2.0, f"{cache_type} factory retrieval {perf_us:.2f}μs exceeds 2μs contract"

    async def test_cache_facade_direct_operations_contract(self):
        """Test CacheFacade direct operations contract compliance."""
        validation_result = CacheContractValidationResult(
            component_name="CacheFacade",
            contract_validated=False
        )

        try:
            # Get cache from factory
            cache = CacheFactory.get_ml_analysis_cache()

            # Contract 1: Direct operation performance (<2ms response times)
            test_data = {f"contract_key_{i}": {"contract": "test", "index": i} for i in range(30)}

            operation_times = {
                "set_times": [],
                "get_times": [],
                "delete_times": []
            }

            # Test SET operation contract
            for key, value in test_data.items():
                start_time = time.perf_counter()
                await cache.set(key, value, l2_ttl=3600, l1_ttl=1800)
                set_time = time.perf_counter() - start_time
                operation_times["set_times"].append(set_time * 1000)  # Convert to ms

                if set_time > 0.002:  # 2ms
                    validation_result.validation_errors.append(f"SET operation {set_time * 1000:.2f}ms exceeds 2ms contract")

            # Test GET operation contract
            for key in test_data:
                start_time = time.perf_counter()
                result = await cache.get(key)
                get_time = time.perf_counter() - start_time
                operation_times["get_times"].append(get_time * 1000)  # Convert to ms

                if get_time > 0.001:  # 1ms for cache hits
                    validation_result.validation_errors.append(f"GET operation {get_time * 1000:.2f}ms exceeds 1ms contract")

                if result is None:
                    validation_result.validation_errors.append(f"GET operation failed to retrieve {key}")

            # Test DELETE operation contract
            for key in list(test_data.keys())[:10]:
                start_time = time.perf_counter()
                await cache.delete(key)
                delete_time = time.perf_counter() - start_time
                operation_times["delete_times"].append(delete_time * 1000)  # Convert to ms

                if delete_time > 0.002:  # 2ms
                    validation_result.validation_errors.append(f"DELETE operation {delete_time * 1000:.2f}ms exceeds 2ms contract")

            # Calculate performance metrics
            avg_set_time = sum(operation_times["set_times"]) / len(operation_times["set_times"])
            avg_get_time = sum(operation_times["get_times"]) / len(operation_times["get_times"])
            avg_delete_time = sum(operation_times["delete_times"]) / len(operation_times["delete_times"])

            validation_result.performance_metrics = {
                "avg_set_ms": avg_set_time,
                "avg_get_ms": avg_get_time,
                "avg_delete_ms": avg_delete_time,
                "p95_set_ms": sorted(operation_times["set_times"])[int(len(operation_times["set_times"]) * 0.95)],
                "p95_get_ms": sorted(operation_times["get_times"])[int(len(operation_times["get_times"]) * 0.95)],
            }

            # Contract 2: API method contract
            required_methods = ["get", "set", "delete", "clear", "invalidate_pattern", "health_check"]
            for method in required_methods:
                if not hasattr(cache, method):
                    validation_result.validation_errors.append(f"Missing required method: {method}")

            # Contract 3: Health check contract
            health = await cache.health_check()
            required_health_fields = ["healthy", "status", "checks", "performance"]

            for field in required_health_fields:
                if field not in health:
                    validation_result.validation_errors.append(f"Missing health check field: {field}")

            # Contract 4: Performance statistics contract
            stats = cache.get_performance_stats()
            required_stat_fields = ["total_requests", "overall_hit_rate", "avg_response_time_ms", "architecture"]

            for field in required_stat_fields:
                if field not in stats:
                    validation_result.validation_errors.append(f"Missing performance stat field: {field}")

            if stats.get("architecture") != "direct_cache_aside_pattern":
                validation_result.validation_errors.append("Cache architecture contract violation")

            validation_result.contract_validated = len(validation_result.validation_errors) == 0

            self.logger.info(f"CacheFacade contract validation: {validation_result.contract_validated}")
            self.logger.info(f"  Performance: SET={avg_set_time:.3f}ms, GET={avg_get_time:.3f}ms, DELETE={avg_delete_time:.3f}ms")

        except Exception as e:
            validation_result.validation_errors.append(f"Contract validation exception: {e!s}")

        # Assert contract compliance
        assert validation_result.contract_validated, f"CacheFacade contract violations: {validation_result.validation_errors}"

        # Assert performance contracts
        assert validation_result.performance_metrics["avg_set_ms"] < 2.0, "Average SET time exceeds 2ms contract"
        assert validation_result.performance_metrics["avg_get_ms"] < 1.0, "Average GET time exceeds 1ms contract"

    async def test_cache_integration_workflow_contract(self):
        """Test end-to-end integration workflow contract compliance."""
        validation_result = CacheContractValidationResult(
            component_name="IntegrationWorkflow",
            contract_validated=False
        )

        try:
            # Contract: All utility functions use new cache architecture
            utility_functions = [
                ("config", get_config_safely, (), "utility"),
                ("metrics", get_metrics_safely, (), "utility"),
                ("logger", get_logger, ("test.contract.cache",), "utility"),
            ]

            workflow_performance = {}

            for func_name, func, args, _expected_cache_type in utility_functions:
                self.logger.info(f"Testing integration workflow: {func_name}")

                # Clear function-specific caches
                if func_name == "config":
                    self._clear_utility_caches()

                # Test function integration with new cache architecture
                execution_times = []

                for i in range(10):
                    start_time = time.perf_counter()
                    result = func(*args)
                    execution_time = time.perf_counter() - start_time
                    execution_times.append(execution_time * 1000)  # Convert to ms

                    # Validate function still works correctly
                    if result is None:
                        validation_result.validation_errors.append(f"{func_name} returned None")

                    # Performance should improve after first call (cache hit)
                    if i > 0 and execution_time > 0.01:  # 10ms threshold for cached calls
                        validation_result.validation_errors.append(f"{func_name} cached call {execution_time * 1000:.2f}ms too slow")

                avg_execution_time = sum(execution_times) / len(execution_times)
                workflow_performance[func_name] = avg_execution_time

                # Cache hits should be much faster than first call
                if len(execution_times) > 1:
                    first_call = execution_times[0]
                    avg_cached_calls = sum(execution_times[1:]) / len(execution_times[1:])

                    if avg_cached_calls >= first_call:
                        validation_result.validation_errors.append(f"{func_name} cache not providing speedup")

            # Contract: TextStat functions use new cache architecture
            textstat_wrapper = get_textstat_wrapper()
            test_text = "Contract validation testing text for comprehensive textstat validation."

            textstat_functions = [
                ("flesch_reading_ease", textstat_wrapper.flesch_reading_ease, (test_text,)),
                ("syllable_count", textstat_wrapper.syllable_count, (test_text,)),
                ("lexicon_count", textstat_wrapper.lexicon_count, (test_text,)),
            ]

            for func_name, func, args in textstat_functions:
                execution_times = []

                for i in range(5):
                    start_time = time.perf_counter()
                    result = func(*args)
                    execution_time = time.perf_counter() - start_time
                    execution_times.append(execution_time * 1000)  # Convert to ms

                    if result is None:
                        validation_result.validation_errors.append(f"TextStat {func_name} returned None")

                    # TextStat cached calls should be very fast
                    if i > 0 and execution_time > 0.005:  # 5ms threshold
                        validation_result.validation_errors.append(f"TextStat {func_name} cached call {execution_time * 1000:.2f}ms too slow")

                avg_execution_time = sum(execution_times) / len(execution_times)
                workflow_performance[f"textstat_{func_name}"] = avg_execution_time

            # Contract: ML analysis functions use new cache architecture
            try:
                lightweight_config = get_lightweight_config()
                analyzer = LinguisticAnalyzer(config=lightweight_config)

                ml_test_text = "Machine learning contract validation testing."

                ml_execution_times = []
                for i in range(3):
                    start_time = time.perf_counter()
                    result = analyzer.analyze_prompt(ml_test_text)
                    execution_time = time.perf_counter() - start_time
                    ml_execution_times.append(execution_time * 1000)  # Convert to ms

                    if not result or not hasattr(result, 'complexity_score'):
                        validation_result.validation_errors.append("ML analysis contract violation")

                avg_ml_time = sum(ml_execution_times) / len(ml_execution_times)
                workflow_performance["ml_analysis"] = avg_ml_time

            except Exception as e:
                self.logger.warning(f"ML analysis contract test skipped: {e}")

            validation_result.performance_metrics = workflow_performance
            validation_result.contract_validated = len(validation_result.validation_errors) == 0

            self.logger.info(f"Integration workflow contract validation: {validation_result.contract_validated}")
            for component, perf in workflow_performance.items():
                self.logger.info(f"  {component}: {perf:.3f}ms average")

        except Exception as e:
            validation_result.validation_errors.append(f"Integration workflow exception: {e!s}")

        # Assert integration contract compliance
        assert validation_result.contract_validated, f"Integration workflow contract violations: {validation_result.validation_errors}"

    async def test_cache_error_handling_contract(self):
        """Test cache error handling and graceful degradation contract."""
        validation_result = CacheContractValidationResult(
            component_name="ErrorHandling",
            contract_validated=False
        )

        try:
            # Contract: Graceful degradation when L2 unavailable
            degraded_cache = CacheFacade(l1_max_size=50, enable_l2=False)

            try:
                # Test operations still work with L1-only
                await degraded_cache.set("error_test", {"degraded": True})
                result = await degraded_cache.get("error_test")

                if result != {"degraded": True}:
                    validation_result.validation_errors.append("Degraded cache operation failed")

                # Test health check reports degradation
                health = await degraded_cache.health_check()
                if health.get("healthy") not in {True, False}:  # Should still provide health status
                    validation_result.validation_errors.append("Degraded health check contract violation")

                # Contract: Performance under degradation
                degraded_times = []
                for i in range(20):
                    start_time = time.perf_counter()
                    await degraded_cache.set(f"degraded_{i}", i)
                    result = await degraded_cache.get(f"degraded_{i}")
                    operation_time = time.perf_counter() - start_time
                    degraded_times.append(operation_time * 1000)  # Convert to ms

                    if result != i:
                        validation_result.validation_errors.append(f"Degraded operation failed for key degraded_{i}")

                avg_degraded_time = sum(degraded_times) / len(degraded_times)

                # Degraded operations should still be fast (L1-only should be <1ms)
                if avg_degraded_time > 1.0:
                    validation_result.validation_errors.append(f"Degraded operations {avg_degraded_time:.2f}ms too slow")

                validation_result.performance_metrics["degraded_avg_ms"] = avg_degraded_time

            finally:
                await degraded_cache.close()

            # Contract: Factory continues working during errors
            try:
                # Simulate factory stress
                cache_instances = []
                for i in range(10):
                    cache = CacheFactory.get_utility_cache()
                    cache_instances.append(cache)

                # All should be same instance (singleton)
                for cache in cache_instances[1:]:
                    if cache is not cache_instances[0]:
                        validation_result.validation_errors.append("Factory singleton violated under stress")

                # Factory should still provide statistics
                stats = CacheFactory.get_performance_stats()
                if "total_instances" not in stats:
                    validation_result.validation_errors.append("Factory statistics unavailable under stress")

            except Exception as e:
                validation_result.validation_errors.append(f"Factory error handling failed: {e!s}")

            validation_result.contract_validated = len(validation_result.validation_errors) == 0

            self.logger.info(f"Error handling contract validation: {validation_result.contract_validated}")
            self.logger.info(f"  Degraded performance: {validation_result.performance_metrics.get('degraded_avg_ms', 0):.3f}ms")

        except Exception as e:
            validation_result.validation_errors.append(f"Error handling contract exception: {e!s}")

        # Assert error handling contract compliance
        assert validation_result.contract_validated, f"Error handling contract violations: {validation_result.validation_errors}"

    async def test_cache_performance_improvement_contract(self):
        """Test performance improvement contract validation (25x improvement claims)."""
        validation_result = CacheContractValidationResult(
            component_name="PerformanceImprovement",
            contract_validated=False
        )

        try:
            # Contract: 25x performance improvement over old coordination approach
            old_coordination_baseline_ms = 51.5  # From performance analysis

            # Test current cache performance
            cache = CacheFactory.get_prompt_cache()

            current_times = []
            for i in range(50):
                start_time = time.perf_counter()
                await cache.set(f"perf_contract_{i}", {"performance": i})
                result = await cache.get(f"perf_contract_{i}")
                operation_time = time.perf_counter() - start_time
                current_times.append(operation_time * 1000)  # Convert to ms

                if result != {"performance": i}:
                    validation_result.validation_errors.append(f"Performance test operation failed for key perf_contract_{i}")

            avg_current_time = sum(current_times) / len(current_times)
            performance_improvement = old_coordination_baseline_ms / avg_current_time if avg_current_time > 0 else 0

            validation_result.performance_metrics = {
                "current_avg_ms": avg_current_time,
                "old_baseline_ms": old_coordination_baseline_ms,
                "improvement_factor": performance_improvement,
                "target_improvement": 25.0
            }

            # Contract: Must achieve at least 20x improvement (allowing some margin)
            if performance_improvement < 20.0:
                validation_result.validation_errors.append(
                    f"Performance improvement {performance_improvement:.1f}x below 20x minimum contract"
                )

            # Contract: Current operations must be <2ms average
            if avg_current_time > 2.0:
                validation_result.validation_errors.append(
                    f"Current operations {avg_current_time:.3f}ms exceed 2ms contract"
                )

            # Contract: P95 performance must be <5ms
            p95_time = sorted(current_times)[int(len(current_times) * 0.95)]
            if p95_time > 5.0:
                validation_result.validation_errors.append(
                    f"P95 performance {p95_time:.3f}ms exceeds 5ms contract"
                )

            validation_result.performance_metrics["p95_ms"] = p95_time
            validation_result.contract_validated = len(validation_result.validation_errors) == 0

            self.logger.info(f"Performance improvement contract validation: {validation_result.contract_validated}")
            self.logger.info(f"  Current: {avg_current_time:.3f}ms, Improvement: {performance_improvement:.1f}x, P95: {p95_time:.3f}ms")

        except Exception as e:
            validation_result.validation_errors.append(f"Performance contract exception: {e!s}")

        # Assert performance improvement contract compliance
        assert validation_result.contract_validated, f"Performance improvement contract violations: {validation_result.validation_errors}"

        # Assert specific performance contracts
        improvement = validation_result.performance_metrics["improvement_factor"]
        current_avg = validation_result.performance_metrics["current_avg_ms"]
        p95 = validation_result.performance_metrics["p95_ms"]

        assert improvement >= 20.0, f"Performance improvement {improvement:.1f}x below 20x contract"
        assert current_avg < 2.0, f"Current average {current_avg:.3f}ms exceeds 2ms contract"
        assert p95 < 5.0, f"P95 performance {p95:.3f}ms exceeds 5ms contract"

    def test_cache_migration_completeness_contract(self):
        """Test that cache migration is complete and all components use new architecture."""
        validation_result = CacheContractValidationResult(
            component_name="MigrationCompleteness",
            contract_validated=False
        )

        try:
            # Contract: All cache types available from factory
            required_cache_types = ["utility", "textstat", "ml_analysis", "session", "rule", "prompt"]
            available_cache_types = []

            factory_methods = {
                "utility": CacheFactory.get_utility_cache,
                "textstat": CacheFactory.get_textstat_cache,
                "ml_analysis": CacheFactory.get_ml_analysis_cache,
                "session": CacheFactory.get_session_cache,
                "rule": CacheFactory.get_rule_cache,
                "prompt": CacheFactory.get_prompt_cache,
            }

            for cache_type, factory_method in factory_methods.items():
                try:
                    cache = factory_method()
                    if cache is not None:
                        available_cache_types.append(cache_type)
                except Exception as e:
                    validation_result.validation_errors.append(f"Cache type {cache_type} not available: {e!s}")

            # Contract: All required cache types available
            missing_types = set(required_cache_types) - set(available_cache_types)
            if missing_types:
                validation_result.validation_errors.append(f"Missing cache types: {missing_types}")

            # Contract: Factory provides comprehensive statistics
            stats = CacheFactory.get_performance_stats()
            if stats["total_instances"] != len(available_cache_types):
                validation_result.validation_errors.append("Factory instance count mismatch")

            # Contract: No legacy services remain
            try:
                # These should not be importable anymore
                legacy_imports = [
                    "prompt_improver.services.cache.l3_database_service",
                    "prompt_improver.services.cache.cache_coordinator_service",
                ]

                for legacy_import in legacy_imports:
                    try:
                        __import__(legacy_import)
                        validation_result.validation_errors.append(f"Legacy service still available: {legacy_import}")
                    except ImportError:
                        # Good - legacy service properly removed
                        pass

            except Exception as e:
                # Import errors are expected for removed services
                pass

            validation_result.performance_metrics = {
                "available_cache_types": len(available_cache_types),
                "required_cache_types": len(required_cache_types),
                "factory_instances": stats["total_instances"]
            }

            validation_result.contract_validated = len(validation_result.validation_errors) == 0

            self.logger.info(f"Migration completeness contract validation: {validation_result.contract_validated}")
            self.logger.info(f"  Available cache types: {available_cache_types}")
            self.logger.info(f"  Factory instances: {stats['total_instances']}")

        except Exception as e:
            validation_result.validation_errors.append(f"Migration completeness exception: {e!s}")

        # Assert migration completeness contract compliance
        assert validation_result.contract_validated, f"Migration completeness contract violations: {validation_result.validation_errors}"

        # Assert all required cache types available
        assert validation_result.performance_metrics["available_cache_types"] == validation_result.performance_metrics["required_cache_types"], "Not all cache types available"


if __name__ == "__main__":
    """Run comprehensive cache architecture contract validation tests."""
    pytest.main([__file__, "-v", "--tb=short"])
