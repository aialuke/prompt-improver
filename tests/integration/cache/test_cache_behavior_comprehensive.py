"""Comprehensive Cache Behavior Testing for Direct Cache Architecture

This module provides comprehensive tests for cache behavior patterns using the new
high-performance direct L1+L2 cache-aside pattern through CacheFactory. Tests validate
the elimination of coordination overhead while maintaining cache effectiveness.

Architecture Tested:
- CacheFactory singleton pattern for optimized instance management
- Direct L1+L2 cache-aside pattern (no coordination layer)
- Performance improvements: <2ms response times, 25x faster cache access
- Eliminated services: L3DatabaseService, CacheCoordinatorService

Cache Behavior Testing Coverage:
1. CacheFactory Pattern - Validate singleton instance management
2. Direct Cache Operations - Test L1+L2 cache-aside pattern
3. Cache Hit/Miss Patterns - Validate cache effectiveness with direct operations
4. Cache Invalidation - Test direct cache clearing and expiration
5. Performance Validation - Test <2ms response times and performance improvements
6. Error Handling - Test cache failure recovery with graceful degradation
7. Memory Management - Test memory efficiency and cleanup
"""

import asyncio
import logging
import time
from typing import Any

import pytest
from tests.fixtures.application.cache import reset_test_caches

from prompt_improver.services.cache.cache_facade import CacheFacade
from prompt_improver.services.cache.cache_factory import CacheFactory


@pytest.mark.integration
@pytest.mark.cache_behavior
@pytest.mark.real_behavior
class TestCacheFactoryBehaviorComprehensive:
    """Comprehensive cache behavior testing for CacheFactory pattern and direct cache operations."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for cache behavior tests."""
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
        """Cleanup after cache behavior tests."""
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

    def test_cache_factory_singleton_behavior(self):
        """Test CacheFactory singleton pattern behavior and performance."""
        # Test singleton pattern across different cache types
        cache_types = [
            ("utility", CacheFactory.get_utility_cache),
            ("textstat", CacheFactory.get_textstat_cache),
            ("ml_analysis", CacheFactory.get_ml_analysis_cache),
            ("session", CacheFactory.get_session_cache),
            ("rule", CacheFactory.get_rule_cache),
            ("prompt", CacheFactory.get_prompt_cache),
        ]

        for cache_type, factory_method in cache_types:
            self.logger.info(f"Testing CacheFactory singleton pattern: {cache_type}")

            # Test first creation
            start_time = time.perf_counter()
            cache1 = factory_method()
            first_creation_time = time.perf_counter() - start_time

            # Test subsequent retrievals (singleton pattern)
            retrieval_times = []
            for _i in range(20):
                start_time = time.perf_counter()
                cache2 = factory_method()
                retrieval_time = time.perf_counter() - start_time
                retrieval_times.append(retrieval_time)

                # Validate same instance returned
                assert cache2 is cache1, f"{cache_type} cache not returning same singleton instance"

            # Performance analysis
            avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
            performance_improvement = first_creation_time / avg_retrieval_time if avg_retrieval_time > 0 else 0

            # Validate singleton performance targets
            assert avg_retrieval_time < 0.000002, f"{cache_type} average retrieval {avg_retrieval_time * 1000000:.2f}μs exceeds 2μs target"
            assert performance_improvement > 10, f"{cache_type} performance improvement {performance_improvement:.1f}x too low"

            self.logger.info(f"{cache_type}: creation={first_creation_time * 1000000:.2f}μs, "
                            f"retrieval={avg_retrieval_time * 1000000:.2f}μs, "
                            f"improvement={performance_improvement:.1f}x")

        # Test factory statistics
        stats = CacheFactory.get_performance_stats()
        assert stats["total_instances"] == len(cache_types)
        assert stats["singleton_pattern"] == "active"
        assert stats["memory_efficient"] is True

    async def test_direct_cache_operations_behavior(self):
        """Test direct cache operations behavior with CacheFactory instances."""
        # Get cache from factory
        cache = CacheFactory.get_utility_cache()

        # Test direct cache operations performance
        test_data = {f"direct_key_{i}": {"operation": "direct", "index": i} for i in range(50)}

        # Test SET operations (direct L1+L2 access)
        set_times = []
        for key, value in test_data.items():
            start_time = time.perf_counter()
            await cache.set(key, value, l2_ttl=3600, l1_ttl=1800)
            set_time = time.perf_counter() - start_time
            set_times.append(set_time)

            # Validate direct operation performance
            assert set_time < 0.002, f"Direct set took {set_time * 1000:.2f}ms, exceeds 2ms target"

        # Test GET operations (direct cache-aside pattern)
        get_times = []
        for key in test_data:
            start_time = time.perf_counter()
            result = await cache.get(key)
            get_time = time.perf_counter() - start_time
            get_times.append(get_time)

            assert result is not None, f"Failed to get key {key}"
            assert get_time < 0.001, f"Direct get took {get_time * 1000:.2f}ms, exceeds 1ms target"

        # Performance analysis
        avg_set_time = sum(set_times) / len(set_times)
        avg_get_time = sum(get_times) / len(get_times)

        self.logger.info(f"Direct cache operations - SET: {avg_set_time * 1000:.3f}ms, GET: {avg_get_time * 1000:.3f}ms")

        # Validate performance improvements
        assert avg_set_time < 0.002, f"Average set time {avg_set_time * 1000:.3f}ms exceeds 2ms"
        assert avg_get_time < 0.001, f"Average get time {avg_get_time * 1000:.3f}ms exceeds 1ms"

    async def test_cache_hit_miss_patterns_with_factory(self):
        """Test cache hit/miss patterns using CacheFactory instances."""
        # Test different cache types from factory
        cache_configs = [
            ("utility", CacheFactory.get_utility_cache()),
            ("textstat", CacheFactory.get_textstat_cache()),
            ("ml_analysis", CacheFactory.get_ml_analysis_cache()),
        ]

        for cache_name, cache in cache_configs:
            self.logger.info(f"Testing cache hit/miss patterns: {cache_name}")

            # Clear cache for clean test
            await cache.clear()

            test_key = f"{cache_name}_hit_miss_test"
            test_value = {"cache_type": cache_name, "pattern": "hit_miss_test"}

            # Test cache miss (first access)
            start_time = time.perf_counter()
            miss_result = await cache.get(test_key)
            miss_time = time.perf_counter() - start_time

            assert miss_result is None, f"{cache_name} should return None on cache miss"

            # Set value in cache
            await cache.set(test_key, test_value)

            # Test cache hits (subsequent accesses)
            hit_times = []
            for _i in range(10):
                start_time = time.perf_counter()
                hit_result = await cache.get(test_key)
                hit_time = time.perf_counter() - start_time
                hit_times.append(hit_time)

                assert hit_result == test_value, f"{cache_name} cache hit returned wrong value"
                assert hit_time < 0.001, f"{cache_name} cache hit took {hit_time * 1000:.3f}ms, too slow"

            avg_hit_time = sum(hit_times) / len(hit_times)

            self.logger.info(f"{cache_name}: miss={miss_time * 1000:.3f}ms, avg_hit={avg_hit_time * 1000:.3f}ms")

            # Validate hit performance
            assert avg_hit_time < 0.001, f"{cache_name} average hit time {avg_hit_time * 1000:.3f}ms exceeds 1ms"

    async def test_cache_invalidation_patterns(self):
        """Test cache invalidation patterns with direct operations."""
        cache = CacheFactory.get_prompt_cache()

        # Set up test data with patterns
        pattern_data = [
            ("user:123:prompt", {"user": 123, "type": "prompt"}),
            ("user:123:rules", {"user": 123, "type": "rules"}),
            ("user:456:prompt", {"user": 456, "type": "prompt"}),
            ("system:config", {"type": "system"}),
        ]

        # Set all test data
        for key, value in pattern_data:
            await cache.set(key, value)

        # Verify all data is cached
        for key, expected_value in pattern_data:
            result = await cache.get(key)
            assert result == expected_value, f"Data not cached for {key}"

        # Test pattern invalidation (direct operation)
        start_time = time.perf_counter()
        invalidated_count = await cache.invalidate_pattern("user:123:*")
        invalidation_time = time.perf_counter() - start_time

        # Validate invalidation performance
        assert invalidation_time < 0.002, f"Pattern invalidation took {invalidation_time * 1000:.2f}ms, exceeds 2ms"
        assert invalidated_count >= 2, f"Expected ≥2 invalidations, got {invalidated_count}"

        # Verify selective invalidation
        assert await cache.get("user:123:prompt") is None
        assert await cache.get("user:123:rules") is None
        assert await cache.get("user:456:prompt") is not None
        assert await cache.get("system:config") is not None

        self.logger.info(f"Pattern invalidation: {invalidation_time * 1000:.3f}ms for {invalidated_count} entries")

    async def test_cache_warming_and_population_behavior(self):
        """Test cache warming and population behavior with direct operations."""
        cache = CacheFactory.get_ml_analysis_cache()

        # Test cache warming through fallback function
        test_key = "ml_warming_test"
        test_value = {"analysis": "ml_result", "warming": True}

        def fallback_computation():
            # Simulate ML computation
            time.sleep(0.001)  # 1ms computation
            return test_value

        # Test cache-aside pattern with fallback
        start_time = time.perf_counter()
        result = await cache.get(test_key, fallback_func=fallback_computation)
        fallback_time = time.perf_counter() - start_time

        assert result == test_value
        assert fallback_time > 0.001, "Fallback computation time too fast"
        assert fallback_time < 0.005, f"Fallback computation took {fallback_time * 1000:.2f}ms, too slow"

        # Subsequent access should be cache hit (much faster)
        start_time = time.perf_counter()
        cached_result = await cache.get(test_key)
        cache_hit_time = time.perf_counter() - start_time

        assert cached_result == test_value
        assert cache_hit_time < 0.001, f"Cache hit took {cache_hit_time * 1000:.3f}ms, too slow"

        # Validate cache warming effectiveness
        speedup = fallback_time / cache_hit_time if cache_hit_time > 0 else 0
        assert speedup > 5, f"Cache warming speedup {speedup:.1f}x too low"

        self.logger.info(f"Cache warming - Fallback: {fallback_time * 1000:.2f}ms, Hit: {cache_hit_time * 1000:.3f}ms, Speedup: {speedup:.1f}x")

    async def test_cache_persistence_across_instances(self):
        """Test cache persistence across CacheFactory instances."""
        # Get same cache type multiple times
        cache1 = CacheFactory.get_session_cache()
        cache2 = CacheFactory.get_session_cache()

        # Verify same instance (singleton pattern)
        assert cache1 is cache2, "CacheFactory not returning same instance"

        # Test data persistence across "different" instances
        test_key = "persistence_test"
        test_value = {"persistence": True, "session": "data"}

        # Set data using first instance
        await cache1.set(test_key, test_value)

        # Get data using second instance (same actual instance)
        result = await cache2.get(test_key)
        assert result == test_value, "Data not persistent across cache instances"

        # Test session-specific operations
        session_id = "test_session_123"
        session_data = {"user_id": 123, "permissions": ["read", "write"]}

        # Set session data
        success = await cache1.set_session(session_id, session_data)
        assert success is True, "Session set failed"

        # Get session data from "different" instance
        retrieved_data = await cache2.get_session(session_id)
        assert retrieved_data == session_data, "Session data not persistent"

        self.logger.info("Cache persistence validated across factory instances")

    async def test_cache_error_handling_and_recovery(self):
        """Test cache error handling and graceful recovery."""
        # Test with L1-only cache (L2 disabled for error scenarios)
        cache = CacheFacade(l1_max_size=50, enable_l2=False)

        try:
            # Test operations under various error conditions
            error_scenarios = [
                ("normal_operation", 20),
                ("large_key_operation", 1),
                ("rapid_operations", 50),
                ("memory_pressure", 100),
            ]

            for scenario_name, operation_count in error_scenarios:
                self.logger.info(f"Testing error handling: {scenario_name}")

                start_time = time.perf_counter()
                successful_ops = 0
                failed_ops = 0

                for i in range(operation_count):
                    try:
                        if scenario_name == "large_key_operation":
                            key = "x" * 1000 + f"_{i}"  # Very long key
                        else:
                            key = f"{scenario_name}_{i}"

                        value = {"scenario": scenario_name, "index": i}

                        # Test set and get operations
                        await cache.set(key, value)
                        result = await cache.get(key)

                        if result == value:
                            successful_ops += 1
                        else:
                            failed_ops += 1

                    except Exception as e:
                        failed_ops += 1
                        self.logger.debug(f"Expected error in {scenario_name}: {e}")

                scenario_time = time.perf_counter() - start_time
                success_rate = successful_ops / (successful_ops + failed_ops) if (successful_ops + failed_ops) > 0 else 0

                # Validate error resilience
                assert success_rate > 0.7, f"{scenario_name} success rate {success_rate:.2%} too low"
                assert scenario_time < 1.0, f"{scenario_name} took {scenario_time:.2f}s, too slow"

                self.logger.info(f"{scenario_name}: {success_rate:.2%} success rate, {scenario_time:.2f}s")

        finally:
            await cache.close()

    async def test_cache_memory_management_efficiency(self):
        """Test cache memory management and efficiency."""
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Test memory efficiency with multiple cache instances
        cache_instances = [
            CacheFactory.get_utility_cache(),
            CacheFactory.get_textstat_cache(),
            CacheFactory.get_ml_analysis_cache(),
        ]

        # Perform operations on all cache instances
        for i, cache in enumerate(cache_instances):
            for j in range(100):
                key = f"memory_test_{i}_{j}"
                value = {"cache": i, "item": j, "data": "x" * 100}
                await cache.set(key, value)
                await cache.get(key)

        # Check memory usage after operations
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        memory_increase_mb = memory_increase / (1024 * 1024)

        # Validate memory efficiency
        assert memory_increase_mb < 10, f"Memory increased by {memory_increase_mb:.1f}MB, too high"

        # Test factory memory efficiency
        factory_stats = CacheFactory.get_performance_stats()
        assert factory_stats["memory_efficient"] is True
        assert factory_stats["total_instances"] == 3  # Should reuse instances

        self.logger.info(f"Memory efficiency: {memory_increase_mb:.1f}MB increase for 300 operations across 3 cache types")

    async def test_cache_concurrent_operations_behavior(self):
        """Test cache behavior under concurrent operations."""
        cache = CacheFactory.get_rule_cache()

        async def concurrent_worker(worker_id: int, operations: int) -> dict[str, Any]:
            """Concurrent worker for cache operations."""
            successful = 0
            failed = 0

            for i in range(operations):
                try:
                    key = f"concurrent_{worker_id}_{i}"
                    value = {"worker": worker_id, "operation": i}

                    await cache.set(key, value)
                    result = await cache.get(key)

                    if result == value:
                        successful += 1
                    else:
                        failed += 1

                except Exception:
                    failed += 1

            return {"worker_id": worker_id, "successful": successful, "failed": failed}

        # Run concurrent workers
        worker_count = 10
        operations_per_worker = 20

        start_time = time.perf_counter()
        tasks = [concurrent_worker(i, operations_per_worker) for i in range(worker_count)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        # Analyze concurrent operation results
        total_successful = sum(r["successful"] for r in results)
        total_failed = sum(r["failed"] for r in results)
        total_operations = total_successful + total_failed
        success_rate = total_successful / total_operations if total_operations > 0 else 0
        throughput = total_operations / total_time

        # Validate concurrent behavior
        assert success_rate > 0.95, f"Concurrent success rate {success_rate:.2%} too low"
        assert throughput > 1000, f"Concurrent throughput {throughput:.0f} ops/sec too low"
        assert total_time < 2.0, f"Concurrent operations took {total_time:.2f}s, too slow"

        self.logger.info(f"Concurrent operations: {success_rate:.2%} success rate, {throughput:.0f} ops/sec, {total_time:.2f}s")

    async def test_cache_performance_validation_comprehensive(self):
        """Comprehensive performance validation for direct cache architecture."""
        # Test performance across all cache types
        cache_performance_data = {}

        cache_types = [
            ("utility", CacheFactory.get_utility_cache()),
            ("textstat", CacheFactory.get_textstat_cache()),
            ("ml_analysis", CacheFactory.get_ml_analysis_cache()),
            ("session", CacheFactory.get_session_cache()),
            ("rule", CacheFactory.get_rule_cache()),
            ("prompt", CacheFactory.get_prompt_cache()),
        ]

        for cache_name, cache in cache_types:
            self.logger.info(f"Performance validation: {cache_name}")

            # Generate test data
            test_data = {f"{cache_name}_perf_{i}": {"performance": i, "cache": cache_name} for i in range(50)}

            # Measure SET performance
            set_times = []
            for key, value in test_data.items():
                start_time = time.perf_counter()
                await cache.set(key, value)
                set_time = time.perf_counter() - start_time
                set_times.append(set_time)

            # Measure GET performance
            get_times = []
            for key in test_data:
                start_time = time.perf_counter()
                result = await cache.get(key)
                get_time = time.perf_counter() - start_time
                get_times.append(get_time)
                assert result is not None

            # Calculate performance metrics
            avg_set_time = sum(set_times) / len(set_times)
            avg_get_time = sum(get_times) / len(get_times)
            p95_set_time = sorted(set_times)[int(len(set_times) * 0.95)]
            p95_get_time = sorted(get_times)[int(len(get_times) * 0.95)]

            cache_performance_data[cache_name] = {
                "avg_set_ms": avg_set_time * 1000,
                "avg_get_ms": avg_get_time * 1000,
                "p95_set_ms": p95_set_time * 1000,
                "p95_get_ms": p95_get_time * 1000,
            }

            # Validate performance targets
            assert avg_set_time < 0.002, f"{cache_name} avg set {avg_set_time * 1000:.3f}ms exceeds 2ms"
            assert avg_get_time < 0.001, f"{cache_name} avg get {avg_get_time * 1000:.3f}ms exceeds 1ms"
            assert p95_set_time < 0.005, f"{cache_name} P95 set {p95_set_time * 1000:.3f}ms too high"
            assert p95_get_time < 0.002, f"{cache_name} P95 get {p95_get_time * 1000:.3f}ms too high"

            self.logger.info(f"{cache_name}: SET={avg_set_time * 1000:.3f}ms, GET={avg_get_time * 1000:.3f}ms")

        # Overall performance summary
        all_set_times = [data["avg_set_ms"] for data in cache_performance_data.values()]
        all_get_times = [data["avg_get_ms"] for data in cache_performance_data.values()]

        overall_avg_set = sum(all_set_times) / len(all_set_times)
        overall_avg_get = sum(all_get_times) / len(all_get_times)

        self.logger.info(f"Overall performance - SET: {overall_avg_set:.3f}ms, GET: {overall_avg_get:.3f}ms")

        # Validate overall performance targets
        assert overall_avg_set < 2.0, f"Overall average set time {overall_avg_set:.3f}ms exceeds 2ms"
        assert overall_avg_get < 1.0, f"Overall average get time {overall_avg_get:.3f}ms exceeds 1ms"


if __name__ == "__main__":
    """Run comprehensive cache behavior tests for CacheFactory pattern."""
    pytest.main([__file__, "-v", "--tb=short"])
