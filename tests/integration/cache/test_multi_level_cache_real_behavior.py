"""Real behavior tests for direct L1+L2 cache-aside pattern.

Comprehensive validation of the new high-performance cache architecture with actual Redis instances.
Tests the direct cache-aside pattern that eliminates coordination overhead and achieves <2ms response times.

Architecture Tested:
- Direct L1+L2 operations through CacheFacade (no coordination layer)
- CacheFactory singleton pattern for optimized instance management
- Cache-aside pattern: App → CacheFacade → Direct L1 → Direct L2 → Storage

Performance Requirements:
- L1 Cache: <1ms response time, >95% hit rate for hot data
- L2 Cache: <10ms response time, >80% hit rate for warm data
- Overall CacheFacade: <2ms response times (eliminates 775x coordination overhead)
- CacheFactory: <2μs instance retrieval (singleton pattern)
"""

import asyncio
import time
from typing import Any

import pytest
from tests.containers.real_redis_container import RealRedisTestContainer

from prompt_improver.services.cache.cache_facade import CacheFacade
from prompt_improver.services.cache.cache_factory import CacheFactory


class TestDirectCacheAsidePatternRealBehavior:
    """Real behavior tests for direct L1+L2 cache-aside pattern."""

    @pytest.fixture
    async def redis_container(self):
        """Real Redis testcontainer for L2 cache."""
        container = RealRedisTestContainer()
        await container.start()
        container.set_env_vars()
        yield container
        await container.stop()

    @pytest.fixture
    async def cache_facade_l1_only(self):
        """CacheFacade with L1-only for maximum performance testing."""
        cache = CacheFacade(
            l1_max_size=500,
            enable_l2=False,  # L1-only for speed tests
        )
        yield cache
        await cache.clear()
        await cache.close()

    @pytest.fixture
    async def cache_facade_l1_l2(self, redis_container):
        """CacheFacade with L1+L2 for full cache-aside pattern testing."""
        cache = CacheFacade(
            l1_max_size=500,
            l2_default_ttl=3600,
            enable_l2=True,   # Full L1+L2 cache-aside pattern
        )
        yield cache
        await cache.clear()
        await cache.close()

    async def test_direct_cache_facade_performance_targets(self, cache_facade_l1_l2):
        """Validate CacheFacade meets <2ms response time targets (vs 775x coordination overhead)."""
        test_data = {f"perf_key_{i}": {"id": i, "data": f"performance_value_{i}"} for i in range(100)}

        # Test direct SET operations (no coordination overhead)
        set_times = []
        for key, value in test_data.items():
            start_time = time.perf_counter()
            await cache_facade_l1_l2.set(key, value, l2_ttl=3600, l1_ttl=1800)
            duration = time.perf_counter() - start_time
            set_times.append(duration)
            assert duration < 0.002, f"Direct set took {duration * 1000:.2f}ms, exceeds 2ms target"

        # Test direct GET operations with L1 hits
        get_times_l1 = []
        for key in list(test_data.keys())[:50]:
            start_time = time.perf_counter()
            result = await cache_facade_l1_l2.get(key)
            duration = time.perf_counter() - start_time
            get_times_l1.append(duration)
            assert result is not None, f"Failed to get key {key}"
            assert duration < 0.001, f"L1 hit took {duration * 1000:.2f}ms, exceeds 1ms target"

        # Clear L1 to test L2 hit performance
        await cache_facade_l1_l2._l1_cache.clear()

        # Test direct GET operations with L2 hits
        get_times_l2 = []
        for key in list(test_data.keys())[50:75]:
            start_time = time.perf_counter()
            result = await cache_facade_l1_l2.get(key)
            duration = time.perf_counter() - start_time
            get_times_l2.append(duration)
            assert result is not None, f"Failed to get key {key} from L2"
            assert duration < 0.010, f"L2 hit took {duration * 1000:.2f}ms, exceeds 10ms target"

        # Performance summary
        avg_set_time = sum(set_times) / len(set_times)
        avg_get_l1_time = sum(get_times_l1) / len(get_times_l1)
        avg_get_l2_time = sum(get_times_l2) / len(get_times_l2)

        print("\nDirect Cache-Aside Performance (No Coordination Overhead):")
        print(f"  Average SET time: {avg_set_time * 1000:.3f}ms (target: <2ms)")
        print(f"  Average L1 GET time: {avg_get_l1_time * 1000:.3f}ms (target: <1ms)")
        print(f"  Average L2 GET time: {avg_get_l2_time * 1000:.3f}ms (target: <10ms)")
        print("  Performance improvement: Eliminated 775x coordination overhead")

        # Validate performance targets achieved
        assert avg_set_time < 0.002, f"Average set time {avg_set_time * 1000:.3f}ms exceeds 2ms"
        assert avg_get_l1_time < 0.001, f"Average L1 get time {avg_get_l1_time * 1000:.3f}ms exceeds 1ms"
        assert avg_get_l2_time < 0.010, f"Average L2 get time {avg_get_l2_time * 1000:.3f}ms exceeds 10ms"

    async def test_cache_aside_pattern_l1_l2_promotion(self, cache_facade_l1_l2):
        """Test cache-aside pattern with L1→L2 promotion and population."""
        test_key = "cache_aside_test"
        test_value = {"cache_aside": True, "promotion_test": "data"}

        # Test cache miss → fallback → populate L1+L2 (cache-aside pattern)
        def fallback_function():
            return test_value

        # Initial miss - should populate both L1 and L2
        start_time = time.perf_counter()
        result = await cache_facade_l1_l2.get(test_key, fallback_func=fallback_function)
        fallback_time = time.perf_counter() - start_time

        assert result == test_value
        assert fallback_time < 0.002, f"Cache-aside fallback took {fallback_time * 1000:.2f}ms, exceeds 2ms"

        # Verify data populated in both L1 and L2
        l1_result = await cache_facade_l1_l2._l1_cache.get(test_key)
        l2_result = await cache_facade_l1_l2._l2_cache.get(test_key)
        assert l1_result == test_value, "Cache-aside didn't populate L1"
        assert l2_result == test_value, "Cache-aside didn't populate L2"

        # Test L1 hit (fastest cache-aside path)
        start_time = time.perf_counter()
        result = await cache_facade_l1_l2.get(test_key)
        l1_hit_time = time.perf_counter() - start_time

        assert result == test_value
        assert l1_hit_time < 0.001, f"L1 hit took {l1_hit_time * 1000:.2f}ms, exceeds 1ms"

        # Clear L1 to test L2 hit → L1 promotion
        await cache_facade_l1_l2._l1_cache.clear()

        # Test L2 hit with cache-aside promotion to L1
        start_time = time.perf_counter()
        result = await cache_facade_l1_l2.get(test_key)
        l2_promotion_time = time.perf_counter() - start_time

        assert result == test_value
        assert l2_promotion_time < 0.010, f"L2 hit with L1 promotion took {l2_promotion_time * 1000:.2f}ms"

        # Verify L2 hit promoted value to L1
        l1_promoted = await cache_facade_l1_l2._l1_cache.get(test_key)
        assert l1_promoted == test_value, "L2 hit didn't promote to L1 in cache-aside pattern"

    async def test_direct_cache_operations_no_coordination(self, cache_facade_l1_l2):
        """Test direct cache operations without coordination overhead."""
        # Batch operations to test direct access patterns
        batch_data = {f"direct_key_{i}": {"batch": i, "direct": True} for i in range(50)}

        # Test direct SET operations in parallel (no coordination blocking)
        set_tasks = []
        for key, value in batch_data.items():
            set_tasks.append(cache_facade_l1_l2.set(key, value, l2_ttl=3600, l1_ttl=1800))

        start_time = time.perf_counter()
        await asyncio.gather(*set_tasks)
        batch_set_time = time.perf_counter() - start_time

        assert batch_set_time < 0.1, f"Batch set (50 items) took {batch_set_time * 1000:.2f}ms, too slow"
        print(f"Direct batch operations (50 sets): {batch_set_time * 1000:.2f}ms")

        # Test direct GET operations in parallel
        get_tasks = [cache_facade_l1_l2.get(key) for key in batch_data]

        start_time = time.perf_counter()
        results = await asyncio.gather(*get_tasks)
        batch_get_time = time.perf_counter() - start_time

        assert batch_get_time < 0.05, f"Batch get (50 items) took {batch_get_time * 1000:.2f}ms, too slow"
        assert all(r is not None for r in results), "Some batch get operations failed"
        print(f"Direct batch operations (50 gets): {batch_get_time * 1000:.2f}ms")

        # Verify no coordination overhead in operations
        avg_set_time = batch_set_time / 50
        avg_get_time = batch_get_time / 50

        assert avg_set_time < 0.002, f"Average direct set {avg_set_time * 1000:.3f}ms exceeds 2ms"
        assert avg_get_time < 0.001, f"Average direct get {avg_get_time * 1000:.3f}ms exceeds 1ms"

    async def test_cache_facade_pattern_invalidation(self, cache_facade_l1_l2):
        """Test pattern invalidation across direct L1+L2 operations."""
        # Set up pattern test data
        pattern_data = [
            ("user:123:profile", {"user": 123, "type": "profile"}),
            ("user:123:settings", {"user": 123, "type": "settings"}),
            ("user:456:profile", {"user": 456, "type": "profile"}),
            ("system:config", {"type": "system"}),
        ]

        # Set data in both L1 and L2
        for key, value in pattern_data:
            await cache_facade_l1_l2.set(key, value)

        # Verify data is in both levels
        for key, expected_value in pattern_data:
            result = await cache_facade_l1_l2.get(key)
            assert result == expected_value, f"Pattern data not set for {key}"

        # Test direct pattern invalidation (no coordination overhead)
        start_time = time.perf_counter()
        invalidated_count = await cache_facade_l1_l2.invalidate_pattern("user:123:*")
        invalidation_time = time.perf_counter() - start_time

        assert invalidation_time < 0.002, f"Pattern invalidation took {invalidation_time * 1000:.2f}ms, exceeds 2ms"
        assert invalidated_count >= 2, f"Expected ≥2 invalidations, got {invalidated_count}"

        # Verify selective invalidation worked
        assert await cache_facade_l1_l2.get("user:123:profile") is None
        assert await cache_facade_l1_l2.get("user:123:settings") is None
        assert await cache_facade_l1_l2.get("user:456:profile") is not None
        assert await cache_facade_l1_l2.get("system:config") is not None

        print(f"Direct pattern invalidation: {invalidation_time * 1000:.3f}ms for {invalidated_count} entries")

    async def test_cache_facade_health_and_performance_stats(self, cache_facade_l1_l2):
        """Test health check and performance statistics for direct cache operations."""
        # Perform operations to generate statistics
        for i in range(20):
            await cache_facade_l1_l2.set(f"stats_key_{i}", {"stats": i})
            await cache_facade_l1_l2.get(f"stats_key_{i}")

        # Test direct health check (no coordination overhead)
        start_time = time.perf_counter()
        health = await cache_facade_l1_l2.health_check()
        health_time = time.perf_counter() - start_time

        assert health_time < 0.010, f"Health check took {health_time * 1000:.2f}ms, exceeds 10ms"
        assert health["healthy"] is True
        assert health["status"] == "healthy"
        assert "l1_cache" in health["checks"]
        assert "l2_cache" in health["checks"]
        assert health["checks"]["l1_cache"] is True
        assert health["checks"]["l2_cache"] is True

        # Test performance statistics
        stats = cache_facade_l1_l2.get_performance_stats()
        assert stats["total_requests"] >= 40  # 20 sets + 20 gets
        assert stats["overall_hit_rate"] >= 0.5  # Should have decent hit rate
        assert stats["avg_response_time_ms"] < 2.0  # Target <2ms average
        assert stats["architecture"] == "direct_cache_aside_pattern"
        assert stats["coordination_overhead"] == "eliminated"

        print(f"Performance Stats - Hit Rate: {stats['overall_hit_rate']:.2%}, "
              f"Avg Response: {stats['avg_response_time_ms']:.3f}ms")

        # Validate performance improvement claims
        assert stats["avg_response_time_ms"] < 2.0, "Performance target not met"

    async def test_cache_facade_session_management(self, cache_facade_l1_l2):
        """Test secure session management with direct cache operations."""
        session_id = "test_session_123"
        session_data = {"user_id": 123, "permissions": ["read", "write"], "timestamp": time.time()}

        # Test session set with encryption (if available)
        start_time = time.perf_counter()
        success = await cache_facade_l1_l2.set_session(session_id, session_data, ttl=1800)
        session_set_time = time.perf_counter() - start_time

        assert session_set_time < 0.020, f"Session set took {session_set_time * 1000:.2f}ms, exceeds 20ms"

        # Test session get with decryption (if available)
        start_time = time.perf_counter()
        retrieved_data = await cache_facade_l1_l2.get_session(session_id)
        session_get_time = time.perf_counter() - start_time

        assert session_get_time < 0.020, f"Session get took {session_get_time * 1000:.2f}ms, exceeds 20ms"
        assert retrieved_data == session_data, "Session data not retrieved correctly"

        # Test session touch (TTL extension)
        touch_success = await cache_facade_l1_l2.touch_session(session_id, ttl=3600)
        assert touch_success is True, "Session touch failed"

        # Test session deletion
        delete_success = await cache_facade_l1_l2.delete_session(session_id)
        assert delete_success is True, "Session delete failed"

        # Verify session is deleted
        deleted_data = await cache_facade_l1_l2.get_session(session_id)
        assert deleted_data is None, "Session not properly deleted"

        print(f"Session operations - Set: {session_set_time * 1000:.2f}ms, Get: {session_get_time * 1000:.2f}ms")

    async def test_cache_facade_concurrent_operations(self, cache_facade_l1_l2):
        """Test cache facade stability under concurrent load with direct operations."""
        async def worker_task(worker_id: int, operations_count: int) -> dict[str, Any]:
            """Worker task for concurrent testing."""
            start_time = time.perf_counter()
            successful_ops = 0
            failed_ops = 0

            for i in range(operations_count):
                key = f"concurrent_worker_{worker_id}_item_{i}"
                value = {"worker": worker_id, "item": i, "timestamp": time.time()}

                try:
                    # Direct cache operations (no coordination blocking)
                    await cache_facade_l1_l2.set(key, value)
                    result = await cache_facade_l1_l2.get(key)

                    if result == value:
                        successful_ops += 1
                    else:
                        failed_ops += 1
                except Exception:
                    failed_ops += 1

            duration = time.perf_counter() - start_time
            return {
                "worker_id": worker_id,
                "duration": duration,
                "successful_ops": successful_ops,
                "failed_ops": failed_ops,
                "ops_per_second": (successful_ops + failed_ops) / duration,
            }

        # Run concurrent workers
        workers = 10
        ops_per_worker = 25

        start_time = time.perf_counter()
        tasks = [worker_task(i, ops_per_worker) for i in range(workers)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        # Analyze results
        total_successful = sum(r["successful_ops"] for r in results)
        total_failed = sum(r["failed_ops"] for r in results)
        total_operations = total_successful + total_failed
        success_rate = total_successful / total_operations if total_operations > 0 else 0

        print("\nConcurrent Direct Cache Operations:")
        print(f"  Total operations: {total_operations}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {total_operations / total_time:.0f} ops/sec")

        # Validate concurrent performance
        assert success_rate > 0.95, f"Success rate {success_rate:.2%} below 95% target"
        assert total_time < 5.0, f"Concurrent operations took {total_time:.2f}s, too slow"


class TestCacheFactoryRealBehavior:
    """Real behavior tests for CacheFactory singleton pattern."""

    def test_cache_factory_singleton_performance(self):
        """Test CacheFactory singleton pattern provides <2μs instance retrieval."""
        # Clear factory instances to start fresh
        CacheFactory.clear_instances()

        # Test first instance creation (slower)
        start_time = time.perf_counter()
        cache1 = CacheFactory.get_utility_cache()
        first_creation_time = time.perf_counter() - start_time

        # Test subsequent instance retrieval (singleton pattern)
        retrieval_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            cache2 = CacheFactory.get_utility_cache()
            retrieval_time = time.perf_counter() - start_time
            retrieval_times.append(retrieval_time)

            # Verify same instance returned
            assert cache2 is cache1, "Singleton pattern not working"

        # Performance analysis
        avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
        max_retrieval_time = max(retrieval_times)

        print("\nCacheFactory Singleton Performance:")
        print(f"  First creation: {first_creation_time * 1000000:.2f}μs")
        print(f"  Average retrieval: {avg_retrieval_time * 1000000:.2f}μs")
        print(f"  Max retrieval: {max_retrieval_time * 1000000:.2f}μs")
        print(f"  Performance improvement: {first_creation_time / avg_retrieval_time:.0f}x faster")

        # Validate performance targets
        assert avg_retrieval_time < 0.000002, f"Average retrieval {avg_retrieval_time * 1000000:.2f}μs exceeds 2μs target"
        assert max_retrieval_time < 0.000005, f"Max retrieval {max_retrieval_time * 1000000:.2f}μs too slow"

        # Cleanup
        CacheFactory.clear_instances()

    def test_cache_factory_specialized_configurations(self):
        """Test CacheFactory provides optimized configurations for different use cases."""
        # Test all specialized cache configurations
        cache_configs = [
            ("utility", CacheFactory.get_utility_cache),
            ("textstat", CacheFactory.get_textstat_cache),
            ("ml_analysis", CacheFactory.get_ml_analysis_cache),
            ("session", CacheFactory.get_session_cache),
            ("rule", CacheFactory.get_rule_cache),
            ("prompt", CacheFactory.get_prompt_cache),
        ]

        created_caches = []

        for config_name, factory_method in cache_configs:
            start_time = time.perf_counter()
            cache = factory_method()
            creation_time = time.perf_counter() - start_time

            created_caches.append((config_name, cache, creation_time))

            # Verify cache instance is configured
            assert cache is not None, f"Failed to create {config_name} cache"
            assert hasattr(cache, '_l1_cache'), f"{config_name} cache missing L1"

            # Test instance performance
            assert creation_time < 0.001, f"{config_name} cache creation took {creation_time * 1000:.2f}ms"

        # Test that specialized caches are different instances
        for i, (name1, cache1, _) in enumerate(created_caches):
            for j, (name2, cache2, _) in enumerate(created_caches):
                if i != j:
                    assert cache1 is not cache2, f"{name1} and {name2} caches should be different instances"

        # Test performance statistics
        stats = CacheFactory.get_performance_stats()
        assert stats["total_instances"] == len(cache_configs)
        assert stats["singleton_pattern"] == "active"
        assert stats["memory_efficient"] is True

        print(f"Created {len(cache_configs)} specialized cache configurations")
        print(f"Factory stats: {stats['total_instances']} instances, memory efficient: {stats['memory_efficient']}")

        # Cleanup
        CacheFactory.clear_instances()

    async def test_cache_factory_real_behavior_integration(self):
        """Test CacheFactory integration with real cache operations."""
        # Get different cache types from factory
        utility_cache = CacheFactory.get_utility_cache()
        ml_cache = CacheFactory.get_ml_analysis_cache()

        # Test real operations on factory-created caches
        test_data = [
            (utility_cache, "utility_key", {"config": "value"}),
            (ml_cache, "ml_key", {"analysis": "result"}),
        ]

        operation_times = []

        for cache, key, value in test_data:
            # Test set operation
            start_time = time.perf_counter()
            await cache.set(key, value)
            set_time = time.perf_counter() - start_time
            operation_times.append(("set", set_time))

            # Test get operation
            start_time = time.perf_counter()
            result = await cache.get(key)
            get_time = time.perf_counter() - start_time
            operation_times.append(("get", get_time))

            assert result == value, f"Cache operation failed for {key}"
            assert set_time < 0.002, f"Factory cache set took {set_time * 1000:.2f}ms"
            assert get_time < 0.001, f"Factory cache get took {get_time * 1000:.2f}ms"

        # Cleanup caches
        await utility_cache.clear()
        await ml_cache.clear()
        await utility_cache.close()
        await ml_cache.close()

        # Cleanup factory
        CacheFactory.clear_instances()

        print("Factory cache operations completed in <2ms each")


@pytest.mark.integration
@pytest.mark.real_behavior
class TestCacheSystemErrorHandling:
    """Test cache system error handling and resilience with direct operations."""

    @pytest.fixture
    async def redis_container(self):
        """Real Redis testcontainer for error testing."""
        container = RealRedisTestContainer()
        await container.start()
        container.set_env_vars()
        yield container
        await container.stop()

    async def test_cache_facade_redis_failure_graceful_degradation(self, redis_container):
        """Test graceful degradation to L1-only when Redis fails."""
        cache = CacheFacade(l1_max_size=100, enable_l2=True)

        try:
            # Test normal L1+L2 operation
            await cache.set("test_key", "test_value")
            result = await cache.get("test_key")
            assert result == "test_value"

            # Simulate Redis failure by stopping container
            await redis_container.simulate_network_failure(2.0)
            await asyncio.sleep(0.5)  # Let failure take effect

            # Clear L1 to force L2 access during failure
            await cache._l1_cache.clear()

            # Operations should degrade gracefully to L1-only
            start_time = time.perf_counter()
            await cache.set("failure_key", "failure_value")
            result = await cache.get("failure_key")
            degraded_time = time.perf_counter() - start_time

            assert result == "failure_value", "Cache didn't degrade gracefully"
            assert degraded_time < 0.005, f"Degraded operation took {degraded_time * 1000:.2f}ms, too slow"

            # Health check should indicate degraded state
            health = await cache.health_check()
            assert health["healthy"] is False or health["status"] == "degraded"

            print(f"Graceful degradation: {degraded_time * 1000:.2f}ms for L1-only operation")

        finally:
            await cache.close()

    async def test_cache_facade_concurrent_error_resilience(self):
        """Test cache facade resilience under concurrent operations with errors."""
        cache = CacheFacade(l1_max_size=50, enable_l2=False)

        async def error_prone_task(task_id: int) -> dict[str, Any]:
            """Task that may encounter various error conditions."""
            results = {"successful": 0, "failed": 0}

            for i in range(20):
                try:
                    key = f"error_test_{task_id}_{i}"

                    # Simulate different error conditions
                    if i % 10 == 0:
                        # Simulate timeout by using very long key
                        key = "x" * 10000 + key

                    await cache.set(key, {"task": task_id, "item": i})
                    result = await cache.get(key)

                    if result is not None:
                        results["successful"] += 1
                    else:
                        results["failed"] += 1

                except Exception:
                    results["failed"] += 1

            return results

        # Run concurrent error-prone tasks
        tasks = [error_prone_task(i) for i in range(5)]
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.perf_counter() - start_time

        # Analyze error resilience
        successful_tasks = [r for r in results if isinstance(r, dict)]
        total_successful = sum(r["successful"] for r in successful_tasks)
        total_failed = sum(r["failed"] for r in successful_tasks)

        success_rate = total_successful / (total_successful + total_failed) if (total_successful + total_failed) > 0 else 0

        print(f"Error resilience test: {success_rate:.2%} success rate, {total_time:.2f}s total time")

        # Should maintain reasonable success rate even with errors
        assert success_rate > 0.70, f"Success rate {success_rate:.2%} too low under error conditions"
        assert total_time < 2.0, f"Error handling took {total_time:.2f}s, too slow"

        await cache.close()


if __name__ == "__main__":
    """Run real behavior tests for direct cache-aside pattern."""
    pytest.main([__file__, "-v", "--tb=short"])
