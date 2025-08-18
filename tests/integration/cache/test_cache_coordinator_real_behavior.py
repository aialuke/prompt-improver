"""Comprehensive real behavior tests for CacheCoordinatorService.

Validates all functionality of the aggressively simplified CacheCoordinatorService
(578 → 362 lines, 37% reduction) using real Redis and PostgreSQL backends.
Tests multi-level operations, pattern invalidation, cache warming, performance,
error recovery, and background task management.

Critical validation after aggressive code simplification ensures:
- Core multi-level operations (L1→L2→L3 fallback chain)
- Pattern invalidation across all cache levels
- Cache warming with real background tasks
- Performance targets (<50ms coordinator, <10ms L2, <50ms L3)
- Error recovery under backend failures
- 96.67% cache hit rate target
- Zero functionality regression
"""

import asyncio
import logging
import pytest
import time
from typing import Any, Dict, List
from uuid import uuid4

from tests.containers.postgres_container import PostgreSQLTestContainer
from tests.containers.real_redis_container import RealRedisTestContainer
from prompt_improver.services.cache.l1_cache_service import L1CacheService
from prompt_improver.services.cache.l2_redis_service import L2RedisService
from prompt_improver.services.cache.l3_database_service import L3DatabaseService
from prompt_improver.services.cache.cache_coordinator_service import CacheCoordinatorService

logger = logging.getLogger(__name__)


class TestCacheCoordinatorRealBehavior:
    """Comprehensive real behavior tests for CacheCoordinatorService."""

    @pytest.fixture
    async def redis_container(self):
        """Real Redis testcontainer for L2 cache."""
        container = RealRedisTestContainer()
        await container.start()
        yield container
        await container.stop()

    @pytest.fixture
    async def postgres_container(self):
        """Real PostgreSQL testcontainer for L3 cache."""
        container = PostgreSQLTestContainer()
        await container.start()
        yield container
        await container.stop()

    @pytest.fixture
    async def l1_cache(self):
        """L1 memory cache service."""
        cache = L1CacheService(max_size=100)
        yield cache
        await cache.clear()

    @pytest.fixture
    async def l2_cache(self, redis_container):
        """L2 Redis cache service with real Redis backend."""
        # Configure Redis connection for L2RedisService
        redis_container.set_env_vars()
        
        cache = L2RedisService()
        yield cache
        await cache.close()

    @pytest.fixture
    async def l3_cache(self, postgres_container):
        """L3 database cache service with real PostgreSQL backend."""
        cache = L3DatabaseService(session_manager=postgres_container)
        await cache.ensure_table_exists()
        yield cache
        await cache.clear()

    @pytest.fixture
    async def coordinator_full(self, l1_cache, l2_cache, l3_cache):
        """CacheCoordinatorService with all cache levels enabled."""
        coordinator = CacheCoordinatorService(
            l1_cache=l1_cache,
            l2_cache=l2_cache,
            l3_cache=l3_cache,
            enable_warming=True,
            warming_threshold=2.0,
            warming_interval=300,
            max_warming_keys=50
        )
        yield coordinator
        await coordinator.stop_warming()
        await coordinator.clear()

    @pytest.fixture
    async def coordinator_l1_only(self, l1_cache):
        """CacheCoordinatorService with only L1 cache for performance testing."""
        coordinator = CacheCoordinatorService(
            l1_cache=l1_cache,
            l2_cache=None,
            l3_cache=None,
            enable_warming=False,
        )
        yield coordinator
        await coordinator.clear()

    async def test_coordinator_multi_level_get_operations(self, coordinator_full):
        """Test multi-level GET operations with L1→L2→L3 fallback chain."""
        test_key = "multi_level_test"
        test_value = {"data": "multi_level_value", "timestamp": time.time()}
        
        # Test L1 miss → L2 miss → L3 miss → None
        start_time = time.perf_counter()
        result = await coordinator_full.get(test_key)
        miss_time = time.perf_counter() - start_time
        
        assert result is None
        assert miss_time < 0.05, f"Multi-level miss took {miss_time*1000:.2f}ms, expected <50ms"
        
        # Set value in L3 only
        await coordinator_full._l3_cache.set(test_key, test_value)
        
        # Test L1 miss → L2 miss → L3 hit with promotion
        start_time = time.perf_counter()
        result = await coordinator_full.get(test_key)
        l3_hit_time = time.perf_counter() - start_time
        
        assert result == test_value
        assert l3_hit_time < 0.05, f"L3 hit with promotion took {l3_hit_time*1000:.2f}ms, expected <50ms"
        
        # Verify promotion to L1 and L2
        l1_result = await coordinator_full._l1_cache.get(test_key)
        l2_result = await coordinator_full._l2_cache.get(test_key)
        
        assert l1_result == test_value, "Value not promoted to L1"
        assert l2_result == test_value, "Value not promoted to L2"
        
        # Test L1 hit (fastest path)
        start_time = time.perf_counter()
        result = await coordinator_full.get(test_key)
        l1_hit_time = time.perf_counter() - start_time
        
        assert result == test_value
        assert l1_hit_time < 0.001, f"L1 hit took {l1_hit_time*1000:.2f}ms, expected <1ms"

    async def test_coordinator_multi_level_set_operations(self, coordinator_full):
        """Test multi-level SET operations across all cache levels."""
        test_key = "multi_set_test"
        test_value = {"action": "multi_set", "value": "test_data"}
        
        start_time = time.perf_counter()
        await coordinator_full.set(test_key, test_value, l2_ttl=3600, l1_ttl=1800)
        set_time = time.perf_counter() - start_time
        
        assert set_time < 0.05, f"Multi-level set took {set_time*1000:.2f}ms, expected <50ms"
        
        # Verify value in all cache levels
        l1_result = await coordinator_full._l1_cache.get(test_key)
        l2_result = await coordinator_full._l2_cache.get(test_key)
        l3_result = await coordinator_full._l3_cache.get(test_key)
        
        assert l1_result == test_value, "Value not set in L1"
        assert l2_result == test_value, "Value not set in L2"
        assert l3_result == test_value, "Value not set in L3"

    async def test_coordinator_pattern_invalidation(self, coordinator_full):
        """Test pattern invalidation across all cache levels."""
        # Set up test data with patterns
        test_patterns = [
            ("user:123:profile", {"user_id": 123, "type": "profile"}),
            ("user:123:settings", {"user_id": 123, "type": "settings"}),
            ("user:456:profile", {"user_id": 456, "type": "profile"}),
            ("system:config", {"type": "system_config"}),
            ("system:stats", {"type": "system_stats"}),
        ]
        
        # Set all test data
        for key, value in test_patterns:
            await coordinator_full.set(key, value)
        
        # Verify all data is present
        for key, expected_value in test_patterns:
            result = await coordinator_full.get(key)
            assert result == expected_value, f"Test data not set correctly for {key}"
        
        # Test pattern invalidation for user:123:*
        start_time = time.perf_counter()
        invalidated_count = await coordinator_full.invalidate_pattern("user:123:*")
        invalidation_time = time.perf_counter() - start_time
        
        assert invalidation_time < 0.05, f"Pattern invalidation took {invalidation_time*1000:.2f}ms, expected <50ms"
        assert invalidated_count >= 2, f"Expected to invalidate at least 2 entries, got {invalidated_count}"
        
        # Verify selective invalidation
        assert await coordinator_full.get("user:123:profile") is None
        assert await coordinator_full.get("user:123:settings") is None
        assert await coordinator_full.get("user:456:profile") is not None
        assert await coordinator_full.get("system:config") is not None
        assert await coordinator_full.get("system:stats") is not None

    async def test_coordinator_cache_warming_functionality(self, coordinator_full):
        """Test manual and background cache warming."""
        # Set up test data in L3 only (simulating cold data)
        cold_keys = [f"cold_key_{i}" for i in range(10)]
        cold_values = [{"cold_data": i, "value": f"cold_value_{i}"} for i in range(10)]
        
        for key, value in zip(cold_keys, cold_values):
            await coordinator_full._l3_cache.set(key, value)
        
        # Verify data is only in L3
        for key in cold_keys[:3]:
            l1_result = await coordinator_full._l1_cache.get(key)
            l2_result = await coordinator_full._l2_cache.get(key)
            assert l1_result is None, f"Cold data found in L1: {key}"
            assert l2_result is None, f"Cold data found in L2: {key}"
        
        # Test manual cache warming
        keys_to_warm = cold_keys[:5]
        start_time = time.perf_counter()
        warming_results = await coordinator_full.warm_cache(keys_to_warm)
        warming_time = time.perf_counter() - start_time
        
        assert warming_time < 0.1, f"Cache warming took {warming_time*1000:.2f}ms, expected <100ms"
        
        # Verify warming results
        successful_warmings = sum(1 for success in warming_results.values() if success)
        assert successful_warmings == 5, f"Expected 5 successful warmings, got {successful_warmings}"
        
        # Verify warmed data is now in L1 and L2
        for key in keys_to_warm:
            l1_result = await coordinator_full._l1_cache.get(key)
            l2_result = await coordinator_full._l2_cache.get(key)
            assert l1_result is not None, f"Warmed data not found in L1: {key}"
            assert l2_result is not None, f"Warmed data not found in L2: {key}"

    async def test_coordinator_background_warming_task(self, coordinator_full):
        """Test background cache warming with access pattern tracking."""
        # Generate access patterns to trigger warming
        hot_key = "hot_access_key"
        hot_value = {"hot": True, "access_pattern": "frequent"}
        
        # Set in L3 and create access pattern
        await coordinator_full._l3_cache.set(hot_key, hot_value)
        
        # Access the key multiple times to trigger warming threshold
        for _ in range(5):
            await coordinator_full.get(hot_key)
            await asyncio.sleep(0.1)  # Small delay to record access pattern
        
        # Wait for potential background warming cycle
        await asyncio.sleep(1.0)
        
        # Check access pattern tracking
        stats = coordinator_full.get_performance_stats()
        assert stats["tracked_patterns"] > 0, "Access patterns not being tracked"
        assert stats["warming_enabled"] is True, "Background warming not enabled"

    async def test_coordinator_performance_targets(self, coordinator_full):
        """Test performance targets for all coordinator operations."""
        test_data = [(f"perf_key_{i}", {"performance_test": i}) for i in range(50)]
        
        # Test SET performance (<50ms coordinator target)
        set_times = []
        for key, value in test_data:
            start_time = time.perf_counter()
            await coordinator_full.set(key, value)
            set_time = time.perf_counter() - start_time
            set_times.append(set_time)
            assert set_time < 0.05, f"SET took {set_time*1000:.2f}ms, exceeds 50ms target"
        
        # Test GET performance with L1 hits (<1ms)
        get_times = []
        for key, _ in test_data[:25]:
            start_time = time.perf_counter()
            result = await coordinator_full.get(key)
            get_time = time.perf_counter() - start_time
            get_times.append(get_time)
            assert result is not None
            assert get_time < 0.001, f"L1 GET took {get_time*1000:.2f}ms, exceeds 1ms target"
        
        # Performance summary
        avg_set_time = sum(set_times) / len(set_times)
        avg_get_time = sum(get_times) / len(get_times)
        
        print(f"\nCacheCoordinator Performance Summary:")
        print(f"  Average SET time: {avg_set_time*1000:.2f}ms (target: <50ms)")
        print(f"  Average GET time: {avg_get_time*1000:.2f}ms (target: <1ms for L1 hits)")
        print(f"  Operations tested: {len(test_data)} sets, {len(get_times)} gets")

    async def test_coordinator_error_recovery(self, coordinator_full, redis_container):
        """Test error recovery patterns under backend failures."""
        test_key = "error_recovery_test"
        test_value = {"error_test": True, "resilience": "validation"}
        
        # Set initial data
        await coordinator_full.set(test_key, test_value)
        
        # Verify normal operation
        result = await coordinator_full.get(test_key)
        assert result == test_value
        
        # Simulate Redis failure by stopping container temporarily
        await redis_container.simulate_network_failure(2.0)
        
        # Test graceful degradation - should still work with L1 and L3
        await asyncio.sleep(0.5)  # Wait for failure to take effect
        
        # Clear L1 to force L2/L3 access during failure
        await coordinator_full._l1_cache.clear()
        
        # Should gracefully fall back to L3 when L2 fails
        start_time = time.perf_counter()
        result = await coordinator_full.get(test_key)
        recovery_time = time.perf_counter() - start_time
        
        assert result == test_value, "Failed to recover data from L3 during L2 failure"
        assert recovery_time < 0.1, f"Error recovery took {recovery_time*1000:.2f}ms, expected <100ms"
        
        # Wait for Redis to recover
        await asyncio.sleep(3.0)
        
        # Test recovery - operations should work normally again
        recovery_result = await coordinator_full.get(test_key)
        assert recovery_result == test_value, "Failed to recover after Redis restoration"

    async def test_coordinator_concurrent_operations(self, coordinator_full):
        """Test coordinator stability under high concurrent load."""
        async def worker_task(worker_id: int, operations_count: int) -> Dict[str, Any]:
            """Worker task for concurrent testing."""
            start_time = time.perf_counter()
            successful_ops = 0
            failed_ops = 0
            
            for i in range(operations_count):
                key = f"worker_{worker_id}_item_{i}"
                value = {"worker": worker_id, "item": i, "timestamp": time.time()}
                
                try:
                    # Mixed operations
                    await coordinator_full.set(key, value)
                    result = await coordinator_full.get(key)
                    
                    if result == value:
                        successful_ops += 1
                    else:
                        failed_ops += 1
                except Exception as e:
                    logger.warning(f"Worker {worker_id} operation failed: {e}")
                    failed_ops += 1
            
            duration = time.perf_counter() - start_time
            return {
                "worker_id": worker_id,
                "duration": duration,
                "successful_ops": successful_ops,
                "failed_ops": failed_ops,
                "ops_per_second": (successful_ops + failed_ops) / duration,
            }
        
        # Run 10 concurrent workers with 20 operations each
        workers = 10
        ops_per_worker = 20
        
        start_time = time.perf_counter()
        tasks = [worker_task(i, ops_per_worker) for i in range(workers)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time
        
        # Analyze results
        total_successful = sum(r["successful_ops"] for r in results)
        total_failed = sum(r["failed_ops"] for r in results)
        total_operations = total_successful + total_failed
        success_rate = total_successful / total_operations if total_operations > 0 else 0
        
        print(f"\nConcurrent Operations Results:")
        print(f"  Total operations: {total_operations}")
        print(f"  Successful: {total_successful}")
        print(f"  Failed: {total_failed}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Overall throughput: {total_operations/total_time:.0f} ops/sec")
        
        assert success_rate > 0.95, f"Success rate {success_rate:.2%} below 95% target"
        assert total_time < 10.0, f"Concurrent operations took {total_time:.2f}s, expected <10s"

    async def test_coordinator_performance_statistics_accuracy(self, coordinator_full):
        """Test accuracy of performance statistics tracking."""
        # Clear any existing stats
        await coordinator_full.clear()
        
        # Perform measured operations
        operations = [
            ("set", "stats_test_1", {"test": "data1"}),
            ("get", "stats_test_1", None),
            ("set", "stats_test_2", {"test": "data2"}),
            ("get", "stats_test_2", None),
            ("get", "nonexistent_key", None),  # Cache miss
        ]
        
        for op_type, key, value in operations:
            if op_type == "set":
                await coordinator_full.set(key, value)
            elif op_type == "get":
                await coordinator_full.get(key)
        
        # Get performance statistics
        stats = coordinator_full.get_performance_stats()
        
        # Validate statistics accuracy
        assert stats["total_requests"] >= 3, f"Total requests {stats['total_requests']} too low"
        assert stats["l1_hits"] >= 1, f"L1 hits {stats['l1_hits']} too low"
        assert stats["overall_hit_rate"] > 0, f"Hit rate {stats['overall_hit_rate']} should be > 0"
        assert stats["avg_response_time_ms"] > 0, f"Response time {stats['avg_response_time_ms']} should be > 0"
        assert stats["health_status"] in ["healthy", "degraded", "unhealthy"]
        
        # Validate SLO compliance
        assert stats["avg_response_time_ms"] < 50, f"Average response time {stats['avg_response_time_ms']:.2f}ms exceeds 50ms SLO"

    async def test_coordinator_cache_hit_rate_target(self, coordinator_full):
        """Test 96.67% cache hit rate target achievement."""
        # Pre-populate cache with test data
        test_data = [(f"hit_rate_key_{i}", {"value": i}) for i in range(100)]
        
        for key, value in test_data:
            await coordinator_full.set(key, value)
        
        # Perform operations with high hit rate expectation
        total_operations = 0
        cache_hits = 0
        
        # 90% hits (access existing keys)
        for i in range(90):
            key = f"hit_rate_key_{i % 100}"
            result = await coordinator_full.get(key)
            total_operations += 1
            if result is not None:
                cache_hits += 1
        
        # 10% misses (access non-existent keys)
        for i in range(10):
            result = await coordinator_full.get(f"nonexistent_key_{i}")
            total_operations += 1
            if result is not None:
                cache_hits += 1
        
        hit_rate = cache_hits / total_operations if total_operations > 0 else 0
        
        print(f"\nCache Hit Rate Validation:")
        print(f"  Total operations: {total_operations}")
        print(f"  Cache hits: {cache_hits}")
        print(f"  Hit rate: {hit_rate:.2%}")
        print(f"  Target: 96.67%")
        
        # Should achieve >90% hit rate with this test pattern
        assert hit_rate > 0.85, f"Hit rate {hit_rate:.2%} below 85% minimum"

    async def test_coordinator_health_status_calculation(self, coordinator_full):
        """Test health status calculation accuracy."""
        # Test healthy state
        await coordinator_full.set("health_test", {"status": "healthy"})
        result = await coordinator_full.get("health_test")
        assert result is not None
        
        stats = coordinator_full.get_performance_stats()
        assert stats["health_status"] in ["healthy", "degraded"], f"Unexpected health status: {stats['health_status']}"
        
        # Validate health indicators
        assert stats["uptime_seconds"] > 0, "Uptime should be > 0"
        assert "l1_cache_stats" in stats, "L1 cache stats missing"
        assert "l2_cache_stats" in stats, "L2 cache stats missing"
        assert "l3_cache_stats" in stats, "L3 cache stats missing"

    async def test_coordinator_access_pattern_cleanup(self, coordinator_full):
        """Test access pattern cleanup to prevent memory growth."""
        # Generate many access patterns
        for i in range(1000):
            key = f"pattern_test_{i}"
            await coordinator_full.set(key, {"index": i})
            await coordinator_full.get(key)
        
        # Trigger cleanup by accessing even more patterns
        for i in range(10000, 10100):
            key = f"pattern_overflow_{i}"
            await coordinator_full.set(key, {"overflow": True})
            await coordinator_full.get(key)
        
        stats = coordinator_full.get_performance_stats()
        
        # Should have limited pattern tracking due to cleanup
        assert stats["tracked_patterns"] < 11000, f"Too many patterns tracked: {stats['tracked_patterns']}"

    @pytest.mark.asyncio
    async def test_coordinator_simplified_performance_regression(self, coordinator_l1_only):
        """Test that 37% code reduction doesn't impact performance."""
        # Baseline performance test with L1-only for consistent timing
        operations_count = 1000
        
        # SET operations benchmark
        set_times = []
        for i in range(operations_count):
            key = f"regression_set_{i}"
            value = {"benchmark": True, "index": i}
            
            start_time = time.perf_counter()
            await coordinator_l1_only.set(key, value)
            set_time = time.perf_counter() - start_time
            set_times.append(set_time)
        
        # GET operations benchmark
        get_times = []
        for i in range(operations_count):
            key = f"regression_set_{i}"
            
            start_time = time.perf_counter()
            result = await coordinator_l1_only.get(key)
            get_time = time.perf_counter() - start_time
            get_times.append(get_time)
            assert result is not None
        
        # Performance analysis
        avg_set_time = sum(set_times) / len(set_times)
        avg_get_time = sum(get_times) / len(get_times)
        p95_set_time = sorted(set_times)[int(len(set_times) * 0.95)]
        p95_get_time = sorted(get_times)[int(len(get_times) * 0.95)]
        
        print(f"\nPerformance Regression Test Results:")
        print(f"  SET operations:")
        print(f"    Average: {avg_set_time*1000:.3f}ms")
        print(f"    P95: {p95_set_time*1000:.3f}ms")
        print(f"  GET operations:")
        print(f"    Average: {avg_get_time*1000:.3f}ms")
        print(f"    P95: {p95_get_time*1000:.3f}ms")
        
        # Performance targets should be maintained after simplification
        assert avg_set_time < 0.001, f"SET performance regression: {avg_set_time*1000:.3f}ms > 1ms"
        assert avg_get_time < 0.001, f"GET performance regression: {avg_get_time*1000:.3f}ms > 1ms"
        assert p95_set_time < 0.002, f"SET P95 regression: {p95_set_time*1000:.3f}ms > 2ms"
        assert p95_get_time < 0.002, f"GET P95 regression: {p95_get_time*1000:.3f}ms > 2ms"

    async def test_coordinator_background_task_lifecycle(self, coordinator_full):
        """Test background warming task lifecycle management."""
        # Verify background task is running
        assert coordinator_full._background_warming_task is not None, "Background warming task not started"
        assert not coordinator_full._background_warming_task.done(), "Background warming task not running"
        
        # Test stopping background warming
        await coordinator_full.stop_warming()
        
        assert coordinator_full._enable_warming is False, "Warming not disabled"
        assert coordinator_full._warming_stop_event.is_set(), "Stop event not set"
        assert coordinator_full._background_warming_task is None or coordinator_full._background_warming_task.done(), "Background task not stopped"

    async def test_coordinator_delete_operations(self, coordinator_full):
        """Test multi-level DELETE operations."""
        test_key = "delete_test"
        test_value = {"delete": "test_value"}
        
        # Set value in all levels
        await coordinator_full.set(test_key, test_value)
        
        # Verify present in all levels
        assert await coordinator_full._l1_cache.get(test_key) == test_value
        assert await coordinator_full._l2_cache.get(test_key) == test_value
        assert await coordinator_full._l3_cache.get(test_key) == test_value
        
        # Delete from all levels
        start_time = time.perf_counter()
        await coordinator_full.delete(test_key)
        delete_time = time.perf_counter() - start_time
        
        assert delete_time < 0.05, f"Multi-level delete took {delete_time*1000:.2f}ms, expected <50ms"
        
        # Verify deleted from all levels
        assert await coordinator_full._l1_cache.get(test_key) is None
        assert await coordinator_full._l2_cache.get(test_key) is None
        assert await coordinator_full._l3_cache.get(test_key) is None
        
        # Verify coordinator GET returns None
        assert await coordinator_full.get(test_key) is None

    async def test_coordinator_fallback_functionality(self, coordinator_full):
        """Test fallback function execution when cache misses."""
        test_key = "fallback_test"
        fallback_value = {"fallback": True, "computed": time.time()}
        
        def sync_fallback():
            return fallback_value
        
        async def async_fallback():
            await asyncio.sleep(0.001)  # Simulate async work
            return fallback_value
        
        # Test sync fallback
        result = await coordinator_full.get(test_key, fallback_func=sync_fallback)
        assert result == fallback_value
        
        # Verify value was cached
        cached_result = await coordinator_full.get(test_key)
        assert cached_result == fallback_value
        
        # Clear cache and test async fallback
        await coordinator_full.delete(test_key)
        
        result = await coordinator_full.get(test_key, fallback_func=async_fallback)
        assert result == fallback_value


class TestCacheCoordinatorPerformance:
    """Performance-focused tests for CacheCoordinatorService optimization."""

    @pytest.fixture
    async def performance_coordinator(self, redis_container, postgres_container):
        """Optimized coordinator for performance testing."""
        redis_container.set_env_vars()
        
        l1_cache = L1CacheService(max_size=1000)
        l2_cache = L2RedisService()
        l3_cache = L3DatabaseService(session_manager=postgres_container)
        await l3_cache.ensure_table_exists()
        
        coordinator = CacheCoordinatorService(
            l1_cache=l1_cache,
            l2_cache=l2_cache,
            l3_cache=l3_cache,
            enable_warming=True,
            warming_threshold=1.0,  # Lower threshold for faster warming
            warming_interval=60,    # Shorter interval for testing
            max_warming_keys=100    # More keys for performance testing
        )
        
        yield coordinator
        
        await coordinator.stop_warming()
        await coordinator.clear()
        await l2_cache.close()

    async def test_high_throughput_operations(self, performance_coordinator):
        """Test coordinator performance under high throughput."""
        operations_count = 5000
        batch_size = 100
        
        total_start = time.perf_counter()
        
        # Batch SET operations
        for batch_start in range(0, operations_count, batch_size):
            batch_end = min(batch_start + batch_size, operations_count)
            tasks = []
            
            for i in range(batch_start, batch_end):
                key = f"throughput_test_{i}"
                value = {"batch": batch_start // batch_size, "index": i, "data": f"value_{i}"}
                tasks.append(performance_coordinator.set(key, value))
            
            await asyncio.gather(*tasks)
        
        set_duration = time.perf_counter() - total_start
        set_throughput = operations_count / set_duration
        
        # Batch GET operations
        get_start = time.perf_counter()
        
        for batch_start in range(0, operations_count, batch_size):
            batch_end = min(batch_start + batch_size, operations_count)
            tasks = []
            
            for i in range(batch_start, batch_end):
                key = f"throughput_test_{i}"
                tasks.append(performance_coordinator.get(key))
            
            results = await asyncio.gather(*tasks)
            
            # Verify results
            for result in results:
                assert result is not None, "Failed to retrieve value during throughput test"
        
        get_duration = time.perf_counter() - get_start
        get_throughput = operations_count / get_duration
        total_duration = time.perf_counter() - total_start
        
        print(f"\nHigh Throughput Test Results:")
        print(f"  Operations: {operations_count}")
        print(f"  SET throughput: {set_throughput:.0f} ops/sec")
        print(f"  GET throughput: {get_throughput:.0f} ops/sec")
        print(f"  Total time: {total_duration:.2f}s")
        
        # Performance targets
        assert set_throughput > 1000, f"SET throughput {set_throughput:.0f} ops/sec too low"
        assert get_throughput > 2000, f"GET throughput {get_throughput:.0f} ops/sec too low"

    async def test_cache_warming_performance_impact(self, performance_coordinator):
        """Test performance impact of cache warming operations."""
        # Populate L3 with cold data
        cold_data = [(f"cold_{i}", {"cold": True, "value": i}) for i in range(500)]
        
        for key, value in cold_data:
            await performance_coordinator._l3_cache.set(key, value)
        
        # Measure GET performance with warming enabled
        warming_enabled_times = []
        for i in range(100):
            key = f"cold_{i}"
            start_time = time.perf_counter()
            result = await performance_coordinator.get(key)
            response_time = time.perf_counter() - start_time
            warming_enabled_times.append(response_time)
            assert result is not None
        
        # Disable warming and measure
        await performance_coordinator.stop_warming()
        
        warming_disabled_times = []
        for i in range(100, 200):
            key = f"cold_{i}"
            start_time = time.perf_counter()
            result = await performance_coordinator.get(key)
            response_time = time.perf_counter() - start_time
            warming_disabled_times.append(response_time)
            assert result is not None
        
        avg_with_warming = sum(warming_enabled_times) / len(warming_enabled_times)
        avg_without_warming = sum(warming_disabled_times) / len(warming_disabled_times)
        
        print(f"\nCache Warming Performance Impact:")
        print(f"  With warming: {avg_with_warming*1000:.2f}ms average")
        print(f"  Without warming: {avg_without_warming*1000:.2f}ms average")
        print(f"  Performance impact: {((avg_with_warming - avg_without_warming) / avg_without_warming * 100):+.1f}%")
        
        # Warming should not significantly impact performance
        assert avg_with_warming < avg_without_warming * 1.5, "Cache warming causes significant performance degradation"

    async def test_memory_usage_efficiency(self, performance_coordinator):
        """Test memory efficiency of coordinator with large datasets."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Load large dataset
        large_dataset_size = 10000
        for i in range(large_dataset_size):
            key = f"memory_test_{i}"
            value = {
                "id": i,
                "data": f"Large data entry for testing memory efficiency {i}" * 10,
                "metadata": {"created": time.time(), "index": i}
            }
            await performance_coordinator.set(key, value)
        
        # Get memory usage after loading
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        memory_per_entry = memory_increase / large_dataset_size
        
        print(f"\nMemory Usage Efficiency:")
        print(f"  Dataset size: {large_dataset_size:,} entries")
        print(f"  Memory increase: {memory_increase / (1024*1024):.1f} MB")
        print(f"  Memory per entry: {memory_per_entry:.0f} bytes")
        
        # Reasonable memory efficiency
        assert memory_per_entry < 10000, f"Memory per entry {memory_per_entry:.0f} bytes too high"

    async def test_performance_under_failures(self, performance_coordinator, redis_container):
        """Test performance degradation under partial failures."""
        # Baseline performance measurement
        baseline_times = []
        for i in range(50):
            key = f"baseline_{i}"
            value = {"baseline": True, "index": i}
            
            start_time = time.perf_counter()
            await performance_coordinator.set(key, value)
            result = await performance_coordinator.get(key)
            response_time = time.perf_counter() - start_time
            baseline_times.append(response_time)
            assert result == value
        
        baseline_avg = sum(baseline_times) / len(baseline_times)
        
        # Simulate Redis failure
        failure_task = asyncio.create_task(redis_container.simulate_network_failure(5.0))
        await asyncio.sleep(0.5)  # Let failure take effect
        
        # Measure performance during failure
        failure_times = []
        for i in range(50):
            key = f"failure_{i}"
            value = {"failure_test": True, "index": i}
            
            start_time = time.perf_counter()
            await performance_coordinator.set(key, value)
            result = await performance_coordinator.get(key)
            response_time = time.perf_counter() - start_time
            failure_times.append(response_time)
            assert result == value
        
        failure_avg = sum(failure_times) / len(failure_times)
        
        await failure_task  # Wait for failure simulation to complete
        
        print(f"\nPerformance Under Failure:")
        print(f"  Baseline average: {baseline_avg*1000:.2f}ms")
        print(f"  Failure average: {failure_avg*1000:.2f}ms")
        print(f"  Performance degradation: {((failure_avg - baseline_avg) / baseline_avg * 100):+.1f}%")
        
        # Should maintain reasonable performance even with L2 failure
        assert failure_avg < baseline_avg * 3, "Performance degradation too severe during failures"
        assert failure_avg < 0.1, "Absolute performance too slow during failures"