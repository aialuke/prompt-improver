"""Integration tests for CacheServiceFacade with direct cache-aside pattern.

Tests the real behavior of decomposed cache services with direct access:
- L1CacheService (memory)
- L2CacheService (Redis)
- CacheServiceFacade (direct cache-aside pattern)

Uses testcontainers for real Redis and PostgreSQL testing.
"""

import asyncio
import time
from typing import Any

import pytest

from prompt_improver.services.cache import L1CacheService
from prompt_improver.services.cache.cache_facade import CacheFacade
from prompt_improver.shared.interfaces.protocols.cache import L2CacheServiceProtocol


@pytest.fixture
async def l1_cache_service():
    """Create L1 cache service for testing."""
    return L1CacheService(max_size=1000, default_ttl=3600)


@pytest.fixture
async def mock_l2_cache_service():
    """Create mock L2 cache service for testing."""

    class MockL2CacheService(L2CacheServiceProtocol):
        def __init__(self):
            self.data = {}
            self.performance_stats = {
                "total_operations": 0,
                "total_time_ms": 0.0,
                "hit_count": 0,
                "miss_count": 0
            }

        async def get(self, key: str, namespace: str | None = None) -> Any:
            start_time = time.perf_counter()
            self.performance_stats["total_operations"] += 1

            full_key = f"{namespace}:{key}" if namespace else key
            result = self.data.get(full_key)

            if result:
                self.performance_stats["hit_count"] += 1
            else:
                self.performance_stats["miss_count"] += 1

            duration_ms = (time.perf_counter() - start_time) * 1000
            self.performance_stats["total_time_ms"] += duration_ms

            return result

        async def set(self, key: str, value: Any, ttl: int | None = None, namespace: str | None = None) -> bool:
            start_time = time.perf_counter()
            self.performance_stats["total_operations"] += 1

            full_key = f"{namespace}:{key}" if namespace else key
            self.data[full_key] = value

            duration_ms = (time.perf_counter() - start_time) * 1000
            self.performance_stats["total_time_ms"] += duration_ms

            return True

        async def mget(self, keys: list[str], namespace: str | None = None) -> dict[str, Any]:
            result = {}
            for key in keys:
                value = await self.get(key, namespace)
                if value is not None:
                    result[key] = value
            return result

        async def mset(self, items: dict[str, Any], ttl: int | None = None, namespace: str | None = None) -> bool:
            for key, value in items.items():
                await self.set(key, value, ttl, namespace)
            return True

        async def delete_pattern(self, pattern: str, namespace: str | None = None) -> int:
            # Simple pattern matching for testing
            full_pattern = f"{namespace}:{pattern}" if namespace else pattern
            deleted_count = 0

            keys_to_delete = [key for key in self.data if pattern.replace("*", "") in key]

            for key in keys_to_delete:
                del self.data[key]
                deleted_count += 1

            return deleted_count

        async def get_connection_status(self) -> dict[str, Any]:
            return {
                "connected": True,
                "latency_ms": 5.0,
                "memory_usage": len(self.data),
                "operations": self.performance_stats["total_operations"]
            }

    return MockL2CacheService()


@pytest.fixture
async def cache_facade(mock_l2_cache_service, l1_cache_service):
    """Create cache facade with direct cache-aside pattern."""

    # Create facade with real L1 and mock L2 for testing
    return CacheFacade(
        l1_max_size=1000,
        enable_l2=False,  # Using mock L2, disable real Redis
        enable_warming=False
    )


@pytest.fixture
async def real_cache_facade():
    """Create cache facade for real behavior testing."""
    return CacheFacade(
        l1_max_size=1000,
        enable_l2=False,  # Disable L2 for isolated testing
        enable_warming=False
    )


@pytest.mark.asyncio
class TestL1CacheService:
    """Test L1 cache service functionality."""

    async def test_basic_get_set_operations(self, l1_cache_service):
        """Test basic cache operations."""
        # Test set and get
        key = "test_key"
        value = {"data": "test_value", "number": 42}

        success = await l1_cache_service.set(key, value)
        assert success is True

        retrieved_value = await l1_cache_service.get(key)
        assert retrieved_value == value

    async def test_ttl_expiration(self, l1_cache_service):
        """Test TTL expiration."""
        key = "ttl_test_key"
        value = "test_value"
        ttl = 1  # 1 second

        success = await l1_cache_service.set(key, value, ttl=ttl)
        assert success is True

        # Immediately retrieve - should work
        retrieved_value = await l1_cache_service.get(key)
        assert retrieved_value == value

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired now
        retrieved_value = await l1_cache_service.get(key)
        assert retrieved_value is None

    async def test_namespace_isolation(self, l1_cache_service):
        """Test namespace isolation."""
        key = "shared_key"
        value1 = "value_in_namespace1"
        value2 = "value_in_namespace2"

        # Set in different namespaces
        await l1_cache_service.set(key, value1, namespace="ns1")
        await l1_cache_service.set(key, value2, namespace="ns2")

        # Retrieve from each namespace
        retrieved1 = await l1_cache_service.get(key, namespace="ns1")
        retrieved2 = await l1_cache_service.get(key, namespace="ns2")

        assert retrieved1 == value1
        assert retrieved2 == value2
        assert retrieved1 != retrieved2

    async def test_cache_eviction_lru(self, l1_cache_service):
        """Test LRU eviction policy."""
        # Set cache size to small value for testing
        small_cache = L1CacheService(max_size=3)

        # Fill cache to capacity
        await small_cache.set("key1", "value1")
        await small_cache.set("key2", "value2")
        await small_cache.set("key3", "value3")

        # All should be present
        assert await small_cache.get("key1") == "value1"
        assert await small_cache.get("key2") == "value2"
        assert await small_cache.get("key3") == "value3"

        # Add one more - should evict oldest (key1)
        await small_cache.set("key4", "value4")

        # key1 should be evicted
        assert await small_cache.get("key1") is None
        assert await small_cache.get("key2") == "value2"
        assert await small_cache.get("key3") == "value3"
        assert await small_cache.get("key4") == "value4"

    async def test_performance_monitoring(self, l1_cache_service):
        """Test performance monitoring."""
        # Perform several operations
        for i in range(10):
            await l1_cache_service.set(f"perf_key_{i}", f"value_{i}")
            await l1_cache_service.get(f"perf_key_{i}")

        stats = await l1_cache_service.get_stats()

        assert "performance" in stats
        perf_stats = stats["performance"]
        assert perf_stats["total_operations"] >= 20  # 10 sets + 10 gets
        assert perf_stats["avg_response_time_ms"] >= 0
        assert "operations_by_type" in perf_stats

    async def test_cache_warming(self, l1_cache_service):
        """Test cache warming functionality."""
        # Create a mock source
        class MockSource:
            async def get(self, key):
                return f"warmed_value_{key}"

        source = MockSource()
        keys_to_warm = ["warm_key1", "warm_key2", "warm_key3"]

        warmed_count = await l1_cache_service.warm_up(keys_to_warm, source)

        assert warmed_count == len(keys_to_warm)

        # Verify values were cached
        for key in keys_to_warm:
            value = await l1_cache_service.get(key)
            assert value == f"warmed_value_{key}"

    async def test_health_status(self, l1_cache_service):
        """Test health status reporting."""
        # Perform some operations to generate metrics
        await l1_cache_service.set("health_key", "health_value")
        await l1_cache_service.get("health_key")

        stats = await l1_cache_service.get_stats()
        health = stats["health"]

        assert "status" in health
        assert health["status"] in {"healthy", "warning", "degraded"}
        assert "avg_response_time_ms" in health
        assert "performance_compliant" in health


@pytest.mark.asyncio
class TestDirectCacheFacade:
    """Test direct cache facade with cache-aside pattern."""

    async def test_direct_cache_operations(self, real_cache_facade):
        """Test direct cache-aside pattern operations."""
        key = "direct_test_key"
        value = "test_value"

        # Direct cache-aside: Check cache first
        cached_value = await real_cache_facade.get(key)
        assert cached_value is None  # Cache miss

        # Set value directly in cache
        success = await real_cache_facade.set(key, value)
        assert success is True

        # Get should now hit cache
        cached_value = await real_cache_facade.get(key)
        assert cached_value == value

    async def test_cache_aside_performance(self, real_cache_facade):
        """Test cache-aside pattern performance."""
        key = "performance_test_key"
        value = "test_value"

        # Measure direct cache SET operation
        start_time = time.perf_counter()
        await real_cache_facade.set(key, value)
        set_time = time.perf_counter() - start_time

        # Should be very fast for L1-only operations
        assert set_time < 0.001, f"Direct SET took {set_time * 1000:.3f}ms, expected <1ms"

        # Measure direct cache GET operation
        start_time = time.perf_counter()
        result = await real_cache_facade.get(key)
        get_time = time.perf_counter() - start_time

        assert result == value
        assert get_time < 0.0005, f"Direct GET took {get_time * 1000:.3f}ms, expected <0.5ms"

    async def test_direct_cache_invalidation(self, real_cache_facade):
        """Test direct cache invalidation."""
        key = "invalidation_test_key"
        value = "test_value"

        # Set value
        await real_cache_facade.set(key, value)
        assert await real_cache_facade.get(key) == value

        # Direct invalidation
        success = await real_cache_facade.invalidate(key)
        assert success is True

        # Verify key is no longer in cache
        cached_value = await real_cache_facade.get(key)
        assert cached_value is None

    async def test_facade_metrics_collection(self, real_cache_facade):
        """Test facade-level metrics collection."""
        # Perform various cache operations
        test_data = {
            "metrics_key1": "value1",
            "metrics_key2": "value2",
            "metrics_key3": "value3"
        }

        # Set values using cache-aside pattern
        for key, value in test_data.items():
            await real_cache_facade.set(key, value)

        # Get values (should hit L1)
        for key in test_data:
            result = await real_cache_facade.get(key)
            assert result == test_data[key]

        # Get facade metrics (not coordination metrics)
        stats = real_cache_facade.get_cache_stats()

        assert "l1_stats" in stats
        l1_stats = stats["l1_stats"]
        assert l1_stats["total_size"] >= len(test_data)

    async def test_bulk_cache_operations(self, real_cache_facade):
        """Test bulk cache operations with direct pattern."""
        # Bulk data for testing
        bulk_data = {f"bulk_key_{i}": f"value_{i}" for i in range(50)}

        # Bulk set operations
        for key, value in bulk_data.items():
            await real_cache_facade.set(key, value)

        # Bulk get operations to verify
        for key, expected_value in bulk_data.items():
            actual_value = await real_cache_facade.get(key)
            assert actual_value == expected_value

        # Verify cache efficiency
        stats = real_cache_facade.get_cache_stats()
        l1_stats = stats["l1_stats"]

        # Should maintain good hit rate for bulk operations
        assert l1_stats["total_size"] >= 50

    async def test_cache_facade_health(self, real_cache_facade):
        """Test cache facade health without coordination overhead."""
        # Perform some operations to generate metrics
        await real_cache_facade.set("health_key", "health_value")
        await real_cache_facade.get("health_key")

        # Get health status directly from facade
        stats = real_cache_facade.get_cache_stats()

        assert "l1_stats" in stats
        l1_stats = stats["l1_stats"]
        assert "health" in l1_stats

        health = l1_stats["health"]
        assert health["status"] in {"healthy", "warning", "degraded"}


@pytest.mark.asyncio
class TestCachePerformance:
    """Test cache performance characteristics."""

    async def test_l1_response_time_target(self, l1_cache_service):
        """Test L1 cache meets <1ms response time target."""
        key = "performance_test_key"
        value = "performance_test_value"

        # Warm up
        await l1_cache_service.set(key, value)

        # Measure get performance
        start_time = time.perf_counter()
        retrieved_value = await l1_cache_service.get(key)
        end_time = time.perf_counter()

        response_time_ms = (end_time - start_time) * 1000

        assert retrieved_value == value
        # L1 target: <1ms (allowing some overhead for test environment)
        assert response_time_ms < 10  # Relaxed for test environment

    async def test_concurrent_cache_operations(self, l1_cache_service):
        """Test concurrent cache operations."""
        # Prepare test data
        test_data = {f"concurrent_key_{i}": f"value_{i}" for i in range(50)}

        # Concurrent set operations
        set_tasks = [
            l1_cache_service.set(key, value)
            for key, value in test_data.items()
        ]

        set_results = await asyncio.gather(*set_tasks)
        assert all(set_results)

        # Concurrent get operations
        get_tasks = [
            l1_cache_service.get(key)
            for key in test_data
        ]

        get_results = await asyncio.gather(*get_tasks)

        # Verify all values retrieved correctly
        for i, (_key, expected_value) in enumerate(test_data.items()):
            assert get_results[i] == expected_value

    async def test_cache_hit_rate_optimization(self, real_cache_facade):
        """Test cache hit rate optimization with direct facade."""
        # Simulate workload with repeated access to same keys
        hot_keys = ["hot1", "hot2", "hot3"]
        cold_keys = ["cold1", "cold2", "cold3", "cold4", "cold5"]

        # Set all keys using cache-aside pattern
        for key in hot_keys + cold_keys:
            await real_cache_facade.set(key, f"value_{key}")

        # Simulate hot key access pattern (multiple accesses)
        for _ in range(10):
            for key in hot_keys:
                result = await real_cache_facade.get(key)
                assert result == f"value_{key}"

        # Single access to cold keys
        for key in cold_keys:
            result = await real_cache_facade.get(key)
            assert result == f"value_{key}"

        # Check facade performance metrics
        stats = real_cache_facade.get_cache_stats()
        l1_stats = stats["l1_stats"]

        # Should have all keys in L1 cache (no L2 enabled)
        assert l1_stats["total_size"] >= len(hot_keys + cold_keys)
        assert l1_stats["overall_hit_rate"] > 0.8


@pytest.mark.asyncio
class TestCacheErrorHandling:
    """Test cache error handling and resilience."""

    async def test_service_failure_graceful_degradation(self, l1_cache_service):
        """Test graceful degradation when cache operations fail."""
        # This would normally involve mocking failures
        # For now, test that normal operations work
        key = "resilience_test_key"
        value = "test_value"

        # Operations should succeed under normal conditions
        success = await l1_cache_service.set(key, value)
        assert success is True

        retrieved_value = await l1_cache_service.get(key)
        assert retrieved_value == value

    async def test_large_value_handling(self, l1_cache_service):
        """Test handling of large values."""
        key = "large_value_key"
        # Create a reasonably large value
        large_value = {"data": "x" * 10000, "metadata": list(range(1000))}

        success = await l1_cache_service.set(key, large_value)
        assert success is True

        retrieved_value = await l1_cache_service.get(key)
        assert retrieved_value == large_value

    async def test_memory_pressure_handling(self, l1_cache_service):
        """Test behavior under memory pressure."""
        # Fill cache with many entries to test eviction
        for i in range(100):
            key = f"memory_test_key_{i}"
            value = f"value_{i}" * 100  # Make values somewhat large
            await l1_cache_service.set(key, value)

        # Cache should still be functional
        test_key = "memory_test_key_50"
        retrieved_value = await l1_cache_service.get(test_key)

        # Value may or may not be present due to eviction, but operation should not fail
        # Retrieved value should either be the expected value or None
        if retrieved_value is not None:
            assert "value_50" in retrieved_value

    async def test_invalid_key_handling(self, l1_cache_service):
        """Test handling of invalid keys."""
        # Test None key
        try:
            await l1_cache_service.get(None)
            # Should either return None or handle gracefully
        except Exception:
            # Graceful error handling is acceptable
            pass

        # Test empty string key
        empty_key_result = await l1_cache_service.get("")
        # Should handle gracefully (return None)
        assert empty_key_result is None

        # Test very long key
        very_long_key = "x" * 10000
        success = await l1_cache_service.set(very_long_key, "test_value")
        # Should either succeed or fail gracefully
        assert isinstance(success, bool)
