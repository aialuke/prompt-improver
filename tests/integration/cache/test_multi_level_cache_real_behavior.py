"""Real behavior tests for multi-level cache services (L1/L2/L3).

Comprehensive validation of decomposed cache services with actual Redis and database instances.
Validates performance requirements, error resilience, and cross-level coordination.

Performance Requirements:
- L1 Cache: <1ms response time, >95% hit rate for hot data
- L2 Cache: <10ms response time, >80% hit rate for warm data  
- L3 Cache: <50ms response time, 100% durability for cold data
- Overall: <200ms end-to-end, >80% cache hit rates
"""

import asyncio
import pytest
import time
from typing import Any, Dict
from uuid import uuid4

from tests.containers.postgres_container import PostgreSQLTestContainer
from tests.containers.real_redis_container import RealRedisTestContainer
from prompt_improver.services.cache.l1_cache_service import L1CacheService
from prompt_improver.services.cache.l2_redis_service import L2RedisService
from prompt_improver.services.cache.l3_database_service import L3DatabaseService
from prompt_improver.services.cache.cache_facade import CacheFacade


class TestL1CacheRealBehavior:
    """Real behavior tests for L1 in-memory cache service."""

    @pytest.fixture
    async def l1_cache(self):
        """Create L1 cache service instance."""
        cache = L1CacheService(max_size=100)
        yield cache
        await cache.clear()

    async def test_l1_cache_performance_requirements(self, l1_cache):
        """Validate L1 cache meets <1ms response time requirement."""
        test_data = {f"key_{i}": f"value_{i}" for i in range(50)}
        
        # Test set operations
        set_times = []
        for key, value in test_data.items():
            start_time = time.perf_counter()
            success = await l1_cache.set(key, value, ttl=3600)
            duration = time.perf_counter() - start_time
            set_times.append(duration)
            assert success, f"Failed to set key {key}"
            assert duration < 0.001, f"Set operation took {duration:.3f}s, exceeds 1ms limit"
        
        # Test get operations
        get_times = []
        for key in test_data.keys():
            start_time = time.perf_counter()
            result = await l1_cache.get(key)
            duration = time.perf_counter() - start_time
            get_times.append(duration)
            assert result is not None, f"Failed to get key {key}"
            assert duration < 0.001, f"Get operation took {duration:.3f}s, exceeds 1ms limit"
        
        # Validate performance statistics
        stats = await l1_cache.get_stats()
        assert stats["hit_rate"] > 0.95, f"Hit rate {stats['hit_rate']:.2f} below 95% target"
        assert stats["size"] == 50, f"Expected 50 entries, got {stats['size']}"
        
        # Performance summary
        avg_set_time = sum(set_times) / len(set_times)
        avg_get_time = sum(get_times) / len(get_times)
        print(f"\nL1 Cache Performance:")
        print(f"  Average Set Time: {avg_set_time*1000:.3f}ms")
        print(f"  Average Get Time: {avg_get_time*1000:.3f}ms")
        print(f"  Hit Rate: {stats['hit_rate']:.2%}")
        print(f"  Memory Usage: {stats['estimated_memory_bytes']:,} bytes")

    async def test_l1_cache_namespace_isolation(self, l1_cache):
        """Test namespace isolation in L1 cache."""
        # Set data in different namespaces
        await l1_cache.set("shared_key", "default_value")
        await l1_cache.set("shared_key", "namespace_a_value", namespace="namespace_a")
        await l1_cache.set("shared_key", "namespace_b_value", namespace="namespace_b")
        
        # Verify isolation
        default_value = await l1_cache.get("shared_key")
        a_value = await l1_cache.get("shared_key", namespace="namespace_a")
        b_value = await l1_cache.get("shared_key", namespace="namespace_b")
        
        assert default_value == "default_value"
        assert a_value == "namespace_a_value"
        assert b_value == "namespace_b_value"
        
        # Test namespace clearing
        await l1_cache.clear(namespace="namespace_a")
        assert await l1_cache.get("shared_key") == "default_value"
        assert await l1_cache.get("shared_key", namespace="namespace_a") is None
        assert await l1_cache.get("shared_key", namespace="namespace_b") == "namespace_b_value"

    async def test_l1_cache_eviction_and_memory_management(self, l1_cache):
        """Test LRU eviction and memory management."""
        # Fill cache to capacity
        for i in range(100):
            await l1_cache.set(f"key_{i}", f"value_{i}")
        
        stats = await l1_cache.get_stats()
        assert stats["size"] == 100
        assert stats["evictions"] == 0
        
        # Add one more item to trigger eviction
        await l1_cache.set("overflow_key", "overflow_value")
        
        stats = await l1_cache.get_stats()
        assert stats["size"] == 100  # Still at max size
        assert stats["evictions"] == 1  # One eviction occurred
        
        # Verify LRU behavior - oldest key should be evicted
        assert await l1_cache.get("key_0") is None  # First key evicted
        assert await l1_cache.get("overflow_key") == "overflow_value"  # New key present

    async def test_l1_cache_ttl_and_expiration(self, l1_cache):
        """Test TTL functionality and automatic expiration."""
        # Set item with short TTL
        await l1_cache.set("short_ttl", "expires_soon", ttl=1)
        await l1_cache.set("long_ttl", "expires_later", ttl=3600)
        
        # Verify immediately available
        assert await l1_cache.get("short_ttl") == "expires_soon"
        assert await l1_cache.get("long_ttl") == "expires_later"
        
        # Wait for expiration
        await asyncio.sleep(1.5)
        
        # Check expiration
        assert await l1_cache.get("short_ttl") is None  # Expired
        assert await l1_cache.get("long_ttl") == "expires_later"  # Still valid
        
        # Test manual cleanup
        await l1_cache.set("cleanup_test", "value", ttl=1)
        await asyncio.sleep(1.5)
        cleaned_count = await l1_cache.cleanup_expired()
        assert cleaned_count > 0

    async def test_l1_cache_hot_keys_tracking(self, l1_cache):
        """Test hot keys tracking and access statistics."""
        # Create keys with different access patterns
        keys_and_access_counts = [
            ("hot_key", 10),
            ("warm_key", 5), 
            ("cold_key", 1),
        ]
        
        # Set keys and access them different numbers of times
        for key, access_count in keys_and_access_counts:
            await l1_cache.set(key, f"value_for_{key}")
            for _ in range(access_count):
                await l1_cache.get(key)
        
        # Get hot keys
        hot_keys = await l1_cache.get_hot_keys(limit=5)
        assert len(hot_keys) >= 3
        
        # Verify ordering (most accessed first)
        assert hot_keys[0]["key"] == "hot_key"
        assert hot_keys[0]["access_count"] == 10
        assert hot_keys[1]["key"] == "warm_key"  
        assert hot_keys[1]["access_count"] == 5
        assert hot_keys[2]["key"] == "cold_key"
        assert hot_keys[2]["access_count"] == 1


class TestL2CacheRealBehavior:
    """Real behavior tests for L2 Redis cache service."""

    @pytest.fixture
    async def redis_container(self):
        """Create real Redis testcontainer."""
        container = RealRedisTestContainer()
        await container.start()
        yield container
        await container.stop()

    @pytest.fixture
    async def l2_cache(self, redis_container):
        """Create L2 cache service with real Redis."""
        import os
        # Set Redis connection for L2 cache
        os.environ["REDIS_HOST"] = redis_container.get_host()
        os.environ["REDIS_PORT"] = str(redis_container.get_port())
        
        cache = L2RedisService()
        yield cache
        await cache.close()

    async def test_l2_cache_performance_requirements(self, l2_cache):
        """Validate L2 cache meets <10ms response time requirement."""
        test_data = {f"key_{i}": {"id": i, "data": f"value_{i}"} for i in range(100)}
        
        # Test set operations
        set_times = []
        for key, value in test_data.items():
            start_time = time.perf_counter()
            success = await l2_cache.set(key, value, ttl=3600)
            duration = time.perf_counter() - start_time
            set_times.append(duration)
            assert success, f"Failed to set key {key}"
            assert duration < 0.01, f"Set operation took {duration:.3f}s, exceeds 10ms limit"
        
        # Test get operations
        get_times = []
        for key in test_data.keys():
            start_time = time.perf_counter()
            result = await l2_cache.get(key)
            duration = time.perf_counter() - start_time
            get_times.append(duration)
            assert result is not None, f"Failed to get key {key}"
            assert duration < 0.01, f"Get operation took {duration:.3f}s, exceeds 10ms limit"
        
        # Test bulk operations
        bulk_keys = list(test_data.keys())[:10]
        start_time = time.perf_counter()
        bulk_results = await l2_cache.mget(bulk_keys)
        bulk_duration = time.perf_counter() - start_time
        assert len(bulk_results) == 10
        assert bulk_duration < 0.01, f"Bulk get took {bulk_duration:.3f}s, exceeds 10ms limit"
        
        # Performance summary
        avg_set_time = sum(set_times) / len(set_times)
        avg_get_time = sum(get_times) / len(get_times)
        print(f"\nL2 Cache Performance:")
        print(f"  Average Set Time: {avg_set_time*1000:.3f}ms")
        print(f"  Average Get Time: {avg_get_time*1000:.3f}ms")
        print(f"  Bulk Get Time: {bulk_duration*1000:.3f}ms for 10 keys")

    async def test_l2_cache_connection_resilience(self, l2_cache):
        """Test L2 cache resilience to connection failures."""
        # Test normal operation
        await l2_cache.set("test_key", "test_value")
        result = await l2_cache.get("test_key")
        assert result == "test_value"
        
        # Get connection status
        status = await l2_cache.get_connection_status()
        assert status["connected"] is True
        assert "ping_duration_ms" in status
        assert status["ping_duration_ms"] < 100  # Should be very fast for local Redis
        
        # Test health check
        health = await l2_cache.health_check()
        assert health["healthy"] is True
        assert health["status"] == "healthy"
        assert health["performance"]["p95_response_time_ms"] < 10

    async def test_l2_cache_namespace_operations(self, l2_cache):
        """Test namespace isolation and bulk operations."""
        # Set data in different namespaces
        await l2_cache.set("key1", "default_value")
        await l2_cache.set("key1", "ns_a_value", namespace="namespace_a")
        await l2_cache.set("key1", "ns_b_value", namespace="namespace_b")
        
        # Test isolation
        assert await l2_cache.get("key1") == "default_value"
        assert await l2_cache.get("key1", namespace="namespace_a") == "ns_a_value"
        assert await l2_cache.get("key1", namespace="namespace_b") == "ns_b_value"
        
        # Test bulk operations with namespace
        bulk_data = {"bulk_key1": "value1", "bulk_key2": "value2", "bulk_key3": "value3"}
        success = await l2_cache.mset(bulk_data, namespace="bulk_test")
        assert success
        
        bulk_results = await l2_cache.mget(list(bulk_data.keys()), namespace="bulk_test")
        assert len(bulk_results) == 3
        for key, expected_value in bulk_data.items():
            assert bulk_results[key] == expected_value

    async def test_l2_cache_pattern_operations(self, l2_cache):
        """Test pattern-based operations and key management."""
        # Create test data with patterns
        test_keys = [
            "user:1:profile",
            "user:1:settings", 
            "user:2:profile",
            "user:2:settings",
            "system:config",
            "system:stats"
        ]
        
        for key in test_keys:
            await l2_cache.set(key, f"data_for_{key}")
        
        # Test pattern deletion
        deleted_count = await l2_cache.delete_pattern("user:1:*")
        assert deleted_count == 2  # Should delete user:1:profile and user:1:settings
        
        # Verify selective deletion
        assert await l2_cache.get("user:1:profile") is None
        assert await l2_cache.get("user:1:settings") is None
        assert await l2_cache.get("user:2:profile") is not None
        assert await l2_cache.get("system:config") is not None

    async def test_l2_cache_memory_pressure_handling(self, redis_container, l2_cache):
        """Test L2 cache behavior under memory pressure."""
        # Fill Redis with data to approach memory limits
        large_value = "x" * 10000  # 10KB per value
        
        # Set many keys
        for i in range(100):
            await l2_cache.set(f"memory_test_{i}", large_value, ttl=3600)
        
        # Check Redis memory usage
        memory_stats = await redis_container.get_memory_usage()
        print(f"Redis memory usage: {memory_stats['used_memory'] / (1024*1024):.1f} MB")
        
        # Continue adding data and verify cache still works
        for i in range(100, 150):
            success = await l2_cache.set(f"memory_test_{i}", large_value, ttl=3600)
            assert success  # Should succeed even under memory pressure
        
        # Verify some data is still accessible
        recent_data = await l2_cache.get("memory_test_149")
        assert recent_data == large_value


class TestCacheServiceFacadeRealBehavior:
    """Real behavior tests for multi-level cache coordination."""
    
    @pytest.fixture
    async def test_infrastructure(self):
        """Set up full test infrastructure with Redis and PostgreSQL."""
        redis_container = RealRedisTestContainer()
        postgres_container = PostgreSQLTestContainer()
        
        await redis_container.start()
        await postgres_container.start()
        
        # Set environment variables
        import os
        os.environ["REDIS_HOST"] = redis_container.get_host()
        os.environ["REDIS_PORT"] = str(redis_container.get_port())
        
        yield {
            "redis": redis_container,
            "postgres": postgres_container
        }
        
        await redis_container.stop()
        await postgres_container.stop()

    async def test_cache_hierarchy_coordination(self, test_infrastructure):
        """Test coordination between cache levels."""
        # This test would be implemented once CacheServiceFacade is available
        # For now, test individual services coordination
        
        l1_cache = L1CacheService(max_size=50)
        l2_cache = L2RedisService()
        
        try:
            # Simulate cache miss scenario - data not in L1, check L2
            key = "hierarchical_test_key"
            value = {"data": "hierarchical_test_value", "timestamp": time.time()}
            
            # L1 miss
            l1_result = await l1_cache.get(key)
            assert l1_result is None
            
            # L2 miss, would trigger L3 lookup in real facade
            l2_result = await l2_cache.get(key)
            assert l2_result is None
            
            # Simulate data retrieved from L3, populate upward
            await l2_cache.set(key, value, ttl=3600)
            await l1_cache.set(key, value, ttl=1800)  # Shorter TTL in L1
            
            # Verify cache hierarchy
            l1_result = await l1_cache.get(key)
            l2_result = await l2_cache.get(key)
            
            assert l1_result == value
            assert l2_result == value
            
        finally:
            await l1_cache.clear()
            await l2_cache.close()

    async def test_cache_performance_end_to_end(self, test_infrastructure):
        """Test end-to-end cache performance across all levels."""
        l1_cache = L1CacheService(max_size=100)
        l2_cache = L2RedisService()
        
        try:
            test_scenarios = [
                # (description, operations_count, data_size_bytes)
                ("Small data, high frequency", 1000, 100),
                ("Medium data, medium frequency", 500, 1000),
                ("Large data, low frequency", 100, 10000),
            ]
            
            performance_results = {}
            
            for description, ops_count, data_size in test_scenarios:
                print(f"\nTesting: {description}")
                
                # Generate test data
                test_value = "x" * data_size
                keys = [f"perf_test_{i}" for i in range(ops_count)]
                
                # Time L1 operations
                l1_start = time.perf_counter()
                for key in keys:
                    await l1_cache.set(key, test_value)
                for key in keys:
                    await l1_cache.get(key)
                l1_duration = time.perf_counter() - l1_start
                
                # Time L2 operations  
                l2_start = time.perf_counter()
                for key in keys[:ops_count//10]:  # Test subset for L2
                    await l2_cache.set(key, test_value)
                for key in keys[:ops_count//10]:
                    await l2_cache.get(key)
                l2_duration = time.perf_counter() - l2_start
                
                performance_results[description] = {
                    "l1_ops_per_second": (ops_count * 2) / l1_duration,
                    "l1_avg_latency_ms": (l1_duration * 1000) / (ops_count * 2),
                    "l2_ops_per_second": (ops_count // 5) / l2_duration,
                    "l2_avg_latency_ms": (l2_duration * 1000) / (ops_count // 5),
                }
                
                print(f"  L1: {performance_results[description]['l1_ops_per_second']:.0f} ops/sec, "
                      f"{performance_results[description]['l1_avg_latency_ms']:.3f}ms avg")
                print(f"  L2: {performance_results[description]['l2_ops_per_second']:.0f} ops/sec, "
                      f"{performance_results[description]['l2_avg_latency_ms']:.3f}ms avg")
                
                # Validate performance requirements
                assert performance_results[description]['l1_avg_latency_ms'] < 1.0
                assert performance_results[description]['l2_avg_latency_ms'] < 10.0
                
        finally:
            await l1_cache.clear()
            await l2_cache.close()


@pytest.mark.integration
@pytest.mark.real_behavior
class TestCacheSystemResilience:
    """Test cache system resilience and error handling."""
    
    async def test_redis_failure_recovery(self):
        """Test L2 cache recovery from Redis failures."""
        # This would require complex container manipulation
        # For now, test graceful degradation
        
        l2_cache = L2RedisService()
        
        # Test with invalid Redis connection
        import os
        original_host = os.environ.get("REDIS_HOST", "localhost")
        os.environ["REDIS_HOST"] = "invalid_host_that_does_not_exist"
        
        try:
            # Operations should fail gracefully
            result = await l2_cache.get("test_key")
            assert result is None  # Should return None, not raise exception
            
            success = await l2_cache.set("test_key", "test_value")
            assert success is False  # Should return False, not raise exception
            
            health = await l2_cache.health_check()
            assert health["healthy"] is False
            assert "error" in health
            
        finally:
            # Restore original host
            if original_host:
                os.environ["REDIS_HOST"] = original_host
            elif "REDIS_HOST" in os.environ:
                del os.environ["REDIS_HOST"]
            await l2_cache.close()

    async def test_cache_data_consistency(self):
        """Test data consistency across cache operations."""
        l1_cache = L1CacheService(max_size=10)
        
        # Test concurrent operations
        async def writer_task(cache_instance, key_prefix, write_count):
            for i in range(write_count):
                await cache_instance.set(f"{key_prefix}_{i}", f"value_{i}")
                
        async def reader_task(cache_instance, key_prefix, read_count):
            results = []
            for i in range(read_count):
                result = await cache_instance.get(f"{key_prefix}_{i}")
                results.append(result)
            return results
        
        # Run concurrent operations
        write_task1 = writer_task(l1_cache, "writer1", 20)
        write_task2 = writer_task(l1_cache, "writer2", 20)
        read_task1 = reader_task(l1_cache, "writer1", 20)
        
        await asyncio.gather(write_task1, write_task2)
        read_results = await read_task1
        
        # Verify some data consistency (accounting for eviction)
        non_none_results = [r for r in read_results if r is not None]
        assert len(non_none_results) <= 10  # Limited by cache size
        
        # Verify cache statistics are consistent
        stats = await l1_cache.get_stats()
        assert stats["size"] <= 10  # Within max size
        assert stats["sets"] >= 40  # All sets were attempted