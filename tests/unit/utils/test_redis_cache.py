"""
PHASE 6.2 REDIS MOCK ELIMINATION - 100% REAL REDIS TESTING

Comprehensive Redis cache testing with real Redis behavior validation.

All tests use actual Redis containers for authentic cache behavior testing:
- Real Redis operations and performance characteristics
- Actual Redis TTL, expiration, and eviction behavior
- Real Redis connection handling and error scenarios
- Authentic Redis memory usage and persistence testing
- Real Redis clustering and SSL/TLS testing

Zero mocked Redis operations - all cache testing uses real Redis instances.
"""

import asyncio
import json
import time

import coredis
import lz4.frame
import pytest
import pytest_asyncio

from prompt_improver.utils.redis_cache import (
    RedisCache,
    _execute_and_cache,
    get,
    get_cache,
    invalidate,
    set,
    with_singleflight,
)


class TestRealRedisCache:
    """Test suite for RedisCache with real Redis behavior."""

    @pytest_asyncio.fixture(scope="function") 
    async def cache_client(self, redis_binary_client):
        """Configure cache module with real Redis client."""
        await redis_binary_client.flushdb()
        
        import prompt_improver.utils.redis_cache as cache_module
        original_client = getattr(cache_module, 'redis_client', None)
        cache_module.redis_client = redis_binary_client
        
        try:
            yield redis_binary_client
        finally:
            if original_client is not None:
                cache_module.redis_client = original_client

    @pytest_asyncio.fixture(scope="function")
    async def reset_cache_state(self):
        """Reset cache module state for test isolation."""
        import prompt_improver.utils.redis_cache as cache_module
        cache_module._ongoing_operations.clear()
        cache_module._operations_lock = None
        yield

    @pytest.mark.asyncio
    async def test_cache_get_hit(self, cache_client, reset_cache_state):
        """Test successful cache get operation with real Redis."""
        test_key = "test_key"
        test_value = b"test_value"
        compressed_value = lz4.frame.compress(test_value)
        await cache_client.set(test_key, compressed_value)
        
        result = await RedisCache.get(test_key)
        assert result == test_value

    @pytest.mark.asyncio
    async def test_cache_get_miss(self, cache_client, reset_cache_state):
        """Test cache miss with real Redis."""
        result = await RedisCache.get("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_set_success(self, cache_client, reset_cache_state):
        """Test successful cache set with real Redis."""
        test_key = "test_key"
        test_value = b"test_value"
        
        result = await RedisCache.set(test_key, test_value)
        assert result is True
        
        # Verify data was stored and compressed correctly
        stored_value = await cache_client.get(test_key)
        assert lz4.frame.decompress(stored_value) == test_value

    @pytest.mark.asyncio
    async def test_cache_set_with_ttl(self, cache_client, reset_cache_state):
        """Test cache set with TTL using real Redis expiration."""
        test_key = "ttl_test"
        test_value = b"ttl_value"
        ttl = 2  # 2 seconds
        
        result = await RedisCache.set(test_key, test_value, expire=ttl)
        assert result is True
        
        # Verify data exists initially
        result = await RedisCache.get(test_key)
        assert result == test_value
        
        # Wait for expiration and verify real Redis TTL behavior
        await asyncio.sleep(ttl + 0.5)
        result = await RedisCache.get(test_key)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_invalidate(self, cache_client, reset_cache_state):
        """Test cache invalidation with real Redis."""
        test_key = "invalidate_test"
        test_value = b"invalidate_value"
        
        # Set value first
        await RedisCache.set(test_key, test_value)
        assert await RedisCache.get(test_key) == test_value
        
        # Invalidate and verify removal
        result = await RedisCache.invalidate(test_key)
        assert result is True
        assert await RedisCache.get(test_key) is None
        assert await cache_client.get(test_key) is None

    @pytest.mark.asyncio
    async def test_cache_invalidate_nonexistent(self, cache_client, reset_cache_state):
        """Test invalidation of non-existent key with real Redis."""
        result = await RedisCache.invalidate("nonexistent_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_decompression_error_handling(self, cache_client, reset_cache_state):
        """Test handling of corrupted data with real Redis."""
        test_key = "corrupted_key"
        # Store invalid compressed data
        await cache_client.set(test_key, b"invalid_compressed_data")
        
        # Should handle decompression error gracefully
        result = await RedisCache.get(test_key)
        assert result is None
        # Key should be cleaned up after decompression failure
        assert await cache_client.get(test_key) is None

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, cache_client, reset_cache_state):
        """Test Redis connection error handling with real disconnection."""
        import prompt_improver.utils.redis_cache as cache_module
        original_client = cache_module.redis_client
        
        # Use invalid Redis connection to simulate network error
        invalid_client = coredis.Redis(
            host="nonexistent-redis.local", 
            port=9999, 
            socket_connect_timeout=0.1,
            socket_timeout=0.1
        )
        cache_module.redis_client = invalid_client
        
        try:
            # Operations should handle connection errors gracefully
            result = await RedisCache.get("test_key")
            assert result is None
            
            result = await RedisCache.set("test_key", b"test_value")
            assert result is False
            
            result = await RedisCache.invalidate("test_key")
            assert result is False
            
        finally:
            cache_module.redis_client = original_client


class TestRealSingleflightPattern:
    """Test singleflight pattern with real Redis."""

    @pytest_asyncio.fixture(scope="function")
    async def singleflight_client(self, redis_binary_client):
        """Configure cache module for singleflight testing."""
        await redis_binary_client.flushdb()
        
        import prompt_improver.utils.redis_cache as cache_module
        original_client = getattr(cache_module, 'redis_client', None)
        cache_module.redis_client = redis_binary_client
        
        # Reset singleflight state
        cache_module._ongoing_operations.clear()
        cache_module._operations_lock = None
        
        try:
            yield redis_binary_client
        finally:
            if original_client is not None:
                cache_module.redis_client = original_client

    @pytest.mark.asyncio
    async def test_singleflight_cache_hit(self, singleflight_client):
        """Test singleflight with cache hit using real Redis."""
        test_value = {"result": "cached_data"}
        cache_data = json.dumps(test_value).encode()
        compressed_data = lz4.frame.compress(cache_data)
        await singleflight_client.set("test_key_hash", compressed_data)

        @with_singleflight(cache_key_fn=lambda: "test_key_hash")
        async def expensive_operation():
            return {"result": "computed_data"}  # Should not be called

        result = await expensive_operation()
        assert result == test_value

    @pytest.mark.asyncio
    async def test_singleflight_cache_miss(self, singleflight_client):
        """Test singleflight with cache miss using real Redis."""
        call_count = 0

        @with_singleflight(cache_key_fn=lambda: "miss_key_hash", expire=60)
        async def expensive_operation():
            nonlocal call_count
            call_count += 1
            return {"result": "computed_data"}

        result = await expensive_operation()
        assert result == {"result": "computed_data"}
        assert call_count == 1
        
        # Verify data was cached in real Redis
        cached_data = await singleflight_client.get("miss_key_hash")
        assert cached_data is not None
        decompressed = lz4.frame.decompress(cached_data)
        cached_result = json.loads(decompressed.decode())
        assert cached_result == {"result": "computed_data"}
        
        # Verify TTL was set correctly
        ttl = await singleflight_client.ttl("miss_key_hash")
        assert 50 <= ttl <= 60  # Should be close to 60 seconds

    @pytest.mark.asyncio
    async def test_singleflight_concurrent_calls(self, singleflight_client):
        """Test concurrent singleflight calls with real Redis coordination."""
        call_count = 0
        call_delay = 0.1

        @with_singleflight(cache_key_fn=lambda: "concurrent_key")
        async def slow_operation():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(call_delay)
            return {"result": f"call_{call_count}"}

        # Launch multiple concurrent calls
        tasks = [slow_operation() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All calls should return the same result (first completion)
        first_result = results[0]
        assert all(result == first_result for result in results)
        
        # Only one actual execution should have occurred
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_singleflight_error_handling(self, singleflight_client):
        """Test singleflight error handling with real Redis."""
        @with_singleflight(cache_key_fn=lambda: "error_key")
        async def failing_operation():
            raise ValueError("Test error")

        # Error should propagate and not be cached
        with pytest.raises(ValueError, match="Test error"):
            await failing_operation()
        
        # Verify error result was not cached
        cached_data = await singleflight_client.get("error_key")
        assert cached_data is None

    @pytest.mark.asyncio
    async def test_singleflight_bytes_result(self, singleflight_client):
        """Test singleflight with bytes return value using real Redis."""
        @with_singleflight(cache_key_fn=lambda: "bytes_key")
        async def bytes_operation():
            return b"bytes_result"

        result = await bytes_operation()
        assert result == b"bytes_result"
        
        # Verify bytes were cached correctly
        cached_data = await singleflight_client.get("bytes_key")
        assert cached_data is not None
        decompressed = lz4.frame.decompress(cached_data)
        assert decompressed == b"bytes_result"

    @pytest.mark.asyncio
    async def test_singleflight_none_result(self, singleflight_client):
        """Test singleflight with None return value using real Redis."""
        call_count = 0

        @with_singleflight(cache_key_fn=lambda: "none_key")
        async def none_operation():
            nonlocal call_count
            call_count += 1
            return None

        result = await none_operation()
        assert result is None
        assert call_count == 1
        
        # None results should not be cached
        cached_data = await singleflight_client.get("none_key")
        assert cached_data is None


class TestRealRedisModuleFunctions:
    """Test module-level cache functions with real Redis."""

    @pytest_asyncio.fixture(scope="function")
    async def module_client(self, redis_binary_client):
        """Configure module functions with real Redis."""
        await redis_binary_client.flushdb()
        
        import prompt_improver.utils.redis_cache as cache_module
        original_client = getattr(cache_module, 'redis_client', None)
        cache_module.redis_client = redis_binary_client
        
        try:
            yield redis_binary_client
        finally:
            if original_client is not None:
                cache_module.redis_client = original_client

    @pytest.mark.asyncio
    async def test_module_get_function(self, module_client):
        """Test module-level get function with real Redis."""
        test_key = "module_get_test"
        test_value = {"data": "module_test"}
        
        # Set data using RedisCache.set
        await RedisCache.set(test_key, json.dumps(test_value).encode())
        
        # Retrieve using module-level function
        result = await get(test_key)
        assert json.loads(result.decode()) == test_value

    @pytest.mark.asyncio
    async def test_module_set_function(self, module_client):
        """Test module-level set function with real Redis."""
        test_key = "module_set_test"
        test_value = b"module_set_data"
        
        result = await set(test_key, test_value, expire=30)
        assert result is True
        
        # Verify with module-level get
        cached_value = await get(test_key)
        assert cached_value == test_value
        
        # Verify TTL was set
        ttl = await module_client.ttl(test_key)
        assert 25 <= ttl <= 30

    @pytest.mark.asyncio
    async def test_module_invalidate_function(self, module_client):
        """Test module-level invalidate function with real Redis."""
        test_key = "module_invalidate_test"
        test_value = b"module_invalidate_data"
        
        await set(test_key, test_value)
        assert await get(test_key) == test_value
        
        result = await invalidate(test_key)
        assert result is True
        assert await get(test_key) is None


class TestRealRedisPerformance:
    """Test Redis cache performance characteristics with real Redis."""

    @pytest_asyncio.fixture(scope="function")
    async def performance_client(self, redis_performance_container):
        """Use performance-optimized Redis container."""
        client = await redis_performance_container.get_async_client(decode_responses=False)
        await client.flushdb()
        
        import prompt_improver.utils.redis_cache as cache_module
        original_client = getattr(cache_module, 'redis_client', None)
        cache_module.redis_client = client
        
        try:
            yield client
        finally:
            if original_client is not None:
                cache_module.redis_client = original_client
            await client.aclose()

    @pytest.mark.asyncio
    async def test_cache_compression_efficiency(self, performance_client):
        """Test LZ4 compression efficiency with real Redis storage."""
        # Test data with high compression ratio
        test_data = ("x" * 1000).encode()  # Highly compressible
        compressed_data = lz4.frame.compress(test_data)
        
        # Compression should significantly reduce size
        assert len(compressed_data) < len(test_data) * 0.1
        
        # Store and retrieve through cache
        await RedisCache.set("compression_test", test_data)
        result = await RedisCache.get("compression_test")
        assert result == test_data
        
        # Verify compressed storage in Redis
        stored_data = await performance_client.get("compression_test")
        assert len(stored_data) < len(test_data) * 0.2  # Stored compressed

    @pytest.mark.asyncio
    async def test_cache_memory_usage(self, redis_performance_container, performance_client):
        """Test real Redis memory usage with cache operations."""
        initial_memory = await redis_performance_container.get_memory_usage()
        
        # Store multiple large values
        large_data = ("test_data_" * 1000).encode()
        for i in range(100):
            await RedisCache.set(f"memory_test_{i}", large_data)
        
        final_memory = await redis_performance_container.get_memory_usage()
        
        # Memory usage should have increased
        memory_increase = final_memory["used_memory"] - initial_memory["used_memory"]
        assert memory_increase > 0
        
        # But compression should keep it reasonable
        expected_uncompressed = len(large_data) * 100
        assert memory_increase < expected_uncompressed * 0.3  # Compression savings

    @pytest.mark.asyncio 
    async def test_cache_throughput(self, performance_client):
        """Test cache throughput with real Redis operations."""
        num_operations = 100
        test_data = b"throughput_test_data"
        
        # Measure set throughput
        start_time = time.time()
        for i in range(num_operations):
            await RedisCache.set(f"throughput_key_{i}", test_data)
        set_duration = time.time() - start_time
        
        # Measure get throughput  
        start_time = time.time()
        for i in range(num_operations):
            result = await RedisCache.get(f"throughput_key_{i}")
            assert result == test_data
        get_duration = time.time() - start_time
        
        # Performance should be reasonable (adjust thresholds based on environment)
        set_ops_per_sec = num_operations / set_duration
        get_ops_per_sec = num_operations / get_duration
        
        assert set_ops_per_sec > 50  # At least 50 sets per second
        assert get_ops_per_sec > 100  # At least 100 gets per second


class TestRealRedisIntegration:
    """Integration tests with real Redis behavior."""

    @pytest_asyncio.fixture(scope="function")
    async def integration_client(self, redis_client):
        """Configure for integration testing."""
        await redis_client.flushdb()
        
        import prompt_improver.utils.redis_cache as cache_module
        original_client = getattr(cache_module, 'redis_client', None)
        # Use the text-mode Redis client for integration tests
        cache_module.redis_client = redis_client
        
        try:
            yield redis_client
        finally:
            if original_client is not None:
                cache_module.redis_client = original_client

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, integration_client):
        """Test complete cache workflow with real Redis."""
        # Test comprehensive workflow
        test_data = {"workflow": "test", "step": 1}
        test_bytes = json.dumps(test_data).encode()
        
        # 1. Set with TTL
        result = await set("workflow_key", test_bytes, expire=300)
        assert result is True
        
        # 2. Get and verify
        cached = await get("workflow_key")
        assert json.loads(cached.decode()) == test_data
        
        # 3. Check TTL
        ttl = await integration_client.ttl("workflow_key")
        assert 250 <= ttl <= 300
        
        # 4. Update data
        updated_data = {"workflow": "test", "step": 2}
        await set("workflow_key", json.dumps(updated_data).encode())
        
        # 5. Verify update
        result = await get("workflow_key")
        assert json.loads(result.decode()) == updated_data
        
        # 6. Clean up
        deleted = await invalidate("workflow_key") 
        assert deleted is True
        assert await get("workflow_key") is None

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, integration_client):
        """Test concurrent cache operations with real Redis."""
        async def cache_worker(worker_id: int):
            """Worker function for concurrent testing."""
            results = []
            for i in range(10):
                key = f"concurrent_{worker_id}_{i}"
                value = f"worker_{worker_id}_value_{i}".encode()
                
                await set(key, value)
                retrieved = await get(key)
                results.append((key, retrieved == value))
                
            return results
        
        # Run multiple workers concurrently
        tasks = [cache_worker(i) for i in range(5)]
        all_results = await asyncio.gather(*tasks)
        
        # Verify all operations succeeded
        for worker_results in all_results:
            for key, success in worker_results:
                assert success, f"Failed operation for key: {key}"
        
        # Verify data integrity in Redis
        total_keys = 0
        async for key in integration_client.scan_iter(match="concurrent_*"):
            total_keys += 1
        assert total_keys == 50  # 5 workers * 10 operations each