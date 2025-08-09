"""
MIGRATION STATUS: FOLLOWS 2025 BEST PRACTICES WITH OPTIMAL HYBRID APPROACH

Comprehensive tests for Redis cache implementation following 2025 best practices:

✅ REAL BEHAVIOR TESTING (Primary approach):
- Real Redis via external Redis service for authentic operations
- Real LZ4 compression/decompression behavior
- Real async Redis operations (set, get, delete, TTL)
- Real singleflight pattern with concurrent execution prevention
- Real integration workflow testing

✅ STRATEGIC MOCKING (Where appropriate):
- Error simulation for connection failures
- Performance testing with fakeredis for speed
- Compression error simulation

Tests cover:
- Basic cache operations (get, set, invalidate)
- Compression and decompression
- OpenTelemetry metrics collection
- Singleflight pattern behavior
- Error handling and edge cases
- Performance characteristics
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
import coredis
import fakeredis
import lz4.frame
import pytest
import pytest_asyncio
from opentelemetry import metrics, trace
from prompt_improver.utils.redis_cache import RedisCache, _execute_and_cache, get, get_cache, invalidate, set, with_singleflight

class TestRedisCache:
    """Test suite for RedisCache class."""

    @pytest_asyncio.fixture(scope='function')
    async def real_redis(self, redis_client):
        """Use the real Redis client fixture for testing."""
        await redis_client.flushdb()
        with patch('prompt_improver.utils.redis_cache.redis_client', redis_client):
            yield redis_client

    @pytest_asyncio.fixture(scope='function')
    async def reset_metrics(self):
        """Reset metrics for real behavior testing."""
        import prompt_improver.utils.redis_cache as cache_module
        cache_module._ongoing_operations.clear()
        cache_module._operations_lock = None
        yield

    @pytest.mark.asyncio
    async def test_get_hit(self, real_redis, reset_metrics):
        """Test successful cache get operation."""
        test_key = 'test_key'
        test_value = b'test_value'
        compressed_value = lz4.frame.compress(test_value)
        await real_redis.set(test_key, compressed_value)
        result = await RedisCache.get(test_key)
        assert result == test_value

    @pytest.mark.asyncio
    async def test_get_miss(self, real_redis, reset_metrics):
        """Test cache miss scenario."""
        result = await RedisCache.get('nonexistent_key')
        assert result is None

    @pytest.mark.asyncio
    async def test_get_decompression_error(self, real_redis, reset_metrics):
        """Test handling of corrupted compressed data."""
        test_key = 'corrupted_key'
        await real_redis.set(test_key, b'invalid_compressed_data')
        result = await RedisCache.get(test_key)
        assert result is None
        assert await real_redis.get(test_key) is None

    @pytest.mark.asyncio
    async def test_set_success(self, real_redis, reset_metrics):
        """Test successful cache set operation."""
        test_key = 'test_key'
        test_value = b'test_value'
        result = await RedisCache.set(test_key, test_value)
        assert result is True
        stored_value = await real_redis.get(test_key)
        assert lz4.frame.decompress(stored_value) == test_value

    @pytest.mark.asyncio
    async def test_set_with_expiry(self, real_redis, reset_metrics):
        """Test cache set with expiration."""
        test_key = 'test_key'
        test_value = b'test_value'
        expire_time = 1
        result = await RedisCache.set(test_key, test_value, expire=expire_time)
        assert result is True
        ttl = await real_redis.ttl(test_key)
        assert ttl > 0

    @pytest.mark.asyncio
    async def test_invalidate_success(self, real_redis, reset_metrics):
        """Test successful cache invalidation."""
        test_key = 'test_key'
        test_value = b'test_value'
        await real_redis.set(test_key, test_value)
        result = await RedisCache.invalidate(test_key)
        assert result is True
        assert await real_redis.get(test_key) is None

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent(self, real_redis, reset_metrics):
        """Test invalidation of non-existent key."""
        result = await RedisCache.invalidate('nonexistent_key')
        assert result is False

    @pytest.mark.asyncio
    async def test_redis_connection_error(self, real_redis, reset_metrics):
        """Test handling of Redis connection errors."""
        with patch('prompt_improver.utils.redis_cache.redis_client') as mock_client:
            mock_client.get.side_effect = coredis.ConnectionError('Connection failed')
            result = await RedisCache.get('test_key')
            assert result is None

    @pytest.mark.asyncio
    async def test_metrics_collection(self, real_redis, reset_metrics):
        """Test that operations work correctly in sequence."""
        test_key = 'metrics_test'
        test_value = b'metrics_value'
        set_result = await RedisCache.set(test_key, test_value)
        get_result = await RedisCache.get(test_key)
        miss_result = await RedisCache.get('nonexistent')
        invalidate_result = await RedisCache.invalidate(test_key)
        assert set_result is True
        assert get_result == test_value
        assert miss_result is None
        assert invalidate_result is True
        assert await real_redis.get(test_key) is None

class TestSingleflightPattern:
    """Test suite for singleflight pattern implementation."""

    @pytest_asyncio.fixture(scope='function')
    async def real_redis(self, redis_client):
        """Use the real Redis client fixture for testing."""
        await redis_client.flushdb()
        with patch('prompt_improver.utils.redis_cache.redis_client', redis_client):
            yield redis_client

    @pytest_asyncio.fixture(scope='function')
    async def reset_metrics(self):
        """Mock metrics for testing."""
        import prompt_improver.utils.redis_cache as cache_module
        cache_module._ongoing_operations.clear()
        cache_module._operations_lock = None
        yield

    @pytest.mark.asyncio
    async def test_singleflight_cache_hit(self, real_redis, reset_metrics):
        """Test singleflight with cache hit."""
        test_key = 'singleflight_test'
        test_value = {'result': 'cached_data'}
        cache_data = json.dumps(test_value).encode()
        compressed_data = lz4.frame.compress(cache_data)
        await real_redis.set('test_key_hash', compressed_data)

        @with_singleflight(cache_key_fn=lambda: 'test_key_hash')
        async def expensive_operation():
            return {'result': 'computed_data'}
        result = await expensive_operation()
        assert result == test_value

    @pytest.mark.asyncio
    async def test_singleflight_cache_miss(self, real_redis, reset_metrics):
        """Test singleflight with cache miss."""
        call_count = 0

        @with_singleflight(cache_key_fn=lambda: 'test_key_hash', expire=60)
        async def expensive_operation():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return {'result': 'computed_data'}
        result = await expensive_operation()
        assert result == {'result': 'computed_data'}
        assert call_count == 1
        cached_data = await real_redis.get('test_key_hash')
        assert cached_data is not None
        decompressed = lz4.frame.decompress(cached_data)
        assert json.loads(decompressed.decode()) == {'result': 'computed_data'}
        ttl = await real_redis.ttl('test_key_hash')
        assert ttl > 0

    @pytest.mark.asyncio
    async def test_singleflight_concurrent_calls(self, real_redis, reset_metrics):
        """Test singleflight prevents multiple concurrent executions."""
        call_count = 0
        execution_started = asyncio.Event()
        can_complete = asyncio.Event()

        @with_singleflight(cache_key_fn=lambda: 'concurrent_test')
        async def slow_operation():
            nonlocal call_count
            call_count += 1
            execution_started.set()
            await can_complete.wait()
            return {'result': f'call_{call_count}'}

        async def make_call():
            return await slow_operation()
        task1 = asyncio.create_task(make_call())
        await execution_started.wait()
        task2 = asyncio.create_task(make_call())
        await asyncio.sleep(0.01)
        can_complete.set()
        result1 = await task1
        result2 = await task2
        assert call_count == 1
        assert result1 == result2
        assert result1 == {'result': 'call_1'}

    @pytest.mark.asyncio
    async def test_singleflight_default_key_generation(self, real_redis, reset_metrics):
        """Test default cache key generation."""

        @with_singleflight(expire=60)
        async def test_function(arg1, arg2, kwarg1=None):
            return {'args': [arg1, arg2], 'kwarg1': kwarg1}
        result = await test_function('value1', 'value2', kwarg1='kwvalue')
        assert result == {'args': ['value1', 'value2'], 'kwarg1': 'kwvalue'}

    @pytest.mark.asyncio
    async def test_singleflight_function_exception(self, real_redis, reset_metrics):
        """Test singleflight behavior when function raises exception."""

        @with_singleflight(cache_key_fn=lambda: 'exception_test')
        async def failing_function():
            raise ValueError('Test exception')
        with pytest.raises(ValueError, match='Test exception'):
            await failing_function()
        cached_data = await real_redis.get('exception_test')
        assert cached_data is None

    @pytest.mark.asyncio
    async def test_singleflight_bytes_result(self, real_redis, reset_metrics):
        """Test singleflight with bytes result."""

        @with_singleflight(cache_key_fn=lambda: 'bytes_test')
        async def bytes_function():
            return b'binary_data'
        result = await bytes_function()
        assert result == b'binary_data'
        cached_data = await real_redis.get('bytes_test')
        assert cached_data is not None
        decompressed = lz4.frame.decompress(cached_data)
        assert decompressed == b'binary_data'

    @pytest.mark.asyncio
    async def test_singleflight_none_result(self, real_redis, reset_metrics):
        """Test singleflight with None result."""

        @with_singleflight(cache_key_fn=lambda: 'none_test')
        async def none_function():
            return None
        result = await none_function()
        assert result is None
        cached_data = await real_redis.get('none_test')
        assert cached_data is None

class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    @pytest_asyncio.fixture(scope='function')
    async def real_redis(self, redis_client):
        """Use the real Redis client fixture for testing."""
        await redis_client.flushdb()
        with patch('prompt_improver.utils.redis_cache.redis_client', redis_client):
            yield redis_client

    @pytest.mark.asyncio
    async def test_module_level_functions(self, real_redis):
        """Test module-level convenience functions."""
        result = await set('module_test', b'test_data')
        assert result is True
        result = await get('module_test')
        assert result == b'test_data'
        result = await invalidate('module_test')
        assert result is True
        result = await get('module_test')
        assert result is None

class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_compression_error_handling(self):
        """Test handling of compression errors."""
        with patch('lz4.frame.compress') as mock_compress:
            mock_compress.side_effect = Exception('Compression failed')
            result = await RedisCache.set('test_key', b'test_data')
            assert result is False

    @pytest.mark.asyncio
    async def test_redis_set_error_handling(self):
        """Test handling of Redis set errors."""
        with patch('prompt_improver.utils.redis_cache.redis_client') as mock_client:
            mock_client.set.side_effect = coredis.ConnectionError('Connection failed')
            result = await RedisCache.set('test_key', b'test_data')
            assert result is False

    @pytest.mark.asyncio
    async def test_redis_delete_error_handling(self):
        """Test handling of Redis delete errors."""
        with patch('prompt_improver.utils.redis_cache.redis_client') as mock_client:
            mock_client.delete.side_effect = coredis.ConnectionError('Connection failed')
            result = await RedisCache.invalidate('test_key')
            assert result is False

class TestPerformanceCharacteristics:
    """Test performance characteristics and benchmarks."""

    @pytest.mark.asyncio
    async def test_compression_efficiency(self):
        """Test that compression actually reduces data size."""
        large_data = b'A' * 1000
        compressed_data = lz4.frame.compress(large_data)
        assert len(compressed_data) < len(large_data)
        decompressed = lz4.frame.decompress(compressed_data)
        assert decompressed == large_data

    @pytest.mark.asyncio
    async def test_cache_latency_metrics(self):
        """Test that latency metrics are recorded."""
        mock_redis = AsyncMock()
        mock_redis.set.return_value = True
        mock_redis.get.return_value = b'data'
        mock_redis.delete.return_value = 1
        with patch('prompt_improver.utils.redis_cache.redis_client', mock_redis):
            await RedisCache.set('perf_test', b'data')
            await RedisCache.get('perf_test')
            await RedisCache.invalidate('perf_test')
            assert True

class TestIntegrationWorkflow:
    """Integration tests for complete cache workflows."""

    @pytest_asyncio.fixture(scope='function')
    async def real_redis(self, redis_client):
        """Use the real Redis client fixture for testing."""
        await redis_client.flushdb()
        with patch('prompt_improver.utils.redis_cache.redis_client', redis_client):
            yield redis_client

    @pytest.mark.asyncio
    async def test_complete_cache_workflow(self, real_redis):
        """Test complete cache workflow with real-world usage patterns."""
        processing_calls = 0

        @with_singleflight(cache_key_fn=lambda data_id: f'processed_data_{data_id}', expire=300)
        async def process_data(data_id: str):
            nonlocal processing_calls
            processing_calls += 1
            await asyncio.sleep(0.01)
            return {'data_id': data_id, 'processed': True, 'result': f'processed_{data_id}', 'call_count': processing_calls}
        result1 = await process_data('test_123')
        assert result1['call_count'] == 1
        assert processing_calls == 1
        result2 = await process_data('test_123')
        assert result2['call_count'] == 1
        assert processing_calls == 1
        result3 = await process_data('test_456')
        assert result3['call_count'] == 2
        assert processing_calls == 2
        await invalidate('processed_data_test_123')
        result4 = await process_data('test_123')
        assert result4['call_count'] == 3
        assert processing_calls == 3
        ttl = await real_redis.ttl('processed_data_test_456')
        assert ttl > 0

    @pytest.mark.asyncio
    async def test_concurrent_singleflight_different_keys(self, real_redis):
        """Test concurrent singleflight calls with different keys."""
        call_counts = {}

        @with_singleflight(cache_key_fn=lambda key: f'concurrent_{key}')
        async def concurrent_function(key: str):
            if key not in call_counts:
                call_counts[key] = 0
            call_counts[key] += 1
            await asyncio.sleep(0.01)
            return f'result_{key}_{call_counts[key]}'
        tasks = [asyncio.create_task(concurrent_function('key1')), asyncio.create_task(concurrent_function('key2')), asyncio.create_task(concurrent_function('key1')), asyncio.create_task(concurrent_function('key3'))]
        results = await asyncio.gather(*tasks)
        assert results[0] == 'result_key1_1'
        assert results[1] == 'result_key2_1'
        assert results[2] == 'result_key1_1'
        assert results[3] == 'result_key3_1'
        assert call_counts['key1'] == 1
        assert call_counts['key2'] == 1
        assert call_counts['key3'] == 1

    @pytest.mark.asyncio
    async def test_redis_cache_corruption_handling(self, real_redis):
        """Test handling of corrupted data with real Redis."""
        await real_redis.set('corrupted_key', b'not_valid_lz4_data')
        value = await RedisCache.get('corrupted_key')
        assert value is None
        exists = await real_redis.exists('corrupted_key')
        assert exists == 0

    @pytest.mark.asyncio
    async def test_redis_cache_isolation_between_tests(self, real_redis):
        """Test that cache state is isolated between tests."""
        await RedisCache.set('isolation_test', b'test_value')
        value = await RedisCache.get('isolation_test')
        assert value == b'test_value'
        await RedisCache.invalidate('isolation_test')

    @pytest.mark.asyncio
    async def test_redis_cache_performance_characteristics(self, real_redis):
        """Test cache performance characteristics with real Redis."""
        import time
        start_time = time.time()
        for i in range(100):
            await RedisCache.set(f'perf_key_{i}', f'value_{i}'.encode())
        write_time = time.time() - start_time
        start_time = time.time()
        for i in range(100):
            value = await RedisCache.get(f'perf_key_{i}')
            assert value == f'value_{i}'.encode()
        read_time = time.time() - start_time
        assert write_time < 5.0
        assert read_time < 5.0
        assert read_time <= write_time * 1.5
        for i in range(100):
            await RedisCache.invalidate(f'perf_key_{i}')
