"""
Integration tests for Redis fixture using Testcontainers.

Tests verify:
- Redis container starts properly
- Redis client connects and operates
- Database flushes between tests
- Proper cleanup after tests
"""
import coredis
import pytest
pytestmark = pytest.mark.redis_integration

@pytest.mark.asyncio
async def test_redis_fixture_basic_operations(redis_client):
    """Test basic Redis operations with the fixture."""
    await redis_client.set('test_key', 'test_value')
    value = await redis_client.get('test_key')
    assert value == 'test_value'
    await redis_client.delete(['test_key'])
    deleted_value = await redis_client.get('test_key')
    assert deleted_value is None

@pytest.mark.asyncio
async def test_redis_fixture_isolation(redis_client):
    """Test that Redis database is clean between tests."""
    value = await redis_client.get('test_key')
    assert value is None
    await redis_client.set('isolation_test', 'should_not_persist')
    value = await redis_client.get('isolation_test')
    assert value == 'should_not_persist'

@pytest.mark.asyncio
async def test_redis_fixture_isolation_verification(redis_client):
    """Test that verifies isolation by ensuring previous test data is gone."""
    value = await redis_client.get('isolation_test')
    assert value is None

@pytest.mark.asyncio
async def test_redis_fixture_hash_operations(redis_client):
    """Test Redis hash operations with the fixture."""
    await redis_client.hset('test_hash', {'field1': 'value1', 'field2': 'value2'})
    value1 = await redis_client.hget('test_hash', 'field1')
    value2 = await redis_client.hget('test_hash', 'field2')
    assert value1 == 'value1'
    assert value2 == 'value2'
    all_fields = await redis_client.hgetall('test_hash')
    assert all_fields == {'field1': 'value1', 'field2': 'value2'}

@pytest.mark.asyncio
async def test_redis_fixture_list_operations(redis_client):
    """Test Redis list operations with the fixture."""
    await redis_client.lpush('test_list', ['item1', 'item2', 'item3'])
    items = await redis_client.lrange('test_list', 0, -1)
    assert items == ['item3', 'item2', 'item1']
    popped = await redis_client.rpop('test_list')
    assert popped == 'item1'
    remaining = await redis_client.lrange('test_list', 0, -1)
    assert remaining == ['item3', 'item2']

@pytest.mark.asyncio
async def test_redis_fixture_expiration(redis_client):
    """Test Redis key expiration with the fixture."""
    await redis_client.setex('expiring_key', 'expiring_value', 1)
    value = await redis_client.get('expiring_key')
    assert value == 'expiring_value'
    ttl = await redis_client.ttl('expiring_key')
    assert ttl > 0

@pytest.mark.asyncio
async def test_redis_fixture_pipeline_operations(redis_client):
    """Test Redis pipeline operations with the fixture."""
    pipe = await redis_client.pipeline()
    pipe.set('pipe_key1', 'value1')
    pipe.set('pipe_key2', 'value2')
    pipe.get('pipe_key1')
    pipe.get('pipe_key2')
    results = await pipe.execute()
    assert results[0] is True
    assert results[1] is True
    assert results[2] == 'value1'
    assert results[3] == 'value2'

@pytest.mark.asyncio
async def test_redis_container_connection_details(redis_container):
    """Test that Redis container provides valid connection details."""
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    assert host is not None
    assert port is not None
    direct_client = coredis.Redis(host=host, port=port, decode_responses=True)
    await direct_client.set('direct_test', 'direct_value')
    value = await direct_client.get('direct_test')
    assert value == 'direct_value'
    direct_client.connection_pool.disconnect()

@pytest.mark.asyncio
async def test_redis_fixture_concurrent_operations(redis_client):
    """Test concurrent Redis operations with the fixture."""
    import asyncio

    async def set_value(key, value):
        await redis_client.set(key, value)
        return await redis_client.get(key)
    tasks = [set_value('concurrent_key1', 'value1'), set_value('concurrent_key2', 'value2'), set_value('concurrent_key3', 'value3')]
    results = await asyncio.gather(*tasks)
    assert results == ['value1', 'value2', 'value3']
    keys = await redis_client.keys('concurrent_key*')
    assert len(keys) == 3
