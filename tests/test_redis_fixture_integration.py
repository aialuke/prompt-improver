"""
Integration tests for Redis fixture using Testcontainers.

Tests verify:
- Redis container starts properly
- Redis client connects and operates
- Database flushes between tests
- Proper cleanup after tests
"""

import pytest
import redis.asyncio as aioredis


pytestmark = pytest.mark.redis_integration


@pytest.mark.asyncio
async def test_redis_fixture_basic_operations(redis_client):
    """Test basic Redis operations with the fixture."""
    # Test SET operation
    await redis_client.set("test_key", "test_value")
    
    # Test GET operation
    value = await redis_client.get("test_key")
    assert value == "test_value"
    
    # Test DELETE operation
    await redis_client.delete("test_key")
    deleted_value = await redis_client.get("test_key")
    assert deleted_value is None


@pytest.mark.asyncio
async def test_redis_fixture_isolation(redis_client):
    """Test that Redis database is clean between tests."""
    # This test should not see data from previous test
    value = await redis_client.get("test_key")
    assert value is None
    
    # Set a value that should not persist to next test
    await redis_client.set("isolation_test", "should_not_persist")
    value = await redis_client.get("isolation_test")
    assert value == "should_not_persist"


@pytest.mark.asyncio
async def test_redis_fixture_isolation_verification(redis_client):
    """Test that verifies isolation by ensuring previous test data is gone."""
    # This should not see data from previous test
    value = await redis_client.get("isolation_test")
    assert value is None


@pytest.mark.asyncio
async def test_redis_fixture_hash_operations(redis_client):
    """Test Redis hash operations with the fixture."""
    # Test HSET operation
    await redis_client.hset("test_hash", "field1", "value1")
    await redis_client.hset("test_hash", "field2", "value2")
    
    # Test HGET operation
    value1 = await redis_client.hget("test_hash", "field1")
    value2 = await redis_client.hget("test_hash", "field2")
    assert value1 == "value1"
    assert value2 == "value2"
    
    # Test HGETALL operation
    all_fields = await redis_client.hgetall("test_hash")
    assert all_fields == {"field1": "value1", "field2": "value2"}


@pytest.mark.asyncio
async def test_redis_fixture_list_operations(redis_client):
    """Test Redis list operations with the fixture."""
    # Test LPUSH operation
    await redis_client.lpush("test_list", "item1", "item2", "item3")
    
    # Test LRANGE operation
    items = await redis_client.lrange("test_list", 0, -1)
    assert items == ["item3", "item2", "item1"]  # LPUSH adds to front
    
    # Test RPOP operation
    popped = await redis_client.rpop("test_list")
    assert popped == "item1"
    
    # Verify list state after pop
    remaining = await redis_client.lrange("test_list", 0, -1)
    assert remaining == ["item3", "item2"]


@pytest.mark.asyncio
async def test_redis_fixture_expiration(redis_client):
    """Test Redis key expiration with the fixture."""
    # Set a key with expiration
    await redis_client.setex("expiring_key", 1, "expiring_value")
    
    # Verify key exists
    value = await redis_client.get("expiring_key")
    assert value == "expiring_value"
    
    # Check TTL
    ttl = await redis_client.ttl("expiring_key")
    assert ttl > 0


@pytest.mark.asyncio
async def test_redis_fixture_pipeline_operations(redis_client):
    """Test Redis pipeline operations with the fixture."""
    # Create a pipeline
    pipe = redis_client.pipeline()
    
    # Add multiple operations to pipeline
    pipe.set("pipe_key1", "value1")
    pipe.set("pipe_key2", "value2")
    pipe.get("pipe_key1")
    pipe.get("pipe_key2")
    
    # Execute pipeline
    results = await pipe.execute()
    
    # Verify results
    assert results[0] is True  # SET result
    assert results[1] is True  # SET result
    assert results[2] == "value1"  # GET result
    assert results[3] == "value2"  # GET result


@pytest.mark.asyncio
async def test_redis_container_connection_details(redis_container):
    """Test that Redis container provides valid connection details."""
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    assert host is not None
    assert port is not None
    
    # Test direct connection using the connection details
    direct_client = aioredis.Redis(host=host, port=port, decode_responses=True)
    await direct_client.set("direct_test", "direct_value")
    value = await direct_client.get("direct_test")
    assert value == "direct_value"
    await direct_client.close()


@pytest.mark.asyncio
async def test_redis_fixture_concurrent_operations(redis_client):
    """Test concurrent Redis operations with the fixture."""
    import asyncio
    
    async def set_value(key, value):
        await redis_client.set(key, value)
        return await redis_client.get(key)
    
    # Execute multiple operations concurrently
    tasks = [
        set_value("concurrent_key1", "value1"),
        set_value("concurrent_key2", "value2"),
        set_value("concurrent_key3", "value3"),
    ]
    
    results = await asyncio.gather(*tasks)
    assert results == ["value1", "value2", "value3"]
    
    # Verify all keys exist
    keys = await redis_client.keys("concurrent_key*")
    assert len(keys) == 3
