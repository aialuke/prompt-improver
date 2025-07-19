#!/usr/bin/env python3
"""
Demo script to show Redis fixture functionality using Testcontainers.

This script demonstrates:
1. Starting a Redis container
2. Connecting to it with a Redis client
3. Performing basic operations
4. Cleanup
"""

import asyncio
import redis.asyncio as aioredis
from testcontainers.redis import RedisContainer


async def demo_redis_fixture():
    """Demo Redis fixture functionality."""
    print("🚀 Starting Redis container...")
    
    try:
        # This is what happens in the session fixture
        with RedisContainer() as redis_container:
            # Get connection details
            host = redis_container.get_container_host_ip()
            port = redis_container.get_exposed_port(6379)
            print(f"✅ Redis container started at {host}:{port}")
            
            # This is what happens in the function fixture
            client = aioredis.Redis(
                host=host,
                port=port,
                decode_responses=True
            )
            
            # Clean the database (like flushdb in fixture)
            await client.flushdb()
            print("🧹 Database flushed")
            
            # Test basic operations
            print("\n📝 Testing basic operations...")
            await client.set("test_key", "test_value")
            value = await client.get("test_key")
            print(f"SET/GET test: {value}")
            
            # Test hash operations
            print("\n🗂️ Testing hash operations...")
            await client.hset("test_hash", "field1", "value1")
            await client.hset("test_hash", "field2", "value2")
            hash_data = await client.hgetall("test_hash")
            print(f"Hash data: {hash_data}")
            
            # Test list operations
            print("\n📋 Testing list operations...")
            await client.lpush("test_list", "item1", "item2", "item3")
            list_data = await client.lrange("test_list", 0, -1)
            print(f"List data: {list_data}")
            
            # Test isolation (flush again)
            await client.flushdb()
            keys_after_flush = await client.keys("*")
            print(f"\n🔄 After flush, keys remaining: {keys_after_flush}")
            
            # Cleanup
            await client.close()
            print("🔌 Client connection closed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("✅ Redis fixture demo completed successfully!")
    return True


if __name__ == "__main__":
    success = asyncio.run(demo_redis_fixture())
    if success:
        print("\n🎉 Redis fixture is working correctly!")
        print("This demonstrates that the pytest fixture will:")
        print("  - Start a Redis container per test session")
        print("  - Provide a clean Redis client per test function")
        print("  - Automatically flush the database between tests")
        print("  - Clean up resources when done")
    else:
        print("\n💥 Redis fixture demo failed!")
        exit(1)
