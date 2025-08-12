"""
Cache Protocol Compliance Tests

Comprehensive test suite to validate that DatabaseServices
properly implements all cache protocol interfaces according to
cache_protocol.py specifications.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

from prompt_improver.core.protocols.cache_protocol import (
    AdvancedCacheProtocol,
    BasicCacheProtocol,
    CacheHealthProtocol,
    CacheLockProtocol,
    CacheSubscriptionProtocol,
    MultiLevelCacheProtocol,
    RedisCacheProtocol,
)
from prompt_improver.database import (
    ManagerMode,
    SecurityContext,
    DatabaseServices,
)


class TestBasicCacheProtocolCompliance:
    """Test BasicCacheProtocol method compliance."""

    @pytest.fixture
    async def manager(self):
        """Create DatabaseServices for testing."""
        manager = DatabaseServices(mode=ManagerMode.ASYNC_MODERN)
        await manager.initialize()
        yield manager
        await manager.close()

    async def test_implements_basic_cache_protocol(self, manager):
        """Test that DatabaseServices implements BasicCacheProtocol."""
        assert isinstance(manager, BasicCacheProtocol)

    async def test_get_method_compliance(self, manager):
        """Test get method signature and behavior compliance."""
        assert hasattr(manager, "get")
        assert callable(manager.get)
        result = await manager.get("non_existent_key")
        assert result is None
        await manager.set("test_key", "test_value")
        result = await manager.get("test_key")
        assert result == "test_value"

    async def test_set_method_compliance(self, manager):
        """Test set method signature and behavior compliance."""
        assert hasattr(manager, "set")
        assert callable(manager.set)
        result = await manager.set("test_key", "test_value")
        assert result is True
        result = await manager.set("ttl_key", "ttl_value", ttl=60)
        assert result is True
        value = await manager.get("test_key")
        assert value == "test_value"

    async def test_delete_method_compliance(self, manager):
        """Test delete method signature and behavior compliance."""
        assert hasattr(manager, "delete")
        assert callable(manager.delete)
        result = await manager.delete("non_existent_key")
        assert result is False
        await manager.set("delete_test", "value")
        result = await manager.delete("delete_test")
        assert result is True
        value = await manager.get("delete_test")
        assert value is None

    async def test_exists_method_compliance(self, manager):
        """Test exists method signature and behavior compliance."""
        assert hasattr(manager, "exists")
        assert callable(manager.exists)
        result = await manager.exists("non_existent_key")
        assert result is False
        await manager.set("exists_test", "value")
        result = await manager.exists("exists_test")
        assert result is True

    async def test_clear_method_compliance(self, manager):
        """Test clear method signature and behavior compliance."""
        assert hasattr(manager, "clear")
        assert callable(manager.clear)
        await manager.set("clear_test_1", "value1")
        await manager.set("clear_test_2", "value2")
        result = await manager.clear()
        assert result is True
        value1 = await manager.get("clear_test_1")
        value2 = await manager.get("clear_test_2")
        assert value1 is None
        assert value2 is None


class TestAdvancedCacheProtocolCompliance:
    """Test AdvancedCacheProtocol method compliance."""

    @pytest.fixture
    async def manager(self):
        """Create DatabaseServices for testing."""
        manager = DatabaseServices(mode=ManagerMode.ASYNC_MODERN)
        await manager.initialize()
        yield manager
        await manager.close()

    async def test_implements_advanced_cache_protocol(self, manager):
        """Test that DatabaseServices implements AdvancedCacheProtocol."""
        assert isinstance(manager, AdvancedCacheProtocol)

    async def test_get_many_method_compliance(self, manager):
        """Test get_many method signature and behavior compliance."""
        assert hasattr(manager, "get_many")
        assert callable(manager.get_many)
        await manager.set("key1", "value1")
        await manager.set("key2", "value2")
        await manager.set("key3", "value3")
        result = await manager.get_many(["key1", "key2", "key4"])
        expected = {"key1": "value1", "key2": "value2"}
        assert result == expected

    async def test_set_many_method_compliance(self, manager):
        """Test set_many method signature and behavior compliance."""
        assert hasattr(manager, "set_many")
        assert callable(manager.set_many)
        mapping = {"batch_key1": "batch_value1", "batch_key2": "batch_value2"}
        result = await manager.set_many(mapping)
        assert result is True
        value1 = await manager.get("batch_key1")
        value2 = await manager.get("batch_key2")
        assert value1 == "batch_value1"
        assert value2 == "batch_value2"

    async def test_delete_many_method_compliance(self, manager):
        """Test delete_many method signature and behavior compliance."""
        assert hasattr(manager, "delete_many")
        assert callable(manager.delete_many)
        await manager.set("del_key1", "value1")
        await manager.set("del_key2", "value2")
        await manager.set("del_key3", "value3")
        result = await manager.delete_many(["del_key1", "del_key2", "del_key4"])
        assert result == 2
        value1 = await manager.get("del_key1")
        value2 = await manager.get("del_key2")
        value3 = await manager.get("del_key3")
        assert value1 is None
        assert value2 is None
        assert value3 == "value3"

    async def test_get_or_set_method_compliance(self, manager):
        """Test get_or_set method signature and behavior compliance."""
        assert hasattr(manager, "get_or_set")
        assert callable(manager.get_or_set)

        def default_func():
            return "computed_value"

        result = await manager.get_or_set("compute_key", default_func)
        assert result == "computed_value"
        cached_value = await manager.get("compute_key")
        assert cached_value == "computed_value"

        def should_not_be_called():
            raise Exception("This should not be called")

        result = await manager.get_or_set("compute_key", should_not_be_called)
        assert result == "computed_value"

    async def test_get_or_set_async_function(self, manager):
        """Test get_or_set with async default function."""

        async def async_default_func():
            await asyncio.sleep(0.01)
            return "async_computed_value"

        result = await manager.get_or_set("async_compute_key", async_default_func)
        assert result == "async_computed_value"

    @pytest.mark.skipif(True, reason="Redis increment requires Redis connection")
    async def test_increment_method_compliance(self, manager):
        """Test increment method signature and behavior compliance."""
        assert hasattr(manager, "increment")
        assert callable(manager.increment)

    @pytest.mark.skipif(True, reason="Redis expire requires Redis connection")
    async def test_expire_method_compliance(self, manager):
        """Test expire method signature and behavior compliance."""
        assert hasattr(manager, "expire")
        assert callable(manager.expire)


class TestCacheHealthProtocolCompliance:
    """Test CacheHealthProtocol method compliance."""

    @pytest.fixture
    async def manager(self):
        """Create DatabaseServices for testing."""
        manager = DatabaseServices(mode=ManagerMode.ASYNC_MODERN)
        await manager.initialize()
        yield manager
        await manager.close()

    async def test_implements_cache_health_protocol(self, manager):
        """Test that DatabaseServices implements CacheHealthProtocol."""
        assert isinstance(manager, CacheHealthProtocol)

    async def test_ping_method_compliance(self, manager):
        """Test ping method signature and behavior compliance."""
        assert hasattr(manager, "ping")
        assert callable(manager.ping)
        result = await manager.ping()
        assert isinstance(result, bool)
        assert result is True

    async def test_get_info_method_compliance(self, manager):
        """Test get_info method signature and behavior compliance."""
        assert hasattr(manager, "get_info")
        assert callable(manager.get_info)
        result = await manager.get_info()
        assert isinstance(result, dict)
        assert "service" in result
        assert "components" in result
        assert "features" in result
        assert result["protocol_compliance"] is True

    async def test_get_stats_method_compliance(self, manager):
        """Test get_stats method signature and behavior compliance."""
        assert hasattr(manager, "get_stats")
        assert callable(manager.get_stats)
        result = await manager.get_stats()
        assert isinstance(result, dict)
        assert "protocol_compliance" in result
        assert "security" in result

    async def test_get_memory_usage_method_compliance(self, manager):
        """Test get_memory_usage method signature and behavior compliance."""
        assert hasattr(manager, "get_memory_usage")
        assert callable(manager.get_memory_usage)
        result = await manager.get_memory_usage()
        assert isinstance(result, dict)
        assert "l1_cache" in result
        assert "l2_cache" in result
        assert "access_patterns" in result


class TestMultiLevelCacheProtocolCompliance:
    """Test MultiLevelCacheProtocol method compliance."""

    @pytest.fixture
    async def manager(self):
        """Create DatabaseServices for testing."""
        manager = DatabaseServices(mode=ManagerMode.ASYNC_MODERN)
        await manager.initialize()
        yield manager
        await manager.close()

    async def test_implements_multilevel_cache_protocol(self, manager):
        """Test that DatabaseServices implements MultiLevelCacheProtocol."""
        assert isinstance(manager, MultiLevelCacheProtocol)

    async def test_get_from_level_method_compliance(self, manager):
        """Test get_from_level method signature and behavior compliance."""
        assert hasattr(manager, "get_from_level")
        assert callable(manager.get_from_level)
        await manager.set("level_test", "level_value")
        result = await manager.get_from_level("level_test", 1)
        assert result == "level_value"
        result = await manager.get_from_level("level_test", 99)
        assert result is None

    async def test_set_to_level_method_compliance(self, manager):
        """Test set_to_level method signature and behavior compliance."""
        assert hasattr(manager, "set_to_level")
        assert callable(manager.set_to_level)
        result = await manager.set_to_level("level_set_test", "level_set_value", 1)
        assert result is True
        value = await manager.get_from_level("level_set_test", 1)
        assert value == "level_set_value"

    async def test_invalidate_levels_method_compliance(self, manager):
        """Test invalidate_levels method signature and behavior compliance."""
        assert hasattr(manager, "invalidate_levels")
        assert callable(manager.invalidate_levels)
        await manager.set_to_level("invalidate_test", "invalidate_value", 1)
        result = await manager.invalidate_levels("invalidate_test", [1])
        assert result is True
        value = await manager.get_from_level("invalidate_test", 1)
        assert value is None

    async def test_get_cache_hierarchy_method_compliance(self, manager):
        """Test get_cache_hierarchy method signature and behavior compliance."""
        assert hasattr(manager, "get_cache_hierarchy")
        assert callable(manager.get_cache_hierarchy)
        result = await manager.get_cache_hierarchy()
        assert isinstance(result, list)
        assert len(result) > 0
        assert any("L1" in level for level in result)


class TestRedisCacheProtocolCompliance:
    """Test RedisCacheProtocol (combined protocol) compliance."""

    @pytest.fixture
    async def manager(self):
        """Create DatabaseServices for testing."""
        manager = DatabaseServices(mode=ManagerMode.ASYNC_MODERN)
        await manager.initialize()
        yield manager
        await manager.close()

    async def test_implements_redis_cache_protocol(self, manager):
        """Test that DatabaseServices implements RedisCacheProtocol."""
        assert isinstance(manager, RedisCacheProtocol)

    async def test_all_protocol_methods_exist(self, manager):
        """Test that all required protocol methods exist."""
        assert hasattr(manager, "get")
        assert hasattr(manager, "set")
        assert hasattr(manager, "delete")
        assert hasattr(manager, "exists")
        assert hasattr(manager, "clear")
        assert hasattr(manager, "get_many")
        assert hasattr(manager, "set_many")
        assert hasattr(manager, "delete_many")
        assert hasattr(manager, "get_or_set")
        assert hasattr(manager, "increment")
        assert hasattr(manager, "expire")
        assert hasattr(manager, "ping")
        assert hasattr(manager, "get_info")
        assert hasattr(manager, "get_stats")
        assert hasattr(manager, "get_memory_usage")
        assert hasattr(manager, "publish")
        assert hasattr(manager, "subscribe")
        assert hasattr(manager, "unsubscribe")
        assert hasattr(manager, "acquire_lock")
        assert hasattr(manager, "release_lock")
        assert hasattr(manager, "extend_lock")


class TestProtocolSecurityCompliance:
    """Test protocol compliance with security context requirements."""

    @pytest.fixture
    async def secure_manager(self):
        """Create DatabaseServices with security requirements."""
        manager = DatabaseServices(mode=ManagerMode.HIGH_AVAILABILITY)
        await manager.initialize()
        yield manager
        await manager.close()

    async def test_protocol_methods_with_security_context(self, secure_manager):
        """Test that protocol methods work with security context enabled."""
        result = await secure_manager.set("secure_test", "secure_value")
        assert result is True
        value = await secure_manager.get("secure_test")
        assert value == "secure_value"
        exists = await secure_manager.exists("secure_test")
        assert exists is True
        deleted = await secure_manager.delete("secure_test")
        assert deleted is True


class TestProtocolPerformanceCompliance:
    """Test protocol compliance under performance constraints."""

    @pytest.fixture
    async def mcp_manager(self):
        """Create MCP-optimized DatabaseServices."""
        manager = DatabaseServices(mode=ManagerMode.MCP_SERVER)
        await manager.initialize()
        yield manager
        await manager.close()

    async def test_protocol_methods_performance(self, mcp_manager):
        """Test that protocol methods meet performance requirements."""
        start_time = time.time()
        await mcp_manager.set("perf_test_1", "value1")
        await mcp_manager.set("perf_test_2", "value2")
        await mcp_manager.get("perf_test_1")
        await mcp_manager.get("perf_test_2")
        await mcp_manager.exists("perf_test_1")
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        assert total_time_ms < 200, (
            f"Protocol operations took {total_time_ms:.2f}ms, expected < 200ms"
        )


@pytest.mark.asyncio
async def test_protocol_compliance_integration():
    """Integration test for complete protocol compliance."""
    manager = DatabaseServices(mode=ManagerMode.ASYNC_MODERN)
    await manager.initialize()
    try:
        assert isinstance(manager, BasicCacheProtocol)
        assert isinstance(manager, AdvancedCacheProtocol)
        assert isinstance(manager, CacheHealthProtocol)
        assert isinstance(manager, CacheSubscriptionProtocol)
        assert isinstance(manager, CacheLockProtocol)
        assert isinstance(manager, RedisCacheProtocol)
        assert isinstance(manager, MultiLevelCacheProtocol)
        await manager.set("integration_test", "integration_value")
        value = await manager.get("integration_test")
        assert value == "integration_value"
        health = await manager.ping()
        assert isinstance(health, bool)
        info = await manager.get_info()
        assert info["protocol_compliance"] is True
        hierarchy = await manager.get_cache_hierarchy()
        assert len(hierarchy) > 0
        print("✅ All cache protocol compliance tests passed!")
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(test_protocol_compliance_integration())
