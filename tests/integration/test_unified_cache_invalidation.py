"""Integration tests for unified cache invalidation using real behavior testing.

This test suite validates that the unified cache architecture properly
handles cache invalidation operations without relying on deprecated
pub/sub patterns.
"""

import pytest
import pytest_asyncio

from prompt_improver.services.cache.l2_redis_service import L2RedisService
from prompt_improver.services.cache.cache_facade import CacheFacade


@pytest.fixture
async def l2_redis_service(redis_client):
    """Create L2RedisService with test Redis client."""
    service = L2RedisService()
    # Inject test Redis client
    service._client = redis_client
    
    try:
        yield service
    finally:
        await service.close()


@pytest.fixture
async def cache_facade(redis_client):
    """Create CacheFacade with test Redis client for integration testing."""
    facade = CacheFacade()
    # Inject test client into L2 service if available
    if hasattr(facade, '_l2_service') and facade._l2_service:
        facade._l2_service._client = redis_client
    
    try:
        yield facade
    finally:
        if hasattr(facade, '_l2_service') and facade._l2_service:
            await facade._l2_service.close()


class TestUnifiedCacheInvalidation:
    """Test unified cache invalidation functionality."""

    async def test_pattern_invalidation_l2_service(self, l2_redis_service):
        """Test pattern-based invalidation using L2RedisService."""
        # Set up test data
        test_data = {
            "apes:pattern:key1": "value1",
            "apes:pattern:key2": "value2", 
            "apes:pattern:key3": "value3",
            "other:key1": "other_value",
        }
        
        # Add all test data
        for key, value in test_data.items():
            await l2_redis_service.set(key, value)
        
        # Verify all keys exist
        for key, expected_value in test_data.items():
            actual_value = await l2_redis_service.get(key)
            assert actual_value == expected_value
        
        # Invalidate pattern
        deleted_count = await l2_redis_service.invalidate_pattern("apes:pattern:*")
        assert deleted_count == 3
        
        # Verify pattern keys are gone
        assert await l2_redis_service.get("apes:pattern:key1") is None
        assert await l2_redis_service.get("apes:pattern:key2") is None
        assert await l2_redis_service.get("apes:pattern:key3") is None
        
        # Verify other keys remain
        assert await l2_redis_service.get("other:key1") == "other_value"
        
        # Cleanup
        await l2_redis_service.delete("other:key1")

    async def test_individual_key_invalidation(self, l2_redis_service):
        """Test individual key deletion."""
        # Set up test data
        await l2_redis_service.set("test:key1", "value1")
        await l2_redis_service.set("test:key2", "value2")
        
        # Verify keys exist
        assert await l2_redis_service.get("test:key1") == "value1"
        assert await l2_redis_service.get("test:key2") == "value2"
        
        # Delete one key
        deleted = await l2_redis_service.delete("test:key1")
        assert deleted is True
        
        # Verify deletion
        assert await l2_redis_service.get("test:key1") is None
        assert await l2_redis_service.get("test:key2") == "value2"
        
        # Cleanup
        await l2_redis_service.delete("test:key2")

    async def test_cache_facade_integration(self, cache_facade):
        """Test cache invalidation through the facade."""
        # This tests that cache facade properly delegates to L2 service
        test_key = "facade:test:key"
        test_value = "facade_test_value"
        
        # Set through facade
        await cache_facade.set(test_key, test_value)
        
        # Get through facade
        retrieved_value = await cache_facade.get(test_key)
        assert retrieved_value == test_value
        
        # Delete through facade  
        await cache_facade.delete(test_key)
        
        # Verify deletion
        deleted_value = await cache_facade.get(test_key)
        assert deleted_value is None

    async def test_multiple_pattern_invalidation(self, l2_redis_service):
        """Test invalidation of multiple different patterns."""
        # Set up test data with different patterns
        test_data = {
            "rule:clarity:v1": "rule_data_1",
            "rule:specificity:v1": "rule_data_2", 
            "model:ensemble:v1": "model_data_1",
            "model:neural:v1": "model_data_2",
            "analytics:performance": "analytics_data",
        }
        
        # Add all test data
        for key, value in test_data.items():
            await l2_redis_service.set(key, value)
        
        # Invalidate rule patterns
        rule_deleted = await l2_redis_service.invalidate_pattern("rule:*")
        assert rule_deleted == 2
        
        # Invalidate model patterns  
        model_deleted = await l2_redis_service.invalidate_pattern("model:*")
        assert model_deleted == 2
        
        # Verify pattern keys are gone
        assert await l2_redis_service.get("rule:clarity:v1") is None
        assert await l2_redis_service.get("rule:specificity:v1") is None
        assert await l2_redis_service.get("model:ensemble:v1") is None
        assert await l2_redis_service.get("model:neural:v1") is None
        
        # Verify analytics key remains
        assert await l2_redis_service.get("analytics:performance") == "analytics_data"
        
        # Cleanup
        await l2_redis_service.delete("analytics:performance")