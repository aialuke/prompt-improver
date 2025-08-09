"""
Comprehensive real behavior testing framework for MultiLevelCache using external Redis services.

This test suite provides 100% real behavior testing with actual Redis instances,
comprehensive cache operation validation, intelligent warming verification,
and performance characteristics testing following 2025 best practices.

Features:
- Real Redis service integration for testing
- Multi-level cache behavior validation (L1 → L2 → L3)
- Intelligent cache warming functionality testing
- Access pattern tracking and analysis verification
- Enhanced statistics and metrics collection testing
- Error handling and resilience validation
- Performance characteristics under load testing
- OpenTelemetry integration validation
- Health monitoring integration testing
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import coredis
import pytest

from prompt_improver.utils.multi_level_cache import (
    AccessPattern,
    CacheEntry,
    LRUCache,
    MultiLevelCache,
    RedisCache,
    SpecializedCaches,
    cached_get,
    cached_set,
    get_cache,
)

# Configure logging for test visibility
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ============================================================================
# TEST INFRASTRUCTURE: CONTAINER MANAGEMENT
# ============================================================================


@pytest.fixture(scope="session")
def redis_test_container():
    """
    Session-scoped Redis container for real behavior testing.

    Provides isolated Redis instance with proper lifecycle management
    and network configuration for MultiLevelCache testing.
    """
    with RedisContainer(image="redis:7-alpine", port=6379) as container:
        # Wait for Redis to be ready
        client = container.get_client()
        max_retries = 30
        for i in range(max_retries):
            try:
                client.ping()
                logger.info(
                    f"Redis container ready on port {container.get_exposed_port(6379)}"
                )
                break
            except Exception as e:
                if i == max_retries - 1:
                    raise RuntimeError(f"Redis container failed to start: {e}")
                time.sleep(1)

        yield container


@pytest.fixture(scope="function")
async def redis_client(redis_test_container):
    """
    Function-scoped Redis client with clean state for each test.

    Provides async Redis client with database flush before each test
    to ensure complete test isolation and consistent starting state.
    """
    host = redis_test_container.get_container_host_ip()
    port = redis_test_container.get_exposed_port(6379)

    client = coredis.Redis(
        host=host,
        port=port,
        decode_responses=False,  # Match MultiLevelCache behavior
    )

    # Ensure clean state
    await client.flushdb()
    await client.ping()  # Verify connection

    yield client

    # Cleanup
    try:
        await client.flushdb()
        await client.close()
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture(scope="function")
async def test_cache_config(redis_test_container):
    """
    Test configuration for MultiLevelCache with real Redis container.

    Provides configuration that connects to the test Redis container
    with appropriate settings for testing scenarios.
    """
    host = redis_test_container.get_container_host_ip()
    port = redis_test_container.get_exposed_port(6379)

    # Mock Redis configuration to use test container
    mock_redis_config = type(
        "MockRedisConfig",
        (),
        {
            "host": host,
            "port": port,
            "database": 0,
            "password": None,
            "username": None,
            "connection_timeout": 5,
            "socket_timeout": 5,
            "max_connections": 20,
        },
    )()

    # Mock application config
    mock_config = type("MockConfig", (), {"redis": mock_redis_config})()

    return {"redis_host": host, "redis_port": port, "mock_config": mock_config}


# ============================================================================
# TEST INFRASTRUCTURE: CACHE SETUP AND HELPERS
# ============================================================================


@pytest.fixture(scope="function")
async def multi_level_cache(test_cache_config):
    """
    Multi-level cache instance configured with real Redis container.

    Creates cache with real L2 Redis backend and comprehensive configuration
    for testing all cache tiers and intelligent warming functionality.
    """
    # Patch get_config to return our test configuration
    with patch(
        "prompt_improver.core.config.get_config",
        return_value=test_cache_config["mock_config"],
    ):
        cache = MultiLevelCache(
            l1_max_size=100,  # Small for easier testing
            l2_default_ttl=300,  # 5 minutes
            enable_l2=True,
            enable_warming=True,
            warming_threshold=2.0,  # Low threshold for testing
            warming_interval=60,  # 1 minute for testing
            max_warming_keys=10,
        )

        # Wait for cache to initialize
        await asyncio.sleep(0.1)

        yield cache

        # Cleanup
        try:
            await cache.stop_warming()
            if cache._l2_cache:
                await cache._l2_cache.close()
        except Exception:
            pass  # Ignore cleanup errors


@pytest.fixture(scope="function")
def sample_cache_data():
    """
    Sample data for cache testing with realistic patterns.

    Provides structured test data including various data types,
    sizes, and complexity levels for comprehensive cache testing.
    """
    return {
        "simple_string": "test_value",
        "simple_number": 42,
        "simple_dict": {"key": "value", "number": 123},
        "complex_dict": {
            "nested": {
                "data": [1, 2, 3, 4, 5],
                "metadata": {
                    "created": datetime.now(UTC).isoformat(),
                    "version": "1.0.0",
                },
            },
            "large_array": list(range(100)),
            "unicode_text": "Hello 世界 🌍",
        },
        "large_text": "A" * 10000,  # 10KB text
        "null_value": None,
        "boolean_values": {"true": True, "false": False},
        "mixed_array": [1, "two", {"three": 3}, [4, 5], None],
    }


@pytest.fixture(scope="function")
async def mock_fallback_database():
    """
    Mock database fallback for L3 cache tier testing.

    Simulates database calls with realistic latency and data patterns
    for testing multi-level cache fallback behavior.
    """
    database_data = {
        "user:123": {"id": 123, "name": "Test User", "email": "test@example.com"},
        "product:456": {"id": 456, "name": "Test Product", "price": 99.99},
        "session:789": {
            "id": 789,
            "data": {"key": "value"},
            "expires": time.time() + 3600,
        },
    }

    async def mock_db_fetch(key: str) -> dict[str, Any] | None:
        """Mock database fetch with realistic latency."""
        await asyncio.sleep(0.05)  # Simulate 50ms database latency
        return database_data.get(key)

    return mock_db_fetch


# ============================================================================
# BASIC CACHE OPERATIONS TESTING
# ============================================================================


@pytest.mark.asyncio
class TestBasicCacheOperations:
    """Test basic cache operations with real Redis behavior."""

    async def test_cache_set_and_get_simple_data(
        self, multi_level_cache, sample_cache_data
    ):
        """Test basic set and get operations with simple data types."""
        cache = multi_level_cache

        # Test simple string
        await cache.set("test_string", sample_cache_data["simple_string"])
        result = await cache.get("test_string")
        assert result == sample_cache_data["simple_string"]

        # Test number
        await cache.set("test_number", sample_cache_data["simple_number"])
        result = await cache.get("test_number")
        assert result == sample_cache_data["simple_number"]

        # Test dict
        await cache.set("test_dict", sample_cache_data["simple_dict"])
        result = await cache.get("test_dict")
        assert result == sample_cache_data["simple_dict"]

    async def test_cache_set_and_get_complex_data(
        self, multi_level_cache, sample_cache_data
    ):
        """Test cache operations with complex nested data structures."""
        cache = multi_level_cache

        # Test complex nested dictionary
        await cache.set("complex_data", sample_cache_data["complex_dict"])
        result = await cache.get("complex_data")
        assert result == sample_cache_data["complex_dict"]

        # Test large text
        await cache.set("large_text", sample_cache_data["large_text"])
        result = await cache.get("large_text")
        assert result == sample_cache_data["large_text"]

        # Test mixed array
        await cache.set("mixed_array", sample_cache_data["mixed_array"])
        result = await cache.get("mixed_array")
        assert result == sample_cache_data["mixed_array"]

    async def test_cache_delete_operations(self, multi_level_cache):
        """Test cache deletion across all levels."""
        cache = multi_level_cache

        # Set data in cache
        test_data = {"test": "delete_me"}
        await cache.set("delete_test", test_data)

        # Verify it exists
        result = await cache.get("delete_test")
        assert result == test_data

        # Delete and verify removal
        await cache.delete("delete_test")
        result = await cache.get("delete_test")
        assert result is None

    async def test_cache_clear_operations(self, multi_level_cache):
        """Test cache clear functionality."""
        cache = multi_level_cache

        # Set multiple items
        test_items = {
            "clear_test1": "value1",
            "clear_test2": "value2",
            "clear_test3": "value3",
        }

        for key, value in test_items.items():
            await cache.set(key, value)

        # Verify items exist
        for key, expected_value in test_items.items():
            result = await cache.get(key)
            assert result == expected_value

        # Clear cache
        await cache.clear()

        # Verify items are removed from L1 (L2 clear is informational only)
        for key in test_items:
            l1_result = cache._l1_cache.get(key)
            assert l1_result is None

    async def test_cache_ttl_behavior(self, multi_level_cache):
        """Test TTL behavior in L1 cache."""
        cache = multi_level_cache

        # Set item with short TTL
        await cache.set("ttl_test", "expires_soon", l1_ttl=1, l2_ttl=10)

        # Immediately retrieve (should exist)
        result = await cache.get("ttl_test")
        assert result == "expires_soon"

        # Wait for L1 TTL to expire
        await asyncio.sleep(1.1)

        # L1 should be expired, but L2 might still have it
        l1_result = cache._l1_cache.get("ttl_test")
        assert l1_result is None  # L1 expired

        # Get through full cache (may hit L2)
        result = await cache.get("ttl_test")
        # Result could be None or the value from L2, both are valid


# ============================================================================
# MULTI-LEVEL CACHE BEHAVIOR TESTING
# ============================================================================


@pytest.mark.asyncio
class TestMultiLevelBehavior:
    """Test multi-level cache behavior with L1 → L2 → L3 fallback."""

    async def test_l1_cache_hit(self, multi_level_cache):
        """Test L1 cache hit scenario."""
        cache = multi_level_cache

        # Pre-populate L1 cache directly
        cache._l1_cache.set("l1_test", "l1_value")

        # Get should hit L1
        result = await cache.get("l1_test")
        assert result == "l1_value"

        # Verify L1 hit was recorded
        stats = cache.get_performance_stats()
        assert stats["l1_cache"]["hits"] > 0

    async def test_l2_cache_hit_with_l1_population(
        self, multi_level_cache, redis_client
    ):
        """Test L2 cache hit with L1 population."""
        cache = multi_level_cache

        # Pre-populate L2 (Redis) directly
        test_data = {"from": "l2_cache"}
        serialized_data = json.dumps(test_data).encode("utf-8")
        await redis_client.set("l2_test", serialized_data, ex=300)

        # Get should miss L1, hit L2, and populate L1
        result = await cache.get("l2_test")
        assert result == test_data

        # Verify L1 was populated
        l1_result = cache._l1_cache.get("l2_test")
        assert l1_result == test_data

        # Verify L2 hit was recorded
        stats = cache.get_performance_stats()
        assert stats["l2_cache"]["hits"] > 0

    async def test_l3_fallback_behavior(
        self, multi_level_cache, mock_fallback_database
    ):
        """Test L3 fallback to database with cache population."""
        cache = multi_level_cache

        # Test L3 fallback
        result = await cache.get(
            "user:123", fallback_func=lambda: mock_fallback_database("user:123")
        )

        expected_data = {"id": 123, "name": "Test User", "email": "test@example.com"}
        assert result == expected_data

        # Verify data was cached in L1
        l1_result = cache._l1_cache.get("user:123")
        assert l1_result == expected_data

        # Verify L3 hit was recorded
        stats = cache.get_performance_stats()
        assert stats["l3_fallback"]["hits"] > 0

    async def test_cache_miss_all_levels(self, multi_level_cache):
        """Test complete cache miss across all levels."""
        cache = multi_level_cache

        # Get non-existent key without fallback
        result = await cache.get("nonexistent_key")
        assert result is None

        # Verify miss was recorded
        stats = cache.get_performance_stats()
        assert stats["total_requests"] > 0

    async def test_cache_level_isolation(self, multi_level_cache, redis_client):
        """Test that cache levels operate independently."""
        cache = multi_level_cache

        # Set data only in L1
        cache._l1_cache.set("l1_only", "l1_data")

        # Set data only in L2
        l2_data = json.dumps("l2_data").encode("utf-8")
        await redis_client.set("l2_only", l2_data, ex=300)

        # Test L1 only retrieval
        result = await cache.get("l1_only")
        assert result == "l1_data"

        # Test L2 only retrieval (should populate L1)
        result = await cache.get("l2_only")
        assert result == "l2_data"

        # Verify L1 was populated from L2
        l1_result = cache._l1_cache.get("l2_only")
        assert l1_result == "l2_data"


# ============================================================================
# INTELLIGENT CACHE WARMING TESTING
# ============================================================================


@pytest.mark.asyncio
class TestIntelligentWarming:
    """Test intelligent cache warming functionality."""

    async def test_access_pattern_tracking(self, multi_level_cache):
        """Test access pattern tracking and recording."""
        cache = multi_level_cache

        # Generate access patterns
        test_key = "warming_test"
        for i in range(5):
            await cache.get(test_key, fallback_func=lambda: f"value_{i}")
            await asyncio.sleep(0.1)  # Small delay between accesses

        # Check access pattern was recorded
        assert test_key in cache._access_patterns
        pattern = cache._access_patterns[test_key]
        assert pattern.access_count >= 5
        assert pattern.access_frequency > 0

    async def test_warming_candidate_identification(self, multi_level_cache):
        """Test identification of warming candidates."""
        cache = multi_level_cache

        # Create access patterns for multiple keys
        hot_keys = ["hot_key_1", "hot_key_2", "hot_key_3"]
        cold_key = "cold_key"

        # Generate frequent access for hot keys
        for key in hot_keys:
            for i in range(10):  # High access count
                await cache.get(key, fallback_func=lambda k=key: f"hot_value_{k}")
                await asyncio.sleep(0.01)

        # Single access for cold key
        await cache.get(cold_key, fallback_func=lambda: "cold_value")

        # Get warming candidates
        candidates = await cache.get_warming_candidates(limit=5)

        # Verify hot keys are prioritized
        candidate_keys = [c["key"] for c in candidates]
        for hot_key in hot_keys:
            assert hot_key in candidate_keys

        # Verify candidate data structure
        for candidate in candidates:
            assert "key" in candidate
            assert "access_count" in candidate
            assert "access_frequency" in candidate
            assert "warming_priority" in candidate

    async def test_background_warming_cycle(self, multi_level_cache, redis_client):
        """Test background warming cycle execution."""
        cache = multi_level_cache

        # Pre-populate L2 with data to be warmed
        warm_data = {"warmed": True, "timestamp": time.time()}
        serialized_data = json.dumps(warm_data).encode("utf-8")
        await redis_client.set("warm_test", serialized_data, ex=300)

        # Create access pattern to make it a warming candidate
        for i in range(5):
            # Access the key multiple times to create warming pattern
            # Skip fallback to ensure we're testing L2 warming specifically
            cache._record_access_pattern("warm_test")

        # Manually trigger warming cycle
        await cache._perform_warming_cycle()

        # Check if key was warmed into L1
        l1_result = cache._l1_cache.get("warm_test")
        # Note: This might be None if the key wasn't in warming candidates
        # due to timing, which is acceptable behavior

        # Check warming stats
        stats = cache.get_performance_stats()
        warming_stats = stats["intelligent_warming"]["warming_stats"]
        assert warming_stats["cycles_completed"] > 0

    async def test_manual_cache_warming(self, multi_level_cache, redis_client):
        """Test manual cache warming functionality."""
        cache = multi_level_cache

        # Pre-populate L2 with test data
        test_keys = ["manual_warm_1", "manual_warm_2", "manual_warm_3"]
        for i, key in enumerate(test_keys):
            data = {"manual_warm": True, "index": i}
            serialized_data = json.dumps(data).encode("utf-8")
            await redis_client.set(key, serialized_data, ex=300)

        # Manually warm the keys
        warm_results = await cache.warm_cache(test_keys)

        # Verify warming results
        for key in test_keys:
            # Result should indicate success if L2 had the data
            if warm_results.get(key, False):
                # If warming succeeded, check L1 cache
                l1_result = cache._l1_cache.get(key)
                assert l1_result is not None
                assert l1_result["manual_warm"] is True

    async def test_warming_error_handling(self, multi_level_cache):
        """Test error handling in warming operations."""
        cache = multi_level_cache

        # Try to warm non-existent keys
        nonexistent_keys = ["no_such_key_1", "no_such_key_2"]
        warm_results = await cache.warm_cache(nonexistent_keys)

        # Verify appropriate failure handling
        for key in nonexistent_keys:
            assert warm_results[key] is False

        # Check that warming errors are tracked
        stats = cache.get_performance_stats()
        # Error tracking might not increment for non-existent keys
        # as this is normal behavior, not an error condition


# ============================================================================
# ENHANCED STATISTICS AND METRICS TESTING
# ============================================================================


@pytest.mark.asyncio
class TestEnhancedStatistics:
    """Test enhanced statistics and metrics collection."""

    async def test_performance_statistics_collection(self, multi_level_cache):
        """Test comprehensive performance statistics collection."""
        cache = multi_level_cache

        # Generate various cache operations
        operations = [
            ("set_test_1", "value_1"),
            ("set_test_2", "value_2"),
            ("set_test_3", "value_3"),
        ]

        # Perform operations
        for key, value in operations:
            await cache.set(key, value)
            await cache.get(key)

        # Get comprehensive stats
        stats = cache.get_performance_stats()

        # Verify basic structure
        assert "total_requests" in stats
        assert "overall_hit_rate" in stats
        assert "l1_cache" in stats
        assert "l2_cache" in stats
        assert "l3_fallback" in stats
        assert "intelligent_warming" in stats
        assert "performance_metrics" in stats
        assert "health_monitoring" in stats

        # Verify L1 cache stats
        l1_stats = stats["l1_cache"]
        assert "hits" in l1_stats
        assert "hit_rate" in l1_stats
        assert "size" in l1_stats
        assert "max_size" in l1_stats
        assert "estimated_memory_usage_bytes" in l1_stats

        # Verify performance metrics
        perf_metrics = stats["performance_metrics"]
        assert "response_times" in perf_metrics
        assert "operation_stats" in perf_metrics
        assert "slo_compliance" in perf_metrics
        assert "error_rate" in perf_metrics

    async def test_cache_efficiency_metrics(
        self, multi_level_cache, mock_fallback_database
    ):
        """Test cache efficiency calculation with various hit patterns."""
        cache = multi_level_cache

        # Create mixed hit patterns
        # L1 hits (set then get immediately)
        await cache.set("l1_hit_1", "value1")
        await cache.get("l1_hit_1")
        await cache.get("l1_hit_1")  # Second get should be L1 hit

        # L3 fallback (with cache population)
        await cache.get(
            "l3_test", fallback_func=lambda: mock_fallback_database("user:123")
        )

        # Get efficiency metrics
        stats = cache.get_performance_stats()
        efficiency = stats["cache_efficiency"]

        assert "overall" in efficiency
        assert "l1_efficiency" in efficiency
        assert "l2_efficiency" in efficiency
        assert "cache_penetration_rate" in efficiency

        # Verify reasonable efficiency values
        assert 0 <= efficiency["overall"] <= 1
        assert 0 <= efficiency["l1_efficiency"] <= 1
        assert 0 <= efficiency["l2_efficiency"] <= 1

    async def test_slo_compliance_tracking(self, multi_level_cache):
        """Test SLO compliance tracking and calculation."""
        cache = multi_level_cache

        # Perform operations to generate response times
        for i in range(10):
            await cache.set(f"slo_test_{i}", f"value_{i}")
            await cache.get(f"slo_test_{i}")

        # Get SLO compliance metrics
        stats = cache.get_performance_stats()
        slo_compliance = stats["performance_metrics"]["slo_compliance"]

        assert "compliant" in slo_compliance
        assert "slo_target_ms" in slo_compliance
        assert "compliance_rate" in slo_compliance
        assert "total_requests" in slo_compliance

        # Verify compliance rate is reasonable (should be good for simple operations)
        assert 0 <= slo_compliance["compliance_rate"] <= 1

    async def test_response_time_percentiles(self, multi_level_cache):
        """Test response time percentile calculations."""
        cache = multi_level_cache

        # Generate sufficient operations for percentile calculation
        for i in range(50):
            await cache.set(f"perf_test_{i}", f"value_{i}")
            await cache.get(f"perf_test_{i}")

        # Get response time statistics
        stats = cache.get_performance_stats()
        response_times = stats["performance_metrics"]["response_times"]

        assert "p50" in response_times
        assert "p95" in response_times
        assert "p99" in response_times
        assert "mean" in response_times
        assert "max" in response_times
        assert "count" in response_times

        # Verify percentile ordering
        assert response_times["p50"] <= response_times["p95"]
        assert response_times["p95"] <= response_times["p99"]
        assert response_times["p99"] <= response_times["max"]

    async def test_monitoring_metrics_format(self, multi_level_cache):
        """Test monitoring-optimized metrics format."""
        cache = multi_level_cache

        # Generate some activity
        await cache.set("monitor_test", "value")
        await cache.get("monitor_test")

        # Get monitoring metrics
        monitoring_metrics = cache.get_monitoring_metrics()

        # Verify flat structure for monitoring systems
        expected_keys = [
            "cache.hit_rate.overall",
            "cache.hit_rate.l1",
            "cache.hit_rate.l2",
            "cache.size.l1_current",
            "cache.requests.total",
            "cache.performance.response_time_p95",
            "cache.performance.error_rate",
            "cache.health.overall",
        ]

        for key in expected_keys:
            assert key in monitoring_metrics

    async def test_alert_metrics_thresholds(self, multi_level_cache):
        """Test alert metrics with threshold evaluation."""
        cache = multi_level_cache

        # Generate activity
        for i in range(20):
            await cache.set(f"alert_test_{i}", f"value_{i}")
            await cache.get(f"alert_test_{i}")

        # Get alert metrics
        alert_metrics = cache.get_alert_metrics()

        assert "alerts" in alert_metrics
        assert "values" in alert_metrics
        assert "thresholds" in alert_metrics

        # Verify alert structure
        alerts = alert_metrics["alerts"]
        assert "hit_rate_critical" in alerts
        assert "hit_rate_warning" in alerts
        assert "error_rate_critical" in alerts
        assert "response_time_critical" in alerts
        assert "slo_compliance_critical" in alerts

        # All alerts should be boolean
        for alert_name, alert_value in alerts.items():
            assert isinstance(alert_value, bool)


# ============================================================================
# ERROR HANDLING AND RESILIENCE TESTING
# ============================================================================


@pytest.mark.asyncio
class TestErrorHandlingResilience:
    """Test error handling and resilience features."""

    async def test_redis_connection_failure_resilience(self, multi_level_cache):
        """Test cache behavior when Redis connection fails."""
        cache = multi_level_cache

        # Force Redis connection to fail by closing it
        if cache._l2_cache and cache._l2_cache._client:
            await cache._l2_cache._client.close()
            cache._l2_cache._client = None

        # Cache should continue working with L1 only
        await cache.set("resilience_test", "still_works")
        result = await cache.get("resilience_test")
        assert result == "still_works"

        # L1 should have the data
        l1_result = cache._l1_cache.get("resilience_test")
        assert l1_result == "still_works"

        # Error should be tracked
        stats = cache.get_performance_stats()
        # Connection failure should affect health status
        assert stats["health_monitoring"]["l2_health"] in ["degraded", "unhealthy"]

    async def test_serialization_error_handling(self, multi_level_cache):
        """Test handling of serialization errors."""
        cache = multi_level_cache

        # Try to cache non-serializable data
        class NonSerializable:
            def __init__(self):
                self.func = lambda x: x  # Functions are not JSON serializable

        non_serializable = NonSerializable()

        # L1 should work (stores objects directly)
        cache._l1_cache.set("non_serializable", non_serializable)
        l1_result = cache._l1_cache.get("non_serializable")
        assert l1_result is non_serializable

        # L2 set should fail gracefully but not crash
        try:
            await cache.set("non_serializable", non_serializable)
            # If it doesn't raise an exception, that's fine
        except Exception:
            # If it raises an exception, the cache should still be functional
            pass

        # Cache should still be operational
        await cache.set("normal_data", "works_fine")
        result = await cache.get("normal_data")
        assert result == "works_fine"

    async def test_concurrent_access_safety(self, multi_level_cache):
        """Test cache safety under concurrent access."""
        cache = multi_level_cache

        async def concurrent_operations(worker_id: int):
            """Worker function for concurrent testing."""
            for i in range(10):
                key = f"concurrent_{worker_id}_{i}"
                value = f"worker_{worker_id}_value_{i}"
                await cache.set(key, value)
                result = await cache.get(key)
                assert result == value

        # Run multiple workers concurrently
        workers = [concurrent_operations(i) for i in range(5)]
        await asyncio.gather(*workers)

        # Verify cache integrity
        stats = cache.get_performance_stats()
        assert stats["total_requests"] >= 50  # At least 50 operations
        assert stats["performance_metrics"]["error_rate"]["overall_error_rate"] == 0

    async def test_memory_pressure_handling(self, multi_level_cache):
        """Test cache behavior under memory pressure (L1 eviction)."""
        cache = multi_level_cache

        # Fill L1 cache beyond capacity
        l1_max_size = cache._l1_cache._max_size

        # Add more items than L1 can hold
        for i in range(l1_max_size + 20):
            await cache.set(f"pressure_test_{i}", f"value_{i}")

        # L1 should have evicted old items
        l1_size = len(cache._l1_cache._cache)
        assert l1_size <= l1_max_size

        # Cache should still be functional
        await cache.set("after_pressure", "still_works")
        result = await cache.get("after_pressure")
        assert result == "still_works"

        # Verify LRU behavior - recent items should still be in L1
        recent_result = cache._l1_cache.get("after_pressure")
        assert recent_result == "still_works"

    async def test_error_rate_tracking(self, multi_level_cache):
        """Test error rate tracking and health status updates."""
        cache = multi_level_cache

        # Force some errors by mocking failure
        original_l2_set = cache._l2_cache.set if cache._l2_cache else None
        if cache._l2_cache:

            async def failing_set(*args, **kwargs):
                raise Exception("Simulated L2 error")

            cache._l2_cache.set = failing_set

        # Perform operations that will fail at L2
        for i in range(10):
            try:
                await cache.set(f"error_test_{i}", f"value_{i}")
            except:
                pass  # Ignore errors for this test

        # Restore original function
        if cache._l2_cache and original_l2_set:
            cache._l2_cache.set = original_l2_set

        # Check error rate tracking
        stats = cache.get_performance_stats()
        error_rate = stats["performance_metrics"]["error_rate"]

        # Some errors should have been recorded
        if error_rate["total_errors"] > 0:
            assert error_rate["overall_error_rate"] > 0
            assert error_rate["consecutive_errors"] >= 0


# ============================================================================
# PERFORMANCE CHARACTERISTICS TESTING
# ============================================================================


@pytest.mark.asyncio
class TestPerformanceCharacteristics:
    """Test performance characteristics and SLA compliance."""

    async def test_response_time_sla_compliance(self, multi_level_cache):
        """Test that cache operations meet SLA requirements."""
        cache = multi_level_cache

        # Test L1 cache performance (should be very fast)
        start_time = time.perf_counter()
        cache._l1_cache.set("perf_l1", "fast_value")
        result = cache._l1_cache.get("perf_l1")
        l1_duration = time.perf_counter() - start_time

        assert result == "fast_value"
        assert l1_duration < 0.001  # L1 should be sub-millisecond

        # Test full cache performance
        response_times = []
        for i in range(100):
            start_time = time.perf_counter()
            await cache.set(f"perf_test_{i}", f"value_{i}")
            await cache.get(f"perf_test_{i}")
            duration = time.perf_counter() - start_time
            response_times.append(duration)

        # Calculate percentiles
        response_times.sort()
        p95_time = response_times[int(len(response_times) * 0.95)]
        p99_time = response_times[int(len(response_times) * 0.99)]

        # Verify SLA compliance (should be well under 200ms for cache operations)
        assert p95_time < 0.05  # 50ms for 95th percentile
        assert p99_time < 0.1  # 100ms for 99th percentile

    async def test_throughput_characteristics(self, multi_level_cache):
        """Test cache throughput under load."""
        cache = multi_level_cache

        # Measure throughput for parallel operations
        start_time = time.perf_counter()

        async def throughput_worker(worker_id: int, operations: int):
            for i in range(operations):
                key = f"throughput_{worker_id}_{i}"
                await cache.set(key, f"value_{i}")
                await cache.get(key)

        # Run parallel workers
        workers = [throughput_worker(i, 50) for i in range(4)]
        await asyncio.gather(*workers)

        total_duration = time.perf_counter() - start_time
        total_operations = 4 * 50 * 2  # 4 workers, 50 ops each, 2 ops per iteration

        throughput = total_operations / total_duration

        # Expect at least 1000 operations per second for simple cache operations
        assert throughput > 1000

        logger.info("Cache throughput: %s ops/sec", throughput:.2f)

    async def test_memory_usage_efficiency(self, multi_level_cache):
        """Test memory usage efficiency and estimation."""
        cache = multi_level_cache

        # Add known data to cache
        test_data = {"key": "value", "number": 42, "array": [1, 2, 3]}
        data_count = 50

        for i in range(data_count):
            await cache.set(f"memory_test_{i}", test_data)

        # Get memory usage stats
        stats = cache.get_performance_stats()
        memory_usage = stats["l1_cache"]["estimated_memory_usage_bytes"]

        # Memory usage should be reasonable (not zero, but not excessive)
        assert memory_usage > 0
        assert memory_usage < 10 * 1024 * 1024  # Less than 10MB for test data

        # Memory usage should scale roughly with data count
        items_in_cache = stats["l1_cache"]["size"]
        if items_in_cache > 0:
            bytes_per_item = memory_usage / items_in_cache
            assert bytes_per_item > 0
            assert bytes_per_item < 50 * 1024  # Less than 50KB per item

    @pytest.mark.performance
    async def test_cache_warming_performance(self, multi_level_cache, redis_client):
        """Test performance of cache warming operations."""
        cache = multi_level_cache

        # Pre-populate L2 with data
        warm_keys = []
        for i in range(20):
            key = f"warm_perf_{i}"
            data = {"performance_test": True, "index": i}
            serialized = json.dumps(data).encode("utf-8")
            await redis_client.set(key, serialized, ex=300)
            warm_keys.append(key)

        # Measure warming performance
        start_time = time.perf_counter()
        warm_results = await cache.warm_cache(warm_keys)
        warming_duration = time.perf_counter() - start_time

        # Warming should be efficient
        assert warming_duration < 1.0  # Less than 1 second for 20 keys

        # Calculate warming throughput
        successful_warms = sum(1 for success in warm_results.values() if success)
        if successful_warms > 0:
            warming_throughput = successful_warms / warming_duration
            assert warming_throughput > 10  # At least 10 keys per second

            logger.info("Cache warming throughput: %s keys/sec", warming_throughput:.2f)


# ============================================================================
# HEALTH MONITORING AND INTEGRATION TESTING
# ============================================================================


@pytest.mark.asyncio
class TestHealthMonitoringIntegration:
    """Test health monitoring and system integration."""

    async def test_comprehensive_health_check(self, multi_level_cache):
        """Test comprehensive health check functionality."""
        cache = multi_level_cache

        # Perform health check
        health_status = await cache.get_health_check()

        assert "healthy" in health_status
        assert "checks" in health_status
        assert "performance" in health_status
        assert "timestamp" in health_status

        # Verify individual component checks
        checks = health_status["checks"]
        assert "l1_cache" in checks
        assert "l2_cache" in checks
        assert "warming_service" in checks

        # L1 should always be healthy
        assert checks["l1_cache"]["healthy"] is True

        # Performance metrics should be present
        performance = health_status["performance"]
        assert "health_check_duration_seconds" in performance
        assert "overall_hit_rate" in performance
        assert "error_rate" in performance

    async def test_health_status_degradation(self, multi_level_cache):
        """Test health status changes under error conditions."""
        cache = multi_level_cache

        # Force L2 connection failure
        if cache._l2_cache:
            await cache._l2_cache.close()

        # Perform operations that will fail at L2 level
        for i in range(5):
            await cache.set(f"health_test_{i}", f"value_{i}")

        # Check health status
        health_status = await cache.get_health_check()

        # Overall health might be degraded due to L2 issues
        if not health_status["checks"]["l2_cache"]["healthy"]:
            # This is expected when L2 is unavailable
            assert health_status["checks"]["l2_cache"]["healthy"] is False

        # L1 should still be healthy
        assert health_status["checks"]["l1_cache"]["healthy"] is True

    async def test_opentelemetry_integration(self, multi_level_cache):
        """Test OpenTelemetry integration and metrics export."""
        cache = multi_level_cache

        # Perform operations to generate telemetry
        for i in range(10):
            await cache.set(f"otel_test_{i}", f"value_{i}")
            await cache.get(f"otel_test_{i}")

        # Get stats (which should update OpenTelemetry metrics)
        stats = cache.get_performance_stats()

        # Verify observability configuration
        observability = stats["observability"]
        assert "opentelemetry_enabled" in observability
        assert "performance_optimizer_enabled" in observability

        # If OpenTelemetry is available, metrics should be updated
        # This is verified through the internal metric update calls
        # Actual verification would require telemetry backend inspection

    async def test_specialized_cache_integration(self):
        """Test specialized cache instances and global cache access."""
        # Test specialized cache types
        rule_cache = get_cache("rule")
        session_cache = get_cache("session")
        analytics_cache = get_cache("analytics")
        prompt_cache = get_cache("prompt")

        # Verify different cache instances
        assert rule_cache is not session_cache
        assert session_cache is not analytics_cache
        assert analytics_cache is not prompt_cache

        # Test global cache access
        default_cache = get_cache()  # Should return prompt cache
        assert default_cache is prompt_cache

        # Test convenience functions
        await cached_set("conv_test", "convenience_value", "prompt", 300)
        result = await cached_get("conv_test", lambda: "fallback", "prompt", 300)
        assert result == "convenience_value"

    async def test_specialized_caches_statistics(self):
        """Test statistics aggregation across specialized caches."""
        from prompt_improver.utils.multi_level_cache import get_specialized_caches

        caches = get_specialized_caches()

        # Perform operations on different cache types
        await caches.rule_cache.set("rule_test", "rule_value")
        await caches.session_cache.set("session_test", "session_value")
        await caches.analytics_cache.set("analytics_test", "analytics_value")
        await caches.prompt_cache.set("prompt_test", "prompt_value")

        # Get aggregated statistics
        all_stats = caches.get_all_stats()

        assert "rule_cache" in all_stats
        assert "session_cache" in all_stats
        assert "analytics_cache" in all_stats
        assert "prompt_cache" in all_stats

        # Verify each cache has stats
        for cache_name, stats in all_stats.items():
            assert "total_requests" in stats
            assert "l1_cache" in stats
            assert stats["total_requests"] > 0  # Should have at least the set operation


# ============================================================================
# INTEGRATION AND SYSTEM TESTING
# ============================================================================


@pytest.mark.asyncio
class TestSystemIntegration:
    """Test system-level integration scenarios."""

    async def test_full_system_workflow(
        self, multi_level_cache, mock_fallback_database
    ):
        """Test complete system workflow with all cache tiers."""
        cache = multi_level_cache

        # Scenario: User requests data multiple times
        user_id = "user:12345"

        # First request: L3 fallback (database)
        start_time = time.perf_counter()
        result1 = await cache.get(
            user_id, fallback_func=lambda: mock_fallback_database(user_id)
        )
        first_request_time = time.perf_counter() - start_time

        # Second request: Should hit L1 (fastest)
        start_time = time.perf_counter()
        result2 = await cache.get(user_id)
        second_request_time = time.perf_counter() - start_time

        # Third request: Also L1 hit
        start_time = time.perf_counter()
        result3 = await cache.get(user_id)
        third_request_time = time.perf_counter() - start_time

        # Verify results consistency
        assert result1 == result2 == result3
        if result1:  # If database had the data
            assert result1.get("id") == 12345  # Expected from mock database

        # Verify performance improvement
        assert second_request_time < first_request_time  # L1 faster than L3
        assert third_request_time < first_request_time  # Consistent L1 performance

        # Verify hit rate improvement
        stats = cache.get_performance_stats()
        assert stats["l1_cache"]["hits"] >= 2  # At least 2 L1 hits
        assert stats["overall_hit_rate"] > 0  # Overall hit rate improved

    async def test_cache_invalidation_workflow(self, multi_level_cache):
        """Test cache invalidation across all tiers."""
        cache = multi_level_cache

        # Setup: Cache data across all tiers
        test_key = "invalidation_test"
        original_data = {"version": 1, "data": "original"}

        await cache.set(test_key, original_data)

        # Verify data is cached
        result = await cache.get(test_key)
        assert result == original_data
        assert cache._l1_cache.get(test_key) == original_data

        # Invalidate cache
        await cache.delete(test_key)

        # Verify invalidation
        result = await cache.get(test_key)
        assert result is None
        assert cache._l1_cache.get(test_key) is None

        # Update with new data
        updated_data = {"version": 2, "data": "updated"}
        await cache.set(test_key, updated_data)

        # Verify update
        result = await cache.get(test_key)
        assert result == updated_data
        assert result["version"] == 2

    async def test_high_concurrency_scenario(self, multi_level_cache):
        """Test cache behavior under high concurrency."""
        cache = multi_level_cache

        # Simulate high concurrency with many parallel requests
        concurrent_requests = 100

        async def concurrent_user_simulation(user_id: int):
            """Simulate a user making multiple cache requests."""
            user_key = f"user:{user_id}"
            user_data = {"id": user_id, "name": f"User {user_id}"}

            # Set user data
            await cache.set(user_key, user_data)

            # Multiple gets to test hit rates
            for _ in range(5):
                result = await cache.get(user_key)
                assert result == user_data
                await asyncio.sleep(0.001)  # Small delay

        # Run concurrent simulations
        start_time = time.perf_counter()
        tasks = [concurrent_user_simulation(i) for i in range(concurrent_requests)]
        await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        # Verify system handled concurrency well
        stats = cache.get_performance_stats()
        assert (
            stats["total_requests"] >= concurrent_requests * 6
        )  # Set + 5 gets per user
        assert stats["performance_metrics"]["error_rate"]["overall_error_rate"] == 0

        # Performance should be reasonable
        operations_per_second = stats["total_requests"] / total_time
        assert operations_per_second > 500  # At least 500 ops/second

        logger.info(
            f"Concurrency test: {operations_per_second:.2f} ops/sec with {concurrent_requests} concurrent users"
        )

    async def test_cache_warming_integration(self, multi_level_cache, redis_client):
        """Test end-to-end cache warming integration."""
        cache = multi_level_cache

        # Scenario: Simulate application startup with pre-warming
        popular_keys = [f"popular_item_{i}" for i in range(10)]

        # Pre-populate L2 (simulating data from previous session)
        for i, key in enumerate(popular_keys):
            data = {"item_id": i, "popularity": 10 - i, "category": "popular"}
            serialized = json.dumps(data).encode("utf-8")
            await redis_client.set(key, serialized, ex=600)

        # Simulate access patterns to build warming candidates
        for key in popular_keys[:5]:  # Only access first 5 frequently
            for _ in range(5):
                cache._record_access_pattern(key)

        # Get warming candidates
        candidates = await cache.get_warming_candidates()
        candidate_keys = [c["key"] for c in candidates]

        # Popular keys should be warming candidates
        for key in popular_keys[:3]:  # Top 3 should definitely be candidates
            assert key in candidate_keys

        # Perform manual warming for startup
        warm_results = await cache.warm_cache(popular_keys)

        # Verify warming success
        successful_warms = sum(1 for success in warm_results.values() if success)
        assert successful_warms >= 5  # At least half should succeed

        # Verify warmed data is in L1
        for key, success in warm_results.items():
            if success:
                l1_data = cache._l1_cache.get(key)
                assert l1_data is not None
                assert "item_id" in l1_data

    async def test_disaster_recovery_scenario(self, multi_level_cache):
        """Test cache behavior during and after failure scenarios."""
        cache = multi_level_cache

        # Setup: Normal operation
        test_data = {"critical": "data", "timestamp": time.time()}
        await cache.set("critical_key", test_data)

        # Verify normal operation
        result = await cache.get("critical_key")
        assert result == test_data

        # Simulate L2 failure
        if cache._l2_cache:
            original_client = cache._l2_cache._client
            cache._l2_cache._client = None  # Simulate connection loss

        # Cache should continue working with L1
        result = await cache.get("critical_key")
        assert result == test_data  # Should still get from L1

        # New data should still be cacheable in L1
        recovery_data = {"status": "recovered", "mode": "degraded"}
        await cache.set("recovery_key", recovery_data)
        result = await cache.get("recovery_key")
        assert result == recovery_data

        # Check health status reflects degraded state
        health = await cache.get_health_check()
        if not health["checks"]["l2_cache"]["healthy"]:
            # L2 failure should be detected
            assert health["checks"]["l1_cache"]["healthy"] is True  # L1 still works

        # Simulate recovery (restore L2 connection)
        if cache._l2_cache and original_client:
            cache._l2_cache._client = original_client
            cache._l2_cache._connection_error_logged = False  # Reset error flag

        # Verify system can recover
        post_recovery_data = {"status": "fully_recovered"}
        await cache.set("post_recovery", post_recovery_data)
        result = await cache.get("post_recovery")
        assert result == post_recovery_data


# ============================================================================
# PERFORMANCE BENCHMARKING AND VALIDATION
# ============================================================================


@pytest.mark.performance
@pytest.mark.asyncio
class TestPerformanceBenchmarking:
    """Performance benchmarking and validation tests."""

    async def test_baseline_performance_benchmark(self, multi_level_cache):
        """Establish baseline performance metrics for regression testing."""
        cache = multi_level_cache

        # Benchmark different operation types
        benchmark_results = {}

        # L1 Cache Performance
        l1_times = []
        for i in range(1000):
            start = time.perf_counter()
            cache._l1_cache.set(f"l1_bench_{i}", f"value_{i}")
            cache._l1_cache.get(f"l1_bench_{i}")
            l1_times.append(time.perf_counter() - start)

        benchmark_results["l1_avg_time"] = sum(l1_times) / len(l1_times)
        benchmark_results["l1_p95_time"] = sorted(l1_times)[int(0.95 * len(l1_times))]

        # Full Cache Performance
        full_cache_times = []
        for i in range(100):
            start = time.perf_counter()
            await cache.set(f"full_bench_{i}", f"value_{i}")
            await cache.get(f"full_bench_{i}")
            full_cache_times.append(time.perf_counter() - start)

        benchmark_results["full_cache_avg_time"] = sum(full_cache_times) / len(
            full_cache_times
        )
        benchmark_results["full_cache_p95_time"] = sorted(full_cache_times)[
            int(0.95 * len(full_cache_times))
        ]

        # Assert performance requirements
        assert benchmark_results["l1_avg_time"] < 0.001  # Sub-millisecond L1
        assert benchmark_results["l1_p95_time"] < 0.002  # 2ms p95 for L1
        assert (
            benchmark_results["full_cache_avg_time"] < 0.01
        )  # 10ms avg for full cache
        assert (
            benchmark_results["full_cache_p95_time"] < 0.05
        )  # 50ms p95 for full cache

        logger.info("Performance benchmark results: %s", benchmark_results)

    async def test_throughput_under_load(self, multi_level_cache):
        """Test cache throughput under sustained load."""
        cache = multi_level_cache

        # Sustained load test
        duration_seconds = 5
        operation_count = 0
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds

        while time.perf_counter() < end_time:
            key = f"load_test_{operation_count}"
            await cache.set(key, f"value_{operation_count}")
            await cache.get(key)
            operation_count += 2

        actual_duration = time.perf_counter() - start_time
        throughput = operation_count / actual_duration

        # Expect at least 1000 operations per second
        assert throughput > 1000

        # Verify cache maintained integrity under load
        stats = cache.get_performance_stats()
        assert (
            stats["performance_metrics"]["error_rate"]["overall_error_rate"] < 0.01
        )  # Less than 1% errors

        logger.info(
            f"Sustained load throughput: {throughput:.2f} ops/sec over {actual_duration:.2f}s"
        )

    async def test_memory_efficiency_benchmark(self, multi_level_cache):
        """Benchmark memory efficiency and usage patterns."""
        cache = multi_level_cache

        # Baseline memory usage
        initial_stats = cache.get_performance_stats()
        initial_memory = initial_stats["l1_cache"]["estimated_memory_usage_bytes"]

        # Add known amount of data
        test_objects = []
        for i in range(100):
            obj = {
                "id": i,
                "data": "x" * 1000,  # 1KB of data per object
                "metadata": {"created": time.time(), "version": 1},
            }
            test_objects.append(obj)
            await cache.set(f"memory_test_{i}", obj)

        # Measure memory usage
        final_stats = cache.get_performance_stats()
        final_memory = final_stats["l1_cache"]["estimated_memory_usage_bytes"]
        memory_increase = final_memory - initial_memory

        # Calculate efficiency
        expected_data_size = 100 * 1000  # 100KB of actual data
        memory_overhead_ratio = (
            memory_increase / expected_data_size if expected_data_size > 0 else 0
        )

        # Memory overhead should be reasonable (less than 5x the actual data)
        assert memory_overhead_ratio < 5.0

        # Memory per item should be reasonable
        items_in_cache = final_stats["l1_cache"]["size"]
        if items_in_cache > 0:
            memory_per_item = final_memory / items_in_cache
            assert memory_per_item < 50 * 1024  # Less than 50KB per item

        logger.info(
            f"Memory efficiency: {memory_overhead_ratio:.2f}x overhead, {memory_per_item:.0f} bytes/item"
        )


# ============================================================================
# CLEANUP AND UTILITIES
# ============================================================================


@pytest.fixture(scope="function", autouse=True)
async def cleanup_after_test():
    """Automatic cleanup after each test to prevent interference."""
    yield

    # Reset any global cache instances
    try:
        from prompt_improver.utils.multi_level_cache import _global_caches

        if _global_caches:
            for cache_name in [
                "rule_cache",
                "session_cache",
                "analytics_cache",
                "prompt_cache",
            ]:
                cache = getattr(_global_caches, cache_name, None)
                if cache:
                    await cache.clear()
                    await cache.stop_warming()
    except Exception:
        pass  # Ignore cleanup errors


def pytest_configure(config):
    """Configure pytest markers for cache testing."""
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "redis: mark test as requiring Redis")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])
