"""Real behavior tests for L1 Memory Cache.

Tests LRU cache, access patterns, and TTL behavior with:
- Real memory usage patterns under load
- Thread safety and concurrent access  
- Cache entry lifecycle and eviction
- Access pattern tracking accuracy
- Performance benchmarks (sub-millisecond access)
- Memory efficiency validation

No mocks - tests actual in-memory cache behavior.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from prompt_improver.database.services.cache.memory_cache import (
    AccessPattern,
    CacheEntry, 
    CacheEventType,
    CacheMetrics,
    EvictionPolicy,
    MemoryCache,
)


class TestCacheEntry:
    """Test CacheEntry functionality."""
    
    def test_cache_entry_creation(self):
        """Test basic cache entry creation."""
        now = datetime.now(UTC)
        entry = CacheEntry(
            value="test_value",
            created_at=now,
            last_accessed=now,
            ttl_seconds=300
        )
        
        assert entry.value == "test_value"
        assert entry.created_at == now
        assert entry.last_accessed == now
        assert entry.access_count == 0
        assert entry.ttl_seconds == 300
        assert entry.size_bytes > 0  # Auto-calculated
        assert isinstance(entry.tags, set)
    
    def test_cache_entry_expiration(self):
        """Test cache entry expiration logic."""
        now = datetime.now(UTC)
        
        # Entry with no TTL never expires
        entry_no_ttl = CacheEntry(
            value="test",
            created_at=now,
            last_accessed=now
        )
        assert not entry_no_ttl.is_expired()
        
        # Entry with future expiry not expired
        entry_future = CacheEntry(
            value="test",
            created_at=now,
            last_accessed=now,
            ttl_seconds=300
        )
        assert not entry_future.is_expired()
        
        # Entry with past expiry is expired
        past_time = now - timedelta(minutes=10)
        entry_expired = CacheEntry(
            value="test",
            created_at=past_time,
            last_accessed=past_time,
            ttl_seconds=300  # 5 minutes, but created 10 minutes ago
        )
        assert entry_expired.is_expired()
    
    def test_cache_entry_touch(self):
        """Test cache entry access tracking."""
        now = datetime.now(UTC)
        entry = CacheEntry(
            value="test",
            created_at=now,
            last_accessed=now
        )
        
        initial_count = entry.access_count
        initial_time = entry.last_accessed
        
        # Small delay to ensure timestamp change
        time.sleep(0.001)
        entry.touch()
        
        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > initial_time
    
    def test_cache_entry_time_until_expiry(self):
        """Test time until expiry calculation."""
        now = datetime.now(UTC)
        
        # Entry with no TTL
        entry_no_ttl = CacheEntry(
            value="test",
            created_at=now,
            last_accessed=now
        )
        assert entry_no_ttl.time_until_expiry() is None
        
        # Entry with TTL
        entry_with_ttl = CacheEntry(
            value="test",
            created_at=now,
            last_accessed=now,
            ttl_seconds=300
        )
        time_left = entry_with_ttl.time_until_expiry()
        assert time_left is not None
        assert 299 <= time_left <= 300  # Should be close to 300 seconds
    
    def test_cache_entry_size_calculation(self):
        """Test cache entry size estimation."""
        # String entry
        string_entry = CacheEntry(
            value="hello world",
            created_at=datetime.now(UTC),
            last_accessed=datetime.now(UTC)
        )
        assert string_entry.size_bytes > 0
        
        # Dictionary entry
        dict_entry = CacheEntry(
            value={"key1": "value1", "key2": "value2"},
            created_at=datetime.now(UTC),
            last_accessed=datetime.now(UTC)
        )
        assert dict_entry.size_bytes > 0
        assert dict_entry.size_bytes > string_entry.size_bytes  # Dict should be larger
        
        # Large data entry
        large_data = "x" * 10000
        large_entry = CacheEntry(
            value=large_data,
            created_at=datetime.now(UTC),
            last_accessed=datetime.now(UTC)
        )
        assert large_entry.size_bytes > 10000


class TestAccessPattern:
    """Test AccessPattern functionality."""
    
    def test_access_pattern_creation(self):
        """Test access pattern creation."""
        pattern = AccessPattern(key="test_key")
        
        assert pattern.key == "test_key"
        assert pattern.access_count == 0
        assert pattern.last_access is None
        assert pattern.access_frequency == 0.0
        assert pattern.warming_priority == 0.0
        assert len(pattern.access_times) == 0
        assert len(pattern.access_intervals) == 0
        assert pattern.predicted_next_access is None
    
    def test_access_pattern_recording(self):
        """Test access pattern recording and calculations."""
        pattern = AccessPattern(key="test_key")
        
        initial_count = pattern.access_count
        initial_time = pattern.last_access
        
        # Record first access
        time.sleep(0.001)  # Ensure timestamp difference
        pattern.record_access()
        
        assert pattern.access_count == initial_count + 1
        assert pattern.last_access is not None
        assert pattern.last_access != initial_time  # Should have changed from None
        assert len(pattern.access_times) == 1
        assert len(pattern.access_intervals) == 0  # No intervals yet on first access
        
        # Record second access after delay
        time.sleep(0.002)
        pattern.record_access()
        
        assert pattern.access_count == initial_count + 2
        assert len(pattern.access_times) == 2
        assert len(pattern.access_intervals) == 1
        assert pattern.access_frequency > 0
        assert pattern.warming_priority > 0
    
    def test_access_pattern_frequency_calculation(self):
        """Test access frequency calculation over time."""
        pattern = AccessPattern(key="test_key")
        
        # Record multiple accesses with known intervals
        for i in range(5):
            pattern.record_access()
            time.sleep(0.001)  # Small consistent interval
        
        assert pattern.access_frequency > 0
        assert pattern.warming_priority > 0
        assert len(pattern.access_intervals) == 4  # N-1 intervals for N accesses
    
    def test_access_pattern_prediction(self):
        """Test access pattern prediction."""
        pattern = AccessPattern(key="test_key")
        
        # Need at least 3 intervals for prediction
        for i in range(4):
            pattern.record_access()
            time.sleep(0.001)
        
        # Should have prediction after enough data points
        assert pattern.predicted_next_access is not None
        assert isinstance(pattern.predicted_next_access, datetime)
    
    def test_access_pattern_should_warm(self):
        """Test warming recommendation logic."""
        pattern = AccessPattern(key="test_key")
        
        # New pattern shouldn't recommend warming
        assert not pattern.should_warm()
        
        # Pattern with frequent access should recommend warming
        for i in range(10):
            pattern.record_access()
            time.sleep(0.001)
        
        # High frequency should trigger warming recommendation
        assert pattern.should_warm(threshold=0.1)


class TestCacheMetrics:
    """Test CacheMetrics functionality."""
    
    def test_cache_metrics_creation(self):
        """Test cache metrics initialization."""
        metrics = CacheMetrics("test_cache")
        
        assert metrics.service_name == "test_cache"
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.evictions == 0
        assert metrics.expirations == 0
        assert metrics.sets == 0
        assert metrics.deletes == 0
        assert metrics.clears == 0
        assert len(metrics.response_times) == 0
        assert len(metrics.size_history) == 0
    
    def test_cache_metrics_recording(self):
        """Test metrics recording."""
        metrics = CacheMetrics("test_cache")
        
        # Record various operations
        metrics.record_operation(CacheEventType.HIT, 1.5)
        metrics.record_operation(CacheEventType.MISS, 0.8)
        metrics.record_operation(CacheEventType.SET, 2.0)
        metrics.record_operation(CacheEventType.EVICTION)
        
        assert metrics.hits == 1
        assert metrics.misses == 1  
        assert metrics.sets == 1
        assert metrics.evictions == 1
        assert len(metrics.response_times) == 3  # Only operations with duration
    
    def test_cache_metrics_stats(self):
        """Test comprehensive metrics statistics."""
        metrics = CacheMetrics("test_cache")
        
        # Record some operations
        for i in range(10):
            metrics.record_operation(CacheEventType.HIT, i + 1.0)
        for i in range(3):
            metrics.record_operation(CacheEventType.MISS, i + 1.0)
        
        stats = metrics.get_stats()
        
        assert stats["service"] == "test_cache"
        assert "operations" in stats
        assert "performance" in stats
        assert "memory" in stats
        
        assert stats["operations"]["hits"] == 10
        assert stats["operations"]["misses"] == 3
        assert stats["operations"]["total"] == 13
        assert stats["performance"]["hit_rate"] > 0.7  # 10/13


class TestMemoryCache:
    """Test MemoryCache core functionality."""
    
    def test_memory_cache_creation(self):
        """Test memory cache initialization."""
        cache = MemoryCache(
            max_size=100,
            eviction_policy=EvictionPolicy.LRU,
            enable_metrics=True
        )
        
        assert cache._max_size == 100
        assert cache._eviction_policy == EvictionPolicy.LRU
        assert cache.metrics is not None
        assert len(cache._cache) == 0
        assert cache._current_memory_bytes == 0
    
    def test_basic_cache_operations(self):
        """Test basic get/set/delete operations."""
        cache = MemoryCache(max_size=10)
        
        # Test set and get
        result = cache.set("key1", "value1")
        assert result is True
        
        value = cache.get("key1")
        assert value == "value1"
        
        # Test cache hit metrics
        assert cache.metrics.hits == 1
        assert cache.metrics.sets == 1
        
        # Test non-existent key
        value = cache.get("nonexistent")
        assert value is None
        assert cache.metrics.misses == 1
        
        # Test delete
        deleted = cache.delete("key1")
        assert deleted is True
        
        # Verify deletion
        value = cache.get("key1")
        assert value is None
        assert cache.metrics.misses == 2
    
    def test_cache_ttl_behavior(self):
        """Test TTL expiration behavior."""
        cache = MemoryCache(max_size=10)
        
        # Set with short TTL
        cache.set("temp_key", "temp_value", ttl_seconds=1)
        
        # Should be available immediately
        value = cache.get("temp_key")
        assert value == "temp_value"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired now
        value = cache.get("temp_key")
        assert value is None
        assert cache.metrics.expirations > 0
    
    def test_cache_size_limit_eviction(self):
        """Test eviction when size limit is reached."""
        cache = MemoryCache(max_size=3, eviction_policy=EvictionPolicy.LRU)
        
        # Fill cache to capacity
        cache.set("key1", "value1")
        cache.set("key2", "value2") 
        cache.set("key3", "value3")
        
        assert len(cache._cache) == 3
        
        # Add one more - should evict LRU
        cache.set("key4", "value4")
        
        assert len(cache._cache) == 3
        assert cache.get("key1") is None  # Should be evicted (LRU)
        assert cache.get("key4") == "value4"  # Should be present
        assert cache.metrics.evictions > 0
    
    def test_lru_eviction_policy(self):
        """Test LRU eviction policy behavior."""
        cache = MemoryCache(max_size=3, eviction_policy=EvictionPolicy.LRU)
        
        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new key - should evict key2 (least recently used)
        cache.set("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Recently accessed
        assert cache.get("key2") is None  # Should be evicted
        assert cache.get("key3") == "value3"  # Should still be there
        assert cache.get("key4") == "value4"  # Newly added
    
    def test_lfu_eviction_policy(self):
        """Test LFU eviction policy behavior."""
        cache = MemoryCache(max_size=3, eviction_policy=EvictionPolicy.LFU)
        
        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Access key1 multiple times to increase frequency
        for _ in range(5):
            cache.get("key1")
        
        # Access key3 once
        cache.get("key3")
        
        # key2 has 0 accesses (least frequent)
        # Add new key - should evict key2
        cache.set("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Most frequent
        assert cache.get("key2") is None  # Should be evicted (LFU)
        assert cache.get("key3") == "value3"  # Accessed once
        assert cache.get("key4") == "value4"  # Newly added
    
    def test_memory_limit_eviction(self):
        """Test eviction based on memory limits."""
        # Small memory limit to trigger memory-based eviction
        cache = MemoryCache(max_size=1000, max_memory_bytes=1024)
        
        # Add large values to exceed memory limit
        large_value = "x" * 500
        
        cache.set("key1", large_value)
        cache.set("key2", large_value)
        cache.set("key3", large_value)  # Should trigger eviction
        
        # At least one eviction should have occurred
        assert cache.metrics.evictions > 0
        assert cache._current_memory_bytes <= cache._max_memory_bytes
    
    def test_cache_exists_method(self):
        """Test cache exists method."""
        cache = MemoryCache(max_size=10)
        
        # Non-existent key
        assert not cache.exists("nonexistent")
        
        # Add key
        cache.set("key1", "value1")
        assert cache.exists("key1")
        
        # Add key with TTL
        cache.set("temp_key", "temp_value", ttl_seconds=1)
        assert cache.exists("temp_key")
        
        # Wait for expiration
        time.sleep(1.1)
        assert not cache.exists("temp_key")  # Should be expired and removed
    
    def test_cache_clear_method(self):
        """Test cache clear method."""
        cache = MemoryCache(max_size=10)
        
        # Add some entries
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        assert len(cache._cache) == 3
        assert cache._current_memory_bytes > 0
        
        # Clear cache
        cache.clear()
        
        assert len(cache._cache) == 0
        assert cache._current_memory_bytes == 0
        assert cache.metrics.clears == 1
    
    def test_tag_based_operations(self):
        """Test tag-based cache operations."""
        cache = MemoryCache(max_size=10)
        
        # Set entries with tags
        cache.set("user:1", "user1_data", tags={"user", "profile"})
        cache.set("user:2", "user2_data", tags={"user", "profile"})
        cache.set("session:1", "session1_data", tags={"session"})
        cache.set("config:1", "config1_data", tags={"config", "system"})
        
        # Get by tags
        user_entries = cache.get_by_tags({"user"})
        assert len(user_entries) == 2
        assert "user:1" in user_entries
        assert "user:2" in user_entries
        
        system_entries = cache.get_by_tags({"system"})
        assert len(system_entries) == 1
        assert "config:1" in system_entries
        
        # Delete by tags
        deleted_count = cache.delete_by_tags({"user"})
        assert deleted_count == 2
        
        # Verify deletion
        assert cache.get("user:1") is None
        assert cache.get("user:2") is None
        assert cache.get("session:1") == "session1_data"  # Should remain
    
    def test_access_pattern_tracking(self):
        """Test access pattern tracking."""
        cache = MemoryCache(max_size=10)
        
        # Set and access keys
        cache.set("pattern_key", "pattern_value")
        
        # Access multiple times
        for i in range(5):
            cache.get("pattern_key")
            time.sleep(0.001)
        
        # Get access patterns
        patterns = cache.get_access_patterns()
        assert "pattern_key" in patterns
        
        pattern = patterns["pattern_key"]
        assert pattern.access_count >= 5  # Including set operation
        assert pattern.access_frequency > 0
        assert pattern.warming_priority > 0
    
    def test_cache_statistics(self):
        """Test comprehensive cache statistics."""
        cache = MemoryCache(max_size=10)
        
        # Perform various operations
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")
        cache.get("nonexistent")
        cache.delete("key2")
        
        stats = cache.get_stats()
        
        # Verify basic stats
        assert "size" in stats
        assert "max_size" in stats
        assert "memory_bytes" in stats
        assert "max_memory_bytes" in stats
        assert "utilization" in stats
        assert "memory_utilization" in stats
        assert "eviction_policy" in stats
        
        # Verify metrics integration
        assert "operations" in stats
        assert "performance" in stats
        
        assert stats["max_size"] == 10
        assert stats["eviction_policy"] == "lru"


class TestMemoryCacheThreadSafety:
    """Test thread safety of MemoryCache."""
    
    def test_concurrent_access(self):
        """Test concurrent cache access from multiple threads."""
        cache = MemoryCache(max_size=1000)
        
        def worker(thread_id: int, operations: int):
            """Worker function for concurrent testing."""
            for i in range(operations):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"
                
                # Set value
                cache.set(key, value)
                
                # Get value
                retrieved = cache.get(key)
                assert retrieved == value
                
                # Delete every other key
                if i % 2 == 0:
                    cache.delete(key)
        
        # Run multiple threads concurrently
        num_threads = 10
        operations_per_thread = 50
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(worker, i, operations_per_thread)
                for i in range(num_threads)
            ]
            
            # Wait for all threads to complete
            for future in futures:
                future.result()
        
        # Verify metrics reflect all operations
        total_expected_sets = num_threads * operations_per_thread
        total_expected_gets = num_threads * operations_per_thread
        
        assert cache.metrics.sets >= total_expected_sets
        assert cache.metrics.hits + cache.metrics.misses >= total_expected_gets
    
    def test_concurrent_eviction(self):
        """Test concurrent access during eviction scenarios."""
        cache = MemoryCache(max_size=50)  # Small size to force evictions
        
        def continuous_writer(thread_id: int, operations: int):
            """Continuously write to cache to trigger evictions."""
            for i in range(operations):
                key = f"writer_{thread_id}_{i}"
                value = f"data_{i}" * 10  # Larger values
                cache.set(key, value)
        
        def continuous_reader(operations: int):
            """Continuously read from cache during evictions."""
            for i in range(operations):
                # Try to read various keys
                key = f"reader_key_{i % 20}"
                cache.get(key)
        
        # Start concurrent writers and readers
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            
            # Start 3 writers
            for i in range(3):
                futures.append(executor.submit(continuous_writer, i, 100))
            
            # Start 2 readers  
            for i in range(2):
                futures.append(executor.submit(continuous_reader, 150))
            
            # Wait for completion
            for future in futures:
                future.result()
        
        # Cache should not exceed max size despite concurrent access
        assert len(cache._cache) <= cache._max_size
        assert cache.metrics.evictions > 0  # Should have triggered evictions


class TestMemoryCachePerformance:
    """Test MemoryCache performance characteristics."""
    
    def test_get_performance(self):
        """Test cache get operation performance."""
        cache = MemoryCache(max_size=10000)
        
        # Pre-populate cache
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")
        
        # Measure get performance
        start_time = time.time()
        num_operations = 10000
        
        for i in range(num_operations):
            key = f"key_{i % 1000}"  # Access existing keys
            cache.get(key)
        
        duration = time.time() - start_time
        operations_per_second = num_operations / duration
        avg_latency_ms = (duration / num_operations) * 1000
        
        # Performance targets for L1 cache
        assert operations_per_second > 100000  # > 100K ops/sec
        assert avg_latency_ms < 0.1  # < 0.1ms average latency
        
        print(f"✅ Get performance: {operations_per_second:.0f} ops/sec, "
              f"avg latency: {avg_latency_ms:.3f}ms")
    
    def test_set_performance(self):
        """Test cache set operation performance."""
        cache = MemoryCache(max_size=10000)
        
        start_time = time.time()
        num_operations = 5000
        
        for i in range(num_operations):
            cache.set(f"perf_key_{i}", f"perf_value_{i}")
        
        duration = time.time() - start_time
        operations_per_second = num_operations / duration
        avg_latency_ms = (duration / num_operations) * 1000
        
        # Performance targets
        assert operations_per_second > 50000  # > 50K ops/sec
        assert avg_latency_ms < 0.2  # < 0.2ms average latency
        
        print(f"✅ Set performance: {operations_per_second:.0f} ops/sec, "
              f"avg latency: {avg_latency_ms:.3f}ms")
    
    def test_memory_efficiency(self):
        """Test cache memory efficiency."""
        cache = MemoryCache(max_size=1000, max_memory_bytes=100 * 1024)  # 100KB limit
        
        # Add entries until memory limit
        entry_size = 100  # Approximate bytes per entry
        max_entries = (100 * 1024) // entry_size
        
        for i in range(max_entries + 10):  # Add more than limit
            value = "x" * entry_size
            cache.set(f"memory_key_{i}", value)
        
        # Should not exceed memory limit
        assert cache._current_memory_bytes <= cache._max_memory_bytes
        assert cache.metrics.evictions > 0  # Should have evicted entries
        
        print(f"✅ Memory efficiency: {cache._current_memory_bytes} bytes used, "
              f"{cache.metrics.evictions} evictions")
    
    def test_concurrent_performance(self):
        """Test cache performance under concurrent load."""
        cache = MemoryCache(max_size=10000)
        
        # Pre-populate cache
        for i in range(1000):
            cache.set(f"concurrent_key_{i}", f"concurrent_value_{i}")
        
        def performance_worker(operations: int) -> float:
            """Worker that measures its own performance."""
            start_time = time.time()
            
            for i in range(operations):
                if i % 3 == 0:
                    cache.set(f"worker_key_{i}", f"worker_value_{i}")
                else:
                    cache.get(f"concurrent_key_{i % 1000}")
            
            return time.time() - start_time
        
        # Run concurrent workers
        operations_per_worker = 1000
        num_workers = 4
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(performance_worker, operations_per_worker)
                for _ in range(num_workers)
            ]
            
            durations = [future.result() for future in futures]
        
        # Calculate aggregate performance
        total_operations = num_workers * operations_per_worker
        max_duration = max(durations)
        aggregate_ops_per_sec = total_operations / max_duration
        
        # Performance should scale reasonably with concurrency
        assert aggregate_ops_per_sec > 50000  # > 50K aggregate ops/sec
        
        print(f"✅ Concurrent performance: {aggregate_ops_per_sec:.0f} aggregate ops/sec "
              f"with {num_workers} threads")


@pytest.mark.asyncio
class TestMemoryCacheAsync:
    """Test MemoryCache async functionality."""
    
    async def test_background_cleanup(self):
        """Test background cleanup of expired entries."""
        cache = MemoryCache(max_size=10)
        
        # Set entries with very short TTL
        cache.set("cleanup_key1", "value1", ttl_seconds=1)
        cache.set("cleanup_key2", "value2", ttl_seconds=1)
        cache.set("permanent_key", "permanent_value")  # No TTL
        
        # Wait for background cleanup (runs every 60s, but entries expire in 1s)
        await asyncio.sleep(1.2)
        
        # Manually trigger cleanup by accessing expired entries
        assert cache.get("cleanup_key1") is None
        assert cache.get("cleanup_key2") is None
        assert cache.get("permanent_key") == "permanent_value"
        
        # Verify cleanup metrics
        assert cache.metrics.expirations >= 2
    
    async def test_cache_shutdown(self):
        """Test cache shutdown and cleanup."""
        cache = MemoryCache(max_size=10)
        
        # Add some entries
        cache.set("shutdown_key1", "value1")
        cache.set("shutdown_key2", "value2")
        
        assert len(cache._cache) == 2
        
        # Shutdown cache
        await cache.shutdown()
        
        # Cache should be empty after shutdown
        assert len(cache._cache) == 0
        assert cache._current_memory_bytes == 0


if __name__ == "__main__":
    print("🔄 Running MemoryCache Tests...")
    
    # Run synchronous tests
    print("\n1. Testing CacheEntry...")
    test_entry = TestCacheEntry()
    test_entry.test_cache_entry_creation()
    test_entry.test_cache_entry_expiration()
    test_entry.test_cache_entry_touch()
    test_entry.test_cache_entry_time_until_expiry()
    test_entry.test_cache_entry_size_calculation()
    print("   ✅ CacheEntry tests passed")
    
    print("2. Testing AccessPattern...")
    test_pattern = TestAccessPattern()
    test_pattern.test_access_pattern_creation()
    test_pattern.test_access_pattern_recording()
    test_pattern.test_access_pattern_frequency_calculation()
    test_pattern.test_access_pattern_prediction()
    test_pattern.test_access_pattern_should_warm()
    print("   ✅ AccessPattern tests passed")
    
    print("3. Testing CacheMetrics...")
    test_metrics = TestCacheMetrics()
    test_metrics.test_cache_metrics_creation()
    test_metrics.test_cache_metrics_recording()
    test_metrics.test_cache_metrics_stats()
    print("   ✅ CacheMetrics tests passed")
    
    print("4. Testing MemoryCache Core...")
    test_cache = TestMemoryCache()
    test_cache.test_memory_cache_creation()
    test_cache.test_basic_cache_operations()
    test_cache.test_cache_ttl_behavior()
    test_cache.test_cache_size_limit_eviction()
    test_cache.test_lru_eviction_policy()
    test_cache.test_lfu_eviction_policy()
    test_cache.test_memory_limit_eviction()
    test_cache.test_cache_exists_method()
    test_cache.test_cache_clear_method()
    test_cache.test_tag_based_operations()
    test_cache.test_access_pattern_tracking()
    test_cache.test_cache_statistics()
    print("   ✅ MemoryCache core tests passed")
    
    print("5. Testing Thread Safety...")
    test_thread_safety = TestMemoryCacheThreadSafety()
    test_thread_safety.test_concurrent_access()
    test_thread_safety.test_concurrent_eviction()
    print("   ✅ Thread safety tests passed")
    
    print("6. Testing Performance...")
    test_performance = TestMemoryCachePerformance()
    test_performance.test_get_performance()
    test_performance.test_set_performance()
    test_performance.test_memory_efficiency()
    test_performance.test_concurrent_performance()
    print("   ✅ Performance tests passed")
    
    print("\n🎯 MemoryCache Testing Complete")
    print("   ✅ All functionality validated with real behavior")
    print("   ✅ Sub-millisecond performance targets met")
    print("   ✅ Thread safety and concurrent access verified")
    print("   ✅ Memory efficiency and eviction policies tested")
    print("   ✅ Access pattern tracking and metrics functional")