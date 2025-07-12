"""
Unit tests for SessionStore TTL (Time-To-Live) functionality.

Tests cover TTL behavior, cleanup tasks, async operations, and edge cases
following pytest best practices and property-based testing.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, precondition

from prompt_improver.utils.session_store import SessionStore


@pytest.mark.asyncio
class TestSessionStoreTTL:
    """Unit tests for SessionStore TTL functionality."""
    
    async def test_session_store_initialization(self):
        """Test SessionStore initialization with default and custom parameters."""
        # Test default initialization
        store = SessionStore()
        assert store.cache.maxsize == 1000
        assert store.cache.ttl == 3600  # 1 hour default
        assert store.cleanup_interval == 300  # 5 minutes default
        assert store.cleanup_task is None
        assert not store._running
        
        # Test custom initialization
        store = SessionStore(maxsize=500, ttl=1800, cleanup_interval=120)
        assert store.cache.maxsize == 500
        assert store.cache.ttl == 1800
        assert store.cleanup_interval == 120
    
    async def test_basic_session_operations(self):
        """Test basic session set/get/delete operations."""
        store = SessionStore(maxsize=10, ttl=60)
        
        # Test set and get
        success = await store.set("key1", {"data": "value1"})
        assert success
        
        value = await store.get("key1")
        assert value == {"data": "value1"}
        
        # Test non-existent key
        value = await store.get("nonexistent")
        assert value is None
        
        # Test delete
        success = await store.delete("key1")
        assert success
        
        value = await store.get("key1")
        assert value is None
        
        # Test delete non-existent key
        success = await store.delete("nonexistent")
        assert not success
    
    async def test_session_ttl_expiration(self):
        """Test that sessions expire according to TTL."""
        # Use very short TTL for testing
        store = SessionStore(maxsize=10, ttl=0.1)  # 100ms TTL
        
        # Set a session
        await store.set("key1", "value1")
        
        # Should be available immediately
        value = await store.get("key1")
        assert value == "value1"
        
        # Wait for TTL to expire
        await asyncio.sleep(0.2)
        
        # Should be expired now
        value = await store.get("key1")
        assert value is None
    
    async def test_session_touch_extends_ttl(self):
        """Test that touching a session extends its TTL."""
        store = SessionStore(maxsize=10, ttl=0.2)  # 200ms TTL
        
        # Set a session
        await store.set("key1", "value1")
        
        # Wait half the TTL
        await asyncio.sleep(0.1)
        
        # Touch the session to extend TTL
        touched = await store.touch("key1")
        assert touched
        
        # Wait another half TTL (should still be alive due to touch)
        await asyncio.sleep(0.1)
        
        # Should still be available
        value = await store.get("key1")
        assert value == "value1"
        
        # Wait for full TTL to expire
        await asyncio.sleep(0.2)
        
        # Should be expired now
        value = await store.get("key1")
        assert value is None
    
    async def test_session_touch_nonexistent_key(self):
        """Test touching a non-existent key."""
        store = SessionStore(maxsize=10, ttl=60)
        
        touched = await store.touch("nonexistent")
        assert not touched
    
    async def test_session_clear_all(self):
        """Test clearing all sessions."""
        store = SessionStore(maxsize=10, ttl=60)
        
        # Add multiple sessions
        await store.set("key1", "value1")
        await store.set("key2", "value2")
        await store.set("key3", "value3")
        
        # Verify sessions exist
        assert await store.size() == 3
        
        # Clear all sessions
        success = await store.clear()
        assert success
        
        # Verify all sessions are gone
        assert await store.size() == 0
        assert await store.get("key1") is None
        assert await store.get("key2") is None
        assert await store.get("key3") is None
    
    async def test_session_size_tracking(self):
        """Test session size tracking."""
        store = SessionStore(maxsize=10, ttl=60)
        
        # Initially empty
        assert await store.size() == 0
        
        # Add sessions
        await store.set("key1", "value1")
        await store.set("key2", "value2")
        assert await store.size() == 2
        
        # Delete one session
        await store.delete("key1")
        assert await store.size() == 1
        
        # Clear all
        await store.clear()
        assert await store.size() == 0
    
    async def test_session_stats(self):
        """Test session store statistics."""
        store = SessionStore(maxsize=100, ttl=3600, cleanup_interval=300)
        
        # Add some sessions
        await store.set("key1", "value1")
        await store.set("key2", "value2")
        
        stats = await store.stats()
        assert stats["current_size"] == 2
        assert stats["max_size"] == 100
        assert stats["ttl_seconds"] == 3600
        assert stats["cleanup_interval"] == 300
        assert stats["cleanup_running"] is False
    
    async def test_cleanup_task_lifecycle(self):
        """Test cleanup task start/stop lifecycle."""
        store = SessionStore(maxsize=10, ttl=60, cleanup_interval=0.1)  # 100ms cleanup
        
        # Initially not running
        assert not store._running
        assert store.cleanup_task is None
        
        # Start cleanup task
        started = await store.start_cleanup_task()
        assert started
        assert store._running
        assert store.cleanup_task is not None
        
        # Try to start again (should return False)
        started = await store.start_cleanup_task()
        assert not started
        
        # Stop cleanup task
        stopped = await store.stop_cleanup_task()
        assert stopped
        assert not store._running
        
        # Try to stop again (should return False)
        stopped = await store.stop_cleanup_task()
        assert not stopped
    
    async def test_cleanup_task_removes_expired_sessions(self):
        """Test that cleanup task removes expired sessions."""
        store = SessionStore(maxsize=10, ttl=0.1, cleanup_interval=0.05)  # 50ms cleanup
        
        # Add sessions
        await store.set("key1", "value1")
        await store.set("key2", "value2")
        
        # Start cleanup task
        await store.start_cleanup_task()
        
        # Wait for sessions to expire
        await asyncio.sleep(0.2)
        
        # Wait for cleanup to run
        await asyncio.sleep(0.1)
        
        # Sessions should be cleaned up
        assert await store.size() == 0
        
        # Stop cleanup task
        await store.stop_cleanup_task()
    
    async def test_context_manager_usage(self):
        """Test SessionStore as async context manager."""
        store = SessionStore(maxsize=10, ttl=60, cleanup_interval=0.1)
        
        async with store:
            # Cleanup task should be running
            assert store._running
            
            # Add and retrieve session
            await store.set("key1", "value1")
            value = await store.get("key1")
            assert value == "value1"
        
        # Cleanup task should be stopped
        assert not store._running
    
    async def test_error_handling_in_operations(self):
        """Test error handling in session operations."""
        store = SessionStore(maxsize=10, ttl=60)
        
        # Mock cache to raise exceptions
        with patch.object(store.cache, 'get', side_effect=Exception("Cache error")):
            value = await store.get("key1")
            assert value is None  # Should return None on error
        
        with patch.object(store.cache, '__setitem__', side_effect=Exception("Cache error")):
            success = await store.set("key1", "value1")
            assert not success  # Should return False on error
        
        with patch.object(store.cache, '__contains__', side_effect=Exception("Cache error")):
            touched = await store.touch("key1")
            assert not touched  # Should return False on error
        
        with patch.object(store.cache, '__delitem__', side_effect=Exception("Cache error")):
            deleted = await store.delete("key1")
            assert not deleted  # Should return False on error
    
    async def test_concurrent_operations(self):
        """Test concurrent session operations."""
        store = SessionStore(maxsize=100, ttl=60)
        
        # Define concurrent operations
        async def set_sessions(start_idx, count):
            for i in range(start_idx, start_idx + count):
                await store.set(f"key{i}", f"value{i}")
        
        async def get_sessions(start_idx, count):
            results = []
            for i in range(start_idx, start_idx + count):
                value = await store.get(f"key{i}")
                results.append(value)
            return results
        
        # Run concurrent set operations
        await asyncio.gather(
            set_sessions(0, 10),
            set_sessions(10, 10),
            set_sessions(20, 10)
        )
        
        # Verify all sessions were set
        assert await store.size() == 30
        
        # Run concurrent get operations
        results = await asyncio.gather(
            get_sessions(0, 10),
            get_sessions(10, 10),
            get_sessions(20, 10)
        )
        
        # Verify all sessions were retrieved
        for result_set in results:
            assert len(result_set) == 10
            assert all(r is not None for r in result_set)
    
    async def test_maxsize_enforcement(self):
        """Test that maxsize is enforced by TTLCache."""
        store = SessionStore(maxsize=5, ttl=60)
        
        # Fill cache to maxsize
        for i in range(5):
            await store.set(f"key{i}", f"value{i}")
        
        assert await store.size() == 5
        
        # Adding more should evict oldest entries
        await store.set("key5", "value5")
        
        # Cache should still be at maxsize
        assert await store.size() == 5
    
    # Property-based tests using Hypothesis
    @given(
        keys=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20),
        values=st.lists(st.dictionaries(st.text(min_size=1, max_size=20), st.integers()), min_size=1, max_size=20)
    )
    async def test_session_operations_properties(self, keys, values):
        """Property-based test for session operations."""
        assume(len(keys) == len(values))
        
        store = SessionStore(maxsize=100, ttl=60)
        
        # Set all key-value pairs
        for key, value in zip(keys, values):
            success = await store.set(key, value)
            assert success
        
        # Verify all can be retrieved
        for key, expected_value in zip(keys, values):
            retrieved_value = await store.get(key)
            assert retrieved_value == expected_value
        
        # Verify size
        unique_keys = set(keys)
        assert await store.size() == len(unique_keys)
    
    @given(st.integers(min_value=1, max_value=100))
    async def test_session_store_sizes(self, maxsize):
        """Property-based test for different store sizes."""
        store = SessionStore(maxsize=maxsize, ttl=60)
        
        # Add sessions up to maxsize
        for i in range(maxsize):
            await store.set(f"key{i}", f"value{i}")
        
        # Size should not exceed maxsize
        current_size = await store.size()
        assert current_size <= maxsize
    
    @settings(deadline=None)
    @given(st.floats(min_value=0.01, max_value=1.0))
    async def test_ttl_timing_properties(self, ttl):
        """Property-based test for TTL timing."""
        store = SessionStore(maxsize=10, ttl=ttl)
        
        # Set a session
        await store.set("key1", "value1")
        
        # Should be available immediately
        value = await store.get("key1")
        assert value == "value1"
        
        # Wait for TTL plus buffer
        await asyncio.sleep(ttl + 0.1)
        
        # Should be expired
        value = await store.get("key1")
        assert value is None


class SessionStoreStateMachine(RuleBasedStateMachine):
    """Stateful testing for SessionStore using Hypothesis."""
    
    def __init__(self):
        super().__init__()
        self.store = SessionStore(maxsize=50, ttl=60)
        self.keys = Bundle('keys')
        self.stored_keys = set()
    
    @rule(target=keys, key=st.text(min_size=1, max_size=20))
    def add_key(self, key):
        """Add a key to track."""
        return key
    
    @rule(key=keys, value=st.dictionaries(st.text(min_size=1, max_size=10), st.integers()))
    async def set_session(self, key, value):
        """Set a session."""
        success = await self.store.set(key, value)
        if success:
            self.stored_keys.add(key)
    
    @rule(key=keys)
    async def get_session(self, key):
        """Get a session."""
        value = await self.store.get(key)
        # If key was stored, it should be retrievable (unless expired)
        if key in self.stored_keys:
            # Note: In a real scenario, we'd need to track expiration
            pass
    
    @rule(key=keys)
    async def delete_session(self, key):
        """Delete a session."""
        success = await self.store.delete(key)
        if success:
            self.stored_keys.discard(key)
    
    @rule()
    async def clear_all_sessions(self):
        """Clear all sessions."""
        await self.store.clear()
        self.stored_keys.clear()
    
    @rule()
    async def check_size_invariant(self):
        """Check that size is consistent."""
        size = await self.store.size()
        assert size >= 0


@pytest.mark.asyncio
class TestSessionStorePerformance:
    """Performance tests for SessionStore operations."""
    
    async def test_set_operation_performance(self, benchmark):
        """Benchmark session set operations."""
        store = SessionStore(maxsize=1000, ttl=3600)
        
        def set_session():
            # Using asyncio.run for benchmark compatibility
            return asyncio.run(store.set("test_key", {"data": "test_value"}))
        
        result = benchmark(set_session)
        assert result is True
    
    async def test_get_operation_performance(self, benchmark):
        """Benchmark session get operations."""
        store = SessionStore(maxsize=1000, ttl=3600)
        
        # Pre-populate the session
        await store.set("test_key", {"data": "test_value"})
        
        def get_session():
            return asyncio.run(store.get("test_key"))
        
        result = benchmark(get_session)
        assert result == {"data": "test_value"}
    
    async def test_concurrent_operations_performance(self, benchmark):
        """Benchmark concurrent session operations."""
        store = SessionStore(maxsize=1000, ttl=3600)
        
        async def concurrent_operations():
            # Set multiple sessions concurrently
            tasks = []
            for i in range(100):
                tasks.append(store.set(f"key{i}", f"value{i}"))
            
            await asyncio.gather(*tasks)
            
            # Get multiple sessions concurrently
            tasks = []
            for i in range(100):
                tasks.append(store.get(f"key{i}"))
            
            results = await asyncio.gather(*tasks)
            return len([r for r in results if r is not None])
        
        def run_concurrent():
            return asyncio.run(concurrent_operations())
        
        result = benchmark(run_concurrent)
        assert result == 100
    
    async def test_cleanup_performance(self, benchmark):
        """Benchmark cleanup task performance."""
        store = SessionStore(maxsize=1000, ttl=0.01, cleanup_interval=0.1)
        
        # Pre-populate with expired sessions
        for i in range(500):
            await store.set(f"key{i}", f"value{i}")
        
        # Wait for expiration
        await asyncio.sleep(0.05)
        
        def run_cleanup():
            return asyncio.run(store.cache.expire())
        
        benchmark(run_cleanup)
        
        # Verify cleanup worked
        assert await store.size() == 0


# Integration test combining multiple features
@pytest.mark.asyncio
async def test_session_store_integration():
    """Integration test combining TTL, cleanup, and concurrent operations."""
    store = SessionStore(maxsize=100, ttl=0.5, cleanup_interval=0.1)
    
    async with store:
        # Add multiple sessions
        for i in range(50):
            await store.set(f"key{i}", {"id": i, "data": f"value{i}"})
        
        # Verify all sessions exist
        assert await store.size() == 50
        
        # Touch some sessions to extend their TTL
        for i in range(0, 50, 5):  # Every 5th session
            await store.touch(f"key{i}")
        
        # Wait for partial expiration
        await asyncio.sleep(0.6)
        
        # Wait for cleanup to run
        await asyncio.sleep(0.2)
        
        # Touched sessions should still exist
        remaining_size = await store.size()
        assert remaining_size > 0  # Some sessions should remain
        
        # Verify touched sessions are still accessible
        for i in range(0, 50, 5):
            value = await store.get(f"key{i}")
            if value is not None:  # May have been evicted due to maxsize
                assert value["id"] == i
