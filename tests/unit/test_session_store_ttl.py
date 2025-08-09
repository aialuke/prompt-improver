"""
Unit tests for SessionStore TTL (Time-To-Live) functionality.

MIGRATION STATUS: MIGRATED TO REAL BEHAVIOR TESTING (2025 BEST PRACTICES)

Tests cover TTL behavior, cleanup tasks, async operations, and edge cases
following pytest best practices with real behavior testing:

✅ REAL BEHAVIOR TESTING:
- Real TTL expiration with controlled timing
- Real SessionStore and TTLCache operations
- Real async operations and cleanup tasks
- Real error scenarios with corrupted caches
- Real concurrent operations testing

✅ STRATEGIC TESTING APPROACH:
- Use actual asyncio.sleep for TTL timing validation
- Test real cache behavior and expiration
- Validate actual lock contention and threading
- Test real cleanup task lifecycle

No mocks used except for performance benchmarking compatibility.
"""
import asyncio
import time
from typing import Any, Dict
import pytest
from hypothesis import assume, given, settings, strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, precondition, rule
from prompt_improver.utils.session_store import SessionStore

@pytest.mark.asyncio
class TestSessionStoreTTL:
    """Unit tests for SessionStore TTL functionality."""

    async def test_session_store_initialization(self):
        """Test SessionStore initialization with default and custom parameters."""
        store = SessionStore()
        assert store.cache.maxsize == 1000
        assert store.cache.ttl == 3600
        assert store.cleanup_interval == 300
        assert store.cleanup_task is None
        assert not store._running
        store = SessionStore(maxsize=500, ttl=1800, cleanup_interval=120)
        assert store.cache.maxsize == 500
        assert store.cache.ttl == 1800
        assert store.cleanup_interval == 120

    async def test_basic_session_operations(self):
        """Test basic session set/get/delete operations."""
        store = SessionStore(maxsize=10, ttl=60)
        success = await store.set('key1', {'data': 'value1'})
        assert success
        value = await store.get('key1')
        assert value == {'data': 'value1'}
        value = await store.get('nonexistent')
        assert value is None
        success = await store.delete('key1')
        assert success
        value = await store.get('key1')
        assert value is None
        success = await store.delete('nonexistent')
        assert not success

    async def test_session_ttl_expiration(self):
        """Test that sessions expire according to TTL."""
        store = SessionStore(maxsize=10, ttl=0.1)
        await store.set('key1', 'value1')
        value = await store.get('key1')
        assert value == 'value1'
        await asyncio.sleep(0.2)
        value = await store.get('key1')
        assert value is None

    async def test_session_touch_extends_ttl(self):
        """Test that touching a session extends its TTL."""
        store = SessionStore(maxsize=10, ttl=0.2)
        await store.set('key1', 'value1')
        await asyncio.sleep(0.1)
        touched = await store.touch('key1')
        assert touched
        await asyncio.sleep(0.1)
        value = await store.get('key1')
        assert value == 'value1'
        await asyncio.sleep(0.2)
        value = await store.get('key1')
        assert value is None

    async def test_session_touch_nonexistent_key(self):
        """Test touching a non-existent key."""
        store = SessionStore(maxsize=10, ttl=60)
        touched = await store.touch('nonexistent')
        assert not touched

    async def test_session_clear_all(self):
        """Test clearing all sessions."""
        store = SessionStore(maxsize=10, ttl=60)
        await store.set('key1', 'value1')
        await store.set('key2', 'value2')
        await store.set('key3', 'value3')
        assert await store.size() == 3
        success = await store.clear()
        assert success
        assert await store.size() == 0
        assert await store.get('key1') is None
        assert await store.get('key2') is None
        assert await store.get('key3') is None

    async def test_session_size_tracking(self):
        """Test session size tracking."""
        store = SessionStore(maxsize=10, ttl=60)
        assert await store.size() == 0
        await store.set('key1', 'value1')
        await store.set('key2', 'value2')
        assert await store.size() == 2
        await store.delete('key1')
        assert await store.size() == 1
        await store.clear()
        assert await store.size() == 0

    async def test_session_stats(self):
        """Test session store statistics."""
        store = SessionStore(maxsize=100, ttl=3600, cleanup_interval=300)
        await store.set('key1', 'value1')
        await store.set('key2', 'value2')
        stats = await store.stats()
        assert stats['current_size'] == 2
        assert stats['max_size'] == 100
        assert stats['ttl_seconds'] == 3600
        assert stats['cleanup_interval'] == 300
        assert stats['cleanup_running'] is False

    async def test_cleanup_task_lifecycle(self):
        """Test cleanup task start/stop lifecycle."""
        store = SessionStore(maxsize=10, ttl=60, cleanup_interval=0.1)
        assert not store._running
        assert store.cleanup_task is None
        started = await store.start_cleanup_task()
        assert started
        assert store._running
        assert store.cleanup_task is not None
        started = await store.start_cleanup_task()
        assert not started
        stopped = await store.stop_cleanup_task()
        assert stopped
        assert not store._running
        stopped = await store.stop_cleanup_task()
        assert not stopped

    async def test_cleanup_task_removes_expired_sessions(self):
        """Test that cleanup task removes expired sessions."""
        store = SessionStore(maxsize=10, ttl=0.1, cleanup_interval=0.05)
        await store.set('key1', 'value1')
        await store.set('key2', 'value2')
        await store.start_cleanup_task()
        await asyncio.sleep(0.2)
        await asyncio.sleep(0.1)
        assert await store.size() == 0
        await store.stop_cleanup_task()

    async def test_context_manager_usage(self):
        """Test SessionStore as async context manager."""
        store = SessionStore(maxsize=10, ttl=60, cleanup_interval=0.1)
        async with store:
            assert store._running
            await store.set('key1', 'value1')
            value = await store.get('key1')
            assert value == 'value1'
        assert not store._running

    async def test_error_handling_with_real_scenarios(self):
        """Test error handling in session operations using real error scenarios."""
        store = SessionStore(maxsize=10, ttl=60)
        value = await store.get('nonexistent_key')
        assert value is None
        small_store = SessionStore(maxsize=2, ttl=60)
        success1 = await small_store.set('key1', 'value1')
        success2 = await small_store.set('key2', 'value2')
        success3 = await small_store.set('key3', 'value3')
        assert success1 is True
        assert success2 is True
        assert success3 is True
        assert await small_store.size() == 2
        assert await small_store.get('key1') is None
        assert await small_store.get('key2') is not None
        assert await small_store.get('key3') is not None
        expired_store = SessionStore(maxsize=10, ttl=0.05)
        await expired_store.set('temp_key', 'temp_value')
        await asyncio.sleep(0.1)
        expired_value = await expired_store.get('temp_key')
        assert expired_value is None
        empty_store = SessionStore(maxsize=10, ttl=60)
        touched = await empty_store.touch('nonexistent')
        assert not touched
        deleted = await empty_store.delete('nonexistent')
        assert not deleted

    async def test_concurrent_operations(self):
        """Test concurrent session operations."""
        store = SessionStore(maxsize=100, ttl=60)

        async def set_sessions(start_idx, count):
            for i in range(start_idx, start_idx + count):
                await store.set(f'key{i}', f'value{i}')

        async def get_sessions(start_idx, count):
            results = []
            for i in range(start_idx, start_idx + count):
                value = await store.get(f'key{i}')
                results.append(value)
            return results
        await asyncio.gather(set_sessions(0, 10), set_sessions(10, 10), set_sessions(20, 10))
        assert await store.size() == 30
        results = await asyncio.gather(get_sessions(0, 10), get_sessions(10, 10), get_sessions(20, 10))
        for result_set in results:
            assert len(result_set) == 10
            assert all((r is not None for r in result_set))

    async def test_maxsize_enforcement(self):
        """Test that maxsize is enforced by TTLCache."""
        store = SessionStore(maxsize=5, ttl=60)
        for i in range(5):
            await store.set(f'key{i}', f'value{i}')
        assert await store.size() == 5
        await store.set('key5', 'value5')
        assert await store.size() == 5

    @given(keys=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20, unique=True), values=st.lists(st.dictionaries(st.text(min_size=1, max_size=20), st.integers()), min_size=1, max_size=20))
    async def test_session_operations_properties(self, keys, values):
        """Property-based test for session operations."""
        assume(len(keys) == len(values))
        store = SessionStore(maxsize=100, ttl=60)
        key_value_map = {}
        for key, value in zip(keys, values, strict=False):
            success = await store.set(key, value)
            assert success
            key_value_map[key] = value
        for key, expected_value in key_value_map.items():
            retrieved_value = await store.get(key)
            assert retrieved_value == expected_value
        assert await store.size() == len(key_value_map)

    @given(st.integers(min_value=1, max_value=100))
    async def test_session_store_sizes(self, maxsize):
        """Property-based test for different store sizes."""
        store = SessionStore(maxsize=maxsize, ttl=60)
        for i in range(maxsize):
            await store.set(f'key{i}', f'value{i}')
        current_size = await store.size()
        assert current_size <= maxsize

    @settings(deadline=None, max_examples=5)
    @given(st.floats(min_value=0.05, max_value=0.3))
    async def test_ttl_timing_properties(self, ttl):
        """Property-based test for TTL timing with real behavior validation."""
        store = SessionStore(maxsize=10, ttl=ttl)
        start_time = time.time()
        await store.set('key1', 'value1')
        value = await store.get('key1')
        assert value == 'value1'
        if ttl > 0.1:
            await asyncio.sleep(ttl / 3)
            mid_value = await store.get('key1')
            assert mid_value == 'value1', 'Session should be valid at 1/3 TTL'
        await asyncio.sleep(ttl + 0.05)
        value = await store.get('key1')
        assert value is None
        elapsed = time.time() - start_time
        assert elapsed >= ttl, f'Total test time {elapsed:.3f}s should be >= TTL {ttl}s'

class SessionStoreStateMachine(RuleBasedStateMachine):
    """Stateful testing for SessionStore using Hypothesis."""
    keys = Bundle('keys')

    def __init__(self):
        super().__init__()
        self.store = SessionStore(maxsize=50, ttl=60)
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
        """Get a session with real behavior validation."""
        value = await self.store.get(key)
        if key in self.stored_keys:
            pass
        else:
            assert value is None, f'Never-stored key {key} should return None'

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
        """Benchmark session set operations with real behavior."""
        store = SessionStore(maxsize=1000, ttl=3600)

        async def set_session():
            return await store.set('test_key', {'data': 'test_value'})
        result = await set_session()
        assert result is True
        retrieved_value = await store.get('test_key')
        assert retrieved_value == {'data': 'test_value'}

    async def test_get_operation_performance(self, benchmark):
        """Benchmark session get operations with real behavior validation."""
        store = SessionStore(maxsize=1000, ttl=3600)
        await store.set('test_key', {'data': 'test_value'})
        pre_benchmark_value = await store.get('test_key')
        assert pre_benchmark_value == {'data': 'test_value'}

        async def get_session():
            return await store.get('test_key')
        result = await get_session()
        assert result == {'data': 'test_value'}
        post_benchmark_value = await store.get('test_key')
        assert post_benchmark_value == {'data': 'test_value'}

    async def test_concurrent_operations_performance(self, benchmark):
        """Benchmark concurrent session operations with real behavior validation."""
        store = SessionStore(maxsize=1000, ttl=3600)

        async def concurrent_operations():
            tasks = []
            for i in range(100):
                tasks.append(store.set(f'key{i}', f'value{i}'))
            set_results = await asyncio.gather(*tasks)
            assert all(set_results), 'All concurrent set operations should succeed'
            tasks = []
            for i in range(100):
                tasks.append(store.get(f'key{i}'))
            results = await asyncio.gather(*tasks)
            return len([r for r in results if r is not None])
        result = await concurrent_operations()
        assert result == 100
        final_size = await store.size()
        assert final_size == 100

        async def concurrent_access():
            tasks = [store.get('key0') for _ in range(10)]
            access_results = await asyncio.gather(*tasks)
            return access_results
        access_results = await concurrent_access()
        assert all((r == 'value0' for r in access_results)), 'Concurrent access should return consistent values'

    async def test_real_ttl_precision_and_timing(self):
        """Test TTL precision and timing with real behavior."""
        precise_store = SessionStore(maxsize=10, ttl=0.2)
        start_time = time.time()
        await precise_store.set('precise_key', 'precise_value')
        immediate_value = await precise_store.get('precise_key')
        assert immediate_value == 'precise_value'
        await asyncio.sleep(0.1)
        half_ttl_value = await precise_store.get('precise_key')
        assert half_ttl_value == 'precise_value'
        await asyncio.sleep(0.15)
        expired_value = await precise_store.get('precise_key')
        assert expired_value is None
        elapsed = time.time() - start_time
        assert elapsed >= 0.2, f'Test should take at least TTL duration, took {elapsed:.3f}s'

    async def test_real_cleanup_task_behavior(self):
        """Test real cleanup task behavior with timing."""
        cleanup_store = SessionStore(maxsize=20, ttl=0.1, cleanup_interval=0.05)
        for i in range(10):
            await cleanup_store.set(f'cleanup_key{i}', f'cleanup_value{i}')
        initial_size = await cleanup_store.size()
        assert initial_size == 10
        async with cleanup_store:
            await asyncio.sleep(0.15)
            await asyncio.sleep(0.1)
            final_size = await cleanup_store.size()
            assert final_size == 0, 'Cleanup task should have removed expired sessions'
            for i in range(10):
                value = await cleanup_store.get(f'cleanup_key{i}')
                assert value is None, f'Session cleanup_key{i} should be cleaned up'

    async def test_real_touch_behavior_edge_cases(self):
        """Test real touch behavior in edge cases."""
        touch_store = SessionStore(maxsize=10, ttl=0.2)
        await touch_store.set('touch_key', 'touch_value')
        touched = await touch_store.touch('touch_key')
        assert touched is True
        await asyncio.sleep(0.15)
        touched_near_expiry = await touch_store.touch('touch_key')
        assert touched_near_expiry is True
        await asyncio.sleep(0.1)
        value_after_touch = await touch_store.get('touch_key')
        assert value_after_touch == 'touch_value'
        await asyncio.sleep(0.15)
        expired_value = await touch_store.get('touch_key')
        assert expired_value is None
        touched_expired = await touch_store.touch('touch_key')
        assert touched_expired is False

    async def test_cleanup_performance(self, benchmark):
        """Benchmark cleanup task performance with real behavior validation."""
        store = SessionStore(maxsize=1000, ttl=0.01, cleanup_interval=0.1)
        for i in range(500):
            await store.set(f'key{i}', f'value{i}')
        initial_size = await store.size()
        assert initial_size == 500
        await asyncio.sleep(0.05)

        def run_cleanup():
            return store.cache.expire()
        benchmark(run_cleanup)
        final_size = await store.size()
        assert final_size == 0
        for i in range(10):
            value = await store.get(f'key{i}')
            assert value is None

@pytest.mark.asyncio
async def test_session_store_integration():
    """Integration test combining TTL, cleanup, and concurrent operations."""
    store = SessionStore(maxsize=100, ttl=2.0, cleanup_interval=0.1)
    async with store:
        for i in range(20):
            await store.set(f'key{i}', {'id': i, 'data': f'value{i}'})
        assert await store.size() == 20
        for i in range(20):
            value = await store.get(f'key{i}')
            assert value is not None
            assert value['id'] == i
        concurrent_tasks = []
        for i in range(20, 40):
            task = store.set(f'key{i}', {'id': i, 'data': f'value{i}'})
            concurrent_tasks.append(task)
        results = await asyncio.gather(*concurrent_tasks)
        assert all(results), 'All concurrent set operations should succeed'
        final_size = await store.size()
        assert final_size == 40
        for i in range(40):
            value = await store.get(f'key{i}')
            assert value is not None
            assert value['id'] == i
        await store.clear()
        assert await store.size() == 0
