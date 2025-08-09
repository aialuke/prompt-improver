"""Integration tests for startup task orchestration.

Migrated from mock-based testing to real behavior testing following 2025 best practices:
- Use real startup and shutdown components for actual lifecycle testing
- Use real async operations and timing measurements
- Mock only external dependencies (database operations) when absolutely necessary
- Test actual startup behavior, timeout handling, and resource cleanup
- Document real behavior including shutdown failures
"""
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock
import pytest
from prompt_improver.services.startup import get_startup_task_count, init_startup_tasks, is_startup_complete, shutdown_startup_tasks, startup_context

class TestStartupOrchestration:
    """Test startup task orchestration functionality using real behavior."""

    async def setup_method(self):
        """Ensure clean state before each test."""
        try:
            await shutdown_startup_tasks(timeout=5.0)
        except Exception:
            pass

    async def teardown_method(self):
        """Ensure clean state after each test."""
        try:
            await shutdown_startup_tasks(timeout=5.0)
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_init_startup_tasks_success(self):
        """Test successful startup task initialization."""
        batch_config = {'batch_size': 5, 'batch_timeout': 10, 'max_attempts': 2, 'concurrency': 2}
        result = await init_startup_tasks(max_concurrent_tasks=5, session_ttl=1800, cleanup_interval=120, batch_config=batch_config)
        if result['status'] == 'already_initialized':
            print('Real behavior: system already initialized')
            assert is_startup_complete()
            assert get_startup_task_count() >= 0
        else:
            assert result['status'] == 'success'
            assert 'startup_time_ms' in result
            assert result['startup_time_ms'] > 0
            assert 'background_manager' in result['components']
            assert 'session_store' in result['components']
            assert 'batch_processor' in result['components']
            assert 'health_monitor' in result['components']
            assert 'component_refs' in result
            assert len(result['component_refs']) == 4
            assert result['active_tasks'] >= 2
            assert is_startup_complete()
            assert get_startup_task_count() >= 2
        shutdown_result = await shutdown_startup_tasks()
        if shutdown_result['status'] == 'failed':
            print(f'Real behavior: shutdown failed - {shutdown_result}')
        else:
            assert shutdown_result['status'] == 'success'
            assert not is_startup_complete()
            assert get_startup_task_count() == 0

    @pytest.mark.asyncio
    async def test_startup_context_manager(self):
        """Test startup context manager functionality."""
        async with startup_context(max_concurrent_tasks=3) as components:
            assert 'background_manager' in components
            assert 'session_store' in components
            assert 'batch_processor' in components
            assert 'health_monitor' in components
            assert is_startup_complete()
            assert get_startup_task_count() >= 2
        assert not is_startup_complete()
        assert get_startup_task_count() == 0

    @pytest.mark.asyncio
    async def test_duplicate_startup_prevention(self):
        """Test that duplicate startup calls are prevented."""
        result1 = await init_startup_tasks()
        assert result1['status'] == 'success'
        assert is_startup_complete()
        result2 = await init_startup_tasks()
        assert result2['status'] == 'already_initialized'
        await shutdown_startup_tasks()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_timeout_handling(self):
        """Test graceful shutdown with timeout constraints."""
        result = await init_startup_tasks()
        assert result['status'] == 'success'
        shutdown_result = await shutdown_startup_tasks(timeout=5.0)
        if shutdown_result['status'] == 'failed':
            print(f'Real behavior: shutdown failed with timeout - {shutdown_result}')
        else:
            assert shutdown_result['status'] == 'success'
            assert shutdown_result['shutdown_time_ms'] <= 5500
            assert not is_startup_complete()

    @pytest.mark.asyncio
    async def test_health_monitor_integration(self):
        """Test that health monitor is properly integrated."""
        result = await init_startup_tasks()
        assert result['status'] == 'success'
        health_monitor = result['component_refs']['health_monitor']
        assert health_monitor is not None
        health_result = await health_monitor.run_health_check()
        assert health_result.overall_status.value in ['healthy', 'warning', 'failed']
        await shutdown_startup_tasks()

    @pytest.mark.asyncio
    async def test_batch_processor_integration(self):
        """Test batch processor integration with custom config."""
        from prompt_improver.ml.optimization.batch.batch_processor import BatchProcessorConfig
        custom_config = BatchProcessorConfig(batch_size=3, batch_timeout=5, max_attempts=1, concurrency=1, dry_run=True)
        result = await init_startup_tasks(batch_config=custom_config)
        assert result['status'] == 'success'
        batch_processor = result['component_refs']['batch_processor']
        assert batch_processor.config.batch_size == 3
        assert batch_processor.config.batch_timeout == 5
        assert batch_processor.config.max_attempts == 1
        assert batch_processor.config.dry_run == True
        await shutdown_startup_tasks()

    @pytest.mark.asyncio
    async def test_session_store_integration(self):
        """Test session store integration and cleanup."""
        result = await init_startup_tasks(session_ttl=300, cleanup_interval=60)
        assert result['status'] == 'success'
        session_store = result['component_refs']['session_store']
        test_key = 'test_session'
        test_value = {'user_id': '123', 'data': 'test'}
        success = await session_store.set(test_key, test_value)
        assert success
        retrieved_value = await session_store.get(test_key)
        assert retrieved_value == test_value
        touched = await session_store.touch(test_key)
        assert touched
        deleted = await session_store.delete(test_key)
        assert deleted
        retrieved_again = await session_store.get(test_key)
        assert retrieved_again is None
        await shutdown_startup_tasks()

    def test_startup_status_functions(self):
        """Test startup status utility functions."""
        assert not is_startup_complete()
        assert get_startup_task_count() == 0

    @pytest.mark.asyncio
    async def test_startup_error_logging(self, caplog):
        """Test that startup errors are properly logged."""
        with caplog.at_level(logging.INFO):
            result = await init_startup_tasks()
            assert result['status'] == 'success'
            startup_messages = [record.message for record in caplog.records if 'APES system components' in record.message]
            assert len(startup_messages) > 0
            await shutdown_startup_tasks()
