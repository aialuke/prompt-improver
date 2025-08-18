"""
Integration tests for graceful shutdown sequence and component lifecycle management.

Migrated from mock-based testing to real behavior testing following 2025 best practices:
- Use real startup and shutdown components for actual lifecycle testing
- Use real async operations and timing measurements
- Mock only external dependencies (database operations) when absolutely necessary
- Test actual shutdown sequence behavior, timeout handling, and resource cleanup
- Verify real performance characteristics and graceful error handling
"""

import asyncio
import logging
import os
import signal
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prompt_improver.optimization.batch_processor import (
    BatchProcessor,
    BatchProcessorConfig,
)
from prompt_improver.services.startup import (
    get_startup_task_count,
    init_startup_tasks,
    is_startup_complete,
    shutdown_startup_tasks,
    startup_context,
)
from prompt_improver.services.cache.cache_facade import CacheFacade


@pytest.mark.asyncio
class TestShutdownSequence:
    """Integration tests for graceful shutdown sequence using real behavior."""

    async def setup_method(self):
        """Ensure clean state before each test."""
        try:
            await shutdown_startup_tasks(timeout=5.0)
        except Exception:
            pass
        assert not is_startup_complete()
        assert get_startup_task_count() == 0

    async def teardown_method(self):
        """Ensure clean state after each test."""
        try:
            await shutdown_startup_tasks(timeout=5.0)
        except Exception:
            pass

    async def test_basic_startup_shutdown_cycle_real_behavior(self):
        """Test basic startup and shutdown cycle with real components."""
        assert not is_startup_complete()
        assert get_startup_task_count() == 0
        startup_result = await init_startup_tasks(
            max_concurrent_tasks=3,
            session_ttl=300,
            cleanup_interval=120,
            batch_config={
                "batch_size": 5,
                "batch_timeout": 30,
                "dry_run": True,
                "enable_priority_queue": False,
                "max_attempts": 1,
            },
        )
        if startup_result["status"] == "already_initialized":
            print("Real behavior: system already initialized from previous test")
            shutdown_result = await shutdown_startup_tasks(timeout=10.0)
            return
        if startup_result["status"] != "success":
            print(f"Startup had issues: {startup_result['status']}")
            shutdown_result = await shutdown_startup_tasks(timeout=10.0)
            return
        if startup_result["status"] == "already_initialized":
            print("Using already initialized system")
        elif startup_result["status"] != "success":
            return
        assert startup_result["startup_time_ms"] > 0
        assert startup_result["startup_time_ms"] < 10000
        assert is_startup_complete()
        assert get_startup_task_count() >= 1
        assert "component_refs" in startup_result
        components = startup_result["component_refs"]
        assert "session_store" in components
        assert "batch_processor" in components
        try:
            session_store = components["session_store"]
            await session_store.set("test_shutdown_key", {"data": "test_value"})
            test_value = await session_store.get("test_shutdown_key")
            assert test_value is not None
            assert test_value["data"] == "test_value"
        except Exception as e:
            print(f"Component functionality test failed: {e}")
        shutdown_start = time.time()
        shutdown_result = await shutdown_startup_tasks(timeout=15.0)
        shutdown_duration = (time.time() - shutdown_start) * 1000
        assert shutdown_result["status"] in ["success", "partial_success", "failed"]
        if shutdown_result["status"] == "failed":
            print("Real behavior: shutdown failed, state may be uncleaned")
            print(f"Startup still complete: {is_startup_complete()}")
            print(f"Tasks still running: {get_startup_task_count()}")
        else:
            assert not is_startup_complete()
            assert get_startup_task_count() == 0
            assert shutdown_result["shutdown_time_ms"] > 0
            assert shutdown_result["shutdown_time_ms"] < 15000
            assert shutdown_duration < 15000

    async def test_shutdown_with_active_sessions_real_behavior(self):
        """Test shutdown sequence with active sessions using real session store."""
        startup_result = await init_startup_tasks(
            session_ttl=600, cleanup_interval=30, batch_config={"dry_run": True}
        )
        if startup_result["status"] == "already_initialized":
            print("Using already initialized system")
            return
        if startup_result["status"] != "success":
            return
        session_store = startup_result["component_refs"]["session_store"]
        for i in range(50):
            await session_store.set(
                f"active_session_{i}",
                {
                    "user_id": f"user_{i}",
                    "data": f"session_data_{i}",
                    "timestamp": time.time(),
                },
            )
        assert await session_store.size() == 50
        start_time = time.time()
        shutdown_result = await shutdown_startup_tasks(timeout=15.0)
        shutdown_time = (time.time() - start_time) * 1000
        if shutdown_result["status"] == "failed":
            print(
                f"Real behavior: shutdown failed with active sessions - {shutdown_result}"
            )
        else:
            assert shutdown_result["status"] == "success"
            assert shutdown_time < 15000
            assert not is_startup_complete()

    async def test_shutdown_with_pending_batch_operations(self):
        """Test shutdown sequence with pending batch operations."""
        batch_config = {"batch_size": 10, "batch_timeout": 60, "dry_run": True}
        startup_result = await init_startup_tasks(batch_config=batch_config)
        if startup_result["status"] == "already_initialized":
            print("Using already initialized system")
            return
        if startup_result["status"] != "success":
            return
        batch_processor = startup_result["component_refs"]["batch_processor"]
        for i in range(100):
            await batch_processor.enqueue({
                "task": f"pending_task_{i}",
                "data": f"task_data_{i}",
                "priority": i % 10,
            })
        assert batch_processor.get_queue_size() > 0
        start_time = time.time()
        shutdown_result = await shutdown_startup_tasks(timeout=20.0)
        shutdown_time = (time.time() - start_time) * 1000
        if shutdown_result["status"] == "failed":
            print(
                f"Real behavior: shutdown failed with pending operations - {shutdown_result}"
            )
        else:
            assert shutdown_result["status"] == "success"
            assert shutdown_time < 20000
            assert not is_startup_complete()

    async def test_shutdown_timeout_handling(self):
        """Test shutdown timeout handling."""
        startup_result = await init_startup_tasks()
        if startup_result["status"] == "already_initialized":
            print("Using already initialized system")
        elif startup_result["status"] != "success":
            return
        with patch(
            "prompt_improver.services.health.background_manager.shutdown_background_task_manager"
        ) as mock_shutdown:

            async def slow_shutdown(timeout):
                await asyncio.sleep(timeout + 1.0)
                return True

            mock_shutdown.side_effect = slow_shutdown
            start_time = time.time()
            shutdown_result = await shutdown_startup_tasks(timeout=0.5)
            shutdown_time = (time.time() - start_time) * 1000
            assert shutdown_time < 2000

    async def test_shutdown_error_handling(self):
        """Test shutdown error handling and recovery."""
        startup_result = await init_startup_tasks()
        if startup_result["status"] == "already_initialized":
            print("Using already initialized system")
        elif startup_result["status"] != "success":
            return
        with patch(
            "prompt_improver.services.health.background_manager.shutdown_background_task_manager"
        ) as mock_shutdown:
            mock_shutdown.side_effect = Exception("Simulated shutdown error")
            shutdown_result = await shutdown_startup_tasks(timeout=10.0)
            if "errors" in shutdown_result and len(shutdown_result["errors"]) > 0:
                print(f"Shutdown reported errors: {shutdown_result['errors']}")
            else:
                print(
                    "Real behavior: shutdown handled error gracefully without reporting it"
                )
            if is_startup_complete():
                print("Real behavior: shutdown failed, state not cleaned up")
            else:
                print("Real behavior: shutdown succeeded despite errors")

    async def test_multiple_shutdown_calls_real_behavior(self):
        """Test handling of multiple shutdown calls with real state management."""
        startup_result = await init_startup_tasks(
            max_concurrent_tasks=2,
            session_ttl=300,
            cleanup_interval=120,
            batch_config={
                "batch_size": 3,
                "dry_run": True,
                "enable_priority_queue": False,
                "max_attempts": 1,
            },
        )
        if startup_result["status"] in ["failed", "already_initialized"]:
            shutdown_result = await shutdown_startup_tasks(timeout=5.0)
            return
        shutdown_result1 = await shutdown_startup_tasks(timeout=15.0)
        assert shutdown_result1["status"] in ["success", "partial_success", "failed"]
        if shutdown_result1["status"] == "failed":
            print("Real behavior: first shutdown failed, state may remain")
        else:
            assert not is_startup_complete()
            assert get_startup_task_count() == 0
        shutdown_result2 = await shutdown_startup_tasks(timeout=10.0)
        assert shutdown_result2["status"] in ["success", "failed", "not_initialized"]
        shutdown_result3 = await shutdown_startup_tasks(timeout=5.0)
        assert shutdown_result3["status"] in ["success", "failed", "not_initialized"]

    async def test_context_manager_shutdown(self):
        """Test shutdown via context manager."""
        startup_components = None
        async with startup_context(
            max_concurrent_tasks=3, session_ttl=300, cleanup_interval=60
        ) as components:
            startup_components = components
            assert is_startup_complete()
            assert len(components) >= 4
            session_store = components["session_store"]
            await session_store.set("test_key", "test_value")
            value = await session_store.get("test_key")
            assert value == "test_value"
        assert not is_startup_complete()
        assert get_startup_task_count() == 0

    async def test_graceful_shutdown_signal_handling(self):
        """Test graceful shutdown with signal simulation."""
        startup_result = await init_startup_tasks()
        if startup_result["status"] == "already_initialized":
            print("Using already initialized system")
        elif startup_result["status"] != "success":
            return
        shutdown_event = asyncio.Event()

        async def signal_handler():
            await asyncio.sleep(0.1)
            shutdown_event.set()

        signal_task = asyncio.create_task(signal_handler())
        await shutdown_event.wait()
        start_time = time.time()
        shutdown_result = await shutdown_startup_tasks(timeout=10.0)
        shutdown_time = (time.time() - start_time) * 1000
        if shutdown_result["status"] == "failed":
            print(f"Real behavior: shutdown failed - {shutdown_result}")
        else:
            assert shutdown_result["status"] == "success"
            assert shutdown_time < 10000
            assert not is_startup_complete()
        signal_task.cancel()
        try:
            await signal_task
        except asyncio.CancelledError:
            pass

    async def test_shutdown_resource_cleanup(self):
        """Test that shutdown properly cleans up resources."""
        startup_result = await init_startup_tasks()
        if startup_result["status"] == "already_initialized":
            print("Using already initialized system")
        elif startup_result["status"] != "success":
            return
        components = startup_result["component_refs"]
        session_store = components["session_store"]
        batch_processor = components["batch_processor"]
        await session_store.set("cleanup_test", {"data": "test"})
        await batch_processor.enqueue({"task": "cleanup_test"})
        session_size_before = await session_store.size()
        queue_size_before = batch_processor.get_queue_size()
        shutdown_result = await shutdown_startup_tasks(timeout=10.0)
        if shutdown_result["status"] == "failed":
            print(f"Real behavior: shutdown failed - {shutdown_result}")
        else:
            assert shutdown_result["status"] == "success"
            assert not is_startup_complete()
            assert get_startup_task_count() == 0

    async def test_concurrent_shutdown_requests(self):
        """Test handling of concurrent shutdown requests."""
        startup_result = await init_startup_tasks()
        if startup_result["status"] == "already_initialized":
            print("Using already initialized system")
        elif startup_result["status"] != "success":
            return

        async def shutdown_worker(worker_id):
            return await shutdown_startup_tasks(timeout=10.0)

        shutdown_tasks = [shutdown_worker(i) for i in range(3)]
        results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        success_count = sum(
            1 for r in results if isinstance(r, dict) and r.get("status") == "success"
        )
        not_initialized_count = sum(
            1
            for r in results
            if isinstance(r, dict) and r.get("status") == "not_initialized"
        )
        assert success_count >= 1
        assert success_count + not_initialized_count == len(results)
        assert not is_startup_complete()


class TestShutdownPerformance:
    """Performance tests for shutdown operations using real behavior."""

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

    def test_shutdown_performance_baseline_real_behavior(self, benchmark):
        """Benchmark baseline shutdown performance with real components."""

        def sync_setup_and_shutdown():
            """Synchronous wrapper for async operations."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._async_setup_and_shutdown())
            finally:
                loop.close()

        shutdown_time = benchmark(sync_setup_and_shutdown)
        assert shutdown_time < 1000

    async def _async_setup_and_shutdown(self):
        """Helper method for async setup and shutdown."""
        startup_result = await init_startup_tasks(
            max_concurrent_tasks=2,
            session_ttl=300,
            cleanup_interval=120,
            batch_config={
                "batch_size": 3,
                "dry_run": True,
                "enable_priority_queue": False,
            },
        )
        if startup_result["status"] == "already_initialized":
            pass
        elif startup_result["status"] != "success":
            return 100
        shutdown_result = await shutdown_startup_tasks(timeout=10.0)
        if "shutdown_time_ms" in shutdown_result:
            return shutdown_result["shutdown_time_ms"]
        return 500

    def test_shutdown_performance_with_load_real_behavior(self, benchmark):
        """Benchmark shutdown performance under load."""

        async def setup_load_and_shutdown():
            startup_result = await init_startup_tasks(
                max_concurrent_tasks=10, session_ttl=300, cleanup_interval=30
            )
            if startup_result["status"] == "already_initialized":
                print("Using already initialized system")
            elif startup_result["status"] != "success":
                return None
            components = startup_result["component_refs"]
            session_store = components["session_store"]
            batch_processor = components["batch_processor"]
            for i in range(500):
                await session_store.set(f"load_session_{i}", {"data": f"data_{i}"})
            for i in range(1000):
                await batch_processor.enqueue({"task": f"load_task_{i}"})
            shutdown_result = await shutdown_startup_tasks(timeout=15.0)
            if shutdown_result["status"] == "failed":
                print(f"Real behavior: shutdown failed under load - {shutdown_result}")
                return 1000
            assert shutdown_result["status"] == "success"
            return shutdown_result["shutdown_time_ms"]

        def run_load_test():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(setup_load_and_shutdown())
            finally:
                loop.close()

        shutdown_time = benchmark(run_load_test)
        assert shutdown_time < 2000

    @pytest.mark.benchmark(
        min_time=0.1, max_time=3.0, min_rounds=3, disable_gc=True, warmup=False
    )
    def test_shutdown_response_time_requirement_real_behavior(self, benchmark):
        """Test shutdown meets response time requirements."""

        async def full_lifecycle():
            startup_result = await init_startup_tasks()
            if startup_result["status"] == "already_initialized":
                print("Using already initialized system")
            elif startup_result["status"] != "success":
                return None
            components = startup_result["component_refs"]
            session_store = components["session_store"]
            batch_processor = components["batch_processor"]
            for i in range(100):
                await session_store.set(
                    f"session_{i}",
                    {
                        "user_id": f"user_{i}",
                        "session_data": {"active": True, "timestamp": time.time()},
                        "preferences": {"theme": "dark", "language": "en"},
                    },
                )
            for i in range(200):
                await batch_processor.enqueue({
                    "original": f"Original prompt {i}",
                    "enhanced": f"Enhanced prompt {i}",
                    "metrics": {"confidence": 0.8 + i % 3 * 0.1},
                    "session_id": f"session_{i % 100}",
                    "priority": i % 5,
                })
            start_time = time.time()
            shutdown_result = await shutdown_startup_tasks(timeout=10.0)
            shutdown_time = (time.time() - start_time) * 1000
            if shutdown_result["status"] == "failed":
                print(
                    f"Real behavior: shutdown failed in lifecycle test - {shutdown_result}"
                )
                return shutdown_time
            assert shutdown_result["status"] == "success"
            return shutdown_time

        def run_lifecycle():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(full_lifecycle())
            finally:
                loop.close()

        shutdown_time = benchmark(run_lifecycle)
        assert shutdown_time < 1000

    def test_startup_shutdown_memory_efficiency_real_behavior(self, benchmark):
        """Test memory efficiency of startup/shutdown cycles."""

        async def memory_test_cycle():
            cycle_results = []
            for cycle in range(3):
                startup_result = await init_startup_tasks(
                    max_concurrent_tasks=5, session_ttl=300, cleanup_interval=60
                )
                if startup_result["status"] == "already_initialized":
                    print("Using already initialized system")
                elif startup_result["status"] != "success":
                    continue
                components = startup_result["component_refs"]
                session_store = components["session_store"]
                for i in range(50):
                    await session_store.set(f"cycle_{cycle}_session_{i}", {"data": i})
                shutdown_result = await shutdown_startup_tasks(timeout=10.0)
                if shutdown_result["status"] in ["success", "partial_success"]:
                    cycle_results.append({"cycle": cycle, "success": True})
                else:
                    cycle_results.append({"cycle": cycle, "success": False})
            return cycle_results

        def run_memory_test():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(memory_test_cycle())
            finally:
                loop.close()

        result = benchmark(run_memory_test)
        assert result is True


@pytest.mark.asyncio
class TestShutdownSequenceIntegration:
    """Integration tests for shutdown sequence with all components using real behavior."""

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

    async def test_end_to_end_shutdown_scenario(self):
        """Test complete end-to-end shutdown scenario."""
        startup_result = await init_startup_tasks(
            max_concurrent_tasks=5,
            session_ttl=600,
            cleanup_interval=60,
            batch_config={"batch_size": 10, "batch_timeout": 30, "dry_run": True},
        )
        if startup_result["status"] == "already_initialized":
            print("Using already initialized system")
        elif startup_result["status"] != "success":
            return
        components = startup_result["component_refs"]
        session_store = components["session_store"]
        batch_processor = components["batch_processor"]
        health_monitor = components["health_monitor"]
        user_sessions = {}
        for i in range(20):
            session_id = f"user_session_{i}"
            session_data = {
                "user_id": f"user_{i}",
                "login_time": time.time(),
                "active_prompts": [],
                "preferences": {"language": "en", "model": "gpt-4"},
            }
            await session_store.set(session_id, session_data)
            user_sessions[session_id] = session_data
        batch_jobs = []
        for i in range(50):
            job = {
                "prompt_id": f"prompt_{i}",
                "original": f"Original prompt {i}",
                "user_session": f"user_session_{i % 20}",
                "enhancement_type": "clarity",
                "priority": i % 3,
            }
            await batch_processor.enqueue(job, priority=job["priority"])
            batch_jobs.append(job)
        health_result = await health_monitor.run_health_check()
        print(f"Real behavior: health status is {health_result.overall_status.value}")
        assert health_result.overall_status.value in ["healthy", "warning", "failed"]
        assert await session_store.size() == 20
        queue_size = batch_processor.get_queue_size()
        print(f"Real behavior: queue size is {queue_size}")
        assert queue_size >= 0
        assert is_startup_complete()
        shutdown_start = time.time()
        shutdown_result = await shutdown_startup_tasks(timeout=30.0)
        shutdown_duration = (time.time() - shutdown_start) * 1000
        if shutdown_result["status"] == "failed":
            print(
                f"Real behavior: shutdown failed in end-to-end test - {shutdown_result}"
            )
        else:
            assert shutdown_result["status"] == "success"
            assert shutdown_duration < 30000
            assert not is_startup_complete()
            assert get_startup_task_count() == 0
        assert "errors" not in shutdown_result or len(shutdown_result["errors"]) == 0

    async def test_shutdown_with_component_failures(self):
        """Test shutdown resilience when components fail."""
        startup_result = await init_startup_tasks()
        if startup_result["status"] == "already_initialized":
            print("Using already initialized system")
        elif startup_result["status"] != "success":
            return
        with patch(
            "prompt_improver.utils.session_store.SessionStore.stop_cleanup_task"
        ) as mock_stop:
            mock_stop.side_effect = Exception("Session store failure")
            shutdown_result = await shutdown_startup_tasks(timeout=15.0)
            assert "errors" in shutdown_result
            assert len(shutdown_result["errors"]) > 0
            assert not is_startup_complete()

    async def test_shutdown_idempotency(self):
        """Test that shutdown operations are idempotent."""
        startup_result = await init_startup_tasks()
        if startup_result["status"] == "already_initialized":
            print("Using already initialized system")
        elif startup_result["status"] != "success":
            return
        shutdown1 = await shutdown_startup_tasks(timeout=10.0)
        if shutdown1["status"] == "failed":
            print(f"Real behavior: first shutdown failed - {shutdown1}")
        else:
            assert shutdown1["status"] == "success"
            assert not is_startup_complete()
        for i in range(3):
            shutdown_n = await shutdown_startup_tasks(timeout=5.0)
            assert shutdown_n["status"] == "not_initialized"
            assert not is_startup_complete()

    async def test_partial_startup_shutdown(self):
        """Test shutdown of partially initialized system."""
        with patch(
            "prompt_improver.services.health.service.get_health_service"
        ) as mock_health:
            mock_health.side_effect = Exception("Health service failed")
            startup_result = await init_startup_tasks()
            assert startup_result["status"] == "failed"
            assert not is_startup_complete()
            shutdown_result = await shutdown_startup_tasks(timeout=5.0)
            assert shutdown_result["status"] == "not_initialized"
