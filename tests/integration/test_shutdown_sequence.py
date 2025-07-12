"""
Integration tests for graceful shutdown sequence and component lifecycle management.

Tests cover startup/shutdown coordination, resource cleanup, timeout handling,
and performance requirements for shutdown operations.
"""

import asyncio
import pytest
import time
import logging
import signal
import os
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager

from prompt_improver.services.startup import (
    init_startup_tasks,
    shutdown_startup_tasks,
    startup_context,
    is_startup_complete,
    get_startup_task_count
)
from prompt_improver.utils.session_store import SessionStore
from prompt_improver.optimization.batch_processor import BatchProcessor


@pytest.mark.asyncio
class TestShutdownSequence:
    """Integration tests for graceful shutdown sequence."""
    
    async def test_basic_startup_shutdown_cycle(self):
        """Test basic startup and shutdown cycle."""
        # Ensure clean state
        assert not is_startup_complete()
        assert get_startup_task_count() == 0
        
        # Startup
        startup_result = await init_startup_tasks(
            max_concurrent_tasks=5,
            session_ttl=300,
            cleanup_interval=60
        )
        
        assert startup_result["status"] == "success"
        assert startup_result["startup_time_ms"] > 0
        assert startup_result["startup_time_ms"] < 5000  # < 5 seconds
        assert is_startup_complete()
        assert get_startup_task_count() >= 2
        
        # Shutdown
        shutdown_result = await shutdown_startup_tasks(timeout=10.0)
        
        assert shutdown_result["status"] == "success"
        assert shutdown_result["shutdown_time_ms"] > 0
        assert shutdown_result["shutdown_time_ms"] < 10000  # < 10 seconds
        assert not is_startup_complete()
        assert get_startup_task_count() == 0
    
    async def test_shutdown_with_active_sessions(self):
        """Test shutdown sequence with active sessions."""
        startup_result = await init_startup_tasks()
        assert startup_result["status"] == "success"
        
        # Get session store from components
        session_store = startup_result["component_refs"]["session_store"]
        
        # Add active sessions
        for i in range(50):
            await session_store.set(f"active_session_{i}", {
                "user_id": f"user_{i}",
                "data": f"session_data_{i}",
                "timestamp": time.time()
            })
        
        # Verify sessions exist
        assert await session_store.size() == 50
        
        # Shutdown with active sessions
        start_time = time.time()
        shutdown_result = await shutdown_startup_tasks(timeout=15.0)
        shutdown_time = (time.time() - start_time) * 1000
        
        assert shutdown_result["status"] == "success"
        assert shutdown_time < 15000  # Should complete within timeout
        assert not is_startup_complete()
    
    async def test_shutdown_with_pending_batch_operations(self):
        """Test shutdown sequence with pending batch operations."""
        batch_config = {
            "batch_size": 10,
            "batch_timeout": 60,  # Long timeout for testing
            "dry_run": True
        }
        
        startup_result = await init_startup_tasks(batch_config=batch_config)
        assert startup_result["status"] == "success"
        
        # Get batch processor from components
        batch_processor = startup_result["component_refs"]["batch_processor"]
        
        # Add pending batch operations
        for i in range(100):
            await batch_processor.enqueue({
                "task": f"pending_task_{i}",
                "data": f"task_data_{i}",
                "priority": i % 10
            })
        
        # Verify tasks are queued
        assert batch_processor.get_queue_size() > 0
        
        # Shutdown with pending operations
        start_time = time.time()
        shutdown_result = await shutdown_startup_tasks(timeout=20.0)
        shutdown_time = (time.time() - start_time) * 1000
        
        assert shutdown_result["status"] == "success"
        assert shutdown_time < 20000  # Should complete within timeout
        assert not is_startup_complete()
    
    async def test_shutdown_timeout_handling(self):
        """Test shutdown timeout handling."""
        startup_result = await init_startup_tasks()
        assert startup_result["status"] == "success"
        
        # Mock a component that doesn't shutdown quickly
        with patch('prompt_improver.services.health.background_manager.shutdown_background_task_manager') as mock_shutdown:
            # Make shutdown take longer than timeout
            async def slow_shutdown(timeout):
                await asyncio.sleep(timeout + 1.0)
                return True
            
            mock_shutdown.side_effect = slow_shutdown
            
            # Shutdown with short timeout
            start_time = time.time()
            shutdown_result = await shutdown_startup_tasks(timeout=0.5)
            shutdown_time = (time.time() - start_time) * 1000
            
            # Should handle timeout gracefully
            assert shutdown_time < 2000  # Should not hang indefinitely
            # Note: Status might be "failed" due to timeout, but system should remain stable
    
    async def test_shutdown_error_handling(self):
        """Test shutdown error handling and recovery."""
        startup_result = await init_startup_tasks()
        assert startup_result["status"] == "success"
        
        # Mock component shutdown to raise an exception
        with patch('prompt_improver.services.health.background_manager.shutdown_background_task_manager') as mock_shutdown:
            mock_shutdown.side_effect = Exception("Simulated shutdown error")
            
            # Shutdown should handle errors gracefully
            shutdown_result = await shutdown_startup_tasks(timeout=10.0)
            
            # Should report errors but continue shutdown
            assert "errors" in shutdown_result
            assert len(shutdown_result["errors"]) > 0
            assert not is_startup_complete()  # Should still mark as shutdown
    
    async def test_multiple_shutdown_calls(self):
        """Test handling of multiple shutdown calls."""
        startup_result = await init_startup_tasks()
        assert startup_result["status"] == "success"
        
        # First shutdown
        shutdown_result1 = await shutdown_startup_tasks(timeout=10.0)
        assert shutdown_result1["status"] == "success"
        assert not is_startup_complete()
        
        # Second shutdown (should handle gracefully)
        shutdown_result2 = await shutdown_startup_tasks(timeout=10.0)
        assert shutdown_result2["status"] == "not_initialized"
        assert not is_startup_complete()
    
    async def test_context_manager_shutdown(self):
        """Test shutdown via context manager."""
        startup_components = None
        
        # Use context manager
        async with startup_context(
            max_concurrent_tasks=3,
            session_ttl=300,
            cleanup_interval=60
        ) as components:
            startup_components = components
            
            # Verify startup
            assert is_startup_complete()
            assert len(components) >= 4  # Should have all components
            
            # Use components
            session_store = components["session_store"]
            await session_store.set("test_key", "test_value")
            
            value = await session_store.get("test_key")
            assert value == "test_value"
        
        # Should be automatically shut down
        assert not is_startup_complete()
        assert get_startup_task_count() == 0
    
    async def test_graceful_shutdown_signal_handling(self):
        """Test graceful shutdown with signal simulation."""
        startup_result = await init_startup_tasks()
        assert startup_result["status"] == "success"
        
        # Simulate receiving shutdown signal
        shutdown_event = asyncio.Event()
        
        async def signal_handler():
            # Simulate signal processing delay
            await asyncio.sleep(0.1)
            shutdown_event.set()
        
        # Start signal handler
        signal_task = asyncio.create_task(signal_handler())
        
        # Wait for signal
        await shutdown_event.wait()
        
        # Perform graceful shutdown
        start_time = time.time()
        shutdown_result = await shutdown_startup_tasks(timeout=10.0)
        shutdown_time = (time.time() - start_time) * 1000
        
        assert shutdown_result["status"] == "success"
        assert shutdown_time < 10000
        assert not is_startup_complete()
        
        # Clean up signal task
        signal_task.cancel()
        try:
            await signal_task
        except asyncio.CancelledError:
            pass
    
    async def test_shutdown_resource_cleanup(self):
        """Test that shutdown properly cleans up resources."""
        startup_result = await init_startup_tasks()
        assert startup_result["status"] == "success"
        
        # Get component references
        components = startup_result["component_refs"]
        session_store = components["session_store"]
        batch_processor = components["batch_processor"]
        
        # Create resources to cleanup
        await session_store.set("cleanup_test", {"data": "test"})
        await batch_processor.enqueue({"task": "cleanup_test"})
        
        # Track resource states before shutdown
        session_size_before = await session_store.size()
        queue_size_before = batch_processor.get_queue_size()
        
        # Shutdown
        shutdown_result = await shutdown_startup_tasks(timeout=10.0)
        assert shutdown_result["status"] == "success"
        
        # Verify cleanup completed
        assert not is_startup_complete()
        assert get_startup_task_count() == 0
        
        # Note: Resources may persist after shutdown (sessions in memory, etc.)
        # The important thing is that cleanup tasks are stopped
    
    async def test_concurrent_shutdown_requests(self):
        """Test handling of concurrent shutdown requests."""
        startup_result = await init_startup_tasks()
        assert startup_result["status"] == "success"
        
        # Launch multiple concurrent shutdown requests
        async def shutdown_worker(worker_id):
            return await shutdown_startup_tasks(timeout=10.0)
        
        # Start multiple shutdown tasks
        shutdown_tasks = [
            shutdown_worker(i) for i in range(3)
        ]
        
        # Wait for all to complete
        results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        # Should handle concurrent requests gracefully
        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        not_initialized_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "not_initialized")
        
        # At least one should succeed, others should be "not_initialized"
        assert success_count >= 1
        assert success_count + not_initialized_count == len(results)
        assert not is_startup_complete()


@pytest.mark.asyncio
class TestShutdownPerformance:
    """Performance tests for shutdown operations."""
    
    async def test_shutdown_performance_baseline(self, benchmark):
        """Benchmark baseline shutdown performance."""
        async def setup_and_shutdown():
            # Setup
            startup_result = await init_startup_tasks(
                max_concurrent_tasks=3,
                session_ttl=300,
                cleanup_interval=60
            )
            assert startup_result["status"] == "success"
            
            # Shutdown
            shutdown_result = await shutdown_startup_tasks(timeout=10.0)
            assert shutdown_result["status"] == "success"
            
            return shutdown_result["shutdown_time_ms"]
        
        def run_test():
            return asyncio.run(setup_and_shutdown())
        
        # Benchmark the operation
        shutdown_time = benchmark(run_test)
        
        # Performance requirement: shutdown should complete in < 200ms for basic case
        assert shutdown_time < 200
    
    async def test_shutdown_performance_with_load(self, benchmark):
        """Benchmark shutdown performance under load."""
        async def setup_load_and_shutdown():
            # Setup with load
            startup_result = await init_startup_tasks(
                max_concurrent_tasks=10,
                session_ttl=300,
                cleanup_interval=30
            )
            assert startup_result["status"] == "success"
            
            # Create load
            components = startup_result["component_refs"]
            session_store = components["session_store"]
            batch_processor = components["batch_processor"]
            
            # Add many sessions
            for i in range(500):
                await session_store.set(f"load_session_{i}", {"data": f"data_{i}"})
            
            # Add many batch tasks
            for i in range(1000):
                await batch_processor.enqueue({"task": f"load_task_{i}"})
            
            # Shutdown under load
            shutdown_result = await shutdown_startup_tasks(timeout=15.0)
            assert shutdown_result["status"] == "success"
            
            return shutdown_result["shutdown_time_ms"]
        
        def run_load_test():
            return asyncio.run(setup_load_and_shutdown())
        
        # Benchmark the operation
        shutdown_time = benchmark(run_load_test)
        
        # Performance requirement: shutdown should complete in reasonable time even under load
        assert shutdown_time < 2000  # < 2 seconds under load
    
    @pytest.mark.benchmark(
        min_time=0.1,
        max_time=2.0,
        min_rounds=3,
        disable_gc=True,
        warmup=False
    )
    async def test_shutdown_response_time_requirement(self, benchmark):
        """Test shutdown meets response time requirements."""
        async def full_lifecycle():
            # Startup
            startup_result = await init_startup_tasks()
            assert startup_result["status"] == "success"
            
            # Add realistic workload
            components = startup_result["component_refs"]
            session_store = components["session_store"]
            batch_processor = components["batch_processor"]
            
            # Realistic session load
            for i in range(100):
                await session_store.set(f"session_{i}", {
                    "user_id": f"user_{i}",
                    "session_data": {"active": True, "timestamp": time.time()},
                    "preferences": {"theme": "dark", "language": "en"}
                })
            
            # Realistic batch load
            for i in range(200):
                await batch_processor.enqueue({
                    "original": f"Original prompt {i}",
                    "enhanced": f"Enhanced prompt {i}",
                    "metrics": {"confidence": 0.8 + (i % 3) * 0.1},
                    "session_id": f"session_{i % 100}",
                    "priority": i % 5
                })
            
            # Measure shutdown time
            start_time = time.time()
            shutdown_result = await shutdown_startup_tasks(timeout=10.0)
            shutdown_time = (time.time() - start_time) * 1000
            
            assert shutdown_result["status"] == "success"
            return shutdown_time
        
        def run_lifecycle():
            return asyncio.run(full_lifecycle())
        
        # Benchmark the operation
        shutdown_time = benchmark(run_lifecycle)
        
        # Critical performance requirement: shutdown < 1 second
        assert shutdown_time < 1000
    
    async def test_startup_shutdown_memory_efficiency(self, benchmark):
        """Test memory efficiency of startup/shutdown cycles."""
        async def memory_test_cycle():
            # Multiple startup/shutdown cycles
            for cycle in range(5):
                # Startup
                startup_result = await init_startup_tasks(
                    max_concurrent_tasks=5,
                    session_ttl=300,
                    cleanup_interval=60
                )
                assert startup_result["status"] == "success"
                
                # Add some data
                components = startup_result["component_refs"]
                session_store = components["session_store"]
                
                for i in range(50):
                    await session_store.set(f"cycle_{cycle}_session_{i}", {"data": i})
                
                # Shutdown
                shutdown_result = await shutdown_startup_tasks(timeout=10.0)
                assert shutdown_result["status"] == "success"
            
            return True
        
        def run_memory_test():
            return asyncio.run(memory_test_cycle())
        
        # Should handle multiple cycles efficiently
        result = benchmark(run_memory_test)
        assert result is True


@pytest.mark.asyncio
class TestShutdownSequenceIntegration:
    """Integration tests for shutdown sequence with all components."""
    
    async def test_end_to_end_shutdown_scenario(self):
        """Test complete end-to-end shutdown scenario."""
        # Startup all components
        startup_result = await init_startup_tasks(
            max_concurrent_tasks=5,
            session_ttl=600,
            cleanup_interval=60,
            batch_config={
                "batch_size": 10,
                "batch_timeout": 30,
                "dry_run": True
            }
        )
        assert startup_result["status"] == "success"
        
        # Get all components
        components = startup_result["component_refs"]
        session_store = components["session_store"]
        batch_processor = components["batch_processor"]
        health_service = components["health_service"]
        
        # Simulate realistic application usage
        # 1. Create user sessions
        user_sessions = {}
        for i in range(20):
            session_id = f"user_session_{i}"
            session_data = {
                "user_id": f"user_{i}",
                "login_time": time.time(),
                "active_prompts": [],
                "preferences": {"language": "en", "model": "gpt-4"}
            }
            await session_store.set(session_id, session_data)
            user_sessions[session_id] = session_data
        
        # 2. Process batch operations
        batch_jobs = []
        for i in range(50):
            job = {
                "prompt_id": f"prompt_{i}",
                "original": f"Original prompt {i}",
                "user_session": f"user_session_{i % 20}",
                "enhancement_type": "clarity",
                "priority": i % 3
            }
            await batch_processor.enqueue(job, priority=job["priority"])
            batch_jobs.append(job)
        
        # 3. Check system health
        health_result = await health_service.run_health_check()
        assert health_result.overall_status.value in ["healthy", "warning"]
        
        # 4. Verify system state before shutdown
        assert await session_store.size() == 20
        assert batch_processor.get_queue_size() > 0
        assert is_startup_complete()
        
        # 5. Initiate graceful shutdown
        shutdown_start = time.time()
        shutdown_result = await shutdown_startup_tasks(timeout=30.0)
        shutdown_duration = (time.time() - shutdown_start) * 1000
        
        # 6. Verify shutdown completed successfully
        assert shutdown_result["status"] == "success"
        assert shutdown_duration < 30000  # Within timeout
        assert not is_startup_complete()
        assert get_startup_task_count() == 0
        
        # 7. Verify clean shutdown state
        assert "errors" not in shutdown_result or len(shutdown_result["errors"]) == 0
    
    async def test_shutdown_with_component_failures(self):
        """Test shutdown resilience when components fail."""
        startup_result = await init_startup_tasks()
        assert startup_result["status"] == "success"
        
        # Simulate component failures during shutdown
        with patch('prompt_improver.utils.session_store.SessionStore.stop_cleanup_task') as mock_stop:
            mock_stop.side_effect = Exception("Session store failure")
            
            # Shutdown should continue despite component failures
            shutdown_result = await shutdown_startup_tasks(timeout=15.0)
            
            # Should report errors but complete shutdown
            assert "errors" in shutdown_result
            assert len(shutdown_result["errors"]) > 0
            assert not is_startup_complete()  # Should still mark as shutdown
    
    async def test_shutdown_idempotency(self):
        """Test that shutdown operations are idempotent."""
        startup_result = await init_startup_tasks()
        assert startup_result["status"] == "success"
        
        # First shutdown
        shutdown1 = await shutdown_startup_tasks(timeout=10.0)
        assert shutdown1["status"] == "success"
        assert not is_startup_complete()
        
        # Multiple subsequent shutdowns should be safe
        for i in range(3):
            shutdown_n = await shutdown_startup_tasks(timeout=5.0)
            assert shutdown_n["status"] == "not_initialized"
            assert not is_startup_complete()
    
    async def test_partial_startup_shutdown(self):
        """Test shutdown of partially initialized system."""
        # This test simulates startup failure and subsequent cleanup
        with patch('prompt_improver.services.health.service.get_health_service') as mock_health:
            mock_health.side_effect = Exception("Health service failed")
            
            # Startup should fail
            startup_result = await init_startup_tasks()
            assert startup_result["status"] == "failed"
            assert not is_startup_complete()
            
            # Shutdown of failed startup should be safe
            shutdown_result = await shutdown_startup_tasks(timeout=5.0)
            assert shutdown_result["status"] == "not_initialized"
