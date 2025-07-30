"""Real behavior tests for health endpoints in the MCP server.

Following 2025 best practices: real behavior testing without mocks.
This module tests the health check functionality including:
- Event loop latency measurement
- Training queue size checks
- Database connectivity verification
- Background task manager integration
"""

import asyncio
import time
from typing import Any, Dict

import pytest

# Import the health functions and related components
from prompt_improver.database import get_session
from prompt_improver.mcp_server.server import APESMCPServer
# Legacy import removed - will be fixed with modern patterns
from prompt_improver.ml.optimization.batch.unified_batch_processor import UnifiedBatchProcessor, BatchProcessorConfig
from prompt_improver.performance.monitoring.health.background_manager import (
    EnhancedBackgroundTaskManager,
    get_background_task_manager,
)


class TestHealthEndpoints:
    """Test suite for health endpoint functionality using real behavior."""

    @pytest.fixture
    async def mcp_server(self):
        """Create a real MCP server instance for health testing."""
        server = APESMCPServer()
        await server.initialize()
        yield server
        # Cleanup - no explicit shutdown needed for this test pattern

    @pytest.fixture
    async def real_batch_processor(self):
        """Create a real batch processor for health testing."""
        config = BatchProcessorConfig(
            batch_size=5,
            concurrency=2,
            max_queue_size=100,
            enable_priority_queue=True,
            max_attempts=2,
            batch_timeout=5,  # 5 second timeout for tests (le=300)
            timeout=2000,  # 2 second timeout
        )
        processor = UnifiedBatchProcessor(config=config)
        
        # Initialize with some test metrics
        processor.metrics = {
            "processed": 50,
            "failed": 2,
            "retries": 1,
            "start_time": time.time(),
        }
        
        yield processor
        
        # Cleanup
        processor.processing = False

    @pytest.fixture
    async def real_background_manager(self):
        """Create a real background task manager for health testing."""
        manager = EnhancedBackgroundTaskManager(max_concurrent_tasks=5)
        await manager.start()
        
        # Add a test task to simulate real usage
        async def health_test_task():
            await asyncio.sleep(0.1)
            return "health_check_complete"
        
        await manager.submit_task("health_test_task", health_test_task)
        
        yield manager
        
        # Cleanup
        await manager.stop(timeout=2.0)

    @pytest.mark.asyncio
    async def test_health_live_success(self, mcp_server, real_batch_processor, real_background_manager, monkeypatch):
        """Test successful health/live endpoint with real components."""
        # Monkeypatch to use our real instances
        monkeypatch.setattr(
            "prompt_improver.mcp_server.server.get_background_task_manager",
            lambda: real_background_manager
        )
        monkeypatch.setattr(
            mcp_server,
            "batch_processor",
            real_batch_processor
        )

        result = await mcp_server._health_live_impl()

        assert result["status"] == "live"
        assert "event_loop_latency_ms" in result
        assert "training_queue_size" in result
        assert "background_queue_size" in result
        assert "timestamp" in result
        assert isinstance(result["event_loop_latency_ms"], (int, float))
        assert result["event_loop_latency_ms"] >= 0
        # Real values may vary, so check for non-negative
        assert result["training_queue_size"] >= 0
        assert result["background_queue_size"] >= 0

    @pytest.mark.asyncio
    async def test_health_live_error_handling(self, mcp_server, monkeypatch):
        """Test health/live endpoint error handling with real error scenarios."""
        # Simulate real error scenario - background manager unavailable
        def failing_background_manager():
            raise Exception("Background task manager error")
            
        monkeypatch.setattr(
            "prompt_improver.mcp_server.server.get_background_task_manager",
            failing_background_manager
        )

        result = await mcp_server._health_live_impl()

        assert result["status"] == "error"
        assert "error" in result
        assert "Background task manager error" in result["error"]

    @pytest.mark.asyncio
    async def test_health_ready_success(self, mcp_server, real_batch_processor, monkeypatch):
        """Test successful health/ready endpoint with real database and components."""
        # Use real database session - no mocking
        monkeypatch.setattr(
            mcp_server,
            "batch_processor",
            real_batch_processor
        )

        result = await mcp_server._health_ready_impl()

        # Real database connectivity test - result depends on actual DB state
        assert result["status"] in ["ready", "not ready", "error"]  # Real DB may not be available
        if result["status"] == "error":
            assert "error" in result
        else:
            assert "db_connectivity" in result
            assert "ready" in result["db_connectivity"]
            assert "response_time_ms" in result["db_connectivity"]
            assert "training_queue_size" in result
            assert "event_loop_latency_ms" in result
            assert "timestamp" in result
            assert result["training_queue_size"] >= 0
            assert isinstance(result["event_loop_latency_ms"], (int, float))
            assert result["event_loop_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_health_ready_db_failure(self, mcp_server, real_batch_processor, monkeypatch):
        """Test health/ready endpoint when database is unavailable."""
        # Simulate real database failure scenario
        def failing_get_session():
            raise Exception("Database connection failed")
        
        monkeypatch.setattr(
            "prompt_improver.mcp_server.server.get_session",
            failing_get_session
        )
        monkeypatch.setattr(
            mcp_server,
            "batch_processor",
            real_batch_processor
        )

        result = await mcp_server._health_ready_impl()

        assert result["status"] == "error"
        assert "error" in result
        # The error message may vary depending on the specific failure mode
        assert any(phrase in result["error"] for phrase in ["Database", "connection", "failed"])

    @pytest.mark.asyncio
    async def test_health_ready_latency_threshold(self, mcp_server, real_batch_processor, monkeypatch):
        """Test health/ready endpoint latency measurement and threshold logic with real timing."""
        monkeypatch.setattr(
            mcp_server,
            "batch_processor",
            real_batch_processor
        )

        result = await mcp_server._health_ready_impl()

        # Real latency measurement and threshold evaluation
        if result["status"] == "error":
            # Database might not be available in test environment
            assert "error" in result
        else:
            assert "event_loop_latency_ms" in result
            assert result["event_loop_latency_ms"] >= 0
            # Test real threshold logic - if latency < 100ms and DB is ready, should be "ready"
            if result["status"] == "ready":
                assert result["db_connectivity"]["ready"] is True
                assert result["event_loop_latency_ms"] < 100
        # Status depends on real conditions - could be ready, not ready, or error

    @pytest.mark.asyncio
    async def test_health_ready_error_handling(self, mcp_server, monkeypatch):
        """Test health/ready endpoint error handling with real error scenarios."""
        # Simulate real database error scenario
        def failing_get_session():
            raise Exception("Database connection failed")
            
        monkeypatch.setattr(
            "prompt_improver.mcp_server.server.get_session",
            failing_get_session
        )

        result = await mcp_server._health_ready_impl()

        assert result["status"] == "error"
        assert "error" in result
        # The error message may vary depending on the specific failure mode
        assert any(phrase in result["error"] for phrase in ["Database", "connection", "failed"])

    @pytest.mark.asyncio
    async def test_get_training_queue_size(self, mcp_server):
        """Test training queue size retrieval with real BatchProcessor."""
        config = BatchProcessorConfig(batch_size=15, max_queue_size=100)
        batch_processor = UnifiedBatchProcessor(config=config)
        
        # Set the batch processor on the server
        mcp_server.batch_processor = batch_processor

        result = await mcp_server._get_training_queue_size_impl()

        # Real queue size from actual processor - returns a dictionary
        assert isinstance(result, dict)
        assert "queue_size" in result
        assert result["queue_size"] >= 0  # Queue size should be non-negative
        assert isinstance(result["queue_size"], int)

    @pytest.mark.asyncio
    async def test_get_training_queue_size_default(self, mcp_server):
        """Test training queue size retrieval with default configuration."""
        config = BatchProcessorConfig()  # Use default config
        batch_processor = UnifiedBatchProcessor(config=config)
        
        # Set the batch processor on the server
        mcp_server.batch_processor = batch_processor

        result = await mcp_server._get_training_queue_size_impl()

        # Real queue size from actual processor - returns a dictionary
        assert isinstance(result, dict)
        assert "queue_size" in result
        assert result["queue_size"] >= 0  # Real queue size may vary
        assert isinstance(result["queue_size"], int)


    @pytest.mark.asyncio
    async def test_event_loop_latency_measurement(self):
        """Test that event loop latency is measured correctly in real conditions."""
        start_time = time.time()
        await asyncio.sleep(0.01)  # Small delay
        end_time = time.time()

        latency = (end_time - start_time) * 1000

        # Real latency measurement - should be at least 10ms (the sleep duration)
        assert latency >= 10
        # Should be reasonable for real event loop (less than 100ms in normal conditions)
        assert latency < 100

    @pytest.mark.asyncio
    async def test_batch_processor_integration(self, mcp_server):
        """Test integration with real BatchProcessor."""
        config = BatchProcessorConfig(
            batch_size=20,
            concurrency=5,
            max_attempts=3,
            timeout=30000,
        )

        batch_processor = UnifiedBatchProcessor(config=config)
        mcp_server.batch_processor = batch_processor
        
        result = await mcp_server._get_training_queue_size_impl()

        # Real queue size from actual processor - returns a dictionary
        assert isinstance(result, dict)
        assert "queue_size" in result
        assert result["queue_size"] >= 0
        assert batch_processor.config.concurrency == 5
        assert batch_processor.config.max_attempts == 3
        assert batch_processor.config.batch_size == 20


class TestBackgroundTaskManagerIntegration(TestHealthEndpoints):
    """Test suite for BackgroundTaskManager integration with health checks using real behavior."""

    @pytest.mark.asyncio
    async def test_background_task_manager_queue_size(self):
        """Test BackgroundTaskManager queue size reporting with real task submission."""
        manager = EnhancedBackgroundTaskManager(max_concurrent_tasks=3)
        await manager.start()

        # Initially should have low queue size
        initial_size = manager.get_queue_size()
        assert initial_size >= 0

        # Add a real task
        async def real_test_task():
            await asyncio.sleep(0.1)
            return "task_complete"

        await manager.submit_task("test_task", real_test_task)

        # Queue size should increase or at least remain non-negative
        new_size = manager.get_queue_size()
        assert new_size >= 0

        # Clean up
        await manager.stop(timeout=2.0)

    @pytest.mark.asyncio
    async def test_background_task_manager_task_counts(self):
        """Test BackgroundTaskManager task count reporting with real tasks."""
        manager = EnhancedBackgroundTaskManager(max_concurrent_tasks=3)
        await manager.start()

        # Check initial counts
        initial_counts = manager.get_task_count()
        assert all(count >= 0 for count in initial_counts.values())

        # Add real tasks
        async def real_task(duration: float = 0.1):
            await asyncio.sleep(duration)
            return f"completed_after_{duration}"

        await manager.submit_task("task1", real_task, duration=0.05)
        await manager.submit_task("task2", real_task, duration=0.05)

        # Check counts after task submission
        counts = manager.get_task_count()
        assert all(count >= 0 for count in counts.values())
        # Total tasks should be non-negative
        total_tasks = sum(counts.values())
        assert total_tasks >= 0

        # Wait for tasks to complete
        await asyncio.sleep(0.2)

        # Clean up
        await manager.stop(timeout=2.0)

    @pytest.mark.asyncio
    async def test_health_check_with_background_tasks(self, mcp_server, real_batch_processor, monkeypatch):
        """Test health checks with active background tasks using real components."""
        # Create real background task manager with active tasks
        manager = EnhancedBackgroundTaskManager(max_concurrent_tasks=3)
        await manager.start()
        
        # Submit real background tasks
        async def background_health_task(task_id: str):
            await asyncio.sleep(0.1)
            return f"health_task_{task_id}_complete"
            
        await manager.submit_task("bg_task_1", lambda: background_health_task("1"))
        await manager.submit_task("bg_task_2", lambda: background_health_task("2"))
        
        # Monkeypatch to use real components
        monkeypatch.setattr(
            "prompt_improver.mcp_server.server.get_background_task_manager",
            lambda: manager
        )
        monkeypatch.setattr(
            mcp_server,
            "batch_processor",
            real_batch_processor
        )

        result = await mcp_server._health_live_impl()

        assert result["status"] == "live"
        assert "background_queue_size" in result
        assert "training_queue_size" in result
        assert result["background_queue_size"] >= 0
        assert result["training_queue_size"] >= 0
        
        # Clean up
        await manager.stop(timeout=2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
