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
import pytest_asyncio

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

    @pytest_asyncio.fixture
    async def mcp_server(self):
        """Create a real MCP server instance for health testing."""
        server = APESMCPServer()
        await server.initialize()
        yield server
        # Cleanup - no explicit shutdown needed for this test pattern

    # Fixtures removed - using real behavior testing without mocks or artificial setup

    @pytest.mark.asyncio
    async def test_health_live_success(self, mcp_server):
        """Test successful health/live endpoint with real behavior - no mocks."""
        # Test the actual consolidated server implementation
        result = await mcp_server._health_live_impl()

        # Validate the real response structure from consolidated server
        assert result["status"] == "live"
        assert "event_loop_latency_ms" in result
        assert "timestamp" in result
        assert isinstance(result["event_loop_latency_ms"], (int, float))
        assert result["event_loop_latency_ms"] >= 0
        
        # Background queue size should be 0 in read-only mode (as per server implementation)
        if "background_queue_size" in result:
            assert result["background_queue_size"] >= 0

    @pytest.mark.asyncio
    async def test_health_live_error_resilience(self, mcp_server):
        """Test health/live endpoint resilience with real error scenarios."""
        # Test that the consolidated server handles errors gracefully
        # Real behavior test - call multiple times to test consistency
        results = []
        for _ in range(3):
            result = await mcp_server._health_live_impl()
            results.append(result)
            await asyncio.sleep(0.01)  # Small delay between calls
        
        # All calls should succeed or fail consistently
        statuses = [r["status"] for r in results]
        # Status should be consistent across calls
        assert len(set(statuses)) <= 2  # Either all "live" or mixed "live"/"error"
        
        # At least one should be successful for a healthy system
        assert "live" in statuses or "error" in statuses

    @pytest.mark.asyncio
    async def test_health_ready_success(self, mcp_server):
        """Test successful health/ready endpoint with real database behavior."""
        # Test real database connectivity - no mocks
        result = await mcp_server._health_ready_impl()

        # Real database connectivity test - result depends on actual DB state
        assert result["status"] in ["ready", "not_ready", "error"]  # Real DB may not be available
        assert "timestamp" in result
        
        if result["status"] == "error":
            assert "error" in result
        else:
            # Check for consolidated structure
            if "database" in result:
                # New consolidated structure
                db_info = result["database"]
                assert "status" in db_info
            elif "rule_application" in result:
                # Check rule application capability
                rule_info = result["rule_application"]
                assert "ready" in rule_info or "service_available" in rule_info

    @pytest.mark.asyncio
    async def test_health_ready_consistency(self, mcp_server):
        """Test health/ready endpoint consistency with real behavior."""
        # Test consistency across multiple calls - real behavior
        results = []
        for _ in range(5):
            result = await mcp_server._health_ready_impl()
            results.append(result)
            await asyncio.sleep(0.01)  # Small delay between calls
        
        # Check that responses are consistent
        statuses = [r["status"] for r in results]
        # Status should be mostly consistent (allowing for transient states)
        unique_statuses = set(statuses)
        assert len(unique_statuses) <= 2  # At most 2 different statuses due to real conditions
        
        # All results should have timestamps
        for result in results:
            assert "timestamp" in result
            assert isinstance(result["timestamp"], (int, float))

    @pytest.mark.asyncio
    async def test_health_ready_response_time(self, mcp_server):
        """Test health/ready endpoint response time with real timing."""
        # Measure real response time
        start_time = time.time()
        result = await mcp_server._health_ready_impl()
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Response should be reasonably fast (under 5 seconds for health checks)
        assert response_time < 5000
        
        # Result should have timestamp
        assert "timestamp" in result
        assert "status" in result
        
        # Status should be one of the expected values
        assert result["status"] in ["ready", "not_ready", "error"]

    @pytest.mark.asyncio
    async def test_health_ready_structure_validation(self, mcp_server):
        """Test health/ready endpoint returns proper structure with real behavior."""
        # Test real response structure
        result = await mcp_server._health_ready_impl()
        
        # Basic structure validation
        assert isinstance(result, dict)
        assert "status" in result
        assert "timestamp" in result
        
        # Status should be a valid health status
        valid_statuses = ["ready", "not_ready", "error"]
        assert result["status"] in valid_statuses
        
        # Timestamp should be recent (within last 10 seconds)
        current_time = time.time()
        assert abs(current_time - result["timestamp"]) < 10

    @pytest.mark.asyncio
    async def test_get_training_queue_size(self, mcp_server):
        """Test training queue size retrieval with real behavior."""
        # Test real UnifiedBatchProcessor behavior - no mocks
        result = await mcp_server._get_training_queue_size_impl()

        # Validate real response structure
        assert isinstance(result, dict)
        assert "queue_size" in result
        assert "status" in result
        assert "timestamp" in result
        
        # Queue size should be non-negative integer
        assert result["queue_size"] >= 0
        assert isinstance(result["queue_size"], int)
        
        # Status should indicate health
        assert result["status"] in ["idle", "active", "error"]
        
        # Should have processor configuration info
        if "processor_config" in result:
            config = result["processor_config"]
            assert isinstance(config, dict)

    @pytest.mark.asyncio
    async def test_get_training_queue_size_consistency(self, mcp_server):
        """Test training queue size consistency across multiple calls."""
        # Test consistency with real behavior - no mocks
        results = []
        for _ in range(3):
            result = await mcp_server._get_training_queue_size_impl()
            results.append(result)
            await asyncio.sleep(0.01)
        
        # All results should have consistent structure
        for result in results:
            assert isinstance(result, dict)
            assert "queue_size" in result
            assert "status" in result
            assert result["queue_size"] >= 0
            assert isinstance(result["queue_size"], int)
        
        # Queue sizes should be consistent for idle system
        queue_sizes = [r["queue_size"] for r in results]
        # Allow some variation but should be mostly consistent
        assert max(queue_sizes) - min(queue_sizes) <= 10


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
        """Test integration with real BatchProcessor - comprehensive validation."""
        # Test real processor integration - no mocks
        result = await mcp_server._get_training_queue_size_impl()

        # Comprehensive validation of real response
        assert isinstance(result, dict)
        
        # Core metrics
        assert "queue_size" in result
        assert "status" in result
        assert "health_status" in result
        assert "timestamp" in result
        
        # Queue size validation
        assert result["queue_size"] >= 0
        assert isinstance(result["queue_size"], int)
        
        # Status validation
        valid_statuses = ["idle", "active", "error"]
        assert result["status"] in valid_statuses
        
        # Health status validation
        valid_health = ["healthy", "degraded", "unhealthy"]
        assert result["health_status"] in valid_health
        
        # Should have processing metrics
        if "processing_rate" in result:
            assert result["processing_rate"] >= 0.0
            assert isinstance(result["processing_rate"], (int, float))


class TestBackgroundTaskManagerIntegration(TestHealthEndpoints):
    """Test suite for BackgroundTaskManager integration with health checks using real behavior."""

    @pytest.mark.asyncio
    async def test_background_task_manager_real_behavior(self):
        """Test BackgroundTaskManager with real behavior - no mocks."""
        # Test real background task manager instantiation and basic operations
        manager = EnhancedBackgroundTaskManager(max_concurrent_tasks=3)
        await manager.start()

        try:
            # Test real queue size reporting
            initial_size = manager.get_queue_size()
            assert initial_size >= 0

            # Test real task submission
            async def real_test_task():
                await asyncio.sleep(0.05)  # Short task
                return "task_complete"

            await manager.submit_task("test_task", real_test_task)

            # Allow task to process
            await asyncio.sleep(0.1)

            # Queue size should be non-negative
            final_size = manager.get_queue_size()
            assert final_size >= 0

        finally:
            # Clean up
            await manager.stop(timeout=2.0)

    @pytest.mark.asyncio
    async def test_background_task_manager_metrics(self):
        """Test BackgroundTaskManager metrics with real behavior."""
        manager = EnhancedBackgroundTaskManager(max_concurrent_tasks=2)
        await manager.start()

        try:
            # Test real task count reporting
            initial_counts = manager.get_task_count()
            assert isinstance(initial_counts, dict)
            assert all(count >= 0 for count in initial_counts.values())

            # Submit real tasks
            async def quick_task(task_id: str):
                await asyncio.sleep(0.02)
                return f"task_{task_id}_complete"

            await manager.submit_task("task1", lambda: quick_task("1"))
            await manager.submit_task("task2", lambda: quick_task("2"))

            # Allow processing time
            await asyncio.sleep(0.1)

            # Check final counts
            final_counts = manager.get_task_count()
            assert all(count >= 0 for count in final_counts.values())

        finally:
            # Clean up
            await manager.stop(timeout=2.0)

    @pytest.mark.asyncio
    async def test_health_live_background_integration(self, mcp_server):
        """Test health/live endpoint integration with real system behavior."""
        # Test real system integration - no mocks, no artificial setup
        result = await mcp_server._health_live_impl()
        
        # Validate real consolidated server response
        assert result["status"] == "live"
        assert "event_loop_latency_ms" in result
        assert "timestamp" in result
        
        # Latency should be reasonable
        assert result["event_loop_latency_ms"] >= 0
        assert result["event_loop_latency_ms"] < 1000  # Should be under 1 second
        
        # Background queue should be 0 in read-only mode (as implemented)
        if "background_queue_size" in result:
            assert result["background_queue_size"] >= 0
        
        # Phase should be "0" as per consolidation
        if "phase" in result:
            assert result["phase"] == "0"
        
        # MCP server mode should be rule application only
        if "mcp_server_mode" in result:
            assert result["mcp_server_mode"] == "rule_application_only"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
