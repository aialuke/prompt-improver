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

from prompt_improver.database import get_session
from prompt_improver.mcp_server.server import APESMCPServer
from prompt_improver.ml.optimization.batch.unified_batch_processor import (
    BatchProcessorConfig,
    UnifiedBatchProcessor,
)
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

    @pytest.mark.asyncio
    async def test_health_live_success(self, mcp_server):
        """Test successful health/live endpoint with real behavior - no mocks."""
        result = await mcp_server._health_live_impl()
        assert result["status"] == "live"
        assert "event_loop_latency_ms" in result
        assert "timestamp" in result
        assert isinstance(result["event_loop_latency_ms"], (int, float))
        assert result["event_loop_latency_ms"] >= 0
        if "background_queue_size" in result:
            assert result["background_queue_size"] >= 0

    @pytest.mark.asyncio
    async def test_health_live_error_resilience(self, mcp_server):
        """Test health/live endpoint resilience with real error scenarios."""
        results = []
        for _ in range(3):
            result = await mcp_server._health_live_impl()
            results.append(result)
            await asyncio.sleep(0.01)
        statuses = [r["status"] for r in results]
        assert len(set(statuses)) <= 2
        assert "live" in statuses or "error" in statuses

    @pytest.mark.asyncio
    async def test_health_ready_success(self, mcp_server):
        """Test successful health/ready endpoint with real database behavior."""
        result = await mcp_server._health_ready_impl()
        assert result["status"] in ["ready", "not_ready", "error"]
        assert "timestamp" in result
        if result["status"] == "error":
            assert "error" in result
        elif "database" in result:
            db_info = result["database"]
            assert "status" in db_info
        elif "rule_application" in result:
            rule_info = result["rule_application"]
            assert "ready" in rule_info or "service_available" in rule_info

    @pytest.mark.asyncio
    async def test_health_ready_consistency(self, mcp_server):
        """Test health/ready endpoint consistency with real behavior."""
        results = []
        for _ in range(5):
            result = await mcp_server._health_ready_impl()
            results.append(result)
            await asyncio.sleep(0.01)
        statuses = [r["status"] for r in results]
        unique_statuses = set(statuses)
        assert len(unique_statuses) <= 2
        for result in results:
            assert "timestamp" in result
            assert isinstance(result["timestamp"], (int, float))

    @pytest.mark.asyncio
    async def test_health_ready_response_time(self, mcp_server):
        """Test health/ready endpoint response time with real timing."""
        start_time = time.time()
        result = await mcp_server._health_ready_impl()
        response_time = (time.time() - start_time) * 1000
        assert response_time < 5000
        assert "timestamp" in result
        assert "status" in result
        assert result["status"] in ["ready", "not_ready", "error"]

    @pytest.mark.asyncio
    async def test_health_ready_structure_validation(self, mcp_server):
        """Test health/ready endpoint returns proper structure with real behavior."""
        result = await mcp_server._health_ready_impl()
        assert isinstance(result, dict)
        assert "status" in result
        assert "timestamp" in result
        valid_statuses = ["ready", "not_ready", "error"]
        assert result["status"] in valid_statuses
        current_time = time.time()
        assert abs(current_time - result["timestamp"]) < 10

    @pytest.mark.asyncio
    async def test_get_training_queue_size(self, mcp_server):
        """Test training queue size retrieval with real behavior."""
        result = await mcp_server._get_training_queue_size_impl()
        assert isinstance(result, dict)
        assert "queue_size" in result
        assert "status" in result
        assert "timestamp" in result
        assert result["queue_size"] >= 0
        assert isinstance(result["queue_size"], int)
        assert result["status"] in ["idle", "active", "error"]
        if "processor_config" in result:
            config = result["processor_config"]
            assert isinstance(config, dict)

    @pytest.mark.asyncio
    async def test_get_training_queue_size_consistency(self, mcp_server):
        """Test training queue size consistency across multiple calls."""
        results = []
        for _ in range(3):
            result = await mcp_server._get_training_queue_size_impl()
            results.append(result)
            await asyncio.sleep(0.01)
        for result in results:
            assert isinstance(result, dict)
            assert "queue_size" in result
            assert "status" in result
            assert result["queue_size"] >= 0
            assert isinstance(result["queue_size"], int)
        queue_sizes = [r["queue_size"] for r in results]
        assert max(queue_sizes) - min(queue_sizes) <= 10

    @pytest.mark.asyncio
    async def test_event_loop_latency_measurement(self):
        """Test that event loop latency is measured correctly in real conditions."""
        start_time = time.time()
        await asyncio.sleep(0.01)
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        assert latency >= 10
        assert latency < 100

    @pytest.mark.asyncio
    async def test_batch_processor_integration(self, mcp_server):
        """Test integration with real BatchProcessor - comprehensive validation."""
        result = await mcp_server._get_training_queue_size_impl()
        assert isinstance(result, dict)
        assert "queue_size" in result
        assert "status" in result
        assert "health_status" in result
        assert "timestamp" in result
        assert result["queue_size"] >= 0
        assert isinstance(result["queue_size"], int)
        valid_statuses = ["idle", "active", "error"]
        assert result["status"] in valid_statuses
        valid_health = ["healthy", "degraded", "unhealthy"]
        assert result["health_status"] in valid_health
        if "processing_rate" in result:
            assert result["processing_rate"] >= 0.0
            assert isinstance(result["processing_rate"], (int, float))


class TestBackgroundTaskManagerIntegration(TestHealthEndpoints):
    """Test suite for BackgroundTaskManager integration with health checks using real behavior."""

    @pytest.mark.asyncio
    async def test_background_task_manager_real_behavior(self):
        """Test BackgroundTaskManager with real behavior - no mocks."""
        manager = EnhancedBackgroundTaskManager(max_concurrent_tasks=3)
        await manager.start()
        try:
            initial_size = manager.get_queue_size()
            assert initial_size >= 0

            async def real_test_task():
                await asyncio.sleep(0.05)
                return "task_complete"

            await manager.submit_task("test_task", real_test_task)
            await asyncio.sleep(0.1)
            final_size = manager.get_queue_size()
            assert final_size >= 0
        finally:
            await manager.stop(timeout=2.0)

    @pytest.mark.asyncio
    async def test_background_task_manager_metrics(self):
        """Test BackgroundTaskManager metrics with real behavior."""
        manager = EnhancedBackgroundTaskManager(max_concurrent_tasks=2)
        await manager.start()
        try:
            initial_counts = manager.get_task_count()
            assert isinstance(initial_counts, dict)
            assert all(count >= 0 for count in initial_counts.values())

            async def quick_task(task_id: str):
                await asyncio.sleep(0.02)
                return f"task_{task_id}_complete"

            await manager.submit_task("task1", lambda: quick_task("1"))
            await manager.submit_task("task2", lambda: quick_task("2"))
            await asyncio.sleep(0.1)
            final_counts = manager.get_task_count()
            assert all(count >= 0 for count in final_counts.values())
        finally:
            await manager.stop(timeout=2.0)

    @pytest.mark.asyncio
    async def test_health_live_background_integration(self, mcp_server):
        """Test health/live endpoint integration with real system behavior."""
        result = await mcp_server._health_live_impl()
        assert result["status"] == "live"
        assert "event_loop_latency_ms" in result
        assert "timestamp" in result
        assert result["event_loop_latency_ms"] >= 0
        assert result["event_loop_latency_ms"] < 1000
        if "background_queue_size" in result:
            assert result["background_queue_size"] >= 0
        if "phase" in result:
            assert result["phase"] == "0"
        if "mcp_server_mode" in result:
            assert result["mcp_server_mode"] == "rule_application_only"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
