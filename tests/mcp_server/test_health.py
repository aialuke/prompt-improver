"""Unit tests for health endpoints in the MCP server.

This module tests the health check functionality including:
- Event loop latency measurement
- Training queue size checks
- Database connectivity verification
- Background task manager integration
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict

# Import the health functions and related components
from prompt_improver.mcp_server.mcp_server import (
    health_live,
    health_ready,
    get_training_queue_size,
    _store_prompt_data,
)
from prompt_improver.optimization.batch_processor import BatchProcessor
from prompt_improver.services.health.background_manager import BackgroundTaskManager


class TestHealthEndpoints:
    """Test suite for health endpoint functionality."""

    @pytest.mark.asyncio
    async def test_health_live_success(self):
        """Test successful health/live endpoint."""
        with patch("prompt_improver.mcp_server.mcp_server.get_background_task_manager") as mock_get_btm, \
             patch("prompt_improver.mcp_server.mcp_server.get_training_queue_size") as mock_get_tqs:
            
            # Mock background task manager
            mock_btm = MagicMock()
            mock_btm.get_queue_size.return_value = 5
            mock_get_btm.return_value = mock_btm
            
            # Mock training queue size
            mock_get_tqs.return_value = 10
            
            result = await health_live()
            
            assert result["status"] == "live"
            assert "event_loop_latency_ms" in result
            assert result["training_queue_size"] == 10
            assert result["background_queue_size"] == 5
            assert "timestamp" in result
            assert isinstance(result["event_loop_latency_ms"], (int, float))
            assert result["event_loop_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_health_live_error_handling(self):
        """Test health/live endpoint error handling."""
        with patch("prompt_improver.mcp_server.mcp_server.get_background_task_manager") as mock_get_btm:
            # Mock an exception
            mock_get_btm.side_effect = Exception("Background task manager error")
            
            result = await health_live()
            
            assert result["status"] == "error"
            assert "error" in result
            assert "Background task manager error" in result["error"]

    @pytest.mark.asyncio
    async def test_health_ready_success(self):
        """Test successful health/ready endpoint."""
        with patch("prompt_improver.mcp_server.mcp_server.get_session") as mock_get_session, \
             patch("prompt_improver.mcp_server.mcp_server.get_training_queue_size") as mock_get_tqs:
            
            # Mock database session
            mock_db_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.fetchone.return_value = (1,)  # Non-None result indicates success
            mock_db_session.execute.return_value = mock_result
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock training queue size
            mock_get_tqs.return_value = 8
            
            result = await health_ready()
            
            assert result["status"] == "ready"
            assert result["db_connectivity"]["ready"] is True
            assert "response_time_ms" in result["db_connectivity"]
            assert result["training_queue_size"] == 8
            assert "event_loop_latency_ms" in result
            assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_health_ready_db_failure(self):
        """Test health/ready endpoint when database is unavailable."""
        with patch("prompt_improver.mcp_server.mcp_server.get_session") as mock_get_session, \
             patch("prompt_improver.mcp_server.mcp_server.get_training_queue_size") as mock_get_tqs:
            
            # Mock database session failure
            mock_db_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.fetchone.return_value = None  # None result indicates failure
            mock_db_session.execute.return_value = mock_result
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock training queue size
            mock_get_tqs.return_value = 3
            
            result = await health_ready()
            
            assert result["status"] == "not ready"
            assert result["db_connectivity"]["ready"] is False
            assert result["training_queue_size"] == 3

    @pytest.mark.asyncio
    async def test_health_ready_latency_threshold(self):
        """Test health/ready endpoint latency measurement and threshold logic."""
        with patch("prompt_improver.mcp_server.mcp_server.get_session") as mock_get_session, \
             patch("prompt_improver.mcp_server.mcp_server.get_training_queue_size") as mock_get_tqs:
            
            # Mock database session
            mock_db_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.fetchone.return_value = (1,)
            mock_db_session.execute.return_value = mock_result
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock training queue size
            mock_get_tqs.return_value = 2
            
            result = await health_ready()
            
            # Should be "ready" under normal conditions
            assert result["status"] == "ready"
            assert result["db_connectivity"]["ready"] is True
            assert "event_loop_latency_ms" in result
            assert result["event_loop_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_health_ready_error_handling(self):
        """Test health/ready endpoint error handling."""
        with patch("prompt_improver.mcp_server.mcp_server.get_session") as mock_get_session:
            # Mock database session error
            mock_get_session.side_effect = Exception("Database connection failed")
            
            result = await health_ready()
            
            assert result["status"] == "error"
            assert "error" in result
            assert "Database connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_get_training_queue_size(self):
        """Test training queue size retrieval."""
        batch_processor = BatchProcessor({"batchSize": 15})
        
        result = await get_training_queue_size(batch_processor)
        
        assert result == 15

    @pytest.mark.asyncio
    async def test_get_training_queue_size_default(self):
        """Test training queue size retrieval with default configuration."""
        batch_processor = BatchProcessor({})
        
        result = await get_training_queue_size(batch_processor)
        
        assert result == 10  # Default batchSize is 10

    @pytest.mark.asyncio
    async def test_store_prompt_data_success(self):
        """Test successful prompt data storage."""
        with patch("prompt_improver.mcp_server.mcp_server.get_session") as mock_get_session:
            # Mock database session
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Test data
            original = "Original prompt"
            enhanced = "Enhanced prompt"
            metrics = {"improvement_score": 0.85}
            session_id = "test_session_123"
            priority = 100
            
            await _store_prompt_data(original, enhanced, metrics, session_id, priority)
            
            # Verify database operations
            mock_db_session.execute.assert_called_once()
            mock_db_session.commit.assert_called_once()
            
            # Verify the SQL query parameters
            call_args = mock_db_session.execute.call_args
            assert call_args[0][1]["prompt_text"] == original
            assert call_args[0][1]["data_source"] == "real"
            assert call_args[0][1]["training_priority"] == priority
            
            # Verify enhancement result structure
            enhancement_result = call_args[0][1]["enhancement_result"]
            assert enhancement_result["enhanced_prompt"] == enhanced
            assert enhancement_result["metrics"] == metrics
            assert enhancement_result["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_store_prompt_data_error_handling(self):
        """Test prompt data storage error handling."""
        with patch("prompt_improver.mcp_server.mcp_server.get_session") as mock_get_session, \
             patch("builtins.print") as mock_print:
            
            # Mock database session error
            mock_db_session = AsyncMock()
            mock_db_session.execute.side_effect = Exception("Database error")
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Should not raise exception, but should print error
            await _store_prompt_data("test", "test", {}, "test", 100)
            
            # Verify error was printed
            mock_print.assert_called_once()
            assert "Error storing prompt data" in mock_print.call_args[0][0]

    @pytest.mark.asyncio
    async def test_event_loop_latency_measurement(self):
        """Test that event loop latency is measured correctly."""
        start_time = time.time()
        await asyncio.sleep(0.01)  # Small delay
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000
        
        # Should be at least 10ms (the sleep duration)
        assert latency >= 10
        # Should be reasonable (less than 100ms in normal conditions)
        assert latency < 100

    @pytest.mark.asyncio
    async def test_batch_processor_integration(self):
        """Test integration with BatchProcessor."""
        config = {
            "batchSize": 20,
            "concurrency": 5,
            "retryAttempts": 3,
            "retryDelay": 1000,
            "timeout": 30000,
        }
        
        batch_processor = BatchProcessor(config)
        queue_size = await get_training_queue_size(batch_processor)
        
        assert queue_size == 20
        assert batch_processor.config["concurrency"] == 5
        assert batch_processor.config["retryAttempts"] == 3


class TestBackgroundTaskManagerIntegration:
    """Test suite for BackgroundTaskManager integration with health checks."""

    @pytest.mark.asyncio
    async def test_background_task_manager_queue_size(self):
        """Test BackgroundTaskManager queue size reporting."""
        manager = BackgroundTaskManager()
        
        # Initially empty
        assert manager.get_queue_size() == 0
        
        # Add a mock task
        await manager.submit_task(
            "test_task",
            lambda: asyncio.sleep(0.1)
        )
        
        # Should have one task in queue
        assert manager.get_queue_size() == 1
        
        # Clean up
        await manager.stop()

    @pytest.mark.asyncio
    async def test_background_task_manager_task_counts(self):
        """Test BackgroundTaskManager task count reporting."""
        manager = BackgroundTaskManager()
        
        # Initially empty
        counts = manager.get_task_count()
        assert all(count == 0 for count in counts.values())
        
        # Add tasks
        await manager.submit_task("task1", lambda: asyncio.sleep(0.1))
        await manager.submit_task("task2", lambda: asyncio.sleep(0.1))
        
        # Check counts
        counts = manager.get_task_count()
        assert counts["running"] + counts["pending"] == 2
        
        # Clean up
        await manager.stop()

    @pytest.mark.asyncio
    async def test_health_check_with_background_tasks(self):
        """Test health checks with active background tasks."""
        with patch("prompt_improver.mcp_server.mcp_server.get_background_task_manager") as mock_get_btm:
            # Mock background task manager with tasks
            mock_btm = MagicMock()
            mock_btm.get_queue_size.return_value = 3
            mock_get_btm.return_value = mock_btm
            
            # Mock other components
            with patch("prompt_improver.mcp_server.mcp_server.get_training_queue_size") as mock_get_tqs:
                mock_get_tqs.return_value = 5
                
                result = await health_live()
                
                assert result["status"] == "live"
                assert result["background_queue_size"] == 3
                assert result["training_queue_size"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
