"""Integration tests for startup task orchestration."""

import asyncio
import pytest
import logging
from unittest.mock import AsyncMock, MagicMock

from prompt_improver.services.startup import (
    init_startup_tasks,
    shutdown_startup_tasks,
    startup_context,
    is_startup_complete,
    get_startup_task_count
)


class TestStartupOrchestration:
    """Test startup task orchestration functionality."""
    
    @pytest.mark.asyncio
    async def test_init_startup_tasks_success(self):
        """Test successful startup task initialization."""
        # Mock configuration for testing
        batch_config = {
            "batch_size": 5,
            "batch_timeout": 10,
            "max_attempts": 2,
            "concurrency": 2
        }
        
        # Initialize startup tasks
        result = await init_startup_tasks(
            max_concurrent_tasks=5,
            session_ttl=1800,
            cleanup_interval=120,
            batch_config=batch_config
        )
        
        # Verify successful startup
        assert result["status"] == "success"
        assert "startup_time_ms" in result
        assert result["startup_time_ms"] > 0
        
        # Verify components were initialized
        assert "background_manager" in result["components"]
        assert "session_store" in result["components"]
        assert "batch_processor" in result["components"]
        assert "health_service" in result["components"]
        
        # Verify component references are available
        assert "component_refs" in result
        assert len(result["component_refs"]) == 4
        
        # Verify startup tasks are running
        assert result["active_tasks"] >= 2  # At least batch processor and health monitor
        assert is_startup_complete()
        assert get_startup_task_count() >= 2
        
        # Clean up
        shutdown_result = await shutdown_startup_tasks()
        assert shutdown_result["status"] == "success"
        assert not is_startup_complete()
        assert get_startup_task_count() == 0
    
    @pytest.mark.asyncio
    async def test_startup_context_manager(self):
        """Test startup context manager functionality."""
        # Use context manager for automatic cleanup
        async with startup_context(max_concurrent_tasks=3) as components:
            # Verify components are available
            assert "background_manager" in components
            assert "session_store" in components
            assert "batch_processor" in components
            assert "health_service" in components
            
            # Verify startup is complete
            assert is_startup_complete()
            assert get_startup_task_count() >= 2
        
        # Verify automatic cleanup occurred
        assert not is_startup_complete()
        assert get_startup_task_count() == 0
    
    @pytest.mark.asyncio
    async def test_duplicate_startup_prevention(self):
        """Test that duplicate startup calls are prevented."""
        # Initialize startup tasks first time
        result1 = await init_startup_tasks()
        assert result1["status"] == "success"
        assert is_startup_complete()
        
        # Try to initialize again
        result2 = await init_startup_tasks()
        assert result2["status"] == "already_initialized"
        
        # Clean up
        await shutdown_startup_tasks()
        assert not is_startup_complete()
    
    @pytest.mark.asyncio 
    async def test_graceful_shutdown_timeout_handling(self):
        """Test graceful shutdown with timeout constraints."""
        # Initialize startup tasks
        result = await init_startup_tasks()
        assert result["status"] == "success"
        
        # Test shutdown with short timeout
        shutdown_result = await shutdown_startup_tasks(timeout=5.0)
        
        # Should still succeed even with short timeout
        assert shutdown_result["status"] == "success"
        assert shutdown_result["shutdown_time_ms"] <= 5500  # Allow some buffer
        assert not is_startup_complete()
    
    @pytest.mark.asyncio
    async def test_health_monitor_integration(self):
        """Test that health monitor is properly integrated."""
        # Initialize with health monitoring
        result = await init_startup_tasks()
        assert result["status"] == "success"
        
        # Verify health service is available
        health_service = result["component_refs"]["health_service"]
        assert health_service is not None
        
        # Run a health check
        health_result = await health_service.run_health_check()
        assert health_result.overall_status.value in ["healthy", "warning", "failed"]
        
        # Clean up
        await shutdown_startup_tasks()
    
    @pytest.mark.asyncio
    async def test_batch_processor_integration(self):
        """Test batch processor integration with custom config."""
        custom_config = {
            "batch_size": 3,
            "batch_timeout": 5,
            "max_attempts": 1,
            "concurrency": 1,
            "dry_run": True  # Safe for testing
        }
        
        result = await init_startup_tasks(batch_config=custom_config)
        assert result["status"] == "success"
        
        # Verify batch processor is configured correctly
        batch_processor = result["component_refs"]["batch_processor"]
        assert batch_processor.config.batch_size == 3
        assert batch_processor.config.batch_timeout == 5
        assert batch_processor.config.max_attempts == 1
        assert batch_processor.config.dry_run == True
        
        # Clean up
        await shutdown_startup_tasks()
    
    @pytest.mark.asyncio
    async def test_session_store_integration(self):
        """Test session store integration and cleanup."""
        result = await init_startup_tasks(
            session_ttl=300,  # 5 minutes
            cleanup_interval=60  # 1 minute
        )
        assert result["status"] == "success"
        
        # Verify session store is working
        session_store = result["component_refs"]["session_store"]
        
        # Test basic session operations
        test_key = "test_session"
        test_value = {"user_id": "123", "data": "test"}
        
        success = await session_store.set(test_key, test_value)
        assert success
        
        retrieved_value = await session_store.get(test_key)
        assert retrieved_value == test_value
        
        # Test session touch
        touched = await session_store.touch(test_key)
        assert touched
        
        # Test session deletion
        deleted = await session_store.delete(test_key)
        assert deleted
        
        # Verify it's gone
        retrieved_again = await session_store.get(test_key)
        assert retrieved_again is None
        
        # Clean up
        await shutdown_startup_tasks()
    
    def test_startup_status_functions(self):
        """Test startup status utility functions."""
        # Initially not started
        assert not is_startup_complete()
        assert get_startup_task_count() == 0
    
    @pytest.mark.asyncio
    async def test_startup_error_logging(self, caplog):
        """Test that startup errors are properly logged."""
        # This test would require mocking components to force failures
        # For now, just verify logging configuration
        with caplog.at_level(logging.INFO):
            result = await init_startup_tasks()
            assert result["status"] == "success"
            
            # Verify startup messages were logged
            startup_messages = [record.message for record in caplog.records 
                             if "APES system components" in record.message]
            assert len(startup_messages) > 0
            
            await shutdown_startup_tasks()
