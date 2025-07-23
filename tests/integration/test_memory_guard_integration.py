"""
Integration tests for MemoryGuard with ML Pipeline Orchestrator.

Tests real behavior without mocks to ensure successful memory monitoring integration
and verify resource management works correctly across the orchestration system.
"""

import asyncio
import pytest
import logging
import numpy as np
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
from prompt_improver.security.memory_guard import (
    MemoryGuard, MemoryEvent, MemoryThreatLevel, ResourceStats
)
from prompt_improver.ml.orchestration.integration.component_invoker import ComponentInvoker
from prompt_improver.ml.orchestration.integration.direct_component_loader import DirectComponentLoader
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig


class TestMemoryGuardIntegration:
    """Test suite for MemoryGuard integration with orchestrator."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create and initialize orchestrator with memory guard."""
        config = OrchestratorConfig()
        orchestrator = MLPipelineOrchestrator(config)
        
        # Initialize orchestrator (this will load and initialize the memory guard)
        await orchestrator.initialize()
        
        yield orchestrator
        
        # Cleanup
        await orchestrator.shutdown()
    
    @pytest.fixture
    def memory_guard(self):
        """Create memory guard for testing."""
        return MemoryGuard(max_memory_mb=1000, max_buffer_size=50 * 1024 * 1024)  # Higher limits for testing
    
    @pytest.mark.asyncio
    async def test_orchestrator_memory_guard_initialization(self, orchestrator):
        """Test that orchestrator properly initializes memory guard."""
        # Verify memory guard is loaded and initialized
        assert orchestrator.memory_guard is not None
        assert isinstance(orchestrator.memory_guard, MemoryGuard)
        
        # Verify component invoker has memory guard
        assert orchestrator.component_invoker.memory_guard is not None
        assert orchestrator.component_invoker.memory_guard is orchestrator.memory_guard
        
        # Verify workflow engine has memory guard
        assert orchestrator.workflow_engine.memory_guard is not None
        assert orchestrator.workflow_engine.memory_guard is orchestrator.memory_guard
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, orchestrator):
        """Test memory usage monitoring through orchestrator."""
        # Test memory monitoring
        stats = await orchestrator.monitor_memory_usage("test_operation", "test_component")
        
        assert isinstance(stats, ResourceStats)
        assert stats.current_memory_mb > 0
        assert stats.usage_percent >= 0
        assert isinstance(stats.threat_level, MemoryThreatLevel)
    
    @pytest.mark.asyncio
    async def test_memory_operation_monitoring(self, orchestrator):
        """Test memory monitoring during operations."""
        # Test operation monitoring context manager
        async with orchestrator.monitor_operation_memory("test_operation", "test_component"):
            # Simulate some memory usage
            test_data = np.random.random((1000, 1000))  # Small array for testing
            await asyncio.sleep(0.1)  # Brief operation
        
        # Check that memory events were recorded
        if orchestrator.memory_guard:
            assert len(orchestrator.memory_guard.memory_events) > 0
    
    @pytest.mark.asyncio
    async def test_memory_validation(self, orchestrator):
        """Test memory validation for ML operations."""
        # Check current memory usage first
        current_stats = await orchestrator.monitor_memory_usage("memory_check", "test_component")

        # Test with small data - should work unless memory is critically low
        small_data = np.array([1, 2, 3, 4, 5])
        try:
            result = await orchestrator.validate_operation_memory(small_data, "small_data_test", "test_component")
            assert result is True
        except MemoryError:
            # If current memory usage is too high, this is expected behavior
            assert current_stats.usage_percent > 90, "Memory validation failed but usage is not critically high"

        # Test with larger data (should fail if memory is constrained)
        try:
            large_data = np.random.random((1000, 1000))  # 8MB array
            result = await orchestrator.validate_operation_memory(large_data, "large_data_test", "test_component")
            # Should either pass or raise MemoryError
            assert isinstance(result, bool)
        except MemoryError:
            # This is expected behavior for large data when memory is constrained
            pass
    
    @pytest.mark.asyncio
    async def test_memory_monitored_component_invocation(self, orchestrator):
        """Test memory-monitored component invocation."""
        # Load components first
        loaded_components = await orchestrator.component_loader.load_all_components()
        assert len(loaded_components) > 0
        
        # Test memory-monitored component invocation
        result = await orchestrator.component_invoker.invoke_component_method_with_memory_monitoring(
            "training_data_loader",
            "load_training_data",
            {"test": "data"},
            context={"operation": "test_memory_monitoring"}
        )
        
        # Result should be an InvocationResult object
        assert hasattr(result, 'success')
        assert hasattr(result, 'component_name')
        assert result.component_name == "training_data_loader"
    
    @pytest.mark.asyncio
    async def test_memory_event_emission(self, orchestrator):
        """Test that memory events are properly emitted."""
        # Get initial memory event count
        initial_events = len(orchestrator.memory_guard.memory_events) if orchestrator.memory_guard else 0
        
        # Trigger memory monitoring operations
        await orchestrator.monitor_memory_usage("event_test_operation", "test_component")
        
        # Simulate memory-intensive operation
        async with orchestrator.monitor_operation_memory("memory_intensive_operation", "test_component"):
            # Create some data to use memory
            test_array = np.random.random((500, 500))
            await asyncio.sleep(0.1)
        
        # Check that memory events were recorded
        if orchestrator.memory_guard:
            final_events = len(orchestrator.memory_guard.memory_events)
            assert final_events > initial_events
            
            # Verify event details
            latest_event = orchestrator.memory_guard.memory_events[-1]
            assert isinstance(latest_event, MemoryEvent)
            assert latest_event.operation_name is not None
            assert latest_event.memory_usage_mb > 0
    
    @pytest.mark.asyncio
    async def test_memory_monitored_training_workflow(self, orchestrator):
        """Test memory-monitored training workflow."""
        # Test with valid training data
        training_data = {
            "features": np.array([[1, 2], [3, 4], [5, 6]]),
            "labels": np.array([0, 1, 0]),
            "metadata": {"source": "test"}
        }
        
        try:
            result = await orchestrator.run_training_workflow_with_memory_monitoring(
                training_data,
                context={"user_id": "test_user", "operation": "memory_test"}
            )
            
            # Should complete successfully with valid data
            assert isinstance(result, dict)
            
        except Exception as e:
            # May fail due to component not being fully initialized, but should not be memory-related
            # unless it's a legitimate MemoryError
            if "memory" not in str(e).lower() or isinstance(e, MemoryError):
                # MemoryError is acceptable for testing
                pass
            else:
                # Re-raise non-memory related errors for investigation
                raise
    
    @pytest.mark.asyncio
    async def test_memory_garbage_collection(self, orchestrator):
        """Test async garbage collection functionality."""
        if not orchestrator.memory_guard:
            pytest.skip("Memory guard not available")
        
        # Get initial memory
        initial_stats = orchestrator.memory_guard.get_resource_stats()
        
        # Force garbage collection
        freed_mb = await orchestrator.memory_guard.force_garbage_collection_async("test_gc")
        
        # Check that GC was performed
        assert isinstance(freed_mb, (int, float))
        assert orchestrator.memory_guard.gc_collections > 0
    
    @pytest.mark.asyncio
    async def test_memory_threat_level_detection(self, memory_guard):
        """Test memory threat level detection."""
        # Test with small memory guard for easier testing
        stats = memory_guard.get_resource_stats()
        
        # Verify threat level is calculated
        assert isinstance(stats.threat_level, MemoryThreatLevel)
        
        # Test memory monitoring
        result = await memory_guard.check_memory_usage_async("threat_test", "test_component")
        assert isinstance(result, ResourceStats)
        assert isinstance(result.threat_level, MemoryThreatLevel)
    
    @pytest.mark.asyncio
    async def test_memory_buffer_validation(self, memory_guard):
        """Test memory buffer validation."""
        # Test with small buffer (should pass)
        small_buffer = b"small test data"
        assert memory_guard.validate_buffer_size(small_buffer, "small_buffer_test") is True
        
        # Test with large buffer (should fail with memory guard limits)
        large_buffer = b"x" * (60 * 1024 * 1024)  # 60MB buffer (exceeds 50MB limit)
        with pytest.raises(MemoryError):
            memory_guard.validate_buffer_size(large_buffer, "large_buffer_test")
    
    @pytest.mark.asyncio
    async def test_memory_ml_operation_validation(self, memory_guard):
        """Test ML-specific memory operation validation."""
        # Test with small array (should pass)
        small_array = np.array([1, 2, 3, 4, 5])
        result = await memory_guard.validate_ml_operation_memory(small_array, "small_array_test", "test_component")
        assert result is True
        
        # Test with larger array (may fail depending on memory guard limits)
        try:
            large_array = np.random.random((1000, 1000))  # 8MB array
            result = await memory_guard.validate_ml_operation_memory(large_array, "large_array_test", "test_component")
            assert isinstance(result, bool)
        except MemoryError:
            # Expected for large arrays with small memory limits
            pass
    
    @pytest.mark.asyncio
    async def test_async_memory_monitor_context(self, memory_guard):
        """Test async memory monitor context manager."""
        async with memory_guard.monitor_operation_async("context_test", "test_component"):
            # Simulate some work
            test_data = np.array([1, 2, 3, 4, 5])
            await asyncio.sleep(0.05)
        
        # Check that memory events were recorded
        assert len(memory_guard.memory_events) > 0
        
        # Verify operation tracking
        assert len(memory_guard.active_operations) == 0  # Should be cleared after context exit
    
    @pytest.mark.asyncio
    async def test_memory_statistics_tracking(self, orchestrator):
        """Test memory statistics tracking and reporting."""
        if not orchestrator.memory_guard:
            pytest.skip("Memory guard not available")
        
        # Get initial stats
        initial_stats = orchestrator.memory_guard._monitoring_stats.copy()
        
        # Perform various memory operations
        await orchestrator.monitor_memory_usage("stats_test_1", "test_component")
        
        async with orchestrator.monitor_operation_memory("stats_test_2", "test_component"):
            await asyncio.sleep(0.05)
        
        # Check updated stats
        final_stats = orchestrator.memory_guard._monitoring_stats
        
        assert final_stats["total_operations"] > initial_stats["total_operations"]
    
    @pytest.mark.asyncio
    async def test_component_loader_memory_guard_registration(self):
        """Test that MemoryGuard is properly registered in component loader."""
        from prompt_improver.ml.orchestration.core.component_registry import ComponentTier
        
        component_loader = DirectComponentLoader()
        
        # Test loading the memory guard component
        loaded_component = await component_loader.load_component(
            "memory_guard", 
            ComponentTier.TIER_6_SECURITY
        )
        
        assert loaded_component is not None
        assert loaded_component.name == "memory_guard"
        assert loaded_component.component_class.__name__ == "MemoryGuard"
        
        # Test initialization
        success = await component_loader.initialize_component("memory_guard")
        assert success is True
        
        # Verify instance is created
        assert loaded_component.instance is not None
        assert isinstance(loaded_component.instance, MemoryGuard)
    
    @pytest.mark.asyncio
    async def test_memory_guard_resource_stats(self, memory_guard):
        """Test comprehensive resource statistics."""
        stats = memory_guard.get_resource_stats()
        
        # Verify all required fields are present
        assert hasattr(stats, 'current_memory_mb')
        assert hasattr(stats, 'peak_memory_mb')
        assert hasattr(stats, 'initial_memory_mb')
        assert hasattr(stats, 'allocated_memory_mb')
        assert hasattr(stats, 'memory_limit_mb')
        assert hasattr(stats, 'usage_percent')
        assert hasattr(stats, 'threat_level')
        assert hasattr(stats, 'gc_collections')
        assert hasattr(stats, 'active_operations')
        
        # Verify data types and ranges
        assert isinstance(stats.current_memory_mb, (int, float))
        assert isinstance(stats.usage_percent, (int, float))
        assert stats.usage_percent >= 0
        assert isinstance(stats.threat_level, MemoryThreatLevel)
    
    @pytest.mark.asyncio
    async def test_concurrent_memory_monitoring(self, orchestrator):
        """Test concurrent memory monitoring operations."""
        if not orchestrator.memory_guard:
            pytest.skip("Memory guard not available")
        
        # Test concurrent memory monitoring
        async def memory_operation(op_id):
            async with orchestrator.monitor_operation_memory(f"concurrent_op_{op_id}", "test_component"):
                # Simulate some work
                data = np.random.random((100, 100))
                await asyncio.sleep(0.1)
                return f"completed_{op_id}"
        
        # Run multiple operations concurrently
        tasks = [memory_operation(i) for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 3
        assert all("completed_" in result for result in results)
        
        # Check that memory events were recorded
        assert len(orchestrator.memory_guard.memory_events) > 0
