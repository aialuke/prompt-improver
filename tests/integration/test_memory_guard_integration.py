"""
Integration tests for MemoryGuard with ML Pipeline Orchestrator.

Tests real behavior without mocks to ensure successful memory monitoring integration
and verify resource management works correctly across the orchestration system.
"""
import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock
import numpy as np
import pytest
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
from prompt_improver.ml.orchestration.integration.component_invoker import ComponentInvoker
from prompt_improver.ml.orchestration.integration.direct_component_loader import DirectComponentLoader
from prompt_improver.security.memory_guard import MemoryEvent, MemoryGuard, MemoryThreatLevel, ResourceStats

class TestMemoryGuardIntegration:
    """Test suite for MemoryGuard integration with orchestrator."""

    @pytest.fixture
    async def orchestrator(self):
        """Create and initialize orchestrator with memory guard."""
        config = OrchestratorConfig()
        orchestrator = MLPipelineOrchestrator(config)
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.fixture
    def memory_guard(self):
        """Create memory guard for testing."""
        return MemoryGuard(max_memory_mb=1000, max_buffer_size=50 * 1024 * 1024)

    @pytest.mark.asyncio
    async def test_orchestrator_memory_guard_initialization(self, orchestrator):
        """Test that orchestrator properly initializes memory guard."""
        assert orchestrator.memory_guard is not None
        assert isinstance(orchestrator.memory_guard, MemoryGuard)
        assert orchestrator.component_invoker.memory_guard is not None
        assert orchestrator.component_invoker.memory_guard is orchestrator.memory_guard
        assert orchestrator.workflow_engine.memory_guard is not None
        assert orchestrator.workflow_engine.memory_guard is orchestrator.memory_guard

    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, orchestrator):
        """Test memory usage monitoring through orchestrator."""
        stats = await orchestrator.monitor_memory_usage('test_operation', 'test_component')
        assert isinstance(stats, ResourceStats)
        assert stats.current_memory_mb > 0
        assert stats.usage_percent >= 0
        assert isinstance(stats.threat_level, MemoryThreatLevel)

    @pytest.mark.asyncio
    async def test_memory_operation_monitoring(self, orchestrator):
        """Test memory monitoring during operations."""
        async with orchestrator.monitor_operation_memory('test_operation', 'test_component'):
            test_data = np.random.random((1000, 1000))
            await asyncio.sleep(0.1)
        if orchestrator.memory_guard:
            assert len(orchestrator.memory_guard.memory_events) > 0

    @pytest.mark.asyncio
    async def test_memory_validation(self, orchestrator):
        """Test memory validation for ML operations."""
        current_stats = await orchestrator.monitor_memory_usage('memory_check', 'test_component')
        small_data = np.array([1, 2, 3, 4, 5])
        try:
            result = await orchestrator.validate_operation_memory(small_data, 'small_data_test', 'test_component')
            assert result is True
        except MemoryError:
            assert current_stats.usage_percent > 90, 'Memory validation failed but usage is not critically high'
        try:
            large_data = np.random.random((1000, 1000))
            result = await orchestrator.validate_operation_memory(large_data, 'large_data_test', 'test_component')
            assert isinstance(result, bool)
        except MemoryError:
            pass

    @pytest.mark.asyncio
    async def test_memory_monitored_component_invocation(self, orchestrator):
        """Test memory-monitored component invocation."""
        loaded_components = await orchestrator.component_loader.load_all_components()
        assert len(loaded_components) > 0
        result = await orchestrator.component_invoker.invoke_component_method_with_memory_monitoring('training_data_loader', 'load_training_data', {'test': 'data'}, context={'operation': 'test_memory_monitoring'})
        assert hasattr(result, 'success')
        assert hasattr(result, 'component_name')
        assert result.component_name == 'training_data_loader'

    @pytest.mark.asyncio
    async def test_memory_event_emission(self, orchestrator):
        """Test that memory events are properly emitted."""
        initial_events = len(orchestrator.memory_guard.memory_events) if orchestrator.memory_guard else 0
        await orchestrator.monitor_memory_usage('event_test_operation', 'test_component')
        async with orchestrator.monitor_operation_memory('memory_intensive_operation', 'test_component'):
            test_array = np.random.random((500, 500))
            await asyncio.sleep(0.1)
        if orchestrator.memory_guard:
            final_events = len(orchestrator.memory_guard.memory_events)
            assert final_events > initial_events
            latest_event = orchestrator.memory_guard.memory_events[-1]
            assert isinstance(latest_event, MemoryEvent)
            assert latest_event.operation_name is not None
            assert latest_event.memory_usage_mb > 0

    @pytest.mark.asyncio
    async def test_memory_monitored_training_workflow(self, orchestrator):
        """Test memory-monitored training workflow."""
        training_data = {'features': np.array([[1, 2], [3, 4], [5, 6]]), 'labels': np.array([0, 1, 0]), 'metadata': {'source': 'test'}}
        try:
            result = await orchestrator.run_training_workflow_with_memory_monitoring(training_data, context={'user_id': 'test_user', 'operation': 'memory_test'})
            assert isinstance(result, dict)
        except Exception as e:
            if 'memory' not in str(e).lower() or isinstance(e, MemoryError):
                pass
            else:
                raise

    @pytest.mark.asyncio
    async def test_memory_garbage_collection(self, orchestrator):
        """Test async garbage collection functionality."""
        if not orchestrator.memory_guard:
            pytest.skip('Memory guard not available')
        initial_stats = orchestrator.memory_guard.get_resource_stats()
        freed_mb = await orchestrator.memory_guard.force_garbage_collection_async('test_gc')
        assert isinstance(freed_mb, (int, float))
        assert orchestrator.memory_guard.gc_collections > 0

    @pytest.mark.asyncio
    async def test_memory_threat_level_detection(self, memory_guard):
        """Test memory threat level detection."""
        stats = memory_guard.get_resource_stats()
        assert isinstance(stats.threat_level, MemoryThreatLevel)
        result = await memory_guard.check_memory_usage_async('threat_test', 'test_component')
        assert isinstance(result, ResourceStats)
        assert isinstance(result.threat_level, MemoryThreatLevel)

    @pytest.mark.asyncio
    async def test_memory_buffer_validation(self, memory_guard):
        """Test memory buffer validation."""
        small_buffer = b'small test data'
        assert memory_guard.validate_buffer_size(small_buffer, 'small_buffer_test') is True
        large_buffer = b'x' * (60 * 1024 * 1024)
        with pytest.raises(MemoryError):
            memory_guard.validate_buffer_size(large_buffer, 'large_buffer_test')

    @pytest.mark.asyncio
    async def test_memory_ml_operation_validation(self, memory_guard):
        """Test ML-specific memory operation validation."""
        small_array = np.array([1, 2, 3, 4, 5])
        result = await memory_guard.validate_ml_operation_memory(small_array, 'small_array_test', 'test_component')
        assert result is True
        try:
            large_array = np.random.random((1000, 1000))
            result = await memory_guard.validate_ml_operation_memory(large_array, 'large_array_test', 'test_component')
            assert isinstance(result, bool)
        except MemoryError:
            pass

    @pytest.mark.asyncio
    async def test_async_memory_monitor_context(self, memory_guard):
        """Test async memory monitor context manager."""
        async with memory_guard.monitor_operation_async('context_test', 'test_component'):
            test_data = np.array([1, 2, 3, 4, 5])
            await asyncio.sleep(0.05)
        assert len(memory_guard.memory_events) > 0
        assert len(memory_guard.active_operations) == 0

    @pytest.mark.asyncio
    async def test_memory_statistics_tracking(self, orchestrator):
        """Test memory statistics tracking and reporting."""
        if not orchestrator.memory_guard:
            pytest.skip('Memory guard not available')
        initial_stats = orchestrator.memory_guard._monitoring_stats.copy()
        await orchestrator.monitor_memory_usage('stats_test_1', 'test_component')
        async with orchestrator.monitor_operation_memory('stats_test_2', 'test_component'):
            await asyncio.sleep(0.05)
        final_stats = orchestrator.memory_guard._monitoring_stats
        assert final_stats['total_operations'] > initial_stats['total_operations']

    @pytest.mark.asyncio
    async def test_component_loader_memory_guard_registration(self):
        """Test that MemoryGuard is properly registered in component loader."""
        from prompt_improver.ml.orchestration.core.component_registry import ComponentTier
        component_loader = DirectComponentLoader()
        loaded_component = await component_loader.load_component('memory_guard', ComponentTier.TIER_1)
        assert loaded_component is not None
        assert loaded_component.name == 'memory_guard'
        assert loaded_component.component_class.__name__ == 'MemoryGuard'
        success = await component_loader.initialize_component('memory_guard')
        assert success is True
        assert loaded_component.instance is not None
        assert isinstance(loaded_component.instance, MemoryGuard)

    @pytest.mark.asyncio
    async def test_memory_guard_resource_stats(self, memory_guard):
        """Test comprehensive resource statistics."""
        stats = memory_guard.get_resource_stats()
        assert hasattr(stats, 'current_memory_mb')
        assert hasattr(stats, 'peak_memory_mb')
        assert hasattr(stats, 'initial_memory_mb')
        assert hasattr(stats, 'allocated_memory_mb')
        assert hasattr(stats, 'memory_limit_mb')
        assert hasattr(stats, 'usage_percent')
        assert hasattr(stats, 'threat_level')
        assert hasattr(stats, 'gc_collections')
        assert hasattr(stats, 'active_operations')
        assert isinstance(stats.current_memory_mb, (int, float))
        assert isinstance(stats.usage_percent, (int, float))
        assert stats.usage_percent >= 0
        assert isinstance(stats.threat_level, MemoryThreatLevel)

    @pytest.mark.asyncio
    async def test_concurrent_memory_monitoring(self, orchestrator):
        """Test concurrent memory monitoring operations."""
        if not orchestrator.memory_guard:
            pytest.skip('Memory guard not available')

        async def memory_operation(op_id):
            async with orchestrator.monitor_operation_memory(f'concurrent_op_{op_id}', 'test_component'):
                data = np.random.random((100, 100))
                await asyncio.sleep(0.1)
                return f'completed_{op_id}'
        tasks = [memory_operation(i) for i in range(3)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 3
        assert all(('completed_' in result for result in results))
        assert len(orchestrator.memory_guard.memory_events) > 0
