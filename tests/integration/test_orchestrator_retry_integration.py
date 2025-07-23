"""
Integration tests for UnifiedRetryManager with ML Pipeline Orchestrator.

Tests real behavior without mocks to ensure successful integration
and verify retry mechanisms work correctly across the orchestration system.
"""

import asyncio
import pytest
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
from prompt_improver.ml.orchestration.core.unified_retry_manager import (
    UnifiedRetryManager, RetryConfig, RetryStrategy, get_retry_manager
)
from prompt_improver.ml.orchestration.integration.component_invoker import ComponentInvoker
from prompt_improver.ml.orchestration.integration.direct_component_loader import DirectComponentLoader
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig


class TestOrchestratorRetryIntegration:
    """Test suite for orchestrator retry integration."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create and initialize orchestrator with retry manager."""
        config = OrchestratorConfig()
        orchestrator = MLPipelineOrchestrator(config)
        
        # Initialize orchestrator (this will load and initialize the retry manager)
        await orchestrator.initialize()
        
        yield orchestrator
        
        # Cleanup
        await orchestrator.shutdown()
    
    @pytest.fixture
    def component_loader(self):
        """Create component loader for testing."""
        return DirectComponentLoader()
    
    @pytest.fixture
    def retry_manager(self):
        """Create retry manager for testing."""
        return UnifiedRetryManager()
    
    @pytest.mark.asyncio
    async def test_orchestrator_retry_manager_initialization(self, orchestrator):
        """Test that orchestrator properly initializes retry manager."""
        # Verify retry manager is loaded and initialized
        assert orchestrator.retry_manager is not None
        assert isinstance(orchestrator.retry_manager, UnifiedRetryManager)
        
        # Verify component invoker has retry manager
        assert orchestrator.component_invoker.retry_manager is not None
        assert orchestrator.component_invoker.retry_manager is orchestrator.retry_manager
        
        # Verify workflow engine has retry manager
        assert orchestrator.workflow_engine.retry_manager is not None
        assert orchestrator.workflow_engine.retry_manager is orchestrator.retry_manager
    
    @pytest.mark.asyncio
    async def test_component_invocation_with_retry(self, orchestrator):
        """Test component invocation with retry logic."""
        # Test that we can use the retry-enabled component invocation
        # This tests real behavior by attempting to invoke a component method
        
        # First ensure we have some components loaded
        loaded_components = await orchestrator.component_loader.load_all_components()
        assert len(loaded_components) > 0
        
        # Try to invoke a method with retry (should handle gracefully even if component not initialized)
        result = await orchestrator.component_invoker.invoke_component_method_with_retry(
            "training_data_loader",
            "load_training_data",
            {"test": "data"},
            max_attempts=2
        )
        
        # Result should be an InvocationResult object
        assert hasattr(result, 'success')
        assert hasattr(result, 'component_name')
        assert hasattr(result, 'method_name')
        assert result.component_name == "training_data_loader"
        assert result.method_name == "load_training_data"
    
    @pytest.mark.asyncio
    async def test_orchestrator_execute_with_retry(self, orchestrator):
        """Test orchestrator's execute_with_retry method."""
        call_count = 0
        
        async def test_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Simulated network error")
            return {"success": True, "attempt": call_count}
        
        # Execute operation with retry
        result = await orchestrator.execute_with_retry(
            test_operation,
            "test_operation",
            max_attempts=3
        )
        
        assert result["success"] is True
        assert result["attempt"] == 2
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_manager_metrics_collection(self, orchestrator):
        """Test that retry metrics are properly collected."""
        call_count = 0
        
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Test failure")
            return "success"
        
        # Execute with retry and collect metrics
        result = await orchestrator.execute_with_retry(
            failing_operation,
            "metrics_test_operation",
            max_attempts=4
        )
        
        assert result == "success"
        assert call_count == 3
        
        # Check that metrics were collected
        metrics = orchestrator.retry_manager.get_retry_metrics("metrics_test_operation")
        assert metrics is not None
        assert metrics["total_attempts"] >= 3
        assert metrics["successful_attempts"] >= 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, orchestrator):
        """Test circuit breaker functionality in orchestrator context."""
        failure_count = 0
        
        async def always_failing_operation():
            nonlocal failure_count
            failure_count += 1
            raise ConnectionError("Persistent failure")
        
        # Configure retry with circuit breaker
        config = RetryConfig(
            max_attempts=3,
            enable_circuit_breaker=True,
            failure_threshold=2,
            operation_name="circuit_breaker_test"
        )
        
        # First few calls should fail and trigger circuit breaker
        with pytest.raises(Exception):
            await orchestrator.retry_manager.retry_async(always_failing_operation, config=config)
        
        # Subsequent calls should be blocked by circuit breaker
        with pytest.raises(Exception):
            await orchestrator.retry_manager.retry_async(always_failing_operation, config=config)
        
        # Verify circuit breaker metrics
        metrics = orchestrator.retry_manager.get_retry_metrics("circuit_breaker_test")
        assert metrics is not None
    
    @pytest.mark.asyncio
    async def test_workflow_engine_retry_integration(self, orchestrator):
        """Test that workflow engine uses retry manager for step execution."""
        # This test verifies the workflow engine integration
        # by checking that the retry manager is properly set

        workflow_engine = orchestrator.workflow_engine
        assert workflow_engine.retry_manager is not None
        assert workflow_engine.retry_manager is orchestrator.retry_manager

        # Test that workflow engine can handle retry configuration
        # (This is a structural test since we don't have actual workflows set up)
        assert hasattr(workflow_engine, 'set_retry_manager')

        # The _execute_step_with_retry method is in WorkflowExecutor, not WorkflowExecutionEngine
        # So we test that the engine can create executors that will have retry capabilities
        assert hasattr(workflow_engine, 'start_workflow')
        assert hasattr(workflow_engine, 'active_executors')
    
    @pytest.mark.asyncio
    async def test_global_retry_manager_consistency(self, orchestrator):
        """Test that global retry manager is consistent across components."""
        # Get global retry manager
        global_manager = get_retry_manager()
        
        # Verify it's the same instance used by orchestrator components
        # (Note: This might not be the same instance if orchestrator creates its own,
        # but the functionality should be consistent)
        assert isinstance(global_manager, UnifiedRetryManager)
        
        # Test that global manager works correctly
        call_count = 0
        
        async def test_global_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Test error")
            return "global_success"
        
        result = await global_manager.retry_async(test_global_operation)
        assert result == "global_success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_observability_integration(self, orchestrator):
        """Test that retry observability is properly integrated."""
        retry_manager = orchestrator.retry_manager
        
        # Verify observability manager is available
        assert hasattr(retry_manager, 'observability_manager')
        assert retry_manager.observability_manager is not None
        
        # Test operation with observability
        async def observable_operation():
            return {"observability": "test"}
        
        result = await retry_manager.retry_async(
            observable_operation,
            config=RetryConfig(operation_name="observability_test")
        )
        
        assert result["observability"] == "test"
        
        # Check that metrics were recorded
        metrics = retry_manager.get_retry_metrics("observability_test")
        assert metrics is not None
    
    @pytest.mark.asyncio
    async def test_component_loader_retry_manager_registration(self, component_loader):
        """Test that UnifiedRetryManager is properly registered in component loader."""
        from prompt_improver.ml.orchestration.core.component_registry import ComponentTier
        
        # Test loading the unified retry manager component
        loaded_component = await component_loader.load_component(
            "unified_retry_manager", 
            ComponentTier.TIER_4_PERFORMANCE
        )
        
        assert loaded_component is not None
        assert loaded_component.name == "unified_retry_manager"
        assert loaded_component.component_class.__name__ == "UnifiedRetryManager"
        
        # Test initialization
        success = await component_loader.initialize_component("unified_retry_manager")
        assert success is True
        
        # Verify instance is created
        assert loaded_component.instance is not None
        assert isinstance(loaded_component.instance, UnifiedRetryManager)
    
    @pytest.mark.asyncio
    async def test_end_to_end_retry_workflow(self, orchestrator):
        """Test end-to-end retry workflow through orchestrator."""
        # This test simulates a complete workflow with retry logic
        
        attempt_count = 0
        
        async def simulated_ml_operation():
            nonlocal attempt_count
            attempt_count += 1
            
            # Simulate transient failures
            if attempt_count < 3:
                raise ConnectionError(f"Simulated failure on attempt {attempt_count}")
            
            return {
                "model_trained": True,
                "accuracy": 0.95,
                "attempts": attempt_count
            }
        
        # Execute through orchestrator with retry
        result = await orchestrator.execute_with_retry(
            simulated_ml_operation,
            "end_to_end_ml_operation",
            max_attempts=5,
            initial_delay_ms=50  # Fast retry for testing
        )
        
        # Verify successful execution after retries
        assert result["model_trained"] is True
        assert result["accuracy"] == 0.95
        assert result["attempts"] == 3
        assert attempt_count == 3
        
        # Verify metrics collection
        metrics = orchestrator.retry_manager.get_retry_metrics("end_to_end_ml_operation")
        assert metrics is not None
        assert metrics["total_attempts"] >= 3
        assert metrics["successful_attempts"] >= 1
        assert metrics["success_rate"] > 0
