"""
Tests for Evaluation Pipeline Manager.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

from prompt_improver.ml.orchestration.coordinators.evaluation_pipeline_manager import (
    EvaluationPipelineManager
)
from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import PipelineState
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from prompt_improver.ml.orchestration.events.event_types import EventType, MLEvent


class TestEvaluationPipelineManager:
    """Test suite for Evaluation Pipeline Manager."""
    
    @pytest.fixture
    async def manager(self):
        """Create manager instance for testing."""
        config = OrchestratorConfig()
        
        # Mock dependencies
        mock_event_bus = Mock()
        mock_event_bus.emit = AsyncMock()
        
        mock_resource_manager = Mock()
        mock_resource_manager.allocate_resources = AsyncMock(return_value=Mock(allocation_id="eval-alloc"))
        mock_resource_manager.deallocate_resources = AsyncMock(return_value=True)
        
        mock_component_registry = Mock()
        mock_component_registry.get_component = AsyncMock()
        
        manager = EvaluationPipelineManager(config, mock_event_bus, mock_resource_manager, mock_component_registry)
        await manager.initialize()
        
        yield manager
        
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert manager._is_initialized is True
        assert manager.active_evaluations == {}
        assert manager.evaluation_history == []
    
    @pytest.mark.asyncio
    async def test_start_evaluation_pipeline(self, manager):
        """Test starting an evaluation pipeline."""
        workflow_id = "eval-workflow-123"
        parameters = {
            "evaluation_type": "a_b_testing",
            "test_data": "validation_set.json",
            "metrics": ["accuracy", "precision", "recall"]
        }
        
        # Mock component responses
        manager.component_registry.get_component.return_value = Mock(
            execute=AsyncMock(return_value={"status": "success", "data": {"evaluated": 100}})
        )
        
        result = await manager.start_evaluation_pipeline(workflow_id, parameters)
        
        assert result is not None
        assert result.workflow_id == workflow_id
        assert result.state == WorkflowState.RUNNING
        assert result.current_step == EvaluationStep.EXPERIMENT_SETUP
    
    @pytest.mark.asyncio
    async def test_a_b_testing_pipeline(self, manager):
        """Test A/B testing evaluation pipeline."""
        workflow_id = "ab-test-456"
        parameters = {
            "evaluation_type": EvaluationType.A_B_TESTING,
            "control_group": "model_v1",
            "treatment_group": "model_v2",
            "sample_size": 1000,
            "confidence_level": 0.95
        }
        
        # Mock ExperimentOrchestrator component
        mock_experiment_orchestrator = Mock()
        mock_experiment_orchestrator.execute = AsyncMock(return_value={
            "status": "success",
            "data": {
                "experiment_id": "ab-exp-123",
                "control_performance": 0.82,
                "treatment_performance": 0.87,
                "statistical_significance": True,
                "p_value": 0.023,
                "confidence_interval": [0.02, 0.08]
            }
        })
        manager.component_registry.get_component.return_value = mock_experiment_orchestrator
        
        result = await manager._execute_ab_testing(workflow_id, parameters)
        
        assert result["status"] == "success"
        assert result["data"]["statistical_significance"] is True
        assert result["data"]["p_value"] < 0.05
        
        # Verify ExperimentOrchestrator was called
        manager.component_registry.get_component.assert_called_with("experiment_orchestrator")
    
    @pytest.mark.asyncio
    async def test_statistical_validation_pipeline(self, manager):
        """Test statistical validation pipeline."""
        workflow_id = "stat-val-789"
        parameters = {
            "evaluation_type": EvaluationType.STATISTICAL_VALIDATION,
            "validation_methods": ["cross_validation", "bootstrap"],
            "folds": 5,
            "bootstrap_samples": 1000
        }
        
        # Mock statistical validator component
        mock_validator = Mock()
        mock_validator.execute = AsyncMock(return_value={
            "status": "success",
            "data": {
                "cross_validation_score": 0.89,
                "cross_validation_std": 0.03,
                "bootstrap_confidence_interval": [0.85, 0.93],
                "validation_passed": True
            }
        })
        manager.component_registry.get_component.return_value = mock_validator
        
        result = await manager._execute_statistical_validation(workflow_id, parameters)
        
        assert result["status"] == "success"
        assert result["data"]["validation_passed"] is True
        assert result["data"]["cross_validation_score"] == 0.89
        
        # Verify correct component was called
        manager.component_registry.get_component.assert_called_with("advanced_statistical_validator")
    
    @pytest.mark.asyncio
    async def test_pattern_analysis_pipeline(self, manager):
        """Test pattern analysis pipeline."""
        workflow_id = "pattern-analysis-321"
        parameters = {
            "evaluation_type": EvaluationType.PATTERN_ANALYSIS,
            "analysis_methods": ["causal_inference", "pattern_significance"],
            "data_source": "training_results.json"
        }
        
        # Mock pattern analyzer component
        mock_analyzer = Mock()
        mock_analyzer.execute = AsyncMock(return_value={
            "status": "success",
            "data": {
                "significant_patterns": 15,
                "causal_relationships": 8,
                "pattern_confidence": 0.92,
                "insights": ["Pattern A improves accuracy", "Feature B is most important"]
            }
        })
        manager.component_registry.get_component.return_value = mock_analyzer
        
        result = await manager._execute_pattern_analysis(workflow_id, parameters)
        
        assert result["status"] == "success"
        assert result["data"]["significant_patterns"] == 15
        assert result["data"]["causal_relationships"] == 8
        
        # Verify correct component was called
        manager.component_registry.get_component.assert_called_with("pattern_significance_analyzer")
    
    @pytest.mark.asyncio
    async def test_complete_evaluation_pipeline(self, manager):
        """Test complete evaluation pipeline execution."""
        workflow_id = "complete-eval-654"
        parameters = {
            "evaluation_type": EvaluationType.A_B_TESTING,
            "control_model": "baseline",
            "treatment_model": "optimized",
            "metrics": ["accuracy", "f1_score", "auc"]
        }
        
        # Mock successful component responses for each step
        mock_component = Mock()
        mock_component.execute = AsyncMock(side_effect=[
            # Experiment setup
            {"status": "success", "data": {"experiment_configured": True, "groups_created": 2}},
            # Statistical analysis
            {"status": "success", "data": {"statistical_power": 0.8, "significance_detected": True}},
            # Result aggregation
            {"status": "success", "data": {"final_results": {"winner": "treatment", "improvement": 0.05}}}
        ])
        manager.component_registry.get_component.return_value = mock_component
        
        # Start evaluation
        evaluation = await manager.start_evaluation_pipeline(workflow_id, parameters)
        
        # Execute complete pipeline
        await manager._execute_evaluation_pipeline(workflow_id, parameters)
        
        # Verify all steps were executed
        assert mock_component.execute.call_count == 3
        
        # Check final evaluation state
        status = await manager.get_evaluation_status(workflow_id)
        assert status.state == WorkflowState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_evaluation_metrics_collection(self, manager):
        """Test evaluation metrics collection."""
        workflow_id = "metrics-eval"
        
        # Create evaluation
        await manager.start_evaluation_pipeline(workflow_id, {"evaluation_type": "a_b_testing"})
        
        # Mock evaluation metrics
        metrics = EvaluationMetrics(
            workflow_id=workflow_id,
            evaluation_type=EvaluationType.A_B_TESTING,
            accuracy=0.91,
            precision=0.89,
            recall=0.88,
            f1_score=0.885,
            statistical_significance=True,
            p_value=0.012,
            effect_size=0.15
        )
        
        # Record metrics
        await manager._record_evaluation_metrics(workflow_id, metrics)
        
        # Retrieve metrics
        recorded_metrics = await manager.get_evaluation_metrics(workflow_id)
        
        assert recorded_metrics is not None
        assert recorded_metrics.accuracy == 0.91
        assert recorded_metrics.statistical_significance is True
        assert recorded_metrics.p_value == 0.012
    
    @pytest.mark.asyncio
    async def test_evaluation_failure_handling(self, manager):
        """Test evaluation failure handling."""
        workflow_id = "fail-eval"
        parameters = {"evaluation_type": "invalid_type"}
        
        # Mock component failure
        mock_component = Mock()
        mock_component.execute = AsyncMock(side_effect=Exception("Invalid evaluation type"))
        manager.component_registry.get_component.return_value = mock_component
        
        # Start evaluation
        evaluation = await manager.start_evaluation_pipeline(workflow_id, parameters)
        
        # Execute pipeline - should handle failure gracefully
        await manager._execute_evaluation_pipeline(workflow_id, parameters)
        
        # Check evaluation was marked as failed
        status = await manager.get_evaluation_status(workflow_id)
        assert status.state == WorkflowState.FAILED
        assert status.error is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self, manager):
        """Test handling multiple concurrent evaluations."""
        evaluation_ids = ["concurrent-eval-1", "concurrent-eval-2", "concurrent-eval-3"]
        parameters = {"evaluation_type": EvaluationType.STATISTICAL_VALIDATION, "data": "test.json"}
        
        # Mock component responses
        mock_component = Mock()
        mock_component.execute = AsyncMock(return_value={"status": "success", "data": {}})
        manager.component_registry.get_component.return_value = mock_component
        
        # Start multiple evaluations concurrently
        tasks = [
            manager.start_evaluation_pipeline(eval_id, parameters)
            for eval_id in evaluation_ids
        ]
        
        evaluations = await asyncio.gather(*tasks)
        
        # All evaluations should be started
        assert len(evaluations) == 3
        assert all(e.state == WorkflowState.RUNNING for e in evaluations)
        
        # Check active evaluations
        active = await manager.list_active_evaluations()
        assert len(active) == 3
    
    @pytest.mark.asyncio
    async def test_experiment_orchestrator_integration(self, manager):
        """Test integration with existing ExperimentOrchestrator."""
        workflow_id = "integration-test"
        parameters = {
            "evaluation_type": EvaluationType.A_B_TESTING,
            "experiment_config": {
                "control_variant": "v1",
                "treatment_variant": "v2",
                "traffic_split": 0.5
            }
        }
        
        # Mock ExperimentOrchestrator with realistic response
        mock_experiment_orchestrator = Mock()
        mock_experiment_orchestrator.execute = AsyncMock(return_value={
            "status": "success",
            "data": {
                "experiment_id": "exp-456",
                "control_metrics": {"accuracy": 0.85, "conversion_rate": 0.12},
                "treatment_metrics": {"accuracy": 0.89, "conversion_rate": 0.15},
                "statistical_analysis": {
                    "significant": True,
                    "p_value": 0.008,
                    "confidence_level": 0.95,
                    "lift": 0.047
                }
            }
        })
        manager.component_registry.get_component.return_value = mock_experiment_orchestrator
        
        # Execute evaluation
        result = await manager._coordinate_with_experiment_orchestrator(workflow_id, parameters)
        
        assert result["status"] == "success"
        assert result["data"]["statistical_analysis"]["significant"] is True
        assert result["data"]["statistical_analysis"]["lift"] > 0
        
        # Verify integration event was emitted
        manager.event_bus.emit.assert_called()
    
    @pytest.mark.asyncio
    async def test_evaluation_step_updates(self, manager):
        """Test evaluation step updates during execution."""
        workflow_id = "step-test"
        parameters = {"evaluation_type": EvaluationType.A_B_TESTING}
        
        # Start evaluation
        evaluation = await manager.start_evaluation_pipeline(workflow_id, parameters)
        assert evaluation.current_step == EvaluationStep.EXPERIMENT_SETUP
        
        # Simulate step progression
        await manager._update_evaluation_step(workflow_id, EvaluationStep.STATISTICAL_ANALYSIS)
        status = await manager.get_evaluation_status(workflow_id)
        assert status.current_step == EvaluationStep.STATISTICAL_ANALYSIS
        
        await manager._update_evaluation_step(workflow_id, EvaluationStep.RESULT_AGGREGATION)
        status = await manager.get_evaluation_status(workflow_id)
        assert status.current_step == EvaluationStep.RESULT_AGGREGATION


class TestEvaluationMetrics:
    """Test suite for EvaluationMetrics."""
    
    def test_metrics_creation(self):
        """Test evaluation metrics creation."""
        metrics = EvaluationMetrics(
            workflow_id="test-eval",
            evaluation_type=EvaluationType.A_B_TESTING,
            accuracy=0.92,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            statistical_significance=True,
            p_value=0.025,
            effect_size=0.12
        )
        
        assert metrics.workflow_id == "test-eval"
        assert metrics.evaluation_type == EvaluationType.A_B_TESTING
        assert metrics.accuracy == 0.92
        assert metrics.statistical_significance is True
        assert metrics.p_value == 0.025
        assert metrics.timestamp is not None
    
    def test_metrics_serialization(self):
        """Test metrics to/from dict conversion."""
        metrics = EvaluationMetrics(
            workflow_id="serialize-eval",
            evaluation_type=EvaluationType.STATISTICAL_VALIDATION,
            accuracy=0.88,
            precision=0.85,
            recall=0.90,
            f1_score=0.875,
            statistical_significance=False,
            p_value=0.08,
            effect_size=0.05
        )
        
        # Convert to dict and back
        metrics_dict = metrics.to_dict()
        restored_metrics = EvaluationMetrics.from_dict(metrics_dict)
        
        assert restored_metrics.workflow_id == metrics.workflow_id
        assert restored_metrics.evaluation_type == metrics.evaluation_type
        assert restored_metrics.accuracy == metrics.accuracy
        assert restored_metrics.statistical_significance == metrics.statistical_significance
    
    def test_metrics_significance_calculation(self):
        """Test statistical significance calculation."""
        # Significant result
        significant_metrics = EvaluationMetrics(
            workflow_id="sig-test",
            evaluation_type=EvaluationType.A_B_TESTING,
            p_value=0.02,
            effect_size=0.15
        )
        assert significant_metrics.is_statistically_significant(alpha=0.05)
        
        # Non-significant result
        non_significant_metrics = EvaluationMetrics(
            workflow_id="non-sig-test",
            evaluation_type=EvaluationType.A_B_TESTING,
            p_value=0.08,
            effect_size=0.03
        )
        assert not non_significant_metrics.is_statistically_significant(alpha=0.05)


if __name__ == "__main__":
    # Run basic smoke test
    async def smoke_test():
        """Basic smoke test for evaluation pipeline manager."""
        print("Running Evaluation Pipeline Manager smoke test...")
        
        # Create mocks
        config = OrchestratorConfig()
        mock_event_bus = Mock()
        mock_event_bus.emit = AsyncMock()
        mock_resource_manager = Mock()
        mock_resource_manager.allocate_resources = AsyncMock(return_value=Mock(allocation_id="test-alloc"))
        mock_resource_manager.deallocate_resources = AsyncMock(return_value=True)
        mock_component_registry = Mock()
        mock_component_registry.get_component = AsyncMock(return_value=Mock(
            execute=AsyncMock(return_value={"status": "success", "data": {}})
        ))
        
        # Create and initialize manager
        manager = EvaluationPipelineManager(config, mock_event_bus, mock_resource_manager, mock_component_registry)
        
        try:
            await manager.initialize()
            print("✓ Manager initialized successfully")
            
            # Test evaluation start
            evaluation = await manager.start_evaluation_pipeline("smoke-test", {"evaluation_type": "a_b_testing"})
            print(f"✓ Evaluation pipeline started: {evaluation.workflow_id}")
            
            # Test status
            status = await manager.get_evaluation_status("smoke-test")
            print(f"✓ Evaluation status: {status.state}")
            
            # Test active evaluations
            active = await manager.list_active_evaluations()
            print(f"✓ Active evaluations: {len(active)}")
            
            # Test stop evaluation
            stopped = await manager.stop_evaluation_pipeline("smoke-test")
            print(f"✓ Evaluation stopped: {stopped}")
            
            print("✓ All basic tests passed!")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            raise
        finally:
            await manager.shutdown()
            print("✓ Manager shut down gracefully")
    
    # Run the smoke test
    asyncio.run(smoke_test())