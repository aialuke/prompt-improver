"""
Evaluation Pipeline Manager for ML Pipeline Orchestration.

Coordinates evaluation components and manages evaluation workflows, including
integration with the specialized ExperimentOrchestrator.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone

from ..events.event_types import EventType, MLEvent

@dataclass
class EvaluationConfig:
    """Configuration for evaluation pipelines."""
    default_timeout: int = 1800  # 30 minutes
    max_concurrent_evaluations: int = 5
    statistical_significance_threshold: float = 0.05
    evaluation_metrics: List[str] = None
    
    def __post_init__(self):
        if self.evaluation_metrics is None:
            self.evaluation_metrics = ["accuracy", "precision", "recall", "f1_score"]

class EvaluationPipelineManager:
    """
    Coordinates evaluation components and manages evaluation workflows.
    
    Integrates with specialized ExperimentOrchestrator while providing
    centralized coordination for all evaluation activities.
    """
    
    def __init__(self, config: EvaluationConfig, event_bus=None, component_registry=None):
        """Initialize the evaluation pipeline manager."""
        self.config = config
        self.event_bus = event_bus
        self.component_registry = component_registry
        self.logger = logging.getLogger(__name__)
        
        # Active evaluation workflows
        self.active_evaluations: Dict[str, Dict[str, Any]] = {}
        
        # Reference to specialized ExperimentOrchestrator (if registered)
        self.experiment_orchestrator = None
        
    async def initialize(self) -> None:
        """Initialize the evaluation pipeline manager."""
        # Try to get reference to ExperimentOrchestrator from component registry
        if self.component_registry:
            experiment_component = await self.component_registry.get_component("experiment_orchestrator")
            if experiment_component:
                self.logger.info("Found registered ExperimentOrchestrator component")
                # In a real implementation, we would get the actual instance
                # For now, we'll simulate this
        
        self.logger.info("Evaluation pipeline manager initialized")
    
    async def start_evaluation_workflow(self, workflow_id: str, parameters: Dict[str, Any]) -> None:
        """Start a new evaluation workflow."""
        self.logger.info(f"Starting evaluation workflow {workflow_id}")
        
        # Register evaluation workflow
        self.active_evaluations[workflow_id] = {
            "status": "running",
            "started_at": datetime.now(timezone.utc),
            "parameters": parameters,
            "current_step": None,
            "evaluation_results": {},
            "experiments": []
        }
        
        try:
            # Step 1: Prepare evaluation data
            await self._prepare_evaluation_data(workflow_id, parameters)
            
            # Step 2: Run statistical validation
            await self._run_statistical_validation(workflow_id, parameters)
            
            # Step 3: Coordinate with ExperimentOrchestrator for A/B testing
            if self.experiment_orchestrator and parameters.get("run_ab_testing", True):
                await self._coordinate_ab_testing(workflow_id, parameters)
            
            # Step 4: Aggregate evaluation results
            await self._aggregate_evaluation_results(workflow_id, parameters)
            
            # Mark workflow as completed
            self.active_evaluations[workflow_id]["status"] = "completed"
            self.active_evaluations[workflow_id]["completed_at"] = datetime.now(timezone.utc)
            
            if self.event_bus:
                await self.event_bus.emit(MLEvent(
                    event_type=EventType.EVALUATION_COMPLETED,
                    source="evaluation_pipeline_manager",
                    data={
                        "workflow_id": workflow_id,
                        "results": self.active_evaluations[workflow_id]["evaluation_results"]
                    }
                ))
            
            self.logger.info(f"Evaluation workflow {workflow_id} completed successfully")
            
        except Exception as e:
            self.active_evaluations[workflow_id]["status"] = "failed"
            self.active_evaluations[workflow_id]["error"] = str(e)
            self.active_evaluations[workflow_id]["completed_at"] = datetime.now(timezone.utc)
            
            if self.event_bus:
                await self.event_bus.emit(MLEvent(
                    event_type=EventType.EVALUATION_FAILED,
                    source="evaluation_pipeline_manager",
                    data={
                        "workflow_id": workflow_id,
                        "error_message": str(e)
                    }
                ))
            
            self.logger.error(f"Evaluation workflow {workflow_id} failed: {e}")
            raise
    
    async def _prepare_evaluation_data(self, workflow_id: str, parameters: Dict[str, Any]) -> None:
        """Prepare evaluation data and metrics."""
        self.logger.info(f"Preparing evaluation data for workflow {workflow_id}")
        self.active_evaluations[workflow_id]["current_step"] = "data_preparation"
        
        # Simulate data preparation
        await asyncio.sleep(0.1)
        
        self.active_evaluations[workflow_id]["evaluation_results"]["data_preparation"] = {
            "status": "completed",
            "data_size": parameters.get("data_size", 1000),
            "metrics_configured": self.config.evaluation_metrics
        }
        
        self.logger.info(f"Evaluation data preparation completed for workflow {workflow_id}")
    
    async def _run_statistical_validation(self, workflow_id: str, parameters: Dict[str, Any]) -> None:
        """Run statistical validation using advanced statistical validators."""
        self.logger.info(f"Running statistical validation for workflow {workflow_id}")
        self.active_evaluations[workflow_id]["current_step"] = "statistical_validation"
        
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.EVALUATION_STARTED,
                source="evaluation_pipeline_manager",
                data={
                    "workflow_id": workflow_id,
                    "step": "statistical_validation"
                }
            ))
        
        # Simulate statistical validation (would call AdvancedStatisticalValidator)
        await asyncio.sleep(0.1)
        
        self.active_evaluations[workflow_id]["evaluation_results"]["statistical_validation"] = {
            "status": "completed",
            "p_value": 0.03,  # Simulated p-value
            "significant": True,
            "confidence_interval": [0.82, 0.89]
        }
        
        self.logger.info(f"Statistical validation completed for workflow {workflow_id}")
    
    async def _coordinate_ab_testing(self, workflow_id: str, parameters: Dict[str, Any]) -> None:
        """Coordinate A/B testing through ExperimentOrchestrator."""
        self.logger.info(f"Coordinating A/B testing for workflow {workflow_id}")
        self.active_evaluations[workflow_id]["current_step"] = "ab_testing"
        
        # In real implementation, this would communicate with the registered ExperimentOrchestrator
        # For now, simulate the coordination
        experiment_id = f"exp_{workflow_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.EXPERIMENT_CREATED,
                source="evaluation_pipeline_manager",
                data={
                    "workflow_id": workflow_id,
                    "experiment_id": experiment_id,
                    "experiment_type": "ab_testing"
                }
            ))
        
        # Simulate A/B test execution
        await asyncio.sleep(0.1)
        
        self.active_evaluations[workflow_id]["experiments"].append(experiment_id)
        self.active_evaluations[workflow_id]["evaluation_results"]["ab_testing"] = {
            "status": "completed",
            "experiment_id": experiment_id,
            "variant_a_performance": 0.85,
            "variant_b_performance": 0.87,
            "winner": "variant_b"
        }
        
        self.logger.info(f"A/B testing coordination completed for workflow {workflow_id}")
    
    async def _aggregate_evaluation_results(self, workflow_id: str, parameters: Dict[str, Any]) -> None:
        """Aggregate all evaluation results."""
        self.logger.info(f"Aggregating evaluation results for workflow {workflow_id}")
        self.active_evaluations[workflow_id]["current_step"] = "result_aggregation"
        
        evaluation_data = self.active_evaluations[workflow_id]["evaluation_results"]
        
        # Simulate result aggregation
        await asyncio.sleep(0.1)
        
        # Calculate overall score based on different evaluation components
        overall_score = 0.0
        component_count = 0
        
        if "statistical_validation" in evaluation_data:
            overall_score += 0.86  # Simulated score
            component_count += 1
        
        if "ab_testing" in evaluation_data:
            overall_score += evaluation_data["ab_testing"]["variant_b_performance"]
            component_count += 1
        
        if component_count > 0:
            overall_score /= component_count
        
        evaluation_data["aggregated_results"] = {
            "overall_score": overall_score,
            "components_evaluated": component_count,
            "recommendation": "deploy" if overall_score > 0.8 else "retrain"
        }
        
        self.logger.info(f"Evaluation result aggregation completed for workflow {workflow_id}")
    
    async def stop_evaluation(self, workflow_id: str) -> None:
        """Stop a running evaluation workflow."""
        if workflow_id not in self.active_evaluations:
            raise ValueError(f"Evaluation workflow {workflow_id} not found")
        
        self.active_evaluations[workflow_id]["status"] = "stopped"
        self.active_evaluations[workflow_id]["completed_at"] = datetime.now(timezone.utc)
        
        self.logger.info(f"Evaluation workflow {workflow_id} stopped")
    
    async def get_evaluation_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the status of an evaluation workflow."""
        if workflow_id not in self.active_evaluations:
            raise ValueError(f"Evaluation workflow {workflow_id} not found")
        
        return self.active_evaluations[workflow_id].copy()
    
    async def list_active_evaluations(self) -> List[str]:
        """List all active evaluation workflows."""
        return [
            eval_id for eval_id, eval_data in self.active_evaluations.items()
            if eval_data["status"] == "running"
        ]
    
    async def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get results for a specific experiment."""
        for evaluation in self.active_evaluations.values():
            if experiment_id in evaluation.get("experiments", []):
                ab_results = evaluation["evaluation_results"].get("ab_testing", {})
                if ab_results.get("experiment_id") == experiment_id:
                    return ab_results
        
        return None