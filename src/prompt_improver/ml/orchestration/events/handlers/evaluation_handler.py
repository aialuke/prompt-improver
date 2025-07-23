"""
Evaluation Event Handler for ML Pipeline Orchestration.

Handles evaluation-related events and coordinates evaluation workflows.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from ..event_types import EventType, MLEvent


class EvaluationEventHandler:
    """
    Handles evaluation-related events in the ML pipeline orchestration.
    
    Responds to evaluation lifecycle events and coordinates evaluation
    activities across components.
    """
    
    def __init__(self, orchestrator=None, event_bus=None):
        """Initialize the evaluation event handler."""
        self.orchestrator = orchestrator
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        
        # Event handling statistics
        self.events_processed = 0
        self.events_failed = 0
        self.last_event_time: Optional[datetime] = None
        
        # Evaluation state tracking
        self.active_evaluations = {}
        self.evaluation_history = []
        
    async def handle_event(self, event: MLEvent) -> None:
        """Handle an evaluation-related event."""
        self.logger.debug(f"Handling evaluation event: {event.event_type}")
        
        try:
            self.events_processed += 1
            self.last_event_time = datetime.now(timezone.utc)
            
            if event.event_type == EventType.EVALUATION_STARTED:
                await self._handle_evaluation_started(event)
            elif event.event_type == EventType.EVALUATION_COMPLETED:
                await self._handle_evaluation_completed(event)
            elif event.event_type == EventType.EVALUATION_FAILED:
                await self._handle_evaluation_failed(event)
            elif event.event_type == EventType.EXPERIMENT_CREATED:
                await self._handle_experiment_created(event)
            elif event.event_type == EventType.EXPERIMENT_COMPLETED:
                await self._handle_experiment_completed(event)
            elif event.event_type == EventType.DATA_VALIDATION_COMPLETED:
                await self._handle_data_validation_completed(event)
            else:
                self.logger.warning(f"Unknown evaluation event type: {event.event_type}")
                
        except Exception as e:
            self.events_failed += 1
            self.logger.error(f"Error handling evaluation event {event.event_type}: {e}")
            
            if self.event_bus:
                await self.event_bus.emit(MLEvent(
                    event_type=EventType.EVENT_HANDLER_ERROR,
                    source="evaluation_handler",
                    data={
                        "original_event": event.event_type.value,
                        "error_message": str(e),
                        "handler": "evaluation_handler"
                    }
                ))
    
    async def _handle_evaluation_started(self, event: MLEvent) -> None:
        """Handle evaluation started event."""
        self.logger.info(f"Evaluation started: {event.data}")
        
        workflow_id = event.data.get("workflow_id")
        if workflow_id:
            self.active_evaluations[workflow_id] = {
                "started_at": datetime.now(timezone.utc),
                "status": "running",
                "type": event.data.get("evaluation_type", "standard"),
                "parameters": event.data.get("parameters", {}),
                "source": event.source,
                "experiments": [],
                "validation_results": {},
                "metrics": {}
            }
        
        # Allocate evaluation resources
        if self.orchestrator and hasattr(self.orchestrator, 'resource_manager'):
            try:
                await self.orchestrator.resource_manager.allocate_evaluation_resources(
                    workflow_id, event.data.get("parameters", {})
                )
            except Exception as e:
                self.logger.warning(f"Failed to allocate evaluation resources: {e}")
        
        self.logger.info(f"Evaluation session {workflow_id} registered")
    
    async def _handle_evaluation_completed(self, event: MLEvent) -> None:
        """Handle evaluation completed event."""
        self.logger.info(f"Evaluation completed: {event.data}")
        
        workflow_id = event.data.get("workflow_id")
        if workflow_id in self.active_evaluations:
            evaluation = self.active_evaluations[workflow_id]
            evaluation["completed_at"] = datetime.now(timezone.utc)
            evaluation["status"] = "completed"
            evaluation["results"] = event.data.get("results", {})
            evaluation["final_metrics"] = event.data.get("metrics", {})
            
            # Move to history
            self.evaluation_history.append(evaluation.copy())
        
        # Release evaluation resources
        if self.orchestrator and hasattr(self.orchestrator, 'resource_manager'):
            try:
                await self.orchestrator.resource_manager.release_evaluation_resources(workflow_id)
            except Exception as e:
                self.logger.warning(f"Failed to release evaluation resources: {e}")
        
        # Trigger deployment if evaluation passed thresholds
        results = event.data.get("results", {})
        auto_deploy = event.data.get("auto_deploy", False)
        if auto_deploy and results.get("passed_thresholds", False) and self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.DEPLOYMENT_REQUESTED,
                source="evaluation_handler",
                data={
                    "workflow_id": workflow_id,
                    "evaluation_results": results,
                    "trigger": "evaluation_success"
                }
            ))
        
        self.logger.info(f"Evaluation session {workflow_id} completed successfully")
    
    async def _handle_evaluation_failed(self, event: MLEvent) -> None:
        """Handle evaluation failed event."""
        self.logger.error(f"Evaluation failed: {event.data}")
        
        workflow_id = event.data.get("workflow_id")
        if workflow_id in self.active_evaluations:
            evaluation = self.active_evaluations[workflow_id]
            evaluation["failed_at"] = datetime.now(timezone.utc)
            evaluation["status"] = "failed"
            evaluation["error_message"] = event.data.get("error_message")
            
            # Move to history
            self.evaluation_history.append(evaluation.copy())
        
        # Release evaluation resources
        if self.orchestrator and hasattr(self.orchestrator, 'resource_manager'):
            try:
                await self.orchestrator.resource_manager.release_evaluation_resources(workflow_id)
            except Exception as e:
                self.logger.warning(f"Failed to release evaluation resources after failure: {e}")
        
        # Trigger failure analysis
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.FAILURE_ANALYSIS_REQUESTED,
                source="evaluation_handler",
                data={
                    "workflow_id": workflow_id,
                    "failure_type": "evaluation_failure",
                    "error_message": event.data.get("error_message"),
                    "context": event.data
                }
            ))
    
    async def _handle_experiment_created(self, event: MLEvent) -> None:
        """Handle experiment created event."""
        self.logger.info(f"Experiment created: {event.data}")
        
        workflow_id = event.data.get("workflow_id")
        experiment_id = event.data.get("experiment_id")
        
        if workflow_id in self.active_evaluations and experiment_id:
            evaluation = self.active_evaluations[workflow_id]
            evaluation["experiments"].append({
                "experiment_id": experiment_id,
                "created_at": datetime.now(timezone.utc),
                "type": event.data.get("experiment_type"),
                "status": "created"
            })
        
        # Log experiment tracking
        self.logger.info(f"Experiment {experiment_id} registered for evaluation {workflow_id}")
    
    async def _handle_experiment_completed(self, event: MLEvent) -> None:
        """Handle experiment completed event."""
        self.logger.info(f"Experiment completed: {event.data}")
        
        workflow_id = event.data.get("workflow_id")
        experiment_id = event.data.get("experiment_id")
        
        if workflow_id in self.active_evaluations:
            evaluation = self.active_evaluations[workflow_id]
            
            # Update experiment status
            for experiment in evaluation["experiments"]:
                if experiment["experiment_id"] == experiment_id:
                    experiment["completed_at"] = datetime.now(timezone.utc)
                    experiment["status"] = "completed"
                    experiment["results"] = event.data.get("results", {})
                    break
            
            # Aggregate experiment results
            experiment_results = event.data.get("results", {})
            if experiment_results:
                evaluation["metrics"].update(experiment_results)
        
        # Check if all experiments are complete for the evaluation
        if workflow_id in self.active_evaluations:
            evaluation = self.active_evaluations[workflow_id]
            all_experiments_complete = all(
                exp["status"] in ["completed", "failed"] 
                for exp in evaluation["experiments"]
            )
            
            if all_experiments_complete and evaluation["status"] == "running":
                # Trigger evaluation completion
                if self.event_bus:
                    await self.event_bus.emit(MLEvent(
                        event_type=EventType.EVALUATION_AGGREGATION_REQUESTED,
                        source="evaluation_handler",
                        data={
                            "workflow_id": workflow_id,
                            "experiments_complete": True
                        }
                    ))
    
    async def _handle_data_validation_completed(self, event: MLEvent) -> None:
        """Handle data validation completed event."""
        self.logger.info(f"Data validation completed: {event.data}")
        
        workflow_id = event.data.get("workflow_id")
        if workflow_id in self.active_evaluations:
            evaluation = self.active_evaluations[workflow_id]
            evaluation["validation_results"] = event.data.get("validation_results", {})
            evaluation["data_validated_at"] = datetime.now(timezone.utc)
        
        # Continue with evaluation if data validation passed
        validation_passed = event.data.get("validation_results", {}).get("passed", True)
        if validation_passed and self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.EVALUATION_CONTINUE,
                source="evaluation_handler",
                data={
                    "workflow_id": workflow_id,
                    "validation_passed": True
                }
            ))
        elif not validation_passed:
            # Fail the evaluation if data validation failed
            if self.event_bus:
                await self.event_bus.emit(MLEvent(
                    event_type=EventType.EVALUATION_FAILED,
                    source="evaluation_handler",
                    data={
                        "workflow_id": workflow_id,
                        "error_message": "Data validation failed",
                        "validation_results": event.data.get("validation_results", {})
                    }
                ))
    
    async def get_evaluation_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an evaluation session."""
        return self.active_evaluations.get(workflow_id, {}).copy()
    
    async def list_active_evaluations(self) -> Dict[str, Dict[str, Any]]:
        """List all active evaluation sessions."""
        active_evaluations = {}
        for workflow_id, evaluation in self.active_evaluations.items():
            if evaluation.get("status") == "running":
                active_evaluations[workflow_id] = evaluation.copy()
        return active_evaluations
    
    async def get_handler_statistics(self) -> Dict[str, Any]:
        """Get event handler statistics."""
        return {
            "events_processed": self.events_processed,
            "events_failed": self.events_failed,
            "success_rate": (self.events_processed - self.events_failed) / max(self.events_processed, 1),
            "last_event_time": self.last_event_time,
            "active_evaluations": len([e for e in self.active_evaluations.values() if e.get("status") == "running"]),
            "total_evaluations": len(self.active_evaluations),
            "evaluation_history_count": len(self.evaluation_history)
        }
    
    def get_supported_events(self) -> List[EventType]:
        """Get list of supported event types."""
        return [
            EventType.EVALUATION_STARTED,
            EventType.EVALUATION_COMPLETED,
            EventType.EVALUATION_FAILED,
            EventType.EXPERIMENT_CREATED,
            EventType.EXPERIMENT_COMPLETED,
            EventType.DATA_VALIDATION_COMPLETED
        ]