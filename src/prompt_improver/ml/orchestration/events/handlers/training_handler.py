"""
Training Event Handler for ML Pipeline Orchestration.

Handles training-related events and coordinates responses across components.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from ..event_types import EventType, MLEvent

class TrainingEventHandler:
    """
    Handles training-related events in the ML pipeline orchestration.
    
    Responds to training lifecycle events and coordinates appropriate actions
    across training components.
    """
    
    def __init__(self, orchestrator=None, event_bus=None):
        """Initialize the training event handler."""
        self.orchestrator = orchestrator
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        
        # Event handling statistics
        self.events_processed = 0
        self.events_failed = 0
        self.last_event_time: Optional[datetime] = None
        
        # Training state tracking
        self.active_training_sessions = {}
        
    async def handle_event(self, event: MLEvent) -> None:
        """Handle a training-related event."""
        self.logger.debug(f"Handling training event: {event.event_type}")
        
        try:
            self.events_processed += 1
            self.last_event_time = datetime.now(timezone.utc)
            
            if event.event_type == EventType.TRAINING_STARTED:
                await self._handle_training_started(event)
            elif event.event_type == EventType.TRAINING_DATA_LOADED:
                await self._handle_training_data_loaded(event)
            elif event.event_type == EventType.TRAINING_COMPLETED:
                await self._handle_training_completed(event)
            elif event.event_type == EventType.TRAINING_FAILED:
                await self._handle_training_failed(event)
            elif event.event_type == EventType.MODEL_TRAINED:
                await self._handle_model_trained(event)
            elif event.event_type == EventType.HYPERPARAMETER_UPDATE:
                await self._handle_hyperparameter_update(event)
            else:
                self.logger.warning(f"Unknown training event type: {event.event_type}")
                
        except Exception as e:
            self.events_failed += 1
            self.logger.error(f"Error handling training event {event.event_type}: {e}")
            
            # Emit error event
            if self.event_bus:
                await self.event_bus.emit(MLEvent(
                    event_type=EventType.EVENT_HANDLER_ERROR,
                    source="training_handler",
                    data={
                        "original_event": event.event_type.value,
                        "error_message": str(e),
                        "handler": "training_handler"
                    }
                ))
    
    async def _handle_training_started(self, event: MLEvent) -> None:
        """Handle training started event."""
        self.logger.info(f"Training started: {event.data}")
        
        workflow_id = event.data.get("workflow_id")
        if workflow_id:
            self.active_training_sessions[workflow_id] = {
                "started_at": datetime.now(timezone.utc),
                "status": "training",
                "parameters": event.data.get("parameters", {}),
                "source": event.source
            }
        
        # Notify resource manager about training start
        if self.orchestrator and hasattr(self.orchestrator, 'resource_manager'):
            try:
                await self.orchestrator.resource_manager.allocate_training_resources(
                    workflow_id, event.data.get("parameters", {})
                )
            except Exception as e:
                self.logger.warning(f"Failed to allocate training resources: {e}")
        
        # Log training metrics
        self.logger.info(f"Training session {workflow_id} registered and resources allocated")
    
    async def _handle_training_data_loaded(self, event: MLEvent) -> None:
        """Handle training data loaded event."""
        self.logger.info(f"Training data loaded: {event.data}")
        
        workflow_id = event.data.get("workflow_id")
        if workflow_id in self.active_training_sessions:
            self.active_training_sessions[workflow_id]["data_loaded_at"] = datetime.now(timezone.utc)
            self.active_training_sessions[workflow_id]["data_size"] = event.data.get("data_size")
        
        # Trigger data validation if configured
        data_validation_enabled = event.data.get("parameters", {}).get("validate_data", True)
        if data_validation_enabled and self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.DATA_VALIDATION_REQUESTED,
                source="training_handler",
                data={
                    "workflow_id": workflow_id,
                    "data_source": event.data.get("data_source"),
                    "validation_type": "training_data"
                }
            ))
    
    async def _handle_training_completed(self, event: MLEvent) -> None:
        """Handle training completed event."""
        self.logger.info(f"Training completed: {event.data}")
        
        workflow_id = event.data.get("workflow_id")
        if workflow_id in self.active_training_sessions:
            session = self.active_training_sessions[workflow_id]
            session["completed_at"] = datetime.now(timezone.utc)
            session["status"] = "completed"
            session["duration"] = event.data.get("duration")
            
            # Calculate training metrics
            if "started_at" in session:
                actual_duration = (session["completed_at"] - session["started_at"]).total_seconds()
                session["actual_duration"] = actual_duration
        
        # Release training resources
        if self.orchestrator and hasattr(self.orchestrator, 'resource_manager'):
            try:
                await self.orchestrator.resource_manager.release_training_resources(workflow_id)
            except Exception as e:
                self.logger.warning(f"Failed to release training resources: {e}")
        
        # Trigger model evaluation if configured
        auto_evaluate = event.data.get("auto_evaluate", True)
        if auto_evaluate and self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.EVALUATION_REQUESTED,
                source="training_handler",
                data={
                    "workflow_id": workflow_id,
                    "model_id": event.data.get("model_id"),
                    "trigger": "training_completion"
                }
            ))
        
        self.logger.info(f"Training session {workflow_id} completed successfully")
    
    async def _handle_training_failed(self, event: MLEvent) -> None:
        """Handle training failed event."""
        self.logger.error(f"Training failed: {event.data}")
        
        workflow_id = event.data.get("workflow_id")
        if workflow_id in self.active_training_sessions:
            session = self.active_training_sessions[workflow_id]
            session["failed_at"] = datetime.now(timezone.utc)
            session["status"] = "failed"
            session["error_message"] = event.data.get("error_message")
        
        # Release training resources
        if self.orchestrator and hasattr(self.orchestrator, 'resource_manager'):
            try:
                await self.orchestrator.resource_manager.release_training_resources(workflow_id)
            except Exception as e:
                self.logger.warning(f"Failed to release training resources after failure: {e}")
        
        # Trigger failure analysis if configured
        analyze_failures = event.data.get("analyze_failures", True)
        if analyze_failures and self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.FAILURE_ANALYSIS_REQUESTED,
                source="training_handler",
                data={
                    "workflow_id": workflow_id,
                    "failure_type": "training_failure",
                    "error_message": event.data.get("error_message"),
                    "context": event.data
                }
            ))
    
    async def _handle_model_trained(self, event: MLEvent) -> None:
        """Handle model trained event."""
        self.logger.info(f"Model trained: {event.data}")
        
        model_id = event.data.get("model_id")
        workflow_id = event.data.get("workflow_id")
        
        # Update session with model information
        if workflow_id in self.active_training_sessions:
            session = self.active_training_sessions[workflow_id]
            session["model_id"] = model_id
            session["model_metrics"] = event.data.get("metrics", {})
        
        # Trigger model registration if configured
        auto_register = event.data.get("auto_register", True)
        if auto_register and self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.MODEL_REGISTRATION_REQUESTED,
                source="training_handler",
                data={
                    "model_id": model_id,
                    "workflow_id": workflow_id,
                    "model_metadata": event.data.get("model_metadata", {}),
                    "trigger": "training_completion"
                }
            ))
    
    async def _handle_hyperparameter_update(self, event: MLEvent) -> None:
        """Handle hyperparameter update event."""
        self.logger.debug(f"Hyperparameter update: {event.data}")
        
        workflow_id = event.data.get("workflow_id")
        if workflow_id in self.active_training_sessions:
            session = self.active_training_sessions[workflow_id]
            if "hyperparameter_updates" not in session:
                session["hyperparameter_updates"] = []
            
            session["hyperparameter_updates"].append({
                "timestamp": datetime.now(timezone.utc),
                "updates": event.data.get("hyperparameters", {}),
                "source": event.source
            })
        
        # Log hyperparameter optimization progress
        optimization_progress = event.data.get("optimization_progress")
        if optimization_progress:
            self.logger.info(f"Hyperparameter optimization progress for {workflow_id}: {optimization_progress}")
    
    async def get_training_session_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a training session."""
        return self.active_training_sessions.get(workflow_id, {}).copy()
    
    async def list_active_training_sessions(self) -> Dict[str, Dict[str, Any]]:
        """List all active training sessions."""
        active_sessions = {}
        for workflow_id, session in self.active_training_sessions.items():
            if session.get("status") in ["training", "data_loading"]:
                active_sessions[workflow_id] = session.copy()
        return active_sessions
    
    async def get_handler_statistics(self) -> Dict[str, Any]:
        """Get event handler statistics."""
        return {
            "events_processed": self.events_processed,
            "events_failed": self.events_failed,
            "success_rate": (self.events_processed - self.events_failed) / max(self.events_processed, 1),
            "last_event_time": self.last_event_time,
            "active_sessions": len([s for s in self.active_training_sessions.values() if s.get("status") in ["training", "data_loading"]]),
            "total_sessions": len(self.active_training_sessions)
        }
    
    async def cleanup_completed_sessions(self, max_history: int = 100) -> None:
        """Clean up completed training sessions to prevent memory growth."""
        completed_sessions = [
            (workflow_id, session) for workflow_id, session in self.active_training_sessions.items()
            if session.get("status") in ["completed", "failed"]
        ]
        
        # Sort by completion time and keep only the most recent
        completed_sessions.sort(key=lambda x: x[1].get("completed_at", x[1].get("failed_at", datetime.min)))
        
        if len(completed_sessions) > max_history:
            sessions_to_remove = completed_sessions[:-max_history]
            for workflow_id, _ in sessions_to_remove:
                del self.active_training_sessions[workflow_id]
            
            self.logger.info(f"Cleaned up {len(sessions_to_remove)} completed training sessions")
    
    def get_supported_events(self) -> List[EventType]:
        """Get list of supported event types."""
        return [
            EventType.TRAINING_STARTED,
            EventType.TRAINING_DATA_LOADED,
            EventType.TRAINING_COMPLETED,
            EventType.TRAINING_FAILED,
            EventType.MODEL_TRAINED,
            EventType.HYPERPARAMETER_UPDATE
        ]