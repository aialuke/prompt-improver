"""
Optimization Event Handler for ML Pipeline Orchestration.

Handles optimization-related events and coordinates optimization workflows.
"""
import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from ..event_types import EventType, MLEvent

class OptimizationEventHandler:
    """
    Handles optimization-related events in the ML pipeline orchestration.
    
    Responds to optimization lifecycle events and coordinates optimization
    algorithms across components.
    """

    def __init__(self, orchestrator=None, event_bus=None):
        """Initialize the optimization event handler."""
        self.orchestrator = orchestrator
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        self.events_processed = 0
        self.events_failed = 0
        self.last_event_time: Optional[datetime] = None
        self.active_optimizations = {}
        self.optimization_history = []

    async def handle_event(self, event: MLEvent) -> None:
        """Handle an optimization-related event."""
        self.logger.debug('Handling optimization event: %s', event.event_type)
        try:
            self.events_processed += 1
            self.last_event_time = datetime.now(timezone.utc)
            if event.event_type == EventType.OPTIMIZATION_STARTED:
                await self._handle_optimization_started(event)
            elif event.event_type == EventType.OPTIMIZATION_COMPLETED:
                await self._handle_optimization_completed(event)
            elif event.event_type == EventType.OPTIMIZATION_FAILED:
                await self._handle_optimization_failed(event)
            elif event.event_type == EventType.HYPERPARAMETER_UPDATE:
                await self._handle_hyperparameter_update(event)
            elif event.event_type == EventType.OPTIMIZATION_ITERATION:
                await self._handle_optimization_iteration(event)
            else:
                self.logger.warning('Unknown optimization event type: %s', event.event_type)
        except Exception as e:
            self.events_failed += 1
            self.logger.error('Error handling optimization event {event.event_type}: %s', e)
            if self.event_bus:
                await self.event_bus.emit(MLEvent(event_type=EventType.EVENT_HANDLER_ERROR, source='optimization_handler', data={'original_event': event.event_type.value, 'error_message': str(e), 'handler': 'optimization_handler'}))

    async def _handle_optimization_started(self, event: MLEvent) -> None:
        """Handle optimization started event."""
        self.logger.info('Optimization started: %s', event.data)
        workflow_id = event.data.get('workflow_id')
        if workflow_id:
            self.active_optimizations[workflow_id] = {'started_at': datetime.now(timezone.utc), 'status': 'running', 'type': event.data.get('type'), 'parameters': event.data.get('parameters', {}), 'source': event.source, 'iterations': 0, 'best_score': None, 'current_parameters': {}}
        if self.orchestrator and hasattr(self.orchestrator, 'resource_manager'):
            try:
                await self.orchestrator.resource_manager.allocate_optimization_resources(workflow_id, event.data.get('parameters', {}))
            except Exception as e:
                self.logger.warning('Failed to allocate optimization resources: %s', e)
        self.logger.info('Optimization session %s registered', workflow_id)

    async def _handle_optimization_completed(self, event: MLEvent) -> None:
        """Handle optimization completed event."""
        self.logger.info('Optimization completed: %s', event.data)
        workflow_id = event.data.get('workflow_id')
        if workflow_id in self.active_optimizations:
            optimization = self.active_optimizations[workflow_id]
            optimization['completed_at'] = datetime.now(timezone.utc)
            optimization['status'] = 'completed'
            optimization['best_result'] = event.data.get('best_result')
            optimization['final_score'] = event.data.get('final_score')
            self.optimization_history.append(optimization.copy())
        if self.orchestrator and hasattr(self.orchestrator, 'resource_manager'):
            try:
                await self.orchestrator.resource_manager.release_optimization_resources(workflow_id)
            except Exception as e:
                self.logger.warning('Failed to release optimization resources: %s', e)
        best_result = event.data.get('best_result', {})
        if best_result.get('improved', False) and self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.MODEL_UPDATE_REQUESTED, source='optimization_handler', data={'workflow_id': workflow_id, 'optimization_result': best_result, 'trigger': 'optimization_improvement'}))
        self.logger.info('Optimization session %s completed successfully', workflow_id)

    async def _handle_optimization_failed(self, event: MLEvent) -> None:
        """Handle optimization failed event."""
        self.logger.error('Optimization failed: %s', event.data)
        workflow_id = event.data.get('workflow_id')
        if workflow_id in self.active_optimizations:
            optimization = self.active_optimizations[workflow_id]
            optimization['failed_at'] = datetime.now(timezone.utc)
            optimization['status'] = 'failed'
            optimization['error_message'] = event.data.get('error_message')
            self.optimization_history.append(optimization.copy())
        if self.orchestrator and hasattr(self.orchestrator, 'resource_manager'):
            try:
                await self.orchestrator.resource_manager.release_optimization_resources(workflow_id)
            except Exception as e:
                self.logger.warning('Failed to release optimization resources after failure: %s', e)
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.FAILURE_ANALYSIS_REQUESTED, source='optimization_handler', data={'workflow_id': workflow_id, 'failure_type': 'optimization_failure', 'error_message': event.data.get('error_message'), 'context': event.data}))

    async def _handle_hyperparameter_update(self, event: MLEvent) -> None:
        """Handle hyperparameter update event."""
        self.logger.debug('Hyperparameter update: %s', event.data)
        workflow_id = event.data.get('workflow_id')
        if workflow_id in self.active_optimizations:
            optimization = self.active_optimizations[workflow_id]
            optimization['current_parameters'] = event.data.get('hyperparameters', {})
            optimization['last_update'] = datetime.now(timezone.utc)
            current_score = event.data.get('score')
            if current_score and (optimization['best_score'] is None or current_score > optimization['best_score']):
                optimization['best_score'] = current_score
                optimization['best_parameters'] = event.data.get('hyperparameters', {})

    async def _handle_optimization_iteration(self, event: MLEvent) -> None:
        """Handle optimization iteration event."""
        self.logger.debug('Optimization iteration: %s', event.data)
        workflow_id = event.data.get('workflow_id')
        if workflow_id in self.active_optimizations:
            optimization = self.active_optimizations[workflow_id]
            optimization['iterations'] += 1
            optimization['last_iteration'] = datetime.now(timezone.utc)
            if optimization['iterations'] % 10 == 0:
                self.logger.info("Optimization {workflow_id} progress: {optimization['iterations']} iterations, best score: %s", optimization.get('best_score', 'N/A'))

    async def get_optimization_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an optimization session."""
        return self.active_optimizations.get(workflow_id, {}).copy()

    async def list_active_optimizations(self) -> Dict[str, Dict[str, Any]]:
        """List all active optimization sessions."""
        active_optimizations = {}
        for workflow_id, optimization in self.active_optimizations.items():
            if optimization.get('status') == 'running':
                active_optimizations[workflow_id] = optimization.copy()
        return active_optimizations

    async def get_handler_statistics(self) -> Dict[str, Any]:
        """Get event handler statistics."""
        return {'events_processed': self.events_processed, 'events_failed': self.events_failed, 'success_rate': (self.events_processed - self.events_failed) / max(self.events_processed, 1), 'last_event_time': self.last_event_time, 'active_optimizations': len([o for o in self.active_optimizations.values() if o.get('status') == 'running']), 'total_optimizations': len(self.active_optimizations), 'optimization_history_count': len(self.optimization_history)}

    def get_supported_events(self) -> List[EventType]:
        """Get list of supported event types."""
        return [EventType.OPTIMIZATION_STARTED, EventType.OPTIMIZATION_COMPLETED, EventType.OPTIMIZATION_FAILED, EventType.HYPERPARAMETER_UPDATE, EventType.OPTIMIZATION_ITERATION]
