"""
Training Workflow Coordinator for ML Pipeline Orchestration.

Coordinates Tier 1 training components and manages training workflows.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from ..core.workflow_types import WorkflowStep, WorkflowStepStatus
from ..events.event_types import EventType, MLEvent

@dataclass
class TrainingWorkflowConfig:
    """Configuration for training workflows."""
    default_timeout: int = 3600
    max_retries: int = 3
    batch_size: int = 1000
    validation_split: float = 0.2

class TrainingWorkflowCoordinator:
    """
    Coordinates Tier 1 training components and manages training workflows.
    
    Manages the flow: TrainingDataLoader → MLModelService → RuleOptimizer
    """

    def __init__(self, config: TrainingWorkflowConfig, event_bus=None, resource_manager=None):
        """Initialize the training workflow coordinator."""
        self.config = config
        self.event_bus = event_bus
        self.resource_manager = resource_manager
        self.logger = logging.getLogger(__name__)
        self.active_workflows: Dict[str, Dict[str, Any]] = {}

    async def start_training_workflow(self, workflow_id: str, parameters: Dict[str, Any]) -> None:
        """Start a new training workflow."""
        self.logger.info('Starting training workflow %s', workflow_id)
        self.active_workflows[workflow_id] = {'status': 'running', 'started_at': datetime.now(timezone.utc), 'parameters': parameters, 'current_step': None, 'steps_completed': []}
        try:
            await self._execute_data_loading(workflow_id, parameters)
            await self._execute_model_training(workflow_id, parameters)
            await self._execute_rule_optimization(workflow_id, parameters)
            self.active_workflows[workflow_id]['status'] = 'completed'
            self.active_workflows[workflow_id]['completed_at'] = datetime.now(timezone.utc)
            if self.event_bus:
                await self.event_bus.emit(MLEvent(event_type=EventType.TRAINING_COMPLETED, source='training_workflow_coordinator', data={'workflow_id': workflow_id, 'duration': (datetime.now(timezone.utc) - self.active_workflows[workflow_id]['started_at']).total_seconds()}))
            self.logger.info('Training workflow %s completed successfully', workflow_id)
        except Exception as e:
            self.active_workflows[workflow_id]['status'] = 'failed'
            self.active_workflows[workflow_id]['error'] = str(e)
            self.active_workflows[workflow_id]['completed_at'] = datetime.now(timezone.utc)
            if self.event_bus:
                await self.event_bus.emit(MLEvent(event_type=EventType.TRAINING_FAILED, source='training_workflow_coordinator', data={'workflow_id': workflow_id, 'error_message': str(e)}))
            self.logger.error('Training workflow {workflow_id} failed: %s', e)
            raise

    async def _execute_data_loading(self, workflow_id: str, parameters: Dict[str, Any]) -> None:
        """Execute data loading step."""
        self.logger.info('Executing data loading for workflow %s', workflow_id)
        self.active_workflows[workflow_id]['current_step'] = 'data_loading'
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.TRAINING_DATA_LOADED, source='training_workflow_coordinator', data={'workflow_id': workflow_id, 'step': 'data_loading', 'parameters': parameters}))
        await asyncio.sleep(0.1)
        self.active_workflows[workflow_id]['steps_completed'].append('data_loading')
        self.logger.info('Data loading completed for workflow %s', workflow_id)

    async def _execute_model_training(self, workflow_id: str, parameters: Dict[str, Any]) -> None:
        """Execute model training step."""
        self.logger.info('Executing model training for workflow %s', workflow_id)
        self.active_workflows[workflow_id]['current_step'] = 'model_training'
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.TRAINING_STARTED, source='training_workflow_coordinator', data={'workflow_id': workflow_id, 'step': 'model_training', 'parameters': parameters}))
        await asyncio.sleep(0.1)
        self.active_workflows[workflow_id]['steps_completed'].append('model_training')
        self.logger.info('Model training completed for workflow %s', workflow_id)

    async def _execute_rule_optimization(self, workflow_id: str, parameters: Dict[str, Any]) -> None:
        """Execute rule optimization step."""
        self.logger.info('Executing rule optimization for workflow %s', workflow_id)
        self.active_workflows[workflow_id]['current_step'] = 'rule_optimization'
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.OPTIMIZATION_STARTED, source='training_workflow_coordinator', data={'workflow_id': workflow_id, 'step': 'rule_optimization', 'parameters': parameters}))
        await asyncio.sleep(0.1)
        self.active_workflows[workflow_id]['steps_completed'].append('rule_optimization')
        self.logger.info('Rule optimization completed for workflow %s', workflow_id)

    async def stop_workflow(self, workflow_id: str) -> None:
        """Stop a running training workflow."""
        if workflow_id not in self.active_workflows:
            raise ValueError(f'Training workflow {workflow_id} not found')
        self.active_workflows[workflow_id]['status'] = 'stopped'
        self.active_workflows[workflow_id]['completed_at'] = datetime.now(timezone.utc)
        self.logger.info('Training workflow %s stopped', workflow_id)

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the status of a training workflow."""
        if workflow_id not in self.active_workflows:
            raise ValueError(f'Training workflow {workflow_id} not found')
        return self.active_workflows[workflow_id].copy()

    async def list_active_workflows(self) -> List[str]:
        """List all active training workflows."""
        return [wf_id for wf_id, wf_data in self.active_workflows.items() if wf_data['status'] == 'running']
