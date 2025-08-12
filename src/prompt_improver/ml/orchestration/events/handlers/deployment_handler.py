"""
Deployment Event Handler for ML Pipeline Orchestration.

Handles deployment-related events and coordinates deployment workflows.
"""
import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from ..event_types import EventType, MLEvent

class DeploymentEventHandler:
    """
    Handles deployment-related events in the ML pipeline orchestration.
    
    Responds to deployment lifecycle events and coordinates deployment
    activities across components.
    """

    def __init__(self, orchestrator=None, event_bus=None):
        """Initialize the deployment event handler."""
        self.orchestrator = orchestrator
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        self.events_processed = 0
        self.events_failed = 0
        self.last_event_time: datetime | None = None
        self.active_deployments = {}
        self.deployment_history = []

    async def handle_event(self, event: MLEvent) -> None:
        """Handle a deployment-related event."""
        self.logger.debug('Handling deployment event: %s', event.event_type)
        try:
            self.events_processed += 1
            self.last_event_time = datetime.now(timezone.utc)
            if event.event_type == EventType.DEPLOYMENT_STARTED:
                await self._handle_deployment_started(event)
            elif event.event_type == EventType.DEPLOYMENT_COMPLETED:
                await self._handle_deployment_completed(event)
            elif event.event_type == EventType.DEPLOYMENT_FAILED:
                await self._handle_deployment_failed(event)
            elif event.event_type == EventType.MODEL_DEPLOYED:
                await self._handle_model_deployed(event)
            elif event.event_type == EventType.DEPLOYMENT_HEALTH_CHECK:
                await self._handle_deployment_health_check(event)
            elif event.event_type == EventType.ROLLBACK_TRIGGERED:
                await self._handle_rollback_triggered(event)
            else:
                self.logger.warning('Unknown deployment event type: %s', event.event_type)
        except Exception as e:
            self.events_failed += 1
            self.logger.error('Error handling deployment event {event.event_type}: %s', e)
            if self.event_bus:
                await self.event_bus.emit(MLEvent(event_type=EventType.EVENT_HANDLER_ERROR, source='deployment_handler', data={'original_event': event.event_type.value, 'error_message': str(e), 'handler': 'deployment_handler'}))

    async def _handle_deployment_started(self, event: MLEvent) -> None:
        """Handle deployment started event."""
        self.logger.info('Deployment started: %s', event.data)
        deployment_id = event.data.get('deployment_id')
        if deployment_id:
            self.active_deployments[deployment_id] = {'started_at': datetime.now(timezone.utc), 'status': 'deploying', 'strategy': event.data.get('strategy', 'blue_green'), 'model_version': event.data.get('model_version'), 'parameters': event.data.get('parameters', {}), 'source': event.source, 'health_checks': [], 'deployment_stages': []}
        if self.orchestrator and hasattr(self.orchestrator, 'resource_manager'):
            try:
                await self.orchestrator.resource_manager.allocate_deployment_resources(deployment_id, event.data.get('parameters', {}))
            except Exception as e:
                self.logger.warning('Failed to allocate deployment resources: %s', e)
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.DEPLOYMENT_MONITORING_STARTED, source='deployment_handler', data={'deployment_id': deployment_id, 'monitoring_config': event.data.get('monitoring_config', {})}))
        self.logger.info('Deployment %s registered and monitoring started', deployment_id)

    async def _handle_deployment_completed(self, event: MLEvent) -> None:
        """Handle deployment completed event."""
        self.logger.info('Deployment completed: %s', event.data)
        deployment_id = event.data.get('deployment_id')
        if deployment_id in self.active_deployments:
            deployment = self.active_deployments[deployment_id]
            deployment['completed_at'] = datetime.now(timezone.utc)
            deployment['status'] = 'completed'
            deployment['final_status'] = event.data.get('final_status', 'success')
            deployment['deployment_url'] = event.data.get('deployment_url')
            self.deployment_history.append(deployment.copy())
        if self.orchestrator and hasattr(self.orchestrator, 'resource_manager'):
            try:
                await self.orchestrator.resource_manager.release_deployment_resources(deployment_id)
            except Exception as e:
                self.logger.warning('Failed to release deployment resources: %s', e)
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.POST_DEPLOYMENT_MONITORING_STARTED, source='deployment_handler', data={'deployment_id': deployment_id, 'deployment_url': event.data.get('deployment_url'), 'monitoring_duration': event.data.get('monitoring_duration', 3600)}))
        self.logger.info('Deployment %s completed successfully', deployment_id)

    async def _handle_deployment_failed(self, event: MLEvent) -> None:
        """Handle deployment failed event."""
        self.logger.error('Deployment failed: %s', event.data)
        deployment_id = event.data.get('deployment_id')
        if deployment_id in self.active_deployments:
            deployment = self.active_deployments[deployment_id]
            deployment['failed_at'] = datetime.now(timezone.utc)
            deployment['status'] = 'failed'
            deployment['error_message'] = event.data.get('error_message')
            deployment['rollback_executed'] = event.data.get('rollback_executed', False)
            self.deployment_history.append(deployment.copy())
        if self.orchestrator and hasattr(self.orchestrator, 'resource_manager'):
            try:
                await self.orchestrator.resource_manager.release_deployment_resources(deployment_id)
            except Exception as e:
                self.logger.warning('Failed to release deployment resources after failure: %s', e)
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.DEPLOYMENT_ALERT, source='deployment_handler', data={'deployment_id': deployment_id, 'alert_type': 'deployment_failure', 'error_message': event.data.get('error_message'), 'severity': 'high'}))
            await self.event_bus.emit(MLEvent(event_type=EventType.FAILURE_ANALYSIS_REQUESTED, source='deployment_handler', data={'deployment_id': deployment_id, 'failure_type': 'deployment_failure', 'error_message': event.data.get('error_message'), 'context': event.data}))

    async def _handle_model_deployed(self, event: MLEvent) -> None:
        """Handle model deployed event."""
        self.logger.info('Model deployed: %s', event.data)
        deployment_id = event.data.get('deployment_id')
        if deployment_id in self.active_deployments:
            deployment = self.active_deployments[deployment_id]
            deployment['model_deployed_at'] = datetime.now(timezone.utc)
            deployment['model_metadata'] = event.data.get('model_metadata', {})
            deployment['registry_info'] = event.data.get('registry_info', {})
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.MODEL_REGISTRATION_COMPLETED, source='deployment_handler', data={'deployment_id': deployment_id, 'model_version': event.data.get('model_version'), 'registry_path': event.data.get('registry_path')}))
        self.logger.info('Model deployment %s registered in production registry', deployment_id)

    async def _handle_deployment_health_check(self, event: MLEvent) -> None:
        """Handle deployment health check event."""
        self.logger.debug('Deployment health check: %s', event.data)
        deployment_id = event.data.get('deployment_id')
        if deployment_id in self.active_deployments:
            deployment = self.active_deployments[deployment_id]
            health_check = {'timestamp': datetime.now(timezone.utc), 'status': event.data.get('status', 'unknown'), 'response_time': event.data.get('response_time'), 'error_rate': event.data.get('error_rate'), 'details': event.data.get('details', {})}
            deployment['health_checks'].append(health_check)
            if len(deployment['health_checks']) > 100:
                deployment['health_checks'] = deployment['health_checks'][-100:]
        health_status = event.data.get('status')
        if health_status in ['unhealthy', 'critical'] and self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.DEPLOYMENT_ALERT, source='deployment_handler', data={'deployment_id': deployment_id, 'alert_type': 'health_check_failure', 'health_status': health_status, 'severity': 'high' if health_status == 'critical' else 'medium'}))

    async def _handle_rollback_triggered(self, event: MLEvent) -> None:
        """Handle rollback triggered event."""
        self.logger.warning('Rollback triggered: %s', event.data)
        deployment_id = event.data.get('deployment_id')
        if deployment_id in self.active_deployments:
            deployment = self.active_deployments[deployment_id]
            deployment['rollback_triggered_at'] = datetime.now(timezone.utc)
            deployment['rollback_reason'] = event.data.get('reason')
            deployment['previous_version'] = event.data.get('previous_version')
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.ROLLBACK_EXECUTION_STARTED, source='deployment_handler', data={'deployment_id': deployment_id, 'rollback_strategy': event.data.get('rollback_strategy', 'immediate'), 'target_version': event.data.get('previous_version')}))
        self.logger.info('Rollback procedure initiated for deployment %s', deployment_id)

    async def get_deployment_status(self, deployment_id: str) -> dict[str, Any] | None:
        """Get status of a deployment."""
        return self.active_deployments.get(deployment_id, {}).copy()

    async def list_active_deployments(self) -> dict[str, dict[str, Any]]:
        """List all active deployments."""
        active_deployments = {}
        for deployment_id, deployment in self.active_deployments.items():
            if deployment.get('status') in ['deploying', 'monitoring']:
                active_deployments[deployment_id] = deployment.copy()
        return active_deployments

    async def get_deployment_health_summary(self, deployment_id: str) -> dict[str, Any] | None:
        """Get health summary for a deployment."""
        if deployment_id not in self.active_deployments:
            return None
        deployment = self.active_deployments[deployment_id]
        health_checks = deployment.get('health_checks', [])
        if not health_checks:
            return {'status': 'unknown', 'checks_count': 0}
        recent_checks = health_checks[-10:]
        healthy_count = sum(1 for check in recent_checks if check['status'] == 'healthy')
        return {'status': 'healthy' if healthy_count >= len(recent_checks) * 0.8 else 'unhealthy', 'checks_count': len(health_checks), 'recent_healthy_ratio': healthy_count / len(recent_checks), 'last_check': recent_checks[-1] if recent_checks else None}

    async def get_handler_statistics(self) -> dict[str, Any]:
        """Get event handler statistics."""
        return {'events_processed': self.events_processed, 'events_failed': self.events_failed, 'success_rate': (self.events_processed - self.events_failed) / max(self.events_processed, 1), 'last_event_time': self.last_event_time, 'active_deployments': len([d for d in self.active_deployments.values() if d.get('status') in ['deploying', 'monitoring']]), 'total_deployments': len(self.active_deployments), 'deployment_history_count': len(self.deployment_history)}

    def get_supported_events(self) -> list[EventType]:
        """Get list of supported event types."""
        return [EventType.DEPLOYMENT_STARTED, EventType.DEPLOYMENT_COMPLETED, EventType.DEPLOYMENT_FAILED, EventType.MODEL_DEPLOYED, EventType.DEPLOYMENT_HEALTH_CHECK, EventType.ROLLBACK_TRIGGERED]
