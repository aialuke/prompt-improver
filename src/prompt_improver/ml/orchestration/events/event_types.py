"""
Event types and definitions for ML pipeline orchestration.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from sqlmodel import SQLModel, Field
from pydantic import BaseModel

class EventType(Enum):
    """ML Pipeline event types."""
    ORCHESTRATOR_INITIALIZED = 'orchestrator.initialized'
    ORCHESTRATOR_SHUTDOWN = 'orchestrator.shutdown'
    WORKFLOW_STARTED = 'workflow.started'
    WORKFLOW_COMPLETED = 'workflow.completed'
    WORKFLOW_FAILED = 'workflow.failed'
    WORKFLOW_STOPPED = 'workflow.stopped'
    WORKFLOW_PAUSED = 'workflow.paused'
    WORKFLOW_RESUMED = 'workflow.resumed'
    COMPONENT_REGISTERED = 'component.registered'
    COMPONENT_CONNECTED = 'component.connected'
    COMPONENT_DISCONNECTED = 'component.disconnected'
    COMPONENT_HEALTH_CHANGED = 'component.health_changed'
    COMPONENT_STARTED = 'component.started'
    COMPONENT_STOPPED = 'component.stopped'
    COMPONENT_ERROR = 'component.error'
    COMPONENT_EXECUTION_COMPLETED = 'component.execution_completed'
    COMPONENT_EXECUTION_FAILED = 'component.execution_failed'
    TRAINING_STARTED = 'training.started'
    TRAINING_PROGRESS = 'training.progress'
    TRAINING_EPOCH_COMPLETED = 'training.epoch_completed'
    TRAINING_COMPLETED = 'training.completed'
    TRAINING_FAILED = 'training.failed'
    TRAINING_DATA_LOADED = 'training.data_loaded'
    EVALUATION_STARTED = 'evaluation.started'
    EVALUATION_COMPLETED = 'evaluation.completed'
    EVALUATION_FAILED = 'evaluation.failed'
    EXPERIMENT_CREATED = 'experiment.created'
    EXPERIMENT_COMPLETED = 'experiment.completed'
    OPTIMIZATION_STARTED = 'optimization.started'
    OPTIMIZATION_ITERATION = 'optimization.iteration'
    OPTIMIZATION_COMPLETED = 'optimization.completed'
    OPTIMIZATION_FAILED = 'optimization.failed'
    HYPERPARAMETER_UPDATE = 'hyperparameter.update'
    DEPLOYMENT_STARTED = 'deployment.started'
    DEPLOYMENT_COMPLETED = 'deployment.completed'
    DEPLOYMENT_FAILED = 'deployment.failed'
    MODEL_REGISTERED = 'model.registered'
    MODEL_DEPLOYED = 'model.deployed'
    RESOURCE_ALLOCATED = 'resource.allocated'
    RESOURCE_RELEASED = 'resource.released'
    RESOURCE_WARNING = 'resource.warning'
    RESOURCE_EXHAUSTED = 'resource.exhausted'
    MEMORY_WARNING = 'memory.warning'
    MEMORY_ALLOCATED = 'memory.allocated'
    MEMORY_RELEASED = 'memory.released'
    GPU_ALLOCATED = 'gpu.allocated'
    PERFORMANCE_DEGRADED = 'performance.degraded'
    METRICS_COLLECTED = 'metrics.collected'
    ALERT_TRIGGERED = 'alert.triggered'
    HEALTH_CHECK_COMPLETED = 'health_check.completed'
    MONITORING_STARTED = 'monitoring.started'
    MONITORING_STOPPED = 'monitoring.stopped'
    HEALTH_STATUS_CHANGED = 'health.status_changed'
    ORCHESTRATOR_ALERT = 'orchestrator.alert'
    COMPONENT_HEALTH_CHECK = 'component.health_check'
    COMPONENT_UNHEALTHY = 'component.unhealthy'
    COMPONENT_RECOVERED = 'component.recovered'
    COMPONENT_INTERVENTION_REQUIRED = 'component.intervention_required'
    PIPELINE_HEALTH_CHANGED = 'pipeline.health_changed'
    PIPELINE_CRITICAL_HEALTH = 'pipeline.critical_health'
    CASCADING_FAILURE_DETECTED = 'pipeline.cascading_failure'
    HEALTH_TREND_ANALYSIS = 'health.trend_analysis'
    METRICS_COLLECTION_STARTED = 'metrics.collection_started'
    METRICS_COLLECTION_STOPPED = 'metrics.collection_stopped'
    METRIC_COLLECTED = 'metric.collected'
    METRICS_EXPORTED = 'metrics.exported'
    ALERT_CREATED = 'alert.created'
    ALERT_ACKNOWLEDGED = 'alert.acknowledged'
    ALERT_RESOLVED = 'alert.resolved'
    ALERT_ESCALATED = 'alert.escalated'
    ALERT_STORM_DETECTED = 'alert.storm_detected'
    SECURITY_ALERT = 'security.alert'
    SECURITY_SCAN_COMPLETED = 'security.scan_completed'
    ADVERSARIAL_ATTACK_DETECTED = 'security.adversarial_attack_detected'
    PRIVACY_AUDIT_COMPLETED = 'security.privacy_audit_completed'
    INPUT_VALIDATION_FAILED = 'security.input_validation_failed'
    PROMPT_INJECTION_DETECTED = 'security.prompt_injection_detected'
    DATABASE_PERFORMANCE_SNAPSHOT_TAKEN = 'database.performance_snapshot_taken'
    DATABASE_CACHE_HIT_RATIO_LOW = 'database.cache_hit_ratio_low'
    DATABASE_SLOW_QUERY_DETECTED = 'database.slow_query_detected'
    DATABASE_CONNECTION_OPTIMIZED = 'database.connection_optimized'
    DATABASE_INDEXES_CREATED = 'database.indexes_created'
    DATABASE_PERFORMANCE_DEGRADED = 'database.performance_degraded'
    DATABASE_PERFORMANCE_IMPROVED = 'database.performance_improved'
    DATABASE_RESOURCE_ANALYSIS_COMPLETED = 'database.resource_analysis_completed'

class MLEvent(BaseModel):
    """ML Pipeline event."""
    event_type: EventType = Field(description='Type of ML pipeline event')
    source: str = Field(description='Event source component')
    data: dict[str, Any] = Field(default_factory=dict, description='Event data payload')
    timestamp: datetime | None = Field(default=None, description='Event timestamp')
    event_id: str | None = Field(default=None, description='Unique event identifier')
    correlation_id: str | None = Field(default=None, description='Correlation ID for event tracing')

    def model_post_init(self, __context: Any) -> None:
        """Set default values after initialization."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.event_id is None:
            ts_str = self.timestamp.strftime('%Y%m%d_%H%M%S_%f')
            self.event_id = f'{self.source}_{ts_str}'

    def model_dump(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {'event_type': self.event_type.value, 'source': self.source, 'data': self.data, 'timestamp': self.timestamp.isoformat() if self.timestamp else None, 'event_id': self.event_id, 'correlation_id': self.correlation_id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'MLEvent':
        """Create event from dictionary."""
        timestamp = None
        if data.get('timestamp'):
            timestamp = datetime.fromisoformat(data['timestamp'])
        return cls(event_type=EventType(data['event_type']), source=data['source'], data=data['data'], timestamp=timestamp, event_id=data.get('event_id'), correlation_id=data.get('correlation_id'))
