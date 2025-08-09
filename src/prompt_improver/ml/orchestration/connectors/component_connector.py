"""
Base Component Connector for ML Pipeline Orchestration.

Provides the base interface for connecting ML components to the central orchestrator.
"""
from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Protocol
from ..events.event_types import EventType, MLEvent
from ..shared.component_types import ComponentTier

class ComponentStatus(Enum):
    """Component status states."""
    UNKNOWN = 'unknown'
    INITIALIZING = 'initializing'
    ready = 'ready'
    RUNNING = 'running'
    busy = 'busy'
    ERROR = 'error'
    STOPPED = 'stopped'

@dataclass
class ComponentCapability:
    """Represents a capability that a component provides."""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class ComponentMetadata:
    """Metadata about a component."""
    name: str
    tier: ComponentTier
    version: str
    capabilities: List[ComponentCapability]
    health_check_endpoint: Optional[str] = None
    api_endpoints: List[str] = None
    resource_requirements: Dict[str, Any] = None

    def __post_init__(self):
        if self.api_endpoints is None:
            self.api_endpoints = []
        if self.resource_requirements is None:
            self.resource_requirements = {}

class ComponentInterface(Protocol):
    """Protocol defining the interface that components must implement."""

    async def initialize(self) -> None:
        """Initialize the component."""
        ...

    async def health_check(self) -> Dict[str, Any]:
        """Check component health."""
        ...

    async def get_status(self) -> ComponentStatus:
        """Get current component status."""
        ...

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute component functionality."""
        ...

    async def stop(self) -> None:
        """Stop the component."""
        ...

class ComponentConnector(ABC):
    """
    Abstract base class for component connectors.
    
    Provides common functionality for connecting ML components to the orchestrator.
    """

    def __init__(self, metadata: ComponentMetadata, event_bus=None):
        """Initialize the component connector."""
        self.metadata = metadata
        self.event_bus = event_bus
        self.logger = logging.getLogger(f'{__name__}.{metadata.name}')
        self.status = ComponentStatus.UNKNOWN
        self.connected_at: Optional[datetime] = None
        self.last_health_check: Optional[datetime] = None
        self.health_status: Dict[str, Any] = {}
        self.component_instance: Optional[ComponentInterface] = None
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []

    async def connect(self) -> None:
        """Connect the component to the orchestrator."""
        self.logger.info('Connecting component %s', self.metadata.name)
        try:
            self.status = ComponentStatus.INITIALIZING
            await self._initialize_component()
            await self._perform_health_check()
            if self.event_bus:
                await self.event_bus.emit(MLEvent(event_type=EventType.COMPONENT_CONNECTED, source=f'connector_{self.metadata.name}', data={'component_name': self.metadata.name, 'tier': self.metadata.tier.value, 'capabilities': [cap.name for cap in self.metadata.capabilities]}))
            self.status = ComponentStatus.ready
            self.connected_at = datetime.now(timezone.utc)
            self.logger.info('Component %s connected successfully', self.metadata.name)
        except Exception as e:
            self.status = ComponentStatus.ERROR
            self.logger.error('Failed to connect component {self.metadata.name}: %s', e)
            raise

    async def disconnect(self) -> None:
        """Disconnect the component from the orchestrator."""
        self.logger.info('Disconnecting component %s', self.metadata.name)
        try:
            for execution_id in list(self.active_executions.keys()):
                await self.stop_execution(execution_id)
            if self.component_instance:
                await self.component_instance.stop()
            if self.event_bus:
                await self.event_bus.emit(MLEvent(event_type=EventType.COMPONENT_DISCONNECTED, source=f'connector_{self.metadata.name}', data={'component_name': self.metadata.name, 'uptime': (datetime.now(timezone.utc) - self.connected_at).total_seconds() if self.connected_at else 0}))
            self.status = ComponentStatus.STOPPED
            self.logger.info('Component %s disconnected', self.metadata.name)
        except Exception as e:
            self.logger.error('Error disconnecting component {self.metadata.name}: %s', e)
            self.status = ComponentStatus.ERROR
            raise

    async def execute_capability(self, capability_name: str, execution_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific capability of the component."""
        self.logger.info('Executing capability {capability_name} for component %s', self.metadata.name)
        capability = self._get_capability(capability_name)
        if not capability:
            raise ValueError(f'Capability {capability_name} not found in component {self.metadata.name}')
        if self.status != ComponentStatus.ready:
            raise RuntimeError(f'Component {self.metadata.name} not ready for execution (status: {self.status})')
        self.active_executions[execution_id] = {'capability': capability_name, 'started_at': datetime.now(timezone.utc), 'parameters': parameters, 'status': 'running'}
        try:
            self.status = ComponentStatus.RUNNING
            result = await self._execute_component(capability_name, parameters)
            self.active_executions[execution_id]['status'] = 'completed'
            self.active_executions[execution_id]['completed_at'] = datetime.now(timezone.utc)
            self.active_executions[execution_id]['result'] = result
            self.execution_history.append(self.active_executions[execution_id].copy())
            del self.active_executions[execution_id]
            self.status = ComponentStatus.ready
            if self.event_bus:
                await self.event_bus.emit(MLEvent(event_type=EventType.COMPONENT_EXECUTION_COMPLETED, source=f'connector_{self.metadata.name}', data={'component_name': self.metadata.name, 'execution_id': execution_id, 'capability': capability_name, 'duration': (datetime.now(timezone.utc) - self.active_executions.get(execution_id, {}).get('started_at', datetime.now(timezone.utc))).total_seconds()}))
            self.logger.info('Capability {capability_name} executed successfully for component %s', self.metadata.name)
            return result
        except Exception as e:
            self.active_executions[execution_id]['status'] = 'failed'
            self.active_executions[execution_id]['error'] = str(e)
            self.active_executions[execution_id]['completed_at'] = datetime.now(timezone.utc)
            self.execution_history.append(self.active_executions[execution_id].copy())
            del self.active_executions[execution_id]
            self.status = ComponentStatus.ERROR
            if self.event_bus:
                await self.event_bus.emit(MLEvent(event_type=EventType.COMPONENT_EXECUTION_FAILED, source=f'connector_{self.metadata.name}', data={'component_name': self.metadata.name, 'execution_id': execution_id, 'capability': capability_name, 'error_message': str(e)}))
            self.logger.error('Capability {capability_name} execution failed for component {self.metadata.name}: %s', e)
            raise

    async def stop_execution(self, execution_id: str) -> None:
        """Stop a specific execution."""
        if execution_id not in self.active_executions:
            raise ValueError(f'Execution {execution_id} not found')
        self.active_executions[execution_id]['status'] = 'stopped'
        self.active_executions[execution_id]['completed_at'] = datetime.now(timezone.utc)
        self.execution_history.append(self.active_executions[execution_id].copy())
        del self.active_executions[execution_id]
        if not self.active_executions:
            self.status = ComponentStatus.ready
        self.logger.info('Execution {execution_id} stopped for component %s', self.metadata.name)

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the component."""
        await self._perform_health_check()
        return self.health_status.copy()

    def get_capabilities(self) -> List[ComponentCapability]:
        """Get list of component capabilities."""
        return self.metadata.capabilities.copy()

    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific execution."""
        return self.active_executions.get(execution_id, {}).copy() if execution_id in self.active_executions else None

    def list_active_executions(self) -> List[str]:
        """List all active execution IDs."""
        return list(self.active_executions.keys())

    def get_execution_history(self, limit: int=10) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.execution_history[-limit:] if limit > 0 else self.execution_history.copy()

    @abstractmethod
    async def _initialize_component(self) -> None:
        """Initialize the specific component. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def _execute_component(self, capability_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the specific component. Must be implemented by subclasses."""
        pass

    async def _perform_health_check(self) -> None:
        """Perform health check on the component."""
        try:
            if self.component_instance:
                health_result = await self.component_instance.health_check()
                self.health_status = {'status': 'healthy', 'timestamp': datetime.now(timezone.utc), 'details': health_result}
            else:
                self.health_status = {'status': 'unhealthy', 'timestamp': datetime.now(timezone.utc), 'details': {'error': 'Component instance not available'}}
            self.last_health_check = datetime.now(timezone.utc)
        except Exception as e:
            self.health_status = {'status': 'unhealthy', 'timestamp': datetime.now(timezone.utc), 'details': {'error': str(e)}}
            self.logger.warning('Health check failed for component {self.metadata.name}: %s', e)

    def _get_capability(self, capability_name: str) -> Optional[ComponentCapability]:
        """Get a specific capability by name."""
        for capability in self.metadata.capabilities:
            if capability.name == capability_name:
                return capability
        return None

    def list_available_components(self) -> List[str]:
        """
        List available components for this connector.

        This is an instance method that provides a consistent interface.
        Subclasses should override this to provide their specific components.

        Returns:
            List of component names available through this connector
        """
        return [self.metadata.name]

    def get_component_capabilities(self) -> List[ComponentCapability]:
        """
        Get the capabilities provided by this component.

        Returns:
            List of capabilities this component provides
        """
        return self.metadata.capabilities

    def get_component_tier(self) -> ComponentTier:
        """
        Get the tier classification of this component.

        Returns:
            The tier this component belongs to
        """
        return self.metadata.tier
