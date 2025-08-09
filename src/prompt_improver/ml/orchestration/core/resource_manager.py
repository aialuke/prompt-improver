"""
Resource Manager for ML Pipeline orchestration.

Manages resource allocation, monitoring, and optimization across all ML components.
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid
import psutil
from ....performance.monitoring.health.background_manager import TaskPriority, get_background_task_manager
from ..config.orchestrator_config import OrchestratorConfig
from ..events.event_types import EventType, MLEvent
try:
    from ..k8s.resource_integration import kubernetes_resource_manager
    KUBERNETES_INTEGRATION_AVAILABLE = True
except ImportError:
    KUBERNETES_INTEGRATION_AVAILABLE = False
    kubernetes_resource_manager = None

class CircuitBreakerState(Enum):
    """Circuit breaker states for resource management."""
    CLOSED = 'closed'
    OPEN = 'open'
    HALF_OPEN = 'half_open'

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3

class ResourceCircuitBreaker:
    """Circuit breaker for resource allocation operations."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.logger = logging.getLogger(__name__)

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                self.logger.info('Circuit breaker transitioning to HALF_OPEN')
            else:
                raise ResourceExhaustionError('Circuit breaker is OPEN - resource allocation blocked')
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except ResourceExhaustionError as e:
            self._on_failure()
            raise
        except Exception as e:
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.recovery_timeout

    def _on_success(self):
        """Handle successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.logger.info('Circuit breaker reset to CLOSED')
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning('Circuit breaker opened after %s failures', self.failure_count)

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {'state': self.state.value, 'failure_count': self.failure_count, 'success_count': self.success_count, 'last_failure_time': self.last_failure_time}

class ResourceType(Enum):
    """Types of resources managed by the orchestrator."""
    CPU = 'cpu'
    memory = 'memory'
    GPU = 'gpu'
    GPU_MEMORY = 'gpu_memory'
    disk = 'disk'
    DATABASE_CONNECTIONS = 'database_connections'
    CACHE_CONNECTIONS = 'cache_connections'
    GPU_MIG_SLICE = 'gpu_mig_slice'
    GPU_TIME_SLICE = 'gpu_time_slice'

@dataclass
class ResourceAllocation:
    """Represents an allocated resource."""
    allocation_id: str
    resource_type: ResourceType
    amount: float
    component_name: str
    workflow_id: Optional[str] = None
    allocated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class ResourceUsage:
    """Current resource usage statistics."""
    resource_type: ResourceType
    total_available: float
    currently_allocated: float
    usage_percentage: float
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class ResourceManager:
    """
    Resource Manager for ML Pipeline orchestration.
    
    Manages:
    - CPU and memory allocation
    - GPU resource coordination
    - Database connection pooling
    - Cache connection management
    - Resource monitoring and optimization
    """

    def __init__(self, config: OrchestratorConfig):
        """Initialize the resource manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.resource_limits: Dict[ResourceType, float] = {}
        self.current_usage: Dict[ResourceType, ResourceUsage] = {}
        self.circuit_breakers: Dict[ResourceType, ResourceCircuitBreaker] = {}
        self._initialize_circuit_breakers()
        self.k8s_manager: Optional[kubernetes_resource_manager] = None
        if KUBERNETES_INTEGRATION_AVAILABLE:
            try:
                self.k8s_manager = kubernetes_resource_manager()
                self.logger.info('Kubernetes resource integration enabled')
            except Exception as e:
                self.logger.warning('Failed to initialize Kubernetes integration: %s', e)
        else:
            self.logger.info('Kubernetes integration not available - running in standalone mode')
        self.event_bus = None
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        self._allocation_counter = 0
        self._is_initialized = False

    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for different resource types."""
        critical_resources = [ResourceType.CPU, ResourceType.memory, ResourceType.GPU, ResourceType.GPU_MEMORY, ResourceType.GPU_MIG_SLICE, ResourceType.GPU_TIME_SLICE]
        for resource_type in critical_resources:
            if resource_type in [ResourceType.GPU, ResourceType.GPU_MEMORY, ResourceType.GPU_MIG_SLICE, ResourceType.GPU_TIME_SLICE]:
                config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0, success_threshold=2)
            else:
                config = CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60.0, success_threshold=3)
            self.circuit_breakers[resource_type] = ResourceCircuitBreaker(config)
            self.logger.info('Initialized circuit breaker for %s', resource_type.value)

    async def initialize(self) -> None:
        """Initialize the resource manager."""
        self.logger.info('Initializing resource manager')
        self._initialize_resource_limits()
        await self._initialize_usage_tracking()
        await self._start_monitoring()
        self._is_initialized = True
        self.logger.info('Resource manager initialized successfully')

    async def shutdown(self) -> None:
        """Shutdown the resource manager."""
        self.logger.info('Shutting down resource manager')
        await self._stop_monitoring()
        await self._release_all_allocations()
        self.logger.info('Resource manager shutdown complete')

    def set_event_bus(self, event_bus) -> None:
        """Set the event bus reference."""
        self.event_bus = event_bus

    async def allocate_resource(self, resource_type: ResourceType, amount: float, component_name: str, workflow_id: Optional[str]=None, timeout: Optional[int]=None) -> str:
        """
        Allocate resources for a component with circuit breaker protection.

        Args:
            resource_type: Type of resource to allocate
            amount: Amount of resource needed
            component_name: Name of the requesting component
            workflow_id: Associated workflow ID (optional)
            timeout: Allocation timeout in seconds

        Returns:
            Allocation ID for tracking and release
        """
        if resource_type in self.circuit_breakers:
            return await self.circuit_breakers[resource_type].call(self._allocate_resource_internal, resource_type, amount, component_name, workflow_id, timeout)
        else:
            return await self._allocate_resource_internal(resource_type, amount, component_name, workflow_id, timeout)

    async def _allocate_resource_internal(self, resource_type: ResourceType, amount: float, component_name: str, workflow_id: Optional[str]=None, timeout: Optional[int]=None) -> str:
        """Internal resource allocation method."""
        if not await self._is_resource_available(resource_type, amount):
            raise ResourceExhaustionError(f'Insufficient {resource_type.value} available: requested {amount}, available {await self._get_available_amount(resource_type)}')
        self._allocation_counter += 1
        allocation_id = f'alloc_{self._allocation_counter}_{resource_type.value}'
        allocation = ResourceAllocation(allocation_id=allocation_id, resource_type=resource_type, amount=amount, component_name=component_name, workflow_id=workflow_id)
        if timeout:
            allocation.expires_at = datetime.now(timezone.utc).add(seconds=timeout)
        self.allocations[allocation_id] = allocation
        await self._update_usage_tracking()
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.RESOURCE_ALLOCATED, source='resource_manager', data={'allocation_id': allocation_id, 'resource_type': resource_type.value, 'amount': amount, 'component_name': component_name, 'workflow_id': workflow_id}))
        self.logger.info('Allocated %s %s to %s (allocation: %s)', amount, resource_type.value, component_name, allocation_id)
        return allocation_id

    async def release_resource(self, allocation_id: str) -> bool:
        """
        Release an allocated resource.
        
        Args:
            allocation_id: ID of the allocation to release
            
        Returns:
            True if resource was found and released
        """
        if allocation_id not in self.allocations:
            return False
        allocation = self.allocations[allocation_id]
        del self.allocations[allocation_id]
        await self._update_usage_tracking()
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.RESOURCE_RELEASED, source='resource_manager', data={'allocation_id': allocation_id, 'resource_type': allocation.resource_type.value, 'amount': allocation.amount, 'component_name': allocation.component_name}))
        self.logger.info('Released %s %s from %s (allocation: %s)', allocation.amount, allocation.resource_type.value, allocation.component_name, allocation_id)
        return True

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get current resource usage statistics."""
        await self._update_usage_tracking()
        stats = {}
        for resource_type, usage in self.current_usage.items():
            stats[resource_type.value] = {'total_available': usage.total_available, 'currently_allocated': usage.currently_allocated, 'usage_percentage': usage.usage_percentage, 'last_updated': usage.last_updated.isoformat()}
        return stats

    async def get_allocations(self, component_name: Optional[str]=None, workflow_id: Optional[str]=None) -> List[ResourceAllocation]:
        """
        Get current resource allocations.
        
        Args:
            component_name: Filter by component name (optional)
            workflow_id: Filter by workflow ID (optional)
            
        Returns:
            List of matching allocations
        """
        allocations = list(self.allocations.values())
        if component_name:
            allocations = [a for a in allocations if a.component_name == component_name]
        if workflow_id:
            allocations = [a for a in allocations if a.workflow_id == workflow_id]
        return allocations

    async def handle_resource_exhaustion(self, resource_type: str) -> None:
        """
        Handle resource exhaustion scenarios.
        
        Args:
            resource_type: Type of resource that is exhausted
        """
        self.logger.warning('Handling resource exhaustion for %s', resource_type)
        await self._cleanup_expired_allocations()
        await self._optimize_resource_usage(ResourceType(resource_type))
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.RESOURCE_EXHAUSTED, source='resource_manager', data={'resource_type': resource_type}))

    def _initialize_resource_limits(self) -> None:
        """Initialize resource limits from configuration."""
        self.resource_limits = {ResourceType.CPU: float(self.config.cpu_limit_cores), ResourceType.memory: self.config.memory_limit_gb * 1024 * 1024 * 1024, ResourceType.GPU: 1.0, ResourceType.GPU_MEMORY: 24.0 * 1024 * 1024 * 1024, ResourceType.GPU_MIG_SLICE: 7.0, ResourceType.GPU_TIME_SLICE: 4.0, ResourceType.disk: 100.0 * 1024 * 1024 * 1024, ResourceType.DATABASE_CONNECTIONS: float(self.config.db_connection_pool_size), ResourceType.CACHE_CONNECTIONS: float(self.config.redis_connection_pool_size)}
        self.logger.info('Initialized resource limits: %s', self.resource_limits)

    async def _initialize_usage_tracking(self) -> None:
        """Initialize resource usage tracking."""
        for resource_type in ResourceType:
            self.current_usage[resource_type] = ResourceUsage(resource_type=resource_type, total_available=self.resource_limits[resource_type], currently_allocated=0.0, usage_percentage=0.0)

    async def _start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.is_monitoring:
            return
        self.is_monitoring = True
        task_manager = get_background_task_manager()
        self.monitoring_task_id = await task_manager.submit_enhanced_task(task_id=f'ml_resource_monitor_{str(uuid.uuid4())[:8]}', coroutine=self._monitoring_loop(), priority=TaskPriority.HIGH, tags={'service': 'ml', 'type': 'monitoring', 'component': 'resource_monitor', 'module': 'resource_manager'})
        self.logger.info('Started resource monitoring')

    async def _stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        if not self.is_monitoring:
            return
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info('Stopped resource monitoring')

    async def _monitoring_loop(self) -> None:
        """Resource monitoring loop."""
        while self.is_monitoring:
            try:
                await self._update_usage_tracking()
                await self._check_resource_thresholds()
                await self._cleanup_expired_allocations()
                await asyncio.sleep(self.config.metrics_collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error('Error in resource monitoring loop: %s', e)
                await asyncio.sleep(5)

    async def _update_usage_tracking(self) -> None:
        """Update current resource usage statistics."""
        for resource_type in ResourceType:
            allocated = sum((alloc.amount for alloc in self.allocations.values() if alloc.resource_type == resource_type))
            total_available = await self._get_total_available(resource_type)
            usage_percentage = allocated / total_available * 100 if total_available > 0 else 0
            self.current_usage[resource_type] = ResourceUsage(resource_type=resource_type, total_available=total_available, currently_allocated=allocated, usage_percentage=usage_percentage)

    async def _get_total_available(self, resource_type: ResourceType) -> float:
        """Get total available amount for a resource type."""
        if resource_type == ResourceType.CPU:
            return float(psutil.cpu_count())
        elif resource_type == ResourceType.memory:
            return float(psutil.virtual_memory().total)
        else:
            return self.resource_limits[resource_type]

    async def _is_resource_available(self, resource_type: ResourceType, amount: float) -> bool:
        """Check if a resource amount is available for allocation."""
        available = await self._get_available_amount(resource_type)
        return available >= amount

    async def _get_available_amount(self, resource_type: ResourceType) -> float:
        """Get currently available amount for a resource type."""
        usage = self.current_usage[resource_type]
        return usage.total_available - usage.currently_allocated

    async def _check_resource_thresholds(self) -> None:
        """Check resource usage against alert thresholds."""
        for resource_type, usage in self.current_usage.items():
            if resource_type == ResourceType.CPU and usage.usage_percentage > self.config.alert_threshold_cpu * 100:
                await self._emit_threshold_alert(resource_type, usage.usage_percentage)
            elif resource_type == ResourceType.memory and usage.usage_percentage > self.config.alert_threshold_memory * 100:
                await self._emit_threshold_alert(resource_type, usage.usage_percentage)

    async def _emit_threshold_alert(self, resource_type: ResourceType, usage_percentage: float) -> None:
        """Emit an alert when resource usage exceeds thresholds."""
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.ALERT_TRIGGERED, source='resource_manager', data={'alert_type': 'resource_threshold_exceeded', 'resource_type': resource_type.value, 'usage_percentage': usage_percentage}))

    async def _cleanup_expired_allocations(self) -> None:
        """Clean up expired resource allocations."""
        now = datetime.now(timezone.utc)
        expired_allocations = []
        for allocation_id, allocation in self.allocations.items():
            if allocation.expires_at and allocation.expires_at < now:
                expired_allocations.append(allocation_id)
        for allocation_id in expired_allocations:
            await self.release_resource(allocation_id)
            self.logger.info('Released expired allocation: %s', allocation_id)

    async def _optimize_resource_usage(self, resource_type: ResourceType) -> None:
        """Implement resource optimization strategies."""
        self.logger.info('Optimizing %s resource usage', resource_type.value)

    async def allocate_gpu_resources(self, component_name: str, gpu_type: str='shared', gpu_memory_gb: float=None, mig_profile: str=None, workflow_id: str=None) -> Optional[ResourceAllocation]:
        """
        Allocate GPU resources with 2025 best practices.
        
        Args:
            component_name: Name of component requesting GPU
            gpu_type: Type of GPU allocation ('shared', 'dedicated', 'mig_slice', 'time_slice')
            gpu_memory_gb: GPU memory requirement in GB
            mig_profile: MIG profile for multi-instance GPU (e.g., '1g.5gb', '2g.10gb')
            workflow_id: Associated workflow ID
            
        Returns:
            Resource allocation if successful, None if failed
        """
        try:
            if gpu_type == 'shared':
                allocation = await self.allocate_resource(resource_type=ResourceType.GPU_TIME_SLICE, amount=0.25, component_name=component_name, workflow_id=workflow_id)
                if gpu_memory_gb and allocation:
                    memory_allocation = await self.allocate_resource(resource_type=ResourceType.GPU_MEMORY, amount=gpu_memory_gb * 1024 * 1024 * 1024, component_name=component_name, workflow_id=workflow_id)
                    allocation.metadata = {'gpu_memory_allocation_id': memory_allocation.allocation_id if memory_allocation else None, 'gpu_memory_gb': gpu_memory_gb}
                return allocation
            elif gpu_type == 'mig_slice' and mig_profile:
                mig_allocation = await self.allocate_resource(resource_type=ResourceType.GPU_MIG_SLICE, amount=1.0, component_name=component_name, workflow_id=workflow_id)
                if mig_allocation:
                    mig_allocation.metadata = {'mig_profile': mig_profile, 'gpu_type': 'mig_slice'}
                return mig_allocation
            elif gpu_type == 'dedicated':
                return await self.allocate_resource(resource_type=ResourceType.GPU, amount=1.0, component_name=component_name, workflow_id=workflow_id)
            else:
                self.logger.error('Unsupported GPU type: %s', gpu_type)
                return None
        except Exception as e:
            self.logger.error('Error allocating GPU resources: %s', e)
            return None

    async def get_gpu_utilization_metrics(self) -> Dict[str, Any]:
        """Get GPU utilization metrics (following 2025 monitoring patterns)."""
        try:
            gpu_metrics = {'gpu_utilization_percentage': 0.0, 'gpu_memory_utilization_percentage': 0.0, 'gpu_temperature_celsius': 0.0, 'gpu_power_usage_watts': 0.0, 'active_allocations': {'dedicated': 0, 'time_sliced': 0, 'mig_slices': 0}, 'last_updated': datetime.now(timezone.utc).isoformat()}
            for allocation in self.allocations.values():
                if allocation.resource_type == ResourceType.GPU:
                    gpu_metrics['active_allocations']['dedicated'] += 1
                elif allocation.resource_type == ResourceType.GPU_TIME_SLICE:
                    gpu_metrics['active_allocations']['time_sliced'] += 1
                elif allocation.resource_type == ResourceType.GPU_MIG_SLICE:
                    gpu_metrics['active_allocations']['mig_slices'] += 1
            total_allocations = sum(gpu_metrics['active_allocations'].values())
            if total_allocations > 0:
                gpu_metrics['gpu_utilization_percentage'] = min(total_allocations * 25.0, 100.0)
                gpu_metrics['gpu_memory_utilization_percentage'] = min(total_allocations * 30.0, 100.0)
            return gpu_metrics
        except Exception as e:
            self.logger.error('Error getting GPU metrics: %s', e)
            return {}

    async def implement_gpu_autoscaling(self, utilization_threshold: float=0.8) -> Dict[str, Any]:
        """
        Implement GPU autoscaling based on utilization (2025 KEDA-style patterns).
        
        Args:
            utilization_threshold: GPU utilization threshold for scaling decisions
            
        Returns:
            Scaling decision and metrics
        """
        try:
            gpu_metrics = await self.get_gpu_utilization_metrics()
            current_utilization = gpu_metrics['gpu_utilization_percentage'] / 100.0
            scaling_decision = {'should_scale_up': False, 'should_scale_down': False, 'current_utilization': current_utilization, 'threshold': utilization_threshold, 'recommended_action': 'maintain', 'timestamp': datetime.now(timezone.utc).isoformat()}
            if current_utilization > utilization_threshold:
                scaling_decision['should_scale_up'] = True
                scaling_decision['recommended_action'] = 'scale_up'
                if self.event_bus:
                    await self.event_bus.emit(MLEvent(event_type=EventType.AUTOSCALING_TRIGGERED, source='resource_manager', data={'resource_type': 'gpu', 'action': 'scale_up', 'current_utilization': current_utilization, 'threshold': utilization_threshold}))
            elif current_utilization < utilization_threshold * 0.5:
                scaling_decision['should_scale_down'] = True
                scaling_decision['recommended_action'] = 'scale_down'
                if self.event_bus:
                    await self.event_bus.emit(MLEvent(event_type=EventType.AUTOSCALING_TRIGGERED, source='resource_manager', data={'resource_type': 'gpu', 'action': 'scale_down', 'current_utilization': current_utilization, 'threshold': utilization_threshold}))
            return scaling_decision
        except Exception as e:
            self.logger.error('Error in GPU autoscaling: %s', e)
            return {'error': str(e)}
        if resource_type == ResourceType.memory:
            import gc
            gc.collect()
            self.logger.info('Triggered garbage collection for memory optimization')

    async def _release_all_allocations(self) -> None:
        """Release all current allocations during shutdown."""
        allocation_ids = list(self.allocations.keys())
        for allocation_id in allocation_ids:
            await self.release_resource(allocation_id)

    async def get_cluster_resource_info(self) -> Dict[str, Any]:
        """Get comprehensive cluster resource information."""
        if not self.k8s_manager or not self.k8s_manager.is_available():
            return {'kubernetes_available': False, 'message': 'Kubernetes integration not available'}
        try:
            cluster_summary = await self.k8s_manager.get_cluster_resource_summary()
            return {'kubernetes_available': True, 'cluster_info': cluster_summary}
        except Exception as e:
            self.logger.error('Error getting cluster resource info: %s', e)
            return {'kubernetes_available': True, 'error': str(e)}

    async def create_hpa_for_component(self, component_name: str, min_replicas: int=1, max_replicas: int=10, target_cpu_percent: int=70) -> bool:
        """Create Horizontal Pod Autoscaler for a component."""
        if not self.k8s_manager or not self.k8s_manager.is_available():
            self.logger.warning('Cannot create HPA - Kubernetes integration not available')
            return False
        try:
            success = await self.k8s_manager.create_hpa(deployment_name=component_name, min_replicas=min_replicas, max_replicas=max_replicas, target_cpu_percent=target_cpu_percent)
            if success:
                self.logger.info('Created HPA for component %s', component_name)
            else:
                self.logger.error('Failed to create HPA for component %s', component_name)
            return success
        except Exception as e:
            self.logger.error('Error creating HPA for {component_name}: %s', e)
            return False

    async def get_namespace_resource_utilization(self, namespace: str=None) -> Dict[str, Any]:
        """Get resource utilization for a specific namespace."""
        if not self.k8s_manager or not self.k8s_manager.is_available():
            return {'error': 'Kubernetes integration not available'}
        try:
            quotas = await self.k8s_manager.get_namespace_resource_quotas(namespace)
            pods = await self.k8s_manager.get_pod_resources(namespace)
            return {'namespace': namespace or self.k8s_manager.namespace, 'resource_quotas': len(quotas), 'active_pods': len(pods), 'quotas': [{'name': quota.quota_name, 'cpu_usage': f'{quota.cpu_used}/{quota.cpu_hard}', 'memory_usage': f'{quota.memory_used}/{quota.memory_hard}', 'gpu_usage': f'{quota.gpu_used}/{quota.gpu_hard}'} for quota in quotas], 'pod_summary': {'total_pods': len(pods), 'total_cpu_requests': sum((self.k8s_manager._parse_cpu_resource(pod.cpu_request) for pod in pods)), 'total_memory_requests': sum((self.k8s_manager._parse_memory_resource(pod.memory_request) for pod in pods)), 'total_gpu_requests': sum((int(pod.gpu_request) for pod in pods))}}
        except Exception as e:
            self.logger.error('Error getting namespace resource utilization: %s', e)
            return {'error': str(e)}

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        return {resource_type.value: breaker.get_state() for resource_type, breaker in self.circuit_breakers.items()}

class ResourceExhaustionError(Exception):
    """Raised when requested resources are not available."""
    pass
