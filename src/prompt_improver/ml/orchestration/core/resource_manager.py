"""
Resource Manager for ML Pipeline orchestration.

Manages resource allocation, monitoring, and optimization across all ML components.
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import logging
import time
from typing import Any, Dict, List, Optional, Protocol, cast
from collections.abc import Awaitable, Callable
import uuid
import psutil
from ....performance.monitoring.health.background_manager import TaskPriority, get_background_task_manager, EnhancedBackgroundTaskManager
from ..config.orchestrator_config import OrchestratorConfig
from ..events.event_types import EventType, MLEvent
k8s_integration_available = False
try:
    from ..k8s.resource_integration import KubernetesResourceManager
    k8s_integration_available = True
except ImportError:
    KubernetesResourceManager = None

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

    async def call(self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
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
        except ResourceExhaustionError:
            self._on_failure()
            raise
        except Exception:
            # Treat other exceptions as failures as well to trip the breaker appropriately
            self._on_failure()
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

    def get_state(self) -> dict[str, Any]:
        """Get current circuit breaker state."""
        return {'state': self.state.value, 'failure_count': self.failure_count, 'success_count': self.success_count, 'last_failure_time': self.last_failure_time}

class ResourceType(Enum):
    """Types of resources managed by the orchestrator (canonical, 2025)."""
    CPU = 'cpu'
    MEMORY = 'memory'
    GPU = 'gpu'
    GPU_MEMORY = 'gpu_memory'
    DISK = 'disk'
    DATABASE_CONNECTIONS = 'database_connections'
    CACHE_CONNECTIONS = 'cache_connections'
    GPU_MIG_SLICE = 'gpu_mig_slice'
    GPU_TIME_SLICE = 'gpu_time_slice'

class EventBusLike(Protocol):
    async def emit(self, event: "MLEvent", priority: int = 5) -> None: ...


@dataclass
class ResourceAllocation:
    """Represents an allocated resource."""
    allocation_id: str
    workflow_id: str
    allocated_resources: dict[ResourceType, float]
    allocation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {}))

    @property
    def age(self) -> timedelta:
        """Calculate age of this allocation."""
        return datetime.now(timezone.utc) - self.allocation_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'allocation_id': self.allocation_id,
            'workflow_id': self.workflow_id,
            'allocated_resources': {rt.value: amount for rt, amount in self.allocated_resources.items()},
            'allocation_time': self.allocation_time.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ResourceAllocation':
        """Create instance from dictionary."""
        if 'allocated_resources' in data and isinstance(data['allocated_resources'], dict):
            # Convert string keys back to ResourceType enums
            allocated_resources: dict[ResourceType, float] = {}
            ar: dict[str, Any] = cast(dict[str, Any], data['allocated_resources'])
            for rt_key, amount in ar.items():
                try:
                    key_str: str = str(rt_key)
                    allocated_resources[ResourceType(key_str)] = float(amount)
                except Exception:
                    continue
            data['allocated_resources'] = allocated_resources
        if isinstance(data.get('allocation_time'), str):
            data['allocation_time'] = datetime.fromisoformat(data['allocation_time'])
        if data.get('expires_at') and isinstance(data['expires_at'], str):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)

@dataclass
class ResourceUsage:
    """Current resource usage statistics."""
    total_cpu_allocated: float
    total_memory_allocated: int
    total_gpu_allocated: int = 0
    total_gpu_memory_allocated: int = 0
    active_allocations: int = 0
    total_allocations: int = 0
    allocation_history_size: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_cpu_allocated': self.total_cpu_allocated,
            'total_memory_allocated': self.total_memory_allocated,
            'total_gpu_allocated': self.total_gpu_allocated,
            'total_gpu_memory_allocated': self.total_gpu_memory_allocated,
            'active_allocations': self.active_allocations,
            'total_allocations': self.total_allocations,
            'allocation_history_size': self.allocation_history_size,
            'last_updated': self.last_updated.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ResourceUsage':
        """Create instance from dictionary."""
        if isinstance(data.get('last_updated'), str):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)

class ResourceService:
    """
    Resource Service for ML Pipeline orchestration.

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
        self.allocations: dict[str, ResourceAllocation] = {}
        self.allocation_history: list[ResourceAllocation] = []
        self.resource_limits: dict[ResourceType, float] = {}
        self.circuit_breakers: dict[ResourceType, ResourceCircuitBreaker] = {}
        self._initialize_circuit_breakers()
        self.k8s_manager: Any = None
        if k8s_integration_available and KubernetesResourceManager is not None:
            try:
                self.k8s_manager = KubernetesResourceManager()
                self.logger.info('Kubernetes resource integration enabled')
            except Exception as e:
                self.logger.warning('Failed to initialize Kubernetes integration: %s', e)
        else:
            self.logger.info('Kubernetes integration not available - running in standalone mode')
        self.event_bus: EventBusLike | None = None
        self._task_manager: EnhancedBackgroundTaskManager | None = None
        self.monitoring_task_id: str | None = None
        self.is_monitoring = False
        self._allocation_counter = 0
        self._is_initialized = False

    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for different resource types."""
        critical_resources = [ResourceType.CPU, ResourceType.MEMORY, ResourceType.GPU, ResourceType.GPU_MEMORY, ResourceType.GPU_MIG_SLICE, ResourceType.GPU_TIME_SLICE]
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

    def set_event_bus(self, event_bus: EventBusLike) -> None:
        """Set the event bus reference."""
        self.event_bus = event_bus

    async def allocate_resource(self, resource_type: ResourceType, amount: float, component_name: str, workflow_id: str | None=None, timeout: int | None=None) -> ResourceAllocation:
        """
        Allocate a single resource for a component with circuit breaker protection.

        Args:
            resource_type: Type of resource to allocate
            amount: Amount of resource needed
            component_name: Name of the requesting component
            workflow_id: Associated workflow ID (optional)
            timeout: Allocation timeout in seconds

        Returns:
            ResourceAllocation object for tracking and release
        """
        if resource_type in self.circuit_breakers:
            return await self.circuit_breakers[resource_type].call(self._allocate_resource_internal, resource_type, amount, component_name, workflow_id, timeout)
        else:
            return await self._allocate_resource_internal(resource_type, amount, component_name, workflow_id, timeout)

    async def _allocate_resource_internal(self, resource_type: ResourceType, amount: float, component_name: str, workflow_id: str | None=None, timeout: int | None=None) -> ResourceAllocation:
        """Internal resource allocation method."""
        if not await self._is_resource_available(resource_type, amount):
            raise ResourceExhaustionError(f'Insufficient {resource_type.value} available: requested {amount}, available {await self._get_available_amount(resource_type)}')
        self._allocation_counter += 1
        allocation_id = f'alloc_{self._allocation_counter}_{resource_type.value}'
        allocated_resources = {resource_type: amount}
        allocation = ResourceAllocation(
            allocation_id=allocation_id,
            workflow_id=workflow_id or f'workflow_{self._allocation_counter}',
            allocated_resources=allocated_resources
        )
        if timeout:
            allocation.expires_at = datetime.now(timezone.utc) + timedelta(seconds=timeout)
        self.allocations[allocation_id] = allocation
        await self._update_usage_tracking()
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.RESOURCE_ALLOCATED, source='resource_manager', data={'allocation_id': allocation_id, 'resource_type': resource_type.value, 'amount': amount, 'component_name': component_name, 'workflow_id': workflow_id}))
        self.logger.info('Allocated %s %s to %s (allocation: %s)', amount, resource_type.value, component_name, allocation_id)
        return allocation

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
        self.allocation_history.append(allocation)
        await self._update_usage_tracking()
        if self.event_bus:
            # Get first resource type and amount for logging
            first_resource = next(iter(allocation.allocated_resources.items()))
            resource_type, amount = first_resource
            await self.event_bus.emit(MLEvent(event_type=EventType.RESOURCE_RELEASED, source='resource_manager', data={'allocation_id': allocation_id, 'resource_type': resource_type.value, 'amount': amount, 'workflow_id': allocation.workflow_id, 'allocated_resources': {rt.value: amt for rt, amt in allocation.allocated_resources.items()}}))

        # Log released resources
        resources_str = ', '.join([f'{amt} {rt.value}' for rt, amt in allocation.allocated_resources.items()])
        self.logger.info('Released %s for workflow %s (allocation: %s)', resources_str, allocation.workflow_id, allocation_id)
        return True

    async def allocate_resources(self, workflow_id: str, requirements: dict[ResourceType, float]) -> ResourceAllocation:
        """
        Allocate multiple resources for a workflow.

        Args:
            workflow_id: ID of the workflow requesting resources
            requirements: Dict mapping resource types to amounts needed

        Returns:
            ResourceAllocation object with allocated resources
        """
        # Allocate up to available amounts per resource (graceful clamping)
        allocated: dict[ResourceType, float] = {}
        for resource_type, requested in requirements.items():
            available = await self._get_available_amount(resource_type)
            amount = min(float(requested), max(float(available), 0.0))
            allocated[resource_type] = amount

        # Create allocation record
        self._allocation_counter += 1
        allocation_id = f'alloc_{self._allocation_counter}_{workflow_id}'
        allocation = ResourceAllocation(
            allocation_id=allocation_id,
            workflow_id=workflow_id,
            allocated_resources=allocated
        )

        self.allocations[allocation_id] = allocation
        await self._update_usage_tracking()

        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.RESOURCE_ALLOCATED,
                source='resource_manager',
                data={
                    'allocation_id': allocation_id,
                    'workflow_id': workflow_id,
                    'allocated_resources': {rt.value: amt for rt, amt in allocated.items()}
                }
            ))

        resources_str = ', '.join([f'{amt} {rt.value}' for rt, amt in allocated.items()])
        self.logger.info('Allocated %s for workflow %s (allocation: %s)', resources_str, workflow_id, allocation_id)
        return allocation

    async def deallocate_resources(self, allocation_id: str) -> bool:
        """Deallocate resources by allocation ID. Alias for release_resource."""
        return await self.release_resource(allocation_id)

    async def get_resource_usage_stats(self) -> ResourceUsage:
        """
        Get aggregate resource usage statistics.

        Returns:
            ResourceUsage object with aggregate statistics
        """
        await self._update_usage_tracking()

        # Calculate totals across all allocations
        total_cpu = sum(
            allocation.allocated_resources.get(ResourceType.CPU, 0)
            for allocation in self.allocations.values()
        )
        total_memory = int(sum(
            allocation.allocated_resources.get(ResourceType.MEMORY, 0)
            for allocation in self.allocations.values()
        ))
        total_gpu = int(sum(
            allocation.allocated_resources.get(ResourceType.GPU, 0)
            for allocation in self.allocations.values()
        ))
        total_gpu_memory = int(sum(
            allocation.allocated_resources.get(ResourceType.GPU_MEMORY, 0)
            for allocation in self.allocations.values()
        ))

        return ResourceUsage(
            total_cpu_allocated=total_cpu,
            total_memory_allocated=total_memory,
            total_gpu_allocated=total_gpu,
            total_gpu_memory_allocated=total_gpu_memory,
            active_allocations=len(self.allocations),
            total_allocations=self._allocation_counter,
            allocation_history_size=len(getattr(self, 'allocation_history', []))
        )

    async def get_usage_stats(self) -> dict[str, Any]:
        """Get current resource usage statistics."""
        await self._update_usage_tracking()

        # Return basic stats based on current allocations
        total_allocations = len(self.allocations)
        active_workflows = len({alloc.workflow_id for alloc in self.allocations.values()})

        stats: dict[str, Any] = {
            'total_active_allocations': total_allocations,
            'active_workflows': active_workflows,
            'allocation_history_size': len(self.allocation_history)
        }

        # Add resource-specific stats
        for resource_type in ResourceType:
            allocated = sum(
                alloc.allocated_resources.get(resource_type, 0)
                for alloc in self.allocations.values()
            )
            if allocated > 0:
                stats[f'{resource_type.value}_allocated'] = allocated

        return stats

    async def check_resource_availability(self, requirements: dict[ResourceType, float]) -> bool:
        """
        Check if requested resources are available.

        Args:
            requirements: Dict mapping resource types to amounts needed

        Returns:
            True if all requested resources are available
        """
        for resource_type, amount in requirements.items():
            if not await self._is_resource_available(resource_type, amount):
                return False
        return True

    async def optimize_memory_usage(self) -> bool:
        """
        Optimize memory usage by cleaning up expired allocations.

        Returns:
            True if optimization was performed
        """
        try:
            cleaned = await self._cleanup_expired_allocations()
            if cleaned > 0:
                self.logger.info(f"Optimized memory usage by cleaning {cleaned} expired allocations")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error during memory optimization: {e}")
            return False

    async def _trigger_garbage_collection(self) -> None:
        """Trigger garbage collection and cleanup."""
        import gc
        try:
            # Force garbage collection
            collected = gc.collect()
            self.logger.debug(f"Garbage collection freed {collected} objects")

            # Clean up expired allocations
            await self._cleanup_expired_allocations()

            # Trim allocation history if it gets too large
            if len(self.allocation_history) > 1000:
                self.allocation_history = self.allocation_history[-500:]
                self.logger.debug("Trimmed allocation history")

        except Exception as e:
            self.logger.error(f"Error during garbage collection: {e}")

    async def _cleanup_expired_allocations(self) -> int:
        """
        Clean up expired resource allocations.

        Returns:
            Number of allocations cleaned up
        """
        current_time = datetime.now(timezone.utc)
        expired_ids: list[str] = []

        for allocation_id, allocation in self.allocations.items():
            if allocation.expires_at and current_time > allocation.expires_at:
                expired_ids.append(allocation_id)

        # Remove expired allocations
        for allocation_id in expired_ids:
            await self.release_resource(allocation_id)

        if expired_ids:
            self.logger.info(f"Cleaned up {len(expired_ids)} expired allocations")

        return len(expired_ids)

    async def get_allocations(self, workflow_id: str | None=None) -> list[ResourceAllocation]:
        """Get current resource allocations, optionally filtered by workflow_id."""
        allocations = list(self.allocations.values())
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
        self.resource_limits = {
            ResourceType.CPU: float(self.config.cpu_limit_cores),
            ResourceType.MEMORY: self.config.memory_limit_gb * 1024 * 1024 * 1024,
            ResourceType.GPU: 1.0,
            ResourceType.GPU_MEMORY: 24.0 * 1024 * 1024 * 1024,
            ResourceType.GPU_MIG_SLICE: 7.0,
            ResourceType.GPU_TIME_SLICE: 4.0,
            ResourceType.DISK: 100.0 * 1024 * 1024 * 1024,
            ResourceType.DATABASE_CONNECTIONS: float(self.config.db_connection_pool_size),
            ResourceType.CACHE_CONNECTIONS: float(self.config.redis_connection_pool_size),
        }
        self.logger.info('Initialized resource limits: %s', self.resource_limits)

    async def _initialize_usage_tracking(self) -> None:
        """Initialize resource usage tracking (no-op; aggregate stats are computed on demand)."""
        return None

    async def _start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.is_monitoring:
            return
        self.is_monitoring = True
        self._task_manager = get_background_task_manager()
        self.monitoring_task_id = await self._task_manager.submit_enhanced_task(
            task_id=f'ml_resource_monitor_{str(uuid.uuid4())[:8]}',
            coroutine=self._monitoring_loop(),
            priority=TaskPriority.HIGH,
            tags={'service': 'ml', 'type': 'monitoring', 'component': 'resource_monitor', 'module': 'resource_manager'}
        )
        self.logger.info('Started resource monitoring')

    async def _stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        if not self.is_monitoring:
            return
        self.is_monitoring = False
        # cancel background monitoring task if registered
        if self._task_manager and self.monitoring_task_id:
            try:
                await self._task_manager.cancel_task(self.monitoring_task_id)
            except Exception as e:
                self.logger.debug('Error cancelling monitoring task %s: %s', self.monitoring_task_id, e)
            finally:
                self.monitoring_task_id = None
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
        # Simplified tracking for new structure - aggregate statistics calculated in get_resource_usage_stats()
        total_allocations = len(self.allocations)
        if total_allocations > 0:
            self.logger.debug(f"Current allocations: {total_allocations}")
        # Note: current_usage dict no longer maintained with new ResourceUsage structure

    async def _get_total_available(self, resource_type: ResourceType) -> float:
        """Get total available amount for a resource type."""
        if resource_type == ResourceType.CPU:
            cpu_count = psutil.cpu_count() or 1
            return float(cpu_count)
        elif resource_type == ResourceType.MEMORY:
            return float(psutil.virtual_memory().total)
        else:
            return float(self.resource_limits[resource_type])

    async def _is_resource_available(self, resource_type: ResourceType, amount: float) -> bool:
        """Check if a resource amount is available for allocation."""
        available = await self._get_available_amount(resource_type)
        return available >= amount

    async def _get_available_amount(self, resource_type: ResourceType) -> float:
        """Get currently available amount for a resource type."""
        total_available = await self._get_total_available(resource_type)
        currently_allocated = sum(
            allocation.allocated_resources.get(resource_type, 0)
            for allocation in self.allocations.values()
        )
        return total_available - currently_allocated

    async def _check_resource_thresholds(self) -> None:
        """Check resource usage against alert thresholds."""
        # Check critical resource types for threshold breaches
        critical_types = [ResourceType.CPU, ResourceType.MEMORY, ResourceType.GPU]

        for resource_type in critical_types:
            total_available = await self._get_total_available(resource_type)
            if total_available <= 0:
                continue

            currently_allocated = sum(
                allocation.allocated_resources.get(resource_type, 0)
                for allocation in self.allocations.values()
            )
            usage_percentage = (currently_allocated / total_available) * 100

            # Check thresholds
            if resource_type == ResourceType.CPU and usage_percentage > self.config.alert_threshold_cpu * 100:
                await self._emit_threshold_alert(resource_type, usage_percentage)
            elif resource_type == ResourceType.MEMORY and usage_percentage > self.config.alert_threshold_memory * 100:
                await self._emit_threshold_alert(resource_type, usage_percentage)
            elif resource_type == ResourceType.GPU:
                gpu_threshold = getattr(self.config, 'alert_threshold_gpu', self.config.alert_threshold_cpu)
                if usage_percentage > gpu_threshold * 100:
                    await self._emit_threshold_alert(resource_type, usage_percentage)

    async def _emit_threshold_alert(self, resource_type: ResourceType, usage_percentage: float) -> None:
        """Emit an alert when resource usage exceeds thresholds."""
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.ALERT_TRIGGERED, source='resource_manager', data={'alert_type': 'resource_threshold_exceeded', 'resource_type': resource_type.value, 'usage_percentage': usage_percentage}))


    async def _optimize_resource_usage(self, resource_type: ResourceType) -> None:
        """Implement resource optimization strategies."""
        self.logger.info('Optimizing %s resource usage', resource_type.value)

    async def allocate_gpu_resources(self, component_name: str, gpu_type: str='shared', gpu_memory_gb: float | None=None, mig_profile: str | None=None, workflow_id: str | None=None) -> ResourceAllocation | None:
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

    async def get_gpu_utilization_metrics(self) -> dict[str, Any]:
        """Get GPU utilization metrics (following 2025 monitoring patterns)."""
        try:
            gpu_metrics: dict[str, Any] = {
                'gpu_utilization_percentage': 0.0,
                'gpu_memory_utilization_percentage': 0.0,
                'gpu_temperature_celsius': 0.0,
                'gpu_power_usage_watts': 0.0,
                'active_allocations': {'dedicated': 0, 'time_sliced': 0, 'mig_slices': 0},
                'last_updated': datetime.now(timezone.utc).isoformat(),
            }
            active = gpu_metrics['active_allocations']
            for allocation in self.allocations.values():
                for rt in allocation.allocated_resources.keys():
                    if rt == ResourceType.GPU:
                        active['dedicated'] += 1
                    elif rt == ResourceType.GPU_TIME_SLICE:
                        active['time_sliced'] += 1
                    elif rt == ResourceType.GPU_MIG_SLICE:
                        active['mig_slices'] += 1
            total_allocations = int(sum(active.values()))
            if total_allocations > 0:
                gpu_metrics['gpu_utilization_percentage'] = min(total_allocations * 25.0, 100.0)
                gpu_metrics['gpu_memory_utilization_percentage'] = min(total_allocations * 30.0, 100.0)
            return gpu_metrics
        except Exception as e:
            self.logger.error('Error getting GPU metrics: %s', e)
            return {}

    async def implement_gpu_autoscaling(self, utilization_threshold: float=0.8) -> dict[str, Any]:
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
                    await self.event_bus.emit(MLEvent(event_type=EventType.ALERT_TRIGGERED, source='resource_manager', data={'resource_type': 'gpu', 'action': 'scale_up', 'current_utilization': current_utilization, 'threshold': utilization_threshold}))
            elif current_utilization < utilization_threshold * 0.5:
                scaling_decision['should_scale_down'] = True
                scaling_decision['recommended_action'] = 'scale_down'
                if self.event_bus:
                    await self.event_bus.emit(MLEvent(event_type=EventType.ALERT_TRIGGERED, source='resource_manager', data={'resource_type': 'gpu', 'action': 'scale_down', 'current_utilization': current_utilization, 'threshold': utilization_threshold}))
            return scaling_decision
        except Exception as e:
            self.logger.error('Error in GPU autoscaling: %s', e)
            return {'error': str(e)}

    async def _release_all_allocations(self) -> None:
        """Release all current allocations during shutdown."""
        allocation_ids = list(self.allocations.keys())
        for allocation_id in allocation_ids:
            await self.release_resource(allocation_id)

    async def get_cluster_resource_info(self) -> dict[str, Any]:
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

    async def get_namespace_resource_utilization(self, namespace: str | None=None) -> dict[str, Any]:
        """Get resource utilization for a specific namespace."""
        if not self.k8s_manager or not self.k8s_manager.is_available():
            return {'error': 'Kubernetes integration not available'}
        try:
            quotas = await self.k8s_manager.get_namespace_resource_quotas(namespace)
            pods = await self.k8s_manager.get_pod_resources(namespace)
            return {'namespace': namespace or self.k8s_manager.namespace, 'resource_quotas': len(quotas), 'active_pods': len(pods), 'quotas': [{'name': quota.quota_name, 'cpu_usage': f'{quota.cpu_used}/{quota.cpu_hard}', 'memory_usage': f'{quota.memory_used}/{quota.memory_hard}', 'gpu_usage': f'{quota.gpu_used}/{quota.gpu_hard}'} for quota in quotas], 'pod_summary': {'total_pods': len(pods), 'total_cpu_requests': sum(self.k8s_manager._parse_cpu_resource(pod.cpu_request) for pod in pods), 'total_memory_requests': sum(self.k8s_manager._parse_memory_resource(pod.memory_request) for pod in pods), 'total_gpu_requests': sum(int(pod.gpu_request) for pod in pods)}}
        except Exception as e:
            self.logger.error('Error getting namespace resource utilization: %s', e)
            return {'error': str(e)}

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        """Get status of all circuit breakers."""
        return {resource_type.value: breaker.get_state() for resource_type, breaker in self.circuit_breakers.items()}

class ResourceExhaustionError(Exception):
    """Raised when requested resources are not available."""
    pass
