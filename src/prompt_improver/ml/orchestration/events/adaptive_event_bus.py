"""
Adaptive Event Bus for ML Pipeline Orchestrator Performance Optimization.

Implements dynamic queue sizing and performance optimization following 2025 best practices.
Target: >8,000 events/second processing with adaptive scaling.

Key Features:
- Dynamic queue sizing based on load (1,000 → 10,000+ events)
- Multi-worker event processing with auto-scaling
- Performance telemetry and real-time monitoring
- Circuit breaker patterns for resilience
- Backpressure handling and flow control
"""
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Set
from ....performance.monitoring.health.background_manager import EnhancedBackgroundTaskManager, TaskPriority, get_background_task_manager
from ..config.orchestrator_config import OrchestratorConfig
from .event_types import EventType, MLEvent

class EventBusState(Enum):
    """Event bus operational states."""
    INITIALIZING = 'initializing'
    RUNNING = 'running'
    SCALING_UP = 'scaling_up'
    SCALING_DOWN = 'scaling_down'
    DEGRADED = 'degraded'
    SHUTDOWN = 'shutdown'

@dataclass
class EventProcessingMetrics:
    """Real-time event processing metrics."""
    events_processed: int = 0
    events_failed: int = 0
    events_dropped: int = 0
    avg_processing_time_ms: float = 0.0
    queue_size: int = 0
    worker_count: int = 0
    throughput_per_second: float = 0.0
    backpressure_events: int = 0
    circuit_breaker_trips: int = 0
    last_scale_event: Optional[datetime] = None
    processing_times: deque = field(default_factory=lambda: deque(maxlen=1000))

@dataclass
class EventSubscription:
    """Enhanced event subscription with performance tracking."""
    event_type: EventType
    handler: Callable[[MLEvent], Any]
    subscription_id: str
    is_async: bool
    priority: int = 5
    max_processing_time_ms: float = 1000.0
    failure_count: int = 0
    success_count: int = 0
    avg_processing_time_ms: float = 0.0

class AdaptiveEventBus:
    """
    High-performance adaptive event bus with dynamic scaling.
    
    Optimizes for >8,000 events/second with intelligent queue management
    and worker auto-scaling based on real-time performance metrics.
    """

    def __init__(self, config: OrchestratorConfig, task_manager: EnhancedBackgroundTaskManager | None=None):
        """Initialize adaptive event bus."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.min_queue_size = 1000
        self.max_queue_size = 50000
        self.current_queue_size = config.event_bus_buffer_size
        self.task_manager = task_manager or get_background_task_manager()
        self.min_workers = 2
        self.max_workers = 20
        self.current_workers = 2
        self.worker_task_ids: List[str] = []
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=self.current_queue_size)
        self.priority_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=1000)
        self.subscriptions: Dict[EventType, List[EventSubscription]] = defaultdict(list)
        self.metrics = EventProcessingMetrics()
        self.performance_window = deque(maxlen=100)
        self.last_metrics_update = time.time()
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.scale_cooldown_seconds = 30
        self.last_scale_time = 0
        self.circuit_breaker_threshold = 10
        self.circuit_breaker_timeout = 60
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = 0
        self.circuit_breaker_open = False
        self.state = EventBusState.INITIALIZING
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        self._subscription_counter = 0
        self.logger.info('AdaptiveEventBus initialized with queue_size={self.current_queue_size}, workers=%s', self.current_workers)

    async def start(self) -> None:
        """Start the adaptive event bus."""
        if self.is_running:
            return
        self.logger.info('Starting adaptive event bus...')
        self.state = EventBusState.RUNNING
        self.is_running = True
        await self._scale_workers(self.current_workers)
        monitoring_task_id = f'event_bus_monitor_{int(time.time())}'
        self._monitoring_task_id = await self.task_manager.submit_enhanced_task(task_id=monitoring_task_id, coroutine=self._performance_monitor(), priority=TaskPriority.HIGH, timeout=24 * 3600, tags={'type': 'event_bus_monitoring', 'component': 'adaptive_event_bus', 'function': 'performance_monitor'})
        self.logger.info('Adaptive event bus started with %s managed workers', len(self.worker_task_ids))

    async def stop(self) -> None:
        """Stop the adaptive event bus gracefully."""
        if not self.is_running:
            return
        self.logger.info('Stopping adaptive event bus...')
        self.state = EventBusState.SHUTDOWN
        self.is_running = False
        self.shutdown_event.set()
        if hasattr(self, '_monitoring_task_id'):
            await self.task_manager.cancel_task(self._monitoring_task_id)
        cancel_tasks = []
        for task_id in self.worker_task_ids:
            cancel_tasks.append(self.task_manager.cancel_task(task_id))
        if cancel_tasks:
            await asyncio.gather(*cancel_tasks, return_exceptions=True)
        self.worker_task_ids.clear()
        self.logger.info('Adaptive event bus stopped with managed task cleanup')

    def subscribe(self, event_type: EventType, handler: Callable[[MLEvent], Any], priority: int=5, max_processing_time_ms: float=1000.0) -> str:
        """
        Subscribe to events with priority and performance constraints.
        
        Args:
            event_type: Type of events to subscribe to
            handler: Event handler function
            priority: Handler priority (1-10, higher = more priority)
            max_processing_time_ms: Maximum allowed processing time
            
        Returns:
            Subscription ID
        """
        self._subscription_counter += 1
        subscription_id = f'sub_{self._subscription_counter}'
        subscription = EventSubscription(event_type=event_type, handler=handler, subscription_id=subscription_id, is_async=asyncio.iscoroutinefunction(handler), priority=priority, max_processing_time_ms=max_processing_time_ms)
        self.subscriptions[event_type].append(subscription)
        self.subscriptions[event_type].sort(key=lambda s: s.priority, reverse=True)
        self.logger.debug('Subscribed {subscription_id} to {event_type.value} with priority %s', priority)
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        for event_type, subscriptions in self.subscriptions.items():
            for i, subscription in enumerate(subscriptions):
                if subscription.subscription_id == subscription_id:
                    subscriptions.pop(i)
                    self.logger.debug('Unsubscribed %s', subscription_id)
                    return True
        return False

    async def emit(self, event: MLEvent, priority: int=5) -> None:
        """
        Emit an event with priority handling.
        
        Args:
            event: Event to emit
            priority: Event priority (1-10, higher = more priority)
        """
        if not self.is_running:
            self.logger.warning('Event bus not running, discarding event')
            return
        if self.circuit_breaker_open:
            if time.time() - self.circuit_breaker_last_failure < self.circuit_breaker_timeout:
                self.logger.warning('Circuit breaker open, dropping event')
                self.metrics.events_dropped += 1
                return
            else:
                self.circuit_breaker_open = False
                self.circuit_breaker_failures = 0
                self.logger.info('Circuit breaker closed, resuming event processing')
        try:
            if priority >= 8:
                await asyncio.wait_for(self.priority_queue.put((10 - priority, event)), timeout=0.1)
            else:
                await asyncio.wait_for(self.event_queue.put(event), timeout=0.1)
            self.logger.debug('Emitted event {event.event_type.value} with priority %s', priority)
        except asyncio.TimeoutError:
            await self._handle_backpressure(event)
        except Exception as e:
            self.logger.error('Error emitting event: %s', e)
            await self._handle_circuit_breaker_failure()

    async def emit_and_wait(self, event: MLEvent, timeout: float=5.0) -> List[Any]:
        """Emit event and wait for all handlers to complete."""
        if not self.is_running:
            return []
        handlers = self.subscriptions.get(event.event_type, [])
        if not handlers:
            return []
        results = []
        start_time = time.time()
        try:
            for subscription in handlers:
                try:
                    handler_start = time.time()
                    if subscription.is_async:
                        result = await asyncio.wait_for(subscription.handler(event), timeout=subscription.max_processing_time_ms / 1000.0)
                    else:
                        result = subscription.handler(event)
                    handler_time = (time.time() - handler_start) * 1000
                    await self._update_subscription_metrics(subscription, handler_time, True)
                    results.append(result)
                except asyncio.TimeoutError:
                    self.logger.warning('Handler %s timed out', subscription.subscription_id)
                    await self._update_subscription_metrics(subscription, subscription.max_processing_time_ms, False)
                except Exception as e:
                    self.logger.error('Handler {subscription.subscription_id} failed: %s', e)
                    await self._update_subscription_metrics(subscription, 0, False)
            total_time = (time.time() - start_time) * 1000
            self.metrics.processing_times.append(total_time)
        except Exception as e:
            self.logger.error('Error in emit_and_wait: %s', e)
        return results

    async def _worker_loop(self, worker_id: int) -> None:
        """Main worker loop for processing events."""
        self.logger.debug('Worker %s started', worker_id)
        while self.is_running:
            try:
                event = None
                try:
                    _, event = await asyncio.wait_for(self.priority_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    try:
                        event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                if event:
                    await self._process_event(event, worker_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error('Worker {worker_id} error: %s', e)
                await self._handle_circuit_breaker_failure()
        self.logger.debug('Worker %s stopped', worker_id)

    async def _process_event(self, event: MLEvent, worker_id: int) -> None:
        """Process a single event."""
        start_time = time.time()
        try:
            handlers = self.subscriptions.get(event.event_type, [])
            for subscription in handlers:
                try:
                    handler_start = time.time()
                    if subscription.is_async:
                        await asyncio.wait_for(subscription.handler(event), timeout=subscription.max_processing_time_ms / 1000.0)
                    else:
                        subscription.handler(event)
                    handler_time = (time.time() - handler_start) * 1000
                    await self._update_subscription_metrics(subscription, handler_time, True)
                except asyncio.TimeoutError:
                    self.logger.warning('Handler %s timed out', subscription.subscription_id)
                    await self._update_subscription_metrics(subscription, subscription.max_processing_time_ms, False)
                except Exception as e:
                    self.logger.error('Handler {subscription.subscription_id} failed: %s', e)
                    await self._update_subscription_metrics(subscription, 0, False)
            processing_time = (time.time() - start_time) * 1000
            self.metrics.events_processed += 1
            self.metrics.processing_times.append(processing_time)
            if self.metrics.processing_times:
                self.metrics.avg_processing_time_ms = statistics.mean(self.metrics.processing_times)
        except Exception as e:
            self.logger.error('Error processing event {event.event_type.value}: %s', e)
            self.metrics.events_failed += 1
            await self._handle_circuit_breaker_failure()

    async def _scale_workers(self, target_count: int) -> None:
        """Scale worker count to target."""
        current_count = len(self.worker_tasks)
        if target_count > current_count:
            for i in range(current_count, target_count):
                worker_task_id = await self.task_manager.submit_enhanced_task(task_id=f'event_bus_worker_{i}_{int(time.time())}', coroutine=self._worker_loop(i), priority=TaskPriority.HIGH, timeout=24 * 3600, tags={'type': 'event_bus_worker', 'component': 'adaptive_event_bus', 'worker_id': str(i)})
                self.worker_task_ids.append(worker_task_id)
            self.logger.info('Scaled up managed workers: {current_count} → %s', target_count)
        elif target_count < current_count:
            task_ids_to_cancel = self.worker_task_ids[target_count:]
            self.worker_task_ids = self.worker_task_ids[:target_count]
            cancel_tasks = []
            for task_id in task_ids_to_cancel:
                cancel_tasks.append(self.task_manager.cancel_task(task_id))
            if cancel_tasks:
                await asyncio.gather(*cancel_tasks, return_exceptions=True)
            self.logger.info('Scaled down managed workers: {current_count} → %s', target_count)
        self.current_workers = target_count
        self.metrics.worker_count = target_count

    async def _performance_monitor(self) -> None:
        """Monitor performance and trigger adaptive scaling."""
        while self.is_running:
            try:
                await asyncio.sleep(5)
                await self._update_performance_metrics()
                await self._evaluate_scaling()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error('Performance monitor error: %s', e)

    async def _update_performance_metrics(self) -> None:
        """Update real-time performance metrics."""
        current_time = time.time()
        time_delta = current_time - self.last_metrics_update
        if time_delta > 0:
            events_in_period = self.metrics.events_processed
            self.metrics.throughput_per_second = events_in_period / time_delta
            self.metrics.queue_size = self.event_queue.qsize() + self.priority_queue.qsize()
            self.performance_window.append({'timestamp': current_time, 'throughput': self.metrics.throughput_per_second, 'queue_size': self.metrics.queue_size, 'worker_count': self.current_workers, 'avg_processing_time': self.metrics.avg_processing_time_ms})
        self.last_metrics_update = current_time

    async def _evaluate_scaling(self) -> None:
        """Evaluate if scaling is needed."""
        if time.time() - self.last_scale_time < self.scale_cooldown_seconds:
            return
        queue_utilization = self.metrics.queue_size / self.current_queue_size
        if queue_utilization > self.scale_up_threshold and self.current_workers < self.max_workers:
            new_worker_count = min(self.current_workers + 2, self.max_workers)
            await self._scale_workers(new_worker_count)
            self.last_scale_time = time.time()
            self.metrics.last_scale_event = datetime.now(timezone.utc)
            self.state = EventBusState.SCALING_UP
        elif queue_utilization < self.scale_down_threshold and self.current_workers > self.min_workers and (self.metrics.throughput_per_second < 1000):
            new_worker_count = max(self.current_workers - 1, self.min_workers)
            await self._scale_workers(new_worker_count)
            self.last_scale_time = time.time()
            self.metrics.last_scale_event = datetime.now(timezone.utc)
            self.state = EventBusState.SCALING_DOWN
        else:
            self.state = EventBusState.RUNNING

    async def _handle_backpressure(self, event: MLEvent) -> None:
        """Handle backpressure when queues are full."""
        self.metrics.backpressure_events += 1
        if self.current_queue_size < self.max_queue_size:
            new_size = min(self.current_queue_size * 2, self.max_queue_size)
            await self._resize_queue(new_size)
            try:
                await asyncio.wait_for(self.event_queue.put(event), timeout=0.1)
                return
            except asyncio.TimeoutError:
                pass
        self.metrics.events_dropped += 1
        self.logger.warning('Dropped event %s due to backpressure', event.event_type.value)

    async def _resize_queue(self, new_size: int) -> None:
        """Resize the event queue."""
        old_queue = self.event_queue
        self.event_queue = asyncio.Queue(maxsize=new_size)
        while not old_queue.empty():
            try:
                event = old_queue.get_nowait()
                await self.event_queue.put(event)
            except asyncio.QueueEmpty:
                break
            except asyncio.QueueFull:
                break
        self.current_queue_size = new_size
        self.logger.info('Resized event queue to %s', new_size)

    async def _handle_circuit_breaker_failure(self) -> None:
        """Handle circuit breaker failure."""
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = time.time()
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            self.circuit_breaker_open = True
            self.metrics.circuit_breaker_trips += 1
            self.state = EventBusState.DEGRADED
            self.logger.error('Circuit breaker opened due to failures')

    async def _update_subscription_metrics(self, subscription: EventSubscription, processing_time_ms: float, success: bool) -> None:
        """Update subscription performance metrics."""
        if success:
            subscription.success_count += 1
            total_time = subscription.avg_processing_time_ms * (subscription.success_count - 1) + processing_time_ms
            subscription.avg_processing_time_ms = total_time / subscription.success_count
        else:
            subscription.failure_count += 1

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {'state': self.state.value, 'events_processed': self.metrics.events_processed, 'events_failed': self.metrics.events_failed, 'events_dropped': self.metrics.events_dropped, 'throughput_per_second': self.metrics.throughput_per_second, 'avg_processing_time_ms': self.metrics.avg_processing_time_ms, 'queue_size': self.metrics.queue_size, 'queue_capacity': self.current_queue_size, 'queue_utilization': self.metrics.queue_size / self.current_queue_size if self.current_queue_size > 0 else 0, 'worker_count': self.current_workers, 'backpressure_events': self.metrics.backpressure_events, 'circuit_breaker_trips': self.metrics.circuit_breaker_trips, 'circuit_breaker_open': self.circuit_breaker_open, 'subscription_count': sum((len(subs) for subs in self.subscriptions.values())), 'last_scale_event': self.metrics.last_scale_event.isoformat() if self.metrics.last_scale_event else None, 'performance_window': list(self.performance_window)}

    async def initialize(self) -> None:
        """Initialize the event bus (alias for start() for compatibility)."""
        await self.start()

    async def shutdown(self) -> None:
        """Shutdown the event bus (alias for stop() for compatibility)."""
        await self.stop()

    def get_event_history(self, event_type: Optional[EventType]=None, limit: int=100) -> List[MLEvent]:
        """
        Get recent event history for compatibility with Basic EventBus.
        
        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return
            
        Returns:
            List of recent events (empty list as this implementation doesn't store history)
        """
        self.logger.debug('Event history not available in AdaptiveEventBus - optimized for performance')
        return []

    def get_subscription_count(self, event_type: Optional[EventType]=None) -> int:
        """
        Get number of active subscriptions.
        
        Args:
            event_type: Count for specific event type (optional)
            
        Returns:
            Number of subscriptions
        """
        if event_type:
            return len(self.subscriptions.get(event_type, []))
        return sum((len(subs) for subs in self.subscriptions.values()))

    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics for compatibility with Basic EventBus."""
        return {'total_events': self.metrics.events_processed, 'failed_events': self.metrics.events_failed, 'active_handlers': self.get_subscription_count(), 'queue_size': self.metrics.queue_size, 'is_running': self.is_running}
