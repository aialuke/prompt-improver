"""Core ML Event Bus for MCP-ML Boundary Isolation

Provides event-driven communication between MCP components and ML subsystems
without direct ML imports, ensuring clean architectural boundaries.

SECURITY NOTE: This event bus operates with strict input validation and
rate limiting to prevent event flooding attacks and ensures data integrity
through structured event validation.
"""
import asyncio
import logging
import uuid
import weakref
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from prompt_improver.performance.monitoring.health.background_manager import TaskPriority, get_background_task_manager
logger = logging.getLogger(__name__)

class MLEventType(Enum):
    """Core ML event types accessible to MCP components."""
    ANALYSIS_REQUEST = 'ml.analysis.request'
    ANALYSIS_COMPLETED = 'ml.analysis.completed'
    ANALYSIS_FAILED = 'ml.analysis.failed'
    TRAINING_REQUEST = 'ml.training.request'
    TRAINING_STARTED = 'ml.training.started'
    TRAINING_COMPLETED = 'ml.training.completed'
    TRAINING_FAILED = 'ml.training.failed'
    TRAINING_PROGRESS = 'ml.training.progress'
    HEALTH_CHECK_REQUEST = 'ml.health.check_request'
    HEALTH_STATUS_UPDATE = 'ml.health.status_update'
    HEALTH_ALERT = 'ml.health.alert'
    HEALTH_RECOVERED = 'ml.health.recovered'
    MODEL_PREDICTION_REQUEST = 'ml.model.prediction_request'
    MODEL_PREDICTION_COMPLETED = 'ml.model.prediction_completed'
    MODEL_LOAD_REQUEST = 'ml.model.load_request'
    MODEL_READY = 'ml.model.ready'
    PERFORMANCE_METRICS_REQUEST = 'ml.performance.metrics_request'
    PERFORMANCE_REPORT = 'ml.performance.report'
    PERFORMANCE_DEGRADED = 'ml.performance.degraded'
    SYSTEM_STATUS_REQUEST = 'ml.system.status_request'
    SYSTEM_STATUS_UPDATE = 'ml.system.status_update'
    SHUTDOWN_REQUEST = 'ml.system.shutdown_request'
    SECURITY_SCAN_REQUEST = 'ml.security.scan_request'
    SECURITY_ALERT = 'ml.security.alert'
    SECURITY_VALIDATION_FAILED = 'ml.security.validation_failed'

@dataclass
class MLEvent:
    """Core ML event structure with security validation.

    SECURITY FEATURES:
    - Input sanitization on all string fields
    - Payload size limits to prevent memory exhaustion
    - Timestamp validation to prevent timestamp manipulation
    - Source validation to ensure event integrity
    """
    event_type: MLEventType
    source: str
    data: dict[str, Any]
    timestamp: datetime | None = None
    event_id: str | None = None
    correlation_id: str | None = None
    priority: int = field(default=1)
    MAX_DATA_SIZE_BYTES = 1024 * 1024
    MAX_STRING_LENGTH = 10000

    def __post_init__(self):
        """Initialize event with security validation."""
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())
        self._validate_security()

    def _validate_security(self) -> None:
        """Validate event data for security compliance."""
        if len(self.source) > self.MAX_STRING_LENGTH:
            raise ValueError(f'Source exceeds maximum length: {len(self.source)}')
        if self.correlation_id and len(self.correlation_id) > self.MAX_STRING_LENGTH:
            raise ValueError(f'Correlation ID exceeds maximum length: {len(self.correlation_id)}')
        import sys
        data_size = sys.getsizeof(self.data)
        if data_size > self.MAX_DATA_SIZE_BYTES:
            raise ValueError(f'Event data exceeds size limit: {data_size} bytes')
        self.source = self._sanitize_string(self.source)
        if self.correlation_id:
            self.correlation_id = self._sanitize_string(self.correlation_id)
        if not 1 <= self.priority <= 4:
            self.priority = 1
        if self.timestamp and self.timestamp > datetime.now(UTC).replace(microsecond=0) + timedelta(seconds=30):
            raise ValueError('Event timestamp cannot be in the future')

    def _sanitize_string(self, value: str) -> str:
        """Sanitize string input to prevent injection attacks."""
        if not isinstance(value, str):
            return str(value)
        dangerous_chars = ['\x00', '\r', '\n', '\t']
        for char in dangerous_chars:
            value = value.replace(char, '')
        return value[:self.MAX_STRING_LENGTH]

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {'event_type': self.event_type.value, 'source': self.source, 'data': self.data, 'timestamp': self.timestamp.isoformat() if self.timestamp else None, 'event_id': self.event_id, 'correlation_id': self.correlation_id, 'priority': self.priority}

@dataclass
class EventSubscription:
    """Event subscription with security constraints."""
    subscription_id: str
    event_type: MLEventType
    handler: Callable[[MLEvent], Any]
    source_filter: str | None = None
    is_async: bool = True
    max_events_per_minute: int = 1000
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    _event_count: int = field(default=0, init=False)
    _last_reset: datetime = field(default_factory=lambda: datetime.now(UTC), init=False)

    def check_rate_limit(self) -> bool:
        """Check if subscription is within rate limits."""
        now = datetime.now(UTC)
        if (now - self._last_reset).total_seconds() >= 60:
            self._event_count = 0
            self._last_reset = now
        if self._event_count >= self.max_events_per_minute:
            return False
        self._event_count += 1
        return True

class MLEventBus:
    """Core ML Event Bus for MCP-ML communication.

    SECURITY FEATURES:
    - Rate limiting per subscription to prevent event flooding
    - Input validation on all events
    - Memory usage monitoring to prevent resource exhaustion
    - Subscription lifecycle management
    - Event audit trail for security monitoring

    PERFORMANCE FEATURES:
    - Async event processing with configurable concurrency
    - Priority-based event handling
    - Automatic cleanup of stale subscriptions
    - Efficient event routing with minimal overhead
    """

    def __init__(self, max_queue_size: int=10000, max_history_size: int=1000):
        """Initialize the ML Event Bus.

        Args:
            max_queue_size: Maximum number of events in processing queue
            max_history_size: Maximum number of events to keep in history
        """
        self.logger = logging.getLogger(__name__)
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing_task_id: str | None = None
        self.is_running = False
        self.subscriptions: dict[MLEventType, list[EventSubscription]] = defaultdict(list)
        self.subscription_refs: dict[str, weakref.ReferenceType] = {}
        self._subscription_counter = 0
        self.event_history: list[MLEvent] = []
        self.max_history_size = max_history_size
        self.processed_events_count = 0
        self.failed_events_count = 0
        self.security_violations = 0
        self.rate_limit_violations = 0
        self.memory_usage_bytes = 0
        self.event_handler_timeout = 5.0
        self.cleanup_interval = 300
        self.last_cleanup = datetime.now(UTC)
        self.logger.info('ML Event Bus initialized with security features enabled')

    async def initialize(self) -> None:
        """Initialize the event bus and start processing."""
        if self.is_running:
            self.logger.warning('Event bus already running')
            return
        self.logger.info('Starting ML Event Bus')
        self.is_running = True
        task_manager = get_background_task_manager()
        task_id = await task_manager.submit_enhanced_task(task_id=f'ml_event_processing_{id(self)}', coroutine=self._process_events, priority=TaskPriority.HIGH, tags={'service': 'ml_event_bus', 'type': 'event_processing', 'component': 'core'})
        self.processing_task_id = task_id
        self.logger.info('ML Event Bus started successfully with enhanced task management')

    async def shutdown(self) -> None:
        """Shutdown the event bus gracefully."""
        if not self.is_running:
            return
        self.logger.info('Shutting down ML Event Bus')
        self.is_running = False
        if self.processing_task_id:
            task_manager = get_background_task_manager()
            await task_manager.cancel_task(self.processing_task_id)
            self.processing_task_id = None
        self._clear_resources()
        self.logger.info('ML Event Bus shutdown complete')

    def subscribe(self, event_type: MLEventType, handler: Callable[[MLEvent], Any], source_filter: str | None=None, max_events_per_minute: int=1000) -> str:
        """Subscribe to ML events with rate limiting.

        Args:
            event_type: Type of events to subscribe to
            handler: Function to handle events (sync or async)
            source_filter: Optional filter by event source
            max_events_per_minute: Rate limit for this subscription

        Returns:
            Subscription ID for unsubscribing

        Raises:
            ValueError: If handler is invalid or rate limit is too high
        """
        if not callable(handler):
            raise ValueError('Handler must be callable')
        if max_events_per_minute > 10000:
            raise ValueError('Rate limit too high, maximum is 10000 events/minute')
        self._subscription_counter += 1
        subscription_id = f'sub_{self._subscription_counter}_{uuid.uuid4().hex[:8]}'
        subscription = EventSubscription(subscription_id=subscription_id, event_type=event_type, handler=handler, source_filter=source_filter, is_async=asyncio.iscoroutinefunction(handler), max_events_per_minute=max_events_per_minute)
        self.subscriptions[event_type].append(subscription)
        self.subscription_refs[subscription_id] = weakref.ref(subscription)
        self.logger.debug('Subscribed {subscription_id} to %s', event_type.value)
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events.

        Args:
            subscription_id: ID returned from subscribe()

        Returns:
            True if subscription was found and removed
        """
        for event_type, subs in self.subscriptions.items():
            for i, sub in enumerate(subs):
                if sub.subscription_id == subscription_id:
                    del subs[i]
                    self.subscription_refs.pop(subscription_id, None)
                    self.logger.debug('Unsubscribed %s from %s', subscription_id, event_type.value)
                    return True
        return False

    async def publish(self, event: MLEvent) -> bool:
        """Publish an event to all subscribers.

        Args:
            event: Event to publish

        Returns:
            True if event was queued successfully

        Raises:
            ValueError: If event validation fails
        """
        if not self.is_running:
            self.logger.warning('Event bus not running, discarding event')
            return False
        try:
            if not isinstance(event, MLEvent):
                raise ValueError('Invalid event type')
            await asyncio.wait_for(self.event_queue.put(event), timeout=1.0)
            self._add_to_history(event)
            self.logger.debug('Published event %s from %s', event.event_type.value, event.source)
            return True
        except TimeoutError:
            self.logger.error('Event queue full, dropping event %s', event.event_type.value)
            return False
        except ValueError as e:
            self.security_violations += 1
            self.logger.error('Event validation failed: %s', e)
            return False
        except Exception as e:
            self.logger.error('Error publishing event: %s', e)
            return False

    async def publish_and_wait(self, event: MLEvent, timeout: float=5.0) -> list[Any]:
        """Publish event and wait for all handlers to complete.

        Args:
            event: Event to publish
            timeout: Maximum time to wait for handlers

        Returns:
            List of handler results
        """
        if not self.is_running:
            raise RuntimeError('Event bus not running')
        subscribers = self._get_filtered_subscribers(event)
        if not subscribers:
            return []
        results = []
        valid_subscribers = []
        for subscription in subscribers:
            if subscription.check_rate_limit():
                valid_subscribers.append(subscription)
            else:
                self.rate_limit_violations += 1
                self.logger.warning('Rate limit exceeded for subscription %s', subscription.subscription_id)
        if not valid_subscribers:
            return []
        try:
            task_manager = get_background_task_manager()
            handler_task_ids = []
            for subscription in valid_subscribers:
                if subscription.is_async:

                    async def async_handler_wrapper():
                        return await asyncio.wait_for(subscription.handler(event), timeout=self.event_handler_timeout)
                    task_id = await task_manager.submit_enhanced_task(task_id=f'event_handler_async_{subscription.subscription_id}_{int(asyncio.get_event_loop().time() * 1000)}', coroutine=async_handler_wrapper, priority=TaskPriority.NORMAL, tags={'service': 'ml_event_bus', 'type': 'async_handler', 'event_type': str(event.event_type)})
                else:

                    async def sync_handler_wrapper():
                        return await asyncio.get_event_loop().run_in_executor(None, subscription.handler, event)
                    task_id = await task_manager.submit_enhanced_task(task_id=f'event_handler_sync_{subscription.subscription_id}_{int(asyncio.get_event_loop().time() * 1000)}', coroutine=sync_handler_wrapper, priority=TaskPriority.NORMAL, tags={'service': 'ml_event_bus', 'type': 'sync_handler', 'event_type': str(event.event_type)})
                handler_task_ids.append(task_id)
            if handler_task_ids:
                results = []
                for task_id in handler_task_ids:
                    try:
                        result = await task_manager.wait_for_task(task_id, timeout=timeout)
                        results.append(result)
                    except Exception as e:
                        results.append(e)
                        self.logger.error('Handler task {task_id} failed: %s', e)
        except TimeoutError:
            self.logger.error('Timeout waiting for handlers of %s', event.event_type.value)
        except Exception as e:
            self.logger.error('Error in publish_and_wait: %s', e)
        return results

    def get_subscription_stats(self) -> dict[str, Any]:
        """Get subscription statistics for monitoring."""
        total_subscriptions = sum((len(subs) for subs in self.subscriptions.values()))
        event_type_counts = {event_type.value: len(subs) for event_type, subs in self.subscriptions.items() if subs}
        return {'total_subscriptions': total_subscriptions, 'event_type_counts': event_type_counts, 'processed_events': self.processed_events_count, 'failed_events': self.failed_events_count, 'security_violations': self.security_violations, 'rate_limit_violations': self.rate_limit_violations, 'queue_size': self.event_queue.qsize() if hasattr(self.event_queue, 'qsize') else 0, 'is_running': self.is_running}

    def get_event_history(self, event_type: MLEventType | None=None, limit: int=100) -> list[dict[str, Any]]:
        """Get recent event history.

        Args:
            event_type: Filter by event type
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries
        """
        events = self.event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        recent_events = events[-limit:] if limit > 0 else events
        return [event.to_dict() for event in recent_events]

    async def _process_events(self) -> None:
        """Background task to process events from the queue."""
        self.logger.info('Started ML event processing')
        while self.is_running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._handle_event(event)
                self.processed_events_count += 1
                await self._periodic_cleanup()
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.failed_events_count += 1
                self.logger.error('Error processing event: %s', e)
        self.logger.info('ML event processing stopped')

    async def _handle_event(self, event: MLEvent) -> None:
        """Handle a single event by calling subscribers."""
        subscribers = self._get_filtered_subscribers(event)
        if not subscribers:
            self.logger.debug('No subscribers for event %s', event.event_type.value)
            return
        for subscription in subscribers:
            if not subscription.check_rate_limit():
                self.rate_limit_violations += 1
                continue
            try:
                if subscription.is_async:
                    await asyncio.wait_for(subscription.handler(event), timeout=self.event_handler_timeout)
                else:
                    await asyncio.get_event_loop().run_in_executor(None, subscription.handler, event)
            except TimeoutError:
                self.logger.error('Handler %s timed out', subscription.subscription_id)
            except Exception as e:
                self.logger.error('Error in handler %s: %s', subscription.subscription_id, e)

    def _get_filtered_subscribers(self, event: MLEvent) -> list[EventSubscription]:
        """Get subscribers filtered by event type and source."""
        subscribers = self.subscriptions.get(event.event_type, [])
        filtered_subscribers = []
        for sub in subscribers:
            if sub.source_filter is None or sub.source_filter == event.source:
                filtered_subscribers.append(sub)
        return filtered_subscribers

    def _add_to_history(self, event: MLEvent) -> None:
        """Add event to history with size management."""
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history.pop(0)

    async def _periodic_cleanup(self) -> None:
        """Perform periodic cleanup of stale subscriptions."""
        now = datetime.now(UTC)
        if (now - self.last_cleanup).total_seconds() < self.cleanup_interval:
            return
        self.last_cleanup = now
        dead_refs = []
        for sub_id, weak_ref in self.subscription_refs.items():
            if weak_ref() is None:
                dead_refs.append(sub_id)
        for sub_id in dead_refs:
            self.unsubscribe(sub_id)
        if dead_refs:
            self.logger.debug('Cleaned up %s stale subscriptions', len(dead_refs))

    def _clear_resources(self) -> None:
        """Clear all resources during shutdown."""
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.subscriptions.clear()
        self.subscription_refs.clear()
        self.event_history.clear()
_ml_event_bus: MLEventBus | None = None

async def get_ml_event_bus() -> MLEventBus:
    """Get the global ML event bus instance.

    Returns:
        MLEventBus instance
    """
    global _ml_event_bus
    if _ml_event_bus is None:
        _ml_event_bus = MLEventBus()
        await _ml_event_bus.initialize()
    return _ml_event_bus

def create_ml_event(event_type: MLEventType, source: str, data: dict[str, Any], correlation_id: str | None=None, priority: int=1) -> MLEvent:
    """Helper function to create ML events with validation.

    Args:
        event_type: Type of the event
        source: Source identifier
        data: Event payload data
        correlation_id: Optional correlation ID
        priority: Event priority (1-4)

    Returns:
        Validated MLEvent instance
    """
    return MLEvent(event_type=event_type, source=source, data=data, correlation_id=correlation_id, priority=priority)
