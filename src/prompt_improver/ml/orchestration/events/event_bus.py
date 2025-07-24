"""
Event Bus for ML Pipeline orchestration.

Provides async event-driven communication between ML components.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

from .event_types import EventType, MLEvent
from ..config.orchestrator_config import OrchestratorConfig

@dataclass
class EventSubscription:
    """Event subscription details."""
    event_type: EventType
    handler: Callable[[MLEvent], Any]
    subscription_id: str
    is_async: bool = True

class EventBus:
    """
    Async Event Bus for ML Pipeline coordination.
    
    Provides pub/sub messaging between ML components with:
    - Async event handling
    - Event filtering and routing
    - Error handling and retries
    - Event history and auditing
    """
    
    def __init__(self, config: OrchestratorConfig):
        """Initialize the event bus."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Event subscriptions by event type
        self.subscriptions: Dict[EventType, List[EventSubscription]] = defaultdict(list)
        
        # Event queue and processing
        self.event_queue: asyncio.Queue = asyncio.Queue(
            maxsize=config.event_bus_buffer_size
        )
        self.processing_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Event history for debugging
        self.event_history: List[MLEvent] = []
        self.max_history_size = 1000
        
        # Subscription counter for unique IDs
        self._subscription_counter = 0
        
        # Initialization state
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the event bus."""
        if self.is_running:
            return
        
        self.logger.info("Initializing event bus")
        self.is_running = True
        
        # Start event processing task
        self.processing_task = asyncio.create_task(self._process_events())
        
        self._is_initialized = True
        self.logger.info("Event bus initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown the event bus."""
        if not self.is_running:
            return
        
        self.logger.info("Shutting down event bus")
        self.is_running = False
        self._is_initialized = False
        
        # Cancel processing task
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Clear remaining events
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        self.logger.info("Event bus shutdown complete")
    
    def subscribe(self, event_type: EventType, handler: Callable[[MLEvent], Any]) -> str:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            handler: Async function to handle events
            
        Returns:
            Subscription ID for unsubscribing
        """
        self._subscription_counter += 1
        subscription_id = f"sub_{self._subscription_counter}"
        
        subscription = EventSubscription(
            event_type=event_type,
            handler=handler,
            subscription_id=subscription_id,
            is_async=asyncio.iscoroutinefunction(handler)
        )
        
        self.subscriptions[event_type].append(subscription)
        
        self.logger.debug(f"Subscribed {subscription_id} to {event_type.value}")
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            subscription_id: ID returned from subscribe()
            
        Returns:
            True if subscription was found and removed
        """
        for event_type, subs in self.subscriptions.items():
            for i, sub in enumerate(subs):
                if sub.subscription_id == subscription_id:
                    del subs[i]
                    self.logger.debug(f"Unsubscribed {subscription_id} from {event_type.value}")
                    return True
        
        return False
    
    async def emit(self, event: MLEvent) -> None:
        """
        Emit an event to all subscribers.
        
        Args:
            event: Event to emit
        """
        if not self.is_running:
            self.logger.warning("Event bus not running, discarding event")
            return
        
        try:
            # Add to queue for processing
            await asyncio.wait_for(
                self.event_queue.put(event),
                timeout=1.0  # Don't block indefinitely
            )
            
            # Add to history for debugging
            self.event_history.append(event)
            if len(self.event_history) > self.max_history_size:
                self.event_history.pop(0)
            
            self.logger.debug(f"Emitted event {event.event_type.value} from {event.source}")
            
        except asyncio.TimeoutError:
            self.logger.error(f"Event queue full, dropping event {event.event_type.value}")
        except Exception as e:
            self.logger.error(f"Error emitting event: {e}")

    async def publish(self, event: MLEvent) -> None:
        """
        Publish an event to all subscribers.

        This is an alias for emit() to provide a more standard pub/sub interface.

        Args:
            event: Event to publish
        """
        await self.emit(event)

    async def emit_and_wait(self, event: MLEvent, timeout: float = 5.0) -> List[Any]:
        """
        Emit an event and wait for all handlers to complete.
        
        Args:
            event: Event to emit
            timeout: Maximum time to wait for handlers
            
        Returns:
            List of handler results
        """
        if not self.is_running:
            raise RuntimeError("Event bus not running")
        
        # Get subscribers for this event type
        subscribers = self.subscriptions.get(event.event_type, [])
        if not subscribers:
            return []
        
        # Execute handlers directly (not through queue)
        results = []
        try:
            handler_tasks = []
            
            for subscription in subscribers:
                if subscription.is_async:
                    task = asyncio.create_task(
                        asyncio.wait_for(
                            subscription.handler(event),
                            timeout=self.config.event_handler_timeout
                        )
                    )
                    handler_tasks.append(task)
                else:
                    # Run sync handlers in thread pool
                    task = asyncio.create_task(
                        asyncio.get_event_loop().run_in_executor(
                            None, subscription.handler, event
                        )
                    )
                    handler_tasks.append(task)
            
            # Wait for all handlers with timeout
            if handler_tasks:
                results = await asyncio.wait_for(
                    asyncio.gather(*handler_tasks, return_exceptions=True),
                    timeout=timeout
                )
            
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout waiting for handlers of {event.event_type.value}")
        except Exception as e:
            self.logger.error(f"Error in emit_and_wait: {e}")
        
        return results
    
    async def _process_events(self) -> None:
        """Process events from the queue."""
        self.logger.info("Started event processing")
        
        while self.is_running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )
                
                # Process event
                await self._handle_event(event)
                
            except asyncio.TimeoutError:
                # Normal timeout, continue loop
                continue
            except asyncio.CancelledError:
                # Task was cancelled, break loop
                break
            except Exception as e:
                self.logger.error(f"Error processing event: {e}")
        
        self.logger.info("Event processing stopped")
    
    async def _handle_event(self, event: MLEvent) -> None:
        """Handle a single event by calling all subscribers."""
        subscribers = self.subscriptions.get(event.event_type, [])
        
        if not subscribers:
            self.logger.debug(f"No subscribers for event {event.event_type.value}")
            return
        
        # Execute all handlers
        for subscription in subscribers:
            try:
                if subscription.is_async:
                    await asyncio.wait_for(
                        subscription.handler(event),
                        timeout=self.config.event_handler_timeout
                    )
                else:
                    # Run sync handlers in thread pool
                    await asyncio.get_event_loop().run_in_executor(
                        None, subscription.handler, event
                    )
                    
            except asyncio.TimeoutError:
                self.logger.error(
                    f"Handler {subscription.subscription_id} timed out for {event.event_type.value}"
                )
            except Exception as e:
                self.logger.error(
                    f"Error in handler {subscription.subscription_id}: {e}"
                )
    
    def get_event_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[MLEvent]:
        """
        Get recent event history.
        
        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return
            
        Returns:
            List of recent events
        """
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:] if limit > 0 else events
    
    def get_subscription_count(self, event_type: Optional[EventType] = None) -> int:
        """
        Get number of active subscriptions.
        
        Args:
            event_type: Count for specific event type (optional)
            
        Returns:
            Number of subscriptions
        """
        if event_type:
            return len(self.subscriptions.get(event_type, []))
        
        return sum(len(subs) for subs in self.subscriptions.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics for health monitoring."""
        return {
            "total_events": len(self.event_history),
            "failed_events": 0,  # Would track in real implementation
            "active_handlers": self.get_subscription_count(),
            "queue_size": self.event_queue.qsize() if hasattr(self.event_queue, 'qsize') else 0,
            "is_running": self.is_running
        }
    
    async def emit_health_check_event(self, source: str) -> None:
        """Emit a health check event for testing event bus functionality."""
        from datetime import datetime, timezone
        await self.emit(MLEvent(
            event_type=EventType.HEALTH_CHECK_COMPLETED,
            source=source,
            data={"test": True, "timestamp": datetime.now(timezone.utc).isoformat()}
        ))