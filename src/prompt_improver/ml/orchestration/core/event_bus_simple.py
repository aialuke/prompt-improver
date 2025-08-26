"""
Simple event bus implementation for testing.
"""
import asyncio
from typing import Any, Callable, Dict, List
from prompt_improver.shared.interfaces.protocols.ml import EventBusProtocol


class SimpleEventBus:
    """Simple in-memory event bus for testing."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the event bus."""
        self._initialized = True
    
    async def shutdown(self):
        """Shutdown the event bus."""
        self._subscribers.clear()
        self._initialized = False
    
    async def emit(self, event_name: str, data: Any = None):
        """Emit an event to all subscribers."""
        if not self._initialized:
            return
            
        subscribers = self._subscribers.get(event_name, [])
        for subscriber in subscribers:
            try:
                if callable(subscriber):
                    await subscriber(data) if asyncio.iscoroutinefunction(subscriber) else subscriber(data)
            except Exception:
                pass  # Ignore subscriber errors in tests
    
    def subscribe(self, event_name: str, callback: Callable):
        """Subscribe to an event."""
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(callback)
    
    def unsubscribe(self, event_name: str, callback: Callable):
        """Unsubscribe from an event."""
        if event_name in self._subscribers:
            try:
                self._subscribers[event_name].remove(callback)
            except ValueError:
                pass