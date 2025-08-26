"""
Mock event bus service for testing.

This module contains event bus service implementations for testing,
extracted from conftest.py to maintain clean architecture.
"""
import asyncio
from typing import Any, Dict, List, Optional
from prompt_improver.utils.datetime_utils import aware_utc_now
from prompt_improver.shared.interfaces.protocols.ml import ServiceStatus


class MockEventBus:
    """Mock event bus for testing event-driven behavior."""
    
    def __init__(self):
        self._subscribers = {}
        self._published_events = []
        self._subscription_counter = 0
        self._is_healthy = True

    async def publish(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Publish an event to all subscribers."""
        event = {
            "type": event_type,
            "data": event_data,
            "timestamp": aware_utc_now().isoformat(),
            "delivered_to": [],
        }
        if event_type in self._subscribers:
            for sub_id, handler in self._subscribers[event_type].items():
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_data)
                    else:
                        handler(event_data)
                    event["delivered_to"].append(sub_id)
                except Exception as e:
                    event["delivery_errors"] = event.get("delivery_errors", [])
                    event["delivery_errors"].append({
                        "subscriber": sub_id,
                        "error": str(e),
                    })
        self._published_events.append(event)

    async def subscribe(self, event_type: str, handler: Any) -> str:
        """Subscribe to events of a specific type."""
        subscription_id = f"sub_{self._subscription_counter}_{event_type}"
        self._subscription_counter += 1
        if event_type not in self._subscribers:
            self._subscribers[event_type] = {}
        self._subscribers[event_type][subscription_id] = handler
        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from events."""
        for event_type, subscribers in self._subscribers.items():
            if subscription_id in subscribers:
                del subscribers[subscription_id]
                break

    async def health_check(self) -> ServiceStatus:
        """Check event bus health."""
        return ServiceStatus.HEALTHY if self._is_healthy else ServiceStatus.ERROR

    def set_health_status(self, healthy: bool):
        """Test helper to control health status."""
        self._is_healthy = healthy

    def get_published_events(self) -> List[Dict[str, Any]]:
        """Test helper to inspect published events."""
        return self._published_events.copy()

    def get_subscription_count(self, event_type: Optional[str] = None) -> int:
        """Test helper to count active subscriptions."""
        if event_type:
            return len(self._subscribers.get(event_type, {}))
        return sum(len(subs) for subs in self._subscribers.values())