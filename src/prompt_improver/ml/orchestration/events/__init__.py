"""Event system for ML pipeline communication."""

from .adaptive_event_bus import AdaptiveEventBus as EventBus  # Use AdaptiveEventBus as the default EventBus
from .event_types import EventType, MLEvent

__all__ = [
    "EventBus",
    "EventType", 
    "MLEvent"
]