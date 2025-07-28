"""Shared types and data structures"""

from .signals import SignalContext, SignalOperation, ShutdownReason, EmergencyOperation
from .events import Event, EventType

__all__ = [
    "SignalContext",
    "SignalOperation", 
    "ShutdownReason",
    "EmergencyOperation",
    "Event",
    "EventType"
]