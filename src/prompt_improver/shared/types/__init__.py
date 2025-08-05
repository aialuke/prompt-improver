"""Shared types and data structures"""

from .signals import SignalContext, SignalOperation, ShutdownReason, EmergencyOperation

__all__ = [
    "SignalContext",
    "SignalOperation", 
    "ShutdownReason",
    "EmergencyOperation"
]