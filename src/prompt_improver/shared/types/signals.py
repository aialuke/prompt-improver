"""Shared signal handling types to break circular dependencies
Extracted from CLI modules following 2025 clean architecture patterns
"""
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, Optional

class ShutdownReason(Enum):
    """Enumeration of shutdown reasons for tracking and reporting"""
    USER_INTERRUPT = 'user_interrupt'
    SYSTEM_SHUTDOWN = 'system_shutdown'
    TIMEOUT = 'timeout'
    ERROR = 'error'
    FORCE = 'force'
    GRACEFUL = 'graceful'

class SignalOperation(Enum):
    """Enumeration of signal-triggered operations"""
    CHECKPOINT = 'checkpoint'
    STATUS_REPORT = 'status_report'
    CONFIG_RELOAD = 'config_reload'
    SHUTDOWN = 'shutdown'
    EMERGENCY_SAVE = 'emergency_save'

class EmergencyOperation(Enum):
    """Types of emergency operations that can be triggered"""
    BACKUP_STATE = 'backup_state'
    SAVE_PROGRESS = 'save_progress'
    EXPORT_LOGS = 'export_logs'
    HEALTH_CHECK = 'health_check'
    RESOURCE_CLEANUP = 'resource_cleanup'

@dataclass
class SignalContext:
    """Context information for signal operations

    Provides all necessary context for handling signal-triggered
    operations without circular dependencies.
    """
    operation: SignalOperation
    timestamp: datetime
    signal_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    force_flag: bool = False

    @classmethod
    def create_shutdown_context(cls, reason: ShutdownReason, signal_num: int | None=None) -> 'SignalContext':
        """Create context for shutdown operations

        Args:
            reason: Reason for shutdown
            signal_num: Signal number that triggered shutdown

        Returns:
            Configured signal context
        """
        return cls(operation=SignalOperation.SHUTDOWN, timestamp=datetime.now(UTC), signal_number=signal_num, metadata={'shutdown_reason': reason.value}, timeout_seconds=30)

    @classmethod
    def create_emergency_context(cls, emergency_op: EmergencyOperation) -> 'SignalContext':
        """Create context for emergency operations

        Args:
            emergency_op: Type of emergency operation

        Returns:
            Configured signal context for emergency operation
        """
        return cls(operation=SignalOperation.EMERGENCY_SAVE, timestamp=datetime.now(UTC), metadata={'emergency_operation': emergency_op.value}, timeout_seconds=60, force_flag=True)

@dataclass
class OperationResult:
    """Result from a signal-triggered operation"""
    success: bool
    operation: SignalOperation
    duration_seconds: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(cls, operation: SignalOperation, duration: float, message: str='', details: dict[str, Any] | None=None) -> 'OperationResult':
        """Create successful operation result"""
        return cls(success=True, operation=operation, duration_seconds=duration, message=message, details=details or {})

    @classmethod
    def error_result(cls, operation: SignalOperation, duration: float, error_message: str, details: dict[str, Any] | None=None) -> 'OperationResult':
        """Create error operation result"""
        return cls(success=False, operation=operation, duration_seconds=duration, message=error_message, details=details or {})
