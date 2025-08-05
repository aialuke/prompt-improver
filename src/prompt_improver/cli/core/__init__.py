"""CLI Core Module - Signal Handler Integration and Shared Components

Provides shared signal handler integration for all CLI components.
Implements 2025 best practices for coordinated signal handling.
"""

import asyncio
from typing import Optional
from rich.console import Console

"""
CLI Core Components for APES Training System

This package contains the core components for the ultra-minimal 3-command CLI:
- Training system management and lifecycle
- CLI orchestration and workflow coordination  
- Progress preservation and session management
- Signal handling and graceful shutdown
- Emergency operations and crash recovery
- Process management and monitoring

All components are designed for clean break modernization with:
- Zero legacy compatibility layers
- Pure training focus without MCP dependencies
- Optimal resource utilization
- 2025 Python best practices
"""

# Core training system components
from .training_system_manager import TrainingSystemManager
from .cli_orchestrator import CLIOrchestrator
from .progress_preservation import ProgressPreservationManager

# Signal handling and emergency operations
from .signal_handler import AsyncSignalHandler, ShutdownContext, ShutdownReason
from .emergency_operations import EmergencyOperationsManager

# Process and session management
from .unified_process_manager import UnifiedProcessManager
from .session_resume import SessionResumeManager

# System monitoring and reporting
from .system_state_reporter import SystemStateReporter
from .enhanced_workflow_manager import EnhancedWorkflowManager

# Validation services
from .rule_validation_service import RuleValidationService

# Signal handler integration
from ...performance.monitoring.health.background_manager import get_background_task_manager

# Global signal handler instance for CLI coordination
_shared_signal_handler: Optional[AsyncSignalHandler] = None

def get_shared_signal_handler() -> AsyncSignalHandler:
    """Get the global shared signal handler for CLI components.
    
    Returns:
        Shared AsyncSignalHandler instance configured for CLI usage
    """
    global _shared_signal_handler
    if _shared_signal_handler is None:
        _shared_signal_handler = AsyncSignalHandler(console=Console())
        
        # Setup signal handlers with asyncio loop if available
        try:
            loop = asyncio.get_running_loop()
            _shared_signal_handler.setup_signal_handlers(loop)
        except RuntimeError:
            # No running loop - will be set up when loop starts
            pass
    
    return _shared_signal_handler

def get_background_manager():
    """Get the enhanced background task manager from Phase 1-2 consolidation.
    
    Returns:
        EnhancedBackgroundTaskManager instance
    """
    return get_background_task_manager()

# Standard CLI Component Signal Integration Pattern
class SignalAwareComponent:
    """Base class for CLI components that need signal handling integration."""
    
    def __init__(self):
        self.signal_handler = get_shared_signal_handler()
        self.background_manager = get_background_manager()
        self._shutdown_priority = 10  # Default shutdown priority
        self._register_signal_handlers()
    
    def _register_signal_handlers(self):
        """Register component-specific signal handlers."""
        # Shutdown coordination
        self.signal_handler.register_shutdown_handler(
            f"{self.__class__.__name__}_shutdown", 
            self.graceful_shutdown
        )
        
        # Emergency operations (SIGUSR1 for checkpoints)
        from .signal_handler import SignalOperation
        self.signal_handler.register_operation_handler(
            SignalOperation.CHECKPOINT,
            self.create_emergency_checkpoint
        )
        
        # Signal chaining for coordinated operations
        import signal
        self.signal_handler.add_signal_chain_handler(
            signal.SIGTERM,
            self.prepare_for_shutdown,
            priority=self._shutdown_priority
        )
    
    async def graceful_shutdown(self, shutdown_context):
        """Override in subclasses for component-specific shutdown."""
        return {"status": "success", "component": self.__class__.__name__}
    
    async def create_emergency_checkpoint(self, signal_context):
        """Override in subclasses for emergency checkpoint creation."""
        return {"status": "checkpoint_created", "component": self.__class__.__name__}
    
    def prepare_for_shutdown(self, signum, signal_name):
        """Override in subclasses for shutdown preparation."""
        return {"prepared": True, "component": self.__class__.__name__}

__all__ = [
    # Core training components
    "TrainingSystemManager",
    "CLIOrchestrator", 
    "ProgressPreservationManager",
    
    # Signal handling
    "AsyncSignalHandler",
    "ShutdownContext",
    "ShutdownReason",
    
    # Emergency operations
    "EmergencyOperationsManager",
    
    # Process management
    "UnifiedProcessManager",
    "SessionResumeManager",
    
    # System monitoring
    "SystemStateReporter",
    "EnhancedWorkflowManager",
    
    # Validation
    "RuleValidationService",
    
    # Signal integration
    "get_shared_signal_handler",
    "get_background_manager",
    "SignalAwareComponent",
]
