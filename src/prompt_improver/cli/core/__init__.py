"""CLI Core Module - Signal Handler Integration and Shared Components.

Provides shared signal handler integration for all CLI components.
Implements 2025 best practices for coordinated signal handling.
"""

import asyncio

from rich.console import Console

from prompt_improver.cli.core.cli_orchestrator import CLIOrchestrator
from prompt_improver.cli.core.emergency_operations import EmergencyService
from prompt_improver.cli.core.enhanced_workflow_manager import WorkflowService
from prompt_improver.cli.core.progress_preservation import ProgressService
from prompt_improver.cli.core.rule_validation_service import RuleValidationService
from prompt_improver.cli.core.session_resume import SessionService
from prompt_improver.cli.core.signal_handler import (
    AsyncSignalHandler,
    ShutdownContext,
    ShutdownReason,
)
from prompt_improver.cli.core.system_state_reporter import SystemStateReporter
from prompt_improver.cli.core.unified_process_manager import ProcessService
from prompt_improver.cli.services.training_orchestrator import (
    TrainingOrchestrator,
)
from prompt_improver.performance.monitoring.health.background_manager import (
    get_background_task_manager,
)

"\nCLI Core Components for APES Training System\n\nThis package contains the core components for the ultra-minimal 3-command CLI:\n- Training system management and lifecycle\n- CLI orchestration and workflow coordination  \n- Progress preservation and session management\n- Signal handling and graceful shutdown\n- Emergency operations and crash recovery\n- Process management and monitoring\n\nAll components are designed for clean break modernization with:\n- Zero legacy compatibility layers\n- Pure training focus without MCP dependencies\n- Optimal resource utilization\n- 2025 Python best practices\n"
_shared_signal_handler: AsyncSignalHandler | None = None


def get_shared_signal_handler() -> AsyncSignalHandler:
    """Get the global shared signal handler for CLI components.

    Returns:
        Shared AsyncSignalHandler instance configured for CLI usage
    """
    global _shared_signal_handler
    if _shared_signal_handler is None:
        _shared_signal_handler = AsyncSignalHandler(console=Console())
        try:
            loop = asyncio.get_running_loop()
            _shared_signal_handler.setup_signal_handlers(loop)
        except RuntimeError:
            pass
    return _shared_signal_handler


def get_background_manager():
    """Get the enhanced background task manager from Phase 1-2 consolidation.

    Returns:
        EnhancedBackgroundTaskManager instance
    """
    return get_background_task_manager()


class SignalAwareComponent:
    """Base class for CLI components that need signal handling integration."""

    def __init__(self) -> None:
        self.signal_handler = get_shared_signal_handler()
        self.background_manager = get_background_manager()
        self._shutdown_priority = 10
        self._register_signal_handlers()

    def _register_signal_handlers(self) -> None:
        """Register component-specific signal handlers."""
        self.signal_handler.register_shutdown_handler(
            f"{self.__class__.__name__}_shutdown", self.graceful_shutdown
        )
        from prompt_improver.cli.core.signal_handler import SignalOperation

        self.signal_handler.register_operation_handler(
            SignalOperation.CHECKPOINT, self.create_emergency_checkpoint
        )
        import signal

        self.signal_handler.add_signal_chain_handler(
            signal.SIGTERM, self.prepare_for_shutdown, priority=self._shutdown_priority
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
    "AsyncSignalHandler",
    "CLIOrchestrator",
    "EmergencyService",
    "ProcessService",
    "ProgressService",
    "RuleValidationService",
    "SessionService",
    "ShutdownContext",
    "ShutdownReason",
    "SignalAwareComponent",
    "SystemStateReporter",
    "TrainingOrchestrator",
    "WorkflowService",
    "get_background_manager",
    "get_shared_signal_handler",
]
