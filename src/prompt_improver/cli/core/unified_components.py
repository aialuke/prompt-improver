"""Unified CLI Components with Facade Pattern - Reduced Coupling Implementation

This is the modernized version of cli/core/__init__.py that uses facade patterns
to reduce coupling from 11 to 3 internal imports while maintaining full functionality.

Key improvements:
- 73% reduction in internal imports (11 → 3)
- Facade-based CLI component coordination
- Protocol-based interfaces for loose coupling
- Signal handling integration through facades
- Zero circular import possibilities
"""

import asyncio
import logging
from typing import Any, Optional

from rich.console import Console

from prompt_improver.core.facades import get_cli_facade
from prompt_improver.core.protocols.facade_protocols import CLIFacadeProtocol

logger = logging.getLogger(__name__)


class UnifiedCLIManager:
    """Unified CLI manager using facade pattern for loose coupling.
    
    This manager provides the same interface as the original CLI core module
    but with dramatically reduced coupling through facade patterns.
    
    Coupling reduction: 11 → 3 internal imports (73% reduction)
    """

    def __init__(self):
        """Initialize the unified CLI manager."""
        self._cli_facade: CLIFacadeProtocol = get_cli_facade()
        self._console = Console()
        self._initialized = False
        logger.debug("UnifiedCLIManager initialized with facade pattern")

    async def initialize(self) -> None:
        """Initialize all CLI components through facade."""
        if self._initialized:
            return
            
        await self._cli_facade.initialize_all()
        self._initialized = True
        logger.info("UnifiedCLIManager initialization complete")

    async def shutdown(self) -> None:
        """Shutdown all CLI components through facade."""
        if not self._initialized:
            return
            
        await self._cli_facade.shutdown_all()
        self._initialized = False
        logger.info("UnifiedCLIManager shutdown complete")

    def get_orchestrator(self) -> Any:
        """Get CLI orchestrator through facade."""
        return self._cli_facade.get_orchestrator()

    def get_workflow_service(self) -> Any:
        """Get workflow service through facade."""
        return self._cli_facade.get_workflow_service()

    def get_progress_service(self) -> Any:
        """Get progress service through facade."""
        return self._cli_facade.get_progress_service()

    def get_session_service(self) -> Any:
        """Get session service through facade."""
        return self._cli_facade.get_session_service()

    def get_training_service(self) -> Any:
        """Get training service through facade."""
        return self._cli_facade.get_training_service()

    def get_signal_handler(self) -> Any:
        """Get signal handler through facade."""
        return self._cli_facade.get_signal_handler()

    def get_background_manager(self) -> Any:
        """Get background manager through facade."""
        return self._cli_facade.get_background_manager()

    def get_emergency_service(self) -> Any:
        """Get emergency service through facade."""
        return self._cli_facade.get_emergency_service()

    def get_rule_validation_service(self) -> Any:
        """Get rule validation service through facade."""
        return self._cli_facade.get_rule_validation_service()

    def get_process_service(self) -> Any:
        """Get process service through facade."""
        return self._cli_facade.get_process_service()

    def get_system_state_reporter(self) -> Any:
        """Get system state reporter through facade."""
        return self._cli_facade.get_system_state_reporter()

    async def create_emergency_checkpoint(self) -> dict[str, Any]:
        """Create emergency checkpoint through facade."""
        return await self._cli_facade.create_emergency_checkpoint()

    def get_component_status(self) -> dict[str, Any]:
        """Get status of all CLI components through facade."""
        status = self._cli_facade.get_component_status()
        status.update({
            "manager_initialized": self._initialized,
            "facade_type": type(self._cli_facade).__name__,
        })
        return status

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on CLI manager through facade."""
        return await self._cli_facade.health_check()


class SignalAwareComponent:
    """Base class for CLI components that need signal handling integration.
    
    Uses facade pattern to reduce dependencies while maintaining signal handling.
    """

    def __init__(self):
        """Initialize signal-aware component with facade."""
        self._cli_manager = get_cli_manager()
        self._shutdown_priority = 10
        self._register_signal_handlers()

    def _register_signal_handlers(self):
        """Register component-specific signal handlers through facade."""
        signal_handler = self._cli_manager.get_signal_handler()
        signal_handler.register_shutdown_handler(
            f"{self.__class__.__name__}_shutdown", self.graceful_shutdown
        )
        
        # Register emergency checkpoint handler
        from prompt_improver.cli.core.signal_handler import SignalOperation
        signal_handler.register_operation_handler(
            SignalOperation.CHECKPOINT, self.create_emergency_checkpoint
        )
        
        # Register SIGTERM handler
        import signal
        signal_handler.add_signal_chain_handler(
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


# Global CLI manager instance
_cli_manager: UnifiedCLIManager | None = None


def get_cli_manager() -> UnifiedCLIManager:
    """Get the global unified CLI manager instance.

    Returns:
        UnifiedCLIManager: Global CLI manager with facade pattern
    """
    global _cli_manager
    if _cli_manager is None:
        _cli_manager = UnifiedCLIManager()
    return _cli_manager


async def initialize_cli_manager() -> None:
    """Initialize the global CLI manager."""
    manager = get_cli_manager()
    await manager.initialize()


async def shutdown_cli_manager() -> None:
    """Shutdown the global CLI manager."""
    global _cli_manager
    if _cli_manager:
        await _cli_manager.shutdown()
        _cli_manager = None


# Convenience functions with facade pattern
def get_shared_signal_handler() -> Any:
    """Get the global shared signal handler for CLI components."""
    return get_cli_manager().get_signal_handler()


def get_background_manager() -> Any:
    """Get the enhanced background task manager."""
    return get_cli_manager().get_background_manager()


def get_orchestrator() -> Any:
    """Get CLI orchestrator."""
    return get_cli_manager().get_orchestrator()


def get_workflow_service() -> Any:
    """Get workflow service."""
    return get_cli_manager().get_workflow_service()


def get_progress_service() -> Any:
    """Get progress service."""
    return get_cli_manager().get_progress_service()


def get_session_service() -> Any:
    """Get session service."""
    return get_cli_manager().get_session_service()


def get_training_service() -> Any:
    """Get training service."""
    return get_cli_manager().get_training_service()


def get_emergency_service() -> Any:
    """Get emergency service."""
    return get_cli_manager().get_emergency_service()


def get_rule_validation_service() -> Any:
    """Get rule validation service."""
    return get_cli_manager().get_rule_validation_service()


def get_process_service() -> Any:
    """Get process service."""
    return get_cli_manager().get_process_service()


def get_system_state_reporter() -> Any:
    """Get system state reporter.""" 
    return get_cli_manager().get_system_state_reporter()


async def create_emergency_checkpoint() -> dict[str, Any]:
    """Create emergency checkpoint."""
    return await get_cli_manager().create_emergency_checkpoint()


def get_component_status() -> dict[str, Any]:
    """Get status of all CLI components."""
    return get_cli_manager().get_component_status()


__all__ = [
    # Manager class
    "UnifiedCLIManager",
    "get_cli_manager",
    "initialize_cli_manager", 
    "shutdown_cli_manager",
    
    # Signal handling
    "SignalAwareComponent",
    
    # Convenience functions
    "get_shared_signal_handler",
    "get_background_manager",
    "get_orchestrator",
    "get_workflow_service",
    "get_progress_service", 
    "get_session_service",
    "get_training_service",
    "get_emergency_service",
    "get_rule_validation_service",
    "get_process_service",
    "get_system_state_reporter",
    "create_emergency_checkpoint",
    "get_component_status",
]