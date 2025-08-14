"""CLI Components Facade - Reduces CLI Core Module Coupling

This facade provides unified CLI component coordination while reducing direct 
imports from 11 to 3 internal dependencies through lazy initialization.

Design:
- Protocol-based interface for loose coupling
- Lazy loading of CLI components
- Signal handling coordination
- Component lifecycle management  
- Zero circular import dependencies
"""

import logging
from typing import Any, Dict, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class CLIFacadeProtocol(Protocol):
    """Protocol for CLI components facade."""
    
    def get_orchestrator(self) -> Any:
        """Get CLI orchestrator."""
        ...
    
    def get_workflow_service(self) -> Any:
        """Get workflow service."""
        ...
    
    def get_progress_service(self) -> Any:
        """Get progress service."""
        ...
    
    def get_session_service(self) -> Any:
        """Get session service."""
        ...
    
    def get_training_service(self) -> Any:
        """Get training service."""
        ...
    
    def get_signal_handler(self) -> Any:
        """Get shared signal handler."""
        ...
    
    async def initialize_all(self) -> None:
        """Initialize all CLI components."""
        ...
    
    async def shutdown_all(self) -> None:
        """Shutdown all CLI components."""
        ...


class CLIFacade(CLIFacadeProtocol):
    """CLI components facade with minimal coupling.
    
    Reduces CLI core module coupling from 11 internal imports to 3.
    Provides unified interface for all CLI component coordination.
    """

    def __init__(self):
        """Initialize facade with lazy loading."""
        self._orchestrator = None
        self._workflow_service = None
        self._progress_service = None
        self._session_service = None
        self._training_service = None
        self._emergency_service = None
        self._rule_validation_service = None
        self._process_service = None
        self._signal_handler = None
        self._background_manager = None
        self._components_initialized = False
        logger.debug("CLIFacade initialized with lazy loading")

    def _ensure_orchestrator(self):
        """Ensure CLI orchestrator is available."""
        if self._orchestrator is None:
            # Only import when needed to reduce coupling
            from prompt_improver.cli.core.cli_orchestrator import CLIOrchestrator
            self._orchestrator = CLIOrchestrator()

    def _ensure_workflow_service(self):
        """Ensure workflow service is available."""
        if self._workflow_service is None:
            from prompt_improver.cli.core.enhanced_workflow_manager import WorkflowService
            self._workflow_service = WorkflowService()

    def _ensure_progress_service(self):
        """Ensure progress service is available."""
        if self._progress_service is None:
            from prompt_improver.cli.core.progress_preservation import ProgressService
            self._progress_service = ProgressService()

    def _ensure_session_service(self):
        """Ensure session service is available."""
        if self._session_service is None:
            from prompt_improver.cli.core.session_resume import SessionService
            self._session_service = SessionService()

    def _ensure_training_service(self):
        """Ensure training service is available."""
        if self._training_service is None:
            from prompt_improver.cli.services.training_orchestrator import TrainingOrchestrator
            self._training_service = TrainingOrchestrator()

    def _ensure_signal_handler(self):
        """Ensure signal handler is available."""
        if self._signal_handler is None:
            from prompt_improver.cli.core import get_shared_signal_handler
            self._signal_handler = get_shared_signal_handler()

    def _ensure_background_manager(self):
        """Ensure background manager is available."""
        if self._background_manager is None:
            from prompt_improver.cli.core import get_background_manager
            self._background_manager = get_background_manager()

    def get_orchestrator(self) -> Any:
        """Get CLI orchestrator."""
        self._ensure_orchestrator()
        return self._orchestrator

    def get_workflow_service(self) -> Any:
        """Get workflow service."""
        self._ensure_workflow_service()
        return self._workflow_service

    def get_progress_service(self) -> Any:
        """Get progress service."""
        self._ensure_progress_service()
        return self._progress_service

    def get_session_service(self) -> Any:
        """Get session service."""
        self._ensure_session_service()
        return self._session_service

    def get_training_service(self) -> Any:
        """Get training service."""
        self._ensure_training_service()
        return self._training_service

    def get_signal_handler(self) -> Any:
        """Get shared signal handler."""
        self._ensure_signal_handler()
        return self._signal_handler

    def get_background_manager(self) -> Any:
        """Get background task manager."""
        self._ensure_background_manager()
        return self._background_manager

    def get_emergency_service(self) -> Any:
        """Get emergency service."""
        if self._emergency_service is None:
            from prompt_improver.cli.core.emergency_operations import EmergencyService
            self._emergency_service = EmergencyService()
        return self._emergency_service

    def get_rule_validation_service(self) -> Any:
        """Get rule validation service."""
        if self._rule_validation_service is None:
            from prompt_improver.cli.core.rule_validation_service import RuleValidationService
            self._rule_validation_service = RuleValidationService()
        return self._rule_validation_service

    def get_process_service(self) -> Any:
        """Get process service."""
        if self._process_service is None:
            from prompt_improver.cli.core.unified_process_manager import ProcessService
            self._process_service = ProcessService()
        return self._process_service

    def get_system_state_reporter(self) -> Any:
        """Get system state reporter."""
        from prompt_improver.cli.core.system_state_reporter import SystemStateReporter
        return SystemStateReporter()

    async def initialize_all(self) -> None:
        """Initialize all CLI components."""
        if self._components_initialized:
            logger.warning("CLI components already initialized")
            return

        logger.info("Initializing CLI components...")

        # Initialize core components
        self._ensure_orchestrator()
        self._ensure_workflow_service()
        self._ensure_progress_service()
        self._ensure_session_service()
        self._ensure_training_service()
        self._ensure_signal_handler()
        self._ensure_background_manager()

        # Initialize components that have initialize methods
        components_to_init = []
        
        if hasattr(self._orchestrator, "initialize"):
            components_to_init.append(self._orchestrator.initialize())
        if hasattr(self._workflow_service, "initialize"):
            components_to_init.append(self._workflow_service.initialize())
        if hasattr(self._progress_service, "initialize"):
            components_to_init.append(self._progress_service.initialize())
        if hasattr(self._session_service, "initialize"):
            components_to_init.append(self._session_service.initialize())
        if hasattr(self._training_service, "initialize"):
            components_to_init.append(self._training_service.initialize())

        # Wait for all initializations
        if components_to_init:
            import asyncio
            await asyncio.gather(*components_to_init, return_exceptions=True)

        self._components_initialized = True
        logger.info("CLI components initialization complete")

    async def shutdown_all(self) -> None:
        """Shutdown all CLI components."""
        if not self._components_initialized:
            return

        logger.info("Shutting down CLI components...")

        # Shutdown components in reverse order
        components_to_shutdown = []
        
        if self._training_service and hasattr(self._training_service, "shutdown"):
            components_to_shutdown.append(self._training_service.shutdown())
        if self._session_service and hasattr(self._session_service, "shutdown"):
            components_to_shutdown.append(self._session_service.shutdown())
        if self._progress_service and hasattr(self._progress_service, "shutdown"):
            components_to_shutdown.append(self._progress_service.shutdown())
        if self._workflow_service and hasattr(self._workflow_service, "shutdown"):
            components_to_shutdown.append(self._workflow_service.shutdown())
        if self._orchestrator and hasattr(self._orchestrator, "shutdown"):
            components_to_shutdown.append(self._orchestrator.shutdown())

        # Wait for all shutdowns
        if components_to_shutdown:
            import asyncio
            await asyncio.gather(*components_to_shutdown, return_exceptions=True)

        # Clear references
        self._orchestrator = None
        self._workflow_service = None
        self._progress_service = None
        self._session_service = None
        self._training_service = None
        self._emergency_service = None
        self._rule_validation_service = None
        self._process_service = None
        self._signal_handler = None
        self._background_manager = None
        self._components_initialized = False

        logger.info("CLI components shutdown complete")

    def get_component_status(self) -> dict[str, Any]:
        """Get status of all CLI components."""
        return {
            "initialized": self._components_initialized,
            "orchestrator": self._orchestrator is not None,
            "workflow_service": self._workflow_service is not None,
            "progress_service": self._progress_service is not None,
            "session_service": self._session_service is not None,
            "training_service": self._training_service is not None,
            "signal_handler": self._signal_handler is not None,
            "background_manager": self._background_manager is not None,
        }

    async def create_emergency_checkpoint(self) -> dict[str, Any]:
        """Create emergency checkpoint across all components."""
        emergency_service = self.get_emergency_service()
        if hasattr(emergency_service, "create_checkpoint"):
            return await emergency_service.create_checkpoint()
        return {"status": "emergency_service_not_available"}


# Global facade instance
_cli_facade: CLIFacade | None = None


def get_cli_facade() -> CLIFacade:
    """Get global CLI facade instance.
    
    Returns:
        CLIFacade with lazy initialization and minimal coupling
    """
    global _cli_facade
    if _cli_facade is None:
        _cli_facade = CLIFacade()
    return _cli_facade


async def initialize_cli_facade() -> None:
    """Initialize the global CLI facade."""
    facade = get_cli_facade()
    await facade.initialize_all()


async def shutdown_cli_facade() -> None:
    """Shutdown the global CLI facade."""
    global _cli_facade
    if _cli_facade:
        await _cli_facade.shutdown_all()
        _cli_facade = None


__all__ = [
    "CLIFacadeProtocol",
    "CLIFacade",
    "get_cli_facade",
    "initialize_cli_facade", 
    "shutdown_cli_facade",
]