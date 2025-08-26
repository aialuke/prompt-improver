"""CLI Service Factory - ServiceRegistry Integration for Circular Import Resolution.

This factory implements the migration from lazy imports to ServiceRegistry pattern,
eliminating 60+ lazy import patterns in CLI components following 2025 best practices.

Key Features:
- ServiceRegistry integration with @service_provider decorators
- Eliminates circular import risks through centralized registration
- Maintains singleton pattern for performance consistency
- Zero backwards compatibility overhead through clean break strategy
"""

import logging
from typing import Any

from prompt_improver.core.services.service_registry import (
    ServiceScope,
    service_provider,
)

logger = logging.getLogger(__name__)


class CLIServiceFactory:
    """CLI service factory implementing ServiceRegistry pattern.

    Replaces 60+ lazy import workarounds with centralized service registration.
    Following factory patterns from RepositoryFactory and CacheFactory.

    Migration Pattern:
    - BEFORE: _ensure_orchestrator() -> manual lazy loading
    - AFTER: get_service("cli_orchestrator") -> ServiceRegistry resolution
    """


# First register the session manager service (required by most CLI services)
@service_provider("session_manager", ServiceScope.SINGLETON)
def create_session_manager() -> Any:
    """Create session manager service.

    Returns:
        DatabaseServices instance implementing SessionManagerProtocol
    """
    import asyncio

    from prompt_improver.database import get_sessionmanager

    try:
        loop = asyncio.get_event_loop()
        session_manager = loop.run_until_complete(get_sessionmanager())
    except RuntimeError:
        # Create new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        session_manager = loop.run_until_complete(get_sessionmanager())

    logger.debug("Session manager service created via ServiceRegistry")
    return session_manager


# CLI Core Services Registration with proper dependency injection
@service_provider("cli_orchestrator", ServiceScope.SINGLETON)
def create_cli_orchestrator() -> Any:
    """Create CLI orchestrator service.

    Returns:
        CLIOrchestrator instance with proper initialization
    """
    from prompt_improver.cli.core.cli_orchestrator import CLIOrchestrator
    orchestrator = CLIOrchestrator()
    logger.debug("CLI orchestrator service created via ServiceRegistry")
    return orchestrator


@service_provider("workflow_service", ServiceScope.SINGLETON, dependencies=["cli_orchestrator"])
def create_workflow_service(cli_orchestrator) -> Any:
    """Create workflow service with CLI orchestrator dependency.

    Args:
        cli_orchestrator: CLIOrchestrator dependency

    Returns:
        WorkflowService instance for enhanced workflow management
    """
    from prompt_improver.cli.core.enhanced_workflow_manager import WorkflowService
    service = WorkflowService(cli_orchestrator)
    logger.debug("Workflow service created via ServiceRegistry")
    return service


@service_provider("progress_service", ServiceScope.SINGLETON, dependencies=["session_manager"])
def create_progress_service(session_manager) -> Any:
    """Create progress preservation service with session manager dependency.

    Args:
        session_manager: SessionManagerProtocol dependency

    Returns:
        ProgressService instance for progress tracking
    """
    from prompt_improver.cli.core.progress_preservation import ProgressService
    service = ProgressService(session_manager)
    logger.debug("Progress service created via ServiceRegistry")
    return service


@service_provider("session_service", ServiceScope.SINGLETON)
def create_session_service() -> Any:
    """Create session resume service.

    NOTE: This service currently uses default constructor as the existing implementation
    does not properly handle session_manager injection. This is a migration compatibility
    fix - the service should be updated to accept session_manager properly in the future.

    Returns:
        SessionService instance for session management
    """
    from prompt_improver.cli.core.session_resume import SessionService
    service = SessionService()
    logger.debug("Session service created via ServiceRegistry (compatibility mode)")
    return service


@service_provider("training_service", ServiceScope.SINGLETON)
def create_training_service() -> Any:
    """Create training orchestrator service.

    Returns:
        TrainingOrchestrator instance for ML training coordination
    """
    from prompt_improver.cli.services.training_orchestrator import TrainingOrchestrator
    service = TrainingOrchestrator()
    logger.debug("Training service created via ServiceRegistry")
    return service


# CLI Support Services Registration with proper dependency injection
@service_provider("emergency_service", ServiceScope.SINGLETON)
def create_emergency_service() -> Any:
    """Create emergency operations service.

    NOTE: This service currently uses default constructor as the existing implementation
    does not properly handle session_manager injection. This is a migration compatibility
    fix - the service should be updated to accept session_manager properly in the future.

    Returns:
        EmergencyService instance for emergency operations
    """
    from prompt_improver.cli.core.emergency_operations import EmergencyService
    service = EmergencyService()
    logger.debug("Emergency service created via ServiceRegistry (compatibility mode)")
    return service


@service_provider("rule_validation_service", ServiceScope.SINGLETON, dependencies=["session_manager"])
def create_rule_validation_service(session_manager) -> Any:
    """Create rule validation service with session manager dependency.

    Args:
        session_manager: SessionManagerProtocol dependency

    Returns:
        RuleValidationService instance for rule validation
    """
    from prompt_improver.cli.core.rule_validation_service import RuleValidationService
    service = RuleValidationService(session_manager)
    logger.debug("Rule validation service created via ServiceRegistry")
    return service


@service_provider("process_service", ServiceScope.SINGLETON)
def create_process_service() -> Any:
    """Create unified process manager service.

    Returns:
        ProcessService instance for process management
    """
    from prompt_improver.cli.core.unified_process_manager import ProcessService
    service = ProcessService()
    logger.debug("Process service created via ServiceRegistry")
    return service


@service_provider("signal_handler", ServiceScope.SINGLETON)
def create_signal_handler() -> Any:
    """Create shared signal handler.

    Returns:
        Signal handler instance for graceful shutdown
    """
    from prompt_improver.cli.core import get_shared_signal_handler
    handler = get_shared_signal_handler()
    logger.debug("Signal handler created via ServiceRegistry")
    return handler


@service_provider("background_manager", ServiceScope.SINGLETON)
def create_background_manager() -> Any:
    """Create background task manager.

    Returns:
        Background manager instance for task coordination
    """
    from prompt_improver.cli.core import get_background_manager
    manager = get_background_manager()
    logger.debug("Background manager created via ServiceRegistry")
    return manager


@service_provider("system_state_reporter", ServiceScope.TRANSIENT)
def create_system_state_reporter() -> Any:
    """Create system state reporter.

    Note: TRANSIENT scope because each report should be fresh.

    Returns:
        SystemStateReporter instance for system reporting
    """
    from prompt_improver.cli.core.system_state_reporter import SystemStateReporter
    reporter = SystemStateReporter()
    logger.debug("System state reporter created via ServiceRegistry")
    return reporter


# Service Registration Helper
def register_all_cli_services() -> None:
    """Register all CLI services with ServiceRegistry.

    This function is called during startup to ensure all CLI services
    are registered with the ServiceRegistry before first use.

    Note: The @service_provider decorators automatically register services
    when this module is imported, but this function provides explicit
    registration confirmation and logging.
    """
    services_registered = [
        "session_manager",
        "cli_orchestrator",
        "workflow_service",
        "progress_service",
        "session_service",
        "training_service",
        "emergency_service",
        "rule_validation_service",
        "process_service",
        "signal_handler",
        "background_manager",
        "system_state_reporter",
    ]

    logger.info(f"CLI services registered with ServiceRegistry: {services_registered}")


# Convenience Functions for Validation
def validate_cli_services() -> dict[str, bool]:
    """Validate that all CLI services can be resolved.

    Returns:
        Dictionary mapping service names to resolution success status
    """
    from prompt_improver.core.services.service_registry import get_service

    services_to_validate = [
        "session_manager",
        "cli_orchestrator",
        "workflow_service",
        "progress_service",
        "session_service",
        "training_service",
        "emergency_service",
        "rule_validation_service",
        "process_service",
        "signal_handler",
        "background_manager",
        "system_state_reporter",
    ]

    validation_results = {}

    for service_name in services_to_validate:
        try:
            service = get_service(service_name)
            validation_results[service_name] = service is not None
            logger.debug(f"Service '{service_name}' resolution: SUCCESS")
        except Exception as e:
            validation_results[service_name] = False
            logger.exception(f"Service '{service_name}' resolution: FAILED - {e}")

    return validation_results


__all__ = [
    "CLIServiceFactory",
    "create_background_manager",
    # Service creation functions for direct access if needed
    "create_cli_orchestrator",
    "create_emergency_service",
    "create_process_service",
    "create_progress_service",
    "create_rule_validation_service",
    "create_session_service",
    "create_signal_handler",
    "create_system_state_reporter",
    "create_training_service",
    "create_workflow_service",
    "register_all_cli_services",
    "validate_cli_services",
]
