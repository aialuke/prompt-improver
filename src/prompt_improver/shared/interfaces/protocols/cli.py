"""CLI Service Protocols (2025) - Consolidated Domain-Modular Location.

Comprehensive protocol definitions for all CLI services following clean architecture
principles. These protocols define the contracts for CLI components without
implementation details, enabling proper dependency injection and eliminating
circular imports.

Consolidated from:
- /core/protocols/cli_protocols.py (comprehensive service protocols)
- Original target stubs (CLI-specific command and interaction protocols)

Organized by functional domains:
- Base protocols for all CLI services
- Command processing and user interaction
- Core service management
- System and background task management
- Rule validation and selection
- Workflow orchestration
"""

from abc import abstractmethod
from typing import Any, Protocol, Union, runtime_checkable

# ============================================================================
# BASE CLI SERVICE PROTOCOL
# ============================================================================


@runtime_checkable
class CLIServiceProtocol(Protocol):
    """Base protocol for all CLI services."""

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """Check service health.

        Returns:
            Health status dictionary
        """
        ...


# ============================================================================
# COMMAND PROCESSING AND USER INTERACTION PROTOCOLS
# ============================================================================

@runtime_checkable
class CommandProcessorProtocol(Protocol):
    """Protocol for command processing services."""

    @abstractmethod
    async def execute_command(self, command: str, args: list[str] | None = None) -> dict[str, Any]:
        """Execute a CLI command with given arguments."""
        ...

    @abstractmethod
    def get_available_commands(self) -> list[str]:
        """Get list of available commands."""
        ...


@runtime_checkable
class UserInteractionProtocol(Protocol):
    """Protocol for user interaction services."""

    @abstractmethod
    async def prompt_user(self, message: str, options: list[str] | None = None) -> str:
        """Prompt user for input with optional choices."""
        ...

    @abstractmethod
    async def display_message(self, message: str, level: str = "info") -> None:
        """Display message to user with specified level."""
        ...


# ============================================================================
# CORE SERVICE MANAGEMENT PROTOCOLS
# ============================================================================

@runtime_checkable
class CLIOrchestratorProtocol(CLIServiceProtocol, Protocol):
    """Protocol for CLI orchestrator service."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the orchestrator."""
        ...

    @abstractmethod
    async def run_command(self, command: str, args: dict[str, Any]) -> Any:
        """Run a CLI command.

        Args:
            command: Command to execute
            args: Command arguments

        Returns:
            Command result
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the orchestrator."""
        ...


@runtime_checkable
class WorkflowServiceProtocol(CLIServiceProtocol, Protocol):
    """Protocol for workflow management service."""

    @abstractmethod
    async def create_workflow(self, workflow_type: str, config: dict[str, Any]) -> str:
        """Create a new workflow.

        Args:
            workflow_type: Type of workflow
            config: Workflow configuration

        Returns:
            Workflow ID
        """
        ...

    @abstractmethod
    async def execute_workflow(self, workflow_id: str) -> Any:
        """Execute a workflow.

        Args:
            workflow_id: ID of workflow to execute

        Returns:
            Workflow result
        """
        ...


@runtime_checkable
class WorkflowManagerProtocol(Protocol):
    """Protocol for CLI workflow management (alternative interface)."""

    @abstractmethod
    async def execute_workflow(self, workflow_name: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute a named workflow with provided context."""
        ...

    @abstractmethod
    def get_workflow_status(self, workflow_id: str) -> dict[str, Any]:
        """Get status of a running workflow."""
        ...


@runtime_checkable
class ProgressServiceProtocol(CLIServiceProtocol, Protocol):
    """Protocol for progress preservation service."""

    @abstractmethod
    async def save_progress(self, session_id: str, progress_data: dict[str, Any]) -> None:
        """Save progress data.

        Args:
            session_id: Session identifier
            progress_data: Progress information to save
        """
        ...

    @abstractmethod
    async def restore_progress(self, session_id: str) -> dict[str, Any] | None:
        """Restore progress data.

        Args:
            session_id: Session identifier

        Returns:
            Restored progress data or None
        """
        ...


@runtime_checkable
class SessionServiceProtocol(CLIServiceProtocol, Protocol):
    """Protocol for session management service."""

    @abstractmethod
    async def create_session(self) -> str:
        """Create a new session.

        Returns:
            Session ID
        """
        ...

    @abstractmethod
    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get session data.

        Args:
            session_id: Session identifier

        Returns:
            Session data or None
        """
        ...

    @abstractmethod
    async def end_session(self, session_id: str) -> None:
        """End a session.

        Args:
            session_id: Session identifier
        """
        ...


@runtime_checkable
class TrainingServiceProtocol(CLIServiceProtocol, Protocol):
    """Protocol for training orchestrator service."""

    @abstractmethod
    async def start_training(self, config: dict[str, Any]) -> str:
        """Start a training job.

        Args:
            config: Training configuration

        Returns:
            Training job ID
        """
        ...

    @abstractmethod
    async def get_training_status(self, job_id: str) -> dict[str, Any]:
        """Get training job status.

        Args:
            job_id: Training job identifier

        Returns:
            Training status information
        """
        ...

    @abstractmethod
    async def stop_training(self, job_id: str) -> None:
        """Stop a training job.

        Args:
            job_id: Training job identifier
        """
        ...


# ============================================================================
# SYSTEM AND BACKGROUND TASK MANAGEMENT PROTOCOLS
# ============================================================================

@runtime_checkable
class SignalHandlerProtocol(CLIServiceProtocol, Protocol):
    """Protocol for signal handling service."""

    @abstractmethod
    def register_handler(self, signal_type: str, handler: Any) -> None:
        """Register a signal handler.

        Args:
            signal_type: Type of signal
            handler: Handler function
        """
        ...

    @abstractmethod
    def handle_signal(self, signal_type: str) -> None:
        """Handle a signal.

        Args:
            signal_type: Type of signal received
        """
        ...


@runtime_checkable
class BackgroundManagerProtocol(CLIServiceProtocol, Protocol):
    """Protocol for background task management."""

    @abstractmethod
    async def create_task(self, task_func: Any, *args, **kwargs) -> str:
        """Create a background task.

        Args:
            task_func: Function to run in background
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Task ID
        """
        ...

    @abstractmethod
    async def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get task status.

        Args:
            task_id: Task identifier

        Returns:
            Task status information
        """
        ...

    @abstractmethod
    async def cancel_task(self, task_id: str) -> None:
        """Cancel a background task.

        Args:
            task_id: Task identifier
        """
        ...


@runtime_checkable
class EmergencyServiceProtocol(CLIServiceProtocol, Protocol):
    """Protocol for emergency operations service."""

    @abstractmethod
    async def create_checkpoint(self) -> dict[str, Any]:
        """Create emergency checkpoint.

        Returns:
            Checkpoint information
        """
        ...

    @abstractmethod
    async def restore_checkpoint(self, checkpoint_id: str) -> None:
        """Restore from checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier
        """
        ...


@runtime_checkable
class ProcessServiceProtocol(CLIServiceProtocol, Protocol):
    """Protocol for process management service."""

    @abstractmethod
    async def start_process(self, command: str, args: list[str]) -> str:
        """Start a new process.

        Args:
            command: Command to execute
            args: Command arguments

        Returns:
            Process ID
        """
        ...

    @abstractmethod
    async def get_process_status(self, process_id: str) -> dict[str, Any]:
        """Get process status.

        Args:
            process_id: Process identifier

        Returns:
            Process status information
        """
        ...


@runtime_checkable
class SystemStateReporterProtocol(CLIServiceProtocol, Protocol):
    """Protocol for system state reporting."""

    @abstractmethod
    async def get_system_state(self) -> dict[str, Any]:
        """Get current system state.

        Returns:
            System state information
        """
        ...

    @abstractmethod
    async def generate_report(self, report_type: str) -> str:
        """Generate a system report.

        Args:
            report_type: Type of report to generate

        Returns:
            Report content
        """
        ...


# ============================================================================
# RULE VALIDATION AND SELECTION PROTOCOLS
# ============================================================================

@runtime_checkable
class RuleValidationServiceProtocol(CLIServiceProtocol, Protocol):
    """Protocol for rule validation service."""

    @abstractmethod
    async def validate_rules(self, rules: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate a set of rules.

        Args:
            rules: Rules to validate

        Returns:
            Validation results
        """
        ...


@runtime_checkable
class RuleSelectionProtocol(Protocol):
    """Protocol for rule selection services."""

    @abstractmethod
    async def select_rules(self, context: dict[str, Any]) -> list[str]:
        """Select appropriate rules based on context."""
        ...

    @abstractmethod
    def get_rule_metadata(self, rule_id: str) -> dict[str, Any]:
        """Get metadata for a specific rule."""
        ...


# ============================================================================
# CONFIGURATION FACADE PROTOCOLS
# ============================================================================

@runtime_checkable
class BaseFacadeProtocol(Protocol):
    """Base protocol for all facades with common lifecycle methods."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the facade and its components."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the facade and cleanup resources."""
        ...

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """Get current status of the facade."""
        ...

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Perform health check on the facade."""
        ...


@runtime_checkable
class ConfigurationAccessProtocol(Protocol):
    """Protocol for configuration access patterns."""

    @abstractmethod
    def get_config(self) -> Any:
        """Get main application configuration."""
        ...

    @abstractmethod
    def get_database_config(self) -> Any:
        """Get database configuration."""
        ...

    @abstractmethod
    def get_security_config(self) -> Any:
        """Get security configuration."""
        ...

    @abstractmethod
    def get_monitoring_config(self) -> Any:
        """Get monitoring configuration."""
        ...

    @abstractmethod
    def get_ml_config(self) -> Any:
        """Get ML configuration."""
        ...

    @abstractmethod
    def create_test_config(self, overrides: dict[str, Any] | None = None) -> Any:
        """Create test configuration."""
        ...

    @abstractmethod
    def validate_configuration(self) -> dict[str, Any]:
        """Validate current configuration."""
        ...

    @abstractmethod
    def get_service_config(self, service_name: str) -> dict[str, Any]:
        """Get configuration for a specific service."""
        ...

    @abstractmethod
    def setup_configuration_logging(self) -> None:
        """Setup configuration logging."""
        ...

    @abstractmethod
    def is_development(self) -> bool:
        """Check if running in development environment."""
        ...

    @abstractmethod
    def is_production(self) -> bool:
        """Check if running in production environment."""
        ...

    @abstractmethod
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        ...


@runtime_checkable
class ConfigFacadeProtocol(BaseFacadeProtocol, ConfigurationAccessProtocol, Protocol):
    """Complete protocol for configuration facade."""

    @abstractmethod
    def reload_config(self) -> None:
        """Reload configuration from sources."""
        ...

    @abstractmethod
    def get_environment_name(self) -> str:
        """Get current environment name."""
        ...


# ============================================================================
# CLI FACADE PROTOCOL
# ============================================================================

@runtime_checkable
class CLIFacadeProtocol(Protocol):
    """Protocol for CLI components facade."""

    @abstractmethod
    def get_orchestrator(self) -> Any:
        """Get CLI orchestrator."""
        ...

    @abstractmethod
    def get_workflow_service(self) -> Any:
        """Get workflow service."""
        ...

    @abstractmethod
    def get_progress_service(self) -> Any:
        """Get progress service."""
        ...

    @abstractmethod
    def get_session_service(self) -> Any:
        """Get session service."""
        ...

    @abstractmethod
    def get_training_service(self) -> Any:
        """Get training service."""
        ...

    @abstractmethod
    def get_signal_handler(self) -> Any:
        """Get shared signal handler."""
        ...

    @abstractmethod
    async def initialize_all(self) -> None:
        """Initialize all CLI components."""
        ...

    @abstractmethod
    async def shutdown_all(self) -> None:
        """Shutdown all CLI components."""
        ...


# ============================================================================
# TYPE UNIONS AND EXPORTS
# ============================================================================

# Union type for any CLI service protocol
CLIService = Union[
    CLIOrchestratorProtocol,
    WorkflowServiceProtocol,
    WorkflowManagerProtocol,
    ProgressServiceProtocol,
    SessionServiceProtocol,
    TrainingServiceProtocol,
    SignalHandlerProtocol,
    BackgroundManagerProtocol,
    EmergencyServiceProtocol,
    RuleValidationServiceProtocol,
    ProcessServiceProtocol,
    SystemStateReporterProtocol,
    CommandProcessorProtocol,
    UserInteractionProtocol,
    RuleSelectionProtocol,
]


# Comprehensive exports for all CLI protocols
__all__ = [
    "BackgroundManagerProtocol",
    # Configuration facade protocols
    "BaseFacadeProtocol",
    # CLI facade protocols
    "CLIFacadeProtocol",
    # Core service management
    "CLIOrchestratorProtocol",
    # Type unions
    "CLIService",
    # Base protocol
    "CLIServiceProtocol",
    # Command processing and user interaction
    "CommandProcessorProtocol",
    "ConfigFacadeProtocol",
    "ConfigurationAccessProtocol",
    "EmergencyServiceProtocol",
    "ProcessServiceProtocol",
    "ProgressServiceProtocol",
    "RuleSelectionProtocol",
    # Rule processing
    "RuleValidationServiceProtocol",
    "SessionServiceProtocol",
    # System and background management
    "SignalHandlerProtocol",
    "SystemStateReporterProtocol",
    "TrainingServiceProtocol",
    "UserInteractionProtocol",
    "WorkflowManagerProtocol",
    "WorkflowServiceProtocol",
]
