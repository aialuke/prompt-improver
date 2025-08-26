"""Training Service Protocols - Clean Architecture Implementation.

Protocol definitions for training system services implementing dependency inversion.
Supports Clean Architecture patterns with protocol-based dependency injection.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol


class TrainingValidatorProtocol(Protocol):
    """Protocol for training validation services."""

    @abstractmethod
    async def validate_ready_for_training(self) -> bool:
        """Validate that the system is ready for training.

        Returns:
            True if ready for training, False otherwise
        """
        ...

    @abstractmethod
    async def validate_database_and_rules(self) -> dict[str, Any]:
        """Validate database connectivity, schema, and seeded rules.

        Returns:
            Database and rule validation results
        """
        ...

    @abstractmethod
    async def assess_data_availability(self) -> dict[str, Any]:
        """Comprehensive training data availability assessment.

        Returns:
            Data availability analysis with quality metrics
        """
        ...

    @abstractmethod
    async def detect_system_state(self) -> dict[str, Any]:
        """Comprehensive system state detection.

        Returns:
            Detailed system state information
        """
        ...

    @abstractmethod
    async def validate_components(self, orchestrator=None, analytics=None, data_generator=None) -> dict[str, Any]:
        """Validate all training system components with health checks.

        Args:
            orchestrator: ML pipeline orchestrator instance
            analytics: Analytics service instance
            data_generator: Data generator instance

        Returns:
            Component validation results
        """
        ...


class TrainingMetricsProtocol(Protocol):
    """Protocol for training metrics services."""

    @abstractmethod
    async def get_resource_usage(self) -> dict[str, float]:
        """Get current system resource usage metrics.

        Returns:
            Resource usage metrics including memory, CPU, and I/O
        """
        ...

    @abstractmethod
    async def get_detailed_training_metrics(self) -> dict[str, Any]:
        """Get comprehensive training performance metrics.

        Returns:
            Detailed training metrics including performance trends and analysis
        """
        ...

    @abstractmethod
    async def get_current_performance_metrics(self) -> dict[str, Any]:
        """Get current performance metrics for checkpoint creation.

        Returns:
            Current performance state for checkpointing
        """
        ...

    @abstractmethod
    async def get_performance_summary(self) -> dict[str, Any]:
        """Get summarized performance metrics for status reporting.

        Returns:
            Performance summary with key metrics and insights
        """
        ...

    @abstractmethod
    async def record_training_iteration(
        self,
        iteration: int,
        metrics: dict[str, float],
        duration_seconds: float
    ) -> None:
        """Record metrics for a training iteration.

        Args:
            iteration: Training iteration number
            metrics: Performance metrics for this iteration
            duration_seconds: Time taken for iteration
        """
        ...


class TrainingPersistenceProtocol(Protocol):
    """Protocol for training persistence services."""

    @abstractmethod
    async def create_training_session(self, training_config: dict[str, Any]) -> str:
        """Create training session with configuration.

        Args:
            training_config: Training configuration dictionary

        Returns:
            Created training session ID
        """
        ...

    @abstractmethod
    async def update_training_progress(
        self,
        session_id: str,
        iteration: int,
        performance_metrics: dict[str, float],
        improvement_score: float = 0.0,
    ) -> bool:
        """Update training progress for a session.

        Args:
            session_id: Training session ID
            iteration: Current iteration number
            performance_metrics: Performance metrics dictionary
            improvement_score: Improvement score for this iteration

        Returns:
            True if updated successfully, False otherwise
        """
        ...

    @abstractmethod
    async def get_training_session_context(self, session_id: str) -> dict[str, Any] | None:
        """Get training session context from database.

        Args:
            session_id: Training session ID

        Returns:
            Training session context if available, None otherwise
        """
        ...

    @abstractmethod
    async def save_training_progress(self, session_id: str) -> bool:
        """Save current training progress to database.

        Args:
            session_id: Training session ID to save progress for

        Returns:
            True if progress saved successfully, False otherwise
        """
        ...

    @abstractmethod
    async def create_checkpoint(self, session_id: str) -> str:
        """Create emergency training checkpoint.

        Args:
            session_id: Training session ID to create checkpoint for

        Returns:
            Checkpoint ID if created successfully
        """
        ...

    @abstractmethod
    async def restore_from_checkpoint(self, checkpoint_id: str) -> dict[str, Any] | None:
        """Restore training session from checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to restore from

        Returns:
            Restored session context if successful, None otherwise
        """
        ...

    @abstractmethod
    async def get_active_sessions(self) -> list[dict[str, Any]]:
        """Get all active training sessions.

        Returns:
            List of active training session dictionaries
        """
        ...

    @abstractmethod
    async def terminate_session(self, session_id: str, reason: str = "manual_termination") -> bool:
        """Terminate a training session.

        Args:
            session_id: Session ID to terminate
            reason: Reason for termination

        Returns:
            True if terminated successfully, False otherwise
        """
        ...


class TrainingOrchestratorProtocol(Protocol):
    """Protocol for training orchestrator services."""

    @abstractmethod
    async def start_training_system(self) -> dict[str, Any]:
        """Start training system components.

        Returns:
            Training system startup results with performance metrics
        """
        ...

    @abstractmethod
    async def stop_training_system(self, graceful: bool = True) -> bool:
        """Stop training system gracefully.

        Args:
            graceful: Whether to perform graceful shutdown

        Returns:
            True if shutdown successful, False otherwise
        """
        ...

    @abstractmethod
    async def get_training_status(self) -> dict[str, Any]:
        """Get training system status.

        Returns:
            Training system status and metrics
        """
        ...

    @property
    @abstractmethod
    def training_status(self) -> str:
        """Get current training status."""
        ...

    @property
    @abstractmethod
    def training_session_id(self) -> str | None:
        """Get current training session ID."""
        ...


class SmartInitializationProtocol(Protocol):
    """Protocol for smart initialization services."""

    @abstractmethod
    async def smart_initialize(self) -> dict[str, Any]:
        """Enhanced smart initialization with comprehensive system state detection.

        Returns:
            Detailed initialization results with system state analysis
        """
        ...

    @abstractmethod
    async def create_initialization_plan(
        self,
        system_state: dict[str, Any],
        component_status: dict[str, Any],
        database_status: dict[str, Any],
        data_status: dict[str, Any],
    ) -> dict[str, Any]:
        """Create intelligent initialization plan based on system analysis.

        Returns:
            Detailed initialization plan with prioritized actions
        """
        ...

    @abstractmethod
    async def execute_initialization_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Execute the initialization plan with progress tracking.

        Returns:
            Execution results with component status
        """
        ...

    @abstractmethod
    async def validate_post_initialization(self) -> dict[str, Any]:
        """Validate system state after initialization.

        Returns:
            Post-initialization validation results
        """
        ...


class DataGenerationProtocol(Protocol):
    """Protocol for synthetic data generation services."""

    @abstractmethod
    async def generate_initial_data(self, data_status: dict[str, Any] | None = None) -> dict[str, Any]:
        """Enhanced synthetic data generation with smart initialization.

        Args:
            data_status: Optional data status from assessment for targeted generation

        Returns:
            Generation results with metrics and recommendations
        """
        ...

    @abstractmethod
    async def determine_generation_strategy(self, data_status: dict[str, Any] | None) -> dict[str, Any]:
        """Determine optimal generation strategy based on data gaps.

        Returns:
            Generation strategy with method, parameters, and targeting
        """
        ...

    @abstractmethod
    async def validate_generated_data_quality(self, generation_data: dict[str, Any]) -> dict[str, Any]:
        """Validate the quality of newly generated data.

        Args:
            generation_data: Generated data to validate

        Returns:
            Quality assessment results
        """
        ...


# Abstract base classes for concrete implementations
class TrainingServiceBase(ABC):
    """Abstract base class for training services providing common functionality."""

    def __init__(self) -> None:
        self.logger = None
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the service gracefully."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized


class TrainingComponentBase(TrainingServiceBase):
    """Abstract base class for training system components."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self._status = "not_initialized"

    @property
    def status(self) -> str:
        """Get component status."""
        return self._status

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on component.

        Returns:
            Health check results
        """
        return {
            "component": self.name,
            "status": self._status,
            "initialized": self._initialized,
            "healthy": self._status == "running" and self._initialized,
        }


# Factory protocol for creating training services
class TrainingServiceFactoryProtocol(Protocol):
    """Protocol for training service factories."""

    def create_validator(self) -> TrainingValidatorProtocol:
        """Create training validator service."""
        ...

    def create_metrics(self) -> TrainingMetricsProtocol:
        """Create training metrics service."""
        ...

    def create_persistence(self) -> TrainingPersistenceProtocol:
        """Create training persistence service."""
        ...

    def create_orchestrator(
        self,
        validator: TrainingValidatorProtocol | None = None,
        metrics: TrainingMetricsProtocol | None = None,
        persistence: TrainingPersistenceProtocol | None = None,
    ) -> TrainingOrchestratorProtocol:
        """Create training orchestrator service with dependencies."""
        ...


# Configuration protocols
class TrainingConfigProtocol(Protocol):
    """Protocol for training configuration."""

    @property
    def max_iterations(self) -> int | None:
        """Maximum number of training iterations."""
        ...

    @property
    def improvement_threshold(self) -> float:
        """Improvement threshold for training continuation."""
        ...

    @property
    def timeout_seconds(self) -> int:
        """Training timeout in seconds."""
        ...

    @property
    def continuous_mode(self) -> bool:
        """Whether to run in continuous mode."""
        ...

    @property
    def auto_init_enabled(self) -> bool:
        """Whether automatic initialization is enabled."""
        ...


# Event protocols for training system events
class TrainingEventProtocol(Protocol):
    """Protocol for training system events."""

    @property
    def event_type(self) -> str:
        """Type of training event."""
        ...

    @property
    def timestamp(self) -> float:
        """Event timestamp."""
        ...

    @property
    def session_id(self) -> str:
        """Associated training session ID."""
        ...

    @property
    def data(self) -> dict[str, Any]:
        """Event data payload."""
        ...


class TrainingEventHandlerProtocol(Protocol):
    """Protocol for training event handlers."""

    @abstractmethod
    async def handle_event(self, event: TrainingEventProtocol) -> None:
        """Handle a training system event.

        Args:
            event: Training event to handle
        """
        ...
