"""Training System Facade - Backwards Compatibility Layer.

Provides backwards compatibility interface while using decomposed training services internally.
Maintains the original TrainingService interface for existing code during transition.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from prompt_improver.cli.services import create_training_system
from prompt_improver.cli.services.training_protocols import (
    TrainingMetricsProtocol,
    TrainingOrchestratorProtocol,
    TrainingPersistenceProtocol,
    TrainingValidatorProtocol,
)


class TrainingServiceFacade:
    """Facade providing backwards compatibility for the original TrainingService interface.

    Uses the decomposed training services internally while maintaining the original API.
    This facilitates gradual migration from the monolithic TrainingService to the
    decomposed architecture.

    Note: This is a transitional facade. New code should use the decomposed services directly.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.logger = logging.getLogger("apes.training_system_facade")

        # Create the decomposed training system internally
        self._orchestrator: TrainingOrchestratorProtocol = create_training_system(console)

        # Extract individual services for direct access (if available via orchestrator)
        self._validator: TrainingValidatorProtocol = getattr(self._orchestrator, 'validator', None)
        self._metrics: TrainingMetricsProtocol = getattr(self._orchestrator, 'metrics', None)
        self._persistence: TrainingPersistenceProtocol = getattr(self._orchestrator, 'persistence', None)

        # Create services directly if not available via orchestrator
        if not self._validator:
            from prompt_improver.cli.services import create_training_validator
            self._validator = create_training_validator()
        if not self._metrics:
            from prompt_improver.cli.services import create_training_metrics
            self._metrics = create_training_metrics()
        if not self._persistence:
            from prompt_improver.cli.services import create_training_persistence
            self._persistence = create_training_persistence()

        # Training system data directory (for compatibility)
        self.training_data_dir = Path.home() / ".local" / "share" / "apes" / "training"
        self.training_data_dir.mkdir(parents=True, exist_ok=True)

    # Main orchestration methods - delegate to orchestrator
    async def start_training_system(self) -> dict[str, Any]:
        """Start training system components - delegates to orchestrator.

        Returns:
            Training system startup results with performance metrics
        """
        return await self._orchestrator.start_training_system()

    async def stop_training_system(self, graceful: bool = True) -> bool:
        """Stop training system gracefully - delegates to orchestrator.

        Args:
            graceful: Whether to perform graceful shutdown

        Returns:
            True if shutdown successful, False otherwise
        """
        return await self._orchestrator.stop_training_system(graceful)

    async def get_training_status(self) -> dict[str, Any]:
        """Get training system status - delegates to orchestrator.

        Returns:
            Training system status and metrics
        """
        return await self._orchestrator.get_training_status()

    # Validation methods - delegate to validator
    async def validate_ready_for_training(self) -> bool:
        """Validate that the system is ready for training.

        Returns:
            True if ready for training, False otherwise
        """
        return await self._validator.validate_ready_for_training()

    async def smart_initialize(self) -> dict[str, Any]:
        """Enhanced smart initialization with comprehensive system state detection.

        Note: This method combines multiple validation and initialization steps
        from the decomposed services.

        Returns:
            Detailed initialization results with system state analysis
        """
        try:
            # Phase 1: System State Detection
            system_state = await self._validator.detect_system_state()

            # Phase 2: Component Validation
            component_status = await self._validator.validate_components(
                orchestrator=self._orchestrator.orchestrator,
                analytics=getattr(self._orchestrator, '_analytics', None),
                data_generator=getattr(self._orchestrator, '_data_generator', None),
            )

            # Phase 3: Database and Rule Validation
            database_status = await self._validator.validate_database_and_rules()

            # Phase 4: Data Availability Assessment
            data_status = await self._validator.assess_data_availability()

            # Phase 5: Create and Execute Initialization Plan
            if all([
                database_status.get("connectivity", {}).get("status") == "healthy",
                data_status.get("minimum_requirements", {}).get("status") == "met"
            ]):
                # System is ready - start training system
                execution_results = await self.start_training_system()
            else:
                # System needs initialization
                execution_results = {
                    "success": False,
                    "message": "System not ready for training",
                    "components_initialized": [],
                }

            # Phase 6: Final Validation
            final_validation = {
                "overall_health": execution_results.get("status") == "success",
                "training_readiness": await self._validator.validate_ready_for_training(),
            }

            return {
                "success": execution_results.get("status") == "success",
                "message": execution_results.get("message", "Initialization completed"),
                "system_state": system_state,
                "component_status": component_status,
                "database_status": database_status,
                "data_status": data_status,
                "execution_results": execution_results,
                "final_validation": final_validation,
                "components_initialized": execution_results.get("components_initialized", []),
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            self.logger.exception(f"Smart initialization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "components_initialized": [],
                "timestamp": datetime.now(UTC).isoformat(),
            }

    # Persistence methods - delegate to persistence service
    async def create_training_session(self, training_config: dict[str, Any]) -> str:
        """Create training session using persistence service.

        Args:
            training_config: Training configuration

        Returns:
            Created training session ID
        """
        return await self._persistence.create_training_session(training_config)

    async def update_training_progress(
        self,
        iteration: int,
        performance_metrics: dict[str, float],
        improvement_score: float = 0.0,
    ) -> bool:
        """Update training progress using persistence service.

        Args:
            iteration: Current iteration
            performance_metrics: Performance metrics
            improvement_score: Improvement score

        Returns:
            True if updated successfully
        """
        session_id = self._orchestrator.training_session_id
        if not session_id:
            self.logger.warning("No active training session for progress update")
            return False

        return await self._persistence.update_training_progress(
            session_id, iteration, performance_metrics, improvement_score
        )

    async def get_training_session_context(self) -> dict[str, Any] | None:
        """Get current training session context from persistence service.

        Returns:
            Training session context if available
        """
        session_id = self._orchestrator.training_session_id
        if not session_id:
            return None

        return await self._persistence.get_training_session_context(session_id)

    async def get_active_sessions(self) -> list:
        """Get all active training sessions.

        Returns:
            List of active training session objects
        """
        return await self._persistence.get_active_sessions()

    # Metrics methods - delegate to metrics service
    async def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status including health and active sessions.

        Returns:
            System status dictionary
        """
        try:
            # Get orchestrator status
            training_status = await self._orchestrator.get_training_status()

            # Get active sessions
            active_sessions = await self._persistence.get_active_sessions()

            # Get performance summary
            performance_summary = await self._metrics.get_performance_summary()

            # Check component health
            components = {
                "training_system": self._orchestrator.training_status,
                "database": "healthy",  # Assume healthy if we got this far
            }

            # Overall health
            healthy = (
                training_status.get("training_system_status") == "running" and
                len([s for s in active_sessions if s.get("status") in {"running", "paused"}]) >= 0
            )

            return {
                "healthy": healthy,
                "status": "healthy" if healthy else "degraded",
                "active_sessions": [
                    {
                        "session_id": session.get("session_id"),
                        "status": session.get("status"),
                        "started_at": session.get("created_at"),
                        "iterations": session.get("current_iteration", 0),
                        "current_performance": session.get("performance_metrics", {}),
                    }
                    for session in active_sessions
                ],
                "components": components,
                "recent_performance": performance_summary,
                "resource_usage": await self._metrics.get_resource_usage(),
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            self.logger.exception(f"Failed to get system status: {e}")
            return {
                "healthy": False,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    # Direct access to decomposed services for advanced use cases
    @property
    def orchestrator(self) -> TrainingOrchestratorProtocol:
        """Access to the orchestrator service."""
        return self._orchestrator

    @property
    def validator(self) -> TrainingValidatorProtocol:
        """Access to the validator service."""
        return self._validator

    @property
    def metrics(self) -> TrainingMetricsProtocol:
        """Access to the metrics service."""
        return self._metrics

    @property
    def persistence(self) -> TrainingPersistenceProtocol:
        """Access to the persistence service."""
        return self._persistence

    # Legacy properties for backwards compatibility
    @property
    def training_status(self) -> str:
        """Get current training status."""
        return self._orchestrator.training_status

    @property
    def training_session_id(self) -> str | None:
        """Get current training session ID."""
        return self._orchestrator.training_session_id

    # Signal handling methods - delegate to orchestrator
    async def graceful_shutdown_handler(self, shutdown_context):
        """Handle graceful shutdown - delegates to orchestrator."""
        return await self._orchestrator.graceful_shutdown_handler(shutdown_context)

    async def create_training_emergency_checkpoint(self, signal_context):
        """Create emergency training checkpoint - delegates to orchestrator."""
        return await self._orchestrator.create_training_emergency_checkpoint(signal_context)

    async def generate_training_status_report(self, signal_context):
        """Generate training status report - delegates to orchestrator."""
        return await self._orchestrator.generate_training_status_report(signal_context)

    def prepare_training_shutdown(self, signum, signal_name):
        """Prepare training system for shutdown - delegates to orchestrator."""
        return self._orchestrator.prepare_training_shutdown(signum, signal_name)

    def prepare_training_interruption(self, signum, signal_name):
        """Prepare training system for interruption - delegates to orchestrator."""
        return self._orchestrator.prepare_training_interruption(signum, signal_name)


# Legacy alias for backwards compatibility
TrainingService = TrainingServiceFacade
