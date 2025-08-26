"""A/B Testing Service Interface.

This module provides abstractions for A/B testing functionality to maintain
clean architecture boundaries and enable dependency injection.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class IABTestingService(ABC):
    """Abstract interface for A/B testing services.

    This interface provides clean abstraction for A/B testing functionality,
    allowing different implementations while maintaining consistent API.
    Following SOLID principles and dependency inversion principle.
    """

    @abstractmethod
    async def create_experiment(
        self,
        db_session: "AsyncSession",
        name: str,
        description: str,
        hypothesis: str,
        control_rule_ids: list[str],
        treatment_rule_ids: list[str],
        success_metric: str = "conversion_rate",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new A/B experiment.

        Args:
            db_session: Database session
            name: Experiment name
            description: Detailed description
            hypothesis: Testable hypothesis
            control_rule_ids: Control group rule IDs
            treatment_rule_ids: Treatment group rule IDs
            success_metric: Primary success metric
            metadata: Additional experiment metadata

        Returns:
            Experiment ID
        """

    @abstractmethod
    async def analyze_experiment(
        self,
        db_session: "AsyncSession",
        experiment_id: str,
        current_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Perform statistical analysis on experiment.

        Args:
            db_session: Database session
            experiment_id: Experiment identifier
            current_time: Analysis timestamp

        Returns:
            Statistical analysis results
        """

    @abstractmethod
    async def check_early_stopping(
        self, experiment_id: str, look_number: int, db_session: "AsyncSession"
    ) -> dict[str, Any]:
        """Check if experiment should be stopped early.

        Args:
            experiment_id: Experiment identifier
            look_number: Sequential look number
            db_session: Database session

        Returns:
            Early stopping decision and reasoning
        """

    @abstractmethod
    async def stop_experiment(
        self, experiment_id: str, reason: str, db_session: "AsyncSession"
    ) -> bool:
        """Stop an experiment with given reason.

        Args:
            experiment_id: Experiment identifier
            reason: Reason for stopping
            db_session: Database session

        Returns:
            Success status
        """

    @abstractmethod
    async def run_orchestrated_analysis(self, config: dict[str, Any]) -> dict[str, Any]:
        """Run orchestrator-compatible A/B testing analysis.

        Args:
            config: Analysis configuration

        Returns:
            Orchestrator-compatible results
        """


class NoOpABTestingService(IABTestingService):
    """No-operation implementation of A/B testing service.

    Provides safe fallback when A/B testing is disabled or unavailable.
    Returns appropriate default responses without performing actual testing.
    """

    async def create_experiment(
        self,
        db_session: "AsyncSession",
        name: str,
        description: str,
        hypothesis: str,
        control_rule_ids: list[str],
        treatment_rule_ids: list[str],
        success_metric: str = "conversion_rate",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create no-op experiment."""
        return "noop-experiment-id"

    async def analyze_experiment(
        self,
        db_session: "AsyncSession",
        experiment_id: str,
        current_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Return no-op analysis results."""
        return {
            "status": "disabled",
            "message": "A/B testing is disabled",
            "experiment_id": experiment_id,
        }

    async def check_early_stopping(
        self, experiment_id: str, look_number: int, db_session: "AsyncSession"
    ) -> dict[str, Any]:
        """Return no-op early stopping check."""
        return {
            "should_stop": False,
            "reason": "ab_testing_disabled",
            "statistical_power": 0.0,
        }

    async def stop_experiment(
        self, experiment_id: str, reason: str, db_session: "AsyncSession"
    ) -> bool:
        """No-op experiment stop."""
        return True

    async def run_orchestrated_analysis(self, config: dict[str, Any]) -> dict[str, Any]:
        """Return no-op orchestrated analysis."""
        return {
            "orchestrator_compatible": True,
            "component_result": {
                "ab_testing_summary": {"status": "disabled"},
                "experiment_data": {"status": "disabled"},
            },
            "local_metadata": {
                "component_version": "noop",
                "framework": "NoOpABTestingService",
            },
        }
