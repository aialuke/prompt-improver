"""A/B Testing Service Adapter

Adapter pattern implementation that wraps the concrete ModernABTestingService
to implement the IABTestingService interface. This provides clean separation
between the performance layer implementation and the core service interface.
"""

from datetime import datetime
from typing import Any, Dict

from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.performance.testing.ab_testing_service import (
    ModernABConfig,
    ModernABTestingService,
)
from prompt_improver.shared.interfaces.ab_testing import IABTestingService


class ModernABTestingServiceAdapter(IABTestingService):
    """Adapter that wraps ModernABTestingService to implement IABTestingService.

    This adapter provides clean separation between the performance layer
    implementation and the core service interface, following the Adapter pattern
    and enabling dependency injection without layering violations.
    """

    def __init__(self, config: ModernABConfig | None = None):
        """Initialize the adapter with underlying A/B testing service.

        Args:
            config: Optional configuration for the A/B testing service
        """
        self._service = ModernABTestingService(config)

    async def create_experiment(
        self,
        db_session: AsyncSession,
        name: str,
        description: str,
        hypothesis: str,
        control_rule_ids: list[str],
        treatment_rule_ids: list[str],
        success_metric: str = "conversion_rate",
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        """Create a new A/B experiment via the underlying service."""
        return await self._service.create_experiment(
            db_session=db_session,
            name=name,
            description=description,
            hypothesis=hypothesis,
            control_rule_ids=control_rule_ids,
            treatment_rule_ids=treatment_rule_ids,
            success_metric=success_metric,
            metadata=metadata,
        )

    async def analyze_experiment(
        self,
        db_session: AsyncSession,
        experiment_id: str,
        current_time: datetime | None = None,
    ) -> Dict[str, Any]:
        """Perform statistical analysis via the underlying service."""
        result = await self._service.analyze_experiment(
            db_session=db_session,
            experiment_id=experiment_id,
            current_time=current_time,
        )

        # Convert the StatisticalResult model to dictionary if needed
        if hasattr(result, "model_dump"):
            return result.model_dump()
        elif hasattr(result, "__dict__"):
            return result.__dict__
        else:
            return result

    async def check_early_stopping(
        self, experiment_id: str, look_number: int, db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Check early stopping via the underlying service."""
        return await self._service.check_early_stopping(
            experiment_id=experiment_id,
            look_number=look_number,
            db_session=db_session,
        )

    async def stop_experiment(
        self, experiment_id: str, reason: str, db_session: AsyncSession
    ) -> bool:
        """Stop experiment via the underlying service."""
        return await self._service.stop_experiment(
            experiment_id=experiment_id,
            reason=reason,
            db_session=db_session,
        )

    async def run_orchestrated_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run orchestrated analysis via the underlying service."""
        return await self._service.run_orchestrated_analysis(config)


def create_ab_testing_service_adapter(
    config: ModernABConfig | None = None,
) -> IABTestingService:
    """Factory function to create A/B testing service adapter.

    Args:
        config: Optional configuration for the A/B testing service

    Returns:
        IABTestingService implementation
    """
    return ModernABTestingServiceAdapter(config)
