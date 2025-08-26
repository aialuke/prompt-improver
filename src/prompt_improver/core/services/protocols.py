"""Service protocols for prompt improvement system decomposition.

These protocols define clean interfaces for the decomposed services,
following the existing patterns established in shared/interfaces.
"""

from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.rule_engine.base import BasePromptRule


class IRuleSelectionService(Protocol):
    """Interface for rule discovery and selection operations."""

    async def get_active_rules(
        self, db_session: AsyncSession | None = None
    ) -> dict[str, BasePromptRule]:
        """Get active rules with caching."""
        ...

    async def get_optimal_rules(
        self,
        prompt_characteristics: dict[str, Any],
        preferred_rules: list[str] | None = None,
        db_session: AsyncSession | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get optimal rules using configured strategy (bandit or traditional)."""
        ...

    async def get_rules_metadata(
        self, enabled_only: bool = True, db_session: AsyncSession | None = None
    ) -> list[dict[str, Any]]:
        """Get rules metadata."""
        ...


class IPersistenceService(Protocol):
    """Interface for database persistence operations."""

    async def store_session(
        self,
        session_id: str,
        original_prompt: str,
        final_prompt: str,
        rules_applied: list[dict[str, Any]],
        user_context: dict[str, Any] | None,
        db_session: AsyncSession,
    ) -> None:
        """Store improvement session in database."""
        ...

    async def store_performance_metrics(
        self, performance_data: list[dict[str, Any]], db_session: AsyncSession
    ) -> None:
        """Store rule performance metrics."""
        ...

    async def store_user_feedback(
        self,
        session_id: str,
        rating: int,
        feedback_text: str | None,
        improvement_areas: list[str] | None,
        db_session: AsyncSession,
    ) -> Any:  # Returns UserFeedback model
        """Store user feedback."""
        ...


class IMLEventingService(Protocol):
    """Interface for ML event handling and coordination."""

    async def initialize_automl(self) -> None:
        """Initialize AutoML via event bus communication."""
        ...

    async def initialize_bandit_experiment(
        self, db_session: AsyncSession | None = None
    ) -> str | None:
        """Initialize bandit experiment and return experiment ID."""
        ...

    async def update_bandit_rewards(
        self,
        applied_rules: list[dict[str, Any]],
        experiment_id: str | None = None,
    ) -> None:
        """Update bandit with rewards via event bus communication."""
        ...

    async def trigger_optimization(
        self, feedback_id: int, db_session: AsyncSession
    ) -> dict[str, Any]:
        """Trigger ML optimization based on feedback."""
        ...

    async def start_automl_optimization(
        self,
        optimization_target: str = "rule_effectiveness",
        experiment_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start AutoML optimization."""
        ...


class IAnalyticsService(Protocol):
    """Interface for analytics and metrics operations."""

    async def analyze_prompt(self, prompt: str) -> dict[str, Any]:
        """Analyze prompt characteristics for rule selection."""
        ...

    def calculate_metrics(self, prompt: str) -> dict[str, float]:
        """Calculate metrics for a prompt."""
        ...

    def calculate_improvement_score(
        self, before: dict[str, float], after: dict[str, float]
    ) -> float:
        """Calculate improvement score based on metrics."""
        ...

    def generate_improvement_summary(
        self, applied_rules: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate a summary of improvements made."""
        ...

    def calculate_overall_confidence(
        self, applied_rules: list[dict[str, Any]]
    ) -> float:
        """Calculate overall confidence score."""
        ...

    def prepare_bandit_context(
        self, prompt_characteristics: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare context for contextual bandit from prompt characteristics."""
        ...
