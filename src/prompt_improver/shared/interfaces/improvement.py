"""Interfaces for prompt improvement services following 2025 best practices"""

from typing import Any, Dict, List, Protocol


class IImprovementService(Protocol):
    """Interface for prompt improvement services

    Following clean architecture principles where business logic
    depends on abstractions, not concrete implementations.
    """

    async def improve_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        """Improve a prompt using available strategies

        Args:
            prompt: Original prompt text
            context: Additional context for improvement (domain, user preferences, etc.)

        Returns:
            Improved prompt text
        """
        ...

    async def get_improvement_suggestions(self, prompt: str) -> list[str]:
        """Get improvement suggestions for a prompt

        Args:
            prompt: Prompt to analyze

        Returns:
            List of specific improvement suggestions
        """
        ...

    async def validate_improvement(
        self, original: str, improved: str
    ) -> dict[str, Any]:
        """Validate that an improvement is actually better

        Args:
            original: Original prompt
            improved: Improved prompt

        Returns:
            Validation results with scores and metrics
        """
        ...


class IMLService(Protocol):
    """Interface for ML services supporting prompt improvement

    Abstracts away specific ML framework implementations to enable
    easy testing and framework switching.
    """

    async def predict_improvement(self, prompt: str) -> dict[str, Any]:
        """Predict improvement metrics for a prompt

        Args:
            prompt: Prompt to analyze

        Returns:
            Prediction results with confidence scores
        """
        ...

    async def train_model(self, training_data: list[dict[str, Any]]) -> bool:
        """Train the improvement model with new data

        Args:
            training_data: List of training examples

        Returns:
            Success status
        """
        ...

    async def get_model_metrics(self) -> dict[str, float]:
        """Get current model performance metrics

        Returns:
            Dictionary of metric names and values
        """
        ...


class IRuleEngine(Protocol):
    """Interface for rule-based improvement engines"""

    async def apply_rules(self, prompt: str, rules: list[str]) -> str:
        """Apply specified rules to improve a prompt

        Args:
            prompt: Prompt to improve
            rules: List of rule names to apply

        Returns:
            Improved prompt
        """
        ...

    async def get_available_rules(self) -> list[str]:
        """Get list of available improvement rules

        Returns:
            List of rule names
        """
        ...
