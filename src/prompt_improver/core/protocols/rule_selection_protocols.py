"""Rule Selection Service Protocols

Protocol interfaces for rule selection services following Clean Architecture principles.
These protocols define the contracts between layers without implementation details.
"""

from typing import Any, Protocol, runtime_checkable

from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.rule_engine.base import BasePromptRule


@runtime_checkable
class RuleSelectionProtocol(Protocol):
    """Protocol for rule selection service operations."""
    
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
        """Get optimal rules using multi-armed bandit or traditional selection."""
        ...
    
    async def get_rules_metadata(
        self, enabled_only: bool = True, db_session: AsyncSession | None = None
    ) -> list[dict[str, Any]]:
        """Get rules metadata."""
        ...
    
    def get_bandit_experiment_id(self) -> str | None:
        """Get current bandit experiment ID."""
        ...
    
    def set_bandit_experiment_id(self, experiment_id: str) -> None:
        """Set bandit experiment ID."""
        ...


@runtime_checkable  
class RuleCacheProtocol(Protocol):
    """Protocol for rule caching operations."""
    
    def get_cached_rules(self, cache_key: str) -> tuple[dict[str, BasePromptRule], float] | None:
        """Get cached rules if available and not expired."""
        ...
    
    def cache_rules(self, cache_key: str, rules: dict[str, BasePromptRule]) -> None:
        """Cache rules with timestamp."""
        ...
    
    def clear_cache(self) -> None:
        """Clear all cached rules."""
        ...


@runtime_checkable
class RuleLoaderProtocol(Protocol):
    """Protocol for rule loading operations."""
    
    async def load_rules_from_database(
        self, db_session: AsyncSession
    ) -> dict[str, BasePromptRule]:
        """Load and instantiate rules from database configuration."""
        ...
    
    def get_fallback_rules(self) -> dict[str, BasePromptRule]:
        """Get fallback rules when database loading fails."""
        ...
    
    def get_rule_class(self, rule_id: str) -> type[BasePromptRule] | None:
        """Map rule_id to actual rule class."""
        ...


@runtime_checkable
class BanditOptimizationProtocol(Protocol):
    """Protocol for bandit optimization operations."""
    
    async def initialize_bandit_experiment(
        self, db_session: AsyncSession | None = None
    ) -> str | None:
        """Initialize bandit experiment via event bus communication."""
        ...
    
    async def get_bandit_optimized_rules(
        self,
        prompt_characteristics: dict[str, Any],
        preferred_rules: list[str] | None = None,
        db_session: AsyncSession | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get rules using event-based bandit optimization."""
        ...
    
    def prepare_bandit_context(
        self, prompt_characteristics: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare context for contextual bandit from prompt characteristics."""
        ...