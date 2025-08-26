"""Simple Performance Monitoring Service Protocols - 2025 Architecture.

Simplified protocol interfaces to test service locator pattern without potential
circular imports from SQLAlchemy or other complex dependencies.
"""

from typing import Any, AsyncContextManager, Protocol


class DatabaseServiceProtocol(Protocol):
    """Protocol for database session access in performance monitoring."""

    async def get_session(self) -> AsyncContextManager[Any]:
        """Get database session for performance monitoring operations.

        Returns:
            Async context manager for database session
        """
        ...


class PromptImprovementServiceProtocol(Protocol):
    """Protocol for prompt improvement operations in performance monitoring."""

    async def improve_prompt(
        self,
        prompt: str,
        context: dict[str, Any],
        session_id: str,
        rate_limit_remaining: int | None = None
    ) -> Any:
        """Improve a prompt using the core prompt improvement service.

        Args:
            prompt: Input prompt to improve
            context: Context dictionary for improvement
            session_id: Session identifier for tracking
            rate_limit_remaining: Optional rate limit info

        Returns:
            Prompt improvement result
        """
        ...


class ConfigurationServiceProtocol(Protocol):
    """Protocol for configuration access in performance monitoring."""

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        ...

    def get_performance_config(self) -> dict[str, Any]:
        """Get performance-specific configuration.

        Returns:
            Performance configuration dictionary
        """
        ...


class MLEventBusServiceProtocol(Protocol):
    """Protocol for ML event bus access in performance monitoring."""

    async def publish(self, event: Any) -> bool:
        """Publish event to ML event bus.

        Args:
            event: Event to publish

        Returns:
            True if event was published successfully
        """
        ...

    async def get_event_bus(self) -> Any:
        """Get the ML event bus instance.

        Returns:
            ML event bus instance
        """
        ...


class SessionStoreServiceProtocol(Protocol):
    """Protocol for session store operations in performance monitoring."""

    async def set(self, session_id: str, data: dict[str, Any]) -> None:
        """Set session data.

        Args:
            session_id: Session identifier
            data: Data to store
        """
        ...

    async def get(self, session_id: str) -> dict[str, Any] | None:
        """Get session data.

        Args:
            session_id: Session identifier

        Returns:
            Session data or None if not found
        """
        ...

    async def touch(self, session_id: str) -> None:
        """Touch session to update last access time.

        Args:
            session_id: Session identifier
        """
        ...

    async def delete(self, session_id: str) -> None:
        """Delete session data.

        Args:
            session_id: Session identifier
        """
        ...
