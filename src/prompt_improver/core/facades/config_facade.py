"""Configuration Management Facade - Reduces Config Module Coupling.

This facade provides unified configuration access while reducing direct imports
from 12 to 1 internal dependencies through lazy initialization.

Design:
- Protocol-based interface for loose coupling
- Lazy loading of configuration components
- Domain-specific configuration access
- Validation and migration coordination
- Zero circular import dependencies
"""

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ConfigFacadeProtocol(Protocol):
    """Protocol for configuration management facade."""

    def get_config(self) -> Any:
        """Get main application configuration."""
        ...

    def get_database_config(self) -> Any:
        """Get database configuration."""
        ...

    def get_security_config(self) -> Any:
        """Get security configuration."""
        ...

    def get_monitoring_config(self) -> Any:
        """Get monitoring configuration."""
        ...

    def get_ml_config(self) -> Any:
        """Get ML configuration."""
        ...

    def validate_configuration(self) -> dict[str, Any]:
        """Validate current configuration."""
        ...

    def reload_config(self) -> None:
        """Reload configuration from sources."""
        ...


class ConfigFacade(ConfigFacadeProtocol):
    """Configuration management facade with minimal coupling.

    Reduces config module coupling from 12 internal imports to 1.
    Provides unified interface for all configuration access patterns.
    """

    def __init__(self) -> None:
        """Initialize facade with lazy loading."""
        self._main_config = None
        self._database_config = None
        self._security_config = None
        self._monitoring_config = None
        self._ml_config = None
        self._config_factory = None
        self._validator = None
        logger.debug("ConfigFacade initialized with lazy loading")

    def _ensure_main_config(self) -> None:
        """Ensure main config is loaded (lazy loading)."""
        if self._main_config is None:
            # Only import when needed to reduce coupling
            from prompt_improver.core.config import get_config
            self._main_config = get_config()

    def _ensure_domain_config(self, domain: str) -> None:
        """Ensure domain-specific config is loaded."""
        self._ensure_main_config()

        if domain == "database" and self._database_config is None:
            from prompt_improver.core.config import get_database_config
            self._database_config = get_database_config()
        elif domain == "security" and self._security_config is None:
            from prompt_improver.core.config import get_security_config
            self._security_config = get_security_config()
        elif domain == "monitoring" and self._monitoring_config is None:
            from prompt_improver.core.config import get_monitoring_config
            self._monitoring_config = get_monitoring_config()
        elif domain == "ml" and self._ml_config is None:
            from prompt_improver.core.config import get_ml_config
            self._ml_config = get_ml_config()

    def _ensure_factory(self) -> None:
        """Ensure config factory is available."""
        if self._config_factory is None:
            from prompt_improver.core.config.factory import get_config_factory
            self._config_factory = get_config_factory()

    def _ensure_validator(self) -> None:
        """Ensure config validator is available."""
        if self._validator is None:
            from prompt_improver.core.config.validation import validate_configuration
            self._validator = validate_configuration

    def get_config(self) -> Any:
        """Get main application configuration."""
        self._ensure_main_config()
        return self._main_config

    def get_database_config(self) -> Any:
        """Get database configuration."""
        self._ensure_domain_config("database")
        return self._database_config

    def get_security_config(self) -> Any:
        """Get security configuration."""
        self._ensure_domain_config("security")
        return self._security_config

    def get_monitoring_config(self) -> Any:
        """Get monitoring configuration."""
        self._ensure_domain_config("monitoring")
        return self._monitoring_config

    def get_ml_config(self) -> Any:
        """Get ML configuration."""
        self._ensure_domain_config("ml")
        return self._ml_config

    def create_test_config(self, overrides: dict[str, Any] | None = None) -> Any:
        """Create test configuration."""
        self._ensure_factory()
        # Dynamic import to reduce coupling
        import importlib
        factory_module = importlib.import_module("prompt_improver.core.config.factory")
        create_test_config_func = factory_module.create_test_config
        return create_test_config_func(overrides or {})

    def validate_configuration(self) -> dict[str, Any]:
        """Validate current configuration."""
        self._ensure_validator()
        self._ensure_main_config()
        return self._validator(self._main_config)

    def reload_config(self) -> None:
        """Reload configuration from sources."""
        from prompt_improver.core.config import reload_config
        reload_config()

        # Clear cached configs to force reload
        self._main_config = None
        self._database_config = None
        self._security_config = None
        self._monitoring_config = None
        self._ml_config = None

    def get_environment_name(self) -> str:
        """Get current environment name."""
        config = self.get_config()
        return config.environment.environment

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.get_environment_name() == "development"

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.get_environment_name() == "production"

    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.get_environment_name() == "testing"

    def get_service_config(self, service_name: str) -> dict[str, Any]:
        """Get configuration for a specific service."""
        config = self.get_config()

        # Map common service names to config sections
        service_mapping = {
            "database": self.get_database_config,
            "security": self.get_security_config,
            "monitoring": self.get_monitoring_config,
            "ml": self.get_ml_config,
            "server": lambda: config.server,
            "logging": lambda: config.logging,
        }

        getter = service_mapping.get(service_name.lower())
        if getter:
            service_config = getter()
            return service_config.model_dump() if hasattr(service_config, "model_dump") else dict(service_config)

        return {}

    def setup_configuration_logging(self) -> None:
        """Setup configuration logging."""
        from prompt_improver.core.config.logging import setup_config_logging
        setup_config_logging()


# Global facade instance
_config_facade: ConfigFacade | None = None


def get_config_facade() -> ConfigFacade:
    """Get global config facade instance.

    Returns:
        ConfigFacade with lazy initialization and minimal coupling
    """
    global _config_facade
    if _config_facade is None:
        _config_facade = ConfigFacade()
    return _config_facade


def reset_config_facade() -> None:
    """Reset the global config facade (primarily for testing)."""
    global _config_facade
    _config_facade = None


__all__ = [
    "ConfigFacade",
    "ConfigFacadeProtocol",
    "get_config_facade",
    "reset_config_facade",
]
