"""Minimal Configuration Facade - Ultimate Loose Coupling Implementation.

This facade provides unified configuration access with zero internal imports at module level.
All imports are done dynamically to achieve maximum decoupling.

Key improvements:
- Zero internal imports at module level
- 100% dynamic imports for loose coupling
- Protocol-based interfaces
- Lazy initialization to minimize startup dependencies
- Zero circular import possibilities
"""

import importlib
import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class MinimalConfigFacadeProtocol(Protocol):
    """Protocol for minimal configuration facade."""

    def get_config(self) -> Any: ...
    def get_database_config(self) -> Any: ...
    def get_security_config(self) -> Any: ...
    def get_monitoring_config(self) -> Any: ...
    def get_ml_config(self) -> Any: ...
    def validate_configuration(self) -> dict[str, Any]: ...
    def reload_config(self) -> None: ...


class MinimalConfigFacade(MinimalConfigFacadeProtocol):
    """Minimal configuration facade with zero coupling at module level."""

    def __init__(self) -> None:
        """Initialize facade with lazy loading."""
        self._cache = {}
        logger.debug("MinimalConfigFacade initialized with zero coupling")

    def _dynamic_import(self, module_path: str, function_name: str):
        """Dynamically import and call a function."""
        try:
            module = importlib.import_module(module_path)
            return getattr(module, function_name)
        except Exception as e:
            logger.exception(f"Dynamic import failed: {module_path}.{function_name} - {e}")
            return None

    def get_config(self) -> Any:
        """Get main configuration via dynamic import."""
        if "main_config" not in self._cache:
            get_config_func = self._dynamic_import("prompt_improver.core.config", "get_config")
            if get_config_func:
                self._cache["main_config"] = get_config_func()
        return self._cache.get("main_config")

    def get_database_config(self) -> Any:
        """Get database configuration via dynamic import."""
        if "database_config" not in self._cache:
            get_db_config_func = self._dynamic_import("prompt_improver.core.config", "get_database_config")
            if get_db_config_func:
                self._cache["database_config"] = get_db_config_func()
        return self._cache.get("database_config")

    def get_security_config(self) -> Any:
        """Get security configuration via dynamic import."""
        if "security_config" not in self._cache:
            get_security_func = self._dynamic_import("prompt_improver.core.config", "get_security_config")
            if get_security_func:
                self._cache["security_config"] = get_security_func()
        return self._cache.get("security_config")

    def get_monitoring_config(self) -> Any:
        """Get monitoring configuration via dynamic import."""
        if "monitoring_config" not in self._cache:
            get_monitoring_func = self._dynamic_import("prompt_improver.core.config", "get_monitoring_config")
            if get_monitoring_func:
                self._cache["monitoring_config"] = get_monitoring_func()
        return self._cache.get("monitoring_config")

    def get_ml_config(self) -> Any:
        """Get ML configuration via dynamic import."""
        if "ml_config" not in self._cache:
            get_ml_func = self._dynamic_import("prompt_improver.core.config", "get_ml_config")
            if get_ml_func:
                self._cache["ml_config"] = get_ml_func()
        return self._cache.get("ml_config")

    def validate_configuration(self) -> dict[str, Any]:
        """Validate configuration via dynamic import."""
        validate_func = self._dynamic_import("prompt_improver.core.config.validation", "validate_configuration")
        if validate_func and self.get_config():
            return validate_func(self.get_config())
        return {"error": "validation_not_available"}

    def reload_config(self) -> None:
        """Reload configuration via dynamic import."""
        reload_func = self._dynamic_import("prompt_improver.core.config", "reload_config")
        if reload_func:
            reload_func()
            self._cache.clear()  # Clear cache to force reload

    def get_environment_name(self) -> str:
        """Get environment name."""
        config = self.get_config()
        return getattr(getattr(config, "environment", None), "environment", "unknown")

    def is_development(self) -> bool:
        """Check if development environment."""
        return self.get_environment_name() == "development"

    def is_production(self) -> bool:
        """Check if production environment."""
        return self.get_environment_name() == "production"

    def is_testing(self) -> bool:
        """Check if testing environment."""
        return self.get_environment_name() == "testing"


# Global minimal facade instance
_minimal_config_facade: MinimalConfigFacade | None = None


def get_minimal_config_facade() -> MinimalConfigFacade:
    """Get global minimal config facade with zero coupling."""
    global _minimal_config_facade
    if _minimal_config_facade is None:
        _minimal_config_facade = MinimalConfigFacade()
    return _minimal_config_facade


__all__ = [
    "MinimalConfigFacade",
    "MinimalConfigFacadeProtocol",
    "get_minimal_config_facade",
]
