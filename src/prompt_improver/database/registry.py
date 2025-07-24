"""Modern SQLAlchemy 2025 Registry Manager

This module provides a centralized registry solution to prevent the
"Multiple classes found for path" error that occurs when SQLAlchemy
model classes are registered multiple times.

Key features:
- Single declarative base with proper registry management
- Clear registry for test isolation
- Fully qualified class path resolution
- Registry inspection utilities
- Best practices for SQLModel integration
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Type

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase, registry
from sqlmodel import SQLModel

logger = logging.getLogger(__name__)

# Create a centralized registry
_centralized_registry = registry(
    type_annotation_map={
        # Add any custom type mappings here if needed
    }
)

# Create centralized metadata
_centralized_metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(column_0_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s"
    }
)

class PromptImproverBase(DeclarativeBase):
    """Single declarative base for all prompt improver models.

    This follows 2025 best practices by:
    - Using a single registry for all models
    - Providing centralized metadata access
    - Supporting proper table extension for tests
    """

    # Use the centralized registry
    registry = _centralized_registry
    metadata = _centralized_metadata

class RegistryManager:
    """Registry manager for handling SQLAlchemy model registration.

    Provides utilities to:
    - Clear registries for test isolation
    - Inspect current registrations
    - Resolve class path conflicts
    """

    def __init__(self, base_class: type[DeclarativeBase] = PromptImproverBase):
        self.base_class = base_class
        self.registry = base_class.registry

    def clear_registry(self) -> None:
        """Clear the registry for test isolation.

        This prevents "Multiple classes found" errors when running tests
        that import models multiple times.
        """
        try:
            # Clear the class registry
            self.registry._class_registry.clear()

            # Clear the metadata
            self.registry.metadata.clear()

            logger.info("Registry cleared successfully")

        except Exception as e:
            logger.error(f"Failed to clear registry: {e}")
            raise

    def get_registered_classes(self) -> dict[str, type]:
        """Get all currently registered classes.

        Returns:
            Dict mapping class names to class objects
        """
        return dict(self.registry._class_registry.items())

    def is_class_registered(self, class_name: str) -> bool:
        """Check if a class is already registered.

        Args:
            class_name: The name of the class to check

        Returns:
            True if class is registered, False otherwise
        """
        return class_name in self.registry._class_registry

    def get_class_by_name(self, class_name: str) -> type | None:
        """Get a registered class by name.

        Args:
            class_name: The name of the class to retrieve

        Returns:
            The class object if found, None otherwise
        """
        return self.registry._class_registry.get(class_name)

    def resolve_class_path_conflict(self, class_name: str) -> str:
        """Resolve class path conflicts by providing fully qualified names.

        Args:
            class_name: The class name that has conflicts

        Returns:
            Fully qualified class path
        """
        # For prompt improver models, use the full module path
        if class_name in ['RulePerformance', 'PromptSession', 'RuleMetadata']:
            return f"prompt_improver.database.models.{class_name}"

        return class_name

    @contextmanager
    def isolated_registry(self) -> Iterator[None]:
        """Context manager for isolated registry operations.

        This is particularly useful for tests that need to ensure
        clean registry state.
        """
        # Save current state
        original_registry = dict(self.registry._class_registry)
        original_metadata = self.registry.metadata

        try:
            # Clear for isolation
            self.clear_registry()
            yield
        finally:
            # Restore original state
            self.registry._class_registry.update(original_registry)
            self.registry.metadata = original_metadata

    def diagnose_registry_conflicts(self) -> dict[str, Any]:
        """Diagnose registry conflicts and provide detailed information.

        Returns:
            Dictionary with conflict information
        """
        conflicts = {}
        registered_classes = self.get_registered_classes()

        # Check for duplicate registrations
        class_counts = {}
        for class_name in registered_classes:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Find conflicts
        for class_name, count in class_counts.items():
            if count > 1:
                conflicts[class_name] = {
                    'count': count,
                    'class_object': registered_classes.get(class_name),
                    'suggested_fix': self.resolve_class_path_conflict(class_name)
                }

        return {
            'conflicts': conflicts,
            'total_registered': len(registered_classes),
            'registry_size': len(self.registry._class_registry)
        }

# Global registry manager instance
_registry_manager: RegistryManager | None = None

def get_registry_manager() -> RegistryManager:
    """Get the global registry manager instance.

    Returns:
        RegistryManager instance
    """
    global _registry_manager
    if _registry_manager is None:
        _registry_manager = RegistryManager()
    return _registry_manager

def clear_registry() -> None:
    """Clear the global registry.

    This is a convenience function for test setup.
    """
    get_registry_manager().clear_registry()

    # Also clear Python's module cache for database models to prevent re-registration
    import sys
    modules_to_clear = []
    for module_name in list(sys.modules.keys()):
        if 'prompt_improver.database.models' in module_name:
            modules_to_clear.append(module_name)

    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]

    # Force reload of SQLModel patching to ensure clean state
    global _registry_manager
    _registry_manager = None
    patch_sqlmodel_registry()

def diagnose_registry() -> dict[str, Any]:
    """Diagnose registry conflicts.

    Returns:
        Dictionary with conflict information
    """
    return get_registry_manager().diagnose_registry_conflicts()

# Integrate with SQLModel
def patch_sqlmodel_registry():
    """Patch SQLModel to use our centralized registry.

    This prevents SQLModel from creating its own registry and
    ensures all models use the same registry.
    """
    # Replace SQLModel's registry with our centralized one
    SQLModel.registry = PromptImproverBase.registry
    SQLModel.metadata = PromptImproverBase.metadata

    logger.info("SQLModel registry patched to use centralized registry")

# Apply the patch at module import
patch_sqlmodel_registry()
