"""Dependency injection container and utilities.

This module provides the unified container orchestrator that delegates to
specialized domain containers following clean architecture principles.
"""

# Export the main orchestrator interface
from prompt_improver.core.di.cli_container import CLIContainer, get_cli_container
from prompt_improver.core.di.container_orchestrator import (
    CircularDependencyError,
    DIContainer,
    ResourceManagementError,
    ServiceInitializationError,
    ServiceLifetime,
    ServiceNotRegisteredError,
    get_container,
    get_datetime_service,
    initialize_container,
    shutdown_container,
)

# Export specialized containers for direct access
from prompt_improver.core.di.core_container import (
    CoreContainer,
    get_core_container,
    shutdown_core_container,
)
from prompt_improver.core.di.database_container import (
    DatabaseContainer,
    get_database_container,
    shutdown_database_container,
)
from prompt_improver.core.di.ml_container import MLServiceContainer
from prompt_improver.core.di.monitoring_container import (
    MonitoringContainer,
    get_monitoring_container,
    shutdown_monitoring_container,
)
from prompt_improver.core.di.protocols import MonitoringContainerProtocol
from prompt_improver.core.di.security_container import (
    SecurityContainer,
    get_security_container,
    shutdown_security_container,
)

__all__ = [
    "CLIContainer",
    "CircularDependencyError",
    # Specialized containers
    "CoreContainer",
    # Main orchestrator interface
    "DIContainer",
    "DatabaseContainer",
    "MLServiceContainer",
    "MonitoringContainer",
    "MonitoringContainerProtocol",
    "ResourceManagementError",
    "SecurityContainer",
    "ServiceInitializationError",
    "ServiceLifetime",
    "ServiceNotRegisteredError",
    "get_cli_container",
    "get_container",
    # Container factories
    "get_core_container",
    "get_database_container",
    "get_datetime_service",
    "get_monitoring_container",
    "get_security_container",
    "initialize_container",
    "shutdown_container",
    # Container lifecycle
    "shutdown_core_container",
    "shutdown_database_container",
    "shutdown_monitoring_container",
    "shutdown_security_container",
]
