"""Dependency injection container and utilities.

This module provides the unified container orchestrator that delegates to
specialized domain containers following clean architecture principles.
"""

# Export the main orchestrator interface for backward compatibility
from prompt_improver.core.di.container_orchestrator import (
    DIContainer,
    ServiceLifetime,
    ServiceNotRegisteredError,
    ServiceInitializationError,
    CircularDependencyError,
    ResourceManagementError,
    get_container,
    get_datetime_service,
    shutdown_container,
    initialize_container,
)

# Export specialized containers for direct access
from prompt_improver.core.di.core_container import (
    CoreContainer,
    get_core_container,
    shutdown_core_container,
)
from prompt_improver.core.di.security_container import (
    SecurityContainer,
    get_security_container,
    shutdown_security_container,
)
from prompt_improver.core.di.database_container import (
    DatabaseContainer,
    get_database_container,
    shutdown_database_container,
)
from prompt_improver.core.di.monitoring_container import (
    MonitoringContainer,
    get_monitoring_container,
    shutdown_monitoring_container,
)
from prompt_improver.core.di.ml_container import MLServiceContainer
from prompt_improver.core.di.protocols import MonitoringContainerProtocol

__all__ = [
    # Main orchestrator interface
    "DIContainer",
    "ServiceLifetime",
    "ServiceNotRegisteredError",
    "ServiceInitializationError",
    "CircularDependencyError",
    "ResourceManagementError",
    "get_container",
    "get_datetime_service",
    "shutdown_container",
    "initialize_container",
    
    # Specialized containers
    "CoreContainer",
    "SecurityContainer",
    "DatabaseContainer",
    "MonitoringContainer",
    "MonitoringContainerProtocol",
    "MLServiceContainer",
    
    # Container factories
    "get_core_container",
    "get_security_container",
    "get_database_container",
    "get_monitoring_container",
    
    # Container lifecycle
    "shutdown_core_container",
    "shutdown_security_container",
    "shutdown_database_container",
    "shutdown_monitoring_container",
]
