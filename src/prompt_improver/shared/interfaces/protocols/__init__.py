"""Consolidated protocol definitions for the prompt improver system.

This module provides a centralized location for all protocol definitions,
organized by domain and functionality. Supports lazy loading for ML protocols
to avoid heavy dependency imports during regular application startup.

Usage:
    # Import core protocols
    from prompt_improver.shared.interfaces.protocols import core, cache, database

    # Import specific protocols
    from prompt_improver.shared.interfaces.protocols.cache import CacheServiceProtocol

    # Lazy load ML protocols when needed
    from prompt_improver.shared.interfaces.protocols.ml import get_ml_protocols
"""


# Core system protocols - always available (fast imports)
from prompt_improver.shared.interfaces.protocols import (
    application,
    cache,
    cli,
    core,
    database,
    mcp,
    security,
)

# Monitoring protocols - lazy loaded to avoid heavy dependencies

# ML protocols - lazy loaded to avoid torch/tensorflow dependencies


def get_ml_protocols():
    """Lazy load ML protocols to avoid heavy dependencies.

    This function imports ML protocols only when needed, preventing
    torch/tensorflow/numpy imports during regular application startup.

    Returns:
        Module containing ML protocol definitions
    """
    from prompt_improver.shared.interfaces.protocols import ml
    return ml


def get_monitoring_protocols():
    """Lazy load monitoring protocols to avoid heavy dependencies.

    This function imports monitoring protocols only when needed, preventing
    heavy monitoring system imports during regular application startup.

    Returns:
        Module containing monitoring protocol definitions
    """
    from prompt_improver.shared.interfaces.protocols import monitoring
    return monitoring


# Export commonly used protocols directly
from prompt_improver.shared.interfaces.protocols import ml, monitoring
from prompt_improver.shared.interfaces.protocols.cache import (
    BasicCacheProtocol,
    CacheHealthProtocol,
    CacheServiceFacadeProtocol,
    CacheServiceProtocol,
)
from prompt_improver.shared.interfaces.protocols.cli import (
    CommandProcessorProtocol,
    UserInteractionProtocol,
    WorkflowManagerProtocol,
)
from prompt_improver.shared.interfaces.protocols.core import (
    DateTimeServiceProtocol,
    DateTimeUtilsProtocol,
    HealthCheckProtocol,
    ServiceProtocol,
    TimeZoneServiceProtocol,
)
from prompt_improver.shared.interfaces.protocols.database import (
    ConnectionPoolCoreProtocol,
    DatabaseProtocol,
    SessionManagerProtocol,
)
from prompt_improver.shared.interfaces.protocols.mcp import (
    LifecycleManagerProtocol,
    MCPResourceProtocol,
    MCPServerProtocol,
    MCPSessionProtocol,
    MCPToolProtocol,
    RuntimeManagerProtocol,
    ServerConfigProtocol,
    ServerFactoryProtocol,
    ServerServicesProtocol,
    ServiceFactoryProtocol,
)

# Monitoring protocols available via lazy loading: get_monitoring_protocols()
from prompt_improver.shared.interfaces.protocols.security import (
    AuthenticationProtocol,
    AuthorizationProtocol,
    EncryptionProtocol,
)

# Application protocols available via lazy loading in interfaces/__init__.py

# Organized exports by domain
__all__ = [
    # Monitoring protocols (available via get_monitoring_protocols())
    # Security protocols
    "AuthenticationProtocol",
    "AuthorizationProtocol",
    # Cache protocols
    "BasicCacheProtocol",
    "CacheHealthProtocol",
    "CacheServiceFacadeProtocol",
    "CacheServiceProtocol",
    # CLI protocols
    "CommandProcessorProtocol",
    "ConnectionPoolCoreProtocol",
    "DatabaseProtocol",
    "DateTimeServiceProtocol",
    "DateTimeUtilsProtocol",
    "EncryptionProtocol",
    "HealthCheckProtocol",
    "LifecycleManagerProtocol",
    "MCPResourceProtocol",
    # MCP protocols
    "MCPServerProtocol",
    "MCPSessionProtocol",
    "MCPToolProtocol",
    "RuntimeManagerProtocol",
    "ServerConfigProtocol",
    "ServerFactoryProtocol",
    "ServerServicesProtocol",
    "ServiceFactoryProtocol",
    # Core protocols
    "ServiceProtocol",
    # Database protocols
    "SessionManagerProtocol",
    "TimeZoneServiceProtocol",
    "UserInteractionProtocol",
    "WorkflowManagerProtocol",
    "application",
    "cache",
    "cli",
    # Modules
    "core",
    "database",
    # Lazy loading
    "get_ml_protocols",
    "get_monitoring_protocols",
    "mcp",
    "ml",
    "monitoring",
    "security",

    # Application protocols available via lazy loading
]
