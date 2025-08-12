"""Deprecated: Protocol definitions have moved to shared.interfaces.

This module is kept for backward compatibility and will be removed
in a future version. Please import from prompt_improver.shared.interfaces instead.
"""

# Re-export from shared interfaces for backward compatibility
from prompt_improver.shared.interfaces.protocols import (
    CacheProtocol,
    ConfigManagerProtocol,
    ConnectionManagerProtocol,
    DatabaseManagerProtocol,
    EventBusProtocol,
    HealthCheckerProtocol,
    LoggerProtocol,
    MetricsCollectorProtocol,
    MLModelProtocol,
    RepositoryProtocol,
    RetryManagerProtocol,
    SecurityManagerProtocol,
    ServiceProtocol,
    WorkflowProtocol,
)

__all__ = [
    "CacheProtocol",
    "ConfigManagerProtocol",
    "ConnectionManagerProtocol",
    "DatabaseManagerProtocol",
    "EventBusProtocol",
    "HealthCheckerProtocol",
    "LoggerProtocol",
    "MetricsCollectorProtocol",
    "MLModelProtocol",
    "RepositoryProtocol",
    "RetryManagerProtocol",
    "SecurityManagerProtocol",
    "ServiceProtocol",
    "WorkflowProtocol",
]
