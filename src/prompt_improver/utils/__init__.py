"""Shared utility modules for APES.

This package provides common utilities for:
- Secure subprocess execution
- Error handling decorators
- Health checking components
"""

from .subprocess_security import SecureSubprocessManager, secure_subprocess
from .error_handlers import (
    handle_database_errors,
    handle_filesystem_errors,
    handle_validation_errors,
    handle_network_errors,
    handle_common_errors,
)
from .health_checks import (
    HealthChecker,
    health_check_component,
    run_health_check,
    check_database_health,
    check_mcp_performance,
    check_analytics_service,
    check_ml_service,
    check_system_resources,
)

__all__ = [
    # Subprocess security
    "SecureSubprocessManager", 
    "secure_subprocess",
    
    # Error handling
    "handle_database_errors",
    "handle_filesystem_errors", 
    "handle_validation_errors",
    "handle_network_errors",
    "handle_common_errors",
    
    # Health checking
    "HealthChecker",
    "health_check_component",
    "run_health_check",
    "check_database_health",
    "check_mcp_performance", 
    "check_analytics_service",
    "check_ml_service",
    "check_system_resources",
]
