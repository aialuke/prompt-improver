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
# Health checks removed to avoid circular import
# Import directly from utils.health_checks if needed
from .session_store import SessionStore

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
    
    # Session management
    "SessionStore",
]
