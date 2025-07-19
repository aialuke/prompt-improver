"""Shared utility modules for APES.

This package provides common utilities for:
- Secure subprocess execution
- Error handling decorators
- Health checking components
"""

from .error_handlers import (
                             handle_common_errors,
                             handle_database_errors,
                             handle_filesystem_errors,
                             handle_network_errors,
                             handle_validation_errors,
)
from .redis_cache import RedisCache, get, invalidate, set, with_singleflight

# Health checks removed to avoid circular import
# Import directly from utils.health_checks if needed
from .session_store import SessionStore
from .sql import fetch_scalar
from .subprocess_security import (
                             SecureSubprocessManager,
                             ensure_running,
                             secure_subprocess,
)

__all__ = [
    # Subprocess security
    "SecureSubprocessManager",
    "secure_subprocess",
    "ensure_running",
    # Error handling
    "handle_database_errors",
    "handle_filesystem_errors",
    "handle_validation_errors",
    "handle_network_errors",
    "handle_common_errors",
    # Session management
    "SessionStore",
    # Redis cache
    "RedisCache",
    "get",
    "set",
    "invalidate",
    "with_singleflight",
    # SQL utilities
    "fetch_scalar",
]
