"""Shared utility modules for APES.

This package provides common utilities for:
- Secure subprocess execution
- Error handling decorators
- Health checking components
"""

# Removed service layer imports to fix circular dependency
# Error handlers should be imported directly from services.error_handling.facade
# SessionStore eliminated - use services.cache.cache_facade.CacheFacade session methods instead
from prompt_improver.utils.subprocess_security import (
    SecureSubprocessManager,
    ensure_running,
    secure_subprocess,
)

__all__ = [
    "SecureSubprocessManager",
    "ensure_running",
    "secure_subprocess",
]
