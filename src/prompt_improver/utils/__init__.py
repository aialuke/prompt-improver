"""Shared utility modules for APES.

This package provides common utilities for:
- Secure subprocess execution
- Error handling decorators
- Health checking components
"""
from prompt_improver.utils.error_handlers import handle_common_errors, handle_database_errors, handle_filesystem_errors, handle_network_errors, handle_validation_errors
from prompt_improver.utils.session_store import SessionStore
from prompt_improver.utils.sql import fetch_scalar
from prompt_improver.utils.subprocess_security import SecureSubprocessManager, ensure_running, secure_subprocess
__all__ = ['SecureSubprocessManager', 'secure_subprocess', 'ensure_running', 'handle_database_errors', 'handle_filesystem_errors', 'handle_validation_errors', 'handle_network_errors', 'handle_common_errors', 'SessionStore', 'RedisCache', 'get', 'set', 'invalidate', 'with_singleflight', 'fetch_scalar']
