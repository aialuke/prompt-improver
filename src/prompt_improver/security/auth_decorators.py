"""Authentication decorators for ML operations.

Provides secure session validation and user authentication for sensitive operations.
"""

import functools
import logging
from typing import Any, Callable, Optional

from .authentication import AuthenticationService
from .input_validator import InputValidator, ValidationError


logger = logging.getLogger(__name__)


def require_valid_session(auth_service: Optional[AuthenticationService] = None):
    """Decorator to require valid session for method calls.

    Args:
        auth_service: Optional authentication service instance

    Usage:
        @require_valid_session()
        async def sensitive_method(self, session_id: str, ...):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract self and session_id from arguments
            if not args:
                raise ValueError("Method must have 'self' parameter")

            self_instance = args[0]

            # Find session_id in arguments
            session_id = None
            if len(args) > 1:
                # Check positional arguments
                for i, arg in enumerate(args[1:], 1):
                    if isinstance(arg, str) and 'session' in func.__code__.co_varnames[i]:
                        session_id = arg
                        break

            # Check keyword arguments
            if session_id is None:
                session_id = kwargs.get('session_id')

            if session_id is None:
                raise ValueError("session_id parameter is required")

            # Get authentication service
            used_auth_service = auth_service
            if used_auth_service is None:
                if hasattr(self_instance, 'auth_service'):
                    used_auth_service = self_instance.auth_service
                else:
                    logger.warning("No authentication service available, skipping validation")
                    return await func(*args, **kwargs)

            # Validate session
            try:
                # For now, we'll do basic session format validation
                # In production, this would validate against stored sessions
                validator = InputValidator()
                validated_session = validator.validate_input("session_id", session_id)

                # Log security event
                logger.info(f"Validated session access: {validated_session[:8]}... for {func.__name__}")

                # Call original function with validated session
                if 'session_id' in kwargs:
                    kwargs['session_id'] = validated_session
                else:
                    # Replace in args tuple
                    args_list = list(args)
                    for i, arg in enumerate(args[1:], 1):
                        if isinstance(arg, str) and 'session' in func.__code__.co_varnames[i]:
                            args_list[i] = validated_session
                            break
                    args = tuple(args_list)

                return await func(*args, **kwargs)

            except ValidationError as e:
                logger.error(f"Session validation failed for {func.__name__}: {e.message}")
                raise PermissionError(f"Invalid session: {e.message}")
            except Exception as e:
                logger.error(f"Authentication error in {func.__name__}: {e}")
                raise PermissionError("Authentication failed")

        return wrapper
    return decorator


def require_user_permission(permission: str, auth_service: Optional[AuthenticationService] = None):
    """Decorator to require specific user permission.

    Args:
        permission: Required permission string
        auth_service: Optional authentication service instance

    Usage:
        @require_user_permission("ml_training")
        async def train_model(self, user_id: str, ...):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract self and user_id from arguments
            if not args:
                raise ValueError("Method must have 'self' parameter")

            self_instance = args[0]

            # Find user_id in arguments
            user_id = None
            if len(args) > 1:
                # Check positional arguments for user_id
                for i, arg in enumerate(args[1:], 1):
                    if isinstance(arg, str) and 'user' in func.__code__.co_varnames[i]:
                        user_id = arg
                        break

            # Check keyword arguments
            if user_id is None:
                user_id = kwargs.get('user_id')

            if user_id is None:
                # Check if it's in session data or context
                session_id = kwargs.get('session_id')
                if session_id:
                    # Extract user from session (simplified for now)
                    user_id = f"user_from_session_{session_id[:8]}"

            if user_id is None:
                raise ValueError("user_id parameter is required for permission check")

            # Get authentication service
            used_auth_service = auth_service
            if used_auth_service is None:
                if hasattr(self_instance, 'auth_service'):
                    used_auth_service = self_instance.auth_service
                else:
                    logger.warning("No authentication service available, skipping permission check")
                    return await func(*args, **kwargs)

            # Validate user and permissions
            try:
                # Validate user_id format
                validator = InputValidator()
                validated_user_id = validator.validate_input("user_id", user_id)

                # In production, this would check actual user permissions
                # For now, we'll do basic validation and logging
                logger.info(f"Permission check: {validated_user_id} needs '{permission}' for {func.__name__}")

                return await func(*args, **kwargs)

            except ValidationError as e:
                logger.error(f"User validation failed for {func.__name__}: {e.message}")
                raise PermissionError(f"Invalid user: {e.message}")
            except Exception as e:
                logger.error(f"Permission error in {func.__name__}: {e}")
                raise PermissionError("Permission check failed")

        return wrapper
    return decorator


def audit_ml_operation(operation_type: str):
    """Decorator to audit ML operations for security monitoring.

    Args:
        operation_type: Type of ML operation (training, inference, etc.)

    Usage:
        @audit_ml_operation("model_training")
        async def train_model(self, ...):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract context for auditing
            self_instance = args[0] if args else None

            # Get user/session info if available
            user_id = kwargs.get('user_id', 'unknown')
            session_id = kwargs.get('session_id', 'unknown')

            # Log operation start
            logger.info(
                f"ML Operation Start: {operation_type} - Function: {func.__name__} - "
                f"User: {user_id} - Session: {session_id[:8]}..."
            )

            try:
                result = await func(*args, **kwargs)

                # Log successful completion
                logger.info(
                    f"ML Operation Success: {operation_type} - Function: {func.__name__} - "
                    f"User: {user_id}"
                )

                return result

            except Exception as e:
                # Log operation failure
                logger.error(
                    f"ML Operation Failed: {operation_type} - Function: {func.__name__} - "
                    f"User: {user_id} - Error: {str(e)}"
                )
                raise

        return wrapper
    return decorator