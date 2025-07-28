"""MCP server middleware for JWT authentication and security.

Provides authentication middleware for FastMCP server with proper error handling
and security headers according to 2025 MCP security standards.
"""

import logging
import time
from typing import Any, Callable, Dict, Optional
from functools import wraps

from .mcp_authentication import MCPAuthenticationService, MCPPermission

logger = logging.getLogger(__name__)

class MCPAuthenticationError(Exception):
    """Custom exception for MCP authentication errors."""

    def __init__(self, message: str, error_code: str = "AUTHENTICATION_FAILED"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)

class MCPAuthMiddleware:
    """JWT authentication middleware for MCP server operations."""

    def __init__(self, auth_service: Optional[MCPAuthenticationService] = None):
        """Initialize MCP authentication middleware.

        Args:
            auth_service: MCP authentication service instance
        """
        self.auth_service = auth_service or MCPAuthenticationService()
        self._authenticated_sessions: Dict[str, Dict[str, Any]] = {}

    def extract_token_from_context(self, context: Dict[str, Any]) -> Optional[str]:
        """Extract JWT token from MCP request context.

        Args:
            context: MCP request context

        Returns:
            JWT token string if found, None otherwise
        """
        # Check for Authorization header
        headers = context.get("headers", {})
        auth_header = headers.get("authorization", "")

        if auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix

        # Check for token in request metadata
        metadata = context.get("metadata", {})
        return metadata.get("jwt_token")

    def authenticate_request(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate MCP request and return token payload.

        Args:
            context: MCP request context

        Returns:
            JWT token payload

        Raises:
            MCPAuthenticationError: If authentication fails
        """
        # Extract token from context
        token = self.extract_token_from_context(context)
        if not token:
            raise MCPAuthenticationError(
                "Missing authentication token. Include JWT token in Authorization header.",
                "MISSING_TOKEN"
            )

        # Validate token
        payload = self.auth_service.validate_agent_token(token)
        if not payload:
            raise MCPAuthenticationError(
                "Invalid or expired authentication token.",
                "INVALID_TOKEN"
            )

        # Cache authenticated session
        session_id = f"{payload['sub']}:{payload['jti']}"
        self._authenticated_sessions[session_id] = payload

        logger.info(
            f"Authenticated MCP request: agent={payload['agent_type']}, "
            f"tier={payload['rate_limit_tier']}, permissions={len(payload['permissions'])}"
        )

        return payload

    def check_permission(self, token_payload: Dict[str, Any], required_permission: MCPPermission) -> None:
        """Check if authenticated agent has required permission.

        Args:
            token_payload: Validated JWT payload
            required_permission: Required permission

        Raises:
            MCPAuthenticationError: If permission denied
        """
        if not self.auth_service.check_permission(token_payload, required_permission):
            raise MCPAuthenticationError(
                f"Permission denied. Required permission: {required_permission.value}",
                "PERMISSION_DENIED"
            )

    def require_authentication(self, required_permission: Optional[MCPPermission] = None):
        """Decorator to require JWT authentication for MCP tools.

        Args:
            required_permission: Optional specific permission required

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract context from MCP call
                # Note: FastMCP provides context in different ways depending on the call type
                context = kwargs.get("context", {})

                try:
                    # Authenticate request
                    token_payload = self.authenticate_request(context)

                    # Check specific permission if required
                    if required_permission:
                        self.check_permission(token_payload, required_permission)

                    # Add authentication info to kwargs for the tool
                    kwargs["auth_payload"] = token_payload
                    kwargs["agent_id"] = token_payload["sub"]
                    kwargs["agent_type"] = token_payload["agent_type"]
                    kwargs["rate_limit_tier"] = token_payload["rate_limit_tier"]

                    # Call the original function
                    return await func(*args, **kwargs)

                except MCPAuthenticationError as e:
                    logger.warning(f"Authentication failed for {func.__name__}: {e.message}")
                    # Return error response instead of raising exception
                    return {
                        "error": "Authentication failed",
                        "message": e.message,
                        "error_code": e.error_code,
                        "timestamp": time.time()
                    }
                except Exception as e:
                    logger.error(f"Unexpected authentication error in {func.__name__}: {e}")
                    # Return error response instead of raising exception
                    return {
                        "error": "Authentication service error",
                        "message": "Internal authentication error occurred",
                        "timestamp": time.time()
                    }

            return wrapper
        return decorator

    def require_rule_access(self):
        """Decorator requiring rule read/apply permissions."""
        return self.require_authentication(MCPPermission.RULE_APPLY)

    def require_feedback_access(self):
        """Decorator requiring feedback write permissions."""
        return self.require_authentication(MCPPermission.FEEDBACK_WRITE)

    def require_performance_access(self):
        """Decorator requiring performance read permissions."""
        return self.require_authentication(MCPPermission.PERFORMANCE_READ)

    def get_authenticated_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get currently authenticated sessions (for monitoring).

        Returns:
            Dict of session_id -> token_payload
        """
        return self._authenticated_sessions.copy()

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired authentication sessions.

        Returns:
            Number of sessions cleaned up
        """
        import time
        current_time = int(time.time())
        expired_sessions = []

        for session_id, payload in self._authenticated_sessions.items():
            if payload.get("exp", 0) <= current_time:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self._authenticated_sessions[session_id]

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired authentication sessions")

        return len(expired_sessions)

# Global middleware instance for MCP server (lazy-loaded)
_mcp_auth_middleware = None

def create_mcp_auth_service() -> MCPAuthenticationService:
    """Create MCP authentication service instance.

    Returns:
        Configured MCPAuthenticationService
    """
    return MCPAuthenticationService()

def get_mcp_auth_middleware() -> MCPAuthMiddleware:
    """Get global MCP authentication middleware instance.

    Returns:
        MCPAuthMiddleware instance
    """
    global _mcp_auth_middleware
    if _mcp_auth_middleware is None:
        _mcp_auth_middleware = MCPAuthMiddleware()
    return _mcp_auth_middleware

# Convenience decorators for common use cases (lazy-loaded)
def require_mcp_auth(required_permission=None):
    """Require MCP authentication decorator."""
    return get_mcp_auth_middleware().require_authentication(required_permission)

def require_rule_access():
    """Require rule access decorator."""
    return get_mcp_auth_middleware().require_rule_access()

def require_feedback_access():
    """Require feedback access decorator."""
    return get_mcp_auth_middleware().require_feedback_access()

def require_performance_access():
    """Require performance access decorator."""
    return get_mcp_auth_middleware().require_performance_access()
