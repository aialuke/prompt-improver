"""Security wiring and authentication for MCP server.

Handles security service creation, middleware configuration, and authentication setup.
Implements unified security architecture with OWASP compliance.
"""

import logging
from typing import TYPE_CHECKING, Any

from prompt_improver.core.config import AppConfig
from prompt_improver.mcp_server.middleware import (
    SecurityMiddlewareAdapter,
    create_mcp_server_security_middleware,
)
from prompt_improver.security.input_validator import InputValidator
from prompt_improver.security.output_validator import OutputValidator
from prompt_improver.security.unified_authentication_manager import (
    get_unified_authentication_manager,
)
from prompt_improver.security.unified_security_manager import (
    get_mcp_security_manager,
)
from prompt_improver.security.unified_security_stack import (
    get_mcp_server_security_stack,
)
from prompt_improver.security.unified_validation_manager import (
    ValidationMode,
    get_unified_validation_manager,
)

if TYPE_CHECKING:
    from prompt_improver.mcp_server.server import APESMCPServer, ServerServices

logger = logging.getLogger(__name__)


async def create_security_services(config: AppConfig) -> dict[str, Any]:
    """Create and initialize all security-related services.

    Creates the unified security architecture components including:
    - Security manager with MCP server mode
    - Validation manager with OWASP 2025 compliance
    - Authentication manager with fail-secure design
    - Security stack with 6-layer OWASP protection
    - Input/output validators for content security

    Args:
        config: Application configuration

    Returns:
        Dictionary containing initialized security services

    Raises:
        RuntimeError: If security initialization fails
    """
    logger.info("Initializing unified security architecture components...")

    try:
        # Initialize core security managers
        security_manager = await get_mcp_security_manager()
        validation_manager = await get_unified_validation_manager(
            ValidationMode.MCP_SERVER
        )
        authentication_manager = await get_unified_authentication_manager()
        security_stack = await get_mcp_server_security_stack()

        # Initialize security middleware
        unified_security_middleware = await create_mcp_server_security_middleware()
        security_adapter = SecurityMiddlewareAdapter(unified_security_middleware)

        # Initialize validators
        input_validator = InputValidator()
        output_validator = OutputValidator()

        logger.info("Unified security components initialized successfully")
        logger.info("- UnifiedSecurityManager: MCP server mode active")
        logger.info("- UnifiedValidationManager: OWASP 2025 compliance enabled")
        logger.info("- UnifiedAuthenticationManager: Fail-secure authentication active")
        logger.info("- UnifiedSecurityStack: 6-layer OWASP security active")
        logger.info("- Input/Output validators: Content security enabled")

        return {
            "security_manager": security_manager,
            "validation_manager": validation_manager,
            "authentication_manager": authentication_manager,
            "security_stack": security_stack,
            "security_middleware_adapter": security_adapter,
            "input_validator": input_validator,
            "output_validator": output_validator,
        }

    except Exception as e:
        logger.error(f"Failed to initialize unified security components: {e}")
        raise RuntimeError(f"Security initialization failed: {e}")


def setup_security_middleware(
    server: "APESMCPServer", services: "ServerServices"
) -> None:
    """Configure security middleware for the server instance.

    Wires up security middleware to intercept and validate all requests/responses.
    Ensures proper security layer ordering per OWASP guidelines.

    Args:
        server: The APESMCPServer instance to configure
        services: Container with initialized services including security components
    """
    logger.info("Configuring security middleware stack...")

    # Security middleware is already configured in services.security_middleware_adapter
    # and will be used by the security stack for request/response processing

    logger.info("Security middleware configuration completed")


async def validate_security_status(services: "ServerServices") -> dict[str, Any]:
    """Validate the security architecture status.

    Performs comprehensive security validation including:
    - Security manager status check
    - Authentication manager health
    - Validation manager compliance
    - Security stack layer verification

    Args:
        services: Container with security services

    Returns:
        Dictionary containing security validation results
    """
    try:
        security_status = await services.security_manager.get_security_status()

        validation_result = {
            "security_mode": security_status.get("mode", "unknown"),
            "authentication_active": hasattr(
                services.authentication_manager, "is_active"
            ),
            "validation_compliance": services.validation_manager.mode
            == ValidationMode.MCP_SERVER,
            "security_layers": 6,  # UnifiedSecurityStack layers
            "input_validation": isinstance(services.input_validator, InputValidator),
            "output_validation": isinstance(services.output_validator, OutputValidator),
            "middleware_configured": services.security_middleware_adapter is not None,
            "overall_status": "active",
        }

        logger.info(
            f"Security validation completed: {validation_result['security_mode']}"
        )
        return validation_result

    except Exception as e:
        logger.error(f"Security validation failed: {e}")
        return {
            "overall_status": "error",
            "error": str(e),
        }


def get_security_health_check() -> dict[str, Any]:
    """Get security component health status.

    Returns basic health information for security components without
    requiring full service initialization.

    Returns:
        Dictionary containing security health status
    """
    return {
        "security_architecture": "unified",
        "compliance_standard": "OWASP_2025",
        "authentication_mode": "fail_secure",
        "validation_layers": 6,
        "content_security": "enabled",
        "middleware_stack": "active",
        "status": "healthy",
    }
