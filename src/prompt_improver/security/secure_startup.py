"""Secure startup module that validates configuration before application startup."""

import logging
import os
import sys

from .config_validator import SecurityConfigValidator

logger = logging.getLogger(__name__)

class SecureStartupError(Exception):
    """Raised when security validation fails during startup."""
    pass

def enforce_secure_configuration(strict_mode: bool = True) -> None:
    """Enforce secure configuration requirements before allowing application startup.

    Args:
        strict_mode: If True, raises exception on security violations.
                    If False, logs warnings but allows startup.

    Raises:
        SecureStartupError: If strict_mode=True and security issues are found.
    """
    logger.info("Performing security configuration validation...")

    validator = SecurityConfigValidator()
    audit_result = validator.audit_configuration()

    if not audit_result["is_secure"]:
        error_msg = "Security configuration validation failed:"
        for issue in audit_result["issues"]:
            error_msg += f"\n  - {issue}"

        error_msg += "\n\nRecommendations:"
        for rec in audit_result["recommendations"]:
            error_msg += f"\n  - {rec}"

        if strict_mode:
            logger.error(error_msg)
            raise SecureStartupError(error_msg)
        else:
            logger.warning(error_msg)
            logger.warning("Continuing startup in non-strict mode - FIX THESE ISSUES IMMEDIATELY")
    else:
        logger.info("✅ Security configuration validation passed")

def check_environment_security() -> bool:
    """Check if environment is securely configured.

    Returns:
        True if secure, False otherwise
    """
    try:
        enforce_secure_configuration(strict_mode=False)
        return True
    except SecureStartupError:
        return False

def get_secure_database_url() -> str:
    """Get database URL using secure environment configuration.

    Returns:
        Secure database URL constructed from environment variables

    Raises:
        SecureStartupError: If required environment variables are missing
    """
    required_vars = ["POSTGRES_USERNAME", "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DATABASE"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise SecureStartupError(
            f"Missing required environment variables for secure database connection: {missing_vars}"
        )

    # Validate password strength
    validator = SecurityConfigValidator()
    password = os.getenv("POSTGRES_PASSWORD")
    password_issues = validator._validate_password_strength(password)

    if password_issues and os.getenv("ENVIRONMENT") == "production":
        raise SecureStartupError(f"Password security issues: {password_issues}")

    username = os.getenv("POSTGRES_USERNAME")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    database = os.getenv("POSTGRES_DATABASE")

    return f"postgresql+psycopg://{username}:{password}@{host}:{port}/{database}"

def setup_secure_logging() -> None:
    """Configure logging to prevent credential leakage."""

    class SecureFormatter(logging.Formatter):
        """Custom formatter that redacts sensitive information."""

        SENSITIVE_PATTERNS = [
            # Password patterns
            (r'password["\s]*[:=]["\s]*([^"\s]+)', r'password=***REDACTED***'),
            (r'POSTGRES_PASSWORD["\s]*[:=]["\s]*([^"\s]+)', r'POSTGRES_PASSWORD=***REDACTED***'),
            # Database URL patterns
            (r'postgresql://([^:]+):([^@]+)@', r'postgresql://\1:***REDACTED***@'),
            (r'postgresql\+psycopg://([^:]+):([^@]+)@', r'postgresql+psycopg://\1:***REDACTED***@'),
            # Authentication tokens
            (r'Bearer\s+([A-Za-z0-9\-._~+/]+)', r'Bearer ***REDACTED***'),
            # API keys
            (r'api[_-]?key["\s]*[:=]["\s]*([^"\s]+)', r'api_key=***REDACTED***'),
        ]

        def format(self, record):
            msg = super().format(record)

            # Redact sensitive information
            for pattern, replacement in self.SENSITIVE_PATTERNS:
                msg = __import__("re").sub(pattern, replacement, msg, flags=__import__("re").IGNORECASE)

            return msg

    # Apply secure formatter to all handlers
    for handler in logging.root.handlers:
        handler.setFormatter(SecureFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

def initialize_secure_environment() -> None:
    """Initialize secure environment configuration."""

    # Set up secure logging first
    setup_secure_logging()

    # Check if we're in development mode
    is_development = os.getenv("DEVELOPMENT_MODE", "true").lower() == "true"
    is_production = os.getenv("ENVIRONMENT", "").lower() == "production"

    # In production, always use strict mode
    strict_mode = is_production or not is_development

    try:
        enforce_secure_configuration(strict_mode=strict_mode)
        logger.info("✅ Secure environment initialization completed")
    except SecureStartupError as e:
        logger.error("❌ Secure environment initialization failed")
        if strict_mode:
            sys.exit(1)

# Auto-initialize when module is imported in production
if os.getenv("ENVIRONMENT", "").lower() == "production":
    initialize_secure_environment()