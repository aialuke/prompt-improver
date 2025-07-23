"""Security configuration validator to ensure no hardcoded credentials are used."""

import logging
import os
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class SecurityConfigValidator:
    """Validates configuration for security compliance."""

    # Patterns for detecting hardcoded credentials
    INSECURE_PATTERNS = [
        (r"password.*=.*[\"'][^\"']+[\"']", "Hardcoded password detected"),
        (r"secret.*=.*[\"'][^\"']+[\"']", "Hardcoded secret detected"),
        (r"key.*=.*[\"'][^\"']+[\"']", "Hardcoded key detected"),
        (r"token.*=.*[\"'][^\"']+[\"']", "Hardcoded token detected"),
        (r"postgresql://[^:]+:[^@]+@", "Database URL with embedded credentials"),
        (r"apes_secure_password", "Default development password in use"),
        (r"YOUR_SECURE_PASSWORD_HERE", "Template password not replaced"),
    ]

    REQUIRED_ENV_VARS = [
        "POSTGRES_PASSWORD",
        "POSTGRES_USERNAME",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DATABASE"
    ]

    def validate_environment(self) -> Tuple[bool, List[str]]:
        """Validate environment variables for security compliance.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check required environment variables are set
        for var in self.REQUIRED_ENV_VARS:
            if not os.getenv(var):
                issues.append(f"Required environment variable {var} is not set")

        # Check password strength
        password = os.getenv("POSTGRES_PASSWORD")
        if password:
            password_issues = self._validate_password_strength(password)
            issues.extend(password_issues)

        # Check for insecure patterns in environment
        for key, value in os.environ.items():
            if key.startswith(("POSTGRES_", "DATABASE_", "DB_")):
                for pattern, message in self.INSECURE_PATTERNS:
                    if re.search(pattern, f"{key}={value}", re.IGNORECASE):
                        issues.append(f"{message} in environment variable {key}")

        return len(issues) == 0, issues

    def _validate_password_strength(self, password: str) -> List[str]:
        """Validate password meets security requirements."""
        issues = []

        if len(password) < 12:
            issues.append("Database password should be at least 12 characters long")

        if password in ["password", "admin", "apes_secure_password_2024", "YOUR_SECURE_PASSWORD_HERE"]:
            issues.append("Database password is using a known weak/default password")

        # Check for basic complexity
        has_upper = bool(re.search(r'[A-Z]', password))
        has_lower = bool(re.search(r'[a-z]', password))
        has_digit = bool(re.search(r'[0-9]', password))
        has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))

        complexity_score = sum([has_upper, has_lower, has_digit, has_special])
        if complexity_score < 3:
            issues.append("Database password should contain uppercase, lowercase, numbers, and special characters")

        return issues

    def validate_database_url(self, database_url: str) -> Tuple[bool, List[str]]:
        """Validate database URL for security issues."""
        issues = []

        # Check for embedded credentials
        if re.search(r'://[^:]+:[^@]+@', database_url):
            issues.append("Database URL contains embedded credentials - use environment variables instead")

        # Check for default passwords
        for pattern, message in self.INSECURE_PATTERNS:
            if re.search(pattern, database_url, re.IGNORECASE):
                issues.append(f"{message} in database URL")

        return len(issues) == 0, issues

    def generate_secure_password(self, length: int = 32) -> str:
        """Generate a cryptographically secure password."""
        import secrets
        import string

        # Use a mix of characters for strong passwords
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(secrets.choice(alphabet) for _ in range(length))

        # Ensure it meets complexity requirements
        while not self._validate_password_strength(password):
            password = ''.join(secrets.choice(alphabet) for _ in range(length))

        return password

    def audit_configuration(self) -> Dict[str, any]:
        """Perform complete security audit of configuration."""
        is_valid, issues = self.validate_environment()

        audit_result = {
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "is_secure": is_valid,
            "issues": issues,
            "recommendations": []
        }

        if not is_valid:
            audit_result["recommendations"] = [
                "Set all required environment variables",
                "Use strong, unique passwords",
                "Never commit credentials to version control",
                "Rotate passwords regularly",
                "Use encrypted secrets management in production"
            ]

        return audit_result


def validate_security_configuration() -> bool:
    """Quick security validation check for startup."""
    validator = SecurityConfigValidator()
    is_valid, issues = validator.validate_environment()

    if not is_valid:
        logger.warning("Security configuration issues detected:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False

    logger.info("Security configuration validation passed")
    return True


def generate_secure_env_template() -> str:
    """Generate a secure .env template with strong passwords."""
    validator = SecurityConfigValidator()
    secure_password = validator.generate_secure_password()

    template = f"""# Secure Database Configuration
# Generated on {__import__("datetime").datetime.now().isoformat()}
# NEVER commit this file to version control

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=apes_production
POSTGRES_USERNAME=apes_user
POSTGRES_PASSWORD={secure_password}

# Alternative: Use full DATABASE_URL (without embedded credentials in code)
# DATABASE_URL=postgresql+psycopg://apes_user:{secure_password}@localhost:5432/apes_production

# Development Mode
DEVELOPMENT_MODE=true
LOG_LEVEL=INFO

# Database Pool Settings
DB_POOL_MIN_SIZE=4
DB_POOL_MAX_SIZE=16
DB_POOL_TIMEOUT=10

# Performance Monitoring
ENABLE_PERFORMANCE_MONITORING=true
SLOW_QUERY_THRESHOLD=1000

# MCP Configuration
MCP_POSTGRES_ENABLED=true

# Test Configuration
TEST_DB_NAME=apes_test
"""
    return template