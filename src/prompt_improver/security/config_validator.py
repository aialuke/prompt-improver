"""Security configuration validator to ensure no hardcoded credentials are used."""
import logging
import os
import re
from typing import Dict, List, Tuple
logger = logging.getLogger(__name__)

class SecurityConfigValidator:
    """Validates configuration for security compliance."""
    INSECURE_PATTERNS = [('password.*=.*[\\"\'][^\\"\']+[\\"\']', 'Hardcoded password detected'), ('secret.*=.*[\\"\'][^\\"\']+[\\"\']', 'Hardcoded secret detected'), ('key.*=.*[\\"\'][^\\"\']+[\\"\']', 'Hardcoded key detected'), ('token.*=.*[\\"\'][^\\"\']+[\\"\']', 'Hardcoded token detected'), ('postgresql://[^:]+:[^@]+@', 'Database URL with embedded credentials'), ('apes_secure_password', 'Default development password in use'), ('YOUR_SECURE_PASSWORD_HERE', 'Template password not replaced')]
    REQUIRED_ENV_VARS = ['POSTGRES_PASSWORD', 'POSTGRES_USERNAME', 'POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_DATABASE']

    def validate_environment(self) -> tuple[bool, list[str]]:
        """Validate environment variables for security compliance.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        for var in self.REQUIRED_ENV_VARS:
            if not os.getenv(var):
                issues.append(f'Required environment variable {var} is not set')
        password = os.getenv('POSTGRES_PASSWORD')
        if password:
            password_issues = self._validate_password_strength(password)
            issues.extend(password_issues)
        for key, value in os.environ.items():
            if key.startswith(('POSTGRES_', 'DATABASE_', 'DB_')):
                for pattern, message in self.INSECURE_PATTERNS:
                    if re.search(pattern, f'{key}={value}', re.IGNORECASE):
                        issues.append(f'{message} in environment variable {key}')
        return (len(issues) == 0, issues)

    def _validate_password_strength(self, password: str) -> list[str]:
        """Validate password meets security requirements."""
        issues = []
        if len(password) < 12:
            issues.append('Database password should be at least 12 characters long')
        if password in ['password', 'admin', 'apes_secure_password_2024', 'YOUR_SECURE_PASSWORD_HERE']:
            issues.append('Database password is using a known weak/default password')
        has_upper = bool(re.search('[A-Z]', password))
        has_lower = bool(re.search('[a-z]', password))
        has_digit = bool(re.search('[0-9]', password))
        has_special = bool(re.search('[!@#$%^&*(),.?":{}|<>]', password))
        complexity_score = sum([has_upper, has_lower, has_digit, has_special])
        if complexity_score < 3:
            issues.append('Database password should contain uppercase, lowercase, numbers, and special characters')
        return issues

    def validate_database_url(self, database_url: str) -> tuple[bool, list[str]]:
        """Validate database URL for security issues."""
        issues = []
        if re.search('://[^:]+:[^@]+@', database_url):
            issues.append('Database URL contains embedded credentials - use environment variables instead')
        for pattern, message in self.INSECURE_PATTERNS:
            if re.search(pattern, database_url, re.IGNORECASE):
                issues.append(f'{message} in database URL')
        return (len(issues) == 0, issues)

    def generate_secure_password(self, length: int=32) -> str:
        """Generate a cryptographically secure password."""
        import secrets
        import string
        alphabet = string.ascii_letters + string.digits + '!@#$%^&*'
        password = ''.join((secrets.choice(alphabet) for _ in range(length)))
        while not self._validate_password_strength(password):
            password = ''.join((secrets.choice(alphabet) for _ in range(length)))
        return password

    def audit_configuration(self) -> dict[str, any]:
        """Perform complete security audit of configuration."""
        is_valid, issues = self.validate_environment()
        audit_result = {'timestamp': __import__('datetime').datetime.now().isoformat(), 'is_secure': is_valid, 'issues': issues, 'recommendations': []}
        if not is_valid:
            audit_result['recommendations'] = ['Set all required environment variables', 'Use strong, unique passwords', 'Never commit credentials to version control', 'Rotate passwords regularly', 'Use encrypted secrets management in production']
        return audit_result

def validate_security_configuration() -> bool:
    """Quick security validation check for startup."""
    validator = SecurityConfigValidator()
    is_valid, issues = validator.validate_environment()
    if not is_valid:
        logger.warning('Security configuration issues detected:')
        for issue in issues:
            logger.warning('  - %s', issue)
        return False
    logger.info('Security configuration validation passed')
    return True

def generate_secure_env_template() -> str:
    """Generate a secure .env template with strong passwords."""
    validator = SecurityConfigValidator()
    secure_password = validator.generate_secure_password()
    template = f"# Secure Database Configuration\n# Generated on {__import__('datetime').datetime.now().isoformat()}\n# NEVER commit this file to version control\n\n# Database Configuration\nPOSTGRES_HOST=localhost\nPOSTGRES_PORT=5432\nPOSTGRES_DATABASE=apes_production\nPOSTGRES_USERNAME=apes_user\nPOSTGRES_PASSWORD={secure_password}\n\n# Alternative: Use full DATABASE_URL (without embedded credentials in code)\n# DATABASE_URL=postgresql+asyncpg://apes_user:{secure_password}@localhost:5432/apes_production\n\n# Development Mode\nDEVELOPMENT_MODE=true\nLOG_LEVEL=INFO\n\n# Database Pool Settings\nDB_POOL_MIN_SIZE=4\nDB_POOL_MAX_SIZE=16\nDB_POOL_TIMEOUT=10\n\n# Performance Monitoring\nENABLE_PERFORMANCE_MONITORING=true\nSLOW_QUERY_THRESHOLD=1000\n\n# MCP Configuration\nMCP_POSTGRES_ENABLED=true\n\n# Test Configuration\nTEST_DB_NAME=apes_test\n"
    return template
