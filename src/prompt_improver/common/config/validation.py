"""Configuration validation utilities for the prompt-improver system."""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from prompt_improver.common.exceptions import ValidationError
from prompt_improver.common.types import ConfigDict

@dataclass
class ValidationRule:
    """A single validation rule."""
    field: str
    validator: Callable[[Any], bool]
    message: str
    required: bool = True

class ConfigValidator:
    """Configuration validator with customizable rules."""

    def __init__(self):
        self.rules: list[ValidationRule] = []

    def add_rule(self, field: str, validator: Callable[[Any], bool], message: str, required: bool=True) -> None:
        """Add a validation rule."""
        rule = ValidationRule(field, validator, message, required)
        self.rules.append(rule)

    def validate(self, config: ConfigDict) -> list[ValidationError]:
        """Validate configuration against all rules."""
        errors = []
        for rule in self.rules:
            value = config.get(rule.field)
            if rule.required and value is None:
                errors.append(ValidationError(field=rule.field, message=f"Required field '{rule.field}' is missing", code='REQUIRED_FIELD_MISSING', value=None))
                continue
            if not rule.required and value is None:
                continue
            try:
                if not rule.validator(value):
                    errors.append(ValidationError(field=rule.field, message=rule.message, code='VALIDATION_FAILED', value=value))
            except Exception as e:
                errors.append(ValidationError(field=rule.field, message=f'Validation error: {e!s}', code='VALIDATION_EXCEPTION', value=value))
        return errors

    def is_valid(self, config: ConfigDict) -> bool:
        """Check if configuration is valid."""
        return len(self.validate(config)) == 0

def is_string(value: Any) -> bool:
    """Check if value is a string."""
    return isinstance(value, str)

def is_non_empty_string(value: Any) -> bool:
    """Check if value is a non-empty string."""
    return isinstance(value, str) and len(value.strip()) > 0

def is_integer(value: Any) -> bool:
    """Check if value is an integer."""
    return isinstance(value, int)

def is_positive_integer(value: Any) -> bool:
    """Check if value is a positive integer."""
    return isinstance(value, int) and value > 0

def is_non_negative_integer(value: Any) -> bool:
    """Check if value is a non-negative integer."""
    return isinstance(value, int) and value >= 0

def is_float(value: Any) -> bool:
    """Check if value is a float."""
    return isinstance(value, (int, float))

def is_positive_float(value: Any) -> bool:
    """Check if value is a positive float."""
    return isinstance(value, (int, float)) and value > 0

def is_boolean(value: Any) -> bool:
    """Check if value is a boolean."""
    return isinstance(value, bool)

def is_port_number(value: Any) -> bool:
    """Check if value is a valid port number."""
    return isinstance(value, int) and 1 <= value <= 65535

def is_url(value: Any) -> bool:
    """Check if value is a valid URL."""
    if not isinstance(value, str):
        return False
    return value.startswith(('http://', 'https://'))

def is_email(value: Any) -> bool:
    """Check if value is a valid email address."""
    if not isinstance(value, str):
        return False
    return '@' in value and '.' in value.split('@')[-1]

def is_in_choices(choices: list[Any]) -> Callable[[Any], bool]:
    """Create a validator that checks if value is in given choices."""

    def validator(value: Any) -> bool:
        return value in choices
    return validator

def min_length(min_len: int) -> Callable[[Any], bool]:
    """Create a validator that checks minimum string length."""

    def validator(value: Any) -> bool:
        return isinstance(value, str) and len(value) >= min_len
    return validator

def max_length(max_len: int) -> Callable[[Any], bool]:
    """Create a validator that checks maximum string length."""

    def validator(value: Any) -> bool:
        return isinstance(value, str) and len(value) <= max_len
    return validator

def min_value(min_val: int | float) -> Callable[[Any], bool]:
    """Create a validator that checks minimum numeric value."""

    def validator(value: Any) -> bool:
        return isinstance(value, (int, float)) and value >= min_val
    return validator

def max_value(max_val: int | float) -> Callable[[Any], bool]:
    """Create a validator that checks maximum numeric value."""

    def validator(value: Any) -> bool:
        return isinstance(value, (int, float)) and value <= max_val
    return validator

def range_value(min_val: int | float, max_val: int | float) -> Callable[[Any], bool]:
    """Create a validator that checks value is within range."""

    def validator(value: Any) -> bool:
        return isinstance(value, (int, float)) and min_val <= value <= max_val
    return validator

def validate_config_dict(config: ConfigDict, validator: ConfigValidator) -> None:
    """Validate a configuration dictionary and raise exception if invalid."""
    errors = validator.validate(config)
    if errors:
        error_messages = [f'{error.field}: {error.message}' for error in errors]
        raise ValidationError(message=f"Configuration validation failed: {'; '.join(error_messages)}", field='config', value=config)

def create_database_validator() -> ConfigValidator:
    """Create a validator for database configuration."""
    validator = ConfigValidator()
    validator.add_rule('host', is_non_empty_string, 'Host must be a non-empty string')
    validator.add_rule('port', is_port_number, 'Port must be between 1 and 65535')
    validator.add_rule('database', is_non_empty_string, 'Database name must be a non-empty string')
    validator.add_rule('username', is_non_empty_string, 'Username must be a non-empty string')
    validator.add_rule('password', is_string, 'Password must be a string', required=False)
    validator.add_rule('pool_size', is_positive_integer, 'Pool size must be a positive integer')
    validator.add_rule('max_overflow', is_non_negative_integer, 'Max overflow must be non-negative')
    return validator

def create_redis_validator() -> ConfigValidator:
    """Create a validator for Redis configuration."""
    validator = ConfigValidator()
    validator.add_rule('host', is_non_empty_string, 'Host must be a non-empty string')
    validator.add_rule('port', is_port_number, 'Port must be between 1 and 65535')
    validator.add_rule('db', is_non_negative_integer, 'Database number must be non-negative')
    validator.add_rule('password', is_string, 'Password must be a string', required=False)
    validator.add_rule('ssl', is_boolean, 'SSL must be a boolean')
    validator.add_rule('max_connections', is_positive_integer, 'Max connections must be positive')
    return validator

def create_security_validator() -> ConfigValidator:
    """Create a validator for security configuration."""
    validator = ConfigValidator()
    validator.add_rule('secret_key', min_length(32), 'Secret key must be at least 32 characters')
    validator.add_rule('token_expiry_seconds', is_positive_integer, 'Token expiry must be positive')
    validator.add_rule('hash_rounds', range_value(4, 20), 'Hash rounds must be between 4 and 20')
    validator.add_rule('max_login_attempts', is_positive_integer, 'Max login attempts must be positive')
    return validator

def create_monitoring_validator() -> ConfigValidator:
    """Create a validator for monitoring configuration."""
    validator = ConfigValidator()
    validator.add_rule('metrics_enabled', is_boolean, 'Metrics enabled must be a boolean')
    validator.add_rule('health_check_interval', is_positive_float, 'Health check interval must be positive')
    validator.add_rule('metrics_collection_interval', is_positive_float, 'Metrics collection interval must be positive')
    validator.add_rule('log_level', is_in_choices(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), 'Invalid log level')
    validator.add_rule('log_format', is_in_choices(['json', 'text']), "Log format must be 'json' or 'text'")
    validator.add_rule('opentelemetry_endpoint', is_url, 'OpenTelemetry endpoint must be a valid URL', required=False)
    return validator
