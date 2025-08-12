"""Configuration validation utilities for the prompt-improver system.

This module has been deprecated in favor of Pydantic-based validation.
Use the configuration classes in core.config with built-in Field constraints.
"""

# This module is deprecated - use Pydantic Field constraints instead
# For example: Field(ge=1, le=65535) instead of custom port validators
