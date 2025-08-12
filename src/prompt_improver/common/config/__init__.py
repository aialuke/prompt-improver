"""Configuration utilities for the prompt-improver system.

This package contains configuration management utilities
that are shared across the system.

Note: Custom ConfigValidator has been deprecated. Use Pydantic-based
configuration classes from core.config with built-in Field constraints.
"""

from prompt_improver.common.config.base import *

# validation module has been deprecated - use core.config instead
__all__ = [
    "BaseConfig",
    "DatabaseConfig",
    "MLConfig",
    "MonitoringConfig",
    "RedisConfig",
    "SecurityConfig",
    "merge_configs",
]
