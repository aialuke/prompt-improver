"""Common utilities and shared patterns to eliminate code duplication.

This module provides centralized implementations for frequently duplicated patterns:
- Logger initialization
- Configuration loading
- Metrics registry access
- Base classes for common patterns
- Error handling utilities

Author: Claude Code
Date: 2025-07-27
"""

from prompt_improver.core.common.base_classes import (
    BaseConfigModel,
    BaseHealthChecker,
    BaseMonitor,
    BaseService,
)
from prompt_improver.core.common.config_utils import ConfigMixin, get_config_safely
from prompt_improver.core.common.error_handling import (
    ErrorCategory,
    InitializationError,
    handle_initialization_error,
    safe_config_load,
)
from prompt_improver.core.common.logging_utils import get_logger
from prompt_improver.core.common.metrics_utils import MetricsMixin, get_metrics_safely

__all__ = [
    "BaseConfigModel",
    "BaseHealthChecker",
    "BaseMonitor",
    "BaseService",
    "ConfigMixin",
    "ErrorCategory",
    "InitializationError",
    "MetricsMixin",
    "get_config_safely",
    "get_logger",
    "get_metrics_safely",
    "handle_initialization_error",
    "safe_config_load",
]
