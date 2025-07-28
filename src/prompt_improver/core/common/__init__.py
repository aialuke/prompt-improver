"""
Common utilities and shared patterns to eliminate code duplication.

This module provides centralized implementations for frequently duplicated patterns:
- Logger initialization
- Configuration loading
- Metrics registry access
- Base classes for common patterns
- Error handling utilities

Author: Claude Code
Date: 2025-07-27
"""

from .logging_utils import get_logger
from .config_utils import ConfigMixin, get_config_safely
from .metrics_utils import MetricsMixin, get_metrics_safely
from .base_classes import (
    BaseHealthChecker,
    BaseConfigModel,
    BaseService,
    BaseMonitor
)
from .error_handling import (
    safe_config_load,
    handle_initialization_error,
    ErrorCategory,
    InitializationError
)

__all__ = [
    # Logging utilities
    'get_logger',
    
    # Configuration utilities
    'ConfigMixin',
    'get_config_safely',
    
    # Metrics utilities
    'MetricsMixin', 
    'get_metrics_safely',
    
    # Base classes
    'BaseHealthChecker',
    'BaseConfigModel',
    'BaseService',
    'BaseMonitor',
    
    # Error handling
    'safe_config_load',
    'handle_initialization_error',
    'ErrorCategory',
    'InitializationError'
]