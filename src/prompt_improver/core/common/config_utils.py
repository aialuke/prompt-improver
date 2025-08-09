"""Centralized configuration utilities to eliminate duplicate config loading patterns.

Consolidates the patterns:
- try: config = get_config() except: fallback
- Duplicate validation logic across config classes
- Repeated configuration initialization
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Type, TypeVar, Union, cast
from prompt_improver.core.common.logging_utils import LoggerMixin, get_logger
logger = get_logger(__name__)
T = TypeVar('T')

@dataclass
class ConfigLoadResult:
    """Result of configuration loading operation."""
    config: Any | None
    success: bool
    error: str | None
    fallback_used: bool = False

@lru_cache(maxsize=1)
def get_config_safely(use_fallback: bool=True) -> ConfigLoadResult:
    """Safely load configuration with fallback handling.

    Consolidates the common pattern:
    try:
        config = get_config()
    except Exception as e:
        logger.warning("Failed to load config: %s", e)
        config = fallback_config

    Args:
        use_fallback: Whether to use fallback values on failure

    Returns:
        ConfigLoadResult with loaded config or fallback
    """
    try:
        from prompt_improver.core.config import get_config
        config = get_config()
        return ConfigLoadResult(config=config, success=True, error=None, fallback_used=False)
    except Exception as e:
        error_msg = f'Failed to load centralized config: {e}'
        logger.warning(error_msg)
        if use_fallback:
            return ConfigLoadResult(config=None, success=False, error=error_msg, fallback_used=True)
        return ConfigLoadResult(config=None, success=False, error=error_msg, fallback_used=False)
