"""Lazy loading for coredis to prevent beartype/NumPy contamination.

This module provides lazy loading for coredis imports to prevent the automatic
loading of beartype's NumPy utilities which causes 1000ms+ startup penalty.

Usage:
    # Instead of: import coredis
    from prompt_improver.core.utils.lazy_coredis_loader import get_coredis

    def my_function():
        coredis = get_coredis()
        return coredis.Redis(...)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module cache
_COREDIS_CACHE: dict[str, Any] = {}


def get_coredis():
    """Lazy load coredis module."""
    if "coredis" in _COREDIS_CACHE:
        return _COREDIS_CACHE["coredis"]

    try:
        import coredis
        _COREDIS_CACHE["coredis"] = coredis
        return coredis
    except ImportError as e:
        logger.exception(f"coredis not available: {e}")
        raise ImportError("coredis is required. Install with: pip install coredis") from e


def get_coredis_redis():
    """Lazy load coredis.Redis class."""
    coredis = get_coredis()
    return coredis.Redis


def get_coredis_exceptions():
    """Lazy load coredis exceptions."""
    if "exceptions" in _COREDIS_CACHE:
        return _COREDIS_CACHE["exceptions"]

    coredis = get_coredis()
    exceptions = {
        'ConnectionError': coredis.exceptions.ConnectionError,
        'TimeoutError': coredis.exceptions.TimeoutError,
        'RedisError': coredis.exceptions.RedisError,
        'ResponseError': coredis.exceptions.ResponseError,
    }
    _COREDIS_CACHE["exceptions"] = exceptions
    return exceptions


# Convenience functions
def create_redis_client(*args, **kwargs):
    """Create Redis client using lazy loading."""
    Redis = get_coredis_redis()
    return Redis(*args, **kwargs)


def create_redis_from_url(url: str, **kwargs):
    """Create Redis client from URL using lazy loading."""
    Redis = get_coredis_redis()
    return Redis.from_url(url, **kwargs)


def clear_coredis_cache():
    """Clear the coredis cache (useful for testing)."""
    global _COREDIS_CACHE
    _COREDIS_CACHE.clear()
