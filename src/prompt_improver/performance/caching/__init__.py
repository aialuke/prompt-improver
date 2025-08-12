"""Performance-optimized caching layer for <200ms response times.

This module provides unified caching facades that integrate with existing
multi-level cache infrastructure for optimal performance.
"""

from .cache_facade import (
    PerformanceCacheFacade,
    get_performance_cache,
    CacheStrategy,
    CacheKey,
)
from .repository_cache import RepositoryCacheDecorator
from .ml_service_cache import MLServiceCache
from .api_cache import APIResponseCache

__all__ = [
    "PerformanceCacheFacade",
    "get_performance_cache", 
    "CacheStrategy",
    "CacheKey",
    "RepositoryCacheDecorator",
    "MLServiceCache",
    "APIResponseCache",
]