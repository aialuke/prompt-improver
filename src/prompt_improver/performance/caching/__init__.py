"""Performance-optimized caching layer for <200ms response times.

This module provides unified caching facades that integrate with existing
multi-level cache infrastructure for optimal performance.
"""

# MIGRATION: Performance cache facade moved to unified services/cache/
from prompt_improver.performance.caching.repository_cache import (
    RepositoryCacheDecorator,
)
from prompt_improver.services.cache.cache_facade import (
    CacheFacade as PerformanceCacheFacade,
)

__all__ = [
    "PerformanceCacheFacade",
    "RepositoryCacheDecorator",
]
