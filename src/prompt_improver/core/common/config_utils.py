"""Centralized configuration utilities to eliminate duplicate config loading patterns.

Consolidates the patterns:
- try: config = get_config() except: fallback
- Duplicate validation logic across config classes
- Repeated configuration initialization
"""

import asyncio
from dataclasses import dataclass
from typing import Any, TypeVar

from prompt_improver.core.common.logging_utils import get_logger
from prompt_improver.services.cache.cache_facade import CacheFacade
from prompt_improver.services.cache.cache_factory import CacheFactory

logger = get_logger(__name__)
T = TypeVar("T")


def get_config_cache():
    """Get optimized config cache using singleton factory pattern.

    Resolves performance issues by using CacheFactory singleton
    instead of creating new instances per call.
    """
    return CacheFactory.get_utility_cache()


class ConfigMixin:
    """Mixin class providing configuration utilities."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize ConfigMixin."""
        super().__init__(*args, **kwargs)
        self._config = None

    @property
    def config(self) -> Any:
        """Get configuration, loading it if necessary."""
        if self._config is None:
            config_result = get_config_safely()
            self._config = config_result.config
        return self._config


@dataclass
class ConfigLoadResult:
    """Result of configuration loading operation."""

    config: Any | None
    success: bool
    error: str | None
    fallback_used: bool = False


def get_config_safely(use_fallback: bool = True) -> ConfigLoadResult:
    """Safely load configuration with fallback handling using unified cache.

    Consolidates the common pattern:
    try:
        config = get_config()
    except Exception as e:
        logger.warning(f"Failed to load config: {e}")
        config = fallback_config

    Args:
        use_fallback: Whether to use fallback values on failure

    Returns:
        ConfigLoadResult with loaded config or fallback
    """
    cache_key = f"util:config:main:fallback={use_fallback}"
    cache = get_config_cache()

    # Try to run in existing event loop or create new one
    try:
        loop = asyncio.get_running_loop()
        # Create task for async cache operation
        task = asyncio.create_task(_get_config_cached(cache_key, cache, use_fallback))
        # Run in current loop context with shorter timeout
        return asyncio.run_coroutine_threadsafe(task, loop).result(timeout=1.0)
    except RuntimeError:
        # No event loop, create one with timeout
        try:
            async def run_with_timeout():
                return await asyncio.wait_for(
                    _get_config_cached(cache_key, cache, use_fallback),
                    timeout=1.0
                )
            return asyncio.run(run_with_timeout())
        except Exception as e:
            logger.warning(f"Async cache operation failed: {e}")
            return _load_config_direct(use_fallback)
    except Exception as e:
        logger.warning(f"Cache operation failed: {e}")
        # Fallback to direct execution
        return _load_config_direct(use_fallback)


async def _get_config_cached(cache_key: str, cache: CacheFacade, use_fallback: bool) -> ConfigLoadResult:
    """Get config from cache or load if not cached."""
    try:
        # Try cache first
        cached_result = await cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Cache miss - load config and cache result
        result = _load_config_direct(use_fallback)

        # Cache for 4 hours with shorter L1 TTL for performance
        await cache.set(cache_key, result, l2_ttl=14400, l1_ttl=3600)

        return result
    except Exception as e:
        logger.exception(f"Config cache operation failed: {e}")
        return _load_config_direct(use_fallback)


def _load_config_direct(use_fallback: bool) -> ConfigLoadResult:
    """Load configuration directly without caching."""
    try:
        from prompt_improver.core.config import get_config

        config = get_config()
        return ConfigLoadResult(
            config=config, success=True, error=None, fallback_used=False
        )
    except Exception as e:
        error_msg = f"Failed to load centralized config: {e}"
        logger.warning(error_msg)
        if use_fallback:
            return ConfigLoadResult(
                config=None, success=False, error=error_msg, fallback_used=True
            )
        return ConfigLoadResult(
            config=None, success=False, error=error_msg, fallback_used=False
        )
