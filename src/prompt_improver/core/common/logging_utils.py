"""Centralized logging utility to eliminate duplicate logger initialization patterns.

Consolidates the pattern: logger = logging.getLogger(__name__)
Found in 100+ files across the codebase.
"""

import asyncio
import logging
from typing import Any

from prompt_improver.services.cache.cache_facade import CacheFacade
from prompt_improver.services.cache.cache_factory import CacheFactory


def get_logging_cache():
    """Get optimized logging cache using singleton factory pattern.

    Resolves performance issues by using CacheFactory singleton
    instead of creating new instances per call.
    """
    return CacheFactory.get_utility_cache()


def get_logger(name: str | None = None, level: str | None = None) -> logging.Logger:
    """Get a configured logger instance with caching to avoid duplicate initialization using unified cache.

    Args:
        name: Logger name (defaults to caller's __name__)
        level: Optional log level override

    Returns:
        Configured logger instance

    Example:
        # Old pattern (duplicated everywhere):
        logger = logging.getLogger(__name__)

        # New pattern (consolidated):
        from prompt_improver.core.common import get_logger
        logger = get_logger(__name__)
    """
    import inspect

    # Determine logger name if not provided
    if name is None:
        frame = inspect.currentframe()
        try:
            if frame is not None:
                caller_frame = frame.f_back
                if caller_frame is not None:
                    name = caller_frame.f_globals.get("__name__", "unknown")
                else:
                    name = "unknown"
            else:
                name = "unknown"
        finally:
            if frame is not None:
                del frame

    # Create cache key that includes level to ensure different levels create different loggers
    cache_key = f"util:logging:{name}:{level or 'default'}"
    cache = get_logging_cache()

    # If cache is not available, use direct execution - critical for logging
    if cache is None:
        return _create_logger_direct(name, level)

    # Try to run in existing event loop or create new one
    try:
        loop = asyncio.get_running_loop()
        # Create task for async cache operation
        task = asyncio.create_task(_get_logger_cached(cache_key, cache, name, level))
        # Run in current loop context with shorter timeout
        return asyncio.run_coroutine_threadsafe(task, loop).result(timeout=0.5)
    except RuntimeError:
        # No event loop, create one with timeout
        try:
            async def run_with_timeout():
                return await asyncio.wait_for(
                    _get_logger_cached(cache_key, cache, name, level),
                    timeout=0.5
                )
            return asyncio.run(run_with_timeout())
        except Exception as e:
            # Fallback to direct execution for logging - must be reliable
            return _create_logger_direct(name, level)
    except Exception as e:
        # Fallback to direct execution for logging - must be reliable
        return _create_logger_direct(name, level)


async def _get_logger_cached(cache_key: str, cache: CacheFacade, name: str, level: str | None) -> logging.Logger:
    """Get logger from cache or create if not cached."""
    try:
        # Try cache first
        cached_logger = await cache.get(cache_key)
        if cached_logger is not None:
            return cached_logger

        # Cache miss - create logger and cache result
        logger_instance = _create_logger_direct(name, level)

        # Cache for 1 hour with shorter L1 TTL for performance
        await cache.set(cache_key, logger_instance, l2_ttl=3600, l1_ttl=900)

        return logger_instance
    except Exception as e:
        # For logging, we must have a fallback that always works
        return _create_logger_direct(name, level)


def _create_logger_direct(name: str, level: str | None) -> logging.Logger:
    """Create logger directly without caching."""
    logger_instance = logging.getLogger(name)
    if level:
        logger_instance.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger_instance


def configure_logging(
    level: str = "INFO",
    format_string: str | None = None,
    include_timestamp: bool = True,
    include_module: bool = True,
) -> None:
    """Configure logging for the entire application.

    Args:
        level: Default logging level
        format_string: Custom format string
        include_timestamp: Include timestamp in logs
        include_module: Include module name in logs
    """
    if format_string is None:
        parts = []
        if include_timestamp:
            parts.append("%(asctime)s")
        parts.append("%(levelname)s")
        if include_module:
            parts.append("%(name)s")
        parts.append("%(message)s")
        format_string = " - ".join(parts)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class LoggerMixin:
    """Mixin class to provide consistent logger access pattern.

    Eliminates the need for manual logger initialization in classes.
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def log_method_entry(self, method_name: str, **kwargs) -> None:
        """Log method entry with parameters."""
        if kwargs:
            self.logger.debug(f"Entering {method_name} with: {kwargs}")
        else:
            self.logger.debug(f"Entering {method_name}")

    def log_method_exit(self, method_name: str, result: Any = None) -> None:
        """Log method exit with result."""
        if result is not None:
            self.logger.debug(
                f"Exiting {method_name} with result: {type(result).__name__}"
            )
        else:
            self.logger.debug(f"Exiting {method_name}")

    def log_error(self, method_name: str, error: Exception) -> None:
        """Log error with context."""
        self.logger.error(f"Error in {method_name}: {type(error).__name__}: {error}")
