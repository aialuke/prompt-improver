"""Centralized logging utility to eliminate duplicate logger initialization patterns.

Consolidates the pattern: logger = logging.getLogger(__name__)
Found in 100+ files across the codebase.
"""
import logging
from functools import lru_cache
from typing import Any, Optional

@lru_cache(maxsize=128)
def get_logger(name: str | None=None, level: str | None=None) -> logging.Logger:
    """Get a configured logger instance with caching to avoid duplicate initialization.

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
    if name is None:
        frame = inspect.currentframe()
        try:
            if frame is not None:
                caller_frame = frame.f_back
                if caller_frame is not None:
                    name = caller_frame.f_globals.get('__name__', 'unknown')
                else:
                    name = 'unknown'
            else:
                name = 'unknown'
        finally:
            if frame is not None:
                del frame
    logger = logging.getLogger(name)
    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger

def configure_logging(level: str='INFO', format_string: str | None=None, include_timestamp: bool=True, include_module: bool=True) -> None:
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
            parts.append('%(asctime)s')
        parts.append('%(levelname)s')
        if include_module:
            parts.append('%(name)s')
        parts.append('%(message)s')
        format_string = ' - '.join(parts)
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=format_string, datefmt='%Y-%m-%d %H:%M:%S')

class LoggerMixin:
    """Mixin class to provide consistent logger access pattern.

    Eliminates the need for manual logger initialization in classes.
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(f'{self.__class__.__module__}.{self.__class__.__name__}')

    def log_method_entry(self, method_name: str, **kwargs) -> None:
        """Log method entry with parameters."""
        if kwargs:
            self.logger.debug('Entering {method_name} with: %s', kwargs)
        else:
            self.logger.debug('Entering %s', method_name)

    def log_method_exit(self, method_name: str, result: Any=None) -> None:
        """Log method exit with result."""
        if result is not None:
            self.logger.debug('Exiting %s with result: %s', method_name, type(result).__name__)
        else:
            self.logger.debug('Exiting %s', method_name)

    def log_error(self, method_name: str, error: Exception) -> None:
        """Log error with context."""
        self.logger.error('Error in {method_name}: {type(error).__name__}: %s', error)
