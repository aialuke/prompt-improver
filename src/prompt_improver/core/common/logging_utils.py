"""
Centralized logging utility to eliminate duplicate logger initialization patterns.

Consolidates the pattern: logger = logging.getLogger(__name__)
Found in 100+ files across the codebase.
"""

import logging
from typing import Optional, Any
from functools import lru_cache

@lru_cache(maxsize=128)
def get_logger(name: Optional[str] = None, level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance with caching to avoid duplicate initialization.
    
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
        # Get caller's module name automatically
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back
            name = caller_frame.f_globals.get('__name__', 'unknown')
        finally:
            del frame
    
    logger = logging.getLogger(name)
    
    # Set level if specified
    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    return logger

def configure_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
    include_module: bool = True
) -> None:
    """
    Configure logging for the entire application.
    
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
        datefmt="%Y-%m-%d %H:%M:%S"
    )

class LoggerMixin:
    """
    Mixin class to provide consistent logger access pattern.
    
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
            self.logger.debug(f"Exiting {method_name} with result: {type(result).__name__}")
        else:
            self.logger.debug(f"Exiting {method_name}")
    
    def log_error(self, method_name: str, error: Exception) -> None:
        """Log error with context."""
        self.logger.error(f"Error in {method_name}: {type(error).__name__}: {error}")