"""Concrete Retry Implementations - 2025 Best Practice

This module provides concrete implementations of the retry protocols defined
in retry_protocols.py. This separation allows protocol consumers to import
interfaces without circular dependencies, while implementations can be
imported separately when needed.

Key Benefits:
- Implements protocols without creating circular imports
- Allows dependency injection of different implementations
- Follows Interface Segregation Principle
- Type-safe with runtime validation
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .protocols.retry_protocols import (
    RetryConfigProtocol,
    RetryStrategy,
    RetryableErrorType,
    RetryConfigFactoryProtocol
)

logger = logging.getLogger(__name__)


@dataclass
class BasicRetryConfig:
    """
    Basic implementation of RetryConfigProtocol.
    
    This provides a concrete implementation that can be used by any component
    that needs retry functionality without creating circular dependencies.
    """
    
    # Basic retry settings
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    
    # Delay settings
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    
    # Advanced settings
    jitter: bool = True
    jitter_factor: float = 0.1  # 10% jitter
    backoff_multiplier: float = 2.0
    
    # Conditional retry settings
    retry_on_exceptions: List[type] = field(default_factory=lambda: [Exception])
    retry_condition: Optional[Callable[[Exception], bool]] = None
    
    # Timeout settings
    operation_timeout: Optional[float] = None  # seconds
    total_timeout: Optional[float] = None  # seconds
    
    # Logging and monitoring
    log_attempts: bool = True
    log_level: str = "INFO"
    track_metrics: bool = True
    
    # Circuit breaker integration
    circuit_breaker_enabled: bool = False
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        
        if self.base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        
        if not 0 <= self.jitter_factor <= 1:
            raise ValueError("jitter_factor must be between 0 and 1")
        
        if self.backoff_multiplier <= 1:
            raise ValueError("backoff_multiplier must be > 1")
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt number.
        
        Args:
            attempt: Attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        if attempt < 0:
            return 0.0
        
        if self.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.base_delay
        
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * (attempt + 1)
        
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (self.backoff_multiplier ** attempt)
        
        elif self.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = self.base_delay * self._fibonacci(attempt + 1)
        
        else:  # CUSTOM or unknown
            delay = self.base_delay
        
        # Apply max delay limit
        delay = min(delay, self.max_delay)
        
        # Apply jitter if enabled
        if self.jitter and delay > 0:
            jitter_amount = delay * self.jitter_factor
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)  # Ensure non-negative
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number"""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if an operation should be retried.
        
        Args:
            exception: The exception that occurred
            attempt: Current attempt number (0-based)
            
        Returns:
            True if should retry, False otherwise
        """
        # Check attempt limit
        if attempt >= self.max_attempts - 1:
            return False
        
        # Check exception type
        if not any(isinstance(exception, exc_type) for exc_type in self.retry_on_exceptions):
            return False
        
        # Check custom retry condition
        if self.retry_condition and not self.retry_condition(exception):
            return False
        
        return True


class StandardRetryConfigFactory:
    """
    Factory for creating standard retry configurations.
    
    This implements RetryConfigFactoryProtocol and provides predefined
    configurations for common use cases.
    """
    
    # Predefined configurations
    _STANDARD_CONFIGS = {
        "fast": BasicRetryConfig(
            max_attempts=3,
            base_delay=0.1,
            max_delay=2.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=True
        ),
        "medium": BasicRetryConfig(
            max_attempts=5,
            base_delay=1.0,
            max_delay=30.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=True
        ),
        "slow": BasicRetryConfig(
            max_attempts=3,
            base_delay=5.0,
            max_delay=300.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=True
        ),
        "critical": BasicRetryConfig(
            max_attempts=10,
            base_delay=0.5,
            max_delay=60.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=True,
            circuit_breaker_enabled=True
        ),
        "background": BasicRetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            jitter=True
        )
    }
    
    def create_config(
        self,
        operation_type: str = "medium",
        max_attempts: Optional[int] = None,
        base_delay: Optional[float] = None,
        **kwargs
    ) -> RetryConfigProtocol:
        """
        Create a retry configuration with sensible defaults.
        
        Args:
            operation_type: Type of operation ("fast", "medium", "slow", "critical", "background")
            max_attempts: Override max attempts
            base_delay: Override base delay
            **kwargs: Additional configuration options
            
        Returns:
            RetryConfigProtocol instance
        """
        # Get base configuration
        base_config = self._STANDARD_CONFIGS.get(operation_type, self._STANDARD_CONFIGS["medium"])
        
        # Create new config with overrides
        config_dict = {
            "max_attempts": max_attempts or base_config.max_attempts,
            "base_delay": base_delay or base_config.base_delay,
            "max_delay": base_config.max_delay,
            "strategy": base_config.strategy,
            "jitter": base_config.jitter,
            "jitter_factor": base_config.jitter_factor,
            "backoff_multiplier": base_config.backoff_multiplier,
            "retry_on_exceptions": base_config.retry_on_exceptions.copy(),
            "retry_condition": base_config.retry_condition,
            "operation_timeout": base_config.operation_timeout,
            "total_timeout": base_config.total_timeout,
            "log_attempts": base_config.log_attempts,
            "log_level": base_config.log_level,
            "track_metrics": base_config.track_metrics,
            "circuit_breaker_enabled": base_config.circuit_breaker_enabled,
            "failure_threshold": base_config.failure_threshold,
            "recovery_timeout": base_config.recovery_timeout,
        }
        
        # Apply overrides
        config_dict.update(kwargs)
        
        return BasicRetryConfig(**config_dict)
    
    def get_standard_config(self, config_name: str) -> RetryConfigProtocol:
        """
        Get a predefined standard configuration.
        
        Args:
            config_name: Name of the standard configuration
            
        Returns:
            RetryConfigProtocol instance
            
        Raises:
            ValueError: If config_name is not found
        """
        if config_name not in self._STANDARD_CONFIGS:
            available = ", ".join(self._STANDARD_CONFIGS.keys())
            raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
        
        return self._STANDARD_CONFIGS[config_name]


# Global factory instance for convenience
_default_factory = StandardRetryConfigFactory()

def create_retry_config(
    operation_type: str = "medium",
    max_attempts: Optional[int] = None,
    base_delay: Optional[float] = None,
    **kwargs
) -> RetryConfigProtocol:
    """
    Convenience function to create retry configurations.
    
    This provides a simple interface for creating retry configurations
    without needing to instantiate the factory directly.
    """
    return _default_factory.create_config(
        operation_type=operation_type,
        max_attempts=max_attempts,
        base_delay=base_delay,
        **kwargs
    )

def get_standard_retry_config(config_name: str) -> RetryConfigProtocol:
    """
    Convenience function to get standard retry configurations.
    """
    return _default_factory.get_standard_config(config_name)


# Backward compatibility aliases
RetryConfig = BasicRetryConfig  # For existing code that imports RetryConfig
