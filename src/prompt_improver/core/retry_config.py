"""Shared Retry Configuration - Circular Import Safe

This module provides shared retry configuration classes that can be imported
by both ML orchestration and performance monitoring without circular dependencies.

2025 Best Practice: Separate configuration from implementation to prevent circular imports.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import timedelta

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy enumeration"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    CUSTOM = "custom"


@dataclass
class RetryConfig:
    """
    Unified retry configuration for all system components.
    
    This configuration is used across ML orchestration, performance monitoring,
    and other system components to ensure consistent retry behavior.
    """
    
    # Basic retry settings
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    
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
            import random
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


# Predefined retry configurations for common use cases
class StandardRetryConfigs:
    """Standard retry configurations for common scenarios"""
    
    # Fast operations (API calls, database queries)
    FAST_OPERATION = RetryConfig(
        max_attempts=3,
        base_delay=0.1,
        max_delay=2.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        jitter=True
    )
    
    # Medium operations (file I/O, network requests)
    MEDIUM_OPERATION = RetryConfig(
        max_attempts=5,
        base_delay=1.0,
        max_delay=30.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        jitter=True
    )
    
    # Slow operations (ML training, large data processing)
    SLOW_OPERATION = RetryConfig(
        max_attempts=3,
        base_delay=5.0,
        max_delay=300.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        jitter=True
    )
    
    # Critical operations (system startup, health checks)
    CRITICAL_OPERATION = RetryConfig(
        max_attempts=10,
        base_delay=0.5,
        max_delay=60.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        jitter=True,
        circuit_breaker_enabled=True
    )
    
    # Background tasks
    BACKGROUND_TASK = RetryConfig(
        max_attempts=5,
        base_delay=2.0,
        max_delay=120.0,
        strategy=RetryStrategy.LINEAR_BACKOFF,
        jitter=True
    )


# Utility functions
def create_retry_config(
    operation_type: str = "medium",
    max_attempts: Optional[int] = None,
    base_delay: Optional[float] = None,
    **kwargs
) -> RetryConfig:
    """
    Create a retry configuration with sensible defaults.
    
    Args:
        operation_type: Type of operation ("fast", "medium", "slow", "critical", "background")
        max_attempts: Override max attempts
        base_delay: Override base delay
        **kwargs: Additional configuration options
        
    Returns:
        RetryConfig instance
    """
    # Get base configuration
    base_configs = {
        "fast": StandardRetryConfigs.FAST_OPERATION,
        "medium": StandardRetryConfigs.MEDIUM_OPERATION,
        "slow": StandardRetryConfigs.SLOW_OPERATION,
        "critical": StandardRetryConfigs.CRITICAL_OPERATION,
        "background": StandardRetryConfigs.BACKGROUND_TASK,
    }
    
    base_config = base_configs.get(operation_type, StandardRetryConfigs.MEDIUM_OPERATION)
    
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
    
    return RetryConfig(**config_dict)
