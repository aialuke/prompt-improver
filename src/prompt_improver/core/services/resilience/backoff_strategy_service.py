"""BackoffStrategyService - High-Performance Delay Calculation Algorithms.

This service provides optimized delay calculation algorithms for retry operations,
focusing on sub-millisecond performance and extensibility.

Key Features:
- <1ms execution performance for all strategy calculations
- Pre-calculated Fibonacci sequences for O(1) lookups
- Intelligent jitter distribution with configurable factors
- Support for custom delay functions with validation
- Strategy-specific optimization and caching
- Performance metrics and validation capabilities

Architecture:
- Protocol-based interface implementation (BackoffStrategyProtocol)
- Pure function implementations for maximum performance
- Pre-computed lookup tables for complex sequences
- Zero-allocation jitter calculations using thread-safe random
- Strategy validation and optimization suggestions

Usage:
    service = BackoffStrategyService()
    delay = service.calculate_delay(strategy, attempt=3, base_delay=1.0, **kwargs)

    # Custom strategy
    service.register_custom_strategy("custom", custom_delay_func)
    delay = service.calculate_delay("custom", attempt=3, base_delay=1.0)
"""

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from prompt_improver.services.cache.cache_facade import get_cache
from prompt_improver.shared.interfaces.protocols.cache import CacheType
from prompt_improver.shared.interfaces.protocols.core import RetryStrategy

logger = logging.getLogger(__name__)


@runtime_checkable
class BackoffStrategyProtocol(Protocol):
    """Protocol for backoff strategy service implementations."""

    def calculate_delay(
        self,
        strategy: RetryStrategy | str,
        attempt: int,
        base_delay: float,
        max_delay: float = 60.0,
        jitter: bool = True,
        jitter_factor: float = 0.1,
        backoff_multiplier: float = 2.0,
        **kwargs: Any,
    ) -> float:
        """Calculate delay for a given retry attempt."""
        ...

    def register_custom_strategy(
        self, name: str, delay_func: Callable[[int, float, dict[str, Any]], float]
    ) -> None:
        """Register a custom delay calculation function."""
        ...

    def validate_strategy(self, strategy: RetryStrategy | str, **params: Any) -> bool:
        """Validate strategy configuration and parameters."""
        ...

    def get_strategy_optimization(self, strategy: RetryStrategy | str) -> dict[str, Any]:
        """Get optimization recommendations for a strategy."""
        ...


@dataclass(frozen=True)
class StrategyMetrics:
    """Performance metrics for strategy calculations."""

    strategy_name: str
    calculation_count: int = 0
    total_time_ns: int = 0
    min_time_ns: int = float('inf')
    max_time_ns: int = 0
    avg_time_ns: float = 0.0

    def add_measurement(self, duration_ns: int) -> 'StrategyMetrics':
        """Add a new timing measurement (returns new instance for immutability)."""
        new_count = self.calculation_count + 1
        new_total = self.total_time_ns + duration_ns
        return StrategyMetrics(
            strategy_name=self.strategy_name,
            calculation_count=new_count,
            total_time_ns=new_total,
            min_time_ns=min(self.min_time_ns, duration_ns),
            max_time_ns=max(self.max_time_ns, duration_ns),
            avg_time_ns=new_total / new_count,
        )


class BackoffStrategyService:
    """High-performance backoff strategy calculation service.

    Provides optimized delay calculations for retry operations with focus
    on sub-millisecond performance and extensive customization options.
    """

    # Pre-calculated Fibonacci sequence for O(1) lookups (up to attempt 50)
    _FIBONACCI_CACHE = [0, 1]
    _MAX_FIBONACCI_CACHE = 50

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._custom_strategies: dict[str, Callable[[int, float, dict[str, Any]], float]] = {}
        self._strategy_metrics: dict[str, StrategyMetrics] = {}
        self._random_state = random.Random()  # Thread-local random state
        self._initialize_fibonacci_cache()
        self.logger.info("BackoffStrategyService initialized with optimized algorithms")

    def _initialize_fibonacci_cache(self) -> None:
        """Pre-calculate Fibonacci sequence for performance optimization."""
        # Extend cache to MAX_FIBONACCI_CACHE for O(1) lookups
        while len(self._FIBONACCI_CACHE) <= self._MAX_FIBONACCI_CACHE:
            next_fib = self._FIBONACCI_CACHE[-1] + self._FIBONACCI_CACHE[-2]
            self._FIBONACCI_CACHE.append(next_fib)

        self.logger.debug(f"Pre-calculated Fibonacci cache up to position {self._MAX_FIBONACCI_CACHE}")

    def calculate_delay(
        self,
        strategy: RetryStrategy | str,
        attempt: int,
        base_delay: float,
        max_delay: float = 60.0,
        jitter: bool = True,
        jitter_factor: float = 0.1,
        backoff_multiplier: float = 2.0,
        **kwargs: Any,
    ) -> float:
        """Calculate delay for a retry attempt with optimal performance.

        Args:
            strategy: Retry strategy to use
            attempt: Attempt number (0-based)
            base_delay: Base delay in seconds
            max_delay: Maximum delay cap in seconds
            jitter: Whether to apply jitter
            jitter_factor: Jitter factor (0.0-1.0)
            backoff_multiplier: Multiplier for exponential backoff
            **kwargs: Additional strategy-specific parameters

        Returns:
            Calculated delay in seconds
        """
        start_time = time.perf_counter_ns()

        try:
            # Input validation (optimized for common cases)
            if attempt < 0:
                return 0.0
            if base_delay <= 0:
                return 0.0

            # Convert string strategy to enum if needed
            if isinstance(strategy, str):
                if strategy in self._custom_strategies:
                    delay = self._calculate_custom_delay(strategy, attempt, base_delay, kwargs)
                else:
                    try:
                        strategy = RetryStrategy(strategy)
                    except ValueError:
                        self.logger.warning(f"Unknown strategy '{strategy}', using FIXED_DELAY")
                        strategy = RetryStrategy.FIXED_DELAY

            # Calculate base delay using optimized algorithms
            if isinstance(strategy, RetryStrategy):
                delay = self._calculate_builtin_delay(strategy, attempt, base_delay, backoff_multiplier)
            else:
                delay = base_delay  # Fallback

            # Apply max_delay cap
            delay = min(delay, max_delay)

            # Apply jitter with optimized calculation
            if jitter and delay > 0 and jitter_factor > 0:
                delay = self._apply_jitter(delay, jitter_factor)

            return max(0.0, delay)

        finally:
            # Record performance metrics
            duration_ns = time.perf_counter_ns() - start_time
            self._record_metrics(str(strategy), duration_ns)

    def _calculate_builtin_delay(
        self, strategy: RetryStrategy, attempt: int, base_delay: float, backoff_multiplier: float
    ) -> float:
        """Calculate delay for built-in strategies with maximum optimization."""
        if strategy == RetryStrategy.FIXED_DELAY:
            return base_delay

        if strategy == RetryStrategy.LINEAR_BACKOFF:
            return base_delay * (attempt + 1)

        if strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            # Optimized for common small exponents
            if attempt <= 10:
                return base_delay * (backoff_multiplier ** attempt)
            # Use pow for larger exponents to prevent overflow
            return base_delay * pow(backoff_multiplier, min(attempt, 20))

        if strategy == RetryStrategy.FIBONACCI_BACKOFF:
            return base_delay * self._get_fibonacci(attempt + 1)

        # Default fallback
        return base_delay

    def _calculate_custom_delay(
        self, strategy_name: str, attempt: int, base_delay: float, kwargs: dict[str, Any]
    ) -> float:
        """Calculate delay using custom strategy function."""
        strategy_func = self._custom_strategies[strategy_name]
        try:
            return strategy_func(attempt, base_delay, kwargs)
        except Exception as e:
            self.logger.exception(f"Custom strategy '{strategy_name}' failed: {e}, using base_delay")
            return base_delay

    def _get_fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number with O(1) lookup for cached values."""
        if n < 0:
            return 0
        if n < len(self._FIBONACCI_CACHE):
            return self._FIBONACCI_CACHE[n]

        # Calculate for values beyond cache (should be rare)
        return self._calculate_fibonacci_iterative(n)

    async def _calculate_fibonacci_iterative_async(self, n: int) -> int:
        """Calculate Fibonacci number iteratively for large n (unified cache)."""
        cache = get_cache(CacheType.APPLICATION)
        cache_key = f"fibonacci_iterative:{n}"

        def compute_fibonacci() -> int:
            """Internal computation function for cache fallback."""
            if n <= 1:
                return n

            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b

        return await cache.get_or_set(
            cache_key,
            compute_fibonacci,
            l1_ttl=3600,  # Cache for 1 hour - mathematical results are stable
            l2_ttl=7200   # Longer TTL in L2 for frequently accessed values
        )

    def _calculate_fibonacci_iterative(self, n: int) -> int:
        """Calculate Fibonacci number iteratively for large n (cached).

        Backward compatible sync wrapper using unified cache infrastructure.
        """
        try:
            # Check if we're in an async context
            loop = asyncio.get_running_loop()
            # If we're already in an event loop, we need to use a different approach
            # For now, fall back to direct computation to avoid blocking the loop
            logger.warning(f"Fibonacci calculation for n={n} called from async context, using direct computation")
            if n <= 1:
                return n
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b
        except RuntimeError:
            # No running event loop, safe to create one
            return asyncio.run(self._calculate_fibonacci_iterative_async(n))

    def _apply_jitter(self, delay: float, jitter_factor: float) -> float:
        """Apply jitter with optimized random calculation."""
        # Optimized jitter calculation - single random call
        jitter_amount = delay * jitter_factor
        # Use uniform distribution centered on original delay
        jitter_delta = self._random_state.uniform(-jitter_amount, jitter_amount)
        return delay + jitter_delta

    def register_custom_strategy(
        self, name: str, delay_func: Callable[[int, float, dict[str, Any]], float]
    ) -> None:
        """Register a custom delay calculation function.

        Args:
            name: Strategy name identifier
            delay_func: Function that takes (attempt, base_delay, kwargs) and returns delay
        """
        if not callable(delay_func):
            raise ValueError("delay_func must be callable")

        # Validate function signature by testing with safe values
        try:
            test_result = delay_func(0, 1.0, {})
            if not isinstance(test_result, (int, float)) or test_result < 0:
                raise ValueError("delay_func must return non-negative number")
        except Exception as e:
            raise ValueError(f"delay_func validation failed: {e}")

        self._custom_strategies[name] = delay_func
        self.logger.info(f"Registered custom strategy: {name}")

    def validate_strategy(self, strategy: RetryStrategy | str, **params: Any) -> bool:
        """Validate strategy configuration and parameters.

        Args:
            strategy: Strategy to validate
            **params: Strategy parameters to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate strategy existence
            if isinstance(strategy, str):
                if strategy not in self._custom_strategies:
                    try:
                        RetryStrategy(strategy)
                    except ValueError:
                        return False

            # Validate common parameters
            base_delay = params.get('base_delay', 1.0)
            max_delay = params.get('max_delay', 60.0)
            jitter_factor = params.get('jitter_factor', 0.1)
            backoff_multiplier = params.get('backoff_multiplier', 2.0)

            if not isinstance(base_delay, (int, float)) or base_delay < 0:
                return False
            if not isinstance(max_delay, (int, float)) or max_delay < base_delay:
                return False
            if not isinstance(jitter_factor, (int, float)) or not 0 <= jitter_factor <= 1:
                return False
            return not (not isinstance(backoff_multiplier, (int, float)) or backoff_multiplier <= 1)

        except Exception as e:
            self.logger.debug(f"Strategy validation failed: {e}")
            return False

    def get_strategy_optimization(self, strategy: RetryStrategy | str) -> dict[str, Any]:
        """Get optimization recommendations for a strategy.

        Args:
            strategy: Strategy to analyze

        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {
            "strategy": str(strategy),
            "performance_class": "unknown",
            "recommended_max_attempts": 10,
            "suggested_base_delay": 1.0,
            "suggested_max_delay": 60.0,
            "jitter_recommended": True,
            "notes": []
        }

        if isinstance(strategy, RetryStrategy):
            if strategy == RetryStrategy.FIXED_DELAY:
                recommendations.update({
                    "performance_class": "optimal",
                    "recommended_max_attempts": 5,
                    "suggested_base_delay": 2.0,
                    "notes": ["Simplest strategy with constant time complexity"]
                })

            elif strategy == RetryStrategy.LINEAR_BACKOFF:
                recommendations.update({
                    "performance_class": "excellent",
                    "recommended_max_attempts": 8,
                    "suggested_base_delay": 1.0,
                    "notes": ["Good for predictable delays, linear growth"]
                })

            elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
                recommendations.update({
                    "performance_class": "good",
                    "recommended_max_attempts": 6,
                    "suggested_base_delay": 0.5,
                    "suggested_max_delay": 30.0,
                    "notes": ["Excellent for network operations, rapid backoff growth"]
                })

            elif strategy == RetryStrategy.FIBONACCI_BACKOFF:
                recommendations.update({
                    "performance_class": "good",
                    "recommended_max_attempts": 7,
                    "suggested_base_delay": 1.0,
                    "notes": ["Balanced growth rate, good for API rate limiting"]
                })

        # Add performance metrics if available
        if str(strategy) in self._strategy_metrics:
            metrics = self._strategy_metrics[str(strategy)]
            recommendations["performance_metrics"] = {
                "avg_time_microseconds": metrics.avg_time_ns / 1000,
                "calculation_count": metrics.calculation_count,
                "performance_rating": "excellent" if metrics.avg_time_ns < 1000 else "good"
            }

        return recommendations

    def _record_metrics(self, strategy_name: str, duration_ns: int) -> None:
        """Record performance metrics for strategy calculations."""
        if strategy_name not in self._strategy_metrics:
            self._strategy_metrics[strategy_name] = StrategyMetrics(strategy_name=strategy_name)

        self._strategy_metrics[strategy_name] = self._strategy_metrics[strategy_name].add_measurement(duration_ns)

    def get_performance_metrics(self) -> dict[str, dict[str, Any]]:
        """Get comprehensive performance metrics for all strategies."""
        return {
            name: {
                "calculation_count": metrics.calculation_count,
                "avg_time_microseconds": metrics.avg_time_ns / 1000,
                "min_time_microseconds": metrics.min_time_ns / 1000,
                "max_time_microseconds": metrics.max_time_ns / 1000,
                "total_time_milliseconds": metrics.total_time_ns / 1_000_000,
            }
            for name, metrics in self._strategy_metrics.items()
        }

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self._strategy_metrics.clear()
        self.logger.info("Performance metrics reset")

    def list_available_strategies(self) -> dict[str, list[str]]:
        """List all available strategies."""
        builtin = [strategy.value for strategy in RetryStrategy]
        custom = list(self._custom_strategies.keys())

        return {
            "builtin_strategies": builtin,
            "custom_strategies": custom,
            "total_count": len(builtin) + len(custom)
        }


# Global service instance
_global_service: BackoffStrategyService | None = None


def get_backoff_strategy_service() -> BackoffStrategyService:
    """Get the global backoff strategy service instance."""
    global _global_service
    if _global_service is None:
        _global_service = BackoffStrategyService()
    return _global_service


def calculate_delay(
    strategy: RetryStrategy | str,
    attempt: int,
    base_delay: float,
    **kwargs: Any,
) -> float:
    """Convenience function for delay calculation."""
    return get_backoff_strategy_service().calculate_delay(
        strategy, attempt, base_delay, **kwargs
    )


__all__ = [
    "BackoffStrategyProtocol",
    "BackoffStrategyService",
    "StrategyMetrics",
    "calculate_delay",
    "get_backoff_strategy_service",
]
