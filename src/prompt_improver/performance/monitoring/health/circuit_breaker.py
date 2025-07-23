"""
Circuit Breaker Pattern Implementation for Health Checkers
Following 2025 best practices for failure isolation
"""

import asyncio
import time
from enum import Enum
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject all calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: int = 60  # Seconds before trying half-open
    half_open_max_calls: int = 3  # Test calls in half-open state
    reset_timeout: int = 120  # Full reset after success streak

    # 2025 addition: SLA-based thresholds
    response_time_threshold_ms: float = 1000  # Consider slow response as failure
    success_rate_threshold: float = 0.95  # Minimum success rate to stay closed


class CircuitBreaker:
    """
    Modern circuit breaker implementation with 2025 observability patterns
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[str, CircuitState], None]] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change

        # State management
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

        # 2025 metrics tracking
        self._call_metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "rejected_calls": 0,
            "total_response_time_ms": 0
        }

    @property
    def state(self) -> CircuitState:
        """Get current circuit state with automatic transition logic"""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery"""
        return (
            self._last_failure_time is not None and
            time.time() - self._last_failure_time >= self.config.recovery_timeout
        )

    def _transition_to(self, new_state: CircuitState):
        """Transition to new state with logging and callback"""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state

            logger.info(
                f"Circuit breaker '{self.name}' transitioned: {old_state.value} -> {new_state.value}",
                extra={
                    "circuit_breaker": self.name,
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                    "failure_count": self._failure_count,
                    "metrics": self._call_metrics
                }
            )

            if self.on_state_change:
                self.on_state_change(self.name, new_state)

            # Reset counters based on state
            if new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0
            elif new_state == CircuitState.CLOSED:
                self._failure_count = 0
                self._half_open_calls = 0

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection

        Args:
            func: Async function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Function result

        Raises:
            CircuitBreakerOpen: If circuit is open
            Original exception: If function fails
        """
        # Check if we should reject immediately
        if self.state == CircuitState.OPEN:
            self._call_metrics["rejected_calls"] += 1
            raise CircuitBreakerOpen(
                f"Circuit breaker '{self.name}' is OPEN. Service is unavailable."
            )

        # Track call metrics
        self._call_metrics["total_calls"] += 1
        start_time = time.time()

        try:
            # Execute the function
            result = await func(*args, **kwargs)

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            self._call_metrics["total_response_time_ms"] += response_time_ms

            # Check if response time exceeds SLA
            if response_time_ms > self.config.response_time_threshold_ms:
                logger.warning(
                    f"Circuit breaker '{self.name}' detected slow response",
                    extra={
                        "response_time_ms": response_time_ms,
                        "threshold_ms": self.config.response_time_threshold_ms
                    }
                )
                # Record as successful call but treat as failure for circuit breaker
                self._record_success()
                self._record_slow_response()
            else:
                self._record_success()

            return result

        except Exception as e:
            self._record_failure()
            raise

    def _record_success(self):
        """Record successful call and update state"""
        self._call_metrics["successful_calls"] += 1

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls >= self.config.half_open_max_calls:
                # Enough successful calls, close the circuit
                self._transition_to(CircuitState.CLOSED)
        elif self._state == CircuitState.CLOSED:
            # Check success rate for SLA monitoring
            success_rate = self._calculate_success_rate()
            if success_rate < self.config.success_rate_threshold:
                logger.warning(
                    f"Circuit breaker '{self.name}' success rate below threshold",
                    extra={
                        "success_rate": success_rate,
                        "threshold": self.config.success_rate_threshold
                    }
                )

    def _record_failure(self, is_timeout: bool = False):
        """Record failed call and update state"""
        self._call_metrics["failed_calls"] += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Failed during recovery attempt, reopen
            self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def _record_slow_response(self):
        """Record slow response for circuit breaker failure counting"""
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Slow response during recovery attempt, reopen
            self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def _calculate_success_rate(self) -> float:
        """Calculate current success rate"""
        total = self._call_metrics["total_calls"]
        if total == 0:
            return 1.0
        return self._call_metrics["successful_calls"] / total

    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics for monitoring"""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_rate": self._calculate_success_rate(),
            "avg_response_time_ms": (
                self._call_metrics["total_response_time_ms"] /
                max(1, self._call_metrics["successful_calls"])
            ),
            **self._call_metrics
        }

    def reset(self):
        """Manually reset the circuit breaker"""
        self._transition_to(CircuitState.CLOSED)
        self._failure_count = 0
        self._last_failure_time = None
        logger.info(f"Circuit breaker '{self.name}' manually reset")


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open"""
    pass


# Circuit breaker registry for centralized management
class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker"""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)
        return self._breakers[name]

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all circuit breakers"""
        return {
            name: breaker.get_metrics()
            for name, breaker in self._breakers.items()
        }

    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self._breakers.values():
            breaker.reset()


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()