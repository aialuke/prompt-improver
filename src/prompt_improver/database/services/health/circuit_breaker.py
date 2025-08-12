"""Circuit breaker implementation for fault tolerance.

This module provides circuit breaker functionality extracted from
unified_connection_manager.py, implementing:

- CircuitBreaker: State-based fault tolerance with automatic recovery
- CircuitBreakerConfig: Configurable thresholds and timeouts
- CircuitBreakerState: Open/Closed/Half-Open states with transitions
- Failure counting and recovery mechanisms

Designed for production resilience with configurable failure thresholds.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, failures counted
    OPEN = "open"  # Failing fast, not calling function
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    # Failure threshold to open circuit
    failure_threshold: int = 5

    # Recovery timeout in seconds
    recovery_timeout_seconds: float = 60.0

    # Success threshold in half-open state to close circuit
    success_threshold: int = 3

    # Maximum response time in milliseconds before considering as failure
    timeout_ms: float = 5000.0

    # Exponential backoff multiplier for recovery timeout
    backoff_multiplier: float = 1.5

    # Maximum recovery timeout in seconds
    max_recovery_timeout_seconds: float = 300.0

    # Enable/disable circuit breaker
    enabled: bool = True

    def __post_init__(self):
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be greater than 0")
        if self.recovery_timeout_seconds <= 0:
            raise ValueError("recovery_timeout_seconds must be greater than 0")
        if self.success_threshold <= 0:
            raise ValueError("success_threshold must be greater than 0")


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(
        self,
        service_name: str,
        failure_count: int,
        next_attempt_time: Optional[datetime] = None,
    ):
        self.service_name = service_name
        self.failure_count = failure_count
        self.next_attempt_time = next_attempt_time

        message = f"Circuit breaker open for {service_name} (failures: {failure_count})"
        if next_attempt_time:
            message += f", next attempt at {next_attempt_time.isoformat()}"

        super().__init__(message)


class CircuitBreaker:
    """Circuit breaker for fault tolerance and automatic recovery.

    Implements the circuit breaker pattern to prevent cascading failures
    by monitoring failure rates and automatically opening/closing the circuit.
    """

    def __init__(
        self, service_name: str, config: Optional[CircuitBreakerConfig] = None
    ):
        self.service_name = service_name
        self.config = config or CircuitBreakerConfig()

        # State tracking
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0  # For half-open state
        self.last_failure_time: Optional[datetime] = None
        self.state_change_time = datetime.now(UTC)

        # Recovery timeout with exponential backoff
        self.current_recovery_timeout = self.config.recovery_timeout_seconds

        # Metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.circuit_opens = 0
        self.circuit_closes = 0

        logger.info(
            f"CircuitBreaker initialized for {service_name}: "
            f"failure_threshold={self.config.failure_threshold}, "
            f"recovery_timeout={self.config.recovery_timeout_seconds}s"
        )

    def is_call_permitted(self) -> bool:
        """Check if a call should be permitted based on circuit state."""
        if not self.config.enabled:
            return True

        current_time = datetime.now(UTC)

        if self.state == CircuitBreakerState.CLOSED:
            return True

        elif self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has elapsed
            if self.last_failure_time:
                time_since_failure = (
                    current_time - self.last_failure_time
                ).total_seconds()
                if time_since_failure >= self.current_recovery_timeout:
                    self._transition_to_half_open(current_time)
                    return True
            return False

        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True

        return False

    def record_success(self, response_time_ms: float = 0) -> None:
        """Record a successful call."""
        self.total_calls += 1
        self.successful_calls += 1

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()

        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            if self.failure_count > 0:
                self.failure_count = max(0, self.failure_count - 1)

    def record_failure(
        self, error: Optional[Exception] = None, response_time_ms: float = 0
    ) -> None:
        """Record a failed call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.failure_count += 1
        self.last_failure_time = datetime.now(UTC)

        # Check for timeout-based failures
        if response_time_ms > self.config.timeout_ms:
            logger.warning(
                f"CircuitBreaker {self.service_name}: Slow response "
                f"({response_time_ms:.1f}ms > {self.config.timeout_ms:.1f}ms)"
            )

        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()

        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Any failure in half-open state transitions back to open
            self._transition_to_open()

    def _transition_to_open(self) -> None:
        """Transition circuit breaker to OPEN state."""
        self.state = CircuitBreakerState.OPEN
        self.state_change_time = datetime.now(UTC)
        self.circuit_opens += 1
        self.success_count = 0

        # Apply exponential backoff to recovery timeout
        self.current_recovery_timeout = min(
            self.current_recovery_timeout * self.config.backoff_multiplier,
            self.config.max_recovery_timeout_seconds,
        )

        logger.warning(
            f"CircuitBreaker OPENED for {self.service_name}: "
            f"failures={self.failure_count}, "
            f"recovery_timeout={self.current_recovery_timeout:.1f}s"
        )

    def _transition_to_half_open(self, current_time: datetime) -> None:
        """Transition circuit breaker to HALF_OPEN state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.state_change_time = current_time
        self.success_count = 0

        logger.info(
            f"CircuitBreaker HALF-OPEN for {self.service_name}: testing recovery"
        )

    def _transition_to_closed(self) -> None:
        """Transition circuit breaker to CLOSED state."""
        self.state = CircuitBreakerState.CLOSED
        self.state_change_time = datetime.now(UTC)
        self.circuit_closes += 1
        self.failure_count = 0
        self.success_count = 0

        # Reset recovery timeout on successful recovery
        self.current_recovery_timeout = self.config.recovery_timeout_seconds

        logger.info(f"CircuitBreaker CLOSED for {self.service_name}: service recovered")

    def get_next_attempt_time(self) -> Optional[datetime]:
        """Get the next time a call will be permitted."""
        if self.state != CircuitBreakerState.OPEN or not self.last_failure_time:
            return None

        return self.last_failure_time + timedelta(seconds=self.current_recovery_timeout)

    def force_open(self) -> None:
        """Force circuit breaker to open (for testing/manual intervention)."""
        logger.warning(f"CircuitBreaker FORCE OPEN for {self.service_name}")
        self._transition_to_open()

    def force_close(self) -> None:
        """Force circuit breaker to close (for testing/manual intervention)."""
        logger.info(f"CircuitBreaker FORCE CLOSE for {self.service_name}")
        self._transition_to_closed()

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state_change_time = datetime.now(UTC)
        self.current_recovery_timeout = self.config.recovery_timeout_seconds

        logger.info(f"CircuitBreaker RESET for {self.service_name}")

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive circuit breaker statistics."""
        failure_rate = (
            self.failed_calls / self.total_calls if self.total_calls > 0 else 0
        )

        current_time = datetime.now(UTC)
        time_in_state = (current_time - self.state_change_time).total_seconds()

        return {
            "service_name": self.service_name,
            "state": self.state.value,
            "time_in_current_state_seconds": time_in_state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "current_recovery_timeout_seconds": self.current_recovery_timeout,
            "next_attempt_time": self.get_next_attempt_time().isoformat()
            if self.get_next_attempt_time()
            else None,
            "metrics": {
                "total_calls": self.total_calls,
                "successful_calls": self.successful_calls,
                "failed_calls": self.failed_calls,
                "failure_rate": failure_rate,
                "circuit_opens": self.circuit_opens,
                "circuit_closes": self.circuit_closes,
            },
            "config": {
                "enabled": self.config.enabled,
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout_seconds": self.config.recovery_timeout_seconds,
                "success_threshold": self.config.success_threshold,
                "timeout_ms": self.config.timeout_ms,
            },
        }

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to wrap functions with circuit breaker protection."""

        async def async_wrapper(*args, **kwargs):
            if not self.is_call_permitted():
                raise CircuitBreakerOpenException(
                    self.service_name, self.failure_count, self.get_next_attempt_time()
                )

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                response_time_ms = (time.time() - start_time) * 1000
                self.record_success(response_time_ms)
                return result

            except Exception as e:
                response_time_ms = (time.time() - start_time) * 1000
                self.record_failure(e, response_time_ms)
                raise

        def sync_wrapper(*args, **kwargs):
            if not self.is_call_permitted():
                raise CircuitBreakerOpenException(
                    self.service_name, self.failure_count, self.get_next_attempt_time()
                )

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                response_time_ms = (time.time() - start_time) * 1000
                self.record_success(response_time_ms)
                return result

            except Exception as e:
                response_time_ms = (time.time() - start_time) * 1000
                self.record_failure(e, response_time_ms)
                raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(service={self.service_name}, "
            f"state={self.state.value}, failures={self.failure_count})"
        )


# Convenience function for creating circuit breakers
def create_circuit_breaker(
    service_name: str,
    failure_threshold: int = 5,
    recovery_timeout_seconds: float = 60.0,
    success_threshold: int = 3,
    timeout_ms: float = 5000.0,
    enabled: bool = True,
    **kwargs,
) -> CircuitBreaker:
    """Create a circuit breaker with simple configuration."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout_seconds=recovery_timeout_seconds,
        success_threshold=success_threshold,
        timeout_ms=timeout_ms,
        enabled=enabled,
        **kwargs,
    )
    return CircuitBreaker(service_name, config)
