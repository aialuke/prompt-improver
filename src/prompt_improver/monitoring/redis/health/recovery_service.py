"""Redis Recovery Service.

Focused Redis recovery service for automatic recovery and circuit breaker patterns.
Designed for resilient operations following SRE best practices with intelligent failover.
"""

import asyncio
import contextlib
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import coredis

from prompt_improver.monitoring.redis.health.types import (
    CircuitBreakerState,
    RecoveryAction,
    RecoveryEvent,
)
from prompt_improver.performance.monitoring.metrics_registry import get_metrics_registry
from prompt_improver.shared.interfaces.protocols.monitoring import (
    RedisClientProviderProtocol,
)

logger = logging.getLogger(__name__)
_metrics_registry = get_metrics_registry()

# Recovery metrics
RECOVERY_ATTEMPTS = _metrics_registry.get_or_create_counter(
    "redis_recovery_attempts_total",
    "Total Redis recovery attempts",
    ["recovery_action", "success"]
)

CIRCUIT_BREAKER_STATE = _metrics_registry.get_or_create_gauge(
    "redis_circuit_breaker_state",
    "Redis circuit breaker state (0=closed, 1=open, 2=half_open)"
)

FAILOVER_DURATION = _metrics_registry.get_or_create_histogram(
    "redis_failover_duration_seconds",
    "Redis failover duration in seconds"
)

RECOVERY_SUCCESS_RATE = _metrics_registry.get_or_create_gauge(
    "redis_recovery_success_rate",
    "Redis recovery success rate percentage"
)


class RedisRecoveryService:
    """Redis recovery service for automatic recovery and circuit breaker patterns.

    Provides intelligent failure recovery with circuit breaker protection,
    automatic failover, and adaptive recovery strategies following SRE practices.
    """

    def __init__(
        self,
        client_provider: RedisClientProviderProtocol,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_timeout_seconds: int = 60,
        max_recovery_attempts: int = 3,
        recovery_timeout_seconds: float = 30.0
    ) -> None:
        """Initialize Redis recovery service.

        Args:
            client_provider: Redis client provider for connections
            circuit_breaker_failure_threshold: Failures before opening circuit breaker
            circuit_breaker_timeout_seconds: Circuit breaker timeout duration
            max_recovery_attempts: Maximum recovery attempts per incident
            recovery_timeout_seconds: Timeout for individual recovery operations
        """
        self.client_provider = client_provider
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_timeout_seconds = recovery_timeout_seconds

        # Circuit breaker state
        self._circuit_breaker = CircuitBreakerState(
            failure_threshold=circuit_breaker_failure_threshold,
            timeout_seconds=circuit_breaker_timeout_seconds
        )

        # Recovery state tracking
        self._recovery_history: list[RecoveryEvent] = []
        self._max_history_size = 100
        self._active_recovery: RecoveryEvent | None = None

        # Backup client tracking
        self._backup_client: coredis.Redis | None = None
        self._is_using_backup = False

        # Statistics
        self._total_recovery_attempts = 0
        self._successful_recoveries = 0

        # Service state
        self._is_monitoring = False
        self._monitor_task: asyncio.Task | None = None

    async def attempt_recovery(self, reason: str) -> RecoveryEvent:
        """Attempt automatic recovery from failure.

        Args:
            reason: Reason for recovery attempt

        Returns:
            Recovery event with results
        """
        start_time = datetime.now(UTC)

        recovery_event = RecoveryEvent(
            id=str(uuid.uuid4()),
            action=RecoveryAction.NONE,
            trigger_reason=reason,
            timestamp=start_time
        )

        try:
            # Check if circuit breaker allows attempts
            if not self._circuit_breaker.can_attempt():
                recovery_event.action = RecoveryAction.CIRCUIT_BREAK
                recovery_event.error_message = "Circuit breaker open - recovery blocked"
                recovery_event.completion_time = datetime.now(UTC)
                self._add_to_history(recovery_event)
                return recovery_event

            self._active_recovery = recovery_event
            self._total_recovery_attempts += 1

            # Determine recovery strategy based on failure reason and history
            recovery_strategy = await self._determine_recovery_strategy(reason)
            recovery_event.action = recovery_strategy

            # Execute recovery action
            success = await self._execute_recovery_action(recovery_strategy, recovery_event)

            recovery_event.success = success
            recovery_event.completion_time = datetime.now(UTC)

            # Update circuit breaker state
            if success:
                self._circuit_breaker.record_success()
                self._successful_recoveries += 1
                logger.info(f"Recovery successful: {recovery_strategy.value} for {reason}")
            else:
                self._circuit_breaker.record_failure()
                logger.warning(f"Recovery failed: {recovery_strategy.value} for {reason}")

            # Update metrics
            RECOVERY_ATTEMPTS.labels(
                recovery_action=recovery_strategy.value,
                success=str(success).lower()
            ).inc()

            self._update_circuit_breaker_metrics()
            self._update_success_rate_metrics()

            # Add to history
            self._add_to_history(recovery_event)

            return recovery_event

        except Exception as e:
            logger.exception(f"Recovery attempt failed with exception: {e}")

            recovery_event.success = False
            recovery_event.error_message = str(e)
            recovery_event.completion_time = datetime.now(UTC)
            recovery_event.action = RecoveryAction.ESCALATE

            self._circuit_breaker.record_failure()
            self._add_to_history(recovery_event)

            return recovery_event

        finally:
            self._active_recovery = None

    async def execute_failover(self) -> bool:
        """Execute failover to backup Redis instance.

        Returns:
            True if failover was successful
        """
        start_time = datetime.now(UTC)

        try:
            logger.info("Executing Redis failover to backup instance")

            # Create backup client if not exists
            if not self._backup_client:
                self._backup_client = await self.client_provider.create_backup_client()

            if not self._backup_client:
                logger.error("No backup Redis client available for failover")
                return False

            # Test backup client connectivity
            if not await self._test_client_connectivity(self._backup_client):
                logger.error("Backup Redis client is not responding")
                return False

            # Switch to backup client
            self._is_using_backup = True

            # Record failover duration
            duration = (datetime.now(UTC) - start_time).total_seconds()
            FAILOVER_DURATION.observe(duration)

            logger.info(f"Failover completed successfully in {duration:.2f}s")
            return True

        except Exception as e:
            logger.exception(f"Failover failed: {e}")
            return False

    def get_circuit_breaker_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state.

        Returns:
            Current circuit breaker state
        """
        return self._circuit_breaker

    def record_operation_result(self, success: bool) -> None:
        """Record operation result for circuit breaker logic.

        Args:
            success: Whether the operation was successful
        """
        if success:
            self._circuit_breaker.record_success()
        else:
            self._circuit_breaker.record_failure()

        self._update_circuit_breaker_metrics()

    async def validate_recovery(self) -> bool:
        """Validate that recovery was successful.

        Returns:
            True if system has recovered
        """
        try:
            # Get current client (primary or backup)
            client = await self._get_current_client()
            if not client:
                return False

            # Test basic connectivity
            if not await self._test_client_connectivity(client):
                return False

            # Test basic operations
            test_key = f"recovery_test_{uuid.uuid4().hex[:8]}"
            test_value = "recovery_validation"

            # Test SET operation
            await asyncio.wait_for(
                client.set(test_key, test_value, ex=10),  # 10 second expiry
                timeout=self.recovery_timeout_seconds
            )

            # Test GET operation
            retrieved_value = await asyncio.wait_for(
                client.get(test_key),
                timeout=self.recovery_timeout_seconds
            )

            if retrieved_value != test_value.encode():
                logger.warning("Recovery validation failed: retrieved value mismatch")
                return False

            # Cleanup test key
            await client.delete(test_key)

            logger.info("Recovery validation successful")
            return True

        except Exception as e:
            logger.exception(f"Recovery validation failed: {e}")
            return False

    def get_recovery_history(self) -> list[RecoveryEvent]:
        """Get history of recovery attempts.

        Returns:
            List of recovery events sorted by timestamp
        """
        return sorted(self._recovery_history, key=lambda x: x.timestamp, reverse=True)

    async def start_monitoring(self) -> None:
        """Start recovery monitoring and circuit breaker management."""
        if self._is_monitoring:
            logger.warning("Recovery monitoring already started")
            return

        self._is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Redis recovery monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop recovery monitoring."""
        if not self._is_monitoring:
            return

        self._is_monitoring = False

        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task
            self._monitor_task = None

        logger.info("Redis recovery monitoring stopped")

    async def _determine_recovery_strategy(self, reason: str) -> RecoveryAction:
        """Determine appropriate recovery strategy based on failure reason.

        Args:
            reason: Failure reason

        Returns:
            Recommended recovery action
        """
        reason_lower = reason.lower()

        # Simple connection issues - try retry first
        if any(keyword in reason_lower for keyword in ["timeout", "connection", "ping"]):
            if self._get_recent_failure_count() < 3:
                return RecoveryAction.RETRY
            return RecoveryAction.FAILOVER

        # Memory or performance issues - may need failover
        if any(keyword in reason_lower for keyword in ["memory", "fragmentation", "slow"]) or any(keyword in reason_lower for keyword in ["unavailable", "failed", "critical"]):
            return RecoveryAction.FAILOVER

        # Default to retry for unknown issues
        return RecoveryAction.RETRY

    async def _execute_recovery_action(
        self,
        action: RecoveryAction,
        recovery_event: RecoveryEvent
    ) -> bool:
        """Execute specific recovery action.

        Args:
            action: Recovery action to execute
            recovery_event: Recovery event to update

        Returns:
            True if recovery action was successful
        """
        try:
            if action == RecoveryAction.RETRY:
                return await self._execute_retry_recovery()

            if action == RecoveryAction.FAILOVER:
                return await self.execute_failover()

            if action == RecoveryAction.CIRCUIT_BREAK:
                # Circuit breaker action - no actual recovery, just state management
                logger.info("Circuit breaker activated - blocking operations")
                return True

            if action == RecoveryAction.ESCALATE:
                # Escalation - log for manual intervention
                logger.critical(f"Recovery escalated for manual intervention: {recovery_event.trigger_reason}")
                return False

            logger.warning(f"Unknown recovery action: {action}")
            return False

        except Exception as e:
            logger.exception(f"Failed to execute recovery action {action}: {e}")
            return False

    async def _execute_retry_recovery(self) -> bool:
        """Execute retry-based recovery.

        Returns:
            True if retry recovery was successful
        """
        try:
            # Get fresh client connection
            client = await self.client_provider.get_client()
            if not client:
                return False

            # Test connectivity with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await asyncio.wait_for(
                        client.ping(),
                        timeout=self.recovery_timeout_seconds
                    )
                    logger.info(f"Retry recovery successful on attempt {attempt + 1}")
                    return True

                except Exception as e:
                    logger.debug(f"Retry attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)  # Brief delay between retries

            return False

        except Exception as e:
            logger.exception(f"Retry recovery failed: {e}")
            return False

    async def _test_client_connectivity(self, client: coredis.Redis) -> bool:
        """Test Redis client connectivity.

        Args:
            client: Redis client to test

        Returns:
            True if client is responsive
        """
        try:
            await asyncio.wait_for(
                client.ping(),
                timeout=self.recovery_timeout_seconds
            )
            return True
        except Exception:
            return False

    async def _get_current_client(self) -> coredis.Redis | None:
        """Get current active Redis client (primary or backup).

        Returns:
            Current Redis client or None
        """
        if self._is_using_backup and self._backup_client:
            return self._backup_client
        return await self.client_provider.get_client()

    async def _monitor_loop(self) -> None:
        """Background monitoring loop for recovery service."""
        logger.info("Starting Redis recovery monitoring loop")

        while self._is_monitoring:
            try:
                # Check if we can switch back from backup to primary
                if self._is_using_backup:
                    await self._check_primary_recovery()

                # Clean up old recovery history
                self._cleanup_old_history()

                # Update metrics
                self._update_success_rate_metrics()

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in recovery monitoring loop: {e}")
                await asyncio.sleep(10)  # Brief delay on error

    async def _check_primary_recovery(self) -> None:
        """Check if primary Redis instance has recovered and switch back."""
        try:
            primary_client = await self.client_provider.get_client()
            if primary_client and await self._test_client_connectivity(primary_client):
                # Primary is back - validate it's stable
                stable_checks = 3
                for _ in range(stable_checks):
                    if not await self._test_client_connectivity(primary_client):
                        return  # Not stable yet
                    await asyncio.sleep(2)

                # Primary is stable - switch back
                self._is_using_backup = False
                logger.info("Switched back to primary Redis instance")

        except Exception as e:
            logger.debug(f"Primary recovery check failed: {e}")

    def _get_recent_failure_count(self) -> int:
        """Get count of recent recovery failures.

        Returns:
            Number of failed recoveries in last 5 minutes
        """
        cutoff_time = datetime.now(UTC) - timedelta(minutes=5)
        return sum(
            1 for event in self._recovery_history
            if event.timestamp >= cutoff_time and not event.success
        )

    def _add_to_history(self, recovery_event: RecoveryEvent) -> None:
        """Add recovery event to history.

        Args:
            recovery_event: Recovery event to add
        """
        self._recovery_history.append(recovery_event)

        # Keep history size manageable
        if len(self._recovery_history) > self._max_history_size:
            self._recovery_history = self._recovery_history[-self._max_history_size:]

    def _cleanup_old_history(self) -> None:
        """Clean up old recovery history to manage memory."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=24)
        self._recovery_history = [
            event for event in self._recovery_history
            if event.timestamp >= cutoff_time
        ]

    def _update_circuit_breaker_metrics(self) -> None:
        """Update circuit breaker state metrics."""
        state_mapping = {
            "closed": 0,
            "open": 1,
            "half_open": 2
        }
        CIRCUIT_BREAKER_STATE.set(state_mapping.get(self._circuit_breaker.state, 0))

    def _update_success_rate_metrics(self) -> None:
        """Update recovery success rate metrics."""
        if self._total_recovery_attempts > 0:
            success_rate = (self._successful_recoveries / self._total_recovery_attempts) * 100
            RECOVERY_SUCCESS_RATE.set(success_rate)

    def get_recovery_statistics(self) -> dict[str, Any]:
        """Get recovery service statistics.

        Returns:
            Recovery statistics dictionary
        """
        recent_history = [
            event for event in self._recovery_history
            if event.timestamp >= datetime.now(UTC) - timedelta(hours=1)
        ]

        success_rate = (
            (self._successful_recoveries / self._total_recovery_attempts * 100)
            if self._total_recovery_attempts > 0 else 0
        )

        return {
            "total_recovery_attempts": self._total_recovery_attempts,
            "successful_recoveries": self._successful_recoveries,
            "success_rate_percent": round(success_rate, 2),
            "circuit_breaker_state": self._circuit_breaker.state,
            "circuit_breaker_failure_count": self._circuit_breaker.failure_count,
            "is_using_backup": self._is_using_backup,
            "recent_recoveries_1h": len(recent_history),
            "recent_failures_5m": self._get_recent_failure_count(),
            "active_recovery": self._active_recovery.id if self._active_recovery else None,
            "monitoring_enabled": self._is_monitoring,
        }

    def get_circuit_breaker_info(self) -> dict[str, Any]:
        """Get detailed circuit breaker information.

        Returns:
            Circuit breaker information dictionary
        """
        return {
            "state": self._circuit_breaker.state,
            "failure_count": self._circuit_breaker.failure_count,
            "failure_threshold": self._circuit_breaker.failure_threshold,
            "timeout_seconds": self._circuit_breaker.timeout_seconds,
            "can_attempt": self._circuit_breaker.can_attempt(),
            "last_failure_time": (
                self._circuit_breaker.last_failure_time.isoformat()
                if self._circuit_breaker.last_failure_time else None
            ),
            "next_attempt_time": (
                self._circuit_breaker.next_attempt_time.isoformat()
                if self._circuit_breaker.next_attempt_time else None
            ),
        }
