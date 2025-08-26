"""Distributed lock management service for database connections.

This module provides distributed locking functionality extracted from
unified_connection_manager.py, implementing:

- DistributedLockManager: Redis-based distributed locking with token-based security
- LockConfig: Configurable timeouts, retry policies, and security settings
- DistributedLock: Context manager for automatic lock acquisition/release
- Lock monitoring with OpenTelemetry metrics integration
- Lua script-based atomic operations for lock safety

Designed for production distributed systems with sub-5ms lock operations.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

from prompt_improver.services.cache.l2_redis_service import L2RedisService

logger = logging.getLogger(__name__)

# Import OpenTelemetry metrics if available
try:
    from opentelemetry import metrics

    OPENTELEMETRY_AVAILABLE = True
    meter = metrics.get_meter(__name__)
    lock_operations_counter = meter.create_counter(
        "distributed_lock_operations_total",
        description="Total distributed lock operations",
    )
    lock_duration_histogram = meter.create_histogram(
        "distributed_lock_duration_seconds", description="Duration of lock hold times"
    )
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    lock_operations_counter = None
    lock_duration_histogram = None


@dataclass
class LockConfig:
    """Configuration for distributed lock behavior."""

    # Lock timeouts
    default_timeout_seconds: int = 30
    max_timeout_seconds: int = 300
    min_timeout_seconds: int = 1

    # Retry behavior
    retry_attempts: int = 3
    retry_delay_seconds: float = 0.1
    retry_backoff_multiplier: float = 2.0
    max_retry_delay_seconds: float = 1.0

    # Lock key prefix
    lock_key_prefix: str = "lock"

    # Monitoring
    enable_metrics: bool = True
    warn_on_long_locks: bool = True
    long_lock_threshold_seconds: float = 60.0

    # Security
    require_security_context: bool = False

    def __post_init__(self):
        if self.default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be greater than 0")
        if self.max_timeout_seconds < self.default_timeout_seconds:
            raise ValueError("max_timeout_seconds must be >= default_timeout_seconds")
        if self.retry_attempts < 0:
            raise ValueError("retry_attempts must be >= 0")


@dataclass
class LockInfo:
    """Information about an active lock."""

    key: str
    token: str
    acquired_at: float
    timeout_seconds: int
    expires_at: float
    owner_context: str | None = None

    @property
    def is_expired(self) -> bool:
        """Check if the lock has expired."""
        return time.time() > self.expires_at

    @property
    def time_remaining(self) -> float:
        """Get remaining time before lock expires."""
        return max(0, self.expires_at - time.time())

    @property
    def duration_held(self) -> float:
        """Get how long the lock has been held."""
        return time.time() - self.acquired_at


class DistributedLockManager:
    """Redis-based distributed lock manager with token-based security.

    Provides distributed locking functionality with:
    - Atomic Redis operations using Lua scripts
    - Token-based lock ownership validation
    - Automatic lock expiration and cleanup
    - Retry logic with exponential backoff
    - OpenTelemetry metrics integration
    - Context manager support for automatic cleanup
    """

    def __init__(
        self, l2_redis_service: L2RedisService, config: LockConfig | None = None, security_context=None
    ) -> None:
        self._l2_redis = l2_redis_service
        self.config = config or LockConfig()
        self.security_context = security_context

        # Active locks tracking
        self._active_locks: dict[str, LockInfo] = {}
        self._lock_tracking_enabled = True

        # Cleanup task
        self._cleanup_task: asyncio.Task | None = None
        self._cleanup_interval = 30.0  # seconds
        self._shutdown_event = asyncio.Event()

        # Performance metrics
        self.total_acquire_attempts = 0
        self.successful_acquisitions = 0
        self.failed_acquisitions = 0
        self.total_releases = 0
        self.total_extensions = 0

        # Lua scripts are now embedded in L2RedisService for reusability

        logger.info(
            f"DistributedLockManager initialized with timeout={self.config.default_timeout_seconds}s"
        )

    async def start_cleanup_task(self) -> None:
        """Start background cleanup task for expired locks."""
        if self._cleanup_task and not self._cleanup_task.done():
            logger.warning("Cleanup task already running")
            return

        self._shutdown_event.clear()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Started distributed lock cleanup task")

    async def stop_cleanup_task(self) -> None:
        """Stop background cleanup task."""
        self._shutdown_event.set()

        if self._cleanup_task and not self._cleanup_task.done():
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=5.0)
            except TimeoutError:
                logger.warning("Cleanup task did not stop gracefully")
                self._cleanup_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._cleanup_task

        logger.info("Stopped distributed lock cleanup task")

    async def _cleanup_loop(self) -> None:
        """Background loop to clean up expired locks."""
        while not self._shutdown_event.is_set():
            try:
                await self._cleanup_expired_locks()

                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=self._cleanup_interval
                    )
                    break  # Shutdown requested
                except TimeoutError:
                    continue  # Normal timeout, continue cleanup

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in lock cleanup loop: {e}")
                await asyncio.sleep(min(self._cleanup_interval, 10.0))

    async def _cleanup_expired_locks(self) -> None:
        """Clean up expired locks from active tracking."""
        current_time = time.time()
        expired_keys = []

        for key, lock_info in self._active_locks.items():
            if lock_info.is_expired:
                expired_keys.append(key)
                if (
                    self.config.warn_on_long_locks
                    and lock_info.duration_held
                    > self.config.long_lock_threshold_seconds
                ):
                    logger.warning(
                        f"Long-held lock expired: {key}, held for {lock_info.duration_held:.1f}s"
                    )

        for key in expired_keys:
            del self._active_locks[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired locks")

    def _generate_lock_key(self, key: str) -> str:
        """Generate Redis key for lock."""
        return f"{self.config.lock_key_prefix}:{key}"

    def _generate_token(self) -> str:
        """Generate unique lock token."""
        return str(uuid.uuid4())

    def _validate_timeout(self, timeout: int) -> int:
        """Validate and clamp timeout value."""
        return max(
            self.config.min_timeout_seconds,
            min(timeout, self.config.max_timeout_seconds),
        )

    async def acquire_lock(
        self, key: str, timeout: int | None = None, retry: bool = True
    ) -> str | None:
        """Acquire distributed lock with retry logic.

        Args:
            key: Lock key (will be prefixed)
            timeout: Lock timeout in seconds
            retry: Whether to retry on failure

        Returns:
            Lock token if acquired, None if failed
        """
        if not self._l2_redis or not self._l2_redis.is_available():
            logger.error("L2 Redis service not available for lock acquisition")
            return None

        timeout = timeout or self.config.default_timeout_seconds
        timeout = self._validate_timeout(timeout)

        self.total_acquire_attempts += 1

        # Retry logic with exponential backoff
        max_attempts = self.config.retry_attempts + 1 if retry else 1
        delay = self.config.retry_delay_seconds

        for attempt in range(max_attempts):
            try:
                token = self._generate_token()
                lock_key = self._generate_lock_key(key)

                # Attempt to acquire lock with Redis SET NX EX through L2RedisService
                acquired = await self._l2_redis.lock_acquire(
                    lock_key, token, timeout
                )

                if acquired:
                    # Track active lock
                    current_time = time.time()
                    self._active_locks[key] = LockInfo(
                        key=key,
                        token=token,
                        acquired_at=current_time,
                        timeout_seconds=timeout,
                        expires_at=current_time + timeout,
                        owner_context=getattr(self.security_context, "agent_id", None)
                        if self.security_context
                        else None,
                    )

                    self.successful_acquisitions += 1

                    # Record metrics
                    if OPENTELEMETRY_AVAILABLE and lock_operations_counter:
                        lock_operations_counter.add(
                            1,
                            {
                                "operation": "acquire",
                                "status": "success",
                                "attempt": attempt + 1,
                            },
                        )

                    logger.debug(
                        f"Acquired lock '{key}' with token {token[:8]}... (timeout: {timeout}s)"
                    )
                    return token

                # Lock acquisition failed
                if attempt < max_attempts - 1:
                    logger.debug(
                        f"Lock '{key}' acquisition attempt {attempt + 1} failed, retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                    delay = min(
                        delay * self.config.retry_backoff_multiplier,
                        self.config.max_retry_delay_seconds,
                    )

            except Exception as e:
                logger.exception(
                    f"Lock acquisition error for '{key}' attempt {attempt + 1}: {e}"
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(delay)
                    delay = min(
                        delay * self.config.retry_backoff_multiplier,
                        self.config.max_retry_delay_seconds,
                    )

        # All attempts failed
        self.failed_acquisitions += 1

        if OPENTELEMETRY_AVAILABLE and lock_operations_counter:
            lock_operations_counter.add(
                1,
                {
                    "operation": "acquire",
                    "status": "failed",
                    "attempts": max_attempts,
                },
            )

        logger.warning(f"Failed to acquire lock '{key}' after {max_attempts} attempts")
        return None

    async def release_lock(self, key: str, token: str) -> bool:
        """Release distributed lock using atomic Lua script.

        Args:
            key: Lock key
            token: Lock token from acquire_lock

        Returns:
            True if successfully released, False otherwise
        """
        if not self._l2_redis or not self._l2_redis.is_available():
            logger.error("L2 Redis service not available for lock release")
            return False

        try:
            lock_key = self._generate_lock_key(key)

            # Use Lua script for atomic release through L2RedisService
            result = await self._l2_redis.lock_release(
                lock_key, token
            )

            released = bool(result)
            self.total_releases += 1

            if released:
                # Update tracking
                if key in self._active_locks:
                    lock_info = self._active_locks[key]
                    duration = lock_info.duration_held
                    del self._active_locks[key]

                    # Record lock duration
                    if OPENTELEMETRY_AVAILABLE and lock_duration_histogram:
                        lock_duration_histogram.record(
                            duration,
                            {
                                "operation": "release",
                                "status": "success",
                            },
                        )

                    if (
                        self.config.warn_on_long_locks
                        and duration > self.config.long_lock_threshold_seconds
                    ):
                        logger.warning(
                            f"Long lock released: '{key}' held for {duration:.1f}s"
                        )

                logger.debug(f"Released lock '{key}'")
            else:
                logger.warning(
                    f"Could not release lock '{key}' - token mismatch or expired"
                )

            # Record metrics
            if OPENTELEMETRY_AVAILABLE and lock_operations_counter:
                lock_operations_counter.add(
                    1,
                    {
                        "operation": "release",
                        "status": "success" if released else "failed",
                    },
                )

            return released

        except Exception as e:
            logger.exception(f"Lock release error for '{key}': {e}")

            if OPENTELEMETRY_AVAILABLE and lock_operations_counter:
                lock_operations_counter.add(
                    1,
                    {
                        "operation": "release",
                        "status": "error",
                    },
                )

            return False

    async def extend_lock(self, key: str, token: str, timeout: int) -> bool:
        """Extend lock timeout using atomic Lua script.

        Args:
            key: Lock key
            token: Lock token from acquire_lock
            timeout: New timeout in seconds

        Returns:
            True if successfully extended, False otherwise
        """
        if not self._l2_redis or not self._l2_redis.is_available():
            logger.error("L2 Redis service not available for lock extension")
            return False

        timeout = self._validate_timeout(timeout)

        try:
            lock_key = self._generate_lock_key(key)

            # Use Lua script for atomic extension through L2RedisService
            result = await self._l2_redis.lock_extend(
                lock_key, token, timeout
            )

            extended = bool(result)
            self.total_extensions += 1

            if extended:
                # Update tracking
                if key in self._active_locks:
                    current_time = time.time()
                    self._active_locks[key].timeout_seconds = timeout
                    self._active_locks[key].expires_at = current_time + timeout

                logger.debug(f"Extended lock '{key}' by {timeout} seconds")
            else:
                logger.warning(
                    f"Could not extend lock '{key}' - token mismatch or expired"
                )

            # Record metrics
            if OPENTELEMETRY_AVAILABLE and lock_operations_counter:
                lock_operations_counter.add(
                    1,
                    {
                        "operation": "extend",
                        "status": "success" if extended else "failed",
                    },
                )

            return extended

        except Exception as e:
            logger.exception(f"Lock extension error for '{key}': {e}")

            if OPENTELEMETRY_AVAILABLE and lock_operations_counter:
                lock_operations_counter.add(
                    1,
                    {
                        "operation": "extend",
                        "status": "error",
                    },
                )

            return False

    @asynccontextmanager
    async def acquire_lock_context(
        self, key: str, timeout: int | None = None, retry: bool = True
    ) -> AsyncIterator[str | None]:
        """Context manager for automatic lock acquisition and release.

        Args:
            key: Lock key
            timeout: Lock timeout in seconds
            retry: Whether to retry on acquisition failure

        Yields:
            Lock token if acquired, None if failed
        """
        token = await self.acquire_lock(key, timeout, retry)
        try:
            yield token
        finally:
            if token:
                await self.release_lock(key, token)

    def get_active_locks(self) -> dict[str, LockInfo]:
        """Get information about all active locks."""
        return dict(self._active_locks)

    def get_lock_info(self, key: str) -> LockInfo | None:
        """Get information about a specific lock."""
        return self._active_locks.get(key)

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive lock manager statistics."""
        current_time = time.time()

        # Analyze active locks
        active_count = len(self._active_locks)
        expired_count = sum(
            1 for lock in self._active_locks.values() if lock.is_expired
        )
        long_locks = [
            lock
            for lock in self._active_locks.values()
            if lock.duration_held > self.config.long_lock_threshold_seconds
        ]

        total_attempts = self.total_acquire_attempts
        success_rate = (
            self.successful_acquisitions / total_attempts if total_attempts > 0 else 0
        )

        return {
            "manager": {
                "redis_available": self._l2_redis is not None and self._l2_redis.is_available(),
                "cleanup_task_running": self._cleanup_task is not None
                and not self._cleanup_task.done(),
                "lock_tracking_enabled": self._lock_tracking_enabled,
            },
            "locks": {
                "active_count": active_count,
                "expired_count": expired_count,
                "long_locks_count": len(long_locks),
                "active_keys": list(self._active_locks.keys()),
            },
            "performance": {
                "total_acquire_attempts": self.total_acquire_attempts,
                "successful_acquisitions": self.successful_acquisitions,
                "failed_acquisitions": self.failed_acquisitions,
                "success_rate": success_rate,
                "total_releases": self.total_releases,
                "total_extensions": self.total_extensions,
            },
            "config": {
                "default_timeout_seconds": self.config.default_timeout_seconds,
                "max_timeout_seconds": self.config.max_timeout_seconds,
                "retry_attempts": self.config.retry_attempts,
                "cleanup_interval": self._cleanup_interval,
            },
            "long_locks": [
                {
                    "key": lock.key,
                    "duration_held": lock.duration_held,
                    "time_remaining": lock.time_remaining,
                    "owner_context": lock.owner_context,
                }
                for lock in long_locks
            ],
        }

    async def cleanup_all_locks(self) -> int:
        """Force cleanup of all tracked locks (for shutdown)."""
        count = len(self._active_locks)
        self._active_locks.clear()
        logger.info(f"Cleaned up {count} tracked locks")
        return count

    async def shutdown(self) -> None:
        """Shutdown lock manager and cleanup resources."""
        logger.info("Shutting down DistributedLockManager")

        await self.stop_cleanup_task()
        await self.cleanup_all_locks()

        logger.info("DistributedLockManager shutdown complete")

    def __repr__(self) -> str:
        return (
            f"DistributedLockManager(active_locks={len(self._active_locks)}, "
            f"success_rate={self.successful_acquisitions}/{self.total_acquire_attempts})"
        )


# Convenience function for creating lock managers
def create_lock_manager(
    l2_redis_service: L2RedisService,
    default_timeout_seconds: int = 30,
    retry_attempts: int = 3,
    enable_metrics: bool = True,
    **kwargs,
) -> DistributedLockManager:
    """Create a distributed lock manager with simple configuration.

    Args:
        l2_redis_service: L2RedisService instance for distributed locking
        default_timeout_seconds: Default lock timeout
        retry_attempts: Number of retry attempts for lock acquisition
        enable_metrics: Enable OpenTelemetry metrics
        **kwargs: Additional LockConfig parameters

    Returns:
        Configured DistributedLockManager instance
    """
    config = LockConfig(
        default_timeout_seconds=default_timeout_seconds,
        retry_attempts=retry_attempts,
        enable_metrics=enable_metrics,
        **kwargs,
    )
    return DistributedLockManager(l2_redis_service, config)


# Legacy factory function for backward compatibility during migration
def create_lock_manager_legacy(
    redis_client,
    default_timeout_seconds: int = 30,
    retry_attempts: int = 3,
    enable_metrics: bool = True,
    **kwargs,
) -> DistributedLockManager:
    """Legacy factory function for backward compatibility.

    DEPRECATED: Use create_lock_manager with L2RedisService instead.
    This function creates a temporary L2RedisService wrapper.
    """
    logger.warning(
        "create_lock_manager_legacy is deprecated. "
        "Migrate to create_lock_manager with L2RedisService."
    )

    # Create temporary L2RedisService for legacy support
    l2_redis_service = L2RedisService()

    config = LockConfig(
        default_timeout_seconds=default_timeout_seconds,
        retry_attempts=retry_attempts,
        enable_metrics=enable_metrics,
        **kwargs,
    )
    return DistributedLockManager(l2_redis_service, config)
