"""Redis health monitoring task for periodic liveness and performance checks."""

import asyncio
import time
from datetime import datetime
from typing import Union

try:
    from ...utils.redis_cache import CACHE_ERRORS, CACHE_LATENCY_MS, redis_client
except ImportError:
    # Fallback if relative imports fail
    from src.prompt_improver.utils.redis_cache import (
        CACHE_ERRORS,
        CACHE_LATENCY_MS,
        redis_client,
    )

import logging

from prometheus_client import Counter

from .background_manager import get_background_task_manager
from .base import HealthChecker, HealthResult, HealthStatus

# Configure logging
logger = logging.getLogger("redis_health_monitor")

# Prometheus metrics for Redis health check
try:
    REDIS_CHECK_FAILURES = Counter('redis_check_failures_total', 'Total number of Redis check failures')
except ValueError:
    # Metric already exists, retrieve it
    from prometheus_client import REGISTRY
    for collector in REGISTRY._collector_to_names.keys():
        if hasattr(collector, '_name') and collector._name == 'redis_check_failures_total':
            REDIS_CHECK_FAILURES = collector
            break
    else:
        # Create with different name if still failing
        REDIS_CHECK_FAILURES = Counter('redis_check_failures_total_v2', 'Total number of Redis check failures')


class RedisHealthMonitor(HealthChecker):
    """Async Redis health monitoring using PING and GET liveness checks."""

    def __init__(self, config):
        super().__init__("redis")
        self.config = config
        self.failure_count = 0
        self.last_check_time = None
        self.is_connected = True

    async def ping_check(self) -> str | None:
        """Perform a Redis PING command to check if the server is alive."""
        try:
            response = await redis_client.ping()
            if response:
                return "PING successful"
        except Exception as e:
            logger.error(f"PING failed: {e}")
            CACHE_ERRORS.labels(operation='ping').inc()
        return None

    async def get_check(self) -> str | None:
        """Perform a Redis GET command to check if data retrieval is working."""
        test_key = "health:check"
        try:
            await redis_client.set(test_key, "ok", ex=60)
            start_time = time.time()
            response = await redis_client.get(test_key)
            end_time = time.time()
            if response == b"ok":
                latency_ms = (end_time - start_time) * 1000
                CACHE_LATENCY_MS.labels(operation='get').observe(latency_ms)
                return "GET successful"
        except Exception as e:
            logger.error(f"GET failed: {e}")
            CACHE_ERRORS.labels(operation='get').inc()
        return None

    async def check(self) -> HealthResult:
        """Perform the health check and return a result."""
        try:
            ping_result = await self.ping_check()
            get_result = await self.get_check()

            if not ping_result or not get_result:
                self.failure_count += 1
                REDIS_CHECK_FAILURES.inc()
                logger.warning(
                    f"Redis check failed: {self.failure_count} failures"
                )

                if self.failure_count >= self.config['failure_threshold']:
                    await self.trigger_reconnection_logic()

                return HealthResult(
                    status=HealthStatus.WARNING,
                    component=self.name,
                    message=f"{self.failure_count} consecutive failures",
                )
            self.failure_count = 0
            return HealthResult(
                status=HealthStatus.HEALTHY,
                component=self.name,
                message="Redis is healthy",
            )
        except Exception as e:
            logger.error(f"Unhandled exception during Redis check: {e}")
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                error=str(e),
                message="Redis check failed due to exception",
            )

    async def trigger_reconnection_logic(self) -> None:
        """Trigger reconnection logic after threshold failures."""
        max_retries = self.config.get('reconnection', {}).get('max_retries', 3)
        backoff_factor = self.config.get('reconnection', {}).get('backoff_factor', 2)

        retry_count = 0
        backoff = 1

        while retry_count < max_retries:
            try:
                await asyncio.sleep(backoff)
                # Attempt reconnection
                pool = redis_client.connection_pool
                pool.disconnect()
                await asyncio.sleep(1)
                self.failure_count = 0  # Reset failure count
                logger.info("Reconnected to Redis")
                break  # Exit loop on successful reconnection
            except Exception as e:
                logger.warning(f"Reconnection attempt {retry_count} failed: {e}")
                retry_count += 1
                backoff *= backoff_factor

        if retry_count == max_retries:
            logger.error("Redis reconnection failed after maximum retries")


async def schedule_redis_health_checks(config):
    """Schedule and manage periodic Redis health checks."""
    task_manager = get_background_task_manager()
    health_monitor = RedisHealthMonitor(config)

    task_id = "redis-health-check"
    check_interval = config.get('check_interval', 60)

    # Run periodic health checks
    while True:
        try:
            await task_manager.submit_task(f"{task_id}-{int(time.time())}", health_monitor.check)
            await asyncio.sleep(check_interval)
        except Exception as e:
            logger.error(f"Error scheduling Redis health check: {e}")
            await asyncio.sleep(check_interval)


# Initialize the Redis Health Monitor
default_monitor_config = {
    'check_interval': 60,  # Time between checks in seconds
    'failure_threshold': 3,
    'latency_threshold': 100,
    'reconnection': {'max_retries': 5, 'backoff_factor': 2}
}


async def start_redis_health_monitor(config=None):
    """Start the Redis health monitoring task based on the configuration."""
    if config is None:
        config = default_monitor_config
    await schedule_redis_health_checks(config)
