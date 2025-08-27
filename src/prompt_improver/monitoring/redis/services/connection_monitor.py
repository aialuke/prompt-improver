"""Redis Connection Monitor Service.

Focused service for Redis connection health and failover detection.
Handles connection status, ping monitoring, and failover detection.
"""

import logging
import time
from datetime import UTC, datetime
from typing import Any

import coredis

from prompt_improver.monitoring.redis.types import (
    RedisConnectionInfo,
    RedisHealthConfig,
    RedisHealthResult,
    RedisHealthStatus,
)

logger = logging.getLogger(__name__)


class RedisConnectionMonitor:
    """Redis connection monitoring service.

    Focused responsibility: Monitor Redis connection health and failover status.
    Performance target: <10ms connection checks.
    """

    def __init__(
        self,
        config: RedisHealthConfig,
        client: coredis.Redis | None = None,
    ) -> None:
        """Initialize Redis connection monitor.

        Args:
            config: Redis health monitoring configuration
            client: Optional Redis client (will create if not provided)
        """
        self.config = config
        self._client = client
        self._connection_info = RedisConnectionInfo(
            host=config.host,
            port=config.port,
        )
        self._last_ping_time = 0.0
        self._connection_start_time = time.time()
        self._error_count = 0
        self._last_error: str | None = None

    async def _ensure_client(self) -> bool:
        """Ensure Redis client is available."""
        if self._client is None:
            try:
                self._client = coredis.Redis(
                    host=self.config.host,
                    port=self.config.port,
                    socket_timeout=self.config.timeout_seconds,
                    socket_connect_timeout=self.config.timeout_seconds,
                )
                return True
            except Exception as e:
                self._last_error = str(e)
                self._error_count += 1
                logger.exception(f"Failed to create Redis client: {e}")
                return False
        return True

    async def check_connection_health(self) -> RedisHealthResult:
        """Check Redis connection health and failover status.

        Returns:
            RedisHealthResult with connection health information
        """
        start_time = time.time()
        result = RedisHealthResult(
            component_name="redis_connection",
            check_type="connection_health",
            timestamp=datetime.now(UTC),
        )

        try:
            if not await self._ensure_client():
                result.status = RedisHealthStatus.FAILED
                result.error = "Failed to create Redis client"
                result.issues.append("Redis client creation failed")
                return result

            # Test basic connectivity
            ping_start = time.time()
            await self._client.ping()
            ping_duration = (time.time() - ping_start) * 1000

            self._last_ping_time = ping_duration
            self._connection_info.last_ping_ms = ping_duration
            self._connection_info.is_connected = True

            # Update connection info
            await self._update_connection_info()

            # Determine health status
            if ping_duration > self.config.max_latency_ms:
                result.status = RedisHealthStatus.WARNING
                result.warnings.append(f"High ping latency: {ping_duration:.1f}ms")
            else:
                result.status = RedisHealthStatus.HEALTHY

            result.success = True

        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            self._connection_info.is_connected = False
            self._connection_info.error_count = self._error_count
            self._connection_info.last_error = str(e)

            result.status = RedisHealthStatus.FAILED
            result.error = str(e)
            result.issues.append(f"Connection failed: {e!s}")

            logger.exception(f"Redis connection health check failed: {e}")

        finally:
            result.duration_ms = (time.time() - start_time) * 1000

        return result

    async def get_connection_info(self) -> RedisConnectionInfo:
        """Get detailed connection information.

        Returns:
            RedisConnectionInfo with current connection details
        """
        self._connection_info.uptime_seconds = time.time() - self._connection_start_time
        return self._connection_info

    async def test_connectivity(self) -> bool:
        """Test basic Redis connectivity.

        Returns:
            True if Redis is reachable, False otherwise
        """
        try:
            if not await self._ensure_client():
                return False

            await self._client.ping()
            return True

        except Exception as e:
            logger.debug(f"Redis connectivity test failed: {e}")
            return False

    async def check_failover_status(self) -> dict[str, Any]:
        """Check Redis failover and sentinel status.

        Returns:
            Dictionary with failover status information
        """
        status = {
            "failover_supported": False,
            "sentinel_configured": False,
            "master_status": "unknown",
            "replica_count": 0,
            "failover_history": [],
        }

        try:
            if not await self._ensure_client():
                return status

            # Get Redis info
            info = await self._client.info()

            # Check replication role
            role = info.get("role", "unknown")
            status["master_status"] = role

            if role == "master":
                status["replica_count"] = int(info.get("connected_slaves", 0))
                status["failover_supported"] = status["replica_count"] > 0

            elif role in {"slave", "replica"}:
                status["master_host"] = info.get("master_host", "unknown")
                status["master_port"] = info.get("master_port", 0)
                status["link_status"] = info.get("master_link_status", "unknown")

            # Check for sentinel configuration (if available)
            try:
                sentinel_info = await self._client.info("sentinel")
                if sentinel_info:
                    status["sentinel_configured"] = True
                    status["sentinel_masters"] = int(sentinel_info.get("sentinel_masters", 0))
            except Exception:
                # Sentinel info not available - not necessarily an error
                pass

        except Exception as e:
            logger.exception(f"Failed to check failover status: {e}")
            status["error"] = str(e)

        return status

    def get_connection_metrics(self) -> dict[str, float]:
        """Get connection performance metrics.

        Returns:
            Dictionary with connection performance metrics
        """
        return {
            "last_ping_ms": self._last_ping_time,
            "uptime_seconds": time.time() - self._connection_start_time,
            "error_count": float(self._error_count),
            "is_connected": float(self._connection_info.is_connected),
            "connection_time_ms": self._connection_info.connection_time_ms,
        }

    async def _update_connection_info(self) -> None:
        """Update internal connection information."""
        try:
            if not self._client:
                return

            # Get Redis info for connection details
            info = await self._client.info()

            self._connection_info.connection_time_ms = self._last_ping_time
            self._connection_info.error_count = self._error_count
            self._connection_info.last_error = self._last_error

            # Get connection pool information if available
            if hasattr(self._client, "connection_pool"):
                pool = self._client.connection_pool
                if hasattr(pool, "_created_connections"):
                    self._connection_info.connection_pool_size = len(pool._created_connections)
                if hasattr(pool, "_in_use_connections"):
                    self._connection_info.active_connections = len(pool._in_use_connections)

        except Exception as e:
            logger.debug(f"Failed to update connection info: {e}")

    async def close(self) -> None:
        """Close Redis connection and cleanup resources."""
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.debug(f"Error closing Redis client: {e}")
            finally:
                self._client = None
