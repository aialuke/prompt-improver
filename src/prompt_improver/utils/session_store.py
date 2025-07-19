"""In-memory session store with TTL and automatic cleanup.

Provides async-safe session management with Time-To-Live (TTL) support
and automatic cleanup of expired sessions.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from cachetools import TTLCache

logger = logging.getLogger(__name__)


class SessionStore:
    """In-memory session store with TTL and automatic cleanup.

    Features:
    - Thread-safe async operations with locks
    - TTL-based expiration for all sessions
    - Automatic cleanup task with configurable interval
    - Session touch functionality to extend TTL
    - Comprehensive error handling and logging
    """

    def __init__(
        self, maxsize: int = 1000, ttl: int = 3600, cleanup_interval: int = 300
    ):
        """Initialize SessionStore.

        Args:
            maxsize: Maximum number of sessions to store
            ttl: Time-to-live in seconds for sessions (default: 1 hour)
            cleanup_interval: Cleanup task interval in seconds (default: 5 minutes)
        """
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.lock = asyncio.Lock()
        self.cleanup_interval = cleanup_interval
        self.cleanup_task: asyncio.Task | None = None
        self._running = False

        logger.info(
            f"SessionStore initialized: maxsize={maxsize}, ttl={ttl}s, cleanup_interval={cleanup_interval}s"
        )

    async def get(self, key: str) -> Any | None:
        """Get a session value by key.

        Args:
            key: Session key

        Returns:
            Session value if found, None otherwise
        """
        async with self.lock:
            try:
                value = self.cache.get(key)
                if value is not None:
                    logger.debug(f"Session retrieved: {key}")
                return value
            except Exception as e:
                logger.error(f"Error retrieving session {key}: {e}")
                return None

    async def set(self, key: str, value: Any) -> bool:
        """Set a session value.

        Args:
            key: Session key
            value: Session value

        Returns:
            True if set successfully, False otherwise
        """
        async with self.lock:
            try:
                self.cache[key] = value
                logger.debug(f"Session stored: {key}")
                return True
            except Exception as e:
                logger.error(f"Error storing session {key}: {e}")
                return False

    async def touch(self, key: str) -> bool:
        """Touch a session to extend its TTL.

        Args:
            key: Session key

        Returns:
            True if touched successfully, False if key not found
        """
        async with self.lock:
            try:
                if key in self.cache:
                    # Remove and re-add to refresh TTL
                    value = self.cache.pop(key)
                    self.cache[key] = value
                    logger.debug(f"Session touched: {key}")
                    return True
                return False
            except Exception as e:
                logger.error(f"Error touching session {key}: {e}")
                return False

    async def delete(self, key: str) -> bool:
        """Delete a session.

        Args:
            key: Session key

        Returns:
            True if deleted successfully, False if key not found
        """
        async with self.lock:
            try:
                if key in self.cache:
                    del self.cache[key]
                    logger.debug(f"Session deleted: {key}")
                    return True
                return False
            except Exception as e:
                logger.error(f"Error deleting session {key}: {e}")
                return False

    async def clear(self) -> bool:
        """Clear all sessions.

        Returns:
            True if cleared successfully
        """
        async with self.lock:
            try:
                self.cache.clear()
                logger.info("All sessions cleared")
                return True
            except Exception as e:
                logger.error(f"Error clearing sessions: {e}")
                return False

    async def size(self) -> int:
        """Get the current number of sessions.

        Returns:
            Number of sessions in store
        """
        async with self.lock:
            return len(self.cache)

    async def stats(self) -> dict[str, Any]:
        """Get session store statistics.

        Returns:
            Dictionary with store statistics
        """
        async with self.lock:
            return {
                "current_size": len(self.cache),
                "max_size": self.cache.maxsize,
                "ttl_seconds": self.cache.ttl,
                "cleanup_interval": self.cleanup_interval,
                "cleanup_running": self._running,
            }

    async def start_cleanup_task(self) -> bool:
        """Start the automatic cleanup task.

        Returns:
            True if started successfully
        """
        if self.cleanup_task is None or self.cleanup_task.done():
            self._running = True
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Session cleanup task started")
            return True
        return False

    async def stop_cleanup_task(self) -> bool:
        """Stop the automatic cleanup task.

        Returns:
            True if stopped successfully
        """
        if self.cleanup_task and not self.cleanup_task.done():
            self._running = False
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Session cleanup task stopped")
            return True
        return False

    async def _cleanup_loop(self):
        """Internal cleanup loop that runs periodically.

        This method expires old sessions and logs cleanup statistics.
        """
        logger.info(
            f"Starting session cleanup loop (interval: {self.cleanup_interval}s)"
        )

        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)

                if not self._running:
                    break

                # Perform cleanup
                async with self.lock:
                    initial_size = len(self.cache)

                    # Force expiration of TTL items
                    self.cache.expire()

                    final_size = len(self.cache)
                    expired_count = initial_size - final_size

                    if expired_count > 0:
                        logger.info(
                            f"Cleaned up {expired_count} expired sessions ({final_size} remaining)"
                        )
                    else:
                        logger.debug(
                            f"No expired sessions to clean up ({final_size} active)"
                        )

            except asyncio.CancelledError:
                logger.info("Session cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in session cleanup loop: {e}")
                # Continue running despite errors
                await asyncio.sleep(self.cleanup_interval)

        logger.info("Session cleanup loop stopped")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_cleanup_task()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_cleanup_task()
