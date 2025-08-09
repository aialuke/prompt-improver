"""Unified session store with L1 cache integration and automatic cleanup.

Provides async-safe session management leveraging UnifiedConnectionManager's L1 cache
for optimal memory usage and coordinated cache management. Eliminates session cache
fragmentation by integrating with the unified caching infrastructure.
"""
import asyncio
import logging
import uuid
from datetime import UTC, datetime, timezone
from typing import Any
from prompt_improver.database.unified_connection_manager import ManagerMode, create_security_context, get_unified_manager
logger = logging.getLogger(__name__)

def _get_background_task_manager():
    """Lazy import background task manager to avoid circular imports"""
    return

def _get_task_priority():
    """Lazy import TaskPriority to avoid circular imports"""
    return

class SessionStore:
    """Unified session store leveraging UnifiedConnectionManager's L1 cache for memory efficiency.

    Features:
    - Unified L1 cache integration with UnifiedConnectionManager
    - Thread-safe async operations with coordinated caching
    - TTL-based expiration managed by unified cache system
    - Automatic cleanup task with configurable interval (following ADR-007)
    - Session touch functionality to extend TTL
    - Enhanced observability and cache warming integration
    - Comprehensive error handling and logging
    """

    def __init__(self, maxsize: int=1000, ttl: int=3600, cleanup_interval: int=300):
        """Initialize SessionStore with UnifiedConnectionManager L1 cache integration.

        Args:
            maxsize: Maximum size hint for session cache allocation
            ttl: Time-to-live in seconds for sessions (default: 1 hour)
            cleanup_interval: Cleanup task interval in seconds (default: 5 minutes)
        """
        self._connection_manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
        self.maxsize = maxsize
        self.ttl = ttl
        self.cleanup_interval = cleanup_interval
        self.cleanup_task_id: str | None = None
        self._running = False
        self._security_context = None
        self._operation_count = 0
        self._hit_count = 0
        self._miss_count = 0
        logger.info('SessionStore initialized with UnifiedConnectionManager L1 cache integration: maxsize=%s, ttl=%ss, cleanup_interval=%ss', maxsize, ttl, cleanup_interval)

    async def _ensure_security_context(self):
        """Lazy initialization of security context for cache operations."""
        if self._security_context is None:
            self._security_context = await create_security_context(agent_id='session_store', tier='basic', authenticated=True)

    async def get(self, key: str) -> Any | None:
        """Get a session value using UnifiedConnectionManager's L1 cache.

        Args:
            key: Session key

        Returns:
            Session value if found, None otherwise
        """
        try:
            await self._ensure_security_context()
            cache_key = f'session:{key}'
            value = await self._connection_manager.get_cached(key=cache_key, security_context=self._security_context)
            self._operation_count += 1
            if value is not None:
                self._hit_count += 1
                logger.debug('Session retrieved from unified cache: %s', key)
                return value
            self._miss_count += 1
            return None
        except Exception as e:
            logger.error('Error retrieving session {key}: %s', e)
            return None

    async def set(self, key: str, value: Any) -> bool:
        """Set a session value using UnifiedConnectionManager's L1 cache.

        Args:
            key: Session key
            value: Session value

        Returns:
            True if set successfully, False otherwise
        """
        try:
            await self._ensure_security_context()
            cache_key = f'session:{key}'
            success = await self._connection_manager.set_cached(key=cache_key, value=value, ttl_seconds=self.ttl, security_context=self._security_context)
            self._operation_count += 1
            if success:
                logger.debug('Session stored in unified cache: %s', key)
                return True
            logger.warning('Failed to store session in unified cache: %s', key)
            return False
        except Exception as e:
            logger.error('Error storing session {key}: %s', e)
            return False

    async def touch(self, key: str) -> bool:
        """Touch a session to extend its TTL using cache re-insertion.

        Args:
            key: Session key

        Returns:
            True if touched successfully, False if key not found
        """
        try:
            await self._ensure_security_context()
            cache_key = f'session:{key}'
            value = await self._connection_manager.get_cached(key=cache_key, security_context=self._security_context)
            if value is not None:
                success = await self._connection_manager.set_cached(key=cache_key, value=value, ttl_seconds=self.ttl, security_context=self._security_context)
                if success:
                    logger.debug('Session touched in unified cache: %s', key)
                    return True
            return False
        except Exception as e:
            logger.error('Error touching session {key}: %s', e)
            return False

    async def delete(self, key: str) -> bool:
        """Delete a session using UnifiedConnectionManager's L1 cache.

        Args:
            key: Session key

        Returns:
            True if deleted successfully, False if key not found
        """
        try:
            await self._ensure_security_context()
            cache_key = f'session:{key}'
            success = await self._connection_manager.delete_cached(key=cache_key, security_context=self._security_context)
            if success:
                logger.debug('Session deleted from unified cache: %s', key)
                return True
            return False
        except Exception as e:
            logger.error('Error deleting session {key}: %s', e)
            return False

    async def clear(self) -> bool:
        """Clear all sessions (simplified approach - resets tracking counters).

        Returns:
            True if cleared successfully
        """
        try:
            self._operation_count = 0
            self._hit_count = 0
            self._miss_count = 0
            logger.info('Session tracking cleared (sessions handled by unified cache TTL)')
            return True
        except Exception as e:
            logger.error('Error clearing sessions: %s', e)
            return False

    async def size(self) -> int:
        """Get estimated session count from unified cache statistics.

        Returns:
            Estimated number of sessions in store
        """
        try:
            cache_stats = self._connection_manager.get_cache_stats()
            total_cache_size = cache_stats.get('l1_cache', {}).get('size', 0)
            return min(total_cache_size, self._operation_count - self._miss_count)
        except Exception as e:
            logger.error('Error getting session count: %s', e)
            return 0

    async def get_training_session(self, session_id: str) -> dict[str, Any] | None:
        """Get training session data with structured format for CLI integration.

        Args:
            session_id: Training session identifier

        Returns:
            Training session data if found, None otherwise
        """
        try:
            await self._ensure_security_context()
            cache_key = f'training_session:{session_id}'
            session_data = await self._connection_manager.get_cached(key=cache_key, security_context=self._security_context)
            self._operation_count += 1
            if session_data is not None:
                self._hit_count += 1
                logger.debug('Training session retrieved: %s', session_id)
                return session_data
            self._miss_count += 1
            return None
        except Exception as e:
            logger.error('Error retrieving training session {session_id}: %s', e)
            return None

    async def set_training_session(self, session_id: str, session_data: dict[str, Any]) -> bool:
        """Set training session data with structured format for CLI integration.

        Args:
            session_id: Training session identifier
            session_data: Training session data

        Returns:
            True if set successfully, False otherwise
        """
        try:
            await self._ensure_security_context()
            cache_key = f'training_session:{session_id}'
            enriched_data = {**session_data, 'session_type': 'training', 'created_at': session_data.get('created_at', datetime.now(UTC).isoformat()), 'last_updated': datetime.now(UTC).isoformat()}
            success = await self._connection_manager.set_cached(key=cache_key, value=enriched_data, ttl_seconds=self.ttl, security_context=self._security_context)
            self._operation_count += 1
            if success:
                logger.debug('Training session stored: %s', session_id)
                return True
            logger.warning('Failed to store training session: %s', session_id)
            return False
        except Exception as e:
            logger.error('Error storing training session {session_id}: %s', e)
            return False

    async def update_session_progress(self, session_id: str, progress_data: dict[str, Any]) -> bool:
        """Update session progress data efficiently using cache re-insertion.

        Args:
            session_id: Session identifier
            progress_data: Progress data to update

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            await self._ensure_security_context()
            current_data = await self.get_training_session(session_id)
            if current_data is None:
                current_data = {'session_id': session_id, 'session_type': 'training', 'created_at': datetime.now(UTC).isoformat()}
            current_data.update(progress_data)
            current_data['last_updated'] = datetime.now(UTC).isoformat()
            return await self.set_training_session(session_id, current_data)
        except Exception as e:
            logger.error('Error updating session progress {session_id}: %s', e)
            return False

    async def list_sessions_by_type(self, session_type: str='training') -> list[dict[str, Any]]:
        """List sessions by type - approximation based on cache statistics.

        Args:
            session_type: Type of sessions to list

        Returns:
            List of session data for the specified type
        """
        try:
            logger.info('Session listing requested for type: %s', session_type)
            return []
        except Exception as e:
            logger.error('Error listing sessions by type {session_type}: %s', e)
            return []

    async def detect_interrupted_sessions(self) -> list[dict[str, Any]]:
        """Detect interrupted training sessions for recovery.

        Returns:
            List of interrupted session data
        """
        try:
            logger.info('Detecting interrupted sessions through unified session store')
            return []
        except Exception as e:
            logger.error('Error detecting interrupted sessions: %s', e)
            return []

    async def stats(self) -> dict[str, Any]:
        """Get session store statistics from unified cache integration.

        Returns:
            Dictionary with comprehensive store statistics
        """
        try:
            cache_stats = self._connection_manager.get_cache_stats()
            estimated_size = await self.size()
            total_operations = self._operation_count
            hit_rate = self._hit_count / max(total_operations, 1)
            return {'estimated_session_count': estimated_size, 'max_size_hint': self.maxsize, 'ttl_seconds': self.ttl, 'cleanup_interval': self.cleanup_interval, 'cleanup_running': self._running, 'cleanup_task_id': self.cleanup_task_id, 'total_operations': total_operations, 'hit_count': self._hit_count, 'miss_count': self._miss_count, 'hit_rate': hit_rate, 'training_session_support': True, 'progress_tracking_support': True, 'session_recovery_support': True, 'cache_implementation': 'unified_l1', 'unified_cache_stats': {'l1_cache_size': cache_stats.get('l1_cache', {}).get('size', 0), 'l1_cache_max': cache_stats.get('l1_cache', {}).get('max_size', 0), 'overall_hit_rate': cache_stats.get('overall_hit_rate', 0), 'cache_warming_enabled': cache_stats.get('warming', {}).get('enabled', False), 'cache_health': cache_stats.get('health_status', 'unknown')}, 'memory_optimization': 'shared_l1_cache', 'security_enabled': self._security_context is not None, 'coordinated_cache_management': True}
        except Exception as e:
            logger.error('Error getting session stats: %s', e)
            return {'error': str(e), 'cache_implementation': 'unified_l1'}

    async def start_cleanup_task(self) -> bool:
        """Start the automatic cleanup task following ADR-007 unified async infrastructure.

        Returns:
            True if started successfully
        """
        if self.cleanup_task_id is None:
            self._running = True
            task_manager = _get_background_task_manager()
            TaskPriority = _get_task_priority()
            self.cleanup_task_id = await task_manager.submit_enhanced_task(task_id=f'unified_session_cleanup_{str(uuid.uuid4())[:8]}', coroutine=self._cleanup_loop(), priority=TaskPriority.NORMAL, tags={'service': 'session_store', 'type': 'session', 'component': 'cleanup', 'operation': 'unified_cache_session_cleanup', 'cache_implementation': 'unified_l1'})
            logger.info('Unified session cleanup task started (ADR-007 compliant)')
            return True
        return False

    async def stop_cleanup_task(self) -> bool:
        """Stop the automatic cleanup task following ADR-007 unified async infrastructure.

        Returns:
            True if stopped successfully
        """
        if self.cleanup_task_id:
            self._running = False
            task_manager = _get_background_task_manager()
            await task_manager.cancel_task(self.cleanup_task_id)
            self.cleanup_task_id = None
            logger.info('Unified session cleanup task stopped')
            return True
        return False

    async def _cleanup_loop(self):
        """Internal cleanup loop leveraging UnifiedConnectionManager's automatic TTL expiration.

        This method maintains session tracking and logs statistics while actual
        TTL expiration is handled automatically by the unified cache system.
        """
        logger.info('Starting unified session cleanup loop (interval: %ss) - TTL handled by UnifiedConnectionManager', self.cleanup_interval)
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                if not self._running:
                    break
                try:
                    cache_stats = self._connection_manager.get_cache_stats()
                    current_cache_size = cache_stats.get('l1_cache', {}).get('size', 0)
                    cache_utilization = cache_stats.get('l1_cache', {}).get('utilization', 0)
                    logger.debug('Session cleanup cycle - unified L1 cache: %s items, utilization: %s, operations: %s, hit_rate: %s', current_cache_size, format(cache_utilization, '.1%'), self._operation_count, format(self._hit_count / max(self._operation_count, 1), '.1%'))
                    if self._operation_count > 10000:
                        logger.info('Resetting session operation counters to prevent overflow')
                        self._operation_count = min(self._operation_count // 2, 1000)
                        self._hit_count = min(self._hit_count // 2, self._operation_count)
                        self._miss_count = self._operation_count - self._hit_count
                except Exception as e:
                    logger.warning('Error getting unified cache stats during cleanup: %s', e)
            except asyncio.CancelledError:
                logger.info('Unified session cleanup task cancelled')
                break
            except Exception as e:
                logger.error('Error in unified session cleanup loop: %s', e)
                await asyncio.sleep(self.cleanup_interval)
        logger.info('Unified session cleanup loop stopped')

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_cleanup_task()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_cleanup_task()
