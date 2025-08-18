"""L3 Database Cache Service for persistent caching with fallback capability.

Provides database-backed caching with 10-50ms response times for persistent
data storage and complex query result caching. Integrates with repository
pattern for clean architecture compliance.
"""

import json
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class DatabaseSessionProtocol(Protocol):
    """Protocol for database session operations."""
    
    async def execute(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        """Execute database query."""
        ...
        
    async def fetch_one(self, query: str, parameters: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Fetch single row from database."""
        ...
        
    async def fetch_all(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Fetch multiple rows from database."""
        ...


class L3DatabaseService:
    """Database cache service for L3 persistent caching operations.
    
    Designed for 10-50ms response times with database-backed persistence.
    Uses dedicated cache tables for storing serialized data with TTL support.
    Follows repository pattern for clean architecture compliance.
    
    Performance targets:
    - GET operations: <30ms
    - SET operations: <50ms
    - Batch operations: <100ms
    """

    def __init__(self, session_manager: Any = None) -> None:
        """Initialize L3 database service.
        
        Args:
            session_manager: Database session manager (repository pattern)
        """
        self._session_manager = session_manager
        self._created_at = datetime.now(UTC)
        
        # Performance tracking
        self._total_operations = 0
        self._successful_operations = 0
        self._failed_operations = 0
        self._total_response_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        
        # SQL queries
        self._init_queries()

    def _track_operation(self, start_time: float, success: bool, operation: str, key: str = "") -> None:
        """Track operation performance and log slow operations."""
        response_time = time.perf_counter() - start_time
        self._total_operations += 1
        self._total_response_time += response_time
        
        if success:
            self._successful_operations += 1
        else:
            self._failed_operations += 1
            
        # Log slow operations (should be <50ms)
        if response_time > 0.05:
            logger.warning(f"L3 Database {operation} took {response_time*1000:.2f}ms (key: {key[:50]}...)")

    def _init_queries(self) -> None:
        """Initialize SQL queries for cache operations."""
        self._queries = {
            "create_table": "CREATE TABLE IF NOT EXISTS cache_l3 (cache_key VARCHAR(255) PRIMARY KEY, cache_value TEXT NOT NULL, created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(), expires_at TIMESTAMP WITH TIME ZONE, access_count INTEGER DEFAULT 1, last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW())",
            "create_index": "CREATE INDEX IF NOT EXISTS idx_cache_l3_expires_at ON cache_l3 (expires_at) WHERE expires_at IS NOT NULL",
            "get": "SELECT cache_value, expires_at FROM cache_l3 WHERE cache_key = :key AND (expires_at IS NULL OR expires_at > NOW())",
            "get_with_update": "UPDATE cache_l3 SET access_count = access_count + 1, last_accessed = NOW() WHERE cache_key = :key AND (expires_at IS NULL OR expires_at > NOW()) RETURNING cache_value",
            "set": "INSERT INTO cache_l3 (cache_key, cache_value, expires_at) VALUES (:key, :value, :expires_at) ON CONFLICT (cache_key) DO UPDATE SET cache_value = EXCLUDED.cache_value, expires_at = EXCLUDED.expires_at, created_at = NOW(), access_count = 1, last_accessed = NOW()",
            "delete": "DELETE FROM cache_l3 WHERE cache_key = :key",
            "cleanup_expired": "DELETE FROM cache_l3 WHERE expires_at IS NOT NULL AND expires_at <= NOW()",
            "clear_all": "DELETE FROM cache_l3",
            
            "exists": "SELECT 1 FROM cache_l3 WHERE cache_key = :key AND (expires_at IS NULL OR expires_at > NOW())",
            "invalidate_pattern": "DELETE FROM cache_l3 WHERE cache_key LIKE :pattern AND (expires_at IS NULL OR expires_at > NOW())",
            "stats": "SELECT COUNT(*) as total_entries, COUNT(CASE WHEN expires_at IS NULL OR expires_at > NOW() THEN 1 END) as valid_entries, AVG(access_count) as avg_access_count, MAX(last_accessed) as last_access_time FROM cache_l3"
        }

    async def ensure_table_exists(self) -> bool:
        """Ensure cache table exists in database.
        
        Returns:
            True if table creation succeeded, False otherwise
        """
        if not self._session_manager:
            logger.warning("L3 Database cache: No session manager configured")
            return False
            
        try:
            async with self._get_session() as session:
                await session.execute(self._queries["create_table"])
                await session.execute(self._queries["create_index"])
                return True
                
        except Exception as e:
            logger.error(f"L3 Database cache table creation failed: {e}")
            return False

    async def get(self, key: str) -> Optional[Any]:
        """Get value from database cache."""
        if not self._session_manager:
            return None
            
        start_time = time.perf_counter()
        
        try:
            async with self._get_session() as session:
                result = await session.fetch_one(
                    self._queries["get_with_update"],
                    {"key": key}
                )
                
                if result:
                    value = json.loads(result["cache_value"])
                    self._cache_hits += 1
                    self._track_operation(start_time, True, "GET", key)
                    return value
                else:
                    self._cache_misses += 1
                    return None
                    
        except Exception as e:
            logger.warning(f"L3 Database GET error for key {key}: {e}")
            self._track_operation(start_time, False, "GET", key)
            return None

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in database cache."""
        if not self._session_manager:
            return False
            
        start_time = time.perf_counter()
        
        try:
            serialized_value = json.dumps(value, default=str)
            expires_at = None
            if ttl_seconds and ttl_seconds > 0:
                expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)
            
            async with self._get_session() as session:
                await session.execute(
                    self._queries["set"],
                    {"key": key, "value": serialized_value, "expires_at": expires_at}
                )
                
            self._track_operation(start_time, True, "SET", key)
            return True
            
        except Exception as e:
            logger.warning(f"L3 Database SET error for key {key}: {e}")
            self._track_operation(start_time, False, "SET", key)
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from database cache."""
        if not self._session_manager:
            return False
            
        start_time = time.perf_counter()
        
        try:
            async with self._get_session() as session:
                result = await session.execute(
                    self._queries["delete"],
                    {"key": key}
                )
                
                success = getattr(result, 'rowcount', 0) > 0
                self._track_operation(start_time, success, "DELETE", key)
                return success
                
        except Exception as e:
            logger.warning(f"L3 Database DELETE error for key {key}: {e}")
            self._track_operation(start_time, False, "DELETE", key)
            return False

    async def clear(self) -> None:
        """Clear all cache entries from database."""
        if not self._session_manager:
            return
            
        start_time = time.perf_counter()
        
        try:
            async with self._get_session() as session:
                await session.execute(self._queries["clear_all"])
                
            self._cache_hits = 0
            self._cache_misses = 0
            self._track_operation(start_time, True, "CLEAR")
            
        except Exception as e:
            logger.error(f"L3 Database CLEAR error: {e}")
            self._track_operation(start_time, False, "CLEAR")

    async def exists(self, key: str) -> bool:
        """Check if key exists in database cache."""
        if not self._session_manager:
            return False
            
        try:
            async with self._get_session() as session:
                result = await session.fetch_one(self._queries["exists"], {"key": key})
                return result is not None
        except Exception as e:
            logger.warning(f"L3 Database EXISTS error for key {key}: {e}")
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate L3 database cache entries matching pattern.
        
        Args:
            pattern: Pattern to match against cache keys (glob patterns converted to SQL LIKE)
            
        Returns:
            Number of entries invalidated
        """
        if not self._session_manager:
            return 0
            
        start_time = time.perf_counter()
        
        try:
            # SECURITY: Validate and sanitize pattern input to prevent SQL injection
            validated_pattern = self._validate_and_sanitize_pattern(pattern)
            if validated_pattern is None:
                logger.warning(f"Invalid cache pattern rejected: {pattern[:50]}...")
                return 0
            
            # Convert glob pattern to SQL LIKE pattern (now safe after validation)
            sql_pattern = validated_pattern.replace('*', '%').replace('?', '_')
            
            async with self._get_session() as session:
                result = await session.execute(
                    self._queries["invalidate_pattern"],
                    {"pattern": sql_pattern}
                )
                
                deleted_count = getattr(result, 'rowcount', 0)
                self._track_operation(start_time, True, "INVALIDATE_PATTERN", pattern)
                return deleted_count
                
        except Exception as e:
            logger.warning(f"L3 Database pattern invalidation failed '{pattern[:50]}...': {e}")
            self._track_operation(start_time, False, "INVALIDATE_PATTERN", pattern)
            return 0

    async def cleanup_expired(self) -> int:
        """Remove expired entries from database cache.
        
        Returns:
            Number of entries removed
        """
        if not self._session_manager:
            return 0
            
        try:
            async with self._get_session() as session:
                result = await session.execute(self._queries["cleanup_expired"])
                removed_count = getattr(result, 'rowcount', 0)
                
                if removed_count > 0:
                    logger.debug(f"L3 Database cache cleanup removed {removed_count} expired entries")
                
                return removed_count
                
        except Exception as e:
            logger.error(f"L3 Database cleanup error: {e}")
            return 0

    def _get_session(self):
        """Get database session using session manager.
        
        Returns:
            Database session context manager
        """
        if hasattr(self._session_manager, 'get_session'):
            return self._session_manager.get_session()
        else:
            # Fallback for different session manager interfaces
            return self._session_manager

    def _validate_and_sanitize_pattern(self, pattern: str) -> str | None:
        """Validate and sanitize cache pattern to prevent SQL injection.
        
        Args:
            pattern: Input pattern to validate
            
        Returns:
            Sanitized pattern or None if invalid
        """
        import re
        
        if not pattern or not isinstance(pattern, str):
            return None
            
        # Maximum pattern length to prevent DoS
        if len(pattern) > 256:
            logger.warning(f"Cache pattern too long: {len(pattern)} characters")
            return None
            
        # Allow only alphanumeric, colons, underscores, hyphens, dots, and glob wildcards
        if not re.match(r'^[a-zA-Z0-9:_\-.*?]+$', pattern):
            logger.warning(f"Cache pattern contains invalid characters: {pattern[:50]}...")
            return None
            
        # Prevent SQL injection attempts
        dangerous_patterns = [
            "'", '"', ';', '--', '/*', '*/', 'DROP', 'DELETE', 'INSERT', 
            'UPDATE', 'CREATE', 'ALTER', 'EXEC', 'UNION', 'SELECT',
            'FROM', 'WHERE', 'OR', 'AND', '=', '<', '>', 'LIKE'
        ]
        
        pattern_upper = pattern.upper()
        for dangerous in dangerous_patterns:
            if dangerous in pattern_upper:
                logger.error(f"Potential SQL injection attempt blocked: {pattern[:50]}...")
                return None
                
        return pattern

    def get_stats(self) -> dict[str, Any]:
        """Get database cache performance statistics."""
        total_ops = self._total_operations
        success_rate = self._successful_operations / total_ops if total_ops > 0 else 0
        avg_response_time = self._total_response_time / total_ops if total_ops > 0 else 0
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "total_operations": total_ops,
            "successful_operations": self._successful_operations,
            "failed_operations": self._failed_operations,
            "success_rate": success_rate,
            "avg_response_time_ms": avg_response_time * 1000,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "session_manager_available": self._session_manager is not None,
            "slo_target_ms": 50.0,
            "slo_compliant": avg_response_time < 0.05,
            "health_status": self._get_health_status(),
            "uptime_seconds": (datetime.now(UTC) - self._created_at).total_seconds(),
        }

    async def get_database_stats(self) -> dict[str, Any]:
        """Get statistics from database cache table."""
        if not self._session_manager:
            return {"error": "No session manager configured"}
            
        try:
            async with self._get_session() as session:
                result = await session.fetch_one(self._queries["stats"])
                
                if result:
                    return {
                        "total_entries": result["total_entries"],
                        "valid_entries": result["valid_entries"],
                        "expired_entries": result["total_entries"] - result["valid_entries"],
                        "avg_access_count": float(result["avg_access_count"] or 0),
                        "last_access_time": result["last_access_time"].isoformat() if result["last_access_time"] else None,
                    }
                return {"error": "Unable to fetch database statistics"}
        except Exception as e:
            logger.error(f"L3 Database stats error: {e}")
            return {"error": str(e)}

    def _get_health_status(self) -> str:
        """Get health status based on performance metrics."""
        if not self._session_manager:
            return "unhealthy"
        if self._total_operations == 0:
            return "healthy"
            
        success_rate = self._successful_operations / self._total_operations
        avg_response_time = self._total_response_time / self._total_operations
        
        if success_rate < 0.5 or avg_response_time > 0.1:
            return "unhealthy"
        elif success_rate < 0.9 or avg_response_time > 0.05:
            return "degraded"
        return "healthy"

    async def health_check(self) -> dict[str, Any]:
        """Perform health check for database cache."""
        start_time = time.perf_counter()
        
        try:
            if not self._session_manager:
                return {"healthy": False, "error": "No session manager configured", "timestamp": datetime.now(UTC).isoformat()}
            
            table_exists = await self.ensure_table_exists()
            test_key = f"health_check_{int(time.time())}"
            test_value = {"test": True}
            
            # Test operations
            set_success = await self.set(test_key, test_value, ttl_seconds=60)
            get_result = await self.get(test_key) if set_success else None
            delete_success = await self.delete(test_key) if set_success else False
            
            operations_success = set_success and get_result and delete_success
            total_time = time.perf_counter() - start_time
            
            return {
                "healthy": table_exists and operations_success,
                "checks": {"table_exists": table_exists, "operations": operations_success},
                "performance": {"total_check_time_ms": total_time * 1000, "meets_slo": total_time < 0.1},
                "timestamp": datetime.now(UTC).isoformat(),
            }
            
        except Exception as e:
            logger.error(f"L3 Database health check failed: {e}")
            return {"healthy": False, "error": str(e), "timestamp": datetime.now(UTC).isoformat()}