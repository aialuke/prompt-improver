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

    def _init_queries(self) -> None:
        """Initialize SQL queries for cache operations."""
        self._queries = {
            "create_table": """
                CREATE TABLE IF NOT EXISTS cache_l3 (
                    cache_key VARCHAR(255) PRIMARY KEY,
                    cache_value TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    expires_at TIMESTAMP WITH TIME ZONE,
                    access_count INTEGER DEFAULT 1,
                    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """,
            
            "create_index": """
                CREATE INDEX IF NOT EXISTS idx_cache_l3_expires_at 
                ON cache_l3 (expires_at) WHERE expires_at IS NOT NULL
            """,
            
            "get": """
                SELECT cache_value, expires_at 
                FROM cache_l3 
                WHERE cache_key = :key 
                AND (expires_at IS NULL OR expires_at > NOW())
            """,
            
            "get_with_update": """
                UPDATE cache_l3 
                SET access_count = access_count + 1, last_accessed = NOW()
                WHERE cache_key = :key 
                AND (expires_at IS NULL OR expires_at > NOW())
                RETURNING cache_value
            """,
            
            "set": """
                INSERT INTO cache_l3 (cache_key, cache_value, expires_at)
                VALUES (:key, :value, :expires_at)
                ON CONFLICT (cache_key) 
                DO UPDATE SET 
                    cache_value = EXCLUDED.cache_value,
                    expires_at = EXCLUDED.expires_at,
                    created_at = NOW(),
                    access_count = 1,
                    last_accessed = NOW()
            """,
            
            "delete": """
                DELETE FROM cache_l3 WHERE cache_key = :key
            """,
            
            "cleanup_expired": """
                DELETE FROM cache_l3 
                WHERE expires_at IS NOT NULL AND expires_at <= NOW()
            """,
            
            "clear_all": """
                DELETE FROM cache_l3
            """,
            
            "exists": """
                SELECT 1 FROM cache_l3 
                WHERE cache_key = :key 
                AND (expires_at IS NULL OR expires_at > NOW())
            """,
            
            "stats": """
                SELECT 
                    COUNT(*) as total_entries,
                    COUNT(CASE WHEN expires_at IS NULL OR expires_at > NOW() THEN 1 END) as valid_entries,
                    AVG(access_count) as avg_access_count,
                    MAX(last_accessed) as last_access_time
                FROM cache_l3
            """
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
        """Get value from database cache.
        
        Args:
            key: Cache key
            
        Returns:
            Deserialized value or None if not found/expired
        """
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
                    # Deserialize cached value
                    value = json.loads(result["cache_value"])
                    self._successful_operations += 1
                    self._cache_hits += 1
                    return value
                else:
                    self._cache_misses += 1
                    return None
                    
        except Exception as e:
            self._failed_operations += 1
            logger.warning(f"L3 Database GET error for key {key}: {e}")
            return None
            
        finally:
            response_time = time.perf_counter() - start_time
            self._total_operations += 1
            self._total_response_time += response_time
            
            # Log slow operations (should be <50ms)
            if response_time > 0.05:
                logger.warning(
                    f"L3 Database GET operation took {response_time*1000:.2f}ms (key: {key[:50]}...)"
                )

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in database cache.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl_seconds: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self._session_manager:
            return False
            
        start_time = time.perf_counter()
        
        try:
            # Serialize value
            serialized_value = json.dumps(value, default=str)
            
            # Calculate expiration
            expires_at = None
            if ttl_seconds and ttl_seconds > 0:
                expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)
            
            async with self._get_session() as session:
                await session.execute(
                    self._queries["set"],
                    {
                        "key": key,
                        "value": serialized_value,
                        "expires_at": expires_at,
                    }
                )
                
            self._successful_operations += 1
            return True
            
        except Exception as e:
            self._failed_operations += 1
            logger.warning(f"L3 Database SET error for key {key}: {e}")
            return False
            
        finally:
            response_time = time.perf_counter() - start_time
            self._total_operations += 1
            self._total_response_time += response_time
            
            if response_time > 0.05:
                logger.warning(
                    f"L3 Database SET operation took {response_time*1000:.2f}ms (key: {key[:50]}...)"
                )

    async def delete(self, key: str) -> bool:
        """Delete key from database cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted, False otherwise
        """
        if not self._session_manager:
            return False
            
        start_time = time.perf_counter()
        
        try:
            async with self._get_session() as session:
                result = await session.execute(
                    self._queries["delete"],
                    {"key": key}
                )
                
                # Check if any rows were affected
                success = getattr(result, 'rowcount', 0) > 0
                
                if success:
                    self._successful_operations += 1
                else:
                    self._failed_operations += 1
                    
                return success
                
        except Exception as e:
            self._failed_operations += 1
            logger.warning(f"L3 Database DELETE error for key {key}: {e}")
            return False
            
        finally:
            response_time = time.perf_counter() - start_time
            self._total_operations += 1
            self._total_response_time += response_time

    async def clear(self) -> None:
        """Clear all cache entries from database."""
        if not self._session_manager:
            return
            
        start_time = time.perf_counter()
        
        try:
            async with self._get_session() as session:
                await session.execute(self._queries["clear_all"])
                
            # Reset statistics
            self._cache_hits = 0
            self._cache_misses = 0
            self._successful_operations += 1
            
        except Exception as e:
            self._failed_operations += 1
            logger.error(f"L3 Database CLEAR error: {e}")
            
        finally:
            response_time = time.perf_counter() - start_time
            self._total_operations += 1
            self._total_response_time += response_time

    async def exists(self, key: str) -> bool:
        """Check if key exists in database cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists and not expired, False otherwise
        """
        if not self._session_manager:
            return False
            
        try:
            async with self._get_session() as session:
                result = await session.fetch_one(
                    self._queries["exists"],
                    {"key": key}
                )
                return result is not None
                
        except Exception as e:
            logger.warning(f"L3 Database EXISTS error for key {key}: {e}")
            return False

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

    def get_stats(self) -> dict[str, Any]:
        """Get database cache performance statistics.
        
        Returns:
            Dictionary with performance and storage statistics
        """
        total_ops = self._total_operations
        success_rate = (
            self._successful_operations / total_ops if total_ops > 0 else 0
        )
        avg_response_time = (
            self._total_response_time / total_ops if total_ops > 0 else 0
        )
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            # Core metrics
            "total_operations": total_ops,
            "successful_operations": self._successful_operations,
            "failed_operations": self._failed_operations,
            "success_rate": success_rate,
            "avg_response_time_ms": avg_response_time * 1000,
            
            # Cache metrics
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            
            # Database specific
            "session_manager_available": self._session_manager is not None,
            
            # SLO compliance
            "slo_target_ms": 50.0,
            "slo_compliant": avg_response_time < 0.05,
            
            # Health indicators
            "health_status": self._get_health_status(),
            "uptime_seconds": (datetime.now(UTC) - self._created_at).total_seconds(),
        }

    async def get_database_stats(self) -> dict[str, Any]:
        """Get statistics from database cache table.
        
        Returns:
            Database-level statistics
        """
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
                else:
                    return {"error": "Unable to fetch database statistics"}
                    
        except Exception as e:
            logger.error(f"L3 Database stats error: {e}")
            return {"error": str(e)}

    def _get_health_status(self) -> str:
        """Get health status based on performance metrics.
        
        Returns:
            Health status: "healthy", "degraded", or "unhealthy"
        """
        if not self._session_manager:
            return "unhealthy"
            
        if self._total_operations == 0:
            return "healthy"
            
        success_rate = self._successful_operations / self._total_operations
        avg_response_time = self._total_response_time / self._total_operations
        
        # Health thresholds
        if success_rate < 0.5 or avg_response_time > 0.1:  # 100ms
            return "unhealthy"
        elif success_rate < 0.9 or avg_response_time > 0.05:  # 50ms
            return "degraded"
        else:
            return "healthy"

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check for database cache.
        
        Returns:
            Health check results with detailed status
        """
        start_time = time.perf_counter()
        
        try:
            if not self._session_manager:
                return {
                    "healthy": False,
                    "error": "No session manager configured",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            
            # Test table existence
            table_exists = await self.ensure_table_exists()
            
            # Test basic operations
            test_key = f"health_check_{int(time.time())}"
            test_value = {"test": True, "timestamp": time.time()}
            
            # Test set
            set_success = await self.set(test_key, test_value, ttl_seconds=60)
            
            # Test get
            get_result = None
            if set_success:
                get_result = await self.get(test_key)
            
            # Test delete
            delete_success = False
            if set_success:
                delete_success = await self.delete(test_key)
            
            operations_success = (
                set_success and 
                get_result is not None and 
                get_result.get("test") is True and
                delete_success
            )
            
            total_time = time.perf_counter() - start_time
            
            return {
                "healthy": table_exists and operations_success,
                "checks": {
                    "table_exists": table_exists,
                    "operations": {
                        "set_success": set_success,
                        "get_success": get_result is not None,
                        "delete_success": delete_success,
                        "value_match": get_result == test_value if get_result else False,
                    },
                },
                "performance": {
                    "total_check_time_ms": total_time * 1000,
                    "meets_slo": total_time < 0.1,  # 100ms health check SLO
                },
                "stats": self.get_stats(),
                "database_stats": await self.get_database_stats(),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            
        except Exception as e:
            logger.error(f"L3 Database health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "stats": self.get_stats(),
                "timestamp": datetime.now(UTC).isoformat(),
            }