"""L3 Database fallback cache with query result caching.

This module provides database-backed caching as the ultimate fallback layer,
extracted from unified_connection_manager.py, implementing:

- DatabaseCache: Query result caching with PostgreSQL backend and compression support
- DatabaseCacheEntry: Persistent cache entry with metadata and JSONB storage
- DatabaseMetrics: Comprehensive database cache performance monitoring
- Transaction-safe cache operations with automatic rollback
- Long-term storage optimization with compression
- Integration with PostgreSQLPoolManager
- Cache invalidation on data changes
- JSONB storage for structured data (PostgreSQL)

Designed for <50ms response times with comprehensive observability and data durability.
"""

import asyncio
import json
import logging
import time
import warnings
import zlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    warnings.warn("asyncpg not available. Install with: pip install asyncpg")
    ASYNCPG_AVAILABLE = False
    asyncpg = Any

try:
    from opentelemetry import metrics, trace
    from opentelemetry.metrics import Counter, Gauge, Histogram

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    Counter = Any
    Gauge = Any
    Histogram = Any

from prompt_improver.common.types import SecurityContext

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Compression algorithms for large cache values."""

    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"


class DatabaseCacheStorageType(Enum):
    """Database storage types for cache values."""

    TEXT = "text"
    JSONB = "jsonb"
    BYTEA = "bytea"


@dataclass
class DatabaseCacheConfig:
    """Configuration for Database L3 cache."""

    # Database connection settings
    host: str = "localhost"
    port: int = 5432
    database: str = "prompt_improver"
    username: str = "postgres"
    password: Optional[str] = None

    # Connection pooling
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: float = 10.0

    # Cache table configuration
    cache_table_name: str = "cache_entries"
    schema_name: str = "public"

    # Storage and compression
    storage_type: DatabaseCacheStorageType = DatabaseCacheStorageType.JSONB
    compression_type: CompressionType = CompressionType.ZLIB
    compression_threshold: int = 1024  # bytes

    # Performance settings
    batch_size: int = 1000
    vacuum_interval_hours: int = 24
    cleanup_expired_interval_hours: int = 6

    # Timeouts (seconds)
    query_timeout: float = 30.0
    transaction_timeout: float = 60.0

    # SSL settings
    ssl_enabled: bool = False
    ssl_require: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.min_connections <= 0:
            raise ValueError("min_connections must be greater than 0")
        if self.max_connections < self.min_connections:
            raise ValueError("max_connections must be >= min_connections")


@dataclass
class DatabaseCacheEntry:
    """Database cache entry with compression and metadata."""

    key: str
    value: Any
    stored_value: Union[str, bytes]
    storage_type: DatabaseCacheStorageType
    compressed: bool
    compression_type: CompressionType
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    size_bytes: int = 0
    checksum: Optional[str] = None
    security_context_id: Optional[str] = None
    tags: Set[str] = field(default_factory=set)

    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = len(
                self.stored_value.encode()
                if isinstance(self.stored_value, str)
                else self.stored_value
            )

    def is_expired(self) -> bool:
        """Check if the cache entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.now(UTC) > self.expires_at

    def time_until_expiry(self) -> Optional[float]:
        """Get seconds until expiry, or None if no expiration."""
        if self.expires_at is None:
            return None
        remaining = (self.expires_at - datetime.now(UTC)).total_seconds()
        return max(0, remaining)

    def touch(self) -> None:
        """Update access tracking."""
        self.last_accessed = datetime.now(UTC)
        self.access_count += 1


class DatabaseMetrics:
    """Comprehensive database cache metrics tracking."""

    def __init__(self, service_name: str = "database_cache", connection_pool=None):
        self.service_name = service_name
        self.connection_pool = connection_pool

        # Basic metrics
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0
        self.compressions = 0
        self.decompressions = 0

        # Performance metrics
        self.response_times: List[float] = []
        self.query_times: List[float] = []
        self.compression_ratios: List[float] = []

        # Storage metrics
        self.total_entries = 0
        self.total_size_bytes = 0
        self.compressed_entries = 0

        # OpenTelemetry setup
        self.operations_counter: Optional[Counter] = None
        self.response_time_histogram: Optional[Histogram] = None
        self.storage_gauge: Optional[Gauge] = None
        self.compression_histogram: Optional[Histogram] = None

        if OPENTELEMETRY_AVAILABLE:
            self._setup_telemetry()

    def _setup_telemetry(self) -> None:
        """Setup OpenTelemetry metrics."""
        try:
            meter = metrics.get_meter(f"prompt_improver.cache.{self.service_name}")

            self.operations_counter = meter.create_counter(
                "database_cache_operations_total",
                description="Total database cache operations by type and result",
                unit="1",
            )

            self.response_time_histogram = meter.create_histogram(
                "database_cache_response_time_seconds",
                description="Database cache operation response times",
                unit="s",
            )

            self.storage_gauge = meter.create_gauge(
                "database_cache_storage_bytes",
                description="Total bytes stored in database cache",
                unit="bytes",
            )

            self.compression_histogram = meter.create_histogram(
                "database_cache_compression_ratio",
                description="Compression ratio achieved for cached values",
                unit="1",
            )

            logger.debug(
                f"OpenTelemetry metrics initialized for Database {self.service_name}"
            )
        except Exception as e:
            logger.warning(f"Failed to setup Database OpenTelemetry metrics: {e}")

    def record_operation(
        self,
        operation: str,
        status: str,
        duration_ms: float = 0,
        security_context_id: Optional[str] = None,
        size_bytes: int = 0,
    ) -> None:
        """Record database cache operation metrics."""
        # Update counters
        if status == "hit":
            self.hits += 1
        elif status == "miss":
            self.misses += 1
        elif status == "error":
            self.errors += 1

        if operation == "set":
            self.sets += 1
            self.total_size_bytes += size_bytes
        elif operation == "delete":
            self.deletes += 1

        # Record response time
        if duration_ms > 0:
            self.response_times.append(duration_ms)
            # Keep only recent times
            if len(self.response_times) > 10000:
                self.response_times = self.response_times[-5000:]

        # OpenTelemetry metrics
        if self.operations_counter:
            self.operations_counter.add(
                1,
                {
                    "operation": operation,
                    "status": status,
                    "security_context": security_context_id or "none",
                },
            )

        if self.response_time_histogram and duration_ms > 0:
            self.response_time_histogram.record(duration_ms / 1000.0)

    def record_compression(self, original_size: int, compressed_size: int) -> None:
        """Record compression metrics."""
        self.compressions += 1
        if original_size > 0:
            ratio = compressed_size / original_size
            self.compression_ratios.append(ratio)

            if self.compression_histogram:
                self.compression_histogram.record(ratio)

    def record_decompression(self) -> None:
        """Record decompression metrics."""
        self.decompressions += 1

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get database storage statistics."""
        if not self.connection_pool:
            return {}

        try:
            async with self.connection_pool.acquire() as conn:
                # Get cache table statistics
                query = """
                SELECT 
                    COUNT(*) as total_entries,
                    SUM(size_bytes) as total_size_bytes,
                    SUM(CASE WHEN compressed THEN 1 ELSE 0 END) as compressed_entries,
                    AVG(size_bytes) as avg_size_bytes,
                    MAX(created_at) as newest_entry,
                    MIN(created_at) as oldest_entry
                FROM cache_entries
                """

                row = await conn.fetchrow(query)
                if row:
                    return {
                        "total_entries": row["total_entries"] or 0,
                        "total_size_bytes": row["total_size_bytes"] or 0,
                        "compressed_entries": row["compressed_entries"] or 0,
                        "avg_size_bytes": float(row["avg_size_bytes"] or 0),
                        "newest_entry": row["newest_entry"].isoformat()
                        if row["newest_entry"]
                        else None,
                        "oldest_entry": row["oldest_entry"].isoformat()
                        if row["oldest_entry"]
                        else None,
                    }
        except Exception as e:
            logger.warning(f"Failed to get database storage stats: {e}")

        return {}

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive database cache statistics."""
        total_operations = self.hits + self.misses
        hit_rate = self.hits / total_operations if total_operations > 0 else 0

        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times
            else 0
        )
        avg_compression_ratio = (
            sum(self.compression_ratios) / len(self.compression_ratios)
            if self.compression_ratios
            else 1.0
        )

        return {
            "service": self.service_name,
            "operations": {
                "hits": self.hits,
                "misses": self.misses,
                "sets": self.sets,
                "deletes": self.deletes,
                "errors": self.errors,
                "total": total_operations,
            },
            "performance": {
                "hit_rate": hit_rate,
                "avg_response_time_ms": avg_response_time,
                "p95_response_time_ms": self._percentile(95)
                if self.response_times
                else 0,
                "p99_response_time_ms": self._percentile(99)
                if self.response_times
                else 0,
            },
            "compression": {
                "compressions": self.compressions,
                "decompressions": self.decompressions,
                "avg_compression_ratio": avg_compression_ratio,
                "compressed_entries": self.compressed_entries,
            },
            "storage": {
                "total_entries": self.total_entries,
                "total_size_bytes": self.total_size_bytes,
            },
        }

    def _percentile(self, p: int) -> float:
        """Calculate percentile of response times."""
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        index = int((p / 100.0) * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)]


class DatabaseCache:
    """High-performance persistent database cache for L3 caching.

    Enhanced database cache with:
    - Transaction-safe operations with automatic rollback
    - JSONB storage for structured data (PostgreSQL)
    - Compression for large cache values
    - Query result caching with TTL
    - Connection pooling and health monitoring
    - Comprehensive metrics and observability
    - Security context validation for multi-tenant isolation
    """

    def __init__(
        self,
        config: DatabaseCacheConfig,
        enable_metrics: bool = True,
        service_name: str = "database_cache",
    ):
        if not ASYNCPG_AVAILABLE:
            raise ImportError(
                "asyncpg is required for DatabaseCache. Install with: pip install asyncpg"
            )

        self.config = config
        self.service_name = service_name

        # Database connection pool (will be initialized in async context)
        self.connection_pool: Optional[asyncpg.Pool] = None

        # Metrics
        self.metrics = DatabaseMetrics(service_name) if enable_metrics else None

        # Security validation
        self._security_validation_enabled = True

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._vacuum_task: Optional[asyncio.Task] = None

        logger.info(
            f"DatabaseCache initialized: {config.host}:{config.port}/{config.database}, "
            f"storage={config.storage_type.value}, compression={config.compression_type.value}"
        )

    async def initialize(self) -> None:
        """Initialize database connection pool and create cache table."""
        try:
            # Create connection pool
            self.connection_pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.query_timeout,
                server_settings={
                    "application_name": f"database_cache_{self.service_name}"
                },
            )

            # Update metrics pool reference
            if self.metrics:
                self.metrics.connection_pool = self.connection_pool

            # Create cache table if it doesn't exist
            await self._create_cache_table()

            # Start background maintenance tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_entries())
            self._vacuum_task = asyncio.create_task(self._periodic_vacuum())

            logger.info(
                f"Database cache connection established to {self.config.host}:{self.config.port}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize database cache connection: {e}")
            raise

    async def _create_cache_table(self) -> None:
        """Create cache table with appropriate indexes."""
        if not self.connection_pool:
            raise RuntimeError("Database connection pool not initialized")

        # Choose column type based on storage configuration
        value_column_type = {
            DatabaseCacheStorageType.TEXT: "TEXT",
            DatabaseCacheStorageType.JSONB: "JSONB",
            DatabaseCacheStorageType.BYTEA: "BYTEA",
        }[self.config.storage_type]

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.config.schema_name}.{self.config.cache_table_name} (
            key VARCHAR(512) PRIMARY KEY,
            stored_value {value_column_type} NOT NULL,
            storage_type VARCHAR(20) NOT NULL DEFAULT '{self.config.storage_type.value}',
            compressed BOOLEAN NOT NULL DEFAULT FALSE,
            compression_type VARCHAR(20) NOT NULL DEFAULT '{self.config.compression_type.value}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMPTZ,
            last_accessed TIMESTAMPTZ,
            access_count INTEGER NOT NULL DEFAULT 0,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            checksum VARCHAR(64),
            security_context_id VARCHAR(128),
            tags TEXT[]
        );
        """

        # Indexes for performance
        create_indexes_sql = [
            f"CREATE INDEX IF NOT EXISTS idx_{self.config.cache_table_name}_expires_at ON {self.config.schema_name}.{self.config.cache_table_name} (expires_at) WHERE expires_at IS NOT NULL;",
            f"CREATE INDEX IF NOT EXISTS idx_{self.config.cache_table_name}_created_at ON {self.config.schema_name}.{self.config.cache_table_name} (created_at);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.config.cache_table_name}_security_context ON {self.config.schema_name}.{self.config.cache_table_name} (security_context_id) WHERE security_context_id IS NOT NULL;",
            f"CREATE INDEX IF NOT EXISTS idx_{self.config.cache_table_name}_tags ON {self.config.schema_name}.{self.config.cache_table_name} USING GIN (tags) WHERE tags IS NOT NULL;",
        ]

        async with self.connection_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(create_table_sql)
                for index_sql in create_indexes_sql:
                    await conn.execute(index_sql)

        logger.debug(f"Cache table {self.config.cache_table_name} created/verified")

    def _validate_security_context(self, security_context: SecurityContext) -> bool:
        """Validate security context for database operations."""
        if not self._security_validation_enabled or not security_context:
            return True

        return (
            security_context.user_id is not None
            and len(security_context.user_id) > 0
            and security_context.permissions is not None
        )

    def _compress_value(self, value: bytes) -> tuple[bytes, bool]:
        """Compress value if it exceeds threshold."""
        if len(value) < self.config.compression_threshold:
            return value, False

        try:
            if self.config.compression_type == CompressionType.ZLIB:
                compressed = zlib.compress(value)
            elif self.config.compression_type == CompressionType.GZIP:
                import gzip

                compressed = gzip.compress(value)
            else:
                return value, False

            # Only use compressed version if it's actually smaller
            if len(compressed) < len(value):
                if self.metrics:
                    self.metrics.record_compression(len(value), len(compressed))
                return compressed, True
            else:
                return value, False

        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return value, False

    def _decompress_value(
        self, value: bytes, compression_type: CompressionType
    ) -> bytes:
        """Decompress value."""
        try:
            if compression_type == CompressionType.ZLIB:
                decompressed = zlib.decompress(value)
            elif compression_type == CompressionType.GZIP:
                import gzip

                decompressed = gzip.decompress(value)
            else:
                decompressed = value

            if self.metrics:
                self.metrics.record_decompression()

            return decompressed

        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return value

    def _serialize_value(
        self, value: Any
    ) -> tuple[Union[str, bytes], DatabaseCacheStorageType]:
        """Serialize value for database storage."""
        if self.config.storage_type == DatabaseCacheStorageType.JSONB:
            try:
                if isinstance(value, (dict, list)):
                    return json.dumps(value), DatabaseCacheStorageType.JSONB
                else:
                    # Wrap non-JSON types in a container
                    return json.dumps({
                        "__value__": value,
                        "__type__": type(value).__name__,
                    }), DatabaseCacheStorageType.JSONB
            except (TypeError, ValueError):
                # Fall back to string representation
                return str(value), DatabaseCacheStorageType.TEXT

        elif self.config.storage_type == DatabaseCacheStorageType.BYTEA:
            import pickle

            try:
                return pickle.dumps(value), DatabaseCacheStorageType.BYTEA
            except Exception:
                return str(value).encode("utf-8"), DatabaseCacheStorageType.BYTEA

        else:  # TEXT
            return str(value), DatabaseCacheStorageType.TEXT

    def _deserialize_value(
        self, stored_value: Union[str, bytes], storage_type: DatabaseCacheStorageType
    ) -> Any:
        """Deserialize value from database storage."""
        try:
            if storage_type == DatabaseCacheStorageType.JSONB:
                if isinstance(stored_value, str):
                    data = json.loads(stored_value)
                else:
                    data = stored_value  # Already parsed by asyncpg

                # Check if it's a wrapped non-JSON type
                if (
                    isinstance(data, dict)
                    and "__value__" in data
                    and "__type__" in data
                ):
                    return data["__value__"]
                return data

            elif storage_type == DatabaseCacheStorageType.BYTEA:
                import pickle

                if isinstance(stored_value, str):
                    stored_value = stored_value.encode("utf-8")
                return pickle.loads(stored_value)

            else:  # TEXT
                return stored_value

        except Exception as e:
            logger.warning(f"Deserialization failed: {e}")
            return stored_value

    async def get(
        self, key: str, security_context: Optional[SecurityContext] = None
    ) -> Any:
        """Get value from database cache with security validation."""
        if not self.connection_pool:
            await self.initialize()

        start_time = time.time()

        try:
            # Security validation
            if security_context and not self._validate_security_context(
                security_context
            ):
                logger.warning(f"Invalid security context for database get key: {key}")
                if self.metrics:
                    self.metrics.record_operation(
                        "get",
                        "error",
                        0,
                        security_context.user_id if security_context else None,
                    )
                return None

            query = f"""
            UPDATE {self.config.schema_name}.{self.config.cache_table_name}
            SET last_accessed = NOW(), access_count = access_count + 1
            WHERE key = $1 AND (expires_at IS NULL OR expires_at > NOW())
            RETURNING stored_value, storage_type, compressed, compression_type
            """

            async with self.connection_pool.acquire() as conn:
                async with conn.transaction():
                    row = await conn.fetchrow(query, key)

                    if row:
                        stored_value = row["stored_value"]
                        storage_type = DatabaseCacheStorageType(row["storage_type"])
                        compressed = row["compressed"]
                        compression_type = CompressionType(row["compression_type"])

                        # Handle decompression if needed
                        if compressed and isinstance(stored_value, (bytes, memoryview)):
                            stored_value = self._decompress_value(
                                bytes(stored_value), compression_type
                            )

                        # Deserialize value
                        value = self._deserialize_value(stored_value, storage_type)

                        duration_ms = (time.time() - start_time) * 1000
                        if self.metrics:
                            self.metrics.record_operation(
                                "get",
                                "hit",
                                duration_ms,
                                security_context.user_id if security_context else None,
                            )

                        return value
                    else:
                        duration_ms = (time.time() - start_time) * 1000
                        if self.metrics:
                            self.metrics.record_operation(
                                "get",
                                "miss",
                                duration_ms,
                                security_context.user_id if security_context else None,
                            )
                        return None

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Database cache get failed for key {key}: {e}")
            if self.metrics:
                self.metrics.record_operation(
                    "get",
                    "error",
                    duration_ms,
                    security_context.user_id if security_context else None,
                )
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        security_context: Optional[SecurityContext] = None,
        tags: Optional[Set[str]] = None,
    ) -> bool:
        """Set value in database cache with security validation."""
        if not self.connection_pool:
            await self.initialize()

        start_time = time.time()

        try:
            # Security validation
            if security_context and not self._validate_security_context(
                security_context
            ):
                logger.warning(f"Invalid security context for database set key: {key}")
                if self.metrics:
                    self.metrics.record_operation(
                        "set",
                        "error",
                        0,
                        security_context.user_id if security_context else None,
                    )
                return False

            # Serialize value
            serialized_value, storage_type = self._serialize_value(value)

            # Convert to bytes for compression if needed
            if isinstance(serialized_value, str):
                value_bytes = serialized_value.encode("utf-8")
            else:
                value_bytes = serialized_value

            # Compress if threshold exceeded
            compressed_value, is_compressed = self._compress_value(value_bytes)

            # Calculate expiration
            expires_at = None
            if ttl_seconds:
                expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)

            # Prepare final stored value based on storage type
            if storage_type == DatabaseCacheStorageType.BYTEA:
                final_stored_value = compressed_value
            else:
                # For JSONB and TEXT, convert back to string if it was compressed as bytes
                if is_compressed:
                    final_stored_value = compressed_value
                    storage_type = (
                        DatabaseCacheStorageType.BYTEA
                    )  # Force BYTEA for compressed text
                else:
                    final_stored_value = serialized_value

            query = f"""
            INSERT INTO {self.config.schema_name}.{self.config.cache_table_name} 
            (key, stored_value, storage_type, compressed, compression_type, expires_at, 
             size_bytes, security_context_id, tags)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (key) DO UPDATE SET
                stored_value = EXCLUDED.stored_value,
                storage_type = EXCLUDED.storage_type,
                compressed = EXCLUDED.compressed,
                compression_type = EXCLUDED.compression_type,
                expires_at = EXCLUDED.expires_at,
                size_bytes = EXCLUDED.size_bytes,
                security_context_id = EXCLUDED.security_context_id,
                tags = EXCLUDED.tags,
                last_accessed = NOW()
            """

            async with self.connection_pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        query,
                        key,
                        final_stored_value,
                        storage_type.value,
                        is_compressed,
                        self.config.compression_type.value,
                        expires_at,
                        len(compressed_value),
                        security_context.user_id if security_context else None,
                        list(tags) if tags else None,
                    )

            duration_ms = (time.time() - start_time) * 1000
            if self.metrics:
                self.metrics.record_operation(
                    "set",
                    "success",
                    duration_ms,
                    security_context.user_id if security_context else None,
                    len(compressed_value),
                )

            return True

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Database cache set failed for key {key}: {e}")
            if self.metrics:
                self.metrics.record_operation(
                    "set",
                    "error",
                    duration_ms,
                    security_context.user_id if security_context else None,
                )
            return False

    async def delete(
        self, key: str, security_context: Optional[SecurityContext] = None
    ) -> bool:
        """Delete key from database cache with security validation."""
        if not self.connection_pool:
            await self.initialize()

        start_time = time.time()

        try:
            # Security validation
            if security_context and not self._validate_security_context(
                security_context
            ):
                logger.warning(
                    f"Invalid security context for database delete key: {key}"
                )
                if self.metrics:
                    self.metrics.record_operation(
                        "delete",
                        "error",
                        0,
                        security_context.user_id if security_context else None,
                    )
                return False

            query = f"DELETE FROM {self.config.schema_name}.{self.config.cache_table_name} WHERE key = $1"

            async with self.connection_pool.acquire() as conn:
                result = await conn.execute(query, key)
                # Extract number of deleted rows from result
                deleted_count = int(result.split()[-1])  # "DELETE N" -> N

            success = deleted_count > 0
            duration_ms = (time.time() - start_time) * 1000

            if self.metrics:
                status = "success" if success else "miss"
                self.metrics.record_operation(
                    "delete",
                    status,
                    duration_ms,
                    security_context.user_id if security_context else None,
                )

            return success

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Database cache delete failed for key {key}: {e}")
            if self.metrics:
                self.metrics.record_operation(
                    "delete",
                    "error",
                    duration_ms,
                    security_context.user_id if security_context else None,
                )
            return False

    async def exists(
        self, key: str, security_context: Optional[SecurityContext] = None
    ) -> bool:
        """Check if key exists in database cache."""
        if not self.connection_pool:
            await self.initialize()

        try:
            # Security validation
            if security_context and not self._validate_security_context(
                security_context
            ):
                logger.warning(
                    f"Invalid security context for database exists key: {key}"
                )
                return False

            query = f"""
            SELECT 1 FROM {self.config.schema_name}.{self.config.cache_table_name} 
            WHERE key = $1 AND (expires_at IS NULL OR expires_at > NOW())
            """

            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow(query, key)
                return row is not None

        except Exception as e:
            logger.error(f"Database cache exists failed for key {key}: {e}")
            return False

    async def get_by_tags(
        self, tags: Set[str], security_context: Optional[SecurityContext] = None
    ) -> Dict[str, Any]:
        """Get all entries matching the specified tags."""
        if not self.connection_pool:
            await self.initialize()

        try:
            # Security validation
            if security_context and not self._validate_security_context(
                security_context
            ):
                logger.warning("Invalid security context for database get_by_tags")
                return {}

            query = f"""
            SELECT key, stored_value, storage_type, compressed, compression_type
            FROM {self.config.schema_name}.{self.config.cache_table_name}
            WHERE tags && $1 AND (expires_at IS NULL OR expires_at > NOW())
            """

            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch(query, list(tags))

                results = {}
                for row in rows:
                    key = row["key"]
                    stored_value = row["stored_value"]
                    storage_type = DatabaseCacheStorageType(row["storage_type"])
                    compressed = row["compressed"]
                    compression_type = CompressionType(row["compression_type"])

                    # Handle decompression if needed
                    if compressed and isinstance(stored_value, (bytes, memoryview)):
                        stored_value = self._decompress_value(
                            bytes(stored_value), compression_type
                        )

                    # Deserialize value
                    value = self._deserialize_value(stored_value, storage_type)
                    results[key] = value

                return results

        except Exception as e:
            logger.error(f"Database cache get_by_tags failed: {e}")
            return {}

    async def delete_by_tags(
        self, tags: Set[str], security_context: Optional[SecurityContext] = None
    ) -> int:
        """Delete all entries matching the specified tags."""
        if not self.connection_pool:
            await self.initialize()

        try:
            # Security validation
            if security_context and not self._validate_security_context(
                security_context
            ):
                logger.warning("Invalid security context for database delete_by_tags")
                return 0

            query = f"DELETE FROM {self.config.schema_name}.{self.config.cache_table_name} WHERE tags && $1"

            async with self.connection_pool.acquire() as conn:
                result = await conn.execute(query, list(tags))
                deleted_count = int(result.split()[-1])  # "DELETE N" -> N

                return deleted_count

        except Exception as e:
            logger.error(f"Database cache delete_by_tags failed: {e}")
            return 0

    async def clear_database(
        self, security_context: Optional[SecurityContext] = None
    ) -> bool:
        """Clear all keys from database cache."""
        if not self.connection_pool:
            await self.initialize()

        try:
            # Security validation
            if security_context and not self._validate_security_context(
                security_context
            ):
                logger.warning("Invalid security context for database clear")
                return False

            query = (
                f"DELETE FROM {self.config.schema_name}.{self.config.cache_table_name}"
            )

            async with self.connection_pool.acquire() as conn:
                await conn.execute(query)
                return True

        except Exception as e:
            logger.error(f"Database cache clear failed: {e}")
            return False

    async def _cleanup_expired_entries(self) -> None:
        """Background task to clean up expired entries."""
        while True:
            try:
                if self.connection_pool:
                    query = f"DELETE FROM {self.config.schema_name}.{self.config.cache_table_name} WHERE expires_at < NOW()"

                    async with self.connection_pool.acquire() as conn:
                        result = await conn.execute(query)
                        deleted_count = int(result.split()[-1]) if result else 0

                        if deleted_count > 0:
                            logger.debug(
                                f"Cleaned up {deleted_count} expired cache entries"
                            )

                # Wait for next cleanup cycle
                await asyncio.sleep(self.config.cleanup_expired_interval_hours * 3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during expired entries cleanup: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def _periodic_vacuum(self) -> None:
        """Background task to vacuum the cache table periodically."""
        while True:
            try:
                if self.connection_pool:
                    query = f"VACUUM ANALYZE {self.config.schema_name}.{self.config.cache_table_name}"

                    async with self.connection_pool.acquire() as conn:
                        await conn.execute(query)
                        logger.debug("Vacuumed database cache table")

                # Wait for next vacuum cycle
                await asyncio.sleep(self.config.vacuum_interval_hours * 3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during vacuum: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry

    async def ping(self) -> bool:
        """Ping database for health check."""
        if not self.connection_pool:
            try:
                await self.initialize()
            except Exception:
                return False

        try:
            async with self.connection_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                return True
        except Exception as e:
            logger.warning(f"Database cache ping failed: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive database cache statistics."""
        base_stats = {
            "connection": {
                "host": self.config.host,
                "port": self.config.port,
                "database": self.config.database,
                "connected": self.connection_pool is not None,
                "storage_type": self.config.storage_type.value,
                "compression_type": self.config.compression_type.value,
            }
        }

        if self.metrics:
            metrics_stats = self.metrics.get_stats()
            base_stats.update(metrics_stats)

            # Add storage statistics
            storage_stats = await self.metrics.get_storage_stats()
            base_stats["storage"].update(storage_stats)

        return base_stats

    async def shutdown(self) -> None:
        """Shutdown database cache and cleanup resources."""
        try:
            # Cancel background tasks
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            if self._vacuum_task:
                self._vacuum_task.cancel()
                try:
                    await self._vacuum_task
                except asyncio.CancelledError:
                    pass

            # Close connection pool
            if self.connection_pool:
                await self.connection_pool.close()

            logger.info("Database cache shutdown complete")

        except Exception as e:
            logger.warning(f"Error during database cache shutdown: {e}")

    def __repr__(self) -> str:
        return (
            f"DatabaseCache({self.config.host}:{self.config.port}/{self.config.database}, "
            f"storage={self.config.storage_type.value}, "
            f"compression={self.config.compression_type.value})"
        )


# Convenience function for easy configuration
def create_database_cache(
    host: str = "localhost",
    port: int = 5432,
    database: str = "prompt_improver",
    username: str = "postgres",
    password: Optional[str] = None,
    **kwargs,
) -> DatabaseCache:
    """Create database cache with simple configuration."""
    config = DatabaseCacheConfig(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        **kwargs,
    )
    return DatabaseCache(config)
