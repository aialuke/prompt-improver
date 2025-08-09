"""Unified Database Connection Manager - Modern Async Connection Management

A clean, modern async-only database connection manager that consolidates
functionality from multiple legacy managers into a single, efficient interface.

Key Features:
- Async-only operations following 2025 best practices
- High availability with automatic failover
- Mode-based access control and optimization
- Advanced connection pooling with intelligent configuration
- Circuit breaker patterns for resilience
- Comprehensive health monitoring and metrics
- Multi-database support (PostgreSQL + Redis)
- Clean, protocol-based interface design
"""
import asyncio
import contextlib
import json
import logging
import os
import statistics
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from typing import TypedDict, Literal

import asyncpg
import coredis
from coredis.sentinel import Sentinel
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool, QueuePool
from prompt_improver.core.protocols.cache_protocol import AdvancedCacheProtocol, BasicCacheProtocol, CacheHealthProtocol, CacheLockProtocol, CacheSubscriptionProtocol, MultiLevelCacheProtocol, RedisCacheProtocol
from prompt_improver.database.registry import RegistryManager, get_registry_manager

class TaskPriority:
    """Task priority levels for cache warming (fallback implementation)."""
    CRITICAL = 'critical'
    HIGH = 'high'
    NORMAL = 'normal'
    LOW = 'low'
    BACKGROUND = 'background'
try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
    cache_tracer = trace.get_tracer(__name__ + '.cache')
    cache_meter = metrics.get_meter(__name__ + '.cache')
    cache_operations_counter = cache_meter.create_counter('unified_cache_operations_total', description='Total unified cache operations by type, level, and status', unit='1')
    cache_hit_ratio_gauge = cache_meter.create_gauge('unified_cache_hit_ratio', description='Unified cache hit ratio by level', unit='ratio')
    cache_latency_histogram = cache_meter.create_histogram('unified_cache_operation_duration_seconds', description='Unified cache operation duration by type and level', unit='s')
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    cache_tracer = None
    cache_meter = None
    cache_operations_counter = None
    cache_hit_ratio_gauge = None
    cache_latency_histogram = None
logger = logging.getLogger(__name__)

@dataclass
class SecurityThreatScore:
    """Security threat assessment for operations."""
    level: str = 'low'
    score: float = 0.0
    factors: list[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

@dataclass
class SecurityValidationResult:
    """Comprehensive security validation results."""
    validated: bool = False
    validation_method: str = 'none'
    validation_timestamp: float = field(default_factory=time.time)
    validation_duration_ms: float = 0.0
    security_incidents: list[str] = field(default_factory=list)
    rate_limit_status: str = 'unknown'
    encryption_required: bool = False
    audit_trail_id: str | None = None

@dataclass
class SecurityPerformanceMetrics:
    """Security operation performance tracking."""
    authentication_time_ms: float = 0.0
    authorization_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    total_security_overhead_ms: float = 0.0
    operations_count: int = 0
    last_performance_check: float = field(default_factory=time.time)

@dataclass
class SecurityContext:
    """Enhanced security context for unified database and application operations.

    Provides comprehensive security information across all layers:
    - Authentication and authorization details
    - Security validation results and threat assessment
    - Performance metrics and audit trail information
    - Unified security policies and enforcement data

    Designed for zero-friction integration between database and security layers.
    """
    agent_id: str
    tier: str = 'basic'
    authenticated: bool = False
    created_at: float = field(default_factory=time.time)
    authentication_method: str = 'none'
    authentication_timestamp: float = field(default_factory=time.time)
    session_id: str | None = None
    permissions: list[str] = field(default_factory=list)
    validation_result: SecurityValidationResult = field(default_factory=SecurityValidationResult)
    threat_score: SecurityThreatScore = field(default_factory=SecurityThreatScore)
    performance_metrics: SecurityPerformanceMetrics = field(default_factory=SecurityPerformanceMetrics)
    audit_metadata: dict[str, Any] = field(default_factory=dict)
    compliance_tags: list[str] = field(default_factory=list)
    security_level: str = 'basic'
    zero_trust_validated: bool = False
    encryption_context: dict[str, str] | None = None
    expires_at: float | None = None
    max_operations: int | None = None
    operations_count: int = 0
    last_used: float = field(default_factory=time.time)

    def is_valid(self) -> bool:
        """Check if security context is still valid."""
        current_time = time.time()
        if self.expires_at and current_time > self.expires_at:
            return False
        if self.max_operations and self.operations_count >= self.max_operations:
            return False
        if not self.authenticated:
            return False
        return True

    def touch(self) -> None:
        """Update last used timestamp and increment operation count."""
        self.last_used = time.time()
        self.operations_count += 1

    def add_audit_event(self, event_type: str, details: dict[str, Any]) -> None:
        """Add audit event to security context."""
        if 'audit_events' not in self.audit_metadata:
            self.audit_metadata['audit_events'] = []
        self.audit_metadata['audit_events'].append({'timestamp': time.time(), 'event_type': event_type, 'details': details})
        if len(self.audit_metadata['audit_events']) > 50:
            self.audit_metadata['audit_events'] = self.audit_metadata['audit_events'][-50:]

    def update_threat_score(self, new_level: str, new_score: float, factors: list[str]) -> None:
        """Update threat assessment with new information."""
        self.threat_score.level = new_level
        self.threat_score.score = new_score
        self.threat_score.factors = factors
        self.threat_score.last_updated = time.time()

    def record_performance_metric(self, operation: str, duration_ms: float) -> None:
        """Record security operation performance metric."""
        if operation == 'authentication':
            self.performance_metrics.authentication_time_ms = duration_ms
        elif operation == 'authorization':
            self.performance_metrics.authorization_time_ms = duration_ms
        elif operation == 'validation':
            self.performance_metrics.validation_time_ms = duration_ms
        self.performance_metrics.total_security_overhead_ms = self.performance_metrics.authentication_time_ms + self.performance_metrics.authorization_time_ms + self.performance_metrics.validation_time_ms
        self.performance_metrics.operations_count += 1
        self.performance_metrics.last_performance_check = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert security context to dictionary for serialization."""
        return {'agent_id': self.agent_id, 'tier': self.tier, 'authenticated': self.authenticated, 'created_at': self.created_at, 'authentication_method': self.authentication_method, 'authentication_timestamp': self.authentication_timestamp, 'session_id': self.session_id, 'permissions': self.permissions, 'validation_result': {'validated': self.validation_result.validated, 'validation_method': self.validation_result.validation_method, 'validation_timestamp': self.validation_result.validation_timestamp, 'validation_duration_ms': self.validation_result.validation_duration_ms, 'security_incidents': self.validation_result.security_incidents, 'rate_limit_status': self.validation_result.rate_limit_status, 'encryption_required': self.validation_result.encryption_required, 'audit_trail_id': self.validation_result.audit_trail_id}, 'threat_score': {'level': self.threat_score.level, 'score': self.threat_score.score, 'factors': self.threat_score.factors, 'last_updated': self.threat_score.last_updated}, 'performance_metrics': {'authentication_time_ms': self.performance_metrics.authentication_time_ms, 'authorization_time_ms': self.performance_metrics.authorization_time_ms, 'validation_time_ms': self.performance_metrics.validation_time_ms, 'total_security_overhead_ms': self.performance_metrics.total_security_overhead_ms, 'operations_count': self.performance_metrics.operations_count, 'last_performance_check': self.performance_metrics.last_performance_check}, 'audit_metadata': self.audit_metadata, 'compliance_tags': self.compliance_tags, 'security_level': self.security_level, 'zero_trust_validated': self.zero_trust_validated, 'encryption_context': self.encryption_context, 'expires_at': self.expires_at, 'max_operations': self.max_operations, 'operations_count': self.operations_count, 'last_used': self.last_used, 'is_valid': self.is_valid()}

class RedisSecurityError(Exception):
    """Security-related Redis operation error."""

@dataclass
class CacheEntry:
    """Cache entry with metadata for L1 memory cache."""
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: int | None = None

    def is_expired(self) -> bool:
        """Check if the cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.now(UTC) > self.created_at + timedelta(seconds=self.ttl_seconds)

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = datetime.now(UTC)
        self.access_count += 1

@dataclass
class AccessPattern:
    """Access pattern tracking for intelligent cache warming."""
    key: str
    access_count: int = 0
    last_access: datetime = None
    access_frequency: float = 0.0
    access_times: list[datetime] = None
    warming_priority: float = 0.0

    def __post_init__(self):
        if self.access_times is None:
            self.access_times = []
        if self.last_access is None:
            self.last_access = datetime.now(UTC)

    def record_access(self) -> None:
        """Record a new access and update frequency metrics."""
        now = datetime.now(UTC)
        self.access_count += 1
        self.last_access = now
        self.access_times.append(now)
        cutoff = now - timedelta(hours=24)
        self.access_times = [t for t in self.access_times if t > cutoff]
        if len(self.access_times) >= 2:
            time_span = (self.access_times[-1] - self.access_times[0]).total_seconds() / 3600
            self.access_frequency = len(self.access_times) / max(time_span, 0.1)
        recency_weight = max(0, 1 - (now - self.last_access).total_seconds() / 3600)
        self.warming_priority = self.access_frequency * (1 + recency_weight)

class LRUCache:
    """High-performance in-memory LRU cache for L1 caching."""

    def __init__(self, max_size: int=1000):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if key in self._cache:
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry.value
        self._misses += 1
        return None

    def set(self, key: str, value: Any, ttl_seconds: int | None=None) -> None:
        """Set value in cache."""
        if key in self._cache:
            entry = self._cache[key]
            entry.value = value
            entry.created_at = datetime.now(UTC)
            entry.ttl_seconds = ttl_seconds
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            entry = CacheEntry(value=value, created_at=datetime.now(UTC), last_accessed=datetime.now(UTC), ttl_seconds=ttl_seconds)
            self._cache[key] = entry

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        return {'size': len(self._cache), 'max_size': self._max_size, 'hits': self._hits, 'misses': self._misses, 'hit_rate': hit_rate, 'utilization': len(self._cache) / self._max_size}

class ConnectionMode(Enum):
    """Connection operation modes"""
    READ_ONLY = 'read_only'
    READ_WRITE = 'read_write'
    BATCH = 'batch'
    TRANSACTIONAL = 'transactional'

class ManagerMode(Enum):
    """Manager operation modes optimized for different use cases"""
    MCP_SERVER = 'mcp_server'
    ML_TRAINING = 'ml_training'
    ADMIN = 'admin'
    ASYNC_MODERN = 'async_modern'
    HIGH_AVAILABILITY = 'ha'

class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = 'healthy'
    DEGRADED = 'degraded'
    UNHEALTHY = 'unhealthy'
    UNKNOWN = 'unknown'

class PoolState(Enum):
    """Connection pool operational states (from AdaptiveConnectionPool)"""
    INITIALIZING = 'initializing'
    HEALTHY = 'healthy'
    SCALING_UP = 'scaling_up'
    SCALING_DOWN = 'scaling_down'
    DEGRADED = 'degraded'
    FAILED = 'failed'
    STRESSED = 'stressed'
    EXHAUSTED = 'exhausted'
    RECOVERING = 'recovering'

@dataclass
class ConnectionInfo:
    """Connection information for age tracking (from ConnectionPoolOptimizer)"""
    connection_id: str
    created_at: datetime
    last_used: datetime
    use_count: int = 0

    @property
    def age_seconds(self) -> float:
        """Connection age in seconds"""
        return (datetime.now(UTC) - self.created_at).total_seconds()

@dataclass
class ConnectionMetrics:
    """Comprehensive connection metrics from all managers"""
    active_connections: int = 0
    idle_connections: int = 0
    total_connections: int = 0
    pool_utilization: float = 0.0
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0
    failed_connections: int = 0
    last_failover: float | None = None
    failover_count: int = 0
    health_check_failures: int = 0
    circuit_breaker_state: str = 'closed'
    circuit_breaker_failures: int = 0
    mode_specific_metrics: dict[str, Any] = field(default_factory=dict)
    sla_compliance_rate: float = 100.0
    registry_conflicts: int = 0
    registered_models: int = 0
    connections_created: int = 0
    connections_closed: int = 0
    connection_failures: int = 0
    queries_executed: int = 0
    queries_failed: int = 0
    wait_time_ms: float = 0.0
    last_scale_event: datetime | None = None
    connection_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    query_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    connection_reuse_count: int = 0
    pool_efficiency: float = 0.0
    connections_saved: int = 0
    database_load_reduction_percent: float = 0.0
    http_pool_health: bool = True
    redis_pool_health: bool = True
    multi_pool_coordination_active: bool = False
    cache_l1_hits: int = 0
    cache_l2_hits: int = 0
    cache_l3_hits: int = 0
    cache_total_requests: int = 0
    cache_hit_rate: float = 0.0
    cache_l1_size: int = 0
    cache_l1_utilization: float = 0.0
    cache_warming_enabled: bool = False
    cache_warming_cycles: int = 0
    cache_warming_keys_warmed: int = 0
    cache_warming_hit_rate: float = 0.0
    cache_response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    cache_operation_stats: dict[str, Any] = field(default_factory=dict)
    cache_health_status: str = 'healthy'

@dataclass
class PoolConfiguration:
    """Intelligent pool configuration based on usage patterns"""
    mode: ManagerMode
    pg_pool_size: int
    pg_max_overflow: int
    pg_timeout: float
    redis_pool_size: int
    enable_ha: bool = False
    enable_circuit_breaker: bool = False
    enable_l1_cache: bool = True
    l1_cache_size: int = 1000
    l2_cache_ttl: int = 3600
    enable_cache_warming: bool = True
    warming_threshold: float = 2.0
    warming_interval: int = 300
    max_warming_keys: int = 50

    @classmethod
    def for_mode(cls, mode: ManagerMode) -> 'PoolConfiguration':
        """Create pool configuration optimized for specific mode"""
        configs = {ManagerMode.MCP_SERVER: cls(mode=mode, pg_pool_size=20, pg_max_overflow=10, pg_timeout=0.2, redis_pool_size=10, enable_circuit_breaker=True, enable_l1_cache=True, l1_cache_size=2000, l2_cache_ttl=900, enable_cache_warming=True, warming_threshold=3.0, warming_interval=120), ManagerMode.ML_TRAINING: cls(mode=mode, pg_pool_size=15, pg_max_overflow=10, pg_timeout=5.0, redis_pool_size=8, enable_ha=True, enable_l1_cache=True, l1_cache_size=1500, l2_cache_ttl=1800, enable_cache_warming=True, warming_threshold=1.5, warming_interval=300), ManagerMode.ADMIN: cls(mode=mode, pg_pool_size=5, pg_max_overflow=2, pg_timeout=10.0, redis_pool_size=3, enable_ha=True, enable_l1_cache=True, l1_cache_size=500, l2_cache_ttl=7200, enable_cache_warming=False), ManagerMode.ASYNC_MODERN: cls(mode=mode, pg_pool_size=12, pg_max_overflow=8, pg_timeout=5.0, redis_pool_size=6, enable_circuit_breaker=True, enable_l1_cache=True, l1_cache_size=1000, l2_cache_ttl=3600, enable_cache_warming=True, warming_threshold=2.0, warming_interval=300), ManagerMode.HIGH_AVAILABILITY: cls(mode=mode, pg_pool_size=20, pg_max_overflow=20, pg_timeout=10.0, redis_pool_size=10, enable_ha=True, enable_circuit_breaker=True, enable_l1_cache=True, l1_cache_size=2500, l2_cache_ttl=1800, enable_cache_warming=True, warming_threshold=1.0, warming_interval=180)}
        return configs.get(mode, configs[ManagerMode.ASYNC_MODERN])

class UnifiedConnectionManager(RedisCacheProtocol, MultiLevelCacheProtocol):
    """Modern async database connection manager.

    Provides a clean, efficient interface for async database operations with
    built-in high availability, health monitoring, and intelligent pooling.
    Follows 2025 best practices with async-only operations.
    """

    def __init__(self, mode: ManagerMode=ManagerMode.ASYNC_MODERN, db_config=None, redis_config=None):
        """Initialize unified connection manager

        Args:
            mode: Manager operation mode
            db_config: Database configuration (auto-detected if None)
            redis_config: Redis configuration (auto-detected if None)
        """
        self.mode = mode
        if db_config is None:
            self.db_config = None
        else:
            self.db_config = db_config
        self.redis_config = redis_config or self._get_redis_config()
        self.pool_config = PoolConfiguration.for_mode(mode)
        self._registry_manager = None
        self._metrics = ConnectionMetrics()
        self.min_pool_size = self.pool_config.pg_pool_size
        self.max_pool_size = min(self.pool_config.pg_pool_size * 5, 100)
        self.current_pool_size = self.pool_config.pg_pool_size
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.scale_cooldown_seconds = 60
        self.last_scale_time = 0
        self._pool_state = PoolState.INITIALIZING
        self.performance_window = deque(maxlen=100)
        self.last_metrics_update = time.time()
        self._connection_registry: dict[str, ConnectionInfo] = {}
        self._connection_id_counter = 0
        self._total_connections_created = 0
        self._async_engine: AsyncEngine | None = None
        self._async_session_factory: async_sessionmaker | None = None
        self._pg_pools: dict[str, asyncpg.Pool] = {}
        self._redis_sentinel: Sentinel | None = None
        self._redis_master: coredis.Redis | None = None
        self._redis_replica: coredis.Redis | None = None
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_timeout = 30
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = 0
        self._health_status = HealthStatus.UNKNOWN
        self._last_health_check = 0
        self._health_check_interval = 10
        self._is_initialized = False
        self._initialization_lock = asyncio.Lock()
        self._health_monitor_task_id: str | None = None
        self._monitoring_task_id: str | None = None
        self._l1_cache: LRUCache | None = None
        self._access_patterns: dict[str, AccessPattern] = {}
        self._cache_warming_task_id: str | None = None
        self._last_warming_time = datetime.now(UTC)
        self._cache_l1_hits = 0
        self._cache_l2_hits = 0
        self._cache_l3_hits = 0
        self._cache_total_requests = 0
        self._cache_response_times = deque(maxlen=1000)
        self._cache_operation_stats = defaultdict(lambda: {'count': 0, 'total_time': 0.0, 'min_time': float('inf'), 'max_time': 0.0, 'error_count': 0})
        self._warming_stats = {'cycles_completed': 0, 'keys_warmed': 0, 'warming_hits': 0, 'warming_errors': 0}
        self._cache_health_status = {'overall_health': 'healthy', 'l1_health': 'healthy', 'l2_health': 'healthy', 'warming_health': 'healthy', 'last_health_check': datetime.now(UTC)}
        self._protocol_security_required = mode in [ManagerMode.HIGH_AVAILABILITY, ManagerMode.MCP_SERVER]
        self._default_security_context = None
        self._active_locks: dict[str, dict[str, Any]] = {}
        self._lock_timeout_default = 30
        self._subscribers: dict[str, list[Any]] = {}
        self._pubsub_connection: Any | None = None
        logger.info('UnifiedConnectionManager initialized for mode: %s with auto-scaling %s-%s connections and enhanced cache (L1: %s)', mode.value, self.min_pool_size, self.max_pool_size, self.pool_config.l1_cache_size if self.pool_config.enable_l1_cache else 0)

    def _get_registry_manager(self):
        """Get registry manager with lazy loading."""
        if self._registry_manager is None:
            from prompt_improver.database.registry import get_registry_manager
            self._registry_manager = get_registry_manager()
        return self._registry_manager

    async def _get_background_task_manager(self):
        """Get background task manager with graceful fallback."""
        try:
            return
        except ImportError as e:
            logger.warning('Background task manager not available: %s', e)
            return
        except Exception as e:
            logger.warning('Failed to get background task manager: %s', e)
            return

    def _get_redis_config(self):
        """Get Redis configuration from AppConfig"""
        try:
            return None
        except Exception:

            class MinimalRedisConfig:
                host = 'localhost'
                port = 6379
                cache_db = 0
                password = None
                socket_timeout = 5.0
                connect_timeout = 5.0
            return MinimalRedisConfig()

    async def initialize(self) -> bool:
        """Initialize all connection components"""
        async with self._initialization_lock:
            if self._is_initialized:
                return True
            try:
                await self._setup_database_connections()
                if self.pool_config.enable_ha:
                    await self._setup_ha_components()
                await self._setup_redis_connections()
                if self.pool_config.enable_l1_cache:
                    self._l1_cache = LRUCache(max_size=self.pool_config.l1_cache_size)
                    logger.info('L1 cache initialized with size %s', self.pool_config.l1_cache_size)
                self._default_security_context = SecurityContext(agent_id='unified_connection_manager_system', tier='system', authenticated=True, created_at=time.time())
                task_manager = await self._get_background_task_manager()
                health_task_id = await task_manager.submit_enhanced_task(task_id=f'db_health_monitor_{id(self)}', coroutine=self._health_monitor_loop, priority=TaskPriority.HIGH, tags={'service': 'database', 'type': 'health_monitoring', 'component': 'unified_connection_manager'})
                monitoring_task_id = await task_manager.submit_enhanced_task(task_id=f'db_monitoring_{id(self)}', coroutine=self._monitoring_loop, priority=TaskPriority.HIGH, tags={'service': 'database', 'type': 'monitoring_loop', 'component': 'unified_connection_manager'})
                if self.pool_config.enable_cache_warming and self._l1_cache and task_manager:
                    try:
                        warming_task_id = await task_manager.submit_enhanced_task(task_id=f'cache_warming_{id(self)}', coroutine=self._cache_warming_loop, priority=TaskPriority.BACKGROUND, tags={'service': 'database', 'type': 'cache_warming', 'component': 'unified_connection_manager'})
                        self._cache_warming_task_id = warming_task_id
                    except Exception as e:
                        logger.warning('Failed to start cache warming task: %s', e)
                self._health_monitor_task_id = health_task_id
                self._monitoring_task_id = monitoring_task_id
                self._is_initialized = True
                self._health_status = HealthStatus.HEALTHY
                logger.info('UnifiedConnectionManager initialized successfully for %s', self.mode.value)
                return True
            except Exception as e:
                logger.error('Failed to initialize UnifiedConnectionManager: %s', e)
                self._health_status = HealthStatus.UNHEALTHY
                return False

    async def shutdown(self) -> bool:
        """Shutdown all connection components and cleanup resources"""
        try:
            logger.info('Shutting down UnifiedConnectionManager for %s', self.mode.value)
            if hasattr(self, '_health_monitor_task_id') and self._health_monitor_task_id:
                pass
            if hasattr(self, '_monitoring_task_id') and self._monitoring_task_id:
                pass
            if self._async_engine:
                await self._async_engine.dispose()
                self._async_engine = None
                logger.info('Async engine disposed')
            if hasattr(self, '_pg_pools') and self._pg_pools:
                for pool_name, pool in self._pg_pools.items():
                    await pool.close()
                    logger.info('HA pool %s closed', pool_name)
                self._pg_pools.clear()
            if hasattr(self, '_redis_client') and self._redis_client:
                await self._redis_client.close()
                self._redis_client = None
                logger.info('Redis client closed')
            self._is_initialized = False
            self._health_status = HealthStatus.UNHEALTHY
            logger.info('UnifiedConnectionManager shutdown completed for %s', self.mode.value)
            return True
        except Exception as e:
            logger.error('Error during shutdown: %s', e)
            return False

    async def _setup_database_connections(self):
        """Setup both sync and async database connections"""
        base_url = f'postgresql://{self.db_config.username}:{self.db_config.password}@{self.db_config.host}:{self.db_config.port}/{self.db_config.database}'
        sync_url = f'postgresql+asyncpg://{self.db_config.username}:{self.db_config.password}@{self.db_config.host}:{self.db_config.port}/{self.db_config.database}'
        async_url = f'postgresql+asyncpg://{self.db_config.username}:{self.db_config.password}@{self.db_config.host}:{self.db_config.port}/{self.db_config.database}'
        poolclass = None
        engine_kwargs = {'pool_size': self.pool_config.pg_pool_size, 'max_overflow': self.pool_config.pg_max_overflow, 'pool_timeout': self.pool_config.pg_timeout, 'pool_pre_ping': True, 'pool_recycle': 3600, 'echo': self.db_config.echo_sql, 'future': True, 'connect_args': {'server_settings': {'application_name': f'apes_unified_{self.mode.value}', 'timezone': 'UTC'}, 'command_timeout': self.pool_config.pg_timeout, 'connect_timeout': 10}}
        self._async_engine = create_async_engine(async_url, **engine_kwargs)
        self._async_session_factory = async_sessionmaker(bind=self._async_engine, class_=AsyncSession, expire_on_commit=False, autoflush=True, autocommit=False)
        self._setup_connection_monitoring()
        await self._test_connections()

    async def _setup_ha_components(self):
        """Setup high availability components (from HAConnectionManager)"""
        if not self.pool_config.enable_ha:
            return
        try:
            primary_dsn = f'postgresql://{self.db_config.username}:{self.db_config.password}@{self.db_config.host}:{self.db_config.port}/{self.db_config.database}'
            primary_pool = await asyncpg.create_pool(dsn=primary_dsn, min_size=2, max_size=self.pool_config.pg_pool_size, command_timeout=self.pool_config.pg_timeout, max_inactive_connection_lifetime=3600, server_settings={'application_name': f'apes_ha_primary_{self.mode.value}', 'timezone': 'UTC'})
            self._pg_pools['primary'] = primary_pool
            replica_hosts = self._get_replica_hosts()
            for i, (host, port) in enumerate(replica_hosts):
                replica_dsn = f'postgresql://{self.db_config.username}:{self.db_config.password}@{host}:{port}/{self.db_config.database}'
                replica_pool = await asyncpg.create_pool(dsn=replica_dsn, min_size=1, max_size=self.pool_config.pg_pool_size // 2, command_timeout=self.pool_config.pg_timeout, server_settings={'application_name': f'apes_ha_replica_{i}_{self.mode.value}', 'timezone': 'UTC'})
                self._pg_pools[f'replica_{i}'] = replica_pool
            logger.info('HA pools initialized: %s pools', len(self._pg_pools))
        except Exception as e:
            logger.warning('HA setup failed, continuing without HA: %s', e)

    def _get_replica_hosts(self) -> list:
        """Get replica host configurations"""
        replicas = os.getenv('POSTGRES_REPLICAS', '').split(',')
        replica_hosts = []
        for replica in replicas:
            if ':' in replica:
                host, port = replica.split(':')
                replica_hosts.append((host.strip(), int(port)))
        return replica_hosts

    async def _setup_redis_connections(self):
        """Setup Redis connections (from HAConnectionManager)"""
        try:
            if self.pool_config.enable_ha:
                await self._setup_redis_sentinel()
            else:
                await self._setup_redis_direct()
        except Exception as e:
            logger.warning('Redis setup failed: %s', e)

    async def _setup_redis_sentinel(self):
        """Setup Redis Sentinel for HA"""
        sentinel_hosts_env = os.getenv('REDIS_SENTINELS', 'redis-sentinel-1:26379,redis-sentinel-2:26379,redis-sentinel-3:26379')
        sentinel_hosts = []
        for host_port in sentinel_hosts_env.split(','):
            if ':' in host_port:
                host, port = host_port.strip().split(':')
                sentinel_hosts.append((host, int(port)))
        if sentinel_hosts:
            self._redis_sentinel = Sentinel(sentinels=sentinel_hosts, stream_timeout=0.1, connect_timeout=0.1)
            self._redis_master = self._redis_sentinel.primary_for('mymaster', stream_timeout=0.1, password=getattr(self.redis_config, 'password', None))
            self._redis_replica = self._redis_sentinel.replica_for('mymaster', stream_timeout=0.1, password=getattr(self.redis_config, 'password', None))
            logger.info('Redis Sentinel initialized')

    async def _setup_redis_direct(self):
        """Setup direct Redis connection"""
        self._redis_master = coredis.Redis(host=self.redis_config.host, port=self.redis_config.port, db=self.redis_config.cache_db, password=getattr(self.redis_config, 'password', None), stream_timeout=self.redis_config.socket_timeout, connect_timeout=self.redis_config.connect_timeout)

    def _setup_connection_monitoring(self):
        """Setup connection monitoring events"""
        if not self._async_engine:
            return

        @event.listens_for(self._async_engine.sync_engine, 'connect')
        def on_async_connect(dbapi_connection, connection_record):
            self._metrics.total_connections += 1
            logger.debug('Async connection created for %s', self.mode.value)

        @event.listens_for(self._async_engine.sync_engine, 'checkout')
        def on_async_checkout(dbapi_connection, connection_record, connection_proxy):
            self._metrics.active_connections += 1
            self._update_pool_utilization()

        @event.listens_for(self._async_engine.sync_engine, 'checkin')
        def on_async_checkin(dbapi_connection, connection_record):
            self._metrics.active_connections = max(0, self._metrics.active_connections - 1)
            self._update_pool_utilization()

    # Structured options for low-level connection behavior
    class ConnectionOptions(TypedDict, total=False):
        connection_type: Literal['session', 'raw']
        pool_name: str


    def _update_pool_utilization(self):
        """Update pool utilization metrics"""
        total_pool_size = self.pool_config.pg_pool_size + self.pool_config.pg_max_overflow
        if total_pool_size > 0:
            self._metrics.pool_utilization = self._metrics.active_connections / total_pool_size * 100

    async def _test_connections(self):
        """Test all connection types"""
        async with self.get_async_session() as session:
            result = await session.execute(text('SELECT 1'))
            assert result.scalar() == 1
        if self._pg_pools:
            primary_pool = self._pg_pools.get('primary')
            if primary_pool:
                async with primary_pool.acquire() as conn:
                    result = await conn.fetchval('SELECT 1')
                    assert result == 1
        logger.info('All connection tests passed')

    async def get_connection(self, mode: ConnectionMode=ConnectionMode.READ_WRITE, **kwargs: ConnectionOptions) -> AsyncIterator[AsyncSession | AsyncConnection]:
        """Get connection implementing ConnectionManagerProtocol"""
        if not self._is_initialized:
            await self.initialize()
        if self._is_circuit_breaker_open():
            raise ConnectionError('Circuit breaker is open')
        connection_type = kwargs.get('connection_type', 'session')
        try:
            if connection_type == 'raw' and self._pg_pools:
                pool_name = 'primary' if mode == ConnectionMode.READ_WRITE else 'replica_0'
                pool = self._pg_pools.get(pool_name) or self._pg_pools.get('primary')
                if pool:
                    async with pool.acquire() as conn:
                        yield conn
                        return
            async with self.get_async_session() as session:
                if mode == ConnectionMode.READ_ONLY:
                    await session.execute(text('SET TRANSACTION READ ONLY'))
                yield session
        except Exception as e:
            self._handle_connection_failure(e)
            raise

    async def health_check(self) -> dict[str, Any]:
        """Comprehensive health check"""
        start_time = time.time()
        health_info = {'status': 'unknown', 'timestamp': start_time, 'mode': self.mode.value, 'components': {}, 'metrics': self._get_metrics_dict(), 'response_time_ms': 0}
        try:
            async with self.get_async_session() as session:
                await session.execute(text('SELECT 1'))
            health_info['components']['async_database'] = 'healthy'
            if self._pg_pools:
                for pool_name, pool in self._pg_pools.items():
                    try:
                        async with pool.acquire() as conn:
                            await conn.execute('SELECT 1')
                        health_info['components'][f'ha_pool_{pool_name}'] = 'healthy'
                    except Exception as e:
                        health_info['components'][f'ha_pool_{pool_name}'] = f'unhealthy: {e}'
            if self._redis_master:
                try:
                    await self._redis_master.ping()
                    health_info['components']['redis_master'] = 'healthy'
                except Exception as e:
                    health_info['components']['redis_master'] = f'unhealthy: {e}'
            unhealthy_components = [k for k, v in health_info['components'].items() if 'unhealthy' in str(v)]
            if not unhealthy_components:
                health_info['status'] = 'healthy'
                self._health_status = HealthStatus.HEALTHY
            elif len(unhealthy_components) < len(health_info['components']) / 2:
                health_info['status'] = 'degraded'
                self._health_status = HealthStatus.DEGRADED
            else:
                health_info['status'] = 'unhealthy'
                self._health_status = HealthStatus.UNHEALTHY
        except Exception as e:
            health_info['status'] = 'unhealthy'
            health_info['error'] = str(e)
            self._health_status = HealthStatus.UNHEALTHY
        health_info['response_time_ms'] = (time.time() - start_time) * 1000
        return health_info

    async def get_health_status(self) -> dict[str, Any]:
        """Get current health status without performing checks"""
        return {'status': self._health_status.value if hasattr(self, '_health_status') else 'unknown', 'mode': self.mode.value, 'pool_info': {'pool_size': self.pool_config.pg_pool_size, 'max_pool_size': self.pool_config.pg_pool_size + self.pool_config.pg_max_overflow, 'timeout': self.pool_config.pg_timeout, 'active_connections': getattr(self._metrics, 'active_connections', 0), 'idle_connections': getattr(self._metrics, 'idle_connections', 0)}, 'metrics': self._get_metrics_dict(), 'initialized': getattr(self, '_is_initialized', False)}

    async def __aenter__(self):
        """Async context manager entry"""
        if not getattr(self, '_is_initialized', False):
            await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""

    async def close(self) -> None:
        """Close all connections and cleanup resources"""
        logger.info('Shutting down UnifiedConnectionManager')
        try:
            try:
                task_manager = await self._get_background_task_manager()
                if task_manager:
                    if self._health_monitor_task_id:
                        await task_manager.cancel_task(self._health_monitor_task_id)
                        self._health_monitor_task_id = None
                    if self._monitoring_task_id:
                        await task_manager.cancel_task(self._monitoring_task_id)
                        self._monitoring_task_id = None
                    if self._cache_warming_task_id:
                        await task_manager.cancel_task(self._cache_warming_task_id)
                        self._cache_warming_task_id = None
            except Exception as e:
                logger.warning('Failed to cancel background tasks: %s', e)
            if self._async_engine:
                await self._async_engine.dispose()
            for pool_name, pool in self._pg_pools.items():
                try:
                    await pool.close()
                except Exception as e:
                    logger.warning('Error closing HA pool {pool_name}: %s', e)
            if self._pubsub_connection:
                try:
                    await self._pubsub_connection.aclose()
                    self._pubsub_connection = None
                except Exception as e:
                    logger.warning('Error closing pubsub connection: %s', e)
            if self._redis_master:
                await self._redis_master.aclose()
            if self._redis_replica:
                await self._redis_replica.aclose()
            self._active_locks.clear()
            self._subscribers.clear()
            if self._l1_cache:
                self._l1_cache.clear()
            self._access_patterns.clear()
            self._cache_response_times.clear()
            self._cache_operation_stats.clear()
            self._is_initialized = False
            logger.info('UnifiedConnectionManager shutdown complete')
        except Exception as e:
            logger.error('Error during shutdown: %s', e)

    async def get_connection_info(self) -> dict[str, Any]:
        """Get current connection pool information"""
        info = {'mode': self.mode.value, 'initialized': self._is_initialized, 'health_status': self._health_status.value, 'pool_config': {'pg_pool_size': self.pool_config.pg_pool_size, 'pg_max_overflow': self.pool_config.pg_max_overflow, 'pg_timeout': self.pool_config.pg_timeout, 'redis_pool_size': self.pool_config.redis_pool_size, 'enable_ha': self.pool_config.enable_ha, 'enable_circuit_breaker': self.pool_config.enable_circuit_breaker}, 'metrics': self._get_metrics_dict()}
        if self._async_engine:
            pool = self._async_engine.pool
            info['async_pool'] = {'size': pool.size(), 'checked_out': pool.checkedout(), 'checked_in': pool.checkedin(), 'overflow': pool.overflow(), 'invalid': pool.invalid()}
        if self._pg_pools:
            info['ha_pools'] = dict.fromkeys(self._pg_pools.keys(), 'active')
        return info

    def is_healthy(self) -> bool:
        """Quick health status check"""
        return self._health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    @contextlib.asynccontextmanager
    async def get_mcp_read_session(self) -> AsyncIterator[AsyncSession]:
        """Get MCP-optimized read-only session with <200ms SLA enforcement (from MCPConnectionPool)"""
        if not self._is_initialized:
            await self.initialize()
        if not self._async_session_factory:
            raise RuntimeError('Async session factory not initialized')
        session = self._async_session_factory()
        start_time = time.time()
        try:
            await session.execute(text('SET TRANSACTION READ ONLY'))
            if self.mode == ManagerMode.MCP_SERVER:
                await session.execute(text("SET statement_timeout = '150ms'"))
            yield session
            response_time = (time.time() - start_time) * 1000
            if self.mode == ManagerMode.MCP_SERVER and response_time > 200:
                logger.warning('MCP read session exceeded 200ms SLA: %sms', format(response_time, '.1f'))
                self._metrics.sla_compliance_rate = max(0.0, self._metrics.sla_compliance_rate - 2.0)
        except Exception as e:
            logger.error('MCP read session error: %s', e)
            raise
        finally:
            await session.close()

    @contextlib.asynccontextmanager
    async def get_feedback_session(self) -> AsyncIterator[AsyncSession]:
        """Get session optimized for feedback data writes (from MCPConnectionPool)"""
        if not self._is_initialized:
            await self.initialize()
        session = self._async_session_factory()
        start_time = time.time()
        try:
            yield session
            await session.commit()
            response_time = (time.time() - start_time) * 1000
            self._update_response_time(response_time)
            self._metrics.queries_executed += 1
        except Exception as e:
            await session.rollback()
            self._metrics.queries_failed += 1
            logger.error('Feedback session error: %s', e)
            raise
        finally:
            await session.close()

    @contextlib.asynccontextmanager
    async def get_async_session(self) -> AsyncIterator[AsyncSession]:
        """Get async session"""
        if not self._is_initialized:
            await self.initialize()
        if not self._async_session_factory:
            raise RuntimeError('Async session factory not initialized')
        session = self._async_session_factory()
        start_time = time.time()
        try:
            yield session
            await session.commit()
            response_time = (time.time() - start_time) * 1000
            self._update_response_time(response_time)
            self._metrics.queries_executed += 1
            self._metrics.connection_reuse_count += 1
            self._metrics.connection_times.append(response_time)
            if len(self._metrics.connection_times) > 0:
                self._metrics.avg_response_time_ms = statistics.mean(list(self._metrics.connection_times)[-100:])
        except Exception as e:
            await session.rollback()
            self._metrics.error_rate += 1
            self._metrics.queries_failed += 1
            self._handle_connection_failure(e)
            logger.error('Async session error in {self.mode.value}: %s', e)
            raise
        finally:
            await session.close()

    async def optimize_pool_size(self) -> dict[str, Any]:
        """Dynamically optimize pool size based on load patterns (from ConnectionPoolOptimizer)"""
        current_metrics = await self._collect_pool_metrics()
        if datetime.now(UTC) - (self._metrics.last_scale_event or datetime.min.replace(tzinfo=UTC)) < timedelta(minutes=5):
            return {'status': 'skipped', 'reason': 'optimization cooldown'}
        utilization = current_metrics.get('utilization', 0) / 100.0
        waiting_requests = current_metrics.get('waiting_requests', 0)
        recommendations = []
        new_pool_size = self.current_pool_size
        if utilization > 0.9 and waiting_requests > 0:
            increase = min(5, self.max_pool_size - self.current_pool_size)
            if increase > 0:
                new_pool_size += increase
                recommendations.append(f'Increase pool size by {increase} (high utilization: {utilization:.1%})')
                self._pool_state = PoolState.STRESSED
        elif utilization < 0.3 and self.current_pool_size > self.min_pool_size:
            decrease = min(3, self.current_pool_size - self.min_pool_size)
            if decrease > 0:
                new_pool_size -= decrease
                recommendations.append(f'Decrease pool size by {decrease} (low utilization: {utilization:.1%})')
        if new_pool_size != self.current_pool_size:
            try:
                await self._scale_pool(new_pool_size)
                return {'status': 'optimized', 'previous_size': self.current_pool_size, 'new_size': new_pool_size, 'utilization': utilization, 'recommendations': recommendations}
            except Exception as e:
                logger.error('Failed to optimize pool size: %s', e)
                return {'status': 'error', 'error': str(e)}
        return {'status': 'no_change_needed', 'current_size': self.current_pool_size, 'utilization': utilization, 'state': self._pool_state.value}

    async def coordinate_pools(self) -> dict[str, Any]:
        """Multi-pool coordination (from ConnectionPoolManager)"""
        coordination_status = {'database_pool': {'healthy': self._health_status == HealthStatus.HEALTHY, 'connections': self._metrics.active_connections, 'utilization': self._metrics.pool_utilization}, 'redis_pool': {'healthy': self._metrics.redis_pool_health, 'connected': self._redis_master is not None}, 'http_pool': {'healthy': self._metrics.http_pool_health}}
        total_healthy_pools = sum((1 for pool in coordination_status.values() if pool['healthy']))
        self._metrics.multi_pool_coordination_active = total_healthy_pools > 1
        return {'status': 'active' if self._metrics.multi_pool_coordination_active else 'limited', 'healthy_pools': total_healthy_pools, 'pool_status': coordination_status, 'load_balancing_active': total_healthy_pools > 1}

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics from all consolidated managers"""
        pool_metrics = await self._collect_pool_metrics()
        return {'state': self._pool_state.value, 'pool_size': self.current_pool_size, 'min_pool_size': self.min_pool_size, 'max_pool_size': self.max_pool_size, 'active_connections': self._metrics.active_connections, 'idle_connections': self._metrics.idle_connections, 'total_connections': self._metrics.total_connections, 'pool_utilization': self._metrics.pool_utilization, 'connections_created': self._metrics.connections_created, 'connections_closed': self._metrics.connections_closed, 'connection_failures': self._metrics.connection_failures, 'avg_response_time_ms': self._metrics.avg_response_time_ms, 'queries_executed': self._metrics.queries_executed, 'queries_failed': self._metrics.queries_failed, 'wait_time_ms': self._metrics.wait_time_ms, 'circuit_breaker_open': self._is_circuit_breaker_open(), 'circuit_breaker_failures': self._circuit_breaker_failures, 'last_scale_event': self._metrics.last_scale_event.isoformat() if self._metrics.last_scale_event else None, 'sla_compliance_rate': self._metrics.sla_compliance_rate, 'pool_efficiency': self._metrics.pool_efficiency, 'database_load_reduction_percent': self._metrics.database_load_reduction_percent, 'connections_saved': self._metrics.connections_saved, 'multi_pool_coordination': self._metrics.multi_pool_coordination_active, 'performance_window': list(self.performance_window), 'cache_stats': self.get_cache_stats()}

    async def test_permissions(self) -> dict[str, Any]:
        """Test database permissions (from MCPConnectionPool)"""
        results = {'read_rule_performance': False, 'read_rule_metadata': False, 'write_prompt_sessions': False, 'denied_rule_write': True}
        try:
            async with self.get_mcp_read_session() as session:
                try:
                    await session.execute(text('SELECT COUNT(*) FROM rule_performance LIMIT 1'))
                    results['read_rule_performance'] = True
                except Exception as e:
                    logger.warning('Cannot read rule_performance: %s', e)
                try:
                    await session.execute(text('SELECT COUNT(*) FROM rule_metadata LIMIT 1'))
                    results['read_rule_metadata'] = True
                except Exception as e:
                    logger.warning('Cannot read rule_metadata: %s', e)
            async with self.get_feedback_session() as session:
                try:
                    await session.execute(text("INSERT INTO prompt_improvement_sessions (original_prompt, enhanced_prompt, applied_rules, response_time_ms) VALUES ('test', 'test', '[]', 100)"))
                    await session.execute(text("DELETE FROM prompt_improvement_sessions WHERE original_prompt = 'test'"))
                    results['write_prompt_sessions'] = True
                except Exception as e:
                    logger.warning('Cannot write to prompt_improvement_sessions: %s', e)
                try:
                    await session.execute(text("INSERT INTO rule_performance (rule_id, rule_name) VALUES ('test', 'test')"))
                    results['denied_rule_write'] = False
                    logger.warning('User can write to rule tables - SECURITY ISSUE!')
                except Exception:
                    pass
        except Exception as e:
            logger.error('Permission test failed: %s', e)
            return {'error': str(e), 'permissions_verified': False}
        return {'permissions_verified': True, 'test_results': results, 'security_compliant': results['read_rule_performance'] and results['read_rule_metadata'] and results['write_prompt_sessions'] and results['denied_rule_write']}

    def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics from multi-level cache system."""
        total_requests = self._cache_total_requests
        overall_hit_rate = 0.0
        if total_requests > 0:
            total_hits = self._cache_l1_hits + self._cache_l2_hits + self._cache_l3_hits
            overall_hit_rate = total_hits / total_requests
        l1_stats = {}
        if self._l1_cache:
            l1_stats = self._l1_cache.get_stats()
        avg_response_time = 0.0
        if self._cache_response_times:
            avg_response_time = sum(self._cache_response_times) / len(self._cache_response_times)
        operation_summaries = {}
        for op_name, stats in self._cache_operation_stats.items():
            if stats['count'] > 0:
                operation_summaries[op_name] = {'count': stats['count'], 'avg_time_ms': stats['total_time'] / stats['count'], 'min_time_ms': stats['min_time'] if stats['min_time'] != float('inf') else 0, 'max_time_ms': stats['max_time'], 'error_rate': stats['error_count'] / stats['count'] if stats['count'] > 0 else 0}
        return {'overall_hit_rate': overall_hit_rate, 'total_requests': total_requests, 'l1_cache': {'hits': self._cache_l1_hits, **l1_stats}, 'l2_cache': {'hits': self._cache_l2_hits, 'enabled': self._redis_master is not None}, 'l3_cache': {'hits': self._cache_l3_hits}, 'performance': {'avg_response_time_ms': avg_response_time, 'total_response_samples': len(self._cache_response_times)}, 'operations': operation_summaries, 'warming': {'enabled': self.pool_config.enable_cache_warming, 'stats': self._warming_stats, 'active_patterns': len(self._access_patterns), 'health': self._cache_health_status}, 'health_status': self._cache_health_status.get('overall_health', 'unknown')}

    async def _collect_pool_metrics(self) -> dict[str, Any]:
        """Collect current pool metrics"""
        if not self._async_engine:
            return {}
        pool = self._async_engine.pool
        return {'pool_size': pool.size(), 'available': pool.checkedin(), 'active': pool.checkedout(), 'utilization': pool.checkedout() / pool.size() * 100 if pool.size() > 0 else 0, 'waiting_requests': 0, 'overflow': pool.overflow(), 'invalid': pool.invalid()}

    async def get_ml_telemetry_metrics(self) -> dict[str, Any]:
        """Get connection pool metrics formatted for ML orchestration telemetry integration.

        This method provides unified pool monitoring for ML orchestration, eliminating
        the need for independent pool registration and monitoring patterns.

        Returns:
            Dict containing ML telemetry-compatible pool metrics
        """
        base_metrics = await self._collect_pool_metrics()
        avg_connection_time = 0.0
        avg_query_time = 0.0
        if self._metrics.connection_times:
            avg_connection_time = sum(self._metrics.connection_times) / len(self._metrics.connection_times)
        if self._metrics.query_times:
            avg_query_time = sum(self._metrics.query_times) / len(self._metrics.query_times)
        return {'pool_utilization': base_metrics.get('utilization', 0) / 100.0, 'avg_connection_time_ms': avg_connection_time, 'avg_query_time_ms': avg_query_time, 'pool_size': base_metrics.get('pool_size', 0), 'active_connections': base_metrics.get('active', 0), 'available_connections': base_metrics.get('available', 0), 'overflow_connections': base_metrics.get('overflow', 0), 'invalid_connections': base_metrics.get('invalid', 0), 'health_status': 'healthy' if self._health_status == HealthStatus.HEALTHY else 'degraded'}

    def _update_response_time(self, response_time_ms: float):
        """Update average response time using exponential moving average"""
        alpha = 0.1
        if self._metrics.avg_response_time_ms == 0:
            self._metrics.avg_response_time_ms = response_time_ms
        else:
            self._metrics.avg_response_time_ms = alpha * response_time_ms + (1 - alpha) * self._metrics.avg_response_time_ms

    def _update_connection_metrics(self, success: bool):
        """Update connection success/failure metrics"""
        if success:
            self._circuit_breaker_failures = 0
        else:
            self._circuit_breaker_failures += 1
            self._circuit_breaker_last_failure = time.time()
            self._metrics.failed_connections += 1

    def _handle_connection_failure(self, error: Exception):
        """Handle connection failure and update circuit breaker"""
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = time.time()
        if self.pool_config.enable_circuit_breaker and self._circuit_breaker_failures >= self._circuit_breaker_threshold:
            self._metrics.circuit_breaker_state = 'open'
            logger.error('Circuit breaker opened due to %s failures', self._circuit_breaker_failures)

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if not self.pool_config.enable_circuit_breaker:
            return False
        if self._metrics.circuit_breaker_state == 'open':
            if time.time() - self._circuit_breaker_last_failure > self._circuit_breaker_timeout:
                self._metrics.circuit_breaker_state = 'half-open'
                logger.info('Circuit breaker moved to half-open state')
                return False
            return True
        return False

    def _get_metrics_dict(self) -> dict[str, Any]:
        """Get metrics as dictionary with all consolidated metrics"""
        return {'active_connections': self._metrics.active_connections, 'idle_connections': self._metrics.idle_connections, 'total_connections': self._metrics.total_connections, 'pool_utilization': self._metrics.pool_utilization, 'avg_response_time_ms': self._metrics.avg_response_time_ms, 'error_rate': self._metrics.error_rate, 'failed_connections': self._metrics.failed_connections, 'failover_count': self._metrics.failover_count, 'last_failover': self._metrics.last_failover, 'health_check_failures': self._metrics.health_check_failures, 'circuit_breaker_state': self._metrics.circuit_breaker_state, 'circuit_breaker_failures': self._circuit_breaker_failures, 'connections_created': self._metrics.connections_created, 'connections_closed': self._metrics.connections_closed, 'connection_failures': self._metrics.connection_failures, 'queries_executed': self._metrics.queries_executed, 'queries_failed': self._metrics.queries_failed, 'wait_time_ms': self._metrics.wait_time_ms, 'last_scale_event': self._metrics.last_scale_event.isoformat() if self._metrics.last_scale_event else None, 'connection_reuse_count': self._metrics.connection_reuse_count, 'pool_efficiency': self._metrics.pool_efficiency, 'connections_saved': self._metrics.connections_saved, 'database_load_reduction_percent': self._metrics.database_load_reduction_percent, 'http_pool_health': self._metrics.http_pool_health, 'redis_pool_health': self._metrics.redis_pool_health, 'multi_pool_coordination_active': self._metrics.multi_pool_coordination_active, 'sla_compliance_rate': self._metrics.sla_compliance_rate, 'registry_conflicts': self._metrics.registry_conflicts, 'registered_models': len(self._get_registry_manager().get_registered_classes()), 'pool_state': self._pool_state.value, 'current_pool_size': self.current_pool_size, 'min_pool_size': self.min_pool_size, 'max_pool_size': self.max_pool_size}

    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while self._is_initialized:
            try:
                await asyncio.sleep(self._health_check_interval)
                health_result = await self.health_check()
                response_time_ms = health_result.get('response_time_ms', 0)
                if self.mode == ManagerMode.MCP_SERVER:
                    if response_time_ms < 200:
                        self._metrics.sla_compliance_rate = min(100.0, self._metrics.sla_compliance_rate + 0.1)
                    else:
                        self._metrics.sla_compliance_rate = max(0.0, self._metrics.sla_compliance_rate - 2.0)
                elif response_time_ms < self.pool_config.pg_timeout * 1000:
                    self._metrics.sla_compliance_rate = min(100.0, self._metrics.sla_compliance_rate + 0.1)
                else:
                    self._metrics.sla_compliance_rate = max(0.0, self._metrics.sla_compliance_rate - 1.0)
                self._last_health_check = time.time()
            except Exception as e:
                logger.error('Health monitoring error: %s', e)
                self._metrics.health_check_failures += 1
                await asyncio.sleep(self._health_check_interval * 2)

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for adaptive scaling and performance tracking (from AdaptiveConnectionPool)"""
        while self._is_initialized:
            try:
                await asyncio.sleep(10)
                await self._update_metrics()
                await self._evaluate_scaling()
                await self._update_connection_efficiency()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error('Monitoring loop error: %s', e)

    async def _update_metrics(self) -> None:
        """Update real-time pool metrics (from AdaptiveConnectionPool)"""
        if not self._async_engine:
            return
        current_time = time.time()
        pool = self._async_engine.pool
        self._metrics.total_connections = pool.size()
        self._metrics.active_connections = pool.checkedout()
        self._metrics.idle_connections = pool.checkedin()
        if self._metrics.total_connections > 0:
            self._metrics.pool_utilization = self._metrics.active_connections / self._metrics.total_connections * 100
        self.performance_window.append({'timestamp': current_time, 'utilization': self._metrics.pool_utilization, 'active_connections': self._metrics.active_connections, 'total_connections': self._metrics.total_connections, 'avg_connection_time': self._metrics.avg_response_time_ms, 'sla_compliance': self._metrics.sla_compliance_rate})
        self.last_metrics_update = current_time

    async def _evaluate_scaling(self) -> None:
        """Evaluate if pool scaling is needed (from AdaptiveConnectionPool)"""
        if time.time() - self.last_scale_time < self.scale_cooldown_seconds:
            return
        utilization = self._metrics.pool_utilization / 100.0
        if utilization > self.scale_up_threshold and self.current_pool_size < self.max_pool_size:
            new_size = min(self.current_pool_size + 10, self.max_pool_size)
            await self._scale_pool(new_size)
            self._pool_state = PoolState.SCALING_UP
        elif utilization < self.scale_down_threshold and self.current_pool_size > self.min_pool_size and (self._metrics.avg_response_time_ms < 50):
            new_size = max(self.current_pool_size - 5, self.min_pool_size)
            await self._scale_pool(new_size)
            self._pool_state = PoolState.SCALING_DOWN
        else:
            self._pool_state = PoolState.HEALTHY

    async def _scale_pool(self, new_size: int) -> None:
        """Scale the connection pool to new size (from AdaptiveConnectionPool)"""
        if not self._async_engine:
            return
        old_size = self.current_pool_size
        try:
            logger.info('Pool scaling requested: {old_size} → %s connections', new_size)
            self.pool_config.pg_pool_size = new_size
            self.current_pool_size = new_size
            self.last_scale_time = time.time()
            self._metrics.last_scale_event = datetime.now(UTC)
            logger.info('Pool size updated: {old_size} → %s connections', new_size)
        except Exception as e:
            logger.error('Failed to scale connection pool: %s', e)
            self._pool_state = PoolState.DEGRADED

    async def _update_connection_efficiency(self) -> None:
        """Update connection efficiency metrics (from ConnectionPoolOptimizer)"""
        if self._total_connections_created > 0:
            reuse_rate = self._metrics.connection_reuse_count / self._total_connections_created
            self._metrics.pool_efficiency = reuse_rate * 100
            base_connections = self._metrics.connection_reuse_count + self._total_connections_created
            if base_connections > 0:
                self._metrics.database_load_reduction_percent = (base_connections - self._total_connections_created) / base_connections * 100
                self._metrics.connections_saved = self._metrics.connection_reuse_count

    async def get(self, key: str) -> Any | None:
        """Get value from cache (BasicCacheProtocol compliance).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        security_context = self._default_security_context if self._protocol_security_required else None
        return await self.get_cached(key, security_context)

    async def set(self, key: str, value: Any, ttl: int | None=None) -> bool:
        """Set value in cache (BasicCacheProtocol compliance).

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds

        Returns:
            True if successfully cached, False otherwise
        """
        security_context = self._default_security_context if self._protocol_security_required else None
        return await self.set_cached(key, value, ttl, security_context)

    async def delete(self, key: str) -> bool:
        """Delete key from cache (BasicCacheProtocol compliance).

        Args:
            key: Cache key to delete

        Returns:
            True if successfully deleted, False otherwise
        """
        security_context = self._default_security_context if self._protocol_security_required else None
        return await self.delete_cached(key, security_context)

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache (BasicCacheProtocol compliance).

        Args:
            key: Cache key to check

        Returns:
            True if key exists, False otherwise
        """
        security_context = self._default_security_context if self._protocol_security_required else None
        return await self.exists_cached(key, security_context)

    async def clear(self) -> bool:
        """Clear all cache entries (BasicCacheProtocol compliance).

        Returns:
            True if successfully cleared, False otherwise
        """
        success = True
        try:
            if self._l1_cache:
                self._l1_cache.clear()
                logger.info('L1 cache cleared')
            if self._redis_master:
                try:
                    if self._protocol_security_required and self._default_security_context:
                        if self._validate_security_context(self._default_security_context):
                            await self._redis_master.flushdb()
                            logger.info('L2 Redis cache cleared')
                        else:
                            logger.warning('Cannot clear Redis cache - invalid security context')
                            success = False
                    else:
                        await self._redis_master.flushdb()
                        logger.info('L2 Redis cache cleared')
                except Exception as e:
                    logger.warning('Failed to clear Redis cache: %s', e)
                    success = False
            self._access_patterns.clear()
            self._cache_response_times.clear()
            self._cache_operation_stats.clear()
            self._cache_l1_hits = 0
            self._cache_l2_hits = 0
            self._cache_l3_hits = 0
            self._cache_total_requests = 0
            return success
        except Exception as e:
            logger.error('Error clearing cache: %s', e)
            return False

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from cache (AdvancedCacheProtocol compliance).

        Args:
            keys: List of cache keys

        Returns:
            Dictionary mapping keys to values (missing keys omitted)
        """
        result = {}
        security_context = self._default_security_context if self._protocol_security_required else None
        for key in keys:
            value = await self.get_cached(key, security_context)
            if value is not None:
                result[key] = value
        return result

    async def set_many(self, mapping: dict[str, Any], ttl: int | None=None) -> bool:
        """Set multiple values in cache (AdvancedCacheProtocol compliance).

        Args:
            mapping: Dictionary of key-value pairs to cache
            ttl: Optional TTL in seconds for all keys

        Returns:
            True if all values were successfully cached, False otherwise
        """
        security_context = self._default_security_context if self._protocol_security_required else None
        success = True
        for key, value in mapping.items():
            if not await self.set_cached(key, value, ttl, security_context):
                success = False
        return success

    async def delete_many(self, keys: list[str]) -> int:
        """Delete multiple keys from cache (AdvancedCacheProtocol compliance).

        Args:
            keys: List of cache keys to delete

        Returns:
            Number of keys successfully deleted
        """
        security_context = self._default_security_context if self._protocol_security_required else None
        deleted_count = 0
        for key in keys:
            if await self.delete_cached(key, security_context):
                deleted_count += 1
        return deleted_count

    async def get_or_set(self, key: str, default_func, ttl: int | None=None) -> Any:
        """Get value or set it using default function (AdvancedCacheProtocol compliance).

        Args:
            key: Cache key
            default_func: Function to call if key doesn't exist (can be async)
            ttl: Optional TTL in seconds

        Returns:
            Cached or computed value
        """
        security_context = self._default_security_context if self._protocol_security_required else None
        value = await self.get_cached(key, security_context)
        if value is not None:
            return value
        if asyncio.iscoroutinefunction(default_func):
            computed_value = await default_func()
        else:
            computed_value = default_func()
        await self.set_cached(key, computed_value, ttl, security_context)
        return computed_value

    async def increment(self, key: str, delta: int=1) -> int:
        """Increment numeric value in cache (AdvancedCacheProtocol compliance).

        Args:
            key: Cache key
            delta: Amount to increment by

        Returns:
            New value after increment
        """
        if self._redis_master:
            try:
                if self._protocol_security_required and self._default_security_context:
                    if not self._validate_security_context(self._default_security_context):
                        raise RedisSecurityError(f'Invalid security context for increment on key {key}')
                new_value = await self._redis_master.incrby(key, delta)
                if self._l1_cache:
                    self._l1_cache.set(key, new_value)
                return new_value
            except Exception as e:
                logger.error('Redis increment failed for key {key}: %s', e)
                raise
        else:
            current_value = await self.get(key)
            if current_value is None:
                current_value = 0
            elif not isinstance(current_value, (int, float)):
                raise ValueError(f'Cannot increment non-numeric value for key {key}')
            new_value = int(current_value) + delta
            await self.set(key, new_value)
            return new_value

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for key (AdvancedCacheProtocol compliance).

        Args:
            key: Cache key
            seconds: Expiration time in seconds

        Returns:
            True if expiration was set, False otherwise
        """
        if self._redis_master:
            try:
                if self._protocol_security_required and self._default_security_context:
                    if not self._validate_security_context(self._default_security_context):
                        logger.warning('Invalid security context for expire on key %s', key)
                        return False
                result = await self._redis_master.expire(key, seconds)
                return bool(result)
            except Exception as e:
                logger.error('Redis expire failed for key {key}: %s', e)
                return False
        else:
            logger.warning('Cannot set expiration on existing key %s without Redis', key)
            return False

    async def ping(self) -> bool:
        """Ping cache service (CacheHealthProtocol compliance).

        Returns:
            True if cache is responsive, False otherwise
        """
        try:
            l1_healthy = True
            if self._l1_cache:
                test_key = '__ping_test__'
                self._l1_cache.set(test_key, 'ping')
                l1_healthy = self._l1_cache.get(test_key) == 'ping'
                self._l1_cache.delete(test_key)
            l2_healthy = True
            if self._redis_master:
                try:
                    if self._protocol_security_required and self._default_security_context:
                        if not self._validate_security_context(self._default_security_context):
                            logger.warning('Invalid security context for ping')
                            l2_healthy = False
                        else:
                            l2_healthy = await self._redis_master.ping()
                    else:
                        l2_healthy = await self._redis_master.ping()
                except Exception as e:
                    logger.warning('Redis ping failed: %s', e)
                    l2_healthy = False
            return l1_healthy and l2_healthy
        except Exception as e:
            logger.error('Cache ping failed: %s', e)
            return False

    async def get_info(self) -> dict[str, Any]:
        """Get cache service information (CacheHealthProtocol compliance).

        Returns:
            Dictionary containing cache service information
        """
        info = {'service': 'UnifiedConnectionManager', 'version': '2.0', 'mode': self.mode.value, 'protocol_compliance': True, 'security_required': self._protocol_security_required, 'components': {'l1_cache': {'enabled': self._l1_cache is not None, 'type': 'LRUCache', 'max_size': self.pool_config.l1_cache_size if self._l1_cache else 0}, 'l2_cache': {'enabled': self._redis_master is not None, 'type': 'Redis', 'connected': False}}, 'features': {'multi_level': True, 'cache_warming': self.pool_config.enable_cache_warming, 'security_context': True, 'distributed_locking': self._redis_master is not None, 'pub_sub': self._redis_master is not None, 'opentelemetry': OPENTELEMETRY_AVAILABLE}}
        if self._redis_master:
            try:
                ping_result = await self.ping()
                info['components']['l2_cache']['connected'] = ping_result
                if ping_result:
                    try:
                        redis_info = await self._redis_master.info()
                        if isinstance(redis_info, dict):
                            info['components']['l2_cache']['redis_version'] = redis_info.get('redis_version', 'unknown')
                            info['components']['l2_cache']['used_memory_human'] = redis_info.get('used_memory_human', 'unknown')
                    except Exception as e:
                        logger.debug('Could not get Redis info: %s', e)
            except Exception as e:
                logger.warning('Could not test Redis connectivity: %s', e)
        return info

    async def get_stats(self) -> dict[str, Any]:
        """Get cache performance statistics (CacheHealthProtocol compliance).

        Returns:
            Dictionary containing performance statistics
        """
        base_stats = self.get_cache_stats()
        protocol_stats = {'protocol_compliance': {'basic_cache': True, 'advanced_cache': True, 'health_monitoring': True, 'pub_sub': self._redis_master is not None, 'distributed_locking': self._redis_master is not None, 'multi_level': True}, 'security': {'security_required': self._protocol_security_required, 'active_locks': len(self._active_locks), 'default_context_valid': self._default_security_context and self._validate_security_context(self._default_security_context) if self._default_security_context else False}, 'connections': {'redis_master_connected': self._redis_master is not None, 'redis_replica_connected': self._redis_replica is not None, 'sentinel_connected': self._redis_sentinel is not None}}
        return {**base_stats, **protocol_stats}

    async def get_memory_usage(self) -> dict[str, Any]:
        """Get memory usage statistics (CacheHealthProtocol compliance).

        Returns:
            Dictionary containing memory usage information
        """
        memory_stats = {'l1_cache': {'enabled': self._l1_cache is not None, 'current_size': 0, 'max_size': 0, 'utilization_percent': 0.0, 'estimated_memory_bytes': 0}, 'l2_cache': {'enabled': self._redis_master is not None, 'memory_info': {}}, 'access_patterns': {'active_patterns': len(self._access_patterns), 'estimated_memory_bytes': len(self._access_patterns) * 1024}, 'response_times': {'samples_stored': len(self._cache_response_times), 'estimated_memory_bytes': len(self._cache_response_times) * 8}}
        if self._l1_cache:
            l1_stats = self._l1_cache.get_stats()
            memory_stats['l1_cache'].update({'current_size': l1_stats['size'], 'max_size': l1_stats['max_size'], 'utilization_percent': l1_stats['utilization'] * 100, 'estimated_memory_bytes': l1_stats['size'] * 512})
        if self._redis_master:
            try:
                if self._protocol_security_required and self._default_security_context:
                    if self._validate_security_context(self._default_security_context):
                        redis_info = await self._redis_master.info('memory')
                        if isinstance(redis_info, dict):
                            memory_stats['l2_cache']['memory_info'] = {'used_memory': redis_info.get('used_memory', 0), 'used_memory_human': redis_info.get('used_memory_human', 'unknown'), 'used_memory_peak': redis_info.get('used_memory_peak', 0), 'used_memory_peak_human': redis_info.get('used_memory_peak_human', 'unknown'), 'maxmemory': redis_info.get('maxmemory', 0), 'maxmemory_human': redis_info.get('maxmemory_human', 'unknown')}
                else:
                    redis_info = await self._redis_master.info('memory')
                    if isinstance(redis_info, dict):
                        memory_stats['l2_cache']['memory_info'] = {'used_memory': redis_info.get('used_memory', 0), 'used_memory_human': redis_info.get('used_memory_human', 'unknown'), 'used_memory_peak': redis_info.get('used_memory_peak', 0), 'used_memory_peak_human': redis_info.get('used_memory_peak_human', 'unknown')}
            except Exception as e:
                logger.warning('Could not get Redis memory info: %s', e)
                memory_stats['l2_cache']['memory_info'] = {'error': str(e)}
        return memory_stats

    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel (CacheSubscriptionProtocol compliance).

        Args:
            channel: Channel name
            message: Message to publish

        Returns:
            Number of subscribers that received the message
        """
        if not self._redis_master:
            logger.warning('Pub/Sub not available - Redis not connected')
            return 0
        try:
            if self._protocol_security_required and self._default_security_context:
                if not self._validate_security_context(self._default_security_context):
                    raise RedisSecurityError(f'Invalid security context for publish to channel {channel}')
            if not isinstance(message, (str, bytes)):
                message = json.dumps(message)
            subscriber_count = await self._redis_master.publish(channel, message)
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {'operation': 'publish', 'level': 'l2', 'status': 'success', 'security_context': self._default_security_context.agent_id if self._default_security_context else 'none'})
            logger.debug('Published message to channel %s, %s subscribers', channel, subscriber_count)
            return subscriber_count
        except Exception as e:
            logger.error('Failed to publish to channel {channel}: %s', e)
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {'operation': 'publish', 'level': 'l2', 'status': 'error', 'security_context': self._default_security_context.agent_id if self._default_security_context else 'none'})
            return 0

    async def subscribe(self, channels: list[str]) -> Any:
        """Subscribe to channels (CacheSubscriptionProtocol compliance).

        Args:
            channels: List of channel names to subscribe to

        Returns:
            Subscription object or None if failed
        """
        if not self._redis_master:
            logger.warning('Pub/Sub not available - Redis not connected')
            return None
        try:
            if self._protocol_security_required and self._default_security_context:
                if not self._validate_security_context(self._default_security_context):
                    raise RedisSecurityError(f'Invalid security context for subscribe to channels {channels}')
            if not self._pubsub_connection:
                self._pubsub_connection = self._redis_master.pubsub()
            await self._pubsub_connection.subscribe(*channels)
            for channel in channels:
                if channel not in self._subscribers:
                    self._subscribers[channel] = []
                self._subscribers[channel].append(self._pubsub_connection)
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {'operation': 'subscribe', 'level': 'l2', 'status': 'success', 'security_context': self._default_security_context.agent_id if self._default_security_context else 'none'})
            logger.debug('Subscribed to channels: %s', channels)
            return self._pubsub_connection
        except Exception as e:
            logger.error('Failed to subscribe to channels {channels}: %s', e)
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {'operation': 'subscribe', 'level': 'l2', 'status': 'error', 'security_context': self._default_security_context.agent_id if self._default_security_context else 'none'})
            return None

    async def unsubscribe(self, channels: list[str]) -> bool:
        """Unsubscribe from channels (CacheSubscriptionProtocol compliance).

        Args:
            channels: List of channel names to unsubscribe from

        Returns:
            True if successfully unsubscribed, False otherwise
        """
        if not self._redis_master or not self._pubsub_connection:
            logger.warning('Pub/Sub not available or not subscribed')
            return False
        try:
            if self._protocol_security_required and self._default_security_context:
                if not self._validate_security_context(self._default_security_context):
                    raise RedisSecurityError(f'Invalid security context for unsubscribe from channels {channels}')
            await self._pubsub_connection.unsubscribe(*channels)
            for channel in channels:
                if channel in self._subscribers:
                    if self._pubsub_connection in self._subscribers[channel]:
                        self._subscribers[channel].remove(self._pubsub_connection)
                    if not self._subscribers[channel]:
                        del self._subscribers[channel]
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {'operation': 'unsubscribe', 'level': 'l2', 'status': 'success', 'security_context': self._default_security_context.agent_id if self._default_security_context else 'none'})
            logger.debug('Unsubscribed from channels: %s', channels)
            return True
        except Exception as e:
            logger.error('Failed to unsubscribe from channels {channels}: %s', e)
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {'operation': 'unsubscribe', 'level': 'l2', 'status': 'error', 'security_context': self._default_security_context.agent_id if self._default_security_context else 'none'})
            return False

    async def acquire_lock(self, key: str, timeout: int=10) -> str | None:
        """Acquire distributed lock (CacheLockProtocol compliance).

        Args:
            key: Lock key
            timeout: Lock timeout in seconds

        Returns:
            Lock token if acquired, None if failed
        """
        if not self._redis_master:
            logger.warning('Distributed locking not available - Redis not connected')
            return None
        try:
            if self._protocol_security_required and self._default_security_context:
                if not self._validate_security_context(self._default_security_context):
                    raise RedisSecurityError(f'Invalid security context for acquire_lock on key {key}')
            import uuid
            lock_token = str(uuid.uuid4())
            lock_key = f'lock:{key}'
            acquired = await self._redis_master.set(lock_key, lock_token, nx=True, ex=timeout)
            if acquired:
                self._active_locks[key] = {'token': lock_token, 'acquired_at': time.time(), 'timeout': timeout, 'expires_at': time.time() + timeout}
                if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                    cache_operations_counter.add(1, {'operation': 'acquire_lock', 'level': 'l2', 'status': 'success', 'security_context': self._default_security_context.agent_id if self._default_security_context else 'none'})
                logger.debug('Acquired lock for key %s with token %s...', key, lock_token[:8])
                return lock_token
            logger.debug('Could not acquire lock for key %s - already locked', key)
            return None
        except Exception as e:
            logger.error('Failed to acquire lock for key {key}: %s', e)
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {'operation': 'acquire_lock', 'level': 'l2', 'status': 'error', 'security_context': self._default_security_context.agent_id if self._default_security_context else 'none'})
            return None

    async def release_lock(self, key: str, token: str) -> bool:
        """Release distributed lock (CacheLockProtocol compliance).

        Args:
            key: Lock key
            token: Lock token returned from acquire_lock

        Returns:
            True if successfully released, False otherwise
        """
        if not self._redis_master:
            logger.warning('Distributed locking not available - Redis not connected')
            return False
        try:
            if self._protocol_security_required and self._default_security_context:
                if not self._validate_security_context(self._default_security_context):
                    raise RedisSecurityError(f'Invalid security context for release_lock on key {key}')
            lock_key = f'lock:{key}'
            lua_script = '\n            if redis.call("get", KEYS[1]) == ARGV[1] then\n                return redis.call("del", KEYS[1])\n            else\n                return 0\n            end\n            '
            result = await self._redis_master.eval(lua_script, 1, lock_key, token)
            released = bool(result)
            if released:
                if key in self._active_locks:
                    del self._active_locks[key]
                if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                    cache_operations_counter.add(1, {'operation': 'release_lock', 'level': 'l2', 'status': 'success', 'security_context': self._default_security_context.agent_id if self._default_security_context else 'none'})
                logger.debug('Released lock for key %s', key)
            else:
                logger.warning('Could not release lock for key %s - token mismatch or lock expired', key)
            return released
        except Exception as e:
            logger.error('Failed to release lock for key {key}: %s', e)
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {'operation': 'release_lock', 'level': 'l2', 'status': 'error', 'security_context': self._default_security_context.agent_id if self._default_security_context else 'none'})
            return False

    async def extend_lock(self, key: str, token: str, timeout: int) -> bool:
        """Extend lock timeout (CacheLockProtocol compliance).

        Args:
            key: Lock key
            token: Lock token returned from acquire_lock
            timeout: New timeout in seconds

        Returns:
            True if successfully extended, False otherwise
        """
        if not self._redis_master:
            logger.warning('Distributed locking not available - Redis not connected')
            return False
        try:
            if self._protocol_security_required and self._default_security_context:
                if not self._validate_security_context(self._default_security_context):
                    raise RedisSecurityError(f'Invalid security context for extend_lock on key {key}')
            lock_key = f'lock:{key}'
            lua_script = '\n            if redis.call("get", KEYS[1]) == ARGV[1] then\n                return redis.call("expire", KEYS[1], ARGV[2])\n            else\n                return 0\n            end\n            '
            result = await self._redis_master.eval(lua_script, 1, lock_key, token, timeout)
            extended = bool(result)
            if extended:
                if key in self._active_locks:
                    self._active_locks[key].update({'timeout': timeout, 'expires_at': time.time() + timeout})
                if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                    cache_operations_counter.add(1, {'operation': 'extend_lock', 'level': 'l2', 'status': 'success', 'security_context': self._default_security_context.agent_id if self._default_security_context else 'none'})
                logger.debug('Extended lock for key {key} by %s seconds', timeout)
            else:
                logger.warning('Could not extend lock for key %s - token mismatch or lock expired', key)
            return extended
        except Exception as e:
            logger.error('Failed to extend lock for key {key}: %s', e)
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {'operation': 'extend_lock', 'level': 'l2', 'status': 'error', 'security_context': self._default_security_context.agent_id if self._default_security_context else 'none'})
            return False

    async def get_from_level(self, key: str, level: int) -> Any | None:
        """Get value from specific cache level (MultiLevelCacheProtocol compliance).

        Args:
            key: Cache key
            level: Cache level (1=L1/memory, 2=L2/Redis, 3=database)

        Returns:
            Cached value or None if not found
        """
        try:
            if level == 1:
                if self._l1_cache:
                    return self._l1_cache.get(key)
                return None
            if level == 2:
                if self._redis_master:
                    if self._protocol_security_required and self._default_security_context:
                        if not self._validate_security_context(self._default_security_context):
                            logger.warning('Invalid security context for get_from_level L2 on key %s', key)
                            return None
                    redis_value = await self._redis_master.get(key)
                    if redis_value is not None:
                        try:
                            return json.loads(redis_value) if isinstance(redis_value, (str, bytes)) else redis_value
                        except (json.JSONDecodeError, TypeError):
                            return redis_value
                return None
            if level == 3:
                logger.debug('L3 cache level not implemented for key %s', key)
                return None
            logger.warning('Invalid cache level {level} for key %s', key)
            return None
        except Exception as e:
            logger.error('Error getting from cache level {level} for key {key}: %s', e)
            return None

    async def set_to_level(self, key: str, value: Any, level: int, ttl: int | None=None) -> bool:
        """Set value to specific cache level (MultiLevelCacheProtocol compliance).

        Args:
            key: Cache key
            value: Value to cache
            level: Cache level (1=L1/memory, 2=L2/Redis, 3=database)
            ttl: Optional TTL in seconds

        Returns:
            True if successfully cached, False otherwise
        """
        try:
            if level == 1:
                if self._l1_cache:
                    self._l1_cache.set(key, value, ttl)
                    return True
                return False
            if level == 2:
                if self._redis_master:
                    if self._protocol_security_required and self._default_security_context:
                        if not self._validate_security_context(self._default_security_context):
                            logger.warning('Invalid security context for set_to_level L2 on key %s', key)
                            return False
                    try:
                        serialized_value = json.dumps(value) if not isinstance(value, (str, bytes)) else value
                    except (TypeError, ValueError):
                        logger.warning('Failed to serialize value for L2 cache key %s', key)
                        return False
                    if ttl:
                        await self._redis_master.setex(key, ttl, serialized_value)
                    else:
                        await self._redis_master.set(key, serialized_value)
                    return True
                return False
            if level == 3:
                logger.debug('L3 cache level not implemented for key %s', key)
                return False
            logger.warning('Invalid cache level {level} for key %s', key)
            return False
        except Exception as e:
            logger.error('Error setting to cache level {level} for key {key}: %s', e)
            return False

    async def invalidate_levels(self, key: str, levels: list[int]) -> bool:
        """Invalidate key from specific levels (MultiLevelCacheProtocol compliance).

        Args:
            key: Cache key to invalidate
            levels: List of cache levels to invalidate from

        Returns:
            True if successfully invalidated from any level, False otherwise
        """
        success = False
        for level in levels:
            try:
                if level == 1:
                    if self._l1_cache:
                        if self._l1_cache.delete(key):
                            success = True
                            logger.debug('Invalidated key %s from L1 cache', key)
                elif level == 2:
                    if self._redis_master:
                        if self._protocol_security_required and self._default_security_context:
                            if not self._validate_security_context(self._default_security_context):
                                logger.warning('Invalid security context for invalidate_levels L2 on key %s', key)
                                continue
                        deleted_count = await self._redis_master.delete(key)
                        if deleted_count > 0:
                            success = True
                            logger.debug('Invalidated key %s from L2 cache', key)
                elif level == 3:
                    logger.debug('L3 cache level invalidation not implemented for key %s', key)
                else:
                    logger.warning('Invalid cache level %s for invalidation of key %s', level, key)
            except Exception as e:
                logger.error('Error invalidating key {key} from level {level}: %s', e)
        if success and key in self._access_patterns:
            del self._access_patterns[key]
        return success

    async def get_cache_hierarchy(self) -> list[str]:
        """Get cache level hierarchy (MultiLevelCacheProtocol compliance).

        Returns:
            List of cache level names in hierarchical order
        """
        hierarchy = []
        if self._l1_cache:
            hierarchy.append('L1_Memory')
        if self._redis_master:
            if self._redis_sentinel:
                hierarchy.append('L2_Redis_Sentinel')
            else:
                hierarchy.append('L2_Redis_Direct')
        hierarchy.append('L3_Database_Fallback')
        return hierarchy

    async def get_cached(self, key: str, security_context: SecurityContext | None=None) -> Any | None:
        """Get value from multi-level cache with optional security context validation.

        Args:
            key: Cache key
            security_context: Optional security context for L2 Redis operations

        Returns:
            Cached value or None if not found or security validation fails
        """
        start_time = time.time()
        try:
            if self._l1_cache:
                l1_value = self._l1_cache.get(key)
                if l1_value is not None:
                    self._cache_l1_hits += 1
                    self._cache_total_requests += 1
                    if key not in self._access_patterns:
                        self._access_patterns[key] = AccessPattern(key=key)
                    self._access_patterns[key].record_access()
                    if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                        cache_operations_counter.add(1, {'operation': 'get', 'level': 'l1', 'status': 'hit', 'security_context': security_context.agent_id if security_context else 'none'})
                    return l1_value
            if self._redis_master and security_context:
                try:
                    if not self._validate_security_context(security_context):
                        logger.warning('Invalid security context for cache key %s', key)
                        return None
                    redis_value = await self._redis_master.get(key)
                    if redis_value is not None:
                        try:
                            deserialized_value = json.loads(redis_value) if isinstance(redis_value, (str, bytes)) else redis_value
                            if self._l1_cache:
                                self._l1_cache.set(key, deserialized_value)
                            self._cache_l2_hits += 1
                            self._cache_total_requests += 1
                            if key not in self._access_patterns:
                                self._access_patterns[key] = AccessPattern(key=key)
                            self._access_patterns[key].record_access()
                            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                                cache_operations_counter.add(1, {'operation': 'get', 'level': 'l2', 'status': 'hit', 'security_context': security_context.agent_id})
                            return deserialized_value
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning('Failed to deserialize cached value for key %s: %s', key, e)
                            return None
                except Exception as e:
                    logger.warning('L2 cache error for key {key}: %s', e)
                    if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                        cache_operations_counter.add(1, {'operation': 'get', 'level': 'l2', 'status': 'error', 'security_context': security_context.agent_id if security_context else 'none'})
                    return None
            self._cache_total_requests += 1
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {'operation': 'get', 'level': 'miss', 'status': 'miss', 'security_context': security_context.agent_id if security_context else 'none'})
            return None
        finally:
            response_time = (time.time() - start_time) * 1000
            self._cache_response_times.append(response_time)
            op_stats = self._cache_operation_stats['get']
            op_stats['count'] += 1
            op_stats['total_time'] += response_time
            op_stats['min_time'] = min(op_stats['min_time'], response_time)
            op_stats['max_time'] = max(op_stats['max_time'], response_time)

    async def set_cached(self, key: str, value: Any, ttl_seconds: int | None=None, security_context: SecurityContext | None=None) -> bool:
        """Set value in multi-level cache with optional security context validation.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Optional TTL in seconds
            security_context: Optional security context for L2 Redis operations

        Returns:
            True if successfully cached, False otherwise
        """
        start_time = time.time()
        success = False
        try:
            if self._l1_cache:
                self._l1_cache.set(key, value, ttl_seconds)
                success = True
                if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                    cache_operations_counter.add(1, {'operation': 'set', 'level': 'l1', 'status': 'success', 'security_context': security_context.agent_id if security_context else 'none'})
            if self._redis_master and security_context:
                try:
                    if not self._validate_security_context(security_context):
                        logger.warning('Invalid security context for cache key %s', key)
                        return success
                    try:
                        serialized_value = json.dumps(value) if not isinstance(value, (str, bytes)) else value
                    except (TypeError, ValueError) as e:
                        logger.warning('Failed to serialize value for key {key}: %s', e)
                        return success
                    if ttl_seconds:
                        await self._redis_master.setex(key, ttl_seconds, serialized_value)
                    else:
                        await self._redis_master.set(key, serialized_value)
                    success = True
                    if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                        cache_operations_counter.add(1, {'operation': 'set', 'level': 'l2', 'status': 'success', 'security_context': security_context.agent_id})
                except Exception as e:
                    logger.warning('L2 cache set error for key {key}: %s', e)
                    if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                        cache_operations_counter.add(1, {'operation': 'set', 'level': 'l2', 'status': 'error', 'security_context': security_context.agent_id if security_context else 'none'})
            return success
        finally:
            response_time = (time.time() - start_time) * 1000
            self._cache_response_times.append(response_time)
            op_stats = self._cache_operation_stats['set']
            op_stats['count'] += 1
            op_stats['total_time'] += response_time
            op_stats['min_time'] = min(op_stats['min_time'], response_time)
            op_stats['max_time'] = max(op_stats['max_time'], response_time)
            if not success:
                op_stats['error_count'] += 1

    async def delete_cached(self, key: str, security_context: SecurityContext | None=None) -> bool:
        """Delete value from multi-level cache with optional security context validation.

        Args:
            key: Cache key to delete
            security_context: Optional security context for L2 Redis operations

        Returns:
            True if successfully deleted from any cache level, False otherwise
        """
        start_time = time.time()
        success = False
        try:
            if self._l1_cache:
                l1_deleted = self._l1_cache.delete(key)
                if l1_deleted:
                    success = True
                    if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                        cache_operations_counter.add(1, {'operation': 'delete', 'level': 'l1', 'status': 'success', 'security_context': security_context.agent_id if security_context else 'none'})
            if self._redis_master and security_context:
                try:
                    if not self._validate_security_context(security_context):
                        logger.warning('Invalid security context for cache key %s', key)
                        return success
                    deleted_count = await self._redis_master.delete(key)
                    if deleted_count > 0:
                        success = True
                        if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                            cache_operations_counter.add(1, {'operation': 'delete', 'level': 'l2', 'status': 'success', 'security_context': security_context.agent_id})
                except Exception as e:
                    logger.warning('L2 cache delete error for key {key}: %s', e)
                    if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                        cache_operations_counter.add(1, {'operation': 'delete', 'level': 'l2', 'status': 'error', 'security_context': security_context.agent_id if security_context else 'none'})
            if key in self._access_patterns:
                del self._access_patterns[key]
            return success
        finally:
            response_time = (time.time() - start_time) * 1000
            self._cache_response_times.append(response_time)
            op_stats = self._cache_operation_stats['delete']
            op_stats['count'] += 1
            op_stats['total_time'] += response_time
            op_stats['min_time'] = min(op_stats['min_time'], response_time)
            op_stats['max_time'] = max(op_stats['max_time'], response_time)
            if not success:
                op_stats['error_count'] += 1

    async def exists_cached(self, key: str, security_context: SecurityContext | None=None) -> bool:
        """Check if key exists in multi-level cache with optional security context validation.

        Args:
            key: Cache key to check
            security_context: Optional security context for L2 Redis operations

        Returns:
            True if key exists in any cache level, False otherwise
        """
        start_time = time.time()
        try:
            if self._l1_cache:
                l1_value = self._l1_cache.get(key)
                if l1_value is not None:
                    if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                        cache_operations_counter.add(1, {'operation': 'exists', 'level': 'l1', 'status': 'found', 'security_context': security_context.agent_id if security_context else 'none'})
                    return True
            if self._redis_master and security_context:
                try:
                    if not self._validate_security_context(security_context):
                        logger.warning('Invalid security context for cache key %s', key)
                        return False
                    exists = await self._redis_master.exists(key)
                    if exists:
                        if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                            cache_operations_counter.add(1, {'operation': 'exists', 'level': 'l2', 'status': 'found', 'security_context': security_context.agent_id})
                        return True
                except Exception as e:
                    logger.warning('L2 cache exists error for key {key}: %s', e)
                    if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                        cache_operations_counter.add(1, {'operation': 'exists', 'level': 'l2', 'status': 'error', 'security_context': security_context.agent_id if security_context else 'none'})
                    return False
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {'operation': 'exists', 'level': 'miss', 'status': 'not_found', 'security_context': security_context.agent_id if security_context else 'none'})
            return False
        finally:
            response_time = (time.time() - start_time) * 1000
            self._cache_response_times.append(response_time)
            op_stats = self._cache_operation_stats['exists']
            op_stats['count'] += 1
            op_stats['total_time'] += response_time
            op_stats['min_time'] = min(op_stats['min_time'], response_time)
            op_stats['max_time'] = max(op_stats['max_time'], response_time)

    def _validate_security_context(self, security_context: SecurityContext) -> bool:
        """Validate security context for Redis operations.

        Args:
            security_context: Security context to validate

        Returns:
            True if valid, False otherwise
        """
        if not security_context:
            return False
        if not security_context.agent_id:
            logger.warning('Security context missing agent_id')
            return False
        if not security_context.authenticated:
            logger.warning('Security context for %s not authenticated', security_context.agent_id)
            return False
        context_age = time.time() - security_context.created_at
        if context_age > 3600:
            logger.warning('Security context for %s expired (age: %ss)', security_context.agent_id, context_age)
            return False
        return True
_unified_managers: dict[ManagerMode, UnifiedConnectionManager] = {}

def get_unified_manager(mode: ManagerMode=ManagerMode.ASYNC_MODERN) -> UnifiedConnectionManager:
    """Get or create unified manager for specified mode"""
    global _unified_managers
    if mode not in _unified_managers:
        _unified_managers[mode] = UnifiedConnectionManager(mode)
    return _unified_managers[mode]

def get_mcp_connection_pool() -> UnifiedConnectionManager:
    """Get MCP-optimized connection manager (replaces MCPConnectionPool)"""
    return get_unified_manager(ManagerMode.MCP_SERVER)

async def get_mcp_session():
    """Get MCP database session for general use (replaces MCPConnectionPool function)"""
    manager = get_unified_manager(ManagerMode.MCP_SERVER)
    return manager.get_async_session()

async def get_mcp_read_session():
    """Get MCP read-only session (replaces MCPConnectionPool function)"""
    manager = get_unified_manager(ManagerMode.MCP_SERVER)
    return manager.get_mcp_read_session()

async def get_mcp_feedback_session():
    """Get MCP feedback session (replaces MCPConnectionPool function)"""
    manager = get_unified_manager(ManagerMode.MCP_SERVER)
    return manager.get_feedback_session()

def get_connection_pool_optimizer() -> UnifiedConnectionManager:
    """Get connection pool optimizer (replaces ConnectionPoolOptimizer)"""
    return get_unified_manager(ManagerMode.ASYNC_MODERN)

def get_adaptive_connection_pool() -> UnifiedConnectionManager:
    """Get adaptive connection pool (replaces AdaptiveConnectionPool)"""
    return get_unified_manager(ManagerMode.ASYNC_MODERN)

def get_connection_pool_manager() -> UnifiedConnectionManager:
    """Get connection pool manager (replaces ConnectionPoolManager)"""
    return get_unified_manager(ManagerMode.ASYNC_MODERN)

async def create_security_context(agent_id: str, tier: str='basic', authenticated: bool=True, authentication_method: str='system', permissions: list[str] | None=None, security_level: str='basic', session_id: str | None=None, expires_minutes: int | None=None) -> SecurityContext:
    """Create enhanced security context for unified database and application operations.

    Default to authenticated=True for secure-by-default behavior.

    Args:
        agent_id: Agent identifier
        tier: Security tier (basic, professional, enterprise)
        authenticated: Authentication status
        authentication_method: Authentication method used
        permissions: List of permissions granted
        security_level: Security level (basic, enhanced, high, critical)
        session_id: Optional session identifier
        expires_minutes: Optional expiration time in minutes

    Returns:
        Enhanced SecurityContext instance
    """
    current_time = time.time()
    expires_at = current_time + expires_minutes * 60 if expires_minutes else None
    return SecurityContext(agent_id=agent_id, tier=tier, authenticated=authenticated, created_at=current_time, authentication_method=authentication_method, authentication_timestamp=current_time, session_id=session_id, permissions=permissions or [], security_level=security_level, expires_at=expires_at, last_used=current_time)

async def create_security_context_from_auth_result(auth_result=None, agent_id: str | None=None, tier: str | None=None, authenticated: bool | None=None, authentication_method: str | None=None, permissions: list[str] | None=None, security_level: str | None=None, expires_minutes: int | None=None, auth_result_metadata: dict[str, Any] | None=None) -> SecurityContext:
    """Create security context from UnifiedAuthenticationManager authentication result.

    Provides seamless integration between authentication and database layers
    with comprehensive security information transfer. Can work with either
    AuthenticationResult objects or direct parameters for flexible integration.

    Args:
        auth_result: AuthenticationResult from UnifiedAuthenticationManager (optional)
        agent_id: Agent identifier (if not using auth_result)
        tier: Security tier (if not using auth_result)
        authenticated: Authentication status (if not using auth_result)
        authentication_method: Authentication method used (if not using auth_result)
        permissions: User permissions (if not using auth_result)
        security_level: Security level (if not using auth_result)
        expires_minutes: Context expiration in minutes (if not using auth_result)
        auth_result_metadata: Additional metadata from authentication process

    Returns:
        Enhanced SecurityContext with authentication details
    """
    current_time = time.time()
    if auth_result:
        auth_time_ms = auth_result.performance_metrics.get('total_auth_time_ms', 0.0)
        agent_id_final = auth_result.agent_id or 'unknown'
        tier_final = auth_result.rate_limit_tier
        authenticated_final = auth_result.success
        auth_method_final = auth_result.authentication_method.value if hasattr(auth_result.authentication_method, 'value') else str(auth_result.authentication_method)
        permissions_final = auth_result.audit_metadata.get('permissions', [])
        session_id_final = auth_result.session_id
        security_level_final = 'basic'
        if auth_result.rate_limit_tier in ['professional', 'enterprise']:
            security_level_final = 'enhanced'
        if auth_method_final == 'api_key':
            security_level_final = 'high' if security_level_final == 'enhanced' else 'enhanced'
        audit_metadata = {'authentication_result_integration': True, 'auth_result_status': auth_result.status.value if hasattr(auth_result.status, 'value') else str(auth_result.status), **auth_result.audit_metadata}
        expires_at = time.time() + expires_minutes * 60 if expires_minutes else None
    else:
        auth_time_ms = 0.0
        agent_id_final = agent_id or 'unknown'
        tier_final = tier or 'basic'
        authenticated_final = authenticated if authenticated is not None else False
        auth_method_final = authentication_method or 'unknown'
        permissions_final = permissions or []
        session_id_final = None
        security_level_final = security_level or 'basic'
        audit_metadata = {'direct_parameter_creation': True, 'authentication_manager_integration': True, **(auth_result_metadata or {})}
        expires_at = time.time() + expires_minutes * 60 if expires_minutes else None
    validation_result = SecurityValidationResult(validated=authenticated_final, validation_method=auth_method_final, validation_timestamp=current_time, validation_duration_ms=auth_time_ms, rate_limit_status='authenticated' if authenticated_final else 'failed', audit_trail_id=audit_metadata.get('audit_trail_id', f'auth_{int(current_time * 1000000)}'))
    performance_metrics = SecurityPerformanceMetrics(authentication_time_ms=auth_time_ms, total_security_overhead_ms=auth_time_ms, operations_count=1, last_performance_check=current_time)
    return SecurityContext(agent_id=agent_id_final, tier=tier_final, authenticated=authenticated_final, created_at=current_time, authentication_method=auth_method_final, authentication_timestamp=current_time, session_id=session_id_final, permissions=permissions_final, validation_result=validation_result, performance_metrics=performance_metrics, audit_metadata=audit_metadata, security_level=security_level_final, zero_trust_validated=authenticated_final, expires_at=expires_at, last_used=current_time)

async def create_security_context_from_security_manager(agent_id: str, security_manager, additional_context: dict[str, Any] | None=None) -> SecurityContext:
    """Create security context with validation from UnifiedSecurityManager.

    Integrates comprehensive security validation and threat assessment
    into database security context for unified security enforcement.

    Args:
        agent_id: Agent identifier
        security_manager: UnifiedSecurityManager instance
        additional_context: Additional security context information

    Returns:
        SecurityContext with comprehensive security validation
    """
    current_time = time.time()
    try:
        security_status = await security_manager.get_security_status()
        validation_result = SecurityValidationResult(validated=True, validation_method='security_manager', validation_timestamp=current_time, validation_duration_ms=0.0, rate_limit_status='validated', encryption_required=security_manager.config.require_encryption, audit_trail_id=f'sm_{int(current_time * 1000000)}')
        threat_level = 'low'
        threat_score = 0.0
        threat_factors = []
        if security_status.get('metrics', {}).get('violation_rate', 0) > 0.1:
            threat_level = 'medium'
            threat_score = 0.3
            threat_factors.append('elevated_violation_rate')
        if security_status.get('metrics', {}).get('active_incidents', 0) > 0:
            threat_level = 'high'
            threat_score = 0.6
            threat_factors.append('active_security_incidents')
        threat_assessment = SecurityThreatScore(level=threat_level, score=threat_score, factors=threat_factors, last_updated=current_time)
        avg_operation_time = security_status.get('performance', {}).get('average_operation_time_ms', 0.0)
        performance_metrics = SecurityPerformanceMetrics(authentication_time_ms=avg_operation_time, authorization_time_ms=avg_operation_time, validation_time_ms=avg_operation_time, total_security_overhead_ms=avg_operation_time * 3, operations_count=1, last_performance_check=current_time)
        authenticated = additional_context.get('authenticated', True) if additional_context else True
        if additional_context and additional_context.get('authentication_method') == 'failed':
            authenticated = False
        security_level = security_manager.config.security_level.value
        if additional_context and additional_context.get('threat_detected'):
            security_level = 'basic'
        tier = security_manager.config.rate_limit_tier.value
        return SecurityContext(agent_id=agent_id, tier=tier, authenticated=authenticated, created_at=current_time, authentication_method=additional_context.get('authentication_method', 'security_manager') if additional_context else 'security_manager', authentication_timestamp=current_time, session_id=additional_context.get('session_id') if additional_context else None, permissions=additional_context.get('permissions', []) if additional_context else [], validation_result=validation_result, threat_score=threat_assessment, performance_metrics=performance_metrics, audit_metadata={'security_manager_integration': True, 'security_manager_mode': security_manager.mode.value, 'security_configuration': {'security_level': security_manager.config.security_level.value, 'rate_limit_tier': security_manager.config.rate_limit_tier.value, 'zero_trust_mode': security_manager.config.zero_trust_mode, 'fail_secure': security_manager.config.fail_secure}, **(additional_context or {})}, security_level=security_level, zero_trust_validated=security_manager.config.zero_trust_mode, encryption_context={'required': security_manager.config.require_encryption}, last_used=current_time)
    except Exception as e:
        logger.error('Failed to create security context from security manager: %s', e)
        return await create_security_context(agent_id=agent_id, tier='basic', authenticated=False, authentication_method='failed', security_level='basic')

async def create_system_security_context(operation_type: str='system', security_level: str='high') -> SecurityContext:
    """Create system-level security context for internal operations.

    Args:
        operation_type: Type of system operation
        security_level: Security level for system operations

    Returns:
        System SecurityContext with elevated privileges
    """
    current_time = time.time()
    validation_result = SecurityValidationResult(validated=True, validation_method='system_internal', validation_timestamp=current_time, validation_duration_ms=0.0, rate_limit_status='system_exempt', encryption_required=False, audit_trail_id=f'sys_{int(current_time * 1000000)}')
    return SecurityContext(agent_id='system', tier='system', authenticated=True, created_at=current_time, authentication_method='system_internal', authentication_timestamp=current_time, session_id=None, permissions=['system:all'], validation_result=validation_result, audit_metadata={'operation_type': operation_type}, compliance_tags=['system_internal'], security_level=security_level, zero_trust_validated=True, last_used=current_time)
logger.info('UnifiedConnectionManager is now the default connection manager with consolidated functionality from MCPConnectionPool, AdaptiveConnectionPool, ConnectionPoolOptimizer, and ConnectionPoolManager')
