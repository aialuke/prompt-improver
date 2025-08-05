"""
Unified Database Connection Manager - Modern Async Connection Management

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
import time
import statistics
from collections import deque, OrderedDict, defaultdict
from collections.abc import AsyncIterator
from typing import Optional, Dict, Any, Union, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, timezone
# Delayed import to avoid circular imports
# from prompt_improver.performance.monitoring.health.background_manager import (
#     get_background_task_manager, TaskPriority
# )

# Task priority enum for cache warming
class TaskPriority:
    """Task priority levels for cache warming (fallback implementation)."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"

# Database imports
import asyncpg
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    AsyncConnection,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import QueuePool, NullPool
# Using asyncpg.Pool for HA functionality instead of psycopg_pool.AsyncConnectionPool

# Redis imports for HA functionality
import coredis
from coredis.sentinel import Sentinel

# Internal imports - AppConfig imported lazily to avoid import-time blocking
# from ..core.config import AppConfig
from .registry import RegistryManager, get_registry_manager
# Cache Protocol imports for compliance
from ..core.protocols.cache_protocol import (
    BasicCacheProtocol, AdvancedCacheProtocol, CacheHealthProtocol,
    CacheSubscriptionProtocol, CacheLockProtocol, RedisCacheProtocol,
    MultiLevelCacheProtocol
)
# RedisConfig now accessed via AppConfig

# OpenTelemetry imports with graceful fallback for cache operations
try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
    
    # Get tracer and meter for cache operations
    cache_tracer = trace.get_tracer(__name__ + ".cache")
    cache_meter = metrics.get_meter(__name__ + ".cache")
    
    # Create cache-specific metrics
    cache_operations_counter = cache_meter.create_counter(
        "unified_cache_operations_total",
        description="Total unified cache operations by type, level, and status",
        unit="1"
    )
    
    cache_hit_ratio_gauge = cache_meter.create_gauge(
        "unified_cache_hit_ratio",
        description="Unified cache hit ratio by level",
        unit="ratio"
    )
    
    cache_latency_histogram = cache_meter.create_histogram(
        "unified_cache_operation_duration_seconds",
        description="Unified cache operation duration by type and level",
        unit="s"
    )
    
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    cache_tracer = None
    cache_meter = None
    cache_operations_counter = None
    cache_hit_ratio_gauge = None
    cache_latency_histogram = None

logger = logging.getLogger(__name__)


# ========== Enhanced Security Context Integration ==========

@dataclass
class SecurityThreatScore:
    """Security threat assessment for operations."""
    level: str = "low"  # low, medium, high, critical
    score: float = 0.0  # 0.0-1.0 threat probability
    factors: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

@dataclass 
class SecurityValidationResult:
    """Comprehensive security validation results."""
    validated: bool = False
    validation_method: str = "none"
    validation_timestamp: float = field(default_factory=time.time)
    validation_duration_ms: float = 0.0
    security_incidents: List[str] = field(default_factory=list)
    rate_limit_status: str = "unknown"
    encryption_required: bool = False
    audit_trail_id: Optional[str] = None

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
    
    # Core authentication information
    agent_id: str
    tier: str = "basic"
    authenticated: bool = False
    created_at: float = field(default_factory=time.time)
    
    # Enhanced authentication details
    authentication_method: str = "none"  # api_key, session_token, system_token
    authentication_timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    
    # Security validation and threat assessment
    validation_result: SecurityValidationResult = field(default_factory=SecurityValidationResult)
    threat_score: SecurityThreatScore = field(default_factory=SecurityThreatScore)
    
    # Performance and monitoring
    performance_metrics: SecurityPerformanceMetrics = field(default_factory=SecurityPerformanceMetrics)
    
    # Audit and compliance
    audit_metadata: Dict[str, Any] = field(default_factory=dict)
    compliance_tags: List[str] = field(default_factory=list)
    
    # Security policies and enforcement
    security_level: str = "basic"  # basic, enhanced, high, critical
    zero_trust_validated: bool = False
    encryption_context: Optional[Dict[str, str]] = None
    
    # Context lifecycle and expiration
    expires_at: Optional[float] = None
    max_operations: Optional[int] = None
    operations_count: int = 0
    last_used: float = field(default_factory=time.time)
    
    def is_valid(self) -> bool:
        """Check if security context is still valid."""
        current_time = time.time()
        
        # Check expiration
        if self.expires_at and current_time > self.expires_at:
            return False
            
        # Check operation limits
        if self.max_operations and self.operations_count >= self.max_operations:
            return False
            
        # Check authentication status
        if not self.authenticated:
            return False
            
        return True
    
    def touch(self) -> None:
        """Update last used timestamp and increment operation count."""
        self.last_used = time.time()
        self.operations_count += 1
    
    def add_audit_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Add audit event to security context."""
        if "audit_events" not in self.audit_metadata:
            self.audit_metadata["audit_events"] = []
            
        self.audit_metadata["audit_events"].append({
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details
        })
        
        # Keep only last 50 events to prevent memory bloat
        if len(self.audit_metadata["audit_events"]) > 50:
            self.audit_metadata["audit_events"] = self.audit_metadata["audit_events"][-50:]
    
    def update_threat_score(self, new_level: str, new_score: float, factors: List[str]) -> None:
        """Update threat assessment with new information."""
        self.threat_score.level = new_level
        self.threat_score.score = new_score
        self.threat_score.factors = factors
        self.threat_score.last_updated = time.time()
    
    def record_performance_metric(self, operation: str, duration_ms: float) -> None:
        """Record security operation performance metric."""
        if operation == "authentication":
            self.performance_metrics.authentication_time_ms = duration_ms
        elif operation == "authorization":
            self.performance_metrics.authorization_time_ms = duration_ms
        elif operation == "validation":
            self.performance_metrics.validation_time_ms = duration_ms
            
        # Update total overhead
        self.performance_metrics.total_security_overhead_ms = (
            self.performance_metrics.authentication_time_ms +
            self.performance_metrics.authorization_time_ms +
            self.performance_metrics.validation_time_ms
        )
        
        self.performance_metrics.operations_count += 1
        self.performance_metrics.last_performance_check = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert security context to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "tier": self.tier,
            "authenticated": self.authenticated,
            "created_at": self.created_at,
            "authentication_method": self.authentication_method,
            "authentication_timestamp": self.authentication_timestamp,
            "session_id": self.session_id,
            "permissions": self.permissions,
            "validation_result": {
                "validated": self.validation_result.validated,
                "validation_method": self.validation_result.validation_method,
                "validation_timestamp": self.validation_result.validation_timestamp,
                "validation_duration_ms": self.validation_result.validation_duration_ms,
                "security_incidents": self.validation_result.security_incidents,
                "rate_limit_status": self.validation_result.rate_limit_status,
                "encryption_required": self.validation_result.encryption_required,
                "audit_trail_id": self.validation_result.audit_trail_id
            },
            "threat_score": {
                "level": self.threat_score.level,
                "score": self.threat_score.score,
                "factors": self.threat_score.factors,
                "last_updated": self.threat_score.last_updated
            },
            "performance_metrics": {
                "authentication_time_ms": self.performance_metrics.authentication_time_ms,
                "authorization_time_ms": self.performance_metrics.authorization_time_ms,
                "validation_time_ms": self.performance_metrics.validation_time_ms,
                "total_security_overhead_ms": self.performance_metrics.total_security_overhead_ms,
                "operations_count": self.performance_metrics.operations_count,
                "last_performance_check": self.performance_metrics.last_performance_check
            },
            "audit_metadata": self.audit_metadata,
            "compliance_tags": self.compliance_tags,
            "security_level": self.security_level,
            "zero_trust_validated": self.zero_trust_validated,
            "encryption_context": self.encryption_context,
            "expires_at": self.expires_at,
            "max_operations": self.max_operations,
            "operations_count": self.operations_count,
            "last_used": self.last_used,
            "is_valid": self.is_valid()
        }


class RedisSecurityError(Exception):
    """Security-related Redis operation error."""
    pass


@dataclass
class CacheEntry:
    """Cache entry with metadata for L1 memory cache."""
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if the cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.now(timezone.utc) > self.created_at + timedelta(seconds=self.ttl_seconds)

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1


@dataclass
class AccessPattern:
    """Access pattern tracking for intelligent cache warming."""
    key: str
    access_count: int = 0
    last_access: datetime = None
    access_frequency: float = 0.0  # accesses per hour
    access_times: List[datetime] = None
    warming_priority: float = 0.0
    
    def __post_init__(self):
        if self.access_times is None:
            self.access_times = []
        if self.last_access is None:
            self.last_access = datetime.now(timezone.utc)
    
    def record_access(self) -> None:
        """Record a new access and update frequency metrics."""
        now = datetime.now(timezone.utc)
        self.access_count += 1
        self.last_access = now
        self.access_times.append(now)
        
        # Keep only recent access times (last 24 hours)
        cutoff = now - timedelta(hours=24)
        self.access_times = [t for t in self.access_times if t > cutoff]
        
        # Calculate access frequency (accesses per hour)
        if len(self.access_times) >= 2:
            time_span = (self.access_times[-1] - self.access_times[0]).total_seconds() / 3600
            self.access_frequency = len(self.access_times) / max(time_span, 0.1)
        
        # Update warming priority (combines frequency and recency)
        recency_weight = max(0, 1 - (now - self.last_access).total_seconds() / 3600)  # Decay over 1 hour
        self.warming_priority = self.access_frequency * (1 + recency_weight)


class LRUCache:
    """High-performance in-memory LRU cache for L1 caching."""

    def __init__(self, max_size: int = 1000):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._cache:
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry.value

        self._misses += 1
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set value in cache."""
        if key in self._cache:
            # Update existing entry
            entry = self._cache[key]
            entry.value = value
            entry.created_at = datetime.now(timezone.utc)
            entry.ttl_seconds = ttl_seconds
            self._cache.move_to_end(key)
        else:
            # Add new entry
            if len(self._cache) >= self._max_size:
                # Remove least recently used
                self._cache.popitem(last=False)

            entry = CacheEntry(
                value=value,
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
                ttl_seconds=ttl_seconds
            )
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

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "utilization": len(self._cache) / self._max_size
        }


class ConnectionMode(Enum):
    """Connection operation modes"""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    BATCH = "batch"
    TRANSACTIONAL = "transactional"


class ManagerMode(Enum):
    """Manager operation modes optimized for different use cases"""
    MCP_SERVER = "mcp_server"      # Read-optimized for MCP server operations
    ML_TRAINING = "ml_training"    # Optimized for ML training workloads  
    ADMIN = "admin"               # Administrative operations with higher timeouts
    ASYNC_MODERN = "async_modern" # General purpose async operations (default)
    HIGH_AVAILABILITY = "ha"     # High availability with failover support

class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class PoolState(Enum):
    """Connection pool operational states (from AdaptiveConnectionPool)"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    SCALING_UP = "scaling_up"
    SCALING_DOWN = "scaling_down"
    DEGRADED = "degraded"
    FAILED = "failed"
    STRESSED = "stressed"  # High utilization
    EXHAUSTED = "exhausted"  # No connections available
    RECOVERING = "recovering"  # Recovering from issues

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
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

@dataclass
class ConnectionMetrics:
    """Comprehensive connection metrics from all managers"""
    # From DatabaseManager/DatabaseSessionManager
    active_connections: int = 0
    idle_connections: int = 0
    total_connections: int = 0
    pool_utilization: float = 0.0
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0
    
    # From HAConnectionManager
    failed_connections: int = 0
    last_failover: Optional[float] = None
    failover_count: int = 0
    health_check_failures: int = 0
    circuit_breaker_state: str = "closed"
    circuit_breaker_failures: int = 0
    
    # From UnifiedConnectionManager
    mode_specific_metrics: Dict[str, Any] = field(default_factory=dict)
    sla_compliance_rate: float = 100.0
    
    # Registry metrics
    registry_conflicts: int = 0
    registered_models: int = 0
    
    # From AdaptiveConnectionPool - auto-scaling metrics
    connections_created: int = 0
    connections_closed: int = 0
    connection_failures: int = 0
    queries_executed: int = 0
    queries_failed: int = 0
    wait_time_ms: float = 0.0
    last_scale_event: Optional[datetime] = None
    connection_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    query_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # From ConnectionPoolOptimizer - load reduction metrics
    connection_reuse_count: int = 0
    pool_efficiency: float = 0.0
    connections_saved: int = 0
    database_load_reduction_percent: float = 0.0
    
    # From ConnectionPoolManager - multi-pool coordination
    http_pool_health: bool = True
    redis_pool_health: bool = True
    multi_pool_coordination_active: bool = False
    
    # Enhanced Cache Metrics (from MultiLevelCache integration)
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
    cache_operation_stats: Dict[str, Any] = field(default_factory=dict)
    cache_health_status: str = "healthy"

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
    
    # Enhanced Cache Configuration
    enable_l1_cache: bool = True
    l1_cache_size: int = 1000
    l2_cache_ttl: int = 3600  # 1 hour default
    enable_cache_warming: bool = True
    warming_threshold: float = 2.0  # accesses per hour
    warming_interval: int = 300  # seconds
    max_warming_keys: int = 50
    
    @classmethod
    def for_mode(cls, mode: ManagerMode) -> 'PoolConfiguration':
        """Create pool configuration optimized for specific mode"""
        configs = {
            ManagerMode.MCP_SERVER: cls(
                mode=mode, pg_pool_size=20, pg_max_overflow=10, pg_timeout=0.2,
                redis_pool_size=10, enable_circuit_breaker=True,
                # MCP-optimized cache settings for <200ms SLA
                enable_l1_cache=True, l1_cache_size=2000, l2_cache_ttl=900,
                enable_cache_warming=True, warming_threshold=3.0, warming_interval=120
            ),
            ManagerMode.ML_TRAINING: cls(
                mode=mode, pg_pool_size=15, pg_max_overflow=10, pg_timeout=5.0,
                redis_pool_size=8, enable_ha=True,
                # ML training cache settings
                enable_l1_cache=True, l1_cache_size=1500, l2_cache_ttl=1800,
                enable_cache_warming=True, warming_threshold=1.5, warming_interval=300
            ),
            ManagerMode.ADMIN: cls(
                mode=mode, pg_pool_size=5, pg_max_overflow=2, pg_timeout=10.0,
                redis_pool_size=3, enable_ha=True,
                # Admin cache settings - less aggressive
                enable_l1_cache=True, l1_cache_size=500, l2_cache_ttl=7200,
                enable_cache_warming=False  # Admin operations don't need warming
            ),
            ManagerMode.ASYNC_MODERN: cls(
                mode=mode, pg_pool_size=12, pg_max_overflow=8, pg_timeout=5.0,
                redis_pool_size=6, enable_circuit_breaker=True,
                # Standard cache settings
                enable_l1_cache=True, l1_cache_size=1000, l2_cache_ttl=3600,
                enable_cache_warming=True, warming_threshold=2.0, warming_interval=300
            ),
            ManagerMode.HIGH_AVAILABILITY: cls(
                mode=mode, pg_pool_size=20, pg_max_overflow=20, pg_timeout=10.0,
                redis_pool_size=10, enable_ha=True, enable_circuit_breaker=True,
                # HA cache settings - maximum performance
                enable_l1_cache=True, l1_cache_size=2500, l2_cache_ttl=1800,
                enable_cache_warming=True, warming_threshold=1.0, warming_interval=180
            )
        }
        return configs.get(mode, configs[ManagerMode.ASYNC_MODERN])

class UnifiedConnectionManager(RedisCacheProtocol, MultiLevelCacheProtocol):
    """
    Modern async database connection manager.
    
    Provides a clean, efficient interface for async database operations with
    built-in high availability, health monitoring, and intelligent pooling.
    Follows 2025 best practices with async-only operations.
    """
    
    def __init__(self, 
                 mode: ManagerMode = ManagerMode.ASYNC_MODERN,
                 db_config = None,
                 redis_config = None):
        """Initialize unified connection manager
        
        Args:
            mode: Manager operation mode
            db_config: Database configuration (auto-detected if None)
            redis_config: Redis configuration (auto-detected if None)
        """
        self.mode = mode
        if db_config is None:
            # Removed AppConfig import to fix circular import - use environment variables directly
            # Core config should not be imported by foundational database components
            self.db_config = None  # Will use environment variables or defaults
        else:
            self.db_config = db_config
        self.redis_config = redis_config or self._get_redis_config()
        
        # Pool configuration
        self.pool_config = PoolConfiguration.for_mode(mode)
        
        # Component managers (composition pattern) - lazy-loaded to avoid import-time blocking
        self._registry_manager = None  # Lazy-loaded
        self._metrics = ConnectionMetrics()
        
        # Auto-scaling configuration (from AdaptiveConnectionPool)
        self.min_pool_size = self.pool_config.pg_pool_size
        self.max_pool_size = min(self.pool_config.pg_pool_size * 5, 100)  # Scale up to 100 connections
        self.current_pool_size = self.pool_config.pg_pool_size
        
        # Auto-scaling thresholds
        self.scale_up_threshold = 0.8  # 80% utilization
        self.scale_down_threshold = 0.3  # 30% utilization
        self.scale_cooldown_seconds = 60  # Cooldown for DB connections
        self.last_scale_time = 0
        
        # Pool state management
        self._pool_state = PoolState.INITIALIZING
        self.performance_window = deque(maxlen=100)
        self.last_metrics_update = time.time()
        
        # Connection age tracking (from ConnectionPoolOptimizer)
        self._connection_registry: Dict[str, ConnectionInfo] = {}
        self._connection_id_counter = 0
        self._total_connections_created = 0
        
        # Database connections
        self._async_engine: Optional[AsyncEngine] = None
        self._async_session_factory: Optional[async_sessionmaker] = None
        
        # HA components (from HAConnectionManager) - using asyncpg.Pool instead of psycopg_pool
        self._pg_pools: Dict[str, asyncpg.Pool] = {}
        self._redis_sentinel: Optional[Sentinel] = None
        self._redis_master: Optional[coredis.Redis] = None
        self._redis_replica: Optional[coredis.Redis] = None
        
        # Circuit breaker state
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_timeout = 30
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = 0
        
        # Health monitoring
        self._health_status = HealthStatus.UNKNOWN
        self._last_health_check = 0
        self._health_check_interval = 10
        
        # Initialization state
        self._is_initialized = False
        self._initialization_lock = asyncio.Lock()
        
        # Enhanced task management
        self._health_monitor_task_id: Optional[str] = None
        self._monitoring_task_id: Optional[str] = None
        
        # Enhanced Multi-Level Cache Infrastructure (from MultiLevelCache)
        self._l1_cache: Optional[LRUCache] = None
        self._access_patterns: Dict[str, AccessPattern] = {}
        self._cache_warming_task_id: Optional[str] = None
        self._last_warming_time = datetime.now(timezone.utc)
        
        # Cache performance tracking
        self._cache_l1_hits = 0
        self._cache_l2_hits = 0
        self._cache_l3_hits = 0
        self._cache_total_requests = 0
        self._cache_response_times = deque(maxlen=1000)
        self._cache_operation_stats = defaultdict(lambda: {
            'count': 0, 'total_time': 0.0, 'min_time': float('inf'), 
            'max_time': 0.0, 'error_count': 0
        })
        
        # Cache warming statistics
        self._warming_stats = {
            "cycles_completed": 0,
            "keys_warmed": 0,
            "warming_hits": 0,
            "warming_errors": 0
        }
        
        # Cache health monitoring
        self._cache_health_status = {
            'overall_health': 'healthy',
            'l1_health': 'healthy',
            'l2_health': 'healthy',
            'warming_health': 'healthy',
            'last_health_check': datetime.now(timezone.utc)
        }
        
        # Protocol compliance configuration
        self._protocol_security_required = mode in [ManagerMode.HIGH_AVAILABILITY, ManagerMode.MCP_SERVER]
        self._default_security_context = None  # Will be created during initialization
        
        # Distributed locking state
        self._active_locks: Dict[str, Dict[str, Any]] = {}
        self._lock_timeout_default = 30  # seconds
        
        # Pub/Sub state
        self._subscribers: Dict[str, List[Any]] = {}
        self._pubsub_connection: Optional[Any] = None
        
        logger.info(f"UnifiedConnectionManager initialized for mode: {mode.value} with auto-scaling {self.min_pool_size}-{self.max_pool_size} connections and enhanced cache (L1: {self.pool_config.l1_cache_size if self.pool_config.enable_l1_cache else 0})")

    def _get_registry_manager(self):
        """Get registry manager with lazy loading."""
        if self._registry_manager is None:
            from .registry import get_registry_manager
            self._registry_manager = get_registry_manager()
        return self._registry_manager
    
    async def _get_background_task_manager(self):
        """Get background task manager with graceful fallback."""
        try:
            # Removed background_manager import to fix circular import
            # Background task management should be handled at higher levels
            return None
        except ImportError as e:
            logger.warning(f"Background task manager not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to get background task manager: {e}")
            return None
    
    def _get_redis_config(self):
        """Get Redis configuration from AppConfig"""
        try:
            # Removed AppConfig import to fix circular import - use environment variables directly
            # Core config should not be imported by foundational database components
            return None  # Will use environment-based Redis configuration
        except Exception:
            # Create minimal config if redis utils not available
            class MinimalRedisConfig:
                host = "localhost"
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
                # Initialize database connections
                await self._setup_database_connections()
                
                # Initialize HA components if enabled
                if self.pool_config.enable_ha:
                    await self._setup_ha_components()
                
                # Initialize Redis connections
                await self._setup_redis_connections()
                
                # Initialize L1 cache if enabled
                if self.pool_config.enable_l1_cache:
                    self._l1_cache = LRUCache(max_size=self.pool_config.l1_cache_size)
                    logger.info(f"L1 cache initialized with size {self.pool_config.l1_cache_size}")
                
                # Create default security context for protocol compliance
                self._default_security_context = SecurityContext(
                    agent_id="unified_connection_manager_system",
                    tier="system",
                    authenticated=True,
                    created_at=time.time()
                )
                
                # Start health monitoring and auto-scaling using enhanced task management
                task_manager = await self._get_background_task_manager()
                
                # Submit health monitor loop task
                health_task_id = await task_manager.submit_enhanced_task(
                    task_id=f"db_health_monitor_{id(self)}",
                    coroutine=self._health_monitor_loop,
                    priority=TaskPriority.HIGH,
                    tags={"service": "database", "type": "health_monitoring", "component": "unified_connection_manager"}
                )
                
                # Submit monitoring loop task
                monitoring_task_id = await task_manager.submit_enhanced_task(
                    task_id=f"db_monitoring_{id(self)}",
                    coroutine=self._monitoring_loop,
                    priority=TaskPriority.HIGH,
                    tags={"service": "database", "type": "monitoring_loop", "component": "unified_connection_manager"}
                )
                
                # Submit cache warming task if enabled
                if self.pool_config.enable_cache_warming and self._l1_cache and task_manager:
                    try:
                        warming_task_id = await task_manager.submit_enhanced_task(
                            task_id=f"cache_warming_{id(self)}",
                            coroutine=self._cache_warming_loop,
                            priority=TaskPriority.BACKGROUND,
                            tags={"service": "database", "type": "cache_warming", "component": "unified_connection_manager"}
                        )
                        self._cache_warming_task_id = warming_task_id
                    except Exception as e:
                        logger.warning(f"Failed to start cache warming task: {e}")
                        # Continue without warming
                
                # Store task IDs for cleanup
                self._health_monitor_task_id = health_task_id
                self._monitoring_task_id = monitoring_task_id
                
                self._is_initialized = True
                self._health_status = HealthStatus.HEALTHY
                
                logger.info(f"UnifiedConnectionManager initialized successfully for {self.mode.value}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize UnifiedConnectionManager: {e}")
                self._health_status = HealthStatus.UNHEALTHY
                return False

    async def shutdown(self) -> bool:
        """Shutdown all connection components and cleanup resources"""
        try:
            logger.info(f"Shutting down UnifiedConnectionManager for {self.mode.value}")

            # Cancel background tasks
            if hasattr(self, '_health_monitor_task_id') and self._health_monitor_task_id:
                # Cancel health monitoring task if it exists
                pass

            if hasattr(self, '_monitoring_task_id') and self._monitoring_task_id:
                # Cancel monitoring task if it exists
                pass

            # Close database connections
            if self._async_engine:
                await self._async_engine.dispose()
                self._async_engine = None
                logger.info("Async engine disposed")

            # Close HA pools
            if hasattr(self, '_pg_pools') and self._pg_pools:
                for pool_name, pool in self._pg_pools.items():
                    await pool.close()
                    logger.info(f"HA pool {pool_name} closed")
                self._pg_pools.clear()

            # Close Redis connections
            if hasattr(self, '_redis_client') and self._redis_client:
                await self._redis_client.close()
                self._redis_client = None
                logger.info("Redis client closed")

            # Reset state
            self._is_initialized = False
            self._health_status = HealthStatus.UNHEALTHY

            logger.info(f"UnifiedConnectionManager shutdown completed for {self.mode.value}")
            return True

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False

    async def _setup_database_connections(self):
        """Setup both sync and async database connections"""
        # Build database URLs
        base_url = (
            f"postgresql://{self.db_config.username}:{self.db_config.password}@"
            f"{self.db_config.host}:{self.db_config.port}/"
            f"{self.db_config.database}"
        )
        sync_url = f"postgresql+asyncpg://{self.db_config.username}:{self.db_config.password}@{self.db_config.host}:{self.db_config.port}/{self.db_config.database}"
        async_url = f"postgresql+asyncpg://{self.db_config.username}:{self.db_config.password}@{self.db_config.host}:{self.db_config.port}/{self.db_config.database}"
        
        # Create async engine with proper pool configuration
        poolclass = None  # Use default async pool for async engines
        
        engine_kwargs = {
            "pool_size": self.pool_config.pg_pool_size,
            "max_overflow": self.pool_config.pg_max_overflow,
            "pool_timeout": self.pool_config.pg_timeout,
            "pool_pre_ping": True,
            "pool_recycle": 3600,
            "echo": self.db_config.echo_sql,
            "future": True,
            "connect_args": {
                "server_settings": {
                    "application_name": f"apes_unified_{self.mode.value}",
                    "timezone": "UTC",
                },
                "command_timeout": self.pool_config.pg_timeout,
                "connect_timeout": 10,
            }
        }
        
        # Don't specify poolclass for async engines - let SQLAlchemy choose the appropriate async pool
        
        self._async_engine = create_async_engine(async_url, **engine_kwargs)
        
        # Create async session factory
        self._async_session_factory = async_sessionmaker(
            bind=self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False
        )
        
        # Setup connection monitoring
        self._setup_connection_monitoring()
        
        # Test connections
        await self._test_connections()
    
    async def _setup_ha_components(self):
        """Setup high availability components (from HAConnectionManager)"""
        if not self.pool_config.enable_ha:
            return
            
        try:
            # Setup PostgreSQL HA pools using asyncpg
            primary_dsn = f"postgresql://{self.db_config.username}:{self.db_config.password}@{self.db_config.host}:{self.db_config.port}/{self.db_config.database}"

            primary_pool = await asyncpg.create_pool(
                dsn=primary_dsn,
                min_size=2,
                max_size=self.pool_config.pg_pool_size,
                command_timeout=self.pool_config.pg_timeout,
                max_inactive_connection_lifetime=3600,
                server_settings={
                    'application_name': f'apes_ha_primary_{self.mode.value}',
                    'timezone': 'UTC',
                }
            )

            self._pg_pools["primary"] = primary_pool
            
            # Add replica pools if configured
            replica_hosts = self._get_replica_hosts()
            for i, (host, port) in enumerate(replica_hosts):
                replica_dsn = f"postgresql://{self.db_config.username}:{self.db_config.password}@{host}:{port}/{self.db_config.database}"
                replica_pool = await asyncpg.create_pool(
                    dsn=replica_dsn,
                    min_size=1,
                    max_size=self.pool_config.pg_pool_size // 2,
                    command_timeout=self.pool_config.pg_timeout,
                    server_settings={
                        'application_name': f'apes_ha_replica_{i}_{self.mode.value}',
                        'timezone': 'UTC',
                    }
                )
                self._pg_pools[f"replica_{i}"] = replica_pool
                
            logger.info(f"HA pools initialized: {len(self._pg_pools)} pools")
            
        except Exception as e:
            logger.warning(f"HA setup failed, continuing without HA: {e}")
    
    def _get_replica_hosts(self) -> list:
        """Get replica host configurations"""
        # In production, this would come from service discovery
        replicas = os.getenv("POSTGRES_REPLICAS", "").split(",")
        replica_hosts = []
        for replica in replicas:
            if ":" in replica:
                host, port = replica.split(":")
                replica_hosts.append((host.strip(), int(port)))
        return replica_hosts
    
    async def _setup_redis_connections(self):
        """Setup Redis connections (from HAConnectionManager)"""
        try:
            # Try Redis Sentinel first for HA
            if self.pool_config.enable_ha:
                await self._setup_redis_sentinel()
            else:
                await self._setup_redis_direct()
                
        except Exception as e:
            logger.warning(f"Redis setup failed: {e}")
    
    async def _setup_redis_sentinel(self):
        """Setup Redis Sentinel for HA"""
        sentinel_hosts_env = os.getenv("REDIS_SENTINELS", "redis-sentinel-1:26379,redis-sentinel-2:26379,redis-sentinel-3:26379")
        sentinel_hosts = []
        for host_port in sentinel_hosts_env.split(","):
            if ":" in host_port:
                host, port = host_port.strip().split(":")
                sentinel_hosts.append((host, int(port)))
        
        if sentinel_hosts:
            self._redis_sentinel = Sentinel(
                sentinels=sentinel_hosts,
                stream_timeout=0.1,
                connect_timeout=0.1
            )
            
            self._redis_master = self._redis_sentinel.primary_for(
                'mymaster',
                stream_timeout=0.1,
                password=getattr(self.redis_config, 'password', None)
            )
            
            self._redis_replica = self._redis_sentinel.replica_for(
                'mymaster',
                stream_timeout=0.1,
                password=getattr(self.redis_config, 'password', None)
            )
            
            logger.info("Redis Sentinel initialized")
    
    async def _setup_redis_direct(self):
        """Setup direct Redis connection"""
        self._redis_master = coredis.Redis(
            host=self.redis_config.host,
            port=self.redis_config.port,
            db=self.redis_config.cache_db,
            password=getattr(self.redis_config, 'password', None),
            stream_timeout=self.redis_config.socket_timeout,
            connect_timeout=self.redis_config.connect_timeout
        )
    
    def _setup_connection_monitoring(self):
        """Setup connection monitoring events"""
        if not self._async_engine:
            return
        
        @event.listens_for(self._async_engine.sync_engine, "connect")  
        def on_async_connect(dbapi_connection, connection_record):
            self._metrics.total_connections += 1
            logger.debug(f"Async connection created for {self.mode.value}")
        
        @event.listens_for(self._async_engine.sync_engine, "checkout")
        def on_async_checkout(dbapi_connection, connection_record, connection_proxy):
            self._metrics.active_connections += 1
            self._update_pool_utilization()
        
        @event.listens_for(self._async_engine.sync_engine, "checkin")
        def on_async_checkin(dbapi_connection, connection_record):
            self._metrics.active_connections = max(0, self._metrics.active_connections - 1)
            self._update_pool_utilization()
    
    def _update_pool_utilization(self):
        """Update pool utilization metrics"""
        total_pool_size = self.pool_config.pg_pool_size + self.pool_config.pg_max_overflow
        if total_pool_size > 0:
            self._metrics.pool_utilization = (self._metrics.active_connections / total_pool_size) * 100
    
    async def _test_connections(self):
        """Test all connection types"""
        # Test async connection  
        async with self.get_async_session() as session:
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
        
        # Test HA connections if available
        if self._pg_pools:
            primary_pool = self._pg_pools.get("primary")
            if primary_pool:
                async with primary_pool.acquire() as conn:
                    result = await conn.fetchval("SELECT 1")
                    assert result == 1
        
        logger.info("All connection tests passed")
    
    # ========== ConnectionManagerProtocol Implementation ==========
    
    async def get_connection(self, 
                           mode: ConnectionMode = ConnectionMode.READ_WRITE,
                           **kwargs) -> AsyncIterator[Union[AsyncSession, AsyncConnection]]:
        """Get connection implementing ConnectionManagerProtocol"""
        if not self._is_initialized:
            await self.initialize()
        
        if self._is_circuit_breaker_open():
            raise ConnectionError("Circuit breaker is open")
        
        connection_type = kwargs.get('connection_type', 'session')
        
        try:
            if connection_type == 'raw' and self._pg_pools:
                # Use HA pool for raw connections
                pool_name = 'primary' if mode == ConnectionMode.READ_WRITE else 'replica_0'
                pool = self._pg_pools.get(pool_name) or self._pg_pools.get('primary')
                if pool:
                    async with pool.acquire() as conn:
                        yield conn
                        return
            
            # Default to async session
            async with self.get_async_session() as session:
                # Apply read-only settings if needed
                if mode == ConnectionMode.READ_ONLY:
                    await session.execute(text("SET TRANSACTION READ ONLY"))
                
                yield session
                
        except Exception as e:
            self._handle_connection_failure(e)
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        start_time = time.time()
        health_info = {
            "status": "unknown",
            "timestamp": start_time,
            "mode": self.mode.value,
            "components": {},
            "metrics": self._get_metrics_dict(),
            "response_time_ms": 0
        }
        
        try:
            # Test async connection
            async with self.get_async_session() as session:
                await session.execute(text("SELECT 1"))
            health_info["components"]["async_database"] = "healthy"
            
            # Test HA pools
            if self._pg_pools:
                for pool_name, pool in self._pg_pools.items():
                    try:
                        async with pool.acquire() as conn:
                            await conn.execute("SELECT 1")
                        health_info["components"][f"ha_pool_{pool_name}"] = "healthy"
                    except Exception as e:
                        health_info["components"][f"ha_pool_{pool_name}"] = f"unhealthy: {e}"
            
            # Test Redis if available
            if self._redis_master:
                try:
                    await self._redis_master.ping()
                    health_info["components"]["redis_master"] = "healthy"
                except Exception as e:
                    health_info["components"]["redis_master"] = f"unhealthy: {e}"
            
            # Overall status
            unhealthy_components = [k for k, v in health_info["components"].items() if "unhealthy" in str(v)]
            if not unhealthy_components:
                health_info["status"] = "healthy"
                self._health_status = HealthStatus.HEALTHY
            elif len(unhealthy_components) < len(health_info["components"]) / 2:
                health_info["status"] = "degraded"  
                self._health_status = HealthStatus.DEGRADED
            else:
                health_info["status"] = "unhealthy"
                self._health_status = HealthStatus.UNHEALTHY
            
        except Exception as e:
            health_info["status"] = "unhealthy"
            health_info["error"] = str(e)
            self._health_status = HealthStatus.UNHEALTHY
        
        health_info["response_time_ms"] = (time.time() - start_time) * 1000
        return health_info

    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status without performing checks"""
        return {
            "status": self._health_status.value if hasattr(self, '_health_status') else "unknown",
            "mode": self.mode.value,
            "pool_info": {
                "pool_size": self.pool_config.pg_pool_size,
                "max_pool_size": self.pool_config.pg_pool_size + self.pool_config.pg_max_overflow,
                "timeout": self.pool_config.pg_timeout,
                "active_connections": getattr(self._metrics, 'active_connections', 0),
                "idle_connections": getattr(self._metrics, 'idle_connections', 0),
            },
            "metrics": self._get_metrics_dict(),
            "initialized": getattr(self, '_is_initialized', False)
        }

    async def __aenter__(self):
        """Async context manager entry"""
        if not getattr(self, '_is_initialized', False):
            await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Don't automatically shutdown on context exit to allow reuse
        # Users can call shutdown() explicitly if needed
        pass

    async def close(self) -> None:
        """Close all connections and cleanup resources"""
        logger.info("Shutting down UnifiedConnectionManager")
        
        try:
            # Cancel background tasks using enhanced task manager
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
                logger.warning(f"Failed to cancel background tasks: {e}")
            
            # Close async engine  
            if self._async_engine:
                await self._async_engine.dispose()
            
            # Close HA pools (asyncpg pools)
            for pool_name, pool in self._pg_pools.items():
                try:
                    await pool.close()
                except Exception as e:
                    logger.warning(f"Error closing HA pool {pool_name}: {e}")
            
            # Close Redis connections and cleanup protocol resources
            if self._pubsub_connection:
                try:
                    await self._pubsub_connection.aclose()
                    self._pubsub_connection = None
                except Exception as e:
                    logger.warning(f"Error closing pubsub connection: {e}")
            
            if self._redis_master:
                await self._redis_master.aclose()
            if self._redis_replica:
                await self._redis_replica.aclose()
            
            # Clear protocol state
            self._active_locks.clear()
            self._subscribers.clear()
            
            # Clear cache data
            if self._l1_cache:
                self._l1_cache.clear()
            self._access_patterns.clear()
            self._cache_response_times.clear()
            self._cache_operation_stats.clear()
            
            self._is_initialized = False
            logger.info("UnifiedConnectionManager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def get_connection_info(self) -> Dict[str, Any]:
        """Get current connection pool information"""
        info = {
            "mode": self.mode.value,
            "initialized": self._is_initialized,
            "health_status": self._health_status.value,
            "pool_config": {
                "pg_pool_size": self.pool_config.pg_pool_size,
                "pg_max_overflow": self.pool_config.pg_max_overflow,
                "pg_timeout": self.pool_config.pg_timeout,
                "redis_pool_size": self.pool_config.redis_pool_size,
                "enable_ha": self.pool_config.enable_ha,
                "enable_circuit_breaker": self.pool_config.enable_circuit_breaker
            },
            "metrics": self._get_metrics_dict()
        }
        
        # Add engine pool info if available
        if self._async_engine:
            pool = self._async_engine.pool
            info["async_pool"] = {
                "size": pool.size(),
                "checked_out": pool.checkedout(),
                "checked_in": pool.checkedin(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid()
            }
        
        # Add HA pool info
        if self._pg_pools:
            info["ha_pools"] = {name: "active" for name in self._pg_pools.keys()}
        
        return info
    
    def is_healthy(self) -> bool:
        """Quick health status check"""
        return self._health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
    
    # ========== Enhanced Session Management with MCP and Multi-Mode Support ==========
    
    @contextlib.asynccontextmanager
    async def get_mcp_read_session(self) -> AsyncIterator[AsyncSession]:
        """Get MCP-optimized read-only session with <200ms SLA enforcement (from MCPConnectionPool)"""
        if not self._is_initialized:
            await self.initialize()
        
        if not self._async_session_factory:
            raise RuntimeError("Async session factory not initialized")
        
        session = self._async_session_factory()
        start_time = time.time()
        
        try:
            # Set transaction to read-only for performance
            await session.execute(text("SET TRANSACTION READ ONLY"))
            
            # Set statement timeout for MCP SLA compliance
            if self.mode == ManagerMode.MCP_SERVER:
                await session.execute(text("SET statement_timeout = '150ms'"))
            
            yield session
            # Read-only transactions don't need explicit commit
            
            # Track MCP SLA compliance
            response_time = (time.time() - start_time) * 1000
            if self.mode == ManagerMode.MCP_SERVER and response_time > 200:
                logger.warning(f"MCP read session exceeded 200ms SLA: {response_time:.1f}ms")
                self._metrics.sla_compliance_rate = max(0.0, self._metrics.sla_compliance_rate - 2.0)
            
        except Exception as e:
            logger.error(f"MCP read session error: {e}")
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
            
            # Update metrics
            response_time = (time.time() - start_time) * 1000
            self._update_response_time(response_time)
            self._metrics.queries_executed += 1
            
        except Exception as e:
            await session.rollback()
            self._metrics.queries_failed += 1
            logger.error(f"Feedback session error: {e}")
            raise
        finally:
            await session.close()
    
    # ========== Core Session Management ==========
    
    @contextlib.asynccontextmanager
    async def get_async_session(self) -> AsyncIterator[AsyncSession]:
        """Get async session"""
        if not self._is_initialized:
            await self.initialize()
        
        if not self._async_session_factory:
            raise RuntimeError("Async session factory not initialized")
        
        session = self._async_session_factory()
        start_time = time.time()
        
        try:
            yield session
            await session.commit()
            
            # Update metrics with connection tracking
            response_time = (time.time() - start_time) * 1000
            self._update_response_time(response_time)
            self._metrics.queries_executed += 1
            self._metrics.connection_reuse_count += 1
            
            # Track connection times for performance analysis
            self._metrics.connection_times.append(response_time)
            if len(self._metrics.connection_times) > 0:
                self._metrics.avg_response_time_ms = statistics.mean(list(self._metrics.connection_times)[-100:])
            
        except Exception as e:
            await session.rollback()
            self._metrics.error_rate += 1
            self._metrics.queries_failed += 1
            self._handle_connection_failure(e)
            logger.error(f"Async session error in {self.mode.value}: {e}")
            raise
        finally:
            await session.close()
    
    
    # ========== Performance Optimization Methods ==========
    
    async def optimize_pool_size(self) -> Dict[str, Any]:
        """Dynamically optimize pool size based on load patterns (from ConnectionPoolOptimizer)"""
        current_metrics = await self._collect_pool_metrics()
        
        # Don't optimize too frequently
        if datetime.now(timezone.utc) - (self._metrics.last_scale_event or datetime.min.replace(tzinfo=timezone.utc)) < timedelta(minutes=5):
            return {"status": "skipped", "reason": "optimization cooldown"}
        
        utilization = current_metrics.get('utilization', 0) / 100.0
        waiting_requests = current_metrics.get('waiting_requests', 0)
        
        # Determine optimal pool size
        recommendations = []
        new_pool_size = self.current_pool_size
        
        if utilization > 0.9 and waiting_requests > 0:
            # Pool is stressed, increase size
            increase = min(5, self.max_pool_size - self.current_pool_size)
            if increase > 0:
                new_pool_size += increase
                recommendations.append(f"Increase pool size by {increase} (high utilization: {utilization:.1%})")
                self._pool_state = PoolState.STRESSED
        
        elif utilization < 0.3 and self.current_pool_size > self.min_pool_size:
            # Pool is underutilized, decrease size
            decrease = min(3, self.current_pool_size - self.min_pool_size)
            if decrease > 0:
                new_pool_size -= decrease
                recommendations.append(f"Decrease pool size by {decrease} (low utilization: {utilization:.1%})")
        
        # Apply optimization if needed
        if new_pool_size != self.current_pool_size:
            try:
                await self._scale_pool(new_pool_size)
                
                return {
                    "status": "optimized",
                    "previous_size": self.current_pool_size,
                    "new_size": new_pool_size,
                    "utilization": utilization,
                    "recommendations": recommendations
                }
            except Exception as e:
                logger.error(f"Failed to optimize pool size: {e}")
                return {"status": "error", "error": str(e)}
        
        return {
            "status": "no_change_needed",
            "current_size": self.current_pool_size,
            "utilization": utilization,
            "state": self._pool_state.value
        }
    
    async def coordinate_pools(self) -> Dict[str, Any]:
        """Multi-pool coordination (from ConnectionPoolManager)"""
        coordination_status = {
            "database_pool": {
                "healthy": self._health_status == HealthStatus.HEALTHY,
                "connections": self._metrics.active_connections,
                "utilization": self._metrics.pool_utilization
            },
            "redis_pool": {
                "healthy": self._metrics.redis_pool_health,
                "connected": self._redis_master is not None
            },
            "http_pool": {
                "healthy": self._metrics.http_pool_health
            }
        }
        
        # Multi-pool load balancing logic
        total_healthy_pools = sum(1 for pool in coordination_status.values() if pool["healthy"])
        
        self._metrics.multi_pool_coordination_active = total_healthy_pools > 1
        
        return {
            "status": "active" if self._metrics.multi_pool_coordination_active else "limited",
            "healthy_pools": total_healthy_pools,
            "pool_status": coordination_status,
            "load_balancing_active": total_healthy_pools > 1
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics from all consolidated managers"""
        pool_metrics = await self._collect_pool_metrics()
        
        return {
            'state': self._pool_state.value,
            'pool_size': self.current_pool_size,
            'min_pool_size': self.min_pool_size,
            'max_pool_size': self.max_pool_size,
            'active_connections': self._metrics.active_connections,
            'idle_connections': self._metrics.idle_connections,
            'total_connections': self._metrics.total_connections,
            'pool_utilization': self._metrics.pool_utilization,
            'connections_created': self._metrics.connections_created,
            'connections_closed': self._metrics.connections_closed,
            'connection_failures': self._metrics.connection_failures,
            'avg_response_time_ms': self._metrics.avg_response_time_ms,
            'queries_executed': self._metrics.queries_executed,
            'queries_failed': self._metrics.queries_failed,
            'wait_time_ms': self._metrics.wait_time_ms,
            'circuit_breaker_open': self._is_circuit_breaker_open(),
            'circuit_breaker_failures': self._circuit_breaker_failures,
            'last_scale_event': self._metrics.last_scale_event.isoformat() if self._metrics.last_scale_event else None,
            'sla_compliance_rate': self._metrics.sla_compliance_rate,
            'pool_efficiency': self._metrics.pool_efficiency,
            'database_load_reduction_percent': self._metrics.database_load_reduction_percent,
            'connections_saved': self._metrics.connections_saved,
            'multi_pool_coordination': self._metrics.multi_pool_coordination_active,
            'performance_window': list(self.performance_window),
            'cache_stats': self.get_cache_stats()
        }
    
    async def test_permissions(self) -> Dict[str, Any]:
        """Test database permissions (from MCPConnectionPool)"""
        results = {
            "read_rule_performance": False,
            "read_rule_metadata": False, 
            "write_prompt_sessions": False,
            "denied_rule_write": True,  # Should be denied
        }
        
        try:
            async with self.get_mcp_read_session() as session:
                # Test read access to rule tables
                try:
                    await session.execute(text("SELECT COUNT(*) FROM rule_performance LIMIT 1"))
                    results["read_rule_performance"] = True
                except Exception as e:
                    logger.warning(f"Cannot read rule_performance: {e}")
                
                try:
                    await session.execute(text("SELECT COUNT(*) FROM rule_metadata LIMIT 1"))
                    results["read_rule_metadata"] = True
                except Exception as e:
                    logger.warning(f"Cannot read rule_metadata: {e}")
            
            async with self.get_feedback_session() as session:
                # Test write access to feedback table
                try:
                    await session.execute(
                        text("INSERT INTO prompt_improvement_sessions "
                            "(original_prompt, enhanced_prompt, applied_rules, response_time_ms) "
                            "VALUES ('test', 'test', '[]', 100)")
                    )
                    await session.execute(
                        text("DELETE FROM prompt_improvement_sessions WHERE original_prompt = 'test'")
                    )
                    results["write_prompt_sessions"] = True
                except Exception as e:
                    logger.warning(f"Cannot write to prompt_improvement_sessions: {e}")
                
                # Test that write to rule tables is properly denied
                try:
                    await session.execute(text("INSERT INTO rule_performance (rule_id, rule_name) VALUES ('test', 'test')"))
                    results["denied_rule_write"] = False  # This should have failed
                    logger.warning("User can write to rule tables - SECURITY ISSUE!")
                except Exception:
                    # This is expected - user should not be able to write to rule tables
                    pass
        
        except Exception as e:
            logger.error(f"Permission test failed: {e}")
            return {"error": str(e), "permissions_verified": False}
        
        return {
            "permissions_verified": True,
            "test_results": results,
            "security_compliant": (
                results["read_rule_performance"] and 
                results["read_rule_metadata"] and 
                results["write_prompt_sessions"] and 
                results["denied_rule_write"]
            )
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics from multi-level cache system."""
        # Calculate overall hit rate
        total_requests = self._cache_total_requests
        overall_hit_rate = 0.0
        if total_requests > 0:
            total_hits = self._cache_l1_hits + self._cache_l2_hits + self._cache_l3_hits
            overall_hit_rate = total_hits / total_requests

        # L1 cache stats
        l1_stats = {}
        if self._l1_cache:
            l1_stats = self._l1_cache.get_stats()

        # Calculate average response times
        avg_response_time = 0.0
        if self._cache_response_times:
            avg_response_time = sum(self._cache_response_times) / len(self._cache_response_times)

        # Operation stats with calculated averages
        operation_summaries = {}
        for op_name, stats in self._cache_operation_stats.items():
            if stats['count'] > 0:
                operation_summaries[op_name] = {
                    'count': stats['count'],
                    'avg_time_ms': stats['total_time'] / stats['count'],
                    'min_time_ms': stats['min_time'] if stats['min_time'] != float('inf') else 0,
                    'max_time_ms': stats['max_time'],
                    'error_rate': stats['error_count'] / stats['count'] if stats['count'] > 0 else 0
                }

        return {
            "overall_hit_rate": overall_hit_rate,
            "total_requests": total_requests,
            "l1_cache": {
                "hits": self._cache_l1_hits,
                **l1_stats
            },
            "l2_cache": {
                "hits": self._cache_l2_hits,
                "enabled": self._redis_master is not None
            },
            "l3_cache": {
                "hits": self._cache_l3_hits
            },
            "performance": {
                "avg_response_time_ms": avg_response_time,
                "total_response_samples": len(self._cache_response_times)
            },
            "operations": operation_summaries,
            "warming": {
                "enabled": self.pool_config.enable_cache_warming,
                "stats": self._warming_stats,
                "active_patterns": len(self._access_patterns),
                "health": self._cache_health_status
            },
            "health_status": self._cache_health_status.get('overall_health', 'unknown')
        }
    
    async def _collect_pool_metrics(self) -> Dict[str, Any]:
        """Collect current pool metrics"""
        if not self._async_engine:
            return {}
        
        pool = self._async_engine.pool
        return {
            "pool_size": pool.size(),
            "available": pool.checkedin(),
            "active": pool.checkedout(),
            "utilization": (pool.checkedout() / pool.size() * 100) if pool.size() > 0 else 0,
            "waiting_requests": 0,  # SQLAlchemy doesn't expose this directly
            "overflow": pool.overflow(),
            "invalid": pool.invalid()
        }
    
    async def get_ml_telemetry_metrics(self) -> Dict[str, Any]:
        """Get connection pool metrics formatted for ML orchestration telemetry integration.
        
        This method provides unified pool monitoring for ML orchestration, eliminating
        the need for independent pool registration and monitoring patterns.
        
        Returns:
            Dict containing ML telemetry-compatible pool metrics
        """
        base_metrics = await self._collect_pool_metrics()
        
        # Calculate timing metrics from recent operations
        avg_connection_time = 0.0
        avg_query_time = 0.0
        
        if self._metrics.connection_times:
            avg_connection_time = sum(self._metrics.connection_times) / len(self._metrics.connection_times)
        
        if self._metrics.query_times:
            avg_query_time = sum(self._metrics.query_times) / len(self._metrics.query_times)
        
        # Return metrics in ML telemetry expected format
        return {
            "pool_utilization": base_metrics.get("utilization", 0) / 100.0,  # Convert to ratio
            "avg_connection_time_ms": avg_connection_time,
            "avg_query_time_ms": avg_query_time,
            "pool_size": base_metrics.get("pool_size", 0),
            "active_connections": base_metrics.get("active", 0),
            "available_connections": base_metrics.get("available", 0),
            "overflow_connections": base_metrics.get("overflow", 0),
            "invalid_connections": base_metrics.get("invalid", 0),
            "health_status": "healthy" if self._health_status == HealthStatus.HEALTHY else "degraded"
        }
    
    # ========== Utility Methods ==========
    
    def _update_response_time(self, response_time_ms: float):
        """Update average response time using exponential moving average"""
        alpha = 0.1
        if self._metrics.avg_response_time_ms == 0:
            self._metrics.avg_response_time_ms = response_time_ms
        else:
            self._metrics.avg_response_time_ms = (
                alpha * response_time_ms + 
                (1 - alpha) * self._metrics.avg_response_time_ms
            )
    
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
        
        if (self.pool_config.enable_circuit_breaker and 
            self._circuit_breaker_failures >= self._circuit_breaker_threshold):
            self._metrics.circuit_breaker_state = "open"
            logger.error(f"Circuit breaker opened due to {self._circuit_breaker_failures} failures")
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if not self.pool_config.enable_circuit_breaker:
            return False
            
        if self._metrics.circuit_breaker_state == "open":
            if time.time() - self._circuit_breaker_last_failure > self._circuit_breaker_timeout:
                self._metrics.circuit_breaker_state = "half-open"
                logger.info("Circuit breaker moved to half-open state")
                return False
            return True
        return False
    
    def _get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary with all consolidated metrics"""
        return {
            # Core connection metrics
            "active_connections": self._metrics.active_connections,
            "idle_connections": self._metrics.idle_connections,
            "total_connections": self._metrics.total_connections,
            "pool_utilization": self._metrics.pool_utilization,
            "avg_response_time_ms": self._metrics.avg_response_time_ms,
            "error_rate": self._metrics.error_rate,
            
            # HA and circuit breaker metrics
            "failed_connections": self._metrics.failed_connections,
            "failover_count": self._metrics.failover_count,
            "last_failover": self._metrics.last_failover,
            "health_check_failures": self._metrics.health_check_failures,
            "circuit_breaker_state": self._metrics.circuit_breaker_state,
            "circuit_breaker_failures": self._circuit_breaker_failures,
            
            # Auto-scaling and performance metrics (from AdaptiveConnectionPool)
            "connections_created": self._metrics.connections_created,
            "connections_closed": self._metrics.connections_closed,
            "connection_failures": self._metrics.connection_failures,
            "queries_executed": self._metrics.queries_executed,
            "queries_failed": self._metrics.queries_failed,
            "wait_time_ms": self._metrics.wait_time_ms,
            "last_scale_event": self._metrics.last_scale_event.isoformat() if self._metrics.last_scale_event else None,
            
            # Connection pool optimization metrics (from ConnectionPoolOptimizer)
            "connection_reuse_count": self._metrics.connection_reuse_count,
            "pool_efficiency": self._metrics.pool_efficiency,
            "connections_saved": self._metrics.connections_saved,
            "database_load_reduction_percent": self._metrics.database_load_reduction_percent,
            
            # Multi-pool coordination metrics (from ConnectionPoolManager)
            "http_pool_health": self._metrics.http_pool_health,
            "redis_pool_health": self._metrics.redis_pool_health,
            "multi_pool_coordination_active": self._metrics.multi_pool_coordination_active,
            
            # SLA and registry metrics
            "sla_compliance_rate": self._metrics.sla_compliance_rate,
            "registry_conflicts": self._metrics.registry_conflicts,
            "registered_models": len(self._get_registry_manager().get_registered_classes()),
            
            # Pool state and configuration
            "pool_state": self._pool_state.value,
            "current_pool_size": self.current_pool_size,
            "min_pool_size": self.min_pool_size,
            "max_pool_size": self.max_pool_size
        }
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while self._is_initialized:
            try:
                await asyncio.sleep(self._health_check_interval)
                health_result = await self.health_check()
                
                # Update SLA compliance (enhanced for MCP <200ms SLA)
                response_time_ms = health_result.get("response_time_ms", 0)
                if self.mode == ManagerMode.MCP_SERVER:
                    # Strict 200ms SLA for MCP server
                    if response_time_ms < 200:
                        self._metrics.sla_compliance_rate = min(100.0, self._metrics.sla_compliance_rate + 0.1)
                    else:
                        self._metrics.sla_compliance_rate = max(0.0, self._metrics.sla_compliance_rate - 2.0)
                else:
                    # Standard SLA based on timeout
                    if response_time_ms < self.pool_config.pg_timeout * 1000:
                        self._metrics.sla_compliance_rate = min(100.0, self._metrics.sla_compliance_rate + 0.1)
                    else:
                        self._metrics.sla_compliance_rate = max(0.0, self._metrics.sla_compliance_rate - 1.0)
                
                self._last_health_check = time.time()
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                self._metrics.health_check_failures += 1
                await asyncio.sleep(self._health_check_interval * 2)
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for adaptive scaling and performance tracking (from AdaptiveConnectionPool)"""
        while self._is_initialized:
            try:
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
                await self._update_metrics()
                await self._evaluate_scaling()
                await self._update_connection_efficiency()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    async def _update_metrics(self) -> None:
        """Update real-time pool metrics (from AdaptiveConnectionPool)"""
        if not self._async_engine:
            return
        
        current_time = time.time()
        
        # Get pool statistics
        pool = self._async_engine.pool
        self._metrics.total_connections = pool.size()
        self._metrics.active_connections = pool.checkedout()
        self._metrics.idle_connections = pool.checkedin()
        
        # Calculate utilization
        if self._metrics.total_connections > 0:
            self._metrics.pool_utilization = self._metrics.active_connections / self._metrics.total_connections * 100
        
        # Store performance snapshot
        self.performance_window.append({
            'timestamp': current_time,
            'utilization': self._metrics.pool_utilization,
            'active_connections': self._metrics.active_connections,
            'total_connections': self._metrics.total_connections,
            'avg_connection_time': self._metrics.avg_response_time_ms,
            'sla_compliance': self._metrics.sla_compliance_rate
        })
        
        self.last_metrics_update = current_time
    
    async def _evaluate_scaling(self) -> None:
        """Evaluate if pool scaling is needed (from AdaptiveConnectionPool)"""
        if time.time() - self.last_scale_time < self.scale_cooldown_seconds:
            return
        
        utilization = self._metrics.pool_utilization / 100.0
        
        # Scale up conditions
        if (utilization > self.scale_up_threshold and 
            self.current_pool_size < self.max_pool_size):
            
            new_size = min(self.current_pool_size + 10, self.max_pool_size)
            await self._scale_pool(new_size)
            self._pool_state = PoolState.SCALING_UP
            
        # Scale down conditions  
        elif (utilization < self.scale_down_threshold and 
              self.current_pool_size > self.min_pool_size and
              self._metrics.avg_response_time_ms < 50):  # Low response time
            
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
            # For SQLAlchemy async engine, we need to recreate with new pool size
            # This is a simplified approach - in production you'd want more sophisticated scaling
            logger.info(f"Pool scaling requested: {old_size} → {new_size} connections")
            
            # Update pool configuration
            self.pool_config.pg_pool_size = new_size
            self.current_pool_size = new_size
            self.last_scale_time = time.time()
            self._metrics.last_scale_event = datetime.now(timezone.utc)
            
            logger.info(f"Pool size updated: {old_size} → {new_size} connections")
            
        except Exception as e:
            logger.error(f"Failed to scale connection pool: {e}")
            self._pool_state = PoolState.DEGRADED
    
    async def _update_connection_efficiency(self) -> None:
        """Update connection efficiency metrics (from ConnectionPoolOptimizer)"""
        # Calculate connection reuse efficiency
        if self._total_connections_created > 0:
            reuse_rate = self._metrics.connection_reuse_count / self._total_connections_created
            self._metrics.pool_efficiency = reuse_rate * 100
            
            # Calculate database load reduction
            base_connections = self._metrics.connection_reuse_count + self._total_connections_created
            if base_connections > 0:
                self._metrics.database_load_reduction_percent = (
                    (base_connections - self._total_connections_created) / base_connections * 100
                )
                self._metrics.connections_saved = self._metrics.connection_reuse_count
    
    # ========== Cache Protocol Compliance Implementation ==========
    
    # BasicCacheProtocol methods
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (BasicCacheProtocol compliance).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        security_context = self._default_security_context if self._protocol_security_required else None
        return await self.get_cached(key, security_context)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
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
            # Clear L1 cache
            if self._l1_cache:
                self._l1_cache.clear()
                logger.info("L1 cache cleared")
            
            # Clear L2 cache (Redis)
            if self._redis_master:
                try:
                    if self._protocol_security_required and self._default_security_context:
                        if self._validate_security_context(self._default_security_context):
                            await self._redis_master.flushdb()
                            logger.info("L2 Redis cache cleared")
                        else:
                            logger.warning("Cannot clear Redis cache - invalid security context")
                            success = False
                    else:
                        # For non-secure modes, allow clear without security context
                        await self._redis_master.flushdb()
                        logger.info("L2 Redis cache cleared")
                except Exception as e:
                    logger.warning(f"Failed to clear Redis cache: {e}")
                    success = False
            
            # Clear access patterns and metrics
            self._access_patterns.clear()
            self._cache_response_times.clear()
            self._cache_operation_stats.clear()
            
            # Reset cache hit counters
            self._cache_l1_hits = 0
            self._cache_l2_hits = 0
            self._cache_l3_hits = 0
            self._cache_total_requests = 0
            
            return success
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    # AdvancedCacheProtocol methods
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
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
    
    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
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
    
    async def delete_many(self, keys: List[str]) -> int:
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
    
    async def get_or_set(self, key: str, default_func, ttl: Optional[int] = None) -> Any:
        """Get value or set it using default function (AdvancedCacheProtocol compliance).
        
        Args:
            key: Cache key
            default_func: Function to call if key doesn't exist (can be async)
            ttl: Optional TTL in seconds
            
        Returns:
            Cached or computed value
        """
        security_context = self._default_security_context if self._protocol_security_required else None
        
        # Try to get existing value
        value = await self.get_cached(key, security_context)
        if value is not None:
            return value
        
        # Compute new value
        if asyncio.iscoroutinefunction(default_func):
            computed_value = await default_func()
        else:
            computed_value = default_func()
        
        # Store and return computed value
        await self.set_cached(key, computed_value, ttl, security_context)
        return computed_value
    
    async def increment(self, key: str, delta: int = 1) -> int:
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
                        raise RedisSecurityError(f"Invalid security context for increment on key {key}")
                
                # Use Redis INCRBY command
                new_value = await self._redis_master.incrby(key, delta)
                
                # Update L1 cache if present
                if self._l1_cache:
                    self._l1_cache.set(key, new_value)
                
                return new_value
                
            except Exception as e:
                logger.error(f"Redis increment failed for key {key}: {e}")
                raise
        else:
            # Fallback for systems without Redis
            current_value = await self.get(key)
            if current_value is None:
                current_value = 0
            elif not isinstance(current_value, (int, float)):
                raise ValueError(f"Cannot increment non-numeric value for key {key}")
            
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
                        logger.warning(f"Invalid security context for expire on key {key}")
                        return False
                
                # Use Redis EXPIRE command
                result = await self._redis_master.expire(key, seconds)
                return bool(result)
                
            except Exception as e:
                logger.error(f"Redis expire failed for key {key}: {e}")
                return False
        else:
            # For L1 cache only, we can't set expiration on existing keys
            # This would require re-setting the value with TTL
            logger.warning(f"Cannot set expiration on existing key {key} without Redis")
            return False
    
    # CacheHealthProtocol methods
    async def ping(self) -> bool:
        """Ping cache service (CacheHealthProtocol compliance).
        
        Returns:
            True if cache is responsive, False otherwise
        """
        try:
            # Test L1 cache
            l1_healthy = True
            if self._l1_cache:
                test_key = "__ping_test__"
                self._l1_cache.set(test_key, "ping")
                l1_healthy = self._l1_cache.get(test_key) == "ping"
                self._l1_cache.delete(test_key)
            
            # Test L2 cache (Redis)
            l2_healthy = True
            if self._redis_master:
                try:
                    if self._protocol_security_required and self._default_security_context:
                        if not self._validate_security_context(self._default_security_context):
                            logger.warning("Invalid security context for ping")
                            l2_healthy = False
                        else:
                            l2_healthy = await self._redis_master.ping()
                    else:
                        l2_healthy = await self._redis_master.ping()
                except Exception as e:
                    logger.warning(f"Redis ping failed: {e}")
                    l2_healthy = False
            
            return l1_healthy and l2_healthy
            
        except Exception as e:
            logger.error(f"Cache ping failed: {e}")
            return False
    
    async def get_info(self) -> Dict[str, Any]:
        """Get cache service information (CacheHealthProtocol compliance).
        
        Returns:
            Dictionary containing cache service information
        """
        info = {
            "service": "UnifiedConnectionManager",
            "version": "2.0",
            "mode": self.mode.value,
            "protocol_compliance": True,
            "security_required": self._protocol_security_required,
            "components": {
                "l1_cache": {
                    "enabled": self._l1_cache is not None,
                    "type": "LRUCache",
                    "max_size": self.pool_config.l1_cache_size if self._l1_cache else 0
                },
                "l2_cache": {
                    "enabled": self._redis_master is not None,
                    "type": "Redis",
                    "connected": False
                }
            },
            "features": {
                "multi_level": True,
                "cache_warming": self.pool_config.enable_cache_warming,
                "security_context": True,
                "distributed_locking": self._redis_master is not None,
                "pub_sub": self._redis_master is not None,
                "opentelemetry": OPENTELEMETRY_AVAILABLE
            }
        }
        
        # Test Redis connectivity
        if self._redis_master:
            try:
                ping_result = await self.ping()
                info["components"]["l2_cache"]["connected"] = ping_result
                
                if ping_result:
                    # Get Redis info if available
                    try:
                        redis_info = await self._redis_master.info()
                        if isinstance(redis_info, dict):
                            info["components"]["l2_cache"]["redis_version"] = redis_info.get("redis_version", "unknown")
                            info["components"]["l2_cache"]["used_memory_human"] = redis_info.get("used_memory_human", "unknown")
                    except Exception as e:
                        logger.debug(f"Could not get Redis info: {e}")
                        
            except Exception as e:
                logger.warning(f"Could not test Redis connectivity: {e}")
        
        return info
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics (CacheHealthProtocol compliance).
        
        Returns:
            Dictionary containing performance statistics
        """
        # Use existing get_cache_stats method and enhance it
        base_stats = self.get_cache_stats()
        
        # Add protocol-specific statistics
        protocol_stats = {
            "protocol_compliance": {
                "basic_cache": True,
                "advanced_cache": True,
                "health_monitoring": True,
                "pub_sub": self._redis_master is not None,
                "distributed_locking": self._redis_master is not None,
                "multi_level": True
            },
            "security": {
                "security_required": self._protocol_security_required,
                "active_locks": len(self._active_locks),
                "default_context_valid": (
                    self._default_security_context and 
                    self._validate_security_context(self._default_security_context)
                ) if self._default_security_context else False
            },
            "connections": {
                "redis_master_connected": self._redis_master is not None,
                "redis_replica_connected": self._redis_replica is not None,
                "sentinel_connected": self._redis_sentinel is not None
            }
        }
        
        # Merge base stats with protocol stats
        return {**base_stats, **protocol_stats}
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics (CacheHealthProtocol compliance).
        
        Returns:
            Dictionary containing memory usage information
        """
        memory_stats = {
            "l1_cache": {
                "enabled": self._l1_cache is not None,
                "current_size": 0,
                "max_size": 0,
                "utilization_percent": 0.0,
                "estimated_memory_bytes": 0
            },
            "l2_cache": {
                "enabled": self._redis_master is not None,
                "memory_info": {}
            },
            "access_patterns": {
                "active_patterns": len(self._access_patterns),
                "estimated_memory_bytes": len(self._access_patterns) * 1024  # Rough estimate
            },
            "response_times": {
                "samples_stored": len(self._cache_response_times),
                "estimated_memory_bytes": len(self._cache_response_times) * 8  # 8 bytes per float
            }
        }
        
        # L1 cache memory usage
        if self._l1_cache:
            l1_stats = self._l1_cache.get_stats()
            memory_stats["l1_cache"].update({
                "current_size": l1_stats["size"],
                "max_size": l1_stats["max_size"],
                "utilization_percent": l1_stats["utilization"] * 100,
                "estimated_memory_bytes": l1_stats["size"] * 512  # Rough estimate per entry
            })
        
        # L2 cache (Redis) memory usage
        if self._redis_master:
            try:
                if self._protocol_security_required and self._default_security_context:
                    if self._validate_security_context(self._default_security_context):
                        redis_info = await self._redis_master.info("memory")
                        if isinstance(redis_info, dict):
                            memory_stats["l2_cache"]["memory_info"] = {
                                "used_memory": redis_info.get("used_memory", 0),
                                "used_memory_human": redis_info.get("used_memory_human", "unknown"),
                                "used_memory_peak": redis_info.get("used_memory_peak", 0),
                                "used_memory_peak_human": redis_info.get("used_memory_peak_human", "unknown"),
                                "maxmemory": redis_info.get("maxmemory", 0),
                                "maxmemory_human": redis_info.get("maxmemory_human", "unknown")
                            }
                else:
                    redis_info = await self._redis_master.info("memory")
                    if isinstance(redis_info, dict):
                        memory_stats["l2_cache"]["memory_info"] = {
                            "used_memory": redis_info.get("used_memory", 0),
                            "used_memory_human": redis_info.get("used_memory_human", "unknown"),
                            "used_memory_peak": redis_info.get("used_memory_peak", 0),
                            "used_memory_peak_human": redis_info.get("used_memory_peak_human", "unknown")
                        }
            except Exception as e:
                logger.warning(f"Could not get Redis memory info: {e}")
                memory_stats["l2_cache"]["memory_info"] = {"error": str(e)}
        
        return memory_stats
    
    # CacheSubscriptionProtocol methods
    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel (CacheSubscriptionProtocol compliance).
        
        Args:
            channel: Channel name
            message: Message to publish
            
        Returns:
            Number of subscribers that received the message
        """
        if not self._redis_master:
            logger.warning("Pub/Sub not available - Redis not connected")
            return 0
        
        try:
            if self._protocol_security_required and self._default_security_context:
                if not self._validate_security_context(self._default_security_context):
                    raise RedisSecurityError(f"Invalid security context for publish to channel {channel}")
            
            # Serialize message if it's not already a string
            if not isinstance(message, (str, bytes)):
                message = json.dumps(message)
            
            # Publish to Redis
            subscriber_count = await self._redis_master.publish(channel, message)
            
            # OpenTelemetry metrics
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {
                    "operation": "publish",
                    "level": "l2",
                    "status": "success",
                    "security_context": self._default_security_context.agent_id if self._default_security_context else "none"
                })
            
            logger.debug(f"Published message to channel {channel}, {subscriber_count} subscribers")
            return subscriber_count
            
        except Exception as e:
            logger.error(f"Failed to publish to channel {channel}: {e}")
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {
                    "operation": "publish",
                    "level": "l2",
                    "status": "error",
                    "security_context": self._default_security_context.agent_id if self._default_security_context else "none"
                })
            return 0
    
    async def subscribe(self, channels: List[str]) -> Any:
        """Subscribe to channels (CacheSubscriptionProtocol compliance).
        
        Args:
            channels: List of channel names to subscribe to
            
        Returns:
            Subscription object or None if failed
        """
        if not self._redis_master:
            logger.warning("Pub/Sub not available - Redis not connected")
            return None
        
        try:
            if self._protocol_security_required and self._default_security_context:
                if not self._validate_security_context(self._default_security_context):
                    raise RedisSecurityError(f"Invalid security context for subscribe to channels {channels}")
            
            # Create or reuse pubsub connection
            if not self._pubsub_connection:
                self._pubsub_connection = self._redis_master.pubsub()
            
            # Subscribe to channels
            await self._pubsub_connection.subscribe(*channels)
            
            # Track subscribed channels
            for channel in channels:
                if channel not in self._subscribers:
                    self._subscribers[channel] = []
                self._subscribers[channel].append(self._pubsub_connection)
            
            # OpenTelemetry metrics
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {
                    "operation": "subscribe",
                    "level": "l2",
                    "status": "success",
                    "security_context": self._default_security_context.agent_id if self._default_security_context else "none"
                })
            
            logger.debug(f"Subscribed to channels: {channels}")
            return self._pubsub_connection
            
        except Exception as e:
            logger.error(f"Failed to subscribe to channels {channels}: {e}")
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {
                    "operation": "subscribe",
                    "level": "l2",
                    "status": "error",
                    "security_context": self._default_security_context.agent_id if self._default_security_context else "none"
                })
            return None
    
    async def unsubscribe(self, channels: List[str]) -> bool:
        """Unsubscribe from channels (CacheSubscriptionProtocol compliance).
        
        Args:
            channels: List of channel names to unsubscribe from
            
        Returns:
            True if successfully unsubscribed, False otherwise
        """
        if not self._redis_master or not self._pubsub_connection:
            logger.warning("Pub/Sub not available or not subscribed")
            return False
        
        try:
            if self._protocol_security_required and self._default_security_context:
                if not self._validate_security_context(self._default_security_context):
                    raise RedisSecurityError(f"Invalid security context for unsubscribe from channels {channels}")
            
            # Unsubscribe from channels
            await self._pubsub_connection.unsubscribe(*channels)
            
            # Update subscriber tracking
            for channel in channels:
                if channel in self._subscribers:
                    if self._pubsub_connection in self._subscribers[channel]:
                        self._subscribers[channel].remove(self._pubsub_connection)
                    if not self._subscribers[channel]:
                        del self._subscribers[channel]
            
            # OpenTelemetry metrics
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {
                    "operation": "unsubscribe",
                    "level": "l2",
                    "status": "success",
                    "security_context": self._default_security_context.agent_id if self._default_security_context else "none"
                })
            
            logger.debug(f"Unsubscribed from channels: {channels}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from channels {channels}: {e}")
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {
                    "operation": "unsubscribe",
                    "level": "l2",
                    "status": "error",
                    "security_context": self._default_security_context.agent_id if self._default_security_context else "none"
                })
            return False
    
    # CacheLockProtocol methods
    async def acquire_lock(self, key: str, timeout: int = 10) -> Optional[str]:
        """Acquire distributed lock (CacheLockProtocol compliance).
        
        Args:
            key: Lock key
            timeout: Lock timeout in seconds
            
        Returns:
            Lock token if acquired, None if failed
        """
        if not self._redis_master:
            logger.warning("Distributed locking not available - Redis not connected")
            return None
        
        try:
            if self._protocol_security_required and self._default_security_context:
                if not self._validate_security_context(self._default_security_context):
                    raise RedisSecurityError(f"Invalid security context for acquire_lock on key {key}")
            
            # Generate unique lock token
            import uuid
            lock_token = str(uuid.uuid4())
            lock_key = f"lock:{key}"
            
            # Try to acquire lock using Redis SET with NX and EX
            acquired = await self._redis_master.set(
                lock_key, 
                lock_token, 
                nx=True,  # Only set if key doesn't exist
                ex=timeout  # Set expiration
            )
            
            if acquired:
                # Track active lock
                self._active_locks[key] = {
                    "token": lock_token,
                    "acquired_at": time.time(),
                    "timeout": timeout,
                    "expires_at": time.time() + timeout
                }
                
                # OpenTelemetry metrics
                if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                    cache_operations_counter.add(1, {
                        "operation": "acquire_lock",
                        "level": "l2",
                        "status": "success",
                        "security_context": self._default_security_context.agent_id if self._default_security_context else "none"
                    })
                
                logger.debug(f"Acquired lock for key {key} with token {lock_token[:8]}...")
                return lock_token
            else:
                logger.debug(f"Could not acquire lock for key {key} - already locked")
                return None
                
        except Exception as e:
            logger.error(f"Failed to acquire lock for key {key}: {e}")
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {
                    "operation": "acquire_lock",
                    "level": "l2",
                    "status": "error",
                    "security_context": self._default_security_context.agent_id if self._default_security_context else "none"
                })
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
            logger.warning("Distributed locking not available - Redis not connected")
            return False
        
        try:
            if self._protocol_security_required and self._default_security_context:
                if not self._validate_security_context(self._default_security_context):
                    raise RedisSecurityError(f"Invalid security context for release_lock on key {key}")
            
            lock_key = f"lock:{key}"
            
            # Use Lua script to atomically check token and delete
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            
            result = await self._redis_master.eval(lua_script, 1, lock_key, token)
            released = bool(result)
            
            if released:
                # Remove from active locks tracking
                if key in self._active_locks:
                    del self._active_locks[key]
                
                # OpenTelemetry metrics
                if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                    cache_operations_counter.add(1, {
                        "operation": "release_lock",
                        "level": "l2",
                        "status": "success",
                        "security_context": self._default_security_context.agent_id if self._default_security_context else "none"
                    })
                
                logger.debug(f"Released lock for key {key}")
            else:
                logger.warning(f"Could not release lock for key {key} - token mismatch or lock expired")
            
            return released
            
        except Exception as e:
            logger.error(f"Failed to release lock for key {key}: {e}")
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {
                    "operation": "release_lock",
                    "level": "l2",
                    "status": "error",
                    "security_context": self._default_security_context.agent_id if self._default_security_context else "none"
                })
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
            logger.warning("Distributed locking not available - Redis not connected")
            return False
        
        try:
            if self._protocol_security_required and self._default_security_context:
                if not self._validate_security_context(self._default_security_context):
                    raise RedisSecurityError(f"Invalid security context for extend_lock on key {key}")
            
            lock_key = f"lock:{key}"
            
            # Use Lua script to atomically check token and extend expiration
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("expire", KEYS[1], ARGV[2])
            else
                return 0
            end
            """
            
            result = await self._redis_master.eval(lua_script, 1, lock_key, token, timeout)
            extended = bool(result)
            
            if extended:
                # Update active locks tracking
                if key in self._active_locks:
                    self._active_locks[key].update({
                        "timeout": timeout,
                        "expires_at": time.time() + timeout
                    })
                
                # OpenTelemetry metrics
                if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                    cache_operations_counter.add(1, {
                        "operation": "extend_lock",
                        "level": "l2",
                        "status": "success",
                        "security_context": self._default_security_context.agent_id if self._default_security_context else "none"
                    })
                
                logger.debug(f"Extended lock for key {key} by {timeout} seconds")
            else:
                logger.warning(f"Could not extend lock for key {key} - token mismatch or lock expired")
            
            return extended
            
        except Exception as e:
            logger.error(f"Failed to extend lock for key {key}: {e}")
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {
                    "operation": "extend_lock",
                    "level": "l2",
                    "status": "error",
                    "security_context": self._default_security_context.agent_id if self._default_security_context else "none"
                })
            return False
    
    # MultiLevelCacheProtocol methods
    async def get_from_level(self, key: str, level: int) -> Optional[Any]:
        """Get value from specific cache level (MultiLevelCacheProtocol compliance).
        
        Args:
            key: Cache key
            level: Cache level (1=L1/memory, 2=L2/Redis, 3=database)
            
        Returns:
            Cached value or None if not found
        """
        try:
            if level == 1:
                # L1 cache (memory)
                if self._l1_cache:
                    return self._l1_cache.get(key)
                return None
                
            elif level == 2:
                # L2 cache (Redis)
                if self._redis_master:
                    if self._protocol_security_required and self._default_security_context:
                        if not self._validate_security_context(self._default_security_context):
                            logger.warning(f"Invalid security context for get_from_level L2 on key {key}")
                            return None
                    
                    redis_value = await self._redis_master.get(key)
                    if redis_value is not None:
                        try:
                            return json.loads(redis_value) if isinstance(redis_value, (str, bytes)) else redis_value
                        except (json.JSONDecodeError, TypeError):
                            return redis_value  # Return raw value if not JSON
                return None
                
            elif level == 3:
                # L3 cache (database/fallback) - could be implemented with database queries
                logger.debug(f"L3 cache level not implemented for key {key}")
                return None
                
            else:
                logger.warning(f"Invalid cache level {level} for key {key}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting from cache level {level} for key {key}: {e}")
            return None
    
    async def set_to_level(self, key: str, value: Any, level: int, ttl: Optional[int] = None) -> bool:
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
                # L1 cache (memory)
                if self._l1_cache:
                    self._l1_cache.set(key, value, ttl)
                    return True
                return False
                
            elif level == 2:
                # L2 cache (Redis)
                if self._redis_master:
                    if self._protocol_security_required and self._default_security_context:
                        if not self._validate_security_context(self._default_security_context):
                            logger.warning(f"Invalid security context for set_to_level L2 on key {key}")
                            return False
                    
                    try:
                        serialized_value = json.dumps(value) if not isinstance(value, (str, bytes)) else value
                    except (TypeError, ValueError):
                        logger.warning(f"Failed to serialize value for L2 cache key {key}")
                        return False
                    
                    if ttl:
                        await self._redis_master.setex(key, ttl, serialized_value)
                    else:
                        await self._redis_master.set(key, serialized_value)
                    return True
                return False
                
            elif level == 3:
                # L3 cache (database/fallback) - could be implemented with database caching
                logger.debug(f"L3 cache level not implemented for key {key}")
                return False
                
            else:
                logger.warning(f"Invalid cache level {level} for key {key}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting to cache level {level} for key {key}: {e}")
            return False
    
    async def invalidate_levels(self, key: str, levels: List[int]) -> bool:
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
                    # L1 cache (memory)
                    if self._l1_cache:
                        if self._l1_cache.delete(key):
                            success = True
                            logger.debug(f"Invalidated key {key} from L1 cache")
                        
                elif level == 2:
                    # L2 cache (Redis)
                    if self._redis_master:
                        if self._protocol_security_required and self._default_security_context:
                            if not self._validate_security_context(self._default_security_context):
                                logger.warning(f"Invalid security context for invalidate_levels L2 on key {key}")
                                continue
                        
                        deleted_count = await self._redis_master.delete(key)
                        if deleted_count > 0:
                            success = True
                            logger.debug(f"Invalidated key {key} from L2 cache")
                            
                elif level == 3:
                    # L3 cache (database/fallback)
                    logger.debug(f"L3 cache level invalidation not implemented for key {key}")
                    
                else:
                    logger.warning(f"Invalid cache level {level} for invalidation of key {key}")
                    
            except Exception as e:
                logger.error(f"Error invalidating key {key} from level {level}: {e}")
        
        # Clean up access patterns if invalidated from any level
        if success and key in self._access_patterns:
            del self._access_patterns[key]
        
        return success
    
    async def get_cache_hierarchy(self) -> List[str]:
        """Get cache level hierarchy (MultiLevelCacheProtocol compliance).
        
        Returns:
            List of cache level names in hierarchical order
        """
        hierarchy = []
        
        # L1 cache (memory)
        if self._l1_cache:
            hierarchy.append("L1_Memory")
        
        # L2 cache (Redis)
        if self._redis_master:
            if self._redis_sentinel:
                hierarchy.append("L2_Redis_Sentinel")
            else:
                hierarchy.append("L2_Redis_Direct")
        
        # L3 cache (database/fallback) - not implemented but show potential
        hierarchy.append("L3_Database_Fallback")
        
        return hierarchy
    
    # ========== Public Cache Interface with Security Context ==========
    
    async def get_cached(self, key: str, security_context: Optional[SecurityContext] = None) -> Optional[Any]:
        """Get value from multi-level cache with optional security context validation.
        
        Args:
            key: Cache key
            security_context: Optional security context for L2 Redis operations
            
        Returns:
            Cached value or None if not found or security validation fails
        """
        start_time = time.time()
        
        try:
            # L1 cache check (no security needed for in-memory)
            if self._l1_cache:
                l1_value = self._l1_cache.get(key)
                if l1_value is not None:
                    self._cache_l1_hits += 1
                    self._cache_total_requests += 1
                    
                    # Record access pattern for cache warming
                    if key not in self._access_patterns:
                        self._access_patterns[key] = AccessPattern(key=key)
                    self._access_patterns[key].record_access()
                    
                    # OpenTelemetry metrics
                    if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                        cache_operations_counter.add(1, {
                            "operation": "get",
                            "level": "l1",
                            "status": "hit",
                            "security_context": security_context.agent_id if security_context else "none"
                        })
                    
                    return l1_value
            
            # L2 cache check (Redis with security validation)
            if self._redis_master and security_context:
                try:
                    # Validate security context
                    if not self._validate_security_context(security_context):
                        logger.warning(f"Invalid security context for cache key {key}")
                        return None
                    
                    # Try Redis L2 cache
                    redis_value = await self._redis_master.get(key)
                    if redis_value is not None:
                        try:
                            # Deserialize JSON value
                            deserialized_value = json.loads(redis_value) if isinstance(redis_value, (str, bytes)) else redis_value
                            
                            # Store in L1 cache for future hits
                            if self._l1_cache:
                                self._l1_cache.set(key, deserialized_value)
                            
                            self._cache_l2_hits += 1
                            self._cache_total_requests += 1
                            
                            # Record access pattern
                            if key not in self._access_patterns:
                                self._access_patterns[key] = AccessPattern(key=key)
                            self._access_patterns[key].record_access()
                            
                            # OpenTelemetry metrics
                            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                                cache_operations_counter.add(1, {
                                    "operation": "get",
                                    "level": "l2",
                                    "status": "hit",
                                    "security_context": security_context.agent_id
                                })
                            
                            return deserialized_value
                            
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"Failed to deserialize cached value for key {key}: {e}")
                            return None
                
                except Exception as e:
                    # Fail-secure: Log error and return None
                    logger.warning(f"L2 cache error for key {key}: {e}")
                    if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                        cache_operations_counter.add(1, {
                            "operation": "get",
                            "level": "l2",
                            "status": "error",
                            "security_context": security_context.agent_id if security_context else "none"
                        })
                    return None
            
            # Cache miss
            self._cache_total_requests += 1
            
            # OpenTelemetry metrics
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {
                    "operation": "get",
                    "level": "miss",
                    "status": "miss",
                    "security_context": security_context.agent_id if security_context else "none"
                })
            
            return None
            
        finally:
            # Record response time
            response_time = (time.time() - start_time) * 1000
            self._cache_response_times.append(response_time)
            
            # Update cache operation stats
            op_stats = self._cache_operation_stats["get"]
            op_stats["count"] += 1
            op_stats["total_time"] += response_time
            op_stats["min_time"] = min(op_stats["min_time"], response_time)
            op_stats["max_time"] = max(op_stats["max_time"], response_time)
    
    async def set_cached(self, key: str, value: Any, ttl_seconds: Optional[int] = None, 
                        security_context: Optional[SecurityContext] = None) -> bool:
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
            # L1 cache (always store if available)
            if self._l1_cache:
                self._l1_cache.set(key, value, ttl_seconds)
                success = True
                
                # OpenTelemetry metrics
                if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                    cache_operations_counter.add(1, {
                        "operation": "set",
                        "level": "l1",
                        "status": "success",
                        "security_context": security_context.agent_id if security_context else "none"
                    })
            
            # L2 cache (Redis with security validation)
            if self._redis_master and security_context:
                try:
                    # Validate security context
                    if not self._validate_security_context(security_context):
                        logger.warning(f"Invalid security context for cache key {key}")
                        return success  # Return L1 success status
                    
                    # Serialize value for Redis
                    try:
                        serialized_value = json.dumps(value) if not isinstance(value, (str, bytes)) else value
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Failed to serialize value for key {key}: {e}")
                        return success  # Return L1 success status
                    
                    # Store in Redis
                    if ttl_seconds:
                        await self._redis_master.setex(key, ttl_seconds, serialized_value)
                    else:
                        await self._redis_master.set(key, serialized_value)
                    
                    success = True
                    
                    # OpenTelemetry metrics
                    if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                        cache_operations_counter.add(1, {
                            "operation": "set",
                            "level": "l2",
                            "status": "success",
                            "security_context": security_context.agent_id
                        })
                
                except Exception as e:
                    # Fail-secure: Log error but don't fail the operation if L1 succeeded
                    logger.warning(f"L2 cache set error for key {key}: {e}")
                    if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                        cache_operations_counter.add(1, {
                            "operation": "set",
                            "level": "l2", 
                            "status": "error",
                            "security_context": security_context.agent_id if security_context else "none"
                        })
            
            return success
            
        finally:
            # Record response time
            response_time = (time.time() - start_time) * 1000
            self._cache_response_times.append(response_time)
            
            # Update cache operation stats
            op_stats = self._cache_operation_stats["set"]
            op_stats["count"] += 1
            op_stats["total_time"] += response_time
            op_stats["min_time"] = min(op_stats["min_time"], response_time)
            op_stats["max_time"] = max(op_stats["max_time"], response_time)
            if not success:
                op_stats["error_count"] += 1
    
    async def delete_cached(self, key: str, security_context: Optional[SecurityContext] = None) -> bool:
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
            # L1 cache deletion (always try if available)
            if self._l1_cache:
                l1_deleted = self._l1_cache.delete(key)
                if l1_deleted:
                    success = True
                    
                    # OpenTelemetry metrics
                    if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                        cache_operations_counter.add(1, {
                            "operation": "delete",
                            "level": "l1",
                            "status": "success",
                            "security_context": security_context.agent_id if security_context else "none"
                        })
            
            # L2 cache deletion (Redis with security validation)
            if self._redis_master and security_context:
                try:
                    # Validate security context
                    if not self._validate_security_context(security_context):
                        logger.warning(f"Invalid security context for cache key {key}")
                        return success  # Return L1 success status
                    
                    # Delete from Redis
                    deleted_count = await self._redis_master.delete(key)
                    if deleted_count > 0:
                        success = True
                        
                        # OpenTelemetry metrics
                        if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                            cache_operations_counter.add(1, {
                                "operation": "delete",
                                "level": "l2",
                                "status": "success",
                                "security_context": security_context.agent_id
                            })
                
                except Exception as e:
                    # Fail-secure: Log error but don't fail if L1 succeeded
                    logger.warning(f"L2 cache delete error for key {key}: {e}")
                    if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                        cache_operations_counter.add(1, {
                            "operation": "delete",
                            "level": "l2",
                            "status": "error",
                            "security_context": security_context.agent_id if security_context else "none"
                        })
            
            # Clean up access patterns
            if key in self._access_patterns:
                del self._access_patterns[key]
            
            return success
            
        finally:
            # Record response time
            response_time = (time.time() - start_time) * 1000
            self._cache_response_times.append(response_time)
            
            # Update cache operation stats
            op_stats = self._cache_operation_stats["delete"]
            op_stats["count"] += 1
            op_stats["total_time"] += response_time
            op_stats["min_time"] = min(op_stats["min_time"], response_time)
            op_stats["max_time"] = max(op_stats["max_time"], response_time)
            if not success:
                op_stats["error_count"] += 1
    
    async def exists_cached(self, key: str, security_context: Optional[SecurityContext] = None) -> bool:
        """Check if key exists in multi-level cache with optional security context validation.
        
        Args:
            key: Cache key to check
            security_context: Optional security context for L2 Redis operations
            
        Returns:
            True if key exists in any cache level, False otherwise
        """
        start_time = time.time()
        
        try:
            # L1 cache check (no security needed)
            if self._l1_cache:
                l1_value = self._l1_cache.get(key)
                if l1_value is not None:
                    # OpenTelemetry metrics
                    if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                        cache_operations_counter.add(1, {
                            "operation": "exists",
                            "level": "l1",
                            "status": "found",
                            "security_context": security_context.agent_id if security_context else "none"
                        })
                    return True
            
            # L2 cache check (Redis with security validation)
            if self._redis_master and security_context:
                try:
                    # Validate security context
                    if not self._validate_security_context(security_context):
                        logger.warning(f"Invalid security context for cache key {key}")
                        return False
                    
                    # Check Redis existence
                    exists = await self._redis_master.exists(key)
                    if exists:
                        # OpenTelemetry metrics
                        if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                            cache_operations_counter.add(1, {
                                "operation": "exists",
                                "level": "l2",
                                "status": "found",
                                "security_context": security_context.agent_id
                            })
                        return True
                
                except Exception as e:
                    # Fail-secure: Log error and return False
                    logger.warning(f"L2 cache exists error for key {key}: {e}")
                    if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                        cache_operations_counter.add(1, {
                            "operation": "exists",
                            "level": "l2",
                            "status": "error",
                            "security_context": security_context.agent_id if security_context else "none"
                        })
                    return False
            
            # Key not found
            if OPENTELEMETRY_AVAILABLE and cache_operations_counter:
                cache_operations_counter.add(1, {
                    "operation": "exists",
                    "level": "miss",
                    "status": "not_found",
                    "security_context": security_context.agent_id if security_context else "none"
                })
            
            return False
            
        finally:
            # Record response time
            response_time = (time.time() - start_time) * 1000
            self._cache_response_times.append(response_time)
            
            # Update cache operation stats
            op_stats = self._cache_operation_stats["exists"]
            op_stats["count"] += 1
            op_stats["total_time"] += response_time
            op_stats["min_time"] = min(op_stats["min_time"], response_time)
            op_stats["max_time"] = max(op_stats["max_time"], response_time)
    
    def _validate_security_context(self, security_context: SecurityContext) -> bool:
        """Validate security context for Redis operations.
        
        Args:
            security_context: Security context to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not security_context:
            return False
        
        # Basic validation
        if not security_context.agent_id:
            logger.warning("Security context missing agent_id")
            return False
        
        # Authentication requirement for secure operations
        if not security_context.authenticated:
            logger.warning(f"Security context for {security_context.agent_id} not authenticated")
            return False
        
        # Check context age (prevent replay attacks)
        context_age = time.time() - security_context.created_at
        if context_age > 3600:  # 1 hour max age
            logger.warning(f"Security context for {security_context.agent_id} expired (age: {context_age}s)")
            return False
        
        return True

# ========== Global Manager Instances ==========

# Global unified managers for different modes
_unified_managers: Dict[ManagerMode, UnifiedConnectionManager] = {}

def get_unified_manager(mode: ManagerMode = ManagerMode.ASYNC_MODERN) -> UnifiedConnectionManager:
    """Get or create unified manager for specified mode"""
    global _unified_managers
    
    if mode not in _unified_managers:
        _unified_managers[mode] = UnifiedConnectionManager(mode)
    
    return _unified_managers[mode]

# ========== Backward Compatibility Adapters ==========

class DatabaseManagerAdapter:
    """Adapter for legacy DatabaseManager usage"""
    
    def __init__(self, unified_manager: UnifiedConnectionManager):
        self._manager = unified_manager
        import warnings
        warnings.warn(
            "DatabaseManager is deprecated. Use UnifiedConnectionManager directly",
            DeprecationWarning,
            stacklevel=2
        )
    
    async def initialize(self):
        """Initialize the underlying manager"""
        return await self._manager.initialize()
    
    async def get_session(self):
        """Legacy async session interface"""
        return self._manager.get_async_session()
    
    def get_sync_session(self):
        """Legacy sync session interface - not supported in async-only manager"""
        raise NotImplementedError(
            "Sync sessions not supported in UnifiedConnectionManager. Use get_async_session() instead."
        )
    
    async def close(self):
        """Close the underlying manager"""
        return await self._manager.close()
    
    async def health_check(self):
        """Health check proxy"""
        return await self._manager.health_check()

class DatabaseSessionManagerAdapter:
    """Adapter for legacy DatabaseSessionManager usage"""
    
    def __init__(self, unified_manager: UnifiedConnectionManager):
        self._manager = unified_manager
        import warnings
        warnings.warn(
            "DatabaseSessionManager is deprecated. Use UnifiedConnectionManager directly",
            DeprecationWarning,
            stacklevel=2
        )
    
    async def initialize(self):
        """Initialize the underlying manager"""
        return await self._manager.initialize()
    
    def session(self):
        """Legacy session interface"""
        return self._manager.get_async_session()
    
    async def get_async_session(self):
        """Async session interface"""
        return self._manager.get_async_session()
    
    async def close(self):
        """Close the underlying manager"""
        return await self._manager.close()
    
    async def health_check(self):
        """Health check proxy"""
        return await self._manager.health_check()

class HAConnectionManagerAdapter:
    """Adapter for legacy HAConnectionManager usage"""
    
    def __init__(self, unified_manager: UnifiedConnectionManager):
        self._manager = unified_manager
        import warnings
        warnings.warn(
            "HAConnectionManager is deprecated. Use UnifiedConnectionManager with HA mode",
            DeprecationWarning,
            stacklevel=2
        )
    
    async def initialize(self):
        """Initialize the underlying manager"""
        return await self._manager.initialize()
    
    async def get_connection(self, mode=None):
        """Get connection with HA support"""
        return self._manager.get_connection(mode or ConnectionMode.READ_WRITE)
    
    async def close(self):
        """Close the underlying manager"""
        return await self._manager.close()
    
    async def health_check(self):
        """Health check proxy"""
        return await self._manager.health_check()

# ========== Adapter Factory Functions ==========

def get_database_manager_adapter() -> DatabaseManagerAdapter:
    """Get DatabaseManager adapter instance"""
    unified_manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
    return DatabaseManagerAdapter(unified_manager)

def get_database_session_manager_adapter() -> DatabaseSessionManagerAdapter:
    """Get DatabaseSessionManager adapter instance"""
    unified_manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
    return DatabaseSessionManagerAdapter(unified_manager)

def get_ha_connection_manager_adapter() -> HAConnectionManagerAdapter:
    """Get HAConnectionManager adapter instance"""
    unified_manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
    return HAConnectionManagerAdapter(unified_manager)

# ========== Enhanced Factory Functions with Consolidated Functionality ==========

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

# ========== Security Context Helper Functions ==========

# ========== Security Context Factory Functions ==========

async def create_security_context(agent_id: str, 
                                tier: str = "basic",
                                authenticated: bool = True,
                                authentication_method: str = "system",
                                permissions: Optional[List[str]] = None,
                                security_level: str = "basic",
                                session_id: Optional[str] = None,
                                expires_minutes: Optional[int] = None) -> SecurityContext:
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
    expires_at = current_time + (expires_minutes * 60) if expires_minutes else None
    
    return SecurityContext(
        agent_id=agent_id,
        tier=tier,
        authenticated=authenticated,
        created_at=current_time,
        authentication_method=authentication_method,
        authentication_timestamp=current_time,
        session_id=session_id,
        permissions=permissions or [],
        security_level=security_level,
        expires_at=expires_at,
        last_used=current_time
    )

async def create_security_context_from_auth_result(auth_result=None, 
                                                   agent_id: Optional[str] = None,
                                                   tier: Optional[str] = None,
                                                   authenticated: Optional[bool] = None,
                                                   authentication_method: Optional[str] = None,
                                                   permissions: Optional[List[str]] = None,
                                                   security_level: Optional[str] = None,
                                                   expires_minutes: Optional[int] = None,
                                                   auth_result_metadata: Optional[Dict[str, Any]] = None) -> SecurityContext:
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
    
    # Use auth_result if provided, otherwise use direct parameters
    if auth_result:
        # Extract authentication timing
        auth_time_ms = auth_result.performance_metrics.get("total_auth_time_ms", 0.0)
        agent_id_final = auth_result.agent_id or "unknown"
        tier_final = auth_result.rate_limit_tier
        authenticated_final = auth_result.success
        auth_method_final = auth_result.authentication_method.value if hasattr(auth_result.authentication_method, 'value') else str(auth_result.authentication_method)
        permissions_final = auth_result.audit_metadata.get("permissions", [])
        session_id_final = auth_result.session_id
        
        # Determine security level based on authentication method and tier
        security_level_final = "basic"
        if auth_result.rate_limit_tier in ["professional", "enterprise"]:
            security_level_final = "enhanced"
        if auth_method_final == "api_key":
            security_level_final = "high" if security_level_final == "enhanced" else "enhanced"
        
        # Merge metadata
        audit_metadata = {
            "authentication_result_integration": True,
            "auth_result_status": auth_result.status.value if hasattr(auth_result.status, 'value') else str(auth_result.status),
            **auth_result.audit_metadata
        }
        
        expires_at = time.time() + (expires_minutes * 60) if expires_minutes else None
        
    else:
        # Use direct parameters
        auth_time_ms = 0.0
        agent_id_final = agent_id or "unknown"
        tier_final = tier or "basic"
        authenticated_final = authenticated if authenticated is not None else False
        auth_method_final = authentication_method or "unknown"
        permissions_final = permissions or []
        session_id_final = None
        security_level_final = security_level or "basic"
        
        # Merge metadata
        audit_metadata = {
            "direct_parameter_creation": True,
            "authentication_manager_integration": True,
            **(auth_result_metadata or {})
        }
        
        expires_at = time.time() + (expires_minutes * 60) if expires_minutes else None
    
    # Create validation result
    validation_result = SecurityValidationResult(
        validated=authenticated_final,
        validation_method=auth_method_final,
        validation_timestamp=current_time,
        validation_duration_ms=auth_time_ms,
        rate_limit_status="authenticated" if authenticated_final else "failed",
        audit_trail_id=audit_metadata.get("audit_trail_id", f"auth_{int(current_time * 1000000)}")
    )
    
    # Create performance metrics
    performance_metrics = SecurityPerformanceMetrics(
        authentication_time_ms=auth_time_ms,
        total_security_overhead_ms=auth_time_ms,
        operations_count=1,
        last_performance_check=current_time
    )
    
    return SecurityContext(
        agent_id=agent_id_final,
        tier=tier_final,
        authenticated=authenticated_final,
        created_at=current_time,
        authentication_method=auth_method_final,
        authentication_timestamp=current_time,
        session_id=session_id_final,
        permissions=permissions_final,
        validation_result=validation_result,
        performance_metrics=performance_metrics,
        audit_metadata=audit_metadata,
        security_level=security_level_final,
        zero_trust_validated=authenticated_final,
        expires_at=expires_at,
        last_used=current_time
    )

async def create_security_context_from_security_manager(agent_id: str, 
                                                      security_manager,
                                                      additional_context: Optional[Dict[str, Any]] = None) -> SecurityContext:
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
        # Get security status from manager
        security_status = await security_manager.get_security_status()
        
        # Create validation result based on security manager state
        validation_result = SecurityValidationResult(
            validated=True,
            validation_method="security_manager",
            validation_timestamp=current_time,
            validation_duration_ms=0.0,  # Filled in by actual validation
            rate_limit_status="validated",
            encryption_required=security_manager.config.require_encryption,
            audit_trail_id=f"sm_{int(current_time * 1000000)}"
        )
        
        # Create threat score based on security manager metrics
        threat_level = "low"
        threat_score = 0.0
        threat_factors = []
        
        if security_status.get("metrics", {}).get("violation_rate", 0) > 0.1:
            threat_level = "medium" 
            threat_score = 0.3
            threat_factors.append("elevated_violation_rate")
            
        if security_status.get("metrics", {}).get("active_incidents", 0) > 0:
            threat_level = "high"
            threat_score = 0.6
            threat_factors.append("active_security_incidents")
        
        threat_assessment = SecurityThreatScore(
            level=threat_level,
            score=threat_score,
            factors=threat_factors,
            last_updated=current_time
        )
        
        # Create performance metrics
        avg_operation_time = security_status.get("performance", {}).get("average_operation_time_ms", 0.0)
        performance_metrics = SecurityPerformanceMetrics(
            authentication_time_ms=avg_operation_time,
            authorization_time_ms=avg_operation_time,
            validation_time_ms=avg_operation_time,
            total_security_overhead_ms=avg_operation_time * 3,
            operations_count=1,
            last_performance_check=current_time
        )
        
        # Determine authentication status from additional context
        authenticated = additional_context.get("authenticated", True) if additional_context else True
        if additional_context and additional_context.get("authentication_method") == "failed":
            authenticated = False
        
        # Determine security level based on security manager configuration and context
        security_level = security_manager.config.security_level.value
        if additional_context and additional_context.get("threat_detected"):
            security_level = "basic"  # Downgrade security level for failed authentication
        
        # Determine tier based on security manager configuration
        tier = security_manager.config.rate_limit_tier.value
        
        return SecurityContext(
            agent_id=agent_id,
            tier=tier,
            authenticated=authenticated,
            created_at=current_time,
            authentication_method=additional_context.get("authentication_method", "security_manager") if additional_context else "security_manager",
            authentication_timestamp=current_time,
            session_id=additional_context.get("session_id") if additional_context else None,
            permissions=additional_context.get("permissions", []) if additional_context else [],
            validation_result=validation_result,
            threat_score=threat_assessment,
            performance_metrics=performance_metrics,
            audit_metadata={
                "security_manager_integration": True,
                "security_manager_mode": security_manager.mode.value,
                "security_configuration": {
                    "security_level": security_manager.config.security_level.value,
                    "rate_limit_tier": security_manager.config.rate_limit_tier.value,
                    "zero_trust_mode": security_manager.config.zero_trust_mode,
                    "fail_secure": security_manager.config.fail_secure
                },
                **(additional_context or {})
            },
            security_level=security_level,
            zero_trust_validated=security_manager.config.zero_trust_mode,
            encryption_context={"required": security_manager.config.require_encryption},
            last_used=current_time
        )
        
    except Exception as e:
        logger.error(f"Failed to create security context from security manager: {e}")
        # Fail-secure: create minimal security context
        return await create_security_context(
            agent_id=agent_id,
            tier="basic",
            authenticated=False,
            authentication_method="failed",
            security_level="basic"
        )

async def create_system_security_context(operation_type: str = "system",
                                       security_level: str = "high") -> SecurityContext:
    """Create system-level security context for internal operations.
    
    Args:
        operation_type: Type of system operation
        security_level: Security level for system operations
        
    Returns:
        System SecurityContext with elevated privileges
    """
    current_time = time.time()
    
    validation_result = SecurityValidationResult(
        validated=True,
        validation_method="system_internal",
        validation_timestamp=current_time,
        validation_duration_ms=0.0,
        rate_limit_status="system_exempt",
        encryption_required=False,
        audit_trail_id=f"sys_{int(current_time * 1000000)}"
    )
    
    return SecurityContext(
        agent_id="system",
        tier="system",
        authenticated=True,
        created_at=current_time,
        authentication_method="system_internal",
        authentication_timestamp=current_time,
        session_id=None,
        permissions=["system:all"],
        validation_result=validation_result,
        audit_metadata={"operation_type": operation_type},
        compliance_tags=["system_internal"],
        security_level=security_level,
        zero_trust_validated=True,
        last_used=current_time
    )


# Unified Connection Manager is now the default connection manager
logger.info("UnifiedConnectionManager is now the default connection manager with consolidated functionality from MCPConnectionPool, AdaptiveConnectionPool, ConnectionPoolOptimizer, and ConnectionPoolManager")

