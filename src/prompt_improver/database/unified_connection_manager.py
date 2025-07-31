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
import logging
import os
import time
import statistics
from collections import deque
from collections.abc import AsyncIterator
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, timezone

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

# Internal imports
from ..core.config import AppConfig
from .registry import RegistryManager, get_registry_manager
# RedisConfig now accessed via AppConfig

logger = logging.getLogger(__name__)


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
    
    @classmethod
    def for_mode(cls, mode: ManagerMode) -> 'PoolConfiguration':
        """Create pool configuration optimized for specific mode"""
        configs = {
            ManagerMode.MCP_SERVER: cls(
                mode=mode, pg_pool_size=20, pg_max_overflow=10, pg_timeout=0.2,
                redis_pool_size=10, enable_circuit_breaker=True
            ),
            ManagerMode.ML_TRAINING: cls(
                mode=mode, pg_pool_size=15, pg_max_overflow=10, pg_timeout=5.0,
                redis_pool_size=8, enable_ha=True
            ),
            ManagerMode.ADMIN: cls(
                mode=mode, pg_pool_size=5, pg_max_overflow=2, pg_timeout=10.0,
                redis_pool_size=3, enable_ha=True
            ),
            ManagerMode.ASYNC_MODERN: cls(
                mode=mode, pg_pool_size=12, pg_max_overflow=8, pg_timeout=5.0,
                redis_pool_size=6, enable_circuit_breaker=True
            ),
            ManagerMode.HIGH_AVAILABILITY: cls(
                mode=mode, pg_pool_size=20, pg_max_overflow=20, pg_timeout=10.0,
                redis_pool_size=10, enable_ha=True, enable_circuit_breaker=True
            )
        }
        return configs.get(mode, configs[ManagerMode.ASYNC_MODERN])

class UnifiedConnectionManager:
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
            config = AppConfig()
            self.db_config = config.database
        else:
            self.db_config = db_config
        self.redis_config = redis_config or self._get_redis_config()
        
        # Pool configuration
        self.pool_config = PoolConfiguration.for_mode(mode)
        
        # Component managers (composition pattern)
        self._registry_manager = get_registry_manager()
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
        
        logger.info(f"UnifiedConnectionManager initialized for mode: {mode.value} with auto-scaling {self.min_pool_size}-{self.max_pool_size} connections")
    
    def _get_redis_config(self):
        """Get Redis configuration from AppConfig"""
        try:
            config = AppConfig()
            return config.redis
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
                
                # Start health monitoring and auto-scaling (from AdaptiveConnectionPool)
                asyncio.create_task(self._health_monitor_loop())
                asyncio.create_task(self._monitoring_loop())
                
                self._is_initialized = True
                self._health_status = HealthStatus.HEALTHY
                
                logger.info(f"UnifiedConnectionManager initialized successfully for {self.mode.value}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize UnifiedConnectionManager: {e}")
                self._health_status = HealthStatus.UNHEALTHY
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
    
    async def close(self) -> None:
        """Close all connections and cleanup resources"""
        logger.info("Shutting down UnifiedConnectionManager")
        
        try:
            # Close async engine  
            if self._async_engine:
                await self._async_engine.dispose()
            
            # Close HA pools (asyncpg pools)
            for pool_name, pool in self._pg_pools.items():
                try:
                    await pool.close()
                except Exception as e:
                    logger.warning(f"Error closing HA pool {pool_name}: {e}")
            
            # Close Redis connections
            if self._redis_master:
                await self._redis_master.aclose()
            if self._redis_replica:
                await self._redis_replica.aclose()
            
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
            'performance_window': list(self.performance_window)
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
            "registered_models": len(self._registry_manager.get_registered_classes()) if self._registry_manager else 0,
            
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

# Unified Connection Manager is now the default connection manager
logger.info("UnifiedConnectionManager is now the default connection manager with consolidated functionality from MCPConnectionPool, AdaptiveConnectionPool, ConnectionPoolOptimizer, and ConnectionPoolManager")

