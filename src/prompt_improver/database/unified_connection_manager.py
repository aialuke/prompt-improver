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
from collections.abc import AsyncIterator
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

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
from psycopg_pool import AsyncConnectionPool

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
        
        # Database connections
        self._async_engine: Optional[AsyncEngine] = None
        self._async_session_factory: Optional[async_sessionmaker] = None
        
        # HA components (from HAConnectionManager)
        self._pg_pools: Dict[str, AsyncConnectionPool] = {}
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
        
        logger.info(f"UnifiedConnectionManager initialized for mode: {mode.value}")
    
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
                
                # Start health monitoring
                asyncio.create_task(self._health_monitor_loop())
                
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
            f"postgresql://{self.db_config.postgres_username}:{self.db_config.postgres_password}@"
            f"{self.db_config.postgres_host}:{self.db_config.postgres_port}/"
            f"{self.db_config.postgres_database}"
        )
        sync_url = f"postgresql+psycopg://{self.db_config.postgres_username}:{self.db_config.postgres_password}@{self.db_config.postgres_host}:{self.db_config.postgres_port}/{self.db_config.postgres_database}"
        async_url = f"postgresql+psycopg://{self.db_config.postgres_username}:{self.db_config.postgres_password}@{self.db_config.postgres_host}:{self.db_config.postgres_port}/{self.db_config.postgres_database}"
        
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
            # Setup PostgreSQL HA pools
            primary_dsn = f"postgresql://{self.db_config.postgres_username}:{self.db_config.postgres_password}@{self.db_config.postgres_host}:{self.db_config.postgres_port}/{self.db_config.postgres_database}"
            
            primary_pool = AsyncConnectionPool(
                conninfo=primary_dsn,
                min_size=2,
                max_size=self.pool_config.pg_pool_size,
                timeout=self.pool_config.pg_timeout,
                max_lifetime=3600,
                max_idle=600,
            )
            
            await primary_pool.__aenter__()
            self._pg_pools["primary"] = primary_pool
            
            # Add replica pools if configured
            replica_hosts = self._get_replica_hosts()
            for i, (host, port) in enumerate(replica_hosts):
                replica_dsn = f"postgresql://{self.db_config.postgres_username}:{self.db_config.postgres_password}@{host}:{port}/{self.db_config.postgres_database}"
                replica_pool = AsyncConnectionPool(
                    conninfo=replica_dsn,
                    min_size=1,
                    max_size=self.pool_config.pg_pool_size // 2,
                    timeout=self.pool_config.pg_timeout,
                )
                await replica_pool.__aenter__()
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
            
            # Close HA pools
            for pool_name, pool in self._pg_pools.items():
                try:
                    await pool.__aexit__(None, None, None)
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
            
            # Update metrics
            response_time = (time.time() - start_time) * 1000
            self._update_response_time(response_time)
            
        except Exception as e:
            await session.rollback()
            self._metrics.error_rate += 1
            logger.error(f"Async session error in {self.mode.value}: {e}")
            raise
        finally:
            await session.close()
    
    
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
        """Get metrics as dictionary"""
        return {
            "active_connections": self._metrics.active_connections,
            "idle_connections": self._metrics.idle_connections,
            "total_connections": self._metrics.total_connections,
            "pool_utilization": self._metrics.pool_utilization,
            "avg_response_time_ms": self._metrics.avg_response_time_ms,
            "error_rate": self._metrics.error_rate,
            "failed_connections": self._metrics.failed_connections,
            "failover_count": self._metrics.failover_count,
            "last_failover": self._metrics.last_failover,
            "health_check_failures": self._metrics.health_check_failures,
            "circuit_breaker_state": self._metrics.circuit_breaker_state,
            "circuit_breaker_failures": self._circuit_breaker_failures,
            "sla_compliance_rate": self._metrics.sla_compliance_rate,
            "registry_conflicts": self._metrics.registry_conflicts,
            "registered_models": len(self._registry_manager.get_registered_classes()) if self._registry_manager else 0
        }
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while self._is_initialized:
            try:
                await asyncio.sleep(self._health_check_interval)
                health_result = await self.health_check()
                
                # Update SLA compliance
                if health_result.get("response_time_ms", 0) < self.pool_config.pg_timeout * 1000:
                    self._metrics.sla_compliance_rate = min(100.0, self._metrics.sla_compliance_rate + 0.1)
                else:
                    self._metrics.sla_compliance_rate = max(0.0, self._metrics.sla_compliance_rate - 1.0)
                
                self._last_health_check = time.time()
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                self._metrics.health_check_failures += 1
                await asyncio.sleep(self._health_check_interval * 2)

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

# Unified Connection Manager is now the default connection manager
logger.info("UnifiedConnectionManager is now the default connection manager")

