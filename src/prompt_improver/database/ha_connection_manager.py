"""High Availability Connection Manager for PostgreSQL and Redis.

This module implements automatic failover and connection management
following 2025 best practices for production ML Pipeline Orchestrator.

Features:
- PostgreSQL streaming replication with automatic failover
- Redis Sentinel integration for cache failover
- Circuit breaker patterns for resilience
- Health monitoring and metrics collection
- Graceful degradation modes
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import time
import json

import asyncpg
import coredis
from coredis.sentinel import Sentinel
from psycopg_pool import AsyncConnectionPool

from .config import DatabaseConfig
from ..utils.redis_cache import RedisConfig


logger = logging.getLogger(__name__)


class DatabaseRole(Enum):
    """Database role enumeration."""
    PRIMARY = "primary"
    REPLICA = "replica"
    UNKNOWN = "unknown"


class ConnectionState(Enum):
    """Connection state enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class ConnectionMetrics:
    """Connection metrics for monitoring."""
    active_connections: int
    idle_connections: int
    failed_connections: int
    avg_response_time: float
    last_failover: Optional[float]
    failover_count: int
    health_check_failures: int


@dataclass
class DatabaseEndpoint:
    """Database endpoint configuration."""
    host: str
    port: int
    role: DatabaseRole
    priority: int = 100  # Lower number = higher priority
    max_connections: int = 20
    state: ConnectionState = ConnectionState.HEALTHY


class HAConnectionManager:
    """High Availability connection manager with automatic failover.
    
    Manages connections to PostgreSQL primary/replica and Redis master/replica
    with automatic failover capabilities.
    """
    
    def __init__(self, 
                 db_config: DatabaseConfig,
                 redis_config: RedisConfig,
                 logger: Optional[logging.Logger] = None):
        """Initialize HA connection manager.
        
        Args:
            db_config: Database configuration
            redis_config: Redis configuration
            logger: Optional logger instance
        """
        self.db_config = db_config
        self.redis_config = redis_config
        self.logger = logger or logging.getLogger(__name__)
        
        # PostgreSQL endpoints
        self.pg_endpoints: List[DatabaseEndpoint] = []
        self.current_pg_primary: Optional[AsyncConnectionPool] = None
        self.pg_replica_pools: List[AsyncConnectionPool] = []
        
        # Redis endpoints
        self.redis_sentinel: Optional[Sentinel] = None
        self.redis_master: Optional[coredis.Redis] = None
        self.redis_replica: Optional[coredis.Redis] = None
        
        # State management
        self.failover_in_progress = False
        self.last_health_check = 0
        self.health_check_interval = 10  # seconds
        
        # Metrics
        self.metrics = ConnectionMetrics(
            active_connections=0,
            idle_connections=0,
            failed_connections=0,
            avg_response_time=0.0,
            last_failover=None,
            failover_count=0,
            health_check_failures=0
        )
        
        # Circuit breaker settings
        self.circuit_breaker_threshold = 5  # failures before opening circuit
        self.circuit_breaker_timeout = 30  # seconds to wait before retry
        self.circuit_breaker_state = "closed"  # closed, open, half-open
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = 0
        
        self.logger.info("HAConnectionManager initialized")
    
    async def initialize(self):
        """Initialize all connections and start health monitoring."""
        try:
            await self._setup_postgresql_endpoints()
            await self._setup_redis_sentinel()
            await self._start_health_monitoring()
            
            self.logger.info("HA connection manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize HA connection manager: {e}")
            raise
    
    async def _setup_postgresql_endpoints(self):
        """Setup PostgreSQL primary and replica endpoints."""
        # Primary endpoint
        primary_endpoint = DatabaseEndpoint(
            host=self.db_config.postgres_host,
            port=self.db_config.postgres_port,
            role=DatabaseRole.PRIMARY,
            priority=1,
            max_connections=self.db_config.pool_max_size
        )
        self.pg_endpoints.append(primary_endpoint)
        
        # Replica endpoints (from environment or config)
        replica_hosts = self._get_replica_hosts()
        for i, (host, port) in enumerate(replica_hosts):
            replica_endpoint = DatabaseEndpoint(
                host=host,
                port=port,
                role=DatabaseRole.REPLICA,
                priority=10 + i,  # Lower priority than primary
                max_connections=self.db_config.pool_max_size // 2
            )
            self.pg_endpoints.append(replica_endpoint)
        
        # Initialize connection pools
        await self._initialize_pg_pools()
    
    def _get_replica_hosts(self) -> List[tuple]:
        """Get replica host configurations from environment."""
        # In production, this would come from service discovery or config
        # For now, use the HA setup from docker-compose
        return [
            ("postgres-replica", 5432),  # Docker service name
            # Add more replicas as needed
        ]
    
    async def _initialize_pg_pools(self):
        """Initialize PostgreSQL connection pools."""
        for endpoint in self.pg_endpoints:
            try:
                dsn = f"postgresql://{self.db_config.postgres_username}:{self.db_config.postgres_password}@{endpoint.host}:{endpoint.port}/{self.db_config.postgres_database}"
                
                pool = AsyncConnectionPool(
                    conninfo=dsn,
                    min_size=2,
                    max_size=endpoint.max_connections,
                    timeout=self.db_config.pool_timeout,
                    max_lifetime=self.db_config.pool_max_lifetime,
                    max_idle=self.db_config.pool_max_idle,
                )
                
                await pool.__aenter__()
                
                if endpoint.role == DatabaseRole.PRIMARY:
                    self.current_pg_primary = pool
                    self.logger.info(f"Primary PostgreSQL pool initialized: {endpoint.host}:{endpoint.port}")
                else:
                    self.pg_replica_pools.append(pool)
                    self.logger.info(f"Replica PostgreSQL pool initialized: {endpoint.host}:{endpoint.port}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize pool for {endpoint.host}:{endpoint.port}: {e}")
                endpoint.state = ConnectionState.FAILED
    
    async def _setup_redis_sentinel(self):
        """Setup Redis Sentinel for automatic failover."""
        try:
            # Sentinel hosts (from docker-compose HA setup)
            sentinel_hosts = [
                ("redis-sentinel-1", 26379),
                ("redis-sentinel-2", 26379),
                ("redis-sentinel-3", 26379),
            ]
            
            self.redis_sentinel = Sentinel(
                sentinels=sentinel_hosts,
                stream_timeout=0.1,
                connect_timeout=0.1
            )
            
            # Get master and replica connections
            self.redis_master = self.redis_sentinel.primary_for(
                'mymaster',
                stream_timeout=0.1,
                password=getattr(self.redis_config, 'password', None)
            )
            
            self.redis_replica = self.redis_sentinel.replica_for(
                'mymaster',
                stream_timeout=0.1,
                password=getattr(self.redis_config, 'password', None)
            )
            
            self.logger.info("Redis Sentinel initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Redis Sentinel: {e}")
            # Fallback to direct Redis connection
            await self._setup_redis_fallback()
    
    async def _setup_redis_fallback(self):
        """Setup direct Redis connection as fallback."""
        try:
            self.redis_master = coredis.Redis(
                host=self.redis_config.host,
                port=self.redis_config.port,
                db=self.redis_config.cache_db,
                password=getattr(self.redis_config, 'password', None),
                stream_timeout=self.redis_config.socket_timeout,
                connect_timeout=self.redis_config.connect_timeout
            )
            
            self.logger.info("Redis fallback connection initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Redis fallback: {e}")
    
    async def get_pg_connection(self, read_only: bool = False, prefer_replica: bool = True) -> asyncpg.Connection:
        """Get PostgreSQL connection with automatic failover.
        
        Args:
            read_only: If True, prefer replica connections
            prefer_replica: If True and read_only, try replica first
            
        Returns:
            PostgreSQL connection
            
        Raises:
            ConnectionError: If no connections available
        """
        if self._is_circuit_breaker_open():
            raise ConnectionError("Circuit breaker is open")
        
        try:
            # For read-only queries, try replicas first if available
            if read_only and prefer_replica and self.pg_replica_pools:
                for replica_pool in self.pg_replica_pools:
                    try:
                        conn = await replica_pool.acquire()
                        await self._update_connection_metrics(success=True)
                        return conn
                    except Exception as e:
                        self.logger.warning(f"Replica connection failed: {e}")
                        continue
            
            # Use primary connection
            if self.current_pg_primary:
                try:
                    conn = await self.current_pg_primary.acquire()
                    await self._update_connection_metrics(success=True)
                    return conn
                except Exception as e:
                    self.logger.error(f"Primary connection failed: {e}")
                    if not self.failover_in_progress:
                        await self._initiate_pg_failover()
            
            # If we reach here, all connections failed
            await self._update_connection_metrics(success=False)
            raise ConnectionError("No PostgreSQL connections available")
            
        except Exception as e:
            await self._handle_connection_failure(e)
            raise
    
    async def get_redis_connection(self, read_only: bool = False) -> coredis.Redis:
        """Get Redis connection with automatic failover.
        
        Args:
            read_only: If True, prefer replica connection
            
        Returns:
            Redis connection
        """
        try:
            if read_only and self.redis_replica:
                try:
                    # Test replica connection
                    await self.redis_replica.ping()
                    return self.redis_replica
                except Exception as e:
                    self.logger.warning(f"Redis replica failed: {e}")
            
            # Use master connection
            if self.redis_master:
                await self.redis_master.ping()
                return self.redis_master
            
            raise ConnectionError("No Redis connections available")
            
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}")
            raise ConnectionError(f"Redis connection failed: {e}")
    
    async def _initiate_pg_failover(self):
        """Initiate PostgreSQL failover to replica."""
        if self.failover_in_progress:
            return
        
        self.failover_in_progress = True
        self.logger.warning("Initiating PostgreSQL failover")
        
        try:
            # Find healthy replica to promote
            for replica_pool in self.pg_replica_pools:
                try:
                    async with replica_pool.acquire() as conn:
                        # Test connection
                        await conn.execute("SELECT 1")
                    
                    # Promote this replica to primary
                    self.current_pg_primary = replica_pool
                    self.pg_replica_pools.remove(replica_pool)
                    
                    self.metrics.failover_count += 1
                    self.metrics.last_failover = time.time()
                    
                    self.logger.info("PostgreSQL failover completed successfully")
                    break
                    
                except Exception as e:
                    self.logger.error(f"Replica failover attempt failed: {e}")
                    continue
            
        finally:
            self.failover_in_progress = False
    
    async def _start_health_monitoring(self):
        """Start background health monitoring task."""
        asyncio.create_task(self._health_monitor_loop())
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval * 2)  # Back off on error
    
    async def _perform_health_checks(self):
        """Perform health checks on all connections."""
        current_time = time.time()
        
        # Check PostgreSQL connections
        await self._check_pg_health()
        
        # Check Redis connections
        await self._check_redis_health()
        
        # Update circuit breaker state
        await self._update_circuit_breaker_state()
        
        self.last_health_check = current_time
    
    async def _check_pg_health(self):
        """Check PostgreSQL connection health."""
        if self.current_pg_primary:
            try:
                async with self.current_pg_primary.acquire() as conn:
                    result = await conn.fetchval("SELECT health_check()")
                    health_data = json.loads(result)
                    
                    if not health_data.get('is_primary', False):
                        self.logger.warning("Current primary is no longer primary, initiating failover")
                        await self._initiate_pg_failover()
                        
            except Exception as e:
                self.logger.error(f"Primary health check failed: {e}")
                self.metrics.health_check_failures += 1
    
    async def _check_redis_health(self):
        """Check Redis connection health."""
        if self.redis_master:
            try:
                await self.redis_master.ping()
            except Exception as e:
                self.logger.error(f"Redis master health check failed: {e}")
                self.metrics.health_check_failures += 1
    
    async def _update_connection_metrics(self, success: bool):
        """Update connection metrics."""
        if success:
            self.circuit_breaker_failures = 0
        else:
            self.circuit_breaker_failures += 1
            self.circuit_breaker_last_failure = time.time()
            self.metrics.failed_connections += 1
    
    async def _handle_connection_failure(self, error: Exception):
        """Handle connection failure and update circuit breaker."""
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = time.time()
        
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            self.circuit_breaker_state = "open"
            self.logger.error(f"Circuit breaker opened due to {self.circuit_breaker_failures} failures")
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.circuit_breaker_state == "open":
            if time.time() - self.circuit_breaker_last_failure > self.circuit_breaker_timeout:
                self.circuit_breaker_state = "half-open"
                self.logger.info("Circuit breaker moved to half-open state")
                return False
            return True
        return False
    
    async def _update_circuit_breaker_state(self):
        """Update circuit breaker state based on recent health."""
        if self.circuit_breaker_state == "half-open":
            # If we've had successful operations, close the circuit
            if self.circuit_breaker_failures == 0:
                self.circuit_breaker_state = "closed"
                self.logger.info("Circuit breaker closed")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "timestamp": time.time(),
            "postgresql": {
                "primary_available": self.current_pg_primary is not None,
                "replica_count": len(self.pg_replica_pools),
                "failover_in_progress": self.failover_in_progress
            },
            "redis": {
                "master_available": self.redis_master is not None,
                "replica_available": self.redis_replica is not None,
                "sentinel_enabled": self.redis_sentinel is not None
            },
            "circuit_breaker": {
                "state": self.circuit_breaker_state,
                "failures": self.circuit_breaker_failures,
                "last_failure": self.circuit_breaker_last_failure
            },
            "metrics": {
                "failover_count": self.metrics.failover_count,
                "last_failover": self.metrics.last_failover,
                "health_check_failures": self.metrics.health_check_failures,
                "failed_connections": self.metrics.failed_connections
            }
        }
    
    async def shutdown(self):
        """Shutdown all connections gracefully."""
        self.logger.info("Shutting down HA connection manager")
        
        # Close PostgreSQL pools
        if self.current_pg_primary:
            await self.current_pg_primary.__aexit__(None, None, None)
        
        for replica_pool in self.pg_replica_pools:
            await replica_pool.__aexit__(None, None, None)
        
        # Close Redis connections
        if self.redis_master:
            await self.redis_master.aclose()
        
        if self.redis_replica:
            await self.redis_replica.aclose()
        
        self.logger.info("HA connection manager shutdown complete")
