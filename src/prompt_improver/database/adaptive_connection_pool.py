"""
Adaptive Database Connection Pool for ML Pipeline Orchestrator Performance Optimization.

Implements auto-scaling connection pool management following 2025 best practices.
Target: Auto-scale from 20 → 100+ connections based on load.

Key Features:
- Dynamic connection pool sizing based on demand
- Real-time performance monitoring and metrics
- Connection health monitoring and auto-recovery
- Circuit breaker patterns for database resilience
- Load-based scaling with intelligent thresholds
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import statistics

try:
    import psycopg
    from psycopg_pool import AsyncConnectionPool
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False
    AsyncConnectionPool = None

from .config import DatabaseConfig


class PoolState(Enum):
    """Connection pool operational states."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    SCALING_UP = "scaling_up"
    SCALING_DOWN = "scaling_down"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass
class ConnectionMetrics:
    """Real-time connection pool metrics."""
    active_connections: int = 0
    idle_connections: int = 0
    total_connections: int = 0
    connections_created: int = 0
    connections_closed: int = 0
    connection_failures: int = 0
    avg_connection_time_ms: float = 0.0
    avg_query_time_ms: float = 0.0
    queries_executed: int = 0
    queries_failed: int = 0
    pool_utilization: float = 0.0
    wait_time_ms: float = 0.0
    last_scale_event: Optional[datetime] = None
    connection_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    query_times: deque = field(default_factory=lambda: deque(maxlen=1000))


class AdaptiveConnectionPool:
    """
    High-performance adaptive database connection pool.
    
    Auto-scales from 20 → 100+ connections based on real-time demand
    with intelligent health monitoring and circuit breaker patterns.
    """
    
    def __init__(self, config: DatabaseConfig):
        """Initialize adaptive connection pool."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Dynamic pool configuration
        self.min_pool_size = config.pool_min_size
        self.max_pool_size = min(config.pool_max_size * 5, 100)  # Scale up to 100 connections
        self.current_pool_size = config.pool_max_size
        
        # Connection pool
        self.pool: Optional[AsyncConnectionPool] = None
        self.connection_string = self._build_connection_string()
        
        # Performance monitoring
        self.metrics = ConnectionMetrics()
        self.performance_window = deque(maxlen=100)
        self.last_metrics_update = time.time()
        
        # Adaptive scaling
        self.scale_up_threshold = 0.8  # 80% utilization
        self.scale_down_threshold = 0.3  # 30% utilization
        self.scale_cooldown_seconds = 60  # Longer cooldown for DB connections
        self.last_scale_time = 0
        
        # Circuit breaker
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 120  # 2 minutes
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = 0
        self.circuit_breaker_open = False
        
        # Health monitoring
        self.health_check_interval = 30
        self.last_health_check = 0
        self.unhealthy_connections: set = set()
        
        # State management
        self.state = PoolState.INITIALIZING
        self.is_initialized = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        self.logger.info(f"AdaptiveConnectionPool initialized: {self.min_pool_size}-{self.max_pool_size} connections")
    
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        return (f"postgresql://{self.config.postgres_username}:{self.config.postgres_password}"
                f"@{self.config.postgres_host}:{self.config.postgres_port}/{self.config.postgres_database}")
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self.is_initialized:
            return
        
        if not PSYCOPG_AVAILABLE:
            raise RuntimeError("psycopg3 not available for adaptive connection pool")
        
        self.logger.info("Initializing adaptive connection pool...")
        
        try:
            # Create initial pool
            self.pool = AsyncConnectionPool(
                conninfo=self.connection_string,
                min_size=self.min_pool_size,
                max_size=self.current_pool_size,
                timeout=self.config.pool_timeout,
                max_lifetime=self.config.pool_max_lifetime,
                max_idle=self.config.pool_max_idle,
            )
            
            await self.pool.__aenter__()
            
            # Start monitoring
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.state = PoolState.HEALTHY
            self.is_initialized = True
            
            self.logger.info(f"Adaptive connection pool initialized with {self.current_pool_size} connections")
            
        except Exception as e:
            self.state = PoolState.FAILED
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    async def close(self) -> None:
        """Close the connection pool."""
        if not self.is_initialized:
            return
        
        self.logger.info("Closing adaptive connection pool...")
        
        # Cancel monitoring
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Close pool
        if self.pool:
            await self.pool.__aexit__(None, None, None)
        
        self.is_initialized = False
        self.state = PoolState.FAILED
        self.logger.info("Adaptive connection pool closed")
    
    async def acquire_connection(self, timeout: Optional[float] = None):
        """
        Acquire a database connection with performance tracking.
        
        Args:
            timeout: Connection acquisition timeout
            
        Returns:
            Database connection
        """
        if not self.is_initialized or self.circuit_breaker_open:
            if self.circuit_breaker_open:
                if time.time() - self.circuit_breaker_last_failure > self.circuit_breaker_timeout:
                    await self._reset_circuit_breaker()
                else:
                    raise ConnectionError("Circuit breaker is open")
            else:
                raise RuntimeError("Connection pool not initialized")
        
        start_time = time.time()
        
        try:
            # Acquire connection from pool
            connection = await asyncio.wait_for(
                self.pool.acquire(),
                timeout=timeout or self.config.pool_timeout
            )
            
            # Track metrics
            connection_time = (time.time() - start_time) * 1000
            self.metrics.connection_times.append(connection_time)
            self.metrics.connections_created += 1
            
            # Update average connection time
            if self.metrics.connection_times:
                self.metrics.avg_connection_time_ms = statistics.mean(self.metrics.connection_times)
            
            return AdaptiveConnection(connection, self)
            
        except asyncio.TimeoutError:
            self.metrics.wait_time_ms = (time.time() - start_time) * 1000
            await self._handle_connection_timeout()
            raise ConnectionError("Connection acquisition timeout")
        except Exception as e:
            await self._handle_connection_failure(e)
            raise
    
    async def execute_query(self, query: str, params: Optional[tuple] = None, timeout: Optional[float] = None) -> Any:
        """
        Execute a query with performance tracking.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            timeout: Query timeout
            
        Returns:
            Query result
        """
        start_time = time.time()
        
        async with await self.acquire_connection(timeout=timeout) as conn:
            try:
                if params:
                    result = await conn.execute(query, params)
                else:
                    result = await conn.execute(query)
                
                # Track query performance
                query_time = (time.time() - start_time) * 1000
                self.metrics.query_times.append(query_time)
                self.metrics.queries_executed += 1
                
                # Update average query time
                if self.metrics.query_times:
                    self.metrics.avg_query_time_ms = statistics.mean(self.metrics.query_times)
                
                return result
                
            except Exception as e:
                self.metrics.queries_failed += 1
                await self._handle_query_failure(e)
                raise
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for adaptive scaling and health checks."""
        while self.is_initialized:
            try:
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
                await self._update_metrics()
                await self._evaluate_scaling()
                await self._health_check()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
    
    async def _update_metrics(self) -> None:
        """Update real-time pool metrics."""
        if not self.pool:
            return
        
        current_time = time.time()
        
        # Get pool statistics
        pool_stats = self.pool.get_stats()
        
        self.metrics.total_connections = pool_stats.pool_size
        self.metrics.active_connections = pool_stats.requests_active
        self.metrics.idle_connections = pool_stats.pool_size - pool_stats.requests_active
        
        # Calculate utilization
        if self.metrics.total_connections > 0:
            self.metrics.pool_utilization = self.metrics.active_connections / self.metrics.total_connections
        
        # Store performance snapshot
        self.performance_window.append({
            'timestamp': current_time,
            'utilization': self.metrics.pool_utilization,
            'active_connections': self.metrics.active_connections,
            'total_connections': self.metrics.total_connections,
            'avg_connection_time': self.metrics.avg_connection_time_ms,
            'avg_query_time': self.metrics.avg_query_time_ms
        })
        
        self.last_metrics_update = current_time
    
    async def _evaluate_scaling(self) -> None:
        """Evaluate if pool scaling is needed."""
        if time.time() - self.last_scale_time < self.scale_cooldown_seconds:
            return
        
        utilization = self.metrics.pool_utilization
        
        # Scale up conditions
        if (utilization > self.scale_up_threshold and 
            self.current_pool_size < self.max_pool_size):
            
            new_size = min(self.current_pool_size + 10, self.max_pool_size)
            await self._scale_pool(new_size)
            self.state = PoolState.SCALING_UP
            
        # Scale down conditions
        elif (utilization < self.scale_down_threshold and 
              self.current_pool_size > self.min_pool_size and
              self.metrics.avg_connection_time_ms < 50):  # Low connection time
            
            new_size = max(self.current_pool_size - 5, self.min_pool_size)
            await self._scale_pool(new_size)
            self.state = PoolState.SCALING_DOWN
        
        else:
            self.state = PoolState.HEALTHY
    
    async def _scale_pool(self, new_size: int) -> None:
        """Scale the connection pool to new size."""
        if not self.pool:
            return
        
        old_size = self.current_pool_size
        
        try:
            # Close current pool
            await self.pool.__aexit__(None, None, None)
            
            # Create new pool with new size
            self.pool = AsyncConnectionPool(
                conninfo=self.connection_string,
                min_size=self.min_pool_size,
                max_size=new_size,
                timeout=self.config.pool_timeout,
                max_lifetime=self.config.pool_max_lifetime,
                max_idle=self.config.pool_max_idle,
            )
            
            await self.pool.__aenter__()
            
            self.current_pool_size = new_size
            self.last_scale_time = time.time()
            self.metrics.last_scale_event = datetime.now(timezone.utc)
            
            self.logger.info(f"Scaled connection pool: {old_size} → {new_size} connections")
            
        except Exception as e:
            self.logger.error(f"Failed to scale connection pool: {e}")
            self.state = PoolState.DEGRADED
    
    async def _health_check(self) -> None:
        """Perform connection pool health check."""
        current_time = time.time()
        
        if current_time - self.last_health_check < self.health_check_interval:
            return
        
        try:
            # Test a simple query
            async with await self.acquire_connection(timeout=5.0) as conn:
                await conn.execute("SELECT 1")
            
            # Reset circuit breaker if health check passes
            if self.circuit_breaker_failures > 0:
                self.circuit_breaker_failures = max(0, self.circuit_breaker_failures - 1)
            
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            await self._handle_connection_failure(e)
        
        self.last_health_check = current_time
    
    async def _handle_connection_timeout(self) -> None:
        """Handle connection acquisition timeout."""
        self.logger.warning("Connection acquisition timeout - considering scale up")
        
        # Trigger immediate scale evaluation
        if (self.current_pool_size < self.max_pool_size and 
            time.time() - self.last_scale_time > 30):  # Shorter cooldown for timeouts
            
            new_size = min(self.current_pool_size + 5, self.max_pool_size)
            await self._scale_pool(new_size)
    
    async def _handle_connection_failure(self, error: Exception) -> None:
        """Handle connection failure."""
        self.metrics.connection_failures += 1
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = time.time()
        
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            self.circuit_breaker_open = True
            self.state = PoolState.DEGRADED
            self.logger.error(f"Circuit breaker opened due to {self.circuit_breaker_failures} failures")
    
    async def _handle_query_failure(self, error: Exception) -> None:
        """Handle query execution failure."""
        self.logger.warning(f"Query execution failed: {error}")
        # Query failures are less severe than connection failures
        self.circuit_breaker_failures += 0.5  # Half weight for query failures
    
    async def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker after timeout."""
        self.circuit_breaker_open = False
        self.circuit_breaker_failures = 0
        self.state = PoolState.HEALTHY
        self.logger.info("Circuit breaker reset - resuming normal operation")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            'state': self.state.value,
            'pool_size': self.current_pool_size,
            'min_pool_size': self.min_pool_size,
            'max_pool_size': self.max_pool_size,
            'active_connections': self.metrics.active_connections,
            'idle_connections': self.metrics.idle_connections,
            'total_connections': self.metrics.total_connections,
            'pool_utilization': self.metrics.pool_utilization,
            'connections_created': self.metrics.connections_created,
            'connections_closed': self.metrics.connections_closed,
            'connection_failures': self.metrics.connection_failures,
            'avg_connection_time_ms': self.metrics.avg_connection_time_ms,
            'avg_query_time_ms': self.metrics.avg_query_time_ms,
            'queries_executed': self.metrics.queries_executed,
            'queries_failed': self.metrics.queries_failed,
            'wait_time_ms': self.metrics.wait_time_ms,
            'circuit_breaker_open': self.circuit_breaker_open,
            'circuit_breaker_failures': self.circuit_breaker_failures,
            'last_scale_event': self.metrics.last_scale_event.isoformat() if self.metrics.last_scale_event else None,
            'performance_window': list(self.performance_window)
        }


class AdaptiveConnection:
    """Wrapper for database connections with performance tracking."""
    
    def __init__(self, connection, pool: AdaptiveConnectionPool):
        """Initialize adaptive connection wrapper."""
        self.connection = connection
        self.pool = pool
        self.acquired_at = time.time()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self.connection
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with performance tracking."""
        try:
            # Track connection usage time
            usage_time = (time.time() - self.acquired_at) * 1000
            
            # Release connection back to pool
            await self.pool.pool.release(self.connection)
            self.pool.metrics.connections_closed += 1
            
        except Exception as e:
            self.pool.logger.error(f"Error releasing connection: {e}")
    
    def __getattr__(self, name):
        """Delegate attribute access to underlying connection."""
        return getattr(self.connection, name)
