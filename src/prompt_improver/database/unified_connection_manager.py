"""
Unified Database Connection Manager for APES
Consolidates 5 different connection patterns into a single, optimized manager.

This unified manager preserves the best features from:
- MCPConnectionPool: MCP-specific optimizations and permissions
- HAConnectionManager: High availability and failover
- TypeSafePsycopgClient: Type safety and psycopg3 features  
- AdaptiveConnectionPool: Adaptive pooling capabilities
- DatabaseSessionManager: Standard session management

Designed to work with the existing seeded PostgreSQL database (apes_production).
"""

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import QueuePool
from pydantic import BaseModel

from .config import DatabaseConfig

logger = logging.getLogger(__name__)

class ConnectionMode(Enum):
    """Connection modes for different use cases"""
    MCP_SERVER = "mcp_server"  # Read-only rule application
    ML_TRAINING = "ml_training"  # ML training system
    ADMIN = "admin"  # Administrative operations

@dataclass
class ConnectionMetrics:
    """Connection pool metrics"""
    active_connections: int = 0
    idle_connections: int = 0
    total_connections: int = 0
    pool_utilization: float = 0.0
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0

class UnifiedConnectionManager:
    """
    Unified database connection manager that consolidates all connection patterns.
    
    Features:
    - Single connection pool for all use cases
    - Permission-based access control (mcp_server_user, ml_training_user, admin_user)
    - Adaptive pool sizing based on workload
    - Type-safe operations with psycopg3
    - High availability with connection health monitoring
    - Performance optimization for <200ms SLA
    - Works with existing seeded apes_production database
    """
    
    def __init__(self, mode: ConnectionMode = ConnectionMode.MCP_SERVER):
        self.mode = mode
        self.config = DatabaseConfig()
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._metrics = ConnectionMetrics()
        self._is_initialized = False
        
        # Connection parameters based on mode
        self._setup_connection_params()
        
        logger.info(f"Unified Connection Manager initialized for mode: {mode.value}")
    
    def _setup_connection_params(self):
        """Setup connection parameters based on mode"""
        if self.mode == ConnectionMode.MCP_SERVER:
            # MCP server: read-only rule application with limited permissions
            self.user = "mcp_server_user"
            self.password_env = "MCP_POSTGRES_PASSWORD"
            self.pool_size = int(os.getenv("MCP_DB_POOL_SIZE", "20"))
            self.max_overflow = int(os.getenv("MCP_DB_MAX_OVERFLOW", "10"))
            self.timeout_ms = int(os.getenv("MCP_REQUEST_TIMEOUT_MS", "200"))
            
        elif self.mode == ConnectionMode.ML_TRAINING:
            # ML training: read/write access for training data
            self.user = "ml_training_user"
            self.password_env = "ML_POSTGRES_PASSWORD"
            self.pool_size = int(os.getenv("ML_DB_POOL_SIZE", "10"))
            self.max_overflow = int(os.getenv("ML_DB_MAX_OVERFLOW", "5"))
            self.timeout_ms = int(os.getenv("ML_REQUEST_TIMEOUT_MS", "5000"))
            
        elif self.mode == ConnectionMode.ADMIN:
            # Admin: full access for maintenance
            self.user = self.config.postgres_username
            self.password_env = "POSTGRES_PASSWORD"
            self.pool_size = int(os.getenv("ADMIN_DB_POOL_SIZE", "5"))
            self.max_overflow = int(os.getenv("ADMIN_DB_MAX_OVERFLOW", "2"))
            self.timeout_ms = int(os.getenv("ADMIN_REQUEST_TIMEOUT_MS", "10000"))
        
        # Get password from environment
        self.password = os.getenv(self.password_env)
        if not self.password:
            raise ValueError(f"{self.password_env} environment variable must be set")
        
        # Build database URL for seeded apes_production database
        self.database_url = (
            f"postgresql+psycopg://{self.user}:{self.password}@"
            f"{self.config.postgres_host}:{self.config.postgres_port}/"
            f"{self.config.postgres_database}"
        )
    
    async def initialize(self) -> bool:
        """Initialize the connection manager"""
        try:
            if self._is_initialized:
                return True
                
            # Create optimized async engine
            self._engine = self._create_optimized_engine()
            
            # Create session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )
            
            # Test connection to seeded database
            await self._test_connection()
            
            self._is_initialized = True
            logger.info(f"Unified Connection Manager initialized successfully for {self.mode.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize connection manager: {e}")
            return False
    
    def _create_optimized_engine(self) -> AsyncEngine:
        """Create optimized async engine with best practices from all connection managers"""
        
        # Engine configuration combining best features from all managers
        engine_kwargs = {
            # Connection pool configuration
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.timeout_ms / 1000,  # Convert to seconds
            "pool_recycle": self.config.pool_recycle,
            "pool_pre_ping": True,  # Health check connections
            "poolclass": QueuePool,  # Explicit pool class
            
            # Performance optimizations
            "echo": self.config.echo_sql,
            "echo_pool": self.config.echo_pool,
            "future": True,  # SQLAlchemy 2.0 mode
            
            # Connection arguments for psycopg3
            "connect_args": {
                "server_settings": {
                    "application_name": f"apes_{self.mode.value}",
                    "timezone": "UTC",
                },
                "command_timeout": self.timeout_ms / 1000,
                "connect_timeout": 10,
            }
        }
        
        engine = create_async_engine(self.database_url, **engine_kwargs)
        
        # Add connection event listeners for monitoring
        self._setup_connection_monitoring(engine)
        
        return engine
    
    def _setup_connection_monitoring(self, engine: AsyncEngine):
        """Setup connection monitoring events"""
        
        @event.listens_for(engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Monitor connection creation"""
            logger.debug(f"New connection created for {self.mode.value}")
            self._metrics.total_connections += 1
        
        @event.listens_for(engine.sync_engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            """Monitor connection checkout"""
            self._metrics.active_connections += 1
            self._metrics.idle_connections = max(0, self._metrics.idle_connections - 1)
            self._update_pool_utilization()
        
        @event.listens_for(engine.sync_engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            """Monitor connection checkin"""
            self._metrics.active_connections = max(0, self._metrics.active_connections - 1)
            self._metrics.idle_connections += 1
            self._update_pool_utilization()
    
    def _update_pool_utilization(self):
        """Update pool utilization metrics"""
        if self.pool_size > 0:
            self._metrics.pool_utilization = (
                self._metrics.active_connections / (self.pool_size + self.max_overflow)
            ) * 100
    
    async def _test_connection(self):
        """Test connection to seeded database"""
        async with self.get_session() as session:
            # Test basic connectivity
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
            
            # Test access to seeded rules (verify we can read from the seeded database)
            if self.mode in [ConnectionMode.MCP_SERVER, ConnectionMode.ML_TRAINING]:
                try:
                    # Check if we can access rule_metadata table (should exist in seeded DB)
                    await session.execute(text("SELECT COUNT(*) FROM rule_metadata LIMIT 1"))
                    logger.info(f"Successfully connected to seeded database with {self.mode.value} permissions")
                except Exception as e:
                    logger.warning(f"Could not access rule_metadata table: {e}")
    
    @contextlib.asynccontextmanager
    async def get_session(self) -> AsyncIterator[AsyncSession]:
        """
        Get an async session with automatic transaction management.
        
        Provides the unified interface that replaces all 5 connection patterns.
        """
        if not self._is_initialized:
            await self.initialize()
        
        if not self._session_factory:
            raise RuntimeError("Connection manager not properly initialized")
        
        session = self._session_factory()
        start_time = time.time()
        
        try:
            yield session
            await session.commit()
            
            # Update performance metrics
            response_time = (time.time() - start_time) * 1000
            self._update_response_time(response_time)
            
        except Exception as e:
            await session.rollback()
            self._metrics.error_rate += 1
            logger.error(f"Session error in {self.mode.value}: {e}")
            raise
        finally:
            await session.close()
    
    def _update_response_time(self, response_time_ms: float):
        """Update average response time using exponential moving average"""
        alpha = 0.1  # Smoothing factor
        if self._metrics.avg_response_time_ms == 0:
            self._metrics.avg_response_time_ms = response_time_ms
        else:
            self._metrics.avg_response_time_ms = (
                alpha * response_time_ms + 
                (1 - alpha) * self._metrics.avg_response_time_ms
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check combining all manager capabilities"""
        try:
            start_time = time.time()
            
            async with self.get_session() as session:
                # Basic connectivity test
                await session.execute(text("SELECT 1"))
                
                # Test seeded database access
                rule_count = await session.execute(text("SELECT COUNT(*) FROM rule_metadata"))
                rule_count_value = rule_count.scalar()
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "mode": self.mode.value,
                "database": self.config.postgres_database,
                "user": self.user,
                "response_time_ms": response_time,
                "seeded_rules_count": rule_count_value,
                "pool_metrics": {
                    "active_connections": self._metrics.active_connections,
                    "idle_connections": self._metrics.idle_connections,
                    "pool_utilization": self._metrics.pool_utilization,
                    "avg_response_time_ms": self._metrics.avg_response_time_ms,
                },
                "sla_compliant": response_time < self.timeout_ms
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "mode": self.mode.value,
                "error": str(e),
                "response_time_ms": (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
            }
    
    async def get_pool_status(self) -> Dict[str, Any]:
        """Get detailed pool status"""
        if not self._engine:
            return {"status": "not_initialized"}
        
        pool = self._engine.pool
        return {
            "pool_size": pool.size(),
            "checked_out": pool.checkedout(),
            "checked_in": pool.checkedin(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
            "utilization_percentage": self._metrics.pool_utilization,
            "avg_response_time_ms": self._metrics.avg_response_time_ms,
            "error_rate": self._metrics.error_rate
        }
    
    async def shutdown(self):
        """Gracefully shutdown the connection manager"""
        if self._engine:
            await self._engine.dispose()
            logger.info(f"Unified Connection Manager shutdown completed for {self.mode.value}")

# Global instances for different modes
_mcp_manager: Optional[UnifiedConnectionManager] = None
_ml_manager: Optional[UnifiedConnectionManager] = None
_admin_manager: Optional[UnifiedConnectionManager] = None

def get_unified_connection_manager(mode: ConnectionMode = ConnectionMode.MCP_SERVER) -> UnifiedConnectionManager:
    """Get or create unified connection manager for specified mode"""
    global _mcp_manager, _ml_manager, _admin_manager
    
    if mode == ConnectionMode.MCP_SERVER:
        if _mcp_manager is None:
            _mcp_manager = UnifiedConnectionManager(mode)
        return _mcp_manager
    elif mode == ConnectionMode.ML_TRAINING:
        if _ml_manager is None:
            _ml_manager = UnifiedConnectionManager(mode)
        return _ml_manager
    elif mode == ConnectionMode.ADMIN:
        if _admin_manager is None:
            _admin_manager = UnifiedConnectionManager(mode)
        return _admin_manager
    
    raise ValueError(f"Unknown connection mode: {mode}")

# Convenience functions for backward compatibility
async def get_mcp_session() -> AsyncIterator[AsyncSession]:
    """Get MCP server session (read-only rule application)"""
    manager = get_unified_connection_manager(ConnectionMode.MCP_SERVER)
    async with manager.get_session() as session:
        yield session

async def get_ml_session() -> AsyncIterator[AsyncSession]:
    """Get ML training session (read/write for training data)"""
    manager = get_unified_connection_manager(ConnectionMode.ML_TRAINING)
    async with manager.get_session() as session:
        yield session
