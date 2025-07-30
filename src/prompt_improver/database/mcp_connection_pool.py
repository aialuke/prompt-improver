"""
Unified Database Connection Pool for MCP Server (Phase 0)
Optimized for mixed read/write workload with permission-based security
"""

import contextlib
import logging
import os
from collections.abc import AsyncIterator
from typing import Optional

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ..core.config import AppConfig

logger = logging.getLogger(__name__)

class MCPConnectionPool:
    """
    Unified connection pool for MCP server with mixed read/write workload optimization.
    
    Implements Phase 0 requirements:
    - 20 connections with 10 max overflow
    - Permission-based security (mcp_server_user)
    - <200ms response time targets
    - Mixed read/write workload optimization
    """
    
    def __init__(self, mcp_user_password: Optional[str] = None):
        self.config = AppConfig().database
        
        # Get MCP-specific environment variables
        self.mcp_user_password = mcp_user_password or os.getenv("MCP_POSTGRES_PASSWORD")
        if not self.mcp_user_password:
            raise ValueError(
                "MCP_POSTGRES_PASSWORD environment variable must be set for MCP server database access"
            )
        
        # MCP server connection parameters from Phase 0 specification
        self.pool_size = int(os.getenv("MCP_DB_POOL_SIZE", "20"))
        self.max_overflow = int(os.getenv("MCP_DB_MAX_OVERFLOW", "10"))
        self.timeout_ms = int(os.getenv("MCP_REQUEST_TIMEOUT_MS", "200"))
        
        # Build MCP server database URL with limited permissions user
        self.database_url = (
            f"postgresql+psycopg://mcp_server_user:{self.mcp_user_password}@"
            f"{self.config.host}:{self.config.port}/"
            f"{self.config.database}"
        )
        
        # Create optimized async engine for MCP workload
        self._engine = self._create_mcp_engine()
        
        # Create session factory
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False
        )
        
        logger.info(
            f"MCP Connection Pool initialized: {self.pool_size} connections, "
            f"{self.max_overflow} overflow, {self.timeout_ms}ms timeout target"
        )
    
    def _create_mcp_engine(self) -> AsyncEngine:
        """Create optimized async engine for MCP server workload."""
        
        # Engine configuration optimized for Phase 0 requirements
        engine_kwargs = {
            # Connection pool configuration (for async engine)
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_pre_ping": True,  # Validate connections
            "pool_recycle": 3600,   # Recycle connections every hour
            "pool_timeout": 30,     # Connection acquisition timeout
            
            # Performance optimizations
            "echo": os.getenv("MCP_LOG_LEVEL", "INFO") == "DEBUG",
            "echo_pool": False,
            "future": True,
            
            # Connection-level timeouts aligned with <200ms SLA
            "connect_args": {
                "command_timeout": 60,  # Connection timeout
                "server_settings": {
                    "application_name": "apes_mcp_server",
                    "statement_timeout": "150ms",  # Query timeout within SLA budget
                    "idle_in_transaction_session_timeout": "300s",
                    "lock_timeout": "10s",
                }
            }
        }
        
        engine = create_async_engine(self.database_url, **engine_kwargs)
        
        # Add connection pool monitoring events
        self._setup_pool_monitoring(engine)
        
        return engine
    
    def _setup_pool_monitoring(self, engine: AsyncEngine) -> None:
        """Setup connection pool monitoring and logging."""
        
        @event.listens_for(engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Log new connections for monitoring."""
            logger.debug(f"New MCP connection established: {connection_record}")
        
        @event.listens_for(engine.sync_engine, "checkout") 
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            """Monitor connection checkout."""
            pool_impl = connection_proxy._pool
            logger.debug(
                f"MCP connection checked out: "
                f"pool_size={pool_impl.size()}, "
                f"overflow={pool_impl.overflow()}, "
                f"invalid={pool_impl.invalidated()}"
            )
        
        @event.listens_for(engine.sync_engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            """Monitor connection checkin."""
            logger.debug(f"MCP connection checked in: {connection_record}")
    
    @contextlib.asynccontextmanager
    async def get_session(self) -> AsyncIterator[AsyncSession]:
        """
        Get an async session optimized for MCP rule application.
        
        Provides automatic transaction management and error handling
        aligned with Phase 0 <200ms response time requirements.
        """
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"MCP session error: {e}")
            raise
        finally:
            await session.close()
    
    @contextlib.asynccontextmanager
    async def get_read_session(self) -> AsyncIterator[AsyncSession]:
        """
        Get a read-only session optimized for rule data access.
        
        Optimized for the MCP server's read-only access to rule tables.
        """
        session = self._session_factory()
        try:
            # Set transaction to read-only for performance
            await session.execute("SET TRANSACTION READ ONLY")
            yield session
            # Read-only transactions don't need explicit commit
        except Exception as e:
            logger.error(f"MCP read session error: {e}")
            raise
        finally:
            await session.close()
    
    @contextlib.asynccontextmanager
    async def get_feedback_session(self) -> AsyncIterator[AsyncSession]:
        """
        Get a session optimized for feedback data writes.
        
        Optimized for writing to prompt_improvement_sessions table.
        """
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"MCP feedback session error: {e}")
            raise
        finally:
            await session.close()
    
    async def get_pool_status(self) -> dict:
        """Get current connection pool status for monitoring."""
        pool_impl = self._engine.pool
        
        return {
            "pool_size": pool_impl.size(),
            "checked_in": pool_impl.checkedin(),
            "overflow": pool_impl.overflow(),
            "checked_out": pool_impl.checkedout(),
            "invalidated": pool_impl.invalidated(),
            "connection_count": pool_impl.size() + pool_impl.overflow(),
            "pool_limit": self.pool_size + self.max_overflow,
            "utilization_percentage": round(
                ((pool_impl.size() + pool_impl.overflow()) / (self.pool_size + self.max_overflow)) * 100, 2
            )
        }
    
    async def health_check(self) -> dict:
        """Perform health check on the connection pool."""
        try:
            async with self.get_session() as session:
                # Simple query to test connectivity and permissions
                result = await session.execute("SELECT 1 as health_check")
                health_result = result.scalar()
                
                if health_result == 1:
                    pool_status = await self.get_pool_status()
                    return {
                        "status": "healthy",
                        "connection_test": "passed",
                        "pool_status": pool_status,
                        "database_user": "mcp_server_user",
                        "permissions": "read_rules_write_feedback"
                    }
                else:
                    return {"status": "unhealthy", "error": "Health check query failed"}
                    
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e),
                "connection_test": "failed"
            }
    
    async def test_permissions(self) -> dict:
        """Test MCP user database permissions."""
        results = {
            "read_rule_performance": False,
            "read_rule_metadata": False,
            "write_prompt_sessions": False,
            "denied_rule_write": True,  # Should be denied
        }
        
        try:
            async with self.get_read_session() as session:
                # Test read access to rule tables
                try:
                    await session.execute("SELECT COUNT(*) FROM rule_performance LIMIT 1")
                    results["read_rule_performance"] = True
                except Exception as e:
                    logger.warning(f"Cannot read rule_performance: {e}")
                
                try:
                    await session.execute("SELECT COUNT(*) FROM rule_metadata LIMIT 1")
                    results["read_rule_metadata"] = True
                except Exception as e:
                    logger.warning(f"Cannot read rule_metadata: {e}")
            
            async with self.get_feedback_session() as session:
                # Test write access to feedback table
                try:
                    await session.execute(
                        "INSERT INTO prompt_improvement_sessions "
                        "(original_prompt, enhanced_prompt, applied_rules, response_time_ms) "
                        "VALUES ('test', 'test', '[]', 100)"
                    )
                    await session.execute(
                        "DELETE FROM prompt_improvement_sessions WHERE original_prompt = 'test'"
                    )
                    results["write_prompt_sessions"] = True
                except Exception as e:
                    logger.warning(f"Cannot write to prompt_improvement_sessions: {e}")
                
                # Test that write to rule tables is properly denied
                try:
                    await session.execute("INSERT INTO rule_performance (rule_id, rule_name) VALUES ('test', 'test')")
                    results["denied_rule_write"] = False  # This should have failed
                    logger.warning("MCP user can write to rule tables - SECURITY ISSUE!")
                except Exception:
                    # This is expected - MCP user should not be able to write to rule tables
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
    
    async def close(self) -> None:
        """Close the connection pool and all connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("MCP connection pool closed")

# Global MCP connection pool instance
_mcp_pool: Optional[MCPConnectionPool] = None

def get_mcp_connection_pool() -> MCPConnectionPool:
    """Get or create the global MCP connection pool."""
    global _mcp_pool
    
    if _mcp_pool is None:
        _mcp_pool = MCPConnectionPool()
    
    return _mcp_pool

async def get_mcp_session() -> AsyncIterator[AsyncSession]:
    """Get an MCP database session for general use."""
    pool = get_mcp_connection_pool()
    async with pool.get_session() as session:
        yield session

async def get_mcp_read_session() -> AsyncIterator[AsyncSession]:
    """Get an MCP database session optimized for reading rule data."""
    pool = get_mcp_connection_pool()
    async with pool.get_read_session() as session:
        yield session

async def get_mcp_feedback_session() -> AsyncIterator[AsyncSession]:
    """Get an MCP database session optimized for writing feedback data."""
    pool = get_mcp_connection_pool()
    async with pool.get_feedback_session() as session:
        yield session