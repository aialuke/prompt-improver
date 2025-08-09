"""Database Test Adapter Interface
Provides controlled database access for performance tests and diagnostics
while maintaining unified connection management for production.

This adapter ensures that performance tests can use direct connections
when needed for accurate benchmarking, while production code benefits
from the unified connection manager's pooling and health monitoring.
"""
import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Dict, Optional
import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession
from prompt_improver.core.config import AppConfig
from prompt_improver.database.unified_connection_manager import ManagerMode, get_unified_manager
logger = logging.getLogger(__name__)

@dataclass
class TestConnectionConfig:
    """Configuration for test database connections."""
    host: str = 'localhost'
    port: int = 5432
    user: str = 'apes_user'
    password: str = 'apes_secure_password_2024'
    database: str = 'apes_production'
    timeout: float = 5.0
    max_connections: int = 5

class DatabaseTestAdapter:
    """Adapter interface for database testing that bridges unified connection
    management with direct connection needs for performance benchmarking.

    Features:
    - Production uses UnifiedConnectionManager for optimal performance
    - Test isolation uses direct connections when needed for accurate benchmarking
    - Automatic fallback between connection methods
    - Health check compatibility across both patterns
    """

    def __init__(self, config: TestConnectionConfig | None=None):
        """Initialize the test adapter with optional configuration."""
        self.config = config or TestConnectionConfig()
        self._manager = None

    async def get_manager(self) -> 'UnifiedConnectionManager':
        """Get or create the unified connection manager."""
        if not self._manager:
            self._manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
        return self._manager

    @contextlib.asynccontextmanager
    async def get_production_session(self) -> AsyncIterator[AsyncSession]:
        """Get database session using unified connection manager.
        Recommended for all production code and integration tests.
        """
        manager = await self.get_manager()
        async with manager.get_async_session() as session:
            yield session

    @contextlib.asynccontextmanager
    async def get_direct_connection(self) -> AsyncIterator[asyncpg.Connection]:
        """Get direct asyncpg connection for performance testing.
        Use only when you need to measure raw connection performance
        without the overhead of connection pooling and session management.
        """
        conn = await asyncpg.connect(host=self.config.host, port=self.config.port, user=self.config.user, password=self.config.password, database=self.config.database, timeout=self.config.timeout)
        try:
            yield conn
        finally:
            await conn.close()

    async def health_check_unified(self) -> dict[str, Any]:
        """Perform health check using unified connection manager.
        Returns detailed health information including pool status.
        """
        try:
            manager = await self.get_manager()
            health_info = await manager.get_health_info()
            return {'method': 'unified_manager', 'status': health_info.get('status', 'unknown'), 'details': health_info, 'recommended': True}
        except Exception as e:
            logger.error('Unified health check failed: %s', e)
            return {'method': 'unified_manager', 'status': 'failed', 'error': str(e), 'recommended': True}

    async def health_check_direct(self) -> dict[str, Any]:
        """Perform health check using direct connection.
        Use for fallback verification or direct connection benchmarking.
        """
        try:
            start_time = asyncio.get_event_loop().time()
            async with self.get_direct_connection() as conn:
                result = await conn.fetchval('SELECT 1')
                connection_time = (asyncio.get_event_loop().time() - start_time) * 1000
                return {'method': 'direct_connection', 'status': 'healthy' if result == 1 else 'degraded', 'connection_time_ms': round(connection_time, 2), 'recommended': False}
        except Exception as e:
            logger.error('Direct health check failed: %s', e)
            return {'method': 'direct_connection', 'status': 'failed', 'error': str(e), 'recommended': False}

    async def comprehensive_health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check using both methods.
        Useful for diagnostic tools and performance comparison.
        """
        unified_result = await self.health_check_unified()
        direct_result = await self.health_check_direct()
        overall_status = 'healthy'
        if unified_result['status'] == 'failed' and direct_result['status'] == 'failed':
            overall_status = 'failed'
        elif unified_result['status'] in ['failed', 'degraded'] or direct_result['status'] in ['failed', 'degraded']:
            overall_status = 'degraded'
        return {'overall_status': overall_status, 'unified_manager': unified_result, 'direct_connection': direct_result, 'recommendation': 'Use unified_manager for production, direct_connection only for performance testing'}

    async def benchmark_connection_methods(self, iterations: int=10) -> dict[str, Any]:
        """Benchmark both connection methods for performance comparison.
        Useful for validating the performance improvements from consolidation.
        """
        unified_times = []
        direct_times = []
        for _ in range(iterations):
            start_time = asyncio.get_event_loop().time()
            try:
                async with self.get_production_session() as session:
                    await session.execute('SELECT 1')
                elapsed = (asyncio.get_event_loop().time() - start_time) * 1000
                unified_times.append(elapsed)
            except Exception as e:
                logger.error('Unified benchmark iteration failed: %s', e)
        for _ in range(iterations):
            start_time = asyncio.get_event_loop().time()
            try:
                async with self.get_direct_connection() as conn:
                    await conn.fetchval('SELECT 1')
                elapsed = (asyncio.get_event_loop().time() - start_time) * 1000
                direct_times.append(elapsed)
            except Exception as e:
                logger.error('Direct benchmark iteration failed: %s', e)
        return {'iterations': iterations, 'unified_manager': {'times_ms': unified_times, 'avg_ms': round(sum(unified_times) / len(unified_times), 2) if unified_times else 0, 'min_ms': round(min(unified_times), 2) if unified_times else 0, 'max_ms': round(max(unified_times), 2) if unified_times else 0}, 'direct_connection': {'times_ms': direct_times, 'avg_ms': round(sum(direct_times) / len(direct_times), 2) if direct_times else 0, 'min_ms': round(min(direct_times), 2) if direct_times else 0, 'max_ms': round(max(direct_times), 2) if direct_times else 0}, 'performance_ratio': round(sum(direct_times) / len(direct_times) / (sum(unified_times) / len(unified_times)), 2) if unified_times and direct_times else 0}

async def get_test_adapter(config: TestConnectionConfig | None=None) -> DatabaseTestAdapter:
    """Get configured database test adapter."""
    return DatabaseTestAdapter(config)

@contextlib.asynccontextmanager
async def production_database_session() -> AsyncIterator[AsyncSession]:
    """Convenience function to get production database session.
    Uses unified connection manager for optimal performance and monitoring.
    """
    adapter = await get_test_adapter()
    async with adapter.get_production_session() as session:
        yield session

@contextlib.asynccontextmanager
async def benchmark_database_connection() -> AsyncIterator[asyncpg.Connection]:
    """Convenience function to get direct database connection for benchmarking.
    Use only for performance testing and diagnostics.
    """
    adapter = await get_test_adapter()
    async with adapter.get_direct_connection() as conn:
        yield conn
__all__ = ['DatabaseTestAdapter', 'TestConnectionConfig', 'benchmark_database_connection', 'get_test_adapter', 'production_database_session']
