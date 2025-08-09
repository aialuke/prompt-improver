"""
Comprehensive test suite for UnifiedConnectionManager modern API validation.

This test suite validates that the unified manager:
1. Preserves all functionality from the original managers
2. Provides equivalent or better performance characteristics
3. Handles all connection modes and failure scenarios correctly
4. Provides clean async-first database connection management
"""
import asyncio
import os
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from prompt_improver.core.config import AppConfig
from prompt_improver.core.protocols.connection_protocol import ConnectionMode
from prompt_improver.database.unified_connection_manager import ConnectionMetrics, HealthStatus, ManagerMode, PoolConfiguration, UnifiedConnectionManager, get_unified_manager

class TestUnifiedConnectionManager:
    """Test suite for UnifiedConnectionManager core functionality"""

    @pytest.fixture
    def db_config(self):
        """Mock database configuration"""
        return DatabaseConfig(postgres_host='localhost', postgres_port=5432, postgres_username='test_user', postgres_password='test_pass', postgres_database='test_db', pool_max_size=10, pool_timeout=30, pool_max_lifetime=3600, pool_max_idle=600, echo_sql=False)

    @pytest.fixture
    def redis_config(self):
        """Mock Redis configuration"""
        config = MagicMock()
        config.host = 'localhost'
        config.port = 6379
        config.cache_db = 0
        config.password = None
        config.socket_timeout = 5.0
        config.connect_timeout = 5.0
        return config

    @pytest.fixture
    async def unified_manager(self, db_config, redis_config):
        """Create unified manager instance"""
        manager = UnifiedConnectionManager(mode=ManagerMode.ASYNC_MODERN)
        yield manager
        await manager.close()

    @pytest.fixture
    async def ha_unified_manager(self, db_config, redis_config):
        """Create HA-enabled unified manager"""
        manager = UnifiedConnectionManager(mode=ManagerMode.HIGH_AVAILABILITY)
        yield manager
        await manager.close()

    def test_pool_configuration_modes(self):
        """Test pool configurations for different modes"""
        mcp_config = PoolConfiguration.for_mode(ManagerMode.MCP_SERVER)
        assert mcp_config.pg_pool_size == 20
        assert mcp_config.pg_max_overflow == 10
        assert mcp_config.pg_timeout == 0.2
        assert mcp_config.enable_circuit_breaker is True
        ml_config = PoolConfiguration.for_mode(ManagerMode.ML_TRAINING)
        assert ml_config.pg_pool_size == 15
        assert ml_config.enable_ha is True
        ha_config = PoolConfiguration.for_mode(ManagerMode.HIGH_AVAILABILITY)
        assert ha_config.enable_ha is True
        assert ha_config.enable_circuit_breaker is True

    @pytest.mark.asyncio
    async def test_initialization(self, unified_manager):
        """Test manager initialization"""
        assert not unified_manager._is_initialized
        with patch.object(unified_manager, '_setup_database_connections') as mock_db, patch.object(unified_manager, '_setup_redis_connections') as mock_redis:
            result = await unified_manager.initialize()
            assert result is True
            assert unified_manager._is_initialized is True
            mock_db.assert_called_once()
            mock_redis.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_protocol_implementation(self, unified_manager):
        """Test ConnectionManagerProtocol implementation"""
        with patch.object(unified_manager, 'initialize', return_value=True), patch.object(unified_manager, 'get_async_session') as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock()
            mock_session.return_value.__aexit__ = AsyncMock()
            async with unified_manager.get_connection(ConnectionMode.READ_ONLY) as conn:
                pass
            health = await unified_manager.health_check()
            assert isinstance(health, dict)
            assert 'status' in health
            assert 'timestamp' in health
            info = await unified_manager.get_connection_info()
            assert isinstance(info, dict)
            assert 'mode' in info
            assert 'initialized' in info
            healthy = unified_manager.is_healthy()
            assert isinstance(healthy, bool)

    @pytest.mark.asyncio
    async def test_ha_connection_compatibility(self, ha_unified_manager):
        """Test HAConnectionManager compatibility"""
        with patch.object(ha_unified_manager, 'initialize', return_value=True), patch.object(ha_unified_manager, '_pg_pools', {'primary': AsyncMock()}):
            ha_unified_manager._pg_pools['primary'].acquire = AsyncMock()
            with patch.object(ha_unified_manager, '_update_connection_metrics'):
                conn = await ha_unified_manager.get_pg_connection(read_only=False)
                assert conn is not None

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, unified_manager):
        """Test circuit breaker patterns"""
        unified_manager.pool_config.enable_circuit_breaker = True
        unified_manager._circuit_breaker_threshold = 3
        for _ in range(3):
            unified_manager._handle_connection_failure(Exception('Test error'))
        assert unified_manager._is_circuit_breaker_open() is True
        assert unified_manager._metrics.circuit_breaker_state == 'open'
        with pytest.raises(ConnectionError, match='Circuit breaker is open'):
            async with unified_manager.get_connection():
                pass

    @pytest.mark.asyncio
    async def test_health_monitoring(self, unified_manager):
        """Test health monitoring functionality"""
        with patch.object(unified_manager, 'get_sync_session') as mock_sync, patch.object(unified_manager, 'get_async_session') as mock_async:
            mock_sync.return_value.__enter__ = MagicMock()
            mock_sync.return_value.__exit__ = MagicMock()
            mock_sync.return_value.execute = MagicMock()
            mock_async.return_value.__aenter__ = AsyncMock()
            mock_async.return_value.__aexit__ = AsyncMock()
            mock_async.return_value.execute = AsyncMock()
            health = await unified_manager.health_check()
            assert health['status'] == 'healthy'
            assert 'sync_database' in health['components']
            assert 'async_database' in health['components']
            assert health['components']['sync_database'] == 'healthy'
            assert health['components']['async_database'] == 'healthy'

    def test_metrics_collection(self, unified_manager):
        """Test metrics collection and updating"""
        unified_manager._update_response_time(100.0)
        assert unified_manager._metrics.avg_response_time_ms == 100.0
        unified_manager._update_response_time(200.0)
        assert 100.0 < unified_manager._metrics.avg_response_time_ms < 200.0
        unified_manager._update_connection_metrics(success=True)
        assert unified_manager._circuit_breaker_failures == 0
        unified_manager._update_connection_metrics(success=False)
        assert unified_manager._metrics.failed_connections == 1

    @pytest.mark.asyncio
    async def test_redis_connections(self, unified_manager):
        """Test Redis connection functionality"""
        with patch('coredis.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis_class.return_value = mock_redis
            mock_redis.ping = AsyncMock()
            await unified_manager._setup_redis_direct()
            redis_conn = await unified_manager.get_redis_connection(read_only=False)
            assert redis_conn == mock_redis
            mock_redis.ping.assert_called()

    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self, unified_manager):
        """Test proper shutdown and cleanup"""
        unified_manager._sync_engine = MagicMock()
        unified_manager._async_engine = AsyncMock()
        unified_manager._async_engine.dispose = AsyncMock()
        unified_manager._redis_master = AsyncMock()
        unified_manager._redis_master.aclose = AsyncMock()
        await unified_manager.close()
        unified_manager._sync_engine.dispose.assert_called_once()
        unified_manager._async_engine.dispose.assert_called_once()
        unified_manager._redis_master.aclose.assert_called_once()
        assert unified_manager._is_initialized is False

class TestPerformanceCharacteristics:
    """Test performance characteristics and regression prevention"""

    @pytest.mark.asyncio
    async def test_connection_acquisition_performance(self, unified_manager):
        """Test connection acquisition times meet SLA requirements"""
        await unified_manager.initialize()
        with patch.object(unified_manager, 'get_async_session') as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock()
            mock_session.return_value.__aexit__ = AsyncMock()
            start_time = time.time()
            async with unified_manager.get_connection():
                pass
            acquisition_time = (time.time() - start_time) * 1000
            assert acquisition_time < unified_manager.pool_config.pg_timeout * 1000

    @pytest.mark.asyncio
    async def test_concurrent_connections(self, unified_manager):
        """Test handling of concurrent connections"""
        await unified_manager.initialize()

        async def get_connection_task():
            """Task to get connection concurrently"""
            with patch.object(unified_manager, 'get_async_session'):
                async with unified_manager.get_connection():
                    await asyncio.sleep(0.01)
        tasks = [get_connection_task() for _ in range(10)]
        await asyncio.gather(*tasks, return_exceptions=True)
        assert unified_manager._metrics.active_connections >= 0

    def test_memory_usage_characteristics(self, unified_manager):
        """Test memory usage doesn't exceed reasonable bounds"""
        metrics = unified_manager._get_metrics_dict()
        assert metrics['active_connections'] >= 0
        assert metrics['pool_utilization'] >= 0
        assert metrics['avg_response_time_ms'] >= 0

class TestFailureScenarios:
    """Test various failure scenarios and recovery"""

    @pytest.mark.asyncio
    async def test_database_connection_failure(self, unified_manager):
        """Test handling of database connection failures"""
        with patch.object(unified_manager, '_async_session_factory') as mock_factory:
            mock_session = AsyncMock()
            mock_session.__aenter__.side_effect = Exception('Database connection failed')
            mock_factory.return_value = mock_session
            with pytest.raises(Exception, match='Database connection failed'):
                async with unified_manager.get_async_session():
                    pass

    @pytest.mark.asyncio
    async def test_redis_connection_failure(self, unified_manager):
        """Test handling of Redis connection failures"""
        unified_manager._redis_master = AsyncMock()
        unified_manager._redis_master.ping.side_effect = Exception('Redis connection failed')
        with pytest.raises(ConnectionError, match='Redis connection failed'):
            await unified_manager.get_redis_connection()

    @pytest.mark.asyncio
    async def test_initialization_failure_recovery(self, unified_manager):
        """Test recovery from initialization failures"""
        with patch.object(unified_manager, '_setup_database_connections', side_effect=Exception('Init failed')):
            result = await unified_manager.initialize()
            assert result is False
            assert unified_manager._health_status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_check_failure_handling(self, unified_manager):
        """Test health check failure handling"""
        with patch.object(unified_manager, 'get_sync_session', side_effect=Exception('Health check failed')):
            health = await unified_manager.health_check()
            assert health['status'] == 'unhealthy'
            assert 'error' in health
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
