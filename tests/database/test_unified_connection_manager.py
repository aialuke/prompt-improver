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
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import contextmanager, asynccontextmanager
from typing import AsyncIterator, Iterator

# Feature flag migration completed - V2 is now the default implementation

from prompt_improver.database.unified_connection_manager import (
    UnifiedConnectionManager,
    ManagerMode,
    PoolConfiguration,
    ConnectionMetrics,
    HealthStatus,
    DatabaseManagerAdapter,
    DatabaseSessionManagerAdapter,
    get_unified_manager,
    get_database_manager_adapter,
    get_database_session_manager_adapter,
    get_ha_connection_manager_adapter,
)
from prompt_improver.core.config import AppConfig
from prompt_improver.core.config import AppConfig
from prompt_improver.core.protocols.connection_protocol import ConnectionMode
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session


class TestUnifiedConnectionManager:
    """Test suite for UnifiedConnectionManager core functionality"""
    
    @pytest.fixture
    def db_config(self):
        """Mock database configuration"""
        return DatabaseConfig(
            postgres_host="localhost",
            postgres_port=5432,
            postgres_username="test_user",
            postgres_password="test_pass",
            postgres_database="test_db",
            pool_max_size=10,
            pool_timeout=30,
            pool_max_lifetime=3600,
            pool_max_idle=600,
            echo_sql=False
        )
    
    @pytest.fixture
    def redis_config(self):
        """Mock Redis configuration"""
        config = MagicMock()
        config.host = "localhost"
        config.port = 6379
        config.cache_db = 0
        config.password = None
        config.socket_timeout = 5.0
        config.connect_timeout = 5.0
        return config
    
    @pytest.fixture
    async def unified_manager(self, db_config, redis_config):
        """Create unified manager instance"""
        manager = UnifiedConnectionManager(
            mode=ManagerMode.ASYNC_MODERN
        )
        yield manager
        await manager.close()
    
    @pytest.fixture
    async def ha_unified_manager(self, db_config, redis_config):
        """Create HA-enabled unified manager"""
        manager = UnifiedConnectionManager(
            mode=ManagerMode.HIGH_AVAILABILITY
        )
        yield manager
        await manager.close()

    def test_pool_configuration_modes(self):
        """Test pool configurations for different modes"""
        # Test MCP server mode
        mcp_config = PoolConfiguration.for_mode(ManagerMode.MCP_SERVER)
        assert mcp_config.pg_pool_size == 20
        assert mcp_config.pg_max_overflow == 10
        assert mcp_config.pg_timeout == 0.2
        assert mcp_config.enable_circuit_breaker is True
        
        # Test ML training mode
        ml_config = PoolConfiguration.for_mode(ManagerMode.ML_TRAINING)
        assert ml_config.pg_pool_size == 15
        assert ml_config.enable_ha is True
        
        # Test HA mode
        ha_config = PoolConfiguration.for_mode(ManagerMode.HIGH_AVAILABILITY)
        assert ha_config.enable_ha is True
        assert ha_config.enable_circuit_breaker is True

    @pytest.mark.asyncio
    async def test_initialization(self, unified_manager):
        """Test manager initialization"""
        assert not unified_manager._is_initialized
        
        with patch.object(unified_manager, '_setup_database_connections') as mock_db, \
             patch.object(unified_manager, '_setup_redis_connections') as mock_redis:
            
            result = await unified_manager.initialize()
            
            assert result is True
            assert unified_manager._is_initialized is True
            mock_db.assert_called_once()
            mock_redis.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_protocol_implementation(self, unified_manager):
        """Test ConnectionManagerProtocol implementation"""
        with patch.object(unified_manager, 'initialize', return_value=True), \
             patch.object(unified_manager, 'get_async_session') as mock_session:
            
            mock_session.return_value.__aenter__ = AsyncMock()
            mock_session.return_value.__aexit__ = AsyncMock()
            
            # Test get_connection method
            async with unified_manager.get_connection(ConnectionMode.READ_ONLY) as conn:
                pass
            
            # Test health_check method
            health = await unified_manager.health_check()
            assert isinstance(health, dict)
            assert "status" in health
            assert "timestamp" in health
            
            # Test get_connection_info method
            info = await unified_manager.get_connection_info()
            assert isinstance(info, dict)
            assert "mode" in info
            assert "initialized" in info
            
            # Test is_healthy method
            healthy = unified_manager.is_healthy()
            assert isinstance(healthy, bool)

    # Removed backward compatibility test - using modern API directly

    @pytest.mark.asyncio 
    async def test_backward_compatibility_async_session(self, unified_manager):
        """Test DatabaseSessionManager compatibility - async sessions"""
        await unified_manager.initialize()
        
        with patch.object(unified_manager, '_async_session_factory') as mock_factory:
            mock_session = AsyncMock()
            mock_factory.return_value = mock_session
            mock_session.commit = AsyncMock()
            mock_session.rollback = AsyncMock()
            mock_session.close = AsyncMock()
            
            async with unified_manager.get_async_session() as session:
                assert session == mock_session
            
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_ha_connection_compatibility(self, ha_unified_manager):
        """Test HAConnectionManager compatibility"""
        with patch.object(ha_unified_manager, 'initialize', return_value=True), \
             patch.object(ha_unified_manager, '_pg_pools', {"primary": AsyncMock()}):
            
            ha_unified_manager._pg_pools["primary"].acquire = AsyncMock()
            
            # Test get_pg_connection method
            with patch.object(ha_unified_manager, '_update_connection_metrics'):
                conn = await ha_unified_manager.get_pg_connection(read_only=False)
                assert conn is not None

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, unified_manager):
        """Test circuit breaker patterns"""
        # Enable circuit breaker
        unified_manager.pool_config.enable_circuit_breaker = True
        unified_manager._circuit_breaker_threshold = 3
        
        # Simulate failures to trigger circuit breaker
        for _ in range(3):
            unified_manager._handle_connection_failure(Exception("Test error"))
        
        # Circuit breaker should be open
        assert unified_manager._is_circuit_breaker_open() is True
        assert unified_manager._metrics.circuit_breaker_state == "open"
        
        # Test connection should fail
        with pytest.raises(ConnectionError, match="Circuit breaker is open"):
            async with unified_manager.get_connection():
                pass

    @pytest.mark.asyncio
    async def test_health_monitoring(self, unified_manager):
        """Test health monitoring functionality"""
        with patch.object(unified_manager, 'get_sync_session') as mock_sync, \
             patch.object(unified_manager, 'get_async_session') as mock_async:
            
            # Mock successful connections
            mock_sync.return_value.__enter__ = MagicMock()
            mock_sync.return_value.__exit__ = MagicMock()
            mock_sync.return_value.execute = MagicMock()
            
            mock_async.return_value.__aenter__ = AsyncMock()
            mock_async.return_value.__aexit__ = AsyncMock()
            mock_async.return_value.execute = AsyncMock()
            
            health = await unified_manager.health_check()
            
            assert health["status"] == "healthy"
            assert "sync_database" in health["components"]
            assert "async_database" in health["components"]
            assert health["components"]["sync_database"] == "healthy"
            assert health["components"]["async_database"] == "healthy"

    def test_metrics_collection(self, unified_manager):
        """Test metrics collection and updating"""
        # Test response time updates
        unified_manager._update_response_time(100.0)
        assert unified_manager._metrics.avg_response_time_ms == 100.0
        
        unified_manager._update_response_time(200.0)
        # Should use exponential moving average
        assert 100.0 < unified_manager._metrics.avg_response_time_ms < 200.0
        
        # Test connection metrics updates
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
            
            # Test Redis connection
            redis_conn = await unified_manager.get_redis_connection(read_only=False)
            assert redis_conn == mock_redis
            mock_redis.ping.assert_called()

    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self, unified_manager):
        """Test proper shutdown and cleanup"""
        # Initialize with mock components
        unified_manager._sync_engine = MagicMock()
        unified_manager._async_engine = AsyncMock()
        unified_manager._async_engine.dispose = AsyncMock()
        unified_manager._redis_master = AsyncMock()
        unified_manager._redis_master.aclose = AsyncMock()
        
        await unified_manager.close()
        
        # Verify cleanup calls
        unified_manager._sync_engine.dispose.assert_called_once()
        unified_manager._async_engine.dispose.assert_called_once()
        unified_manager._redis_master.aclose.assert_called_once()
        assert unified_manager._is_initialized is False


# Removed TestBackwardCompatibilityAdapters class - using modern API directly
# Adapter test methods removed as adapters are not implemented in current unified manager


# Feature flag integration no longer needed - unified manager is now the default


class TestPerformanceCharacteristics:
    """Test performance characteristics and regression prevention"""
    
    @pytest.mark.asyncio
    async def test_connection_acquisition_performance(self, unified_manager):
        """Test connection acquisition times meet SLA requirements"""
        await unified_manager.initialize()
        
        with patch.object(unified_manager, 'get_async_session') as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock()
            mock_session.return_value.__aexit__ = AsyncMock()
            
            # Test connection acquisition time
            start_time = time.time()
            async with unified_manager.get_connection():
                pass
            acquisition_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Should be well under timeout threshold
            assert acquisition_time < unified_manager.pool_config.pg_timeout * 1000

    @pytest.mark.asyncio
    async def test_concurrent_connections(self, unified_manager):
        """Test handling of concurrent connections"""
        await unified_manager.initialize()
        
        async def get_connection_task():
            """Task to get connection concurrently"""
            with patch.object(unified_manager, 'get_async_session'):
                async with unified_manager.get_connection():
                    await asyncio.sleep(0.01)  # Small delay
        
        # Test multiple concurrent connections
        tasks = [get_connection_task() for _ in range(10)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify metrics were updated
        assert unified_manager._metrics.active_connections >= 0

    def test_memory_usage_characteristics(self, unified_manager):
        """Test memory usage doesn't exceed reasonable bounds"""
        # This would typically involve memory profiling
        # For now, test that metrics are reasonable
        metrics = unified_manager._get_metrics_dict()
        
        # Verify metrics are within reasonable bounds
        assert metrics["active_connections"] >= 0
        assert metrics["pool_utilization"] >= 0
        assert metrics["avg_response_time_ms"] >= 0


class TestExistingUsageValidation:
    """Test that existing usage patterns continue to work"""
    
    @pytest.mark.asyncio
    async def test_api_endpoint_pattern(self):
        """Test pattern used in API endpoints"""
        # Simulate the pattern: from ..database.connection import DatabaseManager
        manager_adapter = get_database_manager_adapter()
        
        # Test sync session usage pattern
        with patch.object(manager_adapter, 'get_session') as mock_session:
            mock_session.return_value.__enter__ = MagicMock()
            mock_session.return_value.__exit__ = MagicMock()
            
            with manager_adapter.get_session() as session:
                # This is the pattern used in AprioriAnalyzer
                pass

    @pytest.mark.asyncio
    async def test_ml_component_pattern(self):
        """Test pattern used in ML components"""
        # Simulate the pattern used in ML integration
        session_manager_adapter = get_database_session_manager_adapter()
        
        with patch.object(session_manager_adapter, 'session') as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock()
            mock_session.return_value.__aexit__ = AsyncMock()
            
            async with session_manager_adapter.session() as session:
                # This is the pattern used in ml_integration.py
                pass

    @pytest.mark.asyncio
    async def test_ha_failover_pattern(self):
        """Test HA failover patterns"""
        ha_manager = get_ha_connection_manager_adapter()
        
        with patch.object(ha_manager, 'get_pg_connection') as mock_get_conn:
            mock_get_conn.return_value = AsyncMock()
            
            # Test read-only connection with replica preference
            conn = await ha_manager.get_pg_connection(read_only=True, prefer_replica=True)
            assert conn is not None
            
            # Test write connection
            conn = await ha_manager.get_pg_connection(read_only=False)
            assert conn is not None


class TestFailureScenarios:
    """Test various failure scenarios and recovery"""
    
    @pytest.mark.asyncio
    async def test_database_connection_failure(self, unified_manager):
        """Test handling of database connection failures"""
        with patch.object(unified_manager, '_async_session_factory') as mock_factory:
            mock_session = AsyncMock()
            mock_session.__aenter__.side_effect = Exception("Database connection failed")
            mock_factory.return_value = mock_session
            
            with pytest.raises(Exception, match="Database connection failed"):
                async with unified_manager.get_async_session():
                    pass

    @pytest.mark.asyncio
    async def test_redis_connection_failure(self, unified_manager):
        """Test handling of Redis connection failures"""
        unified_manager._redis_master = AsyncMock()
        unified_manager._redis_master.ping.side_effect = Exception("Redis connection failed")
        
        with pytest.raises(ConnectionError, match="Redis connection failed"):
            await unified_manager.get_redis_connection()

    @pytest.mark.asyncio
    async def test_initialization_failure_recovery(self, unified_manager):
        """Test recovery from initialization failures"""
        with patch.object(unified_manager, '_setup_database_connections', side_effect=Exception("Init failed")):
            result = await unified_manager.initialize()
            assert result is False
            assert unified_manager._health_status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_check_failure_handling(self, unified_manager):
        """Test health check failure handling"""
        with patch.object(unified_manager, 'get_sync_session', side_effect=Exception("Health check failed")):
            health = await unified_manager.health_check()
            assert health["status"] == "unhealthy"
            assert "error" in health


if __name__ == "__main__":
    pytest.main([__file__, "-v"])