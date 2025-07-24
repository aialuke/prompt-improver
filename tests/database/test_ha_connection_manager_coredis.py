"""Tests for HA Connection Manager with coredis migration.

This test suite validates the migration from redis.asyncio to coredis,
ensuring Redis Sentinel integration and basic Redis operations work correctly.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import logging

from src.prompt_improver.database.ha_connection_manager import HAConnectionManager
from src.prompt_improver.database.config import DatabaseConfig
from src.prompt_improver.utils.redis_cache import RedisConfig


class TestHAConnectionManagerCoredis:
    """Test suite for HA Connection Manager with coredis."""
    
    @pytest.fixture
    def db_config(self):
        """Mock database configuration."""
        return DatabaseConfig(
            postgres_host="localhost",
            postgres_port=5432,
            postgres_username="test_user",
            postgres_password="test_pass",
            postgres_database="test_db",
            pool_max_size=10,
            pool_timeout=30,
            pool_max_lifetime=3600,
            pool_max_idle=600
        )
    
    @pytest.fixture
    def redis_config(self):
        """Mock Redis configuration."""
        config = MagicMock()
        config.host = "localhost"
        config.port = 6379
        config.cache_db = 0
        config.password = None
        config.socket_timeout = 5.0
        config.connect_timeout = 5.0
        return config
    
    @pytest.fixture
    def ha_manager(self, db_config, redis_config):
        """Create HA connection manager instance."""
        return HAConnectionManager(
            db_config=db_config,
            redis_config=redis_config,
            logger=logging.getLogger("test")
        )
    
    @pytest.mark.asyncio
    async def test_initialization(self, ha_manager):
        """Test HA manager initialization."""
        assert ha_manager.redis_sentinel is None
        assert ha_manager.redis_master is None
        assert ha_manager.redis_replica is None
        assert ha_manager.failover_in_progress is False
        assert ha_manager.circuit_breaker_state == "closed"
    
    @pytest.mark.asyncio
    @patch('src.prompt_improver.database.ha_connection_manager.Sentinel')
    async def test_redis_sentinel_setup(self, mock_sentinel_class, ha_manager):
        """Test Redis Sentinel setup with coredis."""
        # Mock Sentinel instance
        mock_sentinel = MagicMock()
        mock_sentinel_class.return_value = mock_sentinel
        
        # Mock master and replica connections
        mock_master = AsyncMock()
        mock_replica = AsyncMock()
        mock_sentinel.primary_for.return_value = mock_master
        mock_sentinel.replica_for.return_value = mock_replica
        
        # Call setup method
        await ha_manager._setup_redis_sentinel()
        
        # Verify Sentinel initialization
        mock_sentinel_class.assert_called_once_with(
            sentinels=[
                ("redis-sentinel-1", 26379),
                ("redis-sentinel-2", 26379),
                ("redis-sentinel-3", 26379),
            ],
            stream_timeout=0.1,
            connect_timeout=0.1
        )
        
        # Verify primary and replica connections
        mock_sentinel.primary_for.assert_called_once_with(
            'mymaster',
            stream_timeout=0.1,
            password=None
        )
        mock_sentinel.replica_for.assert_called_once_with(
            'mymaster',
            stream_timeout=0.1,
            password=None
        )
        
        assert ha_manager.redis_sentinel == mock_sentinel
        assert ha_manager.redis_master == mock_master
        assert ha_manager.redis_replica == mock_replica
    
    @pytest.mark.asyncio
    @patch('src.prompt_improver.database.ha_connection_manager.coredis.Redis')
    async def test_redis_fallback_setup(self, mock_redis_class, ha_manager):
        """Test Redis fallback connection setup."""
        # Mock Redis instance
        mock_redis = AsyncMock()
        mock_redis_class.return_value = mock_redis
        
        # Call fallback setup
        await ha_manager._setup_redis_fallback()
        
        # Verify Redis client initialization
        mock_redis_class.assert_called_once_with(
            host=ha_manager.redis_config.host,
            port=ha_manager.redis_config.port,
            db=ha_manager.redis_config.cache_db,
            password=None,
            stream_timeout=ha_manager.redis_config.socket_timeout,
            connect_timeout=ha_manager.redis_config.connect_timeout
        )
        
        assert ha_manager.redis_master == mock_redis
    
    @pytest.mark.asyncio
    async def test_get_redis_connection_master(self, ha_manager):
        """Test getting Redis master connection."""
        # Mock master connection
        mock_master = AsyncMock()
        mock_master.ping = AsyncMock()
        ha_manager.redis_master = mock_master
        
        # Get connection
        connection = await ha_manager.get_redis_connection(read_only=False)
        
        # Verify ping was called and connection returned
        mock_master.ping.assert_called_once()
        assert connection == mock_master
    
    @pytest.mark.asyncio
    async def test_get_redis_connection_replica(self, ha_manager):
        """Test getting Redis replica connection for read-only operations."""
        # Mock replica and master connections
        mock_replica = AsyncMock()
        mock_replica.ping = AsyncMock()
        mock_master = AsyncMock()
        mock_master.ping = AsyncMock()
        
        ha_manager.redis_replica = mock_replica
        ha_manager.redis_master = mock_master
        
        # Get read-only connection
        connection = await ha_manager.get_redis_connection(read_only=True)
        
        # Verify replica was tried first
        mock_replica.ping.assert_called_once()
        assert connection == mock_replica
    
    @pytest.mark.asyncio
    async def test_get_redis_connection_replica_fallback(self, ha_manager):
        """Test fallback to master when replica fails."""
        # Mock failing replica and working master
        mock_replica = AsyncMock()
        mock_replica.ping = AsyncMock(side_effect=Exception("Replica down"))
        mock_master = AsyncMock()
        mock_master.ping = AsyncMock()
        
        ha_manager.redis_replica = mock_replica
        ha_manager.redis_master = mock_master
        
        # Get read-only connection
        connection = await ha_manager.get_redis_connection(read_only=True)
        
        # Verify replica was tried first, then master
        mock_replica.ping.assert_called_once()
        mock_master.ping.assert_called_once()
        assert connection == mock_master
    
    @pytest.mark.asyncio
    async def test_get_redis_connection_no_connections(self, ha_manager):
        """Test error when no Redis connections available."""
        # No connections set
        ha_manager.redis_master = None
        ha_manager.redis_replica = None
        
        # Should raise ConnectionError
        with pytest.raises(ConnectionError, match="No Redis connections available"):
            await ha_manager.get_redis_connection()
    
    @pytest.mark.asyncio
    async def test_redis_health_check(self, ha_manager):
        """Test Redis health check functionality."""
        # Mock master connection
        mock_master = AsyncMock()
        mock_master.ping = AsyncMock()
        ha_manager.redis_master = mock_master
        
        # Run health check
        await ha_manager._check_redis_health()
        
        # Verify ping was called
        mock_master.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_redis_health_check_failure(self, ha_manager):
        """Test Redis health check failure handling."""
        # Mock master connection that fails ping
        mock_master = AsyncMock()
        mock_master.ping = AsyncMock(side_effect=Exception("Connection failed"))
        ha_manager.redis_master = mock_master
        
        # Run health check
        await ha_manager._check_redis_health()
        
        # Verify failure was recorded
        assert ha_manager.metrics.health_check_failures > 0
    
    @pytest.mark.asyncio
    async def test_shutdown_with_coredis(self, ha_manager):
        """Test shutdown with coredis connections."""
        # Mock Redis connections
        mock_master = AsyncMock()
        mock_replica = AsyncMock()
        
        ha_manager.redis_master = mock_master
        ha_manager.redis_replica = mock_replica
        
        # Mock PostgreSQL pools
        ha_manager.current_pg_primary = None
        ha_manager.pg_replica_pools = []
        
        # Call shutdown
        await ha_manager.shutdown()
        
        # Verify aclose was called on Redis connections
        mock_master.aclose.assert_called_once()
        mock_replica.aclose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_status_with_coredis(self, ha_manager):
        """Test health status reporting with coredis connections."""
        # Set up mock connections
        ha_manager.redis_master = AsyncMock()
        ha_manager.redis_replica = AsyncMock()
        ha_manager.redis_sentinel = AsyncMock()
        
        # Get health status
        status = await ha_manager.get_health_status()
        
        # Verify Redis status is reported correctly
        assert status["redis"]["master_available"] is True
        assert status["redis"]["replica_available"] is True
        assert status["redis"]["sentinel_enabled"] is True
    
    @pytest.mark.asyncio
    @patch('src.prompt_improver.database.ha_connection_manager.Sentinel')
    async def test_sentinel_initialization_with_password(self, mock_sentinel_class, ha_manager):
        """Test Sentinel initialization with Redis password."""
        # Set password in config
        ha_manager.redis_config.password = "test_password"
        
        # Mock Sentinel instance
        mock_sentinel = MagicMock()
        mock_sentinel_class.return_value = mock_sentinel
        
        # Mock master and replica connections
        mock_master = AsyncMock()
        mock_replica = AsyncMock()
        mock_sentinel.primary_for.return_value = mock_master
        mock_sentinel.replica_for.return_value = mock_replica
        
        # Call setup method
        await ha_manager._setup_redis_sentinel()
        
        # Verify password was passed to connections
        mock_sentinel.primary_for.assert_called_once_with(
            'mymaster',
            stream_timeout=0.1,
            password="test_password"
        )
        mock_sentinel.replica_for.assert_called_once_with(
            'mymaster',
            stream_timeout=0.1,
            password="test_password"
        )


class TestCoredisIntegration:
    """Integration tests for coredis functionality."""
    
    @pytest.mark.asyncio
    async def test_coredis_import(self):
        """Test that coredis imports work correctly."""
        import coredis
        from coredis.sentinel import Sentinel
        
        # Verify classes exist
        assert hasattr(coredis, 'Redis')
        assert Sentinel is not None
    
    @pytest.mark.asyncio
    async def test_coredis_basic_functionality(self):
        """Test basic coredis functionality (requires running Redis)."""
        pytest.skip("Requires running Redis instance")
        
        # This test would require a real Redis instance
        # Keeping as placeholder for integration testing
        import coredis
        
        client = coredis.Redis(host='localhost', port=6379, db=0)
        try:
            await client.ping()
            await client.set('test_key', 'test_value')
            value = await client.get('test_key')
            assert value == b'test_value'
        finally:
            await client.aclose()
    
    @pytest.mark.asyncio
    async def test_coredis_sentinel_functionality(self):
        """Test coredis Sentinel functionality (requires running Sentinel)."""
        pytest.skip("Requires running Redis Sentinel")
        
        # This test would require a real Redis Sentinel setup
        # Keeping as placeholder for integration testing
        from coredis.sentinel import Sentinel
        
        sentinel = Sentinel(sentinels=[('localhost', 26379)])
        try:
            primary = sentinel.primary_for('mymaster')
            await primary.ping()
            replica = sentinel.replica_for('mymaster')
            await replica.ping()
        except Exception:
            # Expected if no Sentinel is running
            pass


if __name__ == "__main__":
    # Run tests with pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))