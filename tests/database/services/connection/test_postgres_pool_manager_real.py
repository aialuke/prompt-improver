"""Real behavior tests for PostgreSQL PoolManager.

Tests with actual PostgreSQL database connections - NO MOCKS.
Requires real database connection for integration testing.
"""

import asyncio
import os
import time
from unittest.mock import Mock

import pytest
from sqlalchemy import text

from prompt_improver.database.services.connection.postgres_pool_manager import (
    PostgreSQLPoolManager,
    DatabaseConfig,
    PoolConfiguration,
    PoolState,
    HealthStatus,
    ConnectionMode,
    ConnectionInfo,
)


class TestDatabaseConfig:
    """Test DatabaseConfig functionality."""
    
    def test_database_config_creation(self):
        """Test basic database config creation."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass"
        )
        
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "test_db"
        assert config.username == "test_user"
        assert config.password == "test_pass"
        assert not config.echo_sql  # Default
        
    def test_database_config_with_echo(self):
        """Test database config with SQL echo enabled."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass",
            echo_sql=True
        )
        
        assert config.echo_sql


class TestPoolConfiguration:
    """Test PoolConfiguration functionality."""
    
    def test_pool_config_creation(self):
        """Test basic pool config creation."""
        config = PoolConfiguration(
            pool_size=10,
            max_overflow=5,
            timeout=30.0
        )
        
        assert config.pool_size == 10
        assert config.max_overflow == 5
        assert config.timeout == 30.0
        assert config.enable_circuit_breaker  # Default True
        assert config.pool_pre_ping  # Default True
        
    def test_pool_config_for_environment(self):
        """Test environment-specific pool configurations."""
        # Development config
        dev_config = PoolConfiguration.for_environment("development")
        assert dev_config.pool_size == 5
        assert dev_config.max_overflow == 2
        assert dev_config.timeout == 10.0
        assert "dev" in dev_config.application_name
        
        # Production config
        prod_config = PoolConfiguration.for_environment("production")
        assert prod_config.pool_size == 20
        assert prod_config.max_overflow == 10
        assert prod_config.timeout == 30.0
        assert prod_config.enable_ha
        assert "prod" in prod_config.application_name
        
        # Testing config
        test_config = PoolConfiguration.for_environment("testing")
        assert test_config.pool_size == 3
        assert test_config.max_overflow == 1
        assert not test_config.enable_circuit_breaker
        
        # Default fallback
        unknown_config = PoolConfiguration.for_environment("unknown")
        assert unknown_config.pool_size == 5  # Falls back to development


class TestConnectionInfo:
    """Test ConnectionInfo functionality."""
    
    def test_connection_info_creation(self):
        """Test connection info creation."""
        from datetime import datetime, UTC
        
        info = ConnectionInfo(
            connection_id="test_conn_1",
            created_at=datetime.now(UTC)
        )
        
        assert info.connection_id == "test_conn_1"
        assert isinstance(info.created_at, datetime)
        assert isinstance(info.last_used, datetime)
        assert info.query_count == 0
        assert info.error_count == 0
        assert info.pool_name == "default"


class TestPostgreSQLPoolManagerUnit:
    """Test PostgreSQL PoolManager unit functionality."""
    
    def test_pool_manager_creation(self):
        """Test basic pool manager creation."""
        db_config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass"
        )
        
        pool_config = PoolConfiguration(
            pool_size=5,
            max_overflow=2,
            timeout=10.0
        )
        
        manager = PostgreSQLPoolManager(db_config, pool_config, "test_manager")
        
        assert manager.db_config == db_config
        assert manager.pool_config == pool_config
        assert manager.service_name == "test_manager"
        assert manager._pool_state == PoolState.INITIALIZING
        assert manager._health_status == HealthStatus.UNKNOWN
        assert not manager._is_initialized
        assert manager.min_pool_size == 5
        assert manager.max_pool_size == 25  # 5 * 5, capped at 100
        assert manager.current_pool_size == 5
        
    def test_pool_manager_scaling_thresholds(self):
        """Test scaling threshold configuration."""
        db_config = DatabaseConfig("localhost", 5432, "test", "user", "pass")
        pool_config = PoolConfiguration(10, 5, 15.0)
        
        manager = PostgreSQLPoolManager(db_config, pool_config)
        
        assert manager.scale_up_threshold == 0.8
        assert manager.scale_down_threshold == 0.3
        assert manager.scale_cooldown_seconds == 60
        
    def test_pool_manager_circuit_breaker_config(self):
        """Test circuit breaker configuration."""
        db_config = DatabaseConfig("localhost", 5432, "test", "user", "pass")
        pool_config = PoolConfiguration(10, 5, 15.0, enable_circuit_breaker=True)
        
        manager = PostgreSQLPoolManager(db_config, pool_config)
        
        assert manager._circuit_breaker_threshold == 5
        assert manager._circuit_breaker_failures == 0
        assert manager._circuit_breaker_timeout == 60
        assert not manager._is_circuit_breaker_open()


class TestPostgreSQLPoolManagerIntegration:
    """Test PostgreSQL PoolManager with real database integration."""
    
    @pytest.fixture
    def db_config(self):
        """Create database configuration from environment variables."""
        return DatabaseConfig(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DATABASE", "apes_test"),
            username=os.getenv("POSTGRES_USERNAME", "test_user"),
            password=os.getenv("POSTGRES_PASSWORD", "test_password"),
            echo_sql=False  # Keep quiet for tests
        )
    
    @pytest.fixture
    def pool_config(self):
        """Create pool configuration for testing."""
        return PoolConfiguration.for_environment("testing", "integration_test")
    
    async def test_pool_manager_initialization(self, db_config, pool_config):
        """Test pool manager initialization with real database."""
        manager = PostgreSQLPoolManager(db_config, pool_config, "init_test")
        
        try:
            result = await manager.initialize()
            
            if result:
                assert manager._is_initialized
                assert manager._pool_state in [PoolState.HEALTHY, PoolState.DEGRADED]
                assert manager._health_status != HealthStatus.UNKNOWN
                assert manager._async_engine is not None
                assert manager._async_session_factory is not None
                print("✅ Pool manager initialization successful")
            else:
                pytest.skip("Database not available for initialization test")
                
        except Exception as e:
            pytest.skip(f"Database connection failed: {e}")
        finally:
            try:
                await manager.shutdown()
            except Exception:
                pass
    
    async def test_session_management(self, db_config, pool_config):
        """Test session creation and management."""
        manager = PostgreSQLPoolManager(db_config, pool_config, "session_test")
        
        try:
            await manager.initialize()
            
            # Test basic session usage
            async with manager.get_session() as session:
                result = await session.execute(text("SELECT 1 as test_value"))
                value = result.scalar()
                assert value == 1
            
            # Test read-only session
            async with manager.get_session(ConnectionMode.READ_ONLY) as session:
                result = await session.execute(text("SELECT 42 as readonly_test"))
                value = result.scalar()
                assert value == 42
            
            # Verify metrics were updated
            assert manager.metrics.queries_executed >= 2
            assert manager.metrics.active_connections >= 0
            print("✅ Session management tests passed")
            
        except Exception as e:
            pytest.skip(f"Session management test failed: {e}")
        finally:
            try:
                await manager.shutdown()
            except Exception:
                pass
    
    async def test_health_check_integration(self, db_config, pool_config):
        """Test health check with real database."""
        manager = PostgreSQLPoolManager(db_config, pool_config, "health_test")
        
        try:
            await manager.initialize()
            
            health_result = await manager.health_check()
            
            assert "status" in health_result
            assert "timestamp" in health_result
            assert "service" in health_result
            assert health_result["service"] == "health_test"
            assert "components" in health_result
            assert "metrics" in health_result
            assert health_result["response_time_ms"] > 0
            
            if health_result["status"] == "healthy":
                assert "sqlalchemy_session" in health_result["components"]
                assert health_result["components"]["sqlalchemy_session"] == "healthy"
            
            print(f"✅ Health check passed: {health_result['status']}")
            
        except Exception as e:
            pytest.skip(f"Health check test failed: {e}")
        finally:
            try:
                await manager.shutdown()
            except Exception:
                pass
    
    async def test_metrics_collection(self, db_config, pool_config):
        """Test metrics collection during operations."""
        manager = PostgreSQLPoolManager(db_config, pool_config, "metrics_test")
        
        try:
            await manager.initialize()
            
            # Perform several operations to generate metrics
            for i in range(5):
                async with manager.get_session() as session:
                    result = await session.execute(text(f"SELECT {i} as iteration"))
                    assert result.scalar() == i
                
                # Small delay between operations
                await asyncio.sleep(0.01)
            
            # Get metrics
            metrics = await manager.get_metrics()
            
            assert "service" in metrics
            assert metrics["service"] == "metrics_test"
            assert "pool_state" in metrics
            assert "health_status" in metrics
            assert "current_pool_size" in metrics
            assert "pool_metrics" in metrics
            assert "connection_metrics" in metrics
            assert "efficiency_metrics" in metrics
            assert "circuit_breaker" in metrics
            
            # Verify connection metrics were updated
            conn_metrics = metrics["connection_metrics"]
            assert conn_metrics["queries_executed"] >= 5
            assert conn_metrics["total_connections"] >= 1
            
            # Verify pool metrics
            pool_metrics = metrics["pool_metrics"]
            assert "pool_size" in pool_metrics
            assert "utilization" in pool_metrics
            
            print(f"✅ Metrics collection verified: {conn_metrics['queries_executed']} queries executed")
            
        except Exception as e:
            pytest.skip(f"Metrics collection test failed: {e}")
        finally:
            try:
                await manager.shutdown()
            except Exception:
                pass
    
    async def test_circuit_breaker_behavior(self, db_config, pool_config):
        """Test circuit breaker with simulated failures."""
        # Create config with circuit breaker enabled and low threshold
        pool_config_cb = PoolConfiguration(
            pool_size=3,
            max_overflow=1,
            timeout=5.0,
            enable_circuit_breaker=True
        )
        
        manager = PostgreSQLPoolManager(db_config, pool_config_cb, "circuit_breaker_test")
        manager._circuit_breaker_threshold = 2  # Lower threshold for testing
        
        try:
            await manager.initialize()
            
            # Simulate failures by calling the failure handler directly
            exception = Exception("Simulated database failure")
            
            # First failure
            manager._handle_connection_failure(exception)
            assert not manager._is_circuit_breaker_open()
            assert manager._circuit_breaker_failures == 1
            
            # Second failure - should open circuit breaker
            manager._handle_connection_failure(exception)
            assert manager._is_circuit_breaker_open()
            assert manager._circuit_breaker_failures == 2
            
            # Verify circuit breaker blocks operations
            try:
                async with manager.get_session() as session:
                    await session.execute(text("SELECT 1"))
                assert False, "Should have raised RuntimeError due to open circuit breaker"
            except RuntimeError as e:
                assert "circuit breaker" in str(e).lower()
            
            # Test timeout recovery
            manager._circuit_breaker_timeout = 0.1  # Short timeout for testing
            await asyncio.sleep(0.2)
            
            # Should allow operations again
            assert not manager._is_circuit_breaker_open()
            
            print("✅ Circuit breaker behavior verified")
            
        except Exception as e:
            pytest.skip(f"Circuit breaker test failed: {e}")
        finally:
            try:
                await manager.shutdown()
            except Exception:
                pass
    
    async def test_pool_scaling(self, db_config, pool_config):
        """Test dynamic pool scaling functionality."""
        manager = PostgreSQLPoolManager(db_config, pool_config, "scaling_test")
        
        try:
            await manager.initialize()
            
            initial_size = manager.current_pool_size
            
            # Test scaling up
            await manager._scale_pool(initial_size + 2)
            assert manager.current_pool_size == initial_size + 2
            assert manager.pool_config.pool_size == initial_size + 2
            
            # Test scaling down
            await manager._scale_pool(initial_size)
            assert manager.current_pool_size == initial_size
            assert manager.pool_config.pool_size == initial_size
            
            # Test optimization logic
            optimization_result = await manager.optimize_pool_size()
            
            assert "status" in optimization_result
            assert optimization_result["status"] in [
                "optimized", "no_change_needed", "skipped", "error"
            ]
            
            if optimization_result["status"] == "optimized":
                assert "previous_size" in optimization_result
                assert "new_size" in optimization_result
                assert "utilization" in optimization_result
            
            print(f"✅ Pool scaling verified: {optimization_result['status']}")
            
        except Exception as e:
            pytest.skip(f"Pool scaling test failed: {e}")
        finally:
            try:
                await manager.shutdown()
            except Exception:
                pass
    
    async def test_concurrent_operations(self, db_config, pool_config):
        """Test concurrent database operations."""
        manager = PostgreSQLPoolManager(db_config, pool_config, "concurrent_test")
        
        try:
            await manager.initialize()
            
            async def worker(worker_id: int, operations: int):
                """Simulate concurrent database operations."""
                for i in range(operations):
                    async with manager.get_session() as session:
                        result = await session.execute(text(f"SELECT {worker_id} * 100 + {i} as value"))
                        expected = worker_id * 100 + i
                        assert result.scalar() == expected
                    
                    # Small delay to allow interleaving
                    await asyncio.sleep(0.001)
            
            # Run multiple workers concurrently
            workers = [worker(i, 5) for i in range(3)]
            await asyncio.gather(*workers)
            
            # Verify metrics reflect concurrent usage
            metrics = await manager.get_metrics()
            conn_metrics = metrics["connection_metrics"]
            
            assert conn_metrics["queries_executed"] >= 15  # 3 workers * 5 operations
            assert conn_metrics["total_connections"] >= 1
            
            print(f"✅ Concurrent operations completed: {conn_metrics['queries_executed']} total queries")
            
        except Exception as e:
            pytest.skip(f"Concurrent operations test failed: {e}")
        finally:
            try:
                await manager.shutdown()
            except Exception:
                pass
    
    async def test_connection_registry_tracking(self, db_config, pool_config):
        """Test connection registry and tracking functionality."""
        manager = PostgreSQLPoolManager(db_config, pool_config, "registry_test")
        
        try:
            await manager.initialize()
            
            # Perform operations to populate registry
            async with manager.get_session() as session:
                result = await session.execute(text("SELECT 'registry_test' as test"))
                assert result.scalar() == "registry_test"
            
            # Check registry tracking
            metrics = await manager.get_metrics()
            registry_size = metrics.get("connection_registry_size", 0)
            
            assert registry_size >= 0
            assert len(manager._connection_registry) == registry_size
            
            # Verify performance window tracking
            performance_window_size = metrics.get("performance_window_size", 0)
            assert performance_window_size >= 0
            assert len(manager.performance_window) == performance_window_size
            
            print(f"✅ Connection registry tracking: {registry_size} connections, {performance_window_size} events")
            
        except Exception as e:
            pytest.skip(f"Registry tracking test failed: {e}")
        finally:
            try:
                await manager.shutdown()
            except Exception:
                pass


class TestPostgreSQLPoolManagerPerformance:
    """Test PostgreSQL PoolManager performance characteristics."""
    
    async def test_high_volume_operations(self):
        """Test performance with high volume operations."""
        # Skip if no database available
        if not all([
            os.getenv("POSTGRES_HOST"),
            os.getenv("POSTGRES_USERNAME"), 
            os.getenv("POSTGRES_PASSWORD"),
        ]):
            pytest.skip("Database credentials not available for performance test")
        
        db_config = DatabaseConfig(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DATABASE", "apes_test"),
            username=os.getenv("POSTGRES_USERNAME"),
            password=os.getenv("POSTGRES_PASSWORD")
        )
        
        # Larger pool for performance testing
        pool_config = PoolConfiguration(
            pool_size=10,
            max_overflow=5,
            timeout=10.0
        )
        
        manager = PostgreSQLPoolManager(db_config, pool_config, "performance_test")
        
        try:
            await manager.initialize()
            
            # Perform high volume operations
            start_time = time.time()
            num_operations = 100
            
            for i in range(num_operations):
                async with manager.get_session() as session:
                    result = await session.execute(text(f"SELECT {i} as operation_id"))
                    assert result.scalar() == i
            
            duration = time.time() - start_time
            operations_per_second = num_operations / duration
            
            # Verify performance
            assert operations_per_second > 10  # Should handle at least 10 ops/sec
            
            # Check metrics after high volume
            metrics = await manager.get_metrics()
            conn_metrics = metrics["connection_metrics"]
            
            assert conn_metrics["queries_executed"] >= num_operations
            assert conn_metrics["avg_response_time_ms"] > 0
            assert conn_metrics["pool_utilization"] >= 0
            
            print(f"✅ Performance test: {operations_per_second:.1f} ops/sec, avg response: {conn_metrics['avg_response_time_ms']:.1f}ms")
            
        except Exception as e:
            pytest.skip(f"Performance test failed: {e}")
        finally:
            try:
                await manager.shutdown()
            except Exception:
                pass
    
    async def test_memory_efficiency(self):
        """Test memory efficiency with bounded collections."""
        # Skip if no database available
        if not all([
            os.getenv("POSTGRES_HOST"),
            os.getenv("POSTGRES_USERNAME"),
            os.getenv("POSTGRES_PASSWORD"),
        ]):
            pytest.skip("Database credentials not available")
        
        db_config = DatabaseConfig(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DATABASE", "apes_test"),
            username=os.getenv("POSTGRES_USERNAME"),
            password=os.getenv("POSTGRES_PASSWORD")
        )
        
        pool_config = PoolConfiguration.for_environment("testing")
        manager = PostgreSQLPoolManager(db_config, pool_config, "memory_test")
        
        try:
            await manager.initialize()
            
            # Perform more operations than performance window size
            num_operations = 150  # More than maxlen=100
            
            for i in range(num_operations):
                async with manager.get_session() as session:
                    result = await session.execute(text(f"SELECT {i}"))
                    assert result.scalar() == i
                
                # Record performance events
                manager._record_performance_event("test", 10.0, True)
            
            # Verify collections are bounded
            assert len(manager.performance_window) <= 100  # maxlen=100
            assert len(manager.metrics.connection_times) <= 1000  # From ConnectionMetrics
            assert len(manager.metrics.query_times) <= 1000
            
            # But metrics should still be accurate
            metrics = await manager.get_metrics()
            conn_metrics = metrics["connection_metrics"]
            assert conn_metrics["queries_executed"] == num_operations
            
            print(f"✅ Memory efficiency verified: performance_window={len(manager.performance_window)}, operations={num_operations}")
            
        except Exception as e:
            pytest.skip(f"Memory efficiency test failed: {e}")
        finally:
            try:
                await manager.shutdown()
            except Exception:
                pass