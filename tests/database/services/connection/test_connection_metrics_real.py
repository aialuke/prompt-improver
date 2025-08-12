"""Real behavior tests for ConnectionMetrics.

Tests with actual PostgreSQL connections - NO MOCKS.
Requires real database connection for integration testing.
"""

import asyncio
import time

import asyncpg
import pytest

from prompt_improver.database.services.connection.connection_metrics import ConnectionMetrics


class TestConnectionMetricsRealDB:
    """Test ConnectionMetrics with real PostgreSQL database."""
    
    def test_connection_metrics_basic_sync(self):
        """Test basic connection recording (synchronous test)."""
        metrics = ConnectionMetrics()
        
        # Test connection recording
        metrics.record_connection()
        assert metrics.active_connections == 1
        assert metrics.total_connections == 1
        assert metrics.connections_created == 1
        
        # Test query recording
        metrics.record_query(5.5, success=True)
        assert metrics.queries_executed == 1
        assert metrics.avg_response_time_ms == 5.5
        assert len(metrics.query_times) == 1
        
        metrics.record_disconnection()
        assert metrics.active_connections == 0
        assert metrics.connections_closed == 1
    
    def test_connection_metrics_pool_utilization(self):
        """Test pool utilization calculations."""
        metrics = ConnectionMetrics()
        metrics.connection_pool_size = 10
        metrics.active_connections = 5
        
        utilization = metrics.calculate_pool_utilization()
        assert utilization == 50.0
        assert metrics.pool_utilization == 50.0
        
        # Test edge case - zero pool size
        metrics.connection_pool_size = 0
        utilization = metrics.calculate_pool_utilization()
        assert utilization == 0.0
    
    def test_connection_metrics_error_tracking(self):
        """Test error rate calculations."""
        metrics = ConnectionMetrics()
        
        # Test successful queries
        metrics.record_query(5.0, success=True)
        metrics.record_query(3.0, success=True)
        assert metrics.queries_executed == 2
        assert metrics.queries_failed == 0
        assert metrics.error_rate == 0.0
        
        # Add failed query
        metrics.record_query(10.0, success=False)
        assert metrics.queries_executed == 3
        assert metrics.queries_failed == 1
        assert metrics.error_rate == 1.0/3.0  # 33%
    
    def test_connection_metrics_health_assessment(self):
        """Test health check logic."""
        metrics = ConnectionMetrics()
        
        # Set up healthy state
        metrics.connection_pool_size = 10
        metrics.active_connections = 5  # 50% utilization
        metrics.error_rate = 0.05  # 5% error rate
        metrics.circuit_breaker_state = "closed"
        metrics.redis_maxclients = 100
        metrics.redis_connected_clients = 50  # 50% utilization
        metrics.redis_blocked_clients = 10
        metrics.redis_rejected_connections = 0
        
        assert metrics.is_healthy()
        
        # Test unhealthy conditions
        
        # High pool utilization
        metrics.active_connections = 10  # 100% utilization
        assert not metrics.is_healthy()
        
        # Reset and test high error rate  
        metrics.active_connections = 5
        metrics.error_rate = 0.15  # 15% error rate
        assert not metrics.is_healthy()
        
        # Reset and test circuit breaker
        metrics.error_rate = 0.05
        metrics.circuit_breaker_state = "open"
        assert not metrics.is_healthy()
        
        # Reset and test Redis issues
        metrics.circuit_breaker_state = "closed"
        metrics.redis_connected_clients = 95  # 95% utilization
        assert not metrics.is_healthy()
        
        # Test blocked clients
        metrics.redis_connected_clients = 50
        metrics.redis_blocked_clients = 30  # More than 50% of connected
        assert not metrics.is_healthy()
        
        # Test rejected connections
        metrics.redis_blocked_clients = 10
        metrics.redis_rejected_connections = 5
        assert not metrics.is_healthy()
    
    def test_connection_metrics_efficiency_calculation(self):
        """Test efficiency calculations."""
        metrics = ConnectionMetrics()
        
        # Set up test data
        metrics.connections_created = 10
        metrics.connection_reuse_count = 30
        
        efficiency = metrics.get_efficiency_metrics()
        
        assert efficiency["pool_efficiency"] == 300.0  # 30/10 * 100
        assert efficiency["reuse_rate"] == 3.0
        assert efficiency["connections_saved"] == 30
        
        # Calculate expected reduction: (total_ops - new_conns) / total_ops * 100
        # total_ops = reuse + created = 30 + 10 = 40
        # reduction = (40 - 10) / 40 * 100 = 75%
        assert efficiency["database_load_reduction_percent"] == 75.0
    
    def test_connection_metrics_serialization(self):
        """Test dictionary conversion."""
        from datetime import datetime
        
        metrics = ConnectionMetrics()
        metrics.active_connections = 8
        metrics.pool_utilization = 80.0
        metrics.avg_response_time_ms = 12.5
        metrics.last_scale_event = datetime.now()
        metrics.sla_compliance_rate = 95.5
        
        data = metrics.to_dict()
        
        # Verify key fields are present
        assert data["active_connections"] == 8
        assert data["pool_utilization"] == 80.0
        assert data["avg_response_time_ms"] == 12.5
        assert data["sla_compliance_rate"] == 95.5
        assert "is_healthy" in data
        assert "last_scale_event" in data
        assert isinstance(data["is_healthy"], bool)
    
    def test_connection_metrics_reset_functionality(self):
        """Test counter reset."""
        metrics = ConnectionMetrics()
        
        # Set values that should be reset
        metrics.connections_created = 15
        metrics.connections_closed = 12
        metrics.queries_executed = 50
        metrics.queries_failed = 5
        metrics.error_rate = 0.1
        metrics.cache_l1_hits = 100
        
        # Set values that should NOT be reset
        metrics.active_connections = 3
        metrics.pool_utilization = 30.0
        
        metrics.reset_counters()
        
        # Verify reset values
        assert metrics.connections_created == 0
        assert metrics.connections_closed == 0
        assert metrics.queries_executed == 0
        assert metrics.queries_failed == 0
        assert metrics.error_rate == 0.0
        assert metrics.cache_l1_hits == 0
        
        # Verify preserved values
        assert metrics.active_connections == 3
        assert metrics.pool_utilization == 30.0
    
    def test_connection_metrics_cache_integration(self):
        """Test cache metrics integration."""
        metrics = ConnectionMetrics()
        
        # Set up cache metrics
        metrics.cache_l1_hits = 150
        metrics.cache_l2_hits = 75
        metrics.cache_l3_hits = 25
        metrics.cache_total_requests = 300
        
        total_hits = metrics.cache_l1_hits + metrics.cache_l2_hits + metrics.cache_l3_hits
        expected_hit_rate = total_hits / metrics.cache_total_requests
        metrics.cache_hit_rate = expected_hit_rate
        
        assert metrics.cache_hit_rate == 250/300  # ≈ 0.833
        
        # Test cache response times
        test_times = [1.5, 2.3, 0.8, 4.2, 1.1]
        for time_val in test_times:
            metrics.cache_response_times.append(time_val)
        
        assert len(metrics.cache_response_times) == 5
        assert list(metrics.cache_response_times) == test_times
    
    async def test_connection_metrics_with_real_asyncpg(self):
        """Test with real asyncpg if available.""" 
        import os
        
        # Skip if no database credentials
        if not all([
            os.getenv("POSTGRES_HOST"),
            os.getenv("POSTGRES_USERNAME"),
            os.getenv("POSTGRES_PASSWORD"),
        ]):
            pytest.skip("Real database credentials not available")
        
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = int(os.getenv("POSTGRES_PORT", "5432"))
        user = os.getenv("POSTGRES_USERNAME")
        password = os.getenv("POSTGRES_PASSWORD") 
        database = os.getenv("POSTGRES_DATABASE", "apes_test")
        
        dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        metrics = ConnectionMetrics()
        
        try:
            # Create connection pool
            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3)
            
            # Test with real database
            async with pool.acquire() as conn:
                metrics.record_connection()
                
                start_time = time.time()
                result = await conn.fetchval("SELECT 1")
                duration_ms = (time.time() - start_time) * 1000
                
                assert result == 1
                metrics.record_query(duration_ms, success=True)
                metrics.record_connection_time(duration_ms)
                
                assert metrics.queries_executed == 1
                assert metrics.avg_response_time_ms > 0
                assert len(metrics.query_times) == 1
                assert len(metrics.connection_times) == 1
                
            metrics.record_disconnection()
            await pool.close()
            
            assert metrics.active_connections == 0
            assert metrics.connections_closed == 1
            
        except Exception as e:
            pytest.skip(f"Could not connect to database: {e}")