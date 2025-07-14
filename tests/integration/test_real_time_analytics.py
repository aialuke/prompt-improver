"""
Integration tests for Real-time Analytics Dashboard
Tests WebSocket connections, metrics calculation, and dashboard functionality
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import patch, Mock

import websockets
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from src.prompt_improver.services.real_time_analytics import (
    RealTimeAnalyticsService, 
    RealTimeMetrics,
    AlertType
)
from src.prompt_improver.utils.websocket_manager import ConnectionManager
from src.prompt_improver.database.models import ABExperiment, RulePerformance


class TestRealTimeAnalytics:
    """Test suite for real-time analytics functionality"""
    
    @pytest.fixture
    async def analytics_service(self, async_session: AsyncSession):
        """Create analytics service for testing"""
        return RealTimeAnalyticsService(async_session)
    
    @pytest.fixture
    async def sample_experiment(self, async_session: AsyncSession):
        """Create sample experiment for testing"""
        experiment = ABExperiment(
            experiment_name="test_real_time_experiment",
            description="Test experiment for real-time analytics",
            control_rules={"rule_ids": ["rule_1"]},
            treatment_rules={"rule_ids": ["rule_2"]},
            target_metric="improvement_score",
            sample_size_per_group=100,
            status="running",
            started_at=datetime.utcnow() - timedelta(days=1)
        )
        
        async_session.add(experiment)
        await async_session.commit()
        await async_session.refresh(experiment)
        
        return experiment
    
    @pytest.fixture
    async def sample_performance_data(self, async_session: AsyncSession, sample_experiment):
        """Create sample performance data for testing"""
        # Control group data
        control_data = []
        for i in range(50):
            perf = RulePerformance(
                rule_id="rule_1",
                improvement_score=0.7 + (i * 0.001),  # Slight upward trend
                execution_time_ms=100 + (i * 2),
                user_satisfaction_score=0.8,
                created_at=datetime.utcnow() - timedelta(hours=12) + timedelta(minutes=i*10)
            )
            control_data.append(perf)
        
        # Treatment group data (slightly better performance)
        treatment_data = []
        for i in range(45):
            perf = RulePerformance(
                rule_id="rule_2", 
                improvement_score=0.75 + (i * 0.001),  # Slightly higher baseline
                execution_time_ms=95 + (i * 2),
                user_satisfaction_score=0.85,
                created_at=datetime.utcnow() - timedelta(hours=12) + timedelta(minutes=i*12)
            )
            treatment_data.append(perf)
        
        async_session.add_all(control_data + treatment_data)
        await async_session.commit()
        
        return control_data, treatment_data
    
    @pytest.mark.asyncio
    async def test_metrics_calculation(self, analytics_service, sample_experiment, sample_performance_data):
        """Test real-time metrics calculation"""
        control_data, treatment_data = sample_performance_data
        
        # Calculate metrics
        metrics = await analytics_service._calculate_metrics(str(sample_experiment.experiment_id))
        
        assert metrics is not None
        assert isinstance(metrics, RealTimeMetrics)
        assert metrics.experiment_id == str(sample_experiment.experiment_id)
        assert metrics.control_sample_size == len(control_data)
        assert metrics.treatment_sample_size == len(treatment_data)
        assert metrics.total_sample_size == len(control_data) + len(treatment_data)
        
        # Check that means are calculated
        assert metrics.control_mean > 0
        assert metrics.treatment_mean > 0
        
        # Treatment should have higher mean due to our data setup
        assert metrics.treatment_mean > metrics.control_mean
        
        # Effect size should be positive
        assert metrics.effect_size > 0
        
        # Progress should be calculated
        assert 0 <= metrics.completion_percentage <= 100
        
        # Data quality should be reasonable
        assert 0 <= metrics.data_quality_score <= 1
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, analytics_service, sample_experiment, sample_performance_data):
        """Test alert generation for significant results"""
        # Calculate metrics
        metrics = await analytics_service._calculate_metrics(str(sample_experiment.experiment_id))
        
        # Check for alerts
        alerts = await analytics_service._check_for_alerts(str(sample_experiment.experiment_id), metrics)
        
        assert isinstance(alerts, list)
        
        # Should generate alerts if conditions are met
        if metrics.statistical_significance:
            significance_alerts = [a for a in alerts if a.alert_type == AlertType.STATISTICAL_SIGNIFICANCE]
            assert len(significance_alerts) > 0
            
            alert = significance_alerts[0]
            assert alert.experiment_id == str(sample_experiment.experiment_id)
            assert alert.severity in ["info", "warning", "critical"]
            assert "significance" in alert.title.lower()
    
    @pytest.mark.asyncio
    async def test_monitoring_start_stop(self, analytics_service, sample_experiment):
        """Test starting and stopping experiment monitoring"""
        experiment_id = str(sample_experiment.experiment_id)
        
        # Start monitoring
        success = await analytics_service.start_experiment_monitoring(experiment_id, update_interval=1)
        assert success
        assert experiment_id in analytics_service.monitoring_tasks
        
        # Check that task is running
        task = analytics_service.monitoring_tasks[experiment_id]
        assert not task.done()
        
        # Stop monitoring
        success = await analytics_service.stop_experiment_monitoring(experiment_id)
        assert success
        assert experiment_id not in analytics_service.monitoring_tasks
        
        # Task should be cancelled
        assert task.done()
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_updates(self, analytics_service, sample_experiment, sample_performance_data):
        """Test that monitoring loop generates updates"""
        experiment_id = str(sample_experiment.experiment_id)
        
        # Mock Redis publishing to capture updates
        updates_captured = []
        
        async def mock_publish(channel, message):
            updates_captured.append(json.loads(message))
        
        with patch('src.prompt_improver.utils.websocket_manager.publish_experiment_update') as mock_publish_update:
            mock_publish_update.side_effect = lambda exp_id, data, redis_client: updates_captured.append(data)
            
            # Start monitoring with short interval
            await analytics_service.start_experiment_monitoring(experiment_id, update_interval=0.1)
            
            # Wait for a few updates
            await asyncio.sleep(0.3)
            
            # Stop monitoring
            await analytics_service.stop_experiment_monitoring(experiment_id)
        
        # Should have captured some updates
        assert len(updates_captured) > 0
        
        # Check update structure
        update = updates_captured[0]
        assert update["type"] == "metrics_update"
        assert update["experiment_id"] == experiment_id
        assert "metrics" in update
        assert "alerts" in update


class TestWebSocketManager:
    """Test suite for WebSocket connection management"""
    
    @pytest.fixture
    def connection_manager(self):
        """Create connection manager for testing"""
        return ConnectionManager()
    
    @pytest.mark.asyncio
    async def test_connection_tracking(self, connection_manager):
        """Test WebSocket connection tracking"""
        # Mock WebSocket connections
        mock_ws1 = Mock()
        mock_ws2 = Mock()
        
        # Mock accept method
        mock_ws1.accept = Mock(return_value=asyncio.create_task(asyncio.sleep(0)))
        mock_ws2.accept = Mock(return_value=asyncio.create_task(asyncio.sleep(0)))
        
        experiment_id = "test_experiment_123"
        
        # Connect first WebSocket
        await connection_manager.connect(mock_ws1, experiment_id, "user1")
        
        assert experiment_id in connection_manager.experiment_connections
        assert mock_ws1 in connection_manager.experiment_connections[experiment_id]
        assert connection_manager.get_connection_count(experiment_id) == 1
        
        # Connect second WebSocket to same experiment
        await connection_manager.connect(mock_ws2, experiment_id, "user2")
        
        assert connection_manager.get_connection_count(experiment_id) == 2
        
        # Disconnect first WebSocket
        await connection_manager.disconnect(mock_ws1)
        
        assert connection_manager.get_connection_count(experiment_id) == 1
        assert mock_ws1 not in connection_manager.experiment_connections[experiment_id]
        assert mock_ws2 in connection_manager.experiment_connections[experiment_id]
        
        # Disconnect second WebSocket
        await connection_manager.disconnect(mock_ws2)
        
        assert connection_manager.get_connection_count(experiment_id) == 0
        assert experiment_id not in connection_manager.experiment_connections
    
    @pytest.mark.asyncio 
    async def test_broadcast_to_experiment(self, connection_manager):
        """Test broadcasting messages to experiment connections"""
        # Mock WebSocket connections with send_text method
        mock_ws1 = Mock()
        mock_ws2 = Mock()
        
        mock_ws1.accept = Mock(return_value=asyncio.create_task(asyncio.sleep(0)))
        mock_ws2.accept = Mock(return_value=asyncio.create_task(asyncio.sleep(0)))
        mock_ws1.send_text = Mock(return_value=asyncio.create_task(asyncio.sleep(0)))
        mock_ws2.send_text = Mock(return_value=asyncio.create_task(asyncio.sleep(0)))
        
        experiment_id = "test_experiment_456"
        
        # Connect WebSockets
        await connection_manager.connect(mock_ws1, experiment_id, "user1")
        await connection_manager.connect(mock_ws2, experiment_id, "user2")
        
        # Broadcast message
        test_message = {
            "type": "test_update",
            "data": {"metric": "test_value"}
        }
        
        await connection_manager.broadcast_to_experiment(experiment_id, test_message)
        
        # Both WebSockets should have received the message
        assert mock_ws1.send_text.called
        assert mock_ws2.send_text.called
        
        # Check message content (includes timestamp)
        sent_message1 = json.loads(mock_ws1.send_text.call_args[0][0])
        sent_message2 = json.loads(mock_ws2.send_text.call_args[0][0])
        
        assert sent_message1["type"] == "test_update"
        assert sent_message1["data"]["metric"] == "test_value"
        assert "timestamp" in sent_message1
        
        assert sent_message2["type"] == "test_update"
        assert sent_message2["data"]["metric"] == "test_value"
        assert "timestamp" in sent_message2


class TestRealTimeAPI:
    """Test suite for real-time API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from src.prompt_improver.api.real_time_endpoints import real_time_router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(real_time_router)
        
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/api/v1/experiments/real-time/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "services" in data
        assert "websocket_manager" in data["services"]
    
    @pytest.mark.asyncio
    async def test_start_monitoring_endpoint(self, client, sample_experiment):
        """Test start monitoring endpoint"""
        experiment_id = str(sample_experiment.experiment_id)
        
        with patch('src.prompt_improver.services.real_time_analytics.get_real_time_analytics_service') as mock_service:
            mock_analytics = Mock()
            mock_analytics.start_experiment_monitoring = Mock(return_value=asyncio.create_task(asyncio.sleep(0)))
            mock_analytics.start_experiment_monitoring.return_value = True
            mock_service.return_value = mock_analytics
            
            response = client.post(f"/api/v1/experiments/real-time/experiments/{experiment_id}/monitoring/start")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert data["experiment_id"] == experiment_id
    
    def test_get_active_monitoring_endpoint(self, client):
        """Test get active monitoring endpoint"""
        with patch('src.prompt_improver.services.real_time_analytics.get_real_time_analytics_service') as mock_service:
            mock_analytics = Mock()
            mock_analytics.get_active_experiments = Mock(return_value=asyncio.create_task(asyncio.sleep(0)))
            mock_analytics.get_active_experiments.return_value = ["exp1", "exp2"]
            mock_service.return_value = mock_analytics
            
            response = client.get("/api/v1/experiments/real-time/monitoring/active")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "active_experiments" in data
        assert "experiments" in data


@pytest.mark.integration
class TestRealTimeIntegration:
    """Integration tests for the complete real-time system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring(self, async_session, sample_experiment, sample_performance_data):
        """Test complete end-to-end monitoring workflow"""
        # This test would require a full FastAPI app with WebSocket support
        # For now, we'll test the core components in isolation
        
        experiment_id = str(sample_experiment.experiment_id)
        
        # Create analytics service
        analytics_service = RealTimeAnalyticsService(async_session)
        
        # Create connection manager
        connection_manager = ConnectionManager()
        
        # Start monitoring
        success = await analytics_service.start_experiment_monitoring(experiment_id)
        assert success
        
        # Simulate getting metrics
        metrics = await analytics_service.get_real_time_metrics(experiment_id)
        assert metrics is not None
        
        # Simulate broadcasting update
        update_data = {
            "type": "metrics_update",
            "experiment_id": experiment_id,
            "metrics": metrics.__dict__
        }
        
        # This would normally be sent to WebSocket connections
        # For testing, we just verify the data structure
        assert update_data["type"] == "metrics_update"
        assert update_data["experiment_id"] == experiment_id
        assert "metrics" in update_data
        
        # Stop monitoring
        await analytics_service.stop_experiment_monitoring(experiment_id)
        
        # Cleanup
        await analytics_service.cleanup()
        await connection_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_redis_integration(self):
        """Test Redis integration for pub/sub messaging"""
        # This test requires a Redis instance
        # Skip if Redis is not available
        try:
            redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
            await redis_client.ping()
        except:
            pytest.skip("Redis not available for testing")
        
        connection_manager = ConnectionManager(redis_client)
        
        # Test publishing update
        test_data = {
            "type": "test_message",
            "data": {"test": "value"}
        }
        
        experiment_id = "test_experiment_redis"
        
        # Publish update (this would normally trigger WebSocket broadcast)
        from src.prompt_improver.utils.websocket_manager import publish_experiment_update
        await publish_experiment_update(experiment_id, test_data, redis_client)
        
        # Cleanup
        await connection_manager.cleanup()
        await redis_client.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])