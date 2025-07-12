"""Integration tests for queue health monitoring system.

Tests the integration of QueueHealthChecker with the health service and MCP server.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from prompt_improver.services.health import get_health_service, HealthStatus
from prompt_improver.services.health.checkers import QueueHealthChecker
from prompt_improver.optimization.batch_processor import BatchProcessor


class TestQueueHealthIntegration:
    """Test suite for queue health monitoring integration."""

    @pytest.fixture
    def mock_batch_processor(self):
        """Create a mock batch processor for testing."""
        processor = Mock(spec=BatchProcessor)
        processor.get_queue_size.return_value = 10
        processor.config = Mock()
        processor.config.enable_priority_queue = True
        processor.config.max_queue_size = 1000
        processor.config.batch_size = 10
        processor.config.concurrency = 3
        processor.config.max_attempts = 3
        processor.processing = False
        processor.metrics = {
            "processed": 100,
            "failed": 5,
            "retries": 2,
            "start_time": 1000000000
        }
        return processor

    @pytest.fixture
    def mock_task_manager(self):
        """Create a mock background task manager."""
        manager = Mock()
        manager.get_queue_size.return_value = 5
        manager.get_running_tasks.return_value = ["task1", "task2"]
        manager.max_concurrent_tasks = 10
        manager.get_task_count.return_value = {
            "pending": 3,
            "running": 2,
            "completed": 15,
            "failed": 1,
            "cancelled": 0
        }
        return manager

    @pytest.mark.asyncio
    async def test_queue_health_checker_basic_functionality(self, mock_batch_processor):
        """Test basic queue health checker functionality."""
        checker = QueueHealthChecker(batch_processor=mock_batch_processor)
        
        with patch('prompt_improver.services.health.checkers.get_background_task_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_task_manager()
            
            result = await checker.check()
            
            assert result.component == "queue"
            assert result.status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.FAILED]
            assert result.details is not None
            assert "total_queue_backlog" in result.details
            assert "queue_capacity_utilization" in result.details

    @pytest.mark.asyncio
    async def test_queue_health_checker_in_health_service(self, mock_batch_processor):
        """Test queue health checker integration with health service."""
        # Create health service with queue checker
        queue_checker = QueueHealthChecker(batch_processor=mock_batch_processor)
        
        with patch('prompt_improver.services.health.checkers.get_background_task_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_task_manager()
            
            health_service = get_health_service()
            
            # Check if queue checker is available
            available_checks = health_service.get_available_checks()
            assert "queue" in available_checks
            
            # Run specific queue check
            queue_result = await health_service.run_specific_check("queue")
            assert queue_result.component == "queue"

    @pytest.mark.asyncio
    async def test_queue_health_metrics_collection(self, mock_batch_processor):
        """Test comprehensive metrics collection."""
        checker = QueueHealthChecker(batch_processor=mock_batch_processor)
        
        with patch('prompt_improver.services.health.checkers.get_background_task_manager') as mock_get_manager:
            mock_task_manager = self.mock_task_manager()
            mock_get_manager.return_value = mock_task_manager
            
            metrics = await checker._collect_queue_metrics()
            
            # Check batch processor metrics
            assert "training_queue_size" in metrics
            assert "priority_queue_enabled" in metrics
            assert "max_queue_size" in metrics
            assert "processed_count" in metrics
            assert "success_rate" in metrics
            
            # Check background task metrics
            assert "background_queue_size" in metrics
            assert "running_tasks" in metrics
            assert "task_utilization" in metrics
            
            # Check derived metrics
            assert "total_queue_backlog" in metrics
            assert "queue_capacity_utilization" in metrics
            assert "retry_backlog_ratio" in metrics
            assert "avg_processing_latency_ms" in metrics
            assert "throughput_per_second" in metrics

    @pytest.mark.asyncio
    async def test_queue_health_status_evaluation(self, mock_batch_processor):
        """Test health status evaluation logic."""
        checker = QueueHealthChecker(batch_processor=mock_batch_processor)
        
        # Test healthy status
        healthy_metrics = {
            "queue_capacity_utilization": 0.5,
            "retry_backlog_ratio": 0.1,
            "success_rate": 0.98,
            "task_utilization": 0.6
        }
        status = checker._evaluate_queue_health(healthy_metrics)
        assert status == HealthStatus.HEALTHY
        
        # Test warning status - high capacity
        warning_metrics = {
            "queue_capacity_utilization": 0.85,
            "retry_backlog_ratio": 0.1,
            "success_rate": 0.98,
            "task_utilization": 0.6
        }
        status = checker._evaluate_queue_health(warning_metrics)
        assert status == HealthStatus.WARNING
        
        # Test failed status - high retry rate
        failed_metrics = {
            "queue_capacity_utilization": 0.5,
            "retry_backlog_ratio": 0.6,  # Very high retry rate
            "success_rate": 0.98,
            "task_utilization": 0.6
        }
        status = checker._evaluate_queue_health(failed_metrics)
        assert status == HealthStatus.FAILED

    @pytest.mark.asyncio
    async def test_queue_health_mcp_endpoint_integration(self):
        """Test integration with MCP server health endpoint."""
        from prompt_improver.mcp_server.mcp_server import health_queue
        from prompt_improver.services.health import get_health_service
        
        with patch('prompt_improver.services.health.get_health_service') as mock_get_service:
            # Mock health service
            mock_health_service = Mock()
            mock_queue_result = Mock()
            mock_queue_result.status.value = "healthy"
            mock_queue_result.message = "Queue system healthy"
            mock_queue_result.timestamp.isoformat.return_value = "2024-01-01T00:00:00"
            mock_queue_result.response_time_ms = 50.0
            mock_queue_result.error = None
            mock_queue_result.details = {
                "total_queue_backlog": 15,
                "retry_count": 2,
                "avg_processing_latency_ms": 25.5,
                "queue_capacity_utilization": 0.3,
                "success_rate": 0.95,
                "throughput_per_second": 2.5
            }
            
            mock_health_service.run_specific_check.return_value = mock_queue_result
            mock_get_service.return_value = mock_health_service
            
            # Call MCP endpoint
            response = await health_queue()
            
            # Verify response structure
            assert response["status"] == "healthy"
            assert response["message"] == "Queue system healthy"
            assert response["queue_length"] == 15
            assert response["retry_backlog"] == 2
            assert response["avg_latency_ms"] == 25.5
            assert response["capacity_utilization"] == 0.3
            assert response["success_rate"] == 0.95
            assert response["throughput_per_second"] == 2.5
            assert "metrics" in response

    @pytest.mark.asyncio
    async def test_queue_health_error_handling(self):
        """Test error handling in queue health checks."""
        # Test with no batch processor
        checker = QueueHealthChecker(batch_processor=None)
        
        result = await checker.check()
        assert result.status == HealthStatus.WARNING
        assert "not configured" in result.message.lower()

    @pytest.mark.asyncio
    async def test_queue_health_comprehensive_integration(self, mock_batch_processor):
        """Test end-to-end integration of queue health monitoring."""
        with patch('prompt_improver.services.health.checkers.get_background_task_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_task_manager()
            
            # Get health service and run comprehensive health check
            health_service = get_health_service()
            result = await health_service.run_health_check()
            
            # Verify queue health is included in aggregated results
            assert "queue" in result.checks
            queue_check = result.checks["queue"]
            assert queue_check.component == "queue"
            
            # Verify overall status incorporates queue health
            assert result.overall_status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.FAILED]

    @pytest.mark.asyncio
    async def test_queue_health_status_messages(self, mock_batch_processor):
        """Test status message generation for different health states."""
        checker = QueueHealthChecker(batch_processor=mock_batch_processor)
        
        # Test healthy message
        healthy_metrics = {
            "total_queue_backlog": 10,
            "queue_capacity_utilization": 0.3,
            "success_rate": 0.98
        }
        message = checker._get_status_message(HealthStatus.HEALTHY, healthy_metrics)
        assert "healthy" in message.lower()
        assert "10 items" in message
        assert "30.0%" in message
        assert "98.0%" in message
        
        # Test warning message
        warning_metrics = {
            "total_queue_backlog": 800,
            "queue_capacity_utilization": 0.85,
            "success_rate": 0.92
        }
        message = checker._get_status_message(HealthStatus.WARNING, warning_metrics)
        assert "warning" in message.lower()
        assert "800 items" in message

    def mock_task_manager(self):
        """Helper method to create mock task manager."""
        manager = Mock()
        manager.get_queue_size.return_value = 5
        manager.get_running_tasks.return_value = ["task1", "task2"]
        manager.max_concurrent_tasks = 10
        manager.get_task_count.return_value = {
            "pending": 3,
            "running": 2,
            "completed": 15,
            "failed": 1,
            "cancelled": 0
        }
        return manager
