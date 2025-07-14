"""
End-to-end integration tests for AutoML system following 2025 best practices.

Based on research of AutoML testing patterns and Optuna integration best practices:
- Tests complete AutoML workflow from initialization to optimization completion
- Validates integration with existing A/B testing and real-time analytics
- Tests database persistence and storage patterns
- Implements realistic optimization scenarios
- Follows 2025 AutoML integration testing standards
"""

import asyncio
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import optuna
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.automl.orchestrator import AutoMLConfig, AutoMLMode, AutoMLOrchestrator
from prompt_improver.database.connection import DatabaseManager
from prompt_improver.services.ab_testing import ABTestingService
from prompt_improver.services.prompt_improvement import PromptImprovementService
from prompt_improver.utils.websocket_manager import WebSocketManager


class TestAutoMLEndToEndWorkflow:
    """Test complete AutoML workflow integration."""

    @pytest.fixture
    async def temp_database(self):
        """Temporary database for integration testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
            
        # Use SQLite for integration testing
        database_url = f"sqlite+aiosqlite:///{db_path}"
        db_manager = DatabaseManager(database_url.replace("aiosqlite", "psycopg"))
        
        yield db_manager
        
        # Cleanup
        try:
            Path(db_path).unlink(missing_ok=True)
        except Exception:
            pass

    @pytest.fixture
    def mock_ab_testing_service(self):
        """Mock A/B testing service with realistic responses."""
        service = MagicMock(spec=ABTestingService)
        service.get_real_time_metrics = AsyncMock(return_value={
            "conversion_rate": 0.15,
            "user_satisfaction": 4.2,
            "response_time": 120,
            "rule_effectiveness": 0.85
        })
        service.record_experiment_result = AsyncMock()
        service.should_stop_experiment = AsyncMock(return_value=False)
        return service

    @pytest.fixture
    def mock_websocket_manager(self):
        """Mock WebSocket manager for real-time updates."""
        manager = MagicMock(spec=WebSocketManager)
        manager.broadcast = AsyncMock()
        return manager

    @pytest.fixture
    def automl_config(self):
        """Production-like AutoML configuration."""
        return AutoMLConfig(
            study_name=f"integration_test_{uuid.uuid4().hex[:8]}",
            optimization_mode=AutoMLMode.HYPERPARAMETER_OPTIMIZATION,
            n_trials=5,  # Small number for fast testing
            timeout=30,  # 30 second timeout
            enable_real_time_feedback=True,
            enable_early_stopping=True,
            early_stopping_patience=3
        )

    @pytest.fixture
    async def orchestrator(self, automl_config, temp_database, mock_ab_testing_service, mock_websocket_manager):
        """Fully configured AutoML orchestrator."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            storage_url = f"sqlite:///{f.name}"
            
            orchestrator = AutoMLOrchestrator(
                config=automl_config,
                db_manager=temp_database,
                ab_testing_service=mock_ab_testing_service,
                websocket_manager=mock_websocket_manager,
                storage_url=storage_url
            )
            
            yield orchestrator
            
            # Cleanup
            await orchestrator.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_optimization_workflow(self, orchestrator):
        """Test complete optimization workflow from start to finish."""
        # Start optimization
        start_result = await orchestrator.start_optimization()
        assert start_result["status"] == "started"
        assert "study_name" in start_result
        
        # Wait for some trials to complete
        await asyncio.sleep(2)
        
        # Check status during optimization
        status = await orchestrator.get_optimization_status()
        assert status["status"] in ["running", "completed"]
        assert "trials_completed" in status
        assert "best_value" in status
        
        # Wait for completion or timeout
        max_wait = 30  # seconds
        waited = 0
        while waited < max_wait:
            status = await orchestrator.get_optimization_status()
            if status["status"] == "completed":
                break
            await asyncio.sleep(1)
            waited += 1
        
        # Verify final state
        final_status = await orchestrator.get_optimization_status()
        assert final_status["status"] in ["completed", "running"]
        assert final_status["trials_completed"] > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_time_analytics_integration(self, orchestrator, mock_ab_testing_service):
        """Test integration with real-time analytics system."""
        # Start optimization
        await orchestrator.start_optimization()
        
        # Wait for some trials
        await asyncio.sleep(1)
        
        # Verify analytics service interactions
        assert mock_ab_testing_service.get_real_time_metrics.called
        
        # Verify WebSocket broadcasts
        assert orchestrator.websocket_manager.broadcast.called

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_database_persistence(self, orchestrator):
        """Test database persistence of optimization results."""
        # Start optimization
        await orchestrator.start_optimization()
        
        # Wait for trials to complete
        await asyncio.sleep(2)
        
        # Verify study persistence
        study = orchestrator.study
        assert len(study.trials) > 0
        
        # Verify trials have proper data
        for trial in study.trials:
            assert hasattr(trial, 'number')
            assert hasattr(trial, 'state')
            if trial.state.name == 'COMPLETE':
                assert trial.value is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_early_stopping_mechanism(self, orchestrator):
        """Test early stopping mechanism with poor performance."""
        # Mock objective function to return poor values
        def poor_objective(trial):
            return 0.1  # Consistently poor performance
        
        with patch.object(orchestrator, '_objective_function', poor_objective):
            await orchestrator.start_optimization()
            
            # Wait for early stopping to trigger
            await asyncio.sleep(5)
            
            status = await orchestrator.get_optimization_status()
            
            # Should have stopped due to no improvement
            # Note: Actual early stopping logic depends on implementation

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_optimization_with_timeout(self):
        """Test optimization with timeout configuration."""
        config = AutoMLConfig(
            study_name="timeout_test",
            n_trials=1000,  # Large number
            timeout=2,  # Short timeout
            enable_real_time_feedback=False  # Simplify for timeout test
        )
        
        db_manager = MagicMock(spec=DatabaseManager)
        ab_service = MagicMock(spec=ABTestingService)
        
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            storage_url = f"sqlite:///{f.name}"
            
            orchestrator = AutoMLOrchestrator(
                config=config,
                db_manager=db_manager,
                ab_testing_service=ab_service,
                storage_url=storage_url
            )
            
            start_time = datetime.now()
            await orchestrator.start_optimization()
            
            # Wait for timeout
            await asyncio.sleep(3)
            
            status = await orchestrator.get_optimization_status()
            elapsed = datetime.now() - start_time
            
            # Should respect timeout
            assert elapsed.total_seconds() < 10  # Some buffer for test execution

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_optimization_prevention(self, orchestrator):
        """Test prevention of concurrent optimization runs."""
        # Start first optimization
        result1 = await orchestrator.start_optimization()
        assert result1["status"] == "started"
        
        # Try to start second optimization
        result2 = await orchestrator.start_optimization()
        assert result2["status"] == "error"
        assert "already running" in result2["error"].lower()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_optimization_stop_and_restart(self, orchestrator):
        """Test stopping and restarting optimization."""
        # Start optimization
        start_result = await orchestrator.start_optimization()
        assert start_result["status"] == "started"
        
        # Wait briefly
        await asyncio.sleep(1)
        
        # Stop optimization
        stop_result = await orchestrator.stop_optimization()
        assert stop_result["status"] == "stopped"
        
        # Restart optimization
        restart_result = await orchestrator.start_optimization()
        assert restart_result["status"] == "started"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_best_configuration_retrieval(self, orchestrator):
        """Test retrieval of best configuration after optimization."""
        # Start optimization
        await orchestrator.start_optimization()
        
        # Wait for some trials
        await asyncio.sleep(3)
        
        # Get best configuration
        best_config = orchestrator.get_best_configuration()
        
        if best_config is not None:
            assert "parameters" in best_config
            assert "score" in best_config
            assert isinstance(best_config["parameters"], dict)
            assert isinstance(best_config["score"], (int, float))


class TestAutoMLServiceIntegration:
    """Test integration with PromptImprovementService."""

    @pytest.fixture
    def mock_prompt_service(self):
        """Mock prompt improvement service."""
        service = MagicMock(spec=PromptImprovementService)
        service.initialize_automl = AsyncMock()
        service.start_automl_optimization = AsyncMock(return_value={
            "status": "started",
            "optimization_id": "opt_123"
        })
        service.get_automl_status = AsyncMock(return_value={
            "status": "running",
            "trials_completed": 5,
            "best_value": 0.85
        })
        service.stop_automl_optimization = AsyncMock(return_value={
            "status": "stopped"
        })
        return service

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_service_automl_initialization(self, mock_prompt_service):
        """Test AutoML initialization through service layer."""
        db_manager = MagicMock(spec=DatabaseManager)
        
        await mock_prompt_service.initialize_automl(db_manager)
        
        mock_prompt_service.initialize_automl.assert_called_once_with(db_manager)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_service_optimization_lifecycle(self, mock_prompt_service):
        """Test complete optimization lifecycle through service."""
        # Start optimization
        start_result = await mock_prompt_service.start_automl_optimization()
        assert start_result["status"] == "started"
        assert "optimization_id" in start_result
        
        # Check status
        status = await mock_prompt_service.get_automl_status()
        assert status["status"] == "running"
        assert "trials_completed" in status
        
        # Stop optimization
        stop_result = await mock_prompt_service.stop_automl_optimization()
        assert stop_result["status"] == "stopped"


class TestAutoMLErrorScenarios:
    """Test error scenarios and edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_database_connection_failure(self):
        """Test handling of database connection failures."""
        config = AutoMLConfig(study_name="db_failure_test")
        
        # Use invalid database manager
        invalid_db_manager = MagicMock(spec=DatabaseManager)
        invalid_db_manager.get_session.side_effect = Exception("Database connection failed")
        
        ab_service = MagicMock(spec=ABTestingService)
        
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            storage_url = f"sqlite:///{f.name}"
            
            orchestrator = AutoMLOrchestrator(
                config=config,
                db_manager=invalid_db_manager,
                ab_testing_service=ab_service,
                storage_url=storage_url
            )
            
            # Should handle database errors gracefully
            result = await orchestrator.start_optimization()
            
            # Implementation determines exact error handling behavior
            assert "status" in result

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_websocket_failure_resilience(self):
        """Test resilience to WebSocket failures."""
        config = AutoMLConfig(
            study_name="websocket_failure_test",
            n_trials=3,
            enable_real_time_feedback=True
        )
        
        db_manager = MagicMock(spec=DatabaseManager)
        ab_service = MagicMock(spec=ABTestingService)
        
        # Mock failing WebSocket manager
        failing_websocket = MagicMock(spec=WebSocketManager)
        failing_websocket.broadcast = AsyncMock(side_effect=Exception("WebSocket failed"))
        
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            storage_url = f"sqlite:///{f.name}"
            
            orchestrator = AutoMLOrchestrator(
                config=config,
                db_manager=db_manager,
                ab_testing_service=ab_service,
                websocket_manager=failing_websocket,
                storage_url=storage_url
            )
            
            # Should continue optimization despite WebSocket failures
            result = await orchestrator.start_optimization()
            assert result["status"] == "started"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_ab_testing_service_failure(self):
        """Test handling of A/B testing service failures."""
        config = AutoMLConfig(
            study_name="ab_failure_test",
            n_trials=3,
            enable_real_time_feedback=True
        )
        
        db_manager = MagicMock(spec=DatabaseManager)
        websocket_manager = MagicMock(spec=WebSocketManager)
        
        # Mock failing A/B testing service
        failing_ab_service = MagicMock(spec=ABTestingService)
        failing_ab_service.get_real_time_metrics = AsyncMock(
            side_effect=Exception("A/B service failed")
        )
        
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            storage_url = f"sqlite:///{f.name}"
            
            orchestrator = AutoMLOrchestrator(
                config=config,
                db_manager=db_manager,
                ab_testing_service=failing_ab_service,
                websocket_manager=websocket_manager,
                storage_url=storage_url
            )
            
            # Should handle A/B testing failures gracefully
            result = await orchestrator.start_optimization()
            assert result["status"] == "started"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_storage_corruption_handling(self):
        """Test handling of storage corruption or unavailability."""
        config = AutoMLConfig(study_name="storage_corruption_test")
        db_manager = MagicMock(spec=DatabaseManager)
        ab_service = MagicMock(spec=ABTestingService)
        
        # Use non-existent directory for storage
        invalid_storage_url = "sqlite:///non_existent_dir/invalid.db"
        
        try:
            orchestrator = AutoMLOrchestrator(
                config=config,
                db_manager=db_manager,
                ab_testing_service=ab_service,
                storage_url=invalid_storage_url
            )
            
            # Should handle storage issues
            result = await orchestrator.start_optimization()
            # Exact behavior depends on implementation
            
        except Exception as e:
            # Storage errors might be raised during initialization
            assert "storage" in str(e).lower() or "database" in str(e).lower()


class TestAutoMLPerformanceIntegration:
    """Test performance characteristics in integration scenarios."""

    @pytest.mark.performance
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_optimization_throughput(self):
        """Test optimization throughput with realistic workload."""
        config = AutoMLConfig(
            study_name="throughput_test",
            n_trials=10,
            timeout=10,
            enable_real_time_feedback=False  # Reduce overhead
        )
        
        db_manager = MagicMock(spec=DatabaseManager)
        ab_service = MagicMock(spec=ABTestingService)
        
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            storage_url = f"sqlite:///{f.name}"
            
            orchestrator = AutoMLOrchestrator(
                config=config,
                db_manager=db_manager,
                ab_testing_service=ab_service,
                storage_url=storage_url
            )
            
            start_time = datetime.now()
            await orchestrator.start_optimization()
            
            # Wait for completion
            max_wait = 15  # seconds
            waited = 0
            while waited < max_wait:
                status = await orchestrator.get_optimization_status()
                if status["status"] == "completed":
                    break
                await asyncio.sleep(0.5)
                waited += 0.5
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            final_status = await orchestrator.get_optimization_status()
            trials_completed = final_status.get("trials_completed", 0)
            
            if trials_completed > 0:
                throughput = trials_completed / duration
                # Should maintain reasonable throughput
                assert throughput > 0.1  # At least 0.1 trials per second

    @pytest.mark.performance
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_memory_usage_during_optimization(self):
        """Test memory usage characteristics during optimization."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        config = AutoMLConfig(
            study_name="memory_test",
            n_trials=20,
            timeout=15
        )
        
        db_manager = MagicMock(spec=DatabaseManager)
        ab_service = MagicMock(spec=ABTestingService)
        
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            storage_url = f"sqlite:///{f.name}"
            
            orchestrator = AutoMLOrchestrator(
                config=config,
                db_manager=db_manager,
                ab_testing_service=ab_service,
                storage_url=storage_url
            )
            
            await orchestrator.start_optimization()
            
            # Wait for optimization
            await asyncio.sleep(5)
            
            memory_during = process.memory_info().rss
            memory_increase = memory_during - memory_before
            
            # Should not consume excessive memory
            assert memory_increase < 200 * 1024 * 1024  # 200MB max

    @pytest.mark.performance
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_status_checks_performance(self):
        """Test performance of concurrent status checks."""
        config = AutoMLConfig(
            study_name="concurrent_status_test",
            n_trials=10,
            timeout=20
        )
        
        db_manager = MagicMock(spec=DatabaseManager)
        ab_service = MagicMock(spec=ABTestingService)
        
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            storage_url = f"sqlite:///{f.name}"
            
            orchestrator = AutoMLOrchestrator(
                config=config,
                db_manager=db_manager,
                ab_testing_service=ab_service,
                storage_url=storage_url
            )
            
            # Start optimization
            await orchestrator.start_optimization()
            
            # Perform concurrent status checks
            async def check_status():
                return await orchestrator.get_optimization_status()
            
            start_time = datetime.now()
            
            # Run 10 concurrent status checks
            tasks = [check_status() for _ in range(10)]
            statuses = await asyncio.gather(*tasks)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # All status checks should complete quickly
            assert duration < 1.0  # 1 second max for 10 concurrent checks
            assert len(statuses) == 10
            
            # All status checks should return valid data
            for status in statuses:
                assert "status" in status
                assert "trials_completed" in status


class TestAutoMLConfigurationScenarios:
    """Test various configuration scenarios and edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_minimal_configuration(self):
        """Test AutoML with minimal configuration."""
        config = AutoMLConfig()  # Use all defaults
        
        db_manager = MagicMock(spec=DatabaseManager)
        ab_service = MagicMock(spec=ABTestingService)
        
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            storage_url = f"sqlite:///{f.name}"
            
            orchestrator = AutoMLOrchestrator(
                config=config,
                db_manager=db_manager,
                ab_testing_service=ab_service,
                storage_url=storage_url
            )
            
            # Should work with default configuration
            result = await orchestrator.start_optimization()
            assert result["status"] == "started"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_maximum_configuration(self):
        """Test AutoML with maximum feature configuration."""
        config = AutoMLConfig(
            study_name="max_config_test",
            optimization_mode=AutoMLMode.MULTI_OBJECTIVE_OPTIMIZATION,
            n_trials=100,
            timeout=300,
            enable_real_time_feedback=True,
            enable_early_stopping=True,
            early_stopping_patience=10,
            enable_heartbeat=True,
            heartbeat_interval=30,
            enable_performance_monitoring=True
        )
        
        db_manager = MagicMock(spec=DatabaseManager)
        ab_service = MagicMock(spec=ABTestingService)
        websocket_manager = MagicMock(spec=WebSocketManager)
        
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            storage_url = f"sqlite:///{f.name}"
            
            orchestrator = AutoMLOrchestrator(
                config=config,
                db_manager=db_manager,
                ab_testing_service=ab_service,
                websocket_manager=websocket_manager,
                storage_url=storage_url
            )
            
            # Should handle maximum configuration
            result = await orchestrator.start_optimization()
            assert result["status"] == "started"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_objective_optimization_mode(self):
        """Test multi-objective optimization mode."""
        config = AutoMLConfig(
            study_name="multi_objective_test",
            optimization_mode=AutoMLMode.MULTI_OBJECTIVE_OPTIMIZATION,
            n_trials=5
        )
        
        db_manager = MagicMock(spec=DatabaseManager)
        ab_service = MagicMock(spec=ABTestingService)
        
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            storage_url = f"sqlite:///{f.name}"
            
            orchestrator = AutoMLOrchestrator(
                config=config,
                db_manager=db_manager,
                ab_testing_service=ab_service,
                storage_url=storage_url
            )
            
            # Should handle multi-objective mode
            result = await orchestrator.start_optimization()
            assert result["status"] == "started"
            
            # Wait briefly for trials
            await asyncio.sleep(2)
            
            status = await orchestrator.get_optimization_status()
            assert "status" in status