"""
Tests for Deployment Controller.
"""
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock
import pytest
from prompt_improver.ml.orchestration.coordinators.deployment_controller import DeploymentConfig, DeploymentController
from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import PipelineState
from prompt_improver.ml.orchestration.events.event_types import EventType, MLEvent
from prompt_improver.ml.orchestration.metrics import DeploymentMetrics, DeploymentStrategy

class TestDeploymentController:
    """Test suite for Deployment Controller."""

    @pytest.fixture
    async def controller(self):
        """Create controller instance for testing."""
        config = DeploymentConfig()
        mock_event_bus = Mock()
        mock_event_bus.emit = AsyncMock()
        mock_resource_manager = Mock()
        mock_resource_manager.allocate_resources = AsyncMock(return_value=Mock(allocation_id='deploy-alloc'))
        mock_resource_manager.deallocate_resources = AsyncMock(return_value=True)
        mock_component_registry = Mock()
        mock_component_registry.get_component = AsyncMock()
        controller = DeploymentController(config, mock_event_bus, mock_resource_manager)
        await controller.initialize()
        yield controller
        await controller.shutdown()

    @pytest.mark.asyncio
    async def test_controller_initialization(self, controller):
        """Test controller initialization."""
        assert controller._is_initialized is True
        assert controller.active_deployments == {}
        assert controller.deployment_history == []

    @pytest.mark.asyncio
    async def test_start_deployment_workflow(self, controller):
        """Test starting a deployment workflow."""
        workflow_id = 'deploy-workflow-123'
        parameters = {'model_id': 'model-v2.1', 'deployment_strategy': DeploymentStrategy.BLUE_GREEN, 'target_environment': 'production', 'health_check_endpoint': '/health'}
        controller.component_registry.get_component.return_value = Mock(execute=AsyncMock(return_value={'status': 'success', 'data': {'deployed': True}}))
        result = await controller.start_deployment_workflow(workflow_id, parameters)
        assert result is not None
        assert result.workflow_id == workflow_id
        assert result.state == WorkflowState.RUNNING
        assert result.current_step == DeploymentStep.PRE_DEPLOYMENT_VALIDATION
        assert result.deployment_strategy == DeploymentStrategy.BLUE_GREEN

    @pytest.mark.asyncio
    async def test_blue_green_deployment(self, controller):
        """Test blue-green deployment strategy."""
        workflow_id = 'blue-green-deploy'
        parameters = {'model_id': 'model-v3.0', 'deployment_strategy': DeploymentStrategy.BLUE_GREEN, 'blue_environment': 'production-blue', 'green_environment': 'production-green'}
        mock_registry = Mock()
        mock_registry.execute = AsyncMock(return_value={'status': 'success', 'data': {'deployment_id': 'bg-deploy-456', 'blue_version': 'v2.1', 'green_version': 'v3.0', 'traffic_split': {'blue': 0, 'green': 100}, 'rollback_available': True}})
        controller.component_registry.get_component.return_value = mock_registry
        result = await controller._execute_blue_green_deployment(workflow_id, parameters)
        assert result['status'] == 'success'
        assert result['data']['green_version'] == 'v3.0'
        assert result['data']['traffic_split']['green'] == 100
        assert result['data']['rollback_available'] is True
        controller.component_registry.get_component.assert_called_with('production_model_registry')

    @pytest.mark.asyncio
    async def test_canary_deployment(self, controller):
        """Test canary deployment strategy."""
        workflow_id = 'canary-deploy'
        parameters = {'model_id': 'model-v2.5', 'deployment_strategy': DeploymentStrategy.CANARY, 'canary_percentage': 10, 'success_threshold': 0.95}
        mock_canary = Mock()
        mock_canary.execute = AsyncMock(return_value={'status': 'success', 'data': {'deployment_id': 'canary-deploy-789', 'canary_traffic': 10, 'canary_performance': 0.97, 'baseline_performance': 0.94, 'success_metrics': {'error_rate': 0.01, 'response_time': 150, 'throughput': 1000}, 'promotion_recommended': True}})
        controller.component_registry.get_component.return_value = mock_canary
        result = await controller._execute_canary_deployment(workflow_id, parameters)
        assert result['status'] == 'success'
        assert result['data']['canary_performance'] > result['data']['baseline_performance']
        assert result['data']['promotion_recommended'] is True
        controller.component_registry.get_component.assert_called_with('canary_testing')

    @pytest.mark.asyncio
    async def test_rolling_deployment(self, controller):
        """Test rolling deployment strategy."""
        workflow_id = 'rolling-deploy'
        parameters = {'model_id': 'model-v1.8', 'deployment_strategy': DeploymentStrategy.ROLLING, 'batch_size': 3, 'max_unavailable': 1}
        mock_deployer = Mock()
        mock_deployer.execute = AsyncMock(return_value={'status': 'success', 'data': {'deployment_id': 'rolling-deploy-321', 'total_instances': 12, 'updated_instances': 12, 'batches_completed': 4, 'deployment_time': 180, 'zero_downtime': True}})
        controller.component_registry.get_component.return_value = mock_deployer
        result = await controller._execute_rolling_deployment(workflow_id, parameters)
        assert result['status'] == 'success'
        assert result['data']['updated_instances'] == result['data']['total_instances']
        assert result['data']['zero_downtime'] is True

    @pytest.mark.asyncio
    async def test_immediate_deployment(self, controller):
        """Test immediate deployment strategy."""
        workflow_id = 'immediate-deploy'
        parameters = {'model_id': 'hotfix-v1.7.1', 'deployment_strategy': DeploymentStrategy.IMMEDIATE, 'skip_validations': False}
        mock_deployer = Mock()
        mock_deployer.execute = AsyncMock(return_value={'status': 'success', 'data': {'deployment_id': 'immediate-deploy-654', 'deployed_at': datetime.now(timezone.utc).isoformat(), 'deployment_time': 15, 'validation_passed': True, 'rollback_point_created': True}})
        controller.component_registry.get_component.return_value = mock_deployer
        result = await controller._execute_immediate_deployment(workflow_id, parameters)
        assert result['status'] == 'success'
        assert result['data']['deployment_time'] <= 30
        assert result['data']['validation_passed'] is True

    @pytest.mark.asyncio
    async def test_complete_deployment_pipeline(self, controller):
        """Test complete deployment pipeline execution."""
        workflow_id = 'complete-deploy-987'
        parameters = {'model_id': 'model-v4.0', 'deployment_strategy': DeploymentStrategy.BLUE_GREEN, 'target_environment': 'production'}
        mock_component = Mock()
        mock_component.execute = AsyncMock(side_effect=[{'status': 'success', 'data': {'validation_passed': True, 'checks_completed': 5}}, {'status': 'success', 'data': {'model_deployed': True, 'endpoint_active': True}}, {'status': 'success', 'data': {'health_status': 'healthy', 'response_time': 120}}, {'status': 'success', 'data': {'verification_passed': True, 'performance_baseline_met': True}}])
        controller.component_registry.get_component.return_value = mock_component
        deployment = await controller.start_deployment_workflow(workflow_id, parameters)
        await controller._execute_deployment_pipeline(workflow_id, parameters)
        assert mock_component.execute.call_count == 4
        status = await controller.get_deployment_status(workflow_id)
        assert status.state == WorkflowState.COMPLETED

    @pytest.mark.asyncio
    async def test_deployment_health_checks(self, controller):
        """Test deployment health checks."""
        workflow_id = 'health-check-test'
        parameters = {'health_check_endpoint': '/api/health', 'health_check_timeout': 30, 'max_retries': 3}
        mock_health = Mock()
        mock_health.execute = AsyncMock(return_value={'status': 'success', 'data': {'endpoint_healthy': True, 'response_time': 85, 'status_code': 200, 'dependencies_healthy': True, 'checks_completed': 3, 'all_checks_passed': True}})
        controller.component_registry.get_component.return_value = mock_health
        result = await controller._execute_health_checks(workflow_id, parameters)
        assert result['status'] == 'success'
        assert result['data']['endpoint_healthy'] is True
        assert result['data']['all_checks_passed'] is True
        assert result['data']['response_time'] < 100

    @pytest.mark.asyncio
    async def test_deployment_rollback(self, controller):
        """Test deployment rollback functionality."""
        workflow_id = 'rollback-test'
        parameters = {'rollback_to_version': 'v3.2', 'rollback_reason': 'Performance degradation detected'}
        mock_rollback = Mock()
        mock_rollback.execute = AsyncMock(return_value={'status': 'success', 'data': {'rollback_completed': True, 'reverted_to_version': 'v3.2', 'rollback_time': 45, 'health_check_passed': True, 'traffic_restored': True}})
        controller.component_registry.get_component.return_value = mock_rollback
        result = await controller.rollback_deployment(workflow_id, parameters)
        assert result['status'] == 'success'
        assert result['data']['rollback_completed'] is True
        assert result['data']['reverted_to_version'] == 'v3.2'
        assert result['data']['traffic_restored'] is True

    @pytest.mark.asyncio
    async def test_deployment_failure_handling(self, controller):
        """Test deployment failure handling."""
        workflow_id = 'fail-deploy'
        parameters = {'model_id': 'invalid-model'}
        mock_component = Mock()
        mock_component.execute = AsyncMock(side_effect=Exception('Model not found'))
        controller.component_registry.get_component.return_value = mock_component
        deployment = await controller.start_deployment_workflow(workflow_id, parameters)
        await controller._execute_deployment_pipeline(workflow_id, parameters)
        status = await controller.get_deployment_status(workflow_id)
        assert status.state == WorkflowState.FAILED
        assert status.error is not None

    @pytest.mark.asyncio
    async def test_deployment_metrics_collection(self, controller):
        """Test deployment metrics collection."""
        workflow_id = 'metrics-deploy'
        await controller.start_deployment_workflow(workflow_id, {'model_id': 'test-model'})
        metrics = DeploymentMetrics(workflow_id=workflow_id, deployment_strategy=DeploymentStrategy.BLUE_GREEN, deployment_time=240.5, success_rate=0.98, rollback_count=0, health_check_duration=15.2, traffic_migration_time=30.0)
        await controller._record_deployment_metrics(workflow_id, metrics)
        recorded_metrics = await controller.get_deployment_metrics(workflow_id)
        assert recorded_metrics is not None
        assert recorded_metrics.deployment_time == 240.5
        assert recorded_metrics.success_rate == 0.98
        assert recorded_metrics.rollback_count == 0

    @pytest.mark.asyncio
    async def test_concurrent_deployments(self, controller):
        """Test handling multiple concurrent deployments."""
        deployment_ids = ['deploy-1', 'deploy-2', 'deploy-3']
        parameters = {'model_id': 'concurrent-model', 'deployment_strategy': DeploymentStrategy.ROLLING}
        mock_component = Mock()
        mock_component.execute = AsyncMock(return_value={'status': 'success', 'data': {}})
        controller.component_registry.get_component.return_value = mock_component
        tasks = [controller.start_deployment_workflow(deploy_id, parameters) for deploy_id in deployment_ids]
        deployments = await asyncio.gather(*tasks)
        assert len(deployments) == 3
        assert all(d.state == WorkflowState.RUNNING for d in deployments)
        active = await controller.list_active_deployments()
        assert len(active) == 3

class TestDeploymentMetrics:
    """Test suite for DeploymentMetrics."""

    def test_metrics_creation(self):
        """Test deployment metrics creation."""
        metrics = DeploymentMetrics(workflow_id='test-deploy', deployment_strategy=DeploymentStrategy.CANARY, deployment_time=180.0, success_rate=0.96, rollback_count=1, health_check_duration=20.5, traffic_migration_time=60.0)
        assert metrics.workflow_id == 'test-deploy'
        assert metrics.deployment_strategy == DeploymentStrategy.CANARY
        assert metrics.deployment_time == 180.0
        assert metrics.success_rate == 0.96
        assert metrics.rollback_count == 1
        assert metrics.timestamp is not None

    def test_metrics_serialization(self):
        """Test metrics to/from dict conversion."""
        metrics = DeploymentMetrics(workflow_id='serialize-deploy', deployment_strategy=DeploymentStrategy.BLUE_GREEN, deployment_time=300.0, success_rate=0.99, rollback_count=0, health_check_duration=12.0, traffic_migration_time=45.0)
        metrics_dict = metrics.model_dump()
        restored_metrics = DeploymentMetrics.from_dict(metrics_dict)
        assert restored_metrics.workflow_id == metrics.workflow_id
        assert restored_metrics.deployment_strategy == metrics.deployment_strategy
        assert restored_metrics.deployment_time == metrics.deployment_time
        assert restored_metrics.success_rate == metrics.success_rate

    def test_metrics_performance_assessment(self):
        """Test deployment performance assessment."""
        good_metrics = DeploymentMetrics(workflow_id='good-deploy', deployment_strategy=DeploymentStrategy.ROLLING, deployment_time=120.0, success_rate=0.99, rollback_count=0)
        assert good_metrics.is_successful()
        assert good_metrics.is_fast_deployment(threshold=300.0)
        poor_metrics = DeploymentMetrics(workflow_id='poor-deploy', deployment_strategy=DeploymentStrategy.IMMEDIATE, deployment_time=600.0, success_rate=0.85, rollback_count=2)
        assert not poor_metrics.is_successful(threshold=0.95)
        assert not poor_metrics.is_fast_deployment(threshold=300.0)
if __name__ == '__main__':

    async def smoke_test():
        """Basic smoke test for deployment controller."""
        print('Running Deployment Controller smoke test...')
        config = OrchestratorConfig()
        mock_event_bus = Mock()
        mock_event_bus.emit = AsyncMock()
        mock_resource_manager = Mock()
        mock_resource_manager.allocate_resources = AsyncMock(return_value=Mock(allocation_id='test-alloc'))
        mock_resource_manager.deallocate_resources = AsyncMock(return_value=True)
        mock_component_registry = Mock()
        mock_component_registry.get_component = AsyncMock(return_value=Mock(execute=AsyncMock(return_value={'status': 'success', 'data': {}})))
        controller = DeploymentController(config, mock_event_bus, mock_resource_manager)
        try:
            await controller.initialize()
            print('✓ Controller initialized successfully')
            deployment = await controller.start_deployment_workflow('smoke-test', {'model_id': 'test-model', 'deployment_strategy': DeploymentStrategy.BLUE_GREEN})
            print(f'✓ Deployment workflow started: {deployment.workflow_id}')
            status = await controller.get_deployment_status('smoke-test')
            print(f'✓ Deployment status: {status.state}')
            active = await controller.list_active_deployments()
            print(f'✓ Active deployments: {len(active)}')
            stopped = await controller.stop_deployment_workflow('smoke-test')
            print(f'✓ Deployment stopped: {stopped}')
            print('✓ All basic tests passed!')
        except Exception as e:
            print(f'✗ Test failed: {e}')
            raise
        finally:
            await controller.shutdown()
            print('✓ Controller shut down gracefully')
    asyncio.run(smoke_test())
