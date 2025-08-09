"""
Integration tests for evaluation pipeline workflows.
"""
import asyncio
from datetime import datetime, timezone
import pytest
from prompt_improver.core.factories.ml_pipeline_factory import MLPipelineFactory
from prompt_improver.core.protocols.ml_protocols import ServiceContainerProtocol
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import PipelineState
from prompt_improver.ml.orchestration.events.event_types import EventType

class TestEvaluationPipelineIntegration:
    """Integration tests for evaluation pipeline workflows."""

    @pytest.fixture
    async def orchestrator(self, ml_service_container):
        """Create orchestrator for evaluation testing with Protocol-based DI."""
        config = OrchestratorConfig(max_concurrent_workflows=3, evaluation_timeout=180, component_health_check_interval=3)
        factory = MLPipelineFactory()
        orchestrator = await factory.create_from_container(ml_service_container, config.to_dict())
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_complete_ab_testing_evaluation_workflow(self, orchestrator):
        """Test complete A/B testing evaluation workflow."""
        workflow_type = 'tier2_optimization'
        parameters = {'evaluation_type': 'a_b_testing', 'control_model': 'baseline_v1.0', 'treatment_model': 'optimized_v2.0', 'sample_size': 1000, 'confidence_level': 0.95, 'test_duration': '1_hour', 'metrics': ['accuracy', 'precision', 'recall', 'f1_score']}
        workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
        assert workflow_id is not None
        await self._wait_for_workflow_completion(orchestrator, workflow_id, timeout=120)
        final_status = await orchestrator.get_workflow_status(workflow_id)
        assert final_status.state == PipelineState.COMPLETED
        assert final_status.metadata is not None
        result = final_status.metadata
        assert 'evaluation_type' in result
        assert 'control_model' in result
        assert result['evaluation_type'] == 'a_b_testing'
        assert result['control_model'] == 'baseline_v1.0'

    @pytest.mark.asyncio
    async def test_statistical_validation_evaluation_workflow(self, orchestrator):
        """Test statistical validation evaluation workflow."""
        workflow_type = 'tier2_optimization'
        parameters = {'evaluation_type': 'statistical_validation', 'model_id': 'model_to_validate_v1.5', 'validation_methods': ['cross_validation', 'bootstrap', 'holdout'], 'cross_validation_folds': 5, 'bootstrap_samples': 1000, 'holdout_percentage': 0.2, 'significance_threshold': 0.05}
        workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
        await self._wait_for_workflow_completion(orchestrator, workflow_id)
        status = await orchestrator.get_workflow_status(workflow_id)
        assert status.state == PipelineState.COMPLETED
        result = status.metadata
        assert 'evaluation_type' in result
        assert result['evaluation_type'] == 'statistical_validation'

    @pytest.mark.asyncio
    async def test_pattern_analysis_evaluation_workflow(self, orchestrator):
        """Test pattern analysis evaluation workflow."""
        workflow_type = 'tier2_optimization'
        parameters = {'evaluation_type': 'pattern_analysis', 'data_source': 'training_results_comprehensive.json', 'analysis_methods': ['causal_inference', 'pattern_significance', 'structural_analysis'], 'causal_discovery_algorithm': 'pc_algorithm', 'pattern_mining_threshold': 0.1, 'significance_level': 0.05}
        workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
        await self._wait_for_workflow_completion(orchestrator, workflow_id)
        status = await orchestrator.get_workflow_status(workflow_id)
        assert status.state == PipelineState.COMPLETED
        result = status.metadata
        assert 'evaluation_type' in result
        assert result['evaluation_type'] == 'pattern_analysis'

    @pytest.mark.asyncio
    async def test_evaluation_with_experiment_orchestrator_integration(self, orchestrator):
        """Test evaluation workflow integration with ExperimentOrchestrator."""
        workflow_type = 'tier2_optimization'
        parameters = {'evaluation_type': 'a_b_testing', 'use_experiment_orchestrator': True, 'experiment_config': {'experiment_name': 'integration_test_experiment', 'control_variant': 'control_v1', 'treatment_variant': 'treatment_v1', 'traffic_allocation': {'control': 0.5, 'treatment': 0.5}, 'success_metrics': ['conversion_rate', 'click_through_rate'], 'minimum_detectable_effect': 0.05}}
        workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
        events_received = []

        def event_handler(event):
            if 'experiment' in event.data or 'orchestrator' in event.source.lower():
                events_received.append(event)
        await orchestrator.event_bus.subscribe(EventType.EVALUATION_STARTED, event_handler)
        await orchestrator.event_bus.subscribe(EventType.EVALUATION_COMPLETED, event_handler)
        await self._wait_for_workflow_completion(orchestrator, workflow_id)
        status = await orchestrator.get_workflow_status(workflow_id)
        assert status.state == PipelineState.COMPLETED
        await asyncio.sleep(2)
        assert len(events_received) >= 1

    @pytest.mark.asyncio
    async def test_evaluation_pipeline_resource_management(self, orchestrator):
        """Test evaluation pipeline resource allocation and management."""
        workflow_type = 'tier2_optimization'
        parameters = {'evaluation_type': 'statistical_validation', 'resource_intensive': True, 'resource_requirements': {'cpu': 6.0, 'memory': 12288, 'processing_time_estimate': '10_minutes'}, 'parallel_validation': True, 'validation_workers': 4}
        initial_usage = await orchestrator.get_resource_usage()
        workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
        await asyncio.sleep(2)
        active_usage = await orchestrator.get_resource_usage()
        assert active_usage.get('active_allocations', 0) >= initial_usage.get('active_allocations', 0)
        await self._wait_for_workflow_completion(orchestrator, workflow_id)
        await asyncio.sleep(2)
        final_usage = await orchestrator.get_resource_usage()
        assert final_usage.get('active_allocations', 0) <= initial_usage.get('active_allocations', 0) + 1

    @pytest.mark.asyncio
    async def test_concurrent_evaluation_workflows(self, orchestrator):
        """Test multiple concurrent evaluation workflows."""
        workflow_type = 'tier2_optimization'
        evaluations = [{'evaluation_type': 'a_b_testing', 'control_model': 'baseline', 'treatment_model': 'variant_a', 'test_name': 'test_a'}, {'evaluation_type': 'statistical_validation', 'model_id': 'model_b', 'validation_methods': ['cross_validation'], 'test_name': 'test_b'}, {'evaluation_type': 'pattern_analysis', 'data_source': 'dataset_c.json', 'test_name': 'test_c'}]
        workflow_ids = []
        for params in evaluations:
            workflow_id = await orchestrator.start_workflow(workflow_type, params)
            workflow_ids.append(workflow_id)
        assert len(workflow_ids) == 3
        assert len(set(workflow_ids)) == 3
        for workflow_id in workflow_ids:
            await self._wait_for_workflow_completion(orchestrator, workflow_id, timeout=180)
        completed_count = 0
        for workflow_id in workflow_ids:
            status = await orchestrator.get_workflow_status(workflow_id)
            if status.state == PipelineState.COMPLETED:
                completed_count += 1
        assert completed_count >= 2

    @pytest.mark.asyncio
    async def test_evaluation_failure_handling_and_recovery(self, orchestrator):
        """Test evaluation failure handling and recovery."""
        workflow_type = 'tier2_optimization'
        failing_parameters = {'evaluation_type': 'invalid_evaluation_type', 'simulate_failure': True, 'failure_point': 'experiment_setup'}
        failing_workflow_id = await orchestrator.start_workflow(workflow_type, failing_parameters)
        await self._wait_for_workflow_completion(orchestrator, failing_workflow_id, timeout=60)
        failing_status = await orchestrator.get_workflow_status(failing_workflow_id)
        assert failing_status.state == PipelineState.ERROR
        success_parameters = {'evaluation_type': 'a_b_testing', 'control_model': 'simple_control', 'treatment_model': 'simple_treatment'}
        success_workflow_id = await orchestrator.start_workflow(workflow_type, success_parameters)
        await self._wait_for_workflow_completion(orchestrator, success_workflow_id)
        success_status = await orchestrator.get_workflow_status(success_workflow_id)
        assert success_status.state == PipelineState.COMPLETED

    @pytest.mark.asyncio
    async def test_evaluation_with_real_component_interaction(self, orchestrator):
        """Test evaluation workflow with real component interactions."""
        workflow_type = 'tier2_optimization'
        parameters = {'evaluation_type': 'statistical_validation', 'use_real_components': True, 'component_interaction_test': True, 'target_components': ['advanced_statistical_validator', 'causal_inference_analyzer', 'pattern_significance_analyzer']}
        workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
        health_snapshots = []
        for _ in range(3):
            await asyncio.sleep(5)
            health = await orchestrator.get_component_health()
            health_snapshots.append(health)
        await self._wait_for_workflow_completion(orchestrator, workflow_id)
        status = await orchestrator.get_workflow_status(workflow_id)
        assert status.state in [PipelineState.COMPLETED, PipelineState.ERROR]
        assert len(health_snapshots) == 3
        for snapshot in health_snapshots:
            assert isinstance(snapshot, dict)

    @pytest.mark.asyncio
    async def test_evaluation_metrics_and_monitoring(self, orchestrator):
        """Test evaluation workflow metrics collection and monitoring."""
        workflow_type = 'tier2_optimization'
        parameters = {'evaluation_type': 'a_b_testing', 'enable_detailed_monitoring': True, 'collect_performance_metrics': True, 'monitoring_interval': 2}
        workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
        await self._wait_for_workflow_completion(orchestrator, workflow_id)
        status = await orchestrator.get_workflow_status(workflow_id)
        assert status.state == PipelineState.COMPLETED
        assert status.metadata is not None

    async def _wait_for_workflow_completion(self, orchestrator, workflow_id, timeout=120):
        """Helper method to wait for evaluation workflow completion."""
        elapsed_time = 0
        check_interval = 3
        while elapsed_time < timeout:
            status = await orchestrator.get_workflow_status(workflow_id)
            if status.state in [PipelineState.COMPLETED, PipelineState.ERROR]:
                return status
            await asyncio.sleep(check_interval)
            elapsed_time += check_interval
        final_status = await orchestrator.get_workflow_status(workflow_id)
        pytest.fail(f'Evaluation workflow {workflow_id} did not complete within {timeout}s. Final state: {final_status.state}')

class TestEvaluationPipelinePerformance:
    """Performance tests for evaluation pipelines."""

    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator for performance testing."""
        config = OrchestratorConfig(max_concurrent_workflows=5)
        orchestrator = MLPipelineOrchestrator(config)
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_evaluation_pipeline_performance_benchmarks(self, orchestrator):
        """Test evaluation pipeline performance benchmarks."""
        workflow_type = 'tier2_optimization'
        parameters = {'evaluation_type': 'statistical_validation', 'performance_benchmark': True, 'dataset_size': 'medium', 'validation_methods': ['cross_validation']}
        start_time = datetime.now()
        workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
        await self._wait_for_workflow_completion(orchestrator, workflow_id, timeout=180)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 180
        status = await orchestrator.get_workflow_status(workflow_id)
        assert status.state == PipelineState.COMPLETED

    @pytest.mark.asyncio
    async def test_high_throughput_evaluation_processing(self, orchestrator):
        """Test high-throughput evaluation processing."""
        workflow_type = 'tier2_optimization'
        workflow_ids = []
        start_time = datetime.now()
        for i in range(5):
            parameters = {'evaluation_type': 'a_b_testing', 'control_model': f'control_{i}', 'treatment_model': f'treatment_{i}', 'sample_size': 100, 'quick_evaluation': True}
            workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
            workflow_ids.append(workflow_id)
        submission_time = (datetime.now() - start_time).total_seconds()
        assert submission_time < 5
        for workflow_id in workflow_ids:
            await self._wait_for_workflow_completion(orchestrator, workflow_id, timeout=120)
        total_time = (datetime.now() - start_time).total_seconds()
        assert total_time < 120
        successful_count = 0
        for workflow_id in workflow_ids:
            status = await orchestrator.get_workflow_status(workflow_id)
            if status.state == PipelineState.COMPLETED:
                successful_count += 1
        success_rate = successful_count / len(workflow_ids)
        assert success_rate >= 0.8

    async def _wait_for_workflow_completion(self, orchestrator, workflow_id, timeout=120):
        """Helper method for performance tests."""
        elapsed_time = 0
        check_interval = 2
        while elapsed_time < timeout:
            status = await orchestrator.get_workflow_status(workflow_id)
            if status.state in [PipelineState.COMPLETED, PipelineState.ERROR]:
                return status
            await asyncio.sleep(check_interval)
            elapsed_time += check_interval
        pytest.fail(f'Workflow {workflow_id} did not complete within {timeout}s')
if __name__ == '__main__':

    async def integration_test():
        """Basic integration test for evaluation pipelines."""
        print('Running Evaluation Pipeline Integration Test...')
        config = OrchestratorConfig(max_concurrent_workflows=1)
        orchestrator = MLPipelineOrchestrator(config)
        try:
            await orchestrator.initialize()
            print('✓ Orchestrator initialized')
            workflow_id = await orchestrator.start_workflow('tier3_evaluation', {'evaluation_type': 'a_b_testing', 'control_model': 'baseline', 'treatment_model': 'optimized'})
            print(f'✓ Evaluation workflow started: {workflow_id}')
            timeout = 90
            elapsed = 0
            while elapsed < timeout:
                status = await orchestrator.get_workflow_status(workflow_id)
                if status.state in [PipelineState.COMPLETED, PipelineState.ERROR]:
                    break
                await asyncio.sleep(3)
                elapsed += 3
            final_status = await orchestrator.get_workflow_status(workflow_id)
            print(f'✓ Evaluation completed with state: {final_status.state}')
            health = await orchestrator.get_component_health()
            print(f'✓ Component health checked: {len(health)} components')
            print('✓ Integration test passed!')
        except Exception as e:
            print(f'✗ Integration test failed: {e}')
            raise
        finally:
            await orchestrator.shutdown()
            print('✓ Orchestrator shut down')
    asyncio.run(integration_test())
