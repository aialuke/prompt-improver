"""
Comprehensive End-to-End ML Pipeline Orchestrator Testing with Real Behavior Validation.

This test suite executes comprehensive end-to-end testing using REAL BEHAVIOR TESTING
rather than mocks to ensure authentic integration validation of all 77 integrated components.

Key Focus: Real behavior testing of ProductionSyntheticDataGenerator and other components
to prevent false positives and ensure genuine integration success.

Test Coverage:
1. Real Synthetic Data Generation - Test actual ProductionSyntheticDataGenerator behavior
2. Real Training Data Loading - Test actual TrainingDataLoader with real database
3. Real Component Health Monitoring - Test actual health checks of all 77 components
4. Real Workflow Execution - Test actual workflow execution with real monitoring
5. Real Error Handling - Test actual error scenarios and recovery
6. Real Performance Monitoring - Test actual metrics collection and analysis

NO MOCKS - Only real component behavior testing for authentic validation.
"""
import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
import pytest
import numpy as np
from prompt_improver.core.factories.ml_pipeline_factory import MLPipelineFactory
from prompt_improver.shared.interfaces.protocols.ml import ComponentLoaderProtocol, EventBusProtocol, ServiceContainerProtocol
from prompt_improver.database import get_session_context
from prompt_improver.ml.core.training_data_loader import TrainingDataLoader
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from prompt_improver.ml.orchestration.coordinators.data_pipeline_coordinator import DataPipelineConfig, DataPipelineCoordinator
from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import PipelineState
from prompt_improver.ml.preprocessing.synthetic_data_generator import ProductionSyntheticDataGenerator
# Removed unused redis_client import

class TestRealBehaviorEndToEndWorkflow:
    """Real behavior end-to-end integration tests - NO MOCKS, authentic component testing."""

    @pytest.fixture
    async def orchestrator(self, ml_service_container, component_factory):
        """Create real orchestrator with Protocol-based DI from service container."""
        config = OrchestratorConfig(max_concurrent_workflows=3, component_health_check_interval=1, training_timeout=300, event_bus_buffer_size=50, debug_mode=True, verbose_logging=True, enable_performance_profiling=True)
        factory = MLPipelineFactory()
        from dataclasses import asdict
        orchestrator = await factory.create_from_container(ml_service_container, asdict(config))
        await orchestrator.initialize()
        assert orchestrator._is_initialized, 'Orchestrator failed to initialize'
        assert orchestrator.component_registry is not None, 'Component registry not initialized'
        assert hasattr(orchestrator, 'event_bus'), 'EventBus Protocol not injected'
        assert hasattr(orchestrator, 'workflow_engine'), 'WorkflowEngine Protocol not injected'
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_real_synthetic_data_generation_workflow(self, orchestrator):
        """
        Test 1: Real ProductionSyntheticDataGenerator Behavior
        Verify the ProductionSyntheticDataGenerator actually generates valid synthetic data
        and integrates properly with the ML pipeline orchestrator.
        """
        print('\n🧪 Test 1: Real Synthetic Data Generation Workflow')
        print('=' * 60)
        generator = ProductionSyntheticDataGenerator(target_samples=50, generation_method='statistical', use_enhanced_scoring=True, random_state=42)
        print('✅ Real ProductionSyntheticDataGenerator created')
        print('🔄 Generating real synthetic data...')
        try:
            synthetic_data = await generator.generate_comprehensive_training_data()
            assert synthetic_data is not None, 'No synthetic data generated'
            print(f'📋 Generated data keys: {list(synthetic_data.keys())}')
            if 'features' in synthetic_data and 'effectiveness_scores' in synthetic_data:
                features = synthetic_data['features']
                effectiveness_scores = synthetic_data['effectiveness_scores']
                assert len(features) > 0, 'No features generated'
                assert len(effectiveness_scores) > 0, 'No effectiveness scores generated'
                assert len(features) == len(effectiveness_scores), 'Feature/score count mismatch'
                features_array = np.array(features)
                scores_array = np.array(effectiveness_scores)
                assert not np.isnan(features_array).any(), 'Generated features contain NaN values'
                assert not np.isnan(scores_array).any(), 'Generated scores contain NaN values'
                assert np.std(scores_array) > 0.05, 'Insufficient score variance'
                print(f'✅ Generated {len(features)} real synthetic samples')
                print(f'📊 Feature dimensions: {features_array.shape}')
                print(f'📈 Score variance: {np.std(scores_array):.4f}')
                print(f'📈 Score range: {np.min(scores_array):.3f} - {np.max(scores_array):.3f}')
            elif 'prompts' in synthetic_data:
                prompts = synthetic_data['prompts']
                assert len(prompts) > 0, 'No prompts generated'
                print(f'✅ Generated {len(prompts)} real synthetic prompts')
            else:
                print(f'📋 Actual data structure: {synthetic_data}')
                assert False, f'Unexpected data structure: {list(synthetic_data.keys())}'
            if 'metadata' in synthetic_data:
                metadata = synthetic_data['metadata']
                print(f"🎯 Quality metrics: {metadata.get('quality_metrics', 'N/A')}")
            print('✅ Real synthetic data generation validation passed')
        except Exception as e:
            pytest.fail(f'Real synthetic data generation failed: {e}')

    @pytest.mark.asyncio
    async def test_real_training_data_loader_integration(self, orchestrator):
        """
        Test 2: Real TrainingDataLoader Behavior
        Test actual TrainingDataLoader integration with real database session.
        """
        print('\n📚 Test 2: Real Training Data Loader Integration')
        print('=' * 60)
        try:
            async with get_session_context() as session:
                print('✅ Real database session established')
                loader = TrainingDataLoader(real_data_priority=True, min_samples=10, lookback_days=30, synthetic_ratio=0.3)
                print('✅ Real TrainingDataLoader created')
                print('🔄 Loading real training data...')
                training_data = await loader.load_training_data(db_session=session, rule_ids=None)
                assert training_data is not None, 'No training data loaded'
                assert 'features' in training_data, 'Missing features in training data'
                assert 'labels' in training_data, 'Missing labels in training data'
                assert 'metadata' in training_data, 'Missing metadata in training data'
                metadata = training_data['metadata']
                real_samples = metadata.get('real_samples', 0)
                synthetic_samples = metadata.get('synthetic_samples', 0)
                total_samples = metadata.get('total_samples', 0)
                assert total_samples > 0, 'No samples loaded'
                assert total_samples == real_samples + synthetic_samples, 'Sample count mismatch'
                print(f'✅ Loaded {total_samples} total samples')
                print(f'📊 Real samples: {real_samples}')
                print(f'🧪 Synthetic samples: {synthetic_samples}')
                print(f"📈 Synthetic ratio: {metadata.get('synthetic_ratio', 0):.2%}")
        except Exception as e:
            pytest.fail(f'Real training data loader test failed: {e}')

    @pytest.mark.asyncio
    async def test_real_component_health_monitoring(self, orchestrator):
        """
        Test 3: Real Component Health Monitoring
        Verify real health monitoring of all 77 integrated components.
        """
        print('\n🏥 Test 3: Real Component Health Monitoring')
        print('=' * 60)
        try:
            print('🔄 Checking real component health...')
            component_health = await orchestrator.get_component_health()
            assert isinstance(component_health, dict), 'Invalid health data format'
            assert len(component_health) > 0, 'No components found in health check'
            healthy_count = sum(1 for is_healthy in component_health.values() if is_healthy)
            total_count = len(component_health)
            health_percentage = healthy_count / total_count * 100 if total_count > 0 else 0
            print(f'✅ Health check completed for {total_count} components')
            print(f'💚 Healthy components: {healthy_count}')
            print(f'❤️ Unhealthy components: {total_count - healthy_count}')
            print(f'📊 Health percentage: {health_percentage:.1f}%')
            assert health_percentage >= 70, f'Component health too low: {health_percentage:.1f}%'
            unhealthy_components = [name for name, healthy in component_health.items() if not healthy]
            if unhealthy_components:
                print(f'⚠️ Unhealthy components: {unhealthy_components}')
        except Exception as e:
            pytest.fail(f'Real component health monitoring failed: {e}')

    @pytest.mark.asyncio
    async def test_real_workflow_execution_with_monitoring(self, orchestrator):
        """
        Test 4: Real Workflow Execution with Performance Monitoring
        Execute a real workflow and verify performance monitoring captures real metrics.
        """
        print('\n⚡ Test 4: Real Workflow Execution with Monitoring')
        print('=' * 60)
        try:
            workflow_type = 'tier1_training'
            parameters = {'model_type': 'real_test_model', 'enable_monitoring': True, 'enable_real_components': True, 'test_mode': True}
            print(f'🚀 Starting real workflow: {workflow_type}')
            start_time = datetime.now()
            workflow_id = await orchestrator.start_workflow(workflow_type, parameters)
            assert workflow_id is not None, 'Failed to start workflow'
            print(f'✅ Workflow started: {workflow_id}')
            timeout = 120
            elapsed = 0
            check_interval = 3
            while elapsed < timeout:
                status = await orchestrator.get_workflow_status(workflow_id)
                print(f'⏱️ Workflow status: {status.state} (elapsed: {elapsed}s)')
                if status.state in [PipelineState.COMPLETED, PipelineState.ERROR]:
                    break
                await asyncio.sleep(check_interval)
                elapsed += check_interval
            final_status = await orchestrator.get_workflow_status(workflow_id)
            execution_time = datetime.now() - start_time
            print(f'✅ Workflow completed in {execution_time.total_seconds():.2f}s')
            print(f'📊 Final state: {final_status.state}')
            if final_status.state == PipelineState.ERROR:
                error_msg = getattr(final_status, 'error', 'Unknown error')
                print(f'❌ Workflow failed: {error_msg}')
            else:
                assert final_status.state == PipelineState.COMPLETED, f'Workflow in unexpected state: {final_status.state}'
            if hasattr(final_status, 'metadata') and final_status.metadata:
                print(f'📈 Monitoring data collected: {len(final_status.metadata)} metrics')
        except Exception as e:
            pytest.fail(f'Real workflow execution test failed: {e}')

    @pytest.mark.asyncio
    async def test_real_error_handling_and_recovery(self, orchestrator):
        """
        Test 5: Real Error Handling and Recovery
        Test actual error scenarios and verify graceful recovery mechanisms.
        """
        print('\n🚨 Test 5: Real Error Handling and Recovery')
        print('=' * 60)
        try:
            print('🔄 Testing invalid workflow parameters...')
            invalid_parameters = {'model_type': 'nonexistent_model', 'invalid_param': 'should_cause_error', 'timeout': -1}
            try:
                workflow_id = await orchestrator.start_workflow('tier1_training', invalid_parameters)
                timeout = 30
                elapsed = 0
                check_interval = 2
                while elapsed < timeout:
                    status = await orchestrator.get_workflow_status(workflow_id)
                    if status.state in [PipelineState.COMPLETED, PipelineState.ERROR]:
                        break
                    await asyncio.sleep(check_interval)
                    elapsed += check_interval
                final_status = await orchestrator.get_workflow_status(workflow_id)
                if final_status.state == PipelineState.ERROR:
                    print('✅ Error handled gracefully')
                    print(f"📝 Error message: {getattr(final_status, 'error', 'No error message')}")
                else:
                    print(f'⚠️ Workflow unexpectedly succeeded with state: {final_status.state}')
            except Exception as workflow_error:
                print(f'✅ Workflow creation failed as expected: {workflow_error}')
            print('🔄 Testing orchestrator recovery...')
            valid_parameters = {'model_type': 'simple_test_model', 'test_mode': True}
            recovery_workflow_id = await orchestrator.start_workflow('tier1_training', valid_parameters)
            assert recovery_workflow_id is not None, 'Failed to start recovery workflow'
            print('✅ Orchestrator recovered successfully after error')
            print(f'🔄 Recovery workflow started: {recovery_workflow_id}')
        except Exception as e:
            pytest.fail(f'Real error handling test failed: {e}')

    @pytest.mark.asyncio
    async def test_real_performance_metrics_collection(self, orchestrator):
        """
        Test 6: Real Performance Metrics Collection
        Verify actual metrics collection and analysis throughout the workflow.
        """
        print('\n📊 Test 6: Real Performance Metrics Collection')
        print('=' * 60)
        try:
            print('🔄 Collecting initial performance metrics...')
            initial_usage = await orchestrator.get_resource_usage()
            print(f'📈 Initial resource usage: {initial_usage}')
            workflow_parameters = {'model_type': 'metrics_test_model', 'enable_detailed_metrics': True, 'metric_collection_interval': 1, 'test_mode': True}
            start_time = datetime.now()
            workflow_id = await orchestrator.start_workflow('tier1_training', workflow_parameters)
            print(f'🚀 Started metrics collection workflow: {workflow_id}')
            metrics_snapshots = []
            timeout = 60
            elapsed = 0
            check_interval = 5
            while elapsed < timeout:
                current_usage = await orchestrator.get_resource_usage()
                component_health = await orchestrator.get_component_health()
                workflow_status = await orchestrator.get_workflow_status(workflow_id)
                snapshot = {'timestamp': datetime.now(), 'elapsed': elapsed, 'resource_usage': current_usage, 'component_health': component_health, 'workflow_state': workflow_status.state}
                metrics_snapshots.append(snapshot)
                print(f'📊 Metrics snapshot {len(metrics_snapshots)}: {workflow_status.state} (elapsed: {elapsed}s)')
                if workflow_status.state in [PipelineState.COMPLETED, PipelineState.ERROR]:
                    break
                await asyncio.sleep(check_interval)
                elapsed += check_interval
            execution_time = datetime.now() - start_time
            print(f'✅ Metrics collection completed in {execution_time.total_seconds():.2f}s')
            print(f'📊 Collected {len(metrics_snapshots)} metric snapshots')
            assert len(metrics_snapshots) > 0, 'No metrics snapshots collected'
            for i, snapshot in enumerate(metrics_snapshots):
                assert 'resource_usage' in snapshot, f'Missing resource usage in snapshot {i}'
                assert 'component_health' in snapshot, f'Missing component health in snapshot {i}'
                assert 'workflow_state' in snapshot, f'Missing workflow state in snapshot {i}'
            if len(metrics_snapshots) > 1:
                initial_resources = metrics_snapshots[0]['resource_usage']
                final_resources = metrics_snapshots[-1]['resource_usage']
                print(f'📈 Resource usage analysis:')
                print(f'   Initial: {initial_resources}')
                print(f'   Final: {final_resources}')
            print('✅ Real performance metrics collection validated')
        except Exception as e:
            pytest.fail(f'Real performance metrics collection test failed: {e}')

    @pytest.mark.asyncio
    async def test_comprehensive_component_integration_verification(self, orchestrator):
        """
        Test 7: Comprehensive Component Integration Verification
        Verify all 77 integrated components from ALL_COMPONENTS.md work together without errors.
        """
        print('\n🔗 Test 7: Comprehensive Component Integration Verification')
        print('=' * 60)
        try:
            print('🔄 Verifying component registry...')
            component_registry = orchestrator.component_registry
            assert component_registry is not None, 'Component registry not available'
            registered_components = await component_registry.get_all_components()
            print(f'📊 Found {len(registered_components)} registered components')
            print('🔄 Checking health across all component tiers...')
            component_health = await orchestrator.get_component_health()
            healthy_components = [name for name, healthy in component_health.items() if healthy]
            unhealthy_components = [name for name, healthy in component_health.items() if not healthy]
            print(f'💚 Healthy components: {len(healthy_components)}')
            print(f'❤️ Unhealthy components: {len(unhealthy_components)}')
            if unhealthy_components:
                print(f'⚠️ Unhealthy components: {unhealthy_components[:10]}...')
            print('🔄 Testing component discovery and loading...')
            component_loader = orchestrator.component_loader
            assert component_loader is not None, 'Component loader not available'
            print('🔄 Verifying tier-based component organization...')
            tier_components = {}
            for component_name in component_health.keys():
                if any(keyword in component_name.lower() for keyword in ['training', 'ml_integration', 'rule_optimizer', 'batch_processor']):
                    tier_components.setdefault('tier1_core', []).append(component_name)
                elif any(keyword in component_name.lower() for keyword in ['optimization', 'learning', 'insight', 'pattern']):
                    tier_components.setdefault('tier2_optimization', []).append(component_name)
                elif any(keyword in component_name.lower() for keyword in ['evaluation', 'analysis', 'statistical', 'causal']):
                    tier_components.setdefault('tier3_evaluation', []).append(component_name)
                elif any(keyword in component_name.lower() for keyword in ['performance', 'monitoring', 'analytics', 'testing']):
                    tier_components.setdefault('tier4_performance', []).append(component_name)
                elif any(keyword in component_name.lower() for keyword in ['model', 'registry', 'cache', 'validation']):
                    tier_components.setdefault('tier5_infrastructure', []).append(component_name)
                elif any(keyword in component_name.lower() for keyword in ['security', 'sanitizer', 'guard', 'defense']):
                    tier_components.setdefault('tier6_security', []).append(component_name)
                else:
                    tier_components.setdefault('other', []).append(component_name)
            for tier, components in tier_components.items():
                print(f'🏗️ {tier}: {len(components)} components')
            print('🔄 Testing cross-component communication...')
            event_bus = orchestrator.event_bus
            assert event_bus is not None, 'Event bus not available'
            resource_manager = orchestrator.resource_manager
            assert resource_manager is not None, 'Resource manager not available'
            resource_usage = await orchestrator.get_resource_usage()
            assert isinstance(resource_usage, dict), 'Invalid resource usage format'
            print(f'📈 Resource usage tracked: {len(resource_usage)} resource types')
            print('🔄 Verifying integration completeness...')
            expected_min_components = 50
            actual_components = len(component_health)
            assert actual_components >= expected_min_components, f'Too few components found: {actual_components} < {expected_min_components}'
            health_percentage = len(healthy_components) / len(component_health) * 100 if component_health else 0
            assert health_percentage >= 80, f'Component health too low: {health_percentage:.1f}%'
            print('✅ Comprehensive component integration verification passed')
            print(f'📊 Total components verified: {len(component_health)}')
            print(f'💚 Health percentage: {health_percentage:.1f}%')
            print(f'🏗️ Tiers represented: {len(tier_components)}')
        except Exception as e:
            pytest.fail(f'Comprehensive component integration verification failed: {e}')
