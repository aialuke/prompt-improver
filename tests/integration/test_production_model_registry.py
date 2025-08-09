"""Integration tests for Production Model Registry.

Tests alias-based deployment, blue-green deployments, health monitoring,
and rollback capabilities following MLflow best practices.
"""
import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from prompt_improver.services.production_model_registry import DeploymentStrategy, ModelAlias, ModelDeploymentConfig, ModelMetrics, ProductionModelRegistry, get_production_registry

@pytest.fixture
async def temp_mlflow_registry():
    """Create temporary MLflow registry for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tracking_uri = f'file://{temp_dir}/mlruns'
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment('test_production_registry')
        registry = ProductionModelRegistry(tracking_uri=tracking_uri)
        yield registry

@pytest.fixture
def sample_model():
    """Create a sample trained model for testing."""
    X, y = make_classification(n_samples=100, n_features=31, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model

@pytest.mark.asyncio
async def test_production_registry_initialization():
    """Test production registry initialization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tracking_uri = f'file://{temp_dir}/mlruns'
        registry = ProductionModelRegistry(tracking_uri=tracking_uri)
        assert registry.client is not None
        assert registry.model_configs == {}
        assert registry.model_metrics == {}
        assert registry.deployment_history == []

@pytest.mark.asyncio
async def test_model_registration(temp_mlflow_registry, sample_model):
    """Test model registration with enhanced metadata."""
    registry = temp_mlflow_registry
    with mlflow.start_run() as run:
        model_version = await registry.register_model(model=sample_model, model_name='test_rule_optimizer', run_id=run.info.run_id, description='Test model for rule optimization', tags={'environment': 'test', 'version': '1.0'})
        assert model_version is not None
        assert model_version.name == 'test_rule_optimizer'
        assert 'Test model for rule optimization' in model_version.description
        assert 'Registered:' in model_version.description

@pytest.mark.asyncio
async def test_production_deployment(temp_mlflow_registry, sample_model):
    """Test production deployment with alias-based strategy."""
    registry = temp_mlflow_registry
    with mlflow.start_run() as run:
        model_version = await registry.register_model(model=sample_model, model_name='test_optimizer', run_id=run.info.run_id, description='Test model')
        deployment_result = await registry.deploy_model(model_name='test_optimizer', version=model_version.version, alias=ModelAlias.PRODUCTION)
        assert deployment_result['status'] == 'success'
        assert 'deployment' in deployment_result
        assert deployment_result['deployment']['alias'] == 'production'
        assert deployment_result['deployment']['version'] == model_version.version
        assert len(registry.deployment_history) == 1

@pytest.mark.asyncio
async def test_blue_green_deployment(temp_mlflow_registry, sample_model):
    """Test blue-green deployment strategy."""
    registry = temp_mlflow_registry
    with mlflow.start_run() as run1:
        model_v1 = await registry.register_model(model=sample_model, model_name='optimizer_v1', run_id=run1.info.run_id, description='Version 1')
        await registry.deploy_model(model_name='optimizer_v1', version=model_v1.version, alias=ModelAlias.PRODUCTION)
    with mlflow.start_run() as run2:
        model_v2 = await registry.register_model(model=sample_model, model_name='optimizer_v1', run_id=run2.info.run_id, description='Version 2')
        config = ModelDeploymentConfig(model_name='optimizer_v1', alias=ModelAlias.PRODUCTION, strategy=DeploymentStrategy.BLUE_GREEN, rollback_threshold=0.05)
        deployment_result = await registry.deploy_model(model_name='optimizer_v1', version=model_v2.version, alias=ModelAlias.PRODUCTION, config=config)
        assert deployment_result['status'] == 'success'
        assert deployment_result['deployment']['strategy'] == 'blue_green'
        assert len(registry.deployment_history) == 2

@pytest.mark.asyncio
async def test_model_health_monitoring(temp_mlflow_registry, sample_model):
    """Test model health monitoring and alerting."""
    registry = temp_mlflow_registry
    with mlflow.start_run() as run:
        model_version = await registry.register_model(model=sample_model, model_name='health_test_model', run_id=run.info.run_id, description='Health monitoring test')
        await registry.deploy_model(model_name='health_test_model', version=model_version.version, alias=ModelAlias.PRODUCTION)
    healthy_metrics = ModelMetrics(accuracy=0.95, precision=0.93, recall=0.97, f1_score=0.95, latency_p95=150.0, latency_p99=300.0, error_rate=0.001, prediction_count=1000, timestamp=datetime.utcnow())
    health_result = await registry.monitor_model_health(model_name='health_test_model', alias=ModelAlias.PRODUCTION, metrics=healthy_metrics)
    assert health_result['healthy'] is True
    assert len(health_result['issues']) == 0
    assert health_result['metrics']['accuracy'] == 0.95
    unhealthy_metrics = ModelMetrics(accuracy=0.65, precision=0.6, recall=0.7, f1_score=0.65, latency_p95=600.0, latency_p99=1200.0, error_rate=0.05, prediction_count=500, timestamp=datetime.utcnow())
    health_result = await registry.monitor_model_health(model_name='health_test_model', alias=ModelAlias.PRODUCTION, metrics=unhealthy_metrics)
    assert health_result['healthy'] is False
    assert len(health_result['issues']) > 0
    assert 'recommendations' in health_result

@pytest.mark.asyncio
async def test_deployment_rollback(temp_mlflow_registry, sample_model):
    """Test deployment rollback functionality."""
    registry = temp_mlflow_registry
    with mlflow.start_run() as run1:
        model_v1 = await registry.register_model(model=sample_model, model_name='rollback_test', run_id=run1.info.run_id, description='Original version')
        await registry.deploy_model(model_name='rollback_test', version=model_v1.version, alias=ModelAlias.PRODUCTION)
    with mlflow.start_run() as run2:
        model_v2 = await registry.register_model(model=sample_model, model_name='rollback_test', run_id=run2.info.run_id, description='Updated version')
        await registry.deploy_model(model_name='rollback_test', version=model_v2.version, alias=ModelAlias.PRODUCTION)
    rollback_result = await registry.rollback_deployment(model_name='rollback_test', alias=ModelAlias.PRODUCTION, reason='Performance degradation detected')
    assert rollback_result['status'] == 'success'
    assert rollback_result['rollback']['rolled_back_to'] == model_v1.version
    assert 'Performance degradation detected' in rollback_result['rollback']['reason']
    rollback_entries = [h for h in registry.deployment_history if 'rolled_back_to' in h]
    assert len(rollback_entries) == 1

@pytest.mark.asyncio
async def test_multiple_aliases(temp_mlflow_registry, sample_model):
    """Test managing multiple model aliases (champion, production, challenger)."""
    registry = temp_mlflow_registry
    with mlflow.start_run() as run:
        model_version = await registry.register_model(model=sample_model, model_name='multi_alias_test', run_id=run.info.run_id, description='Multi-alias test model')
    aliases = [ModelAlias.CHAMPION, ModelAlias.PRODUCTION, ModelAlias.STAGING]
    for alias in aliases:
        deployment_result = await registry.deploy_model(model_name='multi_alias_test', version=model_version.version, alias=alias)
        assert deployment_result['status'] == 'success'
        assert deployment_result['deployment']['alias'] == alias.value
    assert len(registry.deployment_history) == 3
    deployed_aliases = {d['alias'] for d in registry.deployment_history}
    expected_aliases = {'champion', 'production', 'staging'}
    assert deployed_aliases == expected_aliases

@pytest.mark.asyncio
async def test_model_loading_by_alias(temp_mlflow_registry, sample_model):
    """Test loading production models by alias."""
    registry = temp_mlflow_registry
    with mlflow.start_run() as run:
        model_version = await registry.register_model(model=sample_model, model_name='alias_load_test', run_id=run.info.run_id, description='Alias loading test')
        await registry.deploy_model(model_name='alias_load_test', version=model_version.version, alias=ModelAlias.CHAMPION)
    loaded_model = await registry.get_production_model(model_name='alias_load_test', alias=ModelAlias.CHAMPION)
    assert loaded_model is not None
    X_test = np.random.rand(5, 31)
    predictions = loaded_model.predict(X_test)
    assert len(predictions) == 5

@pytest.mark.asyncio
async def test_deployment_listing(temp_mlflow_registry, sample_model):
    """Test listing all deployments with status."""
    registry = temp_mlflow_registry
    model_names = ['list_test_1', 'list_test_2']
    for name in model_names:
        with mlflow.start_run() as run:
            model_version = await registry.register_model(model=sample_model, model_name=name, run_id=run.info.run_id, description=f'List test model {name}')
            await registry.deploy_model(model_name=name, version=model_version.version, alias=ModelAlias.PRODUCTION)
    deployments = await registry.list_deployments()
    assert isinstance(deployments, list)

@pytest.mark.asyncio
async def test_performance_degradation_detection(temp_mlflow_registry, sample_model):
    """Test automatic performance degradation detection."""
    registry = temp_mlflow_registry
    with mlflow.start_run() as run:
        model_version = await registry.register_model(model=sample_model, model_name='perf_test', run_id=run.info.run_id, description='Performance test')
        config = ModelDeploymentConfig(model_name='perf_test', alias=ModelAlias.PRODUCTION, strategy=DeploymentStrategy.BLUE_GREEN, rollback_threshold=0.05)
        await registry.deploy_model(model_name='perf_test', version=model_version.version, alias=ModelAlias.PRODUCTION, config=config)
    baseline_metrics = [ModelMetrics(accuracy=0.9 + i * 0.001, precision=0.88, recall=0.92, f1_score=0.9, latency_p95=100.0, latency_p99=200.0, error_rate=0.001, prediction_count=1000, timestamp=datetime.utcnow() - timedelta(hours=i)) for i in range(10, 5, -1)]
    for metrics in baseline_metrics:
        await registry.monitor_model_health(model_name='perf_test', alias=ModelAlias.PRODUCTION, metrics=metrics)
    degraded_metrics = [ModelMetrics(accuracy=0.84 + i * 0.001, precision=0.82, recall=0.86, f1_score=0.84, latency_p95=100.0, latency_p99=200.0, error_rate=0.001, prediction_count=1000, timestamp=datetime.utcnow() - timedelta(hours=i)) for i in range(5, 0, -1)]
    for metrics in degraded_metrics:
        health_result = await registry.monitor_model_health(model_name='perf_test', alias=ModelAlias.PRODUCTION, metrics=metrics)
    assert health_result['healthy'] is False
    issues = [issue for issue in health_result['issues'] if 'degraded' in issue.lower()]
    assert len(issues) > 0

@pytest.mark.asyncio
async def test_factory_function():
    """Test the factory function for getting production registry."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tracking_uri = f'file://{temp_dir}/mlruns'
        registry = await get_production_registry(tracking_uri)
        assert isinstance(registry, ProductionModelRegistry)
        assert registry.client is not None
if __name__ == '__main__':

    async def run_test():
        """Run a specific test for debugging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracking_uri = f'file://{temp_dir}/mlruns'
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment('test_debug')
            registry = ProductionModelRegistry(tracking_uri=tracking_uri)
            X, y = make_classification(n_samples=100, n_features=31, random_state=42)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            with mlflow.start_run() as run:
                model_version = await registry.register_model(model=model, model_name='debug_test', run_id=run.info.run_id, description='Debug test model')
                print(f'Registered model version: {model_version.version}')
                deployment_result = await registry.deploy_model(model_name='debug_test', version=model_version.version, alias=ModelAlias.PRODUCTION)
                print(f'Deployment result: {deployment_result}')
