"""
Tests for Pipeline Health Monitor.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock
import pytest
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from prompt_improver.ml.orchestration.events.event_types import EventType, MLEvent
from prompt_improver.ml.orchestration.monitoring.pipeline_health_monitor import HealthTrend, PipelineHealthMonitor, PipelineHealthSnapshot, PipelineHealthStatus

class TestPipelineHealthMonitor:
    """Test suite for Pipeline Health Monitor."""

    @pytest.fixture
    async def health_monitor(self):
        """Create health monitor instance for testing."""
        config = OrchestratorConfig(component_health_check_interval=1, pipeline_status_update_interval=2)
        mock_event_bus = Mock()
        mock_event_bus.emit = AsyncMock()
        mock_event_bus.subscribe = AsyncMock()
        mock_component_registry = Mock()
        mock_component_registry.list_components = AsyncMock(return_value=[Mock(name='training_data_loader', tier='tier1', status='healthy'), Mock(name='ml_integration', tier='tier1', status='healthy'), Mock(name='rule_optimizer', tier='tier1', status='degraded')])
        mock_orchestrator = Mock()
        mock_orchestrator.get_component_health = AsyncMock(return_value={'training_data_loader': True, 'ml_integration': True, 'rule_optimizer': False})
        monitor = PipelineHealthMonitor(config, mock_event_bus, mock_component_registry, mock_orchestrator)
        await monitor.initialize()
        yield monitor
        await monitor.shutdown()

    @pytest.mark.asyncio
    async def test_monitor_initialization(self, health_monitor):
        """Test health monitor initialization."""
        assert health_monitor._is_initialized is True
        assert health_monitor.component_health == {}
        assert len(health_monitor.health_history) == 0

    @pytest.mark.asyncio
    async def test_component_health_aggregation(self, health_monitor):
        """Test component health aggregation."""
        health_data = {'training_data_loader': PipelineHealthSnapshot(component_name='training_data_loader', is_healthy=True, response_time=120, error_rate=0.01, last_check=datetime.now(timezone.utc)), 'ml_integration': PipelineHealthSnapshot(component_name='ml_integration', is_healthy=True, response_time=200, error_rate=0.02, last_check=datetime.now(timezone.utc)), 'rule_optimizer': PipelineHealthSnapshot(component_name='rule_optimizer', is_healthy=False, response_time=500, error_rate=0.15, last_check=datetime.now(timezone.utc))}
        for component_name, health in health_data.items():
            await health_monitor._update_component_health(component_name, health)
        aggregated = await health_monitor.get_aggregated_health()
        assert aggregated.total_components == 3
        assert aggregated.healthy_components == 2
        assert aggregated.unhealthy_components == 1
        assert aggregated.overall_health_percentage == pytest.approx(66.67, rel=0.01)

    @pytest.mark.asyncio
    async def test_pipeline_health_status_determination(self, health_monitor):
        """Test pipeline health status determination."""
        all_healthy = {'comp1': PipelineHealthSnapshot('comp1', True, 100, 0.01), 'comp2': PipelineHealthSnapshot('comp2', True, 150, 0.02), 'comp3': PipelineHealthSnapshot('comp3', True, 120, 0.01)}
        for name, health in all_healthy.items():
            await health_monitor._update_component_health(name, health)
        status = await health_monitor.get_pipeline_health_status()
        assert status == PipelineHealthStatus.HEALTHY
        degraded_health = PipelineHealthSnapshot('comp2', True, 400, 0.08)
        await health_monitor._update_component_health('comp2', degraded_health)
        status = await health_monitor.get_pipeline_health_status()
        assert status == PipelineHealthStatus.DEGRADED
        failed_health = PipelineHealthSnapshot('comp1', False, 1000, 0.25)
        await health_monitor._update_component_health('comp1', failed_health)
        status = await health_monitor.get_pipeline_health_status()
        assert status == PipelineHealthStatus.CRITICAL

    @pytest.mark.asyncio
    async def test_health_trend_analysis(self, health_monitor):
        """Test health trend analysis."""
        component_name = 'test_component'
        base_time = datetime.now(timezone.utc)
        health_samples = [PipelineHealthSnapshot(component_name, False, 800, 0.2, base_time - timedelta(minutes=10)), PipelineHealthSnapshot(component_name, True, 600, 0.15, base_time - timedelta(minutes=8)), PipelineHealthSnapshot(component_name, True, 400, 0.1, base_time - timedelta(minutes=6)), PipelineHealthSnapshot(component_name, True, 200, 0.05, base_time - timedelta(minutes=4)), PipelineHealthSnapshot(component_name, True, 150, 0.02, base_time - timedelta(minutes=2))]
        for health in health_samples:
            await health_monitor._update_component_health(component_name, health)
        trend = await health_monitor.analyze_health_trend(component_name)
        assert trend == HealthTrend.IMPROVING

    @pytest.mark.asyncio
    async def test_degrading_trend_detection(self, health_monitor):
        """Test degrading trend detection."""
        component_name = 'degrading_component'
        base_time = datetime.now(timezone.utc)
        health_samples = [PipelineHealthSnapshot(component_name, True, 100, 0.01, base_time - timedelta(minutes=10)), PipelineHealthSnapshot(component_name, True, 200, 0.03, base_time - timedelta(minutes=8)), PipelineHealthSnapshot(component_name, True, 350, 0.07, base_time - timedelta(minutes=6)), PipelineHealthSnapshot(component_name, True, 500, 0.12, base_time - timedelta(minutes=4)), PipelineHealthSnapshot(component_name, False, 800, 0.2, base_time - timedelta(minutes=2))]
        for health in health_samples:
            await health_monitor._update_component_health(component_name, health)
        trend = await health_monitor.analyze_health_trend(component_name)
        assert trend == HealthTrend.DEGRADING

    @pytest.mark.asyncio
    async def test_critical_condition_monitoring(self, health_monitor):
        """Test critical condition monitoring."""
        critical_conditions = [PipelineHealthSnapshot('high_error_comp', True, 200, 0.25), PipelineHealthSnapshot('slow_comp', True, 2000, 0.05), PipelineHealthSnapshot('down_comp', False, 0, 1.0)]
        for health in critical_conditions:
            await health_monitor._update_component_health(health.component_name, health)
        critical_components = await health_monitor.get_critical_components()
        assert len(critical_components) == 3
        component_names = [comp.component_name for comp in critical_components]
        assert 'high_error_comp' in component_names
        assert 'slow_comp' in component_names
        assert 'down_comp' in component_names

    @pytest.mark.asyncio
    async def test_cascading_failure_detection(self, health_monitor):
        """Test cascading failure detection."""
        failure_sequence = [('primary_service', False, 1000, 1.0), ('dependent_service_1', False, 800, 0.8), ('dependent_service_2', False, 600, 0.6), ('backup_service', True, 300, 0.1)]
        for name, healthy, response_time, error_rate in failure_sequence:
            health = PipelineHealthSnapshot(name, healthy, response_time, error_rate)
            await health_monitor._update_component_health(name, health)
        cascading_detected = await health_monitor.detect_cascading_failures()
        assert cascading_detected is True
        analysis = await health_monitor.get_failure_analysis()
        assert analysis['cascading_failure_detected'] is True
        assert analysis['affected_components'] >= 3

    @pytest.mark.asyncio
    async def test_health_snapshot_creation(self, health_monitor):
        """Test health snapshot creation."""
        components = {'comp1': PipelineHealthSnapshot('comp1', True, 150, 0.02), 'comp2': PipelineHealthSnapshot('comp2', True, 200, 0.03), 'comp3': PipelineHealthSnapshot('comp3', False, 500, 0.15)}
        for name, health in components.items():
            await health_monitor._update_component_health(name, health)
        snapshot = await health_monitor.create_health_snapshot()
        assert snapshot is not None
        assert snapshot.total_components == 3
        assert snapshot.healthy_components == 2
        assert snapshot.pipeline_status == PipelineHealthStatus.DEGRADED
        assert snapshot.timestamp is not None

    @pytest.mark.asyncio
    async def test_health_history_retention(self, health_monitor):
        """Test health history retention."""
        component_name = 'history_test_comp'
        old_time = datetime.now(timezone.utc) - timedelta(hours=25)
        recent_time = datetime.now(timezone.utc) - timedelta(hours=1)
        old_health = PipelineHealthSnapshot(component_name, True, 200, 0.05, old_time)
        recent_health = PipelineHealthSnapshot(component_name, True, 150, 0.02, recent_time)
        await health_monitor._update_component_health(component_name, old_health)
        await health_monitor._update_component_health(component_name, recent_health)
        await health_monitor._cleanup_old_health_data()
        history = await health_monitor.get_component_health_history(component_name)
        assert len(history) == 1
        assert history[0].last_check == recent_time

    @pytest.mark.asyncio
    async def test_event_emission_on_health_changes(self, health_monitor):
        """Test event emission on health status changes."""
        component_name = 'event_test_comp'
        healthy_state = PipelineHealthSnapshot(component_name, True, 150, 0.02)
        unhealthy_state = PipelineHealthSnapshot(component_name, False, 800, 0.2)
        await health_monitor._update_component_health(component_name, healthy_state)
        await health_monitor._update_component_health(component_name, unhealthy_state)
        health_monitor.event_bus.emit.assert_called()
        calls = health_monitor.event_bus.emit.call_args_list
        health_change_events = [call for call in calls if call[0][0].event_type == EventType.COMPONENT_HEALTH_CHANGED]
        assert len(health_change_events) > 0
        event_data = health_change_events[-1][0][0].data
        assert event_data['component_name'] == component_name
        assert event_data['previous_status'] is True
        assert event_data['current_status'] is False

    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, health_monitor):
        """Test performance metrics collection."""
        performance_data = {'fast_component': PipelineHealthSnapshot('fast_component', True, 50, 0.001), 'medium_component': PipelineHealthSnapshot('medium_component', True, 200, 0.02), 'slow_component': PipelineHealthSnapshot('slow_component', True, 800, 0.05)}
        for name, health in performance_data.items():
            await health_monitor._update_component_health(name, health)
        performance = await health_monitor.get_performance_summary()
        assert performance['average_response_time'] > 0
        assert performance['max_response_time'] == 800
        assert performance['min_response_time'] == 50
        assert performance['average_error_rate'] > 0
        assert 'response_time_distribution' in performance

class TestPipelineHealthSnapshot:
    """Test suite for PipelineHealthSnapshot."""

    def test_component_health_creation(self):
        """Test component health creation."""
        health = PipelineHealthSnapshot(component_name='test_component', is_healthy=True, response_time=150, error_rate=0.02, last_check=datetime.now(timezone.utc))
        assert health.component_name == 'test_component'
        assert health.is_healthy is True
        assert health.response_time == 150
        assert health.error_rate == 0.02
        assert health.last_check is not None

    def test_component_health_serialization(self):
        """Test component health to/from dict conversion."""
        health = PipelineHealthSnapshot(component_name='serialize_test', is_healthy=False, response_time=500, error_rate=0.15, last_check=datetime.now(timezone.utc))
        health_dict = health.to_dict()
        restored_health = PipelineHealthSnapshot.from_dict(health_dict)
        assert restored_health.component_name == health.component_name
        assert restored_health.is_healthy == health.is_healthy
        assert restored_health.response_time == health.response_time
        assert restored_health.error_rate == health.error_rate

    def test_health_degradation_assessment(self):
        """Test health degradation assessment."""
        healthy = PipelineHealthSnapshot('healthy_comp', True, 100, 0.01)
        assert healthy.is_performance_degraded() is False
        slow = PipelineHealthSnapshot('slow_comp', True, 600, 0.02)
        assert slow.is_performance_degraded() is True
        errors = PipelineHealthSnapshot('error_comp', True, 200, 0.12)
        assert errors.is_performance_degraded() is True
        down = PipelineHealthSnapshot('down_comp', False, 1000, 0.5)
        assert down.is_performance_degraded() is True
if __name__ == '__main__':

    async def smoke_test():
        """Basic smoke test for pipeline health monitor."""
        print('Running Pipeline Health Monitor smoke test...')
        config = OrchestratorConfig()
        mock_event_bus = Mock()
        mock_event_bus.emit = AsyncMock()
        mock_event_bus.subscribe = AsyncMock()
        mock_component_registry = Mock()
        mock_component_registry.list_components = AsyncMock(return_value=[])
        mock_orchestrator = Mock()
        mock_orchestrator.get_component_health = AsyncMock(return_value={})
        monitor = PipelineHealthMonitor(config, mock_event_bus, mock_component_registry, mock_orchestrator)
        try:
            await monitor.initialize()
            print('✓ Monitor initialized successfully')
            health = PipelineHealthSnapshot('test_comp', True, 150, 0.02)
            await monitor._update_component_health('test_comp', health)
            print('✓ Component health updated')
            aggregated = await monitor.get_aggregated_health()
            print(f'✓ Aggregated health: {aggregated.total_components} components')
            status = await monitor.get_pipeline_health_status()
            print(f'✓ Pipeline status: {status}')
            snapshot = await monitor.create_health_snapshot()
            print(f'✓ Health snapshot created: {snapshot.timestamp}')
            print('✓ All basic tests passed!')
        except Exception as e:
            print(f'✗ Test failed: {e}')
            raise
        finally:
            await monitor.shutdown()
            print('✓ Monitor shut down gracefully')
    asyncio.run(smoke_test())
