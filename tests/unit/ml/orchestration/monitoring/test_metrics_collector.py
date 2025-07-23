"""
Tests for Workflow Metrics Collector.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, Mock

from prompt_improver.ml.orchestration.monitoring.workflow_metrics_collector import (
    WorkflowMetricsCollector, MetricType, WorkflowMetric, MetricAggregation
)
from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from prompt_improver.ml.orchestration.events.event_types import EventType, MLEvent


class TestWorkflowMetricsCollector:
    """Test suite for Workflow Metrics Collector."""
    
    @pytest.fixture
    async def metrics_collector(self):
        """Create metrics collector instance for testing."""
        config = OrchestratorConfig()
        
        # Mock dependencies
        mock_event_bus = Mock()
        mock_event_bus.emit = AsyncMock()
        mock_event_bus.subscribe = AsyncMock()
        
        collector = WorkflowMetricsCollector(config, mock_event_bus)
        await collector.initialize()
        
        yield collector
        
        await collector.shutdown()
    
    @pytest.mark.asyncio
    async def test_collector_initialization(self, metrics_collector):
        """Test metrics collector initialization."""
        assert metrics_collector._is_initialized is True
        assert metrics_collector.metrics == {}
        assert len(metrics_collector.metrics_history) == 0
    
    @pytest.mark.asyncio
    async def test_record_performance_metric(self, metrics_collector):
        """Test recording performance metrics."""
        workflow_id = "perf-test-workflow"
        
        # Record various performance metrics
        await metrics_collector.record_metric(
            workflow_id, MetricType.PERFORMANCE, "cpu_usage", 75.5, {"unit": "percentage"}
        )
        await metrics_collector.record_metric(
            workflow_id, MetricType.PERFORMANCE, "memory_usage", 2048, {"unit": "MB"}
        )
        await metrics_collector.record_metric(
            workflow_id, MetricType.PERFORMANCE, "gpu_utilization", 90.0, {"unit": "percentage"}
        )
        
        # Retrieve metrics
        workflow_metrics = await metrics_collector.get_workflow_metrics(workflow_id)
        
        assert len(workflow_metrics) == 3
        
        # Check specific metrics
        cpu_metrics = [m for m in workflow_metrics if m.metric_name == "cpu_usage"]
        assert len(cpu_metrics) == 1
        assert cpu_metrics[0].value == 75.5
        assert cpu_metrics[0].metadata["unit"] == "percentage"
    
    @pytest.mark.asyncio
    async def test_record_throughput_metric(self, metrics_collector):
        """Test recording throughput metrics."""
        workflow_id = "throughput-test"
        
        # Record throughput metrics over time
        throughput_values = [100, 120, 150, 180, 200]  # Records per second
        
        for i, value in enumerate(throughput_values):
            await metrics_collector.record_metric(
                workflow_id, MetricType.THROUGHPUT, "records_per_second", value,
                {"timestamp_offset": i * 10}  # 10-second intervals
            )
        
        # Get throughput metrics
        throughput_metrics = await metrics_collector.get_metrics_by_type(workflow_id, MetricType.THROUGHPUT)
        
        assert len(throughput_metrics) == 5
        
        # Check throughput trend (should be increasing)
        values = [m.value for m in sorted(throughput_metrics, key=lambda x: x.timestamp)]
        assert values == throughput_values
        assert values[-1] > values[0]  # Throughput improved
    
    @pytest.mark.asyncio
    async def test_record_error_rate_metric(self, metrics_collector):
        """Test recording error rate metrics."""
        workflow_id = "error-rate-test"
        
        # Record error rates
        error_rates = [0.01, 0.02, 0.05, 0.03, 0.01]  # Error rate fluctuation
        
        for rate in error_rates:
            await metrics_collector.record_metric(
                workflow_id, MetricType.ERROR_RATE, "workflow_error_rate", rate,
                {"unit": "percentage", "threshold": 0.05}
            )
        
        # Get error rate metrics
        error_metrics = await metrics_collector.get_metrics_by_type(workflow_id, MetricType.ERROR_RATE)
        
        assert len(error_metrics) == 5
        
        # Check for error spikes
        max_error_rate = max(m.value for m in error_metrics)
        assert max_error_rate == 0.05
        
        # Verify threshold metadata
        assert all(m.metadata["threshold"] == 0.05 for m in error_metrics)
    
    @pytest.mark.asyncio
    async def test_record_resource_utilization_metric(self, metrics_collector):
        """Test recording resource utilization metrics."""
        workflow_id = "resource-util-test"
        
        # Record different resource utilization metrics
        resource_metrics = [
            ("cpu_cores", 4.0, {"allocated": 8.0, "utilization_pct": 50.0}),
            ("memory_gb", 12.5, {"allocated": 16.0, "utilization_pct": 78.125}),
            ("gpu_count", 2.0, {"allocated": 2.0, "utilization_pct": 100.0}),
            ("storage_gb", 250.0, {"allocated": 500.0, "utilization_pct": 50.0}),
        ]
        
        for metric_name, value, metadata in resource_metrics:
            await metrics_collector.record_metric(
                workflow_id, MetricType.RESOURCE_UTILIZATION, metric_name, value, metadata
            )
        
        # Get resource utilization metrics
        resource_util_metrics = await metrics_collector.get_metrics_by_type(
            workflow_id, MetricType.RESOURCE_UTILIZATION
        )
        
        assert len(resource_util_metrics) == 4
        
        # Check GPU utilization (should be 100%)
        gpu_metrics = [m for m in resource_util_metrics if m.metric_name == "gpu_count"]
        assert len(gpu_metrics) == 1
        assert gpu_metrics[0].metadata["utilization_pct"] == 100.0
    
    @pytest.mark.asyncio
    async def test_record_workflow_duration_metric(self, metrics_collector):
        """Test recording workflow duration metrics."""
        workflow_id = "duration-test"
        
        # Record workflow durations for different steps
        step_durations = [
            ("data_loading", 45.2),
            ("model_training", 320.8),
            ("rule_optimization", 180.5),
            ("validation", 60.3),
            ("deployment", 25.1),
        ]
        
        for step_name, duration in step_durations:
            await metrics_collector.record_metric(
                workflow_id, MetricType.WORKFLOW_DURATION, f"{step_name}_duration", duration,
                {"unit": "seconds", "step": step_name}
            )
        
        # Get duration metrics
        duration_metrics = await metrics_collector.get_metrics_by_type(
            workflow_id, MetricType.WORKFLOW_DURATION
        )
        
        assert len(duration_metrics) == 5
        
        # Calculate total workflow duration
        total_duration = sum(m.value for m in duration_metrics)
        assert total_duration == sum(duration for _, duration in step_durations)
        
        # Check longest step (should be model_training)
        longest_step = max(duration_metrics, key=lambda x: x.value)
        assert longest_step.metadata["step"] == "model_training"
        assert longest_step.value == 320.8
    
    @pytest.mark.asyncio
    async def test_record_component_latency_metric(self, metrics_collector):
        """Test recording component latency metrics."""
        workflow_id = "latency-test"
        
        # Record latencies for different components
        component_latencies = [
            ("training_data_loader", 25.5),
            ("ml_integration", 150.2),
            ("rule_optimizer", 200.8),
            ("production_registry", 45.1),
        ]
        
        for component_name, latency in component_latencies:
            await metrics_collector.record_metric(
                workflow_id, MetricType.COMPONENT_LATENCY, f"{component_name}_latency", latency,
                {"unit": "milliseconds", "component": component_name}
            )
        
        # Get latency metrics
        latency_metrics = await metrics_collector.get_metrics_by_type(
            workflow_id, MetricType.COMPONENT_LATENCY
        )
        
        assert len(latency_metrics) == 4
        
        # Check average latency
        avg_latency = sum(m.value for m in latency_metrics) / len(latency_metrics)
        assert avg_latency > 0
        
        # Check fastest component (should be training_data_loader)
        fastest_component = min(latency_metrics, key=lambda x: x.value)
        assert fastest_component.metadata["component"] == "training_data_loader"
    
    @pytest.mark.asyncio
    async def test_metrics_aggregation(self, metrics_collector):
        """Test metrics aggregation functionality."""
        workflow_id = "aggregation-test"
        
        # Record multiple samples of the same metric
        response_times = [100, 120, 80, 150, 110, 95, 130, 105]
        
        for response_time in response_times:
            await metrics_collector.record_metric(
                workflow_id, MetricType.PERFORMANCE, "response_time", response_time,
                {"unit": "milliseconds"}
            )
        
        # Get aggregated metrics
        aggregation = await metrics_collector.get_metrics_aggregation(
            workflow_id, MetricType.PERFORMANCE, "response_time"
        )
        
        assert aggregation is not None
        assert aggregation.count == 8
        assert aggregation.average == sum(response_times) / len(response_times)
        assert aggregation.minimum == min(response_times)
        assert aggregation.maximum == max(response_times)
        
        # Check percentiles
        assert aggregation.percentile_95 > aggregation.percentile_50
        assert aggregation.percentile_99 > aggregation.percentile_95
    
    @pytest.mark.asyncio
    async def test_metrics_aggregation_with_time_window(self, metrics_collector):
        """Test metrics aggregation with time windows."""
        workflow_id = "time-window-test"
        metric_name = "cpu_usage"
        
        # Record metrics over different time periods
        base_time = datetime.now(timezone.utc)
        
        # Recent metrics (within window)
        recent_values = [70, 75, 80, 85]
        for i, value in enumerate(recent_values):
            timestamp = base_time - timedelta(minutes=i)
            metric = WorkflowMetric(
                workflow_id=workflow_id,
                metric_type=MetricType.PERFORMANCE,
                metric_name=metric_name,
                value=value,
                timestamp=timestamp
            )
            await metrics_collector._store_metric(metric)
        
        # Old metrics (outside window)
        old_values = [50, 55, 60]
        for i, value in enumerate(old_values):
            timestamp = base_time - timedelta(hours=2) - timedelta(minutes=i)
            metric = WorkflowMetric(
                workflow_id=workflow_id,
                metric_type=MetricType.PERFORMANCE,
                metric_name=metric_name,
                value=value,
                timestamp=timestamp
            )
            await metrics_collector._store_metric(metric)
        
        # Get aggregation with 1-hour window
        window_minutes = 60
        aggregation = await metrics_collector.get_metrics_aggregation(
            workflow_id, MetricType.PERFORMANCE, metric_name, window_minutes=window_minutes
        )
        
        # Should only include recent values
        assert aggregation.count == 4
        assert aggregation.average == sum(recent_values) / len(recent_values)
    
    @pytest.mark.asyncio
    async def test_external_system_integration(self, metrics_collector):
        """Test external system integration for metrics export."""
        workflow_id = "export-test"
        
        # Record various metrics
        metrics_data = [
            (MetricType.PERFORMANCE, "cpu_usage", 80.0),
            (MetricType.THROUGHPUT, "requests_per_second", 500),
            (MetricType.ERROR_RATE, "error_percentage", 0.02),
        ]
        
        for metric_type, metric_name, value in metrics_data:
            await metrics_collector.record_metric(workflow_id, metric_type, metric_name, value)
        
        # Export to Prometheus format
        prometheus_metrics = await metrics_collector.export_to_prometheus()
        
        assert prometheus_metrics is not None
        assert len(prometheus_metrics) > 0
        
        # Check that metrics are properly formatted
        for metric_line in prometheus_metrics:
            assert isinstance(metric_line, str)
            assert any(name in metric_line for name in ["cpu_usage", "requests_per_second", "error_percentage"])
        
        # Export to Grafana format
        grafana_data = await metrics_collector.export_to_grafana()
        
        assert grafana_data is not None
        assert "targets" in grafana_data
        assert len(grafana_data["targets"]) > 0
    
    @pytest.mark.asyncio
    async def test_metrics_retention_and_cleanup(self, metrics_collector):
        """Test metrics retention and cleanup."""
        workflow_id = "retention-test"
        
        # Create old metrics (beyond retention period)
        old_time = datetime.now(timezone.utc) - timedelta(hours=49)  # Older than 48h default retention
        recent_time = datetime.now(timezone.utc) - timedelta(hours=1)
        
        # Add old metric
        old_metric = WorkflowMetric(
            workflow_id=workflow_id,
            metric_type=MetricType.PERFORMANCE,
            metric_name="old_metric",
            value=100.0,
            timestamp=old_time
        )
        await metrics_collector._store_metric(old_metric)
        
        # Add recent metric
        recent_metric = WorkflowMetric(
            workflow_id=workflow_id,
            metric_type=MetricType.PERFORMANCE,
            metric_name="recent_metric",
            value=200.0,
            timestamp=recent_time
        )
        await metrics_collector._store_metric(recent_metric)
        
        # Trigger cleanup
        cleaned_count = await metrics_collector._cleanup_old_metrics()
        
        assert cleaned_count >= 1  # At least the old metric should be cleaned
        
        # Verify only recent metrics remain
        remaining_metrics = await metrics_collector.get_workflow_metrics(workflow_id)
        assert len(remaining_metrics) == 1
        assert remaining_metrics[0].metric_name == "recent_metric"
    
    @pytest.mark.asyncio
    async def test_high_volume_metrics_performance(self, metrics_collector):
        """Test high-volume metrics collection performance."""
        workflow_id = "performance-test"
        
        # Record large number of metrics quickly
        start_time = datetime.now()
        
        metrics_to_record = 100
        for i in range(metrics_to_record):
            await metrics_collector.record_metric(
                workflow_id, MetricType.PERFORMANCE, f"metric_{i % 10}", float(i),
                {"batch": i // 10}
            )
        
        end_time = datetime.now()
        recording_duration = (end_time - start_time).total_seconds()
        
        # Should be able to record 100 metrics in reasonable time (< 1 second)
        assert recording_duration < 1.0
        
        # Verify all metrics were recorded
        all_metrics = await metrics_collector.get_workflow_metrics(workflow_id)
        assert len(all_metrics) == metrics_to_record
        
        # Test retrieval performance
        start_time = datetime.now()
        retrieved_metrics = await metrics_collector.get_workflow_metrics(workflow_id)
        end_time = datetime.now()
        retrieval_duration = (end_time - start_time).total_seconds()
        
        # Retrieval should be fast (< 0.1 seconds)
        assert retrieval_duration < 0.1
        assert len(retrieved_metrics) == metrics_to_record


class TestWorkflowMetric:
    """Test suite for WorkflowMetric."""
    
    def test_metric_creation(self):
        """Test workflow metric creation."""
        metric = WorkflowMetric(
            workflow_id="test-workflow",
            metric_type=MetricType.PERFORMANCE,
            metric_name="cpu_usage",
            value=75.5,
            metadata={"unit": "percentage", "threshold": 80.0}
        )
        
        assert metric.workflow_id == "test-workflow"
        assert metric.metric_type == MetricType.PERFORMANCE
        assert metric.metric_name == "cpu_usage"
        assert metric.value == 75.5
        assert metric.metadata["unit"] == "percentage"
        assert metric.timestamp is not None
    
    def test_metric_serialization(self):
        """Test metric to/from dict conversion."""
        metric = WorkflowMetric(
            workflow_id="serialize-test",
            metric_type=MetricType.THROUGHPUT,
            metric_name="requests_per_second",
            value=450.0,
            metadata={"target": 500.0}
        )
        
        # Convert to dict and back
        metric_dict = metric.to_dict()
        restored_metric = WorkflowMetric.from_dict(metric_dict)
        
        assert restored_metric.workflow_id == metric.workflow_id
        assert restored_metric.metric_type == metric.metric_type
        assert restored_metric.metric_name == metric.metric_name
        assert restored_metric.value == metric.value
        assert restored_metric.metadata == metric.metadata
    
    def test_metric_age_calculation(self):
        """Test metric age calculation."""
        old_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        metric = WorkflowMetric(
            workflow_id="age-test",
            metric_type=MetricType.PERFORMANCE,
            metric_name="test_metric",
            value=100.0,
            timestamp=old_time
        )
        
        age = metric.age
        assert age.total_seconds() >= 30 * 60  # At least 30 minutes
        assert age.total_seconds() < 31 * 60   # Less than 31 minutes


class TestMetricAggregation:
    """Test suite for MetricAggregation."""
    
    def test_aggregation_creation(self):
        """Test metrics aggregation creation."""
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        aggregation = MetricAggregation(
            metric_name="test_metric",
            metric_type=MetricType.PERFORMANCE,
            count=len(values),
            average=sum(values) / len(values),
            minimum=min(values),
            maximum=max(values),
            percentile_50=50.0,
            percentile_95=95.0,
            percentile_99=99.0
        )
        
        assert aggregation.count == 10
        assert aggregation.average == 55.0
        assert aggregation.minimum == 10
        assert aggregation.maximum == 100
        assert aggregation.percentile_50 == 50.0
        assert aggregation.percentile_95 == 95.0
        assert aggregation.percentile_99 == 99.0
    
    def test_aggregation_serialization(self):
        """Test aggregation to/from dict conversion."""
        aggregation = MetricAggregation(
            metric_name="serialize_metric",
            metric_type=MetricType.ERROR_RATE,
            count=50,
            average=0.05,
            minimum=0.01,
            maximum=0.15,
            percentile_50=0.04,
            percentile_95=0.12,
            percentile_99=0.14
        )
        
        # Convert to dict and back
        agg_dict = aggregation.to_dict()
        restored_agg = MetricAggregation.from_dict(agg_dict)
        
        assert restored_agg.metric_name == aggregation.metric_name
        assert restored_agg.metric_type == aggregation.metric_type
        assert restored_agg.count == aggregation.count
        assert restored_agg.average == aggregation.average


if __name__ == "__main__":
    # Run basic smoke test
    async def smoke_test():
        """Basic smoke test for workflow metrics collector."""
        print("Running Workflow Metrics Collector smoke test...")
        
        # Create and initialize collector
        config = OrchestratorConfig()
        mock_event_bus = Mock()
        mock_event_bus.emit = AsyncMock()
        mock_event_bus.subscribe = AsyncMock()
        
        collector = WorkflowMetricsCollector(config, mock_event_bus)
        
        try:
            await collector.initialize()
            print("✓ Collector initialized successfully")
            
            # Test metric recording
            workflow_id = "smoke-test"
            await collector.record_metric(workflow_id, MetricType.PERFORMANCE, "cpu_usage", 75.0)
            print("✓ Metric recorded successfully")
            
            # Test metric retrieval
            metrics = await collector.get_workflow_metrics(workflow_id)
            print(f"✓ Metrics retrieved: {len(metrics)} metrics")
            
            # Test aggregation
            aggregation = await collector.get_metrics_aggregation(
                workflow_id, MetricType.PERFORMANCE, "cpu_usage"
            )
            print(f"✓ Metrics aggregated: count={aggregation.count}")
            
            # Test export
            prometheus_data = await collector.export_to_prometheus()
            print(f"✓ Prometheus export: {len(prometheus_data)} metric lines")
            
            print("✓ All basic tests passed!")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            raise
        finally:
            await collector.shutdown()
            print("✓ Collector shut down gracefully")
    
    # Run the smoke test
    asyncio.run(smoke_test())