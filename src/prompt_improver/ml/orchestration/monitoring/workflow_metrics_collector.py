"""
Workflow Metrics Collector for ML Pipeline Orchestration.

Collects and aggregates metrics from ML workflows and integrates with existing monitoring systems.
Supports OpenTelemetry-compatible metrics export (2025 best practices).
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import json

from ..events.event_types import EventType, MLEvent


class MetricType(Enum):
    """Types of metrics collected."""
    PERFORMANCE = "performance"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_UTILIZATION = "resource_utilization"
    WORKFLOW_DURATION = "workflow_duration"
    COMPONENT_LATENCY = "component_latency"


@dataclass
class WorkflowMetric:
    """Individual workflow metric."""
    metric_type: MetricType
    metric_name: str
    value: float
    unit: str
    component_name: Optional[str]
    workflow_id: Optional[str]
    tags: Dict[str, str]
    timestamp: datetime


@dataclass
class MetricAggregation:
    """Aggregated metric data."""
    metric_name: str
    count: int
    sum_value: float
    avg_value: float
    min_value: float
    max_value: float
    percentile_95: float
    percentile_99: float
    window_start: datetime
    window_end: datetime


class WorkflowMetricsCollector:
    """
    Collects workflow and component metrics for ML pipeline monitoring.
    
    Integrates with existing monitoring systems and provides pipeline-specific
    metrics aggregation and analysis.
    """
    
    def __init__(self, event_bus=None, config=None):
        """Initialize workflow metrics collector."""
        self.event_bus = event_bus
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.raw_metrics: List[WorkflowMetric] = []
        self.aggregated_metrics: Dict[str, MetricAggregation] = {}
        
        # Configuration
        self.collection_interval = self.config.get("collection_interval", 10)  # seconds
        self.aggregation_window = self.config.get("aggregation_window", 300)  # 5 minutes
        self.retention_hours = self.config.get("retention_hours", 48)  # 48 hours
        self.max_raw_metrics = self.config.get("max_raw_metrics", 10000)
        
        # Integration settings
        self.enable_external_export = self.config.get("enable_external_export", True)
        self.export_interval = self.config.get("export_interval", 60)  # seconds
        
        # Monitoring state
        self.is_collecting = False
        self.collection_task = None
        self.aggregation_task = None
        self.export_task = None
        
        # Performance tracking
        self.collection_stats = {
            "metrics_collected": 0,
            "aggregations_created": 0,
            "exports_completed": 0,
            "last_collection": None,
            "last_aggregation": None,
            "last_export": None
        }
    
    async def start_collection(self) -> None:
        """Start metrics collection."""
        if self.is_collecting:
            self.logger.warning("Metrics collection is already running")
            return
        
        self.logger.info("Starting workflow metrics collection")
        self.is_collecting = True
        
        # Start collection tasks
        self.collection_task = asyncio.create_task(self._collection_loop())
        self.aggregation_task = asyncio.create_task(self._aggregation_loop())
        
        if self.enable_external_export:
            self.export_task = asyncio.create_task(self._export_loop())
        
        # Emit collection started event
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.METRICS_COLLECTION_STARTED,
                source="workflow_metrics_collector",
                data={
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "collection_interval": self.collection_interval,
                    "aggregation_window": self.aggregation_window
                }
            ))
    
    async def stop_collection(self) -> None:
        """Stop metrics collection."""
        if not self.is_collecting:
            return
        
        self.logger.info("Stopping workflow metrics collection")
        self.is_collecting = False
        
        # Cancel tasks
        for task in [self.collection_task, self.aggregation_task, self.export_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Emit collection stopped event
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.METRICS_COLLECTION_STOPPED,
                source="workflow_metrics_collector",
                data={
                    "stopped_at": datetime.now(timezone.utc).isoformat(),
                    "final_stats": self.collection_stats
                }
            ))
    
    async def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        try:
            while self.is_collecting:
                # Collect metrics from various sources
                await self._collect_workflow_metrics()
                await self._collect_component_metrics()
                await self._collect_resource_metrics()
                
                # Clean up old metrics
                await self._cleanup_old_metrics()
                
                # Update collection stats
                self.collection_stats["last_collection"] = datetime.now(timezone.utc)
                
                await asyncio.sleep(self.collection_interval)
                
        except asyncio.CancelledError:
            self.logger.info("Metrics collection loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in metrics collection loop: {e}")
            self.is_collecting = False
    
    async def _aggregation_loop(self) -> None:
        """Metrics aggregation loop."""
        try:
            while self.is_collecting:
                # Perform metric aggregations
                await self._aggregate_metrics()
                
                # Update aggregation stats
                self.collection_stats["last_aggregation"] = datetime.now(timezone.utc)
                
                await asyncio.sleep(self.aggregation_window)
                
        except asyncio.CancelledError:
            self.logger.info("Metrics aggregation loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in metrics aggregation loop: {e}")
    
    async def _export_loop(self) -> None:
        """Metrics export loop."""
        try:
            while self.is_collecting:
                # Export metrics to external systems
                await self._export_metrics()
                
                # Update export stats
                self.collection_stats["last_export"] = datetime.now(timezone.utc)
                
                await asyncio.sleep(self.export_interval)
                
        except asyncio.CancelledError:
            self.logger.info("Metrics export loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in metrics export loop: {e}")
    
    async def _collect_workflow_metrics(self) -> None:
        """Collect workflow-level metrics."""
        try:
            # Collect metrics from workflow execution engine
            # This would integrate with the actual workflow engine
            
            # Simulated workflow metrics for now
            timestamp = datetime.now(timezone.utc)
            
            # Example workflow metrics
            sample_metrics = [
                WorkflowMetric(
                    metric_type=MetricType.WORKFLOW_DURATION,
                    metric_name="training_workflow_duration",
                    value=45.2,
                    unit="seconds",
                    component_name=None,
                    workflow_id="wf_001",
                    tags={"workflow_type": "training", "tier": "1"},
                    timestamp=timestamp
                ),
                WorkflowMetric(
                    metric_type=MetricType.THROUGHPUT,
                    metric_name="workflows_completed_per_minute",
                    value=3.5,
                    unit="count/minute",
                    component_name=None,
                    workflow_id=None,
                    tags={"aggregation": "global"},
                    timestamp=timestamp
                )
            ]
            
            # Add metrics to collection
            for metric in sample_metrics:
                await self._record_metric(metric)
        
        except Exception as e:
            self.logger.error(f"Error collecting workflow metrics: {e}")
    
    async def _collect_component_metrics(self) -> None:
        """Collect component-level metrics."""
        try:
            timestamp = datetime.now(timezone.utc)
            
            # Example component metrics
            component_metrics = [
                WorkflowMetric(
                    metric_type=MetricType.COMPONENT_LATENCY,
                    metric_name="training_data_loader_latency",
                    value=120.5,
                    unit="milliseconds",
                    component_name="training_data_loader",
                    workflow_id="wf_001",
                    tags={"tier": "1", "component_type": "data_loader"},
                    timestamp=timestamp
                ),
                WorkflowMetric(
                    metric_type=MetricType.ERROR_RATE,
                    metric_name="rule_optimizer_error_rate",
                    value=0.02,
                    unit="percentage",
                    component_name="rule_optimizer",
                    workflow_id=None,
                    tags={"tier": "1", "component_type": "optimizer"},
                    timestamp=timestamp
                )
            ]
            
            for metric in component_metrics:
                await self._record_metric(metric)
        
        except Exception as e:
            self.logger.error(f"Error collecting component metrics: {e}")
    
    async def _collect_resource_metrics(self) -> None:
        """Collect resource utilization metrics."""
        try:
            timestamp = datetime.now(timezone.utc)
            
            # Example resource metrics
            resource_metrics = [
                WorkflowMetric(
                    metric_type=MetricType.RESOURCE_UTILIZATION,
                    metric_name="cpu_utilization",
                    value=65.4,
                    unit="percentage",
                    component_name=None,
                    workflow_id=None,
                    tags={"resource_type": "cpu", "scope": "global"},
                    timestamp=timestamp
                ),
                WorkflowMetric(
                    metric_type=MetricType.RESOURCE_UTILIZATION,
                    metric_name="memory_utilization",
                    value=78.2,
                    unit="percentage",
                    component_name=None,
                    workflow_id=None,
                    tags={"resource_type": "memory", "scope": "global"},
                    timestamp=timestamp
                )
            ]
            
            for metric in resource_metrics:
                await self._record_metric(metric)
        
        except Exception as e:
            self.logger.error(f"Error collecting resource metrics: {e}")
    
    async def _record_metric(self, metric: WorkflowMetric) -> None:
        """Record a single metric."""
        try:
            # Add to raw metrics
            self.raw_metrics.append(metric)
            self.collection_stats["metrics_collected"] += 1
            
            # Emit metric collected event
            if self.event_bus:
                await self.event_bus.emit(MLEvent(
                    event_type=EventType.METRIC_COLLECTED,
                    source="workflow_metrics_collector",
                    data={
                        "metric": asdict(metric),
                        "collection_timestamp": datetime.now(timezone.utc).isoformat()
                    }
                ))
            
            # Trim raw metrics if too many
            if len(self.raw_metrics) > self.max_raw_metrics:
                self.raw_metrics = self.raw_metrics[-self.max_raw_metrics:]
        
        except Exception as e:
            self.logger.error(f"Error recording metric: {e}")
    
    async def _aggregate_metrics(self) -> None:
        """Aggregate metrics over the aggregation window."""
        try:
            if not self.raw_metrics:
                return
            
            window_end = datetime.now(timezone.utc)
            window_start = window_end - timedelta(seconds=self.aggregation_window)
            
            # Filter metrics in window
            window_metrics = [
                m for m in self.raw_metrics
                if window_start <= m.timestamp <= window_end
            ]
            
            if not window_metrics:
                return
            
            # Group metrics by name
            metric_groups = {}
            for metric in window_metrics:
                key = metric.metric_name
                if key not in metric_groups:
                    metric_groups[key] = []
                metric_groups[key].append(metric)
            
            # Create aggregations
            for metric_name, metrics in metric_groups.items():
                aggregation = await self._create_aggregation(metric_name, metrics, window_start, window_end)
                if aggregation:
                    self.aggregated_metrics[f"{metric_name}_{int(window_end.timestamp())}"] = aggregation
                    self.collection_stats["aggregations_created"] += 1
            
            # Clean up old aggregations
            await self._cleanup_old_aggregations()
        
        except Exception as e:
            self.logger.error(f"Error aggregating metrics: {e}")
    
    async def _create_aggregation(self, metric_name: str, metrics: List[WorkflowMetric],
                                window_start: datetime, window_end: datetime) -> Optional[MetricAggregation]:
        """Create metric aggregation from list of metrics."""
        try:
            if not metrics:
                return None
            
            values = [m.value for m in metrics]
            values.sort()
            
            count = len(values)
            sum_value = sum(values)
            avg_value = sum_value / count
            min_value = min(values)
            max_value = max(values)
            
            # Calculate percentiles
            p95_idx = int(0.95 * (count - 1))
            p99_idx = int(0.99 * (count - 1))
            percentile_95 = values[p95_idx] if p95_idx < count else max_value
            percentile_99 = values[p99_idx] if p99_idx < count else max_value
            
            return MetricAggregation(
                metric_name=metric_name,
                count=count,
                sum_value=sum_value,
                avg_value=avg_value,
                min_value=min_value,
                max_value=max_value,
                percentile_95=percentile_95,
                percentile_99=percentile_99,
                window_start=window_start,
                window_end=window_end
            )
        
        except Exception as e:
            self.logger.error(f"Error creating aggregation for {metric_name}: {e}")
            return None
    
    async def _export_metrics(self) -> None:
        """Export metrics to external monitoring systems."""
        try:
            if not self.aggregated_metrics:
                return
            
            # Prepare export data
            export_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "aggregated_metrics": {
                    key: asdict(agg) for key, agg in self.aggregated_metrics.items()
                },
                "collection_stats": self.collection_stats
            }
            
            # Export to external systems (implement based on requirements)
            await self._export_to_prometheus(export_data)
            await self._export_to_grafana(export_data)
            
            self.collection_stats["exports_completed"] += 1
            
            # Emit export completed event
            if self.event_bus:
                await self.event_bus.emit(MLEvent(
                    event_type=EventType.METRICS_EXPORTED,
                    source="workflow_metrics_collector",
                    data={
                        "export_timestamp": datetime.now(timezone.utc).isoformat(),
                        "metrics_count": len(self.aggregated_metrics),
                        "stats": self.collection_stats
                    }
                ))
        
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
    
    async def _export_to_prometheus(self, export_data: Dict[str, Any]) -> None:
        """Export metrics to Prometheus format."""
        try:
            # This would implement actual Prometheus integration
            # For now, just log the export
            self.logger.debug(f"Prometheus export: {len(export_data['aggregated_metrics'])} metrics")
        except Exception as e:
            self.logger.error(f"Error exporting to Prometheus: {e}")
    
    async def _export_to_grafana(self, export_data: Dict[str, Any]) -> None:
        """Export metrics to Grafana/InfluxDB."""
        try:
            # This would implement actual Grafana/InfluxDB integration
            # For now, just log the export
            self.logger.debug(f"Grafana export: {len(export_data['aggregated_metrics'])} metrics")
        except Exception as e:
            self.logger.error(f"Error exporting to Grafana: {e}")
    
    async def _cleanup_old_metrics(self) -> None:
        """Clean up old raw metrics."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)
            original_count = len(self.raw_metrics)
            
            self.raw_metrics = [
                metric for metric in self.raw_metrics
                if metric.timestamp > cutoff_time
            ]
            
            cleaned_count = original_count - len(self.raw_metrics)
            if cleaned_count > 0:
                self.logger.debug(f"Cleaned up {cleaned_count} old metrics")
        
        except Exception as e:
            self.logger.error(f"Error cleaning up old metrics: {e}")
    
    # OpenTelemetry-compatible methods (2025 best practices)
    
    async def export_to_otel_format(self) -> Dict[str, Any]:
        """Export metrics in OpenTelemetry format with resource attributes."""
        try:
            current_time = time.time_ns()  # OpenTelemetry uses nanoseconds
            
            # Resource attributes (promoted to labels as per 2025 best practices)
            resource_attributes = {
                "service.name": "ml-pipeline-orchestrator",
                "service.version": "1.0.0",
                "deployment.environment": self.config.get("environment", "production"),
                "ml.workflow.type": "orchestration",
                "k8s.namespace.name": self.config.get("namespace", "default"),
                "host.name": self.config.get("hostname", "unknown")
            }
            
            # Convert metrics to OpenTelemetry format
            metrics_data = []
            
            for metric in self.raw_metrics[-100:]:  # Last 100 metrics
                otel_metric = {
                    "name": f"ml_pipeline_{metric.metric_type.value}_{metric.metric_name}",
                    "description": f"ML Pipeline {metric.metric_type.value} metric",
                    "unit": metric.unit,
                    "data": {
                        "data_points": [{
                            "attributes": {
                                **resource_attributes,
                                **metric.tags,
                                "workflow_id": metric.workflow_id or "unknown",
                                "component_name": metric.component_name or "unknown"
                            },
                            "time_unix_nano": int(metric.timestamp.timestamp() * 1_000_000_000),
                            "value": metric.value
                        }]
                    }
                }
                metrics_data.append(otel_metric)
            
            # Add aggregated metrics
            for name, agg in self.aggregated_metrics.items():
                for stat_name, value in [
                    ("count", agg.count),
                    ("sum", agg.sum_value),
                    ("avg", agg.avg_value),
                    ("min", agg.min_value),
                    ("max", agg.max_value),
                    ("p95", agg.percentile_95),
                    ("p99", agg.percentile_99)
                ]:
                    otel_metric = {
                        "name": f"ml_pipeline_aggregated_{name}_{stat_name}",
                        "description": f"ML Pipeline aggregated {stat_name} for {name}",
                        "unit": "1",
                        "data": {
                            "data_points": [{
                                "attributes": {
                                    **resource_attributes,
                                    "metric_name": name,
                                    "aggregation_type": stat_name
                                },
                                "time_unix_nano": current_time,
                                "value": float(value)
                            }]
                        }
                    }
                    metrics_data.append(otel_metric)
            
            return {
                "resource_metrics": [{
                    "resource": {
                        "attributes": resource_attributes
                    },
                    "scope_metrics": [{
                        "scope": {
                            "name": "ml-pipeline-orchestrator",
                            "version": "1.0.0"
                        },
                        "metrics": metrics_data
                    }]
                }]
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting to OpenTelemetry format: {e}")
            return {}
    
    async def export_to_prometheus_otlp(self) -> List[str]:
        """Export metrics in Prometheus OTLP format (2025 compatible)."""
        try:
            prometheus_metrics = []
            current_timestamp = int(time.time() * 1000)  # Prometheus uses milliseconds
            
            # Export raw metrics
            for metric in self.raw_metrics[-50:]:  # Last 50 metrics
                metric_name = f"ml_pipeline_{metric.metric_type.value}_{metric.metric_name}"
                labels = []
                
                # Add resource attributes as labels
                if metric.workflow_id:
                    labels.append(f'workflow_id="{metric.workflow_id}"')
                if metric.component_name:
                    labels.append(f'component_name="{metric.component_name}"')
                
                # Add custom tags
                for key, value in metric.tags.items():
                    labels.append(f'{key}="{value}"')
                
                labels_str = "{" + ",".join(labels) + "}" if labels else ""
                prometheus_metrics.append(
                    f"{metric_name}{labels_str} {metric.value} {current_timestamp}"
                )
            
            # Export aggregated metrics
            for name, agg in self.aggregated_metrics.items():
                base_name = f"ml_pipeline_aggregated_{name.replace('.', '_')}"
                labels = f'{{metric_name="{name}"}}'
                
                prometheus_metrics.extend([
                    f"{base_name}_count{labels} {agg.count} {current_timestamp}",
                    f"{base_name}_sum{labels} {agg.sum_value} {current_timestamp}",
                    f"{base_name}_avg{labels} {agg.avg_value} {current_timestamp}",
                    f"{base_name}_min{labels} {agg.min_value} {current_timestamp}",
                    f"{base_name}_max{labels} {agg.max_value} {current_timestamp}",
                    f"{base_name}_p95{labels} {agg.percentile_95} {current_timestamp}",
                    f"{base_name}_p99{labels} {agg.percentile_99} {current_timestamp}"
                ])
            
            return prometheus_metrics
            
        except Exception as e:
            self.logger.error(f"Error exporting to Prometheus OTLP format: {e}")
            return []
    
    async def get_drift_detection_metrics(self) -> Dict[str, Any]:
        """Get metrics specifically for ML drift detection (2025 best practice)."""
        try:
            drift_metrics = {
                "model_performance": {},
                "data_distribution": {},
                "prediction_drift": {},
                "feature_drift": {}
            }
            
            # Extract performance metrics for drift detection
            performance_metrics = [
                m for m in self.raw_metrics 
                if m.metric_type == MetricType.PERFORMANCE
            ]
            
            if performance_metrics:
                # Calculate performance drift indicators
                recent_metrics = performance_metrics[-10:]  # Last 10 measurements
                baseline_metrics = performance_metrics[:10] if len(performance_metrics) >= 20 else []
                
                if baseline_metrics and recent_metrics:
                    recent_avg = sum(m.value for m in recent_metrics) / len(recent_metrics)
                    baseline_avg = sum(m.value for m in baseline_metrics) / len(baseline_metrics)
                    
                    drift_metrics["model_performance"] = {
                        "recent_average": recent_avg,
                        "baseline_average": baseline_avg,
                        "drift_ratio": recent_avg / baseline_avg if baseline_avg > 0 else 1.0,
                        "drift_detected": abs(recent_avg - baseline_avg) / baseline_avg > 0.1 if baseline_avg > 0 else False
                    }
            
            # Add timestamp for drift monitoring
            drift_metrics["last_updated"] = datetime.now(timezone.utc).isoformat()
            drift_metrics["metrics_available"] = len(self.raw_metrics)
            
            return drift_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating drift detection metrics: {e}")
            return {}
    
    async def _cleanup_old_aggregations(self) -> None:
        """Clean up old aggregations."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)
            original_count = len(self.aggregated_metrics)
            
            # Remove old aggregations
            keys_to_remove = [
                key for key, agg in self.aggregated_metrics.items()
                if agg.window_end < cutoff_time
            ]
            
            for key in keys_to_remove:
                del self.aggregated_metrics[key]
            
            cleaned_count = len(keys_to_remove)
            if cleaned_count > 0:
                self.logger.debug(f"Cleaned up {cleaned_count} old aggregations")
        
        except Exception as e:
            self.logger.error(f"Error cleaning up old aggregations: {e}")
    
    # Public API methods
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        recent_metrics = [
            m for m in self.raw_metrics
            if m.timestamp > datetime.now(timezone.utc) - timedelta(minutes=5)
        ]
        
        return {
            "raw_metrics_count": len(self.raw_metrics),
            "recent_metrics_count": len(recent_metrics),
            "aggregated_metrics_count": len(self.aggregated_metrics),
            "collection_stats": self.collection_stats,
            "is_collecting": self.is_collecting
        }
    
    async def get_metrics_by_type(self, metric_type: MetricType, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics by type within time window."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        filtered_metrics = [
            asdict(m) for m in self.raw_metrics
            if m.metric_type == metric_type and m.timestamp > cutoff_time
        ]
        return filtered_metrics
    
    async def get_component_metrics(self, component_name: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics for specific component."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        filtered_metrics = [
            asdict(m) for m in self.raw_metrics
            if m.component_name == component_name and m.timestamp > cutoff_time
        ]
        return filtered_metrics
    
    async def get_aggregated_metrics(self, metric_name: str = None) -> Dict[str, Any]:
        """Get aggregated metrics."""
        if metric_name:
            filtered = {
                key: asdict(agg) for key, agg in self.aggregated_metrics.items()
                if agg.metric_name == metric_name
            }
            return filtered
        else:
            return {key: asdict(agg) for key, agg in self.aggregated_metrics.items()}
    
    async def record_custom_metric(self, metric_type: MetricType, metric_name: str, value: float,
                                 unit: str = "count", component_name: str = None,
                                 workflow_id: str = None, tags: Dict[str, str] = None) -> None:
        """Record a custom metric from external sources."""
        metric = WorkflowMetric(
            metric_type=metric_type,
            metric_name=metric_name,
            value=value,
            unit=unit,
            component_name=component_name,
            workflow_id=workflow_id,
            tags=tags or {},
            timestamp=datetime.now(timezone.utc)
        )
        
        await self._record_metric(metric)