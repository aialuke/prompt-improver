"""
Workflow Metrics Collector for ML Pipeline Orchestration.

Collects and aggregates metrics from ML workflows and integrates with existing monitoring systems.
Supports OpenTelemetry-compatible metrics export (2025 best practices).
"""
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
import uuid
from ....performance.monitoring.health.background_manager import TaskPriority, get_background_task_manager
from ..events.event_types import EventType, MLEvent

class MetricType(Enum):
    """Types of metrics collected."""
    PERFORMANCE = 'performance'
    THROUGHPUT = 'throughput'
    ERROR_RATE = 'error_rate'
    RESOURCE_UTILIZATION = 'resource_utilization'
    WORKFLOW_DURATION = 'workflow_duration'
    COMPONENT_LATENCY = 'component_latency'

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
    metrics collection, aggregation, and reporting capabilities.
    """
