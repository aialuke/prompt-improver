"""Performance baseline data models and metrics definitions."""

from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import statistics
import uuid

# Import system monitoring
try:
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Performance monitoring integration
try:
    METRICS_REGISTRY_AVAILABLE = True
except ImportError:
    METRICS_REGISTRY_AVAILABLE = False

class MetricType(Enum):
    """Types of performance metrics."""
    LATENCY = "latency"          # Response time, processing duration
    THROUGHPUT = "throughput"    # Requests per second, operations per minute
    ERROR_RATE = "error_rate"    # Percentage of failed operations
    RESOURCE = "resource"        # CPU, memory, disk, network usage
    BUSINESS = "business"        # Revenue impact, user satisfaction
    SATURATION = "saturation"    # Queue depth, connection pool usage
    AVAILABILITY = "availability" # Uptime percentage

class TrendDirection(Enum):
    """Performance trend directions."""
    IMPROVING = "improving"      # Performance getting better
    STABLE = "stable"           # Performance consistent
    DEGRADING = "degrading"      # Performance declining
    VOLATILE = "volatile"        # Performance inconsistent
    UNKNOWN = "unknown"          # Insufficient data

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class MetricDefinition:
    """Definition of a performance metric."""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    target_value: Optional[float] = None
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    lower_is_better: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate metric definition."""
        if self.metric_type == MetricType.THROUGHPUT:
            self.lower_is_better = False
        elif self.metric_type == MetricType.AVAILABILITY:
            self.lower_is_better = False

@dataclass
class MetricValue:
    """A single metric measurement."""
    metric_name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_name': self.metric_name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricValue':
        """Create from dictionary."""
        return cls(
            metric_name=data['metric_name'],
            value=data['value'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            tags=data.get('tags', {}),
            metadata=data.get('metadata', {})
        )

@dataclass
class BaselineMetrics:
    """Complete set of baseline performance metrics."""
    baseline_id: str
    collection_timestamp: datetime
    duration_seconds: float
    
    # Core performance metrics
    response_times: List[float] = field(default_factory=list)
    error_rates: List[float] = field(default_factory=list)
    throughput_values: List[float] = field(default_factory=list)
    
    # System resource metrics
    cpu_utilization: List[float] = field(default_factory=list)
    memory_utilization: List[float] = field(default_factory=list)
    disk_usage: List[float] = field(default_factory=list)
    network_io: List[float] = field(default_factory=list)
    
    # Application-specific metrics
    database_connection_time: List[float] = field(default_factory=list)
    cache_hit_rate: List[float] = field(default_factory=list)
    queue_depth: List[float] = field(default_factory=list)
    
    # Business metrics
    business_transactions: List[float] = field(default_factory=list)
    user_satisfaction_score: List[float] = field(default_factory=list)
    
    # Additional measurements
    custom_metrics: Dict[str, List[float]] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize baseline ID if not provided."""
        if not self.baseline_id:
            self.baseline_id = str(uuid.uuid4())
    
    def add_metric_value(self, metric_name: str, value: float):
        """Add a metric value to the appropriate list."""
        metric_mapping = {
            'response_time': 'response_times',
            'error_rate': 'error_rates',
            'throughput': 'throughput_values',
            'cpu_utilization': 'cpu_utilization',
            'memory_utilization': 'memory_utilization',
            'disk_usage': 'disk_usage',
            'network_io': 'network_io',
            'database_connection_time': 'database_connection_time',
            'cache_hit_rate': 'cache_hit_rate',
            'queue_depth': 'queue_depth',
            'business_transactions': 'business_transactions',
            'user_satisfaction_score': 'user_satisfaction_score'
        }
        
        if metric_name in metric_mapping:
            attr_name = metric_mapping[metric_name]
            getattr(self, attr_name).append(value)
        else:
            # Store in custom metrics
            if metric_name not in self.custom_metrics:
                self.custom_metrics[metric_name] = []
            self.custom_metrics[metric_name].append(value)
    
    def get_metric_statistics(self, metric_name: str) -> Dict[str, float]:
        """Calculate statistics for a specific metric."""
        values = self._get_metric_values(metric_name)
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'p50': self._percentile(values, 50),
            'p90': self._percentile(values, 90),
            'p95': self._percentile(values, 95),
            'p99': self._percentile(values, 99)
        }
    
    def _get_metric_values(self, metric_name: str) -> List[float]:
        """Get values for a specific metric."""
        metric_mapping = {
            'response_time': self.response_times,
            'error_rate': self.error_rates,
            'throughput': self.throughput_values,
            'cpu_utilization': self.cpu_utilization,
            'memory_utilization': self.memory_utilization,
            'disk_usage': self.disk_usage,
            'network_io': self.network_io,
            'database_connection_time': self.database_connection_time,
            'cache_hit_rate': self.cache_hit_rate,
            'queue_depth': self.queue_depth,
            'business_transactions': self.business_transactions,
            'user_satisfaction_score': self.user_satisfaction_score
        }
        
        if metric_name in metric_mapping:
            return metric_mapping[metric_name]
        elif metric_name in self.custom_metrics:
            return self.custom_metrics[metric_name]
        else:
            return []
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'baseline_id': self.baseline_id,
            'collection_timestamp': self.collection_timestamp.isoformat(),
            'duration_seconds': self.duration_seconds,
            'response_times': self.response_times,
            'error_rates': self.error_rates,
            'throughput_values': self.throughput_values,
            'cpu_utilization': self.cpu_utilization,
            'memory_utilization': self.memory_utilization,
            'disk_usage': self.disk_usage,
            'network_io': self.network_io,
            'database_connection_time': self.database_connection_time,
            'cache_hit_rate': self.cache_hit_rate,
            'queue_depth': self.queue_depth,
            'business_transactions': self.business_transactions,
            'user_satisfaction_score': self.user_satisfaction_score,
            'custom_metrics': self.custom_metrics,
            'tags': self.tags,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaselineMetrics':
        """Create from dictionary."""
        return cls(
            baseline_id=data['baseline_id'],
            collection_timestamp=datetime.fromisoformat(data['collection_timestamp']),
            duration_seconds=data['duration_seconds'],
            response_times=data.get('response_times', []),
            error_rates=data.get('error_rates', []),
            throughput_values=data.get('throughput_values', []),
            cpu_utilization=data.get('cpu_utilization', []),
            memory_utilization=data.get('memory_utilization', []),
            disk_usage=data.get('disk_usage', []),
            network_io=data.get('network_io', []),
            database_connection_time=data.get('database_connection_time', []),
            cache_hit_rate=data.get('cache_hit_rate', []),
            queue_depth=data.get('queue_depth', []),
            business_transactions=data.get('business_transactions', []),
            user_satisfaction_score=data.get('user_satisfaction_score', []),
            custom_metrics=data.get('custom_metrics', {}),
            tags=data.get('tags', {}),
            metadata=data.get('metadata', {})
        )

@dataclass
class PerformanceTrend:
    """Performance trend analysis results."""
    metric_name: str
    direction: TrendDirection
    magnitude: float  # Percentage change
    confidence_score: float  # 0-1 scale
    timeframe_start: datetime
    timeframe_end: datetime
    sample_count: int
    
    # Statistical data
    baseline_mean: float
    current_mean: float
    variance_ratio: float
    p_value: Optional[float] = None
    
    # Trend prediction
    predicted_value_24h: Optional[float] = None
    predicted_value_7d: Optional[float] = None
    
    def is_significant(self, threshold: float = 0.05) -> bool:
        """Check if trend is statistically significant."""
        return self.p_value is not None and self.p_value < threshold

@dataclass
class RegressionAlert:
    """Performance regression alert."""
    alert_id: str
    metric_name: str
    severity: AlertSeverity
    message: str
    
    # Values
    current_value: float
    baseline_value: float
    threshold_value: float
    degradation_percentage: float
    
    # Timing
    detection_timestamp: datetime
    alert_timestamp: datetime
    
    # Context
    affected_operations: List[str] = field(default_factory=list)
    probable_causes: List[str] = field(default_factory=list)
    remediation_suggestions: List[str] = field(default_factory=list)
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize alert ID if not provided."""
        if not self.alert_id:
            self.alert_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'alert_id': self.alert_id,
            'metric_name': self.metric_name,
            'severity': self.severity.value,
            'message': self.message,
            'current_value': self.current_value,
            'baseline_value': self.baseline_value,
            'threshold_value': self.threshold_value,
            'degradation_percentage': self.degradation_percentage,
            'detection_timestamp': self.detection_timestamp.isoformat(),
            'alert_timestamp': self.alert_timestamp.isoformat(),
            'affected_operations': self.affected_operations,
            'probable_causes': self.probable_causes,
            'remediation_suggestions': self.remediation_suggestions,
            'tags': self.tags,
            'metadata': self.metadata
        }

@dataclass
class ProfileData:
    """Code profiling data."""
    profile_id: str
    operation_name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    
    # Call stack information
    function_calls: List[Dict[str, Any]] = field(default_factory=list)
    memory_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    cpu_samples: List[float] = field(default_factory=list)
    
    # Performance hotspots
    slow_functions: List[Dict[str, Any]] = field(default_factory=list)
    memory_allocations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize profile ID if not provided."""
        if not self.profile_id:
            self.profile_id = str(uuid.uuid4())

@dataclass
class BaselineComparison:
    """Comparison between two baselines."""
    baseline_a_id: str
    baseline_b_id: str
    comparison_timestamp: datetime
    
    # Metric comparisons
    metric_comparisons: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Overall assessment
    overall_improvement: bool
    significant_regressions: List[str] = field(default_factory=list)
    significant_improvements: List[str] = field(default_factory=list)
    
    # Summary statistics
    total_metrics_compared: int = 0
    metrics_improved: int = 0
    metrics_degraded: int = 0
    metrics_stable: int = 0
    
    def add_metric_comparison(
        self, 
        metric_name: str, 
        baseline_value: float,
        current_value: float,
        change_percentage: float,
        is_significant: bool,
        lower_is_better: bool = True
    ):
        """Add a metric comparison result."""
        self.metric_comparisons[metric_name] = {
            'baseline_value': baseline_value,
            'current_value': current_value,
            'change_percentage': change_percentage,
            'is_significant': is_significant,
            'lower_is_better': lower_is_better,
            'assessment': self._assess_change(change_percentage, is_significant, lower_is_better)
        }
        
        self.total_metrics_compared += 1
        
        if is_significant:
            if (lower_is_better and change_percentage < 0) or (not lower_is_better and change_percentage > 0):
                self.metrics_improved += 1
                self.significant_improvements.append(metric_name)
            elif (lower_is_better and change_percentage > 0) or (not lower_is_better and change_percentage < 0):
                self.metrics_degraded += 1
                self.significant_regressions.append(metric_name)
        else:
            self.metrics_stable += 1
        
        # Update overall assessment
        self.overall_improvement = self.metrics_improved > self.metrics_degraded
    
    def _assess_change(self, change_percentage: float, is_significant: bool, lower_is_better: bool) -> str:
        """Assess the nature of a metric change."""
        if not is_significant:
            return "stable"
        
        if lower_is_better:
            return "improvement" if change_percentage < 0 else "regression"
        else:
            return "improvement" if change_percentage > 0 else "regression"

# Standard metric definitions for the system
STANDARD_METRICS = {
    'response_time_p95': MetricDefinition(
        name='response_time_p95',
        metric_type=MetricType.LATENCY,
        description='95th percentile response time',
        unit='milliseconds',
        target_value=200.0,
        warning_threshold=150.0,
        critical_threshold=200.0,
        lower_is_better=True
    ),
    'error_rate': MetricDefinition(
        name='error_rate',
        metric_type=MetricType.ERROR_RATE,
        description='Percentage of failed requests',
        unit='percentage',
        target_value=1.0,
        warning_threshold=5.0,
        critical_threshold=10.0,
        lower_is_better=True
    ),
    'throughput_rps': MetricDefinition(
        name='throughput_rps',
        metric_type=MetricType.THROUGHPUT,
        description='Requests processed per second',
        unit='requests/second',
        target_value=10.0,
        warning_threshold=5.0,
        critical_threshold=1.0,
        lower_is_better=False
    ),
    'cpu_utilization': MetricDefinition(
        name='cpu_utilization',
        metric_type=MetricType.RESOURCE,
        description='CPU usage percentage',
        unit='percentage',
        target_value=70.0,
        warning_threshold=80.0,
        critical_threshold=90.0,
        lower_is_better=True
    ),
    'memory_utilization': MetricDefinition(
        name='memory_utilization',
        metric_type=MetricType.RESOURCE,
        description='Memory usage percentage',
        unit='percentage',
        target_value=70.0,
        warning_threshold=80.0,
        critical_threshold=90.0,
        lower_is_better=True
    ),
    'database_response_time': MetricDefinition(
        name='database_response_time',
        metric_type=MetricType.LATENCY,
        description='Database query response time',
        unit='milliseconds',
        target_value=50.0,
        warning_threshold=100.0,
        critical_threshold=200.0,
        lower_is_better=True
    ),
    'cache_hit_rate': MetricDefinition(
        name='cache_hit_rate',
        metric_type=MetricType.THROUGHPUT,
        description='Cache hit rate percentage',
        unit='percentage',
        target_value=90.0,
        warning_threshold=80.0,
        critical_threshold=70.0,
        lower_is_better=False
    )
}

def get_metric_definition(metric_name: str) -> Optional[MetricDefinition]:
    """Get standard metric definition by name."""
    return STANDARD_METRICS.get(metric_name)

def list_standard_metrics() -> List[str]:
    """List all standard metric names."""
    return list(STANDARD_METRICS.keys())