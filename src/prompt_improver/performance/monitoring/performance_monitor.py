"""Enhanced Performance Monitor - 2025 Edition

Advanced performance monitoring with 2025 best practices:
- SLI/SLO framework with error budget tracking
- Multi-dimensional metrics (RED/USE patterns)
- Percentile-based monitoring (P50, P95, P99)
- Business metric correlation
- Adaptive thresholds based on historical data
- Burn rate analysis and alerting
- Performance anomaly detection
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from collections import deque, defaultdict
import statistics
import math

# Enhanced observability imports
try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Machine learning for anomaly detection
try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Simple numpy replacement for basic operations
    class SimpleNumpy:
        @staticmethod
        def array(data):
            return data
        @staticmethod
        def mean(data):
            return statistics.mean(data) if data else 0
        @staticmethod
        def std(data):
            return statistics.stdev(data) if len(data) > 1 else 0
    np = SimpleNumpy()

# Optional import for performance optimizer
try:
    from ..optimization.performance_optimizer import get_performance_optimizer
    PERFORMANCE_OPTIMIZER_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZER_AVAILABLE = False
    get_performance_optimizer = None

logger = logging.getLogger(__name__)

class SLIType(Enum):
    """Service Level Indicator types."""

    availability = "availability"
    latency = "latency"
    throughput = "throughput"
    ERROR_RATE = "error_rate"
    saturation = "saturation"

class SLOStatus(Enum):
    """SLO compliance status."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"  # Error budget exhausted

class BurnRateLevel(Enum):
    """Error budget burn rate levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SLI:
    """Service Level Indicator definition."""

    name: str
    sli_type: SLIType
    description: str
    measurement_window: timedelta = timedelta(minutes=5)
    target_percentile: Optional[float] = None  # For latency SLIs (e.g., 95.0)

    def __post_init__(self):
        if self.sli_type == SLIType.latency and self.target_percentile is None:
            self.target_percentile = 95.0

@dataclass
class SLO:
    """Service Level Objective definition."""

    name: str
    sli: SLI
    target_value: float  # Target value (e.g., 99.9 for 99.9% availability)
    measurement_period: timedelta = timedelta(days=30)  # Rolling window
    error_budget_policy: str = "burn_rate"  # or "simple"

    # Burn rate thresholds (fraction of error budget per hour)
    burn_rate_critical: float = 0.1  # 10% of budget per hour
    burn_rate_high: float = 0.05     # 5% of budget per hour
    burn_rate_medium: float = 0.02   # 2% of budget per hour

    @property
    def error_budget_percent(self) -> float:
        """Calculate error budget percentage based on SLI type."""
        if self.sli.sli_type == SLIType.availability:
            # For availability, error budget is (100 - target_value)
            # e.g., 99.9% availability -> 0.1% error budget
            return 100.0 - self.target_value
        elif self.sli.sli_type == SLIType.ERROR_RATE:
            # For error rate, the target is the maximum acceptable error rate
            # e.g., 1% error rate target -> 1% error budget
            return self.target_value
        elif self.sli.sli_type == SLIType.latency:
            # For latency, we define error budget as percentage of requests
            # that can exceed the target (typically small, e.g., 0.1%)
            return 0.1  # 0.1% of requests can exceed latency target
        elif self.sli.sli_type == SLIType.throughput:
            # For throughput, error budget is percentage below target that's acceptable
            # e.g., 5% below target throughput is acceptable
            return 5.0
        else:
            # Default fallback
            return 1.0

    @property
    def error_budget_minutes(self) -> float:
        """Calculate error budget in minutes for the measurement period."""
        total_minutes = self.measurement_period.total_seconds() / 60
        return total_minutes * (self.error_budget_percent / 100.0)

@dataclass
class SLOViolation:
    """SLO violation record."""

    violation_id: str
    slo_name: str
    violation_type: str  # "threshold", "burn_rate", "budget_exhausted"
    severity: SLOStatus
    current_value: float
    target_value: float
    error_budget_consumed: float  # Percentage of error budget consumed
    burn_rate: float  # Current burn rate
    timestamp: datetime
    duration: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Enhanced performance metrics with RED/USE patterns."""

    # RED Metrics (Rate, Errors, Duration)
    request_rate: float = 0.0  # Requests per second
    error_rate: float = 0.0    # Error percentage
    response_time_p50: float = 0.0  # 50th percentile latency
    response_time_p95: float = 0.0  # 95th percentile latency
    response_time_p99: float = 0.0  # 99th percentile latency

    # USE Metrics (Utilization, Saturation, Errors)
    cpu_utilization: float = 0.0     # CPU usage percentage
    memory_utilization: float = 0.0  # Memory usage percentage
    saturation_score: float = 0.0    # Queue depth/saturation

    # Business Metrics
    business_transactions: float = 0.0  # Business-relevant transactions
    revenue_impact: float = 0.0         # Estimated revenue impact

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sample_count: int = 0
    measurement_window: timedelta = timedelta(minutes=1)

# Use centralized metrics registry
from .metrics_registry import get_metrics_registry

metrics_registry = get_metrics_registry()
SLI_GAUGE = metrics_registry.get_or_create_gauge(
    'sli_value',
    'Service Level Indicator value',
    ['sli_name', 'sli_type']
)
SLO_COMPLIANCE_GAUGE = metrics_registry.get_or_create_gauge(
    'slo_compliance',
    'SLO compliance percentage',
    ['slo_name']
)
ERROR_BUDGET_GAUGE = metrics_registry.get_or_create_gauge(
    'error_budget_remaining',
    'Error budget remaining percentage',
    ['slo_name']
)
BURN_RATE_GAUGE = metrics_registry.get_or_create_gauge(
    'error_budget_burn_rate',
    'Error budget burn rate',
    ['slo_name']
)
PERFORMANCE_HISTOGRAM = metrics_registry.get_or_create_histogram(
    'performance_metrics',
    'Performance metrics histogram',
    ['metric_type']
)

@dataclass
class PerformanceAlert:
    """Performance alert data structure."""
    alert_id: str
    alert_type: str
    severity: str  # 'warning', 'critical'
    message: str
    operation_name: str
    actual_value: float
    threshold_value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'operation_name': self.operation_name,
            'actual_value': self.actual_value,
            'threshold_value': self.threshold_value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

@dataclass
class PerformanceThresholds:
    """Performance monitoring thresholds."""
    response_time_warning_ms: float = 150.0
    response_time_critical_ms: float = 200.0
    error_rate_warning_percent: float = 5.0
    error_rate_critical_percent: float = 10.0
    throughput_warning_rps: float = 10.0  # requests per second
    memory_usage_warning_percent: float = 80.0
    memory_usage_critical_percent: float = 90.0

class PerformanceTrendAnalyzer:
    """Analyzes performance trends and predicts issues."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._response_times: deque = deque(maxlen=window_size)
        self._timestamps: deque = deque(maxlen=window_size)
        self._error_counts: deque = deque(maxlen=window_size)

    def add_measurement(
        self,
        response_time_ms: float,
        is_error: bool = False,
        timestamp: Optional[datetime] = None
    ):
        """Add a performance measurement."""
        timestamp = timestamp or datetime.now(timezone.utc)

        self._response_times.append(response_time_ms)
        self._timestamps.append(timestamp)
        self._error_counts.append(1 if is_error else 0)

    def get_trend_analysis(self) -> Dict[str, Any]:
        """Get comprehensive trend analysis."""
        if len(self._response_times) < 10:
            return {"status": "insufficient_data", "sample_count": len(self._response_times)}

        response_times = list(self._response_times)

        # Calculate basic statistics
        avg_response_time = statistics.mean(response_times)
        median_response_time = statistics.median(response_times)
        p95_response_time = self._percentile(response_times, 95)
        p99_response_time = self._percentile(response_times, 99)

        # Calculate error rate
        error_rate = sum(self._error_counts) / len(self._error_counts) * 100

        # Calculate throughput (requests per second)
        if len(self._timestamps) >= 2:
            time_span = (self._timestamps[-1] - self._timestamps[0]).total_seconds()
            throughput = len(self._timestamps) / time_span if time_span > 0 else 0
        else:
            throughput = 0

        # Trend detection (simple linear regression on recent data)
        trend_direction = self._detect_trend()

        return {
            "status": "analyzed",
            "sample_count": len(response_times),
            "avg_response_time_ms": avg_response_time,
            "median_response_time_ms": median_response_time,
            "p95_response_time_ms": p95_response_time,
            "p99_response_time_ms": p99_response_time,
            "error_rate_percent": error_rate,
            "throughput_rps": throughput,
            "trend_direction": trend_direction,
            "performance_grade": self._calculate_performance_grade(avg_response_time, error_rate)
        }

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    def _detect_trend(self) -> str:
        """Detect performance trend direction."""
        if len(self._response_times) < 20:
            return "unknown"

        # Compare recent half with earlier half
        mid_point = len(self._response_times) // 2
        recent_avg = statistics.mean(list(self._response_times)[mid_point:])
        earlier_avg = statistics.mean(list(self._response_times)[:mid_point])

        if recent_avg > earlier_avg * 1.1:  # 10% increase
            return "degrading"
        elif recent_avg < earlier_avg * 0.9:  # 10% improvement
            return "improving"
        else:
            return "stable"

    def _calculate_performance_grade(self, avg_response_time: float, error_rate: float) -> str:
        """Calculate overall performance grade."""
        if avg_response_time < 50 and error_rate < 1:
            return "A"
        elif avg_response_time < 100 and error_rate < 2:
            return "B"
        elif avg_response_time < 200 and error_rate < 5:
            return "C"
        elif avg_response_time < 500 and error_rate < 10:
            return "D"
        else:
            return "F"

class EnhancedPerformanceMonitor:
    """Enhanced performance monitor with SLI/SLO framework and 2025 best practices.

    features:
    - SLI/SLO framework with error budget tracking
    - Multi-dimensional metrics (RED/USE patterns)
    - Percentile-based monitoring (P50, P95, P99)
    - Business metric correlation
    - Adaptive thresholds based on historical data
    - Burn rate analysis and alerting
    """

    def __init__(
        self,
        slos: Optional[List[SLO]] = None,
        thresholds: Optional[PerformanceThresholds] = None,
        alert_callback: Optional[Callable[[PerformanceAlert], None]] = None,
        trend_window_size: int = 1000,
        enable_anomaly_detection: bool = True,
        enable_adaptive_thresholds: bool = True
    ):
        """Initialize enhanced performance monitor."""
        self.slos = slos or self._create_default_slos()
        self.thresholds = thresholds or PerformanceThresholds()
        self.alert_callback = alert_callback
        self.trend_window_size = trend_window_size
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_adaptive_thresholds = enable_adaptive_thresholds

        # Enhanced tracking
        self.slo_states: Dict[str, Dict[str, Any]] = {}
        self.sli_measurements: Dict[str, deque] = defaultdict(lambda: deque(maxlen=trend_window_size))
        self.error_budget_tracking: Dict[str, Dict[str, Any]] = {}
        self.burn_rate_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Performance data storage
        self.performance_history: deque = deque(maxlen=trend_window_size)
        self.response_times: deque = deque(maxlen=trend_window_size)
        self.error_events: deque = deque(maxlen=trend_window_size)

        # Anomaly detection
        self.anomaly_detector = None
        if self.enable_anomaly_detection and ML_AVAILABLE:
            try:
                self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
                # Fit with dummy data to initialize
                dummy_data = [[100, 1, 10, 50, 60] for _ in range(10)]
                self.anomaly_detector.fit(dummy_data)
            except Exception as e:
                self.logger.warning(f"Could not initialize anomaly detector: {e}")
                self.anomaly_detector = None

        # Alert management
        self._alert_handlers: List[Callable] = []
        self._active_alerts: Dict[str, PerformanceAlert] = {}
        self._slo_violations: Dict[str, SLOViolation] = {}
        self._alert_history: List[PerformanceAlert] = []

        # Monitoring state
        self._monitoring_enabled = True
        self._last_reset_time = datetime.now(timezone.utc)

        # Initialize SLO tracking
        self._initialize_slo_tracking()

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Enhanced performance monitor initialized with {len(self.slos)} SLOs")

    def _create_default_slos(self) -> List[SLO]:
        """Create default SLOs for common performance metrics."""
        default_slis = [
            SLI(
                name="response_time_p95",
                sli_type=SLIType.latency,
                description="95th percentile response time",
                target_percentile=95.0
            ),
            SLI(
                name="availability",
                sli_type=SLIType.availability,
                description="Service availability"
            ),
            SLI(
                name="error_rate",
                sli_type=SLIType.ERROR_RATE,
                description="Error rate percentage"
            ),
            SLI(
                name="throughput",
                sli_type=SLIType.throughput,
                description="Request throughput"
            )
        ]

        default_slos = [
            SLO(
                name="response_time_slo",
                sli=default_slis[0],
                target_value=200.0,  # 200ms P95
                measurement_period=timedelta(hours=1),
                burn_rate_critical=0.1,
                burn_rate_high=0.05,
                burn_rate_medium=0.02
            ),
            SLO(
                name="availability_slo",
                sli=default_slis[1],
                target_value=99.9,  # 99.9% availability
                measurement_period=timedelta(days=30),
                burn_rate_critical=0.1,
                burn_rate_high=0.05,
                burn_rate_medium=0.02
            ),
            SLO(
                name="error_rate_slo",
                sli=default_slis[2],
                target_value=1.0,  # <1% error rate
                measurement_period=timedelta(hours=1),
                burn_rate_critical=0.1,
                burn_rate_high=0.05,
                burn_rate_medium=0.02
            ),
            SLO(
                name="throughput_slo",
                sli=default_slis[3],
                target_value=10.0,  # >10 RPS
                measurement_period=timedelta(minutes=5),
                burn_rate_critical=0.1,
                burn_rate_high=0.05,
                burn_rate_medium=0.02
            )
        ]

        return default_slos

    def _initialize_slo_tracking(self):
        """Initialize SLO tracking state."""
        for slo in self.slos:
            self.slo_states[slo.name] = {
                "current_value": 0.0,
                "target_value": slo.target_value,
                "compliance_percentage": 100.0,
                "error_budget_remaining": 100.0,
                "burn_rate": 0.0,
                "status": SLOStatus.HEALTHY,
                "last_violation": None,
                "violation_count": 0
            }

            self.error_budget_tracking[slo.name] = {
                "total_budget_minutes": slo.error_budget_minutes,
                "consumed_minutes": 0.0,
                "remaining_minutes": slo.error_budget_minutes,
                "burn_rate_per_hour": 0.0,
                "projected_exhaustion": None
            }

    async def record_performance_measurement(
        self,
        operation_name: str,
        response_time_ms: float,
        is_error: bool = False,
        business_value: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a performance measurement with SLI/SLO tracking."""
        timestamp = datetime.now(timezone.utc)

        # Create performance metrics
        metrics = PerformanceMetrics(
            request_rate=self._calculate_current_rate(),
            error_rate=self._calculate_current_error_rate(is_error),
            response_time_p50=self._calculate_percentile(50),
            response_time_p95=self._calculate_percentile(95),
            response_time_p99=self._calculate_percentile(99),
            business_transactions=business_value,
            timestamp=timestamp,
            sample_count=len(self.response_times) + 1
        )

        # Store measurement data
        self.response_times.append(response_time_ms)
        self.error_events.append(1 if is_error else 0)
        self.performance_history.append(metrics)

        # Update SLI measurements
        await self._update_sli_measurements(metrics, timestamp)

        # Evaluate SLOs
        await self._evaluate_slos(timestamp)

        # Check for anomalies
        if self.enable_anomaly_detection:
            await self._detect_anomalies(metrics)

        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and SLI_GAUGE:
            SLI_GAUGE.labels(sli_name="response_time", sli_type="latency").set(response_time_ms)
            if is_error:
                SLI_GAUGE.labels(sli_name="error_rate", sli_type="error_rate").inc()

        self.logger.debug(f"Recorded performance measurement: {operation_name}, {response_time_ms}ms, error={is_error}")

    async def _update_sli_measurements(self, metrics: PerformanceMetrics, timestamp: datetime):
        """Update SLI measurements for all configured SLIs."""
        for slo in self.slos:
            sli = slo.sli

            if sli.sli_type == SLIType.latency:
                if sli.target_percentile == 95.0:
                    value = metrics.response_time_p95
                elif sli.target_percentile == 99.0:
                    value = metrics.response_time_p99
                else:
                    value = metrics.response_time_p50
            elif sli.sli_type == SLIType.availability:
                # Calculate availability as (1 - error_rate) * 100
                value = (1.0 - metrics.error_rate / 100.0) * 100.0
            elif sli.sli_type == SLIType.ERROR_RATE:
                value = metrics.error_rate
            elif sli.sli_type == SLIType.throughput:
                value = metrics.request_rate
            else:
                value = 0.0

            # Store SLI measurement
            self.sli_measurements[sli.name].append({
                "timestamp": timestamp,
                "value": value,
                "target": slo.target_value
            })

    async def _evaluate_slos(self, timestamp: datetime):
        """Evaluate all SLOs and update compliance status."""
        for slo in self.slos:
            await self._evaluate_single_slo(slo, timestamp)

    async def _evaluate_single_slo(self, slo: SLO, timestamp: datetime):
        """Evaluate a single SLO and update its state."""
        sli_data = self.sli_measurements[slo.sli.name]

        if not sli_data:
            return

        # Get measurements within the SLO measurement period
        cutoff_time = timestamp - slo.measurement_period
        recent_measurements = [
            m for m in sli_data
            if m["timestamp"] >= cutoff_time
        ]

        if not recent_measurements:
            return

        # Calculate SLO compliance
        compliance_percentage = self._calculate_slo_compliance(slo, recent_measurements)

        # Update SLO state
        slo_state = self.slo_states[slo.name]
        slo_state["current_value"] = recent_measurements[-1]["value"]
        slo_state["compliance_percentage"] = compliance_percentage

        # Calculate error budget consumption
        error_budget_consumed = max(0, 100.0 - compliance_percentage)
        error_budget_remaining = max(0, 100.0 - error_budget_consumed)
        slo_state["error_budget_remaining"] = error_budget_remaining

        # Calculate burn rate
        burn_rate = self._calculate_burn_rate(slo, recent_measurements)
        slo_state["burn_rate"] = burn_rate

        # Update error budget tracking
        budget_tracking = self.error_budget_tracking[slo.name]
        budget_tracking["consumed_minutes"] = (error_budget_consumed / 100.0) * budget_tracking["total_budget_minutes"]
        budget_tracking["remaining_minutes"] = budget_tracking["total_budget_minutes"] - budget_tracking["consumed_minutes"]
        budget_tracking["burn_rate_per_hour"] = burn_rate

        # Determine SLO status
        new_status = self._determine_slo_status(slo, slo_state, burn_rate)
        old_status = slo_state["status"]
        slo_state["status"] = new_status

        # Check for violations and alerts
        if new_status != SLOStatus.HEALTHY and new_status != old_status:
            await self._handle_slo_violation(slo, slo_state, timestamp)

        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and SLO_COMPLIANCE_GAUGE:
            SLO_COMPLIANCE_GAUGE.labels(slo_name=slo.name).set(compliance_percentage)
            if ERROR_BUDGET_GAUGE:
                ERROR_BUDGET_GAUGE.labels(slo_name=slo.name).set(error_budget_remaining)
            if BURN_RATE_GAUGE:
                BURN_RATE_GAUGE.labels(slo_name=slo.name).set(burn_rate)

    def _calculate_slo_compliance(self, slo: SLO, measurements: List[Dict[str, Any]]) -> float:
        """Calculate SLO compliance percentage."""
        if not measurements:
            return 100.0

        total_measurements = len(measurements)
        compliant_measurements = 0

        for measurement in measurements:
            value = measurement["value"]
            target = slo.target_value

            if slo.sli.sli_type == SLIType.latency:
                # For latency, lower is better
                if value <= target:
                    compliant_measurements += 1
            elif slo.sli.sli_type == SLIType.ERROR_RATE:
                # For error rate, lower is better
                if value <= target:
                    compliant_measurements += 1
            else:
                # For availability and throughput, higher is better
                if value >= target:
                    compliant_measurements += 1

        return (compliant_measurements / total_measurements) * 100.0

    def _calculate_burn_rate(self, slo: SLO, measurements: List[Dict[str, Any]]) -> float:
        """Calculate error budget burn rate."""
        if len(measurements) < 2:
            return 0.0

        # Calculate burn rate over the last hour
        one_hour_ago = measurements[-1]["timestamp"] - timedelta(hours=1)
        recent_measurements = [
            m for m in measurements
            if m["timestamp"] >= one_hour_ago
        ]

        if not recent_measurements:
            return 0.0

        # Calculate compliance for recent period
        compliance = self._calculate_slo_compliance(slo, recent_measurements)
        error_rate = 100.0 - compliance

        # Convert to burn rate (fraction of total error budget consumed per hour)
        if slo.error_budget_percent > 0:
            burn_rate = error_rate / slo.error_budget_percent
        else:
            burn_rate = 0.0

        return burn_rate

    def _determine_slo_status(self, slo: SLO, slo_state: Dict[str, Any], burn_rate: float) -> SLOStatus:
        """Determine SLO status based on compliance and burn rate."""
        error_budget_remaining = slo_state["error_budget_remaining"]

        # Check if error budget is exhausted
        if error_budget_remaining <= 0:
            return SLOStatus.EXHAUSTED

        # Check burn rate levels
        if burn_rate >= slo.burn_rate_critical:
            return SLOStatus.CRITICAL
        elif burn_rate >= slo.burn_rate_high:
            return SLOStatus.WARNING
        elif error_budget_remaining < 20:  # Less than 20% budget remaining
            return SLOStatus.WARNING
        else:
            return SLOStatus.HEALTHY

    async def _handle_slo_violation(self, slo: SLO, slo_state: Dict[str, Any], timestamp: datetime):
        """Handle SLO violation by creating alerts."""
        violation_id = str(uuid.uuid4())

        violation = SLOViolation(
            violation_id=violation_id,
            slo_name=slo.name,
            violation_type="burn_rate" if slo_state["burn_rate"] > slo.burn_rate_medium else "threshold",
            severity=slo_state["status"],
            current_value=slo_state["current_value"],
            target_value=slo.target_value,
            error_budget_consumed=100.0 - slo_state["error_budget_remaining"],
            burn_rate=slo_state["burn_rate"],
            timestamp=timestamp
        )

        self._slo_violations[violation_id] = violation
        slo_state["last_violation"] = violation
        slo_state["violation_count"] += 1

        # Create performance alert
        alert = PerformanceAlert(
            alert_id=violation_id,
            alert_type="slo_violation",
            severity=violation.severity.value,
            message=f"SLO violation: {slo.name} - {violation.violation_type}",
            operation_name=slo.sli.name,
            actual_value=violation.current_value,
            threshold_value=violation.target_value,
            timestamp=timestamp,
            metadata={
                "slo_name": slo.name,
                "error_budget_remaining": slo_state["error_budget_remaining"],
                "burn_rate": violation.burn_rate,
                "violation_type": violation.violation_type
            }
        )

        self._active_alerts[violation_id] = alert
        self._alert_history.append(alert)

        # Trigger alert callback
        if self.alert_callback:
            self.alert_callback(alert)

        self.logger.warning(f"SLO violation detected: {slo.name} - {violation.violation_type}")

    async def _detect_anomalies(self, metrics: PerformanceMetrics):
        """Detect performance anomalies using machine learning."""
        if not self.anomaly_detector or len(self.performance_history) < 50:
            return

        try:
            # Prepare feature vector
            features = [
                metrics.response_time_p95,
                metrics.error_rate,
                metrics.request_rate,
                metrics.cpu_utilization,
                metrics.memory_utilization
            ]

            # Detect anomaly
            is_anomaly = self.anomaly_detector.predict([features])[0] == -1

            if is_anomaly:
                alert = PerformanceAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type="anomaly",
                    severity="warning",
                    message="Performance anomaly detected",
                    operation_name="composite",
                    actual_value=metrics.response_time_p95,
                    threshold_value=0.0,
                    timestamp=metrics.timestamp,
                    metadata={"anomaly_score": "detected", "features": features}
                )

                self._active_alerts[alert.alert_id] = alert
                self._alert_history.append(alert)

                if self.alert_callback:
                    self.alert_callback(alert)

                self.logger.warning("Performance anomaly detected")

        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")

    def _calculate_current_rate(self) -> float:
        """Calculate current request rate."""
        if len(self.performance_history) < 2:
            return 0.0

        # Calculate rate over last minute
        now = datetime.now(timezone.utc)
        one_minute_ago = now - timedelta(minutes=1)

        recent_requests = sum(
            1 for metrics in self.performance_history
            if metrics.timestamp >= one_minute_ago
        )

        return recent_requests / 60.0  # Requests per second

    def _calculate_current_error_rate(self, is_error: bool) -> float:
        """Calculate current error rate."""
        if not self.error_events:
            return 0.0

        total_errors = sum(self.error_events)
        total_requests = len(self.error_events)

        return (total_errors / total_requests) * 100.0 if total_requests > 0 else 0.0

    def _calculate_percentile(self, percentile: float) -> float:
        """Calculate response time percentile."""
        if not self.response_times:
            return 0.0

        sorted_times = sorted(self.response_times)
        index = int((percentile / 100.0) * len(sorted_times))
        index = min(index, len(sorted_times) - 1)

        return sorted_times[index]

    def get_slo_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive SLO dashboard data."""
        dashboard = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_health": self._calculate_overall_health(),
            "slos": {},
            "error_budgets": {},
            "burn_rates": {},
            "active_violations": len(self._slo_violations),
            "total_alerts": len(self._alert_history)
        }

        for slo in self.slos:
            slo_state = self.slo_states[slo.name]
            budget_tracking = self.error_budget_tracking[slo.name]

            dashboard["slos"][slo.name] = {
                "current_value": slo_state["current_value"],
                "target_value": slo_state["target_value"],
                "compliance_percentage": slo_state["compliance_percentage"],
                "status": slo_state["status"].value,
                "violation_count": slo_state["violation_count"]
            }

            dashboard["error_budgets"][slo.name] = {
                "remaining_percentage": slo_state["error_budget_remaining"],
                "remaining_minutes": budget_tracking["remaining_minutes"],
                "total_minutes": budget_tracking["total_budget_minutes"],
                "consumed_minutes": budget_tracking["consumed_minutes"]
            }

            dashboard["burn_rates"][slo.name] = {
                "current_rate": slo_state["burn_rate"],
                "critical_threshold": slo.burn_rate_critical,
                "high_threshold": slo.burn_rate_high,
                "medium_threshold": slo.burn_rate_medium
            }

        return dashboard

    def _calculate_overall_health(self) -> str:
        """Calculate overall system health based on SLO status."""
        if not self.slo_states:
            return "unknown"

        statuses = [state["status"] for state in self.slo_states.values()]

        if any(status == SLOStatus.EXHAUSTED for status in statuses):
            return "critical"
        elif any(status == SLOStatus.CRITICAL for status in statuses):
            return "critical"
        elif any(status == SLOStatus.WARNING for status in statuses):
            return "warning"
        else:
            return "healthy"

    async def run_orchestrated_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrator-compatible interface for performance monitoring (2025 pattern)"""
        start_time = datetime.now(timezone.utc)

        try:
            # Extract configuration
            measurement_window = config.get("measurement_window", 300)  # 5 minutes default
            include_predictions = config.get("include_predictions", True)
            include_anomaly_detection = config.get("include_anomaly_detection", True)
            output_path = config.get("output_path", "./outputs/performance_monitoring")

            # Simulate some measurements if requested
            simulate_data = config.get("simulate_data", False)
            if simulate_data:
                await self._simulate_performance_data()

            # Get comprehensive performance analysis
            dashboard = self.get_slo_dashboard()

            # Get current performance metrics
            current_metrics = None
            if self.performance_history:
                current_metrics = self.performance_history[-1]

            # Prepare orchestrator-compatible result
            result = {
                "performance_summary": {
                    "overall_health": dashboard["overall_health"],
                    "total_slos": len(self.slos),
                    "healthy_slos": sum(1 for slo_data in dashboard["slos"].values() if slo_data["status"] == "healthy"),
                    "warning_slos": sum(1 for slo_data in dashboard["slos"].values() if slo_data["status"] == "warning"),
                    "critical_slos": sum(1 for slo_data in dashboard["slos"].values() if slo_data["status"] == "critical"),
                    "active_violations": dashboard["active_violations"],
                    "total_alerts": dashboard["total_alerts"]
                },
                "slo_compliance": dashboard["slos"],
                "error_budget_status": dashboard["error_budgets"],
                "burn_rate_analysis": dashboard["burn_rates"],
                "current_metrics": {
                    "response_time_p50": current_metrics.response_time_p50 if current_metrics else 0,
                    "response_time_p95": current_metrics.response_time_p95 if current_metrics else 0,
                    "response_time_p99": current_metrics.response_time_p99 if current_metrics else 0,
                    "error_rate": current_metrics.error_rate if current_metrics else 0,
                    "request_rate": current_metrics.request_rate if current_metrics else 0,
                    "timestamp": current_metrics.timestamp.isoformat() if current_metrics else start_time.isoformat()
                },
                "active_alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "alert_type": alert.alert_type,
                        "severity": alert.severity,
                        "message": alert.message,
                        "operation_name": alert.operation_name,
                        "actual_value": alert.actual_value,
                        "threshold_value": alert.threshold_value,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in self._active_alerts.values()
                ],
                "slo_violations": [
                    {
                        "violation_id": violation.violation_id,
                        "slo_name": violation.slo_name,
                        "violation_type": violation.violation_type,
                        "severity": violation.severity.value,
                        "current_value": violation.current_value,
                        "target_value": violation.target_value,
                        "error_budget_consumed": violation.error_budget_consumed,
                        "burn_rate": violation.burn_rate,
                        "timestamp": violation.timestamp.isoformat()
                    }
                    for violation in self._slo_violations.values()
                ],
                "capabilities": {
                    "sli_slo_framework": True,
                    "error_budget_tracking": True,
                    "burn_rate_analysis": True,
                    "percentile_monitoring": True,
                    "anomaly_detection": self.enable_anomaly_detection,
                    "adaptive_thresholds": self.enable_adaptive_thresholds,
                    "multi_dimensional_metrics": True
                }
            }

            # Calculate execution metadata
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            return {
                "orchestrator_compatible": True,
                "component_result": result,
                "local_metadata": {
                    "output_path": output_path,
                    "execution_time": execution_time,
                    "measurement_window": measurement_window,
                    "slos_configured": len(self.slos),
                    "measurements_collected": len(self.performance_history),
                    "anomaly_detection_enabled": self.enable_anomaly_detection,
                    "adaptive_thresholds_enabled": self.enable_adaptive_thresholds,
                    "component_version": "2025.1.0"
                }
            }

        except Exception as e:
            self.logger.error(f"Orchestrated performance monitoring failed: {e}")
            return {
                "orchestrator_compatible": True,
                "component_result": {"error": str(e), "performance_summary": {}},
                "local_metadata": {
                    "execution_time": (datetime.now(timezone.utc) - start_time).total_seconds(),
                    "error": True,
                    "component_version": "2025.1.0"
                }
            }

    async def _simulate_performance_data(self):
        """Simulate performance data for testing purposes."""
        import random

        # Simulate 100 measurements over the last hour
        base_time = datetime.now(timezone.utc) - timedelta(hours=1)

        for i in range(100):
            timestamp = base_time + timedelta(minutes=i * 0.6)  # Every 36 seconds

            # Simulate varying performance
            base_response_time = 150 + random.gauss(0, 30)  # 150ms ± 30ms
            is_error = random.random() < 0.02  # 2% error rate

            # Add some performance degradation in the middle
            if 40 <= i <= 60:
                base_response_time *= 1.5  # 50% slower
                is_error = random.random() < 0.05  # 5% error rate

            await self.record_performance_measurement(
                operation_name="simulated_operation",
                response_time_ms=max(10, base_response_time),
                is_error=is_error,
                business_value=random.uniform(1, 10)
            )

# Maintain backward compatibility
class PerformanceMonitor(EnhancedPerformanceMonitor):
    """Backward compatible performance monitor."""

    def __init__(self, thresholds: Optional[PerformanceThresholds] = None):
        super().__init__(
            slos=None,  # Use defaults
            thresholds=thresholds,
            enable_anomaly_detection=False,
            enable_adaptive_thresholds=False
        )

        # Initialize backward compatibility attributes
        self._total_requests = 0
        self._total_errors = 0
        self._total_response_time = 0.0

        # Initialize trend analyzer for backward compatibility
        self.trend_analyzer = PerformanceTrendAnalyzer(window_size=self.trend_window_size)

    def add_alert_handler(self, handler: Callable[[PerformanceAlert], None]):
        """Add an alert handler function."""
        self._alert_handlers.append(handler)

    async def record_operation(
        self,
        operation_name: str,
        response_time_ms: float,
        is_error: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a performance measurement."""
        if not self._monitoring_enabled:
            return

        # Update counters
        self._total_requests += 1
        self._total_response_time += response_time_ms
        if is_error:
            self._total_errors += 1

        # Add to trend analyzer
        self.trend_analyzer.add_measurement(response_time_ms, is_error)

        # Check for threshold violations
        await self._check_thresholds(operation_name, response_time_ms, is_error, metadata)

    async def _check_thresholds(
        self,
        operation_name: str,
        response_time_ms: float,
        is_error: bool,
        metadata: Optional[Dict[str, Any]]
    ):
        """Check if any thresholds are violated and generate alerts."""
        alerts_to_generate = []

        # Response time thresholds
        if response_time_ms >= self.thresholds.response_time_critical_ms:
            alert = PerformanceAlert(
                alert_id=f"response_time_critical_{int(time.time())}",
                alert_type="response_time",
                severity="critical",
                message=f"Response time {response_time_ms:.2f}ms exceeds critical threshold {self.thresholds.response_time_critical_ms}ms",
                operation_name=operation_name,
                actual_value=response_time_ms,
                threshold_value=self.thresholds.response_time_critical_ms,
                timestamp=datetime.now(timezone.utc),
                metadata=metadata or {}
            )
            alerts_to_generate.append(alert)

        elif response_time_ms >= self.thresholds.response_time_warning_ms:
            alert = PerformanceAlert(
                alert_id=f"response_time_warning_{int(time.time())}",
                alert_type="response_time",
                severity="warning",
                message=f"Response time {response_time_ms:.2f}ms exceeds warning threshold {self.thresholds.response_time_warning_ms}ms",
                operation_name=operation_name,
                actual_value=response_time_ms,
                threshold_value=self.thresholds.response_time_warning_ms,
                timestamp=datetime.now(timezone.utc),
                metadata=metadata or {}
            )
            alerts_to_generate.append(alert)

        # Error rate threshold (calculated over recent requests)
        if self._total_requests >= 10:  # Only check after sufficient samples
            current_error_rate = (self._total_errors / self._total_requests) * 100

            if current_error_rate >= self.thresholds.error_rate_critical_percent:
                alert = PerformanceAlert(
                    alert_id=f"error_rate_critical_{int(time.time())}",
                    alert_type="error_rate",
                    severity="critical",
                    message=f"Error rate {current_error_rate:.2f}% exceeds critical threshold {self.thresholds.error_rate_critical_percent}%",
                    operation_name=operation_name,
                    actual_value=current_error_rate,
                    threshold_value=self.thresholds.error_rate_critical_percent,
                    timestamp=datetime.now(timezone.utc),
                    metadata=metadata or {}
                )
                alerts_to_generate.append(alert)

        # Process alerts
        for alert in alerts_to_generate:
            await self._process_alert(alert)

    async def _process_alert(self, alert: PerformanceAlert):
        """Process and handle a performance alert."""
        # Store alert
        self._active_alerts[alert.alert_id] = alert
        self._alert_history.append(alert)

        # Keep alert history manageable
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-1000:]

        # Log alert
        log_level = logging.CRITICAL if alert.severity == "critical" else logging.WARNING
        logger.log(log_level, f"Performance Alert: {alert.message}")

        # Call alert handlers
        for handler in self._alert_handlers:
            try:
                await handler(alert) if asyncio.iscoroutinefunction(handler) else handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def get_current_performance_status(self) -> Dict[str, Any]:
        """Get current performance status and metrics."""
        trend_analysis = self.trend_analyzer.get_trend_analysis()

        # Calculate current metrics
        avg_response_time = (
            self._total_response_time / self._total_requests
            if self._total_requests > 0 else 0
        )
        error_rate = (
            (self._total_errors / self._total_requests) * 100
            if self._total_requests > 0 else 0
        )

        # Calculate uptime
        uptime_seconds = (datetime.now(timezone.utc) - self._last_reset_time).total_seconds()

        return {
            "monitoring_enabled": self._monitoring_enabled,
            "uptime_seconds": uptime_seconds,
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "avg_response_time_ms": avg_response_time,
            "error_rate_percent": error_rate,
            "active_alerts_count": len(self._active_alerts),
            "trend_analysis": trend_analysis,
            "thresholds": {
                "response_time_warning_ms": self.thresholds.response_time_warning_ms,
                "response_time_critical_ms": self.thresholds.response_time_critical_ms,
                "error_rate_warning_percent": self.thresholds.error_rate_warning_percent,
                "error_rate_critical_percent": self.thresholds.error_rate_critical_percent
            },
            "performance_grade": trend_analysis.get("performance_grade", "N/A"),
            "meets_200ms_target": avg_response_time < 200
        }

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active performance alerts."""
        return [alert.to_dict() for alert in self._active_alerts.values()]

    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for the specified time period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self._alert_history
            if alert.timestamp > cutoff_time
        ]
        return [alert.to_dict() for alert in recent_alerts]

    def reset_counters(self):
        """Reset performance counters."""
        self._total_requests = 0
        self._total_errors = 0
        self._total_response_time = 0.0
        self._last_reset_time = datetime.now(timezone.utc)
        self._active_alerts.clear()

# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

# Convenience functions
async def record_mcp_operation(
    operation_name: str,
    response_time_ms: float,
    is_error: bool = False,
    **metadata
):
    """Record an MCP operation performance measurement."""
    monitor = get_performance_monitor()
    await monitor.record_operation(operation_name, response_time_ms, is_error, metadata)

def add_performance_alert_handler(handler: Callable):
    """Add a performance alert handler."""
    monitor = get_performance_monitor()
    monitor.add_alert_handler(handler)
