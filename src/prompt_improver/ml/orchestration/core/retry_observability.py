"""
Enhanced Observability and Metrics for Unified Retry Manager.

Provides comprehensive monitoring, metrics collection, and alerting
for retry operations following 2025 observability best practices.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Use centralized metrics registry
from ....performance.monitoring.metrics_registry import (
    get_metrics_registry,
    StandardMetrics,
    PROMETHEUS_AVAILABLE
)

try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.metrics import Observation
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class RetryMetrics:
    """Comprehensive retry metrics."""
    
    operation_name: str
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    circuit_breaker_trips: int = 0
    total_delay_ms: float = 0.0
    avg_delay_ms: float = 0.0
    success_rate: float = 0.0
    failure_rate: float = 0.0
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    
    # Performance metrics
    p50_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    # Error breakdown
    error_types: Dict[str, int] = field(default_factory=dict)
    
    def update_success(self, response_time_ms: float):
        """Update metrics for successful operation."""
        self.total_attempts += 1
        self.successful_attempts += 1
        self.last_success_time = datetime.now(timezone.utc)
        self.success_rate = self.successful_attempts / self.total_attempts
        self.failure_rate = 1.0 - self.success_rate
    
    def update_failure(self, error_type: str, delay_ms: float = 0.0):
        """Update metrics for failed operation."""
        self.total_attempts += 1
        self.failed_attempts += 1
        self.total_delay_ms += delay_ms
        self.last_failure_time = datetime.now(timezone.utc)
        self.success_rate = self.successful_attempts / self.total_attempts
        self.failure_rate = 1.0 - self.success_rate
        
        # Update error type counts
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
        
        # Update average delay
        if self.failed_attempts > 0:
            self.avg_delay_ms = self.total_delay_ms / self.failed_attempts
    
    def update_circuit_breaker_trip(self):
        """Update metrics for circuit breaker trip."""
        self.circuit_breaker_trips += 1

@dataclass
class AlertRule:
    """Alert rule configuration."""
    
    name: str
    condition: str  # e.g., "failure_rate > 0.5"
    severity: AlertSeverity
    threshold: float
    window_minutes: int = 5
    enabled: bool = True
    
    def evaluate(self, metrics: RetryMetrics) -> bool:
        """Evaluate if alert should be triggered."""
        if not self.enabled:
            return False
        
        # Simple condition evaluation - in production, use a proper expression evaluator
        if "failure_rate" in self.condition:
            return metrics.failure_rate > self.threshold
        elif "circuit_breaker_trips" in self.condition:
            return metrics.circuit_breaker_trips > self.threshold
        elif "avg_delay_ms" in self.condition:
            return metrics.avg_delay_ms > self.threshold
        
        return False

class RetryObservabilityManager:
    """
    Comprehensive observability manager for retry operations.
    
    features:
    - Prometheus metrics collection
    - OpenTelemetry tracing
    - Custom alerting rules
    - Performance analytics
    - Error pattern analysis
    """
    
    def __init__(self, enable_prometheus: bool = True, enable_otel: bool = True):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.enable_otel = enable_otel and OPENTELEMETRY_AVAILABLE
        
        # Metrics storage
        self.operation_metrics: Dict[str, RetryMetrics] = {}
        self.alert_rules: List[AlertRule] = []
        
        # Initialize observability components
        self._setup_prometheus_metrics()
        self._setup_opentelemetry()
        self._setup_default_alerts()
        
        logger.info("RetryObservabilityManager initialized with 2025 best practices")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics using centralized registry."""
        if not self.enable_prometheus:
            return

        self.metrics_registry = get_metrics_registry()

        # Core retry metrics
        self.retry_attempts_total = self.metrics_registry.get_or_create_counter(
            StandardMetrics.RETRY_ATTEMPTS_TOTAL,
            'Total number of retry attempts',
            ['operation', 'strategy', 'attempt']
        )

        self.retry_success_total = self.metrics_registry.get_or_create_counter(
            StandardMetrics.RETRY_SUCCESS_TOTAL,
            'Total number of successful retries',
            ['operation', 'strategy']
        )

        self.retry_failure_total = self.metrics_registry.get_or_create_counter(
            StandardMetrics.RETRY_FAILURE_TOTAL,
            'Total number of failed retries',
            ['operation', 'strategy', 'error_type']
        )

        # Performance metrics
        self.retry_duration_histogram = self.metrics_registry.get_or_create_histogram(
            StandardMetrics.RETRY_DURATION,
            'Retry operation duration distribution',
            ['operation', 'strategy'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
        )

        self.retry_delay_histogram = self.metrics_registry.get_or_create_histogram(
            StandardMetrics.RETRY_DELAY,
            'Retry delay duration distribution',
            ['operation', 'strategy'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
        )

        # Circuit breaker metrics
        self.circuit_breaker_state_gauge = self.metrics_registry.get_or_create_gauge(
            StandardMetrics.CIRCUIT_BREAKER_STATE,
            'Circuit breaker state (0=closed, 1=half_open, 2=open)',
            ['operation']
        )

        self.circuit_breaker_trips_total = self.metrics_registry.get_or_create_counter(
            StandardMetrics.CIRCUIT_BREAKER_TRIPS_TOTAL,
            'Total number of circuit breaker trips',
            ['operation']
        )

        # Success rate metrics
        self.retry_success_rate_gauge = self.metrics_registry.get_or_create_gauge(
            'retry_success_rate',
            'Current success rate for retry operations',
            ['operation']
        )

        # Error pattern metrics
        self.retry_error_patterns = self.metrics_registry.get_or_create_counter(
            'retry_error_patterns_total',
            'Error patterns in retry operations',
            ['operation', 'error_pattern', 'error_category']
        )

        # SLI/SLO metrics
        self.retry_sli_availability = self.metrics_registry.get_or_create_gauge(
            'retry_sli_availability',
            'Retry operation availability SLI',
            ['operation']
        )

        self.retry_sli_latency = self.metrics_registry.get_or_create_histogram(
            'retry_sli_latency_seconds',
            'Retry operation latency SLI',
            ['operation'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
        )
    
    def _setup_opentelemetry(self):
        """Setup OpenTelemetry tracing and metrics."""
        if not self.enable_otel:
            return
        
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        
        # Create OTEL metrics
        self.otel_retry_counter = self.meter.create_counter(
            "retry_operations_total",
            description="Total retry operations",
            unit="1"
        )
        
        self.otel_retry_duration = self.meter.create_histogram(
            "retry_operation_duration",
            description="Retry operation duration",
            unit="s"
        )
    
    def _setup_default_alerts(self):
        """Setup default alerting rules."""
        self.alert_rules = [
            AlertRule(
                name="high_failure_rate",
                condition="failure_rate > 0.5",
                severity=AlertSeverity.WARNING,
                threshold=0.5,
                window_minutes=5
            ),
            AlertRule(
                name="critical_failure_rate",
                condition="failure_rate > 0.8",
                severity=AlertSeverity.CRITICAL,
                threshold=0.8,
                window_minutes=2
            ),
            AlertRule(
                name="circuit_breaker_trips",
                condition="circuit_breaker_trips > 5",
                severity=AlertSeverity.ERROR,
                threshold=5,
                window_minutes=10
            ),
            AlertRule(
                name="high_retry_delay",
                condition="avg_delay_ms > 10000",
                severity=AlertSeverity.WARNING,
                threshold=10000,
                window_minutes=5
            )
        ]
    
    def record_retry_attempt(
        self,
        operation_name: str,
        strategy: str,
        attempt_number: int,
        success: bool,
        duration_ms: float,
        error_type: Optional[str] = None,
        delay_ms: float = 0.0
    ):
        """Record a retry attempt with comprehensive metrics."""
        # Get or create operation metrics
        if operation_name not in self.operation_metrics:
            self.operation_metrics[operation_name] = RetryMetrics(operation_name)
        
        metrics = self.operation_metrics[operation_name]
        
        # Update Prometheus metrics
        if self.enable_prometheus:
            self.retry_attempts_total.labels(
                operation=operation_name,
                strategy=strategy,
                attempt=str(attempt_number)
            ).inc()
            
            self.retry_duration_histogram.labels(
                operation=operation_name,
                strategy=strategy
            ).observe(duration_ms / 1000.0)
            
            if delay_ms > 0:
                self.retry_delay_histogram.labels(
                    operation=operation_name,
                    strategy=strategy
                ).observe(delay_ms / 1000.0)
        
        # Update operation metrics
        if success:
            metrics.update_success(duration_ms)
            if self.enable_prometheus:
                self.retry_success_total.labels(
                    operation=operation_name,
                    strategy=strategy
                ).inc()
        else:
            error_type = error_type or "unknown"
            metrics.update_failure(error_type, delay_ms)
            if self.enable_prometheus:
                self.retry_failure_total.labels(
                    operation=operation_name,
                    strategy=strategy,
                    error_type=error_type
                ).inc()
        
        # Update success rate gauge
        if self.enable_prometheus:
            self.retry_success_rate_gauge.labels(operation=operation_name).set(metrics.success_rate)
        
        # Record OTEL metrics
        if self.enable_otel:
            self.otel_retry_counter.add(1, {
                "operation": operation_name,
                "strategy": strategy,
                "success": str(success)
            })
            
            self.otel_retry_duration.record(duration_ms / 1000.0, {
                "operation": operation_name,
                "strategy": strategy
            })
        
        # Check alert rules
        self._check_alerts(operation_name, metrics)
    
    def record_circuit_breaker_event(self, operation_name: str, event: str, state: int):
        """Record circuit breaker events."""
        if operation_name not in self.operation_metrics:
            self.operation_metrics[operation_name] = RetryMetrics(operation_name)
        
        metrics = self.operation_metrics[operation_name]
        
        if event == "trip":
            metrics.update_circuit_breaker_trip()
            if self.enable_prometheus:
                self.circuit_breaker_trips_total.labels(operation=operation_name).inc()
        
        if self.enable_prometheus:
            self.circuit_breaker_state_gauge.labels(operation=operation_name).set(state)
    
    def _check_alerts(self, operation_name: str, metrics: RetryMetrics):
        """Check and trigger alerts based on metrics."""
        for rule in self.alert_rules:
            if rule.evaluate(metrics):
                self._trigger_alert(operation_name, rule, metrics)
    
    def _trigger_alert(self, operation_name: str, rule: AlertRule, metrics: RetryMetrics):
        """Trigger an alert."""
        alert_message = (
            f"ALERT: {rule.name} for operation {operation_name} - "
            f"Severity: {rule.severity.value} - "
            f"Failure rate: {metrics.failure_rate:.2%} - "
            f"Circuit breaker trips: {metrics.circuit_breaker_trips}"
        )
        
        if rule.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            logger.error(alert_message)
        else:
            logger.warning(alert_message)
        
        # In production, this would integrate with alerting systems like PagerDuty, Slack, etc.
    
    def get_operation_metrics(self, operation_name: str) -> Optional[RetryMetrics]:
        """Get metrics for a specific operation."""
        return self.operation_metrics.get(operation_name)
    
    def get_all_metrics(self) -> Dict[str, RetryMetrics]:
        """Get all operation metrics."""
        return self.operation_metrics.copy()
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a custom alert rule."""
        self.alert_rules.append(rule)
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule by name."""
        self.alert_rules = [rule for rule in self.alert_rules if rule.name != rule_name]

# Global observability manager instance
_global_observability_manager: Optional[RetryObservabilityManager] = None

def get_observability_manager() -> RetryObservabilityManager:
    """Get global observability manager instance."""
    global _global_observability_manager
    if _global_observability_manager is None:
        _global_observability_manager = RetryObservabilityManager()
    return _global_observability_manager

def set_observability_manager(manager: RetryObservabilityManager):
    """Set global observability manager instance."""
    global _global_observability_manager
    _global_observability_manager = manager
