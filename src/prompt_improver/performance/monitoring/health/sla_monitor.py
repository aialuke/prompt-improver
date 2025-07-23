"""
Service Level Agreement (SLA) Monitoring for Health Checks
2025 Enterprise-grade SLA tracking and alerting
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SLAStatus(Enum):
    """SLA compliance status"""
    MEETING = "meeting"      # Meeting all SLA targets
    AT_RISK = "at_risk"      # Close to breaching SLA
    BREACHING = "breaching"  # Currently breaching SLA


@dataclass
class SLATarget:
    """Definition of an SLA target"""
    name: str
    description: str
    target_value: float
    unit: str  # e.g., "ms", "percent", "count"
    measurement_window_seconds: int = 300  # 5 minutes default

    # Thresholds
    warning_threshold: float = 0.9  # 90% of target = warning
    critical_threshold: float = 1.0  # 100% of target = breach

    # Alert configuration
    alert_on_breach: bool = True
    alert_cooldown_seconds: int = 300  # Don't re-alert for 5 minutes


@dataclass
class SLAConfiguration:
    """Complete SLA configuration for a service"""
    service_name: str

    # Response time SLAs
    response_time_p50_ms: float = 100    # 50th percentile
    response_time_p95_ms: float = 500    # 95th percentile
    response_time_p99_ms: float = 1000   # 99th percentile

    # Availability SLAs
    availability_target: float = 0.999    # 99.9% uptime
    error_rate_threshold: float = 0.01    # 1% error rate max

    # Throughput SLAs (optional)
    min_successful_checks_per_minute: Optional[int] = None

    # Custom SLA targets
    custom_targets: List[SLATarget] = field(default_factory=list)


class SLAMeasurement:
    """Tracks measurements for a specific SLA target"""

    def __init__(self, target: SLATarget):
        self.target = target
        self.measurements = deque(maxlen=1000)  # Keep last 1000 measurements
        self.last_alert_time: Optional[float] = None

    def add_measurement(self, value: float, timestamp: Optional[float] = None):
        """Add a new measurement"""
        timestamp = timestamp or time.time()
        self.measurements.append((timestamp, value))

    def get_current_value(self) -> Optional[float]:
        """Calculate current value within measurement window"""
        if not self.measurements:
            return None

        current_time = time.time()
        window_start = current_time - self.target.measurement_window_seconds

        # Filter measurements within window
        recent_values = [
            value for timestamp, value in self.measurements
            if timestamp >= window_start
        ]

        if not recent_values:
            return None

        # Calculate based on unit type
        if "percent" in self.target.unit or "rate" in self.target.name.lower():
            return statistics.mean(recent_values)
        elif "p50" in self.target.name:
            return statistics.median(recent_values)
        elif "p95" in self.target.name:
            sorted_values = sorted(recent_values)
            index = int(len(sorted_values) * 0.95)
            # Handle edge case for 100th percentile
            if index >= len(sorted_values):
                index = len(sorted_values) - 1
            return sorted_values[index]
        elif "p99" in self.target.name:
            sorted_values = sorted(recent_values)
            index = int(len(sorted_values) * 0.99)
            # Handle edge case for 100th percentile
            if index >= len(sorted_values):
                index = len(sorted_values) - 1
            return sorted_values[index]
        else:
            return statistics.mean(recent_values)

    def get_compliance_status(self) -> SLAStatus:
        """Check if SLA is being met"""
        current_value = self.get_current_value()
        if current_value is None:
            return SLAStatus.MEETING  # No data = assume meeting

        # For availability and success rates, higher values are better
        # For response times and error rates, lower values are better
        is_higher_better = (
            "availability" in self.target.name.lower() or
            "success" in self.target.name.lower() or
            "uptime" in self.target.name.lower()
        )

        if is_higher_better:
            # For availability: current should be >= target
            if current_value < self.target.target_value * self.target.warning_threshold:
                return SLAStatus.BREACHING
            elif current_value < self.target.target_value:
                return SLAStatus.AT_RISK
            else:
                return SLAStatus.MEETING
        else:
            # For response time/error rate: current should be <= target
            ratio = current_value / self.target.target_value
            if ratio >= self.target.critical_threshold:
                return SLAStatus.BREACHING
            elif ratio >= self.target.warning_threshold:
                return SLAStatus.AT_RISK
            else:
                return SLAStatus.MEETING

    def should_alert(self) -> bool:
        """Check if we should send an alert"""
        if not self.target.alert_on_breach:
            return False

        if self.get_compliance_status() != SLAStatus.BREACHING:
            return False

        # Check cooldown
        if self.last_alert_time:
            time_since_alert = time.time() - self.last_alert_time
            if time_since_alert < self.target.alert_cooldown_seconds:
                return False

        return True


class SLAMonitor:
    """
    Monitors SLA compliance for health checks
    """

    def __init__(
        self,
        service_name: str,
        config: SLAConfiguration,
        on_sla_breach: Optional[Callable[[str, SLATarget, float], None]] = None
    ):
        self.service_name = service_name
        self.config = config
        self.on_sla_breach = on_sla_breach

        # Initialize measurements for standard SLAs
        self.measurements: Dict[str, SLAMeasurement] = {}
        self._initialize_standard_slas()

        # Add custom SLA targets
        for target in config.custom_targets:
            self.measurements[target.name] = SLAMeasurement(target)

        # Tracking
        self._check_count = 0
        self._success_count = 0
        self._start_time = time.time()

    def _initialize_standard_slas(self):
        """Initialize standard SLA measurements"""
        # Response time SLAs
        self.measurements["response_time_p50"] = SLAMeasurement(
            SLATarget(
                name="response_time_p50",
                description="50th percentile response time",
                target_value=self.config.response_time_p50_ms,
                unit="ms"
            )
        )

        self.measurements["response_time_p95"] = SLAMeasurement(
            SLATarget(
                name="response_time_p95",
                description="95th percentile response time",
                target_value=self.config.response_time_p95_ms,
                unit="ms"
            )
        )

        self.measurements["response_time_p99"] = SLAMeasurement(
            SLATarget(
                name="response_time_p99",
                description="99th percentile response time",
                target_value=self.config.response_time_p99_ms,
                unit="ms"
            )
        )

        # Availability SLA
        self.measurements["availability"] = SLAMeasurement(
            SLATarget(
                name="availability",
                description="Service availability",
                target_value=self.config.availability_target,
                unit="percent"
            )
        )

        # Error rate SLA
        self.measurements["error_rate"] = SLAMeasurement(
            SLATarget(
                name="error_rate",
                description="Error rate",
                target_value=self.config.error_rate_threshold,
                unit="percent"
            )
        )

    def record_health_check(
        self,
        success: bool,
        response_time_ms: float,
        custom_metrics: Optional[Dict[str, float]] = None
    ):
        """Record a health check result and update SLA metrics"""
        self._check_count += 1
        if success:
            self._success_count += 1

        # Update response time measurements
        self.measurements["response_time_p50"].add_measurement(response_time_ms)
        self.measurements["response_time_p95"].add_measurement(response_time_ms)
        self.measurements["response_time_p99"].add_measurement(response_time_ms)

        # Update availability (per-call basis, not cumulative)
        availability_value = 1.0 if success else 0.0
        self.measurements["availability"].add_measurement(availability_value)

        # Update error rate (per-call basis)
        error_rate_value = 0.0 if success else 1.0
        self.measurements["error_rate"].add_measurement(error_rate_value)

        # Update custom metrics
        if custom_metrics:
            for metric_name, value in custom_metrics.items():
                if metric_name in self.measurements:
                    self.measurements[metric_name].add_measurement(value)

        # Check for SLA breaches
        self._check_sla_compliance()

    def _check_sla_compliance(self):
        """Check all SLAs and trigger alerts if needed"""
        for name, measurement in self.measurements.items():
            if measurement.should_alert():
                current_value = measurement.get_current_value()

                # Log the breach
                logger.warning(
                    f"SLA breach detected for {self.service_name}",
                    extra={
                        "service": self.service_name,
                        "sla_target": name,
                        "target_value": measurement.target.target_value,
                        "current_value": current_value,
                        "unit": measurement.target.unit,
                        "status": "breaching"
                    }
                )

                # Update last alert time
                measurement.last_alert_time = time.time()

                # Trigger callback
                if self.on_sla_breach:
                    self.on_sla_breach(
                        self.service_name,
                        measurement.target,
                        current_value
                    )

    def get_sla_report(self) -> Dict[str, Any]:
        """Generate comprehensive SLA compliance report"""
        report = {
            "service": self.service_name,
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self._start_time,
            "total_checks": self._check_count,
            "successful_checks": self._success_count,
            "overall_availability": self._success_count / max(1, self._check_count),
            "sla_targets": {}
        }

        # Add individual SLA status
        for name, measurement in self.measurements.items():
            current_value = measurement.get_current_value()
            status = measurement.get_compliance_status()

            report["sla_targets"][name] = {
                "description": measurement.target.description,
                "target_value": measurement.target.target_value,
                "current_value": current_value,
                "unit": measurement.target.unit,
                "status": status.value,
                "compliance_ratio": (
                    current_value / measurement.target.target_value
                    if current_value is not None else None
                )
            }

        # Overall SLA status
        statuses = [m.get_compliance_status() for m in self.measurements.values()]
        if any(s == SLAStatus.BREACHING for s in statuses):
            report["overall_status"] = SLAStatus.BREACHING.value
        elif any(s == SLAStatus.AT_RISK for s in statuses):
            report["overall_status"] = SLAStatus.AT_RISK.value
        else:
            report["overall_status"] = SLAStatus.MEETING.value

        return report

    def get_sla_metrics_for_export(self) -> Dict[str, float]:
        """Get SLA metrics in format suitable for Prometheus/Grafana"""
        metrics = {}

        for name, measurement in self.measurements.items():
            current_value = measurement.get_current_value()
            if current_value is not None:
                # Current value
                metrics[f"sla_{name}_current"] = current_value

                # Target value
                metrics[f"sla_{name}_target"] = measurement.target.target_value

                # Compliance ratio (0-1, where 1 is meeting SLA)
                ratio = current_value / measurement.target.target_value
                metrics[f"sla_{name}_compliance_ratio"] = min(1.0, 1.0 / max(0.01, ratio))

                # Status as numeric (0=meeting, 1=at_risk, 2=breaching)
                status = measurement.get_compliance_status()
                status_map = {
                    SLAStatus.MEETING: 0,
                    SLAStatus.AT_RISK: 1,
                    SLAStatus.BREACHING: 2
                }
                metrics[f"sla_{name}_status"] = status_map[status]

        return metrics


# Global SLA monitor registry
sla_monitors: Dict[str, SLAMonitor] = {}


def get_or_create_sla_monitor(
    service_name: str,
    config: Optional[SLAConfiguration] = None
) -> SLAMonitor:
    """Get or create an SLA monitor for a service"""
    if service_name not in sla_monitors:
        if config is None:
            config = SLAConfiguration(service_name=service_name)
        sla_monitors[service_name] = SLAMonitor(service_name, config)
    return sla_monitors[service_name]