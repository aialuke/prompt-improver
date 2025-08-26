"""Service Level Agreement (SLA) Monitoring for Health Checks
2025 Enterprise-grade SLA tracking and alerting.
"""

import logging
import statistics
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from prompt_improver.core.config import get_config

logger = logging.getLogger(__name__)


class SLAStatus(Enum):
    """SLA compliance status."""

    MEETING = "meeting"
    AT_RISK = "at_risk"
    BREACHING = "breaching"


@dataclass
class SLATarget:
    """Definition of an SLA target."""

    name: str
    description: str
    target_value: float
    unit: str
    measurement_window_seconds: int = 300
    warning_threshold: float = 0.9
    critical_threshold: float = 1.0
    alert_on_breach: bool = True
    alert_cooldown_seconds: int = 300

    @classmethod
    def from_global_config(
        cls, name: str, description: str, target_value: float, unit: str
    ) -> "SLATarget":
        """Create SLA target with values from global config."""
        config = get_config()
        sla_config = config.sla_monitoring
        return cls(
            name=name,
            description=description,
            target_value=target_value,
            unit=unit,
            measurement_window_seconds=sla_config.measurement_window_seconds,
            warning_threshold=sla_config.warning_threshold,
            critical_threshold=sla_config.critical_threshold,
            alert_on_breach=True,
            alert_cooldown_seconds=sla_config.alert_cooldown_seconds,
        )


@dataclass
class SLAConfiguration:
    """Complete SLA configuration for a service."""

    service_name: str
    response_time_p50_ms: float = 100
    response_time_p95_ms: float = 500
    response_time_p99_ms: float = 1000
    availability_target: float = 0.999
    error_rate_threshold: float = 0.01
    min_successful_checks_per_minute: int | None = None
    custom_targets: list[SLATarget] = field(default_factory=list)

    @classmethod
    def from_global_config(cls, service_name: str) -> "SLAConfiguration":
        """Create configuration from global config system."""
        config = get_config()
        sla_config = config.sla_monitoring
        return cls(
            service_name=service_name,
            response_time_p50_ms=sla_config.response_time_p50_ms,
            response_time_p95_ms=sla_config.response_time_p95_ms,
            response_time_p99_ms=sla_config.response_time_p99_ms,
            availability_target=sla_config.availability_target,
            error_rate_threshold=sla_config.error_rate_threshold,
            custom_targets=[],
        )


class SLAMeasurement:
    """Tracks measurements for a specific SLA target."""

    def __init__(self, target: SLATarget) -> None:
        self.target = target
        self.measurements = deque(maxlen=1000)
        self.last_alert_time: float | None = None

    def add_measurement(self, value: float, timestamp: float | None = None):
        """Add a new measurement."""
        timestamp = timestamp or time.time()
        self.measurements.append((timestamp, value))

    def get_current_value(self) -> float | None:
        """Calculate current value within measurement window."""
        if not self.measurements:
            return None
        current_time = time.time()
        window_start = current_time - self.target.measurement_window_seconds
        recent_values = [
            value
            for timestamp, value in self.measurements
            if window_start <= timestamp <= current_time
        ]
        if not recent_values:
            return None
        if "percent" in self.target.unit or "rate" in self.target.name.lower():
            return statistics.mean(recent_values)
        if "p50" in self.target.name:
            return statistics.median(recent_values)
        if "p95" in self.target.name:
            sorted_values = sorted(recent_values)
            index = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * 0.95)))
            return sorted_values[index]
        if "p99" in self.target.name:
            sorted_values = sorted(recent_values)
            index = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * 0.99)))
            return sorted_values[index]
        return statistics.mean(recent_values)

    def get_compliance_status(self) -> SLAStatus:
        """Check if SLA is being met."""
        current_value = self.get_current_value()
        if current_value is None:
            return SLAStatus.MEETING
        is_higher_better = (
            "availability" in self.target.name.lower()
            or "success" in self.target.name.lower()
            or "uptime" in self.target.name.lower()
            or ("throughput" in self.target.name.lower())
            or ("hit_rate" in self.target.name.lower())
            or ("accuracy" in self.target.name.lower())
        )
        if is_higher_better:
            if current_value < self.target.target_value * self.target.warning_threshold:
                return SLAStatus.BREACHING
            if current_value < self.target.target_value:
                return SLAStatus.AT_RISK
            return SLAStatus.MEETING
        if current_value >= self.target.target_value * self.target.critical_threshold:
            return SLAStatus.BREACHING
        if current_value >= self.target.target_value * self.target.warning_threshold:
            return SLAStatus.AT_RISK
        return SLAStatus.MEETING

    def should_alert(self) -> bool:
        """Check if we should send an alert."""
        if not self.target.alert_on_breach:
            return False
        if self.get_compliance_status() != SLAStatus.BREACHING:
            return False
        if self.last_alert_time:
            time_since_alert = time.time() - self.last_alert_time
            if time_since_alert < self.target.alert_cooldown_seconds:
                return False
        return True


class SLAMonitor:
    """Monitors SLA compliance for health checks."""

    def __init__(
        self,
        service_name: str,
        config: SLAConfiguration,
        on_sla_breach: Callable[[str, SLATarget, float], None] | None = None,
    ) -> None:
        self.service_name = service_name
        self.config = config
        self.on_sla_breach = on_sla_breach
        self.measurements: dict[str, SLAMeasurement] = {}
        self._initialize_standard_slas()
        for target in config.custom_targets:
            self.measurements[target.name] = SLAMeasurement(target)
        self._check_count = 0
        self._success_count = 0
        self._start_time = time.time()

    def _initialize_standard_slas(self) -> None:
        """Initialize standard SLA measurements."""
        self.measurements["response_time_p50"] = SLAMeasurement(
            SLATarget(
                name="response_time_p50",
                description="50th percentile response time",
                target_value=self.config.response_time_p50_ms,
                unit="ms",
            )
        )
        self.measurements["response_time_p95"] = SLAMeasurement(
            SLATarget(
                name="response_time_p95",
                description="95th percentile response time",
                target_value=self.config.response_time_p95_ms,
                unit="ms",
            )
        )
        self.measurements["response_time_p99"] = SLAMeasurement(
            SLATarget(
                name="response_time_p99",
                description="99th percentile response time",
                target_value=self.config.response_time_p99_ms,
                unit="ms",
            )
        )
        self.measurements["availability"] = SLAMeasurement(
            SLATarget(
                name="availability",
                description="Service availability",
                target_value=self.config.availability_target,
                unit="percent",
            )
        )
        self.measurements["error_rate"] = SLAMeasurement(
            SLATarget(
                name="error_rate",
                description="Error rate",
                target_value=self.config.error_rate_threshold,
                unit="percent",
            )
        )

    def record_health_check(
        self,
        success: bool,
        response_time_ms: float,
        custom_metrics: dict[str, float] | None = None,
    ):
        """Record a health check result and update SLA metrics."""
        self._check_count += 1
        if success:
            self._success_count += 1
        self.measurements["response_time_p50"].add_measurement(response_time_ms)
        self.measurements["response_time_p95"].add_measurement(response_time_ms)
        self.measurements["response_time_p99"].add_measurement(response_time_ms)
        availability_value = 1.0 if success else 0.0
        self.measurements["availability"].add_measurement(availability_value)
        error_rate_value = 0.0 if success else 1.0
        self.measurements["error_rate"].add_measurement(error_rate_value)
        if custom_metrics:
            for metric_name, value in custom_metrics.items():
                if metric_name in self.measurements:
                    self.measurements[metric_name].add_measurement(value)
        self._check_sla_compliance()

    def _check_sla_compliance(self) -> None:
        """Check all SLAs and trigger alerts if needed."""
        for measurement in self.measurements.values():
            if measurement.should_alert():
                current_value = measurement.get_current_value()
                logger.warning(f"SLA breach detected for {self.service_name}")
                measurement.last_alert_time = time.time()
                if self.on_sla_breach:
                    self.on_sla_breach(
                        self.service_name, measurement.target, current_value
                    )

    def get_sla_report(self) -> dict[str, Any]:
        """Generate comprehensive SLA compliance report."""
        report = {
            "service": self.service_name,
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self._start_time,
            "total_checks": self._check_count,
            "successful_checks": self._success_count,
            "overall_availability": self._success_count / max(1, self._check_count),
            "sla_targets": {},
        }
        for name, measurement in self.measurements.items():
            current_value = measurement.get_current_value()
            status = measurement.get_compliance_status()
            report["sla_targets"][name] = {
                "description": measurement.target.description,
                "target_value": measurement.target.target_value,
                "current_value": current_value,
                "unit": measurement.target.unit,
                "status": status.value,
                "compliance_ratio": current_value / measurement.target.target_value
                if current_value is not None
                else None,
            }
        statuses = [m.get_compliance_status() for m in self.measurements.values()]
        if any(s == SLAStatus.BREACHING for s in statuses):
            report["overall_status"] = SLAStatus.BREACHING.value
        elif any(s == SLAStatus.AT_RISK for s in statuses):
            report["overall_status"] = SLAStatus.AT_RISK.value
        else:
            report["overall_status"] = SLAStatus.MEETING.value
        return report

    def get_sla_metrics_for_export(self) -> dict[str, float]:
        """Get SLA metrics in format suitable for OpenTelemetry/Grafana."""
        metrics = {}
        for name, measurement in self.measurements.items():
            current_value = measurement.get_current_value()
            if current_value is not None:
                metrics[f"sla_{name}_current"] = current_value
                metrics[f"sla_{name}_target"] = measurement.target.target_value
                ratio = current_value / measurement.target.target_value
                metrics[f"sla_{name}_compliance_ratio"] = min(
                    1.0, 1.0 / max(0.01, ratio)
                )
                status = measurement.get_compliance_status()
                status_map = {
                    SLAStatus.MEETING: 0,
                    SLAStatus.AT_RISK: 1,
                    SLAStatus.BREACHING: 2,
                }
                metrics[f"sla_{name}_status"] = status_map[status]
        return metrics


sla_monitors: dict[str, SLAMonitor] = {}


def get_or_create_sla_monitor(
    service_name: str, config: SLAConfiguration | None = None
) -> SLAMonitor:
    """Get or create an SLA monitor for a service."""
    if service_name not in sla_monitors:
        if config is None:
            config = SLAConfiguration(service_name=service_name)
        sla_monitors[service_name] = SLAMonitor(service_name, config)
    return sla_monitors[service_name]
