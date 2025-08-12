"""SLO/SLA Framework - Core definitions and models following Google SRE practices

Implements Service Level Objectives (SLOs) and Service Level Agreements (SLAs)
with error budget tracking, burn rate calculation, and multi-window monitoring.
"""

import logging
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SLOType(Enum):
    """Types of SLO measurements"""

    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    QUALITY = "quality"


class SLOTimeWindow(Enum):
    """Time windows for SLO calculations"""

    HOUR_1 = "1h"
    HOUR_6 = "6h"
    DAY_1 = "24h"
    DAY_3 = "3d"
    WEEK_1 = "7d"
    WEEK_2 = "14d"
    MONTH_1 = "30d"
    MONTH_3 = "90d"

    @property
    def seconds(self) -> int:
        """Convert time window to seconds"""
        mapping = {
            "1h": 3600,
            "6h": 21600,
            "24h": 86400,
            "3d": 259200,
            "7d": 604800,
            "14d": 1209600,
            "30d": 2592000,
            "90d": 7776000,
        }
        return mapping[self.value]

    @property
    def human_readable(self) -> str:
        """Human readable time window"""
        mapping = {
            "1h": "1 hour",
            "6h": "6 hours",
            "24h": "24 hours",
            "3d": "3 days",
            "7d": "7 days",
            "14d": "14 days",
            "30d": "30 days",
            "90d": "90 days",
        }
        return mapping[self.value]


class SLOTarget(BaseModel):
    """Defines a specific SLO target with thresholds and alerting"""

    name: str = Field(min_length=1, max_length=255)
    slo_type: SLOType
    target_value: float = Field(gt=0.0, le=100.0)
    time_window: SLOTimeWindow
    description: str = Field(default="", max_length=1000)
    unit: str = Field(default="percent", max_length=50)
    measurement_query: str = Field(default="", max_length=2000)
    warning_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    critical_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    error_budget_burn_rate_1x: float = Field(default=1.0, ge=0.1, le=1000.0)
    error_budget_burn_rate_6x: float = Field(default=6.0, ge=0.1, le=1000.0)
    error_budget_burn_rate_36x: float = Field(default=36.0, ge=0.1, le=1000.0)
    service_name: str = Field(default="", max_length=255)
    component: str = Field(default="", max_length=255)
    labels: dict[str, str] = Field(default_factory=dict)

    def __post_init__(self):
        """Validate SLO target configuration"""
        if self.target_value <= 0 or self.target_value > 100:
            raise ValueError(f"SLO target must be 0-100, got {self.target_value}")
        if self.warning_threshold < self.critical_threshold:
            raise ValueError("Warning threshold must be >= critical threshold")


class ErrorBudget(BaseModel):
    """Error budget tracking for an SLO"""

    slo_target: SLOTarget
    time_window: SLOTimeWindow
    total_budget: float = Field(default=0.0, ge=0.0, le=1000000.0)
    consumed_budget: float = Field(default=0.0, ge=0.0, le=1000000.0)
    remaining_budget: float = Field(default=0.0, ge=0.0, le=1000000.0)
    budget_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    current_burn_rate: float = Field(default=0.0, ge=0.0, le=1000.0)
    historical_burn_rates: list[float] = Field(default_factory=list)
    window_start: datetime | None = Field(default=None)
    window_end: datetime | None = Field(default=None)
    last_updated: datetime | None = Field(default=None)

    def calculate_budget(self, total_requests: int, failed_requests: int) -> None:
        """Calculate error budget based on actual traffic"""
        if total_requests == 0:
            return
        availability_target = self.slo_target.target_value / 100.0
        allowed_failure_rate = 1.0 - availability_target
        self.total_budget = total_requests * allowed_failure_rate
        self.consumed_budget = float(failed_requests)
        self.remaining_budget = max(0.0, self.total_budget - self.consumed_budget)
        if self.total_budget > 0:
            self.budget_percentage = self.consumed_budget / self.total_budget * 100.0
        else:
            self.budget_percentage = 0.0
        self.last_updated = datetime.now(UTC)

    def calculate_burn_rate(self, measurement_period_seconds: int) -> float:
        """Calculate current burn rate (how fast we're consuming budget)"""
        if self.total_budget <= 0 or measurement_period_seconds <= 0:
            return 0.0
        window_seconds = self.time_window.seconds
        expected_consumption = self.total_budget * (
            measurement_period_seconds / window_seconds
        )
        if expected_consumption > 0:
            self.current_burn_rate = self.consumed_budget / expected_consumption
        else:
            self.current_burn_rate = 0.0
        self.historical_burn_rates.append(self.current_burn_rate)
        if len(self.historical_burn_rates) > 100:
            self.historical_burn_rates.pop(0)
        return self.current_burn_rate

    def is_budget_exhausted(self) -> bool:
        """Check if error budget is completely exhausted"""
        return self.remaining_budget <= 0.0

    def time_to_exhaustion(self) -> timedelta | None:
        """Estimate time until budget exhaustion at current burn rate"""
        if self.current_burn_rate <= 0 or self.remaining_budget <= 0:
            return None
        seconds_remaining = self.remaining_budget / (self.current_burn_rate / 3600)
        return timedelta(seconds=seconds_remaining)


class BurnRate(BaseModel):
    """Burn rate calculation and alerting"""

    multiplier: float = Field(ge=0.1, le=1000.0)
    threshold: float = Field(ge=0.0, le=100.0)
    window_size: SLOTimeWindow
    current_rate: float = Field(default=0.0, ge=0.0, le=1000.0)
    is_alerting: bool = Field(default=False)
    alert_start_time: datetime | None = Field(default=None)
    alert_count: int = Field(default=0, ge=0)

    def should_alert(self, current_burn_rate: float) -> bool:
        """Determine if this burn rate threshold should trigger an alert"""
        was_alerting = self.is_alerting
        self.current_rate = current_burn_rate
        exceeds_threshold = current_burn_rate >= self.threshold * self.multiplier
        if exceeds_threshold and (not was_alerting):
            self.is_alerting = True
            self.alert_start_time = datetime.now(UTC)
            self.alert_count += 1
            return True
        if not exceeds_threshold and was_alerting:
            self.is_alerting = False
            self.alert_start_time = None
        return False


class SLODefinition(BaseModel):
    """Complete SLO definition with all configuration"""

    name: str = Field(min_length=1, max_length=255)
    service_name: str = Field(min_length=1, max_length=255)
    description: str = Field(min_length=1, max_length=1000)
    owner_team: str = Field(min_length=1, max_length=255)
    targets: list[SLOTarget] = Field(default_factory=list)
    error_budget_policy: str = Field(default="alerting_only", max_length=100)
    alert_channels: list[str] = Field(default_factory=list)
    severity_levels: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0", max_length=50)
    tags: dict[str, str] = Field(default_factory=dict)

    def add_target(self, target: SLOTarget) -> None:
        """Add an SLO target to this definition"""
        target.service_name = self.service_name
        self.targets.append(target)
        self.updated_at = datetime.now(UTC)

    def get_target(self, name: str) -> SLOTarget | None:
        """Get SLO target by name"""
        for target in self.targets:
            if target.name == name:
                return target
        return None

    def get_targets_by_type(self, slo_type: SLOType) -> list[SLOTarget]:
        """Get all targets of a specific type"""
        return [t for t in self.targets if t.slo_type == slo_type]

    def get_targets_by_window(self, window: SLOTimeWindow) -> list[SLOTarget]:
        """Get all targets for a specific time window"""
        return [t for t in self.targets if t.time_window == window]


class SLOTemplates:
    """Common SLO templates for different service types"""

    @staticmethod
    def web_service_availability(service_name: str) -> SLODefinition:
        """Standard availability SLO for web services"""
        slo_def = SLODefinition(
            name=f"{service_name}_availability",
            service_name=service_name,
            description="Web service availability SLO",
            owner_team="platform",
        )
        for window, target in [
            (SLOTimeWindow.DAY_1, 99.9),
            (SLOTimeWindow.WEEK_1, 99.5),
            (SLOTimeWindow.MONTH_1, 99.0),
        ]:
            slo_def.add_target(
                SLOTarget(
                    name=f"availability_{window.value}",
                    slo_type=SLOType.AVAILABILITY,
                    target_value=target,
                    time_window=window,
                    description=f"Service availability over {window.human_readable}",
                    measurement_query=f'rate(http_requests_total{{service="{service_name}",status!~"5.."}}'
                    + f'[{window.value}]) / rate(http_requests_total{{service="{service_name}"}}[{window.value}])',
                )
            )
        return slo_def

    @staticmethod
    def api_latency(service_name: str) -> SLODefinition:
        """Standard latency SLO for API services"""
        slo_def = SLODefinition(
            name=f"{service_name}_latency",
            service_name=service_name,
            description="API latency SLO",
            owner_team="platform",
        )
        for percentile, target_ms in [("p50", 100), ("p95", 500), ("p99", 1000)]:
            slo_def.add_target(
                SLOTarget(
                    name=f"latency_{percentile}",
                    slo_type=SLOType.LATENCY,
                    target_value=target_ms,
                    time_window=SLOTimeWindow.DAY_1,
                    unit="ms",
                    description=f"API {percentile} latency under {target_ms}ms",
                    measurement_query=f"histogram_quantile(0.{percentile[1:]}, "
                    + f'rate(http_request_duration_seconds_bucket{{service="{service_name}"}}[24h]))',
                )
            )
        return slo_def

    @staticmethod
    def error_rate(service_name: str) -> SLODefinition:
        """Standard error rate SLO"""
        slo_def = SLODefinition(
            name=f"{service_name}_error_rate",
            service_name=service_name,
            description="Service error rate SLO",
            owner_team="platform",
        )
        slo_def.add_target(
            SLOTarget(
                name="error_rate_24h",
                slo_type=SLOType.ERROR_RATE,
                target_value=1.0,
                time_window=SLOTimeWindow.DAY_1,
                description="Error rate under 1% over 24 hours",
                measurement_query=f'rate(http_requests_total{{service="{service_name}",status=~"5.."}}[24h]) '
                + f'/ rate(http_requests_total{{service="{service_name}"}}[24h]) * 100',
            )
        )
        return slo_def
