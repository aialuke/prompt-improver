"""
SLO/SLA Framework - Core definitions and models following Google SRE practices

Implements Service Level Objectives (SLOs) and Service Level Agreements (SLAs)
with error budget tracking, burn rate calculation, and multi-window monitoring.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from enum import Enum
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SLOType(Enum):
    """Types of SLO measurements"""
    AVAILABILITY = "availability"          # Uptime/success rate
    LATENCY = "latency"                   # Response time percentiles  
    ERROR_RATE = "error_rate"             # Error percentage
    THROUGHPUT = "throughput"             # Requests per second
    QUALITY = "quality"                   # Custom quality metrics

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

@dataclass
class SLOTarget:
    """Defines a specific SLO target with thresholds and alerting"""
    
    # Basic definition
    name: str
    slo_type: SLOType
    target_value: float  # e.g., 99.9 for 99.9% availability
    time_window: SLOTimeWindow
    
    # Measurement details
    description: str = ""
    unit: str = "percent"  # percent, ms, count, rate
    measurement_query: str = ""  # Prometheus query or metric name
    
    # Thresholds for alerting
    warning_threshold: float = 0.95  # Alert when SLI drops to 95% of target
    critical_threshold: float = 0.90  # Critical alert at 90% of target
    
    # Error budget configuration
    error_budget_burn_rate_1x: float = 1.0   # Normal burn rate (100% target)
    error_budget_burn_rate_6x: float = 6.0   # Fast burn rate alert  
    error_budget_burn_rate_36x: float = 36.0 # Emergency burn rate alert
    
    # Service details
    service_name: str = ""
    component: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate SLO target configuration"""
        if self.target_value <= 0 or self.target_value > 100:
            raise ValueError(f"SLO target must be 0-100, got {self.target_value}")
            
        if self.warning_threshold < self.critical_threshold:
            raise ValueError("Warning threshold must be >= critical threshold")

@dataclass
class ErrorBudget:
    """Error budget tracking for an SLO"""
    
    slo_target: SLOTarget
    time_window: SLOTimeWindow
    
    # Budget calculation  
    total_budget: float = 0.0      # Total error budget for period
    consumed_budget: float = 0.0   # Budget consumed so far
    remaining_budget: float = 0.0  # Budget remaining
    budget_percentage: float = 0.0 # Percentage of budget consumed
    
    # Burn rate tracking
    current_burn_rate: float = 0.0    # Current rate of budget consumption
    historical_burn_rates: List[float] = field(default_factory=list)
    
    # Time tracking
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    def calculate_budget(self, total_requests: int, failed_requests: int) -> None:
        """Calculate error budget based on actual traffic"""
        if total_requests == 0:
            return
            
        # Calculate total allowed failures for this SLO
        availability_target = self.slo_target.target_value / 100.0
        allowed_failure_rate = 1.0 - availability_target
        self.total_budget = total_requests * allowed_failure_rate
        
        # Calculate consumed budget
        self.consumed_budget = float(failed_requests)
        self.remaining_budget = max(0.0, self.total_budget - self.consumed_budget)
        
        # Calculate percentage consumed
        if self.total_budget > 0:
            self.budget_percentage = (self.consumed_budget / self.total_budget) * 100.0
        else:
            self.budget_percentage = 0.0
            
        self.last_updated = datetime.now(UTC)
    
    def calculate_burn_rate(self, measurement_period_seconds: int) -> float:
        """Calculate current burn rate (how fast we're consuming budget)"""
        if self.total_budget <= 0 or measurement_period_seconds <= 0:
            return 0.0
            
        # Burn rate = (consumed in period) / (expected consumption for entire window)
        window_seconds = self.time_window.seconds
        expected_consumption = self.total_budget * (measurement_period_seconds / window_seconds)
        
        if expected_consumption > 0:
            self.current_burn_rate = self.consumed_budget / expected_consumption
        else:
            self.current_burn_rate = 0.0
            
        # Track historical burn rates for trending
        self.historical_burn_rates.append(self.current_burn_rate)
        if len(self.historical_burn_rates) > 100:  # Keep last 100 measurements
            self.historical_burn_rates.pop(0)
            
        return self.current_burn_rate
    
    def is_budget_exhausted(self) -> bool:
        """Check if error budget is completely exhausted"""
        return self.remaining_budget <= 0.0
    
    def time_to_exhaustion(self) -> Optional[timedelta]:
        """Estimate time until budget exhaustion at current burn rate"""
        if self.current_burn_rate <= 0 or self.remaining_budget <= 0:
            return None
            
        # Calculate time remaining at current consumption rate
        seconds_remaining = self.remaining_budget / (self.current_burn_rate / 3600)  # per hour
        return timedelta(seconds=seconds_remaining)

@dataclass  
class BurnRate:
    """Burn rate calculation and alerting"""
    
    multiplier: float  # 1x, 6x, 36x etc.
    threshold: float   # Threshold for this burn rate
    window_size: SLOTimeWindow  # Measurement window
    
    current_rate: float = 0.0
    is_alerting: bool = False
    alert_start_time: Optional[datetime] = None
    alert_count: int = 0
    
    def should_alert(self, current_burn_rate: float) -> bool:
        """Determine if this burn rate threshold should trigger an alert"""
        was_alerting = self.is_alerting
        self.current_rate = current_burn_rate
        
        # Check if we exceed threshold
        exceeds_threshold = current_burn_rate >= (self.threshold * self.multiplier)
        
        if exceeds_threshold and not was_alerting:
            # Start alerting
            self.is_alerting = True
            self.alert_start_time = datetime.now(UTC)
            self.alert_count += 1
            return True
        elif not exceeds_threshold and was_alerting:
            # Stop alerting
            self.is_alerting = False
            self.alert_start_time = None
            
        return False

@dataclass
class SLODefinition:
    """Complete SLO definition with all configuration"""
    
    # Basic information
    name: str
    service_name: str
    description: str
    owner_team: str
    
    # SLO targets for different time windows
    targets: List[SLOTarget] = field(default_factory=list)
    
    # Error budget configuration
    error_budget_policy: str = "alerting_only"  # alerting_only, block_deploys, rollback_features
    
    # Alerting configuration  
    alert_channels: List[str] = field(default_factory=list)  # slack, pagerduty, email
    severity_levels: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0"
    tags: Dict[str, str] = field(default_factory=dict)
    
    def add_target(self, target: SLOTarget) -> None:
        """Add an SLO target to this definition"""
        target.service_name = self.service_name
        self.targets.append(target)
        self.updated_at = datetime.now(UTC)
    
    def get_target(self, name: str) -> Optional[SLOTarget]:
        """Get SLO target by name"""
        for target in self.targets:
            if target.name == name:
                return target
        return None
    
    def get_targets_by_type(self, slo_type: SLOType) -> List[SLOTarget]:
        """Get all targets of a specific type"""
        return [t for t in self.targets if t.slo_type == slo_type]
    
    def get_targets_by_window(self, window: SLOTimeWindow) -> List[SLOTarget]:
        """Get all targets for a specific time window"""
        return [t for t in self.targets if t.time_window == window]

# Predefined SLO templates following industry best practices
class SLOTemplates:
    """Common SLO templates for different service types"""
    
    @staticmethod
    def web_service_availability(service_name: str) -> SLODefinition:
        """Standard availability SLO for web services"""
        slo_def = SLODefinition(
            name=f"{service_name}_availability",
            service_name=service_name,
            description="Web service availability SLO",
            owner_team="platform"
        )
        
        # Multi-window availability targets
        for window, target in [
            (SLOTimeWindow.DAY_1, 99.9),   # 99.9% daily
            (SLOTimeWindow.WEEK_1, 99.5),  # 99.5% weekly  
            (SLOTimeWindow.MONTH_1, 99.0), # 99.0% monthly
        ]:
            slo_def.add_target(SLOTarget(
                name=f"availability_{window.value}",
                slo_type=SLOType.AVAILABILITY,
                target_value=target,
                time_window=window,
                description=f"Service availability over {window.human_readable}",
                measurement_query=f"rate(http_requests_total{{service=\"{service_name}\",status!~\"5..\"}}" +
                                f"[{window.value}]) / rate(http_requests_total{{service=\"{service_name}\"}}[{window.value}])"
            ))
            
        return slo_def
    
    @staticmethod  
    def api_latency(service_name: str) -> SLODefinition:
        """Standard latency SLO for API services"""
        slo_def = SLODefinition(
            name=f"{service_name}_latency",
            service_name=service_name,
            description="API latency SLO",
            owner_team="platform"
        )
        
        # Latency percentile targets
        for percentile, target_ms in [
            ("p50", 100),   # 50th percentile < 100ms
            ("p95", 500),   # 95th percentile < 500ms  
            ("p99", 1000),  # 99th percentile < 1000ms
        ]:
            slo_def.add_target(SLOTarget(
                name=f"latency_{percentile}",
                slo_type=SLOType.LATENCY,
                target_value=target_ms,
                time_window=SLOTimeWindow.DAY_1,
                unit="ms",
                description=f"API {percentile} latency under {target_ms}ms",
                measurement_query=f"histogram_quantile(0.{percentile[1:]}, " +
                                f"rate(http_request_duration_seconds_bucket{{service=\"{service_name}\"}}[24h]))"
            ))
            
        return slo_def
    
    @staticmethod
    def error_rate(service_name: str) -> SLODefinition:
        """Standard error rate SLO"""
        slo_def = SLODefinition(
            name=f"{service_name}_error_rate",
            service_name=service_name, 
            description="Service error rate SLO",
            owner_team="platform"
        )
        
        slo_def.add_target(SLOTarget(
            name="error_rate_24h",
            slo_type=SLOType.ERROR_RATE,
            target_value=1.0,  # < 1% error rate
            time_window=SLOTimeWindow.DAY_1,
            description="Error rate under 1% over 24 hours",
            measurement_query=f"rate(http_requests_total{{service=\"{service_name}\",status=~\"5..\"}}[24h]) " +
                            f"/ rate(http_requests_total{{service=\"{service_name}\"}}[24h]) * 100"
        ))
        
        return slo_def