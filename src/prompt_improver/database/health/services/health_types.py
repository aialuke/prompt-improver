"""Type definitions and data classes for database health monitoring services.

Defines shared data structures and types used across all health monitoring
services to ensure consistent data exchange and type safety.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, List


@dataclass
class DatabaseHealthMetrics:
    """Comprehensive database health metrics data structure.
    
    This is the main data structure that consolidates all health metrics
    from different services into a single, comprehensive view.
    """
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    # Connection metrics from DatabaseConnectionService
    connection_pool: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics from HealthMetricsService
    query_performance: Dict[str, Any] = field(default_factory=dict)
    storage: Dict[str, Any] = field(default_factory=dict)
    replication: Dict[str, Any] = field(default_factory=dict)
    locks: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)
    transactions: Dict[str, Any] = field(default_factory=dict)
    
    # Health assessment from AlertingService
    health_score: float = 100.0
    issues: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    collection_time_ms: float = 0.0
    version: str = "2025.1.0"


@dataclass
class ConnectionHealthMetrics:
    """Connection pool health metrics data structure."""
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    # Pool configuration
    pool_configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Connection states
    connection_states: Dict[str, int] = field(default_factory=dict)
    
    # Utilization metrics
    utilization_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Age statistics
    age_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Connection details (limited for performance)
    connection_details: List[Dict[str, Any]] = field(default_factory=list)
    
    # Health indicators
    health_indicators: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    collection_time_ms: float = 0.0


@dataclass
class QueryPerformanceMetrics:
    """Query performance metrics data structure."""
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    # Extension availability
    pg_stat_statements_available: bool = False
    
    # Query analysis
    slow_queries: List[Dict[str, Any]] = field(default_factory=list)
    frequent_queries: List[Dict[str, Any]] = field(default_factory=list)
    io_intensive_queries: List[Dict[str, Any]] = field(default_factory=list)
    
    # Cache performance
    cache_performance: Dict[str, Any] = field(default_factory=dict)
    
    # Current activity
    current_activity: List[Dict[str, Any]] = field(default_factory=list)
    
    # Summaries
    performance_summary: str = ""
    overall_assessment: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    collection_time_ms: float = 0.0


@dataclass
class HealthAlert:
    """Health alert data structure."""
    
    severity: str  # "info", "warning", "critical"
    category: str  # "connection_pool", "query_performance", etc.
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metric_name: str = ""
    metric_value: float = 0.0
    threshold: float = 0.0
    source_service: str = ""
    alert_id: str = ""


@dataclass
class HealthRecommendation:
    """Health recommendation data structure."""
    
    category: str  # "connection_pool", "query_performance", etc.
    priority: str  # "low", "medium", "high", "critical"
    action: str
    description: str
    expected_impact: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    recommendation_id: str = ""


@dataclass
class HealthTrend:
    """Health trend analysis data structure."""
    
    period_hours: int
    data_points: int
    metric_name: str
    trend_direction: str  # "improving", "stable", "degrading", "unknown"
    current_value: float
    average_value: float
    min_value: float = 0.0
    max_value: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class HealthThreshold:
    """Health threshold configuration data structure."""
    
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison_operator: str = "greater_than"  # "greater_than", "less_than"
    enabled: bool = True
    description: str = ""


# Type aliases for common data structures
ConnectionDetails = List[Dict[str, Any]]
QueryStats = Dict[str, Any]
MetricsHistory = List[DatabaseHealthMetrics]
AlertList = List[HealthAlert]
RecommendationList = List[HealthRecommendation]
ThresholdConfig = Dict[str, HealthThreshold]