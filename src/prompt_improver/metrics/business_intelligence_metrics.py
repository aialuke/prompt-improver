"""
Business Intelligence Metrics Collectors for Prompt Improver.

Tracks feature adoption rates, user engagement patterns, resource utilization efficiency,
and cost per operation tracking with real-time business insights and ROI analysis.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
import statistics
from collections import defaultdict, deque

from ..performance.monitoring.metrics_registry import get_metrics_registry

class FeatureCategory(Enum):
    """Categories of features for adoption tracking."""
    PROMPT_ENHANCEMENT = "prompt_enhancement"
    ML_ANALYTICS = "ml_analytics"
    REAL_TIME_PROCESSING = "real_time_processing"
    BATCH_PROCESSING = "batch_processing"
    API_INTEGRATION = "api_integration"
    DASHBOARD = "dashboard"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"
    MONITORING = "monitoring"
    ADVANCED_FEATURES = "advanced_features"

class UserTier(Enum):
    """User subscription tiers."""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"

class CostType(Enum):
    """Types of operational costs."""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    ML_INFERENCE = "ml_inference"
    EXTERNAL_API = "external_api"
    DATABASE = "database"
    MONITORING = "monitoring"
    INFRASTRUCTURE = "infrastructure"

class ResourceType(Enum):
    """Types of resources for utilization tracking."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    DATABASE_CONNECTIONS = "database_connections"
    CACHE = "cache"
    QUEUE_CAPACITY = "queue_capacity"

@dataclass
class FeatureAdoptionMetric:
    """Metrics for feature adoption and usage."""
    feature_name: str
    feature_category: FeatureCategory
    user_id: str
    user_tier: UserTier
    session_id: str
    first_use: bool
    usage_count: int
    time_spent_seconds: float
    success: bool
    error_type: Optional[str]
    feature_version: str
    user_cohort: Optional[str]
    onboarding_completed: bool
    conversion_funnel_stage: str
    experiment_variant: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class UserEngagementMetric:
    """Metrics for user engagement patterns."""
    user_id: str
    user_tier: UserTier
    session_id: str
    session_duration_seconds: float
    pages_viewed: int
    features_used: List[str]
    actions_performed: int
    successful_actions: int
    time_to_first_action_seconds: float
    bounce_rate_indicator: bool
    return_user: bool
    days_since_last_visit: Optional[int]
    total_lifetime_value: Optional[float]
    current_streak_days: int
    timestamp: datetime
    user_agent: Optional[str]
    referrer: Optional[str]

@dataclass
class CostTrackingMetric:
    """Metrics for cost per operation tracking."""
    operation_type: str
    cost_type: CostType
    cost_amount: float
    currency: str
    resource_units_consumed: float
    resource_unit_cost: float
    user_id: Optional[str]
    user_tier: Optional[UserTier]
    region: str
    provider: str
    service_name: str
    billing_period: str
    allocation_tags: Dict[str, str]
    timestamp: datetime
    cost_center: Optional[str]
    project_id: Optional[str]

@dataclass
class ResourceUtilizationMetric:
    """Metrics for resource utilization efficiency."""
    resource_type: ResourceType
    resource_name: str
    allocated_capacity: float
    utilized_capacity: float
    peak_utilization: float
    average_utilization: float
    efficiency_score: float
    waste_percentage: float
    cost_per_unit: float
    auto_scaling_triggered: bool
    scaling_direction: Optional[str]  # "up", "down"
    region: str
    availability_zone: Optional[str]
    timestamp: datetime
    forecast_next_hour: Optional[float]
    recommendations: List[str]

class BusinessIntelligenceMetricsCollector:
    """
    Collects and aggregates business intelligence metrics.
    
    Provides real-time tracking of feature adoption, user engagement,
    cost efficiency, and resource utilization with business insights.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize business intelligence metrics collector."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage with size limits
        self.adoption_metrics: deque = deque(maxlen=self.config.get("max_adoption_metrics", 25000))
        self.engagement_metrics: deque = deque(maxlen=self.config.get("max_engagement_metrics", 15000))
        self.cost_metrics: deque = deque(maxlen=self.config.get("max_cost_metrics", 10000))
        self.utilization_metrics: deque = deque(maxlen=self.config.get("max_utilization_metrics", 20000))
        
        # Real-time tracking
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.feature_usage_cache: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_users": set(),
            "total_usage": 0,
            "success_count": 0,
            "first_time_users": set()
        })
        self.cost_accumulator: Dict[str, float] = defaultdict(float)
        
        # Configuration
        self.aggregation_window_minutes = self.config.get("aggregation_window_minutes", 15)
        self.retention_days = self.config.get("retention_days", 30)
        self.roi_calculation_enabled = self.config.get("roi_calculation_enabled", True)
        self.cost_alert_threshold = self.config.get("cost_alert_threshold", 1000.0)
        
        # Collection statistics
        self.collection_stats = {
            "feature_adoptions_tracked": 0,
            "user_engagements_tracked": 0,
            "cost_events_tracked": 0,
            "utilization_events_tracked": 0,
            "active_user_sessions": 0,
            "total_cost_tracked": 0.0,
            "cost_alerts_triggered": 0,
            "last_aggregation": None
        }
        
        # Prometheus metrics integration
        self.metrics_registry = get_metrics_registry()
        self._initialize_prometheus_metrics()
        
        # Background processing
        self.is_running = False
        self.aggregation_task = None
        self.cost_monitoring_task = None
        self.roi_calculation_task = None
    
    def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metrics for business intelligence."""
        # Feature adoption metrics
        self.feature_adoption_rate = self.metrics_registry.get_or_create_gauge(
            "bi_feature_adoption_rate",
            "Feature adoption rate by category and tier",
            ["feature_category", "user_tier", "time_window"]
        )
        
        self.feature_usage_count = self.metrics_registry.get_or_create_counter(
            "bi_feature_usage_total",
            "Total feature usage count",
            ["feature_name", "user_tier", "success"]
        )
        
        self.feature_time_spent = self.metrics_registry.get_or_create_histogram(
            "bi_feature_time_spent_seconds",
            "Time spent using features",
            ["feature_name", "user_tier"],
            buckets=[1, 5, 15, 30, 60, 180, 300, 600, 1800, 3600]
        )
        
        self.first_time_feature_usage = self.metrics_registry.get_or_create_counter(
            "bi_first_time_feature_usage_total",
            "First time feature usage by users",
            ["feature_name", "user_tier"]
        )
        
        # User engagement metrics
        self.user_session_duration = self.metrics_registry.get_or_create_histogram(
            "bi_user_session_duration_seconds",
            "User session duration distribution",
            ["user_tier"],
            buckets=[60, 300, 600, 1200, 1800, 3600, 7200, 14400, 28800]
        )
        
        self.user_actions_per_session = self.metrics_registry.get_or_create_histogram(
            "bi_user_actions_per_session",
            "Number of actions per user session",
            ["user_tier"],
            buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500]
        )
        
        self.user_engagement_score = self.metrics_registry.get_or_create_gauge(
            "bi_user_engagement_score",
            "User engagement score (0-100)",
            ["user_tier", "time_window"]
        )
        
        self.user_retention_rate = self.metrics_registry.get_or_create_gauge(
            "bi_user_retention_rate",
            "User retention rate by time period",
            ["user_tier", "period"]  # period: "daily", "weekly", "monthly"
        )
        
        # Cost tracking metrics
        self.operational_cost_per_hour = self.metrics_registry.get_or_create_gauge(
            "bi_operational_cost_per_hour",
            "Operational cost per hour by type",
            ["cost_type", "region", "currency"]
        )
        
        self.cost_per_user = self.metrics_registry.get_or_create_gauge(
            "bi_cost_per_user",
            "Average cost per user by tier",
            ["user_tier", "time_window", "currency"]
        )
        
        self.cost_efficiency_ratio = self.metrics_registry.get_or_create_gauge(
            "bi_cost_efficiency_ratio",
            "Cost efficiency ratio (revenue/cost)",
            ["cost_type", "time_window"]
        )
        
        self.budget_utilization = self.metrics_registry.get_or_create_gauge(
            "bi_budget_utilization_percent",
            "Budget utilization percentage",
            ["cost_center", "billing_period"]
        )
        
        # Resource utilization metrics
        self.resource_utilization_efficiency = self.metrics_registry.get_or_create_gauge(
            "bi_resource_utilization_efficiency",
            "Resource utilization efficiency score",
            ["resource_type", "region"]
        )
        
        self.resource_waste_percentage = self.metrics_registry.get_or_create_gauge(
            "bi_resource_waste_percentage",
            "Percentage of wasted resources",
            ["resource_type", "region"]
        )
        
        self.auto_scaling_events = self.metrics_registry.get_or_create_counter(
            "bi_auto_scaling_events_total",
            "Total auto-scaling events",
            ["resource_type", "direction", "region"]
        )
        
        # ROI and business value metrics
        self.feature_roi = self.metrics_registry.get_or_create_gauge(
            "bi_feature_roi_ratio",
            "Return on investment for features",
            ["feature_category", "time_window"]
        )
        
        self.user_lifetime_value = self.metrics_registry.get_or_create_gauge(
            "bi_user_lifetime_value",
            "Average user lifetime value",
            ["user_tier", "currency"]
        )
    
    async def start_collection(self):
        """Start background metrics collection and processing."""
        if self.is_running:
            return
        
        self.is_running = True
        self.aggregation_task = asyncio.create_task(self._aggregation_loop())
        self.cost_monitoring_task = asyncio.create_task(self._cost_monitoring_loop())
        
        if self.roi_calculation_enabled:
            self.roi_calculation_task = asyncio.create_task(self._roi_calculation_loop())
        
        self.logger.info("Started business intelligence metrics collection")
    
    async def stop_collection(self):
        """Stop background metrics collection and processing."""
        if not self.is_running:
            return
        
        self.is_running = False
        for task in [self.aggregation_task, self.cost_monitoring_task, self.roi_calculation_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info("Stopped business intelligence metrics collection")
    
    async def _aggregation_loop(self):
        """Background aggregation of metrics."""
        try:
            while self.is_running:
                await self._aggregate_metrics()
                await asyncio.sleep(self.aggregation_window_minutes * 60)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in BI metrics aggregation: {e}")
    
    async def _cost_monitoring_loop(self):
        """Background cost monitoring and alerting."""
        try:
            while self.is_running:
                await self._monitor_costs()
                await asyncio.sleep(300)  # Check costs every 5 minutes
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in cost monitoring: {e}")
    
    async def _roi_calculation_loop(self):
        """Background ROI calculation."""
        try:
            while self.is_running:
                await self._calculate_roi_metrics()
                await asyncio.sleep(3600)  # Calculate ROI every hour
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in ROI calculation: {e}")
    
    async def record_feature_adoption(self, metric: FeatureAdoptionMetric):
        """Record a feature adoption metric."""
        try:
            self.adoption_metrics.append(metric)
            self.collection_stats["feature_adoptions_tracked"] += 1
            
            # Update feature usage cache
            feature_key = f"{metric.feature_category.value}:{metric.feature_name}"
            cache = self.feature_usage_cache[feature_key]
            cache["total_users"].add(metric.user_id)
            cache["total_usage"] += metric.usage_count
            if metric.success:
                cache["success_count"] += 1
            if metric.first_use:
                cache["first_time_users"].add(metric.user_id)
            
            # Update Prometheus metrics
            success_label = "true" if metric.success else "false"
            
            self.feature_usage_count.labels(
                feature_name=metric.feature_name,
                user_tier=metric.user_tier.value,
                success=success_label
            ).inc(metric.usage_count)
            
            self.feature_time_spent.labels(
                feature_name=metric.feature_name,
                user_tier=metric.user_tier.value
            ).observe(metric.time_spent_seconds)
            
            if metric.first_use:
                self.first_time_feature_usage.labels(
                    feature_name=metric.feature_name,
                    user_tier=metric.user_tier.value
                ).inc()
            
            self.logger.debug(f"Recorded feature adoption: {metric.feature_name} by {metric.user_tier.value}")
        
        except Exception as e:
            self.logger.error(f"Error recording feature adoption metric: {e}")
    
    async def record_user_engagement(self, metric: UserEngagementMetric):
        """Record a user engagement metric."""
        try:
            self.engagement_metrics.append(metric)
            self.collection_stats["user_engagements_tracked"] += 1
            
            # Update user session tracking
            self.user_sessions[metric.session_id] = {
                "user_id": metric.user_id,
                "user_tier": metric.user_tier,
                "start_time": metric.timestamp,
                "duration": metric.session_duration_seconds,
                "features_used": metric.features_used,
                "actions": metric.actions_performed
            }
            
            # Update Prometheus metrics
            self.user_session_duration.labels(
                user_tier=metric.user_tier.value
            ).observe(metric.session_duration_seconds)
            
            self.user_actions_per_session.labels(
                user_tier=metric.user_tier.value
            ).observe(metric.actions_performed)
            
            # Calculate engagement score (0-100)
            engagement_score = self._calculate_engagement_score(metric)
            self.user_engagement_score.labels(
                user_tier=metric.user_tier.value,
                time_window="realtime"
            ).set(engagement_score)
            
            self.logger.debug(f"Recorded user engagement: {metric.user_id} ({metric.user_tier.value})")
        
        except Exception as e:
            self.logger.error(f"Error recording user engagement metric: {e}")
    
    async def record_cost_tracking(self, metric: CostTrackingMetric):
        """Record a cost tracking metric."""
        try:
            self.cost_metrics.append(metric)
            self.collection_stats["cost_events_tracked"] += 1
            self.collection_stats["total_cost_tracked"] += metric.cost_amount
            
            # Update cost accumulator for monitoring
            cost_key = f"{metric.cost_type.value}:{metric.region}"
            self.cost_accumulator[cost_key] += metric.cost_amount
            
            # Update Prometheus metrics
            self.operational_cost_per_hour.labels(
                cost_type=metric.cost_type.value,
                region=metric.region,
                currency=metric.currency
            ).set(metric.cost_amount)
            
            self.logger.debug(f"Recorded cost: {metric.cost_amount} {metric.currency} for {metric.cost_type.value}")
        
        except Exception as e:
            self.logger.error(f"Error recording cost tracking metric: {e}")
    
    async def record_resource_utilization(self, metric: ResourceUtilizationMetric):
        """Record a resource utilization metric."""
        try:
            self.utilization_metrics.append(metric)
            self.collection_stats["utilization_events_tracked"] += 1
            
            # Update Prometheus metrics
            self.resource_utilization_efficiency.labels(
                resource_type=metric.resource_type.value,
                region=metric.region
            ).set(metric.efficiency_score)
            
            self.resource_waste_percentage.labels(
                resource_type=metric.resource_type.value,
                region=metric.region
            ).set(metric.waste_percentage)
            
            if metric.auto_scaling_triggered and metric.scaling_direction:
                self.auto_scaling_events.labels(
                    resource_type=metric.resource_type.value,
                    direction=metric.scaling_direction,
                    region=metric.region
                ).inc()
            
            self.logger.debug(f"Recorded resource utilization: {metric.resource_type.value} in {metric.region}")
        
        except Exception as e:
            self.logger.error(f"Error recording resource utilization metric: {e}")
    
    def _calculate_engagement_score(self, metric: UserEngagementMetric) -> float:
        """Calculate user engagement score (0-100)."""
        try:
            # Base score components
            session_score = min(metric.session_duration_seconds / 1800, 1.0) * 30  # Max 30 points for 30+ min sessions
            action_score = min(metric.actions_performed / 20, 1.0) * 25  # Max 25 points for 20+ actions
            success_score = (metric.successful_actions / max(metric.actions_performed, 1)) * 20  # Max 20 points for 100% success
            feature_score = min(len(metric.features_used) / 5, 1.0) * 15  # Max 15 points for 5+ features
            speed_score = max(0, 10 - (metric.time_to_first_action_seconds / 30)) if metric.time_to_first_action_seconds <= 300 else 0  # Max 10 points for quick start
            
            # Bonus points for return users and streaks
            bonus_score = 0
            if metric.return_user:
                bonus_score += 5
            if metric.current_streak_days > 0:
                bonus_score += min(metric.current_streak_days, 5)  # Max 5 points for streaks
            
            total_score = session_score + action_score + success_score + feature_score + speed_score + bonus_score
            return min(total_score, 100.0)
        
        except Exception as e:
            self.logger.error(f"Error calculating engagement score: {e}")
            return 50.0  # Default middle score
    
    async def _aggregate_metrics(self):
        """Aggregate metrics over time windows."""
        try:
            current_time = datetime.now(timezone.utc)
            window_start = current_time - timedelta(minutes=self.aggregation_window_minutes)
            
            # Aggregate feature adoption metrics
            await self._aggregate_adoption_metrics(window_start, current_time)
            
            # Aggregate user engagement metrics
            await self._aggregate_engagement_metrics(window_start, current_time)
            
            # Aggregate cost metrics
            await self._aggregate_cost_metrics(window_start, current_time)
            
            # Aggregate resource utilization metrics
            await self._aggregate_utilization_metrics(window_start, current_time)
            
            # Clean up old metrics
            await self._cleanup_old_metrics(current_time)
            
            self.collection_stats["last_aggregation"] = current_time
            self.collection_stats["active_user_sessions"] = len(self.user_sessions)
            
        except Exception as e:
            self.logger.error(f"Error in BI metrics aggregation: {e}")
    
    async def _aggregate_adoption_metrics(self, window_start: datetime, window_end: datetime):
        """Aggregate feature adoption metrics."""
        window_metrics = [
            m for m in self.adoption_metrics
            if window_start <= m.timestamp <= window_end
        ]
        
        if not window_metrics:
            return
        
        # Group by feature category and user tier
        category_tier_groups = defaultdict(lambda: {
            "total_users": set(),
            "total_usage": 0,
            "first_time_users": set()
        })
        
        for metric in window_metrics:
            key = (metric.feature_category, metric.user_tier)
            group = category_tier_groups[key]
            group["total_users"].add(metric.user_id)
            group["total_usage"] += metric.usage_count
            if metric.first_use:
                group["first_time_users"].add(metric.user_id)
        
        # Calculate adoption rates
        for (category, tier), group in category_tier_groups.items():
            total_users = len(group["total_users"])
            first_time_users = len(group["first_time_users"])
            
            # Adoption rate = new users / total users for this window
            adoption_rate = first_time_users / total_users if total_users > 0 else 0
            
            self.feature_adoption_rate.labels(
                feature_category=category.value,
                user_tier=tier.value,
                time_window=f"{self.aggregation_window_minutes}m"
            ).set(adoption_rate)
        
        self.logger.debug(f"Aggregated adoption metrics: {len(window_metrics)} events")
    
    async def _aggregate_engagement_metrics(self, window_start: datetime, window_end: datetime):
        """Aggregate user engagement metrics."""
        window_metrics = [
            m for m in self.engagement_metrics
            if window_start <= m.timestamp <= window_end
        ]
        
        if not window_metrics:
            return
        
        # Group by user tier
        tier_groups = defaultdict(list)
        for metric in window_metrics:
            tier_groups[metric.user_tier].append(metric)
        
        # Calculate engagement scores
        for tier, metrics in tier_groups.items():
            if not metrics:
                continue
            
            engagement_scores = [self._calculate_engagement_score(m) for m in metrics]
            avg_engagement = statistics.mean(engagement_scores)
            
            self.user_engagement_score.labels(
                user_tier=tier.value,
                time_window=f"{self.aggregation_window_minutes}m"
            ).set(avg_engagement)
        
        self.logger.debug(f"Aggregated engagement metrics: {len(window_metrics)} sessions")
    
    async def _aggregate_cost_metrics(self, window_start: datetime, window_end: datetime):
        """Aggregate cost metrics."""
        window_metrics = [
            m for m in self.cost_metrics
            if window_start <= m.timestamp <= window_end
        ]
        
        if not window_metrics:
            return
        
        # Calculate hourly cost rates
        window_duration_hours = (window_end - window_start).total_seconds() / 3600
        
        cost_groups = defaultdict(float)
        for metric in window_metrics:
            key = (metric.cost_type, metric.region, metric.currency)
            cost_groups[key] += metric.cost_amount
        
        # Update hourly cost metrics
        for (cost_type, region, currency), total_cost in cost_groups.items():
            cost_per_hour = total_cost / window_duration_hours if window_duration_hours > 0 else 0
            
            self.operational_cost_per_hour.labels(
                cost_type=cost_type.value,
                region=region,
                currency=currency
            ).set(cost_per_hour)
        
        self.logger.debug(f"Aggregated cost metrics: {len(window_metrics)} cost events")
    
    async def _aggregate_utilization_metrics(self, window_start: datetime, window_end: datetime):
        """Aggregate resource utilization metrics."""
        window_metrics = [
            m for m in self.utilization_metrics
            if window_start <= m.timestamp <= window_end
        ]
        
        if not window_metrics:
            return
        
        # Group by resource type and region
        resource_groups = defaultdict(list)
        for metric in window_metrics:
            key = (metric.resource_type, metric.region)
            resource_groups[key].append(metric)
        
        # Calculate average efficiency and waste
        for (resource_type, region), metrics in resource_groups.items():
            if not metrics:
                continue
            
            avg_efficiency = statistics.mean(m.efficiency_score for m in metrics)
            avg_waste = statistics.mean(m.waste_percentage for m in metrics)
            
            self.resource_utilization_efficiency.labels(
                resource_type=resource_type.value,
                region=region
            ).set(avg_efficiency)
            
            self.resource_waste_percentage.labels(
                resource_type=resource_type.value,
                region=region
            ).set(avg_waste)
        
        self.logger.debug(f"Aggregated utilization metrics: {len(window_metrics)} resource events")
    
    async def _monitor_costs(self):
        """Monitor costs and trigger alerts if thresholds are exceeded."""
        try:
            # Calculate current hourly cost rate
            current_hour_cost = sum(self.cost_accumulator.values())
            
            if current_hour_cost > self.cost_alert_threshold:
                self.collection_stats["cost_alerts_triggered"] += 1
                self.logger.warning(
                    f"Cost alert: Current hourly cost ({current_hour_cost:.2f}) "
                    f"exceeds threshold ({self.cost_alert_threshold:.2f})"
                )
            
            # Reset accumulator for next period
            self.cost_accumulator.clear()
            
        except Exception as e:
            self.logger.error(f"Error in cost monitoring: {e}")
    
    async def _calculate_roi_metrics(self):
        """Calculate ROI metrics for features and user tiers."""
        try:
            # This is a simplified ROI calculation
            # In practice, you'd integrate with revenue and cost data
            
            current_time = datetime.now(timezone.utc)
            analysis_window = current_time - timedelta(hours=24)
            
            # Calculate feature ROI based on usage and estimated value
            for feature_key, cache in self.feature_usage_cache.items():
                if not cache["total_users"]:
                    continue
                
                category, feature_name = feature_key.split(":", 1)
                
                # Simplified ROI calculation
                user_count = len(cache["total_users"])
                success_rate = cache["success_count"] / cache["total_usage"] if cache["total_usage"] > 0 else 0
                
                # Estimated value based on user engagement
                estimated_value = user_count * success_rate * 10  # $10 per successful user interaction
                estimated_cost = user_count * 0.50  # $0.50 per user cost
                
                roi_ratio = estimated_value / estimated_cost if estimated_cost > 0 else 0
                
                self.feature_roi.labels(
                    feature_category=category,
                    time_window="24h"
                ).set(roi_ratio)
            
            self.logger.debug("Calculated ROI metrics")
            
        except Exception as e:
            self.logger.error(f"Error calculating ROI metrics: {e}")
    
    async def _cleanup_old_metrics(self, current_time: datetime):
        """Clean up metrics older than retention period."""
        cutoff_time = current_time - timedelta(days=self.retention_days)
        
        # Clean adoption metrics
        original_adoption_count = len(self.adoption_metrics)
        self.adoption_metrics = deque(
            (m for m in self.adoption_metrics if m.timestamp > cutoff_time),
            maxlen=self.adoption_metrics.maxlen
        )
        
        # Clean engagement metrics
        original_engagement_count = len(self.engagement_metrics)
        self.engagement_metrics = deque(
            (m for m in self.engagement_metrics if m.timestamp > cutoff_time),
            maxlen=self.engagement_metrics.maxlen
        )
        
        # Clean cost metrics
        original_cost_count = len(self.cost_metrics)
        self.cost_metrics = deque(
            (m for m in self.cost_metrics if m.timestamp > cutoff_time),
            maxlen=self.cost_metrics.maxlen
        )
        
        # Clean utilization metrics
        original_util_count = len(self.utilization_metrics)
        self.utilization_metrics = deque(
            (m for m in self.utilization_metrics if m.timestamp > cutoff_time),
            maxlen=self.utilization_metrics.maxlen
        )
        
        cleaned_total = (
            (original_adoption_count - len(self.adoption_metrics)) +
            (original_engagement_count - len(self.engagement_metrics)) +
            (original_cost_count - len(self.cost_metrics)) +
            (original_util_count - len(self.utilization_metrics))
        )
        
        if cleaned_total > 0:
            self.logger.debug(f"Cleaned up {cleaned_total} old BI metrics")
    
    async def get_feature_adoption_report(self, days: int = 7) -> Dict[str, Any]:
        """Get feature adoption analysis report."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        recent_metrics = [m for m in self.adoption_metrics if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {"status": "no_data", "days": days}
        
        # Analyze feature adoption by category
        category_stats = defaultdict(lambda: {
            "total_users": set(),
            "first_time_users": set(),
            "total_usage": 0,
            "success_rate": 0,
            "features": defaultdict(lambda: {"users": set(), "usage": 0})
        })
        
        for metric in recent_metrics:
            stats = category_stats[metric.feature_category]
            stats["total_users"].add(metric.user_id)
            stats["total_usage"] += metric.usage_count
            
            if metric.first_use:
                stats["first_time_users"].add(metric.user_id)
            
            feature_stats = stats["features"][metric.feature_name]
            feature_stats["users"].add(metric.user_id)
            feature_stats["usage"] += metric.usage_count
        
        # Calculate adoption metrics
        adoption_report = {}
        for category, stats in category_stats.items():
            total_users = len(stats["total_users"])
            new_users = len(stats["first_time_users"])
            
            # Top features in this category
            top_features = sorted(
                stats["features"].items(),
                key=lambda x: len(x[1]["users"]),
                reverse=True
            )[:5]
            
            adoption_report[category.value] = {
                "total_users": total_users,
                "new_users": new_users,
                "adoption_rate": new_users / total_users if total_users > 0 else 0,
                "total_usage": stats["total_usage"],
                "avg_usage_per_user": stats["total_usage"] / total_users if total_users > 0 else 0,
                "top_features": [
                    {
                        "name": name,
                        "users": len(data["users"]),
                        "usage": data["usage"]
                    }
                    for name, data in top_features
                ]
            }
        
        return {
            "total_adoption_events": len(recent_metrics),
            "category_analysis": adoption_report,
            "time_window_days": days,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def get_cost_efficiency_report(self, days: int = 7) -> Dict[str, Any]:
        """Get cost efficiency analysis report."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        recent_metrics = [m for m in self.cost_metrics if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {"status": "no_data", "days": days}
        
        # Analyze costs by type and region
        cost_analysis = defaultdict(lambda: {
            "total_cost": 0,
            "resource_units": 0,
            "regions": defaultdict(float)
        })
        
        total_cost = 0
        for metric in recent_metrics:
            analysis = cost_analysis[metric.cost_type]
            analysis["total_cost"] += metric.cost_amount
            analysis["resource_units"] += metric.resource_units_consumed
            analysis["regions"][metric.region] += metric.cost_amount
            total_cost += metric.cost_amount
        
        # Calculate efficiency metrics
        efficiency_report = {}
        for cost_type, analysis in cost_analysis.items():
            cost_per_unit = (
                analysis["total_cost"] / analysis["resource_units"]
                if analysis["resource_units"] > 0 else 0
            )
            
            efficiency_report[cost_type.value] = {
                "total_cost": analysis["total_cost"],
                "resource_units": analysis["resource_units"],
                "cost_per_unit": cost_per_unit,
                "percentage_of_total": (analysis["total_cost"] / total_cost * 100) if total_cost > 0 else 0,
                "regional_breakdown": dict(analysis["regions"])
            }
        
        return {
            "total_cost": total_cost,
            "cost_breakdown": efficiency_report,
            "daily_average": total_cost / days if days > 0 else 0,
            "time_window_days": days,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get current collection statistics."""
        return {
            **self.collection_stats,
            "current_metrics_count": {
                "feature_adoptions": len(self.adoption_metrics),
                "user_engagements": len(self.engagement_metrics),
                "cost_trackings": len(self.cost_metrics),
                "resource_utilizations": len(self.utilization_metrics)
            },
            "feature_cache_size": len(self.feature_usage_cache),
            "cost_accumulator_size": len(self.cost_accumulator),
            "is_running": self.is_running,
            "config": {
                "aggregation_window_minutes": self.aggregation_window_minutes,
                "retention_days": self.retention_days,
                "roi_calculation_enabled": self.roi_calculation_enabled,
                "cost_alert_threshold": self.cost_alert_threshold
            }
        }

# Global instance
_bi_metrics_collector: Optional[BusinessIntelligenceMetricsCollector] = None

def get_bi_metrics_collector(config: Optional[Dict[str, Any]] = None) -> BusinessIntelligenceMetricsCollector:
    """Get global business intelligence metrics collector instance."""
    global _bi_metrics_collector
    if _bi_metrics_collector is None:
        _bi_metrics_collector = BusinessIntelligenceMetricsCollector(config)
    return _bi_metrics_collector

# Convenience functions for recording metrics
async def record_feature_usage(
    feature_name: str,
    feature_category: FeatureCategory,
    user_id: str,
    user_tier: UserTier,
    session_id: str,
    first_use: bool = False,
    usage_count: int = 1,
    time_spent_seconds: float = 0,
    success: bool = True,
    error_type: Optional[str] = None,
    feature_version: str = "1.0",
    user_cohort: Optional[str] = None,
    onboarding_completed: bool = False,
    conversion_funnel_stage: str = "usage",
    experiment_variant: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Record feature usage (convenience function)."""
    collector = get_bi_metrics_collector()
    metric = FeatureAdoptionMetric(
        feature_name=feature_name,
        feature_category=feature_category,
        user_id=user_id,
        user_tier=user_tier,
        session_id=session_id,
        first_use=first_use,
        usage_count=usage_count,
        time_spent_seconds=time_spent_seconds,
        success=success,
        error_type=error_type,
        feature_version=feature_version,
        user_cohort=user_cohort,
        onboarding_completed=onboarding_completed,
        conversion_funnel_stage=conversion_funnel_stage,
        experiment_variant=experiment_variant,
        timestamp=datetime.now(timezone.utc),
        metadata=metadata or {}
    )
    await collector.record_feature_adoption(metric)

async def record_operational_cost(
    operation_type: str,
    cost_type: CostType,
    cost_amount: float,
    currency: str = "USD",
    resource_units_consumed: float = 1.0,
    resource_unit_cost: Optional[float] = None,
    user_id: Optional[str] = None,
    user_tier: Optional[UserTier] = None,
    region: str = "us-east-1",
    provider: str = "aws",
    service_name: str = "unknown",
    billing_period: str = "hourly",
    allocation_tags: Optional[Dict[str, str]] = None,
    cost_center: Optional[str] = None,
    project_id: Optional[str] = None
):
    """Record operational cost (convenience function)."""
    collector = get_bi_metrics_collector()
    metric = CostTrackingMetric(
        operation_type=operation_type,
        cost_type=cost_type,
        cost_amount=cost_amount,
        currency=currency,
        resource_units_consumed=resource_units_consumed,
        resource_unit_cost=resource_unit_cost or (cost_amount / resource_units_consumed if resource_units_consumed > 0 else 0),
        user_id=user_id,
        user_tier=user_tier,
        region=region,
        provider=provider,
        service_name=service_name,
        billing_period=billing_period,
        allocation_tags=allocation_tags or {},
        timestamp=datetime.now(timezone.utc),
        cost_center=cost_center,
        project_id=project_id
    )
    await collector.record_cost_tracking(metric)