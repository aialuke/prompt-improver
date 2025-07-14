"""
Real-time Analytics Service for A/B Testing
Provides live experiment monitoring, metrics calculation, and statistical analysis
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
from scipy import stats
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from ..database.models import ABExperiment, RulePerformance, PatternEvaluation
from ..services.ab_testing import ABTestingService, ExperimentResult
from ..utils.websocket_manager import publish_experiment_update, connection_manager
from ..utils.error_handlers import handle_database_errors

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of real-time alerts"""
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    EARLY_STOPPING_EFFICACY = "early_stopping_efficacy"
    EARLY_STOPPING_FUTILITY = "early_stopping_futility"
    SAMPLE_SIZE_REACHED = "sample_size_reached"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_QUALITY_ISSUE = "data_quality_issue"


@dataclass
class RealTimeMetrics:
    """Real-time metrics for experiment monitoring"""
    experiment_id: str
    timestamp: datetime
    
    # Sample sizes
    control_sample_size: int
    treatment_sample_size: int
    total_sample_size: int
    
    # Primary metrics
    control_mean: float
    treatment_mean: float
    effect_size: float
    
    # Statistical analysis
    p_value: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    statistical_significance: bool
    statistical_power: float
    
    # Progress tracking
    completion_percentage: float
    estimated_days_remaining: Optional[float]
    
    # Quality metrics
    balance_ratio: float  # treatment/control ratio
    data_quality_score: float
    
    # Early stopping signals
    early_stopping_recommendation: Optional[str]
    early_stopping_confidence: Optional[float]


@dataclass
class RealTimeAlert:
    """Real-time alert for experiment events"""
    alert_id: str
    experiment_id: str
    alert_type: AlertType
    severity: str  # "info", "warning", "critical"
    title: str
    message: str
    timestamp: datetime
    data: Dict[str, Any]
    acknowledged: bool = False


class RealTimeAnalyticsService:
    """Service for real-time A/B testing analytics and monitoring"""
    
    def __init__(self, db_session: AsyncSession, redis_client: Optional[redis.Redis] = None):
        self.db_session = db_session
        self.redis_client = redis_client
        self.ab_testing_service = ABTestingService()
        
        # Cache for experiment metrics to detect changes
        self.metrics_cache: Dict[str, RealTimeMetrics] = {}
        
        # Active monitoring tasks
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        
        # Alert tracking
        self.active_alerts: Dict[str, Set[str]] = {}  # experiment_id -> alert_ids
        
    async def start_experiment_monitoring(self, experiment_id: str, 
                                        update_interval: int = 30) -> bool:
        """Start real-time monitoring for an experiment
        
        Args:
            experiment_id: UUID of experiment to monitor
            update_interval: Update interval in seconds
            
        Returns:
            True if monitoring started successfully
        """
        try:
            # Check if experiment exists and is running
            stmt = select(ABExperiment).where(ABExperiment.experiment_id == experiment_id)
            result = await self.db_session.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                logger.error(f"Experiment {experiment_id} not found")
                return False
                
            if experiment.status != "running":
                logger.warning(f"Experiment {experiment_id} is not running (status: {experiment.status})")
                return False
            
            # Cancel existing monitoring task if running
            await self.stop_experiment_monitoring(experiment_id)
            
            # Start monitoring task
            task = asyncio.create_task(
                self._monitoring_loop(experiment_id, update_interval),
                name=f"monitor_{experiment_id}"
            )
            self.monitoring_tasks[experiment_id] = task
            
            logger.info(f"Started real-time monitoring for experiment {experiment_id}")
            
            # Send initial metrics
            await self._calculate_and_broadcast_metrics(experiment_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring for experiment {experiment_id}: {e}")
            return False
    
    async def stop_experiment_monitoring(self, experiment_id: str) -> bool:
        """Stop real-time monitoring for an experiment"""
        try:
            if experiment_id in self.monitoring_tasks:
                task = self.monitoring_tasks[experiment_id]
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                del self.monitoring_tasks[experiment_id]
                logger.info(f"Stopped monitoring for experiment {experiment_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring for experiment {experiment_id}: {e}")
            return False
    
    async def get_real_time_metrics(self, experiment_id: str) -> Optional[RealTimeMetrics]:
        """Get current real-time metrics for an experiment"""
        try:
            return await self._calculate_metrics(experiment_id)
        except Exception as e:
            logger.error(f"Failed to get real-time metrics for experiment {experiment_id}: {e}")
            return None
    
    async def _monitoring_loop(self, experiment_id: str, update_interval: int):
        """Background monitoring loop for experiment"""
        try:
            while True:
                await self._calculate_and_broadcast_metrics(experiment_id)
                await asyncio.sleep(update_interval)
                
        except asyncio.CancelledError:
            logger.info(f"Monitoring cancelled for experiment {experiment_id}")
            raise
        except Exception as e:
            logger.error(f"Monitoring loop error for experiment {experiment_id}: {e}")
            # Try to continue monitoring after error
            await asyncio.sleep(update_interval)
    
    async def _calculate_and_broadcast_metrics(self, experiment_id: str):
        """Calculate metrics and broadcast to WebSocket connections"""
        try:
            # Calculate current metrics
            metrics = await self._calculate_metrics(experiment_id)
            if not metrics:
                return
            
            # Check for alerts
            alerts = await self._check_for_alerts(experiment_id, metrics)
            
            # Broadcast metrics update
            update_data = {
                "type": "metrics_update",
                "experiment_id": experiment_id,
                "metrics": asdict(metrics),
                "alerts": [asdict(alert) for alert in alerts]
            }
            
            await publish_experiment_update(experiment_id, update_data, self.redis_client)
            
            # Cache metrics for change detection
            self.metrics_cache[experiment_id] = metrics
            
            # Process alerts
            for alert in alerts:
                await self._process_alert(alert)
                
        except Exception as e:
            logger.error(f"Error calculating/broadcasting metrics for experiment {experiment_id}: {e}")
    
    @handle_database_errors(rollback_session=False, return_format="none", operation_name="calculate_metrics")
    async def _calculate_metrics(self, experiment_id: str) -> Optional[RealTimeMetrics]:
        """Calculate real-time metrics for experiment"""
        try:
            # Get experiment details
            stmt = select(ABExperiment).where(ABExperiment.experiment_id == experiment_id)
            result = await self.db_session.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                return None
            
            # Get performance data
            control_data = await self._get_experiment_data(
                experiment.control_rules, experiment.started_at, 
                experiment.target_metric
            )
            treatment_data = await self._get_experiment_data(
                experiment.treatment_rules, experiment.started_at, 
                experiment.target_metric
            )
            
            if len(control_data) < 2 or len(treatment_data) < 2:
                # Not enough data for meaningful analysis
                return RealTimeMetrics(
                    experiment_id=experiment_id,
                    timestamp=datetime.utcnow(),
                    control_sample_size=len(control_data),
                    treatment_sample_size=len(treatment_data),
                    total_sample_size=len(control_data) + len(treatment_data),
                    control_mean=np.mean(control_data) if control_data else 0.0,
                    treatment_mean=np.mean(treatment_data) if treatment_data else 0.0,
                    effect_size=0.0,
                    p_value=1.0,
                    confidence_interval_lower=0.0,
                    confidence_interval_upper=0.0,
                    statistical_significance=False,
                    statistical_power=0.0,
                    completion_percentage=0.0,
                    estimated_days_remaining=None,
                    balance_ratio=len(treatment_data) / max(len(control_data), 1),
                    data_quality_score=0.5,
                    early_stopping_recommendation=None,
                    early_stopping_confidence=None
                )
            
            # Perform statistical analysis
            analysis_result = self.ab_testing_service._perform_statistical_analysis(
                control_data, treatment_data
            )
            
            # Calculate progress metrics
            target_size = experiment.sample_size_per_group * 2
            current_size = len(control_data) + len(treatment_data)
            completion_percentage = min(current_size / target_size * 100, 100.0)
            
            # Estimate days remaining based on current daily rate
            days_running = (datetime.utcnow() - experiment.started_at).days
            if days_running > 0:
                daily_rate = current_size / days_running
                if daily_rate > 0:
                    remaining_samples = max(target_size - current_size, 0)
                    estimated_days_remaining = remaining_samples / daily_rate
                else:
                    estimated_days_remaining = None
            else:
                estimated_days_remaining = None
            
            # Calculate data quality score
            balance_ratio = len(treatment_data) / max(len(control_data), 1)
            balance_score = 1.0 - abs(balance_ratio - 1.0)  # Closer to 1.0 is better
            
            # Simple data quality based on balance and sample size adequacy
            sample_adequacy = min(current_size / max(target_size * 0.1, 10), 1.0)
            data_quality_score = (balance_score + sample_adequacy) / 2
            
            # Check for early stopping recommendation
            early_stopping_rec = None
            early_stopping_conf = None
            
            if hasattr(self.ab_testing_service, 'early_stopping_framework') and \
               self.ab_testing_service.early_stopping_framework:
                # Use existing early stopping logic if available
                early_stopping_result = await self.ab_testing_service.check_early_stopping(
                    experiment_id, look_number=1, db_session=self.db_session
                )
                
                if early_stopping_result.get("status") == "success":
                    if early_stopping_result.get("should_stop", False):
                        early_stopping_rec = early_stopping_result.get("early_stopping_analysis", {}).get("recommendation", "Consider stopping")
                        early_stopping_conf = early_stopping_result.get("early_stopping_analysis", {}).get("confidence", 0.0)
            
            return RealTimeMetrics(
                experiment_id=experiment_id,
                timestamp=datetime.utcnow(),
                control_sample_size=len(control_data),
                treatment_sample_size=len(treatment_data),
                total_sample_size=current_size,
                control_mean=analysis_result.control_mean,
                treatment_mean=analysis_result.treatment_mean,
                effect_size=analysis_result.effect_size,
                p_value=analysis_result.p_value,
                confidence_interval_lower=analysis_result.confidence_interval[0],
                confidence_interval_upper=analysis_result.confidence_interval[1],
                statistical_significance=analysis_result.statistical_significance,
                statistical_power=analysis_result.statistical_power,
                completion_percentage=completion_percentage,
                estimated_days_remaining=estimated_days_remaining,
                balance_ratio=balance_ratio,
                data_quality_score=data_quality_score,
                early_stopping_recommendation=early_stopping_rec,
                early_stopping_confidence=early_stopping_conf
            )
            
        except Exception as e:
            logger.error(f"Error calculating metrics for experiment {experiment_id}: {e}")
            return None
    
    async def _get_experiment_data(self, rule_config: Dict[str, Any], 
                                 start_time: datetime, target_metric: str) -> List[float]:
        """Get performance data for experiment group"""
        try:
            rule_ids = rule_config.get("rule_ids", [])
            if not rule_ids:
                return []
            
            stmt = select(RulePerformance).where(
                RulePerformance.rule_id.in_(rule_ids),
                RulePerformance.created_at >= start_time,
            )
            
            result = await self.db_session.execute(stmt)
            performance_records = result.scalars().all()
            
            metric_values = []
            for record in performance_records:
                if target_metric == "improvement_score":
                    metric_values.append(record.improvement_score or 0.0)
                elif target_metric == "execution_time_ms":
                    metric_values.append(record.execution_time_ms or 0.0)
                elif target_metric == "user_satisfaction_score":
                    metric_values.append(record.user_satisfaction_score or 0.0)
            
            return metric_values
            
        except Exception as e:
            logger.error(f"Error getting experiment data: {e}")
            return []
    
    async def _check_for_alerts(self, experiment_id: str, 
                              current_metrics: RealTimeMetrics) -> List[RealTimeAlert]:
        """Check for alert conditions and generate alerts"""
        alerts = []
        
        try:
            # Statistical significance alert
            if current_metrics.statistical_significance:
                if current_metrics.p_value < 0.01:
                    severity = "critical"
                    title = "Highly Significant Result Detected"
                elif current_metrics.p_value < 0.05:
                    severity = "warning"
                    title = "Statistical Significance Achieved"
                
                alerts.append(RealTimeAlert(
                    alert_id=f"{experiment_id}_significance_{int(datetime.utcnow().timestamp())}",
                    experiment_id=experiment_id,
                    alert_type=AlertType.STATISTICAL_SIGNIFICANCE,
                    severity=severity,
                    title=title,
                    message=f"Experiment has reached statistical significance with p-value {current_metrics.p_value:.4f}",
                    timestamp=datetime.utcnow(),
                    data={
                        "p_value": current_metrics.p_value,
                        "effect_size": current_metrics.effect_size,
                        "confidence_interval": [current_metrics.confidence_interval_lower, 
                                              current_metrics.confidence_interval_upper]
                    }
                ))
            
            # Early stopping alerts
            if current_metrics.early_stopping_recommendation:
                alerts.append(RealTimeAlert(
                    alert_id=f"{experiment_id}_early_stop_{int(datetime.utcnow().timestamp())}",
                    experiment_id=experiment_id,
                    alert_type=AlertType.EARLY_STOPPING_EFFICACY,
                    severity="warning",
                    title="Early Stopping Recommended",
                    message=current_metrics.early_stopping_recommendation,
                    timestamp=datetime.utcnow(),
                    data={
                        "confidence": current_metrics.early_stopping_confidence,
                        "recommendation": current_metrics.early_stopping_recommendation
                    }
                ))
            
            # Sample size completion alert
            if current_metrics.completion_percentage >= 100:
                alerts.append(RealTimeAlert(
                    alert_id=f"{experiment_id}_sample_complete_{int(datetime.utcnow().timestamp())}",
                    experiment_id=experiment_id,
                    alert_type=AlertType.SAMPLE_SIZE_REACHED,
                    severity="info",
                    title="Target Sample Size Reached",
                    message="Experiment has reached its target sample size",
                    timestamp=datetime.utcnow(),
                    data={
                        "total_sample_size": current_metrics.total_sample_size,
                        "completion_percentage": current_metrics.completion_percentage
                    }
                ))
            
            # Data quality alerts
            if current_metrics.data_quality_score < 0.7:
                alerts.append(RealTimeAlert(
                    alert_id=f"{experiment_id}_quality_{int(datetime.utcnow().timestamp())}",
                    experiment_id=experiment_id,
                    alert_type=AlertType.DATA_QUALITY_ISSUE,
                    severity="warning" if current_metrics.data_quality_score < 0.5 else "info",
                    title="Data Quality Issue Detected",
                    message=f"Data quality score is {current_metrics.data_quality_score:.2f}",
                    timestamp=datetime.utcnow(),
                    data={
                        "data_quality_score": current_metrics.data_quality_score,
                        "balance_ratio": current_metrics.balance_ratio
                    }
                ))
            
            # Filter out duplicate alerts (same type for same experiment in last hour)
            alerts = await self._filter_duplicate_alerts(experiment_id, alerts)
            
        except Exception as e:
            logger.error(f"Error checking alerts for experiment {experiment_id}: {e}")
        
        return alerts
    
    async def _filter_duplicate_alerts(self, experiment_id: str, 
                                     new_alerts: List[RealTimeAlert]) -> List[RealTimeAlert]:
        """Filter out duplicate alerts to avoid spam"""
        # Simple implementation - could be enhanced with Redis storage
        if experiment_id not in self.active_alerts:
            self.active_alerts[experiment_id] = set()
        
        filtered_alerts = []
        for alert in new_alerts:
            alert_key = f"{alert.alert_type.value}_{alert.severity}"
            if alert_key not in self.active_alerts[experiment_id]:
                filtered_alerts.append(alert)
                self.active_alerts[experiment_id].add(alert_key)
        
        return filtered_alerts
    
    async def _process_alert(self, alert: RealTimeAlert):
        """Process and potentially act on an alert"""
        try:
            # Log alert
            logger.info(f"Alert generated: {alert.title} for experiment {alert.experiment_id}")
            
            # Could add integrations here:
            # - Send email notifications
            # - Create tickets in issue tracking
            # - Trigger automated actions
            # - Store in database for historical tracking
            
        except Exception as e:
            logger.error(f"Error processing alert: {e}")
    
    async def get_active_experiments(self) -> List[str]:
        """Get list of experiments currently being monitored"""
        return list(self.monitoring_tasks.keys())
    
    async def cleanup(self):
        """Clean up all monitoring tasks"""
        for experiment_id in list(self.monitoring_tasks.keys()):
            await self.stop_experiment_monitoring(experiment_id)


# Global service instance
_real_time_service: Optional[RealTimeAnalyticsService] = None


async def get_real_time_analytics_service(db_session: AsyncSession, 
                                        redis_client: Optional[redis.Redis] = None) -> RealTimeAnalyticsService:
    """Get singleton RealTimeAnalyticsService instance"""
    global _real_time_service
    if _real_time_service is None:
        _real_time_service = RealTimeAnalyticsService(db_session, redis_client)
    return _real_time_service