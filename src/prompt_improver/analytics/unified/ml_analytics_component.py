"""ML Analytics Component for Unified Analytics

This component handles all ML model performance analytics, drift detection,
and predictive analytics with comprehensive monitoring capabilities.
"""

import asyncio
import logging
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, deque
from enum import Enum

import numpy as np
from pydantic import BaseModel
from scipy import stats
from sqlalchemy.ext.asyncio import AsyncSession

from .protocols import (
    AnalyticsComponentProtocol,
    ComponentHealth,
    MLAnalyticsProtocol,
    MLModelMetrics,
)

logger = logging.getLogger(__name__)


class DriftSeverity(Enum):
    """Model drift severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelHealthStatus(Enum):
    """Model health status values"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class DriftDetectionResult(BaseModel):
    """Result of drift detection analysis"""
    model_id: str
    drift_detected: bool
    drift_severity: DriftSeverity
    drift_score: float
    drift_metrics: Dict[str, float]
    timestamp: datetime
    recommendations: List[str]


class ModelComparisonResult(BaseModel):
    """Result of model comparison"""
    model_ids: List[str]
    champion_model: str
    performance_rankings: List[Tuple[str, float]]
    statistical_significance: bool
    recommendations: List[str]
    detailed_comparison: Dict[str, Dict[str, float]]


class ModelIssue(BaseModel):
    """Predicted model issue"""
    issue_type: str
    severity: str
    probability: float
    description: str
    recommended_actions: List[str]
    estimated_impact: str


class MLAnalyticsComponent(MLAnalyticsProtocol, AnalyticsComponentProtocol):
    """
    ML Analytics Component implementing comprehensive ML model monitoring.
    
    Features:
    - Real-time model performance tracking with drift detection
    - Advanced statistical model comparison and ranking
    - Predictive analytics for model health and issues
    - Automated retraining recommendations
    - Performance regression detection
    - Model lifecycle analytics and optimization insights
    """
    
    def __init__(self, db_session: AsyncSession, config: Dict[str, Any]):
        self.db_session = db_session
        self.config = config
        self.logger = logger
        
        # Model metrics storage with sliding windows
        self._metrics_window_size = config.get("metrics_window_size", 1000)
        self._model_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._metrics_window_size)
        )
        
        # Drift detection configuration
        self._drift_detection_enabled = config.get("drift_detection", True)
        self._drift_threshold_low = config.get("drift_threshold_low", 0.05)
        self._drift_threshold_medium = config.get("drift_threshold_medium", 0.1)
        self._drift_threshold_high = config.get("drift_threshold_high", 0.2)
        self._baseline_window_size = config.get("baseline_window_size", 100)
        
        # Model health thresholds
        self._accuracy_threshold = config.get("accuracy_threshold", 0.8)
        self._inference_time_threshold_ms = config.get("inference_time_threshold", 1000)
        self._f1_threshold = config.get("f1_threshold", 0.7)
        
        # Performance tracking
        self._stats = {
            "models_tracked": 0,
            "metrics_processed": 0,
            "drift_detections": 0,
            "comparisons_performed": 0,
            "issues_predicted": 0,
        }
        
        # Model baselines for drift detection
        self._model_baselines: Dict[str, Dict[str, Any]] = {}
        
        # Background monitoring
        self._monitoring_enabled = True
        self._monitoring_interval = config.get("monitoring_interval", 300)  # 5 minutes
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Active model issues tracking
        self._active_issues: Dict[str, List[ModelIssue]] = defaultdict(list)
        
        # Start background monitoring
        self._start_monitoring()
    
    async def track_model_performance(self, metrics: MLModelMetrics) -> bool:
        """
        Track ML model performance metrics with real-time analysis.
        
        Args:
            metrics: ML model performance metrics
            
        Returns:
            Success status
        """
        try:
            model_id = metrics.model_id
            
            # Store metrics
            metric_data = {
                "model_id": model_id,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "inference_time_ms": metrics.inference_time_ms,
                "timestamp": metrics.timestamp,
            }
            
            self._model_metrics[model_id].append(metric_data)
            
            # Update baseline if this is a new model
            if model_id not in self._model_baselines:
                await self._initialize_model_baseline(model_id)
            
            # Update statistics
            self._stats["metrics_processed"] += 1
            if model_id not in [m["model_id"] for m in self._get_recent_metrics(model_id, 1)]:
                self._stats["models_tracked"] += 1
            
            # Real-time drift detection
            if self._drift_detection_enabled and len(self._model_metrics[model_id]) >= 10:
                await self._check_model_drift(model_id)
            
            # Real-time health check
            await self._check_model_health(model_id, metric_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error tracking performance for model {metrics.model_id}: {e}")
            return False
    
    async def analyze_model_drift(
        self,
        model_id: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Analyze model performance drift over time window.
        
        Args:
            model_id: Model identifier
            time_window_hours: Time window for drift analysis
            
        Returns:
            Drift analysis results
        """
        try:
            if model_id not in self._model_metrics:
                return {"error": f"No metrics found for model {model_id}"}
            
            # Get baseline and recent metrics
            baseline_metrics = self._get_baseline_metrics(model_id)
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=time_window_hours)
            
            recent_metrics = [
                m for m in self._model_metrics[model_id]
                if m["timestamp"] >= cutoff_time
            ]
            
            if not recent_metrics or not baseline_metrics:
                return {
                    "error": "Insufficient data for drift analysis",
                    "recent_metrics_count": len(recent_metrics),
                    "baseline_available": bool(baseline_metrics)
                }
            
            # Calculate drift for each metric
            drift_results = {}
            
            metrics_to_analyze = ["accuracy", "precision", "recall", "f1_score", "inference_time_ms"]
            
            for metric_name in metrics_to_analyze:
                if metric_name in baseline_metrics:
                    drift_score = await self._calculate_drift_score(
                        model_id, metric_name, baseline_metrics[metric_name], recent_metrics
                    )
                    drift_results[metric_name] = drift_score
            
            # Determine overall drift severity
            max_drift_score = max(drift_results.values()) if drift_results else 0.0
            drift_severity = self._classify_drift_severity(max_drift_score)
            
            # Generate recommendations
            recommendations = await self._generate_drift_recommendations(
                model_id, drift_results, drift_severity
            )
            
            drift_result = DriftDetectionResult(
                model_id=model_id,
                drift_detected=max_drift_score > self._drift_threshold_low,
                drift_severity=drift_severity,
                drift_score=max_drift_score,
                drift_metrics=drift_results,
                timestamp=current_time,
                recommendations=recommendations
            )
            
            self._stats["drift_detections"] += 1
            
            return drift_result.dict()
            
        except Exception as e:
            self.logger.error(f"Error analyzing drift for model {model_id}: {e}")
            return {"error": str(e)}
    
    async def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple ML models across performance metrics.
        
        Args:
            model_ids: List of model IDs to compare
            
        Returns:
            Model comparison results
        """
        try:
            if len(model_ids) < 2:
                return {"error": "At least 2 models required for comparison"}
            
            # Get recent metrics for all models
            model_performances = {}
            
            for model_id in model_ids:
                if model_id not in self._model_metrics:
                    self.logger.warning(f"No metrics found for model {model_id}")
                    continue
                
                recent_metrics = self._get_recent_metrics(model_id, 50)  # Last 50 data points
                
                if recent_metrics:
                    # Calculate average performance
                    avg_performance = {
                        "accuracy": np.mean([m["accuracy"] for m in recent_metrics]),
                        "precision": np.mean([m["precision"] for m in recent_metrics]),
                        "recall": np.mean([m["recall"] for m in recent_metrics]),
                        "f1_score": np.mean([m["f1_score"] for m in recent_metrics]),
                        "inference_time_ms": np.mean([m["inference_time_ms"] for m in recent_metrics]),
                        "sample_count": len(recent_metrics)
                    }
                    model_performances[model_id] = avg_performance
            
            if len(model_performances) < 2:
                return {"error": "Insufficient data for model comparison"}
            
            # Calculate composite performance scores
            performance_scores = {}
            for model_id, perf in model_performances.items():
                # Weighted composite score (lower inference time is better)
                composite_score = (
                    perf["accuracy"] * 0.3 +
                    perf["precision"] * 0.2 +
                    perf["recall"] * 0.2 +
                    perf["f1_score"] * 0.25 +
                    max(0, 1 - perf["inference_time_ms"] / 2000) * 0.05  # Normalize inference time
                )
                performance_scores[model_id] = composite_score
            
            # Rank models by performance
            performance_rankings = sorted(
                performance_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            champion_model = performance_rankings[0][0]
            
            # Statistical significance testing (simplified)
            statistical_significance = await self._test_model_significance(
                model_performances, champion_model
            )
            
            # Generate recommendations
            recommendations = await self._generate_comparison_recommendations(
                model_performances, performance_rankings
            )
            
            comparison_result = ModelComparisonResult(
                model_ids=list(model_performances.keys()),
                champion_model=champion_model,
                performance_rankings=performance_rankings,
                statistical_significance=statistical_significance,
                recommendations=recommendations,
                detailed_comparison=model_performances
            )
            
            self._stats["comparisons_performed"] += 1
            
            return comparison_result.dict()
            
        except Exception as e:
            self.logger.error(f"Error comparing models {model_ids}: {e}")
            return {"error": str(e)}
    
    async def predict_model_issues(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Predict potential issues for a model using analytics.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of predicted issues
        """
        try:
            if model_id not in self._model_metrics:
                return [{"error": f"No metrics found for model {model_id}"}]
            
            predicted_issues = []
            recent_metrics = self._get_recent_metrics(model_id, 100)
            
            if len(recent_metrics) < 10:
                return [{"warning": "Insufficient data for issue prediction"}]
            
            # Issue 1: Performance degradation trend
            performance_trend = await self._analyze_performance_trend(model_id, recent_metrics)
            if performance_trend["declining"]:
                issue = ModelIssue(
                    issue_type="performance_degradation",
                    severity="medium",
                    probability=performance_trend["confidence"],
                    description=f"Model performance is declining with {performance_trend['metric']} showing downward trend",
                    recommended_actions=[
                        "Review training data for quality issues",
                        "Consider model retraining",
                        "Investigate data drift"
                    ],
                    estimated_impact="medium"
                )
                predicted_issues.append(issue.dict())
            
            # Issue 2: Inference time increasing
            inference_times = [m["inference_time_ms"] for m in recent_metrics]
            if len(inference_times) > 20:
                recent_avg = np.mean(inference_times[-10:])
                historical_avg = np.mean(inference_times[:-10])
                
                if recent_avg > historical_avg * 1.5:  # 50% increase
                    issue = ModelIssue(
                        issue_type="inference_slowdown",
                        severity="high" if recent_avg > self._inference_time_threshold_ms else "medium",
                        probability=0.8,
                        description=f"Inference time increased by {((recent_avg/historical_avg - 1) * 100):.1f}%",
                        recommended_actions=[
                            "Profile model inference pipeline",
                            "Check for resource constraints",
                            "Consider model optimization"
                        ],
                        estimated_impact="high"
                    )
                    predicted_issues.append(issue.dict())
            
            # Issue 3: High variance in metrics
            accuracy_variance = await self._analyze_metric_variance(model_id, "accuracy", recent_metrics)
            if accuracy_variance > 0.05:  # 5% standard deviation
                issue = ModelIssue(
                    issue_type="unstable_performance",
                    severity="medium",
                    probability=0.7,
                    description=f"High variance in accuracy: {accuracy_variance:.3f}",
                    recommended_actions=[
                        "Review input data consistency",
                        "Check for overfitting",
                        "Analyze prediction patterns"
                    ],
                    estimated_impact="medium"
                )
                predicted_issues.append(issue.dict())
            
            # Issue 4: Potential data drift
            drift_result = await self.analyze_model_drift(model_id, 24)
            if not drift_result.get("error") and drift_result.get("drift_detected"):
                issue = ModelIssue(
                    issue_type="data_drift",
                    severity=drift_result["drift_severity"],
                    probability=min(drift_result["drift_score"], 1.0),
                    description="Data drift detected in model inputs",
                    recommended_actions=[
                        "Analyze input data distribution",
                        "Update training dataset",
                        "Implement drift monitoring"
                    ],
                    estimated_impact="high"
                )
                predicted_issues.append(issue.dict())
            
            # Update active issues
            self._active_issues[model_id] = [ModelIssue(**issue) for issue in predicted_issues]
            self._stats["issues_predicted"] += len(predicted_issues)
            
            return predicted_issues
            
        except Exception as e:
            self.logger.error(f"Error predicting issues for model {model_id}: {e}")
            return [{"error": str(e)}]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check component health status"""
        try:
            # Calculate model tracking statistics
            total_models = len(self._model_metrics)
            active_models = sum(
                1 for metrics in self._model_metrics.values()
                if metrics and (datetime.now() - metrics[-1]["timestamp"]).seconds < 3600
            )
            
            # Calculate recent activity
            recent_activity = sum(
                len([
                    m for m in metrics
                    if (datetime.now() - m["timestamp"]).seconds < 300  # Last 5 minutes
                ])
                for metrics in self._model_metrics.values()
            )
            
            # Count active issues
            total_issues = sum(len(issues) for issues in self._active_issues.values())
            critical_issues = sum(
                len([issue for issue in issues if issue.severity == "critical"])
                for issues in self._active_issues.values()
            )
            
            # Determine health status
            status = "healthy"
            alerts = []
            
            if not self._monitoring_enabled:
                status = "unhealthy"
                alerts.append("ML analytics monitoring disabled")
            
            if total_models == 0:
                status = "unhealthy"
                alerts.append("No ML models being tracked")
            
            if critical_issues > 0:
                status = "critical"
                alerts.append(f"{critical_issues} critical model issues detected")
            elif total_issues > total_models * 2:  # More than 2 issues per model on average
                status = "degraded"
                alerts.append("High number of model issues")
            
            if active_models < total_models * 0.5:
                status = "degraded"
                alerts.append("Many models appear inactive")
            
            if recent_activity == 0 and active_models > 0:
                status = "degraded"
                alerts.append("No recent ML metrics activity")
            
            return ComponentHealth(
                component_name="ml_analytics",
                status=status,
                last_check=datetime.now(),
                response_time_ms=0,  # Would measure actual response time
                error_rate=0,  # Would calculate from actual error tracking
                memory_usage_mb=self._estimate_memory_usage(),
                alerts=alerts,
                details={
                    "total_models": total_models,
                    "active_models": active_models,
                    "recent_activity_5min": recent_activity,
                    "total_issues": total_issues,
                    "critical_issues": critical_issues,
                    "monitoring_enabled": self._monitoring_enabled,
                    "stats": self._stats,
                }
            ).dict()
            
        except Exception as e:
            return {
                "component_name": "ml_analytics",
                "status": "error",
                "last_check": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get component performance metrics"""
        return {
            "performance": self._stats.copy(),
            "model_counts": {
                "total_tracked": len(self._model_metrics),
                "with_baselines": len(self._model_baselines),
                "with_active_issues": len([m for m, issues in self._active_issues.items() if issues]),
            },
            "memory_usage_mb": self._estimate_memory_usage(),
            "drift_thresholds": {
                "low": self._drift_threshold_low,
                "medium": self._drift_threshold_medium,
                "high": self._drift_threshold_high,
            },
        }
    
    async def configure(self, config: Dict[str, Any]) -> bool:
        """Configure component with new settings"""
        try:
            # Update configuration
            self.config.update(config)
            
            # Apply configuration changes
            if "drift_detection" in config:
                self._drift_detection_enabled = config["drift_detection"]
            
            if "monitoring_interval" in config:
                self._monitoring_interval = config["monitoring_interval"]
            
            if "accuracy_threshold" in config:
                self._accuracy_threshold = config["accuracy_threshold"]
            
            self.logger.info(f"ML analytics component reconfigured: {config}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error configuring component: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Gracefully shutdown component"""
        try:
            self.logger.info("Shutting down ML analytics component")
            
            # Stop monitoring
            self._monitoring_enabled = False
            
            # Cancel monitoring task
            if self._monitoring_task and not self._monitoring_task.done():
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Clear data
            self._model_metrics.clear()
            self._model_baselines.clear()
            self._active_issues.clear()
            
            self.logger.info("ML analytics component shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    # Private helper methods
    
    def _start_monitoring(self):
        """Start background monitoring tasks"""
        if self._monitoring_enabled:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self._monitoring_enabled:
            try:
                await asyncio.sleep(self._monitoring_interval)
                
                # Check all models for issues
                for model_id in list(self._model_metrics.keys()):
                    await self.predict_model_issues(model_id)
                    
                    # Drift detection for active models
                    if self._drift_detection_enabled:
                        await self._check_model_drift(model_id)
                
                # Cleanup old data
                self._cleanup_old_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in ML analytics monitoring loop: {e}")
                await asyncio.sleep(60)  # 1 minute pause before retrying
    
    async def _initialize_model_baseline(self, model_id: str) -> None:
        """Initialize baseline metrics for a new model"""
        try:
            metrics = self._model_metrics[model_id]
            if len(metrics) >= 10:  # Need sufficient data for baseline
                recent_metrics = list(metrics)[-10:]
                
                baseline = {}
                for metric_name in ["accuracy", "precision", "recall", "f1_score", "inference_time_ms"]:
                    values = [m[metric_name] for m in recent_metrics]
                    baseline[metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "timestamp": datetime.now(),
                    }
                
                self._model_baselines[model_id] = baseline
                self.logger.info(f"Initialized baseline for model {model_id}")
                
        except Exception as e:
            self.logger.error(f"Error initializing baseline for model {model_id}: {e}")
    
    def _get_baseline_metrics(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get baseline metrics for a model"""
        return self._model_baselines.get(model_id)
    
    def _get_recent_metrics(self, model_id: str, count: int) -> List[Dict[str, Any]]:
        """Get recent metrics for a model"""
        if model_id not in self._model_metrics:
            return []
        
        return list(self._model_metrics[model_id])[-count:]
    
    async def _calculate_drift_score(
        self,
        model_id: str,
        metric_name: str,
        baseline_stats: Dict[str, float],
        recent_metrics: List[Dict[str, Any]]
    ) -> float:
        """Calculate drift score for a specific metric"""
        try:
            recent_values = [m[metric_name] for m in recent_metrics if metric_name in m]
            
            if not recent_values:
                return 0.0
            
            # Calculate statistical distance (simplified KL divergence approximation)
            baseline_mean = baseline_stats["mean"]
            baseline_std = baseline_stats["std"]
            
            recent_mean = np.mean(recent_values)
            recent_std = np.std(recent_values) if len(recent_values) > 1 else baseline_std
            
            # Normalized difference in means
            if baseline_std > 0:
                drift_score = abs(recent_mean - baseline_mean) / baseline_std
            else:
                drift_score = abs(recent_mean - baseline_mean)
            
            # Add variance drift component
            if baseline_std > 0:
                variance_drift = abs(recent_std - baseline_std) / baseline_std
                drift_score += variance_drift * 0.5
            
            return min(drift_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating drift score: {e}")
            return 0.0
    
    def _classify_drift_severity(self, drift_score: float) -> DriftSeverity:
        """Classify drift severity based on score"""
        if drift_score >= self._drift_threshold_high:
            return DriftSeverity.CRITICAL
        elif drift_score >= self._drift_threshold_medium:
            return DriftSeverity.HIGH
        elif drift_score >= self._drift_threshold_low:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW
    
    async def _generate_drift_recommendations(
        self,
        model_id: str,
        drift_results: Dict[str, float],
        drift_severity: DriftSeverity
    ) -> List[str]:
        """Generate recommendations based on drift analysis"""
        recommendations = []
        
        if drift_severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            recommendations.append("Immediate retraining recommended")
            recommendations.append("Investigate input data distribution changes")
            
        if drift_severity == DriftSeverity.MEDIUM:
            recommendations.append("Monitor model performance closely")
            recommendations.append("Prepare for potential retraining")
            
        # Metric-specific recommendations
        if drift_results.get("accuracy", 0) > self._drift_threshold_medium:
            recommendations.append("Focus on accuracy improvement during retraining")
            
        if drift_results.get("inference_time_ms", 0) > self._drift_threshold_medium:
            recommendations.append("Investigate performance bottlenecks")
            
        if not recommendations:
            recommendations.append("Continue monitoring - no immediate action required")
            
        return recommendations
    
    async def _check_model_drift(self, model_id: str) -> None:
        """Check for model drift and log if detected"""
        try:
            drift_result = await self.analyze_model_drift(model_id, 6)  # 6 hours
            
            if not drift_result.get("error") and drift_result.get("drift_detected"):
                severity = drift_result["drift_severity"]
                if severity in ["high", "critical"]:
                    self.logger.warning(
                        f"Model {model_id} drift detected: {severity} severity "
                        f"(score: {drift_result['drift_score']:.3f})"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error checking drift for model {model_id}: {e}")
    
    async def _check_model_health(self, model_id: str, metric_data: Dict[str, Any]) -> None:
        """Check model health and log issues"""
        issues = []
        
        if metric_data["accuracy"] < self._accuracy_threshold:
            issues.append(f"Low accuracy: {metric_data['accuracy']:.3f}")
            
        if metric_data["f1_score"] < self._f1_threshold:
            issues.append(f"Low F1 score: {metric_data['f1_score']:.3f}")
            
        if metric_data["inference_time_ms"] > self._inference_time_threshold_ms:
            issues.append(f"High inference time: {metric_data['inference_time_ms']:.0f}ms")
        
        if issues:
            self.logger.warning(f"Model {model_id} health issues: {', '.join(issues)}")
    
    async def _analyze_performance_trend(
        self,
        model_id: str,
        recent_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze performance trend for a model"""
        try:
            if len(recent_metrics) < 20:
                return {"declining": False, "confidence": 0.0, "metric": None}
            
            # Analyze accuracy trend
            accuracies = [m["accuracy"] for m in recent_metrics]
            x = np.arange(len(accuracies))
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, accuracies)
            
            # Trend is declining if slope is significantly negative
            declining = slope < -0.001 and p_value < 0.05  # At least 0.1% decline per observation
            confidence = 1.0 - p_value if p_value < 0.05 else 0.0
            
            return {
                "declining": declining,
                "confidence": confidence,
                "metric": "accuracy",
                "slope": slope,
                "p_value": p_value
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance trend: {e}")
            return {"declining": False, "confidence": 0.0, "metric": None}
    
    async def _analyze_metric_variance(
        self,
        model_id: str,
        metric_name: str,
        recent_metrics: List[Dict[str, Any]]
    ) -> float:
        """Analyze variance in a specific metric"""
        try:
            values = [m[metric_name] for m in recent_metrics if metric_name in m]
            return np.std(values) if len(values) > 1 else 0.0
        except Exception:
            return 0.0
    
    async def _test_model_significance(
        self,
        model_performances: Dict[str, Dict[str, float]],
        champion_model: str
    ) -> bool:
        """Test if champion model is significantly better (simplified)"""
        try:
            champion_score = model_performances[champion_model]["accuracy"]
            
            for model_id, perf in model_performances.items():
                if model_id != champion_model:
                    # Simple threshold-based significance
                    if champion_score - perf["accuracy"] > 0.05:  # 5% difference
                        return True
            
            return False
            
        except Exception:
            return False
    
    async def _generate_comparison_recommendations(
        self,
        model_performances: Dict[str, Dict[str, float]],
        performance_rankings: List[Tuple[str, float]]
    ) -> List[str]:
        """Generate recommendations from model comparison"""
        recommendations = []
        
        try:
            champion_id, champion_score = performance_rankings[0]
            
            recommendations.append(f"Deploy model {champion_id} as primary (score: {champion_score:.3f})")
            
            # Check if champion has specific strengths
            champion_perf = model_performances[champion_id]
            if champion_perf["inference_time_ms"] < 500:
                recommendations.append("Champion model offers excellent inference speed")
            
            if champion_perf["f1_score"] > 0.9:
                recommendations.append("Champion model has excellent F1 score")
            
            # Check for underperforming models
            for model_id, score in performance_rankings[-2:]:  # Bottom 2
                if score < 0.5:
                    recommendations.append(f"Consider retiring model {model_id} (low performance)")
            
        except Exception as e:
            self.logger.error(f"Error generating comparison recommendations: {e}")
            recommendations.append("Manual review of model comparison recommended")
        
        return recommendations
    
    def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics to manage memory"""
        cutoff_time = datetime.now() - timedelta(days=7)  # Keep 7 days of data
        
        for model_id, metrics in self._model_metrics.items():
            # Convert to list for safe iteration and modification
            metrics_list = list(metrics)
            
            # Filter out old metrics
            recent_metrics = [
                m for m in metrics_list
                if m["timestamp"] >= cutoff_time
            ]
            
            # Update deque
            metrics.clear()
            metrics.extend(recent_metrics)
    
    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB"""
        # Simple estimation based on stored data
        total_metrics = sum(len(metrics) for metrics in self._model_metrics.values())
        baseline_count = len(self._model_baselines)
        issues_count = sum(len(issues) for issues in self._active_issues.values())
        
        # Rough estimates: 2KB per metric, 10KB per baseline, 1KB per issue
        return (total_metrics * 0.002) + (baseline_count * 0.01) + (issues_count * 0.001)