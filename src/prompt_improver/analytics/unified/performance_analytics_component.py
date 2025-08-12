"""Performance Analytics Component for Unified Analytics

This component handles all performance metrics collection, analysis, and monitoring
with advanced anomaly detection and trend analysis capabilities.
"""

import asyncio
import logging
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, deque

import numpy as np
from pydantic import BaseModel
from scipy import stats
from sqlalchemy.ext.asyncio import AsyncSession

from .protocols import (
    AnalyticsComponentProtocol,
    ComponentHealth,
    PerformanceAnalyticsProtocol,
    PerformanceMetrics,
)

logger = logging.getLogger(__name__)


class PerformanceThresholds(BaseModel):
    """Performance thresholds for monitoring and alerting"""
    response_time_p95_ms: float = 1000.0
    response_time_p99_ms: float = 2000.0
    throughput_min_rps: float = 10.0
    error_rate_max: float = 0.05
    cpu_usage_max: float = 0.8
    memory_usage_max_mb: float = 1000.0


class AnomalyDetectionResult(BaseModel):
    """Result of anomaly detection analysis"""
    metric_name: str
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    timestamp: datetime
    context: Dict[str, Any]


class TrendAnalysisResult(BaseModel):
    """Result of trend analysis"""
    metric_name: str
    trend_direction: str  # "increasing", "decreasing", "stable", "volatile"
    slope: float
    correlation_coefficient: float
    confidence: float
    prediction: Optional[Dict[str, float]]


class PerformanceAnalyticsComponent(PerformanceAnalyticsProtocol, AnalyticsComponentProtocol):
    """
    Performance Analytics Component implementing comprehensive performance monitoring.
    
    Features:
    - Real-time performance metrics tracking
    - Advanced anomaly detection using statistical methods
    - Trend analysis with predictive capabilities
    - Performance baseline establishment and drift detection
    - Comprehensive reporting and alerting
    - Memory-efficient data processing with sliding windows
    """
    
    def __init__(self, db_session: AsyncSession, config: Dict[str, Any]):
        self.db_session = db_session
        self.config = config
        self.logger = logger
        
        # Performance thresholds
        self.thresholds = PerformanceThresholds(**config.get("thresholds", {}))
        
        # Metrics storage with sliding windows
        self._metrics_window_size = config.get("metrics_window_size", 1000)
        self._metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._metrics_window_size)
        )
        
        # Anomaly detection configuration
        self._anomaly_detection_enabled = config.get("anomaly_detection", True)
        self._anomaly_threshold_sigma = config.get("anomaly_threshold_sigma", 2.5)
        self._baseline_window_size = config.get("baseline_window_size", 100)
        
        # Performance tracking
        self._performance_stats = {
            "metrics_tracked": 0,
            "anomalies_detected": 0,
            "trends_analyzed": 0,
            "reports_generated": 0,
            "last_analysis_time": datetime.now(),
        }
        
        # Background monitoring
        self._monitoring_enabled = True
        self._monitoring_interval = config.get("monitoring_interval", 60)
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Alerts and notifications
        self._active_alerts: Dict[str, Dict[str, Any]] = {}
        self._alert_cooldown_seconds = config.get("alert_cooldown", 300)
        
        # Start background monitoring
        self._start_monitoring()
    
    async def track_performance(self, metrics: PerformanceMetrics) -> bool:
        """
        Track performance metrics with real-time analysis.
        
        Args:
            metrics: Performance metrics to track
            
        Returns:
            Success status
        """
        try:
            # Store metrics in sliding window
            metric_data = {
                "timestamp": metrics.timestamp,
                "response_time_ms": metrics.response_time_ms,
                "throughput_rps": metrics.throughput_rps,
                "error_rate": metrics.error_rate,
                "cpu_usage_percent": metrics.cpu_usage_percent,
                "memory_usage_mb": metrics.memory_usage_mb,
            }
            
            # Add to history for each metric type
            for metric_name, value in metric_data.items():
                if metric_name != "timestamp":
                    self._metrics_history[metric_name].append({
                        "timestamp": metrics.timestamp,
                        "value": value
                    })
            
            # Update performance stats
            self._performance_stats["metrics_tracked"] += 1
            
            # Real-time anomaly detection
            if self._anomaly_detection_enabled:
                await self._check_for_anomalies(metric_data)
            
            # Check threshold violations
            await self._check_thresholds(metric_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error tracking performance metrics: {e}")
            return False
    
    async def analyze_performance_trends(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Analyze performance trends over time period.
        
        Args:
            start_time: Start of analysis period
            end_time: End of analysis period
            
        Returns:
            Trend analysis results
        """
        try:
            trends = {}
            
            # Analyze trends for each metric type
            for metric_name, history in self._metrics_history.items():
                if not history:
                    continue
                
                # Filter data within time range
                filtered_data = [
                    item for item in history
                    if start_time <= item["timestamp"] <= end_time
                ]
                
                if len(filtered_data) < 2:
                    continue
                
                # Perform trend analysis
                trend_result = await self._analyze_metric_trend(metric_name, filtered_data)
                trends[metric_name] = trend_result.dict()
            
            # Overall performance assessment
            overall_assessment = await self._assess_overall_performance(trends)
            
            self._performance_stats["trends_analyzed"] += 1
            
            return {
                "analysis_period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                },
                "trends": trends,
                "overall_assessment": overall_assessment,
                "recommendations": await self._generate_performance_recommendations(trends),
                "metadata": {
                    "analyzed_at": datetime.now().isoformat(),
                    "metrics_analyzed": len(trends),
                    "data_points_processed": sum(len(data["data_points"]) for data in trends.values()),
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance trends: {e}")
            return {
                "error": str(e),
                "analysis_period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                }
            }
    
    async def detect_performance_anomalies(
        self,
        threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detect performance anomalies using statistical methods.
        
        Args:
            threshold: Standard deviation threshold for anomaly detection
            
        Returns:
            List of detected anomalies
        """
        try:
            anomalies = []
            
            # Analyze each metric for anomalies
            for metric_name, history in self._metrics_history.items():
                if len(history) < self._baseline_window_size:
                    continue
                
                # Get recent data for anomaly detection
                recent_data = list(history)[-50:]  # Last 50 data points
                baseline_data = list(history)[-self._baseline_window_size:-50]  # Previous baseline
                
                if len(baseline_data) < 10:
                    continue
                
                # Calculate baseline statistics
                baseline_values = [item["value"] for item in baseline_data]
                baseline_mean = statistics.mean(baseline_values)
                baseline_std = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0
                
                if baseline_std == 0:
                    continue
                
                # Check recent data for anomalies
                for data_point in recent_data:
                    value = data_point["value"]
                    z_score = abs(value - baseline_mean) / baseline_std
                    
                    if z_score > threshold:
                        anomaly = AnomalyDetectionResult(
                            metric_name=metric_name,
                            anomaly_score=z_score,
                            is_anomaly=True,
                            threshold=threshold,
                            timestamp=data_point["timestamp"],
                            context={
                                "value": value,
                                "baseline_mean": baseline_mean,
                                "baseline_std": baseline_std,
                                "severity": "high" if z_score > threshold * 1.5 else "medium"
                            }
                        )
                        anomalies.append(anomaly.dict())
            
            # Sort by anomaly score (most severe first)
            anomalies.sort(key=lambda x: x["anomaly_score"], reverse=True)
            
            self._performance_stats["anomalies_detected"] += len(anomalies)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return []
    
    async def generate_performance_report(
        self,
        report_type: str = "summary"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            report_type: Type of report ("summary", "detailed", "executive")
            
        Returns:
            Performance report
        """
        try:
            current_time = datetime.now()
            
            if report_type == "summary":
                report = await self._generate_summary_report(current_time)
            elif report_type == "detailed":
                report = await self._generate_detailed_report(current_time)
            elif report_type == "executive":
                report = await self._generate_executive_report(current_time)
            else:
                raise ValueError(f"Unknown report type: {report_type}")
            
            self._performance_stats["reports_generated"] += 1
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {
                "error": str(e),
                "report_type": report_type,
                "generated_at": datetime.now().isoformat()
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check component health status"""
        try:
            # Calculate metrics coverage
            total_metrics = len(self._metrics_history)
            active_metrics = sum(1 for history in self._metrics_history.values() if history)
            
            # Calculate recent activity
            recent_activity = sum(
                1 for history in self._metrics_history.values()
                for item in history
                if (datetime.now() - item["timestamp"]).seconds < 300  # Last 5 minutes
            )
            
            # Determine health status
            status = "healthy"
            alerts = []
            
            if not self._monitoring_enabled:
                status = "unhealthy"
                alerts.append("Performance monitoring disabled")
            
            if active_metrics == 0:
                status = "unhealthy"
                alerts.append("No active metrics being tracked")
            elif active_metrics < total_metrics * 0.8:
                status = "degraded"
                alerts.append("Some metrics not being tracked")
            
            if recent_activity == 0:
                status = "degraded"
                alerts.append("No recent metric activity")
            
            return ComponentHealth(
                component_name="performance_analytics",
                status=status,
                last_check=datetime.now(),
                response_time_ms=0,  # Would measure actual response time
                error_rate=0,  # Would calculate from actual error tracking
                memory_usage_mb=self._estimate_memory_usage(),
                alerts=alerts,
                details={
                    "total_metrics_tracked": total_metrics,
                    "active_metrics": active_metrics,
                    "recent_activity_count": recent_activity,
                    "monitoring_enabled": self._monitoring_enabled,
                    "performance_stats": self._performance_stats,
                    "active_alerts_count": len(self._active_alerts),
                }
            ).dict()
            
        except Exception as e:
            return {
                "component_name": "performance_analytics",
                "status": "error",
                "last_check": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get component performance metrics"""
        return {
            "performance": self._performance_stats.copy(),
            "metrics_coverage": {
                metric_name: len(history) for metric_name, history in self._metrics_history.items()
            },
            "memory_usage_mb": self._estimate_memory_usage(),
            "active_alerts": len(self._active_alerts),
            "thresholds": self.thresholds.dict(),
        }
    
    async def configure(self, config: Dict[str, Any]) -> bool:
        """Configure component with new settings"""
        try:
            # Update configuration
            self.config.update(config)
            
            # Apply configuration changes
            if "thresholds" in config:
                self.thresholds = PerformanceThresholds(**config["thresholds"])
            
            if "anomaly_detection" in config:
                self._anomaly_detection_enabled = config["anomaly_detection"]
            
            if "monitoring_interval" in config:
                self._monitoring_interval = config["monitoring_interval"]
            
            self.logger.info(f"Performance analytics component reconfigured: {config}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error configuring component: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Gracefully shutdown component"""
        try:
            self.logger.info("Shutting down performance analytics component")
            
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
            self._metrics_history.clear()
            self._active_alerts.clear()
            
            self.logger.info("Performance analytics component shutdown complete")
            
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
                
                # Perform periodic analysis
                await self._periodic_analysis()
                
                # Clean up old alerts
                await self._cleanup_old_alerts()
                
                self._performance_stats["last_analysis_time"] = datetime.now()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Brief pause before retrying
    
    async def _check_for_anomalies(self, metric_data: Dict[str, Any]) -> None:
        """Check for anomalies in real-time metrics"""
        try:
            anomalies = await self.detect_performance_anomalies(self._anomaly_threshold_sigma)
            
            for anomaly in anomalies:
                if anomaly["context"]["severity"] == "high":
                    await self._trigger_alert("anomaly", anomaly)
                    
        except Exception as e:
            self.logger.error(f"Error checking for anomalies: {e}")
    
    async def _check_thresholds(self, metric_data: Dict[str, Any]) -> None:
        """Check for threshold violations"""
        try:
            violations = []
            
            # Check each threshold
            if metric_data["response_time_ms"] > self.thresholds.response_time_p99_ms:
                violations.append({
                    "metric": "response_time_ms",
                    "value": metric_data["response_time_ms"],
                    "threshold": self.thresholds.response_time_p99_ms,
                    "severity": "high"
                })
            
            if metric_data["throughput_rps"] < self.thresholds.throughput_min_rps:
                violations.append({
                    "metric": "throughput_rps",
                    "value": metric_data["throughput_rps"],
                    "threshold": self.thresholds.throughput_min_rps,
                    "severity": "medium"
                })
            
            if metric_data["error_rate"] > self.thresholds.error_rate_max:
                violations.append({
                    "metric": "error_rate",
                    "value": metric_data["error_rate"],
                    "threshold": self.thresholds.error_rate_max,
                    "severity": "high"
                })
            
            if metric_data["cpu_usage_percent"] > self.thresholds.cpu_usage_max:
                violations.append({
                    "metric": "cpu_usage_percent",
                    "value": metric_data["cpu_usage_percent"],
                    "threshold": self.thresholds.cpu_usage_max,
                    "severity": "medium"
                })
            
            if metric_data["memory_usage_mb"] > self.thresholds.memory_usage_max_mb:
                violations.append({
                    "metric": "memory_usage_mb",
                    "value": metric_data["memory_usage_mb"],
                    "threshold": self.thresholds.memory_usage_max_mb,
                    "severity": "medium"
                })
            
            # Trigger alerts for violations
            for violation in violations:
                await self._trigger_alert("threshold", violation)
                
        except Exception as e:
            self.logger.error(f"Error checking thresholds: {e}")
    
    async def _analyze_metric_trend(
        self, 
        metric_name: str, 
        data: List[Dict[str, Any]]
    ) -> TrendAnalysisResult:
        """Analyze trend for a specific metric"""
        try:
            if len(data) < 2:
                return TrendAnalysisResult(
                    metric_name=metric_name,
                    trend_direction="stable",
                    slope=0.0,
                    correlation_coefficient=0.0,
                    confidence=0.0
                )
            
            # Extract timestamps and values
            timestamps = [(item["timestamp"] - data[0]["timestamp"]).total_seconds() for item in data]
            values = [item["value"] for item in data]
            
            # Calculate linear regression
            if len(timestamps) > 1 and len(set(values)) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, values)
                
                # Determine trend direction
                if abs(slope) < 0.01:  # Threshold for "stable"
                    trend_direction = "stable"
                elif slope > 0:
                    trend_direction = "increasing"
                else:
                    trend_direction = "decreasing"
                
                # Check for volatility
                if len(values) > 5:
                    cv = statistics.stdev(values) / statistics.mean(values)
                    if cv > 0.5:  # High coefficient of variation
                        trend_direction = "volatile"
                
                # Calculate confidence (1 - p_value, capped at 0.95)
                confidence = min(0.95, 1.0 - p_value) if p_value < 1.0 else 0.0
                
                # Generate prediction for next period
                if confidence > 0.5 and trend_direction != "volatile":
                    next_timestamp = timestamps[-1] + 3600  # 1 hour ahead
                    predicted_value = slope * next_timestamp + intercept
                    prediction = {
                        "next_hour_prediction": predicted_value,
                        "confidence": confidence
                    }
                else:
                    prediction = None
                
            else:
                slope = 0.0
                r_value = 0.0
                trend_direction = "stable"
                confidence = 0.0
                prediction = None
            
            return TrendAnalysisResult(
                metric_name=metric_name,
                trend_direction=trend_direction,
                slope=slope,
                correlation_coefficient=r_value,
                confidence=confidence,
                prediction=prediction
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend for {metric_name}: {e}")
            return TrendAnalysisResult(
                metric_name=metric_name,
                trend_direction="unknown",
                slope=0.0,
                correlation_coefficient=0.0,
                confidence=0.0
            )
    
    async def _assess_overall_performance(self, trends: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall performance based on trends"""
        try:
            assessment = {
                "overall_status": "good",
                "key_findings": [],
                "risk_factors": [],
                "performance_score": 0.0
            }
            
            score_components = []
            
            # Analyze each metric trend
            for metric_name, trend_data in trends.items():
                trend_direction = trend_data.get("trend_direction", "stable")
                confidence = trend_data.get("confidence", 0.0)
                
                # Score based on trend direction and metric type
                if metric_name in ["response_time_ms", "error_rate", "cpu_usage_percent", "memory_usage_mb"]:
                    # Lower is better for these metrics
                    if trend_direction == "decreasing":
                        score = 1.0 * confidence
                    elif trend_direction == "stable":
                        score = 0.8
                    elif trend_direction == "increasing":
                        score = 0.2 * (1 - confidence)
                    else:  # volatile
                        score = 0.4
                else:
                    # Higher is better for throughput
                    if trend_direction == "increasing":
                        score = 1.0 * confidence
                    elif trend_direction == "stable":
                        score = 0.8
                    elif trend_direction == "decreasing":
                        score = 0.2 * (1 - confidence)
                    else:  # volatile
                        score = 0.4
                
                score_components.append(score)
                
                # Add key findings
                if confidence > 0.7:
                    if trend_direction in ["increasing", "decreasing"]:
                        assessment["key_findings"].append(
                            f"{metric_name} is {trend_direction} with high confidence"
                        )
                
                # Add risk factors
                if score < 0.3:
                    assessment["risk_factors"].append(
                        f"Poor {metric_name} trend: {trend_direction}"
                    )
            
            # Calculate overall performance score
            if score_components:
                assessment["performance_score"] = statistics.mean(score_components)
                
                # Determine overall status
                if assessment["performance_score"] > 0.8:
                    assessment["overall_status"] = "excellent"
                elif assessment["performance_score"] > 0.6:
                    assessment["overall_status"] = "good"
                elif assessment["performance_score"] > 0.4:
                    assessment["overall_status"] = "fair"
                else:
                    assessment["overall_status"] = "poor"
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing overall performance: {e}")
            return {
                "overall_status": "unknown",
                "error": str(e),
                "performance_score": 0.0
            }
    
    async def _generate_performance_recommendations(self, trends: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on trends"""
        recommendations = []
        
        try:
            for metric_name, trend_data in trends.items():
                trend_direction = trend_data.get("trend_direction", "stable")
                confidence = trend_data.get("confidence", 0.0)
                
                if confidence < 0.5:
                    continue  # Skip low-confidence trends
                
                if metric_name == "response_time_ms" and trend_direction == "increasing":
                    recommendations.append("Response time is increasing - consider performance optimization")
                
                elif metric_name == "error_rate" and trend_direction == "increasing":
                    recommendations.append("Error rate is rising - investigate error causes")
                
                elif metric_name == "throughput_rps" and trend_direction == "decreasing":
                    recommendations.append("Throughput is declining - check system capacity")
                
                elif metric_name in ["cpu_usage_percent", "memory_usage_mb"] and trend_direction == "increasing":
                    recommendations.append(f"{metric_name.replace('_', ' ').title()} is increasing - consider scaling")
                
                elif trend_direction == "volatile":
                    recommendations.append(f"{metric_name.replace('_', ' ').title()} is volatile - investigate instability")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Performance trends are stable - maintain current configuration")
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Error generating recommendations - manual analysis recommended")
        
        return recommendations
    
    async def _generate_summary_report(self, current_time: datetime) -> Dict[str, Any]:
        """Generate summary performance report"""
        # Get latest metrics for each type
        latest_metrics = {}
        for metric_name, history in self._metrics_history.items():
            if history:
                latest_metrics[metric_name] = history[-1]["value"]
        
        # Get recent trends (last hour)
        start_time = current_time - timedelta(hours=1)
        trends = await self.analyze_performance_trends(start_time, current_time)
        
        return {
            "report_type": "summary",
            "generated_at": current_time.isoformat(),
            "latest_metrics": latest_metrics,
            "trends_summary": trends.get("overall_assessment", {}),
            "active_alerts": len(self._active_alerts),
            "recommendations": trends.get("recommendations", [])[:3],  # Top 3
        }
    
    async def _generate_detailed_report(self, current_time: datetime) -> Dict[str, Any]:
        """Generate detailed performance report"""
        # Get trends for last 24 hours
        start_time = current_time - timedelta(hours=24)
        trends = await self.analyze_performance_trends(start_time, current_time)
        
        # Get anomalies
        anomalies = await self.detect_performance_anomalies()
        
        return {
            "report_type": "detailed",
            "generated_at": current_time.isoformat(),
            "analysis_period": "24_hours",
            "trends_analysis": trends,
            "anomalies": anomalies[:10],  # Top 10 anomalies
            "performance_stats": self._performance_stats,
            "thresholds": self.thresholds.dict(),
            "active_alerts": self._active_alerts,
        }
    
    async def _generate_executive_report(self, current_time: datetime) -> Dict[str, Any]:
        """Generate executive-level performance report"""
        # Get trends for last week
        start_time = current_time - timedelta(days=7)
        trends = await self.analyze_performance_trends(start_time, current_time)
        
        overall_assessment = trends.get("overall_assessment", {})
        
        return {
            "report_type": "executive",
            "generated_at": current_time.isoformat(),
            "executive_summary": {
                "performance_status": overall_assessment.get("overall_status", "unknown"),
                "performance_score": overall_assessment.get("performance_score", 0.0),
                "key_metrics_trending": len([
                    t for t in trends.get("trends", {}).values()
                    if t.get("trend_direction") in ["increasing", "decreasing"]
                ]),
                "critical_issues": len([
                    alert for alert in self._active_alerts.values()
                    if alert.get("severity") == "high"
                ]),
            },
            "key_findings": overall_assessment.get("key_findings", []),
            "risk_factors": overall_assessment.get("risk_factors", []),
            "recommendations": trends.get("recommendations", [])[:5],  # Top 5
        }
    
    async def _periodic_analysis(self) -> None:
        """Perform periodic background analysis"""
        try:
            # Detect anomalies
            anomalies = await self.detect_performance_anomalies()
            
            # Log significant anomalies
            high_severity_anomalies = [
                a for a in anomalies 
                if a.get("context", {}).get("severity") == "high"
            ]
            
            if high_severity_anomalies:
                self.logger.warning(
                    f"Detected {len(high_severity_anomalies)} high-severity performance anomalies"
                )
                
        except Exception as e:
            self.logger.error(f"Error in periodic analysis: {e}")
    
    async def _trigger_alert(self, alert_type: str, alert_data: Dict[str, Any]) -> None:
        """Trigger performance alert with cooldown"""
        try:
            alert_key = f"{alert_type}_{alert_data.get('metric', 'unknown')}"
            current_time = datetime.now()
            
            # Check cooldown
            if alert_key in self._active_alerts:
                last_alert_time = datetime.fromisoformat(self._active_alerts[alert_key]["last_triggered"])
                if (current_time - last_alert_time).seconds < self._alert_cooldown_seconds:
                    return  # Still in cooldown period
            
            # Create alert
            alert = {
                "alert_type": alert_type,
                "data": alert_data,
                "triggered_at": current_time.isoformat(),
                "last_triggered": current_time.isoformat(),
                "trigger_count": self._active_alerts.get(alert_key, {}).get("trigger_count", 0) + 1
            }
            
            self._active_alerts[alert_key] = alert
            
            # Log alert
            self.logger.warning(f"Performance alert triggered: {alert_type} - {alert_data}")
            
        except Exception as e:
            self.logger.error(f"Error triggering alert: {e}")
    
    async def _cleanup_old_alerts(self) -> None:
        """Clean up old alerts that are no longer relevant"""
        try:
            current_time = datetime.now()
            alert_retention_hours = 24
            
            expired_alerts = []
            for alert_key, alert in self._active_alerts.items():
                triggered_time = datetime.fromisoformat(alert["triggered_at"])
                if (current_time - triggered_time).hours > alert_retention_hours:
                    expired_alerts.append(alert_key)
            
            for alert_key in expired_alerts:
                del self._active_alerts[alert_key]
                
        except Exception as e:
            self.logger.error(f"Error cleaning up alerts: {e}")
    
    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB"""
        # Simple estimation based on stored data
        total_data_points = sum(len(history) for history in self._metrics_history.values())
        # Rough estimate: 200 bytes per data point
        return total_data_points * 0.0002