"""
ML Health Integration Manager - 2025 Best Practices

Provides seamless integration between ML services and health monitoring,
automatic model registration, and performance tracking integration.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from ...utils.datetime_utils import aware_utc_now
from .ml_health_monitor import get_ml_health_monitor
from .drift_detector import get_drift_detector
from .model_performance_tracker import get_performance_tracker

logger = logging.getLogger(__name__)


class MLHealthIntegrationManager:
    """
    Manages integration between ML services and health monitoring systems.
    
    Provides automatic model registration, performance tracking integration,
    and unified health monitoring for ML operations.
    """
    
    def __init__(self):
        self._initialized = False
        self._health_monitor = None
        self._drift_detector = None
        self._performance_tracker = None
    
    async def initialize(self) -> bool:
        """Initialize all health monitoring components"""
        try:
            if self._initialized:
                return True
            
            # Initialize monitoring components
            self._health_monitor = await get_ml_health_monitor()
            self._drift_detector = await get_drift_detector()
            self._performance_tracker = await get_performance_tracker()
            
            self._initialized = True
            logger.info("ML Health Integration Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ML Health Integration Manager: {e}")
            return False
    
    async def register_model_for_monitoring(
        self,
        model_id: str,
        model: Any,
        model_type: str = "unknown",
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a model for comprehensive health monitoring"""
        try:
            if not self._initialized:
                await self.initialize()
            
            if not self._health_monitor:
                logger.warning("Health monitor not available for model registration")
                return False
            
            # Register with health monitor
            success = await self._health_monitor.register_model(
                model_id=model_id,
                model=model,
                model_type=model_type,
                version=version
            )
            
            if success:
                logger.info(f"Successfully registered model {model_id} for health monitoring")
            else:
                logger.error(f"Failed to register model {model_id} for health monitoring")
            
            return success
            
        except Exception as e:
            logger.error(f"Error registering model {model_id} for monitoring: {e}")
            return False
    
    async def unregister_model(self, model_id: str) -> bool:
        """Unregister a model from health monitoring"""
        try:
            if not self._initialized:
                return False
            
            if self._health_monitor:
                success = await self._health_monitor.unregister_model(model_id)
                if success:
                    logger.info(f"Unregistered model {model_id} from health monitoring")
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Error unregistering model {model_id}: {e}")
            return False
    
    async def track_inference(
        self,
        model_id: str,
        request_id: str,
        prediction: Optional[float] = None,
        confidence: Optional[float] = None,
        features: Optional[list] = None,
        start_performance_tracking: bool = True
    ) -> Optional[str]:
        """Start tracking an inference request"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Start performance tracking
            if start_performance_tracking and self._performance_tracker:
                await self._performance_tracker.start_request_tracking(
                    model_id=model_id,
                    request_id=request_id,
                    metadata={"features_count": len(features) if features else 0}
                )
            
            # Record prediction for drift detection
            if (prediction is not None and confidence is not None and 
                self._drift_detector):
                await self._drift_detector.record_prediction(
                    model_id=model_id,
                    prediction=prediction,
                    confidence=confidence,
                    features=features
                )
            
            return request_id
            
        except Exception as e:
            logger.error(f"Error starting inference tracking for {model_id}: {e}")
            return None
    
    async def complete_inference_tracking(
        self,
        model_id: str,
        request_id: str,
        success: bool,
        error_type: Optional[str] = None,
        latency_ms: Optional[float] = None
    ) -> Optional[float]:
        """Complete tracking for an inference request"""
        try:
            if not self._initialized:
                return None
            
            # Complete performance tracking
            recorded_latency = None
            if self._performance_tracker:
                recorded_latency = await self._performance_tracker.end_request_tracking(
                    model_id=model_id,
                    request_id=request_id,
                    success=success,
                    error_type=error_type
                )
            
            # Use provided latency or recorded latency
            final_latency = latency_ms or recorded_latency
            
            # Record inference metrics in health monitor
            if final_latency is not None and self._health_monitor:
                await self._health_monitor.record_inference(
                    model_id=model_id,
                    latency_ms=final_latency,
                    success=success,
                    error_type=error_type
                )
            
            return final_latency
            
        except Exception as e:
            logger.error(f"Error completing inference tracking for {model_id}: {e}")
            return None
    
    async def get_model_health_summary(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive health summary for a model"""
        try:
            if not self._initialized:
                await self.initialize()
            
            summary = {
                "model_id": model_id, 
                "timestamp": aware_utc_now().isoformat(),
                "components": {}
            }
            
            # Health monitor data
            if self._health_monitor:
                health_metrics = await self._health_monitor.get_model_health(model_id)
                if health_metrics:
                    summary["components"]["health"] = {
                        "status": health_metrics.status,
                        "memory_mb": health_metrics.memory_mb,
                        "total_predictions": health_metrics.total_predictions,
                        "success_rate": health_metrics.success_rate,
                        "error_rate": health_metrics.error_rate,
                        "latency_p95": health_metrics.latency_p95,
                        "version": health_metrics.version
                    }
            
            # Performance tracking data
            if self._performance_tracker:
                perf_summary = await self._performance_tracker.get_model_performance_summary(
                    model_id=model_id,
                    hours=24
                )
                if perf_summary.get("status") != "no_data":
                    summary["components"]["performance"] = {
                        "latency_summary": perf_summary.get("latency_summary", {}),
                        "throughput_summary": perf_summary.get("throughput_summary", {}),
                        "quality_summary": perf_summary.get("quality_summary", {}),
                        "current_status": perf_summary.get("current_status", {})
                    }
            
            # Drift detection data
            if self._drift_detector:
                drift_status = await self._drift_detector.get_drift_status(model_id)
                if not drift_status.get("error"):
                    summary["components"]["drift"] = {
                        "risk_level": drift_status.get("risk_level", "unknown"),
                        "drift_detected": bool(drift_status.get("current_drift", {}).get("drift_detected")),
                        "baseline_established": drift_status.get("baseline_established", False),
                        "recent_drift_events": drift_status.get("recent_drift_events", 0)
                    }
            
            # Overall health assessment
            summary["overall_healthy"] = self._assess_overall_health(summary["components"])
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting health summary for {model_id}: {e}")
            return None
    
    async def get_system_health_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive system health dashboard"""
        try:
            if not self._initialized:
                await self.initialize()
            
            dashboard = {
                "timestamp": aware_utc_now().isoformat(),
                "system_health": {},
                "model_summaries": [],
                "alerts": [],
                "recommendations": []
            }
            
            # System-wide health from health monitor
            if self._health_monitor:
                system_health = await self._health_monitor.get_system_health()
                dashboard["system_health"] = system_health
                
                # Get individual model health
                all_models_health = await self._health_monitor.get_all_models_health()
                for model_health in all_models_health:
                    model_summary = await self.get_model_health_summary(model_health.model_id)
                    if model_summary:
                        dashboard["model_summaries"].append(model_summary)
            
            # System-wide performance metrics
            if self._performance_tracker:
                all_performance = await self._performance_tracker.get_all_models_performance()
                dashboard["performance_overview"] = {
                    "total_models_tracked": len(all_performance),
                    "models_with_issues": len([
                        p for p in all_performance 
                        if p.get("current_status", {}).get("status") in ["warning", "poor"]
                    ])
                }
            
            # System-wide drift status
            if self._drift_detector:
                all_drift_status = await self._drift_detector.get_all_models_drift_status()
                dashboard["drift_overview"] = {
                    "total_models_monitored": len(all_drift_status),
                    "models_with_drift": len([
                        d for d in all_drift_status 
                        if d.get("current_drift", {}).get("drift_detected")
                    ]),
                    "high_risk_models": len([
                        d for d in all_drift_status
                        if d.get("risk_level") == "high"
                    ])
                }
            
            # Generate alerts and recommendations
            dashboard["alerts"] = self._generate_system_alerts(dashboard)
            dashboard["recommendations"] = self._generate_system_recommendations(dashboard)
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error getting system health dashboard: {e}")
            return {
                "error": str(e),
                "timestamp": aware_utc_now().isoformat()
            }
    
    def _assess_overall_health(self, components: Dict[str, Any]) -> bool:
        """Assess overall health based on component data"""
        health_scores = []
        
        # Health component score
        health_data = components.get("health", {})
        if health_data:
            success_rate = health_data.get("success_rate", 0.0)
            health_scores.append(success_rate)
        
        # Performance component score
        perf_data = components.get("performance", {})
        if perf_data:
            current_status = perf_data.get("current_status", {})
            status = current_status.get("status", "unknown")
            if status == "good":
                health_scores.append(1.0)
            elif status == "warning":
                health_scores.append(0.7)
            else:
                health_scores.append(0.3)
        
        # Drift component score
        drift_data = components.get("drift", {})
        if drift_data:
            risk_level = drift_data.get("risk_level", "unknown")
            if risk_level == "low":
                health_scores.append(1.0)
            elif risk_level == "medium":
                health_scores.append(0.6)
            else:
                health_scores.append(0.2)
        
        # Overall health is good if average score > 0.7
        if health_scores:
            avg_score = sum(health_scores) / len(health_scores)
            return avg_score > 0.7
        
        return False  # Unknown health status
    
    def _generate_system_alerts(self, dashboard: Dict[str, Any]) -> list:
        """Generate system-wide alerts"""
        alerts = []
        
        # System health alerts
        system_health = dashboard.get("system_health", {})
        if not system_health.get("healthy", True):
            alerts.append({
                "level": "critical",
                "component": "system",
                "message": "Overall ML system health is degraded",
                "timestamp": aware_utc_now().isoformat()
            })
        
        # Performance alerts
        perf_overview = dashboard.get("performance_overview", {})
        models_with_issues = perf_overview.get("models_with_issues", 0)
        if models_with_issues > 0:
            alerts.append({
                "level": "warning",
                "component": "performance",
                "message": f"{models_with_issues} models have performance issues",
                "timestamp": aware_utc_now().isoformat()
            })
        
        # Drift alerts
        drift_overview = dashboard.get("drift_overview", {})
        high_risk_models = drift_overview.get("high_risk_models", 0)
        if high_risk_models > 0:
            alerts.append({
                "level": "warning",
                "component": "drift",
                "message": f"{high_risk_models} models have high drift risk",
                "timestamp": aware_utc_now().isoformat()
            })
        
        return alerts
    
    def _generate_system_recommendations(self, dashboard: Dict[str, Any]) -> list:
        """Generate system-wide recommendations"""
        recommendations = []
        
        # Health recommendations
        system_health = dashboard.get("system_health", {})
        if system_health.get("health_score", 1.0) < 0.7:
            recommendations.append("Review individual model health metrics and address issues")
        
        # Performance recommendations
        perf_overview = dashboard.get("performance_overview", {})
        if perf_overview.get("models_with_issues", 0) > 0:
            recommendations.append("Investigate performance issues in affected models")
        
        # Drift recommendations  
        drift_overview = dashboard.get("drift_overview", {})
        if drift_overview.get("models_with_drift", 0) > 0:
            recommendations.append("Consider retraining models with detected drift")
        
        # Default recommendation
        if not recommendations:
            recommendations.append("ML system health is optimal - continue monitoring")
        
        return recommendations


# Global integration manager instance
_integration_manager: Optional[MLHealthIntegrationManager] = None

async def get_ml_health_integration_manager() -> MLHealthIntegrationManager:
    """Get or create global ML health integration manager"""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = MLHealthIntegrationManager()
        await _integration_manager.initialize()
    return _integration_manager