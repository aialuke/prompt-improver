"""
ML Health Integration Manager - 2025 Best Practices

Provides seamless integration between ML services and health monitoring,
automatic model registration, and performance tracking integration.
"""
import asyncio
import logging
from typing import Any, Dict, Optional
from ...utils.datetime_utils import aware_utc_now
from .drift_detector import get_drift_detector
from .ml_health_monitor import get_ml_health_monitor
from .model_performance_tracker import get_performance_tracker
logger = logging.getLogger(__name__)

class MLHealthIntegrationService:
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
            self._health_monitor = await get_ml_health_monitor()
            self._drift_detector = await get_drift_detector()
            self._performance_tracker = await get_performance_tracker()
            self._initialized = True
            logger.info('ML Health Integration Manager initialized successfully')
            return True
        except Exception as e:
            logger.error('Failed to initialize ML Health Integration Manager: %s', e)
            return False

    async def register_model_for_monitoring(self, model_id: str, model: Any, model_type: str='unknown', version: str | None=None, metadata: dict[str, Any] | None=None) -> bool:
        """Register a model for comprehensive health monitoring"""
        try:
            if not self._initialized:
                await self.initialize()
            if not self._health_monitor:
                logger.warning('Health monitor not available for model registration')
                return False
            success = await self._health_monitor.register_model(model_id=model_id, model=model, model_type=model_type, version=version)
            if success:
                logger.info('Successfully registered model %s for health monitoring', model_id)
            else:
                logger.error('Failed to register model %s for health monitoring', model_id)
            return success
        except Exception as e:
            logger.error('Error registering model {model_id} for monitoring: %s', e)
            return False

    async def unregister_model(self, model_id: str) -> bool:
        """Unregister a model from health monitoring"""
        try:
            if not self._initialized:
                return False
            if self._health_monitor:
                success = await self._health_monitor.unregister_model(model_id)
                if success:
                    logger.info('Unregistered model %s from health monitoring', model_id)
                return success
            return False
        except Exception as e:
            logger.error('Error unregistering model {model_id}: %s', e)
            return False

    async def track_inference(self, model_id: str, request_id: str, prediction: float | None=None, confidence: float | None=None, features: list | None=None, start_performance_tracking: bool=True) -> str | None:
        """Start tracking an inference request"""
        try:
            if not self._initialized:
                await self.initialize()
            if start_performance_tracking and self._performance_tracker:
                await self._performance_tracker.start_request_tracking(model_id=model_id, request_id=request_id, metadata={'features_count': len(features) if features else 0})
            if prediction is not None and confidence is not None and self._drift_detector:
                await self._drift_detector.record_prediction(model_id=model_id, prediction=prediction, confidence=confidence, features=features)
            return request_id
        except Exception as e:
            logger.error('Error starting inference tracking for {model_id}: %s', e)
            return None

    async def complete_inference_tracking(self, model_id: str, request_id: str, success: bool, error_type: str | None=None, latency_ms: float | None=None) -> float | None:
        """Complete tracking for an inference request"""
        try:
            if not self._initialized:
                return None
            recorded_latency = None
            if self._performance_tracker:
                recorded_latency = await self._performance_tracker.end_request_tracking(model_id=model_id, request_id=request_id, success=success, error_type=error_type)
            final_latency = latency_ms or recorded_latency
            if final_latency is not None and self._health_monitor:
                await self._health_monitor.record_inference(model_id=model_id, latency_ms=final_latency, success=success, error_type=error_type)
            return final_latency
        except Exception as e:
            logger.error('Error completing inference tracking for {model_id}: %s', e)
            return None

    async def get_model_health_summary(self, model_id: str) -> dict[str, Any] | None:
        """Get comprehensive health summary for a model"""
        try:
            if not self._initialized:
                await self.initialize()
            summary = {'model_id': model_id, 'timestamp': aware_utc_now().isoformat(), 'components': {}}
            if self._health_monitor:
                health_metrics = await self._health_monitor.get_model_health(model_id)
                if health_metrics:
                    summary['components']['health'] = {'status': health_metrics.status, 'memory_mb': health_metrics.memory_mb, 'total_predictions': health_metrics.total_predictions, 'success_rate': health_metrics.success_rate, 'error_rate': health_metrics.error_rate, 'latency_p95': health_metrics.latency_p95, 'version': health_metrics.version}
            if self._performance_tracker:
                perf_summary = await self._performance_tracker.get_model_performance_summary(model_id=model_id, hours=24)
                if perf_summary.get('status') != 'no_data':
                    summary['components']['performance'] = {'latency_summary': perf_summary.get('latency_summary', {}), 'throughput_summary': perf_summary.get('throughput_summary', {}), 'quality_summary': perf_summary.get('quality_summary', {}), 'current_status': perf_summary.get('current_status', {})}
            if self._drift_detector:
                drift_status = await self._drift_detector.get_drift_status(model_id)
                if not drift_status.get('error'):
                    summary['components']['drift'] = {'risk_level': drift_status.get('risk_level', 'unknown'), 'drift_detected': bool(drift_status.get('current_drift', {}).get('drift_detected')), 'baseline_established': drift_status.get('baseline_established', False), 'recent_drift_events': drift_status.get('recent_drift_events', 0)}
            summary['overall_healthy'] = self._assess_overall_health(summary['components'])
            return summary
        except Exception as e:
            logger.error('Error getting health summary for {model_id}: %s', e)
            return None

    async def get_system_health_dashboard(self) -> dict[str, Any]:
        """Get comprehensive system health dashboard"""
        try:
            if not self._initialized:
                await self.initialize()
            dashboard = {'timestamp': aware_utc_now().isoformat(), 'system_health': {}, 'model_summaries': [], 'alerts': [], 'recommendations': []}
            if self._health_monitor:
                system_health = await self._health_monitor.get_system_health()
                dashboard['system_health'] = system_health
                all_models_health = await self._health_monitor.get_all_models_health()
                for model_health in all_models_health:
                    model_summary = await self.get_model_health_summary(model_health.model_id)
                    if model_summary:
                        dashboard['model_summaries'].append(model_summary)
            if self._performance_tracker:
                all_performance = await self._performance_tracker.get_all_models_performance()
                dashboard['performance_overview'] = {'total_models_tracked': len(all_performance), 'models_with_issues': len([p for p in all_performance if p.get('current_status', {}).get('status') in ['warning', 'poor']])}
            if self._drift_detector:
                all_drift_status = await self._drift_detector.get_all_models_drift_status()
                dashboard['drift_overview'] = {'total_models_monitored': len(all_drift_status), 'models_with_drift': len([d for d in all_drift_status if d.get('current_drift', {}).get('drift_detected')]), 'high_risk_models': len([d for d in all_drift_status if d.get('risk_level') == 'high'])}
            dashboard['alerts'] = self._generate_system_alerts(dashboard)
            dashboard['recommendations'] = self._generate_system_recommendations(dashboard)
            return dashboard
        except Exception as e:
            logger.error('Error getting system health dashboard: %s', e)
            return {'error': str(e), 'timestamp': aware_utc_now().isoformat()}

    def _assess_overall_health(self, components: dict[str, Any]) -> bool:
        """Assess overall health based on component data"""
        health_scores = []
        health_data = components.get('health', {})
        if health_data:
            success_rate = health_data.get('success_rate', 0.0)
            health_scores.append(success_rate)
        perf_data = components.get('performance', {})
        if perf_data:
            current_status = perf_data.get('current_status', {})
            status = current_status.get('status', 'unknown')
            if status == 'good':
                health_scores.append(1.0)
            elif status == 'warning':
                health_scores.append(0.7)
            else:
                health_scores.append(0.3)
        drift_data = components.get('drift', {})
        if drift_data:
            risk_level = drift_data.get('risk_level', 'unknown')
            if risk_level == 'low':
                health_scores.append(1.0)
            elif risk_level == 'medium':
                health_scores.append(0.6)
            else:
                health_scores.append(0.2)
        if health_scores:
            avg_score = sum(health_scores) / len(health_scores)
            return avg_score > 0.7
        return False

    def _generate_system_alerts(self, dashboard: dict[str, Any]) -> list:
        """Generate system-wide alerts"""
        alerts = []
        system_health = dashboard.get('system_health', {})
        if not system_health.get('healthy', True):
            alerts.append({'level': 'critical', 'component': 'system', 'message': 'Overall ML system health is degraded', 'timestamp': aware_utc_now().isoformat()})
        perf_overview = dashboard.get('performance_overview', {})
        models_with_issues = perf_overview.get('models_with_issues', 0)
        if models_with_issues > 0:
            alerts.append({'level': 'warning', 'component': 'performance', 'message': f'{models_with_issues} models have performance issues', 'timestamp': aware_utc_now().isoformat()})
        drift_overview = dashboard.get('drift_overview', {})
        high_risk_models = drift_overview.get('high_risk_models', 0)
        if high_risk_models > 0:
            alerts.append({'level': 'warning', 'component': 'drift', 'message': f'{high_risk_models} models have high drift risk', 'timestamp': aware_utc_now().isoformat()})
        return alerts

    def _generate_system_recommendations(self, dashboard: dict[str, Any]) -> list:
        """Generate system-wide recommendations"""
        recommendations = []
        system_health = dashboard.get('system_health', {})
        if system_health.get('health_score', 1.0) < 0.7:
            recommendations.append('Review individual model health metrics and address issues')
        perf_overview = dashboard.get('performance_overview', {})
        if perf_overview.get('models_with_issues', 0) > 0:
            recommendations.append('Investigate performance issues in affected models')
        drift_overview = dashboard.get('drift_overview', {})
        if drift_overview.get('models_with_drift', 0) > 0:
            recommendations.append('Consider retraining models with detected drift')
        if not recommendations:
            recommendations.append('ML system health is optimal - continue monitoring')
        return recommendations
_integration_manager: MLHealthIntegrationService | None = None

async def get_ml_health_integration_manager() -> MLHealthIntegrationService:
    """Get or create global ML health integration manager"""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = MLHealthIntegrationService()
        await _integration_manager.initialize()
    return _integration_manager
