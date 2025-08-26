"""
Model Drift Detection - 2025 Best Practices

Comprehensive drift detection for ML models including:
- Prediction distribution drift detection
- Confidence score distribution monitoring  
- Statistical tests for distribution changes
- Early warning system for model degradation
"""
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from sqlmodel import SQLModel, Field
from pydantic import BaseModel
# import numpy as np  # Converted to lazy loading
# from scipy import stats  # Converted to lazy loading
from ...utils.datetime_utils import aware_utc_now
from prompt_improver.core.utils.lazy_ml_loader import get_numpy
from prompt_improver.core.utils.lazy_ml_loader import get_scipy_stats
logger = logging.getLogger(__name__)

class DriftMetrics(BaseModel):
    """Drift detection metrics for a model"""
    model_id: str = Field(description='Model identifier')
    timestamp: datetime = Field(description='Drift analysis timestamp')
    prediction_drift_score: float = Field(ge=0.0, description='Prediction distribution drift score')
    confidence_drift_score: float = Field(ge=0.0, description='Confidence distribution drift score')
    ks_statistic: float = Field(ge=0.0, le=1.0, description='Kolmogorov-Smirnov test statistic')
    ks_p_value: float = Field(ge=0.0, le=1.0, description='KS test p-value')
    js_divergence: float = Field(ge=0.0, description='Jensen-Shannon divergence')
    drift_detected: bool = Field(description='Whether drift was detected')
    alert_level: str = Field(description='Alert level: none, warning, or critical')
    alert_message: str | None = Field(default=None, description='Alert message if applicable')
    baseline_samples: int = Field(default=0, ge=0, description='Number of baseline samples')
    current_samples: int = Field(default=0, ge=0, description='Number of current window samples')

class PredictionWindow(BaseModel):
    """Rolling window of predictions for drift analysis"""
    predictions: list[float] = Field(default_factory=list, description='Prediction values in window')
    confidences: list[float] = Field(default_factory=list, description='Confidence scores in window')
    timestamps: list[datetime] = Field(default_factory=list, description='Timestamps for each prediction')
    max_size: int = Field(default=1000, ge=100, description='Maximum window size')

    def _deque_from_list(self, data: list[Any]) -> deque:
        """Convert list to deque with max size for internal use"""
        return deque(data, maxlen=self.max_size)

    def add_prediction(self, prediction: float, confidence: float, timestamp: datetime):
        """Add a new prediction to the window"""
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        self.timestamps.append(timestamp)

    def get_recent_window(self, hours: int=24) -> tuple[list[float], list[float]]:
        """Get predictions and confidences from the last N hours"""
        cutoff_time = aware_utc_now() - timedelta(hours=hours)
        recent_predictions = []
        recent_confidences = []
        for i, timestamp in enumerate(self.timestamps):
            if timestamp > cutoff_time:
                recent_predictions.append(self.predictions[i])
                recent_confidences.append(self.confidences[i])
        return (recent_predictions, recent_confidences)

class ModelDriftDetector:
    """
    Advanced model drift detection system.
    
    Monitors prediction distributions, confidence scores, and statistical properties
    to detect model drift and performance degradation.
    """

    def __init__(self, baseline_window_hours: int=168, detection_window_hours: int=24, drift_threshold: float=0.1, confidence_threshold: float=0.05):
        self.baseline_window_hours = baseline_window_hours
        self.detection_window_hours = detection_window_hours
        self.drift_threshold = drift_threshold
        self.confidence_threshold = confidence_threshold
        self._model_windows: dict[str, PredictionWindow] = defaultdict(PredictionWindow)
        self._baseline_distributions: dict[str, dict[str, Any]] = {}
        self._drift_history: dict[str, list[DriftMetrics]] = defaultdict(list)
        logger.info('Model Drift Detector initialized')

    async def record_prediction(self, model_id: str, prediction: float, confidence: float, features: list[float] | None=None) -> None:
        """Record a prediction for drift monitoring"""
        try:
            timestamp = aware_utc_now()
            window = self._model_windows[model_id]
            window.add_prediction(prediction, confidence, timestamp)
            if model_id not in self._baseline_distributions and len(window.predictions) >= 100:
                await self._establish_baseline(model_id)
        except Exception as e:
            logger.error('Failed to record prediction for drift detection: %s', e)

    async def detect_drift(self, model_id: str) -> DriftMetrics | None:
        """Detect drift for a specific model"""
        try:
            if model_id not in self._model_windows:
                return None
            window = self._model_windows[model_id]
            if len(window.predictions) < 50:
                return None
            baseline = self._baseline_distributions.get(model_id)
            if not baseline:
                await self._establish_baseline(model_id)
                baseline = self._baseline_distributions.get(model_id)
                if not baseline:
                    return None
            recent_predictions, recent_confidences = window.get_recent_window(self.detection_window_hours)
            if len(recent_predictions) < 20:
                return None
            drift_metrics = await self._calculate_drift_metrics(model_id=model_id, baseline_predictions=baseline['predictions'], baseline_confidences=baseline['confidences'], current_predictions=recent_predictions, current_confidences=recent_confidences)
            self._drift_history[model_id].append(drift_metrics)
            cutoff_time = aware_utc_now() - timedelta(days=30)
            self._drift_history[model_id] = [m for m in self._drift_history[model_id] if m.timestamp > cutoff_time]
            return drift_metrics
        except Exception as e:
            logger.error('Failed to detect drift for model {model_id}: %s', e)
            return None

    async def get_drift_status(self, model_id: str) -> dict[str, Any]:
        """Get comprehensive drift status for a model"""
        try:
            latest_drift = await self.detect_drift(model_id)
            history = self._drift_history.get(model_id, [])
            recent_history = [m for m in history if m.timestamp > aware_utc_now() - timedelta(days=7)]
            drift_trend = self._calculate_drift_trend(recent_history)
            risk_level = self._assess_drift_risk(latest_drift, recent_history)
            return {'model_id': model_id, 'timestamp': aware_utc_now().isoformat(), 'current_drift': latest_drift.__dict__ if latest_drift else None, 'risk_level': risk_level, 'drift_trend': drift_trend, 'baseline_established': model_id in self._baseline_distributions, 'recent_drift_events': len([m for m in recent_history if m.drift_detected]), 'total_samples': len(self._model_windows[model_id].predictions) if model_id in self._model_windows else 0, 'recommendations': self._generate_drift_recommendations(latest_drift, risk_level, drift_trend)}
        except Exception as e:
            logger.error('Failed to get drift status for {model_id}: %s', e)
            return {'model_id': model_id, 'error': str(e), 'timestamp': aware_utc_now().isoformat()}

    async def get_all_models_drift_status(self) -> list[dict[str, Any]]:
        """Get drift status for all monitored models"""
        drift_statuses = []
        for model_id in self._model_windows.keys():
            status = await self.get_drift_status(model_id)
            drift_statuses.append(status)
        return drift_statuses

    async def _establish_baseline(self, model_id: str) -> bool:
        """Establish baseline distribution for a model"""
        try:
            window = self._model_windows[model_id]
            total_samples = len(window.predictions)
            baseline_size = int(total_samples * 0.7)
            if baseline_size < 50:
                return False
            baseline_predictions = list(window.predictions)[:baseline_size]
            baseline_confidences = list(window.confidences)[:baseline_size]
            baseline_stats = {'predictions': baseline_predictions, 'confidences': baseline_confidences, 'prediction_mean': get_numpy().mean(baseline_predictions), 'prediction_std': get_numpy().std(baseline_predictions), 'confidence_mean': get_numpy().mean(baseline_confidences), 'confidence_std': get_numpy().std(baseline_confidences), 'established_at': aware_utc_now(), 'sample_count': len(baseline_predictions)}
            self._baseline_distributions[model_id] = baseline_stats
            logger.info('Established baseline for model {model_id} with %s samples', baseline_size)
            return True
        except Exception as e:
            logger.error('Failed to establish baseline for {model_id}: %s', e)
            return False

    async def _calculate_drift_metrics(self, model_id: str, baseline_predictions: list[float], baseline_confidences: list[float], current_predictions: list[float], current_confidences: list[float]) -> DriftMetrics:
        """Calculate comprehensive drift metrics"""
        ks_stat, ks_p_value = get_scipy_stats().ks_2samp(baseline_predictions, current_predictions)
        js_divergence = self._calculate_js_divergence(baseline_predictions, current_predictions)
        prediction_drift_score = self._calculate_distribution_shift(baseline_predictions, current_predictions)
        confidence_drift_score = self._calculate_distribution_shift(baseline_confidences, current_confidences)
        drift_detected = prediction_drift_score > self.drift_threshold or confidence_drift_score > self.confidence_threshold or ks_p_value < 0.05
        alert_level = 'none'
        alert_message = None
        if drift_detected:
            if prediction_drift_score > self.drift_threshold * 2:
                alert_level = 'critical'
                alert_message = f'Critical prediction drift detected (score: {prediction_drift_score:.3f})'
            else:
                alert_level = 'warning'
                alert_message = f'Prediction drift warning (score: {prediction_drift_score:.3f})'
        return DriftMetrics(model_id=model_id, timestamp=aware_utc_now(), prediction_drift_score=prediction_drift_score, confidence_drift_score=confidence_drift_score, ks_statistic=ks_stat, ks_p_value=ks_p_value, js_divergence=js_divergence, drift_detected=drift_detected, alert_level=alert_level, alert_message=alert_message, baseline_samples=len(baseline_predictions), current_samples=len(current_predictions))

    def _calculate_distribution_shift(self, baseline: list[float], current: list[float]) -> float:
        """Calculate distribution shift score between two distributions"""
        try:
            baseline_arr = get_numpy().array(baseline)
            current_arr = get_numpy().array(current)
            baseline_mean = get_numpy().mean(baseline_arr)
            baseline_std = get_numpy().std(baseline_arr)
            current_mean = get_numpy().mean(current_arr)
            current_std = get_numpy().std(current_arr)
            if baseline_std == 0 or current_std == 0:
                return abs(baseline_mean - current_mean)
            mean_shift = abs(baseline_mean - current_mean) / baseline_std
            std_shift = abs(baseline_std - current_std) / baseline_std
            return (mean_shift + std_shift) / 2
        except Exception as e:
            logger.error('Failed to calculate distribution shift: %s', e)
            return 0.0

    def _calculate_js_divergence(self, baseline: list[float], current: list[float]) -> float:
        """Calculate Jensen-Shannon divergence between two distributions"""
        try:
            combined_data = baseline + current
            bins = get_numpy().histogram_bin_edges(combined_data, bins=20)
            baseline_hist, _ = get_numpy().histogram(baseline, bins=bins, density=True)
            current_hist, _ = get_numpy().histogram(current, bins=bins, density=True)
            baseline_hist = baseline_hist / get_numpy().sum(baseline_hist)
            current_hist = current_hist / get_numpy().sum(current_hist)
            epsilon = 1e-10
            baseline_hist = baseline_hist + epsilon
            current_hist = current_hist + epsilon
            m = 0.5 * (baseline_hist + current_hist)
            js_div = 0.5 * get_scipy_stats().entropy(baseline_hist, m) + 0.5 * get_scipy_stats().entropy(current_hist, m)
            return float(js_div)
        except Exception as e:
            logger.error('Failed to calculate JS divergence: %s', e)
            return 0.0

    def _calculate_drift_trend(self, recent_history: list[DriftMetrics]) -> dict[str, Any]:
        """Calculate drift trend from recent history"""
        if len(recent_history) < 2:
            return {'trend': 'insufficient_data', 'slope': 0.0}
        try:
            sorted_history = sorted(recent_history, key=lambda x: x.timestamp)
            timestamps = [(h.timestamp - sorted_history[0].timestamp).total_seconds() for h in sorted_history]
            drift_scores = [h.prediction_drift_score for h in sorted_history]
            if len(timestamps) > 1:
                slope, intercept, r_value, p_value, std_err = get_scipy_stats().linregress(timestamps, drift_scores)
                if abs(slope) < 0.001:
                    trend = 'stable'
                elif slope > 0:
                    trend = 'increasing'
                else:
                    trend = 'decreasing'
                return {'trend': trend, 'slope': float(slope), 'r_squared': float(r_value ** 2), 'p_value': float(p_value)}
            return {'trend': 'stable', 'slope': 0.0}
        except Exception as e:
            logger.error('Failed to calculate drift trend: %s', e)
            return {'trend': 'unknown', 'slope': 0.0}

    def _assess_drift_risk(self, latest_drift: DriftMetrics | None, recent_history: list[DriftMetrics]) -> str:
        """Assess overall drift risk level"""
        if not latest_drift:
            return 'unknown'
        if latest_drift.alert_level == 'critical':
            return 'high'
        elif latest_drift.alert_level == 'warning':
            risk = 'medium'
        else:
            risk = 'low'
        if len(recent_history) > 0:
            drift_events = [h for h in recent_history if h.drift_detected]
            drift_frequency = len(drift_events) / len(recent_history)
            if drift_frequency > 0.5:
                if risk == 'low':
                    risk = 'medium'
                elif risk == 'medium':
                    risk = 'high'
        return risk

    def _generate_drift_recommendations(self, latest_drift: DriftMetrics | None, risk_level: str, drift_trend: dict[str, Any]) -> list[str]:
        """Generate actionable recommendations based on drift analysis"""
        recommendations = []
        if not latest_drift:
            recommendations.append('Insufficient data for drift analysis. Continue monitoring.')
            return recommendations
        if risk_level == 'high':
            recommendations.extend(['🔴 High drift risk detected - consider model retraining', 'Review recent data quality and feature distributions', 'Implement gradual rollback if performance is degrading'])
        elif risk_level == 'medium':
            recommendations.extend(['🟡 Moderate drift detected - monitor closely', 'Consider collecting additional validation data', 'Prepare for potential model update'])
        else:
            recommendations.append('🟢 Low drift risk - continue normal monitoring')
        trend = drift_trend.get('trend', 'unknown')
        if trend == 'increasing':
            recommendations.append('📈 Drift trend is increasing - proactive intervention recommended')
        elif trend == 'decreasing':
            recommendations.append('📉 Drift trend is decreasing - improvements detected')
        if latest_drift.ks_p_value < 0.01:
            recommendations.append('📊 Strong statistical evidence of distribution change')
        return recommendations

    async def reset_baseline(self, model_id: str) -> bool:
        """Reset baseline distribution for a model"""
        try:
            if model_id in self._baseline_distributions:
                del self._baseline_distributions[model_id]
            if model_id in self._drift_history:
                self._drift_history[model_id] = []
            if model_id in self._model_windows:
                return await self._establish_baseline(model_id)
            return True
        except Exception as e:
            logger.error('Failed to reset baseline for {model_id}: %s', e)
            return False
_drift_detector: ModelDriftDetector | None = None

async def get_drift_detector() -> ModelDriftDetector:
    """Get or create global drift detector instance"""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = ModelDriftDetector()
    return _drift_detector