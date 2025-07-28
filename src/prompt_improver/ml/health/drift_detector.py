"""
Model Drift Detection - 2025 Best Practices

Comprehensive drift detection for ML models including:
- Prediction distribution drift detection
- Confidence score distribution monitoring  
- Statistical tests for distribution changes
- Early warning system for model degradation
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from ...utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)


@dataclass
class DriftMetrics:
    """Drift detection metrics for a model"""
    model_id: str
    timestamp: datetime
    
    # Distribution metrics
    prediction_drift_score: float
    confidence_drift_score: float
    
    # Statistical test results
    ks_statistic: float
    ks_p_value: float
    js_divergence: float
    
    # Alerts
    drift_detected: bool
    alert_level: str  # "none", "warning", "critical"
    alert_message: Optional[str] = None
    
    # Sample counts
    baseline_samples: int = 0
    current_samples: int = 0


@dataclass 
class PredictionWindow:
    """Rolling window of predictions for drift analysis"""
    predictions: deque = field(default_factory=lambda: deque(maxlen=1000))
    confidences: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_prediction(self, prediction: float, confidence: float, timestamp: datetime):
        """Add a new prediction to the window"""
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        self.timestamps.append(timestamp)
    
    def get_recent_window(self, hours: int = 24) -> Tuple[List[float], List[float]]:
        """Get predictions and confidences from the last N hours"""
        cutoff_time = aware_utc_now() - timedelta(hours=hours)
        
        recent_predictions = []
        recent_confidences = []
        
        for i, timestamp in enumerate(self.timestamps):
            if timestamp > cutoff_time:
                recent_predictions.append(self.predictions[i])
                recent_confidences.append(self.confidences[i])
        
        return recent_predictions, recent_confidences


class ModelDriftDetector:
    """
    Advanced model drift detection system.
    
    Monitors prediction distributions, confidence scores, and statistical properties
    to detect model drift and performance degradation.
    """
    
    def __init__(
        self,
        baseline_window_hours: int = 168,  # 1 week
        detection_window_hours: int = 24,   # 1 day
        drift_threshold: float = 0.1,
        confidence_threshold: float = 0.05
    ):
        self.baseline_window_hours = baseline_window_hours
        self.detection_window_hours = detection_window_hours
        self.drift_threshold = drift_threshold
        self.confidence_threshold = confidence_threshold
        
        # Model-specific prediction windows
        self._model_windows: Dict[str, PredictionWindow] = defaultdict(PredictionWindow)
        self._baseline_distributions: Dict[str, Dict[str, Any]] = {}
        
        # Drift history
        self._drift_history: Dict[str, List[DriftMetrics]] = defaultdict(list)
        
        logger.info("Model Drift Detector initialized")
    
    async def record_prediction(
        self,
        model_id: str,
        prediction: float,
        confidence: float,
        features: Optional[List[float]] = None
    ) -> None:
        """Record a prediction for drift monitoring"""
        try:
            timestamp = aware_utc_now()
            
            # Add to model window
            window = self._model_windows[model_id]
            window.add_prediction(prediction, confidence, timestamp)
            
            # Update baseline if we have enough data and no baseline exists
            if (model_id not in self._baseline_distributions and 
                len(window.predictions) >= 100):  # Minimum samples for baseline
                await self._establish_baseline(model_id)
            
        except Exception as e:
            logger.error(f"Failed to record prediction for drift detection: {e}")
    
    async def detect_drift(self, model_id: str) -> Optional[DriftMetrics]:
        """Detect drift for a specific model"""
        try:
            if model_id not in self._model_windows:
                return None
            
            window = self._model_windows[model_id]
            if len(window.predictions) < 50:  # Need minimum samples
                return None
            
            # Get baseline and current distributions
            baseline = self._baseline_distributions.get(model_id)
            if not baseline:
                await self._establish_baseline(model_id)
                baseline = self._baseline_distributions.get(model_id)
                if not baseline:
                    return None
            
            # Get recent predictions for comparison
            recent_predictions, recent_confidences = window.get_recent_window(
                self.detection_window_hours
            )
            
            if len(recent_predictions) < 20:  # Need minimum recent samples
                return None
            
            # Calculate drift metrics
            drift_metrics = await self._calculate_drift_metrics(
                model_id=model_id,
                baseline_predictions=baseline["predictions"],
                baseline_confidences=baseline["confidences"], 
                current_predictions=recent_predictions,
                current_confidences=recent_confidences
            )
            
            # Store in history
            self._drift_history[model_id].append(drift_metrics)
            
            # Keep only recent history (last 30 days)
            cutoff_time = aware_utc_now() - timedelta(days=30)
            self._drift_history[model_id] = [
                m for m in self._drift_history[model_id]
                if m.timestamp > cutoff_time
            ]
            
            return drift_metrics
            
        except Exception as e:
            logger.error(f"Failed to detect drift for model {model_id}: {e}")
            return None
    
    async def get_drift_status(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive drift status for a model"""
        try:
            # Get latest drift metrics
            latest_drift = await self.detect_drift(model_id)
            
            # Get drift history
            history = self._drift_history.get(model_id, [])
            recent_history = [
                m for m in history
                if m.timestamp > aware_utc_now() - timedelta(days=7)
            ]
            
            # Calculate trend
            drift_trend = self._calculate_drift_trend(recent_history)
            
            # Risk assessment
            risk_level = self._assess_drift_risk(latest_drift, recent_history)
            
            return {
                "model_id": model_id,
                "timestamp": aware_utc_now().isoformat(),
                "current_drift": latest_drift.__dict__ if latest_drift else None,
                "risk_level": risk_level,
                "drift_trend": drift_trend,
                "baseline_established": model_id in self._baseline_distributions,
                "recent_drift_events": len([
                    m for m in recent_history if m.drift_detected
                ]),
                "total_samples": len(self._model_windows[model_id].predictions) if model_id in self._model_windows else 0,
                "recommendations": self._generate_drift_recommendations(
                    latest_drift, risk_level, drift_trend
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get drift status for {model_id}: {e}")
            return {
                "model_id": model_id,
                "error": str(e),
                "timestamp": aware_utc_now().isoformat()
            }
    
    async def get_all_models_drift_status(self) -> List[Dict[str, Any]]:
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
            
            # Use older data as baseline (first 70% of available data)
            total_samples = len(window.predictions)
            baseline_size = int(total_samples * 0.7)
            
            if baseline_size < 50:  # Need minimum baseline samples
                return False
            
            baseline_predictions = list(window.predictions)[:baseline_size]
            baseline_confidences = list(window.confidences)[:baseline_size]
            
            # Calculate baseline statistics
            baseline_stats = {
                "predictions": baseline_predictions,
                "confidences": baseline_confidences,
                "prediction_mean": np.mean(baseline_predictions),
                "prediction_std": np.std(baseline_predictions),
                "confidence_mean": np.mean(baseline_confidences),
                "confidence_std": np.std(baseline_confidences),
                "established_at": aware_utc_now(),
                "sample_count": len(baseline_predictions)
            }
            
            self._baseline_distributions[model_id] = baseline_stats
            
            logger.info(f"Established baseline for model {model_id} with {baseline_size} samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish baseline for {model_id}: {e}")
            return False
    
    async def _calculate_drift_metrics(
        self,
        model_id: str,
        baseline_predictions: List[float],
        baseline_confidences: List[float],
        current_predictions: List[float],
        current_confidences: List[float]
    ) -> DriftMetrics:
        """Calculate comprehensive drift metrics"""
        
        # Kolmogorov-Smirnov test for prediction distribution
        ks_stat, ks_p_value = stats.ks_2samp(baseline_predictions, current_predictions)
        
        # Jensen-Shannon divergence
        js_divergence = self._calculate_js_divergence(
            baseline_predictions, current_predictions
        )
        
        # Custom drift scores
        prediction_drift_score = self._calculate_distribution_shift(
            baseline_predictions, current_predictions
        )
        confidence_drift_score = self._calculate_distribution_shift(
            baseline_confidences, current_confidences
        )
        
        # Determine if drift is detected
        drift_detected = (
            prediction_drift_score > self.drift_threshold or
            confidence_drift_score > self.confidence_threshold or
            ks_p_value < 0.05  # Significant distribution difference
        )
        
        # Determine alert level
        alert_level = "none"
        alert_message = None
        
        if drift_detected:
            if prediction_drift_score > self.drift_threshold * 2:
                alert_level = "critical"
                alert_message = f"Critical prediction drift detected (score: {prediction_drift_score:.3f})"
            else:
                alert_level = "warning"
                alert_message = f"Prediction drift warning (score: {prediction_drift_score:.3f})"
        
        return DriftMetrics(
            model_id=model_id,
            timestamp=aware_utc_now(),
            prediction_drift_score=prediction_drift_score,
            confidence_drift_score=confidence_drift_score,
            ks_statistic=ks_stat,
            ks_p_value=ks_p_value,
            js_divergence=js_divergence,
            drift_detected=drift_detected,
            alert_level=alert_level,
            alert_message=alert_message,
            baseline_samples=len(baseline_predictions),
            current_samples=len(current_predictions)
        )
    
    def _calculate_distribution_shift(
        self, baseline: List[float], current: List[float]
    ) -> float:
        """Calculate distribution shift score between two distributions"""
        try:
            # Convert to numpy arrays
            baseline_arr = np.array(baseline)
            current_arr = np.array(current)
            
            # Calculate means and standard deviations
            baseline_mean = np.mean(baseline_arr)
            baseline_std = np.std(baseline_arr)
            current_mean = np.mean(current_arr)
            current_std = np.std(current_arr)
            
            # Avoid division by zero
            if baseline_std == 0 or current_std == 0:
                return abs(baseline_mean - current_mean)
            
            # Normalize difference by baseline standard deviation
            mean_shift = abs(baseline_mean - current_mean) / baseline_std
            std_shift = abs(baseline_std - current_std) / baseline_std
            
            # Combined score
            return (mean_shift + std_shift) / 2
            
        except Exception as e:
            logger.error(f"Failed to calculate distribution shift: {e}")
            return 0.0
    
    def _calculate_js_divergence(
        self, baseline: List[float], current: List[float]
    ) -> float:
        """Calculate Jensen-Shannon divergence between two distributions"""
        try:
            # Create histograms
            combined_data = baseline + current
            bins = np.histogram_bin_edges(combined_data, bins=20)
            
            baseline_hist, _ = np.histogram(baseline, bins=bins, density=True)
            current_hist, _ = np.histogram(current, bins=bins, density=True)
            
            # Normalize to probabilities
            baseline_hist = baseline_hist / np.sum(baseline_hist)
            current_hist = current_hist / np.sum(current_hist)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            baseline_hist = baseline_hist + epsilon
            current_hist = current_hist + epsilon
            
            # Calculate JS divergence
            m = 0.5 * (baseline_hist + current_hist)
            js_div = 0.5 * stats.entropy(baseline_hist, m) + 0.5 * stats.entropy(current_hist, m)
            
            return float(js_div)
            
        except Exception as e:
            logger.error(f"Failed to calculate JS divergence: {e}")
            return 0.0
    
    def _calculate_drift_trend(self, recent_history: List[DriftMetrics]) -> Dict[str, Any]:
        """Calculate drift trend from recent history"""
        if len(recent_history) < 2:
            return {"trend": "insufficient_data", "slope": 0.0}
        
        try:
            # Sort by timestamp
            sorted_history = sorted(recent_history, key=lambda x: x.timestamp)
            
            # Extract drift scores over time
            timestamps = [(h.timestamp - sorted_history[0].timestamp).total_seconds() 
                         for h in sorted_history]
            drift_scores = [h.prediction_drift_score for h in sorted_history]
            
            # Calculate linear regression slope
            if len(timestamps) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    timestamps, drift_scores
                )
                
                # Determine trend
                if abs(slope) < 0.001:
                    trend = "stable"
                elif slope > 0:
                    trend = "increasing"
                else:
                    trend = "decreasing"
                
                return {
                    "trend": trend,
                    "slope": float(slope),
                    "r_squared": float(r_value ** 2),
                    "p_value": float(p_value)
                }
            
            return {"trend": "stable", "slope": 0.0}
            
        except Exception as e:
            logger.error(f"Failed to calculate drift trend: {e}")
            return {"trend": "unknown", "slope": 0.0}
    
    def _assess_drift_risk(
        self, 
        latest_drift: Optional[DriftMetrics], 
        recent_history: List[DriftMetrics]
    ) -> str:
        """Assess overall drift risk level"""
        if not latest_drift:
            return "unknown"
        
        # Base risk on latest drift
        if latest_drift.alert_level == "critical":
            return "high"
        elif latest_drift.alert_level == "warning":
            risk = "medium"
        else:
            risk = "low"
        
        # Adjust based on frequency of drift events
        if len(recent_history) > 0:
            drift_events = [h for h in recent_history if h.drift_detected]
            drift_frequency = len(drift_events) / len(recent_history)
            
            if drift_frequency > 0.5:  # More than 50% of recent checks show drift
                if risk == "low":
                    risk = "medium"
                elif risk == "medium":
                    risk = "high"
        
        return risk
    
    def _generate_drift_recommendations(
        self,
        latest_drift: Optional[DriftMetrics],
        risk_level: str,
        drift_trend: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on drift analysis"""
        recommendations = []
        
        if not latest_drift:
            recommendations.append("Insufficient data for drift analysis. Continue monitoring.")
            return recommendations
        
        if risk_level == "high":
            recommendations.extend([
                "🔴 High drift risk detected - consider model retraining",
                "Review recent data quality and feature distributions",
                "Implement gradual rollback if performance is degrading"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "🟡 Moderate drift detected - monitor closely",
                "Consider collecting additional validation data",
                "Prepare for potential model update"
            ])
        else:
            recommendations.append("🟢 Low drift risk - continue normal monitoring")
        
        # Trend-based recommendations
        trend = drift_trend.get("trend", "unknown")
        if trend == "increasing":
            recommendations.append("📈 Drift trend is increasing - proactive intervention recommended")
        elif trend == "decreasing":
            recommendations.append("📉 Drift trend is decreasing - improvements detected")
        
        # Statistical recommendations
        if latest_drift.ks_p_value < 0.01:
            recommendations.append("📊 Strong statistical evidence of distribution change")
        
        return recommendations
    
    async def reset_baseline(self, model_id: str) -> bool:
        """Reset baseline distribution for a model"""
        try:
            if model_id in self._baseline_distributions:
                del self._baseline_distributions[model_id]
            
            # Clear drift history
            if model_id in self._drift_history:
                self._drift_history[model_id] = []
            
            # Re-establish baseline if we have enough data
            if model_id in self._model_windows:
                return await self._establish_baseline(model_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset baseline for {model_id}: {e}")
            return False


# Global drift detector instance
_drift_detector: Optional[ModelDriftDetector] = None

async def get_drift_detector() -> ModelDriftDetector:
    """Get or create global drift detector instance"""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = ModelDriftDetector()
    return _drift_detector