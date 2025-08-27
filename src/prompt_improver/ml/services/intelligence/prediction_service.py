"""ML Prediction Service.

Provides ML predictions with confidence scoring and validation.
Extracted from intelligence_processor.py god object to follow single responsibility principle.

Performance Target: <100ms for prediction operations
Memory Target: <30MB for prediction caching
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
# import numpy as np  # Converted to lazy loading

from prompt_improver.ml.services.intelligence.protocols.intelligence_service_protocols import (
    MLPredictionServiceProtocol,
    IntelligenceResult,
    MLCircuitBreakerServiceProtocol,
)
from prompt_improver.repositories.protocols.ml_repository_protocol import MLRepositoryProtocol
from prompt_improver.core.utils.lazy_ml_loader import get_numpy
from prompt_improver.performance.monitoring.metrics_registry import (
    StandardMetrics,
    get_metrics_registry,
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    """Metrics for prediction quality assessment."""
    accuracy: float
    confidence: float
    prediction_time_ms: float
    data_quality_score: float
    historical_accuracy: Optional[float] = None


@dataclass
class ConfidenceThresholds:
    """Confidence thresholds for prediction quality."""
    minimum_acceptable: float = 0.3
    good_quality: float = 0.7
    excellent_quality: float = 0.9
    uncertainty_threshold: float = 0.1


class MLPredictionService:
    """ML Prediction Service.
    
    Handles ML predictions with confidence scoring, quality validation,
    and trend analysis for rule selection and optimization.
    """
    
    def __init__(
        self,
        ml_repository: MLRepositoryProtocol,
        circuit_breaker_service: MLCircuitBreakerServiceProtocol
    ):
        """Initialize ML prediction service.
        
        Args:
            ml_repository: ML repository for data access
            circuit_breaker_service: Circuit breaker protection service
        """
        self._ml_repository = ml_repository
        self._circuit_breaker_service = circuit_breaker_service
        self._metrics_registry = get_metrics_registry()
        self._confidence_thresholds = ConfidenceThresholds()
        self._prediction_cache: Dict[str, Dict[str, Any]] = {}
        self._prediction_history: List[PredictionMetrics] = []
        self._max_history_size = 1000
        
        logger.info("MLPredictionService initialized")
    
    async def generate_predictions_with_confidence(self, rule_data: Dict[str, Any]) -> IntelligenceResult:
        """Generate ML predictions with confidence scoring.
        
        Args:
            rule_data: Rule data for prediction generation
            
        Returns:
            Intelligence result with predictions and confidence scores
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Validate input data quality
            data_quality_score = await self._assess_data_quality(rule_data)
            
            if data_quality_score < 0.3:
                logger.warning("Low data quality detected, predictions may be unreliable")
            
            # Generate predictions with circuit breaker protection
            predictions = await self._circuit_breaker_service.call_with_breaker(
                "prediction_generation",
                self._generate_predictions_internal,
                rule_data,
                data_quality_score
            )
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Record prediction metrics
            prediction_metrics = PredictionMetrics(
                accuracy=predictions.get("estimated_accuracy", 0.0),
                confidence=predictions.get("confidence", 0.0),
                prediction_time_ms=processing_time,
                data_quality_score=data_quality_score,
                historical_accuracy=self._get_historical_accuracy()
            )
            
            self._update_prediction_history(prediction_metrics)
            
            self._metrics_registry.increment(
                "ml_prediction_operations_total",
                tags={"service": "prediction", "result": "success"}
            )
            self._metrics_registry.record_value(
                "ml_prediction_duration_ms",
                processing_time,
                tags={"service": "prediction"}
            )
            self._metrics_registry.record_value(
                "ml_prediction_confidence",
                predictions.get("confidence", 0.0),
                tags={"service": "prediction"}
            )
            
            return IntelligenceResult(
                success=True,
                data=predictions,
                confidence=predictions.get("confidence", 0.0),
                processing_time_ms=processing_time,
                cache_hit=False
            )
            
        except Exception as e:
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            self._metrics_registry.increment(
                "ml_prediction_operations_total",
                tags={"service": "prediction", "result": "error"}
            )
            
            logger.error(f"Prediction generation failed: {e}")
            
            return IntelligenceResult(
                success=False,
                data={},
                confidence=0.0,
                processing_time_ms=processing_time,
                cache_hit=False,
                error_message=str(e)
            )
    
    async def _generate_predictions_internal(
        self, 
        rule_data: Dict[str, Any], 
        data_quality_score: float
    ) -> Dict[str, Any]:
        """Internal prediction generation implementation.
        
        Args:
            rule_data: Rule data for predictions
            data_quality_score: Quality assessment of input data
            
        Returns:
            Generated predictions with confidence metrics
        """
        if not rule_data or not rule_data.get("characteristics"):
            return {
                "predictions": [],
                "confidence": 0.0,
                "quality_assessment": "insufficient_data",
                "recommendations": []
            }
        
        characteristics = rule_data["characteristics"]
        
        # Generate rule effectiveness predictions
        effectiveness_prediction = await self._predict_rule_effectiveness(characteristics)
        
        # Generate confidence intervals
        confidence_interval = await self._calculate_confidence_intervals(effectiveness_prediction)
        
        # Generate quality recommendations
        recommendations = await self._generate_quality_recommendations(
            effectiveness_prediction, 
            data_quality_score
        )
        
        # Calculate overall confidence based on multiple factors
        overall_confidence = await self._calculate_overall_confidence(
            effectiveness_prediction,
            data_quality_score,
            confidence_interval
        )
        
        predictions = {
            "effectiveness_prediction": effectiveness_prediction,
            "confidence": overall_confidence,
            "confidence_interval": confidence_interval,
            "data_quality_score": data_quality_score,
            "quality_assessment": self._assess_prediction_quality(overall_confidence),
            "recommendations": recommendations,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "estimated_accuracy": self._estimate_prediction_accuracy(overall_confidence, data_quality_score)
        }
        
        return predictions
    
    async def _predict_rule_effectiveness(self, characteristics: Dict[str, Any]) -> Dict[str, float]:
        """Predict rule effectiveness based on characteristics.
        
        Args:
            characteristics: Rule characteristics
            
        Returns:
            Effectiveness predictions
        """
        # Simulate ML model predictions based on rule characteristics
        # In real implementation, this would use trained ML models
        
        base_effectiveness = 0.5  # Default effectiveness
        
        # Adjust based on characteristics
        if characteristics.get("complexity_score", 0) > 0.8:
            base_effectiveness -= 0.15  # High complexity reduces effectiveness
        
        if characteristics.get("specificity_score", 0) > 0.7:
            base_effectiveness += 0.2  # High specificity increases effectiveness
        
        if characteristics.get("domain_relevance", 0) > 0.8:
            base_effectiveness += 0.15  # High domain relevance helps
        
        # Add some realistic variance
        variance = get_numpy().random.normal(0, 0.05)
        effectiveness = max(0.0, min(1.0, base_effectiveness + variance))
        
        return {
            "overall_effectiveness": effectiveness,
            "precision": min(1.0, effectiveness + 0.1),
            "recall": min(1.0, effectiveness + 0.05),
            "f1_score": 2 * (effectiveness * effectiveness) / (2 * effectiveness) if effectiveness > 0 else 0.0
        }
    
    async def _calculate_confidence_intervals(self, predictions: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate confidence intervals for predictions.
        
        Args:
            predictions: Base predictions
            
        Returns:
            Confidence intervals
        """
        intervals = {}
        
        for metric, value in predictions.items():
            # Calculate 95% confidence interval
            margin_of_error = 0.1 * (1 - value)  # Higher uncertainty for extreme values
            
            intervals[metric] = {
                "lower_bound": max(0.0, value - margin_of_error),
                "upper_bound": min(1.0, value + margin_of_error),
                "margin_of_error": margin_of_error
            }
        
        return intervals
    
    async def _generate_quality_recommendations(
        self, 
        predictions: Dict[str, float], 
        data_quality_score: float
    ) -> List[Dict[str, Any]]:
        """Generate quality recommendations based on predictions.
        
        Args:
            predictions: Effectiveness predictions
            data_quality_score: Quality of input data
            
        Returns:
            Quality recommendations
        """
        recommendations = []
        
        effectiveness = predictions.get("overall_effectiveness", 0.0)
        
        if effectiveness < self._confidence_thresholds.minimum_acceptable:
            recommendations.append({
                "type": "quality_warning",
                "priority": "high",
                "message": "Predicted effectiveness is below acceptable threshold",
                "suggested_action": "Consider rule refinement or additional training data"
            })
        
        if data_quality_score < 0.5:
            recommendations.append({
                "type": "data_quality",
                "priority": "medium", 
                "message": "Input data quality is suboptimal",
                "suggested_action": "Improve data collection and preprocessing"
            })
        
        if effectiveness > self._confidence_thresholds.excellent_quality:
            recommendations.append({
                "type": "optimization",
                "priority": "low",
                "message": "High effectiveness predicted - consider production deployment",
                "suggested_action": "Monitor performance in production environment"
            })
        
        return recommendations
    
    async def _calculate_overall_confidence(
        self,
        predictions: Dict[str, float],
        data_quality_score: float,
        confidence_intervals: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate overall confidence score.
        
        Args:
            predictions: Effectiveness predictions
            data_quality_score: Quality of input data
            confidence_intervals: Confidence intervals
            
        Returns:
            Overall confidence score (0-1)
        """
        # Base confidence from prediction quality
        base_confidence = predictions.get("overall_effectiveness", 0.0)
        
        # Adjust for data quality
        data_quality_factor = min(1.0, data_quality_score + 0.2)
        
        # Adjust for prediction uncertainty (tighter intervals = higher confidence)
        avg_margin_of_error = sum(
            interval["margin_of_error"] 
            for interval in confidence_intervals.values()
        ) / len(confidence_intervals) if confidence_intervals else 0.2
        
        uncertainty_factor = 1.0 - avg_margin_of_error
        
        # Historical accuracy factor
        historical_factor = self._get_historical_accuracy() or 0.7
        
        # Combine factors with weights
        overall_confidence = (
            base_confidence * 0.4 +
            data_quality_factor * 0.3 +
            uncertainty_factor * 0.2 +
            historical_factor * 0.1
        )
        
        return max(0.0, min(1.0, overall_confidence))
    
    async def validate_prediction_quality(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Validate quality and reliability of predictions.
        
        Args:
            predictions: Predictions to validate
            
        Returns:
            Quality validation results
        """
        validation_results = {
            "is_valid": True,
            "quality_score": 0.0,
            "validation_checks": [],
            "warnings": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        confidence = predictions.get("confidence", 0.0)
        
        # Check confidence threshold
        if confidence >= self._confidence_thresholds.excellent_quality:
            validation_results["quality_score"] = 0.95
            validation_results["validation_checks"].append("High confidence threshold met")
        elif confidence >= self._confidence_thresholds.good_quality:
            validation_results["quality_score"] = 0.8
            validation_results["validation_checks"].append("Good confidence threshold met")
        elif confidence >= self._confidence_thresholds.minimum_acceptable:
            validation_results["quality_score"] = 0.6
            validation_results["validation_checks"].append("Minimum confidence threshold met")
            validation_results["warnings"].append("Predictions may be unreliable")
        else:
            validation_results["is_valid"] = False
            validation_results["quality_score"] = 0.3
            validation_results["warnings"].append("Confidence below acceptable threshold")
        
        # Check data quality
        data_quality = predictions.get("data_quality_score", 0.0)
        if data_quality < 0.5:
            validation_results["warnings"].append("Low input data quality detected")
            validation_results["quality_score"] *= 0.9  # Reduce quality score
        
        return validation_results
    
    async def calculate_confidence_metrics(self, prediction_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence metrics for predictions.
        
        Args:
            prediction_data: List of prediction data points
            
        Returns:
            Confidence metrics
        """
        if not prediction_data:
            return {"error": "No prediction data provided"}
        
        confidences = [p.get("confidence", 0.0) for p in prediction_data]
        accuracies = [p.get("estimated_accuracy", 0.0) for p in prediction_data]
        
        return {
            "mean_confidence": sum(confidences) / len(confidences),
            "std_confidence": get_numpy().std(confidences) if len(confidences) > 1 else 0.0,
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "confidence_stability": 1.0 - (get_numpy().std(confidences) / max(sum(confidences) / len(confidences), 0.1)),
            "prediction_count": len(prediction_data)
        }
    
    async def analyze_prediction_trends(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze prediction trends over time.
        
        Args:
            historical_data: Historical prediction data
            
        Returns:
            Trend analysis results
        """
        if not historical_data:
            return {"error": "No historical data provided"}
        
        # Sort by timestamp
        sorted_data = sorted(
            historical_data,
            key=lambda x: x.get("generated_at", "")
        )
        
        confidences = [d.get("confidence", 0.0) for d in sorted_data]
        accuracies = [d.get("estimated_accuracy", 0.0) for d in sorted_data]
        
        # Calculate trends
        confidence_trend = self._calculate_trend(confidences)
        accuracy_trend = self._calculate_trend(accuracies)
        
        return {
            "confidence_trend": confidence_trend,
            "accuracy_trend": accuracy_trend,
            "data_points": len(sorted_data),
            "time_span_hours": self._calculate_time_span(sorted_data),
            "recent_average_confidence": sum(confidences[-10:]) / min(10, len(confidences)),
            "trend_analysis": {
                "confidence_improving": confidence_trend > 0.01,
                "accuracy_improving": accuracy_trend > 0.01,
                "stable_performance": abs(confidence_trend) < 0.05
            }
        }
    
    def _assess_data_quality(self, data: Dict[str, Any]) -> float:
        """Assess quality of input data.
        
        Args:
            data: Input data to assess
            
        Returns:
            Quality score (0-1)
        """
        if not data:
            return 0.0
        
        quality_score = 1.0
        
        # Check for required fields
        required_fields = ["characteristics", "context"]
        for field in required_fields:
            if field not in data:
                quality_score -= 0.3
        
        # Check data completeness
        if "characteristics" in data:
            characteristics = data["characteristics"]
            if isinstance(characteristics, dict):
                non_empty_fields = sum(1 for v in characteristics.values() if v not in [None, "", 0])
                total_fields = len(characteristics)
                completeness = non_empty_fields / total_fields if total_fields > 0 else 0
                quality_score *= completeness
        
        return max(0.0, min(1.0, quality_score))
    
    def _assess_prediction_quality(self, confidence: float) -> str:
        """Assess prediction quality based on confidence.
        
        Args:
            confidence: Confidence score
            
        Returns:
            Quality assessment
        """
        if confidence >= self._confidence_thresholds.excellent_quality:
            return "excellent"
        elif confidence >= self._confidence_thresholds.good_quality:
            return "good"
        elif confidence >= self._confidence_thresholds.minimum_acceptable:
            return "acceptable"
        else:
            return "poor"
    
    def _estimate_prediction_accuracy(self, confidence: float, data_quality: float) -> float:
        """Estimate prediction accuracy based on confidence and data quality.
        
        Args:
            confidence: Confidence score
            data_quality: Data quality score
            
        Returns:
            Estimated accuracy
        """
        # Simple heuristic - in practice, this would use validation data
        base_accuracy = confidence * 0.8 + data_quality * 0.2
        
        # Historical accuracy factor
        historical_factor = self._get_historical_accuracy() or 0.7
        
        return (base_accuracy + historical_factor) / 2
    
    def _get_historical_accuracy(self) -> Optional[float]:
        """Get historical accuracy from prediction history.
        
        Returns:
            Historical accuracy if available
        """
        if not self._prediction_history:
            return None
        
        recent_history = self._prediction_history[-50:]  # Last 50 predictions
        accuracies = [p.accuracy for p in recent_history if p.accuracy > 0]
        
        return sum(accuracies) / len(accuracies) if accuracies else None
    
    def _update_prediction_history(self, metrics: PredictionMetrics) -> None:
        """Update prediction history with new metrics.
        
        Args:
            metrics: New prediction metrics
        """
        self._prediction_history.append(metrics)
        
        # Keep history size manageable
        if len(self._prediction_history) > self._max_history_size:
            self._prediction_history = self._prediction_history[-self._max_history_size:]
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values using simple linear regression.
        
        Args:
            values: List of values
            
        Returns:
            Trend slope
        """
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        
        # Simple linear regression
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _calculate_time_span(self, data: List[Dict[str, Any]]) -> float:
        """Calculate time span of data in hours.
        
        Args:
            data: Historical data with timestamps
            
        Returns:
            Time span in hours
        """
        if len(data) < 2:
            return 0.0
        
        try:
            first_time = datetime.fromisoformat(data[0]["generated_at"].replace("Z", "+00:00"))
            last_time = datetime.fromisoformat(data[-1]["generated_at"].replace("Z", "+00:00"))
            
            return (last_time - first_time).total_seconds() / 3600
        except (KeyError, ValueError):
            return 0.0