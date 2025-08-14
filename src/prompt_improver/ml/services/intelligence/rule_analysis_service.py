"""Rule Analysis Service for ML Intelligence Processing.

Implements focused rule effectiveness analysis extracted from intelligence_processor.py.
Follows clean architecture with protocol-based DI and performance optimization.
Target: <50ms for rule analysis operations.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

from ....repositories.protocols.ml_repository_protocol import MLRepositoryProtocol
from ....ml.learning.patterns.advanced_pattern_discovery import AdvancedPatternDiscovery
from ....ml.optimization.algorithms.rule_optimizer import RuleOptimizer
from .protocols.intelligence_service_protocols import (
    RuleAnalysisServiceProtocol,
    MLCircuitBreakerServiceProtocol,
    IntelligenceResult,
)

logger = logging.getLogger(__name__)


class RuleAnalysisService:
    """Service for ML rule effectiveness analysis.
    
    Extracted from intelligence_processor.py lines 295-646 for focused responsibility.
    Implements clean architecture patterns with repository-based data access.
    """

    def __init__(
        self,
        ml_repository: MLRepositoryProtocol,
        pattern_discovery: AdvancedPatternDiscovery,
        rule_optimizer: RuleOptimizer,
        circuit_breaker: MLCircuitBreakerServiceProtocol,
        batch_size: int = 25,
        cache_ttl_hours: int = 24,
        performance_threshold_ms: float = 50.0,
    ) -> None:
        """Initialize rule analysis service.
        
        Args:
            ml_repository: Repository for ML data access
            pattern_discovery: Advanced pattern discovery component
            rule_optimizer: Rule optimization component
            circuit_breaker: Circuit breaker for fault tolerance
            batch_size: Batch processing size
            cache_ttl_hours: Cache TTL in hours
            performance_threshold_ms: Performance threshold in milliseconds
        """
        self.ml_repository = ml_repository
        self.pattern_discovery = pattern_discovery
        self.rule_optimizer = rule_optimizer
        self.circuit_breaker = circuit_breaker
        self.batch_size = batch_size
        self.cache_ttl_hours = cache_ttl_hours
        self.performance_threshold_ms = performance_threshold_ms
        
        # Performance metrics
        self._processing_stats = {
            "total_rules_processed": 0,
            "total_combinations_processed": 0,
            "avg_processing_time_ms": 0.0,
            "cache_hit_rate": 0.0,
            "error_rate": 0.0,
        }

    async def process_rule_intelligence(
        self, rule_ids: Optional[List[str]] = None
    ) -> IntelligenceResult:
        """Process individual rule effectiveness analysis.
        
        Extracted from intelligence_processor._process_rule_intelligence()
        with performance monitoring and circuit breaker integration.
        
        Args:
            rule_ids: Optional list of specific rule IDs to process
            
        Returns:
            IntelligenceResult with processing metrics
        """
        start_time = time.perf_counter()
        
        try:
            # Execute with circuit breaker protection
            result = await self.circuit_breaker.call_with_breaker(
                "rule_intelligence_processor",
                self._execute_rule_intelligence_processing,
                rule_ids
            )
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Update performance stats
            self._processing_stats["total_rules_processed"] += result.get("rules_processed", 0)
            self._update_avg_processing_time(processing_time)
            
            # Check performance threshold
            if processing_time > self.performance_threshold_ms:
                logger.warning(
                    f"Rule intelligence processing exceeded threshold: {processing_time:.2f}ms > {self.performance_threshold_ms}ms"
                )
            
            return IntelligenceResult(
                success=True,
                data=result,
                confidence=0.85,  # High confidence for rule analysis
                processing_time_ms=processing_time,
                cache_hit=result.get("cache_hits", 0) > 0
            )
            
        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Rule intelligence processing failed: {e}", exc_info=True)
            
            # Update error rate
            self._processing_stats["error_rate"] += 1
            
            return IntelligenceResult(
                success=False,
                data={},
                confidence=0.0,
                processing_time_ms=processing_time,
                error_message=str(e)
            )

    async def process_combination_intelligence(
        self, combination_limit: int = 50
    ) -> IntelligenceResult:
        """Process rule combination synergy analysis.
        
        Extracted from intelligence_processor._process_combination_intelligence()
        with performance optimization and error handling.
        
        Args:
            combination_limit: Maximum combinations to process
            
        Returns:
            IntelligenceResult with processing metrics
        """
        start_time = time.perf_counter()
        
        try:
            # Execute with circuit breaker protection
            result = await self.circuit_breaker.call_with_breaker(
                "combination_intelligence_processor",
                self._execute_combination_intelligence_processing,
                combination_limit
            )
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Update performance stats
            self._processing_stats["total_combinations_processed"] += result.get("combinations_generated", 0)
            self._update_avg_processing_time(processing_time)
            
            return IntelligenceResult(
                success=True,
                data=result,
                confidence=0.80,  # Good confidence for combination analysis
                processing_time_ms=processing_time,
                cache_hit=result.get("cache_hits", 0) > 0
            )
            
        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Combination intelligence processing failed: {e}", exc_info=True)
            
            self._processing_stats["error_rate"] += 1
            
            return IntelligenceResult(
                success=False,
                data={},
                confidence=0.0,
                processing_time_ms=processing_time,
                error_message=str(e)
            )

    async def generate_intelligence_data(
        self, rule_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate ML intelligence for specific rule characteristics.
        
        Args:
            rule_characteristics: Rule characteristic data
            
        Returns:
            Dictionary containing generated intelligence
        """
        try:
            # Generate characteristics hash for caching
            characteristics_hash = self._hash_characteristics_dict(rule_characteristics)
            
            # Check cache freshness if rule_id provided
            rule_id = rule_characteristics.get("rule_id")
            if rule_id:
                is_fresh = await self.ml_repository.check_rule_intelligence_freshness(rule_id)
                if is_fresh:
                    logger.debug(f"Using cached intelligence for rule {rule_id}")
                    return {"cached": True, "characteristics_hash": characteristics_hash}
            
            # Generate comprehensive intelligence
            intelligence_data = {
                "effectiveness_prediction": min(1.0, rule_characteristics.get("effectiveness_ratio", 0.0) * 1.2),
                "confidence_score": rule_characteristics.get("confidence_score", 0.5),
                "context_compatibility": {
                    "general_purpose": rule_characteristics.get("effectiveness_ratio", 0.0) > 0.5,
                    "specialized": rule_characteristics.get("usage_count", 0) > 10,
                },
                "usage_recommendations": self._generate_usage_recommendations(rule_characteristics),
                "pattern_insights": await self._extract_pattern_insights(rule_characteristics),
                "performance_forecast": self._calculate_performance_forecast(rule_characteristics),
                "optimization_suggestions": await self._generate_optimization_suggestions(rule_characteristics),
                "characteristics_hash": characteristics_hash,
            }
            
            return intelligence_data
            
        except Exception as e:
            logger.error(f"Failed to generate intelligence data: {e}", exc_info=True)
            return {"error": str(e)}

    async def analyze_rule_effectiveness(
        self, rule_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze effectiveness of specific rule.
        
        Args:
            rule_id: Rule ID to analyze
            context: Analysis context
            
        Returns:
            Dictionary containing effectiveness analysis
        """
        try:
            # Get historical performance data
            historical_data = await self.ml_repository.get_rule_historical_performance(rule_id)
            
            if not historical_data:
                return {
                    "rule_id": rule_id,
                    "analysis_status": "insufficient_data",
                    "recommendation": "Collect more performance data"
                }
            
            # Calculate effectiveness metrics
            avg_effectiveness = sum(item.get("effectiveness", 0.0) for item in historical_data) / len(historical_data)
            trend_direction = self._calculate_trend_direction(historical_data)
            confidence_level = self._calculate_confidence_level(historical_data)
            
            # Generate analysis
            analysis = {
                "rule_id": rule_id,
                "effectiveness_score": avg_effectiveness,
                "trend_direction": trend_direction,
                "confidence_level": confidence_level,
                "sample_size": len(historical_data),
                "analysis_timestamp": time.time(),
                "recommendations": self._generate_effectiveness_recommendations(
                    avg_effectiveness, trend_direction, confidence_level
                ),
                "performance_insights": {
                    "stability": "stable" if confidence_level > 0.7 else "variable",
                    "usage_pattern": context.get("usage_pattern", "unknown"),
                    "optimization_potential": max(0, 1.0 - avg_effectiveness)
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Rule effectiveness analysis failed for {rule_id}: {e}", exc_info=True)
            return {"rule_id": rule_id, "error": str(e)}

    async def _execute_rule_intelligence_processing(
        self, rule_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute core rule intelligence processing logic.
        
        Extracted from intelligence_processor._process_rule_intelligence()
        """
        logger.info("Processing rule effectiveness intelligence via repository")

        # Get batch of prompt characteristics using repository
        characteristic_groups = await self.ml_repository.get_prompt_characteristics_batch(
            batch_size=self.batch_size
        )

        # Get rule performance data using repository
        rules_data = await self.ml_repository.get_rule_performance_data(
            batch_size=self.batch_size
        )
        
        rules_processed = 0
        intelligence_batch = []
        cache_hits = 0

        # Process rule performance data with ML analysis 
        for rule_data in rules_data:
            try:
                rule_id = rule_data.get("rule_id")
                if not rule_id:
                    continue
                
                # Skip if specific rule IDs requested and this isn't one of them
                if rule_ids and rule_id not in rule_ids:
                    continue
                    
                # Check cache freshness
                if await self.ml_repository.check_rule_intelligence_freshness(rule_id):
                    cache_hits += 1
                    continue
                
                # Generate intelligence for this rule
                intelligence_item = await self._create_rule_intelligence_item(rule_data)
                intelligence_batch.append(intelligence_item)
                rules_processed += 1
                
            except Exception as e:
                logger.warning(f"Failed to generate intelligence for rule {rule_data.get('rule_id', 'unknown')}: {e}")
                continue
        
        # Cache all intelligence data using repository
        if intelligence_batch:
            await self.ml_repository.cache_rule_intelligence(intelligence_batch)
        
        return {
            "rules_processed": rules_processed,
            "intelligence_cached": len(intelligence_batch),
            "cache_hits": cache_hits
        }

    async def _execute_combination_intelligence_processing(
        self, combination_limit: int = 50
    ) -> Dict[str, Any]:
        """Execute core combination intelligence processing logic.
        
        Extracted from intelligence_processor._process_combination_intelligence()
        """
        logger.info("Processing rule combination intelligence via repository")

        # Get rule combinations data using repository
        combinations_data = await self.ml_repository.get_rule_combinations_data(
            batch_size=min(self.batch_size, combination_limit)
        )

        combinations_generated = 0
        combination_intelligence_batch = []
        cache_hits = 0
        
        # Process each rule combination
        for combination_data in combinations_data[:combination_limit]:
            try:
                rule_combination = combination_data.get("rule_combination", [])
                if len(rule_combination) < 2:
                    continue
                    
                # Generate combination intelligence
                combo_intelligence = await self._create_combination_intelligence_item(combination_data)
                combination_intelligence_batch.append(combo_intelligence)
                combinations_generated += 1
                
            except Exception as e:
                logger.warning(f"Failed to process combination {combination_data.get('rule_combination', 'unknown')}: {e}")
                continue
        
        # Cache combination intelligence using repository
        if combination_intelligence_batch:
            await self.ml_repository.cache_combination_intelligence(combination_intelligence_batch)
        
        return {
            "combinations_generated": combinations_generated,
            "intelligence_cached": len(combination_intelligence_batch),
            "cache_hits": cache_hits
        }

    async def _create_rule_intelligence_item(self, rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create intelligence item for individual rule."""
        return {
            "rule_id": rule_data.get("rule_id"),
            "intelligence_data": {
                "usage_count": rule_data.get("usage_count", 0),
                "success_count": rule_data.get("success_count", 0),
                "effectiveness_ratio": rule_data.get("effectiveness_ratio", 0.0),
            },
            "confidence_score": rule_data.get("confidence_score", 0.0),
            "effectiveness_prediction": min(1.0, rule_data.get("effectiveness_ratio", 0.0) * 1.2),
            "context_compatibility": {
                "general_purpose": rule_data.get("effectiveness_ratio", 0.0) > 0.5,
                "specialized": rule_data.get("usage_count", 0) > 10,
            },
            "usage_recommendations": [
                f"Rule shows {rule_data.get('effectiveness_ratio', 0.0):.2%} effectiveness",
                "Consider for similar prompt types" if rule_data.get("effectiveness_ratio", 0.0) > 0.6 else "Use with caution"
            ],
            "pattern_insights": {
                "performance_trend": "stable" if rule_data.get("confidence_score", 0.0) > 0.5 else "variable",
                "usage_frequency": "high" if rule_data.get("usage_count", 0) > 20 else "moderate",
            },
            "performance_forecast": {
                "expected_improvement": rule_data.get("avg_improvement", 0.0),
                "confidence_interval": [
                    max(0, rule_data.get("avg_improvement", 0.0) - 0.1), 
                    min(1, rule_data.get("avg_improvement", 0.0) + 0.1)
                ],
            },
            "optimization_suggestions": [
                "Increase usage in similar contexts" if rule_data.get("effectiveness_ratio", 0.0) > 0.7 else "Review rule parameters",
                "Monitor performance trends" if rule_data.get("confidence_score", 0.0) < 0.6 else "Performance stable"
            ]
        }

    async def _create_combination_intelligence_item(self, combination_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create intelligence item for rule combination."""
        rule_combination = combination_data.get("rule_combination", [])
        
        return {
            "rule_combination": rule_combination,
            "synergy_score": min(1.0, combination_data.get("avg_improvement", 0.0) * 1.1),
            "effectiveness_multiplier": max(1.0, combination_data.get("avg_quality", 0.0) + 0.2),
            "context_suitability": {
                "usage_frequency": "high" if combination_data.get("usage_count", 0) > 5 else "low",
                "performance_stability": "stable" if combination_data.get("avg_improvement", 0.0) > 0.5 else "variable",
                "recommended_contexts": ["general", "technical"] if combination_data.get("avg_quality", 0.0) > 0.6 else ["specific"]
            },
            "performance_data": {
                "avg_improvement": combination_data.get("avg_improvement", 0.0),
                "avg_quality": combination_data.get("avg_quality", 0.0),
                "usage_count": combination_data.get("usage_count", 0),
                "last_used": combination_data.get("last_used"),
            },
            "optimization_insights": {
                "strengths": [
                    f"High synergy between {len(rule_combination)} rules",
                    "Good performance track record" if combination_data.get("avg_improvement", 0.0) > 0.7 else "Moderate performance"
                ],
                "recommendations": [
                    "Continue using this combination" if combination_data.get("avg_improvement", 0.0) > 0.6 else "Monitor performance",
                    "Consider expanding to similar contexts" if combination_data.get("usage_count", 0) > 10 else "Test more thoroughly"
                ]
            }
        }

    def _hash_characteristics_dict(self, characteristics: Dict[str, Any]) -> str:
        """Generate hash for characteristics dictionary."""
        # Sort keys to ensure consistent hashing
        sorted_items = sorted(characteristics.items())
        characteristics_str = json.dumps(sorted_items, sort_keys=True)
        return hashlib.md5(characteristics_str.encode()).hexdigest()

    def _generate_usage_recommendations(self, rule_characteristics: Dict[str, Any]) -> List[str]:
        """Generate usage recommendations based on rule characteristics."""
        recommendations = []
        effectiveness = rule_characteristics.get("effectiveness_ratio", 0.0)
        usage_count = rule_characteristics.get("usage_count", 0)
        
        if effectiveness > 0.8:
            recommendations.append("Highly effective rule - recommended for regular use")
        elif effectiveness > 0.6:
            recommendations.append("Good effectiveness - suitable for most contexts")
        else:
            recommendations.append("Use with caution - monitor performance closely")
        
        if usage_count < 10:
            recommendations.append("Limited usage data - consider more testing")
        
        return recommendations

    async def _extract_pattern_insights(self, rule_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pattern insights using pattern discovery component."""
        try:
            # This would integrate with pattern discovery service
            # For now, return basic insights based on characteristics
            return {
                "performance_stability": "stable" if rule_characteristics.get("confidence_score", 0.0) > 0.7 else "variable",
                "usage_pattern": "frequent" if rule_characteristics.get("usage_count", 0) > 20 else "infrequent",
                "effectiveness_trend": "improving" if rule_characteristics.get("effectiveness_ratio", 0.0) > 0.6 else "stable"
            }
        except Exception as e:
            logger.warning(f"Pattern insight extraction failed: {e}")
            return {"error": "pattern_extraction_failed"}

    def _calculate_performance_forecast(self, rule_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance forecast based on historical data."""
        avg_improvement = rule_characteristics.get("avg_improvement", 0.0)
        confidence = rule_characteristics.get("confidence_score", 0.5)
        
        return {
            "expected_improvement": avg_improvement,
            "confidence_interval": [
                max(0, avg_improvement - (0.1 * (1 - confidence))),
                min(1, avg_improvement + (0.1 * (1 - confidence)))
            ],
            "forecast_confidence": confidence
        }

    async def _generate_optimization_suggestions(self, rule_characteristics: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions using rule optimizer component."""
        try:
            suggestions = []
            effectiveness = rule_characteristics.get("effectiveness_ratio", 0.0)
            
            if effectiveness < 0.5:
                suggestions.append("Consider rule parameter optimization")
                suggestions.append("Review rule applicability conditions")
            elif effectiveness < 0.8:
                suggestions.append("Fine-tune rule parameters for better performance")
            else:
                suggestions.append("Rule performing well - maintain current parameters")
            
            return suggestions
        except Exception as e:
            logger.warning(f"Optimization suggestion generation failed: {e}")
            return ["Unable to generate optimization suggestions"]

    def _calculate_trend_direction(self, historical_data: List[Dict[str, Any]]) -> str:
        """Calculate performance trend direction from historical data."""
        if len(historical_data) < 2:
            return "insufficient_data"
        
        # Simple trend calculation - could be enhanced with more sophisticated analysis
        recent_avg = sum(item.get("effectiveness", 0.0) for item in historical_data[-5:]) / min(5, len(historical_data))
        older_avg = sum(item.get("effectiveness", 0.0) for item in historical_data[:-5]) / max(1, len(historical_data) - 5)
        
        if recent_avg > older_avg + 0.05:
            return "improving"
        elif recent_avg < older_avg - 0.05:
            return "declining"
        else:
            return "stable"

    def _calculate_confidence_level(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate confidence level based on data consistency."""
        if not historical_data:
            return 0.0
        
        # Calculate variance in effectiveness scores
        effectiveness_scores = [item.get("effectiveness", 0.0) for item in historical_data]
        if len(effectiveness_scores) < 2:
            return 0.5
        
        mean_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
        variance = sum((x - mean_effectiveness) ** 2 for x in effectiveness_scores) / len(effectiveness_scores)
        
        # Convert variance to confidence (lower variance = higher confidence)
        confidence = max(0.0, min(1.0, 1.0 - (variance * 4)))  # Scale appropriately
        return confidence

    def _generate_effectiveness_recommendations(
        self, effectiveness: float, trend: str, confidence: float
    ) -> List[str]:
        """Generate recommendations based on effectiveness analysis."""
        recommendations = []
        
        if effectiveness > 0.8:
            recommendations.append("Rule shows excellent effectiveness")
        elif effectiveness > 0.6:
            recommendations.append("Rule shows good effectiveness")
        else:
            recommendations.append("Rule effectiveness needs improvement")
        
        if trend == "improving":
            recommendations.append("Performance trend is positive")
        elif trend == "declining":
            recommendations.append("Monitor rule - performance declining")
        
        if confidence < 0.5:
            recommendations.append("Collect more data for reliable analysis")
        
        return recommendations

    def _update_avg_processing_time(self, processing_time_ms: float) -> None:
        """Update average processing time with exponential smoothing."""
        alpha = 0.1  # Smoothing factor
        current_avg = self._processing_stats["avg_processing_time_ms"]
        self._processing_stats["avg_processing_time_ms"] = (
            alpha * processing_time_ms + (1 - alpha) * current_avg
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            **self._processing_stats,
            "performance_threshold_ms": self.performance_threshold_ms,
            "batch_size": self.batch_size,
            "cache_ttl_hours": self.cache_ttl_hours,
        }