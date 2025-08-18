"""Prompt Analysis Service - Focused service for prompt analysis and improvement logic.

This service handles:
- Prompt analysis and improvement suggestions
- ML-based recommendations
- Quality evaluation
- Rule effectiveness analysis

Part of the PromptServiceFacade decomposition following Clean Architecture principles.
"""

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from datetime import datetime
from uuid import UUID

if TYPE_CHECKING:
    from prompt_improver.core.events.ml_event_bus import (
        MLEvent,
        MLEventType,
    )
    from prompt_improver.core.interfaces.ml_interface import MLModelInterface
from prompt_improver.core.protocols.prompt_service.prompt_protocols import (
    PromptAnalysisServiceProtocol,
)
from prompt_improver.rule_engine.base import BasePromptRule
from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)


class PromptAnalysisService(PromptAnalysisServiceProtocol):
    """Service for prompt analysis and improvement logic."""

    def __init__(
        self,
        ml_interface: Optional["MLModelInterface"] = None,
        enable_automl: bool = True,
    ):
        self.ml_interface = ml_interface
        self.enable_automl = enable_automl
        self.analysis_cache = {}
        self.cache_ttl = 300
        self.logger = logging.getLogger(__name__)

    async def _get_ml_event_bus(self):
        """Lazy load ML event bus to avoid torch dependencies."""
        try:
            from prompt_improver.core.events.ml_event_bus import get_ml_event_bus
            return await get_ml_event_bus()
        except ImportError:
            self.logger.info("ML event bus not available (torch not installed)")
            return None

    def _create_ml_event(self, event_type: str, source: str, data: Dict[str, Any]):
        """Lazy create ML event to avoid torch dependencies."""
        try:
            from prompt_improver.core.events.ml_event_bus import MLEvent, MLEventType
            return MLEvent(
                event_type=getattr(MLEventType, event_type),
                source=source,
                data=data
            )
        except ImportError:
            self.logger.info("ML events not available (torch not installed)")
            return None

    async def analyze_prompt(
        self,
        prompt_id: UUID,
        session_id: Optional[UUID] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze a prompt and generate improvement suggestions."""
        try:
            cache_key = f"analysis_{prompt_id}_{session_id}"
            cached_result = self._get_cached_analysis(cache_key)
            if cached_result:
                return cached_result

            # Extract prompt characteristics
            characteristics = await self._extract_prompt_characteristics(prompt_id, context)
            
            # Analyze structure and clarity
            structure_analysis = await self._analyze_structure(characteristics)
            
            # Analyze clarity and specificity
            clarity_analysis = await self._analyze_clarity(characteristics)
            
            # Generate improvement suggestions
            suggestions = await self._generate_suggestions(
                characteristics, structure_analysis, clarity_analysis
            )
            
            # Calculate analysis metrics
            metrics = await self._calculate_analysis_metrics(
                characteristics, suggestions
            )

            result = {
                "prompt_id": str(prompt_id),
                "session_id": str(session_id) if session_id else None,
                "analysis_timestamp": aware_utc_now().isoformat(),
                "characteristics": characteristics,
                "structure_analysis": structure_analysis,
                "clarity_analysis": clarity_analysis,
                "suggestions": suggestions,
                "metrics": metrics,
                "confidence_score": self._calculate_confidence_score(suggestions)
            }

            self._cache_analysis(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Error analyzing prompt {prompt_id}: {e}")
            raise

    async def generate_improvements(
        self,
        prompt: str,
        rules: List[BasePromptRule],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate improvements for a prompt using specified rules."""
        try:
            improvements = []
            
            for rule in rules:
                try:
                    # Apply rule to generate improvement
                    improvement_result = await self._apply_rule_for_improvement(
                        prompt, rule, context
                    )
                    
                    if improvement_result:
                        improvement = {
                            "rule_id": getattr(rule, 'rule_id', rule.__class__.__name__),
                            "rule_name": getattr(rule, 'name', rule.__class__.__name__),
                            "original_prompt": prompt,
                            "improved_prompt": improvement_result.get("improved_prompt"),
                            "confidence_score": improvement_result.get("confidence", 0.0),
                            "improvement_type": improvement_result.get("type", "unknown"),
                            "explanation": improvement_result.get("explanation", ""),
                            "metrics": improvement_result.get("metrics", {}),
                            "timestamp": aware_utc_now().isoformat()
                        }
                        improvements.append(improvement)
                        
                except Exception as rule_error:
                    logger.warning(f"Rule {rule.__class__.__name__} failed: {rule_error}")
                    continue

            # Sort by confidence score
            improvements.sort(key=lambda x: x["confidence_score"], reverse=True)
            
            return improvements

        except Exception as e:
            logger.error(f"Error generating improvements: {e}")
            raise

    async def evaluate_improvement_quality(
        self,
        original_prompt: str,
        improved_prompt: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Evaluate the quality of an improvement."""
        try:
            quality_scores = {}
            
            # Length improvement score
            quality_scores["length_improvement"] = self._calculate_length_score(
                original_prompt, improved_prompt
            )
            
            # Clarity improvement score
            quality_scores["clarity_improvement"] = self._calculate_clarity_score(
                original_prompt, improved_prompt
            )
            
            # Specificity improvement score
            quality_scores["specificity_improvement"] = self._calculate_specificity_score(
                original_prompt, improved_prompt
            )
            
            # Structure improvement score
            quality_scores["structure_improvement"] = self._calculate_structure_score(
                original_prompt, improved_prompt
            )
            
            # Overall quality score (weighted average)
            weights = {
                "length_improvement": 0.2,
                "clarity_improvement": 0.3,
                "specificity_improvement": 0.3,
                "structure_improvement": 0.2
            }
            
            quality_scores["overall_quality"] = sum(
                score * weights.get(metric, 0.25) 
                for metric, score in quality_scores.items()
            )
            
            # Include provided metrics if available
            if metrics:
                quality_scores.update(metrics)
            
            return quality_scores

        except Exception as e:
            logger.error(f"Error evaluating improvement quality: {e}")
            return {"overall_quality": 0.0}

    async def get_ml_recommendations(
        self,
        prompt: str,
        session_id: UUID,
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get ML-based recommendations for prompt improvement."""
        try:
            if not self.enable_automl or not self.ml_interface:
                return {"status": "ml_disabled", "recommendations": []}

            # Request ML recommendations via event bus
            event_bus = await self._get_ml_event_bus()
            if not event_bus:
                return {"status": "ml_unavailable", "recommendations": []}
                
            ml_event = self._create_ml_event(
                event_type="PREDICTION_REQUEST",
                source="prompt_analysis_service",
                data={
                    "operation": "get_recommendations",
                    "prompt": prompt,
                    "session_id": str(session_id),
                    "model_version": model_version,
                    "context": {
                        "timestamp": aware_utc_now().isoformat(),
                        "service": "prompt_analysis"
                    }
                }
            )
            
            await event_bus.publish(ml_event)
            
            # For now, return simulated ML recommendations
            # In production, this would wait for ML service response
            recommendations = {
                "status": "success",
                "model_version": model_version or "latest",
                "recommendations": [
                    {
                        "type": "structure",
                        "suggestion": "Add more specific context",
                        "confidence": 0.85,
                        "priority": 1
                    },
                    {
                        "type": "clarity",
                        "suggestion": "Use more precise language",
                        "confidence": 0.78,
                        "priority": 2
                    }
                ],
                "timestamp": aware_utc_now().isoformat()
            }
            
            return recommendations

        except Exception as e:
            logger.error(f"Error getting ML recommendations: {e}")
            return {"status": "error", "error": str(e)}

    async def _extract_prompt_characteristics(
        self, 
        prompt_id: UUID, 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract characteristics from prompt for analysis."""
        # This would typically fetch the prompt from repository
        # For now, return basic characteristics
        return {
            "prompt_id": str(prompt_id),
            "length": 0,  # Would be calculated from actual prompt
            "complexity": "medium",
            "domain": context.get("domain", "general") if context else "general",
            "style": "instructional",
            "has_examples": False,
            "has_constraints": False
        }

    async def _analyze_structure(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze prompt structure."""
        return {
            "has_clear_objective": True,
            "has_context": True,
            "has_examples": characteristics.get("has_examples", False),
            "has_constraints": characteristics.get("has_constraints", False),
            "structure_score": 0.75
        }

    async def _analyze_clarity(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze prompt clarity."""
        return {
            "clarity_score": 0.8,
            "specificity_score": 0.7,
            "ambiguity_issues": [],
            "clarity_suggestions": ["Add more specific examples"]
        }

    async def _generate_suggestions(
        self,
        characteristics: Dict[str, Any],
        structure_analysis: Dict[str, Any],
        clarity_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []
        
        if structure_analysis["structure_score"] < 0.8:
            suggestions.append({
                "type": "structure",
                "priority": 1,
                "suggestion": "Improve prompt structure",
                "confidence": 0.85
            })
        
        if clarity_analysis["clarity_score"] < 0.8:
            suggestions.append({
                "type": "clarity",
                "priority": 2,
                "suggestion": "Enhance clarity and specificity",
                "confidence": 0.78
            })
        
        return suggestions

    async def _calculate_analysis_metrics(
        self,
        characteristics: Dict[str, Any],
        suggestions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate metrics for the analysis."""
        return {
            "total_suggestions": len(suggestions),
            "avg_confidence": sum(s.get("confidence", 0) for s in suggestions) / max(len(suggestions), 1),
            "analysis_quality": 0.8,
            "processing_time_ms": 150
        }

    async def _apply_rule_for_improvement(
        self,
        prompt: str,
        rule: BasePromptRule,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Apply a rule to generate improvement."""
        try:
            # Apply rule (simplified for this implementation)
            if hasattr(rule, 'apply'):
                result = await rule.apply(prompt, context or {})
                return {
                    "improved_prompt": result.get("improved_prompt", prompt),
                    "confidence": result.get("confidence", 0.5),
                    "type": result.get("improvement_type", "general"),
                    "explanation": result.get("explanation", "Rule applied"),
                    "metrics": result.get("metrics", {})
                }
            return None
        except Exception as e:
            logger.warning(f"Rule application failed: {e}")
            return None

    def _calculate_length_score(self, original: str, improved: str) -> float:
        """Calculate length improvement score."""
        orig_len = len(original)
        imp_len = len(improved)
        if orig_len == 0:
            return 1.0
        # Prefer moderate length increases
        ratio = imp_len / orig_len
        if 1.1 <= ratio <= 1.5:
            return 1.0
        elif ratio < 1.1:
            return 0.7
        else:
            return max(0.0, 1.0 - (ratio - 1.5) * 0.5)

    def _calculate_clarity_score(self, original: str, improved: str) -> float:
        """Calculate clarity improvement score."""
        # Simplified clarity calculation
        clarity_indicators = ["specific", "clear", "detailed", "example"]
        improved_indicators = sum(1 for indicator in clarity_indicators if indicator in improved.lower())
        original_indicators = sum(1 for indicator in clarity_indicators if indicator in original.lower())
        
        return min(1.0, (improved_indicators + 1) / (original_indicators + 1))

    def _calculate_specificity_score(self, original: str, improved: str) -> float:
        """Calculate specificity improvement score."""
        # Simplified specificity calculation
        specific_words = ["exactly", "precisely", "specifically", "particular"]
        improved_count = sum(1 for word in specific_words if word in improved.lower())
        original_count = sum(1 for word in specific_words if word in original.lower())
        
        return min(1.0, (improved_count + 1) / (original_count + 1))

    def _calculate_structure_score(self, original: str, improved: str) -> float:
        """Calculate structure improvement score."""
        # Simplified structure calculation based on organization
        structure_elements = ["\n\n", "1.", "2.", "-", "*"]
        improved_elements = sum(1 for elem in structure_elements if elem in improved)
        original_elements = sum(1 for elem in structure_elements if elem in original)
        
        return min(1.0, (improved_elements + 1) / (original_elements + 1))

    def _calculate_confidence_score(self, suggestions: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for analysis."""
        if not suggestions:
            return 0.5
        
        confidences = [s.get("confidence", 0.5) for s in suggestions]
        return sum(confidences) / len(confidences)

    def _get_cached_analysis(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result."""
        if cache_key in self.analysis_cache:
            cached_item = self.analysis_cache[cache_key]
            if (datetime.now() - cached_item["timestamp"]).seconds < self.cache_ttl:
                return cached_item["data"]
        return None

    def _cache_analysis(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache analysis result."""
        self.analysis_cache[cache_key] = {
            "data": result,
            "timestamp": datetime.now()
        }
        
        # Simple cache cleanup - keep only last 100 items
        if len(self.analysis_cache) > 100:
            oldest_key = min(self.analysis_cache.keys(), 
                           key=lambda k: self.analysis_cache[k]["timestamp"])
            del self.analysis_cache[oldest_key]