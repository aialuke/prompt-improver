"""ML Pattern Discovery Service.

Provides specialized ML pattern discovery and analysis capabilities.
Extracted from intelligence_processor.py god object to follow single responsibility principle.

Performance Target: <100ms for pattern discovery operations
Memory Target: <50MB for pattern caching
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from prompt_improver.ml.services.intelligence.protocols.intelligence_service_protocols import (
    PatternDiscoveryServiceProtocol,
    IntelligenceResult,
    MLCircuitBreakerServiceProtocol,
)
from prompt_improver.ml.learning.patterns.advanced_pattern_discovery import (
    AdvancedPatternDiscovery,
)
from prompt_improver.repositories.protocols.ml_repository_protocol import MLRepositoryProtocol
from prompt_improver.performance.monitoring.metrics_registry import (
    StandardMetrics,
    get_metrics_registry,
)

logger = logging.getLogger(__name__)


class PatternDiscoveryService:
    """ML Pattern Discovery Service.
    
    Handles ML pattern discovery, analysis, and caching operations with
    performance optimizations and circuit breaker protection.
    """
    
    def __init__(
        self,
        pattern_discovery: AdvancedPatternDiscovery,
        ml_repository: MLRepositoryProtocol,
        circuit_breaker_service: MLCircuitBreakerServiceProtocol
    ):
        """Initialize pattern discovery service.
        
        Args:
            pattern_discovery: Advanced pattern discovery component
            ml_repository: ML repository for data access
            circuit_breaker_service: Circuit breaker protection service
        """
        self._pattern_discovery = pattern_discovery
        self._ml_repository = ml_repository
        self._circuit_breaker_service = circuit_breaker_service
        self._metrics_registry = get_metrics_registry()
        
        logger.info("PatternDiscoveryService initialized")
    
    async def discover_patterns(self, batch_data: List[Dict[str, Any]]) -> IntelligenceResult:
        """Discover ML patterns in batch data.
        
        Args:
            batch_data: Batch of data for pattern discovery
            
        Returns:
            Intelligence result with discovered patterns
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Perform pattern discovery with circuit breaker protection
            patterns = await self._circuit_breaker_service.call_with_breaker(
                "pattern_discovery",
                self._discover_patterns_internal,
                batch_data
            )
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            self._metrics_registry.increment(
                "ml_pattern_discovery_operations_total",
                tags={"service": "pattern_discovery", "result": "success"}
            )
            self._metrics_registry.record_value(
                "ml_pattern_discovery_duration_ms",
                processing_time,
                tags={"service": "pattern_discovery"}
            )
            
            return IntelligenceResult(
                success=True,
                data=patterns,
                confidence=patterns.get("confidence", 0.0),
                processing_time_ms=processing_time,
                cache_hit=False
            )
            
        except Exception as e:
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            self._metrics_registry.increment(
                "ml_pattern_discovery_operations_total",
                tags={"service": "pattern_discovery", "result": "error"}
            )
            
            logger.error(f"Pattern discovery failed: {e}")
            
            return IntelligenceResult(
                success=False,
                data={},
                confidence=0.0,
                processing_time_ms=processing_time,
                cache_hit=False,
                error_message=str(e)
            )
    
    async def _discover_patterns_internal(self, batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Internal pattern discovery implementation.
        
        Args:
            batch_data: Data for pattern discovery
            
        Returns:
            Discovered patterns
        """
        if not batch_data:
            return {"patterns": [], "confidence": 0.0, "insights": []}
        
        # Use advanced pattern discovery for ensemble analysis
        patterns = await self._pattern_discovery.discover_patterns_ensemble(
            user_sessions=batch_data,
            pattern_types=['frequent', 'sequential', 'contextual']
        )
        
        # Analyze discovered patterns for insights
        insights = await self.analyze_pattern_insights(patterns.get("patterns", []))
        
        result = {
            "patterns": patterns.get("patterns", []),
            "confidence": patterns.get("confidence", 0.0),
            "pattern_types": patterns.get("pattern_types", []),
            "insights": insights,
            "discovered_at": datetime.now(timezone.utc).isoformat(),
            "data_size": len(batch_data)
        }
        
        return result
    
    async def analyze_pattern_insights(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze discovered patterns for insights.
        
        Args:
            patterns: List of discovered patterns
            
        Returns:
            Pattern insights and analysis
        """
        if not patterns:
            return {"insights": [], "summary": "No patterns discovered"}
        
        insights = {
            "total_patterns": len(patterns),
            "pattern_categories": {},
            "confidence_distribution": {},
            "actionable_insights": [],
            "recommendations": []
        }
        
        # Categorize patterns
        for pattern in patterns:
            category = pattern.get("category", "unknown")
            if category not in insights["pattern_categories"]:
                insights["pattern_categories"][category] = 0
            insights["pattern_categories"][category] += 1
            
            # Confidence distribution
            confidence = pattern.get("confidence", 0.0)
            confidence_bucket = f"{int(confidence * 10) * 10}%-{int(confidence * 10) * 10 + 10}%"
            if confidence_bucket not in insights["confidence_distribution"]:
                insights["confidence_distribution"][confidence_bucket] = 0
            insights["confidence_distribution"][confidence_bucket] += 1
            
            # Generate actionable insights
            if confidence > 0.8:
                insights["actionable_insights"].append({
                    "pattern_id": pattern.get("id"),
                    "type": "high_confidence",
                    "description": f"Strong pattern detected: {pattern.get('description', 'Unknown')}"
                })
        
        # Generate recommendations
        high_confidence_patterns = [p for p in patterns if p.get("confidence", 0) > 0.7]
        if high_confidence_patterns:
            insights["recommendations"].append({
                "type": "optimization",
                "priority": "high",
                "description": f"Consider implementing rules based on {len(high_confidence_patterns)} high-confidence patterns"
            })
        
        return insights
    
