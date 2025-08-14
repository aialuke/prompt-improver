"""ML Pattern Discovery Service.

Provides specialized ML pattern discovery and analysis capabilities.
Extracted from intelligence_processor.py god object to follow single responsibility principle.

Performance Target: <100ms for pattern discovery operations
Memory Target: <50MB for pattern caching
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib

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
        self._pattern_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_ttl_seconds = 1800  # 30 minutes
        
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
            # Generate cache key for this batch
            cache_key = self._generate_cache_key(batch_data)
            
            # Check cache first
            cached_patterns = await self.get_cached_patterns(cache_key)
            if cached_patterns:
                processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                self._metrics_registry.increment(
                    "ml_pattern_discovery_cache_hits_total",
                    tags={"service": "pattern_discovery"}
                )
                
                return IntelligenceResult(
                    success=True,
                    data=cached_patterns,
                    confidence=cached_patterns.get("confidence", 0.8),
                    processing_time_ms=processing_time,
                    cache_hit=True
                )
            
            # Perform pattern discovery with circuit breaker protection
            patterns = await self._circuit_breaker_service.call_with_breaker(
                "pattern_discovery",
                self._discover_patterns_internal,
                batch_data
            )
            
            # Cache the results
            await self.cache_pattern_results(cache_key, patterns)
            
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
    
    async def cache_pattern_results(self, pattern_key: str, results: Dict[str, Any]) -> None:
        """Cache pattern discovery results.
        
        Args:
            pattern_key: Unique key for the pattern
            results: Pattern discovery results to cache
        """
        try:
            self._pattern_cache[pattern_key] = results
            self._cache_timestamps[pattern_key] = datetime.now(timezone.utc)
            
            # Clean up expired cache entries
            await self._cleanup_expired_cache()
            
            self._metrics_registry.increment(
                "ml_pattern_cache_writes_total",
                tags={"service": "pattern_discovery"}
            )
            
        except Exception as e:
            logger.error(f"Failed to cache pattern results: {e}")
    
    async def get_cached_patterns(self, pattern_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached pattern results.
        
        Args:
            pattern_key: Unique key for the pattern
            
        Returns:
            Cached pattern results if available and not expired
        """
        try:
            if pattern_key not in self._pattern_cache:
                return None
            
            # Check if cache entry is expired
            cached_time = self._cache_timestamps.get(pattern_key)
            if not cached_time:
                return None
            
            age_seconds = (datetime.now(timezone.utc) - cached_time).total_seconds()
            if age_seconds > self._cache_ttl_seconds:
                # Remove expired entry
                self._pattern_cache.pop(pattern_key, None)
                self._cache_timestamps.pop(pattern_key, None)
                return None
            
            return self._pattern_cache[pattern_key]
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached patterns: {e}")
            return None
    
    def _generate_cache_key(self, batch_data: List[Dict[str, Any]]) -> str:
        """Generate cache key for batch data.
        
        Args:
            batch_data: Batch data to generate key for
            
        Returns:
            Unique cache key
        """
        try:
            # Create a hash of the relevant data characteristics
            data_summary = {
                "size": len(batch_data),
                "first_item_keys": list(batch_data[0].keys()) if batch_data else [],
                "data_types": [type(item.get("prompt_text", "")).__name__ for item in batch_data[:5]]
            }
            
            # Generate hash
            data_string = json.dumps(data_summary, sort_keys=True)
            return hashlib.md5(data_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to generate cache key: {e}")
            return f"fallback_{datetime.now(timezone.utc).timestamp()}"
    
    async def _cleanup_expired_cache(self) -> None:
        """Clean up expired cache entries."""
        try:
            current_time = datetime.now(timezone.utc)
            expired_keys = []
            
            for key, timestamp in self._cache_timestamps.items():
                age_seconds = (current_time - timestamp).total_seconds()
                if age_seconds > self._cache_ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._pattern_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring.
        
        Returns:
            Cache statistics
        """
        return {
            "total_entries": len(self._pattern_cache),
            "cache_size_mb": sum(
                len(json.dumps(data).encode()) for data in self._pattern_cache.values()
            ) / (1024 * 1024),
            "oldest_entry_age_seconds": (
                (datetime.now(timezone.utc) - min(self._cache_timestamps.values())).total_seconds()
                if self._cache_timestamps else 0
            ),
            "cache_ttl_seconds": self._cache_ttl_seconds
        }