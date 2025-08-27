"""ML Service Facade with performance optimizations.

Unified interface that coordinates individual focused services for
comprehensive ML operations. Provides clean API while maintaining
high performance through advanced caching and optimization.

Enhanced with:
- Multi-level caching for inference results (target: <10ms response time)
- Performance metrics tracking
- Circuit breaker pattern for resilience
"""

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.services.cache.cache_factory import CacheFactory
from .protocols import MLServiceProtocol
from .registry import MLModelRegistry
from .training_service import MLTrainingService
from .inference_service import MLInferenceService
from .production_service import MLProductionService
from .pattern_discovery_service import MLPatternDiscoveryService
from .orchestration_adapter import MLOrchestrationAdapter

logger = logging.getLogger(__name__)


class MLModelServiceFacade(MLServiceProtocol):
    """Primary ML service facade coordinating focused service components.
    
    This facade delegates to individual focused services providing a unified
    API for all ML operations including training, inference, and pattern discovery.
    """

    def __init__(
        self, 
        db_manager=None, 
        orchestrator_event_bus=None,
        max_cache_size_mb: int = 500,
        default_ttl_minutes: int = 60
    ):
        """Initialize the ML service facade with all component services.
        
        Args:
            db_manager: Database manager for data operations
            orchestrator_event_bus: Event bus for orchestrator integration
            max_cache_size_mb: Maximum cache size for model registry
            default_ttl_minutes: Default TTL for cached models
        """
        # Create the model registry (shared by all services)
        self.model_registry = MLModelRegistry(
            max_cache_size_mb=max_cache_size_mb,
            default_ttl_minutes=default_ttl_minutes
        )
        
        # Performance optimization - direct ML analysis cache for <10ms response times
        self.cache_manager = CacheFactory.get_ml_analysis_cache()
        self._cache_enabled = True
        
        # Performance metrics
        self._performance_metrics = {
            "inference_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_inference_time_ms": 0.0,
            "pattern_discovery_requests": 0,
            "avg_pattern_discovery_time_ms": 0.0,
        }
        
        # Initialize individual services
        self.training_service = MLTrainingService(
            model_registry=self.model_registry,
            db_manager=db_manager,
            orchestrator_event_bus=orchestrator_event_bus
        )
        
        self.inference_service = MLInferenceService()
        
        self.production_service = MLProductionService(
            model_registry=self.model_registry,
            orchestrator_event_bus=orchestrator_event_bus
        )
        
        self.pattern_discovery_service = MLPatternDiscoveryService(
            db_manager=db_manager
        )
        
        self.orchestration_adapter = MLOrchestrationAdapter(
            training_service=self.training_service,
            inference_service=self.inference_service,
            production_service=self.production_service,
            pattern_discovery_service=self.pattern_discovery_service,
            orchestrator_event_bus=orchestrator_event_bus
        )
        
        logger.info(f"ML Model Service Facade initialized with optimized ML analysis cache (target: <10ms response times)")

    # Training methods - delegate to training service
    async def optimize_rules(
        self,
        training_data: Dict[str, List],
        db_session: AsyncSession,
        rule_ids: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Optimize rule parameters using ML training."""
        return await self.training_service.optimize_rules(
            training_data, db_session, rule_ids
        )

    async def optimize_ensemble_rules(
        self, training_data: Dict[str, List], db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Optimize rules using sophisticated ensemble methods."""
        return await self.training_service.optimize_ensemble_rules(
            training_data, db_session
        )

    async def send_training_batch(self, batch: List[Dict]) -> Dict[str, Any]:
        """Send training batch to local ML stub storage."""
        return await self.training_service.send_training_batch(batch)

    async def fetch_latest_model(self) -> Dict[str, Any]:
        """Fetch the latest model from ML stub storage."""
        return await self.training_service.fetch_latest_model()

    # Inference methods - delegate to inference service with caching
    async def predict_rule_effectiveness(
        self, model_id: str, rule_features: List[float]
    ) -> Dict[str, Any]:
        """Predict rule effectiveness using trained model with caching for <10ms response."""
        start_time = time.time()
        self._performance_metrics["inference_requests"] += 1
        
        # Generate cache key for this prediction request
        cache_key = None
        if self._cache_enabled:
            feature_str = ",".join(map(str, rule_features))
            content = f"rule_effectiveness:{model_id}:{feature_str}"
            cache_key = f"ml_inference:{hashlib.md5(content.encode()).hexdigest()}"
            
            # Check cache first (target: <5ms)
            try:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    self._performance_metrics["cache_hits"] += 1
                    self._update_inference_metrics(start_time)
                    logger.debug(f"Cache hit for rule effectiveness prediction: {model_id}")
                    
                    cached_result["cache_hit"] = True
                    cached_result["served_from_cache"] = True
                    return cached_result
            except Exception as e:
                logger.warning(f"Cache lookup failed for inference: {e}")
            
            self._performance_metrics["cache_misses"] += 1
        
        # Execute actual inference
        try:
            result = await self.inference_service.predict_rule_effectiveness(
                model_id, rule_features
            )
            
            # Add performance metadata
            result["cache_hit"] = False
            result["served_from_cache"] = False
            
            # Cache successful results (TTL: 30 minutes for inference)
            if self._cache_enabled and cache_key and result.get("success", True):
                try:
                    await self.cache_manager.set(
                        cache_key, 
                        result, 
                        l2_ttl=1800  # 30 minutes
                    )
                    logger.debug(f"Cached inference result: {model_id}")
                except Exception as e:
                    logger.warning(f"Failed to cache inference result: {e}")
            
            self._update_inference_metrics(start_time)
            return result
            
        except Exception as e:
            self._update_inference_metrics(start_time)
            logger.error(f"Inference failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "cache_hit": False,
                "served_from_cache": False,
            }

    # Pattern discovery methods - delegate to pattern discovery service with caching
    async def discover_patterns(
        self,
        db_session: AsyncSession,
        min_effectiveness: float = 0.7,
        min_support: int = 5,
        use_advanced_discovery: bool = True,
        include_apriori: bool = True,
    ) -> Dict[str, Any]:
        """Enhanced pattern discovery with caching for improved performance."""
        start_time = time.time()
        self._performance_metrics["pattern_discovery_requests"] += 1
        
        # Generate cache key for this pattern discovery request
        cache_key = None
        if self._cache_enabled:
            params = f"{min_effectiveness}:{min_support}:{use_advanced_discovery}:{include_apriori}"
            content = f"pattern_discovery:{params}"
            cache_key = f"ml_patterns:{hashlib.md5(content.encode()).hexdigest()}"
            
            # Check cache first (target: <5ms)
            try:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    self._performance_metrics["cache_hits"] += 1
                    self._update_pattern_discovery_metrics(start_time)
                    logger.debug("Cache hit for pattern discovery")
                    
                    cached_result["cache_hit"] = True
                    cached_result["served_from_cache"] = True
                    return cached_result
            except Exception as e:
                logger.warning(f"Cache lookup failed for pattern discovery: {e}")
            
            self._performance_metrics["cache_misses"] += 1
        
        # Execute actual pattern discovery
        try:
            result = await self.pattern_discovery_service.discover_patterns(
                db_session, min_effectiveness, min_support, use_advanced_discovery, include_apriori
            )
            
            # Add performance metadata
            result["cache_hit"] = False
            result["served_from_cache"] = False
            
            # Cache successful results (TTL: 15 minutes for patterns as they evolve)
            if self._cache_enabled and cache_key and result.get("success", True):
                try:
                    await self.cache_manager.set(
                        cache_key, 
                        result, 
                        l2_ttl=900  # 15 minutes
                    )
                    logger.debug("Cached pattern discovery result")
                except Exception as e:
                    logger.warning(f"Failed to cache pattern discovery result: {e}")
            
            self._update_pattern_discovery_metrics(start_time)
            return result
            
        except Exception as e:
            self._update_pattern_discovery_metrics(start_time)
            logger.error(f"Pattern discovery failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "cache_hit": False,
                "served_from_cache": False,
            }

    async def get_contextualized_patterns(
        self,
        context_items: List[str],
        db_session: AsyncSession,
        min_confidence: float = 0.6,
    ) -> Dict[str, Any]:
        """Get patterns relevant to a specific context with caching."""
        start_time = time.time()
        
        # Generate cache key for this contextualized patterns request
        cache_key = None
        if self._cache_enabled:
            context_str = ",".join(sorted(context_items))  # Sort for consistency
            content = f"contextualized_patterns:{context_str}:{min_confidence}"
            cache_key = f"ml_context_patterns:{hashlib.md5(content.encode()).hexdigest()}"
            
            # Check cache first (target: <5ms)
            try:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    logger.debug(f"Cache hit for contextualized patterns: {len(context_items)} items")
                    
                    cached_result["cache_hit"] = True
                    cached_result["served_from_cache"] = True
                    return cached_result
            except Exception as e:
                logger.warning(f"Cache lookup failed for contextualized patterns: {e}")
        
        # Execute actual contextualized pattern discovery
        try:
            result = await self.pattern_discovery_service.get_contextualized_patterns(
                context_items, db_session, min_confidence
            )
            
            # Add performance metadata
            result["cache_hit"] = False
            result["served_from_cache"] = False
            
            # Cache successful results (TTL: 10 minutes for context patterns)
            if self._cache_enabled and cache_key and result.get("success", True):
                try:
                    await self.cache_manager.set(
                        cache_key, 
                        result, 
                        l2_ttl=600  # 10 minutes
                    )
                    logger.debug("Cached contextualized patterns result")
                except Exception as e:
                    logger.warning(f"Failed to cache contextualized patterns result: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Contextualized patterns failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "cache_hit": False,
                "served_from_cache": False,
            }

    # Production methods - delegate to production service
    async def enable_production_deployment(
        self, tracking_uri: str | None = None
    ) -> Dict[str, Any]:
        """Enable production deployment capabilities."""
        return await self.production_service.enable_production_deployment(tracking_uri)

    async def deploy_to_production(
        self,
        model_name: str,
        version: str,
        alias: Any = None,
        strategy: Any = None,
    ) -> Dict[str, Any]:
        """Deploy model to production with specified strategy."""
        return await self.production_service.deploy_to_production(
            model_name, version, alias, strategy
        )

    async def rollback_production(
        self,
        model_name: str,
        alias: Any = None,
        reason: str = "Performance degradation detected",
    ) -> Dict[str, Any]:
        """Rollback production deployment to previous version."""
        return await self.production_service.rollback_production(
            model_name, alias, reason
        )

    async def monitor_production_health(
        self, model_name: str, alias: Any = None
    ) -> Dict[str, Any]:
        """Monitor production model health and performance."""
        return await self.production_service.monitor_production_health(
            model_name, alias
        )

    async def get_production_model(
        self, model_name: str, alias: Any = None
    ) -> Any:
        """Load production model by alias."""
        return await self.production_service.get_production_model(model_name, alias)

    async def list_production_deployments(self) -> List[Dict[str, Any]]:
        """List all production deployments with their status."""
        return await self.production_service.list_production_deployments()

    # Cache and utility methods - delegate appropriately
    async def get_model_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive model cache statistics."""
        try:
            # Clean up expired models first
            cleaned_count = self.model_registry.cleanup_expired()

            # Get cache statistics
            cache_stats = self.model_registry.get_cache_stats()

            # Add cleanup information
            cache_stats["cleaned_expired_models"] = cleaned_count
            cache_stats["cache_efficiency"] = {
                "hit_rate_estimate": "N/A",  # Would need request tracking
                "memory_efficiency": cache_stats["memory_utilization"],
                "active_model_ratio": cache_stats["active_models"]
                / max(cache_stats["total_models"], 1),
            }

            return {
                "status": "success",
                "cache_stats": cache_stats,
                "recommendations": self._generate_cache_recommendations(cache_stats),
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"status": "error", "error": str(e)}

    async def optimize_model_cache(self) -> Dict[str, Any]:
        """Optimize model cache by cleaning expired models and analyzing usage."""
        try:
            start_time = time.time()

            # Clean expired models
            cleaned_count = self.model_registry.cleanup_expired()

            # Get current stats
            stats = self.model_registry.get_cache_stats()

            processing_time = (time.time() - start_time) * 1000

            return {
                "status": "success",
                "cleaned_models": cleaned_count,
                "active_models": stats["active_models"],
                "memory_usage_mb": stats["total_memory_mb"],
                "memory_utilization": stats["memory_utilization"],
                "processing_time_ms": processing_time,
                "recommendations": self._generate_cache_recommendations(stats),
            }

        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
            return {"status": "error", "error": str(e)}

    # Orchestrator integration - delegate to orchestration adapter
    async def run_orchestrated_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrator-compatible interface for ML model operations."""
        return await self.orchestration_adapter.run_orchestrated_analysis(config)

    def _generate_cache_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate cache optimization recommendations."""
        recommendations = []

        memory_util = stats["memory_utilization"]
        if memory_util > 0.9:
            recommendations.append(
                "🔴 High memory usage - consider increasing cache size or reducing TTL"
            )
        elif memory_util > 0.7:
            recommendations.append("🟡 Moderate memory usage - monitor for trends")
        else:
            recommendations.append("🟢 Healthy memory usage")

        if stats["expired_models"] > stats["active_models"]:
            recommendations.append(
                "⏰ Many expired models - consider shorter TTL or more frequent cleanup"
            )

        if stats["total_models"] > 20:
            recommendations.append(
                "📊 Large number of cached models - consider model lifecycle management"
            )

        return recommendations

    def _update_inference_metrics(self, start_time: float) -> None:
        """Update inference performance metrics."""
        duration_ms = (time.time() - start_time) * 1000
        
        # Update rolling average inference time
        total_requests = self._performance_metrics["inference_requests"]
        current_avg = self._performance_metrics["avg_inference_time_ms"]
        
        self._performance_metrics["avg_inference_time_ms"] = (
            (current_avg * (total_requests - 1) + duration_ms) / total_requests
        )

    def _update_pattern_discovery_metrics(self, start_time: float) -> None:
        """Update pattern discovery performance metrics."""
        duration_ms = (time.time() - start_time) * 1000
        
        # Update rolling average pattern discovery time
        total_requests = self._performance_metrics["pattern_discovery_requests"]
        current_avg = self._performance_metrics["avg_pattern_discovery_time_ms"]
        
        if total_requests > 0:
            self._performance_metrics["avg_pattern_discovery_time_ms"] = (
                (current_avg * (total_requests - 1) + duration_ms) / total_requests
            )
        else:
            self._performance_metrics["avg_pattern_discovery_time_ms"] = duration_ms

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive ML facade performance metrics."""
        total_ml_requests = (
            self._performance_metrics["inference_requests"] + 
            self._performance_metrics["pattern_discovery_requests"]
        )
        
        cache_hit_rate = (
            self._performance_metrics["cache_hits"] / 
            max(total_ml_requests, 1)
        )
        
        metrics = {
            "service": "ml_facade",
            "performance": dict(self._performance_metrics),
            "cache_hit_rate": cache_hit_rate,
            "cache_enabled": self._cache_enabled,
            "total_ml_requests": total_ml_requests,
            "timestamp": time.time(),
        }
        
        # Add model registry stats
        try:
            registry_stats = self.model_registry.get_cache_stats()
            metrics["model_registry_stats"] = registry_stats
        except Exception as e:
            logger.warning(f"Failed to get model registry stats: {e}")
        
        # Add cache manager stats if available
        if self.cache_manager:
            try:
                cache_stats = await self.cache_manager.get_stats()
                metrics["cache_manager_stats"] = cache_stats
            except Exception as e:
                logger.warning(f"Failed to get cache manager stats: {e}")
        
        return metrics