"""ML Service caching for model predictions and computations.

This module provides specialized caching for ML operations including model
predictions, feature computations, and analysis results to achieve <200ms
response times for ML workloads.
"""

import hashlib
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

from prompt_improver.core.interfaces.ml_interface import (
    MLAnalysisResult,
    MLModelType,
    MLHealthReport,
)
from prompt_improver.core.types import CacheType
from prompt_improver.performance.caching.cache_facade import (
    CacheKey,
    CacheStrategy,
    PerformanceCacheFacade,
    get_performance_cache,
    cache_ml_prediction,
)

logger = logging.getLogger(__name__)


class MLServiceCache:
    """High-performance caching for ML service operations.
    
    Provides specialized caching strategies for different ML workload patterns:
    - Model predictions: Fast caching with input-based keys
    - Feature computations: Balanced caching with parameter-based keys
    - Analysis results: Long-term caching for expensive operations
    """

    def __init__(self, cache_facade: Optional[PerformanceCacheFacade] = None):
        self.cache_facade = cache_facade or get_performance_cache()
        self._prediction_stats = {
            "predictions_served": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_prediction_time_ms": 0.0,
            "total_time_saved_ms": 0.0,
        }

    async def cached_model_prediction(
        self,
        model_id: str,
        input_data: Dict[str, Any],
        predict_func: Callable,
        model_type: Optional[MLModelType] = None,
        ttl_seconds: int = 300,  # 5 minutes default
        cache_strategy: CacheStrategy = CacheStrategy.FAST,
        **predict_kwargs,
    ) -> Dict[str, Any]:
        """Cache model predictions with input-based key generation.
        
        Args:
            model_id: Unique identifier for the model
            input_data: Input data for prediction
            predict_func: Function to perform prediction
            model_type: Type of ML model
            ttl_seconds: Cache TTL in seconds
            cache_strategy: Caching strategy
            **predict_kwargs: Additional arguments for prediction function
            
        Returns:
            Cached or computed prediction result
        """
        start_time = time.perf_counter()
        self._prediction_stats["predictions_served"] += 1

        # Generate cache key based on model and input
        cache_key = self._build_prediction_cache_key(
            model_id, input_data, model_type
        )

        # Use the facade for caching
        result = await self.cache_facade.get_or_compute(
            cache_key=cache_key,
            compute_func=predict_func,
            strategy=cache_strategy,
            cache_type=CacheType.ML,
            input_data=input_data,
            **predict_kwargs,
        )

        # Update prediction statistics
        execution_time = (time.perf_counter() - start_time) * 1000
        self._update_prediction_stats(execution_time, cache_hit=result.get("_cached", False))

        # Ensure result has prediction metadata
        if isinstance(result, dict):
            result.setdefault("model_id", model_id)
            result.setdefault("prediction_timestamp", datetime.now(UTC).isoformat())
            result.setdefault("cached", result.get("_cached", False))

        return result

    async def cached_feature_extraction(
        self,
        feature_extractor_name: str,
        input_text: str,
        extraction_func: Callable,
        extraction_params: Optional[Dict[str, Any]] = None,
        ttl_seconds: int = 900,  # 15 minutes default
        **extraction_kwargs,
    ) -> Dict[str, Any]:
        """Cache feature extraction results.
        
        Args:
            feature_extractor_name: Name of the feature extractor
            input_text: Text to extract features from
            extraction_func: Function to perform extraction
            extraction_params: Parameters for extraction
            ttl_seconds: Cache TTL in seconds
            **extraction_kwargs: Additional arguments for extraction function
            
        Returns:
            Cached or computed feature extraction result
        """
        # Generate deterministic key for text + parameters
        cache_key = self._build_feature_cache_key(
            feature_extractor_name, input_text, extraction_params or {}
        )

        return await self.cache_facade.get_or_compute(
            cache_key=cache_key,
            compute_func=extraction_func,
            strategy=CacheStrategy.BALANCED,
            cache_type=CacheType.ML,
            input_text=input_text,
            extraction_params=extraction_params,
            **extraction_kwargs,
        )

    async def cached_ml_analysis(
        self,
        analysis_type: str,
        analysis_data: Dict[str, Any],
        analysis_func: Callable,
        ttl_seconds: int = 1800,  # 30 minutes default
        **analysis_kwargs,
    ) -> MLAnalysisResult:
        """Cache ML analysis results.
        
        Args:
            analysis_type: Type of analysis (e.g., "pattern_analysis")
            analysis_data: Data to analyze
            analysis_func: Function to perform analysis
            ttl_seconds: Cache TTL in seconds
            **analysis_kwargs: Additional arguments for analysis function
            
        Returns:
            Cached or computed analysis result
        """
        cache_key = self._build_analysis_cache_key(analysis_type, analysis_data)

        result = await self.cache_facade.get_or_compute(
            cache_key=cache_key,
            compute_func=analysis_func,
            strategy=CacheStrategy.LONG_TERM,
            cache_type=CacheType.ML,
            analysis_data=analysis_data,
            **analysis_kwargs,
        )

        # Ensure result is properly typed
        if not isinstance(result, MLAnalysisResult):
            if isinstance(result, dict):
                result = MLAnalysisResult(
                    analysis_id=result.get("analysis_id", "cached"),
                    analysis_type=analysis_type,
                    results=result,
                    confidence_score=result.get("confidence_score", 0.0),
                    processing_time_ms=result.get("processing_time_ms", 0),
                    timestamp=datetime.now(UTC),
                )

        return result

    async def cached_batch_predictions(
        self,
        model_id: str,
        batch_inputs: List[Dict[str, Any]],
        batch_predict_func: Callable,
        batch_size: int = 10,
        ttl_seconds: int = 300,
        **predict_kwargs,
    ) -> List[Dict[str, Any]]:
        """Cache batch predictions with intelligent batching.
        
        Args:
            model_id: Model identifier
            batch_inputs: List of input data for batch prediction
            batch_predict_func: Function to perform batch prediction
            batch_size: Size of cache-checking batches
            ttl_seconds: Cache TTL in seconds
            **predict_kwargs: Additional arguments for prediction function
            
        Returns:
            List of cached or computed prediction results
        """
        results = []
        uncached_inputs = []
        uncached_indices = []

        # Check cache for each input
        for i, input_data in enumerate(batch_inputs):
            cache_key = self._build_prediction_cache_key(model_id, input_data)
            
            cached_result = await self.cache_facade.get_or_compute(
                cache_key=cache_key,
                compute_func=lambda: None,  # Don't compute yet
                strategy=CacheStrategy.FAST,
                cache_type=CacheType.ML,
                force_refresh=False,
            )
            
            if cached_result is not None:
                results.append(cached_result)
                self._prediction_stats["cache_hits"] += 1
            else:
                results.append(None)  # Placeholder
                uncached_inputs.append(input_data)
                uncached_indices.append(i)
                self._prediction_stats["cache_misses"] += 1

        # Process uncached inputs in batches
        if uncached_inputs:
            for batch_start in range(0, len(uncached_inputs), batch_size):
                batch_end = min(batch_start + batch_size, len(uncached_inputs))
                batch_data = uncached_inputs[batch_start:batch_end]
                
                # Compute batch predictions
                batch_results = await batch_predict_func(batch_data, **predict_kwargs)
                
                # Cache and store results
                for j, (input_data, prediction_result) in enumerate(zip(batch_data, batch_results)):
                    original_index = uncached_indices[batch_start + j]
                    
                    # Cache the individual result
                    cache_key = self._build_prediction_cache_key(model_id, input_data)
                    await self.cache_facade.set_cached_value(
                        cache_key=cache_key,
                        value=prediction_result,
                        strategy=CacheStrategy.FAST,
                        cache_type=CacheType.ML,
                    )
                    
                    # Store in results
                    results[original_index] = prediction_result

        self._prediction_stats["predictions_served"] += len(batch_inputs)
        return results

    async def invalidate_model_cache(
        self,
        model_id: str,
        invalidate_related: bool = True,
    ) -> int:
        """Invalidate all cached data for a model.
        
        Args:
            model_id: Model ID to invalidate
            invalidate_related: Whether to invalidate related caches
            
        Returns:
            Number of cache entries invalidated
        """
        try:
            from prompt_improver.utils.redis_cache import invalidate
            
            patterns = [f"ml_predictions:predict_{model_id}:*"]
            
            if invalidate_related:
                patterns.extend([
                    f"ml:*model_id*{model_id}*",
                    f"ml_features:*{model_id}*",
                    f"ml_analysis:*{model_id}*",
                ])
            
            total_invalidated = 0
            for pattern in patterns:
                count = await invalidate(pattern)
                total_invalidated += count
                
            logger.info(f"Invalidated {total_invalidated} cache entries for model {model_id}")
            return total_invalidated
            
        except Exception as e:
            logger.error(f"Failed to invalidate model cache for {model_id}: {e}")
            return 0

    async def warm_prediction_cache(
        self,
        model_id: str,
        common_inputs: List[Dict[str, Any]],
        predict_func: Callable,
        ttl_seconds: int = 300,
    ) -> Dict[str, bool]:
        """Warm cache with common prediction inputs.
        
        Args:
            model_id: Model identifier
            common_inputs: List of common input patterns
            predict_func: Function to generate predictions
            ttl_seconds: Cache TTL for warmed entries
            
        Returns:
            Dictionary mapping input hashes to success status
        """
        cache_items = []
        
        for input_data in common_inputs:
            try:
                # Generate prediction
                prediction = await predict_func(input_data)
                
                # Build cache item
                cache_key = self._build_prediction_cache_key(model_id, input_data)
                cache_items.append({
                    "key": cache_key,
                    "value": prediction,
                })
                
            except Exception as e:
                logger.warning(f"Failed to warm cache for input {input_data}: {e}")
        
        # Batch warm the cache
        return await self.cache_facade.warm_cache(
            cache_items=cache_items,
            strategy=CacheStrategy.FAST,
            cache_type=CacheType.ML,
        )

    def _build_prediction_cache_key(
        self,
        model_id: str,
        input_data: Dict[str, Any],
        model_type: Optional[MLModelType] = None,
    ) -> CacheKey:
        """Build cache key for model prediction."""
        # Create deterministic representation of input data
        input_hash = self._hash_input_data(input_data)
        
        parameters = {
            "model_id": model_id,
            "input_hash": input_hash,
            "model_type": model_type.value if model_type else None,
        }

        return CacheKey(
            namespace="ml_predictions",
            operation=f"predict_{model_id}",
            parameters=parameters,
        )

    def _build_feature_cache_key(
        self,
        extractor_name: str,
        input_text: str,
        params: Dict[str, Any],
    ) -> CacheKey:
        """Build cache key for feature extraction."""
        # Hash the input text to handle large inputs
        text_hash = self._hash_text(input_text)
        
        parameters = {
            "extractor": extractor_name,
            "text_hash": text_hash,
            "text_length": len(input_text),
            "params": params,
        }

        return CacheKey(
            namespace="ml_features",
            operation=f"extract_{extractor_name}",
            parameters=parameters,
        )

    def _build_analysis_cache_key(
        self,
        analysis_type: str,
        analysis_data: Dict[str, Any],
    ) -> CacheKey:
        """Build cache key for ML analysis."""
        # Hash the analysis data
        data_hash = self._hash_dict(analysis_data)
        
        parameters = {
            "analysis_type": analysis_type,
            "data_hash": data_hash,
            "data_size": len(str(analysis_data)),
        }

        return CacheKey(
            namespace="ml_analysis",
            operation=f"analyze_{analysis_type}",
            parameters=parameters,
        )

    def _hash_input_data(self, input_data: Dict[str, Any]) -> str:
        """Generate hash for input data."""
        import json
        
        # Normalize and sort for consistent hashing
        normalized = self._normalize_dict(input_data)
        json_str = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
        
        return hashlib.md5(json_str.encode()).hexdigest()

    def _hash_text(self, text: str) -> str:
        """Generate hash for text input."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Generate hash for dictionary data."""
        import json
        
        normalized = self._normalize_dict(data)
        json_str = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
        
        return hashlib.md5(json_str.encode()).hexdigest()

    def _normalize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize dictionary for consistent hashing."""
        normalized = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                normalized[key] = self._normalize_dict(value)
            elif isinstance(value, list):
                # Sort lists if they contain comparable items
                try:
                    normalized[key] = sorted(value) if value and isinstance(value[0], (str, int, float)) else value
                except (TypeError, IndexError):
                    normalized[key] = value
            elif isinstance(value, (str, int, float, bool, type(None))):
                normalized[key] = value
            else:
                # Convert other types to string
                normalized[key] = str(value)
        
        return normalized

    def _update_prediction_stats(self, execution_time_ms: float, cache_hit: bool):
        """Update prediction performance statistics."""
        if cache_hit:
            self._prediction_stats["cache_hits"] += 1
            # Estimate time saved (assume uncached prediction would take 50ms)
            self._prediction_stats["total_time_saved_ms"] += max(0, 50 - execution_time_ms)
        else:
            self._prediction_stats["cache_misses"] += 1

        # Update average
        total_predictions = self._prediction_stats["predictions_served"]
        if total_predictions > 0:
            current_avg = self._prediction_stats["avg_prediction_time_ms"]
            self._prediction_stats["avg_prediction_time_ms"] = (
                (current_avg * (total_predictions - 1) + execution_time_ms) / total_predictions
            )

    async def get_ml_cache_performance(self) -> Dict[str, Any]:
        """Get ML cache performance statistics."""
        cache_facade_stats = await self.cache_facade.get_performance_stats()
        
        total_predictions = self._prediction_stats["predictions_served"]
        cache_hit_rate = (
            self._prediction_stats["cache_hits"] / total_predictions
            if total_predictions > 0 else 0
        )
        
        return {
            "ml_cache_stats": {
                **self._prediction_stats,
                "cache_hit_rate": cache_hit_rate,
                "target_hit_rate": 0.9,
                "meets_hit_rate_target": cache_hit_rate >= 0.9,
                "target_response_time_ms": 200,
                "meets_response_time_target": self._prediction_stats["avg_prediction_time_ms"] <= 200,
            },
            "cache_facade_stats": cache_facade_stats,
            "performance_improvements": {
                "estimated_response_time_improvement": f"{cache_hit_rate * 100:.1f}%",
                "total_time_saved_minutes": self._prediction_stats["total_time_saved_ms"] / 60000,
                "database_load_reduction": f"{cache_hit_rate * 100:.1f}%",
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on ML cache."""
        try:
            # Test prediction caching
            test_input = {"test": "data", "timestamp": datetime.now(UTC).isoformat()}
            test_model_id = "health_check_model"
            
            start_time = time.perf_counter()
            
            result = await self.cached_model_prediction(
                model_id=test_model_id,
                input_data=test_input,
                predict_func=lambda **kwargs: {"prediction": "test", "confidence": 0.95},
                ttl_seconds=60,
            )
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            # Clean up test cache
            await self.invalidate_model_cache(test_model_id)
            
            return {
                "healthy": True,
                "ml_cache_performance": {
                    "test_prediction_time_ms": execution_time,
                    "meets_fast_target": execution_time < 50,
                },
                "performance_stats": await self.get_ml_cache_performance(),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            
        except Exception as e:
            logger.error(f"ML cache health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }


# Global ML service cache instance
_global_ml_cache: Optional[MLServiceCache] = None


def get_ml_service_cache() -> MLServiceCache:
    """Get the global ML service cache instance."""
    global _global_ml_cache
    if _global_ml_cache is None:
        _global_ml_cache = MLServiceCache()
    return _global_ml_cache


# Convenience functions for ML caching

async def cache_model_prediction(
    model_id: str,
    input_data: Dict[str, Any],
    predict_func: Callable,
    ttl_seconds: int = 300,
    **predict_kwargs,
) -> Dict[str, Any]:
    """Cache a model prediction with automatic key generation."""
    ml_cache = get_ml_service_cache()
    return await ml_cache.cached_model_prediction(
        model_id=model_id,
        input_data=input_data,
        predict_func=predict_func,
        ttl_seconds=ttl_seconds,
        **predict_kwargs,
    )


async def cache_feature_extraction(
    extractor_name: str,
    input_text: str,
    extraction_func: Callable,
    ttl_seconds: int = 900,
    **extraction_kwargs,
) -> Dict[str, Any]:
    """Cache feature extraction with automatic key generation."""
    ml_cache = get_ml_service_cache()
    return await ml_cache.cached_feature_extraction(
        feature_extractor_name=extractor_name,
        input_text=input_text,
        extraction_func=extraction_func,
        ttl_seconds=ttl_seconds,
        **extraction_kwargs,
    )


async def cache_analysis_result(
    analysis_type: str,
    analysis_data: Dict[str, Any],
    analysis_func: Callable,
    ttl_seconds: int = 1800,
    **analysis_kwargs,
) -> MLAnalysisResult:
    """Cache ML analysis results with automatic key generation."""
    ml_cache = get_ml_service_cache()
    return await ml_cache.cached_ml_analysis(
        analysis_type=analysis_type,
        analysis_data=analysis_data,
        analysis_func=analysis_func,
        ttl_seconds=ttl_seconds,
        **analysis_kwargs,
    )