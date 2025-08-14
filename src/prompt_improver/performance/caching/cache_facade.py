"""Unified performance caching facade for <200ms response times.

This module integrates with existing multi-level cache infrastructure and 
provides high-performance caching strategies for different workload patterns.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from prompt_improver.core.types import CacheType
from prompt_improver.performance.optimization.performance_optimizer import (
    measure_cache_operation,
    get_performance_optimizer,
)
from prompt_improver.services.cache.cache_facade import (
    CacheFacade as MultiLevelCache, 
    get_cache,
    cached_get,
    cached_set,
)

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Performance-optimized cache strategies."""
    
    # Ultra-fast: <1ms response times
    ULTRA_FAST = "ultra_fast"  # L1 only, 60s TTL
    
    # Fast: <5ms response times  
    FAST = "fast"  # L1 + L2, 300s TTL
    
    # Balanced: <50ms response times
    BALANCED = "balanced"  # All levels, 900s TTL
    
    # Long-term: <200ms response times
    LONG_TERM = "long_term"  # All levels, 3600s TTL


@dataclass
class CacheKey:
    """Structured cache key for performance optimization."""
    
    namespace: str
    operation: str
    parameters: Dict[str, Any]
    version: str = "1.0"
    
    def to_string(self) -> str:
        """Generate cache key string."""
        # Sort parameters for consistent keys
        sorted_params = json.dumps(self.parameters, sort_keys=True, separators=(',', ':'))
        return f"{self.namespace}:{self.operation}:{sorted_params}:v{self.version}"
    
    def get_hash(self) -> str:
        """Get hash for cache key to avoid length issues."""
        import hashlib
        key_string = self.to_string()
        if len(key_string) > 200:  # Redis key length limit
            return f"{self.namespace}:{self.operation}:hash:{hashlib.md5(key_string.encode()).hexdigest()}"
        return key_string


class PerformanceCacheFacade:
    """Unified high-performance caching facade.
    
    Integrates with existing multi-level cache infrastructure for optimal
    performance across different workload patterns and response time targets.
    """
    
    def __init__(self):
        self.performance_optimizer = get_performance_optimizer()
        self._cache_stats = {
            "operations": 0,
            "hits": 0,
            "misses": 0,
            "time_saved_ms": 0.0,
        }
        
    async def get_or_compute(
        self,
        cache_key: Union[CacheKey, str],
        compute_func: Callable,
        strategy: CacheStrategy = CacheStrategy.BALANCED,
        cache_type: CacheType = CacheType.PROMPT,
        force_refresh: bool = False,
        **compute_kwargs,
    ) -> Any:
        """Get value from cache or compute it.
        
        Args:
            cache_key: Cache key (structured or string)
            compute_func: Function to compute value if not cached
            strategy: Caching strategy for performance target
            cache_type: Type of cache to use
            force_refresh: Force cache refresh
            **compute_kwargs: Arguments for compute function
            
        Returns:
            Cached or computed value
        """
        self._cache_stats["operations"] += 1
        
        # Convert cache key
        key_str = cache_key.to_string() if isinstance(cache_key, CacheKey) else cache_key
        
        # Get cache configuration based on strategy
        cache_config = self._get_cache_config(strategy)
        
        if not force_refresh:
            # Try cache first
            async with measure_cache_operation(f"get_{strategy.value}") as perf_metrics:
                cached_value = await cached_get(
                    key_str,
                    fallback_func=None,
                    cache_type=cache_type,
                    ttl=cache_config["ttl"],
                )
                
                if cached_value is not None:
                    self._cache_stats["hits"] += 1
                    # Estimate time saved based on strategy
                    estimated_compute_time = cache_config["estimated_compute_time_ms"]
                    self._cache_stats["time_saved_ms"] += estimated_compute_time
                    
                    perf_metrics.metadata.update({
                        "cache_hit": True,
                        "strategy": strategy.value,
                        "time_saved_ms": estimated_compute_time,
                    })
                    
                    return cached_value
        
        # Cache miss - compute value
        self._cache_stats["misses"] += 1
        
        async with self.performance_optimizer.measure_operation(
            f"compute_{strategy.value}", 
            cache_key=key_str,
            strategy=strategy.value,
        ) as compute_metrics:
            
            if asyncio.iscoroutinefunction(compute_func):
                computed_value = await compute_func(**compute_kwargs)
            else:
                computed_value = compute_func(**compute_kwargs)
            
            # Cache the computed value
            await cached_set(
                key_str,
                computed_value, 
                cache_type=cache_type,
                ttl=cache_config["ttl"],
            )
            
            compute_metrics.metadata.update({
                "cache_miss": True,
                "strategy": strategy.value,
                "computed": True,
            })
            
            return computed_value
    
    async def set_cached_value(
        self,
        cache_key: Union[CacheKey, str],
        value: Any,
        strategy: CacheStrategy = CacheStrategy.BALANCED,
        cache_type: CacheType = CacheType.PROMPT,
    ) -> None:
        """Set value in cache with strategy-based configuration.
        
        Args:
            cache_key: Cache key (structured or string)
            value: Value to cache
            strategy: Caching strategy
            cache_type: Type of cache to use
        """
        key_str = cache_key.to_string() if isinstance(cache_key, CacheKey) else cache_key
        cache_config = self._get_cache_config(strategy)
        
        await cached_set(
            key_str,
            value,
            cache_type=cache_type, 
            ttl=cache_config["ttl"],
        )
    
    async def invalidate(
        self,
        cache_key: Union[CacheKey, str],
        cache_type: CacheType = CacheType.PROMPT,
    ) -> None:
        """Invalidate cache entry.
        
        Args:
            cache_key: Cache key to invalidate
            cache_type: Type of cache
        """
        key_str = cache_key.to_string() if isinstance(cache_key, CacheKey) else cache_key
        cache = get_cache(cache_type)
        await cache.delete(key_str)
    
    async def batch_get(
        self,
        cache_keys: List[Union[CacheKey, str]],
        strategy: CacheStrategy = CacheStrategy.BALANCED,
        cache_type: CacheType = CacheType.PROMPT,
    ) -> Dict[str, Any]:
        """Batch get multiple cache values.
        
        Args:
            cache_keys: List of cache keys
            strategy: Caching strategy
            cache_type: Type of cache
            
        Returns:
            Dictionary mapping keys to values (None if not found)
        """
        cache_config = self._get_cache_config(strategy)
        cache = get_cache(cache_type)
        
        # Convert keys and batch retrieve
        results = {}
        tasks = []
        key_mapping = {}
        
        for cache_key in cache_keys:
            key_str = cache_key.to_string() if isinstance(cache_key, CacheKey) else cache_key
            key_mapping[key_str] = cache_key
            tasks.append(cache.get(key_str))
        
        values = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (key_str, value) in enumerate(zip(key_mapping.keys(), values)):
            original_key = key_mapping[key_str]
            if isinstance(value, Exception):
                logger.warning(f"Failed to get cache key {key_str}: {value}")
                results[str(original_key)] = None
            else:
                results[str(original_key)] = value
                if value is not None:
                    self._cache_stats["hits"] += 1
                else:
                    self._cache_stats["misses"] += 1
        
        self._cache_stats["operations"] += len(cache_keys)
        return results
    
    async def warm_cache(
        self,
        cache_items: List[Dict[str, Any]],
        strategy: CacheStrategy = CacheStrategy.BALANCED,
        cache_type: CacheType = CacheType.PROMPT,
    ) -> Dict[str, bool]:
        """Warm cache with multiple items.
        
        Args:
            cache_items: List of items with 'key' and 'value' 
            strategy: Caching strategy
            cache_type: Type of cache
            
        Returns:
            Dictionary mapping keys to success status
        """
        cache_config = self._get_cache_config(strategy)
        results = {}
        
        # Batch warm cache
        tasks = []
        keys = []
        
        for item in cache_items:
            cache_key = item["key"]
            value = item["value"]
            
            key_str = cache_key.to_string() if isinstance(cache_key, CacheKey) else cache_key
            keys.append(key_str)
            
            tasks.append(cached_set(
                key_str,
                value,
                cache_type=cache_type,
                ttl=cache_config["ttl"],
            ))
        
        # Execute warming operations
        success_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (key_str, result) in enumerate(zip(keys, success_results)):
            if isinstance(result, Exception):
                logger.warning(f"Failed to warm cache key {key_str}: {result}")
                results[key_str] = False
            else:
                results[key_str] = True
        
        return results
    
    def _get_cache_config(self, strategy: CacheStrategy) -> Dict[str, Any]:
        """Get cache configuration for strategy.
        
        Args:
            strategy: Caching strategy
            
        Returns:
            Cache configuration dictionary
        """
        configs = {
            CacheStrategy.ULTRA_FAST: {
                "ttl": 60,  # 1 minute
                "estimated_compute_time_ms": 1,
                "target_response_ms": 1,
                "description": "Ultra-fast L1 only caching",
            },
            CacheStrategy.FAST: {
                "ttl": 300,  # 5 minutes
                "estimated_compute_time_ms": 5,
                "target_response_ms": 5,
                "description": "Fast L1+L2 caching",
            },
            CacheStrategy.BALANCED: {
                "ttl": 900,  # 15 minutes
                "estimated_compute_time_ms": 50,
                "target_response_ms": 50,
                "description": "Balanced multi-level caching",
            },
            CacheStrategy.LONG_TERM: {
                "ttl": 3600,  # 1 hour
                "estimated_compute_time_ms": 200,
                "target_response_ms": 200,
                "description": "Long-term multi-level caching",
            },
        }
        
        return configs[strategy]
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for cache facade.
        
        Returns:
            Performance statistics dictionary
        """
        total_ops = self._cache_stats["operations"]
        hit_rate = self._cache_stats["hits"] / total_ops if total_ops > 0 else 0
        
        return {
            "total_operations": total_ops,
            "cache_hits": self._cache_stats["hits"],
            "cache_misses": self._cache_stats["misses"],
            "hit_rate": hit_rate,
            "total_time_saved_ms": self._cache_stats["time_saved_ms"],
            "avg_time_saved_per_hit_ms": (
                self._cache_stats["time_saved_ms"] / self._cache_stats["hits"]
                if self._cache_stats["hits"] > 0 else 0
            ),
            "performance_improvement": {
                "hit_rate_target": ">90%",
                "hit_rate_actual": f"{hit_rate:.1%}",
                "meets_target": hit_rate >= 0.9,
                "estimated_response_time_improvement": f"{hit_rate * 100:.1f}%",
            },
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache facade.
        
        Returns:
            Health check results
        """
        try:
            # Test cache operation
            test_key = CacheKey(
                namespace="health",
                operation="test",
                parameters={"timestamp": datetime.now(UTC).isoformat()},
            )
            
            test_value = {"test": True, "timestamp": time.time()}
            
            # Test set operation
            start_time = time.perf_counter()
            await self.set_cached_value(
                test_key, 
                test_value,
                strategy=CacheStrategy.FAST,
            )
            set_time = (time.perf_counter() - start_time) * 1000
            
            # Test get operation
            start_time = time.perf_counter()
            retrieved_value = await self.get_or_compute(
                test_key,
                lambda: {"fallback": True},
                strategy=CacheStrategy.FAST,
            )
            get_time = (time.perf_counter() - start_time) * 1000
            
            # Clean up test key
            await self.invalidate(test_key)
            
            return {
                "healthy": True,
                "performance": {
                    "set_operation_ms": set_time,
                    "get_operation_ms": get_time,
                    "meets_fast_target": get_time < 5,
                },
                "cache_stats": await self.get_performance_stats(),
                "test_result": {
                    "set_success": True,
                    "get_success": retrieved_value is not None,
                    "value_match": retrieved_value == test_value,
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Cache facade health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }


# Global cache facade instance
_global_cache_facade: Optional[PerformanceCacheFacade] = None


def get_performance_cache() -> PerformanceCacheFacade:
    """Get the global performance cache facade instance."""
    global _global_cache_facade
    if _global_cache_facade is None:
        _global_cache_facade = PerformanceCacheFacade()
    return _global_cache_facade


# Convenience functions for common patterns

async def cache_expensive_operation(
    operation_name: str,
    compute_func: Callable,
    parameters: Dict[str, Any],
    namespace: str = "operations",
    strategy: CacheStrategy = CacheStrategy.BALANCED,
    **compute_kwargs,
) -> Any:
    """Cache an expensive operation with automatic key generation.
    
    Args:
        operation_name: Name of the operation
        compute_func: Function to compute result
        parameters: Parameters that affect the result
        namespace: Cache namespace
        strategy: Caching strategy
        **compute_kwargs: Arguments for compute function
        
    Returns:
        Cached or computed result
    """
    cache_facade = get_performance_cache()
    cache_key = CacheKey(
        namespace=namespace,
        operation=operation_name,
        parameters=parameters,
    )
    
    return await cache_facade.get_or_compute(
        cache_key,
        compute_func,
        strategy=strategy,
        **compute_kwargs,
    )


async def cache_database_query(
    query_name: str,
    query_func: Callable,
    query_parameters: Dict[str, Any],
    ttl_seconds: int = 900,
    **query_kwargs,
) -> Any:
    """Cache a database query with optimized settings.
    
    Args:
        query_name: Name of the query
        query_func: Function to execute query
        query_parameters: Query parameters
        ttl_seconds: Cache TTL in seconds
        **query_kwargs: Arguments for query function
        
    Returns:
        Cached or fresh query result
    """
    # Choose strategy based on TTL
    if ttl_seconds <= 60:
        strategy = CacheStrategy.ULTRA_FAST
    elif ttl_seconds <= 300:
        strategy = CacheStrategy.FAST
    elif ttl_seconds <= 900:
        strategy = CacheStrategy.BALANCED
    else:
        strategy = CacheStrategy.LONG_TERM
    
    return await cache_expensive_operation(
        operation_name=query_name,
        compute_func=query_func,
        parameters=query_parameters,
        namespace="database",
        strategy=strategy,
        **query_kwargs,
    )


async def cache_ml_prediction(
    model_id: str,
    predict_func: Callable,
    input_data: Dict[str, Any],
    ttl_seconds: int = 300,
    **predict_kwargs,
) -> Any:
    """Cache ML model prediction with optimized settings.
    
    Args:
        model_id: Model identifier
        predict_func: Prediction function
        input_data: Model input data
        ttl_seconds: Cache TTL in seconds
        **predict_kwargs: Arguments for prediction function
        
    Returns:
        Cached or fresh prediction result
    """
    return await cache_expensive_operation(
        operation_name=f"predict_{model_id}",
        compute_func=predict_func,
        parameters={"input": input_data, "model_id": model_id},
        namespace="ml_predictions",
        strategy=CacheStrategy.FAST if ttl_seconds <= 300 else CacheStrategy.BALANCED,
        **predict_kwargs,
    )


async def cache_analytics_computation(
    computation_name: str,
    compute_func: Callable,
    analysis_parameters: Dict[str, Any],
    ttl_seconds: int = 600,
    **compute_kwargs,
) -> Any:
    """Cache analytics computation with optimized settings.
    
    Args:
        computation_name: Name of the computation
        compute_func: Function to perform computation
        analysis_parameters: Analysis parameters
        ttl_seconds: Cache TTL in seconds
        **compute_kwargs: Arguments for computation function
        
    Returns:
        Cached or fresh computation result
    """
    return await cache_expensive_operation(
        operation_name=computation_name,
        compute_func=compute_func,
        parameters=analysis_parameters,
        namespace="analytics",
        strategy=CacheStrategy.BALANCED,
        **compute_kwargs,
    )