"""Repository layer caching for expensive database operations.

This module provides caching decorators and wrappers for repository methods
to achieve <200ms response times with >90% cache hit rates.
"""

import functools
import logging
import time
from datetime import UTC, datetime
from typing import Any, Callable, Dict, List, Optional, Union

from prompt_improver.core.types import CacheType
from prompt_improver.performance.caching.cache_facade import (
    CacheKey,
    CacheStrategy,
    PerformanceCacheFacade,
    get_performance_cache,
)

logger = logging.getLogger(__name__)


class RepositoryCacheDecorator:
    """Caching decorator for repository methods with intelligent invalidation."""

    def __init__(
        self,
        cache_facade: Optional[PerformanceCacheFacade] = None,
        default_strategy: CacheStrategy = CacheStrategy.BALANCED,
        default_namespace: str = "repository",
    ):
        self.cache_facade = cache_facade or get_performance_cache()
        self.default_strategy = default_strategy
        self.default_namespace = default_namespace

    def cache_method(
        self,
        ttl_seconds: Optional[int] = None,
        strategy: Optional[CacheStrategy] = None,
        namespace: Optional[str] = None,
        invalidate_on: Optional[List[str]] = None,
        cache_condition: Optional[Callable] = None,
        key_builder: Optional[Callable] = None,
    ):
        """Decorator for caching repository methods.
        
        Args:
            ttl_seconds: Cache TTL in seconds (overrides strategy)
            strategy: Cache strategy to use
            namespace: Cache namespace
            invalidate_on: List of method names that invalidate this cache
            cache_condition: Function to determine if result should be cached
            key_builder: Custom function to build cache keys
        """
        def decorator(method: Callable) -> Callable:
            # Determine strategy
            cache_strategy = strategy or self.default_strategy
            if ttl_seconds:
                if ttl_seconds <= 60:
                    cache_strategy = CacheStrategy.ULTRA_FAST
                elif ttl_seconds <= 300:
                    cache_strategy = CacheStrategy.FAST
                elif ttl_seconds <= 900:
                    cache_strategy = CacheStrategy.BALANCED
                else:
                    cache_strategy = CacheStrategy.LONG_TERM
            
            # Set up cache namespace
            cache_namespace = namespace or self.default_namespace
            
            @functools.wraps(method)
            async def wrapper(instance, *args, **kwargs):
                # Build cache key
                if key_builder:
                    cache_key = key_builder(instance, method.__name__, args, kwargs)
                else:
                    cache_key = self._build_default_cache_key(
                        instance, method.__name__, args, kwargs, cache_namespace
                    )
                
                # Check if we should cache this call
                if cache_condition and not cache_condition(instance, *args, **kwargs):
                    return await method(instance, *args, **kwargs)
                
                # Get or compute cached value
                result = await self.cache_facade.get_or_compute(
                    cache_key=cache_key,
                    compute_func=method,
                    strategy=cache_strategy,
                    cache_type=CacheType.REPOSITORY,
                    instance=instance,
                    **kwargs,
                )
                
                return result
            
            # Store invalidation info for later use
            wrapper._cache_invalidation = invalidate_on or []
            wrapper._cache_namespace = cache_namespace
            wrapper._cache_method_name = method.__name__
            
            return wrapper
        
        return decorator

    def invalidate_method(
        self,
        invalidate_patterns: Optional[List[str]] = None,
        cascade: bool = False,
    ):
        """Decorator for methods that should invalidate cached data.
        
        Args:
            invalidate_patterns: Patterns of cache keys to invalidate
            cascade: Whether to cascade invalidation to related caches
        """
        def decorator(method: Callable) -> Callable:
            @functools.wraps(method)
            async def wrapper(instance, *args, **kwargs):
                # Execute the method first
                result = await method(instance, *args, **kwargs)
                
                # Perform cache invalidation
                await self._invalidate_caches(
                    instance, method.__name__, invalidate_patterns, cascade
                )
                
                return result
            
            return wrapper
        
        return decorator

    def _build_default_cache_key(
        self, 
        instance: Any,
        method_name: str,
        args: tuple,
        kwargs: dict,
        namespace: str,
    ) -> CacheKey:
        """Build default cache key for repository method."""
        # Get instance identifier
        instance_id = getattr(instance, "__class__", "unknown").__name__
        
        # Build parameters dict
        parameters = {
            "instance_id": instance_id,
            "method": method_name,
            "args": list(args),
            "kwargs": dict(kwargs),
        }
        
        # Remove non-serializable arguments
        parameters = self._sanitize_parameters(parameters)
        
        return CacheKey(
            namespace=namespace,
            operation=f"{instance_id}.{method_name}",
            parameters=parameters,
        )

    def _sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameters to ensure they're serializable."""
        sanitized = {}
        
        for key, value in parameters.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                sanitized[key] = [
                    self._sanitize_value(item) for item in value[:10]  # Limit size
                ]
            elif isinstance(value, dict):
                sanitized[key] = {
                    k: self._sanitize_value(v) for k, v in list(value.items())[:10]
                }
            else:
                # Convert to string representation
                sanitized[key] = str(value)[:100]  # Limit length
        
        return sanitized

    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize individual value."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, datetime):
            return value.isoformat()
        else:
            return str(value)[:100]

    async def _invalidate_caches(
        self,
        instance: Any,
        method_name: str,
        patterns: Optional[List[str]],
        cascade: bool,
    ):
        """Invalidate related caches."""
        try:
            # Get all cached methods on this instance
            cached_methods = [
                attr for attr in dir(instance)
                if hasattr(getattr(instance, attr), '_cache_invalidation')
            ]
            
            # Invalidate based on method relationships
            for cached_method in cached_methods:
                method_obj = getattr(instance, cached_method)
                if hasattr(method_obj, '_cache_invalidation'):
                    invalidation_triggers = method_obj._cache_invalidation
                    if method_name in invalidation_triggers:
                        # Build cache key pattern for this method
                        namespace = getattr(method_obj, '_cache_namespace', 'repository')
                        instance_id = instance.__class__.__name__
                        
                        cache_pattern = f"{namespace}:{instance_id}.{cached_method}:*"
                        
                        # Invalidate matching cache entries
                        await self._invalidate_pattern(cache_pattern)
            
            # Invalidate specific patterns if provided
            if patterns:
                for pattern in patterns:
                    await self._invalidate_pattern(pattern)
                    
        except Exception as e:
            logger.warning(f"Cache invalidation failed for {method_name}: {e}")

    async def _invalidate_pattern(self, pattern: str):
        """Invalidate caches matching a pattern."""
        try:
            from prompt_improver.utils.redis_cache import invalidate
            count = await invalidate(pattern)
            if count > 0:
                logger.debug(f"Invalidated {count} cache entries for pattern: {pattern}")
        except Exception as e:
            logger.warning(f"Failed to invalidate pattern {pattern}: {e}")


# Convenience decorators for common repository caching patterns

def cache_dashboard_data(
    ttl_seconds: int = 300,  # 5 minutes for dashboard data
    strategy: CacheStrategy = CacheStrategy.FAST,
):
    """Cache dashboard data with fast retrieval."""
    decorator = RepositoryCacheDecorator(
        default_strategy=strategy,
        default_namespace="dashboard",
    )
    return decorator.cache_method(
        ttl_seconds=ttl_seconds,
        strategy=strategy,
        invalidate_on=["update_session", "create_session", "delete_session"],
    )


def cache_analytics_query(
    ttl_seconds: int = 900,  # 15 minutes for analytics
    strategy: CacheStrategy = CacheStrategy.BALANCED,
):
    """Cache expensive analytics queries."""
    decorator = RepositoryCacheDecorator(
        default_strategy=strategy,
        default_namespace="analytics",
    )
    return decorator.cache_method(
        ttl_seconds=ttl_seconds,
        strategy=strategy,
        invalidate_on=["bulk_update", "data_import", "recalculate_metrics"],
    )


def cache_time_series_data(
    ttl_seconds: int = 1800,  # 30 minutes for time series
    strategy: CacheStrategy = CacheStrategy.LONG_TERM,
):
    """Cache time-series data queries."""
    def cache_condition(instance, *args, **kwargs) -> bool:
        # Only cache if date range is more than 1 day ago
        start_date = kwargs.get('start_date') or (args[0] if args else None)
        if start_date and hasattr(start_date, 'date'):
            return (datetime.now(UTC).date() - start_date.date()).days >= 1
        return True
    
    decorator = RepositoryCacheDecorator(
        default_strategy=strategy,
        default_namespace="timeseries",
    )
    return decorator.cache_method(
        ttl_seconds=ttl_seconds,
        strategy=strategy,
        cache_condition=cache_condition,
        invalidate_on=["bulk_data_update", "historical_correction"],
    )


def cache_expensive_computation(
    ttl_seconds: int = 3600,  # 1 hour for expensive computations
    strategy: CacheStrategy = CacheStrategy.LONG_TERM,
):
    """Cache expensive computational operations."""
    decorator = RepositoryCacheDecorator(
        default_strategy=strategy,
        default_namespace="computation",
    )
    return decorator.cache_method(
        ttl_seconds=ttl_seconds,
        strategy=strategy,
        invalidate_on=["recalculate", "data_refresh", "model_update"],
    )


def invalidate_on_write(patterns: Optional[List[str]] = None):
    """Decorator to invalidate caches on write operations."""
    decorator = RepositoryCacheDecorator()
    return decorator.invalidate_method(
        invalidate_patterns=patterns,
        cascade=True,
    )


# Enhanced repository base class with automatic caching

class CachedRepositoryMixin:
    """Mixin class to add caching capabilities to repositories."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_facade = get_performance_cache()
        self._cache_stats = {
            "method_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_response_time_ms": 0.0,
        }

    async def _cached_call(
        self,
        method_name: str,
        compute_func: Callable,
        cache_parameters: Dict[str, Any],
        strategy: CacheStrategy = CacheStrategy.BALANCED,
        **compute_kwargs,
    ) -> Any:
        """Make a cached method call with performance tracking."""
        start_time = time.perf_counter()
        self._cache_stats["method_calls"] += 1
        
        cache_key = CacheKey(
            namespace="repository",
            operation=f"{self.__class__.__name__}.{method_name}",
            parameters=cache_parameters,
        )
        
        result = await self._cache_facade.get_or_compute(
            cache_key=cache_key,
            compute_func=compute_func,
            strategy=strategy,
            cache_type=CacheType.REPOSITORY,
            **compute_kwargs,
        )
        
        # Update performance stats
        execution_time = (time.perf_counter() - start_time) * 1000
        self._cache_stats["avg_response_time_ms"] = (
            (self._cache_stats["avg_response_time_ms"] * (self._cache_stats["method_calls"] - 1) 
             + execution_time) / self._cache_stats["method_calls"]
        )
        
        return result

    def get_cache_performance_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics for this repository."""
        return {
            **self._cache_stats,
            "repository_class": self.__class__.__name__,
            "target_response_time_ms": 200,
            "meets_target": self._cache_stats["avg_response_time_ms"] <= 200,
        }


# Example usage: Enhanced Analytics Repository with caching

class CachedAnalyticsRepository(CachedRepositoryMixin):
    """Example of analytics repository with high-performance caching."""

    @cache_dashboard_data(ttl_seconds=300)
    async def get_dashboard_summary_cached(self, period_hours: int = 24) -> Dict[str, Any]:
        """Get dashboard summary with aggressive caching."""
        # This would delegate to the actual repository method
        return await self._cached_call(
            "get_dashboard_summary",
            lambda: self.get_dashboard_summary(period_hours),
            {"period_hours": period_hours},
            strategy=CacheStrategy.FAST,
        )

    @cache_time_series_data(ttl_seconds=1800)
    async def get_time_series_data_cached(
        self,
        metric_name: str,
        start_date: datetime,
        end_date: datetime,
        granularity: str,
    ) -> List[Dict[str, Any]]:
        """Get time-series data with intelligent caching."""
        return await self._cached_call(
            "get_time_series_data",
            lambda: self.get_time_series_data(metric_name, start_date, end_date, granularity),
            {
                "metric_name": metric_name,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "granularity": granularity,
            },
            strategy=CacheStrategy.LONG_TERM,
        )

    @invalidate_on_write(patterns=["dashboard:*", "analytics:*"])
    async def update_analytics_data(self, data: Dict[str, Any]) -> bool:
        """Update analytics data and invalidate related caches."""
        # This would perform the actual update
        return True