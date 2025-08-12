"""API response caching for sub-200ms endpoint response times.

This module provides HTTP response caching with intelligent invalidation
and conditional caching based on request patterns and response characteristics.
"""

import json
import logging
import time
from datetime import UTC, datetime
from typing import Any, Callable, Dict, List, Optional, Union

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from prompt_improver.performance.caching.cache_facade import (
    CacheKey,
    CacheStrategy,
    PerformanceCacheFacade,
    get_performance_cache,
)

logger = logging.getLogger(__name__)


class APIResponseCache:
    """High-performance API response caching with intelligent strategies."""

    def __init__(self, cache_facade: Optional[PerformanceCacheFacade] = None):
        self.cache_facade = cache_facade or get_performance_cache()
        self._api_stats = {
            "requests_served": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_response_time_ms": 0.0,
            "total_time_saved_ms": 0.0,
        }

    async def cached_endpoint_response(
        self,
        endpoint_name: str,
        request: Request,
        response_func: Callable,
        ttl_seconds: int = 300,
        cache_strategy: CacheStrategy = CacheStrategy.BALANCED,
        cache_condition: Optional[Callable[[Request], bool]] = None,
        vary_headers: Optional[List[str]] = None,
        **response_kwargs,
    ) -> Union[Response, JSONResponse, Dict[str, Any]]:
        """Cache API endpoint response with intelligent key generation.
        
        Args:
            endpoint_name: Name of the API endpoint
            request: FastAPI request object
            response_func: Function to generate response
            ttl_seconds: Cache TTL in seconds
            cache_strategy: Caching strategy
            cache_condition: Function to determine if response should be cached
            vary_headers: Headers to include in cache key
            **response_kwargs: Additional arguments for response function
            
        Returns:
            Cached or generated response
        """
        start_time = time.perf_counter()
        self._api_stats["requests_served"] += 1

        # Check if we should cache this request
        if cache_condition and not cache_condition(request):
            return await self._execute_response_func(response_func, request, **response_kwargs)

        # Generate cache key
        cache_key = await self._build_api_cache_key(
            endpoint_name, request, vary_headers or []
        )

        # Get or compute cached response
        cached_response = await self.cache_facade.get_or_compute(
            cache_key=cache_key,
            compute_func=self._execute_and_serialize_response,
            strategy=cache_strategy,
            response_func=response_func,
            request=request,
            **response_kwargs,
        )

        # Update API statistics
        execution_time = (time.perf_counter() - start_time) * 1000
        self._update_api_stats(execution_time, cache_hit=cached_response.get("_cached", False))

        # Deserialize and return response
        return self._deserialize_response(cached_response)

    async def cached_dashboard_endpoint(
        self,
        endpoint_name: str,
        request: Request,
        response_func: Callable,
        **response_kwargs,
    ) -> Union[Response, JSONResponse, Dict[str, Any]]:
        """Cache dashboard endpoint with fast retrieval strategy."""
        return await self.cached_endpoint_response(
            endpoint_name=endpoint_name,
            request=request,
            response_func=response_func,
            ttl_seconds=60,  # 1 minute for dashboard
            cache_strategy=CacheStrategy.ULTRA_FAST,
            cache_condition=self._should_cache_dashboard_request,
            **response_kwargs,
        )

    async def cached_analytics_endpoint(
        self,
        endpoint_name: str,
        request: Request,
        response_func: Callable,
        **response_kwargs,
    ) -> Union[Response, JSONResponse, Dict[str, Any]]:
        """Cache analytics endpoint with balanced strategy."""
        return await self.cached_endpoint_response(
            endpoint_name=endpoint_name,
            request=request,
            response_func=response_func,
            ttl_seconds=600,  # 10 minutes for analytics
            cache_strategy=CacheStrategy.BALANCED,
            cache_condition=self._should_cache_analytics_request,
            vary_headers=["Authorization", "User-Agent"],
            **response_kwargs,
        )

    async def cached_time_series_endpoint(
        self,
        endpoint_name: str,
        request: Request,
        response_func: Callable,
        **response_kwargs,
    ) -> Union[Response, JSONResponse, Dict[str, Any]]:
        """Cache time-series endpoint with long-term strategy."""
        return await self.cached_endpoint_response(
            endpoint_name=endpoint_name,
            request=request,
            response_func=response_func,
            ttl_seconds=1800,  # 30 minutes for time-series
            cache_strategy=CacheStrategy.LONG_TERM,
            cache_condition=self._should_cache_timeseries_request,
            **response_kwargs,
        )

    async def invalidate_endpoint_cache(
        self,
        endpoint_pattern: str,
        user_specific: bool = False,
        cascade: bool = True,
    ) -> int:
        """Invalidate cached responses for endpoint pattern.
        
        Args:
            endpoint_pattern: Pattern to match endpoint caches
            user_specific: Whether to invalidate user-specific caches only
            cascade: Whether to cascade invalidation to related endpoints
            
        Returns:
            Number of cache entries invalidated
        """
        try:
            from prompt_improver.utils.redis_cache import invalidate

            patterns = [f"api_responses:{endpoint_pattern}:*"]
            
            if cascade:
                # Add related endpoint patterns
                if "dashboard" in endpoint_pattern:
                    patterns.extend([
                        "api_responses:*dashboard*:*",
                        "api_responses:*metrics*:*",
                    ])
                elif "analytics" in endpoint_pattern:
                    patterns.extend([
                        "api_responses:*analytics*:*",
                        "api_responses:*performance*:*",
                        "api_responses:*trends*:*",
                    ])

            total_invalidated = 0
            for pattern in patterns:
                count = await invalidate(pattern)
                total_invalidated += count

            logger.info(f"Invalidated {total_invalidated} API cache entries for pattern {endpoint_pattern}")
            return total_invalidated

        except Exception as e:
            logger.error(f"Failed to invalidate API cache for {endpoint_pattern}: {e}")
            return 0

    async def warm_endpoint_cache(
        self,
        endpoint_configs: List[Dict[str, Any]],
    ) -> Dict[str, bool]:
        """Warm cache with common endpoint requests.
        
        Args:
            endpoint_configs: List of endpoint configurations with request data
            
        Returns:
            Dictionary mapping endpoints to success status
        """
        results = {}
        
        for config in endpoint_configs:
            endpoint_name = config.get("endpoint_name")
            if not endpoint_name:
                continue
                
            try:
                # Create mock request object
                mock_request = self._create_mock_request(config.get("request_data", {}))
                
                # Generate response
                response_func = config.get("response_func")
                if response_func:
                    response = await self._execute_response_func(
                        response_func, mock_request, **config.get("response_kwargs", {})
                    )
                    
                    # Cache the response
                    cache_key = await self._build_api_cache_key(
                        endpoint_name, mock_request, config.get("vary_headers", [])
                    )
                    
                    await self.cache_facade.set_cached_value(
                        cache_key=cache_key,
                        value=self._serialize_response(response),
                        strategy=CacheStrategy.BALANCED,
                    )
                    
                    results[endpoint_name] = True
                else:
                    results[endpoint_name] = False
                    
            except Exception as e:
                logger.warning(f"Failed to warm cache for endpoint {endpoint_name}: {e}")
                results[endpoint_name] = False
        
        return results

    async def _build_api_cache_key(
        self,
        endpoint_name: str,
        request: Request,
        vary_headers: List[str],
    ) -> CacheKey:
        """Build cache key for API endpoint."""
        # Get request parameters
        path_params = dict(request.path_params) if hasattr(request, 'path_params') else {}
        query_params = dict(request.query_params) if hasattr(request, 'query_params') else {}
        
        # Get relevant headers
        headers = {}
        for header_name in vary_headers:
            header_value = request.headers.get(header_name)
            if header_value:
                headers[header_name.lower()] = header_value[:100]  # Limit header value length

        # Get user identifier for user-specific caching
        user_id = self._extract_user_id(request)

        parameters = {
            "endpoint": endpoint_name,
            "method": getattr(request, 'method', 'GET'),
            "path_params": path_params,
            "query_params": query_params,
            "headers": headers,
            "user_id": user_id,
        }

        return CacheKey(
            namespace="api_responses",
            operation=endpoint_name,
            parameters=parameters,
        )

    async def _execute_and_serialize_response(
        self,
        response_func: Callable,
        request: Request,
        **response_kwargs,
    ) -> Dict[str, Any]:
        """Execute response function and serialize the result."""
        response = await self._execute_response_func(response_func, request, **response_kwargs)
        return self._serialize_response(response)

    async def _execute_response_func(
        self,
        response_func: Callable,
        request: Request,
        **response_kwargs,
    ) -> Any:
        """Execute response function with proper async handling."""
        import asyncio
        
        if asyncio.iscoroutinefunction(response_func):
            return await response_func(request, **response_kwargs)
        else:
            return response_func(request, **response_kwargs)

    def _serialize_response(self, response: Any) -> Dict[str, Any]:
        """Serialize response for caching."""
        if isinstance(response, Response):
            return {
                "_type": "Response",
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.body.decode() if response.body else None,
                "media_type": getattr(response, 'media_type', None),
            }
        elif isinstance(response, JSONResponse):
            return {
                "_type": "JSONResponse",
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.body.decode() if response.body else None,
                "media_type": response.media_type,
            }
        elif isinstance(response, dict):
            return {
                "_type": "dict",
                "data": response,
            }
        else:
            return {
                "_type": "other",
                "data": str(response),
            }

    def _deserialize_response(self, cached_data: Dict[str, Any]) -> Any:
        """Deserialize cached response."""
        response_type = cached_data.get("_type", "dict")
        
        if response_type == "JSONResponse":
            return JSONResponse(
                content=json.loads(cached_data.get("body", "{}")),
                status_code=cached_data.get("status_code", 200),
                headers=cached_data.get("headers", {}),
                media_type=cached_data.get("media_type"),
            )
        elif response_type == "Response":
            response = Response(
                content=cached_data.get("body", ""),
                status_code=cached_data.get("status_code", 200),
                headers=cached_data.get("headers", {}),
                media_type=cached_data.get("media_type"),
            )
            return response
        elif response_type == "dict":
            return cached_data.get("data", {})
        else:
            return cached_data.get("data", "")

    def _should_cache_dashboard_request(self, request: Request) -> bool:
        """Determine if dashboard request should be cached."""
        # Cache GET requests only
        method = getattr(request, 'method', 'GET')
        if method != 'GET':
            return False
            
        # Don't cache if user requests real-time data
        query_params = dict(request.query_params) if hasattr(request, 'query_params') else {}
        if query_params.get('real_time') == 'true':
            return False
            
        return True

    def _should_cache_analytics_request(self, request: Request) -> bool:
        """Determine if analytics request should be cached."""
        method = getattr(request, 'method', 'GET')
        if method != 'GET':
            return False
            
        query_params = dict(request.query_params) if hasattr(request, 'query_params') else {}
        
        # Don't cache if requesting very recent data
        time_range_hours = query_params.get('time_range_hours')
        if time_range_hours and int(time_range_hours) < 24:
            return False
            
        return True

    def _should_cache_timeseries_request(self, request: Request) -> bool:
        """Determine if time-series request should be cached."""
        method = getattr(request, 'method', 'GET')
        if method != 'GET':
            return False
            
        query_params = dict(request.query_params) if hasattr(request, 'query_params') else {}
        
        # Always cache time-series data older than 1 hour
        start_date = query_params.get('start_date')
        if start_date:
            try:
                from datetime import datetime
                parsed_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                time_diff = datetime.now(UTC) - parsed_date
                return time_diff.total_seconds() > 3600  # 1 hour
            except:
                pass
                
        return True

    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request for user-specific caching."""
        # Try to get user ID from headers
        auth_header = request.headers.get('Authorization', '')
        if auth_header:
            # Simple extraction - in real implementation, decode JWT token
            return auth_header[:50]  # Limit length
            
        # Try session or cookie
        user_id = request.headers.get('X-User-ID')
        if user_id:
            return user_id[:50]
            
        return None

    def _create_mock_request(self, request_data: Dict[str, Any]) -> Request:
        """Create mock request for cache warming."""
        # This is a simplified mock - in real implementation,
        # you'd create a proper Request object
        class MockRequest:
            def __init__(self, data):
                self.method = data.get('method', 'GET')
                self.path_params = data.get('path_params', {})
                self.query_params = data.get('query_params', {})
                self.headers = data.get('headers', {})
        
        return MockRequest(request_data)

    def _update_api_stats(self, execution_time_ms: float, cache_hit: bool):
        """Update API cache performance statistics."""
        if cache_hit:
            self._api_stats["cache_hits"] += 1
            # Estimate time saved (assume uncached response would take 100ms)
            self._api_stats["total_time_saved_ms"] += max(0, 100 - execution_time_ms)
        else:
            self._api_stats["cache_misses"] += 1

        # Update average response time
        total_requests = self._api_stats["requests_served"]
        if total_requests > 0:
            current_avg = self._api_stats["avg_response_time_ms"]
            self._api_stats["avg_response_time_ms"] = (
                (current_avg * (total_requests - 1) + execution_time_ms) / total_requests
            )

    async def get_api_cache_performance(self) -> Dict[str, Any]:
        """Get API cache performance statistics."""
        total_requests = self._api_stats["requests_served"]
        cache_hit_rate = (
            self._api_stats["cache_hits"] / total_requests
            if total_requests > 0 else 0
        )
        
        cache_facade_stats = await self.cache_facade.get_performance_stats()
        
        return {
            "api_cache_stats": {
                **self._api_stats,
                "cache_hit_rate": cache_hit_rate,
                "target_hit_rate": 0.8,  # Lower target for API responses
                "meets_hit_rate_target": cache_hit_rate >= 0.8,
                "target_response_time_ms": 200,
                "meets_response_time_target": self._api_stats["avg_response_time_ms"] <= 200,
            },
            "cache_facade_stats": cache_facade_stats,
            "performance_improvements": {
                "estimated_response_time_improvement": f"{cache_hit_rate * 100:.1f}%",
                "total_time_saved_minutes": self._api_stats["total_time_saved_ms"] / 60000,
                "server_load_reduction": f"{cache_hit_rate * 100:.1f}%",
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on API cache."""
        try:
            # Test API response caching
            mock_request = self._create_mock_request({
                "method": "GET",
                "query_params": {"test": "health_check"},
            })
            
            start_time = time.perf_counter()
            
            result = await self.cached_endpoint_response(
                endpoint_name="health_check",
                request=mock_request,
                response_func=lambda req: {"status": "ok", "timestamp": datetime.now(UTC).isoformat()},
                ttl_seconds=60,
            )
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return {
                "healthy": True,
                "api_cache_performance": {
                    "test_response_time_ms": execution_time,
                    "meets_fast_target": execution_time < 50,
                },
                "performance_stats": await self.get_api_cache_performance(),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            
        except Exception as e:
            logger.error(f"API cache health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }


# Global API response cache instance
_global_api_cache: Optional[APIResponseCache] = None


def get_api_response_cache() -> APIResponseCache:
    """Get the global API response cache instance."""
    global _global_api_cache
    if _global_api_cache is None:
        _global_api_cache = APIResponseCache()
    return _global_api_cache


# Convenience decorators for FastAPI endpoints

def cached_api_endpoint(
    ttl_seconds: int = 300,
    cache_strategy: CacheStrategy = CacheStrategy.BALANCED,
    cache_condition: Optional[Callable] = None,
    vary_headers: Optional[List[str]] = None,
):
    """Decorator for caching FastAPI endpoint responses.
    
    Args:
        ttl_seconds: Cache TTL in seconds
        cache_strategy: Caching strategy
        cache_condition: Function to determine if response should be cached
        vary_headers: Headers to include in cache key
    """
    def decorator(endpoint_func: Callable) -> Callable:
        import functools
        
        @functools.wraps(endpoint_func)
        async def wrapper(request: Request, *args, **kwargs):
            api_cache = get_api_response_cache()
            
            return await api_cache.cached_endpoint_response(
                endpoint_name=endpoint_func.__name__,
                request=request,
                response_func=endpoint_func,
                ttl_seconds=ttl_seconds,
                cache_strategy=cache_strategy,
                cache_condition=cache_condition,
                vary_headers=vary_headers,
                *args,
                **kwargs,
            )
        
        return wrapper
    
    return decorator


def cached_dashboard_endpoint():
    """Decorator for caching dashboard endpoints with ultra-fast strategy."""
    return cached_api_endpoint(
        ttl_seconds=60,
        cache_strategy=CacheStrategy.ULTRA_FAST,
    )


def cached_analytics_endpoint():
    """Decorator for caching analytics endpoints with balanced strategy."""
    return cached_api_endpoint(
        ttl_seconds=600,
        cache_strategy=CacheStrategy.BALANCED,
        vary_headers=["Authorization"],
    )


def cached_time_series_endpoint():
    """Decorator for caching time-series endpoints with long-term strategy."""
    return cached_api_endpoint(
        ttl_seconds=1800,
        cache_strategy=CacheStrategy.LONG_TERM,
    )