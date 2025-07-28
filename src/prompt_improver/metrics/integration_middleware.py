"""
Business Metrics Integration Middleware for Prompt Improver.

Provides decorators, middleware, and instrumentation to automatically collect
business metrics from actual application operations without requiring manual
metric recording throughout the codebase.
"""

import asyncio
import functools
import logging
import time
import uuid
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import inspect

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware

from .ml_metrics import (
    get_ml_metrics_collector, 
    record_prompt_improvement,
    record_model_inference,
    PromptCategory,
    ModelInferenceStage
)
from .api_metrics import (
    get_api_metrics_collector,
    record_api_request,
    record_user_journey_event,
    EndpointCategory,
    HTTPMethod,
    UserJourneyStage,
    AuthenticationMethod
)
from .performance_metrics import (
    get_performance_metrics_collector,
    record_pipeline_stage_timing,
    PipelineStage,
    DatabaseOperation,
    CacheType,
    ExternalAPIType
)
from .business_intelligence_metrics import (
    get_bi_metrics_collector,
    record_feature_usage,
    record_operational_cost,
    FeatureCategory,
    UserTier,
    CostType,
    ResourceType
)

logger = logging.getLogger(__name__)

class BusinessMetricsMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that automatically collects API usage metrics
    and user journey data from all HTTP requests.
    """
    
    def __init__(self, app, config: Optional[Dict[str, Any]] = None):
        super().__init__(app)
        self.config = config or {}
        self.api_collector = get_api_metrics_collector()
        self.performance_collector = get_performance_metrics_collector()
        
        # Endpoint categorization mapping
        self.endpoint_categories = {
            "/api/v1/prompt": EndpointCategory.PROMPT_IMPROVEMENT,
            "/api/v1/ml": EndpointCategory.ML_ANALYTICS,
            "/api/v1/users": EndpointCategory.USER_MANAGEMENT,
            "/health": EndpointCategory.HEALTH_CHECK,
            "/api/v1/auth": EndpointCategory.AUTHENTICATION,
            "/api/v1/realtime": EndpointCategory.REAL_TIME,
            "/api/v1/batch": EndpointCategory.BATCH_PROCESSING,
            "/api/v1/config": EndpointCategory.CONFIGURATION,
        }
        
    async def dispatch(self, request: Request, call_next):
        """Process HTTP request and collect metrics."""
        start_time = datetime.now(timezone.utc)
        request_id = str(uuid.uuid4())
        
        # Extract request metadata
        endpoint = str(request.url.path)
        method = HTTPMethod(request.method)
        user_id = self._extract_user_id(request)
        session_id = self._extract_session_id(request)
        ip_address = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent")
        
        # Categorize endpoint
        category = self._categorize_endpoint(endpoint)
        
        # Determine authentication method
        auth_method = self._determine_auth_method(request)
        
        # Track request start
        await record_pipeline_stage_timing(
            request_id=request_id,
            stage=PipelineStage.INGRESS,
            start_time=start_time,
            end_time=start_time,  # Will be updated
            endpoint=endpoint,
            method=request.method,
            user_id=user_id,
            session_id=session_id
        )
        
        # Process request
        try:
            response = await call_next(request)
            success = True
            error_type = None
        except Exception as e:
            # Create error response
            response = Response(
                content=f"Internal server error: {str(e)}",
                status_code=500,
                media_type="text/plain"
            )
            success = False
            error_type = type(e).__name__
            logger.error(f"Request {request_id} failed: {e}")
        
        end_time = datetime.now(timezone.utc)
        response_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Extract response metadata
        status_code = response.status_code
        request_size = self._get_request_size(request)
        response_size = self._get_response_size(response)
        
        # Check if request was rate limited
        rate_limited = status_code == 429
        
        # Check cache hit (from headers or response metadata)
        cache_hit = response.headers.get("X-Cache-Hit", "false").lower() == "true"
        
        # Record API usage metric
        await record_api_request(
            endpoint=endpoint,
            method=method,
            category=category,
            status_code=status_code,
            response_time_ms=response_time_ms,
            request_size_bytes=request_size,
            response_size_bytes=response_size,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            rate_limited=rate_limited,
            cache_hit=cache_hit,
            authentication_method=auth_method
        )
        
        # Record egress pipeline stage
        await record_pipeline_stage_timing(
            request_id=request_id,
            stage=PipelineStage.EGRESS,
            start_time=end_time,
            end_time=end_time,
            success=success,
            error_type=error_type,
            endpoint=endpoint,
            method=request.method,
            user_id=user_id,
            session_id=session_id
        )
        
        # Track user journey if applicable
        if user_id and category != EndpointCategory.HEALTH_CHECK:
            journey_stage = self._determine_journey_stage(endpoint, user_id)
            await record_user_journey_event(
                user_id=user_id,
                session_id=session_id or f"session_{user_id}_{int(time.time())}",
                journey_stage=journey_stage,
                event_type="api_request",
                endpoint=endpoint,
                success=success,
                time_to_action_seconds=response_time_ms / 1000.0
            )
        
        return response
    
    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request."""
        # Check JWT token
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            # In a real implementation, decode JWT to get user_id
            return "user_from_jwt"
        
        # Check session cookie
        session_cookie = request.cookies.get("session_id")
        if session_cookie:
            # In a real implementation, lookup user from session
            return f"user_from_session_{session_cookie[:8]}"
        
        # Check API key header
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"user_from_api_key_{api_key[:8]}"
        
        return None
    
    def _extract_session_id(self, request: Request) -> Optional[str]:
        """Extract session ID from request."""
        # Check custom session header
        session_id = request.headers.get("x-session-id")
        if session_id:
            return session_id
        
        # Check session cookie
        session_cookie = request.cookies.get("session_id")
        if session_cookie:
            return session_cookie
        
        # Generate session ID from IP and user agent
        ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        return f"session_{hash(f'{ip}_{user_agent}')}"[:16]
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to client host
        return request.client.host if request.client else "unknown"
    
    def _categorize_endpoint(self, endpoint: str) -> EndpointCategory:
        """Categorize endpoint based on path."""
        for path_prefix, category in self.endpoint_categories.items():
            if endpoint.startswith(path_prefix):
                return category
        
        # Default categorization based on path patterns
        if "/api/" in endpoint:
            return EndpointCategory.PROMPT_IMPROVEMENT  # Default API category
        
        return EndpointCategory.CONFIGURATION  # Default for non-API endpoints
    
    def _determine_auth_method(self, request: Request) -> AuthenticationMethod:
        """Determine authentication method used."""
        auth_header = request.headers.get("authorization", "")
        
        if auth_header.startswith("Bearer "):
            return AuthenticationMethod.JWT_TOKEN
        elif auth_header.startswith("Basic "):
            return AuthenticationMethod.BASIC_AUTH
        elif request.headers.get("x-api-key"):
            return AuthenticationMethod.API_KEY
        elif request.cookies.get("session_id"):
            return AuthenticationMethod.SESSION_COOKIE
        
        return AuthenticationMethod.ANONYMOUS
    
    def _determine_journey_stage(self, endpoint: str, user_id: str) -> UserJourneyStage:
        """Determine user journey stage based on endpoint and user context."""
        # In a real implementation, this would check user history and behavior
        # For now, use simple heuristics based on endpoint
        
        if "onboard" in endpoint.lower():
            return UserJourneyStage.ONBOARDING
        elif "tutorial" in endpoint.lower() or "help" in endpoint.lower():
            return UserJourneyStage.FIRST_USE
        elif "advanced" in endpoint.lower() or "batch" in endpoint.lower():
            return UserJourneyStage.ADVANCED_USE
        
        # Default to regular use
        return UserJourneyStage.REGULAR_USE
    
    def _get_request_size(self, request: Request) -> int:
        """Get request size in bytes."""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                return int(content_length)
            except ValueError:
                pass
        
        # Estimate based on headers and query parameters
        header_size = sum(len(k) + len(v) for k, v in request.headers.items())
        query_size = len(str(request.query_params))
        return header_size + query_size
    
    def _get_response_size(self, response: Response) -> int:
        """Get response size in bytes."""
        if hasattr(response, 'body') and response.body:
            return len(response.body)
        
        # Estimate based on headers
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                return int(content_length)
            except ValueError:
                pass
        
        return 0

def track_ml_operation(
    category: PromptCategory,
    stage: Optional[ModelInferenceStage] = None,
    model_name: Optional[str] = None
):
    """
    Decorator to automatically track ML operations and record metrics.
    
    Args:
        category: The prompt improvement category
        stage: Model inference stage (if applicable)
        model_name: Name of the model being used
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Extract parameters
            original_prompt = kwargs.get('prompt', '') or (args[0] if args else '')
            user_id = kwargs.get('user_id')
            session_id = kwargs.get('session_id')
            
            try:
                result = await func(*args, **kwargs)
                success = True
                error_type = None
            except Exception as e:
                result = None
                success = False
                error_type = type(e).__name__
                raise
            finally:
                end_time = time.time()
                processing_time_ms = (end_time - start_time) * 1000
                
                # Extract result metadata
                improved_prompt = ''
                confidence_score = 0.8  # Default confidence
                improvement_ratio = 1.0
                
                if result and isinstance(result, dict):
                    improved_prompt = result.get('improved_prompt', '')
                    confidence_score = result.get('confidence', 0.8)
                    if improved_prompt and original_prompt:
                        improvement_ratio = len(improved_prompt) / len(original_prompt)
                
                # Record prompt improvement metric
                await record_prompt_improvement(
                    category=category,
                    original_length=len(original_prompt),
                    improved_length=len(improved_prompt),
                    improvement_ratio=improvement_ratio,
                    success=success,
                    processing_time_ms=processing_time_ms,
                    confidence_score=confidence_score,
                    user_id=user_id,
                    session_id=session_id
                )
                
                # Record model inference if stage is specified
                if stage and model_name:
                    input_tokens = len(original_prompt.split()) if original_prompt else 0
                    output_tokens = len(improved_prompt.split()) if improved_prompt else 0
                    
                    await record_model_inference(
                        model_name=model_name,
                        inference_stage=stage,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        latency_ms=processing_time_ms,
                        memory_usage_mb=0,  # Would need actual memory tracking
                        success=success,
                        error_type=error_type,
                        confidence_distribution=[confidence_score]
                    )
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, create async wrapper
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def track_feature_usage(
    feature_name: str,
    feature_category: FeatureCategory,
    user_tier: UserTier = UserTier.FREE
):
    """
    Decorator to automatically track feature usage and business metrics.
    
    Args:
        feature_name: Name of the feature being used
        feature_category: Category of the feature
        user_tier: User's subscription tier
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Extract user context
            user_id = kwargs.get('user_id', 'anonymous')
            session_id = kwargs.get('session_id', f'session_{user_id}_{int(time.time())}')
            
            try:
                result = await func(*args, **kwargs)
                success = True
                error_type = None
            except Exception as e:
                result = None
                success = False
                error_type = type(e).__name__
                raise
            finally:
                end_time = time.time()
                time_spent = end_time - start_time
                
                # Record feature usage
                await record_feature_usage(
                    feature_name=feature_name,
                    feature_category=feature_category,
                    user_id=user_id,
                    user_tier=user_tier,
                    session_id=session_id,
                    time_spent_seconds=time_spent,
                    success=success,
                    error_type=error_type
                )
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

@asynccontextmanager
async def track_pipeline_stage(
    request_id: str,
    stage: PipelineStage,
    endpoint: str = "unknown",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """
    Context manager to track pipeline stage execution time and success.
    
    Usage:
        async with track_pipeline_stage(request_id, PipelineStage.BUSINESS_LOGIC):
            # Your business logic here
            result = await process_business_logic()
    """
    start_time = datetime.now(timezone.utc)
    success = True
    error_type = None
    
    try:
        yield
    except Exception as e:
        success = False
        error_type = type(e).__name__
        raise
    finally:
        end_time = datetime.now(timezone.utc)
        
        await record_pipeline_stage_timing(
            request_id=request_id,
            stage=stage,
            start_time=start_time,
            end_time=end_time,
            success=success,
            error_type=error_type,
            endpoint=endpoint,
            user_id=user_id,
            session_id=session_id
        )

def track_cost_operation(
    operation_type: str,
    cost_type: CostType,
    estimated_cost_per_unit: float = 0.001,  # Default cost estimation
    currency: str = "USD"
):
    """
    Decorator to track operational costs for business intelligence.
    
    Args:
        operation_type: Type of operation being performed
        cost_type: Category of cost (compute, storage, etc.)
        estimated_cost_per_unit: Estimated cost per operation
        currency: Currency for cost calculation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Calculate resource usage and cost
                end_time = time.time()
                duration_seconds = end_time - start_time
                
                # Estimate cost based on duration and type
                estimated_cost = duration_seconds * estimated_cost_per_unit
                
                # Record cost metric
                await record_operational_cost(
                    operation_type=operation_type,
                    cost_type=cost_type,
                    cost_amount=estimated_cost,
                    currency=currency,
                    resource_units_consumed=duration_seconds,
                    resource_unit_cost=estimated_cost_per_unit,
                    user_id=kwargs.get('user_id'),
                    user_tier=kwargs.get('user_tier')
                )
                
                return result
                
            except Exception as e:
                # Still record cost for failed operations
                end_time = time.time()
                duration_seconds = end_time - start_time
                estimated_cost = duration_seconds * estimated_cost_per_unit
                
                await record_operational_cost(
                    operation_type=f"{operation_type}_failed",
                    cost_type=cost_type,
                    cost_amount=estimated_cost,
                    currency=currency,
                    resource_units_consumed=duration_seconds,
                    resource_unit_cost=estimated_cost_per_unit,
                    user_id=kwargs.get('user_id'),
                    user_tier=kwargs.get('user_tier')
                )
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

class DatabaseMetricsInstrumentation:
    """
    Instrumentation class for automatically tracking database operations.
    Can be integrated with SQLAlchemy, asyncpg, or other database libraries.
    """
    
    def __init__(self):
        self.performance_collector = get_performance_metrics_collector()
    
    async def track_query(
        self,
        query: str,
        operation_type: DatabaseOperation,
        table_name: str = "unknown",
        execution_time_ms: float = 0,
        rows_affected: int = 0,
        success: bool = True,
        error_type: Optional[str] = None
    ):
        """Track database query execution."""
        from .performance_metrics import DatabasePerformanceMetric
        
        metric = DatabasePerformanceMetric(
            operation_type=operation_type,
            table_name=table_name,
            query_hash=str(hash(query))[:16],
            execution_time_ms=execution_time_ms,
            rows_affected=rows_affected,
            rows_examined=rows_affected,  # Simplification
            index_usage=[],
            query_plan_type="unknown",
            connection_pool_size=10,  # Would get from actual pool
            active_connections=5,  # Would get from actual pool
            wait_time_ms=0,
            lock_time_ms=0,
            temp_tables_created=0,
            bytes_sent=len(query),
            bytes_received=rows_affected * 100,  # Estimation
            success=success,
            error_type=error_type,
            timestamp=datetime.now(timezone.utc),
            transaction_id=None
        )
        
        await self.performance_collector.record_database_operation(metric)

class CacheMetricsInstrumentation:
    """
    Instrumentation class for automatically tracking cache operations.
    Can be integrated with Redis, Memcached, or other caching systems.
    """
    
    def __init__(self):
        self.performance_collector = get_performance_metrics_collector()
    
    async def track_cache_operation(
        self,
        cache_type: CacheType,
        operation: str,
        key: str,
        hit: bool,
        response_time_ms: float,
        cache_size_bytes: Optional[int] = None,
        success: bool = True
    ):
        """Track cache operation."""
        from .performance_metrics import CachePerformanceMetric
        
        metric = CachePerformanceMetric(
            cache_type=cache_type,
            operation=operation,
            key_pattern=key[:50],  # Truncate for privacy
            hit=hit,
            response_time_ms=response_time_ms,
            cache_size_bytes=cache_size_bytes,
            eviction_triggered=False,
            ttl_remaining_seconds=None,
            serialization_time_ms=None,
            network_time_ms=None,
            compression_used=False,
            compression_ratio=None,
            timestamp=datetime.now(timezone.utc),
            user_id=None,
            session_id=None
        )
        
        await self.performance_collector.record_cache_operation(metric)

# Global instances for easy access
db_metrics = DatabaseMetricsInstrumentation()
cache_metrics = CacheMetricsInstrumentation()

# Convenience functions for manual metric recording
async def track_user_action(
    user_id: str,
    action_type: str,
    feature_category: FeatureCategory,
    session_id: Optional[str] = None,
    success: bool = True,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Manually track a user action for business intelligence.
    
    Args:
        user_id: ID of the user performing the action
        action_type: Type of action being performed
        feature_category: Category of feature being used
        session_id: User's session ID
        success: Whether the action was successful
        metadata: Additional metadata about the action
    """
    await record_feature_usage(
        feature_name=action_type,
        feature_category=feature_category,
        user_id=user_id,
        user_tier=UserTier.FREE,  # Would determine actual tier
        session_id=session_id or f"session_{user_id}_{int(time.time())}",
        success=success,
        metadata=metadata or {}
    )

async def initialize_metrics_collection():
    """Initialize all metrics collectors and start background aggregation."""
    try:
        # Initialize all collectors
        ml_collector = get_ml_metrics_collector()
        api_collector = get_api_metrics_collector()
        performance_collector = get_performance_metrics_collector()
        bi_collector = get_bi_metrics_collector()
        
        # Start background collection
        await ml_collector.start_aggregation()
        await api_collector.start_collection()
        await performance_collector.start_collection()
        await bi_collector.start_collection()
        
        logger.info("Business metrics collection initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize metrics collection: {e}")
        raise

async def shutdown_metrics_collection():
    """Shutdown all metrics collectors."""
    try:
        # Get all collectors
        ml_collector = get_ml_metrics_collector()
        api_collector = get_api_metrics_collector()
        performance_collector = get_performance_metrics_collector()
        bi_collector = get_bi_metrics_collector()
        
        # Stop background collection
        await ml_collector.stop_aggregation()
        await api_collector.stop_collection()
        await performance_collector.stop_collection()
        await bi_collector.stop_collection()
        
        logger.info("Business metrics collection shutdown successfully")
        
    except Exception as e:
        logger.error(f"Failed to shutdown metrics collection: {e}")