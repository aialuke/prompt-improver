"""RateLimitingComponent - Request Rate Limiting Service

Complete implementation extracted from UnifiedRateLimiter. This component handles 
sliding window rate limiting, burst control, and traffic throttling with Redis-based 
persistence using secure atomic Lua operations.

Features:
- Sliding window rate limiting with configurable bucket sizes
- Multi-tier rate limiting (basic, professional, enterprise)
- Burst capacity handling with separate limits
- Atomic Redis operations preventing race conditions
- Fail-secure policy (deny on error)
- Comprehensive metrics and monitoring
- Authentication-based access control
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from prompt_improver.database import (
    ManagerMode,
    RedisSecurityError,
    SecurityContext,
    SecurityPerformanceMetrics,
    create_security_context,
    get_database_services,
)
from prompt_improver.security.unified.protocols import (
    RateLimitingProtocol,
    SecurityComponentStatus,
    SecurityOperationResult,
)
from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)


class RateLimitResult(str, Enum):
    """Rate limit check results."""
    ALLOWED = "allowed"
    RATE_LIMITED = "rate_limited"
    BURST_LIMITED = "burst_limited"
    ERROR = "error"
    AUTHENTICATION_REQUIRED = "authentication_required"


class RateLimitTier(str, Enum):
    """Rate limiting tiers with different request allowances."""
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class RateLimitStatus(BaseModel):
    """Rate limit status information."""
    result: RateLimitResult = Field(description="Rate limit check result")
    requests_remaining: int = Field(ge=0, description="Number of requests remaining in current window")
    burst_remaining: int = Field(ge=0, description="Number of burst requests remaining")
    reset_time: float = Field(gt=0, description="Timestamp when rate limit resets")
    retry_after: int | None = Field(default=None, ge=0, description="Seconds to wait before retry")
    current_requests: int = Field(ge=0, description="Current request count in window")
    window_start: float = Field(gt=0, description="Timestamp of current window start")
    agent_id: str = Field(min_length=1, max_length=100, description="Authenticated agent identifier")
    tier: str = Field(min_length=1, max_length=50, description="Rate limiting tier")


class RateLimitingComponent:
    """Complete rate limiting component implementing RateLimitingProtocol."""
    
    def __init__(self):
        self._initialized = False
        self._connection_manager = None
        self._metrics = {
            "rate_limit_checks": 0,
            "rate_limit_updates": 0,
            "rate_limits_exceeded": 0,
            "rate_limit_resets": 0,
            "total_rate_limit_time_ms": 0.0,
        }
        
        # Rate limiting configuration by tier
        self._tier_configs = {
            RateLimitTier.BASIC.value: {
                "rate_limit_per_minute": 60,
                "burst_capacity": 90,
                "window_size_seconds": 60,
                "bucket_size_seconds": 10,
            },
            RateLimitTier.PROFESSIONAL.value: {
                "rate_limit_per_minute": 300,
                "burst_capacity": 450,
                "window_size_seconds": 60,
                "bucket_size_seconds": 10,
            },
            RateLimitTier.ENTERPRISE.value: {
                "rate_limit_per_minute": 1000,
                "burst_capacity": 1500,
                "window_size_seconds": 60,
                "bucket_size_seconds": 5,
            },
        }
        
        # Atomic Lua script for sliding window rate limiting
        self._sliding_window_script = """
        local key = KEYS[1]
        local window_size = tonumber(ARGV[1])
        local bucket_size = tonumber(ARGV[2])
        local rate_limit = tonumber(ARGV[3])
        local burst_capacity = tonumber(ARGV[4])
        local current_time = tonumber(ARGV[5])
        
        -- Calculate window boundaries
        local window_start = current_time - window_size
        local bucket_start = math.floor(current_time / bucket_size) * bucket_size
        
        -- Remove old buckets outside window
        redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)
        
        -- Count current requests in window
        local current_requests = redis.call('ZCOUNT', key, window_start, current_time)
        
        -- Check rate limits
        if current_requests >= burst_capacity then
            -- Burst limit exceeded
            local ttl = redis.call('TTL', key)
            return {3, current_requests, 0, window_start, ttl > 0 and ttl or window_size}
        elseif current_requests >= rate_limit then
            -- Rate limit exceeded  
            local ttl = redis.call('TTL', key)
            return {2, current_requests, burst_capacity - current_requests, window_start, ttl > 0 and ttl or window_size}
        else
            -- Request allowed - add to current bucket
            redis.call('ZADD', key, current_time, current_time .. ':' .. math.random())
            redis.call('EXPIRE', key, window_size * 2)  -- Safety margin for cleanup
            
            local remaining = rate_limit - current_requests - 1
            local burst_remaining = burst_capacity - current_requests - 1
            return {1, remaining, burst_remaining, window_start, 0}
        end
        """
    
    async def initialize(self) -> bool:
        """Initialize the rate limiting component with database services."""
        try:
            self._connection_manager = await get_database_services(ManagerMode.ASYNC_MODERN)
            self._initialized = True
            logger.info("RateLimitingComponent initialized with enhanced DatabaseServices")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize rate limiting component: {e}")
            return False
    
    async def health_check(self) -> Tuple[SecurityComponentStatus, Dict[str, Any]]:
        """Check component health status."""
        if not self._initialized:
            return SecurityComponentStatus.UNHEALTHY, {
                "initialized": False,
                "error": "Component not initialized"
            }
        
        try:
            # Test Redis connectivity
            if not self._connection_manager or not self._connection_manager.cache.redis_client:
                return SecurityComponentStatus.UNHEALTHY, {
                    "initialized": self._initialized,
                    "error": "Redis connection not available"
                }
            
            # Test basic Redis operation
            await self._connection_manager.cache.redis_client.ping()
            
            return SecurityComponentStatus.HEALTHY, {
                "initialized": self._initialized,
                "metrics": self._metrics.copy(),
                "redis_connected": True,
                "tier_configs": list(self._tier_configs.keys())
            }
        except Exception as e:
            return SecurityComponentStatus.UNHEALTHY, {
                "initialized": self._initialized,
                "error": f"Health check failed: {e}"
            }
    
    async def get_metrics(self) -> SecurityPerformanceMetrics:
        """Get security performance metrics."""
        total_ops = sum([
            self._metrics["rate_limit_checks"],
            self._metrics["rate_limit_updates"],
            self._metrics["rate_limit_resets"]
        ])
        avg_latency = (
            self._metrics["total_rate_limit_time_ms"] / total_ops 
            if total_ops > 0 else 0.0
        )
        
        return SecurityPerformanceMetrics(
            operation_count=total_ops,
            average_latency_ms=avg_latency,
            error_rate=0.0,
            threat_detection_count=self._metrics["rate_limits_exceeded"],
            last_updated=aware_utc_now()
        )
    
    async def check_rate_limit(
        self,
        security_context: SecurityContext,
        operation_type: str = "default"
    ) -> SecurityOperationResult:
        """Check if request is within rate limits using sliding window algorithm."""
        start_time = time.perf_counter()
        self._metrics["rate_limit_checks"] += 1
        
        try:
            if not self._initialized:
                await self.initialize()
                
            # Fail secure if not authenticated
            if not security_context.authenticated:
                execution_time = (time.perf_counter() - start_time) * 1000
                self._metrics["total_rate_limit_time_ms"] += execution_time
                return SecurityOperationResult(
                    success=False,
                    operation_type="check_rate_limit",
                    execution_time_ms=execution_time,
                    security_context=security_context,
                    metadata={
                        "operation_type": operation_type,
                        "rate_limit_status": "authentication_required",
                        "error": "Authentication required for rate limiting"
                    }
                )
            
            # Get tier configuration
            tier = getattr(security_context, 'tier', 'basic')
            if tier not in self._tier_configs:
                logger.warning(f"Unknown tier {tier}, defaulting to basic")
                tier = "basic"
            
            config = self._tier_configs[tier]
            
            # Check Redis availability
            if not self._connection_manager.cache.redis_client:
                # Fail secure on Redis unavailability
                execution_time = (time.perf_counter() - start_time) * 1000
                self._metrics["total_rate_limit_time_ms"] += execution_time
                self._metrics["rate_limits_exceeded"] += 1
                return SecurityOperationResult(
                    success=False,
                    operation_type="check_rate_limit",
                    execution_time_ms=execution_time,
                    security_context=security_context,
                    metadata={
                        "operation_type": operation_type,
                        "rate_limit_status": "error",
                        "error": "Redis not available - failing secure"
                    }
                )
            
            # Execute sliding window rate limit check
            current_time = time.time()
            key = f"rate_limit:{security_context.agent_id}:{tier}:{operation_type}"
            
            result = await self._connection_manager.cache.redis_client.eval(
                self._sliding_window_script,
                keys=[key],
                args=[
                    config["window_size_seconds"],
                    config["bucket_size_seconds"],
                    config["rate_limit_per_minute"],
                    config["burst_capacity"],
                    current_time,
                ],
            )
            
            status_code, current_requests, remaining, window_start, retry_after = result
            execution_time = (time.perf_counter() - start_time) * 1000
            self._metrics["total_rate_limit_time_ms"] += execution_time
            
            if status_code == 1:  # Allowed
                return SecurityOperationResult(
                    success=True,
                    operation_type="check_rate_limit",
                    execution_time_ms=execution_time,
                    security_context=security_context,
                    metadata={
                        "operation_type": operation_type,
                        "rate_limit_status": "allowed",
                        "requests_remaining": remaining,
                        "burst_remaining": remaining,
                        "reset_time": window_start + config["window_size_seconds"],
                        "current_requests": current_requests,
                        "tier": tier
                    }
                )
            else:  # Rate limited or burst limited
                self._metrics["rate_limits_exceeded"] += 1
                limit_type = "rate_limited" if status_code == 2 else "burst_limited"
                return SecurityOperationResult(
                    success=False,
                    operation_type="check_rate_limit",
                    execution_time_ms=execution_time,
                    security_context=security_context,
                    metadata={
                        "operation_type": operation_type,
                        "rate_limit_status": limit_type,
                        "requests_remaining": 0 if status_code == 3 else 0,
                        "burst_remaining": remaining if status_code == 2 else 0,
                        "reset_time": window_start + config["window_size_seconds"],
                        "retry_after": retry_after,
                        "current_requests": current_requests,
                        "tier": tier
                    }
                )
                
        except RedisSecurityError as e:
            logger.error(f"Redis security error in rate limiting: {e}")
            execution_time = (time.perf_counter() - start_time) * 1000
            self._metrics["total_rate_limit_time_ms"] += execution_time
            self._metrics["rate_limits_exceeded"] += 1
            return SecurityOperationResult(
                success=False,
                operation_type="check_rate_limit",
                execution_time_ms=execution_time,
                security_context=security_context,
                metadata={
                    "operation_type": operation_type,
                    "rate_limit_status": "error",
                    "error": f"Security error: {e}"
                }
            )
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            execution_time = (time.perf_counter() - start_time) * 1000
            self._metrics["total_rate_limit_time_ms"] += execution_time
            self._metrics["rate_limits_exceeded"] += 1
            return SecurityOperationResult(
                success=False,
                operation_type="check_rate_limit",
                execution_time_ms=execution_time,
                security_context=security_context,
                metadata={
                    "operation_type": operation_type,
                    "rate_limit_status": "error",
                    "error": f"Rate limit check failed: {e}"
                }
            )
    
    async def update_rate_limit(
        self,
        security_context: SecurityContext,
        operation_type: str = "default"
    ) -> SecurityOperationResult:
        """Update rate limit counters after successful operation (already handled in check)."""
        start_time = time.perf_counter()
        self._metrics["rate_limit_updates"] += 1
        
        # Note: The sliding window implementation updates counters during check_rate_limit
        # This method exists for protocol compliance but doesn't need separate logic
        execution_time = (time.perf_counter() - start_time) * 1000
        self._metrics["total_rate_limit_time_ms"] += execution_time
        
        return SecurityOperationResult(
            success=True,
            operation_type="update_rate_limit",
            execution_time_ms=execution_time,
            security_context=security_context,
            metadata={
                "operation_type": operation_type,
                "updated": True,
                "note": "Rate limits updated atomically during check operation"
            }
        )
    
    async def get_rate_limit_status(
        self,
        agent_id: str,
        operation_type: str = "default"
    ) -> SecurityOperationResult:
        """Get current rate limit status for agent without consuming quota."""
        start_time = time.perf_counter()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Create minimal security context for status check
            security_context = await create_security_context(
                agent_id=agent_id,
                authenticated=True  # Assume authenticated for status check
            )
            
            tier = getattr(security_context, 'tier', 'basic')
            if tier not in self._tier_configs:
                tier = "basic"
            
            config = self._tier_configs[tier]
            
            if not self._connection_manager.cache.redis_client:
                execution_time = (time.perf_counter() - start_time) * 1000
                return SecurityOperationResult(
                    success=False,
                    operation_type="get_rate_limit_status",
                    execution_time_ms=execution_time,
                    metadata={
                        "agent_id": agent_id,
                        "operation_type": operation_type,
                        "error": "Redis not available"
                    }
                )
            
            # Check current usage without updating
            current_time = time.time()
            key = f"rate_limit:{agent_id}:{tier}:{operation_type}"
            window_start = current_time - config["window_size_seconds"]
            
            # Count current requests in window
            current_requests = await self._connection_manager.cache.redis_client.zcount(
                key, window_start, current_time
            )
            
            requests_remaining = max(0, config["rate_limit_per_minute"] - current_requests)
            burst_remaining = max(0, config["burst_capacity"] - current_requests)
            reset_time = window_start + config["window_size_seconds"]
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return SecurityOperationResult(
                success=True,
                operation_type="get_rate_limit_status",
                execution_time_ms=execution_time,
                metadata={
                    "agent_id": agent_id,
                    "operation_type": operation_type,
                    "tier": tier,
                    "current_requests": current_requests,
                    "requests_remaining": requests_remaining,
                    "burst_remaining": burst_remaining,
                    "rate_limit_per_minute": config["rate_limit_per_minute"],
                    "burst_capacity": config["burst_capacity"],
                    "reset_time": reset_time,
                    "window_size_seconds": config["window_size_seconds"]
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get rate limit status: {e}")
            execution_time = (time.perf_counter() - start_time) * 1000
            return SecurityOperationResult(
                success=False,
                operation_type="get_rate_limit_status",
                execution_time_ms=execution_time,
                metadata={
                    "agent_id": agent_id,
                    "operation_type": operation_type,
                    "error": f"Status check failed: {e}"
                }
            )
    
    async def reset_rate_limit(
        self,
        agent_id: str,
        operation_type: str = "default"
    ) -> SecurityOperationResult:
        """Reset rate limit counters for agent (administrative operation)."""
        start_time = time.perf_counter()
        self._metrics["rate_limit_resets"] += 1
        
        try:
            if not self._initialized:
                await self.initialize()
                
            if not self._connection_manager.cache.redis_client:
                execution_time = (time.perf_counter() - start_time) * 1000
                return SecurityOperationResult(
                    success=False,
                    operation_type="reset_rate_limit",
                    execution_time_ms=execution_time,
                    metadata={
                        "agent_id": agent_id,
                        "operation_type": operation_type,
                        "error": "Redis not available"
                    }
                )
            
            # Reset rate limits for all tiers and operation types for this agent
            keys_deleted = 0
            for tier in self._tier_configs.keys():
                key = f"rate_limit:{agent_id}:{tier}:{operation_type}"
                deleted = await self._connection_manager.cache.redis_client.delete(key)
                keys_deleted += deleted
            
            execution_time = (time.perf_counter() - start_time) * 1000
            self._metrics["total_rate_limit_time_ms"] += execution_time
            
            return SecurityOperationResult(
                success=True,
                operation_type="reset_rate_limit",
                execution_time_ms=execution_time,
                metadata={
                    "agent_id": agent_id,
                    "operation_type": operation_type,
                    "keys_deleted": keys_deleted,
                    "reset": True
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")
            execution_time = (time.perf_counter() - start_time) * 1000
            self._metrics["total_rate_limit_time_ms"] += execution_time
            return SecurityOperationResult(
                success=False,
                operation_type="reset_rate_limit",
                execution_time_ms=execution_time,
                metadata={
                    "agent_id": agent_id,
                    "operation_type": operation_type,
                    "error": f"Reset failed: {e}"
                }
            )
    
    async def cleanup(self) -> bool:
        """Cleanup component resources and reset state."""
        try:
            if self._connection_manager:
                # Connection cleanup is handled by DatabaseServices
                self._connection_manager = None
                
            self._metrics = {
                "rate_limit_checks": 0,
                "rate_limit_updates": 0,
                "rate_limits_exceeded": 0,
                "rate_limit_resets": 0,
                "total_rate_limit_time_ms": 0.0,
            }
            self._initialized = False
            logger.info("RateLimitingComponent cleanup completed")
            return True
        except Exception as e:
            logger.error(f"Error during rate limiting component cleanup: {e}")
            return False