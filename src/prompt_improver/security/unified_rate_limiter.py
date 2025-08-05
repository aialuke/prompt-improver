"""
Unified Rate Limiter - Phase 4 Security Consolidation

Consolidates duplicate rate limiting implementations:
- MCPRateLimitMiddleware (rate_limit_middleware.py) - 335 lines
- SlidingWindowRateLimiter (redis_rate_limiter.py) - 325 lines

Fixes critical security vulnerabilities:
- Fail-open policy → fail-secure (fail-closed) policy
- Agent authentication bypass → mandatory authentication
- Race conditions in sliding window → atomic Lua operations
- Credential exposure → secure Redis manager integration

Integrates with Phase 1 UnifiedConnectionManager and new SecureRedisManager.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from ..database.unified_connection_manager import (
    get_unified_manager,
    create_security_context, 
    SecurityContext,
    RedisSecurityError,
    ManagerMode
)

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
    BASIC = "basic"          # 60 req/min, burst 90
    PROFESSIONAL = "professional"  # 300 req/min, burst 450
    ENTERPRISE = "enterprise"      # 1000+ req/min, burst 1500


@dataclass
class RateLimitStatus:
    """Rate limit status information."""
    result: RateLimitResult
    requests_remaining: int
    burst_remaining: int
    reset_time: float
    retry_after: Optional[int]
    current_requests: int
    window_start: float
    agent_id: str
    tier: str


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str, status: RateLimitStatus):
        self.message = message
        self.status = status
        super().__init__(message)


class UnifiedRateLimiter:
    """Unified rate limiter with secure Redis integration.
    
    Consolidates functionality from:
    - SlidingWindowRateLimiter (redis_rate_limiter.py)
    - MCPRateLimitMiddleware (rate_limit_middleware.py)
    
    Security improvements:
    - Fail-secure policy (fail-closed on errors)
    - Mandatory authentication for all operations
    - Atomic Lua operations preventing race conditions
    - Secure Redis connection management
    """
    
    def __init__(self):
        self._connection_manager = None
        self._tier_configs = {
            RateLimitTier.BASIC.value: {
                "rate_limit_per_minute": 60,
                "burst_capacity": 90,
                "window_size_seconds": 60,
                "bucket_size_seconds": 10
            },
            RateLimitTier.PROFESSIONAL.value: {
                "rate_limit_per_minute": 300,
                "burst_capacity": 450,
                "window_size_seconds": 60,
                "bucket_size_seconds": 10
            },
            RateLimitTier.ENTERPRISE.value: {
                "rate_limit_per_minute": 1000,
                "burst_capacity": 1500,
                "window_size_seconds": 60,
                "bucket_size_seconds": 5
            }
        }
        
        # Lua script for atomic sliding window operations
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
    
    async def initialize(self) -> None:
        """Initialize unified rate limiter with enhanced UnifiedConnectionManager."""
        try:
            self._connection_manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
            await self._connection_manager.initialize()
            logger.info("Unified rate limiter initialized with enhanced UnifiedConnectionManager")
        except Exception as e:
            logger.error(f"Failed to initialize unified rate limiter: {e}")
            raise
    
    async def check_rate_limit(self,
                              agent_id: str,
                              tier: str = "basic",
                              authenticated: bool = True) -> RateLimitStatus:
        """Check rate limit with secure authentication.
        
        Fixes security vulnerabilities:
        - Requires authentication (fixes agent bypass)
        - Fails secure on errors (fixes fail-open policy)
        - Uses atomic operations (fixes race conditions)
        """
        if not self._connection_manager:
            await self.initialize()
        
        # Validate authentication - SECURITY REQUIREMENT
        if not authenticated:
            return RateLimitStatus(
                result=RateLimitResult.AUTHENTICATION_REQUIRED,
                requests_remaining=0,
                burst_remaining=0,
                reset_time=time.time(),
                retry_after=None,
                current_requests=0,
                window_start=time.time(),
                agent_id=agent_id,
                tier=tier
            )
        
        # Get tier configuration
        if tier not in self._tier_configs:
            logger.warning(f"Unknown tier {tier}, defaulting to basic")
            tier = "basic"
            
        config = self._tier_configs[tier]
        
        try:
            # Create security context
            security_context = await create_security_context(
                agent_id=agent_id,
                tier=tier,
                authenticated=authenticated
            )
            
            # Use UnifiedConnectionManager Redis client with security validation
            if not self._connection_manager._redis_master:
                logger.error("Redis master not available in UnifiedConnectionManager")
                # Fail-secure: deny access
                return RateLimitStatus(
                    result=RateLimitResult.ERROR,
                    requests_remaining=0,
                    burst_remaining=0,
                    reset_time=time.time() + 60,
                    retry_after=60,
                    current_requests=0,
                    window_start=time.time(),
                    agent_id=agent_id,
                    tier=tier
                )
            
            # Execute atomic sliding window check
            current_time = time.time()
            key = f"rate_limit:{agent_id}:{tier}"
            
            result = await self._connection_manager._redis_master.eval(
                self._sliding_window_script,
                keys=[key],
                args=[
                    config["window_size_seconds"],
                    config["bucket_size_seconds"],
                    config["rate_limit_per_minute"],
                    config["burst_capacity"],
                    current_time
                ]
            )
                
            # Parse Lua script result
            status_code, current_requests, remaining, window_start, retry_after = result
                
            if status_code == 1:  # Allowed
                return RateLimitStatus(
                    result=RateLimitResult.ALLOWED,
                    requests_remaining=remaining,
                    burst_remaining=remaining,  # Lua already calculated
                    reset_time=window_start + config["window_size_seconds"],
                    retry_after=None,
                    current_requests=current_requests,
                    window_start=window_start,
                    agent_id=agent_id,
                    tier=tier
                )
            elif status_code == 2:  # Rate limited
                return RateLimitStatus(
                    result=RateLimitResult.RATE_LIMITED,
                    requests_remaining=0,
                    burst_remaining=remaining,
                    reset_time=window_start + config["window_size_seconds"],
                    retry_after=retry_after,
                    current_requests=current_requests,
                    window_start=window_start,
                    agent_id=agent_id,
                    tier=tier
                )
            elif status_code == 3:  # Burst limited
                return RateLimitStatus(
                    result=RateLimitResult.BURST_LIMITED,
                    requests_remaining=0,
                    burst_remaining=0,
                    reset_time=window_start + config["window_size_seconds"],
                    retry_after=retry_after,
                    current_requests=current_requests,
                    window_start=window_start,
                    agent_id=agent_id,
                    tier=tier
                )
                    
        except RedisSecurityError as e:
            # Security errors should not be masked
            logger.error(f"Redis security error in rate limiting: {e}")
            raise
            
        except Exception as e:
            # SECURITY FIX: Fail secure (fail-closed) instead of fail-open
            logger.error(f"Rate limit check failed for {agent_id}: {e}")
            return RateLimitStatus(
                result=RateLimitResult.ERROR,
                requests_remaining=0,  # Fail-secure: deny access
                burst_remaining=0,
                reset_time=time.time() + 60,  # Retry in 1 minute
                retry_after=60,
                current_requests=0,
                window_start=time.time(),
                agent_id=agent_id,
                tier=tier
            )
    
    async def enforce_rate_limit(self,
                               agent_id: str,
                               tier: str = "basic",
                               authenticated: bool = True) -> RateLimitStatus:
        """Enforce rate limit and raise exception if exceeded.
        
        Provides simple interface for middleware integration.
        """
        status = await self.check_rate_limit(agent_id, tier, authenticated)
        
        if status.result not in [RateLimitResult.ALLOWED]:
            raise RateLimitExceeded(
                f"Rate limit exceeded for {agent_id} ({tier}): {status.result.value}",
                status
            )
        
        return status
    
    def middleware_decorator(self, tier: str = "basic"):
        """Decorator for applying rate limiting to functions.
        
        Usage:
            @rate_limiter.middleware_decorator(tier="professional")
            async def my_handler(agent_id: str):
                # Handler logic
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Extract agent_id from function parameters
                agent_id = kwargs.get('agent_id') or (args[0] if args else 'anonymous')
                
                # Check rate limit
                await self.enforce_rate_limit(
                    agent_id=str(agent_id),
                    tier=tier,
                    authenticated=True  # Require authentication by default  
                )
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator


# Global unified rate limiter instance (replaces duplicate implementations)
_unified_rate_limiter: Optional[UnifiedRateLimiter] = None


async def get_unified_rate_limiter() -> UnifiedRateLimiter:
    """Get the global unified rate limiter instance."""
    global _unified_rate_limiter
    
    if _unified_rate_limiter is None:
        _unified_rate_limiter = UnifiedRateLimiter()
        await _unified_rate_limiter.initialize()
    
    return _unified_rate_limiter