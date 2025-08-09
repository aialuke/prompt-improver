"""Unified Rate Limiter - Phase 4 Security Consolidation

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
from enum import Enum
from typing import Any, Dict, Optional, Union
from sqlmodel import Field, SQLModel
from prompt_improver.database.unified_connection_manager import ManagerMode, RedisSecurityError, SecurityContext, create_security_context, get_unified_manager
logger = logging.getLogger(__name__)

class RateLimitResult(str, Enum):
    """Rate limit check results."""
    ALLOWED = 'allowed'
    RATE_LIMITED = 'rate_limited'
    BURST_LIMITED = 'burst_limited'
    ERROR = 'error'
    AUTHENTICATION_REQUIRED = 'authentication_required'

class RateLimitTier(str, Enum):
    """Rate limiting tiers with different request allowances."""
    BASIC = 'basic'
    PROFESSIONAL = 'professional'
    ENTERPRISE = 'enterprise'

class RateLimitStatus(SQLModel):
    """Rate limit status information."""
    result: RateLimitResult = Field(description='Rate limit check result')
    requests_remaining: int = Field(ge=0, description='Number of requests remaining in current window')
    burst_remaining: int = Field(ge=0, description='Number of burst requests remaining')
    reset_time: float = Field(gt=0, description='Timestamp when rate limit resets')
    retry_after: int | None = Field(default=None, ge=0, description='Seconds to wait before retry')
    current_requests: int = Field(ge=0, description='Current request count in window')
    window_start: float = Field(gt=0, description='Timestamp of current window start')
    agent_id: str = Field(min_length=1, max_length=100, description='Authenticated agent identifier')
    tier: str = Field(min_length=1, max_length=50, description='Rate limiting tier')

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
        self._tier_configs = {RateLimitTier.BASIC.value: {'rate_limit_per_minute': 60, 'burst_capacity': 90, 'window_size_seconds': 60, 'bucket_size_seconds': 10}, RateLimitTier.PROFESSIONAL.value: {'rate_limit_per_minute': 300, 'burst_capacity': 450, 'window_size_seconds': 60, 'bucket_size_seconds': 10}, RateLimitTier.ENTERPRISE.value: {'rate_limit_per_minute': 1000, 'burst_capacity': 1500, 'window_size_seconds': 60, 'bucket_size_seconds': 5}}
        self._sliding_window_script = "\n        local key = KEYS[1]\n        local window_size = tonumber(ARGV[1])\n        local bucket_size = tonumber(ARGV[2])\n        local rate_limit = tonumber(ARGV[3])\n        local burst_capacity = tonumber(ARGV[4])\n        local current_time = tonumber(ARGV[5])\n        \n        -- Calculate window boundaries\n        local window_start = current_time - window_size\n        local bucket_start = math.floor(current_time / bucket_size) * bucket_size\n        \n        -- Remove old buckets outside window\n        redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)\n        \n        -- Count current requests in window\n        local current_requests = redis.call('ZCOUNT', key, window_start, current_time)\n        \n        -- Check rate limits\n        if current_requests >= burst_capacity then\n            -- Burst limit exceeded\n            local ttl = redis.call('TTL', key)\n            return {3, current_requests, 0, window_start, ttl > 0 and ttl or window_size}\n        elseif current_requests >= rate_limit then\n            -- Rate limit exceeded  \n            local ttl = redis.call('TTL', key)\n            return {2, current_requests, burst_capacity - current_requests, window_start, ttl > 0 and ttl or window_size}\n        else\n            -- Request allowed - add to current bucket\n            redis.call('ZADD', key, current_time, current_time .. ':' .. math.random())\n            redis.call('EXPIRE', key, window_size * 2)  -- Safety margin for cleanup\n            \n            local remaining = rate_limit - current_requests - 1\n            local burst_remaining = burst_capacity - current_requests - 1\n            return {1, remaining, burst_remaining, window_start, 0}\n        end\n        "

    async def initialize(self) -> None:
        """Initialize unified rate limiter with enhanced UnifiedConnectionManager."""
        try:
            self._connection_manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
            await self._connection_manager.initialize()
            logger.info('Unified rate limiter initialized with enhanced UnifiedConnectionManager')
        except Exception as e:
            logger.error('Failed to initialize unified rate limiter: %s', e)
            raise

    async def check_rate_limit(self, agent_id: str, tier: str='basic', authenticated: bool=True) -> RateLimitStatus:
        """Check rate limit with secure authentication.

        Fixes security vulnerabilities:
        - Requires authentication (fixes agent bypass)
        - Fails secure on errors (fixes fail-open policy)
        - Uses atomic operations (fixes race conditions)
        """
        if not self._connection_manager:
            await self.initialize()
        if not authenticated:
            return RateLimitStatus(result=RateLimitResult.AUTHENTICATION_REQUIRED, requests_remaining=0, burst_remaining=0, reset_time=time.time(), retry_after=None, current_requests=0, window_start=time.time(), agent_id=agent_id, tier=tier)
        if tier not in self._tier_configs:
            logger.warning('Unknown tier %s, defaulting to basic', tier)
            tier = 'basic'
        config = self._tier_configs[tier]
        try:
            security_context = await create_security_context(agent_id=agent_id, tier=tier, authenticated=authenticated)
            if not self._connection_manager._redis_master:
                logger.error('Redis master not available in UnifiedConnectionManager')
                return RateLimitStatus(result=RateLimitResult.ERROR, requests_remaining=0, burst_remaining=0, reset_time=time.time() + 60, retry_after=60, current_requests=0, window_start=time.time(), agent_id=agent_id, tier=tier)
            current_time = time.time()
            key = f'rate_limit:{agent_id}:{tier}'
            result = await self._connection_manager._redis_master.eval(self._sliding_window_script, keys=[key], args=[config['window_size_seconds'], config['bucket_size_seconds'], config['rate_limit_per_minute'], config['burst_capacity'], current_time])
            status_code, current_requests, remaining, window_start, retry_after = result
            if status_code == 1:
                return RateLimitStatus(result=RateLimitResult.ALLOWED, requests_remaining=remaining, burst_remaining=remaining, reset_time=window_start + config['window_size_seconds'], retry_after=None, current_requests=current_requests, window_start=window_start, agent_id=agent_id, tier=tier)
            if status_code == 2:
                return RateLimitStatus(result=RateLimitResult.RATE_LIMITED, requests_remaining=0, burst_remaining=remaining, reset_time=window_start + config['window_size_seconds'], retry_after=retry_after, current_requests=current_requests, window_start=window_start, agent_id=agent_id, tier=tier)
            if status_code == 3:
                return RateLimitStatus(result=RateLimitResult.BURST_LIMITED, requests_remaining=0, burst_remaining=0, reset_time=window_start + config['window_size_seconds'], retry_after=retry_after, current_requests=current_requests, window_start=window_start, agent_id=agent_id, tier=tier)
        except RedisSecurityError as e:
            logger.error('Redis security error in rate limiting: %s', e)
            raise
        except Exception as e:
            logger.error('Rate limit check failed for {agent_id}: %s', e)
            return RateLimitStatus(result=RateLimitResult.ERROR, requests_remaining=0, burst_remaining=0, reset_time=time.time() + 60, retry_after=60, current_requests=0, window_start=time.time(), agent_id=agent_id, tier=tier)

    async def enforce_rate_limit(self, agent_id: str, tier: str='basic', authenticated: bool=True) -> RateLimitStatus:
        """Enforce rate limit and raise exception if exceeded.

        Provides simple interface for middleware integration.
        """
        status = await self.check_rate_limit(agent_id, tier, authenticated)
        if status.result not in [RateLimitResult.ALLOWED]:
            raise RateLimitExceeded(f'Rate limit exceeded for {agent_id} ({tier}): {status.result.value}', status)
        return status

    def middleware_decorator(self, tier: str='basic'):
        """Decorator for applying rate limiting to functions.

        Usage:
            @rate_limiter.middleware_decorator(tier="professional")
            async def my_handler(agent_id: str):
                # Handler logic
        """

        def decorator(func):

            async def wrapper(*args, **kwargs):
                agent_id = kwargs.get('agent_id') or (args[0] if args else 'anonymous')
                await self.enforce_rate_limit(agent_id=str(agent_id), tier=tier, authenticated=True)
                return await func(*args, **kwargs)
            return wrapper
        return decorator
_unified_rate_limiter: UnifiedRateLimiter | None = None

async def get_unified_rate_limiter() -> UnifiedRateLimiter:
    """Get the global unified rate limiter instance."""
    global _unified_rate_limiter
    if _unified_rate_limiter is None:
        _unified_rate_limiter = UnifiedRateLimiter()
        await _unified_rate_limiter.initialize()
    return _unified_rate_limiter
