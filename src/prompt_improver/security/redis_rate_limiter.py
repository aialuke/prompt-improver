"""Redis-based sliding window rate limiter with burst handling.

Implements 2025 best practices for distributed rate limiting using Redis
with Lua scripts for atomic operations and sliding window algorithm.

MIGRATED: Now uses UnifiedConnectionManager instead of hardcoded Redis URLs.
Fixes CVSS 7.5 authentication bypass vulnerability with fail-secure policy.
"""
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
from coredis.exceptions import ConnectionError, TimeoutError
from prompt_improver.database.unified_connection_manager import ManagerMode, SecurityContext, create_security_context, get_unified_manager
logger = logging.getLogger(__name__)

class RateLimitResult(str, Enum):
    """Rate limit check results."""
    ALLOWED = 'allowed'
    RATE_LIMITED = 'rate_limited'
    BURST_LIMITED = 'burst_limited'
    ERROR = 'error'

@dataclass
class RateLimitStatus:
    """Rate limit status information."""
    result: RateLimitResult
    requests_remaining: int
    burst_remaining: int
    reset_time: float
    retry_after: int | None
    current_requests: int
    window_start: float

class SlidingWindowRateLimiter:
    """Redis-based sliding window rate limiter with burst handling.

    Implements a sliding window algorithm that:
    - Tracks requests in time-based buckets
    - Supports burst capacity for short-term spikes
    - Uses Lua scripts for atomic operations
    - Provides accurate rate limiting across distributed systems

    MIGRATED: Now uses UnifiedConnectionManager for secure Redis connections.
    Fixes hardcoded Redis URLs and implements fail-secure policy.
    """

    def __init__(self):
        """Initialize sliding window rate limiter.

        Uses UnifiedConnectionManager for all Redis connections.
        """
        self._connection_manager = None
        self.rate_limit_script = '\n        local key = KEYS[1]\n        local window_size = tonumber(ARGV[1])\n        local rate_limit = tonumber(ARGV[2])\n        local burst_limit = tonumber(ARGV[3])\n        local current_time = tonumber(ARGV[4])\n        local bucket_size = tonumber(ARGV[5])\n\n        -- Calculate window boundaries\n        local window_start = current_time - window_size\n        local current_bucket = math.floor(current_time / bucket_size)\n        local bucket_key = key .. ":" .. current_bucket\n\n        -- Clean up old buckets (older than window_size)\n        local cleanup_before = math.floor(window_start / bucket_size)\n        for i = 0, 10 do\n            local old_bucket = cleanup_before - i\n            local old_key = key .. ":" .. old_bucket\n            redis.call(\'DEL\', old_key)\n        end\n\n        -- Count requests in current window\n        local total_requests = 0\n        local buckets_to_check = math.ceil(window_size / bucket_size)\n\n        for i = 0, buckets_to_check do\n            local bucket_time = current_bucket - i\n            local check_key = key .. ":" .. bucket_time\n            local bucket_count = redis.call(\'GET\', check_key)\n            if bucket_count then\n                -- Weight the bucket based on how much of it falls within our window\n                local bucket_start_time = bucket_time * bucket_size\n                local bucket_end_time = bucket_start_time + bucket_size\n\n                if bucket_end_time > window_start then\n                    if bucket_start_time >= window_start then\n                        -- Entire bucket is within window\n                        total_requests = total_requests + tonumber(bucket_count)\n                    else\n                        -- Partial bucket - estimate based on overlap\n                        local overlap = (bucket_end_time - window_start) / bucket_size\n                        total_requests = total_requests + (tonumber(bucket_count) * overlap)\n                    end\n                end\n            end\n        end\n\n        -- Check rate limits\n        local rate_exceeded = total_requests >= rate_limit\n        local burst_exceeded = total_requests >= burst_limit\n\n        if burst_exceeded then\n            return {\n                "result", "burst_limited",\n                "current_requests", math.floor(total_requests),\n                "requests_remaining", 0,\n                "burst_remaining", 0,\n                "window_start", window_start,\n                "retry_after", math.ceil(bucket_size)\n            }\n        elseif rate_exceeded then\n            return {\n                "result", "rate_limited",\n                "current_requests", math.floor(total_requests),\n                "requests_remaining", 0,\n                "burst_remaining", burst_limit - math.floor(total_requests),\n                "window_start", window_start,\n                "retry_after", math.ceil(bucket_size)\n            }\n        else\n            -- Allow request and increment counter\n            redis.call(\'INCR\', bucket_key)\n            redis.call(\'EXPIRE\', bucket_key, window_size + bucket_size)\n\n            return {\n                "result", "allowed",\n                "current_requests", math.floor(total_requests) + 1,\n                "requests_remaining", rate_limit - math.floor(total_requests) - 1,\n                "burst_remaining", burst_limit - math.floor(total_requests) - 1,\n                "window_start", window_start,\n                "retry_after", 0\n            }\n        end\n        '

    async def _get_redis(self):
        """Get Redis client instance via UnifiedConnectionManager.

        MIGRATED: Now uses UnifiedConnectionManager instead of direct Redis connections.
        Eliminates hardcoded Redis URLs and provides secure connection management.
        """
        if self._connection_manager is None:
            try:
                self._connection_manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
                await self._connection_manager.initialize()
                logger.debug('SlidingWindowRateLimiter: Initialized UnifiedConnectionManager')
            except Exception as e:
                logger.error('Failed to initialize UnifiedConnectionManager: %s', e)
                raise
        if not self._connection_manager._redis_master:
            raise ConnectionError('Redis master connection not available in UnifiedConnectionManager')
        return self._connection_manager._redis_master

    async def check_rate_limit(self, identifier: str, rate_limit_per_minute: int, burst_capacity: int, window_size_seconds: int=60, bucket_size_seconds: int=10) -> RateLimitStatus:
        """Check rate limit for identifier using sliding window algorithm.

        Args:
            identifier: Unique identifier (e.g., user_id, ip_address, agent_id)
            rate_limit_per_minute: Maximum requests per minute
            burst_capacity: Maximum burst requests allowed
            window_size_seconds: Size of sliding window in seconds
            bucket_size_seconds: Size of time buckets in seconds

        Returns:
            RateLimitStatus with current status and remaining capacity
        """
        try:
            redis = await self._get_redis()
            current_time = time.time()
            key = f'rate_limit:sliding:{identifier}'
            result = await redis.eval(self.rate_limit_script, keys=[key], args=[window_size_seconds, rate_limit_per_minute, burst_capacity, current_time, bucket_size_seconds])
            result_dict = {}
            for i in range(0, len(result), 2):
                result_dict[result[i]] = result[i + 1]
            return RateLimitStatus(result=RateLimitResult(result_dict['result']), requests_remaining=int(result_dict['requests_remaining']), burst_remaining=int(result_dict['burst_remaining']), reset_time=current_time + window_size_seconds, retry_after=int(result_dict['retry_after']) if result_dict['retry_after'] > 0 else None, current_requests=int(result_dict['current_requests']), window_start=float(result_dict['window_start']))
        except (ConnectionError, TimeoutError) as e:
            logger.error('Redis connection error in rate limiter (UnifiedConnectionManager): %s', e)
            return RateLimitStatus(result=RateLimitResult.ERROR, requests_remaining=0, burst_remaining=0, reset_time=time.time() + 60, retry_after=60, current_requests=0, window_start=time.time() - window_size_seconds)
        except Exception as e:
            logger.error('Unexpected error in rate limiter (UnifiedConnectionManager): %s', e)
            return RateLimitStatus(result=RateLimitResult.ERROR, requests_remaining=0, burst_remaining=0, reset_time=time.time() + 60, retry_after=60, current_requests=0, window_start=time.time() - window_size_seconds)

    async def get_rate_limit_info(self, identifier: str, window_size_seconds: int=60, bucket_size_seconds: int=10) -> dict[str, Any]:
        """Get current rate limit information without incrementing counters.

        Args:
            identifier: Unique identifier
            window_size_seconds: Size of sliding window in seconds
            bucket_size_seconds: Size of time buckets in seconds

        Returns:
            Dict with current rate limit information
        """
        try:
            redis = await self._get_redis()
            current_time = time.time()
            window_start = current_time - window_size_seconds
            current_bucket = int(current_time // bucket_size_seconds)
            buckets_to_check = int(window_size_seconds // bucket_size_seconds) + 1
            total_requests = 0
            for i in range(buckets_to_check):
                bucket_time = current_bucket - i
                bucket_key = f'rate_limit:sliding:{identifier}:{bucket_time}'
                bucket_count = await redis.get(bucket_key)
                if bucket_count:
                    bucket_start_time = bucket_time * bucket_size_seconds
                    bucket_end_time = bucket_start_time + bucket_size_seconds
                    if bucket_end_time > window_start:
                        if bucket_start_time >= window_start:
                            total_requests += int(bucket_count)
                        else:
                            overlap = (bucket_end_time - window_start) / bucket_size_seconds
                            total_requests += int(bucket_count) * overlap
            return {'identifier': identifier, 'current_requests': int(total_requests), 'window_start': window_start, 'window_end': current_time, 'window_size_seconds': window_size_seconds, 'bucket_size_seconds': bucket_size_seconds}
        except Exception as e:
            logger.error('Error getting rate limit info (UnifiedConnectionManager): %s', e)
            return {'identifier': identifier, 'current_requests': 0, 'window_start': current_time - window_size_seconds, 'window_end': current_time, 'window_size_seconds': window_size_seconds, 'bucket_size_seconds': bucket_size_seconds, 'error': str(e)}

    async def reset_rate_limit(self, identifier: str) -> bool:
        """Reset rate limit for identifier (admin function).

        Args:
            identifier: Unique identifier to reset

        Returns:
            True if successful, False otherwise
        """
        try:
            redis = await self._get_redis()
            pattern = f'rate_limit:sliding:{identifier}:*'
            keys = []
            async for key in redis.scan_iter(match=pattern):
                keys.append(key)
            if keys:
                await redis.delete(*keys)
                logger.info('Reset rate limit for %s, deleted %s buckets', identifier, len(keys))
            return True
        except Exception as e:
            logger.error('Error resetting rate limit for %s (UnifiedConnectionManager): %s', identifier, e)
            return False
