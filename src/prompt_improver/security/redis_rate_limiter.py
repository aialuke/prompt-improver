"""Redis-based sliding window rate limiter with burst handling.

Implements 2025 best practices for distributed rate limiting using Redis
with Lua scripts for atomic operations and sliding window algorithm.
"""

import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

import coredis
from coredis import Redis
from coredis.exceptions import ConnectionError, TimeoutError

logger = logging.getLogger(__name__)

class RateLimitResult(str, Enum):
    """Rate limit check results."""
    ALLOWED = "allowed"
    RATE_LIMITED = "rate_limited"
    BURST_LIMITED = "burst_limited"
    ERROR = "error"

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

class SlidingWindowRateLimiter:
    """Redis-based sliding window rate limiter with burst handling.

    Implements a sliding window algorithm that:
    - Tracks requests in time-based buckets
    - Supports burst capacity for short-term spikes
    - Uses Lua scripts for atomic operations
    - Provides accurate rate limiting across distributed systems
    """

    def __init__(self, redis_client: Optional[Redis] = None, redis_url: str = "redis://localhost:6379/2"):
        """Initialize sliding window rate limiter.

        Args:
            redis_client: Optional Redis client instance
            redis_url: Redis connection URL (used if redis_client not provided)
        """
        self.redis_client = redis_client
        self.redis_url = redis_url
        self._redis = None

        # Lua script for atomic sliding window rate limiting
        self.rate_limit_script = """
        local key = KEYS[1]
        local window_size = tonumber(ARGV[1])
        local rate_limit = tonumber(ARGV[2])
        local burst_limit = tonumber(ARGV[3])
        local current_time = tonumber(ARGV[4])
        local bucket_size = tonumber(ARGV[5])

        -- Calculate window boundaries
        local window_start = current_time - window_size
        local current_bucket = math.floor(current_time / bucket_size)
        local bucket_key = key .. ":" .. current_bucket

        -- Clean up old buckets (older than window_size)
        local cleanup_before = math.floor(window_start / bucket_size)
        for i = 0, 10 do
            local old_bucket = cleanup_before - i
            local old_key = key .. ":" .. old_bucket
            redis.call('DEL', old_key)
        end

        -- Count requests in current window
        local total_requests = 0
        local buckets_to_check = math.ceil(window_size / bucket_size)

        for i = 0, buckets_to_check do
            local bucket_time = current_bucket - i
            local check_key = key .. ":" .. bucket_time
            local bucket_count = redis.call('GET', check_key)
            if bucket_count then
                -- Weight the bucket based on how much of it falls within our window
                local bucket_start_time = bucket_time * bucket_size
                local bucket_end_time = bucket_start_time + bucket_size

                if bucket_end_time > window_start then
                    if bucket_start_time >= window_start then
                        -- Entire bucket is within window
                        total_requests = total_requests + tonumber(bucket_count)
                    else
                        -- Partial bucket - estimate based on overlap
                        local overlap = (bucket_end_time - window_start) / bucket_size
                        total_requests = total_requests + (tonumber(bucket_count) * overlap)
                    end
                end
            end
        end

        -- Check rate limits
        local rate_exceeded = total_requests >= rate_limit
        local burst_exceeded = total_requests >= burst_limit

        if burst_exceeded then
            return {
                "result", "burst_limited",
                "current_requests", math.floor(total_requests),
                "requests_remaining", 0,
                "burst_remaining", 0,
                "window_start", window_start,
                "retry_after", math.ceil(bucket_size)
            }
        elseif rate_exceeded then
            return {
                "result", "rate_limited",
                "current_requests", math.floor(total_requests),
                "requests_remaining", 0,
                "burst_remaining", burst_limit - math.floor(total_requests),
                "window_start", window_start,
                "retry_after", math.ceil(bucket_size)
            }
        else
            -- Allow request and increment counter
            redis.call('INCR', bucket_key)
            redis.call('EXPIRE', bucket_key, window_size + bucket_size)

            return {
                "result", "allowed",
                "current_requests", math.floor(total_requests) + 1,
                "requests_remaining", rate_limit - math.floor(total_requests) - 1,
                "burst_remaining", burst_limit - math.floor(total_requests) - 1,
                "window_start", window_start,
                "retry_after", 0
            }
        end
        """

    async def _get_redis(self) -> Redis:
        """Get Redis client instance."""
        if self._redis is None:
            if self.redis_client:
                self._redis = self.redis_client
            else:
                self._redis = coredis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    async def check_rate_limit(
        self,
        identifier: str,
        rate_limit_per_minute: int,
        burst_capacity: int,
        window_size_seconds: int = 60,
        bucket_size_seconds: int = 10
    ) -> RateLimitStatus:
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

            # Create rate limit key
            key = f"rate_limit:sliding:{identifier}"

            # Execute Lua script atomically
            result = await redis.eval(
                self.rate_limit_script,
                keys=[key],
                args=[
                    window_size_seconds,
                    rate_limit_per_minute,
                    burst_capacity,
                    current_time,
                    bucket_size_seconds
                ]
            )

            # Parse result from Lua script
            result_dict = {}
            for i in range(0, len(result), 2):
                result_dict[result[i]] = result[i + 1]

            return RateLimitStatus(
                result=RateLimitResult(result_dict["result"]),
                requests_remaining=int(result_dict["requests_remaining"]),
                burst_remaining=int(result_dict["burst_remaining"]),
                reset_time=current_time + window_size_seconds,
                retry_after=int(result_dict["retry_after"]) if result_dict["retry_after"] > 0 else None,
                current_requests=int(result_dict["current_requests"]),
                window_start=float(result_dict["window_start"])
            )

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error in rate limiter: {e}")
            # Fail open - allow request when Redis is unavailable
            return RateLimitStatus(
                result=RateLimitResult.ERROR,
                requests_remaining=rate_limit_per_minute,
                burst_remaining=burst_capacity,
                reset_time=time.time() + window_size_seconds,
                retry_after=None,
                current_requests=0,
                window_start=time.time() - window_size_seconds
            )
        except Exception as e:
            logger.error(f"Unexpected error in rate limiter: {e}")
            # Fail open for unexpected errors
            return RateLimitStatus(
                result=RateLimitResult.ERROR,
                requests_remaining=rate_limit_per_minute,
                burst_remaining=burst_capacity,
                reset_time=time.time() + window_size_seconds,
                retry_after=None,
                current_requests=0,
                window_start=time.time() - window_size_seconds
            )

    async def get_rate_limit_info(
        self,
        identifier: str,
        window_size_seconds: int = 60,
        bucket_size_seconds: int = 10
    ) -> Dict[str, Any]:
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

            # Calculate current requests in window
            current_bucket = int(current_time // bucket_size_seconds)
            buckets_to_check = int(window_size_seconds // bucket_size_seconds) + 1

            total_requests = 0
            for i in range(buckets_to_check):
                bucket_time = current_bucket - i
                bucket_key = f"rate_limit:sliding:{identifier}:{bucket_time}"

                bucket_count = await redis.get(bucket_key)
                if bucket_count:
                    bucket_start_time = bucket_time * bucket_size_seconds
                    bucket_end_time = bucket_start_time + bucket_size_seconds

                    if bucket_end_time > window_start:
                        if bucket_start_time >= window_start:
                            total_requests += int(bucket_count)
                        else:
                            # Partial bucket
                            overlap = (bucket_end_time - window_start) / bucket_size_seconds
                            total_requests += int(bucket_count) * overlap

            return {
                "identifier": identifier,
                "current_requests": int(total_requests),
                "window_start": window_start,
                "window_end": current_time,
                "window_size_seconds": window_size_seconds,
                "bucket_size_seconds": bucket_size_seconds
            }

        except Exception as e:
            logger.error(f"Error getting rate limit info: {e}")
            return {
                "identifier": identifier,
                "current_requests": 0,
                "window_start": current_time - window_size_seconds,
                "window_end": current_time,
                "window_size_seconds": window_size_seconds,
                "bucket_size_seconds": bucket_size_seconds,
                "error": str(e)
            }

    async def reset_rate_limit(self, identifier: str) -> bool:
        """Reset rate limit for identifier (admin function).

        Args:
            identifier: Unique identifier to reset

        Returns:
            True if successful, False otherwise
        """
        try:
            redis = await self._get_redis()
            pattern = f"rate_limit:sliding:{identifier}:*"

            # Get all keys matching pattern
            keys = []
            async for key in redis.scan_iter(match=pattern):
                keys.append(key)

            # Delete all matching keys
            if keys:
                await redis.delete(*keys)
                logger.info(f"Reset rate limit for {identifier}, deleted {len(keys)} buckets")

            return True

        except Exception as e:
            logger.error(f"Error resetting rate limit for {identifier}: {e}")
            return False
