"""
Integration tests for Phase 1 Redis Rate Limiting System
Tests sliding window rate limiting with real Redis connections and concurrent requests
"""

import asyncio
import os
import pytest
import time
from typing import List

# Import the components we're testing
from prompt_improver.security.redis_rate_limiter import (
    SlidingWindowRateLimiter,
    RateLimitResult,
    RateLimitStatus
)
from prompt_improver.security.rate_limit_middleware import (
    MCPRateLimitMiddleware,
    RateLimitExceeded
)
from prompt_improver.security.mcp_authentication import RateLimitTier

# Test configuration
TEST_REDIS_URL = "redis://localhost:6379/15"  # Use test database


class TestPhase1RateLimiting:
    """Integration tests for Phase 1 rate limiting system."""
    
    @pytest.fixture(scope="class")
    def rate_limiter(self):
        """Create sliding window rate limiter for testing."""
        return SlidingWindowRateLimiter(redis_url=TEST_REDIS_URL)
    
    @pytest.fixture(scope="class")
    def rate_limit_middleware(self):
        """Create rate limiting middleware for testing."""
        return MCPRateLimitMiddleware(redis_url=TEST_REDIS_URL)
    
    @pytest.fixture(autouse=True)
    async def cleanup_redis(self, rate_limiter):
        """Clean up Redis test data before each test."""
        try:
            redis = await rate_limiter._get_redis()
            await redis.flushdb()  # Clear test database
        except Exception:
            pass  # Ignore cleanup errors

    def test_sliding_window_basic_functionality(self, rate_limiter):
        """Test basic sliding window rate limiting functionality."""
        async def run_test():
            identifier = "test_basic_user"
            rate_limit = 5
            burst_capacity = 8
            
            # First 5 requests should be allowed
            for i in range(5):
                status = await rate_limiter.check_rate_limit(
                    identifier=identifier,
                    rate_limit_per_minute=rate_limit,
                    burst_capacity=burst_capacity
                )
                assert status.result == RateLimitResult.ALLOWED
                assert status.requests_remaining == rate_limit - i - 1
                assert status.current_requests == i + 1
            
            # 6th request should be rate limited (exceeds rate limit but within burst)
            status = await rate_limiter.check_rate_limit(
                identifier=identifier,
                rate_limit_per_minute=rate_limit,
                burst_capacity=burst_capacity
            )
            assert status.result == RateLimitResult.RATE_LIMITED
            assert status.requests_remaining == 0
            assert status.burst_remaining > 0
        
        asyncio.run(run_test())

    def test_burst_capacity_handling(self, rate_limiter):
        """Test burst capacity handling for short-term spikes."""
        async def run_test():
            identifier = "test_burst_user"
            rate_limit = 3
            burst_capacity = 6
            
            # Send requests up to burst capacity
            for i in range(6):
                status = await rate_limiter.check_rate_limit(
                    identifier=identifier,
                    rate_limit_per_minute=rate_limit,
                    burst_capacity=burst_capacity
                )
                if i < 3:
                    assert status.result == RateLimitResult.ALLOWED
                else:
                    assert status.result == RateLimitResult.RATE_LIMITED
                    assert status.burst_remaining > 0
            
            # 7th request should be burst limited
            status = await rate_limiter.check_rate_limit(
                identifier=identifier,
                rate_limit_per_minute=rate_limit,
                burst_capacity=burst_capacity
            )
            assert status.result == RateLimitResult.BURST_LIMITED
            assert status.burst_remaining == 0
        
        asyncio.run(run_test())

    def test_sliding_window_time_decay(self, rate_limiter):
        """Test that requests decay over time in sliding window."""
        async def run_test():
            identifier = "test_decay_user"
            rate_limit = 3
            burst_capacity = 5
            window_size = 10  # 10 seconds
            bucket_size = 2   # 2 second buckets
            
            # Fill up the rate limit
            for i in range(3):
                status = await rate_limiter.check_rate_limit(
                    identifier=identifier,
                    rate_limit_per_minute=rate_limit,
                    burst_capacity=burst_capacity,
                    window_size_seconds=window_size,
                    bucket_size_seconds=bucket_size
                )
                assert status.result == RateLimitResult.ALLOWED
            
            # Next request should be rate limited
            status = await rate_limiter.check_rate_limit(
                identifier=identifier,
                rate_limit_per_minute=rate_limit,
                burst_capacity=burst_capacity,
                window_size_seconds=window_size,
                bucket_size_seconds=bucket_size
            )
            assert status.result == RateLimitResult.RATE_LIMITED
            
            # Wait for one bucket to expire
            await asyncio.sleep(3)
            
            # Should now allow requests again
            status = await rate_limiter.check_rate_limit(
                identifier=identifier,
                rate_limit_per_minute=rate_limit,
                burst_capacity=burst_capacity,
                window_size_seconds=window_size,
                bucket_size_seconds=bucket_size
            )
            assert status.result == RateLimitResult.ALLOWED
        
        asyncio.run(run_test())

    def test_concurrent_requests_same_user(self, rate_limiter):
        """Test concurrent requests from the same user."""
        async def run_test():
            identifier = "test_concurrent_user"
            rate_limit = 10
            burst_capacity = 15
            
            async def make_request():
                return await rate_limiter.check_rate_limit(
                    identifier=identifier,
                    rate_limit_per_minute=rate_limit,
                    burst_capacity=burst_capacity
                )
            
            # Send 20 concurrent requests
            tasks = [make_request() for _ in range(20)]
            results = await asyncio.gather(*tasks)
            
            # Count allowed vs rate limited
            allowed_count = sum(1 for r in results if r.result == RateLimitResult.ALLOWED)
            rate_limited_count = sum(1 for r in results if r.result == RateLimitResult.RATE_LIMITED)
            burst_limited_count = sum(1 for r in results if r.result == RateLimitResult.BURST_LIMITED)
            
            # Should allow up to rate limit, then rate limit until burst, then burst limit
            assert allowed_count == rate_limit
            assert rate_limited_count == burst_capacity - rate_limit
            assert burst_limited_count == 20 - burst_capacity
        
        asyncio.run(run_test())

    def test_different_users_isolated(self, rate_limiter):
        """Test that different users have isolated rate limits."""
        async def run_test():
            user1 = "test_user_1"
            user2 = "test_user_2"
            rate_limit = 3
            burst_capacity = 5
            
            # Fill rate limit for user1
            for i in range(3):
                status = await rate_limiter.check_rate_limit(
                    identifier=user1,
                    rate_limit_per_minute=rate_limit,
                    burst_capacity=burst_capacity
                )
                assert status.result == RateLimitResult.ALLOWED
            
            # User1 should now be rate limited
            status = await rate_limiter.check_rate_limit(
                identifier=user1,
                rate_limit_per_minute=rate_limit,
                burst_capacity=burst_capacity
            )
            assert status.result == RateLimitResult.RATE_LIMITED
            
            # User2 should still be allowed
            status = await rate_limiter.check_rate_limit(
                identifier=user2,
                rate_limit_per_minute=rate_limit,
                burst_capacity=burst_capacity
            )
            assert status.result == RateLimitResult.ALLOWED
        
        asyncio.run(run_test())

    def test_middleware_tier_based_limits(self, rate_limit_middleware):
        """Test middleware with different tier-based rate limits."""
        async def run_test():
            # Test basic tier
            basic_agent = "basic_agent_001"
            basic_tier = RateLimitTier.BASIC.value
            
            # Should allow up to 60 requests per minute for basic tier
            for i in range(60):
                status = await rate_limit_middleware.check_rate_limit(
                    agent_id=basic_agent,
                    rate_limit_tier=basic_tier
                )
                assert status.result == RateLimitResult.ALLOWED
            
            # 61st request should be rate limited
            try:
                await rate_limit_middleware.check_rate_limit(
                    agent_id=basic_agent,
                    rate_limit_tier=basic_tier
                )
                assert False, "Should have raised RateLimitExceeded"
            except RateLimitExceeded as e:
                assert e.tier == basic_tier
                assert e.status.result == RateLimitResult.RATE_LIMITED
            
            # Test professional tier has higher limits
            pro_agent = "pro_agent_001"
            pro_tier = RateLimitTier.PROFESSIONAL.value
            
            # Should allow more requests for professional tier
            for i in range(100):  # Test first 100 requests
                status = await rate_limit_middleware.check_rate_limit(
                    agent_id=pro_agent,
                    rate_limit_tier=pro_tier
                )
                assert status.result == RateLimitResult.ALLOWED
        
        asyncio.run(run_test())

    def test_middleware_burst_handling(self, rate_limit_middleware):
        """Test middleware burst handling for different tiers."""
        async def run_test():
            agent_id = "burst_test_agent"
            tier = RateLimitTier.BASIC.value
            
            # Basic tier: 60 req/min, 90 burst
            # Send 90 requests quickly
            allowed_count = 0
            rate_limited_count = 0
            burst_limited_count = 0
            
            for i in range(95):
                try:
                    status = await rate_limit_middleware.check_rate_limit(
                        agent_id=agent_id,
                        rate_limit_tier=tier
                    )
                    allowed_count += 1
                except RateLimitExceeded as e:
                    if e.status.result == RateLimitResult.RATE_LIMITED:
                        rate_limited_count += 1
                    elif e.status.result == RateLimitResult.BURST_LIMITED:
                        burst_limited_count += 1
            
            # Should allow 60, rate limit 30, burst limit 5
            assert allowed_count == 60
            assert rate_limited_count == 30
            assert burst_limited_count == 5
        
        asyncio.run(run_test())

    def test_rate_limit_info_retrieval(self, rate_limit_middleware):
        """Test rate limit information retrieval."""
        async def run_test():
            agent_id = "info_test_agent"
            tier = RateLimitTier.PROFESSIONAL.value
            
            # Make some requests
            for i in range(10):
                await rate_limit_middleware.check_rate_limit(
                    agent_id=agent_id,
                    rate_limit_tier=tier
                )
            
            # Get rate limit info
            info = await rate_limit_middleware.get_rate_limit_info(agent_id, tier)
            
            assert info["tier"] == tier
            assert info["current_requests"] == 10
            assert info["rate_limit_per_minute"] == 300
            assert info["burst_capacity"] == 450
            assert info["requests_remaining"] == 290
            assert info["burst_remaining"] == 440
        
        asyncio.run(run_test())

    def test_rate_limit_reset(self, rate_limit_middleware):
        """Test rate limit reset functionality."""
        async def run_test():
            agent_id = "reset_test_agent"
            tier = RateLimitTier.BASIC.value
            
            # Fill up rate limit
            for i in range(60):
                await rate_limit_middleware.check_rate_limit(
                    agent_id=agent_id,
                    rate_limit_tier=tier
                )
            
            # Should be rate limited now
            try:
                await rate_limit_middleware.check_rate_limit(
                    agent_id=agent_id,
                    rate_limit_tier=tier
                )
                assert False, "Should have been rate limited"
            except RateLimitExceeded:
                pass
            
            # Reset rate limit
            success = await rate_limit_middleware.reset_agent_rate_limit(agent_id)
            assert success
            
            # Should now be allowed again
            status = await rate_limit_middleware.check_rate_limit(
                agent_id=agent_id,
                rate_limit_tier=tier
            )
            assert status.result == RateLimitResult.ALLOWED
        
        asyncio.run(run_test())

    def test_redis_failure_handling(self, rate_limiter):
        """Test graceful handling of Redis failures (fail-open behavior)."""
        async def run_test():
            # Create rate limiter with invalid Redis URL
            failing_limiter = SlidingWindowRateLimiter(redis_url="redis://invalid:9999/0")
            
            # Should fail open and allow requests
            status = await failing_limiter.check_rate_limit(
                identifier="test_user",
                rate_limit_per_minute=10,
                burst_capacity=15
            )
            
            assert status.result == RateLimitResult.ERROR
            assert status.requests_remaining > 0  # Should allow requests on error
        
        asyncio.run(run_test())

    def test_performance_under_load(self, rate_limiter):
        """Test rate limiter performance under concurrent load."""
        async def run_test():
            start_time = time.time()
            
            async def make_requests(user_id: str, count: int):
                for i in range(count):
                    await rate_limiter.check_rate_limit(
                        identifier=f"load_test_user_{user_id}",
                        rate_limit_per_minute=100,
                        burst_capacity=150
                    )
            
            # Simulate 10 users making 50 requests each concurrently
            tasks = [make_requests(str(i), 50) for i in range(10)]
            await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should complete 500 requests in reasonable time (< 5 seconds)
            assert total_time < 5.0, f"Rate limiting took too long: {total_time:.2f}s"
            
            # Calculate requests per second
            rps = 500 / total_time
            assert rps > 100, f"Rate limiter too slow: {rps:.1f} req/s"
        
        asyncio.run(run_test())
