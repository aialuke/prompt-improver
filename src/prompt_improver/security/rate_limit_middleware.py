"""Rate limiting middleware for MCP server with JWT integration.

Provides rate limiting middleware that integrates with JWT authentication
to enforce tier-based rate limits with burst handling.
"""

import logging
import os
import time
from typing import Any, Callable, Dict, Optional
from functools import wraps

from .redis_rate_limiter import SlidingWindowRateLimiter, RateLimitResult, RateLimitStatus
from .mcp_authentication import RateLimitTier

logger = logging.getLogger(__name__)

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str, status: RateLimitStatus, tier: str):
        self.message = message
        self.status = status
        self.tier = tier
        super().__init__(message)

class MCPRateLimitMiddleware:
    """Rate limiting middleware for MCP server with JWT tier integration."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize rate limiting middleware.
        
        Args:
            redis_url: Redis connection URL (defaults to environment variable)
        """
        self.redis_url = redis_url or os.getenv("MCP_RATE_LIMIT_REDIS_URL", "redis://localhost:6379/2")
        self.rate_limiter = SlidingWindowRateLimiter(redis_url=self.redis_url)
        
        # Rate limit configurations by tier (per minute)
        self.tier_configs = {
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
                "bucket_size_seconds": 10
            }
        }
        
        # Performance tracking
        self._rate_limit_checks = 0
        self._rate_limit_blocks = 0
        self._rate_limit_errors = 0

    async def check_rate_limit(
        self,
        agent_id: str,
        rate_limit_tier: str,
        additional_identifier: Optional[str] = None
    ) -> RateLimitStatus:
        """Check rate limit for agent with tier-based configuration.
        
        Args:
            agent_id: Agent identifier from JWT token
            rate_limit_tier: Rate limit tier from JWT token
            additional_identifier: Optional additional identifier (e.g., IP address)
            
        Returns:
            RateLimitStatus with current rate limit status
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        self._rate_limit_checks += 1
        
        # Get configuration for tier
        config = self.tier_configs.get(rate_limit_tier, self.tier_configs[RateLimitTier.BASIC.value])
        
        # Create composite identifier
        identifier = f"agent:{agent_id}"
        if additional_identifier:
            identifier += f":ip:{additional_identifier}"
        
        try:
            # Check rate limit
            status = await self.rate_limiter.check_rate_limit(
                identifier=identifier,
                rate_limit_per_minute=config["rate_limit_per_minute"],
                burst_capacity=config["burst_capacity"],
                window_size_seconds=config["window_size_seconds"],
                bucket_size_seconds=config["bucket_size_seconds"]
            )
            
            # Log rate limit events
            if status.result in [RateLimitResult.RATE_LIMITED, RateLimitResult.BURST_LIMITED]:
                self._rate_limit_blocks += 1
                logger.warning(
                    f"Rate limit exceeded for {agent_id} (tier: {rate_limit_tier}) - "
                    f"Result: {status.result}, Current: {status.current_requests}, "
                    f"Remaining: {status.requests_remaining}, Burst: {status.burst_remaining}"
                )
                raise RateLimitExceeded(
                    f"Rate limit exceeded for tier {rate_limit_tier}",
                    status,
                    rate_limit_tier
                )
            elif status.result == RateLimitResult.ERROR:
                self._rate_limit_errors += 1
                logger.error(f"Rate limiter error for {agent_id}, allowing request (fail-open)")
            
            # Log successful rate limit check
            if status.current_requests % 10 == 0:  # Log every 10th request to avoid spam
                logger.info(
                    f"Rate limit check passed for {agent_id} (tier: {rate_limit_tier}) - "
                    f"Current: {status.current_requests}, Remaining: {status.requests_remaining}"
                )
            
            return status
            
        except RateLimitExceeded:
            raise
        except Exception as e:
            self._rate_limit_errors += 1
            logger.error(f"Unexpected rate limiter error for {agent_id}: {e}")
            # Fail open - return success status when rate limiter fails
            return RateLimitStatus(
                result=RateLimitResult.ERROR,
                requests_remaining=config["rate_limit_per_minute"],
                burst_remaining=config["burst_capacity"],
                reset_time=time.time() + config["window_size_seconds"],
                retry_after=None,
                current_requests=0,
                window_start=time.time() - config["window_size_seconds"]
            )

    def require_rate_limit_check(self, include_ip: bool = False):
        """Decorator to require rate limit checking for MCP tools.
        
        Args:
            include_ip: Whether to include IP address in rate limiting
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract authentication info from kwargs (added by auth middleware)
                auth_payload = kwargs.get("auth_payload")
                agent_id = kwargs.get("agent_id")
                rate_limit_tier = kwargs.get("rate_limit_tier")
                
                if not auth_payload or not agent_id or not rate_limit_tier:
                    logger.error(f"Missing authentication info for rate limiting in {func.__name__}")
                    return {
                        "error": "Authentication required",
                        "message": "Rate limiting requires valid authentication",
                        "timestamp": time.time()
                    }
                
                # Extract IP address if requested
                additional_identifier = None
                if include_ip:
                    context = kwargs.get("context", {})
                    headers = context.get("headers", {})
                    additional_identifier = headers.get("x-forwarded-for") or headers.get("x-real-ip")
                
                try:
                    # Check rate limit
                    rate_limit_status = await self.check_rate_limit(
                        agent_id=agent_id,
                        rate_limit_tier=rate_limit_tier,
                        additional_identifier=additional_identifier
                    )
                    
                    # Add rate limit info to kwargs
                    kwargs["rate_limit_status"] = rate_limit_status
                    kwargs["rate_limit_remaining"] = rate_limit_status.requests_remaining
                    kwargs["rate_limit_reset"] = rate_limit_status.reset_time
                    
                    # Call the original function
                    result = await func(*args, **kwargs)
                    
                    # Add rate limit headers to response if it's a dict
                    if isinstance(result, dict):
                        result["rate_limit"] = {
                            "requests_remaining": rate_limit_status.requests_remaining,
                            "burst_remaining": rate_limit_status.burst_remaining,
                            "reset_time": rate_limit_status.reset_time,
                            "tier": rate_limit_tier,
                            "current_requests": rate_limit_status.current_requests
                        }
                    
                    return result
                    
                except RateLimitExceeded as e:
                    logger.warning(f"Rate limit exceeded in {func.__name__}: {e.message}")
                    return {
                        "error": "Rate limit exceeded",
                        "message": f"Too many requests for {e.tier} tier",
                        "rate_limit": {
                            "requests_remaining": e.status.requests_remaining,
                            "burst_remaining": e.status.burst_remaining,
                            "reset_time": e.status.reset_time,
                            "retry_after": e.status.retry_after,
                            "tier": e.tier,
                            "current_requests": e.status.current_requests
                        },
                        "timestamp": time.time()
                    }
                except Exception as e:
                    logger.error(f"Unexpected rate limiting error in {func.__name__}: {e}")
                    # Continue with request on unexpected errors (fail-open)
                    return await func(*args, **kwargs)
            
            return wrapper
        return decorator

    async def get_rate_limit_info(self, agent_id: str, rate_limit_tier: str) -> Dict[str, Any]:
        """Get current rate limit information for agent.
        
        Args:
            agent_id: Agent identifier
            rate_limit_tier: Rate limit tier
            
        Returns:
            Dict with rate limit information
        """
        config = self.tier_configs.get(rate_limit_tier, self.tier_configs[RateLimitTier.BASIC.value])
        identifier = f"agent:{agent_id}"
        
        info = await self.rate_limiter.get_rate_limit_info(
            identifier=identifier,
            window_size_seconds=config["window_size_seconds"],
            bucket_size_seconds=config["bucket_size_seconds"]
        )
        
        info.update({
            "tier": rate_limit_tier,
            "rate_limit_per_minute": config["rate_limit_per_minute"],
            "burst_capacity": config["burst_capacity"],
            "requests_remaining": max(0, config["rate_limit_per_minute"] - info["current_requests"]),
            "burst_remaining": max(0, config["burst_capacity"] - info["current_requests"])
        })
        
        return info

    async def reset_agent_rate_limit(self, agent_id: str) -> bool:
        """Reset rate limit for specific agent (admin function).
        
        Args:
            agent_id: Agent identifier to reset
            
        Returns:
            True if successful, False otherwise
        """
        identifier = f"agent:{agent_id}"
        return await self.rate_limiter.reset_rate_limit(identifier)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get rate limiting performance metrics.
        
        Returns:
            Dict with performance metrics
        """
        return {
            "total_checks": self._rate_limit_checks,
            "total_blocks": self._rate_limit_blocks,
            "total_errors": self._rate_limit_errors,
            "block_rate": self._rate_limit_blocks / max(1, self._rate_limit_checks),
            "error_rate": self._rate_limit_errors / max(1, self._rate_limit_checks),
            "success_rate": (self._rate_limit_checks - self._rate_limit_blocks - self._rate_limit_errors) / max(1, self._rate_limit_checks)
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check for rate limiting system.
        
        Returns:
            Dict with health status
        """
        try:
            # Test Redis connection
            redis = await self.rate_limiter._get_redis()
            await redis.ping()
            
            return {
                "status": "healthy",
                "redis_connected": True,
                "metrics": self.get_performance_metrics(),
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "redis_connected": False,
                "error": str(e),
                "metrics": self.get_performance_metrics(),
                "timestamp": time.time()
            }

# Global rate limiting middleware instance
_mcp_rate_limit_middleware = None

def get_mcp_rate_limit_middleware() -> MCPRateLimitMiddleware:
    """Get global MCP rate limiting middleware instance.
    
    Returns:
        MCPRateLimitMiddleware instance
    """
    global _mcp_rate_limit_middleware
    if _mcp_rate_limit_middleware is None:
        _mcp_rate_limit_middleware = MCPRateLimitMiddleware()
    return _mcp_rate_limit_middleware

# Convenience decorator for rate limiting
def require_rate_limiting(include_ip: bool = False):
    """Convenience decorator for rate limiting."""
    return get_mcp_rate_limit_middleware().require_rate_limit_check(include_ip)
