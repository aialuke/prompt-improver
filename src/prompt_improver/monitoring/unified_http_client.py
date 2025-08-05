"""
Unified HTTP Client Factory - Phase 3 Session Management Consolidation
Extends ExternalAPIHealthMonitor patterns for standardized HTTP client usage across the codebase
"""

import asyncio
import ssl
import time
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from urllib.parse import urlparse
from enum import Enum

import aiohttp
from ..performance.monitoring.health.background_manager import get_background_task_manager, TaskPriority
from ..performance.monitoring.health.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, circuit_breaker_registry
from .external_api_health import ExternalAPIHealthMonitor, APIEndpoint, ResponseMetrics

logger = logging.getLogger(__name__)

class HTTPClientUsage(Enum):
    """HTTP client usage patterns"""
    WEBHOOK_ALERTS = "webhook_alerts"
    HEALTH_CHECKS = "health_checks"
    API_CALLS = "api_calls"
    DOWNLOADS = "downloads"
    MONITORING = "monitoring"
    TESTING = "testing"

@dataclass
class HTTPClientConfig:
    """Configuration for unified HTTP client"""
    name: str
    usage_type: HTTPClientUsage
    base_url: Optional[str] = None
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_backoff_factor: float = 1.0
    ssl_verify: bool = True
    headers: Dict[str, str] = field(default_factory=dict)
    auth_header: Optional[str] = None
    
    # Circuit breaker configuration
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    response_time_threshold_ms: float = 5000.0
    
    # Monitoring configuration
    collect_metrics: bool = True
    sla_targets: Optional[Dict[str, float]] = None
    rate_limit_aware: bool = True

@dataclass
class HTTPRequestContext:
    """Context for HTTP request tracking"""
    request_id: str
    client_name: str
    method: str
    url: str
    start_time: datetime
    usage_type: HTTPClientUsage
    tags: Dict[str, str] = field(default_factory=dict)

class UnifiedHTTPClientFactory:
    """
    Unified HTTP Client Factory using ExternalAPIHealthMonitor patterns
    Provides standardized HTTP clients with circuit breakers, monitoring, and SLA tracking
    """
    
    def __init__(self):
        self.clients: Dict[str, HTTPClientConfig] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.metrics: Dict[str, List[ResponseMetrics]] = {}
        self.active_sessions: Dict[str, aiohttp.ClientSession] = {}
        self._external_monitor: Optional[ExternalAPIHealthMonitor] = None
        
    def register_client(
        self, 
        config: HTTPClientConfig,
        endpoint_config: Optional[APIEndpoint] = None
    ) -> None:
        """Register a new HTTP client configuration"""
        self.clients[config.name] = config
        
        # Set up circuit breaker
        if config.circuit_breaker_enabled:
            cb_config = CircuitBreakerConfig(
                failure_threshold=config.failure_threshold,
                recovery_timeout=config.recovery_timeout_seconds,
                response_time_threshold_ms=config.response_time_threshold_ms
            )
            self.circuit_breakers[config.name] = circuit_breaker_registry.get_or_create(
                f"http_client_{config.name}", cb_config
            )
        
        # Initialize metrics tracking
        if config.collect_metrics:
            self.metrics[config.name] = []
            
        # Create endpoint for external monitoring if needed
        if endpoint_config and config.collect_metrics:
            if not self._external_monitor:
                self._external_monitor = ExternalAPIHealthMonitor()
            # Add endpoint to monitor for comprehensive health tracking
            
        logger.info(f"Registered HTTP client: {config.name} ({config.usage_type.value})")
    
    @asynccontextmanager
    async def get_session(self, client_name: str):
        """Get a configured HTTP session with circuit breaker and monitoring"""
        if client_name not in self.clients:
            raise ValueError(f"HTTP client '{client_name}' not registered")
        
        config = self.clients[client_name]
        
        # Configure SSL context
        ssl_context = None
        if config.ssl_verify:
            ssl_context = ssl.create_default_context()
        else:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        
        # Configure timeout
        timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
        
        # Configure headers
        headers = config.headers.copy()
        if config.auth_header:
            headers['Authorization'] = config.auth_header
        
        session = None
        try:
            session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
                connector=aiohttp.TCPConnector(ssl=ssl_context)
            )
            
            # Track active session
            self.active_sessions[client_name] = session
            
            yield session
            
        finally:
            if session:
                await session.close()
                if client_name in self.active_sessions:
                    del self.active_sessions[client_name]
    
    async def make_request(
        self,
        client_name: str,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """Make an HTTP request with circuit breaker and monitoring"""
        config = self.clients[client_name]
        request_context = HTTPRequestContext(
            request_id=str(uuid.uuid4())[:8],
            client_name=client_name,
            method=method.upper(),
            url=url,
            start_time=datetime.now(timezone.utc),
            usage_type=config.usage_type,
            tags=kwargs.pop('_tags', {})
        )
        
        # Use circuit breaker if enabled
        if config.circuit_breaker_enabled and client_name in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[client_name]
            try:
                response = await circuit_breaker.call(
                    self._execute_request, config, request_context, method, url, **kwargs
                )
                return response
            except Exception as e:
                await self._record_request_metrics(config, request_context, None, str(e))
                raise
        else:
            return await self._execute_request(config, request_context, method, url, **kwargs)
    
    async def _execute_request(
        self,
        config: HTTPClientConfig,
        context: HTTPRequestContext,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """Execute HTTP request with monitoring"""
        async with self.get_session(config.name) as session:
            start_time = time.time()
            
            try:
                # Add base URL if configured
                full_url = url
                if config.base_url and not url.startswith(('http://', 'https://')):
                    full_url = f"{config.base_url.rstrip('/')}/{url.lstrip('/')}"
                
                # Execute request
                async with session.request(method, full_url, **kwargs) as response:
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    # Record metrics
                    if config.collect_metrics:
                        await self._record_request_metrics(
                            config, context, response, None, response_time_ms
                        )
                    
                    # Log request
                    logger.debug(
                        f"HTTP {method} {full_url} -> {response.status} ({response_time_ms:.1f}ms)",
                        extra={
                            "client_name": config.name,
                            "request_id": context.request_id,
                            "method": method,
                            "url": full_url,
                            "status_code": response.status,
                            "response_time_ms": response_time_ms,
                            "usage_type": config.usage_type.value
                        }
                    )
                    
                    return response
                    
            except Exception as e:
                response_time_ms = (time.time() - start_time) * 1000
                
                # Record error metrics
                if config.collect_metrics:
                    await self._record_request_metrics(
                        config, context, None, str(e), response_time_ms
                    )
                
                logger.error(
                    f"HTTP {method} {url} failed: {e} ({response_time_ms:.1f}ms)",
                    extra={
                        "client_name": config.name,
                        "request_id": context.request_id,
                        "method": method,
                        "url": url,
                        "error": str(e),
                        "response_time_ms": response_time_ms,
                        "usage_type": config.usage_type.value
                    }
                )
                raise
    
    async def _record_request_metrics(
        self,
        config: HTTPClientConfig,
        context: HTTPRequestContext,
        response: Optional[aiohttp.ClientResponse],
        error: Optional[str],
        response_time_ms: Optional[float] = None
    ) -> None:
        """Record request metrics for monitoring and SLA tracking"""
        if not config.collect_metrics:
            return
        
        # Calculate response time if not provided
        if response_time_ms is None:
            response_time_ms = (datetime.now(timezone.utc) - context.start_time).total_seconds() * 1000
        
        # Extract rate limiting headers
        rate_limit_remaining = None
        rate_limit_reset = None
        rate_limit_limit = None
        response_headers = {}
        
        if response:
            response_headers = dict(response.headers)
            rate_limit_remaining = self._parse_rate_limit_header(response_headers, 'remaining')
            rate_limit_limit = self._parse_rate_limit_header(response_headers, 'limit')
            rate_limit_reset = self._parse_rate_limit_reset_header(response_headers)
        
        # Create metrics record
        metrics = ResponseMetrics(
            timestamp=context.start_time,
            response_time_ms=response_time_ms,
            status_code=response.status if response else 0,
            success=(response is not None and response.status < 400 and error is None),
            error=error,
            headers=response_headers,
            rate_limit_remaining=rate_limit_remaining,
            rate_limit_reset=rate_limit_reset,
            rate_limit_limit=rate_limit_limit
        )
        
        # Store metrics
        if config.name not in self.metrics:
            self.metrics[config.name] = []
        
        self.metrics[config.name].append(metrics)
        
        # Keep only recent metrics (last 1000 requests per client)
        if len(self.metrics[config.name]) > 1000:
            self.metrics[config.name] = self.metrics[config.name][-1000:]
    
    def _parse_rate_limit_header(self, headers: Dict[str, str], limit_type: str) -> Optional[int]:
        """Parse rate limit headers (reusing ExternalAPIHealthMonitor logic)"""
        patterns = {
            'remaining': [
                'x-ratelimit-remaining',
                'x-rate-limit-remaining',
                'ratelimit-remaining',
                'rate-limit-remaining'
            ],
            'limit': [
                'x-ratelimit-limit',
                'x-rate-limit-limit',
                'ratelimit-limit',
                'rate-limit-limit'
            ]
        }
        
        for header_name in patterns.get(limit_type, []):
            value = headers.get(header_name) or headers.get(header_name.upper())
            if value:
                try:
                    return int(value)
                except ValueError:
                    continue
        return None
    
    def _parse_rate_limit_reset_header(self, headers: Dict[str, str]) -> Optional[datetime]:
        """Parse rate limit reset headers (reusing ExternalAPIHealthMonitor logic)"""
        reset_patterns = [
            'x-ratelimit-reset',
            'x-rate-limit-reset',
            'ratelimit-reset',
            'rate-limit-reset'
        ]
        
        for header_name in reset_patterns:
            value = headers.get(header_name) or headers.get(header_name.upper())
            if value:
                try:
                    timestamp = int(value)
                    return datetime.fromtimestamp(timestamp, tz=timezone.utc)
                except ValueError:
                    try:
                        return datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except ValueError:
                        continue
        return None
    
    async def get_client_metrics(self, client_name: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a client"""
        if client_name not in self.metrics:
            return {}
        
        client_metrics = self.metrics[client_name]
        if not client_metrics:
            return {}
        
        # Calculate basic statistics
        recent_metrics = client_metrics[-100:]  # Last 100 requests
        response_times = [m.response_time_ms for m in recent_metrics if m.success]
        
        success_count = sum(1 for m in recent_metrics if m.success)
        total_requests = len(recent_metrics)
        
        metrics_data = {
            "client_name": client_name,
            "total_requests": total_requests,
            "successful_requests": success_count,
            "failed_requests": total_requests - success_count,
            "success_rate": success_count / total_requests if total_requests > 0 else 0.0,
            "error_rate": (total_requests - success_count) / total_requests if total_requests > 0 else 0.0
        }
        
        # Calculate response time percentiles
        if response_times:
            response_times.sort()
            n = len(response_times)
            metrics_data["response_times"] = {
                "min": min(response_times),
                "max": max(response_times),
                "mean": sum(response_times) / n,
                "p50": response_times[n // 2] if n > 0 else 0.0,
                "p95": response_times[int(n * 0.95)] if n > 0 else 0.0,
                "p99": response_times[int(n * 0.99)] if n > 0 else 0.0
            }
        
        # Circuit breaker status
        if client_name in self.circuit_breakers:
            cb = self.circuit_breakers[client_name]
            metrics_data["circuit_breaker"] = {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "last_failure_time": cb.last_failure_time.isoformat() if cb.last_failure_time else None,
                "metrics": cb.get_metrics()
            }
        
        return metrics_data
    
    async def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all registered clients"""
        all_metrics = {}
        for client_name in self.clients.keys():
            all_metrics[client_name] = await self.get_client_metrics(client_name)
        
        # Calculate overall statistics
        total_requests = sum(m.get("total_requests", 0) for m in all_metrics.values())
        total_successful = sum(m.get("successful_requests", 0) for m in all_metrics.values())
        
        summary = {
            "total_clients": len(self.clients),
            "total_requests": total_requests,
            "overall_success_rate": total_successful / total_requests if total_requests > 0 else 0.0,
            "clients": all_metrics
        }
        
        return summary
    
    async def health_check_all_clients(self) -> Dict[str, bool]:
        """Perform health check on all registered clients"""
        health_status = {}
        
        for client_name, config in self.clients.items():
            try:
                # Simple health check - if circuit breaker is open, client is unhealthy
                if client_name in self.circuit_breakers:
                    cb = self.circuit_breakers[client_name]
                    health_status[client_name] = cb.state.value != "open"
                else:
                    health_status[client_name] = True
                    
            except Exception as e:
                logger.error(f"Health check failed for client {client_name}: {e}")
                health_status[client_name] = False
        
        return health_status

# Global factory instance
_global_factory: Optional[UnifiedHTTPClientFactory] = None

def get_http_client_factory() -> UnifiedHTTPClientFactory:
    """Get the global HTTP client factory instance"""
    global _global_factory
    if _global_factory is None:
        _global_factory = UnifiedHTTPClientFactory()
        # Register default clients
        _register_default_clients(_global_factory)
    return _global_factory

def _register_default_clients(factory: UnifiedHTTPClientFactory) -> None:
    """Register default HTTP client configurations"""
    
    # Webhook alerts client
    factory.register_client(HTTPClientConfig(
        name="webhook_alerts",
        usage_type=HTTPClientUsage.WEBHOOK_ALERTS,
        timeout_seconds=30.0,
        max_retries=3,
        circuit_breaker_enabled=True,
        failure_threshold=3,
        recovery_timeout_seconds=300,  # 5 minutes
        headers={"Content-Type": "application/json"}
    ))
    
    # Health monitoring client
    factory.register_client(HTTPClientConfig(
        name="health_monitoring",
        usage_type=HTTPClientUsage.HEALTH_CHECKS,
        timeout_seconds=10.0,
        max_retries=2,
        circuit_breaker_enabled=True,
        failure_threshold=5,
        recovery_timeout_seconds=60,
        headers={"User-Agent": "prompt-improver-health-monitor/1.0"}
    ))
    
    # API calls client
    factory.register_client(HTTPClientConfig(
        name="api_calls",
        usage_type=HTTPClientUsage.API_CALLS,
        timeout_seconds=30.0,
        max_retries=3,
        circuit_breaker_enabled=True,
        failure_threshold=5,
        recovery_timeout_seconds=120,
        rate_limit_aware=True
    ))
    
    # Downloads client
    factory.register_client(HTTPClientConfig(
        name="downloads",
        usage_type=HTTPClientUsage.DOWNLOADS,
        timeout_seconds=300.0,  # 5 minutes for downloads
        max_retries=3,
        circuit_breaker_enabled=True,
        failure_threshold=3,
        recovery_timeout_seconds=600  # 10 minutes
    ))
    
    # Testing client
    factory.register_client(HTTPClientConfig(
        name="testing",
        usage_type=HTTPClientUsage.TESTING,
        timeout_seconds=30.0,
        max_retries=1,
        circuit_breaker_enabled=False,  # Disable for testing
        collect_metrics=False
    ))

# Convenience functions for common HTTP operations
async def make_webhook_request(url: str, data: Dict[str, Any], **kwargs) -> aiohttp.ClientResponse:
    """Make a webhook request with standardized configuration"""
    factory = get_http_client_factory()
    return await factory.make_request(
        "webhook_alerts", "POST", url, json=data, **kwargs
    )

async def make_health_check_request(url: str, **kwargs) -> aiohttp.ClientResponse:
    """Make a health check request with standardized configuration"""
    factory = get_http_client_factory()
    return await factory.make_request(
        "health_monitoring", "GET", url, **kwargs
    )

async def make_api_request(method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
    """Make an API request with standardized configuration"""
    factory = get_http_client_factory()
    return await factory.make_request(
        "api_calls", method, url, **kwargs
    )

async def download_file(url: str, **kwargs) -> aiohttp.ClientResponse:
    """Download a file with standardized configuration"""
    factory = get_http_client_factory()
    return await factory.make_request(
        "downloads", "GET", url, **kwargs
    )

@asynccontextmanager
async def get_http_session(client_name: str):
    """Get an HTTP session for advanced usage"""
    factory = get_http_client_factory()
    async with factory.get_session(client_name) as session:
        yield session