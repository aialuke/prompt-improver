"""External API Health Monitoring - 2025 SRE Best Practices
Real connectivity monitoring with response time tracking, rate limit awareness, and circuit breaker integration
"""
import asyncio
import json
import logging
import socket
import ssl
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urlparse
import aiohttp
import dns.exception
import dns.resolver
from cryptography import x509
from cryptography.x509.oid import NameOID
from prompt_improver.core.config import AppConfig, get_config
from prompt_improver.performance.monitoring.health.background_manager import TaskPriority, get_background_task_manager
from prompt_improver.performance.monitoring.health.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, circuit_breaker_registry
logger = logging.getLogger(__name__)

class APIStatus(Enum):
    """API health status levels"""
    HEALTHY = 'healthy'
    DEGRADED = 'degraded'
    UNHEALTHY = 'unhealthy'
    UNKNOWN = 'unknown'

class SLACompliance(Enum):
    """SLA compliance levels"""
    EXCEEDING = 'exceeding'
    MEETING = 'meeting'
    VIOLATED = 'violated'

@dataclass
class APIEndpoint:
    """Configuration for an external API endpoint"""
    name: str
    url: str
    timeout_seconds: float = 10.0
    expected_status_codes: list[int] = field(default_factory=lambda: [200])
    headers: dict[str, str] = field(default_factory=dict)
    auth_header: str | None = None
    check_rate_limits: bool = True
    ssl_verify: bool = True
    dns_check_enabled: bool = True
    p50_target_ms: float = 100.0
    p95_target_ms: float = 500.0
    p99_target_ms: float = 1000.0
    availability_target: float = 0.999
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60

@dataclass
class ResponseMetrics:
    """Response time and status metrics"""
    timestamp: datetime
    response_time_ms: float
    status_code: int
    success: bool
    error: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    rate_limit_remaining: int | None = None
    rate_limit_reset: datetime | None = None
    rate_limit_limit: int | None = None

@dataclass
class DNSMetrics:
    """DNS resolution metrics"""
    timestamp: datetime
    resolution_time_ms: float
    resolved_ips: list[str]
    success: bool
    error: str | None = None

@dataclass
class SSLMetrics:
    """SSL certificate metrics"""
    timestamp: datetime
    certificate_expiry: datetime
    days_until_expiry: int
    issuer: str
    subject: str
    valid: bool
    error: str | None = None

@dataclass
class APIHealthSnapshot:
    """Complete health snapshot for an API"""
    endpoint_name: str
    status: APIStatus
    sla_compliance: SLACompliance
    last_check: datetime
    response_times: dict[str, float]
    availability: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    current_response_time_ms: float | None
    current_status_code: int | None
    current_error: str | None
    rate_limit_status: dict[str, Any] | None
    dns_status: dict[str, Any] | None
    ssl_status: dict[str, Any] | None
    circuit_breaker_state: str
    circuit_breaker_metrics: dict[str, Any]

class RollingWindow:
    """Rolling window for time-series metrics"""

    def __init__(self, window_size_minutes: int=60):
        self.window_size = timedelta(minutes=window_size_minutes)
        self.data: deque = deque()

    def add(self, metric: ResponseMetrics | DNSMetrics | SSLMetrics):
        """Add a metric to the rolling window"""
        now = datetime.now(UTC)
        while self.data and now - self.data[0].timestamp > self.window_size:
            self.data.popleft()
        self.data.append(metric)

    def get_percentiles(self, percentiles: list[float]) -> dict[str, float]:
        """Calculate response time percentiles"""
        response_times = [m.response_time_ms for m in self.data if isinstance(m, ResponseMetrics) and m.success]
        if not response_times:
            return {f'p{int(p * 100)}': 0.0 for p in percentiles}
        response_times.sort()
        n = len(response_times)
        result = {}
        for p in percentiles:
            if p == 0.5:
                idx = n // 2
                result['p50'] = response_times[idx] if n > 0 else 0.0
            elif p == 0.95:
                idx = int(n * 0.95)
                result['p95'] = response_times[min(idx, n - 1)] if n > 0 else 0.0
            elif p == 0.99:
                idx = int(n * 0.99)
                result['p99'] = response_times[min(idx, n - 1)] if n > 0 else 0.0
        return result

    def get_availability(self) -> float:
        """Calculate availability percentage"""
        if not self.data:
            return 1.0
        response_metrics = [m for m in self.data if isinstance(m, ResponseMetrics)]
        if not response_metrics:
            return 1.0
        successful = sum((1 for m in response_metrics if m.success))
        return successful / len(response_metrics)

    def get_success_count(self) -> int:
        """Get count of successful requests"""
        return sum((1 for m in self.data if isinstance(m, ResponseMetrics) and m.success))

    def get_failure_count(self) -> int:
        """Get count of failed requests"""
        return sum((1 for m in self.data if isinstance(m, ResponseMetrics) and (not m.success)))

class ExternalAPIHealthMonitor:
    """Comprehensive external API health monitoring system
    Implements real connectivity tests with SLA tracking and circuit breaker integration
    """

    def __init__(self, config_endpoints: list[APIEndpoint] | None=None):
        self.endpoints = config_endpoints or self._get_default_endpoints()
        self.metrics_windows: dict[str, RollingWindow] = {}
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.dns_resolver = dns.resolver.Resolver()
        self._last_dependency_scan = None
        self._discovered_dependencies: set[str] = set()
        for endpoint in self.endpoints:
            self.metrics_windows[endpoint.name] = RollingWindow()
            if endpoint.circuit_breaker_enabled:
                config = CircuitBreakerConfig(failure_threshold=endpoint.failure_threshold, recovery_timeout=endpoint.recovery_timeout_seconds, response_time_threshold_ms=endpoint.p95_target_ms)
                self.circuit_breakers[endpoint.name] = circuit_breaker_registry.get_or_create(f'external_api_{endpoint.name}', config)

    def _get_default_endpoints(self) -> list[APIEndpoint]:
        """Get default API endpoints from configuration"""
        config = get_config()
        endpoints = []
        endpoints.append(APIEndpoint(name='openai', url='https://api.openai.com/v1/models', timeout_seconds=10.0, expected_status_codes=[200, 401], headers={'User-Agent': 'prompt-improver-health-check/1.0'}, p50_target_ms=200.0, p95_target_ms=800.0, p99_target_ms=2000.0, availability_target=0.995))
        endpoints.append(APIEndpoint(name='huggingface', url='https://huggingface.co/api/models', timeout_seconds=15.0, expected_status_codes=[200], headers={'User-Agent': 'prompt-improver-health-check/1.0'}, p50_target_ms=500.0, p95_target_ms=2000.0, p99_target_ms=5000.0, availability_target=0.99))
        if config.ml.mlflow_tracking_uri:
            mlflow_url = config.ml.mlflow_tracking_uri.rstrip('/') + '/api/2.0/mlflow/experiments/list'
            endpoints.append(APIEndpoint(name='mlflow', url=mlflow_url, timeout_seconds=10.0, expected_status_codes=[200, 401, 403], headers={'User-Agent': 'prompt-improver-health-check/1.0'}, p50_target_ms=300.0, p95_target_ms=1000.0, p99_target_ms=3000.0, availability_target=0.999))
        return endpoints

    async def check_all_endpoints(self) -> dict[str, APIHealthSnapshot]:
        """Perform health checks on all configured endpoints in parallel
        Returns comprehensive health snapshots
        """
        logger.info('Starting health checks for %s external APIs', len(self.endpoints))
        task_manager = get_background_task_manager()
        tasks = []
        task_ids = []
        for endpoint in self.endpoints:
            task_id = await task_manager.submit_enhanced_task(task_id=f'api_health_check_{endpoint.name}_{str(uuid.uuid4())[:8]}', coroutine=self._check_single_endpoint(endpoint), priority=TaskPriority.HIGH, tags={'service': 'monitoring', 'type': 'health_monitoring', 'component': 'external_api_health', 'endpoint': endpoint.name, 'operation': 'endpoint_health_check'})
            task_ids.append(task_id)
            task_status = await task_manager.get_task_status(task_id)
            if task_status and 'task' in task_status:
                tasks.append(task_status['task'])
        results = await asyncio.gather(*tasks, return_exceptions=True)
        health_snapshots = {}
        for endpoint, result in zip(self.endpoints, results, strict=False):
            if isinstance(result, Exception):
                logger.error('Health check failed for {endpoint.name}: %s', result)
                health_snapshots[endpoint.name] = self._create_error_snapshot(endpoint, str(result))
            else:
                health_snapshots[endpoint.name] = result
        await self._store_health_snapshots(health_snapshots)
        logger.info('Completed health checks for %s APIs', len(health_snapshots))
        return health_snapshots

    async def _check_single_endpoint(self, endpoint: APIEndpoint) -> APIHealthSnapshot:
        """Perform comprehensive health check for a single endpoint"""
        start_time = time.time()
        dns_metrics = None
        if endpoint.dns_check_enabled:
            dns_metrics = await self._check_dns(endpoint.url)
        ssl_metrics = None
        if endpoint.url.startswith('https://'):
            ssl_metrics = await self._check_ssl_certificate(endpoint.url)
        response_metrics = None
        if endpoint.circuit_breaker_enabled and endpoint.name in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[endpoint.name]
            try:
                response_metrics = await circuit_breaker.call(self._perform_http_check, endpoint)
            except Exception as e:
                response_metrics = ResponseMetrics(timestamp=datetime.now(UTC), response_time_ms=0.0, status_code=0, success=False, error=str(e))
        else:
            response_metrics = await self._perform_http_check(endpoint)
        window = self.metrics_windows[endpoint.name]
        if response_metrics:
            window.add(response_metrics)
        if dns_metrics:
            window.add(dns_metrics)
        if ssl_metrics:
            window.add(ssl_metrics)
        percentiles = window.get_percentiles([0.5, 0.95, 0.99])
        availability = window.get_availability()
        status = self._determine_api_status(endpoint, response_metrics, availability, percentiles)
        sla_compliance = self._determine_sla_compliance(endpoint, percentiles, availability)
        cb_state = 'disabled'
        cb_metrics = {}
        if endpoint.name in self.circuit_breakers:
            cb = self.circuit_breakers[endpoint.name]
            cb_state = cb.state.value
            cb_metrics = cb.get_metrics()
        snapshot = APIHealthSnapshot(endpoint_name=endpoint.name, status=status, sla_compliance=sla_compliance, last_check=datetime.now(UTC), response_times=percentiles, availability=availability, total_requests=window.get_success_count() + window.get_failure_count(), successful_requests=window.get_success_count(), failed_requests=window.get_failure_count(), current_response_time_ms=response_metrics.response_time_ms if response_metrics else None, current_status_code=response_metrics.status_code if response_metrics else None, current_error=response_metrics.error if response_metrics else None, rate_limit_status=self._extract_rate_limit_info(response_metrics) if response_metrics else None, dns_status=asdict(dns_metrics) if dns_metrics else None, ssl_status=asdict(ssl_metrics) if ssl_metrics else None, circuit_breaker_state=cb_state, circuit_breaker_metrics=cb_metrics)
        total_time = (time.time() - start_time) * 1000
        logger.info('Completed health check for %s in %sms - Status: %s', endpoint.name, format(total_time, '.2f'), status.value, extra={'endpoint': endpoint.name, 'status': status.value, 'response_time_ms': response_metrics.response_time_ms if response_metrics else None, 'availability': availability, 'circuit_breaker_state': cb_state})
        return snapshot

    async def _perform_http_check(self, endpoint: APIEndpoint) -> ResponseMetrics:
        """Perform HTTP connectivity check with response time monitoring"""
        start_time = time.time()
        headers = endpoint.headers.copy()
        if endpoint.auth_header:
            headers['Authorization'] = endpoint.auth_header
        ssl_context = None
        if endpoint.url.startswith('https://'):
            ssl_context = ssl.create_default_context()
            if not endpoint.ssl_verify:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
        timeout = aiohttp.ClientTimeout(total=endpoint.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(ssl=ssl_context)) as session, session.get(endpoint.url, headers=headers, allow_redirects=True) as response:
                response_time_ms = (time.time() - start_time) * 1000
                response_headers = dict(response.headers)
                success = response.status in endpoint.expected_status_codes
                return ResponseMetrics(timestamp=datetime.now(UTC), response_time_ms=response_time_ms, status_code=response.status, success=success, headers=response_headers, rate_limit_remaining=self._parse_rate_limit(response_headers, 'remaining'), rate_limit_reset=self._parse_rate_limit_reset(response_headers), rate_limit_limit=self._parse_rate_limit(response_headers, 'limit'))
        except TimeoutError:
            response_time_ms = (time.time() - start_time) * 1000
            return ResponseMetrics(timestamp=datetime.now(UTC), response_time_ms=response_time_ms, status_code=0, success=False, error=f'Timeout after {endpoint.timeout_seconds}s')
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return ResponseMetrics(timestamp=datetime.now(UTC), response_time_ms=response_time_ms, status_code=0, success=False, error=str(e))

    async def _check_dns(self, url: str) -> DNSMetrics:
        """Check DNS resolution time and IP addresses"""
        start_time = time.time()
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return DNSMetrics(timestamp=datetime.now(UTC), resolution_time_ms=0.0, resolved_ips=[], success=False, error='No hostname found in URL')
        try:
            answers = await asyncio.to_thread(self.dns_resolver.resolve, hostname, 'A')
            resolution_time_ms = (time.time() - start_time) * 1000
            resolved_ips = [str(answer) for answer in answers]
            return DNSMetrics(timestamp=datetime.now(UTC), resolution_time_ms=resolution_time_ms, resolved_ips=resolved_ips, success=True)
        except Exception as e:
            resolution_time_ms = (time.time() - start_time) * 1000
            return DNSMetrics(timestamp=datetime.now(UTC), resolution_time_ms=resolution_time_ms, resolved_ips=[], success=False, error=str(e))

    async def _check_ssl_certificate(self, url: str) -> SSLMetrics:
        """Check SSL certificate validity and expiration"""
        parsed = urlparse(url)
        hostname = parsed.hostname
        port = parsed.port or 443
        if not hostname:
            return SSLMetrics(timestamp=datetime.now(UTC), certificate_expiry=datetime.now(UTC), days_until_expiry=0, issuer='', subject='', valid=False, error='No hostname found in URL')
        try:
            context = ssl.create_default_context()

            def get_cert():
                with socket.create_connection((hostname, port), timeout=10.0) as sock:
                    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                        return ssock.getpeercert(binary_form=True)
            cert_der = await asyncio.to_thread(get_cert)
            cert = x509.load_der_x509_certificate(cert_der)
            expiry = cert.not_valid_after.replace(tzinfo=UTC)
            now = datetime.now(UTC)
            days_until_expiry = (expiry - now).days
            subject = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
            issuer = cert.issuer.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
            valid = now >= cert.not_valid_before.replace(tzinfo=UTC) and now <= expiry and (days_until_expiry > 0)
            return SSLMetrics(timestamp=now, certificate_expiry=expiry, days_until_expiry=days_until_expiry, issuer=issuer, subject=subject, valid=valid)
        except Exception as e:
            return SSLMetrics(timestamp=datetime.now(UTC), certificate_expiry=datetime.now(UTC), days_until_expiry=0, issuer='', subject='', valid=False, error=str(e))

    def _parse_rate_limit(self, headers: dict[str, str], limit_type: str) -> int | None:
        """Parse rate limit information from response headers"""
        patterns = {'remaining': ['x-ratelimit-remaining', 'x-rate-limit-remaining', 'ratelimit-remaining', 'rate-limit-remaining'], 'limit': ['x-ratelimit-limit', 'x-rate-limit-limit', 'ratelimit-limit', 'rate-limit-limit']}
        for header_name in patterns.get(limit_type, []):
            value = headers.get(header_name) or headers.get(header_name.upper())
            if value:
                try:
                    return int(value)
                except ValueError:
                    continue
        return None

    def _parse_rate_limit_reset(self, headers: dict[str, str]) -> datetime | None:
        """Parse rate limit reset time from response headers"""
        reset_patterns = ['x-ratelimit-reset', 'x-rate-limit-reset', 'ratelimit-reset', 'rate-limit-reset']
        for header_name in reset_patterns:
            value = headers.get(header_name) or headers.get(header_name.upper())
            if value:
                try:
                    timestamp = int(value)
                    return datetime.fromtimestamp(timestamp, tz=UTC)
                except ValueError:
                    try:
                        return datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except ValueError:
                        continue
        return None

    def _extract_rate_limit_info(self, metrics: ResponseMetrics) -> dict[str, Any]:
        """Extract rate limit information from response metrics"""
        if not metrics:
            return {}
        info = {}
        if metrics.rate_limit_remaining is not None:
            info['remaining'] = metrics.rate_limit_remaining
        if metrics.rate_limit_limit is not None:
            info['limit'] = metrics.rate_limit_limit
        if metrics.rate_limit_reset is not None:
            info['reset_time'] = metrics.rate_limit_reset.isoformat()
            info['reset_in_seconds'] = (metrics.rate_limit_reset - datetime.now(UTC)).total_seconds()
        if metrics.rate_limit_remaining is not None and metrics.rate_limit_limit is not None:
            used = metrics.rate_limit_limit - metrics.rate_limit_remaining
            info['utilization'] = used / metrics.rate_limit_limit if metrics.rate_limit_limit > 0 else 0.0
        return info

    def _determine_api_status(self, endpoint: APIEndpoint, metrics: ResponseMetrics | None, availability: float, percentiles: dict[str, float]) -> APIStatus:
        """Determine overall API status based on metrics"""
        if not metrics:
            return APIStatus.UNKNOWN
        if not metrics.success:
            return APIStatus.UNHEALTHY
        if availability < 0.95:
            return APIStatus.UNHEALTHY
        if availability < 0.99:
            return APIStatus.DEGRADED
        p95 = percentiles.get('p95', 0.0)
        p99 = percentiles.get('p99', 0.0)
        if p99 > endpoint.p99_target_ms * 2:
            return APIStatus.UNHEALTHY
        if p95 > endpoint.p95_target_ms * 1.5 or p95 > endpoint.p95_target_ms:
            return APIStatus.DEGRADED
        return APIStatus.HEALTHY

    def _determine_sla_compliance(self, endpoint: APIEndpoint, percentiles: dict[str, float], availability: float) -> SLACompliance:
        """Determine SLA compliance level"""
        if availability < endpoint.availability_target:
            return SLACompliance.VIOLATED
        p50 = percentiles.get('p50', 0.0)
        p95 = percentiles.get('p95', 0.0)
        p99 = percentiles.get('p99', 0.0)
        if p50 > endpoint.p50_target_ms or p95 > endpoint.p95_target_ms or p99 > endpoint.p99_target_ms:
            return SLACompliance.VIOLATED
        if p50 < endpoint.p50_target_ms * 0.5 and p95 < endpoint.p95_target_ms * 0.5 and (availability > endpoint.availability_target + 0.001):
            return SLACompliance.EXCEEDING
        return SLACompliance.MEETING

    def _create_error_snapshot(self, endpoint: APIEndpoint, error: str) -> APIHealthSnapshot:
        """Create health snapshot for failed health check"""
        return APIHealthSnapshot(endpoint_name=endpoint.name, status=APIStatus.UNKNOWN, sla_compliance=SLACompliance.VIOLATED, last_check=datetime.now(UTC), response_times={'p50': 0.0, 'p95': 0.0, 'p99': 0.0}, availability=0.0, total_requests=0, successful_requests=0, failed_requests=1, current_response_time_ms=None, current_status_code=None, current_error=error, rate_limit_status=None, dns_status=None, ssl_status=None, circuit_breaker_state='unknown', circuit_breaker_metrics={})

    async def _store_health_snapshots(self, snapshots: dict[str, APIHealthSnapshot]):
        """Store health snapshots in Redis for historical tracking"""
        try:
            timestamp = datetime.now(UTC).isoformat()

            def json_serializer(obj):
                if isinstance(obj, (datetime, timezone)):
                    return obj.isoformat()
                if hasattr(obj, 'value') or isinstance(obj, (APIStatus, SLACompliance)):
                    return obj.value
                return str(obj)
            for name, snapshot in snapshots.items():
                key = f'external_api_health:{name}:{timestamp}'
                snapshot_dict = asdict(snapshot)
                snapshot_dict['status'] = snapshot.status.value
                snapshot_dict['sla_compliance'] = snapshot.sla_compliance.value
                await redis_client.setex(key, 3600 * 24 * 7, json.dumps(snapshot_dict, default=json_serializer))
            summary_key = f'external_api_health:summary:{timestamp}'
            summary = {'timestamp': timestamp, 'total_apis': len(snapshots), 'healthy_apis': sum((1 for s in snapshots.values() if s.status == APIStatus.HEALTHY)), 'degraded_apis': sum((1 for s in snapshots.values() if s.status == APIStatus.DEGRADED)), 'unhealthy_apis': sum((1 for s in snapshots.values() if s.status == APIStatus.UNHEALTHY)), 'sla_compliant': sum((1 for s in snapshots.values() if s.sla_compliance != SLACompliance.VIOLATED))}
            await redis_client.setex(summary_key, 3600 * 24 * 30, json.dumps(summary, default=json_serializer))
        except Exception as e:
            logger.warning('Failed to store health snapshots in Redis: %s', e)

    async def discover_dependencies(self) -> set[str]:
        """Automatically discover external dependencies by scanning code
        Returns set of discovered URLs/domains
        """
        if self._last_dependency_scan and datetime.now(UTC) - self._last_dependency_scan < timedelta(hours=6):
            return self._discovered_dependencies
        logger.info('Starting automatic dependency discovery')
        discovered = set()
        config = get_config()
        for attr in dir(config):
            if attr.endswith('_url') or attr.endswith('_uri') or 'endpoint' in attr.lower():
                value = getattr(config, attr, None)
                if isinstance(value, str) and value.startswith(('http://', 'https://')):
                    discovered.add(value)
        common_domains = ['api.openai.com', 'huggingface.co', 'github.com', 'pypi.org', 'registry.npmjs.org']
        for domain in common_domains:
            discovered.add(f'https://{domain}')
        self._discovered_dependencies = discovered
        self._last_dependency_scan = datetime.now(UTC)
        logger.info('Discovered %s external dependencies', len(discovered))
        return discovered

    async def get_historical_metrics(self, endpoint_name: str, hours_back: int=24) -> list[dict[str, Any]]:
        """Get historical metrics for an endpoint from Redis"""
        try:
            cutoff_time = datetime.now(UTC) - timedelta(hours=hours_back)
            pattern = f'external_api_health:{endpoint_name}:*'
            keys = await redis_client.keys(pattern)
            metrics = []
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                timestamp_str = key_str.split(':')[-1]
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if timestamp >= cutoff_time:
                        data = await redis_client.get(key)
                        if data:
                            metrics.append(json.loads(data))
                except (ValueError, json.JSONDecodeError):
                    continue
            metrics.sort(key=lambda x: x.get('last_check', ''))
            return metrics
        except Exception as e:
            logger.warning('Failed to retrieve historical metrics for %s: %s', endpoint_name, e)
            return []
