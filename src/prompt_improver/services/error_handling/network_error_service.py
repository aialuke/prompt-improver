"""Network Error Service - Specialized Network and HTTP Error Handling.

Provides comprehensive network error handling with:
- HTTP client/server error classification with status code analysis
- Circuit breaker patterns for external API calls and network operations
- Intelligent retry logic with exponential backoff for transient failures
- Security-aware sanitization of HTTP headers, URLs, and response data
- API rate limiting detection and adaptive throttling
- Network connectivity issue handling (DNS, timeouts, connection failures)

Security Features:
- HTTP header sanitization to remove sensitive authentication tokens
- URL parameter redaction for sensitive query strings
- Response payload sanitization for error messages containing PII
- Request/response size monitoring for DoS attack detection
- User-Agent and referrer header validation

Performance Target: <2ms error classification, <5ms circuit breaker operations
Memory Target: <15MB for error tracking and circuit breaker state
"""

import asyncio
import contextlib
import logging
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import aiohttp
import httpx
import requests

from prompt_improver.core.services.resilience.retry_service_facade import (
    get_retry_service as get_retry_manager,
)
from prompt_improver.database.services.health.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState as CBState,
)
from prompt_improver.performance.monitoring.metrics_registry import (
    StandardMetrics,
    get_metrics_registry,
)
from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)


class NetworkErrorCategory(Enum):
    """Network-specific error categories for precise handling."""

    HTTP_CLIENT_ERROR = "http_client_error"  # 4xx status codes
    HTTP_SERVER_ERROR = "http_server_error"  # 5xx status codes
    CONNECTION_TIMEOUT = "connection_timeout"
    DNS_RESOLUTION_FAILURE = "dns_resolution_failure"
    CONNECTION_REFUSED = "connection_refused"
    SSL_CERTIFICATE_ERROR = "ssl_certificate_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    REQUEST_TOO_LARGE = "request_too_large"
    RESPONSE_TOO_LARGE = "response_too_large"
    PROTOCOL_ERROR = "protocol_error"
    PROXY_ERROR = "proxy_error"
    NETWORK_UNREACHABLE = "network_unreachable"
    HOST_UNREACHABLE = "host_unreachable"
    UNKNOWN_NETWORK_ERROR = "unknown_network_error"


class NetworkErrorSeverity(Enum):
    """Error severity levels for network operations."""

    CRITICAL = "critical"  # Service-wide network failure
    HIGH = "high"         # External service degradation
    MEDIUM = "medium"     # Recoverable network issues
    LOW = "low"          # Expected network errors (rate limits, etc.)
    INFO = "info"        # Informational network events


@dataclass
class NetworkErrorContext:
    """Comprehensive error context for network operations."""

    error_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    operation_name: str = ""
    category: NetworkErrorCategory = NetworkErrorCategory.UNKNOWN_NETWORK_ERROR
    severity: NetworkErrorSeverity = NetworkErrorSeverity.MEDIUM
    original_exception: Exception | None = None
    sanitized_message: str = ""
    is_retryable: bool = False
    retry_after_seconds: float | None = None

    # HTTP-specific context
    http_method: str | None = None
    url: str | None = None
    sanitized_url: str | None = None
    status_code: int | None = None
    response_headers: dict[str, Any] = field(default_factory=dict)
    request_headers: dict[str, Any] = field(default_factory=dict)
    response_size_bytes: int | None = None
    request_size_bytes: int | None = None

    # Network-specific context
    host: str | None = None
    port: int | None = None
    ip_address: str | None = None
    dns_resolution_time_ms: float | None = None
    connection_time_ms: float | None = None
    total_request_time_ms: float | None = None

    timestamp: datetime = field(default_factory=aware_utc_now)
    correlation_id: str | None = None
    circuit_breaker_triggered: bool = False


@dataclass
class NetworkCircuitBreakerConfig:
    """Circuit breaker configuration for network operations."""

    failure_threshold: int = 3  # Lower threshold for network failures
    recovery_timeout_seconds: float = 15.0  # Faster recovery for network
    success_threshold: int = 2
    timeout_ms: float = 30000.0  # 30 second timeout for network operations
    slow_call_threshold_ms: float = 10000.0  # 10 second slow call threshold
    enable_half_open_max_calls: int = 2


class NetworkErrorService:
    """Specialized network error handling service.

    Provides comprehensive error handling for network operations including:
    - HTTP status code classification and intelligent response handling
    - Circuit breaker protection for external API calls
    - Security-aware sanitization of network data (URLs, headers, payloads)
    - Adaptive retry logic based on network error patterns
    - API rate limiting detection and throttling management
    """

    # HTTP header patterns for security sanitization
    SENSITIVE_HEADER_PATTERNS = [
        re.compile(r"(authorization|auth|token|key|secret|password)", re.IGNORECASE),
        re.compile(r"(x-api-key|api-key|apikey)", re.IGNORECASE),
        re.compile(r"(cookie|set-cookie)", re.IGNORECASE),
        re.compile(r"(x-auth.*|x-token.*|bearer)", re.IGNORECASE),
    ]

    # URL patterns for sensitive parameter redaction
    URL_SENSITIVE_PATTERNS = [
        (re.compile(r"([?&])(token|key|password|secret|auth)=([^&]*)", re.IGNORECASE), r"\1\2=[REDACTED]"),
        (re.compile(r"([?&])(api[-_]?key)=([^&]*)", re.IGNORECASE), r"\1\2=[REDACTED]"),
        (re.compile(r"([?&])(access[-_]?token)=([^&]*)", re.IGNORECASE), r"\1\2=[REDACTED]"),
        (re.compile(r"://([^:]+):([^@]+)@"), r"://[USER]:[PASS]@"),  # Basic auth in URL
    ]

    # Response body patterns for PII redaction
    RESPONSE_PII_PATTERNS = [
        (re.compile(r'"(email|mail)"\s*:\s*"[^"]*"', re.IGNORECASE), r'"\1":"[EMAIL_REDACTED]"'),
        (re.compile(r'"(phone|tel)"\s*:\s*"[^"]*"', re.IGNORECASE), r'"\1":"[PHONE_REDACTED]"'),
        (re.compile(r'"(ssn|social)"\s*:\s*"[^"]*"', re.IGNORECASE), r'"\1":"[SSN_REDACTED]"'),
        (re.compile(r'"(credit[-_]?card|card[-_]?number)"\s*:\s*"[^"]*"', re.IGNORECASE), r'"\1":"[CARD_REDACTED]"'),
    ]

    # Error mapping for HTTP status codes and network exceptions
    HTTP_STATUS_MAPPINGS = {
        # Client errors (4xx)
        400: (NetworkErrorCategory.HTTP_CLIENT_ERROR, NetworkErrorSeverity.LOW),
        401: (NetworkErrorCategory.HTTP_CLIENT_ERROR, NetworkErrorSeverity.MEDIUM),
        403: (NetworkErrorCategory.HTTP_CLIENT_ERROR, NetworkErrorSeverity.MEDIUM),
        404: (NetworkErrorCategory.HTTP_CLIENT_ERROR, NetworkErrorSeverity.LOW),
        408: (NetworkErrorCategory.CONNECTION_TIMEOUT, NetworkErrorSeverity.MEDIUM),
        413: (NetworkErrorCategory.REQUEST_TOO_LARGE, NetworkErrorSeverity.LOW),
        429: (NetworkErrorCategory.RATE_LIMIT_EXCEEDED, NetworkErrorSeverity.LOW),

        # Server errors (5xx)
        500: (NetworkErrorCategory.HTTP_SERVER_ERROR, NetworkErrorSeverity.HIGH),
        502: (NetworkErrorCategory.HTTP_SERVER_ERROR, NetworkErrorSeverity.HIGH),
        503: (NetworkErrorCategory.HTTP_SERVER_ERROR, NetworkErrorSeverity.HIGH),
        504: (NetworkErrorCategory.CONNECTION_TIMEOUT, NetworkErrorSeverity.HIGH),
    }

    NETWORK_EXCEPTION_MAPPINGS = {
        # aiohttp exceptions
        aiohttp.ClientConnectionError: (NetworkErrorCategory.CONNECTION_REFUSED, NetworkErrorSeverity.HIGH),
        aiohttp.ClientTimeout: (NetworkErrorCategory.CONNECTION_TIMEOUT, NetworkErrorSeverity.MEDIUM),
        aiohttp.ClientSSLError: (NetworkErrorCategory.SSL_CERTIFICATE_ERROR, NetworkErrorSeverity.HIGH),
        aiohttp.ClientProxyConnectionError: (NetworkErrorCategory.PROXY_ERROR, NetworkErrorSeverity.MEDIUM),
        aiohttp.ClientPayloadError: (NetworkErrorCategory.PROTOCOL_ERROR, NetworkErrorSeverity.MEDIUM),

        # httpx exceptions
        httpx.ConnectTimeout: (NetworkErrorCategory.CONNECTION_TIMEOUT, NetworkErrorSeverity.MEDIUM),
        httpx.ReadTimeout: (NetworkErrorCategory.CONNECTION_TIMEOUT, NetworkErrorSeverity.MEDIUM),
        httpx.ConnectError: (NetworkErrorCategory.CONNECTION_REFUSED, NetworkErrorSeverity.HIGH),
        httpx.HTTPStatusError: (NetworkErrorCategory.HTTP_SERVER_ERROR, NetworkErrorSeverity.HIGH),

        # requests exceptions
        requests.exceptions.ConnectionError: (NetworkErrorCategory.CONNECTION_REFUSED, NetworkErrorSeverity.HIGH),
        requests.exceptions.Timeout: (NetworkErrorCategory.CONNECTION_TIMEOUT, NetworkErrorSeverity.MEDIUM),
        requests.exceptions.SSLError: (NetworkErrorCategory.SSL_CERTIFICATE_ERROR, NetworkErrorSeverity.HIGH),
        requests.exceptions.ProxyError: (NetworkErrorCategory.PROXY_ERROR, NetworkErrorSeverity.MEDIUM),
        requests.exceptions.HTTPError: (NetworkErrorCategory.HTTP_SERVER_ERROR, NetworkErrorSeverity.HIGH),

        # Built-in exceptions
        ConnectionRefusedError: (NetworkErrorCategory.CONNECTION_REFUSED, NetworkErrorSeverity.HIGH),
        ConnectionAbortedError: (NetworkErrorCategory.CONNECTION_REFUSED, NetworkErrorSeverity.MEDIUM),
        ConnectionResetError: (NetworkErrorCategory.CONNECTION_REFUSED, NetworkErrorSeverity.MEDIUM),
        OSError: (NetworkErrorCategory.NETWORK_UNREACHABLE, NetworkErrorSeverity.HIGH),
        asyncio.TimeoutError: (NetworkErrorCategory.CONNECTION_TIMEOUT, NetworkErrorSeverity.MEDIUM),
    }

    def __init__(self, correlation_id: str | None = None) -> None:
        """Initialize network error service.

        Args:
            correlation_id: Optional correlation ID for request tracking
        """
        self.correlation_id = correlation_id or str(uuid.uuid4())[:8]
        self._metrics_registry = get_metrics_registry()
        self._retry_manager = get_retry_manager()
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._circuit_breaker_configs: dict[str, NetworkCircuitBreakerConfig] = {}
        self._rate_limit_cache: dict[str, dict[str, Any]] = {}

        # Initialize default circuit breaker configuration
        self._default_cb_config = NetworkCircuitBreakerConfig()

        logger.info(f"NetworkErrorService initialized with correlation_id: {self.correlation_id}")

    async def handle_network_error(
        self,
        error: Exception,
        operation_name: str,
        url: str | None = None,
        method: str | None = None,
        status_code: int | None = None,
        response_headers: dict[str, str] | None = None,
        request_headers: dict[str, str] | None = None,
        response_body: str | None = None,
        **context_kwargs: Any
    ) -> NetworkErrorContext:
        """Handle network error with comprehensive processing.

        Args:
            error: The network exception that occurred
            operation_name: Name of the network operation
            url: Request URL
            method: HTTP method
            status_code: HTTP response status code
            response_headers: HTTP response headers
            request_headers: HTTP request headers
            response_body: HTTP response body
            **context_kwargs: Additional context information

        Returns:
            NetworkErrorContext with processed error information
        """
        start_time = time.time()

        # Classify the error
        category, severity = self._classify_network_error(error, status_code)

        # Create error context
        error_context = NetworkErrorContext(
            operation_name=operation_name,
            category=category,
            severity=severity,
            original_exception=error,
            sanitized_message=self._sanitize_error_message(str(error)),
            is_retryable=self._is_retryable_network_error(category, severity, status_code),
            http_method=method,
            url=url,
            sanitized_url=self._sanitize_url(url) if url else None,
            status_code=status_code,
            response_headers=self._sanitize_headers(response_headers or {}),
            request_headers=self._sanitize_headers(request_headers or {}),
            correlation_id=self.correlation_id,
            **context_kwargs
        )

        # Extract additional network context
        if url:
            parsed_url = urlparse(url)
            error_context.host = parsed_url.hostname
            error_context.port = parsed_url.port

        # Handle rate limiting
        if category == NetworkErrorCategory.RATE_LIMIT_EXCEEDED:
            await self._handle_rate_limit(error_context, response_headers or {})

        # Check for security threats
        await self._check_network_security_threats(error_context, response_body)

        # Update circuit breaker
        circuit_breaker = await self._get_or_create_circuit_breaker(operation_name, error_context.host)
        if circuit_breaker:
            error_context.circuit_breaker_triggered = await self._update_circuit_breaker(
                circuit_breaker, False, error_context
            )

        # Record metrics
        await self._record_network_error_metrics(error_context)

        # Log the error
        await self._log_network_error(error_context)

        # Calculate processing time
        processing_time = time.time() - start_time
        self._metrics_registry.record_value(
            "network_error_processing_duration_seconds",
            processing_time,
            tags={"operation": operation_name, "category": category.value}
        )

        return error_context

    async def execute_with_circuit_breaker(
        self,
        operation: Callable[..., Any],
        operation_name: str,
        host: str | None = None,
        *args,
        **kwargs
    ) -> Any:
        """Execute network operation with circuit breaker protection.

        Args:
            operation: Network operation to execute
            operation_name: Name of the operation for monitoring
            host: Target host for circuit breaker scoping
            *args, **kwargs: Operation arguments

        Returns:
            Operation result

        Raises:
            Exception: If operation fails or circuit breaker is open
        """
        circuit_breaker = await self._get_or_create_circuit_breaker(operation_name, host)

        if not circuit_breaker:
            # Fallback to direct execution if circuit breaker unavailable
            return await operation(*args, **kwargs)

        # Check circuit breaker state
        if circuit_breaker.state == CBState.OPEN:
            error_msg = f"Circuit breaker OPEN for network operation: {operation_name}"
            if host:
                error_msg += f" (host: {host})"
            logger.warning(error_msg)

            self._metrics_registry.increment(
                StandardMetrics.CIRCUIT_BREAKER_CALLS_TOTAL,
                tags={"operation": operation_name, "host": host or "unknown", "state": "open", "result": "rejected"}
            )
            raise RuntimeError(error_msg)

        start_time = time.time()
        try:
            # Execute operation
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)

            # Record success
            execution_time = time.time() - start_time
            await self._update_circuit_breaker(circuit_breaker, True, None, execution_time)

            self._metrics_registry.increment(
                StandardMetrics.CIRCUIT_BREAKER_CALLS_TOTAL,
                tags={"operation": operation_name, "host": host or "unknown", "state": circuit_breaker.state.value, "result": "success"}
            )

            return result

        except Exception as error:
            # Handle failure
            execution_time = time.time() - start_time
            error_context = await self.handle_network_error(
                error,
                operation_name,
                execution_time=execution_time
            )

            await self._update_circuit_breaker(circuit_breaker, False, error_context, execution_time)

            self._metrics_registry.increment(
                StandardMetrics.CIRCUIT_BREAKER_CALLS_TOTAL,
                tags={"operation": operation_name, "host": host or "unknown", "state": circuit_breaker.state.value, "result": "failure"}
            )

            raise

    def _classify_network_error(
        self,
        error: Exception,
        status_code: int | None = None
    ) -> tuple[NetworkErrorCategory, NetworkErrorSeverity]:
        """Classify network error into category and severity.

        Args:
            error: Network exception
            status_code: HTTP status code if applicable

        Returns:
            Tuple of (category, severity)
        """
        # HTTP status code classification first
        if status_code and status_code in self.HTTP_STATUS_MAPPINGS:
            return self.HTTP_STATUS_MAPPINGS[status_code]

        # General status code ranges
        if status_code:
            if 400 <= status_code < 500:
                return (NetworkErrorCategory.HTTP_CLIENT_ERROR, NetworkErrorSeverity.LOW)
            if 500 <= status_code < 600:
                return (NetworkErrorCategory.HTTP_SERVER_ERROR, NetworkErrorSeverity.HIGH)

        # Exception type mapping
        error_type = type(error)
        if error_type in self.NETWORK_EXCEPTION_MAPPINGS:
            return self.NETWORK_EXCEPTION_MAPPINGS[error_type]

        # Check inheritance hierarchy
        for mapped_type, (category, severity) in self.NETWORK_EXCEPTION_MAPPINGS.items():
            if isinstance(error, mapped_type):
                return (category, severity)

        # Pattern-based classification for unknown errors
        error_msg = str(error).lower()

        if any(pattern in error_msg for pattern in ["dns", "name resolution", "hostname"]):
            return (NetworkErrorCategory.DNS_RESOLUTION_FAILURE, NetworkErrorSeverity.HIGH)

        if any(pattern in error_msg for pattern in ["ssl", "certificate", "tls"]):
            return (NetworkErrorCategory.SSL_CERTIFICATE_ERROR, NetworkErrorSeverity.HIGH)

        if any(pattern in error_msg for pattern in ["proxy", "tunnel"]):
            return (NetworkErrorCategory.PROXY_ERROR, NetworkErrorSeverity.MEDIUM)

        if any(pattern in error_msg for pattern in ["too large", "payload", "size"]):
            return (NetworkErrorCategory.REQUEST_TOO_LARGE, NetworkErrorSeverity.LOW)

        return (NetworkErrorCategory.UNKNOWN_NETWORK_ERROR, NetworkErrorSeverity.MEDIUM)

    def _is_retryable_network_error(
        self,
        category: NetworkErrorCategory,
        severity: NetworkErrorSeverity,
        status_code: int | None = None
    ) -> bool:
        """Determine if network error is retryable.

        Args:
            category: Error category
            severity: Error severity
            status_code: HTTP status code

        Returns:
            True if error should be retried
        """
        # Non-retryable categories
        non_retryable_categories = {
            NetworkErrorCategory.HTTP_CLIENT_ERROR,  # 4xx errors generally not retryable
            NetworkErrorCategory.SSL_CERTIFICATE_ERROR,
            NetworkErrorCategory.REQUEST_TOO_LARGE,
        }

        # Non-retryable HTTP status codes
        non_retryable_status_codes = {400, 401, 403, 404, 405, 406, 409, 410, 413, 414, 415, 422}

        if status_code and status_code in non_retryable_status_codes:
            return False

        if category in non_retryable_categories:
            # Exception: 408 Request Timeout is retryable
            return status_code == 408

        # Retryable categories
        retryable_categories = {
            NetworkErrorCategory.HTTP_SERVER_ERROR,  # 5xx errors
            NetworkErrorCategory.CONNECTION_TIMEOUT,
            NetworkErrorCategory.DNS_RESOLUTION_FAILURE,
            NetworkErrorCategory.CONNECTION_REFUSED,
            NetworkErrorCategory.RATE_LIMIT_EXCEEDED,
            NetworkErrorCategory.NETWORK_UNREACHABLE,
            NetworkErrorCategory.HOST_UNREACHABLE,
        }

        return category in retryable_categories

    def _sanitize_url(self, url: str) -> str:
        """Sanitize URL to remove sensitive parameters and credentials.

        Args:
            url: Raw URL

        Returns:
            Sanitized URL
        """
        sanitized = url

        # Apply all URL sensitive patterns
        for pattern, replacement in self.URL_SENSITIVE_PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)

        return sanitized

    def _sanitize_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Sanitize HTTP headers to remove sensitive information.

        Args:
            headers: Raw HTTP headers

        Returns:
            Sanitized headers dictionary
        """
        sanitized_headers = {}

        for key, value in headers.items():
            # Check if header is sensitive
            is_sensitive = any(pattern.search(key) for pattern in self.SENSITIVE_HEADER_PATTERNS)

            if is_sensitive:
                sanitized_headers[key] = "[REDACTED]"
            # Still redact very long header values that might contain tokens
            elif len(str(value)) > 100:
                sanitized_headers[key] = f"{str(value)[:50]}...[TRUNCATED]"
            else:
                sanitized_headers[key] = value

        return sanitized_headers

    def _sanitize_error_message(self, error_message: str) -> str:
        """Sanitize error message to remove sensitive information.

        Args:
            error_message: Raw error message

        Returns:
            Sanitized error message
        """
        sanitized = error_message

        # Apply URL sanitization to any URLs in error messages
        for pattern, replacement in self.URL_SENSITIVE_PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)

        # Remove potential API keys or tokens from error messages
        sanitized = re.sub(r'\b[A-Za-z0-9_-]{32,}\b', '[TOKEN_REDACTED]', sanitized)

        # Remove IP addresses
        return re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP_REDACTED]', sanitized)

    async def _handle_rate_limit(
        self,
        error_context: NetworkErrorContext,
        response_headers: dict[str, str]
    ) -> None:
        """Handle rate limiting with adaptive backoff.

        Args:
            error_context: Network error context
            response_headers: HTTP response headers
        """
        # Extract rate limit information from headers
        retry_after = None
        rate_limit_reset = None

        # Common rate limit headers
        for header_name, header_value in response_headers.items():
            header_lower = header_name.lower()
            if header_lower in {'retry-after', 'x-retry-after'}:
                with contextlib.suppress(ValueError, TypeError):
                    retry_after = float(header_value)
            elif header_lower in {'x-ratelimit-reset', 'x-rate-limit-reset'}:
                with contextlib.suppress(ValueError, TypeError):
                    rate_limit_reset = int(header_value)

        # Calculate retry delay
        if retry_after:
            error_context.retry_after_seconds = retry_after
        elif rate_limit_reset:
            # Calculate seconds until reset
            current_timestamp = int(time.time())
            error_context.retry_after_seconds = max(0, rate_limit_reset - current_timestamp)
        else:
            # Default exponential backoff for rate limits
            error_context.retry_after_seconds = min(60.0, 2.0 ** (
                self._rate_limit_cache.get(error_context.host or 'default', {}).get('consecutive_limits', 0)
            ))

        # Update rate limit cache
        host_key = error_context.host or 'default'
        if host_key not in self._rate_limit_cache:
            self._rate_limit_cache[host_key] = {}

        self._rate_limit_cache[host_key]['consecutive_limits'] = (
            self._rate_limit_cache[host_key].get('consecutive_limits', 0) + 1
        )
        self._rate_limit_cache[host_key]['last_limit_time'] = time.time()

        logger.info(
            f"Rate limit detected for {error_context.operation_name} "
            f"(host: {error_context.host}). Retry after: {error_context.retry_after_seconds}s",
            extra={"correlation_id": self.correlation_id}
        )

    async def _check_network_security_threats(
        self,
        error_context: NetworkErrorContext,
        response_body: str | None
    ) -> None:
        """Check for potential security threats in network errors.

        Args:
            error_context: Error context to analyze
            response_body: HTTP response body to check
        """
        threat_detected = False
        threat_types = []

        # Check for suspicious error patterns that might indicate attacks
        error_msg = str(error_context.original_exception)

        # SQL injection attempts in network requests
        if re.search(r"(?i)(union|select|insert|update|delete|drop).*from", error_msg):
            threat_detected = True
            threat_types.append("sql_injection_in_network_request")

        # XSS attempts
        if re.search(r"(?i)<script|javascript:|vbscript:|onload=|onerror=", error_msg):
            threat_detected = True
            threat_types.append("xss_attempt")

        # Command injection attempts
        if re.search(r"(?i)(;|\||&|\$\(|`)(rm|cat|ls|wget|curl|nc|bash)", error_msg):
            threat_detected = True
            threat_types.append("command_injection_attempt")

        # Check response body for sensitive data leaks
        if response_body and len(response_body) < 10000:  # Only check reasonably sized responses
            for pattern, _ in self.RESPONSE_PII_PATTERNS:
                if pattern.search(response_body):
                    threat_detected = True
                    threat_types.append("pii_in_error_response")
                    break

        if threat_detected:
            logger.warning(
                f"NETWORK SECURITY ALERT: Potential threats detected in network error for operation {error_context.operation_name}: {threat_types}",
                extra={
                    "correlation_id": self.correlation_id,
                    "threat_types": threat_types,
                    "error_id": error_context.error_id,
                    "host": error_context.host
                }
            )

            self._metrics_registry.increment(
                "network_security_threats_total",
                tags={
                    "operation": error_context.operation_name,
                    "host": error_context.host or "unknown",
                    "threat_type": ",".join(threat_types)
                }
            )

    async def _get_or_create_circuit_breaker(
        self,
        operation_name: str,
        host: str | None = None
    ) -> CircuitBreaker | None:
        """Get or create circuit breaker for network operation.

        Args:
            operation_name: Name of the network operation
            host: Target host for scoped circuit breakers

        Returns:
            Circuit breaker instance or None if creation fails
        """
        # Create unique key for operation + host combination
        cb_key = f"{operation_name}:{host}" if host else operation_name

        if cb_key not in self._circuit_breakers:
            try:
                config = self._circuit_breaker_configs.get(cb_key, self._default_cb_config)

                cb_config = CircuitBreakerConfig(
                    failure_threshold=config.failure_threshold,
                    recovery_timeout_seconds=config.recovery_timeout_seconds,
                    success_threshold=config.success_threshold,
                    timeout_ms=config.timeout_ms
                )

                self._circuit_breakers[cb_key] = CircuitBreaker(
                    name=f"net_{cb_key}",
                    config=cb_config
                )

                logger.info(f"Created circuit breaker for network operation: {cb_key}")

            except Exception as e:
                logger.exception(f"Failed to create circuit breaker for {cb_key}: {e}")
                return None

        return self._circuit_breakers.get(cb_key)

    async def _update_circuit_breaker(
        self,
        circuit_breaker: CircuitBreaker,
        success: bool,
        error_context: NetworkErrorContext | None,
        execution_time: float | None = None
    ) -> bool:
        """Update circuit breaker state based on operation result.

        Args:
            circuit_breaker: Circuit breaker to update
            success: Whether operation was successful
            error_context: Error context if operation failed
            execution_time: Operation execution time in seconds

        Returns:
            True if circuit breaker was triggered (state changed)
        """
        previous_state = circuit_breaker.state

        try:
            if success:
                circuit_breaker.record_success(execution_time)
            else:
                circuit_breaker.record_failure()

            current_state = circuit_breaker.state
            state_changed = previous_state != current_state

            if state_changed:
                logger.info(
                    f"Network circuit breaker state changed for {circuit_breaker.name}: "
                    f"{previous_state.value} -> {current_state.value}"
                )

                self._metrics_registry.increment(
                    StandardMetrics.CIRCUIT_BREAKER_STATE_TRANSITIONS,
                    tags={
                        "name": circuit_breaker.name,
                        "from_state": previous_state.value,
                        "to_state": current_state.value,
                        "type": "network"
                    }
                )

            return state_changed

        except Exception as e:
            logger.exception(f"Failed to update network circuit breaker {circuit_breaker.name}: {e}")
            return False

    async def _record_network_error_metrics(self, error_context: NetworkErrorContext) -> None:
        """Record network error metrics for monitoring.

        Args:
            error_context: Error context with metrics data
        """
        try:
            self._metrics_registry.increment(
                "network_errors_total",
                tags={
                    "operation": error_context.operation_name,
                    "category": error_context.category.value,
                    "severity": error_context.severity.value,
                    "retryable": str(error_context.is_retryable).lower(),
                    "status_code": str(error_context.status_code) if error_context.status_code else "none",
                    "host": error_context.host or "unknown"
                }
            )

            # Record response time metrics if available
            if error_context.total_request_time_ms:
                self._metrics_registry.record_value(
                    "network_request_duration_ms",
                    error_context.total_request_time_ms,
                    tags={
                        "operation": error_context.operation_name,
                        "host": error_context.host or "unknown",
                        "result": "failure"
                    }
                )

        except Exception as e:
            logger.exception(f"Failed to record network error metrics: {e}")

    async def _log_network_error(self, error_context: NetworkErrorContext) -> None:
        """Log network error with appropriate level and context.

        Args:
            error_context: Error context to log
        """
        log_data = {
            "correlation_id": self.correlation_id,
            "error_id": error_context.error_id,
            "operation": error_context.operation_name,
            "category": error_context.category.value,
            "severity": error_context.severity.value,
            "retryable": error_context.is_retryable,
            "status_code": error_context.status_code,
            "host": error_context.host,
            "method": error_context.http_method,
            "circuit_breaker_triggered": error_context.circuit_breaker_triggered,
        }

        log_message = (
            f"Network error in {error_context.operation_name}: "
            f"{error_context.sanitized_message}"
        )

        if error_context.sanitized_url:
            log_message += f" (URL: {error_context.sanitized_url})"

        if error_context.severity == NetworkErrorSeverity.CRITICAL:
            logger.critical(log_message, extra=log_data)
        elif error_context.severity == NetworkErrorSeverity.HIGH:
            logger.error(log_message, extra=log_data)
        elif error_context.severity == NetworkErrorSeverity.MEDIUM:
            logger.warning(log_message, extra=log_data)
        else:
            logger.info(log_message, extra=log_data)

    def get_network_error_statistics(self) -> dict[str, Any]:
        """Get network error statistics for monitoring dashboard.

        Returns:
            Dictionary with network error statistics
        """
        circuit_breaker_states = {}
        for name, cb in self._circuit_breakers.items():
            circuit_breaker_states[name] = {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "success_count": getattr(cb, 'success_count', 0),
                "last_failure_time": getattr(cb, 'last_failure_time', None),
            }

        rate_limit_info = {}
        for host, info in self._rate_limit_cache.items():
            rate_limit_info[host] = {
                "consecutive_limits": info.get('consecutive_limits', 0),
                "last_limit_time": info.get('last_limit_time'),
            }

        return {
            "correlation_id": self.correlation_id,
            "active_circuit_breakers": len(self._circuit_breakers),
            "circuit_breaker_states": circuit_breaker_states,
            "rate_limit_info": rate_limit_info,
            "service_health": "operational",
        }
