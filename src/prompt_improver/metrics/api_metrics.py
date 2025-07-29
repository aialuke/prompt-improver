"""
API Usage Business Metrics Collectors for Prompt Improver.

Tracks endpoint popularity, user journey analytics, rate limiting effectiveness,
and authentication patterns with real-time aggregation and conversion tracking.
"""

from __future__ import annotations

import asyncio
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, DefaultDict, Deque, Protocol

# Use consolidated common utilities
from ..core.common import ConfigMixin, MetricsMixin, get_logger

class EndpointCategory(Enum):
    """Categories of API endpoints."""
    PROMPT_IMPROVEMENT = "prompt_improvement"
    ML_ANALYTICS = "ml_analytics"
    USER_MANAGEMENT = "user_management"
    HEALTH_CHECK = "health_check"
    AUTHENTICATION = "authentication"
    REAL_TIME = "real_time"
    BATCH_PROCESSING = "batch_processing"
    CONFIGURATION = "configuration"
    ML_TRAINING = "ml_training"
    RULE_APPLICATION = "rule_application"

class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

class UserJourneyStage(Enum):
    """Stages in user journey."""
    ONBOARDING = "onboarding"
    FIRST_USE = "first_use"
    REGULAR_USE = "regular_use"
    ADVANCED_USE = "advanced_use"
    CHURNED = "churned"
    REACTIVATED = "reactivated"
    ML_TRAINING = "ml_training"

class AuthenticationMethod(Enum):
    """Authentication methods."""
    SIMPLIFIED_AUTH = "simplified_auth"
    API_KEY = "api_key"
    OAUTH = "oauth"
    SESSION_COOKIE = "session_cookie"
    BASIC_AUTH = "basic_auth"
    ANONYMOUS = "anonymous"
    LOCAL_AUTH = "local_auth"

@dataclass
class APIUsageMetric:
    """Metrics for API endpoint usage."""
    endpoint: str
    method: HTTPMethod
    category: EndpointCategory
    status_code: int
    response_time_ms: float
    request_size_bytes: int
    response_size_bytes: int
    user_id: str | None
    session_id: str | None
    ip_address: str
    user_agent: str | None
    timestamp: datetime
    query_parameters_count: int
    payload_type: str | None
    rate_limited: bool
    cache_hit: bool
    authentication_method: AuthenticationMethod
    api_version: str | None

@dataclass
class UserJourneyMetric:
    """Metrics for user journey tracking."""
    user_id: str
    session_id: str
    journey_stage: UserJourneyStage
    event_type: str
    endpoint: str
    success: bool
    conversion_value: float | None
    time_to_action_seconds: float | None
    previous_stage: UserJourneyStage | None
    feature_flags_active: list[str]
    cohort_id: str | None
    timestamp: datetime
    metadata: dict[str, Any]

@dataclass
class RateLimitMetric:
    """Metrics for rate limiting effectiveness."""
    user_id: str | None
    ip_address: str
    endpoint: str
    limit_type: str  # "user", "ip", "endpoint"
    limit_value: int
    current_usage: int
    time_window_seconds: int
    blocked: bool
    burst_detected: bool
    timestamp: datetime
    user_tier: str | None
    override_applied: bool

@dataclass
class AuthenticationMetric:
    """Metrics for authentication operations."""
    user_id: str | None
    authentication_method: AuthenticationMethod
    success: bool
    failure_reason: str | None
    ip_address: str
    user_agent: str | None
    session_duration_seconds: float | None
    mfa_used: bool
    token_type: str | None
    timestamp: datetime
    geo_location: str | None
    device_fingerprint: str | None

# Define metric protocol for type safety
class MetricProtocol(Protocol):
    """Protocol for metric objects with common methods."""

    def inc(self, *args: Any, **kwargs: Any) -> None:
        """Increment metric."""
        ...

    def set(self, *args: Any, **kwargs: Any) -> None:
        """Set metric value."""
        ...

    def observe(self, *args: Any, **kwargs: Any) -> None:
        """Observe metric value."""
        ...

    def labels(self, *args: Any, **kwargs: Any) -> MetricProtocol:
        """Get labeled metric."""
        ...


class APIMetricsCollector(MetricsMixin, ConfigMixin):
    """
    Collects and aggregates API usage business metrics.

    Provides real-time tracking of endpoint popularity, user behavior patterns,
    and system effectiveness metrics with conversion and journey analytics.

    Uses consolidated MetricsMixin and ConfigMixin for common patterns.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize API metrics collector."""
        # Initialize mixins properly
        MetricsMixin.__init__(self)
        ConfigMixin.__init__(self)

        # Use provided config or fall back to centralized config
        self.local_config = config or {}

        # Initialize logger using consolidated utility
        self.logger = get_logger(__name__)

        # Metrics storage with size limits - properly typed deques
        self.api_usage_metrics: Deque[APIUsageMetric] = deque(maxlen=self.local_config.get("max_api_metrics", 20000))
        self.journey_metrics: Deque[UserJourneyMetric] = deque(maxlen=self.local_config.get("max_journey_metrics", 10000))
        self.rate_limit_metrics: Deque[RateLimitMetric] = deque(maxlen=self.local_config.get("max_rate_limit_metrics", 5000))
        self.auth_metrics: Deque[AuthenticationMetric] = deque(maxlen=self.local_config.get("max_auth_metrics", 10000))

        # Real-time tracking
        self.active_sessions: dict[str, dict[str, Any]] = {}
        self.user_journey_states: dict[str, UserJourneyStage] = {}
        self.endpoint_popularity_cache: DefaultDict[str, int] = defaultdict(int)

        # Configuration with fallbacks to centralized config
        self.aggregation_window_minutes = self.local_config.get("aggregation_window_minutes", 5)
        self.retention_hours = self.local_config.get("retention_hours", 72)
        self.journey_timeout_minutes = self.local_config.get("journey_timeout_minutes", 30)

        # Collection statistics - properly typed
        self.collection_stats: dict[str, Any] = {
            "api_calls_tracked": 0,
            "journey_events_tracked": 0,
            "rate_limit_events_tracked": 0,
            "auth_events_tracked": 0,
            "active_sessions_count": 0,
            "last_aggregation": None
        }

        # Initialize Prometheus metrics using MetricsMixin
        self._initialize_prometheus_metrics()

        # Background processing
        self.is_running = False
        self.aggregation_task: asyncio.Task[None] | None = None
        self.session_cleanup_task: asyncio.Task[None] | None = None

    def _initialize_prometheus_metrics(self) -> None:
        """Initialize real metrics for API operations."""
        # Always use real metrics - either Prometheus or in-memory fallback
        # The metrics registry handles the fallback automatically

        # Ensure we have a metrics registry (property will create one if needed)
        registry = self.metrics_registry
        if registry is None:
            # Force creation of a new registry
            from ..performance.monitoring.metrics_registry import get_metrics_registry
            self._metrics_registry = get_metrics_registry()
            self._metrics_available = True
            registry = self._metrics_registry
            self.logger.info("Created new metrics registry for API metrics")

        self.logger.info("Initializing real metrics (Prometheus or in-memory fallback)")

        # API endpoint metrics - using real metrics (Prometheus or in-memory)
        self.endpoint_request_count = registry.get_or_create_counter(
            "api_endpoint_requests_total",
            "Total number of API requests by endpoint",
            ["endpoint", "method", "status_code", "category"]
        )

        self.endpoint_response_time = registry.get_or_create_histogram(
            "api_endpoint_response_time_seconds",
            "API endpoint response time distribution",
            ["endpoint", "method", "category"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )

        self.endpoint_payload_size = registry.get_or_create_histogram(
            "api_endpoint_payload_size_bytes",
            "API request/response payload size distribution",
            ["endpoint", "direction"],  # direction: "request" or "response"
            buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        )

        self.endpoint_popularity = registry.get_or_create_gauge(
            "api_endpoint_popularity_score",
            "Relative popularity score of API endpoints",
            ["endpoint", "time_window"]
        )

        # User journey metrics - using real metrics
        self.user_journey_conversion_rate = registry.get_or_create_gauge(
            "api_user_journey_conversion_rate",
            "Conversion rate between journey stages",
            ["from_stage", "to_stage", "time_window"]
        )

        self.user_journey_duration = registry.get_or_create_histogram(
            "api_user_journey_stage_duration_seconds",
            "Time spent in each journey stage",
            ["stage"],
            buckets=[60, 300, 900, 1800, 3600, 7200, 14400, 28800, 86400]
        )

        self.active_user_sessions = registry.get_or_create_gauge(
            "api_active_user_sessions",
            "Number of currently active user sessions",
            ["stage"]
        )

        # Rate limiting metrics - using real metrics
        self.rate_limit_blocks = registry.get_or_create_counter(
            "api_rate_limit_blocks_total",
            "Total rate limit blocks by type",
            ["limit_type", "endpoint", "user_tier"]
        )

        self.rate_limit_utilization = registry.get_or_create_gauge(
            "api_rate_limit_utilization_ratio",
            "Current rate limit utilization ratio",
            ["limit_type", "endpoint"]
        )

        self.burst_detection_events = registry.get_or_create_counter(
            "api_burst_detection_events_total",
            "Total burst detection events",
            ["endpoint", "severity"]
        )

        # Authentication metrics - using real metrics
        self.auth_success_rate = registry.get_or_create_gauge(
            "api_authentication_success_rate",
            "Authentication success rate by method",
            ["method", "time_window"]
        )

        self.auth_session_duration = registry.get_or_create_histogram(
            "api_authentication_session_duration_seconds",
            "Authentication session duration distribution",
            ["method"],
            buckets=[300, 900, 1800, 3600, 7200, 14400, 28800, 86400]
        )

        self.mfa_usage_rate = registry.get_or_create_gauge(
            "api_mfa_usage_rate",
            "Multi-factor authentication usage rate",
            ["auth_method"]
        )

        self.logger.info("Successfully initialized all real metrics for API operations")

    async def start_collection(self) -> None:
        """Start background metrics collection and processing."""
        if self.is_running:
            return

        self.is_running = True
        self.aggregation_task = asyncio.create_task(self._aggregation_loop())
        self.session_cleanup_task = asyncio.create_task(self._session_cleanup_loop())
        self.logger.info("Started API metrics collection")

    async def stop_collection(self) -> None:
        """Stop background metrics collection and processing."""
        if not self.is_running:
            return

        self.is_running = False
        for task in [self.aggregation_task, self.session_cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.logger.info("Stopped API metrics collection")

    async def _aggregation_loop(self) -> None:
        """Background aggregation of metrics."""
        try:
            while self.is_running:
                await self._aggregate_metrics()
                await asyncio.sleep(self.aggregation_window_minutes * 60)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in API metrics aggregation: {e}")

    async def _session_cleanup_loop(self) -> None:
        """Background cleanup of expired sessions."""
        try:
            while self.is_running:
                await self._cleanup_expired_sessions()
                await asyncio.sleep(300)  # Clean up every 5 minutes
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in session cleanup: {e}")

    async def record_api_usage(self, metric: APIUsageMetric) -> None:
        """Record an API usage metric."""
        try:
            self.api_usage_metrics.append(metric)
            self.collection_stats["api_calls_tracked"] += 1

            # Update endpoint popularity cache
            endpoint_key = f"{metric.method.value}:{metric.endpoint}"
            self.endpoint_popularity_cache[endpoint_key] += 1

            # Update active sessions
            if metric.session_id:
                self.active_sessions[metric.session_id] = {
                    "user_id": metric.user_id,
                    "last_activity": metric.timestamp,
                    "endpoint_count": self.active_sessions.get(metric.session_id, {}).get("endpoint_count", 0) + 1
                }

            # Update Prometheus metrics immediately
            self.endpoint_request_count.labels(
                endpoint=metric.endpoint,
                method=metric.method.value,
                status_code=str(metric.status_code),
                category=metric.category.value
            ).inc()

            self.endpoint_response_time.labels(
                endpoint=metric.endpoint,
                method=metric.method.value,
                category=metric.category.value
            ).observe(metric.response_time_ms / 1000.0)

            # Payload size metrics
            self.endpoint_payload_size.labels(
                endpoint=metric.endpoint,
                direction="request"
            ).observe(metric.request_size_bytes)

            self.endpoint_payload_size.labels(
                endpoint=metric.endpoint,
                direction="response"
            ).observe(metric.response_size_bytes)

            self.logger.debug(f"Recorded API usage: {metric.method.value} {metric.endpoint}")

        except Exception as e:
            self.logger.error(f"Error recording API usage metric: {e}")

    async def record_user_journey(self, metric: UserJourneyMetric) -> None:
        """Record a user journey metric."""
        try:
            self.journey_metrics.append(metric)
            self.collection_stats["journey_events_tracked"] += 1

            # Update user journey state
            self.user_journey_states[metric.user_id] = metric.journey_stage

            # Calculate stage transition if available
            if metric.previous_stage and metric.previous_stage != metric.journey_stage:
                # This is a stage transition
                self.user_journey_conversion_rate.labels(
                    from_stage=metric.previous_stage.value,
                    to_stage=metric.journey_stage.value,
                    time_window="realtime"
                ).set(1.0)  # Will be aggregated later

            # Record time to action if available
            if metric.time_to_action_seconds:
                self.user_journey_duration.labels(
                    stage=metric.journey_stage.value
                ).observe(metric.time_to_action_seconds)

            self.logger.debug(f"Recorded user journey: {metric.user_id} -> {metric.journey_stage.value}")

        except Exception as e:
            self.logger.error(f"Error recording user journey metric: {e}")

    async def record_rate_limit(self, metric: RateLimitMetric) -> None:
        """Record a rate limiting metric."""
        try:
            self.rate_limit_metrics.append(metric)
            self.collection_stats["rate_limit_events_tracked"] += 1

            # Update Prometheus metrics
            if metric.blocked:
                self.rate_limit_blocks.labels(
                    limit_type=metric.limit_type,
                    endpoint=metric.endpoint,
                    user_tier=metric.user_tier or "default"
                ).inc()

            # Rate limit utilization
            utilization_ratio = metric.current_usage / metric.limit_value if metric.limit_value > 0 else 0
            self.rate_limit_utilization.labels(
                limit_type=metric.limit_type,
                endpoint=metric.endpoint
            ).set(utilization_ratio)

            # Burst detection
            if metric.burst_detected:
                severity = "high" if utilization_ratio > 0.9 else "medium"
                self.burst_detection_events.labels(
                    endpoint=metric.endpoint,
                    severity=severity
                ).inc()

            self.logger.debug(f"Recorded rate limit: {metric.limit_type} for {metric.endpoint}")

        except Exception as e:
            self.logger.error(f"Error recording rate limit metric: {e}")

    async def record_authentication(self, metric: AuthenticationMetric) -> None:
        """Record an authentication metric."""
        try:
            self.auth_metrics.append(metric)
            self.collection_stats["auth_events_tracked"] += 1

            # Update Prometheus metrics
            success_value = 1.0 if metric.success else 0.0
            self.auth_success_rate.labels(
                method=metric.authentication_method.value,
                time_window="realtime"
            ).set(success_value)

            # Session duration
            if metric.session_duration_seconds:
                self.auth_session_duration.labels(
                    method=metric.authentication_method.value
                ).observe(metric.session_duration_seconds)

            # MFA usage tracking
            if metric.authentication_method != AuthenticationMethod.ANONYMOUS:
                mfa_value = 1.0 if metric.mfa_used else 0.0
                self.mfa_usage_rate.labels(
                    auth_method=metric.authentication_method.value
                ).set(mfa_value)

            self.logger.debug(f"Recorded authentication: {metric.authentication_method.value}")

        except Exception as e:
            self.logger.error(f"Error recording authentication metric: {e}")

    async def _aggregate_metrics(self) -> None:
        """Aggregate metrics over time windows."""
        try:
            current_time = datetime.now(timezone.utc)
            window_start = current_time - timedelta(minutes=self.aggregation_window_minutes)

            # Aggregate API usage metrics
            await self._aggregate_api_usage_metrics(window_start, current_time)

            # Aggregate user journey metrics
            await self._aggregate_journey_metrics(window_start, current_time)

            # Aggregate rate limiting metrics
            await self._aggregate_rate_limit_metrics(window_start, current_time)

            # Aggregate authentication metrics
            await self._aggregate_auth_metrics(window_start, current_time)

            # Update active session counts
            await self._update_active_session_metrics()

            # Clean up old metrics
            await self._cleanup_old_metrics(current_time)

            self.collection_stats["last_aggregation"] = current_time

        except Exception as e:
            self.logger.error(f"Error in API metrics aggregation: {e}")

    async def _aggregate_api_usage_metrics(self, window_start: datetime, window_end: datetime) -> None:
        """Aggregate API usage metrics."""
        window_metrics = [
            m for m in self.api_usage_metrics
            if window_start <= m.timestamp <= window_end
        ]

        if not window_metrics:
            return

        # Calculate endpoint popularity scores
        endpoint_counts: DefaultDict[str, int] = defaultdict(int)
        total_requests = len(window_metrics)

        for metric in window_metrics:
            endpoint_key = f"{metric.method.value}:{metric.endpoint}"
            endpoint_counts[endpoint_key] += 1

        # Update popularity scores (normalized to 0-100)
        for endpoint_key, count in endpoint_counts.items():
            popularity_score = (count / total_requests) * 100 if total_requests > 0 else 0
            _, endpoint = endpoint_key.split(":", 1)  # method not used, but needed for unpacking

            self.endpoint_popularity.labels(
                endpoint=endpoint,
                time_window=f"{self.aggregation_window_minutes}m"
            ).set(popularity_score)

        # Calculate success rates by endpoint
        endpoint_success_rates: DefaultDict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})
        for metric in window_metrics:
            key = f"{metric.method.value}:{metric.endpoint}"
            endpoint_success_rates[key]["total"] += 1
            if 200 <= metric.status_code < 400:
                endpoint_success_rates[key]["success"] += 1

        self.logger.debug(
            f"Aggregated API usage metrics: {len(window_metrics)} requests, "
            f"{len(endpoint_counts)} unique endpoints"
        )

    async def _aggregate_journey_metrics(self, window_start: datetime, window_end: datetime) -> None:
        """Aggregate user journey metrics."""
        window_metrics = [
            m for m in self.journey_metrics
            if window_start <= m.timestamp <= window_end
        ]

        if not window_metrics:
            return

        # Calculate stage transitions and conversion rates
        stage_transitions: DefaultDict[UserJourneyStage, DefaultDict[UserJourneyStage, int]] = defaultdict(lambda: defaultdict(int))
        stage_durations: DefaultDict[UserJourneyStage, list[float]] = defaultdict(list)

        for metric in window_metrics:
            if metric.previous_stage and metric.previous_stage != metric.journey_stage:
                stage_transitions[metric.previous_stage][metric.journey_stage] += 1

            if metric.time_to_action_seconds:
                stage_durations[metric.journey_stage].append(metric.time_to_action_seconds)

        # Update conversion rate metrics
        for from_stage, to_stages in stage_transitions.items():
            total_transitions = sum(to_stages.values())
            for to_stage, count in to_stages.items():
                conversion_rate = count / total_transitions if total_transitions > 0 else 0

                self.user_journey_conversion_rate.labels(
                    from_stage=from_stage.value,
                    to_stage=to_stage.value,
                    time_window=f"{self.aggregation_window_minutes}m"
                ).set(conversion_rate)

        self.logger.debug(
            f"Aggregated journey metrics: {len(window_metrics)} events, "
            f"{len(stage_transitions)} stage transitions"
        )

    async def _aggregate_rate_limit_metrics(self, window_start: datetime, window_end: datetime) -> None:
        """Aggregate rate limiting metrics."""
        window_metrics = [
            m for m in self.rate_limit_metrics
            if window_start <= m.timestamp <= window_end
        ]

        if not window_metrics:
            return

        # Calculate average utilization by endpoint and limit type
        utilization_groups: DefaultDict[tuple[str, str], list[float]] = defaultdict(list)

        for metric in window_metrics:
            key = (metric.limit_type, metric.endpoint)
            utilization = metric.current_usage / metric.limit_value if metric.limit_value > 0 else 0
            utilization_groups[key].append(utilization)

        # Update average utilization metrics
        for (limit_type, endpoint), utilizations in utilization_groups.items():
            avg_utilization = statistics.mean(utilizations)

            self.rate_limit_utilization.labels(
                limit_type=limit_type,
                endpoint=endpoint
            ).set(avg_utilization)

        self.logger.debug(f"Aggregated rate limit metrics: {len(window_metrics)} events")

    async def _aggregate_auth_metrics(self, window_start: datetime, window_end: datetime) -> None:
        """Aggregate authentication metrics."""
        window_metrics = [
            m for m in self.auth_metrics
            if window_start <= m.timestamp <= window_end
        ]

        if not window_metrics:
            return

        # Calculate success rates by authentication method
        method_stats: DefaultDict[AuthenticationMethod, dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0, "mfa_used": 0})

        for metric in window_metrics:
            method = metric.authentication_method
            method_stats[method]["total"] += 1
            if metric.success:
                method_stats[method]["success"] += 1
            if metric.mfa_used:
                method_stats[method]["mfa_used"] += 1

        # Update aggregated success rates
        for method, stats in method_stats.items():
            success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
            mfa_rate = stats["mfa_used"] / stats["total"] if stats["total"] > 0 else 0

            self.auth_success_rate.labels(
                method=method.value,
                time_window=f"{self.aggregation_window_minutes}m"
            ).set(success_rate)

            self.mfa_usage_rate.labels(
                auth_method=method.value
            ).set(mfa_rate)

        self.logger.debug(f"Aggregated auth metrics: {len(window_metrics)} events")

    async def _update_active_session_metrics(self) -> None:
        """Update active session metrics."""
        active_count = len(self.active_sessions)

        # Count sessions by journey stage
        stage_counts: DefaultDict[UserJourneyStage, int] = defaultdict(int)
        for session_data in self.active_sessions.values():
            user_id = session_data.get("user_id")
            if user_id and user_id in self.user_journey_states:
                stage = self.user_journey_states[user_id]
                stage_counts[stage] += 1
            else:
                stage_counts[UserJourneyStage.REGULAR_USE] += 1  # Default

        # Update metrics
        for stage in UserJourneyStage:
            count = stage_counts[stage]
            self.active_user_sessions.labels(stage=stage.value).set(count)

        self.collection_stats["active_sessions_count"] = active_count

    async def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        current_time = datetime.now(timezone.utc)
        timeout_duration = timedelta(minutes=self.journey_timeout_minutes)

        expired_sessions = [
            session_id for session_id, session_data in self.active_sessions.items()
            if current_time - session_data["last_activity"] > timeout_duration
        ]

        for session_id in expired_sessions:
            del self.active_sessions[session_id]

        if expired_sessions:
            self.logger.debug(f"Cleaned up {len(expired_sessions)} expired sessions")

    async def _cleanup_old_metrics(self, current_time: datetime) -> None:
        """Clean up metrics older than retention period."""
        cutoff_time = current_time - timedelta(hours=self.retention_hours)

        # Clean API usage metrics
        original_api_count = len(self.api_usage_metrics)
        self.api_usage_metrics = deque(
            (m for m in self.api_usage_metrics if m.timestamp > cutoff_time),
            maxlen=self.api_usage_metrics.maxlen
        )

        # Clean journey metrics
        original_journey_count = len(self.journey_metrics)
        self.journey_metrics = deque(
            (m for m in self.journey_metrics if m.timestamp > cutoff_time),
            maxlen=self.journey_metrics.maxlen
        )

        # Clean rate limit metrics
        original_rate_count = len(self.rate_limit_metrics)
        self.rate_limit_metrics = deque(
            (m for m in self.rate_limit_metrics if m.timestamp > cutoff_time),
            maxlen=self.rate_limit_metrics.maxlen
        )

        # Clean auth metrics
        original_auth_count = len(self.auth_metrics)
        self.auth_metrics = deque(
            (m for m in self.auth_metrics if m.timestamp > cutoff_time),
            maxlen=self.auth_metrics.maxlen
        )

        cleaned_total = (
            (original_api_count - len(self.api_usage_metrics)) +
            (original_journey_count - len(self.journey_metrics)) +
            (original_rate_count - len(self.rate_limit_metrics)) +
            (original_auth_count - len(self.auth_metrics))
        )

        if cleaned_total > 0:
            self.logger.debug(f"Cleaned up {cleaned_total} old API metrics")

    async def get_endpoint_analytics(self, hours: int = 1) -> dict[str, Any]:
        """Get analytics for API endpoints."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_metrics = [m for m in self.api_usage_metrics if m.timestamp > cutoff_time]

        if not recent_metrics:
            return {"status": "no_data", "hours": hours}

        # Group by endpoint - properly typed
        endpoint_stats: DefaultDict[str, dict[str, Any]] = defaultdict(lambda: {
            "request_count": 0,
            "total_response_time": 0.0,
            "status_codes": defaultdict(int),
            "methods": defaultdict(int),
            "total_request_bytes": 0,
            "total_response_bytes": 0,
            "unique_users": set(),
            "rate_limited_count": 0,
            "cache_hit_count": 0
        })

        for metric in recent_metrics:
            stats = endpoint_stats[metric.endpoint]
            stats["request_count"] += 1
            stats["total_response_time"] += metric.response_time_ms
            stats["status_codes"][metric.status_code] += 1
            stats["methods"][metric.method.value] += 1
            stats["total_request_bytes"] += metric.request_size_bytes
            stats["total_response_bytes"] += metric.response_size_bytes

            if metric.user_id:
                stats["unique_users"].add(metric.user_id)
            if metric.rate_limited:
                stats["rate_limited_count"] += 1
            if metric.cache_hit:
                stats["cache_hit_count"] += 1

        # Calculate derived metrics
        endpoint_analytics = {}
        for endpoint, stats in endpoint_stats.items():
            count = stats["request_count"]
            endpoint_analytics[endpoint] = {
                "request_count": count,
                "avg_response_time_ms": stats["total_response_time"] / count if count > 0 else 0,
                "requests_per_hour": count / hours,
                "unique_users": len(stats["unique_users"]),
                "success_rate": sum(
                    count for status, count in stats["status_codes"].items()
                    if 200 <= status < 400
                ) / count if count > 0 else 0,
                "rate_limit_rate": stats["rate_limited_count"] / count if count > 0 else 0,
                "cache_hit_rate": stats["cache_hit_count"] / count if count > 0 else 0,
                "avg_request_size_bytes": stats["total_request_bytes"] / count if count > 0 else 0,
                "avg_response_size_bytes": stats["total_response_bytes"] / count if count > 0 else 0,
                "status_distribution": dict(stats["status_codes"]),
                "method_distribution": dict(stats["methods"])
            }

        return {
            "total_requests": len(recent_metrics),
            "unique_endpoints": len(endpoint_analytics),
            "endpoint_analytics": endpoint_analytics,
            "time_window_hours": hours,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    async def get_user_journey_analytics(self, hours: int = 24) -> dict[str, Any]:
        """Get user journey analytics."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_metrics = [m for m in self.journey_metrics if m.timestamp > cutoff_time]

        if not recent_metrics:
            return {"status": "no_data", "hours": hours}

        # Analyze stage transitions and conversions - properly typed
        stage_stats: dict[UserJourneyStage, dict[str, set[str] | int | float | list[float]]] = defaultdict(lambda: {
            "users": set(),
            "events": 0,
            "successful_events": 0,
            "total_conversion_value": 0.0,
            "times_to_action": []
        })

        transition_matrix: DefaultDict[UserJourneyStage, DefaultDict[UserJourneyStage, int]] = defaultdict(lambda: defaultdict(int))

        for metric in recent_metrics:
            stage = metric.journey_stage
            stats = stage_stats[stage]

            # Type-safe operations
            users_set = stats["users"]
            if isinstance(users_set, set):
                users_set.add(metric.user_id)

            events_count = stats["events"]
            if isinstance(events_count, int):
                stats["events"] = events_count + 1

            if metric.success:
                successful_events = stats["successful_events"]
                if isinstance(successful_events, int):
                    stats["successful_events"] = successful_events + 1

            if metric.conversion_value:
                total_conversion = stats["total_conversion_value"]
                if isinstance(total_conversion, (int, float)):
                    stats["total_conversion_value"] = total_conversion + metric.conversion_value

            if metric.time_to_action_seconds:
                times_list = stats["times_to_action"]
                if isinstance(times_list, list):
                    times_list.append(metric.time_to_action_seconds)

            # Track transitions
            if metric.previous_stage and metric.previous_stage != metric.journey_stage:
                transition_matrix[metric.previous_stage][metric.journey_stage] += 1

        # Calculate stage analytics with proper type handling
        stage_analytics = {}
        for stage, stats in stage_stats.items():
            # Type-safe extraction of values
            users_set = stats["users"]
            users_count = len(users_set) if isinstance(users_set, set) else 0

            events_count = stats["events"]
            events_count = events_count if isinstance(events_count, int) else 0

            successful_events = stats["successful_events"]
            successful_events = successful_events if isinstance(successful_events, int) else 0

            total_conversion_value = stats["total_conversion_value"]
            total_conversion_value = total_conversion_value if isinstance(total_conversion_value, (int, float)) else 0.0

            times_to_action = stats["times_to_action"]
            times_to_action = times_to_action if isinstance(times_to_action, list) else []

            stage_analytics[stage.value] = {
                "unique_users": users_count,
                "total_events": events_count,
                "success_rate": successful_events / events_count if events_count > 0 else 0,
                "avg_conversion_value": total_conversion_value / users_count if users_count > 0 else 0,
                "avg_time_to_action_seconds": statistics.mean(times_to_action) if times_to_action else None,
                "median_time_to_action_seconds": statistics.median(times_to_action) if times_to_action else None
            }

        # Calculate transition rates
        transition_analytics = {}
        for from_stage, to_stages in transition_matrix.items():
            total_transitions = sum(to_stages.values())
            transition_analytics[from_stage.value] = {
                to_stage.value: {
                    "count": count,
                    "rate": count / total_transitions if total_transitions > 0 else 0
                }
                for to_stage, count in to_stages.items()
            }

        return {
            "total_journey_events": len(recent_metrics),
            "stage_analytics": stage_analytics,
            "transition_analytics": transition_analytics,
            "active_sessions": len(self.active_sessions),
            "time_window_hours": hours,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    def get_collection_stats(self) -> dict[str, Any]:
        """Get current collection statistics."""
        return {
            **self.collection_stats,
            "current_metrics_count": {
                "api_usage": len(self.api_usage_metrics),
                "user_journey": len(self.journey_metrics),
                "rate_limiting": len(self.rate_limit_metrics),
                "authentication": len(self.auth_metrics)
            },
            "endpoint_popularity_cache_size": len(self.endpoint_popularity_cache),
            "is_running": self.is_running,
            "config": {
                "aggregation_window_minutes": self.aggregation_window_minutes,
                "retention_hours": self.retention_hours,
                "journey_timeout_minutes": self.journey_timeout_minutes
            }
        }

# Global instance
_api_metrics_collector: APIMetricsCollector | None = None

def get_api_metrics_collector(config: dict[str, Any] | None = None) -> APIMetricsCollector:
    """Get global API metrics collector instance."""
    global _api_metrics_collector
    if _api_metrics_collector is None:
        _api_metrics_collector = APIMetricsCollector(config)
    return _api_metrics_collector

# Convenience functions for recording metrics
async def record_api_request(
    endpoint: str,
    method: HTTPMethod,
    category: EndpointCategory,
    status_code: int,
    response_time_ms: float,
    request_size_bytes: int = 0,
    response_size_bytes: int = 0,
    user_id: str | None = None,
    session_id: str | None = None,
    ip_address: str = "unknown",
    user_agent: str | None = None,
    query_parameters_count: int = 0,
    payload_type: str | None = None,
    rate_limited: bool = False,
    cache_hit: bool = False,
    authentication_method: AuthenticationMethod = AuthenticationMethod.ANONYMOUS,
    api_version: str | None = None
) -> None:
    """Record an API request metric (convenience function)."""
    collector = get_api_metrics_collector()
    metric = APIUsageMetric(
        endpoint=endpoint,
        method=method,
        category=category,
        status_code=status_code,
        response_time_ms=response_time_ms,
        request_size_bytes=request_size_bytes,
        response_size_bytes=response_size_bytes,
        user_id=user_id,
        session_id=session_id,
        ip_address=ip_address,
        user_agent=user_agent,
        timestamp=datetime.now(timezone.utc),
        query_parameters_count=query_parameters_count,
        payload_type=payload_type,
        rate_limited=rate_limited,
        cache_hit=cache_hit,
        authentication_method=authentication_method,
        api_version=api_version
    )
    await collector.record_api_usage(metric)

async def record_user_journey_event(
    user_id: str,
    session_id: str,
    journey_stage: UserJourneyStage,
    event_type: str,
    endpoint: str,
    success: bool = True,
    conversion_value: float | None = None,
    time_to_action_seconds: float | None = None,
    previous_stage: UserJourneyStage | None = None,
    feature_flags_active: list[str] | None = None,
    cohort_id: str | None = None,
    metadata: dict[str, Any] | None = None
) -> None:
    """Record a user journey event (convenience function)."""
    collector = get_api_metrics_collector()
    metric = UserJourneyMetric(
        user_id=user_id,
        session_id=session_id,
        journey_stage=journey_stage,
        event_type=event_type,
        endpoint=endpoint,
        success=success,
        conversion_value=conversion_value,
        time_to_action_seconds=time_to_action_seconds,
        previous_stage=previous_stage,
        feature_flags_active=feature_flags_active or [],
        cohort_id=cohort_id,
        timestamp=datetime.now(timezone.utc),
        metadata=metadata or {}
    )
    await collector.record_user_journey(metric)
