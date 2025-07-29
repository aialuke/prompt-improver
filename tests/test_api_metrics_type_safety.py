"""
Real behavior testing for API metrics type safety fixes.

Tests that all type annotation fixes work correctly with actual metrics collection
scenarios, using real components instead of mocked objects.
"""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from typing import Any

from prompt_improver.metrics.api_metrics import (
    APIMetricsCollector,
    APIUsageMetric,
    UserJourneyMetric,
    RateLimitMetric,
    AuthenticationMetric,
    HTTPMethod,
    EndpointCategory,
    UserJourneyStage,
    AuthenticationMethod,
    record_api_request,
    record_user_journey_event,
    get_api_metrics_collector
)


class TestAPIMetricsTypeSafety:
    """Test suite for API metrics type safety and real behavior validation."""

    @pytest_asyncio.fixture
    async def metrics_collector(self) -> APIMetricsCollector:
        """Create a real metrics collector instance."""
        config = {
            "max_api_metrics": 1000,
            "max_journey_metrics": 500,
            "max_rate_limit_metrics": 200,
            "max_auth_metrics": 300,
            "aggregation_window_minutes": 1,
            "retention_hours": 1,
            "journey_timeout_minutes": 5
        }
        collector = APIMetricsCollector(config)
        await collector.start_collection()
        yield collector
        await collector.stop_collection()

    @pytest.mark.asyncio
    async def test_api_usage_metric_creation_and_recording(self, metrics_collector: APIMetricsCollector) -> None:
        """Test that API usage metrics can be created and recorded with proper types."""
        # Create a real API usage metric
        metric = APIUsageMetric(
            endpoint="/api/v1/prompts",
            method=HTTPMethod.POST,
            category=EndpointCategory.PROMPT_IMPROVEMENT,
            status_code=200,
            response_time_ms=150.5,
            request_size_bytes=1024,
            response_size_bytes=2048,
            user_id="user123",
            session_id="session456",
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
            timestamp=datetime.now(timezone.utc),
            query_parameters_count=3,
            payload_type="application/json",
            rate_limited=False,
            cache_hit=True,
            authentication_method=AuthenticationMethod.JWT_TOKEN,
            api_version="v1"
        )

        # Record the metric
        await metrics_collector.record_api_usage(metric)

        # Verify it was recorded
        assert len(metrics_collector.api_usage_metrics) == 1
        assert metrics_collector.collection_stats["api_calls_tracked"] == 1

        # Verify the metric data integrity
        recorded_metric = metrics_collector.api_usage_metrics[0]
        assert recorded_metric.endpoint == "/api/v1/prompts"
        assert recorded_metric.method == HTTPMethod.POST
        assert recorded_metric.status_code == 200
        assert recorded_metric.user_id == "user123"

    @pytest.mark.asyncio
    async def test_user_journey_metric_creation_and_recording(self, metrics_collector: APIMetricsCollector) -> None:
        """Test that user journey metrics can be created and recorded with proper types."""
        # Create a real user journey metric
        metric = UserJourneyMetric(
            user_id="user123",
            session_id="session456",
            journey_stage=UserJourneyStage.FIRST_USE,
            event_type="prompt_creation",
            endpoint="/api/v1/prompts",
            success=True,
            conversion_value=10.5,
            time_to_action_seconds=30.2,
            previous_stage=UserJourneyStage.ONBOARDING,
            feature_flags_active=["new_ui", "advanced_features"],
            cohort_id="cohort_a",
            timestamp=datetime.now(timezone.utc),
            metadata={"experiment": "test_group", "version": "1.0"}
        )

        # Record the metric
        await metrics_collector.record_user_journey(metric)

        # Verify it was recorded
        assert len(metrics_collector.journey_metrics) == 1
        assert metrics_collector.collection_stats["journey_events_tracked"] == 1

        # Verify user journey state tracking
        assert metrics_collector.user_journey_states["user123"] == UserJourneyStage.FIRST_USE

    @pytest.mark.asyncio
    async def test_rate_limit_metric_creation_and_recording(self, metrics_collector: APIMetricsCollector) -> None:
        """Test that rate limit metrics can be created and recorded with proper types."""
        # Create a real rate limit metric
        metric = RateLimitMetric(
            user_id="user123",
            ip_address="192.168.1.1",
            endpoint="/api/v1/prompts",
            limit_type="user",
            limit_value=100,
            current_usage=75,
            time_window_seconds=3600,
            blocked=False,
            burst_detected=True,
            timestamp=datetime.now(timezone.utc),
            user_tier="premium",
            override_applied=False
        )

        # Record the metric
        await metrics_collector.record_rate_limit(metric)

        # Verify it was recorded
        assert len(metrics_collector.rate_limit_metrics) == 1
        assert metrics_collector.collection_stats["rate_limit_events_tracked"] == 1

    @pytest.mark.asyncio
    async def test_authentication_metric_creation_and_recording(self, metrics_collector: APIMetricsCollector) -> None:
        """Test that authentication metrics can be created and recorded with proper types."""
        # Create a real authentication metric
        metric = AuthenticationMetric(
            user_id="user123",
            authentication_method=AuthenticationMethod.JWT_TOKEN,
            success=True,
            failure_reason=None,
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
            session_duration_seconds=3600.0,
            mfa_used=True,
            token_type="Bearer",
            timestamp=datetime.now(timezone.utc),
            geo_location="US-CA",
            device_fingerprint="device123"
        )

        # Record the metric
        await metrics_collector.record_authentication(metric)

        # Verify it was recorded
        assert len(metrics_collector.auth_metrics) == 1
        assert metrics_collector.collection_stats["auth_events_tracked"] == 1

    @pytest.mark.asyncio
    async def test_metrics_aggregation_with_real_data(self, metrics_collector: APIMetricsCollector) -> None:
        """Test that metrics aggregation works with real data and proper types."""
        # Create multiple API usage metrics
        for i in range(5):
            metric = APIUsageMetric(
                endpoint=f"/api/v1/endpoint{i % 2}",
                method=HTTPMethod.GET if i % 2 == 0 else HTTPMethod.POST,
                category=EndpointCategory.PROMPT_IMPROVEMENT,
                status_code=200 if i < 4 else 500,
                response_time_ms=100.0 + i * 10,
                request_size_bytes=512 + i * 100,
                response_size_bytes=1024 + i * 200,
                user_id=f"user{i}",
                session_id=f"session{i}",
                ip_address="192.168.1.1",
                user_agent="TestAgent/1.0",
                timestamp=datetime.now(timezone.utc),
                query_parameters_count=i,
                payload_type="application/json",
                rate_limited=i == 4,
                cache_hit=i % 2 == 0,
                authentication_method=AuthenticationMethod.JWT_TOKEN,
                api_version="v1"
            )
            await metrics_collector.record_api_usage(metric)

        # Trigger aggregation manually
        await metrics_collector._aggregate_metrics()

        # Verify aggregation completed without errors
        assert metrics_collector.collection_stats["last_aggregation"] is not None
        assert len(metrics_collector.api_usage_metrics) == 5

    @pytest.mark.asyncio
    async def test_endpoint_analytics_with_real_data(self, metrics_collector: APIMetricsCollector) -> None:
        """Test that endpoint analytics work with real data and return proper types."""
        # Add some test data
        for i in range(3):
            metric = APIUsageMetric(
                endpoint="/api/v1/test",
                method=HTTPMethod.GET,
                category=EndpointCategory.PROMPT_IMPROVEMENT,
                status_code=200,
                response_time_ms=100.0,
                request_size_bytes=512,
                response_size_bytes=1024,
                user_id=f"user{i}",
                session_id=f"session{i}",
                ip_address="192.168.1.1",
                user_agent="TestAgent/1.0",
                timestamp=datetime.now(timezone.utc),
                query_parameters_count=0,
                payload_type="application/json",
                rate_limited=False,
                cache_hit=False,
                authentication_method=AuthenticationMethod.JWT_TOKEN,
                api_version="v1"
            )
            await metrics_collector.record_api_usage(metric)

        # Get analytics
        analytics = await metrics_collector.get_endpoint_analytics(hours=1)

        # Verify analytics structure and types
        assert isinstance(analytics, dict)
        assert "total_requests" in analytics
        assert "unique_endpoints" in analytics
        assert "endpoint_analytics" in analytics
        assert analytics["total_requests"] == 3
        assert analytics["unique_endpoints"] == 1

    @pytest.mark.asyncio
    async def test_convenience_functions_type_safety(self) -> None:
        """Test that convenience functions work with proper type annotations."""
        # Test record_api_request convenience function
        await record_api_request(
            endpoint="/api/v1/test",
            method=HTTPMethod.POST,
            category=EndpointCategory.PROMPT_IMPROVEMENT,
            status_code=201,
            response_time_ms=200.5,
            user_id="user123",
            session_id="session456"
        )

        # Test record_user_journey_event convenience function
        await record_user_journey_event(
            user_id="user123",
            session_id="session456",
            journey_stage=UserJourneyStage.FIRST_USE,
            event_type="test_event",
            endpoint="/api/v1/test",
            success=True,
            feature_flags_active=["flag1", "flag2"],
            metadata={"test": "data"}
        )

        # Verify global collector was used
        collector = get_api_metrics_collector()
        assert len(collector.api_usage_metrics) >= 1
        assert len(collector.journey_metrics) >= 1

    @pytest.mark.asyncio
    async def test_collection_stats_type_safety(self, metrics_collector: APIMetricsCollector) -> None:
        """Test that collection stats return proper types."""
        stats = metrics_collector.get_collection_stats()

        # Verify stats structure and types
        assert isinstance(stats, dict)
        assert isinstance(stats["api_calls_tracked"], int)
        assert isinstance(stats["journey_events_tracked"], int)
        assert isinstance(stats["rate_limit_events_tracked"], int)
        assert isinstance(stats["auth_events_tracked"], int)
        assert isinstance(stats["is_running"], bool)
        assert isinstance(stats["current_metrics_count"], dict)
        assert isinstance(stats["config"], dict)

    def test_metric_protocol_compliance(self) -> None:
        """Test that metrics comply with the MetricProtocol."""
        from prompt_improver.performance.monitoring.metrics_registry import MockMetric

        # Test MockMetric implements the protocol correctly
        mock_metric = MockMetric()

        # These should not raise type errors
        mock_metric.inc()
        mock_metric.set(1.0)
        mock_metric.observe(2.5)
        labeled = mock_metric.labels(endpoint="test")
        assert isinstance(labeled, MockMetric)
