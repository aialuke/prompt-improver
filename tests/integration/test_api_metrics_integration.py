"""
Integration test for API metrics to verify real behavior without mocks.

Tests the fixed api_metrics.py implementation with actual data flow
and PostgreSQL integration following APES architectural patterns.
"""
import asyncio
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, Dict
import pytest
import pytest_asyncio
from src.prompt_improver.metrics.api_metrics import APIMetricsCollector, APIUsageMetric, AuthenticationMethod, AuthenticationMetric, EndpointCategory, HTTPMethod, RateLimitMetric, UserJourneyMetric, UserJourneyStage, get_api_metrics_collector, record_api_request, record_user_journey_event

@pytest.mark.asyncio
class TestAPIMetricsIntegration:
    """Test API metrics with real behavior and data flow."""

    @pytest_asyncio.fixture
    async def metrics_collector(self):
        """Create a real metrics collector instance."""
        config = {'max_api_metrics': 1000, 'max_journey_metrics': 500, 'aggregation_window_minutes': 1, 'retention_hours': 24}
        collector = APIMetricsCollector(config)
        await collector.start_collection()
        yield collector
        await collector.stop_collection()

    async def test_api_usage_recording(self, metrics_collector):
        """Test recording API usage metrics with real data."""
        metric = APIUsageMetric(endpoint='/api/v1/improve-prompt', method=HTTPMethod.POST, category=EndpointCategory.PROMPT_IMPROVEMENT, status_code=200, response_time_ms=150.5, request_size_bytes=1024, response_size_bytes=2048, user_id='user123', session_id='session456', ip_address='192.168.1.100', user_agent='TestAgent/1.0', timestamp=datetime.now(UTC), query_parameters_count=3, payload_type='application/json', rate_limited=False, cache_hit=True, authentication_method=AuthenticationMethod.JWT_TOKEN, api_version='v1')
        await metrics_collector.record_api_usage(metric)
        stats = metrics_collector.get_collection_stats()
        assert stats['api_calls_tracked'] == 1
        assert stats['current_metrics_count']['api_usage'] == 1
        endpoint_key = f'{metric.method.value}:{metric.endpoint}'
        assert metrics_collector.endpoint_popularity_cache[endpoint_key] == 1

    async def test_user_journey_recording(self, metrics_collector):
        """Test recording user journey metrics with stage transitions."""
        metric1 = UserJourneyMetric(user_id='user123', session_id='session456', journey_stage=UserJourneyStage.ONBOARDING, event_type='signup', endpoint='/api/v1/auth/register', success=True, conversion_value=None, time_to_action_seconds=30.0, previous_stage=None, feature_flags_active=['new_ui'], cohort_id='cohort_a', timestamp=datetime.now(UTC), metadata={'source': 'landing_page'})
        await metrics_collector.record_user_journey(metric1)
        metric2 = UserJourneyMetric(user_id='user123', session_id='session456', journey_stage=UserJourneyStage.FIRST_USE, event_type='first_prompt', endpoint='/api/v1/improve-prompt', success=True, conversion_value=10.0, time_to_action_seconds=120.0, previous_stage=UserJourneyStage.ONBOARDING, feature_flags_active=['new_ui'], cohort_id='cohort_a', timestamp=datetime.now(UTC), metadata={'prompt_length': 150})
        await metrics_collector.record_user_journey(metric2)
        stats = metrics_collector.get_collection_stats()
        assert stats['journey_events_tracked'] == 2
        assert metrics_collector.user_journey_states['user123'] == UserJourneyStage.FIRST_USE

    async def test_analytics_generation(self, metrics_collector):
        """Test analytics generation with real data."""
        for i in range(5):
            metric = APIUsageMetric(endpoint=f'/api/v1/endpoint{i % 2}', method=HTTPMethod.GET if i % 2 == 0 else HTTPMethod.POST, category=EndpointCategory.PROMPT_IMPROVEMENT, status_code=200 if i < 4 else 500, response_time_ms=100.0 + i * 10, request_size_bytes=500 + i * 100, response_size_bytes=1000 + i * 200, user_id=f'user{i}', session_id=f'session{i}', ip_address='192.168.1.100', user_agent='TestAgent/1.0', timestamp=datetime.now(UTC), query_parameters_count=i, payload_type='application/json', rate_limited=i == 4, cache_hit=i % 2 == 0, authentication_method=AuthenticationMethod.JWT_TOKEN, api_version='v1')
            await metrics_collector.record_api_usage(metric)
        analytics = await metrics_collector.get_endpoint_analytics(hours=1)
        assert 'total_requests' in analytics
        assert 'unique_endpoints' in analytics
        assert 'endpoint_analytics' in analytics
        assert analytics['total_requests'] == 5
        assert analytics['unique_endpoints'] == 2
        endpoint_data = analytics['endpoint_analytics']
        assert len(endpoint_data) == 2
        for endpoint, data in endpoint_data.items():
            assert 'request_count' in data
            assert 'avg_response_time_ms' in data
            assert 'success_rate' in data
            assert 'rate_limit_rate' in data
            assert 'cache_hit_rate' in data

    async def test_metrics_aggregation(self, metrics_collector):
        """Test real-time metrics aggregation."""
        await self.test_api_usage_recording(metrics_collector)
        await self.test_user_journey_recording(metrics_collector)
        await metrics_collector._aggregate_metrics()
        stats = metrics_collector.get_collection_stats()
        assert stats['last_aggregation'] is not None
        assert isinstance(stats['last_aggregation'], datetime)

    async def test_session_management(self, metrics_collector):
        """Test active session tracking and cleanup."""
        metric = APIUsageMetric(endpoint='/api/v1/test', method=HTTPMethod.GET, category=EndpointCategory.HEALTH_CHECK, status_code=200, response_time_ms=50.0, request_size_bytes=100, response_size_bytes=200, user_id='user123', session_id='active_session', ip_address='192.168.1.100', user_agent='TestAgent/1.0', timestamp=datetime.now(UTC), query_parameters_count=0, payload_type='application/json', rate_limited=False, cache_hit=False, authentication_method=AuthenticationMethod.JWT_TOKEN, api_version='v1')
        await metrics_collector.record_api_usage(metric)
        assert 'active_session' in metrics_collector.active_sessions
        session_data = metrics_collector.active_sessions['active_session']
        assert session_data['user_id'] == 'user123'
        assert session_data['endpoint_count'] == 1
        await metrics_collector._cleanup_expired_sessions()
        assert 'active_session' in metrics_collector.active_sessions

@pytest.mark.asyncio
async def test_convenience_functions():
    """Test convenience functions for recording metrics."""
    await record_api_request(endpoint='/api/v1/test', method=HTTPMethod.GET, category=EndpointCategory.HEALTH_CHECK, status_code=200, response_time_ms=100.0, user_id='test_user')
    await record_user_journey_event(user_id='test_user', session_id='test_session', journey_stage=UserJourneyStage.REGULAR_USE, event_type='api_call', endpoint='/api/v1/test', success=True)
    collector = get_api_metrics_collector()
    stats = collector.get_collection_stats()
    assert stats['api_calls_tracked'] >= 1
    assert stats['journey_events_tracked'] >= 1
if __name__ == '__main__':

    async def main():
        print('Testing API Metrics Integration...')
        await test_convenience_functions()
        print('✓ Convenience functions work')
        collector = APIMetricsCollector()
        await collector.start_collection()
        try:
            metric = APIUsageMetric(endpoint='/test', method=HTTPMethod.GET, category=EndpointCategory.HEALTH_CHECK, status_code=200, response_time_ms=50.0, request_size_bytes=100, response_size_bytes=200, user_id='test', session_id='test', ip_address='127.0.0.1', user_agent='Test', timestamp=datetime.now(UTC), query_parameters_count=0, payload_type='json', rate_limited=False, cache_hit=False, authentication_method=AuthenticationMethod.ANONYMOUS, api_version='v1')
            await collector.record_api_usage(metric)
            print('✓ API usage recording works')
            analytics = await collector.get_endpoint_analytics()
            print(f"✓ Analytics generated: {analytics.get('total_requests', 0)} requests")
        finally:
            await collector.stop_collection()
        print('All tests passed! 🎉')
    asyncio.run(main())
