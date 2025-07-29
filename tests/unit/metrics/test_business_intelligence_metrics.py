"""
Unit tests for Business Intelligence Metrics.

Tests the fixed implementation with real behavior rather than mocked components.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

from prompt_improver.metrics.business_intelligence_metrics import (
    BusinessIntelligenceMetricsCollector,
    FeatureAdoptionMetric,
    UserEngagementMetric,
    CostTrackingMetric,
    ResourceUtilizationMetric,
    FeatureCategory,
    UserTier,
    CostType,
    ResourceType
)


class TestBusinessIntelligenceMetrics:
    """Test business intelligence metrics functionality."""

    def test_collector_initialization(self):
        """Test that the collector initializes correctly."""
        # Test with default config
        collector = BusinessIntelligenceMetricsCollector()
        assert collector.config == {}
        assert len(collector.adoption_metrics) == 0
        assert len(collector.engagement_metrics) == 0
        assert len(collector.cost_metrics) == 0
        assert len(collector.utilization_metrics) == 0

        # Test with custom config
        config = {
            "max_adoption_metrics": 1000,
            "max_engagement_metrics": 500,
            "aggregation_window_minutes": 10
        }
        collector_custom = BusinessIntelligenceMetricsCollector(config)
        assert collector_custom.config == config
        assert collector_custom.aggregation_window_minutes == 10

    def test_feature_adoption_metric_creation(self):
        """Test creating feature adoption metrics."""
        metric = FeatureAdoptionMetric(
            feature_name="prompt_enhancement",
            feature_category=FeatureCategory.PROMPT_ENHANCEMENT,
            user_id="user123",
            user_tier=UserTier.PROFESSIONAL,
            session_id="session456",
            first_use=True,
            usage_count=1,
            time_spent_seconds=30.5,
            success=True,
            error_type=None,
            feature_version="1.0.0",
            user_cohort="beta",
            onboarding_completed=True,
            conversion_funnel_stage="activated",
            experiment_variant="A",
            timestamp=datetime.now(timezone.utc),
            metadata={"source": "test"}
        )

        assert metric.feature_name == "prompt_enhancement"
        assert metric.feature_category == FeatureCategory.PROMPT_ENHANCEMENT
        assert metric.user_tier == UserTier.PROFESSIONAL
        assert metric.first_use is True
        assert metric.success is True

    def test_user_engagement_metric_creation(self):
        """Test creating user engagement metrics."""
        metric = UserEngagementMetric(
            user_id="user123",
            user_tier=UserTier.PROFESSIONAL,
            session_id="session456",
            session_duration_seconds=300.0,
            pages_viewed=5,
            features_used=["feature1", "feature2"],
            actions_performed=10,
            successful_actions=8,
            time_to_first_action_seconds=5.0,
            bounce_rate_indicator=False,
            return_user=True,
            days_since_last_visit=2,
            total_lifetime_value=100.0,
            current_streak_days=5,
            timestamp=datetime.now(timezone.utc),
            user_agent="test-agent",
            referrer="test-referrer"
        )

        assert metric.user_id == "user123"
        assert metric.user_tier == UserTier.PROFESSIONAL
        assert metric.session_duration_seconds == 300.0
        assert metric.actions_performed == 10
        assert metric.successful_actions == 8

    @pytest.mark.asyncio
    async def test_record_feature_adoption(self):
        """Test recording feature adoption metrics."""
        collector = BusinessIntelligenceMetricsCollector()

        metric = FeatureAdoptionMetric(
            feature_name="test_feature",
            feature_category=FeatureCategory.PROMPT_ENHANCEMENT,
            user_id="user123",
            user_tier=UserTier.PROFESSIONAL,
            session_id="session456",
            first_use=True,
            usage_count=1,
            time_spent_seconds=30.5,
            success=True,
            error_type=None,
            feature_version="1.0.0",
            user_cohort="beta",
            onboarding_completed=True,
            conversion_funnel_stage="activated",
            experiment_variant="A",
            timestamp=datetime.now(timezone.utc),
            metadata={"source": "test"}
        )

        await collector.record_feature_adoption(metric)

        # Verify metric was recorded
        assert len(collector.adoption_metrics) == 1
        assert collector.collection_stats["feature_adoptions_tracked"] == 1

        # Verify feature cache was updated
        feature_key = f"{metric.feature_category.value}:{metric.feature_name}"
        assert feature_key in collector.feature_usage_cache

    @pytest.mark.asyncio
    async def test_record_user_engagement(self):
        """Test recording user engagement metrics."""
        collector = BusinessIntelligenceMetricsCollector()

        metric = UserEngagementMetric(
            user_id="user123",
            user_tier=UserTier.PROFESSIONAL,
            session_id="session456",
            session_duration_seconds=300.0,
            pages_viewed=5,
            features_used=["feature1", "feature2"],
            actions_performed=10,
            successful_actions=8,
            time_to_first_action_seconds=5.0,
            bounce_rate_indicator=False,
            return_user=True,
            days_since_last_visit=2,
            total_lifetime_value=100.0,
            current_streak_days=5,
            timestamp=datetime.now(timezone.utc),
            user_agent="test-agent",
            referrer="test-referrer"
        )

        await collector.record_user_engagement(metric)

        # Verify metric was recorded
        assert len(collector.engagement_metrics) == 1
        assert collector.collection_stats["user_engagements_tracked"] == 1

        # Verify user session was tracked
        assert metric.session_id in collector.user_sessions

    def test_collection_stats(self):
        """Test getting collection statistics."""
        collector = BusinessIntelligenceMetricsCollector()
        stats = collector.get_collection_stats()

        # Check that all expected keys are present
        expected_keys = [
            "feature_adoptions_tracked",
            "user_engagements_tracked",
            "cost_events_tracked",
            "utilization_events_tracked",
            "active_user_sessions",
            "total_cost_tracked",
            "cost_alerts_triggered",
            "last_aggregation",
            "current_metrics_count",
            "feature_cache_size",
            "cost_accumulator_size"
        ]

        for key in expected_keys:
            assert key in stats

        # Check initial values
        assert stats["feature_adoptions_tracked"] == 0
        assert stats["user_engagements_tracked"] == 0
        assert stats["current_metrics_count"]["feature_adoptions"] == 0

    @pytest.mark.asyncio
    async def test_feature_adoption_report(self):
        """Test generating feature adoption report."""
        collector = BusinessIntelligenceMetricsCollector()

        # Add some test metrics
        for i in range(3):
            metric = FeatureAdoptionMetric(
                feature_name=f"feature_{i}",
                feature_category=FeatureCategory.PROMPT_ENHANCEMENT,
                user_id=f"user_{i}",
                user_tier=UserTier.PROFESSIONAL,
                session_id=f"session_{i}",
                first_use=i == 0,  # Only first metric is first use
                usage_count=1,
                time_spent_seconds=30.0,
                success=True,
                error_type=None,
                feature_version="1.0.0",
                user_cohort="test",
                onboarding_completed=True,
                conversion_funnel_stage="activated",
                experiment_variant="A",
                timestamp=datetime.now(timezone.utc),
                metadata={}
            )
            await collector.record_feature_adoption(metric)

        # Generate report
        report = await collector.get_feature_adoption_report(days=7)

        # Verify report structure
        assert "total_adoption_events" in report
        assert "category_analysis" in report
        assert "time_window_days" in report
        assert "generated_at" in report

        assert report["total_adoption_events"] == 3
        assert report["time_window_days"] == 7

    @pytest.mark.asyncio
    async def test_cost_efficiency_report_no_data(self):
        """Test cost efficiency report with no data."""
        collector = BusinessIntelligenceMetricsCollector()

        report = await collector.get_cost_efficiency_report(days=7)

        # Should return no_data status when no metrics
        assert report["status"] == "no_data"
        assert report["days"] == 7

    def test_engagement_score_calculation(self):
        """Test engagement score calculation."""
        collector = BusinessIntelligenceMetricsCollector()

        metric = UserEngagementMetric(
            user_id="user123",
            user_tier=UserTier.PROFESSIONAL,
            session_id="session456",
            session_duration_seconds=300.0,
            pages_viewed=5,
            features_used=["feature1", "feature2"],
            actions_performed=10,
            successful_actions=8,
            time_to_first_action_seconds=5.0,
            bounce_rate_indicator=False,
            return_user=True,
            days_since_last_visit=2,
            total_lifetime_value=100.0,
            current_streak_days=5,
            timestamp=datetime.now(timezone.utc),
            user_agent="test-agent",
            referrer="test-referrer"
        )

        score = collector._calculate_engagement_score(metric)

        # Score should be between 0 and 100
        assert 0 <= score <= 100
        assert isinstance(score, (int, float))
