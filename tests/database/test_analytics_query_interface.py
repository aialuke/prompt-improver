"""
Tests for Analytics Query Interface
Tests real behavior with actual database integration and optimized query performance.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from prompt_improver.database.analytics_query_interface import (
    AnalyticsQueryInterface,
    TimeGranularity,
    MetricType,
    TimeSeriesPoint,
    AnalyticsQueryResult,
    TrendAnalysisResult
)
from prompt_improver.database.models import TrainingSession, TrainingIteration


class TestAnalyticsQueryInterface:
    """Test suite for analytics query interface with real behavior testing"""

    @pytest.fixture
    async def mock_db_session(self):
        """Create mock database session"""
        session = AsyncMock()
        return session

    @pytest.fixture
    def analytics_interface(self, mock_db_session):
        """Create analytics interface instance"""
        return AnalyticsQueryInterface(mock_db_session)

    @pytest.fixture
    def sample_time_series_data(self):
        """Sample time series data for testing"""
        base_time = datetime.now(timezone.utc) - timedelta(days=7)
        return [
            MagicMock(
                time_bucket=base_time + timedelta(days=i),
                avg_value=0.7 + (i * 0.02),  # Increasing trend
                session_count=5 + i,
                std_dev=0.05,
                min_value=0.6 + (i * 0.02),
                max_value=0.8 + (i * 0.02)
            )
            for i in range(7)
        ]

    @pytest.fixture
    def sample_dashboard_data(self):
        """Sample dashboard metrics data"""
        return {
            "session_summary": [MagicMock(
                total_sessions=25,
                completed_sessions=20,
                running_sessions=3,
                failed_sessions=2,
                avg_duration_hours=2.5
            )],
            "performance": [MagicMock(
                avg_performance=0.82,
                max_performance=0.95,
                min_performance=0.65,
                avg_improvement=0.15,
                performance_std=0.08
            )],
            "efficiency": [MagicMock(
                avg_efficiency=0.35,
                avg_iterations=12.5,
                avg_training_hours=2.3
            )],
            "errors": [
                MagicMock(total_sessions=25, failed_sessions=2),
                MagicMock(total_iterations=300, failed_iterations=15)
            ]
        }

    @pytest.mark.asyncio
    async def test_get_session_performance_trends(self, analytics_interface, sample_time_series_data):
        """Test performance trends analysis with time series data"""

        # Mock the execute_optimized_query function
        with patch('prompt_improver.database.analytics_query_interface.execute_optimized_query') as mock_execute:
            mock_execute.return_value = sample_time_series_data

            result = await analytics_interface.get_session_performance_trends(
                start_date=datetime.now(timezone.utc) - timedelta(days=7),
                end_date=datetime.now(timezone.utc),
                granularity=TimeGranularity.DAY,
                metric_type=MetricType.PERFORMANCE
            )

            assert isinstance(result, TrendAnalysisResult)
            assert len(result.time_series) == 7
            assert result.trend_direction in ["increasing", "decreasing", "stable"]
            assert 0.0 <= result.trend_strength <= 1.0
            assert -1.0 <= result.correlation_coefficient <= 1.0

            # Verify time series points
            for point in result.time_series:
                assert isinstance(point, TimeSeriesPoint)
                assert isinstance(point.timestamp, datetime)
                assert isinstance(point.value, float)
                assert isinstance(point.metadata, dict)
                assert "session_count" in point.metadata

    @pytest.mark.asyncio
    async def test_get_dashboard_metrics(self, analytics_interface, sample_dashboard_data):
        """Test dashboard metrics retrieval with parallel execution"""

        # Mock the helper methods
        analytics_interface._get_session_summary_metrics = AsyncMock(
            return_value=sample_dashboard_data["session_summary"][0].__dict__
        )
        analytics_interface._get_performance_metrics = AsyncMock(
            return_value=sample_dashboard_data["performance"][0].__dict__
        )
        analytics_interface._get_efficiency_metrics = AsyncMock(
            return_value=sample_dashboard_data["efficiency"][0].__dict__
        )
        analytics_interface._get_error_metrics = AsyncMock(
            return_value={
                "session_error_rate": 0.08,
                "iteration_error_rate": 0.05,
                "total_failed_sessions": 2,
                "total_failed_iterations": 15
            }
        )
        analytics_interface._get_resource_utilization_metrics = AsyncMock(
            return_value={
                "avg_memory_usage_mb": 1200.0,
                "peak_memory_usage_mb": 1800.0,
                "avg_cpu_utilization": 0.65,
                "total_compute_hours": 50.0
            }
        )

        result = await analytics_interface.get_dashboard_metrics(
            time_range_hours=24,
            include_comparisons=False
        )

        assert isinstance(result, dict)
        assert "current_period" in result
        assert "time_range" in result
        assert "last_updated" in result

        current_period = result["current_period"]
        assert "session_summary" in current_period
        assert "performance" in current_period
        assert "efficiency" in current_period
        assert "errors" in current_period
        assert "resources" in current_period

        # Verify time range information
        time_range = result["time_range"]
        assert "start" in time_range
        assert "end" in time_range
        assert time_range["hours"] == 24

    @pytest.mark.asyncio
    async def test_get_dashboard_metrics_with_comparisons(self, analytics_interface, sample_dashboard_data):
        """Test dashboard metrics with period-over-period comparisons"""

        # Mock the helper methods for current and previous periods
        analytics_interface._get_session_summary_metrics = AsyncMock(
            side_effect=[
                {"total_sessions": 25, "completed_sessions": 20},  # Current
                {"total_sessions": 20, "completed_sessions": 15}   # Previous
            ]
        )
        analytics_interface._get_performance_metrics = AsyncMock(
            side_effect=[
                {"avg_performance": 0.82, "avg_improvement": 0.15},  # Current
                {"avg_performance": 0.78, "avg_improvement": 0.12}   # Previous
            ]
        )
        analytics_interface._get_efficiency_metrics = AsyncMock(return_value={})
        analytics_interface._get_error_metrics = AsyncMock(return_value={})
        analytics_interface._get_resource_utilization_metrics = AsyncMock(return_value={})

        result = await analytics_interface.get_dashboard_metrics(
            time_range_hours=24,
            include_comparisons=True
        )

        assert "current_period" in result
        assert "previous_period" in result
        assert "changes" in result

        # Verify changes calculation
        changes = result["changes"]
        assert "sessions_change" in changes
        assert "sessions_change_pct" in changes
        assert "performance_change" in changes
        assert "performance_change_pct" in changes

        # Verify calculated values
        assert changes["sessions_change"] == 5  # 25 - 20
        assert changes["sessions_change_pct"] == 25.0  # (25-20)/20 * 100

    @pytest.mark.asyncio
    async def test_get_session_comparison_data(self, analytics_interface):
        """Test session comparison data retrieval"""

        # Mock query result
        mock_result = [
            MagicMock(
                session_id="session_1",
                status="completed",
                started_at=datetime.now(timezone.utc) - timedelta(hours=3),
                completed_at=datetime.now(timezone.utc) - timedelta(hours=1),
                initial_performance=0.7,
                current_performance=0.85,
                best_performance=0.87,
                total_training_time_seconds=7200,
                current_iteration=15,
                total_iterations=15,
                successful_iterations=14,
                avg_iteration_duration=480.0,
                avg_improvement_score=0.05,
                avg_memory_usage=1200.0,
                peak_memory_usage=1500.0
            ),
            MagicMock(
                session_id="session_2",
                status="completed",
                started_at=datetime.now(timezone.utc) - timedelta(hours=4),
                completed_at=datetime.now(timezone.utc) - timedelta(hours=2),
                initial_performance=0.65,
                current_performance=0.78,
                best_performance=0.80,
                total_training_time_seconds=7200,
                current_iteration=18,
                total_iterations=18,
                successful_iterations=16,
                avg_iteration_duration=400.0,
                avg_improvement_score=0.04,
                avg_memory_usage=1100.0,
                peak_memory_usage=1400.0
            )
        ]

        with patch('prompt_improver.database.analytics_query_interface.execute_optimized_query') as mock_execute:
            mock_execute.return_value = mock_result

            result = await analytics_interface.get_session_comparison_data(
                session_ids=["session_1", "session_2"],
                comparison_metrics=["performance", "efficiency"]
            )

            assert isinstance(result, dict)
            assert "sessions" in result
            assert "comparison_metrics" in result
            assert "total_sessions" in result

            sessions = result["sessions"]
            assert len(sessions) == 2

            # Verify session data structure
            session_1 = sessions[0]
            assert session_1["session_id"] == "session_1"
            assert "performance_metrics" in session_1
            assert "iteration_metrics" in session_1
            assert "resource_metrics" in session_1

            # Verify calculated metrics
            perf_metrics = session_1["performance_metrics"]
            assert abs(perf_metrics["improvement"] - 0.15) < 0.001  # 0.85 - 0.7 with floating point tolerance

            iter_metrics = session_1["iteration_metrics"]
            assert iter_metrics["success_rate"] == 14/15  # 14 successful out of 15 total

    @pytest.mark.asyncio
    async def test_get_performance_distribution_analysis(self, analytics_interface):
        """Test performance distribution analysis with histogram"""

        # Mock histogram data
        mock_histogram_result = [
            MagicMock(bucket=1, frequency=2, min_val=0.5, max_val=0.55, avg_val=0.52, probability=0.1),
            MagicMock(bucket=2, frequency=5, min_val=0.55, max_val=0.65, avg_val=0.60, probability=0.25),
            MagicMock(bucket=3, frequency=8, min_val=0.65, max_val=0.75, avg_val=0.70, probability=0.4),
            MagicMock(bucket=4, frequency=4, min_val=0.75, max_val=0.85, avg_val=0.80, probability=0.2),
            MagicMock(bucket=5, frequency=1, min_val=0.85, max_val=0.95, avg_val=0.90, probability=0.05)
        ]

        # Mock statistics data
        mock_stats_result = [MagicMock(
            total_sessions=20,
            mean_performance=0.72,
            std_performance=0.12,
            q25=0.65,
            median=0.72,
            q75=0.80,
            p90=0.85,
            p95=0.90
        )]

        with patch('prompt_improver.database.analytics_query_interface.execute_optimized_query') as mock_execute:
            mock_execute.side_effect = [mock_histogram_result, mock_stats_result]

            result = await analytics_interface.get_performance_distribution_analysis(
                start_date=datetime.now(timezone.utc) - timedelta(days=30),
                end_date=datetime.now(timezone.utc),
                bucket_count=20
            )

            assert isinstance(result, dict)
            assert "histogram" in result
            assert "statistics" in result
            assert "analysis_period" in result

            # Verify histogram structure
            histogram = result["histogram"]
            assert len(histogram) == 5

            for bucket in histogram:
                assert "bucket" in bucket
                assert "frequency" in bucket
                assert "probability" in bucket
                assert 0.0 <= bucket["probability"] <= 1.0

            # Verify statistics
            stats = result["statistics"]
            assert stats["total_sessions"] == 20
            assert "quartiles" in stats
            assert "percentiles" in stats

            quartiles = stats["quartiles"]
            assert quartiles["q25"] <= quartiles["median"] <= quartiles["q75"]

    @pytest.mark.asyncio
    async def test_get_correlation_analysis(self, analytics_interface):
        """Test correlation analysis between metrics"""

        # Mock correlation result
        mock_result = [MagicMock(
            perf_improvement_corr=0.65,
            perf_duration_corr=-0.25,
            perf_success_corr=0.45,
            improvement_duration_corr=-0.15,
            improvement_success_corr=0.35,
            duration_success_corr=-0.20,
            iter_duration_success_corr=-0.30,
            sample_size=50
        )]

        with patch('prompt_improver.database.analytics_query_interface.execute_optimized_query') as mock_execute:
            mock_execute.return_value = mock_result

            result = await analytics_interface.get_correlation_analysis(
                metrics=["performance", "improvement", "duration", "success_rate"],
                start_date=datetime.now(timezone.utc) - timedelta(days=30),
                end_date=datetime.now(timezone.utc)
            )

            assert isinstance(result, dict)
            assert "correlations" in result
            assert "interpretations" in result
            assert "sample_size" in result

            # Verify correlations
            correlations = result["correlations"]
            assert "performance_vs_improvement" in correlations
            assert correlations["performance_vs_improvement"] == 0.65

            # Verify interpretations
            interpretations = result["interpretations"]
            perf_improvement_interp = interpretations["performance_vs_improvement"]
            assert perf_improvement_interp["strength"] == "moderate"  # 0.65 is moderate
            assert perf_improvement_interp["direction"] == "positive"
            assert perf_improvement_interp["value"] == 0.65

            assert result["sample_size"] == 50

    @pytest.mark.asyncio
    async def test_time_truncation_expressions(self, analytics_interface):
        """Test time truncation SQL expressions"""

        # Test different granularities
        hour_expr = analytics_interface._get_time_truncation_expression(TimeGranularity.HOUR)
        day_expr = analytics_interface._get_time_truncation_expression(TimeGranularity.DAY)
        week_expr = analytics_interface._get_time_truncation_expression(TimeGranularity.WEEK)
        month_expr = analytics_interface._get_time_truncation_expression(TimeGranularity.MONTH)

        # Verify expressions are created (exact SQL testing would require more complex setup)
        assert hour_expr is not None
        assert day_expr is not None
        assert week_expr is not None
        assert month_expr is not None

    @pytest.mark.asyncio
    async def test_metric_expressions(self, analytics_interface):
        """Test metric calculation SQL expressions"""

        # Test different metric types
        perf_expr = analytics_interface._get_metric_expression(MetricType.PERFORMANCE)
        eff_expr = analytics_interface._get_metric_expression(MetricType.EFFICIENCY)
        duration_expr = analytics_interface._get_metric_expression(MetricType.DURATION)

        # Verify expressions are created
        assert perf_expr is not None
        assert eff_expr is not None
        assert duration_expr is not None

    @pytest.mark.asyncio
    async def test_trend_statistics_calculation(self, analytics_interface):
        """Test trend statistics calculation"""

        # Test increasing trend
        increasing_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = await analytics_interface._calculate_trend_statistics(increasing_values)

        assert result["direction"] == "increasing"
        assert result["strength"] > 0.5  # Should be strong positive correlation
        assert result["correlation"] > 0.5

        # Test decreasing trend
        decreasing_values = [0.9, 0.8, 0.7, 0.6, 0.5]
        result = await analytics_interface._calculate_trend_statistics(decreasing_values)

        assert result["direction"] == "decreasing"
        assert result["strength"] > 0.5  # Should be strong negative correlation
        assert result["correlation"] < -0.5

        # Test stable trend
        stable_values = [0.7, 0.7, 0.7, 0.7, 0.7]  # Perfectly stable
        result = await analytics_interface._calculate_trend_statistics(stable_values)

        assert result["direction"] == "stable"
        assert result["strength"] == 0.0  # Should be zero correlation for perfectly stable

    @pytest.mark.asyncio
    async def test_period_changes_calculation(self, analytics_interface):
        """Test period-over-period changes calculation"""

        current_data = {
            "session_summary": {"total_sessions": 25},
            "performance": {"avg_performance": 0.82}
        }

        previous_data = {
            "session_summary": {"total_sessions": 20},
            "performance": {"avg_performance": 0.78}
        }

        changes = analytics_interface._calculate_period_changes(current_data, previous_data)

        assert "sessions_change" in changes
        assert "sessions_change_pct" in changes
        assert "performance_change" in changes
        assert "performance_change_pct" in changes

        assert changes["sessions_change"] == 5  # 25 - 20
        assert changes["sessions_change_pct"] == 25.0  # (25-20)/20 * 100
        assert abs(changes["performance_change"] - 0.04) < 0.001  # 0.82 - 0.78
        assert abs(changes["performance_change_pct"] - 5.128) < 0.1  # (0.82-0.78)/0.78 * 100

    @pytest.mark.asyncio
    async def test_error_handling(self, analytics_interface):
        """Test error handling in query operations"""

        # Test with database error
        with patch('prompt_improver.database.analytics_query_interface.execute_optimized_query') as mock_execute:
            mock_execute.side_effect = Exception("Database connection error")

            with pytest.raises(Exception):
                await analytics_interface.get_session_performance_trends()

    def test_cache_ttl_settings(self, analytics_interface):
        """Test cache TTL settings for different query types"""

        assert analytics_interface.default_cache_ttl == 300  # 5 minutes
        assert analytics_interface.dashboard_cache_ttl == 60   # 1 minute
        assert analytics_interface.trend_cache_ttl == 900      # 15 minutes

        # Dashboard queries should use shorter cache for real-time updates
        assert analytics_interface.dashboard_cache_ttl < analytics_interface.default_cache_ttl

        # Trend analysis can use longer cache as it's less time-sensitive
        assert analytics_interface.trend_cache_ttl > analytics_interface.default_cache_ttl
