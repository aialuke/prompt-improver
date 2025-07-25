"""
Tests for Analytics API Endpoints
Tests real behavior with actual API integration and comprehensive endpoint functionality.
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from prompt_improver.api.analytics_endpoints import (
    analytics_router,
    TimeRangeRequest,
    TrendAnalysisRequest,
    SessionComparisonRequest
)
from prompt_improver.database.analytics_query_interface import TimeGranularity, MetricType
from prompt_improver.ml.analytics.session_comparison_analyzer import ComparisonDimension, ComparisonMethod


class TestAnalyticsEndpoints:
    """Test suite for analytics API endpoints with real behavior testing"""

    @pytest.fixture
    def app(self):
        """Create FastAPI test app"""
        app = FastAPI()
        app.include_router(analytics_router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_dashboard_data(self):
        """Mock dashboard metrics data"""
        return {
            "current_period": {
                "session_summary": {
                    "total_sessions": 25,
                    "completed_sessions": 20,
                    "running_sessions": 3,
                    "failed_sessions": 2,
                    "avg_duration_hours": 2.5
                },
                "performance": {
                    "avg_performance": 0.82,
                    "max_performance": 0.95,
                    "min_performance": 0.65,
                    "avg_improvement": 0.15,
                    "performance_std": 0.08
                },
                "efficiency": {
                    "avg_efficiency": 0.35,
                    "avg_iterations": 12.5,
                    "avg_training_hours": 2.3
                },
                "errors": {
                    "session_error_rate": 0.08,
                    "iteration_error_rate": 0.05,
                    "total_failed_sessions": 2,
                    "total_failed_iterations": 15
                },
                "resources": {
                    "avg_memory_usage_mb": 1200.0,
                    "peak_memory_usage_mb": 1800.0,
                    "avg_cpu_utilization": 0.65,
                    "total_compute_hours": 50.0
                }
            },
            "time_range": {
                "start": (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(),
                "end": datetime.now(timezone.utc).isoformat(),
                "hours": 24
            },
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

    @pytest.fixture
    def mock_trend_data(self):
        """Mock trend analysis data"""
        base_time = datetime.now(timezone.utc) - timedelta(days=7)
        return {
            "time_series": [
                {
                    "timestamp": (base_time + timedelta(days=i)).isoformat(),
                    "value": 0.7 + (i * 0.02),
                    "metadata": {"session_count": 5 + i}
                }
                for i in range(7)
            ],
            "trend_direction": "increasing",
            "trend_strength": 0.85,
            "correlation_coefficient": 0.92,
            "seasonal_patterns": {},
            "metadata": {
                "granularity": "day",
                "metric_type": "performance",
                "session_count": None
            }
        }

    @pytest.fixture
    def mock_session_comparison(self):
        """Mock session comparison data"""
        return {
            "session_a_id": "session_a",
            "session_b_id": "session_b",
            "comparison_dimension": "performance",
            "statistical_significance": True,
            "p_value": 0.023,
            "effect_size": 0.65,
            "winner": "session_a",
            "insights": ["Session A shows significantly better performance"],
            "recommendations": ["Adopt configuration from session_a for better performance"]
        }

    def test_get_dashboard_metrics(self, client, mock_dashboard_data):
        """Test dashboard metrics endpoint"""

        from prompt_improver.api.analytics_endpoints import get_analytics_interface

        # Mock the dependency function to return our mock analytics interface
        def mock_get_analytics_interface():
            mock_analytics = AsyncMock()
            mock_analytics.get_dashboard_metrics.return_value = mock_dashboard_data
            return mock_analytics

        # Override the dependency
        client.app.dependency_overrides[get_analytics_interface] = mock_get_analytics_interface

        try:
            response = client.get("/api/v1/analytics/dashboard/metrics?time_range_hours=24&include_comparisons=true")
        finally:
            # Clean up the override
            client.app.dependency_overrides.clear()

            assert response.status_code == 200
            data = response.json()

            assert "current_period" in data
            assert "time_range" in data
            assert "last_updated" in data

            # Verify session summary data
            session_summary = data["current_period"]["session_summary"]
            assert session_summary["total_sessions"] == 25
            assert session_summary["completed_sessions"] == 20

            # Verify performance data
            performance = data["current_period"]["performance"]
            assert performance["avg_performance"] == 0.82

    def test_analyze_performance_trends(self, client, mock_trend_data):
        """Test performance trends analysis endpoint"""

        with patch('prompt_improver.api.analytics_endpoints.get_analytics_interface') as mock_get_analytics:
            mock_analytics = AsyncMock()

            # Mock the trend analysis result
            mock_result = MagicMock()
            mock_result.time_series = [
                MagicMock(
                    timestamp=datetime.fromisoformat(point["timestamp"].replace('Z', '+00:00')),
                    value=point["value"],
                    metadata=point["metadata"]
                )
                for point in mock_trend_data["time_series"]
            ]
            mock_result.trend_direction = mock_trend_data["trend_direction"]
            mock_result.trend_strength = mock_trend_data["trend_strength"]
            mock_result.correlation_coefficient = mock_trend_data["correlation_coefficient"]
            mock_result.seasonal_patterns = mock_trend_data["seasonal_patterns"]

            mock_analytics.get_session_performance_trends.return_value = mock_result
            mock_get_analytics.return_value = mock_analytics

            request_data = {
                "time_range": {"hours": 24},
                "granularity": "day",
                "metric_type": "performance"
            }

            response = client.post("/analytics/trends/analysis", json=request_data)

            assert response.status_code == 200
            data = response.json()

            assert "time_series" in data
            assert "trend_direction" in data
            assert "trend_strength" in data
            assert "correlation_coefficient" in data

            assert data["trend_direction"] == "increasing"
            assert data["trend_strength"] == 0.85
            assert len(data["time_series"]) == 7

    def test_compare_sessions(self, client, mock_session_comparison):
        """Test session comparison endpoint"""

        with patch('prompt_improver.api.analytics_endpoints.get_comparison_analyzer') as mock_get_analyzer:
            mock_analyzer = AsyncMock()

            # Mock the comparison result
            mock_result = MagicMock()
            mock_result.session_a_id = mock_session_comparison["session_a_id"]
            mock_result.session_b_id = mock_session_comparison["session_b_id"]
            mock_result.comparison_dimension = MagicMock(value=mock_session_comparison["comparison_dimension"])
            mock_result.statistical_significance = mock_session_comparison["statistical_significance"]
            mock_result.p_value = mock_session_comparison["p_value"]
            mock_result.effect_size = mock_session_comparison["effect_size"]
            mock_result.winner = mock_session_comparison["winner"]
            mock_result.insights = mock_session_comparison["insights"]
            mock_result.recommendations = mock_session_comparison["recommendations"]

            mock_analyzer.compare_sessions.return_value = mock_result
            mock_get_analyzer.return_value = mock_analyzer

            request_data = {
                "session_a_id": "session_a",
                "session_b_id": "session_b",
                "dimension": "performance",
                "method": "t_test"
            }

            response = client.post("/analytics/sessions/compare", json=request_data)

            assert response.status_code == 200
            data = response.json()

            assert data["session_a_id"] == "session_a"
            assert data["session_b_id"] == "session_b"
            assert data["comparison_dimension"] == "performance"
            assert data["statistical_significance"] == True
            assert data["winner"] == "session_a"
            assert len(data["insights"]) > 0
            assert len(data["recommendations"]) > 0

    def test_get_session_summary(self, client):
        """Test session summary endpoint"""

        with patch('prompt_improver.api.analytics_endpoints.get_session_reporter') as mock_get_reporter:
            mock_reporter = AsyncMock()

            # Mock session summary
            mock_summary = MagicMock()
            mock_summary.session_id = "test_session_1"
            mock_summary.status = "completed"
            mock_summary.performance_score = 0.85
            mock_summary.improvement_velocity = 0.75
            mock_summary.efficiency_rating = 0.80
            mock_summary.quality_index = 0.90
            mock_summary.success_rate = 0.88
            mock_summary.initial_performance = 0.7
            mock_summary.final_performance = 0.85
            mock_summary.best_performance = 0.87
            mock_summary.total_improvement = 0.15
            mock_summary.improvement_rate = 0.075
            mock_summary.performance_trend = "improving"
            mock_summary.total_iterations = 15
            mock_summary.successful_iterations = 14
            mock_summary.failed_iterations = 1
            mock_summary.average_iteration_duration = 480.0
            mock_summary.key_insights = ["Strong performance improvement achieved"]
            mock_summary.recommendations = ["Continue current approach"]
            mock_summary.anomalies_detected = []
            mock_summary.started_at = datetime.now(timezone.utc) - timedelta(hours=2)
            mock_summary.completed_at = datetime.now(timezone.utc)
            mock_summary.total_duration_hours = 2.0
            mock_summary.configuration = {"continuous_mode": True}

            mock_reporter.generate_session_summary.return_value = mock_summary
            mock_get_reporter.return_value = mock_reporter

            response = client.get("/analytics/sessions/test_session_1/summary")

            assert response.status_code == 200
            data = response.json()

            assert data["session_id"] == "test_session_1"
            assert data["status"] == "completed"

            # Verify executive KPIs
            kpis = data["executive_kpis"]
            assert kpis["performance_score"] == 0.85
            assert kpis["improvement_velocity"] == 0.75

            # Verify performance metrics
            perf_metrics = data["performance_metrics"]
            assert perf_metrics["total_improvement"] == 0.15
            assert perf_metrics["performance_trend"] == "improving"

            # Verify insights
            insights = data["insights"]
            assert len(insights["key_insights"]) > 0
            assert len(insights["recommendations"]) > 0

    def test_export_session_report(self, client):
        """Test session report export endpoint"""

        with patch('prompt_improver.api.analytics_endpoints.get_session_reporter') as mock_get_reporter:
            mock_reporter = AsyncMock()
            mock_reporter.export_session_report.return_value = "/tmp/session_report_test.json"
            mock_get_reporter.return_value = mock_reporter

            response = client.get("/analytics/sessions/test_session_1/export?format=json")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "success"
            assert data["export_path"] == "/tmp/session_report_test.json"
            assert data["format"] == "json"
            assert data["session_id"] == "test_session_1"
            assert "exported_at" in data

    def test_get_performance_distribution(self, client):
        """Test performance distribution endpoint"""

        with patch('prompt_improver.api.analytics_endpoints.get_analytics_interface') as mock_get_analytics:
            mock_analytics = AsyncMock()

            mock_distribution_data = {
                "histogram": [
                    {"bucket": 1, "frequency": 2, "min_value": 0.5, "max_value": 0.6, "probability": 0.1},
                    {"bucket": 2, "frequency": 5, "min_value": 0.6, "max_value": 0.7, "probability": 0.25},
                    {"bucket": 3, "frequency": 8, "min_value": 0.7, "max_value": 0.8, "probability": 0.4},
                    {"bucket": 4, "frequency": 4, "min_value": 0.8, "max_value": 0.9, "probability": 0.2},
                    {"bucket": 5, "frequency": 1, "min_value": 0.9, "max_value": 1.0, "probability": 0.05}
                ],
                "statistics": {
                    "total_sessions": 20,
                    "mean": 0.72,
                    "std_dev": 0.12,
                    "quartiles": {"q25": 0.65, "median": 0.72, "q75": 0.80},
                    "percentiles": {"p90": 0.85, "p95": 0.90}
                }
            }

            mock_analytics.get_performance_distribution_analysis.return_value = mock_distribution_data
            mock_get_analytics.return_value = mock_analytics

            response = client.get("/analytics/distribution/performance?bucket_count=20")

            assert response.status_code == 200
            data = response.json()

            assert "histogram" in data
            assert "statistics" in data

            histogram = data["histogram"]
            assert len(histogram) == 5
            assert histogram[0]["frequency"] == 2

            stats = data["statistics"]
            assert stats["total_sessions"] == 20
            assert stats["mean"] == 0.72

    def test_get_correlation_analysis(self, client):
        """Test correlation analysis endpoint"""

        with patch('prompt_improver.api.analytics_endpoints.get_analytics_interface') as mock_get_analytics:
            mock_analytics = AsyncMock()

            mock_correlation_data = {
                "correlations": {
                    "performance_vs_improvement": 0.65,
                    "performance_vs_duration": -0.25,
                    "performance_vs_success_rate": 0.45
                },
                "interpretations": {
                    "performance_vs_improvement": {
                        "strength": "moderate",
                        "direction": "positive",
                        "value": 0.65
                    }
                },
                "sample_size": 50
            }

            mock_analytics.get_correlation_analysis.return_value = mock_correlation_data
            mock_get_analytics.return_value = mock_analytics

            response = client.get("/analytics/correlation/analysis?metrics=performance&metrics=improvement&metrics=duration")

            assert response.status_code == 200
            data = response.json()

            assert "correlations" in data
            assert "interpretations" in data
            assert "sample_size" in data

            correlations = data["correlations"]
            assert "performance_vs_improvement" in correlations
            assert correlations["performance_vs_improvement"] == 0.65

            assert data["sample_size"] == 50

    def test_analytics_health_check(self, client):
        """Test analytics health check endpoint"""

        with patch('prompt_improver.api.analytics_endpoints.connection_manager') as mock_connection_manager:
            mock_connection_manager.active_connections = []

            response = client.get("/analytics/health")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert "services" in data

            services = data["services"]
            assert services["analytics_query_interface"] == "operational"
            assert services["session_reporter"] == "operational"
            assert services["comparison_analyzer"] == "operational"
            assert "websocket_connections" in services

    def test_get_dashboard_config(self, client):
        """Test dashboard configuration endpoint"""

        response = client.get("/api/v1/analytics/dashboard/config")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert "config" in data

        config = data["config"]
        assert "dashboard_settings" in config
        assert "websocket_endpoints" in config
        assert "api_endpoints" in config
        assert "user_roles" in config
        assert "comparison_dimensions" in config
        assert "metric_types" in config
        assert "time_granularities" in config

        # Verify dashboard settings
        dashboard_settings = config["dashboard_settings"]
        assert dashboard_settings["auto_refresh"] == True
        assert dashboard_settings["refresh_interval"] == 30

        # Verify API endpoints
        api_endpoints = config["api_endpoints"]
        assert "dashboard_metrics" in api_endpoints
        assert "trend_analysis" in api_endpoints
        assert "session_comparison" in api_endpoints

    def test_error_handling(self, client):
        """Test error handling in endpoints"""

        with patch('prompt_improver.api.analytics_endpoints.get_analytics_interface') as mock_get_analytics:
            mock_analytics = AsyncMock()
            mock_analytics.get_dashboard_metrics.side_effect = Exception("Database connection error")
            mock_get_analytics.return_value = mock_analytics

            response = client.get("/analytics/dashboard/metrics")

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Failed to retrieve dashboard metrics" in data["detail"]

    def test_request_validation(self, client):
        """Test request validation for POST endpoints"""

        # Test invalid trend analysis request
        invalid_request = {
            "time_range": {"hours": -1},  # Invalid negative hours
            "granularity": "invalid_granularity"
        }

        response = client.post("/analytics/trends/analysis", json=invalid_request)
        assert response.status_code == 422  # Validation error

        # Test invalid session comparison request
        invalid_comparison = {
            "session_a_id": "",  # Empty session ID
            "session_b_id": "session_b"
        }

        response = client.post("/analytics/sessions/compare", json=invalid_comparison)
        assert response.status_code == 422  # Validation error

    def test_query_parameters(self, client, mock_dashboard_data):
        """Test query parameter handling"""

        with patch('prompt_improver.api.analytics_endpoints.get_analytics_interface') as mock_get_analytics:
            mock_analytics = AsyncMock()
            mock_analytics.get_dashboard_metrics.return_value = mock_dashboard_data
            mock_get_analytics.return_value = mock_analytics

            # Test with different time ranges
            response = client.get("/analytics/dashboard/metrics?time_range_hours=168")  # 1 week
            assert response.status_code == 200

            # Test with comparisons disabled
            response = client.get("/analytics/dashboard/metrics?include_comparisons=false")
            assert response.status_code == 200

            # Test with invalid time range (should be clamped)
            response = client.get("/analytics/dashboard/metrics?time_range_hours=10000")  # Too large
            assert response.status_code == 422  # Validation error
