"""Smoke tests for decomposed database health monitoring services.

These tests verify that the new services can be instantiated and basic
operations work without requiring a real database connection.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from prompt_improver.database.health.services import (
    AlertingService,
    DatabaseConnectionService,
    DatabaseHealthService,
    HealthMetricsService,
    HealthReportingService,
    create_database_health_service,
)


class MockSessionManager:
    """Mock session manager for testing."""

    async def health_check(self):
        return True

    async def get_connection_info(self):
        return {
            "pool_size": 10,
            "pool_min_size": 2,
            "pool_max_size": 20,
            "pool_timeout": 30,
            "requests_waiting": 0,
        }

    def session_context(self):
        return AsyncMock()


@pytest.fixture
def mock_session_manager():
    return MockSessionManager()


class TestServiceInstantiation:
    """Test that all services can be instantiated correctly."""

    def test_database_connection_service_creation(self, mock_session_manager):
        """Test DatabaseConnectionService can be created."""
        service = DatabaseConnectionService(mock_session_manager)
        assert service.session_manager == mock_session_manager
        assert service.max_connection_lifetime_seconds == 1800
        assert service.utilization_warning_threshold == 80.0

    def test_health_metrics_service_creation(self, mock_session_manager):
        """Test HealthMetricsService can be created."""
        service = HealthMetricsService(mock_session_manager)
        assert service.session_manager == mock_session_manager
        assert service.slow_query_threshold_ms == 1000.0
        assert service.cache_hit_ratio_threshold == 95.0

    def test_alerting_service_creation(self):
        """Test AlertingService can be created."""
        service = AlertingService()
        assert len(service.thresholds) > 0
        assert "connection_pool_utilization" in service.thresholds
        assert "slow_queries_count" in service.thresholds

    def test_health_reporting_service_creation(self):
        """Test HealthReportingService can be created."""
        service = HealthReportingService()
        assert service._metrics_history == []
        assert service._max_history_size == 1000

    def test_database_health_service_creation(self, mock_session_manager):
        """Test unified DatabaseHealthService can be created."""
        service = DatabaseHealthService(mock_session_manager)
        assert service.session_manager == mock_session_manager
        assert isinstance(service.connection_service, DatabaseConnectionService)
        assert isinstance(service.metrics_service, HealthMetricsService)
        assert isinstance(service.alerting_service, AlertingService)
        assert isinstance(service.reporting_service, HealthReportingService)

    def test_create_database_health_service_factory(self, mock_session_manager):
        """Test factory function works."""
        service = create_database_health_service(mock_session_manager)
        assert isinstance(service, DatabaseHealthService)


class TestAlertingService:
    """Test AlertingService functionality without database."""

    @pytest.fixture
    def alerting_service(self):
        return AlertingService()

    def test_calculate_health_score_good_metrics(self, alerting_service):
        """Test health score calculation with good metrics."""
        metrics = {
            "connection_pool": {"utilization_percent": 50.0},
            "query_performance": {"slow_queries_count": 0},
            "cache": {"overall_cache_hit_ratio_percent": 98.0},
            "replication": {"replication_enabled": False},
            "locks": {"blocking_locks": 0, "long_running_locks": 0},
            "transactions": {"rollback_ratio_percent": 2.0},
        }

        score = alerting_service.calculate_health_score(metrics)
        assert isinstance(score, float)
        assert 80 <= score <= 100

    def test_calculate_health_score_bad_metrics(self, alerting_service):
        """Test health score calculation with poor metrics."""
        metrics = {
            "connection_pool": {"utilization_percent": 98.0},
            "query_performance": {"slow_queries_count": 20},
            "cache": {"overall_cache_hit_ratio_percent": 80.0},
            "replication": {"replication_enabled": True, "lag_seconds": 500.0},
            "locks": {"blocking_locks": 5, "long_running_locks": 3},
            "transactions": {"rollback_ratio_percent": 30.0},
        }

        score = alerting_service.calculate_health_score(metrics)
        assert isinstance(score, float)
        assert 0 <= score < 50

    def test_identify_health_issues(self, alerting_service):
        """Test health issues identification."""
        metrics = {
            "connection_pool": {"utilization_percent": 92.0},
            "query_performance": {"slow_queries_count": 8},
            "cache": {"overall_cache_hit_ratio_percent": 88.0},
        }

        issues = alerting_service.identify_health_issues(metrics)
        assert isinstance(issues, list)
        assert len(issues) > 0

        # Verify issue structure
        for issue in issues:
            assert "severity" in issue
            assert "category" in issue
            assert "message" in issue
            assert "timestamp" in issue

    def test_generate_recommendations(self, alerting_service):
        """Test recommendation generation."""
        metrics = {
            "connection_pool": {"utilization_percent": 85.0},
            "query_performance": {"slow_queries_count": 5},
            "cache": {"overall_cache_hit_ratio_percent": 92.0},
        }

        recommendations = alerting_service.generate_recommendations(metrics)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Verify recommendation structure
        for rec in recommendations:
            assert "category" in rec
            assert "priority" in rec
            assert "action" in rec
            assert "description" in rec
            assert "expected_impact" in rec

    def test_set_threshold(self, alerting_service):
        """Test threshold setting."""
        alerting_service.set_threshold("test_metric", 70.0, 90.0)

        assert "test_metric" in alerting_service.thresholds
        threshold = alerting_service.thresholds["test_metric"]
        assert threshold.warning_threshold == 70.0
        assert threshold.critical_threshold == 90.0


class TestHealthReportingService:
    """Test HealthReportingService functionality."""

    @pytest.fixture
    def reporting_service(self):
        return HealthReportingService()

    def test_add_metrics_to_history(self, reporting_service):
        """Test adding metrics to history."""
        metrics = {
            "timestamp": datetime.now(UTC).isoformat(),
            "health_score": 85.0,
            "connection_pool": {"utilization_percent": 60.0},
        }

        reporting_service.add_metrics_to_history(metrics)
        history = reporting_service.get_metrics_history()

        assert len(history) == 1
        assert history[0]["health_score"] == 85.0

    def test_generate_health_report(self, reporting_service):
        """Test health report generation."""
        metrics = {
            "timestamp": datetime.now(UTC).isoformat(),
            "health_score": 78.0,
            "issues": [
                {"severity": "warning", "category": "cache", "message": "Low hit ratio"}
            ],
            "recommendations": [
                {"category": "cache", "priority": "medium", "action": "tune_cache"}
            ],
            "connection_pool": {"utilization_percent": 75.0},
        }

        report = reporting_service.generate_health_report(metrics)

        assert isinstance(report, dict)
        assert "report_metadata" in report
        assert "executive_summary" in report
        assert "component_health" in report

        # Verify executive summary
        exec_summary = report["executive_summary"]
        assert exec_summary["overall_health_score"] == 78.0
        assert exec_summary["overall_status"] == "good"

    def test_export_metrics_json(self, reporting_service):
        """Test JSON metrics export."""
        metrics = {"timestamp": datetime.now(UTC).isoformat(), "health_score": 90.0}
        reporting_service.add_metrics_to_history(metrics)

        json_export = reporting_service.export_metrics("json")
        assert isinstance(json_export, str)
        assert "health_score" in json_export

    def test_export_metrics_csv(self, reporting_service):
        """Test CSV metrics export."""
        metrics = {"timestamp": datetime.now(UTC).isoformat(), "health_score": 90.0}
        reporting_service.add_metrics_to_history(metrics)

        csv_export = reporting_service.export_metrics("csv")
        assert isinstance(csv_export, str)
        assert "timestamp,health_score" in csv_export

    def test_get_health_trends_insufficient_data(self, reporting_service):
        """Test trend analysis with insufficient data."""
        trends = reporting_service.get_health_trends(hours=24)

        assert trends["status"] == "insufficient_data"
        assert "data_points_found" in trends


class TestProtocolCompliance:
    """Test that services implement their protocols correctly."""

    def test_connection_service_protocol_compliance(self, mock_session_manager):
        """Test DatabaseConnectionService implements its protocol."""
        from prompt_improver.shared.interfaces.protocols.database import (
            DatabaseConnectionServiceProtocol,
        )

        service = DatabaseConnectionService(mock_session_manager)
        assert isinstance(service, DatabaseConnectionServiceProtocol)

    def test_metrics_service_protocol_compliance(self, mock_session_manager):
        """Test HealthMetricsService implements its protocol."""
        from prompt_improver.shared.interfaces.protocols.database import (
            HealthMetricsServiceProtocol,
        )

        service = HealthMetricsService(mock_session_manager)
        assert isinstance(service, HealthMetricsServiceProtocol)

    def test_alerting_service_protocol_compliance(self):
        """Test AlertingService implements its protocol."""
        from prompt_improver.shared.interfaces.protocols.database import (
            AlertingServiceProtocol,
        )

        service = AlertingService()
        assert isinstance(service, AlertingServiceProtocol)

    def test_reporting_service_protocol_compliance(self):
        """Test HealthReportingService implements its protocol."""
        from prompt_improver.shared.interfaces.protocols.database import (
            HealthReportingServiceProtocol,
        )

        service = HealthReportingService()
        assert isinstance(service, HealthReportingServiceProtocol)


class TestBackwardCompatibility:
    """Test backward compatibility with existing interfaces."""

    def test_unified_service_has_legacy_methods(self, mock_session_manager):
        """Test that DatabaseHealthService has all legacy methods."""
        service = DatabaseHealthService(mock_session_manager)

        # Check for backward compatibility methods
        assert hasattr(service, 'calculate_health_score')
        assert hasattr(service, 'identify_health_issues')
        assert hasattr(service, 'generate_recommendations')
        assert hasattr(service, 'add_to_history')
        assert hasattr(service, 'get_metrics_history')

        # These should be callable
        assert callable(service.calculate_health_score)
        assert callable(service.identify_health_issues)
        assert callable(service.generate_recommendations)

    @pytest.mark.asyncio
    async def test_unified_service_async_methods(self, mock_session_manager):
        """Test that async methods exist and are callable."""
        service = DatabaseHealthService(mock_session_manager)

        # Check async methods exist
        assert hasattr(service, 'collect_comprehensive_metrics')
        assert hasattr(service, 'get_comprehensive_health')
        assert hasattr(service, 'health_check')
        assert hasattr(service, 'get_connection_pool_health_summary')
        assert hasattr(service, 'analyze_query_performance')

        # These should be async callables
        assert callable(service.collect_comprehensive_metrics)
        assert callable(service.get_comprehensive_health)
        assert callable(service.health_check)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
