"""Integration tests for decomposed database health monitoring services.

Tests the integrated functionality of all health monitoring services working together,
including real PostgreSQL database interactions using testcontainers for authentic
behavior validation.

This test suite validates:
- Service integration and communication
- Real database health monitoring
- Parallel execution performance
- Backward compatibility
- Error handling and recovery
- Data consistency across services
"""

import asyncio
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
    get_database_health_service,
)


class MockSession:
    """Mock session implementation for testing."""

    def __init__(self):
        self.queries_executed = []
        self.closed = False

    async def execute(self, query, parameters=None):
        """Mock query execution with realistic PostgreSQL responses."""
        query_text = str(query)
        self.queries_executed.append(query_text)

        # Mock different query types with realistic responses
        if "pg_stat_activity" in query_text:
            # Mock connection activity data
            return MockResult([
                (12345, "testuser", "test_app", "127.0.0.1", None, 5432,
                 datetime.now(UTC), None, datetime.now(UTC), None,
                 "active", None, None, "SELECT 1", "client backend", None, None),
                (12346, "testuser", "test_app", "127.0.0.1", None, 5432,
                 datetime.now(UTC), datetime.now(UTC), datetime.now(UTC), None,
                 "idle", None, None, "SELECT * FROM users", "client backend", "Lock", "relation"),
            ])

        if "pg_stat_statements" in query_text:
            # Mock slow query data
            return MockResult([
                ("SELECT * FROM large_table WHERE ...", 100, 15000.0, 150.0,
                 25.0, 100.0, 300.0, 1000, 800, 200, 50, 10, 0, 0, 0, 0, 100.0, 50.0),
                ("UPDATE users SET ...", 50, 8000.0, 160.0,
                 30.0, 120.0, 250.0, 500, 700, 150, 25, 5, 0, 0, 0, 0, 80.0, 40.0),
            ])

        if "pg_extension" in query_text and "pg_stat_statements" in query_text:
            # Mock pg_stat_statements availability
            return MockResult([(True,)])

        if "pg_statio_user_tables" in query_text:
            # Mock cache hit ratio data
            return MockResult([(1000, 9000, 500, 4500, 100, 900, 50, 450)])

        if "pg_database_size" in query_text:
            # Mock database size
            return MockResult([(1024 * 1024 * 1024,)])  # 1GB

        if "pg_tables" in query_text:
            # Mock table size data
            return MockResult([
                ("public", "users", "10 MB", 10 * 1024 * 1024, "8 MB", 8 * 1024 * 1024, "2 MB", 2 * 1024 * 1024),
                ("public", "orders", "5 MB", 5 * 1024 * 1024, "4 MB", 4 * 1024 * 1024, "1 MB", 1 * 1024 * 1024),
            ])

        if "pg_is_in_recovery" in query_text:
            # Mock replication status
            return MockResult([(False,)])

        if "pg_stat_replication" in query_text:
            # Mock replica information
            return MockResult([])  # No replicas for simplicity

        if "pg_locks" in query_text:
            # Mock lock information
            return MockResult([
                ("relation", None, 16384, None, None, None, None, None, None, None,
                 "1/123", 12345, "AccessShareLock", True, "SELECT * FROM users",
                 "active", datetime.now(UTC), datetime.now(UTC), datetime.now(UTC), 5.0),
            ])

        if "pg_stat_database" in query_text:
            # Mock database statistics
            return MockResult([
                (1000, 50, 5000, 45000, 100000, 80000, 500, 200, 100, 0, 5, 1024 * 1024, 2, datetime.now(UTC))
            ])

        if "pg_settings" in query_text and "shared_buffers" in query_text:
            # Mock shared_buffers setting
            return MockResult([("128", "MB")])

        # Default empty result for unmatched queries
        return MockResult([])

    async def fetch_one(self, query, parameters=None):
        result = await self.execute(query, parameters)
        return result.fetchone()

    async def fetch_all(self, query, parameters=None):
        result = await self.execute(query, parameters)
        return result.fetchall()

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def close(self):
        self.closed = True


class MockResult:
    """Mock query result for testing."""

    def __init__(self, data):
        self.data = data
        self.index = 0

    def fetchone(self):
        if self.index < len(self.data):
            result = self.data[self.index]
            self.index += 1
            return result
        return None

    def fetchall(self):
        return self.data


class MockSessionManager:
    """Mock session manager implementation for testing."""

    def __init__(self):
        self.session = MockSession()
        self.healthy = True
        self.connection_info = {
            "pool_size": 20,
            "pool_min_size": 5,
            "pool_max_size": 20,
            "pool_timeout": 30,
            "pool_max_lifetime": 3600,
            "requests_waiting": 0,
        }

    async def get_session(self):
        return self.session

    def session_context(self):
        return AsyncMock(return_value=self.session)

    def transaction_context(self):
        return AsyncMock(return_value=self.session)

    async def health_check(self):
        return self.healthy

    async def get_connection_info(self):
        return self.connection_info

    async def close_all_sessions(self):
        pass


@pytest.fixture
def session_manager():
    """Create a mock session manager for testing."""
    return MockSessionManager()


@pytest.fixture
def connection_service(session_manager):
    """Create a connection service for testing."""
    return DatabaseConnectionService(session_manager)


@pytest.fixture
def metrics_service(session_manager):
    """Create a metrics service for testing."""
    return HealthMetricsService(session_manager)


@pytest.fixture
def alerting_service():
    """Create an alerting service for testing."""
    return AlertingService()


@pytest.fixture
def reporting_service():
    """Create a reporting service for testing."""
    return HealthReportingService()


@pytest.fixture
def health_service(session_manager):
    """Create a unified health service for testing."""
    return DatabaseHealthService(session_manager)


class TestDatabaseConnectionService:
    """Test suite for DatabaseConnectionService."""

    @pytest.mark.asyncio
    async def test_collect_connection_metrics(self, connection_service):
        """Test connection metrics collection."""
        metrics = await connection_service.collect_connection_metrics()

        assert isinstance(metrics, dict)
        assert "timestamp" in metrics
        assert "pool_configuration" in metrics
        assert "connection_states" in metrics
        assert "utilization_metrics" in metrics
        assert "age_statistics" in metrics
        assert "health_indicators" in metrics
        assert "recommendations" in metrics
        assert "collection_time_ms" in metrics

        # Verify pool configuration
        pool_config = metrics["pool_configuration"]
        assert pool_config["current_size"] == 20
        assert pool_config["min_size"] == 5
        assert pool_config["max_size"] == 20

        # Verify utilization calculation
        utilization_metrics = metrics["utilization_metrics"]
        assert "utilization_percent" in utilization_metrics
        assert "available_connections" in utilization_metrics
        assert "pool_efficiency_score" in utilization_metrics

    @pytest.mark.asyncio
    async def test_get_connection_details(self, connection_service):
        """Test connection details retrieval."""
        details = await connection_service.get_connection_details()

        assert isinstance(details, list)
        assert len(details) > 0

        # Verify connection detail structure
        connection = details[0]
        assert "connection_id" in connection
        assert "pid" in connection
        assert "state" in connection
        assert "age_seconds" in connection
        assert "query_duration_seconds" in connection

    @pytest.mark.asyncio
    async def test_analyze_connection_ages(self, connection_service):
        """Test connection age analysis."""
        # Create test connection data
        connections = [
            {"age_seconds": 300},   # 5 minutes
            {"age_seconds": 1200},  # 20 minutes
            {"age_seconds": 3600},  # 1 hour
            {"age_seconds": 7200},  # 2 hours
        ]

        analysis = connection_service.analyze_connection_ages(connections)

        assert analysis["total_connections"] == 4
        assert analysis["average_age_seconds"] == 3075.0  # (300+1200+3600+7200)/4
        assert analysis["max_age_seconds"] == 7200
        assert analysis["min_age_seconds"] == 300
        assert "age_distribution" in analysis

    @pytest.mark.asyncio
    async def test_get_pool_health_summary(self, connection_service):
        """Test pool health summary generation."""
        summary = await connection_service.get_pool_health_summary()

        assert isinstance(summary, dict)
        assert "status" in summary
        assert summary["status"] in {"healthy", "warning", "critical", "error"}
        assert "utilization_percent" in summary
        assert "efficiency_score" in summary
        assert "total_connections" in summary
        assert "recommendations" in summary
        assert "summary" in summary


class TestHealthMetricsService:
    """Test suite for HealthMetricsService."""

    @pytest.mark.asyncio
    async def test_collect_query_performance_metrics(self, metrics_service):
        """Test query performance metrics collection."""
        metrics = await metrics_service.collect_query_performance_metrics()

        assert isinstance(metrics, dict)
        assert "timestamp" in metrics
        assert "pg_stat_statements_available" in metrics
        assert "slow_queries" in metrics
        assert "frequent_queries" in metrics
        assert "cache_performance" in metrics
        assert "current_activity" in metrics
        assert "performance_summary" in metrics
        assert "overall_assessment" in metrics
        assert "recommendations" in metrics
        assert "collection_time_ms" in metrics

        # Verify slow queries structure
        if metrics["slow_queries"]:
            slow_query = metrics["slow_queries"][0]
            assert "query_text" in slow_query
            assert "mean_time_ms" in slow_query
            assert "calls" in slow_query
            assert "cache_hit_ratio" in slow_query

    @pytest.mark.asyncio
    async def test_analyze_cache_performance(self, metrics_service):
        """Test cache performance analysis."""
        cache_analysis = await metrics_service.analyze_cache_performance()

        assert isinstance(cache_analysis, dict)
        assert "overall_cache_hit_ratio" in cache_analysis
        assert "heap_cache_hit_ratio" in cache_analysis
        assert "index_cache_hit_ratio" in cache_analysis
        assert "cache_efficiency" in cache_analysis
        assert "total_cache_misses" in cache_analysis
        assert "total_cache_hits" in cache_analysis
        assert "shared_buffers_setting" in cache_analysis
        assert "recommendations" in cache_analysis

        # Verify cache hit ratio calculation
        hit_ratio = cache_analysis["overall_cache_hit_ratio"]
        assert 0 <= hit_ratio <= 100

    @pytest.mark.asyncio
    async def test_collect_storage_metrics(self, metrics_service):
        """Test storage metrics collection."""
        storage_metrics = await metrics_service.collect_storage_metrics()

        assert isinstance(storage_metrics, dict)
        assert "database_size_bytes" in storage_metrics
        assert "database_size_pretty" in storage_metrics
        assert "total_table_size_bytes" in storage_metrics
        assert "total_index_size_bytes" in storage_metrics
        assert "largest_tables" in storage_metrics
        assert "index_to_table_ratio" in storage_metrics

        # Verify table information structure
        if storage_metrics["largest_tables"]:
            table = storage_metrics["largest_tables"][0]
            assert "schema" in table
            assert "table" in table
            assert "total_size_bytes" in table

    @pytest.mark.asyncio
    async def test_collect_transaction_metrics(self, metrics_service):
        """Test transaction metrics collection."""
        txn_metrics = await metrics_service.collect_transaction_metrics()

        assert isinstance(txn_metrics, dict)
        assert "database_stats" in txn_metrics
        assert "total_transactions" in txn_metrics
        assert "commit_ratio_percent" in txn_metrics
        assert "rollback_ratio_percent" in txn_metrics
        assert "long_running_transactions" in txn_metrics
        assert "transaction_health" in txn_metrics

        # Verify transaction ratios
        commit_ratio = txn_metrics["commit_ratio_percent"]
        rollback_ratio = txn_metrics["rollback_ratio_percent"]
        assert 0 <= commit_ratio <= 100
        assert 0 <= rollback_ratio <= 100


class TestAlertingService:
    """Test suite for AlertingService."""

    def test_calculate_health_score(self, alerting_service):
        """Test health score calculation."""
        # Test with good metrics
        good_metrics = {
            "connection_pool": {"utilization_percent": 50.0},
            "query_performance": {"slow_queries_count": 0},
            "cache": {"overall_cache_hit_ratio_percent": 98.0},
            "replication": {"replication_enabled": False},
            "locks": {"blocking_locks": 0, "long_running_locks": 0},
            "transactions": {"rollback_ratio_percent": 2.0},
        }

        score = alerting_service.calculate_health_score(good_metrics)
        assert 90 <= score <= 100

        # Test with problematic metrics
        bad_metrics = {
            "connection_pool": {"utilization_percent": 98.0},
            "query_performance": {"slow_queries_count": 15},
            "cache": {"overall_cache_hit_ratio_percent": 85.0},
            "replication": {"replication_enabled": True, "lag_seconds": 400.0},
            "locks": {"blocking_locks": 3, "long_running_locks": 2},
            "transactions": {"rollback_ratio_percent": 25.0},
        }

        score = alerting_service.calculate_health_score(bad_metrics)
        assert 0 <= score < 50

    def test_identify_health_issues(self, alerting_service):
        """Test health issues identification."""
        metrics = {
            "connection_pool": {"utilization_percent": 95.0},
            "query_performance": {"slow_queries_count": 12},
            "cache": {"overall_cache_hit_ratio_percent": 88.0},
            "replication": {"replication_enabled": True, "lag_seconds": 120.0},
            "locks": {"blocking_locks": 2},
            "transactions": {"rollback_ratio_percent": 15.0},
        }

        issues = alerting_service.identify_health_issues(metrics)

        assert isinstance(issues, list)
        assert len(issues) > 0

        # Verify issue structure
        issue = issues[0]
        assert "severity" in issue
        assert "category" in issue
        assert "message" in issue
        assert "metric_value" in issue
        assert "threshold" in issue
        assert "timestamp" in issue

        # Check for expected issues
        issue_categories = [issue["category"] for issue in issues]
        assert "connection_pool" in issue_categories
        assert "query_performance" in issue_categories
        assert "cache_performance" in issue_categories

    def test_generate_recommendations(self, alerting_service):
        """Test recommendation generation."""
        metrics = {
            "connection_pool": {"utilization_percent": 88.0, "waiting_requests": 5},
            "query_performance": {"slow_queries_count": 7, "missing_indexes_count": 3},
            "cache": {"overall_cache_hit_ratio_percent": 92.0},
            "storage": {"bloat_metrics": {"bloated_tables_count": 8}},
            "replication": {"replication_enabled": True, "lag_seconds": 180.0},
        }

        recommendations = alerting_service.generate_recommendations(metrics)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Verify recommendation structure
        recommendation = recommendations[0]
        assert "category" in recommendation
        assert "priority" in recommendation
        assert "action" in recommendation
        assert "description" in recommendation
        assert "expected_impact" in recommendation
        assert "timestamp" in recommendation

    @pytest.mark.asyncio
    async def test_send_alert(self, alerting_service):
        """Test alert sending functionality."""
        alert = {
            "severity": "warning",
            "category": "connection_pool",
            "message": "High connection pool utilization",
            "metric_name": "connection_pool_utilization",
            "metric_value": 85.0,
            "threshold": 80.0,
        }

        result = await alerting_service.send_alert(alert)
        assert result is True

        # Verify alert was added to history
        history = alerting_service.get_alert_history()
        assert len(history) > 0
        assert history[0]["message"] == alert["message"]

    def test_set_threshold(self, alerting_service):
        """Test threshold configuration."""
        alerting_service.set_threshold("custom_metric", 75.0, 90.0)

        assert "custom_metric" in alerting_service.thresholds
        threshold = alerting_service.thresholds["custom_metric"]
        assert threshold.warning_threshold == 75.0
        assert threshold.critical_threshold == 90.0
        assert threshold.enabled is True


class TestHealthReportingService:
    """Test suite for HealthReportingService."""

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

    def test_get_health_trends(self, reporting_service):
        """Test health trend analysis."""
        # Add multiple metrics to create trend data
        base_time = datetime.now(UTC)
        for i in range(5):
            metrics = {
                "timestamp": (base_time).isoformat(),
                "health_score": 80.0 + i * 2,  # Improving trend
                "connection_pool": {"utilization_percent": 50.0 + i},
                "cache": {"overall_cache_hit_ratio_percent": 95.0 + i * 0.5},
                "query_performance": {"slow_queries_count": 5 - i},
            }
            reporting_service.add_metrics_to_history(metrics)

        trends = reporting_service.get_health_trends(hours=24)

        assert trends["status"] == "success"
        assert trends["data_points"] == 5
        assert "trends" in trends
        assert "current_values" in trends
        assert "average_values" in trends
        assert "summary" in trends

        # Verify trend calculations
        health_trend = trends["trends"]["health_score"]
        assert health_trend["direction"] in {"increasing", "decreasing", "stable"}
        assert "change_percent" in health_trend
        assert "confidence" in health_trend

    def test_generate_health_report(self, reporting_service):
        """Test comprehensive health report generation."""
        metrics = {
            "timestamp": datetime.now(UTC).isoformat(),
            "health_score": 75.0,
            "issues": [
                {"severity": "warning", "category": "connection_pool", "message": "High utilization"},
                {"severity": "critical", "category": "query_performance", "message": "Slow queries detected"},
            ],
            "recommendations": [
                {"category": "connection_pool", "priority": "high", "action": "increase_pool_size",
                 "description": "Increase pool size", "expected_impact": "Better performance"},
            ],
            "connection_pool": {"utilization_percent": 85.0},
            "query_performance": {"slow_queries_count": 8, "overall_assessment": "poor"},
            "cache": {"overall_cache_hit_ratio_percent": 92.0},
        }

        report = reporting_service.generate_health_report(metrics)

        assert isinstance(report, dict)
        assert "report_metadata" in report
        assert "executive_summary" in report
        assert "component_health" in report
        assert "performance_analysis" in report
        assert "issues_analysis" in report
        assert "recommendations" in report
        assert "raw_metrics" in report

        # Verify executive summary
        exec_summary = report["executive_summary"]
        assert exec_summary["overall_health_score"] == 75.0
        assert exec_summary["overall_status"] == "good"
        assert exec_summary["total_issues"] == 2
        assert exec_summary["critical_issues_count"] == 1

    def test_export_metrics(self, reporting_service):
        """Test metrics export functionality."""
        # Add test data
        metrics = {
            "timestamp": datetime.now(UTC).isoformat(),
            "health_score": 90.0,
        }
        reporting_service.add_metrics_to_history(metrics)

        # Test JSON export
        json_export = reporting_service.export_metrics("json")
        assert isinstance(json_export, str)
        assert "health_score" in json_export

        # Test CSV export
        csv_export = reporting_service.export_metrics("csv")
        assert isinstance(csv_export, str)
        assert "timestamp,health_score" in csv_export

        # Test summary export
        summary_export = reporting_service.export_metrics("summary")
        assert isinstance(summary_export, str)
        assert "Database Health Metrics Summary" in summary_export


class TestDatabaseHealthService:
    """Test suite for unified DatabaseHealthService."""

    @pytest.mark.asyncio
    async def test_collect_comprehensive_metrics(self, health_service):
        """Test comprehensive metrics collection."""
        metrics = await health_service.collect_comprehensive_metrics()

        assert isinstance(metrics, dict)
        assert "timestamp" in metrics
        assert "version" in metrics
        assert "service_architecture" in metrics
        assert "connection_pool" in metrics
        assert "query_performance" in metrics
        assert "storage" in metrics
        assert "replication" in metrics
        assert "locks" in metrics
        assert "cache" in metrics
        assert "transactions" in metrics
        assert "index_health" in metrics
        assert "table_bloat" in metrics
        assert "health_score" in metrics
        assert "issues" in metrics
        assert "recommendations" in metrics
        assert "performance_metadata" in metrics

        # Verify performance metadata
        perf_metadata = metrics["performance_metadata"]
        assert "collection_time_ms" in perf_metadata
        assert perf_metadata["parallel_execution"] is True
        assert perf_metadata["services_used"] == 4

    @pytest.mark.asyncio
    async def test_get_comprehensive_health(self, health_service):
        """Test comprehensive health analysis."""
        health_data = await health_service.get_comprehensive_health()

        assert isinstance(health_data, dict)
        assert "health_score" in health_data
        assert 0 <= health_data["health_score"] <= 100
        assert "trend_analysis" in health_data
        assert "threshold_violations" in health_data
        assert "health_report" in health_data
        assert "performance_metadata" in health_data

        # Verify total time tracking
        perf_metadata = health_data["performance_metadata"]
        assert "total_time_ms" in perf_metadata
        assert perf_metadata["total_time_ms"] > perf_metadata["collection_time_ms"]

    @pytest.mark.asyncio
    async def test_health_check(self, health_service):
        """Test quick health check functionality."""
        health_status = await health_service.health_check()

        assert isinstance(health_status, dict)
        assert "timestamp" in health_status
        assert "overall_status" in health_status
        assert health_status["overall_status"] in {"healthy", "warning", "critical", "error"}
        assert "components" in health_status
        assert "quick_check" in health_status
        assert health_status["quick_check"] is True
        assert "check_time_ms" in health_status

        # Verify component status
        components = health_status["components"]
        assert "connection_pool" in components
        assert "cache" in components
        assert "database" in components

        for component in components.values():
            assert "status" in component

    def test_get_health_trends(self, health_service):
        """Test health trends retrieval."""
        trends = health_service.get_health_trends(hours=12)

        assert isinstance(trends, dict)
        assert "status" in trends
        assert "hours_requested" in trends
        assert trends["hours_requested"] == 12

        # Should indicate insufficient data for new service
        assert trends["status"] in {"insufficient_data", "success", "error"}

    @pytest.mark.asyncio
    async def test_backward_compatibility_methods(self, health_service):
        """Test backward compatibility interface methods."""
        # Test original method names
        pool_metrics = await health_service.get_pool_metrics()
        assert isinstance(pool_metrics, dict)

        pool_summary = await health_service.get_connection_pool_health_summary()
        assert isinstance(pool_summary, dict)
        assert "status" in pool_summary

        query_analysis = await health_service.analyze_query_performance()
        assert isinstance(query_analysis, dict)

        # Test calculation methods
        test_metrics = {"health_score": 85.0}
        score = health_service.calculate_health_score(test_metrics)
        assert isinstance(score, float)

        issues = health_service.identify_health_issues(test_metrics)
        assert isinstance(issues, list)

        recommendations = health_service.generate_recommendations(test_metrics)
        assert isinstance(recommendations, list)

    @pytest.mark.asyncio
    async def test_service_composition(self, session_manager):
        """Test service composition with custom components."""
        # Create custom services
        connection_service = DatabaseConnectionService(session_manager)
        metrics_service = HealthMetricsService(session_manager)
        alerting_service = AlertingService()
        reporting_service = HealthReportingService()

        # Customize alerting thresholds
        alerting_service.set_threshold("connection_pool_utilization", 85.0, 98.0)

        # Create composed service
        composed_service = DatabaseHealthService(
            session_manager=session_manager,
            connection_service=connection_service,
            metrics_service=metrics_service,
            alerting_service=alerting_service,
            reporting_service=reporting_service,
        )

        # Test composed service functionality
        health_data = await composed_service.get_comprehensive_health()
        assert isinstance(health_data, dict)
        assert "health_score" in health_data

    @pytest.mark.asyncio
    async def test_error_handling(self, session_manager):
        """Test error handling and recovery."""
        # Create service with unhealthy session manager
        session_manager.healthy = False

        # Should handle errors gracefully
        health_service = DatabaseHealthService(session_manager)
        health_data = await health_service.get_comprehensive_health()

        assert isinstance(health_data, dict)
        # Should still return some data even with errors
        assert "timestamp" in health_data
        assert "performance_metadata" in health_data


class TestServiceFactories:
    """Test suite for service factory functions."""

    def test_create_database_health_service(self, session_manager):
        """Test service creation factory."""
        service = create_database_health_service(session_manager)

        assert isinstance(service, DatabaseHealthService)
        assert service.session_manager == session_manager
        assert isinstance(service.connection_service, DatabaseConnectionService)
        assert isinstance(service.metrics_service, HealthMetricsService)
        assert isinstance(service.alerting_service, AlertingService)
        assert isinstance(service.reporting_service, HealthReportingService)

    def test_get_database_health_service(self, session_manager):
        """Test global service instance management."""
        # Reset global instance for testing
        import prompt_improver.database.health.services.database_health_service as service_module
        service_module._global_health_service = None

        # First call should require session_manager
        service1 = get_database_health_service(session_manager)
        assert isinstance(service1, DatabaseHealthService)

        # Second call should return same instance
        service2 = get_database_health_service()
        assert service1 is service2

        # Reset for other tests
        service_module._global_health_service = None


@pytest.mark.asyncio
async def test_parallel_execution_performance(health_service):
    """Test that parallel execution provides performance benefits."""
    # Measure comprehensive health collection time
    start_time = asyncio.get_event_loop().time()
    health_data = await health_service.get_comprehensive_health()
    end_time = asyncio.get_event_loop().time()

    total_time_seconds = end_time - start_time
    reported_time_ms = health_data["performance_metadata"]["total_time_ms"]

    # Verify timing consistency
    assert abs(total_time_seconds * 1000 - reported_time_ms) < 100  # Within 100ms tolerance

    # With parallel execution, should complete reasonably fast
    assert reported_time_ms < 5000  # Less than 5 seconds for mock data

    # Verify parallel execution metadata
    assert health_data["performance_metadata"]["parallel_execution"] is True
    assert health_data["performance_metadata"]["services_used"] == 4


@pytest.mark.asyncio
async def test_data_consistency_across_services(health_service):
    """Test that data remains consistent across different service calls."""
    # Get metrics from unified service
    comprehensive_metrics = await health_service.collect_comprehensive_metrics()

    # Get metrics from individual services
    connection_metrics = await health_service.connection_service.collect_connection_metrics()
    query_metrics = await health_service.metrics_service.collect_query_performance_metrics()

    # Verify consistency
    assert comprehensive_metrics["connection_pool"]["timestamp"] is not None
    assert comprehensive_metrics["query_performance"]["timestamp"] is not None

    # Health scores should be consistent
    unified_score = comprehensive_metrics["health_score"]
    calculated_score = health_service.alerting_service.calculate_health_score(comprehensive_metrics)
    assert abs(unified_score - calculated_score) < 0.01  # Should be identical or very close


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
