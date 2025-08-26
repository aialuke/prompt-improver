"""
Tests for Analytics API Endpoints - Real Behavior Integration
Tests with actual FastAPI TestClient, real service calls, and comprehensive validation.
Eliminated all mocking in favor of real backend service integration.

Key Features:
- Real FastAPI TestClient with full service stack
- Real database and cache integration
- Real ML service calls and analytics processing
- Performance and reliability testing
- WebSocket endpoint testing
- Circuit breaker and timeout validation
"""

import asyncio

from fastapi.testclient import TestClient


class TestAnalyticsEndpointsRealBehavior:
    """Test suite for analytics API endpoints with real service integration.

    Tests actual API behavior using real FastAPI TestClient with:
    - Real database connections and transactions
    - Real cache integration with proper isolation
    - Real ML service calls and analytics processing
    - Real WebSocket connections and streaming
    - Performance and reliability validation
    """

    async def test_dashboard_metrics_real_integration(
        self, real_api_client: TestClient, api_helpers, api_performance_monitor
    ):
        """Test dashboard metrics endpoint with real service integration."""
        # Wait for services to be ready
        await api_helpers.wait_for_service_ready(real_api_client)

        # Monitor performance
        api_performance_monitor.start_request()

        # Make real API call
        response = real_api_client.get("/api/v1/analytics/dashboard/metrics")

        # Validate response
        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        expected_fields = ["current_period", "time_range", "last_updated"]
        api_helpers.assert_api_response_structure(data, expected_fields)

        # Validate performance
        duration_ms = api_performance_monitor.end_request(
            "/api/v1/analytics/dashboard/metrics", response.status_code
        )
        api_helpers.assert_api_performance(duration_ms, max_response_time_ms=2000)

        # Validate data types and structure
        assert isinstance(data["current_period"], dict)
        assert isinstance(data["time_range"], dict)
        assert "last_updated" in data

    async def test_dashboard_metrics_with_custom_time_range(
        self, real_api_client: TestClient, api_helpers
    ):
        """Test dashboard metrics with custom time range parameters."""
        # Make real API call with parameters
        response = real_api_client.get(
            "/api/v1/analytics/dashboard/metrics?time_range_hours=168&include_comparisons=true"
        )

        # Validate response
        assert response.status_code == 200
        data = response.json()

        # Validate time range is respected
        time_range = data.get("time_range", {})
        assert time_range.get("hours") == 168

    async def test_performance_trends_real_integration(
        self, real_api_client: TestClient, api_test_data, api_helpers
    ):
        """Test performance trends analysis with real ML service integration."""
        # Prepare request data
        request_data = api_test_data["analytics"]["trend_analysis"]

        # Make real API call
        response = real_api_client.post(
            "/api/v1/analytics/trends/analysis",
            json=request_data
        )

        # Validate response
        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        expected_fields = [
            "time_series", "trend_direction", "trend_strength",
            "correlation_coefficient", "seasonal_patterns", "metadata"
        ]
        api_helpers.assert_api_response_structure(data, expected_fields)

        # Validate data types
        assert isinstance(data["time_series"], list)
        assert isinstance(data["trend_strength"], (int, float))
        assert data["trend_direction"] in {"increasing", "decreasing", "stable"}

        # Validate time series data structure
        if data["time_series"]:
            time_point = data["time_series"][0]
            assert "timestamp" in time_point
            assert "value" in time_point

    async def test_session_comparison_real_integration(
        self, real_api_client: TestClient, api_test_data, api_helpers
    ):
        """Test session comparison with real ML analysis service integration."""
        # Prepare comparison request
        request_data = api_test_data["analytics"]["session_comparison"]

        # Make real API call
        response = real_api_client.post(
            "/api/v1/analytics/sessions/compare",
            json=request_data
        )

        # Validate response
        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        expected_fields = [
            "session_a_id", "session_b_id", "comparison_dimension",
            "statistical_significance", "p_value", "effect_size",
            "winner", "insights", "recommendations"
        ]
        api_helpers.assert_api_response_structure(data, expected_fields)

        # Validate session IDs match request
        assert data["session_a_id"] == request_data["session_a_id"]
        assert data["session_b_id"] == request_data["session_b_id"]

        # Validate statistical fields
        assert isinstance(data["statistical_significance"], bool)
        assert isinstance(data["p_value"], (int, float))
        assert isinstance(data["effect_size"], (int, float))

    async def test_session_summary_real_integration(
        self, real_api_client: TestClient, api_helpers
    ):
        """Test session summary endpoint with real ML service integration."""
        session_id = "test_session_integration_001"

        # Create test session data through real services
        test_session_data = await api_helpers.create_test_session_data(
            real_api_client, session_id
        )

        # Make real API call
        response = real_api_client.get(f"/api/v1/analytics/sessions/{session_id}/summary")

        # Validate response
        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        expected_fields = [
            "session_id", "status", "executive_kpis", "performance_metrics",
            "training_statistics", "insights", "metadata"
        ]
        api_helpers.assert_api_response_structure(data, expected_fields)

        # Validate session ID matches
        assert data["session_id"] == session_id

        # Validate KPIs structure
        assert isinstance(data["executive_kpis"], dict)
        assert isinstance(data["performance_metrics"], dict)
        assert isinstance(data["training_statistics"], dict)

    async def test_export_session_report_real_integration(
        self, real_api_client: TestClient, api_helpers
    ):
        """Test session report export with real service integration."""
        session_id = "test_session_export_001"

        # Make real API call
        response = real_api_client.get(
            f"/api/v1/analytics/sessions/{session_id}/export?format=json"
        )

        # Validate response
        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        expected_fields = [
            "status", "request_id", "format", "session_id", "requested_at"
        ]
        api_helpers.assert_api_response_structure(data, expected_fields)

        # Validate export request status
        assert data["status"] in {"export_requested", "completed"}
        assert data["session_id"] == session_id
        assert data["format"] == "json"

    async def test_performance_distribution_real_integration(
        self, real_api_client: TestClient, api_helpers
    ):
        """Test performance distribution analysis with real analytics service."""
        # Make real API call
        response = real_api_client.get(
            "/api/v1/analytics/distribution/performance?bucket_count=20"
        )

        # Validate response
        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        expected_fields = [
            "distribution", "total_sessions", "avg_performance", "bucket_count"
        ]
        api_helpers.assert_api_response_structure(data, expected_fields)

        # Validate data types
        assert isinstance(data["total_sessions"], int)
        assert isinstance(data["avg_performance"], (int, float))
        assert data["bucket_count"] == 20

    async def test_correlation_analysis_real_integration(
        self, real_api_client: TestClient, api_helpers
    ):
        """Test correlation analysis with real analytics service."""
        # Make real API call
        response = real_api_client.get(
            "/api/v1/analytics/correlation/analysis?metrics=performance,efficiency,duration"
        )

        # Validate response
        assert response.status_code == 200
        data = response.json()

        # Validate data structure (response from real service)
        assert isinstance(data, dict)

    async def test_analytics_health_check_real_integration(
        self, real_api_client: TestClient, api_helpers
    ):
        """Test analytics health check endpoint with real service validation."""
        # Make real API call
        response = real_api_client.get("/api/v1/analytics/health")

        # Validate response
        assert response.status_code == 200
        data = response.json()

        # Validate health check structure
        expected_fields = ["status", "timestamp", "services"]
        api_helpers.assert_api_response_structure(data, expected_fields)

        # Validate health status
        assert data["status"] == "healthy"
        assert isinstance(data["services"], dict)

    async def test_dashboard_config_real_integration(
        self, real_api_client: TestClient, api_helpers
    ):
        """Test dashboard configuration endpoint."""
        # Make real API call
        response = real_api_client.get("/api/v1/analytics/dashboard/config")

        # Validate response
        assert response.status_code == 200
        data = response.json()

        # Validate config structure
        expected_fields = ["status", "config"]
        api_helpers.assert_api_response_structure(data, expected_fields)

        # Validate config contents
        config = data["config"]
        assert "dashboard_settings" in config
        assert "websocket_endpoints" in config
        assert "api_endpoints" in config

    async def test_websocket_dashboard_real_integration(
        self, websocket_test_client: TestClient, api_helpers
    ):
        """Test WebSocket dashboard endpoint with real service integration."""
        with websocket_test_client.websocket_connect(
            "/api/v1/analytics/live/dashboard?user_id=test_user"
        ) as websocket:
            # Send request for data
            websocket.send_json({"type": "request_update", "time_range_hours": 24})

            # Receive response
            data = websocket.receive_json()

            # Validate WebSocket response
            assert data["type"] == "dashboard_data"
            assert "data" in data
            assert "timestamp" in data

    async def test_websocket_session_real_integration(
        self, websocket_test_client: TestClient, api_helpers
    ):
        """Test WebSocket session endpoint with real service integration."""
        session_id = "test_websocket_session_001"

        with websocket_test_client.websocket_connect(
            f"/api/v1/analytics/live/session/{session_id}?user_id=test_user"
        ) as websocket:
            # Send request for session update
            websocket.send_json({"type": "request_update"})

            # Receive response
            data = websocket.receive_json()

            # Validate WebSocket response
            assert data["type"] == "session_update"
            assert data["session_id"] == session_id
            assert "data" in data

    async def test_api_error_handling_real_integration(
        self, real_api_client: TestClient, api_helpers
    ):
        """Test API error handling with real service errors."""
        # Test invalid session ID
        response = real_api_client.get("/api/v1/analytics/sessions/invalid_session/summary")

        # Should return proper error response (not necessarily 404)
        assert response.status_code in {404, 500}

        # Test malformed request data
        invalid_request = {"invalid": "data"}
        response = real_api_client.post(
            "/api/v1/analytics/trends/analysis",
            json=invalid_request
        )

        # Should return validation error
        assert response.status_code in {400, 422, 500}

    async def test_api_performance_under_load(
        self, real_api_client: TestClient, api_performance_monitor, api_helpers
    ):
        """Test API performance under concurrent load."""
        import asyncio

        async def make_dashboard_request():
            api_performance_monitor.start_request()
            response = real_api_client.get("/api/v1/analytics/dashboard/metrics")
            duration_ms = api_performance_monitor.end_request(
                "/api/v1/analytics/dashboard/metrics", response.status_code
            )
            return response.status_code, duration_ms

        # Make 10 concurrent requests
        tasks = [make_dashboard_request() for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Validate all requests succeeded
        successful_requests = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_requests) >= 8, "At least 80% of requests should succeed"

        # Validate performance
        response_times = [duration for status, duration in successful_requests if status == 200]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            assert avg_response_time < 3000, f"Average response time {avg_response_time}ms too high"

    async def test_circuit_breaker_behavior(
        self, real_api_client: TestClient, api_circuit_breaker_tester, api_helpers
    ):
        """Test circuit breaker behavior with service failures."""
        # Simulate service failure
        await api_circuit_breaker_tester.simulate_service_failure("analytics_service", 5)

        # Verify circuit breaker opens
        await api_circuit_breaker_tester.verify_circuit_breaker_open(
            real_api_client, "/api/v1/analytics/dashboard/metrics"
        )

        # Wait for service recovery
        await asyncio.sleep(6)

        # Verify circuit breaker recovers
        await api_circuit_breaker_tester.verify_circuit_breaker_recovery(
            real_api_client, "/api/v1/analytics/dashboard/metrics"
        )

    async def test_cleanup_test_data(
        self, api_helpers, api_database_manager
    ):
        """Clean up test data after API integration tests."""
        test_session_ids = [
            "test_session_integration_001",
            "test_session_export_001",
            "test_websocket_session_001"
        ]

        await api_helpers.cleanup_test_data(api_database_manager, test_session_ids)
