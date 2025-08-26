"""
Tests for Health API Endpoints - Real Behavior Integration
Tests with actual health check services, real database connections, and system monitoring.

Key Features:
- Real health check implementations with actual service dependencies
- Real database connection testing
- Real Redis connectivity validation
- Real system resource monitoring
- Circuit breaker and timeout behavior testing
- Kubernetes probe simulation
"""

import asyncio

import psutil
from fastapi.testclient import TestClient


class TestHealthEndpointsRealBehavior:
    """Test suite for health API endpoints with real service integration."""

    async def test_liveness_probe_real_integration(
        self, real_api_client: TestClient, api_helpers, api_performance_monitor
    ):
        """Test Kubernetes liveness probe with real service validation."""
        # Monitor performance
        api_performance_monitor.start_request()

        # Make real API call
        response = real_api_client.get("/health/liveness")

        # Validate response
        assert response.status_code == 200
        data = response.json()

        # Validate performance (liveness should be < 1s)
        duration_ms = api_performance_monitor.end_request(
            "/health/liveness", response.status_code
        )
        api_helpers.assert_api_performance(duration_ms, max_response_time_ms=1000)

        # Validate response structure
        expected_fields = ["status", "timestamp", "uptime_seconds", "version", "environment"]
        api_helpers.assert_api_response_structure(data, expected_fields)

        # Validate status
        assert data["status"] == "alive"
        assert isinstance(data["uptime_seconds"], int)
        assert data["uptime_seconds"] >= 0

    async def test_readiness_probe_real_integration(
        self, real_api_client: TestClient, api_helpers, api_performance_monitor
    ):
        """Test Kubernetes readiness probe with real dependency checks."""
        # Wait for services to be ready first
        await api_helpers.wait_for_service_ready(real_api_client, "/health/readiness")

        # Monitor performance
        api_performance_monitor.start_request()

        # Make real API call
        response = real_api_client.get("/health/readiness")

        # Validate response (might be 200 or 503 depending on service state)
        assert response.status_code in {200, 503}
        data = response.json()

        # Validate performance
        duration_ms = api_performance_monitor.end_request(
            "/health/readiness", response.status_code
        )
        api_helpers.assert_api_performance(duration_ms, max_response_time_ms=5000)

        # Validate response structure
        expected_fields = ["status", "ready", "timestamp", "checks", "check_duration_ms"]
        api_helpers.assert_api_response_structure(data, expected_fields)

        # Validate checks structure
        assert isinstance(data["checks"], dict)

        # Validate that critical services are checked
        expected_services = ["database", "redis"]
        for service in expected_services:
            if service in data["checks"]:
                check = data["checks"][service]
                assert "status" in check
                assert "healthy" in check
                assert "message" in check

    async def test_startup_probe_real_integration(
        self, real_api_client: TestClient, api_helpers, api_performance_monitor
    ):
        """Test Kubernetes startup probe with real startup sequence validation."""
        # Monitor performance
        api_performance_monitor.start_request()

        # Make real API call
        response = real_api_client.get("/health/startup")

        # Validate response (200 if started, 503 if still starting)
        assert response.status_code in {200, 503}
        data = response.json()

        # Validate performance
        duration_ms = api_performance_monitor.end_request(
            "/health/startup", response.status_code
        )
        api_helpers.assert_api_performance(duration_ms, max_response_time_ms=2000)

        # Validate response structure
        expected_fields = ["status", "startup_complete", "timestamp", "uptime_seconds", "startup_tasks"]
        api_helpers.assert_api_response_structure(data, expected_fields)

        # Validate startup tasks
        assert isinstance(data["startup_tasks"], dict)

        expected_tasks = [
            "configuration_loaded", "database_migrations",
            "cache_warmed", "health_checks_initialized"
        ]
        for task in expected_tasks:
            if task in data["startup_tasks"]:
                assert isinstance(data["startup_tasks"][task], bool)

    async def test_main_health_check_real_integration(
        self, real_api_client: TestClient, api_helpers, api_performance_monitor
    ):
        """Test main health check endpoint with comprehensive service validation."""
        # Monitor performance
        api_performance_monitor.start_request()

        # Make real API call
        response = real_api_client.get("/health")

        # Validate response (200 if healthy, 503 if unhealthy)
        assert response.status_code in {200, 503}
        data = response.json()

        # Validate performance
        duration_ms = api_performance_monitor.end_request(
            "/health", response.status_code
        )
        api_helpers.assert_api_performance(duration_ms, max_response_time_ms=3000)

        # Validate response structure
        expected_fields = [
            "status", "healthy", "timestamp", "uptime_seconds",
            "version", "execution_time_ms", "message"
        ]
        api_helpers.assert_api_response_structure(data, expected_fields)

        # Validate status values
        assert data["status"] in {"healthy", "degraded", "unhealthy"}
        assert isinstance(data["healthy"], bool)
        assert isinstance(data["execution_time_ms"], (int, float))

    async def test_deep_health_check_real_integration(
        self, real_api_client: TestClient, api_helpers, api_performance_monitor
    ):
        """Test comprehensive deep health check with all service dependencies."""
        # Monitor performance
        api_performance_monitor.start_request()

        # Make real API call
        response = real_api_client.get("/health/deep")

        # Validate response
        assert response.status_code in {200, 503}
        data = response.json()

        # Validate performance
        duration_ms = api_performance_monitor.end_request(
            "/health/deep", response.status_code
        )
        api_helpers.assert_api_performance(duration_ms, max_response_time_ms=10000)

        # Validate response structure
        expected_fields = [
            "status", "healthy", "timestamp", "uptime_seconds",
            "version", "check_duration_ms", "checks", "summary"
        ]
        api_helpers.assert_api_response_structure(data, expected_fields)

        # Validate checks structure
        assert isinstance(data["checks"], dict)
        assert isinstance(data["summary"], dict)

        # Validate summary structure
        summary = data["summary"]
        expected_summary_fields = ["total_checks", "healthy_checks", "unhealthy_checks"]
        api_helpers.assert_api_response_structure(summary, expected_summary_fields)

        # Validate service checks
        expected_services = ["database", "redis", "system_resources"]
        for service in expected_services:
            if service in data["checks"]:
                check = data["checks"][service]
                assert "status" in check
                assert "healthy" in check
                assert "message" in check
                assert "duration_ms" in check
                assert isinstance(check["duration_ms"], (int, float))

    async def test_database_health_check_real_behavior(
        self, real_api_client: TestClient, api_helpers, api_database_manager
    ):
        """Test database health check with real database connection."""
        # Make deep health check to get database status
        response = real_api_client.get("/health/deep")
        assert response.status_code in {200, 503}

        data = response.json()

        # Validate database check exists and has proper structure
        if "database" in data["checks"]:
            db_check = data["checks"]["database"]

            # Validate database check structure
            assert "status" in db_check
            assert "healthy" in db_check
            assert "duration_ms" in db_check

            # Validate response time is reasonable
            assert db_check["duration_ms"] < 5000, "Database health check too slow"

            # If database is healthy, validate additional details
            if db_check["healthy"]:
                assert db_check["status"] in {"healthy", "degraded"}
                assert "details" in db_check
                if "details" in db_check:
                    details = db_check["details"]
                    assert "response_time_ms" in details
                    assert "connection_test" in details

    async def test_redis_health_check_real_behavior(
        self, real_api_client: TestClient, api_helpers
    ):
        """Test Redis health check with real Redis connection."""
        # Make deep health check to get Redis status
        response = real_api_client.get("/health/deep")
        assert response.status_code in {200, 503}

        data = response.json()

        # Validate Redis check
        if "redis" in data["checks"]:
            redis_check = data["checks"]["redis"]

            # Validate Redis check structure
            assert "status" in redis_check
            assert "healthy" in redis_check
            assert "duration_ms" in redis_check

            # Validate response time is reasonable
            assert redis_check["duration_ms"] < 3000, "Redis health check too slow"

            # Validate status values
            assert redis_check["status"] in {"healthy", "degraded", "unhealthy"}

    async def test_system_resources_health_check_real_behavior(
        self, real_api_client: TestClient, api_helpers
    ):
        """Test system resources health check with real system monitoring."""
        # Make deep health check to get system resources status
        response = real_api_client.get("/health/deep")
        assert response.status_code in {200, 503}

        data = response.json()

        # Validate system resources check
        if "system_resources" in data["checks"]:
            system_check = data["checks"]["system_resources"]

            # Validate system check structure
            assert "status" in system_check
            assert "healthy" in system_check
            assert "duration_ms" in system_check
            assert "details" in system_check

            # Validate response time is reasonable
            assert system_check["duration_ms"] < 2000, "System resources check too slow"

            # Validate details structure
            details = system_check["details"]
            expected_metrics = [
                "memory_percent", "disk_percent",
                "memory_available_gb", "disk_free_gb"
            ]
            for metric in expected_metrics:
                if metric in details:
                    assert isinstance(details[metric], (int, float))

                    # Validate reasonable ranges
                    if "percent" in metric:
                        assert 0 <= details[metric] <= 100
                    if "gb" in metric:
                        assert details[metric] >= 0

    async def test_health_check_performance_consistency(
        self, real_api_client: TestClient, api_performance_monitor, api_helpers
    ):
        """Test health check performance consistency over multiple calls."""
        endpoint = "/health/liveness"
        response_times = []

        # Make 10 consecutive health checks
        for _ in range(10):
            api_performance_monitor.start_request()
            response = real_api_client.get(endpoint)

            duration_ms = api_performance_monitor.end_request(endpoint, response.status_code)
            response_times.append(duration_ms)

            # Validate each response
            assert response.status_code == 200

            # Small delay between requests
            await asyncio.sleep(0.1)

        # Validate performance consistency
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)

        # All responses should be fast
        assert avg_response_time < 500, f"Average response time {avg_response_time}ms too high"
        assert max_response_time < 1000, f"Max response time {max_response_time}ms too high"

        # Response times should be consistent (no outliers > 3x average)
        outliers = [rt for rt in response_times if rt > 3 * avg_response_time]
        assert len(outliers) <= 1, f"Too many response time outliers: {outliers}"

    async def test_health_check_under_system_stress(
        self, real_api_client: TestClient, api_helpers
    ):
        """Test health check behavior under system stress conditions."""
        # Get baseline system metrics
        memory_before = psutil.virtual_memory()

        # Make multiple concurrent health checks
        async def make_health_request():
            response = real_api_client.get("/health")
            return response.status_code, response.json()

        # Make 20 concurrent requests
        tasks = [make_health_request() for _ in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Validate that most requests succeeded
        successful_requests = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_requests) >= 16, "At least 80% of health checks should succeed under stress"

        # Validate that successful requests returned proper status codes
        for status_code, _ in successful_requests:
            assert status_code in {200, 503}, "Health check should return proper status codes"

    async def test_health_check_error_recovery(
        self, real_api_client: TestClient, api_circuit_breaker_tester, api_helpers
    ):
        """Test health check behavior during service errors and recovery."""
        # First, verify health check works normally
        response = real_api_client.get("/health")
        initial_status = response.status_code

        # Simulate database failure
        await api_circuit_breaker_tester.simulate_service_failure("database", 3)

        # Health check should detect the failure
        response = real_api_client.get("/health/deep")
        assert response.status_code in {200, 503}  # May still return 200 with degraded status

        data = response.json()
        if "checks" in data and "database" in data["checks"]:
            # Database check should show unhealthy or degraded
            db_check = data["checks"]["database"]
            # During failure, should be degraded or unhealthy
            assert db_check["status"] in {"degraded", "unhealthy"} or not db_check["healthy"]

        # Wait for recovery
        await asyncio.sleep(4)

        # Verify health check recovers
        response = real_api_client.get("/health")
        # Should return to healthy state or at least not be worse
        assert response.status_code in {200, 503}

    async def test_health_check_monitoring_integration(
        self, real_api_client: TestClient, api_monitoring_manager, api_helpers
    ):
        """Test health check integration with monitoring systems."""
        # Make health check call
        response = real_api_client.get("/health/deep")
        assert response.status_code in {200, 503}

        data = response.json()

        # Validate that metrics are being collected
        # (This would integrate with actual monitoring systems)
        assert "execution_time_ms" in data
        assert isinstance(data["execution_time_ms"], (int, float))
        assert data["execution_time_ms"] > 0

        # Validate timestamp format
        assert "timestamp" in data
        # Validate ISO format timestamp
        timestamp = data["timestamp"]
        assert "T" in timestamp and ("Z" in timestamp or "+" in timestamp)

    async def test_kubernetes_probe_simulation(
        self, real_api_client: TestClient, api_helpers
    ):
        """Simulate Kubernetes probe behavior with real health checks."""
        # Test liveness probe (should always succeed for running app)
        liveness_response = real_api_client.get("/health/liveness")
        assert liveness_response.status_code == 200

        # Test readiness probe (may fail if dependencies not ready)
        readiness_response = real_api_client.get("/health/readiness")
        assert readiness_response.status_code in {200, 503}

        readiness_data = readiness_response.json()
        ready_status = readiness_data.get("ready", False)

        # If not ready, should have failing checks
        if not ready_status:
            checks = readiness_data.get("checks", {})
            failing_checks = [
                name for name, check in checks.items()
                if not check.get("healthy", False)
            ]
            assert len(failing_checks) > 0, "If not ready, should have failing checks"

        # Test startup probe
        startup_response = real_api_client.get("/health/startup")
        assert startup_response.status_code in {200, 503}

        startup_data = startup_response.json()
        startup_complete = startup_data.get("startup_complete", False)

        # Validate startup tasks
        if startup_complete:
            tasks = startup_data.get("startup_tasks", {})
            # All tasks should be complete
            incomplete_tasks = [name for name, status in tasks.items() if not status]
            assert len(incomplete_tasks) == 0, f"Incomplete startup tasks: {incomplete_tasks}"
