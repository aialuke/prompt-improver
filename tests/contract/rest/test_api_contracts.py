"""
REST API contract tests.

Validates API contracts, schemas, and backward compatibility.
Tests ensure API responses meet specified contracts and handle edge cases properly.
"""

import time
from uuid import uuid4

import pytest


@pytest.mark.contract
@pytest.mark.api
class TestPromptImprovementAPIContract:
    """Contract tests for prompt improvement REST API."""

    def test_improve_prompt_request_schema(self, api_client, schema_validator, contract_test_data):
        """Test improve prompt request adheres to schema contract."""
        # Arrange
        valid_request = contract_test_data["valid_prompt_request"]

        # Validate request schema
        schema_validator(valid_request, "prompt_improvement_request")

        # Act
        response = api_client.post("/api/v1/improve", data=valid_request)

        # Assert
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"

    def test_improve_prompt_response_schema(self, api_client, schema_validator, contract_test_data):
        """Test improve prompt response adheres to schema contract."""
        # Arrange
        request = contract_test_data["valid_prompt_request"]

        # Act
        response = api_client.post("/api/v1/improve", data=request)

        # Assert
        assert response.status_code == 200
        response_data = response.json()

        # Validate response schema
        schema_validator(response_data, "prompt_improvement_response")

        # Verify required fields
        assert "improved_prompt" in response_data
        assert "confidence" in response_data
        assert "processing_time_ms" in response_data

        # Verify field types and constraints
        assert isinstance(response_data["improved_prompt"], str)
        assert 0 <= response_data["confidence"] <= 1
        assert response_data["processing_time_ms"] >= 0

    def test_invalid_request_schema_validation(self, api_client, contract_test_data):
        """Test API properly rejects invalid request schemas."""
        # Arrange
        invalid_request = contract_test_data["invalid_prompt_request"]

        # Act
        response = api_client.post("/api/v1/improve", data=invalid_request)

        # Assert
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data
        assert "validation" in error_data["error"].lower()

    def test_health_check_contract(self, api_client, schema_validator):
        """Test health check endpoint contract."""
        # Act
        response = api_client.get("/health")

        # Assert
        assert response.status_code == 200
        response_data = response.json()

        # Validate response schema
        schema_validator(response_data, "health_check_response")

        # Verify required fields
        assert "status" in response_data
        assert "timestamp" in response_data
        assert response_data["status"] in {"healthy", "degraded", "unhealthy"}

    @pytest.mark.performance
    def test_response_time_contract(self, api_client, contract_test_data, performance_requirements):
        """Test API response time meets contract requirements."""
        # Arrange
        request = contract_test_data["valid_prompt_request"]
        max_response_time = performance_requirements["api_response_time_ms"]

        # Act
        start_time = time.time()
        response = api_client.post("/api/v1/improve", data=request)
        duration = (time.time() - start_time) * 1000

        # Assert
        assert response.status_code == 200
        assert duration <= max_response_time, \
            f"API response time {duration:.2f}ms exceeds contract requirement {max_response_time}ms"

        # Verify response time is also reported in response
        response_data = response.json()
        reported_time = response_data["processing_time_ms"]
        assert reported_time <= duration  # Should be less than or equal to total request time

    def test_concurrent_request_handling(self, api_client, contract_test_data, performance_requirements):
        """Test API handles concurrent requests per contract."""
        import concurrent.futures

        # Arrange
        request = contract_test_data["valid_prompt_request"]
        concurrent_requests = performance_requirements["concurrent_requests"]
        results = []
        errors = []

        def make_request():
            try:
                # Use unique session_id for each request
                test_request = request.copy()
                test_request["session_id"] = str(uuid4())

                start_time = time.time()
                response = api_client.post("/api/v1/improve", data=test_request)
                duration = (time.time() - start_time) * 1000

                results.append({
                    "status_code": response.status_code,
                    "duration": duration,
                    "response": response.json() if response.status_code == 200 else None
                })
            except Exception as e:
                errors.append(str(e))

        # Act
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(concurrent_requests)]
            concurrent.futures.wait(futures)

        # Assert
        assert len(errors) == 0, f"Errors occurred during concurrent requests: {errors}"
        assert len(results) == concurrent_requests

        # All requests should succeed
        successful_requests = [r for r in results if r["status_code"] == 200]
        assert len(successful_requests) == concurrent_requests

        # Response times should be reasonable under load
        avg_response_time = sum(r["duration"] for r in results) / len(results)
        assert avg_response_time <= 1000, f"Average response time {avg_response_time:.2f}ms too high under load"

    def test_session_history_contract(self, api_client, contract_test_data):
        """Test session history endpoint contract."""
        # Arrange
        session_id = str(uuid4())
        request = contract_test_data["valid_prompt_request"]
        request["session_id"] = session_id

        # Create some history
        prompts = ["First prompt", "Second prompt", "Third prompt"]
        for prompt in prompts:
            request["prompt"] = prompt
            response = api_client.post("/api/v1/improve", data=request)
            assert response.status_code == 200

        # Act
        response = api_client.get(f"/api/v1/sessions/{session_id}/history")

        # Assert
        assert response.status_code == 200
        history_data = response.json()

        assert "history" in history_data
        assert len(history_data["history"]) == 3

        # Verify history item structure
        for item in history_data["history"]:
            assert "original_prompt" in item
            assert "improved_prompt" in item
            assert "confidence" in item
            assert "timestamp" in item
            assert "processing_time_ms" in item

        # Verify chronological order
        timestamps = [item["timestamp"] for item in history_data["history"]]
        assert timestamps == sorted(timestamps), "History should be in chronological order"

    def test_error_response_contract(self, api_client):
        """Test error responses follow contract specification."""
        # Test various error scenarios
        error_cases = [
            ("/api/v1/improve", {"prompt": ""}, 400, "validation"),
            ("/api/v1/improve", {}, 400, "validation"),
            ("/api/v1/sessions/invalid-uuid/history", None, 404, "not found"),
            ("/api/v1/nonexistent", None, 404, "not found")
        ]

        for endpoint, data, expected_status, expected_error_type in error_cases:
            # Act
            if data is not None:
                response = api_client.post(endpoint, data=data)
            else:
                response = api_client.get(endpoint)

            # Assert
            assert response.status_code == expected_status
            error_data = response.json()
            assert "error" in error_data
            assert expected_error_type.lower() in error_data["error"].lower()
            assert "timestamp" in error_data  # Error responses should include timestamp

    def test_api_versioning_contract(self, api_client, api_versions):
        """Test API versioning follows contract specification."""
        # Test current version
        response = api_client.get("/api/v1/health")
        assert response.status_code == 200

        # Test version in response headers
        assert "API-Version" in response.headers
        assert response.headers["API-Version"] == api_versions["current"]

        # Test supported versions
        for version in api_versions["supported"]:
            response = api_client.get(f"/api/{version}/health")
            assert response.status_code == 200

    def test_content_type_negotiation(self, api_client, contract_test_data):
        """Test content type negotiation contract."""
        request = contract_test_data["valid_prompt_request"]

        # Test JSON content type (default)
        response = api_client.post("/api/v1/improve", data=request)
        assert response.status_code == 200
        assert "application/json" in response.headers["Content-Type"]

        # Test unsupported content type
        headers = {"Content-Type": "text/plain", "Accept": "application/json"}
        response = api_client.session.post(
            f"{api_client.base_url}/api/v1/improve",
            data="invalid data",
            headers=headers
        )
        assert response.status_code == 415  # Unsupported Media Type

    def test_rate_limiting_contract(self, api_client, contract_test_data):
        """Test rate limiting behavior follows contract."""
        request = contract_test_data["valid_prompt_request"]

        # Make rapid requests to trigger rate limiting
        responses = []
        for _i in range(20):  # Exceed typical rate limit
            request["session_id"] = str(uuid4())
            response = api_client.post("/api/v1/improve", data=request)
            responses.append(response)

            if response.status_code == 429:  # Rate limited
                break

        # Verify rate limiting headers are present when limit is hit
        rate_limited_response = next((r for r in responses if r.status_code == 429), None)
        if rate_limited_response:
            assert "X-RateLimit-Limit" in rate_limited_response.headers
            assert "X-RateLimit-Remaining" in rate_limited_response.headers
            assert "Retry-After" in rate_limited_response.headers
