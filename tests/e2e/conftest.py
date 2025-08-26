"""
End-to-end test configuration and fixtures.

Provides complete system testing infrastructure for full workflow validation.
Tests verify complete user scenarios with real deployment configurations.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager


# System under test management
@pytest.fixture(scope="session")
def docker_compose_setup():
    """Set up complete system using Docker Compose for E2E testing."""
    compose_file = Path(__file__).parent.parent.parent / "docker-compose.test.yml"

    # Start services
    subprocess.run([
        "docker-compose", "-f", str(compose_file), "up", "-d"
    ], check=True, capture_output=True)

    # Wait for services to be ready
    _wait_for_service_health("http://localhost:8000/health", timeout=60)
    _wait_for_service_health("http://localhost:8001/health", timeout=30)

    yield {
        "api_url": "http://localhost:8000",
        "mcp_port": 8001,
        "database_url": "postgresql://test:test@localhost:5433/test_db",
        "redis_url": "redis://localhost:6380"
    }

    # Cleanup
    subprocess.run([
        "docker-compose", "-f", str(compose_file), "down", "-v"
    ], check=False, capture_output=True)


def _wait_for_service_health(url: str, timeout: int = 30):
    """Wait for service to become healthy."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)

    raise TimeoutError(f"Service at {url} did not become healthy within {timeout}s")


# Browser automation
@pytest.fixture(scope="session")
def browser_driver():
    """Set up browser driver for E2E UI testing."""
    options = ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(10)

    yield driver

    driver.quit()


# Complete workflow client
@pytest.fixture
def e2e_client(docker_compose_setup):
    """Complete system client for E2E testing."""
    class E2EClient:
        def __init__(self, config: dict[str, Any]):
            self.api_base_url = config["api_url"]
            self.mcp_port = config["mcp_port"]
            self.session = requests.Session()
            self.session.headers.update({
                "Content-Type": "application/json",
                "Accept": "application/json"
            })

        def create_session(self, user_id: str | None = None) -> str:
            """Create user session."""
            response = self.session.post(f"{self.api_base_url}/sessions", json={
                "user_id": user_id or f"e2e_user_{uuid4()}"
            })
            response.raise_for_status()
            return response.json()["session_id"]

        def improve_prompt(self, session_id: str, prompt: str, context: dict | None = None):
            """Improve prompt via API."""
            data = {
                "prompt": prompt,
                "session_id": session_id,
                "context": context or {}
            }
            response = self.session.post(f"{self.api_base_url}/improve", json=data)
            response.raise_for_status()
            return response.json()

        def get_session_history(self, session_id: str):
            """Get session improvement history."""
            response = self.session.get(f"{self.api_base_url}/sessions/{session_id}/history")
            response.raise_for_status()
            return response.json()

        def get_analytics(self, session_id: str):
            """Get session analytics."""
            response = self.session.get(f"{self.api_base_url}/analytics/{session_id}")
            response.raise_for_status()
            return response.json()

        async def mcp_improve_prompt(self, prompt: str, context: dict | None = None):
            """Improve prompt via MCP protocol."""
            from mcp import ClientSession
            from mcp.client.stdio import stdio_client

            # Connect to MCP server
            client = await stdio_client([
                "python", "-m", "prompt_improver.mcp_server",
                "--port", str(self.mcp_port)
            ])

            session = ClientSession(client)
            await session.initialize()

            try:
                return await session.call_tool("improve_prompt", {
                    "prompt": prompt,
                    "context": context or {}
                })
            finally:
                await client.close()

    return E2EClient(docker_compose_setup)


# Test data generators for E2E scenarios
@pytest.fixture
def e2e_test_scenarios():
    """Complete test scenarios for E2E validation."""
    return {
        "software_development": {
            "prompts": [
                "Fix bug",
                "Make code better",
                "Refactor this function",
                "Add error handling",
                "Write unit tests"
            ],
            "context": {
                "domain": "software_development",
                "language": "python",
                "complexity": "medium"
            },
            "expected_improvements": [
                "clarity",
                "specificity",
                "structure"
            ]
        },
        "data_science": {
            "prompts": [
                "Analyze data",
                "Build model",
                "Clean dataset",
                "Visualize results",
                "Find patterns"
            ],
            "context": {
                "domain": "data_science",
                "language": "python",
                "complexity": "high"
            },
            "expected_improvements": [
                "specificity",
                "methodology",
                "structure"
            ]
        },
        "general": {
            "prompts": [
                "Help me",
                "Do this task",
                "Fix problem",
                "Make better",
                "Explain this"
            ],
            "context": {
                "domain": "general",
                "complexity": "low"
            },
            "expected_improvements": [
                "clarity",
                "specificity"
            ]
        }
    }


# Performance monitoring for E2E tests
@pytest.fixture
def e2e_performance_monitor():
    """Performance monitoring for E2E tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}

        def start_measurement(self, name: str):
            """Start performance measurement."""
            self.metrics[name] = {"start": time.time()}

        def end_measurement(self, name: str):
            """End performance measurement."""
            if name in self.metrics:
                self.metrics[name]["duration"] = time.time() - self.metrics[name]["start"]

        def get_duration(self, name: str) -> float:
            """Get measurement duration in seconds."""
            return self.metrics.get(name, {}).get("duration", 0)

        def assert_performance(self, name: str, max_duration: float):
            """Assert performance requirement."""
            duration = self.get_duration(name)
            assert duration <= max_duration, \
                f"Performance requirement failed: {name} took {duration:.2f}s (max: {max_duration}s)"

    return PerformanceMonitor()


# Load testing support
@pytest.fixture
def load_test_config():
    """Configuration for load testing scenarios."""
    return {
        "concurrent_users": 10,
        "test_duration_seconds": 60,
        "ramp_up_time_seconds": 10,
        "success_rate_threshold": 0.95,
        "response_time_p95_ms": 1000,
        "response_time_p99_ms": 2000
    }


# Scenario validation helpers
@pytest.fixture
def scenario_validator():
    """Helper for validating complete user scenarios."""
    class ScenarioValidator:
        def __init__(self):
            self.results = []

        def validate_improvement_quality(self, original: str, improved: str, confidence: float):
            """Validate improvement quality."""
            # Length should increase (more detail)
            assert len(improved) > len(original), "Improved prompt should be longer"

            # Confidence should be reasonable
            assert 0.5 <= confidence <= 1.0, f"Confidence {confidence} should be between 0.5 and 1.0"

            # Should not be identical
            assert improved != original, "Improved prompt should be different from original"

        def validate_response_time(self, duration_ms: float, max_ms: float):
            """Validate response time requirement."""
            assert duration_ms <= max_ms, \
                f"Response time {duration_ms}ms exceeded maximum {max_ms}ms"

        def validate_session_consistency(self, session_data: dict):
            """Validate session data consistency."""
            assert "session_id" in session_data
            assert "history" in session_data
            assert "analytics" in session_data

            # History should be chronological
            history = session_data["history"]
            if len(history) > 1:
                timestamps = [item["timestamp"] for item in history]
                assert timestamps == sorted(timestamps), "History should be chronological"

    return ScenarioValidator()


# Environment configuration
@pytest.fixture(autouse=True)
def e2e_environment():
    """Set up E2E test environment."""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "INFO"
    os.environ["METRICS_ENABLED"] = "true"

    yield

    # Cleanup environment
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


def pytest_configure(config):
    """Configure E2E test markers."""
    config.addinivalue_line("markers", "e2e: end-to-end test")
    config.addinivalue_line("markers", "workflow: complete workflow test")
    config.addinivalue_line("markers", "scenario: user scenario test")
    config.addinivalue_line("markers", "performance: performance requirement test")
    config.addinivalue_line("markers", "load: load testing")
    config.addinivalue_line("markers", "ui: user interface test")
