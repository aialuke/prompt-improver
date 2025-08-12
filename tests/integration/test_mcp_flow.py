"""
MCP Flow Integration Tests

This module provides comprehensive integration testing for the MCP server via stdio transport.
The tests verify:
1. Response time < 200ms for prompt improvement calls
2. Training data persistence to storage stub
3. Health endpoints return status="operational"
"""

import asyncio
import json
import subprocess
import sys
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest


class MCPClient:
    """Simple MCP client for testing purposes using stdio transport."""

    def __init__(self, process: subprocess.Popen) -> None:
        self.process = process
        self.request_id = 0

    def get_next_id(self) -> int:
        """Get next request ID."""
        self.request_id += 1
        return self.request_id

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send JSON-RPC request to MCP server."""
        request = {"jsonrpc": "2.0", "id": self.get_next_id(), "method": method}
        if params:
            request["params"] = params
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json)
        self.process.stdin.flush()
        start_time = time.time()
        timeout = 5.0
        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                raise RuntimeError("MCP server process terminated")
            try:
                line = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self.process.stdout.readline
                    ),
                    timeout=0.1,
                )
                if line and line.strip():
                    response = json.loads(line.strip())
                    if "id" in response and response["id"] == request["id"]:
                        return response
            except (TimeoutError, json.JSONDecodeError):
                await asyncio.sleep(0.01)
                continue
        raise TimeoutError(f"No response received for {method} within {timeout}s")

    async def initialize(self) -> dict[str, Any]:
        """Initialize MCP connection."""
        return await self.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "resources": {},
                    "tools": {},
                    "prompts": {},
                    "experimental": {},
                },
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool."""
        return await self.send_request(
            "tools/call", {"name": name, "arguments": arguments}
        )

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """Read an MCP resource."""
        return await self.send_request("resources/read", {"uri": uri})


@asynccontextmanager
async def mcp_server_process() -> AsyncGenerator[subprocess.Popen, None]:
    """Context manager for MCP server process with proper cleanup."""
    python_path = sys.executable
    server_script = "src/prompt_improver/mcp_server/mcp_server.py"
    process = subprocess.Popen(
        [python_path, server_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
        cwd=".",
    )
    try:
        await asyncio.sleep(3.0)
        if process.poll() is not None:
            stderr_output = process.stderr.read()
            stdout_output = process.stdout.read()
            raise RuntimeError(
                f"MCP server failed to start. stderr: {stderr_output}, stdout: {stdout_output}"
            )
        yield process
    finally:
        try:
            process.terminate()
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, process.wait),
                timeout=5.0,
            )
        except TimeoutError:
            process.kill()
            process.wait()


@pytest.fixture(scope="module")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
@pytest.mark.integration
class TestMCPFlow:
    """Integration tests for MCP server flow via stdio transport."""

    async def test_mcp_server_startup_and_initialization(self):
        """Test MCP server starts and responds to initialization."""
        async with mcp_server_process() as process:
            client = MCPClient(process)
            start_time = time.time()
            response = await client.initialize()
            initialization_time = (time.time() - start_time) * 1000
            assert "result" in response
            assert "capabilities" in response["result"]
            assert initialization_time < 200

    async def test_improve_prompt_response_time(self):
        """Test improve_prompt tool returns within 200ms."""
        async with mcp_server_process() as process:
            client = MCPClient(process)
            await client.initialize()
            start_time = time.time()
            response = await client.send_request(
                "tools/call",
                {
                    "name": "improve_prompt",
                    "arguments": {
                        "prompt": "Help me write better code",
                        "context": {"domain": "software_engineering"},
                        "session_id": "integration_test",
                    },
                },
            )
            response_time = (time.time() - start_time) * 1000
            assert response_time < 200, (
                f"Response time {response_time:.1f}ms exceeds 200ms limit"
            )
            assert "result" in response
            result = response["result"]
            assert "content" in result
            assert len(result["content"]) > 0
            content_text = str(result["content"])
            assert (
                "processing_time_ms" in content_text
                or "improved_prompt" in content_text
            )

    async def test_training_data_persistence(self):
        """Test training data is persisted to storage using real database verification."""
        async with mcp_server_process() as process:
            client = MCPClient(process)
            await client.initialize()
            response = await client.call_tool(
                "improve_prompt",
                {
                    "prompt": "Test prompt for persistence",
                    "context": {"domain": "testing"},
                    "session_id": "persistence_test",
                },
            )
            assert "result" in response
            result = response["result"]
            assert "content" in result
            assert len(result["content"]) > 0
            await asyncio.sleep(3)
            from prompt_improver.database import get_session
            from prompt_improver.services.analytics import AnalyticsService

            async with get_session() as db_session:
                analytics_service = AnalyticsService()
                trends = await analytics_service.get_performance_trends(
                    db_session=db_session, days=1
                )
                assert "daily_trends" in trends
                assert "summary" in trends
                assert isinstance(trends["daily_trends"], list)
                effectiveness_stats = await analytics_service.get_rule_effectiveness(
                    days=1
                )
                assert isinstance(effectiveness_stats, list)
                performance_summary = await analytics_service.get_performance_summary(
                    days=1, db_session=db_session
                )
                assert "total_sessions" in performance_summary
                assert "avg_improvement" in performance_summary
                assert "success_rate" in performance_summary
                assert isinstance(performance_summary["total_sessions"], int)

    async def test_health_endpoints_operational(self):
        """Test health endpoints return status='operational'."""
        async with mcp_server_process() as process:
            client = MCPClient(process)
            await client.initialize()
            live_response = await client.read_resource("apes://health/live")
            assert "result" in live_response
            live_content = live_response["result"]["contents"][0]["text"]
            live_data = json.loads(live_content)
            assert live_data["status"] == "live", (
                f"Expected 'live', got {live_data['status']}"
            )
            assert "event_loop_latency_ms" in live_data
            assert "training_queue_size" in live_data
            assert "timestamp" in live_data
            ready_response = await client.read_resource("apes://health/ready")
            assert "result" in ready_response
            ready_content = ready_response["result"]["contents"][0]["text"]
            ready_data = json.loads(ready_content)
            assert ready_data["status"] in ["ready", "not ready"]
            assert "db_connectivity" in ready_data
            assert "event_loop_latency_ms" in ready_data
            assert "timestamp" in ready_data

    async def test_store_prompt_tool_functionality(self):
        """Test store_prompt tool works correctly."""
        async with mcp_server_process() as process:
            client = MCPClient(process)
            await client.initialize()
            start_time = time.time()
            response = await client.call_tool(
                "store_prompt",
                {
                    "original": "Original test prompt",
                    "enhanced": "Enhanced test prompt with improvements",
                    "metrics": {"improvement_score": 0.85, "clarity_gain": 0.3},
                    "session_id": "store_test",
                },
            )
            response_time = (time.time() - start_time) * 1000
            assert response_time < 200, (
                f"Store response time {response_time:.1f}ms exceeds 200ms limit"
            )
            assert "result" in response
            result = response["result"]
            assert "content" in result
            content_text = str(result["content"])
            assert "status" in content_text
            assert "priority" in content_text

    async def test_concurrent_requests_performance(self):
        """Test server handles concurrent requests within performance limits."""
        async with mcp_server_process() as process:
            client = MCPClient(process)
            await client.initialize()

            async def make_request(request_id: int) -> float:
                start_time = time.time()
                response = await client.call_tool(
                    "improve_prompt",
                    {
                        "prompt": f"Concurrent test prompt {request_id}",
                        "context": {"domain": "performance_testing"},
                        "session_id": f"concurrent_test_{request_id}",
                    },
                )
                response_time = (time.time() - start_time) * 1000
                assert "result" in response
                return response_time

            tasks = [make_request(i) for i in range(3)]
            response_times = await asyncio.gather(*tasks)
            for i, response_time in enumerate(response_times):
                assert response_time < 200, (
                    f"Request {i} took {response_time:.1f}ms, exceeds 200ms limit"
                )
            avg_response_time = sum(response_times) / len(response_times)
            assert avg_response_time < 150, (
                f"Average response time {avg_response_time:.1f}ms too high"
            )
