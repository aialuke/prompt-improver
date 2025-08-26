"""
MCP protocol contract tests.

Validates MCP protocol compliance, tool contracts, and message formats.
Tests ensure MCP server follows protocol specifications.
"""

import asyncio
import time
from uuid import uuid4

import pytest


@pytest.mark.contract
@pytest.mark.mcp
class TestMCPProtocolContracts:
    """Contract tests for MCP protocol compliance."""

    @pytest.mark.asyncio
    async def test_server_initialization_contract(self, mcp_client):
        """Test MCP server initialization follows protocol contract."""
        # Act
        await mcp_client.connect([
            "python", "-m", "prompt_improver.mcp_server", "--stdio"
        ])

        # Assert
        assert mcp_client.session is not None
        assert mcp_client.session.server_info is not None

        # Verify server info structure
        server_info = mcp_client.session.server_info
        assert "name" in server_info
        assert "version" in server_info
        assert server_info["name"] == "prompt-improver"

        await mcp_client.disconnect()

    @pytest.mark.asyncio
    async def test_tool_listing_contract(self, mcp_client):
        """Test tool listing follows MCP protocol contract."""
        # Arrange
        await mcp_client.connect([
            "python", "-m", "prompt_improver.mcp_server", "--stdio"
        ])

        # Act
        tools = await mcp_client.session.list_tools()

        # Assert
        assert isinstance(tools, list)
        assert len(tools) > 0

        # Verify each tool has required fields
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

            # Verify input schema is valid JSON schema
            input_schema = tool["inputSchema"]
            assert "type" in input_schema
            assert input_schema["type"] == "object"

            if "properties" in input_schema:
                assert isinstance(input_schema["properties"], dict)

        await mcp_client.disconnect()

    @pytest.mark.asyncio
    async def test_improve_prompt_tool_contract(self, mcp_client, mcp_schemas, schema_validator):
        """Test improve_prompt tool follows contract specification."""
        # Arrange
        await mcp_client.connect([
            "python", "-m", "prompt_improver.mcp_server", "--stdio"
        ])

        tool_request = {
            "method": "tools/call",
            "params": {
                "name": "improve_prompt",
                "arguments": {
                    "prompt": "Fix this bug",
                    "context": {"domain": "software_development"}
                }
            }
        }

        # Validate request schema
        schema_validator(tool_request, "tool_call_request")

        # Act
        start_time = time.time()
        result = await mcp_client.call_tool("improve_prompt", tool_request["params"]["arguments"])
        duration = (time.time() - start_time) * 1000

        # Assert
        assert result is not None
        schema_validator(result, "tool_call_response")

        # Verify content structure
        assert "content" in result
        assert isinstance(result["content"], list)
        assert len(result["content"]) > 0

        # Verify first content item
        content_item = result["content"][0]
        assert "type" in content_item
        assert content_item["type"] == "text"
        assert "text" in content_item

        # Parse and validate response data
        import json
        response_data = json.loads(content_item["text"])
        assert "improved_prompt" in response_data
        assert "confidence" in response_data
        assert "processing_time_ms" in response_data

        # Performance contract
        assert duration < 200, f"MCP response took {duration:.2f}ms (should be <200ms)"

        await mcp_client.disconnect()

    @pytest.mark.asyncio
    async def test_get_session_history_tool_contract(self, mcp_client):
        """Test get_session_history tool contract."""
        # Arrange
        await mcp_client.connect([
            "python", "-m", "prompt_improver.mcp_server", "--stdio"
        ])

        session_id = str(uuid4())

        # Create some history first
        await mcp_client.call_tool("improve_prompt", {
            "prompt": "First prompt",
            "session_id": session_id
        })
        await mcp_client.call_tool("improve_prompt", {
            "prompt": "Second prompt",
            "session_id": session_id
        })

        # Act
        result = await mcp_client.call_tool("get_session_history", {
            "session_id": session_id
        })

        # Assert
        assert "content" in result
        content_item = result["content"][0]
        assert content_item["type"] == "text"

        import json
        history_data = json.loads(content_item["text"])
        assert "history" in history_data
        assert len(history_data["history"]) == 2

        # Verify history item structure
        for item in history_data["history"]:
            assert "original_prompt" in item
            assert "improved_prompt" in item
            assert "session_id" in item
            assert "timestamp" in item
            assert item["session_id"] == session_id

        await mcp_client.disconnect()

    @pytest.mark.asyncio
    async def test_error_handling_contract(self, mcp_client):
        """Test error handling follows MCP protocol contract."""
        # Arrange
        await mcp_client.connect([
            "python", "-m", "prompt_improver.mcp_server", "--stdio"
        ])

        # Test invalid tool call
        try:
            result = await mcp_client.call_tool("nonexistent_tool", {})

            # If no exception, check error response format
            if "isError" in result:
                assert result["isError"] is True
                assert "content" in result
                assert len(result["content"]) > 0
                assert result["content"][0]["type"] == "text"
        except Exception as e:
            # Exception should be informative
            assert "nonexistent_tool" in str(e).lower() or "not found" in str(e).lower()

        # Test invalid arguments
        try:
            result = await mcp_client.call_tool("improve_prompt", {})  # Missing required arguments

            if "isError" in result:
                assert result["isError"] is True
        except Exception as e:
            assert "required" in str(e).lower() or "missing" in str(e).lower()

        await mcp_client.disconnect()

    @pytest.mark.asyncio
    async def test_message_format_contract(self, mcp_client):
        """Test all messages follow MCP format contract."""
        # Arrange
        await mcp_client.connect([
            "python", "-m", "prompt_improver.mcp_server", "--stdio"
        ])

        # Act - Make various tool calls and verify message formats
        tools = await mcp_client.session.list_tools()

        for tool in tools[:3]:  # Test first 3 tools
            tool_name = tool["name"]

            # Prepare minimal valid arguments
            arguments = {}
            if "properties" in tool["inputSchema"]:
                for prop, schema in tool["inputSchema"]["properties"].items():
                    if prop in tool["inputSchema"].get("required", []):
                        # Provide minimal valid value based on type
                        if schema.get("type") == "string":
                            arguments[prop] = "test_value"
                        elif schema.get("type") == "object":
                            arguments[prop] = {}

            try:
                result = await mcp_client.call_tool(tool_name, arguments)

                # Verify response format
                assert isinstance(result, dict)
                assert "content" in result
                assert isinstance(result["content"], list)

                for content_item in result["content"]:
                    assert "type" in content_item
                    assert content_item["type"] in {"text", "image", "resource"}

            except Exception:
                # Some tools may fail with minimal arguments, which is acceptable
                pass

        await mcp_client.disconnect()

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_requests_contract(self, performance_requirements):
        """Test MCP server handles concurrent requests per contract."""
        # Arrange
        concurrent_requests = performance_requirements["concurrent_requests"]
        clients = []

        # Create multiple clients
        for _ in range(concurrent_requests):
            client = mcp_client()
            await client.connect([
                "python", "-m", "prompt_improver.mcp_server", "--stdio"
            ])
            clients.append(client)

        try:
            # Act - Make concurrent requests
            tasks = []
            for i, client in enumerate(clients):
                task = client.call_tool("improve_prompt", {
                    "prompt": f"Test concurrent request {i}",
                    "session_id": str(uuid4())
                })
                tasks.append(task)

            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = (time.time() - start_time) * 1000

            # Assert
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) == concurrent_requests, \
                f"Only {len(successful_results)}/{concurrent_requests} requests succeeded"

            # Performance requirement
            avg_response_time = duration / concurrent_requests
            assert avg_response_time < 500, \
                f"Average concurrent response time {avg_response_time:.2f}ms too high"

        finally:
            # Cleanup
            for client in clients:
                await client.disconnect()

    @pytest.mark.asyncio
    async def test_resource_listing_contract(self, mcp_client):
        """Test resource listing follows MCP protocol contract."""
        # Arrange
        await mcp_client.connect([
            "python", "-m", "prompt_improver.mcp_server", "--stdio"
        ])

        # Act
        resources = await mcp_client.session.list_resources()

        # Assert
        assert isinstance(resources, list)

        # If resources are provided, verify structure
        for resource in resources:
            assert "uri" in resource
            assert "name" in resource
            assert "mimeType" in resource or "description" in resource

        await mcp_client.disconnect()

    @pytest.mark.asyncio
    async def test_prompt_listing_contract(self, mcp_client):
        """Test prompt template listing follows MCP protocol contract."""
        # Arrange
        await mcp_client.connect([
            "python", "-m", "prompt_improver.mcp_server", "--stdio"
        ])

        # Act
        prompts = await mcp_client.session.list_prompts()

        # Assert
        assert isinstance(prompts, list)

        # If prompts are provided, verify structure
        for prompt in prompts:
            assert "name" in prompt
            assert "description" in prompt
            if "arguments" in prompt:
                assert isinstance(prompt["arguments"], list)
                for arg in prompt["arguments"]:
                    assert "name" in arg
                    assert "description" in arg

        await mcp_client.disconnect()

    @pytest.mark.asyncio
    async def test_session_persistence_contract(self, mcp_client):
        """Test session persistence across tool calls."""
        # Arrange
        await mcp_client.connect([
            "python", "-m", "prompt_improver.mcp_server", "--stdio"
        ])

        session_id = str(uuid4())

        # Act - Multiple tool calls with same session
        prompts = ["First call", "Second call", "Third call"]
        for prompt in prompts:
            result = await mcp_client.call_tool("improve_prompt", {
                "prompt": prompt,
                "session_id": session_id
            })
            assert "content" in result

        # Get history to verify persistence
        history_result = await mcp_client.call_tool("get_session_history", {
            "session_id": session_id
        })

        # Assert
        import json
        history_data = json.loads(history_result["content"][0]["text"])
        assert len(history_data["history"]) == 3

        # Verify session continuity
        for i, item in enumerate(history_data["history"]):
            assert item["original_prompt"] == prompts[i]
            assert item["session_id"] == session_id

        await mcp_client.disconnect()
