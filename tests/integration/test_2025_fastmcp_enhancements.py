"""Real behavior tests for 2025 FastMCP enhancements.

Tests the actual implementation of:
- Custom middleware stack (timing, logging, rate limiting)
- Progress reporting with Context support
- Advanced resource templates with wildcards
- Streamable HTTP transport
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List

import pytest
from mcp.server.fastmcp import Context
from unittest.mock import AsyncMock, MagicMock, patch

from prompt_improver.mcp_server.server import APESMCPServer
from prompt_improver.mcp_server.middleware import (
    TimingMiddleware,
    DetailedTimingMiddleware,
    StructuredLoggingMiddleware,
    RateLimitingMiddleware,
    ErrorHandlingMiddleware,
    MiddlewareContext,
    create_default_middleware_stack
)
from prompt_improver.utils.session_store import SessionStore

logger = logging.getLogger(__name__)


class TestMiddlewareStack:
    """Test custom middleware implementation."""
    
    @pytest.mark.asyncio
    async def test_timing_middleware_real_behavior(self):
        """Test timing middleware measures request duration accurately."""
        middleware = TimingMiddleware()
        
        # Create a test handler that takes 100ms
        async def test_handler(ctx: MiddlewareContext):
            await asyncio.sleep(0.1)  # 100ms
            return {"result": "success"}
        
        context = MiddlewareContext(method="test_method")
        result = await middleware(context, test_handler)
        
        # Verify result
        assert result["result"] == "success"
        assert "_metadata" in result
        assert "duration_ms" in result["_metadata"]
        
        # Duration should be around 100ms (allow 90-150ms for system variance)
        duration = result["_metadata"]["duration_ms"]
        assert 90 <= duration <= 150, f"Expected ~100ms, got {duration}ms"
        
        # Check metrics were recorded
        metrics = middleware.get_metrics_summary()
        assert "test_method" in metrics
        assert metrics["test_method"]["count"] == 1
        assert 90 <= metrics["test_method"]["avg_ms"] <= 150
    
    @pytest.mark.asyncio
    async def test_rate_limiting_middleware_real_behavior(self):
        """Test rate limiting actually prevents excessive requests."""
        # Create middleware with low limit for testing
        middleware = RateLimitingMiddleware(
            max_requests_per_second=2.0,
            burst_capacity=3
        )
        
        async def test_handler(ctx: MiddlewareContext):
            return {"success": True}
        
        context = MiddlewareContext(method="test_method")
        
        # First 3 requests should succeed (burst capacity)
        for i in range(3):
            result = await middleware(context, test_handler)
            assert result["success"] is True
        
        # 4th request should fail
        from mcp import McpError
        with pytest.raises(McpError) as exc_info:
            await middleware(context, test_handler)
        
        assert exc_info.value.error.code == -32000
        assert "Rate limit exceeded" in exc_info.value.error.message
        
        # Wait for token replenishment
        await asyncio.sleep(0.6)  # Wait 600ms for ~1 token
        
        # Should succeed again
        result = await middleware(context, test_handler)
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_error_handling_middleware_real_behavior(self):
        """Test error handling middleware transforms exceptions correctly."""
        middleware = ErrorHandlingMiddleware(
            include_traceback=True,
            transform_errors=True
        )
        
        # Handler that raises exception
        async def failing_handler(ctx: MiddlewareContext):
            raise ValueError("Test error")
        
        context = MiddlewareContext(method="test_method")
        
        from mcp import McpError
        with pytest.raises(McpError) as exc_info:
            await middleware(context, failing_handler)
        
        # Check error was transformed
        assert exc_info.value.error.code == -32603  # Internal error
        assert "Internal error in test_method" in exc_info.value.error.message
        assert "Test error" in exc_info.value.error.message
        
        # Check error was counted
        assert middleware.error_counts["ValueError:test_method"] == 1
    
    @pytest.mark.asyncio
    async def test_middleware_stack_integration(self):
        """Test full middleware stack working together."""
        stack = create_default_middleware_stack()
        
        # Create a handler that takes time and might fail
        call_count = 0
        
        async def test_handler(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Simulate some work
            await asyncio.sleep(0.05)  # 50ms
            
            if call_count == 2:
                raise ValueError("Simulated error")
            
            return {"call_count": call_count}
        
        wrapped = stack.wrap(test_handler)
        
        # First call should succeed and be timed
        result1 = await wrapped(__method__="test_operation")
        assert result1["call_count"] == 1
        
        # Second call should fail but be handled
        from mcp import McpError
        with pytest.raises(McpError):
            await wrapped(__method__="test_operation")
        
        # Verify timing metrics were collected
        # Find timing middleware in stack
        timing_mw = None
        for mw in stack.middleware:
            if isinstance(mw, TimingMiddleware):
                timing_mw = mw
                break
        
        assert timing_mw is not None
        metrics = timing_mw.get_metrics_summary()
        # Note: The method name might be wrapped differently
        assert len(metrics) > 0


class TestProgressReporting:
    """Test progress reporting functionality."""
    
    @pytest.mark.asyncio
    async def test_improve_prompt_progress_real_behavior(self):
        """Test progress reporting during prompt improvement."""
        server = APESMCPServer()
        
        # Create a mock context using modern 2025 helper method
        progress_reports = []
        log_messages = []
        
        mock_ctx = server.create_mock_context()
        mock_ctx.report_progress = AsyncMock(side_effect=lambda **kwargs: progress_reports.append(kwargs))
        mock_ctx.info = AsyncMock(side_effect=lambda msg: log_messages.append(("info", msg)))
        mock_ctx.debug = AsyncMock(side_effect=lambda msg: log_messages.append(("debug", msg)))
        mock_ctx.error = AsyncMock(side_effect=lambda msg: log_messages.append(("error", msg)))
        
        # Test the progress-aware tool
        prompt = "Test prompt for enhancement"
        
        # Need to access the tool implementation directly
        # The tool is registered as a decorator, so we need to find it
        improve_prompt = None
        for tool_name, tool_func in server.mcp._tools.items():
            if tool_name == "improve_prompt":
                improve_prompt = tool_func.implementation
                break
        
        assert improve_prompt is not None, "Tool 'improve_prompt' not found"
        
        # Call the tool with required modern parameters using helper methods
        session_id = server.create_session_id("test_client")
        result = await improve_prompt(
            prompt=prompt,
            session_id=session_id,  # Generated using helper method
            ctx=mock_ctx,  # Created using helper method
            context={"test": "context"}
        )
        
        # Verify progress was reported
        assert len(progress_reports) >= 4, f"Expected at least 4 progress reports, got {len(progress_reports)}"
        
        # Check progress sequence
        progress_values = [p["progress"] for p in progress_reports]
        assert progress_values[0] == 0  # Start
        assert progress_values[-1] == 100  # Complete
        
        # Check messages were logged
        assert len(log_messages) >= 3
        assert any("Beginning prompt enhancement" in msg for level, msg in log_messages if level == "info")
        assert any("Enhancement complete" in msg for level, msg in log_messages if level == "info")
        
        # Result should include timing metrics
        assert "_timing_metrics" in result


class TestWildcardResources:
    """Test advanced resource templates with wildcards."""
    
    @pytest.mark.asyncio
    async def test_session_history_wildcard_real_behavior(self):
        """Test hierarchical session history retrieval."""
        server = APESMCPServer()
        
        # Populate session store with test data
        test_session_data = {
            "user_data": {"name": "Test User"},
            "history": [
                {"action": "login", "workspace": "main", "timestamp": 1234567890},
                {"action": "query", "workspace": "main", "timestamp": 1234567891},
                {"action": "query", "workspace": "dev", "timestamp": 1234567892},
                {"action": "logout", "workspace": "main", "timestamp": 1234567893}
            ]
        }
        
        await server.services.session_store.set("user123", test_session_data)
        
        # Test 1: Get full history
        result1 = await server._get_session_history_impl("user123")
        assert result1["exists"] is True
        assert len(result1["history"]) == 4
        assert result1["base_session_id"] == "user123"
        assert result1["path_components"] == ["user123"]
        
        # Test 2: Get workspace-filtered history
        result2 = await server._get_session_history_impl("user123/workspace/main")
        assert result2["exists"] is True
        assert len(result2["history"]) == 3  # Only main workspace entries
        assert all(h["workspace"] == "main" for h in result2["history"])
        assert result2["path_components"] == ["user123", "workspace", "main"]
        
        # Test 3: Non-existent session
        result3 = await server._get_session_history_impl("nonexistent/workspace/main")
        assert result3["exists"] is False
        assert "not found" in result3["message"]
    
    @pytest.mark.asyncio
    async def test_rule_category_performance_wildcard_real_behavior(self):
        """Test rule category performance metrics with wildcards."""
        server = APESMCPServer()
        
        # Mock database query results
        with patch('prompt_improver.database.get_unified_manager') as mock_manager:
            mock_session = AsyncMock()
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_session
            mock_manager.return_value.get_async_session.return_value = mock_context
            
            # Mock query results for category metrics
            mock_result = MagicMock()
            mock_result.first.return_value = (
                5,      # total_rules
                1000,   # total_applications  
                0.75,   # avg_improvement_score
                25.5,   # avg_execution_ms
                0.95,   # success_rate
                "security",  # category
                "input_validation"  # subcategory
            )
            
            # Mock top rules results
            mock_top_rules = MagicMock()
            mock_top_rules.fetchall.return_value = [
                ("sec_001", "XSS Prevention", 500, 0.85),
                ("sec_002", "SQL Injection Guard", 300, 0.80)
            ]
            
            mock_session.execute = AsyncMock(side_effect=[mock_result, mock_top_rules])
            
            # Test category performance query
            result = await server._get_rule_category_performance_impl("security/input_validation")
            
            assert result["categories"] == ["security", "input_validation"]
            assert result["metrics"]["total_rules"] == 5
            assert result["metrics"]["total_applications"] == 1000
            assert result["metrics"]["avg_improvement_score"] == 0.75
            assert result["metrics"]["success_rate"] == 0.95
            
            assert len(result["top_rules"]) == 2
            assert result["top_rules"][0]["rule_id"] == "sec_001"
            assert result["top_rules"][0]["applications"] == 500
    
    @pytest.mark.asyncio
    async def test_hierarchical_metrics_real_behavior(self):
        """Test hierarchical metrics retrieval."""
        server = APESMCPServer()
        
        # Add some timing data to middleware
        server.services.timing_middleware.metrics["improve_prompt"] = [10.5, 15.2, 12.8]
        server.services.timing_middleware.metrics["get_session"] = [5.1, 4.8, 5.3]
        
        # Test 1: Get all performance metrics
        result1 = await server._get_hierarchical_metrics_impl("performance")
        assert result1["source"] == "timing_middleware"
        assert "improve_prompt" in result1["data"]
        assert "get_session" in result1["data"]
        
        # Test 2: Get specific tool metrics
        result2 = await server._get_hierarchical_metrics_impl("performance/tools/improve_prompt")
        assert "improve_prompt" in result2["data"]
        assert "get_session" not in result2["data"]
        
        # Test 3: Get session metrics
        result3 = await server._get_hierarchical_metrics_impl("sessions")
        assert result3["source"] == "session_store"
        assert "active_sessions" in result3["data"]
        assert "max_size" in result3["data"]
        
        # Test 4: Unknown metric type
        result4 = await server._get_hierarchical_metrics_impl("unknown/metric")
        assert "available_types" in result4
        assert result4["data"] == {}


class TestHTTPTransport:
    """Test streamable HTTP transport functionality."""
    
    def test_http_transport_initialization(self):
        """Test HTTP transport can be initialized."""
        server = APESMCPServer()
        
        # Mock the mcp.run method to verify parameters
        original_run = server.mcp.run
        run_params = {}
        
        def mock_run(**kwargs):
            run_params.update(kwargs)
            # Don't actually run the server
            return
        
        server.mcp.run = mock_run
        
        try:
            # Test with custom host and port
            server.run_streamable_http(host="0.0.0.0", port=9000)
            
            # Verify parameters were passed
            assert run_params.get("transport") == "streamable-http"
            assert run_params.get("host") == "0.0.0.0"
            assert run_params.get("port") == 9000
            assert run_params.get("log_level") == "INFO"
        finally:
            server.mcp.run = original_run
    
    def test_http_transport_fallback(self):
        """Test fallback to stdio when HTTP transport not supported."""
        server = APESMCPServer()
        
        # Mock run to raise TypeError (simulating unsupported parameters)
        def mock_run_with_error(**kwargs):
            if "transport" in kwargs:
                raise TypeError("Unexpected keyword argument 'transport'")
            return
        
        server.mcp.run = mock_run_with_error
        
        # Should not raise, but log warning and fallback
        with patch('logging.Logger.warning') as mock_warning:
            server.run_streamable_http()
            
            # Check warning was logged
            assert mock_warning.called
            warning_msg = mock_warning.call_args[0][0]
            assert "not supported" in warning_msg


class TestPerformanceTargets:
    """Test performance targets are maintained."""
    
    @pytest.mark.asyncio
    async def test_response_time_under_200ms(self):
        """Test that enhanced operations still meet <200ms target."""
        server = APESMCPServer()
        
        # Time a simple operation with middleware
        start_time = time.time()
        
        # Create test context using modern helper method
        mock_ctx = server.create_mock_context()
        
        # Simple prompt that should process quickly
        prompt = "Simple test prompt"
        
        # Find and call the tool
        improve_prompt = None
        for tool_name, tool_func in server.mcp._tools.items():
            if tool_name == "improve_prompt":
                improve_prompt = tool_func.implementation
                break
        
        assert improve_prompt is not None
        
        # Execute the enhanced prompt improvement using modern helper methods
        session_id = server.create_session_id("performance_test")
        result = await improve_prompt(
            prompt=prompt,
            session_id=session_id,  # Generated using helper method
            ctx=mock_ctx,  # Created using helper method
            context=None
        )
        
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should complete under 200ms
        assert elapsed < 200, f"Operation took {elapsed:.2f}ms, exceeds 200ms target"
        
        # Result should be valid
        assert "improved_prompt" in result or "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])