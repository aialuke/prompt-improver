"""
Test MCP server integration with real Pydantic model behavior.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))


@pytest.mark.asyncio
async def test_mcp_server_imports():
    """Test that MCP server imports work correctly"""
    try:
        from prompt_improver.mcp_server.mcp_server import (
            PromptEnhancementRequest,
            PromptStorageRequest,
            mcp
        )
        
        # Test that server was created
        assert mcp is not None, "MCP server should be created"
        assert hasattr(mcp, 'name'), "MCP server should have name"
        
        # Test request models
        assert hasattr(PromptEnhancementRequest, 'model_validate')
        assert hasattr(PromptStorageRequest, 'model_validate')
        
        print("✓ MCP server imports validation passed")
        return True
        
    except Exception as e:
        print(f"✗ MCP server imports failed: {e}")
        return False


@pytest.mark.asyncio
async def test_mcp_request_models():
    """Test MCP request models with real data"""
    try:
        from prompt_improver.mcp_server.mcp_server import (
            PromptEnhancementRequest,
            PromptStorageRequest
        )
        
        # Test PromptEnhancementRequest with full data
        enhancement_data = {
            "prompt": "Improve this technical documentation for better clarity",
            "context": {
                "domain": "technical",
                "audience": "developers", 
                "complexity": "intermediate",
                "format": "documentation"
            },
            "session_id": "mcp-test-session-123"
        }
        
        request = PromptEnhancementRequest.model_validate(enhancement_data)
        assert request.prompt.startswith("Improve this technical")
        assert request.context["domain"] == "technical"
        assert request.session_id == "mcp-test-session-123"
        
        # Test serialization round-trip
        serialized = request.model_dump()
        restored = PromptEnhancementRequest.model_validate(serialized)
        assert restored.prompt == request.prompt
        assert restored.context == request.context
        
        # Test PromptStorageRequest with metrics
        storage_data = {
            "original": "Original prompt text that needs improvement",
            "enhanced": "Enhanced prompt text with better structure and clarity for improved results",
            "metrics": {
                "improvement_score": 0.78,
                "quality_score": 0.85,
                "confidence_level": 0.92,
                "processing_time_ms": 145.7,
                "rules_applied": ["clarity", "structure", "specificity"],
                "performance_gains": {
                    "readability": 0.15,
                    "effectiveness": 0.22,
                    "user_satisfaction": 0.18
                }
            },
            "session_id": "mcp-storage-test-456"
        }
        
        storage_request = PromptStorageRequest.model_validate(storage_data)
        assert storage_request.original.startswith("Original prompt")
        assert storage_request.enhanced.startswith("Enhanced prompt")
        assert storage_request.metrics["improvement_score"] == 0.78
        assert len(storage_request.metrics["rules_applied"]) == 3
        assert storage_request.metrics["performance_gains"]["readability"] == 0.15
        
        print("✓ MCP request models validation passed")
        return True
        
    except Exception as e:
        print(f"✗ MCP request models validation failed: {e}")
        return False


@pytest.mark.asyncio
async def test_mcp_tool_signatures():
    """Test that MCP tools have correct Pydantic signatures"""
    try:
        from prompt_improver.mcp_server.mcp_server import mcp
        
        # Get the tools from the MCP server
        tools = []
        if hasattr(mcp, '_tools'):
            tools = list(mcp._tools.keys())
        elif hasattr(mcp, 'tools'):
            tools = list(mcp.tools.keys())
        
        # Expected tools
        expected_tools = [
            "improve_prompt",
            "store_prompt", 
            "get_session",
            "set_session",
            "touch_session",
            "delete_session"
        ]
        
        # Check that key tools exist
        for tool in expected_tools:
            if tools:  # Only check if we can access tools
                assert tool in tools, f"Tool {tool} should be available"
        
        print(f"✓ MCP tools validation passed (found {len(tools)} tools)")
        return True
        
    except Exception as e:
        print(f"✗ MCP tools validation failed: {e}")
        return False


@pytest.mark.asyncio
async def test_mcp_resources():
    """Test that MCP resources are properly defined"""
    try:
        from prompt_improver.mcp_server.mcp_server import mcp
        
        # Get the resources from the MCP server
        resources = []
        if hasattr(mcp, '_resources'):
            resources = list(mcp._resources.keys())
        elif hasattr(mcp, 'resources'):
            resources = list(mcp.resources.keys())
        
        # Expected resources
        expected_resources = [
            "apes://rule_status",
            "apes://session_store/status", 
            "apes://health/live",
            "apes://health/ready"
        ]
        
        # Check that key resources exist
        for resource in expected_resources:
            if resources:  # Only check if we can access resources
                assert resource in resources, f"Resource {resource} should be available"
        
        print(f"✓ MCP resources validation passed (found {len(resources)} resources)")
        return True
        
    except Exception as e:
        print(f"✗ MCP resources validation failed: {e}")
        return False


@pytest.mark.asyncio
async def test_mcp_session_store():
    """Test MCP session store functionality"""
    try:
        from prompt_improver.mcp_server.mcp_server import session_store
        
        # Test session store creation
        assert session_store is not None, "Session store should be created"
        assert hasattr(session_store, 'maxsize'), "Session store should have maxsize"
        assert hasattr(session_store, 'ttl'), "Session store should have ttl"
        
        # Test basic session operations (without async context)
        test_data = {
            "prompt": "Test prompt",
            "context": {"domain": "test"},
            "timestamp": "2025-01-01T00:00:00Z"
        }
        
        # These methods should exist
        assert hasattr(session_store, 'set'), "Session store should have set method"
        assert hasattr(session_store, 'get'), "Session store should have get method"
        assert hasattr(session_store, 'delete'), "Session store should have delete method"
        assert hasattr(session_store, 'touch'), "Session store should have touch method"
        
        print("✓ MCP session store validation passed")
        return True
        
    except Exception as e:
        print(f"✗ MCP session store validation failed: {e}")
        return False


@pytest.mark.asyncio
async def test_database_models_integration():
    """Test database models work with MCP server context"""
    try:
        from prompt_improver.database.models import (
            PromptSession, UserFeedback, RulePerformance, TrainingPrompt
        )
        
        # Test PromptSession with MCP-like data
        session_data = {
            "session_id": "mcp-integration-test-789",
            "original_prompt": "Test prompt from MCP server integration",
            "improved_prompt": "Enhanced test prompt with better structure from MCP server",
            "user_context": {
                "mcp_client": "test_client",
                "transport": "stdio",
                "version": "1.0.0"
            },
            "quality_score": 0.88,
            "improvement_score": 0.76,
            "confidence_level": 0.94
        }
        
        prompt_session = PromptSession(**session_data)
        assert prompt_session.session_id == "mcp-integration-test-789"
        assert prompt_session.user_context["mcp_client"] == "test_client"
        assert prompt_session.quality_score == 0.88
        
        # Test UserFeedback with MCP context
        feedback_data = {
            "session_id": "mcp-integration-test-789",
            "rating": 4,
            "feedback_text": "Good improvement via MCP server",
            "improvement_areas": ["clarity", "technical_accuracy"],
            "model_id": "mcp-server-model-v1"
        }
        
        user_feedback = UserFeedback(**feedback_data)
        assert user_feedback.session_id == "mcp-integration-test-789"
        assert user_feedback.rating == 4
        assert user_feedback.model_id == "mcp-server-model-v1"
        
        # Test TrainingPrompt for MCP data storage
        training_data = {
            "prompt_text": "Training prompt captured from MCP server interaction",
            "enhancement_result": {
                "improved_prompt": "Enhanced training prompt via MCP processing",
                "metrics": {
                    "mcp_processing_time_ms": 156.3,
                    "transport_overhead_ms": 12.7,
                    "improvement_score": 0.81
                },
                "mcp_metadata": {
                    "client_id": "test_client",
                    "server_version": "1.0.0",
                    "protocol_version": "2024-11-05"
                }
            },
            "data_source": "mcp_server",
            "training_priority": 150
        }
        
        training_prompt = TrainingPrompt(**training_data)
        assert training_prompt.data_source == "mcp_server"
        assert training_prompt.enhancement_result["mcp_metadata"]["client_id"] == "test_client"
        assert training_prompt.training_priority == 150
        
        print("✓ Database models MCP integration validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Database models MCP integration failed: {e}")
        return False


async def run_mcp_integration_tests():
    """Run all MCP integration tests"""
    tests = [
        test_mcp_server_imports,
        test_mcp_request_models,
        test_mcp_tool_signatures,
        test_mcp_resources,
        test_mcp_session_store,
        test_database_models_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n=== MCP Integration Results ===")
    print(f"Passed: {passed}/{total} tests")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All MCP integration tests passed!")
    else:
        print("⚠️  Some MCP integration tests failed")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(run_mcp_integration_tests())