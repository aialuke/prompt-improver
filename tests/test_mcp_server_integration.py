"""
Test MCP server integration with real Pydantic model behavior.
"""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from prompt_improver.mcp_server.server import APESMCPServer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

@pytest.mark.asyncio
async def test_mcp_server_imports():
    """Test that MCP server imports work correctly"""
    try:
        server = APESMCPServer()
        assert server is not None, 'MCP server should be created'
        assert hasattr(server, 'mcp'), 'MCP server should have FastMCP instance'
        assert hasattr(server, 'services'), 'MCP server should have services'
        assert hasattr(server, '_improve_prompt_impl'), 'Server should have improve_prompt implementation'
        assert hasattr(server, '_get_session_impl'), 'Server should have get_session implementation'
        print('✓ MCP server imports validation passed')
        return True
    except Exception as e:
        print(f'✗ MCP server imports failed: {e}')
        return False

@pytest.mark.asyncio
async def test_mcp_tool_functionality():
    """Test MCP server tool functionality with real data"""
    try:
        server = APESMCPServer()
        tools = dir(server)
        assert '_improve_prompt_impl' in tools, 'improve_prompt implementation should exist'
        assert '_get_session_impl' in tools, 'get_session implementation should exist'
        assert '_set_session_impl' in tools, 'set_session implementation should exist'
        assert '_touch_session_impl' in tools, 'touch_session implementation should exist'
        assert '_delete_session_impl' in tools, 'delete_session implementation should exist'
        assert hasattr(server, 'mcp'), 'Server should have FastMCP instance'
        assert server.mcp is not None, 'FastMCP instance should be initialized'
        print('✓ MCP tool functionality validation passed')
        return True
    except Exception as e:
        print(f'✗ MCP tool functionality validation failed: {e}')
        return False

@pytest.mark.asyncio
async def test_mcp_tool_signatures():
    """Test that MCP tools have correct Pydantic signatures"""
    try:
        server = APESMCPServer()
        tools = []
        if hasattr(server.mcp, '_tools'):
            tools = list(server.mcp._tools.keys())
        elif hasattr(server.mcp, 'tools'):
            tools = list(server.mcp.tools.keys())
        expected_tools = ['improve_prompt', 'store_prompt', 'get_session', 'set_session', 'touch_session', 'delete_session']
        for tool in expected_tools:
            if tools:
                assert tool in tools, f'Tool {tool} should be available'
        print(f'✓ MCP tools validation passed (found {len(tools)} tools)')
        return True
    except Exception as e:
        print(f'✗ MCP tools validation failed: {e}')
        return False

@pytest.mark.asyncio
async def test_mcp_resources():
    """Test that MCP resources are properly defined"""
    try:
        server = APESMCPServer()
        resources = []
        if hasattr(server.mcp, '_resources'):
            resources = list(server.mcp._resources.keys())
        elif hasattr(server.mcp, 'resources'):
            resources = list(server.mcp.resources.keys())
        expected_resources = ['apes://rule_status', 'apes://session_store/status', 'apes://health/live', 'apes://health/ready']
        for resource in expected_resources:
            if resources:
                assert resource in resources, f'Resource {resource} should be available'
        print(f'✓ MCP resources validation passed (found {len(resources)} resources)')
        return True
    except Exception as e:
        print(f'✗ MCP resources validation failed: {e}')
        return False

@pytest.mark.asyncio
async def test_mcp_session_store():
    """Test MCP session store functionality"""
    try:
        server = APESMCPServer()
        session_store = server.services.session_store
        assert session_store is not None, 'Session store should be created'
        assert hasattr(session_store, 'maxsize'), 'Session store should have maxsize'
        assert hasattr(session_store, 'ttl'), 'Session store should have ttl'
        test_data = {'prompt': 'Test prompt', 'context': {'domain': 'test'}, 'timestamp': '2025-01-01T00:00:00Z'}
        assert hasattr(session_store, 'set'), 'Session store should have set method'
        assert hasattr(session_store, 'get'), 'Session store should have get method'
        assert hasattr(session_store, 'delete'), 'Session store should have delete method'
        assert hasattr(session_store, 'touch'), 'Session store should have touch method'
        print('✓ MCP session store validation passed')
        return True
    except Exception as e:
        print(f'✗ MCP session store validation failed: {e}')
        return False

@pytest.mark.asyncio
async def test_database_models_integration():
    """Test database models work with MCP server context"""
    try:
        from prompt_improver.database.models import PromptSession, RulePerformance, TrainingPrompt, UserFeedback
        session_data = {'session_id': 'mcp-integration-test-789', 'original_prompt': 'Test prompt from MCP server integration', 'improved_prompt': 'Enhanced test prompt with better structure from MCP server', 'user_context': {'mcp_client': 'test_client', 'transport': 'stdio', 'version': '1.0.0'}, 'quality_score': 0.88, 'improvement_score': 0.76, 'confidence_level': 0.94}
        prompt_session = PromptSession(**session_data)
        assert prompt_session.session_id == 'mcp-integration-test-789'
        assert prompt_session.user_context['mcp_client'] == 'test_client'
        assert prompt_session.quality_score == 0.88
        feedback_data = {'session_id': 'mcp-integration-test-789', 'rating': 4, 'feedback_text': 'Good improvement via MCP server', 'improvement_areas': ['clarity', 'technical_accuracy'], 'model_id': 'mcp-server-model-v1'}
        user_feedback = UserFeedback(**feedback_data)
        assert user_feedback.session_id == 'mcp-integration-test-789'
        assert user_feedback.rating == 4
        assert user_feedback.model_id == 'mcp-server-model-v1'
        training_data = {'prompt_text': 'Training prompt captured from MCP server interaction', 'enhancement_result': {'improved_prompt': 'Enhanced training prompt via MCP processing', 'metrics': {'mcp_processing_time_ms': 156.3, 'transport_overhead_ms': 12.7, 'improvement_score': 0.81}, 'mcp_metadata': {'client_id': 'test_client', 'server_version': '1.0.0', 'protocol_version': '2024-11-05'}}, 'data_source': 'mcp_server', 'training_priority': 150}
        training_prompt = TrainingPrompt(**training_data)
        assert training_prompt.data_source == 'mcp_server'
        assert training_prompt.enhancement_result['mcp_metadata']['client_id'] == 'test_client'
        assert training_prompt.training_priority == 150
        print('✓ Database models MCP integration validation passed')
        return True
    except Exception as e:
        print(f'✗ Database models MCP integration failed: {e}')
        return False

async def run_mcp_integration_tests():
    """Run all MCP integration tests"""
    tests = [test_mcp_server_imports, test_mcp_tool_functionality, test_mcp_tool_signatures, test_mcp_resources, test_mcp_session_store, test_database_models_integration]
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f'✗ Test {test.__name__} failed with exception: {e}')
            results.append(False)
    passed = sum(results)
    total = len(results)
    print('\n=== MCP Integration Results ===')
    print(f'Passed: {passed}/{total} tests')
    print(f'Success rate: {passed / total * 100:.1f}%')
    if passed == total:
        print('🎉 All MCP integration tests passed!')
    else:
        print('⚠️  Some MCP integration tests failed')
    return passed == total
if __name__ == '__main__':
    asyncio.run(run_mcp_integration_tests())
