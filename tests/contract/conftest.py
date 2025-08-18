"""
Contract test configuration and fixtures.

Provides contract testing infrastructure for API and service boundary validation.
Tests verify API contracts, schema compliance, and protocol adherence.
"""

import json
import os
from typing import Dict, Any
from uuid import uuid4

import pytest
import requests
import msgspec
from msgspec import ValidationError as MsgspecValidationError
from pydantic import BaseModel, ValidationError as PydanticValidationError

# API Contract Testing
@pytest.fixture
def api_base_url():
    """Base URL for API contract testing."""
    return os.environ.get("TEST_API_BASE_URL", "http://localhost:8000")

@pytest.fixture
def api_client(api_base_url):
    """HTTP client for API contract testing."""
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "Accept": "application/json"
    })
    
    class APIClient:
        def __init__(self, base_url: str, session: requests.Session):
            self.base_url = base_url
            self.session = session
        
        def post(self, endpoint: str, data: Dict[str, Any] = None, **kwargs):
            url = f"{self.base_url}{endpoint}"
            return self.session.post(url, json=data, **kwargs)
        
        def get(self, endpoint: str, params: Dict[str, Any] = None, **kwargs):
            url = f"{self.base_url}{endpoint}"
            return self.session.get(url, params=params, **kwargs)
        
        def put(self, endpoint: str, data: Dict[str, Any] = None, **kwargs):
            url = f"{self.base_url}{endpoint}"
            return self.session.put(url, json=data, **kwargs)
        
        def delete(self, endpoint: str, **kwargs):
            url = f"{self.base_url}{endpoint}"
            return self.session.delete(url, **kwargs)
    
    return APIClient(api_base_url, session)

# Schema Validation
@pytest.fixture
def api_schemas():
    """API schema definitions for contract validation."""
    return {
        "prompt_improvement_request": {
            "type": "object",
            "required": ["prompt", "session_id"],
            "properties": {
                "prompt": {"type": "string", "minLength": 1},
                "session_id": {"type": "string"},
                "context": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string"},
                        "language": {"type": "string"},
                        "complexity": {"type": "string", "enum": ["low", "medium", "high"]}
                    }
                }
            }
        },
        "prompt_improvement_response": {
            "type": "object",
            "required": ["improved_prompt", "confidence", "processing_time_ms"],
            "properties": {
                "improved_prompt": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rules_applied": {"type": "array", "items": {"type": "string"}},
                "processing_time_ms": {"type": "integer", "minimum": 0},
                "metadata": {"type": "object"}
            }
        },
        "health_check_response": {
            "type": "object",
            "required": ["status", "timestamp"],
            "properties": {
                "status": {"type": "string", "enum": ["healthy", "degraded", "unhealthy"]},
                "timestamp": {"type": "string"},
                "services": {
                    "type": "object",
                    "properties": {
                        "database": {"type": "string", "enum": ["healthy", "unhealthy"]},
                        "redis": {"type": "string", "enum": ["healthy", "unhealthy"]},
                        "ml_service": {"type": "string", "enum": ["healthy", "unhealthy"]}
                    }
                },
                "version": {"type": "string"}
            }
        }
    }

@pytest.fixture
def schema_validator(api_schemas):
    """Schema validation helper."""
    def validate_schema(data: Dict[str, Any], schema_name: str) -> bool:
        """Validate data against named schema."""
        if schema_name not in api_schemas:
            raise ValueError(f"Unknown schema: {schema_name}")
        
        try:
            # Convert schema to msgspec Struct for validation
            schema_struct = msgspec.convert(api_schemas[schema_name], type=dict)
            msgspec.validate(data, type=type(schema_struct))
            return True
        except MsgspecValidationError as e:
            pytest.fail(f"Schema validation failed for {schema_name}: {str(e)}")
    
    return validate_schema

# MCP Protocol Testing
@pytest.fixture
def mcp_client():
    """MCP client for protocol contract testing."""
    import asyncio
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    
    class MCPTestClient:
        def __init__(self):
            self.client = None
            self.session = None
        
        async def connect(self, command: list):
            """Connect to MCP server."""
            self.client = await stdio_client(command)
            self.session = ClientSession(self.client)
            await self.session.initialize()
        
        async def call_tool(self, name: str, arguments: Dict[str, Any]):
            """Call MCP tool."""
            return await self.session.call_tool(name, arguments)
        
        async def disconnect(self):
            """Disconnect from MCP server."""
            if self.client:
                await self.client.close()
    
    return MCPTestClient()

@pytest.fixture
def mcp_schemas():
    """MCP protocol schemas for validation."""
    return {
        "tool_call_request": {
            "type": "object",
            "required": ["method", "params"],
            "properties": {
                "method": {"type": "string", "const": "tools/call"},
                "params": {
                    "type": "object",
                    "required": ["name", "arguments"],
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "object"}
                    }
                }
            }
        },
        "tool_call_response": {
            "type": "object",
            "required": ["content"],
            "properties": {
                "content": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {"type": "string", "enum": ["text", "image", "resource"]},
                            "text": {"type": "string"}
                        }
                    }
                },
                "isError": {"type": "boolean"}
            }
        }
    }

# Contract Test Data
@pytest.fixture
def contract_test_data():
    """Test data for contract validation."""
    return {
        "valid_prompt_request": {
            "prompt": "Fix this bug in the code",
            "session_id": str(uuid4()),
            "context": {
                "domain": "software_development",
                "language": "python",
                "complexity": "medium"
            }
        },
        "invalid_prompt_request": {
            "prompt": "",  # Invalid: empty prompt
            "session_id": str(uuid4())
        },
        "valid_mcp_tool_call": {
            "method": "tools/call",
            "params": {
                "name": "improve_prompt",
                "arguments": {
                    "prompt": "Make this better",
                    "context": {"domain": "general"}
                }
            }
        }
    }

# Performance requirements for contract tests
@pytest.fixture
def performance_requirements():
    """Performance requirements for contract validation."""
    return {
        "api_response_time_ms": 500,  # API responses should be < 500ms
        "mcp_response_time_ms": 200,  # MCP responses should be < 200ms
        "concurrent_requests": 10     # Should handle 10 concurrent requests
    }

# Backward compatibility testing
@pytest.fixture
def api_versions():
    """API version compatibility matrix."""
    return {
        "current": "v1",
        "supported": ["v1"],
        "deprecated": [],
        "breaking_changes": {}
    }

def pytest_configure(config):
    """Configure contract test markers."""
    config.addinivalue_line("markers", "contract: contract test")
    config.addinivalue_line("markers", "api: API contract test")
    config.addinivalue_line("markers", "mcp: MCP protocol contract test")
    config.addinivalue_line("markers", "schema: schema validation test")
    config.addinivalue_line("markers", "performance: performance contract test")