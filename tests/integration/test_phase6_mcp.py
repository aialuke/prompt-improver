#!/usr/bin/env python3
"""
Test Phase 6 MCP tools directly.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_mcp_tools():
    """Test the MCP orchestrator tools."""
    print("🔧 Testing MCP Orchestrator Tools...")
    
    # Import MCP server class
    from prompt_improver.mcp_server.server import APESMCPServer
    
    # Create server instance
    server = APESMCPServer()
    
    # Test basic server functionality - using available MCP tools
    print("\n1. Testing server initialization...")
    try:
        # Initialize server
        initialized = await server.initialize()
        print(f"  ✅ Server initialized: {initialized}")
        print(f"  Server running: {server._is_running}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test available MCP tools
    print("\n2. Testing get_performance_status tool...")
    try:
        result = await server._get_performance_status_impl()
        print(f"  ✅ Performance status retrieved")
        print(f"  Status keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test training queue monitoring (read-only)
    print("\n3. Testing get_training_queue_size tool...")
    try:
        result = await server._get_training_queue_size_impl()
        print(f"  ✅ Queue size: {result.get('queue_size', 0)}")
        print(f"  Status: {result.get('status', 'unknown')}")
        print(f"  Processing rate: {result.get('processing_rate', 0.0)} items/sec")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test health check resource
    print("\n4. Testing health check resource...")
    try:
        result = await server._health_ready_impl()
        print(f"  ✅ Health status: {result.get('status', 'unknown')}")
        print(f"  Rule application ready: {result.get('rule_application', {}).get('ready', False)}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Cleanup
    await server.shutdown()

if __name__ == "__main__":
    asyncio.run(test_mcp_tools())