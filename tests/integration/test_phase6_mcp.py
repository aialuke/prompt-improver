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
    
    # Import MCP server module
    from prompt_improver.mcp_server import mcp_server
    
    # Test get_orchestrator_status
    print("\n1. Testing get_orchestrator_status()...")
    try:
        result = await mcp_server.get_orchestrator_status()
        print(f"  ✅ Status: {result.get('state', 'unknown')}")
        print(f"  Initialized: {result.get('initialized', False)}")
        print(f"  Active workflows: {result.get('active_workflows', 0)}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test initialize_orchestrator
    print("\n2. Testing initialize_orchestrator()...")
    try:
        result = await mcp_server.initialize_orchestrator()
        print(f"  ✅ Success: {result.get('success', False)}")
        print(f"  Loaded components: {result.get('loaded_components', 0)}")
        if result.get('component_list'):
            print(f"  Sample components: {result['component_list'][:3]}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test run_ml_training_workflow
    print("\n3. Testing run_ml_training_workflow()...")
    try:
        result = await mcp_server.run_ml_training_workflow("test training data")
        print(f"  ✅ Success: {result.get('success', False)}")
        if result.get('success'):
            print(f"  Execution time: {result.get('execution_time', 0):.2f}s")
            print(f"  Steps completed: {result.get('steps_completed', 0)}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test invoke_ml_component
    print("\n4. Testing invoke_ml_component()...")
    try:
        # Try to invoke a simple component method
        result = await mcp_server.invoke_ml_component(
            component_name="training_data_loader",
            method_name="load_training_data"
        )
        print(f"  Success: {result.get('success', False)}")
        if not result.get('success'):
            print(f"  Error: {result.get('error', 'Unknown')}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_mcp_tools())