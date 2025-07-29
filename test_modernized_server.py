#!/usr/bin/env python3
"""
Test script for the modernized MCP server.
Verifies that the class-based architecture works correctly.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_server_import():
    """Test that the modernized server can be imported."""
    try:
        from prompt_improver.mcp_server.mcp_server import APESMCPServer
        print("✅ APESMCPServer class imports successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import APESMCPServer: {e}")
        return False

def test_server_instantiation():
    """Test that the server can be instantiated (without running)."""
    try:
        # Set minimal required environment variables
        os.environ.setdefault('MCP_JWT_SECRET_KEY', 'test-key-for-testing-only')
        os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost:5432/test')
        
        from prompt_improver.mcp_server.mcp_server import APESMCPServer
        
        # Try to create server instance
        server = APESMCPServer()
        print("✅ APESMCPServer instantiates successfully")
        print(f"✅ Server has {len(server.mcp.tools)} tools registered")
        print(f"✅ Server has {len(server.mcp.resources)} resources registered")
        
        # Verify key components
        assert hasattr(server, 'services'), "Server should have services container"
        assert hasattr(server, 'mcp'), "Server should have FastMCP instance"
        assert hasattr(server, 'initialize'), "Server should have initialize method"
        assert hasattr(server, 'shutdown'), "Server should have shutdown method"
        assert hasattr(server, 'run'), "Server should have run method"
        
        print("✅ All required methods and attributes present")
        return True
        
    except Exception as e:
        print(f"❌ Failed to instantiate APESMCPServer: {e}")
        return False

def test_architecture_features():
    """Test that modern architecture features are present."""
    try:
        from prompt_improver.mcp_server.mcp_server import APESMCPServer, ServerServices
        
        print("✅ ServerServices dataclass available")
        print("✅ Class-based architecture implemented")
        
        # Check that old functional code is removed
        try:
            from prompt_improver.mcp_server.mcp_server import mcp
            print("❌ Old functional 'mcp' variable still exists")
            return False
        except ImportError:
            print("✅ Old functional code properly removed")
        
        return True
        
    except Exception as e:
        print(f"❌ Architecture test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Modernized MCP Server")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_server_import),
        ("Instantiation Test", test_server_instantiation),
        ("Architecture Test", test_architecture_features),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"💥 {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"🏆 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Modernized server is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
