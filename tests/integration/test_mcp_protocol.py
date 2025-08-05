#!/usr/bin/env python3
"""
Test MCP protocol compliance for APES MCP Server.
This script tests the actual stdio MCP protocol interaction.
"""

import asyncio
import json
import os
import subprocess
import sys
import time

def test_mcp_protocol():
    """Test MCP protocol compliance with real stdio interaction."""
    print("=" * 60)
    print("APES MCP Server Protocol Compliance Test")
    print("=" * 60)
    
    # Set environment variables
    env = os.environ.copy()
    env['DATABASE_URL'] = "postgresql://apes_user:apes_secure_password_2024@localhost:5432/apes_production"
    env['PYTHONPATH'] = "/Users/lukemckenzie/prompt-improver/src"
    
    try:
        print("🚀 Starting MCP server process...")
        
        # Start the MCP server process
        process = subprocess.Popen(
            [sys.executable, '-m', 'prompt_improver.mcp_server.server'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            cwd="/Users/lukemckenzie/prompt-improver"
        )
        
        # Test initialization
        print("📤 Sending initialization request...")
        
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        # Send request
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        # Wait for response with timeout
        start_time = time.time()
        timeout = 10  # 10 seconds
        
        response_line = None
        while time.time() - start_time < timeout:
            if process.poll() is not None:
                # Process ended
                break
                
            # Try to read a line
            try:
                response_line = process.stdout.readline()
                if response_line.strip():
                    break
            except:
                pass
            
            time.sleep(0.1)
        
        if response_line and response_line.strip():
            try:
                response = json.loads(response_line.strip())
                print(f"✅ Received initialization response: {response.get('id')} - {response.get('result', {}).get('serverInfo', {}).get('name', 'Unknown')}")
                
                # Test tool listing
                print("📤 Requesting tools list...")
                
                tools_request = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list"
                }
                
                process.stdin.write(json.dumps(tools_request) + "\n")
                process.stdin.flush()
                
                # Read tools response
                tools_response_line = process.stdout.readline()
                if tools_response_line.strip():
                    tools_response = json.loads(tools_response_line.strip())
                    tools = tools_response.get('result', {}).get('tools', [])
                    print(f"✅ Tools available: {len(tools)} tools")
                    for tool in tools[:5]:  # Show first 5 tools
                        print(f"   - {tool.get('name', 'unknown')}: {tool.get('description', 'no description')[:50]}")
                    if len(tools) > 5:
                        print(f"   ... and {len(tools) - 5} more tools")
                
                # Test resource listing
                print("📤 Requesting resources list...")
                
                resources_request = {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "resources/list"
                }
                
                process.stdin.write(json.dumps(resources_request) + "\n")
                process.stdin.flush()
                
                # Read resources response
                resources_response_line = process.stdout.readline()
                if resources_response_line.strip():
                    resources_response = json.loads(resources_response_line.strip())
                    resources = resources_response.get('result', {}).get('resources', [])
                    print(f"✅ Resources available: {len(resources)} resources")
                    for resource in resources[:5]:  # Show first 5 resources
                        print(f"   - {resource.get('name', 'unknown')}: {resource.get('description', 'no description')[:50]}")
                    if len(resources) > 5:
                        print(f"   ... and {len(resources) - 5} more resources")
                
                print("✅ MCP protocol compliance test PASSED")
                result = True
                
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON response: {e}")
                print(f"Raw response: {response_line}")
                result = False
        else:
            print("❌ No response received from MCP server")
            result = False
        
        # Clean up
        print("🧹 Terminating server process...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        
        # Check for any errors
        stderr_output = process.stderr.read()
        if stderr_output:
            print(f"\n📋 Server stderr output (first 500 chars):\n{stderr_output[:500]}")
        
        return result
        
    except Exception as e:
        print(f"❌ Protocol test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mcp_protocol()
    if success:
        print("\n🎉 MCP Protocol compliance test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ MCP Protocol compliance test failed!")
        sys.exit(1)