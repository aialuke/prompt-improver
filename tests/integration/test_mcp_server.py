#!/usr/bin/env python3
"""
Quick test script to verify MCP server behavior
"""
import asyncio
import json
import subprocess
import sys
import time

async def test_mcp_server():
    """Test the MCP server startup and basic functionality"""
    # Start the MCP server
    process = subprocess.Popen(
        [sys.executable, "src/prompt_improver/mcp_server/mcp_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    try:
        # Wait for server startup
        await asyncio.sleep(2)
        
        if process.poll() is not None:
            stderr = process.stderr.read()
            stdout = process.stdout.read()
            print(f"Server failed to start. stderr: {stderr}")
            print(f"stdout: {stdout}")
            return False
        
        # Test initialization
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}, "resources": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        # Send initialization request
        request_json = json.dumps(init_request) + "\n"
        print(f"Sending: {request_json}")
        process.stdin.write(request_json)
        process.stdin.flush()
        
        # Wait for response
        await asyncio.sleep(1)
        
        # Try to read response
        if process.stdout.readable():
            response_line = process.stdout.readline()
            print(f"Response: {response_line}")
            if response_line:
                response = json.loads(response_line.strip())
                print(f"Parsed response: {response}")
                
                if "result" in response:
                    print("✓ Server initialized successfully")
                    return True
                else:
                    print("✗ Server initialization failed")
                    return False
        
        print("✗ No response received")
        return False
        
    except Exception as e:
        print(f"Error during test: {e}")
        return False
        
    finally:
        process.terminate()
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, process.wait),
                timeout=3
            )
        except asyncio.TimeoutError:
            process.kill()

if __name__ == "__main__":
    result = asyncio.run(test_mcp_server())
    print(f"Test result: {'PASSED' if result else 'FAILED'}")
