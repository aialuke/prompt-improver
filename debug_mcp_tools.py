#!/usr/bin/env python3
"""
Debug script to check MCP server capabilities and tool formats
"""
import asyncio
import json
import subprocess
import sys
import time

async def debug_mcp_server():
    """Debug MCP server capabilities and tool formats"""
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
                "clientInfo": {"name": "debug-client", "version": "1.0.0"}
            }
        }
        
        # Send initialization request
        request_json = json.dumps(init_request) + "\n"
        print(f"Sending init: {request_json}")
        process.stdin.write(request_json)
        process.stdin.flush()
        
        # Read init response
        await asyncio.sleep(1)
        if process.stdout.readable():
            response_line = process.stdout.readline()
            print(f"Init response: {response_line}")
            if response_line:
                response = json.loads(response_line.strip())
                print(f"Init parsed: {json.dumps(response, indent=2)}")
        
        # Test tools/list to see available tools
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        request_json = json.dumps(tools_request) + "\n"
        print(f"Sending tools/list: {request_json}")
        process.stdin.write(request_json)
        process.stdin.flush()
        
        # Read tools response
        await asyncio.sleep(1)
        if process.stdout.readable():
            response_line = process.stdout.readline()
            print(f"Tools response: {response_line}")
            if response_line:
                response = json.loads(response_line.strip())
                print(f"Tools parsed: {json.dumps(response, indent=2)}")
        
        # Test a simple tool call with various formats
        test_formats = [
            # Format 1: Direct arguments
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "improve_prompt",
                    "arguments": {
                        "prompt": "Help me debug this",
                        "context": {"domain": "debugging"},
                        "session_id": "debug_test"
                    }
                }
            },
            # Format 2: Arguments in different structure
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "improve_prompt",
                    "prompt": "Help me debug this",
                    "context": {"domain": "debugging"},
                    "session_id": "debug_test"
                }
            }
        ]
        
        for i, test_format in enumerate(test_formats):
            request_json = json.dumps(test_format) + "\n"
            print(f"Testing format {i+1}: {request_json}")
            process.stdin.write(request_json)
            process.stdin.flush()
            
            # Read response
            await asyncio.sleep(1)
            if process.stdout.readable():
                response_line = process.stdout.readline()
                print(f"Format {i+1} response: {response_line}")
                if response_line:
                    try:
                        response = json.loads(response_line.strip())
                        print(f"Format {i+1} parsed: {json.dumps(response, indent=2)}")
                    except json.JSONDecodeError:
                        print(f"Format {i+1} JSON decode error")
        
        return True
        
    except Exception as e:
        print(f"Error during debugging: {e}")
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
    result = asyncio.run(debug_mcp_server())
    print(f"Debug result: {'SUCCESS' if result else 'FAILED'}")
