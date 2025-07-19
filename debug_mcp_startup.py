#!/usr/bin/env python3
"""
Enhanced debug script to capture server startup logs and diagnose issues
"""
import asyncio
import json
import subprocess
import sys
import time
import os

async def debug_mcp_startup():
    """Debug MCP server startup with detailed logging"""
    # Set environment variable
    env = os.environ.copy()
    env['DATABASE_URL'] = "postgresql+asyncpg://apes_user:apes_secure_password_2024@localhost:5432/apes_production"
    
    # Start the MCP server with stderr capture
    process = subprocess.Popen(
        [sys.executable, "src/prompt_improver/mcp_server/mcp_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env
    )
    
    try:
        # Wait for server startup and capture stderr
        await asyncio.sleep(3)
        
        # Check if process is still running
        if process.poll() is not None:
            stderr = process.stderr.read()
            stdout = process.stdout.read()
            print(f"Server terminated early!")
            print(f"Return code: {process.returncode}")
            print(f"STDERR: {stderr}")
            print(f"STDOUT: {stdout}")
            return False
        
        # Capture any stderr output (server logs)
        stderr_output = ""
        if process.stderr.readable():
            # Non-blocking read of stderr
            import select
            ready, _, _ = select.select([process.stderr], [], [], 0.1)
            if ready:
                stderr_output = process.stderr.read(1024)
                if stderr_output:
                    print(f"Server logs (stderr): {stderr_output}")
        
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
        response_line = process.stdout.readline()
        print(f"Init response: {response_line}")
        
        if response_line:
            response = json.loads(response_line.strip())
            print(f"Init parsed: {json.dumps(response, indent=2)}")
            
            if "result" in response:
                print("✓ Server initialization successful")
                
                # Now test tools/list with proper params
                tools_request = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {}
                }
                
                request_json = json.dumps(tools_request) + "\n"
                print(f"Sending tools/list (no params): {request_json}")
                process.stdin.write(request_json)
                process.stdin.flush()
                
                await asyncio.sleep(1)
                response_line = process.stdout.readline()
                print(f"Tools response: {response_line}")
                
                if response_line:
                    response = json.loads(response_line.strip())
                    print(f"Tools parsed: {json.dumps(response, indent=2)}")
                    
                    if "result" in response:
                        print("✓ Tools listing successful")
                        tools = response["result"]["tools"]
                        print(f"Available tools: {[tool['name'] for tool in tools]}")
                        
                        # Test actual tool call
                        if tools:
                            tool_name = tools[0]['name']
                            tool_call = {
                                "jsonrpc": "2.0",
                                "id": 3,
                                "method": "tools/call",
                                "params": {
                                    "name": tool_name,
                                    "arguments": {}
                                }
                            }
                            
                            request_json = json.dumps(tool_call) + "\n"
                            print(f"Testing tool call: {request_json}")
                            process.stdin.write(request_json)
                            process.stdin.flush()
                            
                            await asyncio.sleep(2)
                            response_line = process.stdout.readline()
                            print(f"Tool call response: {response_line}")
                    else:
                        print("✗ Tools listing failed")
                        print(f"Error: {response.get('error', 'Unknown error')}")
            else:
                print("✗ Server initialization failed")
                print(f"Error: {response.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"Error during debugging: {e}")
        import traceback
        traceback.print_exc()
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
    result = asyncio.run(debug_mcp_startup())
    print(f"Debug result: {'SUCCESS' if result else 'FAILED'}")
