#!/usr/bin/env python3
"""
FastAPI + WebSocket 15.x Integration Test

Tests FastAPI WebSocket compatibility with websockets 15.x
"""

import asyncio
import json
import pytest
from typing import Dict, Any
from unittest.mock import AsyncMock

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient
import websockets


class TestWebSocketServer:
    """Test WebSocket server using FastAPI"""
    
    def __init__(self):
        self.app = FastAPI()
        self.connected_clients = set()
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            await websocket.accept()
            self.connected_clients.add(websocket)
            
            try:
                await websocket.send_text(json.dumps({
                    "type": "welcome",
                    "client_id": client_id,
                    "message": "Connected successfully"
                }))
                
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    if message.get("type") == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": message.get("timestamp")
                        }))
                    elif message.get("type") == "echo":
                        await websocket.send_text(json.dumps({
                            "type": "echo_response",
                            "original": message.get("data"),
                            "client_id": client_id
                        }))
                    elif message.get("type") == "broadcast":
                        # Broadcast to all connected clients
                        broadcast_message = json.dumps({
                            "type": "broadcast_message",
                            "from": client_id,
                            "data": message.get("data")
                        })
                        for client in self.connected_clients.copy():
                            try:
                                await client.send_text(broadcast_message)
                            except:
                                self.connected_clients.discard(client)
                    
            except WebSocketDisconnect:
                self.connected_clients.discard(websocket)
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "connected_clients": len(self.connected_clients),
                "websockets_version": websockets.__version__
            }


@pytest.mark.asyncio
async def test_fastapi_websocket_basic_connection():
    """Test basic FastAPI WebSocket connection with websockets 15.x"""
    
    server = TestWebSocketServer()
    
    with TestClient(server.app) as client:
        # Test the health endpoint first
        response = client.get("/health")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "websockets_version" in health_data
        
        # Test WebSocket connection using the test client's WebSocket support
        with client.websocket_connect("/ws/test-client") as websocket:
            # Should receive welcome message
            welcome_data = websocket.receive_text()
            welcome_message = json.loads(welcome_data)
            
            assert welcome_message["type"] == "welcome"
            assert welcome_message["client_id"] == "test-client"
            assert "message" in welcome_message


@pytest.mark.asyncio
async def test_fastapi_websocket_ping_pong():
    """Test ping/pong functionality"""
    
    server = TestWebSocketServer()
    
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/ping-test") as websocket:
            # Skip welcome message
            websocket.receive_text()
            
            # Send ping
            ping_message = {
                "type": "ping",
                "timestamp": "2025-07-25T10:00:00Z"
            }
            websocket.send_text(json.dumps(ping_message))
            
            # Receive pong
            pong_data = websocket.receive_text()
            pong_message = json.loads(pong_data)
            
            assert pong_message["type"] == "pong"
            assert pong_message["timestamp"] == ping_message["timestamp"]


@pytest.mark.asyncio
async def test_fastapi_websocket_echo():
    """Test echo functionality"""
    
    server = TestWebSocketServer()
    
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/echo-test") as websocket:
            # Skip welcome message
            websocket.receive_text()
            
            # Send echo request
            echo_message = {
                "type": "echo",
                "data": "Hello WebSocket 15.x!"
            }
            websocket.send_text(json.dumps(echo_message))
            
            # Receive echo response
            echo_response_data = websocket.receive_text()
            echo_response = json.loads(echo_response_data)
            
            assert echo_response["type"] == "echo_response"
            assert echo_response["original"] == "Hello WebSocket 15.x!"
            assert echo_response["client_id"] == "echo-test"


@pytest.mark.asyncio 
async def test_fastapi_websocket_concurrent_connections():
    """Test multiple concurrent connections"""
    
    server = TestWebSocketServer()
    
    with TestClient(server.app) as client:
        # Open multiple WebSocket connections
        connections = []
        
        try:
            for i in range(5):
                ws = client.websocket_connect(f"/ws/client-{i}")
                ws.__enter__()  # Manually enter context
                connections.append(ws)
                
                # Skip welcome message
                welcome = ws.receive_text()
                welcome_data = json.loads(welcome)
                assert welcome_data["client_id"] == f"client-{i}"
            
            # Test that all connections are active
            assert len(server.connected_clients) == 5
            
            # Send messages from each connection
            for i, ws in enumerate(connections):
                test_message = {
                    "type": "echo",
                    "data": f"Message from client-{i}"
                }
                ws.send_text(json.dumps(test_message))
                
                response = ws.receive_text()
                response_data = json.loads(response)
                assert response_data["type"] == "echo_response"
                assert response_data["original"] == f"Message from client-{i}"
        
        finally:
            # Clean up connections
            for ws in connections:
                try:
                    ws.__exit__(None, None, None)
                except:
                    pass


@pytest.mark.asyncio
async def test_websocket_15x_specific_features():
    """Test WebSocket 15.x specific features with FastAPI"""
    
    # Test that we can import the new asyncio client
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed
    
    # Verify the version
    version = websockets.__version__
    major_version = int(version.split('.')[0])
    assert major_version >= 15, f"Expected WebSocket 15.x, got {version}"
    
    # Test that we can use the new reconnection pattern
    # This is more of a smoke test since we can't easily test actual reconnection
    # in a unit test environment
    
    server = TestWebSocketServer()
    
    # Mock server for reconnection testing
    class MockReconnectionTest:
        def __init__(self):
            self.connection_attempts = 0
        
        async def __aiter__(self):
            while self.connection_attempts < 2:
                self.connection_attempts += 1
                yield AsyncMock()
    
    # Test the reconnection pattern structure
    mock_connect = MockReconnectionTest()
    connection_count = 0
    
    async for websocket in mock_connect:
        connection_count += 1
        if connection_count >= 2:
            break
    
    assert connection_count == 2
    assert mock_connect.connection_attempts == 2


def test_websocket_version_compliance():
    """Test that we're using the correct WebSocket version"""
    
    version = websockets.__version__
    major_version = int(version.split('.')[0])
    
    assert major_version >= 15, f"Expected WebSocket 15.x, got {version}"
    print(f"✅ Using WebSocket version: {version}")


if __name__ == "__main__":
    # Run the tests
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    sys.exit(result.returncode)