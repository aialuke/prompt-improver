#!/usr/bin/env python3
"""
Test Phase 6 API endpoints.
"""

import asyncio
import httpx

async def test_api_endpoints():
    """Test the orchestrator API endpoints."""
    print("🌐 Testing Orchestrator API Endpoints...")
    
    base_url = "http://localhost:5000/api/v1/experiments/real-time"
    
    async with httpx.AsyncClient() as client:
        # Test orchestrator status endpoint
        try:
            response = await client.get(f"{base_url}/orchestrator/status")
            print(f"\n✅ GET /orchestrator/status - Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  State: {data['data']['state']}")
                print(f"  Initialized: {data['data']['initialized']}")
        except Exception as e:
            print(f"\n❌ GET /orchestrator/status - Error: {e}")
        
        # Test orchestrator components endpoint
        try:
            response = await client.get(f"{base_url}/orchestrator/components")
            print(f"\n✅ GET /orchestrator/components - Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  Total components: {data['data']['total_components']}")
                if data['data']['components']:
                    print(f"  Sample component: {data['data']['components'][0]['name']}")
        except Exception as e:
            print(f"\n❌ GET /orchestrator/components - Error: {e}")
        
        # Test orchestrator history endpoint
        try:
            response = await client.get(f"{base_url}/orchestrator/history?limit=10")
            print(f"\n✅ GET /orchestrator/history - Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  Total invocations: {data['data']['total_invocations']}")
                print(f"  Success rate: {data['data']['success_rate']}")
        except Exception as e:
            print(f"\n❌ GET /orchestrator/history - Error: {e}")

if __name__ == "__main__":
    print("Note: This test requires the API server to be running on port 5000")
    print("You can start it with: python -m prompt_improver.cli start")
    asyncio.run(test_api_endpoints())