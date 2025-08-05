#!/usr/bin/env python3
"""
Simple test health server for production readiness validation testing
"""

import asyncio
import json
from datetime import datetime
from aiohttp import web
import aiohttp_cors


async def health_handler(request):
    """Health check endpoint"""
    return web.json_response({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2025.1.0",
        "services": {
            "database": "connected",
            "cache": "connected",
            "ml_pipeline": "ready"
        }
    })


async def ready_handler(request):
    """Readiness check endpoint"""
    return web.json_response({
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "database_migration": "complete",
            "model_loading": "complete",
            "cache_warming": "complete"
        }
    })


async def live_handler(request):
    """Liveness check endpoint"""
    return web.json_response({
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": 3600
    })


async def status_handler(request):
    """API status endpoint"""
    return web.json_response({
        "api_version": "v1",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": [
            "/health",
            "/health/ready",
            "/health/live",
            "/api/v1/status"
        ]
    })


async def metrics_handler(request):
    """Metrics endpoint (Prometheus format simulation)"""
    metrics = """# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",endpoint="/health"} 1234
http_requests_total{method="GET",endpoint="/api/v1/status"} 567

# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.1"} 800
http_request_duration_seconds_bucket{le="0.2"} 900
http_request_duration_seconds_bucket{le="0.5"} 950
http_request_duration_seconds_bucket{le="+Inf"} 1000

# HELP process_resident_memory_bytes Resident memory size
# TYPE process_resident_memory_bytes gauge
process_resident_memory_bytes 41943040
"""
    return web.Response(text=metrics, content_type='text/plain')


async def create_app():
    """Create the test application"""
    app = web.Application()
    
    # Configure CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    # Add routes
    app.router.add_get('/health', health_handler)
    app.router.add_get('/health/ready', ready_handler)
    app.router.add_get('/health/live', live_handler)
    app.router.add_get('/api/v1/health', health_handler)
    app.router.add_get('/api/v1/status', status_handler)
    app.router.add_get('/api/v1/metrics', metrics_handler)
    app.router.add_get('/metrics', metrics_handler)
    
    # Add CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)
    
    return app


async def main():
    """Main function to run the test server"""
    app = await create_app()
    
    # Start the server
    runner = web.AppRunner(app)
    await runner.setup()
    
    # Use a different port to avoid conflicts
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()
    
    print("🚀 Test health server started on http://localhost:8080")
    print("📊 Available endpoints:")
    print("   - GET /health")
    print("   - GET /health/ready")
    print("   - GET /health/live")
    print("   - GET /api/v1/health")
    print("   - GET /api/v1/status")
    print("   - GET /api/v1/metrics")
    print("   - GET /metrics")
    print("\n⏹️  Press Ctrl+C to stop")
    
    try:
        # Keep the server running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down test server...")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
