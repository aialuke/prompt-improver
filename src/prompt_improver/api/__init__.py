"""API Module for Prompt Improver
Provides REST and WebSocket API endpoints for the application
"""

from fastapi import APIRouter

from .apriori_endpoints import apriori_router
from .real_time_endpoints import real_time_router

# Main API router
api_router = APIRouter(prefix="/api/v1")

# Include sub-routers
api_router.include_router(apriori_router, tags=["apriori"])
api_router.include_router(real_time_router, tags=["real-time"])

__all__ = ["api_router", "apriori_router", "real_time_router"]
