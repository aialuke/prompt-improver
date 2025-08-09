"""API Module for Prompt Improver
Provides REST and WebSocket API endpoints for the application
"""
from fastapi import APIRouter
from prompt_improver.api.analytics_endpoints import analytics_router
from prompt_improver.api.apriori_endpoints import apriori_router
from prompt_improver.api.real_time_endpoints import real_time_router
api_router = APIRouter(prefix='/api/v1')
api_router.include_router(apriori_router, tags=['apriori'])
api_router.include_router(real_time_router, tags=['real-time'])
api_router.include_router(analytics_router, tags=['analytics'])
__all__ = ['analytics_router', 'api_router', 'apriori_router', 'real_time_router']
