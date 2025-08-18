"""API Layer Factories - Factory Functions for API Components.

This module contains factory functions for API layer components,
following Clean Architecture principles by keeping API-specific
logic within the presentation layer.
"""

from .analytics_api_factory import get_analytics_router

__all__ = [
    "get_analytics_router",
]