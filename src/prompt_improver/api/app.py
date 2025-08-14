"""FastAPI Application Factory for Real API Testing

Production-ready FastAPI application with real service integrations,
comprehensive health checks, and observability features.

Key Features:
- Real database, Redis, and ML service integrations
- API request/response monitoring and metrics
- WebSocket support for real-time analytics
- Security middleware and authentication
- Circuit breaker patterns for reliability
- OpenTelemetry instrumentation
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from prompt_improver.api import api_router
from prompt_improver.api.health import health_router
from prompt_improver.utils.error_handlers import (
    authentication_exception_handler,
    authorization_exception_handler,
    create_correlation_middleware,
    prompt_improver_exception_handler,
    rate_limit_exception_handler,
    validation_exception_handler,
)
from prompt_improver.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    PromptImproverError,
    RateLimitError,
    ValidationError,
)
from prompt_improver.core.config import get_config
from prompt_improver.core.di.container_orchestrator import get_container
from prompt_improver.core.di.clean_container import initialize_clean_container
from prompt_improver.repositories.protocols.session_manager_protocol import (
    SessionManagerProtocol,
)
from prompt_improver.monitoring.opentelemetry.integration import setup_fastapi_telemetry
from prompt_improver.monitoring.unified.facade import UnifiedMonitoringFacade
from prompt_improver.performance.monitoring.health.unified_health_system import (
    get_unified_health_monitor,
)
from prompt_improver.security.unified_security_orchestrator import (
    get_security_orchestrator,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with proper startup/shutdown sequences."""
    try:
        # Startup sequence
        logger.info("Starting FastAPI application...")

        # Initialize core services
        config = get_config()
        logger.info(
            f"Configuration loaded for environment: {config.environment.environment}"
        )

        # Initialize dependency injection container
        await initialize_clean_container()
        container = await get_container()
        logger.info("Dependency injection container initialized")

        # Initialize database connections
        db_manager = await get_database_services()
        await db_manager.initialize()
        logger.info("Database manager initialized")

        # Initialize monitoring and health systems
        monitoring_manager = UnifiedMonitoringFacade()
        await monitoring_manager.start_monitoring()
        logger.info("Unified monitoring initialized")

        health_monitor = get_unified_health_monitor()
        await health_monitor.initialize()
        logger.info("Health monitor initialized")

        # Initialize security orchestrator
        security_orchestrator = get_security_orchestrator()
        await security_orchestrator.initialize()
        logger.info("Security orchestrator initialized")

        # Setup telemetry if enabled
        if config.monitoring.enable_tracing:
            setup_fastapi_telemetry(app, "prompt-improver-api")
            logger.info("OpenTelemetry instrumentation enabled")

        logger.info("FastAPI application startup completed successfully")
        yield

    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    finally:
        # Shutdown sequence
        logger.info("Starting FastAPI application shutdown...")

        try:
            # Cleanup resources in reverse order
            if hasattr(security_orchestrator, "cleanup"):
                await security_orchestrator.cleanup()
                logger.info("Security orchestrator cleaned up")

            if hasattr(health_monitor, "cleanup"):
                await health_monitor.cleanup()
                logger.info("Health monitor cleaned up")

            await monitoring_manager.stop_monitoring()
            logger.info("Unified monitoring cleaned up")

            if hasattr(db_manager, "cleanup"):
                await db_manager.cleanup()
                logger.info("Database manager cleaned up")

            if hasattr(container, "cleanup"):
                await container.cleanup()
                logger.info("DI container cleaned up")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        logger.info("FastAPI application shutdown completed")


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled exceptions."""
    import uuid
    from datetime import UTC, datetime
    
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={"correlation_id": correlation_id}
    )

    # Return appropriate error response
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "correlation_id": correlation_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        },
        headers={"X-Correlation-ID": correlation_id}
    )


def create_app(
    enable_cors: bool = True, enable_docs: bool = True, testing: bool = False
) -> FastAPI:
    """Create and configure FastAPI application with real service integrations.

    Args:
        enable_cors: Enable CORS middleware for development
        enable_docs: Enable OpenAPI documentation endpoints
        testing: Configure for testing environment

    Returns:
        Configured FastAPI application instance
    """
    config = get_config()

    # Create FastAPI app with lifespan management
    app = FastAPI(
        title="Prompt Improver API",
        description="Production-ready API for prompt improvement with ML analytics",
        version=config.environment.version,
        lifespan=lifespan,
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
        openapi_url="/openapi.json" if enable_docs else None,
    )

    # Add correlation middleware first
    app.add_middleware(lambda request, call_next: create_correlation_middleware()(request, call_next))
    
    # Add specific exception handlers
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(AuthenticationError, authentication_exception_handler)
    app.add_exception_handler(AuthorizationError, authorization_exception_handler)
    app.add_exception_handler(RateLimitError, rate_limit_exception_handler)
    app.add_exception_handler(PromptImproverError, prompt_improver_exception_handler)
    
    # Add global exception handler for unhandled exceptions
    app.add_exception_handler(Exception, global_exception_handler)

    # Configure CORS for development
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://localhost:8080"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

    # Include all API routers
    app.include_router(health_router)  # Health checks at /health/*
    app.include_router(api_router)  # API endpoints at /api/v1/*

    @app.get("/")
    async def root():
        """Root endpoint with service information."""
        return {
            "service": "Prompt Improver API",
            "version": config.environment.version,
            "environment": config.environment.environment,
            "status": "operational",
            "documentation": "/docs" if enable_docs else "disabled",
            "health_check": "/health",
            "api_prefix": "/api/v1",
        }

    return app


def create_test_app() -> FastAPI:
    """Create FastAPI application configured for testing.

    This creates an app with real service integrations but configured
    for testing environments (test databases, mock external services, etc.)
    """
    return create_app(
        enable_cors=True,
        enable_docs=False,  # Disable docs in tests for performance
        testing=True,
    )


# Global app instance for production deployment
app = create_app()

if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(
        "prompt_improver.api.app:app",
        host="0.0.0.0",
        port=config.server.port,
        reload=config.environment.environment == "development",
        log_level="info",
        access_log=True,
    )
