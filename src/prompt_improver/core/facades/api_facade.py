"""API Services Facade - Reduces FastAPI App Module Coupling

This facade provides unified API service coordination while reducing direct
imports from 12 to 4 internal dependencies through lazy initialization.

Design:
- Protocol-based interface for loose coupling
- Lazy loading of API service integrations
- Service lifecycle coordination  
- Health check and monitoring integration
- Zero circular import dependencies
"""

import logging
from typing import Any, Dict, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class APIFacadeProtocol(Protocol):
    """Protocol for API services facade."""
    
    def get_config(self) -> Any:
        """Get application configuration."""
        ...
    
    async def get_container(self) -> Any:
        """Get dependency injection container."""
        ...
    
    async def get_database_services(self) -> Any:
        """Get database services manager."""
        ...
    
    async def get_monitoring_facade(self) -> Any:
        """Get unified monitoring facade."""
        ...
    
    async def get_security_orchestrator(self) -> Any:
        """Get security orchestrator."""
        ...
    
    def setup_telemetry(self, app: Any, service_name: str) -> None:
        """Setup OpenTelemetry instrumentation."""
        ...
    
    async def initialize_all_services(self) -> None:
        """Initialize all API services."""
        ...
    
    async def shutdown_all_services(self) -> None:
        """Shutdown all API services."""
        ...


class APIFacade(APIFacadeProtocol):
    """API services facade with minimal coupling.
    
    Reduces FastAPI app module coupling from 12 internal imports to 4.
    Provides unified interface for all API service coordination.
    """

    def __init__(self):
        """Initialize facade with lazy loading."""
        self._config = None
        self._container = None
        self._database_services = None
        self._monitoring_facade = None
        self._security_orchestrator = None
        self._health_monitor = None
        self._services_initialized = False
        logger.debug("APIFacade initialized with lazy loading")

    def _ensure_config(self):
        """Ensure configuration is loaded."""
        if self._config is None:
            # Only import when needed to reduce coupling
            from prompt_improver.core.config import get_config
            self._config = get_config()

    async def _ensure_container(self):
        """Ensure DI container is available."""
        if self._container is None:
            from prompt_improver.core.di.container_orchestrator import get_container
            from prompt_improver.core.di.clean_container import initialize_clean_container
            await initialize_clean_container()
            self._container = await get_container()

    async def _ensure_database_services(self):
        """Ensure database services are available."""
        if self._database_services is None:
            from prompt_improver.database import get_database_services
            self._database_services = await get_database_services()

    async def _ensure_monitoring_facade(self):
        """Ensure monitoring facade is available."""
        if self._monitoring_facade is None:
            from prompt_improver.monitoring.unified.facade import UnifiedMonitoringFacade
            self._monitoring_facade = UnifiedMonitoringFacade()

    async def _ensure_security_orchestrator(self):
        """Ensure security orchestrator is available."""
        if self._security_orchestrator is None:
            from prompt_improver.security.unified_security_orchestrator import get_security_orchestrator
            self._security_orchestrator = get_security_orchestrator()

    def get_config(self) -> Any:
        """Get application configuration."""
        self._ensure_config()
        return self._config

    async def get_container(self) -> Any:
        """Get dependency injection container."""
        await self._ensure_container()
        return self._container

    async def get_database_services(self) -> Any:
        """Get database services manager."""
        await self._ensure_database_services()
        return self._database_services

    async def get_monitoring_facade(self) -> Any:
        """Get unified monitoring facade."""
        await self._ensure_monitoring_facade()
        return self._monitoring_facade

    async def get_security_orchestrator(self) -> Any:
        """Get security orchestrator."""
        await self._ensure_security_orchestrator()
        return self._security_orchestrator

    async def get_health_monitor(self) -> Any:
        """Get unified health monitor."""
        if self._health_monitor is None:
            from prompt_improver.performance.monitoring.health.unified_health_system import get_unified_health_monitor
            self._health_monitor = get_unified_health_monitor()
        return self._health_monitor

    def setup_telemetry(self, app: Any, service_name: str) -> None:
        """Setup OpenTelemetry instrumentation."""
        config = self.get_config()
        if config.monitoring.enable_tracing:
            from prompt_improver.monitoring.opentelemetry.integration import setup_fastapi_telemetry
            setup_fastapi_telemetry(app, service_name)
            logger.info(f"OpenTelemetry instrumentation enabled for {service_name}")

    def setup_error_handlers(self, app: Any) -> None:
        """Setup error handlers for the FastAPI app."""
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
        
        # Add correlation middleware
        app.add_middleware(lambda request, call_next: create_correlation_middleware()(request, call_next))
        
        # Add exception handlers
        app.add_exception_handler(ValidationError, validation_exception_handler)
        app.add_exception_handler(AuthenticationError, authentication_exception_handler)
        app.add_exception_handler(AuthorizationError, authorization_exception_handler)
        app.add_exception_handler(RateLimitError, rate_limit_exception_handler)
        app.add_exception_handler(PromptImproverError, prompt_improver_exception_handler)

    def setup_cors(self, app: Any, enable_cors: bool = True) -> None:
        """Setup CORS middleware."""
        if enable_cors:
            from fastapi.middleware.cors import CORSMiddleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["http://localhost:3000", "http://localhost:8080"],
                allow_credentials=True,
                allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                allow_headers=["*"],
            )

    def setup_routers(self, app: Any) -> None:
        """Setup API routers.""" 
        from prompt_improver.api import api_router
        from prompt_improver.api.health import health_router
        
        app.include_router(health_router)  # Health checks at /health/*
        app.include_router(api_router)  # API endpoints at /api/v1/*

    async def initialize_all_services(self) -> None:
        """Initialize all API services."""
        if self._services_initialized:
            logger.warning("API services already initialized")
            return

        logger.info("Initializing API services...")

        # Initialize core services
        await self._ensure_container()
        await self._ensure_database_services()
        await self._ensure_monitoring_facade()
        await self._ensure_security_orchestrator()

        # Initialize database connections
        await self._database_services.initialize()
        logger.info("✓ Database services initialized")

        # Initialize monitoring and health systems
        await self._monitoring_facade.start_monitoring()
        logger.info("✓ Unified monitoring initialized")

        health_monitor = await self.get_health_monitor()
        await health_monitor.initialize()
        logger.info("✓ Health monitor initialized")

        # Initialize security orchestrator
        await self._security_orchestrator.initialize()
        logger.info("✓ Security orchestrator initialized")

        self._services_initialized = True
        logger.info("API services initialization complete")

    async def shutdown_all_services(self) -> None:
        """Shutdown all API services."""
        if not self._services_initialized:
            return

        logger.info("Shutting down API services...")

        try:
            # Cleanup resources in reverse order
            if self._security_orchestrator and hasattr(self._security_orchestrator, "cleanup"):
                await self._security_orchestrator.cleanup()
                logger.info("✓ Security orchestrator cleaned up")

            if self._health_monitor and hasattr(self._health_monitor, "cleanup"):
                await self._health_monitor.cleanup()
                logger.info("✓ Health monitor cleaned up")

            if self._monitoring_facade:
                await self._monitoring_facade.stop_monitoring()
                logger.info("✓ Unified monitoring cleaned up")

            if self._database_services and hasattr(self._database_services, "cleanup"):
                await self._database_services.cleanup()
                logger.info("✓ Database services cleaned up")

            if self._container and hasattr(self._container, "cleanup"):
                await self._container.cleanup()
                logger.info("✓ DI container cleaned up")

        except Exception as e:
            logger.error(f"Error during API services shutdown: {e}")

        # Clear references
        self._container = None
        self._database_services = None
        self._monitoring_facade = None
        self._security_orchestrator = None
        self._health_monitor = None
        self._services_initialized = False

        logger.info("API services shutdown complete")

    def get_service_status(self) -> dict[str, Any]:
        """Get status of all API services."""
        return {
            "initialized": self._services_initialized,
            "config": self._config is not None,
            "container": self._container is not None,
            "database_services": self._database_services is not None,
            "monitoring_facade": self._monitoring_facade is not None,
            "security_orchestrator": self._security_orchestrator is not None,
            "health_monitor": self._health_monitor is not None,
        }

    async def health_check_all_services(self) -> dict[str, Any]:
        """Perform health check on all API services."""
        results = {
            "api_facade_status": "healthy",
            "services": {},
        }
        
        try:
            if self._container:
                container_health = await self._container.health_check_all()
                results["services"]["container"] = container_health

            if self._database_services and hasattr(self._database_services, "health_check"):
                db_health = await self._database_services.health_check()
                results["services"]["database"] = db_health

            if self._monitoring_facade and hasattr(self._monitoring_facade, "health_check"):
                monitoring_health = await self._monitoring_facade.health_check()
                results["services"]["monitoring"] = monitoring_health

            if self._security_orchestrator and hasattr(self._security_orchestrator, "health_check"):
                security_health = await self._security_orchestrator.health_check()
                results["services"]["security"] = security_health

        except Exception as e:
            results["api_facade_status"] = "unhealthy"
            results["error"] = str(e)

        return results


# Global facade instance
_api_facade: APIFacade | None = None


def get_api_facade() -> APIFacade:
    """Get global API facade instance.
    
    Returns:
        APIFacade with lazy initialization and minimal coupling
    """
    global _api_facade
    if _api_facade is None:
        _api_facade = APIFacade()
    return _api_facade


async def initialize_api_facade() -> None:
    """Initialize the global API facade."""
    facade = get_api_facade()
    await facade.initialize_all_services()


async def shutdown_api_facade() -> None:
    """Shutdown the global API facade."""
    global _api_facade
    if _api_facade:
        await _api_facade.shutdown_all_services()
        _api_facade = None


__all__ = [
    "APIFacadeProtocol",
    "APIFacade",
    "get_api_facade",
    "initialize_api_facade",
    "shutdown_api_facade",
]