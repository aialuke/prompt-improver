"""
MCP Service Facade

Provides a clean interface to all MCP server dependencies using dependency inversion.
This facade reduces coupling and makes the MCP server more testable and maintainable.
"""

from typing import Protocol, Any, Dict, Optional
import logging

# Import protocols for dependency inversion
from prompt_improver.core.protocols import (
    DatabaseProtocol,
    RedisCacheProtocol,
    DateTimeUtilsProtocol,
    HealthServiceProtocol,
    PerformanceMonitorProtocol
)

class ConfigServiceProtocol(Protocol):
    """Protocol for configuration service"""

    def get_mcp_config(self) -> Dict[str, Any]:
        """Get MCP-specific configuration"""
        ...

    @property
    def mcp_batch_size(self) -> int:
        """Get MCP batch size"""
        ...

    @property
    def mcp_session_maxsize(self) -> int:
        """Get MCP session max size"""
        ...

    @property
    def mcp_session_ttl(self) -> int:
        """Get MCP session TTL"""
        ...

class SecurityServiceProtocol(Protocol):
    """Protocol for security services"""

    async def validate_input(self, input_data: str) -> bool:
        """Validate input using OWASP standards"""
        ...

    async def validate_output(self, output_data: str) -> bool:
        """Validate output for safety"""
        ...

    async def check_rate_limit(self, client_id: str) -> bool:
        """Check rate limiting"""
        ...

    async def authenticate_request(self, auth_data: Dict[str, Any]) -> bool:
        """Authenticate MCP request"""
        ...

# MLServiceProtocol removed per architectural separation requirements
# MCP server is read-only for rule application only

# FeedbackServiceProtocol removed per architectural separation requirements
# Feedback collection should be handled by separate ML training system

class MCPServiceFacade:
    """
    Service facade that abstracts all MCP server dependencies.

    This facade implements the Facade pattern to provide a simplified
    interface to the complex subsystem of services required by the MCP server.
    """

    def __init__(self,
                 config_service: ConfigServiceProtocol,
                 database_service: DatabaseProtocol,
                 cache_service: RedisCacheProtocol,
                 datetime_service: DateTimeUtilsProtocol,
                 health_service: HealthServiceProtocol,
                 performance_service: PerformanceMonitorProtocol,
                 security_service: SecurityServiceProtocol):

        self.config = config_service
        self.database = database_service
        self.cache = cache_service
        self.datetime = datetime_service
        self.health = health_service
        self.performance = performance_service
        self.security = security_service
        # ML and feedback services removed per architectural separation

        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> bool:
        """Initialize all services"""
        try:
            # Perform health checks on all services
            health_results = await self.health.run_all_checks()

            # Log initialization status
            failed_checks = [name for name, result in health_results.items()
                           if result.status.value != 'healthy']

            if failed_checks:
                self.logger.warning(f"Some services failed health check: {failed_checks}")
                return False

            self.logger.info("MCP Service Facade initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize MCP services: {e}")
            return False

    async def shutdown(self) -> bool:
        """Gracefully shutdown all services"""
        try:
            # Stop background tasks, close connections, etc.
            # Implementation depends on specific service shutdown procedures
            self.logger.info("MCP Service Facade shutdown completed")
            return True
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return False

    async def process_prompt_request(self,
                                   prompt: str,
                                   context: Optional[Dict[str, Any]] = None,
                                   client_id: str = "anonymous") -> Dict[str, Any]:
        """
        Process a prompt improvement request through the service facade.

        Read-only rule application mode:
        1. Validate security
        2. Check rate limits
        3. Apply pre-computed rules (no ML training)
        4. Return optimized response
        """
        start_time = self.datetime.aware_utc_now()

        try:
            # Security validation
            if not await self.security.validate_input(prompt):
                return {"error": "Input validation failed", "status": "rejected"}

            # Rate limiting check
            if not await self.security.check_rate_limit(client_id):
                return {"error": "Rate limit exceeded", "status": "rate_limited"}

            # Record performance metrics
            await self.performance.record_counter("mcp.request.start", tags={"client": client_id})

            # Apply pre-computed rules (read-only rule application)
            # This should use the existing prompt improvement service for rule application
            improved_prompt = prompt  # Placeholder - implement rule application

            # Validate output
            if not await self.security.validate_output(improved_prompt):
                return {"error": "Output validation failed", "status": "rejected"}

            # Calculate processing time
            end_time = self.datetime.aware_utc_now()
            processing_time = (end_time - start_time).total_seconds() * 1000

            # Record performance metrics
            await self.performance.record_timer("mcp.request.duration", processing_time)
            await self.performance.record_counter("mcp.request.success")

            # ML training and feedback collection removed per architectural separation

            return {
                "original_prompt": prompt,
                "improved_prompt": improved_prompt,
                "processing_time_ms": processing_time,
                "status": "success",
                "mode": "read_only_rule_application",
                "timestamp": self.datetime.format_iso(end_time)
            }

        except Exception as e:
            await self.performance.record_counter("mcp.request.error")
            self.logger.error(f"Error processing prompt request: {e}")
            return {"error": str(e), "status": "error"}

    async def get_server_status(self) -> Dict[str, Any]:
        """Get comprehensive server status (read-only rule application mode)"""
        try:
            health_status = await self.health.get_overall_health()
            performance_metrics = await self.performance.get_metrics_summary()

            return {
                "status": health_status.status.value,
                "timestamp": self.datetime.format_iso(self.datetime.aware_utc_now()),
                "mode": "read_only_rule_application",
                "health": {
                    "overall": health_status.status.value,
                    "message": health_status.message,
                    "details": health_status.details
                },
                "performance": performance_metrics,
                "ml_analytics": "Moved to separate ML training system",
                "feedback": "Moved to separate ML training system",
                "config": {
                    "batch_size": self.config.mcp_batch_size,
                    "session_maxsize": self.config.mcp_session_maxsize,
                    "session_ttl": self.config.mcp_session_ttl
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting server status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "mode": "read_only_rule_application",
                "timestamp": self.datetime.format_iso(self.datetime.aware_utc_now())
            }

    # Batch processing removed per architectural separation requirements
    # Batch processing should be handled by separate ML training system

# Factory function for creating the facade with real implementations
def create_mcp_service_facade() -> MCPServiceFacade:
    """Create MCP service facade with real service implementations (read-only mode)"""
    from .concrete_services import create_concrete_services

    services = create_concrete_services()

    return MCPServiceFacade(
        config_service=services['config_service'],
        database_service=services['database_service'],
        cache_service=services['cache_service'],
        datetime_service=services['datetime_service'],
        health_service=services['health_service'],
        performance_service=services['performance_service'],
        security_service=services['security_service']
        # ML and feedback services removed per architectural separation
    )
