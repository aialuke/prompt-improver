"""MCP server protocol definitions.

Consolidated protocols for all Model Context Protocol server functionality
including tool management, resource handling, server lifecycle, and operational excellence.

This module serves as the authoritative source for all MCP-related protocols,
eliminating circular dependencies and providing comprehensive server functionality
following SRE best practices.
"""

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ServerConfigProtocol(Protocol):
    """Protocol for MCP server configuration."""

    @property
    def security_config(self) -> dict[str, Any]:
        """Get security configuration."""
        ...

    @property
    def transport_config(self) -> dict[str, Any]:
        """Get transport configuration."""
        ...


@runtime_checkable
class ServerServicesProtocol(Protocol):
    """Protocol for MCP server services container.

    Replaces direct imports of ServerServices class to eliminate
    circular dependencies between MCP components.
    """

    config: Any
    database: Any
    cache: Any
    security: Any
    health_monitor: Any
    analytics: Any

    # MCP-specific service access for tools
    security_facade: Any
    security_manager: Any
    validation_manager: Any
    authentication_manager: Any
    authorization_manager: Any
    security_stack: Any
    input_validator: Any
    output_validator: Any
    performance_optimizer: Any
    performance_monitor: Any
    sla_monitor: Any
    prompt_service: Any
    session_store: Any
    event_loop_manager: Any
    security_middleware_adapter: Any

    @property
    def is_initialized(self) -> bool:
        """Check if services are fully initialized."""
        ...

    @abstractmethod
    async def get_database_session(self):
        """Get database session context manager for MCP operations.

        Replaces direct calls to get_database_services(ManagerMode.MCP_SERVER)
        to eliminate circular dependencies.

        Returns:
            Database session context manager for async operations
        """
        ...


@runtime_checkable
class MCPServerProtocol(Protocol):
    """Protocol for MCP server interface.

    Comprehensive server protocol that combines lifecycle management,
    operational excellence, and MCP specification compliance.
    """

    config: Any
    services: ServerServicesProtocol

    # Core server lifecycle methods
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the MCP server."""
        ...

    @abstractmethod
    async def start(self) -> None:
        """Start the MCP server."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the MCP server."""
        ...

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Perform server health check."""
        ...

    # MCP specification compliance methods
    @abstractmethod
    async def start_server(self) -> None:
        """Start the MCP server (MCP spec compliance)."""
        ...

    @abstractmethod
    async def stop_server(self) -> None:
        """Stop the MCP server gracefully (MCP spec compliance)."""
        ...

    @abstractmethod
    def get_server_info(self) -> dict[str, Any]:
        """Get server information and status (MCP spec compliance)."""
        ...


@runtime_checkable
class ServiceFactoryProtocol(Protocol):
    """Protocol for creating MCP server services.

    Eliminates circular dependencies by providing factory interface
    for service creation without direct imports.
    """

    @abstractmethod
    async def create_services(self, config: Any, **service_dependencies) -> ServerServicesProtocol:
        """Create server services container with dependencies.

        Args:
            config: Application configuration
            **service_dependencies: Additional service dependencies

        Returns:
            ServerServicesProtocol: Configured services container
        """
        ...


@runtime_checkable
class ServerFactoryProtocol(Protocol):
    """Protocol for creating and initializing MCP server instances.

    Provides factory interface for server creation following SRE best practices.
    """

    @abstractmethod
    def create_server(self) -> MCPServerProtocol:
        """Create a new MCP server instance.

        Returns:
            MCPServerProtocol: Uninitialized server instance
        """
        ...

    @abstractmethod
    async def initialize_server(self, server: MCPServerProtocol) -> bool:
        """Initialize server instance with async components.

        Args:
            server: Server instance to initialize

        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        ...


@runtime_checkable
class LifecycleManagerProtocol(Protocol):
    """Protocol for MCP server lifecycle management.

    Implements SRE best practices for reliable service lifecycle:
    - Graceful startup sequencing
    - Signal handling for SIGTERM/SIGINT
    - Health check integration
    - Resource cleanup and connection draining
    - Process monitoring hooks
    """

    @abstractmethod
    def setup_lifecycle_handlers(self, server: MCPServerProtocol) -> None:
        """Setup lifecycle handlers for server instance.

        Configures:
        - Signal handlers for graceful shutdown
        - Initialization and shutdown procedures
        - Health monitoring integration

        Args:
            server: Server instance to configure lifecycle for
        """
        ...

    @abstractmethod
    async def initialize_server_instance(self, server: MCPServerProtocol) -> bool:
        """Initialize server instance and all services.

        Implements proper startup sequencing following SRE practices:
        - Service dependency resolution
        - Health check validation
        - Resource allocation and validation
        - Performance baseline establishment

        Args:
            server: Server instance to initialize

        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        ...

    @abstractmethod
    async def shutdown_server_instance(self, server: MCPServerProtocol) -> None:
        """Gracefully shutdown server instance.

        Implements proper shutdown procedures:
        - Connection draining
        - Resource cleanup
        - Service termination sequencing
        - Final health status reporting

        Args:
            server: Server instance to shutdown
        """
        ...

    @abstractmethod
    def setup_signal_handlers(self, server: MCPServerProtocol) -> None:
        """Setup signal handlers for graceful shutdown.

        Configures handlers for:
        - SIGTERM: Graceful shutdown
        - SIGINT: Interrupt handling
        - Custom signals for operational control

        Args:
            server: Server instance to setup signal handling for
        """
        ...


@runtime_checkable
class RuntimeManagerProtocol(Protocol):
    """Protocol for MCP server runtime operations.

    Handles runtime aspects following SRE operational excellence:
    - Process monitoring and health reporting
    - Performance metrics collection
    - Incident response coordination
    - Recovery and restart procedures
    """

    @abstractmethod
    async def monitor_server_health(self, server: MCPServerProtocol) -> dict[str, Any]:
        """Monitor server health and performance metrics.

        Args:
            server: Server instance to monitor

        Returns:
            Dict[str, Any]: Health status and performance metrics
        """
        ...

    @abstractmethod
    async def handle_server_incident(self, server: MCPServerProtocol, incident_type: str, details: dict[str, Any]) -> None:
        """Handle server incidents following SRE procedures.

        Args:
            server: Server instance experiencing incident
            incident_type: Type of incident (performance, availability, security)
            details: Incident details and context
        """
        ...

    @abstractmethod
    def get_server_runtime_status(self, server: MCPServerProtocol) -> dict[str, Any]:
        """Get current runtime status and metrics.

        Args:
            server: Server instance to check

        Returns:
            Dict[str, Any]: Runtime status including uptime, connections, performance
        """
        ...


@runtime_checkable
class MCPToolProtocol(Protocol):
    """Protocol for MCP tool management.

    Provides tool execution and management capabilities following
    MCP specification requirements.
    """

    @abstractmethod
    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool with provided arguments."""
        ...

    @abstractmethod
    def list_available_tools(self) -> list[str]:
        """List all available tools."""
        ...

    @abstractmethod
    def get_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Get schema for a specific tool."""
        ...


@runtime_checkable
class MCPResourceProtocol(Protocol):
    """Protocol for MCP resource management.

    Provides resource discovery, access, and management capabilities
    following MCP specification requirements.
    """

    @abstractmethod
    async def get_resource(self, resource_uri: str) -> Any:
        """Get a resource by URI."""
        ...

    @abstractmethod
    def list_resources(self) -> list[str]:
        """List all available resources."""
        ...

    @abstractmethod
    async def update_resource(self, resource_uri: str, data: Any) -> None:
        """Update a resource with new data."""
        ...


@runtime_checkable
class MCPSessionProtocol(Protocol):
    """Protocol for MCP session management.

    Provides session lifecycle management capabilities for
    MCP client-server communication sessions.
    """

    @abstractmethod
    async def create_session(self, session_config: dict[str, Any]) -> str:
        """Create a new MCP session."""
        ...

    @abstractmethod
    async def end_session(self, session_id: str) -> None:
        """End an MCP session."""
        ...

    @abstractmethod
    def get_session_info(self, session_id: str) -> dict[str, Any]:
        """Get information about a session."""
        ...


# Export all MCP protocols for consolidated access
__all__ = [
    "LifecycleManagerProtocol",
    "MCPResourceProtocol",
    "MCPServerProtocol",
    "MCPSessionProtocol",
    "MCPToolProtocol",
    "RuntimeManagerProtocol",
    "ServerConfigProtocol",
    "ServerFactoryProtocol",
    "ServerServicesProtocol",
    "ServiceFactoryProtocol",
]
