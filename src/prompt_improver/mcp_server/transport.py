"""Transport layer management for MCP server.

Handles transport selection, fallback logic, and HTTP transport configuration.
Separated from main server for better organization and testability.
"""

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_improver.mcp_server.server import APESMCPServer

logger = logging.getLogger(__name__)


def setup_transport_handlers(server: "APESMCPServer") -> None:
    """Add transport-related methods to the server instance.

    Args:
        server: The APESMCPServer instance to configure
    """
    # Bind transport methods to server instance
    server.run_streamable_http = lambda host=None, port=None: run_streamable_http(
        server, host, port
    )


def run_streamable_http(
    server: "APESMCPServer", host: str = None, port: int = None
) -> None:
    """Run server with Streamable HTTP transport (2025-03-26 spec).

    This enables HTTP-based communication instead of stdio,
    supporting both SSE and regular HTTP responses for better
    client compatibility and production deployments.

    Args:
        server: The APESMCPServer instance to run
        host: Host to bind to (env: MCP_SERVER_HOST, required)
        port: Port to bind to (env: MCP_SERVER_PORT, default: 8080)

    Raises:
        ValueError: If MCP_SERVER_HOST environment variable is not set and host is None
    """
    host = host or os.getenv("MCP_SERVER_HOST")
    if not host:
        raise ValueError("MCP_SERVER_HOST environment variable is required")

    port = port or int(os.getenv("MCP_SERVER_PORT", "8080"))

    logger.info(
        f"Starting APES MCP Server with Streamable HTTP transport on {host}:{port}"
    )

    try:
        server.mcp.run(
            transport="streamable-http", host=host, port=port, log_level="INFO"
        )
    except TypeError as e:
        logger.warning(
            f"Streamable HTTP transport parameters not supported in current MCP SDK: {e}"
        )
        logger.info("Falling back to standard stdio transport")
        server.mcp.run()
    except Exception as e:
        logger.error(f"Failed to start with HTTP transport: {e}")
        raise


def get_transport_config() -> dict[str, str | int | None]:
    """Get transport configuration from environment variables.

    Returns:
        Dictionary containing transport configuration
    """
    return {
        "host": os.getenv("MCP_SERVER_HOST"),
        "port": int(os.getenv("MCP_SERVER_PORT", "8080")),
        "transport_type": os.getenv("MCP_TRANSPORT_TYPE", "stdio"),
        "log_level": os.getenv("MCP_LOG_LEVEL", "INFO"),
    }


def validate_http_config(host: str = None, port: int = None) -> tuple[bool, str]:
    """Validate HTTP transport configuration.

    Args:
        host: Host to validate
        port: Port to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not host and not os.getenv("MCP_SERVER_HOST"):
        return (
            False,
            "MCP_SERVER_HOST environment variable is required for HTTP transport",
        )

    actual_port = port or int(os.getenv("MCP_SERVER_PORT", "8080"))
    if not (1 <= actual_port <= 65535):
        return False, f"Invalid port number: {actual_port}. Must be between 1-65535"

    return True, ""


def select_transport_mode(args) -> str:
    """Select appropriate transport mode based on arguments and environment.

    Args:
        args: Parsed command line arguments

    Returns:
        Transport mode string ('http' or 'stdio')
    """
    if args.http:
        return "http"

    if os.getenv("MCP_FORCE_HTTP", "").lower() in ("1", "true", "yes"):
        return "http"

    return "stdio"
