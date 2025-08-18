"""MCP Server module for APES - Modernized Architecture."""

def get_server_class():
    """Get the APESMCPServer class with dynamic import."""
    from prompt_improver.mcp_server.server import APESMCPServer
    return APESMCPServer

def get_main_function():
    """Get the main function with dynamic import."""
    from prompt_improver.mcp_server.lifecycle import main
    return main

# For backward compatibility
def __getattr__(name):
    if name == "APESMCPServer":
        return get_server_class()
    elif name == "main":
        return get_main_function()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["APESMCPServer", "main", "get_server_class", "get_main_function"]
