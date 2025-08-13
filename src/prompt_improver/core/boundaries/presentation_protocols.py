"""Presentation Layer Boundary Protocols.

These protocols define the contracts between the presentation layer
and application layer, ensuring clean separation of concerns.

Clean Architecture Rule: Presentation layer depends only on application layer.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from uuid import UUID

from prompt_improver.core.domain.types import (
    SessionId,
    UserId,
    ImprovementSessionData,
    HealthStatusData,
)


@runtime_checkable
class APIEndpointProtocol(Protocol):
    """Protocol for REST API endpoints."""
    
    async def handle_request(
        self,
        request_data: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Handle incoming API request.
        
        Args:
            request_data: Request payload and parameters
            user_context: Optional user authentication context
            
        Returns:
            API response data
        """
        ...
    
    async def validate_request(
        self,
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate incoming request format and data.
        
        Args:
            request_data: Request to validate
            
        Returns:
            Validation results with any errors
        """
        ...
    
    def get_supported_methods(self) -> List[str]:
        """Get HTTP methods supported by this endpoint.
        
        Returns:
            List of supported HTTP methods
        """
        ...


@runtime_checkable
class CLICommandProtocol(Protocol):
    """Protocol for CLI command handlers."""
    
    async def execute(
        self,
        args: List[str],
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute CLI command with given arguments.
        
        Args:
            args: Command line arguments
            options: Command line options
            
        Returns:
            Command execution results
        """
        ...
    
    def get_help_text(self) -> str:
        """Get help text for this command.
        
        Returns:
            Help text describing command usage
        """
        ...
    
    def validate_args(
        self,
        args: List[str],
        options: Dict[str, Any]
    ) -> bool:
        """Validate command arguments and options.
        
        Args:
            args: Arguments to validate
            options: Options to validate
            
        Returns:
            Whether arguments are valid
        """
        ...


@runtime_checkable
class TUIWidgetProtocol(Protocol):
    """Protocol for TUI (Text User Interface) widgets."""
    
    async def render(
        self,
        context: Dict[str, Any]
    ) -> str:
        """Render widget content.
        
        Args:
            context: Rendering context and data
            
        Returns:
            Rendered widget content
        """
        ...
    
    async def handle_input(
        self,
        input_event: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle user input event.
        
        Args:
            input_event: Input event data
            
        Returns:
            Input handling results
        """
        ...
    
    async def update_state(
        self,
        new_data: Dict[str, Any]
    ) -> None:
        """Update widget state with new data.
        
        Args:
            new_data: New data to display
        """
        ...


@runtime_checkable
class WebSocketProtocol(Protocol):
    """Protocol for WebSocket connection handlers."""
    
    async def on_connect(
        self,
        connection_data: Dict[str, Any]
    ) -> bool:
        """Handle new WebSocket connection.
        
        Args:
            connection_data: Connection metadata
            
        Returns:
            Whether to accept the connection
        """
        ...
    
    async def on_message(
        self,
        message: Dict[str, Any],
        connection_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Handle incoming WebSocket message.
        
        Args:
            message: Message data
            connection_id: Connection identifier
            
        Returns:
            Optional response message
        """
        ...
    
    async def on_disconnect(
        self,
        connection_id: str,
        reason: str,
    ) -> None:
        """Handle WebSocket disconnection.
        
        Args:
            connection_id: Connection identifier
            reason: Disconnection reason
        """
        ...
    
    async def broadcast(
        self,
        message: Dict[str, Any],
        target_connections: Optional[List[str]] = None,
    ) -> None:
        """Broadcast message to connections.
        
        Args:
            message: Message to broadcast
            target_connections: Specific connections or None for all
        """
        ...


@runtime_checkable 
class ResponseFormatterProtocol(Protocol):
    """Protocol for formatting responses to presentation formats."""
    
    def format_success_response(
        self,
        data: Dict[str, Any],
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format successful response.
        
        Args:
            data: Response data
            message: Optional success message
            
        Returns:
            Formatted response
        """
        ...
    
    def format_error_response(
        self,
        error: Exception,
        error_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format error response.
        
        Args:
            error: Exception that occurred
            error_code: Optional error code
            
        Returns:
            Formatted error response
        """
        ...
    
    def format_health_response(
        self,
        health_data: HealthStatusData
    ) -> Dict[str, Any]:
        """Format health check response.
        
        Args:
            health_data: Health status data
            
        Returns:
            Formatted health response
        """
        ...


@runtime_checkable
class AuthenticationProtocol(Protocol):
    """Protocol for presentation layer authentication."""
    
    async def authenticate_request(
        self,
        credentials: Dict[str, Any]
    ) -> Optional[UserId]:
        """Authenticate a request using provided credentials.
        
        Args:
            credentials: Authentication credentials
            
        Returns:
            User ID if authenticated, None otherwise
        """
        ...
    
    async def authorize_action(
        self,
        user_id: UserId,
        action: str,
        resource: Optional[str] = None,
    ) -> bool:
        """Check if user is authorized for an action.
        
        Args:
            user_id: User identifier
            action: Action to authorize
            resource: Optional resource identifier
            
        Returns:
            Whether action is authorized
        """
        ...
    
    async def create_session(
        self,
        user_id: UserId
    ) -> SessionId:
        """Create a new user session.
        
        Args:
            user_id: User identifier
            
        Returns:
            New session identifier
        """
        ...