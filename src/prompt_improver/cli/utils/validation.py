"""
Validation utilities for CLI operations.
Extracted from existing patterns in the APES codebase.
"""

import re
from pathlib import Path
from typing import Union


def validate_path(path: Union[str, Path]) -> Path:
    """
    Validate and sanitize file path to prevent directory traversal attacks.
    
    Extracted from existing file operation patterns in the codebase.
    
    Args:
        path: File path to validate
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid or contains security risks
    """
    if not path:
        raise ValueError("Path cannot be empty")
    
    # Convert to string for processing
    path_str = str(path)
    
    # Remove directory traversal patterns
    sanitized = re.sub(r'\.\.\/|\.\.\\', '', path_str)
    
    # Remove null bytes
    sanitized = sanitized.replace('\x00', '')
    
    # Remove leading/trailing whitespace
    sanitized = sanitized.strip()
    
    # Validate path doesn't contain dangerous patterns
    if '..' in sanitized or sanitized != path_str:
        raise ValueError(f"Path contains invalid characters or traversal patterns: {path}")
    
    # Convert to Path object and validate it exists or parent exists
    validated_path = Path(sanitized)
    
    # For file paths, ensure parent directory exists or is creatable
    if not validated_path.exists() and not validated_path.parent.exists():
        try:
            validated_path.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot create parent directory for path {validated_path}: {e}")
    
    return validated_path


def validate_port(port: int) -> int:
    """
    Validate port number for network services.
    
    Extracted from existing port validation in CLI start commands.
    
    Args:
        port: Port number to validate
        
    Returns:
        Validated port number
        
    Raises:
        ValueError: If port is invalid
    """
    if not isinstance(port, int):
        try:
            port = int(port)
        except (ValueError, TypeError):
            raise ValueError(f"Port must be an integer, got: {type(port).__name__}")
    
    # Validate port range (1024-65535 for non-privileged ports)
    if port < 1024:
        raise ValueError(f"Port {port} is reserved (must be >= 1024)")
    
    if port > 65535:
        raise ValueError(f"Port {port} is invalid (must be <= 65535)")
    
    # Common ports that might conflict
    reserved_ports = {
        3306: "MySQL",
        5432: "PostgreSQL", 
        6379: "Redis",
        8080: "HTTP Proxy",
        9200: "Elasticsearch"
    }
    
    if port in reserved_ports:
        # Warning but not error - user might intentionally use these
        pass
    
    return port


def validate_timeout(timeout: int) -> int:
    """
    Validate timeout value for operations.
    
    Extracted from existing timeout validation patterns in service management.
    
    Args:
        timeout: Timeout in seconds
        
    Returns:
        Validated timeout value
        
    Raises:
        ValueError: If timeout is invalid
    """
    if not isinstance(timeout, int):
        try:
            timeout = int(timeout)
        except (ValueError, TypeError):
            raise ValueError(f"Timeout must be an integer, got: {type(timeout).__name__}")
    
    # Validate timeout range
    if timeout < 0:
        raise ValueError(f"Timeout cannot be negative: {timeout}")
    
    if timeout == 0:
        raise ValueError("Timeout cannot be zero (use None for no timeout)")
    
    # Reasonable upper limit (24 hours)
    max_timeout = 24 * 60 * 60  # 86400 seconds
    if timeout > max_timeout:
        raise ValueError(f"Timeout {timeout}s exceeds maximum allowed {max_timeout}s (24 hours)")
    
    # Warn about very short timeouts
    if timeout < 5:
        # Very short timeout - might be intentional for testing
        pass
    
    return timeout
