"""Security service protocol definitions.

Consolidated protocols for all security-related services including
authentication, authorization, encryption, and security monitoring.
"""

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

# Protocols will be migrated here from:
# - /core/protocols/security_service/security_protocols.py
# - Any security-related protocols scattered across the system


@runtime_checkable
class AuthenticationProtocol(Protocol):
    """Protocol for authentication services."""

    @abstractmethod
    async def authenticate(self, credentials: dict[str, Any]) -> dict[str, Any] | None:
        """Authenticate user credentials."""
        ...

    @abstractmethod
    async def validate_token(self, token: str) -> dict[str, Any] | None:
        """Validate authentication token."""
        ...


@runtime_checkable
class AuthorizationProtocol(Protocol):
    """Protocol for authorization services."""

    @abstractmethod
    async def authorize(self, user_id: str, resource: str, action: str) -> bool:
        """Check if user is authorized for action on resource."""
        ...

    @abstractmethod
    async def get_user_permissions(self, user_id: str) -> list[str]:
        """Get all permissions for a user."""
        ...


@runtime_checkable
class EncryptionProtocol(Protocol):
    """Protocol for encryption services."""

    @abstractmethod
    async def encrypt(self, data: str, key_id: str | None = None) -> str:
        """Encrypt data using specified or default key."""
        ...

    @abstractmethod
    async def decrypt(self, encrypted_data: str, key_id: str | None = None) -> str:
        """Decrypt data using specified or default key."""
        ...


@runtime_checkable
class SecurityMonitoringProtocol(Protocol):
    """Protocol for security monitoring services."""

    @abstractmethod
    async def log_security_event(self, event_type: str, details: dict[str, Any]) -> None:
        """Log a security-related event."""
        ...

    @abstractmethod
    async def detect_anomaly(self, data: Any) -> dict[str, Any] | None:
        """Detect security anomalies in data."""
        ...


# Additional security protocols to be migrated during consolidation phase
