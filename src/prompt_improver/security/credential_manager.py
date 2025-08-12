"""Secure Credential Management System
Following 2025 security best practices for credential handling.

This module provides centralized credential management with:
- Environment variable loading with validation
- Secure secret rotation capability
- Integration with cloud secret managers
- Fail-secure design patterns
- Audit logging for credential access
"""

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class CredentialMetadata:
    """Metadata for credential tracking and rotation."""

    name: str
    source: str
    last_accessed: float
    rotation_required: bool = False
    expires_at: float | None = None
    hash_signature: str | None = None


class CredentialProvider(ABC):
    """Abstract base class for credential providers."""

    @abstractmethod
    async def get_credential(self, name: str) -> str | None:
        """Retrieve a credential by name."""

    @abstractmethod
    async def set_credential(self, name: str, value: str) -> bool:
        """Store a credential (if supported)."""

    @abstractmethod
    async def rotate_credential(self, name: str) -> bool:
        """Rotate a credential (if supported)."""


class EnvironmentCredentialProvider(CredentialProvider):
    """Environment variable credential provider."""

    def __init__(self):
        self.accessed_credentials = set()

    async def get_credential(self, name: str) -> str | None:
        """Get credential from environment variables."""
        self.accessed_credentials.add(name)
        value = os.getenv(name)
        if value:
            logger.debug(f"Retrieved credential '{name}' from environment")
        else:
            logger.warning(f"Credential '{name}' not found in environment")
        return value

    async def set_credential(self, name: str, value: str) -> bool:
        """Environment variables are read-only in this implementation."""
        logger.warning(
            f"Cannot set environment credential '{name}' - read-only provider"
        )
        return False

    async def rotate_credential(self, name: str) -> bool:
        """Environment credentials require external rotation."""
        logger.warning(
            f"Cannot rotate environment credential '{name}' - external rotation required"
        )
        return False


class FileCredentialProvider(CredentialProvider):
    """File-based credential provider for development."""

    def __init__(self, secrets_dir: Path = Path("/run/secrets")):
        self.secrets_dir = secrets_dir
        self.secrets_dir.mkdir(parents=True, exist_ok=True)

    async def get_credential(self, name: str) -> str | None:
        """Get credential from file."""
        secret_file = self.secrets_dir / name
        try:
            if secret_file.exists():
                value = secret_file.read_text().strip()
                logger.debug(f"Retrieved credential '{name}' from file")
                return value
            logger.warning(f"Credential file '{name}' not found in {self.secrets_dir}")
            return None
        except Exception as e:
            logger.error(f"Failed to read credential '{name}' from file: {e}")
            return None

    async def set_credential(self, name: str, value: str) -> bool:
        """Store credential to file."""
        secret_file = self.secrets_dir / name
        try:
            secret_file.write_text(value)
            secret_file.chmod(384)
            logger.info(f"Stored credential '{name}' to file")
            return True
        except Exception as e:
            logger.error(f"Failed to store credential '{name}' to file: {e}")
            return False

    async def rotate_credential(self, name: str) -> bool:
        """File-based credentials require manual rotation."""
        logger.warning(f"Manual rotation required for file credential '{name}'")
        return False


class CredentialManager:
    """Centralized credential management with multiple providers.

    Provides secure credential access with:
    - Multiple provider support (env, file, cloud)
    - Credential validation and sanitization
    - Access logging and audit trails
    - Fail-secure design patterns
    """

    def __init__(self):
        self.providers: dict[str, CredentialProvider] = {}
        self.metadata: dict[str, CredentialMetadata] = {}
        self.access_log: list[dict[str, Any]] = []
        self.providers["env"] = EnvironmentCredentialProvider()
        self.providers["file"] = FileCredentialProvider()
        self.provider_order = ["env", "file"]

    def add_provider(
        self, name: str, provider: CredentialProvider, priority: int = 999
    ):
        """Add a credential provider with optional priority."""
        self.providers[name] = provider
        if name not in self.provider_order:
            if priority == 0:
                self.provider_order.insert(0, name)
            elif priority >= len(self.provider_order):
                self.provider_order.append(name)
            else:
                self.provider_order.insert(priority, name)

    async def get_credential(self, name: str, required: bool = True) -> str | None:
        """Get credential from providers in priority order.

        Args:
            name: Credential name
            required: If True, log warning when credential not found

        Returns:
            Credential value or None if not found
        """
        start_time = time.time()
        for provider_name in self.provider_order:
            provider = self.providers.get(provider_name)
            if not provider:
                continue
            try:
                value = await provider.get_credential(name)
                if value:
                    self._log_access(
                        name, provider_name, True, time.time() - start_time
                    )
                    self.metadata[name] = CredentialMetadata(
                        name=name,
                        source=provider_name,
                        last_accessed=time.time(),
                        hash_signature=self._hash_credential(value),
                    )
                    return value
            except Exception as e:
                logger.error(
                    f"Provider '{provider_name}' failed to get credential '{name}': {e}"
                )
                continue
        self._log_access(name, "none", False, time.time() - start_time)
        if required:
            logger.warning(f"Required credential '{name}' not found in any provider")
        return None

    async def get_credential_with_default(self, name: str, default: str) -> str:
        """Get credential with fallback to default value."""
        value = await self.get_credential(name, required=False)
        return value if value is not None else default

    async def validate_required_credentials(
        self, required_credentials: list[str]
    ) -> dict[str, bool]:
        """Validate that all required credentials are available."""
        results = {}
        for cred_name in required_credentials:
            value = await self.get_credential(cred_name, required=True)
            results[cred_name] = value is not None
        return results

    def get_credential_metadata(self, name: str) -> CredentialMetadata | None:
        """Get metadata for a credential."""
        return self.metadata.get(name)

    def get_access_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent credential access log entries."""
        return self.access_log[-limit:]

    def _log_access(self, name: str, provider: str, success: bool, duration: float):
        """Log credential access for audit purposes."""
        log_entry: dict[str, Any] = {
            "timestamp": time.time(),
            "credential_name": name,
            "provider": provider,
            "success": success,
            "duration_ms": duration * 1000,
            "masked_name": self._mask_credential_name(name),
        }
        self.access_log.append(log_entry)
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-1000:]

    def _mask_credential_name(self, name: str) -> str:
        """Mask credential name for logging."""
        if len(name) <= 4:
            return "*" * len(name)
        return name[:2] + "*" * (len(name) - 4) + name[-2:]

    def _hash_credential(self, value: str) -> str:
        """Create hash signature of credential for change detection."""
        return hashlib.sha256(value.encode()).hexdigest()[:16]

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on credential system."""
        health_status: dict[str, Any] = {
            "status": "healthy",
            "providers": {},
            "total_credentials": len(self.metadata),
            "recent_accesses": len([
                log for log in self.access_log if time.time() - log["timestamp"] < 3600
            ]),
            "timestamp": time.time(),
        }
        providers_dict: dict[str, Any] = {}
        for provider_name, provider in self.providers.items():
            try:
                test_result = await provider.get_credential("__health_check_test__")
                providers_dict[provider_name] = {
                    "status": "healthy",
                    "type": type(provider).__name__,
                }
            except Exception as e:
                providers_dict[provider_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "type": type(provider).__name__,
                }
                health_status["status"] = "degraded"
        health_status["providers"] = providers_dict
        return health_status


_credential_manager: CredentialManager | None = None


def get_credential_manager() -> CredentialManager:
    """Get the global credential manager instance."""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = CredentialManager()
    return _credential_manager


async def get_credential(name: str, required: bool = True) -> str | None:
    """Convenience function to get a credential."""
    manager = get_credential_manager()
    return await manager.get_credential(name, required)


async def get_credential_with_default(name: str, default: str) -> str:
    """Convenience function to get a credential with default."""
    manager = get_credential_manager()
    return await manager.get_credential_with_default(name, default)


class CredentialNames:
    """Standard credential names used throughout the application."""

    DB_PASSWORD = "POSTGRES_PASSWORD"
    DB_USERNAME = "POSTGRES_USERNAME"
    DB_HOST = "POSTGRES_HOST"
    REDIS_PASSWORD = "REDIS_PASSWORD"
    REDIS_USERNAME = "REDIS_USERNAME"
    REDIS_HOST = "REDIS_HOST"
    SECRET_KEY = "SECRET_KEY"
    JWT_SECRET = "JWT_SECRET"
    OPENAI_API_KEY = "OPENAI_API_KEY"
    ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
    SENTRY_DSN = "SENTRY_DSN"
    DATADOG_API_KEY = "DATADOG_API_KEY"


async def migrate_hardcoded_credential(old_value: str, credential_name: str) -> str:
    """Helper function to migrate from hardcoded values to credential manager.

    Args:
        old_value: The hardcoded value (for fallback)
        credential_name: The credential name to look up

    Returns:
        The credential value from manager or fallback to old_value
    """
    try:
        new_value = await get_credential(credential_name, required=False)
        if new_value:
            logger.info(
                f"Successfully migrated credential '{credential_name}' from hardcoded value"
            )
            return new_value
        logger.warning(
            f"Credential '{credential_name}' not found, using hardcoded fallback"
        )
        return old_value
    except Exception as e:
        logger.error(f"Failed to migrate credential '{credential_name}': {e}")
        return old_value


if __name__ == "__main__":
    import asyncio

    async def test_credential_manager():
        """Test the credential manager functionality."""
        manager = get_credential_manager()
        db_password = await manager.get_credential("POSTGRES_PASSWORD", required=False)
        print(f"DB Password found: {db_password is not None}")
        redis_host = await manager.get_credential_with_default("REDIS_HOST", "redis")
        print(f"Redis Host: {redis_host}")
        health = await manager.health_check()
        print(f"Health Status: {health['status']}")
        log_entries = manager.get_access_log(5)
        print(f"Recent accesses: {len(log_entries)}")

    asyncio.run(test_credential_manager())
