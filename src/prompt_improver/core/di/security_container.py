"""Security Services Dependency Injection Container (2025).

Specialized DI container for security services including authentication,
authorization, validation, and cryptography services.
"""

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from prompt_improver.core.di.protocols import (
    ContainerRegistryProtocol,
    SecurityContainerProtocol,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


class ServiceLifetime(Enum):
    """Service lifetime management options."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


@dataclass
class SecurityServiceRegistration:
    """Security service registration information."""
    interface: type[Any]
    implementation: type[Any] | None
    lifetime: ServiceLifetime
    factory: Callable[[], Any] | None = None
    initialized: bool = False
    instance: Any = None
    tags: set[str] = field(default_factory=set)
    health_check: Callable[[], Any] | None = None


class SecurityContainer(SecurityContainerProtocol, ContainerRegistryProtocol):
    """Specialized DI container for security services.

    Manages security-related services including:
    - Authentication and authorization
    - Cryptography and key management
    - Input validation and sanitization
    - Security configuration and policies
    - API security and rate limiting

    Follows clean architecture with protocol-based dependencies.
    """

    def __init__(self, name: str = "security") -> None:
        """Initialize security services container.

        Args:
            name: Container identifier for logging
        """
        self.name = name
        self.logger = logger.getChild(f"container.{name}")
        self._services: dict[type[Any], SecurityServiceRegistration] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._initialization_order: list[type[Any]] = []
        self._register_default_services()
        self.logger.debug(f"Security container '{self.name}' initialized")

    def _register_default_services(self) -> None:
        """Register default security services."""
        # Self-registration for dependency injection
        self.register_instance(SecurityContainer, self, tags={"container", "security"})

        # Authentication service factory
        self.register_authentication_service_factory()

        # Authorization service factory
        self.register_authorization_service_factory()

        # Crypto service factory
        self.register_crypto_service_factory()

        # Validation service factory
        self.register_validation_service_factory()

        # API security service factory
        self.register_api_security_service_factory()

        # Rate limiting service factory
        self.register_rate_limiting_service_factory()

        # Security config service factory
        self.register_security_config_service_factory()

    def register_singleton(
        self,
        interface: type[T],
        implementation: type[T],
        tags: set[str] | None = None,
    ) -> None:
        """Register a singleton service.

        Args:
            interface: Service interface/protocol
            implementation: Concrete implementation class
            tags: Optional tags for service categorization
        """
        self._services[interface] = SecurityServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SINGLETON,
            tags=tags or set(),
        )
        self.logger.debug(
            f"Registered singleton: {interface.__name__} -> {implementation.__name__}"
        )

    def register_transient(
        self,
        interface: type[T],
        implementation_or_factory: Any,
        tags: set[str] | None = None,
    ) -> None:
        """Register a transient service.

        Args:
            interface: Service interface/protocol
            implementation_or_factory: Implementation class or factory
            tags: Optional tags for service categorization
        """
        self._services[interface] = SecurityServiceRegistration(
            interface=interface,
            implementation=implementation_or_factory if not callable(implementation_or_factory) else None,
            factory=implementation_or_factory if callable(implementation_or_factory) else None,
            lifetime=ServiceLifetime.TRANSIENT,
            tags=tags or set(),
        )
        self.logger.debug(f"Registered transient: {interface.__name__}")

    def register_factory(
        self,
        interface: type[T],
        factory: Any,
        tags: set[str] | None = None,
    ) -> None:
        """Register a service factory.

        Args:
            interface: Service interface/protocol
            factory: Factory function to create service
            tags: Optional tags for service categorization
        """
        self._services[interface] = SecurityServiceRegistration(
            interface=interface,
            implementation=None,
            lifetime=ServiceLifetime.SINGLETON,
            factory=factory,
            tags=tags or set(),
        )
        interface_name = interface.__name__ if hasattr(interface, "__name__") else str(interface)
        self.logger.debug(f"Registered factory: {interface_name}")

    def register_instance(
        self,
        interface: type[T],
        instance: T,
        tags: set[str] | None = None,
    ) -> None:
        """Register a pre-created service instance.

        Args:
            interface: Service interface/protocol
            instance: Pre-created service instance
            tags: Optional tags for service categorization
        """
        registration = SecurityServiceRegistration(
            interface=interface,
            implementation=type(instance),
            lifetime=ServiceLifetime.SINGLETON,
            initialized=True,
            instance=instance,
            tags=tags or set(),
        )
        self._services[interface] = registration
        self.logger.debug(f"Registered instance: {interface.__name__}")

    async def get(self, interface: type[T]) -> T:
        """Resolve service instance.

        Args:
            interface: Service interface to resolve

        Returns:
            Service instance

        Raises:
            KeyError: If service is not registered
        """
        async with self._lock:
            return await self._resolve_service(interface)

    async def _resolve_service(self, interface: type[T]) -> T:
        """Internal service resolution with lifecycle management.

        Args:
            interface: Service interface to resolve

        Returns:
            Service instance
        """
        if interface not in self._services:
            raise KeyError(f"Security service not registered: {interface.__name__}")

        registration = self._services[interface]

        # Return existing singleton instance
        if (registration.lifetime == ServiceLifetime.SINGLETON and
            registration.initialized and registration.instance is not None):
            return registration.instance

        # Create new instance
        if registration.factory:
            instance = await self._create_from_factory(registration.factory)
        elif registration.implementation:
            instance = await self._create_from_class(registration.implementation)
        else:
            raise ValueError(f"No factory or implementation for {interface.__name__}")

        # Initialize if needed
        if hasattr(instance, "initialize") and asyncio.iscoroutinefunction(instance.initialize):
            await instance.initialize()

        # Store singleton
        if registration.lifetime == ServiceLifetime.SINGLETON:
            registration.instance = instance
            registration.initialized = True
            self._initialization_order.append(interface)

        self.logger.debug(f"Resolved security service: {interface.__name__}")
        return instance

    async def _create_from_factory(self, factory: Callable[[], Any]) -> Any:
        """Create service instance from factory.

        Args:
            factory: Factory function

        Returns:
            Service instance
        """
        if asyncio.iscoroutinefunction(factory):
            return await factory()
        return factory()

    async def _create_from_class(self, implementation: type[Any]) -> Any:
        """Create service instance from class constructor.

        Args:
            implementation: Implementation class

        Returns:
            Service instance
        """
        import inspect

        sig = inspect.signature(implementation.__init__)
        kwargs = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param.annotation != inspect.Parameter.empty:
                try:
                    dependency = await self._resolve_service(param.annotation)
                    kwargs[param_name] = dependency
                except KeyError:
                    if param.default != inspect.Parameter.empty:
                        continue
                    raise

        return implementation(**kwargs)

    def is_registered(self, interface: type[T]) -> bool:
        """Check if service is registered.

        Args:
            interface: Service interface to check

        Returns:
            True if registered, False otherwise
        """
        return interface in self._services

    # Security service factory methods
    def register_authentication_service_factory(self) -> None:
        """Register factory for authentication service."""
        def create_authentication_service():
            try:
                from prompt_improver.security.unified_authentication_manager import (
                    UnifiedAuthenticationManager,
                )
                return UnifiedAuthenticationManager()
            except ImportError:
                # Fallback to basic authentication
                from prompt_improver.security.services.basic_auth_service import (
                    BasicAuthService,
                )
                return BasicAuthService()

        self.register_factory(
            "authentication_service",
            create_authentication_service,
            tags={"authentication", "security", "auth"}
        )
        self.logger.debug("Registered authentication service factory")

    def register_authorization_service_factory(self) -> None:
        """Register factory for authorization service."""
        def create_authorization_service():
            try:
                from prompt_improver.security.unified.authorization_component import (
                    AuthorizationComponent,
                )
                return AuthorizationComponent()
            except ImportError:
                # Fallback to basic authorization
                from prompt_improver.security.services.basic_authz_service import (
                    BasicAuthzService,
                )
                return BasicAuthzService()

        self.register_factory(
            "authorization_service",
            create_authorization_service,
            tags={"authorization", "security", "authz"}
        )
        self.logger.debug("Registered authorization service factory")

    def register_crypto_service_factory(self) -> None:
        """Register factory for cryptography service."""
        def create_crypto_service():
            try:
                from prompt_improver.security.unified_crypto_manager import (
                    UnifiedCryptoManager,
                )
                return UnifiedCryptoManager()
            except ImportError:
                # Fallback to basic crypto
                from prompt_improver.security.services.basic_crypto_service import (
                    BasicCryptoService,
                )
                return BasicCryptoService()

        self.register_factory(
            "crypto_service",
            create_crypto_service,
            tags={"cryptography", "security", "crypto"}
        )
        self.logger.debug("Registered crypto service factory")

    def register_validation_service_factory(self) -> None:
        """Register factory for validation service."""
        def create_validation_service():
            try:
                from prompt_improver.security.unified_validation_manager import (
                    UnifiedValidationManager,
                )
                return UnifiedValidationManager()
            except ImportError:
                # Fallback to basic validation
                from prompt_improver.security.services.basic_validation_service import (
                    BasicValidationService,
                )
                return BasicValidationService()

        self.register_factory(
            "validation_service",
            create_validation_service,
            tags={"validation", "security", "input_sanitization"}
        )
        self.logger.debug("Registered validation service factory")

    def register_api_security_service_factory(self) -> None:
        """Register factory for API security service."""
        def create_api_security_service():
            try:
                from prompt_improver.security.services.api_security_service import (
                    ApiSecurityService,
                )
                return ApiSecurityService()
            except ImportError:
                # Fallback to no-op API security
                from prompt_improver.security.services.noop_api_security import (
                    NoOpApiSecurityService,
                )
                return NoOpApiSecurityService()

        self.register_factory(
            "api_security_service",
            create_api_security_service,
            tags={"api_security", "security", "web"}
        )
        self.logger.debug("Registered API security service factory")

    def register_rate_limiting_service_factory(self) -> None:
        """Register factory for rate limiting service."""
        def create_rate_limiting_service():
            try:
                from prompt_improver.security.services.rate_limiting_service import (
                    RateLimitingService,
                )
                return RateLimitingService()
            except ImportError:
                # Fallback to no-op rate limiting
                from prompt_improver.security.services.noop_rate_limiting import (
                    NoOpRateLimitingService,
                )
                return NoOpRateLimitingService()

        self.register_factory(
            "rate_limiting_service",
            create_rate_limiting_service,
            tags={"rate_limiting", "security", "throttling"}
        )
        self.logger.debug("Registered rate limiting service factory")

    def register_security_config_service_factory(self) -> None:
        """Register factory for security configuration service."""
        def create_security_config_service():
            try:
                from prompt_improver.security.unified_security_config import (
                    UnifiedSecurityConfig,
                )
                return UnifiedSecurityConfig()
            except ImportError:
                # Fallback to environment-based security config
                from prompt_improver.security.services.env_security_config import (
                    EnvSecurityConfig,
                )
                return EnvSecurityConfig()

        self.register_factory(
            "security_config_service",
            create_security_config_service,
            tags={"config", "security"}
        )
        self.logger.debug("Registered security config service factory")

    # SecurityContainerProtocol implementation
    async def get_authentication_service(self) -> Any:
        """Get authentication service instance."""
        return await self.get("authentication_service")

    async def get_authorization_service(self) -> Any:
        """Get authorization service instance."""
        return await self.get("authorization_service")

    async def get_crypto_service(self) -> Any:
        """Get cryptography service instance."""
        return await self.get("crypto_service")

    async def get_validation_service(self) -> Any:
        """Get validation service instance."""
        return await self.get("validation_service")

    async def get_api_security_service(self) -> Any:
        """Get API security service instance."""
        return await self.get("api_security_service")

    async def get_rate_limiting_service(self) -> Any:
        """Get rate limiting service instance."""
        return await self.get("rate_limiting_service")

    async def get_security_config_service(self) -> Any:
        """Get security config service instance."""
        return await self.get("security_config_service")

    # Container lifecycle management
    async def initialize(self) -> None:
        """Initialize all security services."""
        if self._initialized:
            return

        self.logger.info(f"Initializing security container '{self.name}'")

        # Initialize all registered services
        for interface in list(self._services.keys()):
            try:
                await self.get(interface)
            except Exception as e:
                self.logger.exception(f"Failed to initialize {interface}: {e}")
                raise

        self._initialized = True
        self.logger.info(f"Security container '{self.name}' initialization complete")

    async def shutdown(self) -> None:
        """Shutdown all security services gracefully."""
        self.logger.info(f"Shutting down security container '{self.name}'")

        # Shutdown in reverse initialization order
        for interface in reversed(self._initialization_order):
            registration = self._services.get(interface)
            if registration and registration.instance:
                try:
                    if hasattr(registration.instance, "shutdown"):
                        if asyncio.iscoroutinefunction(registration.instance.shutdown):
                            await registration.instance.shutdown()
                        else:
                            registration.instance.shutdown()
                    self.logger.debug(f"Shutdown service: {interface.__name__}")
                except Exception as e:
                    self.logger.exception(f"Error shutting down {interface.__name__}: {e}")

        self._services.clear()
        self._initialization_order.clear()
        self._initialized = False
        self.logger.info(f"Security container '{self.name}' shutdown complete")

    async def health_check(self) -> dict[str, Any]:
        """Check health of all security services."""
        results = {
            "container_status": "healthy",
            "container_name": self.name,
            "initialized": self._initialized,
            "registered_services": len(self._services),
            "services": {},
        }

        for interface, registration in self._services.items():
            service_name = interface.__name__ if hasattr(interface, "__name__") else str(interface)
            try:
                if (registration.initialized and registration.instance and
                    hasattr(registration.instance, "health_check")):

                    health_check = registration.instance.health_check
                    if asyncio.iscoroutinefunction(health_check):
                        service_health = await health_check()
                    else:
                        service_health = health_check()
                    results["services"][service_name] = service_health
                else:
                    results["services"][service_name] = {
                        "status": "healthy",
                        "initialized": registration.initialized,
                    }
            except Exception as e:
                results["services"][service_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                results["container_status"] = "degraded"

        return results

    def get_registration_info(self) -> dict[str, Any]:
        """Get information about registered security services."""
        info = {
            "container_name": self.name,
            "initialized": self._initialized,
            "services": {},
        }

        for interface, registration in self._services.items():
            service_name = interface.__name__ if hasattr(interface, "__name__") else str(interface)
            info["services"][service_name] = {
                "implementation": registration.implementation.__name__ if registration.implementation else "Factory",
                "lifetime": registration.lifetime.value,
                "initialized": registration.initialized,
                "has_instance": registration.instance is not None,
                "tags": list(registration.tags),
            }

        return info

    @asynccontextmanager
    async def managed_lifecycle(self):
        """Managed lifecycle context for security container."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.shutdown()


# Global security container instance
_security_container: SecurityContainer | None = None


def get_security_container() -> SecurityContainer:
    """Get the global security container instance.

    Returns:
        SecurityContainer: Global security container instance
    """
    global _security_container
    if _security_container is None:
        _security_container = SecurityContainer()
    return _security_container


async def shutdown_security_container() -> None:
    """Shutdown the global security container."""
    global _security_container
    if _security_container:
        await _security_container.shutdown()
        _security_container = None
