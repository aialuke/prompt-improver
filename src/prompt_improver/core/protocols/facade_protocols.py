"""Core Facade Protocols - Protocol-Based Interface Definitions

This module defines all facade protocols to ensure loose coupling throughout 
the application. All facades must implement these protocols to maintain
architectural consistency and enable easy testing/mocking.

Features:
- Protocol-based interfaces for all facades
- Type safety through runtime_checkable protocols  
- Clear contract definitions for facade implementations
- Support for dependency injection testing
- Zero implementation dependencies
"""

from typing import Any, Dict, List, Optional, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class BaseFacadeProtocol(Protocol):
    """Base protocol for all facades with common lifecycle methods."""
    
    async def initialize(self) -> None:
        """Initialize the facade and its components."""
        ...
    
    async def shutdown(self) -> None:
        """Shutdown the facade and cleanup resources."""
        ...
    
    def get_status(self) -> dict[str, Any]:
        """Get current status of the facade."""
        ...
    
    async def health_check(self) -> dict[str, Any]:
        """Perform health check on the facade."""
        ...


@runtime_checkable  
class ConfigurationAccessProtocol(Protocol):
    """Protocol for configuration access patterns."""
    
    def get_config(self) -> Any:
        """Get main application configuration."""
        ...
    
    def get_database_config(self) -> Any:
        """Get database configuration."""
        ...
    
    def get_security_config(self) -> Any:
        """Get security configuration."""
        ...
    
    def get_monitoring_config(self) -> Any:
        """Get monitoring configuration."""
        ...
    
    def get_ml_config(self) -> Any:
        """Get ML configuration."""
        ...
    
    def validate_configuration(self) -> dict[str, Any]:
        """Validate current configuration."""
        ...


@runtime_checkable
class ServiceResolutionProtocol(Protocol):
    """Protocol for service resolution patterns."""
    
    async def get_service(self, service_type: type[T]) -> T:
        """Get service instance by type."""
        ...
    
    async def register_singleton(self, interface: type[T], implementation: type[T]) -> None:
        """Register singleton service."""
        ...
    
    async def register_transient(self, interface: type[T], implementation: type[T]) -> None:
        """Register transient service."""
        ...


@runtime_checkable
class RepositoryAccessProtocol(Protocol):
    """Protocol for repository access patterns."""
    
    async def get_repository(self, protocol_type: type[T]) -> T:
        """Get repository by protocol type."""
        ...
    
    async def get_repository_by_name(self, name: str) -> Any:
        """Get repository by name."""
        ...
    
    def get_available_repositories(self) -> list[str]:
        """Get list of available repository names."""
        ...


@runtime_checkable
class ComponentCoordinationProtocol(Protocol):
    """Protocol for component coordination patterns."""
    
    async def initialize_all(self) -> None:
        """Initialize all components."""
        ...
    
    async def shutdown_all(self) -> None:
        """Shutdown all components."""
        ...
    
    def get_component_status(self) -> dict[str, Any]:
        """Get status of all components."""
        ...


@runtime_checkable
class PerformanceMonitoringProtocol(Protocol):
    """Protocol for performance monitoring patterns."""
    
    async def record_operation(self, operation_name: str, response_time_ms: float, **kwargs) -> None:
        """Record operation performance."""
        ...
    
    async def analyze_trends(self, hours: int = 24) -> dict[str, Any]:
        """Analyze performance trends."""
        ...
    
    async def check_regressions(self) -> list[dict[str, Any]]:
        """Check for performance regressions."""
        ...


@runtime_checkable
class ServiceIntegrationProtocol(Protocol):
    """Protocol for service integration patterns."""
    
    async def get_database_services(self) -> Any:
        """Get database services."""
        ...
    
    async def get_monitoring_facade(self) -> Any:
        """Get monitoring facade."""
        ...
    
    async def get_security_orchestrator(self) -> Any:
        """Get security orchestrator."""
        ...
    
    async def initialize_all_services(self) -> None:
        """Initialize all integrated services."""
        ...


# Composite protocols for specific facade types
@runtime_checkable
class DIFacadeProtocol(BaseFacadeProtocol, ServiceResolutionProtocol, Protocol):
    """Complete protocol for dependency injection facade."""
    
    async def get_core_service(self, service_name: str) -> Any:
        """Get core service by name."""
        ...
    
    async def get_ml_service(self, service_name: str) -> Any:
        """Get ML service by name."""
        ...


@runtime_checkable  
class ConfigFacadeProtocol(BaseFacadeProtocol, ConfigurationAccessProtocol, Protocol):
    """Complete protocol for configuration facade."""
    
    def reload_config(self) -> None:
        """Reload configuration from sources."""
        ...
    
    def get_environment_name(self) -> str:
        """Get current environment name."""
        ...


@runtime_checkable
class RepositoryFacadeProtocol(BaseFacadeProtocol, RepositoryAccessProtocol, Protocol):
    """Complete protocol for repository facade."""
    
    async def get_analytics_repository(self) -> Any:
        """Get analytics repository."""
        ...
    
    async def get_ml_repository(self) -> Any:
        """Get ML repository."""
        ...
    
    def cleanup(self) -> None:
        """Cleanup repository resources."""
        ...


@runtime_checkable
class CLIFacadeProtocol(BaseFacadeProtocol, ComponentCoordinationProtocol, Protocol):
    """Complete protocol for CLI facade."""
    
    def get_orchestrator(self) -> Any:
        """Get CLI orchestrator."""
        ...
    
    def get_signal_handler(self) -> Any:
        """Get signal handler."""
        ...
    
    async def create_emergency_checkpoint(self) -> dict[str, Any]:
        """Create emergency checkpoint."""
        ...


@runtime_checkable
class APIFacadeProtocol(BaseFacadeProtocol, ServiceIntegrationProtocol, Protocol):
    """Complete protocol for API facade."""
    
    async def get_container(self) -> Any:
        """Get dependency injection container."""
        ...
    
    def setup_telemetry(self, app: Any, service_name: str) -> None:
        """Setup telemetry for application."""
        ...


@runtime_checkable
class PerformanceFacadeProtocol(BaseFacadeProtocol, PerformanceMonitoringProtocol, Protocol):
    """Complete protocol for performance facade."""
    
    async def get_baseline_system(self) -> Any:
        """Get baseline system."""
        ...
    
    async def get_profiler(self) -> Any:
        """Get profiler."""
        ...
    
    async def generate_report(self, report_type: str = "daily") -> dict[str, Any]:
        """Generate performance report."""
        ...


# Unified facade coordinator protocol
@runtime_checkable
class FacadeCoordinatorProtocol(Protocol):
    """Protocol for coordinating multiple facades together."""
    
    def get_di_facade(self) -> DIFacadeProtocol:
        """Get dependency injection facade."""
        ...
    
    def get_config_facade(self) -> ConfigFacadeProtocol:
        """Get configuration facade."""
        ...
    
    def get_repository_facade(self) -> RepositoryFacadeProtocol:
        """Get repository facade."""
        ...
    
    def get_cli_facade(self) -> CLIFacadeProtocol:
        """Get CLI facade."""
        ...
    
    def get_api_facade(self) -> APIFacadeProtocol:
        """Get API facade."""
        ...
    
    def get_performance_facade(self) -> PerformanceFacadeProtocol:
        """Get performance facade."""
        ...
    
    async def initialize_all_facades(self) -> None:
        """Initialize all facades."""
        ...
    
    async def shutdown_all_facades(self) -> None:
        """Shutdown all facades."""
        ...
    
    async def health_check_all_facades(self) -> dict[str, Any]:
        """Health check all facades."""
        ...


__all__ = [
    # Base protocols
    "BaseFacadeProtocol",
    "ConfigurationAccessProtocol", 
    "ServiceResolutionProtocol",
    "RepositoryAccessProtocol",
    "ComponentCoordinationProtocol",
    "PerformanceMonitoringProtocol",
    "ServiceIntegrationProtocol",
    
    # Composite facade protocols
    "DIFacadeProtocol",
    "ConfigFacadeProtocol",
    "RepositoryFacadeProtocol",
    "CLIFacadeProtocol", 
    "APIFacadeProtocol",
    "PerformanceFacadeProtocol",
    
    # Coordinator protocol
    "FacadeCoordinatorProtocol",
]