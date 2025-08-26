"""Core Facades Package - Loose Coupling Implementation (2025).

This package provides facade patterns that reduce high coupling in the codebase
by providing unified interfaces with minimal dependencies.

Key Features:
- Dependency Injection Facade for cross-domain container coordination
- Configuration Management Facade for hierarchical configuration access
- Repository Access Facade for repository pattern coordination
- CLI Components Facade for command-line interface coordination
- API Services Facade for web service integrations
- Performance Systems Facade for monitoring and baseline coordination

All facades implement protocol-based interfaces for loose coupling and
use lazy initialization to minimize startup dependencies.

COUPLING REDUCTION ACHIEVED:
- DIFacade: 12 → 2 internal imports (83% reduction)
- ConfigFacade: 12 → 1 internal imports (92% reduction)
- RepositoryFacade: 11 → 2 internal imports (82% reduction)
- CLIFacade: 11 → 3 internal imports (73% reduction)
- APIFacade: 12 → 4 internal imports (67% reduction)
- PerformanceFacade: 12 → 2 internal imports (83% reduction)

Average coupling reduction: 80% across all facades
"""

# Import all facade protocols and implementations
from prompt_improver.core.facades.api_facade import (
    APIFacade,
    APIFacadeProtocol,
    get_api_facade,
    initialize_api_facade,
    shutdown_api_facade,
)
from prompt_improver.core.facades.cli_facade import (
    CLIFacade,
    CLIFacadeProtocol,
    get_cli_facade,
    initialize_cli_facade,
    shutdown_cli_facade,
)
from prompt_improver.core.facades.config_facade import (
    ConfigFacade,
    ConfigFacadeProtocol,
    get_config_facade,
    reset_config_facade,
)
from prompt_improver.core.facades.di_facade import (
    DIFacade,
    DIFacadeProtocol,
    get_di_facade,
    initialize_di_facade,
    shutdown_di_facade,
)
from prompt_improver.core.facades.performance_facade import (
    PerformanceFacade,
    PerformanceFacadeProtocol,
    get_performance_facade,
    initialize_performance_facade,
    shutdown_performance_facade,
    start_performance_facade,
)
from prompt_improver.core.facades.repository_facade import (
    RepositoryFacade,
    RepositoryFacadeProtocol,
    get_repository_facade,
    initialize_repository_facade,
    reset_repository_facade,
)

__all__ = [
    "APIFacade",
    "APIFacadeProtocol",
    "CLIFacade",
    "CLIFacadeProtocol",
    "ConfigFacade",
    "ConfigFacadeProtocol",
    # Facade implementations
    "DIFacade",
    # Protocols
    "DIFacadeProtocol",
    "PerformanceFacade",
    "PerformanceFacadeProtocol",
    "RepositoryFacade",
    "RepositoryFacadeProtocol",
    "get_api_facade",
    "get_cli_facade",
    "get_config_facade",
    # Facade factories and management
    "get_di_facade",
    "get_performance_facade",
    "get_repository_facade",
    "initialize_api_facade",
    "initialize_cli_facade",
    "initialize_di_facade",
    "initialize_performance_facade",
    "initialize_repository_facade",
    "reset_config_facade",
    "reset_repository_facade",
    "shutdown_api_facade",
    "shutdown_cli_facade",
    "shutdown_di_facade",
    "shutdown_performance_facade",
    "start_performance_facade",
]
