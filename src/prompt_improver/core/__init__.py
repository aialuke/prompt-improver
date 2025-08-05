"""Core System Components

Core services, setup utilities, and system-wide functionality.
"""

# DI components - these are the new implementations that work independently
from .interfaces.datetime_service import DateTimeServiceProtocol, MockDateTimeService
from .services.datetime_service import DateTimeService
from .di.container import DIContainer, get_datetime_service, get_container

# Core services - import with proper error handling
try:
    from .services.prompt_improvement import PromptImprovementService
    from .services.startup import (
        StartupOrchestrator,
        init_startup_tasks,
        shutdown_startup_tasks,
        startup_context,
        is_startup_complete
    )
    from .services.manager import APESServiceManager
    # Removed PromptDataProtection import to fix circular import - security must be foundational
    from .setup.initializer import SystemInitializer
    from .setup.migration import MigrationManager

    # All legacy services imported successfully
    LEGACY_SERVICES_AVAILABLE = True

except ImportError as e:
    # Set all legacy services to None if any import fails
    PromptImprovementService = None
    StartupOrchestrator = None
    init_startup_tasks = None
    shutdown_startup_tasks = None
    startup_context = None
    is_startup_complete = None
    APESServiceManager = None
    PromptDataProtection = None
    SystemInitializer = None
    MigrationManager = None

    LEGACY_SERVICES_AVAILABLE = False

    import warnings
    warnings.warn(f"Legacy core services not available: {e}")

__all__: list[str] = [
    # DI Components (always available)
    "DateTimeServiceProtocol",
    "MockDateTimeService",
    "DateTimeService",
    "DIContainer",
    "get_datetime_service",
    "get_container",
    # Services (may be None if imports fail)
    "PromptImprovementService",
    "StartupOrchestrator",
    "init_startup_tasks",
    "shutdown_startup_tasks",
    "startup_context",
    "is_startup_complete",
    "APESServiceManager",
    "PromptDataProtection",
    "SystemInitializer",
    "MigrationManager",
    # Status flag
    "LEGACY_SERVICES_AVAILABLE",
]
