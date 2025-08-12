"""Core System Components

Core services, setup utilities, and system-wide functionality.
"""

from prompt_improver.core.di.container import (
    DIContainer,
    get_container,
    get_datetime_service,
)
from prompt_improver.core.interfaces.datetime_service import (
    DateTimeServiceProtocol,
    MockDateTimeService,
)
from prompt_improver.core.services.datetime_service import DateTimeService
from prompt_improver.core.services.manager import APESServiceManager
from prompt_improver.core.services.prompt_improvement import PromptImprovementService
from prompt_improver.core.services.startup import (
    StartupOrchestrator,
    init_startup_tasks,
    is_startup_complete,
    shutdown_startup_tasks,
    startup_context,
)
from prompt_improver.core.setup import SystemInitializer

__all__: list[str] = [
    "APESServiceManager",
    "DIContainer",
    "DateTimeService",
    "DateTimeServiceProtocol",
    "MockDateTimeService",
    "PromptDataProtection",
    "PromptImprovementService",
    "StartupOrchestrator",
    "SystemInitializer",
    "get_container",
    "get_datetime_service",
    "init_startup_tasks",
    "is_startup_complete",
    "shutdown_startup_tasks",
    "startup_context",
]
