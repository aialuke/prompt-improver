"""Core Services.

System services including prompt improvement, startup orchestration,
service management, and security.

New decomposed services following SOLID principles:
- PersistenceService: Database operations
- MLEventingService: ML event coordination
- AnalyticsService: Unified analytics facade (v2.0)
- CLIServiceFactory: CLI service registration with ServiceRegistry (2025)

ServiceRegistry Migration Status:
- ✅ CLI service factory integrated for lazy import elimination
- ✅ Service registration during startup orchestration
- ✅ Backward compatibility maintained

NOTE: RuleSelectionService moved to application layer (Clean Architecture 2025)
"""

from typing import TYPE_CHECKING

# MIGRATION NOTE: AnalyticsService now points to unified service
# Move to TYPE_CHECKING to break circular import chain
if TYPE_CHECKING:
    pass
else:
    AnalyticsService = None
from prompt_improver.core.services.manager import (
    OrchestrationService,
)

if TYPE_CHECKING:
    from prompt_improver.core.services.ml_eventing_service import MLEventingService
else:
    MLEventingService = None
# CLEAN ARCHITECTURE 2025: RuleSelectionService moved to application layer
# REMOVED: Direct import violates layer separation and causes circular imports
# Applications should import from application layer directly
from prompt_improver.analytics import AnalyticsServiceFacade
from prompt_improver.core.services.cli_service_factory import CLIServiceFactory
from prompt_improver.core.services.persistence_service import PersistenceService

# NOTE: PromptImprovementService moved to services.prompt.facade to break circular imports
from prompt_improver.core.services.protocols import (
    IAnalyticsService,
    IMLEventingService,
    IPersistenceService,
    IRuleSelectionService,
)
from prompt_improver.core.services.security import PromptDataProtection
from prompt_improver.core.services.startup import (
    StartupOrchestrator,
    init_startup_tasks,
    is_startup_complete,
    shutdown_startup_tasks,
    startup_context,
)

__all__ = [
    # Main services
    # Decomposed services
    "AnalyticsServiceFacade",
    # "RuleSelectionService",  # REMOVED: Moved to application layer
    # ServiceRegistry integration (2025 Migration)
    "CLIServiceFactory",
    # Service protocols
    "IAnalyticsService",
    "IMLEventingService",
    "IPersistenceService",
    "IRuleSelectionService",
    "MLEventingService",
    "OrchestrationService",
    "PersistenceService",
    "PromptDataProtection",
    # "PromptImprovementService", # Moved to services.prompt.facade
    "StartupOrchestrator",
    # Startup functions
    "init_startup_tasks",
    "is_startup_complete",
    "shutdown_startup_tasks",
    "startup_context",
]
