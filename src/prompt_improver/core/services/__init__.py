"""Core Services

System services including prompt improvement, startup orchestration,
service management, and security.

New decomposed services following SOLID principles:
- RuleSelectionService: Rule discovery and selection
- PersistenceService: Database operations
- MLEventingService: ML event coordination
- AnalyticsService: Unified analytics facade (v2.0)
"""

# MIGRATION NOTE: AnalyticsService now points to unified service
from prompt_improver.analytics import AnalyticsServiceFacade as AnalyticsService
from prompt_improver.core.services.manager import OrchestrationService as APESServiceManager
from prompt_improver.core.services.ml_eventing_service import MLEventingService
from prompt_improver.core.services.persistence_service import PersistenceService
# NOTE: PromptImprovementService moved to services.prompt.facade to break circular imports
from prompt_improver.core.services.protocols import (
    IAnalyticsService,
    IMLEventingService,
    IPersistenceService,
    IRuleSelectionService,
)
from prompt_improver.core.services.rule_selection_service import RuleSelectionService
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
    "APESServiceManager",
    "PromptDataProtection",
    # "PromptImprovementService", # Moved to services.prompt.facade
    "StartupOrchestrator",
    # Decomposed services
    "AnalyticsService",
    "MLEventingService",
    "PersistenceService",
    "RuleSelectionService",
    # Service protocols
    "IAnalyticsService",
    "IMLEventingService",
    "IPersistenceService",
    "IRuleSelectionService",
    # Startup functions
    "init_startup_tasks",
    "is_startup_complete",
    "shutdown_startup_tasks",
    "startup_context",
]
