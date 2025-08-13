"""Clean Architecture Boundary Protocols.

This package defines the protocol boundaries between Clean Architecture layers:
- Presentation Layer (API, CLI, TUI) 
- Application Layer (Services, Workflows)
- Domain Layer (Business Logic, Rules)
- Infrastructure Layer (Repositories, Database, External Services)

CRITICAL: All protocols in this package are infrastructure-agnostic and only
use domain types to prevent circular import risks.
"""

from prompt_improver.core.boundaries.presentation_protocols import (
    APIEndpointProtocol,
    CLICommandProtocol,
    TUIWidgetProtocol,
    WebSocketProtocol,
)

from prompt_improver.core.boundaries.application_protocols import (
    ApplicationServiceProtocol,
    WorkflowOrchestratorProtocol,
    BusinessProcessProtocol,
    ApplicationEventProtocol,
)

from prompt_improver.core.boundaries.domain_protocols import (
    DomainServiceProtocol,
    RuleEngineProtocol,
    DomainEventProtocol,
    BusinessRuleProtocol,
)

from prompt_improver.core.boundaries.infrastructure_protocols import (
    RepositoryProtocol,
    ExternalServiceProtocol,
    CacheServiceProtocol,
    MonitoringServiceProtocol,
)

from prompt_improver.core.boundaries.layer_enforcement import (
    LayerBoundaryEnforcer,
    DependencyValidator,
    CircularImportDetector,
)

__all__ = [
    # Presentation layer protocols
    "APIEndpointProtocol",
    "CLICommandProtocol", 
    "TUIWidgetProtocol",
    "WebSocketProtocol",
    
    # Application layer protocols
    "ApplicationServiceProtocol",
    "WorkflowOrchestratorProtocol",
    "BusinessProcessProtocol",
    "ApplicationEventProtocol",
    
    # Domain layer protocols
    "DomainServiceProtocol",
    "RuleEngineProtocol",
    "DomainEventProtocol",
    "BusinessRuleProtocol",
    
    # Infrastructure layer protocols
    "RepositoryProtocol",
    "ExternalServiceProtocol",
    "CacheServiceProtocol",
    "MonitoringServiceProtocol",
    
    # Layer enforcement
    "LayerBoundaryEnforcer",
    "DependencyValidator", 
    "CircularImportDetector",
]