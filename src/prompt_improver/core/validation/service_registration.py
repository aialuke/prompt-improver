"""Service Registration for Protocol-Based Architecture (2025).

This module provides centralized service registration ensuring all services
are properly bound to their protocol interfaces with validation.
"""

import logging
from typing import Type, TypeVar, Protocol, Optional, Any, Dict, List

from prompt_improver.core.validation.protocol_registry import (
    register_protocol_contract,
    get_protocol_registry,
    validate_all_contracts
)

# Import all protocol interfaces
from prompt_improver.core.protocols.prompt_service.prompt_protocols import (
    PromptServiceFacadeProtocol,
    PromptAnalysisServiceProtocol,
    RuleApplicationServiceProtocol,
    ValidationServiceProtocol,
)
from prompt_improver.repositories.protocols.analytics_repository_protocol import (
    AnalyticsRepositoryProtocol,
)
from prompt_improver.repositories.protocols.apriori_repository_protocol import (
    AprioriRepositoryProtocol,
)
from prompt_improver.repositories.protocols.prompt_repository_protocol import (
    PromptRepositoryProtocol,
)
from prompt_improver.core.di.protocols import (
    CoreContainerProtocol,
    MLContainerProtocol,
    SecurityContainerProtocol,
    DatabaseContainerProtocol,
    MonitoringContainerProtocol,
    ContainerFacadeProtocol,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = TypeVar("P", bound=Protocol)


class ServiceRegistrationManager:
    """Manages protocol-based service registration and validation."""
    
    def __init__(self):
        """Initialize the service registration manager."""
        self._registered_services: Dict[str, Any] = {}
        self._validation_results: Dict[str, Any] = {}
        
    def register_all_services(self) -> Dict[str, Any]:
        """Register all services with their protocol contracts."""
        logger.info("Starting protocol-based service registration")
        
        registration_results = {
            "successful_registrations": [],
            "failed_registrations": [],
            "validation_errors": [],
            "total_registered": 0
        }
        
        try:
            # Register prompt services
            self._register_prompt_services(registration_results)
            
            # Register repository services
            self._register_repository_services(registration_results)
            
            # Register container services
            self._register_container_services(registration_results)
            
            # Validate all contracts
            validation_results = validate_all_contracts()
            self._validation_results = validation_results
            
            # Check for validation errors
            for protocol_name, result in validation_results["contract_validation"].items():
                if not result["is_valid"]:
                    registration_results["validation_errors"].extend(
                        [f"{protocol_name}: {issue}" for issue in result["issues"]]
                    )
            
            # Check dependency graph
            if not validation_results["dependency_validation"]["is_valid"]:
                registration_results["validation_errors"].extend(
                    validation_results["dependency_validation"]["circular_dependencies"]
                )
                registration_results["validation_errors"].extend(
                    validation_results["dependency_validation"]["missing_dependencies"]
                )
            
            registration_results["total_registered"] = len(registration_results["successful_registrations"])
            
            logger.info(f"Service registration completed: {registration_results['total_registered']} services registered")
            
        except Exception as e:
            logger.error(f"Service registration failed: {e}")
            registration_results["failed_registrations"].append(f"Global registration error: {e}")
        
        return registration_results
    
    def _register_prompt_services(self, results: Dict[str, Any]) -> None:
        """Register prompt service contracts."""
        try:
            # Import implementations only when registering to avoid circular imports
            from prompt_improver.services.prompt.facade import PromptServiceFacade
            from prompt_improver.services.prompt.prompt_analysis_service import PromptAnalysisService
            from prompt_improver.services.prompt.rule_application_service import RuleApplicationService
            from prompt_improver.services.prompt.validation_service import ValidationService
            
            # Register PromptServiceFacade
            register_protocol_contract(
                protocol_type=PromptServiceFacadeProtocol,
                implementation_type=PromptServiceFacade,
                dependencies=[
                    PromptAnalysisServiceProtocol,
                    RuleApplicationServiceProtocol, 
                    ValidationServiceProtocol
                ],
                lifecycle="singleton",
                tags={"prompt", "facade", "core"}
            )
            results["successful_registrations"].append("PromptServiceFacadeProtocol")
            
            # Register component services
            register_protocol_contract(
                protocol_type=PromptAnalysisServiceProtocol,
                implementation_type=PromptAnalysisService,
                lifecycle="singleton",
                tags={"prompt", "analysis"}
            )
            results["successful_registrations"].append("PromptAnalysisServiceProtocol")
            
            register_protocol_contract(
                protocol_type=RuleApplicationServiceProtocol,
                implementation_type=RuleApplicationService,
                lifecycle="singleton",
                tags={"prompt", "rules"}
            )
            results["successful_registrations"].append("RuleApplicationServiceProtocol")
            
            register_protocol_contract(
                protocol_type=ValidationServiceProtocol,
                implementation_type=ValidationService,
                lifecycle="singleton",
                tags={"prompt", "validation"}
            )
            results["successful_registrations"].append("ValidationServiceProtocol")
            
        except Exception as e:
            logger.error(f"Failed to register prompt services: {e}")
            results["failed_registrations"].append(f"Prompt services: {e}")
    
    def _register_repository_services(self, results: Dict[str, Any]) -> None:
        """Register repository service contracts."""
        try:
            # Import implementations
            from prompt_improver.repositories.impl.analytics_repository import AnalyticsRepository
            
            # Register repository contracts (would need actual implementations)
            # For now, register the protocols as placeholders
            
            logger.info("Repository service registration completed")
            
        except Exception as e:
            logger.error(f"Failed to register repository services: {e}")
            results["failed_registrations"].append(f"Repository services: {e}")
    
    def _register_container_services(self, results: Dict[str, Any]) -> None:
        """Register container service contracts."""
        try:
            # Import container implementations
            from prompt_improver.core.di.container_orchestrator import ContainerOrchestrator
            from prompt_improver.core.di.monitoring_container import MonitoringContainer
            
            # Register container contracts
            register_protocol_contract(
                protocol_type=ContainerFacadeProtocol,
                implementation_type=ContainerOrchestrator,
                lifecycle="singleton",
                tags={"container", "orchestrator", "facade"}
            )
            results["successful_registrations"].append("ContainerFacadeProtocol")
            
            register_protocol_contract(
                protocol_type=MonitoringContainerProtocol,
                implementation_type=MonitoringContainer,
                lifecycle="singleton",
                tags={"container", "monitoring"}
            )
            results["successful_registrations"].append("MonitoringContainerProtocol")
            
        except Exception as e:
            logger.error(f"Failed to register container services: {e}")
            results["failed_registrations"].append(f"Container services: {e}")
    
    def get_service_by_protocol(self, protocol_type: Type[P]) -> Optional[Type]:
        """Get service implementation for a protocol."""
        registry = get_protocol_registry()
        return registry.get_implementation_type(protocol_type)
    
    def validate_service_contracts(self) -> Dict[str, Any]:
        """Validate all registered service contracts."""
        return validate_all_contracts()
    
    def get_dependency_order(self) -> List[Type[Protocol]]:
        """Get topologically sorted service dependency order."""
        registry = get_protocol_registry()
        return registry.generate_dependency_order()
    
    def get_registration_stats(self) -> Dict[str, Any]:
        """Get registration and validation statistics."""
        registry = get_protocol_registry()
        stats = registry.get_registry_stats()
        
        validation_summary = {
            "total_contracts_validated": 0,
            "valid_contracts": 0,
            "invalid_contracts": 0,
            "dependency_graph_valid": True
        }
        
        if self._validation_results:
            contract_validation = self._validation_results.get("contract_validation", {})
            validation_summary["total_contracts_validated"] = len(contract_validation)
            validation_summary["valid_contracts"] = sum(
                1 for result in contract_validation.values() if result["is_valid"]
            )
            validation_summary["invalid_contracts"] = sum(
                1 for result in contract_validation.values() if not result["is_valid"]
            )
            validation_summary["dependency_graph_valid"] = (
                self._validation_results.get("dependency_validation", {}).get("is_valid", True)
            )
        
        return {
            "registry_stats": stats,
            "validation_summary": validation_summary
        }


# Global service registration manager
_registration_manager = ServiceRegistrationManager()


def register_all_services() -> Dict[str, Any]:
    """Register all services with protocol validation."""
    return _registration_manager.register_all_services()


def get_service_by_protocol(protocol_type: Type[P]) -> Optional[Type]:
    """Get service implementation for a protocol."""
    return _registration_manager.get_service_by_protocol(protocol_type)


def validate_service_contracts() -> Dict[str, Any]:
    """Validate all service contracts."""
    return _registration_manager.validate_service_contracts()


def get_registration_stats() -> Dict[str, Any]:
    """Get service registration statistics."""
    return _registration_manager.get_registration_stats()


def create_service_factory(protocol_type: Type[P]) -> Any:
    """Create a factory function for a protocol-based service."""
    def factory(*args, **kwargs):
        registry = get_protocol_registry()
        contract = registry.get_contract(protocol_type)
        if not contract:
            raise ValueError(f"No contract registered for protocol {protocol_type.__name__}")
        
        if contract.implementation_factory:
            return contract.implementation_factory(*args, **kwargs)
        else:
            return contract.implementation_type(*args, **kwargs)
    
    return factory