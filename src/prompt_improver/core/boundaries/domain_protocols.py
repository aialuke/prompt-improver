"""Domain Layer Boundary Protocols.

These protocols define the contracts for domain services and business logic.
Domain layer should be the most stable layer with minimal external dependencies.

Clean Architecture Rule: Domain layer has no external dependencies.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from datetime import datetime

from prompt_improver.core.domain.types import (
    RuleId,
    SessionId,
    UserId,
    BusinessRuleData,
    ValidationConstraintData,
)
from prompt_improver.core.domain.enums import ValidationLevel


@runtime_checkable
class DomainServiceProtocol(Protocol):
    """Base protocol for domain services containing business logic."""
    
    def get_service_name(self) -> str:
        """Get the name of this domain service.
        
        Returns:
            Service name
        """
        ...
    
    async def validate_business_invariants(
        self,
        entity_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate business invariants for an entity.
        
        Args:
            entity_data: Entity data to validate
            
        Returns:
            Validation results with any violations
        """
        ...


@runtime_checkable
class RuleEngineProtocol(Protocol):
    """Protocol for rule engine operations."""
    
    async def evaluate_rule(
        self,
        rule_id: RuleId,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single rule against input data.
        
        Args:
            rule_id: Rule to evaluate
            input_data: Data to evaluate
            context: Optional evaluation context
            
        Returns:
            Rule evaluation results
        """
        ...
    
    async def evaluate_rule_set(
        self,
        rule_ids: List[RuleId],
        input_data: Dict[str, Any],
        evaluation_strategy: str = "all",
    ) -> Dict[str, Any]:
        """Evaluate a set of rules.
        
        Args:
            rule_ids: Rules to evaluate
            input_data: Data to evaluate
            evaluation_strategy: How to combine results ("all", "any", "priority")
            
        Returns:
            Combined evaluation results
        """
        ...
    
    async def get_applicable_rules(
        self,
        input_data: Dict[str, Any],
        rule_category: Optional[str] = None,
    ) -> List[RuleId]:
        """Get rules applicable to the input data.
        
        Args:
            input_data: Data to find rules for
            rule_category: Optional category filter
            
        Returns:
            List of applicable rule IDs
        """
        ...
    
    async def register_rule(
        self,
        rule_data: BusinessRuleData
    ) -> RuleId:
        """Register a new business rule.
        
        Args:
            rule_data: Rule definition data
            
        Returns:
            ID of the registered rule
        """
        ...


@runtime_checkable
class DomainEventProtocol(Protocol):
    """Protocol for domain event handling."""
    
    async def emit_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        aggregate_id: Optional[str] = None,
    ) -> None:
        """Emit a domain event.
        
        Args:
            event_type: Type of domain event
            event_data: Event payload
            aggregate_id: Optional aggregate identifier
        """
        ...
    
    async def handle_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        event_id: str,
    ) -> Dict[str, Any]:
        """Handle a domain event.
        
        Args:
            event_type: Type of event to handle
            event_data: Event payload
            event_id: Event identifier
            
        Returns:
            Event handling results
        """
        ...
    
    def get_event_handlers(
        self,
        event_type: str
    ) -> List[str]:
        """Get registered handlers for an event type.
        
        Args:
            event_type: Event type to query
            
        Returns:
            List of handler names
        """
        ...


@runtime_checkable
class BusinessRuleProtocol(Protocol):
    """Protocol for individual business rules."""
    
    def get_rule_id(self) -> RuleId:
        """Get unique identifier for this rule.
        
        Returns:
            Rule identifier
        """
        ...
    
    def get_rule_name(self) -> str:
        """Get human-readable name for this rule.
        
        Returns:
            Rule name
        """
        ...
    
    def get_rule_description(self) -> str:
        """Get description of what this rule does.
        
        Returns:
            Rule description
        """
        ...
    
    async def applies_to(
        self,
        input_data: Dict[str, Any]
    ) -> bool:
        """Check if this rule applies to the input data.
        
        Args:
            input_data: Data to check
            
        Returns:
            Whether rule applies to this data
        """
        ...
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the business rule.
        
        Args:
            input_data: Data to process
            context: Optional execution context
            
        Returns:
            Rule execution results
        """
        ...
    
    def get_priority(self) -> int:
        """Get rule execution priority.
        
        Returns:
            Priority value (higher = more important)
        """
        ...


@runtime_checkable
class ValidationServiceProtocol(Protocol):
    """Protocol for domain validation logic."""
    
    async def validate_entity(
        self,
        entity_type: str,
        entity_data: Dict[str, Any],
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
    ) -> Dict[str, Any]:
        """Validate an entity according to business rules.
        
        Args:
            entity_type: Type of entity to validate
            entity_data: Entity data
            validation_level: Level of validation to apply
            
        Returns:
            Validation results
        """
        ...
    
    async def validate_operation(
        self,
        operation: str,
        operation_data: Dict[str, Any],
        user_id: Optional[UserId] = None,
    ) -> Dict[str, Any]:
        """Validate a business operation.
        
        Args:
            operation: Operation name
            operation_data: Operation parameters
            user_id: Optional user performing operation
            
        Returns:
            Operation validation results
        """
        ...
    
    async def get_validation_constraints(
        self,
        entity_type: str
    ) -> ValidationConstraintData:
        """Get validation constraints for an entity type.
        
        Args:
            entity_type: Entity type to query
            
        Returns:
            Validation constraints
        """
        ...
    
    async def add_custom_validator(
        self,
        validator_name: str,
        validator_function: Any,  # Callable
        entity_types: List[str],
    ) -> bool:
        """Add a custom validation function.
        
        Args:
            validator_name: Name for the validator
            validator_function: Validation function
            entity_types: Entity types this validator applies to
            
        Returns:
            Whether validator was added successfully
        """
        ...


@runtime_checkable
class AggregateRootProtocol(Protocol):
    """Protocol for aggregate root entities in domain model."""
    
    def get_aggregate_id(self) -> str:
        """Get unique identifier for this aggregate.
        
        Returns:
            Aggregate identifier
        """
        ...
    
    def get_aggregate_version(self) -> int:
        """Get current version of this aggregate.
        
        Returns:
            Version number
        """
        ...
    
    def get_uncommitted_events(self) -> List[Dict[str, Any]]:
        """Get domain events that haven't been committed.
        
        Returns:
            List of uncommitted domain events
        """
        ...
    
    def mark_events_as_committed(self) -> None:
        """Mark all uncommitted events as committed."""
        ...
    
    async def apply_event(
        self,
        event: Dict[str, Any]
    ) -> None:
        """Apply a domain event to this aggregate.
        
        Args:
            event: Domain event to apply
        """
        ...


@runtime_checkable
class ValueObjectProtocol(Protocol):
    """Protocol for value objects in domain model."""
    
    def equals(self, other: Any) -> bool:
        """Check equality with another value object.
        
        Args:
            other: Object to compare with
            
        Returns:
            Whether objects are equal
        """
        ...
    
    def get_hash(self) -> int:
        """Get hash code for this value object.
        
        Returns:
            Hash code
        """
        ...
    
    def is_valid(self) -> bool:
        """Check if this value object is in a valid state.
        
        Returns:
            Whether object is valid
        """
        ...


@runtime_checkable
class DomainCalculationProtocol(Protocol):
    """Protocol for domain-specific calculations and algorithms."""
    
    async def calculate(
        self,
        input_data: Dict[str, Any],
        calculation_type: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform a domain-specific calculation.
        
        Args:
            input_data: Data to calculate from
            calculation_type: Type of calculation
            parameters: Optional calculation parameters
            
        Returns:
            Calculation results
        """
        ...
    
    def get_supported_calculations(self) -> List[str]:
        """Get list of supported calculation types.
        
        Returns:
            List of calculation type names
        """
        ...
    
    async def validate_calculation_input(
        self,
        input_data: Dict[str, Any],
        calculation_type: str,
    ) -> bool:
        """Validate input data for a calculation.
        
        Args:
            input_data: Data to validate
            calculation_type: Type of calculation
            
        Returns:
            Whether input is valid
        """
        ...