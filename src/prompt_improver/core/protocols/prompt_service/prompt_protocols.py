"""Protocol interfaces for PromptServiceFacade decomposition.

Following Clean Architecture principles with protocol-based dependency injection.
These protocols define the contracts for the decomposed prompt improvement services.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from datetime import datetime
from uuid import UUID

from prompt_improver.core.domain.types import (
    ImprovementSessionData,
    PromptSessionData,
    TrainingSessionData,
    UserFeedbackData,
)
from prompt_improver.rule_engine.base import BasePromptRule


@runtime_checkable
class PromptAnalysisServiceProtocol(Protocol):
    """Protocol for prompt analysis and improvement logic."""
    
    async def analyze_prompt(
        self,
        prompt_id: UUID,
        session_id: Optional[UUID] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze a prompt and generate improvement suggestions.
        
        Args:
            prompt_id: Unique identifier for the prompt
            session_id: Optional session identifier for tracking
            context: Additional context for analysis
            
        Returns:
            Analysis results including suggestions and metrics
        """
        ...
    
    async def generate_improvements(
        self,
        prompt: str,
        rules: List[BasePromptRule],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate improvements for a prompt using specified rules.
        
        Args:
            prompt: The prompt text to improve
            rules: List of rules to apply
            context: Additional context for improvement
            
        Returns:
            List of improvement suggestions with confidence scores
        """
        ...
    
    async def evaluate_improvement_quality(
        self,
        original_prompt: str,
        improved_prompt: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Evaluate the quality of an improvement.
        
        Args:
            original_prompt: The original prompt text
            improved_prompt: The improved prompt text
            metrics: Optional metrics to consider
            
        Returns:
            Quality scores and metrics
        """
        ...
    
    async def get_ml_recommendations(
        self,
        prompt: str,
        session_id: UUID,
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get ML-based recommendations for prompt improvement.
        
        Args:
            prompt: The prompt to analyze
            session_id: Session identifier
            model_version: Optional specific model version
            
        Returns:
            ML recommendations and confidence scores
        """
        ...


@runtime_checkable
class RuleApplicationServiceProtocol(Protocol):
    """Protocol for rule execution and validation."""
    
    async def apply_rules(
        self,
        prompt: str,
        rules: List[BasePromptRule],
        session_id: Optional[UUID] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Apply a set of rules to a prompt.
        
        Args:
            prompt: The prompt to process
            rules: Rules to apply
            session_id: Optional session for tracking
            config: Rule application configuration
            
        Returns:
            Rule application results and transformations
        """
        ...
    
    async def validate_rule_compatibility(
        self,
        rules: List[BasePromptRule],
        prompt_type: Optional[str] = None
    ) -> Dict[str, bool]:
        """Validate if rules are compatible with each other.
        
        Args:
            rules: List of rules to validate
            prompt_type: Optional prompt type for context
            
        Returns:
            Compatibility matrix and warnings
        """
        ...
    
    async def execute_rule_chain(
        self,
        prompt: str,
        rule_chain: List[BasePromptRule],
        stop_on_error: bool = False
    ) -> List[Dict[str, Any]]:
        """Execute a chain of rules in sequence.
        
        Args:
            prompt: Initial prompt
            rule_chain: Ordered list of rules
            stop_on_error: Whether to stop on first error
            
        Returns:
            List of results from each rule execution
        """
        ...
    
    async def get_rule_performance_metrics(
        self,
        rule_id: str,
        time_range: Optional[tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """Get performance metrics for a specific rule.
        
        Args:
            rule_id: Rule identifier
            time_range: Optional time range for metrics
            
        Returns:
            Performance metrics and statistics
        """
        ...


@runtime_checkable
class ValidationServiceProtocol(Protocol):
    """Protocol for input validation and business rule checking."""
    
    async def validate_prompt_input(
        self,
        prompt: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate prompt input against constraints.
        
        Args:
            prompt: Prompt to validate
            constraints: Validation constraints
            
        Returns:
            Validation results and any violations
        """
        ...
    
    async def check_business_rules(
        self,
        operation: str,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if an operation complies with business rules.
        
        Args:
            operation: Operation type
            data: Operation data
            context: Additional context
            
        Returns:
            Whether the operation is allowed
        """
        ...
    
    async def validate_improvement_session(
        self,
        session: ImprovementSessionData
    ) -> Dict[str, Any]:
        """Validate an improvement session.
        
        Args:
            session: Session to validate
            
        Returns:
            Validation results and warnings
        """
        ...
    
    async def sanitize_prompt_content(
        self,
        prompt: str,
        sanitization_level: str = "standard"
    ) -> str:
        """Sanitize prompt content for safety.
        
        Args:
            prompt: Prompt to sanitize
            sanitization_level: Level of sanitization
            
        Returns:
            Sanitized prompt
        """
        ...


@runtime_checkable
class PromptServiceFacadeProtocol(Protocol):
    """Protocol for the unified PromptServiceFacade."""
    
    async def improve_prompt(
        self,
        prompt: str,
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
        rules: Optional[List[BasePromptRule]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Main method to improve a prompt.
        
        Args:
            prompt: Prompt to improve
            user_id: Optional user identifier
            session_id: Optional session identifier
            rules: Optional specific rules to apply
            config: Optional configuration
            
        Returns:
            Improvement results including improved prompt and metrics
        """
        ...
    
    async def get_session_summary(
        self,
        session_id: UUID
    ) -> Dict[str, Any]:
        """Get summary of an improvement session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary with metrics
        """
        ...
    
    async def process_feedback(
        self,
        feedback: UserFeedbackData,
        session_id: UUID
    ) -> Dict[str, Any]:
        """Process user feedback for a session.
        
        Args:
            feedback: User feedback
            session_id: Associated session
            
        Returns:
            Feedback processing results
        """
        ...
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all prompt services.
        
        Returns:
            Health status of each component
        """
        ...