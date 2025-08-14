"""Training Services Module - Clean Architecture Implementation

Provides factory and initialization for decomposed training system services.
Extracted from training_system_manager.py (2109 lines) as part of clean architecture refactoring.
"""

from typing import Optional

from prompt_improver.cli.services.training_metrics import TrainingMetrics
from prompt_improver.cli.services.training_orchestrator import TrainingOrchestrator
from prompt_improver.cli.services.training_persistence import TrainingPersistence
from prompt_improver.cli.services.training_protocols import (
    TrainingMetricsProtocol,
    TrainingOrchestratorProtocol,
    TrainingPersistenceProtocol,
    TrainingServiceFactoryProtocol,
    TrainingValidatorProtocol,
)
from prompt_improver.cli.services.training_validator import TrainingValidator


class TrainingServiceFactory:
    """Factory for creating training system services with dependency injection.
    
    Implements Clean Architecture patterns:
    - Protocol-based dependency injection
    - Factory pattern for service creation
    - Dependency inversion principle
    - Single responsibility principle
    """

    def create_validator(self) -> TrainingValidatorProtocol:
        """Create training validator service.
        
        Returns:
            Configured training validator instance
        """
        return TrainingValidator()

    def create_metrics(self) -> TrainingMetricsProtocol:
        """Create training metrics service.
        
        Returns:
            Configured training metrics instance
        """
        return TrainingMetrics()

    def create_persistence(self) -> TrainingPersistenceProtocol:
        """Create training persistence service.
        
        Returns:
            Configured training persistence instance
        """
        return TrainingPersistence()

    def create_orchestrator(
        self,
        console=None,
        validator: Optional[TrainingValidatorProtocol] = None,
        metrics: Optional[TrainingMetricsProtocol] = None,
        persistence: Optional[TrainingPersistenceProtocol] = None,
    ) -> TrainingOrchestratorProtocol:
        """Create training orchestrator service with dependencies.
        
        Args:
            console: Rich console for output
            validator: Training validator service
            metrics: Training metrics service  
            persistence: Training persistence service
            
        Returns:
            Configured training orchestrator instance
        """
        # Create dependencies if not provided
        if validator is None:
            validator = self.create_validator()
        if metrics is None:
            metrics = self.create_metrics()
        if persistence is None:
            persistence = self.create_persistence()

        return TrainingOrchestrator(
            console=console,
            validator=validator,
            metrics=metrics,
            persistence=persistence,
        )

    def create_complete_training_system(self, console=None) -> TrainingOrchestratorProtocol:
        """Create complete training system with all dependencies.
        
        Args:
            console: Rich console for output
            
        Returns:
            Fully configured training orchestrator with all services
        """
        # Create all services with proper dependency injection
        validator = self.create_validator()
        metrics = self.create_metrics()
        persistence = self.create_persistence()
        
        orchestrator = self.create_orchestrator(
            console=console,
            validator=validator,
            metrics=metrics,
            persistence=persistence,
        )
        
        return orchestrator


# Global factory instance for easy access
_training_service_factory = TrainingServiceFactory()


def get_training_service_factory() -> TrainingServiceFactory:
    """Get the global training service factory instance.
    
    Returns:
        Global training service factory
    """
    return _training_service_factory


def create_training_system(console=None) -> TrainingOrchestratorProtocol:
    """Convenience function to create complete training system.
    
    Args:
        console: Rich console for output
        
    Returns:
        Fully configured training orchestrator
    """
    return _training_service_factory.create_complete_training_system(console)


# Individual service creation functions for backwards compatibility
def create_training_validator() -> TrainingValidatorProtocol:
    """Create training validator service.
    
    Returns:
        Training validator instance
    """
    return _training_service_factory.create_validator()


def create_training_metrics() -> TrainingMetricsProtocol:
    """Create training metrics service.
    
    Returns:
        Training metrics instance
    """
    return _training_service_factory.create_metrics()


def create_training_persistence() -> TrainingPersistenceProtocol:
    """Create training persistence service.
    
    Returns:
        Training persistence instance
    """
    return _training_service_factory.create_persistence()


# Export all classes and protocols for direct import
__all__ = [
    # Main services
    "TrainingOrchestrator",
    "TrainingValidator", 
    "TrainingMetrics",
    "TrainingPersistence",
    # Protocols
    "TrainingOrchestratorProtocol",
    "TrainingValidatorProtocol",
    "TrainingMetricsProtocol", 
    "TrainingPersistenceProtocol",
    "TrainingServiceFactoryProtocol",
    # Factory
    "TrainingServiceFactory",
    "get_training_service_factory",
    # Convenience functions
    "create_training_system",
    "create_training_validator",
    "create_training_metrics",
    "create_training_persistence",
]