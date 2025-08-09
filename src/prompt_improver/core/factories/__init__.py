"""Component and Pipeline Factory Module (2025).

Provides factory pattern implementations for creating ML pipeline components
and orchestrators with proper dependency injection and Protocol compliance.

Exports:
    ComponentFactory: Factory for creating ML components with dependency injection
    DependencyValidator: Utility for validating component dependencies
    MLPipelineOrchestratorFactory: Factory for creating ML pipeline orchestrators
"""
from prompt_improver.core.factories.component_factory import ComponentFactory, DependencyValidator, create_component_factory, register_default_component_specs
from prompt_improver.core.factories.ml_pipeline_factory import ComponentLoaderFactory, ExternalServiceFactory, MLPipelineOrchestratorFactory, create_ml_pipeline_orchestrator, create_production_orchestrator, ml_pipeline_context
__all__ = ['ComponentFactory', 'DependencyValidator', 'create_component_factory', 'register_default_component_specs', 'MLPipelineOrchestratorFactory', 'ComponentLoaderFactory', 'ExternalServiceFactory', 'create_ml_pipeline_orchestrator', 'create_production_orchestrator', 'ml_pipeline_context']
