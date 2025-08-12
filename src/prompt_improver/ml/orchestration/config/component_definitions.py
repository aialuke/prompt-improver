"""
Component definitions for all 50+ ML components across 6 tiers.
"""
from typing import Any, Dict, List
from sqlmodel import SQLModel, Field
from pydantic import BaseModel
from ..shared.component_types import ComponentCapability, ComponentInfo, ComponentTier

class ComponentDefinitions(BaseModel):
    """Central registry of all ML component definitions."""
    tier1_core_components: dict[str, dict[str, Any]] = Field(default_factory=lambda: {'training_data_loader': {'description': 'Central training data hub for ML pipeline', 'file_path': 'ml/core/training_data_loader.py', 'capabilities': ['data_loading', 'data_preprocessing', 'training_data_management'], 'dependencies': ['database', 'file_system'], 'resource_requirements': {'memory': '1GB', 'cpu': '1 core'}}, 'ml_integration': {'description': 'Core ML service processing engine', 'file_path': 'ml/core/ml_integration.py', 'capabilities': ['model_training', 'prediction', 'inference'], 'dependencies': ['training_data_loader', 'model_manager'], 'resource_requirements': {'memory': '2GB', 'cpu': '2 cores', 'gpu': 'optional'}}, 'rule_optimizer': {'description': 'Multi-objective optimization for rules', 'file_path': 'ml/optimization/algorithms/rule_optimizer.py', 'capabilities': ['optimization', 'rule_tuning', 'parameter_search'], 'dependencies': ['ml_integration'], 'resource_requirements': {'memory': '1GB', 'cpu': '2 cores'}}, 'multi_armed_bandit': {'description': 'Thompson Sampling and UCB algorithms', 'file_path': 'ml/optimization/algorithms/multi_armed_bandit.py', 'capabilities': ['exploration', 'exploitation', 'adaptive_selection'], 'dependencies': [], 'resource_requirements': {'memory': '512MB', 'cpu': '1 core'}}, 'apriori_analyzer': {'description': 'Association rule mining and pattern discovery', 'file_path': 'ml/learning/patterns/apriori_analyzer.py', 'capabilities': ['pattern_mining', 'association_rules', 'frequent_patterns'], 'dependencies': ['training_data_loader'], 'resource_requirements': {'memory': '1GB', 'cpu': '1 core'}}, 'batch_processor': {'description': 'Unified batch processing component with OpenTelemetry metrics', 'file_path': 'ml/optimization/batch/unified_batch_processor.py', 'capabilities': ['batch_processing', 'parallel_execution', 'resource_management'], 'dependencies': [], 'resource_requirements': {'memory': '2GB', 'cpu': '2 cores'}}}, description='Core ML pipeline component definitions')
    tier2_learning_components: dict[str, dict[str, Any]] = Field(default_factory=dict, description='Learning component definitions')
    tier3_orchestration_components: dict[str, dict[str, Any]] = Field(default_factory=dict, description='Orchestration component definitions')
    tier4_monitoring_components: dict[str, dict[str, Any]] = Field(default_factory=dict, description='Monitoring component definitions')
    tier5_integration_components: dict[str, dict[str, Any]] = Field(default_factory=dict, description='Integration component definitions')
    tier6_deployment_components: dict[str, dict[str, Any]] = Field(default_factory=dict, description='Deployment component definitions')

    def get_all_components(self) -> dict[str, dict[str, Any]]:
        """Get all component definitions across all tiers."""
        all_components = {}
        all_components.update(self.tier1_core_components)
        all_components.update(self.tier2_learning_components)
        all_components.update(self.tier3_orchestration_components)
        all_components.update(self.tier4_monitoring_components)
        all_components.update(self.tier5_integration_components)
        all_components.update(self.tier6_deployment_components)
        return all_components

    def get_component_by_name(self, name: str) -> dict[str, Any]:
        """Get specific component definition by name."""
        all_components = self.get_all_components()
        return all_components.get(name, {})
